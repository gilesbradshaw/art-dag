# primitive/server.py
"""
HTTP server for primitive execution engine.

Provides a REST API for submitting DAGs and retrieving results.

Endpoints:
    POST /execute     - Submit DAG for execution
    GET  /status/:id  - Get execution status
    GET  /result/:id  - Get execution result
    GET  /cache/stats - Get cache statistics
    DELETE /cache     - Clear cache
"""

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .dag import DAG
from .engine import Engine, ExecutionResult
from . import nodes  # Register built-in executors

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """A pending or completed execution job."""
    job_id: str
    dag: DAG
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[ExecutionResult] = None
    error: Optional[str] = None


class PrimitiveServer:
    """
    HTTP server for the primitive engine.

    Usage:
        server = PrimitiveServer(cache_dir="/tmp/primitive_cache", port=8080)
        server.start()  # Blocking
    """

    def __init__(self, cache_dir: Path | str, host: str = "127.0.0.1", port: int = 8080):
        self.cache_dir = Path(cache_dir)
        self.host = host
        self.port = port
        self.engine = Engine(self.cache_dir)
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit_job(self, dag: DAG) -> str:
        """Submit a DAG for execution, return job ID."""
        job_id = str(uuid.uuid4())[:8]
        job = Job(job_id=job_id, dag=dag)

        with self._lock:
            self.jobs[job_id] = job

        # Execute in background thread
        thread = threading.Thread(target=self._execute_job, args=(job_id,))
        thread.daemon = True
        thread.start()

        return job_id

    def _execute_job(self, job_id: str):
        """Execute a job in background."""
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.status = "running"

        try:
            result = self.engine.execute(job.dag)
            with self._lock:
                job.result = result
                job.status = "completed" if result.success else "failed"
                if not result.success:
                    job.error = result.error
        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            with self._lock:
                job.status = "failed"
                job.error = str(e)

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def _create_handler(server_instance):
        """Create request handler with access to server instance."""

        class RequestHandler(BaseHTTPRequestHandler):
            server_ref = server_instance

            def log_message(self, format, *args):
                logger.debug(format % args)

            def _send_json(self, data: Any, status: int = 200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _send_error(self, message: str, status: int = 400):
                self._send_json({"error": message}, status)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path

                if path.startswith("/status/"):
                    job_id = path[8:]
                    job = self.server_ref.get_job(job_id)
                    if not job:
                        self._send_error("Job not found", 404)
                        return
                    self._send_json({
                        "job_id": job.job_id,
                        "status": job.status,
                        "error": job.error,
                    })

                elif path.startswith("/result/"):
                    job_id = path[8:]
                    job = self.server_ref.get_job(job_id)
                    if not job:
                        self._send_error("Job not found", 404)
                        return
                    if job.status == "pending" or job.status == "running":
                        self._send_json({
                            "job_id": job.job_id,
                            "status": job.status,
                            "ready": False,
                        })
                        return

                    result = job.result
                    self._send_json({
                        "job_id": job.job_id,
                        "status": job.status,
                        "ready": True,
                        "success": result.success if result else False,
                        "output_path": str(result.output_path) if result and result.output_path else None,
                        "error": job.error,
                        "execution_time": result.execution_time if result else 0,
                        "nodes_executed": result.nodes_executed if result else 0,
                        "nodes_cached": result.nodes_cached if result else 0,
                    })

                elif path == "/cache/stats":
                    stats = self.server_ref.engine.get_cache_stats()
                    self._send_json({
                        "total_entries": stats.total_entries,
                        "total_size_bytes": stats.total_size_bytes,
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "hit_rate": stats.hit_rate,
                    })

                elif path == "/health":
                    self._send_json({"status": "ok"})

                else:
                    self._send_error("Not found", 404)

            def do_POST(self):
                if self.path == "/execute":
                    try:
                        content_length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(content_length).decode()
                        data = json.loads(body)

                        dag = DAG.from_dict(data)
                        job_id = self.server_ref.submit_job(dag)

                        self._send_json({
                            "job_id": job_id,
                            "status": "pending",
                        })
                    except json.JSONDecodeError as e:
                        self._send_error(f"Invalid JSON: {e}")
                    except Exception as e:
                        self._send_error(str(e), 500)
                else:
                    self._send_error("Not found", 404)

            def do_DELETE(self):
                if self.path == "/cache":
                    self.server_ref.engine.clear_cache()
                    self._send_json({"status": "cleared"})
                else:
                    self._send_error("Not found", 404)

        return RequestHandler

    def start(self):
        """Start the HTTP server (blocking)."""
        handler = self._create_handler()
        server = HTTPServer((self.host, self.port), handler)
        logger.info(f"Primitive server starting on {self.host}:{self.port}")
        print(f"Primitive server running on http://{self.host}:{self.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.shutdown()

    def start_background(self) -> threading.Thread:
        """Start the server in a background thread."""
        thread = threading.Thread(target=self.start)
        thread.daemon = True
        thread.start()
        return thread


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Primitive execution server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--cache-dir", default="/tmp/primitive_cache", help="Cache directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = PrimitiveServer(
        cache_dir=args.cache_dir,
        host=args.host,
        port=args.port,
    )
    server.start()


if __name__ == "__main__":
    main()
