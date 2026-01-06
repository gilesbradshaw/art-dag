# primitive/client.py
"""
Client SDK for the primitive execution server.

Provides a simple API for submitting DAGs and retrieving results.

Usage:
    client = PrimitiveClient("http://localhost:8080")

    # Build a DAG
    builder = DAGBuilder()
    source = builder.source("/path/to/video.mp4")
    segment = builder.segment(source, duration=5.0)
    builder.set_output(segment)
    dag = builder.build()

    # Execute and wait for result
    result = client.execute(dag)
    print(f"Output: {result.output_path}")
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

from .dag import DAG, DAGBuilder


@dataclass
class ExecutionResult:
    """Result from server execution."""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    nodes_executed: int = 0
    nodes_cached: int = 0


@dataclass
class CacheStats:
    """Cache statistics from server."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0


class PrimitiveClient:
    """
    Client for the primitive execution server.

    Args:
        base_url: Server URL (e.g., "http://localhost:8080")
        timeout: Request timeout in seconds
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make HTTP request to server."""
        url = f"{self.base_url}{path}"

        if data is not None:
            body = json.dumps(data).encode()
            headers = {"Content-Type": "application/json"}
        else:
            body = None
            headers = {}

        req = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except HTTPError as e:
            error_body = e.read().decode()
            try:
                error_data = json.loads(error_body)
                raise RuntimeError(error_data.get("error", str(e)))
            except json.JSONDecodeError:
                raise RuntimeError(f"HTTP {e.code}: {error_body}")
        except URLError as e:
            raise ConnectionError(f"Failed to connect to server: {e}")

    def health(self) -> bool:
        """Check if server is healthy."""
        try:
            result = self._request("GET", "/health")
            return result.get("status") == "ok"
        except Exception:
            return False

    def submit(self, dag: DAG) -> str:
        """
        Submit a DAG for execution.

        Args:
            dag: The DAG to execute

        Returns:
            Job ID for tracking
        """
        result = self._request("POST", "/execute", dag.to_dict())
        return result["job_id"]

    def status(self, job_id: str) -> str:
        """
        Get job status.

        Args:
            job_id: Job ID from submit()

        Returns:
            Status: "pending", "running", "completed", or "failed"
        """
        result = self._request("GET", f"/status/{job_id}")
        return result["status"]

    def result(self, job_id: str) -> Optional[ExecutionResult]:
        """
        Get job result (non-blocking).

        Args:
            job_id: Job ID from submit()

        Returns:
            ExecutionResult if complete, None if still running
        """
        data = self._request("GET", f"/result/{job_id}")

        if not data.get("ready", False):
            return None

        return ExecutionResult(
            success=data.get("success", False),
            output_path=Path(data["output_path"]) if data.get("output_path") else None,
            error=data.get("error"),
            execution_time=data.get("execution_time", 0),
            nodes_executed=data.get("nodes_executed", 0),
            nodes_cached=data.get("nodes_cached", 0),
        )

    def wait(self, job_id: str, poll_interval: float = 0.5) -> ExecutionResult:
        """
        Wait for job completion and return result.

        Args:
            job_id: Job ID from submit()
            poll_interval: Seconds between status checks

        Returns:
            ExecutionResult
        """
        while True:
            result = self.result(job_id)
            if result is not None:
                return result
            time.sleep(poll_interval)

    def execute(self, dag: DAG, poll_interval: float = 0.5) -> ExecutionResult:
        """
        Submit DAG and wait for result.

        Convenience method combining submit() and wait().

        Args:
            dag: The DAG to execute
            poll_interval: Seconds between status checks

        Returns:
            ExecutionResult
        """
        job_id = self.submit(dag)
        return self.wait(job_id, poll_interval)

    def cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        data = self._request("GET", "/cache/stats")
        return CacheStats(
            total_entries=data.get("total_entries", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            hits=data.get("hits", 0),
            misses=data.get("misses", 0),
            hit_rate=data.get("hit_rate", 0.0),
        )

    def clear_cache(self) -> None:
        """Clear the server cache."""
        self._request("DELETE", "/cache")


# Re-export DAGBuilder for convenience
__all__ = ["PrimitiveClient", "ExecutionResult", "CacheStats", "DAGBuilder"]
