# primitive/engine.py
"""
DAG execution engine.

Executes DAGs by:
1. Resolving nodes in topological order
2. Checking cache for each node
3. Running executors for cache misses
4. Storing results in cache
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .dag import DAG, Node, NodeType
from .cache import Cache
from .executor import Executor, get_executor

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of executing a DAG."""
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    nodes_executed: int = 0
    nodes_cached: int = 0
    node_results: Dict[str, Path] = field(default_factory=dict)


@dataclass
class NodeProgress:
    """Progress update for a node."""
    node_id: str
    node_type: str
    status: str  # "pending", "running", "cached", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""


# Progress callback type
ProgressCallback = Callable[[NodeProgress], None]


class Engine:
    """
    DAG execution engine.

    Manages cache, resolves dependencies, and runs executors.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache = Cache(cache_dir)
        self._progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: ProgressCallback):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _report_progress(self, progress: NodeProgress):
        """Report progress to callback if set."""
        if self._progress_callback:
            try:
                self._progress_callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def execute(self, dag: DAG) -> ExecutionResult:
        """
        Execute a DAG and return the result.

        Args:
            dag: The DAG to execute

        Returns:
            ExecutionResult with output path or error
        """
        start_time = time.time()
        node_results: Dict[str, Path] = {}
        nodes_executed = 0
        nodes_cached = 0

        # Validate DAG
        errors = dag.validate()
        if errors:
            return ExecutionResult(
                success=False,
                error=f"Invalid DAG: {errors}",
                execution_time=time.time() - start_time,
            )

        # Get topological order
        try:
            order = dag.topological_order()
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Failed to order DAG: {e}",
                execution_time=time.time() - start_time,
            )

        # Execute each node
        for node_id in order:
            node = dag.get_node(node_id)
            type_str = node.node_type.name if isinstance(node.node_type, NodeType) else str(node.node_type)

            # Report starting
            self._report_progress(NodeProgress(
                node_id=node_id,
                node_type=type_str,
                status="pending",
                message=f"Processing {type_str}",
            ))

            # Check cache first
            cached_path = self.cache.get(node_id)
            if cached_path is not None:
                node_results[node_id] = cached_path
                nodes_cached += 1
                self._report_progress(NodeProgress(
                    node_id=node_id,
                    node_type=type_str,
                    status="cached",
                    progress=1.0,
                    message="Using cached result",
                ))
                continue

            # Get executor
            executor = get_executor(node.node_type)
            if executor is None:
                return ExecutionResult(
                    success=False,
                    error=f"No executor for node type: {node.node_type}",
                    execution_time=time.time() - start_time,
                    node_results=node_results,
                )

            # Resolve input paths
            input_paths = []
            for input_id in node.inputs:
                if input_id not in node_results:
                    return ExecutionResult(
                        success=False,
                        error=f"Missing input {input_id} for node {node_id}",
                        execution_time=time.time() - start_time,
                        node_results=node_results,
                    )
                input_paths.append(node_results[input_id])

            # Determine output path
            output_path = self.cache.get_output_path(node_id, ".mkv")

            # Execute
            self._report_progress(NodeProgress(
                node_id=node_id,
                node_type=type_str,
                status="running",
                progress=0.5,
                message=f"Executing {type_str}",
            ))

            node_start = time.time()
            try:
                result_path = executor.execute(
                    config=node.config,
                    inputs=input_paths,
                    output_path=output_path,
                )
                node_time = time.time() - node_start

                # Store in cache (file is already at output_path)
                self.cache.put(
                    node_id=node_id,
                    source_path=result_path,
                    node_type=type_str,
                    execution_time=node_time,
                    move=False,  # Already in place
                )

                node_results[node_id] = result_path
                nodes_executed += 1

                self._report_progress(NodeProgress(
                    node_id=node_id,
                    node_type=type_str,
                    status="completed",
                    progress=1.0,
                    message=f"Completed in {node_time:.2f}s",
                ))

            except Exception as e:
                logger.error(f"Node {node_id} failed: {e}")
                self._report_progress(NodeProgress(
                    node_id=node_id,
                    node_type=type_str,
                    status="failed",
                    message=str(e),
                ))
                return ExecutionResult(
                    success=False,
                    error=f"Node {node_id} ({type_str}) failed: {e}",
                    execution_time=time.time() - start_time,
                    node_results=node_results,
                    nodes_executed=nodes_executed,
                    nodes_cached=nodes_cached,
                )

        # Get final output
        output_path = node_results.get(dag.output_id)

        return ExecutionResult(
            success=True,
            output_path=output_path,
            execution_time=time.time() - start_time,
            nodes_executed=nodes_executed,
            nodes_cached=nodes_cached,
            node_results=node_results,
        )

    def execute_node(self, node: Node, inputs: List[Path]) -> Path:
        """
        Execute a single node (bypassing DAG structure).

        Useful for testing individual executors.
        """
        executor = get_executor(node.node_type)
        if executor is None:
            raise ValueError(f"No executor for node type: {node.node_type}")

        output_path = self.cache.get_output_path(node.node_id, ".mkv")
        return executor.execute(node.config, inputs, output_path)

    def get_cache_stats(self):
        """Get cache statistics."""
        return self.cache.get_stats()

    def clear_cache(self):
        """Clear the cache."""
        self.cache.clear()
