# primitive/executor.py
"""
Executor base class and registry.

Executors implement the actual operations for each node type.
They are registered by node type and looked up during execution.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .dag import NodeType

logger = logging.getLogger(__name__)

# Global executor registry
_EXECUTORS: Dict[NodeType | str, Type["Executor"]] = {}


class Executor(ABC):
    """
    Base class for node executors.

    Subclasses implement execute() to perform the actual operation.
    """

    @abstractmethod
    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        """
        Execute the node operation.

        Args:
            config: Node configuration
            inputs: Paths to input files (from resolved input nodes)
            output_path: Where to write the output

        Returns:
            Path to the output file
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate node configuration.

        Returns list of error messages (empty if valid).
        Override in subclasses for specific validation.
        """
        return []

    def estimate_output_size(self, config: Dict[str, Any], input_sizes: List[int]) -> int:
        """
        Estimate output size in bytes.

        Override for better estimates. Default returns sum of inputs.
        """
        return sum(input_sizes) if input_sizes else 0


def register_executor(node_type: NodeType | str) -> Callable:
    """
    Decorator to register an executor for a node type.

    Usage:
        @register_executor(NodeType.SOURCE)
        class SourceExecutor(Executor):
            ...
    """
    def decorator(cls: Type[Executor]) -> Type[Executor]:
        if node_type in _EXECUTORS:
            logger.warning(f"Overwriting executor for {node_type}")
        _EXECUTORS[node_type] = cls
        return cls
    return decorator


def get_executor(node_type: NodeType | str) -> Optional[Executor]:
    """
    Get an executor instance for a node type.

    Returns None if no executor is registered.
    """
    executor_cls = _EXECUTORS.get(node_type)
    if executor_cls is None:
        return None
    return executor_cls()


def list_executors() -> Dict[str, Type[Executor]]:
    """List all registered executors."""
    return {
        (k.name if isinstance(k, NodeType) else k): v
        for k, v in _EXECUTORS.items()
    }


def clear_executors():
    """Clear all registered executors (for testing)."""
    _EXECUTORS.clear()
