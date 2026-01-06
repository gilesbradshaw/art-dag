# primitive/nodes/source.py
"""
Source executors: Load media from paths.

Primitives: SOURCE
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..dag import NodeType
from ..executor import Executor, register_executor

logger = logging.getLogger(__name__)


@register_executor(NodeType.SOURCE)
class SourceExecutor(Executor):
    """
    Load source media from a path.

    Config:
        path: Path to source file

    Creates a symlink to the source file for zero-copy loading.
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        source_path = Path(config["path"])

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use symlink for zero-copy
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()

        # Preserve extension from source
        actual_output = output_path.with_suffix(source_path.suffix)
        if actual_output.exists() or actual_output.is_symlink():
            actual_output.unlink()

        os.symlink(source_path.resolve(), actual_output)
        logger.debug(f"SOURCE: {source_path.name} -> {actual_output}")

        return actual_output

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        if "path" not in config:
            errors.append("SOURCE requires 'path' config")
        return errors
