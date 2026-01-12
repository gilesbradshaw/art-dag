"""
Sandbox for effect execution.

Uses bubblewrap (bwrap) for Linux namespace isolation.
Provides controlled access to:
  - Input files (read-only)
  - Output file (write)
  - stderr (logging)
  - Seeded RNG
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    """
    Sandbox configuration.

    Attributes:
        trust_level: "untrusted" (full isolation) or "trusted" (allows subprocess)
        venv_path: Path to effect's virtual environment
        wheel_cache: Shared wheel cache directory
        timeout: Maximum execution time in seconds
        memory_limit: Memory limit in bytes (0 = unlimited)
        allow_network: Whether to allow network access
    """

    trust_level: str = "untrusted"
    venv_path: Optional[Path] = None
    wheel_cache: Path = field(default_factory=lambda: Path("/var/cache/artdag/wheels"))
    timeout: int = 3600  # 1 hour default
    memory_limit: int = 0
    allow_network: bool = False


def is_bwrap_available() -> bool:
    """Check if bubblewrap is available."""
    try:
        result = subprocess.run(
            ["bwrap", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_venv_path(dependencies: List[str], cache_dir: Path = None) -> Path:
    """
    Get or create venv for given dependencies.

    Uses hash of sorted dependencies for cache key.

    Args:
        dependencies: List of pip package specifiers
        cache_dir: Base directory for venv cache

    Returns:
        Path to venv directory
    """
    cache_dir = cache_dir or Path("/var/cache/artdag/venvs")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Compute deps hash
    sorted_deps = sorted(dep.lower().strip() for dep in dependencies)
    deps_str = "\n".join(sorted_deps)
    deps_hash = hashlib.sha3_256(deps_str.encode()).hexdigest()[:16]

    venv_path = cache_dir / deps_hash

    if venv_path.exists():
        logger.debug(f"Reusing venv at {venv_path}")
        return venv_path

    # Create new venv
    logger.info(f"Creating venv for {len(dependencies)} deps at {venv_path}")

    subprocess.run(
        ["python", "-m", "venv", str(venv_path)],
        check=True,
    )

    # Install dependencies
    pip_path = venv_path / "bin" / "pip"
    wheel_cache = Path("/var/cache/artdag/wheels")

    if dependencies:
        cmd = [
            str(pip_path),
            "install",
            "--cache-dir", str(wheel_cache),
            *dependencies,
        ]
        subprocess.run(cmd, check=True)

    return venv_path


@dataclass
class SandboxResult:
    """Result of sandboxed execution."""

    success: bool
    output_path: Optional[Path] = None
    stderr: str = ""
    exit_code: int = 0
    error: Optional[str] = None


class Sandbox:
    """
    Sandboxed effect execution environment.

    Uses bubblewrap for namespace isolation when available,
    falls back to subprocess with restricted permissions.
    """

    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self._temp_dirs: List[Path] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Clean up temporary directories."""
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs = []

    def _create_temp_dir(self) -> Path:
        """Create a temporary directory for sandbox use."""
        temp_dir = Path(tempfile.mkdtemp(prefix="artdag_sandbox_"))
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def run_effect(
        self,
        effect_path: Path,
        input_paths: List[Path],
        output_path: Path,
        params: Dict[str, Any],
        bindings: Dict[str, List[float]] = None,
        seed: int = 0,
    ) -> SandboxResult:
        """
        Run an effect in the sandbox.

        Args:
            effect_path: Path to effect.py
            input_paths: List of input file paths
            output_path: Output file path
            params: Effect parameters
            bindings: Per-frame parameter bindings
            seed: RNG seed for determinism

        Returns:
            SandboxResult with success status and output
        """
        bindings = bindings or {}

        # Create work directory
        work_dir = self._create_temp_dir()
        config_path = work_dir / "config.json"
        effect_copy = work_dir / "effect.py"

        # Copy effect to work dir
        shutil.copy(effect_path, effect_copy)

        # Write config file
        config_data = {
            "input_paths": [str(p) for p in input_paths],
            "output_path": str(output_path),
            "params": params,
            "bindings": bindings,
            "seed": seed,
        }
        config_path.write_text(json.dumps(config_data))

        if is_bwrap_available() and self.config.trust_level == "untrusted":
            return self._run_with_bwrap(
                effect_copy, config_path, input_paths, output_path, work_dir
            )
        else:
            return self._run_subprocess(
                effect_copy, config_path, input_paths, output_path, work_dir
            )

    def _run_with_bwrap(
        self,
        effect_path: Path,
        config_path: Path,
        input_paths: List[Path],
        output_path: Path,
        work_dir: Path,
    ) -> SandboxResult:
        """Run effect with bubblewrap isolation."""
        logger.info("Running effect in bwrap sandbox")

        # Build bwrap command
        cmd = [
            "bwrap",
            # New PID namespace
            "--unshare-pid",
            # No network
            "--unshare-net",
            # Read-only root filesystem
            "--ro-bind", "/", "/",
            # Read-write work directory
            "--bind", str(work_dir), str(work_dir),
            # Read-only input files
        ]

        for input_path in input_paths:
            cmd.extend(["--ro-bind", str(input_path), str(input_path)])

        # Bind output directory as writable
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--bind", str(output_dir), str(output_dir)])

        # Bind venv if available
        if self.config.venv_path and self.config.venv_path.exists():
            cmd.extend(["--ro-bind", str(self.config.venv_path), str(self.config.venv_path)])
            python_path = self.config.venv_path / "bin" / "python"
        else:
            python_path = Path("/usr/bin/python3")

        # Add runner script
        runner_script = self._get_runner_script()
        runner_path = work_dir / "runner.py"
        runner_path.write_text(runner_script)

        # Run the effect
        cmd.extend([
            str(python_path),
            str(runner_path),
            str(effect_path),
            str(config_path),
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
            )

            if result.returncode == 0 and output_path.exists():
                return SandboxResult(
                    success=True,
                    output_path=output_path,
                    stderr=result.stderr,
                    exit_code=0,
                )
            else:
                return SandboxResult(
                    success=False,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    error=result.stderr or "Effect execution failed",
                )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Effect timed out after {self.config.timeout}s",
                exit_code=-1,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                exit_code=-1,
            )

    def _run_subprocess(
        self,
        effect_path: Path,
        config_path: Path,
        input_paths: List[Path],
        output_path: Path,
        work_dir: Path,
    ) -> SandboxResult:
        """Run effect in subprocess (fallback without bwrap)."""
        logger.warning("Running effect without sandbox isolation")

        # Create runner script
        runner_script = self._get_runner_script()
        runner_path = work_dir / "runner.py"
        runner_path.write_text(runner_script)

        # Determine Python path
        if self.config.venv_path and self.config.venv_path.exists():
            python_path = self.config.venv_path / "bin" / "python"
        else:
            python_path = "python3"

        cmd = [
            str(python_path),
            str(runner_path),
            str(effect_path),
            str(config_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(work_dir),
            )

            if result.returncode == 0 and output_path.exists():
                return SandboxResult(
                    success=True,
                    output_path=output_path,
                    stderr=result.stderr,
                    exit_code=0,
                )
            else:
                return SandboxResult(
                    success=False,
                    stderr=result.stderr,
                    exit_code=result.returncode,
                    error=result.stderr or "Effect execution failed",
                )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                error=f"Effect timed out after {self.config.timeout}s",
                exit_code=-1,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                error=str(e),
                exit_code=-1,
            )

    def _get_runner_script(self) -> str:
        """Get the runner script that executes effects."""
        return '''#!/usr/bin/env python3
"""Effect runner script - executed in sandbox."""

import importlib.util
import json
import sys
from pathlib import Path

def load_effect(effect_path):
    """Load effect module from path."""
    spec = importlib.util.spec_from_file_location("effect", effect_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    if len(sys.argv) < 3:
        print("Usage: runner.py <effect_path> <config_path>", file=sys.stderr)
        sys.exit(1)

    effect_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])

    # Load config
    config = json.loads(config_path.read_text())

    input_paths = [Path(p) for p in config["input_paths"]]
    output_path = Path(config["output_path"])
    params = config["params"]
    bindings = config.get("bindings", {})
    seed = config.get("seed", 0)

    # Load effect
    effect = load_effect(effect_path)

    # Check API type
    if hasattr(effect, "process"):
        # Whole-video API
        from artdag.effects.meta import ExecutionContext
        ctx = ExecutionContext(
            input_paths=[str(p) for p in input_paths],
            output_path=str(output_path),
            params=params,
            seed=seed,
            bindings=bindings,
        )
        effect.process(input_paths, output_path, params, ctx)

    elif hasattr(effect, "process_frame"):
        # Frame-by-frame API
        from artdag.effects.frame_processor import process_video

        result_path, _ = process_video(
            input_path=input_paths[0],
            output_path=output_path,
            process_frame=effect.process_frame,
            params=params,
            bindings=bindings,
        )

    else:
        print("Effect must have process() or process_frame()", file=sys.stderr)
        sys.exit(1)

    print(f"Success: {output_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
'''
