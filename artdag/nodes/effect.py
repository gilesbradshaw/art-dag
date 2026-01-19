# artdag/nodes/effect.py
"""
Effect executor: Apply effects from the registry or IPFS.

Primitives: EFFECT

Effects can be:
1. Built-in (registered with @register_effect)
2. Stored in IPFS (referenced by CID)
"""

import importlib.util
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, TypeVar

import requests

from ..executor import Executor, register_executor

logger = logging.getLogger(__name__)

# Type alias for effect functions: (input_path, output_path, config) -> output_path
EffectFn = Callable[[Path, Path, Dict[str, Any]], Path]

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])

# IPFS API multiaddr - same as ipfs_client.py for consistency
# Docker uses /dns/ipfs/tcp/5001, local dev uses /ip4/127.0.0.1/tcp/5001
IPFS_API = os.environ.get("IPFS_API", "/ip4/127.0.0.1/tcp/5001")

# Connection timeout in seconds
IPFS_TIMEOUT = int(os.environ.get("IPFS_TIMEOUT", "30"))


def _get_ipfs_base_url() -> str:
    """
    Convert IPFS multiaddr to HTTP URL.

    Matches the conversion logic in ipfs_client.py for consistency.
    """
    multiaddr = IPFS_API

    # Handle /dns/hostname/tcp/port format (Docker)
    dns_match = re.match(r"/dns[46]?/([^/]+)/tcp/(\d+)", multiaddr)
    if dns_match:
        return f"http://{dns_match.group(1)}:{dns_match.group(2)}"

    # Handle /ip4/address/tcp/port format (local)
    ip4_match = re.match(r"/ip4/([^/]+)/tcp/(\d+)", multiaddr)
    if ip4_match:
        return f"http://{ip4_match.group(1)}:{ip4_match.group(2)}"

    # Fallback: assume it's already a URL or use default
    if multiaddr.startswith("http"):
        return multiaddr
    return "http://127.0.0.1:5001"


def _get_effects_cache_dir() -> Optional[Path]:
    """Get the effects cache directory from environment or default."""
    # Check both env var names (CACHE_DIR used by art-celery, ARTDAG_CACHE_DIR for standalone)
    for env_var in ["CACHE_DIR", "ARTDAG_CACHE_DIR"]:
        cache_dir = os.environ.get(env_var)
        if cache_dir:
            effects_dir = Path(cache_dir) / "_effects"
            if effects_dir.exists():
                return effects_dir

    # Try default locations
    for base in [Path.home() / ".artdag" / "cache", Path("/var/cache/artdag")]:
        effects_dir = base / "_effects"
        if effects_dir.exists():
            return effects_dir

    return None


def _fetch_effect_from_ipfs(cid: str, effect_path: Path) -> bool:
    """
    Fetch an effect from IPFS and cache locally.

    Uses the IPFS API endpoint (/api/v0/cat) for consistency with ipfs_client.py.
    This works reliably in Docker where IPFS_API=/dns/ipfs/tcp/5001.

    Returns True on success, False on failure.
    """
    try:
        # Use IPFS API (same as ipfs_client.py)
        base_url = _get_ipfs_base_url()
        url = f"{base_url}/api/v0/cat"
        params = {"arg": cid}

        response = requests.post(url, params=params, timeout=IPFS_TIMEOUT)
        response.raise_for_status()

        # Cache locally
        effect_path.parent.mkdir(parents=True, exist_ok=True)
        effect_path.write_bytes(response.content)
        logger.info(f"Fetched effect from IPFS: {cid[:16]}...")
        return True

    except Exception as e:
        logger.error(f"Failed to fetch effect from IPFS {cid[:16]}...: {e}")
        return False


def _parse_pep723_dependencies(source: str) -> List[str]:
    """
    Parse PEP 723 dependencies from effect source code.

    Returns list of package names (e.g., ["numpy", "opencv-python"]).
    """
    match = re.search(r"# /// script\n(.*?)# ///", source, re.DOTALL)
    if not match:
        return []

    block = match.group(1)
    deps_match = re.search(r'# dependencies = \[(.*?)\]', block, re.DOTALL)
    if not deps_match:
        return []

    return re.findall(r'"([^"]+)"', deps_match.group(1))


def _ensure_dependencies(dependencies: List[str], effect_cid: str) -> bool:
    """
    Ensure effect dependencies are installed.

    Installs missing packages using pip. Returns True on success.
    """
    if not dependencies:
        return True

    missing = []
    for dep in dependencies:
        # Extract package name (strip version specifiers)
        pkg_name = re.split(r'[<>=!]', dep)[0].strip()
        # Normalize name (pip uses underscores, imports use underscores or hyphens)
        pkg_name_normalized = pkg_name.replace('-', '_').lower()

        try:
            __import__(pkg_name_normalized)
        except ImportError:
            # Some packages have different import names
            try:
                # Try original name with hyphens replaced
                __import__(pkg_name.replace('-', '_'))
            except ImportError:
                missing.append(dep)

    if not missing:
        return True

    logger.info(f"Installing effect dependencies for {effect_cid[:16]}...: {missing}")

    try:
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.error(f"Failed to install dependencies: {result.stderr}")
            return False

        logger.info(f"Installed dependencies: {missing}")
        return True

    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False


def _load_cached_effect(effect_cid: str) -> Optional[EffectFn]:
    """
    Load an effect by CID, fetching from IPFS if not cached locally.

    Returns the effect function or None if not found.
    """
    effects_dir = _get_effects_cache_dir()

    # Create cache dir if needed
    if not effects_dir:
        # Try to create default cache dir
        for env_var in ["CACHE_DIR", "ARTDAG_CACHE_DIR"]:
            cache_dir = os.environ.get(env_var)
            if cache_dir:
                effects_dir = Path(cache_dir) / "_effects"
                effects_dir.mkdir(parents=True, exist_ok=True)
                break

        if not effects_dir:
            effects_dir = Path.home() / ".artdag" / "cache" / "_effects"
            effects_dir.mkdir(parents=True, exist_ok=True)

    effect_path = effects_dir / effect_cid / "effect.py"

    # If not cached locally, fetch from IPFS
    if not effect_path.exists():
        if not _fetch_effect_from_ipfs(effect_cid, effect_path):
            logger.warning(f"Effect not found: {effect_cid[:16]}...")
            return None

    # Parse and install dependencies before loading
    try:
        source = effect_path.read_text()
        dependencies = _parse_pep723_dependencies(source)
        if dependencies:
            logger.info(f"Effect {effect_cid[:16]}... requires: {dependencies}")
            if not _ensure_dependencies(dependencies, effect_cid):
                logger.error(f"Failed to install dependencies for effect {effect_cid[:16]}...")
                return None
    except Exception as e:
        logger.error(f"Error parsing effect dependencies: {e}")
        # Continue anyway - the effect might work without the deps check

    # Load the effect module
    try:
        spec = importlib.util.spec_from_file_location("cached_effect", effect_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check for frame-by-frame API
        if hasattr(module, "process_frame"):
            return _wrap_frame_effect(module, effect_path)

        # Check for whole-video API
        if hasattr(module, "process"):
            return _wrap_video_effect(module)

        # Check for old-style effect function
        if hasattr(module, "effect"):
            return module.effect

        logger.warning(f"Effect has no recognized API: {effect_cid[:16]}...")
        return None

    except Exception as e:
        logger.error(f"Failed to load effect {effect_cid[:16]}...: {e}")
        return None


def _wrap_frame_effect(module: ModuleType, effect_path: Path) -> EffectFn:
    """Wrap a frame-by-frame effect to work with the executor API."""

    def wrapped_effect(input_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
        """Run frame-by-frame effect through FFmpeg pipes."""
        try:
            from ..effects.frame_processor import process_video
        except ImportError:
            logger.error("Frame processor not available - falling back to copy")
            shutil.copy2(input_path, output_path)
            return output_path

        # Extract params from config (excluding internal keys)
        params = {k: v for k, v in config.items()
                  if k not in ("effect", "hash", "_binding")}

        # Get bindings if present
        bindings = {}
        for key, value in config.items():
            if isinstance(value, dict) and value.get("_resolved_values"):
                bindings[key] = value["_resolved_values"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_output = output_path.with_suffix(".mp4")

        process_video(
            input_path=input_path,
            output_path=actual_output,
            process_frame=module.process_frame,
            params=params,
            bindings=bindings,
        )

        return actual_output

    return wrapped_effect


def _wrap_video_effect(module: ModuleType) -> EffectFn:
    """Wrap a whole-video effect to work with the executor API."""

    def wrapped_effect(input_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
        """Run whole-video effect."""
        from ..effects.meta import ExecutionContext

        params = {k: v for k, v in config.items()
                  if k not in ("effect", "hash", "_binding")}

        output_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = ExecutionContext(
            input_paths=[str(input_path)],
            output_path=str(output_path),
            params=params,
            seed=hash(str(input_path)) & 0xFFFFFFFF,
        )

        module.process([input_path], output_path, params, ctx)
        return output_path

    return wrapped_effect


# Effect registry - maps effect names to implementations
_EFFECTS: Dict[str, EffectFn] = {}


def register_effect(name: str) -> Callable[[F], F]:
    """Decorator to register an effect implementation."""
    def decorator(func: F) -> F:
        _EFFECTS[name] = func  # type: ignore[assignment]
        return func
    return decorator


def get_effect(name: str) -> Optional[EffectFn]:
    """Get an effect implementation by name."""
    return _EFFECTS.get(name)


# Built-in effects

@register_effect("identity")
def effect_identity(input_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
    """
    Identity effect - returns input unchanged.

    This is the foundational effect: identity(x) = x
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing output if any
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()

    # Preserve extension from input
    actual_output = output_path.with_suffix(input_path.suffix)
    if actual_output.exists() or actual_output.is_symlink():
        actual_output.unlink()

    # Symlink to input (zero-copy identity)
    os.symlink(input_path.resolve(), actual_output)
    logger.debug(f"EFFECT identity: {input_path.name} -> {actual_output}")

    return actual_output


def _get_sexp_effect(effect_path: str, recipe_dir: Path = None) -> Optional[EffectFn]:
    """
    Load a sexp effect from a .sexp file.

    Args:
        effect_path: Relative path to the .sexp effect file
        recipe_dir: Base directory for resolving paths

    Returns:
        Effect function or None if not a sexp effect
    """
    if not effect_path or not effect_path.endswith(".sexp"):
        return None

    try:
        from ..sexp.effect_loader import SexpEffectLoader
    except ImportError:
        logger.warning("Sexp effect loader not available")
        return None

    try:
        loader = SexpEffectLoader(recipe_dir or Path.cwd())
        return loader.load_effect_path(effect_path)
    except Exception as e:
        logger.error(f"Failed to load sexp effect from {effect_path}: {e}")
        return None


def _get_python_primitive_effect(effect_name: str) -> Optional[EffectFn]:
    """
    Get a Python primitive frame processor effect.

    Checks if the effect has a python_primitive in FFmpegCompiler.EFFECT_MAPPINGS
    and wraps it for the executor API.
    """
    try:
        from ..sexp.ffmpeg_compiler import FFmpegCompiler
        from ..sexp.primitives import get_primitive
        from ..effects.frame_processor import process_video
    except ImportError:
        return None

    compiler = FFmpegCompiler()
    primitive_name = compiler.has_python_primitive(effect_name)
    if not primitive_name:
        return None

    primitive_fn = get_primitive(primitive_name)
    if not primitive_fn:
        logger.warning(f"Python primitive '{primitive_name}' not found for effect '{effect_name}'")
        return None

    def wrapped_effect(input_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
        """Run Python primitive effect via frame processor."""
        # Extract params (excluding internal keys)
        params = {k: v for k, v in config.items()
                  if k not in ("effect", "cid", "hash", "effect_path", "_binding")}

        # Get bindings if present
        bindings = {}
        for key, value in config.items():
            if isinstance(value, dict) and value.get("_resolved_values"):
                bindings[key] = value["_resolved_values"]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        actual_output = output_path.with_suffix(".mp4")

        # Wrap primitive to match frame processor signature
        def process_frame(frame, frame_params, state):
            # Call primitive with frame and params
            result = primitive_fn(frame, **frame_params)
            return result, state

        process_video(
            input_path=input_path,
            output_path=actual_output,
            process_frame=process_frame,
            params=params,
            bindings=bindings,
        )

        logger.info(f"Processed effect '{effect_name}' via Python primitive '{primitive_name}'")
        return actual_output

    return wrapped_effect


@register_executor("EFFECT")
class EffectExecutor(Executor):
    """
    Apply an effect from the registry or IPFS.

    Config:
        effect: Name of the effect to apply
        cid: IPFS CID for the effect (fetched from IPFS if not cached)
        hash: Legacy alias for cid (backwards compatibility)
        params: Optional parameters for the effect

    Inputs:
        Single input file to transform
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        effect_name = config.get("effect")
        # Support both "cid" (new) and "hash" (legacy)
        effect_cid = config.get("cid") or config.get("hash")

        if not effect_name:
            raise ValueError("EFFECT requires 'effect' config")

        if len(inputs) != 1:
            raise ValueError(f"EFFECT expects 1 input, got {len(inputs)}")

        # Try IPFS effect first if CID provided
        effect_fn: Optional[EffectFn] = None
        if effect_cid:
            effect_fn = _load_cached_effect(effect_cid)
            if effect_fn:
                logger.info(f"Running effect '{effect_name}' (cid={effect_cid[:16]}...)")

        # Try sexp effect from effect_path (.sexp file)
        if effect_fn is None:
            effect_path = config.get("effect_path")
            if effect_path and effect_path.endswith(".sexp"):
                effect_fn = _get_sexp_effect(effect_path)
                if effect_fn:
                    logger.info(f"Running effect '{effect_name}' via sexp definition")

        # Try Python primitive (from FFmpegCompiler.EFFECT_MAPPINGS)
        if effect_fn is None:
            effect_fn = _get_python_primitive_effect(effect_name)
            if effect_fn:
                logger.info(f"Running effect '{effect_name}' via Python primitive")

        # Fall back to built-in effect
        if effect_fn is None:
            effect_fn = get_effect(effect_name)

        if effect_fn is None:
            raise ValueError(f"Unknown effect: {effect_name}")

        # Pass full config (effect can extract what it needs)
        return effect_fn(inputs[0], output_path, config)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        if "effect" not in config:
            errors.append("EFFECT requires 'effect' config")
        else:
            # If CID provided, we'll load from IPFS - skip built-in check
            has_cid = config.get("cid") or config.get("hash")
            if not has_cid and get_effect(config["effect"]) is None:
                errors.append(f"Unknown effect: {config['effect']}")
        return errors
