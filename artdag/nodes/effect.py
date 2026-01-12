# artdag/nodes/effect.py
"""
Effect executor: Apply effects from the registry or cache.

Primitives: EFFECT

Effects can be:
1. Built-in (registered with @register_effect)
2. Cached (referenced by content hash)
"""

import importlib.util
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..executor import Executor, register_executor

logger = logging.getLogger(__name__)


def _get_effects_cache_dir() -> Optional[Path]:
    """Get the effects cache directory from environment or default."""
    cache_dir = os.environ.get("ARTDAG_CACHE_DIR")
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


def _load_cached_effect(effect_hash: str) -> Optional[callable]:
    """
    Load a cached effect by content hash.

    Returns the effect function or None if not found.
    """
    effects_dir = _get_effects_cache_dir()
    if not effects_dir:
        logger.warning("Effects cache directory not found")
        return None

    effect_path = effects_dir / effect_hash / "effect.py"
    if not effect_path.exists():
        logger.warning(f"Cached effect not found: {effect_hash[:16]}...")
        return None

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

        logger.warning(f"Cached effect has no recognized API: {effect_hash[:16]}...")
        return None

    except Exception as e:
        logger.error(f"Failed to load cached effect {effect_hash[:16]}...: {e}")
        return None


def _wrap_frame_effect(module, effect_path: Path) -> callable:
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


def _wrap_video_effect(module) -> callable:
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
_EFFECTS: Dict[str, callable] = {}


def register_effect(name: str):
    """Decorator to register an effect implementation."""
    def decorator(func):
        _EFFECTS[name] = func
        return func
    return decorator


def get_effect(name: str):
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


@register_executor("EFFECT")
class EffectExecutor(Executor):
    """
    Apply an effect from the registry or cache.

    Config:
        effect: Name of the effect to apply
        hash: Optional content hash for cached effects
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
        effect_hash = config.get("hash")

        if not effect_name:
            raise ValueError("EFFECT requires 'effect' config")

        if len(inputs) != 1:
            raise ValueError(f"EFFECT expects 1 input, got {len(inputs)}")

        # Try cached effect first if hash provided
        effect_fn = None
        if effect_hash:
            effect_fn = _load_cached_effect(effect_hash)
            if effect_fn:
                logger.info(f"Running cached effect '{effect_name}' ({effect_hash[:16]}...)")

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
            # If hash provided, we'll load from cache - skip built-in check
            if not config.get("hash") and get_effect(config["effect"]) is None:
                errors.append(f"Unknown effect: {config['effect']}")
        return errors
