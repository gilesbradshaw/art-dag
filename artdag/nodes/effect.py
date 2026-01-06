# artdag/nodes/effect.py
"""
Effect executor: Apply effects from the registry.

Primitives: EFFECT
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from ..executor import Executor, register_executor

logger = logging.getLogger(__name__)


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
    Apply an effect from the registry.

    Config:
        effect: Name of the effect to apply
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
        if not effect_name:
            raise ValueError("EFFECT requires 'effect' config")

        effect_fn = get_effect(effect_name)
        if effect_fn is None:
            raise ValueError(f"Unknown effect: {effect_name}")

        if len(inputs) != 1:
            raise ValueError(f"EFFECT expects 1 input, got {len(inputs)}")

        effect_params = config.get("params", {})
        return effect_fn(inputs[0], output_path, effect_params)

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        if "effect" not in config:
            errors.append("EFFECT requires 'effect' config")
        elif get_effect(config["effect"]) is None:
            errors.append(f"Unknown effect: {config['effect']}")
        return errors
