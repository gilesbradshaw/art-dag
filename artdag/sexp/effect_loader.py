"""
Sexp effect loader.

Loads sexp effect definitions (define-effect forms) and creates
frame processors that evaluate the sexp body with primitives.

Usage:
    loader = SexpEffectLoader()
    effect_fn = loader.load_effect_file(Path("effects/ascii_art.sexp"))
    output = effect_fn(input_path, output_path, config)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .parser import parse_all, Symbol, Keyword
from .evaluator import evaluate
from .primitives import PRIMITIVES

logger = logging.getLogger(__name__)


def _parse_define_effect(sexp) -> tuple:
    """
    Parse a define-effect form.

    (define-effect name
      ((param1 default1) (param2 default2) ...)
      body)

    Returns (name, params_with_defaults, body)
    """
    if not isinstance(sexp, list) or len(sexp) < 4:
        raise ValueError(f"Invalid define-effect form: {sexp}")

    head = sexp[0]
    if not (isinstance(head, Symbol) and head.name == "define-effect"):
        raise ValueError(f"Expected define-effect, got {head}")

    name = sexp[1]
    if isinstance(name, Symbol):
        name = name.name

    # Parse params with defaults (evaluating default values)
    params_list = sexp[2]
    params_with_defaults = {}

    # Create minimal env for evaluating defaults
    default_env = {
        "list": lambda *args: tuple(args),  # (list 0 0 0) -> (0, 0, 0)
    }

    if isinstance(params_list, list):
        for param in params_list:
            if isinstance(param, list) and len(param) == 2:
                param_name = param[0].name if isinstance(param[0], Symbol) else param[0]
                param_default = param[1]
                # Evaluate default if it's an expression
                if isinstance(param_default, list) and param_default:
                    try:
                        param_default = evaluate(param_default, default_env)
                    except Exception:
                        pass  # Keep as-is if eval fails
                params_with_defaults[param_name] = param_default
            elif isinstance(param, Symbol):
                params_with_defaults[param.name] = None

    body = sexp[3]

    return name, params_with_defaults, body


def _create_process_frame(
    effect_name: str,
    params_with_defaults: Dict[str, Any],
    body: Any,
) -> Callable:
    """
    Create a process_frame function that evaluates the sexp body.

    The function signature is: (frame, params, state) -> (frame, state)
    """
    import math

    def process_frame(frame: np.ndarray, params: Dict[str, Any], state: Any):
        """Evaluate sexp effect body on a frame."""
        # Build environment with primitives
        env = dict(PRIMITIVES)

        # Add math functions
        env["floor"] = lambda x: int(math.floor(x))
        env["ceil"] = lambda x: int(math.ceil(x))
        env["round"] = lambda x: int(round(x))
        env["abs"] = abs
        env["min"] = min
        env["max"] = max
        env["sqrt"] = math.sqrt
        env["sin"] = math.sin
        env["cos"] = math.cos

        # Add list operations
        env["list"] = lambda *args: tuple(args)
        env["nth"] = lambda coll, i: coll[int(i)] if coll else None

        # Bind frame
        env["frame"] = frame

        # Bind parameters (defaults + overrides from config)
        for param_name, default in params_with_defaults.items():
            # Use config value if provided, otherwise default
            if param_name in params:
                env[param_name] = params[param_name]
            elif default is not None:
                env[param_name] = default

        # Also copy any extra params from config
        for k, v in params.items():
            if k not in env:
                env[k] = v

        # Evaluate the body
        try:
            result = evaluate(body, env)
            if isinstance(result, np.ndarray):
                return result, state
            else:
                logger.warning(f"Effect {effect_name} returned {type(result)}, expected ndarray")
                return frame, state
        except Exception as e:
            logger.error(f"Error evaluating effect {effect_name}: {e}")
            raise

    return process_frame


def load_sexp_effect(source: str, base_path: Optional[Path] = None) -> tuple:
    """
    Load a sexp effect from source code.

    Args:
        source: Sexp source code
        base_path: Base path for resolving relative imports

    Returns:
        (effect_name, process_frame_fn, params_with_defaults)
    """
    exprs = parse_all(source)

    # Find define-effect form
    define_effect = None
    if isinstance(exprs, list):
        for expr in exprs:
            if isinstance(expr, list) and expr and isinstance(expr[0], Symbol):
                if expr[0].name == "define-effect":
                    define_effect = expr
                    break
    elif isinstance(exprs, list) and exprs and isinstance(exprs[0], Symbol):
        if exprs[0].name == "define-effect":
            define_effect = exprs

    if not define_effect:
        raise ValueError("No define-effect form found in sexp effect")

    name, params_with_defaults, body = _parse_define_effect(define_effect)
    process_frame = _create_process_frame(name, params_with_defaults, body)

    return name, process_frame, params_with_defaults


def load_sexp_effect_file(path: Path) -> tuple:
    """
    Load a sexp effect from file.

    Returns:
        (effect_name, process_frame_fn, params_with_defaults)
    """
    source = path.read_text()
    return load_sexp_effect(source, base_path=path.parent)


class SexpEffectLoader:
    """
    Loader for sexp effect definitions.

    Creates effect functions compatible with the EffectExecutor.
    """

    def __init__(self, recipe_dir: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            recipe_dir: Base directory for resolving relative effect paths
        """
        self.recipe_dir = recipe_dir or Path.cwd()

    def load_effect_path(self, effect_path: str) -> Callable:
        """
        Load a sexp effect from a relative path.

        Args:
            effect_path: Relative path to effect .sexp file

        Returns:
            Effect function (input_path, output_path, config) -> output_path
        """
        from ..effects.frame_processor import process_video

        full_path = self.recipe_dir / effect_path
        if not full_path.exists():
            raise FileNotFoundError(f"Sexp effect not found: {full_path}")

        name, process_frame_fn, params_defaults = load_sexp_effect_file(full_path)
        logger.info(f"Loaded sexp effect: {name} from {effect_path}")

        def effect_fn(input_path: Path, output_path: Path, config: Dict[str, Any]) -> Path:
            """Run sexp effect via frame processor."""
            # Extract params (excluding internal keys)
            params = dict(params_defaults)  # Start with defaults
            for k, v in config.items():
                if k not in ("effect", "cid", "hash", "effect_path", "_binding"):
                    params[k] = v

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
                process_frame=process_frame_fn,
                params=params,
                bindings=bindings,
            )

            logger.info(f"Processed sexp effect '{name}' from {effect_path}")
            return actual_output

        return effect_fn


def get_sexp_effect_loader(recipe_dir: Optional[Path] = None) -> SexpEffectLoader:
    """Get a sexp effect loader instance."""
    return SexpEffectLoader(recipe_dir)
