"""
Sexp effect loader.

Loads sexp effect definitions (define-effect forms) and creates
frame processors that evaluate the sexp body with primitives.

Effects must use :params syntax:

    (define-effect name
      :params (
        (param1 :type int :default 8 :range [4 32] :desc "description")
        (param2 :type string :default "value" :desc "description")
      )
      body)

For effects with no parameters, use empty :params ():

    (define-effect name
      :params ()
      body)

Unknown parameters passed to effects will raise an error.

Usage:
    loader = SexpEffectLoader()
    effect_fn = loader.load_effect_file(Path("effects/ascii_art.sexp"))
    output = effect_fn(input_path, output_path, config)
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .parser import parse_all, Symbol, Keyword
from .evaluator import evaluate
from .primitives import PRIMITIVES
from .compiler import ParamDef, _parse_params, CompileError

logger = logging.getLogger(__name__)


def _parse_define_effect(sexp) -> tuple:
    """
    Parse a define-effect form.

    Required syntax:
        (define-effect name
          :params (
            (param1 :type int :default 8 :range [4 32] :desc "description")
          )
          body)

    Effects MUST use :params syntax. Legacy ((param default) ...) syntax is not supported.

    Returns (name, params_with_defaults, param_defs, body)
    where param_defs is a list of ParamDef objects
    """
    if not isinstance(sexp, list) or len(sexp) < 3:
        raise ValueError(f"Invalid define-effect form: {sexp}")

    head = sexp[0]
    if not (isinstance(head, Symbol) and head.name == "define-effect"):
        raise ValueError(f"Expected define-effect, got {head}")

    name = sexp[1]
    if isinstance(name, Symbol):
        name = name.name

    params_with_defaults = {}
    param_defs: List[ParamDef] = []
    body = None
    found_params = False

    # Parse :params and body
    i = 2
    while i < len(sexp):
        item = sexp[i]
        if isinstance(item, Keyword) and item.name == "params":
            # :params syntax
            if i + 1 >= len(sexp):
                raise ValueError(f"Effect '{name}': Missing params list after :params keyword")
            try:
                param_defs = _parse_params(sexp[i + 1])
                # Build params_with_defaults from ParamDef objects
                for pd in param_defs:
                    params_with_defaults[pd.name] = pd.default
            except CompileError as e:
                raise ValueError(f"Effect '{name}': Error parsing :params: {e}")
            found_params = True
            i += 2
        elif isinstance(item, Keyword):
            # Skip other keywords we don't recognize
            i += 2
        elif body is None:
            # First non-keyword item is the body
            if isinstance(item, list) and item:
                first_elem = item[0]
                # Check for legacy syntax and reject it
                if isinstance(first_elem, list) and len(first_elem) >= 2:
                    raise ValueError(
                        f"Effect '{name}': Legacy parameter syntax ((name default) ...) is not supported. "
                        f"Use :params block instead:\n"
                        f"  :params (\n"
                        f"    (param_name :type int :default 0 :desc \"description\")\n"
                        f"  )"
                    )
            body = item
            i += 1
        else:
            i += 1

    if body is None:
        raise ValueError(f"Effect '{name}': No body found")

    if not found_params:
        raise ValueError(
            f"Effect '{name}': Missing :params block. Effects must declare parameters.\n"
            f"For effects with no parameters, use empty :params ():\n"
            f"  (define-effect {name}\n"
            f"    :params ()\n"
            f"    body)"
        )

    return name, params_with_defaults, param_defs, body


def _create_process_frame(
    effect_name: str,
    params_with_defaults: Dict[str, Any],
    param_defs: List[ParamDef],
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

        # Validate that all provided params are known
        known_params = set(params_with_defaults.keys())
        for k in params.keys():
            if k not in known_params:
                raise ValueError(
                    f"Effect '{effect_name}': Unknown parameter '{k}'. "
                    f"Valid parameters are: {', '.join(sorted(known_params)) if known_params else '(none)'}"
                )

        # Bind parameters (defaults + overrides from config)
        for param_name, default in params_with_defaults.items():
            # Use config value if provided, otherwise default
            if param_name in params:
                env[param_name] = params[param_name]
            elif default is not None:
                env[param_name] = default

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
        (effect_name, process_frame_fn, params_with_defaults, param_defs)
        where param_defs is a list of ParamDef objects for introspection
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

    name, params_with_defaults, param_defs, body = _parse_define_effect(define_effect)
    process_frame = _create_process_frame(name, params_with_defaults, param_defs, body)

    return name, process_frame, params_with_defaults, param_defs


def load_sexp_effect_file(path: Path) -> tuple:
    """
    Load a sexp effect from file.

    Returns:
        (effect_name, process_frame_fn, params_with_defaults, param_defs)
        where param_defs is a list of ParamDef objects for introspection
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
        # Cache loaded effects with their param_defs for introspection
        self._loaded_effects: Dict[str, tuple] = {}

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

        name, process_frame_fn, params_defaults, param_defs = load_sexp_effect_file(full_path)
        logger.info(f"Loaded sexp effect: {name} from {effect_path}")

        # Cache for introspection
        self._loaded_effects[effect_path] = (name, params_defaults, param_defs)

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

    def get_effect_params(self, effect_path: str) -> List[ParamDef]:
        """
        Get parameter definitions for an effect.

        Args:
            effect_path: Relative path to effect .sexp file

        Returns:
            List of ParamDef objects describing the effect's parameters
        """
        if effect_path not in self._loaded_effects:
            # Load the effect to get its params
            full_path = self.recipe_dir / effect_path
            if not full_path.exists():
                raise FileNotFoundError(f"Sexp effect not found: {full_path}")
            name, _, params_defaults, param_defs = load_sexp_effect_file(full_path)
            self._loaded_effects[effect_path] = (name, params_defaults, param_defs)

        return self._loaded_effects[effect_path][2]


def get_sexp_effect_loader(recipe_dir: Optional[Path] = None) -> SexpEffectLoader:
    """Get a sexp effect loader instance."""
    return SexpEffectLoader(recipe_dir)
