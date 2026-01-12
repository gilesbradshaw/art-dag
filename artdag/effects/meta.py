"""
Effect metadata types.

Defines the core dataclasses for effect metadata:
- ParamSpec: Parameter specification with type, range, and default
- EffectMeta: Complete effect metadata including params and flags
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union


@dataclass
class ParamSpec:
    """
    Specification for an effect parameter.

    Attributes:
        name: Parameter name (used in recipes as :name)
        param_type: Python type (float, int, bool, str)
        default: Default value if not specified
        range: Optional (min, max) tuple for numeric types
        description: Human-readable description
        choices: Optional list of allowed values (for enums)
    """

    name: str
    param_type: Type
    default: Any = None
    range: Optional[Tuple[float, float]] = None
    description: str = ""
    choices: Optional[List[Any]] = None

    def validate(self, value: Any) -> Any:
        """
        Validate and coerce a parameter value.

        Args:
            value: Input value to validate

        Returns:
            Validated and coerced value

        Raises:
            ValueError: If value is invalid
        """
        if value is None:
            if self.default is not None:
                return self.default
            raise ValueError(f"Parameter '{self.name}' requires a value")

        # Type coercion
        try:
            if self.param_type == bool:
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                else:
                    value = bool(value)
            elif self.param_type == int:
                value = int(value)
            elif self.param_type == float:
                value = float(value)
            elif self.param_type == str:
                value = str(value)
            else:
                value = self.param_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Parameter '{self.name}' expects {self.param_type.__name__}, "
                f"got {type(value).__name__}: {e}"
            )

        # Range check for numeric types
        if self.range is not None and self.param_type in (int, float):
            min_val, max_val = self.range
            if value < min_val or value > max_val:
                raise ValueError(
                    f"Parameter '{self.name}' must be in range "
                    f"[{min_val}, {max_val}], got {value}"
                )

        # Choices check
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Parameter '{self.name}' must be one of {self.choices}, got {value}"
            )

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "name": self.name,
            "type": self.param_type.__name__,
            "description": self.description,
        }
        if self.default is not None:
            d["default"] = self.default
        if self.range is not None:
            d["range"] = list(self.range)
        if self.choices is not None:
            d["choices"] = self.choices
        return d


@dataclass
class EffectMeta:
    """
    Complete metadata for an effect.

    Attributes:
        name: Effect name (used in recipes)
        version: Semantic version string
        temporal: If True, effect needs complete input (can't be collapsed)
        params: List of parameter specifications
        author: Optional author identifier
        description: Human-readable description
        examples: List of example S-expression usages
        dependencies: List of Python package dependencies
        requires_python: Minimum Python version
        api_type: "frame" for frame-by-frame, "video" for whole-video
    """

    name: str
    version: str = "1.0.0"
    temporal: bool = False
    params: List[ParamSpec] = field(default_factory=list)
    author: str = ""
    description: str = ""
    examples: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    requires_python: str = ">=3.10"
    api_type: str = "frame"  # "frame" or "video"

    def get_param(self, name: str) -> Optional[ParamSpec]:
        """Get a parameter spec by name."""
        for param in self.params:
            if param.name == name:
                return param
        return None

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all parameters.

        Args:
            params: Dictionary of parameter values

        Returns:
            Dictionary with validated/coerced values including defaults

        Raises:
            ValueError: If any parameter is invalid
        """
        result = {}
        for spec in self.params:
            value = params.get(spec.name)
            result[spec.name] = spec.validate(value)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "temporal": self.temporal,
            "params": [p.to_dict() for p in self.params],
            "author": self.author,
            "description": self.description,
            "examples": self.examples,
            "dependencies": self.dependencies,
            "requires_python": self.requires_python,
            "api_type": self.api_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectMeta":
        """Create from dictionary."""
        params = []
        for p in data.get("params", []):
            # Map type name back to Python type
            type_map = {"float": float, "int": int, "bool": bool, "str": str}
            param_type = type_map.get(p.get("type", "float"), float)
            params.append(
                ParamSpec(
                    name=p["name"],
                    param_type=param_type,
                    default=p.get("default"),
                    range=tuple(p["range"]) if p.get("range") else None,
                    description=p.get("description", ""),
                    choices=p.get("choices"),
                )
            )

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            temporal=data.get("temporal", False),
            params=params,
            author=data.get("author", ""),
            description=data.get("description", ""),
            examples=data.get("examples", []),
            dependencies=data.get("dependencies", []),
            requires_python=data.get("requires_python", ">=3.10"),
            api_type=data.get("api_type", "frame"),
        )


@dataclass
class ExecutionContext:
    """
    Context passed to effect execution.

    Provides controlled access to resources within sandbox.
    """

    input_paths: List[str]
    output_path: str
    params: Dict[str, Any]
    seed: int  # Deterministic seed for RNG
    frame_rate: float = 30.0
    width: int = 1920
    height: int = 1080

    # Resolved bindings (frame -> param value lookup)
    bindings: Dict[str, List[float]] = field(default_factory=dict)

    def get_param_at_frame(self, param_name: str, frame: int) -> Any:
        """
        Get parameter value at a specific frame.

        If parameter has a binding, looks up the bound value.
        Otherwise returns the static parameter value.
        """
        if param_name in self.bindings:
            binding_values = self.bindings[param_name]
            if frame < len(binding_values):
                return binding_values[frame]
            # Past end of binding data, use last value
            return binding_values[-1] if binding_values else self.params.get(param_name)
        return self.params.get(param_name)

    def get_rng(self) -> "random.Random":
        """Get a seeded random number generator."""
        import random

        return random.Random(self.seed)
