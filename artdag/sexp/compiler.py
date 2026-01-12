"""
Compiler for S-expression recipes.

Transforms S-expression recipes into internal DAG format.
Handles:
- Threading macro expansion (->)
- def bindings for named nodes
- Registry resolution (assets, effects)
- Node ID generation (content-addressed)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json

from .parser import Symbol, Keyword, parse, serialize


class CompileError(Exception):
    """Error during recipe compilation."""
    pass


@dataclass
class CompiledRecipe:
    """Result of compiling an S-expression recipe."""
    name: str
    version: str
    description: str
    owner: Optional[str]
    registry: Dict[str, Dict[str, Any]]  # {assets: {...}, effects: {...}}
    nodes: List[Dict[str, Any]]  # List of node definitions
    output_node_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (compatible with YAML structure)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "owner": self.owner,
            "registry": self.registry,
            "dag": {
                "nodes": self.nodes,
                "output": self.output_node_id,
            },
            "metadata": self.metadata,
        }


@dataclass
class CompilerContext:
    """Compilation context tracking bindings and nodes."""
    registry: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"assets": {}, "effects": {}})
    bindings: Dict[str, str] = field(default_factory=dict)  # name -> node_id
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> node
    node_counter: int = 0

    def gen_node_id(self, prefix: str = "n") -> str:
        """Generate a unique node ID."""
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def add_node(self, node_type: str, config: Dict[str, Any],
                 inputs: List[str] = None, name: str = None) -> str:
        """Add a node and return its ID."""
        node_id = self.gen_node_id(node_type.lower())
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "config": config,
            "inputs": inputs or [],
            "name": name,
        }
        return node_id


def compile_recipe(sexp: Any) -> CompiledRecipe:
    """
    Compile an S-expression recipe into internal format.

    Args:
        sexp: Parsed S-expression (list starting with 'recipe' symbol)

    Returns:
        CompiledRecipe with nodes and registry

    Example:
        >>> sexp = parse('(recipe "test" :version "1.0" (-> (source cat) (effect identity)))')
        >>> result = compile_recipe(sexp)
    """
    if not isinstance(sexp, list) or len(sexp) < 2:
        raise CompileError("Recipe must be a list starting with 'recipe'")

    head = sexp[0]
    if not (isinstance(head, Symbol) and head.name == "recipe"):
        raise CompileError(f"Expected 'recipe', got {head}")

    # Extract recipe name
    if len(sexp) < 2 or not isinstance(sexp[1], str):
        raise CompileError("Recipe name must be a string")
    name = sexp[1]

    # Parse keyword arguments and body
    ctx = CompilerContext()
    version = "1.0"
    description = ""
    owner = None
    body_exprs = []

    i = 2
    while i < len(sexp):
        item = sexp[i]

        if isinstance(item, Keyword):
            if i + 1 >= len(sexp):
                raise CompileError(f"Keyword {item.name} missing value")
            value = sexp[i + 1]

            if item.name == "version":
                version = str(value)
            elif item.name == "description":
                description = str(value)
            elif item.name == "owner":
                owner = str(value)
            else:
                raise CompileError(f"Unknown keyword :{item.name}")
            i += 2
        else:
            # Body expression
            body_exprs.append(item)
            i += 1

    # Compile body expressions
    output_node_id = None
    for expr in body_exprs:
        result = _compile_expr(expr, ctx)
        if result is not None:
            output_node_id = result

    if output_node_id is None:
        raise CompileError("Recipe has no output (no DAG expression)")

    return CompiledRecipe(
        name=name,
        version=version,
        description=description,
        owner=owner,
        registry=ctx.registry,
        nodes=list(ctx.nodes.values()),
        output_node_id=output_node_id,
    )


def _compile_expr(expr: Any, ctx: CompilerContext) -> Optional[str]:
    """
    Compile an expression, returning node_id if it produces a node.

    Handles:
    - (asset name :hash "..." :url "...")
    - (effect name :hash "..." :url "...")
    - (def name expr)
    - (-> expr expr ...)
    - (source ...), (effect ...), (sequence ...), etc.
    """
    if not isinstance(expr, list) or len(expr) == 0:
        # Atom - could be a reference
        if isinstance(expr, Symbol):
            # Look up binding
            if expr.name in ctx.bindings:
                return ctx.bindings[expr.name]
            raise CompileError(f"Undefined symbol: {expr.name}")
        return None

    head = expr[0]
    if not isinstance(head, Symbol):
        raise CompileError(f"Expected symbol at head of expression, got {head}")

    name = head.name

    # Registry declarations
    if name == "asset":
        return _compile_asset(expr, ctx)
    if name == "effect":
        return _compile_effect_decl(expr, ctx)

    # Binding
    if name == "def":
        return _compile_def(expr, ctx)

    # Threading macro
    if name == "->":
        return _compile_threading(expr, ctx)

    # Node types
    if name == "source":
        return _compile_source(expr, ctx)
    if name in ("effect", "fx"):
        return _compile_effect_node(expr, ctx)
    if name == "segment":
        return _compile_segment(expr, ctx)
    if name == "resize":
        return _compile_resize(expr, ctx)
    if name == "transform":
        return _compile_transform(expr, ctx)
    if name == "sequence":
        return _compile_sequence(expr, ctx)
    if name == "layer":
        return _compile_layer(expr, ctx)
    if name == "blend":
        return _compile_blend(expr, ctx)
    if name == "mux":
        return _compile_mux(expr, ctx)
    if name == "analyze":
        return _compile_analyze(expr, ctx)

    # Binding expression for parameter linking
    if name == "bind":
        return _compile_bind(expr, ctx)

    raise CompileError(f"Unknown expression type: {name}")


def _parse_kwargs(expr: List, start: int = 1) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Parse positional args and keyword args from expression.

    Returns (positional_args, keyword_dict)
    """
    positional = []
    kwargs = {}

    i = start
    while i < len(expr):
        item = expr[i]
        if isinstance(item, Keyword):
            if i + 1 >= len(expr):
                raise CompileError(f"Keyword :{item.name} missing value")
            kwargs[item.name] = expr[i + 1]
            i += 2
        else:
            positional.append(item)
            i += 1

    return positional, kwargs


def _compile_asset(expr: List, ctx: CompilerContext) -> None:
    """Compile (asset name :hash "..." :url "...")"""
    if len(expr) < 2:
        raise CompileError("asset requires a name")

    name = expr[1]
    if isinstance(name, Symbol):
        name = name.name

    _, kwargs = _parse_kwargs(expr, 2)

    if "hash" not in kwargs:
        raise CompileError(f"asset {name} requires :hash")

    ctx.registry["assets"][name] = {
        "hash": kwargs["hash"],
        "url": kwargs.get("url"),
    }
    return None


def _compile_effect_decl(expr: List, ctx: CompilerContext) -> Optional[str]:
    """
    Compile effect - either declaration or node.

    Declaration: (effect name :hash "..." :url "...")
    Node: (effect effect-name) or (effect effect-name input-node)
    """
    if len(expr) < 2:
        raise CompileError("effect requires at least a name")

    # Check if this is a declaration (has :hash)
    _, kwargs = _parse_kwargs(expr, 2)

    if "hash" in kwargs:
        # Declaration
        name = expr[1]
        if isinstance(name, Symbol):
            name = name.name

        # Handle temporal flag - could be Symbol('true') or Python bool
        temporal = kwargs.get("temporal", False)
        if isinstance(temporal, Symbol):
            temporal = temporal.name.lower() == "true"

        ctx.registry["effects"][name] = {
            "hash": kwargs["hash"],
            "url": kwargs.get("url"),
            "temporal": temporal,
        }
        return None

    # Otherwise it's a node - delegate to effect node compiler
    return _compile_effect_node(expr, ctx)


def _compile_def(expr: List, ctx: CompilerContext) -> None:
    """Compile (def name expr)"""
    if len(expr) != 3:
        raise CompileError("def requires exactly 2 arguments: name and expression")

    name = expr[1]
    if not isinstance(name, Symbol):
        raise CompileError(f"def name must be a symbol, got {name}")

    body = expr[2]
    node_id = _compile_expr(body, ctx)

    if node_id is None:
        raise CompileError(f"def body must produce a node")

    ctx.bindings[name.name] = node_id
    return None


def _compile_threading(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (-> expr1 expr2 expr3 ...)

    Each expression's output becomes the implicit first input of the next.
    """
    if len(expr) < 2:
        raise CompileError("-> requires at least one expression")

    prev_node_id = None

    for i, sub_expr in enumerate(expr[1:]):
        if prev_node_id is not None:
            # Inject previous node as first input
            sub_expr = _inject_input(sub_expr, prev_node_id)

        prev_node_id = _compile_expr(sub_expr, ctx)

        if prev_node_id is None:
            raise CompileError(f"Expression {i} in -> chain produced no node")

    return prev_node_id


def _inject_input(expr: Any, input_id: str) -> List:
    """Inject an input node ID into an expression."""
    if not isinstance(expr, list):
        # Symbol reference - wrap in a node that takes input
        if isinstance(expr, Symbol):
            # Assume it's an effect name
            return [Symbol("effect"), expr, Symbol(f"__input_{input_id}")]
        raise CompileError(f"Cannot inject input into {expr}")

    # For node expressions, we'll handle the input in the compiler
    # Mark it with a special __prev__ reference
    return expr + [Symbol("__prev__"), input_id]


def _resolve_input(arg: Any, ctx: CompilerContext, prev_id: str = None) -> str:
    """Resolve an argument to a node ID."""
    if isinstance(arg, Symbol):
        if arg.name == "__prev__":
            if prev_id is None:
                raise CompileError("__prev__ used outside threading context")
            return prev_id
        if arg.name.startswith("__input_"):
            return arg.name[8:]  # Strip __input_ prefix
        if arg.name in ctx.bindings:
            return ctx.bindings[arg.name]
        raise CompileError(f"Undefined reference: {arg.name}")

    if isinstance(arg, str):
        # Direct node ID
        return arg

    if isinstance(arg, list):
        # Nested expression
        return _compile_expr(arg, ctx)

    raise CompileError(f"Cannot resolve input: {arg}")


def _extract_prev_id(args: List, kwargs: Dict) -> Tuple[List, Dict, Optional[str]]:
    """Extract __prev__ marker from args if present."""
    prev_id = None
    new_args = []

    i = 0
    while i < len(args):
        if isinstance(args[i], Symbol) and args[i].name == "__prev__":
            if i + 1 < len(args):
                prev_id = args[i + 1]
                i += 2
                continue
        new_args.append(args[i])
        i += 1

    return new_args, kwargs, prev_id


def _compile_source(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (source asset-name) or (source :input "name" :description "...").
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, _ = _extract_prev_id(args, kwargs)

    if "input" in kwargs:
        # Variable input - :input can be followed by a name string
        input_val = kwargs["input"]
        if isinstance(input_val, str):
            # (source :input "User Video" :description "...")
            name = input_val
        else:
            # (source :input true :name "User Video")
            name = kwargs.get("name", "Input")
        config = {
            "input": True,
            "name": name,
            "description": kwargs.get("description", ""),
        }
    elif args:
        # Asset reference
        asset_name = args[0]
        if isinstance(asset_name, Symbol):
            asset_name = asset_name.name
        config = {"asset": asset_name}
    else:
        raise CompileError("source requires asset name or :input flag")

    return ctx.add_node("SOURCE", config)


def _compile_effect_node(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (effect effect-name [input-node] :param value ...).

    Parameters can be literals or bind expressions:
        (fx brightness :level 0.5)
        (fx brightness :level (bind analysis :energy :range [0 1]))
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    if not args:
        raise CompileError("effect requires effect name")

    effect_name = args[0]
    if isinstance(effect_name, Symbol):
        effect_name = effect_name.name

    config = {"effect": effect_name}

    # Process parameter values, looking for bind expressions
    for k, v in kwargs.items():
        if k not in ("hash", "url"):
            config[k] = _process_value(v, ctx)

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args[1:]:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("EFFECT", config, inputs)


def _compile_segment(expr: List, ctx: CompilerContext) -> str:
    """Compile (segment :start 0.0 :end 2.0 [input])."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    config = {}
    if "start" in kwargs:
        config["start"] = float(kwargs["start"])
    if "end" in kwargs:
        config["end"] = float(kwargs["end"])
    if "duration" in kwargs:
        config["duration"] = float(kwargs["duration"])

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("SEGMENT", config, inputs)


def _compile_resize(expr: List, ctx: CompilerContext) -> str:
    """Compile (resize width height :mode "fit" [input])."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    if len(args) < 2:
        raise CompileError("resize requires width and height")

    config = {
        "width": int(args[0]),
        "height": int(args[1]),
        "mode": kwargs.get("mode", "fit"),
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args[2:]:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("RESIZE", config, inputs)


def _compile_transform(expr: List, ctx: CompilerContext) -> str:
    """Compile (transform :saturation 1.5 :brightness 0.8 [input])."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    # All kwargs except special ones become effect parameters
    effects = {k: v for k, v in kwargs.items()}
    config = {"effects": effects}

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("TRANSFORM", config, inputs)


def _compile_sequence(expr: List, ctx: CompilerContext) -> str:
    """Compile (sequence node1 node2 ... :transition {...})."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    config = {
        "transition": kwargs.get("transition", {"type": "cut"}),
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("SEQUENCE", config, inputs)


def _compile_layer(expr: List, ctx: CompilerContext) -> str:
    """Compile (layer node1 node2 ...)."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    config = {"inputs": [{}] * len(inputs)}

    return ctx.add_node("LAYER", config, inputs)


def _compile_blend(expr: List, ctx: CompilerContext) -> str:
    """Compile (blend node1 node2 :mode "overlay" :opacity 0.5)."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    config = {
        "mode": kwargs.get("mode", "overlay"),
        "opacity": float(kwargs.get("opacity", 0.5)),
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    if len(inputs) < 2:
        raise CompileError("blend requires two inputs")

    return ctx.add_node("BLEND", config, inputs)


def _compile_mux(expr: List, ctx: CompilerContext) -> str:
    """Compile (mux video-node audio-node)."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    config = {
        "video_stream": 0,
        "audio_stream": 1,
        "shortest": kwargs.get("shortest", True),
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    if len(inputs) < 2:
        raise CompileError("mux requires video and audio inputs")

    return ctx.add_node("MUX", config, inputs)


def _compile_analyze(expr: List, ctx: CompilerContext) -> str:
    """Compile (analyze input :beats :energy ...)."""
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    # Keywords become analysis types
    analysis_types = list(kwargs.keys())

    config = {
        "analysis_types": analysis_types,
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("ANALYZE", config, inputs)


def _compile_bind(expr: List, ctx: CompilerContext) -> Dict[str, Any]:
    """
    Compile (bind source feature :option value ...).

    Returns a binding specification dict (not a node ID).

    Examples:
        (bind analysis :energy)
        (bind analysis :energy :range [0 1])
        (bind analysis :beats :on-event 1.0 :decay 0.1)
        (bind analysis :energy :range [0 1] :smooth 0.05 :noise 0.1 :seed 42)
    """
    args, kwargs = _parse_kwargs(expr, 1)

    if len(args) < 2:
        raise CompileError("bind requires source and feature: (bind source :feature ...)")

    source = args[0]
    feature = args[1]

    # Source can be a symbol reference
    source_ref = None
    if isinstance(source, Symbol):
        if source.name in ctx.bindings:
            source_ref = ctx.bindings[source.name]
        else:
            source_ref = source.name

    # Feature should be a keyword
    feature_name = None
    if isinstance(feature, Keyword):
        feature_name = feature.name
    elif isinstance(feature, Symbol):
        feature_name = feature.name
    else:
        raise CompileError(f"bind feature must be a keyword, got {feature}")

    binding = {
        "_binding": True,  # Marker for binding resolution
        "source": source_ref,
        "feature": feature_name,
    }

    # Add optional binding modifiers
    if "range" in kwargs:
        range_val = kwargs["range"]
        if isinstance(range_val, list) and len(range_val) == 2:
            binding["range"] = [float(range_val[0]), float(range_val[1])]
        else:
            raise CompileError("bind :range must be [lo hi]")

    if "smooth" in kwargs:
        binding["smooth"] = float(kwargs["smooth"])

    if "offset" in kwargs:
        binding["offset"] = float(kwargs["offset"])

    if "on-event" in kwargs:
        binding["on_event"] = float(kwargs["on-event"])

    if "decay" in kwargs:
        binding["decay"] = float(kwargs["decay"])

    if "noise" in kwargs:
        binding["noise"] = float(kwargs["noise"])

    if "seed" in kwargs:
        binding["seed"] = int(kwargs["seed"])

    return binding


def _process_value(value: Any, ctx: CompilerContext) -> Any:
    """
    Process a value, resolving nested expressions like bind.

    Returns the processed value (could be a binding dict, node ref, or literal).
    """
    if isinstance(value, list) and len(value) > 0:
        head = value[0]
        if isinstance(head, Symbol) and head.name == "bind":
            return _compile_bind(value, ctx)
        # Could be other nested expressions
        return _compile_expr(value, ctx)
    return value


def compile_string(text: str) -> CompiledRecipe:
    """
    Compile an S-expression recipe string.

    Convenience function combining parse + compile.
    """
    sexp = parse(text)
    return compile_recipe(sexp)
