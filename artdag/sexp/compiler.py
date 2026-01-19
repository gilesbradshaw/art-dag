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

from .parser import Symbol, Keyword, Lambda, parse, serialize


def _serialize_for_hash(obj) -> str:
    """Serialize any value to canonical S-expression string for hashing."""
    if obj is None:
        return "nil"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, str):
        escaped = obj.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(obj, Symbol):
        return obj.name
    if isinstance(obj, Keyword):
        return f":{obj.name}"
    if isinstance(obj, Lambda):
        params = " ".join(obj.params)
        body = _serialize_for_hash(obj.body)
        return f"(fn [{params}] {body})"
    if isinstance(obj, dict):
        items = []
        for k, v in sorted(obj.items()):
            items.append(f":{k} {_serialize_for_hash(v)}")
        return "{" + " ".join(items) + "}"
    if isinstance(obj, list):
        items = [_serialize_for_hash(x) for x in obj]
        return "(" + " ".join(items) + ")"
    return str(obj)


class CompileError(Exception):
    """Error during recipe compilation."""
    pass


@dataclass
class ParamDef:
    """Definition of a recipe parameter."""
    name: str
    param_type: str  # "string", "int", "float", "bool"
    default: Any
    description: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    choices: Optional[List[str]] = None  # For enum-like params


@dataclass
class CompiledStage:
    """A compiled stage with dependencies and outputs."""
    name: str
    requires: List[str]  # Names of required stages
    inputs: List[str]  # Names of bindings consumed from required stages
    outputs: List[str]  # Names of bindings produced by this stage
    node_ids: List[str]  # Node IDs created in this stage
    output_bindings: Dict[str, str]  # output_name -> node_id mapping


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
    encoding: Dict[str, Any] = field(default_factory=dict)  # {codec, crf, preset, audio_codec}
    metadata: Dict[str, Any] = field(default_factory=dict)
    params: List[ParamDef] = field(default_factory=list)  # Declared parameters
    stages: List[CompiledStage] = field(default_factory=list)  # Compiled stages
    stage_order: List[str] = field(default_factory=list)  # Topologically sorted stage names

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
            "encoding": self.encoding,
            "metadata": self.metadata,
        }


@dataclass
class CompilerContext:
    """Compilation context tracking bindings and nodes."""
    registry: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {"assets": {}, "effects": {}, "analyzers": {}, "constructs": {}})
    bindings: Dict[str, str] = field(default_factory=dict)  # name -> node_id
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> node

    # Stage tracking
    current_stage: Optional[str] = None  # Name of stage currently being compiled
    defined_stages: Dict[str, 'CompiledStage'] = field(default_factory=dict)  # stage_name -> CompiledStage
    stage_bindings: Dict[str, Dict[str, str]] = field(default_factory=dict)  # stage_name -> {binding_name -> node_id}
    pre_stage_bindings: Dict[str, Any] = field(default_factory=dict)  # bindings defined before any stage
    stage_node_ids: List[str] = field(default_factory=list)  # node IDs created in current stage

    def add_node(self, node_type: str, config: Dict[str, Any],
                 inputs: List[str] = None, name: str = None) -> str:
        """
        Add a node and return its code-addressed ID.

        The node_id is a hash of the S-expression subtree (type, config, inputs),
        creating a Merkle-tree like a blockchain - each node's hash includes all
        upstream hashes. This is computed purely from the plan, before execution.

        The node_id is a pre-computed "bucket" where the computation result will
        be stored. Same plan = same buckets = automatic cache reuse.
        """
        # Build canonical S-expression for hashing
        # Inputs are already code-addressed node IDs (hashes)
        canonical = {
            "type": node_type,
            "config": config,
            "inputs": inputs or [],
        }
        # Hash the canonical S-expression form using SHA3-256
        canonical_sexp = _serialize_for_hash(canonical)
        node_id = hashlib.sha3_256(canonical_sexp.encode()).hexdigest()

        # Check for collision (same hash = same computation, reuse)
        if node_id in self.nodes:
            return node_id

        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "config": config,
            "inputs": inputs or [],
            "name": name,
        }

        # Track node in current stage
        if self.current_stage is not None:
            self.stage_node_ids.append(node_id)

        return node_id

    def get_accessible_bindings(self, stage_inputs: List[str] = None) -> Dict[str, Any]:
        """
        Get bindings accessible to the current stage.

        If inside a stage with declared inputs, only those inputs plus pre-stage
        bindings are accessible. If outside a stage, all bindings are accessible.
        """
        if self.current_stage is None:
            return dict(self.bindings)

        # Start with pre-stage bindings (sources, etc.)
        accessible = dict(self.pre_stage_bindings)

        # Add declared inputs from required stages
        if stage_inputs:
            for input_name in stage_inputs:
                # Look for the binding in required stages
                for stage_name, stage in self.defined_stages.items():
                    if input_name in stage.output_bindings:
                        accessible[input_name] = stage.output_bindings[input_name]
                        break
                else:
                    # Check if it's in pre-stage bindings (might be a source)
                    if input_name not in accessible:
                        raise CompileError(
                            f"Stage '{self.current_stage}' declares input '{input_name}' "
                            f"but it's not produced by any required stage"
                        )

        return accessible


def _topological_sort_stages(stages: Dict[str, 'CompiledStage']) -> List[str]:
    """
    Topologically sort stages by their dependencies.

    Returns list of stage names in execution order (dependencies first).
    """
    if not stages:
        return []

    # Build dependency graph
    in_degree = {name: 0 for name in stages}
    dependents = {name: [] for name in stages}

    for name, stage in stages.items():
        for req in stage.requires:
            if req in stages:
                dependents[req].append(name)
                in_degree[name] += 1

    # Kahn's algorithm
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        # Sort for deterministic ordering
        queue.sort()
        current = queue.pop(0)
        result.append(current)

        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(stages):
        # This shouldn't happen if we validated cycles earlier
        missing = set(stages.keys()) - set(result)
        raise CompileError(f"Circular stage dependency detected: {missing}")

    return result


def _parse_encoding(value: Any) -> Dict[str, Any]:
    """
    Parse encoding settings from S-expression.

    Expects a list like: (:codec "libx264" :crf 18 :preset "fast" :audio-codec "aac")
    Returns: {"codec": "libx264", "crf": 18, "preset": "fast", "audio_codec": "aac"}
    """
    if not isinstance(value, list):
        raise CompileError(f"Encoding must be a list, got {type(value).__name__}")

    result = {}
    i = 0
    while i < len(value):
        item = value[i]
        if isinstance(item, Keyword):
            if i + 1 >= len(value):
                raise CompileError(f"Encoding keyword {item.name} missing value")
            # Convert kebab-case to snake_case for Python
            key = item.name.replace("-", "_")
            result[key] = value[i + 1]
            i += 2
        else:
            raise CompileError(f"Expected keyword in encoding, got {type(item).__name__}")
    return result


def _parse_params(value: Any) -> List[ParamDef]:
    """
    Parse parameter definitions from S-expression.

    Syntax:
        :params (
          (param_name :type string :default "value" :desc "Description")
          (param_name :type float :default 1.0 :range [0 10] :desc "Description")
          (param_name :type string :default "a" :choices ["a" "b" "c"] :desc "Description")
        )

    Supported types: string, int, float, bool
    Optional: :range [min max], :choices [...], :desc "..."
    """
    if not isinstance(value, list):
        raise CompileError(f"Params must be a list, got {type(value).__name__}")

    params = []
    for param_def in value:
        if not isinstance(param_def, list) or len(param_def) < 1:
            raise CompileError(f"Invalid param definition: {param_def}")

        # First element is the parameter name
        first = param_def[0]
        if isinstance(first, Symbol):
            param_name = first.name
        elif isinstance(first, str):
            param_name = first
        else:
            raise CompileError(f"Param name must be symbol or string, got {type(first).__name__}")

        # Parse keyword arguments
        param_type = "string"
        default = None
        desc = ""
        range_min = None
        range_max = None
        choices = None

        i = 1
        while i < len(param_def):
            item = param_def[i]
            if isinstance(item, Keyword):
                if i + 1 >= len(param_def):
                    raise CompileError(f"Param keyword {item.name} missing value")
                kw_value = param_def[i + 1]

                if item.name == "type":
                    if isinstance(kw_value, Symbol):
                        param_type = kw_value.name
                    else:
                        param_type = str(kw_value)
                elif item.name == "default":
                    default = kw_value
                elif item.name == "desc" or item.name == "description":
                    desc = str(kw_value)
                elif item.name == "range":
                    if isinstance(kw_value, list) and len(kw_value) >= 2:
                        range_min = float(kw_value[0])
                        range_max = float(kw_value[1])
                    else:
                        raise CompileError(f"Param range must be [min max], got {kw_value}")
                elif item.name == "choices":
                    if isinstance(kw_value, list):
                        choices = [str(c) if not isinstance(c, Symbol) else c.name for c in kw_value]
                    else:
                        raise CompileError(f"Param choices must be a list, got {kw_value}")
                else:
                    raise CompileError(f"Unknown param keyword :{item.name}")
                i += 2
            else:
                i += 1

        # Convert default to appropriate type
        if default is not None:
            if param_type == "int":
                default = int(default)
            elif param_type == "float":
                default = float(default)
            elif param_type == "bool":
                if isinstance(default, (int, float)):
                    default = bool(default)
                elif isinstance(default, str):
                    default = default.lower() in ("true", "1", "yes")
            elif param_type == "string":
                default = str(default)

        params.append(ParamDef(
            name=param_name,
            param_type=param_type,
            default=default,
            description=desc,
            range_min=range_min,
            range_max=range_max,
            choices=choices,
        ))

    return params


def compile_recipe(sexp: Any, initial_bindings: Dict[str, Any] = None) -> CompiledRecipe:
    """
    Compile an S-expression recipe into internal format.

    Args:
        sexp: Parsed S-expression (list starting with 'recipe' symbol)
        initial_bindings: Optional dict of name -> value bindings to inject before compilation.
                         These can be referenced as variables in the recipe.

    Returns:
        CompiledRecipe with nodes and registry

    Example:
        >>> sexp = parse('(recipe "test" :version "1.0" (-> (source cat) (effect identity)))')
        >>> result = compile_recipe(sexp)
        >>> # With parameters:
        >>> result = compile_recipe(sexp, {"effect_num": 5})
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
    encoding = {}
    params = []
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
            elif item.name == "encoding":
                encoding = _parse_encoding(value)
            elif item.name == "params":
                params = _parse_params(value)
            else:
                raise CompileError(f"Unknown keyword :{item.name}")
            i += 2
        else:
            # Body expression
            body_exprs.append(item)
            i += 1

    # Create bindings from params with their default values
    # Initial bindings override param defaults
    for param in params:
        if initial_bindings and param.name in initial_bindings:
            ctx.bindings[param.name] = initial_bindings[param.name]
        else:
            ctx.bindings[param.name] = param.default

    # Inject any additional initial bindings not covered by params
    if initial_bindings:
        for k, v in initial_bindings.items():
            if k not in ctx.bindings:
                ctx.bindings[k] = v

    # Compile body expressions
    # Track when we encounter the first stage to capture pre-stage bindings
    output_node_id = None
    first_stage_seen = False

    for expr in body_exprs:
        # Check if this is a stage form
        is_stage_form = (
            isinstance(expr, list) and
            len(expr) > 0 and
            isinstance(expr[0], Symbol) and
            expr[0].name == "stage"
        )

        # Before the first stage, capture bindings as pre-stage bindings
        if is_stage_form and not first_stage_seen:
            first_stage_seen = True
            ctx.pre_stage_bindings = dict(ctx.bindings)

        result = _compile_expr(expr, ctx)
        if result is not None:
            output_node_id = result

    if output_node_id is None:
        raise CompileError("Recipe has no output (no DAG expression)")

    # Build stage order (topological sort)
    stage_order = _topological_sort_stages(ctx.defined_stages)

    # Collect stages in order
    stages = [ctx.defined_stages[name] for name in stage_order]

    return CompiledRecipe(
        name=name,
        version=version,
        description=description,
        owner=owner,
        registry=ctx.registry,
        nodes=list(ctx.nodes.values()),
        output_node_id=output_node_id,
        encoding=encoding,
        params=params,
        stages=stages,
        stage_order=stage_order,
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
    if name == "analyzer":
        return _compile_analyzer_decl(expr, ctx)
    if name == "construct":
        return _compile_construct_decl(expr, ctx)

    # Include - load and evaluate external sexp file
    if name == "include":
        return _compile_include(expr, ctx)

    # Binding
    if name == "def":
        return _compile_def(expr, ctx)

    # Stage form
    if name == "stage":
        return _compile_stage(expr, ctx)

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

    # Check if it's a registered construct call BEFORE built-in slice-on
    # This allows user-defined constructs to override built-ins
    if name in ctx.registry.get("constructs", {}):
        return _compile_construct_call(expr, ctx)

    if name == "slice-on":
        return _compile_slice_on(expr, ctx)

    # Binding expression for parameter linking
    if name == "bind":
        return _compile_bind(expr, ctx)

    # Pure functions that can be evaluated at compile time
    PURE_FUNCTIONS = {
        "max", "min", "floor", "ceil", "round", "abs",
        "+", "-", "*", "/", "mod", "sqrt", "pow",
        "len", "get", "first", "last", "nth",
        "=", "<", ">", "<=", ">=", "not=",
        "and", "or", "not",
        "inc", "dec",
        "chunk-every",
        "list", "dict",
        "assert",
    }
    if name in PURE_FUNCTIONS:
        # Evaluate using the evaluator
        from .evaluator import evaluate
        # Build env from ctx.bindings
        env = dict(ctx.bindings)
        try:
            result = evaluate(expr, env)
            return result
        except Exception as e:
            raise CompileError(f"Error evaluating {name}: {e}")

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
    """Compile (asset name :cid "..." :url "...") or legacy (asset name :hash "...")"""
    if len(expr) < 2:
        raise CompileError("asset requires a name")

    name = expr[1]
    if isinstance(name, Symbol):
        name = name.name

    _, kwargs = _parse_kwargs(expr, 2)

    # Support both :cid (new IPFS) and :hash (legacy SHA3-256)
    asset_cid = kwargs.get("cid") or kwargs.get("hash")
    if not asset_cid:
        raise CompileError(f"asset {name} requires :cid or :hash")

    ctx.registry["assets"][name] = {
        "cid": asset_cid,
        "url": kwargs.get("url"),
    }
    return None


def _compile_effect_decl(expr: List, ctx: CompilerContext) -> Optional[str]:
    """
    Compile effect - either declaration or node.

    Declaration: (effect name :cid "..." :url "...") or legacy (effect name :hash "...")
    Node: (effect effect-name) or (effect effect-name input-node)
    """
    if len(expr) < 2:
        raise CompileError("effect requires at least a name")

    # Check if this is a declaration (has :cid or :hash)
    _, kwargs = _parse_kwargs(expr, 2)

    # Support both :cid (new) and :hash (legacy)
    effect_cid = kwargs.get("cid") or kwargs.get("hash")

    if effect_cid or "path" in kwargs:
        # Declaration
        name = expr[1]
        if isinstance(name, Symbol):
            name = name.name

        # Handle temporal flag - could be Symbol('true') or Python bool
        temporal = kwargs.get("temporal", False)
        if isinstance(temporal, Symbol):
            temporal = temporal.name.lower() == "true"

        ctx.registry["effects"][name] = {
            "cid": effect_cid,
            "path": kwargs.get("path"),  # Local path for development
            "url": kwargs.get("url"),
            "temporal": temporal,
        }
        return None

    # Otherwise it's a node - delegate to effect node compiler
    return _compile_effect_node(expr, ctx)


def _compile_analyzer_decl(expr: List, ctx: CompilerContext) -> Optional[str]:
    """
    Compile analyzer declaration.

    Declaration: (analyzer name :path "..." :cid "...")

    Example:
        (analyzer beats :path "../analyzers/beats/analyzer.py")
    """
    if len(expr) < 2:
        raise CompileError("analyzer requires at least a name")

    _, kwargs = _parse_kwargs(expr, 2)

    name = expr[1]
    if isinstance(name, Symbol):
        name = name.name

    ctx.registry["analyzers"][name] = {
        "cid": kwargs.get("cid"),
        "path": kwargs.get("path"),
        "url": kwargs.get("url"),
    }
    return None


def _compile_construct_decl(expr: List, ctx: CompilerContext) -> Optional[str]:
    """
    Compile construct declaration.

    Declaration: (construct name :path "...")

    Example:
        (construct beat-alternate :path "constructs/beat-alternate.sexp")
    """
    if len(expr) < 2:
        raise CompileError("construct requires at least a name")

    _, kwargs = _parse_kwargs(expr, 2)

    name = expr[1]
    if isinstance(name, Symbol):
        name = name.name

    ctx.registry["constructs"][name] = {
        "path": kwargs.get("path"),
        "cid": kwargs.get("cid"),
        "url": kwargs.get("url"),
    }
    return None


def _compile_construct_call(expr: List, ctx: CompilerContext) -> str:
    """
    Compile a call to a user-defined construct.

    Creates a CONSTRUCT node that will be expanded at plan time.

    Example:
        (beat-alternate beats-data (list video-a video-b))
    """
    name = expr[0].name
    construct_info = ctx.registry["constructs"][name]

    # Get positional args and kwargs
    args, kwargs = _parse_kwargs(expr, 1)

    # Resolve input references
    resolved_args = []
    node_inputs = []  # Track actual node IDs for inputs

    for arg in args:
        if isinstance(arg, Symbol) and arg.name in ctx.bindings:
            node_id = ctx.bindings[arg.name]
            resolved_args.append(node_id)
            node_inputs.append(node_id)
        elif isinstance(arg, list) and arg and isinstance(arg[0], Symbol):
            # Check if it's a literal list expression like (list video-a video-b)
            if arg[0].name == "list":
                # Resolve each element of the list
                list_items = []
                for item in arg[1:]:
                    if isinstance(item, Symbol) and item.name in ctx.bindings:
                        list_items.append(ctx.bindings[item.name])
                        node_inputs.append(ctx.bindings[item.name])
                    else:
                        list_items.append(item)
                resolved_args.append(list_items)
            else:
                # Try to compile as an expression
                try:
                    node_id = _compile_expr(arg, ctx)
                    if node_id:
                        resolved_args.append(node_id)
                        node_inputs.append(node_id)
                    else:
                        resolved_args.append(arg)
                except CompileError:
                    resolved_args.append(arg)
        else:
            resolved_args.append(arg)

    # Also scan kwargs for Symbol references to nodes (like analysis nodes)
    # Helper to extract node IDs from a value (handles nested lists/dicts)
    def extract_node_ids(val):
        if isinstance(val, str) and len(val) == 64:
            return [val]
        elif isinstance(val, list):
            ids = []
            for item in val:
                ids.extend(extract_node_ids(item))
            return ids
        elif isinstance(val, dict):
            ids = []
            for v in val.values():
                ids.extend(extract_node_ids(v))
            return ids
        return []

    for key, value in kwargs.items():
        if isinstance(value, Symbol) and value.name in ctx.bindings:
            binding_value = ctx.bindings[value.name]
            # If it's a node ID (string hash), add to inputs
            if isinstance(binding_value, str) and len(binding_value) == 64:
                node_inputs.append(binding_value)
            # Also scan lists/dicts for node IDs (e.g., video_infos list)
            elif isinstance(binding_value, (list, dict)):
                node_inputs.extend(extract_node_ids(binding_value))

    node_id = ctx.add_node(
        "CONSTRUCT",
        {
            "construct_name": name,
            "construct_path": construct_info.get("path"),
            "args": resolved_args,
            # Include bindings so reducer lambda can reference video sources etc.
            "bindings": dict(ctx.bindings),
            **kwargs,
        },
        inputs=node_inputs,
    )
    return node_id


def _compile_include(expr: List, ctx: CompilerContext) -> None:
    """
    Compile (include :path "...") or (include name :path "...").

    Loads an external .sexp file and processes its declarations/definitions.
    Supports analyzer, effect, construct declarations and def bindings.

    Forms:
        (include :path "libs/standard-effects.sexp")     ; declaration-only
        (include :cid "bafy...")                          ; from L1/L2 cache
        (include preset-name :path "presets/all.sexp")   ; binds result to name

    Included files can contain:
        - (analyzer name :path "...") declarations
        - (effect name :path "...") declarations
        - (construct name :path "...") declarations
        - (def name value) bindings

    For web-based systems:
        - :cid loads from L1 local cache or L2 shared cache
        - :path is for local development

    Example library file (libs/standard-analyzers.sexp):
        ;; Standard audio analyzers
        (analyzer beats :path "../artdag-analyzers/beats/analyzer.py")
        (analyzer bass :path "../artdag-analyzers/bass/analyzer.py")
        (analyzer energy :path "../artdag-analyzers/energy/analyzer.py")

    Example usage:
        (include :path "libs/standard-analyzers.sexp")
        (include :path "libs/all-effects.sexp")
        ;; Now beats, bass, energy analyzers and all effects are available
    """
    from pathlib import Path
    from .parser import parse_all
    from .evaluator import evaluate

    _, kwargs = _parse_kwargs(expr, 1)

    # Name is optional - check if first arg is a symbol (name) or keyword
    name = None
    if len(expr) >= 2 and isinstance(expr[1], Symbol) and not str(expr[1].name).startswith(":"):
        name = expr[1].name
        _, kwargs = _parse_kwargs(expr, 2)

    path = kwargs.get("path")
    cid = kwargs.get("cid")

    if not path and not cid:
        raise CompileError("include requires :path or :cid")

    content = None

    if cid:
        # Load from content-addressed cache (L1 local / L2 shared)
        content = _load_from_cache(cid, ctx)

    if content is None and path:
        # Load from local path
        include_path = Path(path)

        # Try relative to recipe directory first
        if hasattr(ctx, 'recipe_dir') and ctx.recipe_dir:
            recipe_relative = ctx.recipe_dir / path
            if recipe_relative.exists():
                include_path = recipe_relative

        # Try relative to cwd
        if not include_path.exists():
            import os
            cwd = Path(os.getcwd())
            include_path = cwd / path

        if not include_path.exists():
            raise CompileError(f"Include file not found: {path}")

        content = include_path.read_text()

    if content is None:
        raise CompileError(f"Could not load include: path={path}, cid={cid}")

    # Parse the included file
    sexp_list = parse_all(content)
    if not isinstance(sexp_list, list):
        sexp_list = [sexp_list]

    # Build an environment from current bindings
    env = dict(ctx.bindings)

    for sexp in sexp_list:
        if isinstance(sexp, list) and sexp and isinstance(sexp[0], Symbol):
            form = sexp[0].name

            if form == "def":
                # (def name value) - evaluate and add to bindings
                if len(sexp) != 3:
                    raise CompileError(f"Invalid def in include: {sexp}")
                def_name = sexp[1]
                if isinstance(def_name, Symbol):
                    def_name = def_name.name
                def_value = evaluate(sexp[2], env)
                env[def_name] = def_value
                ctx.bindings[def_name] = def_value

            elif form == "analyzer":
                # (analyzer name :path "..." [:cid "..."])
                _compile_analyzer_decl(sexp, ctx)

            elif form == "effect":
                # (effect name :path "..." [:cid "..."])
                _compile_effect_decl(sexp, ctx)

            elif form == "construct":
                # (construct name :path "..." [:cid "..."])
                _compile_construct_decl(sexp, ctx)

            else:
                # Try to evaluate as expression
                result = evaluate(sexp, env)
                # If a name was provided, bind the last result
                if name and result is not None:
                    ctx.bindings[name] = result
        else:
            # Evaluate as expression (e.g., bare list literal)
            result = evaluate(sexp, env)
            if name and result is not None:
                ctx.bindings[name] = result

    return None


def _load_from_cache(cid: str, ctx: CompilerContext) -> Optional[str]:
    """
    Load content from L1 (local) or L2 (shared) cache by CID.

    Cache hierarchy:
        L1: Local file cache (~/.artdag/cache/{cid})
        L2: Shared/network cache (IPFS, HTTP gateway, etc.)

    Returns file content as string, or None if not found.
    """
    from pathlib import Path
    import os

    # L1: Local cache directory
    cache_dir = Path(os.path.expanduser("~/.artdag/cache"))
    l1_path = cache_dir / cid

    if l1_path.exists():
        return l1_path.read_text()

    # L2: Try shared cache sources
    content = _load_from_l2(cid, ctx)

    if content:
        # Store in L1 for future use
        cache_dir.mkdir(parents=True, exist_ok=True)
        l1_path.write_text(content)

    return content


def _load_from_l2(cid: str, ctx: CompilerContext) -> Optional[str]:
    """
    Load content from L2 shared cache.

    Supports:
        - IPFS gateways (if CID starts with 'bafy' or 'Qm')
        - HTTP URLs (if configured in ctx.l2_sources)
        - Custom backends (extensible)

    Returns content as string, or None if not available.
    """
    import urllib.request
    import urllib.error

    # IPFS gateway (public, for development)
    if cid.startswith("bafy") or cid.startswith("Qm"):
        gateways = [
            f"https://ipfs.io/ipfs/{cid}",
            f"https://dweb.link/ipfs/{cid}",
            f"https://cloudflare-ipfs.com/ipfs/{cid}",
        ]
        for gateway_url in gateways:
            try:
                with urllib.request.urlopen(gateway_url, timeout=10) as response:
                    return response.read().decode('utf-8')
            except (urllib.error.URLError, urllib.error.HTTPError):
                continue

    # Custom L2 sources from context (e.g., private cache server)
    l2_sources = getattr(ctx, 'l2_sources', [])
    for source in l2_sources:
        try:
            url = f"{source}/{cid}"
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.read().decode('utf-8')
        except (urllib.error.URLError, urllib.error.HTTPError):
            continue

    return None


def _compile_def(expr: List, ctx: CompilerContext) -> None:
    """Compile (def name expr)"""
    if len(expr) != 3:
        raise CompileError("def requires exactly 2 arguments: name and expression")

    name = expr[1]
    if not isinstance(name, Symbol):
        raise CompileError(f"def name must be a symbol, got {name}")

    # If binding already exists (e.g. from command-line param), don't override
    # This allows recipes to specify defaults that command-line params can override
    if name.name in ctx.bindings:
        return None

    body = expr[2]

    # Check if body is a simple value (number, string, etc.)
    if isinstance(body, (int, float, str, bool)):
        ctx.bindings[name.name] = body
        return None

    node_id = _compile_expr(body, ctx)

    # If result is a simple value (from evaluated pure function), store it directly
    # This includes lists, tuples, dicts from pure functions like `list`
    if isinstance(node_id, (int, float, str, bool, list, tuple, dict)):
        ctx.bindings[name.name] = node_id
        return None

    if node_id is None:
        raise CompileError(f"def body must produce a node or value")

    # Store binding for reference resolution
    ctx.bindings[name.name] = node_id

    # Also store the name on the node so planner can reference it
    if node_id in ctx.nodes:
        ctx.nodes[node_id]["name"] = name.name

    return None


def _compile_stage(expr: List, ctx: CompilerContext) -> Optional[str]:
    """
    Compile (stage :name :requires [...] :inputs [...] :outputs [...] body...).

    Stage form enables explicit dependency declaration, parallel execution,
    and variable scoping.

    Example:
        (stage :analyze-a
          :outputs [beats-a]
          (def beats-a (-> audio-a (analyze beats))))

        (stage :plan-a
          :requires [:analyze-a]
          :inputs [beats-a]
          :outputs [segments-a]
          (def segments-a (make-segments :beats beats-a)))
    """
    if len(expr) < 2:
        raise CompileError("stage requires at least a name")

    # Parse stage name (first element after 'stage' should be a keyword like :analyze-a)
    # The stage name is NOT a key-value pair - it's a standalone keyword
    stage_name = None
    start_idx = 1

    if len(expr) > 1:
        first_arg = expr[1]
        if isinstance(first_arg, Keyword):
            stage_name = first_arg.name
            start_idx = 2
        elif isinstance(first_arg, Symbol):
            stage_name = first_arg.name
            start_idx = 2

    if stage_name is None:
        raise CompileError("stage requires a name (e.g., (stage :analyze-a ...))")

    # Now parse remaining kwargs and body
    args, kwargs = _parse_kwargs(expr, start_idx)

    # Parse requires, inputs, outputs
    requires = []
    if "requires" in kwargs:
        req_val = kwargs["requires"]
        if isinstance(req_val, list):
            for r in req_val:
                if isinstance(r, Keyword):
                    requires.append(r.name)
                elif isinstance(r, Symbol):
                    requires.append(r.name)
                elif isinstance(r, str):
                    requires.append(r)
                else:
                    raise CompileError(f"Invalid require: {r}")
        else:
            raise CompileError(":requires must be a list")

    inputs = []
    if "inputs" in kwargs:
        inp_val = kwargs["inputs"]
        if isinstance(inp_val, list):
            for i in inp_val:
                if isinstance(i, Symbol):
                    inputs.append(i.name)
                elif isinstance(i, str):
                    inputs.append(i)
                else:
                    raise CompileError(f"Invalid input: {i}")
        else:
            raise CompileError(":inputs must be a list")

    outputs = []
    if "outputs" in kwargs:
        out_val = kwargs["outputs"]
        if isinstance(out_val, list):
            for o in out_val:
                if isinstance(o, Symbol):
                    outputs.append(o.name)
                elif isinstance(o, str):
                    outputs.append(o)
                else:
                    raise CompileError(f"Invalid output: {o}")
        else:
            raise CompileError(":outputs must be a list")

    # Validate requires - must reference defined stages
    for req in requires:
        if req not in ctx.defined_stages:
            raise CompileError(
                f"Stage '{stage_name}' requires undefined stage '{req}'"
            )

    # Validate inputs - must be produced by required stages
    for inp in inputs:
        found = False
        for req in requires:
            if inp in ctx.defined_stages[req].output_bindings:
                found = True
                break
        if not found and inp not in ctx.pre_stage_bindings:
            raise CompileError(
                f"Stage '{stage_name}' declares input '{inp}' "
                f"which is not an output of any required stage or pre-stage binding"
            )

    # Check for circular dependencies (simple check for now)
    # A more thorough check would use topological sort
    visited = set()
    def check_cycle(stage: str, path: List[str]):
        if stage in path:
            cycle = " -> ".join(path + [stage])
            raise CompileError(f"Circular stage dependency: {cycle}")
        if stage in visited:
            return
        visited.add(stage)
        if stage in ctx.defined_stages:
            for req in ctx.defined_stages[stage].requires:
                check_cycle(req, path + [stage])

    for req in requires:
        check_cycle(req, [stage_name])

    # Save context state before entering stage
    prev_stage = ctx.current_stage
    prev_stage_node_ids = ctx.stage_node_ids

    # Enter stage context
    ctx.current_stage = stage_name
    ctx.stage_node_ids = []

    # Build accessible bindings for this stage
    stage_ctx_bindings = dict(ctx.pre_stage_bindings)

    # Add input bindings from required stages
    for inp in inputs:
        for req in requires:
            if inp in ctx.defined_stages[req].output_bindings:
                stage_ctx_bindings[inp] = ctx.defined_stages[req].output_bindings[inp]
                break

    # Save current bindings and set up stage bindings
    prev_bindings = ctx.bindings
    ctx.bindings = stage_ctx_bindings

    # Compile body expressions
    # Body expressions are lists or symbols after the stage name and kwargs
    # Start from index 2 (after 'stage' and stage name)
    body_exprs = []
    i = 2  # Skip 'stage' and stage name
    while i < len(expr):
        item = expr[i]
        if isinstance(item, Keyword):
            # Skip keyword and its value
            i += 2
        elif isinstance(item, (list, Symbol)):
            # Include both list expressions and symbol references
            body_exprs.append(item)
            i += 1
        else:
            i += 1

    last_result = None
    for body_expr in body_exprs:
        result = _compile_expr(body_expr, ctx)
        if result is not None:
            last_result = result

    # Collect output bindings
    output_bindings = {}
    for out in outputs:
        if out in ctx.bindings:
            output_bindings[out] = ctx.bindings[out]
        else:
            raise CompileError(
                f"Stage '{stage_name}' declares output '{out}' "
                f"but it was not defined in the stage body"
            )

    # Create CompiledStage
    compiled_stage = CompiledStage(
        name=stage_name,
        requires=requires,
        inputs=inputs,
        outputs=outputs,
        node_ids=ctx.stage_node_ids,
        output_bindings=output_bindings,
    )

    # Register the stage
    ctx.defined_stages[stage_name] = compiled_stage
    ctx.stage_bindings[stage_name] = output_bindings

    # Restore context state
    ctx.current_stage = prev_stage
    ctx.stage_node_ids = prev_stage_node_ids
    ctx.bindings = prev_bindings

    # Make stage outputs available to subsequent stages via bindings
    ctx.bindings.update(output_bindings)

    return last_result


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
    Compile (source asset-name), (source :input "name" ...), or (source :path "file.mkv" ...).
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
    elif "path" in kwargs:
        # Local file path - for development/testing
        # (source :path "dog.mkv" :description "Input video")
        path = kwargs["path"]
        config = {
            "path": path,
            "description": kwargs.get("description", ""),
        }
    elif args:
        # Asset reference
        asset_name = args[0]
        if isinstance(asset_name, Symbol):
            asset_name = asset_name.name
        config = {"asset": asset_name}
    else:
        raise CompileError("source requires asset name, :input flag, or :path")

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
    """
    Compile (resize width height :mode "linear" [input]).

    Resize is now an EFFECT that uses the sexp resize-frame effect.
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    if len(args) < 2:
        raise CompileError("resize requires width and height")

    # Create EFFECT node with resize effect
    # Note: param names match resize.sexp (target-w, target-h to avoid primitive conflict)
    config = {
        "effect": "resize-frame",
        "effect_path": "sexp_effects/effects/resize-frame.sexp",
        "target-w": int(args[0]),
        "target-h": int(args[1]),
        "mode": kwargs.get("mode", "linear"),
    }

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args[2:]:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("EFFECT", config, inputs)


def _compile_sequence(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (sequence node1 node2 ... :resize-mode :fit :priority :width).

    Options:
        :transition    - transition between clips (default: cut)
        :resize-mode   - fit | crop | stretch | cover (default: none)
        :priority      - width | height (which dimension to match exactly)
        :target-width  - explicit target width
        :target-height - explicit target height
        :pad-color     - color for fit mode padding (default: black)
        :crop-gravity  - center | top | bottom | left | right (default: center)
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    config = {
        "transition": kwargs.get("transition", {"type": "cut"}),
    }

    # Add normalize config if specified
    resize_mode = kwargs.get("resize-mode")
    if isinstance(resize_mode, (Symbol, Keyword)):
        resize_mode = resize_mode.name
    if resize_mode:
        config["resize_mode"] = resize_mode

        priority = kwargs.get("priority")
        if isinstance(priority, (Symbol, Keyword)):
            priority = priority.name
        if priority:
            config["priority"] = priority

        if kwargs.get("target-width"):
            config["target_width"] = kwargs["target-width"]
        if kwargs.get("target-height"):
            config["target_height"] = kwargs["target-height"]

        pad_color = kwargs.get("pad-color")
        if isinstance(pad_color, (Symbol, Keyword)):
            pad_color = pad_color.name
        config["pad_color"] = pad_color or "black"

        crop_gravity = kwargs.get("crop-gravity")
        if isinstance(crop_gravity, (Symbol, Keyword)):
            crop_gravity = crop_gravity.name
        config["crop_gravity"] = crop_gravity or "center"

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    return ctx.add_node("SEQUENCE", config, inputs)


def _compile_layer(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (layer bg-node fg-node :x 0 :y 0 :opacity 1.0 :mode "alpha").

    Layer is now a multi-input EFFECT that uses the sexp layer effect.
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    if len(inputs) < 2:
        raise CompileError("layer requires two inputs (background, foreground)")

    # Create EFFECT node with layer effect
    config = {
        "effect": "layer",
        "effect_path": "sexp_effects/effects/layer.sexp",
        "multi_input": True,
        "x": kwargs.get("x", 0),
        "y": kwargs.get("y", 0),
        "opacity": float(kwargs.get("opacity", 1.0)),
        "mode": kwargs.get("mode", "alpha"),
    }

    return ctx.add_node("EFFECT", config, inputs)


def _compile_blend(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (blend node1 node2 :mode "overlay" :opacity 0.5).

    Blend is now a multi-input EFFECT that uses the sexp blend effect.
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args:
        inputs.append(_resolve_input(arg, ctx, prev_id))

    if len(inputs) < 2:
        raise CompileError("blend requires two inputs")

    # Create EFFECT node with blend effect
    config = {
        "effect": "blend",
        "effect_path": "sexp_effects/effects/blend.sexp",
        "multi_input": True,
        "mode": kwargs.get("mode", "overlay"),
        "opacity": float(kwargs.get("opacity", 0.5)),
    }

    return ctx.add_node("EFFECT", config, inputs)


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


def _compile_slice_on(expr: List, ctx: CompilerContext) -> str:
    """
    Compile slice-on with either legacy or lambda syntax.

    Legacy syntax:
        (slice-on video analysis :times path :effect fx :pattern pat)

    Lambda syntax:
        (slice-on analysis
          :times times
          :init 0
          :fn (lambda [acc i start end]
                {:source video
                 :effects (if (odd? i) [invert] [])
                 :acc (inc acc)}))

    Args:
        video: input video node (legacy) or omitted (lambda)
        analysis: analysis node with times array
        :times - path to times array in analysis
        :effect - effect to apply (legacy, optional)
        :pattern - all, odd, even, alternate (legacy, default: all)
        :init - initial accumulator value (lambda)
        :fn - reducer lambda function (lambda)
    """
    from .parser import Lambda

    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    # Check for lambda mode
    reducer_fn = kwargs.get("fn")

    # Parse lambda if it's a list
    if isinstance(reducer_fn, list):
        reducer_fn = _parse_lambda(reducer_fn)

    # Lambda mode: only analysis input required (sources come from fn)
    # Legacy mode: requires video and analysis inputs
    if reducer_fn is not None:
        # Lambda mode - just need analysis input
        if len(args) < 1:
            raise CompileError("slice-on requires analysis input")
        analysis_input = _resolve_input(args[0], ctx, prev_id)
        inputs = [analysis_input]
    else:
        # Legacy mode - need video and analysis inputs
        if len(args) < 2:
            raise CompileError("slice-on requires video and analysis inputs")
        video_input = _resolve_input(args[0], ctx, prev_id)
        analysis_input = _resolve_input(args[1], ctx, prev_id)
        inputs = [video_input, analysis_input]

    times_path = kwargs.get("times", "times")
    if isinstance(times_path, Symbol):
        times_path = times_path.name

    config = {
        "times_path": times_path,
        "fn": reducer_fn,
        "init": kwargs.get("init", 0),
        # Include bindings so lambda can reference video sources etc.
        "bindings": dict(ctx.bindings),
    }

    return ctx.add_node("SLICE_ON", config, inputs)


def _parse_lambda(expr: List):
    """Parse a lambda expression list into a Lambda object."""
    from .parser import Lambda, Symbol

    if not expr or not isinstance(expr[0], Symbol):
        raise CompileError("Invalid lambda expression")

    name = expr[0].name
    if name not in ("lambda", "fn"):
        raise CompileError(f"Expected lambda or fn, got {name}")

    if len(expr) < 3:
        raise CompileError("lambda requires params and body")

    params = expr[1]
    if not isinstance(params, list):
        raise CompileError("lambda params must be a list")

    param_names = []
    for p in params:
        if isinstance(p, Symbol):
            param_names.append(p.name)
        elif isinstance(p, str):
            param_names.append(p)
        else:
            raise CompileError(f"Invalid lambda param: {p}")

    return Lambda(param_names, expr[2])


def _compile_analyze(expr: List, ctx: CompilerContext) -> str:
    """
    Compile (analyze analyzer-name :param value ...).

    Example:
        (analyze beats)
        (analyze beats :min-bpm 120 :max-bpm 180)
    """
    args, kwargs = _parse_kwargs(expr, 1)
    args, kwargs, prev_id = _extract_prev_id(args, kwargs)

    # First arg is analyzer name
    if not args:
        raise CompileError("analyze requires analyzer name")

    analyzer_name = args[0]
    if isinstance(analyzer_name, Symbol):
        analyzer_name = analyzer_name.name

    # Look up analyzer in registry
    analyzer_entry = ctx.registry.get("analyzers", {}).get(analyzer_name, {})

    config = {
        "analyzer": analyzer_name,
        "analyzer_path": analyzer_entry.get("path"),
        "cid": analyzer_entry.get("cid"),
    }
    # Add params (kwargs) to config
    config.update(kwargs)

    inputs = []
    if prev_id:
        inputs.append(prev_id if isinstance(prev_id, str) else str(prev_id))
    for arg in args[1:]:  # Skip analyzer name
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
    if isinstance(value, Symbol):
        # Resolve symbol from bindings
        if value.name in ctx.bindings:
            return ctx.bindings[value.name]
        # Return as-is if not found (could be an effect reference, etc.)
        return value
    if isinstance(value, list) and len(value) > 0:
        head = value[0]
        if isinstance(head, Symbol) and head.name == "bind":
            return _compile_bind(value, ctx)
        # Handle lambda expressions - parse but don't compile
        if isinstance(head, Symbol) and head.name in ("lambda", "fn"):
            return _parse_lambda(value)
        # Could be other nested expressions
        return _compile_expr(value, ctx)
    return value


def compile_string(text: str, initial_bindings: Dict[str, Any] = None) -> CompiledRecipe:
    """
    Compile an S-expression recipe string.

    Convenience function combining parse + compile.

    Args:
        text: S-expression recipe string
        initial_bindings: Optional dict of name -> value bindings to inject before compilation.
                         These can be referenced as variables in the recipe.
    """
    sexp = parse(text)
    return compile_recipe(sexp, initial_bindings)
