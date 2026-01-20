"""
Execution plan generation from S-expression recipes.

The planner:
1. Takes a compiled recipe + input content hashes
2. Runs analyzers to get concrete data (beat times, etc.)
3. Expands dynamic nodes (SLICE_ON) into primitive operations
4. Resolves all registry references to content hashes
5. Generates an execution plan with pre-computed cache IDs

Plans are S-expressions with all references resolved to hashes,
ready for distribution to Celery workers.
"""

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .parser import Symbol, Keyword, Binding, serialize
from .compiler import CompiledRecipe


# Node types that can be collapsed into a single FFmpeg filter chain
COLLAPSIBLE_TYPES = {"EFFECT", "SEGMENT"}

# Node types that are boundaries (sources, merges, or special processing)
BOUNDARY_TYPES = {"SOURCE", "SEQUENCE", "MUX", "ANALYZE", "LIST"}

# Node types that need expansion during planning
EXPANDABLE_TYPES = {"SLICE_ON", "CONSTRUCT"}


def _load_module(module_path: Path, module_name: str = "module"):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_analyzer(
    analyzer_path: Path,
    input_path: Path,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Run an analyzer module and return results."""
    analyzer = _load_module(analyzer_path, "analyzer")
    return analyzer.analyze(input_path, params)


def _pre_execute_segment(
    node: Dict,
    input_path: Path,
    work_dir: Path,
) -> Path:
    """
    Pre-execute a SEGMENT node during planning.

    This is needed when ANALYZE depends on a SEGMENT output.
    Returns path to the segmented file.
    """
    import subprocess
    import tempfile

    config = node.get("config", {})
    start = config.get("start", 0)
    duration = config.get("duration")
    end = config.get("end")

    # Detect if input is audio-only
    suffix = input_path.suffix.lower()
    is_audio = suffix in ('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')

    if is_audio:
        output_ext = ".m4a"  # Use m4a for aac codec
    else:
        output_ext = ".mp4"

    output_path = work_dir / f"segment_{node['id'][:16]}{output_ext}"

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if start:
        cmd.extend(["-ss", str(start)])
    if duration:
        cmd.extend(["-t", str(duration)])
    elif end:
        cmd.extend(["-t", str(end - start)])

    if is_audio:
        cmd.extend(["-c:a", "aac", str(output_path)])
    else:
        cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18",
                   "-c:a", "aac", str(output_path)])

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _serialize_for_hash(obj) -> str:
    """Serialize any value to canonical S-expression string for hashing."""
    from .parser import Lambda

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
    if isinstance(obj, Binding):
        # analysis_ref can be a string, node ID, or dict - serialize it properly
        if isinstance(obj.analysis_ref, str):
            ref_str = f'"{obj.analysis_ref}"'
        else:
            ref_str = _serialize_for_hash(obj.analysis_ref)
        return f"(bind {ref_str} :range [{obj.range_min} {obj.range_max}])"
    if isinstance(obj, dict):
        items = []
        for k, v in sorted(obj.items()):
            items.append(f":{k} {_serialize_for_hash(v)}")
        return "{" + " ".join(items) + "}"
    if isinstance(obj, list):
        items = [_serialize_for_hash(x) for x in obj]
        return "(" + " ".join(items) + ")"
    return str(obj)


def _stable_hash(data: Any, cluster_key: str = None) -> str:
    """Create stable SHA3-256 hash from data using S-expression serialization."""
    if cluster_key:
        data = {"_cluster_key": cluster_key, "_data": data}
    sexp_str = _serialize_for_hash(data)
    return hashlib.sha3_256(sexp_str.encode()).hexdigest()


@dataclass
class PlanStep:
    """A step in the execution plan."""
    step_id: str
    node_type: str
    config: Dict[str, Any]
    inputs: List[str]  # List of input step_ids
    cache_id: str
    level: int = 0
    stage: Optional[str] = None  # Stage this step belongs to
    stage_cache_id: Optional[str] = None  # Cache ID for the stage

    def to_sexp(self) -> List:
        """Convert to S-expression."""
        sexp = [Symbol("step"), self.step_id]

        # Add cache-id
        sexp.extend([Keyword("cache-id"), self.cache_id])

        # Add level if > 0
        if self.level > 0:
            sexp.extend([Keyword("level"), self.level])

        # Add stage info if present
        if self.stage:
            sexp.extend([Keyword("stage"), self.stage])
        if self.stage_cache_id:
            sexp.extend([Keyword("stage-cache-id"), self.stage_cache_id])

        # Add the node expression
        node_sexp = [Symbol(self.node_type.lower())]

        # Add config as keywords
        for key, value in self.config.items():
            # Convert Binding to sexp form
            if isinstance(value, Binding):
                value = [Symbol("bind"), value.analysis_ref,
                        Keyword("range"), [value.range_min, value.range_max]]
            node_sexp.extend([Keyword(key), value])

        # Add inputs if any
        if self.inputs:
            node_sexp.extend([Keyword("inputs"), self.inputs])

        sexp.append(node_sexp)
        return sexp


@dataclass
class StagePlan:
    """A stage in the execution plan with its own cache ID."""
    stage_name: str
    cache_id: str
    steps: List[PlanStep]
    requires: List[str]  # Names of required stages
    output_bindings: Dict[str, str]  # binding_name -> cache_id of output
    level: int = 0  # Stage level for parallel execution


@dataclass
class ExecutionPlanSexp:
    """Execution plan as S-expression."""
    plan_id: str
    recipe_id: str
    recipe_hash: str
    steps: List[PlanStep]
    output_step_id: str
    inputs: Dict[str, str] = field(default_factory=dict)  # name -> hash
    analysis: Dict[str, Dict] = field(default_factory=dict)  # name -> {times, values}
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_plans: List[StagePlan] = field(default_factory=list)  # Stage-level plans
    stage_order: List[str] = field(default_factory=list)  # Topologically sorted stage names
    stage_levels: Dict[str, int] = field(default_factory=dict)  # stage_name -> level
    stage_cache_ids: Dict[str, str] = field(default_factory=dict)  # stage_name -> cache_id
    effects_registry: Dict[str, Dict] = field(default_factory=dict)  # effect_name -> {path, cid, ...}
    minimal_primitives: bool = False  # If True, interpreter uses only core primitives

    def to_sexp(self) -> List:
        """Convert entire plan to S-expression."""
        sexp = [Symbol("plan")]

        # Metadata
        sexp.extend([Keyword("id"), self.plan_id])
        sexp.extend([Keyword("recipe"), self.recipe_id])
        sexp.extend([Keyword("recipe-hash"), self.recipe_hash])

        # Input bindings
        if self.inputs:
            inputs_sexp = [Symbol("inputs")]
            for name, hash_val in self.inputs.items():
                inputs_sexp.append([Symbol(name), hash_val])
            sexp.append(inputs_sexp)

        # Analysis data (for effect parameter bindings)
        if self.analysis:
            analysis_sexp = [Symbol("analysis")]
            for name, data in self.analysis.items():
                track_sexp = [Symbol(name)]
                if "times" in data:
                    track_sexp.extend([Keyword("times"), data["times"]])
                if "values" in data:
                    track_sexp.extend([Keyword("values"), data["values"]])
                analysis_sexp.append(track_sexp)
            sexp.append(analysis_sexp)

        # Stage information
        if self.stage_plans:
            stages_sexp = [Symbol("stages")]
            for stage_plan in self.stage_plans:
                stage_sexp = [
                    Keyword("name"), stage_plan.stage_name,
                    Keyword("cache-id"), stage_plan.cache_id,
                    Keyword("level"), stage_plan.level,
                ]
                if stage_plan.requires:
                    stage_sexp.extend([Keyword("requires"), stage_plan.requires])
                if stage_plan.output_bindings:
                    outputs_sexp = []
                    for name, cache_id in stage_plan.output_bindings.items():
                        outputs_sexp.append([Symbol(name), Keyword("cache-id"), cache_id])
                    stage_sexp.extend([Keyword("outputs"), outputs_sexp])
                stages_sexp.append(stage_sexp)
            sexp.append(stages_sexp)

        # Effects registry - for loading explicitly declared effects
        if self.effects_registry:
            registry_sexp = [Symbol("effects-registry")]
            for name, info in self.effects_registry.items():
                effect_sexp = [Symbol(name)]
                if info.get("path"):
                    effect_sexp.extend([Keyword("path"), info["path"]])
                if info.get("cid"):
                    effect_sexp.extend([Keyword("cid"), info["cid"]])
                registry_sexp.append(effect_sexp)
            sexp.append(registry_sexp)

        # Minimal primitives flag
        if self.minimal_primitives:
            sexp.extend([Keyword("minimal-primitives"), True])

        # Steps
        for step in self.steps:
            sexp.append(step.to_sexp())

        # Output reference
        sexp.extend([Keyword("output"), self.output_step_id])

        return sexp

    def to_string(self, pretty: bool = True) -> str:
        """Serialize plan to S-expression string."""
        return serialize(self.to_sexp(), pretty=pretty)


def _expand_list_inputs(nodes: List[Dict]) -> List[Dict]:
    """
    Expand LIST node inputs in SEQUENCE nodes.

    When a SEQUENCE has a LIST as input, replace it with all the LIST's inputs.
    LIST nodes are virtual and removed from the final node list.
    """
    nodes_by_id = {n["id"]: n for n in nodes}
    list_nodes = {n["id"]: n for n in nodes if n["type"] == "LIST"}

    if not list_nodes:
        return nodes

    result = []
    for node in nodes:
        if node["type"] == "LIST":
            # Skip LIST nodes - they're virtual containers
            continue

        if node["type"] == "SEQUENCE":
            # Expand any LIST inputs
            new_inputs = []
            for inp in node.get("inputs", []):
                if inp in list_nodes:
                    # Replace LIST with its contents
                    new_inputs.extend(list_nodes[inp].get("inputs", []))
                else:
                    new_inputs.append(inp)

            # Create updated node
            result.append({
                **node,
                "inputs": new_inputs,
            })
        else:
            result.append(node)

    return result


def _collapse_effect_chains(nodes: List[Dict], registry: Dict = None) -> List[Dict]:
    """
    Collapse sequential effect chains into single COMPOUND nodes.

    A chain is a sequence of single-input collapsible nodes where:
    - Each node has exactly one input
    - No node in the chain is referenced by multiple other nodes
    - The chain ends at a boundary or multi-ref node
    - No node in the chain is marked as temporal

    Effects can declare :temporal true to prevent collapsing (e.g., reverse).

    Returns a new node list with chains collapsed.
    """
    if not nodes:
        return nodes

    registry = registry or {}
    nodes_by_id = {n["id"]: n for n in nodes}

    # Build reference counts: how many nodes reference each node as input
    ref_count = {n["id"]: 0 for n in nodes}
    for node in nodes:
        for inp in node.get("inputs", []):
            if inp in ref_count:
                ref_count[inp] += 1

    # Track which nodes are consumed by chains
    consumed = set()
    compound_nodes = []

    def is_temporal(node: Dict) -> bool:
        """Check if a node is temporal (needs complete input)."""
        config = node.get("config", {})
        # Check node-level temporal flag
        if config.get("temporal"):
            return True
        # Check effect registry for temporal flag
        if node["type"] == "EFFECT":
            effect_name = config.get("effect")
            if effect_name:
                effect_meta = registry.get("effects", {}).get(effect_name, {})
                if effect_meta.get("temporal"):
                    return True
        return False

    def is_collapsible(node_id: str) -> bool:
        """Check if a node can be part of a chain."""
        if node_id in consumed:
            return False
        node = nodes_by_id.get(node_id)
        if not node:
            return False
        if node["type"] not in COLLAPSIBLE_TYPES:
            return False
        # Temporal effects can't be collapsed
        if is_temporal(node):
            return False
        # Effects CAN be collapsed if they have an FFmpeg mapping
        # Only fall back to Python interpreter if no mapping exists
        config = node.get("config", {})
        if node["type"] == "EFFECT":
            effect_name = config.get("effect")
            # Import here to avoid circular imports
            from .ffmpeg_compiler import FFmpegCompiler
            compiler = FFmpegCompiler()
            if compiler.get_mapping(effect_name):
                return True  # Has FFmpeg mapping, can collapse
            elif config.get("effect_path"):
                return False  # No FFmpeg mapping, has Python path, can't collapse
        return True

    def is_chain_boundary(node_id: str) -> bool:
        """Check if a node is a chain boundary (can't be collapsed into)."""
        node = nodes_by_id.get(node_id)
        if not node:
            return True  # Unknown node is a boundary
        # Boundary if: it's a boundary type, or referenced by multiple nodes
        return node["type"] in BOUNDARY_TYPES or ref_count.get(node_id, 0) > 1

    def collect_chain(start_id: str) -> List[str]:
        """Collect a chain of collapsible nodes starting from start_id."""
        chain = [start_id]
        current = start_id

        while True:
            node = nodes_by_id[current]
            inputs = node.get("inputs", [])

            # Must have exactly one input
            if len(inputs) != 1:
                break

            next_id = inputs[0]

            # Stop if next is a boundary or already consumed
            if is_chain_boundary(next_id) or not is_collapsible(next_id):
                break

            # Stop if next is referenced by others besides current
            if ref_count.get(next_id, 0) > 1:
                break

            chain.append(next_id)
            current = next_id

        return chain

    # Process nodes in reverse order (from outputs toward inputs)
    # This ensures we find complete chains starting from their end
    # First, topologically sort to get dependency order
    sorted_ids = []
    visited = set()

    def topo_visit(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)
        node = nodes_by_id.get(node_id)
        if node:
            for inp in node.get("inputs", []):
                topo_visit(inp)
            sorted_ids.append(node_id)

    for node in nodes:
        topo_visit(node["id"])

    # Process in reverse topological order (outputs first)
    result_nodes = []

    for node_id in reversed(sorted_ids):
        node = nodes_by_id[node_id]

        if node_id in consumed:
            continue

        if not is_collapsible(node_id):
            # Keep boundary nodes as-is
            result_nodes.append(node)
            continue

        # Check if this node is the start of a chain (output end)
        # A node is a chain start if it's collapsible and either:
        # - Referenced by a boundary node
        # - Referenced by multiple nodes
        # - Is the output node
        # For now, collect chain going backwards from this node

        chain = collect_chain(node_id)

        if len(chain) == 1:
            # Single node, no collapse needed
            result_nodes.append(node)
            continue

        # Collapse the chain into a COMPOUND node
        # Chain is [end, ..., start] order (backwards from output)
        # The compound node:
        # - Has the same ID as the chain end (for reference stability)
        # - Takes input from what the chain start originally took
        # - Has a filter_chain config with all the nodes in order

        chain_start = chain[-1]  # First to execute
        chain_end = chain[0]     # Last to execute

        start_node = nodes_by_id[chain_start]
        end_node = nodes_by_id[chain_end]

        # Build filter chain config (in execution order: start to end)
        filter_chain = []
        for chain_node_id in reversed(chain):
            chain_node = nodes_by_id[chain_node_id]
            filter_chain.append({
                "type": chain_node["type"],
                "config": chain_node.get("config", {}),
            })

        compound_node = {
            "id": chain_end,  # Keep the end ID for reference stability
            "type": "COMPOUND",
            "config": {
                "filter_chain": filter_chain,
                # Include effects registry so executor can load only declared effects
                "effects_registry": registry.get("effects", {}),
            },
            "inputs": start_node.get("inputs", []),
            "name": f"compound_{len(filter_chain)}_effects",
        }

        result_nodes.append(compound_node)

        # Mark all chain nodes as consumed
        for chain_node_id in chain:
            consumed.add(chain_node_id)

    return result_nodes


def _expand_slice_on(
    node: Dict,
    analysis_data: Dict[str, Any],
    registry: Dict,
    sources: Dict[str, str] = None,
    cluster_key: str = None,
    encoding: Dict = None,
) -> List[Dict]:
    """
    Expand a SLICE_ON node into primitive SEGMENT + EFFECT + SEQUENCE nodes.

    Supports two modes:
    1. Legacy: :effect and :pattern parameters
    2. Lambda: :fn parameter with reducer function

    Lambda syntax:
        (slice-on analysis
          :times times
          :init 0
          :fn (lambda [acc i start end]
                {:source video
                 :effects (if (odd? i) [invert] [])
                 :acc (inc acc)}))

    Args:
        node: The SLICE_ON node to expand
        analysis_data: Analysis results containing times array
        registry: Recipe registry with effect definitions
        sources: Map of source names to node IDs
        cluster_key: Optional cluster key for hashing

    Returns:
        List of expanded nodes (segments, effects, sequence)
    """
    from .evaluator import evaluate, EvalError
    from .parser import Lambda, Symbol

    config = node.get("config", {})
    node_inputs = node.get("inputs", [])
    sources = sources or {}

    # Extract times
    times_path = config.get("times_path", "times")
    times = analysis_data
    for key in times_path.split("."):
        times = times[key]

    if not times:
        raise ValueError(f"No times found at path '{times_path}' in analysis")

    # Default video input (first input after analysis)
    default_video = node_inputs[0] if node_inputs else None

    expanded_nodes = []
    sequence_inputs = []
    base_id = node["id"][:8]

    # Check for lambda-based reducer
    reducer_fn = config.get("fn")

    if isinstance(reducer_fn, Lambda):
        # Lambda mode - evaluate function for each slice
        acc = config.get("init", 0)
        slice_times = list(zip([0] + times[:-1], times))

        # Frame-accurate timing calculation
        # Align ALL times to frame boundaries to prevent accumulating drift
        fps = (encoding or {}).get("fps", 30)
        frame_duration = 1.0 / fps

        # Get total duration from analysis data (beats analyzer includes this)
        # Falls back to config target_duration for backwards compatibility
        total_duration = analysis_data.get("duration") or config.get("target_duration")

        # Pre-compute frame-aligned cumulative times
        cumulative_frames = [0]  # Start at frame 0
        for t in times:
            # Round to nearest frame boundary
            frames = round(t * fps)
            cumulative_frames.append(frames)

        # If total duration known, ensure last segment extends to it exactly
        if total_duration is not None:
            target_frames = round(total_duration * fps)
            if target_frames > cumulative_frames[-1]:
                cumulative_frames[-1] = target_frames

        # Pre-compute frame-aligned start times and durations for each slice
        frame_aligned_starts = []
        frame_aligned_durations = []
        for i in range(len(cumulative_frames) - 1):
            start_frames = cumulative_frames[i]
            end_frames = cumulative_frames[i + 1]
            frame_aligned_starts.append(start_frames * frame_duration)
            frame_aligned_durations.append((end_frames - start_frames) * frame_duration)

        for i, (start, end) in enumerate(slice_times):
            if start >= end:
                continue

            # Build environment with sources, effects, and builtins
            env = dict(sources)

            # Add effect names so they can be referenced as symbols
            for effect_name in registry.get("effects", {}):
                env[effect_name] = effect_name

            env["acc"] = acc
            env["i"] = i
            env["start"] = start
            env["end"] = end

            # Evaluate the reducer
            result = evaluate([reducer_fn, Symbol("acc"), Symbol("i"),
                              Symbol("start"), Symbol("end")], env)

            if not isinstance(result, dict):
                raise ValueError(f"Reducer must return a dict, got {type(result)}")

            # Extract result
            source_name = result.get("source")
            effects = result.get("effects", [])
            acc = result.get("acc", acc)

            # Segment timing: use frame-aligned values to prevent drift
            # Lambda can override with explicit start/duration/end
            if result.get("start") is not None or result.get("duration") is not None or result.get("end") is not None:
                # Explicit timing from lambda - use as-is
                seg_start = result.get("start", start)
                seg_duration = result.get("duration")
                if seg_duration is None:
                    if result.get("end") is not None:
                        seg_duration = result["end"] - seg_start
                    else:
                        seg_duration = end - start
            else:
                # Default: use frame-aligned start and duration to prevent accumulated drift
                seg_start = frame_aligned_starts[i] if i < len(frame_aligned_starts) else start
                seg_duration = frame_aligned_durations[i] if i < len(frame_aligned_durations) else (end - start)

            # Resolve source to node ID
            if isinstance(source_name, Symbol):
                source_name = source_name.name
            # source_name could be a name (lookup in sources) or already a node ID
            # Build set of valid node IDs from sources values
            valid_node_ids = set(sources.values())
            if source_name in sources:
                video_input = sources[source_name]
            elif source_name in valid_node_ids:
                # Already a node ID
                video_input = source_name
            else:
                video_input = default_video

            # Create SEGMENT node
            segment_id = f"{base_id}_seg_{i:04d}"
            segment_node = {
                "id": segment_id,
                "type": "SEGMENT",
                "config": {
                    "start": seg_start,
                    "duration": seg_duration,
                },
                "inputs": [video_input],
            }
            expanded_nodes.append(segment_node)

            # Apply effects chain
            current_input = segment_id
            for j, effect in enumerate(effects):
                # Effect can be: string name, Symbol, or dict with params
                effect_name = None
                effect_params = {}

                if isinstance(effect, Symbol):
                    effect_name = effect.name
                elif isinstance(effect, str):
                    effect_name = effect
                elif isinstance(effect, dict):
                    # Dict form: {:effect invert :intensity (bind bass)}
                    effect_name = effect.get("effect")
                    if isinstance(effect_name, Symbol):
                        effect_name = effect_name.name
                    # Copy all params except 'effect'
                    for k, v in effect.items():
                        if k != "effect":
                            effect_params[k] = v

                if not effect_name:
                    continue

                effect_id = f"{base_id}_fx_{i:04d}_{j}"
                effect_entry = registry.get("effects", {}).get(effect_name, {})

                effect_config = {
                    "effect": effect_name,
                    "effect_path": effect_entry.get("path"),
                }
                # Add any params (may include Binding objects)
                effect_config.update(effect_params)

                effect_node = {
                    "id": effect_id,
                    "type": "EFFECT",
                    "config": effect_config,
                    "inputs": [current_input],
                }
                expanded_nodes.append(effect_node)
                current_input = effect_id

            sequence_inputs.append(current_input)

    else:
        # Legacy mode - :effect and :pattern
        effect_name = config.get("effect")
        effect_path = config.get("effect_path")
        pattern = config.get("pattern", "all")
        video_input = default_video

        if not video_input:
            raise ValueError("SLICE_ON requires video input")

        slice_times = list(zip([0] + times[:-1], times))

        for i, (start, end) in enumerate(slice_times):
            if start >= end:
                continue

            # Determine if effect should be applied
            apply_effect = False
            if effect_name:
                if pattern == "all":
                    apply_effect = True
                elif pattern == "odd":
                    apply_effect = (i % 2 == 1)
                elif pattern == "even":
                    apply_effect = (i % 2 == 0)
                elif pattern == "alternate":
                    apply_effect = (i % 2 == 1)

            # Create SEGMENT node
            segment_id = f"{base_id}_seg_{i:04d}"
            segment_node = {
                "id": segment_id,
                "type": "SEGMENT",
                "config": {
                    "start": start,
                    "duration": end - start,
                },
                "inputs": [video_input],
            }
            expanded_nodes.append(segment_node)

            if apply_effect:
                effect_id = f"{base_id}_fx_{i:04d}"
                effect_config = {"effect": effect_name}
                if effect_path:
                    effect_config["effect_path"] = effect_path

                effect_node = {
                    "id": effect_id,
                    "type": "EFFECT",
                    "config": effect_config,
                    "inputs": [segment_id],
                }
                expanded_nodes.append(effect_node)
                sequence_inputs.append(effect_id)
            else:
                sequence_inputs.append(segment_id)
    # Create LIST node to hold all slices (user must explicitly sequence them)
    list_node = {
        "id": node["id"],  # Keep original ID for reference stability
        "type": "LIST",
        "config": {},
        "inputs": sequence_inputs,
    }
    expanded_nodes.append(list_node)

    return expanded_nodes


def _parse_construct_params(params_list: list) -> tuple:
    """
    Parse :params block in a construct definition.

    Syntax:
        (
          (param_name :type string :default "value" :desc "description")
        )

    Returns:
        (param_names, param_defaults) where param_names is a list of strings
        and param_defaults is a dict of param_name -> default_value
    """
    param_names = []
    param_defaults = {}

    for param_def in params_list:
        if not isinstance(param_def, list) or len(param_def) < 1:
            continue

        # First element is the parameter name
        first = param_def[0]
        if isinstance(first, Symbol):
            param_name = first.name
        elif isinstance(first, str):
            param_name = first
        else:
            continue

        param_names.append(param_name)

        # Parse keyword arguments
        default = None
        i = 1
        while i < len(param_def):
            item = param_def[i]
            if isinstance(item, Keyword):
                if i + 1 >= len(param_def):
                    break
                kw_value = param_def[i + 1]

                if item.name == "default":
                    default = kw_value
                # We could also parse :type, :range, :choices, :desc here
                i += 2
            else:
                i += 1

        param_defaults[param_name] = default

    return param_names, param_defaults


def _expand_construct(
    node: Dict,
    registry: Dict,
    sources: Dict[str, str],
    analysis_data: Dict[str, Dict],
    recipe_dir: Path,
    cluster_key: str = None,
    encoding: Dict = None,
) -> List[Dict]:
    """
    Expand a user-defined CONSTRUCT node.

    Loads the construct definition from .sexp file, evaluates it with
    the provided arguments, and converts the result into segment nodes.

    Args:
        node: The CONSTRUCT node to expand
        registry: Recipe registry
        sources: Map of source names to node IDs
        analysis_data: Analysis results (analysis_id -> {times, values})
        recipe_dir: Recipe directory for resolving paths
        cluster_key: Optional cluster key for hashing
        encoding: Encoding config

    Returns:
        List of expanded nodes (segments, effects, list)
    """
    from .parser import parse_all, Symbol
    from .evaluator import evaluate

    config = node.get("config", {})
    construct_name = config.get("construct_name")
    construct_path = config.get("construct_path")
    args = config.get("args", [])

    # Load construct definition
    full_path = recipe_dir / construct_path
    if not full_path.exists():
        raise ValueError(f"Construct file not found: {full_path}")

    print(f"  Loading construct: {construct_name} from {construct_path}", file=sys.stderr)

    construct_text = full_path.read_text()
    construct_sexp = parse_all(construct_text)

    # Parse define-construct: (define-construct name "desc" (params...) body)
    if not isinstance(construct_sexp, list):
        construct_sexp = [construct_sexp]

    # Process imports (effect, construct declarations) in the construct file
    # These extend the registry for this construct's scope
    local_registry = dict(registry)  # Copy parent registry
    construct_def = None

    for expr in construct_sexp:
        if isinstance(expr, list) and expr and isinstance(expr[0], Symbol):
            form_name = expr[0].name

            if form_name == "effect":
                # (effect name :path "...")
                effect_name = expr[1].name if isinstance(expr[1], Symbol) else expr[1]
                # Parse kwargs
                i = 2
                kwargs = {}
                while i < len(expr):
                    if isinstance(expr[i], Keyword):
                        kwargs[expr[i].name] = expr[i + 1] if i + 1 < len(expr) else None
                        i += 2
                    else:
                        i += 1
                local_registry.setdefault("effects", {})[effect_name] = {
                    "path": kwargs.get("path"),
                    "cid": kwargs.get("cid"),
                }
                print(f"    Construct imports effect: {effect_name}", file=sys.stderr)

            elif form_name == "define-construct":
                construct_def = expr

    if not construct_def:
        raise ValueError(f"No define-construct found in {construct_path}")

    # Use local_registry instead of registry from here
    registry = local_registry

    # Parse define-construct - requires :params syntax:
    #   (define-construct name
    #     :params (
    #       (param1 :type string :default "value" :desc "description")
    #     )
    #     body)
    #
    # Legacy syntax (define-construct name "desc" (param1 param2) body) is not supported.
    def_name = construct_def[1].name if isinstance(construct_def[1], Symbol) else construct_def[1]

    params = []  # List of param names
    param_defaults = {}  # param_name -> default value
    body = None
    found_params = False

    idx = 2
    while idx < len(construct_def):
        item = construct_def[idx]
        if isinstance(item, Keyword) and item.name == "params":
            # :params syntax
            if idx + 1 >= len(construct_def):
                raise ValueError(f"Construct '{def_name}': Missing params list after :params keyword")
            params_list = construct_def[idx + 1]
            params, param_defaults = _parse_construct_params(params_list)
            found_params = True
            idx += 2
        elif isinstance(item, Keyword):
            # Skip other keywords (like :desc)
            idx += 2
        elif isinstance(item, str):
            # Skip description strings (but warn about legacy format)
            print(f"  Warning: Description strings in define-construct are deprecated", file=sys.stderr)
            idx += 1
        elif body is None:
            # First non-keyword, non-string item is the body
            if isinstance(item, list) and item:
                first_elem = item[0]
                # Check for legacy params syntax and reject it
                if isinstance(first_elem, Symbol) and first_elem.name not in ("let", "let*", "if", "when", "do", "begin", "->", "map", "filter", "fn", "reduce", "nth"):
                    # Could be legacy params if all items are just symbols
                    if all(isinstance(p, Symbol) for p in item):
                        raise ValueError(
                            f"Construct '{def_name}': Legacy parameter syntax (param1 param2) is not supported. "
                            f"Use :params block instead."
                        )
            body = item
            idx += 1
        else:
            idx += 1

    if body is None:
        raise ValueError(f"No body found in define-construct {def_name}")

    # Build environment with sources and analysis data
    env = dict(sources)

    # Add bindings from compiler (video-a, video-b, etc.)
    if "bindings" in config:
        env.update(config["bindings"])

    # Add effect names so they can be referenced as symbols
    for effect_name in registry.get("effects", {}):
        env[effect_name] = effect_name

    # Map analysis node IDs to their data with :times and :values
    for analysis_id, data in analysis_data.items():
        # Find the name this analysis was bound to
        for name, node_id in sources.items():
            if node_id == analysis_id or name.endswith("-data"):
                env[name] = data
        env[analysis_id] = data

    # Apply param defaults first (for :params syntax)
    for param_name, default_value in param_defaults.items():
        if default_value is not None:
            env[param_name] = default_value

    # Bind positional args to params (overrides defaults)
    param_names = [p.name if isinstance(p, Symbol) else p for p in params]
    for i, param in enumerate(param_names):
        if i < len(args):
            arg = args[i]
            # Resolve node IDs to their data if it's analysis
            if isinstance(arg, str) and arg in analysis_data:
                env[param] = analysis_data[arg]
            else:
                env[param] = arg

    # Helper to resolve node IDs to analysis data recursively
    def resolve_value(val):
        """Resolve node IDs and symbols in a value, including inside dicts/lists."""
        if isinstance(val, str) and val in analysis_data:
            return analysis_data[val]
        elif isinstance(val, str) and val in env:
            return env[val]
        elif isinstance(val, Symbol):
            if val.name in env:
                return env[val.name]
            return val
        elif isinstance(val, dict):
            return {k: resolve_value(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [resolve_value(v) for v in val]
        return val

    # Validate and bind keyword arguments from the config (excluding internal keys)
    # These may be S-expressions that need evaluation (e.g., lambdas)
    # or Symbols that need resolution from bindings
    internal_keys = {"construct_name", "construct_path", "args", "bindings"}
    known_params = set(param_names) | set(param_defaults.keys())
    for key, value in config.items():
        if key not in internal_keys:
            # Convert key to valid identifier (replace - with _) for checking
            param_key = key.replace("-", "_")
            if param_key not in known_params:
                raise ValueError(
                    f"Construct '{def_name}': Unknown parameter '{key}'. "
                    f"Valid parameters are: {', '.join(sorted(known_params)) if known_params else '(none)'}"
                )
            # Evaluate if it's an expression (list starting with Symbol)
            if isinstance(value, list) and value and isinstance(value[0], Symbol):
                env[param_key] = evaluate(value, env)
            elif isinstance(value, Symbol):
                # Resolve Symbol from env/bindings, then resolve any node IDs in the value
                if value.name in env:
                    env[param_key] = resolve_value(env[value.name])
                else:
                    raise ValueError(f"Undefined symbol in construct arg: {value.name}")
            else:
                # Resolve node IDs inside dicts/lists
                env[param_key] = resolve_value(value)

    # Evaluate construct body
    print(f"  Evaluating construct with params: {param_names}", file=sys.stderr)
    segments = evaluate(body, env)

    if not isinstance(segments, list):
        raise ValueError(f"Construct must return a list of segments, got {type(segments)}")

    print(f"  Construct produced {len(segments)} segments", file=sys.stderr)

    # Convert segment descriptors to plan nodes
    expanded_nodes = []
    sequence_inputs = []
    base_id = node["id"][:8]

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        source_ref = seg.get("source")
        start = seg.get("start", 0)
        print(f"    DEBUG segment {i}: source={str(source_ref)[:20]}... start={start}", file=sys.stderr)
        end = seg.get("end")
        duration = seg.get("duration") or (end - start if end else 1.0)
        effects = seg.get("effects", [])

        # Resolve source reference to node ID
        source_id = sources.get(source_ref, source_ref) if isinstance(source_ref, str) else source_ref

        # Create segment node
        segment_id = f"{base_id}_seg_{i:04d}"
        segment_node = {
            "id": segment_id,
            "type": "SEGMENT",
            "config": {
                "start": start,
                "duration": duration,
            },
            "inputs": [source_id] if source_id else [],
        }
        expanded_nodes.append(segment_node)

        # Add effects if specified
        if effects:
            prev_id = segment_id
            for j, eff in enumerate(effects):
                effect_name = eff.get("effect") if isinstance(eff, dict) else eff
                effect_id = f"{base_id}_fx_{i:04d}_{j:02d}"
                # Look up effect_path from registry (prevents collapsing Python effects)
                effect_entry = registry.get("effects", {}).get(effect_name, {})
                effect_config = {
                    "effect": effect_name,
                    **{k: v for k, v in (eff.items() if isinstance(eff, dict) else []) if k != "effect"},
                }
                if effect_entry.get("path"):
                    effect_config["effect_path"] = effect_entry["path"]
                effect_node = {
                    "id": effect_id,
                    "type": "EFFECT",
                    "config": effect_config,
                    "inputs": [prev_id],
                }
                expanded_nodes.append(effect_node)
                prev_id = effect_id
            sequence_inputs.append(prev_id)
        else:
            sequence_inputs.append(segment_id)

    # Create LIST node
    list_node = {
        "id": node["id"],
        "type": "LIST",
        "config": {},
        "inputs": sequence_inputs,
    }
    expanded_nodes.append(list_node)

    return expanded_nodes


def _expand_nodes(
    nodes: List[Dict],
    registry: Dict,
    recipe_dir: Path,
    source_paths: Dict[str, Path],
    work_dir: Path = None,
    cluster_key: str = None,
    on_analysis: Callable[[str, Dict], None] = None,
    encoding: Dict = None,
    pre_analysis: Dict[str, Dict] = None,
) -> List[Dict]:
    """
    Expand dynamic nodes (SLICE_ON) by running analyzers.

    Processes nodes in dependency order:
    1. SOURCE nodes: resolve file paths
    2. SEGMENT nodes: pre-execute if needed for analysis
    3. ANALYZE nodes: run analyzers (or use pre_analysis), store results
    4. SLICE_ON nodes: expand using analysis results

    Args:
        nodes: List of compiled nodes
        registry: Recipe registry
        recipe_dir: Directory for resolving relative paths
        source_paths: Resolved source paths (id -> path)
        work_dir: Working directory for temporary files (created if None)
        cluster_key: Optional cluster key
        on_analysis: Callback when analysis completes (node_id, results)
        pre_analysis: Pre-computed analysis data (name -> results)

    Returns:
        Tuple of (expanded_nodes, named_analysis) where:
        - expanded_nodes: List with SLICE_ON replaced by primitives
        - named_analysis: Dict of analyzer_name -> {times, values}
    """
    import tempfile

    nodes_by_id = {n["id"]: n for n in nodes}
    sorted_ids = _topological_sort(nodes)

    # Create work directory if needed
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="artdag_plan_"))

    # Track outputs and analysis results
    outputs = {}  # node_id -> output path or analysis data
    analysis_results = {}  # node_id -> analysis dict
    named_analysis = {}  # analyzer_name -> analysis dict (for effect bindings)
    pre_executed = set()  # nodes pre-executed during planning
    expanded = []
    expanded_ids = set()

    for node_id in sorted_ids:
        node = nodes_by_id[node_id]
        node_type = node["type"]

        if node_type == "SOURCE":
            # Resolve source path
            config = node.get("config", {})
            if "path" in config:
                path = recipe_dir / config["path"]
                outputs[node_id] = path.resolve()
                source_paths[node_id] = outputs[node_id]
            expanded.append(node)
            expanded_ids.add(node_id)

        elif node_type == "SEGMENT":
            # Check if this segment's input is resolved
            inputs = node.get("inputs", [])
            if inputs and inputs[0] in outputs:
                input_path = outputs[inputs[0]]
                if isinstance(input_path, Path):
                    # Pre-execute segment to get output path
                    # This is needed if ANALYZE depends on this segment
                    import sys
                    print(f"  Pre-executing segment: {node_id[:16]}...", file=sys.stderr)
                    output_path = _pre_execute_segment(node, input_path, work_dir)
                    outputs[node_id] = output_path
                    pre_executed.add(node_id)
            expanded.append(node)
            expanded_ids.add(node_id)

        elif node_type == "ANALYZE":
            # Get or run analysis
            config = node.get("config", {})
            analysis_name = node.get("name") or config.get("analyzer")

            # Check for pre-computed analysis first
            if pre_analysis and analysis_name and analysis_name in pre_analysis:
                import sys
                print(f"  Using pre-computed analysis: {analysis_name}", file=sys.stderr)
                results = pre_analysis[analysis_name]
            else:
                # Run analyzer to get concrete data
                analyzer_path = config.get("analyzer_path")
                node_inputs = node.get("inputs", [])

                if not node_inputs:
                    raise ValueError(f"ANALYZE node {node_id} has no inputs")

                # Get input path - could be SOURCE or pre-executed SEGMENT
                input_id = node_inputs[0]
                input_path = outputs.get(input_id)

                if input_path is None:
                    raise ValueError(
                        f"ANALYZE input {input_id} not resolved. "
                        "Check that input SOURCE or SEGMENT exists."
                    )

                if not isinstance(input_path, Path):
                    raise ValueError(
                        f"ANALYZE input {input_id} is not a file path: {type(input_path)}"
                    )

                if analyzer_path:
                    full_path = recipe_dir / analyzer_path
                    params = {k: v for k, v in config.items()
                             if k not in ("analyzer", "analyzer_path", "cid")}
                    import sys
                    print(f"  Running analyzer: {config.get('analyzer', 'unknown')}", file=sys.stderr)
                    results = _run_analyzer(full_path, input_path, params)
                else:
                    raise ValueError(f"ANALYZE node {node_id} missing analyzer_path")

            analysis_results[node_id] = results
            outputs[node_id] = results

            # Store by name for effect binding resolution
            if analysis_name:
                named_analysis[analysis_name] = results

            if on_analysis:
                on_analysis(node_id, results)

            # Keep ANALYZE node in plan (it produces a JSON artifact)
            expanded.append(node)
            expanded_ids.add(node_id)

        elif node_type == "SLICE_ON":
            # Expand into primitives using analysis results
            inputs = node.get("inputs", [])
            config = node.get("config", {})

            # Lambda mode can have just 1 input (analysis), legacy needs 2 (video + analysis)
            has_lambda = "fn" in config
            if has_lambda:
                if len(inputs) < 1:
                    raise ValueError(f"SLICE_ON {node_id} requires analysis input")
                analysis_id = inputs[-1]  # Last input is analysis
            else:
                if len(inputs) < 2:
                    raise ValueError(f"SLICE_ON {node_id} requires video and analysis inputs")
                analysis_id = inputs[1]

            if analysis_id not in analysis_results:
                raise ValueError(
                    f"SLICE_ON {node_id} analysis input {analysis_id} not found"
                )

            # Build sources map: name -> node_id
            # This lets the lambda reference videos by name
            sources = {}
            for n in nodes:
                if n.get("name"):
                    sources[n["name"]] = n["id"]

            analysis_data = analysis_results[analysis_id]
            slice_nodes = _expand_slice_on(node, analysis_data, registry, sources, cluster_key, encoding)

            for sn in slice_nodes:
                if sn["id"] not in expanded_ids:
                    expanded.append(sn)
                    expanded_ids.add(sn["id"])

        elif node_type == "CONSTRUCT":
            # Expand user-defined construct
            config = node.get("config", {})
            construct_name = config.get("construct_name")
            construct_path = config.get("construct_path")

            if not construct_path:
                raise ValueError(f"CONSTRUCT {node_id} missing path")

            # Build sources map
            sources = {}
            for n in nodes:
                if n.get("name"):
                    sources[n["name"]] = n["id"]

            # Get analysis data if referenced
            inputs = node.get("inputs", [])
            analysis_data = {}
            for inp in inputs:
                if inp in analysis_results:
                    analysis_data[inp] = analysis_results[inp]

            construct_nodes = _expand_construct(
                node, registry, sources, analysis_data, recipe_dir, cluster_key, encoding
            )

            for cn in construct_nodes:
                if cn["id"] not in expanded_ids:
                    expanded.append(cn)
                    expanded_ids.add(cn["id"])

        else:
            # Keep other nodes as-is
            expanded.append(node)
            expanded_ids.add(node_id)

    return expanded, named_analysis


def create_plan(
    recipe: CompiledRecipe,
    inputs: Dict[str, str] = None,
    recipe_dir: Path = None,
    cluster_key: str = None,
    on_analysis: Callable[[str, Dict], None] = None,
    pre_analysis: Dict[str, Dict] = None,
) -> ExecutionPlanSexp:
    """
    Create an execution plan from a compiled recipe.

    Args:
        recipe: Compiled S-expression recipe
        inputs: Mapping of input names to content hashes
        recipe_dir: Directory for resolving relative paths (required for analyzers)
        cluster_key: Optional cluster key for cache isolation
        on_analysis: Callback when analysis completes (node_id, results)
        pre_analysis: Pre-computed analysis data (name -> results), skips running analyzers

    Returns:
        ExecutionPlanSexp with all cache IDs computed

    Example:
        >>> recipe = compile_string('(recipe "test" (-> (source cat) (effect identity)))')
        >>> plan = create_plan(recipe, inputs={}, recipe_dir=Path("."))
        >>> print(plan.to_string())
    """
    inputs = inputs or {}

    # Compute recipe hash
    recipe_hash = _stable_hash(recipe.to_dict(), cluster_key)

    # Check if recipe has expandable nodes (SLICE_ON, etc.)
    has_expandable = any(n["type"] in EXPANDABLE_TYPES for n in recipe.nodes)
    named_analysis = {}

    if has_expandable:
        if recipe_dir is None:
            raise ValueError("recipe_dir required for recipes with SLICE_ON nodes")

        # Expand dynamic nodes (runs analyzers, expands SLICE_ON)
        source_paths = {}
        expanded_nodes, named_analysis = _expand_nodes(
            recipe.nodes,
            recipe.registry,
            recipe_dir,
            source_paths,
            cluster_key=cluster_key,
            on_analysis=on_analysis,
            encoding=recipe.encoding,
            pre_analysis=pre_analysis,
        )
        # Expand LIST inputs in SEQUENCE nodes
        expanded_nodes = _expand_list_inputs(expanded_nodes)
        # Collapse effect chains after expansion
        collapsed_nodes = _collapse_effect_chains(expanded_nodes, recipe.registry)
    else:
        # No expansion needed
        collapsed_nodes = _collapse_effect_chains(recipe.nodes, recipe.registry)

    # Build node lookup from collapsed nodes
    nodes_by_id = {node["id"]: node for node in collapsed_nodes}

    # Topological sort
    sorted_ids = _topological_sort(collapsed_nodes)

    # Create steps with resolved hashes
    steps = []
    cache_ids = {}  # step_id -> cache_id

    for node_id in sorted_ids:
        node = nodes_by_id[node_id]
        step = _create_step(
            node,
            recipe.registry,
            inputs,
            cache_ids,
            cluster_key,
        )
        steps.append(step)
        cache_ids[node_id] = step.cache_id

    # Compute levels
    _compute_levels(steps, nodes_by_id)

    # Handle stage-aware planning if recipe has stages
    stage_plans = []
    stage_order = []
    stage_levels = {}
    stage_cache_ids = {}

    if recipe.stages:
        # Build mapping from node_id to stage
        node_to_stage = {}
        for stage in recipe.stages:
            for node_id in stage.node_ids:
                node_to_stage[node_id] = stage.name

        # Compute stage levels (for parallel execution)
        stage_levels = _compute_stage_levels(recipe.stages)

        # Compute stage cache IDs
        # Stage cache ID = hash of: stage_name + required stage cache_ids + node content
        stage_cache_ids = {}
        for stage_name in recipe.stage_order:
            stage = next(s for s in recipe.stages if s.name == stage_name)
            stage_cache_ids[stage_name] = _compute_stage_cache_id(
                stage,
                stage_cache_ids,
                cache_ids,
                cluster_key,
            )

        # Tag each step with stage info
        for step in steps:
            if step.step_id in node_to_stage:
                step.stage = node_to_stage[step.step_id]
                step.stage_cache_id = stage_cache_ids.get(step.stage)

        # Build stage plans
        for stage_name in recipe.stage_order:
            stage = next(s for s in recipe.stages if s.name == stage_name)
            stage_steps = [s for s in steps if s.stage == stage_name]

            # Build output bindings with cache IDs
            output_cache_ids = {}
            for out_name, node_id in stage.output_bindings.items():
                if node_id in cache_ids:
                    output_cache_ids[out_name] = cache_ids[node_id]

            stage_plans.append(StagePlan(
                stage_name=stage_name,
                cache_id=stage_cache_ids[stage_name],
                steps=stage_steps,
                requires=stage.requires,
                output_bindings=output_cache_ids,
                level=stage_levels.get(stage_name, 0),
            ))

        stage_order = recipe.stage_order

    # Compute plan ID
    plan_content = {
        "recipe_hash": recipe_hash,
        "steps": [{"id": s.step_id, "cache_id": s.cache_id} for s in steps],
        "inputs": inputs,
    }
    if stage_cache_ids:
        plan_content["stage_cache_ids"] = stage_cache_ids
    plan_id = _stable_hash(plan_content, cluster_key)

    return ExecutionPlanSexp(
        plan_id=plan_id,
        recipe_id=recipe.name,
        recipe_hash=recipe_hash,
        steps=steps,
        output_step_id=recipe.output_node_id,
        inputs=inputs,
        analysis=named_analysis,
        stage_plans=stage_plans,
        stage_order=stage_order,
        stage_levels=stage_levels,
        stage_cache_ids=stage_cache_ids,
        effects_registry=recipe.registry.get("effects", {}),
        minimal_primitives=recipe.minimal_primitives,
    )


def _topological_sort(nodes: List[Dict]) -> List[str]:
    """Sort nodes in dependency order."""
    nodes_by_id = {n["id"]: n for n in nodes}
    visited = set()
    order = []

    def visit(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)
        node = nodes_by_id.get(node_id)
        if node:
            for input_id in node.get("inputs", []):
                visit(input_id)
            order.append(node_id)

    for node in nodes:
        visit(node["id"])

    return order


def _create_step(
    node: Dict,
    registry: Dict,
    inputs: Dict[str, str],
    cache_ids: Dict[str, str],
    cluster_key: str = None,
) -> PlanStep:
    """Create a PlanStep from a node definition."""
    node_id = node["id"]
    node_type = node["type"]
    config = dict(node.get("config", {}))
    node_inputs = node.get("inputs", [])

    # Resolve registry references
    resolved_config = _resolve_config(config, registry, inputs)

    # Get input cache IDs
    input_cache_ids = [cache_ids[inp] for inp in node_inputs if inp in cache_ids]

    # Compute cache ID
    cache_content = {
        "node_type": node_type,
        "config": resolved_config,
        "inputs": sorted(input_cache_ids),
    }
    cache_id = _stable_hash(cache_content, cluster_key)

    return PlanStep(
        step_id=node_id,
        node_type=node_type,
        config=resolved_config,
        inputs=node_inputs,
        cache_id=cache_id,
    )


def _resolve_config(
    config: Dict,
    registry: Dict,
    inputs: Dict[str, str],
) -> Dict:
    """Resolve registry references in config to content hashes."""
    resolved = {}

    for key, value in config.items():
        if key == "filter_chain" and isinstance(value, list):
            # Resolve each filter in the chain (for COMPOUND nodes)
            resolved_chain = []
            for filter_item in value:
                filter_config = filter_item.get("config", {})
                resolved_filter_config = _resolve_config(filter_config, registry, inputs)
                resolved_chain.append({
                    "type": filter_item["type"],
                    "config": resolved_filter_config,
                })
            resolved["filter_chain"] = resolved_chain

        elif key == "asset" and isinstance(value, str):
            # Resolve asset reference - use CID from registry
            if value in registry.get("assets", {}):
                resolved["cid"] = registry["assets"][value]["cid"]
            else:
                resolved["asset"] = value  # Keep as-is if not in registry

        elif key == "effect" and isinstance(value, str):
            # Resolve effect reference - keep name AND add CID/path
            resolved["effect"] = value
            if value in registry.get("effects", {}):
                effect_entry = registry["effects"][value]
                if effect_entry.get("cid"):
                    resolved["cid"] = effect_entry["cid"]
                if effect_entry.get("path"):
                    resolved["effect_path"] = effect_entry["path"]

        elif key == "input" and value is True:
            # Variable input - resolve from inputs dict
            input_name = config.get("name", "input")
            if input_name in inputs:
                resolved["hash"] = inputs[input_name]
            else:
                resolved["input"] = True
                resolved["name"] = input_name

        elif key == "path":
            # Local file path - keep as-is for local execution
            resolved["path"] = value

        else:
            resolved[key] = value

    return resolved


def _compute_levels(steps: List[PlanStep], nodes_by_id: Dict) -> None:
    """Compute dependency levels for steps."""
    levels = {}

    def compute_level(step_id: str) -> int:
        if step_id in levels:
            return levels[step_id]

        node = nodes_by_id.get(step_id)
        if not node or not node.get("inputs"):
            levels[step_id] = 0
            return 0

        max_input = max(compute_level(inp) for inp in node["inputs"])
        levels[step_id] = max_input + 1
        return levels[step_id]

    for step in steps:
        step.level = compute_level(step.step_id)


def _compute_stage_levels(stages: List) -> Dict[str, int]:
    """
    Compute stage levels for parallel execution.

    Stages at the same level have no dependencies between them
    and can run in parallel.
    """
    from .compiler import CompiledStage

    levels = {}

    def compute_level(stage_name: str) -> int:
        if stage_name in levels:
            return levels[stage_name]

        stage = next((s for s in stages if s.name == stage_name), None)
        if not stage or not stage.requires:
            levels[stage_name] = 0
            return 0

        max_req = max(compute_level(req) for req in stage.requires)
        levels[stage_name] = max_req + 1
        return levels[stage_name]

    for stage in stages:
        compute_level(stage.name)

    return levels


def _compute_stage_cache_id(
    stage,
    stage_cache_ids: Dict[str, str],
    node_cache_ids: Dict[str, str],
    cluster_key: str = None,
) -> str:
    """
    Compute cache ID for a stage.

    Stage cache ID = SHA3-256 of:
    - stage name
    - required stage cache IDs
    - node cache IDs in the stage
    """
    cache_content = {
        "stage": stage.name,
        "requires": {req: stage_cache_ids.get(req, "") for req in stage.requires},
        "nodes": sorted([
            node_cache_ids.get(node_id, node_id)
            for node_id in stage.node_ids
        ]),
    }
    return _stable_hash(cache_content, cluster_key)


def step_to_task_sexp(step: PlanStep) -> List:
    """
    Convert a step to a minimal S-expression for Celery task.

    This is the S-expression that gets sent to a worker.
    The worker hashes this to verify cache_id.
    """
    sexp = [Symbol(step.node_type.lower())]

    # Add resolved config
    for key, value in step.config.items():
        sexp.extend([Keyword(key), value])

    # Add input cache IDs (not step IDs)
    if step.inputs:
        sexp.extend([Keyword("inputs"), step.inputs])

    return sexp


def task_cache_id(task_sexp: List, cluster_key: str = None) -> str:
    """
    Compute cache ID from task S-expression.

    This allows workers to verify they're executing the right task.
    """
    # Serialize S-expression to canonical form
    canonical = serialize(task_sexp)
    return _stable_hash({"sexp": canonical}, cluster_key)
