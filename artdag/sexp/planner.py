"""
Execution plan generation from S-expression recipes.

The planner:
1. Takes a compiled recipe + input content hashes
2. Resolves all registry references to content hashes
3. Generates an execution plan (also as S-expression)
4. Computes cache IDs for each step

Plans are S-expressions with all references resolved to hashes,
ready for distribution to Celery workers.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .parser import Symbol, Keyword, serialize
from .compiler import CompiledRecipe


# Node types that can be collapsed into a single FFmpeg filter chain
COLLAPSIBLE_TYPES = {"EFFECT", "TRANSFORM", "RESIZE", "SEGMENT"}

# Node types that are boundaries (sources, merges, or special processing)
BOUNDARY_TYPES = {"SOURCE", "SEQUENCE", "LAYER", "BLEND", "MUX", "ANALYZE"}


def _stable_hash(data: Any, cluster_key: str = None) -> str:
    """Create stable SHA3-256 hash from data."""
    if cluster_key:
        data = {"_cluster_key": cluster_key, "_data": data}
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha3_256(json_str.encode()).hexdigest()


@dataclass
class PlanStep:
    """A step in the execution plan."""
    step_id: str
    node_type: str
    config: Dict[str, Any]
    inputs: List[str]  # List of input step_ids
    cache_id: str
    level: int = 0

    def to_sexp(self) -> List:
        """Convert to S-expression."""
        sexp = [Symbol("step"), self.step_id]

        # Add cache-id
        sexp.extend([Keyword("cache-id"), self.cache_id])

        # Add level if > 0
        if self.level > 0:
            sexp.extend([Keyword("level"), self.level])

        # Add the node expression
        node_sexp = [Symbol(self.node_type.lower())]

        # Add config as keywords
        for key, value in self.config.items():
            node_sexp.extend([Keyword(key), value])

        # Add inputs if any
        if self.inputs:
            node_sexp.extend([Keyword("inputs"), self.inputs])

        sexp.append(node_sexp)
        return sexp


@dataclass
class ExecutionPlanSexp:
    """Execution plan as S-expression."""
    plan_id: str
    recipe_id: str
    recipe_hash: str
    steps: List[PlanStep]
    output_step_id: str
    inputs: Dict[str, str] = field(default_factory=dict)  # name -> hash
    metadata: Dict[str, Any] = field(default_factory=dict)

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

        # Steps
        for step in self.steps:
            sexp.append(step.to_sexp())

        # Output reference
        sexp.extend([Keyword("output"), self.output_step_id])

        return sexp

    def to_string(self, pretty: bool = True) -> str:
        """Serialize plan to S-expression string."""
        return serialize(self.to_sexp(), pretty=pretty)


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
            },
            "inputs": start_node.get("inputs", []),
            "name": f"compound_{len(filter_chain)}_effects",
        }

        result_nodes.append(compound_node)

        # Mark all chain nodes as consumed
        for chain_node_id in chain:
            consumed.add(chain_node_id)

    return result_nodes


def create_plan(
    recipe: CompiledRecipe,
    inputs: Dict[str, str] = None,
    cluster_key: str = None,
) -> ExecutionPlanSexp:
    """
    Create an execution plan from a compiled recipe.

    Args:
        recipe: Compiled S-expression recipe
        inputs: Mapping of input names to content hashes
        cluster_key: Optional cluster key for cache isolation

    Returns:
        ExecutionPlanSexp with all cache IDs computed

    Example:
        >>> recipe = compile_string('(recipe "test" (-> (source cat) (effect identity)))')
        >>> plan = create_plan(recipe, inputs={})
        >>> print(plan.to_string())
    """
    inputs = inputs or {}

    # Compute recipe hash
    recipe_hash = _stable_hash(recipe.to_dict(), cluster_key)

    # Collapse sequential effect chains into compound nodes
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

    # Compute plan ID
    plan_content = {
        "recipe_hash": recipe_hash,
        "steps": [{"id": s.step_id, "cache_id": s.cache_id} for s in steps],
        "inputs": inputs,
    }
    plan_id = _stable_hash(plan_content, cluster_key)

    return ExecutionPlanSexp(
        plan_id=plan_id,
        recipe_id=recipe.name,
        recipe_hash=recipe_hash,
        steps=steps,
        output_step_id=recipe.output_node_id,
        inputs=inputs,
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
            # Resolve asset reference
            if value in registry.get("assets", {}):
                resolved["hash"] = registry["assets"][value]["hash"]
            else:
                resolved["asset"] = value  # Keep as-is if not in registry

        elif key == "effect" and isinstance(value, str):
            # Resolve effect reference - keep name AND add CID
            resolved["effect"] = value
            if value in registry.get("effects", {}):
                resolved["cid"] = registry["effects"][value]["cid"]

        elif key == "input" and value is True:
            # Variable input - resolve from inputs dict
            input_name = config.get("name", "input")
            if input_name in inputs:
                resolved["hash"] = inputs[input_name]
            else:
                resolved["input"] = True
                resolved["name"] = input_name

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
