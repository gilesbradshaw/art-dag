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

    # Build node lookup
    nodes_by_id = {node["id"]: node for node in recipe.nodes}

    # Topological sort
    sorted_ids = _topological_sort(recipe.nodes)

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
        if key == "asset" and isinstance(value, str):
            # Resolve asset reference
            if value in registry.get("assets", {}):
                resolved["hash"] = registry["assets"][value]["hash"]
            else:
                resolved["asset"] = value  # Keep as-is if not in registry

        elif key == "effect" and isinstance(value, str):
            # Resolve effect reference
            if value in registry.get("effects", {}):
                resolved["hash"] = registry["effects"][value]["hash"]
            else:
                resolved["effect"] = value  # Keep as-is if not in registry

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
