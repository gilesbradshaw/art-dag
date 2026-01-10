# artdag/planning/planner.py
"""
Recipe planner - converts recipes into execution plans.

The planner is the second phase of the 3-phase execution model.
It takes a recipe and analysis results and generates a complete
execution plan with pre-computed cache IDs.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .schema import ExecutionPlan, ExecutionStep
from .tree_reduction import TreeReducer, reduce_sequence
from ..analysis import AnalysisResult

logger = logging.getLogger(__name__)


def _stable_hash(data: Any, algorithm: str = "sha3_256") -> str:
    """Create stable hash from arbitrary data."""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode())
    return hasher.hexdigest()


@dataclass
class RecipeNode:
    """A node in the recipe DAG."""
    id: str
    type: str
    config: Dict[str, Any]
    inputs: List[str]


@dataclass
class Recipe:
    """Parsed recipe structure."""
    name: str
    version: str
    description: str
    nodes: List[RecipeNode]
    output: str
    registry: Dict[str, Any]
    owner: str
    raw_yaml: str

    @property
    def recipe_hash(self) -> str:
        """Compute hash of recipe content."""
        return _stable_hash({"yaml": self.raw_yaml})

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "Recipe":
        """Parse recipe from YAML string."""
        data = yaml.safe_load(yaml_content)

        nodes = []
        for node_data in data.get("dag", {}).get("nodes", []):
            # Handle both 'inputs' as list and 'inputs' as dict
            inputs = node_data.get("inputs", [])
            if isinstance(inputs, dict):
                # Extract input references from dict structure
                input_list = []
                for key, value in inputs.items():
                    if isinstance(value, str):
                        input_list.append(value)
                    elif isinstance(value, list):
                        input_list.extend(value)
                inputs = input_list
            elif isinstance(inputs, str):
                inputs = [inputs]

            nodes.append(RecipeNode(
                id=node_data["id"],
                type=node_data["type"],
                config=node_data.get("config", {}),
                inputs=inputs,
            ))

        return cls(
            name=data.get("name", "unnamed"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            nodes=nodes,
            output=data.get("dag", {}).get("output", ""),
            registry=data.get("registry", {}),
            owner=data.get("owner", ""),
            raw_yaml=yaml_content,
        )

    @classmethod
    def from_file(cls, path: Path) -> "Recipe":
        """Load recipe from YAML file."""
        with open(path, "r") as f:
            return cls.from_yaml(f.read())


class RecipePlanner:
    """
    Generates execution plans from recipes.

    The planner:
    1. Parses the recipe
    2. Resolves fixed inputs from registry
    3. Maps variable inputs to provided hashes
    4. Expands MAP/iteration nodes
    5. Applies tree reduction for SEQUENCE nodes
    6. Computes cache IDs for all steps
    """

    def __init__(self, use_tree_reduction: bool = True):
        """
        Initialize the planner.

        Args:
            use_tree_reduction: Whether to use tree reduction for SEQUENCE
        """
        self.use_tree_reduction = use_tree_reduction

    def plan(
        self,
        recipe: Recipe,
        input_hashes: Dict[str, str],
        analysis: Optional[Dict[str, AnalysisResult]] = None,
    ) -> ExecutionPlan:
        """
        Generate an execution plan from a recipe.

        Args:
            recipe: The parsed recipe
            input_hashes: Mapping from input name to content hash
            analysis: Analysis results for inputs (keyed by hash)

        Returns:
            ExecutionPlan with pre-computed cache IDs
        """
        logger.info(f"Planning recipe: {recipe.name}")

        # Build node lookup
        nodes_by_id = {n.id: n for n in recipe.nodes}

        # Topologically sort nodes
        sorted_ids = self._topological_sort(recipe.nodes)

        # Resolve registry references
        registry_hashes = self._resolve_registry(recipe.registry)

        # Generate steps
        steps = []
        step_id_map = {}  # Maps recipe node ID to step ID(s)
        analysis_cache_ids = {}

        for node_id in sorted_ids:
            node = nodes_by_id[node_id]
            logger.debug(f"Processing node: {node.id} ({node.type})")

            new_steps, output_step_id = self._process_node(
                node=node,
                step_id_map=step_id_map,
                input_hashes=input_hashes,
                registry_hashes=registry_hashes,
                analysis=analysis or {},
            )

            steps.extend(new_steps)
            step_id_map[node_id] = output_step_id

        # Find output step
        output_step = step_id_map.get(recipe.output)
        if not output_step:
            raise ValueError(f"Output node '{recipe.output}' not found")

        # Build analysis cache IDs
        if analysis:
            analysis_cache_ids = {
                h: a.cache_id for h, a in analysis.items()
                if a.cache_id
            }

        # Create plan
        plan = ExecutionPlan(
            plan_id=None,  # Computed in __post_init__
            recipe_id=recipe.name,
            recipe_hash=recipe.recipe_hash,
            steps=steps,
            output_step=output_step,
            analysis_cache_ids=analysis_cache_ids,
            input_hashes=input_hashes,
            metadata={
                "recipe_version": recipe.version,
                "recipe_description": recipe.description,
                "owner": recipe.owner,
            },
        )

        # Compute all cache IDs
        plan.compute_all_cache_ids()
        plan.compute_levels()

        logger.info(f"Generated plan with {len(steps)} steps")
        return plan

    def plan_from_yaml(
        self,
        yaml_content: str,
        input_hashes: Dict[str, str],
        analysis: Optional[Dict[str, AnalysisResult]] = None,
    ) -> ExecutionPlan:
        """
        Generate plan from YAML string.

        Args:
            yaml_content: Recipe YAML content
            input_hashes: Mapping from input name to content hash
            analysis: Analysis results

        Returns:
            ExecutionPlan
        """
        recipe = Recipe.from_yaml(yaml_content)
        return self.plan(recipe, input_hashes, analysis)

    def plan_from_file(
        self,
        recipe_path: Path,
        input_hashes: Dict[str, str],
        analysis: Optional[Dict[str, AnalysisResult]] = None,
    ) -> ExecutionPlan:
        """
        Generate plan from recipe file.

        Args:
            recipe_path: Path to recipe YAML file
            input_hashes: Mapping from input name to content hash
            analysis: Analysis results

        Returns:
            ExecutionPlan
        """
        recipe = Recipe.from_file(recipe_path)
        return self.plan(recipe, input_hashes, analysis)

    def _topological_sort(self, nodes: List[RecipeNode]) -> List[str]:
        """Topologically sort recipe nodes."""
        nodes_by_id = {n.id: n for n in nodes}
        visited = set()
        order = []

        def visit(node_id: str):
            if node_id in visited:
                return
            if node_id not in nodes_by_id:
                return  # External input
            visited.add(node_id)
            node = nodes_by_id[node_id]
            for input_id in node.inputs:
                visit(input_id)
            order.append(node_id)

        for node in nodes:
            visit(node.id)

        return order

    def _resolve_registry(self, registry: Dict[str, Any]) -> Dict[str, str]:
        """
        Resolve registry references to content hashes.

        Args:
            registry: Registry section from recipe

        Returns:
            Mapping from name to content hash
        """
        hashes = {}

        # Assets
        for name, asset_data in registry.get("assets", {}).items():
            if isinstance(asset_data, dict) and "hash" in asset_data:
                hashes[name] = asset_data["hash"]
            elif isinstance(asset_data, str):
                hashes[name] = asset_data

        # Effects
        for name, effect_data in registry.get("effects", {}).items():
            if isinstance(effect_data, dict) and "hash" in effect_data:
                hashes[f"effect:{name}"] = effect_data["hash"]
            elif isinstance(effect_data, str):
                hashes[f"effect:{name}"] = effect_data

        return hashes

    def _process_node(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
        input_hashes: Dict[str, str],
        registry_hashes: Dict[str, str],
        analysis: Dict[str, AnalysisResult],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process a recipe node into execution steps.

        Args:
            node: Recipe node to process
            step_id_map: Mapping from processed node IDs to step IDs
            input_hashes: User-provided input hashes
            registry_hashes: Registry-resolved hashes
            analysis: Analysis results

        Returns:
            Tuple of (new steps, output step ID)
        """
        # SOURCE nodes
        if node.type == "SOURCE":
            return self._process_source(node, input_hashes, registry_hashes)

        # SOURCE_LIST nodes
        if node.type == "SOURCE_LIST":
            return self._process_source_list(node, input_hashes)

        # ANALYZE nodes
        if node.type == "ANALYZE":
            return self._process_analyze(node, step_id_map, analysis)

        # MAP nodes
        if node.type == "MAP":
            return self._process_map(node, step_id_map, input_hashes, analysis)

        # SEQUENCE nodes (may use tree reduction)
        if node.type == "SEQUENCE":
            return self._process_sequence(node, step_id_map)

        # SEGMENT_AT nodes
        if node.type == "SEGMENT_AT":
            return self._process_segment_at(node, step_id_map, analysis)

        # Standard nodes (SEGMENT, RESIZE, TRANSFORM, LAYER, MUX, BLEND, etc.)
        return self._process_standard(node, step_id_map)

    def _process_source(
        self,
        node: RecipeNode,
        input_hashes: Dict[str, str],
        registry_hashes: Dict[str, str],
    ) -> Tuple[List[ExecutionStep], str]:
        """Process SOURCE node."""
        config = dict(node.config)

        # Variable input?
        if config.get("input"):
            # Look up in user-provided inputs
            if node.id not in input_hashes:
                raise ValueError(f"Missing input for SOURCE node '{node.id}'")
            content_hash = input_hashes[node.id]
        # Fixed asset from registry?
        elif config.get("asset"):
            asset_name = config["asset"]
            if asset_name not in registry_hashes:
                raise ValueError(f"Asset '{asset_name}' not found in registry")
            content_hash = registry_hashes[asset_name]
        else:
            raise ValueError(f"SOURCE node '{node.id}' has no input or asset")

        step = ExecutionStep(
            step_id=node.id,
            node_type="SOURCE",
            config={"input_ref": node.id, "content_hash": content_hash},
            input_steps=[],
            cache_id=content_hash,  # SOURCE cache_id is just the content hash
            name=config.get("name", node.id),
        )

        return [step], step.step_id

    def _process_source_list(
        self,
        node: RecipeNode,
        input_hashes: Dict[str, str],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process SOURCE_LIST node.

        Creates individual SOURCE steps for each item in the list.
        """
        # Look for list input
        if node.id not in input_hashes:
            raise ValueError(f"Missing input for SOURCE_LIST node '{node.id}'")

        input_value = input_hashes[node.id]

        # Parse as comma-separated list if string
        if isinstance(input_value, str):
            items = [h.strip() for h in input_value.split(",")]
        else:
            items = list(input_value)

        steps = []
        for i, content_hash in enumerate(items):
            step = ExecutionStep(
                step_id=f"{node.id}_{i}",
                node_type="SOURCE",
                config={"input_ref": f"{node.id}[{i}]", "content_hash": content_hash},
                input_steps=[],
                cache_id=content_hash,
                name=f"{node.config.get('name', node.id)}[{i}]",
            )
            steps.append(step)

        # Return list marker as output
        list_step = ExecutionStep(
            step_id=node.id,
            node_type="_LIST",
            config={"items": [s.step_id for s in steps]},
            input_steps=[s.step_id for s in steps],
            name=node.config.get("name", node.id),
        )
        steps.append(list_step)

        return steps, list_step.step_id

    def _process_analyze(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
        analysis: Dict[str, AnalysisResult],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process ANALYZE node.

        ANALYZE nodes reference pre-computed analysis results.
        """
        input_step = step_id_map.get(node.inputs[0]) if node.inputs else None
        if not input_step:
            raise ValueError(f"ANALYZE node '{node.id}' has no input")

        feature = node.config.get("feature", "all")

        step = ExecutionStep(
            step_id=node.id,
            node_type="ANALYZE",
            config={
                "feature": feature,
                **node.config,
            },
            input_steps=[input_step],
            name=f"Analyze {feature}",
        )

        return [step], step.step_id

    def _process_map(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
        input_hashes: Dict[str, str],
        analysis: Dict[str, AnalysisResult],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process MAP node - expand iteration over list.

        MAP applies an operation to each item in a list.
        """
        operation = node.config.get("operation", "TRANSFORM")

        # Get items input
        items_ref = node.config.get("items") or (
            node.inputs[0] if isinstance(node.inputs, list) else
            node.inputs.get("items") if isinstance(node.inputs, dict) else None
        )

        if not items_ref:
            raise ValueError(f"MAP node '{node.id}' has no items input")

        # Resolve items to list of step IDs
        if items_ref in step_id_map:
            # Reference to SOURCE_LIST output
            items_step = step_id_map[items_ref]
            # TODO: expand list items
            logger.warning(f"MAP node '{node.id}' references list step, expansion TBD")
            item_steps = [items_step]
        else:
            item_steps = [items_ref]

        # Generate step for each item
        steps = []
        output_steps = []

        for i, item_step in enumerate(item_steps):
            step_id = f"{node.id}_{i}"

            if operation == "RANDOM_SLICE":
                step = ExecutionStep(
                    step_id=step_id,
                    node_type="SEGMENT",
                    config={
                        "random": True,
                        "seed_from": node.config.get("seed_from"),
                        "index": i,
                    },
                    input_steps=[item_step],
                    name=f"Random slice {i}",
                )
            elif operation == "TRANSFORM":
                step = ExecutionStep(
                    step_id=step_id,
                    node_type="TRANSFORM",
                    config=node.config.get("effects", {}),
                    input_steps=[item_step],
                    name=f"Transform {i}",
                )
            elif operation == "ANALYZE":
                step = ExecutionStep(
                    step_id=step_id,
                    node_type="ANALYZE",
                    config={"feature": node.config.get("feature", "all")},
                    input_steps=[item_step],
                    name=f"Analyze {i}",
                )
            else:
                step = ExecutionStep(
                    step_id=step_id,
                    node_type=operation,
                    config=node.config,
                    input_steps=[item_step],
                    name=f"{operation} {i}",
                )

            steps.append(step)
            output_steps.append(step_id)

        # Create list output
        list_step = ExecutionStep(
            step_id=node.id,
            node_type="_LIST",
            config={"items": output_steps},
            input_steps=output_steps,
            name=f"MAP results",
        )
        steps.append(list_step)

        return steps, list_step.step_id

    def _process_sequence(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process SEQUENCE node.

        Uses tree reduction for parallel composition if enabled.
        """
        # Resolve input steps
        input_steps = []
        for input_id in node.inputs:
            if input_id in step_id_map:
                input_steps.append(step_id_map[input_id])
            else:
                input_steps.append(input_id)

        if len(input_steps) == 0:
            raise ValueError(f"SEQUENCE node '{node.id}' has no inputs")

        if len(input_steps) == 1:
            # Single input, no sequence needed
            return [], input_steps[0]

        transition_config = node.config.get("transition", {"type": "cut"})
        config = {"transition": transition_config}

        if self.use_tree_reduction and len(input_steps) > 2:
            # Use tree reduction
            reduction_steps, output_id = reduce_sequence(
                input_steps,
                transition_config=config,
                id_prefix=node.id,
            )

            steps = []
            for step_id, inputs, step_config in reduction_steps:
                step = ExecutionStep(
                    step_id=step_id,
                    node_type="SEQUENCE",
                    config=step_config,
                    input_steps=inputs,
                    name=f"Sequence reduction",
                )
                steps.append(step)

            return steps, output_id
        else:
            # Direct sequence
            step = ExecutionStep(
                step_id=node.id,
                node_type="SEQUENCE",
                config=config,
                input_steps=input_steps,
                name="Sequence",
            )
            return [step], step.step_id

    def _process_segment_at(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
        analysis: Dict[str, AnalysisResult],
    ) -> Tuple[List[ExecutionStep], str]:
        """
        Process SEGMENT_AT node - cut at specific times.

        Creates SEGMENT steps for each time range.
        """
        times_from = node.config.get("times_from")
        distribute = node.config.get("distribute", "round_robin")

        # TODO: Resolve times from analysis
        # For now, create a placeholder
        step = ExecutionStep(
            step_id=node.id,
            node_type="SEGMENT_AT",
            config=node.config,
            input_steps=[step_id_map.get(i, i) for i in node.inputs],
            name="Segment at times",
        )

        return [step], step.step_id

    def _process_standard(
        self,
        node: RecipeNode,
        step_id_map: Dict[str, str],
    ) -> Tuple[List[ExecutionStep], str]:
        """Process standard transformation/composition node."""
        input_steps = [step_id_map.get(i, i) for i in node.inputs]

        step = ExecutionStep(
            step_id=node.id,
            node_type=node.type,
            config=node.config,
            input_steps=input_steps,
            name=node.type,
        )

        return [step], step.step_id
