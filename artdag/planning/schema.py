# artdag/planning/schema.py
"""
Data structures for execution plans.

An ExecutionPlan contains all steps needed to execute a recipe,
with pre-computed cache IDs for each step.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def _stable_hash(data: Any, algorithm: str = "sha3_256") -> str:
    """Create stable hash from arbitrary data."""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode())
    return hasher.hexdigest()


class StepStatus(Enum):
    """Status of an execution step."""
    PENDING = "pending"
    CLAIMED = "claimed"
    RUNNING = "running"
    COMPLETED = "completed"
    CACHED = "cached"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepOutput:
    """
    A single output from an execution step.

    Nodes may produce multiple outputs (e.g., split_on_beats produces N segments).
    Each output has a human-readable name and a cache_id for storage.

    Attributes:
        name: Human-readable name (e.g., "beats.split.segment[0]")
        cache_id: Content-addressed hash for caching
        media_type: MIME type of the output (e.g., "video/mp4", "audio/wav")
        index: Output index for multi-output nodes
        metadata: Optional additional metadata (time_range, etc.)
    """
    name: str
    cache_id: str
    media_type: str = "application/octet-stream"
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cache_id": self.cache_id,
            "media_type": self.media_type,
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepOutput":
        return cls(
            name=data["name"],
            cache_id=data["cache_id"],
            media_type=data.get("media_type", "application/octet-stream"),
            index=data.get("index", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class StepInput:
    """
    Reference to an input for a step.

    Inputs can reference outputs from other steps by name.

    Attributes:
        name: Input slot name (e.g., "video", "audio", "segments")
        source: Source output name (e.g., "beats.split.segment[0]")
        cache_id: Resolved cache_id of the source (populated during planning)
    """
    name: str
    source: str
    cache_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "cache_id": self.cache_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepInput":
        return cls(
            name=data["name"],
            source=data["source"],
            cache_id=data.get("cache_id"),
        )


@dataclass
class ExecutionStep:
    """
    A single step in the execution plan.

    Each step has a pre-computed cache_id that uniquely identifies
    its output based on its configuration and input cache_ids.

    Steps can produce multiple outputs (e.g., split_on_beats produces N segments).
    Each output has its own cache_id derived from the step's cache_id + index.

    Attributes:
        name: Human-readable name relating to recipe (e.g., "beats.split")
        step_id: Unique identifier (hash) for this step
        node_type: The primitive type (SOURCE, SEQUENCE, TRANSFORM, etc.)
        config: Configuration for the primitive
        input_steps: IDs of steps this depends on (legacy, use inputs for new code)
        inputs: Structured input references with names and sources
        cache_id: Pre-computed cache ID (hash of config + input cache_ids)
        outputs: List of outputs this step produces
        estimated_duration: Optional estimated execution time
        level: Dependency level (0 = no dependencies, higher = more deps)
    """
    step_id: str
    node_type: str
    config: Dict[str, Any]
    input_steps: List[str] = field(default_factory=list)
    inputs: List[StepInput] = field(default_factory=list)
    cache_id: Optional[str] = None
    outputs: List[StepOutput] = field(default_factory=list)
    name: Optional[str] = None
    estimated_duration: Optional[float] = None
    level: int = 0

    def compute_cache_id(self, input_cache_ids: Dict[str, str]) -> str:
        """
        Compute cache ID from configuration and input cache IDs.

        cache_id = SHA3-256(node_type + config + sorted(input_cache_ids))

        Args:
            input_cache_ids: Mapping from input step_id/name to their cache_id

        Returns:
            The computed cache_id
        """
        # Use structured inputs if available, otherwise fall back to input_steps
        if self.inputs:
            resolved_inputs = [
                inp.cache_id or input_cache_ids.get(inp.source, inp.source)
                for inp in sorted(self.inputs, key=lambda x: x.name)
            ]
        else:
            resolved_inputs = [input_cache_ids.get(s, s) for s in sorted(self.input_steps)]

        content = {
            "node_type": self.node_type,
            "config": self.config,
            "inputs": resolved_inputs,
        }
        self.cache_id = _stable_hash(content)
        return self.cache_id

    def compute_output_cache_id(self, index: int) -> str:
        """
        Compute cache ID for a specific output index.

        output_cache_id = SHA3-256(step_cache_id + index)

        Args:
            index: The output index

        Returns:
            Cache ID for that output
        """
        if not self.cache_id:
            raise ValueError("Step cache_id must be computed first")
        content = {"step_cache_id": self.cache_id, "output_index": index}
        return _stable_hash(content)

    def add_output(
        self,
        name: str,
        media_type: str = "application/octet-stream",
        index: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StepOutput:
        """
        Add an output to this step.

        Args:
            name: Human-readable output name
            media_type: MIME type of the output
            index: Output index (defaults to next available)
            metadata: Optional metadata

        Returns:
            The created StepOutput
        """
        if index is None:
            index = len(self.outputs)

        cache_id = self.compute_output_cache_id(index)
        output = StepOutput(
            name=name,
            cache_id=cache_id,
            media_type=media_type,
            index=index,
            metadata=metadata or {},
        )
        self.outputs.append(output)
        return output

    def get_output(self, index: int = 0) -> Optional[StepOutput]:
        """Get output by index."""
        if index < len(self.outputs):
            return self.outputs[index]
        return None

    def get_output_by_name(self, name: str) -> Optional[StepOutput]:
        """Get output by name."""
        for output in self.outputs:
            if output.name == name:
                return output
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "node_type": self.node_type,
            "config": self.config,
            "input_steps": self.input_steps,
            "inputs": [inp.to_dict() for inp in self.inputs],
            "cache_id": self.cache_id,
            "outputs": [out.to_dict() for out in self.outputs],
            "estimated_duration": self.estimated_duration,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionStep":
        inputs = [StepInput.from_dict(i) for i in data.get("inputs", [])]
        outputs = [StepOutput.from_dict(o) for o in data.get("outputs", [])]
        return cls(
            step_id=data["step_id"],
            node_type=data["node_type"],
            config=data.get("config", {}),
            input_steps=data.get("input_steps", []),
            inputs=inputs,
            cache_id=data.get("cache_id"),
            outputs=outputs,
            name=data.get("name"),
            estimated_duration=data.get("estimated_duration"),
            level=data.get("level", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionStep":
        return cls.from_dict(json.loads(json_str))


@dataclass
class PlanInput:
    """
    An input to the execution plan.

    Attributes:
        name: Human-readable name from recipe (e.g., "source_video")
        cache_id: Content hash of the input file
        content_hash: Same as cache_id (for clarity)
        media_type: MIME type of the input
    """
    name: str
    cache_id: str
    content_hash: str
    media_type: str = "application/octet-stream"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cache_id": self.cache_id,
            "content_hash": self.content_hash,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanInput":
        return cls(
            name=data["name"],
            cache_id=data["cache_id"],
            content_hash=data.get("content_hash", data["cache_id"]),
            media_type=data.get("media_type", "application/octet-stream"),
        )


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a recipe.

    Contains all steps in topological order with pre-computed cache IDs.
    The plan is deterministic: same recipe + same inputs = same plan.

    Attributes:
        name: Human-readable plan name from recipe
        plan_id: Hash of the entire plan (for deduplication)
        recipe_id: Source recipe identifier
        recipe_name: Human-readable recipe name
        recipe_hash: Hash of the recipe content
        seed: Random seed used for planning
        steps: List of steps in execution order
        output_step: ID of the final output step
        output_name: Human-readable name of the final output
        inputs: Structured input definitions
        analysis_cache_ids: Cache IDs of analysis results used
        input_hashes: Content hashes of input files (legacy, use inputs)
        created_at: When the plan was generated
        metadata: Optional additional metadata
    """
    plan_id: Optional[str]
    recipe_id: str
    recipe_hash: str
    steps: List[ExecutionStep]
    output_step: str
    name: Optional[str] = None
    recipe_name: Optional[str] = None
    seed: Optional[int] = None
    output_name: Optional[str] = None
    inputs: List[PlanInput] = field(default_factory=list)
    analysis_cache_ids: Dict[str, str] = field(default_factory=dict)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.plan_id is None:
            self.plan_id = self._compute_plan_id()

    def _compute_plan_id(self) -> str:
        """Compute plan ID from contents."""
        content = {
            "recipe_hash": self.recipe_hash,
            "steps": [s.to_dict() for s in self.steps],
            "input_hashes": self.input_hashes,
            "analysis_cache_ids": self.analysis_cache_ids,
        }
        return _stable_hash(content)

    def compute_all_cache_ids(self) -> None:
        """
        Compute cache IDs for all steps in dependency order.

        Must be called after all steps are added to ensure
        cache IDs propagate correctly through dependencies.
        """
        # Build step lookup
        step_by_id = {s.step_id: s for s in self.steps}

        # Cache IDs start with input hashes
        cache_ids = dict(self.input_hashes)

        # Process in order (assumes topological order)
        for step in self.steps:
            # For SOURCE steps referencing inputs, use input hash
            if step.node_type == "SOURCE" and step.config.get("input_ref"):
                ref = step.config["input_ref"]
                if ref in self.input_hashes:
                    step.cache_id = self.input_hashes[ref]
                    cache_ids[step.step_id] = step.cache_id
                    continue

            # For other steps, compute from inputs
            input_cache_ids = {}
            for input_step_id in step.input_steps:
                if input_step_id in cache_ids:
                    input_cache_ids[input_step_id] = cache_ids[input_step_id]
                elif input_step_id in step_by_id:
                    # Step should have been processed already
                    input_cache_ids[input_step_id] = step_by_id[input_step_id].cache_id
                else:
                    raise ValueError(f"Input step {input_step_id} not found for {step.step_id}")

            step.compute_cache_id(input_cache_ids)
            cache_ids[step.step_id] = step.cache_id

        # Recompute plan_id with final cache IDs
        self.plan_id = self._compute_plan_id()

    def compute_levels(self) -> int:
        """
        Compute dependency levels for all steps.

        Level 0 = no dependencies (can start immediately)
        Level N = depends on steps at level N-1

        Returns:
            Maximum level (number of sequential dependency levels)
        """
        step_by_id = {s.step_id: s for s in self.steps}
        levels = {}

        def compute_level(step_id: str) -> int:
            if step_id in levels:
                return levels[step_id]

            step = step_by_id.get(step_id)
            if step is None:
                return 0  # Input from outside the plan

            if not step.input_steps:
                levels[step_id] = 0
                step.level = 0
                return 0

            max_input_level = max(compute_level(s) for s in step.input_steps)
            level = max_input_level + 1
            levels[step_id] = level
            step.level = level
            return level

        for step in self.steps:
            compute_level(step.step_id)

        return max(levels.values()) if levels else 0

    def get_steps_by_level(self) -> Dict[int, List[ExecutionStep]]:
        """
        Group steps by dependency level.

        Steps at the same level can execute in parallel.

        Returns:
            Dict mapping level -> list of steps at that level
        """
        by_level: Dict[int, List[ExecutionStep]] = {}
        for step in self.steps:
            by_level.setdefault(step.level, []).append(step)
        return by_level

    def get_step(self, step_id: str) -> Optional[ExecutionStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_by_cache_id(self, cache_id: str) -> Optional[ExecutionStep]:
        """Get step by cache ID."""
        for step in self.steps:
            if step.cache_id == cache_id:
                return step
        return None

    def get_step_by_name(self, name: str) -> Optional[ExecutionStep]:
        """Get step by human-readable name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_all_outputs(self) -> Dict[str, StepOutput]:
        """
        Get all outputs from all steps, keyed by output name.

        Returns:
            Dict mapping output name -> StepOutput
        """
        outputs = {}
        for step in self.steps:
            for output in step.outputs:
                outputs[output.name] = output
        return outputs

    def get_output_cache_ids(self) -> Dict[str, str]:
        """
        Get mapping of output names to cache IDs.

        Returns:
            Dict mapping output name -> cache_id
        """
        return {
            output.name: output.cache_id
            for step in self.steps
            for output in step.outputs
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "recipe_id": self.recipe_id,
            "recipe_name": self.recipe_name,
            "recipe_hash": self.recipe_hash,
            "seed": self.seed,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "output_step": self.output_step,
            "output_name": self.output_name,
            "analysis_cache_ids": self.analysis_cache_ids,
            "input_hashes": self.input_hashes,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        inputs = [PlanInput.from_dict(i) for i in data.get("inputs", [])]
        return cls(
            plan_id=data.get("plan_id"),
            name=data.get("name"),
            recipe_id=data["recipe_id"],
            recipe_name=data.get("recipe_name"),
            recipe_hash=data["recipe_hash"],
            seed=data.get("seed"),
            inputs=inputs,
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
            output_step=data["output_step"],
            output_name=data.get("output_name"),
            analysis_cache_ids=data.get("analysis_cache_ids", {}),
            input_hashes=data.get("input_hashes", {}),
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionPlan":
        return cls.from_dict(json.loads(json_str))

    def summary(self) -> str:
        """Get a human-readable summary of the plan."""
        by_level = self.get_steps_by_level()
        max_level = max(by_level.keys()) if by_level else 0

        lines = [
            f"Execution Plan: {self.plan_id[:16]}...",
            f"Recipe: {self.recipe_id}",
            f"Steps: {len(self.steps)}",
            f"Levels: {max_level + 1}",
            "",
        ]

        for level in sorted(by_level.keys()):
            steps = by_level[level]
            lines.append(f"Level {level}: ({len(steps)} steps, can run in parallel)")
            for step in steps:
                cache_status = f"[{step.cache_id[:8]}...]" if step.cache_id else "[no cache_id]"
                lines.append(f"  - {step.step_id}: {step.node_type} {cache_status}")

        return "\n".join(lines)
