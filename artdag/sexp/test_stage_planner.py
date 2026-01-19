"""
Tests for stage-aware planning.

Tests stage topological sorting, level computation, cache ID computation,
and plan metadata generation.
"""

import pytest
from pathlib import Path

from .parser import parse
from .compiler import compile_recipe, CompiledStage
from .planner import (
    create_plan,
    StagePlan,
    _compute_stage_levels,
    _compute_stage_cache_id,
)


class TestStagePlanning:
    """Test stage-aware plan creation."""

    def test_stage_topological_sort_in_plan(self):
        """Stages sorted by dependencies in plan."""
        recipe = '''
        (recipe "test-sort"
          (def audio (source :path "test.mp3"))

          (stage :analyze
            :outputs [beats]
            (def beats (-> audio (analyze beats))))

          (stage :output
            :requires [:analyze]
            :inputs [beats]
            (-> audio (segment :times beats) (sequence))))
        '''
        compiled = compile_recipe(parse(recipe))
        # Note: create_plan needs recipe_dir for analysis, we'll test the ordering differently
        assert compiled.stage_order.index("analyze") < compiled.stage_order.index("output")

    def test_stage_level_computation(self):
        """Independent stages get same level."""
        stages = [
            CompiledStage(name="a", requires=[], inputs=[], outputs=["x"],
                         node_ids=["n1"], output_bindings={"x": "n1"}),
            CompiledStage(name="b", requires=[], inputs=[], outputs=["y"],
                         node_ids=["n2"], output_bindings={"y": "n2"}),
            CompiledStage(name="c", requires=["a", "b"], inputs=["x", "y"], outputs=["z"],
                         node_ids=["n3"], output_bindings={"z": "n3"}),
        ]
        levels = _compute_stage_levels(stages)

        assert levels["a"] == 0
        assert levels["b"] == 0
        assert levels["c"] == 1  # Depends on a and b

    def test_stage_level_chain(self):
        """Chain stages get increasing levels."""
        stages = [
            CompiledStage(name="a", requires=[], inputs=[], outputs=["x"],
                         node_ids=["n1"], output_bindings={"x": "n1"}),
            CompiledStage(name="b", requires=["a"], inputs=["x"], outputs=["y"],
                         node_ids=["n2"], output_bindings={"y": "n2"}),
            CompiledStage(name="c", requires=["b"], inputs=["y"], outputs=["z"],
                         node_ids=["n3"], output_bindings={"z": "n3"}),
        ]
        levels = _compute_stage_levels(stages)

        assert levels["a"] == 0
        assert levels["b"] == 1
        assert levels["c"] == 2

    def test_stage_cache_id_deterministic(self):
        """Same stage = same cache ID."""
        stage = CompiledStage(
            name="analyze",
            requires=[],
            inputs=[],
            outputs=["beats"],
            node_ids=["abc123"],
            output_bindings={"beats": "abc123"},
        )

        cache_id_1 = _compute_stage_cache_id(
            stage,
            stage_cache_ids={},
            node_cache_ids={"abc123": "nodeabc"},
            cluster_key=None,
        )
        cache_id_2 = _compute_stage_cache_id(
            stage,
            stage_cache_ids={},
            node_cache_ids={"abc123": "nodeabc"},
            cluster_key=None,
        )

        assert cache_id_1 == cache_id_2

    def test_stage_cache_id_includes_requires(self):
        """Cache ID changes when required stage cache ID changes."""
        stage = CompiledStage(
            name="process",
            requires=["analyze"],
            inputs=["beats"],
            outputs=["result"],
            node_ids=["def456"],
            output_bindings={"result": "def456"},
        )

        cache_id_1 = _compute_stage_cache_id(
            stage,
            stage_cache_ids={"analyze": "req_cache_a"},
            node_cache_ids={"def456": "node_def"},
            cluster_key=None,
        )
        cache_id_2 = _compute_stage_cache_id(
            stage,
            stage_cache_ids={"analyze": "req_cache_b"},
            node_cache_ids={"def456": "node_def"},
            cluster_key=None,
        )

        # Different required stage cache IDs should produce different cache IDs
        assert cache_id_1 != cache_id_2

    def test_stage_cache_id_cluster_key(self):
        """Cache ID changes with cluster key."""
        stage = CompiledStage(
            name="analyze",
            requires=[],
            inputs=[],
            outputs=["beats"],
            node_ids=["abc123"],
            output_bindings={"beats": "abc123"},
        )

        cache_id_no_key = _compute_stage_cache_id(
            stage,
            stage_cache_ids={},
            node_cache_ids={"abc123": "nodeabc"},
            cluster_key=None,
        )
        cache_id_with_key = _compute_stage_cache_id(
            stage,
            stage_cache_ids={},
            node_cache_ids={"abc123": "nodeabc"},
            cluster_key="cluster123",
        )

        # Cluster key should change the cache ID
        assert cache_id_no_key != cache_id_with_key


class TestStagePlanMetadata:
    """Test stage metadata in execution plans."""

    def test_plan_without_stages(self):
        """Plan without stages has empty stage fields."""
        recipe = '''
        (recipe "no-stages"
          (-> (source :path "test.mp3") (effect gain :amount 0.5)))
        '''
        compiled = compile_recipe(parse(recipe))
        assert compiled.stages == []
        assert compiled.stage_order == []


class TestStagePlanDataclass:
    """Test StagePlan dataclass."""

    def test_stage_plan_creation(self):
        """StagePlan can be created with all fields."""
        from .planner import PlanStep

        step = PlanStep(
            step_id="step1",
            node_type="ANALYZE",
            config={"analyzer": "beats"},
            inputs=["input1"],
            cache_id="cache123",
            level=0,
            stage="analyze",
            stage_cache_id="stage_cache_123",
        )

        stage_plan = StagePlan(
            stage_name="analyze",
            cache_id="stage_cache_123",
            steps=[step],
            requires=[],
            output_bindings={"beats": "cache123"},
            level=0,
        )

        assert stage_plan.stage_name == "analyze"
        assert stage_plan.cache_id == "stage_cache_123"
        assert len(stage_plan.steps) == 1
        assert stage_plan.level == 0


class TestExplicitDataRouting:
    """Test that plan includes explicit data routing."""

    def test_plan_step_includes_stage_info(self):
        """PlanStep includes stage and stage_cache_id."""
        from .planner import PlanStep

        step = PlanStep(
            step_id="step1",
            node_type="ANALYZE",
            config={},
            inputs=[],
            cache_id="cache123",
            level=0,
            stage="analyze",
            stage_cache_id="stage_cache_abc",
        )

        sexp = step.to_sexp()
        # Convert to string to check for stage info
        from .parser import serialize
        sexp_str = serialize(sexp)

        assert "stage" in sexp_str
        assert "analyze" in sexp_str
        assert "stage-cache-id" in sexp_str
