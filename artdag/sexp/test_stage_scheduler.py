"""
Tests for stage-aware scheduler.

Tests stage cache hit/miss, stage execution ordering,
and parallel stage support.
"""

import pytest
import tempfile
from unittest.mock import Mock, MagicMock, patch

from .scheduler import (
    StagePlanScheduler,
    StageResult,
    StagePlanResult,
    create_stage_scheduler,
    schedule_staged_plan,
)
from .planner import ExecutionPlanSexp, PlanStep, StagePlan
from .stage_cache import StageCache, StageCacheEntry, StageOutput


class TestStagePlanScheduler:
    """Test stage-aware scheduling."""

    def test_plan_without_stages_uses_regular_scheduling(self):
        """Plans without stages fall back to regular scheduling."""
        plan = ExecutionPlanSexp(
            plan_id="test_plan",
            recipe_id="test_recipe",
            recipe_hash="abc123",
            steps=[],
            output_step_id="output",
            stage_plans=[],  # No stages
        )

        scheduler = StagePlanScheduler()
        # This will use PlanScheduler internally
        # Without Celery, it just returns completed status
        result = scheduler.schedule(plan)

        assert isinstance(result, StagePlanResult)

    def test_stage_cache_hit_skips_execution(self):
        """Cached stage not re-executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_cache = StageCache(tmpdir)

            # Pre-populate cache
            entry = StageCacheEntry(
                stage_name="analyze",
                cache_id="stage_cache_123",
                outputs={"beats": StageOutput(cache_id="beats_out", output_type="analysis")},
            )
            stage_cache.save_stage(entry)

            step = PlanStep(
                step_id="step1",
                node_type="ANALYZE",
                config={},
                inputs=[],
                cache_id="step_cache",
                level=0,
                stage="analyze",
                stage_cache_id="stage_cache_123",
            )

            stage_plan = StagePlan(
                stage_name="analyze",
                cache_id="stage_cache_123",
                steps=[step],
                requires=[],
                output_bindings={"beats": "beats_out"},
                level=0,
            )

            plan = ExecutionPlanSexp(
                plan_id="test_plan",
                recipe_id="test_recipe",
                recipe_hash="abc123",
                steps=[step],
                output_step_id="step1",
                stage_plans=[stage_plan],
                stage_order=["analyze"],
                stage_levels={"analyze": 0},
                stage_cache_ids={"analyze": "stage_cache_123"},
            )

            scheduler = StagePlanScheduler(stage_cache=stage_cache)
            result = scheduler.schedule(plan)

            assert result.stages_cached == 1
            assert result.stages_completed == 0

    def test_stage_inputs_loaded_from_cache(self):
        """Stage receives inputs from required stage cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_cache = StageCache(tmpdir)

            # Pre-populate upstream stage cache
            upstream_entry = StageCacheEntry(
                stage_name="analyze",
                cache_id="upstream_cache",
                outputs={"beats": StageOutput(cache_id="beats_data", output_type="analysis")},
            )
            stage_cache.save_stage(upstream_entry)

            # Steps for stages
            upstream_step = PlanStep(
                step_id="analyze_step",
                node_type="ANALYZE",
                config={},
                inputs=[],
                cache_id="analyze_cache",
                level=0,
                stage="analyze",
                stage_cache_id="upstream_cache",
            )

            downstream_step = PlanStep(
                step_id="process_step",
                node_type="SEGMENT",
                config={},
                inputs=["analyze_step"],
                cache_id="process_cache",
                level=1,
                stage="process",
                stage_cache_id="downstream_cache",
            )

            upstream_plan = StagePlan(
                stage_name="analyze",
                cache_id="upstream_cache",
                steps=[upstream_step],
                requires=[],
                output_bindings={"beats": "beats_data"},
                level=0,
            )

            downstream_plan = StagePlan(
                stage_name="process",
                cache_id="downstream_cache",
                steps=[downstream_step],
                requires=["analyze"],
                output_bindings={"result": "process_cache"},
                level=1,
            )

            plan = ExecutionPlanSexp(
                plan_id="test_plan",
                recipe_id="test_recipe",
                recipe_hash="abc123",
                steps=[upstream_step, downstream_step],
                output_step_id="process_step",
                stage_plans=[upstream_plan, downstream_plan],
                stage_order=["analyze", "process"],
                stage_levels={"analyze": 0, "process": 1},
                stage_cache_ids={"analyze": "upstream_cache", "process": "downstream_cache"},
            )

            scheduler = StagePlanScheduler(stage_cache=stage_cache)
            result = scheduler.schedule(plan)

            # Upstream should be cached, downstream executed
            assert result.stages_cached == 1
            assert "analyze" in result.stage_results
            assert result.stage_results["analyze"].status == "cached"

    def test_parallel_stages_same_level(self):
        """Stages at same level can run in parallel."""
        step_a = PlanStep(
            step_id="step_a",
            node_type="ANALYZE",
            config={},
            inputs=[],
            cache_id="cache_a",
            level=0,
            stage="analyze-a",
            stage_cache_id="stage_a",
        )

        step_b = PlanStep(
            step_id="step_b",
            node_type="ANALYZE",
            config={},
            inputs=[],
            cache_id="cache_b",
            level=0,
            stage="analyze-b",
            stage_cache_id="stage_b",
        )

        stage_a = StagePlan(
            stage_name="analyze-a",
            cache_id="stage_a",
            steps=[step_a],
            requires=[],
            output_bindings={"beats-a": "cache_a"},
            level=0,
        )

        stage_b = StagePlan(
            stage_name="analyze-b",
            cache_id="stage_b",
            steps=[step_b],
            requires=[],
            output_bindings={"beats-b": "cache_b"},
            level=0,
        )

        plan = ExecutionPlanSexp(
            plan_id="test_plan",
            recipe_id="test_recipe",
            recipe_hash="abc123",
            steps=[step_a, step_b],
            output_step_id="step_b",
            stage_plans=[stage_a, stage_b],
            stage_order=["analyze-a", "analyze-b"],
            stage_levels={"analyze-a": 0, "analyze-b": 0},
            stage_cache_ids={"analyze-a": "stage_a", "analyze-b": "stage_b"},
        )

        scheduler = StagePlanScheduler()
        # Group stages by level
        stages_by_level = scheduler._group_stages_by_level(plan.stage_plans)

        # Both stages should be at level 0
        assert len(stages_by_level[0]) == 2

    def test_stage_outputs_cached_after_execution(self):
        """Stage outputs written to cache after completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_cache = StageCache(tmpdir)

            step = PlanStep(
                step_id="step1",
                node_type="ANALYZE",
                config={},
                inputs=[],
                cache_id="step_cache",
                level=0,
                stage="analyze",
                stage_cache_id="new_stage_cache",
            )

            stage_plan = StagePlan(
                stage_name="analyze",
                cache_id="new_stage_cache",
                steps=[step],
                requires=[],
                output_bindings={"beats": "step_cache"},
                level=0,
            )

            plan = ExecutionPlanSexp(
                plan_id="test_plan",
                recipe_id="test_recipe",
                recipe_hash="abc123",
                steps=[step],
                output_step_id="step1",
                stage_plans=[stage_plan],
                stage_order=["analyze"],
                stage_levels={"analyze": 0},
                stage_cache_ids={"analyze": "new_stage_cache"},
            )

            scheduler = StagePlanScheduler(stage_cache=stage_cache)
            result = scheduler.schedule(plan)

            # Stage should now be cached
            assert stage_cache.has_stage("new_stage_cache")


class TestStageResult:
    """Test StageResult dataclass."""

    def test_stage_result_creation(self):
        """StageResult can be created with all fields."""
        result = StageResult(
            stage_name="test",
            cache_id="cache123",
            status="completed",
            step_results={},
            outputs={"out": "out_cache"},
        )

        assert result.stage_name == "test"
        assert result.status == "completed"
        assert result.outputs["out"] == "out_cache"


class TestStagePlanResult:
    """Test StagePlanResult dataclass."""

    def test_stage_plan_result_creation(self):
        """StagePlanResult can be created with all fields."""
        result = StagePlanResult(
            plan_id="plan123",
            status="completed",
            stages_completed=2,
            stages_cached=1,
            stages_failed=0,
        )

        assert result.plan_id == "plan123"
        assert result.stages_completed == 2
        assert result.stages_cached == 1


class TestSchedulerFactory:
    """Test scheduler factory functions."""

    def test_create_stage_scheduler(self):
        """create_stage_scheduler returns StagePlanScheduler."""
        scheduler = create_stage_scheduler()
        assert isinstance(scheduler, StagePlanScheduler)

    def test_create_stage_scheduler_with_cache(self):
        """create_stage_scheduler accepts stage_cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stage_cache = StageCache(tmpdir)
            scheduler = create_stage_scheduler(stage_cache=stage_cache)
            assert scheduler.stage_cache is stage_cache
