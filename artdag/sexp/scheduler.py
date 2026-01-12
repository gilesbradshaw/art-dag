"""
Celery scheduler for S-expression execution plans.

Distributes plan steps to workers as S-expressions.
The S-expression is the canonical format - workers receive
serialized S-expressions and can verify cache_ids by hashing them.

Usage:
    from artdag.sexp import compile_string, create_plan
    from artdag.sexp.scheduler import schedule_plan

    recipe = compile_string(sexp_content)
    plan = create_plan(recipe, inputs={'video': 'abc123...'})
    result = schedule_plan(plan)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .parser import Symbol, Keyword, serialize, parse
from .planner import ExecutionPlanSexp, PlanStep

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from executing a step."""
    step_id: str
    cache_id: str
    status: str  # 'completed', 'cached', 'failed', 'pending'
    output_path: Optional[str] = None
    error: Optional[str] = None
    ipfs_cid: Optional[str] = None


@dataclass
class PlanResult:
    """Result from executing a complete plan."""
    plan_id: str
    status: str  # 'completed', 'failed', 'partial'
    steps_completed: int = 0
    steps_cached: int = 0
    steps_failed: int = 0
    output_cache_id: Optional[str] = None
    output_path: Optional[str] = None
    output_ipfs_cid: Optional[str] = None
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    error: Optional[str] = None


def step_to_sexp(step: PlanStep) -> List:
    """
    Convert a PlanStep to minimal S-expression for worker.

    This is the canonical form that workers receive.
    Workers can verify cache_id by hashing this S-expression.
    """
    sexp = [Symbol(step.node_type.lower())]

    # Add config as keywords
    for key, value in step.config.items():
        sexp.extend([Keyword(key.replace('_', '-')), value])

    # Add inputs as cache IDs (not step IDs)
    if step.inputs:
        sexp.extend([Keyword("inputs"), step.inputs])

    return sexp


def step_sexp_to_string(step: PlanStep) -> str:
    """Serialize step to S-expression string for Celery task."""
    return serialize(step_to_sexp(step))


def verify_step_cache_id(step_sexp: str, expected_cache_id: str, cluster_key: str = None) -> bool:
    """
    Verify that a step's cache_id matches its S-expression.

    Workers should call this to verify they're executing the correct task.
    """
    data = {"sexp": step_sexp}
    if cluster_key:
        data = {"_cluster_key": cluster_key, "_data": data}

    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    computed = hashlib.sha3_256(json_str.encode()).hexdigest()
    return computed == expected_cache_id


class PlanScheduler:
    """
    Schedules execution of S-expression plans on Celery workers.

    The scheduler:
    1. Groups steps by dependency level
    2. Checks cache for already-computed results
    3. Dispatches uncached steps to workers
    4. Waits for completion before proceeding to next level
    """

    def __init__(
        self,
        cache_manager=None,
        celery_app=None,
        execute_task_name: str = 'tasks.execute_step_sexp',
    ):
        """
        Initialize the scheduler.

        Args:
            cache_manager: L1 cache manager for checking cached results
            celery_app: Celery application instance
            execute_task_name: Name of the Celery task for step execution
        """
        self.cache_manager = cache_manager
        self.celery_app = celery_app
        self.execute_task_name = execute_task_name

    def schedule(
        self,
        plan: ExecutionPlanSexp,
        timeout: int = 3600,
    ) -> PlanResult:
        """
        Schedule and execute a plan.

        Args:
            plan: The execution plan (S-expression format)
            timeout: Timeout in seconds for the entire plan

        Returns:
            PlanResult with execution results
        """
        from celery import group

        logger.info(f"Scheduling plan {plan.plan_id[:16]}... ({len(plan.steps)} steps)")

        # Build step lookup and group by level
        steps_by_id = {s.step_id: s for s in plan.steps}
        steps_by_level = self._group_by_level(plan.steps)
        max_level = max(steps_by_level.keys()) if steps_by_level else 0

        # Track results
        result = PlanResult(
            plan_id=plan.plan_id,
            status="pending",
        )

        # Map step_id -> cache_id for resolving inputs
        cache_ids = dict(plan.inputs)  # Start with input hashes
        for step in plan.steps:
            cache_ids[step.step_id] = step.cache_id

        # Execute level by level
        for level in range(max_level + 1):
            level_steps = steps_by_level.get(level, [])
            if not level_steps:
                continue

            logger.info(f"Level {level}: {len(level_steps)} steps")

            # Check cache for each step
            steps_to_run = []
            for step in level_steps:
                if self._is_cached(step.cache_id):
                    result.steps_cached += 1
                    result.step_results[step.step_id] = StepResult(
                        step_id=step.step_id,
                        cache_id=step.cache_id,
                        status="cached",
                        output_path=self._get_cached_path(step.cache_id),
                    )
                else:
                    steps_to_run.append(step)

            if not steps_to_run:
                logger.info(f"Level {level}: all {len(level_steps)} steps cached")
                continue

            # Dispatch uncached steps to workers
            logger.info(f"Level {level}: dispatching {len(steps_to_run)} steps")

            tasks = []
            for step in steps_to_run:
                # Build task arguments
                step_sexp = step_sexp_to_string(step)
                input_cache_ids = {
                    inp: cache_ids.get(inp, inp)
                    for inp in step.inputs
                }

                task = self._get_execute_task().s(
                    step_sexp=step_sexp,
                    step_id=step.step_id,
                    cache_id=step.cache_id,
                    plan_id=plan.plan_id,
                    input_cache_ids=input_cache_ids,
                )
                tasks.append(task)

            # Execute in parallel
            job = group(tasks)
            async_result = job.apply_async()

            try:
                step_results = async_result.get(timeout=timeout)
            except Exception as e:
                logger.error(f"Level {level} failed: {e}")
                result.status = "failed"
                result.error = f"Level {level} failed: {e}"
                return result

            # Process results
            for step_result in step_results:
                step_id = step_result.get("step_id")
                status = step_result.get("status")

                result.step_results[step_id] = StepResult(
                    step_id=step_id,
                    cache_id=step_result.get("cache_id"),
                    status=status,
                    output_path=step_result.get("output_path"),
                    error=step_result.get("error"),
                    ipfs_cid=step_result.get("ipfs_cid"),
                )

                if status in ("completed", "cached", "completed_by_other"):
                    result.steps_completed += 1
                elif status == "failed":
                    result.steps_failed += 1
                    result.status = "failed"
                    result.error = step_result.get("error")
                    return result

        # Get final output
        output_step = steps_by_id.get(plan.output_step_id)
        if output_step:
            output_result = result.step_results.get(output_step.step_id)
            if output_result:
                result.output_cache_id = output_step.cache_id
                result.output_path = output_result.output_path
                result.output_ipfs_cid = output_result.ipfs_cid

        result.status = "completed"
        logger.info(
            f"Plan {plan.plan_id[:16]}... completed: "
            f"{result.steps_completed} executed, {result.steps_cached} cached"
        )
        return result

    def _group_by_level(self, steps: List[PlanStep]) -> Dict[int, List[PlanStep]]:
        """Group steps by dependency level."""
        by_level = {}
        for step in steps:
            by_level.setdefault(step.level, []).append(step)
        return by_level

    def _is_cached(self, cache_id: str) -> bool:
        """Check if a cache_id exists in cache."""
        if self.cache_manager is None:
            return False
        path = self.cache_manager.get_by_cid(cache_id)
        return path is not None

    def _get_cached_path(self, cache_id: str) -> Optional[str]:
        """Get the path for a cached item."""
        if self.cache_manager is None:
            return None
        path = self.cache_manager.get_by_cid(cache_id)
        return str(path) if path else None

    def _get_execute_task(self):
        """Get the Celery task for step execution."""
        if self.celery_app is None:
            raise RuntimeError("Celery app not configured")
        return self.celery_app.tasks[self.execute_task_name]


def create_scheduler(cache_manager=None, celery_app=None) -> PlanScheduler:
    """
    Create a scheduler with the given dependencies.

    If not provided, attempts to import from art-celery.
    """
    if celery_app is None:
        try:
            from celery_app import app as celery_app
        except ImportError:
            pass

    if cache_manager is None:
        try:
            from cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
        except ImportError:
            pass

    return PlanScheduler(
        cache_manager=cache_manager,
        celery_app=celery_app,
    )


def schedule_plan(
    plan: ExecutionPlanSexp,
    cache_manager=None,
    celery_app=None,
    timeout: int = 3600,
) -> PlanResult:
    """
    Convenience function to schedule a plan.

    Args:
        plan: The execution plan
        cache_manager: Optional cache manager
        celery_app: Optional Celery app
        timeout: Execution timeout

    Returns:
        PlanResult
    """
    scheduler = create_scheduler(cache_manager, celery_app)
    return scheduler.schedule(plan, timeout=timeout)
