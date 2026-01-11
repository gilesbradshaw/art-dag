"""
S-expression parsing, compilation, and planning for ArtDAG.

This module provides:
- parser: Parse S-expression text into Python data structures
- compiler: Compile recipe S-expressions into DAG format
- planner: Generate execution plans from recipes
"""

from .parser import (
    parse,
    parse_all,
    serialize,
    Symbol,
    Keyword,
    ParseError,
)

from .compiler import (
    compile_recipe,
    compile_string,
    CompiledRecipe,
    CompileError,
)

from .planner import (
    create_plan,
    ExecutionPlanSexp,
    PlanStep,
    step_to_task_sexp,
    task_cache_id,
)

from .scheduler import (
    PlanScheduler,
    PlanResult,
    StepResult,
    schedule_plan,
    step_to_sexp,
    step_sexp_to_string,
    verify_step_cache_id,
)

__all__ = [
    # Parser
    'parse',
    'parse_all',
    'serialize',
    'Symbol',
    'Keyword',
    'ParseError',
    # Compiler
    'compile_recipe',
    'compile_string',
    'CompiledRecipe',
    'CompileError',
    # Planner
    'create_plan',
    'ExecutionPlanSexp',
    'PlanStep',
    'step_to_task_sexp',
    'task_cache_id',
    # Scheduler
    'PlanScheduler',
    'PlanResult',
    'StepResult',
    'schedule_plan',
    'step_to_sexp',
    'step_sexp_to_string',
    'verify_step_cache_id',
]
