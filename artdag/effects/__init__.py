"""
Cacheable effect system.

Effects are single Python files with:
- PEP 723 embedded dependencies
- @-tag metadata in docstrings
- Frame-by-frame or whole-video API

Effects are cached by content hash (SHA3-256) and executed in
sandboxed environments for determinism.
"""

from .meta import EffectMeta, ParamSpec, ExecutionContext
from .loader import load_effect, load_effect_file, LoadedEffect, compute_cid
from .binding import (
    AnalysisData,
    ResolvedBinding,
    resolve_binding,
    resolve_all_bindings,
    bindings_to_lookup_table,
    has_bindings,
    extract_binding_sources,
)
from .sandbox import Sandbox, SandboxConfig, SandboxResult, is_bwrap_available, get_venv_path
from .runner import run_effect, run_effect_from_cache, EffectExecutor

__all__ = [
    # Meta types
    "EffectMeta",
    "ParamSpec",
    "ExecutionContext",
    # Loader
    "load_effect",
    "load_effect_file",
    "LoadedEffect",
    "compute_cid",
    # Binding
    "AnalysisData",
    "ResolvedBinding",
    "resolve_binding",
    "resolve_all_bindings",
    "bindings_to_lookup_table",
    "has_bindings",
    "extract_binding_sources",
    # Sandbox
    "Sandbox",
    "SandboxConfig",
    "SandboxResult",
    "is_bwrap_available",
    "get_venv_path",
    # Runner
    "run_effect",
    "run_effect_from_cache",
    "EffectExecutor",
]
