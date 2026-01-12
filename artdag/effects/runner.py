"""
Effect runner.

Main entry point for executing cached effects with sandboxing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .binding import AnalysisData, bindings_to_lookup_table, resolve_all_bindings
from .loader import load_effect, LoadedEffect
from .meta import ExecutionContext
from .sandbox import Sandbox, SandboxConfig, SandboxResult, get_venv_path

logger = logging.getLogger(__name__)


def run_effect(
    effect_source: str,
    input_paths: List[Path],
    output_path: Path,
    params: Dict[str, Any],
    analysis: Optional[AnalysisData] = None,
    cache_id: str = None,
    seed: int = 0,
    trust_level: str = "untrusted",
    timeout: int = 3600,
) -> SandboxResult:
    """
    Run an effect with full sandboxing.

    This is the main entry point for effect execution.

    Args:
        effect_source: Effect source code
        input_paths: List of input file paths
        output_path: Output file path
        params: Effect parameters (may contain bindings)
        analysis: Optional analysis data for binding resolution
        cache_id: Cache ID for deterministic seeding
        seed: RNG seed (overrides cache_id-based seed)
        trust_level: "untrusted" or "trusted"
        timeout: Maximum execution time in seconds

    Returns:
        SandboxResult with success status and output
    """
    # Load and validate effect
    loaded = load_effect(effect_source)
    logger.info(f"Running effect '{loaded.meta.name}' v{loaded.meta.version}")

    # Resolve bindings if analysis data available
    bindings = {}
    if analysis:
        resolved = resolve_all_bindings(params, analysis, cache_id)
        bindings = bindings_to_lookup_table(resolved)
        # Remove binding dicts from params, keeping only resolved values
        params = {
            k: v for k, v in params.items()
            if not (isinstance(v, dict) and v.get("_binding"))
        }

    # Validate parameters
    validated_params = loaded.meta.validate_params(params)

    # Get or create venv for dependencies
    venv_path = None
    if loaded.dependencies:
        venv_path = get_venv_path(loaded.dependencies)

    # Configure sandbox
    config = SandboxConfig(
        trust_level=trust_level,
        venv_path=venv_path,
        timeout=timeout,
    )

    # Write effect to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as f:
        f.write(effect_source)
        effect_path = Path(f.name)

    try:
        with Sandbox(config) as sandbox:
            result = sandbox.run_effect(
                effect_path=effect_path,
                input_paths=input_paths,
                output_path=output_path,
                params=validated_params,
                bindings=bindings,
                seed=seed,
            )
    finally:
        effect_path.unlink(missing_ok=True)

    return result


def run_effect_from_cache(
    cache,
    effect_hash: str,
    input_paths: List[Path],
    output_path: Path,
    params: Dict[str, Any],
    analysis: Optional[AnalysisData] = None,
    cache_id: str = None,
    seed: int = 0,
    trust_level: str = "untrusted",
    timeout: int = 3600,
) -> SandboxResult:
    """
    Run an effect from cache by content hash.

    Args:
        cache: Cache instance
        effect_hash: Content hash of effect
        input_paths: Input file paths
        output_path: Output file path
        params: Effect parameters
        analysis: Optional analysis data
        cache_id: Cache ID for seeding
        seed: RNG seed
        trust_level: "untrusted" or "trusted"
        timeout: Max execution time

    Returns:
        SandboxResult
    """
    effect_source = cache.get_effect(effect_hash)
    if not effect_source:
        return SandboxResult(
            success=False,
            error=f"Effect not found in cache: {effect_hash[:16]}...",
        )

    return run_effect(
        effect_source=effect_source,
        input_paths=input_paths,
        output_path=output_path,
        params=params,
        analysis=analysis,
        cache_id=cache_id,
        seed=seed,
        trust_level=trust_level,
        timeout=timeout,
    )


def check_effect_temporal(cache, effect_hash: str) -> bool:
    """
    Check if an effect is temporal (can't be collapsed).

    Args:
        cache: Cache instance
        effect_hash: Content hash of effect

    Returns:
        True if effect is temporal
    """
    metadata = cache.get_effect_metadata(effect_hash)
    if not metadata:
        return False

    meta = metadata.get("meta", {})
    return meta.get("temporal", False)


def get_effect_api_type(cache, effect_hash: str) -> str:
    """
    Get the API type of an effect.

    Args:
        cache: Cache instance
        effect_hash: Content hash of effect

    Returns:
        "frame" or "video"
    """
    metadata = cache.get_effect_metadata(effect_hash)
    if not metadata:
        return "frame"

    meta = metadata.get("meta", {})
    return meta.get("api_type", "frame")


class EffectExecutor:
    """
    Executor for cached effects.

    Provides a higher-level interface for effect execution.
    """

    def __init__(self, cache, trust_level: str = "untrusted"):
        """
        Initialize executor.

        Args:
            cache: Cache instance
            trust_level: Default trust level
        """
        self.cache = cache
        self.trust_level = trust_level

    def execute(
        self,
        effect_hash: str,
        input_paths: List[Path],
        output_path: Path,
        params: Dict[str, Any],
        analysis: Optional[AnalysisData] = None,
        step_cache_id: str = None,
    ) -> SandboxResult:
        """
        Execute an effect.

        Args:
            effect_hash: Content hash of effect
            input_paths: Input file paths
            output_path: Output path
            params: Effect parameters
            analysis: Analysis data for bindings
            step_cache_id: Step cache ID for seeding

        Returns:
            SandboxResult
        """
        # Check effect metadata for trust level override
        metadata = self.cache.get_effect_metadata(effect_hash)
        trust_level = self.trust_level
        if metadata:
            # L1 owner can mark effect as trusted
            if metadata.get("trust_level") == "trusted":
                trust_level = "trusted"

        return run_effect_from_cache(
            cache=self.cache,
            effect_hash=effect_hash,
            input_paths=input_paths,
            output_path=output_path,
            params=params,
            analysis=analysis,
            cache_id=step_cache_id,
            trust_level=trust_level,
        )

    def is_temporal(self, effect_hash: str) -> bool:
        """Check if effect is temporal."""
        return check_effect_temporal(self.cache, effect_hash)

    def get_api_type(self, effect_hash: str) -> str:
        """Get effect API type."""
        return get_effect_api_type(self.cache, effect_hash)
