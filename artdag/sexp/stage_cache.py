"""
Stage-level cache layer using S-expression storage.

Provides caching for stage outputs, enabling:
- Stage-level cache hits (skip entire stage execution)
- Analysis result persistence as sexp
- Cross-worker stage cache sharing (for L1 Celery integration)

All cache files use .sexp extension - no JSON in the pipeline.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .parser import Symbol, Keyword, parse, serialize


@dataclass
class StageOutput:
    """A single output from a stage."""
    cache_id: Optional[str] = None  # For artifacts (files, analysis data)
    value: Any = None  # For scalar values
    output_type: str = "artifact"  # "artifact", "analysis", "scalar"

    def to_sexp(self) -> List:
        """Convert to S-expression."""
        sexp = []
        if self.cache_id:
            sexp.extend([Keyword("cache-id"), self.cache_id])
        if self.value is not None:
            sexp.extend([Keyword("value"), self.value])
        sexp.extend([Keyword("type"), Keyword(self.output_type)])
        return sexp

    @classmethod
    def from_sexp(cls, sexp: List) -> 'StageOutput':
        """Parse from S-expression list."""
        cache_id = None
        value = None
        output_type = "artifact"

        i = 0
        while i < len(sexp):
            if isinstance(sexp[i], Keyword):
                key = sexp[i].name
                if i + 1 < len(sexp):
                    val = sexp[i + 1]
                    if key == "cache-id":
                        cache_id = val
                    elif key == "value":
                        value = val
                    elif key == "type":
                        if isinstance(val, Keyword):
                            output_type = val.name
                        else:
                            output_type = str(val)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        return cls(cache_id=cache_id, value=value, output_type=output_type)


@dataclass
class StageCacheEntry:
    """Cached result of a stage execution."""
    stage_name: str
    cache_id: str
    outputs: Dict[str, StageOutput]  # binding_name -> output info
    completed_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_sexp(self) -> List:
        """
        Convert to S-expression for storage.

        Format:
            (stage-result
              :name "analyze-a"
              :cache-id "abc123..."
              :completed-at 1705678900.123
              :outputs
                ((beats-a :cache-id "def456..." :type :analysis)
                 (tempo :value 120.5 :type :scalar)))
        """
        sexp = [Symbol("stage-result")]
        sexp.extend([Keyword("name"), self.stage_name])
        sexp.extend([Keyword("cache-id"), self.cache_id])
        sexp.extend([Keyword("completed-at"), self.completed_at])

        if self.outputs:
            outputs_sexp = []
            for name, output in self.outputs.items():
                output_sexp = [Symbol(name)] + output.to_sexp()
                outputs_sexp.append(output_sexp)
            sexp.extend([Keyword("outputs"), outputs_sexp])

        if self.metadata:
            sexp.extend([Keyword("metadata"), self.metadata])

        return sexp

    def to_string(self, pretty: bool = True) -> str:
        """Serialize to S-expression string."""
        return serialize(self.to_sexp(), pretty=pretty)

    @classmethod
    def from_sexp(cls, sexp: List) -> 'StageCacheEntry':
        """Parse from S-expression."""
        if not sexp or not isinstance(sexp[0], Symbol) or sexp[0].name != "stage-result":
            raise ValueError("Invalid stage-result sexp")

        stage_name = None
        cache_id = None
        completed_at = time.time()
        outputs = {}
        metadata = {}

        i = 1
        while i < len(sexp):
            if isinstance(sexp[i], Keyword):
                key = sexp[i].name
                if i + 1 < len(sexp):
                    val = sexp[i + 1]
                    if key == "name":
                        stage_name = val
                    elif key == "cache-id":
                        cache_id = val
                    elif key == "completed-at":
                        completed_at = float(val)
                    elif key == "outputs":
                        if isinstance(val, list):
                            for output_sexp in val:
                                if isinstance(output_sexp, list) and output_sexp:
                                    out_name = output_sexp[0]
                                    if isinstance(out_name, Symbol):
                                        out_name = out_name.name
                                    outputs[out_name] = StageOutput.from_sexp(output_sexp[1:])
                    elif key == "metadata":
                        metadata = val if isinstance(val, dict) else {}
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if not stage_name or not cache_id:
            raise ValueError("stage-result missing required fields (name, cache-id)")

        return cls(
            stage_name=stage_name,
            cache_id=cache_id,
            outputs=outputs,
            completed_at=completed_at,
            metadata=metadata,
        )

    @classmethod
    def from_string(cls, text: str) -> 'StageCacheEntry':
        """Parse from S-expression string."""
        sexp = parse(text)
        return cls.from_sexp(sexp)


class StageCache:
    """
    Stage-level cache manager using S-expression files.

    Cache structure:
        cache_dir/
            _stages/
                {cache_id}.sexp   <- Stage result files
    """

    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize stage cache.

        Args:
            cache_dir: Base cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.stages_dir = self.cache_dir / "_stages"
        self.stages_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, cache_id: str) -> Path:
        """Get the path for a stage cache file."""
        return self.stages_dir / f"{cache_id}.sexp"

    def has_stage(self, cache_id: str) -> bool:
        """Check if a stage result is cached."""
        return self.get_cache_path(cache_id).exists()

    def load_stage(self, cache_id: str) -> Optional[StageCacheEntry]:
        """
        Load a cached stage result.

        Args:
            cache_id: Stage cache ID

        Returns:
            StageCacheEntry if found, None otherwise
        """
        path = self.get_cache_path(cache_id)
        if not path.exists():
            return None

        try:
            content = path.read_text()
            return StageCacheEntry.from_string(content)
        except Exception as e:
            # Corrupted cache file - log and return None
            import sys
            print(f"Warning: corrupted stage cache {cache_id}: {e}", file=sys.stderr)
            return None

    def save_stage(self, entry: StageCacheEntry) -> Path:
        """
        Save a stage result to cache.

        Args:
            entry: Stage cache entry to save

        Returns:
            Path to the saved cache file
        """
        path = self.get_cache_path(entry.cache_id)
        content = entry.to_string(pretty=True)
        path.write_text(content)
        return path

    def delete_stage(self, cache_id: str) -> bool:
        """
        Delete a cached stage result.

        Args:
            cache_id: Stage cache ID

        Returns:
            True if deleted, False if not found
        """
        path = self.get_cache_path(cache_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_stages(self) -> List[str]:
        """List all cached stage IDs."""
        return [
            p.stem for p in self.stages_dir.glob("*.sexp")
        ]

    def clear(self) -> int:
        """
        Clear all cached stages.

        Returns:
            Number of entries cleared
        """
        count = 0
        for path in self.stages_dir.glob("*.sexp"):
            path.unlink()
            count += 1
        return count


@dataclass
class AnalysisResult:
    """
    Analysis result stored as S-expression.

    Format:
        (analysis-result
          :analyzer "beats"
          :input-hash "abc123..."
          :duration 120.5
          :tempo 128.0
          :times (0.0 0.468 0.937 1.406 ...)
          :values (0.8 0.9 0.7 0.85 ...))
    """
    analyzer: str
    input_hash: str
    data: Dict[str, Any]  # Analysis data (times, values, duration, etc.)
    computed_at: float = field(default_factory=time.time)

    def to_sexp(self) -> List:
        """Convert to S-expression."""
        sexp = [Symbol("analysis-result")]
        sexp.extend([Keyword("analyzer"), self.analyzer])
        sexp.extend([Keyword("input-hash"), self.input_hash])
        sexp.extend([Keyword("computed-at"), self.computed_at])

        # Add all data fields
        for key, value in self.data.items():
            # Convert key to keyword
            sexp.extend([Keyword(key.replace("_", "-")), value])

        return sexp

    def to_string(self, pretty: bool = True) -> str:
        """Serialize to S-expression string."""
        return serialize(self.to_sexp(), pretty=pretty)

    @classmethod
    def from_sexp(cls, sexp: List) -> 'AnalysisResult':
        """Parse from S-expression."""
        if not sexp or not isinstance(sexp[0], Symbol) or sexp[0].name != "analysis-result":
            raise ValueError("Invalid analysis-result sexp")

        analyzer = None
        input_hash = None
        computed_at = time.time()
        data = {}

        i = 1
        while i < len(sexp):
            if isinstance(sexp[i], Keyword):
                key = sexp[i].name
                if i + 1 < len(sexp):
                    val = sexp[i + 1]
                    if key == "analyzer":
                        analyzer = val
                    elif key == "input-hash":
                        input_hash = val
                    elif key == "computed-at":
                        computed_at = float(val)
                    else:
                        # Convert kebab-case back to snake_case
                        data_key = key.replace("-", "_")
                        data[data_key] = val
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if not analyzer:
            raise ValueError("analysis-result missing analyzer field")

        return cls(
            analyzer=analyzer,
            input_hash=input_hash or "",
            data=data,
            computed_at=computed_at,
        )

    @classmethod
    def from_string(cls, text: str) -> 'AnalysisResult':
        """Parse from S-expression string."""
        sexp = parse(text)
        return cls.from_sexp(sexp)


def save_analysis_result(
    cache_dir: Union[str, Path],
    node_id: str,
    result: AnalysisResult,
) -> Path:
    """
    Save an analysis result as S-expression.

    Args:
        cache_dir: Base cache directory
        node_id: Node ID for the analysis
        result: Analysis result to save

    Returns:
        Path to the saved file
    """
    cache_dir = Path(cache_dir)
    node_dir = cache_dir / node_id
    node_dir.mkdir(parents=True, exist_ok=True)

    path = node_dir / "analysis.sexp"
    content = result.to_string(pretty=True)
    path.write_text(content)
    return path


def load_analysis_result(
    cache_dir: Union[str, Path],
    node_id: str,
) -> Optional[AnalysisResult]:
    """
    Load an analysis result from cache.

    Args:
        cache_dir: Base cache directory
        node_id: Node ID for the analysis

    Returns:
        AnalysisResult if found, None otherwise
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / node_id / "analysis.sexp"

    if not path.exists():
        return None

    try:
        content = path.read_text()
        return AnalysisResult.from_string(content)
    except Exception as e:
        import sys
        print(f"Warning: corrupted analysis cache {node_id}: {e}", file=sys.stderr)
        return None
