"""
Tests for stage cache layer.

Tests S-expression storage for stage results and analysis data.
"""

import pytest
import tempfile
from pathlib import Path

from .stage_cache import (
    StageCache,
    StageCacheEntry,
    StageOutput,
    AnalysisResult,
    save_analysis_result,
    load_analysis_result,
)
from .parser import parse, serialize


class TestStageOutput:
    """Test StageOutput dataclass and serialization."""

    def test_stage_output_artifact(self):
        """StageOutput can represent an artifact."""
        output = StageOutput(
            cache_id="abc123",
            output_type="artifact",
        )
        assert output.cache_id == "abc123"
        assert output.output_type == "artifact"

    def test_stage_output_scalar(self):
        """StageOutput can represent a scalar value."""
        output = StageOutput(
            value=120.5,
            output_type="scalar",
        )
        assert output.value == 120.5
        assert output.output_type == "scalar"

    def test_stage_output_to_sexp(self):
        """StageOutput serializes to sexp."""
        output = StageOutput(
            cache_id="abc123",
            output_type="artifact",
        )
        sexp = output.to_sexp()
        sexp_str = serialize(sexp)

        assert "cache-id" in sexp_str
        assert "abc123" in sexp_str
        assert "type" in sexp_str
        assert "artifact" in sexp_str

    def test_stage_output_from_sexp(self):
        """StageOutput parses from sexp."""
        sexp = parse('(:cache-id "def456" :type :analysis)')
        output = StageOutput.from_sexp(sexp)

        assert output.cache_id == "def456"
        assert output.output_type == "analysis"


class TestStageCacheEntry:
    """Test StageCacheEntry serialization."""

    def test_stage_cache_entry_to_sexp(self):
        """StageCacheEntry serializes to sexp."""
        entry = StageCacheEntry(
            stage_name="analyze-a",
            cache_id="stage_abc123",
            outputs={
                "beats": StageOutput(cache_id="beats_def456", output_type="analysis"),
                "tempo": StageOutput(value=120.5, output_type="scalar"),
            },
            completed_at=1705678900.123,
        )

        sexp = entry.to_sexp()
        sexp_str = serialize(sexp)

        assert "stage-result" in sexp_str
        assert "analyze-a" in sexp_str
        assert "stage_abc123" in sexp_str
        assert "outputs" in sexp_str
        assert "beats" in sexp_str

    def test_stage_cache_entry_roundtrip(self):
        """save -> load produces identical data."""
        entry = StageCacheEntry(
            stage_name="analyze-b",
            cache_id="stage_xyz789",
            outputs={
                "segments": StageOutput(cache_id="seg_123", output_type="artifact"),
            },
            completed_at=1705678900.0,
        )

        sexp_str = entry.to_string()
        loaded = StageCacheEntry.from_string(sexp_str)

        assert loaded.stage_name == entry.stage_name
        assert loaded.cache_id == entry.cache_id
        assert "segments" in loaded.outputs
        assert loaded.outputs["segments"].cache_id == "seg_123"

    def test_stage_cache_entry_from_sexp(self):
        """StageCacheEntry parses from sexp."""
        sexp_str = '''
        (stage-result
          :name "test-stage"
          :cache-id "cache123"
          :completed-at 1705678900.0
          :outputs ((beats :cache-id "beats123" :type :analysis)))
        '''
        entry = StageCacheEntry.from_string(sexp_str)

        assert entry.stage_name == "test-stage"
        assert entry.cache_id == "cache123"
        assert "beats" in entry.outputs
        assert entry.outputs["beats"].cache_id == "beats123"


class TestStageCache:
    """Test StageCache file operations."""

    def test_save_and_load_stage(self):
        """Save and load a stage result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            entry = StageCacheEntry(
                stage_name="analyze",
                cache_id="test_cache_id",
                outputs={
                    "beats": StageOutput(cache_id="beats_out", output_type="analysis"),
                },
            )

            path = cache.save_stage(entry)
            assert path.exists()
            assert path.suffix == ".sexp"

            loaded = cache.load_stage("test_cache_id")
            assert loaded is not None
            assert loaded.stage_name == "analyze"
            assert "beats" in loaded.outputs

    def test_has_stage(self):
        """Check if stage is cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            assert not cache.has_stage("nonexistent")

            entry = StageCacheEntry(
                stage_name="test",
                cache_id="exists_cache_id",
                outputs={},
            )
            cache.save_stage(entry)

            assert cache.has_stage("exists_cache_id")

    def test_delete_stage(self):
        """Delete a cached stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            entry = StageCacheEntry(
                stage_name="test",
                cache_id="to_delete",
                outputs={},
            )
            cache.save_stage(entry)

            assert cache.has_stage("to_delete")
            result = cache.delete_stage("to_delete")
            assert result is True
            assert not cache.has_stage("to_delete")

    def test_list_stages(self):
        """List all cached stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            for i in range(3):
                entry = StageCacheEntry(
                    stage_name=f"stage{i}",
                    cache_id=f"cache_{i}",
                    outputs={},
                )
                cache.save_stage(entry)

            stages = cache.list_stages()
            assert len(stages) == 3
            assert "cache_0" in stages
            assert "cache_1" in stages
            assert "cache_2" in stages

    def test_clear(self):
        """Clear all cached stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            for i in range(3):
                entry = StageCacheEntry(
                    stage_name=f"stage{i}",
                    cache_id=f"cache_{i}",
                    outputs={},
                )
                cache.save_stage(entry)

            count = cache.clear()
            assert count == 3
            assert len(cache.list_stages()) == 0

    def test_cache_file_extension(self):
        """Cache files use .sexp extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)
            path = cache.get_cache_path("test_id")
            assert path.suffix == ".sexp"

    def test_invalid_sexp_error_handling(self):
        """Graceful error on corrupted cache file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = StageCache(tmpdir)

            # Write corrupted content
            corrupt_path = cache.get_cache_path("corrupted")
            corrupt_path.write_text("this is not valid sexp )()(")

            # Should return None, not raise
            result = cache.load_stage("corrupted")
            assert result is None


class TestAnalysisResult:
    """Test AnalysisResult serialization."""

    def test_analysis_result_to_sexp(self):
        """AnalysisResult serializes to sexp."""
        result = AnalysisResult(
            analyzer="beats",
            input_hash="input_abc123",
            data={
                "duration": 120.5,
                "tempo": 128.0,
                "times": [0.0, 0.468, 0.937, 1.406],
                "values": [0.8, 0.9, 0.7, 0.85],
            },
        )

        sexp = result.to_sexp()
        sexp_str = serialize(sexp)

        assert "analysis-result" in sexp_str
        assert "beats" in sexp_str
        assert "duration" in sexp_str
        assert "tempo" in sexp_str
        assert "times" in sexp_str

    def test_analysis_result_roundtrip(self):
        """Analysis result round-trips through sexp."""
        original = AnalysisResult(
            analyzer="scenes",
            input_hash="video_xyz",
            data={
                "scene_count": 5,
                "scene_times": [0.0, 10.5, 25.0, 45.2, 60.0],
            },
        )

        sexp_str = original.to_string()
        loaded = AnalysisResult.from_string(sexp_str)

        assert loaded.analyzer == original.analyzer
        assert loaded.input_hash == original.input_hash
        assert loaded.data["scene_count"] == 5

    def test_save_and_load_analysis_result(self):
        """Save and load analysis result from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = AnalysisResult(
                analyzer="beats",
                input_hash="audio_123",
                data={
                    "tempo": 120.0,
                    "times": [0.0, 0.5, 1.0],
                },
            )

            path = save_analysis_result(tmpdir, "node_abc", result)
            assert path.exists()
            assert path.name == "analysis.sexp"

            loaded = load_analysis_result(tmpdir, "node_abc")
            assert loaded is not None
            assert loaded.analyzer == "beats"
            assert loaded.data["tempo"] == 120.0

    def test_analysis_result_kebab_case(self):
        """Keys convert between snake_case and kebab-case."""
        result = AnalysisResult(
            analyzer="test",
            input_hash="hash",
            data={
                "scene_count": 5,
                "beat_times": [1, 2, 3],
            },
        )

        sexp_str = result.to_string()
        # Kebab case in sexp
        assert "scene-count" in sexp_str
        assert "beat-times" in sexp_str

        # Back to snake_case after parsing
        loaded = AnalysisResult.from_string(sexp_str)
        assert "scene_count" in loaded.data
        assert "beat_times" in loaded.data
