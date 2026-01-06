# tests/test_primitive_new/test_engine.py
"""Tests for primitive engine execution."""

import pytest
import subprocess
import tempfile
from pathlib import Path

from artdag.dag import DAG, DAGBuilder, Node, NodeType
from artdag.engine import Engine
from artdag import nodes  # Register executors


@pytest.fixture
def cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def engine(cache_dir):
    """Create engine instance."""
    return Engine(cache_dir)


@pytest.fixture
def test_video(cache_dir):
    """Create a test video file."""
    video_path = cache_dir / "test_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        str(video_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return video_path


@pytest.fixture
def test_audio(cache_dir):
    """Create a test audio file."""
    audio_path = cache_dir / "test_audio.mp3"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=5",
        "-c:a", "libmp3lame",
        str(audio_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_path


class TestEngineBasic:
    """Test basic engine functionality."""

    def test_engine_creation(self, cache_dir):
        """Test engine creation."""
        engine = Engine(cache_dir)
        assert engine.cache is not None

    def test_invalid_dag(self, engine):
        """Test executing invalid DAG."""
        dag = DAG()  # No nodes, no output
        result = engine.execute(dag)

        assert not result.success
        assert "Invalid DAG" in result.error

    def test_missing_executor(self, engine):
        """Test executing node with missing executor."""
        dag = DAG()
        node = Node(node_type="UNKNOWN_TYPE", config={})
        node_id = dag.add_node(node)
        dag.set_output(node_id)

        result = engine.execute(dag)

        assert not result.success
        assert "No executor" in result.error


class TestSourceExecutor:
    """Test SOURCE node executor."""

    def test_source_creates_symlink(self, engine, test_video):
        """Test source node creates symlink."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        builder.set_output(source)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success
        assert result.output_path.exists()
        assert result.output_path.is_symlink()

    def test_source_missing_file(self, engine):
        """Test source with missing file."""
        builder = DAGBuilder()
        source = builder.source("/nonexistent/file.mp4")
        builder.set_output(source)
        dag = builder.build()

        result = engine.execute(dag)

        assert not result.success
        assert "not found" in result.error.lower()


class TestSegmentExecutor:
    """Test SEGMENT node executor."""

    def test_segment_duration(self, engine, test_video):
        """Test segment extracts correct duration."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        segment = builder.segment(source, duration=2.0)
        builder.set_output(segment)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success

        # Verify duration
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(result.output_path)
        ], capture_output=True, text=True)
        duration = float(probe.stdout.strip())
        assert abs(duration - 2.0) < 0.1

    def test_segment_with_offset(self, engine, test_video):
        """Test segment with offset."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        segment = builder.segment(source, offset=1.0, duration=2.0)
        builder.set_output(segment)
        dag = builder.build()

        result = engine.execute(dag)
        assert result.success


class TestResizeExecutor:
    """Test RESIZE node executor."""

    def test_resize_dimensions(self, engine, test_video):
        """Test resize to specific dimensions."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        resized = builder.resize(source, width=640, height=480, mode="fit")
        builder.set_output(resized)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success

        # Verify dimensions
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0:s=x",
            str(result.output_path)
        ], capture_output=True, text=True)
        dimensions = probe.stdout.strip().split("\n")[0]
        assert "640x480" in dimensions


class TestTransformExecutor:
    """Test TRANSFORM node executor."""

    def test_transform_saturation(self, engine, test_video):
        """Test transform with saturation effect."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        transformed = builder.transform(source, effects={"saturation": 1.5})
        builder.set_output(transformed)
        dag = builder.build()

        result = engine.execute(dag)
        assert result.success
        assert result.output_path.exists()

    def test_transform_multiple_effects(self, engine, test_video):
        """Test transform with multiple effects."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        transformed = builder.transform(source, effects={
            "saturation": 1.2,
            "contrast": 1.1,
            "brightness": 0.05,
        })
        builder.set_output(transformed)
        dag = builder.build()

        result = engine.execute(dag)
        assert result.success


class TestSequenceExecutor:
    """Test SEQUENCE node executor."""

    def test_sequence_cut(self, engine, test_video):
        """Test sequence with cut transition."""
        builder = DAGBuilder()
        s1 = builder.source(str(test_video))
        seg1 = builder.segment(s1, duration=2.0)
        seg2 = builder.segment(s1, offset=2.0, duration=2.0)
        seq = builder.sequence([seg1, seg2], transition={"type": "cut"})
        builder.set_output(seq)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success

        # Verify combined duration
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(result.output_path)
        ], capture_output=True, text=True)
        duration = float(probe.stdout.strip())
        assert abs(duration - 4.0) < 0.2

    def test_sequence_crossfade(self, engine, test_video):
        """Test sequence with crossfade transition."""
        builder = DAGBuilder()
        s1 = builder.source(str(test_video))
        seg1 = builder.segment(s1, duration=3.0)
        seg2 = builder.segment(s1, offset=1.0, duration=3.0)
        seq = builder.sequence([seg1, seg2], transition={"type": "crossfade", "duration": 0.5})
        builder.set_output(seq)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success

        # Duration should be sum minus crossfade
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(result.output_path)
        ], capture_output=True, text=True)
        duration = float(probe.stdout.strip())
        # 3 + 3 - 0.5 = 5.5
        assert abs(duration - 5.5) < 0.3


class TestMuxExecutor:
    """Test MUX node executor."""

    def test_mux_video_audio(self, engine, test_video, test_audio):
        """Test muxing video and audio."""
        builder = DAGBuilder()
        video = builder.source(str(test_video))
        audio = builder.source(str(test_audio))
        muxed = builder.mux(video, audio)
        builder.set_output(muxed)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success
        assert result.output_path.exists()


class TestCaching:
    """Test engine caching behavior."""

    def test_cache_reuse(self, engine, test_video):
        """Test that cached results are reused."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        builder.set_output(source)
        dag = builder.build()

        # First execution
        result1 = engine.execute(dag)
        assert result1.success
        assert result1.nodes_cached == 0
        assert result1.nodes_executed == 1

        # Second execution should use cache
        result2 = engine.execute(dag)
        assert result2.success
        assert result2.nodes_cached == 1
        assert result2.nodes_executed == 0

    def test_clear_cache(self, engine, test_video):
        """Test clearing cache."""
        builder = DAGBuilder()
        source = builder.source(str(test_video))
        builder.set_output(source)
        dag = builder.build()

        engine.execute(dag)
        assert engine.cache.stats.total_entries == 1

        engine.clear_cache()
        assert engine.cache.stats.total_entries == 0


class TestProgressCallback:
    """Test progress callback functionality."""

    def test_progress_callback(self, engine, test_video):
        """Test that progress callback is called."""
        progress_updates = []

        def callback(progress):
            progress_updates.append((progress.node_id, progress.status))

        engine.set_progress_callback(callback)

        builder = DAGBuilder()
        source = builder.source(str(test_video))
        builder.set_output(source)
        dag = builder.build()

        result = engine.execute(dag)

        assert result.success
        assert len(progress_updates) > 0
        # Should have pending, running, completed
        statuses = [p[1] for p in progress_updates]
        assert "pending" in statuses
        assert "completed" in statuses


class TestFullWorkflow:
    """Test complete workflow."""

    def test_full_pipeline(self, engine, test_video, test_audio):
        """Test complete video processing pipeline."""
        builder = DAGBuilder()

        # Load sources
        video = builder.source(str(test_video))
        audio = builder.source(str(test_audio))

        # Extract segment
        segment = builder.segment(video, duration=3.0)

        # Resize
        resized = builder.resize(segment, width=640, height=480)

        # Apply effects
        transformed = builder.transform(resized, effects={"saturation": 1.3})

        # Mux with audio
        final = builder.mux(transformed, audio)
        builder.set_output(final)

        dag = builder.build()

        result = engine.execute(dag)

        assert result.success
        assert result.output_path.exists()
        assert result.nodes_executed == 6  # source, source, segment, resize, transform, mux
