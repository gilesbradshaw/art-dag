"""
Tests for Python primitive effects.

Tests that ascii_art, ascii_zones, and other Python primitives
can be executed via the EffectExecutor.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

try:
    import numpy as np
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from .primitives import (
    ascii_art_frame,
    ascii_zones_frame,
    get_primitive,
    list_primitives,
)
from .ffmpeg_compiler import FFmpegCompiler


def create_test_video(path: Path, duration: float = 0.5, size: str = "64x64") -> bool:
    """Create a short test video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={size}:rate=10",
        "-c:v", "libx264", "-preset", "ultrafast",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


@pytest.mark.skipif(not HAS_DEPS, reason="numpy/PIL not available")
class TestPrimitives:
    """Test primitive functions directly."""

    def test_ascii_art_frame_basic(self):
        """Test ascii_art_frame produces output of same shape."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = ascii_art_frame(frame, char_size=8)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_ascii_zones_frame_basic(self):
        """Test ascii_zones_frame produces output of same shape."""
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = ascii_zones_frame(frame, char_size=8)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_get_primitive(self):
        """Test primitive lookup."""
        assert get_primitive("ascii_art_frame") is ascii_art_frame
        assert get_primitive("ascii_zones_frame") is ascii_zones_frame
        assert get_primitive("nonexistent") is None

    def test_list_primitives(self):
        """Test listing primitives."""
        primitives = list_primitives()
        assert "ascii_art_frame" in primitives
        assert "ascii_zones_frame" in primitives
        assert len(primitives) > 5


class TestFFmpegCompilerPrimitives:
    """Test FFmpegCompiler python_primitive mappings."""

    def test_has_python_primitive_ascii_art(self):
        """Test ascii_art has python_primitive."""
        compiler = FFmpegCompiler()
        assert compiler.has_python_primitive("ascii_art") == "ascii_art_frame"

    def test_has_python_primitive_ascii_zones(self):
        """Test ascii_zones has python_primitive."""
        compiler = FFmpegCompiler()
        assert compiler.has_python_primitive("ascii_zones") == "ascii_zones_frame"

    def test_has_python_primitive_ffmpeg_effect(self):
        """Test FFmpeg effects don't have python_primitive."""
        compiler = FFmpegCompiler()
        assert compiler.has_python_primitive("brightness") is None
        assert compiler.has_python_primitive("blur") is None

    def test_compile_effect_returns_none_for_primitives(self):
        """Test compile_effect returns None for primitive effects."""
        compiler = FFmpegCompiler()
        assert compiler.compile_effect("ascii_art", {}) is None
        assert compiler.compile_effect("ascii_zones", {}) is None


@pytest.mark.skipif(not HAS_DEPS, reason="numpy/PIL not available")
class TestEffectExecutorPrimitives:
    """Test EffectExecutor with Python primitives."""

    def test_executor_loads_primitive(self):
        """Test that executor finds primitive effects."""
        from ..nodes.effect import _get_python_primitive_effect

        effect_fn = _get_python_primitive_effect("ascii_art")
        assert effect_fn is not None

        effect_fn = _get_python_primitive_effect("ascii_zones")
        assert effect_fn is not None

    def test_executor_rejects_unknown_effect(self):
        """Test that executor returns None for unknown effects."""
        from ..nodes.effect import _get_python_primitive_effect

        effect_fn = _get_python_primitive_effect("nonexistent_effect")
        assert effect_fn is None

    def test_execute_ascii_art_effect(self, tmp_path):
        """Test executing ascii_art effect on a video."""
        from ..nodes.effect import EffectExecutor

        # Create test video
        input_path = tmp_path / "input.mp4"
        if not create_test_video(input_path):
            pytest.skip("Could not create test video")

        output_path = tmp_path / "output.mkv"
        executor = EffectExecutor()

        result = executor.execute(
            config={"effect": "ascii_art", "char_size": 8},
            inputs=[input_path],
            output_path=output_path,
        )

        assert result.exists()
        assert result.stat().st_size > 0


def run_all_tests():
    """Run tests manually."""
    import sys

    # Check dependencies
    if not HAS_DEPS:
        print("SKIP: numpy/PIL not available")
        return

    print("Testing primitives...")

    # Test primitive functions
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    print("  ascii_art_frame...", end=" ")
    result = ascii_art_frame(frame, char_size=8)
    assert result.shape == frame.shape
    print("PASS")

    print("  ascii_zones_frame...", end=" ")
    result = ascii_zones_frame(frame, char_size=8)
    assert result.shape == frame.shape
    print("PASS")

    # Test FFmpegCompiler mappings
    print("\nTesting FFmpegCompiler mappings...")
    compiler = FFmpegCompiler()

    print("  ascii_art python_primitive...", end=" ")
    assert compiler.has_python_primitive("ascii_art") == "ascii_art_frame"
    print("PASS")

    print("  ascii_zones python_primitive...", end=" ")
    assert compiler.has_python_primitive("ascii_zones") == "ascii_zones_frame"
    print("PASS")

    # Test executor lookup
    print("\nTesting EffectExecutor...")
    try:
        from ..nodes.effect import _get_python_primitive_effect

        print("  _get_python_primitive_effect(ascii_art)...", end=" ")
        effect_fn = _get_python_primitive_effect("ascii_art")
        assert effect_fn is not None
        print("PASS")

        print("  _get_python_primitive_effect(ascii_zones)...", end=" ")
        effect_fn = _get_python_primitive_effect("ascii_zones")
        assert effect_fn is not None
        print("PASS")

    except ImportError as e:
        print(f"SKIP: {e}")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    run_all_tests()
