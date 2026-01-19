"""
Tests for FFmpeg filter compilation.

Validates that each filter mapping produces valid FFmpeg commands.
"""

import subprocess
import tempfile
from pathlib import Path

from .ffmpeg_compiler import FFmpegCompiler, EFFECT_MAPPINGS


def test_filter_syntax(filter_str: str, duration: float = 0.1, is_complex: bool = False) -> tuple[bool, str]:
    """
    Test if an FFmpeg filter string is valid by running it on a test pattern.

    Args:
        filter_str: The filter string to test
        duration: Duration of test video
        is_complex: If True, use -filter_complex instead of -vf

    Returns (success, error_message)
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        output_path = f.name

    try:
        if is_complex:
            # Complex filter graph needs -filter_complex and explicit output mapping
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size=64x64:rate=10',
                '-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}',
                '-filter_complex', f'[0:v]{filter_str}[out]',
                '-map', '[out]', '-map', '1:a',
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac',
                '-t', str(duration),
                output_path
            ]
        else:
            # Simple filter uses -vf
            cmd = [
                'ffmpeg', '-y',
                '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size=64x64:rate=10',
                '-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}',
                '-vf', filter_str,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac',
                '-t', str(duration),
                output_path
            ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            return True, ""
        else:
            # Extract relevant error
            stderr = result.stderr
            for line in stderr.split('\n'):
                if 'Error' in line or 'error' in line or 'Invalid' in line:
                    return False, line.strip()
            return False, stderr[-500:] if len(stderr) > 500 else stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        Path(output_path).unlink(missing_ok=True)


def run_all_tests():
    """Test all effect mappings."""
    compiler = FFmpegCompiler()
    results = []

    for effect_name, mapping in EFFECT_MAPPINGS.items():
        filter_name = mapping.get("filter")

        # Skip effects with no FFmpeg equivalent (external tools or python primitives)
        if filter_name is None:
            reason = "No FFmpeg equivalent"
            if mapping.get("external_tool"):
                reason = f"External tool: {mapping['external_tool']}"
            elif mapping.get("python_primitive"):
                reason = f"Python primitive: {mapping['python_primitive']}"
            results.append((effect_name, "SKIP", reason))
            continue

        # Check if complex filter
        is_complex = mapping.get("complex", False)

        # Build filter string
        if "static" in mapping:
            filter_str = f"{filter_name}={mapping['static']}"
        else:
            filter_str = filter_name

        # Test it
        success, error = test_filter_syntax(filter_str, is_complex=is_complex)

        if success:
            results.append((effect_name, "PASS", filter_str))
        else:
            results.append((effect_name, "FAIL", f"{filter_str} -> {error}"))

    return results


def print_results(results):
    """Print test results."""
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    skipped = sum(1 for _, status, _ in results if status == "SKIP")

    print(f"\n{'='*60}")
    print(f"FFmpeg Filter Tests: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}\n")

    # Print failures first
    if failed > 0:
        print("FAILURES:")
        for name, status, msg in results:
            if status == "FAIL":
                print(f"  {name}: {msg}")
        print()

    # Print passes
    print("PASSED:")
    for name, status, msg in results:
        if status == "PASS":
            print(f"  {name}: {msg}")

    # Print skips
    if skipped > 0:
        print("\nSKIPPED (Python fallback):")
        for name, status, msg in results:
            if status == "SKIP":
                print(f"  {name}")


if __name__ == "__main__":
    results = run_all_tests()
    print_results(results)
