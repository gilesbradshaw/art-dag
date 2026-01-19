"""
External tool runners for effects that can't be done in FFmpeg.

Supports:
- datamosh: via ffglitch or datamoshing Python CLI
- pixelsort: via Rust pixelsort or Python pixelsort-cli
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_tool(tool_names: List[str]) -> Optional[str]:
    """Find first available tool from a list of candidates."""
    for name in tool_names:
        path = shutil.which(name)
        if path:
            return path
    return None


def check_python_package(package: str) -> bool:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            ["python3", "-c", f"import {package}"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# Tool detection
DATAMOSH_TOOLS = ["ffgac", "ffedit"]  # ffglitch tools
PIXELSORT_TOOLS = ["pixelsort"]  # Rust CLI


def get_available_tools() -> Dict[str, Optional[str]]:
    """Get dictionary of available external tools."""
    return {
        "datamosh": find_tool(DATAMOSH_TOOLS),
        "pixelsort": find_tool(PIXELSORT_TOOLS),
        "datamosh_python": "datamoshing" if check_python_package("datamoshing") else None,
        "pixelsort_python": "pixelsort" if check_python_package("pixelsort") else None,
    }


def run_datamosh(
    input_path: Path,
    output_path: Path,
    params: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Run datamosh effect using available tool.

    Args:
        input_path: Input video file
        output_path: Output video file
        params: Effect parameters (corruption, block_size, etc.)

    Returns:
        (success, error_message)
    """
    tools = get_available_tools()

    corruption = params.get("corruption", 0.3)

    # Try ffglitch first
    if tools.get("datamosh"):
        ffgac = tools["datamosh"]
        try:
            # ffglitch approach: remove I-frames to create datamosh effect
            # This is a simplified version - full datamosh needs custom scripts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                # Write a simple ffglitch script that corrupts motion vectors
                f.write(f"""
// Datamosh script - corrupt motion vectors
let corruption = {corruption};

export function glitch_frame(frame, stream) {{
    if (frame.pict_type === 'P' && Math.random() < corruption) {{
        // Corrupt motion vectors
        let dominated = frame.mv?.forward?.overflow;
        if (dominated) {{
            for (let i = 0; i < dominated.length; i++) {{
                if (Math.random() < corruption) {{
                    dominated[i] = [
                        Math.floor(Math.random() * 64 - 32),
                        Math.floor(Math.random() * 64 - 32)
                    ];
                }}
            }}
        }}
    }}
    return frame;
}}
""")
                script_path = f.name

            cmd = [
                ffgac,
                "-i", str(input_path),
                "-s", script_path,
                "-o", str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            Path(script_path).unlink(missing_ok=True)

            if result.returncode == 0:
                return True, ""
            return False, result.stderr[:500]

        except subprocess.TimeoutExpired:
            return False, "Datamosh timeout"
        except Exception as e:
            return False, str(e)

    # Fall back to Python datamoshing package
    if tools.get("datamosh_python"):
        try:
            cmd = [
                "python3", "-m", "datamoshing",
                str(input_path),
                str(output_path),
                "--mode", "iframe_removal",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[:500]
        except Exception as e:
            return False, str(e)

    return False, "No datamosh tool available. Install ffglitch or: pip install datamoshing"


def run_pixelsort(
    input_path: Path,
    output_path: Path,
    params: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Run pixelsort effect using available tool.

    Args:
        input_path: Input image/frame file
        output_path: Output image file
        params: Effect parameters (sort_by, threshold_low, threshold_high, angle)

    Returns:
        (success, error_message)
    """
    tools = get_available_tools()

    sort_by = params.get("sort_by", "lightness")
    threshold_low = params.get("threshold_low", 50)
    threshold_high = params.get("threshold_high", 200)
    angle = params.get("angle", 0)

    # Try Rust pixelsort first (faster)
    if tools.get("pixelsort"):
        try:
            cmd = [
                tools["pixelsort"],
                str(input_path),
                "-o", str(output_path),
                "--sort", sort_by,
                "-r", str(angle),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[:500]
        except Exception as e:
            return False, str(e)

    # Fall back to Python pixelsort-cli
    if tools.get("pixelsort_python"):
        try:
            cmd = [
                "python3", "-m", "pixelsort",
                "--image_path", str(input_path),
                "--output", str(output_path),
                "--angle", str(angle),
                "--threshold", str(threshold_low / 255),  # Normalize to 0-1
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True, ""
            return False, result.stderr[:500]
        except Exception as e:
            return False, str(e)

    return False, "No pixelsort tool available. Install: cargo install pixelsort or pip install pixelsort-cli"


def run_pixelsort_video(
    input_path: Path,
    output_path: Path,
    params: Dict[str, Any],
    fps: float = 30,
) -> Tuple[bool, str]:
    """
    Run pixelsort on a video by processing each frame.

    This extracts frames, processes them, then reassembles.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_in = tmpdir / "frame_%04d.png"
        frames_out = tmpdir / "sorted_%04d.png"

        # Extract frames
        extract_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", f"fps={fps}",
            str(frames_in),
        ]
        result = subprocess.run(extract_cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            return False, "Failed to extract frames"

        # Process each frame
        frame_files = sorted(tmpdir.glob("frame_*.png"))
        for i, frame in enumerate(frame_files):
            out_frame = tmpdir / f"sorted_{i:04d}.png"
            success, error = run_pixelsort(frame, out_frame, params)
            if not success:
                return False, f"Frame {i}: {error}"

        # Reassemble
        # Get audio from original
        reassemble_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "sorted_%04d.png"),
            "-i", str(input_path),
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            str(output_path),
        ]
        result = subprocess.run(reassemble_cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            return False, "Failed to reassemble video"

        return True, ""


def run_external_effect(
    effect_name: str,
    input_path: Path,
    output_path: Path,
    params: Dict[str, Any],
    is_video: bool = True,
) -> Tuple[bool, str]:
    """
    Run an external effect tool.

    Args:
        effect_name: Name of effect (datamosh, pixelsort)
        input_path: Input file
        output_path: Output file
        params: Effect parameters
        is_video: Whether input is video (vs single image)

    Returns:
        (success, error_message)
    """
    if effect_name == "datamosh":
        return run_datamosh(input_path, output_path, params)
    elif effect_name == "pixelsort":
        if is_video:
            return run_pixelsort_video(input_path, output_path, params)
        else:
            return run_pixelsort(input_path, output_path, params)
    else:
        return False, f"Unknown external effect: {effect_name}"


if __name__ == "__main__":
    # Print available tools
    print("Available external tools:")
    for name, path in get_available_tools().items():
        status = path if path else "NOT INSTALLED"
        print(f"  {name}: {status}")
