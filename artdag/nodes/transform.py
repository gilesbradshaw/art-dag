# primitive/nodes/transform.py
"""
Transform executors: Modify single media inputs.

Primitives: SEGMENT, RESIZE, TRANSFORM
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..dag import NodeType
from ..executor import Executor, register_executor

logger = logging.getLogger(__name__)


@register_executor(NodeType.SEGMENT)
class SegmentExecutor(Executor):
    """
    Extract a time segment from media.

    Config:
        offset: Start time in seconds (default: 0)
        duration: Duration in seconds (optional, default: to end)
        precise: Use frame-accurate seeking (default: True)
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) != 1:
            raise ValueError("SEGMENT requires exactly one input")

        input_path = inputs[0]
        offset = config.get("offset", 0)
        duration = config.get("duration")
        precise = config.get("precise", True)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if precise:
            # Frame-accurate: decode-seek (slower but precise)
            cmd = ["ffmpeg", "-y", "-i", str(input_path)]
            if offset > 0:
                cmd.extend(["-ss", str(offset)])
            if duration:
                cmd.extend(["-t", str(duration)])
            cmd.extend([
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac",
                str(output_path)
            ])
        else:
            # Fast: input-seek at keyframes (may be slightly off)
            cmd = ["ffmpeg", "-y"]
            if offset > 0:
                cmd.extend(["-ss", str(offset)])
            cmd.extend(["-i", str(input_path)])
            if duration:
                cmd.extend(["-t", str(duration)])
            cmd.extend(["-c", "copy", str(output_path)])

        logger.debug(f"SEGMENT: offset={offset}, duration={duration}, precise={precise}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Segment failed: {result.stderr}")

        return output_path


@register_executor(NodeType.RESIZE)
class ResizeExecutor(Executor):
    """
    Resize media to target dimensions.

    Config:
        width: Target width
        height: Target height
        mode: "fit" (letterbox), "fill" (crop), "stretch", "pad"
        background: Background color for pad mode (default: black)
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) != 1:
            raise ValueError("RESIZE requires exactly one input")

        input_path = inputs[0]
        width = config["width"]
        height = config["height"]
        mode = config.get("mode", "fit")
        background = config.get("background", "black")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "fit":
            # Scale to fit, add letterboxing
            vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background}"
        elif mode == "fill":
            # Scale to fill, crop excess
            vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        elif mode == "stretch":
            # Stretch to exact size
            vf = f"scale={width}:{height}"
        elif mode == "pad":
            # Scale down only if larger, then pad
            vf = f"scale='min({width},iw)':'min({height},ih)':force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={background}"
        else:
            raise ValueError(f"Unknown resize mode: {mode}")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "copy",
            str(output_path)
        ]

        logger.debug(f"RESIZE: {width}x{height} ({mode})")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Resize failed: {result.stderr}")

        return output_path


@register_executor(NodeType.TRANSFORM)
class TransformExecutor(Executor):
    """
    Apply visual effects to media.

    Config:
        effects: Dict of effect -> value
            saturation: 0.0-2.0 (1.0 = normal)
            contrast: 0.0-2.0 (1.0 = normal)
            brightness: -1.0 to 1.0 (0.0 = normal)
            gamma: 0.1-10.0 (1.0 = normal)
            hue: degrees shift
            blur: blur radius
            sharpen: sharpen amount
            speed: playback speed multiplier
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) != 1:
            raise ValueError("TRANSFORM requires exactly one input")

        input_path = inputs[0]
        effects = config.get("effects", {})

        if not effects:
            # No effects - just copy
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, output_path)
            return output_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build filter chain
        vf_parts = []
        af_parts = []

        # Color adjustments via eq filter
        eq_parts = []
        if "saturation" in effects:
            eq_parts.append(f"saturation={effects['saturation']}")
        if "contrast" in effects:
            eq_parts.append(f"contrast={effects['contrast']}")
        if "brightness" in effects:
            eq_parts.append(f"brightness={effects['brightness']}")
        if "gamma" in effects:
            eq_parts.append(f"gamma={effects['gamma']}")
        if eq_parts:
            vf_parts.append(f"eq={':'.join(eq_parts)}")

        # Hue adjustment
        if "hue" in effects:
            vf_parts.append(f"hue=h={effects['hue']}")

        # Blur
        if "blur" in effects:
            vf_parts.append(f"boxblur={effects['blur']}")

        # Sharpen
        if "sharpen" in effects:
            vf_parts.append(f"unsharp=5:5:{effects['sharpen']}:5:5:0")

        # Speed change
        if "speed" in effects:
            speed = effects["speed"]
            vf_parts.append(f"setpts={1/speed}*PTS")
            af_parts.append(f"atempo={speed}")

        cmd = ["ffmpeg", "-y", "-i", str(input_path)]

        if vf_parts:
            cmd.extend(["-vf", ",".join(vf_parts)])
        if af_parts:
            cmd.extend(["-af", ",".join(af_parts)])

        cmd.extend([
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
            "-c:a", "aac",
            str(output_path)
        ])

        logger.debug(f"TRANSFORM: {list(effects.keys())}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Transform failed: {result.stderr}")

        return output_path
