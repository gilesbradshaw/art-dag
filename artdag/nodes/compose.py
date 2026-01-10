# primitive/nodes/compose.py
"""
Compose executors: Combine multiple media inputs.

Primitives: SEQUENCE, LAYER, MUX, BLEND
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from ..dag import NodeType
from ..executor import Executor, register_executor
from .encoding import WEB_ENCODING_ARGS_STR, get_web_encoding_args

logger = logging.getLogger(__name__)


def _get_duration(path: Path) -> float:
    """Get media duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def _get_video_info(path: Path) -> dict:
    """Get video width, height, frame rate, and sample rate."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    width = int(parts[0]) if len(parts) > 0 and parts[0] else 1920
    height = int(parts[1]) if len(parts) > 1 and parts[1] else 1080
    fps_str = parts[2] if len(parts) > 2 else "30/1"
    # Parse frame rate (e.g., "30/1" or "30000/1001")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30
    else:
        fps = float(fps_str) if fps_str else 30

    # Get audio sample rate
    cmd_audio = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "csv=p=0",
        str(path)
    ]
    result_audio = subprocess.run(cmd_audio, capture_output=True, text=True)
    sample_rate = int(result_audio.stdout.strip()) if result_audio.stdout.strip() else 44100

    return {"width": width, "height": height, "fps": fps, "sample_rate": sample_rate}


@register_executor(NodeType.SEQUENCE)
class SequenceExecutor(Executor):
    """
    Concatenate inputs in time order.

    Config:
        transition: Transition config
            type: "cut" | "crossfade" | "fade"
            duration: Transition duration in seconds
        target_size: How to determine output dimensions when inputs differ
            "first": Use first input's dimensions (default)
            "last": Use last input's dimensions
            "largest": Use largest width and height from all inputs
            "explicit": Use width/height config values
        width: Target width (when target_size="explicit")
        height: Target height (when target_size="explicit")
        background: Padding color for letterbox/pillarbox (default: "black")
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) < 1:
            raise ValueError("SEQUENCE requires at least one input")

        if len(inputs) == 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inputs[0], output_path)
            return output_path

        transition = config.get("transition", {"type": "cut"})
        transition_type = transition.get("type", "cut")
        transition_duration = transition.get("duration", 0.5)

        # Size handling config
        target_size = config.get("target_size", "first")
        width = config.get("width")
        height = config.get("height")
        background = config.get("background", "black")

        if transition_type == "cut":
            return self._concat_cut(inputs, output_path, target_size, width, height, background)
        elif transition_type == "crossfade":
            return self._concat_crossfade(inputs, output_path, transition_duration)
        elif transition_type == "fade":
            return self._concat_fade(inputs, output_path, transition_duration)
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")

    def _concat_cut(
        self,
        inputs: List[Path],
        output_path: Path,
        target_size: str = "first",
        width: int = None,
        height: int = None,
        background: str = "black",
    ) -> Path:
        """
        Concatenate with scaling/padding to handle different resolutions.

        Args:
            inputs: Input video paths
            output_path: Output path
            target_size: How to determine output size:
                - "first": Use first input's dimensions (default)
                - "last": Use last input's dimensions
                - "largest": Use largest dimensions from all inputs
                - "explicit": Use width/height params
            width: Explicit width (when target_size="explicit")
            height: Explicit height (when target_size="explicit")
            background: Padding color (default: black)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(inputs)
        input_args = []
        for p in inputs:
            input_args.extend(["-i", str(p)])

        # Get video info for all inputs
        infos = [_get_video_info(p) for p in inputs]

        # Determine target dimensions
        if target_size == "explicit" and width and height:
            target_w, target_h = width, height
        elif target_size == "last":
            target_w, target_h = infos[-1]["width"], infos[-1]["height"]
        elif target_size == "largest":
            target_w = max(i["width"] for i in infos)
            target_h = max(i["height"] for i in infos)
        else:  # "first" or default
            target_w, target_h = infos[0]["width"], infos[0]["height"]

        # Use common frame rate (from first input) and sample rate
        target_fps = infos[0]["fps"]
        target_sr = max(i["sample_rate"] for i in infos)

        # Build filter for each input: scale to fit + pad to target size
        filter_parts = []
        for i in range(n):
            # Scale to fit within target, maintaining aspect ratio, then pad
            vf = (
                f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color={background},"
                f"setsar=1,fps={target_fps:.6f}[v{i}]"
            )
            # Resample audio to common rate
            af = f"[{i}:a]aresample={target_sr}[a{i}]"
            filter_parts.append(vf)
            filter_parts.append(af)

        # Build concat filter
        stream_labels = "".join(f"[v{i}][a{i}]" for i in range(n))
        filter_parts.append(f"{stream_labels}concat=n={n}:v=1:a=1[outv][outa]")

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", "[outa]",
            *get_web_encoding_args(),
            str(output_path)
        ]

        logger.debug(f"SEQUENCE cut: {n} clips -> {target_w}x{target_h} (web-optimized)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Concat failed: {result.stderr}")

        return output_path

    def _concat_crossfade(
        self,
        inputs: List[Path],
        output_path: Path,
        duration: float,
    ) -> Path:
        """Concatenate with crossfade transitions."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        durations = [_get_duration(p) for p in inputs]
        n = len(inputs)
        input_args = " ".join(f"-i {p}" for p in inputs)

        # Build xfade filter chain
        filter_parts = []
        current = "[0:v]"

        for i in range(1, n):
            offset = sum(durations[:i]) - duration * i
            next_input = f"[{i}:v]"
            output_label = f"[v{i}]" if i < n - 1 else "[outv]"
            filter_parts.append(
                f"{current}{next_input}xfade=transition=fade:duration={duration}:offset={offset}{output_label}"
            )
            current = output_label

        # Audio crossfade chain
        audio_current = "[0:a]"
        for i in range(1, n):
            next_input = f"[{i}:a]"
            output_label = f"[a{i}]" if i < n - 1 else "[outa]"
            filter_parts.append(
                f"{audio_current}{next_input}acrossfade=d={duration}{output_label}"
            )
            audio_current = output_label

        filter_complex = ";".join(filter_parts)

        cmd = f'ffmpeg -y {input_args} -filter_complex "{filter_complex}" -map [outv] -map [outa] {WEB_ENCODING_ARGS_STR} {output_path}'

        logger.debug(f"SEQUENCE crossfade: {n} clips (web-optimized)")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.warning(f"Crossfade failed, falling back to cut: {result.stderr[:200]}")
            return self._concat_cut(inputs, output_path)

        return output_path

    def _concat_fade(
        self,
        inputs: List[Path],
        output_path: Path,
        duration: float,
    ) -> Path:
        """Concatenate with fade out/in transitions."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        faded_paths = []
        for i, path in enumerate(inputs):
            clip_dur = _get_duration(path)
            faded_path = output_path.parent / f"_faded_{i}.mkv"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(path),
                "-vf", f"fade=in:st=0:d={duration},fade=out:st={clip_dur - duration}:d={duration}",
                "-af", f"afade=in:st=0:d={duration},afade=out:st={clip_dur - duration}:d={duration}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac",
                str(faded_path)
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            faded_paths.append(faded_path)

        result = self._concat_cut(faded_paths, output_path)

        for p in faded_paths:
            p.unlink()

        return result


@register_executor(NodeType.LAYER)
class LayerExecutor(Executor):
    """
    Layer inputs spatially (overlay/composite).

    Config:
        inputs: List of per-input configs
            position: [x, y] offset
            opacity: 0.0-1.0
            scale: Scale factor
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) < 1:
            raise ValueError("LAYER requires at least one input")

        if len(inputs) == 1:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inputs[0], output_path)
            return output_path

        input_configs = config.get("inputs", [{}] * len(inputs))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_args = " ".join(f"-i {p}" for p in inputs)
        n = len(inputs)
        filter_parts = []
        current = "[0:v]"

        for i in range(1, n):
            cfg = input_configs[i] if i < len(input_configs) else {}
            x, y = cfg.get("position", [0, 0])
            opacity = cfg.get("opacity", 1.0)
            scale = cfg.get("scale", 1.0)

            scale_label = f"[s{i}]"
            if scale != 1.0:
                filter_parts.append(f"[{i}:v]scale=iw*{scale}:ih*{scale}{scale_label}")
                overlay_input = scale_label
            else:
                overlay_input = f"[{i}:v]"

            output_label = f"[v{i}]" if i < n - 1 else "[outv]"

            if opacity < 1.0:
                filter_parts.append(
                    f"{overlay_input}format=rgba,colorchannelmixer=aa={opacity}[a{i}]"
                )
                overlay_input = f"[a{i}]"

            filter_parts.append(
                f"{current}{overlay_input}overlay=x={x}:y={y}:format=auto{output_label}"
            )
            current = output_label

        filter_complex = ";".join(filter_parts)

        cmd = f'ffmpeg -y {input_args} -filter_complex "{filter_complex}" -map [outv] -map 0:a? {WEB_ENCODING_ARGS_STR} {output_path}'

        logger.debug(f"LAYER: {n} inputs (web-optimized)")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Layer failed: {result.stderr}")

        return output_path


@register_executor(NodeType.MUX)
class MuxExecutor(Executor):
    """
    Combine video and audio streams.

    Config:
        video_stream: Index of video input (default: 0)
        audio_stream: Index of audio input (default: 1)
        shortest: End when shortest stream ends (default: True)
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) < 2:
            raise ValueError("MUX requires at least 2 inputs (video + audio)")

        video_idx = config.get("video_stream", 0)
        audio_idx = config.get("audio_stream", 1)
        shortest = config.get("shortest", True)

        video_path = inputs[video_idx]
        audio_path = inputs[audio_idx]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

        if shortest:
            cmd.append("-shortest")

        cmd.append(str(output_path))

        logger.debug(f"MUX: video={video_path.name} + audio={audio_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Mux failed: {result.stderr}")

        return output_path


@register_executor(NodeType.BLEND)
class BlendExecutor(Executor):
    """
    Blend two inputs using a blend mode.

    Config:
        mode: Blend mode (multiply, screen, overlay, add, etc.)
        opacity: 0.0-1.0 for second input
    """

    BLEND_MODES = {
        "multiply": "multiply",
        "screen": "screen",
        "overlay": "overlay",
        "add": "addition",
        "subtract": "subtract",
        "average": "average",
        "difference": "difference",
        "lighten": "lighten",
        "darken": "darken",
    }

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) != 2:
            raise ValueError("BLEND requires exactly 2 inputs")

        mode = config.get("mode", "overlay")
        opacity = config.get("opacity", 0.5)

        if mode not in self.BLEND_MODES:
            raise ValueError(f"Unknown blend mode: {mode}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        blend_mode = self.BLEND_MODES[mode]

        if opacity < 1.0:
            filter_complex = (
                f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[b];"
                f"[0:v][b]blend=all_mode={blend_mode}"
            )
        else:
            filter_complex = f"[0:v][1:v]blend=all_mode={blend_mode}"

        cmd = [
            "ffmpeg", "-y",
            "-i", str(inputs[0]),
            "-i", str(inputs[1]),
            "-filter_complex", filter_complex,
            "-map", "0:a?",
            *get_web_encoding_args(),
            str(output_path)
        ]

        logger.debug(f"BLEND: {mode} (opacity={opacity}) (web-optimized)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Blend failed: {result.stderr}")

        return output_path


@register_executor(NodeType.AUDIO_MIX)
class AudioMixExecutor(Executor):
    """
    Mix multiple audio streams.

    Config:
        gains: List of gain values per input (0.0-2.0, default 1.0)
        normalize: Normalize output to prevent clipping (default True)
    """

    def execute(
        self,
        config: Dict[str, Any],
        inputs: List[Path],
        output_path: Path,
    ) -> Path:
        if len(inputs) < 2:
            raise ValueError("AUDIO_MIX requires at least 2 inputs")

        gains = config.get("gains", [1.0] * len(inputs))
        normalize = config.get("normalize", True)

        # Pad gains list if too short
        while len(gains) < len(inputs):
            gains.append(1.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build filter: apply volume to each input, then mix
        filter_parts = []
        mix_inputs = []

        for i, gain in enumerate(gains[:len(inputs)]):
            if gain != 1.0:
                filter_parts.append(f"[{i}:a]volume={gain}[a{i}]")
                mix_inputs.append(f"[a{i}]")
            else:
                mix_inputs.append(f"[{i}:a]")

        # amix filter
        normalize_flag = 1 if normalize else 0
        mix_filter = f"{''.join(mix_inputs)}amix=inputs={len(inputs)}:normalize={normalize_flag}[aout]"
        filter_parts.append(mix_filter)

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
        ]
        for p in inputs:
            cmd.extend(["-i", str(p)])

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-c:a", "aac",
            str(output_path)
        ])

        logger.debug(f"AUDIO_MIX: {len(inputs)} inputs, gains={gains[:len(inputs)]}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Audio mix failed: {result.stderr}")

        return output_path
