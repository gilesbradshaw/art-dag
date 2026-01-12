"""
FFmpeg pipe-based frame processing.

Processes video through Python frame-by-frame effects using FFmpeg pipes:
  FFmpeg decode -> Python process_frame -> FFmpeg encode

This avoids writing intermediate frames to disk.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video metadata."""

    width: int
    height: int
    frame_rate: float
    total_frames: int
    duration: float
    pixel_format: str = "rgb24"


def probe_video(path: Path) -> VideoInfo:
    """
    Get video information using ffprobe.

    Args:
        path: Path to video file

    Returns:
        VideoInfo with dimensions, frame rate, etc.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "csv=p=0",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    parts = result.stdout.strip().split(",")
    if len(parts) < 4:
        raise RuntimeError(f"Unexpected ffprobe output: {result.stdout}")

    width = int(parts[0])
    height = int(parts[1])

    # Parse frame rate (could be "30/1" or "30")
    fr_parts = parts[2].split("/")
    if len(fr_parts) == 2:
        frame_rate = float(fr_parts[0]) / float(fr_parts[1])
    else:
        frame_rate = float(fr_parts[0])

    # nb_frames might be N/A
    total_frames = 0
    duration = 0.0
    try:
        total_frames = int(parts[3])
    except (ValueError, IndexError):
        pass

    try:
        duration = float(parts[4]) if len(parts) > 4 else 0.0
    except (ValueError, IndexError):
        pass

    if total_frames == 0 and duration > 0:
        total_frames = int(duration * frame_rate)

    return VideoInfo(
        width=width,
        height=height,
        frame_rate=frame_rate,
        total_frames=total_frames,
        duration=duration,
    )


FrameProcessor = Callable[[np.ndarray, Dict[str, Any], Any], Tuple[np.ndarray, Any]]


def process_video(
    input_path: Path,
    output_path: Path,
    process_frame: FrameProcessor,
    params: Dict[str, Any],
    bindings: Dict[str, List[float]] = None,
    initial_state: Any = None,
    pixel_format: str = "rgb24",
    output_codec: str = "libx264",
    output_options: List[str] = None,
) -> Tuple[Path, Any]:
    """
    Process video through frame-by-frame effect.

    Args:
        input_path: Input video path
        output_path: Output video path
        process_frame: Function (frame, params, state) -> (frame, state)
        params: Static parameter dict
        bindings: Per-frame parameter lookup tables
        initial_state: Initial state for process_frame
        pixel_format: Pixel format for frame data
        output_codec: Video codec for output
        output_options: Additional ffmpeg output options

    Returns:
        Tuple of (output_path, final_state)
    """
    bindings = bindings or {}
    output_options = output_options or []

    # Probe input
    info = probe_video(input_path)
    logger.info(f"Processing {info.width}x{info.height} @ {info.frame_rate}fps")

    # Calculate bytes per frame
    if pixel_format == "rgb24":
        bytes_per_pixel = 3
    elif pixel_format == "rgba":
        bytes_per_pixel = 4
    else:
        bytes_per_pixel = 3  # Default to RGB

    frame_size = info.width * info.height * bytes_per_pixel

    # Start decoder process
    decode_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
        "-",
    ]

    # Start encoder process
    encode_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
        "-s", f"{info.width}x{info.height}",
        "-r", str(info.frame_rate),
        "-i", "-",
        "-i", str(input_path),  # For audio
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", output_codec,
        "-c:a", "aac",
        *output_options,
        str(output_path),
    ]

    logger.debug(f"Decoder: {' '.join(decode_cmd)}")
    logger.debug(f"Encoder: {' '.join(encode_cmd)}")

    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    state = initial_state
    frame_idx = 0

    try:
        while True:
            # Read frame from decoder
            raw_frame = decoder.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break

            # Convert to numpy
            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((info.height, info.width, bytes_per_pixel))

            # Build per-frame params
            frame_params = dict(params)
            for param_name, values in bindings.items():
                if frame_idx < len(values):
                    frame_params[param_name] = values[frame_idx]

            # Process frame
            processed, state = process_frame(frame, frame_params, state)

            # Ensure correct shape and dtype
            if processed.shape != frame.shape:
                raise ValueError(
                    f"Frame shape mismatch: {processed.shape} vs {frame.shape}"
                )
            processed = processed.astype(np.uint8)

            # Write to encoder
            encoder.stdin.write(processed.tobytes())
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.debug(f"Processed frame {frame_idx}")

    except Exception as e:
        logger.error(f"Frame processing failed at frame {frame_idx}: {e}")
        raise
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        stderr = encoder.stderr.read().decode() if encoder.stderr else ""
        raise RuntimeError(f"Encoder failed: {stderr}")

    logger.info(f"Processed {frame_idx} frames")
    return output_path, state


def process_video_batch(
    input_path: Path,
    output_path: Path,
    process_frames: Callable[[List[np.ndarray], Dict[str, Any]], List[np.ndarray]],
    params: Dict[str, Any],
    batch_size: int = 30,
    pixel_format: str = "rgb24",
    output_codec: str = "libx264",
) -> Path:
    """
    Process video in batches for effects that need temporal context.

    Args:
        input_path: Input video path
        output_path: Output video path
        process_frames: Function (frames_batch, params) -> processed_batch
        params: Parameter dict
        batch_size: Number of frames per batch
        pixel_format: Pixel format
        output_codec: Output codec

    Returns:
        Output path
    """
    info = probe_video(input_path)

    if pixel_format == "rgb24":
        bytes_per_pixel = 3
    elif pixel_format == "rgba":
        bytes_per_pixel = 4
    else:
        bytes_per_pixel = 3

    frame_size = info.width * info.height * bytes_per_pixel

    decode_cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
        "-",
    ]

    encode_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", pixel_format,
        "-s", f"{info.width}x{info.height}",
        "-r", str(info.frame_rate),
        "-i", "-",
        "-i", str(input_path),
        "-map", "0:v",
        "-map", "1:a?",
        "-c:v", output_codec,
        "-c:a", "aac",
        str(output_path),
    ]

    decoder = subprocess.Popen(
        decode_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    encoder = subprocess.Popen(
        encode_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    batch = []
    total_processed = 0

    try:
        while True:
            raw_frame = decoder.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                # Process remaining batch
                if batch:
                    processed = process_frames(batch, params)
                    for frame in processed:
                        encoder.stdin.write(frame.astype(np.uint8).tobytes())
                        total_processed += 1
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((info.height, info.width, bytes_per_pixel))
            batch.append(frame)

            if len(batch) >= batch_size:
                processed = process_frames(batch, params)
                for frame in processed:
                    encoder.stdin.write(frame.astype(np.uint8).tobytes())
                    total_processed += 1
                batch = []

    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()

    if encoder.returncode != 0:
        stderr = encoder.stderr.read().decode() if encoder.stderr else ""
        raise RuntimeError(f"Encoder failed: {stderr}")

    logger.info(f"Processed {total_processed} frames in batches of {batch_size}")
    return output_path
