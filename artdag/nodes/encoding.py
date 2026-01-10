# artdag/nodes/encoding.py
"""
Web-optimized video encoding settings.

Provides common FFmpeg arguments for producing videos that:
- Stream efficiently (faststart)
- Play on all browsers (H.264 High profile)
- Support seeking (regular keyframes)
"""

from typing import List

# Standard web-optimized video encoding arguments
WEB_VIDEO_ARGS: List[str] = [
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "18",
    "-profile:v", "high",
    "-level", "4.1",
    "-pix_fmt", "yuv420p",  # Ensure broad compatibility
    "-movflags", "+faststart",  # Enable streaming before full download
    "-g", "48",  # Keyframe every ~2 seconds at 24fps (for seeking)
]

# Standard audio encoding arguments
WEB_AUDIO_ARGS: List[str] = [
    "-c:a", "aac",
    "-b:a", "192k",
]


def get_web_encoding_args() -> List[str]:
    """Get FFmpeg args for web-optimized video+audio encoding."""
    return WEB_VIDEO_ARGS + WEB_AUDIO_ARGS


def get_web_video_args() -> List[str]:
    """Get FFmpeg args for web-optimized video encoding only."""
    return WEB_VIDEO_ARGS.copy()


def get_web_audio_args() -> List[str]:
    """Get FFmpeg args for web-optimized audio encoding only."""
    return WEB_AUDIO_ARGS.copy()


# For shell commands (string format)
WEB_VIDEO_ARGS_STR = " ".join(WEB_VIDEO_ARGS)
WEB_AUDIO_ARGS_STR = " ".join(WEB_AUDIO_ARGS)
WEB_ENCODING_ARGS_STR = f"{WEB_VIDEO_ARGS_STR} {WEB_AUDIO_ARGS_STR}"
