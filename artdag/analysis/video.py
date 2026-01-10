# artdag/analysis/video.py
"""
Video feature extraction.

Uses ffprobe for basic metadata and optional OpenCV for motion analysis.
"""

import json
import logging
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import List, Optional

from .schema import VideoFeatures

logger = logging.getLogger(__name__)

# Feature names
FEATURE_METADATA = "metadata"
FEATURE_MOTION_TEMPO = "motion_tempo"
FEATURE_SCENE_CHANGES = "scene_changes"
FEATURE_ALL = "all"


def _parse_frame_rate(rate_str: str) -> float:
    """Parse frame rate string like '30000/1001' or '30'."""
    try:
        if "/" in rate_str:
            frac = Fraction(rate_str)
            return float(frac)
        return float(rate_str)
    except (ValueError, ZeroDivisionError):
        return 30.0  # Default


def analyze_metadata(path: Path) -> VideoFeatures:
    """
    Extract video metadata using ffprobe.

    Args:
        path: Path to video file

    Returns:
        VideoFeatures with basic metadata
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        "-select_streams", "v:0",
        str(path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        raise ValueError(f"Could not read video info: {e}")

    if not data.get("streams"):
        raise ValueError("No video stream found")

    stream = data["streams"][0]
    fmt = data.get("format", {})

    # Get duration from format or stream
    duration = float(fmt.get("duration", stream.get("duration", 0)))

    # Parse frame rate
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate", "30"))

    return VideoFeatures(
        duration=duration,
        frame_rate=frame_rate,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        codec=stream.get("codec_name", ""),
    )


def analyze_scene_changes(path: Path, threshold: float = 0.3) -> List[float]:
    """
    Detect scene changes using ffmpeg scene detection.

    Args:
        path: Path to video file
        threshold: Scene change threshold (0-1, lower = more sensitive)

    Returns:
        List of scene change times in seconds
    """
    cmd = [
        "ffmpeg", "-i", str(path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
    except subprocess.CalledProcessError as e:
        logger.warning(f"Scene detection failed: {e}")
        return []

    # Parse scene change times from ffmpeg output
    scene_times = []
    for line in stderr.split("\n"):
        if "pts_time:" in line:
            try:
                # Extract pts_time value
                for part in line.split():
                    if part.startswith("pts_time:"):
                        time_str = part.split(":")[1]
                        scene_times.append(float(time_str))
                        break
            except (ValueError, IndexError):
                continue

    return scene_times


def analyze_motion_tempo(path: Path, sample_duration: float = 30.0) -> Optional[float]:
    """
    Estimate tempo from video motion periodicity.

    Analyzes optical flow or frame differences to detect rhythmic motion.
    This is useful for matching video speed to audio tempo.

    Args:
        path: Path to video file
        sample_duration: Duration to analyze (seconds)

    Returns:
        Estimated motion tempo in BPM, or None if not detectable
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("OpenCV not available, skipping motion tempo analysis")
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {path}")
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        max_frames = int(sample_duration * fps)
        frame_diffs = []
        prev_gray = None

        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and resize for speed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))

            if prev_gray is not None:
                # Calculate frame difference
                diff = cv2.absdiff(gray, prev_gray)
                frame_diffs.append(np.mean(diff))

            prev_gray = gray
            frame_count += 1

        if len(frame_diffs) < 60:  # Need at least 2 seconds at 30fps
            return None

        # Convert to numpy array
        motion = np.array(frame_diffs)

        # Normalize
        motion = motion - motion.mean()
        if motion.std() > 0:
            motion = motion / motion.std()

        # Autocorrelation to find periodicity
        n = len(motion)
        acf = np.correlate(motion, motion, mode="full")[n-1:]
        acf = acf / acf[0]  # Normalize

        # Find peaks in autocorrelation (potential beat periods)
        # Look for periods between 0.3s (200 BPM) and 2s (30 BPM)
        min_lag = int(0.3 * fps)
        max_lag = min(int(2.0 * fps), len(acf) - 1)

        if max_lag <= min_lag:
            return None

        # Find the highest peak in the valid range
        search_range = acf[min_lag:max_lag]
        if len(search_range) == 0:
            return None

        peak_idx = np.argmax(search_range) + min_lag
        peak_value = acf[peak_idx]

        # Only report if peak is significant
        if peak_value < 0.1:
            return None

        # Convert lag to BPM
        period_seconds = peak_idx / fps
        bpm = 60.0 / period_seconds

        # Sanity check
        if 30 <= bpm <= 200:
            return round(bpm, 1)

        return None

    finally:
        cap.release()


def analyze_video(
    path: Path,
    features: Optional[List[str]] = None,
) -> VideoFeatures:
    """
    Extract video features from file.

    Args:
        path: Path to video file
        features: List of features to extract. Options:
            - "metadata": Basic video info (always included)
            - "motion_tempo": Estimated tempo from motion
            - "scene_changes": Scene change detection
            - "all": All features

    Returns:
        VideoFeatures with requested analysis
    """
    if features is None:
        features = [FEATURE_METADATA]

    if FEATURE_ALL in features:
        features = [FEATURE_METADATA, FEATURE_MOTION_TEMPO, FEATURE_SCENE_CHANGES]

    # Basic metadata is always extracted
    result = analyze_metadata(path)

    if FEATURE_MOTION_TEMPO in features:
        try:
            result.motion_tempo = analyze_motion_tempo(path)
        except Exception as e:
            logger.warning(f"Motion tempo analysis failed: {e}")

    if FEATURE_SCENE_CHANGES in features:
        try:
            result.scene_changes = analyze_scene_changes(path)
        except Exception as e:
            logger.warning(f"Scene change detection failed: {e}")

    return result
