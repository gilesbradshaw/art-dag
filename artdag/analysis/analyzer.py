# artdag/analysis/analyzer.py
"""
Main Analyzer class for the Analysis phase.

Coordinates audio and video feature extraction with caching.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .schema import AnalysisResult, AudioFeatures, VideoFeatures
from .audio import analyze_audio, FEATURE_ALL as AUDIO_ALL
from .video import analyze_video, FEATURE_ALL as VIDEO_ALL

logger = logging.getLogger(__name__)


class AnalysisCache:
    """
    Simple file-based cache for analysis results.

    Stores results as JSON files keyed by analysis cache_id.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, cache_id: str) -> Path:
        """Get cache file path for a cache_id."""
        return self.cache_dir / f"{cache_id}.json"

    def get(self, cache_id: str) -> Optional[AnalysisResult]:
        """Retrieve cached analysis result."""
        path = self._path_for(cache_id)
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return AnalysisResult.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load analysis cache {cache_id}: {e}")
            return None

    def put(self, result: AnalysisResult) -> None:
        """Store analysis result in cache."""
        path = self._path_for(result.cache_id)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def has(self, cache_id: str) -> bool:
        """Check if analysis result is cached."""
        return self._path_for(cache_id).exists()

    def remove(self, cache_id: str) -> bool:
        """Remove cached analysis result."""
        path = self._path_for(cache_id)
        if path.exists():
            path.unlink()
            return True
        return False


class Analyzer:
    """
    Analyzes media inputs to extract features.

    The Analyzer is the first phase of the 3-phase execution model.
    It extracts features from inputs that inform downstream processing.

    Example:
        analyzer = Analyzer(cache_dir=Path("./analysis_cache"))

        # Analyze a music file for beats
        result = analyzer.analyze(
            input_path=Path("/path/to/music.mp3"),
            input_hash="abc123...",
            features=["beats", "energy"]
        )

        print(f"Tempo: {result.tempo} BPM")
        print(f"Beats: {result.beat_times}")
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        content_cache: Optional["Cache"] = None,  # artdag.Cache for input lookup
    ):
        """
        Initialize the Analyzer.

        Args:
            cache_dir: Directory for analysis cache. If None, no caching.
            content_cache: artdag Cache for looking up inputs by hash
        """
        self.cache = AnalysisCache(cache_dir) if cache_dir else None
        self.content_cache = content_cache

    def get_input_path(self, input_hash: str, input_path: Optional[Path] = None) -> Path:
        """
        Resolve input to a file path.

        Args:
            input_hash: Content hash of the input
            input_path: Optional direct path to file

        Returns:
            Path to the input file

        Raises:
            ValueError: If input cannot be resolved
        """
        if input_path and input_path.exists():
            return input_path

        if self.content_cache:
            entry = self.content_cache.get(input_hash)
            if entry:
                return Path(entry.output_path)

        raise ValueError(f"Cannot resolve input {input_hash}: no path provided and not in cache")

    def analyze(
        self,
        input_hash: str,
        features: List[str],
        input_path: Optional[Path] = None,
        media_type: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Analyze an input file and extract features.

        Args:
            input_hash: Content hash of the input (for cache key)
            features: List of features to extract:
                Audio: "beats", "tempo", "energy", "spectrum", "onsets"
                Video: "metadata", "motion_tempo", "scene_changes"
                Meta: "all" (extracts all relevant features)
            input_path: Optional direct path to file
            media_type: Optional hint ("audio", "video", or None for auto-detect)

        Returns:
            AnalysisResult with extracted features
        """
        # Compute cache ID
        temp_result = AnalysisResult(
            input_hash=input_hash,
            features_requested=sorted(features),
        )
        cache_id = temp_result.cache_id

        # Check cache
        if self.cache and self.cache.has(cache_id):
            cached = self.cache.get(cache_id)
            if cached:
                logger.info(f"Analysis cache hit: {cache_id[:16]}...")
                return cached

        # Resolve input path
        path = self.get_input_path(input_hash, input_path)
        logger.info(f"Analyzing {path} for features: {features}")

        # Detect media type if not specified
        if media_type is None:
            media_type = self._detect_media_type(path)

        # Extract features
        audio_features = None
        video_features = None

        # Normalize features
        if "all" in features:
            audio_features_list = [AUDIO_ALL]
            video_features_list = [VIDEO_ALL]
        else:
            audio_features_list = [f for f in features if f in ("beats", "tempo", "energy", "spectrum", "onsets")]
            video_features_list = [f for f in features if f in ("metadata", "motion_tempo", "scene_changes")]

        if media_type in ("audio", "video") and audio_features_list:
            try:
                audio_features = analyze_audio(path, features=audio_features_list)
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")

        if media_type == "video" and video_features_list:
            try:
                video_features = analyze_video(path, features=video_features_list)
            except Exception as e:
                logger.warning(f"Video analysis failed: {e}")

        result = AnalysisResult(
            input_hash=input_hash,
            features_requested=sorted(features),
            audio=audio_features,
            video=video_features,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Cache result
        if self.cache:
            self.cache.put(result)

        return result

    def analyze_multiple(
        self,
        inputs: Dict[str, Path],
        features: List[str],
    ) -> Dict[str, AnalysisResult]:
        """
        Analyze multiple inputs.

        Args:
            inputs: Dict mapping input_hash to file path
            features: Features to extract from all inputs

        Returns:
            Dict mapping input_hash to AnalysisResult
        """
        results = {}
        for input_hash, input_path in inputs.items():
            try:
                results[input_hash] = self.analyze(
                    input_hash=input_hash,
                    features=features,
                    input_path=input_path,
                )
            except Exception as e:
                logger.error(f"Analysis failed for {input_hash}: {e}")
                raise

        return results

    def _detect_media_type(self, path: Path) -> str:
        """
        Detect if file is audio or video.

        Args:
            path: Path to media file

        Returns:
            "audio" or "video"
        """
        import subprocess
        import json

        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            has_video = any(s.get("codec_type") == "video" for s in streams)
            has_audio = any(s.get("codec_type") == "audio" for s in streams)

            if has_video:
                return "video"
            elif has_audio:
                return "audio"
            else:
                return "unknown"

        except (subprocess.CalledProcessError, json.JSONDecodeError):
            # Fall back to extension-based detection
            ext = path.suffix.lower()
            if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                return "video"
            elif ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
                return "audio"
            return "unknown"
