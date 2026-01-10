# artdag/analysis/schema.py
"""
Data structures for analysis results.

Analysis extracts features from input media that inform downstream processing.
Results are cached by: analysis_cache_id = SHA3-256(input_hash + sorted(features))
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def _stable_hash(data: Any, algorithm: str = "sha3_256") -> str:
    """Create stable hash from arbitrary data."""
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"))
    hasher = hashlib.new(algorithm)
    hasher.update(json_str.encode())
    return hasher.hexdigest()


@dataclass
class BeatInfo:
    """
    Beat detection results.

    Attributes:
        beat_times: List of beat positions in seconds
        tempo: Estimated tempo in BPM
        confidence: Tempo detection confidence (0-1)
        downbeat_times: First beat of each bar (if detected)
        time_signature: Detected or assumed time signature (e.g., 4)
    """
    beat_times: List[float]
    tempo: float
    confidence: float = 1.0
    downbeat_times: Optional[List[float]] = None
    time_signature: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "beat_times": self.beat_times,
            "tempo": self.tempo,
            "confidence": self.confidence,
            "downbeat_times": self.downbeat_times,
            "time_signature": self.time_signature,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeatInfo":
        return cls(
            beat_times=data["beat_times"],
            tempo=data["tempo"],
            confidence=data.get("confidence", 1.0),
            downbeat_times=data.get("downbeat_times"),
            time_signature=data.get("time_signature", 4),
        )


@dataclass
class EnergyEnvelope:
    """
    Energy (loudness) over time.

    Attributes:
        times: Time points in seconds
        values: Energy values (0-1, normalized)
        window_ms: Analysis window size in milliseconds
    """
    times: List[float]
    values: List[float]
    window_ms: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": self.times,
            "values": self.values,
            "window_ms": self.window_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EnergyEnvelope":
        return cls(
            times=data["times"],
            values=data["values"],
            window_ms=data.get("window_ms", 50.0),
        )

    def at_time(self, t: float) -> float:
        """Interpolate energy value at given time."""
        if not self.times:
            return 0.0
        if t <= self.times[0]:
            return self.values[0]
        if t >= self.times[-1]:
            return self.values[-1]

        # Binary search for bracketing indices
        lo, hi = 0, len(self.times) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.times[mid] <= t:
                lo = mid
            else:
                hi = mid

        # Linear interpolation
        t0, t1 = self.times[lo], self.times[hi]
        v0, v1 = self.values[lo], self.values[hi]
        alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
        return v0 + alpha * (v1 - v0)


@dataclass
class SpectrumBands:
    """
    Frequency band envelopes over time.

    Attributes:
        bass: Low frequency envelope (20-200 Hz typical)
        mid: Mid frequency envelope (200-2000 Hz typical)
        high: High frequency envelope (2000-20000 Hz typical)
        times: Time points in seconds
        band_ranges: Frequency ranges for each band in Hz
    """
    bass: List[float]
    mid: List[float]
    high: List[float]
    times: List[float]
    band_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "bass": (20, 200),
        "mid": (200, 2000),
        "high": (2000, 20000),
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bass": self.bass,
            "mid": self.mid,
            "high": self.high,
            "times": self.times,
            "band_ranges": self.band_ranges,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectrumBands":
        return cls(
            bass=data["bass"],
            mid=data["mid"],
            high=data["high"],
            times=data["times"],
            band_ranges=data.get("band_ranges", {
                "bass": (20, 200),
                "mid": (200, 2000),
                "high": (2000, 20000),
            }),
        )


@dataclass
class AudioFeatures:
    """
    All extracted audio features.

    Attributes:
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
        beats: Beat detection results
        energy: Energy envelope
        spectrum: Frequency band envelopes
        onsets: Note/sound onset times
    """
    duration: float
    sample_rate: int
    channels: int
    beats: Optional[BeatInfo] = None
    energy: Optional[EnergyEnvelope] = None
    spectrum: Optional[SpectrumBands] = None
    onsets: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "beats": self.beats.to_dict() if self.beats else None,
            "energy": self.energy.to_dict() if self.energy else None,
            "spectrum": self.spectrum.to_dict() if self.spectrum else None,
            "onsets": self.onsets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioFeatures":
        return cls(
            duration=data["duration"],
            sample_rate=data["sample_rate"],
            channels=data["channels"],
            beats=BeatInfo.from_dict(data["beats"]) if data.get("beats") else None,
            energy=EnergyEnvelope.from_dict(data["energy"]) if data.get("energy") else None,
            spectrum=SpectrumBands.from_dict(data["spectrum"]) if data.get("spectrum") else None,
            onsets=data.get("onsets"),
        )


@dataclass
class VideoFeatures:
    """
    Extracted video features.

    Attributes:
        duration: Video duration in seconds
        frame_rate: Frames per second
        width: Frame width in pixels
        height: Frame height in pixels
        codec: Video codec name
        motion_tempo: Estimated tempo from motion analysis (optional)
        scene_changes: Times of detected scene changes
    """
    duration: float
    frame_rate: float
    width: int
    height: int
    codec: str = ""
    motion_tempo: Optional[float] = None
    scene_changes: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "frame_rate": self.frame_rate,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "motion_tempo": self.motion_tempo,
            "scene_changes": self.scene_changes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoFeatures":
        return cls(
            duration=data["duration"],
            frame_rate=data["frame_rate"],
            width=data["width"],
            height=data["height"],
            codec=data.get("codec", ""),
            motion_tempo=data.get("motion_tempo"),
            scene_changes=data.get("scene_changes"),
        )


@dataclass
class AnalysisResult:
    """
    Complete analysis result for an input.

    Combines audio and video features with metadata for caching.

    Attributes:
        input_hash: Content hash of the analyzed input
        features_requested: List of features that were requested
        audio: Audio features (if input has audio)
        video: Video features (if input has video)
        cache_id: Computed cache ID for this analysis
        analyzed_at: Timestamp of analysis
    """
    input_hash: str
    features_requested: List[str]
    audio: Optional[AudioFeatures] = None
    video: Optional[VideoFeatures] = None
    cache_id: Optional[str] = None
    analyzed_at: Optional[str] = None

    def __post_init__(self):
        """Compute cache_id if not provided."""
        if self.cache_id is None:
            self.cache_id = self._compute_cache_id()

    def _compute_cache_id(self) -> str:
        """
        Compute cache ID from input hash and requested features.

        cache_id = SHA3-256(input_hash + sorted(features_requested))
        """
        content = {
            "input_hash": self.input_hash,
            "features": sorted(self.features_requested),
        }
        return _stable_hash(content)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_hash": self.input_hash,
            "features_requested": self.features_requested,
            "audio": self.audio.to_dict() if self.audio else None,
            "video": self.video.to_dict() if self.video else None,
            "cache_id": self.cache_id,
            "analyzed_at": self.analyzed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        return cls(
            input_hash=data["input_hash"],
            features_requested=data["features_requested"],
            audio=AudioFeatures.from_dict(data["audio"]) if data.get("audio") else None,
            video=VideoFeatures.from_dict(data["video"]) if data.get("video") else None,
            cache_id=data.get("cache_id"),
            analyzed_at=data.get("analyzed_at"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "AnalysisResult":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # Convenience accessors
    @property
    def tempo(self) -> Optional[float]:
        """Get tempo if beats were analyzed."""
        return self.audio.beats.tempo if self.audio and self.audio.beats else None

    @property
    def beat_times(self) -> Optional[List[float]]:
        """Get beat times if beats were analyzed."""
        return self.audio.beats.beat_times if self.audio and self.audio.beats else None

    @property
    def downbeat_times(self) -> Optional[List[float]]:
        """Get downbeat times if analyzed."""
        return self.audio.beats.downbeat_times if self.audio and self.audio.beats else None

    @property
    def duration(self) -> float:
        """Get duration from video or audio."""
        if self.video:
            return self.video.duration
        if self.audio:
            return self.audio.duration
        return 0.0

    @property
    def dimensions(self) -> Optional[Tuple[int, int]]:
        """Get video dimensions if available."""
        if self.video:
            return (self.video.width, self.video.height)
        return None
