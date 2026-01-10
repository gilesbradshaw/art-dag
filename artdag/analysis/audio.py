# artdag/analysis/audio.py
"""
Audio feature extraction.

Uses librosa for beat detection, energy analysis, and spectral features.
Falls back to basic ffprobe if librosa is not available.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .schema import AudioFeatures, BeatInfo, EnergyEnvelope, SpectrumBands

logger = logging.getLogger(__name__)

# Feature names for requesting specific analysis
FEATURE_BEATS = "beats"
FEATURE_TEMPO = "tempo"
FEATURE_ENERGY = "energy"
FEATURE_SPECTRUM = "spectrum"
FEATURE_ONSETS = "onsets"
FEATURE_ALL = "all"


def _get_audio_info_ffprobe(path: Path) -> Tuple[float, int, int]:
    """Get basic audio info using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a:0",
        str(path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        if not data.get("streams"):
            raise ValueError("No audio stream found")

        stream = data["streams"][0]
        duration = float(stream.get("duration", 0))
        sample_rate = int(stream.get("sample_rate", 44100))
        channels = int(stream.get("channels", 2))
        return duration, sample_rate, channels
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"ffprobe failed: {e}")
        raise ValueError(f"Could not read audio info: {e}")


def _extract_audio_to_wav(path: Path, duration: Optional[float] = None) -> Path:
    """Extract audio to temporary WAV file for librosa processing."""
    import tempfile
    wav_path = Path(tempfile.mktemp(suffix=".wav"))

    cmd = ["ffmpeg", "-y", "-i", str(path)]
    if duration:
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", "22050",  # Resample to 22050 Hz for librosa
        "-ac", "1",  # Mono
        str(wav_path)
    ])

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return wav_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio extraction failed: {e.stderr}")
        raise ValueError(f"Could not extract audio: {e}")


def analyze_beats(path: Path, sample_rate: int = 22050) -> BeatInfo:
    """
    Detect beats and tempo using librosa.

    Args:
        path: Path to audio file (or pre-extracted WAV)
        sample_rate: Sample rate for analysis

    Returns:
        BeatInfo with beat times, tempo, and confidence
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required for beat detection. Install with: pip install librosa")

    # Load audio
    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)

    # Detect tempo and beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # Convert frames to times
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Estimate confidence from onset strength consistency
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = onset_env[beat_frames] if len(beat_frames) > 0 else []
    confidence = float(beat_strength.mean() / onset_env.max()) if len(beat_strength) > 0 and onset_env.max() > 0 else 0.5

    # Detect downbeats (first beat of each bar)
    # Use beat phase to estimate bar positions
    downbeat_times = None
    if len(beat_times) >= 4:
        # Assume 4/4 time signature, downbeats every 4 beats
        downbeat_times = [beat_times[i] for i in range(0, len(beat_times), 4)]

    return BeatInfo(
        beat_times=beat_times,
        tempo=float(tempo) if hasattr(tempo, '__float__') else float(tempo[0]) if len(tempo) > 0 else 120.0,
        confidence=min(1.0, max(0.0, confidence)),
        downbeat_times=downbeat_times,
        time_signature=4,
    )


def analyze_energy(path: Path, window_ms: float = 50.0, sample_rate: int = 22050) -> EnergyEnvelope:
    """
    Extract energy (loudness) envelope.

    Args:
        path: Path to audio file
        window_ms: Analysis window size in milliseconds
        sample_rate: Sample rate for analysis

    Returns:
        EnergyEnvelope with times and normalized values
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise ImportError("librosa and numpy required. Install with: pip install librosa numpy")

    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)

    # Calculate frame size from window_ms
    hop_length = int(sr * window_ms / 1000)

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Normalize to 0-1
    rms_max = rms.max()
    if rms_max > 0:
        rms_normalized = rms / rms_max
    else:
        rms_normalized = rms

    # Generate time points
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    return EnergyEnvelope(
        times=times.tolist(),
        values=rms_normalized.tolist(),
        window_ms=window_ms,
    )


def analyze_spectrum(
    path: Path,
    band_ranges: Optional[dict] = None,
    window_ms: float = 50.0,
    sample_rate: int = 22050
) -> SpectrumBands:
    """
    Extract frequency band envelopes.

    Args:
        path: Path to audio file
        band_ranges: Dict mapping band name to (low_hz, high_hz)
        window_ms: Analysis window size
        sample_rate: Sample rate

    Returns:
        SpectrumBands with bass, mid, high envelopes
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        raise ImportError("librosa and numpy required")

    if band_ranges is None:
        band_ranges = {
            "bass": (20, 200),
            "mid": (200, 2000),
            "high": (2000, 20000),
        }

    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    hop_length = int(sr * window_ms / 1000)

    # Compute STFT
    n_fft = 2048
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    def band_energy(low_hz: float, high_hz: float) -> List[float]:
        """Sum energy in frequency band."""
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not mask.any():
            return [0.0] * stft.shape[1]
        band = stft[mask, :].sum(axis=0)
        # Normalize
        band_max = band.max()
        if band_max > 0:
            band = band / band_max
        return band.tolist()

    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)

    return SpectrumBands(
        bass=band_energy(*band_ranges["bass"]),
        mid=band_energy(*band_ranges["mid"]),
        high=band_energy(*band_ranges["high"]),
        times=times.tolist(),
        band_ranges=band_ranges,
    )


def analyze_onsets(path: Path, sample_rate: int = 22050) -> List[float]:
    """
    Detect onset times (note/sound starts).

    Args:
        path: Path to audio file
        sample_rate: Sample rate

    Returns:
        List of onset times in seconds
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa required")

    y, sr = librosa.load(str(path), sr=sample_rate, mono=True)

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times.tolist()


def analyze_audio(
    path: Path,
    features: Optional[List[str]] = None,
) -> AudioFeatures:
    """
    Extract audio features from file.

    Args:
        path: Path to audio/video file
        features: List of features to extract. Options:
            - "beats": Beat detection (tempo, beat times)
            - "energy": Loudness envelope
            - "spectrum": Frequency band envelopes
            - "onsets": Note onset times
            - "all": All features

    Returns:
        AudioFeatures with requested analysis
    """
    if features is None:
        features = [FEATURE_ALL]

    # Normalize features
    if FEATURE_ALL in features:
        features = [FEATURE_BEATS, FEATURE_ENERGY, FEATURE_SPECTRUM, FEATURE_ONSETS]

    # Get basic info via ffprobe
    duration, sample_rate, channels = _get_audio_info_ffprobe(path)

    result = AudioFeatures(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
    )

    # Check if librosa is available for advanced features
    try:
        import librosa  # noqa: F401
        has_librosa = True
    except ImportError:
        has_librosa = False
        if any(f in features for f in [FEATURE_BEATS, FEATURE_ENERGY, FEATURE_SPECTRUM, FEATURE_ONSETS]):
            logger.warning("librosa not available, skipping advanced audio features")

    if not has_librosa:
        return result

    # Extract audio to WAV for librosa
    wav_path = None
    try:
        wav_path = _extract_audio_to_wav(path)

        if FEATURE_BEATS in features or FEATURE_TEMPO in features:
            try:
                result.beats = analyze_beats(wav_path)
            except Exception as e:
                logger.warning(f"Beat detection failed: {e}")

        if FEATURE_ENERGY in features:
            try:
                result.energy = analyze_energy(wav_path)
            except Exception as e:
                logger.warning(f"Energy analysis failed: {e}")

        if FEATURE_SPECTRUM in features:
            try:
                result.spectrum = analyze_spectrum(wav_path)
            except Exception as e:
                logger.warning(f"Spectrum analysis failed: {e}")

        if FEATURE_ONSETS in features:
            try:
                result.onsets = analyze_onsets(wav_path)
            except Exception as e:
                logger.warning(f"Onset detection failed: {e}")

    finally:
        # Clean up temporary WAV file
        if wav_path and wav_path.exists():
            wav_path.unlink()

    return result
