"""
Parameter binding resolution.

Resolves bind expressions to per-frame lookup tables at plan time.
Binding options:
  - :range [lo hi] - map 0-1 to output range
  - :smooth N - smoothing window in seconds
  - :offset N - time offset in seconds
  - :on-event V - value on discrete events
  - :decay N - exponential decay after event
  - :noise N - add deterministic noise (seeded)
  - :seed N - explicit RNG seed
"""

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class AnalysisData:
    """
    Analysis data for binding resolution.

    Attributes:
        frame_rate: Video frame rate
        total_frames: Total number of frames
        features: Dict mapping feature name to per-frame values
        events: Dict mapping event name to list of frame indices
    """

    frame_rate: float
    total_frames: int
    features: Dict[str, List[float]]  # feature -> [value_per_frame]
    events: Dict[str, List[int]]  # event -> [frame_indices]

    def get_feature(self, name: str, frame: int) -> float:
        """Get feature value at frame, interpolating if needed."""
        if name not in self.features:
            return 0.0
        values = self.features[name]
        if not values:
            return 0.0
        if frame >= len(values):
            return values[-1]
        return values[frame]

    def get_events_in_range(
        self, name: str, start_frame: int, end_frame: int
    ) -> List[int]:
        """Get event frames in range."""
        if name not in self.events:
            return []
        return [f for f in self.events[name] if start_frame <= f < end_frame]


@dataclass
class ResolvedBinding:
    """
    Resolved binding with per-frame values.

    Attributes:
        param_name: Parameter this binding applies to
        values: List of values, one per frame
    """

    param_name: str
    values: List[float]

    def get(self, frame: int) -> float:
        """Get value at frame."""
        if frame >= len(self.values):
            return self.values[-1] if self.values else 0.0
        return self.values[frame]


def resolve_binding(
    binding: Dict[str, Any],
    analysis: AnalysisData,
    param_name: str,
    cache_id: str = None,
) -> ResolvedBinding:
    """
    Resolve a binding specification to per-frame values.

    Args:
        binding: Binding spec with source, feature, and options
        analysis: Analysis data with features and events
        param_name: Name of the parameter being bound
        cache_id: Cache ID for deterministic seeding

    Returns:
        ResolvedBinding with values for each frame
    """
    feature = binding.get("feature")
    if not feature:
        raise ValueError(f"Binding for {param_name} missing feature")

    # Get base values
    values = []
    is_event = feature in analysis.events

    if is_event:
        # Event-based binding
        on_event = binding.get("on_event", 1.0)
        decay = binding.get("decay", 0.0)
        values = _resolve_event_binding(
            analysis.events.get(feature, []),
            analysis.total_frames,
            analysis.frame_rate,
            on_event,
            decay,
        )
    else:
        # Continuous feature binding
        feature_values = analysis.features.get(feature, [])
        if not feature_values:
            # No data, use zeros
            values = [0.0] * analysis.total_frames
        else:
            # Extend to total frames if needed
            values = list(feature_values)
            while len(values) < analysis.total_frames:
                values.append(values[-1] if values else 0.0)

    # Apply offset
    offset = binding.get("offset")
    if offset:
        offset_frames = int(offset * analysis.frame_rate)
        values = _apply_offset(values, offset_frames)

    # Apply smoothing
    smooth = binding.get("smooth")
    if smooth:
        window_frames = int(smooth * analysis.frame_rate)
        values = _apply_smoothing(values, window_frames)

    # Apply range mapping
    range_spec = binding.get("range")
    if range_spec:
        lo, hi = range_spec
        values = _apply_range(values, lo, hi)

    # Apply noise
    noise = binding.get("noise")
    if noise:
        seed = binding.get("seed")
        if seed is None and cache_id:
            # Derive seed from cache_id for determinism
            seed = int(hashlib.sha256(cache_id.encode()).hexdigest()[:8], 16)
        values = _apply_noise(values, noise, seed or 0)

    return ResolvedBinding(param_name=param_name, values=values)


def _resolve_event_binding(
    event_frames: List[int],
    total_frames: int,
    frame_rate: float,
    on_event: float,
    decay: float,
) -> List[float]:
    """
    Resolve event-based binding with optional decay.

    Args:
        event_frames: List of frame indices where events occur
        total_frames: Total number of frames
        frame_rate: Video frame rate
        on_event: Value at event
        decay: Decay time constant in seconds (0 = instant)

    Returns:
        List of values per frame
    """
    values = [0.0] * total_frames

    if not event_frames:
        return values

    event_set = set(event_frames)

    if decay <= 0:
        # No decay - just mark event frames
        for f in event_frames:
            if 0 <= f < total_frames:
                values[f] = on_event
    else:
        # Apply exponential decay
        decay_frames = decay * frame_rate
        for f in event_frames:
            if f < 0 or f >= total_frames:
                continue
            # Apply decay from this event forward
            for i in range(f, total_frames):
                elapsed = i - f
                decayed = on_event * math.exp(-elapsed / decay_frames)
                if decayed < 0.001:
                    break
                values[i] = max(values[i], decayed)

    return values


def _apply_offset(values: List[float], offset_frames: int) -> List[float]:
    """Shift values by offset frames (positive = delay)."""
    if offset_frames == 0:
        return values

    n = len(values)
    result = [0.0] * n

    for i in range(n):
        src = i - offset_frames
        if 0 <= src < n:
            result[i] = values[src]

    return result


def _apply_smoothing(values: List[float], window_frames: int) -> List[float]:
    """Apply moving average smoothing."""
    if window_frames <= 1:
        return values

    n = len(values)
    result = []
    half = window_frames // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        avg = sum(values[start:end]) / (end - start)
        result.append(avg)

    return result


def _apply_range(values: List[float], lo: float, hi: float) -> List[float]:
    """Map values from 0-1 to lo-hi range."""
    return [lo + v * (hi - lo) for v in values]


def _apply_noise(values: List[float], amount: float, seed: int) -> List[float]:
    """Add deterministic noise to values."""
    rng = random.Random(seed)
    return [v + rng.uniform(-amount, amount) for v in values]


def resolve_all_bindings(
    config: Dict[str, Any],
    analysis: AnalysisData,
    cache_id: str = None,
) -> Dict[str, ResolvedBinding]:
    """
    Resolve all bindings in a config dict.

    Looks for values with _binding: True marker.

    Args:
        config: Node config with potential bindings
        analysis: Analysis data
        cache_id: Cache ID for seeding

    Returns:
        Dict mapping param name to resolved binding
    """
    resolved = {}

    for key, value in config.items():
        if isinstance(value, dict) and value.get("_binding"):
            resolved[key] = resolve_binding(value, analysis, key, cache_id)

    return resolved


def bindings_to_lookup_table(
    bindings: Dict[str, ResolvedBinding],
) -> Dict[str, List[float]]:
    """
    Convert resolved bindings to simple lookup tables.

    Returns dict mapping param name to list of per-frame values.
    This format is JSON-serializable for inclusion in execution plans.
    """
    return {name: binding.values for name, binding in bindings.items()}


def has_bindings(config: Dict[str, Any]) -> bool:
    """Check if config contains any bindings."""
    for value in config.values():
        if isinstance(value, dict) and value.get("_binding"):
            return True
    return False


def extract_binding_sources(config: Dict[str, Any]) -> List[str]:
    """
    Extract all analysis source references from bindings.

    Returns list of node IDs that provide analysis data.
    """
    sources = []
    for value in config.values():
        if isinstance(value, dict) and value.get("_binding"):
            source = value.get("source")
            if source and source not in sources:
                sources.append(source)
    return sources
