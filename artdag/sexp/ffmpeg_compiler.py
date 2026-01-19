"""
FFmpeg filter compiler for sexp effects.

Compiles sexp effect definitions to FFmpeg filter expressions,
with support for dynamic parameters via sendcmd scripts.

Usage:
    compiler = FFmpegCompiler()

    # Compile an effect with static params
    filter_str = compiler.compile_effect("brightness", {"amount": 50})
    # -> "eq=brightness=0.196"

    # Compile with dynamic binding to analysis data
    filter_str, sendcmd = compiler.compile_effect_with_binding(
        "brightness",
        {"amount": {"_bind": "bass-data", "range_min": 0, "range_max": 100}},
        analysis_data={"bass-data": {"times": [...], "values": [...]}},
        segment_start=0.0,
        segment_duration=5.0,
    )
    # -> ("eq=brightness=0.5", "0.0 [eq] brightness 0.5;\n0.05 [eq] brightness 0.6;...")
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# FFmpeg filter mappings for common effects
# Maps effect name -> {filter: str, params: {param_name: {ffmpeg_param, scale, offset}}}
EFFECT_MAPPINGS = {
    "invert": {
        "filter": "negate",
        "params": {},
    },
    "grayscale": {
        "filter": "colorchannelmixer",
        "static": "0.3:0.4:0.3:0:0.3:0.4:0.3:0:0.3:0.4:0.3",
        "params": {},
    },
    "sepia": {
        "filter": "colorchannelmixer",
        "static": "0.393:0.769:0.189:0:0.349:0.686:0.168:0:0.272:0.534:0.131",
        "params": {},
    },
    "brightness": {
        "filter": "eq",
        "params": {
            "amount": {"ffmpeg_param": "brightness", "scale": 1/255, "offset": 0},
        },
    },
    "contrast": {
        "filter": "eq",
        "params": {
            "amount": {"ffmpeg_param": "contrast", "scale": 1.0, "offset": 0},
        },
    },
    "saturation": {
        "filter": "eq",
        "params": {
            "amount": {"ffmpeg_param": "saturation", "scale": 1.0, "offset": 0},
        },
    },
    "hue_shift": {
        "filter": "hue",
        "params": {
            "degrees": {"ffmpeg_param": "h", "scale": 1.0, "offset": 0},
        },
    },
    "blur": {
        "filter": "gblur",
        "params": {
            "radius": {"ffmpeg_param": "sigma", "scale": 1.0, "offset": 0},
        },
    },
    "sharpen": {
        "filter": "unsharp",
        "params": {
            "amount": {"ffmpeg_param": "la", "scale": 1.0, "offset": 0},
        },
    },
    "pixelate": {
        # Scale down then up to create pixelation effect
        "filter": "scale",
        "static": "iw/8:ih/8:flags=neighbor,scale=iw*8:ih*8:flags=neighbor",
        "params": {},
    },
    "vignette": {
        "filter": "vignette",
        "params": {
            "strength": {"ffmpeg_param": "a", "scale": 1.0, "offset": 0},
        },
    },
    "noise": {
        "filter": "noise",
        "params": {
            "amount": {"ffmpeg_param": "alls", "scale": 1.0, "offset": 0},
        },
    },
    "flip": {
        "filter": "hflip",  # Default horizontal
        "params": {},
    },
    "mirror": {
        "filter": "hflip",
        "params": {},
    },
    "rotate": {
        "filter": "rotate",
        "params": {
            "angle": {"ffmpeg_param": "a", "scale": math.pi/180, "offset": 0},  # degrees to radians
        },
    },
    "zoom": {
        "filter": "zoompan",
        "params": {
            "factor": {"ffmpeg_param": "z", "scale": 1.0, "offset": 0},
        },
    },
    "posterize": {
        # Use lutyuv to quantize levels (approximate posterization)
        "filter": "lutyuv",
        "static": "y=floor(val/32)*32:u=floor(val/32)*32:v=floor(val/32)*32",
        "params": {},
    },
    "threshold": {
        # Use geq for thresholding
        "filter": "geq",
        "static": "lum='if(gt(lum(X,Y),128),255,0)':cb=128:cr=128",
        "params": {},
    },
    "edge_detect": {
        "filter": "edgedetect",
        "params": {
            "low": {"ffmpeg_param": "low", "scale": 1/255, "offset": 0},
            "high": {"ffmpeg_param": "high", "scale": 1/255, "offset": 0},
        },
    },
    "swirl": {
        "filter": "lenscorrection",  # Approximate with lens distortion
        "params": {
            "strength": {"ffmpeg_param": "k1", "scale": 0.1, "offset": 0},
        },
    },
    "fisheye": {
        "filter": "lenscorrection",
        "params": {
            "strength": {"ffmpeg_param": "k1", "scale": 1.0, "offset": 0},
        },
    },
    "wave": {
        # Wave displacement using geq - need r/g/b for RGB mode
        "filter": "geq",
        "static": "r='r(X+10*sin(Y/20),Y)':g='g(X+10*sin(Y/20),Y)':b='b(X+10*sin(Y/20),Y)'",
        "params": {},
    },
    "rgb_split": {
        # Chromatic aberration using geq
        "filter": "geq",
        "static": "r='p(X+5,Y)':g='p(X,Y)':b='p(X-5,Y)'",
        "params": {},
    },
    "scanlines": {
        "filter": "drawgrid",
        "params": {
            "spacing": {"ffmpeg_param": "h", "scale": 1.0, "offset": 0},
        },
    },
    "film_grain": {
        "filter": "noise",
        "params": {
            "intensity": {"ffmpeg_param": "alls", "scale": 100, "offset": 0},
        },
    },
    "crt": {
        "filter": "vignette",  # Simplified - just vignette for CRT look
        "params": {},
    },
    "bloom": {
        "filter": "gblur",  # Simplified bloom = blur overlay
        "params": {
            "radius": {"ffmpeg_param": "sigma", "scale": 1.0, "offset": 0},
        },
    },
    "color_cycle": {
        "filter": "hue",
        "params": {
            "speed": {"ffmpeg_param": "h", "scale": 360.0, "offset": 0, "time_expr": True},
        },
        "time_based": True,  # Uses time expression
    },
    "strobe": {
        # Strobe using select to drop frames
        "filter": "select",
        "static": "'mod(n,4)'",
        "params": {},
    },
    "echo": {
        # Echo using tmix
        "filter": "tmix",
        "static": "frames=4:weights='1 0.5 0.25 0.125'",
        "params": {},
    },
    "trails": {
        # Trails using tblend
        "filter": "tblend",
        "static": "all_mode=average",
        "params": {},
    },
    "kaleidoscope": {
        # 4-way mirror kaleidoscope using FFmpeg filter chain
        # Crops top-left quadrant, mirrors horizontally, then vertically
        "filter": "crop",
        "complex": True,
        "static": "iw/2:ih/2:0:0[q];[q]split[q1][q2];[q1]hflip[qr];[q2][qr]hstack[top];[top]split[t1][t2];[t2]vflip[bot];[t1][bot]vstack",
        "params": {},
    },
    "emboss": {
        "filter": "convolution",
        "static": "-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2:-2 -1 0 -1 1 1 0 1 2",
        "params": {},
    },
    "neon_glow": {
        # Edge detect + negate for neon-like effect
        "filter": "edgedetect",
        "static": "mode=colormix:high=0.1",
        "params": {},
    },
    "ascii_art": {
        # Requires Python frame processing - no FFmpeg equivalent
        "filter": None,
        "python_primitive": "ascii_art_frame",
        "params": {
            "char_size": {"default": 8},
            "alphabet": {"default": "standard"},
            "color_mode": {"default": "color"},
        },
    },
    "ascii_zones": {
        # Requires Python frame processing - zone-based ASCII
        "filter": None,
        "python_primitive": "ascii_zones_frame",
        "params": {
            "char_size": {"default": 8},
            "zone_threshold": {"default": 128},
        },
    },
    "datamosh": {
        # External tool: ffglitch or datamoshing CLI, falls back to Python
        "filter": None,
        "external_tool": "datamosh",
        "python_primitive": "datamosh_frame",
        "params": {
            "block_size": {"default": 32},
            "corruption": {"default": 0.3},
        },
    },
    "pixelsort": {
        # External tool: pixelsort CLI (Rust or Python), falls back to Python
        "filter": None,
        "external_tool": "pixelsort",
        "python_primitive": "pixelsort_frame",
        "params": {
            "sort_by": {"default": "lightness"},
            "threshold_low": {"default": 50},
            "threshold_high": {"default": 200},
            "angle": {"default": 0},
        },
    },
    "ripple": {
        # Use geq for ripple displacement
        "filter": "geq",
        "static": "lum='lum(X+5*sin(hypot(X-W/2,Y-H/2)/10),Y+5*cos(hypot(X-W/2,Y-H/2)/10))'",
        "params": {},
    },
    "tile_grid": {
        # Use tile filter for grid
        "filter": "tile",
        "static": "2x2",
        "params": {},
    },
    "outline": {
        "filter": "edgedetect",
        "params": {},
    },
    "color-adjust": {
        "filter": "eq",
        "params": {
            "brightness": {"ffmpeg_param": "brightness", "scale": 1/255, "offset": 0},
            "contrast": {"ffmpeg_param": "contrast", "scale": 1.0, "offset": 0},
        },
    },
}


class FFmpegCompiler:
    """Compiles sexp effects to FFmpeg filters with sendcmd support."""

    def __init__(self, effect_mappings: Dict = None):
        self.mappings = effect_mappings or EFFECT_MAPPINGS

    def get_full_mapping(self, effect_name: str) -> Optional[Dict]:
        """Get full mapping for an effect (including external tools and python primitives)."""
        mapping = self.mappings.get(effect_name)
        if not mapping:
            # Try with underscores/hyphens converted
            normalized = effect_name.replace("-", "_").replace(" ", "_").lower()
            mapping = self.mappings.get(normalized)
        return mapping

    def get_mapping(self, effect_name: str) -> Optional[Dict]:
        """Get FFmpeg filter mapping for an effect (returns None for non-FFmpeg effects)."""
        mapping = self.get_full_mapping(effect_name)
        # Return None if no mapping or no FFmpeg filter
        if mapping and mapping.get("filter") is None:
            return None
        return mapping

    def has_external_tool(self, effect_name: str) -> Optional[str]:
        """Check if effect uses an external tool. Returns tool name or None."""
        mapping = self.get_full_mapping(effect_name)
        if mapping:
            return mapping.get("external_tool")
        return None

    def has_python_primitive(self, effect_name: str) -> Optional[str]:
        """Check if effect uses a Python primitive. Returns primitive name or None."""
        mapping = self.get_full_mapping(effect_name)
        if mapping:
            return mapping.get("python_primitive")
        return None

    def is_complex_filter(self, effect_name: str) -> bool:
        """Check if effect uses a complex filter chain."""
        mapping = self.get_full_mapping(effect_name)
        return bool(mapping and mapping.get("complex"))

    def compile_effect(
        self,
        effect_name: str,
        params: Dict[str, Any],
    ) -> Optional[str]:
        """
        Compile an effect to an FFmpeg filter string with static params.

        Returns None if effect has no FFmpeg mapping.
        """
        mapping = self.get_mapping(effect_name)
        if not mapping:
            return None

        filter_name = mapping["filter"]

        # Handle static filters (no params)
        if "static" in mapping:
            return f"{filter_name}={mapping['static']}"

        if not mapping.get("params"):
            return filter_name

        # Build param string
        filter_params = []
        for param_name, param_config in mapping["params"].items():
            if param_name in params:
                value = params[param_name]
                # Skip if it's a binding (handled separately)
                if isinstance(value, dict) and ("_bind" in value or "_binding" in value):
                    continue
                ffmpeg_param = param_config["ffmpeg_param"]
                scale = param_config.get("scale", 1.0)
                offset = param_config.get("offset", 0)
                # Handle various value types
                if isinstance(value, (int, float)):
                    ffmpeg_value = value * scale + offset
                    filter_params.append(f"{ffmpeg_param}={ffmpeg_value:.4f}")
                elif isinstance(value, str):
                    filter_params.append(f"{ffmpeg_param}={value}")
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    ffmpeg_value = value[0] * scale + offset
                    filter_params.append(f"{ffmpeg_param}={ffmpeg_value:.4f}")

        if filter_params:
            return f"{filter_name}={':'.join(filter_params)}"
        return filter_name

    def compile_effect_with_bindings(
        self,
        effect_name: str,
        params: Dict[str, Any],
        analysis_data: Dict[str, Dict],
        segment_start: float,
        segment_duration: float,
        sample_interval: float = 0.04,  # ~25 fps
    ) -> Tuple[Optional[str], Optional[str], List[str]]:
        """
        Compile an effect with dynamic bindings to a filter + sendcmd script.

        Returns:
            (filter_string, sendcmd_script, bound_param_names)
            - filter_string: Initial FFmpeg filter (may have placeholder values)
            - sendcmd_script: Script content for sendcmd filter
            - bound_param_names: List of params that have bindings
        """
        mapping = self.get_mapping(effect_name)
        if not mapping:
            return None, None, []

        filter_name = mapping["filter"]
        static_params = []
        bound_params = []
        sendcmd_lines = []

        # Handle time-based effects (use FFmpeg expressions with 't')
        if mapping.get("time_based"):
            for param_name, param_config in mapping.get("params", {}).items():
                if param_name in params:
                    value = params[param_name]
                    ffmpeg_param = param_config["ffmpeg_param"]
                    scale = param_config.get("scale", 1.0)
                    if isinstance(value, (int, float)):
                        # Create time expression: h='t*speed*scale'
                        static_params.append(f"{ffmpeg_param}='t*{value}*{scale}'")
                    else:
                        static_params.append(f"{ffmpeg_param}='t*{scale}'")
            if static_params:
                filter_str = f"{filter_name}={':'.join(static_params)}"
            else:
                filter_str = f"{filter_name}=h='t*360'"  # Default rotation
            return filter_str, None, []

        # Process each param
        for param_name, param_config in mapping.get("params", {}).items():
            if param_name not in params:
                continue

            value = params[param_name]
            ffmpeg_param = param_config["ffmpeg_param"]
            scale = param_config.get("scale", 1.0)
            offset = param_config.get("offset", 0)

            # Check if it's a binding
            if isinstance(value, dict) and ("_bind" in value or "_binding" in value):
                bind_ref = value.get("_bind") or value.get("_binding")
                range_min = value.get("range_min", 0.0)
                range_max = value.get("range_max", 1.0)
                transform = value.get("transform")

                # Get analysis data
                analysis = analysis_data.get(bind_ref)
                if not analysis:
                    # Try without -data suffix
                    analysis = analysis_data.get(bind_ref.replace("-data", ""))

                if analysis and "times" in analysis and "values" in analysis:
                    times = analysis["times"]
                    values = analysis["values"]

                    # Generate sendcmd entries for this segment
                    segment_end = segment_start + segment_duration
                    t = 0.0  # Time relative to segment start

                    while t < segment_duration:
                        abs_time = segment_start + t

                        # Find analysis value at this time
                        raw_value = self._interpolate_value(times, values, abs_time)

                        # Apply transform
                        if transform == "sqrt":
                            raw_value = math.sqrt(max(0, raw_value))
                        elif transform == "pow2":
                            raw_value = raw_value ** 2
                        elif transform == "log":
                            raw_value = math.log(max(0.001, raw_value))

                        # Map to range
                        mapped_value = range_min + raw_value * (range_max - range_min)

                        # Apply FFmpeg scaling
                        ffmpeg_value = mapped_value * scale + offset

                        # Add sendcmd line (time relative to segment)
                        sendcmd_lines.append(f"{t:.3f} [{filter_name}] {ffmpeg_param} {ffmpeg_value:.4f};")

                        t += sample_interval

                    bound_params.append(param_name)
                    # Use initial value for the filter string
                    initial_value = self._interpolate_value(times, values, segment_start)
                    initial_mapped = range_min + initial_value * (range_max - range_min)
                    initial_ffmpeg = initial_mapped * scale + offset
                    static_params.append(f"{ffmpeg_param}={initial_ffmpeg:.4f}")
                else:
                    # No analysis data, use range midpoint
                    mid_value = (range_min + range_max) / 2
                    ffmpeg_value = mid_value * scale + offset
                    static_params.append(f"{ffmpeg_param}={ffmpeg_value:.4f}")
            else:
                # Static value - handle various types
                if isinstance(value, (int, float)):
                    ffmpeg_value = value * scale + offset
                    static_params.append(f"{ffmpeg_param}={ffmpeg_value:.4f}")
                elif isinstance(value, str):
                    # String value - use as-is (e.g., for direction parameters)
                    static_params.append(f"{ffmpeg_param}={value}")
                elif isinstance(value, list) and value:
                    # List - try to use first numeric element
                    first = value[0]
                    if isinstance(first, (int, float)):
                        ffmpeg_value = first * scale + offset
                        static_params.append(f"{ffmpeg_param}={ffmpeg_value:.4f}")
                # Skip other types

        # Handle static filters
        if "static" in mapping:
            filter_str = f"{filter_name}={mapping['static']}"
        elif static_params:
            filter_str = f"{filter_name}={':'.join(static_params)}"
        else:
            filter_str = filter_name

        # Combine sendcmd lines
        sendcmd_script = "\n".join(sendcmd_lines) if sendcmd_lines else None

        return filter_str, sendcmd_script, bound_params

    def _interpolate_value(
        self,
        times: List[float],
        values: List[float],
        target_time: float,
    ) -> float:
        """Interpolate a value from analysis data at a given time."""
        if not times or not values:
            return 0.5

        # Find surrounding indices
        if target_time <= times[0]:
            return values[0]
        if target_time >= times[-1]:
            return values[-1]

        # Binary search for efficiency
        lo, hi = 0, len(times) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if times[mid] <= target_time:
                lo = mid
            else:
                hi = mid

        # Linear interpolation
        t0, t1 = times[lo], times[hi]
        v0, v1 = values[lo], values[hi]

        if t1 == t0:
            return v0

        alpha = (target_time - t0) / (t1 - t0)
        return v0 + alpha * (v1 - v0)


def generate_sendcmd_filter(
    effects: List[Dict],
    analysis_data: Dict[str, Dict],
    segment_start: float,
    segment_duration: float,
) -> Tuple[str, Optional[Path]]:
    """
    Generate FFmpeg filter chain with sendcmd for dynamic effects.

    Args:
        effects: List of effect configs with name and params
        analysis_data: Analysis data keyed by name
        segment_start: Segment start time in source
        segment_duration: Segment duration

    Returns:
        (filter_chain_string, sendcmd_file_path or None)
    """
    import tempfile

    compiler = FFmpegCompiler()
    filters = []
    all_sendcmd_lines = []

    for effect in effects:
        effect_name = effect.get("effect")
        params = {k: v for k, v in effect.items() if k != "effect"}

        filter_str, sendcmd, _ = compiler.compile_effect_with_bindings(
            effect_name,
            params,
            analysis_data,
            segment_start,
            segment_duration,
        )

        if filter_str:
            filters.append(filter_str)
            if sendcmd:
                all_sendcmd_lines.append(sendcmd)

    if not filters:
        return "", None

    filter_chain = ",".join(filters)

    # NOTE: sendcmd disabled - FFmpeg's sendcmd filter has compatibility issues.
    # Bindings use their initial value (sampled at segment start time).
    # This is acceptable since each segment is only ~8 seconds.
    # The binding value is still music-reactive (varies per segment).
    sendcmd_path = None

    return filter_chain, sendcmd_path
