# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
@effect invert
@version 1.0.0
@author @giles@artdag.rose-ash.com
@temporal false

@description
Inverts video colors. Each pixel value is subtracted from 255,
creating a photographic negative effect.

@param intensity float
  @range 0.0 1.0
  @default 1.0
  Blend between original (0.0) and fully inverted (1.0).

@example
  (fx invert)

@example
  (fx invert :intensity 0.5)

@example
  (fx invert :intensity (bind analysis :energy :range [0.3 1.0]))
"""

import numpy as np

from artdag.effects import EffectMeta, ParamSpec

META = EffectMeta(
    name="invert",
    version="1.0.0",
    temporal=False,
    params=[
        ParamSpec("intensity", float, default=1.0, range=(0.0, 1.0)),
    ],
)


def process_frame(frame: np.ndarray, params: dict, state) -> tuple:
    """
    Invert frame colors.

    Args:
        frame: RGB frame as numpy array (H, W, 3)
        params: {"intensity": 0.0-1.0}
        state: Unused, passed through

    Returns:
        (inverted_frame, state)
    """
    intensity = params.get("intensity", 1.0)

    # Invert: 255 - pixel
    inverted = 255 - frame

    # Blend based on intensity
    if intensity < 1.0:
        result = (frame * (1 - intensity) + inverted * intensity).astype(np.uint8)
    else:
        result = inverted.astype(np.uint8)

    return result, state
