"""
Frame processing primitives for sexp effects.

These primitives are called by sexp effect definitions and operate on
numpy arrays (frames). They're used when falling back to Python rendering
instead of FFmpeg.

Required: numpy, PIL
"""

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ASCII character sets for different styles
ASCII_ALPHABETS = {
    "standard": " .:-=+*#%@",
    "blocks": " ░▒▓█",
    "simple": " .-:+=xX#",
    "detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "binary": " █",
}


def check_deps():
    """Check that required dependencies are available."""
    if not HAS_NUMPY:
        raise ImportError("numpy required for frame primitives: pip install numpy")
    if not HAS_PIL:
        raise ImportError("PIL required for frame primitives: pip install Pillow")


def frame_to_image(frame: np.ndarray) -> Image.Image:
    """Convert numpy frame to PIL Image."""
    check_deps()
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def image_to_frame(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy frame."""
    check_deps()
    return np.array(img)


# ============================================================================
# ASCII Art Primitives
# ============================================================================

def cell_sample(frame: np.ndarray, cell_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample frame into cells, returning average colors and luminances.

    Args:
        frame: Input frame (H, W, C)
        cell_size: Size of each cell

    Returns:
        (colors, luminances) - colors is (rows, cols, 3), luminances is (rows, cols)
    """
    check_deps()
    h, w = frame.shape[:2]
    rows = h // cell_size
    cols = w // cell_size

    colors = np.zeros((rows, cols, 3), dtype=np.float32)
    luminances = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_size, (r + 1) * cell_size
            x0, x1 = c * cell_size, (c + 1) * cell_size
            cell = frame[y0:y1, x0:x1]

            # Average color
            avg_color = cell.mean(axis=(0, 1))
            colors[r, c] = avg_color[:3]  # RGB only

            # Luminance (ITU-R BT.601)
            lum = 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
            luminances[r, c] = lum

    return colors, luminances


def luminance_to_chars(
    luminances: np.ndarray,
    alphabet: str = "standard",
    contrast: float = 1.0,
) -> List[List[str]]:
    """
    Convert luminance values to ASCII characters.

    Args:
        luminances: 2D array of luminance values (0-255)
        alphabet: Name of character set or custom string
        contrast: Contrast multiplier

    Returns:
        2D list of characters
    """
    check_deps()
    chars = ASCII_ALPHABETS.get(alphabet, alphabet)
    n_chars = len(chars)

    rows, cols = luminances.shape
    result = []

    for r in range(rows):
        row_chars = []
        for c in range(cols):
            lum = luminances[r, c]
            # Apply contrast around midpoint
            lum = 128 + (lum - 128) * contrast
            lum = np.clip(lum, 0, 255)
            # Map to character index
            idx = int(lum / 256 * n_chars)
            idx = min(idx, n_chars - 1)
            row_chars.append(chars[idx])
        result.append(row_chars)

    return result


def render_char_grid(
    frame: np.ndarray,
    chars: List[List[str]],
    colors: np.ndarray,
    char_size: int = 8,
    color_mode: str = "color",
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Render character grid to an image.

    Args:
        frame: Original frame (for dimensions)
        chars: 2D list of characters
        colors: Color for each cell (rows, cols, 3)
        char_size: Size of each character cell
        color_mode: "color", "white", or "green"
        background: Background RGB color

    Returns:
        Rendered frame
    """
    check_deps()
    h, w = frame.shape[:2]
    rows = len(chars)
    cols = len(chars[0]) if chars else 0

    # Create output image
    img = Image.new("RGB", (w, h), background)
    draw = ImageDraw.Draw(img)

    # Try to get a monospace font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", char_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", char_size)
        except (IOError, OSError):
            font = ImageFont.load_default()

    for r in range(rows):
        for c in range(cols):
            char = chars[r][c]
            if char == ' ':
                continue

            x = c * char_size
            y = r * char_size

            if color_mode == "color":
                color = tuple(int(v) for v in colors[r, c])
            elif color_mode == "green":
                color = (0, 255, 0)
            else:  # white
                color = (255, 255, 255)

            draw.text((x, y), char, fill=color, font=font)

    return np.array(img)


def ascii_art_frame(
    frame: np.ndarray,
    char_size: int = 8,
    alphabet: str = "standard",
    color_mode: str = "color",
    contrast: float = 1.5,
    background: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Apply ASCII art effect to a frame.

    This is the main entry point for the ascii_art effect.
    """
    check_deps()
    colors, luminances = cell_sample(frame, char_size)
    chars = luminance_to_chars(luminances, alphabet, contrast)
    return render_char_grid(frame, chars, colors, char_size, color_mode, background)


# ============================================================================
# ASCII Zones Primitives
# ============================================================================

def ascii_zones_frame(
    frame: np.ndarray,
    char_size: int = 8,
    zone_threshold: int = 128,
    dark_chars: str = " .-:",
    light_chars: str = "=+*#",
) -> np.ndarray:
    """
    Apply zone-based ASCII art effect.

    Different character sets for dark vs light regions.
    """
    check_deps()
    colors, luminances = cell_sample(frame, char_size)

    rows, cols = luminances.shape
    chars = []

    for r in range(rows):
        row_chars = []
        for c in range(cols):
            lum = luminances[r, c]
            if lum < zone_threshold:
                # Dark zone
                charset = dark_chars
                local_lum = lum / zone_threshold  # 0-1 within zone
            else:
                # Light zone
                charset = light_chars
                local_lum = (lum - zone_threshold) / (255 - zone_threshold)

            idx = int(local_lum * len(charset))
            idx = min(idx, len(charset) - 1)
            row_chars.append(charset[idx])
        chars.append(row_chars)

    return render_char_grid(frame, chars, colors, char_size, "color", (0, 0, 0))


# ============================================================================
# Kaleidoscope Primitives (Python fallback)
# ============================================================================

def kaleidoscope_displace(
    w: int,
    h: int,
    segments: int = 6,
    rotation: float = 0,
    cx: float = None,
    cy: float = None,
    zoom: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute kaleidoscope displacement coordinates.

    Returns (x_coords, y_coords) arrays for remapping.
    """
    check_deps()
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    # Center coordinates
    x_centered = x_grid - cx
    y_centered = y_grid - cy

    # Convert to polar
    r = np.sqrt(x_centered**2 + y_centered**2) / zoom
    theta = np.arctan2(y_centered, x_centered)

    # Apply rotation
    theta = theta - np.radians(rotation)

    # Kaleidoscope: fold angle into segment
    segment_angle = 2 * np.pi / segments
    theta = np.abs(np.mod(theta, segment_angle) - segment_angle / 2)

    # Convert back to cartesian
    x_new = r * np.cos(theta) + cx
    y_new = r * np.sin(theta) + cy

    return x_new, y_new


def remap(
    frame: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> np.ndarray:
    """
    Remap frame using coordinate arrays.

    Uses bilinear interpolation.
    """
    check_deps()
    from scipy import ndimage

    h, w = frame.shape[:2]

    # Clip coordinates
    x_coords = np.clip(x_coords, 0, w - 1)
    y_coords = np.clip(y_coords, 0, h - 1)

    # Remap each channel
    if len(frame.shape) == 3:
        result = np.zeros_like(frame)
        for c in range(frame.shape[2]):
            result[:, :, c] = ndimage.map_coordinates(
                frame[:, :, c],
                [y_coords, x_coords],
                order=1,
                mode='reflect',
            )
        return result
    else:
        return ndimage.map_coordinates(frame, [y_coords, x_coords], order=1, mode='reflect')


def kaleidoscope_frame(
    frame: np.ndarray,
    segments: int = 6,
    rotation: float = 0,
    center_x: float = 0.5,
    center_y: float = 0.5,
    zoom: float = 1.0,
) -> np.ndarray:
    """
    Apply kaleidoscope effect to a frame.

    This is a Python fallback - FFmpeg version is faster.
    """
    check_deps()
    h, w = frame.shape[:2]
    cx = w * center_x
    cy = h * center_y

    x_coords, y_coords = kaleidoscope_displace(w, h, segments, rotation, cx, cy, zoom)
    return remap(frame, x_coords, y_coords)


# ============================================================================
# Datamosh Primitives (simplified Python version)
# ============================================================================

def datamosh_frame(
    frame: np.ndarray,
    prev_frame: Optional[np.ndarray],
    block_size: int = 32,
    corruption: float = 0.3,
    max_offset: int = 50,
    color_corrupt: bool = True,
) -> np.ndarray:
    """
    Simplified datamosh effect using block displacement.

    This is a basic approximation - real datamosh works on compressed video.
    """
    check_deps()
    if prev_frame is None:
        return frame.copy()

    h, w = frame.shape[:2]
    result = frame.copy()

    # Process in blocks
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if np.random.random() < corruption:
                # Random offset
                ox = np.random.randint(-max_offset, max_offset + 1)
                oy = np.random.randint(-max_offset, max_offset + 1)

                # Source from previous frame with offset
                src_y = np.clip(y + oy, 0, h - block_size)
                src_x = np.clip(x + ox, 0, w - block_size)

                block = prev_frame[src_y:src_y+block_size, src_x:src_x+block_size]

                # Color corruption
                if color_corrupt and np.random.random() < 0.3:
                    # Swap or shift channels
                    block = np.roll(block, np.random.randint(1, 3), axis=2)

                result[y:y+block_size, x:x+block_size] = block

    return result


# ============================================================================
# Pixelsort Primitives (Python version)
# ============================================================================

def pixelsort_frame(
    frame: np.ndarray,
    sort_by: str = "lightness",
    threshold_low: float = 50,
    threshold_high: float = 200,
    angle: float = 0,
    reverse: bool = False,
) -> np.ndarray:
    """
    Apply pixel sorting effect to a frame.
    """
    check_deps()
    from scipy import ndimage

    # Rotate if needed
    if angle != 0:
        frame = ndimage.rotate(frame, -angle, reshape=False, mode='reflect')

    h, w = frame.shape[:2]
    result = frame.copy()

    # Compute sort key
    if sort_by == "lightness":
        key = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
    elif sort_by == "hue":
        # Simple hue approximation
        key = np.arctan2(
            np.sqrt(3) * (frame[:,:,1].astype(float) - frame[:,:,2]),
            2 * frame[:,:,0].astype(float) - frame[:,:,1] - frame[:,:,2]
        )
    elif sort_by == "saturation":
        mx = frame.max(axis=2).astype(float)
        mn = frame.min(axis=2).astype(float)
        key = np.where(mx > 0, (mx - mn) / mx, 0)
    else:
        key = frame[:,:,0]  # Red channel

    # Sort each row
    for y in range(h):
        row = result[y]
        row_key = key[y]

        # Find sortable intervals (pixels within threshold)
        mask = (row_key >= threshold_low) & (row_key <= threshold_high)

        # Find runs of True in mask
        runs = []
        start = None
        for x in range(w):
            if mask[x] and start is None:
                start = x
            elif not mask[x] and start is not None:
                if x - start > 1:
                    runs.append((start, x))
                start = None
        if start is not None and w - start > 1:
            runs.append((start, w))

        # Sort each run
        for start, end in runs:
            indices = np.argsort(row_key[start:end])
            if reverse:
                indices = indices[::-1]
            result[y, start:end] = row[start:end][indices]

    # Rotate back
    if angle != 0:
        result = ndimage.rotate(result, angle, reshape=False, mode='reflect')

    return result


# ============================================================================
# Primitive Registry
# ============================================================================

def map_char_grid(
    chars,
    luminances,
    fn,
):
    """
    Map a function over each cell of a character grid.

    Args:
        chars: 2D array/list of characters (rows, cols)
        luminances: 2D array of luminance values
        fn: Function or Lambda (row, col, char, luminance) -> new_char

    Returns:
        New character grid with mapped values (list of lists)
    """
    from .parser import Lambda
    from .evaluator import evaluate

    # Handle both list and numpy array inputs
    if isinstance(chars, np.ndarray):
        rows, cols = chars.shape[:2]
    else:
        rows = len(chars)
        cols = len(chars[0]) if rows > 0 and isinstance(chars[0], (list, tuple, str)) else 1

    # Get luminances as 2D
    if isinstance(luminances, np.ndarray):
        lum_arr = luminances
    else:
        lum_arr = np.array(luminances)

    # Check if fn is a Lambda (from sexp) or a Python callable
    is_lambda = isinstance(fn, Lambda)

    result = []
    for r in range(rows):
        row_result = []
        for c in range(cols):
            # Get character
            if isinstance(chars, np.ndarray):
                ch = chars[r, c] if len(chars.shape) > 1 else chars[r]
            elif isinstance(chars[r], str):
                ch = chars[r][c] if c < len(chars[r]) else ' '
            else:
                ch = chars[r][c] if c < len(chars[r]) else ' '

            # Get luminance
            if len(lum_arr.shape) > 1:
                lum = lum_arr[r, c]
            else:
                lum = lum_arr[r]

            # Call the function
            if is_lambda:
                # Evaluate the Lambda with arguments bound
                call_env = dict(fn.closure) if fn.closure else {}
                for param, val in zip(fn.params, [r, c, ch, float(lum)]):
                    call_env[param] = val
                new_ch = evaluate(fn.body, call_env)
            else:
                new_ch = fn(r, c, ch, float(lum))

            row_result.append(new_ch)
        result.append(row_result)

    return result


def alphabet_char(alphabet: str, index: int) -> str:
    """
    Get a character from an alphabet at a given index.

    Args:
        alphabet: Alphabet name (from ASCII_ALPHABETS) or literal string
        index: Index into the alphabet (clamped to valid range)

    Returns:
        Character at the index
    """
    # Get alphabet string
    if alphabet in ASCII_ALPHABETS:
        chars = ASCII_ALPHABETS[alphabet]
    else:
        chars = alphabet

    # Clamp index to valid range
    index = int(index)
    index = max(0, min(index, len(chars) - 1))

    return chars[index]


PRIMITIVES = {
    # ASCII
    "cell-sample": cell_sample,
    "luminance-to-chars": luminance_to_chars,
    "render-char-grid": render_char_grid,
    "map-char-grid": map_char_grid,
    "alphabet-char": alphabet_char,
    "ascii_art_frame": ascii_art_frame,
    "ascii_zones_frame": ascii_zones_frame,

    # Kaleidoscope
    "kaleidoscope-displace": kaleidoscope_displace,
    "remap": remap,
    "kaleidoscope_frame": kaleidoscope_frame,

    # Datamosh
    "datamosh": datamosh_frame,
    "datamosh_frame": datamosh_frame,

    # Pixelsort
    "pixelsort": pixelsort_frame,
    "pixelsort_frame": pixelsort_frame,
}


def get_primitive(name: str):
    """Get a primitive function by name."""
    return PRIMITIVES.get(name)


def list_primitives() -> List[str]:
    """List all available primitives."""
    return list(PRIMITIVES.keys())
