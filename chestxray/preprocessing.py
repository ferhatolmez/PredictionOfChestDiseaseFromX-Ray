"""Image preprocessing: white balance (per-channel contrast stretch)."""

import numpy as np

CHUNK_SIZE = 256  # images per chunk, keeps the float64 copy of np.percentile small


def white_balance(images: np.ndarray, percentile: float = 0.05) -> np.ndarray:
    """Stretch each channel of each image to the full 0-255 range.

    For every channel, the value at ``percentile`` becomes 0 and the value at
    ``100 - percentile`` becomes 255; everything outside is clipped. This
    removes a gray cast and makes dark lungs and bright bone easier to tell
    apart.

    Args:
        images: uint8 array, one image ``(H, W, C)`` or a batch ``(N, H, W, C)``.
        percentile: percentile (0-100) clipped at each end.

    Returns:
        uint8 array of the same shape.
    """
    if images.ndim == 3:
        return white_balance(images[None], percentile)[0]
    if images.ndim != 4:
        raise ValueError(f"expected (H, W, C) or (N, H, W, C), got {images.shape}")

    out = np.empty_like(images, dtype=np.uint8)
    for start in range(0, len(images), CHUNK_SIZE):
        chunk = images[start : start + CHUNK_SIZE]
        # low/high have shape (n, 1, 1, C) so they broadcast over each image
        low, high = np.percentile(
            chunk, [percentile, 100 - percentile], axis=(1, 2), keepdims=True
        )
        scale = 255.0 / np.maximum(high - low, 1e-6)  # avoid dividing by zero
        stretched = np.clip((chunk - low) * scale, 0, 255)
        out[start : start + CHUNK_SIZE] = stretched.astype(np.uint8)
    return out
