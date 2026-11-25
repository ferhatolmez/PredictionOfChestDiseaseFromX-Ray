"""Sobel edge detection written from scratch with NumPy only.

Idea
----
An edge is a place where brightness changes quickly. The Sobel operator
estimates the image gradient with two 3x3 kernels:

    Kx = [[-1, 0, 1],        Ky = [[-1, -2, -1],
          [-2, 0, 2],              [ 0,  0,  0],
          [-1, 0, 1]]              [ 1,  2,  1]]

Kx responds to horizontal changes (vertical edges), Ky to vertical changes
(horizontal edges). Each kernel is a central difference in one direction
combined with a [1, 2, 1] smoothing in the other, which makes it less
sensitive to noise than a plain difference. The edge strength is the gradient
magnitude sqrt(Gx^2 + Gy^2) and the edge direction is atan2(Gy, Gx).

Borders use reflect-101 padding (``np.pad(mode="reflect")``), the same default
as OpenCV, so Gx/Gy can be checked against ``cv2.Sobel`` (see tests/).
"""

import numpy as np

SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T.copy()


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D cross-correlation with reflect-101 padding; output has the input's shape.

    Instead of looping over every pixel, we loop over the (few) kernel cells and
    add a shifted copy of the whole padded image weighted by that cell. For a
    3x3 kernel that is 9 vectorized array operations.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("kernel size must be odd")

    ph, pw = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float64), ((ph, ph), (pw, pw)), mode="reflect")
    h, w = image.shape

    out = np.zeros((h, w), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


def sobel_gradients(image: np.ndarray):
    """Return (Gx, Gy) as float64 arrays."""
    return correlate2d(image, SOBEL_X), correlate2d(image, SOBEL_Y)


def sobel_edges(image: np.ndarray, normalize: bool = True):
    """Return (magnitude, direction_in_radians).

    If ``normalize`` is True the magnitude is scaled to uint8 0..255 so it can
    be displayed or saved directly.
    """
    gx, gy = sobel_gradients(image)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    if normalize:
        peak = magnitude.max()
        magnitude = (magnitude / peak * 255).astype(np.uint8) if peak > 0 \
            else np.zeros_like(magnitude, dtype=np.uint8)
    return magnitude, direction
