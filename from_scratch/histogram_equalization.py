"""Histogram equalization written from scratch with NumPy only.

Idea
----
A low-contrast X-ray uses only a narrow band of gray levels. Histogram
equalization spreads those levels over the full 0-255 range by mapping every
gray level through the image's cumulative distribution function (CDF):

    new_level(v) = round( (CDF(v) - CDF_min) / (N - CDF_min) * 255 )

where N is the number of pixels and CDF_min is the CDF value of the darkest
gray level that actually appears in the image. Frequent gray levels are pushed
apart and rare ones squeezed together, which increases global contrast.

The implementation follows the same rule as ``cv2.equalizeHist`` so the
result can be checked pixel by pixel against OpenCV (see tests/).
"""

import numpy as np


def compute_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """Count how many pixels have each gray level (0..bins-1)."""
    _check_gray_uint8(image)
    hist = np.zeros(bins, dtype=np.int64)
    # np.add.at accumulates repeated indices correctly (hist[idx] += 1 would not)
    np.add.at(hist, image.ravel(), 1)
    return hist


def compute_cdf(hist: np.ndarray) -> np.ndarray:
    """Cumulative distribution function: cdf[v] = number of pixels <= v."""
    return np.cumsum(hist)


def equalization_lut(image: np.ndarray) -> np.ndarray:
    """Build the 256-entry lookup table that maps old gray levels to new ones."""
    hist = compute_histogram(image)
    cdf = compute_cdf(hist)
    total = cdf[-1]

    first = int(np.flatnonzero(hist)[0])  # darkest gray level that appears
    cdf_min = cdf[first]

    if cdf_min == total:  # constant image: nothing to stretch
        return np.full(256, first, dtype=np.uint8)

    # float32 + round-half-to-even mirrors OpenCV's arithmetic exactly
    scale = np.float32(255.0) / np.float32(total - cdf_min)
    lut = np.rint((cdf - cdf_min).astype(np.float32) * scale)
    lut[:first] = 0
    return np.clip(lut, 0, 255).astype(np.uint8)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Return the histogram-equalized version of a grayscale uint8 image."""
    lut = equalization_lut(image)
    return lut[image]  # fancy indexing applies the LUT to every pixel at once


def _check_gray_uint8(image: np.ndarray) -> None:
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"expected dtype uint8, got {image.dtype}")
