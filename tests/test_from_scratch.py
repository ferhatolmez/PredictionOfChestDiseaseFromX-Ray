"""Check the from-scratch NumPy implementations against OpenCV.

Run with:  pytest tests/
"""

import cv2
import numpy as np
import pytest

from from_scratch import compute_histogram, equalize_histogram, sobel_gradients, sobel_edges

RNG = np.random.default_rng(0)


def low_contrast_image(h=128, w=96):
    """Random image squeezed into gray levels 60..120 (like a dull X-ray)."""
    return RNG.integers(60, 121, size=(h, w), dtype=np.uint8)


# ---------- histogram equalization ----------

def test_histogram_matches_numpy_bincount():
    img = low_contrast_image()
    assert np.array_equal(compute_histogram(img), np.bincount(img.ravel(), minlength=256))


@pytest.mark.parametrize("seed", range(5))
def test_equalize_matches_opencv(seed):
    rng = np.random.default_rng(seed)
    lo, hi = sorted(rng.integers(0, 256, size=2))
    img = rng.integers(lo, hi + 1, size=(100, 150), dtype=np.uint8)
    assert np.array_equal(equalize_histogram(img), cv2.equalizeHist(img))


def test_equalize_stretches_contrast():
    img = low_contrast_image()
    eq = equalize_histogram(img)
    assert eq.min() == 0 and eq.max() == 255


def test_equalize_constant_image_is_unchanged():
    img = np.full((10, 10), 77, dtype=np.uint8)
    assert np.array_equal(equalize_histogram(img), img)


def test_equalize_rejects_color_image():
    with pytest.raises(ValueError):
        equalize_histogram(np.zeros((4, 4, 3), dtype=np.uint8))


# ---------- Sobel ----------

def test_sobel_matches_opencv():
    img = RNG.integers(0, 256, size=(64, 80), dtype=np.uint8)
    gx, gy = sobel_gradients(img)
    assert np.allclose(gx, cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    assert np.allclose(gy, cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))


def test_sobel_finds_vertical_edge():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[:, 10:] = 255  # dark left half, bright right half
    mag, _ = sobel_edges(img)
    strongest_cols = np.unique(np.argwhere(mag == mag.max())[:, 1])
    assert set(strongest_cols) <= {9, 10}


def test_sobel_flat_image_has_no_edges():
    mag, _ = sobel_edges(np.full((16, 16), 128, dtype=np.uint8))
    assert mag.max() == 0
