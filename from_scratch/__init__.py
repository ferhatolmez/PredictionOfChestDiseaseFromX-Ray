"""Classic image-processing algorithms implemented from scratch with NumPy."""

from .histogram_equalization import compute_histogram, compute_cdf, equalize_histogram
from .sobel import correlate2d, sobel_gradients, sobel_edges

__all__ = [
    "compute_histogram",
    "compute_cdf",
    "equalize_histogram",
    "correlate2d",
    "sobel_gradients",
    "sobel_edges",
]
