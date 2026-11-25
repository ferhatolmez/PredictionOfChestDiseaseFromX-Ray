"""Run both from-scratch algorithms on one X-ray and save a comparison figure.

Usage:
    python -m from_scratch.demo data/PNEUMONIA/<some_image>.jpeg
    python -m from_scratch.demo path/to/xray.png --out assets/from_scratch_demo.png

OpenCV is used only to read the file and as a reference for comparison;
the algorithms themselves are pure NumPy.
"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from from_scratch import compute_histogram, equalize_histogram, sobel_edges


def main():
    """Parse arguments, run both algorithms and save the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="path to a chest X-ray image")
    parser.add_argument("--out", default="assets/from_scratch_demo.png")
    args = parser.parse_args()

    gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"could not read image: {args.image}")
    gray = cv2.resize(gray, (512, 512))

    eq = equalize_histogram(gray)
    edges, _ = sobel_edges(eq)

    # sanity check against OpenCV
    diff = np.abs(eq.astype(int) - cv2.equalizeHist(gray).astype(int)).max()
    print(f"max |ours - cv2.equalizeHist| = {diff}")

    fig, ax = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    for col, (img, title) in enumerate(
        [
            (gray, "Original"),
            (eq, "Histogram equalized (NumPy)"),
            (edges, "Sobel edges (NumPy)"),
        ]
    ):
        ax[0, col].imshow(img, cmap="gray")
        ax[0, col].set_title(title)
        ax[0, col].axis("off")
    ax[1, 0].bar(range(256), compute_histogram(gray), width=1)
    ax[1, 0].set_title("Histogram before")
    ax[1, 1].bar(range(256), compute_histogram(eq), width=1)
    ax[1, 1].set_title("Histogram after")
    for a in ax[1, :2]:
        a.set_xlabel("Gray level")
        a.set_ylabel("Pixel count")
    ax[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
