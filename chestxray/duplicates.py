"""Finding near-identical images and keeping them in the same split.

Public datasets are often merged from other datasets, so the same X-ray can
appear twice under different names. If one copy is in the training set and
the other in the test set, the test score is too optimistic.
"""

import numpy as np

HASH_SIZE = 8


def average_hash(image: np.ndarray, hash_size: int = HASH_SIZE) -> bytes:
    """Return a perceptual hash: which blocks are brighter than the mean.

    The image is averaged down to ``hash_size x hash_size`` blocks; each block
    gives one bit. Resized or re-compressed copies of the same image get the
    same hash, unlike an exact file hash.
    """
    gray = image.mean(axis=2) if image.ndim == 3 else image
    h, w = gray.shape
    if h % hash_size or w % hash_size:
        raise ValueError(f"image size {gray.shape} is not divisible by {hash_size}")
    blocks = gray.reshape(hash_size, h // hash_size, hash_size, w // hash_size)
    small = blocks.mean(axis=(1, 3))
    return np.packbits(small > small.mean()).tobytes()


def find_duplicates(hashes: list[bytes]) -> list[list[int]]:
    """Group the indices of images that share a hash (groups of two or more)."""
    by_hash: dict[bytes, list[int]] = {}
    for index, value in enumerate(hashes):
        by_hash.setdefault(value, []).append(index)
    return [group for group in by_hash.values() if len(group) > 1]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the pixel values of two images (1.0 = same picture)."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def confirmed_duplicates(
    images: np.ndarray, groups: list[list[int]], threshold: float = 0.95
) -> list[list[int]]:
    """Keep the hash groups whose images are really near-identical.

    A 64-bit hash can collide for different images, so each image is compared
    with the first of its group pixel by pixel.
    """
    confirmed = []
    for group in groups:
        first = group[0]
        same = [
            i for i in group[1:] if correlation(images[first], images[i]) > threshold
        ]
        if same:
            confirmed.append([first, *same])
    return confirmed


def duplicate_sets(images: np.ndarray) -> list[list[int]]:
    """Return the sets of near-identical images (indices) in ``images``."""
    hashes = [average_hash(image) for image in images]
    return confirmed_duplicates(images, find_duplicates(hashes))


def merge_groups(groups: np.ndarray, duplicates: list[list[int]]) -> np.ndarray:
    """Give duplicate images, and all images of their patients, one group.

    Uses union-find: every duplicate set joins the groups of its members, so
    if image A of patient 1 equals image B of patient 2, all images of both
    patients end up in the same group.
    """
    parent: dict[str, str] = {}

    def find(group: str) -> str:
        while parent.get(group, group) != group:
            group = parent[group]
        return group

    for members in duplicates:
        root = find(groups[members[0]])
        for index in members[1:]:
            other = find(groups[index])
            if other != root:
                parent[other] = root
    return np.array([find(group) for group in groups])
