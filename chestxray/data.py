"""Listing and loading images, patient IDs, and the patient-grouped split."""

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from chestxray.config import Config
from chestxray.preprocessing import white_balance

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Patient IDs in file names (only the pediatric NORMAL/PNEUMONIA source has them):
#   person123_bacteria_45.jpeg -> person123
#   IM-0605-0001.jpeg          -> IM-0605
#   NORMAL2-IM-0359-0001.jpeg  -> NORMAL2-IM-0359
# An optional "train_"/"test_"/"val_" prefix was added by download_data.py when
# two splits had the same file name, so those are kept as separate patients.
PATIENT_PATTERNS = (
    re.compile(r"^((?:train_|test_|val_)?person\d+)_"),
    re.compile(r"^((?:train_|test_|val_)?(?:NORMAL2-)?IM-\d+)-"),
)


@dataclass
class XrayData:
    """Images and everything known about them, row by row."""

    images: np.ndarray  # (N, size, size, 3) uint8, RGB, white balanced
    labels: np.ndarray  # (N,) class names
    groups: np.ndarray  # (N,) patient IDs
    shapes: np.ndarray  # (N, 2) original height and width
    paths: np.ndarray  # (N,) file paths


@dataclass
class Split:
    """Row indices of the three sets."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def list_images(
    data_dir: Path, classes: tuple[str, ...]
) -> tuple[list[Path], list[str]]:
    """Return the image paths and their class names, sorted for reproducibility."""
    paths, labels = [], []
    for name in classes:
        for path in sorted((Path(data_dir) / name).iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
                labels.append(name)
    return paths, labels


def patient_id(path: Path) -> str:
    """Return the patient ID in a file name, or the file name if it has none.

    The class folder is part of the ID, so equal names in two classes never
    become one patient.
    """
    path = Path(path)
    for pattern in PATIENT_PATTERNS:
        match = pattern.match(path.name)
        if match:
            return f"{path.parent.name}/{match.group(1)}"
    return f"{path.parent.name}/{path.name}"


def load_images(paths: list[Path], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read and resize images.

    Returns:
        ``images`` (N, size, size, 3) uint8 in RGB order, and ``shapes``
        (N, 2) with each original height and width. Unreadable files raise.
    """
    images = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    shapes = np.empty((len(paths), 2), dtype=np.int32)
    for i, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read image: {path}")
        shapes[i] = image.shape[:2]
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        images[i] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Keras models use RGB
    return images, shapes


def load_dataset(config: Config, use_cache: bool = True) -> XrayData:
    """Load, resize and white-balance all images, using a cache file if present.

    Reading and preprocessing 7,000 images takes minutes; the cache brings
    that down to seconds for every later run.
    """
    cache = Path(config.cache_file)
    if use_cache and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        return XrayData(**{key: stored[key] for key in stored.files})

    paths, labels = list_images(config.data_dir, config.classes)
    images, shapes = load_images(paths, config.image_size)
    data = XrayData(
        images=white_balance(images),
        labels=np.array(labels),
        groups=np.array([patient_id(p) for p in paths]),
        shapes=shapes,
        paths=np.array([str(p) for p in paths]),
    )
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **vars(data))
    return data


def split_by_patient(
    labels: np.ndarray,
    groups: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Split:
    """Split into train/validation/test, stratified by class and grouped by patient.

    All images of one patient end up in the same set, so the model is never
    tested on a patient it has already seen in training. Class proportions are
    kept as equal as the groups allow. ``StratifiedGroupKFold`` makes folds of
    equal size, so the fractions are rounded to ``1 / n_splits``.
    """
    indices = np.arange(len(labels))

    test_folds = StratifiedGroupKFold(
        n_splits=round(1 / test_fraction), shuffle=True, random_state=seed
    )
    rest, test = next(test_folds.split(indices, labels, groups))

    # validation is a fraction of the whole dataset, so of the rest it is larger
    val_of_rest = val_fraction / (1 - test_fraction)
    val_folds = StratifiedGroupKFold(
        n_splits=round(1 / val_of_rest), shuffle=True, random_state=seed
    )
    train_pos, val_pos = next(val_folds.split(rest, labels[rest], groups[rest]))
    return Split(train=rest[train_pos], val=rest[val_pos], test=test)
