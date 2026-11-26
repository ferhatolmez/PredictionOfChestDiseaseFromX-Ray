"""Tests for chestxray.preprocessing, chestxray.data and chestxray.audit."""

from pathlib import Path

import numpy as np
import pytest

from chestxray.data import patient_id, split_by_patient
from chestxray.duplicates import average_hash, find_duplicates, merge_groups
from chestxray.preprocessing import white_balance

RNG = np.random.default_rng(0)


def reference_white_balance_channel(channel, perc=0.05):
    """Stretch one channel like the original notebook did (reference)."""
    mi, ma = np.percentile(channel, perc), np.percentile(channel, 100 - perc)
    return np.uint8(np.clip((channel - mi) * 255.0 / (ma - mi), 0, 255))


# ---------- white balance ----------


def test_white_balance_matches_original_notebook():
    images = RNG.integers(40, 200, size=(3, 32, 32, 3), dtype=np.uint8)
    expected = np.stack(
        [
            np.dstack([reference_white_balance_channel(img[..., c]) for c in range(3)])
            for img in images
        ]
    )
    assert np.array_equal(white_balance(images), expected)


def test_white_balance_single_image_and_constant_image():
    flat = np.full((16, 16, 3), 90, dtype=np.uint8)
    out = white_balance(flat)  # max == min must not divide by zero
    assert out.shape == flat.shape and out.dtype == np.uint8


# ---------- patient IDs ----------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("data/PNEUMONIA/person1_bacteria_1.jpeg", "PNEUMONIA/person1"),
        ("data/PNEUMONIA/person1_virus_6.jpeg", "PNEUMONIA/person1"),
        ("data/NORMAL/IM-0605-0001.jpeg", "NORMAL/IM-0605"),
        ("data/NORMAL/NORMAL2-IM-0359-0001.jpeg", "NORMAL/NORMAL2-IM-0359"),
        ("data/NORMAL/test_IM-0001-0001.jpeg", "NORMAL/test_IM-0001"),
        ("data/COVID19/COVID19(486).jpg", "COVID19/COVID19(486).jpg"),
    ],
)
def test_patient_id(path, expected):
    assert patient_id(Path(path)) == expected


# ---------- split ----------


def make_labels_and_groups(n_patients=300):
    labels, groups = [], []
    for p in range(n_patients):
        label = ["A", "B", "C"][p % 3]
        for _ in range(RNG.integers(1, 4)):  # 1-3 images per patient
            labels.append(label)
            groups.append(f"patient{p}")
    return np.array(labels), np.array(groups)


def test_split_keeps_each_patient_in_one_set():
    labels, groups = make_labels_and_groups()
    split = split_by_patient(labels, groups, 0.15, 0.15, seed=0)
    sets = [set(groups[s]) for s in (split.train, split.val, split.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_split_covers_everything_once_with_all_classes():
    labels, groups = make_labels_and_groups()
    split = split_by_patient(labels, groups, 0.15, 0.15, seed=0)
    all_idx = np.concatenate([split.train, split.val, split.test])
    assert sorted(all_idx) == list(range(len(labels)))
    for part in (split.train, split.val, split.test):
        assert set(labels[part]) == {"A", "B", "C"}
    assert 0.10 < len(split.test) / len(labels) < 0.20


# ---------- duplicates ----------


def test_average_hash_finds_rescaled_copy():
    image = RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    brighter = np.clip(image.astype(int) + 10, 0, 255).astype(np.uint8)
    other = RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    hashes = [average_hash(image), average_hash(brighter), average_hash(other)]
    assert find_duplicates(hashes) == [[0, 1]]


def test_merge_groups_joins_patients_of_duplicates():
    groups = np.array(["p1", "p1", "p2", "p3"])
    merged = merge_groups(groups, [[1, 2]])  # image 1 (p1) == image 2 (p2)
    assert merged[0] == merged[1] == merged[2]
    assert merged[3] == "p3"
