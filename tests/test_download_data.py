"""Check that download_data.flatten builds the folder layout the notebook expects."""

from download_data import CLASSES, flatten, is_ready


def make_fake_archive(root):
    """Create a tiny copy of the Kaggle layout: <split>/<folder>/<image>."""
    layout = {
        "train": ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"],
        "test": ["COVID19", "NORMAL", "PNEUMONIA", "TURBERCULOSIS"],
    }
    for split, folders in layout.items():
        for folder in folders:
            (root / split / folder).mkdir(parents=True)
            (root / split / folder / "img_1.jpeg").write_bytes(b"x")
    (root / "archive.zip").write_bytes(b"zip")


def test_flatten_merges_splits_and_fixes_spelling(tmp_path):
    make_fake_archive(tmp_path)
    counts = flatten(tmp_path)

    assert counts == {name: 2 for name in CLASSES}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(CLASSES)
    # duplicate file names from different splits are kept, not overwritten
    names = sorted(p.name for p in (tmp_path / "NORMAL").iterdir())
    assert len(names) == 2 and "img_1.jpeg" in names
    assert any(n in names for n in ("test_img_1.jpeg", "train_img_1.jpeg"))
    assert is_ready(tmp_path)


def test_is_ready_false_for_empty_folder(tmp_path):
    assert not is_ready(tmp_path)
