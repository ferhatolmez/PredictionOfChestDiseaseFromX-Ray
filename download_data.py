"""Download the Kaggle chest X-ray dataset and flatten it into ``data/<CLASS>/``.

The Kaggle archive is split into ``train/``, ``test/`` and ``val/`` folders.
The notebook makes its own stratified train/validation/test split, so all
images are merged into one folder per class.

Usage:
    pip install kaggle
    python download_data.py

The Kaggle API needs a token in ``~/.kaggle/kaggle.json``.
"""

import shutil
import subprocess
from pathlib import Path

DATASET = "jtiptj/chest-xray-pneumoniacovid19tuberculosis"
DATA_DIR = Path("data")
CLASSES = ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS")

# Folder names found in the archive -> class name used in the notebook.
# "TURBERCULOSIS" is a spelling mistake in the original dataset.
FOLDER_TO_CLASS = {
    "COVID19": "COVID19",
    "COVID-19": "COVID19",
    "NORMAL": "NORMAL",
    "PNEUMONIA": "PNEUMONIA",
    "TUBERCULOSIS": "TUBERCULOSIS",
    "TURBERCULOSIS": "TUBERCULOSIS",
}


def is_ready(data_dir: Path = DATA_DIR) -> bool:
    """Return True if every class folder exists and contains images."""
    return all(
        (data_dir / name).is_dir() and any((data_dir / name).iterdir())
        for name in CLASSES
    )


def download(data_dir: Path = DATA_DIR) -> None:
    """Download and unzip the dataset with the Kaggle CLI."""
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )


def flatten(data_dir: Path = DATA_DIR) -> dict[str, int]:
    """Move every image into ``data_dir/<CLASS>/`` and return the counts."""
    for name in CLASSES:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    targets = {data_dir / name for name in CLASSES}

    for folder in list(data_dir.rglob("*")):
        class_name = FOLDER_TO_CLASS.get(folder.name.upper())
        if not folder.is_dir() or class_name is None or folder in targets:
            continue
        split = folder.parent.name  # train / test / val
        for image in folder.iterdir():
            if not image.is_file():
                continue
            target = data_dir / class_name / image.name
            if target.exists():  # same file name in two splits
                target = target.with_name(f"{split}_{image.name}")
            shutil.move(str(image), str(target))

    # remove the now-empty split folders and any leftover files
    for item in data_dir.iterdir():
        if item in targets:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    return {name: len(list((data_dir / name).iterdir())) for name in CLASSES}


def main() -> None:
    """Download the dataset unless it is already in place."""
    if is_ready():
        print(f"Dataset already available in '{DATA_DIR}/'.")
        return
    print("Downloading dataset from Kaggle...")
    download()
    print("Flattening folder structure...")
    counts = flatten()
    for name, count in counts.items():
        print(f"  {name:<13} {count:>5} images")
    print(f"Done: {sum(counts.values())} images in '{DATA_DIR}/'.")


if __name__ == "__main__":
    main()
