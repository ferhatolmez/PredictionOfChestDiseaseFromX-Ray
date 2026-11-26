"""Dataset checks: patients, duplicate images and non-medical shortcuts.

The four classes come from different public datasets. If a simple model can
tell the classes apart from things like image size or brightness alone, a CNN
may do the same, and a high test score may partly measure *where* an image
comes from rather than *what disease* it shows ("shortcut learning").

Usage:
    python -m chestxray.audit
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from chestxray.config import Config
from chestxray.data import XrayData, load_dataset
from chestxray.duplicates import duplicate_sets


def non_medical_features(data: XrayData) -> np.ndarray:
    """Return features that say nothing about disease.

    Columns: original height, width, aspect ratio (not visible to the CNN,
    which sees 224x224), then mean brightness, contrast and border brightness
    of the 224x224 image (visible to the CNN).
    """
    heights, widths = data.shapes[:, 0], data.shapes[:, 1]
    gray = data.images.mean(axis=3, dtype=np.float32)  # float32 halves the memory
    border = np.concatenate(
        [gray[:, :8].reshape(len(gray), -1), gray[:, -8:].reshape(len(gray), -1)],
        axis=1,
    )
    return np.column_stack(
        [
            heights,
            widths,
            widths / heights,
            gray.mean(axis=(1, 2)),
            gray.std(axis=(1, 2)),
            border.mean(axis=1),
        ]
    )


def shortcut_probe(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> float:
    """Return the mean 5-fold accuracy of a random forest on the given features."""
    forest = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(forest, features, labels, cv=folds).mean())


def run_audit(data: XrayData) -> dict:
    """Collect class sizes, patients, image sizes, duplicates and the probes."""
    duplicates = duplicate_sets(data.images)
    cross_class = [g for g in duplicates if len({data.labels[i] for i in g}) > 1]
    cross_patient = [g for g in duplicates if len({data.groups[i] for i in g}) > 1]

    per_class = {}
    for name in sorted(set(data.labels)):
        mask = data.labels == name
        patients = Counter(data.groups[mask])
        per_class[name] = {
            "images": int(mask.sum()),
            "patients": len(patients),
            "max_images_per_patient": max(patients.values()),
            "file_types": dict(
                Counter(Path(p).suffix.lower() for p in data.paths[mask])
            ),
            "median_height": int(np.median(data.shapes[mask, 0])),
            "median_width": int(np.median(data.shapes[mask, 1])),
        }

    features = non_medical_features(data)
    return {
        "per_class": per_class,
        "duplicate_sets": len(duplicates),
        "images_in_duplicate_sets": sum(len(g) for g in duplicates),
        "duplicate_sets_across_patients": len(cross_patient),
        "duplicate_sets_across_classes": len(cross_class),
        "duplicate_examples_across_classes": [
            [str(data.paths[i]) for i in g] for g in cross_class
        ],
        "majority_class_accuracy": max(Counter(data.labels).values())
        / len(data.labels),
        "probe_accuracy_size_only": shortcut_probe(features[:, :3], data.labels),
        "probe_accuracy_brightness_only": shortcut_probe(features[:, 3:], data.labels),
        "probe_accuracy_size_and_brightness": shortcut_probe(features, data.labels),
    }


def main() -> None:
    """Run the audit and save it to results/audit.json."""
    report = run_audit(load_dataset(Config()))
    out = Path("results/audit.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
