"""Two-phase training of the chest X-ray classifier.

Usage:
    python -m chestxray.train --seed 0
    python -m chestxray.train --seed 0 --limit 200 --head-epochs 1 --fine-tune-epochs 1

Writes to ``results/seed_<seed>/``: the best model, the training history,
the split and the test metrics.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from chestxray.config import Config
from chestxray.data import XrayData, load_dataset, split_by_patient
from chestxray.duplicates import duplicate_sets, merge_groups
from chestxray.evaluate import compute_metrics
from chestxray.model import build_model, compile_model, unfreeze_top_layers
from chestxray.pipeline import REGIONS, build_augmentation, make_dataset, region_mask


def make_callbacks(config: Config, checkpoint: Path, best_so_far: float | None = None):
    """Return the callbacks; all of them watch the validation loss only."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
        # best_so_far: phase 2 only overwrites the file if it beats phase 1
        keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_loss",
            save_best_only=True,
            initial_value_threshold=best_so_far,
        ),
    ]


def train(config: Config, data: XrayData, output_dir: Path) -> dict:
    """Train in two phases, evaluate the best model once on the test set.

    Phase 1 trains only the new head on the frozen backbone. Phase 2 unfreezes
    the last layers of the backbone and fine-tunes with a much smaller
    learning rate. The checkpoint with the lowest validation loss over both
    phases is evaluated on the test set, which is not used before.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(config.seed)

    classes = config.classes
    class_index = {name: i for i, name in enumerate(classes)}
    y = np.array([class_index[label] for label in data.labels])  # names -> 0..C-1
    one_hot = keras.utils.to_categorical(y, num_classes=len(classes))
    # near-identical images (and their patients) must stay in the same set
    groups = merge_groups(data.groups, duplicate_sets(data.images))
    split = split_by_patient(
        data.labels, groups, config.val_fraction, config.test_fraction, config.seed
    )

    mask = region_mask(config.image_size, config.region)

    def batches(idx, training=False):
        return make_dataset(
            data.images[idx],
            one_hot[idx],
            config.batch_size,
            augmentation=build_augmentation(config) if training else None,
            shuffle=training,
            seed=config.seed,
            mask=mask,
        )

    train_ds, val_ds = batches(split.train, training=True), batches(split.val)
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y[split.train])
    class_weight = dict(enumerate(weights))
    checkpoint = output_dir / "best_model.keras"

    model, backbone = build_model(len(classes), config)

    compile_model(model, config.head_learning_rate)
    phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs,
        class_weight=class_weight,
        callbacks=make_callbacks(config, checkpoint),
        verbose=2,
    )

    phase1_epochs = len(phase1.history["loss"])
    phase2_history = {key: [] for key in phase1.history}
    if config.fine_tune_epochs > 0:  # 0 = train the head only
        unfreeze_top_layers(backbone, config.fine_tune_layers)
        compile_model(model, config.fine_tune_learning_rate)
        best_val_loss = min(phase1.history["val_loss"])
        phase2_history = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=phase1_epochs,
            epochs=phase1_epochs + config.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(config, checkpoint, best_val_loss),
            verbose=2,
        ).history

    best = keras.models.load_model(checkpoint)
    test_images = make_dataset(
        data.images[split.test], batch_size=config.batch_size, mask=mask
    )
    probs = best.predict(test_images, verbose=0)
    metrics = compute_metrics(y[split.test], probs, config.classes)

    history = {
        key: [float(v) for v in phase1.history[key] + phase2_history[key]]
        for key in ("loss", "accuracy", "val_loss", "val_accuracy")
    }
    np.savez(output_dir / "split.npz", **vars(split), test_probs=probs)
    (output_dir / "history.json").write_text(
        json.dumps({"phase1_epochs": phase1_epochs, **history}, indent=2)
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    """Read the command line options."""
    parser = argparse.ArgumentParser(description="Train the chest X-ray classifier.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--head-epochs", type=int, default=Config.head_epochs)
    parser.add_argument("--fine-tune-epochs", type=int, default=Config.fine_tune_epochs)
    parser.add_argument(
        "--region",
        choices=REGIONS,
        default=Config.region,
        help="part of the image the model sees (masking experiment)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="use only N random images (quick test)"
    )
    return parser.parse_args()


def main() -> None:
    """Train one model from the command line and print its test metrics."""
    args = parse_args()
    config = dataclasses.replace(
        Config(),
        seed=args.seed,
        head_epochs=args.head_epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        region=args.region,
    )
    data = load_dataset(config)
    if args.limit:
        keep = np.random.default_rng(0).choice(
            len(data.labels), args.limit, replace=False
        )
        data = XrayData(**{k: v[np.sort(keep)] for k, v in vars(data).items()})

    output_dir = args.output_dir or Path("results") / f"seed_{args.seed}"
    metrics = train(config, data, output_dir)
    print(
        f"seed {args.seed}: accuracy {metrics['accuracy']:.4f}, "
        f"macro F1 {metrics['macro_f1']:.4f}, ROC-AUC {metrics['roc_auc_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
