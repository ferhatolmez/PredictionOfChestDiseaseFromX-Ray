"""All paths and hyperparameters in one place."""

from dataclasses import dataclass
from pathlib import Path

CLASSES = ("COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS")


@dataclass(frozen=True)
class Config:
    """Settings for loading the data and training the model.

    The class is frozen, so a run cannot change its own settings by accident.
    Use ``dataclasses.replace(config, seed=1)`` to make a modified copy.
    """

    # data
    data_dir: Path = Path("data")
    cache_file: Path = Path("data/cache_224.npz")
    classes: tuple[str, ...] = CLASSES
    image_size: int = 224

    # split (the rest is the training set)
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0

    # model
    weights: str | None = "imagenet"
    dense_units: int = 256
    dropout: float = 0.5

    # training, phase 1: frozen backbone
    batch_size: int = 32
    head_epochs: int = 15
    head_learning_rate: float = 1e-3

    # training, phase 2: fine-tuning
    fine_tune_epochs: int = 10
    fine_tune_learning_rate: float = 1e-5
    fine_tune_layers: int = 30

    # callbacks
    patience: int = 4

    # which part of the image the model may see (masking experiment):
    # "all", "lungs" (only a central ellipse) or "outside" (ellipse blacked out)
    region: str = "all"

    # augmentation: X-rays are taken in a standard position, so keep it light
    rotation: float = 10 / 360  # fraction of a full turn (RandomRotation)
    shift: float = 0.05  # fraction of the image size
    zoom: float = 0.1
