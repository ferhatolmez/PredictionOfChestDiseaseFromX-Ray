"""Predict the class of new X-ray images with a trained model.

The preprocessing is exactly the one used in training (resize to 224x224
with area interpolation, RGB, white balance), otherwise the model would see
images unlike the ones it learned from.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras

from chestxray.config import CLASSES
from chestxray.gradcam import grad_cam, overlay
from chestxray.preprocessing import white_balance

DEFAULT_MODEL = Path("results/seed_0/best_model.keras")


@dataclass
class Prediction:
    """The result for one image."""

    probabilities: dict[str, float]  # class name -> probability, sums to 1
    predicted: str
    heatmap_overlay: np.ndarray  # RGB uint8 image with the Grad-CAM heatmap


def prepare(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Turn an RGB (or grayscale) image of any size into the model's input.

    Returns a uint8 RGB image (size, size, 3), resized and white balanced.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # PNG with transparency
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return white_balance(image.astype(np.uint8))


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class Predictor:
    """Load a trained model once and use it for many images."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, classes=CLASSES):
        """Load the model from ``model_path``."""
        self.model = keras.models.load_model(model_path)
        self.classes = tuple(classes)
        self.size = self.model.input_shape[1]

    def predict(self, image: np.ndarray, explain: bool = True) -> Prediction:
        """Classify one RGB image; with ``explain`` also compute Grad-CAM."""
        prepared = prepare(image, self.size)
        if explain:
            heatmap, probs, _ = grad_cam(self.model, prepared)
            shown = overlay(prepared, heatmap)
        else:
            probs = self.model(prepared[None].astype(np.float32) / 255.0)[0].numpy()
            shown = prepared
        probabilities = {
            name: float(p) for name, p in zip(self.classes, probs, strict=True)
        }
        return Prediction(
            probabilities=probabilities,
            predicted=max(probabilities, key=probabilities.get),
            heatmap_overlay=shown,
        )
