"""Tests for the model, the input pipeline, Grad-CAM and the metrics.

Skipped when TensorFlow is not installed (the CI installs only light packages).
"""

import dataclasses

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from chestxray.config import Config  # noqa: E402
from chestxray.evaluate import compute_metrics  # noqa: E402
from chestxray.gradcam import grad_cam, overlay  # noqa: E402
from chestxray.model import build_model, unfreeze_top_layers  # noqa: E402
from chestxray.pipeline import (  # noqa: E402
    build_augmentation,
    make_dataset,
    region_mask,
)

# small and without downloading ImageNet weights, so the test is fast
CONFIG = dataclasses.replace(Config(), image_size=64, weights=None, dense_units=8)
RNG = np.random.default_rng(0)


def test_model_outputs_probabilities():
    model, _ = build_model(4, CONFIG)
    images = RNG.integers(0, 256, size=(5, 64, 64, 3), dtype=np.uint8)
    probs = model.predict(make_dataset(images, batch_size=2), verbose=0)
    assert probs.shape == (5, 4)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-5)


def test_unfreeze_top_layers():
    _, backbone = build_model(4, CONFIG)
    unfreeze_top_layers(backbone, 5)
    trainable = [layer.trainable for layer in backbone.layers]
    assert trainable[-5:] == [True] * 5 and not any(trainable[:-5])


def test_pipeline_scales_and_augments():
    images = np.full((4, 64, 64, 3), 255, dtype=np.uint8)
    labels = np.eye(4, dtype=np.float32)
    x, y = next(iter(make_dataset(images, labels, batch_size=4)))
    assert x.dtype == tf.float32 and float(tf.reduce_max(x)) == 1.0
    assert y.shape == (4, 4)
    x_aug, _ = next(iter(make_dataset(images, labels, 4, build_augmentation(CONFIG))))
    assert x_aug.shape == (4, 64, 64, 3)


def test_grad_cam_heatmap_shape_and_range():
    model, _ = build_model(4, CONFIG)
    image = RNG.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    heatmap, probs, index = grad_cam(model, image)
    assert heatmap.shape == (64, 64)
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    assert index == int(np.argmax(probs))
    assert overlay(image, heatmap).shape == image.shape


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 3, 0, 1])
    probs = np.eye(4)[y_true]
    metrics = compute_metrics(y_true, probs, ("a", "b", "c", "d"))
    assert metrics["accuracy"] == 1.0 and metrics["macro_f1"] == 1.0
    assert metrics["roc_auc_macro"] == 1.0


def test_region_masks_are_complementary():
    lungs = region_mask(64, "lungs")
    outside = region_mask(64, "outside")
    assert region_mask(64, "all") is None
    assert np.array_equal(lungs + outside, np.ones((64, 64, 1)))
    assert lungs[32, 32, 0] == 1 and outside[0, 0, 0] == 1
    images = np.full((2, 64, 64, 3), 255, dtype=np.uint8)
    x = next(iter(make_dataset(images, batch_size=2, mask=outside)))
    assert float(x[0, 32, 32, 0]) == 0.0 and float(x[0, 0, 0, 0]) == 1.0
