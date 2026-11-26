"""Grad-CAM: which part of the X-ray made the model choose a class.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

The backbone's last feature maps keep spatial layout (7x7 for a 224x224
image). Grad-CAM weights each feature map by how much the class score
changes with it (the mean gradient), sums the weighted maps and keeps the
positive part. High values mark regions that pushed the score up.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _split_model(model: keras.Model):
    """Return (layers before the backbone, backbone, layers after it)."""
    backbone_index = next(
        i for i, layer in enumerate(model.layers) if isinstance(layer, keras.Model)
    )
    before = [
        layer
        for layer in model.layers[:backbone_index]
        if not isinstance(layer, keras.layers.InputLayer)
    ]
    return before, model.layers[backbone_index], model.layers[backbone_index + 1 :]


def grad_cam(
    model: keras.Model, image: np.ndarray, class_index: int | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute a Grad-CAM heatmap for one image.

    Args:
        model: classifier built by ``chestxray.model.build_model``.
        image: uint8 image (H, W, 3).
        class_index: class to explain; the predicted class if None.

    Returns:
        ``heatmap`` (H, W) in [0, 1], the class probabilities, and the class
        index that was explained.
    """
    before, backbone, after = _split_model(model)
    x = tf.convert_to_tensor(image[None].astype(np.float32) / 255.0)

    with tf.GradientTape() as tape:
        for layer in before:
            x = layer(x)
        features = backbone(x, training=False)  # (1, h, w, channels)
        tape.watch(features)
        out = features
        for layer in after:
            out = layer(out, training=False)
        if class_index is None:
            class_index = int(tf.argmax(out[0]))
        score = out[:, class_index]

    grads = tape.gradient(score, features)  # d score / d feature map
    weights = tf.reduce_mean(grads, axis=(1, 2))  # one weight per feature map
    cam = tf.nn.relu(tf.einsum("bhwc,bc->bhw", features, weights))[0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    heatmap = cv2.resize(cam, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return heatmap, out[0].numpy(), class_index


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a heatmap (jet colors) over an RGB uint8 image."""
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
