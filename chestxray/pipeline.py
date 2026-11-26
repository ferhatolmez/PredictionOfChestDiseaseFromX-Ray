"""tf.data input pipeline, data augmentation and region masks.

This replaces Keras' deprecated ``ImageDataGenerator``: augmentation runs as
Keras preprocessing layers inside a ``tf.data`` pipeline, in parallel with
training.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from chestxray.config import Config

AUTOTUNE = tf.data.AUTOTUNE
REGIONS = ("all", "lungs", "outside")

# Ellipse that covers both lungs on a centred frontal chest X-ray, as fractions
# of the image size: centre (x, y) and semi-axes (a, b).
LUNG_ELLIPSE = (0.5, 0.5, 0.40, 0.40)


def region_mask(size: int, region: str) -> np.ndarray | None:
    """Return a (size, size, 1) float32 mask of 0/1, or None for "all".

    "lungs" keeps only the central ellipse; "outside" keeps everything except
    it (borders, corners, text markers). Training on "outside" tests whether
    the model can find the class without seeing the lungs at all.
    """
    if region not in REGIONS:
        raise ValueError(f"region must be one of {REGIONS}, got {region!r}")
    if region == "all":
        return None
    cx, cy, a, b = LUNG_ELLIPSE
    y, x = np.mgrid[0:size, 0:size] / size
    inside = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
    keep = inside if region == "lungs" else ~inside
    return keep[..., None].astype(np.float32)


def build_augmentation(config: Config) -> keras.Sequential:
    """Return the random transformations applied to training images only.

    No horizontal flip: the heart is on the left, so a mirrored chest X-ray
    is not a realistic image.
    """
    return keras.Sequential(
        [
            keras.layers.RandomRotation(
                config.rotation, fill_mode="constant", seed=config.seed
            ),
            keras.layers.RandomTranslation(
                config.shift, config.shift, fill_mode="constant", seed=config.seed
            ),
            keras.layers.RandomZoom(
                config.zoom, fill_mode="constant", seed=config.seed
            ),
        ],
        name="augmentation",
    )


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray | None = None,
    batch_size: int = 32,
    augmentation: keras.Sequential | None = None,
    shuffle: bool = False,
    seed: int = 0,
    mask: np.ndarray | None = None,
) -> tf.data.Dataset:
    """Build batches of float32 images in [0, 1] (and labels, if given).

    Only indices are shuffled; each batch is gathered from the uint8 array,
    so the images are never copied as float32 all at once. A ``mask`` is
    applied after augmentation, so the hidden region is always the same.
    """
    images_t = tf.convert_to_tensor(images)
    labels_t = None if labels is None else tf.convert_to_tensor(labels)
    mask_t = None if mask is None else tf.convert_to_tensor(mask)

    ds = tf.data.Dataset.range(len(images))
    if shuffle:
        ds = ds.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def gather(idx):
        x = tf.cast(tf.gather(images_t, idx), tf.float32) / 255.0
        if augmentation is not None:
            x = augmentation(x, training=True)
        if mask_t is not None:
            x = x * mask_t
        return x if labels_t is None else (x, tf.gather(labels_t, idx))

    return ds.map(gather, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
