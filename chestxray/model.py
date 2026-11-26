"""MobileNetV2 transfer-learning model."""

from tensorflow import keras

from chestxray.config import Config


def build_model(num_classes: int, config: Config) -> tuple[keras.Model, keras.Model]:
    """Build the classifier and return it together with its backbone.

    Input: float32 images in [0, 1]. The ``Rescaling`` layer maps them to
    [-1, 1], the range MobileNetV2 saw during ImageNet pretraining (the same
    as ``mobilenet_v2.preprocess_input``).

    The backbone starts frozen (phase 1). It is called with
    ``training=False``, so its BatchNorm layers keep their ImageNet
    statistics even after it is unfrozen for fine-tuning.
    """
    size = config.image_size
    backbone = keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights=config.weights
    )
    backbone.trainable = False

    inputs = keras.Input(shape=(size, size, 3), name="image")
    x = keras.layers.Rescaling(2.0, offset=-1.0, name="to_minus_one_one")(inputs)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = keras.layers.Dense(config.dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(config.dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(
        x
    )
    return keras.Model(inputs, outputs, name="chest_xray_classifier"), backbone


def unfreeze_top_layers(backbone: keras.Model, n_layers: int) -> None:
    """Make only the last ``n_layers`` of the backbone trainable (phase 2)."""
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False


def compile_model(model: keras.Model, learning_rate: float) -> None:
    """Compile with Adam and categorical cross-entropy.

    Compiling again is required after changing which layers are trainable.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
