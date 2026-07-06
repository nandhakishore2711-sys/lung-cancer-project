"""
TensorFlow / Keras model definitions for lung cancer CT scan classification.

Two architectures are provided:

1. build_custom_cnn   - a from-scratch CNN (the "novel" architecture for the
                         project). Four convolutional blocks with batch
                         normalization + dropout, trained end-to-end.

2. build_transfer_cnn - a transfer-learning model (EfficientNetB0 backbone)
                         for a stronger accuracy baseline / comparison table
                         in your report ("proposed model vs. transfer
                         learning baseline" is a very common results section).

Both return a compiled tf.keras.Model with an identical input/output
contract, so app.py and train_tensorflow.py don't need to know which one
was used.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=3):
    """A compact custom CNN purpose-built for this task."""
    inputs = layers.Input(shape=input_shape, name="ct_scan_input")

    x = layers.Rescaling(1.0)(inputs)  # no-op placeholder; images already scaled to [0,1] upstream

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv_block")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="last_conv")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="lung_cancer_custom_cnn")
    return model


def build_transfer_cnn(input_shape=(224, 224, 3), num_classes=3, fine_tune=False):
    """EfficientNetB0-backed transfer learning model."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = fine_tune  # freeze by default; unfreeze in a later fine-tuning phase

    inputs = layers.Input(shape=input_shape, name="ct_scan_input")
    x = layers.Rescaling(255.0)(inputs)  # EfficientNet preprocessing expects 0-255 range
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=fine_tune)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="lung_cancer_efficientnet")
    return model


def build_model(architecture="custom", input_shape=(224, 224, 3), num_classes=3):
    if architecture == "custom":
        return build_custom_cnn(input_shape, num_classes)
    elif architecture == "efficientnet":
        return build_transfer_cnn(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_last_conv_layer_name(model):
    """Used by Grad-CAM to find the layer to hook into automatically."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


if __name__ == "__main__":
    m = build_model("custom")
    m.summary()
