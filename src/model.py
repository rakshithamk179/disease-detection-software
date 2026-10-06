"""MobileNetV3 and MobileNetV2 architecture definitions with two-stage training support."""

from typing import Optional, Tuple
import tensorflow as tf


def build_classifier_model(
    num_classes: int,
    architecture: str = "MobileNetV3Small",
    pretrained_weights: str = "imagenet",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_1: float = 0.30,
    dropout_2: float = 0.20,
) -> tf.keras.Model:
    """Build transfer learning model with frozen pretrained backbone and custom classification head.

    Raises RuntimeError if pretrained weights fail to load.
    """
    if not pretrained_weights:
        raise ValueError("Scientifically sound transfer learning requires pretrained_weights (e.g., 'imagenet').")

    inputs = tf.keras.Input(shape=input_shape, name="image_input")

    arch_lower = architecture.lower()
    try:
        if arch_lower in ["mobilenetv3small", "mobilenetv3_small"]:
            preprocessed = tf.keras.layers.Rescaling(scale=1.0, offset=0.0, name="preprocess_input")(inputs)
            base_model = tf.keras.applications.MobileNetV3Small(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights,
            )
        elif arch_lower in ["mobilenetv2", "mobilenet_v2"]:
            preprocessed = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess_input")(inputs)
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights,
            )
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'. Supported: MobileNetV3Small, MobileNetV2.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load pretrained weights '{pretrained_weights}' for architecture '{architecture}'. "
            f"Do not train from random initialization. Error: {str(e)}"
        ) from e

    # Stage 1: Freeze base model
    base_model.trainable = False

    # Pass through base model ensuring training=False to protect batch norm statistics
    features = base_model(preprocessed, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(features)
    x = tf.keras.layers.Dropout(dropout_1, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_128")(x)
    x = tf.keras.layers.Dropout(dropout_2, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classification_output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{architecture}_leaf_classifier")
    return model


def prepare_model_for_finetuning(
    model: tf.keras.Model,
    fine_tune_last_n_layers: int = 25,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreeze the top N layers of the base feature extractor while keeping

    BatchNormalization layers frozen to preserve running statistics.
    """
    # Locate base model layer
    base_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenet" in layer.name.lower():
            base_layer = layer
            break

    if base_layer is None:
        raise ValueError("Could not find backbone feature extractor in model layers.")

    base_layer.trainable = True

    # Keep initial layers frozen
    total_backbone_layers = len(base_layer.layers)
    freeze_until = max(0, total_backbone_layers - fine_tune_last_n_layers)

    for i, layer in enumerate(base_layer.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            # BatchNormalization layers must remain frozen during fine-tuning
            if isinstance(layer, (tf.keras.layers.BatchNormalization,)):
                layer.trainable = False
            else:
                layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
