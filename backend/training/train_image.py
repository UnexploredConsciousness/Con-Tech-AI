"""
Guardian AI - Image Deepfake Detection Model Training
Fine-tunes EfficientNetB0 for AI-generated image detection.
"""

import os
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train-image")

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent / "data" / "images"


def build_efficientnet_model(num_classes: int = 2, image_size: int = 224):
    """Build EfficientNetB0-based deepfake classifier."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB0

    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )

    # Freeze base model initially for feature extraction
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = base(inputs, training=False)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    logger.info(f"Model built. Trainable params: {model.count_params():,}")
    return model, base


def build_data_pipeline(data_dir: Path, image_size: int = 224, batch_size: int = 32):
    """Build tf.data pipeline from directory structure:
    data/images/
        real/   ← genuine photos
        fake/   ← AI-generated images
    """
    import tensorflow as tf

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_dir),
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
    )

    AUTOTUNE = tf.data.AUTOTUNE

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),
    ])

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds


def train(epochs_frozen: int = 10, epochs_finetuned: int = 5, image_size: int = 224):
    import tensorflow as tf

    if not DATA_DIR.exists():
        logger.error(
            f"Data directory not found: {DATA_DIR}\n"
            "Create the following structure:\n"
            "  training/data/images/real/  ← genuine photos\n"
            "  training/data/images/fake/  ← AI-generated images"
        )
        return

    logger.info("Building data pipeline...")
    train_ds, val_ds = build_data_pipeline(DATA_DIR, image_size)

    logger.info("Building EfficientNet model...")
    model, base_model = build_efficientnet_model(num_classes=2, image_size=image_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
    ]

    # Phase 1 — Feature extraction (frozen base)
    logger.info(f"Phase 1: Training classification head for {epochs_frozen} epochs...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen, callbacks=callbacks)

    # Phase 2 — Fine-tuning (unfreeze top layers of base)
    logger.info(f"Phase 2: Fine-tuning top 30 layers for {epochs_finetuned} epochs...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetuned, callbacks=callbacks)

    # Save feature extractor (removes classification head)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    feature_model.save(str(MODELS_DIR / "efficientnet_features.h5"))

    # Save full model too
    model.save(str(MODELS_DIR / "efficientnet_classifier.h5"))
    logger.info(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    train()