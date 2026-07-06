"""
Train the lung cancer CNN using TensorFlow / Keras.

Expected data layout (see data/README.md for how to get this from the
IQ-OTH/NCCD dataset or your own data):

    data/
      train/
        Benign/    *.png
        Malignant/ *.png
        Normal/    *.png
      val/
        Benign/ ...
        Malignant/ ...
        Normal/ ...
      test/
        Benign/ ...
        Malignant/ ...
        Normal/ ...

Usage:
    python train_tensorflow.py --data_dir data --epochs 30 --architecture custom
    python train_tensorflow.py --data_dir data --epochs 15 --architecture efficientnet
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe: never opens a window, always writes files
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from models.tensorflow_model import build_model
from utils.preprocessing import IMG_SIZE, CLASS_NAMES


def build_datasets(data_dir, img_size, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )
    test_dir = os.path.join(data_dir, "test")
    test_ds = None
    if os.path.isdir(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir, image_size=img_size, batch_size=batch_size, label_mode="int", shuffle=False
        )

    class_names = train_ds.class_names  # alphabetical, e.g. ['Benign', 'Malignant', 'Normal']

    normalize = tf.keras.layers.Rescaling(1.0 / 255)

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (normalize(x), y))
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalize(x), y))
    if test_ds is not None:
        test_ds = test_ds.map(lambda x, y: (normalize(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_weights(data_dir, class_names):
    """Class weights from folder counts — handles the class imbalance that
    is common in medical imaging datasets (e.g. far fewer malignant scans)."""
    labels = []
    train_dir = os.path.join(data_dir, "train")
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(train_dir, cname)
        if os.path.isdir(cdir):
            n = len([f for f in os.listdir(cdir) if not f.startswith(".")])
            labels.extend([idx] * n)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=labels)
    return {i: w for i, w in enumerate(weights)}


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data", help="Root folder containing train/val/test subfolders")
    parser.add_argument("--architecture", default="custom", choices=["custom", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="saved_models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = build_datasets(args.data_dir, IMG_SIZE, args.batch_size)
    print("Detected classes:", class_names)

    class_weights = compute_weights(args.data_dir, class_names)
    print("Class weights:", class_weights)

    model = build_model(args.architecture, input_shape=IMG_SIZE + (3,), num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    ckpt_path = os.path.join(args.output_dir, f"lung_cnn_tf_{args.architecture}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(os.path.join(args.output_dir, "training_log.csv")),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    plot_history(history, os.path.join(args.output_dir, "training_curves.png"))

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(class_names, f)

    # Final evaluation on held-out test set, if present
    eval_ds = test_ds if test_ds is not None else val_ds
    y_true, y_pred = [], []
    for batch_x, batch_y in eval_ds:
        preds = model.predict(batch_x, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_y.numpy())

    print("\nClassification report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix.png"))

    print(f"\nSaved best model to: {ckpt_path}")
    print(f"Artifacts written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
