"""
Train the lung cancer CNN using PyTorch.

Expected data layout (identical to the TensorFlow script — same dataset
folder works for both):

    data/
      train/{Benign,Malignant,Normal}/*.png
      val/{Benign,Malignant,Normal}/*.png
      test/{Benign,Malignant,Normal}/*.png

Usage:
    python train_pytorch.py --data_dir data --epochs 30 --architecture custom
    python train_pytorch.py --data_dir data --epochs 15 --architecture efficientnet
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.pytorch_model import build_model
from utils.preprocessing import IMG_SIZE


def build_loaders(data_dir, img_size, batch_size):
    train_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ColorJitter(contrast=0.1, brightness=0.1),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_tf)

    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf) if os.path.isdir(test_dir) else None

    class_names = train_dataset.classes  # alphabetical, matches ImageFolder convention

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if test_dataset else None

    return train_loader, val_loader, test_loader, class_names, train_dataset


def compute_weights(train_dataset, num_classes, device):
    labels = [label for _, label in train_dataset.samples]
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            output = model(images)
            logits = output[0] if _model_returns_features(model) else output
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def _model_returns_features(model):
    """Custom LungCancerCNN returns (logits, features); torchvision models return logits only."""
    return hasattr(model, "get_last_conv_layer")


def plot_history(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
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
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--architecture", default="custom", choices=["custom", "efficientnet"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="saved_models")
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names, train_dataset = build_loaders(
        args.data_dir, IMG_SIZE, args.batch_size
    )
    print("Detected classes:", class_names)

    class_weights = compute_weights(train_dataset, len(class_names), device)
    print("Class weights:", class_weights.tolist())

    model = build_model(args.architecture, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_acc = 0.0
    best_path = os.path.join(args.output_dir, f"lung_cnn_pt_{args.architecture}.pt")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": args.architecture,
                "class_names": class_names,
            }, best_path)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping.")
                break

    plot_history(history, os.path.join(args.output_dir, "training_curves_pytorch.png"))

    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(class_names, f)

    # Reload best checkpoint for final evaluation
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_loader = test_loader if test_loader is not None else val_loader
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            output = model(images)
            logits = output[0] if _model_returns_features(model) else output
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print("\nClassification report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(args.output_dir, "classification_report_pytorch.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, "confusion_matrix_pytorch.png"))

    print(f"\nSaved best model to: {best_path}")
    print(f"Artifacts written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
