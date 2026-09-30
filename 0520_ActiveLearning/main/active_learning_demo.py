#!/usr/bin/env python3
"""
Active Learning Simulation — image-level selection, crop-level training.
Classes: BC, ETB, YM, XSA
"""

import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Try to import DINOv2 (official)
try:
    import dinov2  # noqa: F401
    from dinov2.models import build_model_from_cfg  # noqa: F401
    DINO_AVAILABLE = True
except ImportError:
    print("DINOv2 not installed. Install with: pip install dinov2")
    DINO_AVAILABLE = False
    import torchvision.models as models

# ---------------------------
# 1. Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_DIR = "./dataset/images/"
ANNOT_DIR = "./dataset/annotations/"

CLASSES = ["BC", "ETB", "YM", "XSA"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Image-level active learning
INITIAL_PER_CLASS = 5
QUERY_IMAGES_PER_ROUND = 10
NUM_QUERY_ROUNDS = 5
ACC_STOP_THRESHOLD = 0.95
TEST_IMAGES_PER_CLASS = 4  # 16 test images total (20% of 80)

MAX_EPOCHS = 20
BATCH_SIZE_TRAIN = 8
LEARNING_RATE = 1e-4


def extract_plate_class(filename: str) -> str:
    """Map filename prefix to plate class (BC, ETB, YM, XSA)."""
    name = filename.lower()
    if "bc" in name:
        return "BC"
    if "etb" in name:
        return "ETB"
    if "ym" in name:
        return "YM"
    if "xsa" in name:
        return "XSA"
    raise ValueError(f"Cannot determine class from filename: {filename}")


# ---------------------------
# 2. Dataset Loading with Bounding Boxes
# ---------------------------
class BacteriaDataset(Dataset):
    """Assumes each image has a corresponding .json annotation file."""

    def __init__(self, image_dir, annot_dir, transform=None):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.all_crops = []  # list of (image_path, bbox, class_idx)

        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for fname in image_files:
            img_path = os.path.join(image_dir, fname)
            base = os.path.splitext(fname)[0]
            annot_file = os.path.join(annot_dir, base + ".json")
            if not os.path.exists(annot_file):
                print(f"Warning: annotation missing for {fname}")
                continue

            with open(annot_file, "r") as f:
                ann_data = json.load(f)
            for obj in ann_data:
                bbox = obj["bbox"]
                cls_name = obj["class"]
                if cls_name not in CLASSES:
                    continue
                self.all_crops.append((img_path, bbox, CLASS_TO_IDX[cls_name]))

        print(
            f"Loaded {len(self.all_crops)} colony crops from {len(image_files)} images."
        )

    def __len__(self):
        return len(self.all_crops)

    def __getitem__(self, idx):
        img_path, bbox_norm, label = self.all_crops[idx]
        image = Image.open(img_path).convert("RGB")
        w_img, h_img = image.size
        cx, cy, bw, bh = bbox_norm
        left = (cx - bw / 2) * w_img
        top = (cy - bh / 2) * h_img
        right = (cx + bw / 2) * w_img
        bottom = (cy + bh / 2) * h_img
        crop = image.crop((left, top, right, bottom))
        if self.transform:
            crop = self.transform(crop)
        return crop, label


def pil_to_tensor(pic):
    """PIL -> CHW tensor without NumPy (works when PyTorch NumPy ABI is broken)."""
    if pic.mode != "RGB":
        pic = pic.convert("RGB")
    w, h = pic.size
    data = torch.tensor(bytearray(pic.tobytes()), dtype=torch.uint8)
    return data.view(h, w, 3).permute(2, 0, 1).float().div(255.0)


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(pil_to_tensor),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------
# 3. Image index (image-level AL, crop-level training)
# ---------------------------
def build_image_index(dataset):
    """
    Returns:
        images: list of unique image paths
        image_to_class: class index per image (from filename)
        image_to_crop_indices: image_id -> list of crop indices in dataset
        path_to_image_id: image path -> image_id
    """
    path_to_image_id = {}
    images = []
    image_to_class = []
    image_to_crop_indices = defaultdict(list)

    for crop_idx, (img_path, _, _) in enumerate(dataset.all_crops):
        if img_path not in path_to_image_id:
            path_to_image_id[img_path] = len(images)
            images.append(img_path)
            cls_name = extract_plate_class(os.path.basename(img_path))
            image_to_class.append(CLASS_TO_IDX[cls_name])
        image_to_crop_indices[path_to_image_id[img_path]].append(crop_idx)

    return images, image_to_class, dict(image_to_crop_indices), path_to_image_id


def crop_indices_for_images(image_ids, image_to_crop_indices):
    indices = []
    for img_id in image_ids:
        indices.extend(image_to_crop_indices[img_id])
    return indices


def stratified_image_split(image_to_class, per_class_count, exclude_ids=None):
    """Pick per_class_count image ids per class. exclude_ids are not chosen."""
    exclude = set(exclude_ids or [])
    by_class = defaultdict(list)
    for img_id, cls_idx in enumerate(image_to_class):
        if img_id in exclude:
            continue
        by_class[cls_idx].append(img_id)

    selected = []
    for cls_idx in range(NUM_CLASSES):
        pool = by_class[cls_idx]
        random.shuffle(pool)
        if len(pool) < per_class_count:
            raise ValueError(
                f"Not enough images for class {CLASSES[cls_idx]}: "
                f"need {per_class_count}, have {len(pool)}"
            )
        selected.extend(pool[:per_class_count])
    return selected


def seed_images_stratified(image_to_class, per_class=5, pool_ids=None):
    """Pick per_class random image ids per class from pool_ids."""
    pool_set = set(pool_ids) if pool_ids is not None else set(range(len(image_to_class)))
    by_class = defaultdict(list)
    for img_id in pool_set:
        by_class[image_to_class[img_id]].append(img_id)

    selected = []
    for cls_idx in range(NUM_CLASSES):
        pool = by_class[cls_idx]
        random.shuffle(pool)
        selected.extend(pool[:per_class])
    return selected


# ---------------------------
# 4. DINOv2 Backbone (Frozen) + Lightweight Classifier
# ---------------------------
class DINOv3Classifier(nn.Module):
    def __init__(self, num_classes, backbone_name="dinov2_vits14"):
        super().__init__()
        if DINO_AVAILABLE:
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            feat_dim = 384
        else:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feat_dim = 2048
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            if not DINO_AVAILABLE:
                features = features.flatten(1)
            else:
                features = features["x_norm_patchtokens"]
                features = features.mean(dim=1)
        logits = self.classifier(features)
        return logits


# ---------------------------
# 5. Training and evaluation
# ---------------------------
def train_model(model, train_loader, epochs=MAX_EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")
    return model


def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_y.tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return acc, f1


# ---------------------------
# 6. Uncertainty sampling (per image)
# ---------------------------
def compute_image_uncertainties(model, dataset, image_ids, image_to_crop_indices, batch_size=32):
    """Mean entropy over all crops in each image."""
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for img_id in image_ids:
            crop_ids = image_to_crop_indices[img_id]
            if not crop_ids:
                uncertainties.append(0.0)
                continue
            loader = DataLoader(
                Subset(dataset, crop_ids), batch_size=batch_size, shuffle=False
            )
            entropies = []
            for batch_x, _ in loader:
                batch_x = batch_x.to(DEVICE)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropies.extend(entropy.cpu().tolist())
            uncertainties.append(float(np.mean(entropies)))
    return uncertainties


def query_images(model, dataset, unlabeled_image_ids, image_to_crop_indices, n_query=10):
    """Select n_query images with highest mean crop entropy."""
    uncertainties = compute_image_uncertainties(
        model, dataset, unlabeled_image_ids, image_to_crop_indices
    )
    pairs = list(zip(unlabeled_image_ids, uncertainties))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [img_id for img_id, _ in pairs[:n_query]]


# ---------------------------
# 7. Main active learning simulation
# ---------------------------
def main():
    print("Loading dataset...")
    transform = get_transform()
    full_dataset = BacteriaDataset(IMAGE_DIR, ANNOT_DIR, transform=transform)
    images, image_to_class, image_to_crop_indices, _ = build_image_index(full_dataset)
    num_images = len(images)
    print(f"Indexed {num_images} unique images.")

    test_image_ids = stratified_image_split(
        image_to_class, TEST_IMAGES_PER_CLASS
    )
    test_set = set(test_image_ids)
    pool_image_ids = [i for i in range(num_images) if i not in test_set]
    print(f"Test images: {len(test_image_ids)}, AL pool: {len(pool_image_ids)}")

    labeled_images = seed_images_stratified(
        image_to_class, per_class=INITIAL_PER_CLASS, pool_ids=pool_image_ids
    )
    remaining_images = [i for i in pool_image_ids if i not in set(labeled_images)]

    test_crop_indices = crop_indices_for_images(test_image_ids, image_to_crop_indices)
    test_loader = DataLoader(
        Subset(full_dataset, test_crop_indices),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=False,
    )

    def get_loader(image_ids):
        crop_idx = crop_indices_for_images(image_ids, image_to_crop_indices)
        return DataLoader(
            Subset(full_dataset, crop_idx),
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
        )

    accuracies = []
    num_labeled_images = []
    model = DINOv3Classifier(NUM_CLASSES).to(DEVICE)

    for round_idx in range(NUM_QUERY_ROUNDS + 1):
        print(f"\n--- Round {round_idx} ({len(labeled_images)} labeled images) ---")
        train_loader = get_loader(labeled_images)
        model = train_model(model, train_loader)
        acc, f1 = evaluate_model(model, test_loader)
        num_labeled_images.append(len(labeled_images))
        accuracies.append(acc)
        print(f"Round {round_idx}: images={len(labeled_images)}, acc={acc:.4f}, f1={f1:.4f}")

        if acc >= ACC_STOP_THRESHOLD:
            print(f"Stopped: accuracy {acc:.4f} >= {ACC_STOP_THRESHOLD}")
            break
        if round_idx >= NUM_QUERY_ROUNDS:
            break
        if not remaining_images:
            print("Stopped: no remaining images in pool.")
            break

        n_query = min(QUERY_IMAGES_PER_ROUND, len(remaining_images))
        selected = query_images(
            model, full_dataset, remaining_images, image_to_crop_indices, n_query
        )
        labeled_images.extend(selected)
        remaining_set = set(selected)
        remaining_images = [i for i in remaining_images if i not in remaining_set]
        print(f"Queried {len(selected)} images (uncertainty sampling).")

    plt.figure(figsize=(8, 6))
    plt.plot(num_labeled_images, accuracies, "bo-", label="Uncertainty sampling")
    plt.xlabel("Number of Labeled Images")
    plt.ylabel("Test Accuracy")
    plt.title("Image-Level Active Learning — Bacteria Classification")
    plt.legend()
    plt.grid(True)
    plt.savefig("active_learning_results.png")
    plt.close()
    print("Saved active_learning_results.png")

    import pandas as pd

    df = pd.DataFrame({
        "num_labeled_images": num_labeled_images,
        "accuracy": accuracies,
    })
    df.to_csv("active_learning_simulation.csv", index=False)
    print("Results saved to active_learning_simulation.csv")


if __name__ == "__main__":
    main()
