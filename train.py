"""
Train Mask R-CNN (torchvision, ResNet-50-FPN backbone) for weed vs. crop
instance segmentation.

Expects a COCO-style dataset:
    data/images/train/*.jpg
    data/images/val/*.jpg
    data/annotations/train.json
    data/annotations/val.json

Outputs (matching what the other scripts in this repo expect):
    checkpoints/best.pth        <- loaded by benchmark_fps.py / detect_weeds.py
    logs/metrics.json           <- JSON-lines log, read by plot_training_curves.py
                                    keys: iteration, total_loss, loss_classifier,
                                    loss_box_reg, loss_mask, loss_objectness,
                                    loss_rpn_box_reg, [val_map]

Usage:
    python train.py --config configs/mrcnn_weed.yaml --epochs 50
    python train.py --epochs 50 --batch_size 4 --lr 0.005 --device cuda
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

try:
    import yaml
except ImportError:
    yaml = None


# --------------------------------------------------------------------------
# Reproducibility (per README's "Reproducibility Notes")
# --------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class CocoWeedDataset(Dataset):
    """
    Loads a COCO-style instance segmentation dataset.
    category_id 0 is reserved for background by torchvision's convention;
    real category ids from the annotation file are remapped to 1..N.
    """

    def __init__(self, images_dir: str, ann_file: str, train: bool = True):
        self.images_dir = images_dir
        self.coco = COCO(ann_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.train = train

        cat_ids = sorted(self.coco.getCatIds())
        # Map original category ids -> contiguous 1..N (0 = background)
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_name = {
            self.cat_id_to_label[c["id"]]: c["name"] for c in self.coco.loadCats(cat_ids)
        }
        self.num_classes = len(cat_ids) + 1  # +1 for background

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        image = F.to_tensor(image)

        if len(boxes) == 0:
            # No annotations on this image -> empty target (rare, but handle it)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8),
                "image_id": torch.tensor([image_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([image_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }

        if self.train and random.random() < 0.5:
            image = image.flip(-1)
            w_img = image.shape[-1]
            if target["boxes"].numel():
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w_img - boxes[:, [2, 0]]
                target["boxes"] = boxes
                target["masks"] = target["masks"].flip(-1)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes: int):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

    # Replace the box predictor head for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor head for our number of classes
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def append_log(log_path: str, record: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True) if os.path.dirname(log_path) else None
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Train / validate loops
# --------------------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device, epoch, log_path, global_step, print_every=20):
    model.train()
    epoch_loss = 0.0

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        epoch_loss += loss_value
        global_step += 1

        record = {"iteration": global_step, "epoch": epoch, "total_loss": round(loss_value, 4)}
        record.update({k: round(v.item(), 4) for k, v in loss_dict.items()})
        append_log(log_path, record)

        if i % print_every == 0:
            print(f"  epoch {epoch} | step {i}/{len(loader)} | loss {loss_value:.4f}")

    return epoch_loss / max(len(loader), 1), global_step


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """
    Quick validation signal: average of the same loss terms used in training,
    computed on the val set (model.train() mode is required for torchvision
    detection models to return losses instead of predictions).
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# --------------------------------------------------------------------------
# Config loading (optional YAML, CLI args take precedence if both given)
# --------------------------------------------------------------------------
def load_config(config_path):
    if not config_path:
        return {}
    if yaml is None:
        print("Warning: pyyaml not installed, skipping --config file. `pip install pyyaml` to enable it.")
        return {}
    if not os.path.exists(config_path):
        print(f"Warning: config file {config_path} not found, using CLI args/defaults only.")
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config (e.g. configs/mrcnn_weed.yaml)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--train_images", default=None, help="Default: <data_dir>/images/train")
    parser.add_argument("--train_ann", default=None, help="Default: <data_dir>/annotations/train.json")
    parser.add_argument("--val_images", default=None, help="Default: <data_dir>/images/val")
    parser.add_argument("--val_ann", default=None, help="Default: <data_dir>/annotations/val.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--lr_step_size", type=int, default=15, help="Epochs between LR decay steps")
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--log_path", default="logs/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, value in cfg.items():
        if hasattr(args, key) and parser.get_default(key) == getattr(args, key):
            # Only let config override args the user didn't explicitly pass on CLI
            setattr(args, key, value)

    set_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print("CUDA not available, falling back to CPU. Training will be significantly slower.")

    train_images = args.train_images or os.path.join(args.data_dir, "images", "train")
    train_ann = args.train_ann or os.path.join(args.data_dir, "annotations", "train.json")
    val_images = args.val_images or os.path.join(args.data_dir, "images", "val")
    val_ann = args.val_ann or os.path.join(args.data_dir, "annotations", "val.json")

    print(f"Loading train set from {train_images} / {train_ann}")
    train_ds = CocoWeedDataset(train_images, train_ann, train=True)
    print(f"Loading val set from {val_images} / {val_ann}")
    val_ds = CocoWeedDataset(val_images, val_ann, train=False)

    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | Classes (incl. bg): {train_ds.num_classes}")
    print(f"Class mapping: {train_ds.label_to_name}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = build_model(train_ds.num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"\nStarting training for {args.epochs} epochs on {device}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, optimizer, train_loader, device, epoch, args.log_path, global_step
        )
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs - 1} done in {elapsed:.1f}s | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        append_log(args.log_path, {
            "iteration": global_step,
            "epoch": epoch,
            "epoch_train_loss": round(train_loss, 4),
            "epoch_val_loss": round(val_loss, 4),
        })

        # Always save the latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "last.pth")
        torch.save(model.state_dict(), latest_path)

        # Save best-so-far separately (this is what benchmark_fps.py / inference.py load)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  New best val_loss={val_loss:.4f} -> saved {best_path}")

        # Full checkpoint (for resuming), separate from the plain state_dict files above
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, os.path.join(args.checkpoint_dir, "resume_state.pth"))

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Best weights: {os.path.join(args.checkpoint_dir, 'best.pth')}")
    print(f"Metrics log: {args.log_path}")
    print(
        "\nNext steps:\n"
        f"  python plot_training_curves.py --log {args.log_path} --out results/training_curves.png\n"
        "  python detect_weeds.py --weights "
        f"{os.path.join(args.checkpoint_dir, 'best.pth')} --images data/valid --export_coco\n"
    )


if __name__ == "__main__":
    main()
