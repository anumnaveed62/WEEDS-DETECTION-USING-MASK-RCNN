"""
Train Mask R-CNN (torchvision, ResNet-50-FPN backbone) on the Crops & Weeds
COCO-segmentation dataset exported from Roboflow.

What this script does, step by step:
  1. Reads dataset/model/training settings from data.yaml.
  2. Builds train/val DataLoaders from the COCO json annotations.
  3. Loads a COCO-pretrained Mask R-CNN and swaps its box + mask prediction
     heads for 3 classes (background, crop, weed) -- this is the standard
     transfer-learning recipe and matters a lot with only 40 images.
  4. Trains with SGD + step LR decay, logging every-iteration losses to a
     JSON-lines file that plot_training_curves.py (already in your repo)
     can plot directly.
  5. Every few epochs, runs validation mAP (COCOeval, bbox) and checkpoints
     the best-performing weights to checkpoints/best.pth -- the exact path
     benchmark_fps.py and detect_weeds.py expect.

Usage:
    python train_maskrcnn.py --config data.yaml
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset_coco import CocoSegmentationDataset, collate_fn
from transforms import get_train_transforms, get_eval_transforms


def build_model(num_classes):
    # Start from a Mask R-CNN pretrained on COCO -- the backbone already
    # knows general visual features (edges, textures, plant-like shapes),
    # so with 40 images we're really only fine-tuning the heads.
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    # Replace the box classification/regression head for our 3 classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask head for our 3 classes.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


@torch.no_grad()
def export_predictions(model, loader, device):
    """Run the model over a DataLoader and return COCO 'results'-format predictions."""
    model.eval()
    results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            masks = output["masks"].cpu().numpy()  # [N, 1, H, W] soft masks in [0,1]

            for box, score, label, mask in zip(boxes, scores, labels, masks):
                x1, y1, x2, y2 = box
                binary_mask = np.asfortranarray((mask[0] > 0.5).astype(np.uint8))
                rle = mask_utils.encode(binary_mask)
                rle["counts"] = rle["counts"].decode("utf-8")
                results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "segmentation": rle,
                })
    return results


def evaluate_map(model, loader, coco_gt: COCO, device, out_path):
    """Quick val mAP (bbox) using the same COCOeval machinery as compute_map.py."""
    results = export_predictions(model, loader, device)
    if not results:
        return {"AP@[.5:.95]": 0.0, "AP@0.5": 0.0}
    with open(out_path, "w") as f:
        json.dump(results, f)
    coco_dt = coco_gt.loadRes(out_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {"AP@[.5:.95]": float(coco_eval.stats[0]), "AP@0.5": float(coco_eval.stats[1])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data.yaml")
    parser.add_argument("--eval_every", type=int, default=5, help="Run val mAP every N epochs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_classes = cfg["classes"]["num_classes"]
    train_cfg = cfg["training"]
    paths = cfg["paths"]

    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    log_dir = os.path.dirname(train_cfg["log_file"])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    train_ds = CocoSegmentationDataset(
        paths["train"]["images"], paths["train"]["annotations"], transforms=get_train_transforms()
    )
    val_ds = CocoSegmentationDataset(
        paths["val"]["images"], paths["val"]["annotations"], transforms=get_eval_transforms()
    )
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    model = build_model(num_classes).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=train_cfg["lr"], momentum=train_cfg["momentum"], weight_decay=train_cfg["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=train_cfg["lr_step_size"], gamma=train_cfg["lr_gamma"]
    )

    log_path = train_cfg["log_file"]
    open(log_path, "w").close()  # start a fresh log each run

    iteration = 0
    best_map = -1.0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_start = time.time()
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            record = {
                "iteration": iteration,
                "epoch": epoch,
                "total_loss": float(total_loss.item()),
                "loss_cls": float(loss_dict.get("loss_classifier", torch.tensor(0.0)).item()),
                "loss_box_reg": float(loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()),
                "loss_mask": float(loss_dict.get("loss_mask", torch.tensor(0.0)).item()),
                "loss_rpn_cls": float(loss_dict.get("loss_objectness", torch.tensor(0.0)).item()),
                "loss_rpn_loc": float(loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0)).item()),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            if iteration % 10 == 0:
                print(f"epoch {epoch} iter {iteration} loss {total_loss.item():.4f}")
            iteration += 1

        lr_scheduler.step()
        print(f"Epoch {epoch} done in {time.time() - epoch_start:.1f}s")

        do_eval = (epoch + 1) % args.eval_every == 0 or epoch == train_cfg["epochs"] - 1
        if do_eval:
            metrics = evaluate_map(
                model, val_loader, val_ds.coco, device,
                out_path=os.path.join(train_cfg["checkpoint_dir"], "val_preds.json"),
            )
            print(f"Val mAP@[.5:.95]={metrics['AP@[.5:.95]']:.4f}  mAP@0.5={metrics['AP@0.5']:.4f}")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "iteration": iteration, "epoch": epoch, "val_map": metrics["AP@[.5:.95]"]
                }) + "\n")

            if metrics["AP@[.5:.95]"] > best_map:
                best_map = metrics["AP@[.5:.95]"]
                torch.save(model.state_dict(), os.path.join(train_cfg["checkpoint_dir"], "best.pth"))
                print(f"  -> new best model saved (mAP={best_map:.4f})")

        torch.save(model.state_dict(), os.path.join(train_cfg["checkpoint_dir"], "last.pth"))

    print("Training complete.")
    print(f"Best checkpoint: {os.path.join(train_cfg['checkpoint_dir'], 'best.pth')} (mAP={best_map:.4f})")


if __name__ == "__main__":
    main()
