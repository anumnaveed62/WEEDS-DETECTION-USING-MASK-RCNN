"""
Run the trained Mask R-CNN weed/crop detector on new images.

Draws bounding boxes + instance masks + class label + confidence on each
image and saves annotated copies. With --export_coco it also writes all
detections to predictions.json in COCO "results" format, ready to feed
into compute_map.py, compute_iou.py, or error_analysis.py.

Usage:
    python detect_weeds.py --weights checkpoints/best.pth --images data/test \
        --out results/inference_samples --score_thresh 0.5 --export_coco
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

CLASS_NAMES = {1: "crop", 2: "weed"}
CLASS_COLORS = {1: (46, 204, 113), 2: (231, 76, 60)}  # green=crop, red=weed


def build_model(num_classes, weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def overlay_predictions(image: Image.Image, boxes, labels, scores, masks, score_thresh):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < score_thresh:
            continue
        color = CLASS_COLORS.get(int(label), (52, 152, 219))

        binary_mask = mask[0] > 0.5
        mask_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
        mask_rgba[binary_mask] = (*color, 100)
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, mode="RGBA"))

        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
        text = f"{CLASS_NAMES.get(int(label), str(label))} {score:.2f}"
        draw.rectangle([x1, y1 - 14, x1 + 8 * len(text), y1], fill=color + (255,))
        draw.text((x1 + 2, y1 - 13), text, fill=(255, 255, 255, 255))

    return Image.alpha_composite(image, overlay).convert("RGB")


@torch.no_grad()
def run_inference(model, image_paths, device, score_thresh, out_dir, export_coco):
    os.makedirs(out_dir, exist_ok=True)
    coco_results = []

    for idx, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB")
        tensor = F.to_tensor(image).to(device)
        output = model([tensor])[0]

        boxes = output["boxes"].cpu().numpy()
        labels = output["labels"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        masks = output["masks"].cpu().numpy()

        annotated = overlay_predictions(image, boxes, labels, scores, masks, score_thresh)
        out_path = os.path.join(out_dir, os.path.basename(path))
        annotated.save(out_path)
        n_kept = int((scores >= score_thresh).sum())
        print(f"[{idx + 1}/{len(image_paths)}] {os.path.basename(path)}: {n_kept} detections -> {out_path}")

        if export_coco:
            for box, label, score, mask in zip(boxes, labels, scores, masks):
                if score < score_thresh:
                    continue
                x1, y1, x2, y2 = box
                binary_mask = np.asfortranarray((mask[0] > 0.5).astype(np.uint8))
                rle = mask_utils.encode(binary_mask)
                rle["counts"] = rle["counts"].decode("utf-8")
                coco_results.append({
                    # NOTE: image_id here is just the file's position in this run.
                    # If you plan to feed predictions.json into compute_map.py /
                    # compute_iou.py against your test.json ground truth, replace
                    # this with the actual COCO image_id from that annotation file
                    # so predictions line up with the right ground-truth image.
                    "image_id": idx,
                    "file_name": os.path.basename(path),
                    "category_id": int(label),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                    "segmentation": rle,
                })

    if export_coco:
        pred_path = os.path.join(out_dir, "predictions.json")
        with open(pred_path, "w") as f:
            json.dump(coco_results, f)
        print(f"\nSaved COCO-format predictions to {pred_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", required=True, help="Image file or directory")
    parser.add_argument("--num_classes", type=int, default=3, help="background + crop + weed")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--out", default="results/inference_samples")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--export_coco", action="store_true")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print("CUDA not available, falling back to CPU.")

    if os.path.isdir(args.images):
        image_paths = sorted(glob.glob(f"{args.images}/*.jpg") + glob.glob(f"{args.images}/*.png"))
    else:
        image_paths = [args.images]

    if not image_paths:
        raise FileNotFoundError(f"No images found at {args.images}")

    model = build_model(args.num_classes, args.weights, device)
    run_inference(model, image_paths, device, args.score_thresh, args.out, args.export_coco)


if __name__ == "__main__":
    main()