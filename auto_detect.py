"""
Automatic weed/crop detection - no command-line arguments needed.

Runs continuously, watching a folder for new images. As soon as an image
shows up, it's automatically run through the trained Mask R-CNN model and
an annotated copy is saved to the output folder - no commands to type per
image.

Setup: edit the SETTINGS block below once (weights path already matches
your project), then just run it:
    venv\\Scripts\\python.exe auto_detect.py
or double-click run_auto_detect.bat.

Leave it running and drop new images into WATCH_FOLDER at any time -
they'll be picked up within POLL_SECONDS. Press Ctrl+C to stop.
"""
import glob
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F

# --------------------------------------------------------------------------
# SETTINGS - edit these once to match your project, then just run the script.
# --------------------------------------------------------------------------
WEIGHTS_PATH = "checkpoints/best.pth"
WATCH_FOLDER = "dataset/test"           # drop new images here (or point at any folder)
OUTPUT_FOLDER = "results/auto_detect"   # annotated images land here automatically
NUM_CLASSES = 5                         # background + your 4 trained classes
SCORE_THRESH = 0.5
POLL_SECONDS = 5                        # how often to check for new images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# From your training log's class mapping - edit if your categories differ.
CLASS_NAMES = {1: "crops-weeds-W8PH-WGI7", 2: "Foxtail", 3: "MintLeaf", 4: "MintStem"}
CLASS_COLORS = {1: (52, 152, 219), 2: (231, 76, 60), 3: (46, 204, 113), 4: (241, 196, 15)}


def build_model():
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, NUM_CLASSES)

    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def overlay_predictions(image, boxes, labels, scores, masks):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for box, label, score, mask in zip(boxes, labels, scores, masks):
        if score < SCORE_THRESH:
            continue
        color = CLASS_COLORS.get(int(label), (149, 165, 166))

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
def process_image(model, path, out_dir):
    image = Image.open(path).convert("RGB")
    tensor = F.to_tensor(image).to(DEVICE)
    output = model([tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    labels = output["labels"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output["masks"].cpu().numpy()

    annotated = overlay_predictions(image, boxes, labels, scores, masks)
    out_path = os.path.join(out_dir, os.path.basename(path))
    annotated.save(out_path)

    n_kept = int((scores >= SCORE_THRESH).sum())
    print(f"[{time.strftime('%H:%M:%S')}] {os.path.basename(path)}: {n_kept} detections -> {out_path}")


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Loading model from {WEIGHTS_PATH} on {DEVICE} ...")
    model = build_model()
    print(f"Watching {WATCH_FOLDER} for images (checking every {POLL_SECONDS}s). Press Ctrl+C to stop.\n")

    processed = set()

    while True:
        current = set(glob.glob(f"{WATCH_FOLDER}/*.jpg") + glob.glob(f"{WATCH_FOLDER}/*.png"))
        new_files = sorted(current - processed)

        for path in new_files:
            try:
                process_image(model, path, OUTPUT_FOLDER)
            except Exception as e:
                print(f"  Failed on {path}: {e}")
            processed.add(path)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
