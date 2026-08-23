"""
COCO-format instance segmentation dataset loader for Mask R-CNN.

Loads images + polygon/RLE segmentation masks from a Roboflow-exported
COCO Segmentation dataset (one `_annotations.coco.json` per split).

Expected folder layout (default Roboflow COCO export):
    data/train/_annotations.coco.json
    data/train/img1.jpg ...
    data/valid/_annotations.coco.json
    data/test/_annotations.coco.json
"""
import os

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CocoSegmentationDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms

        # Map raw COCO category_id -> contiguous model label.
        # Model label 0 is reserved for background by torchvision's Mask R-CNN,
        # so real classes start at 1 (this matches classes.names in data.yaml:
        # 1=crop, 2=weed, as long as your annotations only contain those 2 cats).
        cat_ids = sorted(self.coco.getCatIds())
        self.catid_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_name = {
            self.catid_to_label[c["id"]]: c["name"] for c in self.coco.loadCats(cat_ids)
        }

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = [a for a in self.coco.loadAnns(ann_ids) if a.get("iscrowd", 0) == 0]

        boxes, labels, masks, areas = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid_to_label[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            masks.append(mask_utils.decode(self.coco.annToRLE(ann)))

        image = F.to_tensor(image)

        if len(boxes) == 0:
            # Image with no annotated instances still needs valid (empty) tensors,
            # or the Mask R-CNN loss functions will error out on this batch item.
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
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
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """Mask R-CNN takes lists of variable-sized images/targets, not a stacked batch tensor."""
    return tuple(zip(*batch))
