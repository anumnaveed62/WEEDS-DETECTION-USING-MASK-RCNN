"""
Simple joint image+target transforms for Mask R-CNN training on COCO-format
instance segmentation data. Unlike torchvision.transforms, these operate on
(image, target) pairs together so that boxes/masks stay aligned with the
image after geometric transforms like flips.
"""
import random


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Flip left-right. Useful since field photos have no fixed left/right orientation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            image = image.flip(-1)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class RandomVerticalFlip:
    """Flip top-bottom. Useful for overhead/drone field imagery which has no fixed 'up'."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, h, w = image.shape
            image = image.flip(-2)
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes
            if target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-2)
        return image, target


def get_train_transforms():
    return Compose([RandomHorizontalFlip(0.5), RandomVerticalFlip(0.2)])


def get_eval_transforms():
    return Compose([])
