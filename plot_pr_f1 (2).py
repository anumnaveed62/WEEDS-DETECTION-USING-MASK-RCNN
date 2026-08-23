"""
Plot precision, recall, F1-score (and a standard precision-recall curve) for
the weed detection model, using predictions already exported in COCO
"results" format (e.g. from detect_weeds.py --export_coco).

Unlike compute_map.py (which reports a single mAP number) or error_analysis.py
(which reports precision/recall at ONE fixed score threshold), this script
sweeps across all confidence thresholds so you can see how precision, recall,
and F1 trade off, and picks the threshold that maximizes F1.

Produces:
  results/metrics_vs_threshold.png   - precision / recall / F1 vs confidence threshold
  results/precision_recall_curve.png - standard precision-recall curve (with AP)
  results/per_class_metrics.csv      - precision/recall/F1/support per class at the best threshold
  results/metrics_summary.json       - best threshold + overall precision/recall/F1/AP

Usage:
    python plot_pr_f1.py --gt dataset/test/_annotations.coco.json \
        --pred results/inference_samples/predictions.json --iou_thresh 0.5
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO


def box_iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def greedy_match(coco_gt, preds, iou_thresh, category_id=None):
    """
    Standard COCO-style greedy matching: predictions are processed highest-score
    first; each is a TP if it hits an unclaimed ground-truth box of the same
    class at IoU >= iou_thresh, otherwise FP. Returns per-prediction TP flags
    (aligned with `preds`, already sorted by score descending) and the total
    number of ground-truth instances (recall denominator).
    """
    preds = sorted(preds, key=lambda p: -p.get("score", 1.0))
    if category_id is not None:
        preds = [p for p in preds if p["category_id"] == category_id]

    matched_gt = defaultdict(set)  # image_id -> set of matched gt ids
    tp_flags = []

    preds_by_image = defaultdict(list)
    for p in preds:
        preds_by_image[p["image_id"]].append(p)

    gt_cache = {}

    def get_gts(image_id):
        if image_id not in gt_cache:
            ann_ids = coco_gt.getAnnIds(imgIds=image_id, catIds=[category_id] if category_id else None)
            gt_cache[image_id] = coco_gt.loadAnns(ann_ids)
        return gt_cache[image_id]

    for p in preds:
        image_id = p["image_id"]
        gts = get_gts(image_id)
        best_iou, best_gt = 0.0, None
        for gt in gts:
            if gt["id"] in matched_gt[image_id]:
                continue
            if category_id is None and gt["category_id"] != p["category_id"]:
                continue
            iou = box_iou(p["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_gt = iou, gt
        if best_gt is not None and best_iou >= iou_thresh:
            matched_gt[image_id].add(best_gt["id"])
            tp_flags.append(1)
        else:
            tp_flags.append(0)

    if category_id is not None:
        n_gt = len(coco_gt.getAnnIds(catIds=[category_id]))
    else:
        n_gt = len(coco_gt.getAnnIds())

    scores = [p.get("score", 1.0) for p in preds]
    return np.array(scores), np.array(tp_flags), n_gt


def precision_recall_f1_curves(scores, tp_flags, n_gt):
    eps = 1e-9
    cum_tp = np.cumsum(tp_flags)
    cum_fp = np.cumsum(1 - tp_flags)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, eps)
    recall = cum_tp / max(n_gt, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, eps)
    return precision, recall, f1


def plot_metrics_vs_threshold(scores, precision, recall, f1, best_idx, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(scores, precision, label="Precision", color="#3498db", linewidth=1.8)
    plt.plot(scores, recall, label="Recall", color="#2ecc71", linewidth=1.8)
    plt.plot(scores, f1, label="F1-score", color="#e74c3c", linewidth=2.2)
    plt.axvline(scores[best_idx], color="gray", linestyle="--", linewidth=1,
                label=f"Best F1 @ score>={scores[best_idx]:.2f}")
    plt.xlabel("Confidence score threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Precision / Recall / F1 vs Confidence Threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_pr_curve(precision, recall, ap, out_path):
    plt.figure(figsize=(6, 6))
    order = np.argsort(recall)
    plt.plot(recall[order], precision[order], color="#9b59b6", linewidth=2)
    plt.fill_between(recall[order], precision[order], alpha=0.15, color="#9b59b6")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.05)
    plt.title(f"Precision-Recall Curve (AP={ap:.3f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def average_precision(precision, recall):
    """Area under the PR curve via the trapezoidal rule over recall-sorted points."""
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    return float(np.trapz(p, r))


def per_class_report(coco_gt, preds, iou_thresh, score_thresh):
    cat_ids = coco_gt.getCatIds()
    cat_names = {c["id"]: c["name"] for c in coco_gt.loadCats(cat_ids)}
    rows = []
    for cat_id in cat_ids:
        scores, tp_flags, n_gt = greedy_match(coco_gt, preds, iou_thresh, category_id=cat_id)
        keep = scores >= score_thresh
        tp = int(tp_flags[keep].sum())
        fp = int((1 - tp_flags[keep]).sum())
        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "class": cat_names[cat_id],
            "support_gt": n_gt,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to ground-truth COCO json (e.g. dataset/test/_annotations.coco.json)")
    parser.add_argument("--pred", required=True, help="Path to predictions in COCO results format")
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    coco_gt = COCO(args.gt)
    with open(args.pred) as f:
        preds = json.load(f)

    if not preds:
        raise ValueError("predictions file is empty - nothing to score")

    scores, tp_flags, n_gt = greedy_match(coco_gt, preds, args.iou_thresh)
    precision, recall, f1 = precision_recall_f1_curves(scores, tp_flags, n_gt)

    best_idx = int(np.argmax(f1))
    ap = average_precision(precision, recall)

    plot_metrics_vs_threshold(
        scores, precision, recall, f1, best_idx,
        os.path.join(args.out, "metrics_vs_threshold.png"),
    )
    plot_pr_curve(precision, recall, ap, os.path.join(args.out, "precision_recall_curve.png"))

    per_class = per_class_report(coco_gt, preds, args.iou_thresh, score_thresh=scores[best_idx])
    csv_path = os.path.join(args.out, "per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "support_gt", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(per_class)
    print(f"Saved {csv_path}")

    summary = {
        "iou_thresh": args.iou_thresh,
        "best_score_threshold": round(float(scores[best_idx]), 4),
        "precision_at_best": round(float(precision[best_idx]), 4),
        "recall_at_best": round(float(recall[best_idx]), 4),
        "f1_at_best": round(float(f1[best_idx]), 4),
        "average_precision": round(ap, 4),
        "n_ground_truth_instances": n_gt,
        "n_predictions": len(preds),
        "per_class": per_class,
    }
    summary_path = os.path.join(args.out, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest F1={summary['f1_at_best']:.4f} at score>={summary['best_score_threshold']:.2f} "
          f"(precision={summary['precision_at_best']:.4f}, recall={summary['recall_at_best']:.4f})")
    print(f"Average Precision (AP@{args.iou_thresh}) = {ap:.4f}")
    print(f"\nPer-class breakdown (at best threshold):")
    for row in per_class:
        print(f"  {row['class']:<25} P={row['precision']:.3f}  R={row['recall']:.3f}  "
              f"F1={row['f1']:.3f}  support={row['support_gt']}")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
