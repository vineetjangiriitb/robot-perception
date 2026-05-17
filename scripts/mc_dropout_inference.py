"""
Phase 5 — Monte Carlo Dropout inference.

Runs N stochastic forward passes with dropout kept active at inference time.
Outputs per-detection uncertainty scores (std dev of confidence across passes).

Usage:
    from scripts.mc_dropout_inference import mc_dropout_inference, compute_ECE

    detections = mc_dropout_inference(model, image_tensor, n_passes=30)
    ece = compute_ECE(predictions_with_correctness)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enable dropout at inference
# ---------------------------------------------------------------------------

def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Keep all Dropout layers in train mode so they remain stochastic at inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def verify_mc_dropout_active(model: nn.Module, image: torch.Tensor, device: str = "mps") -> bool:
    """
    Sanity check: run the same image twice; outputs should differ if dropout is active.
    Returns True if MC Dropout is working correctly.
    """
    model.eval()
    enable_mc_dropout(model)

    with torch.no_grad():
        out1 = model(image.to(device))
        out2 = model(image.to(device))

    # Compare first detection confidence between passes
    try:
        c1 = float(out1[0].boxes.conf[0]) if len(out1[0].boxes) > 0 else 0.0
        c2 = float(out2[0].boxes.conf[0]) if len(out2[0].boxes) > 0 else 0.0
        active = abs(c1 - c2) > 1e-6
        if active:
            print(f"MC Dropout ACTIVE — pass1 conf={c1:.4f}, pass2 conf={c2:.4f}")
        else:
            print(f"WARNING: MC Dropout may not be active — outputs identical ({c1:.4f})")
            print("Check that the model has Dropout layers and they are set to train mode.")
        return active
    except (IndexError, AttributeError):
        print("Could not verify (no detections on this image). Try a different image.")
        return False


# ---------------------------------------------------------------------------
# Multi-pass inference + aggregation
# ---------------------------------------------------------------------------

def mc_dropout_inference(
    model,
    image: torch.Tensor,
    n_passes: int = 30,
    iou_match_threshold: float = 0.4,
    device: str = "mps",
) -> list[dict]:
    """
    Run N stochastic forward passes and aggregate per-box uncertainty.

    Args:
        model: Ultralytics RTDETR model (already loaded)
        image: image tensor [1, 3, H, W] or PIL Image / numpy array
        n_passes: number of stochastic forward passes (30 recommended)
        iou_match_threshold: IoU threshold to match same box across passes

    Returns:
        List of dicts, one per detection in the reference pass:
            {
                'box': [x1, y1, x2, y2],      # pixel coords from reference pass
                'class': int,
                'mean_confidence': float,       # mean across passes
                'uncertainty': float,           # std dev across passes
                'confidences': list[float],     # raw per-pass values
            }
    """
    model.eval()
    enable_mc_dropout(model)

    all_passes = []
    with torch.no_grad():
        for _ in range(n_passes):
            results = model(image, verbose=False)
            all_passes.append(results[0])  # single image

    # Use pass 0 as the reference (anchor boxes)
    ref = all_passes[0]
    if len(ref.boxes) == 0:
        return []

    ref_boxes = ref.boxes.xyxy.cpu().numpy()      # [N, 4]
    ref_classes = ref.boxes.cls.cpu().numpy().astype(int)

    aggregated = []
    for i, (box, cls) in enumerate(zip(ref_boxes, ref_classes)):
        confs = []
        for pass_result in all_passes:
            if len(pass_result.boxes) == 0:
                confs.append(0.0)
                continue
            # Find best IoU match in this pass
            candidate_boxes = pass_result.boxes.xyxy.cpu().numpy()
            candidate_confs = pass_result.boxes.conf.cpu().numpy()
            best_iou = 0.0
            best_conf = 0.0
            for j, (cbox, cconf) in enumerate(zip(candidate_boxes, candidate_confs)):
                iou = _box_iou(box, cbox)
                if iou > best_iou:
                    best_iou = iou
                    best_conf = float(cconf)
            confs.append(best_conf if best_iou >= iou_match_threshold else 0.0)

        aggregated.append({
            "box": box.tolist(),
            "class": int(cls),
            "mean_confidence": float(np.mean(confs)),
            "uncertainty": float(np.std(confs)),
            "confidences": confs,
        })

    return aggregated


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def compute_ECE(predictions: list[dict], n_buckets: int = 10) -> float:
    """
    Compute Expected Calibration Error.

    Each prediction dict must have:
        'mean_confidence': float in [0, 1]
        'is_correct': bool — True if IoU with ground truth ≥ 0.5

    Returns ECE (lower is better; target ≤ 0.10).
    """
    bucket_size = 1.0 / n_buckets
    ece = 0.0
    n = len(predictions)
    if n == 0:
        return 0.0

    for i in range(n_buckets):
        low = i * bucket_size
        high = (i + 1) * bucket_size
        bucket = [p for p in predictions if low <= p["mean_confidence"] < high]
        if not bucket:
            continue
        avg_conf = np.mean([p["mean_confidence"] for p in bucket])
        avg_acc = np.mean([float(p["is_correct"]) for p in bucket])
        ece += (len(bucket) / n) * abs(avg_conf - avg_acc)

    return float(ece)


# ---------------------------------------------------------------------------
# Quick CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python mc_dropout_inference.py <weights.pt> <image_path>")
        sys.exit(1)

    from ultralytics import RTDETR

    weights, img_path = sys.argv[1], sys.argv[2]
    model = RTDETR(weights)

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_path}")
        sys.exit(1)

    print(f"Running {30} MC Dropout passes on {img_path}...")
    detections = mc_dropout_inference(model, img_path, n_passes=30)

    print(f"\nDetections: {len(detections)}")
    for d in detections:
        print(f"  class={d['class']}  mean_conf={d['mean_confidence']:.3f}  uncertainty={d['uncertainty']:.3f}")
