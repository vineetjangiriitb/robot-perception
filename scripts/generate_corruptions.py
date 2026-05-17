"""
Phase 3 — Corruption generator (ImageNet-C methodology).

Generates 25 corrupted copies of the test set:
  5 corruption types × 5 severity levels

Usage:
    python scripts/generate_corruptions.py

Output structure:
    data/corrupted/{corruption_type}/severity_{1-5}/
        *.jpg  (corrupted copies of test images)
    Labels are shared with the clean test set (corruptions don't move boxes).
"""

import os
import cv2
import numpy as np
from pathlib import Path

TEST_IMG_DIR = Path("data/annotated/images/test")
CORRUPTED_DIR = Path("data/corrupted")

CORRUPTIONS = ["gaussian_noise", "motion_blur", "gaussian_blur", "brightness", "occlusion"]
SEVERITIES = [1, 2, 3, 4, 5]


def apply_corruption(image: np.ndarray, corruption_type: str, severity: int) -> np.ndarray:
    """Apply one corruption at one severity level. severity: 1=mild, 5=severe."""
    s = severity

    if corruption_type == "gaussian_noise":
        std = [0.04, 0.08, 0.12, 0.18, 0.26][s - 1]
        noise = np.random.normal(0, std * 255, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif corruption_type == "motion_blur":
        k = [3, 5, 7, 11, 15][s - 1]
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(image, -1, kernel)

    elif corruption_type == "gaussian_blur":
        sigma = [0.5, 1.0, 1.5, 2.5, 4.0][s - 1]
        k = int(6 * sigma + 1) | 1  # must be odd
        return cv2.GaussianBlur(image, (k, k), sigma)

    elif corruption_type == "brightness":
        factor = [1.3, 1.6, 2.0, 2.5, 3.0][s - 1]
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    elif corruption_type == "occlusion":
        h, w = image.shape[:2]
        result = image.copy()
        n_patches = [1, 2, 3, 5, 8][s - 1]
        rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
        for _ in range(n_patches):
            pw, ph = int(w * 0.08), int(h * 0.08)
            x = rng.integers(0, max(1, w - pw))
            y = rng.integers(0, max(1, h - ph))
            result[y : y + ph, x : x + pw] = 0
        return result

    return image


def generate_all(test_img_dir: Path = TEST_IMG_DIR, output_dir: Path = CORRUPTED_DIR):
    img_paths = sorted(
        p for p in test_img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if not img_paths:
        print(f"No images found in {test_img_dir}. Run this after the test split is in place.")
        return

    total = len(CORRUPTIONS) * len(SEVERITIES)
    done = 0

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            out_dir = output_dir / corruption / f"severity_{severity}"
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_path in img_paths:
                dest = out_dir / img_path.name
                if dest.exists():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"  WARNING: cannot read {img_path}")
                    continue
                corrupted = apply_corruption(img, corruption, severity)
                cv2.imwrite(str(dest), corrupted)

            done += 1
            print(f"  [{done}/{total}] {corruption}/severity_{severity} — {len(img_paths)} images")

    print(f"\nDone. Corrupted sets written to {output_dir}/")
    print("Labels remain in data/annotated/labels/test/ (boxes don't change under corruption).")


if __name__ == "__main__":
    print("=== Phase 3: Generating corrupted test sets ===")
    generate_all()
