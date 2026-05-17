"""
Phase 3 & 4 — Robustness evaluator.

Runs a trained RT-DETR model on all 25 corrupted test sets and saves
the mAP@0.5 grid to JSON. Also computes mPC (mean Performance under Corruption).

Usage:
    # Evaluate baseline model
    python scripts/evaluate_robustness.py --weights models/baseline/best.pt --out results/robustness_baseline.json

    # Evaluate robust (aug-trained) model
    python scripts/evaluate_robustness.py --weights models/robust/best.pt --out results/robustness_robust.json
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

CORRUPTIONS = ["gaussian_noise", "motion_blur", "gaussian_blur", "brightness", "occlusion"]
SEVERITIES = [1, 2, 3, 4, 5]

CORRUPTED_DIR = Path("data/corrupted")
LABEL_DIR = Path("data/annotated/labels/test")  # labels shared across all corruptions
DATASET_YAML = Path("robot_parts.yaml")


def build_temp_yaml(corruption: str, severity: int, tmp_dir: Path) -> Path:
    """
    Build a temporary dataset yaml for one corrupted set.

    `path:` points at the severity dir; train/val/test all = 'images'.
    With split='test', Ultralytics resolves {sev}/images/*.jpg and then
    finds labels by replacing '/images/' with '/labels/' → {sev}/labels/*.txt.
    train/val are set only so the yaml validates; they are never read.
    """
    sev_dir = (CORRUPTED_DIR / corruption / f"severity_{severity}").resolve()
    yaml_content = f"""path: {sev_dir}
train: images
val: images
test: images

nc: 5
names: ['arm', 'leg', 'torso', 'head', 'sensor']
"""
    yaml_path = tmp_dir / f"{corruption}_s{severity}.yaml"
    yaml_path.write_text(yaml_content)
    return yaml_path


def evaluate_model(weights_path: str, output_json: str):
    from ultralytics import RTDETR

    model = RTDETR(weights_path)

    # First: evaluate on clean test set
    print("\n--- Clean test set ---")
    clean_metrics = model.val(data=str(DATASET_YAML), split="test", verbose=False)
    clean_mAP = float(clean_metrics.box.map50)
    print(f"Clean mAP@0.5: {clean_mAP:.4f}")

    results_grid = {}
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        total = len(CORRUPTIONS) * len(SEVERITIES)
        done = 0

        for corruption in CORRUPTIONS:
            results_grid[corruption] = {}
            for severity in SEVERITIES:
                sev_dir = CORRUPTED_DIR / corruption / f"severity_{severity}"
                img_dir = sev_dir / "images"
                lbl_dir = sev_dir / "labels"
                if not img_dir.exists():
                    print(f"  MISSING: {img_dir} — run generate_corruptions.py first")
                    results_grid[corruption][severity] = None
                    continue

                # Drop any stale Ultralytics label cache so counts stay correct
                for cache in lbl_dir.glob("*.cache"):
                    cache.unlink()

                yaml_path = build_temp_yaml(corruption, severity, tmp_dir)

                try:
                    metrics = model.val(data=str(yaml_path), split="test", verbose=False)
                    mAP = float(metrics.box.map50)
                except Exception as e:
                    print(f"  ERROR on {corruption}/severity_{severity}: {e}")
                    mAP = 0.0

                results_grid[corruption][severity] = mAP
                done += 1
                print(f"  [{done}/{total}] {corruption}/severity_{severity}: mAP={mAP:.4f}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Compute mPC
    all_mAPs = [
        results_grid[c][s]
        for c in CORRUPTIONS
        for s in SEVERITIES
        if results_grid[c][s] is not None
    ]
    mPC = float(np.mean(all_mAPs))
    relative_mPC = mPC / clean_mAP if clean_mAP > 0 else 0.0

    output = {
        "weights": weights_path,
        "clean_mAP50": clean_mAP,
        "mPC": mPC,
        "relative_mPC": relative_mPC,
        "results_grid": {c: {str(s): v for s, v in sv.items()} for c, sv in results_grid.items()},
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Robustness Summary ===")
    print(f"Clean mAP@0.5:  {clean_mAP:.4f}")
    print(f"mPC:            {mPC:.4f}")
    print(f"Relative mPC:   {relative_mPC:.4f}  ({relative_mPC*100:.1f}% of clean retained)")
    print(f"Saved to: {output_json}")

    # Worst corruption
    per_corruption_mPC = {
        c: np.mean([results_grid[c][s] for s in SEVERITIES if results_grid[c][s] is not None])
        for c in CORRUPTIONS
    }
    worst = min(per_corruption_mPC, key=per_corruption_mPC.get)
    print(f"Worst corruption: {worst} (avg mAP={per_corruption_mPC[worst]:.4f})")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()
    evaluate_model(args.weights, args.out)
