# CLAUDE.md — Robust + Uncertainty-Aware Robot Perception

## Project Overview

This project builds an uncertainty-aware humanoid robot component detector that:
1. Detects robot parts (arm, leg, torso, head) — 4 classes
2. Degrades gracefully under real-world corruptions (blur, noise, occlusion)
3. Outputs calibrated uncertainty alongside predictions

**Architecture:** RT-DETR (transformer-based, no NMS)
**Uncertainty:** MC Dropout (primary) + Conformal Prediction (required)
**Benchmark:** ImageNet-C methodology applied to the robot dataset

---

## Compute Infrastructure

All GPU training and heavy inference runs on **RunPod** via SSH. Jupyter notebooks are kept as reference/documentation only — actual execution happens through SSH terminal.

**Pod specs:**
- GPU: 1× RTX 3090 (24 GB VRAM)
- RAM: 125 GB | vCPU: 32 | Template: RunPod PyTorch 2.4.0 (CUDA 12.4.1, Ubuntu 22.04, Python 3.11)
- Container disk: 20 GB (ephemeral — erased on pod stop)
- Pricing: On-Demand $0.46/hr

**SSH workflow rules:**
1. **Always sync dataset and weights to pod before starting work** — container disk is wiped on stop
2. **Always sync results back to Mac before stopping pod** — use `rsync` or `scp`
3. **Scripts live in `scripts/`** — each phase has a dedicated Python script; run it via SSH, not in a notebook cell
4. **Stop the pod when training finishes** — leaving it idle burns $0.46/hr for nothing
5. **Never store irreplaceable outputs only on the pod** — treat container disk as a temp workspace

**Standard session workflow:**
```bash
# 1. SSH into pod
ssh root@<pod-ip> -p <port>

# 2. Sync dataset up (first time or after dataset changes)
rsync -avz --progress data/annotated/ root@<pod-ip>:<port>:/workspace/data/annotated/

# 3. Sync scripts up
rsync -avz scripts/ root@<pod-ip>:<port>:/workspace/scripts/

# 4. Run training / evaluation
python /workspace/scripts/train_baseline.py

# 5. Sync results back to Mac
rsync -avz root@<pod-ip>:<port>:/workspace/results/ ./results/
rsync -avz root@<pod-ip>:<port>:/workspace/models/ ./models/
```

**rsync shortcut** — add to your Mac's `~/.ssh/config`:
```
Host runpod
    HostName <pod-ip>
    Port <ssh-port>
    User root
    IdentityFile ~/.ssh/id_rsa
```
Then use `rsync -avz data/ runpod:/workspace/data/`

---

## CRITICAL: Commitments Made to Professor

The following outcomes were stated in the project proposal email to the professor. The project is **NOT complete** until all of these are achieved:

1. **Fine-tuned RT-DETR** on a 4-class humanoid robot dataset (arm, leg, torso, head — sensor class dropped due to absence in source dataset)
2. **Robustness benchmarked** across 25 corruption scenarios (5 corruption types × 5 severities, ImageNet-C methodology)
3. **Monte Carlo Dropout** with N=30 stochastic forward passes for uncertainty quantification
4. **ECE < 0.10** — uncertainty must be well-calibrated
5. **Uncertainty increases monotonically** with corruption severity (verified empirically)
6. **Conformal prediction** (error rate ε = 0.05) providing distribution-free prediction-set coverage guarantees
7. **≥ 95% empirical coverage** guaranteed at deployment

Do not mark any phase complete unless its metric targets are hit. Do not skip phases.

---

## Execution Plan

The complete phase-by-phase plan is in [`robust_perception_project_plan.md`](robust_perception_project_plan.md).
Every agent working on this project must treat this file as the authoritative source of truth.

### Phase targets (hard requirements)

| Phase | Key metric | Target |
|---|---|---|
| 1 — Dataset | Total annotated images | ≥ 300 |
| 1 — Dataset | Min examples per class (train) | ≥ 25 |
| 2 — Baseline | mAP@0.5 (clean test set) | ≥ 0.70 |
| 2 — Baseline | mAP@0.5:0.95 | ≥ 0.50 |
| 3 — Corruption benchmark | mPC recorded for all 25 sets | all 25 evaluated |
| 4 — Robust training | mPC improvement over baseline | ≥ 10% relative |
| 4 — Robust training | Clean mAP drop | < 0.03 |
| 5 — MC Dropout | ECE on clean test set | ≤ 0.10 |
| 5 — MC Dropout | Uncertainty correlation with error | Spearman ρ > 0.3 |
| 5 — MC Dropout | Uncertainty at severity 5 vs 1 | ≥ 1.5× |
| 6 — Conformal | Empirical coverage (clean test) | ≥ 0.95 |
| 6 — Conformal | Avg set size clean | ≤ 1.5 |
| 6 — Conformal | Avg set size severity 5 | > 2.0 |

**If mAP@0.5 < 0.60 after Phase 2, do NOT proceed to Phase 3.**
Diagnose annotation errors and class balance first.

---

## Required Visualizations

Each phase has mandatory visualizations that must be saved before marking the phase done.

| Phase | Visualization | File |
|---|---|---|
| 1 | Class distribution bar chart | `results/figures/viz1a_class_distribution.png` |
| 1 | 20 random annotated images | `results/figures/viz1b_annotation_sample.png` |
| 1 | Bounding box size distribution | `results/figures/viz1c_box_sizes.png` |
| 2 | Training curve (loss + val mAP) | `results/figures/viz2a_training_curve.png` |
| 2 | Confusion matrix on test set | `results/figures/viz2b_confusion_matrix.png` |
| 2 | 10 qualitative detections | `results/figures/viz2c_qualitative_detections.png` |
| 3 | Corruption strip (all types × severities) | `results/figures/viz3a_corruption_strip.png` |
| 3 | Robustness curve + heatmap | `results/figures/robustness_curve_baseline.png` |
| 4 | Side-by-side robustness comparison | `results/figures/robustness_comparison.png` |
| 5 | Reliability diagram | `results/figures/viz5a_reliability_diagram.png` |
| 5 | Uncertainty overlay on 6 images | `results/figures/viz5b_uncertainty_overlay.png` |
| 5 | Uncertainty vs severity | `results/figures/uncertainty_vs_severity.png` |
| 6 | Prediction set size distribution | `results/figures/viz6a_conformal_set_sizes.png` |
| 6 | Example prediction sets on images | `results/figures/viz6b_conformal_examples.png` |

---

## Repository Structure

```
robust-uncertainty-aware-robot-perception/
├── CLAUDE.md                          ← this file
├── robust_perception_project_plan.md  ← authoritative phase-by-phase plan
├── data/
│   ├── raw/              # original images
│   ├── annotated/        # YOLO-format labels (train/val/test/calibration splits)
│   └── corrupted/        # generated corruption sets (ephemeral on pod, not committed)
├── models/
│   ├── baseline/         # clean-trained weights (sync back from pod after Phase 2)
│   └── robust/           # aug-trained weights (sync back from pod after Phase 4)
├── notebooks/
│   ├── 00_colab_setup.ipynb           # DEPRECATED — kept for reference only
│   ├── 01_data_collection_and_audit.ipynb
│   ├── 02_baseline_train.ipynb        # reference only — use scripts/train_baseline.py on pod
│   ├── 03_corruption_benchmark.ipynb  # reference only — use scripts/evaluate_robustness.py on pod
│   ├── 04_robust_training.ipynb       # reference only — use scripts/train_robust.py on pod
│   ├── 05_mc_dropout.ipynb            # reference only — use scripts/mc_dropout_inference.py on pod
│   ├── 06_conformal.ipynb             # reference only — use scripts/conformal.py on pod
│   └── 07_analysis.ipynb
├── scripts/                           ← PRIMARY EXECUTION — run these on RunPod via SSH
│   ├── train_baseline.py              # Phase 2: train RT-DETR-L from COCO pretrained
│   ├── train_robust.py                # Phase 4: retrain with corruption augmentation
│   ├── generate_corruptions.py        # Phase 3: generate 25 corrupted test sets
│   ├── evaluate_robustness.py         # Phase 3+4: evaluate model on all 25 sets
│   ├── mc_dropout_inference.py        # Phase 5: N=30 stochastic passes + ECE
│   └── conformal.py                   # Phase 6: calibration + prediction sets
└── results/
    ├── robustness_baseline.json
    ├── robustness_robust.json
    └── figures/
```

---

## Dataset Split Requirements

```
Total: ≥ 300 images
Train:       70% (≥ 210 images)
Val:         15% (≥ 45 images)   — hyperparameter decisions only
Test:        15% (≥ 45 images)   — held out, touch only at evaluation time
Calibration: 100 images from train — required for conformal prediction (Phase 6)
```

**Classes (YOLO format IDs):**
```
0: arm
1: leg
2: torso/body
3: head
4: sensor/camera
```

---

## Model Specs

- **Architecture:** RT-DETR-L (use R50 if compute-limited), fine-tuned from COCO pretrained
- **Input size:** 640×640
- **Training epochs:** 100 (early stopping patience=20 on val mAP)
- **MC Dropout passes:** N=30 at inference
- **Conformal error rate:** ε = 0.05 (95% coverage target)

---

## Key Notes for All Agents

- Always start from COCO pretrained weights — never train from scratch
- The calibration set (100 images) must be carved from train in Phase 1 and never touched for training after that
- When evaluating robustness, save results to JSON before plotting (`results/robustness_baseline.json`, `results/robustness_robust.json`)
- Temperature scaling is the recommended fix if ECE > 0.15
- If MC Dropout uncertainty doesn't increase with corruption, verify `module.train()` is called on all Dropout layers
- For conformal quantile computation, use exactly: `q_level = np.ceil((n+1) * (1-epsilon)) / n`

---

## Parallel Execution & Agent Delegation

When planning next steps, AI must analyse the workflow for parallelisation opportunities before recommending a sequential plan.

**Required analysis every time a multi-step plan is proposed:**
1. List all upcoming tasks and their dependencies (what must finish before what can start)
2. Identify tasks with no dependency on each other — these are candidates for parallel execution
3. Estimate time saved by running them in parallel vs sequentially
4. If meaningful time can be saved (>20% of total estimated time), AI must proactively recommend spawning multiple agents and running those tasks in parallel — do not wait for the user to ask

**Format for the recommendation:**
- Show a dependency graph or table: task → depends on → can run in parallel with
- State the estimated sequential time vs parallel time
- Explicitly say which tasks to delegate to which agents and what each agent's prompt should cover

**Rule:** Never silently choose sequential execution when parallel is viable. Always surface the analysis and let the user decide.

---

## Git Workflow

- Create a new branch for each phase
- Merge to main only after all metric targets for that phase are hit
- Commit after each completed sub-task
- Keep `requirements.txt` and `README.md` up to date throughout

---

*Project owner: Vineet Jangir, IIT Bombay*
*Committed outcomes documented in professor communication, May 2026*
