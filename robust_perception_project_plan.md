# Robust + Uncertainty-Aware Robot Perception
## Project Execution Plan

> **Analogy:** This document is your textbook. Each Phase = a Chapter. Each metric target = a problem set. You know you've completed the chapter when you hit the target. You know how well you did from the numbers.

---

## What This Project Builds

A humanoid robot component detector (arms, legs, cameras, sensors) that:
1. Works well on clean images (baseline)
2. Degrades **gracefully** under real-world corruptions — blur, noise, occlusion — not catastrophically
3. Outputs **calibrated uncertainty** alongside predictions — so downstream systems know when to trust the detector

**Architecture:** RT-DETR (transformer-based, no NMS, more principled than YOLO for uncertainty heads)  
**Uncertainty method:** MC Dropout (primary) + optional Conformal Prediction (stretch goal)  
**Corruption benchmark:** ImageNet-C methodology applied to your robot dataset

---

## Project Phases Overview

| Phase | Name | What you build | Signal you're done |
|---|---|---|---|
| 0 | Setup & Literature | Environment, papers read | Checklist complete |
| 1 | Dataset | Clean annotated dataset | ≥300 images, mAP baseline |
| 2 | Baseline Detector | RT-DETR trained, clean perf | mAP ≥ 0.70 on clean test set |
| 3 | Corruption Benchmark | Robustness curve | mPC metric across 5 corruption types |
| 4 | Robust Training | Aug-improved model | mPC improves by ≥ 10% over baseline |
| 5 | MC Dropout Head | Uncertainty outputs | ECE ≤ 0.10, uncertainty correlates with error |
| 6 | Conformal Prediction | Guaranteed prediction sets | Coverage ≥ 95% at ε = 0.05 |
| 7 | Analysis & Write-up | Repo + resume bullet | Clean README, 3 figures, bullet drafted |

---

## Phase 0 — Setup & Literature

**Goal:** Know the terrain before writing a single line of code.

### 0.1 Environment Setup

**Compute platform: RunPod RTX 3090 via SSH** (replaces Colab)

```bash
# SSH into your RunPod pod
ssh root@<pod-ip> -p <port>

# Pod already has PyTorch 2.4.0 + CUDA 12.4.1 installed via template
# Install project dependencies
pip install ultralytics
pip install albumentations       # corruption augmentations
pip install scikit-learn matplotlib pandas
pip install scipy                # for ECE + conformal quantile computation

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 3090
```

**Sync your project to the pod:**
```bash
# From your Mac (run once per pod session)
rsync -avz --progress \
  /Users/vineetjangir/robust\&uncertainty-aware-robot-perception/data/annotated/ \
  root@<pod-ip>:<port>:/workspace/data/annotated/

rsync -avz --progress \
  /Users/vineetjangir/robust\&uncertainty-aware-robot-perception/scripts/ \
  root@<pod-ip>:<port>:/workspace/scripts/
```

Set up your project repo:
```
robot-perception/
├── data/
│   ├── raw/              # original images
│   ├── annotated/        # YOLO-format labels
│   └── corrupted/        # generated corruption sets
├── models/
│   ├── baseline/         # clean-trained weights
│   └── robust/           # aug-trained weights
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_train.ipynb
│   ├── 03_corruption_benchmark.ipynb
│   ├── 04_mc_dropout.ipynb
│   └── 05_conformal.ipynb
├── scripts/
│   ├── generate_corruptions.py
│   ├── evaluate_robustness.py
│   └── mc_dropout_inference.py
└── results/
    └── figures/
```

### 0.2 Papers to Read (in this order)

| Paper | Why | Time |
|---|---|---|
| **RT-DETR** (Zhao et al., Baidu 2023) — "DETRs Beat YOLOs on Real-time Object Detection" | Your base architecture | 1.5 hrs |
| **ImageNet-C** (Hendrycks & Dietterich 2019) — "Benchmarking Neural Network Robustness to Common Corruptions" | Your corruption benchmark method | 1 hr |
| **MC Dropout** (Gal & Ghahramani, 2016) — "Dropout as a Bayesian Approximation" | Your uncertainty method | 1.5 hrs |
| **Conformal Prediction** (Angelopoulos & Bates 2021) — "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" | Stretch goal, but read it | 1 hr |

### ✅ Phase 0 Completion Checklist

- [ ] Repo initialized and pushed to GitHub
- [ ] Environment working: `import torch; torch.cuda.is_available()` returns True
- [ ] RT-DETR repo cloned and demo inference runs without error
- [ ] All 4 papers read — for each paper write 3 sentences: (1) what it proposes, (2) the key mechanism, (3) how it applies to your project. Put this in a `notes/papers.md` file.

**Problem Set 0:** In your own words, answer these in `notes/papers.md`:
1. Why does RT-DETR not need NMS? What does it use instead?
2. What is the difference between aleatoric and epistemic uncertainty?
3. How does ImageNet-C generate corruptions? Name 3 corruption types and their severity levels.
4. What is the nonconformity score in conformal prediction?

---

## Phase 1 — Dataset Construction

**Goal:** A clean, well-annotated dataset of humanoid robot components ready for training and evaluation.

### 1.1 Dataset Sourcing

You already have a starting set from your YOLO project. Expand it:

**Sources:**
- Your existing annotated dataset (starting point)
- Google Images / Bing Images search: "humanoid robot arm", "Boston Dynamics Spot leg", "robot torso", "humanoid robot joint"
- YouTube frames: extract frames from Boston Dynamics videos, Figure AI demos, 1X Technologies footage
- Roboflow Universe: search for existing robot part datasets (can merge with yours)
- Synthetic: Blender renders of robot meshes (optional, high effort)

**Target class taxonomy (keep it tight):**
```
0: arm
1: leg
2: torso/body
3: head
4: sensor/camera
```
5 classes maximum. Too many classes = insufficient examples per class at ≤500 images.

### 1.2 Annotation

Use **Roboflow** (free tier) or **LabelImg** for annotation in YOLO format.

Annotation rules:
- Tight bounding boxes — no large padding
- Partial objects: annotate if ≥ 50% of the part is visible
- Overlapping parts: annotate each separately
- Minimum 20 examples per class in test set

### 1.3 Dataset Split

```
Total: ≥ 300 images
Train: 70% (≥ 210 images)
Val:   15% (≥ 45 images)   ← used for hyperparameter decisions
Test:  15% (≥ 45 images)   ← held out, only touch at evaluation time
Calibration: carve 100 images from train ← needed for conformal prediction in Phase 6
```

### 1.4 Dataset Audit

Before training, run a data audit notebook. **All four visualizations below are mandatory — do not skip any.**

#### VIZ 1.A — Class distribution bar chart
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections, os, random
import cv2, numpy as np

CLASS_NAMES = ['arm', 'leg', 'torso', 'head', 'sensor']

def count_class_distribution(labels_dir):
    counts = collections.Counter()
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(labels_dir, label_file)) as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counts[class_id] += 1
    return counts

train_counts = count_class_distribution('data/annotated/labels/train')
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(CLASS_NAMES, [train_counts[i] for i in range(5)], color='steelblue', edgecolor='white')
ax.axhline(y=25, color='red', linestyle='--', label='min threshold (25)')
for bar, count in zip(bars, [train_counts[i] for i in range(5)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(count), ha='center', va='bottom', fontsize=11)
ax.set_ylabel('Number of annotated instances (train set)')
ax.set_title('Class distribution — training set')
ax.legend()
plt.tight_layout()
plt.savefig('results/figures/viz1a_class_distribution.png', dpi=150)
plt.show()
# ⚠️  MANUAL CHECK: Any bar below the red line = not enough data for that class. Fix before training.
```

#### VIZ 1.B — 20 random annotated images (ground truth boxes overlaid)
```python
def draw_annotations(img_path, label_path, class_names):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    colors = [(255,80,80),(80,200,80),(80,80,255),(255,200,0),(200,80,255)]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw/2) * w); y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w); y2 = int((cy + bh/2) * h)
            color = colors[cls]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img, class_names[cls], (x1, max(y1-6,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img

# Sample 20 random images from train set
img_dir = 'data/annotated/images/train'
lbl_dir = 'data/annotated/labels/train'
img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.png'))]
sample = random.sample(img_files, min(20, len(img_files)))

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
for ax, fname in zip(axes.flatten(), sample):
    stem = os.path.splitext(fname)[0]
    lbl_path = os.path.join(lbl_dir, stem + '.txt')
    img_path = os.path.join(img_dir, fname)
    img = draw_annotations(img_path, lbl_path, CLASS_NAMES)
    ax.imshow(img)
    ax.set_title(fname[:20], fontsize=8)
    ax.axis('off')
plt.suptitle('VIZ 1.B — 20 random annotated training images', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/viz1b_annotation_sample.png', dpi=120)
plt.show()
# ⚠️  MANUAL CHECK: Look at every box. Are the boxes tight? Are class labels correct?
#    Common errors: wrong class ID, box covering the whole image, missing annotations.
#    Fix any errors in Roboflow/LabelImg before proceeding.
```

#### VIZ 1.C — Bounding box size distribution (catches annotation errors)
```python
widths, heights = [], []
for label_file in os.listdir('data/annotated/labels/train'):
    if not label_file.endswith('.txt'): continue
    with open(os.path.join('data/annotated/labels/train', label_file)) as f:
        for line in f:
            parts = line.strip().split()
            widths.append(float(parts[3]))   # normalized width
            heights.append(float(parts[4]))  # normalized height

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths, bins=30, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Normalized box width'); axes[0].set_ylabel('Count')
axes[0].set_title('Box width distribution')
axes[0].axvline(x=0.8, color='red', linestyle='--', label='suspiciously large (>0.8)')
axes[0].legend()

axes[1].hist(heights, bins=30, color='darkorange', edgecolor='white')
axes[1].set_xlabel('Normalized box height'); axes[1].set_ylabel('Count')
axes[1].set_title('Box height distribution')
axes[1].axvline(x=0.8, color='red', linestyle='--', label='suspiciously large (>0.8)')
axes[1].legend()

plt.suptitle('VIZ 1.C — Bounding box size distribution')
plt.tight_layout()
plt.savefig('results/figures/viz1c_box_sizes.png', dpi=150)
plt.show()
# ⚠️  MANUAL CHECK: Boxes with width or height > 0.8 are likely annotation errors
#    (box covering entire image). Boxes < 0.02 are likely too small to be useful.
```

### ✅ Phase 1 Completion Checklist

- [ ] ≥ 300 images collected
- [ ] All images annotated in YOLO format
- [ ] Train/val/test/calibration split created and saved
- [ ] **VIZ 1.A saved** — class distribution bar chart, all bars above red threshold line
- [ ] **VIZ 1.B saved** — 20 annotated images visually inspected, boxes look correct and tight
- [ ] **VIZ 1.C saved** — box size distribution looks reasonable (no mass of boxes > 0.8)
- [ ] Dataset pushed to a private Roboflow project OR stored in Google Drive with a reproducible download script

**Problem Set 1 (Metrics):**

| Metric | Target | How to measure |
|---|---|---|
| Total annotated images | ≥ 300 | count files |
| Min examples per class (train) | ≥ 25 | class distribution plot |
| Annotation coverage | 100% of images have ≥ 1 label | script check |
| Zero-shot YOLOv8n mAP@0.5 | Record it, any value | run inference on test set, establishes baseline-of-baselines |

> **The zero-shot mAP on your test set is your first real number.** It tells you how much work the fine-tuning has to do.

---

## Phase 2 — Baseline Detector (Clean Performance)

**Goal:** Train RT-DETR on your clean dataset and establish a solid clean-image performance number. This is your Chapter 2 exam score that everything else is measured against.

### 2.1 Model Setup

Use RT-DETR-L (large variant). RTX 3090 has 24 GB VRAM — use batch=16.

**Run on RunPod via SSH:**
```bash
# On pod
python /workspace/scripts/train_baseline.py
```

`scripts/train_baseline.py` does:
```python
from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt')  # pretrained on COCO, auto-downloaded
results = model.train(
    data='/workspace/data/annotated/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,          # RTX 3090 can handle 16
    lr0=1e-4,
    weight_decay=1e-4,
    warmup_epochs=3,
    patience=20,
    device=0,
    project='/workspace/models/baseline',
    name='run1'
)
```

**After training finishes — sync back to Mac before stopping pod:**
```bash
rsync -avz root@<pod-ip>:<port>:/workspace/models/baseline/ ./models/baseline/
rsync -avz root@<pod-ip>:<port>:/workspace/results/ ./results/
```

Your `robot_parts.yaml`:
```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

nc: 5
names: ['arm', 'leg', 'torso', 'head', 'sensor']
```

### 2.2 Training Protocol

- **Epochs:** 100 (use early stopping with patience=20 on val mAP)
- **Pretrained weights:** Always start from COCO pretrained — you're fine-tuning, not training from scratch
- **Monitor:** Log val mAP@0.5 and val loss every epoch. Use W&B or just matplotlib.
- **Save:** Best checkpoint by val mAP@0.5

### 2.3 Evaluation on Clean Test Set

```python
# After training, evaluate on test set
metrics = model.val(data='robot_parts.yaml', split='test')
print(metrics.box.map50)    # mAP@0.5
print(metrics.box.map)      # mAP@0.5:0.95
print(metrics.box.p)        # Precision per class
print(metrics.box.r)        # Recall per class
```

#### VIZ 2.A — Training curve
```python
# If using Ultralytics, results are saved automatically to runs/detect/train/results.csv
import pandas as pd

results_df = pd.read_csv('runs/detect/train/results.csv')
results_df.columns = results_df.columns.str.strip()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(results_df['epoch'], results_df['train/box_loss'], label='train box loss')
axes[0].plot(results_df['epoch'], results_df['val/box_loss'], label='val box loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('VIZ 2.A — Box loss over training')
axes[0].legend()

axes[1].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='val mAP@0.5', color='green')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mAP@0.5')
axes[1].set_title('Validation mAP over training')
axes[1].axhline(y=0.70, color='red', linestyle='--', label='target (0.70)')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/figures/viz2a_training_curve.png', dpi=150)
plt.show()
# ⚠️  MANUAL CHECK: Is val mAP still rising at the end? If yes, train longer.
#    Is val loss rising while train loss falls? → Overfitting, add regularization.
#    Did mAP plateau early (< epoch 30)? → Check learning rate.
```

#### VIZ 2.B — Confusion matrix on test set
```python
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Collect true and predicted class labels from test set detections
# A detection is matched to ground truth if IoU ≥ 0.5
y_true, y_pred = [], []
for img, labels in test_set:
    preds = model(img)
    for gt_box, gt_cls in zip(labels['boxes'], labels['classes']):
        best_pred_cls = match_gt_to_prediction(gt_box, preds)  # returns class or None
        if best_pred_cls is not None:
            y_true.append(gt_cls)
            y_pred.append(best_pred_cls)

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('VIZ 2.B — Confusion matrix (test set)')
plt.tight_layout()
plt.savefig('results/figures/viz2b_confusion_matrix.png', dpi=150)
plt.show()
# ⚠️  MANUAL CHECK: The diagonal should be brightest. Off-diagonal bright cells =
#    the model confuses those two classes. Common: arm/leg confusion (visually similar).
#    If a whole row is dark → model never predicts that class (recall problem).
#    If a whole column is dark → model never detects those as that class (precision problem).
```

#### VIZ 2.C — 10 qualitative detection examples
```python
# Run inference on 10 test images and display predictions alongside ground truth
sample = random.sample(list(test_set), 10)
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

for ax, (img, labels) in zip(axes.flatten(), sample):
    preds = model(img)
    vis = draw_predictions_vs_gt(img, preds, labels, CLASS_NAMES)
    # Color code: GREEN box = ground truth, BLUE box = prediction
    ax.imshow(vis)
    ax.axis('off')

plt.suptitle('VIZ 2.C — Green: ground truth | Blue: predictions\n'
             'Look for: correct boxes, false positives, missed objects', fontsize=12)
plt.tight_layout()
plt.savefig('results/figures/viz2c_qualitative_detections.png', dpi=120)
plt.show()
# ⚠️  MANUAL CHECK: For each image, classify it as:
#    GOOD — boxes match ground truth tightly
#    FALSE POSITIVE — blue box with no matching green box
#    MISS — green box with no matching blue box
#    WRONG CLASS — box in right place but wrong label color
#    If misses dominate → lower confidence threshold. If FPs dominate → raise it.
```

Plot and save all three before evaluating metrics.

### ✅ Phase 2 Completion Checklist

- [ ] RT-DETR fine-tuned and best checkpoint saved
- [ ] Test set evaluation complete
- [ ] **VIZ 2.A saved** — training curve (loss + val mAP over epochs): val mAP trending upward, not plateauing at epoch 10
- [ ] **VIZ 2.B saved** — confusion matrix on test set: diagonal should dominate
- [ ] **VIZ 2.C saved** — 10 qualitative detections manually inspected, labeled as: correct / false positive / miss

**Problem Set 2 (Metrics):**

| Metric | Target | Notes |
|---|---|---|
| mAP@0.5 (clean test set) | ≥ 0.70 | If < 0.60, diagnose before moving on |
| mAP@0.5:0.95 (clean test set) | ≥ 0.50 | Stricter IoU thresholds |
| Precision per class | All ≥ 0.65 | Flag any class that underperforms |
| Recall per class | All ≥ 0.60 | Flag any class that underperforms |
| Inference speed (ms/image) | ≤ 50ms | On your GPU |

> **If mAP@0.5 < 0.60:** Do NOT proceed to Phase 3. Diagnose: (a) check for annotation errors in the failing classes, (b) check class balance, (c) try training for 50 more epochs. Phase 3 robustness numbers are meaningless if the clean baseline is weak.

---

## Phase 3 — Corruption Benchmark (The Robustness Curve)

**Goal:** Systematically measure how much your clean model degrades under different corruptions. This is the scientific contribution of the project — you're producing a robustness profile, not just a single number.

### 3.1 Corruption Types to Implement

Apply ImageNet-C methodology to your test set. For each corruption, implement 5 severity levels (1=mild, 5=severe).

```python
import albumentations as A
import cv2
import numpy as np

def apply_corruption(image, corruption_type, severity):
    """
    Apply a specific corruption at a specific severity to an image.
    severity: 1-5 (1=mild, 5=severe)
    """
    s = severity  # shorthand
    
    if corruption_type == 'gaussian_noise':
        std = [0.04, 0.08, 0.12, 0.18, 0.26][s-1]
        noise = np.random.normal(0, std * 255, image.shape).astype(np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    elif corruption_type == 'motion_blur':
        k = [3, 5, 7, 11, 15][s-1]
        kernel = np.zeros((k, k))
        kernel[k//2, :] = 1.0 / k
        return cv2.filter2D(image, -1, kernel)
    
    elif corruption_type == 'gaussian_blur':
        sigma = [0.5, 1.0, 1.5, 2.5, 4.0][s-1]
        k = int(6*sigma+1) | 1  # must be odd
        return cv2.GaussianBlur(image, (k, k), sigma)
    
    elif corruption_type == 'brightness':
        factor = [1.3, 1.6, 2.0, 2.5, 3.0][s-1]
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    elif corruption_type == 'occlusion':
        # Random rectangular patches zeroed out
        h, w = image.shape[:2]
        result = image.copy()
        n_patches = [1, 2, 3, 5, 8][s-1]
        for _ in range(n_patches):
            pw, ph = int(w*0.08), int(h*0.08)
            x = np.random.randint(0, w - pw)
            y = np.random.randint(0, h - ph)
            result[y:y+ph, x:x+pw] = 0
        return result
    
    return image

CORRUPTIONS = ['gaussian_noise', 'motion_blur', 'gaussian_blur', 'brightness', 'occlusion']
SEVERITIES = [1, 2, 3, 4, 5]
```

### 3.2 Generate Corrupted Test Sets

```python
# For each corruption × severity, generate a corrupted copy of your test set
# Save to: data/corrupted/{corruption_type}/severity_{s}/

for corruption in CORRUPTIONS:
    for severity in SEVERITIES:
        output_dir = f"data/corrupted/{corruption}/severity_{severity}"
        os.makedirs(output_dir, exist_ok=True)
        for img_path in test_images:
            img = cv2.imread(img_path)
            corrupted = apply_corruption(img, corruption, severity)
            cv2.imwrite(f"{output_dir}/{os.path.basename(img_path)}", corrupted)
```

#### VIZ 3.A — Corruption strip (visual sanity check before evaluating anything)

**Run this before any model evaluation.** You need to see what your corruptions actually look like at each severity level.

```python
# Pick one representative test image and show all corruptions × severities in a grid
sample_img_path = test_images[0]
sample_img = cv2.imread(sample_img_path)
sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(len(CORRUPTIONS), len(SEVERITIES) + 1,
                          figsize=(18, len(CORRUPTIONS) * 3))

for i, corruption in enumerate(CORRUPTIONS):
    # Column 0: original
    axes[i, 0].imshow(sample_img)
    axes[i, 0].set_title('Original' if i == 0 else '', fontsize=9)
    axes[i, 0].set_ylabel(corruption, fontsize=10, fontweight='bold')
    axes[i, 0].axis('off')

    for j, severity in enumerate(SEVERITIES):
        corrupted = apply_corruption(sample_img.copy(), corruption, severity)
        axes[i, j+1].imshow(corrupted)
        axes[i, j+1].set_title(f'S{severity}' if i == 0 else '', fontsize=9)
        axes[i, j+1].axis('off')

plt.suptitle('VIZ 3.A — Corruption strip: all types × all severities', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/viz3a_corruption_strip.png', dpi=120)
plt.show()
# ⚠️  MANUAL CHECK: Does severity 1 look mild and severity 5 look severe?
#    Does occlusion look like realistic partial blocking, not just random noise?
#    Does motion blur look like what you'd see from a fast-moving robot limb?
#    If any severity looks wrong, fix apply_corruption() before evaluating.
```

### 3.3 Evaluate Baseline Model on All Corrupted Sets

```python
results_grid = {}  # {corruption: {severity: mAP}}

for corruption in CORRUPTIONS:
    results_grid[corruption] = {}
    for severity in SEVERITIES:
        # Run your baseline model on this corrupted test set
        # Record mAP@0.5
        mAP = evaluate_model(model, f"data/corrupted/{corruption}/severity_{severity}")
        results_grid[corruption][severity] = mAP
```

### 3.4 Compute the mPC Metric

mPC = mean Performance under Corruption. This is your single headline robustness number.

```python
# Mean Relative Performance under Corruption
clean_mAP = 0.75  # your Phase 2 result

all_corrupted_mAPs = []
for corruption in CORRUPTIONS:
    for severity in SEVERITIES:
        all_corrupted_mAPs.append(results_grid[corruption][severity])

mPC = np.mean(all_corrupted_mAPs)
relative_mPC = mPC / clean_mAP  # fraction of clean performance retained
print(f"Clean mAP: {clean_mAP:.3f}")
print(f"mPC: {mPC:.3f}")
print(f"Relative mPC: {relative_mPC:.3f} ({relative_mPC*100:.1f}% of clean retained)")
```

### 3.5 Plot the Robustness Curve

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: mAP vs Severity for each corruption type
ax = axes[0]
for corruption in CORRUPTIONS:
    mAPs = [results_grid[corruption][s] for s in SEVERITIES]
    ax.plot(SEVERITIES, mAPs, marker='o', label=corruption)
ax.axhline(y=clean_mAP, color='k', linestyle='--', label='clean baseline')
ax.set_xlabel('Corruption Severity')
ax.set_ylabel('mAP@0.5')
ax.set_title('Robustness Curve — Baseline Model')
ax.legend()

# Plot 2: Performance drop heatmap
drops = np.array([[clean_mAP - results_grid[c][s] for s in SEVERITIES] for c in CORRUPTIONS])
ax = axes[1]
im = ax.imshow(drops, cmap='Reds', aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels([f'S{i}' for i in SEVERITIES])
ax.set_yticks(range(len(CORRUPTIONS))); ax.set_yticklabels(CORRUPTIONS)
ax.set_title('mAP Drop Heatmap (redder = worse)')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('results/figures/robustness_curve_baseline.png', dpi=150)
```

### ✅ Phase 3 Completion Checklist

- [ ] All 5 corruption types implemented
- [ ] **VIZ 3.A saved** — corruption strip visually inspected: severities look realistic and progressive
- [ ] All 25 corrupted test sets generated (5 corruptions × 5 severities)
- [ ] Baseline model evaluated on all 25 sets
- [ ] results_grid saved as `results/robustness_baseline.json`
- [ ] Robustness curve plot saved
- [ ] mPC computed and recorded

**Problem Set 3 (Metrics):**

| Metric | What to record | Notes |
|---|---|---|
| mPC (baseline) | Record whatever it is | This is your Chapter 3 baseline — no target, just measure |
| Worst corruption type | Which causes biggest drop? | Guides Phase 4 augmentation |
| Worst severity | At severity 5, what is average mAP? | Expect large drop |
| Relative mPC | mPC / clean_mAP | Anything below 0.60 = model is brittle |

> **The goal of Phase 3 is pure measurement, not improvement.** You're answering: "how brittle is my model?" If relative mPC ≥ 0.85 (model retains 85% of clean performance), your model was already robust — check your corruption implementations. Most models fall to 0.50–0.70 relative mPC.

---

## Phase 4 — Robust Training (Corruption-Aware Augmentation)

**Goal:** Retrain the model with corruption-aware augmentation during training. Measure whether mPC improves.

### 4.1 Augmentation Pipeline

```python
import albumentations as A

def get_robust_augmentation_pipeline():
    return A.Compose([
        # Geometric augmentations (standard)
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomCrop(height=580, width=580, p=0.3),
        
        # Corruption-mimicking augmentations
        A.OneOf([
            A.GaussNoise(var_limit=(10, 80), p=1.0),
            A.MotionBlur(blur_limit=(3, 15), p=1.0),
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
        ], p=0.5),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.4,
            contrast_limit=0.3,
            p=0.4
        ),
        
        # Occlusion simulation
        A.CoarseDropout(
            max_holes=8,
            max_height=40,
            max_width=40,
            p=0.3
        ),
        
        # Resize to model input
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### 4.2 Retrain

Train the same RT-DETR-L architecture with this augmentation pipeline. Use the same pretrained COCO weights as starting point.

Keep all other hyperparameters identical to Phase 2. The only variable that changes is the augmentation pipeline.

```python
# Retrain with augmentation
model_robust = RTDETR('rtdetr-l.pt')
results = model_robust.train(
    data='robot_parts.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=1e-4,
    augment=True,     # enable built-in augmentations
    # + pass your custom albumentations pipeline via a custom dataset class
)
```

### 4.3 Evaluate Both Models Side-by-Side

Run the full Phase 3 evaluation script again on the robust model. Compare:

```python
# Load results from Phase 3 (baseline) and new evaluation (robust)
# Produce a side-by-side robustness curve
fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(20, 4))
for i, corruption in enumerate(CORRUPTIONS):
    baseline_mAPs = [baseline_results[corruption][s] for s in SEVERITIES]
    robust_mAPs = [robust_results[corruption][s] for s in SEVERITIES]
    axes[i].plot(SEVERITIES, baseline_mAPs, 'r-o', label='baseline')
    axes[i].plot(SEVERITIES, robust_mAPs, 'g-o', label='robust')
    axes[i].axhline(y=clean_mAP, color='k', linestyle='--')
    axes[i].set_title(corruption)
    axes[i].legend()
plt.savefig('results/figures/robustness_comparison.png', dpi=150)
```

### ✅ Phase 4 Completion Checklist

- [ ] Robust model trained with augmentation pipeline
- [ ] Robust model evaluated on clean test set — verify clean mAP doesn't drop significantly (< 3% drop is acceptable)
- [ ] Robust model evaluated on all 25 corrupted test sets
- [ ] Side-by-side comparison plot saved
- [ ] robust_results saved as `results/robustness_robust.json`

**Problem Set 4 (Metrics):**

| Metric | Target | Notes |
|---|---|---|
| mPC (robust model) | ≥ mPC_baseline × 1.10 | At least 10% relative improvement |
| Clean mAP (robust) | ≥ clean_mAP − 0.03 | Don't sacrifice clean performance |
| Biggest improvement | Which corruption improved most? | Should match what you augmented |
| Relative mPC (robust) | Record | Ideally ≥ 0.75 |

> **The 10% mPC improvement target is the headline result of this project.** If you hit it, the narrative is clean: "corruption-aware augmentation improved robustness by X% without sacrificing clean performance." If you don't hit it, diagnose: are your training augmentations actually matching your test corruptions at similar severity?

---

## Phase 5 — MC Dropout Uncertainty Head

**Goal:** Add MC Dropout to your trained model. At inference, run N forward passes and compute the variance — your uncertainty estimate. Validate that high uncertainty correlates with actual errors.

### 5.1 Enable Dropout at Inference

RT-DETR's backbone likely already has dropout layers. The modification needed:

```python
def enable_mc_dropout(model):
    """Set all dropout layers to training mode (keeps them active at inference)"""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    return model

def mc_dropout_inference(model, image, n_passes=30):
    """
    Run N stochastic forward passes, return mean prediction and uncertainty.
    """
    model.eval()
    enable_mc_dropout(model)  # dropout ON, batchnorm OFF
    
    all_predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred = model(image)
            all_predictions.append(pred)
    
    # Aggregate predictions
    # For detection: match boxes across passes (by IoU), compute per-box confidence variance
    mean_confidences = aggregate_confidences(all_predictions)
    uncertainty = compute_variance(all_predictions)
    
    return mean_confidences, uncertainty
```

### 5.2 Uncertainty Aggregation for Detection

Object detection aggregation is trickier than classification — you need to match the same object across N passes.

```python
def aggregate_detection_passes(all_predictions, iou_threshold=0.5):
    """
    For each detected box in pass 1 (reference), find the matching box in all other passes.
    Compute the variance of the confidence scores across passes for that box.
    
    Returns: list of (box, mean_confidence, uncertainty_score)
    """
    # Use reference pass as anchor
    reference_boxes = all_predictions[0]  # boxes from first pass
    
    results = []
    for ref_box in reference_boxes:
        confidences_across_passes = []
        for pred in all_predictions:
            # Find best-matching box in this pass
            best_match = find_best_iou_match(ref_box, pred)
            if best_match is not None:
                confidences_across_passes.append(best_match['confidence'])
            else:
                confidences_across_passes.append(0.0)  # object not detected in this pass
        
        mean_conf = np.mean(confidences_across_passes)
        uncertainty = np.std(confidences_across_passes)  # std dev = uncertainty
        results.append({
            'box': ref_box,
            'mean_confidence': mean_conf,
            'uncertainty': uncertainty
        })
    
    return results
```

### 5.3 Validation — Does Uncertainty Correlate with Errors?

This is your most important sanity check. A good uncertainty estimator should be HIGH when the model is wrong and LOW when it's correct.

```python
# Run MC Dropout on your test set (clean)
# For each detection, record: (is_correct, uncertainty_score)
# A detection is "correct" if IoU with ground truth ≥ 0.5

# Then compute:
# 1. ECE (Expected Calibration Error) — lower is better
# 2. Correlation between uncertainty and error

# Split predictions into N buckets by confidence level
# Within each bucket: does the model's mean confidence match actual accuracy?

def compute_ECE(predictions, n_buckets=10):
    bucket_size = 1.0 / n_buckets
    ece = 0.0
    for i in range(n_buckets):
        low = i * bucket_size
        high = (i+1) * bucket_size
        bucket = [p for p in predictions if low <= p['mean_confidence'] < high]
        if len(bucket) == 0:
            continue
        avg_confidence = np.mean([p['mean_confidence'] for p in bucket])
        avg_accuracy = np.mean([p['is_correct'] for p in bucket])
        ece += (len(bucket) / len(predictions)) * abs(avg_confidence - avg_accuracy)
    return ece
```

#### VIZ 5.A — Reliability diagram (calibration plot)

```python
def plot_reliability_diagram(predictions, n_buckets=10):
    """
    predictions: list of dicts with keys 'mean_confidence' and 'is_correct'
    A well-calibrated model follows the diagonal — confidence matches actual accuracy.
    """
    bucket_size = 1.0 / n_buckets
    bucket_confidences = []
    bucket_accuracies = []
    bucket_sizes = []

    for i in range(n_buckets):
        low = i * bucket_size
        high = (i + 1) * bucket_size
        bucket = [p for p in predictions if low <= p['mean_confidence'] < high]
        if len(bucket) == 0:
            continue
        bucket_confidences.append(np.mean([p['mean_confidence'] for p in bucket]))
        bucket_accuracies.append(np.mean([p['is_correct'] for p in bucket]))
        bucket_sizes.append(len(bucket))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.bar(bucket_confidences, bucket_accuracies, width=bucket_size*0.8,
           alpha=0.6, color='steelblue', label='Model')
    ax.set_xlabel('Mean predicted confidence')
    ax.set_ylabel('Actual accuracy')
    ax.set_title('VIZ 5.A — Reliability diagram\n(bars should follow the diagonal)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend()

    # Right: confidence histogram (shows where most predictions fall)
    ax2 = axes[1]
    all_confs = [p['mean_confidence'] for p in predictions]
    ax2.hist(all_confs, bins=20, color='darkorange', edgecolor='white')
    ax2.set_xlabel('Predicted confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence distribution\n(should not be all bunched at 0.9+)')

    plt.tight_layout()
    plt.savefig('results/figures/viz5a_reliability_diagram.png', dpi=150)
    plt.show()
    # ⚠️  MANUAL CHECK: If bars are consistently above the diagonal → overconfident
    #    (model says 0.9 but is right only 70% of the time). Apply temperature scaling.
    #    If bars are below → underconfident. Either is bad; diagonal = good.

plot_reliability_diagram(test_predictions)
```

#### VIZ 5.B — Uncertainty overlaid on detections (qualitative check)

```python
def visualize_uncertainty_on_image(img_path, detections, class_names):
    """
    Draw bounding boxes colored by uncertainty:
    GREEN = low uncertainty (trust this detection)
    ORANGE = medium uncertainty
    RED = high uncertainty (treat with caution)
    Also prints the uncertainty score on each box.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    h, w = img.shape[:2]

    # Normalize uncertainty scores to [0,1] for coloring
    uncertainties = [d['uncertainty'] for d in detections]
    if max(uncertainties) > 0:
        u_norm = [(u - min(uncertainties)) / (max(uncertainties) - min(uncertainties) + 1e-8)
                  for u in uncertainties]
    else:
        u_norm = [0.0] * len(uncertainties)

    for det, u in zip(detections, u_norm):
        box = det['box']  # [x1, y1, x2, y2] in pixel coords
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Color: green → orange → red as uncertainty increases
        r = int(255 * u)
        g = int(255 * (1 - u))
        color = (r, g, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label = f"{class_names[det['class']]} | u={det['uncertainty']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# Run on 6 test images — 3 clean, 3 heavily corrupted (severity 5)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, (img_path, is_corrupted) in enumerate(selected_pairs):
    detections = mc_dropout_inference(model, img_path, n_passes=30)
    vis = visualize_uncertainty_on_image(img_path, detections, CLASS_NAMES)
    row = 0 if not is_corrupted else 1
    col = i % 3
    axes[row, col].imshow(vis)
    axes[row, col].set_title('Clean' if not is_corrupted else 'Corrupted (severity 5)', fontsize=10)
    axes[row, col].axis('off')

axes[0, 0].set_ylabel('Low corruption', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('High corruption', fontsize=12, fontweight='bold')
plt.suptitle('VIZ 5.B — Uncertainty overlay: GREEN=confident, RED=uncertain', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/viz5b_uncertainty_overlay.png', dpi=120)
plt.show()
# ⚠️  MANUAL CHECK: Corrupted images (row 2) should have more red boxes than clean (row 1).
#    If both rows look equally green, MC Dropout is not sensitive to corruption → diagnose.
#    If both rows are mostly red, the dropout rate may be too high.
```

```python
# Per-severity uncertainty plot
for corruption in CORRUPTIONS:
    uncertainties_per_severity = []
    for severity in SEVERITIES:
        # Run MC Dropout on this corrupted set
        preds = mc_dropout_on_set(model, corrupted_sets[corruption][severity])
        mean_uncertainty = np.mean([p['uncertainty'] for p in preds])
        uncertainties_per_severity.append(mean_uncertainty)
    plt.plot(SEVERITIES, uncertainties_per_severity, label=corruption)
plt.xlabel('Corruption Severity')
plt.ylabel('Mean Uncertainty')
plt.title('Uncertainty Increases with Corruption Severity')
plt.legend()
plt.savefig('results/figures/uncertainty_vs_severity.png', dpi=150)
```

### ✅ Phase 5 Completion Checklist

- [ ] MC Dropout enabled correctly — verify dropout is ON at inference (check output is different across two runs of same image)
- [ ] N=30 passes implemented and timing measured
- [ ] Uncertainty aggregation working for bounding box predictions
- [ ] ECE computed on clean test set
- [ ] **VIZ 5.A saved** — reliability diagram: bars roughly following the diagonal
- [ ] **VIZ 5.B saved** — uncertainty overlay on 6 images: corrupted row has visibly more red than clean row
- [ ] Uncertainty vs severity plot saved — shows monotonically increasing uncertainty with severity for at least 3/5 corruptions

**Problem Set 5 (Metrics):**

| Metric | Target | Notes |
|---|---|---|
| ECE on clean test set | ≤ 0.10 | Expected Calibration Error. 0 = perfectly calibrated |
| Uncertainty correlation with error | Positive (Spearman ρ > 0.3) | High uncertainty should predict errors |
| Uncertainty at severity 5 vs severity 1 | Severity 5 uncertainty ≥ 1.5× severity 1 | Model should express more doubt on harder inputs |
| Latency per image (N=30 passes) | Record it | Will be slow — document the tradeoff |

> **The ECE ≤ 0.10 target is the hardest one.** If ECE is high (>0.15), your model's confidence scores are poorly calibrated. Consider post-hoc temperature scaling: `calibrated_logit = raw_logit / T` where T is a scalar fit on a held-out set.

---

## Phase 6 — Conformal Prediction (Stretch Goal)

**Goal:** Apply conformal prediction to guarantee that your prediction sets contain the true label ≥ 95% of the time, without any distributional assumptions.

> **This is a stretch goal.** Complete Phases 0–5 first. If you complete Phase 5 successfully, Phase 6 is what separates this project from "good" to "exceptional."

### 6.1 Setup

You need your calibration set (100 images carved out in Phase 1).

For each calibration image:
1. Run standard inference (no MC Dropout needed here)
2. Get the softmax probabilities for each detected class
3. Compute the nonconformity score: `s_i = 1 - p(true_class | image_i)`

This score is high when the model is wrong or uncertain (low probability assigned to the true class).

```python
calibration_scores = []
for img, label in calibration_set:
    pred = model(img)
    true_class_prob = pred['class_probs'][label['true_class']]
    nonconformity_score = 1 - true_class_prob
    calibration_scores.append(nonconformity_score)
```

### 6.2 Compute the Threshold

```python
# User-chosen error rate
epsilon = 0.05  # 5% — we want 95% coverage

# Find the (1-epsilon) quantile of calibration nonconformity scores
n = len(calibration_scores)
q_level = np.ceil((n+1) * (1-epsilon)) / n
threshold = np.quantile(calibration_scores, q_level)
print(f"Threshold: {threshold:.4f}")
```

### 6.3 Generate Prediction Sets at Test Time

```python
def conformal_prediction_set(model, image, threshold):
    """
    Returns a SET of classes (not just one), guaranteed to contain the true class
    with probability ≥ 1 - epsilon.
    """
    pred = model(image)
    class_probs = pred['class_probs']  # probabilities for each class
    
    prediction_set = []
    for class_idx, prob in enumerate(class_probs):
        nonconformity = 1 - prob
        if nonconformity <= threshold:  # this class is "conforming"
            prediction_set.append(class_idx)
    
    return prediction_set
```

On easy images: set size = 1 (just "leg")
On hard/corrupted images: set size = 3 ({"leg", "arm", "torso"}) — the model is saying "it's one of these"

### 6.4 Validate Coverage

Run on test set and verify the guarantee holds:

```python
correct_coverage = 0
set_sizes = []

for img, label in test_set:
    pred_set = conformal_prediction_set(model, img, threshold)
    if label['true_class'] in pred_set:
        correct_coverage += 1
    set_sizes.append(len(pred_set))

coverage = correct_coverage / len(test_set)
avg_set_size = np.mean(set_sizes)
print(f"Empirical coverage: {coverage:.3f} (target: {1-epsilon:.3f})")
print(f"Average prediction set size: {avg_set_size:.2f}")
```

Also run on corrupted test sets — do prediction sets get larger as corruption increases?

#### VIZ 6.A — Prediction set size distribution (clean vs corrupted)

```python
def get_set_sizes(model, dataset, threshold, class_names):
    set_sizes = []
    prediction_sets = []
    for img, label in dataset:
        pred_set = conformal_prediction_set(model, img, threshold)
        set_sizes.append(len(pred_set))
        prediction_sets.append({'set': pred_set, 'true': label['true_class']})
    return set_sizes, prediction_sets

# Collect set sizes across clean + each severity level
all_set_sizes = {'clean': get_set_sizes(model, test_set, threshold)[0]}
for severity in SEVERITIES:
    corrupted_ds = load_corrupted_set('gaussian_noise', severity)  # use worst corruption
    all_set_sizes[f'S{severity}'] = get_set_sizes(model, corrupted_ds, threshold)[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: set size histograms stacked
ax = axes[0]
labels = list(all_set_sizes.keys())
colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(labels)))
for label, color in zip(labels, colors):
    sizes = all_set_sizes[label]
    ax.hist(sizes, bins=range(1, len(CLASS_NAMES)+2), alpha=0.6,
            label=label, color=color, edgecolor='white')
ax.set_xlabel('Prediction set size')
ax.set_ylabel('Count')
ax.set_title('VIZ 6.A — Prediction set sizes\n(larger sets = more uncertain)')
ax.legend()

# Right: mean set size vs severity (key result)
ax = axes[1]
mean_sizes = [np.mean(all_set_sizes['clean'])] + \
             [np.mean(all_set_sizes[f'S{s}']) for s in SEVERITIES]
x_labels = ['clean'] + [f'S{s}' for s in SEVERITIES]
ax.plot(x_labels, mean_sizes, 'o-', color='steelblue', linewidth=2, markersize=8)
ax.axhline(y=1.0, color='green', linestyle='--', label='ideal (set size=1)')
ax.set_xlabel('Corruption severity')
ax.set_ylabel('Mean prediction set size')
ax.set_title('Mean set size grows with corruption\n(model becomes less certain)')
ax.legend()

plt.suptitle('VIZ 6.A — Conformal prediction: set sizes under corruption', fontsize=13)
plt.tight_layout()
plt.savefig('results/figures/viz6a_conformal_set_sizes.png', dpi=150)
plt.show()
# ⚠️  MANUAL CHECK: Clean images should have mostly set size=1.
#    Set sizes should grow as severity increases (right plot trending upward).
#    If set sizes are always 1 even at severity 5 → threshold may be too loose.
#    If set sizes are always 5 (all classes) → threshold may be too tight.
```

#### VIZ 6.B — Example prediction sets on real images

```python
# Show 8 test images with their conformal prediction set displayed
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
sample_indices = random.sample(range(len(test_set)), 8)

for ax, idx in zip(axes.flatten(), sample_indices):
    img, label = test_set[idx]
    pred_set = conformal_prediction_set(model, img, threshold)
    true_class = label['true_class']

    ax.imshow(img)
    set_str = '{' + ', '.join([CLASS_NAMES[c] for c in pred_set]) + '}'
    true_str = CLASS_NAMES[true_class]
    covered = true_class in pred_set

    title = f"True: {true_str}\nPred set: {set_str}"
    color = 'green' if covered else 'red'
    ax.set_title(title, fontsize=9, color=color)
    ax.axis('off')

plt.suptitle('VIZ 6.B — Conformal prediction sets on examples\n'
             '(green title = true class in set, red = not covered)', fontsize=12)
plt.tight_layout()
plt.savefig('results/figures/viz6b_conformal_examples.png', dpi=120)
plt.show()
# ⚠️  MANUAL CHECK: Red titles (not covered) should be rare — target < 5% of images.
#    On easy clean images, the set should typically contain just one class.
#    On hard/occluded images, sets of 2-3 are expected and acceptable.
```

### ✅ Phase 6 Completion Checklist

- [ ] Calibration scores computed for all 100 calibration images
- [ ] Threshold computed at ε = 0.05
- [ ] Conformal prediction sets generated for all test images
- [ ] Empirical coverage validated on test set
- [ ] **VIZ 6.A saved** — set size histogram and mean set size vs severity trend (upward slope visible)
- [ ] **VIZ 6.B saved** — example images with prediction sets, red titles < 5%

**Problem Set 6 (Metrics):**

| Metric | Target | Notes |
|---|---|---|
| Empirical coverage (clean test) | ≥ 0.95 (matches 1-ε guarantee) | If < 0.95, something is wrong in implementation |
| Avg set size (clean images) | ≤ 1.5 | Model should be confident on clean data |
| Avg set size (severity 5) | > 2.0 | Model should produce larger sets on hard inputs |
| Coverage under corruption | ≥ 0.90 even at severity 5 | The guarantee should be robust |

---

## Phase 7 — Analysis, Write-up & Resume Bullet

**Goal:** Package the work into a portfolio-ready artifact.

### 7.1 Final Results Table

Produce this table in your README:

| Model | Clean mAP | mPC | Relative mPC | ECE |
|---|---|---|---|---|
| RT-DETR baseline | X.XX | X.XX | X.XX | — |
| RT-DETR + robust aug | X.XX | X.XX | X.XX | — |
| RT-DETR + MC Dropout | X.XX | X.XX | X.XX | X.XX |

### 7.2 Three Key Figures (must have these)

1. **Robustness curve** — mAP vs severity for baseline vs robust model, for all 5 corruptions
2. **Reliability diagram** — model confidence vs actual accuracy (calibration plot)
3. **Uncertainty vs severity** — mean uncertainty score as corruption severity increases

### 7.3 README Structure

```markdown
# Robust + Uncertainty-Aware Robot Perception

## What this does
One paragraph. Clear problem statement.

## Results
| Model | Clean mAP | mPC | ECE |
[your table]

## Key finding
"Corruption-aware augmentation improved mean performance under corruption by X% 
(mPC: 0.XX → 0.XX) without sacrificing clean accuracy. 
MC Dropout provides calibrated uncertainty (ECE: 0.XX) that increases 
monotonically with corruption severity."

## Methods
- Dataset: N images, 5 classes, humanoid robot components
- Architecture: RT-DETR-L fine-tuned from COCO pretrained
- Corruption benchmark: ImageNet-C methodology, 5 corruption types × 5 severities
- Uncertainty: MC Dropout, N=30 passes
- [Optional] Conformal prediction: ε=0.05, empirical coverage: 0.XX

## Setup & Usage
[install + run instructions]

## Figures
[embed your 3 key figures]
```

### 7.4 Resume Bullet (draft)

Once you have numbers, fill this template:

```
Robust Humanoid Robot Perception — RT-DETR · MC Dropout · Conformal Prediction · PyTorch
• Built an uncertainty-aware RT-DETR detector for humanoid robot components; 
  benchmarked robustness across 25 corruption scenarios (5 types × 5 severities, 
  ImageNet-C methodology), achieving mAP@0.5: X.XX on clean images
• Designed corruption-aware augmentation pipeline that improved mean performance 
  under corruption (mPC) by X% relative to baseline without sacrificing clean accuracy
• Implemented MC Dropout uncertainty quantification (N=30 stochastic forward passes);
  achieved ECE ≤ X.XX with uncertainty that increases monotonically with 
  corruption severity — enabling principled confidence thresholding for robot deployment
```

### ✅ Phase 7 Completion Checklist

- [ ] Final results table complete with real numbers
- [ ] Three key figures saved at high resolution (300 DPI)
- [ ] README written and pushed
- [ ] Resume bullet drafted with actual numbers filled in
- [ ] GitHub repo is public, clean, and has a working `requirements.txt`

---

## Diagnostic Guide — When Things Go Wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| mAP < 0.50 after 100 epochs | Dataset quality issue or class imbalance | Audit annotations, check class distribution |
| mPC barely changes after robust training | Augmentation severity doesn't match test corruption severity | Increase augmentation strength |
| ECE > 0.20 | Model overconfident — softmax scores not calibrated | Apply temperature scaling post-hoc |
| Uncertainty doesn't increase with severity | MC Dropout not actually active at inference | Verify `module.train()` on all Dropout layers |
| Coverage < 0.95 in conformal prediction | Off-by-one in quantile computation | Use `np.ceil((n+1)*(1-eps))/n` exactly |
| Inference too slow (N=30 passes) | Normal — document it | Report single-pass speed separately, note the tradeoff |

---

## Time Estimate

| Phase | Estimated time | GPU time (RTX 3090) | RunPod cost |
|---|---|---|---|
| 0 — Setup | 1–2 hrs | — | — |
| 1 — Dataset | done | — | — |
| 2 — Baseline | 1 hr setup + train | ~45–60 min | ~$0.50 |
| 3 — Corruption benchmark | 1 hr setup + eval | ~20–30 min | ~$0.25 |
| 4 — Robust training | 30 min setup + train | ~45–60 min | ~$0.50 |
| 5 — MC Dropout | 2–3 hrs | ~30 min (N=30 passes on test set) | ~$0.25 |
| 6 — Conformal (stretch) | 2 hrs | ~10 min | ~$0.10 |
| 7 — Write-up | 2–3 hrs | — | — |

**Total GPU cost estimate: ~$2–3 total on RunPod**
**Rule: stop the pod the moment training/eval finishes. Never leave it idle.**

---

## Working with Claude Code

When starting a new Claude Code session on any phase, paste this at the top:

```
Project: Robust + Uncertainty-Aware Robot Perception
Current phase: [PHASE NUMBER] — [PHASE NAME]
What I'm trying to do: [specific task]
Current error / question: [paste it]
```

For evaluation runs, always paste:
- Your current metric numbers
- The metric targets from the Problem Set
- What you're trying to debug or improve

---

## RunPod SSH Quick Reference

```bash
# Connect
ssh root@<pod-ip> -p <port>

# Sync dataset TO pod (from Mac)
rsync -avz data/annotated/ root@<pod-ip>:<port>:/workspace/data/annotated/

# Sync scripts TO pod
rsync -avz scripts/ root@<pod-ip>:<port>:/workspace/scripts/

# Sync model weights TO pod (if restarting pod mid-project)
rsync -avz models/ root@<pod-ip>:<port>:/workspace/models/

# Sync results FROM pod (always do before stopping pod)
rsync -avz root@<pod-ip>:<port>:/workspace/results/ ./results/
rsync -avz root@<pod-ip>:<port>:/workspace/models/ ./models/

# Check GPU
nvidia-smi

# Run training in background (survives SSH disconnect)
nohup python /workspace/scripts/train_baseline.py > /workspace/train.log 2>&1 &
tail -f /workspace/train.log
```

**Phase → script mapping:**

| Phase | Script to run on pod |
|---|---|
| 2 — Baseline training | `python scripts/train_baseline.py` |
| 3 — Corruption benchmark | `python scripts/evaluate_robustness.py --model models/baseline/run1/weights/best.pt` |
| 4 — Robust training | `python scripts/train_robust.py` |
| 4 — Robust benchmark | `python scripts/evaluate_robustness.py --model models/robust/run1/weights/best.pt --out results/robustness_robust.json` |
| 5 — MC Dropout | `python scripts/mc_dropout_inference.py --model models/robust/run1/weights/best.pt` |
| 6 — Conformal | `python scripts/conformal.py --model models/robust/run1/weights/best.pt` |

---

*Last updated: May 2026*
*Project owner: Vineet Jangir, IIT Bombay*
