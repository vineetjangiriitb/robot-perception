"""
Phase 1 — Image collection script.

Downloads robot images from Roboflow Universe datasets and organizes them
into data/raw/ for manual review before annotation.

Usage:
    python scripts/download_images.py

After running:
1. Review images in data/raw/  — delete blurry/duplicate/irrelevant ones
2. Upload the keepers to your Roboflow project for annotation
3. Annotate with classes: 0=arm, 1=leg, 2=torso, 3=head, 4=sensor
"""

import os
import sys
import urllib.request
import json
import time
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Option A: Download via Roboflow Python SDK (requires API key)
# ---------------------------------------------------------------------------

def download_via_roboflow(api_key: str, workspace: str, project: str, version: int = 1):
    """
    Download a Roboflow Universe dataset (images only, we re-annotate ourselves).

    Install: pip install roboflow
    Get your API key from: https://app.roboflow.com/ → Settings → API
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8", location=str(RAW_DIR / project))
    print(f"Downloaded to {RAW_DIR / project}")
    return dataset


# ---------------------------------------------------------------------------
# Option B: Download images from direct URLs (no API key needed)
# ---------------------------------------------------------------------------

# Curated list of royalty-free / open-licensed robot image URLs.
# These are from public datasets and open web sources.
# Add more URLs here as you find good images.
ROBOT_IMAGE_URLS = [
    # Boston Dynamics Spot — publicly released press images
    # Add direct image URLs here, e.g.:
    # "https://example.com/robot1.jpg",
]


def download_from_urls(urls: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for i, url in enumerate(urls):
        ext = url.split(".")[-1].split("?")[0]
        if ext not in ("jpg", "jpeg", "png", "webp"):
            ext = "jpg"
        dest = output_dir / f"robot_{i:04d}.{ext}"
        if dest.exists():
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            downloaded += 1
            print(f"  [{i+1}/{len(urls)}] {dest.name}")
            time.sleep(0.3)  # be polite
        except Exception as e:
            print(f"  FAILED {url}: {e}")
    print(f"\nDownloaded {downloaded} images to {output_dir}")


# ---------------------------------------------------------------------------
# Option C: Extract frames from a local video file
# ---------------------------------------------------------------------------

def extract_video_frames(video_path: str, output_dir: Path, every_n_frames: int = 30):
    """
    Extract one frame every N frames from a video.
    Useful for YouTube downloads (use yt-dlp to download first).

    Install yt-dlp: pip install yt-dlp
    Download video: yt-dlp -o data/raw/video.mp4 <youtube-url>
    """
    try:
        import cv2
    except ImportError:
        print("Run: pip install opencv-python")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            dest = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(dest), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames from {video_path} → {output_dir}")


# ---------------------------------------------------------------------------
# Audit: count and report what's in data/raw/
# ---------------------------------------------------------------------------

def audit_raw():
    total = 0
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        total += len(list(RAW_DIR.rglob(ext)))
    print(f"\n--- data/raw/ audit ---")
    print(f"Total images found: {total}")
    if total < 300:
        print(f"Need {300 - total} more images to hit the ≥300 target.")
    else:
        print("Target met. Ready to upload to Roboflow for annotation.")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 1 Image Collection ===")
    print()
    print("Choose a collection method:")
    print("  A) Roboflow SDK  — needs API key, downloads existing datasets")
    print("  B) Direct URLs   — add URLs to ROBOT_IMAGE_URLS list in this file")
    print("  C) Video frames  — extract from a downloaded YouTube video")
    print()
    print("RECOMMENDED WORKFLOW:")
    print("1. Go to https://universe.roboflow.com/search?q=robot")
    print("2. Find datasets with robot images (even if classes differ)")
    print("3. Get your Roboflow API key from https://app.roboflow.com/settings/api")
    print("4. Call download_via_roboflow(api_key, workspace, project) from this script")
    print()
    print("Then upload the raw images to YOUR Roboflow project and annotate with:")
    print("  0=arm  1=leg  2=torso  3=head  4=sensor")
    print()

    # Example usage (uncomment and fill in your API key + dataset slugs):
    # API_KEY = "your_api_key_here"
    # download_via_roboflow(API_KEY, "lab-4ifnn", "humanoid-robots-and-people-a2inv", version=1)

    # For video frame extraction (after downloading a video with yt-dlp):
    # extract_video_frames("data/raw/boston_dynamics.mp4", RAW_DIR / "bd_frames", every_n_frames=30)

    audit_raw()
