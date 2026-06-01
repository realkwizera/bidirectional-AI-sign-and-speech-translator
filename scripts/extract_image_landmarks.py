#!/usr/bin/env python3
"""
Extract and cache real hand landmarks for data/images.

This script intentionally does not use contour fallback. It requires a real
MediaPipe HandLandmarker task model so the cached features represent wrist,
palm, finger, and finger-joint landmarks.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.real_dataset import MultiSourceSignLanguageDataset
from utils.hand_landmarks import HandLandmarkExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract real hand landmarks for data/images")
    parser.add_argument("--image-root", default="data/images", help="Root folder of letter image classes")
    parser.add_argument("--cache-dir", default="data/landmark_cache", help="Landmark cache directory")
    parser.add_argument("--num-frames", type=int, default=30, help="Repeated sequence length for static images")
    parser.add_argument("--image-size", type=int, default=64, help="Image size used by the CNN path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of image samples")
    parser.add_argument("--report", default="data/landmark_cache/image_landmark_report.json")
    args = parser.parse_args()

    task_model = HandLandmarkExtractor._resolve_task_model(None)
    if task_model is None:
        raise SystemExit(
            "No real MediaPipe hand landmark model found. Place hand_landmarker.task in "
            "models/ or assets/, or set HAND_LANDMARKER_TASK to the model file path. "
            "Contour fallback is disabled."
        )

    dataset = MultiSourceSignLanguageDataset(
        image_root=args.image_root,
        num_frames=args.num_frames,
        image_size=args.image_size,
        include_sources=("images",),
        use_landmarks=True,
        landmark_cache_dir=args.cache_dir,
    )

    total = len(dataset) if args.limit is None else min(len(dataset), args.limit)
    detected = 0
    missing = []

    for idx in range(total):
        item = dataset[idx]
        presence = float(item["landmarks"][:, -1].max().item())
        if presence > 0.0:
            detected += 1
        else:
            missing.append(
                {
                    "index": idx,
                    "text": item["text"],
                    "source": item["source"],
                    "path": dataset.samples[idx]["path"],
                }
            )

        if (idx + 1) % 100 == 0 or idx + 1 == total:
            print(f"Processed {idx + 1}/{total} images, detected={detected}, missing={len(missing)}")

    report = {
        "task_model": str(task_model),
        "total": total,
        "detected": detected,
        "missing": len(missing),
        "missing_samples": missing,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Report written to {report_path}")
    if missing:
        print("Some images did not produce hand landmarks. Review the report before training.")


if __name__ == "__main__":
    main()
