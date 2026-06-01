#!/usr/bin/env python3
"""
Precompute real MediaPipe hand landmarks for training samples.

Contour fallback is disabled. Samples without real hand landmarks are reported.
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


def parse_sources(value):
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser(description="Cache real hand landmarks for dataset samples")
    parser.add_argument("--root-dir", default="data/videos")
    parser.add_argument("--image-root", default="data/images")
    parser.add_argument("--cache-dir", default="data/landmark_cache")
    parser.add_argument("--sources", default="words,letters,sentences,images")
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default="data/landmark_cache/landmark_report.json")
    args = parser.parse_args()

    task_model = HandLandmarkExtractor._resolve_task_model(None)
    if task_model is None:
        raise SystemExit(
            "No real MediaPipe hand landmark model found. Place hand_landmarker.task in "
            "models/ or assets/, or set HAND_LANDMARKER_TASK to the model file path."
        )

    dataset = MultiSourceSignLanguageDataset(
        root_dir=args.root_dir,
        image_root=args.image_root,
        num_frames=args.num_frames,
        image_size=args.image_size,
        include_sources=parse_sources(args.sources),
        use_landmarks=True,
        landmark_cache_dir=args.cache_dir,
        drop_missing_image_landmarks=False,
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
            sample = dataset.samples[idx]
            missing.append(
                {
                    "index": idx,
                    "text": item["text"],
                    "source": item["source"],
                    "path": sample["path"],
                }
            )

        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"Processed {idx + 1}/{total}, detected={detected}, missing={len(missing)}")

    report = {
        "task_model": str(task_model),
        "sources": list(parse_sources(args.sources)),
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
        print("Some samples did not produce hand landmarks. Review the report.")


if __name__ == "__main__":
    main()
