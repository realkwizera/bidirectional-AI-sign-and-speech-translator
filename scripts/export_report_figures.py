from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.hand_landmarks import HandLandmarkExtractor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_sample_image(images_root: Path, preferred_letter: str | None = None) -> Path:
    folders = []
    if preferred_letter:
        folders = [images_root / preferred_letter, images_root / preferred_letter.upper(), images_root / preferred_letter.lower()]
    elif images_root.exists():
        folders = sorted([p for p in images_root.iterdir() if p.is_dir()])

    for folder in folders:
        if not folder.exists() or not folder.is_dir():
            continue
        for image_path in sorted(folder.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTS:
                return image_path
    raise FileNotFoundError(f"No image files found under {images_root}")


def draw_landmarks(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    pts = []
    for x, y, _z in landmarks:
        pts.append((int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))))

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]
    for a, b in connections:
        cv2.line(image, pts[a], pts[b], (0, 204, 255), 2, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        color = (29, 75, 255) if idx in {4, 8, 12, 16, 20} else (0, 204, 223)
        cv2.circle(image, pt, 4, color, -1, cv2.LINE_AA)
    return image


def fit_height(image_bgr: np.ndarray, height: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    scale = height / max(1, h)
    return cv2.resize(image_bgr, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)


def make_side_by_side(left_bgr: np.ndarray, right_bgr: np.ndarray, left_title: str, right_title: str) -> np.ndarray:
    target_h = 430
    left = fit_height(left_bgr, target_h)
    right = fit_height(right_bgr, target_h)
    pad = 28
    title_h = 58
    width = left.shape[1] + right.shape[1] + pad * 3
    canvas = np.full((target_h + title_h + pad, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, left_title, (pad, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(canvas, right_title, (pad * 2 + left.shape[1], 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (15, 15, 15), 2, cv2.LINE_AA)
    canvas[title_h:title_h + left.shape[0], pad:pad + left.shape[1]] = left
    x2 = pad * 2 + left.shape[1]
    canvas[title_h:title_h + right.shape[0], x2:x2 + right.shape[1]] = right
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Export report figures 4.2 and 4.3 from a dataset image.")
    parser.add_argument("--image", type=Path, default=None, help="Specific source image to use.")
    parser.add_argument("--letter", default=None, help="Preferred data/images letter folder, for example A or b.")
    parser.add_argument("--images-root", type=Path, default=PROJECT_ROOT / "data" / "images")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "report_figures")
    parser.add_argument("--padding", type=float, default=0.35, help="Crop padding around the landmark bounding box.")
    args = parser.parse_args()

    source = args.image if args.image else find_sample_image(args.images_root, args.letter)
    if not source.exists():
        raise FileNotFoundError(source)

    bgr = cv2.imread(str(source))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {source}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    extractor = HandLandmarkExtractor(static_image_mode=True, allow_contour_fallback=False)
    try:
        result = extractor.extract(rgb)
    finally:
        extractor.close()

    if result.landmarks is None or result.presence <= 0:
        raise RuntimeError(
            "No real MediaPipe hand landmarks were detected in this image. Try another image, "
            "for example: py scripts/export_report_figures.py --letter b"
        )

    landmark_bgr = draw_landmarks(rgb, result.landmarks)
    crop_rgb, _crop_coords, bounds = HandLandmarkExtractor.crop_to_landmarks(rgb, result.landmarks, padding=args.padding)
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig42_path = args.out_dir / "figure_4_2_landmark_before_after.png"
    fig43_path = args.out_dir / "figure_4_3_tight_hand_crop.png"
    crop_only_path = args.out_dir / "figure_4_3_crop_only.png"

    cv2.imwrite(str(fig42_path), make_side_by_side(bgr, landmark_bgr, "Original training image", "After hand landmark detection"))
    cv2.imwrite(str(fig43_path), make_side_by_side(bgr, crop_bgr, "Original training image", "Tight hand crop"))
    cv2.imwrite(str(crop_only_path), crop_bgr)

    print(f"Source image: {source}")
    print(f"Hand presence: {result.presence:.3f}")
    print(f"Crop bounds: {bounds}")
    print(f"Saved Figure 4.2: {fig42_path}")
    print(f"Saved Figure 4.3: {fig43_path}")
    print(f"Saved crop only: {crop_only_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
