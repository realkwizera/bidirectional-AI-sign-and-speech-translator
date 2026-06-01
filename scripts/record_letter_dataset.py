"""
Record live webcam letter clips for data/videos/letters.

The recorder saves the same unmirrored hand ROI that live prediction uses by
default. The display can still be mirrored so recording feels natural.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_roi(value):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)


def roi_bounds(frame, roi):
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    if max(roi) <= 1.0:
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + rw) * w)
        y2 = int((y + rh) * h)
    else:
        x1 = int(x)
        y1 = int(y)
        x2 = int(x + rw)
        y2 = int(y + rh)

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


def mirrored_roi_bounds(frame, roi):
    x1, y1, x2, y2 = roi_bounds(frame, roi)
    width = frame.shape[1]
    return width - x2, y1, width - x1, y2


def existing_count(output_root, letter):
    folder = output_root / letter
    if not folder.exists():
        return 0
    return len([path for path in folder.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov"}])


def draw_hud(frame, letter, count, recording, remaining, roi, mirror_display):
    display = cv2.flip(frame, 1) if mirror_display else frame.copy()
    x1, y1, x2, y2 = mirrored_roi_bounds(frame, roi) if mirror_display else roi_bounds(frame, roi)
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    cv2.putText(display, f"Letter: {letter}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(display, f"Saved clips: {count}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if recording:
        cv2.putText(display, f"Recording {remaining:.1f}s", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(display, "SPACE record | N/P letter | Q quit", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return display


def main():
    parser = argparse.ArgumentParser(description="Record live ASL letter clips")
    parser.add_argument("--output-root", default="data/videos/letters")
    parser.add_argument("--seconds", type=float, default=2.0, help="Seconds per clip")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", type=parse_roi, default=parse_roi("0.42,0.05,0.55,0.90"))
    parser.add_argument("--start-letter", default="A")
    parser.add_argument("--mirror-display", action="store_true", default=True)
    parser.add_argument("--no-mirror-display", action="store_false", dest="mirror_display")
    parser.add_argument("--save-mirrored", action="store_true", help="Only use if live prediction also uses --mirror-prediction")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open webcam")

    letter_index = max(0, LETTERS.find(args.start_letter.upper()[:1]))
    recording = False
    writer = None
    clip_path = None
    started_at = 0.0
    saved_counts = {letter: existing_count(output_root, letter) for letter in LETTERS}

    print("Recorder opened")
    print("  SPACE: record one clip")
    print("  N/P: next/previous letter")
    print("  Q: quit")
    print("Record 10-20 clips per letter if possible, especially A/E and Q/W.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break

            letter = LETTERS[letter_index]
            now = time.monotonic()
            remaining = max(0.0, args.seconds - (now - started_at)) if recording else 0.0

            if recording:
                save_frame = cv2.flip(frame, 1) if args.save_mirrored else frame
                x1, y1, x2, y2 = roi_bounds(save_frame, args.roi)
                crop = save_frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (224, 224))
                writer.write(crop)
                if now - started_at >= args.seconds:
                    writer.release()
                    writer = None
                    recording = False
                    saved_counts[letter] += 1
                    print(f"Saved {clip_path}")

            display = draw_hud(
                frame,
                letter,
                saved_counts[letter],
                recording,
                remaining,
                args.roi,
                args.mirror_display,
            )
            cv2.imshow("ASL Letter Dataset Recorder", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if recording:
                continue
            if key == ord("n"):
                letter_index = (letter_index + 1) % len(LETTERS)
            elif key == ord("p"):
                letter_index = (letter_index - 1) % len(LETTERS)
            elif key == ord(" "):
                letter_folder = output_root / letter
                letter_folder.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                clip_path = letter_folder / f"live_{letter}_{stamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(clip_path), fourcc, args.fps, (224, 224))
                if not writer.isOpened():
                    print(f"Could not open writer for {clip_path}")
                    writer = None
                    continue
                started_at = time.monotonic()
                recording = True
                print(f"Recording {letter} -> {clip_path}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

    print("Counts:")
    for letter in LETTERS:
        print(f"  {letter}: {saved_counts[letter]}")


if __name__ == "__main__":
    main()
