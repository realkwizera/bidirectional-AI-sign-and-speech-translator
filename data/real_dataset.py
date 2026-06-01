import hashlib
import os
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils.hand_landmarks import HAND_LANDMARK_DIM, HandLandmarkExtractor


BLANK_ID = 0
LANDMARK_CACHE_VERSION = "hand-landmarks-tight-crop-v3"
CHAR_TO_ID = {" ": 1}
for i, char in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    CHAR_TO_ID[char] = i + 2
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
ID_TO_CHAR[BLANK_ID] = ""


class MultiSourceSignLanguageDataset(Dataset):
    """Letter images + letter videos prepared with the same MediaPipe hand crop."""

    def __init__(
        self,
        root_dir="data/videos",
        image_root="data/images",
        num_frames=30,
        image_size=64,
        include_sources=("images", "letters"),
        use_landmarks=True,
        landmark_cache_dir="data/landmark_cache",
        drop_missing_image_landmarks=False,
        shuffle_seed=42,
        crop_padding=0.35,
    ):
        self.num_frames = num_frames
        self.image_size = image_size
        self.include_sources = set(include_sources)
        self.use_landmarks = use_landmarks
        self.landmark_cache_dir = Path(landmark_cache_dir)
        self.drop_missing_image_landmarks = drop_missing_image_landmarks
        self.shuffle_seed = shuffle_seed
        self.crop_padding = crop_padding
        self.landmark_extractor = None
        self.samples = []

        video_root = Path(root_dir)
        for source in ("words", "letters", "sentences"):
            if source in self.include_sources:
                self._index_video_source(video_root / source, source)
        if "images" in self.include_sources:
            self._index_image_source(Path(image_root))
        if self.drop_missing_image_landmarks:
            self._drop_missing_image_landmarks()

        self.samples.sort(key=lambda sample: (sample["source"], sample["label"], sample["path"]))
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(self.samples)
        counts = Counter(sample["source"] for sample in self.samples)
        summary = ", ".join(f"{source}={count}" for source, count in sorted(counts.items()))
        print(f"Indexed {len(self.samples)} samples ({summary})")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["landmark_extractor"] = None
        return state

    @staticmethod
    def normalize_label(label):
        normalized = label.upper().replace("_", " ").replace("-", " ")
        return "".join(char for char in normalized if char in CHAR_TO_ID)

    def _index_video_source(self, source_dir, source):
        if not source_dir.exists():
            return
        count = 0
        for label_path in sorted(source_dir.iterdir()):
            if not label_path.is_dir():
                continue
            label = self.normalize_label(label_path.name)
            if not label:
                continue
            for video_file in sorted(label_path.iterdir()):
                if video_file.suffix.lower() in {".mp4", ".avi", ".mov"}:
                    self.samples.append({
                        "path": str(video_file),
                        "label": label,
                        "raw_label": label_path.name,
                        "source": source,
                        "kind": "video",
                    })
                    count += 1
        print(f"Indexed {count} {source} videos")

    def _index_image_source(self, image_dir):
        if not image_dir.exists():
            return
        count = 0
        for label_path in sorted(image_dir.iterdir()):
            if not label_path.is_dir():
                continue
            label = self.normalize_label(label_path.name)
            if not label:
                continue
            for image_file in sorted(label_path.iterdir()):
                if image_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append({
                        "path": str(image_file),
                        "label": label,
                        "raw_label": label_path.name,
                        "source": "images",
                        "kind": "image",
                    })
                    count += 1
        print(f"Indexed {count} letter images")

    def _drop_missing_image_landmarks(self):
        import json

        report_path = self.landmark_cache_dir / "image_landmark_report.json"
        if not report_path.exists():
            print("Image landmark report not found; run python scripts/extract_image_landmarks.py before training.")
            return
        report = json.loads(report_path.read_text(encoding="utf-8"))
        missing_paths = {str(Path(item["path"])) for item in report.get("missing_samples", [])}
        before = len(self.samples)
        self.samples = [
            sample for sample in self.samples
            if sample["source"] != "images" or str(Path(sample["path"])) not in missing_paths
        ]
        removed = before - len(self.samples)
        if removed:
            print(f"Dropped {removed} image samples without real hand landmarks")

    def __len__(self):
        return len(self.samples)

    def _ensure_landmark_extractor(self):
        if self.landmark_extractor is None:
            self.landmark_extractor = HandLandmarkExtractor(
                static_image_mode=False,
                allow_contour_fallback=False,
            )

    def _prepare_bgr_frame(self, frame):
        import cv2
        import numpy as np

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = np.zeros(HAND_LANDMARK_DIM, dtype=np.float32)
        if self.use_landmarks:
            self._ensure_landmark_extractor()
            result = self.landmark_extractor.extract(frame_rgb)
            if result.landmarks is not None and result.presence > 0.0:
                frame_rgb, crop_points, _ = HandLandmarkExtractor.crop_to_landmarks(
                    frame_rgb,
                    result.landmarks,
                    padding=self.crop_padding,
                )
                landmarks = HandLandmarkExtractor.features_from_landmarks(crop_points, result.presence)

        frame_rgb = cv2.resize(frame_rgb, (self.image_size, self.image_size))
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        return frame_rgb, landmarks.astype(np.float32)

    def _extract_frames(self, video_path):
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        landmarks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb, frame_landmarks = self._prepare_bgr_frame(frame)
            frames.append(frame_rgb)
            landmarks.append(frame_landmarks)
        cap.release()
        return np.array(frames, dtype=np.float32), np.array(landmarks, dtype=np.float32)

    def _load_image_sequence(self, image_path):
        import cv2
        import numpy as np

        frame = cv2.imread(str(image_path))
        if frame is None:
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            landmarks = np.zeros(HAND_LANDMARK_DIM, dtype=np.float32)
        else:
            frame, landmarks = self._prepare_bgr_frame(frame)
        return (
            np.repeat(frame[None, ...], self.num_frames, axis=0),
            np.repeat(landmarks[None, ...], self.num_frames, axis=0),
        )

    def _pad_or_trim(self, frames, landmarks):
        import numpy as np

        if len(frames) == 0:
            return (
                np.zeros((self.num_frames, self.image_size, self.image_size, 3), dtype=np.float32),
                np.zeros((self.num_frames, HAND_LANDMARK_DIM), dtype=np.float32),
            )
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames).astype(int)
            frames = frames[indices]
            landmarks = landmarks[indices]
        elif len(frames) < self.num_frames:
            pad_count = self.num_frames - len(frames)
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad_count, axis=0)], axis=0)
            landmarks = np.concatenate([landmarks, np.repeat(landmarks[-1:], pad_count, axis=0)], axis=0)
        return frames, landmarks

    def _prepared_cache_path(self, sample):
        source_path = Path(sample["path"])
        stat = source_path.stat() if source_path.exists() else None
        task_model = HandLandmarkExtractor._resolve_task_model(None)
        task_stat = task_model.stat() if task_model and task_model.exists() else None
        landmark_backend_key = (
            f"task:{task_model.resolve()}:{task_stat.st_mtime_ns}:{task_stat.st_size}"
            if task_model and task_stat
            else "real-landmark-model-missing"
        )
        key = "|".join([
            LANDMARK_CACHE_VERSION,
            landmark_backend_key,
            str(source_path.resolve()),
            str(stat.st_mtime_ns if stat else 0),
            str(stat.st_size if stat else 0),
            str(self.num_frames),
            str(self.image_size),
            str(self.crop_padding),
            sample["kind"],
        ])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.landmark_cache_dir / f"{digest}.prepared.npz"

    def _load_prepared_sample(self, sample):
        import numpy as np

        cache_path = self._prepared_cache_path(sample)
        if not cache_path.exists():
            return None
        try:
            with np.load(cache_path) as cached:
                frames = cached["frames"].astype(np.float32)
                landmarks = cached["landmarks"].astype(np.float32)
        except Exception:
            return None
        if frames.shape != (self.num_frames, self.image_size, self.image_size, 3):
            return None
        if landmarks.shape != (self.num_frames, HAND_LANDMARK_DIM):
            return None
        return frames, landmarks

    def _save_prepared_sample(self, sample, frames, landmarks):
        import numpy as np

        cache_path = self._prepared_cache_path(sample)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temp_path, frames=frames, landmarks=landmarks)
        os.replace(temp_path, cache_path)

    def get_sample_weights(self):
        label_counts = Counter((sample["source"], sample["label"]) for sample in self.samples)
        labels_by_source = {}
        for sample in self.samples:
            labels_by_source.setdefault(sample["source"], set()).add(sample["label"])

        weights = []
        for sample in self.samples:
            source_label_count = max(len(labels_by_source[sample["source"]]), 1)
            label_count = max(label_counts[(sample["source"], sample["label"])], 1)
            weights.append(1.0 / (source_label_count * label_count))
        return torch.tensor(weights, dtype=torch.double)

    def get_class_weights(self):
        counts = torch.ones(len(ID_TO_CHAR), dtype=torch.float32)
        for sample in self.samples:
            for char in sample["label"]:
                if char in CHAR_TO_ID:
                    counts[CHAR_TO_ID[char]] += 1.0
        weights = counts.sum() / (counts * len(counts))
        weights[BLANK_ID] = 0.0
        return weights / weights[1:].mean()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cached = self._load_prepared_sample(sample)
        if cached is not None:
            frames, landmarks = cached
        elif sample["kind"] == "image":
            frames, landmarks = self._load_image_sequence(sample["path"])
            self._save_prepared_sample(sample, frames, landmarks)
        else:
            frames, landmarks = self._extract_frames(sample["path"])
            frames, landmarks = self._pad_or_trim(frames, landmarks)
            self._save_prepared_sample(sample, frames, landmarks)

        char_labels = torch.tensor(
            [CHAR_TO_ID[char] for char in sample["label"] if char in CHAR_TO_ID],
            dtype=torch.long,
        )
        return {
            "frames": torch.tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2),
            "labels": char_labels,
            "landmarks": torch.tensor(landmarks, dtype=torch.float32),
            "frame_len": torch.tensor(frames.shape[0]),
            "label_len": torch.tensor(len(char_labels)),
            "text": sample["label"],
            "source": sample["source"],
        }
