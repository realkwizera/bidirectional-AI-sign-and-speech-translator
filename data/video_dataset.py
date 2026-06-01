import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VideoASLDataset(Dataset):

    def __init__(
        self,
        root_dir,
        label_map=None,
        num_frames=30,
        image_size=64,
        mode="words"
    ):
        """
        root_dir:
            data/videos/words OR sentences OR letters

        label_map:
            dictionary {label: index}

        num_frames:
            fixed sequence length (IMPORTANT for CTC)

        image_size:
            resize frames (64 or 128 recommended)
        """

        self.root_dir = os.path.join(root_dir, mode)
        self.num_frames = num_frames
        self.image_size = image_size

        self.samples = []

        # build dataset index
        for label in sorted(os.listdir(self.root_dir)):

            label_path = os.path.join(self.root_dir, label)

            if not os.path.isdir(label_path):
                continue

            for video_file in os.listdir(label_path):

                if video_file.endswith((".mp4", ".avi", ".mov")):

                    self.samples.append({
                        "path": os.path.join(label_path, video_file),
                        "label": label
                    })

        # build label map if not provided
        if label_map is None:
            labels = sorted(list(set([s["label"] for s in self.samples])))

            self.label_map = {
                label: idx + 1  # 0 reserved for CTC blank
                for idx, label in enumerate(labels)
            }
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.samples)

    def _extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (self.image_size, self.image_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0

            frames.append(frame)

        cap.release()

        frames = np.array(frames)

        return frames

    def _pad_or_trim(self, frames):

        T = len(frames)

        if T > self.num_frames:
            frames = frames[:self.num_frames]

        elif T < self.num_frames:

            pad = np.zeros(
                (self.num_frames - T,
                 self.image_size,
                 self.image_size,
                 3)
            )

            frames = np.concatenate([frames, pad], axis=0)

        return frames

    def __getitem__(self, idx):

        sample = self.samples[idx]

        video_path = sample["path"]
        label = sample["label"]

        frames = self._extract_frames(video_path)
        frames = self._pad_or_trim(frames)

        # convert to tensor: (T, C, H, W)
        frames = torch.tensor(frames, dtype=torch.float32)
        frames = frames.permute(0, 3, 1, 2)

        label_id = self.label_map[label]

        return frames, torch.tensor(label_id, dtype=torch.long)