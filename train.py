from torch.utils.data import DataLoader
from data.video_dataset import VideoASLDataset

dataset = VideoASLDataset(
    root_dir="data/videos",
    mode="words",
    num_frames=30,
    image_size=64
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

for frames, labels in loader:
    print(frames.shape)  # (B, T, C, H, W)
    print(labels)
    break