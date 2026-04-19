import random

import torch
from torch.utils.data import Dataset


class TinyDetectionDataset(Dataset):
    def __init__(self, split="train", image_size=256, num_samples=16, num_classes=4, seed=42):
        self.split = split
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_classes = num_classes
        base_seed = seed if split == "train" else seed + 1000
        self.samples = [self._build_sample(base_seed + i) for i in range(num_samples)]

    def _build_sample(self, seed):
        rng = random.Random(seed)
        image = torch.full((3, self.image_size, self.image_size), 0.08, dtype=torch.float32)

        num_objects = rng.randint(1, 3)
        boxes = []
        labels = []

        palette = {
            1: torch.tensor([1.0, 0.25, 0.25], dtype=torch.float32).view(3, 1, 1),
            2: torch.tensor([0.2, 0.9, 0.3], dtype=torch.float32).view(3, 1, 1),
            3: torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32).view(3, 1, 1),
        }

        for _ in range(num_objects):
            label = rng.randint(1, self.num_classes - 1)
            width = rng.randint(self.image_size // 8, self.image_size // 3)
            height = rng.randint(self.image_size // 8, self.image_size // 3)
            x1 = rng.randint(0, self.image_size - width - 1)
            y1 = rng.randint(0, self.image_size - height - 1)
            x2 = x1 + width
            y2 = y1 + height

            image[:, y1:y2, x1:x2] = palette[label]
            boxes.append([x1, y1, x2, y2])
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, target = self.samples[index]
        return image.clone(), {k: v.clone() for k, v in target.items()}
