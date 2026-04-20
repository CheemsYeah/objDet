import ast
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _read_text(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read()


def _extract_names_from_data_yaml(dataset_root):
    yaml_path = Path(dataset_root) / "data.yaml"
    if not yaml_path.exists():
        return None

    names_value = None
    for line in _read_text(yaml_path).splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            names_value = stripped.split(":", 1)[1].strip()
            break

    if not names_value:
        return None

    try:
        parsed = ast.literal_eval(names_value)
    except (SyntaxError, ValueError):
        return None

    if isinstance(parsed, dict):
        return [parsed[key] for key in sorted(parsed)]
    if isinstance(parsed, list):
        return [str(name) for name in parsed]
    return None


def infer_num_classes_from_labels(labels_dir):
    max_class_id = -1
    for label_path in Path(labels_dir).glob("*.txt"):
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                max_class_id = max(max_class_id, int(float(parts[0])))
    return max_class_id + 2 if max_class_id >= 0 else 1


class YOLODetectionDataset(Dataset):
    def __init__(self, dataset_root, split="train"):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_dir = self.dataset_root / "images" / split
        self.label_dir = self.dataset_root / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        self.image_paths = sorted(
            [path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        self.to_tensor = transforms.ToTensor()
        self.class_names = _extract_names_from_data_yaml(self.dataset_root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size
            image_tensor = self.to_tensor(img)

        boxes = []
        labels = []
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, cx, cy, bw, bh = map(float, parts)
                    x1 = (cx - bw / 2.0) * width
                    y1 = (cy - bh / 2.0) * height
                    x2 = (cx + bw / 2.0) * width
                    y2 = (cy + bh / 2.0) * height
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(class_id) + 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        return image_tensor, target
