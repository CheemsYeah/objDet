from pathlib import Path

import torch
from torchvision import datasets, transforms

from utils.toy_dataset import TinyDetectionDataset
from utils.yolo_detection_dataset import YOLODetectionDataset, infer_num_classes_from_labels


def custom_collate_fn(batch):
    return tuple(zip(*batch))


def coco_target_transform(target):
    boxes = []
    labels = []
    for obj in target:
        x, y, w, h = obj["bbox"]
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

    if boxes:
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.int64),
    }


class VOCDetectionTransform:
    def __call__(self, image, target):
        img_tensor = transforms.ToTensor()(image)

        boxes = []
        labels = []
        voc_classes = {
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20,
        }

        objects = target["annotation"].get("object", [])
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            name = obj["name"].lower()
            if name not in voc_classes:
                continue

            bndbox = obj["bndbox"]
            xmin = float(bndbox["xmin"]) - 1
            ymin = float(bndbox["ymin"]) - 1
            xmax = float(bndbox["xmax"]) - 1
            ymax = float(bndbox["ymax"]) - 1

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(voc_classes[name])

        return img_tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


class COCODetectionTransform:
    def __call__(self, image, target):
        img_tensor = transforms.ToTensor()(image)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annotation in target:
            bbox = annotation.get("bbox")
            category_id = annotation.get("category_id")
            if bbox is None or category_id is None:
                continue

            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                continue

            boxes.append([x, y, x + width, y + height])
            labels.append(int(category_id))
            areas.append(float(annotation.get("area", width * height)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        return img_tensor, {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }


def get_construction_ppe_root(root_dir="./datasets"):
    return Path(root_dir) / "construction_ppe"


def resolve_construction_ppe_split(root_dir, train):
    dataset_root = get_construction_ppe_root(root_dir)
    if train:
        return "train"
    for split_name in ("valid", "val", "test"):
        if (dataset_root / "images" / split_name).exists():
            return split_name
    raise FileNotFoundError(
        f"Could not find validation split under {dataset_root / 'images'}. "
        f"Expected one of: valid, val, test."
    )


def get_num_classes(dataset_name, root_dir="./datasets"):
    mapping = {
        "toy": 4,
        "VOC": 21,
        "COCO": 91,
    }
    if dataset_name == "CONSTRUCTION_PPE":
        labels_dir = get_construction_ppe_root(root_dir) / "labels" / "train"
        return infer_num_classes_from_labels(labels_dir)
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported Dataset: {dataset_name}")
    return mapping[dataset_name]


def get_dataloader(
    dataset_name="toy",
    root_dir="./datasets",
    batch_size=4,
    train=True,
    num_workers=4,
    pin_memory=False,
    toy_image_size=256,
    toy_train_samples=16,
    toy_val_samples=8,
):
    if dataset_name == "toy":
        dataset = TinyDetectionDataset(
            split="train" if train else "val",
            image_size=toy_image_size,
            num_samples=toy_train_samples if train else toy_val_samples,
        )
    elif dataset_name == "VOC":
        image_set = "trainval" if train else "test"
        dataset = datasets.VOCDetection(
            root=root_dir,
            year="2007",
            image_set=image_set,
            download=True,
            transforms=VOCDetectionTransform(),
        )
    elif dataset_name == "COCO":
        split = "train2017" if train else "val2017"
        ann_file = f"{root_dir}/coco/annotations/instances_{split}.json"
        img_dir = f"{root_dir}/coco/{split}"
        dataset = datasets.CocoDetection(
            root=img_dir,
            annFile=ann_file,
            transform=transforms.ToTensor(),
            target_transform=coco_target_transform
        )
    elif dataset_name == "CONSTRUCTION_PPE":
        split = resolve_construction_ppe_split(root_dir, train)
        dataset = YOLODetectionDataset(get_construction_ppe_root(root_dir), split=split)
    else:
        raise ValueError("Unsupported Dataset")

    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": custom_collate_fn,
    }

    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = 4

    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
