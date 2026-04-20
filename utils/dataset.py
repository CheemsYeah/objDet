import torch
from torchvision import datasets, transforms

from utils.toy_dataset import TinyDetectionDataset


def custom_collate_fn(batch):
    return tuple(zip(*batch))


def coco_target_transform(target):
    """专门为 COCO 数据集处理标签的转换器"""
    boxes = []
    labels = []
    for obj in target:
        # COCO 原生格式是 [x_min, y_min, width, height]
        x, y, w, h = obj['bbox']

        # 转换为 PyTorch 需要的 [x_min, y_min, x_max, y_max]
        if w > 0 and h > 0:
            boxes.append([x, y, x + w, y + h])
            # COCO 的类别 ID 并不是连续的 (1~90中间有跳过)
            labels.append(obj['category_id'])

    target_dict = {}
    if len(boxes) > 0:
        target_dict['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target_dict['labels'] = torch.tensor(labels, dtype=torch.int64)
    else:
        # 🔥 关键修复：如果没有边界框，必须生成 [0, 4] 形状的空张量！
        target_dict['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target_dict['labels'] = torch.zeros((0,), dtype=torch.int64)

    return target_dict

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

        target_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img_tensor, target_dict


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

        target_dict = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
        }
        return img_tensor, target_dict


def get_num_classes(dataset_name):
    mapping = {
        "toy": 4,
        "VOC": 21,
        "COCO": 91,
    }
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
            target_transform=coco_target_transform  # <--- 新增这一行
        )
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

    return torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
