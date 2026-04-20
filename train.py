import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from tqdm import tqdm

from models.backbone import MODEL_BACKBONE_CHOICES, MODEL_RECOMMENDED_BACKBONES
from models.detr_resnet import DETRDetector
from models.fast_rcnn import FastRCNNDetector
from models.faster_rcnn import FasterRCNNDetector
from models.rcnn import RCNNDetector
from models.ssd_resnet import SSDDetector
from models.yolo_resnet import YOLODetector
from utils.dataset import get_dataloader, get_num_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Training Settings")
    parser.add_argument(
        "--model",
        type=str,
        default="faster_rcnn",
        choices=["rcnn", "fast_rcnn", "faster_rcnn", "yolo", "ssd", "detr"],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="auto",
        choices=["auto", "resnet18", "resnet34", "resnet50", "mobilenetv3", "cspdarknet"],
    )
    parser.add_argument("--dataset", type=str, default="VOC", choices=["toy", "VOC", "COCO", "CONSTRUCTION_PPE"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--toy_train_samples", type=int, default=16)
    parser.add_argument("--toy_val_samples", type=int, default=8)
    return parser.parse_args()


def ensure_output_dir(output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_log_path(output_dir, model_name, dataset_name, backbone_name):
    return output_dir / f"{model_name}_{backbone_name}_{dataset_name.lower()}_training_log.csv"


def append_epoch_log(log_path, row):
    file_exists = log_path.exists()
    fieldnames = [
        "epoch",
        "model",
        "backbone",
        "dataset",
        "train_loss",
        "val_loss",
        "lr",
        "epoch_time_sec",
        "iter_time_sec",
        "checkpoint",
    ]

    with log_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_epoch_zero_if_needed(
    log_path,
    args,
    model,
    val_loader,
    device,
    use_amp,
    num_classes,
    non_blocking,
):
    if args.resume:
        return
    if log_path.exists():
        return

    val_loss = validate_epoch(
        args.model,
        model,
        val_loader,
        device,
        use_amp,
        args.input_size,
        num_classes,
        non_blocking,
    )

    append_epoch_log(log_path, {
        "epoch": 0,
        "model": args.model,
        "backbone": args.backbone,
        "dataset": args.dataset,
        "train_loss": "",
        "val_loss": f"{val_loss:.6f}",
        "lr": f"{0.0:.8f}",
        "epoch_time_sec": f"{0.0:.4f}",
        "iter_time_sec": f"{0.0:.4f}",
        "checkpoint": "",
    })


def save_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, scaler, epoch, args, best_val_loss):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, lr_scheduler, scaler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    return start_epoch, best_val_loss


def create_model(model_name, num_classes, pretrained_backbone, backbone_name, input_size):
    if backbone_name not in MODEL_BACKBONE_CHOICES[model_name]:
        supported = ", ".join(MODEL_BACKBONE_CHOICES[model_name])
        raise ValueError(f"Backbone {backbone_name} is not supported for {model_name}. Supported: {supported}.")
    if model_name == "rcnn":
        return RCNNDetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone, backbone_name=backbone_name)
    if model_name == "fast_rcnn":
        return FastRCNNDetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone, backbone_name=backbone_name)
    if model_name == "faster_rcnn":
        return FasterRCNNDetector(
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            backbone_name=backbone_name,
            input_size=input_size,
        )
    if model_name == "yolo":
        return YOLODetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone, backbone_name=backbone_name)
    if model_name == "ssd":
        return SSDDetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone, backbone_name=backbone_name)
    if model_name == "detr":
        return DETRDetector(num_classes=num_classes, pretrained_backbone=pretrained_backbone, backbone_name=backbone_name)
    raise ValueError(f"Unsupported model: {model_name}")


def get_effective_input_size(model_name, input_size):
    if model_name == "ssd":
        return max(input_size, 512)
    if model_name == "yolo":
        return max(input_size, 256)
    return input_size


def resize_images_and_targets(images, targets, input_size, device, non_blocking):
    resized_images = []
    resized_targets = []
    scale = torch.tensor([input_size, input_size, input_size, input_size], dtype=torch.float32, device=device)

    for image, target in zip(images, targets):
        image = image.to(device, non_blocking=non_blocking)
        target_on_device = {k: v.to(device, non_blocking=non_blocking) for k, v in target.items()}

        orig_h, orig_w = image.shape[-2:]
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        boxes = target_on_device["boxes"].clone()
        if boxes.numel() > 0:
            boxes[:, [0, 2]] *= input_size / orig_w
            boxes[:, [1, 3]] *= input_size / orig_h

        resized_images.append(resized)
        resized_targets.append({
            "boxes": boxes,
            "labels": target_on_device["labels"],
            "norm_boxes": boxes / scale,
        })

    return torch.stack(resized_images, dim=0), resized_targets


def build_roi_batch(images, targets, crop_size=224):
    crops = []
    labels = []
    boxes = []

    for image, target in zip(images, targets):
        image_h, image_w = image.shape[-2:]
        if target["boxes"].numel() == 0:
            continue

        for box, label in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(int(x1), image_w - 1))
            y1 = max(0, min(int(y1), image_h - 1))
            x2 = max(x1 + 1, min(int(x2), image_w))
            y2 = max(y1 + 1, min(int(y2), image_h))

            crop = TF.resized_crop(image, y1, x1, y2 - y1, x2 - x1, [crop_size, crop_size])
            crops.append(crop)
            labels.append(label)
            boxes.append(box / torch.tensor([image_w, image_h, image_w, image_h], device=image.device))

    if not crops:
        return None, None, None

    return torch.stack(crops, dim=0), torch.stack(labels), torch.stack(boxes)


def regression_loss_for_class_logits(bbox_deltas, labels, target_boxes, num_classes):
    pred_boxes = bbox_deltas.view(-1, num_classes, 4)
    batch_indices = torch.arange(labels.size(0), device=labels.device)
    chosen_pred_boxes = pred_boxes[batch_indices, labels]
    return F.smooth_l1_loss(chosen_pred_boxes, target_boxes)


def yolo_smoke_loss(predictions, targets, num_classes):
    batch_size, grid_h, grid_w, channels = predictions.shape
    device = predictions.device
    box_dim = 5
    num_bboxes = (channels - num_classes) // box_dim
    target_tensor = torch.zeros_like(predictions)
    object_mask = torch.zeros((batch_size, grid_h, grid_w), dtype=torch.bool, device=device)

    for batch_idx, target in enumerate(targets):
        if target["boxes"].numel() == 0:
            continue

        box = target["norm_boxes"][0].clamp(0.0, 1.0)
        label = int(target["labels"][0].item())
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) * 0.5).item()
        cy = ((y1 + y2) * 0.5).item()
        bw = (x2 - x1).item()
        bh = (y2 - y1).item()

        cell_x = min(grid_w - 1, int(cx * grid_w))
        cell_y = min(grid_h - 1, int(cy * grid_h))
        object_mask[batch_idx, cell_y, cell_x] = True

        for box_idx in range(num_bboxes):
            base = box_idx * box_dim
            target_tensor[batch_idx, cell_y, cell_x, base:base + 5] = torch.tensor(
                [cx, cy, bw, bh, 1.0],
                device=device,
                dtype=predictions.dtype,
            )

        target_tensor[batch_idx, cell_y, cell_x, num_bboxes * box_dim + label] = 1.0

    box_pred = predictions[..., :num_bboxes * box_dim]
    box_target = target_tensor[..., :num_bboxes * box_dim]
    cls_pred = predictions[..., num_bboxes * box_dim:]
    cls_target = target_tensor[..., num_bboxes * box_dim:]

    if object_mask.any():
        obj_mask = object_mask.unsqueeze(-1)
        loss_box = F.mse_loss(box_pred[obj_mask.expand_as(box_pred)], box_target[obj_mask.expand_as(box_target)])
        loss_cls = F.mse_loss(cls_pred[obj_mask.expand_as(cls_pred)], cls_target[obj_mask.expand_as(cls_target)])
    else:
        loss_box = box_pred.sum() * 0.0
        loss_cls = cls_pred.sum() * 0.0

    return loss_box + loss_cls


def ssd_smoke_loss(locs, confs, targets, num_classes):
    all_loc_losses = []
    all_cls_losses = []

    for batch_idx, target in enumerate(targets):
        if target["boxes"].numel() == 0:
            continue

        count = min(target["boxes"].size(0), locs.size(1))
        gt_boxes = target["norm_boxes"][:count]
        gt_labels = target["labels"][:count]
        pred_boxes = locs[batch_idx, :count]
        pred_labels = confs[batch_idx, :count]

        all_loc_losses.append(F.smooth_l1_loss(pred_boxes, gt_boxes))
        all_cls_losses.append(F.cross_entropy(pred_labels, gt_labels))

    if not all_loc_losses:
        return locs.sum() * 0.0

    return torch.stack(all_loc_losses).mean() + torch.stack(all_cls_losses).mean()


def detr_smoke_loss(out_class, out_bbox, targets):
    cls_losses = []
    box_losses = []
    background_class = out_class.size(-1) - 1

    for batch_idx, target in enumerate(targets):
        if target["boxes"].numel() == 0:
            cls_target = torch.full(
                (out_class.size(1),),
                background_class,
                dtype=torch.long,
                device=out_class.device,
            )
            cls_losses.append(F.cross_entropy(out_class[batch_idx], cls_target))
            continue

        count = min(target["boxes"].size(0), out_bbox.size(1))
        cls_target = torch.full(
            (out_class.size(1),),
            background_class,
            dtype=torch.long,
            device=out_class.device,
        )
        cls_target[:count] = target["labels"][:count]
        cls_losses.append(F.cross_entropy(out_class[batch_idx], cls_target))
        box_losses.append(F.l1_loss(out_bbox[batch_idx, :count], target["norm_boxes"][:count]))

    loss_cls = torch.stack(cls_losses).mean() if cls_losses else out_class.sum() * 0.0
    loss_box = torch.stack(box_losses).mean() if box_losses else out_bbox.sum() * 0.0
    return loss_cls + loss_box


def compute_batch_loss(model_name, model, images, targets, device, use_amp, input_size, num_classes, non_blocking):
    if model_name == "faster_rcnn":
        images = [image.to(device, non_blocking=non_blocking) for image in images]
        targets = [{k: v.to(device, non_blocking=non_blocking) for k, v in t.items()} for t in targets]
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
        return loss

    effective_input_size = get_effective_input_size(model_name, input_size)
    batch_images, resized_targets = resize_images_and_targets(
        images, targets, effective_input_size, device, non_blocking
    )

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        if model_name == "rcnn":
            roi_crops, labels, target_boxes = build_roi_batch(batch_images, resized_targets)
            if roi_crops is None:
                return batch_images.sum() * 0.0
            cls_scores, bbox_deltas = model(roi_crops)
            loss_cls = F.cross_entropy(cls_scores, labels)
            loss_box = regression_loss_for_class_logits(bbox_deltas, labels, target_boxes, num_classes)
            return loss_cls + loss_box

        if model_name == "fast_rcnn":
            rois = [target["boxes"] for target in resized_targets]
            cls_scores, bbox_deltas = model(batch_images, rois)
            labels = torch.cat([target["labels"] for target in resized_targets], dim=0)
            target_boxes = torch.cat([target["norm_boxes"] for target in resized_targets], dim=0)
            loss_cls = F.cross_entropy(cls_scores, labels)
            loss_box = regression_loss_for_class_logits(bbox_deltas, labels, target_boxes, num_classes)
            return loss_cls + loss_box

        if model_name == "yolo":
            predictions = model(batch_images)
            return yolo_smoke_loss(predictions, resized_targets, num_classes)

        if model_name == "ssd":
            locs, confs = model(batch_images)
            return ssd_smoke_loss(locs, confs, resized_targets, num_classes)

        if model_name == "detr":
            out_class, out_bbox = model(batch_images)
            return detr_smoke_loss(out_class, out_bbox, resized_targets)

    raise ValueError(f"Unsupported model: {model_name}")


def resolve_backbone_name(model_name, backbone_name):
    if backbone_name == "auto":
        return MODEL_RECOMMENDED_BACKBONES[model_name]
    return backbone_name


def set_batchnorm_eval(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def validate_epoch(model_name, model, val_loader, device, use_amp, input_size, num_classes, non_blocking):
    if model_name == "faster_rcnn":
        model.train()
        model.apply(set_batchnorm_eval)
    else:
        model.eval()

    total_val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating", leave=False):
            loss = compute_batch_loss(
                model_name,
                model,
                images,
                targets,
                device,
                use_amp,
                input_size,
                num_classes,
                non_blocking,
            )
            total_val_loss += loss.item()

    return total_val_loss / max(1, len(val_loader))


def main():
    args = parse_args()
    args.backbone = resolve_backbone_name(args.model, args.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory
    use_amp = args.amp and device.type == "cuda"

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Running model: {args.model}")
    print(f"Backbone: {args.backbone}")
    print(f"Dataset: {args.dataset}")
    print(f"AMP enabled: {use_amp}")

    output_dir = ensure_output_dir(args.output_dir)
    log_path = get_log_path(output_dir, args.model, args.dataset, args.backbone)

    train_loader = get_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        toy_image_size=args.input_size,
        toy_train_samples=args.toy_train_samples,
        toy_val_samples=args.toy_val_samples,

    )
    val_loader = get_dataloader(
        args.dataset,
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        toy_image_size=args.input_size,
        toy_train_samples=args.toy_train_samples,
        toy_val_samples=args.toy_val_samples,

    )

    num_classes = get_num_classes(args.dataset)
    model = create_model(args.model, num_classes, args.pretrained_backbone, args.backbone, args.input_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_val_loss = load_checkpoint(
            args.resume,
            model,
            optimizer,
            lr_scheduler,
            scaler,
            device,
        )
        print(f"Resume start epoch: {start_epoch + 1}")

    log_epoch_zero_if_needed(
        log_path,
        args,
        model,
        val_loader,
        device,
        use_amp,
        num_classes,
        non_blocking,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True, mininterval=0.5)

        for step, (images, targets) in enumerate(pbar, start=1):
            iter_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            loss = compute_batch_loss(
                args.model,
                model,
                images,
                targets,
                device,
                use_amp,
                args.input_size,
                num_classes,
                non_blocking,
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            iter_time = time.time() - iter_start
            if step == 1 or step % 10 == 0 or step == len(train_loader):
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "iter_s": f"{iter_time:.2f}",
                })

        lr_scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(1, len(train_loader))
        avg_iter_time = epoch_time / max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss = validate_epoch(
            args.model,
            model,
            val_loader,
            device,
            use_amp,
            args.input_size,
            num_classes,
            non_blocking,
        )

        print(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f}")
        print(f"Epoch {epoch + 1} Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch + 1} Time: {epoch_time:.2f}s, Avg Iter Time: {avg_iter_time:.2f}s")

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1):
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                args,
                best_val_loss,
            )

        save_checkpoint(
            output_dir / "last_checkpoint.pth",
            model,
            optimizer,
            lr_scheduler,
            scaler,
            epoch,
            args,
            best_val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                lr_scheduler,
                scaler,
                epoch,
                args,
                best_val_loss,
            )

        append_epoch_log(log_path, {
            "epoch": epoch + 1,
            "model": args.model,
            "backbone": args.backbone,
            "dataset": args.dataset,
            "train_loss": f"{avg_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
            "lr": f"{current_lr:.8f}",
            "epoch_time_sec": f"{epoch_time:.4f}",
            "iter_time_sec": f"{avg_iter_time:.4f}",
            "checkpoint": str(checkpoint_path),
        })

        with (output_dir / "latest_run_summary.json").open("w", encoding="utf-8") as f:
            json.dump({
                "epoch": epoch + 1,
                "model": args.model,
                "backbone": args.backbone,
                "dataset": args.dataset,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "lr": current_lr,
                "epoch_time_sec": epoch_time,
                "iter_time_sec": avg_iter_time,
                "last_checkpoint": str(output_dir / "last_checkpoint.pth"),
            }, f, indent=2)


if __name__ == "__main__":
    main()
