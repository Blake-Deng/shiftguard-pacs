#!/usr/bin/env python3
"""Corrected ShiftGuard experiments with target-blind source validation.

Screening does not enumerate or load target images. Formal evaluation restores
one source-validation-selected checkpoint, then constructs and evaluates the
target dataset exactly once.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIMM_VIT_MEAN = (0.5, 0.5, 0.5)
TIMM_VIT_STD = (0.5, 0.5, 0.5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_domains(root: Path) -> dict[str, Path]:
    resolved = {}
    for domain in DOMAINS:
        options = (domain, domain.lower(), domain.replace("_", " "), domain.replace("_", "-"))
        path = next((root / name for name in options if (root / name).is_dir()), None)
        if path is None:
            raise FileNotFoundError(f"Missing PACS domain {domain} under {root}")
        resolved[domain] = path
    return resolved


def source_classes(domain_dirs: dict[str, Path], target: str) -> list[str]:
    return sorted({p.name for d, root in domain_dirs.items() if d != target for p in root.iterdir() if p.is_dir()})


def collect_source_samples(root: Path, target: str, seed: int, val_fraction: float):
    domain_dirs = resolve_domains(root)
    classes = source_classes(domain_dirs, target)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    source_train, source_val = [], []
    rng = random.Random(seed)
    for domain, domain_root in domain_dirs.items():
        if domain == target:
            continue
        for class_name in classes:
            class_root = domain_root / class_name
            files = [] if not class_root.is_dir() else [p for p in class_root.rglob("*") if p.suffix.lower() in EXTENSIONS]
            items = [(str(path), class_to_idx[class_name]) for path in sorted(files)]
            if not items:
                continue
            rng.shuffle(items)
            n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
            source_val.extend(items[:n_val])
            source_train.extend(items[n_val:])
    if not source_train or not source_val:
        raise RuntimeError(f"Found {len(source_train)} source-train and {len(source_val)} source-val images")
    return source_train, source_val, class_to_idx


def collect_target_samples(root: Path, target: str, class_to_idx: dict[str, int]):
    target_root = resolve_domains(root)[target]
    samples = []
    for class_name, label in class_to_idx.items():
        class_root = target_root / class_name
        if not class_root.is_dir():
            continue
        files = [p for p in class_root.rglob("*") if p.suffix.lower() in EXTENSIONS]
        samples.extend((str(path), label) for path in sorted(files))
    if not samples:
        raise RuntimeError(f"No target images found for {target}")
    return samples


class PACSDataset(Dataset):
    def __init__(self, samples, weak, strong=None):
        self.samples = samples
        self.weak = weak
        self.strong = strong

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        weak = self.weak(image)
        strong = self.strong(image) if self.strong is not None else weak
        return weak, strong, label


def make_transforms(image_size: int, preprocessing: str, augmentation_m: int = 9):
    if preprocessing == "legacy_imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        interpolation = transforms.InterpolationMode.BILINEAR
        evaluation_resize = 256
    elif preprocessing == "timm_vit_standard":
        mean, std = TIMM_VIT_MEAN, TIMM_VIT_STD
        interpolation = transforms.InterpolationMode.BICUBIC
        evaluation_resize = int(image_size / 0.9)
    else:
        raise ValueError(f"Unknown preprocessing profile: {preprocessing}")
    weak = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    strong = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=augmentation_m),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    evaluation = transforms.Compose([
        transforms.Resize(evaluation_resize, interpolation=interpolation, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return weak, strong, evaluation


def build_model(name: str, num_classes: int, pretrained: bool):
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "vit-small":
        model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
        if pretrained:
            from timm.models._builder import load_pretrained
            cfg = timm.get_pretrained_cfg("vit_small_patch16_224").to_dict()
            cfg["file"] = str(Path(__file__).resolve().parent / "weights" / "vit_small_patch16_224.npz")
            cfg["url"] = ""
            cfg["hf_hub_id"] = ""
            load_pretrained(model, cfg, num_classes=num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


def forward_features(model, images):
    if hasattr(model, "forward_features") and model.__class__.__module__.startswith("timm"):
        features = model.forward_features(images)
        if features.ndim == 3:
            features = features[:, 0]
        return features, model.get_classifier()(features)
    if isinstance(model, models.ResNet):
        x = model.conv1(images)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        features = model.avgpool(x).flatten(1)
        return features, model.fc(features)
    raise TypeError(type(model))


def one_way_kl(teacher_logits, student_logits, temperature: float):
    teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student, teacher, reduction="none").sum(1) * (temperature ** 2)


def consistency_terms(weak_features, strong_features, weak_logits, strong_logits, args):
    feature_loss = 1.0 - F.cosine_similarity(weak_features.detach(), strong_features, dim=1)
    kl_loss = one_way_kl(weak_logits, strong_logits, args.temperature)
    with torch.no_grad():
        disagreement = 1.0 - F.cosine_similarity(weak_features, strong_features, dim=1)
        confidence = F.softmax(weak_logits, dim=1).amax(dim=1)
        reliability = confidence * torch.exp(-disagreement / args.gate_tau)
        weight = reliability / reliability.mean().clamp_min(1e-6)
        weight = weight.clamp(args.weight_min, args.weight_max)
    return feature_loss, kl_loss, weight


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for weak, _, labels in loader:
        weak = weak.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(weak)
        loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": correct / total, "loss": loss_sum / total}


def run(args):
    if args.gate_tau <= 0:
        raise ValueError("gate_tau must be positive")
    seed_everything(args.seed)
    started = time.time()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weak_tf, strong_tf, eval_tf = make_transforms(args.image_size, args.preprocessing, args.augmentation_m)
    train_samples, val_samples, class_to_idx = collect_source_samples(
        Path(args.data_root), args.target, args.seed, args.val_fraction
    )
    train_ds = PACSDataset(train_samples, weak_tf, strong_tf)
    val_ds = PACSDataset(val_samples, eval_tf, eval_tf)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model = build_model(args.model, len(class_to_idx), not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val = -1.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "cls": 0.0, "feat": 0.0, "kl": 0.0, "weight": 0.0}
        ramp = 1.0 if args.warmup_epochs <= 0 else min(1.0, epoch / args.warmup_epochs)
        for weak, strong, labels in train_loader:
            weak = weak.to(device, non_blocking=True)
            strong = strong.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                weak_features, weak_logits = forward_features(model, weak)
                strong_features, strong_logits = forward_features(model, strong)
                classification = 0.5 * (
                    F.cross_entropy(weak_logits, labels) + F.cross_entropy(strong_logits, labels)
                )
                feature_loss, kl_loss, adaptive_weight = consistency_terms(
                    weak_features, strong_features, weak_logits, strong_logits, args
                )
                if args.method == "aug":
                    loss = classification
                elif args.method == "kl":
                    loss = classification + ramp * args.lambda_kl * kl_loss.mean()
                elif args.method == "feat":
                    loss = classification + ramp * args.lambda_feat * feature_loss.mean()
                elif args.method == "feat_kl":
                    loss = classification + ramp * (
                        args.lambda_feat * feature_loss.mean() + args.lambda_kl * kl_loss.mean()
                    )
                elif args.method == "adaptive":
                    loss = classification + ramp * (
                        args.lambda_feat * (adaptive_weight * feature_loss).mean()
                        + args.lambda_kl * (adaptive_weight * kl_loss).mean()
                    )
                else:
                    raise ValueError(args.method)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += loss.item()
            totals["cls"] += classification.item()
            totals["feat"] += feature_loss.mean().item()
            totals["kl"] += kl_loss.mean().item()
            totals["weight"] += adaptive_weight.mean().item()
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        batches = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / batches,
            "train_cls": totals["cls"] / batches,
            "train_feat": totals["feat"] / batches,
            "train_kl": totals["kl"] / batches,
            "train_weight": totals["weight"] / batches,
            "ramp": ramp,
            "val_accuracy": validation["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={record['train_loss']:.4f} "
            f"val={validation['accuracy']:.4f} ramp={ramp:.2f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("No source-validation checkpoint was selected")
    model.load_state_dict(best_state)
    target_result = None
    target_count = 0
    if not args.skip_target_eval:
        target_samples = collect_target_samples(Path(args.data_root), args.target, class_to_idx)
        target_ds = PACSDataset(target_samples, eval_tf, eval_tf)
        target_loader = DataLoader(target_ds, shuffle=False, **loader_args)
        target_result = evaluate(model, target_loader, device)
        target_count = len(target_ds)

    result = {
        "target": args.target,
        "method": args.method,
        "run_name": args.run_name,
        "model": args.model,
        "seed": args.seed,
        "best_source_val": best_val,
        "target_accuracy": None if target_result is None else target_result["accuracy"],
        "target_loss": None if target_result is None else target_result["loss"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": target_count,
        "target_evaluations": 0 if target_result is None else 1,
        "elapsed_seconds": time.time() - started,
        "config": {
            "lambda_feat": args.lambda_feat,
            "lambda_kl": args.lambda_kl,
            "temperature": args.temperature,
            "gate_tau": args.gate_tau,
            "weight_min": args.weight_min,
            "weight_max": args.weight_max,
            "warmup_epochs": args.warmup_epochs,
            "preprocessing": args.preprocessing,
            "augmentation_N": 2,
            "augmentation_M": args.augmentation_m,
        },
        "history": history,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.target}_{args.model}_{args.run_name}_seed{args.seed}"
    (output / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_checkpoint:
        torch.save(
            {"model": best_state, "args": vars(args), "result": {k: v for k, v in result.items() if k != "history"}},
            output / f"{stem}.pt",
        )
    csv_path = output / "results.csv"
    fields = [
        "target", "method", "run_name", "model", "seed", "best_source_val", "target_accuracy",
        "target_loss", "n_train", "n_val", "n_test", "target_evaluations", "elapsed_seconds",
        "lambda_feat", "lambda_kl", "temperature", "gate_tau", "warmup_epochs",
    ]
    row = {key: result.get(key) for key in fields}
    row.update(result["config"])
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({key: row[key] for key in fields})
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target", required=True, choices=DOMAINS)
    parser.add_argument("--method", choices=["aug", "kl", "feat", "feat_kl", "adaptive"], default="adaptive")
    parser.add_argument("--run-name", default="corrected")
    parser.add_argument("--model", choices=["resnet50", "vit-small"], default="resnet50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--preprocessing",
        choices=["legacy_imagenet", "timm_vit_standard"],
        default="legacy_imagenet",
        help="Explicit preprocessing profile; the default preserves all historical runs.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-feat", type=float, default=0.05)
    parser.add_argument("--lambda-kl", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--augmentation-m", type=int, default=9)
    parser.add_argument("--gate-tau", type=float, default=0.5)
    parser.add_argument("--weight-min", type=float, default=0.5)
    parser.add_argument("--weight-max", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--output", default="runs/corrected")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-target-eval", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
