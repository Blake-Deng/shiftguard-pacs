#!/usr/bin/env python3
"""Target-isolated VLCS and OfficeHome revision experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from shiftguard_corrected import build_model, forward_features, make_transforms, one_way_kl
from mixstyle_adapter import inject_resnet50
from swad_adapter import LossValleyEpoch

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DATASETS = {
    "pacs": {
        "domains": {
            "Photo": ("Photo", "photo"),
            "Art_Painting": ("Art_Painting", "art_painting", "Art Painting"),
            "Cartoon": ("Cartoon", "cartoon"),
            "Sketch": ("Sketch", "sketch"),
        },
        "classes": 7,
    },
    "vlcs": {
        "domains": {
            "Caltech101": ("Caltech101", "CALTECH", "Caltech", "C"),
            "LabelMe": ("LabelMe", "LABELME", "Labelme", "L"),
            "SUN09": ("SUN09", "SUN", "Sun", "S"),
            "VOC2007": ("VOC2007", "PASCAL", "Pascal", "VOC", "V"),
        },
        "classes": 5,
    },
    "officehome": {
        "domains": {
            "Art": ("Art",),
            "Clipart": ("Clipart", "ClipArt"),
            "Product": ("Product",),
            "Real_World": ("Real_World", "Real World", "RealWorld"),
        },
        "classes": 65,
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_domains(root: Path, dataset: str) -> dict[str, Path]:
    children = {path.name.casefold(): path for path in root.iterdir() if path.is_dir()}
    resolved = {}
    for canonical, aliases in DATASETS[dataset]["domains"].items():
        path = next((children[alias.casefold()] for alias in aliases if alias.casefold() in children), None)
        if path is None:
            raise FileNotFoundError(f"missing {dataset} domain {canonical} under {root}")
        resolved[canonical] = path
    return resolved


def shared_classes(domain_dirs: dict[str, Path]) -> list[str]:
    domain_classes = {
        domain: {path.name for path in domain_root.iterdir() if path.is_dir()}
        for domain, domain_root in domain_dirs.items()
    }
    reference = next(iter(domain_classes.values()))
    for domain, classes in domain_classes.items():
        if classes != reference:
            raise RuntimeError(f"class mismatch in {domain}")
    return sorted(reference)


def exclusion_paths(path: Path, dataset: str) -> tuple[frozenset[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("datasets", {}).get(dataset, [])
    excluded = [entry["path"] for entry in entries]
    if len(excluded) != len(set(excluded)):
        raise RuntimeError(f"duplicate exclusions declared for {dataset}")
    return frozenset(excluded), payload["policy_version"]


def collect_source_samples(root: Path, dataset: str, target: str, seed: int, val_fraction: float, excluded: frozenset[str] = frozenset()):
    domain_dirs = resolve_domains(root, dataset)
    source_domains = [domain for domain in domain_dirs if domain != target]
    source_dirs = {domain: domain_dirs[domain] for domain in source_domains}
    classes = shared_classes(source_dirs)
    if len(classes) != DATASETS[dataset]["classes"]:
        raise RuntimeError(f"expected {DATASETS[dataset]['classes']} classes, found {len(classes)}")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    domain_to_idx = {name: index for index, name in enumerate(source_domains)}
    source_train, source_val = [], []
    rng = random.Random(seed)
    for domain in source_domains:
        for class_name in classes:
            files = [
                path for path in (domain_dirs[domain] / class_name).rglob("*")
                if path.is_file() and path.suffix.lower() in EXTENSIONS
                and path.relative_to(root).as_posix() not in excluded
            ]
            items = [(str(path), class_to_idx[class_name], domain_to_idx[domain]) for path in sorted(files)]
            rng.shuffle(items)
            n_val = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
            source_val.extend(items[:n_val])
            source_train.extend(items[n_val:])
    if not source_train or not source_val:
        raise RuntimeError(f"found {len(source_train)} train and {len(source_val)} validation images")
    return source_train, source_val, class_to_idx, source_domains


def collect_target_samples(root: Path, dataset: str, target: str, class_to_idx: dict[str, int], excluded: frozenset[str] = frozenset()):
    target_root = resolve_domains(root, dataset)[target]
    samples = []
    for class_name, label in class_to_idx.items():
        for path in sorted((target_root / class_name).rglob("*")):
            if (path.is_file() and path.suffix.lower() in EXTENSIONS
                    and path.relative_to(root).as_posix() not in excluded):
                samples.append((str(path), label, -1))
    if not samples:
        raise RuntimeError(f"no target images found for {dataset}/{target}")
    return samples


class DomainImageDataset(Dataset):
    def __init__(self, samples, weak, strong=None):
        self.samples = samples
        self.weak = weak
        self.strong = strong

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, domain = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        weak = self.weak(image)
        strong = self.strong(image) if self.strong is not None else weak
        return weak, strong, label, domain


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for weak, _, labels, _ in loader:
        weak = weak.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(weak)
        loss_sum += F.cross_entropy(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"accuracy": correct / total, "loss": loss_sum / total}


def run(args) -> None:
    if args.target not in DATASETS[args.dataset]["domains"]:
        raise ValueError(f"invalid target {args.target} for {args.dataset}")
    seed_everything(args.seed)
    started = time.time()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    excluded, exclusion_policy = exclusion_paths(Path(args.exclusions), args.dataset)
    weak_tf, strong_tf, eval_tf = make_transforms(args.image_size, "legacy_imagenet")
    train_samples, val_samples, class_to_idx, source_domains = collect_source_samples(
        Path(args.data_root), args.dataset, args.target, args.seed, args.val_fraction, excluded
    )
    uses_two_views = args.method in {"strong_aug", "feature_plus_kl"}
    train_ds = DomainImageDataset(train_samples, weak_tf, strong_tf if uses_two_views else None)
    val_ds = DomainImageDataset(val_samples, eval_tf)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model = build_model("resnet50", len(class_to_idx), not args.no_pretrained)
    if args.method == "mixstyle":
        model = inject_resnet50(model, p=args.mixstyle_p, alpha=args.mixstyle_alpha)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val, best_state = -1.0, None
    swad_valley = LossValleyEpoch() if args.method == "swad" else None
    swad_start_epoch = swad_end_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "cls": 0.0, "feat": 0.0, "kl": 0.0}
        ramp = 1.0 if args.warmup_epochs <= 0 else min(1.0, epoch / args.warmup_epochs)
        for weak, strong, labels, _ in train_loader:
            weak = weak.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                weak_features, weak_logits = forward_features(model, weak)
                if args.method in {"erm", "mixstyle", "swad"}:
                    classification = F.cross_entropy(weak_logits, labels)
                    feature_loss = weak_logits.new_zeros(())
                    kl_loss = weak_logits.new_zeros(())
                    loss = classification
                else:
                    strong = strong.to(device, non_blocking=True)
                    strong_features, strong_logits = forward_features(model, strong)
                    classification = 0.5 * (
                        F.cross_entropy(weak_logits, labels) + F.cross_entropy(strong_logits, labels)
                    )
                    feature_loss = (1.0 - F.cosine_similarity(weak_features.detach(), strong_features, dim=1)).mean()
                    kl_loss = one_way_kl(weak_logits, strong_logits, args.temperature).mean()
                    loss = classification
                    if args.method == "feature_plus_kl":
                        loss = loss + ramp * (args.lambda_feat * feature_loss + args.lambda_kl * kl_loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += loss.item()
            totals["cls"] += classification.item()
            totals["feat"] += feature_loss.item()
            totals["kl"] += kl_loss.item()
        scheduler.step()
        validation = evaluate(model, val_loader, device)
        batches = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "train_loss": totals["loss"] / batches,
            "train_cls": totals["cls"] / batches,
            "train_feat": totals["feat"] / batches,
            "train_kl": totals["kl"] / batches,
            "ramp": ramp,
            "val_accuracy": validation["accuracy"],
            "val_loss": validation["loss"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)
        if swad_valley is not None:
            epoch_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            swad_valley.update(epoch, epoch_state, validation["loss"])
        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={record['train_loss']:.4f} "
            f"val={validation['accuracy']:.4f} ramp={ramp:.2f}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no source-validation checkpoint selected")
    if swad_valley is not None:
        best_state, swad_start_epoch, swad_end_epoch = swad_valley.final(best_state)
    model.load_state_dict(best_state)
    target_result, target_count = None, 0
    if not args.skip_target_eval:
        target_samples = collect_target_samples(Path(args.data_root), args.dataset, args.target, class_to_idx, excluded)
        target_ds = DomainImageDataset(target_samples, eval_tf)
        target_loader = DataLoader(target_ds, shuffle=False, **loader_args)
        target_result = evaluate(model, target_loader, device)
        target_count = len(target_ds)

    result = {
        "dataset": args.dataset,
        "target": args.target,
        "source_domains": source_domains,
        "method": args.method,
        "run_name": args.run_name,
        "model": "resnet50",
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
            "val_fraction": args.val_fraction,
            "lambda_feat": args.lambda_feat,
            "lambda_kl": args.lambda_kl,
            "temperature": args.temperature,
            "warmup_epochs": args.warmup_epochs,
            "augmentation_M": 9,
            "preprocessing": "legacy_imagenet",
            "exclusion_policy": exclusion_policy,
            "excluded_images": len(excluded),
            "mixstyle_p": args.mixstyle_p,
            "mixstyle_alpha": args.mixstyle_alpha,
            "swad_n_converge": 3,
            "swad_n_tolerance": 6,
            "swad_tolerance_ratio": 0.3,
            "swad_start_epoch": swad_start_epoch,
            "swad_end_epoch": swad_end_epoch,
        },
        "history": history,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.target}_resnet50_{args.run_name}_seed{args.seed}"
    result_path = output / f"{stem}.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    if args.save_checkpoint:
        torch.save(
            {"model": best_state, "args": vars(args), "result": {k: v for k, v in result.items() if k != "history"}},
            output / f"{stem}.pt",
        )
    csv_path = output / "results.csv"
    fields = [
        "dataset", "target", "method", "run_name", "model", "seed", "best_source_val",
        "target_accuracy", "target_loss", "n_train", "n_val", "n_test", "target_evaluations",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in fields})
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--method", required=True, choices=["erm", "strong_aug", "feature_plus_kl", "mixstyle", "swad"])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-feat", type=float, default=0.10)
    parser.add_argument("--lambda-kl", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--mixstyle-p", type=float, default=0.5)
    parser.add_argument("--mixstyle-alpha", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
