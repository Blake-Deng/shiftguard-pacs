#!/usr/bin/env python3
"""PACS leave-one-domain-out experiments for ShiftGuard.

Examples:
  python shiftguard.py --data-root data/PACS --target Sketch --method erm
  python shiftguard.py --data-root data/PACS --target Sketch --method shiftguard --epochs 30
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import timm


DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PACSDataset(Dataset):
    def __init__(self, samples, class_to_idx, weak, strong=None):
        self.samples, self.class_to_idx = samples, class_to_idx
        self.weak, self.strong = weak, strong

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            image = im.convert("RGB")
        weak = self.weak(image)
        strong = self.strong(image) if self.strong is not None else weak
        return weak, strong, label


def collect_samples(root: Path, target: str, seed: int, val_fraction: float):
    if target not in DOMAINS:
        raise ValueError(f"target must be one of {DOMAINS}, got {target}")
    domain_dirs = {d: root / d for d in DOMAINS}
    # Accept common lowercase/spaced spellings.
    for d in DOMAINS:
        if not domain_dirs[d].is_dir():
            alternatives = [root / d.lower(), root / d.replace("_", " "), root / d.replace("_", "-")]
            hit = next((p for p in alternatives if p.is_dir()), None)
            if hit is not None:
                domain_dirs[d] = hit
    missing = [d for d, p in domain_dirs.items() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing PACS domain directories under {root}: {missing}")
    classes = sorted({p.name for d in DOMAINS for p in domain_dirs[d].iterdir() if p.is_dir()})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    source_train, source_val, target_test = [], [], []
    rng = random.Random(seed)
    for domain, dpath in domain_dirs.items():
        for cls in classes:
            files = [p for p in (dpath / cls).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            if not files:
                continue
            items = [(str(p), class_to_idx[cls]) for p in sorted(files)]
            if domain == target:
                target_test.extend(items)
            else:
                rng.shuffle(items)
                nval = max(1, int(round(len(items) * val_fraction))) if len(items) > 1 else 0
                source_val.extend(items[:nval])
                source_train.extend(items[nval:])
    if not source_train or not target_test:
        raise RuntimeError(f"Found {len(source_train)} source-train and {len(target_test)} target images; check layout.")
    return source_train, source_val, target_test, class_to_idx


def make_transforms(image_size: int):
    weak = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    strong = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1), transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return weak, strong, eval_tf


def build_model(name: str, num_classes: int, pretrained: bool):
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    if name == "vit-small":
        net = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)
        if pretrained:
            from timm.models._builder import load_pretrained
            cfg = timm.get_pretrained_cfg("vit_small_patch16_224").to_dict()
            cfg["file"] = str(Path(__file__).resolve().parent / "weights" / "vit_small_patch16_224.npz")
            cfg["url"] = ""; cfg["hf_hub_id"] = ""
            load_pretrained(net, cfg, num_classes=num_classes)
        return net
    if name == "vit_b16":
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
        return net
    raise ValueError(f"unknown model {name}")


def forward_features(model, x):
    if hasattr(model, "forward_features") and model.__class__.__module__.startswith("timm"):
        z = model.forward_features(x)
        if z.ndim == 3: z = z[:, 0]
        return z, model.get_classifier()(z)
    if isinstance(model, models.ResNet):
        y = model.conv1(x); y = model.bn1(y); y = model.relu(y); y = model.maxpool(y)
        y = model.layer1(y); y = model.layer2(y); y = model.layer3(y); y = model.layer4(y)
        z = model.avgpool(y).flatten(1)
        return z, model.fc(z)
    if isinstance(model, models.VisionTransformer):
        z = model._process_input(x); n = z.shape[0]
        cls = model.class_token.expand(n, -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = model.encoder(z)[:, 0]
        return z, model.heads(z)
    raise TypeError(type(model))


def symmetric_kl(logits_a, logits_b):
    logpa, logpb = logits_a.log_softmax(1), logits_b.log_softmax(1)
    pa, pb = logpa.exp(), logpb.exp()
    return 0.5 * ((pa * (logpa - logpb)).sum(1) + (pb * (logpb - logpa)).sum(1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); correct = total = 0; loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for weak, _, y in loader:
        weak, y = weak.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(weak); loss_sum += ce(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return {"accuracy": correct / total, "loss": loss_sum / total}


def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    weak, strong, eval_tf = make_transforms(args.image_size)
    train_s, val_s, test_s, class_to_idx = collect_samples(Path(args.data_root), args.target, args.seed, args.val_fraction)
    train_ds = PACSDataset(train_s, class_to_idx, weak, strong)
    val_ds = PACSDataset(val_s, class_to_idx, eval_tf, eval_tf)
    test_ds = PACSDataset(test_s, class_to_idx, eval_tf, eval_tf)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=(device.type == "cuda"), persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
    model = build_model(args.model, len(class_to_idx), not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()
    best_val, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for xw, xs, y in train_loader:
            xw, xs, y = xw.to(device, non_blocking=True), xs.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                zw, pw = forward_features(model, xw); zs, ps = forward_features(model, xs)
                if args.method == "erm":
                    loss = ce(pw, y)
                else:
                    lcls = 0.5 * (ce(pw, y) + ce(ps, y))
                    d = 1.0 - nn.functional.cosine_similarity(zw, zs, dim=1)
                    lfeat = d
                    lkl = symmetric_kl(pw, ps)
                    if args.method == "aug":
                        loss = lcls
                    else:
                        if args.method == "feat":
                            loss = lcls + args.lambda_feat * lfeat.mean()
                        elif args.method == "feat_kl":
                            loss = lcls + args.lambda_feat * lfeat.mean() + args.lambda_kl * lkl.mean()
                        elif args.method == "shiftguard":
                            w = (d.detach() / (d.detach().mean() + 1e-6)).clamp(0.5, 2.0)
                            loss = lcls + args.lambda_feat * (w * lfeat).mean() + args.lambda_kl * (w * lkl).mean()
                        else:
                            raise ValueError(args.method)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); running += loss.item()
        scheduler.step()
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch, "train_loss": running / max(1, len(train_loader)), "val_accuracy": val["accuracy"], "lr": scheduler.get_last_lr()[0]}
        history.append(rec)
        if val["accuracy"] > best_val:
            best_val, best_state = val["accuracy"], {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:03d}/{args.epochs} loss={rec['train_loss']:.4f} val={val['accuracy']:.4f}", flush=True)
    if best_state is not None: model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    result = {"target": args.target, "method": args.method, "model": args.model, "seed": args.seed, "best_source_val": best_val, "target_accuracy": test["accuracy"], "target_loss": test["loss"], "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds), "history": history}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.target}_{args.model}_{args.method}_seed{args.seed}"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "args": vars(args), "result": {k: result[k] for k in result if k != "history"}}, out / f"{stem}.pt")
    with (out / "results.csv").open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "method", "model", "seed", "best_source_val", "target_accuracy", "target_loss", "n_train", "n_val", "n_test"])
        if f.tell() == 0: writer.writeheader()
        writer.writerow({k: result[k] for k in writer.fieldnames})
    print(json.dumps({k: result[k] for k in result if k != "history"}, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--target", required=True, choices=DOMAINS)
    p.add_argument("--method", choices=["erm", "aug", "feat", "feat_kl", "shiftguard"], default="shiftguard")
    p.add_argument("--model", choices=["resnet50", "vit_b16", "vit-small"], default="resnet50")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64); p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--image-size", type=int, default=224); p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-feat", type=float, default=0.5); p.add_argument("--lambda-kl", type=float, default=0.5)
    p.add_argument("--output", default="runs"); p.add_argument("--device", default=None); p.add_argument("--no-pretrained", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
