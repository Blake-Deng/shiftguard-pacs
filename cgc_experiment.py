#!/usr/bin/env python3
"""Target-isolated experiments for component-wise conflict-gated consistency."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shiftguard_corrected import build_model, forward_features, make_transforms
from shiftguard_multidataset import (
    DATASETS,
    DomainImageDataset,
    collect_source_samples,
    collect_target_samples,
    evaluate,
    exclusion_paths,
    resolve_domains,
    shared_classes,
)

METHODS = (
    "strong_aug",
    "feature_plus_kl",
    "cgc",
    "mean_teacher",
    "cgc_feature_gate",
    "cgc_kl_gate",
    "cgc_combined",
    "cgc_random",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_full_domains(
    root: Path,
    dataset: str,
    domains: list[str],
    class_to_idx: dict[str, int],
    domain_to_idx: dict[str, int],
    excluded: frozenset[str],
) -> list[tuple[str, int, int]]:
    domain_dirs = resolve_domains(root, dataset)
    samples = []
    for domain in domains:
        for class_name, label in class_to_idx.items():
            for path in sorted((domain_dirs[domain] / class_name).rglob("*")):
                if (
                    path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    and path.relative_to(root).as_posix() not in excluded
                ):
                    samples.append((str(path), label, domain_to_idx[domain]))
    if not samples:
        raise RuntimeError(f"no samples found for {dataset} domains {domains}")
    return samples


def nested_source_split(args, excluded: frozenset[str]):
    if not args.train_domains or not args.validation_domain:
        raise ValueError("nested screening requires --train-domains and --validation-domain")
    train_domains = args.train_domains.split(",")
    known = DATASETS[args.dataset]["domains"]
    if args.target in train_domains or args.validation_domain == args.target:
        raise ValueError("outer target entered nested source split")
    if args.validation_domain in train_domains:
        raise ValueError("inner validation domain entered inner training domains")
    if any(domain not in known for domain in train_domains + [args.validation_domain]):
        raise ValueError("unknown nested source domain")
    expected = set(known) - {args.target, args.validation_domain}
    if set(train_domains) != expected:
        raise ValueError(f"inner train domains must be exactly {sorted(expected)}")
    domain_dirs = resolve_domains(Path(args.data_root), args.dataset)
    visible_dirs = {domain: domain_dirs[domain] for domain in train_domains + [args.validation_domain]}
    classes = shared_classes(visible_dirs)
    if len(classes) != DATASETS[args.dataset]["classes"]:
        raise RuntimeError(f"expected {DATASETS[args.dataset]['classes']} classes, found {len(classes)}")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    domain_to_idx = {name: index for index, name in enumerate(train_domains)}
    domain_to_idx[args.validation_domain] = -1
    train = collect_full_domains(
        Path(args.data_root), args.dataset, train_domains, class_to_idx, domain_to_idx, excluded
    )
    validation = collect_full_domains(
        Path(args.data_root), args.dataset, [args.validation_domain], class_to_idx, domain_to_idx, excluded
    )
    return train, validation, class_to_idx, train_domains


def ema_update(teacher, student, alpha: float) -> None:
    with torch.no_grad():
        for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
            teacher_parameter.mul_(alpha).add_(student_parameter, alpha=1.0 - alpha)
        for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
            if teacher_buffer.is_floating_point():
                teacher_buffer.mul_(alpha).add_(student_buffer, alpha=1.0 - alpha)
            else:
                teacher_buffer.copy_(student_buffer)


def component_terms(weak_features, strong_features, weak_logits, strong_logits, labels, temperature):
    weak_features_f = weak_features.detach().float()
    strong_features_f = strong_features.float()
    weak_logits_f = weak_logits.float()
    strong_logits_f = strong_logits.float()
    ce_w = F.cross_entropy(weak_logits_f, labels, reduction="none")
    ce_s = F.cross_entropy(strong_logits_f, labels, reduction="none")
    feature = 1.0 - F.cosine_similarity(weak_features_f, strong_features_f, dim=1)
    teacher = F.softmax(weak_logits_f.detach() / temperature, dim=1)
    student = F.log_softmax(strong_logits_f / temperature, dim=1)
    kl = F.kl_div(student, teacher, reduction="none").sum(1) * (temperature ** 2)
    return 0.5 * (ce_w + ce_s), ce_s, feature, kl


def conflict_masks(ce_s, feature, kl, strong_features, args):
    g_ce = torch.autograd.grad(ce_s.sum(), strong_features, retain_graph=True, create_graph=False)[0]
    g_feature = torch.autograd.grad(feature.sum(), strong_features, retain_graph=True, create_graph=False)[0]
    g_kl = torch.autograd.grad(kl.sum(), strong_features, retain_graph=True, create_graph=False)[0]
    alignment_feature = F.cosine_similarity(g_ce.float(), g_feature.float(), dim=1, eps=1e-8).detach()
    alignment_kl = F.cosine_similarity(g_ce.float(), g_kl.float(), dim=1, eps=1e-8).detach()
    mask_feature = (alignment_feature >= 0).to(feature.dtype).detach()
    mask_kl = (alignment_kl >= 0).to(kl.dtype).detach()

    if args.method == "cgc_feature_gate":
        mask_kl = torch.ones_like(mask_kl)
    elif args.method == "cgc_kl_gate":
        mask_feature = torch.ones_like(mask_feature)
    elif args.method == "cgc_combined":
        combined = args.lambda_feat * feature + args.lambda_kl * kl
        g_combined = torch.autograd.grad(
            combined.sum(), strong_features, retain_graph=True, create_graph=False
        )[0]
        combined_alignment = F.cosine_similarity(
            g_ce.float(), g_combined.float(), dim=1, eps=1e-8
        ).detach()
        combined_mask = (combined_alignment >= 0).to(feature.dtype).detach()
        mask_feature = combined_mask
        mask_kl = combined_mask.to(kl.dtype)
    elif args.method == "cgc_random":
        mask_feature = mask_feature[torch.randperm(mask_feature.numel(), device=mask_feature.device)]
        mask_kl = mask_kl[torch.randperm(mask_kl.numel(), device=mask_kl.device)]

    return mask_feature, mask_kl, alignment_feature, alignment_kl


def median_or_nan(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def run(args) -> None:
    if args.target not in DATASETS[args.dataset]["domains"]:
        raise ValueError(f"invalid target {args.target} for {args.dataset}")
    if args.method not in METHODS:
        raise ValueError(args.method)
    if args.ema_alpha <= 0 or args.ema_alpha >= 1:
        raise ValueError("EMA alpha must be in (0, 1)")
    seed_everything(args.seed)
    started = time.time()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda_index = None
    if device.type == "cuda":
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)

    excluded, exclusion_policy = exclusion_paths(Path(args.exclusions), args.dataset)
    weak_tf, strong_tf, eval_tf = make_transforms(
        args.image_size, args.preprocessing, args.augmentation_m
    )
    if args.selection_protocol == "strict_nested_inner":
        train_samples, val_samples, class_to_idx, source_domains = nested_source_split(args, excluded)
    else:
        train_samples, val_samples, class_to_idx, source_domains = collect_source_samples(
            Path(args.data_root), args.dataset, args.target, args.seed, args.val_fraction, excluded
        )

    train_ds = DomainImageDataset(train_samples, weak_tf, strong_tf)
    val_ds = DomainImageDataset(val_samples, eval_tf)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model = build_model(args.model, len(class_to_idx), not args.no_pretrained).to(device)
    teacher = None
    if args.method == "mean_teacher":
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_val, best_state, best_epoch = -1.0, None, None
    history = []

    gated_methods = {"cgc", "cgc_feature_gate", "cgc_kl_gate", "cgc_combined", "cgc_random"}
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        model.train()
        if teacher is not None:
            teacher.eval()
        totals = {"loss": 0.0, "cls": 0.0, "feat": 0.0, "kl": 0.0}
        feature_alignments: list[float] = []
        kl_alignments: list[float] = []
        per_domain = {
            domain: {"feature_conflicts": 0, "kl_conflicts": 0, "count": 0}
            for domain in source_domains
        }
        feature_kept = kl_kept = diagnostic_count = 0
        ramp = 1.0 if args.warmup_epochs <= 0 else min(1.0, epoch / args.warmup_epochs)

        for weak, strong, labels, domain_ids in train_loader:
            weak = weak.to(device, non_blocking=True)
            strong = strong.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                weak_features, weak_logits = forward_features(model, weak)
                strong_features, strong_logits = forward_features(model, strong)
            ce, ce_s, feature, kl = component_terms(
                weak_features, strong_features, weak_logits, strong_logits, labels, args.temperature
            )
            mask_feature = torch.ones_like(feature)
            mask_kl = torch.ones_like(kl)

            if args.method in gated_methods:
                mask_feature, mask_kl, alignment_feature, alignment_kl = conflict_masks(
                    ce_s, feature, kl, strong_features, args
                )
                feature_cpu = alignment_feature.float().cpu().tolist()
                kl_cpu = alignment_kl.float().cpu().tolist()
                feature_alignments.extend(feature_cpu)
                kl_alignments.extend(kl_cpu)
                feature_kept += int(mask_feature.sum().item())
                kl_kept += int(mask_kl.sum().item())
                diagnostic_count += labels.numel()
                domain_cpu = domain_ids.tolist()
                for index, domain_id in enumerate(domain_cpu):
                    if 0 <= domain_id < len(source_domains):
                        stats = per_domain[source_domains[domain_id]]
                        stats["count"] += 1
                        stats["feature_conflicts"] += int(feature_cpu[index] < 0)
                        stats["kl_conflicts"] += int(kl_cpu[index] < 0)

            if args.method == "strong_aug":
                loss_per_sample = ce
            elif args.method == "feature_plus_kl":
                loss_per_sample = ce + ramp * (args.lambda_feat * feature + args.lambda_kl * kl)
            elif args.method in gated_methods:
                loss_per_sample = ce + ramp * (
                    args.lambda_feat * mask_feature * feature + args.lambda_kl * mask_kl * kl
                )
            elif args.method == "mean_teacher":
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    teacher_logits = teacher(weak)
                teacher_prob = F.softmax(teacher_logits.float() / args.temperature, dim=1)
                student_log_prob = F.log_softmax(strong_logits.float() / args.temperature, dim=1)
                teacher_kl = F.kl_div(
                    student_log_prob, teacher_prob, reduction="none"
                ).sum(1) * (args.temperature ** 2)
                loss_per_sample = ce + ramp * args.lambda_kl * teacher_kl
                kl = teacher_kl
            else:
                raise ValueError(args.method)

            loss = loss_per_sample.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if teacher is not None:
                ema_update(teacher, model, args.ema_alpha)
            totals["loss"] += loss.item()
            totals["cls"] += ce.mean().item()
            totals["feat"] += feature.mean().item()
            totals["kl"] += kl.mean().item()

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
            "source_val_accuracy": validation["accuracy"],
            "source_val_loss": validation["loss"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - epoch_started,
        }
        if diagnostic_count:
            record.update({
                "mean_a_feat": statistics.fmean(feature_alignments),
                "median_a_feat": median_or_nan(feature_alignments),
                "mean_a_kl": statistics.fmean(kl_alignments),
                "median_a_kl": median_or_nan(kl_alignments),
                "feature_conflict_rate": sum(value < 0 for value in feature_alignments) / diagnostic_count,
                "kl_conflict_rate": sum(value < 0 for value in kl_alignments) / diagnostic_count,
                "feature_keep_rate": feature_kept / diagnostic_count,
                "kl_keep_rate": kl_kept / diagnostic_count,
                "source_domain_conflict_rates": {
                    domain: {
                        "feature": values["feature_conflicts"] / values["count"],
                        "kl": values["kl_conflicts"] / values["count"],
                        "count": values["count"],
                    }
                    for domain, values in per_domain.items() if values["count"]
                },
            })
        history.append(record)
        if validation["accuracy"] > best_val:
            best_val = validation["accuracy"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        diagnostic_text = ""
        if diagnostic_count:
            diagnostic_text = (
                f" keep_f={record['feature_keep_rate']:.3f}"
                f" keep_k={record['kl_keep_rate']:.3f}"
            )
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={record['train_loss']:.4f} "
            f"val={validation['accuracy']:.4f}{diagnostic_text}",
            flush=True,
        )

    if best_state is None:
        raise RuntimeError("no source-validation checkpoint selected")
    model.load_state_dict(best_state)
    target_result, target_count = None, 0
    if not args.skip_target_eval:
        target_samples = collect_target_samples(
            Path(args.data_root), args.dataset, args.target, class_to_idx, excluded
        )
        target_ds = DomainImageDataset(target_samples, eval_tf)
        target_loader = DataLoader(target_ds, shuffle=False, **loader_args)
        target_result = evaluate(model, target_loader, device)
        target_count = len(target_ds)

    peak_memory = torch.cuda.max_memory_allocated(cuda_index) if device.type == "cuda" else 0
    result = {
        "protocol_version": "cgc-v2-zero-threshold-2026-09-04",
        "dataset": args.dataset,
        "target": args.target,
        "source_domains": source_domains,
        "inner_validation_domain": args.validation_domain,
        "selection_protocol": args.selection_protocol,
        "method": args.method,
        "run_name": args.run_name,
        "model": args.model,
        "seed": args.seed,
        "best_source_val": best_val,
        "checkpoint_epoch": best_epoch,
        "target_accuracy": None if target_result is None else target_result["accuracy"],
        "target_loss": None if target_result is None else target_result["loss"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": target_count,
        "target_evaluations": 0 if target_result is None else 1,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": peak_memory,
        "config": {
            "val_fraction": args.val_fraction,
            "lambda_feat": args.lambda_feat,
            "lambda_kl": args.lambda_kl,
            "temperature": args.temperature,
            "warmup_epochs": args.warmup_epochs,
            "augmentation_N": 2,
            "augmentation_M": args.augmentation_m,
            "gradient_gate_threshold": 0.0,
            "gradient_reference": "strong_view_supervised_ce_wrt_strong_features",
            "component_masks_detached": True,
            "ema_alpha": args.ema_alpha,
            "preprocessing": args.preprocessing,
            "exclusion_policy": exclusion_policy,
            "excluded_images": len(excluded),
        },
        "history": history,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.target}_{args.model}_{args.run_name}_seed{args.seed}"
    result_path = output / f"{stem}.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(result_path)
    if args.save_checkpoint:
        torch.save(
            {"model": best_state, "args": vars(args), "result": {k: v for k, v in result.items() if k != "history"}},
            output / f"{stem}.pt",
        )
    if args.diagnostics_output and args.method in gated_methods:
        diagnostics = Path(args.diagnostics_output)
        diagnostics.mkdir(parents=True, exist_ok=True)
        diagnostics_path = diagnostics / f"{args.dataset}_{args.target}_{args.model}_{args.run_name}_seed{args.seed}.json"
        diagnostics_path.write_text(json.dumps({
            "dataset": args.dataset,
            "target": args.target,
            "seed": args.seed,
            "method": args.method,
            "model": args.model,
            "history": history,
        }, indent=2, allow_nan=False), encoding="utf-8")
    fields = [
        "dataset", "target", "method", "run_name", "model", "seed", "best_source_val",
        "checkpoint_epoch", "target_accuracy", "target_loss", "n_train", "n_val", "n_test",
        "target_evaluations", "elapsed_seconds", "peak_gpu_memory_bytes", "selection_protocol",
        "inner_validation_domain",
    ]
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: result[key] for key in fields})
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=["resnet50", "vit-small"], default="resnet50")
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
    parser.add_argument("--augmentation-m", type=int, default=9)
    parser.add_argument("--ema-alpha", type=float, default=0.999)
    parser.add_argument("--preprocessing", choices=["legacy_imagenet", "timm_vit_standard"], default="legacy_imagenet")
    parser.add_argument("--selection-protocol", choices=["fixed_source_validation", "strict_nested_inner"], default="fixed_source_validation")
    parser.add_argument("--train-domains", default=None)
    parser.add_argument("--validation-domain", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--diagnostics-output", default=None)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--skip-target-eval", action="store_true")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
