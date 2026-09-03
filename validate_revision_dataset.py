#!/usr/bin/env python3
"""Validate local VLCS or OfficeHome data before revision experiments.

This script is intentionally download-source agnostic. It accepts extracted
DomainBed-compatible data, validates its domains/classes/images, and writes a
deterministic inventory that can be compared after upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPECS = {
    "vlcs": {
        "display_name": "VLCS",
        "expected_images": 10729,
        "expected_classes": 5,
        "domains": {
            "Caltech101": ("Caltech101", "CALTECH", "Caltech", "C"),
            "LabelMe": ("LabelMe", "LABELME", "Labelme", "L"),
            "SUN09": ("SUN09", "SUN", "Sun", "S"),
            "VOC2007": ("VOC2007", "PASCAL", "Pascal", "VOC", "V"),
        },
        "class_names": {"bird", "car", "chair", "dog", "person"},
    },
    "officehome": {
        "display_name": "OfficeHome",
        "expected_images": 15588,
        "expected_classes": 65,
        "domains": {
            "Art": ("Art",),
            "Clipart": ("Clipart", "ClipArt"),
            "Product": ("Product",),
            "Real_World": ("Real_World", "Real World", "RealWorld"),
        },
        "class_names": None,
    },
}


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_directories(root: Path, max_depth: int = 4) -> list[Path]:
    root = root.resolve()
    candidates = [root]
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            candidates.append(path)
    return candidates


def resolve_domains(root: Path, aliases: dict[str, tuple[str, ...]]) -> tuple[Path, dict[str, Path]]:
    matches = []
    for parent in candidate_directories(root):
        children = {path.name.casefold(): path for path in parent.iterdir() if path.is_dir()}
        resolved = {}
        for canonical, options in aliases.items():
            path = next((children[name.casefold()] for name in options if name.casefold() in children), None)
            if path is None:
                break
            resolved[canonical] = path
        if len(resolved) == len(aliases):
            matches.append((parent, resolved))
    if not matches:
        expected = {name: list(options) for name, options in aliases.items()}
        raise RuntimeError(f"could not locate all domain directories under {root}; expected aliases: {expected}")
    matches.sort(key=lambda item: (len(item[0].parts), str(item[0])))
    return matches[0]


def load_declared_exclusions(path: Path | None, dataset: str) -> tuple[set[str], dict]:
    if path is None:
        return set(), {"policy_version": "none", "reason": "", "entries": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("datasets", {}).get(dataset, [])
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"duplicate exclusions declared for {dataset}")
    return set(paths), {
        "policy_version": payload["policy_version"],
        "reason": payload["reason"],
        "entries": entries,
    }


def validate(dataset: str, root: Path, verify_images: bool, exclusions_path: Path | None = None) -> dict:
    spec = SPECS[dataset]
    dataset_root, domain_dirs = resolve_domains(root, spec["domains"])
    excluded_paths, exclusion_policy = load_declared_exclusions(exclusions_path, dataset)
    domain_classes = {
        domain: {path.name for path in domain_root.iterdir() if path.is_dir()}
        for domain, domain_root in domain_dirs.items()
    }
    first_classes = next(iter(domain_classes.values()))
    for domain, classes in domain_classes.items():
        if classes != first_classes:
            missing = sorted(first_classes - classes)
            extra = sorted(classes - first_classes)
            raise RuntimeError(f"class mismatch in {domain}: missing={missing}, extra={extra}")
    if len(first_classes) != spec["expected_classes"]:
        raise RuntimeError(f"expected {spec['expected_classes']} shared classes, found {len(first_classes)}")
    if spec["class_names"] is not None and {name.casefold() for name in first_classes} != spec["class_names"]:
        raise RuntimeError(f"unexpected VLCS classes: {sorted(first_classes)}")

    inventory, invalid, discovered = [], [], set()
    counts = Counter()
    raw_counts = Counter()
    for domain, domain_root in sorted(domain_dirs.items()):
        for class_name in sorted(first_classes):
            for path in sorted((domain_root / class_name).rglob("*")):
                if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                    continue
                relative = path.relative_to(dataset_root).as_posix()
                discovered.add(relative)
                raw_counts[domain] += 1
                if verify_images:
                    try:
                        with Image.open(path) as image:
                            decoded = image.convert("RGB")
                            decoded.load()
                    except Exception as error:
                        invalid.append({"path": relative, "error": str(error)})
                if relative in excluded_paths:
                    continue
                inventory.append({
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "domain": domain,
                    "class": class_name,
                })
                counts[domain] += 1
    missing_exclusions = excluded_paths - discovered
    if missing_exclusions:
        raise RuntimeError(f"declared exclusions do not exist: {sorted(missing_exclusions)}")
    if verify_images:
        invalid_paths = {entry["path"] for entry in invalid}
        if invalid_paths != excluded_paths:
            raise RuntimeError(
                "full-decode failures differ from the frozen exclusion list; "
                f"undeclared={sorted(invalid_paths - excluded_paths)}, "
                f"declared_but_readable={sorted(excluded_paths - invalid_paths)}"
            )
    raw_image_count = len(inventory) + len(excluded_paths)
    if raw_image_count != spec["expected_images"]:
        raise RuntimeError(
            f"expected {spec['expected_images']} images for standard {spec['display_name']}, "
            f"found {raw_image_count}; per-domain={dict(raw_counts)}"
        )
    fingerprint_payload = {
        "dataset": spec["display_name"],
        "inventory": inventory,
        "exclusion_policy": exclusion_policy,
    }
    return {
        "dataset": spec["display_name"],
        "resolved_root": str(dataset_root),
        "raw_image_count": raw_image_count,
        "image_count": len(inventory),
        "excluded_image_count": len(excluded_paths),
        "exclusion_policy": exclusion_policy,
        "class_count": len(first_classes),
        "classes": sorted(first_classes),
        "domain_counts": dict(sorted(counts.items())),
        "dataset_fingerprint": canonical_json_hash(fingerprint_payload),
        "inventory": inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--exclusions", type=Path)
    args = parser.parse_args()
    result = validate(args.dataset, args.root, args.verify_images, args.exclusions)
    destination = args.inventory or Path(f"{args.dataset}_inventory.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(destination)
    summary = {key: value for key, value in result.items() if key != "inventory"}
    summary["inventory_file"] = str(destination.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
