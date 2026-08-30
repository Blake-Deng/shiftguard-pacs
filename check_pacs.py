#!/usr/bin/env python3
"""Validate a PACS extraction before launching a multi-day run."""
import argparse
from pathlib import Path

DOMAINS = ("Photo", "Art_Painting", "Cartoon", "Sketch")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def resolve(root, domain):
    for name in (domain, domain.lower(), domain.replace("_", " "), domain.replace("_", "-")):
        p = root / name
        if p.is_dir(): return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("data_root"); args = ap.parse_args()
    root = Path(args.data_root)
    total = 0; ok = True
    for domain in DOMAINS:
        path = resolve(root, domain)
        if path is None:
            print(f"MISSING {domain}"); ok = False; continue
        classes = sorted(p.name for p in path.iterdir() if p.is_dir())
        count = sum(1 for p in path.rglob("*") if p.suffix.lower() in EXTS)
        total += count
        print(f"{domain:13s} {count:6d} images, {len(classes)} classes: {', '.join(classes)}")
        if not classes or not count: ok = False
    print(f"total: {total} images")
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__": main()
