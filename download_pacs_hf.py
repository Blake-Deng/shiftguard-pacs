#!/usr/bin/env python3
"""Download PACS from Hugging Face and export it to the folder layout used by this repo.

Usage:
    python download_pacs_hf.py --output data/PACS
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='data/PACS')
    ap.add_argument('--dataset', default='flwrlabs/pacs')
    args = ap.parse_args()
    ds = load_dataset(args.dataset)
    out = Path(args.output)
    label_feature = ds[next(iter(ds))].features['label']
    label_names = getattr(label_feature, 'names', None)
    count = 0
    for split, part in ds.items():
        for idx, row in enumerate(part):
            domain = str(row['domain']).replace(' ', '_')
            label_value = row['label']
            label = label_names[label_value] if label_names else str(label_value)
            image = row['image'].convert('RGB')
            target = out / domain / label
            target.mkdir(parents=True, exist_ok=True)
            image.save(target / f'{split}_{idx:06d}.jpg', quality=95)
            count += 1
    print(f'Exported {count} images to {out}')


if __name__ == '__main__':
    main()
