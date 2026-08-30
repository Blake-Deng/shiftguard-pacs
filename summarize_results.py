#!/usr/bin/env python3
"""Print a compact PACS table from runs/results.csv."""
import argparse
import csv
from collections import defaultdict
import math

def main():
    p = argparse.ArgumentParser(); p.add_argument("csv", nargs="?", default="runs/results.csv"); args = p.parse_args()
    rows = list(csv.DictReader(open(args.csv, newline="")))
    grouped = defaultdict(list)
    for r in rows: grouped[(r["target"], r["model"], r["method"])].append(float(r["target_accuracy"]))
    print("target\tmodel\tmethod\tn\taccuracy_mean\taccuracy_std")
    for key in sorted(grouped):
        vals = grouped[key]; mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        print("%s\t%s\t%s\t%d\t%.4f\t%.4f" % (*key, len(vals), mean, std))

if __name__ == "__main__": main()
