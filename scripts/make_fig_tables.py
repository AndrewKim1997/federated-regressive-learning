#!/usr/bin/env python3
"""
Read one or more CSV logs and produce simple figures and summary tables.
- Input: CSV files with columns [round, acc, ece, aggregator, ...]
- Output: figures/*.pdf and a summary CSV with final metrics per run.
Notes:
  1) Uses matplotlib only (no seaborn).
  2) Single-axes charts, no custom colors.
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_logs(patterns: List[str]) -> List[pd.DataFrame]:
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No CSV matched: {patterns}")
    return [pd.read_csv(f).assign(_src=os.path.basename(f)) for f in files]


def plot_metric_over_rounds(dfs: List[pd.DataFrame], metric: str, out_path: str) -> None:
    plt.figure()
    for df in dfs:
        x = df["round"].values
        y = df[metric].values
        label = f"{df['aggregator'].iloc[0]} | {df['_src'].iloc[0]}"
        plt.plot(x, y, label=label)
    plt.xlabel("Round")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Make figures and summary tables from CSV logs.")
    ap.add_argument("--glob", type=str, nargs="+", required=True, help="CSV glob(s), e.g., results/logs/*.csv")
    ap.add_argument("--outdir", type=str, default="results/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    dfs = load_logs(args.glob)

    # Plots
    plot_metric_over_rounds(dfs, metric="acc", out_path=os.path.join(args.outdir, "acc_over_rounds.pdf"))
    plot_metric_over_rounds(dfs, metric="ece", out_path=os.path.join(args.outdir, "ece_over_rounds.pdf"))

    # Summary table (final-epoch metrics per run)
    rows = []
    for df in dfs:
        df = df.sort_values("round")
        last = df.iloc[-1]
        rows.append({
            "src": df["_src"].iloc[0],
            "aggregator": last["aggregator"],
            "rounds": int(last["round"]),
            "acc_final": float(last["acc"]),
            "ece_final": float(last["ece"]),
        })
    summary = pd.DataFrame(rows).sort_values(["aggregator", "src"]).reset_index(drop=True)
    out_csv = os.path.join(args.outdir, "summary_final.csv")
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {os.path.join(args.outdir, 'acc_over_rounds.pdf')}")
    print(f"Saved: {os.path.join(args.outdir, 'ece_over_rounds.pdf')}")


if __name__ == "__main__":
    main()
