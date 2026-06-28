"""Generate publication charts from a unified benchmark results CSV.

Understands the multi-suite schema written by benchmark.py (columns: suite,
operation, index_type, metric, wal, concurrency, scenario, selectivity,
throughput_ops_sec, recall_at_k, p50/p95/p99_ms, ...).

Produces (only for suites present in the CSV):
  vector_search:
    vector_recall_vs_qps.png, vector_qps_vs_concurrency.png,
    vector_latency_vs_concurrency.png, vector_ingest_throughput.png
  metadata_filter:
    metadata_recall_vs_qps.png, metadata_qps_vs_concurrency.png
  key_value:
    kv_throughput.png, kv_latency.png

Usage:
  python plot.py --csv results/run.csv --out-dir results/charts
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return 0.0


def wal_tag(row):
    return "WAL" if row.get("wal") in ("True", "true", "1") else "noWAL"


def _save(fig, out_dir, name):
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


# --------------------------------------------------------------------------- #
# Vector search
# --------------------------------------------------------------------------- #
def plot_vector(rows, out_dir):
    search = [r for r in rows if r["suite"] == "vector_search" and r["operation"] == "search"]
    inserts = [r for r in rows if r["suite"] == "vector_search" and r["operation"] == "insert"]
    if not search:
        return
    k = search[0]["k"]

    # recall vs QPS (one series per index_type+wal)
    fig, ax = plt.subplots(figsize=(9, 6))
    series = defaultdict(list)
    for r in search:
        series[f"{r['index_type']} ({wal_tag(r)})"].append(r)
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "throughput_ops_sec"))
        ax.plot([fnum(r, "throughput_ops_sec") for r in items],
                [fnum(r, "recall_at_k") for r in items], marker="o", label=label)
    ax.set_xlabel("Throughput (QPS)")
    ax.set_ylabel(f"Recall@{k}")
    ax.set_title("Vector search: recall vs QPS (up & right is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, "vector_recall_vs_qps.png")

    # QPS vs concurrency
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "concurrency"))
        ax.plot([fnum(r, "concurrency") for r in items],
                [fnum(r, "throughput_ops_sec") for r in items], marker="o", label=label)
    ax.set_xlabel("Concurrency (connections)")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Vector search: throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, "vector_qps_vs_concurrency.png")

    # latency vs concurrency (p99)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "concurrency"))
        ax.plot([fnum(r, "concurrency") for r in items],
                [fnum(r, "p99_ms") for r in items], marker="s", label=label)
    ax.set_xlabel("Concurrency (connections)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Vector search: p99 latency vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out_dir, "vector_latency_vs_concurrency.png")

    # ingest throughput bar
    if inserts:
        labels = [f"{r['index_type']}\n{wal_tag(r)}" for r in inserts]
        vals = [fnum(r, "throughput_ops_sec") for r in inserts]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.6), 6))
        ax.bar(range(len(labels)), vals)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Ingest throughput (vectors/sec)")
        ax.set_title("Vector ingest throughput per index type")
        ax.grid(True, axis="y", alpha=0.3)
        _save(fig, out_dir, "vector_ingest_throughput.png")


# --------------------------------------------------------------------------- #
# Metadata filter
# --------------------------------------------------------------------------- #
def plot_metadata(rows, out_dir):
    fs = [r for r in rows if r["suite"] == "metadata_filter" and r["operation"] == "filtered_search"]
    if not fs:
        return
    k = fs[0]["k"]
    series = defaultdict(list)
    for r in fs:
        series[f"{r['scenario']} ({wal_tag(r)})"].append(r)

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "throughput_ops_sec"))
        ax.plot([fnum(r, "throughput_ops_sec") for r in items],
                [fnum(r, "recall_at_k") for r in items], marker="o", label=label)
    ax.set_xlabel("Throughput (QPS)")
    ax.set_ylabel(f"Filtered recall@{k}")
    ax.set_title("Metadata-filtered search: recall vs QPS")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "metadata_recall_vs_qps.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "concurrency"))
        ax.plot([fnum(r, "concurrency") for r in items],
                [fnum(r, "throughput_ops_sec") for r in items], marker="o", label=label)
    ax.set_xlabel("Concurrency (connections)")
    ax.set_ylabel("Throughput (QPS)")
    ax.set_title("Metadata-filtered search: throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, out_dir, "metadata_qps_vs_concurrency.png")


# --------------------------------------------------------------------------- #
# Key-value
# --------------------------------------------------------------------------- #
def plot_kv(rows, out_dir):
    kv = [r for r in rows if r["suite"] == "key_value"]
    if not kv:
        return

    # throughput vs concurrency, one series per (operation, wal)
    fig, ax = plt.subplots(figsize=(9, 6))
    series = defaultdict(list)
    for r in kv:
        series[f"{r['operation']} ({wal_tag(r)})"].append(r)
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "concurrency"))
        ax.plot([fnum(r, "concurrency") for r in items],
                [fnum(r, "throughput_ops_sec") for r in items], marker="o", label=label)
    ax.set_xlabel("Concurrency (connections)")
    ax.set_ylabel("Throughput (ops/sec)")
    ax.set_title("Key-value: throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "kv_throughput.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, items in sorted(series.items()):
        items = sorted(items, key=lambda r: fnum(r, "concurrency"))
        ax.plot([fnum(r, "concurrency") for r in items],
                [fnum(r, "p99_ms") for r in items], marker="s", label=label)
    ax.set_xlabel("Concurrency (connections)")
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Key-value: p99 latency vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, out_dir, "kv_latency.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/run.csv")
    ap.add_argument("--out-dir", default="results/charts")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"no rows in {args.csv}")
    os.makedirs(args.out_dir, exist_ok=True)

    plot_vector(rows, args.out_dir)
    plot_metadata(rows, args.out_dir)
    plot_kv(rows, args.out_dir)


if __name__ == "__main__":
    main()
