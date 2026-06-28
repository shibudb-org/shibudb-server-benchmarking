"""ShibuDB full-system benchmark orchestrator.

Runs one or more benchmark suites over TCP through the official `shibudb-client`
SDK and writes a single unified results CSV + JSON (with captured environment).

Suites:
  vector    : every vector index type x {WAL on/off} x concurrency
              -> ingest throughput, recall@k, QPS, p50/p95/p99
  metadata  : Flat + indexed metadata fields, filtered search x {WAL on/off}
              x filter-scenario x concurrency -> filtered recall, QPS, latency
  kv        : key-value PUT/GET/DELETE x {WAL on/off} x concurrency
              -> ops/sec, latency

Examples:
  # Everything, with and without WAL (quick local subset):
  python benchmark.py --suites all --both-wal \
      --num-base 20000 --num-queries 1000 --kv-keys 20000 \
      --concurrency 1 8 32 --out results/local_all.csv

  # Full vector sweep on a dedicated server:
  python benchmark.py --suites vector --both-wal \
      --num-base 1000000 --num-queries 10000 \
      --concurrency 1 8 32 64 128 --out results/sift1m_vector.csv
"""

from __future__ import annotations

import argparse
import json
import os

import bench_kv
import bench_metadata
import bench_vector
from common import ResourceMonitor, ResultWriter, capture_env, find_server_process
from dataset import load_sift


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Connection
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--timeout", type=int, default=120)
    # Suite selection
    ap.add_argument("--suites", nargs="+", default=["all"],
                    choices=["all", "vector", "metadata", "kv"])
    # Dataset
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.shibudb-benchmarks/data"))
    ap.add_argument("--num-base", type=int, default=100_000, help="vectors to ingest (vector/metadata suites)")
    ap.add_argument("--num-queries", type=int, default=1_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dimension", type=int, default=128)
    # Vector index sweep
    ap.add_argument("--index-types", nargs="+", default=bench_vector.DEFAULT_INDEX_TYPES)
    ap.add_argument("--metric", default="L2",
                    help="server metric name: L2 | InnerProduct | L1 | Lp | Canberra | BrayCurtis | JensenShannon | Linf")
    # WAL
    ap.add_argument("--both-wal", action="store_true", help="run each config with WAL off AND on")
    ap.add_argument("--enable-wal", action="store_true", help="if not --both-wal, run with WAL on (default off)")
    # Key-value sizing
    ap.add_argument("--kv-keys", type=int, default=100_000)
    ap.add_argument("--kv-value-size", type=int, default=100, help="value size in bytes")
    # Concurrency
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 8, 32, 64])
    ap.add_argument("--ingest-concurrency", type=int, default=16)
    # Lifecycle
    ap.add_argument("--settle-timeout", type=int, default=600)
    ap.add_argument("--drop-existing", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="delete benchmark spaces after each config")
    ap.add_argument("--space-prefix", default="bench")
    # Server resource monitoring (CPU / memory) -- requires psutil; same-host
    ap.add_argument("--monitor", dest="monitor", action="store_true", default=True,
                    help="sample server CPU/memory during the run (on by default)")
    ap.add_argument("--no-monitor", dest="monitor", action="store_false",
                    help="disable server CPU/memory sampling")
    ap.add_argument("--server-pid", type=int, default=None,
                    help="PID of the ShibuDB server process to monitor (else auto-detect by name)")
    ap.add_argument("--server-name", default="shibudb",
                    help="substring used to auto-detect the server process")
    ap.add_argument("--monitor-interval", type=float, default=0.5,
                    help="seconds between server CPU/memory samples")
    # Output
    ap.add_argument("--out", default="results/run.csv")
    return ap


def main():
    args = build_parser().parse_args()
    suites = set(args.suites)
    if "all" in suites:
        suites = {"vector", "metadata", "kv"}

    env = capture_env(args)
    print(json.dumps(env, indent=2), flush=True)
    writer = ResultWriter(args.out, env)

    need_dataset = bool({"vector", "metadata"} & suites)
    base = queries = sift_gt = None
    if need_dataset:
        base, queries, sift_gt = load_sift(args.data_dir)

    monitor = None
    if args.monitor:
        proc = find_server_process(args.server_pid, args.server_name)
        if proc is None:
            print("[monitor] no server process found on this host (install psutil and run the "
                  "client on the server host, or pass --server-pid); CPU/memory will not be recorded",
                  flush=True)
        else:
            print(f"[monitor] sampling server pid={proc.pid} every {args.monitor_interval}s", flush=True)
            monitor = ResourceMonitor(proc, interval=args.monitor_interval).start()

    res_path = None
    try:
        if "vector" in suites:
            bench_vector.run(args, base, queries, sift_gt, writer)
        if "metadata" in suites:
            bench_metadata.run(args, base, queries, writer)
        if "kv" in suites:
            bench_kv.run(args, writer)
    finally:
        if monitor is not None:
            monitor.stop()
            res_path = os.path.splitext(args.out)[0] + ".resources.csv"
            monitor.save(res_path)
            s = monitor.summary()
            if s:
                print(f"[monitor] server CPU mean={s['cpu_percent_mean']:.1f}% peak={s['cpu_percent_peak']:.1f}% "
                      f"| RSS mean={s['rss_mb_mean']:.0f}MB peak={s['rss_mb_peak']:.0f}MB "
                      f"-> {res_path}", flush=True)

    print("\n=== Summary ===", flush=True)
    for r in writer.rows:
        recall = f" recall@{r.k}={r.recall_at_k:.4f}" if r.recall_at_k >= 0 else ""
        scenario = f" [{r.scenario}]" if r.scenario else ""
        print(f"{r.suite:16s} {r.operation:16s} {r.index_type:14s} wal={int(r.wal)} "
              f"c={r.concurrency:<4d}{scenario} {r.throughput_ops_sec:9.1f} ops/s "
              f"p99={r.p99_ms:.2f}ms{recall}", flush=True)
    print(f"\nWrote {args.out} and {os.path.splitext(args.out)[0] + '.json'}", flush=True)
    if res_path:
        print(f"Wrote {res_path} and {os.path.splitext(res_path)[0] + '.json'} (server CPU/memory)", flush=True)


if __name__ == "__main__":
    main()
