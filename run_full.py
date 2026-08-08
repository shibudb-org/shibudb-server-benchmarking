"""Full benchmark matrix runner.

Runs each suite at multiple data sizes (fresh DB per test: spaces are dropped
before and deleted after every config), writing one result file per
suite × size named `{suite}_{size}_{num_queries}.csv`, then charts and a
single combined report + zip bundle.

Defaults (all configurable):
  sizes        10000 100000 500000 1000000
  num-queries  5000
  concurrency  1 2 4 8 16
  suites       vector kv
  index types  all 12 (vector suite)
  WAL          both off and on (--no-both-wal to disable)

Examples:
  python run_full.py                                   # the full matrix
  python run_full.py --sizes 10000 100000 --suites vector
  python run_full.py --suites vector metadata kv --index-types Flat HNSW32
  python run_full.py --dry-run                         # print commands only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=HERE)


def run_logged(cmd: list[str], log_path: str, dry_run: bool) -> None:
    """Run a command, streaming output to console AND a log file.

    The log starts with the exact command and timestamps, so the full text
    report of every run (ingest/query progress + summary) is preserved and
    can be published alongside the CSV.
    """
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    start = time.time()
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(f"# started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = proc.wait()
        elapsed = time.time() - start
        log.write(f"\n# finished: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} "
                  f"(elapsed {elapsed:.0f}s, exit code {rc})\n")
    print(f"Log -> {log_path}", flush=True)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Matrix
    ap.add_argument("--suites", nargs="+", default=["vector", "kv"],
                    choices=["vector", "metadata", "kv"])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[10_000, 100_000, 500_000, 1_000_000],
                    help="data sizes: vectors ingested (vector/metadata) or KV keys (kv)")
    ap.add_argument("--num-queries", type=int, default=5_000)
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--index-types", nargs="+", default=None,
                    help="vector suite index types (default: all 12)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--kv-value-size", type=int, default=100)
    ap.add_argument("--ingest-concurrency", type=int, default=16)
    ap.add_argument("--both-wal", action=argparse.BooleanOptionalAction, default=True,
                    help="run each config with WAL off AND on (--no-both-wal: off only)")
    # Connection
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    # Server metadata (forwarded to benchmark.py)
    ap.add_argument("--server-version", default="")
    ap.add_argument("--server-commit", default="")
    ap.add_argument("--server-info-file", default="")
    # Output
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                    help="generate charts + combined report + zip at the end")
    ap.add_argument("--dry-run", action="store_true", help="print commands without running")
    return ap


def benchmark_cmd(args, suite: str, size: int, out_csv: str,
                  index_type: str | None = None) -> list[str]:
    cmd = [
        sys.executable, os.path.join(HERE, "benchmark.py"),
        "--suites", suite,
        "--concurrency", *map(str, args.concurrency),
        "--host", args.host, "--port", str(args.port),
        "--user", args.user, "--password", args.password,
        # Fresh DB per test: recreate spaces at start, delete them when done.
        "--drop-existing", "--cleanup",
        "--out", out_csv,
    ]
    if args.both_wal:
        cmd.append("--both-wal")
    if suite == "kv":
        cmd += ["--kv-keys", str(size), "--kv-value-size", str(args.kv_value_size)]
    else:
        cmd += ["--num-base", str(size), "--num-queries", str(args.num_queries),
                "--k", str(args.k), "--ingest-concurrency", str(args.ingest_concurrency)]
        if suite == "vector" and index_type:
            cmd += ["--index-types", index_type]
    if args.server_version:
        cmd += ["--server-version", args.server_version]
    if args.server_commit:
        cmd += ["--server-commit", args.server_commit]
    if args.server_info_file:
        cmd += ["--server-info-file", args.server_info_file]
    return cmd


def make_report(outputs: list[tuple[str, str]], args) -> None:
    """(Re)generate charts for the newest run and the combined report + zip
    from all runs completed so far."""
    charts_root = os.path.join(args.out_dir, "charts")
    stem, out_csv = outputs[-1]
    run([sys.executable, os.path.join(HERE, "plot.py"),
         "--csv", out_csv, "--out-dir", os.path.join(charts_root, stem)],
        args.dry_run)
    run([sys.executable, os.path.join(HERE, "report.py"),
         "--csv", *[c for _, c in outputs],
         "--charts-dir", charts_root,
         "--charts-subdir", *[s for s, _ in outputs],
         "--zip", os.path.join(args.out_dir, "benchmark_report.zip")],
        args.dry_run)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from bench_vector import DEFAULT_INDEX_TYPES
    index_types = args.index_types or DEFAULT_INDEX_TYPES

    # One run per test unit: vector runs are split per index type so each
    # index gets its own CSV/log ({suite}_{index}_{size}_{queries}).
    runs: list[tuple[str, int, str | None]] = []
    for size in args.sizes:
        for suite in args.suites:
            if suite == "vector":
                runs.extend((suite, size, idx) for idx in index_types)
            else:
                runs.append((suite, size, None))

    print(f"Matrix: {len(runs)} run(s) — sizes={args.sizes} suites={args.suites} "
          f"index_types={index_types} concurrency={args.concurrency} "
          f"num_queries={args.num_queries} both_wal={args.both_wal}", flush=True)

    outputs: list[tuple[str, str]] = []  # (stem, csv path)
    for i, (suite, size, index_type) in enumerate(runs, 1):
        if index_type:
            stem = f"{suite}_{index_type.replace(',', '-')}_{size}_{args.num_queries}"
        else:
            stem = f"{suite}_{size}_{args.num_queries}"
        out_csv = os.path.join(args.out_dir, f"{stem}.csv")
        log_path = os.path.join(args.out_dir, f"{stem}.log")
        print(f"\n=== [{i}/{len(runs)}] {stem} ===", flush=True)
        run_logged(benchmark_cmd(args, suite, size, out_csv, index_type),
                   log_path, args.dry_run)
        outputs.append((stem, out_csv))

        if args.report:
            # Publish incrementally: after each run, the report + zip already
            # cover everything completed so far.
            make_report(outputs, args)
            print(f"Report updated ({i}/{len(runs)} runs included): "
                  f"docs/BENCHMARKS.md + {args.out_dir}/benchmark_report.zip", flush=True)

    print(f"\nAll {len(outputs)} run(s) complete. Results in {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
