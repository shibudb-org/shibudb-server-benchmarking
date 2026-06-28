"""Vector similarity-search benchmark suite.

Sweeps every requested vector index type x {WAL on, WAL off} x concurrency.
For each combination it measures:
  - ingest throughput (vectors/sec) and ingest latency percentiles
  - query throughput (QPS), recall@k (vs exact ground-truth), and p50/p95/p99

Ground-truth: uses the canonical SIFT ground-truth when the full 1M base is
ingested with the L2 metric; otherwise computes exact top-k with numpy over the
ingested subset (so recall stays valid for any --num-base).

NOTE (server behavior): in the current ShibuDB, only `Flat` and `HNSW*` exercise
their real index. `IVF*`/`PQ*` spaces ingest and search through a Flat hot
segment (the trained index is only built for sealed segments, which are disabled
for those types), so their recall/latency will resemble Flat. This is reported
faithfully rather than hidden.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    Result,
    ResultWriter,
    compute_ground_truth,
    create_space,
    make_client,
    parse_search_ids,
    recall_at_k,
    run_phase,
    summarize,
    wait_for_searchable,
)

DEFAULT_INDEX_TYPES = [
    "Flat",
    "HNSW8", "HNSW16", "HNSW32", "HNSW64",
    "IVF64,Flat", "IVF128,Flat", "IVF256,Flat",
    "IVF256,PQ8", "IVF256,PQ16",
    "PQ8", "PQ16",
]


def _ground_truth(args, base, queries, sift_gt, full_base_n, cache: dict):
    """Resolve ground-truth for the current base/query/metric, caching it.

    The canonical SIFT ground-truth is only valid when the ENTIRE base set is
    ingested (its neighbor ids index into the full 1M base). For any subset we
    must compute exact ground-truth over the ingested vectors instead.
    """
    key = (args.num_base, args.k, args.metric)
    if key in cache:
        return cache[key]
    full = sift_gt is not None and args.num_base >= full_base_n and args.metric == "L2"
    if full:
        gt = sift_gt[:, : args.k]
        print("[gt] using canonical SIFT ground-truth", flush=True)
    else:
        print(f"[gt] computing exact ground-truth (num_base={args.num_base}, metric={args.metric})...", flush=True)
        t0 = time.perf_counter()
        gt = compute_ground_truth(base, queries, args.k, metric=args.metric)
        print(f"[gt] done in {time.perf_counter() - t0:.1f}s", flush=True)
    cache[key] = gt
    return gt


def run(args, base, queries, sift_gt, writer: ResultWriter):
    full_base_n = base.shape[0]
    base = base[: args.num_base]
    queries = queries[: args.num_queries]
    gt_cache: dict = {}
    gt = _ground_truth(args, base, queries, sift_gt, full_base_n, gt_cache)

    base_list = [base[i].astype(np.float32).tolist() for i in range(base.shape[0])]
    query_list = [queries[i].astype(np.float32).tolist() for i in range(queries.shape[0])]
    sample_ids = [0, args.num_base // 2, args.num_base - 1]

    wal_modes = [False, True] if args.both_wal else [args.enable_wal]

    for index_type in args.index_types:
        for wal in wal_modes:
            space = f"{args.space_prefix}_vec_{index_type.replace(',', '_')}_wal{int(wal)}_{int(time.time())}"
            print(f"\n=== VECTOR index={index_type} wal={wal} ===", flush=True)
            create_space(args, space, "vector", dimension=args.dimension,
                         index_type=index_type, metric=args.metric, enable_wal=wal)

            # --- ingest ---
            def insert_task(client, i):
                client.insert_vector(i, base_list[i])

            lat, _, failed, wall = run_phase(args, space, base.shape[0],
                                             args.ingest_concurrency, insert_task)
            ing = summarize(lat, wall, base.shape[0])
            writer.add(Result(
                suite="vector_search", operation="insert", engine="vector",
                index_type=index_type, metric=args.metric, wal=wal,
                num_base=args.num_base, concurrency=args.ingest_concurrency,
                ops=base.shape[0] - failed, throughput_ops_sec=ing["throughput_ops_sec"],
                mean_ms=ing["mean_ms"], p50_ms=ing["p50_ms"], p95_ms=ing["p95_ms"],
                p99_ms=ing["p99_ms"], failed=failed,
            ))
            print(f"[ingest] {ing['throughput_ops_sec']:.0f} vec/s ({wall:.1f}s, {failed} failed)", flush=True)

            wait_for_searchable(args, space, sample_ids, vector=True)

            # --- query sweep ---
            for c in args.concurrency:
                def search_task(client, i):
                    return parse_search_ids(client.search_topk(query_list[i], k=args.k))

                lat, results, failed, wall = run_phase(args, space, queries.shape[0], c,
                                                       search_task, collect=True)
                s = summarize(lat, wall, queries.shape[0])
                recalls = [recall_at_k(results[i] or [], gt[i], args.k) for i in range(queries.shape[0])]
                recall = float(np.mean(recalls)) if recalls else 0.0
                writer.add(Result(
                    suite="vector_search", operation="search", engine="vector",
                    index_type=index_type, metric=args.metric, wal=wal,
                    num_base=args.num_base, k=args.k, concurrency=c,
                    ops=queries.shape[0] - failed, throughput_ops_sec=s["throughput_ops_sec"],
                    recall_at_k=recall, mean_ms=s["mean_ms"], p50_ms=s["p50_ms"],
                    p95_ms=s["p95_ms"], p99_ms=s["p99_ms"], failed=failed,
                ))
                print(f"[query] c={c:<4d} QPS={s['throughput_ops_sec']:8.1f} "
                      f"recall@{args.k}={recall:.4f} p99={s['p99_ms']:.2f}ms failed={failed}", flush=True)

            if args.cleanup:
                try:
                    make_client(args).delete_space(space)
                except Exception:
                    pass
