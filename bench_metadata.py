"""Metadata-filtered vector search benchmark suite.

Metadata filtering is only supported on `Flat` vector spaces declared with
`indexed_metadata_fields` (the server's filterable Flat engine does exact
brute-force distance over a metadata pre-filtered candidate set).

For each {WAL on, WAL off} x filter-scenario x concurrency it measures filtered
QPS, p50/p95/p99 latency, and filtered recall@k against an exact ground-truth
computed over only the matching subset (numpy). It also records the
ingest-with-metadata throughput.
"""

from __future__ import annotations

import time

import numpy as np

from common import (
    Result,
    ResultWriter,
    compute_ground_truth,
    create_space,
    filter_scenarios,
    make_client,
    metadata_field_spec,
    metadata_for_id,
    parse_search_ids,
    recall_at_k,
    run_phase,
    summarize,
    wait_for_searchable,
)


def run(args, base, queries, writer: ResultWriter):
    base = base[: args.num_base]
    queries = queries[: args.num_queries]

    base_list = [base[i].astype(np.float32).tolist() for i in range(base.shape[0])]
    query_list = [queries[i].astype(np.float32).tolist() for i in range(queries.shape[0])]
    meta_list = [metadata_for_id(i) for i in range(base.shape[0])]
    sample_ids = [0, args.num_base // 2, args.num_base - 1]

    scenarios = filter_scenarios(base.shape[0])

    # Precompute filtered ground-truth per scenario (reused across WAL/concurrency).
    print("[gt] computing filtered ground-truth per scenario...", flush=True)
    for sc in scenarios:
        cand = np.nonzero(sc["mask"])[0]
        sc["candidates"] = cand
        sc["gt"] = compute_ground_truth(base, queries, args.k, metric=args.metric,
                                        candidate_ids=cand)
        print(f"[gt]   {sc['label']}: {len(cand)} matching vectors "
              f"(selectivity~{sc['selectivity']:.3f})", flush=True)

    wal_modes = [False, True] if args.both_wal else [args.enable_wal]

    for wal in wal_modes:
        space = f"{args.space_prefix}_meta_wal{int(wal)}_{int(time.time())}"
        print(f"\n=== METADATA-FILTER (Flat) wal={wal} ===", flush=True)
        create_space(args, space, "vector", dimension=args.dimension,
                     index_type="Flat", metric=args.metric, enable_wal=wal,
                     indexed_metadata_fields=metadata_field_spec())

        # --- ingest with metadata ---
        def insert_task(client, i):
            client.insert_vector(i, base_list[i], metadata=meta_list[i])

        lat, _, failed, wall = run_phase(args, space, base.shape[0],
                                         args.ingest_concurrency, insert_task)
        ing = summarize(lat, wall, base.shape[0])
        writer.add(Result(
            suite="metadata_filter", operation="insert", engine="vector",
            index_type="Flat", metric=args.metric, wal=wal, num_base=args.num_base,
            concurrency=args.ingest_concurrency, ops=base.shape[0] - failed,
            throughput_ops_sec=ing["throughput_ops_sec"], mean_ms=ing["mean_ms"],
            p50_ms=ing["p50_ms"], p95_ms=ing["p95_ms"], p99_ms=ing["p99_ms"], failed=failed,
        ))
        print(f"[ingest+meta] {ing['throughput_ops_sec']:.0f} vec/s ({wall:.1f}s, {failed} failed)", flush=True)

        wait_for_searchable(args, space, sample_ids, vector=True)

        # --- filtered query sweep ---
        for sc in scenarios:
            where = sc["where"]
            gt = sc["gt"]
            for c in args.concurrency:
                def search_task(client, i):
                    return parse_search_ids(client.search_topk(query_list[i], k=args.k, where=where))

                lat, results, failed, wall = run_phase(args, space, queries.shape[0], c,
                                                       search_task, collect=True)
                s = summarize(lat, wall, queries.shape[0])
                recalls = [recall_at_k(results[i] or [], gt[i], args.k) for i in range(queries.shape[0])]
                recall = float(np.mean(recalls)) if recalls else 0.0
                writer.add(Result(
                    suite="metadata_filter", operation="filtered_search", engine="vector",
                    index_type="Flat", metric=args.metric, wal=wal, num_base=args.num_base,
                    k=args.k, concurrency=c, scenario=sc["label"], selectivity=sc["selectivity"],
                    ops=queries.shape[0] - failed, throughput_ops_sec=s["throughput_ops_sec"],
                    recall_at_k=recall, mean_ms=s["mean_ms"], p50_ms=s["p50_ms"],
                    p95_ms=s["p95_ms"], p99_ms=s["p99_ms"], failed=failed,
                ))
                print(f"[filtered] {sc['label']:<28s} c={c:<4d} QPS={s['throughput_ops_sec']:8.1f} "
                      f"recall@{args.k}={recall:.4f} p99={s['p99_ms']:.2f}ms failed={failed}", flush=True)

        if args.cleanup:
            try:
                make_client(args).delete_space(space)
            except Exception:
                pass
