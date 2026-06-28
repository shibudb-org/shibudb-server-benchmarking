"""Key-value engine benchmark suite.

For each {WAL on, WAL off} x concurrency it measures throughput (ops/sec) and
p50/p95/p99 latency for PUT, GET, and DELETE against a key-value space, using
synthetic fixed-size values.
"""

from __future__ import annotations

import time

from common import (
    Result,
    ResultWriter,
    create_space,
    make_client,
    mark_phase,
    run_phase,
    summarize,
)


def run(args, writer: ResultWriter):
    n = args.kv_keys
    value = "x" * args.kv_value_size
    keys = [f"key_{i}" for i in range(n)]

    wal_modes = [False, True] if args.both_wal else [args.enable_wal]

    for wal in wal_modes:
        space = f"{args.space_prefix}_kv_wal{int(wal)}_{int(time.time())}"
        print(f"\n=== KEY-VALUE wal={wal} ===", flush=True)
        mark_phase(f"kv wal{int(wal)}")
        create_space(args, space, "key-value", enable_wal=wal)

        # PUT uses ingest concurrency (write load); GET/DELETE use the query sweep.
        def put_task(client, i):
            client.put(keys[i], value)

        lat, _, failed, wall = run_phase(args, space, n, args.ingest_concurrency, put_task)
        s = summarize(lat, wall, n)
        writer.add(Result(
            suite="key_value", operation="put", engine="key-value", wal=wal,
            num_base=n, concurrency=args.ingest_concurrency, ops=n - failed,
            throughput_ops_sec=s["throughput_ops_sec"], mean_ms=s["mean_ms"],
            p50_ms=s["p50_ms"], p95_ms=s["p95_ms"], p99_ms=s["p99_ms"], failed=failed,
        ))
        print(f"[put] {s['throughput_ops_sec']:.0f} ops/s ({wall:.1f}s, {failed} failed)", flush=True)

        for c in args.concurrency:
            def get_task(client, i):
                client.get(keys[i])

            lat, _, failed, wall = run_phase(args, space, n, c, get_task)
            s = summarize(lat, wall, n)
            writer.add(Result(
                suite="key_value", operation="get", engine="key-value", wal=wal,
                num_base=n, concurrency=c, ops=n - failed,
                throughput_ops_sec=s["throughput_ops_sec"], mean_ms=s["mean_ms"],
                p50_ms=s["p50_ms"], p95_ms=s["p95_ms"], p99_ms=s["p99_ms"], failed=failed,
            ))
            print(f"[get] c={c:<4d} {s['throughput_ops_sec']:8.0f} ops/s p99={s['p99_ms']:.2f}ms failed={failed}", flush=True)

        # DELETE last (destroys the keys). Single concurrency level to keep it simple.
        dc = args.concurrency[-1]

        def delete_task(client, i):
            client.delete(keys[i])

        lat, _, failed, wall = run_phase(args, space, n, dc, delete_task)
        s = summarize(lat, wall, n)
        writer.add(Result(
            suite="key_value", operation="delete", engine="key-value", wal=wal,
            num_base=n, concurrency=dc, ops=n - failed,
            throughput_ops_sec=s["throughput_ops_sec"], mean_ms=s["mean_ms"],
            p50_ms=s["p50_ms"], p95_ms=s["p95_ms"], p99_ms=s["p99_ms"], failed=failed,
        ))
        print(f"[delete] c={dc:<4d} {s['throughput_ops_sec']:8.0f} ops/s p99={s['p99_ms']:.2f}ms failed={failed}", flush=True)

        if args.cleanup:
            try:
                make_client(args).delete_space(space)
            except Exception:
                pass
