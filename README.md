# ShibuDB Benchmarks

Full-system performance benchmarking for ShibuDB, run over TCP through the
official [`shibudb-client`](https://pypi.org/project/shibudb-client/) Python SDK
against the standard **SIFT1M** dataset (1M × 128-dim vectors with precomputed
ground-truth).

It measures the database the way real clients use it and reports the three
numbers that make a database benchmark credible — **recall**, **throughput
(ops/sec)**, and **latency percentiles (p50/p95/p99)** — never throughput alone.

## Coverage

Three suites, each swept across **WAL off and WAL on** and multiple concurrency levels:

| Suite | What it covers | Metrics |
|-------|----------------|---------|
| `vector` | Every vector **index type** (`Flat`, `HNSW8/16/32/64`, `IVF*,Flat`, `IVF*,PQ*`, `PQ*`) | ingest vec/s, recall@k, QPS, p50/p95/p99 |
| `metadata` | **Metadata-filtered** search on a `Flat` space with indexed fields, across several filter selectivities | ingest vec/s, filtered recall@k, QPS, latency |
| `kv` | **Key-value** PUT / GET / DELETE | ops/sec, latency |

## Layout

| File | Purpose |
|------|---------|
| `common.py` | Shared: client, concurrent runner, latency stats, exact ground-truth (numpy), metadata generation, unified result schema, env capture |
| `dataset.py` | Downloads + loads SIFT1M (`.fvecs`/`.ivecs`) |
| `bench_vector.py` | Vector index-type × WAL × concurrency sweep |
| `bench_metadata.py` | Metadata-filtered search sweep |
| `bench_kv.py` | Key-value PUT/GET/DELETE sweep |
| `benchmark.py` | Orchestrator (`--suites`), writes unified CSV + JSON |
| `plot.py` | Charts for all suites |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start a ShibuDB server (from a checkout of the separate `shibudb-server` repo):

```bash
make start-local-server   # localhost:4444, admin/admin
```

## Quick local run (validate cheaply)

Small subset so the full matrix finishes in minutes:

```bash
python benchmark.py --suites all --both-wal \
  --num-base 20000 --num-queries 1000 --kv-keys 20000 \
  --index-types Flat HNSW32 IVF256,Flat \
  --concurrency 1 8 32 --out results/local_all.csv

python plot.py --csv results/local_all.csv --out-dir results/charts
```

The first run downloads SIFT1M (~250 MB) into `~/.shibudb-benchmarks/data`.

## Full publishable run (dedicated server; client on a separate machine)

```bash
# Vector: all index types, with and without WAL
python benchmark.py --suites vector --both-wal \
  --num-base 1000000 --num-queries 10000 --k 10 \
  --concurrency 1 8 32 64 128 --ingest-concurrency 32 \
  --out results/sift1m_vector.csv

# Metadata-filtered search + key-value
python benchmark.py --suites metadata kv --both-wal \
  --num-base 1000000 --kv-keys 1000000 \
  --concurrency 1 8 32 64 128 --out results/sift1m_meta_kv.csv

python plot.py --csv results/sift1m_vector.csv --out-dir results/charts/vector
python plot.py --csv results/sift1m_meta_kv.csv --out-dir results/charts/meta_kv
```

## Output

- `results/<name>.csv` — unified rows (`suite, operation, index_type, metric, wal,
  concurrency, scenario, selectivity, throughput_ops_sec, recall_at_k, p50/p95/p99_ms, ...`).
- `results/<name>.json` — same results plus a captured environment block (git
  commit, SDK/Python versions, host, CPU, OS, all args) for reproducibility.
- `results/charts/*.png` — per-suite recall/throughput/latency/ingest charts.

## Methodology notes / caveats (read before publishing)

Honest, ShibuDB-specific facts that affect interpretation:

- **IVF/PQ currently run via the Flat hot-path.** In this server version, only
  `Flat` and `HNSW*` exercise their real index. `IVF*`/`PQ*` spaces ingest and
  search through a Flat hot segment (the trained index is only built for sealed
  segments, which are disabled for those types), so expect their recall ≈ 1.0 and
  latency ≈ Flat. The suite still benchmarks them — and the results make this
  behavior visible rather than hiding it.
- **No query-time accuracy knobs.** The server searches with a fixed
  `searchK = max(k*8, 32)`; there is no `efSearch`/`nprobe`. The recall/QPS
  frontier is therefore traced by sweeping **index build params** (one operating
  point per config), not by tuning a single index — a scatter, not the smooth
  per-index curves ANN-Benchmarks draws.
- **Index string rules.** Suffixes must be a **power of two in [2, 256]** (so the
  max IVF is `IVF256`); IVF/PQ need a full FAISS descriptor (`IVF256,Flat`,
  `IVF256,PQ8`); PQ subquantizers must divide the dimension (128).
- **Recall ground-truth.** Uses the canonical SIFT ground-truth when the full 1M
  base is ingested with `L2`; otherwise computes **exact** top-k with numpy over
  the actually-ingested subset, so recall stays valid for any `--num-base`,
  metric, or metadata filter.
- **Asynchronous persistence.** Inserted vectors aren't instantly searchable; the
  suite waits (sampled `get_vector`/`get`) before querying.
- **WAL** is off by default and dominates insert latency when on — hence
  `--both-wal` to report it as an explicit dimension.
- **Client must not be the bottleneck.** Watch client-host CPU during the sweep;
  if a single Python process can't saturate the server, run the client on a
  separate machine (recommended) and/or shard across processes.

## Reproducibility checklist (for credible publishing)

- [x] Public dataset with verifiable ground-truth (SIFT1M)
- [x] Recall + throughput + latency reported together
- [x] All index types, metadata filtering, and KV — each with/without WAL
- [x] Environment (hardware, versions, commit) captured into every result file
- [x] Exact commands + scripts committed here
- [ ] Pinned, dedicated hardware (planned: `infra/` Terraform)
- [ ] Charts + writeup published in `docs/BENCHMARKS.md`
