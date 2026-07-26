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
| `report.py` | Markdown report for GitHub Pages (`docs/BENCHMARKS.md`) |
| `docs/` | GitHub Pages site (`index.md` + generated `BENCHMARKS.md`) |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Start a ShibuDB server (from a checkout of the separate `shibudb-server` repo):

```bash
make start-local-server   # localhost:4444, admin/admin
```

## ▶ Generate the FULL ShibuDB report — run these commands

> [!IMPORTANT]
> This is the canonical, complete benchmark: **all 12 index types × WAL off/on ×
> concurrency 1–128** on the full 1M-vector SIFT1M base, plus metadata-filtered
> search and key-value — ending with the publishable report and a downloadable
> zip bundle. Run it on a dedicated server with the client on a separate machine;
> it takes many hours. All smaller / customized runs come in the next section.

```bash
# 0. Record server build + hardware once (edit examples/server-info.example.json for your host)
SERVER_INFO=examples/server-info.example.json
SERVER_VER=v1.0.0
SERVER_SHA=abc123def456

# 1. Vector suite: ALL index types (Flat, HNSW8-64, IVF*, PQ*), WAL off + on
python benchmark.py --suites vector --both-wal \
  --num-base 1000000 --num-queries 10000 --k 10 \
  --concurrency 1 8 32 64 128 --ingest-concurrency 32 \
  --server-version "$SERVER_VER" --server-commit "$SERVER_SHA" \
  --server-info-file "$SERVER_INFO" \
  --out results/sift1m_vector.csv

# 2. Metadata-filtered search + key-value at full scale
python benchmark.py --suites metadata kv --both-wal \
  --num-base 1000000 --kv-keys 1000000 \
  --concurrency 1 8 32 64 128 \
  --server-version "$SERVER_VER" --server-commit "$SERVER_SHA" \
  --server-info-file "$SERVER_INFO" \
  --out results/sift1m_meta_kv.csv

# 3. Charts
python plot.py --csv results/sift1m_vector.csv --out-dir results/charts/vector
python plot.py --csv results/sift1m_meta_kv.csv --out-dir results/charts/meta_kv

# 4. Publishable Markdown report (docs/BENCHMARKS.md) + downloadable zip bundle
python report.py --csv results/sift1m_vector.csv results/sift1m_meta_kv.csv \
  --charts-dir results/charts --charts-subdir vector meta_kv --zip
```

The first run downloads SIFT1M (~250 MB) into `~/.shibudb-benchmarks/data`.
The final step writes `docs/BENCHMARKS.md` (GitHub-ready, with the generated-report
banner) and `results/benchmark_report_<timestamp>.zip` for download/sharing.

## Customized / smaller runs

### Quick local run (validate cheaply)

Small subset so the full matrix finishes in minutes:

```bash
python benchmark.py --suites all --both-wal \
  --num-base 20000 --num-queries 1000 --kv-keys 20000 \
  --index-types Flat HNSW32 IVF256,Flat \
  --concurrency 1 8 32 --out results/local_all.csv

python plot.py --csv results/local_all.csv --out-dir results/charts
python report.py --csv results/local_all.csv --charts-dir results/charts --zip
```

### Common customizations

| What to change | Flag |
|----------------|------|
| Suites to run | `--suites all` \| `vector` \| `metadata` \| `kv` |
| Index types swept | `--index-types Flat HNSW32 ...` (omit for all 12) |
| Data scale | `--num-base`, `--num-queries`, `--kv-keys`, `--kv-value-size` |
| Concurrency sweep | `--concurrency 1 8 32 64 128`, `--ingest-concurrency` |
| WAL | `--both-wal` (off **and** on) or `--enable-wal` (on only) |
| Distance metric | `--metric L2` \| `InnerProduct` \| `L1` \| ... |
| Remote server | `--host`, `--port`, `--user`, `--password` |

Run `python benchmark.py --help` for the full list.

## Publish report (GitHub Pages tab)

After benchmark + plot, generate the Markdown report and commit it under `docs/`:

```bash
python report.py --csv results/run.csv --charts-dir results/charts
# writes docs/BENCHMARKS.md and copies charts to docs/charts/

git add docs/
git commit -m "Update benchmark report"
git push
```

### Downloadable zip bundle

Pass `--zip` to also package everything into a single downloadable archive —
the Markdown report (at the archive root, with working image links), all chart
PNGs, and the raw CSV/JSON result files under `raw-results/`:

```bash
python report.py --csv results/run.csv --charts-dir results/charts --zip
# writes results/benchmark_report_<UTC timestamp>.zip

python report.py --csv results/run.csv --charts-dir results/charts --zip results/my_report.zip
# writes to an explicit path instead
```

Then enable **GitHub Pages** once per repo:

1. Repo **Settings → Pages**
2. **Build and deployment → Source:** Deploy from a branch
3. **Branch:** `main` (or your default), folder **`/docs`**
4. Save — the site appears at `https://<org>.github.io/<repo>/`

| Page | URL | Role |
|------|-----|------|
| Site home | `…/shibudb-server-benchmarking/` | `docs/index.md` — links to the report |
| Benchmark report | `…/shibudb-server-benchmarking/BENCHMARKS` | Generated `docs/BENCHMARKS.md` |

Add the Pages URL to the repo **About → Website** field so it shows next to the README on GitHub.

## Output

- `results/<name>.csv` — unified rows (`suite, operation, index_type, metric, wal,
  concurrency, scenario, selectivity, throughput_ops_sec, recall_at_k, p50/p95/p99_ms, ...`).
- `results/<name>.json` — same results plus a captured environment block (git
  commit, ShibuDB server version/commit, client + server hardware, data scale,
  SDK/Python versions, all args) for reproducibility.
- `results/charts/*.png` — per-suite recall/throughput/latency/ingest charts.
- `docs/BENCHMARKS.md` — publishable report (data scale, versions, hardware,
  result tables, embedded charts) for GitHub Pages.
- `results/benchmark_report_<timestamp>.zip` — optional (`report.py --zip`)
  self-contained bundle of the report + charts + raw results for download/sharing.

### Server metadata flags

| Flag | Purpose |
|------|---------|
| `--server-version` | ShibuDB server release/tag (e.g. `v1.2.3`) |
| `--server-commit` | ShibuDB server git SHA |
| `--server-info-file` | JSON with server CPU, RAM, disk, network (see `examples/server-info.example.json`) |

These are captured into the results JSON at benchmark time and rendered prominently
in the generated report. You can also pass them to `report.py` when regenerating a
report from older result files.

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
- [x] Environment (client + server hardware, ShibuDB version, data scale) captured into every result file
- [x] Exact commands + scripts committed here
- [ ] Pinned, dedicated hardware (planned: `infra/` Terraform)
- [x] Report generator (`report.py` → `docs/BENCHMARKS.md` + GitHub Pages)
- [ ] Latest full-run report committed under `docs/`
