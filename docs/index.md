# ShibuDB Benchmarks

Published performance results for [ShibuDB](https://github.com/shibudb/shibudb-server), measured
end-to-end through the official Python client against the **SIFT1M** dataset.

## Latest report

**[Benchmark Report →](BENCHMARKS.md)**

The report includes recall@k, throughput, latency percentiles, environment metadata, and charts
for vector search, metadata-filtered search, and key-value workloads.

> **Note:** `BENCHMARKS.md` is generated after a benchmark run. If you see a 404, generate it
> locally with `python report.py` (see the [repository README](../README.md)).

## Reproduce

```bash
python benchmark.py --suites all --both-wal --out results/run.csv
python plot.py --csv results/run.csv --out-dir results/charts
python report.py --csv results/run.csv --charts-dir results/charts
git add docs/ && git commit -m "Update benchmark report"
```

Enable **GitHub Pages** (Settings → Pages → deploy from branch `main`, folder `/docs`) to publish
this site at `https://<org>.github.io/shibudb-server-benchmarking/`.
