"""Generate a publishable Markdown benchmark report for GitHub Pages.

Reads one or more result CSV/JSON files produced by benchmark.py and writes
`docs/BENCHMARKS.md` (by default), optionally copying chart PNGs into
`docs/charts/` so images render on GitHub Pages.

Typical workflow:
  python benchmark.py ... --out results/run.csv
  python plot.py --csv results/run.csv --out-dir results/charts
  python report.py --csv results/run.csv --charts-dir results/charts

GitHub Pages (repo Settings → Pages → branch main, folder /docs) serves
`docs/index.md` as the site home and links to the generated report.

Usage:
  python report.py --csv results/local_all.csv --charts-dir results/charts
  python report.py --csv results/sift1m_vector.csv results/sift1m_meta_kv.csv \\
      --charts-dir results/charts --charts-subdir vector meta_kv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone


REPO_URL = "https://github.com/shibudb-org/shibudb-server-benchmarking"

METHODOLOGY = """\
## Methodology

- **Dataset:** [SIFT1M](http://corpus-texmex.irisa.fr/) (1M × 128-dim vectors, 10k queries).
- **Transport:** Official [`shibudb-client`](https://pypi.org/project/shibudb-client/) over TCP.
- **Metrics:** Recall@k, throughput (ops/sec), and latency percentiles (p50/p95/p99) are
  always reported together for search workloads.
- **Concurrency:** Process-based client parallelism (one OS process per worker) so Python's
  GIL does not cap measured server throughput.
- **Ground truth:** Canonical SIFT neighbors at full 1M + L2; otherwise exact numpy top-k over
  the ingested subset (and over metadata-filtered candidates for filtered search).

### Caveats

- **IVF/PQ** currently run through a Flat hot segment in this server version — expect recall
  ≈ 1.0 and latency ≈ Flat until the trained index is used on the hot path.
- **No query-time accuracy knobs** (`efSearch` / `nprobe`); one operating point per index config.
- **WAL** is reported as an explicit dimension (`--both-wal`).
- **Async persistence:** the suite waits until sample inserts are retrievable before querying.
"""


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _fnum(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def _wal_label(row: dict) -> str:
    return "on" if str(row.get("wal", "")).lower() in ("true", "1") else "off"


def _fmt_throughput(v: float) -> str:
    if v >= 10_000:
        return f"{v:,.0f}"
    if v >= 100:
        return f"{v:,.1f}"
    return f"{v:.2f}"


def _fmt_ms(v: float) -> str:
    return f"{v:.2f}"


def _fmt_recall(v: float) -> str:
    if v < 0:
        return "—"
    return f"{v:.4f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No results._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _human_bytes(n: int | float) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def _fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def _normalize_env(env: dict, overrides: dict | None = None) -> dict:
    """Merge legacy flat env keys and optional report-time overrides."""
    overrides = overrides or {}
    out = dict(env)

    client = dict(env.get("client") or {})
    if not client:
        client = {
            "host": env.get("client_host", "unknown"),
            "platform": env.get("platform", "unknown"),
            "processor": env.get("processor") or "unknown",
            "cpu_count": env.get("cpu_count"),
        }

    server = dict(env.get("server") or {})
    if not server:
        server = {
            "host": env.get("server_host", "unknown"),
            "port": env.get("server_port"),
            "version": "unknown",
            "commit": "unknown",
        }
    if overrides.get("server_version"):
        server["version"] = overrides["server_version"]
    if overrides.get("server_commit"):
        server["commit"] = overrides["server_commit"]
    if overrides.get("server_info"):
        server.update(overrides["server_info"])

    data_scale = dict(env.get("data_scale") or {})
    args = env.get("args") or {}
    if not data_scale and args:
        data_scale = {
            "dataset": env.get("dataset", "SIFT1M"),
            "num_base": args.get("num_base", 0),
            "num_queries": args.get("num_queries", 0),
            "dimension": args.get("dimension", 128),
            "k": args.get("k", 10),
            "metric": args.get("metric", "L2"),
            "kv_keys": args.get("kv_keys", 0),
            "kv_value_size_bytes": args.get("kv_value_size", 100),
        }
        nb = int(data_scale["num_base"] or 0)
        dim = int(data_scale["dimension"] or 128)
        nq = int(data_scale["num_queries"] or 0)
        data_scale["full_sift_base"] = nb >= 1_000_000
        data_scale["vector_base_bytes"] = nb * dim * 4
        data_scale["vector_query_bytes"] = nq * dim * 4
        kv_keys = int(data_scale.get("kv_keys") or 0)
        kv_val = int(data_scale.get("kv_value_size_bytes") or 100)
        data_scale["kv_total_bytes_approx"] = kv_keys * (8 + kv_val)

    out["client"] = client
    out["server"] = server
    out["data_scale"] = data_scale
    out["args"] = args
    return out


def _infer_data_scale_from_rows(rows: list[dict], data_scale: dict) -> dict:
    """Fill missing data_scale fields from result rows when env is sparse."""
    out = dict(data_scale)
    suites = {r.get("suite") for r in rows if r.get("suite")}
    if suites:
        out["suites"] = sorted(suites)

    base_vals = [int(_fnum(r, "num_base")) for r in rows if _fnum(r, "num_base") > 0]
    if base_vals and not out.get("num_base"):
        out["num_base"] = max(base_vals)

    k_vals = [int(_fnum(r, "k")) for r in rows if _fnum(r, "k") > 0]
    if k_vals and not out.get("k"):
        out["k"] = k_vals[0]

    metrics = [r.get("metric") for r in rows if r.get("metric")]
    if metrics and not out.get("metric"):
        out["metric"] = metrics[0]

    return out


def _data_scale_section(data_scale: dict, rows: list[dict]) -> str:
    ds = _infer_data_scale_from_rows(rows, data_scale)
    lines = ["### Data scale", ""]

    dataset = ds.get("dataset", "SIFT1M")
    suites = ds.get("suites") or []
    suite_labels = {
        "vector_search": "vector search",
        "metadata_filter": "metadata-filtered search",
        "key_value": "key-value",
    }
    if suites:
        labels = [suite_labels.get(s, s) for s in suites]
        lines.append(f"**Suites in this run:** {', '.join(labels)}.")
        lines.append("")

    num_base = int(ds.get("num_base") or 0)
    num_queries = int(ds.get("num_queries") or 0)
    dimension = int(ds.get("dimension") or 128)
    k = int(ds.get("k") or 10)
    metric = ds.get("metric") or "L2"
    kv_keys = int(ds.get("kv_keys") or 0)
    kv_val = int(ds.get("kv_value_size_bytes") or ds.get("kv_value_size") or 100)

    vector_lines = []
    if num_base > 0 and ("vector_search" in suites or "metadata_filter" in suites or not suites):
        full_note = "full SIFT1M base" if ds.get("full_sift_base") else f"{_fmt_int(num_base)}-vector subset"
        base_bytes = int(ds.get("vector_base_bytes") or num_base * dimension * 4)
        vector_lines.append(
            f"- **Vector base:** {_fmt_int(num_base)} vectors × {dimension}-dim float32 "
            f"({_human_bytes(base_bytes)}; {full_note} of {dataset})"
        )
    if num_queries > 0:
        query_bytes = int(ds.get("vector_query_bytes") or num_queries * dimension * 4)
        vector_lines.append(
            f"- **Queries:** {_fmt_int(num_queries)} vectors × {dimension}-dim float32 "
            f"({_human_bytes(query_bytes)})"
        )
    if k > 0 and num_queries > 0:
        vector_lines.append(f"- **Search accuracy target:** recall@{k} ({metric} metric)")

    if vector_lines:
        lines.append("**Vector workloads**")
        lines.append("")
        lines.extend(vector_lines)
        lines.append("")

    if kv_keys > 0 and ("key_value" in suites or not suites):
        kv_bytes = int(ds.get("kv_total_bytes_approx") or kv_keys * (8 + kv_val))
        lines.extend([
            "**Key-value workload**",
            "",
            f"- **Keys:** {_fmt_int(kv_keys)}",
            f"- **Value size:** {_fmt_int(kv_val)} bytes per key",
            f"- **Approx. payload:** {_human_bytes(kv_bytes)} (keys + values, excluding protocol overhead)",
            "",
        ])

    if not vector_lines and kv_keys <= 0:
        lines.append("_Data scale not recorded — pass `--num-base`, `--num-queries`, and/or `--kv-keys` to benchmark.py._")
        lines.append("")

    return "\n".join(lines)


def _system_table(title: str, fields: list[tuple[str, str]]) -> list[str]:
    lines = [f"### {title}", "", "| | |", "|---|---|"]
    for label, value in fields:
        if value in (None, "", "unknown") and label.endswith("*"):
            continue
        lines.append(f"| **{label.rstrip('*')}** | {value or 'unknown'} |")
    lines.append("")
    return lines


def _env_section(env: dict, rows: list[dict] | None = None) -> str:
    env = _normalize_env(env)
    rows = rows or []
    args = env.get("args") or {}
    client = env.get("client") or {}
    server = env.get("server") or {}
    data_scale = env.get("data_scale") or {}

    server_endpoint = f"{server.get('host', 'unknown')}:{server.get('port', '?')}"
    server_version = server.get("version", "unknown")
    server_commit = server.get("commit", "unknown")
    version_line = f"`{server_version}`"
    if server_commit and server_commit != "unknown":
        short = server_commit[:12]
        version_line += f" (commit `{short}`)"

    lines = [
        "## Environment",
        "",
        "| | |",
        "|---|---|",
        f"| **Report generated** | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} |",
        f"| **Benchmark run** | {env.get('timestamp', 'unknown')} |",
        f"| **Benchmark harness commit** | `{env.get('git_commit', 'unknown')}` |",
        f"| **ShibuDB server** | {version_line} |",
        f"| **Client SDK** | `{env.get('shibudb_client_version', 'unknown')}` |",
        f"| **Python (client)** | {env.get('python_version', 'unknown')} |",
        f"| **Server endpoint** | {server_endpoint} |",
        "",
    ]
    lines.append(_data_scale_section(data_scale, rows))

    lines.extend(_system_table("Client machine (load generator)", [
        ("Hostname", client.get("host", env.get("client_host", "unknown"))),
        ("Platform", client.get("platform", env.get("platform", "unknown"))),
        ("CPU", f"{client.get('processor') or env.get('processor') or 'unknown'} "
                f"({client.get('cpu_count') or env.get('cpu_count') or '?'} cores)"),
        ("RAM*", _ram_field(client)),
    ]))

    server_fields = [
        ("Hostname*", server.get("hostname")),
        ("Platform*", server.get("platform")),
        ("CPU*", _cpu_field(server)),
        ("RAM*", _ram_field(server)),
        ("Storage*", server.get("disk") or server.get("storage")),
        ("Network*", server.get("network")),
        ("Notes*", server.get("notes")),
    ]
    lines.extend(_system_table("Server machine (ShibuDB)", server_fields))
    if all(server.get(k) in (None, "", "unknown") for k in ("platform", "processor", "hostname")):
        lines.extend([
            "> Server hardware was not supplied. Re-run with `--server-info-file` (see "
            "`examples/server-info.example.json`) or pass the same flag to `report.py`.",
            "",
        ])

    skip_args = {
        "password", "num_base", "num_queries", "dimension", "k", "metric",
        "kv_keys", "kv_value_size", "server_version", "server_commit", "server_info_file",
        "host", "port", "user", "data_dir", "out",
    }
    extra_args = [(k, args[k]) for k in sorted(args) if k not in skip_args]
    if extra_args:
        lines.extend(["### Other run parameters", "", "| Parameter | Value |", "|---|---|"])
        for key, val in extra_args:
            if isinstance(val, list):
                val = ", ".join(str(x) for x in val)
            lines.append(f"| `{key}` | {val} |")
        lines.append("")

    return "\n".join(lines)


def _cpu_field(info: dict) -> str | None:
    proc = info.get("processor") or info.get("cpu")
    cores = info.get("cpu_count") or info.get("cores")
    if proc and cores:
        return f"{proc} ({cores} cores)"
    return proc or (f"{cores} cores" if cores else None)


def _ram_field(info: dict) -> str | None:
    if info.get("ram_gb"):
        return f"{info['ram_gb']} GB"
    if info.get("ram"):
        return str(info["ram"])
    return None


def _machine_summary(info: dict) -> str:
    """One-line 'OS · CPU · RAM' summary of a machine, from env info."""
    parts = []
    if info.get("platform") not in (None, "", "unknown"):
        parts.append(str(info["platform"]))
    cpu = _cpu_field(info)
    if cpu:
        parts.append(cpu)
    ram = _ram_field(info)
    if ram:
        parts.append(f"{ram} RAM")
    return " · ".join(parts)


def _banner(env: dict | None) -> list[str]:
    """Highlighted GitHub alert stating this report is generated, with
    tool link, timestamps, and machine OS/CPU/RAM."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    facts = [
        f"**This is an auto-generated benchmark report** produced by the "
        f"[ShibuDB benchmarking tool]({REPO_URL}) on **{generated}**.",
    ]
    if env:
        run_ts = env.get("timestamp")
        if run_ts:
            facts.append(f"**Benchmark run:** {run_ts}")
        server = env.get("server") or {}
        server_summary = _machine_summary(server)
        version = server.get("version")
        if version and version != "unknown":
            server_summary = f"ShibuDB `{version}`" + (f" — {server_summary}" if server_summary else "")
        if server_summary:
            facts.append(f"**Server:** {server_summary}")
        client_summary = _machine_summary(env.get("client") or {})
        if client_summary:
            facts.append(f"**Client (load generator):** {client_summary}")

    lines = ["> [!NOTE]"]
    for i, fact in enumerate(facts):
        if i:
            lines.append(">")  # blank quote line: keeps each fact on its own line on GitHub
        lines.append(f"> {fact}")
    return lines + [""]


def _pick_highlight_concurrency(rows: list[dict]) -> int:
    """Prefer the highest concurrency that had zero failures, else the max seen."""
    concurrencies = sorted({_fnum(r, "concurrency") for r in rows if _fnum(r, "concurrency") > 0})
    if not concurrencies:
        return 0
    for c in reversed(concurrencies):
        subset = [r for r in rows if int(_fnum(r, "concurrency")) == int(c)]
        if subset and all(int(_fnum(r, "failed")) == 0 for r in subset):
            return int(c)
    return int(concurrencies[-1])


def _vector_section(rows: list[dict], chart_prefix: str) -> str:
    inserts = [r for r in rows if r["suite"] == "vector_search" and r["operation"] == "insert"]
    searches = [r for r in rows if r["suite"] == "vector_search" and r["operation"] == "search"]
    if not inserts and not searches:
        return ""

    k = searches[0]["k"] if searches else "?"
    highlight_c = _pick_highlight_concurrency(searches)

    parts = ["## Vector search", ""]
    if highlight_c:
        parts.append(f"Search highlights at **concurrency = {highlight_c}** "
                     f"(highest concurrency with zero failures).")
        parts.append("")

    # Ingest table
    parts.append("### Ingest throughput")
    parts.append("")
    ingest_rows = []
    for r in sorted(inserts, key=lambda x: (x["index_type"], _wal_label(x))):
        ingest_rows.append([
            r["index_type"],
            _wal_label(r),
            str(int(_fnum(r, "concurrency"))),
            _fmt_throughput(_fnum(r, "throughput_ops_sec")),
            _fmt_ms(_fnum(r, "p99_ms")),
            str(int(_fnum(r, "failed"))),
        ])
    parts.append(_table(
        ["Index", "WAL", "Concurrency", "Throughput (vec/s)", "p99 (ms)", "Failed"],
        ingest_rows,
    ))

    # Search table (highlight concurrency)
    parts.append(f"### Search (recall@{k}, throughput, latency)")
    parts.append("")
    search_rows = []
    for r in sorted(searches, key=lambda x: (x["index_type"], _wal_label(x), _fnum(x, "concurrency"))):
        if highlight_c and int(_fnum(r, "concurrency")) != highlight_c:
            continue
        search_rows.append([
            r["index_type"],
            _wal_label(r),
            str(int(_fnum(r, "concurrency"))),
            _fmt_throughput(_fnum(r, "throughput_ops_sec")),
            _fmt_recall(_fnum(r, "recall_at_k")),
            _fmt_ms(_fnum(r, "p50_ms")),
            _fmt_ms(_fnum(r, "p95_ms")),
            _fmt_ms(_fnum(r, "p99_ms")),
            str(int(_fnum(r, "failed"))),
        ])
    parts.append(_table(
        [f"Index", "WAL", "Concurrency", "QPS", f"Recall@{k}", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Failed"],
        search_rows,
    ))

    # Full concurrency matrix per index (compact)
    parts.append("### Search by concurrency")
    parts.append("")
    by_index = defaultdict(list)
    for r in searches:
        by_index[(r["index_type"], _wal_label(r))].append(r)
    matrix_rows = []
    for (index_type, wal), items in sorted(by_index.items()):
        for r in sorted(items, key=lambda x: _fnum(x, "concurrency")):
            matrix_rows.append([
                index_type,
                wal,
                str(int(_fnum(r, "concurrency"))),
                _fmt_throughput(_fnum(r, "throughput_ops_sec")),
                _fmt_recall(_fnum(r, "recall_at_k")),
                _fmt_ms(_fnum(r, "p99_ms")),
            ])
    parts.append(_table(
        ["Index", "WAL", "Concurrency", "QPS", f"Recall@{k}", "p99 (ms)"],
        matrix_rows,
    ))

    if chart_prefix:
        parts.extend([
            "### Charts",
            "",
            f"![Vector recall vs QPS]({chart_prefix}/vector_recall_vs_qps.png)",
            "",
            f"![Vector QPS vs concurrency]({chart_prefix}/vector_qps_vs_concurrency.png)",
            "",
            f"![Vector p99 latency vs concurrency]({chart_prefix}/vector_latency_vs_concurrency.png)",
            "",
            f"![Vector ingest throughput]({chart_prefix}/vector_ingest_throughput.png)",
            "",
        ])
    return "\n".join(parts)


def _metadata_section(rows: list[dict], chart_prefix: str) -> str:
    inserts = [r for r in rows if r["suite"] == "metadata_filter" and r["operation"] == "insert"]
    searches = [r for r in rows if r["suite"] == "metadata_filter" and r["operation"] == "filtered_search"]
    if not inserts and not searches:
        return ""

    k = searches[0]["k"] if searches else "?"
    highlight_c = _pick_highlight_concurrency(searches)

    parts = ["## Metadata-filtered search", ""]
    parts.append("Flat index with indexed metadata fields; ground truth computed over the "
                 "filter-matched candidate set only.")
    parts.append("")
    if highlight_c:
        parts.append(f"Highlights at **concurrency = {highlight_c}**.")
        parts.append("")

    if inserts:
        parts.append("### Ingest (with metadata)")
        parts.append("")
        ingest_rows = [[
            _wal_label(r),
            str(int(_fnum(r, "concurrency"))),
            _fmt_throughput(_fnum(r, "throughput_ops_sec")),
            _fmt_ms(_fnum(r, "p99_ms")),
        ] for r in inserts]
        parts.append(_table(
            ["WAL", "Concurrency", "Throughput (vec/s)", "p99 (ms)"],
            ingest_rows,
        ))

    parts.append(f"### Filtered search (recall@{k})")
    parts.append("")
    search_rows = []
    for r in sorted(searches, key=lambda x: (x["scenario"], _wal_label(x), _fnum(x, "concurrency"))):
        if highlight_c and int(_fnum(r, "concurrency")) != highlight_c:
            continue
        sel = _fnum(r, "selectivity")
        sel_s = f"{sel:.3f}" if sel >= 0 else "—"
        search_rows.append([
            r["scenario"],
            sel_s,
            _wal_label(r),
            str(int(_fnum(r, "concurrency"))),
            _fmt_throughput(_fnum(r, "throughput_ops_sec")),
            _fmt_recall(_fnum(r, "recall_at_k")),
            _fmt_ms(_fnum(r, "p99_ms")),
        ])
    parts.append(_table(
        ["Scenario", "Selectivity", "WAL", "Concurrency", "QPS", f"Recall@{k}", "p99 (ms)"],
        search_rows,
    ))

    if chart_prefix:
        parts.extend([
            "### Charts",
            "",
            f"![Metadata recall vs QPS]({chart_prefix}/metadata_recall_vs_qps.png)",
            "",
            f"![Metadata QPS vs concurrency]({chart_prefix}/metadata_qps_vs_concurrency.png)",
            "",
        ])
    return "\n".join(parts)


def _kv_section(rows: list[dict], chart_prefix: str) -> str:
    kv = [r for r in rows if r["suite"] == "key_value"]
    if not kv:
        return ""

    parts = ["## Key-value", ""]
    table_rows = []
    for r in sorted(kv, key=lambda x: (x["operation"], _wal_label(x), _fnum(x, "concurrency"))):
        table_rows.append([
            r["operation"].upper(),
            _wal_label(r),
            str(int(_fnum(r, "concurrency"))),
            _fmt_throughput(_fnum(r, "throughput_ops_sec")),
            _fmt_ms(_fnum(r, "p50_ms")),
            _fmt_ms(_fnum(r, "p99_ms")),
            str(int(_fnum(r, "failed"))),
        ])
    parts.append(_table(
        ["Operation", "WAL", "Concurrency", "Throughput (ops/s)", "p50 (ms)", "p99 (ms)", "Failed"],
        table_rows,
    ))

    if chart_prefix:
        parts.extend([
            "### Charts",
            "",
            f"![KV throughput vs concurrency]({chart_prefix}/kv_throughput.png)",
            "",
            f"![KV p99 latency vs concurrency]({chart_prefix}/kv_latency.png)",
            "",
        ])
    return "\n".join(parts)


def _copy_charts(sources: list[tuple[str, str]], dest_dir: str) -> None:
    """Copy chart PNGs from source dirs into dest_dir (flat or prefixed subdirs)."""
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    for charts_dir, subdir in sources:
        src_root = os.path.join(charts_dir, subdir) if subdir else charts_dir
        if not src_root or not os.path.isdir(src_root):
            continue
        target_root = os.path.join(dest_dir, subdir) if subdir else dest_dir
        os.makedirs(target_root, exist_ok=True)
        for name in os.listdir(src_root):
            if not name.endswith(".png"):
                continue
            src = os.path.join(src_root, name)
            dst = os.path.join(target_root, name)
            shutil.copy2(src, dst)
            copied += 1
    if copied:
        print(f"Copied {copied} chart(s) -> {dest_dir}", flush=True)


def _create_zip(
    zip_path: str,
    report_path: str,
    csv_paths: list[str],
    charts_dirs: list[str],
    chart_prefixes: list[str],
) -> None:
    """Package the report, charts, and raw result files into one archive.

    The report Markdown sits at the archive root and charts keep the same
    relative prefix used inside the report, so images still resolve when the
    zip is extracted and the Markdown is previewed.
    """
    zip_dir = os.path.dirname(os.path.abspath(zip_path))
    if zip_dir:
        os.makedirs(zip_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report_path, arcname=os.path.basename(report_path))

        seen: set[str] = set()
        for charts_dir, prefix in zip(charts_dirs, chart_prefixes):
            if not charts_dir or not os.path.isdir(charts_dir):
                continue
            # A prefix that escapes the report dir can't be mirrored in a zip.
            arc_root = "charts" if (not prefix or prefix.startswith("..")) else prefix
            for root, _, files in os.walk(charts_dir):
                for name in files:
                    if not name.endswith(".png"):
                        continue
                    src = os.path.join(root, name)
                    rel = os.path.relpath(src, charts_dir)
                    arcname = os.path.join(arc_root, rel).replace(os.sep, "/")
                    if arcname in seen:
                        continue
                    seen.add(arcname)
                    zf.write(src, arcname=arcname)

        for path in csv_paths:
            if os.path.isfile(path):
                zf.write(path, arcname=f"raw-results/{os.path.basename(path)}")
            for ext in (".json", ".log"):
                sidecar = os.path.splitext(path)[0] + ext
                if os.path.isfile(sidecar):
                    zf.write(sidecar, arcname=f"raw-results/{os.path.basename(sidecar)}")

    size = os.path.getsize(zip_path)
    print(f"Wrote {zip_path} ({_human_bytes(size)})", flush=True)


_MAX_LOG_LINES = 3000


def _logs_section(csv_paths: list[str]) -> str:
    """Embed each run's full console log (command + progress + summary) in
    collapsible blocks, from the `.log` sidecar next to each CSV."""
    blocks = []
    for path in csv_paths:
        log_path = os.path.splitext(path)[0] + ".log"
        if not os.path.isfile(log_path):
            continue
        with open(log_path, errors="replace") as f:
            lines = f.read().splitlines()
        if len(lines) > _MAX_LOG_LINES:
            half = _MAX_LOG_LINES // 2
            lines = (lines[:half]
                     + ["", f"[... {len(lines) - _MAX_LOG_LINES} lines truncated — "
                            "full log in the zip bundle ...]", ""]
                     + lines[-half:])
        content = "\n".join(lines).replace("````", "```\u200b`")
        blocks.extend([
            "<details>",
            f"<summary><b><code>{os.path.basename(log_path)}</code></b> — full run output "
            "(command, progress, summary)</summary>",
            "",
            "````text",
            content,
            "````",
            "",
            "</details>",
            "",
        ])
    if not blocks:
        return ""
    return "\n".join([
        "## Run logs",
        "",
        "Complete console output of every benchmark run, including the exact "
        "command that produced it.",
        "",
        *blocks,
    ])


def _resolve_inputs(csv_paths: list[str]) -> list[tuple[list[dict], dict | None, str]]:
    """Return (rows, env, label) for each input file."""
    out = []
    for path in csv_paths:
        rows = _load_csv(path)
        json_path = os.path.splitext(path)[0] + ".json"
        payload = _load_json(json_path)
        env = payload.get("env") if payload else None
        label = os.path.basename(path)
        out.append((rows, env, label))
    return out


def generate_report(
    inputs: list[tuple[list[dict], dict | None, str]],
    chart_prefixes: list[str],
    env_overrides: dict | None = None,
    log_sources: list[str] | None = None,
) -> str:
    all_rows: list[dict] = []
    env_blocks: list[tuple[str, dict]] = []
    for rows, env, label in inputs:
        all_rows.extend(rows)
        if env:
            env_blocks.append((label, env))

    primary_env = _normalize_env(env_blocks[0][1], env_overrides) if env_blocks else None

    parts = [
        "# ShibuDB Benchmark Report",
        "",
        *_banner(primary_env),
        "Full-system performance results over TCP via the official Python client.",
        "Recall, throughput, and latency are reported together for every search workload.",
        "",
    ]

    if len(inputs) > 1:
        parts.extend([
            "### Source files",
            "",
        ])
        for _, _, label in inputs:
            parts.append(f"- `{label}`")
        parts.append("")

    if env_blocks:
        primary_rows = inputs[0][0]
        parts.append(_env_section(primary_env, primary_rows))
        if len(env_blocks) > 1:
            rows_by_label = {label: rows for rows, _, label in inputs}
            parts.extend(["", "<details>", "<summary>Additional run environments</summary>", ""])
            for label, env in env_blocks[1:]:
                parts.append(f"**{label}**")
                parts.append("")
                parts.append(_env_section(_normalize_env(env, env_overrides),
                                          rows_by_label.get(label, [])))
            parts.extend(["</details>", ""])
    else:
        parts.extend([
            "## Environment",
            "",
            "_No JSON sidecar found — re-run benchmark.py to capture environment metadata._",
            "",
        ])

    # One chart prefix per input when multiple, else single prefix for merged rows.
    if len(chart_prefixes) == 1:
        cp = chart_prefixes[0]
        for section_fn in (_vector_section, _metadata_section, _kv_section):
            block = section_fn(all_rows, cp)
            if block:
                parts.extend([block, ""])
    else:
        for (rows, _, label), cp in zip(inputs, chart_prefixes):
            parts.extend([f"## Results from `{label}`", ""])
            for section_fn in (_vector_section, _metadata_section, _kv_section):
                block = section_fn(rows, cp)
                if block:
                    parts.extend([block, ""])

    if log_sources:
        logs_block = _logs_section(log_sources)
        if logs_block:
            parts.extend([logs_block, ""])

    parts.append(METHODOLOGY)
    parts.extend([
        "",
        "---",
        "",
        "_Report generated by "
        "[`report.py`](https://github.com/shibudb-org/shibudb-server-benchmarking/blob/main/report.py). "
        "Reproduce with the commands in the repository README._",
        "",
    ])
    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more result CSV files from benchmark.py")
    ap.add_argument("--out", default="docs/BENCHMARKS.md",
                    help="Output Markdown path (default: docs/BENCHMARKS.md)")
    ap.add_argument("--charts-dir", default="",
                    help="Directory with PNGs from plot.py (copied into docs/charts/)")
    ap.add_argument("--charts-subdir", nargs="*", default=[],
                    help="Optional subdir name per --csv when using multiple result files "
                         "(e.g. --charts-dir results/charts --charts-subdir vector meta_kv)")
    ap.add_argument("--charts-dest", default="docs/charts",
                    help="Destination for copied charts (default: docs/charts)")
    ap.add_argument("--no-copy-charts", action="store_true",
                    help="Do not copy PNGs; only embed paths (use if charts already in docs/charts/)")
    ap.add_argument("--server-version", default="", help="Override ShibuDB server version in the report")
    ap.add_argument("--server-commit", default="", help="Override ShibuDB server git commit in the report")
    ap.add_argument("--server-info-file", default="",
                    help="JSON with server hardware facts (merged into the report if missing from results JSON)")
    ap.add_argument("--zip", nargs="?", const="", default=None, metavar="PATH",
                    help="Also package the report, charts, and raw CSV/JSON results into a "
                         "downloadable zip (default path: results/benchmark_report_<UTC timestamp>.zip)")
    return ap


def main():
    args = build_parser().parse_args()
    inputs = _resolve_inputs(args.csv)
    if not any(rows for rows, _, _ in inputs):
        raise SystemExit("No result rows found in the given CSV file(s).")

    env_overrides = {}
    if args.server_version:
        env_overrides["server_version"] = args.server_version
    if args.server_commit:
        env_overrides["server_commit"] = args.server_commit
    if args.server_info_file:
        with open(args.server_info_file) as f:
            env_overrides["server_info"] = json.load(f)

    chart_prefixes: list[str] = []
    if args.charts_dir and not args.no_copy_charts:
        subdirs = args.charts_subdir
        if len(args.csv) > 1:
            if subdirs and len(subdirs) != len(args.csv):
                raise SystemExit("--charts-subdir must have one entry per --csv when multiple CSVs are given")
            if not subdirs:
                subdirs = [""] * len(args.csv)
            sources = list(zip([args.charts_dir] * len(args.csv), subdirs))
        else:
            sources = [(args.charts_dir, subdirs[0] if subdirs else "")]
        _copy_charts(sources, args.charts_dest)
        for sub in (subdirs if len(args.csv) > 1 else [""]):
            rel = os.path.relpath(
                os.path.join(args.charts_dest, sub) if sub else args.charts_dest,
                os.path.dirname(os.path.abspath(args.out)) or ".",
            )
            chart_prefixes.append(rel.replace(os.sep, "/"))
    elif args.charts_dir:
        rel = os.path.relpath(args.charts_dest, os.path.dirname(os.path.abspath(args.out)) or ".")
        chart_prefixes = [rel.replace(os.sep, "/")] * len(args.csv)
    else:
        chart_prefixes = [""] * len(args.csv)

    if len(chart_prefixes) == 1 and len(args.csv) > 1:
        chart_prefixes = chart_prefixes * len(args.csv)

    report = generate_report(
        inputs,
        chart_prefixes if len(args.csv) > 1 else chart_prefixes[:1],
        env_overrides=env_overrides or None,
        log_sources=args.csv,
    )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"Wrote {args.out}", flush=True)

    if args.zip is not None:
        zip_path = args.zip
        if not zip_path:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join("results", f"benchmark_report_{stamp}.zip")
        if args.charts_dir:
            base_prefix = os.path.relpath(
                args.charts_dest, os.path.dirname(os.path.abspath(args.out)) or "."
            ).replace(os.sep, "/")
            charts_dirs, zip_prefixes = [args.charts_dest], [base_prefix]
        else:
            charts_dirs, zip_prefixes = [], []
        _create_zip(zip_path, args.out, args.csv, charts_dirs, zip_prefixes)

    print("\nGitHub Pages: commit docs/ and enable Settings → Pages → branch main → /docs", flush=True)
    print("Site home: docs/index.md  |  Report: docs/BENCHMARKS.md", flush=True)


if __name__ == "__main__":
    main()
