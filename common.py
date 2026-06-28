"""Shared utilities for the ShibuDB benchmark suites.

Provides: client construction, a concurrent phase runner (one dedicated
connection per worker), latency/throughput summarization, exact ground-truth
computation (numpy), synthetic metadata generation, a unified result schema,
and reproducible environment capture.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np

try:
    from shibudb_client import ShibuDbClient
except ImportError:  # pragma: no cover
    raise SystemExit("shibudb-client is not installed. Run: pip install -r requirements.txt")

# The SDK logs every connect/close at INFO; silence it (noise + per-op overhead).
logging.getLogger("shibudb_client").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #
def make_client(args, space: str | None = None) -> ShibuDbClient:
    client = ShibuDbClient(args.host, args.port, timeout=args.timeout)
    client.authenticate(args.user, args.password)
    if space:
        client.use_space(space)
    return client


# --------------------------------------------------------------------------- #
# Concurrent phase runner
# --------------------------------------------------------------------------- #
def run_phase(args, space: str, n: int, concurrency: int, task_fn, collect: bool = False):
    """Execute `task_fn(client, i)` for i in [0, n) across `concurrency` workers.

    Each worker owns one authenticated connection (the SDK serializes I/O per
    connection, so concurrency == number of connections). Per-call latency is
    timed. `task_fn` should raise on failure. If `collect`, the per-item return
    value of `task_fn` is stored and returned.

    Returns: (latencies_ms[list], results[list|None], failed[int], wall_seconds).
    """
    latencies = [0.0] * n
    results = [None] * n if collect else None
    failed = {"n": 0}
    lock = threading.Lock()

    def worker(wid: int):
        client = make_client(args, space)
        try:
            for i in range(wid, n, concurrency):
                t0 = time.perf_counter()
                try:
                    r = task_fn(client, i)
                    latencies[i] = (time.perf_counter() - t0) * 1000.0
                    if collect:
                        results[i] = r
                except Exception:
                    latencies[i] = (time.perf_counter() - t0) * 1000.0
                    with lock:
                        failed["n"] += 1
        finally:
            client.close()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(concurrency)))
    wall = time.perf_counter() - t0
    return latencies, results, failed["n"], wall


def summarize(latencies: list[float], wall: float, n: int) -> dict:
    lat = sorted(latencies)

    def pct(p):
        if not lat:
            return 0.0
        idx = min(len(lat) - 1, int(round(p / 100.0 * (len(lat) - 1))))
        return lat[idx]

    return {
        "throughput_ops_sec": (n / wall) if wall > 0 else 0.0,
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
    }


# --------------------------------------------------------------------------- #
# Server resource monitoring (CPU / memory)
# --------------------------------------------------------------------------- #
try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def find_server_process(pid: int | None = None, name_match: str = "shibudb"):
    """Locate the ShibuDB server process to monitor.

    Returns a psutil.Process or None. Resource monitoring only works when the
    server runs on the SAME host as this benchmark client (or when an explicit
    --server-pid is given). For a remote server, sample on that machine instead.
    """
    if psutil is None:
        return None
    if pid is not None:
        try:
            return psutil.Process(pid)
        except Exception:
            return None
    me = os.getpid()
    candidates = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.pid == me:
                continue
            hay = (p.info.get("name") or "") + " " + " ".join(p.info.get("cmdline") or [])
            if name_match.lower() in hay.lower():
                candidates.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None

    def _rss(p):
        try:
            return p.memory_info().rss
        except Exception:
            return 0

    # If multiple match (e.g. a launcher + the server), pick the heaviest one.
    return max(candidates, key=_rss)


@dataclass
class ResourceSample:
    t: float                  # seconds since monitor start
    cpu_percent: float        # server-process CPU% (can exceed 100 on multicore)
    rss_mb: float             # server-process resident memory (MB)
    sys_cpu_percent: float    # whole-host CPU%
    sys_mem_used_mb: float    # whole-host memory used (MB)


# Module-level handle so suites can annotate phases without plumbing the monitor
# object through every call. Set by ResourceMonitor.start(), cleared by stop().
_active_monitor: "ResourceMonitor | None" = None


def mark_phase(label: str) -> None:
    """Record a labeled time marker on the active monitor (no-op if none)."""
    if _active_monitor is not None:
        _active_monitor.mark(label)


class ResourceMonitor:
    """Background sampler of a server process's CPU% and RSS over time.

    Samples every `interval` seconds in a daemon thread until stopped, keeping a
    full time-series plus optional phase markers. Saving writes a `.resources.csv`
    (time-series) and `.resources.json` (samples + marks + summary).
    """

    def __init__(self, proc, interval: float = 0.5):
        self.proc = proc
        self.interval = interval
        self.samples: list[ResourceSample] = []
        self.marks: list[tuple[float, str]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0: float | None = None

    @property
    def active(self) -> bool:
        return self.proc is not None and psutil is not None

    def start(self) -> "ResourceMonitor":
        global _active_monitor
        if not self.active:
            return self
        _active_monitor = self
        self._t0 = time.perf_counter()
        # Prime cpu_percent counters (first call always returns 0.0).
        try:
            self.proc.cpu_percent(None)
        except Exception:
            pass
        psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def mark(self, label: str) -> None:
        if self.active and self._t0 is not None:
            self.marks.append((time.perf_counter() - self._t0, label))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            t = time.perf_counter() - self._t0
            try:
                with self.proc.oneshot():
                    cpu = self.proc.cpu_percent(None)
                    rss = self.proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            vm = psutil.virtual_memory()
            self.samples.append(ResourceSample(
                t=t,
                cpu_percent=cpu,
                rss_mb=rss,
                sys_cpu_percent=psutil.cpu_percent(None),
                sys_mem_used_mb=(vm.total - vm.available) / (1024 * 1024),
            ))

    def stop(self) -> "ResourceMonitor":
        global _active_monitor
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.interval * 2 + 1)
        _active_monitor = None
        return self

    def summary(self) -> dict:
        if not self.samples:
            return {}
        cpu = [s.cpu_percent for s in self.samples]
        rss = [s.rss_mb for s in self.samples]
        return {
            "samples": len(self.samples),
            "duration_s": self.samples[-1].t,
            "interval_s": self.interval,
            "ncpu": psutil.cpu_count() if psutil else None,
            "cpu_percent_mean": statistics.fmean(cpu),
            "cpu_percent_peak": max(cpu),
            "rss_mb_mean": statistics.fmean(rss),
            "rss_mb_peak": max(rss),
        }

    def save(self, path: str) -> None:
        import csv as _csv

        if not self.samples:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["t_seconds", "proc_cpu_percent", "proc_rss_mb",
                        "sys_cpu_percent", "sys_mem_used_mb"])
            for s in self.samples:
                w.writerow([f"{s.t:.3f}", f"{s.cpu_percent:.2f}", f"{s.rss_mb:.2f}",
                            f"{s.sys_cpu_percent:.2f}", f"{s.sys_mem_used_mb:.2f}"])
        json_path = os.path.splitext(path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump({
                "summary": self.summary(),
                "marks": [{"t": t, "label": lbl} for t, lbl in self.marks],
                "samples": [asdict(s) for s in self.samples],
            }, f, indent=2)


# --------------------------------------------------------------------------- #
# Search-result parsing
# --------------------------------------------------------------------------- #
def parse_search_ids(resp: dict) -> list[int]:
    """Extract the ordered result ids from a search_topk/range_search response.

    The server returns results as a JSON string in `message`, e.g.
    '[{"id": 12, "distance": 0.34}, ...]'.
    """
    payload = resp.get("message", resp)
    if isinstance(payload, str):
        payload = json.loads(payload) if payload.strip() else []
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    ids = []
    for item in payload:
        if isinstance(item, dict) and "id" in item:
            ids.append(int(item["id"]))
        elif isinstance(item, (int, float)):
            ids.append(int(item))
    return ids


# --------------------------------------------------------------------------- #
# Exact ground-truth (for recall) via batched numpy
# --------------------------------------------------------------------------- #
def compute_ground_truth(base: np.ndarray, queries: np.ndarray, k: int,
                         metric: str = "L2", candidate_ids: np.ndarray | None = None,
                         batch: int = 64) -> np.ndarray:
    """Exact top-k neighbor ids per query (ids are indices into `base`).

    If `candidate_ids` is given, the search is restricted to that subset (used
    for filtered-search recall). Supports L2 (smaller is better) and
    InnerProduct (larger is better).
    """
    if candidate_ids is None:
        cand = base
        cand_ids = np.arange(base.shape[0])
    else:
        cand_ids = np.asarray(candidate_ids)
        cand = base[cand_ids]

    m = cand.shape[0]
    kk = min(k, m)
    out = np.full((queries.shape[0], k), -1, dtype=np.int64)
    cand_sq = np.sum(cand * cand, axis=1)[None, :] if metric != "InnerProduct" else None

    for i in range(0, queries.shape[0], batch):
        qb = queries[i:i + batch]
        if metric == "InnerProduct":
            score = qb @ cand.T          # larger is better
            part = np.argpartition(-score, kth=kk - 1, axis=1)[:, :kk]
            order = np.argsort(-np.take_along_axis(score, part, axis=1), axis=1)
        else:
            score = -2.0 * (qb @ cand.T) + cand_sq   # rank by squared L2
            part = np.argpartition(score, kth=kk - 1, axis=1)[:, :kk]
            order = np.argsort(np.take_along_axis(score, part, axis=1), axis=1)
        topk_local = np.take_along_axis(part, order, axis=1)
        out[i:i + qb.shape[0], :kk] = cand_ids[topk_local]
    return out


def recall_at_k(returned_ids: list[int], gt_row: np.ndarray, k: int) -> float:
    gt = set(int(x) for x in gt_row[:k] if x >= 0)
    if not gt:
        return 0.0
    hit = len(gt.intersection(returned_ids[:k]))
    return hit / len(gt)


# --------------------------------------------------------------------------- #
# Synthetic metadata (for metadata-filter benchmarks)
# --------------------------------------------------------------------------- #
N_CATEGORIES = 10
N_YEARS = 25
PRICE_RANGE = 1000


def metadata_for_id(i: int) -> dict:
    """Deterministic metadata for vector id i (reproducible across runs)."""
    return {
        "category": f"cat_{i % N_CATEGORIES}",
        "price": float((i * 37) % PRICE_RANGE),
        "year": 2000 + (i % N_YEARS),
    }


def metadata_field_spec() -> dict:
    return {"category": "string", "price": "float", "year": "int"}


def metadata_arrays(n: int):
    """Vectorized metadata for ids [0, n) for fast filtered-GT computation."""
    ids = np.arange(n)
    return {
        "category": ids % N_CATEGORIES,
        "price": (ids * 37) % PRICE_RANGE,
        "year": 2000 + (ids % N_YEARS),
    }


# Filter scenarios: (label, where-string, predicate over metadata_arrays -> bool mask).
def filter_scenarios(n: int) -> list[dict]:
    md = metadata_arrays(n)
    return [
        {
            "label": "category_eq",
            "where": "category=cat_0",
            "mask": md["category"] == 0,
            "selectivity": 1.0 / N_CATEGORIES,
        },
        {
            "label": "year_between",
            "where": "year BETWEEN 2010 AND 2014",
            "mask": (md["year"] >= 2010) & (md["year"] <= 2014),
            "selectivity": 5.0 / N_YEARS,
        },
        {
            "label": "price_lt_500_and_cat_in",
            "where": "price<500 AND category IN (cat_0, cat_1, cat_2)",
            "mask": (md["price"] < 500) & (md["category"] < 3),
            "selectivity": 0.5 * (3.0 / N_CATEGORIES),
        },
    ]


# --------------------------------------------------------------------------- #
# Unified result schema + writer
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    suite: str                # vector_search | metadata_filter | key_value
    operation: str            # insert | search | filtered_search | put | get | delete
    engine: str               # vector | key-value
    index_type: str = ""
    metric: str = ""
    wal: bool = False
    num_base: int = 0
    k: int = 0
    concurrency: int = 0
    scenario: str = ""        # filter label (metadata suite)
    selectivity: float = -1.0
    ops: int = 0
    throughput_ops_sec: float = 0.0
    recall_at_k: float = -1.0  # -1 == not applicable
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    failed: int = 0


class ResultWriter:
    """Accumulates Result rows and saves CSV + JSON (with env) incrementally."""

    def __init__(self, out_path: str, env: dict):
        self.out_path = out_path
        self.env = env
        self.rows: list[Result] = []
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    def add(self, r: Result):
        self.rows.append(r)
        self.flush()

    def flush(self):
        import csv

        if not self.rows:
            return
        fields = list(asdict(self.rows[0]).keys())
        with open(self.out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.rows:
                w.writerow(asdict(r))
        json_path = os.path.splitext(self.out_path)[0] + ".json"
        with open(json_path, "w") as f:
            json.dump({"env": self.env, "results": [asdict(r) for r in self.rows]}, f, indent=2)


# --------------------------------------------------------------------------- #
# Environment capture (reproducibility)
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _client_version() -> str:
    try:
        import shibudb_client
        return getattr(shibudb_client, "__version__", "unknown")
    except Exception:
        return "unknown"


def capture_env(args) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "shibudb_client_version": _client_version(),
        "client_host": socket.gethostname(),
        "server_host": args.host,
        "server_port": args.port,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "dataset": "SIFT1M",
        "args": {k: v for k, v in vars(args).items() if k != "password"},
    }


# --------------------------------------------------------------------------- #
# Space helpers
# --------------------------------------------------------------------------- #
def create_space(args, space: str, engine_type: str, **kwargs) -> None:
    client = make_client(args)
    try:
        if getattr(args, "drop_existing", False):
            try:
                client.delete_space(space)
            except Exception:
                pass
        client.create_space(space, engine_type=engine_type, **kwargs)
    finally:
        client.close()


def wait_for_searchable(args, space: str, sample_ids: list[int], vector: bool = True) -> float:
    """Block until a sample of inserted ids is retrievable (persistence lag)."""
    client = make_client(args, space)
    t0 = time.perf_counter()
    deadline = t0 + args.settle_timeout
    try:
        while time.perf_counter() < deadline:
            ok = True
            for sid in sample_ids:
                try:
                    if vector:
                        client.get_vector(sid)
                    else:
                        client.get(str(sid))
                except Exception:
                    ok = False
                    break
            if ok:
                return time.perf_counter() - t0
            time.sleep(1.0)
    finally:
        client.close()
    print(f"[settle] WARNING: sample not retrievable within {args.settle_timeout}s", flush=True)
    return time.perf_counter() - t0
