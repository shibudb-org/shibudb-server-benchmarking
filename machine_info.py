"""Auto-detect this machine's hardware and emit a server-info JSON.

Stdlib-only — copy this single file to the ShibuDB server host and run it
there to generate the file consumed by `benchmark.py --server-info-file`
(no pip install needed).

Usage:
  python3 machine_info.py                            # print JSON to stdout
  python3 machine_info.py --out server-info.json     # write to a file
  python3 machine_info.py --server-repo ~/src/shibudb-server \
      --out server-info.json                         # also fill version/commit from git
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess


def cpu_model() -> str:
    """Human-readable CPU model (e.g. 'Apple M2 Pro'), best effort."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            name = out.stdout.strip()
            if name:
                return name
        except (OSError, subprocess.SubprocessError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def total_ram_gb() -> float | None:
    """Total physical RAM of this machine in GB, or None if undetectable."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS fallback
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        return round(int(out.stdout.strip()) / (1024 ** 3), 1)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def disk_summary(path: str = "/") -> str | None:
    """Total size of the filesystem holding `path`, best effort."""
    try:
        total = shutil.disk_usage(path).total
        return f"{total / (1024 ** 3):.0f} GB total ({path})"
    except OSError:
        return None


def _git(repo: str, *argv: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", os.path.expanduser(repo), *argv],
                             capture_output=True, text=True, timeout=10)
        val = out.stdout.strip()
        return val or None
    except (OSError, subprocess.SubprocessError):
        return None


def collect(server_repo: str | None = None) -> dict:
    """Detect hardware facts for this machine, in server-info.json schema."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "disk": disk_summary(),
        "notes": "auto-detected by machine_info.py",
    }
    if server_repo:
        version = _git(server_repo, "describe", "--tags", "--always")
        commit = _git(server_repo, "rev-parse", "HEAD")
        if version:
            info["version"] = version
        if commit:
            info["commit"] = commit
    return {k: v for k, v in info.items() if v is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="", help="write JSON here (default: stdout)")
    ap.add_argument("--server-repo", default="",
                    help="path to a shibudb-server git checkout; fills version/commit")
    args = ap.parse_args()

    info = collect(args.server_repo or None)
    payload = json.dumps(info, indent=2) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        print(f"Wrote {args.out}")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
