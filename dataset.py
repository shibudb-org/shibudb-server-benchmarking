"""SIFT1M dataset download and loading helpers.

SIFT1M (a.k.a. ANN_SIFT1M) is the standard 128-dimensional benchmark dataset
used across the vector-search literature (ANN-Benchmarks, FAISS, etc.). It ships
with precomputed ground-truth nearest neighbors, which is what makes recall
measurable and the results credible.

Layout after download/extract (under --data-dir):
    sift/sift_base.fvecs        1,000,000 x 128  float32   (the database)
    sift/sift_query.fvecs          10,000 x 128  float32   (query vectors)
    sift/sift_groundtruth.ivecs    10,000 x 100  int32     (true neighbor ids)
    sift/sift_learn.fvecs         100,000 x 128  float32   (training set; unused here)

The .fvecs/.ivecs format is: for each vector, one int32 dimension `d` followed
by `d` values (float32 for fvecs, int32 for ivecs), packed back-to-back.
"""

from __future__ import annotations

import os
import sys
import tarfile
import urllib.request

import numpy as np

# Default mirror. The canonical source is the TEXMEX corpus at IRISA (FTP).
# Override with --dataset-url if this is slow/unavailable in your environment.
DEFAULT_SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"


def read_fvecs(path: str) -> np.ndarray:
    """Read a .fvecs file into an (n, d) float32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty fvecs file: {path}")
    d = int(raw[0])
    arr = raw.reshape(-1, d + 1)[:, 1:]
    return arr.copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    """Read an .ivecs file into an (n, d) int32 array."""
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"empty ivecs file: {path}")
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def _download(url: str, dest: str) -> None:
    print(f"Downloading {url} -> {dest}", flush=True)

    def _progress(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100.0 * done / total_size
        sys.stdout.write(f"\r  {done / 1e6:8.1f} MB / {total_size / 1e6:.1f} MB ({pct:5.1f}%)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, _progress)
    sys.stdout.write("\n")


def ensure_sift(data_dir: str, url: str = DEFAULT_SIFT_URL) -> dict:
    """Ensure SIFT1M is present under data_dir, downloading/extracting if needed.

    Returns a dict of resolved file paths: base, query, groundtruth.
    """
    os.makedirs(data_dir, exist_ok=True)
    sift_dir = os.path.join(data_dir, "sift")
    base = os.path.join(sift_dir, "sift_base.fvecs")
    query = os.path.join(sift_dir, "sift_query.fvecs")
    gt = os.path.join(sift_dir, "sift_groundtruth.ivecs")

    if not (os.path.exists(base) and os.path.exists(query) and os.path.exists(gt)):
        tarball = os.path.join(data_dir, "sift.tar.gz")
        if not os.path.exists(tarball):
            _download(url, tarball)
        print(f"Extracting {tarball}", flush=True)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(data_dir)

    for label, p in (("base", base), ("query", query), ("groundtruth", gt)):
        if not os.path.exists(p):
            raise FileNotFoundError(f"expected {label} file not found after extract: {p}")

    return {"base": base, "query": query, "groundtruth": gt}


def load_sift(data_dir: str, url: str = DEFAULT_SIFT_URL):
    """Load SIFT1M into memory. Returns (base, query, groundtruth) numpy arrays."""
    paths = ensure_sift(data_dir, url)
    base = read_fvecs(paths["base"])
    query = read_fvecs(paths["query"])
    gt = read_ivecs(paths["groundtruth"])
    print(
        f"Loaded SIFT1M: base={base.shape} query={query.shape} groundtruth={gt.shape}",
        flush=True,
    )
    return base, query, gt


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download/verify the SIFT1M dataset.")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/.shibudb-benchmarks/data"))
    ap.add_argument("--dataset-url", default=DEFAULT_SIFT_URL)
    args = ap.parse_args()

    b, q, g = load_sift(args.data_dir, args.dataset_url)
    print("OK")
