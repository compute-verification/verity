"""The gpt2 training stream's duplication, before any GPU screen of deduplication (CPU, numpy; reads only TrainData's file).

    python stream_stats.py [--windows 300000] [--gram 32] [--every 16] [--seed 1] --out stats.json

For the first `--windows` windows of TrainData's seed order (a full-budget s plan is ~304k), it hashes every `--gram`-token
n-gram, keeps the ones whose hash is 0 mod `--every` (content-defined anchors, so two copies of a passage share anchors at any
offset), and reports how many anchors occur more than once and the share of windows whose anchors are mostly repeats, at
several thresholds; plus the stream's tokens, documents (end-of-text tokens) and exact duplicate windows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import prepare  # noqa: E402

CTX = prepare.EVAL_CTX
MULT = np.uint64(0x9E3779B97F4A7C15)


def anchors(rows: np.ndarray, gram: int, every: int) -> tuple[np.ndarray, np.ndarray]:
    """(hash, row index) of every anchored n-gram of each row."""
    t = rows.astype(np.uint64)
    n = t.shape[1] - gram + 1
    h = np.zeros((t.shape[0], n), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(gram):
            h = h * MULT + t[:, j: j + n] + np.uint64(1)
        h ^= h >> np.uint64(29)
        h *= MULT
    keep = (h % np.uint64(every)) == 0
    r, _ = np.nonzero(keep)
    return h[keep], r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", type=int, default=300_000)
    ap.add_argument("--gram", type=int, default=32)
    ap.add_argument("--every", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.perf_counter()
    data = prepare.TrainData("gpt2", seed=a.seed)
    tok = data.tokens
    eot = 0
    for i in range(0, len(tok), 1 << 27):
        eot += int(np.count_nonzero(tok[i: i + (1 << 27)] == prepare.EOT["gpt2"]))
    ids = np.sort(data.order[: a.windows])
    hs, owners, exact = [], [], []
    for i in range(0, len(ids), a.chunk):
        idx = ids[i: i + a.chunk]
        rows = np.stack([tok[j * CTX: j * CTX + CTX] for j in idx])
        h, r = anchors(rows, a.gram, a.every)
        hs.append(h)
        owners.append(r + i)
        with np.errstate(over="ignore"):
            exact.append((rows.astype(np.uint64) * MULT).sum(1) ^ rows[:, :8].astype(np.uint64).sum(1))
    h = np.concatenate(hs)
    own = np.concatenate(owners)
    uniq, inv, counts = np.unique(h, return_inverse=True, return_counts=True)
    first = np.full(len(uniq), len(ids), dtype=np.int64)
    np.minimum.at(first, inv, own)
    repeat = own != first[inv]
    per_window = np.bincount(own, minlength=len(ids)).astype(np.float64)
    rep_window = np.bincount(own, weights=repeat.astype(np.float64), minlength=len(ids))
    share = np.divide(rep_window, per_window, out=np.zeros_like(rep_window), where=per_window > 0)
    ex = np.concatenate(exact)
    out = {"stream_tokens": int(len(tok)), "windows_total": int(len(data.order)), "documents": eot,
           "mean_doc_tokens": len(tok) / max(1, eot), "sampled_windows": int(len(ids)), "gram": a.gram, "every": a.every,
           "anchors": int(len(h)), "distinct_anchors": int(len(uniq)), "repeated_anchor_share": float(repeat.mean()),
           "windows_with_repeat_share_over": {str(q): float((share > q).mean()) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
           "repeated_tokens_estimate": float((share * 1).mean()),
           "exact_duplicate_windows": int(len(ex) - len(np.unique(ex))), "seconds": round(time.perf_counter() - t0, 1)}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
