"""GLM-5.3's down projections as one TP8 rank's Pearl-C B rows at k = 256: does every row of every rank's
256-wide slice pass TT_OUT's per-row rules, the noise floor ρ·α ≥ 1 and liveness (at most k/64 = 4 entries with
x² ≥ 64ρ², ρ the RMS over every 8th entry)?

The weights are `mlp.experts.E.down_proj` (each routed expert's w2) and `mlp.shared_experts.down_proj` of every MoE layer,
read from a local snapshot of the pinned checkpoint.  A row is what Pearl-C on the H200 commits: each E4M3 code times its
128 x 128 block's FP32 `weight_scale_inv`, rounded to FP32 (`pearl_c_sm90/run.py`'s `fp8_block` widening), and rank r's
slice is columns [256r, 256r + 256).  The rules are computed vectorized in float64 on the FP32 values; a row within 2^-20 of
either threshold is decided again by the scheme's own `pearl_c.s5_stats` and `pearl_c.live_row`.

Writes a JSON report (per role, per layer and per rank: rows, rows failing each rule, the worst slice) and, with --samples,
an .npz of 64-row slices for `pearlc_k256_gate.py --b-samples`.

  uv run python benchmarks/pouw/pearl_c_sm90/glm53_slices.py SNAPSHOT_DIR OUT.json [--samples S.npz] [--jobs 64]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_kw as P

TP, K, BLOCK = 8, 256, 128
NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.(?:experts\.(\d+)|shared_experts)\.down_proj\.weight$")
MARGIN = 2.0 ** -20
#: the slices `--samples` keeps, (layer, expert or None for the shared expert, rank), 64 rows from a fixed offset
SAMPLES = ((3, 0, 0), (20, 17, 2), (40, 7, 3), (40, None, 5), (60, 200, 6), (77, 255, 7), (10, None, 1), (50, 128, 4))


def _mag() -> np.ndarray:
    m = np.zeros(256, dtype=np.float32)
    for c in range(128):
        e, f = c >> 3, c & 7
        m[c] = f * 2.0 ** -9 if e == 0 else (8 + f) * 2.0 ** (e - 10)
    m[0x7F] = np.nan
    m[128:] = -m[:128]
    return m


E4M3 = _mag()
DTYPES = {"F32": np.float32, "BF16": None, "F8_E4M3": np.uint8}


def header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def tensor(path: Path, name: str) -> np.ndarray:
    h, base = header(path)
    e = h[name]
    lo, hi = e["data_offsets"]
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=base + lo, shape=(hi - lo,))
    if e["dtype"] == "BF16":
        return (np.asarray(raw).view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
    return np.asarray(raw).view(DTYPES[e["dtype"]]).reshape(e["shape"])


def widen(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """code x its block's FP32 scale, RN to FP32 (one FP32 product)."""
    s = np.repeat(np.repeat(scale.astype(np.float32), BLOCK, 0), BLOCK, 1)[:codes.shape[0], :codes.shape[1]]
    return (E4M3[codes] * s).astype(np.float32)


def rules(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row of x (rows x 256, FP32): (floor_ok, live_ok, near) in float64, `near` the rows within MARGIN of a threshold."""
    v = x.astype(np.float64)
    s = np.abs(v).max(1)
    S = (v[:, ::8] ** 2).sum(1)
    rho = np.sqrt(S / (K // 8))
    alpha = 448.0 / np.maximum(s + 4 * rho, 2.0 ** -32)
    fl = rho * alpha
    sq = v * v * (K // 8)
    dead = (sq >= 64 * S[:, None]).sum(1)
    near_l = (np.abs(sq - 64 * S[:, None]) <= MARGIN * 64 * S[:, None]).any(1)
    return fl >= 1.0, 64 * dead <= K, (np.abs(fl - 1.0) <= MARGIN) | near_l


def exact(row: np.ndarray) -> tuple[bool, bool]:
    words = [int(w) for w in row.view(np.uint32)]
    _, rho, alpha, _ = C.s5_stats(words)
    return P.f32_of_bits(rho) * P.f32_of_bits(alpha) >= P.SIGMA_MIN, C.live_row(words)


def one(job: tuple[str, str]) -> dict:
    path, name = Path(job[0]), job[1]
    m = NAME.match(name)
    layer, expert = int(m.group(1)), (None if m.group(2) is None else int(m.group(2)))
    sp = Path(job[2]) if len(job) > 2 else path
    w = widen(tensor(path, name), tensor(sp, name + "_scale_inv"))
    out = {"layer": layer, "expert": expert, "shape": list(w.shape), "ranks": [], "samples": {}}
    for r in range(TP):
        x = np.ascontiguousarray(w[:, K * r:K * r + K])
        f_ok, l_ok, near = rules(x)
        for i in np.flatnonzero(near):
            f_ok[i], l_ok[i] = exact(x[i])
        zero = ~(x != 0).any(1)
        out["ranks"].append({"rank": r, "rows": int(x.shape[0]), "floor_fail": int((~f_ok).sum()), "live_fail": int((~l_ok).sum()),
                             "either_fail": int((~(f_ok & l_ok)).sum()), "zero_rows": int(zero.sum()), "near": int(near.sum())})
        if (layer, expert, r) in SAMPLES:
            off = (7919 * (layer + 1) * (r + 1)) % (x.shape[0] - 64)
            out["samples"][f"L{layer}-{'s' if expert is None else f'e{expert}'}-r{r}"] = x[off:off + 64]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--samples", default="")
    ap.add_argument("--jobs", type=int, default=64)
    a = ap.parse_args()
    snap = Path(a.snapshot)
    cfg = json.loads((snap / "config.json").read_text())
    wm = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    jobs = [(str(snap / f), n, str(snap / wm[n + "_scale_inv"])) for n, f in sorted(wm.items()) if NAME.match(n)]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        res = pool.map(one, jobs, chunksize=4)
    layers_main = cfg.get("num_hidden_layers")
    agg: dict = defaultdict(lambda: {"slices": 0, "rows": 0, "floor_fail": 0, "live_fail": 0, "either_fail": 0, "zero_rows": 0,
                                     "near": 0, "slices_with_a_failing_row": 0, "slices_over_1_64": 0, "worst_slice": None})
    by_layer: dict = defaultdict(lambda: {"rows": 0, "either_fail": 0})
    by_rank: dict = defaultdict(lambda: {"rows": 0, "either_fail": 0})
    samples = {}
    for t in res:
        role = ("routed" if t["expert"] is not None else "shared") + ("" if layers_main is None or t["layer"] < layers_main else "-mtp")
        for rk in t["ranks"]:
            g = agg[role]
            g["slices"] += 1
            for x in ("rows", "floor_fail", "live_fail", "either_fail", "zero_rows", "near"):
                g[x] += rk[x]
            g["slices_with_a_failing_row"] += rk["either_fail"] > 0
            g["slices_over_1_64"] += rk["either_fail"] * 64 > rk["rows"]
            share = rk["either_fail"] / rk["rows"]
            if g["worst_slice"] is None or share > g["worst_slice"]["share"]:
                g["worst_slice"] = {"layer": t["layer"], "expert": t["expert"], "rank": rk["rank"], "share": share,
                                    "floor_fail": rk["floor_fail"], "live_fail": rk["live_fail"]}
            by_layer[f"{role}/{t['layer']}"]["rows"] += rk["rows"]
            by_layer[f"{role}/{t['layer']}"]["either_fail"] += rk["either_fail"]
            by_rank[f"{role}/{rk['rank']}"]["rows"] += rk["rows"]
            by_rank[f"{role}/{rk['rank']}"]["either_fail"] += rk["either_fail"]
        samples.update(t["samples"])
    for g in agg.values():
        g["fail_share"] = g["either_fail"] / g["rows"] if g["rows"] else 0.0
    rows = sum(g["rows"] for g in agg.values())
    fail = sum(g["either_fail"] for g in agg.values())
    doc = {"tool": "glm53_slices", "snapshot": str(snap), "tp": TP, "k": K, "tensors": len(res), "num_hidden_layers": layers_main,
           "rows": rows, "either_fail": fail, "fail_share": fail / rows if rows else 0.0, "roles": dict(agg),
           "by_layer": dict(by_layer), "by_rank": dict(by_rank), "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    if a.samples:
        np.savez(a.samples, **samples)
    print(json.dumps({x: v for x, v in doc.items() if x not in ("by_layer",)}, indent=1))
    sys.exit(0 if rows else 1)


if __name__ == "__main__":
    main()
