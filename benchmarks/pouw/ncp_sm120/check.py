"""Off the card: run.py's outputs against NCP-FP8's chain recomputed from verity's sm_120 atom and verity's BLAKE3.

Every check run (OUT/check/, small shapes) is compared whole: C (plain: one accumulator chained over the first K columns;
every other arm: R_(T-1)) and, for the hash arms, every lane's digest.  Every timed arm's sampled lanes (OUT/arms.json, the
full shapes) are compared: their C cells and digests.  Operands are regenerated from gen_codes's rule (ncp_sm120.cu).

  uv run python benchmarks/pouw/ncp_sm120/check.py OUT     exit 0 when every comparison matches, 1 otherwise
"""

from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

from verity.commitments import blake3
from verity.ml.kernels import group_sum_total_batch
from verity.ml.tc.models import BLACKWELL_SM120_E4M3_M16N8K32 as ATOM

NEVER = 1 << 30


def codes(seed: int, rows: np.ndarray, K: int, cols: slice = slice(None)) -> np.ndarray:
    """gen_codes's bytes for the given rows of a (· x K) operand: code(idx) of h = mix32(idx * 0x9E3779B9 + seed)."""
    idx = (rows.astype(np.uint64)[:, None] * np.uint64(K) + np.arange(K, dtype=np.uint64)[None, cols]).astype(np.uint32)
    x = idx * np.uint32(0x9E3779B9) + np.uint32(seed)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return ((x & 0x80) | ((4 + ((x >> 8) & 7)) << 3) | ((x >> 16) & 7)).astype(np.uint8)


def operands(meta: dict, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """X~'s rows and Y~'s rows (padding rows of X~ zero)."""
    K = meta["K"]
    a = codes(meta["seeds"][0], rows, K)
    a[rows >= meta["rows"]] = 0
    return a, codes(meta["seeds"][1], cols, K)


def chain(a: np.ndarray, b: np.ndarray, atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Per output (a[i], b[i]): I (outputs x atoms) and R (outputs x atoms), FP32 bit patterns."""
    n = a.shape[0]
    seg_a = a[:, : 32 * atoms].reshape(n * atoms, 32).astype(np.int64)
    seg_b = b[:, : 32 * atoms].reshape(n * atoms, 32).astype(np.int64)
    words, _ = group_sum_total_batch(ATOM, np.zeros(n * atoms, dtype=np.uint32), seg_a, seg_b)
    I = words.astype(np.uint32).reshape(n, atoms)
    R = np.empty_like(I)
    acc = I[:, 0].view(np.float32).copy()
    R[:, 0] = acc.view(np.uint32)
    for t in range(1, atoms):
        acc = (acc + I[:, t].view(np.float32)).astype(np.float32)
        R[:, t] = acc.view(np.uint32)
    return I, R


def plain(a: np.ndarray, b: np.ndarray, atoms: int) -> np.ndarray:
    acc = np.zeros(a.shape[0], dtype=np.uint32)
    for t in range(atoms):
        acc, _ = group_sum_total_batch(ATOM, acc, a[:, 32 * t:32 * t + 32].astype(np.int64),
                                       b[:, 32 * t:32 * t + 32].astype(np.int64))
        acc = acc.astype(np.uint32)
    return acc


def lane_cells(meta: dict, tm: int, tn: int, tid: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """A lane's outputs (rows, cols) in fragment order (mi, ni, e), and whether its warp holds a row that exists."""
    lay = meta["layout"]
    warp, lane = tid // 32, tid % 32
    wm, wn, g, q = warp // lay["warps_n"], warp % lay["warps_n"], lane >> 2, lane & 3
    rows, cols = [], []
    for mi in range(lay["mi"]):
        for ni in range(lay["ni"]):
            for e in range(4):
                rows.append(tm * meta["tm"] + wm * lay["wm"] + 16 * mi + g + 8 * (e >> 1))
                cols.append(tn * meta["bn"] + wn * 32 + 8 * ni + 2 * q + (e & 1))
    live = tm * meta["tm"] + wm * lay["wm"] < meta["rows"]
    return np.array(rows), np.array(cols), live


def digest(I: np.ndarray, R: np.ndarray, tile: int, tid: int, S: int, words: int, live: bool) -> bytes:
    """The lane's leaf: BLAKE3 compressions chained over its bound atoms' R words then I words (16 a block, fragment order),
    keyed by IV xor (tile, tid, S | words << 16, 0, ...); I and R are (lane outputs x atoms)."""
    cv = list(blake3.IV)
    cv[0] ^= tile
    cv[1] ^= tid
    cv[2] ^= (S | words << 16) & 0xFFFFFFFF
    ctr = 0
    if live:
        for t in range(I.shape[1]):
            if (t + 1) % S:
                continue
            for w in ([R[:, t], I[:, t]] if words else [R[:, t]]):
                for b in range(0, len(w), 16):
                    cv = blake3.compress(tuple(cv), struct.pack("<16I", *(int(v) for v in w[b:b + 16])), ctr, 64,
                                         blake3.KEYED_HASH)[:8]
                    ctr += 1
    return struct.pack("<8I", *cv)


def check_small(out: str, kind: str) -> list[str]:
    meta = json.load(open(os.path.join(out, "check", f"{kind}.json")))
    m, n, K = meta["m"], meta["n"], meta["K"]
    T = K // 32
    rr, cc = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    a, b = operands(meta, np.arange(m), np.arange(n))
    A, B = a[rr.ravel()], b[cc.ravel()]
    I, R = chain(A, B, T)
    want_r = R[:, -1].reshape(m, n)
    fails = []
    tiles_n = n // meta["bn"]
    for run in meta["runs"]:
        got = np.frombuffer(open(os.path.join(out, "check", run["tag"] + ".c.bin"), "rb").read(), dtype=np.uint32).reshape(m, n)
        want = plain(A, B, run["K"] // 32).reshape(m, n) if run["arm"] == "plain" else want_r
        bad = int((got != want).sum())
        if bad:
            fails.append(f"{run['tag']}: C {bad} of {m * n} words differ")
        if run["arm"].startswith("hash"):
            dg = open(os.path.join(out, "check", run["tag"] + ".dg.bin"), "rb").read()
            words, wrong = int(run["arm"] == "hash_ir"), 0
            for t in range(meta["tiles"]):
                tm, tn = t // tiles_n, t % tiles_n
                for tid in range(meta["nct"]):
                    rows, cols, live = lane_cells(meta, tm, tn, tid)
                    sel = rows * n + cols
                    d = digest(I[sel], R[sel], t, tid, run["S"], words, live)
                    wrong += d != dg[(t * meta["nct"] + tid) * 32:(t * meta["nct"] + tid + 1) * 32]
            if wrong:
                fails.append(f"{run['tag']}: {wrong} of {meta['tiles'] * meta['nct']} lane digests differ")
        print(f"check {run['tag']}: {'ok' if not fails or not fails[-1].startswith(run['tag']) else fails[-1]}", flush=True)
    return fails


def check_arms(out: str) -> list[str]:
    res = json.load(open(os.path.join(out, "arms.json")))
    fails = []
    for shape in res["shapes"]:
        n, T = shape["n"], shape["T"]
        cache = {}
        for arm in shape["arms"]:
            for ln in arm["lanes"]:
                key = (ln["tm"], ln["tn"], ln["tid"])
                if key not in cache:
                    rows, cols, live = lane_cells(shape, *key)
                    a, b = operands(shape, rows, cols)
                    I, R = chain(a, b, T)
                    cache[key] = (rows, cols, live, a, b, I, R)
                rows, cols, live, a, b, I, R = cache[key]
                got = np.array([int(h, 16) for h in ln["cells"]], dtype=np.uint32)
                want = plain(a, b, arm["K"] // 32) if arm["arm"] == "plain" else R[:, -1]
                tag = f"{shape['kind']} {arm['arm']} S={arm['S']} K={arm['K']} tile {ln['tile']} lane {ln['tid']}"
                if (got != want).any():
                    fails.append(f"{tag}: {int((got != want).sum())} of {len(got)} cells differ")
                if "digest" in ln:
                    d = digest(I, R, ln["tile"], ln["tid"], arm["S"], int(arm["arm"] == "hash_ir"), live)
                    if d.hex() != ln["digest"]:
                        fails.append(f"{tag}: digest differs")
            print(f"arms {shape['kind']} {arm['arm']} S={arm['S']} K={arm['K']}: {len(arm['lanes'])} lanes checked", flush=True)
    return fails


def main(argv: list[str]) -> int:
    out = argv[1]
    fails = []
    for kind in ("p", "d"):
        if os.path.exists(os.path.join(out, "check", f"{kind}.json")):
            fails += check_small(out, kind)
    if os.path.exists(os.path.join(out, "arms.json")):
        fails += check_arms(out)
    for f in fails:
        print("FAIL", f)
    print("all match" if not fails else f"{len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
