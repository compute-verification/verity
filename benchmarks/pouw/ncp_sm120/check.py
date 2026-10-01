"""Off the card: run.py's outputs against NCP-FP8's chain recomputed from verity's sm_120 atom and verity's BLAKE3.

Every check run (OUT/check/, small shapes) is compared whole: C (plain and chash: one accumulator chained over the first K
columns; every other arm: R_(T-1)) and, for the hashing arms, every lane's digest.  Every timed arm's sampled lanes (OUT/arms.json, the
full shapes) are compared: their C cells and digests.  The negative control (an arms.json row `control`: C and the
digests poisoned, nothing launched) must fail every comparison against the arm it stands as (`as`).  Operands are
regenerated from gen_codes's rule (ncp_sm120.cu).

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


def chained(a: np.ndarray, b: np.ndarray, atoms: int) -> np.ndarray:
    """The plain chained accumulator after each atom (outputs x atoms), FP32 bit patterns."""
    acc = np.zeros(a.shape[0], dtype=np.uint32)
    out = np.empty((a.shape[0], atoms), dtype=np.uint32)
    for t in range(atoms):
        acc, _ = group_sum_total_batch(ATOM, acc, a[:, 32 * t:32 * t + 32].astype(np.int64),
                                       b[:, 32 * t:32 * t + 32].astype(np.int64))
        acc = acc.astype(np.uint32)
        out[:, t] = acc
    return out


def plain(a: np.ndarray, b: np.ndarray, atoms: int) -> np.ndarray:
    return chained(a, b, atoms)[:, -1]


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


WORDS = {"hash_r": 0, "hash_ir": 1, "chash": 2, "hash_ir_lb": 1, "chash_lb": 2}


def digest(I: np.ndarray, R: np.ndarray, Q: np.ndarray, tile: int, tid: int, arm: str, S: int, stagger: int,
           live: bool) -> bytes:
    """The lane's leaf: BLAKE3 compressions chained over its bound atoms' words (16 a block, fragment order): R then I
    (hash_ir; no R at atom 0, the duplicate of I), R (hash_r), or the chained accumulator Q (chash); atom t is bound when
    (t + 1 + phase) % S == 0, phase = (tid / 32) % S when staggered.  `_lb`: each fragment's four words as one word of their
    bits 8 ... 15, 16 a block, the last block zero-filled.  Keyed by IV xor (tile, tid, S | words << 16 | lb << 24, stagger,
    0, ...); I, R and Q are (lane outputs x atoms)."""
    lb = arm.endswith("_lb")
    base = arm.removesuffix("_lb")
    cv = list(blake3.IV)
    cv[0] ^= tile
    cv[1] ^= tid
    cv[2] ^= (S | WORDS[arm] << 16 | lb << 24) & 0xFFFFFFFF
    cv[3] ^= stagger
    phase = (tid // 32) % S if stagger else 0
    stream = []
    if live:
        for t in range(I.shape[1]):
            if (t + 1 + phase) % S:
                continue
            group = {"hash_r": [R[:, t]], "hash_ir": [I[:, t]] if t == 0 else [R[:, t], I[:, t]], "chash": [Q[:, t]]}[base]
            for w in group:
                w = [int(v) for v in w]
                stream += [sum((w[f + e] >> 8 & 0xFF) << 8 * e for e in range(4)) for f in range(0, len(w), 4)] if lb else w
    if len(stream) % 16:
        stream += [0] * (16 - len(stream) % 16)
    for ctr, b in enumerate(range(0, len(stream), 16)):
        cv = blake3.compress(tuple(cv), struct.pack("<16I", *stream[b:b + 16]), ctr, 64, blake3.KEYED_HASH)[:8]
    return struct.pack("<8I", *cv)


def check_small(out: str, kind: str) -> list[str]:
    meta = json.load(open(os.path.join(out, "check", f"{kind}.json")))
    m, n, K = meta["m"], meta["n"], meta["K"]
    T = K // 32
    rr, cc = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
    a, b = operands(meta, np.arange(m), np.arange(n))
    A, B = a[rr.ravel()], b[cc.ravel()]
    I, R = chain(A, B, T)
    Q = chained(A, B, T)
    want_r = R[:, -1].reshape(m, n)
    fails = []
    tiles_n = n // meta["bn"]
    for run in meta["runs"]:
        got = np.frombuffer(open(os.path.join(out, "check", run["tag"] + ".c.bin"), "rb").read(), dtype=np.uint32).reshape(m, n)
        if run["arm"].startswith("chash"):
            want = Q[:, -1].reshape(m, n)
        elif run["arm"] == "plain":
            want = plain(A, B, run["K"] // 32).reshape(m, n)
        else:
            want = want_r
        bad = int((got != want).sum())
        if bad:
            fails.append(f"{run['tag']}: C {bad} of {m * n} words differ")
        if run["arm"] in WORDS:
            dg = open(os.path.join(out, "check", run["tag"] + ".dg.bin"), "rb").read()
            wrong = 0
            for t in range(meta["tiles"]):
                tm, tn = t // tiles_n, t % tiles_n
                for tid in range(meta["nct"]):
                    rows, cols, live = lane_cells(meta, tm, tn, tid)
                    sel = rows * n + cols
                    d = digest(I[sel], R[sel], Q[sel], t, tid, run["arm"], run["S"], run.get("stagger", 0), live)
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
            control = arm["arm"] == "control"
            name = arm.get("as", arm["arm"])
            matched = 0
            for ln in arm["lanes"]:
                key = (ln["tm"], ln["tn"], ln["tid"])
                if key not in cache:
                    rows, cols, live = lane_cells(shape, *key)
                    a, b = operands(shape, rows, cols)
                    I, R = chain(a, b, T)
                    cache[key] = (rows, cols, live, a, b, I, R, chained(a, b, T))
                rows, cols, live, a, b, I, R, Q = cache[key]
                got = np.array([int(h, 16) for h in ln["cells"]], dtype=np.uint32)
                if name.startswith("chash"):
                    want = Q[:, -1]
                elif name == "plain":
                    want = plain(a, b, arm["K"] // 32)
                else:
                    want = R[:, -1]
                stagger = arm.get("stagger", 0)
                tag = (f"{shape['kind']} {arm['arm']} S={arm['S']} K={arm['K']}{' stagger' if stagger else ''} "
                       f"tile {ln['tile']} lane {ln['tid']}")
                ok = not (got != want).any()
                if not ok and not control:
                    fails.append(f"{tag}: {int((got != want).sum())} of {len(got)} cells differ")
                matched += ok
                if "digest" in ln:
                    d = digest(I, R, Q, ln["tile"], ln["tid"], name, arm["S"], stagger, live)
                    ok = d.hex() == ln["digest"]
                    if not ok and not control:
                        fails.append(f"{tag}: digest differs")
                    matched += ok
            if control:
                if matched:
                    fails.append(f"{shape['kind']} control: {matched} comparisons of poisoned, unwritten buffers match")
                print(f"arms {shape['kind']} control (as {name} S={arm['S']}): "
                      f"{'rejected' if not matched else 'ACCEPTED'} on {len(arm['lanes'])} lanes", flush=True)
                continue
            print(f"arms {shape['kind']} {arm['arm']} S={arm['S']} K={arm['K']} stagger={arm.get('stagger', 0)}: "
                  f"{len(arm['lanes'])} lanes checked", flush=True)
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
