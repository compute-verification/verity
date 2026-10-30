"""ovl.cu's edited arms' receipts judged by EGEMM's reference: sampled tiles of each dumped call (EG_DUMP; X, W' and the
receipts of chunk 0's layer-31 call of a linear) recomputed by ``segment.PouwSegCheck_v2{S, duo}``'s own fold
(``fold_keys``, ``fold_absorb``, ``fold_final`` of `verity.ml.protocols.pouw.circuit.segment`, as hashed_gemm's
hgref.py reads it) from the running sums of X W'^T, every word of
those tiles compared; another seed's fold must differ (the control).

    python3 eref.py DUMP...   a dump directory each (dump.json, X.bin, Wp.bin, operand.bin: the receipts, a 2 KiB row a
                              window of a tile, R[(tile nw + w) 512 + thread]); JSON lines, exit 1 on any difference

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

from verity.ml.protocols.pouw.circuit import segment as SG

NCT = 512


def thread_entries() -> tuple[np.ndarray, np.ndarray]:
    """Every thread's 32 entries' (row, col) in its tile (hgref.py's)."""
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, qq = warp // 4, warp % 4, lane >> 2, lane & 3
    rs = np.array([SG.row_slot(q) for q in range(32)])
    cs = np.array([SG.col_slot(q) for q in range(32)])
    rows = wm[:, None] * 32 + g[:, None] + 8 * rs[None, :]
    cols = wn[:, None] * 32 + 2 * qq[:, None] + 8 * (cs[None, :] >> 1) + (cs[None, :] & 1)
    return rows, cols


def tile_receipts(job: tuple) -> np.ndarray:
    """One tile's receipts by segment's duo fold, (nw, NCT), from X's and W''s rows of the tile (hgref.py's, S a
    parameter)."""
    seed, tile, S, xc, wc = job
    k = xc.shape[1]
    ks, nw = k // 32, k // (32 * S)
    inc = np.einsum("isk,jsk->sij", xc.reshape(128, ks, 32).astype(np.int64), wc.reshape(128, ks, 32).astype(np.int64))
    Ts = (np.cumsum(inc, axis=0) & 0xFFFFFFFF).astype(np.uint32)
    rows, cols = thread_entries()
    P = np.concatenate([np.zeros((1, NCT, 32), np.uint32), Ts[:, rows, cols]])   # P[j + 1] = T_j
    out = np.zeros((nw, NCT), np.uint32)
    for ctid in range(NCT):
        Pt = P[:, ctid, :].tolist()
        for w in range(nw):
            st = SG.fold_keys("duo", seed, tile, ctid, w, nw)
            for i in range(S + 1):
                st = SG.fold_absorb("duo", st, Pt[w * S + i])
            out[w, ctid] = SG.fold_final("duo", st)
    return out


def jobs_of(d: Path) -> tuple[dict, list]:
    meta = json.loads((d / "dump.json").read_text())
    m, n, k, S, seed = meta["m"], meta["n"], meta["k"], meta["S"], meta["seed"]
    tiles, tiles_n = (m // 128) * (n // 128), n // 128
    X = np.fromfile(d / "X.bin", np.int8).reshape(m, k)
    W = np.fromfile(d / "Wp.bin", np.int8).reshape(n, k)
    rng = random.Random(seed)
    picks = sorted({0, tiles - 1, rng.randrange(tiles)})
    out = []
    for tile in picks:
        tm, tn = divmod(tile, tiles_n)
        xc, wc = X[128 * tm:128 * tm + 128], W[128 * tn:128 * tn + 128]
        out.append((tile, False, (seed, tile, S, xc, wc)))
    tm, tn = divmod(picks[0], tiles_n)
    out.append((picks[0], True, (seed ^ 1, picks[0], S, X[128 * tm:128 * tm + 128], W[128 * tn:128 * tn + 128])))
    return meta, out


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    plan = [(Path(a), *jobs_of(Path(a))) for a in argv[1:]]
    flat = [j for _, _, js in plan for j in js]
    with mp.get_context("fork").Pool() as pool:
        got = pool.map(tile_receipts, [j[2] for j in flat])
    it = iter(got)
    failed = 0
    for d, meta, js in plan:
        m, n, k, S = meta["m"], meta["n"], meta["k"], meta["S"]
        tiles, nw = (m // 128) * (n // 128), k // (32 * S)
        R = np.fromfile(d / "operand.bin", np.uint32).reshape(tiles * nw, NCT)
        rec, ok = [], nw == meta["nw"]
        for tile, control, _ in js:
            want = next(it)
            bad = int((R[tile * nw:(tile + 1) * nw] != want).sum())
            rec.append({"tile": tile, "control": control, "words": int(want.size), "bad": bad})
            ok &= bad > want.size // 2 if control else bad == 0
        failed += not ok
        print(json.dumps({"dir": str(d), "stage": meta["stage"], "m": m, "n": n, "k": k, "S": S, "nw": nw, "tiles": rec, "ok": bool(ok)}),
              flush=True)
    print(json.dumps({"summary": {"dumps": len(plan), "failed": failed, "ref_s": round(time.perf_counter() - t0, 1)}}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
