"""SERVE-REG's gated lane against option A's prover side, byte for byte: the rows `partials.lane_partials` computes (the
cpu's chain split by lane, the gpu's `kh_lanes`) are the rows `h1zy.PouwH1LaneZy_v1` checks.

SERVE-REG's lane (cursor/ca-h1-lane-ea24) checks a leaf's committed partials row lane by lane: the row is the lane's
chain words ANDed with the OR of the leaf's R admit bits, so a dead leaf's row is zero.  Its references on words are
`h1tile.lane_partials(a, b, salt, position, roots, gated=admit)`, the leaf's 32 lanes, and `h1zy.lane_partial(a, b,
admit, salt, position, roots, lane, tile)`, one lane from its 4 A rows and 8 B rows by slot (`references`).

A served step's admit bits are its slots < m, so a leaf's OR is 32 · row block < m, partials.py's live rule.  At
m = 32 r + j (0 < j < 8) a leaf of row block r is live with only its rows 0 .. j - 1 admitted, and a lane g >= j (rows
g + 8 s) holds no admitted row but is ungated all the same: `served` builds such steps on a served step's own operands.
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1zy as ZY

R = 32
PARTLY = (1, 3, 7)     # j of m = 32 r + j: 28, 20 and 4 of a leaf's 32 lanes hold no admitted row


def _u8(t) -> list[list[int]]:
    return (t.cpu().numpy().astype(np.int64) & 0xFF).tolist()


def _job(a: list, b: list, admit: list, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> tuple[list, list]:
    lay = HT.Layout(R, R, len(a[0]))
    tile = HT.lane_partials(a, b, salt, position, roots, gated=admit)
    lanes = [ZY.lane_partial([a[i] for i in lay.rows(ln)], [b[j] for j in lay.cols(ln)], admit, salt, position, roots,
                             ln, (R, R)) for ln in range(lay.lanes)]
    return tile, lanes


def inputs(win, leaf) -> tuple[list, list, list, bytes, int, tuple[bytes, bytes]]:
    """The leaf's arguments to SERVE-REG's references: its A and B row blocks (u8 encodings), its admit bits (slots < m),
    the step's salt, the leaf's position and its (A, B) roots."""
    import partials as PT
    x, st = PT._at(win, leaf), win.steps[leaf.step]
    a = _u8(st.a[leaf.call][R * x.ri:R * x.ri + R])
    b = _u8(win.b[leaf.call][R * x.cj:R * x.cj + R])
    admit = [int(R * x.ri + i < st.m) for i in range(R)]
    return a, b, admit, st.salt, x.position, (st.a_roots[leaf.call], win.b_roots[leaf.call])


def references(win, leaves, workers: int = 1) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each leaf's (`h1tile.lane_partials`, `h1zy.lane_partial` lane by lane), (32, W) uint32 each, on `workers`
    spawned processes (1: here)."""
    args = [inputs(win, x) for x in leaves]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            got = list(ex.map(_job, *zip(*args)))
    else:
        got = [_job(*x) for x in args]
    return [(np.array(t, dtype=np.uint32), np.array(ln, dtype=np.uint32)) for t, ln in got]


def compare(win, leaves, kernel=None, workers: int = 1, receipts: dict | None = None) -> dict:
    """`partials.lane_partials` on the cpu (and the gpu, with `kernel`) against SERVE-REG's two references on each leaf:
    every mismatch by leaf (step, call, unit), lane and first word.  `receipts` {leaf: the committed receipt (W,)} adds
    the references' XOR against it."""
    import partials as PT
    cpu = PT.lane_partials_many(win, leaves)
    gpu = PT.lane_partials_many(win, leaves, "gpu", kernel) if kernel is not None else None
    refs = references(win, leaves, workers)
    bad, xor_bad, by_k = [], [], {}
    live = partly = no_admit = 0
    for i, leaf in enumerate(leaves):
        x, st = PT._at(win, leaf), win.steps[leaf.step]
        k = win.k(leaf.call)
        admit = [R * x.ri + j < st.m for j in range(R)]
        lay = HT.Layout(R, R, k)
        empty = [ln for ln in range(lay.lanes) if not any(admit[r] for r in lay.rows(ln))] if x.live else []
        live += x.live
        partly += bool(empty)
        no_admit += len(empty)
        by_k[str(k)] = by_k.get(str(k), 0) + 1
        tile, lanes = refs[i]
        rows = {"cpu": cpu[i], **({"gpu": gpu[i]} if gpu is not None else {}), "lane_partial": lanes}
        for name, got in rows.items():
            for ln in np.flatnonzero((got != tile).any(axis=1)):
                w = int(np.flatnonzero(got[ln] != tile[ln])[0])
                bad.append({"leaf": [leaf.step, leaf.call, leaf.unit], "k": k, "live": x.live,
                            "admitted": int(sum(admit)), "lane": int(ln), "lane_has_admitted_row": int(ln) not in empty,
                            "row": name, "against": "h1tile.lane_partials", "word": w,
                            "got": f"{int(got[ln, w]):#010x}", "want": f"{int(tile[ln, w]):#010x}"})
        if receipts is not None and leaf in receipts and not np.array_equal(PT.combine(tile), receipts[leaf]):
            xor_bad.append([leaf.step, leaf.call, leaf.unit])
    doc = {"leaves": len(leaves), "live": live, "dead": len(leaves) - live, "partly_admitted": partly,
           "lanes_without_an_admitted_row": no_admit, "by_k": by_k, "backends": ["cpu"] + (["gpu"] if gpu is not None else []),
           "mismatches": len(bad), "first_mismatches": bad[:16]}
    if receipts is not None:
        doc["reference_xor_vs_receipt_bad"] = xor_bad
    doc["ok"] = not bad and not xor_bad
    return doc


def served(win, g: dict, salt: bytes, lib=None, per_call: int = 1, workers: int = 32) -> dict:
    """On a served step's capture `g` (dither_one's Window `win`, the step's salt; `lib` the served h1u build for the
    gpu): every call's `per_call` live leaves and one dead leaf (when the step has a dead row block), each against the
    references and the committed receipt; then, on the step's own operands at m = 32 r + j (j in `PARTLY`,
    r = min(m // 32, t / 32 - 1)), one leaf of row block r on calls 0, 2, the first of the widest K and the last, whose
    lanes g >= j hold no admitted row."""
    import partials as PT
    s, m, t, L = g["s"], g["m"], win.t, win.L
    kern = PT.Kernel(lib) if lib is not None else None
    step = PT.Step(list(g["a"]), list(g["a_roots"]), m, salt)
    pw = PT.Window(list(win.wt), list(g["b_roots"]), list(win.offs), win.u_step, {s: step})
    rng = np.random.default_rng(s + 7)
    nrb, r = t // R, -(-min(m, t) // R)
    leaves, rcpt = [], {}
    for w in range(L):
        ncb = win.nk[w][0] // R
        us = [int(u) for u in rng.choice(r * ncb, size=min(per_call, r * ncb), replace=False)]
        if r < nrb:
            us.append(int(r * ncb + rng.integers((nrb - r) * ncb)))
        o_, units, nw = win.plan.offs[w]
        got = g["receipts"][o_:o_ + units * nw].view(units, nw)
        for u in us:
            leaf = PT.Leaf(s, w, u)
            leaves.append(leaf)
            rcpt[leaf] = got[u].cpu().numpy().view(np.uint32)
    doc = {"s": s, "m": m, "t": t, "real": compare(pw, leaves, kern, workers, rcpt), "partly": {}}
    rb = min(m // R, nrb - 1)
    wide = [w for w in range(L) if win.nk[w][1] == max(k for _, k in win.nk)][:1]
    for j in PARTLY:
        mj = R * rb + j
        tw = PT.Window(pw.b, pw.b_roots, pw.offs, pw.u_step, {s: PT.Step(step.a, step.a_roots, mj, salt)})
        ls = [PT.Leaf(s, w, int(rb * (win.nk[w][0] // R) + rng.integers(win.nk[w][0] // R)))
              for w in sorted({0, 2 % L, L - 1, *wide})]
        doc["partly"][str(mj)] = compare(tw, ls, kern, workers)
    doc["ok"] = doc["real"]["ok"] and all(x["ok"] and x["partly_admitted"] == x["leaves"] for x in doc["partly"].values())
    return doc


__all__ = ["PARTLY", "compare", "inputs", "references", "served"]
