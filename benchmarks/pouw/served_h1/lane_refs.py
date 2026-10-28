"""SERVE-REG's gated lane against option A's prover side, byte for byte: the rows `partials.lane_partials` computes (the
cpu's chain split by lane, the gpu's `kh_lanes`) are the rows `h1zy.PouwH1LaneZy_v1` checks.

SERVE-REG's lane (`h1zy.PouwH1LaneZy_v1`) checks a leaf's committed partials row lane by lane: the row is the lane's
chain words ANDed with the OR of the leaf's R admit bits, so a dead leaf's row is zero.  Its references on words are
`h1tile.lane_partials(a, b, salt, position, roots, gated=admit)`, the leaf's 32 lanes, and `h1zy.lane_partial(a, b,
admit, salt, position, roots, lane, tile)`, one lane from its 4 A rows and 8 B rows by slot.  The references take any
K; `h1zy.lane` and `h1zy.combine` stage a leaf at a positive K and an R that is a multiple of 16, and the combine only at
K >= 384 (`h1zy.LEAST_K`).

A served step's admit bits are its slots < m, so a leaf's OR is 32 · row block < m, partials.py's live rule.  At
m = 32 r + j (0 < j < 8) a leaf of row block r is live with only its rows 0 .. j - 1 admitted, and a lane g >= j (rows
g + 8 s) holds no admitted row but is ungated all the same: `served` builds such steps on a served step's own operands.

A leaf's record (`records`) holds its inputs to the references, its partials (cpu, and gpu with a kernel) and, for a
served leaf, its committed receipt; `check` runs the references on records (pure Python, on a pool of processes where
the caller may have children) and reports every mismatch by leaf, lane and word.  The gate's process is a daemon that
may not, so `served` with a path saves the records (`save`) for `python lane_refs.py CAPTURE` after serving.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1zy as ZY

R = 32
PARTLY = (1, 3, 7)     # j of m = 32 r + j: 28, 20 and 4 of a leaf's 32 lanes hold no admitted row


def _job(a: list, b: list, admit: list, salt: bytes, position: int, roots: tuple[bytes, bytes]) -> tuple[list, list]:
    lay = HT.TileShape(R, R, len(a[0]))
    tile = HT.lane_partials(a, b, salt, position, roots, gated=admit)
    lanes = [ZY.lane_partial([a[i] for i in lay.rows(ln)], [b[j] for j in lay.cols(ln)], admit, salt, position, roots,
                             ln, (R, R)) for ln in range(lay.lanes)]
    return tile, lanes


def records(win, leaves, kernel=None, receipts: dict | None = None, group: str = "") -> list[dict]:
    """Each leaf's record: its step, call and unit, K, live bit and admit bits (slots < m), the step's salt, its position
    and (A, B) roots, its A and B row blocks (u8), its partials on the cpu and (with `kernel`) the gpu, and its
    committed receipt from `receipts` {leaf: (W,)}."""
    import partials as PT
    cpu = PT.lane_partials_many(win, leaves)
    gpu = PT.lane_partials_many(win, leaves, "gpu", kernel) if kernel is not None else [None] * len(leaves)
    out = []
    for leaf, c, g in zip(leaves, cpu, gpu):
        x, st = PT._at(win, leaf), win.steps[leaf.step]
        out.append({"leaf": [leaf.step, leaf.call, leaf.unit], "group": group, "k": win.k(leaf.call), "live": x.live,
                    "admit": [int(R * x.ri + i < st.m) for i in range(R)], "salt": st.salt, "position": x.position,
                    "roots": (st.a_roots[leaf.call], win.b_roots[leaf.call]),
                    "a": st.a[leaf.call][R * x.ri:R * x.ri + R].cpu().numpy().view(np.uint8),
                    "b": win.b[leaf.call][R * x.cj:R * x.cj + R].cpu().numpy().view(np.uint8), "cpu": c, "gpu": g,
                    "receipt": None if receipts is None else receipts.get(leaf)})
    return out


def save(path, recs: list[dict]) -> None:
    """The records as one npz: their arrays by index, the rest as JSON."""
    arrays, meta = {}, []
    for i, r in enumerate(recs):
        for key in ("a", "b", "cpu", "gpu", "receipt"):
            if r[key] is not None:
                arrays[f"{i}.{key}"] = r[key]
        meta.append({**{k: v for k, v in r.items() if k not in ("a", "b", "cpu", "gpu", "receipt", "salt", "roots")},
                     "salt": r["salt"].hex(), "roots": [x.hex() for x in r["roots"]]})
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)


def load(path) -> list[dict]:
    z = np.load(path)
    out = []
    for i, r in enumerate(json.loads(str(z["meta"]))):
        out.append({**r, "salt": bytes.fromhex(r["salt"]), "roots": tuple(bytes.fromhex(x) for x in r["roots"]),
                    **{k: z[f"{i}.{k}"] if f"{i}.{k}" in z else None for k in ("a", "b", "cpu", "gpu", "receipt")}})
    return out


def check(recs: list[dict], workers: int = 1) -> dict:
    """SERVE-REG's two references on each record against its cpu and gpu partials, and the `h1zy.lane_partial` rows and
    the receipt against `h1tile.lane_partials`'; by group: leaves, live, dead, partly admitted (live with a lane holding
    no admitted row), those lanes, and the mismatches."""
    args = [((r["a"].astype(np.int64)).tolist(), (r["b"].astype(np.int64)).tolist(), r["admit"], r["salt"],
             r["position"], tuple(r["roots"])) for r in recs]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            refs = list(ex.map(_job, *zip(*args)))
    else:
        refs = [_job(*x) for x in args]
    groups: dict[str, dict] = {}
    bad = []
    for r, (tile, lanes) in zip(recs, refs):
        tile, lanes = np.array(tile, dtype=np.uint32), np.array(lanes, dtype=np.uint32)
        lay = HT.TileShape(R, R, r["k"])
        empty = [ln for ln in range(lay.lanes) if not any(r["admit"][i] for i in lay.rows(ln))] if r["live"] else []
        gd = groups.setdefault(r["group"], {"leaves": 0, "live": 0, "dead": 0, "partly_admitted": 0,
                                            "lanes_without_an_admitted_row": 0, "by_k": {}, "backends": [],
                                            "mismatches": 0, "reference_xor_vs_receipt_bad": []})
        gd["leaves"] += 1
        gd["live" if r["live"] else "dead"] += 1
        gd["partly_admitted"] += bool(empty)
        gd["lanes_without_an_admitted_row"] += len(empty)
        gd["by_k"][str(r["k"])] = gd["by_k"].get(str(r["k"]), 0) + 1
        rows = {"cpu": r["cpu"], "gpu": r["gpu"], "lane_partial": lanes}
        for name, got in rows.items():
            if got is None:
                continue
            if name not in gd["backends"]:
                gd["backends"].append(name)
            for ln in np.flatnonzero((got != tile).any(axis=1)):
                w = int(np.flatnonzero(got[ln] != tile[ln])[0])
                gd["mismatches"] += 1
                bad.append({"leaf": r["leaf"], "group": r["group"], "k": r["k"], "live": r["live"],
                            "admitted": int(sum(r["admit"])), "lane": int(ln), "lane_has_admitted_row": int(ln) not in empty,
                            "row": name, "against": "h1tile.lane_partials", "word": w,
                            "got": f"{int(got[ln, w]):#010x}", "want": f"{int(tile[ln, w]):#010x}"})
        if r["receipt"] is not None and not np.array_equal(np.bitwise_xor.reduce(tile, axis=0), r["receipt"]):
            gd["reference_xor_vs_receipt_bad"].append(r["leaf"])
    for gd in groups.values():
        gd["ok"] = not gd["mismatches"] and not gd["reference_xor_vs_receipt_bad"]
    return {"groups": groups, "mismatches": len(bad), "first_mismatches": bad[:16],
            "ok": all(gd["ok"] for gd in groups.values())}


def compare(win, leaves, kernel=None, workers: int = 1, receipts: dict | None = None) -> dict:
    """`check` of the leaves' records, as one group."""
    doc = check(records(win, leaves, kernel, receipts), workers)
    return {**doc.pop("groups")[""], **doc}


def verdict(doc: dict) -> bool:
    """A served step's check: every group's rows equal, and every partly admitted leaf is one."""
    gs = doc["groups"]
    return bool(doc["ok"] and "real" in gs and all(gs[g]["partly_admitted"] == gs[g]["leaves"]
                                                    for g in gs if g.startswith("partly")))


def served(win, g: dict, salt: bytes, lib=None, per_call: int = 1, path=None, workers: int = 1) -> dict:
    """On a served step's capture `g` (dither_one's DitherState `win`, the step's salt; `lib` the served h1u build for the
    gpu): every call's `per_call` live leaves and one dead leaf (when the step has a dead row block), group "real", with
    their committed receipts; then, on the step's own operands at m = 32 r + j (j in `PARTLY`,
    r = min(m // 32, t / 32 - 1)), one leaf of row block r on calls 0, 2, the first of the widest K and the last, group
    "partly:m", whose lanes g >= j hold no admitted row.  Here: the cpu against the gpu and the partials' XOR against the
    receipt; then with `path` the records saved for `check` after serving, else `check` here."""
    import partials as PT
    s, m, t, L = g["s"], g["m"], win.t, win.L
    kern = PT.Kernel(lib) if lib is not None else None
    step = PT.Step(list(g["a"]), list(g["a_roots"]), m, salt)
    pw = PT.WindowCalls(list(win.wt), list(g["b_roots"]), list(win.offs), win.u_step, {s: step})
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
    recs = records(pw, leaves, kern, rcpt, "real")
    rb = min(m // R, nrb - 1)
    wide = [w for w in range(L) if win.nk[w][1] == max(k for _, k in win.nk)][:1]
    for j in PARTLY:
        mj = R * rb + j
        tw = PT.WindowCalls(pw.b, pw.b_roots, pw.offs, pw.u_step, {s: PT.Step(step.a, step.a_roots, mj, salt)})
        ls = [PT.Leaf(s, w, int(rb * (win.nk[w][0] // R) + rng.integers(win.nk[w][0] // R)))
              for w in sorted({0, 2 % L, L - 1, *wide})]
        recs += records(tw, ls, kern, None, f"partly:{mj}")
    here = {"s": s, "m": m, "t": t, "leaves": len(recs),
            "cpu_vs_gpu_bad": [x["leaf"] for x in recs if x["gpu"] is not None and x["cpu"].tobytes() != x["gpu"].tobytes()],
            "xor_vs_receipt_bad": [x["leaf"] for x in recs if x["receipt"] is not None
                                   and not np.array_equal(np.bitwise_xor.reduce(x["cpu"], axis=0), x["receipt"])],
            "gpu": kern is not None}
    here["here_ok"] = not here["cpu_vs_gpu_bad"] and not here["xor_vs_receipt_bad"]
    if path is not None:
        save(path, recs)
        return {**here, "captured": str(path), "ok": here["here_ok"]}
    doc = check(recs, workers)
    return {**here, **doc, "ok": here["here_ok"] and verdict(doc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SERVE-REG's gated lane on a saved capture (`served` with a path)")
    ap.add_argument("capture")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    recs = load(a.capture)
    doc = check(recs, a.workers)
    doc["served_ok"] = verdict(doc)
    doc["ok"] = doc["served_ok"]
    text = json.dumps(doc, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if doc["ok"] else 1


__all__ = ["PARTLY", "check", "compare", "load", "records", "save", "served", "verdict"]

if __name__ == "__main__":
    sys.exit(main())
