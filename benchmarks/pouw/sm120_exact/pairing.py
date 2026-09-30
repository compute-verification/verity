#!/usr/bin/env python3
"""`fp8-merge-rate/sm120` under the greedy adversarial pairing (server.md 09:08Z), on the CPU: one Strassen level may take any
row permutation of A, any column permutation of B and one k permutation shared by both, so the prover chooses which rows,
columns and k indices meet in each pre-add. How large a block can it make whose pre-adds are all E4M3?

A pairs row i (to A11/A12) with row i' (to A21/A22); the k permutation pairs index c (to the first half) with c' (the second).
With u = A[i], v = A[i']: A21 - A11 = v[c] - u[c] and A12 - A22 = u[c'] - v[c'] read one index each, and A11 + A22 =
u[c] + v[c'], A21 + A22 = v[c] + v[c'], A11 + A12 = u[c] + u[c'] read the pair. B likewise pairs weight rows j, j' (columns of
B) over the same k pairs: with u = Bs[j], v = Bs[j'], B11 + B12 = u[c] + v[c] and B21 + B22 = u[c'] + v[c'] read one index,
B11 + B22 = u[c] + v[c'], B12 - B22 = v[c] - v[c'] and B21 - B11 = u[c'] - u[c] the pair. A 16 x 32 A fragment is 16 row
pairs over 32 k pairs, all 5 exact on each; an 8-column B fragment is 8 column pairs over the same 32 k pairs.

The greedy, per item: start from every directed k pair of --k-search indices; add the row pair (A-only, B-only, or either
side for the joint search) that keeps the most k pairs, from --cands candidates (random pairs of admitted rows, and pairs of
the rows that share the most codes); after each step, a greedy matching (lowest degree first) gives the k pairs usable at
once. It stops at the fragment's size (16 A pairs, 8 B pairs) or when fewer than 32 k pairs remain, and restarts from the
--seeds first pairs that keep the most k pairs and the --seeds that share the most codes, keeping the best run (or the first
to reach a fragment). The joint search takes each step's pair from either side; --joint-policies adds joint searches that
choose the side otherwise: `B-first` (B's pairs until it has its 8, then A's) and `balanced` (the side furthest from its
need). Candidates
are ranked on samples (the first step on 512 x 512 of its index pairs, later steps on 65,536 of the surviving k pairs) and the
chosen one is applied exactly. The report per item: each step's pairs, surviving k pairs and matching, the largest block
(pairs x matched k pairs), and whether a fragment was reached. A lower bound on the adversary: greedy, over --rows sampled
rows per side.

Items: `unit:<family>` (the census family's rows, formed by census.form_s5 at delta 1, line norm 16; run 1's seeds) and, with
--acts DIR, `acts:<linear>` (a pearlc_capture directory's token rows and the linear's weight rows, admitted rows only).
`pearlc_activations.py --pairing` runs `pair_unit` on a live model's linears.
Restartable as run.py: one JSON per item in --out, state.json, exit 99 once --budget-min is spent while items remain.
--selftest checks the pre-add conditions against merge_stats' quadrants on the identity pairing, and that the search finds a
planted all-exact fragment.

    python benchmarks/pouw/sm120_exact/pairing.py --selftest
    python benchmarks/pouw/sm120_exact/pairing.py --families gaussian,spikes-first --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import pearlc_census as census  # noqa: E402
import run_result  # noqa: E402
from run import ATOM, LINE_NORM, LUT_DIFF, LUT_SUM, UNIT_FAMILIES, capture_linear  # noqa: E402

NEED = {"A": 16, "B": 8}
KPAIRS = 32
SAMPLE = 512
RANK_EDGES = 1 << 16


def conds(side: str, u: np.ndarray, v: np.ndarray):
    """The row pair (u, v)'s conditions: on a first index c, on a second index c', and on the pair (c, c')."""
    if side == "A":
        return (LUT_DIFF[v, u], LUT_DIFF[u, v],
                lambda c, d: LUT_SUM[u[c], v[d]] & LUT_SUM[v[c], v[d]] & LUT_SUM[u[c], u[d]])
    return (LUT_SUM[u, v], LUT_SUM[u, v],
            lambda c, d: LUT_SUM[u[c], v[d]] & LUT_DIFF[v[c], v[d]] & LUT_DIFF[u[d], u[c]])


def evaluate(side: str, X: np.ndarray, p: tuple[int, int], ks: np.ndarray, edges):
    """The k pairs that survive adding row pair p: from every directed pair of ks (edges None) or from `edges`."""
    first, second, pair = conds(side, X[p[0]], X[p[1]])
    if edges is None:
        c1, c2 = ks[first[ks]], ks[second[ks]]
        ok = pair(c1[:, None], c2[None, :]) & (c1[:, None] != c2[None, :])
        i, j = np.nonzero(ok)
        return c1[i], c2[j]
    e1, e2 = edges
    keep = first[e1] & second[e2] & pair(e1, e2)
    return e1[keep], e2[keep]


def estimate(side: str, X: np.ndarray, p: tuple[int, int], ks: np.ndarray, edges, rng: np.random.Generator) -> float:
    """evaluate's count, estimated on at most SAMPLE first and second indices (edges None), or exact on `edges`."""
    first, second, pair = conds(side, X[p[0]], X[p[1]])
    if edges is None:
        c1, c2 = ks[first[ks]], ks[second[ks]]
        s1 = c1 if len(c1) <= SAMPLE else rng.choice(c1, SAMPLE, replace=False)
        s2 = c2 if len(c2) <= SAMPLE else rng.choice(c2, SAMPLE, replace=False)
        if not len(s1) or not len(s2):
            return 0.0
        ok = pair(s1[:, None], s2[None, :]) & (s1[:, None] != s2[None, :])
        return float(ok.mean()) * len(c1) * len(c2)
    return float(len(evaluate(side, X, p, ks, edges)[0]))


def matching(edges, k: int) -> int:
    """A greedy matching of the directed k pairs (each index used once, as first or second), lowest degree first."""
    e1, e2 = edges
    if not len(e1):
        return 0
    deg = np.bincount(e1, minlength=k) + np.bincount(e2, minlength=k)
    order = np.argsort(deg[e1] + deg[e2], kind="stable")
    a, b = e1[order], e2[order]
    used = np.zeros(k, bool)
    n = 0
    for i in range(0, len(a), 1 << 16):
        ca, cb = a[i:i + (1 << 16)], b[i:i + (1 << 16)]
        live = ~used[ca] & ~used[cb]
        for x, y in zip(ca[live].tolist(), cb[live].tolist()):
            if not used[x] and not used[y]:
                used[x] = used[y] = True
                n += 1
    return n


def candidates(X: np.ndarray, ok: np.ndarray, used: set, n: int, rng: np.random.Generator, similar: np.ndarray):
    rows = np.array([r for r in np.flatnonzero(ok) if r not in used])
    if len(rows) < 2:
        return []
    out = set()
    for _ in range(n):
        a, b = rng.choice(rows, 2, replace=False)
        out.add((int(a), int(b)))
    for a in rng.choice(rows, min(n, len(rows)), replace=False):
        order = [int(b) for b in np.argsort(-similar[a], kind="stable") if b != a and b not in used and ok[b]]
        if order:
            out.add((int(a), order[0]))
            out.add((order[0], int(a)))
    return sorted(out)


def first_steps(sides: dict, ks: np.ndarray, cands: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """Per side: the codes-shared counts over ks[:512], and the first step's candidates with their estimated k pairs."""
    sim = {s: (X[:, ks[:512]][:, None, :] == X[:, ks[:512]][None, :, :]).sum(2) for s, (X, _) in sides.items()}
    first = {s: [(s, p, estimate(s, X, p, ks, None, rng)) for p in candidates(X, ok, set(), cands, rng, sim[s])]
             for s, (X, ok) in sides.items()}
    return sim, first


POLICIES = ("max", "B-first", "balanced")


def search(sides: dict, ks: np.ndarray, k: int, which: tuple[str, ...], cands: int, seeds: int, rng: np.random.Generator,
           sim: dict, first: dict, policy: str = "max") -> dict:
    """Greedy over row pairs of the sides in `which` (A, B or both), sharing one k pairing, restarted from `seeds` first pairs
    with the most surviving k pairs and `seeds` with the most shared codes; the best run is kept. A joint search's `policy`
    picks the side of each step: `max` whichever pair keeps the most k pairs, `B-first` B's pairs until it has its 8 (starting
    from B's first pairs), `balanced` the side furthest from its need."""
    pool = [t for s in (("B",) if policy == "B-first" else which) for t in first[s]]
    starts = [t[:2] for t in sorted(pool, key=lambda t: -t[2])[:seeds]]
    starts += [t[:2] for t in sorted(pool, key=lambda t: -int(sim[t[0]][t[1]]))[:seeds] if t[:2] not in starts]
    best = None
    for s, p in starts:
        res = grow(sides, ks, k, which, cands, rng, sim, s, p, policy)
        if best is None or rank(res) > rank(best):
            best = res
        if best["fragment"]:
            break
    return {**(best or grow(sides, ks, k, which, cands, rng, sim, None, None)), "starts": len(starts)}


def rank(res: dict) -> tuple:
    r = res["reached_with_32_k_pairs"]
    return res["fragment"], r[0] + r[1], r[2]


def grow(sides, ks, k, which, cands, rng, sim, s0, p0, policy: str = "max") -> dict:
    chosen = {s: [] for s in which}
    used = {s: set() for s in which}
    edges, steps = None, []
    if s0 is not None:
        edges = evaluate(s0, sides[s0][0], p0, ks, None)
        chosen[s0].append(p0)
        used[s0].update(p0)
        steps.append({**{f"n{x}": len(chosen[x]) for x in which}, "k_pairs": len(edges[0]), "matched": matching(edges, k)})
    while not steps or (steps[-1]["matched"] >= KPAIRS and any(len(chosen[s]) < NEED[s] for s in which)):
        sub = edges
        if edges is not None and len(edges[0]) > RANK_EDGES:
            pick = rng.choice(len(edges[0]), RANK_EDGES, replace=False)
            sub = (edges[0][pick], edges[1][pick])
        best = None
        open_ = [s for s in which if len(chosen[s]) < NEED[s]]
        if policy == "B-first" and "B" in open_:
            open_ = ["B"]
        elif policy == "balanced" and open_:
            open_ = [min(open_, key=lambda s: (len(chosen[s]) / NEED[s], s))]
        for s in open_:
            X, ok = sides[s]
            for p in candidates(X, ok, used[s], cands, rng, sim[s]):
                n = estimate(s, X, p, ks, sub, rng)
                if best is None or n > best[2]:
                    best = (s, p, n)
        if best is None:
            break
        s, p, _ = best
        edges = evaluate(s, sides[s][0], p, ks, edges)
        chosen[s].append(p)
        used[s].update(p)
        steps.append({**{f"n{x}": len(chosen[x]) for x in which}, "k_pairs": len(edges[0]), "matched": matching(edges, k)})
    blocks = [(st.get("nA", 0), st.get("nB", 0), st["matched"]) for st in steps]
    ok_steps = [b for b in blocks if b[2] >= KPAIRS]
    reached = max(ok_steps, key=lambda b: (b[0] + b[1], b[2]), default=(0, 0, 0))
    return {"steps": steps, "largest_block": max(blocks, key=lambda b: (b[0] + b[1]) * b[2], default=(0, 0, 0)),
            "reached_with_32_k_pairs": reached,
            "fragment": all(reached[i] >= NEED[s] for i, s in enumerate(("A", "B")) if s in which)}


def rules(X: np.ndarray) -> np.ndarray:
    import pearlc_activations as PA
    f, lv, _ = PA.row_rules(X, 1.0)
    return f & lv


def form(X: np.ndarray, Fc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return census.form_s5(X, Fc, 1.0, LINE_NORM, rng)[0]


def item_sides(item: str, k: int, rows: int, acts: Path | None, seed: int) -> tuple[dict, int, dict]:
    kind, name = item.split(":", 1)
    rng = census._rng("pairing", item, seed)
    if kind == "unit":
        XA, XB, _, _ = census.family_rows(name, k, 1.0, rows, rows, 1)
        info = {"family": name}
    else:
        X, W = capture_linear(acts, name)
        okX, okW = rules(X), rules(W)
        XA = X[rng.choice(np.flatnonzero(okX), min(rows, int(okX.sum())), replace=False)]
        XB = W[rng.choice(np.flatnonzero(okW), min(rows, int(okW.sum())), replace=False)]
        k = X.shape[1]
        info = {"linear": name, "rows_admitted": {"A": int(okX.sum()), "B": int(okW.sum())}}
    FcA, FcB = census.s5_lines(k, 1, LINE_NORM)
    sides = {"A": (form(XA, FcA, rng), rules(XA)), "B": (form(XB, FcB, rng), rules(XB))}
    info.update({"k": k, "rows_sampled": {s: len(v[0]) for s, v in sides.items()},
                 "rows_ok": {s: int(v[1].sum()) for s, v in sides.items()}})
    return sides, k, info


def joint_label(policy: str) -> str:
    return "joint" if policy == "max" else "joint_" + policy.replace("-", "_")


def pair_unit(sides: dict, k: int, k_search: int, cands: int, seeds: int, rng: np.random.Generator,
              policies: tuple[str, ...] = ("max",)) -> dict:
    """The A-only, B-only and joint searches on one unit's codes: sides = {"A": (codes, row_ok), "B": (codes, row_ok)}; a
    joint search per policy, `max`'s as `joint` and the others' after it (so `joint` is the same with or without them)."""
    t = time.time()
    ks = np.sort(rng.choice(k, min(k, k_search), replace=False))
    sim, first = first_steps(sides, ks, cands, rng)
    res = {"k_search": len(ks), "cands": cands, "seeds": seeds, "seconds_first_step": round(time.time() - t, 1)}
    runs = [("A_only", ("A",), "max"), ("B_only", ("B",), "max"), ("joint", ("A", "B"), "max")]
    runs += [(joint_label(p), ("A", "B"), p) for p in policies if p != "max"]
    for label, which, policy in runs:
        ts = time.time()
        res[label] = search(sides, ks, k, which, cands, seeds, rng, sim, first, policy)
        res[label]["seconds"] = round(time.time() - ts, 1)
    return res


def run_item(item: str, args) -> dict:
    t = time.time()
    sides, k, info = item_sides(item, args.k, args.rows, Path(args.acts) if args.acts else None, args.seed)
    res = {"item": item, **info, "seconds_forming": round(time.time() - t, 1)}
    res.update(pair_unit(sides, k, args.k_search, args.cands, args.seeds, census._rng("pairing-search", item, args.seed),
                         tuple(args.joint_policies.split(","))))
    res["seconds"] = round(time.time() - t, 1)
    return res


def selftest() -> None:
    census.set_atom(ATOM)
    from run import frag_count, merge_stats
    rng = np.random.default_rng(1)
    X = census.enc(rng.standard_normal((64, 256)) * 40)
    Y = census.enc(rng.standard_normal((32, 256)) * 40)
    ms = merge_stats(X, Y, 0)
    h, kh, nh = 32, 128, 16
    bad = 0
    for s, M, half in (("A", X, h), ("B", Y, nh)):
        if s == "A":
            A11, A12, A21, A22 = M[:h, :kh], M[:h, kh:], M[h:, :kh], M[h:, kh:]
            want = (LUT_SUM[A11, A22] & LUT_SUM[A21, A22] & LUT_SUM[A11, A12] & LUT_DIFF[A21, A11] & LUT_DIFF[A12, A22])
        else:
            B11, B12, B21, B22 = M[:nh, :kh], M[nh:, :kh], M[:nh, kh:], M[nh:, kh:]
            want = (LUT_SUM[B11, B22] & LUT_DIFF[B12, B22] & LUT_DIFF[B21, B11] & LUT_SUM[B11, B12] & LUT_SUM[B21, B22])
        c, d = np.arange(kh), np.arange(kh) + kh
        for r in range(half):
            first, second, pair = conds(s, M[r], M[r + half])
            got = first[c] & second[d] & pair(c, d)
            bad += int((got != want[r]).sum())
        bad += int(ms["all5_" + s][0] != frag_count(want, 16 if s == "A" else 8, 32)[0])
    print(json.dumps({"selftest": "identity pairing against merge_stats' quadrants", "mismatches": bad}))
    assert bad == 0
    # a planted fragment: 40 rows and 80 indices carry one value x whose double is exact (x + x, x - x = 0)
    A = census.enc(rng.standard_normal((256, 1024)) * 40)
    B = census.enc(rng.standard_normal((256, 1024)) * 40)
    rows, cols = rng.choice(256, 40, replace=False), rng.choice(1024, 80, replace=False)
    x = int(census.enc(np.array([3.0]))[0])
    A[np.ix_(rows, cols)] = x
    B[np.ix_(rows[:24], cols)] = x
    ks = np.arange(1024)
    sides = {"A": (A, np.ones(256, bool)), "B": (B, np.ones(256, bool))}
    for policy in POLICIES:
        r2 = np.random.default_rng(2)
        out = search(sides, ks, 1024, ("A", "B"), 64, 4, r2, *first_steps(sides, ks, 64, r2), policy)
        print(json.dumps({"selftest": "planted 40 x 80 block (A) and 24 x 80 (B)", "policy": policy,
                          "joint": out["reached_with_32_k_pairs"], "fragment": out["fragment"]}))
        assert out["fragment"]
    sides = {"A": (census.enc(rng.standard_normal((256, 1024)) * 40), np.ones(256, bool))}
    r3 = np.random.default_rng(3)
    rand = search(sides, ks, 1024, ("A",), 64, 4, r3, *first_steps(sides, ks, 64, r3))
    print(json.dumps({"selftest": "no planted block", "A_only": rand["reached_with_32_k_pairs"], "fragment": rand["fragment"]}))
    assert not rand["fragment"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(UNIT_FAMILIES))
    ap.add_argument("--acts", default=None, help="a pearlc_capture directory: each captured linear is an item")
    ap.add_argument("--acts-layers", default=".", help="regex over the capture's linear names")
    ap.add_argument("--k", type=int, default=8192)
    ap.add_argument("--rows", type=int, default=512, help="rows sampled per side")
    ap.add_argument("--k-search", type=int, default=8192, help="k indices the pairing may use")
    ap.add_argument("--cands", type=int, default=128, help="random row-pair candidates per step (plus as many similar pairs)")
    ap.add_argument("--seeds", type=int, default=4, help="restarts from the first pairs with the most k pairs (and as many "
                    "with the most shared codes)")
    ap.add_argument("--joint-policies", default="max", help=f"comma list of the joint search's side policies, of {POLICIES}")
    ap.add_argument("--budget-min", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if not set(args.joint_policies.split(",")) <= set(POLICIES):
        ap.error(f"--joint-policies: each of {POLICIES}")
    if args.selftest:
        selftest()
        return
    t0 = time.time()
    census.set_atom(ATOM)
    out_dir = Path(args.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    items = [f"unit:{f}" for f in args.families.split(",") if f]
    if args.acts:
        names = json.loads((Path(args.acts) / "manifest.json").read_text())["inputs"]
        items += [f"acts:{x}" for x in names if re.search(args.acts_layers, x)]
    state_path = out_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"done": [], "starts": {}}
    secs: list[float] = []
    for item in [x for x in items if x not in state["done"]]:
        if time.time() - t0 + (max(secs) if secs else 60.0) > 60 * args.budget_min:
            print(json.dumps({"chunk_end": item, "remaining": len([x for x in items if x not in state["done"]])}), flush=True)
            sys.exit(99)
        state["starts"][item] = state["starts"].get(item, 0) + 1
        state_path.write_text(json.dumps(state))
        if state["starts"][item] > 3:
            print(json.dumps({"failed": item, "reason": "started 3 times without finishing"}), flush=True)
            sys.exit(1)
        ts = time.time()
        res = run_item(item, args)
        (out_dir / f"{item.replace(':', '-')}.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({"done": item, "A": res["A_only"]["largest_block"], "B": res["B_only"]["largest_block"],
                          "joint": res["joint"]["reached_with_32_k_pairs"], "seconds": res["seconds"]}), flush=True)
        state["done"].append(item)
        state_path.write_text(json.dumps(state))
        secs.append(time.time() - ts)
    rows = [json.loads((out_dir / f"{x.replace(':', '-')}.json").read_text()) for x in items]
    joints = [joint_label(p) for p in args.joint_policies.split(",")]
    modes = ["A_only", "B_only", *dict.fromkeys(["joint", *joints])]
    summary = {"tool": "pouw_sm120_pairing", "items": [
        {"item": r["item"], "rows_ok": r["rows_ok"], **{s: r[s]["largest_block"] for s in ("A_only", "B_only")},
         **{j: r[j]["reached_with_32_k_pairs"] for j in modes[2:]}, "fragment": {s: r[s]["fragment"] for s in modes}}
        for r in rows]}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    frags = {s: sum(r[s]["fragment"] for r in rows) for s in modes}
    bestA = max((r["A_only"]["largest_block"] for r in rows), key=lambda b: b[0] * b[2], default=(0, 0, 0))
    bestB = max((r["B_only"]["largest_block"] for r in rows), key=lambda b: b[1] * b[2], default=(0, 0, 0))
    run_result.write(path, True, "sm120-pairing",
                     f"{len(rows)} items; fragments reached " + ", ".join(f"{s} {frags[s]}" for s in modes) +
                     f"; largest A block {bestA[0]} row pairs x {bestA[2]} k pairs, B {bestB[1]} x {bestB[2]} "
                     f"(a fragment needs 16 x 32 and 8 x 32)",
                     [("items", len(rows), "count"), *((f"fragments_{s}", frags[s], "count") for s in modes)])


if __name__ == "__main__":
    main()
