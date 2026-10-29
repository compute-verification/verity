"""COUNTER (LEAN program): `pouw-tile-grain`'s falsification test, the measured prices that back charging a tile whole.

`TileGrain` (`Security/Definitions/Pouw/Assumptions/Lead.lean`) charges a committed tile that isn't a whole-tile copy
as its 128 registers, while at W1 one IADD patches one word for one register. W1 doesn't back the charge: it prices a
predicate-only write, a predicated-off instruction and a branch at 0. The card does, and this module states what it
rests on and judges each reading against the price past which copy-and-patch on the lead's own words (`counter_patch`'s
lane model, the prover knowing every match for free) saves more than `coinBranchGap` (0.084% of stock) at a searched
call.

**A skip form.** A lane of the m16n8 fragment holds 4 words of a tile copied whole from a source inside its coin tile,
at `COPY` a word. It keeps its `k` matching words and patches the rest at 32 (one W1 register each). A form that
skips the patches costs `p` a lane beyond its arms (the decision) and `c` a skipped word (the patch arm issuing
anyway), so a lane saves `max(0, k (32 − c) − p)` and a tile `max(0, Σ_lanes − 128 COPY)`, against patching all 128.
Sources: `uniform`, one a tile (a warp-uniform choice, unpriced); `own+load`, each lane its own at 32 more (a
data-dependent load); `own`, each lane its own unpriced, a bound past what's realizable.

**What falsifies it** (`judge`): a measured skip form whose `(p, c)` puts the copy term above `coinBranchGap` at any
searched call, `uniform` or `own+load`. Every measured form issues the patch arm it would skip (`c = 32`), so it saves
nothing a lane at any `K`; that is the first line. The boundary is computed on the lead's words (`thresholds`, by `K`):

- `p_star`: with `c = 0` (an arm that doesn't issue), the least decision price that fits. A decision cheaper than it
  falsifies the charge once arms stop issuing. It grows as `K` shrinks: early steps' words match more.
- `c_star`: at the cheapest measured decision (a predicate, `p = 13.5`) and at W1's (`p = 0`), the least cost of a
  skipped word's arm that fits. A predicated-off patch or an untaken divergent arm cheaper than it falsifies.
- A warp-uniform branch skips whole registers (32 words, one a lane), which no per-lane form needs. With branches
  free they save at most `register_bound`, inside `coinBranchGap`, so no branch price falsifies the bound. A branch
  under `3 · 1,024 − 128 COPY` units makes a 3-register skip pay, which falsifies "only a whole m16n8 tile pays"
  but not `G`.

The readings (`READINGS`) are units of W1's issue pricing (32 a register), measured alone where issue slots bind.
Beside the lead's `wgmma` a selection costs 0.12 a lane-entry and a copy 0.40 a word (N-FLOOR time7, time6): the
figure is priced W1 work, so time beside `wgmma` doesn't enter, except the copy, taken at its cheapest.

`python counter_tile_grain.py --json OUT` runs the searched calls (`counter_patch`'s spread columns, in-tile sources);
`--readings FILE` judges new measurements (a JSON list of `READINGS`' fields) in place of these.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_patch as cpatch

H32 = cpatch.H32
COPY = cpatch.COPY
LOAD = cpatch.OWN_LOAD
COIN_BRANCH = cpatch.COIN_BRANCH
REGISTER = 32 * H32
HEAD = "d5263f864b"


@dataclass(frozen=True)
class Reading:
    """A skip form as measured: `p` a lane beyond its arms, `c` a word it skips, and where it was measured."""

    name: str
    p: float
    c: float
    what: str
    source: str


READINGS = (
    Reading("selp", 61.8, H32, "selp on a warp-uniform bit, both arms computed (1,978 a warp decision)",
            "N-FLOOR `sel`, time1 r20261029-095917-70b4, time2 r20261029-100522-711d"),
    Reading("lane-bit", 71.8, H32, "@p bra on the lane's own bit, which ptxas if-converts: both arms issue predicated",
            "N-FLOOR `brd`, time1 r20261029-095917-70b4, time2 r20261029-100522-711d"),
    Reading("predicate", 13.5, H32, "a word steered by a predicate its digit's LOP3 writes: the predicated IMAD issues "
            "for the warp anyway", "N-FLOOR time5 r20261029-110224-c481"),
    Reading("divergent", 70.9, H32, "a divergent coin branch, if-converted: 326.88 a lane with its 8 arm ops, 70.9 "
            "beyond them; with BSSY/BSYNC both arms run (b_dbra16, 1,346 a lane)", "N-W1 b_dbra r20261029-094825-8394"),
)
WARP_BRANCH = Reading("warp-branch", 3814.0, 0.0, "@p bra.uni between two real arms (`lu`), a warp decision",
                      "N-FLOOR time1 r20261029-095917-70b4")
COPY_SOURCE = "N-FLOOR time6 r20261029-110822-fc03 (a tile copied beside the lead's wgmma, 0.40 a word)"


def histograms(w: np.ndarray) -> dict:
    """Per tile, the lanes by matching words `k = 1..4`: `uniform` a source (offset, s, n/8, 4), `own` at each lane's
    best source (s, n/8, 4), and `best`, the most words a source matches."""
    m, s, n = w.shape
    out = []
    for rb in range(m // 16):
        offs = cpatch._offsets(rb, m)
        if not offs:
            out.append(None)
            continue
        counts = np.stack([cpatch.lane_counts(w, rb, d) for d in offs])
        uni = np.stack([(counts == k).sum(axis=(1, 4)) for k in range(1, 5)], axis=-1).astype(np.int16)
        own = np.stack([(counts.max(axis=0) == k).sum(axis=(0, 3)) for k in range(1, 5)], axis=-1).astype(np.int16)
        out.append({"uniform": uni, "own": own, "best": counts.sum(axis=(1, 4)).max(axis=0)})
    return {"blocks": out}


def _lane_gain(p: float, c: float) -> np.ndarray:
    k = np.arange(1, 5, dtype=np.float64)
    return np.maximum(0.0, k * (H32 - c) - p)


def share(h: dict, stock: int, p: float, c: float, src: str, copy: float = COPY) -> float:
    """The copy term (a share of stock) of the skip form `(p, c)` with sources `src`, each tile on its own (`r1`)."""
    tot = 0.0
    for b in h["blocks"]:
        if b is None:
            continue
        if src == "uniform":
            gain = (b["uniform"] @ _lane_gain(p, c)).max(axis=0)
        else:
            gain = b["own"] @ _lane_gain(p + (LOAD if src == "own+load" else 0.0), c)
        tot += float(np.maximum(0.0, gain - 128 * copy).sum())
    return tot / stock


def register_share(h: dict, stock: int, branch: float = 0.0, copy: float = COPY) -> float:
    """Whole-register skips under a warp-uniform branch of price `branch`: a tile whose best source matches `b` words
    has at most `⌊b / 32⌋` registers whose 32 words all match, and saves at most `r · 1,024 − 128 COPY − branch`."""
    tot = 0.0
    for b in h["blocks"]:
        if b is None:
            continue
        r = b["best"] // 32
        tot += float(np.maximum(0.0, np.where(r > 0, r * REGISTER - 128 * copy - branch, 0.0)).sum())
    return tot / stock


def _least(f, lo: float, hi: float, tol: float = 1e-3) -> float | None:
    """The least `x` in `[lo, hi]` with `f(x) ≤ coinBranchGap`, `f` nonincreasing; None if `f(hi)` exceeds it."""
    if f(hi) > COIN_BRANCH:
        return None
    if f(lo) <= COIN_BRANCH:
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) <= COIN_BRANCH else (mid, hi)
    return hi


def thresholds(h: dict, stock: int) -> dict:
    out = {}
    for src in ("uniform", "own+load", "own"):
        out[f"p_star/{src}"] = _least(lambda p: share(h, stock, p, 0.0, src), 0.0, 128.0)
        out[f"p_star/{src}/copy0"] = _least(lambda p: share(h, stock, p, 0.0, src, copy=0.0), 0.0, 128.0)
        for p in (0.0, 13.5):
            out[f"c_star/{src}/p{p:g}"] = _least(lambda c: share(h, stock, p, c, src), 0.0, H32)
    out["register_bound"] = register_share(h, stock)
    out["register_bound/copy0"] = register_share(h, stock, copy=0.0)
    out["register_share/warp_branch"] = register_share(h, stock, WARP_BRANCH.p)
    return out


def judge(h: dict, stock: int, readings=READINGS) -> dict:
    """Each reading's copy term at `uniform` and `own+load` sources (and `own`, past what's realizable)."""
    return {r.name: {src: share(h, stock, r.p, r.c, src) for src in ("uniform", "own+load", "own")}
            | {"relaxed": {src: share(h, stock, r.p, 0.0, src) for src in ("uniform", "own+load", "own")}}
            for r in readings}


def call(m: int, n: int, k: int, seed: int, pattern: str, readings=READINGS) -> dict:
    w = cpatch.instance(m, n, k, seed, pattern)
    stock = m * n * k
    h = histograms(w)
    del w
    return {"m": m, "n": n, "k": k, "stock": stock, "thresholds": thresholds(h, stock),
            "judge": judge(h, stock, readings)}


def _worst(calls) -> dict:
    th = {key: [c["thresholds"][key] for c in calls] for key in calls[0]["thresholds"]}
    return {key: max(vals) if key.startswith("register") or None not in vals else None for key, vals in th.items()}


def verdict(calls: dict, readings=READINGS) -> dict:
    """The falsification test over the searched calls: each threshold at its worst call (and by `K`), and each reading's
    worst copy term against `coinBranchGap`."""
    worst = _worst(list(calls.values()))
    by_k = {k: _worst([c for c in calls.values() if c["k"] == k]) for k in sorted({c["k"] for c in calls.values()})}
    forms = {}
    for r in readings:
        realizable = max(c["judge"][r.name][src] for c in calls.values() for src in ("uniform", "own+load"))
        relaxed = max(c["judge"][r.name]["relaxed"][src] for c in calls.values() for src in ("uniform", "own+load"))
        forms[r.name] = asdict(r) | {"copy_term": realizable, "copy_term_if_arms_free": relaxed,
                                     "holds": realizable <= COIN_BRANCH}
    def star(w: dict, key: str) -> float | None:
        vals = [w[f"{key}/{src}" if key == "p_star" else f"c_star/{src}/p13.5"] for src in ("uniform", "own+load")]
        return None if None in vals else max(vals)

    p_star, c_star = star(worst, "p_star"), star(worst, "c_star")
    least_p = min(readings, key=lambda r: r.p)
    least_c = min(readings, key=lambda r: r.c)
    letter = 3 * REGISTER - 128 * COPY
    return {
        "coin_branch_gap": COIN_BRANCH, "worst": worst, "by_k": by_k, "forms": forms,
        "p_star": p_star, "c_star": c_star,
        "p_star_by_k": {k: star(w, "p_star") for k, w in by_k.items()},
        "c_star_by_k": {k: star(w, "c_star") for k, w in by_k.items()},
        "least_decision": {"name": least_p.name, "p": least_p.p,
                           "headroom": least_p.p / p_star - 1 if p_star else None},
        "least_arm": {"name": least_c.name, "c": least_c.c, "headroom": least_c.c / c_star - 1 if c_star else None},
        "warp_branch": asdict(WARP_BRANCH) | {"bound_with_branches_free": worst["register_bound"],
                                              "only_whole_tiles_pay_from": letter,
                                              "headroom": WARP_BRANCH.p / letter - 1},
        "copy": {"price": COPY, "source": COPY_SOURCE},
        "holds": all(f["holds"] for f in forms.values()) and worst["register_bound"] <= COIN_BRANCH,
    }


POINTS = cpatch.POINTS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", nargs="+", default=list(POINTS))
    ap.add_argument("--ks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--ks-8192", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--cols", type=int, default=256)
    ap.add_argument("--cols-8192", type=int, default=64)
    ap.add_argument("--patterns", nargs="+", default=["zero", "a2c", "a2r"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seeds-8192", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--spread", type=int, default=4096)
    ap.add_argument("--readings", default=None, help="a JSON list of readings ({name, p, c, what, source})")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    readings = READINGS
    if args.readings:
        with open(args.readings) as f:
            readings = tuple(Reading(**r) for r in json.load(f))
    cpatch.PARTIAL = True
    cpatch.use_head(HEAD)
    rep: dict = {"config": vars(args), "head": HEAD, "port": cpatch.check_port()}
    assert rep["port"]["word_mismatches"] == 0 and rep["port"]["tags_equal"] and rep["port"]["code_equal"]
    calls = {}
    for name in args.points:
        m = POINTS[name]
        n = args.cols_8192 if m == 8192 else args.cols
        cpatch.COLUMNS = cpatch.spread_columns(n, args.spread)
        for k in args.ks_8192 if m == 8192 else args.ks:
            for pattern in args.patterns if m != 8192 else args.patterns[:1]:
                for seed in args.seeds_8192 if m == 8192 else args.seeds:
                    key = f"{name}/K{k}/{pattern}/s{seed}"
                    calls[key] = call(m, n, k, seed, pattern, readings)
                    t = calls[key]["thresholds"]
                    print(f"{key}: p* uniform {t['p_star/uniform']} own+load {t['p_star/own+load']} own "
                          f"{t['p_star/own']}, c*(13.5) {t['c_star/uniform/p13.5']}, register bound "
                          f"{100 * t['register_bound']:.5f}%", flush=True)
    rep["calls"] = calls
    rep["verdict"] = verdict(calls, readings)
    v = rep["verdict"]
    print(f"p* {v['p_star']} by K {v['p_star_by_k']} (least decision {v['least_decision']}), c* {v['c_star']} by K "
          f"{v['c_star_by_k']}, register bound {100 * v['worst']['register_bound']:.5f}%, holds {v['holds']}",
          flush=True)
    for name, f in v["forms"].items():
        print(f"  {name}: copy term {100 * f['copy_term']:.5f}%, with free arms "
              f"{100 * f['copy_term_if_arms_free']:.5f}%, holds {f['holds']}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
