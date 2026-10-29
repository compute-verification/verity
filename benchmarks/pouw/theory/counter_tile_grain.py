"""COUNTER (LEAN program): `pouw-tile-grain`'s falsification test, the measured prices that back charging a tile whole.

`TileGrain` (`Security/Definitions/Pouw/Assumptions/Lead.lean`) charges a committed tile that isn't a whole-tile copy
as its 128 registers, while at W1 one IADD patches one word for one register. W1 doesn't back the charge: it prices a
predicate-only write, a predicated-off instruction and a branch at 0. The card does, and this module states what it
rests on and judges each reading against the price past which copy-and-patch on the lead's own words (`counter_patch`'s
lane model, the prover knowing every match for free) saves more than a budget at a searched call, which is one coin
value: `coinBranchGap` (0.084% of stock), the term `G` carries for it, and `coinSlack` (0.08%), the slack `t` of
`G_tail`'s fixed gap, which carries no copy term, so a saving past the slack at every coin puts the work under
`G_tail`'s threshold on all of them.

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
- `b_star`: a warp-uniform branch skips whole registers (32 words, one a lane), at most `⌊b / 32⌋` of a tile whose
  best source matches `b` words; the least branch price at which those skips fit (`register_bound` is their saving
  with branches free). A branch under `3 · 1,024 − 128 COPY` units makes a 3-register skip pay, which falsifies "only
  a whole m16n8 tile pays" whether or not it breaks the budget.

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
COIN_SLACK = 8 / 10000
BUDGETS = {"coinBranchGap": COIN_BRANCH, "coinSlack": COIN_SLACK}
SOURCES = ("uniform", "own+load")
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


def _least(f, lo: float, hi: float, budget: float, tol: float = 1e-3) -> float | None:
    """The least `x` in `[lo, hi]` with `f(x) ≤ budget`, `f` nonincreasing; None if `f(hi)` exceeds it."""
    if f(hi) > budget:
        return None
    if f(lo) <= budget:
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) <= budget else (mid, hi)
    return hi


def _prices(readings) -> list[float]:
    return sorted({0.0, 13.5} | {r.p for r in readings})


def thresholds(h: dict, stock: int, readings=READINGS, budgets=None) -> dict:
    """Per budget: `p_star` by source, `c_star` at W1's, the predicate's and each reading's decision price, and
    `b_star`, the least warp-branch price at which whole-register skips fit."""
    out: dict = {}
    for name, cap in (budgets or BUDGETS).items():
        t: dict = {}
        for src in ("uniform", "own+load", "own"):
            t[f"p_star/{src}"] = _least(lambda p: share(h, stock, p, 0.0, src), 0.0, 128.0, cap)
            t[f"p_star/{src}/copy0"] = _least(lambda p: share(h, stock, p, 0.0, src, copy=0.0), 0.0, 128.0, cap)
        for src in SOURCES:
            for p in _prices(readings):
                t[f"c_star/{src}/p{p:g}"] = _least(lambda c: share(h, stock, p, c, src), 0.0, H32, cap)
        t["b_star"] = _least(lambda b: register_share(h, stock, b), 0.0, 4 * REGISTER, cap)
        out[name] = t
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
    return {"m": m, "n": n, "k": k, "stock": stock, "thresholds": thresholds(h, stock, readings),
            "judge": judge(h, stock, readings)}


def _star(group: list, budget: str, key: str, sources=SOURCES) -> float | None:
    """A threshold at its worst call and source; None if some call has none (no price in range fits)."""
    vals = [c["thresholds"][budget][key.format(src=src)] for c in group for src in sources]
    return None if None in vals else max(vals)


def _headroom(x: float, star: float | None) -> float | None:
    return x / star - 1 if star else None


def verdict(calls: dict, readings=READINGS, budgets=None) -> dict:
    """The falsification test over the searched calls, per budget and by `K`: each reading's worst copy term, its
    decision and arm headroom (once its arm stops issuing, and at its own decision price), and the warp branch's."""
    budgets = budgets or BUDGETS
    cs = list(calls.values())
    by_k = {k: [c for c in cs if c["k"] == k] for k in sorted({c["k"] for c in cs})}
    forms = {r.name: asdict(r) | {
        "copy_term": max(c["judge"][r.name][src] for c in cs for src in SOURCES),
        "copy_term_if_arms_free": max(c["judge"][r.name]["relaxed"][src] for c in cs for src in SOURCES)}
        for r in readings}
    out: dict = {"forms": forms, "by_budget": {}}
    for name, cap in budgets.items():
        p_star = {k: _star(g, name, "p_star/{src}") for k, g in by_k.items()}
        judged = {}
        for r in readings:
            c_star = {k: _star(g, name, f"c_star/{{src}}/p{r.p:g}") for k, g in by_k.items()}
            judged[r.name] = {
                "holds": forms[r.name]["copy_term"] <= cap,
                "holds_if_arms_free": forms[r.name]["copy_term_if_arms_free"] <= cap,
                "decision_headroom": {k: _headroom(r.p, v) for k, v in p_star.items()},
                "c_star": c_star, "arm_headroom": {k: _headroom(r.c, v) for k, v in c_star.items()}}
        b_star = {k: max(c["thresholds"][name]["b_star"] for c in g) for k, g in by_k.items()}
        at_reading = max(c["thresholds"]["register_share/warp_branch"] for c in cs)
        out["by_budget"][name] = {
            "budget": cap, "p_star": p_star,
            "p_star_own_unpriced": {k: _star(g, name, "p_star/{src}", ("own",)) for k, g in by_k.items()},
            "c_star_p0": {k: _star(g, name, "c_star/{src}/p0") for k, g in by_k.items()},
            "c_star_p13.5": {k: _star(g, name, "c_star/{src}/p13.5") for k, g in by_k.items()},
            "forms": judged,
            "warp_branch": {"b_star": b_star, "headroom": {k: _headroom(WARP_BRANCH.p, v) for k, v in b_star.items()},
                            "share_at_reading": at_reading, "holds": at_reading <= cap},
            "holds": all(f["holds"] for f in judged.values()) and at_reading <= cap,
        }
    out |= {
        "register_bound": {k: max(c["thresholds"]["register_bound"] for c in g) for k, g in by_k.items()},
        "warp_branch": asdict(WARP_BRANCH) | {"only_whole_tiles_pay_from": 3 * REGISTER - 128 * COPY},
        "copy": {"price": COPY, "source": COPY_SOURCE},
        "holds": all(b["holds"] for b in out["by_budget"].values()),
    }
    return out


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
                    print(key, "register bound", f"{100 * t['register_bound']:.5f}%", flush=True)
                    for b in BUDGETS:
                        print(f"  {b}: p* uniform {t[b]['p_star/uniform']} own+load {t[b]['p_star/own+load']} own "
                              f"{t[b]['p_star/own']}, c*(13.5) {t[b]['c_star/uniform/p13.5']}, b* {t[b]['b_star']}",
                              flush=True)
    rep["calls"] = calls
    rep["verdict"] = verdict(calls, readings)
    v = rep["verdict"]
    print(f"register bound by K {v['register_bound']}, holds {v['holds']}", flush=True)
    for b, w in v["by_budget"].items():
        print(f"{b} ({100 * w['budget']:.3f}%): p* by K {w['p_star']}, c*(13.5) by K {w['c_star_p13.5']}, b* by K "
              f"{w['warp_branch']['b_star']}, holds {w['holds']}", flush=True)
        for name, f in w["forms"].items():
            print(f"  {name}: copy term {100 * v['forms'][name]['copy_term']:.5f}%, with free arms "
                  f"{100 * v['forms'][name]['copy_term_if_arms_free']:.5f}% (holds {f['holds_if_arms_free']}), "
                  f"c* at its p by K {f['c_star']}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=1, default=str)


if __name__ == "__main__":
    main()
