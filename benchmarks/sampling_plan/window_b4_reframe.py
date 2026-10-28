"""`window_b4.py`'s planner with Λ per certification period (the golden demo's reframe, BRIEF.md 22:25Z): a period is a
year of the world's inference, about C = 8.8e30 forward FLOP, so the demo window (the B4 window 4,32,4, F_w forward FLOP)
is one of n = C / F_w, and δ is per period.

The period's population is the window's units times n (`Candidate.units`), each window a replay block (as
`optimize.replay_cost` prices a `ReplayDomain`): under lazy commitment (tokens at serve time, interior values when challenged) a drawn unit's witness is rebuilt by
re-running its window, at the window's stock GPU-s. Targets, each at δ ∈ {2^-40, 1%} per period:

  (a) a fraction: free compute at most ε ∈ {1e-3, 1e-4} of the period's work;
  (b) absolute: at most W_max = 2.8e19 FLOP (one optimizer step), ε = W_max / C of the period.

Credited work is ANDs (`window_b4`), so W_max enters as its fraction of the period. Each plan is the planner's (`for_target`,
density bound) under C-Flock's fit and GD-PROVER's warm refit; it reports draws a period and a window, and proving, verifying
(at the CPU price) and replay GPU-s per stock GPU-s. Mode 2 adds the values commitment's serve-time hashing at its
`class_rows` compressions (a row a unit), at GD-HASH's exclusive cost a compression, and `balance` gives each linear
family's grain N where that hashing (falling as 1/N) and proving (rising with N) balance, below and past the cap.

    uv run python benchmarks/sampling_plan/window_b4_reframe.py CENSUS.json --out DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from math import ceil, expm1, fsum, inf, log, log1p
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import window_b4 as W
import window_b4_warm as WW

from verity.core.protocols.sampled_proofs.optimize import Candidate, CostModel, Objective, for_target

#: a year of the world's inference, forward FLOP (BRIEF.md REFRAME)
PERIOD_FLOP = 8.8e30
#: one optimizer step, the absolute target (b)
W_MAX_FLOP = 2.8e19
DELTAS = {"2^-40": 2.0 ** -40, "1%": 0.01}
FRACTIONS = (1e-3, 1e-4)
#: the window's forward FLOP: 2 × (1.36e11 int8 + 3.68e10 LM head MACs), 140 token positions (an estimate from the census)
WINDOW_FLOP = 3.46e11
WINDOW_TOKENS = 140
#: the served step at T 8,192: 26.14 ms (BENCH-34, r…b57f), so 3.19 µs a token on one H200
STOCK_S_A_TOKEN = 26.14e-3 / 8192
#: GD-HASH (note:compute-research/20261028T2045Z-report-gd-hash): h_x at 4 KiB rows, 2,703 MAC-times a byte, a leaf of
#: 42 compressions a 4,096-byte row, a MAC-time 1 / (4,096 × 132 × 1.965e9) s: GPU-s a compression, the tree included
MAC_TIME_S = 1 / (4096 * 132 * 1.965e9)
COMPRESSION_S = 2703 * 4096 / 42 * MAC_TIME_S


def replay_gpu_s(sizes: dict[str, int], chosen: list[Candidate], n: int, stock_w: float) -> float:
    """The windows a draw meets, each re-run at `stock_w`: `optimize.replay_cost` with each window a replay block and a
    template's units spread evenly over the n windows. Computed here, since `_log_miss`'s midpoint form leaves log1p's
    domain when the frontier's scan draws nearly every unit of a population this large."""
    log_miss = fsum(c.units / n * log1p(-min(1.0, sizes[c.template] / c.units)) if sizes[c.template] < c.units else -inf
                    for c in chosen)
    return n * -expm1(log_miss) * stock_w if log_miss > -inf else n * stock_w


def plan(cands: list[Candidate], cm: CostModel, split: dict[str, CostModel], eps: float, delta: float, *, n: int,
         stock_w: float, grid: int) -> dict:
    pc = [replace(c, units=c.units * n) for c in cands]
    F = W.window_work(pc)
    p = for_target(pc, cm, Objective(delta, work=1.0), eps * F, by="free_compute", exact=False, grid=grid)
    chosen = [c for c in pc if c.template in p.sizes]
    draws = sum(p.sizes.values())
    stock = n * stock_w
    replay = replay_gpu_s(dict(p.sizes), chosen, n, stock_w)
    parts = {k: fsum(W.template_cost(c, m, p.sizes[c.template])[0] for c in chosen) for k, m in split.items()}
    grains = {c.op.split(",DOT")[0]: c.cut for c in chosen if c.cut}
    proving = p.cost["statements"] + p.cost["units"]
    return {"eps": eps, "delta": delta, "lambda": log(1 / delta), "n": n, "draws_period": draws,
            "draws_lambda_over_eps": log(1 / delta) / eps, "draws_window": draws / n,
            "eps_certified": p.certificate.free_compute / F, "grains": grains,
            "proving_ands": sum(p.sizes[c.template] * cm.ands(c) for c in chosen),
            "statements_gpu_s": p.cost["statements"], "units_gpu_s": p.cost["units"], "replay_gpu_s": replay,
            "commit_gpu_s": p.cost["commit"], "total_gpu_s": p.cost["total"] + replay, **parts,
            "stock_gpu_s": stock, "per_stock": {"proving+verify": proving / stock, "replay": replay / stock,
                                               "commit": p.cost["commit"] / stock,
                                               "J_inf": (p.cost["total"] + replay) / stock}}


def balance(cands: list[Candidate], cm: CostModel, eps: float, delta: float, stock_w: float) -> list[dict]:
    """Each linear family at each grain N, per stock GPU-s: its serve-time hashing (its units' compressions at
    `COMPRESSION_S`) and its proving (its share Λ/ε · W_f/W of the draws at a drawn block's cost, the statement spread over
    the cap), both constant in n; and N*, where the two would balance, from the family's two widest admitted grains'
    power laws (an extrapolation past the cap)."""
    F = W.window_work(cands)
    lam = log(1 / delta)
    fams: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.cut.startswith("N="):
            fams.setdefault(c.op, []).append(c)
    out = []
    for op, cs in fams.items():
        cs.sort(key=lambda c: int(c.cut[2:]))
        share = cs[0].units * cs[0].work / F
        rows = []
        for c in cs:
            a0, a1 = cm.statement(c)
            per_draw = cm.unit(c) + a1 + a0 / c.cap
            prove = lam / eps * share * per_draw / (PERIOD_FLOP / WINDOW_FLOP * stock_w)
            hashing = c.units * c.commit_rows * COMPRESSION_S / stock_w
            leaf = c.units * (ceil((c.rows[-1] + 17) / W.COMPRESSION) + 9) * COMPRESSION_S / stock_w
            rows.append({"N": int(c.cut[2:]), "units_window": c.units, "proved_ands": cm.ands(c), "cap": c.cap,
                         "per_draw_gpu_s": per_draw, "hash_per_stock": hashing, "hash_leaf_per_stock": leaf,
                         "prove_per_stock": prove, "sum": hashing + prove, "sum_leaf": leaf + prove})
        out.append({"op": op.split(",DOT")[0], "work_share": share, "rows": rows,
                    **{k: _meet(rows, h) for k, h in (("class_rows", "hash_per_stock"), ("leaf", "hash_leaf_per_stock"))}})
    return out


def _meet(rows: list[dict], h: str) -> dict:
    """hash = H N^ah and prove = P N^ap through the two widest admitted grains: N where they're equal, and where their sum
    is least (ah·hash + ap·prove = 0), extrapolated past the cap."""
    lo, hi = rows[-2], rows[-1]
    r = log(hi["N"] / lo["N"])
    ah, ap = log(hi[h] / lo[h]) / r, log(hi["prove_per_stock"] / lo["prove_per_stock"]) / r
    H, P = hi[h] / hi["N"] ** ah, hi["prove_per_stock"] / hi["N"] ** ap
    n_min = (-ah * H / (ap * P)) ** (1 / (ap - ah)) if ah < 0 < ap else None
    return {"exponents": [ah, ap], "N_balance": (H / P) ** (1 / (ap - ah)), "N_min": n_min,
            "J_at_N_min": None if n_min is None else H * n_min ** ah + P * n_min ** ap,
            "best_admitted": min(rows, key=lambda x: x[h] + x["prove_per_stock"])["N"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("census")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--compression-ands", type=int, default=W.FLOCK_SHA512_ANDS)
    args = ap.parse_args()
    before = [W.IN.s8_coord_ands(k) for k in WW.MEASURED_K]
    W.price_openings(args.compression_ands)
    scale = sum(b / W.IN.s8_coord_ands(k) for b, k in zip(before, WW.MEASURED_K)) / len(before)
    census = json.loads(Path(args.census).read_text())
    f = W.fitted()
    cm_warm, split_warm = WW.warm(f.cpu, scale)
    models = {"fitted": (W.costs(), W.parts(f)), "warm": (cm_warm, split_warm)}
    cands = W.candidates(census)
    n = round(PERIOD_FLOP / WINDOW_FLOP)
    stock_w = WINDOW_TOKENS * STOCK_S_A_TOKEN
    h = COMPRESSION_S / W.COMPRESSION
    eps_b = W_MAX_FLOP / PERIOD_FLOP
    targets = {f"(a) eps {e:g}": e for e in FRACTIONS} | {f"(b) W_max {W_MAX_FLOP:g} FLOP": eps_b}
    doc = {"period_flop": PERIOD_FLOP, "window_flop": WINDOW_FLOP, "n": n, "stock_gpu_s_window": stock_w,
           "compression_gpu_s": COMPRESSION_S, "eps_b": eps_b, "compression_ands": args.compression_ands,
           "warm_per_and_scale": scale, "runs": {}, "balance": {}, "n1": {}}
    for name, (cm, split) in models.items():
        for tname, eps in targets.items():
            for dname, delta in DELTAS.items():
                key = f"{name} {tname} delta {dname}"
                doc["runs"][f"{key} mode1"] = plan(cands, cm, split, eps, delta, n=n, stock_w=stock_w, grid=args.grid)
                cm2 = replace(cm, commit_per_row=h * W.COMPRESSION)
                doc["runs"][f"{key} mode2"] = plan(cands, cm2, split, eps, delta, n=n, stock_w=stock_w, grid=args.grid)
                if tname.startswith("(b)"):
                    doc["balance"][key] = balance(cands, cm, eps, delta, stock_w)
        for e in (*FRACTIONS, W.EPS):
            for dname, delta in DELTAS.items():
                doc["n1"][f"{name} eps {e:g} delta {dname}"] = plan(cands, cm, split, e, delta, n=1, stock_w=stock_w,
                                                                    grid=args.grid)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "window_b4_reframe.json").write_text(json.dumps(doc, indent=1, default=str))
    for name, x in (*doc["runs"].items(), *doc["n1"].items()):
        s = x["per_stock"]
        g = ", ".join(f"{op[:16]} {cut}" for op, cut in x["grains"].items())
        print(f"{name:52s} draws {x['draws_period']:.4g} ({x['draws_window']:.3g}/window) eps {x['eps_certified']:.3g} | "
              f"prove+verify {s['proving+verify']:.3g}, replay {s['replay']:.3g}, commit {s['commit']:.3g}, "
              f"J {s['J_inf']:.4g} per stock | {g}")
    for key, fams in doc["balance"].items():
        for fam in fams:
            for m in ("class_rows", "leaf"):
                b = fam[m]
                print(f"balance {key:44s} {fam['op'][:30]:30s} {m:10s} share {fam['work_share']:.3f} best admitted N "
                      f"{b['best_admitted']} N_balance {b['N_balance']:.4g} N_min {b['N_min']:.4g} J there {b['J_at_N_min']:.3g} "
                      f"exps {b['exponents'][0]:.2f},{b['exponents'][1]:.2f}")
            for r in fam["rows"]:
                print(f"    N {r['N']:4d} hash {r['hash_per_stock']:.4g} leaf {r['hash_leaf_per_stock']:.4g} "
                      f"prove {r['prove_per_stock']:.4g} per draw {r['per_draw_gpu_s']:.4g} GPU-s, cap {r['cap']}")


if __name__ == "__main__":
    main()
