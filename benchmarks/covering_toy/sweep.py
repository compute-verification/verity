"""The sweep: each instance's cheapest covering against the bounds, as `k` and `s` vary.

    uv run --no-project --with ortools --with scipy --with numpy --with networkx \\
        python benchmarks/covering_toy/sweep.py --out OUT [--instance NAME ...] [--jobs J] [--budget S]

For each instance, `k` and gate budget `s` (`eta1`: `s` matched to honest inference on `k` words, `bounds.
honest_block_size`; `eta2`: twice that; `inf`: no gate budget) it solves three covering problems, each by the
heuristics and then column generation seeded with their blocks:

- `full`: every non-source gate (Lean's `Covers`);
- `V`: only the counted values (the production bound's object);
- `M`: only `M*`, the chained multiplication blocks at the Lean bound's `D` with `c_w = 1`.

and evaluates every bound in `bounds`. Writes `OUT/<instance>.json` (every point, with each best covering's blocks) and
`OUT/table.md`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bounds as BD  # noqa: E402
import cover as CV  # noqa: E402
import toygraph as TG  # noqa: E402

INSTANCES = {
    "mlp222-b1-t2": dict(dims=(2, 2, 2), batch=1, steps=2),
    "mlp222-b2-t2": dict(dims=(2, 2, 2), batch=2, steps=2),
    "mlp222-b1-t3": dict(dims=(2, 2, 2), batch=1, steps=3),
    "mlp2222-b1-t2": dict(dims=(2, 2, 2, 2), batch=1, steps=2),
    "mlp333-b1-t2": dict(dims=(3, 3, 3), batch=1, steps=2),
    "mlp444-b2-t2": dict(dims=(4, 4, 4), batch=2, steps=2),
    "mlp4444-b1-t2": dict(dims=(4, 4, 4, 4), batch=1, steps=2),
    "mlp4444-b4-t2": dict(dims=(4, 4, 4, 4), batch=4, steps=2),
}
KS = {
    "mlp222-b1-t2": [2, 3, 4, 5, 6, 8, 10, 12],
    "mlp222-b2-t2": [2, 3, 4, 6, 8, 12, 16],
    "mlp222-b1-t3": [2, 3, 4, 6, 8, 12, 16],
    "mlp2222-b1-t2": [2, 3, 4, 6, 8, 12],
    "mlp333-b1-t2": [2, 3, 6, 12],
    "mlp444-b2-t2": [2, 4, 8, 16],
    "mlp4444-b1-t2": [2, 4, 8, 16],
    "mlp4444-b4-t2": [2, 3, 4, 6, 8],
}
MODES = ("eta1", "eta2", "inf")
LARGE = {"mlp333-b1-t2", "mlp444-b2-t2", "mlp4444-b1-t2", "mlp4444-b4-t2"}
#: Instances solved only for `M*` and the per-block caps: the Lean bound's own object, where it is not vacuous.
M_ONLY = {"mlp4444-b4-t2"}


MODE_OVERRIDE: dict[str, tuple[str, ...]] = {}
#: Which covering problems to solve (`full`, `V`, `M`); `--problems` narrows it.
PROBLEMS = {"full", "V", "M"}


def modes(name: str) -> tuple[str, ...]:
    return MODE_OVERRIDE.get(name) or (("eta1", "inf") if name in LARGE else MODES)


def build(name: str) -> TG.Graph:
    return TG.mlp(**INSTANCES[name])


def gate_budget(G: TG.Graph, k: int, mode: str) -> int | None:
    if mode == "inf":
        return None
    return BD.honest_block_size(G, k) * (1 if mode == "eta1" else 2)


def describe(G: TG.Graph, B: CV.Block, targets: set[int]) -> dict:
    """A block's unregistered inputs by kind, and the target gates it computes by step."""
    ins: dict[str, int] = {}
    for g in B.inputs:
        if G.theta[g]:
            continue
        tg = G.tag[g]
        if G.op[g] == "data":
            kind = f"data@{tg['step']}"
        elif tg.get("kind") == "w":
            kind = f"w@{tg['step'] + 1}"
        elif tg.get("act"):
            kind = f"act@{tg['step']}"
        elif "value" in tg:
            kind = f"pre@{tg['step']}"
        elif tg.get("phase") in ("bwd",):
            kind = f"bwd@{tg['step']}"
        else:
            kind = f"{tg.get('phase', G.op[g])}@{tg.get('step')}"
        ins[kind] = ins.get(kind, 0) + 1
    by_step: dict[int, int] = {}
    for g in B.computed & targets:
        st = G.tag[g]["step"]
        by_step[st] = by_step.get(st, 0) + 1
    return {"inputs": dict(sorted(ins.items())), "targets_by_step": dict(sorted(by_step.items())),
            "size": B.size(), "unregistered": B.unregistered(G)}


def solve(G: TG.Graph, k: int, s: int | None, targets: list[int] | None, budget: float, workers: int) -> dict:
    large = G.N > 400
    h = CV.best_heuristic(G, k, s, targets=targets, restarts=2 if large else 4, iters=40 if large else 120)
    cols = list(h["cover"] or [])
    cg = CV.colgen(G, k, s, cols, targets=targets, price_time=min(15.0, budget / 4), ip_time=min(120.0, budget / 2),
                   workers=workers, time_budget=budget)
    cands = [c for c in (h["cover"], cg.get("cover")) if c]
    best = min(cands, key=len)
    tset = set(G.inner() if targets is None else targets)
    return {
        "n": len(best),
        "lb": cg["lb"],
        "optimal": cg["lb"] >= len(best),
        "lp": cg["lp"],
        "lp_closed": cg["lp_closed"],
        "heuristics": {a: b for a, b in h.items() if a != "cover"},
        "colgen_n": cg.get("n"),
        "blocks": [describe(G, B, tset) for B in sorted(best, key=lambda b: -len(b.computed & tset))],
        "cover": [[sorted(B.computed), sorted(B.inputs)] for B in best],
    }


def cw_measured(G: TG.Graph, s: int | None, workers: int, qs: list[int] | None = None) -> dict:
    """`c_w(s) = min_q f(q) / q`, `f(q)` the fewest unregistered inputs completing products of `q` slots. Uses CP-SAT's
    proven lower bound on `f`, so the result is a valid parameter-hardness constant. On sampled `qs`, `f` being
    nondecreasing gives `f(q) / q >= f(q_i) / q_{i+1}` for `q_i <= q <= q_{i+1}`, which is what `cw` takes; `cw_sampled`
    is the plain minimum over the samples."""
    Q = len({key[0] for key in G.mul})
    qs = sorted(set(qs or range(1, Q + 1)) | {1, Q})
    pts = {}
    for q in qs:
        r = CV.min_inputs_for_slots(G, s, q, time_limit=20.0, workers=workers)
        pts[q] = {"f": r["n"], "f_lb": r["lb"], "optimal": r["optimal"]}
    lows = [pts[a]["f_lb"] / b for a, b in zip(qs, qs[1:])] + [pts[Q]["f_lb"] / Q]
    q_best = min(pts, key=lambda q: pts[q]["f_lb"] / q)
    return {"cw": min(lows), "cw_sampled": pts[q_best]["f_lb"] / q_best, "q": q_best, "f": pts}


def point(args) -> dict:
    name, k, mode, budget, workers, cw = args
    G = build(name)
    s = gate_budget(G, k, mode)
    out = {"instance": name, "k": k, "mode": mode, "s": s, "gates": G.N, "inner": len(G.inner())}
    if name not in M_ONLY:
        if "full" in PROBLEMS:
            out["full"] = solve(G, k, s, None, budget, workers)
        if "V" in PROBLEMS:
            out["V"] = solve(G, k, s, G.values, budget, workers)
    lean1 = BD.lean_two_costs(G, k, cw=1.0)
    mstar = BD.chained(G, lean1["D"])
    out["lean_cw1"] = lean1
    out["lean_add_cw1"] = BD.lean_additive(G, k, cw=1.0)
    out["M"] = solve(G, k, s, mstar, budget, workers) if mstar and "M" in PROBLEMS else None
    out["cap"] = {} if name in M_ONLY or "V" not in PROBLEMS else {
        "inner": CV.block_capacity(G, k, s, G.inner(), time_limit=30.0, workers=workers),
        "V": CV.block_capacity(G, k, s, G.values, time_limit=30.0, workers=workers),
    }
    if mstar:
        out["cap"]["M"] = CV.block_capacity(G, k, s, mstar, time_limit=60.0, workers=workers)
    out["production_beta1"] = BD.production(G, k, 1.0)
    out["trivial"] = BD.trivial(G, s)
    if cw is not None:
        out["cw"] = cw
        lm = BD.lean_two_costs(G, k, cw=cw["cw"]) if cw["cw"] > 0 else None
        out["lean_measured"] = lm
        if lm and lm["M_star"] and "M" in PROBLEMS:
            ms = BD.chained(G, lm["D"])
            out["cap"]["M_measured"] = CV.block_capacity(G, k, s, ms, time_limit=30.0, workers=workers)
            out["M_measured"] = solve(G, k, s, ms, budget, workers)
    if s is not None and "full" in out:
        out["rho"] = s * out["full"]["n"] / G.N
        out["rho_lb"] = s * out["full"]["lb"] / G.N
    return out


def run(names: list[str], outdir: Path, jobs: int, budget: float, workers: int, cw_instances: set[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for name in names:
        G = build(name)
        cws = {}
        if name in cw_instances:
            Q = len({key[0] for key in G.mul})
            qs = None if Q <= 16 else sorted({1, 2, 4, 8, 12, 16, 24, 32, 40, Q} & set(range(1, Q + 1)))
            for k in KS[name]:
                for mode in modes(name):
                    s = gate_budget(G, k, mode)
                    if s not in cws:
                        cws[s] = cw_measured(G, s, workers, qs)
                        print(f"{name} c_w(s={s}) = {cws[s]['cw']:.3f}", flush=True)
        meta = {"instance": name, **G.summary(), "upstream": sorted(BD.upstream(G).values()),
                "cw": {str(s): v for s, v in cws.items()}}
        (outdir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1))
        for k in KS[name]:
            for mode in modes(name):
                tasks.append((name, k, mode, budget, workers, cws.get(gate_budget(G, k, mode))))
    results: dict[str, list] = {n: [] for n in names}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(point, tasks):
            results[r["instance"]].append(r)
            sol = r.get("full") or r.get("M") or {"n": None, "lb": None}
            print(f"{r['instance']} k={r['k']} {r['mode']} s={r['s']}: n={sol['n']} lb={sol['lb']}", flush=True)
            (outdir / f"{r['instance']}.json").write_text(json.dumps(results[r["instance"]], indent=1))
    (outdir / "table.md").write_text(table(outdir, names))


def _br(sol: dict | None) -> str:
    if not sol:
        return "-"
    return f"{sol['n']}" if sol["optimal"] else f"[{sol['lb']}, {sol['n']}]"


def why(G: TG.Graph, comp: set[int], ins: set[int], targets: set[int]) -> str:
    """Which section-4 assumption a block that beats a word-per-value count breaks, from the gates it computes."""
    steps = {G.tag[g]["step"] for g in comp & targets}
    if max(steps, default=0) >= 1 and any(G.tag[g].get("phase") == "upd" for g in comp):
        return "decoding (it recomputes the update from earlier steps' data within s)"
    if steps <= {0}:
        return "cap (cheap inputs: the registered weights' values on token data)"
    if any(G.tag[g].get("kind") == "w" and G.preds[g] for g in ins):
        return "cap (a tile: the update's words plus cheap token inputs)"
    return "cap (cheap inputs: activations and partial sums it reads)"


def densest(G: TG.Graph, sol: dict, targets: set[int]) -> tuple[set[int], set[int]]:
    """The block of a solution with the most target gates per unregistered input."""
    best = max(sol["cover"], key=lambda b: len(set(b[0]) & targets) / max(1, sum(not G.theta[g] for g in b[1])))
    return set(best[0]), set(best[1])


def flags(r: dict) -> list[str]:
    G = build(r["instance"])
    out = []
    k = r["k"]
    V = set(G.values)
    dV = why(G, *densest(G, r["V"], V), V)
    if r["V"]["n"] < r["production_beta1"]["n_lb"] - 1e-9:
        out.append("P-total: " + dV)
    if r["cap"]["V"]["n"] and r["cap"]["V"]["n"] > k:
        out.append(f"P-block: one block computes {r['cap']['V']['n']} values from {k} words; densest V-block: " + dV)
    if r.get("M") and r["M"]["n"] < r["lean_cw1"]["n_lb"] - 1e-9:
        M = set(BD.chained(G, r["lean_cw1"]["D"]))
        out.append("L-c_w=1: " + why(G, *densest(G, r["M"], M), M))
    lm = r.get("lean_measured")
    if lm and r.get("M_measured") and r["M_measured"]["n"] < lm["n_lb"] - 1e-9:
        out.append("L-measured: THEOREM VIOLATED (a solver or statement bug)")
    return out


def table(outdir: Path, names: list[str]) -> str:
    lines = [
        "| instance | k | s | n* | rho | n*/trivial | n*_V | \\|V\\|/k | cap_V | \\|M*\\| (D) | n*_M | Lean c_w=1 | c_w(s) | Lean c_w(s) | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        p = outdir / f"{name}.json"
        if not p.exists():
            continue
        for r in sorted(json.loads(p.read_text()), key=lambda r: (r["k"], MODES.index(r["mode"]))):
            lean1 = r["lean_cw1"]
            lm = r.get("lean_measured")
            rho = f"{r['rho']:.2f}" if "rho" in r else "-"
            triv = f"{r['full']['n'] / r['trivial']:.1f}" if r["trivial"] else "-"
            cw = f"{r['cw']['cw']:.2f}" if r.get("cw") else "-"
            lms = f"{lm['n_lb']:.2f} (D={lm['D']}, M*={lm['M_star']}, n*_M={_br(r.get('M_measured'))})" if lm else "-"
            lines.append(
                f"| {name} | {r['k']} | {r['s'] if r['s'] is not None else 'inf'} | {_br(r['full'])} | {rho} | {triv} | "
                f"{_br(r['V'])} | {r['production_beta1']['n_lb']:.2f} | {r['cap']['V']['ub']} | "
                f"{lean1['M_star']} ({lean1['D']}) | {_br(r['M'])} | {lean1['n_lb']:.2f} | {cw} | {lms} | "
                f"{'; '.join(f.split(':')[0] for f in flags(r)) or '-'} |"
            )
    return "\n".join(lines) + "\n"


def caps(name: str, ks: list[int], workers: int, cw: dict) -> list[dict]:
    """The Lean bound's counting step, per block: the most `M*` products one `(s, k)`-block computes (CP-SAT's proven
    cap) against `(k / c_w)(k / c_a) + k`, at `c_w = 1` and at the measured `c_w(s)` (`cw`: `s` to `c_w`), with `M*` at
    each one's smallest admissible `D`; and the most forward products of updated weights, chained or not (`P`)."""
    G = build(name)
    P = list(G.mul.values())
    rows = []
    for k in ks:
        for mode in modes(name):
            s = gate_budget(G, k, mode)
            r = {"instance": name, "k": k, "mode": mode, "s": s,
                 "cap_P": CV.block_capacity(G, k, s, P, time_limit=60.0, workers=workers)}
            for label, c in (("cw1", 1.0), ("cw_meas", cw.get(s))):
                if not c:
                    continue
                lb = BD.lean_two_costs(G, k, cw=c)
                ms = BD.chained(G, lb["D"])
                r[label] = {**lb, "cw": c,
                            "cap": CV.block_capacity(G, k, s, ms, time_limit=60.0, workers=workers) if ms else None}
            rows.append(r)
            print(json.dumps(r), flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--instance", action="append")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=120.0, help="column generation's seconds per covering problem")
    ap.add_argument("--cw", action="append", default=None, help="instances whose c_w(s) to measure (default: the smallest)")
    ap.add_argument("--ks", type=lambda t: [int(x) for x in t.split(",")], help="override every instance's k grid")
    ap.add_argument("--modes", type=lambda t: tuple(t.split(",")), help="override the gate budgets (eta1,eta2,inf)")
    ap.add_argument("--problems", type=lambda t: set(t.split(",")), help="solve only these (full,V,M)")
    ap.add_argument("--caps", action="store_true", help="only the per-block cap test (`caps`), from OUT's meta files")
    a = ap.parse_args()
    names = a.instance or list(INSTANCES)
    if a.problems:
        PROBLEMS.intersection_update(a.problems)
    for name in names:
        if a.ks:
            KS[name] = a.ks
        if a.modes:
            MODE_OVERRIDE[name] = a.modes
    if a.caps:
        out = []
        for name in names:
            meta = a.out / f"{name}.meta.json"
            cw = {}
            if meta.exists():
                cw = {(None if s == "None" else int(s)): v["cw"] for s, v in json.loads(meta.read_text())["cw"].items()}
            out += caps(name, KS[name], a.workers, cw)
        (a.out / "caps.json").write_text(json.dumps(out, indent=1))
        return
    cw = set(a.cw) if a.cw else {"mlp222-b1-t2", "mlp2222-b1-t2", "mlp4444-b1-t2"}
    run(names, a.out, a.jobs, a.budget, a.workers, cw)


if __name__ == "__main__":
    main()
