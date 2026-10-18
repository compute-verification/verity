"""The sampling plan for the int8 Llama-3.1-8B row under C-Flock (`verity.core.protocols.sampled_proofs`): the fitted cost
model, the cuts the generator finds, the frontier of ε against the verification overhead B, the break-even granularity
by window, ε and commitment regime, and the frontier's sensitivity to its inputs.

Units are decision 106's: every s8 dot unit checks its seed and derives its own noise (2 + 2 ceil(L b / 512) SHA-512
compressions), opens its operand rows and its output in circuit, and carries its own work (MACs, W_ref's proxy).  A dot
may be cut into K-chunks (`S8Dot{T0 = input}`), each committing its running sum; tiles of r x c dots are offered so the
width rule's price is reported.  B is GPU-seconds of overhead per plain GPU-second over a window (BUDGET's unit).

    uv run python benchmarks/sampling_plan/int8_llama.py --out DIR      # DIR/plan.json, DIR/cost-model.json, tables
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from math import ceil, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import (A6_ANDS, EPILOGUE, HEAD_DIM, HEADS, I, KEY, LAYERS, LINEARS, S8_STEP, SHA512, observations)  # noqa: E402

from verity.core.protocols.sampled_proofs.candidates import Shape, cuts, pressure  # noqa: E402
from verity.core.protocols.sampled_proofs.costfit import Fitted, fit  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (Candidate, CostModel, Objective, ReplayDomain,  # noqa: E402
                                                           for_budget, for_target, frontier, needs_benchmark, perturb,
                                                           row_ands, structure)

DELTA = 2.0 ** -40
CHUNKS = (2048, 1024, 512, 256, 128, 64, 32)
WINDOWS = {"round": 1, "hour": 3600 / 12.6, "day": 86400 / 12.6, "week": 48_000}
TARGETS = (1e-3, 1e-4)
REGIMES = (("plain", 1.0, 1.0), ("prefix", 0.0, 1.0), ("prefix, sessions/256", 0.0, 1 / 256))
BUDGETS = (0.065, 0.07, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0)
OPERATING = 0.1


def fitted(v: dict) -> Fitted:
    obs = observations()
    sweep = [o for o in obs if o.source.startswith("proof_throughput")]
    slope = fit(sweep, "prove", ["one", "ands*drawn"]).coef["ands*drawn"] * v["zk_unit"]
    zk = [o for o in obs if o.source == "r20261017-202742-da01"]
    prove = fit(zk, "prove", ["one", "ands", "ands*drawn"], fixed={"ands*drawn": slope})
    verify = fit([o for o in obs if o.verify is not None and not o.source.startswith("proof")], "verify",
                 ["one", "ands*staged"])
    stage = fit(zk, "stage", ["one", "ands*staged"])
    return Fitted(prove, verify, stage, v["cpu"], KEY)


def cap(ands: int) -> int:
    """A statement's units under m = k_log + nbl <= 35, k_log = ceil(log2 ands) + 1 (the sweep's 25 at 14.8M ANDs and
    A6's 23 at K = 2048; A6's K = 8192 cap is twice this)."""
    return 1 << max(0, 35 - (ceil(log2(ands)) + 1))


def s8_unit(name: str, length: int, out_bits: int, *, extra_rows: tuple[int, ...], commit_bits: int, bits: int,
            ways=()) -> Shape:
    noise = 2 + 2 * ceil(length * bits / 512)
    rows = (length, length) + extra_rows
    ands = (length // 32) * S8_STEP + noise * SHA512 + (EPILOGUE if out_bits == 16 else 0)
    total = ands + sum(row_ands(r) for r in rows)
    return Shape(name, ands, length, out_bits, 1, rows, commit_bits, 0.0, cap=min(cap(total), 1 << 20), ways=ways)


def linear(k: int, bits: int, chunks=CHUNKS, tiles=((2, 1), (16, 16))) -> Shape:
    """One output coordinate of `ScaledMmS8Coordinate{K}` with its noise, its scales' rows and its bf16 output, cut into
    K-chunks that carry the running sum; the root is a 16 x 16 tile (refused by the width rule) over 2 x 1 tiles."""
    ways = tuple((f"K={c}", ((s8_unit(f"S8Chunk{{K={k},L={c}}}", c, 32, extra_rows=(4, 4),
                                      commit_bits=round(32 * (1 - c / k)), bits=bits), k // c),))
                 for c in chunks if c < k and k % c == 0)
    coord = s8_unit(f"S8Coord{{K={k}}}", k, 16, extra_rows=(2, 4, 4), commit_bits=0, bits=bits, ways=ways)
    shape = coord
    for r, c in sorted(tiles, key=lambda t: t[0] * t[1]):
        n = r * c
        tile_ands = n * (coord.ands - (2 + 2 * ceil(k * bits / 512)) * SHA512) \
            + (2 + (r + c) * ceil(k * bits / 512)) * SHA512
        shape = Shape(f"S8Tile{{K={k},{r}x{c}}}", tile_ands, n * k, 16 * n, n, (k,) * (r + c) + (2 * n, 4 * r, 4 * c),
                      0, 0.0, cap=min(cap(tile_ands + sum(row_ands(x) for x in (k,) * (r + c))), 1 << 20),
                      parts=((shape, n // (shape.work // k)),))
    return shape


def pv(length: int, bits: int) -> Shape:
    ways = tuple((f"K={c}", ((s8_unit(f"PvChunk{{L={length},C={c}}}", c, 32, extra_rows=(4, 4),
                                      commit_bits=round(32 * (1 - c / length)), bits=bits), length // c),))
                 for c in CHUNKS if c < length and length % c == 0)
    return s8_unit(f"PV{{L={length}}}", length, 32, extra_rows=(4,), commit_bits=0, bits=bits, ways=ways)


def classes(context: int) -> dict[int, float]:
    """P.V's padded power-of-two length classes (from 32) and each one's share of positions uniform on 1..context."""
    out, lo = {}, 0
    length = 32
    while lo < context:
        hi = min(length, context)
        out[length] = (hi - lo) / context
        lo, length = hi, length * 2
    return out


def workload(v: dict, window_rounds: float) -> dict:
    """Per op: its root shape, its instances over the window, the replay domain's units, and the window's work."""
    bits = int(v["noise_bits"])
    tokens = v["tokens_per_s"] * v["round"] * window_rounds
    step = int(v["step_tokens"])
    steps = max(1, round(tokens / step))
    replay_units = steps * LAYERS
    ops = {}
    for k, coords in LINEARS.items():
        root = linear(k, bits)
        ops[f"linear K={k}"] = (root, step * coords * k // root.work)
    shares = classes(int(v["context"]))
    mean_pos = (int(v["context"]) + 1) / 2
    ops["attention S"] = (s8_unit("S{K=128}", HEAD_DIM, 32, extra_rows=(4,), commit_bits=0, bits=bits),
                          round(step * HEADS * mean_pos))
    for length, share in shares.items():
        per_step = round(step * share) * HEADS * HEAD_DIM
        if per_step:
            ops[f"attention PV L={length}"] = (pv(length, bits), per_step)
    total_work = sum(root.work * per for root, per in ops.values()) * replay_units
    return {"ops": ops, "replay_units": replay_units, "tokens": tokens, "work": total_work,
            "window_gpu_s": v["round"] * window_rounds}


def model(v: dict, fit_: Fitted, replay_units: int, *, commit: float = 1.0, statement: float = 1.0) -> CostModel:
    m = fit_.model()
    m = replace(m, commit_per_bit=v["commit_per_byte"] / 8 * commit,
                replay={"layer-step": ReplayDomain(replay_units, v["replay_layer_step"])})
    return perturb(m, "statement", statement) if statement != 1.0 else m


def candidates(w: dict, costs: CostModel, objective: Objective, grid: int = 31):
    out, alts, hints = [], {}, []
    for op, (root, per) in w["ops"].items():
        cs, a, h = cuts(op, root, per * w["replay_units"], costs, objective, grid=grid,
                        replay=("layer-step", w["replay_units"]))
        out += cs
        alts[op] = {label: [list(u) for u in units] for label, units in a.items()}
        hints += [{"op": op, **h.__dict__} for h in h]
    return out, alts, hints


def cut_of(p, alts: dict) -> dict:
    return {op: [u[0] for u in alts[op][label]] for op, label in p.choice.items()}


def at_budget(b: float, plans, cands, costs: CostModel, objective: Objective, w: dict, v: dict, extra: float):
    """The least-ε plan whose B, the commitment floor and the zero-work templates included, is at most `b`."""
    return for_budget(plans, (b - v["commit_floor"]) * w["window_gpu_s"] - extra, candidates=cands, costs=costs,
                      objective=objective, exact=False)


def summarize(p, w: dict, v: dict, extra: float, alts: dict) -> dict:
    gpu_s = w["window_gpu_s"]
    cut = cut_of(p, alts)
    return {"B": (p.cost["total"] + extra) / gpu_s + v["commit_floor"], "eps": p.certificate.harm / w["work"],
            "cost": {k: c / gpu_s for k, c in p.cost.items()}, "cut": cut,
            "draws": {t: k for t, k in p.sizes.items()}, "statement": dict(p.statement_size),
            "floor_only": sorted(p.floor_only), "full": sorted(p.full)}


def zero_work_cost(costs: CostModel, v: dict) -> float:
    """The zero-work templates at their floor: one unit each in a one-unit statement of a small template."""
    small = A6_ANDS[2048]
    return v["zero_work_templates"] * (costs.statement_fixed + costs.statement_per_and * small
                                       + costs.staged_per_and * small + costs.per_and * small)


def run(v: dict, window: str = "week", *, commit: float = 1.0, statement: float = 1.0, grid: int = 61,
        cut_grid: int = 31):
    fit_ = fitted(v)
    w = workload(v, WINDOWS[window])
    costs = model(v, fit_, w["replay_units"], commit=commit, statement=statement)
    objective = Objective(DELTA)
    cands, alts, hints = candidates(w, costs, objective, cut_grid)
    plans = frontier(cands, costs, objective, grid=grid, exact=False)
    return fit_, w, costs, objective, cands, alts, hints, plans


def _as_candidate(op: str, s: Shape, n: int, replay_units: int) -> Candidate:
    return Candidate(op, f"{op}:{s.id}", n, s.work, s.out_bits, s.out_gates, "", s.influence, s.ands, s.rows,
                     s.commit_bits, s.commit_rows, s.cap, "layer-step", n // replay_units)


def forced(w: dict, grain: int | None) -> list[Candidate]:
    """Every op whole, the linears cut at `grain` (K-chunks of that length; None, whole coordinates)."""
    R, out = w["replay_units"], []
    for op, (root, per) in w["ops"].items():
        n = per * R
        if op.startswith("linear"):
            coord = root
            while coord.parts:
                coord, k = coord.parts[0][0], coord.parts[0][1]
                n *= k
            ways = dict(coord.ways)
            if grain is not None and f"K={grain}" in ways:
                (chunk, k), = ways[f"K={grain}"]
                out.append(_as_candidate(op, chunk, n * k, R))
                continue
            out.append(_as_candidate(op, coord, n, R))
        else:
            out.append(_as_candidate(op, root, n, R))
    return out


def breakeven(v: dict) -> list[dict]:
    """Per regime, window and ε: B with the linears forced to each grain, and the cheapest grain."""
    fit_, objective, rows = fitted(v), Objective(DELTA), []
    for regime, commit, statement in REGIMES:
        for window, rounds in WINDOWS.items():
            w = workload(v, rounds)
            costs = model(v, fit_, w["replay_units"], commit=commit, statement=statement)
            extra = zero_work_cost(costs, v)
            for target in TARGETS:
                by = {}
                for grain in (None,) + CHUNKS:
                    cands = forced(w, grain)
                    p = for_target(cands, costs, objective, target * w["work"], grid=31, exact=False)
                    by["coord" if grain is None else str(grain)] = None if p is None else \
                        (p.cost["total"] + extra) / w["window_gpu_s"] + v["commit_floor"]
                ok = {g: b for g, b in by.items() if b is not None}
                rows.append({"regime": regime, "window": window, "eps": target, "B": by,
                             "best": min(ok, key=ok.get) if ok else None})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = {k: i.value for k, i in I.items()}

    fit_, w, costs, objective, cands, alts, hints, plans = run(v, grid=args.grid)
    (args.out / "cost-model.json").write_text(json.dumps(fit_.record(observations()), indent=1))
    extra = zero_work_cost(costs, v)
    front = [summarize(p, w, v, extra, alts) for p in plans]
    for h in hints:
        h["B"] = front[min(range(len(plans)), key=lambda i: abs(log(plans[i].lam / h["lam"])))]["B"]
    budgets = {}
    for b in BUDGETS:
        p = at_budget(b, plans, cands, costs, objective, w, v, extra)
        budgets[b] = None if p is None else summarize(p, w, v, extra, alts)

    table = breakeven(v)
    base_plan = at_budget(OPERATING, plans, cands, costs, objective, w, v, extra)
    hints_at = [] if base_plan is None else [
        {"op": op, **h.__dict__} for op, (root, per) in w["ops"].items()
        for h in pressure(op, root, per * w["replay_units"], costs, objective, base_plan.lam,
                          replay=("layer-step", w["replay_units"]))]
    benchmark = [] if base_plan is None else needs_benchmark(
        cands, costs, objective, (OPERATING - v["commit_floor"]) * w["window_gpu_s"] - extra, grid=31, exact=False)

    sens = []
    base_cut = None if base_plan is None else cut_of(base_plan, alts)
    for name, factors in (("prove", (0.5, 2.0)), ("statement", (0.01, 0.5, 2.0)), ("commit", (0.5, 2.0)),
                          ("replay", (0.5, 4.0))):
        for f in factors:
            pc = perturb(costs, name, f, cands)
            pl = frontier(cands, pc, objective, grid=args.grid, exact=False)
            p = at_budget(OPERATING, pl, cands, pc, objective, w, v, extra)
            sens.append({"what": name, "factor": f, "eps": None if p is None else p.certificate.harm / w["work"],
                         "same_cut": p is not None and cut_of(p, alts) == base_cut,
                         "same_structure": p is not None and base_plan is not None
                         and structure(p) == structure(base_plan)})
    for name in ("tokens_per_s", "replay_layer_step", "commit_per_byte", "zk_unit", "cpu", "step_tokens"):
        for end in ("low", "high"):
            vv = {**v, name: getattr(I[name], end)}
            _, ww, cc, oo, cs, aa, _, pl = run(vv, grid=31, cut_grid=21)
            p = at_budget(OPERATING, pl, cs, cc, oo, ww, vv, zero_work_cost(cc, vv))
            sens.append({"what": name, "value": getattr(I[name], end),
                         "eps": None if p is None else p.certificate.harm / ww["work"],
                         "same_cut": p is not None and cut_of(p, aa) == base_cut})

    out = {"inputs": {k: i.__dict__ for k, i in I.items()}, "delta": DELTA, "window": "week",
           "tokens": w["tokens"], "work": w["work"], "cost_model": fit_.record([])["fits"],
           "rates": {"statement_fixed": costs.statement_fixed, "statement_per_and": costs.statement_per_and,
                     "staged_per_and": costs.staged_per_and, "per_and": costs.per_and,
                     "commit_per_bit": costs.commit_per_bit},
           "alternatives": alts, "hints": hints, "operating": OPERATING, "hints_at_operating": hints_at,
           "needs_benchmark": benchmark, "frontier": front,
           "at_budget": {str(b): f for b, f in budgets.items()}, "breakeven": table, "sensitivity": sens}
    (args.out / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    print(render(out))


def render(out: dict) -> str:
    lines = ["# int8 Llama-3.1-8B under C-Flock: the sampling plan", ""]
    r = out["rates"]
    lines.append(f"rates (GPU-s): a0 {r['statement_fixed']:.3g} + {r['statement_per_and']:.3g}/AND a statement, "
                 f"{r['staged_per_and']:.3g}/AND a staged unit, {r['per_and']:.3g}/AND a proved unit, "
                 f"commit {r['commit_per_bit']:.3g}/bit")
    lines += ["", "| B budget | ε at 2^-40 | draws a week | statements | units | commit | replay | linear cut | "
              "statement sizes |", "|---|---|---|---|---|---|---|---|---|"]
    for b, f in out["at_budget"].items():
        if f is None:
            lines.append(f"| {b} | (none affordable) | | | | | | | |")
            continue
        c = f["cost"]
        cut = ", ".join(f"{op.split()[-1]}: {'+'.join(u)}" for op, u in f["cut"].items() if op.startswith("linear"))
        sizes = sorted(set(f["statement"].values()))
        lines.append(f"| {b} | {f['eps']:.4%} | {sum(f['draws'].values()):,} | {c['statements']:.4f} | "
                     f"{c['units']:.2e} | {c['commit']:.2e} | {c['replay']:.2e} | {cut} | {sizes} |")
    grains = ["coord"] + [str(c) for c in CHUNKS]
    lines += ["", "B with the linears at each grain (K-chunk length), the cheapest marked:", "",
              "| regime | window | ε | " + " | ".join(grains) + " |", "|---|---|---|" + "---|" * len(grains)]
    for e in out["breakeven"]:
        cells = ["-" if e["B"][g] is None else (f"**{e['B'][g]:.4g}**" if g == e["best"] else f"{e['B'][g]:.4g}")
                 for g in grains]
        lines.append(f"| {e['regime']} | {e['window']} | {e['eps']:.2%} | " + " | ".join(cells) + " |")
    lines += ["", f"| input | value | ε at B = {out['operating']} | same cut |", "|---|---|---|---|"]
    for s in out["sensitivity"]:
        val = s.get("factor", s.get("value"))
        lines.append(f"| {s['what']} | {val:.3g} | {'-' if s['eps'] is None else format(s['eps'], '.4%')} | "
                     f"{s.get('same_cut', '')} |")
    best = {}
    for h in out["hints"]:
        key = (h["op"], h["kind"], h["node"])
        if key not in best or h["saving"] > best[key]["saving"]:
            best[key] = h
    lines += ["", "hints over the sweep (per node its largest saving, at the frontier's B for that density):"] + [
        f"- {h['op']}: {h['kind']} {h['node']} -> {h['child']} split {h['split']} saves {h['saving']:.1%} at B "
        f"{h['B']:.3g}" for h in best.values()]
    at = out["hints_at_operating"]
    lines += ["", f"hints at B = {out['operating']}:"] + [
        f"- {h['op']}: {h['kind']} {h['node']} {h['child']} split {h['split']} saves {h['saving']:.1%}" for h in at] \
        + ["- none"] * (not at)
    lines += ["", f"templates worth benchmarking at B = {out['operating']}: "
              + (", ".join(out["needs_benchmark"]) or "none")]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
