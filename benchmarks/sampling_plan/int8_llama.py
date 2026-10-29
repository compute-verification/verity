"""The sampling plan for an int8 row under Flock (`verity.core.protocols.sampled_proofs`), Llama-3.1-8B or Qwen3-8B: the
fitted cost model, the cuts the generator finds, the frontier of ε against the verification overhead B, the break-even
granularity by window, ε and commitment regime, the frontier's sensitivity to its inputs, the tensors committed at more
than one scale, and the weight classes' weights at each target.

Units: every s8 dot unit checks its seed and derives its own noise (2 + 2 ceil(L b / 512) SHA-512
compressions), opens its operand rows and its output in circuit, and carries its own work (MACs, W_ref's proxy).  A dot
may be cut into K-chunks (`S8Dot{T0 = input}`), each committing its running sum, and tiles of r x c dots are offered,
each a unit when its outputs fit a size class.  B is GPU-seconds of overhead per plain GPU-second over a window (the budget model's unit),
the total: everything is committed at serve time, so B holds every scale's serve-time commitments under the PoUW
scheme (`inputs.serve_rows`: the Commit's taps, PoUW's work blocks, the bits they pin, the remote audit's receipts) beside
the proving, the units' commitments, the witnesses' replay and the zero-work subcircuits.  The `floor` scheme is the earlier
accounting, its one measured term (`commit_floor`) and nothing else.

    uv run python benchmarks/sampling_plan/int8_llama.py --out DIR [--model qwen3-8b] [--scheme pearl-c] [--weights]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from math import ceil, log, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import (EPILOGUE, I, KEY, MODELS, S8_STEP, SCHEMES, SHA512, Model, leaf_chain, observations,  # noqa: E402
                    s8_coord_ands, serve_rows, tree_chain)

from verity.core.protocols.sampled_proofs.candidates import Shape, cuts, pressure  # noqa: E402
from verity.core.protocols.sampled_proofs.costfit import Fitted, fit  # noqa: E402
from verity.core.protocols.sampled_proofs.optimize import (Candidate, Commitment, CostModel, Objective,  # noqa: E402
                                                           ReplayDomain, for_budget, for_target, frontier,
                                                           needs_benchmark, overlaps, perturb, row_ands, structure,
                                                           weigh)

DELTA = 2.0 ** -40
CHUNKS = (2048, 1024, 512, 256, 128, 64, 32)
WINDOWS = {"round": 1, "hour": 3600 / 12.6, "day": 86400 / 12.6, "week": 48_000}
TARGETS = (1e-3, 1e-4)
REGIMES = (("plain", 1.0, 1.0), ("prefix", 0.0, 1.0), ("prefix, sessions/256", 0.0, 1 / 256))
BUDGETS = (0.065, 0.07, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 5.0)
#: the sampling spend at the `floor` scheme's operating point, B = 0.1 less its 0.060 floor: the operating B is the serve-time total plus it
SPEND = 0.04
#: a layer's linear calls (qkv, o_proj, gate_up, down), each hashing its tile leaves as it ends
CALLS = 4


def fitted(v: dict) -> Fitted:
    obs = observations()
    return Fitted(fit(obs, "prove", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "verify", ["one", "ands", "ands*staged", "ands*drawn"]),
                  fit(obs, "stage", ["one", "ands", "ands*staged"]), v["cpu"], KEY)


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
    K-chunks that carry the running sum; the root is a 16 x 16 tile over 2 x 1 tiles."""
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


def workload(v: dict, window_rounds: float, m: Model) -> dict:
    """Per op: its root shape, its instances over the window, the replay domain's units, and the window's work."""
    bits = int(v["noise_bits"])
    tokens = v["tokens_per_s"] * v["round"] * window_rounds
    step = int(v["step_tokens"])
    steps = max(1, round(tokens / step))
    replay_units = steps * m.layers
    ops = {}
    for k, coords in m.linears.items():
        root = linear(k, bits)
        ops[f"linear K={k}"] = (root, step * coords * k // root.work)
    shares = classes(int(v["context"]))
    mean_pos = (int(v["context"]) + 1) / 2
    ops["attention S"] = (s8_unit(f"S{{K={m.head_dim}}}", m.head_dim, 32, extra_rows=(4,), commit_bits=0, bits=bits),
                          round(step * m.heads * mean_pos))
    for length, share in shares.items():
        per_step = round(step * share) * m.heads * m.head_dim
        if per_step:
            ops[f"attention PV L={length}"] = (pv(length, bits), per_step)
    total_work = sum(root.work * per for root, per in ops.values()) * replay_units
    return {"ops": ops, "replay_units": replay_units, "tokens": tokens, "work": total_work,
            "window_gpu_s": v["round"] * window_rounds}


def serve(v: dict, m: Model, scheme: str | None, w: dict, *, kv: bool = False) -> tuple[Commitment, ...]:
    """The window's serve-time commitments under `scheme` (None: none, a plan's sampling spend alone), each priced at
    `commit_per_byte` (`floor`: its measured B), with its decode bytes and gating chain at a B32 step: one tree a call over
    its leaves, or a tile leaf a call."""
    if scheme is None:
        return ()
    if scheme == "floor":
        a = (m.hidden + m.heads * m.head_dim + m.hidden + m.intermediate) * w["tokens"] * m.layers
        return (Commitment("floor", "s8 A operands, R = 128 leaves (measured)", a,
                           cost=v["commit_floor"] * w["window_gpu_s"]),)
    rate, step = v["commit_per_byte"], int(v["decode_tokens"])
    out = []
    for r in serve_rows(m, scheme, kv=kv):
        per_call = (r.decode or r.nbytes) * step
        chain = CALLS * leaf_chain(r.leaf, r.fmt) if r.tile else leaf_chain(r.leaf, r.fmt) + tree_chain(per_call / r.leaf)
        nbytes = r.nbytes * w["tokens"] * m.layers
        out.append(Commitment(r.scale, r.tensor, nbytes, cost=nbytes * rate, decode_bytes=per_call * m.layers,
                              chain=chain * m.layers, gates=r.gates))
    receipts = v["remote_receipts"] * v["passes_per_s"] * w["window_gpu_s"]
    out.append(Commitment("remote", "store-block receipts", receipts, cost=receipts * rate))
    return tuple(out)


def model(v: dict, fit_: Fitted, replay_units: int, *, commitments: tuple[Commitment, ...] = (), commit: float = 1.0,
          statement: float = 1.0) -> CostModel:
    m = fit_.model()
    m = replace(m, commit_per_bit=v["commit_per_byte"] / 8 * commit, serve=commitments,
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


def at_budget(b: float, plans, cands, costs: CostModel, objective: Objective, w: dict, extra: float):
    """The least-ε plan whose B, the serve-time commitments and the zero-work subcircuits included, is at most `b`."""
    return for_budget(plans, b * w["window_gpu_s"] - extra, candidates=cands, costs=costs, objective=objective,
                      exact=False)


def summarize(p, w: dict, extra: float, alts: dict) -> dict:
    gpu_s = w["window_gpu_s"]
    cut = cut_of(p, alts)
    return {"B": (p.cost["total"] + extra) / gpu_s, "eps": p.certificate.harm / w["work"],
            "cost": {k: c / gpu_s for k, c in p.cost.items()}, "cut": cut,
            "draws": {t: k for t, k in p.sizes.items()}, "statement": dict(p.statement_size),
            "floor_only": sorted(p.floor_only), "full": sorted(p.full), "weights": dict(p.weights)}


def zero_work_cost(costs: CostModel, v: dict) -> float:
    """The zero-work subcircuits at their floor: one unit each in a one-unit statement of a small subcircuit."""
    small = s8_coord_ands(2048)
    return v["zero_work_templates"] * (costs.statement_fixed + costs.statement_per_and * small
                                       + costs.staged_per_and * small + costs.per_and * small)


def run(v: dict, window: str = "week", *, model_name: str = "llama-3.1-8b", scheme: str | None = "h1",
        kv: bool = False, commit: float = 1.0, statement: float = 1.0, grid: int = 61, cut_grid: int = 31):
    m = MODELS[model_name]
    fit_ = fitted(v)
    w = workload(v, WINDOWS[window], m)
    costs = model(v, fit_, w["replay_units"], commitments=serve(v, m, scheme, w, kv=kv), commit=commit,
                  statement=statement)
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


def breakeven(v: dict, model_name: str, scheme: str | None, *, kv: bool = False) -> list[dict]:
    """Per regime, window and ε: B with the linears forced to each grain, and the cheapest grain."""
    m = MODELS[model_name]
    fit_, objective, rows = fitted(v), Objective(DELTA), []
    for regime, commit, statement in REGIMES:
        for window, rounds in WINDOWS.items():
            w = workload(v, rounds, m)
            costs = model(v, fit_, w["replay_units"], commitments=serve(v, m, scheme, w, kv=kv), commit=commit,
                          statement=statement)
            extra = zero_work_cost(costs, v)
            for target in TARGETS:
                by = {}
                for grain in (None,) + CHUNKS:
                    cands = forced(w, grain)
                    p = for_target(cands, costs, objective, target * w["work"], grid=31, exact=False)
                    by["coord" if grain is None else str(grain)] = None if p is None else \
                        (p.cost["total"] + extra) / w["window_gpu_s"]
                ok = {g: b for g, b in by.items() if b is not None}
                rows.append({"regime": regime, "window": window, "eps": target, "B": by,
                             "best": min(ok, key=ok.get) if ok else None})
    return rows


def serve_table(costs: CostModel, w: dict, m: Model) -> list[dict]:
    per = w["tokens"] * m.layers
    return [{"scale": s.scale, "tensor": s.tensor, "B": costs.committed(s) / w["window_gpu_s"],
             "bytes_token_layer": s.nbytes / per if s.scale != "remote" else None, "gates": s.gates}
            for s in costs.serve]


def overlap_table(v: dict, m: Model, scheme: str, w: dict) -> list[dict]:
    """`overlaps` on the serve-time commitments with the KV append counted (it is one copy only when KV is credited), in
    B at prefill, bytes a token a layer, and decode's chain and seconds a step."""
    costs = CostModel(commit_per_bit=v["commit_per_byte"] / 8, serve=serve(v, m, scheme, w, kv=True))
    per, gpu_s = w["tokens"] * m.layers, w["window_gpu_s"]
    out = []
    for r in overlaps(costs, latency=v["compress_latency"]):
        out.append({"tensor": r["tensor"], "kept": r["kept"],
                    "copies": [{**c, "bytes": c["bytes"] / per, "prefill": c["prefill"] / gpu_s} for c in r["copies"]],
                    "saving": {**r["saving"], "bytes": r["saving"]["bytes"] / per,
                               "prefill": r["saving"]["prefill"] / gpu_s}})
    return out


def weights_at(cands, costs: CostModel, objective: Objective, w: dict, extra: float, alts: dict) -> list[dict]:
    """At each ε target, the plan at equal weights and the one `weigh` chooses, each class (subcircuit) weighed alone."""
    out = []
    for target in TARGETS:
        flat = for_target(cands, costs, objective, target * w["work"], grid=31, exact=False)
        best = weigh(cands, costs, objective, target * w["work"], grid=31, exact=False)
        out.append({"eps": target, "equal": None if flat is None else summarize(flat, w, extra, alts),
                    "weighed": None if best is None else summarize(best, w, extra, alts)})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=sorted(MODELS), default="llama-3.1-8b")
    ap.add_argument("--scheme", choices=SCHEMES, default="h1", help="the PoUW scheme whose serve-time commitments B holds")
    ap.add_argument("--kv", action="store_true", help="count the remote KV append (live KV credited) in B")
    ap.add_argument("--weights", action="store_true", help="choose the weight classes' weights at each ε target")
    ap.add_argument("--grid", type=int, default=61)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = {k: i.value for k, i in I.items()}
    m = MODELS[args.model]

    def run_(vv: dict, **kw):
        return run(vv, model_name=args.model, scheme=args.scheme, kv=args.kv, **kw)

    fit_, w, costs, objective, cands, alts, hints, plans = run_(v, grid=args.grid)
    (args.out / "cost-model.json").write_text(json.dumps(fit_.record(observations()), indent=1))
    extra = zero_work_cost(costs, v)
    serve_b = costs.serve_cost() / w["window_gpu_s"]
    operating = serve_b + SPEND
    front = [summarize(p, w, extra, alts) for p in plans]
    for h in hints:
        h["B"] = front[min(range(len(plans)), key=lambda i: abs(log(plans[i].lam / h["lam"])))]["B"]
    budgets = {}
    for b in BUDGETS:
        p = at_budget(b, plans, cands, costs, objective, w, extra)
        budgets[b] = None if p is None else summarize(p, w, extra, alts)

    table = breakeven(v, args.model, args.scheme, kv=args.kv)
    base_plan = at_budget(operating, plans, cands, costs, objective, w, extra)
    hints_at = [] if base_plan is None else [
        {"op": op, **h.__dict__} for op, (root, per) in w["ops"].items()
        for h in pressure(op, root, per * w["replay_units"], costs, objective, base_plan.lam,
                          replay=("layer-step", w["replay_units"]))]
    benchmark = [] if base_plan is None else needs_benchmark(
        cands, costs, objective, operating * w["window_gpu_s"] - extra, grid=31, exact=False)

    sens = []
    base_cut = None if base_plan is None else cut_of(base_plan, alts)
    for name, factors in (("prove", (0.5, 2.0)), ("statement", (0.01, 0.5, 2.0)), ("commit", (0.5, 2.0)),
                          ("replay", (0.5, 4.0))):
        for f in factors:
            pc = perturb(costs, name, f, cands)
            pl = frontier(cands, pc, objective, grid=args.grid, exact=False)
            p = at_budget(operating, pl, cands, pc, objective, w, extra)
            sens.append({"what": name, "factor": f, "eps": None if p is None else p.certificate.harm / w["work"],
                         "same_cut": p is not None and cut_of(p, alts) == base_cut,
                         "same_structure": p is not None and base_plan is not None
                         and structure(p) == structure(base_plan)})
    for name in ("tokens_per_s", "replay_layer_step", "commit_per_byte", "cpu", "step_tokens", "passes_per_s"):
        for end in ("low", "high"):
            vv = {**v, name: getattr(I[name], end)}
            _, ww, cc, oo, cs, aa, _, pl = run_(vv, grid=31, cut_grid=21)
            p = at_budget(operating, pl, cs, cc, oo, ww, zero_work_cost(cc, vv))
            sens.append({"what": name, "value": getattr(I[name], end),
                         "eps": None if p is None else p.certificate.harm / ww["work"],
                         "same_cut": p is not None and cut_of(p, aa) == base_cut})

    out = {"model": args.model, "scheme": args.scheme, "kv": args.kv,
           "inputs": {k: i.__dict__ for k, i in I.items()}, "delta": DELTA, "window": "week",
           "tokens": w["tokens"], "work": w["work"], "cost_model": fit_.record([])["fits"],
           "rates": {"statement_fixed": costs.statement_fixed, "statement_per_and": costs.statement_per_and,
                     "staged_per_and": costs.staged_per_and, "per_and": costs.per_and,
                     "commit_per_bit": costs.commit_per_bit},
           "serve_B": serve_b, "serve": serve_table(costs, w, m),
           "overlaps": [] if args.scheme == "floor" else overlap_table(v, m, args.scheme, w),
           "alternatives": alts, "hints": hints, "operating": operating, "hints_at_operating": hints_at,
           "needs_benchmark": benchmark, "frontier": front,
           "at_budget": {str(b): f for b, f in budgets.items()}, "breakeven": table, "sensitivity": sens,
           "weights": weights_at(cands, costs, objective, w, extra, alts) if args.weights else None}
    (args.out / "plan.json").write_text(json.dumps(out, indent=1, default=str))
    print(render(out))


def _ms(seconds: float) -> str:
    return f"{seconds * 1e3:.2f}"


def render(out: dict) -> str:
    lines = [f"# int8 {out['model']} under C-Flock, PoUW scheme {out['scheme']}"
             + (", KV credited" if out["kv"] else "") + ": the sampling plan", ""]
    r = out["rates"]
    lines.append(f"rates (GPU-s): a0 {r['statement_fixed']:.3g} + {r['statement_per_and']:.3g}/AND a statement, "
                 f"{r['staged_per_and']:.3g}/AND a staged unit, {r['per_and']:.3g}/AND a proved unit, "
                 f"commit {r['commit_per_bit']:.3g}/bit")
    lines += ["", f"serve-time commitments: B {out['serve_B']:.4f} (every plan pays them)", "",
              "| scale | tensor | bytes a token a layer | B | gates a decode step |", "|---|---|---|---|---|"]
    for s in out["serve"]:
        size = "-" if s["bytes_token_layer"] is None else f"{s['bytes_token_layer']:,.0f}"
        lines.append(f"| {s['scale']} | {s['tensor']} | {size} | {s['B']:.4f} | {'yes' if s['gates'] else ''} |")
    lines += ["", "| B budget | ε at 2^-40 | draws a week | statements | units | commit | replay | serve | linear cut | "
              "statement sizes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b, f in out["at_budget"].items():
        if f is None:
            lines.append(f"| {b} | (none affordable) | | | | | | | | |")
            continue
        c = f["cost"]
        cut = ", ".join(f"{op.split()[-1]}: {'+'.join(u)}" for op, u in f["cut"].items() if op.startswith("linear"))
        sizes = sorted(set(f["statement"].values()))
        lines.append(f"| {b} | {f['eps']:.4%} | {sum(f['draws'].values()):,} | {c['statements']:.4f} | "
                     f"{c['units']:.2e} | {c['commit']:.2e} | {c['replay']:.2e} | {c['serve']:.4f} | {cut} | {sizes} |")
    if out["overlaps"]:
        lines += ["", "Tensors committed at more than one scale, ranked by what committing once saves at prefill (the "
                  "KV append counted: it is a copy only when KV is credited). Bytes a token a layer; prefill in B; "
                  "decode a B32 step: the gating chain in compressions and its latency in ms.", "",
                  "| tensor | copies (scale: bytes, prefill B, decode chain, ms if it gates) | kept | saving: bytes | "
                  "prefill B | decode ms |", "|---|---|---|---|---|---|"]
        for o in out["overlaps"]:
            copies = "; ".join(f"{c['scale']}: {c['bytes']:,.0f}, {c['prefill']:.4f}, {c['chain']:,.0f}"
                               + (f", {_ms(c['decode'])}" if c["gates"] else ", trails") for c in o["copies"])
            s = o["saving"]
            lines.append(f"| {o['tensor']} | {copies} | {o['kept']} | {s['bytes']:,.0f} | {s['prefill']:.4f} | "
                         f"{_ms(s['decode'])} |")
    if out["weights"]:
        lines += ["", "Block kinds' weights at each target (each template its own kind):", "",
                  "| ε | B at equal weights | B weighed | weights |", "|---|---|---|---|"]
        for e in out["weights"]:
            eq, wd = e["equal"], e["weighed"]
            lines.append(f"| {e['eps']:.2%} | {'-' if eq is None else format(eq['B'], '.5g')} | "
                         f"{'-' if wd is None else format(wd['B'], '.5g')} | "
                         f"{'-' if wd is None else (wd['weights'] or 'all 1')} |")
    grains = ["coord"] + [str(c) for c in CHUNKS]
    lines += ["", "B with the linears at each grain (K-chunk length), the cheapest marked:", "",
              "| regime | window | ε | " + " | ".join(grains) + " |", "|---|---|---|" + "---|" * len(grains)]
    for e in out["breakeven"]:
        cells = ["-" if e["B"][g] is None else (f"**{e['B'][g]:.4g}**" if g == e["best"] else f"{e['B'][g]:.4g}")
                 for g in grains]
        lines.append(f"| {e['regime']} | {e['window']} | {e['eps']:.2%} | " + " | ".join(cells) + " |")
    lines += ["", f"| input | value | ε at B = {out['operating']:.4g} | same cut |", "|---|---|---|---|"]
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
    lines += ["", f"hints at B = {out['operating']:.4g}:"] + [
        f"- {h['op']}: {h['kind']} {h['node']} {h['child']} split {h['split']} saves {h['saving']:.1%}" for h in at] \
        + ["- none"] * (not at)
    lines += ["", f"templates worth benchmarking at B = {out['operating']:.4g}: "
              + (", ".join(out["needs_benchmark"]) or "none")]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
