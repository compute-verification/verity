"""OPT5: the least serve-time and proving total that keeps every policy within its bound, for the int8 rows (Llama-3.1-8B,
Qwen3-8B) under C-Flock: note:proofs/20261018T2305Z-draft-iop-core-objects's tier 2, d_P <= A_P H_c(delta), with the
draw weighted by the envelope of the policies' public costs (`optimize.Policy`).  Each policy is held to a share `eps` of
its total over the window:

- skippable work: c(u) = wt(u), A = F / (1 - p), F = 1 (every operand is committed at serve time and each dot unit
  derives its own noise), p the luck (the spec sheet's 7/8; p = 0 is OPT4's ε); of the window's credited work;
- control: c(u) = w(u) + log2(|U| + 1), the location term absorbed (the 23:02Z schema), A = 1; of the egress bits, the
  outputs at ceil(log2 vocab) bits;
- integrity: c(u) = the outputs in u's cone, a request being `context` prompt tokens and `decode` generated ones (a
  prompt unit reaches every output, a decode unit the ones after it), A = 1; of the outputs.

The decision variables: the cut (the unit's size), the replay block's tokens, the draw (the envelope), the always-proved
set (templates proved whole), clustered draws (k units checked in each drawn replay block, in closed form), and the
serve-time options (`OPTIONS`), each priced in B.  No plan depends on the serve-time set, which every plan pays, so each
plan is found once and each option set's B added.  The zero-work templates stay at their floor (`zero_work_cost`), so a
control or integrity row is a lower bound.

    uv run python benchmarks/sampling_plan/opt5.py --out DIR --label main
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from itertools import product
from math import ceil, exp, inf, lgamma, log, log1p, log2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inputs import ACT, ATTN, I, MODELS, Input, Model, serve_rows  # noqa: E402
from int8_llama import (DELTA, WINDOWS, candidates, cut_of, fitted, model, serve, workload,  # noqa: E402
                        zero_work_cost)

from verity.core.protocols.sampled_proofs.optimize import (Objective, Policy, certify, for_target,  # noqa: E402
                                                           replay_cost, template_cost)

P_INPUTS = {
    "decode": Input(256, 32, 2048, "generated tokens a request, beside `context` prompt tokens", "A"),
    "luck": Input(7 / 8, 0.0, 7 / 8, "p, the chance a skipped unit's output is right anyway (the spec sheet's page 3: "
                  "'p = 7/8 gives x8'); 0 is OPT4's ε", "A"),
    "late_cone": Input(1, 1, 1, "F: each dot unit derives its own noise (decision 106) and every operand is committed at "
                       "serve time (ASK-25), so a unit's late cone is itself", "D"),
    "weights_s": Input(1.21e-4, 1.21e-4, 1.21e-4, "a replay block's weight read, 218 MB at 1.8 TB/s, its fixed part "
                       "(I['replay_layer_step'] less 256 tokens at the plain rate)", "D"),
}
VOCAB = {"llama-3.1-8b": 128_256, "qwen3-8b": 151_936}
EPS = (1e-3, 1e-4)
STEPS = (32, 256, 2048)
SCHEMES = ("h1", "pearl-c")
#: the share of the Commit's tap hashing that hash's rule-keeping changes save (H200: 2.50 of 11.92 points of serve, and
#: 31-32% with q and k committed once around RoPE)
FORMAT, QK_ROPE = 2.50 / 11.92, 0.315


@dataclass(frozen=True)
class Option:
    name: str
    schemes: tuple[str, ...]
    owner: str
    what: str
    excludes: tuple[str, ...] = ()


OPTIONS = (
    Option("commit-once", ("pearl-c",), "compute-accounting (Pearl-C's A-leaf format)",
           "Pearl-C's seed reads down's and o_proj's A from the Commit's BF16 rows: their FP32 A rows go, and the "
           "Commit's copies gate decode in their place; the PoUW proof opens BF16 rows widened in circuit"),
    Option("format", ("h1", "h1-bf16", "pearl-c"), "hash", "the Commit's taps in hash's rule-keeping format",
           ("qk-rope",)),
    Option("qk-rope", ("pearl-c",), "hash", "the format, and q and k committed once around RoPE; H1 pins rope_q as "
           "QK^T's A, so not under h1", ("format",)),
)


def option_sets(scheme: str) -> list[tuple[str, ...]]:
    mine = [o for o in OPTIONS if scheme in o.schemes]
    out = []
    for picks in product((False, True), repeat=len(mine)):
        chosen = tuple(o.name for o, on in zip(mine, picks) if on)
        if not any(x in chosen for o in mine if o.name in chosen for x in o.excludes):
            out.append(chosen)
    return out


def serve_set(v: dict, m: Model, scheme: str, w: dict, chosen: tuple[str, ...]):
    rows = serve_rows(m, scheme)
    if "commit-once" in chosen:
        rows = [replace(r, gates=True) if r.scale == "commit" and r.tensor in (ATTN, ACT) else r for r in rows
                if not (r.scale == "pearl-c:A" and r.tensor in (ATTN, ACT))]
    out = serve(v, m, scheme, w, rows=rows)
    keep = 1 - (FORMAT if "format" in chosen else QK_ROPE if "qk-rope" in chosen else 0.0)
    return tuple(replace(s, cost=s.cost * keep) if s.scale == "commit" and s.tensor != "CP-W receipts" else s
                 for s in out)


def serve_table(v: dict, m: Model, w: dict) -> list[dict]:
    out = []
    for scheme in SCHEMES:
        for chosen in option_sets(scheme):
            s = serve_set(v, m, scheme, w, chosen)
            out.append({"scheme": scheme, "options": list(chosen), "B": sum(c.cost for c in s) / w["window_gpu_s"],
                        "decode_ms": 1e3 * v["compress_latency"] * sum(c.chain for c in s if c.gates)})
    return out


def totals(v: dict, model_name: str, w: dict) -> dict:
    context, decode = int(v["context"]), int(v["decode"])
    outputs = w["tokens"] / (context + decode) * decode
    return {"work": w["work"], "outputs": outputs, "egress": outputs * ceil(log2(VOCAB[model_name])),
            "cone": (context * decode + decode * (decode + 1) / 2) / (context + decode),
            "location": log2(w["work"] / 32 + 1)}


def policies(v: dict, t: dict, eps: float, which: str, luck: float) -> tuple[Policy, ...]:
    """`which` is a string of W, C and I; `eps` 1 gives each policy's total as its target (a level, not a bound)."""
    out = {"W": Policy("work", eps * t["work"], amplification=v["late_cone"] / (1 - luck), work=1.0),
           "C": Policy("control", eps * t["egress"], bits=1.0, unit=t["location"]),
           "I": Policy("integrity", eps * t["outputs"], unit=t["cone"])}
    return tuple(out[k] for k in which)


TOTAL = {"work": "work", "control": "egress", "integrity": "outputs"}


def plan_row(v: dict, model_name: str, which: str, eps: float, *, luck: float, step: int | None = None, grid: int = 31,
             cut_grid: int = 21) -> dict:
    vv = dict(v)
    if step is not None:
        vv["step_tokens"] = step
        vv["replay_layer_step"] = v["weights_s"] + step * (v["replay_layer_step"] - v["weights_s"]) / 256
    fit_ = fitted(vv)
    w = workload(vv, WINDOWS["week"], MODELS[model_name])
    costs = model(vv, fit_, w["replay_units"])
    t = totals(vv, model_name, w)
    objective = Objective(DELTA, policies=policies(vv, t, eps, which, luck))
    cands, alts, _ = candidates(w, costs, objective, cut_grid)
    p = for_target(cands, costs, objective, 1.0, grid=grid, exact=False)
    extra = zero_work_cost(costs, vv)
    gpu_s = w["window_gpu_s"]
    chosen = [c for c in cands if c.template in p.sizes]
    levels = certify(p.sizes, cands, Objective(DELTA, policies=policies(vv, t, 1.0, "WCI", luck)), exact=False)
    cert = p.certificate.policies
    binding = max(objective.policies, key=lambda q: cert[q.name] / q.target).name
    drawn_by = {}
    for c in chosen:
        if p.sizes[c.template] > 1:
            drawn_by[objective.binding(c)] = drawn_by.get(objective.binding(c), 0) + p.sizes[c.template]
    full_all = sum(template_cost(c, costs, c.units)[0] + costs.commit(c) for c in chosen) \
        + replay_cost({c.template: c.units for c in chosen}, chosen, costs)
    dominant = max(chosen, key=lambda c: p.sizes[c.template])
    a0, a1 = costs.statement(dominant)
    return {"which": which, "eps": eps, "luck": luck, "step": vv["step_tokens"], "replay_s": vv["replay_layer_step"],
            "B_proving": (p.cost["total"] + extra) / gpu_s, "cost": {k: c / gpu_s for k, c in p.cost.items()},
            "draws": sum(p.sizes.values()), "binding": binding,
            "use": {q.name: cert[q.name] / q.target for q in objective.policies},
            "levels": {q: levels.policies[q] / t[TOTAL[q]] for q in levels.policies},
            "drawn_by": drawn_by, "cut": cut_of(p, alts), "full": sorted(p.full), "floor_only": sorted(p.floor_only),
            "B_full": (full_all + extra) / gpu_s, "totals": t, "gpu_s": gpu_s,
            "replay_units_met": p.cost["replay"] / vv["replay_layer_step"],
            "dominant": {"template": dominant.template, "a0": a0, "a1": a1,
                         "b": costs.unit(dominant) + costs.of(dominant).replay,
                         "per_replay": dominant.per_replay},
            "units_per_replay": sum(c.per_replay for c in chosen if not c.op.startswith("attention PV"))}


def cluster(row: dict, ks=(1, 2, 8, 64)) -> list[dict]:
    """Clustered draws at the row's ε: each drawn replay block checks k units.  With every value committed before the
    first coins, a block holding m wrong units escapes with at most 1 - q m / n_v (the chord of a concave function), so
    the wrong work is at most n_v ln(1/delta) / q: ε = k ln(1/delta) / M at M = q k N_R proofs, and holding ε holds the
    blocks drawn while proving k times the units.  `twoStage_profile`, which charges each faulty block its n_v whenever
    its interior is registered after the first coins, gives n_v ln(1/delta) / M at any k."""
    m, d = row["draws"], row["dominant"]
    out = []
    for k in ks:
        extra = m * (k - 1) * (d["a1"] + d["b"]) / row["gpu_s"]
        out.append({"k": k, "proofs": m * k, "blocks": m, "B_proving": row["B_proving"] + extra,
                    "eps_at_equal_proving": min(1.0, k * row["eps"])})
    return out


def _block_rate(n_v: int, k: int, q: float) -> float:
    """The least -ln(escape) a wrong unit costs in a block drawn with probability q and k of its n_v units checked, its m
    wrong units committed before the first coins: min over m of -ln(1 - q (1 - C(n_v - m, k) / C(n_v, k))) / m, over a
    log-spaced grid of m and both ends."""
    ms = sorted({1, n_v, *(round(n_v ** (i / 400)) for i in range(401))})
    best = inf
    for m in ms:
        miss = 0.0 if m > n_v - k else exp(lgamma(n_v - m + 1) - lgamma(n_v - m - k + 1) - lgamma(n_v + 1)
                                           + lgamma(n_v - k + 1))
        caught = q * (1 - miss)
        escape = 1 - q + q * miss
        best = min(best, (-log1p(-caught) if caught < 0.5 else inf if escape <= 0 else -log(escape)) / m)
    return best


def ts1(n_r: int = 442_368, n_v: int = 35_840, ms=(27_713, 1e6, 1e7, 1e8, 9.93e8)) -> list[dict]:
    """TS1's table (note:proofs/20261018T1640Z-draft-two-stage-sampling) with every value committed at serve time: one
    stage; two stage as `twoStage_profile` charges it (each faulty block its n_v, the interiors registered after the first
    coins), which is n_v ln(1/delta) / M at any k; two stage with every value committed, from the cheapest wrong count a
    block (`_block_rate`), at the better of the least k that draws M proofs (q = M / (k N_R) <= 1) and every block drawn
    with the k it affords (q = 1); and check-all."""
    lnd, n = log(1 / DELTA), n_r * n_v
    out = []
    for m in ms:
        t1 = lnd / -log1p(-m / n)
        up = max(1, ceil(m / n_r))
        ways = [(lnd / _block_rate(n_v, up, m / (up * n_r)) / n, up, m / (up * n_r))]
        if m >= n_r:
            ways.append((lnd / _block_rate(n_v, int(m // n_r), 1.0) / n, int(m // n_r), 1.0))
        eps, k, q = min(ways)
        out.append({"M": m, "one_stage": t1 / n, "deferred": min(1.0, t1 * n_v / n), "k": k, "q": q,
                    "all_committed": min(1.0, eps), "check_all": min(1.0, n_v * lnd / m)})
    return out


def run(v: dict, model_name: str, *, grid: int, cut_grid: int) -> dict:
    m = MODELS[model_name]
    w = workload(v, WINDOWS["week"], m)
    rows = []
    for which, luck in (("W", v["luck"]), ("W", 0.0), ("WC", v["luck"]), ("WCI", v["luck"])):
        for eps in EPS:
            rows.append(plan_row(v, model_name, which, eps, luck=luck, grid=grid, cut_grid=cut_grid))
    steps = [plan_row(v, model_name, "W", eps, luck=v["luck"], step=s, grid=grid, cut_grid=cut_grid)
             for s in STEPS if s != int(v["step_tokens"]) for eps in EPS]
    serve_ = serve_table(v, m, w)
    best = {s: min((r for r in serve_ if r["scheme"] == s), key=lambda r: r["B"]) for s in SCHEMES}
    base = {s: next(r for r in serve_ if r["scheme"] == s and not r["options"]) for s in SCHEMES}
    optimum = []
    for r in rows + steps:
        for s in SCHEMES:
            optimum.append({"scheme": s, "options": best[s]["options"], "which": r["which"], "eps": r["eps"],
                            "luck": r["luck"], "step": r["step"], "B": best[s]["B"] + r["B_proving"],
                            "B_no_options": base[s]["B"] + r["B_proving"], "binding": r["binding"]})
    return {"model": model_name, "rows": rows, "steps": steps, "serve": serve_, "optimum": optimum,
            "cluster": {f"{r['eps']:g}": cluster(r) for r in rows if r["which"] == "W" and r["luck"] == v["luck"]},
            "ts1": ts1()}


def render(out: dict) -> str:
    lines = [f"# OPT5, {out['model']} ({out['label']})", "", "Serve-time sets (B at prefill, decode's gating chain a B32 "
             "step):", "", "| scheme | options | serve B | decode ms |", "|---|---|---|---|"]
    for s in out["serve"]:
        lines.append(f"| {s['scheme']} | {', '.join(s['options']) or 'none'} | {s['B']:.4f} | {s['decode_ms']:.2f} |")
    lines += ["", "Plans (proving B: statements, units, the units' commitments, replay and the zero-work floor):", "",
              "| policies | ε | p | step | proving B | draws | binding | use (damage / target) | levels at this plan "
              "(work, control, integrity) | linear cut | whole |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out["rows"] + out["steps"]:
        use = ", ".join(f"{k} {u:.2f}" for k, u in r["use"].items())
        lev = ", ".join(f"{r['levels'][k]:.3g}" for k in ("work", "control", "integrity"))
        cut = "; ".join(f"{op.split()[-1]}: {'+'.join(u)}" for op, u in r["cut"].items() if op.startswith("linear"))
        lines.append(f"| {r['which']} | {r['eps']:g} | {r['luck']:.3g} | {r['step']} | {r['B_proving']:.4g} | "
                     f"{r['draws']:.3g} | {r['binding']} | {use} | {lev} | {cut} | {len(r['full'])} |")
    lines += ["", f"Proving every unit: B {out['rows'][0]['B_full']:.4g}.", "", "Optimum, B the serve-time and proving "
              "total with the scheme's cheapest serve-time set:", "",
              "| scheme | options | policies | ε | p | step | B | B without options | binding |",
              "|---|---|---|---|---|---|---|---|---|"]
    for o in out["optimum"]:
        lines.append(f"| {o['scheme']} | {', '.join(o['options']) or 'none'} | {o['which']} | {o['eps']:g} | "
                     f"{o['luck']:.3g} | {o['step']} | {o['B']:.4g} | {o['B_no_options']:.4g} | {o['binding']} |")
    for eps, rows in out["cluster"].items():
        lines += ["", f"Clustered draws at ε = {eps} (work, all values committed): k units a drawn replay block", "",
                  "| k | proofs a week | blocks | proving B at equal ε | ε at equal proving |", "|---|---|---|---|---|"]
        for c in rows:
            lines.append(f"| {c['k']} | {c['proofs']:.3g} | {c['blocks']:.3g} | {c['B_proving']:.4g} | "
                         f"{c['eps_at_equal_proving']:.3g} |")
    lines += ["", "TS1's table, every value committed at serve time (N_R = 442,368, n_v = 35,840):", "",
              "| M | one-stage | two-stage, interiors after the coins (twoStage_profile) | k | q | two-stage, all "
              "committed | check-all |", "|---|---|---|---|---|---|---|"]
    for t in out["ts1"]:
        lines.append(f"| {t['M']:.4g} | {t['one_stage']:.3g} | {t['deferred']:.3g} | {t['k']} | {t['q']:.3g} | "
                     f"{t['all_committed']:.3g} | {t['check_all']:.3g} |")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="main", help="the cost fit's name, for the report")
    ap.add_argument("--models", default=",".join(sorted(MODELS)))
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--cut-grid", type=int, default=21)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    v = {**{k: i.value for k, i in I.items()}, **{k: i.value for k, i in P_INPUTS.items()}}
    for name in args.models.split(","):
        out = {"label": args.label, **run(v, name, grid=args.grid, cut_grid=args.cut_grid),
               "inputs": {k: i.__dict__ for k, i in {**I, **P_INPUTS}.items()}}
        (args.out / f"opt5-{name}.json").write_text(json.dumps(out, indent=1, default=str))
        text = render(out)
        (args.out / f"opt5-{name}.md").write_text(text + "\n")
        print(text, flush=True)


if __name__ == "__main__":
    main()
