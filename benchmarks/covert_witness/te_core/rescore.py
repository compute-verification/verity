"""The night's v2.3 records scored by the training formalism v2, for the scorer, in one command.

    python -m te_core.rescore --out DIR [--root LEDGER] [--pin COMMIT ...] [--facts FACTS.json]
                              [--quantities measured|analysis|stubs] [--tail literal|frozen] [--charge-context]
                              [--analysis analysis/rescore_v1.py --scorer ~/.research/te-scorer]
                              [--policy reference] [--drop 0.01] [--g G --from-scratch F.json] [--source TEXT]

It reads te_core's ledger (`--root`, default `$TE_LEDGER`), keeps the valid records measured at the pin by
te_core.frontier's rules (by default the pin pin.py reads, v2.3's tonight; `--facts` as for te_core.frontier), and
scores every checkpoint (te_core.formalism: Fit, Slowdown's bound from A1's and A1''s upper bounds, and the attack
side's best-known values beside it), with H and W* over every checkpoint of every valid record (the coordinator's
19:50Z rule). It writes into DIR:

- `frontier.md` and `frontier.json`: Fit, Slowdown's ρ and Security's S = W*/H per start, target and policy cell, with
  W*'s route and its x0 and s* bounds and best-known values (te_core.frontier's `formalism_report`);
- the five panels, each `<name>.json` (a panels-route body) and `<name>.png` drawn from it (te_core.plots), and
  `console/<name>.json`, each cut to the route's limits (te_core.console); they are off the console, so nothing is PUT
  (its one training panel is te_core.panels', which reads the ledger and the quantities through `arguments` and `load`);
- `formalism.jsonl`: each scored row's block and its source;
- with the measured quantities and the analysis in the checkout, `compare.json` and `compare.md`: each scored row's
  quantities beside the analysis's for the same checkpoint (their ratios), the policy's S and route per start and
  target under each, and each route row's Slowdown bound measured over analysis as the ratios of its factors
  (`FACTORS`), which say why S moved.

`--quantities measured` (the default) scores the rows whose `formalism` block ingest measured on the record's traced
step (te_core.capture.circuit through `annotate.with_structures`: σ, the affected share, the consumed tokens, τ, a,
u_in, u, t, Δ_grad; Δ stays 8 |E|); a valid row with none is left out, listed in `frontier.json` as `not_measured`.
`--tail frozen` reads t as the trace's contiguous frozen tail above the forward's last update read
(`frozen_tail_share`) instead of every op that reads no update value, the comparison with the analysis's t.
`--charge-context` scores with te_core.formalism's context charge, a sensitivity for A1: every earlier token's
affected keys and values a block reads, at each layer it works on for a token it doesn't compute from its inputs
(`with_context`, from each run's changed tensors in `--facts`, which it requires; off by default).

`--quantities analysis` attaches the per-record quantities of covert_witness's `analysis/rescore_v1.py` instead
(`--analysis`; it reads the scorer's directory, `--scorer`, and the same ledger), joined to a ledger row by the
record's run and the checkpoint's MACs, and leaves out the rows it has none for (`not_analysed`); the analysis has no
formula of its own, te_core.formalism is the one:

- one part: the checkpoint's gates (the analysis prices each counted op by its circuit Definition), its tokens
  (forward gates over the start's forward gates per token), its token blocks' gates (forward and backward, so
  σ = their share of |C|), their affected gates, and Δ_grad (the run's optimizer state at its widths);
- τ (one token id), a (d_model x 16), Δ = 8 |E|, and u(1) = Δ;
- u and u_in, A1's and A1''s profiles: evenly spread where the run has no frozen tail; with one (its share t above the
  run's highest update, the tied head included), u is 0 up to t and u_in jumps to the full update at once.

u, u_in, u(1), a and Δ are assumption profiles or upper bounds from those witnesses, and their labels say so. Each
analysed row also carries the analysis's `family` (P4's and A2's). `--quantities stubs` (or `--stubs`) gives every
valid row te_core.formalism's stubs from its own fields. Every panel's note and figure's footer names the stubs and
bounds its rows rest on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from . import VERSION, console, formalism, frontier, plots, policy
from .ledger import Ledger

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "rescore_v1.py"
SCORER = Path.home() / ".research" / "te-scorer"
SOURCE = "analysis (rescore_v1)"
MEASURED, ANALYSED, STUBS = "measured", "analysis", "stubs"
LABELS = {
    "gates": f"{SOURCE}: the ledger's counted ops in gates, each by its circuit Definition",
    "tokens": f"{SOURCE}: forward gates over the start's forward gates per token",
    "token_gates": f"{SOURCE}: the ledger's forward and backward ops in gates (the optimizer in no token block)",
    "affected_gates": f"{SOURCE}: token gates less nu F (nu: the forward's share below the lowest update)",
    "delta_grad_bits": f"{SOURCE}: the gradient and optimizer state at the run's optimizer widths",
    "tau_bits": f"stub: one token id per token block, ceil(log2 V) bits ({SOURCE})",
    "a_bits": f"upper bound (witness): d_model x 16, one token's bf16 residual stream ({SOURCE})",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "u1_bits": f"upper bound (witness): u(1) = Delta, every token's path reads the whole update ({SOURCE})",
    "decoder_gates": "stub: the decoder's gates d unmeasured, taken as 0 (rho 1 where Fit holds)",
}
EVEN = {"u": f"assumption (A1): evenly spread, u(x) = x * u(1) ({SOURCE})",
        "u_in": f"assumption (A1'): evenly spread, u_in = u ({SOURCE})"}
TAIL = {"u": f"assumption (A1): frozen layers above the run's highest update, their share t of a token's affected work "
             f"done reading no update bits, the update even over the rest ({SOURCE}'s t; the tied head is one for a run "
             f"that leaves the embedding frozen)",
        "u_in": f"assumption (A1'): frozen layers above changed ones, u_in jumps to the full update at once ({SOURCE})"}
FROZEN = {"tail_share": "measured (trace): the contiguous frozen tail, the token's affected work between the forward's "
                        "last update read and the backward's first (--tail frozen)",
          "u": "assumption (A1), from the trace: 0 up to the contiguous frozen tail's share, then u_in (--tail frozen)"}
UNCHARGED = ("uncharged (--charge-context): the run's changed tensors are not in --facts, so v2's s* (the choice "
             "favourable to the block)")
COMPARED = ("sigma", "affected_share", "w", "tokens", "gates", "tau_bits", "a_bits", "tail_share", "u1_bits",
            "delta_bits", "delta_grad_bits")
TRACE_FIELDS = ("sigma", "affected_share", "affected_all_share", "token_factor", "frozen_tail_share", "a_x",
                "update_cut_bits", "grad_cut_bits", "state_bits", "a_cut_bits")
CELL = ("rho", "fit", "x0", "s_star", "slowdown", "sigma", "s", "alpha_ratio")
#: Slowdown's bound σ (1 − x0) (s/s*) (ᾱ/α_max) with s* = room(u, k)/a, as the factors whose ratios multiply to its ratio
FACTORS = ("sigma", "reach", "s", "a", "room", "alpha")


def load_analysis(path: Path):
    spec = importlib.util.spec_from_file_location("rescore_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block(c: dict, r: dict, tau: float) -> dict:
    """One analysed checkpoint `c` of record `r` as a formalism block (module docstring)."""
    t = float(r.get("t") or 0.0)
    t = t if 0 < t < 1 else 0.0
    return {"parts": [{"gates": c["gates"], "tokens": c["tokens"], "token_gates": c["token_gates"],
                       "affected_gates": c["affected_gates"], "steps": 1, "delta_grad_bits": r["delta_grad"]}],
            "tau_bits": tau, "a_bits": r["a"], "delta_bits": c["delta"], "u1_bits": c["delta"], "u": c["u"],
            "u_in": c["u_in"], "tail_share": t, "decoder_gates": None, "labels": LABELS | (TAIL if t else EVEN)}


def analysed(path: Path, scorer: Path, ledger: Path, cache: Path) -> dict[tuple, tuple[dict, str | None]]:
    """(run id, MACs) -> the analysis's formalism block for that checkpoint, and its record's family."""
    mod = load_analysis(path)
    g = mod.gate_factors(cache)
    subs, launch, starts, counted = mod.load(scorer, ledger)
    records, cps = mod.build(subs, launch, starts, counted, g)
    return {(launch[c["id"]]["run"], c["macs"]): (block(c, records[c["id"]], mod.TAU), records[c["id"]].get("family"))
            for c in cps}


def attach(rows: list[dict], blocks: dict[tuple, tuple[dict, str | None]]) -> tuple[list[dict], dict[str, str]]:
    """The rows, each one the analysis has a block for with that block as `formalism` and its `family`; and each row's
    source."""
    out, source = [], {}
    for r in rows:
        b = blocks.get((r.get("run_id"), r.get("macs")))
        out.append(r | {"formalism": b[0], "family": b[1]} if b else r)
        source[r["id"]] = ANALYSED if b else STUBS
    return out, source


def frozen_tail(rows: list[dict]) -> list[dict]:
    """The measured rows with t the trace's contiguous frozen tail and u rebuilt from u_in on it (`--tail frozen`)."""
    out = []
    for r in rows:
        if frontier.measured(r):
            f = r["formalism"]
            t = f["trace"]["frozen_tail_share"]
            t = t if t < 1 else 0.0
            u = formalism.with_tail(tuple(map(tuple, f["u_in"])), t)
            r = r | {"formalism": f | {"tail_share": t, "u": [list(p) for p in u],
                                       "labels": f.get("labels", {}) | FROZEN}}
        out.append(r)
    return out


def with_context(rows: list[dict], shapes: dict) -> list[dict]:
    """The rows with the context charge's quantities in their formalism block (`--charge-context`,
    te_core.formalism.context), from each row's changed tensors (`--facts`) and its start's shape. A row whose run has
    no changed tensors in the facts stays uncharged, labelled (`UNCHARGED`): the choice favourable to the block."""
    out = []
    for r in rows:
        shape, f = shapes.get(r["start"]), r.get("formalism") or {}
        if shape is not None and r.get("changed") is not None:
            r = r | {"formalism": f | {"context": formalism.context(r["changed"], shape)}}
        elif shape is not None:
            r = r | {"formalism": f | {"labels": f.get("labels", {}) | {"context": UNCHARGED}}}
        out.append(r)
    return out


def quantities(m: formalism.Measured) -> dict:
    return {"sigma": m.sigma, "affected_share": frontier.affected_share(m), "w": m.w, "tokens": m.tokens,
            "gates": m.gates, "tau_bits": m.tau_bits, "a_bits": m.a_bits, "tail_share": m.tail_share,
            "u1_bits": m.u1_bits, "delta_bits": m.delta_bits, "delta_grad_bits": m.delta_grad_bits}


def _ratio(a, b):
    if a is None or b is None:
        return None
    return a / b if b else (1.0 if a == b else math.inf)


def reference_s(rep: dict, name: str) -> dict:
    """(start, drop) -> the `name` cell's S, ρ, basis, Slowdown terms and the route that sets W* in `rep`
    (`formalism_report`), with the route's row id."""
    out = {}
    for start, v in rep["starts"].items():
        for t in v["targets"]:
            c = t["cells"].get(name) or {}
            out[(start, t["drop"])] = {"S": c.get("S"), "rho": c.get("rho"), "basis": c.get("basis"),
                                       "route": frontier._route(c.get("of")), "id": (c.get("of") or {}).get("id"),
                                       "honest": t["honest"], **{n: c.get(n) for n in frontier.TERMS}}
    return out


def factors(v: dict, va: dict, a: float, a_analysis: float) -> dict:
    """Slowdown's bound in cell `v` over cell `va`, factor by factor (`FACTORS`): reach is 1 − x0 and room is s*·a;
    σ·reach·s·a·alpha over room is the bound's ratio."""
    room, room_a = v["s_star"] * a, va["s_star"] * a_analysis
    return {"sigma": _ratio(v["sigma"], va["sigma"]), "reach": _ratio(1 - v["x0"], 1 - va["x0"]),
            "s": _ratio(v["s"], va["s"]), "a": _ratio(a, a_analysis), "room": _ratio(room, room_a),
            "alpha": _ratio(v["alpha_ratio"], va["alpha_ratio"])}


def compare(scored: list[dict], analysed_rows: list[dict], shapes: dict, rep: dict, rep_a: dict, name: str) -> dict:
    """Each scored row's measured quantities beside the analysis's for the same checkpoint and their ratios, ρ and
    Slowdown's terms in the `name` cell under each, and the `name` cell's S and route per start and target under each."""
    by = {r["id"]: r for r in analysed_rows}
    rows = []
    for r in scored:
        shape = shapes.get(r["start"])
        a = by.get(r["id"])
        if shape is None or a is None:
            continue
        c, k = policy.cells(shape)[name]
        m, ma = formalism.measure(r, shape), formalism.measure(a, shape)
        q, qa = quantities(m), quantities(ma)
        v, va = formalism.at(m, c, k), formalism.at(ma, c, k)
        tr = r["formalism"]["trace"]
        rows.append({"id": r["id"], "run_id": r.get("run_id"), "start": r["start"], "direction": r.get("direction"),
                     "mechanism": r.get("mechanism"), "checkpoint_step": r.get("checkpoint_step"),
                     "family": a.get("family"), "measured": q, "analysis": qa,
                     "ratio": {x: _ratio(q[x], qa[x]) for x in COMPARED},
                     "trace": {x: tr.get(x) for x in TRACE_FIELDS},
                     "update_cut_over_delta": _ratio(tr.get("update_cut_bits"), m.delta_bits),
                     "cell": {x: v[x] for x in CELL}, "cell_analysis": {x: va[x] for x in CELL},
                     "factors": factors(v, va, m.a_bits, ma.a_bits)})
    s, sa = reference_s(rep, name), reference_s(rep_a, name)
    by_row = {r["id"]: r for r in rows}
    cells = [{"start": x[0], "drop": x[1], "measured": s[x], "analysis": sa.get(x),
              "routes": {k: _why(by_row.get((e or {}).get("id"))) for k, e in (("measured", s[x]), ("analysis", sa.get(x)))}}
             for x in s]
    return {"cell": name, "rows": rows, "S": cells}


def _why(r: dict | None) -> dict | None:
    """A route row's ρ and Slowdown's bound under each set of quantities, and the bound's factors."""
    if r is None:
        return None
    return {"id": r["id"], "checkpoint_step": r["checkpoint_step"], "rho": r["cell"]["rho"],
            "rho_analysis": r["cell_analysis"]["rho"], "slowdown": r["cell"]["slowdown"],
            "slowdown_analysis": r["cell_analysis"]["slowdown"], "fit": r["cell"]["fit"],
            "fit_analysis": r["cell_analysis"]["fit"], "factors": r["factors"]}


def _g(x) -> str:
    return "—" if x is None else "inf" if x == math.inf else f"{x:.3g}"


def compare_markdown(c: dict) -> str:
    lines = [f"Measured (trace) over analysis (rescore_v1) per checkpoint; ρ, x0 and s* at the {c['cell']} cell.", "",
             "| record | start | route | σ | affected share | w | a | t (measured, analysis) | u(1) | Δ | Δ_grad | gates "
             "| ρ measured (x0, s*) | ρ analysis (x0, s*) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c["rows"]:
        q, qa, x, v, va = r["measured"], r["analysis"], r["ratio"], r["cell"], r["cell_analysis"]
        lines.append(
            f"| {r['id'][-12:]}@{r['checkpoint_step']} | {r['start']} | {r['direction']}/{r['mechanism']} | "
            f"{_g(x['sigma'])} | {_g(x['affected_share'])} | {_g(x['w'])} | {_g(x['a_bits'])} | "
            f"{_g(q['tail_share'])}, {_g(qa['tail_share'])} | {_g(x['u1_bits'])} | {_g(x['delta_bits'])} | "
            f"{_g(x['delta_grad_bits'])} | {_g(x['gates'])} | {_g(v['rho'])} ({_g(v['x0'])}, {_g(v['s_star'])}) | "
            f"{_g(va['rho'])} ({_g(va['x0'])}, {_g(va['s_star'])}) |")
    lines += ["", f"S at the {c['cell']} cell, measured and analysis, with W*'s route.", "",
              "| start | target | S measured | route | S analysis | route |", "|---|---|---|---|---|---|"]
    for e in c["S"]:
        m, a = e["measured"], e["analysis"] or {}
        lines.append(f"| {e['start']} | -{100 * e['drop']:g}% | {_g(m['S'])} | {m['route']} | "
                     f"{_g(a.get('S'))} | {a.get('route', '')} |")
    lines += ["", "Why S moved: each route row's ρ measured (analysis), and Slowdown's bound measured over analysis as "
              "its factors (σ, reach 1 − x0, s, a, room = s*·a, ᾱ/α_max; σ·reach·s·a·alpha/room is the ratio).", "",
              "| start | target | S measured / analysis | route | ρ | σ | reach | s | a | room | alpha |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in c["S"]:
        m, a = e["measured"], e["analysis"] or {}
        for k in ("measured", "analysis"):
            w = e["routes"][k]
            if w is None:
                continue
            f = w["factors"]
            lines.append(f"| {e['start']} | -{100 * e['drop']:g}% | {_g(_ratio(m['S'], a.get('S')))} | {k}: "
                         f"{(m if k == 'measured' else a)['route']} | {_g(w['rho'])} ({_g(w['rho_analysis'])}) | "
                         + " | ".join(_g(f[x]) for x in FACTORS) + " |")
    return "\n".join(lines)


def arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ledger and quantities options `load` reads (te_core.panels takes them too)."""
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="te_core's ledger (default $TE_LEDGER)")
    ap.add_argument("--pin", action="append", default=None, help="a measurement commit, repeated (default pin.py's)")
    ap.add_argument("--facts", type=Path, default=None, help="as te_core.frontier's --facts")
    ap.add_argument("--quantities", choices=(MEASURED, ANALYSED, STUBS), default=MEASURED)
    ap.add_argument("--tail", choices=("literal", "frozen"), default="literal",
                    help="measured t: every op reading no update value, or the contiguous frozen tail")
    ap.add_argument("--charge-context", action="store_true",
                    help="charge earlier tokens' affected keys and values (te_core.formalism's context charge; "
                         "reads each run's changed tensors from --facts)")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="the per-record quantities (rescore_v1.py)")
    ap.add_argument("--scorer", type=Path, default=SCORER, help="the scorer's directory, which the analysis reads")
    ap.add_argument("--stubs", action="store_true", help="--quantities stubs")
    ap.add_argument("--source", default=None, help="the panels' `source` (default this command and the pin)")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    ap.add_argument("--ladder-fit", type=Path, default=None,
                    help="the ladder's fit the scoring cap reads (te_core.policy's `ladder_fit`; default none: the "
                         "start's forward caps every checkpoint's)")
    return ap


def load(a) -> dict:
    """The ledger's rows with their quantities (`--quantities`: the measured ones, t by `--tail`; the analysis's blocks
    attached, `attach`; or the stubs), the analysis's blocks wherever the checkout has it (`blocks`, the gate factors
    cached in `a.out`), the pin, the starts' shapes, which rows to score (`keep`: those with the quantities, None for
    every row), each row's `source`, how the quantities were got (`how`, and `quantities` in a word or two), the
    list in `frontier.json` the valid rows without them go in (`left`) and the ladder's fit the scoring cap reads
    (`ladder`, None without `--ladder-fit`)."""
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.facts:
        rows = frontier.with_facts(rows, json.loads(a.facts.read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    mode = STUBS if a.stubs else a.quantities
    if mode == ANALYSED and not a.analysis.exists():
        mode = STUBS
    blocks = analysed(a.analysis, a.scorer, ledger.root, a.out / "analysis-gates.json") \
        if mode != STUBS and a.analysis.exists() else {}
    if mode == MEASURED:
        rows = frozen_tail(rows) if a.tail == "frozen" else rows
        source = {r["id"]: MEASURED if frontier.measured(r) else STUBS for r in rows}
        keep, how, word = frontier.measured, f"measured (trace; t {a.tail})", f"measured, t {a.tail}"
    else:
        rows, source = attach(rows, blocks)
        keep = (lambda r: source[r["id"]] == ANALYSED) if mode == ANALYSED else None
        how = str(a.analysis) if mode == ANALYSED else \
            "stubs: " + ("--stubs" if a.stubs or a.quantities == STUBS else f"no {a.analysis} in this checkout")
        word = "rescore_v1" if mode == ANALYSED else "stubs"
    shapes = policy.starts(a.manifest)
    if a.charge_context:
        if not a.facts:
            raise SystemExit("--charge-context reads each run's changed tensors from --facts")
        rows, how, word = with_context(rows, shapes), f"{how}; context charged", f"{word}, context charged"
    return {"rows": rows, "blocks": blocks, "mode": mode, "pin": tuple(a.pin) if a.pin else frontier.pinned(),
            "shapes": shapes, "source": source, "keep": keep, "use": mode != STUBS, "how": how,
            "quantities": word, "left": {MEASURED: "not_measured", ANALYSED: "not_analysed"}.get(mode),
            "ladder": policy.ladder_fit(a.ladder_fit) if a.ladder_fit else None}


def main(argv=None) -> int:
    ap = arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--policy", default=policy.REFERENCE)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--from-scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load(a)
    rows, blocks, mode, pin, shapes, source, keep, how, left, ladder = (
        d[k] for k in ("rows", "blocks", "mode", "pin", "shapes", "source", "keep", "how", "left", "ladder"))
    rep = frontier.formalism_report(rows, pin, shapes, keep, ladder)
    left_out = frontier.status(rows, pin)
    valid = [r for r in rows if r["id"] not in left_out]
    kept = [r for r in valid if keep is None or keep(r)]
    missing = sorted(r["id"] for r in valid if keep is not None and not keep(r))
    (a.out / "frontier.json").write_text(json.dumps(
        {"pin": list(pin), "left_out": left_out, "quantities": how,
         "not_analysed": missing if mode == ANALYSED else [], "not_measured": missing if mode == MEASURED else []}
        | rep, indent=1, default=str) + "\n")
    (a.out / "frontier.md").write_text(frontier.formalism_markdown(rep) + "\n")
    with (a.out / "formalism.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "run_id": r.get("run_id"), "start": r["start"],
                                "checkpoint_step": r.get("checkpoint_step"), "source": source[r["id"]],
                                "family": r.get("family"), "formalism": r.get("formalism")}, default=str) + "\n")
    if mode == MEASURED and blocks:
        arows, asource = attach(rows, blocks)
        arows = with_context(arows, shapes) if a.charge_context else arows
        rep_a = frontier.formalism_report(arows, pin, shapes, lambda r: asource[r["id"]] == ANALYSED, ladder)
        c = compare(kept, [r for r in arows if asource[r["id"]] == ANALYSED], shapes, rep, rep_a, a.policy)
        (a.out / "compare.json").write_text(json.dumps(c, indent=1, default=str) + "\n")
        (a.out / "compare.md").write_text(compare_markdown(c) + "\n")
    scored, stubs = plots.load(rows, pin, shapes, keep, ladder)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    what = a.source or (f"python -m te_core.rescore (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                        f"{len(kept)} valid checkpoints, quantities {how})")
    written = plots.write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop, what)
    console.write(a.out, a.out / "console")
    print(frontier.formalism_markdown(rep))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(valid)} valid rows, {len(kept)} scored with {how}"
          + (f", {len(missing)} left out (frontier.json's {left})" if left else "")
          + f" -> {a.out}: frontier.md, frontier.json, formalism.jsonl, "
          + ("compare.md, compare.json, " if mode == MEASURED and blocks else "")
          + ", ".join(f"{p.stem}.json/.png" for p in written) + ", console/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
