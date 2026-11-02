"""The night's v2.3 records scored by the training formalism v1, for the scorer, in one command.

    python -m te_core.rescore --out DIR [--root LEDGER] [--pin COMMIT ...] [--facts FACTS.json]
                              [--quantities measured|analysis|stubs] [--tail literal|frozen]
                              [--analysis analysis/rescore_v1.py --scorer ~/.research/te-scorer]
                              [--reading literal-x0|tail-charged ...] [--policy reference] [--drop 0.01]
                              [--g G --from-scratch F.json] [--source TEXT]

It reads te_core's ledger (`--root`, default `$TE_LEDGER`), keeps the valid records measured at the pin by
te_core.frontier's rules (by default the pin pin.py reads, v2.3's tonight; `--facts` as for te_core.frontier),
scores every checkpoint under each reading of Slowdown's frozen-tail case (`--reading`, repeated; default both,
te_core.formalism's READINGS, of which neither is the headline), with H and W* over every checkpoint of every valid
record (the coordinator's 19:50Z rule). It writes into DIR:

- `frontier.md` and `frontier.json`: Fit, Slowdown's ρ and Security's S = W*/H per start, target, reading and policy
  cell, with W*'s route (te_core.frontier's `formalism_report`);
- the console's five panels, each `<name>.json` (the panels route's body, `reading` a column with each reading's rows)
  and `<name>.png` drawn from it (te_core.plots);
- `formalism.jsonl`: each scored row's block and its source;
- with the measured quantities and the analysis in the checkout, `compare.json` and `compare.md`: each scored row's
  quantities beside the analysis's for the same checkpoint (their ratios), and the reference policy's S and route per
  start, target and reading under each.

`--quantities measured` (the default) scores the rows whose `formalism` block ingest measured on the record's traced
step (te_core.capture.circuit through `annotate.with_structures`: φ, the consumed tokens, τ, a, u, t, Δ_grad; Δ stays
8 |E|); a valid row with none is left out, listed in `frontier.json` as `not_measured`. `--tail frozen` reads t as the
trace's contiguous frozen tail above the forward's last update read (`frozen_tail_share`) instead of every op that
reads no update value, the comparison with the analysis's t.

`--quantities analysis` attaches the per-record quantities of covert_witness's `analysis/rescore_v1.py` instead
(`--analysis`; it reads the scorer's directory, `--scorer`, and the same ledger), joined to a ledger row by the
record's run and the checkpoint's MACs, and leaves out the rows it has none for (`not_analysed`):

- one part: the checkpoint's gates (the analysis prices each counted op by its circuit Definition), its tokens
  (forward gates over the start's forward gates per token), its dependent gates φ·|C|, and Δ_grad (the run's optimizer
  state at its widths);
- τ (one token id), a (d_model x 16), Δ = 8 |E|, and u(1) = Δ;
- u, evenly spread from the token's start, and the frozen tail's share t above the run's highest update (the
  analysis's t, the tied head included; 0 where there is none).

`--quantities stubs` (or `--stubs`) gives every valid row te_core.formalism's stubs from its own fields. Every panel's
note and figure's footer names the stubs and bounds its rows rest on; the analysis has no formula of its own,
te_core.formalism is the one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

from . import VERSION, formalism, frontier, plots, policy
from .ledger import Ledger

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "rescore_v1.py"
SCORER = Path.home() / ".research" / "te-scorer"
SOURCE = "analysis (rescore_v1)"
MEASURED, ANALYSED, STUBS = "measured", "analysis", "stubs"
LABELS = {
    "gates": f"{SOURCE}: the ledger's counted ops in gates, each by its circuit Definition",
    "tokens": f"{SOURCE}: forward gates over the start's forward gates per token",
    "dep_gates": f"{SOURCE}: phi |C|, phi = 1 - nu F/G (nu: the forward's share below the lowest update)",
    "delta_grad_bits": f"{SOURCE}: the gradient and optimizer state at the run's optimizer widths",
    "tau_bits": f"stub: one token id per token block, ceil(log2 V) bits ({SOURCE})",
    "a_bits": f"upper bound (witness): d_model x 16, one token's bf16 residual stream ({SOURCE})",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "u1_bits": f"upper bound (witness): u(1) = Delta, every token's path reads the whole update ({SOURCE})",
    "u": formalism.STUBS["u"],
    "decoder_gates": formalism.STUBS["decoder_gates"],
}
TAIL = (f"upper bound (witness): the frozen tail above the run's highest update, its share t of a token's dependent "
        f"work, reads no update bits ({SOURCE}'s t; the tied head is one for a run that leaves the embedding frozen)")
FROZEN = ("measured (trace): the contiguous frozen tail, the token's dependent work between the forward's last update "
          "read and the backward's first (--tail frozen)")
COMPARED = ("phi", "w", "tokens", "gates", "tau_bits", "a_bits", "tail_share", "u1_bits", "delta_bits",
            "delta_grad_bits")


def load_analysis(path: Path):
    spec = importlib.util.spec_from_file_location("rescore_v1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def block(c: dict, r: dict, tau: float) -> dict:
    """One analysed checkpoint `c` of record `r` as a formalism block (module docstring)."""
    t = float(r.get("t") or 0.0)
    return {"parts": [{"gates": c["gates"], "tokens": c["tokens"], "dep_gates": c["phi"] * c["gates"], "steps": 1,
                       "delta_grad_bits": r["delta_grad"]}],
            "tau_bits": tau, "a_bits": r["a"], "delta_bits": c["delta"], "u1_bits": c["delta"],
            "u": [list(p) for p in formalism.even(c["delta"])], "tail_share": t if 0 < t < 1 else 0.0,
            "decoder_gates": None, "labels": LABELS | ({"tail_share": TAIL} if 0 < t < 1 else {})}


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
    """The measured rows with t the trace's contiguous frozen tail (`--tail frozen`)."""
    out = []
    for r in rows:
        if frontier.measured(r):
            f = r["formalism"]
            t = f["trace"]["frozen_tail_share"]
            r = r | {"formalism": f | {"tail_share": t if t < 1 else 0.0,
                                       "labels": f.get("labels", {}) | {"tail_share": FROZEN}}}
        out.append(r)
    return out


def quantities(m: formalism.Measured) -> dict:
    return {"phi": m.phi, "w": m.w, "tokens": m.tokens, "gates": m.gates, "tau_bits": m.tau_bits, "a_bits": m.a_bits,
            "tail_share": m.tail_share, "u1_bits": m.u1_bits, "delta_bits": m.delta_bits,
            "delta_grad_bits": m.delta_grad_bits}


def _ratio(a, b):
    if a is None or b is None:
        return None
    return a / b if b else (1.0 if a == b else math.inf)


def reference_s(rep: dict, name: str) -> dict:
    """(start, drop, reading) -> the `name` cell's S and the route that sets W* in `rep` (`formalism_report`)."""
    out = {}
    for start, v in rep["starts"].items():
        for t in v["targets"]:
            c = t["cells"].get(name) or {}
            out[(start, t["drop"], t["reading"])] = {"S": c.get("S"), "rho": c.get("rho"), "basis": c.get("basis"),
                                                     "route": frontier._route(c.get("of")), "honest": t["honest"]}
    return out


def compare(scored: list[dict], analysed_rows: list[dict], shapes: dict, rep: dict, rep_a: dict,
            name: str, readings) -> dict:
    """Each scored row's measured quantities beside the analysis's for the same checkpoint and their ratios, ρ in the
    `name` cell under each reading, and the `name` cell's S and route per start, target and reading under each."""
    by = {r["id"]: r for r in analysed_rows}
    rows = []
    for r in scored:
        shape = shapes.get(r["start"])
        a = by.get(r["id"])
        if shape is None or a is None:
            continue
        cell = {name: policy.cells(shape)[name]}
        m, ma = formalism.measure(r, shape), formalism.measure(a, shape)
        q, qa = quantities(m), quantities(ma)
        rho, _ = formalism.rhos(m, cell, readings)
        rho_a, _ = formalism.rhos(ma, cell, readings)
        tr = r["formalism"]["trace"]
        rows.append({"id": r["id"], "run_id": r.get("run_id"), "start": r["start"], "direction": r.get("direction"),
                     "mechanism": r.get("mechanism"), "checkpoint_step": r.get("checkpoint_step"),
                     "family": a.get("family"), "measured": q, "analysis": qa,
                     "ratio": {k: _ratio(q[k], qa[k]) for k in COMPARED},
                     "trace": {k: tr.get(k) for k in ("sigma", "dep_share", "token_factor", "frozen_tail_share",
                                                      "update_cut_bits", "grad_cut_bits", "state_bits", "a_cut_bits")},
                     "update_cut_over_delta": _ratio(tr.get("update_cut_bits"), m.delta_bits),
                     "rho": {k: rho[k][name] for k in readings}, "rho_analysis": {k: rho_a[k][name] for k in readings}})
    s, sa = reference_s(rep, name), reference_s(rep_a, name)
    cells = [{"start": k[0], "drop": k[1], "reading": k[2], "measured": s[k], "analysis": sa.get(k)} for k in s]
    return {"cell": name, "rows": rows, "S": cells}


def _g(x) -> str:
    return "—" if x is None else "inf" if x == math.inf else f"{x:.3g}"


def compare_markdown(c: dict) -> str:
    lines = [f"Measured (trace) over analysis (rescore_v1) per checkpoint; ρ at the {c['cell']} cell under each reading.",
             "", "| record | start | route | φ | w | a | t (measured, analysis) | u(1) | Δ | Δ_grad | gates | ρ measured | "
             "ρ analysis |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in c["rows"]:
        q, qa, x = r["measured"], r["analysis"], r["ratio"]
        lines.append(
            f"| {r['id'][-12:]}@{r['checkpoint_step']} | {r['start']} | {r['direction']}/{r['mechanism']} | "
            f"{_g(x['phi'])} | {_g(x['w'])} | {_g(x['a_bits'])} | {_g(q['tail_share'])}, {_g(qa['tail_share'])} | "
            f"{_g(x['u1_bits'])} | {_g(x['delta_bits'])} | {_g(x['delta_grad_bits'])} | {_g(x['gates'])} | "
            + " / ".join(_g(v) for v in r["rho"].values()) + " | " + " / ".join(_g(v) for v in r["rho_analysis"].values())
            + " |")
    lines += ["", f"S at the {c['cell']} cell, measured and analysis, with W*'s route.", "",
              "| start | target | reading | S measured | route | S analysis | route |", "|---|---|---|---|---|---|---|"]
    for e in c["S"]:
        m, a = e["measured"], e["analysis"] or {}
        lines.append(f"| {e['start']} | -{100 * e['drop']:g}% | {e['reading']} | {_g(m['S'])} | {m['route']} | "
                     f"{_g(a.get('S'))} | {a.get('route', '')} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="te_core's ledger (default $TE_LEDGER)")
    ap.add_argument("--pin", action="append", default=None, help="a measurement commit, repeated (default pin.py's)")
    ap.add_argument("--facts", type=Path, default=None, help="as te_core.frontier's --facts")
    ap.add_argument("--quantities", choices=(MEASURED, ANALYSED, STUBS), default=MEASURED)
    ap.add_argument("--tail", choices=("literal", "frozen"), default="literal",
                    help="measured t: every op reading no update value, or the contiguous frozen tail")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="the per-record quantities (rescore_v1.py)")
    ap.add_argument("--scorer", type=Path, default=SCORER, help="the scorer's directory, which the analysis reads")
    ap.add_argument("--stubs", action="store_true", help="--quantities stubs")
    ap.add_argument("--reading", action="append", choices=formalism.READINGS, default=None,
                    help="one reading of Slowdown's frozen-tail case, repeated (default both)")
    ap.add_argument("--policy", default=policy.REFERENCE)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--from-scratch", type=Path, default=None)
    ap.add_argument("--source", default=None, help="the panels' `source` (default this command and the pin)")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    a = ap.parse_args(argv)
    mode = STUBS if a.stubs else a.quantities
    readings = tuple(a.reading or formalism.READINGS)
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.facts:
        rows = frontier.with_facts(rows, json.loads(a.facts.read_text()))
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    shapes = policy.starts(a.manifest)
    a.out.mkdir(parents=True, exist_ok=True)
    if mode == ANALYSED and not a.analysis.exists():
        mode = STUBS
    blocks = analysed(a.analysis, a.scorer, ledger.root, a.out / "analysis-gates.json") \
        if mode != STUBS and a.analysis.exists() else {}
    if mode == MEASURED:
        frontier_rows = frozen_tail(rows) if a.tail == "frozen" else rows
        source = {r["id"]: MEASURED if frontier.measured(r) else STUBS for r in rows}
        keep, how = frontier.measured, f"measured (trace; t {a.tail})"
    else:
        frontier_rows, source = attach(rows, blocks)
        keep = (lambda r: source[r["id"]] == ANALYSED) if mode == ANALYSED else None
        how = str(a.analysis) if mode == ANALYSED else \
            "stubs: " + ("--stubs" if a.stubs or a.quantities == STUBS else f"no {a.analysis} in this checkout")
    rep = frontier.formalism_report(frontier_rows, pin, shapes, keep, readings)
    left_out = frontier.status(frontier_rows, pin)
    valid = [r for r in frontier_rows if r["id"] not in left_out]
    kept = [r for r in valid if keep is None or keep(r)]
    missing = sorted(r["id"] for r in valid if keep is not None and not keep(r))
    left = {MEASURED: "not_measured", ANALYSED: "not_analysed"}.get(mode)
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
        rep_a = frontier.formalism_report(arows, pin, shapes, lambda r: asource[r["id"]] == ANALYSED, readings)
        c = compare(kept, [r for r in arows if asource[r["id"]] == ANALYSED], shapes, rep, rep_a, a.policy, readings)
        (a.out / "compare.json").write_text(json.dumps(c, indent=1, default=str) + "\n")
        (a.out / "compare.md").write_text(compare_markdown(c) + "\n")
    scored, stubs = plots.load(frontier_rows, pin, shapes, keep, readings)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    what = a.source or (f"python -m te_core.rescore (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                        f"{len(kept)} valid checkpoints, quantities {how})")
    written = plots.write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop, readings, what)
    print(frontier.formalism_markdown(rep))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(valid)} valid rows, {len(kept)} scored with {how} under "
          f"{', '.join(readings)}" + (f", {len(missing)} left out (frontier.json's {left})" if left else "")
          + f" -> {a.out}: frontier.md, frontier.json, formalism.jsonl, "
          + ("compare.md, compare.json, " if mode == MEASURED and blocks else "")
          + ", ".join(f"{p.stem}.json/.png" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
