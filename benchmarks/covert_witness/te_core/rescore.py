"""The night's v2.3 records scored by the training formalism v1, for the scorer, in one command.

    python -m te_core.rescore --out DIR [--root LEDGER] [--pin COMMIT ...] [--facts FACTS.json]
                              [--analysis analysis/rescore_v1.py --scorer ~/.research/te-scorer | --stubs]
                              [--reading literal-x0|tail-charged ...] [--policy reference] [--drop 0.01]
                              [--g G --from-scratch F.json] [--source TEXT]

It reads te_core's ledger (`--root`, default `$TE_LEDGER`), keeps the valid records measured at the pin by
te_core.frontier's rules (by default the pin pin.py reads, v2.3's tonight; `--facts` as for te_core.frontier),
attaches each one's `formalism` block, and scores every checkpoint under each reading of Slowdown's frozen-tail case
(`--reading`, repeated; default both, te_core.formalism's READINGS, of which neither is the headline), with H and W*
over every checkpoint of every valid record (the coordinator's 19:50Z rule). It writes into DIR:

- `frontier.md` and `frontier.json`: Fit, Slowdown's ρ and Security's S = W*/H per start, target, reading and policy
  cell, with W*'s route (te_core.frontier's `formalism_report`);
- the console's five panels, each `<name>.json` (the panels route's body, `reading` a column with each reading's rows)
  and `<name>.png` drawn from it (te_core.plots);
- `formalism.jsonl`: each scored row's attached block and its source.

The block holds the per-record quantities of covert_witness's `analysis/rescore_v1.py` where that file is present
(`--analysis`, default benchmarks/covert_witness/analysis/rescore_v1.py; it reads the scorer's directory, `--scorer`,
and the same ledger), joined to a ledger row by the record's run and the checkpoint's MACs; the analysis has no
formula of its own, te_core.formalism is the one:

- one part: the checkpoint's gates (the analysis prices each counted op by its circuit Definition), its tokens
  (forward gates over the start's forward gates per token), its dependent gates φ·|C|, and Δ_grad (the run's optimizer
  state at its widths);
- τ (one token id), a (d_model x 16), Δ = 8 |E|, and u(1) = Δ;
- u, evenly spread from the token's start, and the frozen tail's share t above the run's highest update (the
  analysis's t, the tied head included; 0 where there is none), which the two readings read differently.

u, u(1), t, a and Δ are upper bounds from those witnesses, and their labels say so. Each row also carries the
analysis's `family` (P4's and A2's). With the analysis, a valid row it has no block for (a run outside the scorer's
regime, or submitted after the scorer's snapshot) is left out, listed in `frontier.json` as `not_analysed`, so no score
mixes the two sources. With `--stubs`, or without the analysis in the checkout, every valid row gets
te_core.formalism's stubs from its own fields. Every panel's note and figure's footer names the stubs and bounds its
rows rest on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import VERSION, formalism, frontier, plots, policy
from .ledger import Ledger

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "rescore_v1.py"
SCORER = Path.home() / ".research" / "te-scorer"
SOURCE = "analysis (rescore_v1)"
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
        source[r["id"]] = "analysis" if b else "stubs"
    return out, source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="te_core's ledger (default $TE_LEDGER)")
    ap.add_argument("--pin", action="append", default=None, help="a measurement commit, repeated (default pin.py's)")
    ap.add_argument("--facts", type=Path, default=None, help="as te_core.frontier's --facts")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="the per-record quantities (rescore_v1.py)")
    ap.add_argument("--scorer", type=Path, default=SCORER, help="the scorer's directory, which the analysis reads")
    ap.add_argument("--stubs", action="store_true", help="te_core.formalism's stubs for every row, no analysis")
    ap.add_argument("--reading", action="append", choices=formalism.READINGS, default=None,
                    help="one reading of Slowdown's frozen-tail case, repeated (default both)")
    ap.add_argument("--policy", default=policy.REFERENCE)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--from-scratch", type=Path, default=None)
    ap.add_argument("--source", default=None, help="the panels' `source` (default this command and the pin)")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    a = ap.parse_args(argv)
    readings = tuple(a.reading or formalism.READINGS)
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.facts:
        rows = frontier.with_facts(rows, json.loads(a.facts.read_text()))
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    shapes = policy.starts(a.manifest)
    a.out.mkdir(parents=True, exist_ok=True)
    use = not a.stubs and a.analysis.exists()
    blocks = analysed(a.analysis, a.scorer, ledger.root, a.out / "analysis-gates.json") if use else {}
    rows, source = attach(rows, blocks)
    keep = (lambda r: source[r["id"]] == "analysis") if use else None
    rep = frontier.formalism_report(rows, pin, shapes, keep, readings)
    left_out = frontier.status(rows, pin)
    valid = [r for r in rows if r["id"] not in left_out]
    kept = [r for r in valid if keep is None or keep(r)]
    not_analysed = sorted(r["id"] for r in valid if keep is not None and not keep(r))
    how = str(a.analysis) if use else "stubs: " + ("--stubs" if a.stubs else f"no {a.analysis} in this checkout")
    (a.out / "frontier.json").write_text(json.dumps({"pin": list(pin), "left_out": left_out, "quantities": how,
                                                     "not_analysed": not_analysed} | rep, indent=1, default=str) + "\n")
    (a.out / "frontier.md").write_text(frontier.formalism_markdown(rep) + "\n")
    with (a.out / "formalism.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "run_id": r.get("run_id"), "start": r["start"],
                                "checkpoint_step": r.get("checkpoint_step"), "source": source[r["id"]],
                                "family": r.get("family"), "formalism": r.get("formalism")}, default=str) + "\n")
    scored, stubs = plots.load(rows, pin, shapes, keep, readings)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    what = a.source or (f"python -m te_core.rescore (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                        f"{len(kept)} valid checkpoints, quantities {'rescore_v1' if use else 'stubs'})")
    written = plots.write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop, readings, what)
    print(frontier.formalism_markdown(rep))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(valid)} valid rows, {len(kept)} scored with {how} under "
          f"{', '.join(readings)}" + (f", {len(not_analysed)} not analysed (left out, frontier.json's not_analysed)"
                                      if use else "")
          + f" -> {a.out}: frontier.md, frontier.json, formalism.jsonl, "
          + ", ".join(f"{p.stem}.json/.png" for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
