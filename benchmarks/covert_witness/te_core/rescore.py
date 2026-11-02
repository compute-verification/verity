"""The night's v2.3 records scored by the training formalism v1, for the scorer, in one command.

    python -m te_core.rescore --out DIR [--root LEDGER] [--pin COMMIT ...] [--facts FACTS.json]
                              [--analysis analysis/rescore_v1.py --scorer ~/.research/te-scorer | --stubs]
                              [--policy reference] [--drop 0.01] [--g G --from-scratch F.json]

It reads te_core's ledger (`--root`, default `$TE_LEDGER`), keeps the valid records measured at the pin by
te_core.frontier's rules (by default the pin pin.py reads, v2.3's tonight; `--facts` as for te_core.frontier),
attaches each one's `formalism` block, and writes into DIR:

- `frontier.md` and `frontier.json`: Fit, Slowdown's ρ and Security's S = W*/H per start, target and policy cell
  (te_core.frontier's `formalism_report`);
- `P1`-`P4` and `A2`, each a PNG and the JSON of its numbers (te_core.plots);
- `formalism.jsonl`: each valid row's attached block and its source.

The block holds the analytic v1 quantities of covert_witness's `analysis/rescore_v1.py` where that file is present
(`--analysis`, default benchmarks/covert_witness/analysis/rescore_v1.py; it reads the scorer's directory, `--scorer`,
and the same ledger), joined to a ledger row by the record's run and the checkpoint's MACs:

- one part: the checkpoint's gates (the analysis prices each counted op by its circuit Definition), its tokens
  (forward gates over the start's forward gates per token), its dependent gates φ·|C|, and Δ_grad (the run's optimizer
  state at its widths);
- τ (one token id), a (d_model x 16), Δ = 8 |E|, and u(1) = Δ;
- u, the best-known profile from witnesses: evenly spread, and, where the run leaves a frozen tail above its highest
  update (the analysis's share t), 0 up to t (te_core.formalism's `frozen_tail`).

u, u(1), a and Δ are upper bounds from those witnesses, and their labels say so. With the analysis, a valid row it has
no block for (a run outside the scorer's regime, or submitted after the scorer's snapshot) is left out, listed in
`frontier.json` as `not_analysed`, so no score mixes the two sources; complete schedules are still read over every
row. With `--stubs`, or without the
analysis in the checkout, every valid row gets te_core.formalism's stubs from its own fields. Every figure's footer
names the stubs and bounds its points rest on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import formalism, frontier, plots, policy
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
    "decoder_gates": formalism.STUBS["decoder_gates"],
}
U_EVEN = formalism.STUBS["u"]
U_TAIL = (f"upper bound (witnesses): the least of evenly spread x u(1) and the frozen tail's 0 up to its share t "
          f"({SOURCE}'s t)")


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
            "u": [list(p) for p in formalism.frozen_tail(c["delta"], t)], "decoder_gates": None,
            "labels": LABELS | {"u": U_TAIL if 0 < t < 1 else U_EVEN}}


def analysed(path: Path, scorer: Path, ledger: Path, cache: Path) -> dict[tuple, dict]:
    """(run id, MACs) -> the analysis's formalism block for that checkpoint."""
    mod = load_analysis(path)
    g = mod.gate_factors(cache)
    subs, launch, starts, _night, counted = mod.load(scorer, ledger)
    records, cps = mod.build(subs, launch, starts, counted, g)
    return {(launch[c["id"]]["run"], c["macs"]): block(c, records[c["id"]], mod.TAU) for c in cps}


def attach(rows: list[dict], blocks: dict[tuple, dict]) -> tuple[list[dict], dict[str, str]]:
    """The rows, each one the analysis has a block for with that block as `formalism`; and each row's source."""
    out, source = [], {}
    for r in rows:
        b = blocks.get((r.get("run_id"), r.get("macs")))
        out.append(r | {"formalism": b} if b else r)
        source[r["id"]] = "analysis" if b else "stubs"
    return out, source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="te_core's ledger (default $TE_LEDGER)")
    ap.add_argument("--pin", action="append", default=None, help="a measurement commit, repeated (default pin.py's)")
    ap.add_argument("--facts", type=Path, default=None, help="as te_core.frontier's --facts")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="the analytic v1 quantities (rescore_v1.py)")
    ap.add_argument("--scorer", type=Path, default=SCORER, help="the scorer's directory, which the analysis reads")
    ap.add_argument("--stubs", action="store_true", help="te_core.formalism's stubs for every row, no analysis")
    ap.add_argument("--policy", default=policy.REFERENCE)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--from-scratch", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    a = ap.parse_args(argv)
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
    rep = frontier.formalism_report(rows, pin, shapes, keep)
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
                                "formalism": r.get("formalism")}, default=str) + "\n")
    scored, stubs = plots.load(rows, pin, shapes, keep)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    written = plots.write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop)
    print(frontier.formalism_markdown(rep))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(valid)} valid rows, {len(kept)} scored with {how}"
          + (f", {len(not_analysed)} not analysed (left out, frontier.json's not_analysed)" if use else "")
          + f" -> {a.out}: frontier.md, frontier.json, formalism.jsonl, " + ", ".join(p.name for p in written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
