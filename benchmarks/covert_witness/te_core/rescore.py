"""The night's v2.3 records scored by the training formalism v2, for the scorer, in one command.

    python -m te_core.rescore --out DIR [--root LEDGER] [--pin COMMIT ...] [--facts FACTS.json]
                              [--analysis analysis/rescore_v1.py --scorer ~/.research/te-scorer | --stubs]
                              [--policy reference] [--drop 0.01] [--g G --from-scratch F.json] [--source TEXT]

It reads te_core's ledger (`--root`, default `$TE_LEDGER`), keeps the valid records measured at the pin by
te_core.frontier's rules (by default the pin pin.py reads, v2.3's tonight; `--facts` as for te_core.frontier),
attaches each one's `formalism` block, and scores every checkpoint (te_core.formalism: Fit, Slowdown's bound from A1's
and A1''s upper bounds, and the attack side's best-known values beside it), with H and W* over every checkpoint of every
valid record (the coordinator's 19:50Z rule). It writes into DIR:

- `frontier.md` and `frontier.json`: Fit, Slowdown's ρ and Security's S = W*/H per start, target and policy cell, with
  W*'s route and its x0 and s* bounds and best-known values (te_core.frontier's `formalism_report`);
- the five panels, each `<name>.json` (a panels-route body) and `<name>.png` drawn from it (te_core.plots), and
  `console/<name>.json`, each cut to the route's limits (te_core.console); they are off the console, so nothing is PUT
  (its one training panel is te_core.panels', which reads the ledger through `arguments` and `load`);
- `formalism.jsonl`: each scored row's attached block and its source.

The block holds the per-record quantities of covert_witness's `analysis/rescore_v1.py` where that file is present
(`--analysis`, default benchmarks/covert_witness/analysis/rescore_v1.py; it reads the scorer's directory, `--scorer`,
and the same ledger), joined to a ledger row by the record's run and the checkpoint's MACs; the analysis has no
formula of its own, te_core.formalism is the one:

- one part: the checkpoint's gates (the analysis prices each counted op by its circuit Definition), its tokens
  (forward gates over the start's forward gates per token), its token blocks' gates (forward and backward, so
  σ = their share of |C|), their affected gates, and Δ_grad (the run's optimizer state at its widths);
- τ (one token id), a (d_model x 16), Δ = 8 |E|, and u(1) = Δ;
- Δ_t, the update application's weights (the coordinator's 00:47Z ruling): a straight-through run's fp32 latents, 32
  bits a value as its record counts them, else Δ;
- u and u_in, A1's and A1''s profiles: evenly spread where the run has no frozen tail; with one (its share t above the
  run's highest update, the tied head included), u is 0 up to t and u_in jumps to the full update at once.

u, u_in, u(1), a and Δ are assumption profiles or upper bounds from those witnesses, and their labels say so. Each row
also carries the analysis's `family` (P4's and A2's). With the analysis, a valid row it has no block for (a run outside
the scorer's regime, or submitted after the scorer's snapshot) is left out, listed in `frontier.json` as
`not_analysed`, so no score mixes the two sources. With `--stubs`, or without the analysis in the checkout, every valid
row gets te_core.formalism's stubs from its own fields. Every panel's note and figure's footer names the stubs and
bounds its rows rest on.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import VERSION, console, frontier, plots, policy
from .ledger import Ledger

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis" / "rescore_v1.py"
SCORER = Path.home() / ".research" / "te-scorer"
SOURCE = "analysis (rescore_v1)"
LABELS = {
    "gates": f"{SOURCE}: the ledger's counted ops in gates, each by its circuit Definition",
    "tokens": f"{SOURCE}: forward gates over the start's forward gates per token",
    "token_gates": f"{SOURCE}: the ledger's forward and backward ops in gates (the optimizer in no token block)",
    "affected_gates": f"{SOURCE}: token gates less nu F (nu: the forward's share below the lowest update)",
    "delta_grad_bits": f"{SOURCE}: the gradient and optimizer state at the run's optimizer widths",
    "tau_bits": f"stub: one token id per token block, ceil(log2 V) bits ({SOURCE})",
    "a_bits": f"upper bound (witness): d_model x 16, one token's bf16 residual stream ({SOURCE})",
    "delta_bits": "upper bound (witness): 8 |E|, the submitted update code",
    "delta_t_bits": f"{SOURCE}: a straight-through run's fp32 latents, 32 bits a value as its record counts them; "
                    "every other run's is Delta",
    "u1_bits": f"upper bound (witness): u(1) = Delta, every token's path reads the whole update ({SOURCE})",
    "decoder_gates": "stub: the decoder's gates d unmeasured, taken as 0 (rho 1 where Fit holds)",
}
EVEN = {"u": f"assumption (A1): evenly spread, u(x) = x * u(1) ({SOURCE})",
        "u_in": f"assumption (A1'): evenly spread, u_in = u ({SOURCE})"}
TAIL = {"u": f"assumption (A1): frozen layers above the run's highest update, their share t of a token's affected work "
             f"done reading no update bits, the update even over the rest ({SOURCE}'s t; the tied head is one for a run "
             f"that leaves the embedding frozen)",
        "u_in": f"assumption (A1'): frozen layers above changed ones, u_in jumps to the full update at once ({SOURCE})"}


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
            "tau_bits": tau, "a_bits": r["a"], "delta_bits": c["delta"], "delta_t_bits": r.get("delta_t"),
            "u1_bits": c["delta"], "u": c["u"], "u_in": c["u_in"], "tail_share": t, "decoder_gates": None,
            "labels": LABELS | (TAIL if t else EVEN)}


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


def arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ledger and analysis options `load` reads (te_core.panels takes them too)."""
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None, help="te_core's ledger (default $TE_LEDGER)")
    ap.add_argument("--pin", action="append", default=None, help="a measurement commit, repeated (default pin.py's)")
    ap.add_argument("--facts", type=Path, default=None, help="as te_core.frontier's --facts")
    ap.add_argument("--analysis", type=Path, default=ANALYSIS, help="the per-record quantities (rescore_v1.py)")
    ap.add_argument("--scorer", type=Path, default=SCORER, help="the scorer's directory, which the analysis reads")
    ap.add_argument("--stubs", action="store_true", help="te_core.formalism's stubs for every row, no analysis")
    ap.add_argument("--source", default=None, help="the panels' `source` (default this command and the pin)")
    ap.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    return ap


def load(a) -> dict:
    """The ledger's rows with the analysis's blocks attached (`attach`; the gate factors cached in `a.out`), the pin,
    the starts' shapes, which rows to score (`keep`: those the analysis covers, None for every row) and how the
    quantities were got (`how`)."""
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.facts:
        rows = frontier.with_facts(rows, json.loads(a.facts.read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    use = not a.stubs and a.analysis.exists()
    blocks = analysed(a.analysis, a.scorer, ledger.root, a.out / "analysis-gates.json") if use else {}
    rows, source = attach(rows, blocks)
    how = str(a.analysis) if use else "stubs: " + ("--stubs" if a.stubs else f"no {a.analysis} in this checkout")
    return {"rows": rows, "pin": tuple(a.pin) if a.pin else frontier.pinned(), "shapes": policy.starts(a.manifest),
            "source": source, "keep": (lambda r: source[r["id"]] == "analysis") if use else None, "use": use,
            "how": how}


def main(argv=None) -> int:
    ap = arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--policy", default=policy.REFERENCE)
    ap.add_argument("--drop", type=float, default=0.01)
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--from-scratch", type=Path, default=None)
    a = ap.parse_args(argv)
    d = load(a)
    rows, pin, shapes, source, keep, use, how = (d[k] for k in ("rows", "pin", "shapes", "source", "keep", "use", "how"))
    rep = frontier.formalism_report(rows, pin, shapes, keep)
    left_out = frontier.status(rows, pin)
    valid = [r for r in rows if r["id"] not in left_out]
    kept = [r for r in valid if keep is None or keep(r)]
    not_analysed = sorted(r["id"] for r in valid if keep is not None and not keep(r))
    (a.out / "frontier.json").write_text(json.dumps({"pin": list(pin), "left_out": left_out, "quantities": how,
                                                     "not_analysed": not_analysed} | rep, indent=1, default=str) + "\n")
    (a.out / "frontier.md").write_text(frontier.formalism_markdown(rep) + "\n")
    with (a.out / "formalism.jsonl").open("w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "run_id": r.get("run_id"), "start": r["start"],
                                "checkpoint_step": r.get("checkpoint_step"), "source": source[r["id"]],
                                "family": r.get("family"), "formalism": r.get("formalism")}, default=str) + "\n")
    scored, stubs = plots.load(rows, pin, shapes, keep)
    fs = json.loads(a.from_scratch.read_text()) if a.from_scratch else {}
    what = a.source or (f"python -m te_core.rescore (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                        f"{len(kept)} valid checkpoints, quantities {'rescore_v1' if use else 'stubs'})")
    written = plots.write(scored, shapes, stubs, pin, a.out, a.policy, fs, a.g, a.drop, what)
    console.write(a.out, a.out / "console")
    print(frontier.formalism_markdown(rep))
    print(f"pin {' + '.join(c[:12] for c in pin)}: {len(valid)} valid rows, {len(kept)} scored with {how}"
          + (f", {len(not_analysed)} not analysed (left out, frontier.json's not_analysed)" if use else "")
          + f" -> {a.out}: frontier.md, frontier.json, formalism.jsonl, "
          + ", ".join(f"{p.stem}.json/.png" for p in written) + ", console/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
