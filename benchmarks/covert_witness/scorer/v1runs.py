"""The console's A2 and P4 with one row per valid run (coordinator 20:52Z, the dashboard's request), from te_core's rows.

    python v1runs.py --out DIR --scorer STATE [--root LEDGER] [--pin COMMIT ...]

Run from benchmarks/covert_witness of te_core's checkout (v1cycle's RESCORE_WT), with DIR the directory
`python -m te_core.rescore` just wrote (its analysis-gates.json cache is reused). te_core.rescore's A2 is the floor's
staircase and its P4 one point per algorithm; the console draws the floor itself, as the frontier of every valid run,
so these carry one row per run, target and reading, at the run's least-compute checkpoint reaching the target, with
`variant` (the run's launch note) so a method's runs join. Every number is te_core's: the rows are te_core.plots.load's
(rescore_v1.py's quantities attached by te_core.rescore, Fit and rho under both readings by te_core.formalism, H by
te_core.frontier.target), selected here, not re-scored. Writes DIR/runs-a2-efficiency-floor.json and
DIR/runs-p4-decomposition.json, each a whole panel in the route's form.
"""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from te_core import formalism, frontier, plots, policy, rescore
from te_core.ledger import Ledger

VARIANT_MAX = 200
HEAD = ["start", "start_id", "size", "saturation", "target_pct", "reading", "run", "method", "variant", "family",
        "record", "checkpoint", "checkpoint_frac"]
SIZES = ["delta_bits", "delta_grad_bits", "compute_gates", "compute_frac", "fit_run_k_bits", "fit_train_k_bits"]


def variants(state: Path) -> dict[str, str | None]:
    out = {}
    for line in (state / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py" and x.get("run"):
            out[x["run"]] = (x.get("note") or "")[:VARIANT_MAX] or None
    return out


def least_per_run(st: plots.Start, l_star: float) -> list[dict]:
    """Each run's least-compute checkpoint at or below l_star."""
    best: dict[str, dict] = {}
    for r in st.on:
        if r.get("val_bpb") is not None and r["val_bpb"] <= l_star:
            b = best.get(r["run_id"])
            if b is None or r["gates"] < b["gates"]:
                best[r["run_id"]] = r
    return sorted(best.values(), key=lambda r: r["gates"])


def panels(sts: list[plots.Start], var: dict, readings, ctx: dict) -> dict[str, dict]:
    a2, p4 = [], []
    for st in sts:
        c_ref = st.cells[st.ref][0]
        size, sat = st.shape.get("parameters"), plots.saturation(st.shape)
        for tp, g, l_star in st.levels(None):
            h = frontier.target(st.on, l_star, [], size="gates")["honest"]
            for r in least_per_run(st, l_star):
                m = r["measured"]
                run_k, train_k = formalism.fit_k(m, c_ref)
                sizes = [m.delta_bits, m.delta_grad_bits, r["gates"], st.frac(r["gates"]), run_k, train_k]
                for name in readings:
                    head = [*st.head, size, sat, tp, name, r["run_id"], plots.method(r), var.get(r["run_id"]),
                            r.get("family"), r["id"], r.get("checkpoint_step"), r.get("fraction")]
                    a2.append(head + sizes)
                    s = r["rho_by"][name][st.ref]
                    p4.append(head + [g, r["gates"] / h if h else None, s, r["gates"] * s / h if h else None,
                                      plots.basis(r, st.ref), 1] + sizes)
    every = ("one row per valid run, target and reading, at the run's least-compute checkpoint reaching the target "
             "(over every checkpoint of every valid record, 19:50Z); variant = the run's launch note")
    return {
        "a2-efficiency-floor": plots.panel(
            "a2-efficiency-floor", "Efficiency floor: least compute to each target against the update's size, per run",
            HEAD + SIZES, a2,
            f"Per start and target, {every}; the floor is these rows' frontier in (delta_bits, compute_gates). "
            "delta_bits = Delta (Fit's quantity), delta_grad_bits = the gradient and optimizer state passed between "
            "blocks, compute in Boolean gates, checkpoint_frac = the checkpoint's share of its run. fit_run_k_bits and "
            "fit_train_k_bits: the least k at which Fit holds at the reference c, for running the updated model and "
            "for the training. A2 reads no rho: its readings' rows agree.", ctx),
        "p4-decomposition": plots.panel(
            "p4-decomposition", "p against s per run at the reference policy", HEAD + ["g", "p", "s", "S", "basis",
                                                                                        "seeds"] + SIZES, p4,
            f"Per start and target, {every}; p = |C|/H with H over every checkpoint of every valid record, s = rho at "
            "the reference policy (1 by Fit, else Slowdown's bound floored at 1), S = p s. One seed per point.", ctx)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None)
    ap.add_argument("--source", default="")
    a = ap.parse_args(argv)
    ledger = Ledger(a.root)
    rows = ledger.index()
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    shapes = policy.starts()
    blocks = rescore.analysed(rescore.ANALYSIS, a.scorer, ledger.root, a.out / "analysis-gates.json")
    rows, source = rescore.attach(rows, blocks)
    scored, stubs = plots.load(rows, pin, shapes, lambda r: source[r["id"]] == "analysis", formalism.READINGS)
    sts = plots.starts_of(scored, shapes, policy.REFERENCE, {})
    tail = (f" Readings: {', '.join(formalism.READINGS)} (Slowdown's frozen-tail case; Daniel decides which the theorem "
            "means, so both are listed and neither is the headline). basis: fit = rho 1 by Fit for the training; target "
            "= Slowdown's bound (stated, not proved in Lean). Pin " + " + ".join(c[:12] for c in pin) + ". Stubs and "
            "bounds: " + "; ".join(f"{s} ({n} rows)" for s, n in sorted(stubs.items())) + ".")
    ctx = {"stamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": a.source, "tail": tail}
    for name, p in panels(sts, variants(a.scorer), formalism.READINGS, ctx).items():
        (a.out / f"runs-{name}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
        print(f"{name}: {len(p['rows'])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
