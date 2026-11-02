"""P4 with one row per valid run (coordinator 20:52Z), from te_core's rows, for the final table (v1cycle's final_md).

    python v1runs.py --out DIR --scorer STATE [--root LEDGER] [--pin COMMIT ...] [--final FINAL_DIR] [--prefix P]

Run from benchmarks/covert_witness of te_core's checkout (v1cycle's RESCORE_WT), with DIR the directory
`python -m te_core.rescore` just wrote (its analysis-gates.json cache is reused). te_core.rescore's P4 is one point per
algorithm; this one has one row per run and target, at the run's least-compute checkpoint reaching the target, with
`variant` (the run's launch note). Every number is te_core's: the rows are te_core.plots.load's (rescore_v1.py's
quantities attached by te_core.rescore, Fit and rho by te_core.formalism with the frozen tail charged, H by
te_core.frontier.target), selected here, not re-scored. With --final, the rows carry FINAL_DIR's final values as
v1final.py gives them to te_core.rescore. Writes DIR/<P>p4-decomposition.json (P default runs-), a whole panel in the
route's form; nothing PUTs it.
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
HEAD = ["start", "start_id", "size", "saturation", "target_pct", "run", "method", "variant", "family", "record",
        "checkpoint", "checkpoint_frac"]
SIZES = ["delta_bits", "delta_grad_bits", "compute_gates", "compute_frac", "fit_run_k_bits", "fit_train_k_bits",
         "sigma"]


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


def p4_runs(sts: list[plots.Start], var: dict, ctx: dict) -> dict:
    rows = []
    for st in sts:
        c_ref = st.cells[st.ref][0]
        size, sat = st.shape.get("parameters"), plots.saturation(st.shape)
        for tp, g, l_star in st.levels(None):
            h = frontier.target(st.on, l_star, [], size="gates")["honest"]
            for r in least_per_run(st, l_star):
                m, s = r["measured"], r["rho"][st.ref]
                run_k, train_k = formalism.fit_k(m, c_ref, st.w)
                rows.append([*st.head, size, sat, tp, r["run_id"], plots.method(r), var.get(r["run_id"]),
                             r.get("family"), r["id"], r.get("checkpoint_step"), r.get("fraction"),
                             g, r["gates"] / h if h else None, s, r["gates"] * s / h if h else None,
                             plots.basis(r, st.ref), len(r["seeds"]) if r.get("seeds") else 1,
                             m.delta_bits, m.delta_grad_bits, r["gates"], st.frac(r["gates"]), run_k, train_k, m.sigma,
                             *plots.terms(r, st.ref)])
    return plots.panel(
        "p4-decomposition", "p against s per run at the reference policy",
        HEAD + ["g", "p", "s", "S", "basis", "seeds"] + SIZES + list(frontier.TERMS), rows,
        "Per start and target, one row per valid run at its least-compute checkpoint reaching the target (over every "
        "checkpoint of every valid record, 19:50Z); variant = the run's launch note; p = |C|/H with H over every "
        "checkpoint of every valid record, s = rho at the reference policy (c/(c - d) by Fit, else Slowdown's bound "
        "floored at 1), S = p s; seeds = the point's; fit_*_k_bits: the least k where Fit holds at the reference c.",
        ctx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None)
    ap.add_argument("--source", default="")
    ap.add_argument("--final", type=Path, default=None, help="score with FINAL_DIR's final values (v1final.final_rows)")
    ap.add_argument("--prefix", default="runs-", help="the written file's prefix")
    a = ap.parse_args(argv)
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.final:
        from v1final import final_rows
        rows = final_rows(rows, a.final, a.scorer)[0]
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    shapes = policy.starts()
    blocks = rescore.analysed(rescore.ANALYSIS, a.scorer, ledger.root, a.out / "analysis-gates.json")
    rows, source = rescore.attach(rows, blocks)
    scored, stubs = plots.load(rows, pin, shapes, lambda r: source[r["id"]] == "analysis")
    sts = plots.starts_of(scored, shapes, policy.REFERENCE, {})
    tail = (" The frozen tail's work always charged (Daniel, 22:00Z). basis: fit = rho c/(c - d) by Fit for the "
            "training; target = Slowdown's bound (stated, not proved in Lean). Pin " + " + ".join(c[:12] for c in pin)
            + ". Stubs and bounds: " + "; ".join(f"{s} ({n} rows)" for s, n in sorted(stubs.items())) + ".")
    ctx = {"stamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": a.source, "tail": tail}
    p = p4_runs(sts, variants(a.scorer), ctx)
    (a.out / f"{a.prefix}p4-decomposition.json").write_text(json.dumps(p, allow_nan=False) + "\n")
    print(f"p4-decomposition: {len(p['rows'])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
