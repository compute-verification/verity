"""The console's A2 and P4 with one row per valid run (coordinator 20:52Z, the dashboard's request), from te_core's rows.

    python v1runs.py --out DIR --scorer STATE [--root LEDGER] [--pin COMMIT ...] [--final FINAL_DIR] [--prefix P]

Run from benchmarks/covert_witness of te_core's checkout (v1cycle's RESCORE_WT), with DIR the directory
`python -m te_core.rescore` just wrote (its analysis-gates.json cache is reused). te_core.rescore's A2 is the floor's
staircase and its P4 one point per algorithm; the console draws the floor itself, as the frontier of every valid run,
so these carry one row per run, target and reading, at the run's least-compute checkpoint reaching the target, with
`variant` (the run's launch note) so a method's runs join. Every number is te_core's: the rows are te_core.plots.load's
(rescore_v1.py's quantities attached by te_core.rescore, Fit and rho under both readings by te_core.formalism, H by
te_core.frontier.target), selected here, not re-scored. With --final, the rows carry FINAL_DIR's final values as
v1final.py gives them to te_core.rescore. Writes DIR/<P>a2-efficiency-floor.json and DIR/<P>p4-decomposition.json (P
default runs-), each a whole panel in the route's form.
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
BRANCHES = ["curve", "start", "start_compute_gates", "start_loss", "c_tokens_per_block", "k_bits", "compute_total_gates",
            "compute_after_gates", "loss", "S", "run", "method", "delta_bits", "basis", "start_id", "policy", "record",
            "checkpoint"]


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
                                      plots.basis(r, st.ref), len(r["seeds"]) if r.get("seeds") else 1] + sizes)
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
            "the reference policy (1 by Fit, else Slowdown's bound floored at 1), S = p s; seeds = the point's.", ctx)}


def branches(sts: list[plots.Start], name: str, ctx: dict) -> dict:
    """Daniel's headline panel (note:training-20261102T2226Z-draft-frontier-branches-panel) under reading `name`: per
    start, its branch point (C0, l0), the honest staircase after it and the throttled one (|C| rho) at every (c, k) of
    te_core.policy.cells, S = W*/H at each throttled point's loss against the same start's honest staircase; the
    from-scratch frontier is the starts' own points until the ladder's curve comes."""
    rows = []
    for x, y, st in plots.staircase([(st.c0, st.l0, st) for st in sts]):
        rows.append(["frontier", "", None, None, None, None, x, None, y, None, st.name, "the start's own training",
                     None, None, st.name, None, None, None])
    for st in sts:
        def row(curve, x, y, r, key=None, s=None, st=st):
            c, k = st.cells[key] if key else (None, None)
            return [curve, st.head[0], st.c0, st.l0, c and c / st.w, k, st.c0 + x, x, y, s, r["run_id"],
                    plots.method(r), r["measured"].delta_bits, plots.basis(r, key) if key else None, st.name,
                    plots.policy_label(key) if key else None, r["id"], r.get("checkpoint_step")]
        honest = plots.staircase([(r["gates"], r["val_bpb"], r) for r in st.on])
        rows += [row("honest", x, y, r) for x, y, r in honest]
        for key in st.cells:
            pts = [(r["gates"] * r["rho_by"][name][key], r["val_bpb"], r) for r in st.on
                   if r["rho_by"][name].get(key) is not None]
            for x, y, r in plots.staircase(pts):
                h = min((hx for hx, hy, _ in honest if hy <= y), default=None)
                rows.append(row("throttled", x, y, r, key, x / h if h else None))
    how = {policy.c0(st.shape)[1] for st in sts}
    return plots.panel(
        "frontier-branches", "The frontier and its branches: loss against total training compute", BRANCHES, rows,
        "Every curve a staircase (the least compute reaching each loss) over every checkpoint of every valid record "
        "(19:50Z). frontier = from scratch: the starts' own points until the ladder's curve comes; honest = |C| after "
        "the start; throttled = |C| rho at each (c, k) (c in tokens per block of the start's forward, k in bits); "
        "compute_total = C0 + compute after; S = W*/H at the point's loss against the same start's honest. C0: "
        + "; ".join(sorted(how)) + ".", ctx)


def draw_branches(p: dict, out: Path, ref: str = policy.REFERENCE) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = plots.frame(p)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    starts = sorted({r["start_id"] for r in d if r["curve"] != "frontier"})
    for ax, xk, title in ((axes[0], "compute_total_gates", "total training compute (C0 + after), gates"),
                          (axes[1], "compute_after_gates", "compute after the start, gates")):
        fr = [r for r in d if r["curve"] == "frontier"]
        if xk == "compute_total_gates" and fr:
            ax.plot([r[xk] for r in fr], [r["loss"] for r in fr], "--", color="C0", lw=1.5, label="frontier (starts)")
        for i, s in enumerate(starts):
            h = [r for r in d if r["start_id"] == s and r["curve"] == "honest"]
            t = [r for r in d if r["start_id"] == s and r["curve"] == "throttled" and r["policy"] == plots.policy_label(ref)]
            lab = h[0]["start"] if h else s
            if xk == "compute_total_gates" and h:
                ax.plot([h[0]["start_compute_gates"]], [h[0]["start_loss"]], "o", color="k", ms=6)
                ax.annotate(f"{s} ({lab})", (h[0]["start_compute_gates"], h[0]["start_loss"]), fontsize=7,
                            xytext=(4, 4), textcoords="offset points")
            for rows, color, lw, what in ((h, "C0", 0.9, "honest"), (t, "C3", 2.2, "throttled")):
                if rows:
                    ax.step([r[xk] for r in rows], [r["loss"] for r in rows], where="post", color=color, lw=lw,
                            alpha=0.6 + 0.4 * (i == 0), label=f"{what} {s}")
        ax.set_xscale("log")
        ax.set_xlabel(title)
        ax.set_ylabel("held-out bits per byte")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"{p['title']} (throttled at {plots.policy_label(ref)}; every (c, k) in the panel)", fontsize=10)
    fig.text(0.01, 0.005, p["note"][:300], fontsize=5, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scorer", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--pin", action="append", default=None)
    ap.add_argument("--source", default="")
    ap.add_argument("--final", type=Path, default=None, help="score with FINAL_DIR's final values (v1final.final_rows)")
    ap.add_argument("--prefix", default="runs-", help="the written files' prefix")
    ap.add_argument("--reading", action="append", choices=formalism.READINGS, default=None,
                    help="a reading of Slowdown's frozen-tail case, repeated (default te_core's)")
    a = ap.parse_args(argv)
    readings = tuple(a.reading or formalism.READINGS)
    ledger = Ledger(a.root)
    rows = ledger.index()
    if a.final:
        from v1final import final_rows
        rows = final_rows(rows, a.final, a.scorer)[0]
    pin = tuple(a.pin) if a.pin else frontier.pinned()
    shapes = policy.starts()
    blocks = rescore.analysed(rescore.ANALYSIS, a.scorer, ledger.root, a.out / "analysis-gates.json")
    rows, source = rescore.attach(rows, blocks)
    scored, stubs = plots.load(rows, pin, shapes, lambda r: source[r["id"]] == "analysis", readings)
    sts = plots.starts_of(scored, shapes, policy.REFERENCE, {})
    which = (" Reading: tail-charged, the frozen tail's work always charged (Daniel's ruling, coordinator 22:00Z)."
             if readings == ("tail-charged",) else
             f" Readings: {', '.join(readings)} (Slowdown's frozen-tail case; neither is the headline).")
    tail = (which + " basis: fit = rho 1 by Fit for the training; target = Slowdown's bound (stated, not proved in "
            "Lean). Pin " + " + ".join(c[:12] for c in pin) + ". Stubs and bounds: "
            + "; ".join(f"{s} ({n} rows)" for s, n in sorted(stubs.items())) + ".")
    ctx = {"stamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": a.source, "tail": tail}
    out = panels(sts, variants(a.scorer), readings, ctx) | {"frontier-branches": branches(sts, readings[0], ctx)}
    for name, p in out.items():
        (a.out / f"{a.prefix}{name}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
        print(f"{name}: {len(p['rows'])} rows")
    draw_branches(out["frontier-branches"], a.out / f"{a.prefix}frontier-branches.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
