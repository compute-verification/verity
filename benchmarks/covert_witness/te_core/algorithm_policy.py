"""The console's second training panel, `training/algorithm-policy`: each run's efficiency loss from its algorithm
against its efficiency loss from the policy, as Daniel asked for it.

    python -m te_core.algorithm_policy --out DIR [--publish] [--ladder-fit FILE]
                                       [te_core.rescore's ledger and analysis options]

It reads what te_core.panels' frontier-branches reads: the valid records of the pin and of the pilot (pin.py's `pilot`)
unless `--pin` names the commits, each checkpoint on its data (`panels.data_of`: `v3`, harness v3's OLMo 2 mix, or
`v2.3-pilot`; a loss on one is no loss on the other), scored where the scoring cap admits it (te_core.policy's
`eligible`). At an end loss l, one of te_core.frontier's targets (a start's l0 on that data less 0.5%, 1%, 2% and 5%),
H is the least compute after the start of any valid checkpoint from it on the same data that reaches l (the honest
envelope te_core.panels draws). A run that reaches l does so first at a checkpoint of compute C; its algorithm
inefficiency is p = C/H >= 1, its policy slowdown there ρ (te_core.formalism.at at the dial's (c, k): c/(c − d) where
Fit holds for the training, basis `fit`, else Slowdown's bound, a target, basis `target`), and its constrained cost
over H is S = p·ρ.

The panel (`COLUMNS`) has one row per run (its run id) and end loss it reaches, taken at that first checkpoint: the
start (te_core.panels' label, C0 and l0), the end loss, C and p, the inputs ρ is computed from there (`INPUTS`, as
the training notes' draft on ρ's inputs and formula has them), which of those are stubs, bounds, assumptions or
estimates rather than measured (`estimated`, from the record's labels: `KINDS`), ρ and its basis at the dial's default
(te_core.panels' `DEFAULT`: c = 4,096 tokens of the start's forward, k = 80,000 bits), and the run's method, family,
record (run id@step) and data. The route takes at most 24 columns, so H (C/p) and the target (`target`, from l0 and
the end loss) aren't columns of their own. `page_rho` is the formula the console's dial runs on a row's columns alone,
and `build` refuses a row whose columns don't reproduce te_core.formalism.at's ρ and basis at the default.

Its console copy (`cut`) is the whole table when it fits the panels route (te_core.console), else, per data, start,
target and family, the rows on the lower-left (p, rho_default) front plus the family's least-p and least-delta rows
(the headline data's alone when every data's don't fit), with the full table's art id in the note; the ρ inputs keep
6 significant digits so the dial reproduces rho_default.
DIR gets `algorithm-policy.json` (uncut), `console/algorithm-policy.json` (the cut) and `algorithm-policy.png` (p
against ρ at the default dial from the uncut rows on the headline data, te_core.panels': v3 once it has rows, else the
pilot; a panel per target, with iso-S diagonals). `--publish` puts the uncut table in the evidence store when the cut
leaves rows out, then PUTs the cut.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import VERSION, console, formalism, frontier, panels, plots, policy, rescore

NAME = "algorithm-policy"
PANEL = f"training/{NAME}"
#: the inputs of ρ at any (c, k), each at the run's checkpoint (the draft note's)
INPUTS = ["w_gates", "w_fwd_gates", "sigma", "tau_bits", "a_bits", "u_points", "u_in_points", "delta_bits",
          "delta_t_bits", "d_gates", "delta_grad_bits"]
COLUMNS = ["start", "start_compute_gates", "start_loss", "end_loss", "C_gates", "p", *INPUTS, "estimated",
           "rho_default", "basis_default", "method", "family", "record", "data"]
LOSSES = ("start_loss", "end_loss")
#: significant digits of the ρ inputs in the console's copy, and of its other numbers
INPUT_DIGITS, DIGITS = 6, 4
#: each column whose value rests on a record's labelled quantities, and those quantities (te_core.formalism's names), the
#: column's own first
LABELLED = {"C_gates": ("gates",), "w_gates": ("tokens", "token_gates"), "sigma": ("token_gates", "gates"),
            "tau_bits": ("tau_bits",), "a_bits": ("a_bits",), "u_points": ("u", "u1_bits"),
            "u_in_points": ("u_in", "u1_bits"), "delta_bits": ("delta_bits",), "delta_t_bits": ("delta_t_bits",),
            "d_gates": ("decoder_gates",), "delta_grad_bits": ("delta_grad_bits",)}
#: a label's first words, and the tag `estimated` gives it; any other label not measured or counted is "est"
KINDS = (("stub", "stub"), ("upper bound", "bound"), ("assumption (A1')", "A1'"), ("assumption (A1)", "A1"),
         ("assumption", "asm"), ("unmeasured", "none"))
#: te_core.rescore's labels of quantities the ledger counted (its op counts priced in gates)
COUNTED = (rescore.LABELS["gates"], rescore.LABELS["token_gates"])
LEVELS = plots.LEVELS
MARKERS = ("o", "s", "^", "D", "v", "P")


def default(w_fwd: float) -> tuple[float, float]:
    """The dial's default (c, k) for a start whose forward is `w_fwd` gates a token (te_core.panels' `DEFAULT`, tokens
    per block and bits)."""
    return panels.DEFAULT[0] * w_fwd, float(panels.DEFAULT[1])


def target(start_loss: float, end_loss: float) -> float:
    """The row's end loss as te_core.frontier's target it is, in percent of l0 (-0.5, -1, -2, -5)."""
    return -100 * min(frontier.TARGETS, key=lambda t: abs(end_loss - start_loss * (1 - t)))


def num(v: float, digits: int = INPUT_DIGITS) -> str:
    """`v` to `digits` significant digits, compact: 24601880 -> 2.46019e7, 0.25 -> 0.25."""
    return f"{v:.{digits}g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def points(u) -> str:
    """A profile's points as the console's string, "x:bits;..." ("0:0;0.25:1.2e6;1:4.8e6")."""
    return ";".join(f"{num(x)}:{num(b)}" for x, b in u)


def profile(text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(b)) for x, b in (p.split(":") for p in text.split(";")))


def kind(label: str | None) -> str | None:
    """What a labelled quantity is: None when measured or counted, else `KINDS`' tag, or "est" (an estimate: derived
    by the analysis, not measured) for any other label and for none."""
    if label is not None and (label.startswith(("measured", "counted")) or label in COUNTED):
        return None
    return next((tag for start, tag in KINDS if label is not None and label.startswith(start)), "est")


def estimated(m: formalism.Measured) -> str:
    """The columns whose values are stubs, bounds, assumptions or estimates rather than measured, each with its tag
    ("tau_bits:stub a_bits:bound ..."). Δ_t that is Δ (no latents) is Δ's."""
    out = []
    for col, keys in LABELLED.items():
        if col == "delta_t_bits" and m.delta_t_bits is None:
            keys = LABELLED["delta_bits"]
        tag = next((t for t in (kind(m.labels.get(k)) for k in keys) if t), None)
        if tag:
            out.append(f"{col}:{tag}")
    return " ".join(out)


def inputs(m: formalism.Measured, w_fwd: float) -> list:
    """`INPUTS` for a checkpoint's quantities: d None while unmeasured (no block fits, as `formalism.at` reads it), u
    and u_in capped at Δ (te_core.formalism's `capped`) where Δ counts at every c (d = 0; `page_rho` caps them wherever
    d < c), and Δ_grad the one Fit binds on (`Measured.fit_delta_grad_bits`)."""
    u, u_in = (formalism.capped(m.u, m.delta_bits), formalism.capped(m.u_in, m.delta_bits)) \
        if m.delta_bits is not None and m.decoder_gates == 0 else (m.u, m.u_in)
    return [m.w, w_fwd, m.sigma, m.tau_bits, m.a_bits, points(u), points(u_in), m.delta_bits, m.delta_t,
            m.decoder_gates, m.fit_delta_grad_bits]


def page_rho(d: dict, c: float, k: float) -> tuple[float, str]:
    """ρ and its basis at (c, k) from a row's columns alone, the formula the console's dial runs (the draft note's,
    with te_core 4.x's two terms where a row has them, te_core.panels' candidates: `w_fit_gates`, the least w of a
    part, which Fit's compute block binds on, else w; and `alpha_ratio`, ᾱ/α_max, else 1): Fit for the training,
    delta + (c/w_fit)·tau <= k, 2·delta_grad <= k and delta_t + delta_grad <= k (a null delta_grad fails), with
    d < c (a null d, unmeasured, fails), gives c/(c − d); else s = c/w, x0 = the largest x with u_in(x) <= k,
    s* = max_x x(k − u(x))/a and ρ = max(1, sigma·(1 − x0)·s·alpha_ratio/s*), infinite where s* is 0 and the rest
    isn't; where d < c and delta is known, u and u_in are at most delta."""
    s, dg = c / d["w_gates"], d["delta_grad_bits"]
    s_fit = c / d.get("w_fit_gates", d["w_gates"])
    counts = d["d_gates"] is not None and d["d_gates"] < c and d["delta_bits"] is not None
    if counts and d["delta_bits"] + s_fit * d["tau_bits"] <= k \
            and dg is not None and 2 * dg <= k and d["delta_t_bits"] + dg <= k:
        return c / (c - d["d_gates"]), "fit"
    u, u_in = profile(d["u_points"]), profile(d["u_in_points"])
    if counts:
        u, u_in = formalism.capped(u, d["delta_bits"]), formalism.capped(u_in, d["delta_bits"])
    x0 = formalism.bottom_share(u_in, k)
    s_star = formalism.room(u, k) / d["a_bits"]
    lead = d["sigma"] * (1 - x0) * s * d.get("alpha_ratio", 1.0)
    return max(1.0, lead / s_star if s_star > 0 else math.inf if lead else 0.0), "target"


def build(scored: list[dict], shapes: dict[str, dict], source: str, stamp: str | None = None) -> dict:
    """The panel, uncut: one row per run and end loss it reaches, per data and start (module docstring)."""
    rate, _ = panels.dense_rate(scored, shapes)
    rows, wrong = [], []
    for data, name in sorted({(panels.data_of(r), r["start"]) for r in scored},
                             key=lambda dn: (panels.DATA.index(dn[0]), dn[1])):
        on = [r for r in scored if (panels.data_of(r), r["start"]) == (data, name)]
        l0 = frontier.start_bpb(on)
        if l0 is None:
            continue
        shape = shapes[name]
        w_fwd = policy.w_fwd(shape)
        head = [plots.start_label(shape), panels.c0(shape, rate), l0]
        honest = plots.staircase([(r["gates"], r["val_bpb"], r) for r in on])
        runs: dict[str, list[dict]] = {}
        for r in sorted(on, key=lambda r: (r.get("checkpoint_step") or 0, r["gates"])):
            runs.setdefault(r.get("run_id") or r["id"], []).append(r)
        for t in frontier.TARGETS:
            end = l0 * (1 - t)
            h = panels.least(honest, end)
            got = []
            for cps in runs.values():
                r = next((r for r in cps if r["val_bpb"] <= end), None)
                if r is None:
                    continue
                m = r["measured"]
                v = formalism.at(m, *default(w_fwd))
                basis = "fit" if v["fit"] else "target"
                row = [*head, end, r["gates"], r["gates"] / h, *inputs(m, w_fwd), estimated(m),
                       None if math.isinf(v["rho"]) else v["rho"], basis, plots.method(r), r.get("family"),
                       f"{r.get('run_id') or r['id']}@{r.get('checkpoint_step')}", data]
                again, b = page_rho(dict(zip(COLUMNS, row)), *default(w_fwd))
                if b != basis or not math.isclose(again, v["rho"], rel_tol=1e-4):
                    wrong.append(f"{row[-2]} at {end:.4f} (te_core {v['rho']:.6g} {basis}, columns {again:.6g} {b})")
                got.append(row)
            rows += sorted(got, key=lambda x: (x[COLUMNS.index("p")], x[COLUMNS.index("rho_default")] or math.inf))
    if wrong:
        raise ValueError(f"{PANEL}: {len(wrong)} rows' columns don't give te_core.formalism.at's rho at the default "
                         f"({'; '.join(wrong[:3])}): a record with several parts needs per-part columns")
    return {"id": PANEL, "title": "Algorithm against policy: each run's p = C/H against its slowdown rho",
            "kind": "table", "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source,
            "note": note(), "columns": COLUMNS, "rows": rows}


def note(art: str | None = None, cut: str | None = None) -> str:
    """The panel's note; `cut` names what a cut keeps (None: no cut; "every": every data's; else that data's)."""
    out = ("Per run and end loss, its first checkpoint reaching it: p = C/H, H the least C of any checkpoint from the "
           "start on its data reaching it; rho by te_core.formalism.at (fit: c/(c-d); target: Slowdown's bound); "
           f"S = p rho. *_default: c = {panels.DEFAULT[0]} w_fwd, k = {panels.DEFAULT[1]:,} bits; formula "
           "note:training-20261102T2334Z-draft-rho-inputs-and-formula. One seed.")
    if cut:
        out += (" Cut " + ("per data, " if cut == "every" else f"to {cut}, per ")
                + "start, target, family: (p, rho) front, least p, least delta; full table "
                + (art or f"{NAME}.json (unpublished)") + ".")
    return out


def headline(p: dict) -> str:
    """The data the figure shows, and the one a cut keeps when every data's doesn't fit (te_core.panels' headline): v3
    once the table has v3 rows, else the pilot."""
    have = {r[COLUMNS.index("data")] for r in p["rows"]}
    return next((d for d in panels.DATA if d in have), panels.DATA[-1])


def rounded(row: list) -> list:
    def val(c, v):
        if not isinstance(v, float):
            return v
        if c in LOSSES:
            return round(v, 5)
        return float(num(v, INPUT_DIGITS if c in INPUTS else DIGITS))
    return [val(c, v) for c, v in zip(COLUMNS, row)]


def kept(rows: list[list]) -> list[list]:
    """Per data, start, target and family, the rows on the lower-left (p, rho_default) front (te_core.plots' staircase)
    and the family's least-p and least-delta rows, in the table's order."""
    ix = {c: i for i, c in enumerate(COLUMNS)}
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (r[ix["data"]], r[ix["start"]], target(r[ix["start_loss"]], r[ix["end_loss"]]), r[ix["family"]])
        groups.setdefault(key, []).append(i)
    keep = set()
    for g in groups.values():
        rho = {i: rows[i][ix["rho_default"]] if rows[i][ix["rho_default"]] is not None else math.inf for i in g}
        keep |= {i for _, _, i in plots.staircase([(rows[i][ix["p"]], rho[i], i) for i in g])}
        keep.add(min(g, key=lambda i: (rows[i][ix["p"]], rho[i])))
        deltas = [i for i in g if rows[i][ix["delta_bits"]] is not None]
        if deltas:
            keep.add(min(deltas, key=lambda i: (rows[i][ix["delta_bits"]], rows[i][ix["p"]])))
    return [r for i, r in enumerate(rows) if i in keep]


def checked(q: dict) -> dict:
    if len(q["note"]) > console.NOTE_MAX:
        raise ValueError(f"{PANEL}: its note is {len(q['note'])} characters, over {console.NOTE_MAX}")
    if bad := console.refused(q):
        raise ValueError(f"{PANEL}: the panels route refuses it ({'; '.join(bad[:3])}; te_core.console.refused): at "
                         f"most {console.COLUMNS_MAX} columns, each cell a number, a string of at most "
                         f"{console.CELL_MAX} characters or null")
    return q


def whole(p: dict) -> dict:
    """`p` rounded, every row, if its body fits the panels route, else None."""
    q = checked(p | {"rows": [rounded(r) for r in p["rows"]], "note": note()})
    return q if len(console.body(q)) <= console.BODY_MAX else None


def cut(p: dict, art: str | None = None) -> dict:
    """The console's copy: `whole` when it fits, else the rows `kept` keeps, every data's when they fit and else the
    headline data's, with the full table's art id in the note."""
    if q := whole(p):
        return q
    rows = kept(p["rows"])
    q = checked(p | {"rows": [rounded(r) for r in rows], "note": note(art, cut="every")})
    if len(console.body(q)) > console.BODY_MAX and len({r[COLUMNS.index("data")] for r in rows}) > 1:
        data = headline(p)
        q = checked(p | {"rows": [rounded(r) for r in rows if r[COLUMNS.index("data")] == data],
                         "note": note(art, cut=data)})
    if len(console.body(q)) > console.BODY_MAX:
        raise ValueError(f"{PANEL}: the cut is {len(console.body(q))} bytes, over {console.BODY_MAX}: keep fewer rows "
                         "per family (te_core.algorithm_policy.kept)")
    return q


def store(path: Path, meta: dict) -> str:
    """`path` in the evidence store, preserved (`research data put`); its art id."""
    exe = shutil.which("research") or str(Path(sys.executable).with_name("research"))
    out = subprocess.run([exe, "data", "put", "--kind", "evidence/v1", "--file", str(path), "--preserve", "--json",
                          "--meta", json.dumps(meta)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)["id"]


def finite(p: dict) -> list[dict]:
    """`p`'s rows with a finite rho_default, as dicts, each with its `target_pct`."""
    return [d | {"target_pct": target(d["start_loss"], d["end_loss"])} for d in plots.frame(p)
            if d["rho_default"] is not None]


def figure(p: dict, out: Path) -> Path:
    """p against ρ at the default dial on the headline data, log-log, one panel per target, a colour per family and a
    marker per start, with the iso-S diagonals S = p ρ at `LEVELS`."""
    data = headline(p)
    rows = [d for d in finite(p) if d["data"] == data]
    families = sorted({d["family"] or "?" for d in rows})
    starts = list(dict.fromkeys(d["start"] for d in rows))
    targets = [-100 * t for t in frontier.TARGETS]
    xhi = 10 ** math.ceil(math.log10(1.5 * max([d["p"] for d in rows] + [5.0])))
    yhi = 10 ** math.ceil(math.log10(1.5 * max([d["rho_default"] for d in rows] + [float(max(LEVELS))])))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    xs = [10 ** (i / 20) for i in range(-2, int(20 * math.log10(xhi)) + 1)]
    for ax, tp in zip(axes.flat, targets):
        mine = [d for d in rows if d["target_pct"] == tp]
        for s in LEVELS:
            ax.plot(xs, [s / x for x in xs], color="0.75", lw=0.7, zorder=0)
            ax.annotate(f"S = {s:,}", (1.0, s), xytext=(2, 2), textcoords="offset points", fontsize=6, color="0.45")
        for d in mine:
            ax.plot(d["p"], d["rho_default"], MARKERS[starts.index(d["start"]) % len(MARKERS)], ms=4.5, alpha=0.7,
                    color=f"C{families.index(d['family'] or '?') % 10}", mec="k" if d["basis_default"] == "fit" else
                    "none", mew=0.6)
        if mine:
            best = min(mine, key=lambda d: d["p"] * d["rho_default"])
            ax.plot(best["p"], best["rho_default"], "o", mfc="none", mec="red", ms=11)
            ax.set_title(f"end loss {tp:g}%: {len(mine)} runs; least S {best['p'] * best['rho_default']:.3g} "
                         f"({best['start']}, {best['method']})", fontsize=8)
        else:
            ax.set_title(f"end loss {tp:g}%: no run reaches it", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.8, xhi)
        ax.set_ylim(0.8, yhi)
        ax.grid(alpha=0.2, which="major")
    for ax in axes[1]:
        ax.set_xlabel("p = C/H, the algorithm's inefficiency")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"rho at c = {panels.DEFAULT[0]} w_fwd, k = {panels.DEFAULT[1]:,} bits")
    keys = [Line2D([], [], marker="o", ls="none", color=f"C{i % 10}", label=f) for i, f in enumerate(families)]
    keys += [Line2D([], [], marker=MARKERS[i % len(MARKERS)], ls="none", color="0.3", label=s)
             for i, s in enumerate(starts)]
    keys.append(Line2D([], [], marker="o", ls="none", mfc="none", mec="red", ms=9, label="least S at the end loss"))
    if any(d["basis_default"] == "fit" for d in rows):
        keys.append(Line2D([], [], marker="o", ls="none", color="0.6", mec="k", label="basis fit (black edge)"))
    fig.legend(handles=keys, loc="lower center", ncol=min(len(keys), 6), fontsize=7, bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(f"{p['title']}, {data} data (grey: iso-S, S = p rho)", fontsize=10)
    fig.text(0.01, 0.005, p["note"], fontsize=5.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    path = out / f"{NAME}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary(p: dict) -> str:
    """Per data, start and end loss, the runs that reach it and the least S, and each family's least S there."""
    rows = finite(p)
    lines = ["| data | start | end loss | runs | least S (p, rho, basis; method) | each family's least S |",
             "|---|---|---|---|---|---|"]
    for ds in dict.fromkeys((d["data"], d["start"]) for d in rows):
        for tp in dict.fromkeys(d["target_pct"] for d in rows if (d["data"], d["start"]) == ds):
            mine = sorted((d for d in rows if (d["data"], d["start"]) == ds and d["target_pct"] == tp),
                          key=lambda d: d["p"] * d["rho_default"])
            b = mine[0]
            fam = {}
            for d in mine:
                fam.setdefault(d["family"], d)
            lines.append(f"| {ds[0]} | {ds[1]} | {tp:g}% | {len(mine)} | {b['p'] * b['rho_default']:.3g} ({b['p']:.3g}, "
                         f"{b['rho_default']:.3g}, {b['basis_default']}; {b['method']} {b['record']}) | "
                         + ", ".join(f"{f} {d['p'] * d['rho_default']:.3g}" for f, d in fam.items()) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = rescore.arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--publish", action="store_true",
                    help=f"put the uncut table in the evidence store if the cut leaves rows out, then PUT the cut to "
                         f"{console.PANELS_URL}/{PANEL}")
    a = ap.parse_args(argv)
    d = rescore.load(a)
    pin, shapes = d["pin"] if a.pin else frontier.pinned(pilot=True), d["shapes"]
    scored, _ = plots.load(d["rows"], pin, shapes, d["keep"], d["ladder"])
    source = a.source or (f"python -m te_core.{NAME.replace('-', '_')} (te_core {VERSION}; pin "
                          f"{' + '.join(c[:12] for c in pin)}; {len(scored)} valid checkpoints, quantities "
                          f"{d['quantities']})")
    p = build(scored, shapes, source)
    full = a.out / f"{NAME}.json"
    full.write_text(json.dumps(p, allow_nan=False) + "\n")
    art = None
    if a.publish and whole(p) is None:
        art = store(full, {"what": f"{PANEL} uncut (te_core.algorithm_policy)", "rows": len(p["rows"]),
                           "source": source})
    q = cut(p, art)
    (a.out / "console").mkdir(exist_ok=True)
    (a.out / "console" / f"{NAME}.json").write_bytes(console.body(q))
    png = figure(p, a.out)
    print(summary(p))
    print(f"{PANEL}: {len(p['rows'])} rows uncut{f' ({art})' if art else ''}; console {len(q['rows'])} rows, "
          f"{len(console.body(q))} bytes, note {len(q['note'])} characters -> {a.out}: {NAME}.json, "
          f"console/{NAME}.json, {png.name}")
    if a.publish:
        status = console.put(q)
        print(f"PUT {console.PANELS_URL}/{PANEL}: {status}")
        return 0 if status.startswith("2") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
