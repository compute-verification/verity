"""The console's second training panel, `training/algorithm-policy`: each run's efficiency loss from its algorithm
against its efficiency loss from the policy (Daniel, 2 Nov, 5:04 PM PST).

    python -m te_core.algorithm_policy --out DIR [--publish] [--data LABEL]
                                       [te_core.rescore's ledger and analysis options]

At an end loss l, one of te_core.frontier's targets (a start's l0 less 0.5%, 1%, 2% and 5%), H is the least compute
after the start of any valid checkpoint from it that reaches l (the honest envelope te_core.panels draws). A run that
reaches l does so first at a checkpoint of compute C; its algorithm inefficiency is p = C/H >= 1, its policy slowdown
there ρ (te_core.formalism.at at the dial's (c, k): c/(c − d) where Fit holds for the training, basis `fit`, else
Slowdown's bound, a target, basis `target`), and its constrained cost over H is S = p·ρ.

The panel (`COLUMNS`) has one row per run (its run id) and end loss it reaches, taken at that first checkpoint: the
start (te_core.panels' label, C0 and l0), the end loss, C, H and p, the inputs ρ is computed from there (`INPUTS`, as
note:training-20261102T2334Z-draft-rho-inputs-and-formula has them), which of those are stubs, bounds, assumptions or
estimates rather than measured (`estimated`, from the record's labels: `KINDS`), ρ and its basis at the dial's default
(te_core.panels' `DEFAULT`: c = 4,096 tokens of the start's forward, k = 80,000 bits), and the run's method, family,
record (run id@step) and data. `page_rho` is the formula the console's dial runs on a row's columns alone, and `build`
refuses a row whose columns don't reproduce te_core.formalism.at's ρ and basis at the default.

Its console copy (`cut`) is the whole table when it fits the panels route (te_core.console), else, per start, target
and family, the rows on the lower-left (p, rho_default) front plus the family's least-p and least-delta rows, with the
full table's art id in the note; the ρ inputs keep 6 significant digits so the dial reproduces rho_default. DIR gets
`algorithm-policy.json` (uncut), `console/algorithm-policy.json` (the cut) and `algorithm-policy.png` (p against ρ at
the default dial from the uncut rows, a panel per target, with iso-S diagonals). `--publish` puts the uncut table in
the evidence store when the cut leaves rows out, then PUTs the cut.
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

from . import VERSION, console, formalism, frontier, panels, plots, rescore

NAME = "algorithm-policy"
PANEL = f"training/{NAME}"
#: the inputs of ρ at any (c, k), each at the run's checkpoint (the draft note's)
INPUTS = ["w_gates", "w_fwd_gates", "sigma", "tau_bits", "a_bits", "u_points", "u_in_points", "delta_bits",
          "delta_t_bits", "d_gates", "delta_grad_bits"]
COLUMNS = ["start", "start_compute_gates", "start_loss", "end_loss", "target_pct", "C_gates", "H_gates", "p", *INPUTS,
           "estimated", "rho_default", "basis_default", "method", "family", "record", "data"]
LOSSES = ("start_loss", "end_loss")
#: significant digits of the ρ inputs in the console's copy, and of its other numbers
INPUT_DIGITS, DIGITS = 6, 4
#: the records' data: v2.3's pins score fineweb-edu (data v2.2); "v3" names the OLMo-mix records
DATA = "v2.3-pilot"
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
    """The dial's default (c, k) for a start whose forward is `w_fwd` gates a token."""
    return panels.DEFAULT[0] * w_fwd, 8.0 * panels.DEFAULT[1]


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
    """`INPUTS` for a checkpoint's quantities, d = 0 while unmeasured."""
    return [m.w, w_fwd, m.sigma, m.tau_bits, m.a_bits, points(m.u), points(m.u_in), m.delta_bits, m.delta_t,
            m.decoder_gates or 0.0, m.delta_grad_bits]


def page_rho(d: dict, c: float, k: float) -> tuple[float, str]:
    """ρ and its basis at (c, k) from a row's columns alone, the formula the console's dial runs (the draft note's):
    s = c/w; Fit for the training, delta + s·tau <= k, 2·delta_grad <= k and delta_t + delta_grad <= k (a null
    delta_grad fails), gives c/(c − d); else x0 = the largest x with u_in(x) <= k, s* = max_x x(k − u(x))/a and
    ρ = max(1, sigma·(1 − x0)·s/s*), infinite where s* is 0 and the rest isn't."""
    s, dg = c / d["w_gates"], d["delta_grad_bits"]
    if d["d_gates"] < c and d["delta_bits"] is not None and d["delta_bits"] + s * d["tau_bits"] <= k \
            and dg is not None and 2 * dg <= k and d["delta_t_bits"] + dg <= k:
        return c / (c - d["d_gates"]), "fit"
    x0 = formalism.bottom_share(profile(d["u_in_points"]), k)
    s_star = formalism.room(profile(d["u_points"]), k) / d["a_bits"]
    lead = d["sigma"] * (1 - x0) * s
    return max(1.0, lead / s_star if s_star > 0 else math.inf if lead else 0.0), "target"


def build(scored: list[dict], shapes: dict[str, dict], source: str, data: str = DATA, stamp: str | None = None) -> dict:
    """The panel, uncut: one row per run and end loss it reaches (module docstring)."""
    rate, _ = panels.dense_rate(scored, shapes)
    rows, wrong = [], []
    for name in sorted({r["start"] for r in scored}):
        on = [r for r in scored if r["start"] == name]
        l0 = frontier.start_bpb(on)
        if l0 is None:
            continue
        shape = shapes[name]
        w_fwd = panels.w_fwd(shape)
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
                row = [*head, end, -100 * t, r["gates"], h, r["gates"] / h, *inputs(m, w_fwd), estimated(m),
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


def note(art: str | None = None, cut: bool = False) -> str:
    out = ("Per run and end loss, its first checkpoint reaching it: p = C/H, H the least C of any valid checkpoint from "
           "the start reaching it; rho by te_core.formalism.at (fit: c/(c-d); target: Slowdown's bound); S = p rho. "
           "*_default: c = 4096 w_fwd, k = 80,000 bits; formula "
           "note:training-20261102T2334Z-draft-rho-inputs-and-formula. One seed.")
    if cut:
        out += (" Cut per start, target, family: (p, rho) front, least p, least delta; full table "
                + (art or f"{NAME}.json (unpublished)") + ".")
    return out


def rounded(row: list) -> list:
    def val(c, v):
        if not isinstance(v, float):
            return v
        if c in LOSSES:
            return round(v, 5)
        return float(num(v, INPUT_DIGITS if c in INPUTS else DIGITS))
    return [val(c, v) for c, v in zip(COLUMNS, row)]


def kept(rows: list[list]) -> list[list]:
    """Per start, target and family, the rows on the lower-left (p, rho_default) front (te_core.plots' staircase) and
    the family's least-p and least-delta rows, in the table's order."""
    ix = {c: i for i, c in enumerate(COLUMNS)}
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r[ix[c]] for c in ("start", "target_pct", "family")), []).append(i)
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
        raise ValueError(f"{PANEL}: the panels route refuses {len(bad)} cells ({'; '.join(bad[:3])}): make each a "
                         f"number, a string of at most {console.CELL_MAX} characters or null")
    return q


def whole(p: dict) -> dict:
    """`p` rounded, every row, if its body fits the panels route, else None."""
    q = checked(p | {"rows": [rounded(r) for r in p["rows"]], "note": note()})
    return q if len(console.body(q)) <= console.BODY_MAX else None


def cut(p: dict, art: str | None = None) -> dict:
    """The console's copy: `whole` when it fits, else the rows `kept` keeps with the full table's art id in the note."""
    if q := whole(p):
        return q
    q = checked(p | {"rows": [rounded(r) for r in kept(p["rows"])], "note": note(art, cut=True)})
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


def figure(p: dict, out: Path) -> Path:
    """p against ρ at the default dial, log-log, one panel per target, a colour per family and a marker per start,
    with the iso-S diagonals S = p ρ at `LEVELS`."""
    rows = [d for d in plots.frame(p) if d["rho_default"] is not None]
    families = sorted({d["family"] or "?" for d in rows})
    starts = list(dict.fromkeys(d["start"] for d in rows))
    targets = [-100 * t for t in frontier.TARGETS]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5), sharex=True, sharey=True)
    hi = max([d["p"] for d in rows] + [d["rho_default"] for d in rows] + [10.0]) * 2
    xs = [10 ** (i / 20) for i in range(0, int(20 * math.log10(hi)) + 1)]
    for ax, tp in zip(axes.flat, targets):
        mine = [d for d in rows if d["target_pct"] == tp]
        for s in LEVELS:
            ax.plot(xs, [s / x for x in xs], color="0.75", lw=0.7, zorder=0)
            ax.annotate(f"S = {s:,}", (1.05, s / 1.05 if s > 1 else 1.05), fontsize=6, color="0.45")
        for d in mine:
            ax.plot(d["p"], d["rho_default"], MARKERS[starts.index(d["start"]) % len(MARKERS)], ms=4.5, alpha=0.75,
                    color=f"C{families.index(d['family'] or '?') % 10}", mec="k" if d["basis_default"] == "fit" else
                    "none", mew=0.6)
        if mine:
            best = min(mine, key=lambda d: d["p"] * d["rho_default"])
            ax.plot(best["p"], best["rho_default"], "o", mfc="none", mec="red", ms=11)
            ax.set_title(f"end loss {tp:g}% ({len(mine)} runs; least S {best['p'] * best['rho_default']:.3g}, "
                         f"{best['method']})", fontsize=8)
        else:
            ax.set_title(f"end loss {tp:g}%: no run reaches it", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.9, hi)
        ax.set_ylim(0.9, hi)
        ax.grid(alpha=0.2, which="major")
    for ax in axes[1]:
        ax.set_xlabel("p = C/H, the algorithm's inefficiency")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"rho at c = {panels.DEFAULT[0]} w_fwd, k = {8 * panels.DEFAULT[1]:,} bits")
    for i, f in enumerate(families):
        axes[0, 0].plot([], [], "o", color=f"C{i % 10}", label=f)
    for i, s in enumerate(starts):
        axes[0, 0].plot([], [], MARKERS[i % len(MARKERS)], color="0.3", label=s)
    axes[0, 0].plot([], [], "o", mfc="none", mec="red", ms=9, label="least S at the end loss")
    axes[0, 0].plot([], [], "o", color="0.6", mec="k", label="basis fit (black edge)")
    axes[0, 0].legend(fontsize=6, loc="upper right", ncol=2)
    fig.suptitle(f"{p['title']} (grey: iso-S, S = p rho)", fontsize=10)
    fig.text(0.01, 0.005, p["note"], fontsize=5.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = out / f"{NAME}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary(p: dict) -> str:
    """Per start and end loss, the runs that reach it and the least S, and each family's least S there."""
    rows = [d for d in plots.frame(p) if d["rho_default"] is not None]
    lines = ["| start | end loss | runs | least S (p, rho, basis; method) | each family's least S |", "|---|---|---|---|---|"]
    for s in dict.fromkeys(d["start"] for d in rows):
        for tp in dict.fromkeys(d["target_pct"] for d in rows if d["start"] == s):
            mine = sorted((d for d in rows if d["start"] == s and d["target_pct"] == tp),
                          key=lambda d: d["p"] * d["rho_default"])
            b = mine[0]
            fam = {}
            for d in mine:
                fam.setdefault(d["family"], d)
            lines.append(f"| {s} | {tp:g}% | {len(mine)} | {b['p'] * b['rho_default']:.3g} ({b['p']:.3g}, "
                         f"{b['rho_default']:.3g}, {b['basis_default']}; {b['method']} {b['record']}) | "
                         + ", ".join(f"{f} {d['p'] * d['rho_default']:.3g}" for f, d in fam.items()) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = rescore.arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    ap.add_argument("--publish", action="store_true",
                    help=f"put the uncut table in the evidence store if the cut leaves rows out, then PUT the cut to "
                         f"{console.PANELS_URL}/{PANEL}")
    ap.add_argument("--data", default=DATA, help="the records' data (default %(default)s; v3: the OLMo-mix records)")
    a = ap.parse_args(argv)
    d = rescore.load(a)
    pin, shapes = d["pin"], d["shapes"]
    scored, _ = plots.load(d["rows"], pin, shapes, d["keep"])
    source = a.source or (f"python -m te_core.{NAME.replace('-', '_')} (te_core {VERSION}; pin "
                          f"{' + '.join(c[:12] for c in pin)}; {len(scored)} valid checkpoints, quantities "
                          f"{'rescore_v1' if d['use'] else 'stubs'})")
    p = build(scored, shapes, source, a.data)
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
