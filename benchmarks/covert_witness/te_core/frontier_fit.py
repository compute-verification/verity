"""The console's training panel `training/frontier-fit`: per data, the scaling law L(C) = E + A·C^(−γ) fitted to its
unconstrained frontier, through which te_core.algorithm_policy maps a loss to effective compute.

    python -m te_core.frontier_fit --out DIR [--publish] [--pretraining DIR] [--ladder-points FILE ...]
                                   [te_core.rescore's ledger and quantities options]

A data's unconstrained frontier is the least held-out loss (bits per byte) reached at each total training compute C
(gates) by anything on that data (`points`): from scratch, every start at (C0, l0) and every scored checkpoint from a
start at C0 + |C|, as te_core.panels draws it. From scratch on the pilot is learning_frontier's dense AdamW
pretraining runs, their loss converted as te_core.panels converts it (`panels.scratch`); on v3 it is the ladder's
runs: `--ladder-points` reads the ladder report's annealed runs (its F.json or report.json, each at its counted MACs)
and the compute-optimal points of its FIT.json, which `--ladder-fit` names as well once the ladder has three sizes
(`ladder_points`). Every C is in te_core's gates: a start's C0 is te_core.panels' `c0`, a ladder point's MACs are at
the bf16 GEMM price, as a v3 start's counted C0 is.

The fit (`fit`) reads the frontier from the least C0 of the data's starts in the manifest up: gains are read at no
less compute, and the from-scratch runs' first evaluations below it are warm-up, not a frontier. It takes one point per
1/`PER_DECADE` decade of C (the least loss there, so a stretch of checkpoints close in compute counts once) and fits E,
A and γ by least squares in relative loss, E in [0, the least loss) and A > 0, γ on a grid refined by golden section.
A data with fewer than `FIT_MIN` such points, or spanning less than a factor `FIT_SPAN` in C, has no fit, and so does
one with neither a start that has a scored checkpoint nor a ladder point: the pilot's from-scratch losses are converted
through its starts' (`panels.scratch`), and its starts' C0 rest on the dense rate their checkpoints give. Effective
compute C_eff(l) = (A/(l − E))^(1/γ) is the compute at which the fit reaches loss l, and a point's gain C_eff(l)/C its
residual in compute (1 on the fit).

The panel (`COLUMNS`) has a `fit` row per data (`row`): the form, E, A and γ, the points used, their span and the
residuals' rms and largest in relative loss, or why there is no fit (`status`); a `point` row per point used, with its
C, loss, the fit's loss there, the residual and its gain; and a `start` row per start, at (C0, l0), whose gain is the
start's own. DIR gets `frontier-fit.json` (the panel), `frontier-points.json` (every frontier point, per data) and
`console/frontier-fit.json`; `--publish` PUTs the console copy.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import VERSION, console, frontier, panels, plots, policy, rescore
from .gates import per_mac

NAME = "frontier-fit"
PANEL = f"training/{NAME}"
FORM = "L(C) = E + A*C^-gamma"
COLUMNS = ["row", "data", "form", "E_bpb", "A", "gamma", "points", "C_lo_gates", "C_hi_gates", "rms_pct",
           "max_abs_pct", "C_gates", "loss_bpb", "fit_bpb", "residual_pct", "gain", "source", "status"]
#: the fit's points per decade of compute, the least loss in each
PER_DECADE = 20
#: the fewest points a fit takes, and the least factor in C they span
FIT_MIN, FIT_SPAN = 5, 10.0
#: γ's grid, refined about its best by golden section
GAMMAS = np.linspace(0.01, 2.0, 399)
DIGITS = 6


@dataclass
class Start:
    """One start on one data: its scored checkpoints there, l0, C0, label, forward w and honest envelope (te_core.plots'
    staircase of |C| against loss)."""
    name: str
    shape: dict
    data: str
    rows: list[dict]
    rate: float
    l0: float = field(init=False)
    c0: float = field(init=False)
    label: str = field(init=False)
    w: float = field(init=False)
    honest: list = field(init=False)

    def __post_init__(self):
        self.l0, self.c0 = frontier.start_bpb(self.rows), panels.c0(self.shape, self.rate)
        self.label, self.w = plots.start_label(self.shape), policy.w_fwd(self.shape)
        self.honest = plots.staircase([(r["gates"], r["val_bpb"], r) for r in self.rows])


def starts(scored: list[dict], shapes: dict[str, dict], rate: float) -> list[Start]:
    """Each start with scored checkpoints and an l0, on its data (`data_of_shape`), the headline data first
    (te_core.panels' `DATA`). A checkpoint is on its start's data whether or not its record carries `data`: the scorer's
    measured copy of a traced record need not, and te_core.frontier scores no v3 record from a start of another
    tokenizer."""
    out = [Start(s, shapes[s], data_of_shape(shapes[s]), [r for r in scored if r["start"] == s], rate)
           for s in sorted({r["start"] for r in scored},
                           key=lambda s: (panels.DATA.index(data_of_shape(shapes[s])), s))]
    return [s for s in out if s.l0 is not None]


def data_of_shape(shape: dict) -> str:
    """The data a start's records are on: v3's for a start with data v3's tokenizer, else the pilot's."""
    return panels.DATA[0] if shape.get("tokenizer") == policy.V3_TOKENIZER else panels.DATA[1]


def ladder_points(path: Path) -> list[tuple[float, float, dict]]:
    """The v3 ladder's from-scratch points in `path`, at C in gates: its annealed runs (te-ladder/F/v1, or report/v2's
    F), each at its counted MACs, and FIT.json's compute-optimal models (te-ladder/fit/v1) at c(N)·D MACs."""
    doc = json.loads(path.read_text())
    gates = per_mac("gemm/bf16")
    schema = doc.get("schema", "")
    f = doc if schema == "te-ladder/F/v1" else doc.get("F") or {}
    out = [(p["macs"] * gates, p["bpb"], {"data": panels.DATA[0], "method_key": panels.SCRATCH,
                                          "source": f"ladder {p.get('branch') or p.get('size')}, annealed"})
           for p in f.get("annealed", []) if p.get("bpb") and p.get("macs")]
    co = doc if schema == "te-ladder/fit/v1" else doc.get("compute_optimal") or {}
    per = co.get("costs", {}).get("gates_per_mac", gates)
    out += [(q["train_macs"] * per, q["loss"], {"data": panels.DATA[0], "method_key": panels.SCRATCH,
                                                "source": f"ladder fit, N {q['non_embedding_params']:.3g} optimal"})
            for q in co.get("points", [])]
    if not out:
        raise SystemExit(f"frontier_fit: {path} holds no ladder point (te-ladder/F/v1, report/v2 or fit/v1)")
    return out


def points(sts: list[Start], runs: list[dict], shapes: dict[str, dict], rate: float,
           extra: list[tuple] = ()) -> tuple[dict[str, list], float]:
    """Each data's frontier points (C, loss, info with its `source` and `method_key`), every one of them, and the
    pilot's loss ratio (te_core.panels' `scratch`)."""
    pts, ratio = panels.scratch(runs, sts, shapes, rate)
    pts = [(x, y, r | {"source": f"start {r['start']}" if r.get("start") else f"{r['run']}: {r['method']}"})
           for x, y, r in pts]
    for s in sts:
        pts += [(s.c0 + x, y, {"data": s.data, "method_key": plots.method(r),
                               "source": f"{plots.method(r)} from {s.name} ({r['id']}@{r.get('checkpoint_step')})"})
                for x, y, r in s.honest]
    pts += list(extra)
    return {d: [p for p in pts if p[2]["data"] == d] for d in panels.DATA}, ratio


def used(pts: list[tuple], c_lo: float) -> list[tuple]:
    """The points a fit reads: the frontier (te_core.plots' staircase) from `c_lo` up, the least loss per
    1/`PER_DECADE` decade of C."""
    bins: dict[int, tuple] = {}
    for x, y, r in plots.staircase(pts):
        if x >= c_lo * (1 - 1e-12):
            b = math.floor(PER_DECADE * math.log10(x))
            if b not in bins or y < bins[b][1]:
                bins[b] = (x, y, r)
    return sorted(bins.values(), key=lambda p: p[0])


def fit(pts: list[tuple]) -> dict | None:
    """L(C) = E + A·C^(−γ) by least squares in relative loss over `pts` (C, loss, ...; by C): for each γ of `GAMMAS`
    and then by golden section about the best, (E, A) by weighted linear least squares, E held in [0, the least loss)
    and A > 0; None with fewer than `FIT_MIN` points or a span under `FIT_SPAN`. E, A and γ are kept to `DIGITS`
    significant digits, the values the panel shows and every gain reads, and the residuals are theirs."""
    if len(pts) < FIT_MIN or pts[-1][0] < FIT_SPAN * pts[0][0]:
        return None
    c, l = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    z, w, top = np.log(c / c.max()), 1 / l, l.min() * (1 - 1e-9)

    def solve(g: float) -> tuple[float, float, float, float] | None:
        x = np.exp(-g * z)
        (e, a), *_ = np.linalg.lstsq(np.stack([w, x * w], 1), np.ones_like(l), rcond=None)
        if not 0 <= e < top:
            e = min(max(float(e), 0.0), top)
            a = float(((l - e) * x * w * w).sum() / (x * x * w * w).sum())
        if a <= 0:
            return None
        r = (e + a * x) * w - 1
        return float(r @ r), g, float(e), float(a)

    tried = [t for g in GAMMAS if (t := solve(float(g)))]
    if not tried:
        return None
    step, phi = float(GAMMAS[1] - GAMMAS[0]), (math.sqrt(5) - 1) / 2
    lo, hi = max(1e-6, min(tried)[1] - step), min(tried)[1] + step
    for _ in range(60):
        m1, m2 = hi - phi * (hi - lo), lo + phi * (hi - lo)
        t1, t2 = solve(m1), solve(m2)
        tried += [t for t in (t1, t2) if t]
        if t2 is None or t1 is not None and t1[0] <= t2[0]:
            hi = m2
        else:
            lo = m1
    _, g, e, a = min(tried)
    out = {"form": FORM, "E": num(e), "A": num(a * float(c.max()) ** g), "gamma": num(g), "points": len(pts),
           "C_lo": float(c.min()), "C_hi": float(c.max())}
    res = [100 * (y / loss_at(out, x) - 1) for x, y in zip(c, l)]
    return out | {"rms_pct": math.sqrt(sum(v * v for v in res) / len(res)), "max_abs_pct": max(map(abs, res))}


def loss_at(f: dict, c: float) -> float:
    """The fit's loss at compute `c`."""
    return f["E"] + f["A"] * c ** -f["gamma"]


def c_eff(f: dict, loss: float) -> float:
    """The compute at which the fit reaches `loss` (infinite at or below E)."""
    return (f["A"] / (loss - f["E"])) ** (1 / f["gamma"]) if loss > f["E"] else math.inf


def gain(f: dict, loss: float, c0: float) -> float:
    """Effective compute over `c0`: C_eff(loss)/c0."""
    return c_eff(f, loss) / c0


def c_lo(shapes: dict[str, dict], data: str, rate: float) -> float | None:
    """The least C0 of `data`'s starts in the manifest, where its fit's domain starts."""
    c0s = [panels.c0(s, rate) for s in shapes.values() if data_of_shape(s) == data]
    return min(c0s) if c0s else None


def num(v: float, digits: int = DIGITS) -> float:
    return float(f"{v:.{digits}g}")


def row(**values) -> list:
    return [values.get(c) for c in COLUMNS]


def build(scored: list[dict], shapes: dict[str, dict], runs: list[dict], source: str, extra: list[tuple] = (),
          stamp: str | None = None) -> tuple[dict, dict]:
    """The panel and what it rests on (`info`): the rate, the ratio, the starts, each data's frontier points and its fit
    (None where there is none) with the points it read."""
    rate, rate_how = panels.dense_rate(scored, shapes)
    sts = starts(scored, shapes, rate)
    pts, ratio = points(sts, runs, shapes, rate, extra)
    rows, fits, read = [], {}, {}
    for d in panels.DATA:
        lo = c_lo(shapes, d, rate)
        anchored = any(s.data == d for s in sts) or any(p[2]["data"] == d for p in extra)
        read[d] = used(pts[d], lo) if lo is not None and anchored else []
        fits[d] = f = fit(read[d])
        if f is None:
            why = ("no fit: no start has a scored checkpoint on this data and no ladder point is given, so nothing "
                   "converts its from-scratch losses or prices its starts" if not anchored else
                   f"no fit: {len(read[d])} frontier points from the least start's C0 up"
                   + (f" ({lo:.4g} gates)" if lo is not None else " (no start)")
                   + f"; a fit takes {FIT_MIN} spanning {FIT_SPAN:g}x in C")
            rows.append(row(row="fit", data=d, form=FORM, points=len(read[d]), status=why))
            continue
        spread = [gain(f, y, x) for x, y, _ in read[d]]
        rows.append(row(row="fit", data=d, form=FORM, E_bpb=f["E"], A=f["A"], gamma=f["gamma"], points=f["points"],
                        C_lo_gates=num(f["C_lo"]), C_hi_gates=num(f["C_hi"]), rms_pct=num(f["rms_pct"], 3),
                        max_abs_pct=num(f["max_abs_pct"], 3),
                        status=f"fit; its points' gains {min(spread):.3g} to {max(spread):.3g}"))
        for x, y, r in read[d]:
            rows.append(row(row="point", data=d, C_gates=num(x), loss_bpb=round(y, 5), fit_bpb=round(loss_at(f, x), 5),
                            residual_pct=num(100 * (y / loss_at(f, x) - 1), 3), gain=num(gain(f, y, x), 4),
                            source=r["source"][:console.CELL_MAX]))
        for s in (s for s in sts if s.data == d):
            rows.append(row(row="start", data=d, C_gates=num(s.c0), loss_bpb=round(s.l0, 5),
                            fit_bpb=round(loss_at(f, s.c0), 5), residual_pct=num(100 * (s.l0 / loss_at(f, s.c0) - 1), 3),
                            gain=num(gain(f, s.l0, s.c0), 4), source=f"start {s.name} ({s.label})"))
    p = {"id": PANEL, "title": "The frontier's fit: L(C) = E + A C^-gamma per data", "kind": "table",
         "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source,
         "note": note(), "columns": COLUMNS, "rows": rows}
    return checked(p), {"rate": rate, "rate_how": rate_how, "ratio": ratio, "starts": sts, "points": pts,
                        "read": read, "fits": fits}


def note() -> str:
    return ("Per data, held-out bits/byte L against total training gates C: E + A*C^-gamma by least squares in relative "
            "loss over its unconstrained frontier (the least loss at each C: from scratch, the starts at C0 and every "
            "scored checkpoint from a start at C0 + |C|), from the least start's C0 up, one point per 1/"
            f"{PER_DECADE} decade. gain = C_eff(L)/C, C_eff = (A/(L - E))^(1/gamma): 1 on the fit; a start's row "
            "is its own. training/algorithm-policy's gains read it.")


def checked(p: dict) -> dict:
    if len(p["note"]) > console.NOTE_MAX:
        raise ValueError(f"{PANEL}: its note is {len(p['note'])} characters, over {console.NOTE_MAX}")
    if bad := console.refused(p):
        raise ValueError(f"{PANEL}: the panels route refuses it ({'; '.join(bad[:3])}; te_core.console.refused)")
    if len(console.body(p)) > console.BODY_MAX:
        raise ValueError(f"{PANEL}: {len(console.body(p))} bytes, over {console.BODY_MAX}: fewer points a decade "
                         "(PER_DECADE)")
    return p


def frontier_json(info: dict) -> dict:
    """Every frontier point per data, as `frontier-points.json` holds them."""
    return {d: [{"C_gates": x, "loss_bpb": y, "source": r["source"], "method_key": r["method_key"]}
                for x, y, r in plots.staircase(info["points"][d])] for d in panels.DATA}


def summary(info: dict) -> str:
    lines = [f"C0 rate: {info['rate']:.4g} forward passes per training token ({info['rate_how']}); from-scratch loss "
             f"ratio {info['ratio']:.5f}"]
    for d, f in info["fits"].items():
        if f is None:
            lines.append(f"{d}: no fit ({len(info['read'][d])} points read)")
            continue
        lines.append(f"{d}: L = {f['E']:.5f} + {f['A']:.5g} C^-{f['gamma']:.5f} over {f['points']} points, "
                     f"C {f['C_lo']:.4g}..{f['C_hi']:.4g}; rms {f['rms_pct']:.3f}%, max {f['max_abs_pct']:.3f}%; "
                     "starts' own gains " + ", ".join(f"{s.name} {gain(f, s.l0, s.c0):.3f}"
                                                       for s in info["starts"] if s.data == d))
    return "\n".join(lines)


def arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """`--pretraining` and `--ladder-points`, beside te_core.rescore's options."""
    ap.add_argument("--pretraining", type=Path, default=None,
                    help=f"the pilot's pretraining runs, a directory per run id (default {panels.PRETRAINING}, fetched)")
    ap.add_argument("--ladder-points", type=Path, action="append", default=[],
                    help="the v3 ladder's report.json, F.json or FIT.json, repeated: its from-scratch points")
    return ap


def inputs(a, scored: list[dict], shapes: dict[str, dict], d: dict, command: str = "te_core.frontier_fit"
           ) -> tuple[list[dict], list[tuple], str]:
    """The pretraining runs, the ladder's points (`--ladder-points`, and `--ladder-fit`'s FIT.json) and the panels'
    source for a loaded ledger `d` (te_core.rescore's `load`)."""
    runs = panels.pretraining(a.pretraining or panels.fetch(panels.PRETRAINING))
    extra = [p for path in [*a.ladder_points, *([a.ladder_fit] if a.ladder_fit else [])] for p in ladder_points(path)]
    pin = d["pin"] if a.pin else frontier.pinned(pilot=True)
    source = a.source or (f"python -m {command} (te_core {VERSION}; pin {' + '.join(c[:12] for c in pin)}; "
                          f"{len(scored)} valid checkpoints, quantities {d['quantities']}; "
                          f"{len(d.get('refused', {}))} refused (te_core.formalism.refusal); "
                          f"{len(extra)} ladder points)")
    return runs, extra, source


def scored_rows(a) -> tuple[dict, list[dict]]:
    """te_core.rescore's loading for `a` and its scored rows (te_core.plots' `load`, the pin's and the pilot's unless
    `--pin` names the commits), each with its record's family where it has none: the analysis classifies every record
    it has (`rescore.analysed`), whichever quantities score it. The valid rows te_core.formalism refuses are never
    scored; `d["refused"]` holds each one's reason by row id (te_core.frontier's `formalism_rows`)."""
    d = rescore.load(a)
    family = {run: f for (run, _), (_, f) in d["blocks"].items()}
    rows = [r | {"family": family[r["run_id"]]} if not r.get("family") and r.get("run_id") in family else r
            for r in d["rows"]]
    pin = d["pin"] if a.pin else frontier.pinned(pilot=True)
    keep = [r for r in frontier.valid(rows, pin) if d["keep"] is None or d["keep"](r)]
    scored, skipped = frontier.formalism_rows(keep, d["shapes"], d["ladder"])
    d["refused"] = {i: v[len(frontier.REFUSED):] for i, v in skipped.items() if v.startswith(frontier.REFUSED)}
    return d, scored


def refusals(refused: dict[str, str]) -> str:
    """How many rows each kind of refusal refuses (a reason up to its first colon), the most first."""
    kinds: dict[str, int] = {}
    for why in refused.values():
        kinds[why.split(":", 1)[0]] = kinds.get(why.split(":", 1)[0], 0) + 1
    return ", ".join(f"{n} {k}" for k, n in sorted(kinds.items(), key=lambda kn: (-kn[1], kn[0])))


def write(p: dict, info: dict, tables: Path, out: Path) -> None:
    """The panel and every frontier point into `tables`, and its console copy into `out`/console."""
    tables.mkdir(parents=True, exist_ok=True)
    (out / "console").mkdir(parents=True, exist_ok=True)
    (tables / f"{NAME}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
    (tables / "frontier-points.json").write_text(json.dumps(frontier_json(info), allow_nan=False) + "\n")
    (out / "console" / f"{NAME}.json").write_bytes(console.body(p))


def main(argv=None) -> int:
    ap = arguments(rescore.arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0])))
    ap.add_argument("--publish", action="store_true", help=f"PUT the panel to {console.PANELS_URL}/{PANEL}")
    a = ap.parse_args(argv)
    d, scored = scored_rows(a)
    runs, extra, source = inputs(a, scored, d["shapes"], d)
    p, info = build(scored, d["shapes"], runs, source, extra)
    write(p, info, a.out, a.out)
    print(summary(info))
    if d["refused"]:
        print(f"{len(d['refused'])} valid checkpoints refused, never scored: {refusals(d['refused'])}")
    print(f"{PANEL}: {len(p['rows'])} rows, {len(console.body(p))} bytes -> {a.out}: {NAME}.json, "
          f"frontier-points.json, console/{NAME}.json")
    if a.publish:
        status = console.put(p)
        print(f"PUT {console.PANELS_URL}/{PANEL}: {status}")
        return 0 if status.startswith("2") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
