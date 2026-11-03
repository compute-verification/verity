"""The console's training panel `training/algorithm-policy`: per run, start and gain target, the algorithm's
inefficiency p = C/H against the policy's slowdown ρ.

    python -m te_core.algorithm_policy --out DIR [--publish] [--pretraining DIR] [--ladder-points FILE ...]
                                       [te_core.rescore's ledger and quantities options]

It reads what te_core.frontier_fit reads, and that fit: the valid records of the pin and of the pilot (pin.py's
`pilot`) unless `--pin` names the commits, scored on their measured quantities unless `--quantities` says otherwise,
where the scoring cap admits them (te_core.policy's `eligible`), each checkpoint on its start's data
(te_core.frontier_fit's `data_of_shape`: `v3`, harness v3's OLMo 2 mix, else `v2.3-pilot`). A checkpoint's gain is
Daniel's C(L1)/C(L0) (TRN-40, 7:05 PM PST; coordinator 08:09Z and 11:16Z): the compute at which its data's fit
(te_core.frontier_fit, `training/frontier-fit`) reaches L1, its held-out loss, over the compute at which it reaches L0,
its start's l0 (te_core.frontier_fit's `gain_from`); a row carries C0 (`start_compute_gates`) and the start's own
effective compute C_eff(l0) (`start_effective_gates`, over C0 the start's own gain, which the gain no longer reads). At
a gain target g (`GAINS`: 1.1, 2 and 11, so +10%, +100% and +1000%) a start's target loss is E + (l0 − E)·g^(−γ),
below l0 at every g (te_core.frontier_fit's `target`), and H is the least |C| of any scored checkpoint from the start
on the same data reaching it, every checkpoint of every run pooled (null when none does). Every run from the start has
a row at every target: `reached` "yes" at its first checkpoint reaching it, C that checkpoint's |C| and p = C/H >= 1;
else "no" at its last checkpoint, C its total |C| and p = C/H a lower bound on its p (null without H). A start at or
below its fit's E has no rows, and a data with no fit none.

Each row's ρ is te_core.formalism's `published` at its checkpoint at the console's default policy (`CONSOLE`,
te_core.policy's: one full session of the 45M reference, attention included), the one default of every ρ column
(coordinator 08:09Z and 10:23Z: rows are reported at that c only); `rho_served` is the same at the registered serving
block (te_core.panels' `SERVED`), the figure beside every headline. `rho_default` is it as a number: under reading B
c/(c − d) where Fit holds for the training (basis `fit`), else Slowdown's token form (`target`), null where infinite;
under reading A, Fit's, else the one bound's max over the kinds valid on their own (basis its binding kind, `none` at
1), and null with basis `unmeasured` where no such kind is measured. `rho_inputs`, one JSON cell, carries every input
the console's dial needs to recompute the token form at any (c, k), a row's ρ under reading B and a sensitivity under
A: `panels.PACKED` and `SPLIT`, u_in through its points at the dial's k (`panels.through`, the console's among them),
where x0 is exact; `page_rho` is the dial's formula on them. `build` refuses a row whose rounded columns don't give
formalism.at's token form and basis at every cell of te_core.panels' dial and at `CONSOLE` (`panels.mismatches`).
`estimated` names the inputs that are stubs, bounds, assumptions or estimates, and gain:provisional where its data's
fit is provisional (te_core.frontier_fit's `FIT_TARGET`), which the note names. The dashboard's cells (coordinator
08:26Z and the dashboard's ruling): `rho_console`, the same ρ ("inf" where infinite, null where reading A
measures no kind, never 1; coordinator 09:46Z), `reading` (A where formalism.READING_A and the row's token kind
is measured, its trace's entries and an op log, else B, whose ρ is Slowdown's token form, a target), `te_core` the
VERSION that priced it, `binding_kind` and `rests_on`, and `kinds`, a JSON cell of the terms its max takes and the
sensitivities it leaves out.

The coverage (`coverage`) is, per gain target, data and decade of ρ (`DECADES`: from 1, 10, ... and 1e5 and above,
an infinite ρ there, and an unmeasured one, or one resting on a stub input (basis `stub`), in none, which the table
counts), the reached row with the least p, else the failed attempt with the greatest gain, else EMPTY.
The console's copy (`cut`) is the whole table when it fits the panels route (te_core.console), else the coverage's
rows and, at the first tier of `TIERS` that fits, per data, start and target (and family, at the first tier) the
reached rows on the lower-left (p, rho_default) front and the failed attempt with the greatest gain, without the
columns of `CONSOLE_DROP`, which the route's 24 columns leave no room for. DIR gets the
uncut tables in `tables/` (`algorithm-policy.json`, `coverage.json`, `coverage.md`, `refused.json`, each valid
checkpoint te_core.formalism refuses with its reason, which no row holds, and te_core.frontier_fit's
`frontier-fit.json` and `frontier-points.json`), both panels' console copies in `console/` and `algorithm-policy.png`
(p against ρ at the console's default on the headline data, v3 once it has rows, else the pilot: a panel per gain
target, with iso-S diagonals). `--publish` GETs what the route serves (`console.served`; no PUT when that fails), puts
`tables/` in the evidence store, whose art id the cut's note names, and PUTs `training/frontier-fit`, then the cut. A
version switch never empties a populated panel: on each data whose served rows are at a newer te_core, or at an older
one whose coverage cells there this build fills fewer of (`kept_data`), both panels keep the served rows (`guarded`),
which the note and the board line name. The coverage's table and board line give the reached cells' ρ at the
registered serving block beside the console's default (`served_text`).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import VERSION, console, formalism, frontier_fit, panels, plots, policy, rescore
from .gates import UNIT

NAME = "algorithm-policy"
PANEL = f"training/{NAME}"
#: the gain targets, C(L1)/C(L0) on the fit (te_core.frontier_fit's `gain_from`)
GAINS = (1.1, 2, 11)
#: the coverage's decades of ρ, each from its value to the next's, the last with every larger and an infinite ρ
DECADES = (1, 10, 100, 1_000, 10_000, 100_000)
#: the inputs of ρ at any (c, k), each at the run's checkpoint (te_core.panels' `rho_columns` reads them)
INPUTS = ["w_gates", "w_fwd_gates", "sigma", "tau_bits", "a_bits", "u_points", "u_in_points", "delta_bits",
          "delta_t_bits", "d_gates", "delta_grad_bits"]
#: the inputs of ρ te_core.panels' candidate layout gives columns of their own, which `rho_inputs` holds here, the one
#: JSON cell the dashboard's dial reads (coordinator 07:57Z: 4,096-character cells)
SPLIT = ("delta_bits", "w_fwd_gates", "u_points", "u_in_points")
COLUMNS = ["start", "start_compute_gates", "start_effective_gates", "start_loss", "end_loss", "gain_target", "gain",
           "reached", "C_gates", "H_gates", "p", "rho_default", "basis_default", "method_key", "family", "record", "data",
           "estimated", "rho_inputs", "reading", "te_core", "rho_console", "binding_kind", "rests_on", "kinds",
           "rho_served"]
#: the console's default policy, the one default of `rho_default` and `rho_console` (coordinator 08:09Z): te_core.policy's
#: `CONSOLE`, one full session of the 45M reference with 4,096 input and 1,024 output tokens (`policy.session_gates`)
#: and k = 65,536 bits
CONSOLE = policy.CONSOLE
#: the columns the console's copy leaves out (the stored tables keep them), so its rows fit the route's
#: `console.COLUMNS_MAX` and its finer tiers `console.BODY_MAX`: H is C/p, and what ρ's inputs rest on is the same on
#: every row of a te_core version
CONSOLE_DROP = ("H_gates", "estimated")
CONSOLE_COLUMNS = [c for c in COLUMNS if c not in CONSOLE_DROP]
LOSSES = ("start_loss", "end_loss")
#: significant digits of the console copy's compute and ρ inputs, so its dial reproduces ρ and its gains and p
#: recompute, and of its other numbers
INPUT_DIGITS, DIGITS = 6, 4
EXACT = ("start_compute_gates", "start_effective_gates", "C_gates", "H_gates", "delta_bits", "w_fwd_gates")
#: the cut's tiers, the finest first: the groups whose (p, ρ) fronts and best failed attempts it keeps
TIERS = ("family", "start", "coverage")
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
EMPTY = "EMPTY"
#: basis_default where reading A measures no kind valid on its own, its ρ null: in no decade, never an infinite ρ
UNMEASURED = "unmeasured"
#: basis_default where a row's ρ rests on a stub input (`estimated`'s stub tag; coordinator 08:09Z item 2): in no
#: decade, and off the cut's fronts and the figure; the stored table keeps it
STUB = "stub"


def finite(rho: float | None) -> float | None:
    """A published ρ as a number column: null where infinite or unmeasured."""
    return None if rho is None or math.isinf(rho) else rho


def num(v: float, digits: int = INPUT_DIGITS) -> str:
    """`v` to `digits` significant digits, compact: 24601880 -> 2.46019e7, 0.25 -> 0.25."""
    return f"{v:.{digits}g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def points(u) -> str:
    """A profile's points as the console's string, "x:bits;..." ("0:0;0.25:1.2e6;1:4.8e6")."""
    return ";".join(f"{num(x)}:{num(b)}" for x, b in u)


def profile(text: str) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(b)) for x, b in (p.split(":") for p in text.split(";")))


def simplified(u) -> tuple[tuple[float, float], ...]:
    """A profile without its redundant points, the same function piece by piece: a repeat of the point before, or an
    inner point of a flat or vertical run (a profile capped at Δ has long flat runs at Δ)."""
    out = [tuple(u[0])]
    for i, p in enumerate(map(tuple, u[1:]), 1):
        q = out[-1]
        if p == q or i + 1 < len(u) and (q[1] == p[1] == u[i + 1][1] or q[0] == p[0] == u[i + 1][0]):
            continue
        out.append(p)
    return tuple(out)


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
    d < c), without redundant points (`simplified`), and Δ_grad the one Fit binds on (`Measured.fit_delta_grad_bits`)."""
    u, u_in = (formalism.capped(m.u, m.delta_bits), formalism.capped(m.u_in, m.delta_bits)) \
        if m.delta_bits is not None and m.decoder_gates == 0 else (m.u, m.u_in)
    return [m.w, w_fwd, m.sigma, m.tau_bits, m.a_bits, points(simplified(u)), points(simplified(u_in)), m.delta_bits,
            m.delta_t, m.decoder_gates, m.fit_delta_grad_bits]


def page_rho(d: dict, c: float, k: float) -> tuple[float, str]:
    """ρ and its basis at (c, k) from a row's inputs alone (te_core.panels' `page_inputs` gives them from its
    columns), the formula the console's dial runs (the draft note's, with te_core 4.x's two terms where a row has them:
    `w_fit_gates`, the least w of a part, which Fit's compute block binds on, else w; and `alpha_ratio`, ᾱ/α_max, else
    1): Fit for the training, delta + (c/w_fit)·tau <= k, 2·delta_grad <= k and delta_t + delta_grad <= k (a null
    delta_grad fails), with d < c (a null d, unmeasured, fails), gives c/(c − d); else s = c/w, x0 = the largest x with
    u_in(x) <= k, s* = max_x x(k − u(x))/a and ρ = max(1, sigma·(1 − x0)·s·alpha_ratio/s*), infinite where s* is 0 and
    the rest isn't; where d < c and delta is known, u and u_in are at most delta."""
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


def rounded(row: list) -> list:
    def val(c, v):
        if not isinstance(v, float):
            return v
        if c in LOSSES:
            return round(v, 5)
        return float(num(v, INPUT_DIGITS if c in EXACT else DIGITS))
    return [val(c, v) for c, v in zip(COLUMNS, row)]


def rho_inputs(m: formalism.Measured, w_fwd: float) -> str:
    """The dial's inputs as one JSON cell: te_core.panels' `rho_columns` (`panels.PACKED`) with SPLIT's, numbers to
    INPUT_DIGITS significant digits."""
    cols = panels.rho_columns(m, w_fwd)
    nums = ",".join(f'"{k}":' + ("null" if v is None else num(v)) for k, v in (("delta_bits", m.delta_bits),
                                                                             ("w_fwd_gates", w_fwd)))
    return (cols["rho_inputs"][:-1] + f",{nums},\"u_points\":{json.dumps(cols['u_points'])},"
            f"\"u_in_points\":{json.dumps(cols['u_in_points'])}}}")


def kinds(one: dict) -> str:
    """A published ρ's kinds as one JSON cell (te_core.formalism's `published`): `bound`, the terms its max takes, and
    `sensitivity`, those it leaves out; null where unmeasured, "inf" where infinite."""
    def cell(x: float | None) -> str:
        return "null" if x is None else '"inf"' if math.isinf(x) else num(x, DIGITS)

    def part(d: dict) -> str:
        return "{" + ",".join(f'"{k}":{cell(x)}' for k, x in d.items()) + "}"
    return f'{{"bound":{part(one["terms"])},"sensitivity":{part(one["sensitivities"])}}}'


def runs_of(st: frontier_fit.Start) -> dict[str, list[dict]]:
    """A start's runs (run id, else record id), each its checkpoints by step."""
    out: dict[str, list[dict]] = {}
    for r in sorted(st.rows, key=lambda r: (r.get("checkpoint_step") or 0, r["gates"])):
        out.setdefault(r.get("run_id") or r["id"], []).append(r)
    return out


def build(info: dict, source: str, stamp: str | None = None) -> dict:
    """The panel, uncut, from te_core.frontier_fit's `build` (`info`: its starts and fits): a row per run, start and
    gain target (module docstring), reached rows first by p, then the rest by gain."""
    ix = {c: i for i, c in enumerate(COLUMNS)}

    def order(x: list) -> tuple:
        if x[ix["reached"]] == "yes":
            return False, math.inf if x[ix["p"]] is None else x[ix["p"]], x[ix["C_gates"]]
        return True, -x[ix["gain"]], x[ix["C_gates"]]

    rows, wrong, rhos = [], [], {}
    for st in info["starts"]:
        f = info["fits"].get(st.data)
        if f is None or st.l0 <= f["E"]:
            continue
        runs, dials, own = runs_of(st), {}, frontier_fit.c_eff(f, st.l0)
        for g in GAINS:
            target = frontier_fit.target(f, st.l0, g)
            h = panels.least(st.honest, target)
            got = []
            for run, cps in runs.items():
                hit = next((r for r in cps if r["val_bpb"] <= target), None)
                r = hit or cps[-1]
                m = r["measured"]
                if r["id"] not in dials:
                    dials[r["id"]] = panels.dial(m, st.w) | {(CONSOLE[0] / st.w, CONSOLE[1]): formalism.at(m, *CONSOLE)}
                    rhos[r["id"]] = formalism.published(m), formalism.published(m, *panels.SERVED)
                one, served = rhos[r["id"]]
                gain, est = frontier_fit.gain_from(f, st.l0, r["val_bpb"]), estimated(m)
                basis = (UNMEASURED if one["rho"] is None else STUB if any(t.endswith(f":{STUB}") for t in est.split())
                         else one["binding_kind"])
                row = [st.label, st.c0, None if math.isinf(own) else own, st.l0, r["val_bpb"], g,
                       None if math.isinf(gain) else gain,
                       "yes" if hit else "no", r["gates"], h, r["gates"] / h if h else None, finite(one["rho"]),
                       basis, plots.method(r), r.get("family"), f"{run}@{r.get('checkpoint_step')}", st.data,
                       " ".join(filter(None, (est, "gain:provisional" if f.get("provisional") else "")))]
                cols = {"rho_inputs": rho_inputs(m, st.w), "reading": one["reading"], "te_core": VERSION,
                        "rho_console": None if one["rho"] is None else "inf" if math.isinf(one["rho"]) else one["rho"],
                        "binding_kind": one["binding_kind"], "rests_on": one["rests_on"], "kinds": kinds(one),
                        "rho_served": finite(served["rho"])}
                row += [cols[c] for c in COLUMNS[len(row):]]
                d = dict(zip(COLUMNS, rounded(row)))
                wrong += panels.mismatches(d | {"run": d["record"]}, dials[r["id"]], st.w)
                got.append(row)
            rows += sorted(got, key=order)
    if wrong:
        raise ValueError(f"{PANEL}: {len(wrong)} cells where a row's columns don't give te_core.formalism.at's rho "
                         f"({'; '.join(wrong[:3])}): the dial (page_rho) has no context charge (--charge-context), "
                         f"and its inputs have {INPUT_DIGITS} significant digits")
    return {"id": PANEL, "title": f"Algorithm against policy: p = C/H against rho at each gain target (C, H and c in "
                                  f"{UNIT})",
            "kind": "table", "updated_at": stamp or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "source": source,
            "note": note(), "columns": COLUMNS, "rows": rows}


def note(art: str | None = None, tier: str | None = None, kept: str = "", provisional: tuple = ()) -> str:
    """The panel's note; `tier` names what a cut keeps (None: no cut), `art` the stored tables, `kept` the data whose
    served rows stay (`kept_data`) and `provisional` the data whose gains read a provisional fit. Over the route's
    `console.NOTE_MAX` it drops the kept data's versions, then the dial's sentence."""
    def text(kept: str, dial: bool) -> str:
        out = (f"Per run, start and g = {', '.join(f'{g:g}' for g in GAINS)} (gain C(L)/C(l0), training/frontier-fit): "
               "yes at the first checkpoint reaching g, else no at the last. p = C/H, H the start's least C to g (a "
               "lower bound if no). rho_default = rho_console (console default); rho_served (registered serving "
               "block)." + (" Dial: rho_inputs (u_in exact at its k)." if dial else ""))
        if provisional:
            out += f" {', '.join(provisional)}: provisional fit."
        if tier:
            out += " Cut: coverage" + ("." if tier == "coverage" else " + per start, g"
                                       + (", family" if tier == "family" else "") + ": (p, rho) front, best no.")
        if kept:
            out += f" Kept as served: {kept}."
        if art:
            out += f" Tables {art}."
        return out

    bare = re.sub(r" \(te_core [\d.]+\)", "", kept)
    return next((t for t in (text(kept, True), text(bare, True)) if len(t) <= console.NOTE_MAX), text(bare, False))


def decade(rho: float | None, basis: str | None = None) -> int | None:
    """The coverage's decade of ρ (`DECADES`) a row's rho_default is in: an infinite one (null) is in the last, and an
    unmeasured one (null, basis `UNMEASURED`) or one resting on a stub (basis `STUB`) in none."""
    if basis in (UNMEASURED, STUB):
        return None
    if rho is None:
        return len(DECADES) - 1
    return max(i for i, lo in enumerate(DECADES) if rho >= lo or i == 0)


def unmeasured_rows(p: dict, basis: str = UNMEASURED) -> int:
    """How many of a panel's rows have no measured ρ (basis_default `UNMEASURED`), or rest on a stub (`STUB`), which
    no decade holds."""
    return sum(d.get("basis_default") == basis for d in plots.frame(p))


def label(i: int) -> str:
    lo = DECADES[i]
    return (f"{lo:g}" if lo < 1000 else f"1e{round(math.log10(lo))}") + ("+" if i == len(DECADES) - 1 else "")


def coverage(p: dict) -> list[dict]:
    """Each cell of gain target by data by decade of ρ: the reached row with the least p (then ρ, then C), else the
    failed attempt with the greatest gain (then the least C), else none; `row` its index in `p`'s rows."""
    rows = plots.frame(p)
    out = []
    for g in GAINS:
        for data in panels.DATA:
            for i in range(len(DECADES)):
                on = [(j, d) for j, d in enumerate(rows) if d["gain_target"] == g and d["data"] == data
                      and decade(d["rho_default"], d.get("basis_default")) == i]
                yes = [(j, d) for j, d in on if d["reached"] == "yes"]
                no = [(j, d) for j, d in on if d["reached"] == "no"]
                best = (min(yes, key=lambda jd: (jd[1]["p"], jd[1]["rho_default"] or math.inf, jd[1]["C_gates"]))
                        if yes else max(no, key=lambda jd: (jd[1]["gain"], -jd[1]["C_gates"])) if no else None)
                out.append({"gain_target": g, "data": data, "rho_decade": label(i),
                            "state": "empty" if best is None else "reached" if yes else "failed",
                            "row": None if best is None else best[0], "rows": len(on), "reached": len(yes),
                            **({} if best is None else {k: best[1].get(k) for k in (
                                "start", "record", "method_key", "family", "p", "gain", "rho_default", "basis_default",
                                "rho_served", "C_gates", "H_gates", "end_loss")})})
    return out


def rho_text(rho: float | None) -> str:
    return "inf" if rho is None else f"{rho:.3g}"


def cell_text(c: dict, short: bool = False) -> str:
    """A coverage cell as text: `EMPTY`, "yes p ..." or "no gain ...", with its ρ at the console's default and at the
    registered serving block and its run unless `short`."""
    if c["state"] == "empty":
        return EMPTY
    head = (f"yes p {c['p']:.3g}" if c["state"] == "reached" else
            f"no gain {c['gain']:.3g}" + ("" if c["p"] is None else f", p >= {c['p']:.3g}"))
    return head if short else (f"{head}, rho {rho_text(c['rho_default'])}, served {rho_text(c.get('rho_served'))} "
                               f"({c['start']}, {c['method_key']}, {c['record']})")


def served_text(cov: list[dict]) -> str:
    """The reached cells' ρ per data at the console's default and at the registered serving block (te_core.panels'
    `SERVED`), the figure beside every headline (coordinator 08:09Z), "" without a reached cell."""
    def span(xs: list) -> str:
        fin = [x for x in xs if x is not None]
        return (frontier_fit.span(min(fin), max(fin)) if fin else "none finite") + (", and inf" if None in xs else "")

    parts = [f"{data} {span([c['rho_default'] for c in yes])} at the console's default, "
             f"{span([c.get('rho_served') for c in yes])} at the registered serving block"
             for data in panels.DATA if (yes := [c for c in cov if c["data"] == data and c["state"] == "reached"])]
    return (f"Reached cells' rho: {'; '.join(parts)} (c {panels.SERVED[0]:.3g} gates, k {panels.SERVED[1]:,} bits)."
            if parts else "")


def coverage_md(cov: list[dict], fits: dict, stamp: str, refused: dict[str, str] | None = None,
                unmeasured: int = 0, stubs: int = 0) -> str:
    """The coverage table in Markdown, a row per gain target and data, a column per decade of ρ, and how many valid
    checkpoints te_core.formalism refused (`refused`, row id -> reason) and rows have no measured ρ (`unmeasured`) or
    rest on a stub (`stubs`), which no cell can hold."""
    lines = [f"# training/algorithm-policy coverage, {stamp}", "",
             "Per gain target, data and decade of rho_default: the reached row with the least p, else the failed "
             f"attempt with the greatest gain (p a lower bound), else **{EMPTY}**.", "",
             "| gain target | data | " + " | ".join(f"rho {label(i)}" for i in range(len(DECADES))) + " |",
             "|---|---|" + "---|" * len(DECADES)]
    for g in GAINS:
        for data in panels.DATA:
            cells = [c for c in cov if (c["gain_target"], c["data"]) == (g, data)]
            lines.append(f"| {g:g} | {data}{'' if fits.get(data) else ' (no fit)'} | "
                         + " | ".join(f"**{EMPTY}**" if c["state"] == "empty" else cell_text(c) for c in cells) + " |")
    n = sum(c["state"] == "empty" for c in cov)
    lines += ["", f"{n} of {len(cov)} cells {EMPTY}; {sum(c['state'] == 'failed' for c in cov)} hold only failed "
                  "attempts."]
    if served := served_text(cov):
        lines.append(served)
    if refused:
        lines.append(f"{len(refused)} valid checkpoints are refused by te_core.formalism, never scored, so in no cell "
                     f"({frontier_fit.refusals(refused)}; refused.json has each one's reason).")
    if stubs:
        lines.append(f"{stubs} {'row rests' if stubs == 1 else 'rows rest'} on a stub input (basis_default {STUB}), "
                     "in no decade.")
    if unmeasured:
        lines.append(f"{unmeasured} {'row' if unmeasured == 1 else 'rows'} with no measured rho under reading A "
                     f"(rho_default null, basis_default {UNMEASURED}), in no decade.")
    return "\n".join(lines) + "\n"


def board_line(cov: list[dict], fits: dict, stamp: str, art: str | None, refused: dict[str, str] | None = None,
               unmeasured: int = 0, stubs: int = 0) -> str:
    """The coverage as one board line's text, after its time and tag."""
    parts = []
    for g in GAINS:
        for data in panels.DATA:
            cells = [c for c in cov if (c["gain_target"], c["data"]) == (g, data)]
            if not fits.get(data):
                parts.append(f"g {g:g} {data}: no fit, all {len(cells)} {EMPTY}")
                continue
            parts.append(f"g {g:g} {data}: " + " | ".join(cell_text(c, short=True) for c in cells))
    n = sum(c["state"] == "empty" for c in cov)
    return (f"coverage at {stamp} ({art or 'unpublished'}), gain target x data x decade of rho_default (1, 10, 100, "
            f"1e3, 1e4, 1e5+): the least-p reached row, else the failed attempt with the greatest gain, else {EMPTY}. "
            + "; ".join(parts) + f". {n} of {len(cov)} cells {EMPTY}."
            + (f" {served}" if (served := served_text(cov)) else "")
            + (f" {len(refused)} valid checkpoints refused by te_core.formalism, never scored "
               f"({frontier_fit.refusals(refused)})." if refused else "")
            + (f" {stubs} {'row rests' if stubs == 1 else 'rows rest'} on a stub, in no decade." if stubs else "")
            + (f" {unmeasured} {'row' if unmeasured == 1 else 'rows'} unmeasured under reading A, in no decade."
               if unmeasured else ""))


def headline(p: dict) -> str:
    """The data the figure shows (te_core.panels' headline): v3 once the table has v3 rows, else the pilot."""
    have = {r[COLUMNS.index("data")] for r in p["rows"]}
    return next((d for d in panels.DATA if d in have), panels.DATA[-1])


def kept(rows: list[list], tier: str, cov: set[int]) -> list[list]:
    """The coverage's rows (`cov`, indices into `rows`) and, unless `tier` is "coverage", per data, start, gain target
    and (at tier "family") family, the reached rows on the lower-left (p, rho_default) front (te_core.plots'
    staircase, an infinite ρ above every finite one) and the failed attempt with the greatest gain, neither resting on
    a stub (`STUB`), in the table's order."""
    ix = {c: i for i, c in enumerate(COLUMNS)}

    def big(v):
        return sys.float_info.max if v is None else v

    keep = set(cov)
    if tier != "coverage":
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(rows):
            key = (r[ix["data"]], r[ix["start"]], r[ix["gain_target"]]) + ((r[ix["family"]],) if tier == "family" else ())
            groups.setdefault(key, []).append(i)
        for g in groups.values():
            yes = [i for i in g if rows[i][ix["reached"]] == "yes" and rows[i][ix["basis_default"]] != STUB]
            keep |= {i for _, _, i in plots.staircase([(big(rows[i][ix["p"]]), big(rows[i][ix["rho_default"]]), i)
                                                       for i in yes])}
            no = [i for i in g if rows[i][ix["reached"]] == "no" and rows[i][ix["basis_default"]] != STUB]
            if no:
                keep.add(max(no, key=lambda i: (rows[i][ix["gain"]], -rows[i][ix["C_gates"]])))
    return [r for i, r in enumerate(rows) if i in keep]


def checked(q: dict) -> dict:
    if len(q["note"]) > console.NOTE_MAX:
        raise ValueError(f"{PANEL}: its note is {len(q['note'])} characters, over {console.NOTE_MAX}")
    if bad := console.refused(q):
        raise ValueError(f"{PANEL}: the panels route refuses it ({'; '.join(bad[:3])}; te_core.console.refused): a "
                         f"source of at most {console.SOURCE_MAX} characters, at most {console.COLUMNS_MAX} columns, "
                         f"each cell a number, a string of at most {console.CELL_MAX} characters or null")
    return q


def console_rows(rows: list[list]) -> list[list]:
    """`rows` rounded, on `CONSOLE_COLUMNS`."""
    on = [COLUMNS.index(c) for c in CONSOLE_COLUMNS]
    return [[x[i] for i in on] for x in map(rounded, rows)]


def cut(p: dict, art: str | None = None, served: str = "") -> dict:
    """The console's copy: every row rounded when it fits the panels route, else the rows `kept` keeps at the first of
    `TIERS` that fits, on `CONSOLE_COLUMNS`, the note naming the data whose gains read a provisional fit, the tier, the
    data whose served rows it keeps (`served`) and the stored tables' art id."""
    prov = tuple(dict.fromkeys(d["data"] for d in plots.frame(p) if "gain:provisional" in (d.get("estimated") or "")))
    q = checked(p | {"columns": CONSOLE_COLUMNS, "rows": console_rows(p["rows"]),
                     "note": note(art, kept=served, provisional=prov)})
    if len(console.body(q)) <= console.BODY_MAX:
        return q
    cov = {c["row"] for c in coverage(p) if c["row"] is not None}
    for tier in TIERS:
        q = checked(p | {"columns": CONSOLE_COLUMNS, "rows": console_rows(kept(p["rows"], tier, cov)),
                         "note": note(art, tier, served, prov)})
        if len(console.body(q)) <= console.BODY_MAX:
            return q
    raise ValueError(f"{PANEL}: the coverage's rows alone are {len(console.body(q))} bytes, over {console.BODY_MAX}")


def version_of(p: dict, data: str) -> tuple[int, ...] | None:
    """The te_core version a panel's rows on `data` are priced at: their `te_core` cell, else the one the panel's source
    names (a panel from before the cell); None without rows on `data`."""
    ds = [d for d in plots.frame(p) if d.get("data") == data]
    if not ds:
        return None
    v = next((str(d["te_core"]) for d in ds if d.get("te_core")), None)
    v = v or next(iter(re.findall(r"te_core (\d+(?:\.\d+)*)", p.get("source") or "")), None)
    return tuple(int(x) for x in v.split(".")) if v else None


def filled(p: dict, data: str) -> int:
    """The coverage cells on `data` a panel's rows fill (`coverage`)."""
    return sum(c["state"] != "empty" for c in coverage(p) if c["data"] == data)


def kept_data(p: dict, served: dict | None) -> list[str]:
    """The data whose rows the route serves stay (coordinator 07:58Z: a version switch never replaces a populated panel
    with an emptier one): served at a newer te_core than this one, or at an older one whose coverage cells on it
    this build fills fewer of. The same te_core's rows are replaced."""
    mine = tuple(int(x) for x in VERSION.split("."))
    out = []
    for data in panels.DATA:
        was = version_of(served, data) if served else None
        if was is not None and (was > mine or was < mine and filled(p, data) < filled(served, data)):
            out.append(data)
    return out


def merged(p: dict, served: dict, keep: list[str]) -> dict:
    """`p` with its rows on each data of `keep` that `served` has rows on replaced by `served`'s, by column name (null
    where `served` has no such column), by data in te_core.panels' `DATA` order."""
    at = p["columns"].index("data")
    theirs = [d for d in plots.frame(served) if d.get("data") in keep]
    on = {d["data"] for d in theirs}
    rows = [r for r in p["rows"] if r[at] not in on] + [[d.get(c) for c in p["columns"]] for d in theirs]
    order = {d: i for i, d in enumerate(panels.DATA)}
    return p | {"rows": sorted(rows, key=lambda r: order.get(r[at], len(order)))}


def guarded(p: dict, fp: dict, served: dict[str, dict]) -> tuple[dict, dict, str]:
    """`p` and te_core.frontier_fit's panel `fp` as they go over what the route serves (`served`, by id): on each data
    `kept_data` keeps, both panels' served rows, and the note's words for those data ("" when none)."""
    keep = kept_data(p, ap := served.get(PANEL))
    if not keep:
        return p, fp, ""
    fit = served.get(frontier_fit.PANEL)
    return (merged(p, ap, keep), merged(fp, fit, keep) if fit else fp,
            ", ".join(f"{d} (te_core {'.'.join(map(str, version_of(ap, d)))})" for d in keep))


def store(path: Path, meta: dict) -> str:
    """The directory `path` in the evidence store, preserved (`research data put --tree`); its art id."""
    exe = shutil.which("research") or str(Path(sys.executable).with_name("research"))
    out = subprocess.run([exe, "data", "put", "--kind", "evidence/v1", "--tree", str(path), "--preserve", "--json",
                          "--meta", json.dumps(meta)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)["id"]


def figure(p: dict, out: Path) -> Path:
    """p against ρ at the console's default on the headline data, log-log, one panel per gain target: a colour per family,
    a marker per start, filled where reached and open at a failed attempt's lower bound on p, the coverage's decades
    of ρ dotted and the iso-S diagonals S = p ρ at `LEVELS`."""
    data = headline(p)
    every = [d for d in plots.frame(p) if d["data"] == data]
    rows = [d for d in every if d["p"] is not None and d["rho_default"] is not None and d["basis_default"] != STUB]
    families = sorted({d["family"] or "?" for d in rows})
    starts = list(dict.fromkeys(d["start"] for d in rows))
    xlo = 10 ** math.floor(math.log10(min([d["p"] for d in rows] + [0.8])))
    xhi = 10 ** math.ceil(math.log10(1.5 * max([d["p"] for d in rows] + [5.0])))
    yhi = 10 ** math.ceil(math.log10(1.5 * max([d["rho_default"] for d in rows] + [float(max(LEVELS))])))
    fig, axes = plt.subplots(1, len(GAINS), figsize=(15, 6), sharex=True, sharey=True)
    xs = [10 ** (i / 20) for i in range(int(20 * math.log10(xlo)), int(20 * math.log10(xhi)) + 1)]
    for ax, g in zip(axes, GAINS):
        mine = [d for d in rows if d["gain_target"] == g]
        for s in LEVELS:
            ax.plot(xs, [s / x for x in xs], color="0.8", lw=0.7, zorder=0)
            ax.annotate(f"S = {s:,}", (max(1.0, xlo), s / max(1.0, xlo)), xytext=(2, 2), textcoords="offset points",
                        fontsize=6, color="0.45")
        for lo in DECADES[1:]:
            ax.axhline(lo, color="0.6", lw=0.6, ls=":", zorder=0)
        for d in mine:
            yes = d["reached"] == "yes"
            colour = f"C{families.index(d['family'] or '?') % 10}"
            ax.plot(d["p"], d["rho_default"], MARKERS[starts.index(d["start"]) % len(MARKERS)], ms=4.5, alpha=0.75,
                    color=colour, mfc=colour if yes else "none", mec="k" if yes and d["basis_default"] == "fit" else
                    colour, mew=0.7)
        reached = [d for d in mine if d["reached"] == "yes"]
        at = [d for d in every if d["gain_target"] == g]
        unplotted = len(at) - len(mine)
        if reached:
            best = min(reached, key=lambda d: d["p"] * d["rho_default"])
            ax.plot(best["p"], best["rho_default"], "o", mfc="none", mec="red", ms=11)
            ax.set_title(f"gain {g:g}: {len(reached)} reached, {len(mine) - len(reached)} not"
                         + (f" ({unplotted} more: rho infinite, unmeasured or on a stub, or no p)" if unplotted else "")
                         + f";\nleast S {best['p'] * best['rho_default']:.3g} ({best['start']}, {best['method_key']})",
                         fontsize=8)
        elif at:
            top = max(at, key=lambda d: d["gain"])
            ax.set_title(f"gain {g:g}: none of {len(at)} attempts reached it, so no H and no p", fontsize=8)
            ax.text(0.5, 0.5, f"best gain {top['gain']:.3g}\n({top['start']}, {top['method_key']}, {top['record']})",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8, color="0.3")
        else:
            ax.set_title(f"gain {g:g}: no attempts, no scored row on {data}", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(0.8, yhi)
        ax.grid(alpha=0.15, which="major")
        ax.set_xlabel("p = C/H (open: not reached, p a lower bound)")
    axes[0].set_ylabel(f"rho at the console's default (c = {CONSOLE[0]:.3g} gates, k = {CONSOLE[1]:,.0f} bits)")
    keys = [Line2D([], [], marker="o", ls="none", color=f"C{i % 10}", label=f) for i, f in enumerate(families)]
    keys += [Line2D([], [], marker=MARKERS[i % len(MARKERS)], ls="none", color="0.3", label=s)
             for i, s in enumerate(starts)]
    keys.append(Line2D([], [], marker="o", ls="none", mfc="none", mec="red", ms=9, label="least S reached"))
    fig.legend(handles=keys, loc="lower center", ncol=min(len(keys), 7), fontsize=7, bbox_to_anchor=(0.5, 0.04))
    fig.suptitle(f"{p['title']}, {data} data\n(grey: iso-S, S = p rho; dotted: the coverage's decades of rho)",
                 fontsize=10)
    fig.text(0.01, 0.005, p["note"], fontsize=5.5, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    path = out / f"{NAME}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def summary(p: dict, info: dict) -> str:
    """Per data, start and gain target: its target loss, the runs, how many reach it, the least p and the least S
    among them (S on no row resting on a stub), and the best failed attempt."""
    rows = plots.frame(p)
    lines = ["| data | start | gain | own gain | runs | reached | least p (run) | least S (p, rho) | best not reached |",
             "|---|---|---|---|---|---|---|---|---|"]
    for st in info["starts"]:
        f = info["fits"].get(st.data)
        if f is None:
            continue
        for g in GAINS:
            mine = [d for d in rows if (d["data"], d["start"], d["gain_target"]) == (st.data, st.label, g)]
            yes = [d for d in mine if d["reached"] == "yes"]
            no = [d for d in mine if d["reached"] == "no"]
            lp = min(yes, key=lambda d: d["p"]) if yes else None
            fin = [d for d in yes if d["rho_default"] is not None and d["basis_default"] != STUB]
            ls = min(fin, key=lambda d: d["p"] * d["rho_default"]) if fin else None
            bn = max(no, key=lambda d: d["gain"]) if no else None
            lines.append(f"| {st.data} | {st.label} | {g:g} | {frontier_fit.gain(f, st.l0, st.c0):.3f} | {len(mine)} | "
                         f"{len(yes)} | " + (f"{lp['p']:.3g} ({lp['method_key']} {lp['record']})" if lp else "-")
                         + " | " + (f"{ls['p'] * ls['rho_default']:.3g} ({ls['p']:.3g}, {ls['rho_default']:.3g})"
                                    if ls else "-")
                         + " | " + (f"gain {bn['gain']:.3g} ({bn['method_key']} {bn['record']})" if bn else "-")
                         + " |" + ("" if mine else " no rows: its loss is at or below the fit's E"))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = frontier_fit.arguments(rescore.arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0])))
    ap.add_argument("--publish", action="store_true",
                    help=f"put the tables in the evidence store and PUT {frontier_fit.PANEL} and the cut to "
                         f"{console.PANELS_URL}/{PANEL}")
    a = ap.parse_args(argv)
    d, scored = frontier_fit.scored_rows(a)
    me = f"te_core.{NAME.replace('-', '_')}"
    runs, extra, source = frontier_fit.inputs(a, scored, d["shapes"], d, me)
    fp, info = frontier_fit.build(scored, d["shapes"], runs, a.source or source.replace(me, "te_core.frontier_fit", 1),
                                  extra, gains=GAINS)
    p = build(info, source, fp["updated_at"])
    own = coverage(p)
    tables = a.out / "tables"
    frontier_fit.write(fp, info, tables, a.out)
    (tables / f"{NAME}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
    (tables / "coverage.json").write_text(json.dumps(own, allow_nan=False, indent=1) + "\n")
    (tables / "refused.json").write_text(json.dumps(d["refused"], indent=1, sort_keys=True) + "\n")
    (tables / "coverage.md").write_text(coverage_md(own, info["fits"], p["updated_at"], d["refused"],
                                                    unmeasured_rows(p), unmeasured_rows(p, STUB)))
    served = {}
    if a.publish:
        try:
            served = console.served()
        except Exception as e:  # noqa: BLE001 (whatever fails, nothing keeps the PUT from emptying a served panel)
            print(f"not PUT: GET {console.PANELS_URL}?producer=training failed ({type(e).__name__}: {e}), and without "
                  "it no PUT can keep the rows a newer or fuller panel serves", file=sys.stderr)
            return 1
    pub, fpub, held = guarded(p, fp, served)
    cov = own if pub is p else coverage(pub)
    md = coverage_md(cov, info["fits"], p["updated_at"], d["refused"], unmeasured_rows(pub), unmeasured_rows(pub, STUB))
    art = store(tables, {"what": f"{PANEL} uncut, its coverage and {frontier_fit.PANEL} (te_core.algorithm_policy)",
                         "rows": len(p["rows"]), "source": source}) if a.publish else None
    q = cut(pub, art, held)
    (a.out / "console" / f"{NAME}.json").write_bytes(console.body(q))
    (a.out / "console" / f"{frontier_fit.NAME}.json").write_bytes(console.body(fpub))
    png = figure(p, a.out)
    # first: the scorer's refresh posts the first Markdown table under the first line that names coverage
    print(md)
    print(frontier_fit.summary(info))
    if d["refused"]:
        print(f"{len(d['refused'])} valid checkpoints refused, never scored: {frontier_fit.refusals(d['refused'])}")
    print(summary(p, info))
    reached = {dd: sum(r[COLUMNS.index("reached")] == "yes" for r in p["rows"] if r[COLUMNS.index("data")] == dd)
               for dd in panels.DATA}
    every = {dd: sum(r[COLUMNS.index("data")] == dd for r in p["rows"]) for dd in panels.DATA}
    print(f"{PANEL}: {len(p['rows'])} rows uncut (" + ", ".join(f"{dd} {reached[dd]} reached of {every[dd]}"
                                                               for dd in panels.DATA)
          + f"){f' ({art})' if art else ''}; console {len(q['rows'])} rows, {len(console.body(q))} bytes, note "
          f"{len(q['note'])} characters" + (f", served rows kept on {held}" if held else "")
          + f" -> {a.out}: tables/, console/{NAME}.json, console/{frontier_fit.NAME}.json, {png.name}")
    print("board: " + board_line(cov, info["fits"], p["updated_at"], art, d["refused"], unmeasured_rows(pub),
                                 unmeasured_rows(pub, STUB))
          + (f" Kept as served: {held}." if held else ""))
    if a.publish:
        status = []
        for panel in (fpub, q):
            status.append(console.put(panel))
            print(f"PUT {console.PANELS_URL}/{panel['id']}: {status[-1]}")
        return 0 if all(s.startswith("2") for s in status) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
