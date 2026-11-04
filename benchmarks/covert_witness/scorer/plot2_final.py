"""Plot 2's final version: training/algorithm-policy from the integrator's finals table, a row per finals point and gain
target, on te_core.algorithm_policy's columns and cut, with its own figure (`figure`).

    python plot2_final.py build --finals FINALS_JSON --policy TABLES_JSON --out DIR
    python plot2_final.py put DIR

Run from benchmarks/covert_witness of te_core's checkout (node_loop's RESCORE). FINALS_JSON is
islands/integrator/finals_table.py's out/finals-table.json, TABLES_JSON the search table it read
(te_core.algorithm_policy's tables/algorithm-policy.json). The rules (coordinator 13:12Z, 13:36Z):

- p is p_any: the mean |C| of the point's finished seeds reaching g over H_any's, the least mean |C| over every route;
  the record cell carries p's range over seeds and the finals table's honest-only p (p_honest, H over
  te_core.frontier's REF_DIRECTIONS) with its range or its split;
- the search H-setter of a (data, start, g) is its reached row with the least |C| in the search table. Where that
  point isn't a finals point or has no finished seed (the finals table's p_any_basis, else the same rule on its
  points), or no finals point reaches g, every row there is the search table's row for its run and g, p the search
  split's, so a finals |C| is never divided by a search H; so is a row whose own point has no finished seed. The
  record cell says which split a row's p is and why, and final.json's split_differs lists each (start, g) where the
  finals table's split isn't the one its points give;
- C is its reaching seeds' mean |C| and H H_any's; gain, start_loss and end_loss are their means. A point whose
  finished seeds all miss g is a "no" row at their last checkpoints' mean |C|, p = C/H_any a lower bound;
- every ρ column is the search row's for its run and g, ρ from the search run's trace: S = p·rho_own is null where
  rho_own is unmeasured (`UNTRACED`, or a refused run's row, `REFUSED`; drawn in the strip under its panel) and only a
  lower bound on a partial row.

`build` writes DIR/tables/ (algorithm-policy.json uncut, the finals table it read and final.json: each (data, start,
g)'s split and why), DIR/console/algorithm-policy.json (the cut, its note naming no art) and DIR/algorithm-policy.png
(`figure`, both axes log at the same length per decade);
`put` stores DIR/tables/ in the evidence store, cuts again with its art id in the note and PUTs the cut. It PUTs
nothing where the route serves algorithm-policy rows priced by a newer te_core (te_core.algorithm_policy's
`kept_data`). training/frontier-fit stays the search refresh's.
"""
import argparse
import json
import math
import shutil
import statistics
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from te_core import VERSION, console
from te_core import algorithm_policy as ap
from te_core.frontier import REF_DIRECTIONS
from te_core.gates import UNIT
from matplotlib.lines import Line2D

IX = {c: i for i, c in enumerate(ap.COLUMNS)}
#: the ρ columns a search row carries for its trace's checkpoint, null on a row built for an untraced run
RHO = ("rho_default", "rho_inputs", "reading", "rho_console", "binding_kind", "rests_on", "kinds", "rho_served",
       "rho_own")
TITLE = f"Finals: algorithm vs policy, p = C/H against rho per gain target ({UNIT})"
FINAL, SEARCH = "final", "search split"
#: the finals table's p_any_basis where its p reads the final set (islands/integrator/finals_table.py's FINAL_SET)
FINAL_SET = "final set"
#: how a refused run's row's basis_own opens (v1cycle's REFUSED_ROWS): a row with p from its record and no ρ
REFUSED = "refused: "
#: the figure's inches per decade, on both axes, so an iso-S line S = p·rho_own runs at 45 degrees
DECADE_IN = 1.5
#: the figure's strip under each panel, the rows with p and no finite rho_own, each kind's marker
STRIP_IN, NO_RHO = 0.4, {"untraced": ("x", "reached, not yet traced"), "refused": ("+", "refused (data-dependent)"),
                         "none": ("|", "rho_own infinite or unmeasured")}


def run_of(row: list) -> str:
    return row[IX["record"]].rsplit("@", 1)[0]


def mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def honest(row: list) -> bool:
    return (row[IX["method_key"]] or "").split("/")[0] in REF_DIRECTIONS


def template(by_run: dict[str, dict[float, list]], run: str, g: float) -> list | None:
    """The search row for `run` at `g`; an untraced run has rows only where it reached, so elsewhere its row at another
    target with its ρ columns null and no trace (`UNTRACED`), whose |C|, gain and losses only a finals row replaces."""
    rows = by_run.get(run, {})
    if g in rows:
        return list(rows[g])
    if not rows:
        return None
    out = list(next(iter(rows.values())))
    for c in RHO:
        out[IX[c]] = None
    why = out[IX["basis_own"]]
    out[IX["basis_default"]] = out[IX["basis_own"]] = why if str(why).startswith(REFUSED) else ap.UNTRACED
    out[IX["gain_target"]] = g
    return out


def span(lo_hi: list | None) -> str:
    return f"[{lo_hi[0]:.3g}, {lo_hi[1]:.3g}]" if lo_hi and None not in lo_hi else ""


def searched(row: list, group: list[list], why: str) -> list:
    """A search table row as the final table's, its record cell naming the split and the honest-only p."""
    hh = min((r[IX["C_gates"]] for r in group if r[IX["reached"]] == "yes" and honest(r)), default=None)
    ph = f"p honest {row[IX['C_gates']] / hh:.3g}" if hh else "p honest: no honest route reached g"
    out = list(row)
    out[IX["record"]] = f"{row[IX['record']]}; {SEARCH}: {why}; {ph}"
    return out


def finals(row: list, fr: dict, at: dict, h_any: float) -> list:
    """The finals point's row at g (module docstring) on its search row `row`'s other columns."""
    seeds = list(at["seeds"].values())
    hit = [s for s in seeds if s["reached"]]
    use = hit or seeds
    out = list(row)
    c = mean([s["C_gates"] for s in use])
    out[IX["start_loss"]] = mean([s["l0"] for s in use])
    out[IX["end_loss"]] = mean([s["val_bpb"] for s in use])
    out[IX["gain"]] = mean([s["gain"] for s in use])
    out[IX["reached"]] = "yes" if hit else "no"
    out[IX["C_gates"]], out[IX["H_gates"]], out[IX["p"]] = c, h_any, c / h_any
    seen = f"{at['reached']} of {at['finished']} finished seeds reach g ({at['planned']} planned)"
    hp = fr.get("p_honest", fr.get("p"))
    ph = ("p honest: no honest point's finished seed reaches g" if hp is None else
          f"p honest {hp:.3g} {span(fr.get('p_honest_range', fr.get('p_range')))}".rstrip()
          if fr.get("p_honest_basis", FINAL_SET) == FINAL_SET else f"p honest {hp:.3g} ({SEARCH})")
    out[IX["record"]] = "; ".join(filter(None, (
        row[IX["record"]], f"{FINAL}: {seen}", f"p over seeds {span(fr.get('p_any_range'))}" if hit else
        "p a lower bound", ph)))
    return out


def build(fin: dict, search: dict) -> tuple[dict, dict]:
    """The final table (module docstring) and final.json: each (data, start, g)'s split and why, and the rows whose ρ
    differs from the finals table's (none when it read `search`)."""
    by_run: dict[str, dict[float, list]] = {}
    groups: dict[tuple, list[list]] = {}
    for r in search["rows"]:
        by_run.setdefault(run_of(r), {})[r[IX["gain_target"]]] = r
        groups.setdefault((r[IX["data"]], r[IX["start"]], r[IX["gain_target"]]), []).append(r)
    order = {k: i for i, k in enumerate(groups)}
    points = {p["run"]: p for p in fin["points"]}
    rows, cells, differ, missing, split_differs = [], [], [], [], []
    for t in fin["tables"]:
        g = t["gain"]
        mine = []
        for fr in t["rows"]:
            row = template(by_run, fr["run"], g)
            if row is None:
                missing.append(f"{fr['run']}@{g:g}")
                continue
            if fr.get("rho_own") != row[IX["rho_own"]]:
                differ.append({"run": fr["run"], "g": g, "finals": fr.get("rho_own"), "search": row[IX["rho_own"]]})
            mine.append((fr, row))
        if not mine:
            continue
        key = (t["data"], mine[0][1][IX["start"]], g)
        group = groups.get(key, [])
        yes = [r for r in group if r[IX["reached"]] == "yes"]
        hs = run_of(min(yes, key=lambda r: r[IX["C_gates"]])) if yes else None
        hs_at = (points.get(hs) or {}).get("at", {}).get(f"{g:g}") if hs else None
        h_any = (t.get("H_any") or {}).get("C", {}).get("mean")
        own = (f"the search H-setter {hs[-4:]} is not a finals point" if hs and hs not in points else
               f"the search H-setter {hs[-4:]} has no finished seed" if hs and not (hs_at or {}).get("finished")
               else None)
        basis = t.get("p_any_basis")
        if basis is not None and (basis != FINAL_SET) != bool(own):
            split_differs.append({"start": t["start"], "g": g, "finals_table": basis, "points": own})
        split = own if basis is None else None if basis == FINAL_SET else own or basis
        why = split or (None if h_any else "no finals point's finished seed reaches g")
        got = []
        for fr, row in mine:
            at = points[fr["run"]]["at"][f"{g:g}"]
            if not why and at["finished"]:
                got.append(finals(row, fr, at, h_any))
            elif g not in by_run.get(fr["run"], {}):
                missing.append(f"{fr['run']}@{g:g}")
            else:
                got.append(searched(row, group, why or "this point has no finished seed"))
        got.sort(key=lambda r: (r[IX["reached"]] != "yes", r[IX["p"]] if r[IX["reached"]] == "yes"
                                and r[IX["p"]] is not None else -(r[IX["gain"]] or 0.0)))
        rows += [(order.get(key, len(order)), i, r) for i, r in enumerate(got)]
        cells.append({"data": t["data"], "start": t["start"], "label": key[1], "g": g, "split": SEARCH if why else FINAL,
                      "why": why, "finals_table_basis": basis, "search_H": hs,
                      "finals_H_any": (t.get("H_any") or {}).get("run"), "rows": len(got),
                      "search_rows": sum(SEARCH in r[IX["record"]] for r in got)})
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    p = {"id": ap.PANEL, "title": TITLE, "kind": "table", "updated_at": stamp,
         "source": (f"scorer/plot2_final.py: the finals table built {fin['built_at']} (te_core {fin['te_core']}) on the "
                    f"search table of {search['updated_at']}; te_core {VERSION}'s cut"),
         "note": "", "columns": ap.COLUMNS, "rows": [r for *_, r in sorted(rows, key=lambda x: x[:2])]}
    return p | {"note": note(cells)}, {"cells": cells, "rho_differs": differ, "no_search_row": missing,
                                       "split_differs": split_differs}


def note(cells: list[dict], art: str | None = None, tier: str | None = None) -> str:
    """The panel's note within the route's `console.NOTE_MAX`: the split's cells by start, else their count, and the
    cut's words, else only that it is one; the headline's label and the art id stay whole."""
    split = {}
    for c in cells:
        if c["split"] == SEARCH:
            split.setdefault(c["start"], []).append(f"{c['g']:g}x")
    every = {c["start"]: [f"{x['g']:g}x" for x in cells if x["start"] == c["start"]] for c in cells}
    where = "; ".join(f"{s} {'all g' if gs == every[s] else ', '.join(gs)}" for s, gs in split.items())
    count = f"{sum(c['split'] == SEARCH for c in cells)} of {len(cells)} cells"
    cut = "" if not tier else " Cut: coverage" + ("." if tier == "coverage" else " + per start, g" + (
        ", family" if tier == "family" else "") + ": (p, rho) front, best no.")

    def text(where: str, cut: str) -> str:
        return ("Finals: a row per finals point, g. p = p_any: its reaching seeds' mean C over H, the least mean C of "
                f"any route; honest p in record. Search split where the search H-setter has no finished seed: {where} "
                f"(record says). rho_own from the search run's trace, the start's own session, {ap.OWN_K_LABEL}. "
                "S = p rho_own, none where unmeasured." + cut + (f" Tables {art}." if art else ""))
    tries = [(where, cut), (count, cut), (count, " Cut." if tier else "")]
    return next((t for t in (text(*x) for x in tries) if len(t) <= console.NOTE_MAX), text(*tries[-1]))


def cut(p: dict, cells: list[dict], art: str | None = None) -> dict:
    """te_core.algorithm_policy's `cut` with this table's note: every row when it fits the route, else `kept`'s."""
    q = ap.checked(p | {"columns": ap.CONSOLE_COLUMNS, "rows": ap.console_rows(p["rows"]), "note": note(cells, art)})
    if len(console.body(q)) <= console.BODY_MAX:
        return q
    cov = {c["row"] for c in ap.coverage(p) if c["row"] is not None}
    for tier in ap.TIERS:
        q = ap.checked(p | {"columns": ap.CONSOLE_COLUMNS, "rows": ap.console_rows(ap.kept(p["rows"], tier, cov)),
                            "note": note(cells, art, tier)})
        if len(console.body(q)) <= console.BODY_MAX:
            return q
    raise SystemExit(f"plot2_final: the coverage's rows alone are {len(console.body(q))} bytes, over {console.BODY_MAX}")


def colour_key(d: dict) -> str:
    """A row's family, else its method's direction (a v3 record names no family)."""
    return d["family"] or (d["method_key"] or "").split("/")[0] or "unnamed"


def no_rho(d: dict) -> str:
    return ("untraced" if d["basis_own"] == ap.UNTRACED else
            "refused" if str(d["basis_own"]).startswith(REFUSED) else "none")


def bounds(xs: list[float], *also: float) -> tuple[float, float]:
    """Log10 limits at half decades around `xs` and `also`, at least two decades apart."""
    xs = [*xs, *also] or [1.0]
    a = math.floor(2 * math.log10(min(xs) / 1.2)) / 2
    return a, max(math.ceil(2 * math.log10(max(xs) * 1.2)) / 2, a + 2)


def figure(p: dict, out: Path) -> Path:
    """p against the headline rho_own on the headline data (te_core.algorithm_policy's `headline`), a panel per gain
    target, both axes log at DECADE_IN inches a decade, so the iso-S lines S = p·rho_own run at 45 degrees: a colour per
    family (`colour_key`), a marker per start, filled where reached and open at a failed attempt's lower bound on p,
    faded where p is the search split's, a caret up from a partial row, the least S ringed; the rows with p and no
    finite rho_own (`NO_RHO`) in the strip under their panel. The axes are placed in inches, so nothing overlaps."""
    plt, fs = ap.plt, 8
    y, b, data = "rho_own", "basis_own", ap.headline(p)
    every = [d for d in ap.plots.frame(p) if d["data"] == data and d["p"] is not None]
    drawn = [d for d in every if d[y] is not None and d[b] != ap.STUB]
    strip = [d for d in every if d[y] is None]
    keys = sorted({colour_key(d) for d in every})
    starts = list(dict.fromkeys(d["start"] for d in every))
    (x0, x1), (y0, y1) = bounds([d["p"] for d in every], 1.0), bounds([d[y] for d in drawn])
    w, h, left, gap, right = (x1 - x0) * DECADE_IN, (y1 - y0) * DECADE_IN, 1.0, 0.3, 0.3
    width = left + len(ap.GAINS) * w + (len(ap.GAINS) - 1) * gap + right
    wrap = lambda text, size: textwrap.wrap(text, int(0.95 * width * 72 / (0.6 * size)))
    own = {d["start"]: (d["own_c_gates"], d["own_k_bits"]) for d in every if d.get("own_c_gates") is not None}
    head = [*wrap(f"{p['title']}, {data} data", fs + 2),
            *wrap(f"Headline rho_own: {ap.POLICIES['rho_own']}.", fs),
            *wrap("c, packed k per start: " + "; ".join(f"{s} {c:.3g} gates, {k:,} bits" for s, (c, k) in own.items())
                  + ".", fs),
            *wrap("Both axes log at the same length per decade, so the grey iso-S lines (S = p·rho_own) run at 45°. "
                  f"Sensitivities in the tables: rho_default, {ap.POLICIES['rho_default']}; rho_served, "
                  f"{ap.POLICIES['rho_served']}.", fs)]
    marks = [Line2D([], [], marker="o", ls="none", color=f"C{i % 10}", label=k) for i, k in enumerate(keys)]
    marks += [Line2D([], [], marker=ap.MARKERS[i % len(ap.MARKERS)], ls="none", color="0.3", label=s)
              for i, s in enumerate(starts)]
    marks += [Line2D([], [], marker="o", ls="none", color="0.3", label="filled: reached"),
              Line2D([], [], marker="o", ls="none", mfc="none", color="0.3", label="open: not reached (p a lower bound)"),
              Line2D([], [], marker="o", ls="none", color="0.3", alpha=0.35, label=f"faded: p on the {SEARCH}"),
              Line2D([], [], marker="o", ls="none", mfc="none", mec="red", ms=9, label="least S")]
    if any(d[b] in ap.LOWER for d in drawn):
        marks.append(Line2D([], [], marker=ap.LOWER[ap.PARTIAL], ls="none", color="0.3", ms=8,
                            label="partial: rho a lower bound"))
    marks += [Line2D([], [], marker=m, ls="none", color="0.3", label=f"strip: {what}")
              for k, (m, what) in NO_RHO.items() if any(no_rho(d) == k for d in strip)]
    ncol = max(1, int(width // 2.4))
    note = wrap(p["note"], 6)
    line = lambda size: 1.45 * size / 72
    tail = 0.35 + 0.45 + -(-len(marks) // ncol) * line(fs) * 1.3 + 0.15 + len(note) * line(6) + 0.1
    top = 0.1 + len(head) * line(fs) + 0.15 + 3 * line(fs) + 0.1
    height = top + h + 0.06 + STRIP_IN + tail
    fig = plt.figure(figsize=(width, height))
    frac = lambda x, y_, w_, h_: [x / width, 1 - (y_ + h_) / height, w_ / width, h_ / height]
    for i, text in enumerate(head):
        fig.text(0.5, 1 - (0.1 + (i + 0.8) * line(fs)) / height, text, ha="center", va="baseline",
                 fontsize=fs + 2 if i == 0 else fs)
    lo, hi = 10 ** x0, 10 ** x1
    for i, g in enumerate(ap.GAINS):
        x = left + i * (w + gap)
        ax = fig.add_axes(frac(x, top, w, h))
        sx = fig.add_axes(frac(x, top + h + 0.06, w, STRIP_IN), sharex=ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(10 ** y0, 10 ** y1)
        ax.set_aspect("equal")
        for k in range(math.floor(x0 + y0), math.ceil(x1 + y1) + 1):
            ax.plot([lo, hi], [10 ** k / lo, 10 ** k / hi], color="0.82", lw=0.7, zorder=0)
            at = next(((10 ** px, 10 ** (k - px)) for px in (x0 + 0.12, k - y1 + 0.12)
                       if x0 < px < x1 - 0.3 and y0 + 0.3 < k - px < y1), None)
            if at:
                ax.annotate(f"S = 1e{k}", at, rotation=-45, fontsize=6, color="0.5", ha="left", va="bottom",
                            rotation_mode="anchor")
        ax.grid(alpha=0.2, which="major")
        mine = [d for d in drawn if d["gain_target"] == g]
        for d in mine:
            yes, c = d["reached"] == "yes", f"C{keys.index(colour_key(d)) % 10}"
            ax.plot(d["p"], d[y], ap.MARKERS[starts.index(d["start"]) % len(ap.MARKERS)], ms=5, color=c,
                    mfc=c if yes else "none", mec=c, mew=0.8, alpha=0.35 if SEARCH in d["record"] else 0.85)
            if d[b] in ap.LOWER:
                ax.plot(d["p"], d[y], marker=ap.LOWER[d[b]], ms=9, color=c, mew=1.0)
        whole = [d for d in mine if d["reached"] == "yes" and d[b] not in ap.LOWER]
        best = min(whole, key=lambda d: d["p"] * d[y]) if whole else None
        if best:
            ax.plot(best["p"], best[y], "o", mfc="none", mec="red", ms=12, mew=1.2)
        bare = [d for d in strip if d["gain_target"] == g]
        for d in bare:
            sx.plot(d["p"], 0.5, NO_RHO[no_rho(d)][0], ms=6, mew=1.3, color=f"C{keys.index(colour_key(d)) % 10}",
                    alpha=0.35 if SEARCH in d["record"] else 0.9)
        sx.set_ylim(0, 1)
        sx.set_yticks([])
        sx.grid(alpha=0.2, which="major", axis="x")
        ax.tick_params(labelbottom=False, labelsize=fs - 1)
        sx.tick_params(labelsize=fs - 1)
        if i:
            ax.tick_params(labelleft=False)
        reached = sum(d["reached"] == "yes" for d in mine) + sum(d["reached"] == "yes" for d in bare)
        n = len(mine) + len(bare)
        more = ", ".join(f"{sum(no_rho(d) == k for d in bare)} {NO_RHO[k][1]}" for k in NO_RHO
                         if any(no_rho(d) == k for d in bare))
        title = [f"gain {g:g}: {reached} of {n} rows reached", f"in the strip: {more}" if more else "no row without rho",
                 f"least S {best['p'] * best[y]:.3g} ({best['start']}, {best['method_key']})" if best else
                 "no least S: no reached row has a whole rho_own"]
        ax.set_title("\n".join(title), fontsize=fs, loc="left")
    fig.axes[0].set_ylabel("\n".join(textwrap.wrap(f"rho_own, the start's own session ({ap.OWN_K_LABEL})",
                                                    int(0.9 * h * 72 / (0.6 * fs)))), fontsize=fs)
    fig.axes[1].set_ylabel("no\nrho_own", fontsize=fs - 1, rotation=0, ha="right", va="center")
    under = top + h + 0.06 + STRIP_IN + 0.35
    fig.text(0.5, 1 - under / height, "p = C/H, relative to the best route found (H)", ha="center", va="top",
             fontsize=fs + 1)
    fig.legend(handles=marks, loc="upper center", ncol=ncol, fontsize=fs - 1, frameon=False,
               bbox_to_anchor=(0.5, 1 - (under + 0.3) / height))
    for i, text in enumerate(note):
        fig.text(0.01, (0.1 + (len(note) - 1 - i) * line(6)) / height, text, fontsize=6, va="bottom")
    path = out / f"{ap.NAME}.png"
    fig.savefig(path, dpi=130)
    decade = fig.axes[0].transData.transform([(lo, 10 ** y0), (lo * 10, 10 ** (y0 + 1))])
    plt.close(fig)
    dx, dy = decade[1] - decade[0]
    if not math.isclose(dx, dy, rel_tol=1e-3):
        raise SystemExit(f"plot2_final: a decade is {dx:.1f} px on x and {dy:.1f} px on y; the figure needs them equal")
    return path


def summary(p: dict, cells: list[dict]) -> str:
    """Per (data, start, g): its split, rows, reached, the least p and the least S (p, rho_own) over rows not partial."""
    frame = ap.plots.frame(p)
    lines = ["| data | start | g | split | rows | reached | least p (record) | least S (p, rho_own) |",
             "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        mine = [d for d in frame if (d["data"], d["start"], d["gain_target"]) == (c["data"], c["label"], c["g"])]
        yes = [d for d in mine if d["reached"] == "yes" and d["p"] is not None]
        lp = min(yes, key=lambda d: d["p"]) if yes else None
        fin = [d for d in yes if d["rho_own"] is not None and d["basis_own"] not in (ap.STUB, *ap.LOWER)]
        ls = min(fin, key=lambda d: d["p"] * d["rho_own"]) if fin else None
        split = c["split"] + (f" ({c['why']})" if c["why"] else "")
        lines.append(f"| {c['data']} | {c['start']} | {c['g']:g} | {split} | {len(mine)} | {len(yes)} | "
                     + (f"{lp['p']:.3g} ({lp['record'].split(';')[0]})" if lp else "-")
                     + " | " + (f"{ls['p'] * ls['rho_own']:.3g} ({ls['p']:.3g}, {ls['rho_own']:.3g})" if ls else "-")
                     + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = a.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--finals", type=Path, required=True)
    b.add_argument("--policy", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("put")
    u.add_argument("out", type=Path)
    args = a.parse_args(argv)
    tables = args.out / "tables"
    if args.cmd == "build":
        fin, search = json.loads(args.finals.read_text()), json.loads(args.policy.read_text())
        p, info = build(fin, search)
        tables.mkdir(parents=True, exist_ok=True)
        (args.out / "console").mkdir(exist_ok=True)
        (tables / f"{ap.NAME}.json").write_text(json.dumps(p, allow_nan=False) + "\n")
        (tables / "final.json").write_text(json.dumps(info, indent=1) + "\n")
        shutil.copyfile(args.finals, tables / "finals-table.json")
        q = cut(p, info["cells"])
        (args.out / "console" / f"{ap.NAME}.json").write_bytes(console.body(q))
        png = figure(p, args.out)
        shutil.copyfile(png, tables / png.name)
        print(summary(p, info["cells"]))
        print(f"{ap.PANEL} finals: {len(p['rows'])} rows uncut, console {len(q['rows'])} rows, "
              f"{len(console.body(q))} bytes, note {len(q['note'])} characters; "
              f"{sum(c['split'] == SEARCH for c in info['cells'])} of {len(info['cells'])} cells on the search split; "
              f"rho differs from the finals table's on {len(info['rho_differs'])} rows; "
              f"the split differs from the finals table's at {len(info['split_differs'])} (start, g); "
              f"{len(info['no_search_row'])} finals rows with no search row -> {args.out}")
        return 0
    p = json.loads((tables / f"{ap.NAME}.json").read_text())
    cells = json.loads((tables / "final.json").read_text())["cells"]
    served = console.served()
    if keep := ap.kept_data(p, served.get(ap.PANEL)):
        print(f"not PUT: the route serves {ap.PANEL} rows on {', '.join(keep)} priced by a newer te_core than {VERSION}")
        return 1
    art = ap.store(tables, {"what": f"{ap.PANEL} finals (scorer/plot2_final.py), uncut, and the finals table it read",
                            "rows": len(p["rows"]), "source": p["source"]})
    q = cut(p, cells, art)
    (args.out / "console" / f"{ap.NAME}.json").write_bytes(console.body(q))
    status = console.put(q)
    print(f"{datetime.now(UTC).strftime('%H:%M:%SZ')} PUT {console.PANELS_URL}/{ap.PANEL}: {status} ({art}; "
          f"{len(q['rows'])} rows, {len(console.body(q))} bytes)")
    return 0 if status.startswith("2") else 1


if __name__ == "__main__":
    sys.exit(main())
