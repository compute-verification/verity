"""Plot 2's final version: training/algorithm-policy from the integrator's finals table, a row per finals point and gain
target, on te_core.algorithm_policy's columns, cut and figure.

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
  rho_own is unmeasured (`UNTRACED`, drawn on the floor) and only a lower bound on a partial row.

`build` writes DIR/tables/ (algorithm-policy.json uncut, the finals table it read and final.json: each (data, start,
g)'s split and why), DIR/console/algorithm-policy.json (the cut, its note naming no art) and DIR/algorithm-policy.png;
`put` stores DIR/tables/ in the evidence store, cuts again with its art id in the note and PUTs the cut. It PUTs
nothing where the route serves algorithm-policy rows priced by a newer te_core (te_core.algorithm_policy's
`kept_data`). training/frontier-fit stays the search refresh's.
"""
import argparse
import json
import shutil
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from te_core import VERSION, console
from te_core import algorithm_policy as ap
from te_core.frontier import REF_DIRECTIONS
from te_core.gates import UNIT

IX = {c: i for i, c in enumerate(ap.COLUMNS)}
#: the ρ columns a search row carries for its trace's checkpoint, null on a row built for an untraced run
RHO = ("rho_default", "rho_inputs", "reading", "rho_console", "binding_kind", "rests_on", "kinds", "rho_served",
       "rho_own")
TITLE = f"Finals: algorithm vs policy, p = C/H against rho per gain target ({UNIT})"
FINAL, SEARCH = "final", "search split"
#: the finals table's p_any_basis where its p reads the final set (islands/integrator/finals_table.py's FINAL_SET)
FINAL_SET = "final set"


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
    out[IX["basis_default"]] = out[IX["basis_own"]] = ap.UNTRACED
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
                    f"search table of {search['updated_at']}; te_core {VERSION}'s cut and figure"),
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
        png = ap.figure(p, args.out)
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
