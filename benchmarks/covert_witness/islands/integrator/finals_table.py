"""The finals table: every finals point (FINALS.json) scored on the sealed final set, per (start, gain) and seed, with
p = C/H against the best route found and the search run's ρ beside it.

    python islands/integrator/finals_table.py fetch --launches JOB.jsonl --to DIR [--arts ARTS.json]
    python islands/integrator/finals_table.py build --records DIR --policy TABLES --launches JOB.jsonl --out OUT
                                                    [--finals FINALS.json] [--manifest M] [--ladder-fit CAP.json]

Both run from benchmarks/covert_witness, on a node (the agent VM only edits). A finals rerun runs in final mode
(CW_FINAL=1): every checkpoint and start_bpb is scored on the sealed final set, and nothing ingests it into the scoring
ledger. `fetch` copies each finished rerun's record from the evidence store into DIR/RUN/out/ (`FILES`: the scored
run.json, the submission's meta.json and the witness), since a node's copy of a finished run's big files may be
evicted once preserved; `--arts` gives each rerun's run-record art id where the store's lookup by run id is
unavailable. `build` reads te_core and changes nothing in it:

- the reruns are the finals job's launches (`--launches`, its launch file): per point and seed, the last launch not
  superseded by its step check or exited 75;
- each rerun's checkpoints are its record as te_core.ledger.ingest writes a search run's, into a scratch ledger, each
  with its |C| in gates (te_core.formalism's `measure`, as the search rows' and the scorer's untraced rows'), or the
  reason none is scored (an invalid record, unscored, te_core.frontier's `left_out` or a refusal);
- a seed's final-set target at gain g is te_core.frontier_fit's `target` on the start's data fit, E and γ unchanged
  from the scorer's stored tables (`--policy`'s frontier-fit.json), and l0 that seed's own start_bpb on the final set:
  the final set is the search pool's distribution, so the fit carries over and only l0 moves. The seed reaches g at its
  first checkpoint at or below its target (C that checkpoint's |C|), else it is "not reached" at its last;
- per point: the seeds reached (k of the seeds finished, of those planned: its FINALS.json reruns but a seed dropped or
  skipped there and never launched), C's mean and spread over the reached seeds;
- H at (start, gain) is the point with the least mean C over its reached seeds, every route's finals seeds counted,
  covert ones included (the scorer's rule: te_core.frontier's `target` and te_core.algorithm_policy's H), with its own
  spread; the headline p, `p_any` (plot 2's "p relative to the best route found"), is C/H. `H_honest`, the least
  over the honest points (`honest`: its island a reference direction, te_core.frontier's `REF_DIRECTIONS`), and
  `p_honest` ("p against the honest reference") are beside it as a sensitivity;
- each H has its search split's (`search_h`): over every route the scorer's H_gates and the reached row at it, over
  the honest ones the reached row with the least C_gates among the reference routes (a row's method_key direction,
  as te_core.plots reads one). Where that H's point has no finished finals seed when the table builds (not a finals
  point, not rerun, or its seeds unfinished), the final set's p would read against a different H, so every row of
  that (start, gain) takes the search split's p, the scorer's stored p (`p_honest`: C_gates over the honest H's),
  labelled why (`p_basis`), and no finals C is divided by a different H (`p_any_final` and `p_honest_final`, the
  final set's p where the group reads it, are then None); where that point finished but no seed reached g,
  `H_caveat` says so;
- ρ is the search run's at its first checkpoint reaching g in the scorer's stored table (`--policy`'s
  algorithm-policy.json): ρ is a property of the computation, not of the data (`RHO_LABEL`); where the search run
  didn't reach g it is its last checkpoint's, labelled so, and S = p_any·ρ on the headline rho_own (rho_default, so
  labelled, from a table older than rho_own).

Nothing is left out silently: each point at each gain carries its seeds' states and every reason (`label`), and a
point the markdown leaves out of a (start, gain) is counted there with why. OUT gets finals-table.json and
finals-table.md.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from te_core import VERSION, formalism, frontier, frontier_fit, policy
from te_core.algorithm_policy import GAINS
from te_core.ledger import Ledger, ingest

HERE = Path(__file__).resolve().parent
FINALS = HERE / "FINALS.json"
SCHEMA = "covert-witness/finals-table/v1"
#: the files of a rerun's record `fetch` copies, under its run dir
FILES = ("out/run/run.json", "out/meta.json", "out/witness.json")
SEEDS = ("own", "101", "102")
RHO_LABEL = "rho from the search run's trace"
#: FINALS.json's label for a point left without its three seeds
SINGLE = "single seed: a rerun doesn't fit before the nodes end"
#: the headline p's basis where the search split's H has a finished finals seed
FINAL_SET = "final set"
#: a launched rerun's states while its record isn't in (`checkpoints`', and `fetch`'s why)
NOT_YET = ("no finished record", "no run record in the store")
#: the two H, the headline's first: (key, the scorer's name for its p)
KINDS = (("any", "p relative to the best route found"), ("honest", "p against the honest reference"))


def research() -> list[str]:
    exe = shutil.which("research")
    return [exe] if exe else [sys.executable, "-m", "research"]


def reruns(launches: Path) -> dict[str, dict[str, str]]:
    """Per point (its search run) and seed, the rerun the finals job launched last for it, unless its step check
    superseded it or it exited 75."""
    rows = [json.loads(ln) for ln in launches.read_text().splitlines() if ln.strip()]
    gone = {r["run"] for r in rows if r.get("superseded") or r.get("exited_75")}
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if "node" in r and r.get("run") and r.get("rc", 0) == 0 and r["run"] not in gone:
            point, _, seed = r["key"].rpartition(" seed ")
            out.setdefault(point, {})[seed] = r["run"]
    return out


def fetch(launches: Path, to: Path, arts: dict[str, str]) -> dict[str, str]:
    """Each rerun's record (`FILES`) from the store into `to`/RUN/, and why for each not fetched."""
    why = {}
    for seeds in reruns(launches).values():
        for run in seeds.values():
            d = to / run
            if (d / FILES[0]).exists():
                continue
            art = arts.get(run)
            if art is None:
                p = subprocess.run([*research(), "data", "show", run, "--json"], capture_output=True, text=True)
                try:
                    art = (json.loads(p.stdout).get("outputs") or {}).get("run_record")
                except ValueError:
                    art = None
            if not art:
                why[run] = "no run record in the store (still running or waiting, or not uploaded yet)"
                continue
            p = subprocess.run([*research(), "data", "fetch", art, "--to", str(d),
                                *(x for f in FILES for x in ("--path", f))], capture_output=True, text=True)
            if not (d / FILES[0]).exists():
                why[run] = f"its record {art} has no {FILES[0]} (fetch rc {p.returncode}: {p.stderr.strip()[-200:]})"
    return why


def honest(point: dict) -> bool:
    """te_core.frontier's reference rule (`REF_DIRECTIONS`), on the point's island: the pilot's `class` is its own
    classification, not honesty."""
    return point.get("island") in frontier.REF_DIRECTIONS


def checkpoints(run_dir: Path, point: dict, run: str, shape: dict, ladder=None, pins: tuple[str, ...] = ()
                ) -> tuple[list[dict], str | None]:
    """The rerun's checkpoints by step, each its step, final-set bpb, start_bpb, counted MACs and FLOPs (the witness's
    curve) and |C| in gates, or why it has none; and why the rerun has no record, else None. A checkpoint measured at a
    commit outside `pins` says so (`version`), as te_core.frontier's `other-version`, and is still scored."""
    if not (run_dir / FILES[0]).exists():
        return [], "no finished record"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            subs = ingest(run_dir / "out" / "run", Ledger(Path(tmp) / "ledger", "finals-table"),
                          direction=point.get("island") or "", mechanism=point.get("mechanism") or "", run_id=run)
    except (ValueError, KeyError) as e:
        return [], f"its record doesn't ingest ({type(e).__name__}: {e})"
    w = run_dir / FILES[2]
    flops = {c["step"]: c.get("train_flops") for c in json.loads(w.read_text()).get("curve", [])} if w.exists() else {}
    out = []
    for r in sorted(subs, key=lambda r: r["checkpoint_step"]):
        cp = {"step": r["checkpoint_step"], "val_bpb": r.get("val_bpb"), "start_bpb": r.get("start_bpb"),
              "macs": r.get("macs"), "flops": flops.get(r["checkpoint_step"]), "gates": None}
        if pins and r.get("measurement_commit") not in pins:
            cp["version"] = f"measured at {str(r.get('measurement_commit'))[:12]}, outside the pin's version"
        why = ("invalid: " + "; ".join(map(str, r["invalid"])) if r.get("invalid") else
               "unscored" if r.get("val_bpb") is None else frontier.left_out(r, shape, ladder))
        if why is None:
            try:
                cp["gates"] = formalism.measure(r, shape).gates
            except formalism.Refused as e:
                why = f"refused: {e}"
            except ValueError as e:
                why = f"unmeasured: {e}"
        out.append(cp | ({"why": why} if why else {}))
    return out, None


def seed_at(cps: list[dict], f: dict | None, g: float) -> dict:
    """One seed at gain `g`: its l0 (the final set's start_bpb), target, and the first scored checkpoint reaching it,
    else 'not reached' at its last."""
    scored = [c for c in cps if c["gates"] is not None]
    l0 = next((c["start_bpb"] for c in cps if c.get("start_bpb") is not None), None)
    if f is None:
        return {"reached": False, "why": "its start's data has no fit in the scorer's tables"}
    if l0 is None:
        return {"reached": False, "why": "its record has no start_bpb"}
    t = frontier_fit.target(f, l0, g)
    if t is None:
        return {"l0": l0, "reached": False, "why": f"its l0 {l0:g} is at or below the fit's E {f['E']:g}"}
    out = {"l0": l0, "target": t}
    if not scored:
        return out | {"reached": False, "why": "no scored checkpoint: " + "; ".join(
            sorted({c.get("why", "") for c in cps})) if cps else "no checkpoint"}
    hit = next((c for c in scored if c["val_bpb"] <= t), None)
    c = hit or scored[-1]
    return out | {"reached": hit is not None, "step": c["step"], "val_bpb": c["val_bpb"], "C_gates": c["gates"],
                  "macs": c.get("macs"), "flops": c.get("flops"),
                  "gain": frontier_fit.gain_from(f, l0, c["val_bpb"]) if c["val_bpb"] > f["E"] else math.inf}


def spread(xs: list[float]) -> dict:
    return {"mean": statistics.fmean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else None, "min": min(xs),
            "max": max(xs), "n": len(xs)}


def fits_of(panel: dict) -> dict[str, dict]:
    """Per data, the scorer's stored fit (training/frontier-fit's `fit` rows): E, A, γ and its status."""
    ix = {c: i for i, c in enumerate(panel["columns"])}
    return {r[ix["data"]]: {"E": r[ix["E_bpb"]], "A": r[ix["A"]], "gamma": r[ix["gamma"]],
                            "points": r[ix["points"]], "status": r[ix["status"]]}
            for r in panel["rows"] if r[ix["row"]] == "fit" and r[ix["E_bpb"]] is not None}


def policy_rows(panel: dict) -> dict[tuple[str, float], dict]:
    """The scorer's stored algorithm-policy rows by (search run, gain target)."""
    cols = panel["columns"]
    out = {}
    for r in panel["rows"]:
        d = dict(zip(cols, r))
        out[(d["record"].rpartition("@")[0], float(d["gain_target"]))] = d
    return out


def rho_of(row: dict | None, run: str, g: float) -> dict:
    """The search run's ρ at `g` from the scorer's row (`policy_rows`), with its label."""
    if row is None:
        return {"rho_own": None, "label": f"no row for {run} in the scorer's table (left out or refused there, or "
                                          "not ingested at its pin)"}
    at = "at its first checkpoint reaching it" if row["reached"] == "yes" else \
        f"at its last checkpoint: the search run didn't reach {g:g}x"
    reads = "rho_own" if "rho_own" in row else "rho_default"
    own = "" if reads == "rho_own" else f"; the table (te_core {row.get('te_core')}) has no rho_own, so S reads rho_default"
    return {k: row.get(k) for k in ("rho_own", "basis_own", "rho_default", "basis_default", "record", "te_core")} | {
        "S_reads": reads, "label": f"{RHO_LABEL}, {at}{own}"}


def search_h(prows: dict[tuple[str, float], dict], label: str | None, g: float, kind: str) -> dict | None:
    """The search split's H at the scorer's start `label` and gain `g`: over every route (`kind` "any") the scorer's
    H_gates and the reached row at it (record None if no stored row is), over the honest ones the reached row with the
    least C_gates among the reference routes (its method_key's direction in te_core.frontier's `REF_DIRECTIONS`)."""
    group = [r for r in prows.values() if label is not None and r.get("start") == label and float(r["gain_target"]) == g]
    best = min((r for r in group if r.get("reached") == "yes" and r.get("C_gates") and (
        kind == "any" or str(r.get("method_key") or "").split("/")[0] in frontier.REF_DIRECTIONS)),
        key=lambda r: r["C_gates"], default=None)
    c = best and best["C_gates"]
    if kind == "any":
        c = next((r["H_gates"] for r in group if r.get("H_gates")), c)
        best = best if best and math.isclose(best["C_gates"], c, rel_tol=1e-9) else None
    return {"record": best and best["record"], "C_gates": c} if c else None


def p_basis(s: dict | None, hp: dict | None, k: str) -> str:
    """What every row's p at gain `k` reads: the final set (`FINAL_SET`), unless the search split's H `s` (its finals
    point `hp`, None if it isn't one) has no finished finals seed, when it is the search split's p, labelled why."""
    if s is None or (hp and hp["at"][k]["finished"]):
        return FINAL_SET
    if hp is None:
        return f"search split, H ({s['record'] or 'C %.3g gates, no stored row at it' % s['C_gates']}) is not a finals point"
    if hp["seeds"]:
        return "search split, H's seeds " + ", ".join(
            f"{seed} ({v['rerun'] or 'not launched'}" + ("" if not v["rerun"] or v["state"].startswith(NOT_YET) else
                                                          f": {v['state']}") + ")"
            for seed, v in hp["seeds"].items()) + " unfinished"
    return f"search split, H ({hp['point']}, {s['record']}) not rerun in the finals"


def build(finals: dict, runs: dict[str, dict[str, str]], records: Path, fits: dict[str, dict],
          prows: dict[tuple[str, float], dict], shapes: dict[str, dict], ladder=None, fetched: dict | None = None,
          pins: tuple[str, ...] = ()) -> dict:
    """The table (module docstring): per point its seeds at every gain, per (start, gain) H and the rows."""
    fetched = fetched or {}
    points = []
    for pt in finals["points"]:
        shape = shapes.get(pt["start"])
        data = frontier_fit.data_of_shape(shape) if shape else None
        launched = runs.get(pt["run"], {})
        planned = [x["seed"] for x in pt.get("reruns", []) if x["seed"] in SEEDS
                   and (x["seed"] in launched or not {"dropped", "skipped"} & x.keys())]
        seeds = {}
        for s in planned:
            run = launched.get(s)
            if run is None:
                seeds[s] = {"rerun": None, "state": "not launched"}
                continue
            if shape is None:
                seeds[s] = {"rerun": run, "state": f"no shape for start {pt['start']} in the starts manifest"}
                continue
            cps, why = checkpoints(records / run, pt, run, shape, ladder, pins)
            seeds[s] = {"rerun": run, "state": fetched.get(run) or why or "finished", "checkpoints": cps}
        finished = [s for s, v in seeds.items() if v["state"] == "finished"]
        versions = sorted({f"seed {s} {c['version']}" for s in finished for c in seeds[s]["checkpoints"]
                           if c.get("version")})
        at = {}
        for g in GAINS:
            one = {s: seed_at(seeds[s]["checkpoints"], fits.get(data), g) for s in finished}
            hit = [v["C_gates"] for v in one.values() if v["reached"]]
            at[str(g)] = {"seeds": one, "reached": len(hit), "finished": len(finished), "planned": len(planned),
                          "C": spread(hit) if hit else None,
                          "rho": rho_of(prows.get((pt["run"], float(g))), pt["run"], g)}
        label = [x for x in (pt.get("label"),) if x]
        if len(finished) < len(planned):
            label.append(f"{len(finished)} of {len(planned)} seeds finished: " + "; ".join(
                f"seed {s} {v['state']}" + (f" ({v['rerun']})" if v.get("rerun") else "")
                for s, v in seeds.items() if v["state"] != "finished"))
        if not planned:
            label.append("no finals rerun planned")
        label += versions
        points.append({"point": pt["point"], "run": pt["run"], "start": pt["start"], "data": data,
                       "island": pt.get("island"), "mechanism": pt.get("mechanism"), "class": pt.get("class"),
                       "honest": honest(pt), "rank": pt.get("rank"),
                       "seeds": {s: {k: v for k, v in x.items() if k != "checkpoints"} for s, x in seeds.items()},
                       "checkpoints": {s: x.get("checkpoints", []) for s, x in seeds.items()},
                       "at": at, "label": label})
    tables = []
    for start in sorted({p["start"] for p in points}):
        mine = [p for p in points if p["start"] == start]
        by_run = {p["run"]: p for p in mine}
        scorers = next((r["start"] for p in mine for g in GAINS if (r := prows.get((p["run"], float(g))))), None)
        for g in GAINS:
            k = str(g)
            got = [p for p in mine if p["at"][k]["C"]]
            hf = {"any": min(got, key=lambda p: p["at"][k]["C"]["mean"], default=None),
                  "honest": min((p for p in got if p["honest"]), key=lambda p: p["at"][k]["C"]["mean"], default=None)}
            hs, hp, basis = {}, {}, {}
            for kind, _ in KINDS:
                hs[kind] = search_h(prows, scorers, float(g), kind)
                rec = (hs[kind] or {}).get("record")
                hp[kind] = by_run.get(rec.rpartition("@")[0]) if rec else None
                basis[kind] = p_basis(hs[kind], hp[kind], k)
            rows, left = [], {}
            for p in mine:
                a = p["at"][k]
                searched = (a["rho"].get("record") or "") and "first checkpoint" in a["rho"]["label"]
                if not (a["reached"] or searched):
                    why = ("no seed finished" if not a["finished"] else
                           f"neither the search run nor any of its {a['finished']} finished seeds reached {g:g}x")
                    left[why] = left.get(why, 0) + 1
                    continue
                c = a["C"]
                rho = a["rho"].get(a["rho"].get("S_reads", "rho_own"))
                row = {"point": p["point"], "run": p["run"], "island": p["island"], "honest": p["honest"],
                       "reached": f"{a['reached']} of {a['finished']} finished ({a['planned']} planned)", "C": c,
                       "flops": spread(fl) if (fl := [v["flops"] for v in a["seeds"].values()
                                                      if v["reached"] and v.get("flops") is not None]) else None,
                       "bpb": spread(b) if (b := [v["val_bpb"] for v in a["seeds"].values() if v["reached"]]) else None,
                       "below_target": spread(b) if (b := [v["target"] - v["val_bpb"] for v in a["seeds"].values()
                                                          if v["reached"]]) else None,
                       "rho_own": a["rho"].get("rho_own"), "basis_own": a["rho"].get("basis_own"),
                       "rho_default": a["rho"].get("rho_default"), "basis_default": a["rho"].get("basis_default"),
                       "S_reads": a["rho"].get("S_reads"), "rho_label": a["rho"]["label"],
                       "rho_record": a["rho"].get("record"), "label": p["label"]}
                srow = prows.get((p["run"], float(g)))
                for kind, _ in KINDS:
                    on = basis[kind] == FINAL_SET
                    hc = on and hf[kind] and hf[kind]["at"][k]["C"]
                    fin = c["mean"] / hc["mean"] if c and hc else None
                    rng = [c["min"] / hc["max"], c["max"] / hc["min"]] if fin is not None else None
                    sea = (srow.get("p") if kind == "any" else srow["C_gates"] / hs[kind]["C_gates"]
                           if srow.get("C_gates") and hs[kind] else None) if srow else None
                    row |= {f"p_{kind}": fin if on else sea, f"p_{kind}_basis": basis[kind], f"p_{kind}_range": rng,
                            f"p_{kind}_final": fin, f"p_{kind}_search": sea}
                row["search_reached"] = srow and srow.get("reached")
                row["S"] = row["p_any"] * rho if row["p_any"] is not None and rho is not None else None
                rows.append(row)
            rows.sort(key=lambda r: (*(math.inf if r[f"p_{kind}"] is None else r[f"p_{kind}"] for kind, _ in KINDS),
                                     r["point"]))
            caveat = [f"the search split's H{'' if kind == 'any' else ' honest'} at {g:g}x, {x['point']} "
                      f"({hs[kind]['record']}, C {hs[kind]['C_gates']:.3g} gates), has {x['at'][k]['finished']} of "
                      f"{x['at'][k]['planned']} seeds finished and none reaching it on the final set"
                      for kind, _ in KINDS if (x := hp[kind]) and x["at"][k]["finished"] and not x["at"][k]["reached"]]
            tables.append({"start": start, "gain": g, "data": mine[0]["data"], "search_start": scorers}
                          | {f"p_{kind}_basis": basis[kind] for kind, _ in KINDS}
                          | {f"H_{kind}": h_at(hf[kind], k) for kind, _ in KINDS}
                          | {f"H_{kind}_why": None if hf[kind] else
                             f"no{' honest' if kind == 'honest' else ''} point has a finished seed reaching it"
                             for kind, _ in KINDS}
                          | {f"H_{kind}_search": hs[kind] and hs[kind] | (
                              {"point": x["point"], "seeds": x["seeds"]}
                              | {n: x["at"][k][n] for n in ("finished", "planned")} if (x := hp[kind]) else
                              {"point": None}) for kind, _ in KINDS}
                          | {"H_caveat": "; ".join(caveat) or None, "rows": rows, "not_shown": left})
    return {"points": points, "tables": tables}


def h_at(p: dict | None, k: str) -> dict | None:
    return p and {"point": p["point"], "run": p["run"], "island": p["island"], "C": p["at"][k]["C"]}


def g3(x) -> str:
    return "-" if x is None else "inf" if isinstance(x, float) and math.isinf(x) else f"{x:.3g}"


def p_cell(r: dict, kind: str) -> str:
    """A row's p of `kind`: on the final set with its range and the search split's beside it ("≥" where the search
    run didn't reach g: its last checkpoint's), else the search split's, so marked."""
    rng, sea = r[f"p_{kind}_range"], r[f"p_{kind}_search"]
    if r[f"p_{kind}_basis"] != FINAL_SET:
        return f"{g3(r[f'p_{kind}'])} (search split)"
    return (g3(r[f"p_{kind}"]) + (f" [{g3(rng[0])}, {g3(rng[1])}]" if rng else "") + " (search split "
            + ("" if r["search_reached"] in ("yes", None) or sea is None else "≥") + f"{g3(sea)})")


def markdown(t: dict) -> str:
    """The table's markdown: per (start, gain) each H on the final set and the search split, what its p reads, and a
    row per point shown, the headline p_any first."""
    lines = [f"# Finals table ({t['built_at']}, te_core {t['te_core']})", "",
             f"Targets: {t['rule']['target']}", f"H: {t['rule']['H']}", f"rho: {t['rule']['rho']}", "",
             f"Policy tables: {t['inputs']['policy']}", ""]
    for tb in t["tables"]:
        f = t["fits"].get(tb["data"]) or {}
        lines += [f"## {tb['start']} at {tb['gain']:g}x ({tb['data']}; fit E {g3(f.get('E'))}, gamma {g3(f.get('gamma'))})",
                  ""]
        for kind, name in KINDS:
            h, s = tb[f"H_{kind}"], tb[f"H_{kind}_search"]
            lines += [f"- p_{kind}, {name}, reads: {tb[f'p_{kind}_basis']}",
                      f"  - H{'' if kind == 'any' else ' honest'} on the final set: " + (
                          f"{h['point']} ({h['island']}, {h['run']}), C {g3(h['C']['mean'])} gates, "
                          f"sd {g3(h['C']['sd'])} over {h['C']['n']} seeds" if h else tb[f"H_{kind}_why"]),
                      "  - on the search split: " + (
                          f"{s['record'] or '-'}, C {g3(s['C_gates'])} gates, " + (
                              f"{s['point']}, {s['finished']} of {s['planned']} finals seeds finished" if s["point"]
                              else "not a finals point") if s else
                          f"none: no reached row at {tb['search_start'] or 'this start'} in the scorer's table")]
        if tb.get("H_caveat"):
            lines.append(f"- CAVEAT: {tb['H_caveat']}")
        lines += ["", "| point | island | seeds reached | C gates, mean (sd) | counted FLOPs, mean | "
                  f"p_any, {KINDS[0][1]} | p_honest, {KINDS[1][1]} | rho (which, basis) | S = p_any rho | label |",
                  "|---|---|---|---|---|---|---|---|---|---|"] if tb["rows"] else ["", "No point shown."]
        for r in tb["rows"]:
            c, which = r["C"], r["S_reads"] or "rho_own"
            basis = r["basis_own"] if which == "rho_own" else r["basis_default"]
            lines.append(
                f"| {r['point']} ({r['run'][-4:]}) | {r['island']}{' (honest)' if r['honest'] else ''} | {r['reached']} | "
                + (f"{g3(c['mean'])} ({g3(c['sd'])})" if c else "-") + " | "
                + (g3(r["flops"]["mean"]) if r["flops"] else "-") + " | "
                + f"{p_cell(r, 'any')} | {p_cell(r, 'honest')} | {g3(r[which])} ({which}, {basis or '-'}) | "
                + f"{g3(r['S'])} | " + "; ".join([r["rho_label"], *r["label"]]) + " |")
        if tb["not_shown"]:
            lines.append("")
            lines.append("Not shown: " + "; ".join(f"{n} point(s): {why}" for why, n in tb["not_shown"].items()))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="each finished rerun's record from the store")
    f.add_argument("--launches", type=Path, required=True)
    f.add_argument("--to", type=Path, required=True)
    f.add_argument("--arts", type=Path, default=None, help="JSON, rerun id -> its run-record art id")
    b = sub.add_parser("build", help="the table")
    b.add_argument("--records", type=Path, required=True, help="fetch's directory: RUN/out/run/run.json per rerun")
    b.add_argument("--policy", type=Path, required=True,
                   help="the scorer's stored tables (algorithm-policy.json and frontier-fit.json)")
    b.add_argument("--launches", type=Path, required=True)
    b.add_argument("--finals", type=Path, default=FINALS)
    b.add_argument("--manifest", type=Path, default=policy.MANIFEST)
    b.add_argument("--ladder-fit", type=Path, default=None, help="the scoring cap's fit, as te_core.rescore reads it")
    b.add_argument("--policy-art", default="", help="the stored tables' art id, for the table's inputs")
    b.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        why = fetch(a.launches, a.to, json.loads(a.arts.read_text()) if a.arts else {})
        a.to.mkdir(parents=True, exist_ok=True)
        (a.to / "fetch.json").write_text(json.dumps(why, indent=1, sort_keys=True) + "\n")
        print(f"finals_table fetch: {len(why)} reruns without a record" + "".join(f"\n  {r}: {w}" for r, w in why.items()))
        return 0
    fetched = json.loads((a.records / "fetch.json").read_text()) if (a.records / "fetch.json").exists() else {}
    ap_panel = json.loads((a.policy / "algorithm-policy.json").read_text())
    fits = fits_of(json.loads((a.policy / "frontier-fit.json").read_text()))
    t = build(json.loads(a.finals.read_text()), reruns(a.launches), a.records, fits, policy_rows(ap_panel),
              policy.starts(a.manifest), policy.ladder_fit(a.ladder_fit) if a.ladder_fit else None, fetched,
              frontier.pinned(pilot=True))
    t = {"schema": SCHEMA, "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "te_core": VERSION,
         "inputs": {"finals": str(a.finals), "launches": str(a.launches), "records": str(a.records),
                    "policy": " ".join(filter(None, (a.policy_art, f"({ap_panel.get('source', '')}; updated "
                                                                   f"{ap_panel.get('updated_at')})"))),
                    "ladder_fit": str(a.ladder_fit) if a.ladder_fit else None},
         "rule": {"target": "frontier_fit.target(fit, l0, g) = E + (l0 - E) g^-gamma, E and gamma the start's data "
                            "fit in the scorer's tables, l0 each seed's start_bpb on the final set",
                  "H": "H_any, the point with the least mean C over its reached seeds, every route's finals seeds "
                       "counted (the scorer's rule, te_core.algorithm_policy's H), and the headline p_any = C/H_any, "
                       f"{KINDS[0][1]}; beside it H_honest over the honest points (their island a reference "
                       f"direction, te_core.frontier's REF_DIRECTIONS) and p_honest, {KINDS[1][1]}. Where the search "
                       "split's H at a (start, gain) has no finished finals seed, every row of it takes the search "
                       "split's p (the scorer's stored p; p_honest C_gates over its honest H's), so labelled in "
                       "p_any_basis and p_honest_basis, and no finals C is divided by a different H",
                  "rho": RHO_LABEL + " (the scorer's stored table); S = p_any * rho_own, rho_default where the table "
                                     "has no rho_own", "single_seed": SINGLE},
         "fits": fits} | t
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "finals-table.json").write_text(json.dumps(t, indent=1, allow_nan=True) + "\n")
    md = markdown(t)
    (a.out / "finals-table.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
