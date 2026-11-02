"""Night 2's formalism-v1 refresh (note:training-20261102T1817Z-report-training-formalism-v1), every 90 minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post]
    python v1cycle.py --reput OUT

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore` with its readings). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core or analysis/rescore_v1.py on
   cursor/covert-witness-4207 or cursor/training-efficiency-4207.
2. Runs its `python -m te_core.rescore` on the scorer's ledger at tonight's pin (PINS): per checkpoint of every valid
   v2.3 record, rescore_v1.py's quantities (Delta, u(1), Delta_grad, phi, the frozen tail's share t), Fit (running and
   training) and the Slowdown theorem's rho (a target) on te_core.policy's (c, k) grid under both readings of its
   frozen-tail case (literal-x0 and tail-charged, no headline S), H and W* over every checkpoint of every valid record
   (19:50Z), S = W*/H; frontier.md/json, formalism.jsonl and the console's five panels with their PNGs. Then
   v1runs.py gives A2 and P4 one row per valid run and target from te_core's same rows (coordinator 20:52Z), which
   replace te_core's staircase and per-algorithm rows in those two panels (kept beside them as te_core-<plot>.json).
3. Publishes the tables and each panel (its JSON and PNG) to the evidence store, writes each panel's console cut (the
   route's body and note limits, `console`) to /opt/cursor/artifacts/te-panels/, PUTs them, and posts one DIGEST line
   with the art ids. `--reput OUT` re-cuts a finished refresh's panels and PUTs them, with nothing re-scored.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import scorer as S

HERE = Path(__file__).resolve().parent
RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
RESCORE_PATHS = ("benchmarks/covert_witness/te_core", "benchmarks/covert_witness/analysis/rescore_v1.py")
RESCORE_REFS = ("cursor/covert-witness-4207", "cursor/training-efficiency-4207")
PINS = os.environ.get("TE_V1_PINS", "11d96c1b8745d8093746ea11e3e33fe6f07f7d3b,"
                                    "36aac280a62be7a04cbf510b50636b5ad875ba54").split(",")
PANELS_DIR = Path(os.environ.get("TE_PANELS_DIR", "/opt/cursor/artifacts/te-panels"))
PANELS_URL = "https://website-docs-sage.vercel.app/api/panels"
PLOT_OF = {"p1": "p1-loss-vs-compute", "p2": "p2-slowdown-vs-gain", "p3": "p3-policy-plane", "p4": "p4-decomposition",
           "a2": "a2-efficiency-floor"}
READINGS = ("literal-x0", "tail-charged")
REFERENCE = "c=4096w k=10kB"
REF_TARGET = -1.0
NIGHT2 = "2026-11-02T19:15:00Z"
BINS = (("<=10kB", 8e4), ("<=100kB", 8e5), ("<=1MB", 8e6))
BODY_MAX = 65536
NOTE_MAX = 500
COMMON = ("reading: literal-x0, tail-charged (Slowdown's frozen-tail case; Daniel decides; no headline S). basis: fit = "
          "rho 1 by Fit for the training; target = Slowdown's bound, unproved. Delta, u(1), t, a: witness upper bounds; "
          "tau a stub. One seed. Full panels: evidence store.")
PER_RUN = ("Per valid v2.3 run, target and reading, its least-compute checkpoint reaching the target (19:50Z); "
           "variant: its launch note's label.")
SHORT = {
    "p1-loss-vs-compute": "Loss (val bpb) against compute after theta, every checkpoint of every valid v2.3 record: "
                          "honest |C| (gates), covert |C| rho at c=4096w k=10kB, covert-policy at other (c, k).",
    "p2-slowdown-vs-gain": "S = W*/H at c=4096w k=10kB per start and target; H and W* over every checkpoint of every "
                           "valid v2.3 record, so S >= 1. g null until the ladder's F.",
    "p3-policy-plane": "S = W*/H over te_core.policy's (c, k) grid and each start's marked shapes, k in bits; H and W* "
                       "over every checkpoint of every valid v2.3 record.",
    "p4-decomposition": f"{PER_RUN} At c=4096w k=10kB: p = |C|/H, s = rho, S = p s.",
    "a2-efficiency-floor": f"{PER_RUN} The floor is their frontier; fit_*_k_bits: least k where Fit holds at c=4096w.",
}
BAND_FEW = ("c1e15/k1e5", "c1e15/k1e6", "c1e14/k1e4")
LEAN = ("start_id", "checkpoint", "method", "family")
RUNS_LEAN = ("start_id", "record", "checkpoint", "size", "saturation", "compute_frac", "g", "seeds")
PER_RUN_PANELS = ("a2-efficiency-floor", "p4-decomposition")
FINAL_TITLE = "FINAL (sealed shard: FINAL entries' 3-seed means, other records at the same drop): "
VARIANT_LABEL = 40


def fmt(x) -> str:
    if x is None:
        return "–"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.2e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str, cwd: Path = RESCORE_WT) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def rescore_head() -> str:
    """Check out, detached, the newest commit touching te_core or the analysis on either branch; returns it."""
    for ref in RESCORE_REFS:
        subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    commit = git("log", "-1", "--format=%H", *(f"origin/{r}" for r in RESCORE_REFS), "--", *RESCORE_PATHS)
    git("checkout", "-q", "--detach", commit)
    return commit


def rescore(out: Path, gates: Path, final: Path | None = None) -> str:
    """te_core.rescore into `out` (module docstring, 2), through v1final.py with FINAL_DIR's values for `final`, and
    v1runs.py's per-run A2 and P4 (in final mode also the search's, prefixed search-runs-); returns its last line."""
    out.mkdir(parents=True, exist_ok=True)
    if gates.exists() and not (out / "analysis-gates.json").exists():
        (out / "analysis-gates.json").write_bytes(gates.read_bytes())
    ledger = os.environ.get("TE_LEDGER", str(Path.home() / ".research" / "te-ledger"))
    args = ["--out", str(out), "--root", ledger, "--scorer", str(S.STATE), *(x for p in PINS for x in ("--pin", p))]
    runs = [sys.executable, str(HERE / "v1runs.py"), *args]
    cmds = ([[sys.executable, str(HERE / "v1final.py"), str(final), *args], [*runs, "--final", str(final)],
             [*runs, "--prefix", "search-runs-"]] if final else [[sys.executable, "-m", "te_core.rescore", *args], runs])
    logs = []
    for c in cmds:
        r = subprocess.run(c, cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True,
                           check=False)
        logs.append(r.stdout + r.stderr)
        (out / "rescore.log").write_text("\n".join(logs))
        if r.returncode:
            raise SystemExit(f"v1cycle: {' '.join(c[1:3])} failed (rc {r.returncode}):\n{r.stderr[-2000:]}")
    return logs[0].strip().splitlines()[-1]


def body(p: dict) -> bytes:
    return json.dumps(p, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


def frame(p: dict) -> list[dict]:
    ix = {c: i for i, c in enumerate(p["columns"])}
    return [{c: r[i] for c, i in ix.items()} for r in p["rows"]]


def compact(p: dict, keep=lambda r: True, drop: tuple = (), fix=None) -> dict:
    """`p` with the rows `keep` takes (each through `fix`), without the columns `drop`, booleans as yes/no and numbers
    to 4 significant digits (losses 6)."""
    cols = [c for c in p["columns"] if c not in drop]

    def val(c, v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return round(v, 6) if c in ("loss", "loss_lo", "loss_hi", "l0") else float(f"{v:.4g}")
        return v
    rows = []
    for d in frame(p):
        if keep(d):
            d = fix(d) if fix else d
            rows.append([val(c, d[c]) for c in cols])
    return p | {"columns": cols, "rows": rows}


def variant_label(d: dict) -> dict:
    """The row with its variant cut to the launch note's label (the text before its first colon), at most
    VARIANT_LABEL characters."""
    v = d.get("variant")
    return d | {"variant": v.split(":")[0][:VARIANT_LABEL]} if v else d


def thin(p: dict) -> dict:
    """P1's curves at the points that lower the loss by 0.25% or reach a deeper target, each curve's ends kept."""
    ix = {c: i for i, c in enumerate(p["columns"])}
    groups: dict[tuple, list] = defaultdict(list)
    for r in p["rows"]:
        groups[(r[ix["start_id"]], r[ix["curve"]], r[ix["policy"]], r[ix["reading"]])].append(r)
    rows = []
    for g in groups.values():
        kept = [g[0]]
        for i, r in enumerate(g[1:], 1):
            last = kept[-1]
            if i == len(g) - 1 or r[ix["target_pct"]] != last[ix["target_pct"]] or \
                    r[ix["loss"]] <= last[ix["loss"]] * 0.9975:
                kept.append(r)
        rows += kept
    return p | {"rows": rows}


def console(P: dict, final: bool = False) -> dict:
    """Each panel cut to the route's limits (a body of BODY_MAX bytes, a note of NOTE_MAX characters): the short note,
    and the first of its cuts that fits, which the note names. Every cut keeps both readings, except A2's second and
    third, whose rows agree under both (A2 reads no rho) and so are one row each, `reading` both."""
    def ref(r):
        return r["target_pct"] == REF_TARGET

    def one(r):
        return r["reading"] == READINGS[0]

    def both(d):
        return variant_label(d) | {"reading": "both"}
    out = {}
    for pid, p in P.items():
        tries = [("", lambda r: True, LEAN, None)]
        if pid == "p1-loss-vs-compute":
            p = thin(p)
            tries += [(f" The band here: {', '.join(BAND_FEW)}.",
                       lambda r: r["curve"] != "covert-policy" or r["policy"] in BAND_FEW, LEAN, None),
                      (f" The band here: {BAND_FEW[1]}.",
                       lambda r: r["curve"] != "covert-policy" or r["policy"] == BAND_FEW[1], LEAN, None),
                      (" No band here.", lambda r: r["curve"] != "covert-policy", LEAN, None)]
        elif pid == "p3-policy-plane":
            tries += [(" Here -1% only.", ref, LEAN, None),
                      (" Here -1% only, no record column.", ref, (*LEAN, "record"), None),
                      (" Here -1% only, no record, g, k_bytes or p (= S/rho).", ref,
                       (*LEAN, "record", "g", "k_bytes", "p"), None)]
        elif pid == "a2-efficiency-floor":
            tries = [("", lambda r: True, RUNS_LEAN, variant_label),
                     (" Rows: reading both (A2 reads no rho).", one, RUNS_LEAN, both),
                     (" Rows: reading both (A2 reads no rho); here -1% only.", lambda r: one(r) and ref(r), RUNS_LEAN,
                      both)]
        elif pid == "p4-decomposition":
            tries = [("", lambda r: True, RUNS_LEAN, variant_label),
                     (" No fit_*_k_bits here (A2 has them).", lambda r: True,
                      (*RUNS_LEAN, "fit_run_k_bits", "fit_train_k_bits"), variant_label),
                     (" Here -1% only.", ref, RUNS_LEAN, variant_label),
                     (" Here -1% only, no variant.", ref, (*RUNS_LEAN, "variant"), None)]
        common = COMMON.replace(" One seed.", "") if final else COMMON
        for why, keep, drop, fix in tries:
            q = compact(p, keep, drop, fix)
            note = f"{SHORT[pid]}{why} {common}"
            q["note"] = note if len(note) <= NOTE_MAX else note[:NOTE_MAX - 1] + "…"
            if len(body(q)) <= BODY_MAX:
                break
        out[pid] = q
    return out


def put_data(tree: Path, kind: str, meta: dict) -> str:
    r = subprocess.run(["research", "data", "put", "--kind", kind, "--tree", str(tree), "--preserve", "--json",
                        "--meta", json.dumps(meta)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"data put {tree}: {r.stderr[-400:]}")
    return json.loads(r.stdout)["id"]


def put_panel(panel: dict) -> str:
    import urllib.error
    import urllib.request

    from research.slack import oidc_token
    req = urllib.request.Request(f"{PANELS_URL}/{panel['id']}", body(panel), method="PUT",
                                 headers={"Authorization": f"Bearer {oidc_token(PANELS_URL)}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return f"{r.status}"
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.read()[:400].decode(errors='replace')}"


def night2_runs() -> set[str]:
    out = set()
    for line in (S.STATE / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py" and (x.get("launched") or "") >= NIGHT2:
            out.add(x.get("run"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="default /tmp/scorer/v1/<stamp>")
    ap.add_argument("--gates", type=Path, default=Path("/tmp/scorer/v1/trial/gates.json"),
                    help="the analysis's gate factors, seeded into the re-score's cache")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-put", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--reput", type=Path, metavar="OUT", help="re-cut a refresh's full panels (OUT/publish) and PUT them")
    ap.add_argument("--final", type=Path, metavar="FINAL_DIR",
                    help="the final table: te_core.rescore with FINAL_DIR's values (scorer.py final table; v1final.py); "
                         "posts FINAL-TABLE, not DIGEST")
    args = ap.parse_args()
    if args.reput:
        return reput(args.reput)
    stamp, final = now(), args.final
    out = args.out or Path("/tmp/scorer/v1") / f"{'final-' if final else ''}{stamp.replace(':', '')}"
    commit = rescore_head()
    last = rescore(out / "rescore", args.gates, final)
    rs = out / "rescore"
    what = "final table" if final else "refresh"
    fr = json.loads((rs / "frontier.json").read_text())
    scored = [json.loads(line) for line in (rs / "formalism.jsonl").open()]
    runs, n2 = {r["run_id"] for r in scored}, night2_runs()
    meta = {"stamp": stamp, "te_core": commit, "pin": fr["pin"], "readings": fr["readings"],
            "checkpoints": len(scored), "records": len(runs), "night2_records": len(runs & n2),
            "not_analysed": len(fr["not_analysed"]), "rule": "H and W* over every checkpoint (19:50Z)"}
    trees = {k: out / "publish" / k for k in ("tables", *PLOT_OF)}
    for t in trees.values():
        t.mkdir(parents=True, exist_ok=True)
    for name in ("frontier.md", "frontier.json", "formalism.jsonl", "rescore.log"):
        (trees["tables"] / name).write_bytes((rs / name).read_bytes())
    fin = json.loads((rs / "final_rows.json").read_text()) if final else None
    if final:
        ents = fin["entries"]
        meta |= {"final": str(final), "entries": len(ents), "entries_matched": sum(e["matched"] > 0 for e in ents.values()),
                 "final_checkpoints": len(fin["rows"])}
        (trees["tables"] / "final.md").write_text(final_md(rs, fin, json.loads((final / "table.json").read_text())))
        (trees["tables"] / "final_rows.json").write_bytes((rs / "final_rows.json").read_bytes())
        for name in ("table.md", "table.json", "final_submissions.json", "results.jsonl", "plan.json"):
            if (final / name).exists():
                (trees["tables"] / f"scorer-final-{name}").write_bytes((final / name).read_bytes())
    (trees["tables"] / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    ids = {}
    if not args.no_publish:
        ids["tables"] = put_data(trees["tables"], "evidence/v1", {"what": f"te formalism v1 {what}: te_core.rescore's "
                                                                          "frontier and per-checkpoint blocks", **meta})
    source = (f"{ids.get('tables', 'unpublished')[:16]} (scorer v1cycle {what} {stamp}: te_core.rescore "
              f"{commit[:10]}, {len(scored)} valid v2.3 checkpoints)")
    P = {}
    for pid in PLOT_OF.values():
        own = rs / (f"runs-{pid}.json" if pid in PER_RUN_PANELS else f"{pid}.json")
        P[pid] = json.loads(own.read_text()) | {"source": source}
        if final:
            P[pid]["title"] = FINAL_TITLE + P[pid]["title"]
    if final:
        P["p1-loss-vs-compute"] = seeds_range(P["p1-loss-vs-compute"], fin["rows"])
    C = console(P, bool(final))
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, pid in PLOT_OF.items():
        if pid in PER_RUN_PANELS:
            (trees[key] / f"te_core-{pid}.json").write_bytes((rs / f"{pid}.json").read_bytes())
        (trees[key] / f"{pid}.json").write_text(json.dumps(P[pid]) + "\n")
        (trees[key] / f"{pid}.console.json").write_bytes(body(C[pid]) + b"\n")
        (trees[key] / f"{pid}.png").write_bytes((rs / f"{pid}.png").read_bytes())
        (PANELS_DIR / f"{pid}.json").write_bytes(body(C[pid]) + b"\n")
        (PANELS_DIR / f"{pid}.png").write_bytes((rs / f"{pid}.png").read_bytes())
        if not args.no_publish:
            ids[key] = put_data(trees[key], "evidence/v1", {"what": f"te formalism v1 {what} {pid}", **meta})
    puts = {} if args.no_put else {pid: put_panel(C[pid]) for pid in PLOT_OF.values()}
    summary = meta | {"art": ids, "panels_put": puts, "rows": {k: len(v["rows"]) for k, v in P.items()},
                      "console_rows": {k: len(v["rows"]) for k, v in C.items()},
                      "console_bytes": {k: len(body(v)) for k, v in C.items()}, "rescore": last, "out": str(out)}
    (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
    line = digest(P, meta, ids, puts)
    (out / "digest.txt").write_text(line + "\n")
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "FINAL-TABLE" if final else "DIGEST", line],
                       check=True)
    return 0


def seeds_range(p: dict, rows: dict) -> dict:
    """P1 with each final checkpoint's seeds' range as loss_lo and loss_hi."""
    ix = {c: i for i, c in enumerate(p["columns"])}
    out = []
    for r in p["rows"]:
        f = rows.get(r[ix["record"]])
        out.append([*r[:ix["loss_lo"]], f["bpb_min"], f["bpb_max"], *r[ix["loss_hi"] + 1:]] if f else r)
    return p | {"rows": out}


def final_md(rs: Path, fin: dict, tab: dict) -> str:
    """Per FINAL entry and target: its least compute reaching the target on the sealed shard (the seeds' mean) beside
    the search's, and its S at that checkpoint under each reading (v1runs.py's per-run P4, final and search)."""
    p4 = frame(json.loads((rs / "runs-p4-decomposition.json").read_text()))
    s4 = frame(json.loads((rs / "search-runs-p4-decomposition.json").read_text()))
    L = ["## FINAL entries on the sealed shard", "",
         (f"{len(fin['entries'])} entries; targets from the final starts' bpb ("
          + ", ".join(f"{s} {v:.4f}" for s, v in sorted(fin["final_starts"].items())) + "). A checkpoint reaches a "
          "target when its seeds' mean bpb does (all: the worst seed does too). Gates and every quantity but the bpb "
          "are the search checkpoint's at the nearest MACs (v1final.py); S = p s at that checkpoint, at c = 4096w, "
          "k = 10 kB, over the final H (te_core.rescore)."), "",
         ("| entry | start | target | final: least gates (frac; bpb mean [lo, hi]; seeds) | search: least gates (frac) "
          "| final/search | S literal-x0 (ρ, basis) | S tail-charged (ρ, basis) |"), "|---|---|---|---|---|---|---|---|"]
    for e, x in sorted(fin["entries"].items(), key=lambda t: str(t[1]["run"])):
        mine = [r for r in p4 if r["run"] == x["run"]]
        name = f"{mine[0]['method'] if mine else '?'} {e[4:16]} ({x['run']})"
        if not x["matched"] or not mine:
            why = "no final row at a search checkpoint's MACs" if not x["matched"] else "reaches no target"
            L.append(f"| {name} | | | {why} | | | | |")
            continue
        for tp in sorted({r["target_pct"] for r in mine}, reverse=True):
            at = {r["reading"]: r for r in mine if r["target_pct"] == tp}
            f = next(iter(at.values()))
            rec = fin["rows"].get(f["record"], {})
            lf = fin["final_starts"][f["start_id"]] * (1 + tp / 100)
            g = next((r for r in s4 if r["run"] == x["run"] and r["target_pct"] == tp), None)
            seeds = rec.get("seeds") or []
            fs = (f"{fmt(f['compute_gates'])} ({fmt(f['checkpoint_frac'])}; {rec.get('final_mean_bpb', 0):.4f} "
                  f"[{rec.get('bpb_min', 0):.4f}, {rec.get('bpb_max', 0):.4f}]; "
                  f"{'all' if rec.get('bpb_max', 9) <= lf else 'not all'} of {len(seeds)})" if rec else
                  f"{fmt(f['compute_gates'])} (search value)")
            gs = f"{fmt(g['compute_gates'])} ({fmt(g['checkpoint_frac'])})" if g else "not reached"
            ratio = fmt(f["compute_gates"] / g["compute_gates"]) if g else "–"
            vs = [f"{fmt(at[m]['S'])} ({fmt(at[m]['s'])}, {at[m]['basis']})" if m in at else "–" for m in READINGS]
            L.append(f"| {name} | {f['start_id']} | {tp:g}% | {fs} | {gs} | {ratio} | " + " | ".join(vs) + " |")
    L.append("")
    bad = [x for x in tab.get("reruns", []) if x.get("problems")]
    if bad:
        L += ["### Reruns left out", "", "| entry | seed | run | problems |", "|---|---|---|---|"]
        L += [f"| {x['id'][4:16]} | {x.get('seed')} | {x.get('run')} | {'; '.join(x['problems'])} |" for x in bad]
        L.append("")
    return "\n".join(L)


def reput(out: Path) -> int:
    """The console's cut of a finished refresh's full panels, written to PANELS_DIR and PUT, with nothing re-scored."""
    C = console({pid: json.loads((out / "publish" / key / f"{pid}.json").read_text()) for key, pid in PLOT_OF.items()})
    puts = {}
    for pid, q in C.items():
        (PANELS_DIR / f"{pid}.json").write_bytes(body(q) + b"\n")
        puts[pid] = put_panel(q)
    print(json.dumps({"panels_put": puts, "console_bytes": {k: len(body(v)) for k, v in C.items()}}))
    return 0 if all(str(v).startswith("2") for v in puts.values()) else 1


def digest(P: dict, meta: dict, ids: dict, puts: dict) -> str:
    """At the reference target and policy, per start: H, each reading's S (rho, route) from P2's rows, and A2's least
    gates at the first Delta bins."""
    p2, a2 = frame(P["p2-slowdown-vs-gain"]), frame(P["a2-efficiency-floor"])
    parts = []
    for s in ("s100", "m100", "s50", "m50"):
        at = {r["reading"]: r for r in p2 if r["start_id"] == s and r["target_pct"] == REF_TARGET
              and r["policy"] == REFERENCE}
        if not at:
            continue
        h = next(iter(at.values()))["H_gates"]
        if h is None:
            parts.append(f"{s} not reached")
            continue
        q = " / ".join(f"{fmt(at[m]['S'])} {m} (ρ {fmt(at[m]['rho'])}, {at[m]['basis']}; {at[m]['method']} "
                       f"{at[m]['record']}@{at[m]['checkpoint']})" if m in at and at[m]["S"] is not None else f"– {m}"
                       for m in READINGS)
        fl = [r for r in a2 if r["start_id"] == s and r["target_pct"] == REF_TARGET and r["reading"] == READINGS[0]]
        floor = []
        for n, b in BINS:
            ok = [r["compute_gates"] for r in fl if r["delta_bits"] <= b]
            floor.append(f"{n} {fmt(min(ok))}" if ok else f"{n} none")
        parts.append(f"{s}: H {fmt(h)}; S {q}; A2 least gates at " + ", ".join(floor))
    arts = ", ".join(f"{k} {v[:16]}" for k, v in ids.items())
    ok = sum(str(v).startswith("2") for v in puts.values())
    head = (f"v1 final table (sealed shard: {meta['entries']} FINAL entries' reruns, {meta['entries_matched']} matched "
            f"at {meta['final_checkpoints']} checkpoints, 3 seeds' mean; every other record's search value at the same "
            "drop; " if "entries" in meta else "v1 refresh (")
    return (f"{head}te_core.rescore {meta['te_core'][:10]}: {meta['records']} valid v2.3 records, "
            f"{meta['night2_records']} from night 2, {meta['checkpoints']} checkpoints, {meta['not_analysed']} not "
            "analysed yet; H and W* over every checkpoint, so S >= 1; both readings of Slowdown's frozen-tail case, "
            "literal-x0 / tail-charged, no headline S; rho a target unless Fit holds for the training) at −1% and "
            f"{REFERENCE}: " + "; ".join(parts) + f". art: {arts}. Console panels PUT {ok}/{len(puts)} (p1-p4, a2).")


if __name__ == "__main__":
    sys.exit(main())
