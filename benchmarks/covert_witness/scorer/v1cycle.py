"""Night 2's refresh on the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2), every 90
minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post]
    python v1cycle.py --final FINAL_DIR [--out DIR] [--no-publish] [--no-post]

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore`; 22:16Z: through the integrator's
v2 te_core). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core or analysis/rescore_v1.py on
   cursor/covert-witness-4207 or cursor/training-efficiency-4207.
2. Runs its `python -m te_core.rescore` on the scorer's ledger at tonight's pin (PINS): per checkpoint of every valid
   v2.3 record, rescore_v1.py's quantities (Delta, u(1), Delta_grad, phi, sigma, the frozen tail's share t), Fit per
   kind of block and rho (c/(c - d) where every part fits, else Slowdown's bound, a target) on te_core.policy's (c, k)
   grid with the frozen tail always charged (Daniel, 22:00Z), H and W* over every checkpoint of every valid record
   (19:50Z), S = W*/H; frontier.md/json and formalism.jsonl.
3. Runs its `python -m te_core.panels frontier-branches --publish` in the same directory: the console's one training
   panel (Daniel, coordinator 23:56Z; the integrator's builder, 23:59Z), PUT. A refused PUT is reported, not fatal.
   Nothing else is PUT, and te_core's five old panels stay in the re-score's directory, unpublished (every other plot
   is dropped, 23:56Z).
4. Publishes the tables and frontier-branches (uncut, its console cut and PNG) to the evidence store, copies the cut to
   PANELS_DIR, and posts one DIGEST line with the art ids.

`--final FINAL_DIR` scores the sealed shard's values (v1final.py, frontier-branches included, not PUT) and v1runs.py's
per-run P4 (final and search) for final.md, and posts FINAL-TABLE instead of DIGEST.
"""
import argparse
import json
import os
import subprocess
import sys
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
FB = "frontier-branches"
OLD_PANELS = ("p1-loss-vs-compute", "p2-slowdown-vs-gain", "p3-policy-plane", "p4-decomposition",
              "a2-efficiency-floor")
REFERENCE = "c=4096w k=10kB"
REF_TARGET = -1.0
NIGHT2 = "2026-11-02T19:15:00Z"
FINAL_TITLE = "FINAL (sealed shard: FINAL entries' 3-seed means, other records at the same drop): "


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


def ledger_args(out: Path) -> list[str]:
    ledger = os.environ.get("TE_LEDGER", str(Path.home() / ".research" / "te-ledger"))
    return ["--out", str(out), "--root", ledger, "--scorer", str(S.STATE), *(x for p in PINS for x in ("--pin", p))]


def rescore(out: Path, gates: Path, final: Path | None = None) -> str:
    """te_core.rescore into `out` (module docstring, 2), or with FINAL_DIR's values for `final` through v1final.py and
    v1runs.py's per-run P4 (final, and the search's prefixed search-runs-); returns its last line."""
    out.mkdir(parents=True, exist_ok=True)
    if gates.exists() and not (out / "analysis-gates.json").exists():
        (out / "analysis-gates.json").write_bytes(gates.read_bytes())
    args = ledger_args(out)
    runs = [sys.executable, str(HERE / "v1runs.py"), *args]
    cmds = ([[sys.executable, str(HERE / "v1final.py"), str(final), *args], [*runs, "--final", str(final)],
             [*runs, "--prefix", "search-runs-"]] if final else [[sys.executable, "-m", "te_core.rescore", *args]])
    logs = []
    for c in cmds:
        r = subprocess.run(c, cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True,
                           check=False)
        logs.append(r.stdout + r.stderr)
        (out / "rescore.log").write_text("\n".join(logs))
        if r.returncode:
            raise SystemExit(f"v1cycle: {' '.join(c[1:3])} failed (rc {r.returncode}):\n{r.stderr[-2000:]}")
    return logs[0].strip().splitlines()[-1]


def branches(out: Path, put: bool) -> str:
    """te_core.panels' frontier-branches into `out` after the re-score (its gate cache reused), PUT with `put`; the
    PUT's status, or why there is none."""
    r = subprocess.run([sys.executable, "-m", "te_core.panels", FB, *ledger_args(out), *(["--publish"] if put else [])],
                       cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True, check=False)
    (out / "panels.log").write_text(r.stdout + r.stderr)
    done = next((x for x in reversed(r.stdout.splitlines()) if x.startswith("PUT ")), None)
    if done:
        return done.split(f"{FB}: ", 1)[-1][:300]
    why = (r.stderr.strip().splitlines() or [f"rc {r.returncode}"])[-1][:300]
    if not (out / "console" / f"{FB}.json").exists():
        return f"not built: {why}"
    return f"not PUT: {why}" if r.returncode else "not PUT (--no-put)"


def frame(p: dict) -> list[dict]:
    ix = {c: i for i, c in enumerate(p["columns"])}
    return [{c: r[i] for c, i in ix.items()} for r in p["rows"]]


def put_data(tree: Path, kind: str, meta: dict) -> str:
    r = subprocess.run(["research", "data", "put", "--kind", kind, "--tree", str(tree), "--preserve", "--json",
                        "--meta", json.dumps(meta)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"data put {tree}: {r.stderr[-400:]}")
    return json.loads(r.stdout)["id"]


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
    ap.add_argument("--no-put", action="store_true", help="build frontier-branches but don't PUT it")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--final", type=Path, metavar="FINAL_DIR",
                    help="the final table: te_core.rescore with FINAL_DIR's values (scorer.py final table; v1final.py); "
                         "posts FINAL-TABLE, not DIGEST")
    args = ap.parse_args()
    stamp, final = now(), args.final
    out = args.out or Path("/tmp/scorer/v1") / f"{'final-' if final else ''}{stamp.replace(':', '')}"
    commit = rescore_head()
    last = rescore(out / "rescore", args.gates, final)
    rs = out / "rescore"
    put = "not PUT (final values)" if final else branches(rs, not args.no_put)
    what = "final table" if final else "refresh"
    fr = json.loads((rs / "frontier.json").read_text())
    scored = [json.loads(line) for line in (rs / "formalism.jsonl").open()]
    runs, n2 = {r["run_id"] for r in scored}, night2_runs()
    meta = {"stamp": stamp, "te_core": commit, "pin": fr["pin"], "formalism": "v2, the frozen tail always charged",
            "checkpoints": len(scored), "records": len(runs), "night2_records": len(runs & n2),
            "not_analysed": len(fr["not_analysed"]), "rule": "H and W* over every checkpoint (19:50Z)"}
    trees = {k: out / "publish" / k for k in ("tables", "fb")}
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
        for name in ("final_rows.json", "runs-p4-decomposition.json", "search-runs-p4-decomposition.json"):
            (trees["tables"] / name).write_bytes((rs / name).read_bytes())
        for name in ("table.md", "table.json", "final_submissions.json", "results.jsonl", "plan.json"):
            if (final / name).exists():
                (trees["tables"] / f"scorer-final-{name}").write_bytes((final / name).read_bytes())
    (trees["tables"] / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    ids = {}
    if not args.no_publish:
        ids["tables"] = put_data(trees["tables"], "evidence/v1", {"what": f"te formalism v2 {what}: te_core.rescore's "
                                                                          "frontier and per-checkpoint blocks", **meta})
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for pid in OLD_PANELS:
        for stale in PANELS_DIR.glob(f"{pid}.*"):
            stale.unlink()
    fb = None
    if (rs / "console" / f"{FB}.json").exists():
        fb = json.loads((rs / f"{FB}.json").read_text())
        if final:
            fb["title"] = FINAL_TITLE + fb["title"]
        cut, png = (rs / "console" / f"{FB}.json").read_bytes(), (rs / f"{FB}.png").read_bytes()
        (trees["fb"] / f"{FB}.json").write_text(json.dumps(fb) + "\n")
        (trees["fb"] / f"{FB}.console.json").write_bytes(cut)
        (trees["fb"] / f"{FB}.png").write_bytes(png)
        (trees["fb"] / "panels.log").write_bytes((rs / "panels.log").read_bytes() if (rs / "panels.log").exists()
                                                  else b"")
        if not final:
            (PANELS_DIR / f"{FB}.json").write_bytes(cut)
            (PANELS_DIR / f"{FB}.png").write_bytes(png)
        if not args.no_publish:
            ids["fb"] = put_data(trees["fb"], "evidence/v1", {"what": f"te formalism v2 {what} {FB}", "put": put,
                                                              **meta})
    fb_rows = {"uncut": len(fb["rows"]), "console": len(json.loads(cut)["rows"])} if fb else None
    summary = meta | {"art": ids, "frontier_branches": {"put": put, "rows": fb_rows}, "rescore": last,
                      "out": str(out)}
    (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
    line = digest(json.loads((rs / "p2-slowdown-vs-gain.json").read_text()), meta, ids, put, fb_rows)
    (out / "digest.txt").write_text(line + "\n")
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "FINAL-TABLE" if final else "DIGEST", line],
                       check=True)
    return 0


def final_md(rs: Path, fin: dict, tab: dict) -> str:
    """Per FINAL entry and target: its least compute reaching the target on the sealed shard (the seeds' mean) beside
    the search's, and its S at that checkpoint (v1runs.py's per-run P4, final and search)."""
    p4 = frame(json.loads((rs / "runs-p4-decomposition.json").read_text()))
    s4 = frame(json.loads((rs / "search-runs-p4-decomposition.json").read_text()))
    L = ["## FINAL entries on the sealed shard", "",
         (f"{len(fin['entries'])} entries; targets from the final starts' bpb ("
          + ", ".join(f"{s} {v:.4f}" for s, v in sorted(fin["final_starts"].items())) + "). A checkpoint reaches a "
          "target when its seeds' mean bpb does (all: the worst seed does too). Gates and every quantity but the bpb "
          "are the search checkpoint's at the nearest MACs (v1final.py); S = p s at that checkpoint, at c = 4096w, "
          "k = 10 kB, over the final H (te_core.rescore; formalism v2, the frozen tail always charged)."), "",
         ("| entry | start | target | final: least gates (frac; bpb mean [lo, hi]; seeds) | search: least gates (frac) "
          "| final/search | S (ρ, basis) |"),
         "|---" * 7 + "|"]
    for e, x in sorted(fin["entries"].items(), key=lambda t: str(t[1]["run"])):
        mine = [r for r in p4 if r["run"] == x["run"]]
        name = f"{mine[0]['method'] if mine else '?'} {e[4:16]} ({x['run']})"
        if not x["matched"] or not mine:
            why = "no final row at a search checkpoint's MACs" if not x["matched"] else "reaches no target"
            L.append(f"| {name} | | | {why} | | | |")
            continue
        for f in sorted(mine, key=lambda r: -r["target_pct"]):
            tp = f["target_pct"]
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
            L.append(f"| {name} | {f['start_id']} | {tp:g}% | {fs} | {gs} | {ratio} | "
                     f"{fmt(f['S'])} ({fmt(f['s'])}, {f['basis']}) |")
    L.append("")
    bad = [x for x in tab.get("reruns", []) if x.get("problems")]
    if bad:
        L += ["### Reruns left out", "", "| entry | seed | run | problems |", "|---|---|---|---|"]
        L += [f"| {x['id'][4:16]} | {x.get('seed')} | {x.get('run')} | {'; '.join(x['problems'])} |" for x in bad]
        L.append("")
    return "\n".join(L)


def digest(p: dict, meta: dict, ids: dict, put: str, fb_rows: dict | None) -> str:
    """At the reference target and policy, per start: H and S (rho, basis, W*'s checkpoint) from te_core's P2 rows
    (te_core.frontier's cells), and frontier-branches' rows and PUT."""
    p2 = frame(p)
    parts = []
    for s in ("s100", "m100", "s50", "m50"):
        at = next((r for r in p2 if r["start_id"] == s and r["target_pct"] == REF_TARGET and r["policy"] == REFERENCE),
                  None)
        if at is None:
            continue
        if at["H_gates"] is None:
            parts.append(f"{s} not reached")
            continue
        q = (f"{fmt(at['S'])} (ρ {fmt(at['rho'])}, {at['basis']}; {at['method']} {at['record']}@{at['checkpoint']})"
             if at["S"] is not None else "–")
        parts.append(f"{s}: H {fmt(at['H_gates'])}; S {q}")
    arts = ", ".join(f"{k} {v[:16]}" for k, v in ids.items())
    put = f"PUT {put}" if put[:1].isdigit() else put
    fb = (f"training/frontier-branches (te_core.panels): {fb_rows['uncut']:,} rows, the console's cut "
          f"{fb_rows['console']}, {put}" if fb_rows else f"training/frontier-branches {put}")
    head = (f"v2 final table (sealed shard: {meta['entries']} FINAL entries' reruns, {meta['entries_matched']} matched "
            f"at {meta['final_checkpoints']} checkpoints, 3 seeds' mean; every other record's search value at the same "
            "drop; " if "entries" in meta else "v2 refresh (")
    return (f"{head}te_core {meta['te_core'][:10]}, formalism v2 with the frozen tail always charged: "
            f"{meta['records']} valid v2.3 records, {meta['night2_records']} from night 2, {meta['checkpoints']} "
            f"checkpoints, {meta['not_analysed']} not analysed yet; H and W* over every checkpoint, so S >= 1; ρ "
            "c/(c − d) where Fit holds for the training, else Slowdown's bound, a target) at −1% and "
            f"{REFERENCE}: " + "; ".join(parts) + f". {fb}; nothing else PUT. art: {arts}.")


if __name__ == "__main__":
    sys.exit(main())
