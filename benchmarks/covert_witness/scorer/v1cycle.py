"""Night 2's refresh on the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2), every 90
minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post] [--no-trace]
    python v1cycle.py --final FINAL_DIR [--out DIR] [--no-publish] [--no-post]

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore`; 22:16Z: through the integrator's
v2 te_core; 01:22Z: on te_core 4.0). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core or analysis/rescore_v1.py on
   cursor/covert-witness-4207 or cursor/training-efficiency-4207 whose te_core is 4.x or later.
2. Builds the measured ledger (trace40.py): every valid run's step traced, measured and ingested again on TRACE_NODE,
   the runs traced at earlier refreshes kept.
3. Runs its `python -m te_core.rescore` on that ledger at tonight's pin (PINS): per checkpoint of every valid v2.3
   record measured, formalism v2's quantities measured on its traced step (t literal), Fit per kind of block and rho
   (c/(c - d) where every part fits, else Slowdown's bound, a target) on te_core.policy's (c, k) grid with the frozen
   tail always charged (Daniel, 22:00Z), H and W* over every checkpoint of every valid record (19:50Z), S = W*/H;
   frontier.md/json, formalism.jsonl and compare.md/json; then `--tail frozen` and `--quantities analysis`, which go
   to the evidence store only (01:22Z).
4. Runs its `python -m te_core.panels frontier-branches --publish` on the same ledger: the console's one training
   panel (Daniel, coordinator 23:56Z; the integrator's builder, 23:59Z), PUT. A refused PUT is reported, not fatal.
   Nothing else is PUT, and te_core's five old panels stay in the re-score's directory, unpublished (every other plot
   is dropped, 23:56Z).
5. Publishes the tables and frontier-branches (uncut, its console cut and PNG) to the evidence store, copies the cut to
   PANELS_DIR, and posts one DIGEST line with the art ids; an S whose rho is Slowdown's bound carries
   loose-end-attention (01:22Z: A1 charges no read of earlier tokens' keys and values).

`--final FINAL_DIR` scores the sealed shard's values (v1final.py, frontier-branches included, not PUT) and v1runs.py's
per-run P4 (final and search) for final.md, and posts FINAL-TABLE instead of DIGEST.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import scorer as S
import trace40

HERE = Path(__file__).resolve().parent
RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
RESCORE_PATHS = ("benchmarks/covert_witness/te_core", "benchmarks/covert_witness/analysis/rescore_v1.py")
RESCORE_REFS = ("cursor/covert-witness-4207", "cursor/training-efficiency-4207")
#: the refresh runs on te_core 4.0 or later (coordinator 01:22Z): formalism v2's quantities measured on traced steps
TE_MAJOR = 4
# the scorer's te_core change awaiting the integrator's merge (Delta_t, coordinator 00:47Z), used while it holds theirs
RESCORE_AHEAD = os.environ.get("TE_RESCORE_AHEAD", "cursor/te-delta-t-40-516a")
TRACE_NODE = os.environ.get("TE_TRACE_NODE", "vy-nebius-cpu-9")
#: how long a refresh waits for its node runs; one still tracing is taken at the next refresh
TRACE_WAIT_S = 1080
#: what 4.0's rescore also writes, to the evidence store only (coordinator 01:22Z): t as the frozen tail, and the analysis
VARIANTS = {"frozen": ["--tail", "frozen"], "analysis": ["--quantities", "analysis"]}
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


def te_version(commit: str) -> str:
    init = git("show", f"{commit}:benchmarks/covert_witness/te_core/__init__.py")
    return next((x.split('"')[1] for x in init.splitlines() if x.startswith("VERSION = ")), "0")


def te_major(commit: str) -> int:
    return int(te_version(commit).split(".")[0])


def rescore_head() -> str:
    """Check out, detached, the newest commit touching te_core or the analysis on either branch whose te_core is
    TE_MAJOR or later, or RESCORE_AHEAD's head when it contains that commit; returns it."""
    for ref in (*RESCORE_REFS, RESCORE_AHEAD):
        subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    heads = [git("log", "-1", "--format=%H", f"origin/{r}", "--", *RESCORE_PATHS) for r in RESCORE_REFS]
    heads = [c for c in heads if c and te_major(c) >= TE_MAJOR]
    if not heads:
        raise SystemExit(f"v1cycle: no te_core {TE_MAJOR}.x on {', '.join(RESCORE_REFS)}")
    commit = git("log", "--no-walk=sorted", "-1", "--format=%H", *heads)
    ahead = subprocess.run(["git", "-C", str(RESCORE_WT), "rev-parse", "-q", "--verify", f"origin/{RESCORE_AHEAD}"],
                           capture_output=True, text=True, check=False).stdout.strip()
    if ahead and subprocess.run(["git", "-C", str(RESCORE_WT), "merge-base", "--is-ancestor", commit, ahead],
                                check=False).returncode == 0:
        commit = ahead
    git("checkout", "-q", "--detach", commit)
    return commit


def ledger_args(out: Path, root: Path | None = None) -> list[str]:
    ledger = str(root or os.environ.get("TE_LEDGER", str(Path.home() / ".research" / "te-ledger")))
    return ["--out", str(out), "--root", ledger, "--scorer", str(S.STATE), *(x for p in PINS for x in ("--pin", p))]


#: the states `research fetch` observes a node run to have ended in
ENDED = ("done", "failed", "cancelled", "refused", "dead", "timed_out", "exited")


def node_state(run: str) -> str:
    """The node run's state as `research fetch RUN` observes it (`... observed T: STATE rc=...`): one of ENDED, else
    running."""
    r = subprocess.run(["research", "fetch", run], capture_output=True, text=True, check=False)
    m = re.search(r" observed \S+: (\w+) rc=", r.stdout + r.stderr)
    return m.group(1) if m and m.group(1) in ENDED else "running"


def trace_resources(runs: int) -> tuple[int, int]:
    """Workers and GB to ask TRACE_NODE for a pack of `runs` runs: 24 workers peaked at 70 GB PSS (88 GB RSS) on
    r20261103-024017-a670, and a fixed 200 GB was refused at 03:34Z behind other reservations."""
    workers = max(1, min(24, runs))
    return workers, min(200, 24 + 6 * workers)


def measured_ledger(out: Path, launch: bool = True) -> dict:
    """trace40.py's measured ledger: the scorer's ledger indexed (rescore reads the measured copy, so nothing else
    rewrites the index), the valid runs with no trace yet and no node run tracing them launched on TRACE_NODE, every
    node run (this one and earlier refreshes') waited for up to TRACE_WAIT_S and taken as it ends (one still tracing
    is taken at a later refresh), and the copy built."""
    t40, cw = trace40.TE40_WT / "benchmarks" / "covert_witness", [sys.executable, str(HERE / "trace40.py")]
    subprocess.run([sys.executable, "-m", "te_core.ledger", "--root", str(trace40.LEDGER), "index"], cwd=t40,
                   check=True, capture_output=True)
    cache = trace40.cache()
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    pf = cache / "pending.json"
    pending = json.loads(pf.read_text()) if pf.exists() else {}
    took, log, deadline = [], {}, time.monotonic() + TRACE_WAIT_S

    def settle() -> None:
        for run in list(pending):
            state = node_state(run)
            if state == "running":
                continue
            # a failed or timed-out node run keeps the runs it finished (trace40.py's node writes each as it ends)
            t = subprocess.run([*cw, "take", run], capture_output=True, text=True, check=False)
            if t.returncode == 0:
                took.append(run)
            else:
                log.setdefault("not_taken", {})[run] = f"{state}: {(t.stderr.strip().splitlines() or [''])[-1][:200]}"
            del pending[run]

    # the ended node runs taken first, so the runs they failed (under an older CODE) are traced again in this one
    settle()
    pack = out / "trace40-pack.tar"
    skip = ",".join(r for runs in pending.values() for r in runs)
    p = subprocess.run([*cw, "pack", "--out", str(pack), "--skip", skip], capture_output=True, text=True, check=False) \
        if launch else None
    if p and p.returncode:
        # nothing new traced this time; the refresh scores the runs traced so far
        log["pack_failed"] = (p.stderr.strip().splitlines() or [f"rc {p.returncode}"])[-3:]
    packed = json.loads(p.stdout.strip().splitlines()[-1]) if p and not p.returncode else {"runs": 0}
    if packed["runs"]:
        runs = [r["run_id"] for r in json.loads(subprocess.run(["tar", "-xOf", str(pack), "./runs.json"],
                                                               capture_output=True, check=True).stdout)]
        workers, gb = trace_resources(len(runs))
        r = subprocess.run(["research", "run", "--on", TRACE_NODE, "--project", "verity", "--source", str(HERE.parents[2]),
                            "--cwd", "source", "--send", str(pack), "--mem-gb", str(gb), "--env", f"WORKERS={workers}",
                            "--timeout", "150m", "--", "bash", "benchmarks/covert_witness/scorer/job40.sh"],
                           capture_output=True, text=True, check=False, env=os.environ | {"AGENT_GUARD_LOCAL_OK": "1"})
        said = r.stdout + r.stderr
        run = next((w for w in said.split() if w.startswith("r20") and w.count("-") == 2), None)
        log["launch"] = said.strip().splitlines()[-3:]
        # a refused launch traces nothing, so its runs stay out of pending and the next refresh packs them again
        if run and not r.returncode and " refused " not in said:
            pending[run] = runs
        else:
            log["launch_refused"] = run
    pack.unlink(missing_ok=True)
    while launch and pending and time.monotonic() < deadline:
        time.sleep(60)
        settle()
    pf.write_text(json.dumps(pending, indent=1))
    built = json.loads(subprocess.run([*cw, "ledger"], capture_output=True, text=True, check=True).stdout)
    traced = json.loads((cache / "traced.json").read_text()) if (cache / "traced.json").exists() else {}
    waiting = {k: sum(r not in traced or bool(traced[r].get("error")) for r in v) for k, v in pending.items()}
    return {"te_tree": cache.name, "took": took, "tracing": waiting,
            "packed": packed["runs"], "traced_runs": sum(not v.get("error") for v in traced.values()),
            "untraced_runs": sum(bool(v.get("error")) for v in traced.values()), "ledger": built, **log}


def rescore(out: Path, gates: Path, final: Path | None = None, root: Path | None = None, extra: list[str] = ()) -> str:
    """te_core.rescore into `out` (module docstring, 2) on the ledger at `root` with `extra` options, or with
    FINAL_DIR's values for `final` through v1final.py and v1runs.py's per-run P4 (final, and the search's prefixed
    search-runs-); returns its last line."""
    out.mkdir(parents=True, exist_ok=True)
    if gates.exists() and not (out / "analysis-gates.json").exists():
        (out / "analysis-gates.json").write_bytes(gates.read_bytes())
    args = [*ledger_args(out, root), *extra]
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


def branches(out: Path, put: bool, root: Path | None = None) -> str:
    """te_core.panels' frontier-branches into `out` after the re-score (its gate cache reused), on the ledger at
    `root`, PUT with `put`; the PUT's status, or why there is none."""
    r = subprocess.run([sys.executable, "-m", "te_core.panels", FB, *ledger_args(out, root),
                        *(["--publish"] if put else [])],
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
    ap.add_argument("--no-trace", action="store_true", help="launch no node trace (finished ones are still taken)")
    ap.add_argument("--final", type=Path, metavar="FINAL_DIR",
                    help="the final table: te_core.rescore with FINAL_DIR's values (scorer.py final table; v1final.py); "
                         "posts FINAL-TABLE, not DIGEST")
    args = ap.parse_args()
    stamp, final = now(), args.final
    out = args.out or Path("/tmp/scorer/v1") / f"{'final-' if final else ''}{stamp.replace(':', '')}"
    commit = rescore_head()
    rs = out / "rescore"
    traced, root = (None, None) if final else (measured_ledger(out, not args.no_trace), trace40.MEASURED)
    last = rescore(rs, args.gates, final, root)
    put = "not PUT (final values)" if final else branches(rs, not args.no_put, root)
    variants = {}
    for name, extra in ({} if final else VARIANTS).items():
        try:
            variants[name] = rescore(rs / name, args.gates, None, root, extra)
        except SystemExit as e:
            variants[name] = str(e)[:300]
    what = "final table" if final else "refresh"
    fr = json.loads((rs / "frontier.json").read_text())
    scored = [json.loads(line) for line in (rs / "formalism.jsonl").open()]
    runs, n2 = {r["run_id"] for r in scored}, night2_runs()
    left = "not_analysed" if final else "not_measured"
    meta = {"stamp": stamp, "te_core": commit, "te_core_version": te_version(commit),
            "quantities": ("rescore_v1's analysis" if final else
                           "measured on each record's traced step (te_core 4.0, t literal; trace40.py)"),
            "formalism": "v2, the frozen tail always charged", "checkpoints": len(scored), "records": len(runs),
            "night2_records": len(runs & n2), left: len(fr.get(left) or []),
            "rule": "H and W* over every checkpoint (19:50Z)"}
    if traced:
        meta |= {"trace": traced, "variants": {k: str(v)[:200] for k, v in variants.items()},
                 "delta_t": "delta_t_bits" in git("show", f"{commit}:benchmarks/covert_witness/te_core/formalism.py")}
    trees = {k: out / "publish" / k for k in ("tables", "fb")}
    for t in trees.values():
        t.mkdir(parents=True, exist_ok=True)
    for name in ("frontier.md", "frontier.json", "formalism.jsonl", "rescore.log", "compare.md", "compare.json"):
        if (rs / name).exists():
            (trees["tables"] / name).write_bytes((rs / name).read_bytes())
    if traced and (trace40.cache() / "traced.json").exists():
        (trees["tables"] / "trace40-traced.json").write_bytes((trace40.cache() / "traced.json").read_bytes())
    for v in variants:
        for name in ("frontier.md", "frontier.json", "rescore.log", "formalism.jsonl"):
            if (rs / v / name).exists():
                (trees["tables"] / v).mkdir(exist_ok=True)
                (trees["tables"] / v / name).write_bytes((rs / v / name).read_bytes())
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
    measured = "trace" in meta
    parts = []
    for s in ("s100", "m100", "s50", "m50"):
        at = next((r for r in p2 if r["start_id"] == s and r["target_pct"] == REF_TARGET and r["policy"] == REFERENCE),
                  None)
        if at is None:
            continue
        if at["H_gates"] is None:
            parts.append(f"{s} not reached")
            continue
        loose = "; loose-end-attention" if measured and at["basis"] == "target" else ""
        q = (f"{fmt(at['S'])} (ρ {fmt(at['rho'])}, {at['basis']}; {at['method']} {at['record']}@{at['checkpoint']}"
             f"{loose})" if at["S"] is not None else "–")
        parts.append(f"{s}: H {fmt(at['H_gates'])}; S {q}")
    arts = ", ".join(f"{k} {v[:16]}" for k, v in ids.items())
    put = f"PUT {put}" if put[:1].isdigit() else put
    fb = (f"training/frontier-branches (te_core.panels): {fb_rows['uncut']:,} rows, the console's cut "
          f"{fb_rows['console']}, {put}" if fb_rows else f"training/frontier-branches {put}")
    head = (f"v2 final table (sealed shard: {meta['entries']} FINAL entries' reruns, {meta['entries_matched']} matched "
            f"at {meta['final_checkpoints']} checkpoints, 3 seeds' mean; every other record's search value at the same "
            "drop; " if "entries" in meta else "v2 refresh (")
    if not measured:
        return (f"{head}te_core {meta['te_core'][:10]}, formalism v2 with the frozen tail always charged: "
                f"{meta['records']} valid v2.3 records, {meta['night2_records']} from night 2, {meta['checkpoints']} "
                f"checkpoints, {meta['not_analysed']} not analysed yet; H and W* over every checkpoint, so S >= 1; ρ "
                "c/(c − d) where Fit holds for the training, else Slowdown's bound, a target) at −1% and "
                f"{REFERENCE}: " + "; ".join(parts) + f". {fb}; nothing else PUT. art: {arts}.")
    tr = meta["trace"]
    tracing = sum(tr["tracing"].values())
    dt = ("; Δ_t a straight-through run's fp32 latents, else Δ (00:47Z)" if meta.get("delta_t")
          else "; Δ_t not priced (this te_core has none)")
    return (f"{head}te_core {meta['te_core'][:10]} ({meta['te_core_version']}), formalism v2's quantities measured on "
            f"each record's traced step, t literal, the frozen tail always charged{dt}: {meta['records']} valid v2.3 "
            f"records measured, {meta['night2_records']} from night 2, {meta['checkpoints']} checkpoints; "
            f"{meta['not_measured']} valid checkpoints not measured yet"
            + (f" ({tracing} runs tracing on {TRACE_NODE})" if tracing else "")
            + (f" ({tr['packed']} runs' trace refused on {TRACE_NODE}, packed again at the next refresh)"
               if "launch_refused" in tr else "")
            + (f", {tr['untraced_runs']} runs whose trace failed (trace40-traced.json)" if tr["untraced_runs"] else "")
            + "; H and W* over every checkpoint, so S >= 1; ρ c/(c − d) where Fit holds for the training, else "
            "Slowdown's bound, a target, whose A1 charges no read of earlier tokens' keys and values "
            f"(loose-end-attention)) at −1% and {REFERENCE}: " + "; ".join(parts)
            + f". {fb}; nothing else PUT; t as the frozen tail and rescore_v1's analysis are in the tables art only. "
            f"art: {arts}.")


if __name__ == "__main__":
    sys.exit(main())
