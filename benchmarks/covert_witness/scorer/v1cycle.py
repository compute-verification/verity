"""Night 2's refresh on the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2), every 90
minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post] [--no-trace]
    python v1cycle.py --final FINAL_DIR [--out DIR] [--no-publish] [--no-post]

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore`; 22:16Z: through the integrator's
v2 te_core; 01:22Z: on te_core 4.0). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core, analysis/rescore_v1.py or the pin and
   starts (RESCORE_PATHS) on cursor/covert-witness-4207 or cursor/training-efficiency-4207 whose te_core is 4.x or
   later, or RESCORE_AHEAD holding it (merged with it when it doesn't, if they merge cleanly). Its pinned.json names
   the records scored (PINS: the pin, same_version and the pilot, trace40.pins). TE40_WT's capture traces while the
   integrator's te_core VERSION captures alike (CAPTURES). A new capture (`next_capture`) traces into its own cache,
   and until it has traced as many runs in scope as TE40_WT's and as many of v3's (`replaces`), the refresh scores
   on LAST's re-score commit and TE40_WT's traces; then TE40_WT moves to it.
2. Builds the measured ledger (trace40.py): v3's valid runs (the pilot's too under CAPTURE_FINAL's capture or with
   TE_TRACE_PILOT=1) traced, measured and ingested again in shards of TRACE_SHARD runs over the CPU pool (`research
   run --on auto`), at most TRACE_MAX_SHARDS a refresh, the runs traced at earlier refreshes kept.
3. Runs its `python -m te_core.rescore` on that ledger at PINS: per checkpoint of every valid record, v3's and the
   pilot's, measured, formalism v2's quantities measured on its traced step (t literal), Fit per kind of block and rho
   (c/(c - d) where every part fits, else Slowdown's bound, a target) on te_core.policy's (c, k) grid with the frozen
   tail always charged (Daniel, 22:00Z), H and W* over every checkpoint of every valid record (19:50Z), S = W*/H;
   frontier.md/json, formalism.jsonl and compare.md/json; then `--tail frozen` and `--quantities analysis`, which go
   to the evidence store only (01:22Z).
4. Once the builder's v2 is on that te_core (`policy_panels`), runs its `python -m te_core.algorithm_policy
   --publish` on the same ledger, with ladder-v3's points files as `--ladder-points` (`ladder_points`, 07:57Z),
   which PUTs training/algorithm-policy and training/frontier-fit; then `python -m
   te_core.panels frontier-branches` (Daniel, coordinator 23:56Z; the integrator's builder, 23:59Z), PUT until
   algorithm-policy is PUT with gain, gain_target and start_effective_gates on its rows (`fb_retired`; 05:22Z,
   07:48Z), then stored only. A refused PUT is reported, not fatal. te_core's five old panels stay in the re-score's
   directory, unpublished (23:56Z).
5. Publishes the tables and frontier-branches (uncut, its console cut and PNG) to the evidence store, copies the cut to
   PANELS_DIR, and posts one DIGEST line with the art ids, and the builder's coverage table (gain size by decade of
   rho) as one more when algorithm-policy writes one; an S whose rho is Slowdown's bound carries loose-end-attention
   (01:22Z: A1 charges no read of earlier tokens' keys and values).

`--final FINAL_DIR` scores the sealed shard's values (v1final.py, frontier-branches included, not PUT) and v1runs.py's
per-run P4 (final and search) for final.md, and posts FINAL-TABLE instead of DIGEST.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import scorer as S
import trace40

HERE = Path(__file__).resolve().parent
RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
#: what a re-score reads from its checkout: te_core, the analysis, and the pin and starts te_core reads (pin.py's
#: pinned.json, te_core.policy's starts_store.json)
RESCORE_PATHS = ("benchmarks/covert_witness/te_core", "benchmarks/covert_witness/analysis/rescore_v1.py",
                 "benchmarks/covert_witness/pinned.json", "benchmarks/covert_witness/pin.py",
                 "benchmarks/covert_witness/starts_store.json")
RESCORE_REFS = ("cursor/covert-witness-4207", "cursor/training-efficiency-4207")
#: the refresh runs on te_core 4.0 or later (coordinator 01:22Z): formalism v2's quantities measured on traced steps
TE_MAJOR = 4
# the scorer's te_core change awaiting the integrator's merge (Delta_t, coordinator 00:47Z), used while it holds theirs
RESCORE_AHEAD = os.environ.get("TE_RESCORE_AHEAD", "cursor/te-delta-t-40-516a")
#: te_core VERSIONs that capture alike (4.1's capture differs from 4.0's in prose only; 4.3 prices the gradient cut at
#: coded width from 4.2's traces, coordinator 07:48Z); a VERSION outside TE40_WT's group is a new capture, whose cache
#: traces every run again (`next_capture`)
CAPTURES = (("4.0", "4.1"), ("4.2", "4.3"))
#: the capture from which every later VERSION captures alike, and which traces the pilot's runs too, after v3's: the
#: toolsmith's 08:14Z line (te_core 4.4: the capture is final for tonight, retrace every run once)
CAPTURE_FINAL = "4.4"
#: where node traces run: `auto` spreads shards of TRACE_SHARD runs over the CPU pool (coordinator 07:48Z)
TRACE_NODE = os.environ.get("TE_TRACE_NODE", "auto")
TRACE_SHARD = 6
#: the most shards a refresh launches, so the pool (about 27 free slots at 08:50Z) keeps room for other lanes' runs;
#: the next refresh launches the rest, v3's first
TRACE_MAX_SHARDS = 30
#: a shard's custody key lifetime: `--on auto` adds its 4 h queue wait to the 150 min timeout (23,400 s), past the
#: default key's 6 h, so `research run` refuses the launch without it
TRACE_CUSTODY_TTL = "8h"
#: how long a refresh waits for its node runs; one still tracing is taken at the next refresh
TRACE_WAIT_S = 1080
#: the integrator's te_core whose capture TE40_WT's isn't (`next_capture`), checked out with no files: its runs traced
#: into its own cache while the refresh scores on LAST's re-score commit and TE40_WT's traces, until they replace those
#: (`replaces`)
NEXT_WT = Path(os.environ.get("TE_NEXT_WT", "/tmp/scorer/te-next"))
#: the re-score commit of the last refresh scored with TE40_WT's capture
LAST = trace40.STATE / "trace40" / "rescored.json"
#: what 4.0's rescore also writes, to the evidence store only (coordinator 01:22Z): t as the frozen tail, and the analysis
VARIANTS = {"frozen": ["--tail", "frozen"], "analysis": ["--quantities", "analysis"]}
#: the measurement commits scored: the re-score commit's pin, same_version and pilot (trace40.pins), set by `main`
PINS: list[str] = []
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


def rescore_head() -> tuple[str, str, str]:
    """Check out, detached, the newest commit touching RESCORE_PATHS on either branch whose te_core is TE_MAJOR or
    later, or RESCORE_AHEAD's head when it contains that commit, else RESCORE_AHEAD merged with it when they merge
    cleanly (the merge pushed to RESCORE_AHEAD); returns (the checkout, that newest commit, how it was chosen)."""
    for ref in (*RESCORE_REFS, RESCORE_AHEAD):
        subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    heads = [git("log", "-1", "--format=%H", f"origin/{r}", "--", *RESCORE_PATHS) for r in RESCORE_REFS]
    heads = [c for c in heads if c and te_major(c) >= TE_MAJOR]
    if not heads:
        raise SystemExit(f"v1cycle: no te_core {TE_MAJOR}.x on {', '.join(RESCORE_REFS)}")
    newest = git("log", "--no-walk=sorted", "-1", "--format=%H", *heads)
    ahead = subprocess.run(["git", "-C", str(RESCORE_WT), "rev-parse", "-q", "--verify", f"origin/{RESCORE_AHEAD}"],
                           capture_output=True, text=True, check=False).stdout.strip()
    if not ahead:
        git("checkout", "-q", "--detach", newest)
        return newest, newest, "the integrator's"
    if subprocess.run(["git", "-C", str(RESCORE_WT), "merge-base", "--is-ancestor", newest, ahead],
                      check=False).returncode == 0:
        git("checkout", "-q", "--detach", ahead)
        return ahead, newest, RESCORE_AHEAD
    git("checkout", "-q", "--detach", ahead)
    m = subprocess.run(["git", "-C", str(RESCORE_WT), "merge", "-q", "--no-edit", "-m",
                        f"Merge {newest[:11]} (te_core {te_version(newest)}) into {RESCORE_AHEAD} (v1cycle)", newest],
                       capture_output=True, text=True, check=False)
    if m.returncode:
        subprocess.run(["git", "-C", str(RESCORE_WT), "merge", "--abort"], capture_output=True, check=False)
        git("checkout", "-q", "--detach", newest)
        return newest, newest, f"the integrator's ({RESCORE_AHEAD} doesn't merge with it cleanly)"
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "-C", str(RESCORE_WT), "push", "-q", "origin", f"HEAD:refs/heads/{RESCORE_AHEAD}"],
                   capture_output=True, check=False)
    return head, newest, f"{RESCORE_AHEAD} merged with it"


def capture_version(te: Path) -> str:
    init = git("show", f"HEAD:{trace40.CW}/te_core/__init__.py", cwd=te)
    return next((x.split('"')[1] for x in init.splitlines() if x.startswith("VERSION = ")), "0")


def same_capture(a: str, b: str) -> bool:
    def final(v):
        return tuple(map(int, v.split("."))) >= tuple(map(int, CAPTURE_FINAL.split(".")))
    return a == b or any({a, b} <= set(g) for g in CAPTURES) or (final(a) and final(b))


def next_capture(newest: str) -> Path | None:
    """NEXT_WT when the integrator's newest te_core's capture isn't TE40_WT's (`CAPTURES`), else None: kept at the
    first commit seen with that capture, since a later commit of it captures alike and its cache (per te_core tree)
    would trace every run again. Tracing reads only its git objects, so it has no files."""
    have, want = capture_version(trace40.TE40_WT), te_version(newest)
    if same_capture(have, want):
        return None
    if not NEXT_WT.exists():
        git("worktree", "add", "-q", "--no-checkout", "--detach", str(NEXT_WT), newest, cwd=trace40.REPO)
    if not same_capture(capture_version(NEXT_WT), want):
        git("reset", "-q", "--soft", newest, cwd=NEXT_WT)
    return NEXT_WT


def v3_scope() -> set[str]:
    """The runs in scope measured at the pin's commit or its same_version (not the pilot's)."""
    p = trace40.at_pin("pinned.json", os.environ.get("TE_PIN_REV", trace40.PIN_REV))
    return {r["run_id"] for r in trace40.scope() if r["meas"] in {p["commit"], *p.get("same_version", [])}}


def covered(te: Path) -> tuple[int, int, int]:
    """(the v3 runs in scope, how many of them `te`'s capture has traced or failed under CODE, how many it traced)."""
    c = trace40.cache(te)
    new = json.loads((c / "traced.json").read_text()) if (c / "traced.json").exists() else {}
    want = v3_scope()
    return len(want), len(want & trace40.done(c)), sum(not (new.get(k) or {"error": 1}).get("error") for k in want)


def traced_runs(te: Path) -> tuple[int, int]:
    """(the runs in scope `te`'s capture traced, those of v3's)."""
    f = trace40.cache(te) / "traced.json"
    ok = {k for k, v in (json.loads(f.read_text()) if f.exists() else {}).items() if not v.get("error")}
    return len(ok & {r["run_id"] for r in trace40.scope()}), len(ok & v3_scope())


def replaces(new: Path) -> bool:
    """Whether the capture `new` replaces TE40_WT's: it traced at least as many of the runs in scope and of v3's, so
    neither the pilot nor v3 loses a row by the switch (every row traced before a capture is refused under its
    te_core). A v3 scope that grows with each launch never waits on its last run."""
    return all(a >= b for a, b in zip(traced_runs(new), traced_runs(trace40.TE40_WT)))


def to_trace(te: Path, busy: set[str]) -> list[str]:
    """The runs in scope with no trace in `te`'s cache and no node run tracing them, v3's first, then the pilot's
    under CAPTURE_FINAL's capture or when TE_TRACE_PILOT is set."""
    v3, done = v3_scope(), trace40.done(trace40.cache(te)) | busy
    pilot = os.environ.get("TE_TRACE_PILOT") == "1" or same_capture(capture_version(te), CAPTURE_FINAL)
    runs = [r["run_id"] for r in trace40.scope() if r["run_id"] not in done and (r["run_id"] in v3 or pilot)]
    return sorted(runs, key=lambda r: r not in v3)


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
    """Workers (CPUs) and GB to ask TRACE_NODE for a shard of `runs` runs: 24 workers peaked at 70 GB PSS (88 GB RSS) on
    r20261103-024017-a670's m100-heavy bulk, so 4 GB a worker; a shard of 6 asks 36 GB, under a research pool slot's
    40 (vy-nebius-cpu-1's), which `--on auto` needs to place it."""
    workers = max(1, min(24, runs))
    return workers, min(200, 12 + 4 * workers)


def measured_ledger(out: Path, launch: bool = True, te: Path = trace40.TE40_WT, build: bool = True,
                    over: Path | None = None) -> dict:
    """trace40.py's measured ledger: the scorer's ledger indexed (rescore reads the measured copy, so nothing else
    rewrites the index), the valid runs with no trace yet in `te`'s cache and no node run tracing them launched on
    TRACE_NODE, every node run (this one and earlier refreshes') waited for up to TRACE_WAIT_S and taken as it ends (one
    still tracing is taken at a later refresh), and the copy built (`build`), with the runs `over`'s capture traced
    taken from there (trace40.ledger)."""
    cw = [sys.executable, str(HERE / "trace40.py")]
    if build:
        subprocess.run([sys.executable, "-m", "te_core.ledger", "--root", str(trace40.LEDGER), "index"],
                       cwd=te / "benchmarks" / "covert_witness", check=True, capture_output=True)
    cache = trace40.cache(te)
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
            t = subprocess.run([*cw, "take", run, "--te", str(te)], capture_output=True, text=True, check=False)
            if t.returncode == 0:
                took.append(run)
            else:
                log.setdefault("not_taken", {})[run] = f"{state}: {(t.stderr.strip().splitlines() or [''])[-1][:200]}"
            del pending[run]

    # the ended node runs taken first, so the runs they failed (under an older CODE) are traced again in this one
    settle()
    todo = to_trace(te, {r for runs in pending.values() for r in runs})[:TRACE_SHARD * TRACE_MAX_SHARDS] if launch else []
    packed = {"runs": len(todo)}
    for i in range(0, len(todo), TRACE_SHARD):
        shard, pack = todo[i:i + TRACE_SHARD], out / f"trace40-pack-{i // TRACE_SHARD}.tar"
        p = subprocess.run([*cw, "pack", "--out", str(pack), "--te", str(te), "--only", ",".join(shard)],
                           capture_output=True, text=True, check=False)
        if p.returncode:
            log.setdefault("pack_failed", []).append((p.stderr.strip().splitlines() or [f"rc {p.returncode}"])[-1])
            continue
        runs = [r["run_id"] for r in json.loads(subprocess.run(["tar", "-xOf", str(pack), "./runs.json"],
                                                               capture_output=True, check=True).stdout)]
        workers, gb = trace_resources(len(runs))
        r = subprocess.run(["research", "run", "--on", TRACE_NODE, "--project", "verity", "--source", str(HERE.parents[2]),
                            "--cwd", "source", "--send", str(pack), "--cpus", str(workers), "--mem-gb", str(gb),
                            "--env", f"WORKERS={workers}", "--timeout", "150m", "--custody-ttl", TRACE_CUSTODY_TTL, "--",
                            "bash", "benchmarks/covert_witness/scorer/job40.sh"],
                           capture_output=True, text=True, check=False, env=os.environ | {"AGENT_GUARD_LOCAL_OK": "1"})
        said = r.stdout + r.stderr
        run = next((w for w in said.split() if w.startswith("r20") and w.count("-") == 2), None)
        log.setdefault("launch", []).append(said.strip().splitlines()[-1:])
        # a refused launch traces nothing, so its runs stay out of pending and the next refresh packs them again
        if run and not r.returncode and " refused" not in said:
            pending[run] = runs
        else:
            log.setdefault("launch_refused", []).append(run)
        pack.unlink(missing_ok=True)
    while launch and pending and time.monotonic() < deadline:
        time.sleep(60)
        settle()
    pf.write_text(json.dumps(pending, indent=1))
    built = json.loads(subprocess.run([*cw, "ledger", "--te", str(te), *(["--over", str(over)] if over else [])],
                                      capture_output=True, text=True, check=True).stdout) if build else None
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


def v3_starts(commit: str) -> list[str]:
    """The starts data v3's branches launch from (starts_store.json's entries on V3_TOKENIZER at `commit`)."""
    return [n for n, s in trace40.at_pin("starts_store.json", commit)["starts"].items()
            if (s.get("shape") or {}).get("tokenizer") == trace40.V3_TOKENIZER]


#: what marks the builder's v2 in te_core.algorithm_policy (gains at 1.1, 2 and 11, coordinator 04:35Z)
AP_V2 = "gain_target"
FIT = "frontier-fit"
#: where ladder-v3 writes the v3 ladder's points (coordinator 07:57Z: every cycle passes them as --ladder-points), and
#: the schemas te_core.frontier_fit's `ladder_points` reads
LADDER = ("cursor/te-ladder-v3-4207", "benchmarks/covert_witness/islands/ladder-v3")
LADDER_SCHEMAS = ("te-ladder/F/v1", "te-ladder/fit/v1", "te-ladder/report/v2")


def ladder_points(out: Path) -> list[Path]:
    """The ladder's points files on LADDER's branch, each JSON file of its directory whose schema `ladder_points` reads
    (or that carries its `F` or `compute_optimal`), written into `out`, and TE_LADDER_POINTS's paths (os.pathsep)."""
    ref, where = LADDER
    subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    names = subprocess.run(["git", "-C", str(RESCORE_WT), "ls-tree", "-r", "--name-only", f"origin/{ref}", "--", where],
                           capture_output=True, text=True, check=False).stdout.split()
    found = [Path(p) for p in os.environ.get("TE_LADDER_POINTS", "").split(os.pathsep) if p]
    for name in (n for n in names if n.endswith(".json")):
        try:
            doc = json.loads(git("show", f"origin/{ref}:{name}"))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict) and (doc.get("schema") in LADDER_SCHEMAS or "F" in doc or "compute_optimal" in doc):
            out.mkdir(parents=True, exist_ok=True)
            (out / Path(name).name).write_text(json.dumps(doc))
            found.append(out / Path(name).name)
    return found


def policy_panels(out: Path, commit: str, put: bool, root: Path | None) -> dict:
    """Once the re-score's te_core holds the builder's v2 (AP_V2 in te_core.algorithm_policy): its run on the same
    ledger, with `--publish` under `put`, which builds and PUTs training/algorithm-policy and training/frontier-fit
    both; {panel: its PUT status or why there is none} and `coverage`, the coverage (gain size by decade of rho) as
    the board line it prints, else its table's rows."""
    src = lambda p: subprocess.run(["git", "-C", str(RESCORE_WT), "show", f"{commit}:benchmarks/covert_witness/te_core/{p}"],
                                   capture_output=True, text=True, check=False).stdout
    if AP_V2 not in src("algorithm_policy.py"):
        return {"algorithm-policy": "not run (the builder's v2 isn't on this te_core yet)"}
    d = out / "algorithm-policy"
    ladder = ladder_points(out / "ladder")
    r = subprocess.run([sys.executable, "-m", "te_core.algorithm_policy", *ledger_args(d, root),
                        *(x for p in ladder for x in ("--ladder-points", str(p))), *(["--publish"] if put else [])],
                       cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True, check=False)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.log").write_text(r.stdout + r.stderr)
    why = (f"not PUT: {(r.stderr.strip().splitlines() or [f'rc {r.returncode}'])[-1][:200]}" if r.returncode
           else "not PUT (--no-put)")
    res = {"algorithm-policy": why, FIT: why}
    for x in r.stdout.splitlines():
        if x.startswith("PUT ") and ": " in x:
            url, status = x[4:].rsplit(": ", 1)
            res[url.rstrip("/").rsplit("/", 1)[-1]] = status[:200]
    board = next((x.removeprefix("board: ") for x in r.stdout.splitlines() if x.startswith("board: coverage")), None)
    res["coverage"] = [board] if board else coverage(d, r.stdout)
    res["ladder"] = [p.name for p in ladder]
    return res


#: what training/algorithm-policy's console copy carries once frontier-branches is no longer PUT, only stored
#: (coordinator 05:22Z and 07:48Z)
FB_SUCCESSOR = ("gain", "gain_target", "start_effective_gates")
FB_RETIRED = ("training/algorithm-policy is PUT with " + ", ".join(FB_SUCCESSOR)
              + " on its rows; frontier-branches is in the evidence store only")


def fb_retired(out: Path, policy: dict) -> bool:
    """Whether this refresh PUT training/algorithm-policy with FB_SUCCESSOR among its console copy's columns."""
    f = out / "algorithm-policy" / "console" / "algorithm-policy.json"
    return (str(policy.get("algorithm-policy", ""))[:1] == "2" and f.exists()
            and set(FB_SUCCESSOR) <= set(json.loads(f.read_text())["columns"]))


def coverage(d: Path, stdout: str) -> list[str]:
    """The builder's coverage table (gain size by decade of rho) as one line a row: a `coverage.md` it wrote under `d`,
    else the Markdown table under a line naming coverage in its output; [] when it has none."""
    files = sorted(d.rglob("*coverage*.md"))
    lines = files[0].read_text().splitlines() if files else stdout.splitlines()
    if not files:
        at = next((i for i, x in enumerate(lines) if "coverage" in x.lower()), None)
        lines = [] if at is None else lines[at + 1:]
    table = []
    for x in lines:
        if x.startswith("|"):
            if not re.fullmatch(r"[|\s:-]+", x):
                table.append(", ".join(c.strip() for c in x.strip("|").split("|")))
        elif table:
            break
    return table


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
    commit, newest, chosen = rescore_head()
    os.environ["TE_PIN_REV"] = commit
    PINS[:] = trace40.pins(commit)
    nxt, retrace = (None, None) if final else (next_capture(newest), None)
    if nxt:
        # every row traced before a new capture is refused under its te_core, so the refresh scores on the last
        # capture until the new one's traces replace it
        retrace = measured_ledger(out / "next-capture", not args.no_trace, nxt, build=False)
        retrace |= dict(zip(("of", "covered", "traced"), covered(nxt)), te_core=te_version(newest),
                        commit=git("rev-parse", "HEAD", cwd=nxt))
        if replaces(nxt) or not LAST.exists():
            git("checkout", "-q", "--detach", retrace["commit"], cwd=trace40.TE40_WT)
            nxt, retrace = None, None
        else:
            commit = json.loads(LAST.read_text())["commit"]
            git("checkout", "-q", "--detach", commit)
            chosen = (f"the last re-score on te_core {capture_version(trace40.TE40_WT)}'s capture, until te_core "
                      f"{te_version(newest)}'s traces replace it")
            os.environ["TE_PIN_REV"] = commit
            PINS[:] = trace40.pins(commit)
    if not nxt and not final:
        LAST.write_text(json.dumps({"commit": commit, "te_core": te_version(commit), "stamp": stamp}) + "\n")
    trace_version = capture_version(trace40.TE40_WT)
    # the runs the new capture has traced are scored on its traces already, where the re-score's te_core reads them
    over = nxt if nxt and same_capture(te_version(commit), capture_version(nxt)) else None
    rs = out / "rescore"
    traced, root = (None, None) if final else (measured_ledger(out, not args.no_trace and not nxt, over=over),
                                               trace40.MEASURED)
    last = rescore(rs, args.gates, final, root)
    policy = {} if final else policy_panels(out, commit, not args.no_put, root)
    retired = not final and fb_retired(out, policy)
    put = "not PUT (final values)" if final else branches(rs, not args.no_put and not retired, root)
    if retired:
        put = f"not PUT ({FB_RETIRED})"
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
    v3 = v3_starts(commit)
    by_start = {}
    for r in scored:
        by_start.setdefault(r["start"], set()).add(r["run_id"])
    meta = {"stamp": stamp, "te_core": commit, "te_core_version": te_version(commit), "te_core_chosen": chosen,
            "capture": trace_version, "over": bool(over),
            "integrator_te_core": newest, "pins": PINS,
            "quantities": ("rescore_v1's analysis" if final else
                           f"measured on each record's traced step (te_core {trace_version}'s capture, t literal; "
                           "trace40.py)"),
            "formalism": "v2, the frozen tail always charged", "checkpoints": len(scored), "records": len(runs),
            "night2_records": len(runs & n2), left: len(fr.get(left) or []),
            "v3": {s: {"records": len(by_start.get(s, ())), "checkpoints": sum(r["start"] == s for r in scored)}
                   for s in v3},
            "pilot_start": sum(v == "pilot-start" for v in (fr.get("skipped") or {}).values()),
            "rule": "H and W* over every checkpoint (19:50Z)"}
    if policy:
        meta["algorithm_policy"] = {k: v for k, v in policy.items() if k not in ("coverage", "ladder")}
        if "ladder" in policy:
            meta["ladder_points"] = policy["ladder"]
        apc = out / "algorithm-policy" / "console" / "algorithm-policy.json"
        if apc.exists():
            q = json.loads(apc.read_text())
            meta["algorithm_policy_cells"] = {c: sorted({str(r[q["columns"].index(c)]) for r in q["rows"]})
                                              for c in ("reading", "te_core", "binding_kind") if c in q["columns"]}
            meta["algorithm_policy_cells"]["rho_console"] = "rho_console" in q["columns"]
    if retrace:
        meta["next_capture"] = retrace
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
    for name in ("algorithm-policy", FIT):
        if (out / name).is_dir():
            shutil.copytree(out / name, trees["tables"] / name, dirs_exist_ok=True)
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
    line = digest(json.loads((rs / "p2-slowdown-vs-gain.json").read_text()),
                  meta | {"version_cell": bool(fb and "te_core_version" in fb["columns"])}, ids, put, fb_rows)
    (out / "digest.txt").write_text(line + "\n")
    said = policy.get("coverage") or []
    cov = (f"training/algorithm-policy (te_core {meta['te_core_version']}; tables art {ids.get('tables', '–')[:16]}) "
           + said[0] if said[:1] and said[0].startswith("coverage at ") else
           f"coverage (te_core.algorithm_policy, te_core {meta['te_core_version']}: gain size by decade of ρ, the "
           f"frontier point or the best failed attempt in each cell; tables art {ids.get('tables', '–')[:16]}): "
           + "; ".join(said) + "." if said else None)
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "FINAL-TABLE" if final else "DIGEST", line],
                       check=True)
        if cov:
            subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "DIGEST", cov], check=True)
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
    v3 = list(meta.get("v3") or {})
    for s in (*v3, "s100", "m100", "s50", "m50"):
        at = next((r for r in p2 if r["start_id"] == s and r["target_pct"] == REF_TARGET and r["policy"] == REFERENCE),
                  None)
        if at is None:
            if s in v3 and meta["v3"][s]["records"]:
                parts.append(f"{s} (v3) not reached")
            continue
        s = f"{s} (v3)" if s in v3 else s
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
    where = "the CPU pool" if TRACE_NODE == "auto" else TRACE_NODE
    refused = tr["ledger"].get("refused_runs") or []
    nx = meta.get("next_capture")
    nx_refused = len(nx.get("launch_refused") or []) if nx else 0
    nx_said = (f" (te_core {nx['te_core']} prices a row from its own capture ({nx['commit'][:11]}), which is tracing "
               f"every run over the CPU pool, v3's first: {nx['covered']} of v3's {nx['of']} done so far, "
               f"{nx['traced']} traced"
               + (f", {nx_refused} of {len(nx.get('launch') or [])} shard launches refused, packed again at the next "
                  "refresh" if nx_refused else "") + "; "
               f"this refresh scores on te_core {meta['te_core_version']} over {meta.get('capture', '?')}'s capture"
               + (f", with the new capture's traces of its {nx['traced_runs']} traced runs over it (their rows reading "
                  "A)" if meta.get("over") else "")
               + " until it traces as many runs, and as many of v3's)" if nx else "")
    dt = ("; Δ_t a straight-through run's fp32 latents, else Δ (00:47Z)" if meta.get("delta_t")
          else "; Δ_t not priced (this te_core has none)")
    v3_said = ", ".join(f"{s} {x['records']} ({x['checkpoints']} checkpoints)" for s, x in meta["v3"].items()
                        if x["records"]) or "none measured yet"
    ap, cells = meta.get("algorithm_policy") or {}, meta.get("algorithm_policy_cells") or {}
    cells_said = (f" (every row: reading {'/'.join(cells.get('reading') or ['–'])}, te_core "
                  f"{'/'.join(cells.get('te_core') or ['–'])}"
                  + (", rho_console at c = 5,120 x 2 x 45e6 x 740 gates, k = 65,536 bits" if cells.get("rho_console")
                     else "")
                  + (f", the one bound's binding kind {'/'.join(cells['binding_kind'])} with what it rests on and each "
                     "kind's term (reduction and update as sensitivities, 08:26Z)" if cells.get("binding_kind") else "")
                  + ", rho_default)" if cells else "")
    ladder = meta.get("ladder_points")
    ladder_said = ("" if ladder is None else f" (v3's fit with ladder-v3's points: {', '.join(ladder)})" if ladder
                   else f" (no ladder points file on {LADDER[0]} yet, so v3's fit has only its starts)")
    ap_said = "; ".join(f"training/{k} {'PUT ' if v[:1].isdigit() else ''}{v}"
                        + (cells_said if k == "algorithm-policy" else ladder_said if k == FIT else "")
                        for k, v in ap.items())
    return (f"{head}te_core {meta['te_core'][:10]} ({meta['te_core_version']}){nx_said}, formalism v2's quantities "
            f"measured on each record's traced step, t literal, the frozen tail always charged{dt}: {meta['records']} valid "
            f"records measured at the pin's commits and the pilot's, {meta['night2_records']} from night 2, "
            f"{meta['checkpoints']} checkpoints; v3: {v3_said}"
            + (f"; {meta['pilot_start']} v3 checkpoints from a pilot start left out (pilot-start)"
               if meta.get("pilot_start") else "")
            + f"; {meta['not_measured']} valid checkpoints not measured yet"
            + (f" ({tracing} runs tracing on {where})" if tracing else "")
            + (f" ({tr['packed']} runs' trace refused on {where}, packed again at the next refresh)"
               if tr.get("launch_refused") else "")
            + (f", {tr['untraced_runs']} runs whose trace failed (trace40-traced.json)" if tr["untraced_runs"] else "")
            + (f"; {len(refused)} run{'s' * (len(refused) != 1)} refused, not scored, a trace showing an affected "
               f"share of 0 or u(1) = 0 beside a nonzero E (04:40Z): {', '.join(refused)}" if refused else "")
            + "; H and W* over every checkpoint, so S >= 1; ρ c/(c − d) where Fit holds for the training, else "
            "Slowdown's bound, a target, whose A1 charges no read of earlier tokens' keys and values "
            f"(loose-end-attention)) at −1% and {REFERENCE}: " + "; ".join(parts)
            + f". {fb}" + (f"; {ap_said}" if ap_said else "")
            + "; nothing else PUT; t as the frozen tail and rescore_v1's analysis are in the tables art only"
            + ("; every row carries te_core_version" if meta.get("version_cell") else "") + f". art: {arts}.")


if __name__ == "__main__":
    sys.exit(main())
