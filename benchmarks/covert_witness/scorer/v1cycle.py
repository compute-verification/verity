"""Night 2's refresh on the training formalism v2 (note:training-20261102T2215Z-report-training-formalism-v2), every 90
minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post] [--no-trace]
    python v1cycle.py --final FINAL_DIR [--out DIR] [--no-publish] [--no-post]

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore`; 22:16Z: through the integrator's
v2 te_core; 01:22Z: on te_core 4.0). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core, analysis/rescore_v1.py or the pin and
   starts (RESCORE_PATHS) on cursor/covert-witness-4207 (RESCORE_REFS) whose te_core is 4.x or later, or RESCORE_AHEAD holding it (merged with it when it doesn't, if they merge cleanly). Its pinned.json names
   the records scored (PINS: the pin, same_version and the pilot, trace40.pins). TE40_WT's capture traces while the
   integrator's te_core VERSION captures alike (CAPTURES). A new capture (`next_capture`) traces into its own cache,
   and until it has traced as many runs in scope as TE40_WT's and as many of v3's (`replaces`), the refresh scores
   TE40_WT's traces with the new capture's over them, on that checkout where TE40_WT's capture is READS_SINCE's or
   later, else on LAST's re-score commit; then TE40_WT moves to it.
2. Builds the measured ledger (trace40.py): v3's valid runs (the pilot's too under CAPTURE_FINAL's capture or with
   TE_TRACE_PILOT=1) traced, measured and ingested again in shards of TRACE_SHARD runs over TRACE_POOLS' free slots,
   a long-tier record in its tier's layer (TIER_FROM), the runs traced at earlier refreshes kept, and each run whose
   trace lists an elementwise product retraced in the fix layer over its layer (FIX, `listed`), scored from there.
3. Runs its `python -m te_core.rescore` on that ledger at PINS: per checkpoint of every valid record, v3's and the
   pilot's, measured, formalism v2's quantities measured on its traced step (t literal), Fit per kind of block and rho
   (c/(c - d) where every part fits, else Slowdown's bound, a target) on te_core.policy's (c, k) grid with the frozen
   tail always charged (Daniel, 22:00Z), H and W* over every checkpoint of every valid record (19:50Z), S = W*/H;
   frontier.md/json, formalism.jsonl and compare.md/json; then `--tail frozen` and `--quantities analysis`, which go
   to the evidence store only (01:22Z).
4. Once the builder's v2 is on that te_core (`policy_panels`), runs its `python -m te_core.algorithm_policy
   --publish` on the same ledger, with ladder-v3's points files as `--ladder-points` (`ladder_points`, 07:57Z),
   else    te_core.frontier_fit's own (its LADDER, built from ladder-v3's manifest), which PUTs training/algorithm-policy and training/frontier-fit,
   its rows labelled first (POLICY, v1labels: a void row unpriced, a frozen embedding's price under review) and the
   coverage line saying which of its points are priced (`v1labels.priced`); then `python -m
   te_core.panels frontier-branches` (Daniel, coordinator 23:56Z; the integrator's builder, 23:59Z), stored only
   since algorithm-policy carries gain, gain_target and start_effective_gates (FB_RETIRED; 05:22Z, 07:48Z), so a
   refresh whose algorithm-policy isn't PUT PUTs no frontier either. A refused PUT is reported, not fatal. te_core's
   five old panels stay in the re-score's directory, unpublished (23:56Z).
5. Deletes the measured ledger, which the next refresh builds afresh, and publishes the tables (the re-score's files
   linked into its tree) and frontier-branches (uncut, its console cut and PNG) to the evidence store, copies the cut to
   PANELS_DIR, deletes its re-score and publish directories once both are in the store, and posts one DIGEST line with the art ids, and the builder's coverage table (gain size by decade of
   rho) as one more when algorithm-policy writes one; an S whose rho is Slowdown's bound carries loose-end-attention
   (01:22Z: A1 charges no read of earlier tokens' keys and values).

On a CPU node (refresh-node.sh, which sets TE_TRACE_HERE: the agent VM only edits) a refresh holds no store write key,
no console token and no ssh: its shards trace beside it (`trace_here`) until TE_HERE_UNTIL, the node runs of earlier
refreshes are taken from their run records in the store (`take_src`), its tables and frontier-branches go to its run's
out/, which custody stores (`here_out`), and algorithm-policy builds its console bodies on the panels the VM read as
served (TE_SERVED), leaving their PUT and post.json's board lines to the VM's loop (node_loop.py).

`--final FINAL_DIR` scores the sealed shard's values (v1final.py, frontier-branches included, not PUT) and v1runs.py's
per-run P4 (final and search) for final.md, and posts FINAL-TABLE instead of DIGEST.
"""
import argparse
import atexit
import collections
import gzip
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
# research from this tree: pool_room sends this tree's slot.py, and an older research.auto reads a slot without mem= as 0 GB
sys.path.insert(0, str(HERE.parents[2] / "tools" / "research" / "src"))

import scorer as S
import trace40
import v1labels

RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
#: what a re-score reads from its checkout: te_core, the analysis, and the pin and starts te_core reads (pin.py's
#: pinned.json, te_core.policy's starts_store.json)
RESCORE_PATHS = ("benchmarks/covert_witness/te_core", "benchmarks/covert_witness/analysis/rescore_v1.py",
                 "benchmarks/covert_witness/pinned.json", "benchmarks/covert_witness/pin.py",
                 "benchmarks/covert_witness/starts_store.json")
#: the integrator's branch, where the scorer's te_core is merged (training-efficiency-4207's te_core lacks
#: formalism.published, the integrator's 10:24Z)
RESCORE_REFS = ("cursor/covert-witness-4207",)
#: the refresh runs on te_core 4.0 or later (coordinator 01:22Z): formalism v2's quantities measured on traced steps
TE_MAJOR = 4
# the scorer's te_core change awaiting the integrator's merge (Delta_t, coordinator 00:47Z), used while it holds theirs
RESCORE_AHEAD = os.environ.get("TE_RESCORE_AHEAD", "cursor/te-delta-t-40-516a")
#: te_core VERSIONs that capture alike (4.1's capture differs from 4.0's in prose only; 4.3 prices the gradient cut at
#: coded width from 4.2's traces, coordinator 07:48Z); a VERSION outside TE40_WT's group is a new capture, whose cache
#: traces every run again (`next_capture`)
CAPTURES = (("4.0", "4.1"), ("4.2", "4.3"), ("4.4", "4.5", "4.6", "4.7"), ("4.8", "4.9", "4.10"))
#: the capture from which every later VERSION captures alike, and which traces the pilot's runs too, after v3's: te_core
#: 4.8 saves a traced step's product rows in every trace (the toolsmith's 66162c92b1 line; coordinator 12:20Z), and
#: 4.12.1's stores everything 4.12.x reads, each row's fed, P and gathered and the rows at each k of the console's dial
#: (the toolsmith's 16:34Z line), and 4.12.7's lists the matrix products a step writes with elementwise ops: a trace
#: captured before 4.12.7 reads unmeasured for the fresh term (the toolsmith's 18:53Z line); 4.12.9's finds a broadcast
#: product through a map, a copy or a multiplication by ones, and reads every earlier trace 'retrace' for the products
#: term. The capture is frozen at 4.12.9 through Wednesday's finals (coordinator 20:11Z)
CAPTURE_FINAL = "4.12.9"
#: the (machine, pool, cap) node traces go to, each pool's research slots any lane's (auto.py's SPILL, and cpu-9's):
#: a shard takes a slot free there now past its waiters (`pool_room`), at most `cap` shards in flight there across every
#: layer's cache, half the pool's slots, so the rest stays other lanes'. Named machines, never `--on auto`, which
#: misplaces runs until #7113 lands (coordinator 15:08Z); not only cpu-9, where they waited 19th in line (17:5xZ).
#: TE_TRACE_POOLS overrides it: MACHINE:POOL:CAP,...
TRACE_POOLS = [(m, p, int(c)) for m, p, c in (x.split(":") for x in os.environ.get(
    "TE_TRACE_POOLS", "vy-nebius-cpu-5:research:6,vy-nebius-cpu-6:research:4,vy-nebius-cpu-9:research:4,"
    "vy-nebius-cpu-1:research:10,vy-nebius-cpu-3:research:4,vy-nebius-cpu-2:research:2").split(","))]
#: the lane a pool slot names (`research run --pool` refuses a run without one)
TRACE_LANE = os.environ.get("RESEARCH_LANE", "te-scorer")
TRACE_SHARD = 6
#: each node run in flight and the machine it went to ({run: machine}; a list, before TRACE_POOLS, was cpu-9's)
ON_POOL = trace40.STATE / "trace40" / "on-pool.json"
#: budget tiers whose records a capture replays at their own tier only from this te_core on (4.12.3's
#: `capture.script.tier_of` sets CW_TIER from run.json's evaluation.budget.tier; 4.12.1's sets none, so a long-tier
#: record replays at the search tier and refuses at step 0 or ends before its traced step, coordinator 17:5xZ). An
#: older layer traces none of them; its tier's layer (NEXT_WT-VERSION-TIER, `tier_layer`) traces only them, scored over
#: the layer whose capture it shares
TIER_FROM = {"long": "4.12.3"}
#: a shard's custody key lifetime: `--on auto` adds its 4 h queue wait to the 150 min timeout (23,400 s), past the
#: default key's 6 h, so `research run` refuses the launch without it
TRACE_CUSTODY_TTL = "8h"
#: how long a refresh waits for its node runs; one still tracing is taken at the next refresh
TRACE_WAIT_S = 1080
#: the integrator's te_core whose capture TE40_WT's isn't (`next_capture`), checked out with no files: its runs traced
#: into its own cache while the refresh scores TE40_WT's traces with them over those, until they replace those
#: (`replaces`)
NEXT_WT = Path(os.environ.get("TE_NEXT_WT", "/tmp/scorer/te-next"))
#: the re-score commit of the last refresh scored with TE40_WT's capture
LAST = trace40.STATE / "trace40" / "rescored.json"
#: the first capture whose traces a later te_core doesn't refuse for their capture (formalism.refusal: read-backs
#: unchecked before 4.2): over one, the refresh scores on the integrator's newest te_core, not LAST's
READS_SINCE = "4.2"
#: what 4.0's rescore also writes, to the evidence store only (coordinator 01:22Z): t as the frozen tail, and the analysis
VARIANTS = {"frozen": ["--tail", "frozen"], "analysis": ["--quantities", "analysis"]}
#: the measurement commits scored: the re-score commit's pin, same_version and pilot (trace40.pins), set by `main`
PINS: list[str] = []
PANELS_DIR = Path(os.environ.get("TE_PANELS_DIR", "/opt/cursor/artifacts/te-panels"))
#: where each refresh writes its directory
OUT_ROOT = Path(os.environ.get("TE_OUT_ROOT", "/tmp/scorer/v1"))
#: a refresh on a CPU node (refresh-node.sh: the agent VM only edits), which can launch no research run and holds no
#: store write key, no console token and no ssh: its shards trace here, beside it, in this kept directory
#: (`trace_here`), the earlier shards' run records come from the store (`take_src`), and its tables stay in the run's
#: out/ for custody (`here_out`), the VM PUTting the console's copies (`VM_PUT`) and posting post.json's lines
HERE_DIR = Path(os.environ["TE_TRACE_HERE"]) if os.environ.get("TE_TRACE_HERE") else None
#: the cores and GB of its reservation a refresh on the node keeps for its re-score; its shards trace on the rest
HERE_CORES, HERE_GB = 4, 48
#: what such a refresh's algorithm-policy run says for each panel it would PUT, the bodies in its console/: the VM's
#: PUT status replaces it in the board line
VM_PUT = "left to the VM's PUT"
#: the shards tracing here ({name: (process, GB asked)}, `trace_here`)
LOCAL: dict[str, tuple[subprocess.Popen, int]] = {}
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


def version(v: str) -> tuple[int, ...]:
    return tuple(map(int, v.split(".")))


def same_capture(a: str, b: str) -> bool:
    def final(v):
        return version(v) >= version(CAPTURE_FINAL)
    return a == b or any({a, b} <= set(g) for g in CAPTURES) or (final(a) and final(b))


#: a later commit of a layer's capture that traces only runs the layer's capture failed on or listed falsely
#: (NEXT_WT-VERSION-fix, `fixed`): scored over that layer, never the layer that traces (te_core 4.11's bf16 mv, the
#: toolsmith's 14:37Z line). 4.12.13's code drops 4.12.9's false elementwise listings, so its fix layer retraces each
#: run its layer's trace lists one for (`listed`, coordinator 23:59Z); a trace there reads 4.12.9 as its capture, so
#: the cache's traced.json names the capture code (`mark`)
FIX = "-fix"


def layer_tier(te: Path) -> str | None:
    """The tier whose records `te` alone traces (NEXT_WT-VERSION-TIER, TIER_FROM), None for a capture's own layer."""
    return next((t for t in TIER_FROM if te.name.endswith(f"-{t}")), None)


def layers() -> list[Path]:
    """The captures newer than TE40_WT's, oldest first, a fix layer over the layer it fixes and a tier's layer over the
    layer whose capture it shares: NEXT_WT, NEXT_WT-VERSION (`next_capture`), NEXT_WT-VERSION-fix (`FIX`) and
    NEXT_WT-VERSION-TIER (`tier_layer`)."""
    have = capture_version(trace40.TE40_WT)
    got = [p for p in (NEXT_WT, *NEXT_WT.parent.glob(NEXT_WT.name + "-*")) if p.exists()]
    return sorted((p for p in got if version(capture_version(p)) > version(have)
                   and (layer_tier(p) or p.name.endswith(FIX) or not same_capture(capture_version(p), have))),
                  key=lambda p: version(capture_version(p)))


def fixed(te: Path) -> Path:
    """The layer fix layer `te` is scored over: the newest layer before it that is neither a fix nor a tier's, else
    TE40_WT's."""
    below = [p for p in layers() if not p.name.endswith(FIX) and not layer_tier(p)
             and version(capture_version(p)) < version(capture_version(te))]
    return below[-1] if below else trace40.TE40_WT


def listed(te: Path) -> set[str]:
    """The runs whose trace in `te`'s cache lists a matrix product a step writes with elementwise ops (4.12.7's
    `elementwise_products`), read from each records file's first row, since a run's checkpoints share its trace."""
    out = set()
    for p in (trace40.cache(te) / "records").glob("*.jsonl.gz"):
        with gzip.open(p, "rt") as f:
            line = f.readline()
        if not line:
            continue
        at = ((json.loads(line).get("formalism") or {}).get("products") or {}).get("at") or {}
        if any(v.get("elementwise_products") for v in at.values()):
            out.add(p.name.removesuffix(".jsonl.gz"))
    return out


def mark(te: Path) -> None:
    """Names fix layer `te`'s capture code in each traced run's entry of its traced.json: a trace from before te_core
    4.12.14 names only its frozen capture (CAPTURE_FINAL), not the code that wrote it (coordinator 23:59Z)."""
    f = trace40.cache(te) / "traced.json"
    traced = json.loads(f.read_text()) if f.exists() else {}
    code = {"capture_code": capture_version(te),
            "provenance": (f"te_core {capture_version(te)}'s capture code ({git('rev-parse', '--short', 'HEAD', cwd=te)}"
                           f", tree {trace40.cache(te).name}); the trace reads {CAPTURE_FINAL} as its capture")}
    new = {k: v | code for k, v in traced.items() if not v.get("error") and "capture_code" not in v}
    if new:
        f.write_text(json.dumps(traced | new, indent=1, sort_keys=True, default=str))


def tier_layer(tier: str, top: Path, newest: str) -> Path | None:
    """The layer that traces `tier`'s records over `top`, the layer that traces the rest: the newest
    NEXT_WT-VERSION-TIER whose capture is top's, else one made at the integrator's newest te_core, if that is at or after
    TIER_FROM's and captures as top does; None before then, and where top's own capture replays the tier. Like
    `next_capture`'s, it stays at its first commit (its cache is per te_core tree) and has no files."""
    if version(capture_version(top)) >= version(TIER_FROM[tier]):
        return None
    have = [p for p in layers() if layer_tier(p) == tier and same_capture(capture_version(p), capture_version(top))]
    if have:
        return have[-1]
    want = te_version(newest)
    if version(want) < version(TIER_FROM[tier]) or not same_capture(want, capture_version(top)):
        return None
    new = NEXT_WT.with_name(f"{NEXT_WT.name}-{want}-{tier}")
    git("worktree", "add", "-q", "--no-checkout", "--detach", str(new), newest, cwd=trace40.REPO)
    return new


def next_capture(newest: str) -> Path | None:
    """The capture that traces when the integrator's newest te_core's capture isn't TE40_WT's (`CAPTURES`), else
    None: kept at the first commit seen with that capture, since a later commit of it captures alike and its cache (per
    te_core tree) would trace every run again. A newer capture than the newest layer's gets a worktree of its own
    (NEXT_WT-VERSION), so the older layer's traces stay under it (`layers`) until it replaces TE40_WT's; a layer that
    has traced nothing moves to it instead. Tracing reads only its git objects, so it has no files."""
    have, want = capture_version(trace40.TE40_WT), te_version(newest)
    if same_capture(have, want):
        return None
    if not NEXT_WT.exists():
        git("worktree", "add", "-q", "--no-checkout", "--detach", str(NEXT_WT), newest, cwd=trace40.REPO)
        return NEXT_WT
    top = ([p for p in layers() if not p.name.endswith(FIX) and not layer_tier(p)] or [NEXT_WT])[-1]
    if same_capture(capture_version(top), want):
        return top
    c = trace40.cache(top)
    if not trace40.done(c) and not (json.loads((c / "pending.json").read_text()) if (c / "pending.json").exists()
                                    else {}):
        git("reset", "-q", "--soft", newest, cwd=top)
        return top
    new = NEXT_WT.with_name(f"{NEXT_WT.name}-{want}")
    git("worktree", "add", "-q", "--no-checkout", "--detach", str(new), newest, cwd=trace40.REPO)
    return new


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


def traced_runs(te: Path, *more: Path) -> tuple[int, int]:
    """(the runs in scope `te`'s capture, or a layer of `more`, traced, those of v3's)."""
    ok = set()
    for t in (te, *more):
        f = trace40.cache(t) / "traced.json"
        ok |= {k for k, v in (json.loads(f.read_text()) if f.exists() else {}).items() if not v.get("error")}
    return len(ok & {r["run_id"] for r in trace40.scope()}), len(ok & v3_scope())


def replaces(new: Path, tiered: list[Path] = ()) -> bool:
    """Whether the capture `new` replaces TE40_WT's: it, with its tiers' layers (`tier_layer`), traced at least as many
    of the runs in scope and of v3's, so neither the pilot nor v3 loses a row by the switch (every row traced before a
    capture is refused under its te_core). A v3 scope that grows with each launch never waits on its last run."""
    return all(a >= b for a, b in zip(traced_runs(new, *tiered), traced_runs(trace40.TE40_WT)))


def published_runs(table: Path, p: dict) -> set[str]:
    """The runs of what the console shows from `table` (`p`): each coverage cell's row (console/algorithm-policy.json
    beside it), the frontier's, the reached rows with a measured rho_default that no other such row of their data and
    gain target beats on both p and rho_default, and the partial reached rows' (a lower bound, beside the frontier)."""
    ix = {c: i for i, c in enumerate(p["columns"])}
    shown = table.parent.parent / "console" / "algorithm-policy.json"
    q = json.loads(shown.read_text()) if shown.exists() else {"columns": ["record"], "rows": []}
    runs = {r[q["columns"].index("record")].rsplit("@", 1)[0] for r in q["rows"]}
    by: dict[tuple, list] = collections.defaultdict(list)
    for r in p["rows"]:
        if r[ix["reached"]] != "yes" or not isinstance(r[ix["rho_default"]], (int, float)) or r[ix["p"]] is None:
            continue
        if r[ix["basis_default"]] == "partial":
            runs.add(r[ix["record"]].rsplit("@", 1)[0])
        else:
            by[(r[ix["data"]], r[ix["gain_target"]])].append((r[ix["p"]], r[ix["rho_default"]], r[ix["record"]]))
    for grp in by.values():
        runs |= {rec.rsplit("@", 1)[0] for a, b, rec in grp
                 if not any(c <= a and d <= b and (c, d) != (a, b) for c, d, _ in grp)}
    return runs


def retrace_order(v3: set[str]) -> dict[str, int]:
    """The runs a retrace takes first (coordinator 12:20Z), from the newest algorithm-policy table a refresh PUT under
    OUT_ROOT (else the newest it wrote): -1 a published row's (`published_runs`, coordinator 20:56Z); 0 a run that
    reached a gain, v3's or the pilot's, by its row or, a v3 run with no row (refused or not yet traced, such as
    sparse's and quant's first 1.1x points, coordinator 15:08Z), by a checkpoint's loss in the ledger at or under a
    reached row's at the least gain target from its start; 1 each v3 family's best not-reached run (by gain;
    method_key where the family is unnamed). The rest of v3 follows, then the pilot."""
    def put(t: Path) -> bool:
        # vm-put.log: the VM's PUT of a refresh on the node (refresh-node.sh writes it from TE_VM_PUT)
        logs = (t.parents[1] / "run.log", t.parents[1] / "vm-put.log")
        return any(x.exists() and "/training/algorithm-policy: 200" in x.read_text() for x in logs)
    tables = sorted(OUT_ROOT.glob("2026-*/algorithm-policy/tables/algorithm-policy.json"), reverse=True)
    for t in sorted(tables, key=lambda t: not put(t)):
        try:
            p = json.loads(t.read_text())
        except ValueError:
            continue
        ix = {c: i for i, c in enumerate(p["columns"])}
        rows = [r for r in p["rows"] if r[ix["record"]].rsplit("@", 1)[0] in v3]
        rank = {r[ix["record"]].rsplit("@", 1)[0]: 0 for r in p["rows"] if r[ix["reached"]] == "yes"}
        rank |= {run: -1 for run in published_runs(t, p)}
        start = {r["run_id"]: r["start"] for r in trace40.scope()}
        least = min((r[ix["gain_target"]] for r in rows), default=None)
        bar: dict[str, float] = {}
        for r in rows:
            if r[ix["reached"]] == "yes" and r[ix["gain_target"]] == least:
                s = start.get(r[ix["record"]].rsplit("@", 1)[0])
                bar[s] = max(bar.get(s, -math.inf), r[ix["end_loss"]])
        for line in (trace40.LEDGER / "index.jsonl").open():
            x = json.loads(line)
            if x.get("run_id") in v3 and (x.get("val_bpb") or math.inf) <= bar.get(start.get(x["run_id"]), -math.inf):
                rank.setdefault(x["run_id"], 0)
        best: dict[str, tuple[float, str]] = {}
        for r in rows:
            run, fam, gain = r[ix["record"]].rsplit("@", 1)[0], r[ix["family"]] or r[ix["method_key"]], r[ix["gain"]]
            if run not in rank and gain is not None and gain > best.get(fam, (-math.inf, ""))[0]:
                best[fam] = (gain, run)
        return rank | {run: 1 for _, run in best.values() if run not in rank}
    return {}


def run_tiers() -> dict[str, str | None]:
    """Each run's budget tier, as its ledger rows carry it (`budget.tier`: run.json's evaluation.budget.tier, which
    te_core's capture reads, for all 574 runs in scope at 18:00Z)."""
    out = {}
    for line in (trace40.LEDGER / "index.jsonl").open():
        x = json.loads(line)
        if x.get("run_id") and x["run_id"] not in out:
            out[x["run_id"]] = (x.get("budget") or {}).get("tier")
    return out


def to_trace(te: Path, busy: set[str]) -> list[tuple[int, str]]:
    """The runs in scope with no trace in `te`'s cache and no node run tracing them, each with its rank: v3's first
    (`retrace_order`), then the pilot's under CAPTURE_FINAL's capture or when TE_TRACE_PILOT is set. A tier's layer
    takes only that tier's records, and a layer before TIER_FROM's none of them; a fix layer only the runs its layer's
    traces list elementwise products for (`listed`)."""
    v3, done = v3_scope(), trace40.done(trace40.cache(te)) | busy
    pilot = os.environ.get("TE_TRACE_PILOT") == "1" or same_capture(capture_version(te), CAPTURE_FINAL)
    tier, tiers, at = layer_tier(te), run_tiers(), version(capture_version(te))
    fix = listed(fixed(te)) if te.name.endswith(FIX) else None

    def here(run: str) -> bool:
        if fix is not None:
            return run in fix
        t = tiers.get(run)
        return t == tier if tier else not (t in TIER_FROM and at < version(TIER_FROM[t]))
    runs = [r["run_id"] for r in trace40.scope() if r["run_id"] not in done and (r["run_id"] in v3 or pilot)
            and here(r["run_id"])]
    first = retrace_order(v3)
    return sorted(((first[r] if first.get(r, 1) <= 0 else 2 * (r not in v3) + first.get(r, 2), r) for r in runs),
                  key=lambda x: x[0])


def reached_left(te: Path) -> list[str]:
    """The published and reached runs (`retrace_order`'s ranks 0 and below) in scope that `te`'s cache hasn't traced
    or failed: while any is left, a refresh over CAPTURE_FINAL's capture PUTs nothing, so the console's products terms
    move to the new capture together and its frontier doesn't flip empty and back (coordinator 20:56Z)."""
    done = trace40.done(trace40.cache(te))
    scope = {r["run_id"] for r in trace40.scope()}
    return sorted(r for r, k in retrace_order(v3_scope()).items() if k <= 0 and r in scope and r not in done)


def ledger_args(out: Path, root: Path | None = None) -> list[str]:
    ledger = str(root or os.environ.get("TE_LEDGER", str(Path.home() / ".research" / "te-ledger")))
    return ["--out", str(out), "--root", ledger, "--scorer", str(S.STATE), *(x for p in PINS for x in ("--pin", p))]


#: the states `research fetch` observes a node run to have ended in
ENDED = ("done", "failed", "cancelled", "refused", "dead", "timed_out", "exited")


def attempt(run: str) -> dict:
    """A node run's attempt record in the evidence store (attempts/RUN.json), {} before custody writes it."""
    with tempfile.TemporaryDirectory() as d:
        try:
            S.store_remote().get(f"attempts/{run}.json", Path(d) / "a.json")
            return json.loads((Path(d) / "a.json").read_text())
        except Exception:  # noqa: BLE001 - an attempt not yet in the store (or not readable) is a run still going
            return {}


def node_state(run: str) -> str:
    """The node run's state as `research fetch RUN` observes it (`... observed T: STATE rc=...`): one of ENDED, else
    running. On the node: a shard traced here by its process, one of an earlier refresh ended (its process ended with
    that run), and another node run by its attempt in the store."""
    if HERE_DIR:
        if run in LOCAL:
            return "running" if LOCAL[run][0].poll() is None else "exited"
        if run.startswith("here-"):
            return "exited"
        state = (attempt(run).get("status") or {}).get("state")
        return state if state in ENDED else "running"
    r = subprocess.run(["research", "fetch", run], capture_output=True, text=True, check=False)
    m = re.search(r" observed \S+: (\w+) rc=", r.stdout + r.stderr)
    return m.group(1) if m and m.group(1) in ENDED else "running"


def take_src(run: str) -> list[str]:
    """`trace40.py take`'s --src on the node: a shard's out/ here, else the out/ of the node run's run record, fetched
    from the store (a missing one fails the take, so its runs are traced again)."""
    if not HERE_DIR:
        return []
    if run.startswith("here-"):
        return ["--src", str(HERE_DIR / run / "out")]
    art = (attempt(run).get("outputs") or {}).get("run_record")
    r = subprocess.run(["research", "data", "fetch", art], capture_output=True, text=True, check=False) if art else None
    got = Path(r.stdout.strip().splitlines()[-1]) if r and not r.returncode and r.stdout.strip() else HERE_DIR / run
    return ["--src", str(got / "out" if (got / "out").is_dir() else got)]


def trace_here(pack: Path, workers: int, gb: int) -> str:
    """Shard `pack` traced here by job40.sh in its own session (so `stop_here` ends its captures with it), into
    HERE_DIR/NAME/out; NAME, its node run's name in pending.json."""
    run = f"here-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{len(LOCAL)}"
    d = HERE_DIR / run
    (d / "tmp").mkdir(parents=True)
    shutil.move(pack, d / "pack.tar")
    with (d / "trace.log").open("w") as log:
        p = subprocess.Popen(["bash", str(HERE / "job40.sh")], stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=os.environ | {
                                 "PACK": str(d / "pack.tar"), "OUT": str(d / "out"), "WORKERS": str(workers),
                                 "TMPDIR": str(d / "tmp")})
    LOCAL[run] = (p, gb)
    return run


def stop_here(until: float) -> dict:
    """Waits for the shards tracing here until `until` (epoch seconds), then ends each one still tracing with its
    captures; the runs a stopped shard finished are in its out/, taken at the next refresh, and the rest traced again."""
    stopped = []
    for run, (p, _) in LOCAL.items():
        try:
            p.wait(timeout=max(0.0, until - time.time()))
        except subprocess.TimeoutExpired:
            for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
                try:
                    os.killpg(p.pid, sig)
                    p.wait(timeout=grace)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    continue
            stopped.append(run)
    return {"here": sorted(LOCAL), "stopped": stopped}


def trace_resources(runs: int) -> tuple[int, int]:
    """Workers (CPUs) and GB to ask TRACE_NODE for a shard of `runs` runs: 24 workers peaked at 70 GB PSS (88 GB RSS) on
    r20261103-024017-a670's m100-heavy bulk, so 4 GB a worker; a shard of 6 asks 36 GB, under a research pool slot's
    40 (vy-nebius-cpu-1's), which `--on auto` needs to place it."""
    workers = max(1, min(24, runs))
    return workers, min(200, 12 + 4 * workers)


#: the captures whose node runs size a shard (`run_peaks`): on 4.12.7's a run's capture alone took 28 to 404 GB, past
#: its shard's 36, and r20261103-083440-8d5e's grew to 404 GB until the kernel killed it at 20:45Z on cpu-1
#: (r20261103-193201-8666; coordinator 20:56Z). A request is a reservation, not a bound
PEAKS_SINCE = "4.12.7"
#: each taken node run's {run: GB} (`node_peaks`)
PEAKS = ON_POOL.with_name("peaks.json")
#: a capture that peaks at this many GB or more is a heavy run's: a shard of its own, asking for it (`shard_resources`)
HEAVY_GB = 8
#: the share of a machine's available memory the shards placed on it at once may ask (`plan`)
MEM_SHARE = 0.8
MEMINFO = b"print(int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:'))) >> 20)\n"


def node_peaks(node: str) -> dict[str, float] | None:
    """The peak RSS (GB) of each heavy run's capture in node run `node`, from its fetched resources.jsonl, None when
    that isn't here: a capture is a subprocess that lives as long as its run traces, so the longest-lived subprocesses
    at HEAVY_GB or more are the slowest runs' of the node's pack (one unfinished or killed the slowest, a timed out
    one TRACE_TIMEOUT, another error none)."""
    d = Path.home() / ".research" / "runs" / node
    packs = sorted((d / "inputs").glob("trace40-pack-*.tar"))
    if not (d / "resources.jsonl").exists() or not packs:
        return None
    runs = [r["run_id"] for r in json.loads(subprocess.run(["tar", "-xOf", str(packs[0]), "./runs.json"],
                                                           capture_output=True, check=True).stdout)]
    traced = json.loads((d / "out" / "traced.json").read_text()) if (d / "out" / "traced.json").exists() else {}

    def took(run: str) -> float:
        err = (traced.get(run) or {}).get("error") or ""
        if run not in traced or "killed" in err:
            return math.inf
        return float(trace40.TRACE_TIMEOUT) if "timed out" in err else traced[run].get("wall") or 0.0
    life: dict[int, list[float]] = {}
    for line in (d / "resources.jsonl").open():
        r = json.loads(line)
        if r.get("kind") != "process":
            continue
        for p in r.get("processes") or []:
            x = life.setdefault(p["pid"], [r["t_mono"], r["t_mono"], 0.0])
            x[1], x[2] = r["t_mono"], max(x[2], (p.get("rss_bytes") or 0) / 1e9)
    heavy = sorted((x for x in life.values() if x[2] >= HEAVY_GB), key=lambda x: x[0] - x[1])
    return {run: round(x[2], 1) for run, x in zip(sorted(runs, key=lambda r: -took(r)), heavy)}


def run_peaks() -> dict[str, float]:
    """Each run's greatest capture peak (`node_peaks`) over the node runs of PEAKS_SINCE's captures and later, those
    taken (cached in PEAKS) and those pending, fetched here."""
    seen = json.loads(PEAKS.read_text()) if PEAKS.exists() else {}
    taken, pending = set(), set()
    for te in layers():
        if version(capture_version(te)) >= version(PEAKS_SINCE):
            c = trace40.cache(te)
            if (c / "traced.json").exists():
                taken |= {v.get("by") for v in json.loads((c / "traced.json").read_text()).values()} - {None}
            if (c / "pending.json").exists():
                pending |= set(json.loads((c / "pending.json").read_text()))
    for n in sorted(taken - set(seen)):
        if (got := node_peaks(n)) is not None:
            seen[n] = got
    PEAKS.write_text(json.dumps(seen, indent=1, sort_keys=True))
    out: dict[str, float] = {}
    for v in (*seen.values(), *(node_peaks(n) or {} for n in sorted(pending - taken))):
        for run, gb in v.items():
            out[run] = max(out.get(run, 0.0), gb)
    return out


def shard_resources(runs: list[str], peaks: dict[str, float]) -> tuple[int, int]:
    """trace_resources' for a shard of `runs`, and for a heavy run (`run_peaks`) half again its capture's peak past the
    4 GB a worker it gives."""
    workers, gb = trace_resources(len(runs))
    return workers, gb + sum(max(0, math.ceil(1.5 * peaks[r]) - 4) for r in runs if r in peaks)


def on_pool() -> dict[str, str]:
    """ON_POOL's node runs still pending in some layer's cache, each with its machine."""
    x = json.loads(ON_POOL.read_text()) if ON_POOL.exists() else {}
    x = x if isinstance(x, dict) else {r: "vy-nebius-cpu-9" for r in x}
    flying = {r for f in ON_POOL.parent.glob("*/pending.json") for r in json.loads(f.read_text())}
    return {r: m for r, m in x.items() if r in flying}


def pool_room() -> tuple[list[tuple[str, str, int | None]], dict[str, str], dict[str, float]]:
    """A (machine, pool, the slot's GB, None where the pool gives a run what it asks) per shard that may launch now, in
    TRACE_POOLS' order: each pool's free slots that fit a shard, past its waiters (auto.py's read of `slot.py --status
    --json` over the ssh `research run --on` uses), at most its cap less the shards in flight there (`on_pool`); a pool
    that can't be read takes none. Also what each pool said, and the GB shards may ask on each machine (MEM_SHARE of
    its available memory). On the node, one slot ("here") per TRACE_SHARD of the cores past HERE_CORES and the
    reservation's GB past HERE_GB, less the shards tracing here."""
    if HERE_DIR:
        cores, flying = len(os.sched_getaffinity(0)), [gb for p, gb in LOCAL.values() if p.poll() is None]
        n = max(0, (cores - HERE_CORES) // TRACE_SHARD - len(flying))
        gb = float(os.environ.get("RESEARCH_RUN_MEM_GB") or 0) - HERE_GB - sum(flying)
        return ([("here", "here", None)] * n, {"here": f"{cores} cores, {len(flying)} shards tracing, {n} more; "
                                                        f"{gb:.0f} GB to ask"}, {"here": gb})
    from research import auto
    if not Path(auto.__file__).resolve().is_relative_to(HERE.parents[2]):
        return [], {m: f"not read: research.auto is {auto.__file__}, not {HERE.parents[2]}'s"
                    for m, _, _ in TRACE_POOLS}, {}
    workers, gb = trace_resources(TRACE_SHARD)
    flying = collections.Counter(on_pool().values())

    def one(item: tuple[str, str, int]) -> tuple[int, str, int | None, float]:
        m, p, _ = item
        try:
            r = auto._ssh(auto._reach(m), auto.payload(HERE.parents[2], workers, None, [p]))
            doc = json.loads(r.stdout).get(p)
            _, free, waiting, why = auto._pool(doc, workers, gb)
            avail = int(auto._ssh(auto._reach(m), MEMINFO).stdout)
        except Exception as e:  # noqa: BLE001 - a pool that can't be read takes no shard, whatever stopped the read
            return 0, f"not read: {str(e)[:120] or type(e).__name__}", None, 0.0
        mems = [s.get("mem") for s in doc.get("slots") or [] if s.get("holder") is None and not s.get("blocked")]
        return max(0, free - waiting), why, None if None in mems else max(mems, default=gb), MEM_SHARE * avail
    with ThreadPoolExecutor(max_workers=len(TRACE_POOLS)) as ex:
        seen = list(ex.map(one, TRACE_POOLS))
    slots, said, room = [], {}, {}
    for (m, p, cap), (n, why, slot_gb, avail) in zip(TRACE_POOLS, seen):
        n = min(n, max(0, cap - flying[m]))
        slots += [(m, p, slot_gb)] * n
        room[m] = avail
        said[m] = f"{why}; {avail:.0f} GB to ask; {flying[m]} of ours in flight, cap {cap}, {n} launched"
    return slots, said, room


def plan(tes: list[Path], slots: list[tuple[str, str, int | None]], room: dict[str, float]
         ) -> tuple[dict[Path, list[tuple[list[str], str, str]]], dict[str, int]]:
    """Each launching layer's shards, each with the slot it takes: every layer's runs to trace (`to_trace`) in its own
    order, a heavy run (`run_peaks`) in a shard of its own and the rest in shards of TRACE_SHARD, the shards taken in
    retrace order across layers (by their runs' ranks, then the layer's place in `tes`), each in the first of `slots`
    whose slot gives what it asks (`shard_resources`) on a machine with that much `room` left. Also each heavy run left
    waiting for memory, with the GB it asks."""
    peaks, shards = run_peaks(), []
    for i, te in enumerate(tes):
        pf = trace40.cache(te) / "pending.json"
        todo = to_trace(te, {r for runs in (json.loads(pf.read_text()) if pf.exists() else {}).values() for r in runs})
        light = [x for x in todo if x[1] not in peaks]
        shards += [((k,), i, -1, te, [r]) for k, r in todo if r in peaks]
        shards += [(tuple(k for k, _ in light[j:j + TRACE_SHARD]), i, j, te, [r for _, r in light[j:j + TRACE_SHARD]])
                   for j in range(0, len(light), TRACE_SHARD)]
    out, waiting, left, room = {te: [] for te in tes}, {}, list(slots), dict(room)
    for *_, te, runs in sorted(shards, key=lambda s: s[:3]):
        gb = shard_resources(runs, peaks)[1]
        at = next((s for s in left if (s[2] is None or s[2] >= gb) and room.get(s[0], 0) >= gb), None)
        if at is None:
            if left and len(runs) == 1 and runs[0] in peaks:
                waiting[runs[0]] = gb
            continue
        left.remove(at)
        room[at[0]] -= gb
        out[te].append((runs, at[0], at[1]))
    return out, waiting


def measured_ledger(out: Path, shards: list[tuple[list[str], str, str]] = (), te: Path = trace40.TE40_WT,
                    build: bool = True, over: list[Path] = (), wait: bool = True) -> dict:
    """trace40.py's measured ledger: the scorer's ledger indexed (rescore reads the measured copy, so nothing else
    rewrites the index), each of `shards` (runs, machine, pool; `plan`) packed from `te`'s cache and launched in its
    pool, every node run (this one and earlier refreshes') taken as it ends, waited for up to TRACE_WAIT_S (`wait`; one
    still tracing is taken at a later refresh), and the copy built (`build`), with the runs each capture of `over`
    (oldest first) traced taken from there (trace40.ledger)."""
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
            t = subprocess.run([*cw, "take", run, "--te", str(te), *take_src(run)], capture_output=True, text=True,
                               check=False)
            if t.returncode == 0:
                took.append(run)
            else:
                log.setdefault("not_taken", {})[run] = f"{state}: {(t.stderr.strip().splitlines() or [''])[-1][:200]}"
            del pending[run]
            if HERE_DIR and run.startswith("here-"):
                shutil.rmtree(HERE_DIR / run, ignore_errors=True)

    # the ended node runs taken first, so the runs they failed (under an older CODE) are traced again in this one
    settle()
    pf.write_text(json.dumps(pending, indent=1))
    packed, peaks = {"runs": sum(len(s[0]) for s in shards)}, run_peaks() if shards else {}
    for i, (shard, machine, pool) in enumerate(shards):
        pack = out / f"trace40-pack-{i}.tar"
        p = subprocess.run([*cw, "pack", "--out", str(pack), "--te", str(te), "--only", ",".join(shard)],
                           capture_output=True, text=True, check=False)
        if p.returncode:
            log.setdefault("pack_failed", []).append((p.stderr.strip().splitlines() or [f"rc {p.returncode}"])[-1])
            continue
        runs = [r["run_id"] for r in json.loads(subprocess.run(["tar", "-xOf", str(pack), "./runs.json"],
                                                               capture_output=True, check=True).stdout)]
        workers, gb = shard_resources(runs, peaks)
        if HERE_DIR:
            run, rc = trace_here(pack, workers, gb), 0
            said = f"tracing here: {run}, {len(runs)} runs, {workers} workers, {gb} GB"
        else:
            # the pool slot's cores are the run's, so no --cpus
            r = subprocess.run(["research", "run", "--on", machine, "--pool", pool, "--as", TRACE_LANE,
                                "--project", "verity", "--source", str(HERE.parents[2]),
                                "--cwd", "source", "--send", str(pack), "--mem-gb", str(gb),
                                "--env", f"WORKERS={workers}", "--timeout", "150m", "--custody-ttl", TRACE_CUSTODY_TTL,
                                "--", "bash", "benchmarks/covert_witness/scorer/job40.sh"],
                               capture_output=True, text=True, check=False,
                               env=os.environ | {"AGENT_GUARD_LOCAL_OK": "1"})
            said, rc = r.stdout + r.stderr, r.returncode
            run = next((w for w in said.split() if w.startswith("r20") and w.count("-") == 2), None)
        log.setdefault("launch", []).append([machine, *said.strip().splitlines()[-1:]])
        # a refused launch traces nothing, so its runs stay out of pending and the next refresh packs them again
        if run and not rc and " refused" not in said:
            pending[run] = runs
            # written at once, so the next launch (this layer's or another's) counts it in flight
            pf.write_text(json.dumps(pending, indent=1))
            ON_POOL.write_text(json.dumps(on_pool() | {run: machine}, indent=1))
        else:
            log.setdefault("launch_refused", []).append(run)
        pack.unlink(missing_ok=True)
    while wait and pending and time.monotonic() < deadline:
        time.sleep(60)
        settle()
    pf.write_text(json.dumps(pending, indent=1))
    built = json.loads(subprocess.run([*cw, "ledger", "--te", str(te), *(x for o in over for x in ("--over", str(o)))],
                                      capture_output=True, text=True, check=True).stdout) if build else None
    traced = json.loads((cache / "traced.json").read_text()) if (cache / "traced.json").exists() else {}
    waiting = {k: sum(r not in traced or bool(traced[r].get("error")) for r in v) for k, v in pending.items()}
    return {"te_tree": cache.name, "took": took, "tracing": waiting,
            "packed": packed["runs"], "traced_runs": sum(not v.get("error") for v in traced.values()),
            "untraced_runs": sum(bool(v.get("error")) for v in traced.values()), "ledger": built, **log}


#: `python -c LEAN MODULE ARGS` runs te_core's MODULE with each record's copy of its run's trace swapped, as the ledger
#: reads it, for one object per distinct trace: at 21:56Z 5598 records carried 1.16 GB of traces, 230 distinct (46 MB),
#: and the copies took the 21:31Z re-score past this VM's memory (killed at 9.4 GB). A module that changed a shared
#: trace would change every record that shares it, so a changed one stops the module before any PUT and at its exit
#: (rc 70)
LEAN_SETUP = """
import atexit, hashlib, json, os, runpy, sys
from te_core import console, ledger
shared = {}
def key(tr):
    return hashlib.sha256(json.dumps(tr).encode()).hexdigest()
class Json:
    def __getattr__(self, name):
        return getattr(json, name)
    def loads(self, s, *a, **k):
        x = json.loads(s, *a, **k)
        f = x.get("formalism") if isinstance(x, dict) else None
        tr = f.get("trace") if isinstance(f, dict) else None
        if isinstance(tr, dict):
            f["trace"] = shared.setdefault(key(tr), tr)
        return x
def check():
    changed = sum(key(tr) != k for k, tr in shared.items())
    if changed:
        print(f"v1cycle LEAN: {changed} shared traces changed in place; this run's output doesn't hold",
              file=sys.stderr, flush=True)
        os._exit(70)
put = console.put
def checked_put(*a, **k):
    check()
    return put(*a, **k)
ledger.json, console.put = Json(), checked_put
atexit.register(check)
"""
LEAN = LEAN_SETUP + 'runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)\n'
#: LEAN for te_core.algorithm_policy with its rows labelled before it writes, stores or PUTs them (v1labels.run):
#: argv[1] this directory, argv[2] v1labels.frozen_runs' JSON file; v1labels only, since scorer.py puts its own te_core
#: first on sys.path, and this directory last, so no module of it shadows one te_core imports
POLICY = LEAN_SETUP + """
from pathlib import Path
sys.path.append(sys.argv.pop(1))
import v1labels
from te_core import algorithm_policy
if os.environ.get("TE_SERVED"):
    # a refresh on the node (HERE_DIR) holds no console token and no store write key: the panels the VM read as
    # served before launching it, the tables cited in the run's out/, and the PUT of the console/ bodies left to the VM
    console.served = lambda producer="training": json.load(open(os.environ["TE_SERVED"]))
    algorithm_policy.store = lambda path, meta: os.environ["TE_TABLES_CITE"]
    put = lambda p: os.environ["TE_VM_PUT"]
sys.exit(v1labels.run(algorithm_policy, Path(sys.argv.pop(1))))
"""


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
             [*runs, "--prefix", "search-runs-"]] if final else [[sys.executable, "-c", LEAN, "te_core.rescore", *args]])
    logs = []
    for c in cmds:
        r = subprocess.run(c, cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True,
                           check=False)
        logs.append(r.stdout + r.stderr)
        (out / "rescore.log").write_text("\n".join(logs))
        if r.returncode:
            what = c[3] if c[1] == "-c" else " ".join(c[1:3])
            raise SystemExit(f"v1cycle: {what} failed (rc {r.returncode}):\n{r.stderr[-2000:]}")
    return logs[0].strip().splitlines()[-1]


def branches(out: Path, root: Path | None = None) -> str:
    """te_core.panels' frontier-branches into `out` after the re-score (its gate cache reused), on the ledger at
    `root`, never PUT (FB_RETIRED); why there is no PUT, or why there is no panel."""
    r = subprocess.run([sys.executable, "-c", LEAN, "te_core.panels", FB, *ledger_args(out, root), *ladder_fit()],
                       cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True, check=False)
    (out / "panels.log").write_text(r.stdout + r.stderr)
    why = (r.stderr.strip().splitlines() or [f"rc {r.returncode}"])[-1][:300]
    if not (out / "console" / f"{FB}.json").exists():
        return f"not built: {why}"
    return f"not PUT: {why}" if r.returncode else f"not PUT ({FB_RETIRED})"


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
#: the scoring cap's ladder (the integrator's CAP.json in policy.ladder_fit's format), which every panel build reads as
#: --ladder-fit (coordinator 20:11Z)
LADDER_FIT = os.environ.get("TE_LADDER_FIT", "art:031cf1c6aa64e69fe5affa8f8ba20fa90df25f780d89e7b85951997af9d438c3")


def ladder_fit() -> list[str]:
    """`--ladder-fit` and LADDER_FIT's CAP.json, fetched from the evidence store; none when it can't be fetched."""
    r = subprocess.run(["research", "data", "fetch", LADDER_FIT], capture_output=True, text=True, check=False)
    cap = Path(r.stdout.strip().splitlines()[-1]) / "CAP.json" if not r.returncode and r.stdout.strip() else None
    return ["--ladder-fit", str(cap)] if cap and cap.exists() else []


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


def policy_panels(out: Path, commit: str, put: bool, root: Path | None, gates: Path | None = None) -> dict:
    """Once the re-score's te_core holds the builder's v2 (AP_V2 in te_core.algorithm_policy): its run on the same
    ledger, with `--publish` under `put`, which builds and PUTs training/algorithm-policy and training/frontier-fit
    both; {panel: its PUT status or why there is none} and `coverage`, the coverage (gain size by decade of rho) as
    the board line it prints, else its table's rows."""
    src = lambda p: subprocess.run(["git", "-C", str(RESCORE_WT), "show", f"{commit}:benchmarks/covert_witness/te_core/{p}"],
                                   capture_output=True, text=True, check=False).stdout
    if AP_V2 not in src("algorithm_policy.py"):
        return {"algorithm-policy": "not run (the builder's v2 isn't on this te_core yet)"}
    d = out / "algorithm-policy"
    d.mkdir(parents=True, exist_ok=True)
    if gates and gates.exists() and not (d / "analysis-gates.json").exists():
        (d / "analysis-gates.json").write_bytes(gates.read_bytes())
    frozen = d / "frozen-embedding.json"
    frozen.write_text(json.dumps(v1labels.frozen_runs(trace40.STATE / "trace40")) + "\n")
    ladder, cap = ladder_points(out / "ladder"), ladder_fit()
    here = {"TE_VM_PUT": VM_PUT, "TE_TABLES_CITE": f"{here_cite('tables')}/algorithm-policy/tables"} if HERE_DIR else {}
    r = subprocess.run([sys.executable, "-c", POLICY, str(HERE), str(frozen), *ledger_args(d, root),
                        *(x for p in ladder for x in ("--ladder-points", str(p))), *cap,
                        *(["--publish"] if put else [])],
                       cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True, check=False,
                       env=os.environ | here)
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
    res["ladder_fit"] = LADDER_FIT if cap else None
    fit = d / "tables" / f"{FIT}.json"
    used = re.search(r"(\d+) ladder points(?: \((art:\w+)\))?", json.loads(fit.read_text()).get("source") or ""
                     ) if fit.exists() else None
    res["ladder_used"] = {"points": int(used[1]), "art": used[2]} if used else None
    labels = d / "labels.json"
    res["labels"] = json.loads(labels.read_text()) if labels.exists() else None
    res["priced"] = v1labels.priced(d)
    console = "import json; from te_core import algorithm_policy as a; print(json.dumps(list(a.CONSOLE)))"
    c = subprocess.run([sys.executable, "-c", console], cwd=RESCORE_WT / "benchmarks" / "covert_witness",
                       capture_output=True, text=True, check=False)
    res["console"] = json.loads(c.stdout) if not c.returncode else None
    table = d / "tables" / "algorithm-policy.json"
    art = re.search(r"\((art:\w+|run:[\w./-]+)\)", board or "")
    c = subprocess.run([sys.executable, "-c", CONSOLE_COVERAGE, str(table), now(), art[1] if art else ""],
                       cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True,
                       check=False) if table.exists() else None
    res["coverage_console"] = (json.loads(c.stdout.strip().splitlines()[-1]) if c and not c.returncode else
                               {"error": (c.stderr.strip().splitlines() or ["rc"])[-1][:200]} if c else None)
    return res


#: what training/algorithm-policy's console copy carries once frontier-branches is no longer PUT, only stored
#: (coordinator 05:22Z and 07:48Z)
FB_SUCCESSOR = ("gain", "gain_target", "start_effective_gates")
#: the coverage at the console default (coordinator 11:46Z): te_core.algorithm_policy's own on its uncut table, each
#: row's rho_console in rho_default's place ("inf" infinite; a null one, unmeasured, and a partial one, only a lower
#: bound, in no cell or frontier: coordinator 12:20Z); argv: the table, the stamp and the tables art
CONSOLE_COVERAGE = """
import json, math, sys
from te_core import algorithm_policy as a
p = json.load(open(sys.argv[1]))
c = p["columns"]
i, j, d, kc, rc = (c.index(x) for x in ("rho_default", "rho_console", "data", "kinds", "record"))
cell = [json.loads(r[kc]) if r[kc] else {} for r in p["rows"]]
null = sum(r[j] is None for r in p["rows"])
partial = sum(bool(x.get("partial")) for x in cell)
fresh = len({r[rc] for r, x in zip(p["rows"], cell) if x.get("bound", {}).get("fresh_products") is not None})
fits = {r[d]: True for r in p["rows"]}
p["rows"] = [r[:i] + [math.inf if r[j] == "inf" else r[j]] + r[i + 1:] for r, x in zip(p["rows"], cell)
             if r[j] is not None and not x.get("partial")]
cov = a.coverage(p)
line = a.board_line(cov, fits, sys.argv[2], sys.argv[3] or None)
print(json.dumps({"line": line.replace("decade of rho_default", "decade of rho_console"), "unmeasured": null,
                  "partial": partial, "fresh_records": fresh, "empty": sum(x["state"] == "empty" for x in cov),
                  "cells": len(cov)}))
"""
FB_RETIRED = ("training/algorithm-policy carries " + ", ".join(FB_SUCCESSOR)
              + " on its rows; frontier-branches is in the evidence store only")


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


def link(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    os.link(src, dst)


def here_cite(name: str) -> str:
    """How a refresh on the node cites `name` under its run's out/, which custody puts in the evidence store with the
    run's record."""
    run = os.environ.get("RESEARCH_RUN_ID") or Path(os.environ["RESEARCH_RUN_DIR"]).name
    return f"run:{run}/out/{name}"


def here_out(tree: Path, name: str) -> str:
    """`tree` moved to `name` under the node run's out/ (a node holds no store write key: custody stores it); its
    citation (`here_cite`)."""
    dst = Path(os.environ["RESEARCH_RUN_DIR"]) / "out" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tree), str(dst))
    return here_cite(name)


def put_data(tree: Path, kind: str, meta: dict) -> str:
    # the store's local index is SQLite, which another writer on this VM holds for a while ("database is locked",
    # the 17:00Z refresh's publish); the same tree puts to the same id, so a retry is safe
    for wait in (0, 15, 60, 180):
        time.sleep(wait)
        r = subprocess.run(["research", "data", "put", "--kind", kind, "--tree", str(tree), "--preserve", "--json",
                            "--meta", json.dumps(meta)], capture_output=True, text=True, check=False)
        if not r.returncode or "database is locked" not in r.stderr:
            break
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
    ap.add_argument("--out", type=Path, help="default OUT_ROOT/<stamp>")
    ap.add_argument("--gates", type=Path, default=OUT_ROOT / "trial" / "gates.json",
                    help="the analysis's gate factors, seeded into the re-score's cache")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-put", action="store_true", help="build algorithm-policy and frontier-fit but don't PUT them")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--no-trace", action="store_true", help="launch no node trace (finished ones are still taken)")
    ap.add_argument("--final", type=Path, metavar="FINAL_DIR",
                    help="the final table: te_core.rescore with FINAL_DIR's values (scorer.py final table; v1final.py); "
                         "posts FINAL-TABLE, not DIGEST")
    args = ap.parse_args()
    if HERE_DIR:
        # a shard traces in its own session, so nothing ends it with the refresh's run: a refresh that exits or is
        # stopped (its run's timeout) ends its shards first
        atexit.register(stop_here, 0.0)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    stamp, final = now(), args.final
    out = args.out or OUT_ROOT / f"{'final-' if final else ''}{stamp.replace(':', '')}"
    commit, newest, chosen = rescore_head()
    os.environ["TE_PIN_REV"] = commit
    PINS[:] = trace40.pins(commit)
    nxt, retrace, held = (None, None, {}) if final else (next_capture(newest), None, {})
    top = nxt or trace40.TE40_WT
    tiered = [] if final else [p for p in (tier_layer(t, top, newest) for t in TIER_FROM) if p]
    shards, pools, tiers, mem_wait = {}, {}, {}, {}
    if not final and not args.no_trace:
        # the launching layers' ended node runs taken first, so the runs they failed (under an older CODE) are
        # planned again in this refresh
        for te in (*tiered, top):
            measured_ledger(out / f"settle-{te.name}", (), te, build=False, wait=False)
        slots, pools, room = pool_room()
        shards, mem_wait = plan([*tiered, top], slots, room)
    for te in tiered:
        tiers[layer_tier(te)] = measured_ledger(out / f"tier-{te.name}", shards.get(te, []), te, build=False,
                                                wait=False) | {"te_core": capture_version(te),
                                                               "commit": git("rev-parse", "HEAD", cwd=te)}
    if nxt:
        # an older layer launches nothing, and its node runs are taken as they end
        held = {capture_version(h): measured_ledger(out / f"held-{capture_version(h)}", (), h, build=False, wait=False)
                for h in layers() if h != nxt and h not in tiered}
        # a row traced under a capture before READS_SINCE's is refused under a later te_core, so over such a capture
        # the refresh scores on the last re-score until the new capture's traces replace it
        retrace = measured_ledger(out / "next-capture", shards.get(nxt, []), nxt, build=False, wait=not args.no_trace)
        retrace |= dict(zip(("of", "covered", "traced"), covered(nxt)), te_core=te_version(newest),
                        commit=git("rev-parse", "HEAD", cwd=nxt))
        if replaces(nxt, tiered) or not LAST.exists():
            git("checkout", "-q", "--detach", retrace["commit"], cwd=trace40.TE40_WT)
            nxt, retrace, held = None, None, {}
        elif version(capture_version(trace40.TE40_WT)) < version(READS_SINCE):
            commit = json.loads(LAST.read_text())["commit"]
            git("checkout", "-q", "--detach", commit)
            chosen = (f"the last re-score on te_core {capture_version(trace40.TE40_WT)}'s capture, until te_core "
                      f"{te_version(newest)}'s traces replace it")
            os.environ["TE_PIN_REV"] = commit
            PINS[:] = trace40.pins(commit)
    # a fix layer plans after the wait, so the runs its layer traced during it are retraced too; its node runs are
    # taken at the next refresh
    fixes = [] if final else [p for p in layers() if p.name.endswith(FIX)]
    got = {}
    if fixes and not args.no_trace:
        slots, _, room = pool_room()
        got, waits = plan(fixes, slots, room)
        mem_wait |= waits
    for h in fixes:
        held[capture_version(h)] = measured_ledger(out / f"held-{capture_version(h)}", got.get(h, []), h,
                                                   build=False, wait=False)
        mark(h)
    if not final:
        LAST.write_text(json.dumps({"commit": commit, "te_core": te_version(commit), "stamp": stamp}) + "\n")
    trace_version = capture_version(trace40.TE40_WT)
    # the runs each newer capture has traced are scored on its traces already, where the re-score's te_core reads them
    over = [p for p in layers() if (nxt or p in tiered or p.name.endswith(FIX))
            and version(capture_version(p)) <= version(te_version(commit))]
    rs = out / "rescore"
    # the tiers' node runs that ended during the wait taken too
    for te in tiered:
        t, again = tiers[layer_tier(te)], measured_ledger(out / f"tier-{te.name}", (), te, build=False, wait=False)
        tiers[layer_tier(te)] = (t | {k: v for k, v in again.items() if k not in ("took", "packed")}
                                 | {"took": t["took"] + again["took"]})
    traced, root = (None, None) if final else (
        measured_ledger(out, shards.get(trace40.TE40_WT, []), over=over, wait=not args.no_trace and not nxt),
        trace40.MEASURED)
    last = rescore(rs, args.gates, final, root)
    on = nxt or trace40.TE40_WT
    untraced = [] if final or not same_capture(capture_version(on), CAPTURE_FINAL) else reached_left(on)
    hold = (f"{len(untraced)} of the published and reached runs not yet traced on {CAPTURE_FINAL}'s capture"
            if untraced else None)
    if HERE_DIR and not os.environ.get("TE_SERVED") and not hold:
        hold = "the VM read no served panels for this refresh on the node (TE_SERVED)"
    policy = {} if final else policy_panels(out, commit, not args.no_put and not hold, root, args.gates)
    if hold:
        policy |= {k: f"not PUT ({hold})" + (f"; its build failed: {v.removeprefix('not PUT: ')}"
                                             if str(v).startswith("not PUT: ") else "")
                   for k, v in policy.items() if k in ("algorithm-policy", FIT)}
    put = "not PUT (final values)" if final else branches(rs, root)
    variants = {}
    for name, extra in ({} if final else VARIANTS).items():
        try:
            variants[name] = rescore(rs / name, args.gates, None, root, extra)
        except SystemExit as e:
            variants[name] = str(e)[:300]
    # every refresh builds the measured ledger afresh and nothing after the re-scores reads it: its 4.7 GB go before
    # the tables' put, which needs the tree's size free in the store (this VM's disk is shared)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)
    what = "final table" if final else "refresh"
    fr = json.loads((rs / "frontier.json").read_text())
    # each row carries its checkpoint's trace: 1.5 GB at 00:30Z, which parsed whole took the refresh past this VM's
    # memory (killed at 6.9 GB), so a row's run and start are all that's kept
    with (rs / "formalism.jsonl").open() as f:
        scored = [(x["run_id"], x["start"]) for x in map(json.loads, f)]
    runs, n2 = {run for run, _ in scored}, night2_runs()
    left = "not_analysed" if final else "not_measured"
    v3 = v3_starts(commit)
    by_start = {}
    for run, start in scored:
        by_start.setdefault(start, set()).add(run)
    meta = {"stamp": stamp, "te_core": commit, "te_core_version": te_version(commit), "te_core_chosen": chosen,
            "capture": trace_version, "over": [capture_version(p) for p in over],
            "integrator_te_core": newest, "pins": PINS,
            "quantities": ("rescore_v1's analysis" if final else
                           f"measured on each record's traced step (te_core {trace_version}'s capture, t literal; "
                           "trace40.py)"),
            "formalism": "v2, the frozen tail always charged", "checkpoints": len(scored), "records": len(runs),
            "night2_records": len(runs & n2), left: len(fr.get(left) or []),
            "v3": {s: {"records": len(by_start.get(s, ())), "checkpoints": sum(start == s for _, start in scored)}
                   for s in v3},
            "pilot_start": sum(v == "pilot-start" for v in (fr.get("skipped") or {}).values()),
            "rule": "H and W* over every checkpoint (19:50Z)"}
    if policy:
        meta["algorithm_policy"] = {k: v for k, v in policy.items()
                                    if k not in ("coverage", "ladder", "ladder_used", "console", "coverage_console",
                                                 "labels", "priced")}
        meta["console"], meta["coverage_console"] = policy.get("console"), policy.get("coverage_console")
        meta["labels"], meta["priced"] = policy.get("labels"), policy.get("priced")
        if "ladder" in policy:
            meta["ladder_points"], meta["ladder_used"] = policy["ladder"], policy.get("ladder_used")
        apc = out / "algorithm-policy" / "console" / "algorithm-policy.json"
        if apc.exists():
            q = json.loads(apc.read_text())
            meta["algorithm_policy_cells"] = {c: sorted({str(r[q["columns"].index(c)]) for r in q["rows"]})
                                              for c in ("reading", "te_core", "binding_kind") if c in q["columns"]}
            meta["algorithm_policy_cells"]["rho_console"] = "rho_console" in q["columns"]
    if retrace:
        meta["next_capture"] = retrace
    if tiers:
        meta["tiers"] = tiers
    if pools:
        meta["pools"] = pools
    if untraced:
        meta["put_held_for"] = untraced
    if mem_wait:
        meta["trace_mem_wait"] = mem_wait
    if held:
        meta["held"] = held
    if traced:
        meta |= {"trace": traced, "variants": {k: str(v)[:200] for k, v in variants.items()},
                 "delta_t": "delta_t_bits" in git("show", f"{commit}:benchmarks/covert_witness/te_core/formalism.py")}
    trees = {k: out / "publish" / k for k in ("tables", "fb")}
    for t in trees.values():
        t.mkdir(parents=True, exist_ok=True)
    # the re-score's files are linked, not copied (two formalism.jsonl of 2.1 GB each): nothing writes them after the
    # re-scores, and both directories go once the trees are in the store
    for name in ("frontier.md", "frontier.json", "formalism.jsonl", "rescore.log", "compare.md", "compare.json"):
        if (rs / name).exists():
            link(rs / name, trees["tables"] / name)
    if traced and (trace40.cache() / "traced.json").exists():
        shutil.copyfile(trace40.cache() / "traced.json", trees["tables"] / "trace40-traced.json")
    for v in variants:
        for name in ("frontier.md", "frontier.json", "rescore.log", "formalism.jsonl"):
            if (rs / v / name).exists():
                (trees["tables"] / v).mkdir(exist_ok=True)
                link(rs / v / name, trees["tables"] / v / name)
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
        ids["tables"] = here_out(trees["tables"], "tables") if HERE_DIR else put_data(
            trees["tables"], "evidence/v1", {"what": f"te formalism v2 {what}: te_core.rescore's frontier and "
                                                     "per-checkpoint blocks", **meta})
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
            ids["fb"] = here_out(trees["fb"], "fb") if HERE_DIR else put_data(
                trees["fb"], "evidence/v1", {"what": f"te formalism v2 {what} {FB}", "put": put, **meta})
    fb_rows = {"uncut": len(fb["rows"]), "console": len(json.loads(cut)["rows"])} if fb else None
    summary = meta | {"art": ids, "frontier_branches": {"put": put, "rows": fb_rows}, "rescore": last,
                      "out": str(out)}
    (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
    line = digest(json.loads((rs / "p2-slowdown-vs-gain.json").read_text()),
                  meta | {"version_cell": bool(fb and "te_core_version" in fb["columns"])}, ids, put, fb_rows)
    (out / "digest.txt").write_text(line + "\n")
    # this VM's disk is shared and a refresh's copies are about 5 GB; what later refreshes read of one (`retrace_order`)
    # is its algorithm-policy directory, which stays
    if ids.get("tables") and (fb is None or ids.get("fb")):
        for d in (rs, out / "publish"):
            shutil.rmtree(d, ignore_errors=True)
    said, tables = policy.get("coverage") or [], S.short(ids.get("tables", "–"))
    cov = (f"training/algorithm-policy (te_core {meta['te_core_version']}; tables {tables}) "
           + said[0] if said[:1] and said[0].startswith("coverage at ") else
           f"coverage (te_core.algorithm_policy, te_core {meta['te_core_version']}: gain size by decade of ρ, the "
           f"frontier point or the best failed attempt in each cell; tables {tables}): "
           + "; ".join(said) + "." if said else None)
    cc = meta.get("coverage_console") or {}
    if cc.get("line"):
        ck = meta.get("console")
        at = f" (c = {ck[0]:.6g} gates, k = {ck[1]:,.0f} bits)" if ck else ""
        cov = ((cov + " " if cov else "") + f"At the console default{at} (te_core.formalism.published: reading A "
               f"rows at the measured kinds' max where it exceeds 1, reading B rows at the token form, a target), "
               f"{cc['unmeasured']} rows unmeasured (null) and {cc.get('partial', 0)} partial (a lower bound, a "
               f"valid kind unmeasured) in no cell or frontier; {cc.get('fresh_records', 0)} records with a measured "
               f"fresh-products term: {cc['line']}")
    if meta.get("priced"):
        cov = (cov + " " if cov else "") + meta["priced"]
    posts = [["FINAL-TABLE" if final else "DIGEST", line], *([["DIGEST", cov]] if cov else [])]
    (out / "post.json").write_text(json.dumps(posts) + "\n")
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        for tag, text in posts:
            subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", tag, text], check=True)
    if HERE_DIR:
        summary["here"] = stop_here(float(os.environ.get("TE_HERE_UNTIL") or time.time()))
        (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
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
    arts = ", ".join(f"{k} {S.short(v)}" for k, v in ids.items())
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
    where = ("this refresh's node, beside it" if HERE_DIR else
             "the research pools of " + ", ".join(sorted({m.removeprefix("vy-nebius-") for m, _, _ in TRACE_POOLS})))
    refused = tr["ledger"].get("refused_runs") or []
    nx = meta.get("next_capture")
    nx_refused = len(nx.get("launch_refused") or []) if nx else 0
    layered = [f"te_core {v}'s capture's traces of its {h['traced_runs']} traced runs"
               for v, h in (meta.get("held") or {}).items()]
    layered += [f"the new capture's traces of its {nx['traced_runs']} traced runs"] if nx else []
    nx_said = (f" (te_core {nx['te_core']} prices a row from its own capture ({nx['commit'][:11]}), which is tracing "
               f"every run over the CPU pool, v3's runs that reached a gain first, then each family's best run that "
               f"didn't, the rest of v3's, the pilot's last: {nx['covered']} of v3's {nx['of']} done so far, "
               f"{nx['traced']} traced"
               + (f", {nx_refused} of {len(nx.get('launch') or [])} shard launches refused, packed again at the next "
                  "refresh" if nx_refused else "") + "; "
               f"this refresh scores on te_core {meta['te_core_version']} over {meta.get('capture', '?')}'s capture"
               + (", with " + " and ".join(layered) + " over it" if meta.get("over") else "")
               + " until it traces as many runs, and as many of v3's)" if nx else "")
    dt = ("; Δ_t a straight-through run's fp32 latents, else Δ (00:47Z)" if meta.get("delta_t")
          else "; Δ_t not priced (this te_core has none)")
    v3_said = ", ".join(f"{s} {x['records']} ({x['checkpoints']} checkpoints)" for s, x in meta["v3"].items()
                        if x["records"]) or "none measured yet"
    ap, cells, ck = meta.get("algorithm_policy") or {}, meta.get("algorithm_policy_cells") or {}, meta.get("console")
    cells_said = (f" (every row: reading {'/'.join(cells.get('reading') or ['–'])}, te_core "
                  f"{'/'.join(cells.get('te_core') or ['–'])}"
                  + ((f", rho_console at te_core's console c = {ck[0]:.6g} gates, k = {ck[1]:,.0f} bits" if ck
                      else ", rho_console at te_core's console c") if cells.get("rho_console") else "")
                  + (f", the one bound's binding kind {'/'.join(cells['binding_kind'])} with what it rests on and each "
                     "kind's term (reduction and update as sensitivities, 08:26Z)" if cells.get("binding_kind") else "")
                  + ", rho_default)" if cells else "")
    lab = meta.get("labels") or {}
    cells_said += (f", labelled in rests_on (v1labels): {lab.get('unpriced', 0)} void rows \""
                   f"{v1labels.UNPRICED.format('CAUSE')}\", {lab.get('under_review', 0)} rows with a frozen embedding "
                   f"\"{v1labels.UNDER_REVIEW}\"" if lab else "")
    ladder, used = meta.get("ladder_points"), meta.get("ladder_used") or {}
    n = f"{used['points']} " if used else ""
    ladder_said = ("" if ladder is None else f" (v3's fit with ladder-v3's {n}points: {', '.join(ladder)})" if ladder
                   else f" (v3's fit with te_core.frontier_fit's {n}ladder points, {used['art'][:16]})"
                   if used.get("art") and used["points"] else " (no ladder points, so v3's fit has only its starts)")
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
