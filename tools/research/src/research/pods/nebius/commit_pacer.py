#!/usr/bin/env python3
"""Node 1's Commit pacer: the one controller that decides when a Commit in Kueue's `deployments-gpu` may start.

    commit_pacer.py run  [--mode hold|gate] [--every 10]   tick forever (commit-pacer.service)
    commit_pacer.py once [--mode hold|gate] [--dry-run]    one tick; --dry-run prints what it would do
    commit_pacer.py status                                 every unfinished Commit: running, waiting or queued, and why
    commit_pacer.py cleanup [--delete]                     failed Commits' replay bundles past retention (lists unless --delete)

It replaces `~research/commit-release/release.py` (circuits' rule, 4:33 PM PDT 30 Sep; sole pacer by the 4:41 PM PDT ruling).

**Commits.** A `deployments-gpu` Workload whose pod set carries `verity.dev/stage: commit` (config-run's gpu task and config-run-row
set it, so a copy of their Job keeps it), or, for an item pinned before that label, whose Job is task 1 of config-run or task 0 of
config-run-row. Its workstream is its item key's first part, else its pod set's `verity.dev/workstream` label; its coverage id the
key's second part, else its SWEEP_DIR's name. A Commit is paced when its workstream is in PACER_SCOPE (comma-separated; `*` for
every Commit; default circuits' three, as release.py). Everything else in `deployments-gpu` is passed.

**The rule** (`decide`). A waiting Commit is released, in circuits' order (WAVES, then batch), only while
  - node 1's `/workspace` is under DISK_PAUSE_PCT (80, as df reports it);
  - fewer than MAX_INFLIGHT (6) paced Commits are in flight (released or admitted, unfinished);
  - it is below batch 8, or fewer than MAX_BIG (4) paced Commits at batch 8 or more are in flight;
  - the replay bundles on disk, plus what the Commits in flight have yet to write, plus its own estimate, stay within the cap:
    CAP_GB (300) until `/workspace` first reaches CAP_DROP_PCT (78%), CAP_LOW_GB (150) from then on (a latch, PACER_STATE/cap-150;
    delete it to go back). The estimate is infra's 120 GB x batch/8 x (input + output tokens)/1152 for Phi-3-mini, scaled by the
    model's hidden size x layers against Phi-3-mini's, and at least 10 GB; a Commit in flight has yet to write its estimate less
    its bundle so far;
  - it isn't kept: on KEEP, at batch 64, or at batch 16+ of a 7B+ or MoE model (BIG_MODELS).
The rule is release.py's as of 01:09Z 1 Oct (md5 535db739: circuits' 300/150 GB cap at 6:04 PM PDT, 6 and 4 at 6:09 PM PDT). The
limits are defaults that PACER_MAX_INFLIGHT, PACER_MAX_BIG, PACER_DISK_PAUSE_PCT, PACER_CAP_GB, PACER_CAP_LOW_GB and
PACER_CAP_DROP_PCT override (the unit reads them from /etc/default/commit-pacer); `run` logs the rule it starts with.
A tick whose disk or bundle reading failed releases nothing. Nothing admitted is evicted, and the pacer deletes nothing but
`cleanup --delete`'s bundles.

**Modes.** `hold` is release.py's: a paced Commit that is active, not admitted and not released is deactivated (`spec.active=false`,
which gives its quota back), and released by reactivation; one that someone else reactivates counts as released. `gate` adds
Kueue's AdmissionCheck `commit-pacer` (controllerName verity.dev/commit-pacer, on the ClusterQueue; commit-pacer-admission.yaml),
so a Workload Kueue reserves quota for starts only once the pacer marks its check Ready. The pacer keeps the AdmissionCheck
Active, marks the check Ready on every passed workload, on every admitted one and on its own releases, and deactivates any other
paced Commit, a hand reactivation included. A Workload's `status.admission` is set at quota reservation, before its checks, so
in-flight counts read the `Admitted` condition. If the pacer is down, nothing new in `deployments-gpu` starts.

**What a waiting Commit shows.** The pacer writes its verdict on the Workload as the annotation `verity.dev/commit-pacer`
(`held <utc>: <reason>` or `released <utc>: ...`), and in gate mode the check's message; `status` prints both, and every
decision is a line of the pacer's log (journalctl -u commit-pacer).

**Cleanup** (`cleanup_candidates`). A replay bundle is `intermediate` (the steward's retention classes): its replay's job deletes it
on success. A Job is gone 6 h after it ends, so each tick records every Commit's item key and commit directory
(`SWEEP_DIR/ROW/commit`) in PACER_STATE/commits.json. A bundle `<commit dir>/replay_bundle_p<N>[.partial]` under a bundle root goes
only once every recorded item of its directory ended failed (its last `done.jsonl` line) at least RETAIN_FAILED_H (6) ago, no
unfinished Job, ready item or pack-spool entry names the directory (a retry under a new id included), and nothing has it open
(`lsof +D`). Each deletion is a line of the hold log.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

WAVES = ["cov-n082", "cov-n083", "cov-n084", "cov-n085", "cov-n087", "cov-n088", "cov-n111", "cov-n116", "cov-n117", "cov-n139-2",
         "cov-n140-2", "cov-n145-2", "cov-n146-2", "cov-g167", "cov-g176", "cov-g181", "cov-g173", "cov-g153",
         "cov-g089", "cov-g090", "cov-g092", "cov-g043", "cov-g116", "cov-g072", "cov-g145", "cov-g164", "cov-n100", "cov-n128",
         "cov-g189", "cov-g055", "cov-g056"]
KEEP = {"cov-n133", "cov-g108", "cov-n002", "cov-g019", "cov-g061", "cov-n158", "cov-g084", "cov-g128"}
BIG_MODELS = ("mistral-7b", "qwen25-7b", "qwen3-30b", "olmoe", "llama31-8b", "gemma2-9b")
HIDDEN_X_LAYERS = {"smollm2-135m": 576 * 30, "smollm2-360m": 960 * 32, "qwen25-05b": 896 * 24, "qwen25-15b": 1536 * 28,
                   "qwen25-7b": 3584 * 28, "qwen3-4b": 2560 * 36, "qwen3-30b-a3b": 2048 * 48, "olmoe-1b-7b": 2048 * 16,
                   "llama32-1b": 2048 * 16, "tinyllama-11b": 2048 * 22, "phi3-mini": 3072 * 32, "mistral-7b": 4096 * 32,
                   "gemma2-2b": 2304 * 26, "pythia-160m": 768 * 12, "llama31-8b": 4096 * 32}
REF = HIDDEN_X_LAYERS["phi3-mini"]          # unknown models are estimated as Phi-3-mini
MAX_INFLIGHT = int(os.environ.get("PACER_MAX_INFLIGHT", "6"))
MAX_BIG = int(os.environ.get("PACER_MAX_BIG", "4"))
DISK_PAUSE_PCT = float(os.environ.get("PACER_DISK_PAUSE_PCT", "80"))
CAP_GB = float(os.environ.get("PACER_CAP_GB", "300"))
CAP_LOW_GB = float(os.environ.get("PACER_CAP_LOW_GB", "150"))
CAP_DROP_PCT = float(os.environ.get("PACER_CAP_DROP_PCT", "78"))
RETAIN_FAILED_H = 6

QUEUE = "deployments-gpu"
CHECK = "commit-pacer"
CONTROLLER = "verity.dev/commit-pacer"
ANNOTATION = "verity.dev/commit-pacer"
STAGE = "verity.dev/stage"
WORKSTREAM = "verity.dev/workstream"
LEGACY_COMMIT_TASKS = {("config-run", "1"), ("config-run-row", "0")}
CIRCUITS = "vllm-epoch-run,n2-build,vllm-coverage-defs"

SCOPE = tuple(s for s in os.environ.get("PACER_SCOPE", CIRCUITS).split(",") if s)
KUBECTL = shlex.split(os.environ.get("PACER_KUBECTL", "k3s kubectl"))
WORKSPACE = os.environ.get("PACER_WORKSPACE", "/workspace")
BUNDLE_ROOTS = [p for p in os.environ.get("PACER_BUNDLE_ROOTS", "/workspace/jobs/cov,/workspace/jobs/probe-jit").split(",") if p]
STATE = Path(os.environ.get("PACER_STATE", "/var/lib/commit-pacer"))
DISPATCH_ROOT = Path(os.environ.get("VY_DISPATCH_ROOT", "/workspace/jobs/dispatch"))
READY = Path(os.environ.get("VY_READY", "/workspace/jobs/ready"))
PACK_SPOOL = Path(os.environ.get("PACK_SPOOL", str(DISPATCH_ROOT / "pack")))
HOLD_LOG = Path(os.environ.get("PACER_HOLD_LOG", "/workspace/research/infra-kueue-hold.txt"))
BUNDLE_NAME = re.compile(r"replay_bundle_p\d+(\.partial)?")
TAKEN_STAMP = re.compile(r"\.\d{8}T\d{6}Z$")


def utc(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_utc(s: str) -> float:
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def log(msg: str, dry: bool = False) -> None:
    print(f"{utc()} {'DRY ' if dry else ''}{msg}", flush=True)


# ---- the rule: pure functions of what one tick read --------------------------------------------------------------------------

def batch(row: str) -> int:
    m = re.search(r"__b(\d+)__", row)
    return int(m.group(1)) if m else 0


def estimate_gb(row: str) -> float:
    m = re.search(r"__i(\d+)__o(\d+)__", row)
    tokens = int(m.group(1)) + int(m.group(2)) if m else 1152
    size = next((v for k, v in HIDDEN_X_LAYERS.items() if row.startswith(k + "__")), REF) / REF
    return max(10.0, 120 * max(batch(row), 1) / 8 * tokens / 1152 * size)


def kept(cov: str, row: str) -> bool:
    b = batch(row)
    return cov in KEEP or b >= 64 or (b >= 16 and row.startswith(BIG_MODELS))


def in_scope(workstream: str, scope: tuple[str, ...] = SCOPE) -> bool:
    return "*" in scope or workstream in scope


def pod_labels(workload: dict) -> dict:
    out: dict = {}
    for ps in workload["spec"].get("podSets") or []:
        out.update(((ps.get("template") or {}).get("metadata") or {}).get("labels") or {})
    return out


def is_commit(workload: dict, job: dict | None) -> bool:
    """A Commit by its pod set's `verity.dev/stage: commit`, else (items pinned before the label) by its Job's template and task."""
    if pod_labels(workload).get(STAGE) == "commit":
        return True
    meta = (job or {}).get("metadata") or {}
    template = (meta.get("annotations") or {}).get("verity.dev/template", "").split("@")[0]
    return (template, (meta.get("labels") or {}).get("verity.dev/task", "")) in LEGACY_COMMIT_TASKS


def env_of(spec: dict) -> dict[str, str]:
    """The env of a pod spec's containers."""
    return {e["name"]: e.get("value") or "" for c in spec.get("containers") or [] for e in c.get("env") or []}


def commit_dir(env: dict) -> str:
    return f"{env['SWEEP_DIR']}/{env['ROW']}/commit" if env.get("SWEEP_DIR") and env.get("ROW") else ""


def view(workload: dict, job: dict | None, scope: tuple[str, ...] = SCOPE) -> dict:
    """The fields of one unfinished `deployments-gpu` Workload that the rule reads."""
    st = workload.get("status") or {}
    conds = {c["type"]: c["status"] == "True" for c in st.get("conditions") or []}
    key = (((job or {}).get("metadata") or {}).get("annotations") or {}).get("verity.dev/item", "")
    env: dict[str, str] = {}
    for ps in workload["spec"].get("podSets") or []:
        env.update(env_of(ps["template"]["spec"]))
    workstream = key.split("/", 1)[0] if "/" in key else pod_labels(workload).get(WORKSTREAM, "")
    cov = key.split("/", 1)[1] if "/" in key else Path(env.get("SWEEP_DIR", "")).name
    checks = st.get("admissionChecks") or []
    commit = is_commit(workload, job)
    return {"name": workload["metadata"]["name"], "key": key, "cov": cov, "row": env.get("ROW", ""), "dir": commit_dir(env),
            "commit": commit, "paced": commit and in_scope(workstream, scope), "active": workload["spec"].get("active", True),
            "reserved": bool(st.get("admission")) or conds.get("QuotaReserved", False), "admitted": conds.get("Admitted", False),
            "check": next((c["state"] for c in checks if c["name"] == CHECK), None), "checks": checks,
            "note": (workload["metadata"].get("annotations") or {}).get(ANNOTATION) or ""}


def decide(ws: list[dict], released: set[str], seen: set[str], bundles_gb: float | None, disk_pct: float | None,
           mode: str = "hold", written: dict[str, float] | None = None, latched: bool = False) -> dict:
    """One tick's actions, each (verb, workload, reason): `hold` (deactivate), `activate` (reactivate), `ready` (mark the check
    Ready); a `wait` reason per waiting paced Commit; the new released and seen sets and cap latch. `ws` are unfinished
    deployments-gpu views, `written` the GB of bundles on disk per commit directory, `latched` whether the cap has dropped."""
    gate = mode == "gate"
    acts: list[tuple[str, dict, str]] = []
    paced = [w for w in ws if w["paced"]]
    active = {w["name"] for w in paced if w["active"]}
    released = released & active                    # finished or re-held releases drop out
    seen = (seen & {w["name"] for w in paced}) | {w["name"] for w in paced if not w["active"]}
    for w in paced:
        if w["admitted"] or not w["active"] or w["name"] in released:
            continue
        if not gate and w["name"] in seen:          # hold mode: reactivated by someone else, so it counts as released
            released.add(w["name"])
            continue
        acts.append(("hold", w, "activated by hand; waits for the pacer" if w["name"] in seen else "arrived; waits for the pacer"))
        w["active"] = False
        seen.add(w["name"])
    if gate:
        for w in ws:
            if w["check"] == "Pending" and w["reserved"] and w["active"]:
                why = ("admitted" if w["admitted"] else "released" if w["name"] in released
                       else "" if w["paced"] else "not a paced Commit")
                if why:
                    acts.append(("ready", w, why))
    flying = [w for w in paced if w["admitted"] or w["name"] in released]
    inflight = len(flying)
    big = sum(1 for w in flying if batch(w["row"]) >= 8)
    written = written or {}
    projected = None if bundles_gb is None else \
        bundles_gb + sum(max(estimate_gb(w["row"]) - written.get(w["dir"], 0.0), 0.0) for w in flying)
    latch = latched or (disk_pct is not None and disk_pct >= CAP_DROP_PCT)
    cap = CAP_LOW_GB if latch else CAP_GB
    order = {c: i for i, c in enumerate(WAVES)}
    waiting = sorted((w for w in paced if not w["active"] and not w["admitted"]),
                     key=lambda w: (order.get(w["cov"], len(WAVES)), batch(w["row"])))
    wait: dict[str, str] = {}
    for w in waiting:
        est = estimate_gb(w["row"])
        if kept(w["cov"], w["row"]):
            wait[w["name"]] = "kept (keep-list, batch 64, or batch 16+ of a 7B+/MoE model)"
        elif disk_pct is None or projected is None:
            wait[w["name"]] = "a reading failed this tick (disk or bundles): nothing is released"
        elif disk_pct >= DISK_PAUSE_PCT:
            wait[w["name"]] = f"/workspace at {disk_pct:.1f}% (pause at {DISK_PAUSE_PCT:g}%)"
        elif inflight >= MAX_INFLIGHT:
            wait[w["name"]] = f"{inflight} Commits in flight (max {MAX_INFLIGHT})"
        elif batch(w["row"]) >= 8 and big >= MAX_BIG:
            wait[w["name"]] = f"{big} Commits at batch 8+ in flight (max {MAX_BIG})"
        elif projected + est > cap:
            wait[w["name"]] = (f"bundles {bundles_gb:.0f} GB + in flight {projected - bundles_gb:.0f} GB + this {est:.0f} GB "
                               f"> {cap:g} GB")
        else:
            acts.append(("activate", w, (f"in flight {inflight}/{MAX_INFLIGHT}, batch-8+ {big}/{MAX_BIG}, "
                                         f"projected {projected:.0f} + {est:.0f}/{cap:g} GB, disk {disk_pct:.1f}%")))
            released.add(w["name"])
            inflight += 1
            big += batch(w["row"]) >= 8
            projected += est
    summary = (f"cap {cap:g} GB, disk {'?' if disk_pct is None else f'{disk_pct:.0f}%'}"
               f"{' PAUSED' if disk_pct is not None and disk_pct >= DISK_PAUSE_PCT else ''}, in flight {inflight}, batch-8+ {big}, "
               f"bundles {'?' if bundles_gb is None else f'{bundles_gb:.0f}'} GB, "
               f"projected {'?' if projected is None else f'{projected:.0f}'} GB, waiting {len(wait)}: "
               + " ".join(w["cov"] for w in waiting if w["name"] in wait))
    return {"acts": acts, "wait": wait, "released": released, "seen": seen, "latch": latch, "summary": summary}


def cleanup_candidates(done: list[dict], owners: dict[str, set[str]], busy: set[str], now: float,
                       bundles: list[Path]) -> list[tuple[Path, str]]:
    """(bundle, reason) for each bundle whose commit directory's recorded items all ended failed at least RETAIN_FAILED_H ago and
    that nothing unfinished or queued names. `done` is done.jsonl in order, `owners` the item keys recorded per commit directory,
    `busy` the commit directories of unfinished Jobs, ready items and pack-spool entries. lsof is the caller's."""
    last: dict[str, dict] = {}
    for d in done:
        last[d["key"]] = d
    out = []
    for p in bundles:
        cdir = str(p.parent)
        keys = owners.get(cdir)
        if not BUNDLE_NAME.fullmatch(p.name) or p.parent.name != "commit" or not keys or cdir in busy:
            continue
        ends = [last.get(key) for key in sorted(keys)]
        if any(e is None or e.get("state") != "failed" or not e.get("t") for e in ends):
            continue
        age_h = min(now - parse_utc(e["t"]) for e in ends) / 3600
        if age_h >= RETAIN_FAILED_H:
            out.append((p, ", ".join(f"{e['key']} failed at {e['t']} (task {e.get('task')})" for e in ends)
                        + f"; {age_h:.1f} h ago, nothing queued"))
    return out


# ---- reads and writes --------------------------------------------------------------------------------------------------------

def k(*args: str) -> str:
    r = subprocess.run([*KUBECTL, *args], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def disk_pct(path: str = WORKSPACE) -> float | None:
    try:
        s = os.statvfs(path)
    except OSError:
        return None
    used = s.f_blocks - s.f_bfree
    return 100 * used / (used + s.f_bavail)


def bundle_dirs(roots: list[str] = BUNDLE_ROOTS) -> list[Path] | None:
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        return []
    r = subprocess.run(["find", *roots, "-xdev", "-name", "replay_bundle_p*", "-prune", "-print0"], capture_output=True, check=False)
    if r.returncode:
        return None
    return [Path(p) for p in r.stdout.decode().split("\0") if p]


def bundle_sizes(dirs: list[Path] | None) -> dict[Path, float] | None:
    """GB per bundle directory, by one `du -sb`."""
    if dirs is None:
        return None
    if not dirs:
        return {}
    r = subprocess.run(["du", "-sb", "--", *map(str, dirs)], capture_output=True, text=True, check=False)
    if r.returncode and not r.stdout:
        return None
    return {Path(p): int(n) / 1e9 for n, p in (line.split("\t", 1) for line in r.stdout.splitlines() if line.strip())}


def bundles_gb(dirs: list[Path] | None) -> float | None:
    sizes = bundle_sizes(dirs)
    return None if sizes is None else sum(sizes.values())


def read_cluster(scope: tuple[str, ...] = SCOPE) -> list[dict]:
    jobs = {j["metadata"]["name"]: j for j in json.loads(k("get", "jobs", "-o", "json"))["items"]}
    ws = []
    for w in json.loads(k("get", "workloads", "-o", "json"))["items"]:
        st = w.get("status") or {}
        if w["spec"].get("queueName") != QUEUE or any(c["type"] == "Finished" and c["status"] == "True"
                                                      for c in st.get("conditions") or []):
            continue
        own = [o["name"] for o in w["metadata"].get("ownerReferences") or [] if o["kind"] == "Job"]
        ws.append(view(w, jobs.get(own[0]) if own else None, scope))
    return ws


def ensure_active(dry: bool) -> None:
    """Mark the AdmissionCheck Active: Kueue admits nothing in a ClusterQueue while a check it lists is inactive. Never False."""
    ac = json.loads(k("get", "admissioncheck", CHECK, "-o", "json"))
    if ac["spec"].get("controllerName") != CONTROLLER:
        raise RuntimeError(f"admissioncheck {CHECK} has controllerName {ac['spec'].get('controllerName')!r}, not {CONTROLLER}")
    if any(c["type"] == "Active" and c["status"] == "True" for c in (ac.get("status") or {}).get("conditions") or []):
        return
    log(f"marking admissioncheck {CHECK} Active", dry)
    if not dry:
        cond = {"type": "Active", "status": "True", "reason": "Active", "message": "commit-pacer is running",
                "lastTransitionTime": utc()}
        k("patch", "admissioncheck", CHECK, "--subresource=status", "--type=merge", "-p",
          json.dumps({"status": {"conditions": [cond]}}))


def same_note(note_text: str, verdict: str, reason: str) -> bool:
    """Whether an annotation already gives this verdict and reason, its numbers aside (the bundle reading moves every tick)."""
    def shape(s: str) -> str:
        return re.sub(r"\d+(\.\d+)?", "#", s)
    return note_text.split(" ", 1)[0] == verdict and shape(note_text.split(": ", 1)[-1]) == shape(reason)


def note(w: dict, verdict: str, reason: str) -> None:
    """Annotate the Workload with the pacer's verdict, unless it already says the same."""
    if not same_note(w["note"], verdict, reason):
        k("annotate", "workload", w["name"], "--overwrite", f"{ANNOTATION}={verdict} {utc()}: {reason}")
        w["note"] = f"{verdict} {utc()}: {reason}"


def apply(plan: dict, dry: bool) -> None:
    for verb, w, why in plan["acts"]:
        log(f"{verb} {w['cov'] or '-'} ({w['row'][:40]}) {w['name']}: {why}", dry)
        if dry:
            continue
        if verb in ("hold", "activate"):
            k("patch", "workload", w["name"], "--type=merge", "-p", json.dumps({"spec": {"active": verb == "activate"}}))
            if verb == "activate":
                note(w, "released", why)
        elif verb == "ready":
            mine = {"name": CHECK, "state": "Ready", "message": f"commit-pacer: {why}", "lastTransitionTime": utc()}
            checks = [mine if c["name"] == CHECK else c for c in w["checks"]]
            k("patch", "workload", w["name"], "--subresource=status", "--type=merge", "-p",
              json.dumps({"status": {"admissionChecks": checks}}))
    if not dry:
        by_name = {w["name"]: w for _, w, _ in plan["acts"]} | plan.get("views", {})
        for name, why in plan["wait"].items():
            note(by_name[name], "held", why)


def load_set(name: str) -> set[str]:
    p = STATE / name
    return set(p.read_text().split()) if p.exists() else set()


def write_state(name: str, text: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / f".{name}.tmp"
    tmp.write_text(text)
    tmp.replace(STATE / name)


def load_commits() -> dict[str, dict]:
    p = STATE / "commits.json"
    return json.loads(p.read_text()) if p.exists() else {}


def record_commits(ws: list[dict], dry: bool) -> None:
    """Remember each Commit's item key and commit directory, which cleanup needs after its Job is gone."""
    commits = load_commits()
    new = {w["key"]: {"dir": w["dir"], "workload": w["name"], "first": utc()}
           for w in ws if w["commit"] and w["key"] and w["dir"] and w["key"] not in commits}
    if new and not dry:
        write_state("commits.json", json.dumps(commits | new, indent=0, sort_keys=True) + "\n")


_last = {"summary": None, "t": 0.0}


def tick(mode: str, dry: bool) -> dict:
    if mode == "gate":
        ensure_active(dry)
    ws = read_cluster()
    record_commits(ws, dry)
    sizes, disk = bundle_sizes(bundle_dirs()), disk_pct()
    written: dict[str, float] = {}
    for p, gb in (sizes or {}).items():
        written[str(p.parent)] = written.get(str(p.parent), 0.0) + gb
    latched = (STATE / "cap-150").exists()
    plan = decide(ws, load_set("released"), load_set("seen-inactive"), None if sizes is None else sum(sizes.values()), disk, mode,
                  written, latched)
    plan["views"] = {w["name"]: w for w in ws}
    if plan["latch"] and not latched:
        log(f"disk {disk:.1f}% reached {CAP_DROP_PCT:g}%: the bundle cap drops to {CAP_LOW_GB:g} GB until {STATE / 'cap-150'} "
            "is removed", dry)
        if not dry:
            write_state("cap-150", f"{utc()} disk {disk:.1f}%\n")
    apply(plan, dry)
    if not dry:
        write_state("released", "".join(f"{n}\n" for n in sorted(plan["released"])))
        write_state("seen-inactive", "".join(f"{n}\n" for n in sorted(plan["seen"])))
    if plan["summary"] != _last["summary"] or time.monotonic() - _last["t"] >= 60:
        log(f"tick ({mode}): {plan['summary']}", dry)
        _last.update(summary=plan["summary"], t=time.monotonic())
    return plan


def status() -> None:
    for w in read_cluster():
        if w["commit"]:
            state = "running" if w["admitted"] else "waiting" if not w["active"] else "queued"
            print(f"{w['name']:52} {state:8} {'paced' if w['paced'] else 'passed':6} check={w['check'] or '-':8} "
                  f"{w['cov'] or '-':14} {w['note'] or '-'}")


def lsof_clean(path: Path) -> bool:
    r = subprocess.run(["lsof", "+D", str(path)], capture_output=True, text=True, check=False)
    return r.returncode == 1 and not r.stdout.strip()   # lsof exits 1, printing nothing, when no process has a file open


def read_json(f: Path) -> dict | None:
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def cleanup(delete: bool) -> None:
    done = [json.loads(line) for line in (DISPATCH_ROOT / "done.jsonl").read_text().splitlines() if line.strip()]
    owners: dict[str, set[str]] = {}
    for key, rec in load_commits().items():
        owners.setdefault(rec["dir"], set()).add(key)
    for f in (DISPATCH_ROOT / "taken").glob("*/*.json"):
        cdir = commit_dir((read_json(f) or {}).get("env") or {})
        if cdir:
            owners.setdefault(cdir, set()).add(f"{f.parent.name}/{TAKEN_STAMP.sub('', f.stem)}")
    busy = {commit_dir(env_of(j["spec"]["template"]["spec"])) for j in json.loads(k("get", "jobs", "-o", "json"))["items"]
            if not any(c["type"] in ("Complete", "Failed") and c["status"] == "True"
                       for c in (j.get("status") or {}).get("conditions") or [])}
    busy |= {commit_dir((read_json(f) or {}).get("env") or {}) for f in READY.glob("*/*.json")}
    for sub in ("queue", "claimed"):
        busy |= {commit_dir(((read_json(f) or {}).get("item") or {}).get("env") or {}) for f in (PACK_SPOOL / sub).glob("*.json")}
    dirs = bundle_dirs()
    if dirs is None:
        raise RuntimeError("bundle scan failed")
    roots = [Path(r).resolve() for r in BUNDLE_ROOTS]
    for path, why in cleanup_candidates(done, owners, busy - {""}, time.time(), dirs):
        real = path.resolve()
        if not any(root in real.parents for root in roots) or not lsof_clean(path):
            log(f"cleanup: keep {path}: open, or outside the bundle roots")
            continue
        size = bundles_gb([path]) or 0.0
        log(f"cleanup: {'delete' if delete else 'would delete'} {path} ({size:.1f} GB): {why}")
        if delete:
            subprocess.run(["nice", "-n", "19", "ionice", "-c3", "rm", "-rf", "--", str(real)], check=True)
            with HOLD_LOG.open("a") as h:
                h.write(f"{utc()} commit-pacer cleanup: deleted {path} ({size:.1f} GB): {why}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "once"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=("hold", "gate"), default=os.environ.get("PACER_MODE", "hold"))
        p.add_argument("--every", type=float, default=10.0)
        p.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    c = sub.add_parser("cleanup")
    c.add_argument("--delete", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        status()
        return 0
    if a.cmd == "cleanup":
        cleanup(a.delete)
        return 0
    log(f"{a.cmd} ({a.mode}): scope {','.join(SCOPE)}; in flight < {MAX_INFLIGHT}, batch-8+ < {MAX_BIG}, pause at "
        f"{DISK_PAUSE_PCT:g}%, cap {CAP_GB:g} GB, {CAP_LOW_GB:g} GB from {CAP_DROP_PCT:g}%", a.dry_run)
    while True:
        try:
            tick(a.mode, a.dry_run)
        except Exception as e:  # noqa: BLE001  (a failed tick releases nothing; the next one retries)
            log(f"tick failed: {e}", a.dry_run)
            if a.cmd == "once":
                return 1
        if a.cmd == "once":
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
