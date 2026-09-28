#!/usr/bin/env python3
"""Pace circuits' Commits back onto node 1 after the disk hold (circuits, Slack 1790811206.136429, 4:33 PM PDT).

Every tick: circuits' Commit workloads in deployments-gpu (items vllm-epoch-run/* and n2-build/*) that this controller didn't
release stay deactivated (spec.active=false), new arrivals included. It reactivates them in circuits' order while
  - at most MAX_INFLIGHT (6) of its releases are unfinished,
  - at most MAX_BIG (6) Commits at batch 8 or more are unfinished (released or admitted),
  - node 1's unreplayed replay bundles (jobs/cov, jobs/probe-jit), plus the estimated bundles of the unfinished Commits and of
    the candidate, stay under the cap: what is left under CAP_ROOM_PCT (75) of /workspace after everything that is not a bundle,
    at most CAP_GB (2000), recomputed every tick; 150 GB once the disk has reached 78%. A model's estimate is learned from its
    finished bundles (the largest GB per batch x tokens seen, x1.25; bundle-sizes.json). Until then it is infra's 120 GB x batch/8 x (input + output tokens)/1152
    for Phi-3-mini, scaled by the model's hidden size x layers against Phi-3-mini's and at least 10 GB (it stays above every bundle
    measured on Sep 30: Phi-3-mini, Mistral-7B, TinyLlama, Llama-3.2-1B, Qwen2.5-0.5B and -1.5B),
  - a model the pacer has no size for (none learned, and not in the formula's table, so the formula guesses it as Phi-3-mini:
    on 2 Oct that guess ran 1.3-6x low for qwen25-3b, yi15-6b and gemma2-9b, and node 1's disk went from 61% to 80% in 26 min)
    has at most one Commit unfinished until one has succeeded and taught its size; while that one is admitted the LocalQueue
    stays held, so a second can't be admitted the moment it arrives,
  - node 1's /workspace is under DISK_PAUSE_PCT (80, as df reports it): at or over it nothing is released (infra, 4:46 PM PDT).
Rows circuits keeps held (batch 64, batch 16+ of 7B+ or MoE models, Gemma-2 at batch 16+ with 1024 input tokens, and the named ones) are never reactivated. It evicts nothing
that Kueue has admitted, and deletes nothing. Since Kueue admits a leased Commit before a tick can hold it, the pacer holds
the deployments-gpu LocalQueue itself (stopPolicy Hold) unless the next Commit's projected bundle fits under the cap, and
releases its own hold once it does (gate()); the lease pools' GPU holders use their own LocalQueue, so it gates Commits only. With no Commit in flight it always lets one through, outside the 78% latch, so a big one can't starve. It ticks every TICK_S (10) s and logs a tick when its summary changes or once a minute;
a Job created while GPU quota is free is admitted in the same second, before any tick, so a lane must not resubmit a held Commit. `--dry-run` prints what it would do.

Runs on vy-nebius-1 as research, from the deployed sky/ (tmux `commit-release`):
    python3 /workspace/jobs/dispatch/infra/nebius/sky/release.py 2>&1 | tee -a ~/commit-release/log
Its state (released, seen-inactive, no-release, cap-150, gpu-held) is in $VY_PACER_DIR, default ~/commit-release.
"""
import json
import os
import re
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
CIRCUITS = ("vllm-epoch-run/", "n2-build/", "vllm-coverage-defs/")
MAX_INFLIGHT, MAX_BIG, DISK_PAUSE_PCT = 6, 6, 80   # circuits, 2:04 AM PDT Oct 2: batch 8+ as many as in flight (was 4, a
                                                   # disk guard the disk-following cap and the 80% pause now cover)
BOOST_UNTIL, BOOST_INFLIGHT, BOOST_BIG = 1790866200, 8, 6   # circuits, 6:22 AM PDT Oct 1: 8 and 6 until 7:50 AM PDT (14:50Z)


def limits():
    return (BOOST_INFLIGHT, BOOST_BIG) if time.time() < BOOST_UNTIL else (MAX_INFLIGHT, MAX_BIG)
CAP_GB, CAP_LOW_GB, CAP_DROP_PCT = 2000, 150, 78  # the 78% drop latch and the 80% pause stay the guard
CAP_ROOM_PCT = 75   # root, 6:31 PM PDT Oct 1: the cap follows the disk (was a fixed 1 TB, top-level 7:24 PM PDT Sep 30); 3 points under the latch
ROOT = Path(os.environ.get("VY_PACER_DIR", str(Path.home() / "commit-release")))
NO_RELEASE = ROOT / "no-release"   # while it exists, nothing is released (new arrivals still wait deactivated)
CAP_LATCH = ROOT / "cap-150"  # the drop is a latch: delete this file to go back to CAP_GB
TICK_S, LOG_EVERY_S = 10, 60
_last_tick = {"summary": None, "t": 0.0}
STATE = ROOT / "released"
SEEN = ROOT / "seen-inactive"      # workloads once deactivated: reactivated by anyone, they count as released
DRY = "--dry-run" in sys.argv


def k(*args, stdin=None):
    r = subprocess.run(["sudo", "k3s", "kubectl", *args], input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {'DRY ' if DRY else ''}{msg}", flush=True)


def bundles_gb():
    r = subprocess.run("sudo find /workspace/jobs/cov /workspace/jobs/probe-jit -xdev -name 'replay_bundle_p*' -prune -print0 2>/dev/null"
                       " | sudo xargs -0r du -sb 2>/dev/null | awk '{t+=$1} END {print t+0}'", shell=True, capture_output=True, text=True)
    return int(r.stdout.strip() or 0) / 1e9


def room_cap(gb):
    """The bundle cap, GB: CAP_ROOM_PCT of /workspace less everything on it that is not a bundle (gb), at most CAP_GB."""
    s = os.statvfs("/workspace")
    used, total = (s.f_blocks - s.f_bfree) * s.f_frsize / 1e9, (s.f_blocks - s.f_bfree + s.f_bavail) * s.f_frsize / 1e9
    return int(max(0, min(CAP_GB, CAP_ROOM_PCT / 100 * total - (used - gb))))


def disk_pct():
    s = os.statvfs("/workspace")
    used = s.f_blocks - s.f_bfree
    return 100 * used / (used + s.f_bavail)


def written_gb(w):
    """What an unfinished Commit's bundle already holds on disk (it's counted in bundles_gb), so its estimate counts only the rest."""
    if not w.get("sweep") or not w.get("row"):
        return 0.0
    r = subprocess.run(f"sudo du -sbc {w['sweep']}/{w['row']}/commit/replay_bundle_p* 2>/dev/null | tail -1", shell=True,
                       capture_output=True, text=True)
    parts = r.stdout.split()
    return int(parts[0]) / 1e9 if parts and parts[0].isdigit() else 0.0


def batch(row):
    m = re.search(r"__b(\d+)__", row)
    return int(m.group(1)) if m else 0


HIDDEN_X_LAYERS = {"smollm2-135m": 576 * 30, "smollm2-360m": 960 * 32, "qwen25-05b": 896 * 24, "qwen25-15b": 1536 * 28,
                   "qwen25-7b": 3584 * 28, "qwen3-4b": 2560 * 36, "qwen3-30b-a3b": 2048 * 48, "olmoe-1b-7b": 2048 * 16,
                   "llama32-1b": 2048 * 16, "tinyllama-11b": 2048 * 22, "phi3-mini": 3072 * 32, "mistral-7b": 4096 * 32,
                   "gemma2-2b": 2304 * 26, "pythia-160m": 768 * 12, "llama31-8b": 4096 * 32}
REF = HIDDEN_X_LAYERS["phi3-mini"]          # unknown models are estimated as Phi-3-mini


SIZES = ROOT / "bundle-sizes.json"    # learned: per model, the largest succeeded Commit's bundle in GB per (batch x tokens)
TRACK = ROOT / "bundle-track.json"    # per Commit in flight: the most its bundle has held, and its Job
# SIZES' mirror under /workspace, which the pack pods mount (their home isn't research's): commit_pack.py reads it
SHARED = Path(os.environ.get("VY_BUNDLE_SIZES", "/workspace/jobs/dispatch/bundle-sizes.json"))
LEARN_MARGIN = 1.25
_learned: dict[str, float] | None = None


def tokens(row):
    m = re.search(r"__i(\d+)__o(\d+)__", row)
    return int(m.group(1)) + int(m.group(2)) if m else 1152


def model(row):
    return row.split("__", 1)[0]


def learned():
    global _learned
    if _learned is None:
        try:
            _learned = {k: float(v) for k, v in json.loads(SIZES.read_text()).items()}
        except (OSError, ValueError):
            _learned = {}
    return _learned


def learn(flying, written, jobs):
    """Track the most each in-flight Commit's bundle has held (written), and once its workload has finished, record its model's
    GB per (batch x tokens) if its Job succeeded. A bundle on disk is no measure: its replay deletes it and may keep leaves, and
    a failed Commit's is partial."""
    try:
        track = json.loads(TRACK.read_text())
    except (OSError, ValueError):
        track = {}
    for w in flying:
        t = track.setdefault(w["row"], {"gb": 0.0, "job": w.get("job", "")})
        t["gb"] = max(t["gb"], written.get(w["name"], 0.0))
    table, changed, live = learned(), False, {w["row"] for w in flying}
    for row in [r for r in track if r not in live]:
        st = (jobs.get(track[row]["job"]) or {}).get("status") or {}
        if st.get("succeeded") and batch(row) > 0 and track[row]["gb"] > 0:
            rate = track[row]["gb"] / (batch(row) * tokens(row))
            if rate > table.get(model(row), 0.0):
                table[model(row)], changed = rate, True
        if st.get("succeeded") or st.get("failed") or track[row]["job"] not in jobs:
            del track[row]
    if not DRY:
        TRACK.parent.mkdir(parents=True, exist_ok=True)
        TRACK.write_text(json.dumps(track, indent=1, sort_keys=True) + "\n")
        if changed:
            SIZES.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
        share(table)


_share_failed = False


def share(table):
    """Mirror the learned table to SHARED when it differs, so the pack pods (commit_pack.py) estimate as the pacer does; written to
    a temporary name and renamed, so a pod never reads half of it."""
    global _share_failed
    text = json.dumps(table, indent=1, sort_keys=True) + "\n"
    try:
        if SHARED.exists() and SHARED.read_text() == text:
            return
        SHARED.parent.mkdir(parents=True, exist_ok=True)
        tmp = SHARED.with_name(SHARED.name + ".tmp")
        tmp.write_text(text)
        tmp.replace(SHARED)
        _share_failed = False
    except OSError as e:
        if not _share_failed:
            log(f"could not mirror the learned bundle sizes to {SHARED}: {e}")
        _share_failed = True


def sized(row):
    """Whether the pacer has a size for `row`'s model it trusts: a rate learned from a succeeded Commit, or the formula's own
    row (HIDDEN_X_LAYERS). Any other model is estimated as Phi-3-mini, a guess."""
    return model(row) in learned() or any(row.startswith(k + "__") for k in HIDDEN_X_LAYERS)


def estimate_gb(row):
    """A Commit's bundle, GB: from the largest bundle of its model a succeeded Commit wrote (learn()) with LEARN_MARGIN, else
    infra's formula scaled from Phi-3-mini, which runs 4-9x over the grid's small models (Oct 1: 51 and 107 GB against 480)."""
    rate = learned().get(model(row))
    if rate:
        return max(10.0, rate * max(batch(row), 1) * tokens(row) * LEARN_MARGIN)
    size = next((v for k, v in HIDDEN_X_LAYERS.items() if row.startswith(k + "__")), REF) / REF
    return max(10.0, 120 * max(batch(row), 1) / 8 * tokens(row) / 1152 * size)


def kept(cov, row):
    b = batch(row)
    return cov in KEEP or b >= 64 or (b >= 16 and row.startswith(BIG_MODELS)) \
        or (b >= 16 and row.startswith("gemma2-") and "__i1024__" in row)   # top-level, 1:30 AM PDT: Gemma-2 b16+ i1024 stays held


GPU_QUEUE = "deployments-gpu"                                  # the LocalQueue (namespace default) circuits' Commits are submitted to
GATE = ROOT / "gpu-held"                                      # exists while the pacer itself holds GPU_QUEUE


def gate(close, projected, cap, probing=()):
    """Kueue admits a leased Commit (it requests no GPU) the moment it arrives, before a tick can deactivate it, so the pacer holds
    the GPU_QUEUE LocalQueue itself (stopPolicy Hold: nothing new is admitted from it, nothing running is touched) while `close`,
    and lets it go otherwise. tick() closes it while the admitted Commits' projection plus the next Commit's (the next released
    or waiting one, else the largest admitted, standing for the next arrival) would pass the cap, or the disk is at
    DISK_PAUSE_PCT, or a Commit of a model with no size (`probing`, sized()) is admitted: the next arrival may be another of
    that model. With no Commit admitted it always opens (outside the 78% latch), so a big one can't starve.
    The hold gates Commits only, by construction: the lease pools' GPU holders are submitted to their own LocalQueue
    (deployments-gpu-pool, n1_lease.py's VY_POOL_LOCAL_QUEUE), and the quiet hour and the disk guard hold the ClusterQueue, a
    different object. It only undoes its own hold and never touches a HoldAndDrain. `projected` is for the log."""
    policy = k("get", "localqueue", GPU_QUEUE, "-n", "default", "-o", "jsonpath={.spec.stopPolicy}").strip() or "None"
    if close:
        if policy == "None":
            why = (f"a Commit of {', '.join(probing)} (no size yet) is admitted" if probing else
                   f"projected {projected:.0f} GB with the next Commit is over the {cap} GB cap (or the disk at {DISK_PAUSE_PCT}%)")
            log(f"{why}: holding the {GPU_QUEUE} LocalQueue, so Kueue admits no new Commit")
            if not DRY:
                k("patch", "localqueue", GPU_QUEUE, "-n", "default", "--type=merge", "-p", '{"spec":{"stopPolicy":"Hold"}}')
                GATE.parent.mkdir(parents=True, exist_ok=True)
                GATE.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        return
    if not GATE.exists():
        return
    if policy == "Hold":
        log(f"projected {projected:.0f} GB with the next Commit fits under the {cap} GB cap: releasing the {GPU_QUEUE} LocalQueue")
        if not DRY:
            k("patch", "localqueue", GPU_QUEUE, "-n", "default", "--type=merge", "-p", '{"spec":{"stopPolicy":"None"}}')
    if not DRY:
        GATE.unlink(missing_ok=True)


def tick():
    jobs = {j["metadata"]["name"]: j for j in json.loads(k("get", "jobs", "-o", "json"))["items"]}
    released = set(STATE.read_text().split()) if STATE.exists() else set()
    seen = set(SEEN.read_text().split()) if SEEN.exists() else set()
    ws = []
    for w in json.loads(k("get", "workloads", "-o", "json"))["items"]:
        st = w.get("status") or {}
        if w["spec"].get("queueName") != "deployments-gpu" or any(c["type"] == "Finished" and c["status"] == "True"
                                                                  for c in st.get("conditions", [])):
            continue
        own = [o["name"] for o in w["metadata"].get("ownerReferences", []) if o["kind"] == "Job"]
        key = ((jobs.get(own[0]) or {}).get("metadata", {}).get("annotations", {}) if own else {}).get("verity.dev/item", "")
        if not key.startswith(CIRCUITS):
            continue
        env = {e["name"]: e.get("value") or "" for c in w["spec"]["podSets"][0]["template"]["spec"]["containers"] for e in c.get("env", [])}
        ws.append({"name": w["metadata"]["name"], "cov": key.split("/", 1)[1], "row": env.get("ROW", ""), "sweep": env.get("SWEEP_DIR", ""),
                   "active": w["spec"].get("active", True), "admitted": bool(st.get("admission")), "job": own[0] if own else ""})
    live = {w["name"] for w in ws}
    released = {w["name"] for w in ws if w["name"] in released and w["active"]}   # finished or re-held releases drop out
    seen = (seen & live) | {w["name"] for w in ws if not w["active"]}
    for w in ws:                                                # a new arrival waits deactivated; one someone reactivated is released
        if w["active"] and not w["admitted"] and w["name"] in seen:
            released.add(w["name"])
        elif w["active"] and not w["admitted"] and w["name"] not in released:
            log(f"hold {w['cov']} ({w['row'][:40]}) {w['name']}")
            if not DRY:
                k("patch", "workload", w["name"], "--type=merge", "-p", '{"spec":{"active":false}}')
            w["active"] = False
    inflight = len(released | {w["name"] for w in ws if w["admitted"]})     # every unfinished circuits Commit, whoever released it
    big = sum(1 for w in ws if (w["admitted"] or w["name"] in released) and batch(w["row"]) >= 8)
    gb = bundles_gb()
    disk = disk_pct()
    if disk >= CAP_DROP_PCT and not CAP_LATCH.exists():
        log(f"disk {disk:.1f}% reached {CAP_DROP_PCT}%: the bundle cap drops to {CAP_LOW_GB} GB until {CAP_LATCH} is removed")
        if not DRY:
            CAP_LATCH.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} disk {disk:.1f}%\n")
    latched = CAP_LATCH.exists() or disk >= CAP_DROP_PCT
    cap = CAP_LOW_GB if latched else room_cap(gb)
    flying = [w for w in ws if w["admitted"] or w["name"] in released]
    written = {w["name"]: written_gb(w) for w in flying}
    learn(flying, written, jobs)
    rest = {w["name"]: max(estimate_gb(w["row"]) - written[w["name"]], 0) for w in flying}   # a Commit's bundle so far is already in gb
    projected = gb + sum(rest.values())
    probing = {model(w["row"]) for w in flying if not sized(w["row"])}     # one Commit at a time of a model with no size yet
    order = {c: i for i, c in enumerate(WAVES)}
    cands = sorted((w for w in ws if not w["active"] and not w["admitted"] and not kept(w["cov"], w["row"])),
                   key=lambda w: (order.get(w["cov"], len(WAVES)), batch(w["row"])))
    for w in cands:
        if inflight >= limits()[0] or disk >= DISK_PAUSE_PCT or NO_RELEASE.exists():
            break
        fits = projected + estimate_gb(w["row"]) <= cap or (inflight == 0 and not latched)   # with none in flight, a big one can't starve
        if (batch(w["row"]) >= 8 and big >= limits()[1]) or not fits or (not sized(w["row"]) and model(w["row"]) in probing):
            continue
        log(f"release {w['cov']} ({w['row'][:40]}) {w['name']}: in flight {inflight}, batch-8+ {big}, bundles {gb:.0f} GB, "
            f"projected {projected:.0f} + {estimate_gb(w['row']):.0f} GB")
        if not DRY:
            k("patch", "workload", w["name"], "--type=merge", "-p", '{"spec":{"active":true}}')
        released.add(w["name"])
        inflight += 1
        projected += estimate_gb(w["row"])
        big += batch(w["row"]) >= 8
        if not sized(w["row"]):
            probing.add(model(w["row"]))
    admitted = [w for w in ws if w["admitted"]]
    waiting = [w for w in ws if w["name"] in released and not w["admitted"]] + [w for w in cands if w["name"] not in released]
    base = gb + sum(rest.get(w["name"], 0) for w in admitted)
    reserve = estimate_gb(waiting[0]["row"]) if waiting else max((estimate_gb(w["row"]) for w in admitted), default=0.0)
    room = (not admitted and not latched) or base + reserve <= cap
    probe = sorted({model(w["row"]) for w in admitted if not sized(w["row"])})
    gate(not room or disk >= DISK_PAUSE_PCT or bool(probe), base + reserve, cap, probe)
    if not DRY:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text("\n".join(sorted(released)) + "\n")
        SEEN.write_text("\n".join(sorted(seen)) + "\n")
    held = [w["cov"] for w in ws if not w["active"] and not w["admitted"]]
    summary = (f"tick: {'NO-RELEASE, ' if NO_RELEASE.exists() else ''}cap {cap} GB, disk {disk:.0f}%{' PAUSED' if disk >= DISK_PAUSE_PCT else ''}, in flight {inflight}, batch-8+ {big}, bundles {gb:.0f} GB, projected {projected:.0f} GB, waiting {len(held)}: {' '.join(held)}")
    if summary != _last_tick["summary"] or time.monotonic() - _last_tick["t"] >= LOG_EVERY_S:
        log(summary)
        _last_tick.update(summary=summary, t=time.monotonic())


if __name__ == "__main__":
    once = "--once" in sys.argv or DRY
    while True:
        try:
            tick()
        except Exception as e:  # noqa: BLE001  (a failed tick is retried; nothing is released on an error)
            log(f"tick failed: {e}")
        if once:
            break
        time.sleep(TICK_S)
