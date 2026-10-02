#!/usr/bin/env python3
"""Pace circuits' Commits back onto node 1 after the disk hold (circuits, Slack 1790811206.136429, 4:33 PM PDT).

Every tick: circuits' Commit workloads in deployments-gpu (items vllm-epoch-run/* and n2-build/*) that this controller didn't
release stay deactivated (spec.active=false), new arrivals included. It reactivates them in circuits' order while
  - at most MAX_INFLIGHT (6) of its releases are unfinished,
  - at most MAX_BIG (4) Commits at batch 8 or more are unfinished (released or admitted),
  - node 1's unreplayed replay bundles (jobs/cov, jobs/probe-jit), plus the estimated bundles of the unfinished Commits and of
    the candidate, stay under the cap: what is left under CAP_ROOM_PCT (75) of /workspace after everything that is not a bundle,
    at most CAP_GB (2000), recomputed every tick; 150 GB once the disk has reached 78%. The estimate is infra's 120 GB x batch/8 x (input + output tokens)/1152
    for Phi-3-mini, scaled by the model's hidden size x layers against Phi-3-mini's and at least 10 GB (it stays above every bundle
    measured on Sep 30: Phi-3-mini, Mistral-7B, TinyLlama, Llama-3.2-1B, Qwen2.5-0.5B and -1.5B),
  - node 1's /workspace is under DISK_PAUSE_PCT (80, as df reports it): at or over it nothing is released (infra, 4:46 PM PDT).
Rows circuits keeps held (batch 64, batch 16+ of 7B+ or MoE models, Gemma-2 at batch 16+ with 1024 input tokens, and the named ones) are never reactivated. It evicts nothing
that Kueue has admitted, and deletes nothing. Since Kueue admits a leased Commit before a tick can hold it, the pacer holds
deployments-gpu itself (stopPolicy Hold) unless the next Commit's projected bundle fits under the cap, and releases its own hold
once it does (gate()). With no Commit in flight it always lets one through, outside the 78% latch, so a big one can't starve. It ticks every TICK_S (10) s and logs a tick when its summary changes or once a minute;
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
MAX_INFLIGHT, MAX_BIG, DISK_PAUSE_PCT = 6, 4, 80   # circuits, 6:09 PM PDT: 6 in flight, 4 at batch 8+ (was 3 and 2)
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


def estimate_gb(row):
    m = re.search(r"__i(\d+)__o(\d+)__", row)
    tokens = int(m.group(1)) + int(m.group(2)) if m else 1152
    size = next((v for k, v in HIDDEN_X_LAYERS.items() if row.startswith(k + "__")), REF) / REF
    return max(10.0, 120 * max(batch(row), 1) / 8 * tokens / 1152 * size)


def kept(cov, row):
    b = batch(row)
    return cov in KEEP or b >= 64 or (b >= 16 and row.startswith(BIG_MODELS)) \
        or (b >= 16 and row.startswith("gemma2-") and "__i1024__" in row)   # top-level, 1:30 AM PDT: Gemma-2 b16+ i1024 stays held


GPU_QUEUE = "deployments-gpu"
GATE = ROOT / "gpu-held"                                      # exists while the pacer itself holds GPU_QUEUE
GUARD_STATE = "/var/lib/vy-disk-guard/held"                   # the queues the disk guard holds


def pool_holders_pending():
    """Whether a lease pool's GPU holder (n1_lease.py's `gpu-pool-*` Jobs, which queue in GPU_QUEUE) waits for admission. It
    brings a GPU to Commits already admitted, so the gate opens for it: held, those Commits would wait for a GPU forever, and the
    projection that keeps the gate closed would never fall (Oct 1, 9:09-9:31 PM PDT)."""
    for w in json.loads(k("get", "workloads", "-o", "json"))["items"]:
        st = w.get("status") or {}
        own = [o["name"] for o in w["metadata"].get("ownerReferences", []) if o["kind"] == "Job"]
        if (w["spec"].get("queueName") == GPU_QUEUE and own and own[0].startswith("gpu-pool-") and not st.get("admission")
                and w["spec"].get("active", True)
                and not any(c["type"] == "Finished" and c["status"] == "True" for c in st.get("conditions", []))):
            return True
    return False


def guard_holds():
    """The queues the disk guard holds (its state file is root's)."""
    return set(subprocess.run(["sudo", "cat", GUARD_STATE], capture_output=True, text=True).stdout.split())


def gate(close, projected, cap, now=None, why=None):
    """Kueue admits a leased Commit (it requests no GPU) the moment it arrives, before a tick can deactivate it, so the pacer holds
    GPU_QUEUE itself (stopPolicy Hold: nothing new is admitted, nothing running is touched) while `close`, and lets it go
    otherwise. tick() closes it while the admitted Commits' projection plus the next Commit's (the next released or waiting one,
    else the largest admitted, standing for the next arrival) would pass the cap, or the disk is at DISK_PAUSE_PCT. With no Commit
    admitted it always opens (outside the 78% latch), so a big one can't starve. It only undoes its own hold, never in the quiet
    hour (12:30-13:30Z, whose own timers hold and release) or while the disk guard lists the queue. A lease pool's GPU holder
    waiting for admission opens it too (pool_holders_pending()). `projected` (the projection with the next Commit) and `why`
    are for the log."""
    policy = k("get", "clusterqueue", GPU_QUEUE, "-o", "jsonpath={.spec.stopPolicy}").strip() or "None"
    g = time.gmtime(now)
    quiet = (12, 30) <= (g.tm_hour, g.tm_min) < (13, 30)
    if close:
        if policy == "None" and not quiet:
            log(f"projected {projected:.0f} GB with the next Commit is over the {cap} GB cap (or the disk at {DISK_PAUSE_PCT}%): "
                f"holding {GPU_QUEUE}, so Kueue admits no new Commit")
            if not DRY:
                k("patch", "clusterqueue", GPU_QUEUE, "--type=merge", "-p", '{"spec":{"stopPolicy":"Hold"}}')
                GATE.parent.mkdir(parents=True, exist_ok=True)
                GATE.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", g) + "\n")
        return
    if not GATE.exists():
        return
    if policy == "Hold" and not quiet and GPU_QUEUE not in guard_holds():
        log(f"{why or f'projected {projected:.0f} GB with the next Commit fits under the {cap} GB cap'}: releasing {GPU_QUEUE}")
        if not DRY:
            k("patch", "clusterqueue", GPU_QUEUE, "--type=merge", "-p", '{"spec":{"stopPolicy":"None"}}')
    if not DRY:
        GATE.unlink(missing_ok=True)


def tick(now=None):
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
                   "active": w["spec"].get("active", True), "admitted": bool(st.get("admission"))})
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
    rest = {w["name"]: max(estimate_gb(w["row"]) - written_gb(w), 0) for w in flying}   # a Commit's bundle so far is already in gb
    projected = gb + sum(rest.values())
    order = {c: i for i, c in enumerate(WAVES)}
    cands = sorted((w for w in ws if not w["active"] and not w["admitted"] and not kept(w["cov"], w["row"])),
                   key=lambda w: (order.get(w["cov"], len(WAVES)), batch(w["row"])))
    for w in cands:
        if inflight >= limits()[0] or disk >= DISK_PAUSE_PCT or NO_RELEASE.exists():
            break
        fits = projected + estimate_gb(w["row"]) <= cap or (inflight == 0 and not latched)   # with none in flight, a big one can't starve
        if (batch(w["row"]) >= 8 and big >= limits()[1]) or not fits:
            continue
        log(f"release {w['cov']} ({w['row'][:40]}) {w['name']}: in flight {inflight}, batch-8+ {big}, bundles {gb:.0f} GB, "
            f"projected {projected:.0f} + {estimate_gb(w['row']):.0f} GB")
        if not DRY:
            k("patch", "workload", w["name"], "--type=merge", "-p", '{"spec":{"active":true}}')
        released.add(w["name"])
        inflight += 1
        projected += estimate_gb(w["row"])
        big += batch(w["row"]) >= 8
    admitted = [w for w in ws if w["admitted"]]
    waiting = [w for w in ws if w["name"] in released and not w["admitted"]] + [w for w in cands if w["name"] not in released]
    base = gb + sum(rest.get(w["name"], 0) for w in admitted)
    reserve = estimate_gb(waiting[0]["row"]) if waiting else max((estimate_gb(w["row"]) for w in admitted), default=0.0)
    room = (not admitted and not latched) or base + reserve <= cap
    full = not room or disk >= DISK_PAUSE_PCT
    holders = full and pool_holders_pending()
    gate(full and not holders, base + reserve, cap, now, why="a lease pool's GPU holder waits for admission" if holders else None)
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
