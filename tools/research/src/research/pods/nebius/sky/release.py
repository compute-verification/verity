#!/usr/bin/env python3
"""Paces circuits' Commits back onto node 1 after the disk hold."""
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
MAX_INFLIGHT, MAX_BIG, DISK_PAUSE_PCT = 6, 6, 80
BOOST_UNTIL, BOOST_INFLIGHT, BOOST_BIG = 1790866200, 8, 6


def limits():
    return (BOOST_INFLIGHT, BOOST_BIG) if time.time() < BOOST_UNTIL else (MAX_INFLIGHT, MAX_BIG)
LIMIT_GB, LIMIT_LOW_GB, LIMIT_DROP_PCT = 2000, 150, 78
LIMIT_ROOM_PCT = 75
ROOT = Path(os.environ.get("VY_PACER_DIR", str(Path.home() / "commit-release")))
NO_RELEASE = ROOT / "no-release"
LIMIT_LATCH = ROOT / "limit-150"
OLD_LATCH = ROOT / "cap-150"
TICK_S, LOG_EVERY_S = 10, 60
_last_tick = {"summary": None, "t": 0.0}
STATE = ROOT / "released"
SEEN = ROOT / "seen-inactive"
DRY = "--dry-run" in sys.argv


def k(*args, stdin=None):
    r = subprocess.run(["sudo", "k3s", "kubectl", *args], input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def log(msg):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {'DRY ' if DRY else ''}{msg}", flush=True)


FIND, XARGS, DU = "/usr/bin/find", "/usr/bin/xargs", "/usr/bin/du"
BUNDLES = (f"sudo -n {FIND} /workspace/jobs/cov /workspace/jobs/probe-jit -xdev -name 'replay_bundle_p*' -prune -print0 2>/dev/null"
           f" | sudo -n {XARGS} -0r {DU} -sb 2>/dev/null | awk '{{t+=$1}} END {{print t+0}}'")
WRITTEN = f"sudo -n {DU} -sbc {{}}/commit/replay_bundle_p* 2>/dev/null | tail -1"


def bundles_gb():
    """Raises where sudo refuses a read (BUNDLES, and WRITTEN in written_gb) to this login, as after a lock-out
    (pods/nebius/lockout): their errors go to /dev/null, so a refusal would read as 0 GB and release Commits into a full disk."""
    for tool in (FIND, XARGS, DU):
        if subprocess.run(["sudo", "-n", "-l", tool], capture_output=True).returncode:
            raise SystemExit(f"release.py: sudo refuses {tool} to this login, so the bundles can't be measured; nothing released. "
                             "A locked node grants only `vy-sudo du-bundles` (pods/nebius/lockout): run the pacer as a root unit")
    r = subprocess.run(BUNDLES, shell=True, capture_output=True, text=True)
    return int(r.stdout.strip() or 0) / 1e9


def room_limit(gb):
    s = os.statvfs("/workspace")
    used, total = (s.f_blocks - s.f_bfree) * s.f_frsize / 1e9, (s.f_blocks - s.f_bfree + s.f_bavail) * s.f_frsize / 1e9
    return int(max(0, min(LIMIT_GB, LIMIT_ROOM_PCT / 100 * total - (used - gb))))


def disk_pct():
    s = os.statvfs("/workspace")
    used = s.f_blocks - s.f_bfree
    return 100 * used / (used + s.f_bavail)


def written_gb(w):
    if not w.get("sweep") or not w.get("row"):
        return 0.0
    r = subprocess.run(WRITTEN.format(f"{w['sweep']}/{w['row']}"), shell=True, capture_output=True, text=True)
    parts = r.stdout.split()
    return int(parts[0]) / 1e9 if parts and parts[0].isdigit() else 0.0


def batch(row):
    m = re.search(r"__b(\d+)__", row)
    return int(m.group(1)) if m else 0


HIDDEN_X_LAYERS = {"smollm2-135m": 576 * 30, "smollm2-360m": 960 * 32, "qwen25-05b": 896 * 24, "qwen25-15b": 1536 * 28,
                   "qwen25-7b": 3584 * 28, "qwen3-4b": 2560 * 36, "qwen3-30b-a3b": 2048 * 48, "olmoe-1b-7b": 2048 * 16,
                   "llama32-1b": 2048 * 16, "tinyllama-11b": 2048 * 22, "phi3-mini": 3072 * 32, "mistral-7b": 4096 * 32,
                   "gemma2-2b": 2304 * 26, "pythia-160m": 768 * 12, "llama31-8b": 4096 * 32}
REF = HIDDEN_X_LAYERS["phi3-mini"]


SIZES = ROOT / "bundle-sizes.json"
TRACK = ROOT / "bundle-track.json"
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
            _learned = {k: float(v) for k, v in json.loads(SIZES.read_text()).items()} if SIZES.exists() else {}
        except (OSError, ValueError) as e:
            log(f"{SIZES} unread, so no size is learned yet ({type(e).__name__}: {e})")
            _learned = {}
    return _learned


def learn(flying, written, jobs):
    try:
        track = json.loads(TRACK.read_text()) if TRACK.exists() else {}
    except (OSError, ValueError) as e:
        log(f"{TRACK} unread, so tracking starts again ({type(e).__name__}: {e})")
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


def tp(row):
    m = re.search(r"__tp(\d+)__", row)
    return int(m.group(1)) if m else 1


def sized(row):
    return tp(row) > 1 or model(row) in learned() or any(row.startswith(k + "__") for k in HIDDEN_X_LAYERS)


def estimate_gb(row):
    if tp(row) > 1:
        return 0.0
    rate = learned().get(model(row))
    if rate:
        return max(10.0, rate * max(batch(row), 1) * tokens(row) * LEARN_MARGIN)
    size = next((v for k, v in HIDDEN_X_LAYERS.items() if row.startswith(k + "__")), REF) / REF
    return max(10.0, 120 * max(batch(row), 1) / 8 * tokens(row) / 1152 * size)


def kept(cov, row):
    b = batch(row)
    return cov in KEEP or b >= 64 or (b >= 16 and row.startswith(BIG_MODELS)) \
        or (b >= 16 and row.startswith("gemma2-") and "__i1024__" in row)


GPU_QUEUE = "deployments-gpu"
GATE = ROOT / "gpu-held"


def gate(close, projected, limit, probing=()):
    """Kueue admits a leased Commit (it requests no GPU) the moment it arrives, before a tick can deactivate it, so the pacer holds
    the GPU_QUEUE LocalQueue itself (stopPolicy Hold: nothing new is admitted from it, nothing running is touched) while `close`,
    and lets it go otherwise. tick() closes it while the admitted Commits' projection plus the next Commit's (the next released
    or waiting one, else the largest admitted, standing for the next arrival) would pass the limit, or the disk is at
    DISK_PAUSE_PCT, or a Commit of a model with no size (`probing`, sized()) is admitted: the next arrival may be another of
    that model. With no Commit admitted it always opens (outside the 78% latch), so a big one can't starve.
    The hold gates Commits only, by construction: the lease pools' GPU holders are submitted to their own LocalQueue
    (deployments-gpu-pool, n1_lease.py's VY_POOL_LOCAL_QUEUE), and the quiet hour and the disk guard hold the ClusterQueue, a
    different object. It only undoes its own hold and never touches a HoldAndDrain. `projected` is for the log."""
    policy = k("get", "localqueue", GPU_QUEUE, "-n", "default", "-o", "jsonpath={.spec.stopPolicy}").strip() or "None"
    if close:
        if policy == "None":
            why = (f"a Commit of {', '.join(probing)} (no size yet) is admitted" if probing else
                   f"projected {projected:.0f} GB with the next Commit is over the {limit} GB limit (or the disk at {DISK_PAUSE_PCT}%)")
            log(f"{why}: holding the {GPU_QUEUE} LocalQueue, so Kueue admits no new Commit")
            if not DRY:
                k("patch", "localqueue", GPU_QUEUE, "-n", "default", "--type=merge", "-p", '{"spec":{"stopPolicy":"Hold"}}')
                GATE.parent.mkdir(parents=True, exist_ok=True)
                GATE.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
        return
    if not GATE.exists():
        return
    if policy == "Hold":
        log(f"projected {projected:.0f} GB with the next Commit fits under the {limit} GB limit: releasing the {GPU_QUEUE} LocalQueue")
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
    released = {w["name"] for w in ws if w["name"] in released and w["active"]}
    seen = (seen & live) | {w["name"] for w in ws if not w["active"]}
    for w in ws:
        if w["active"] and not w["admitted"] and w["name"] in seen:
            released.add(w["name"])
        elif w["active"] and not w["admitted"] and w["name"] not in released:
            log(f"hold {w['cov']} ({w['row'][:40]}) {w['name']}")
            if not DRY:
                k("patch", "workload", w["name"], "--type=merge", "-p", '{"spec":{"active":false}}')
            w["active"] = False
    inflight = len(released | {w["name"] for w in ws if w["admitted"]})
    big = sum(1 for w in ws if (w["admitted"] or w["name"] in released) and batch(w["row"]) >= 8)
    gb = bundles_gb()
    disk = disk_pct()
    if OLD_LATCH.exists() and not DRY:
        log(f"{OLD_LATCH} is the latch under its older name: renamed {LIMIT_LATCH}")
        OLD_LATCH.replace(LIMIT_LATCH)
    if disk >= LIMIT_DROP_PCT and not (LIMIT_LATCH.exists() or OLD_LATCH.exists()):
        log(f"disk {disk:.1f}% reached {LIMIT_DROP_PCT}%: the bundle limit drops to {LIMIT_LOW_GB} GB until {LIMIT_LATCH} is removed")
        if not DRY:
            LIMIT_LATCH.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} disk {disk:.1f}%\n")
    latched = LIMIT_LATCH.exists() or OLD_LATCH.exists() or disk >= LIMIT_DROP_PCT
    limit = LIMIT_LOW_GB if latched else room_limit(gb)
    flying = [w for w in ws if w["admitted"] or w["name"] in released]
    written = {w["name"]: written_gb(w) for w in flying}
    learn(flying, written, jobs)
    rest = {w["name"]: max(estimate_gb(w["row"]) - written[w["name"]], 0) for w in flying}
    projected = gb + sum(rest.values())
    probing = {model(w["row"]) for w in flying if not sized(w["row"])}
    order = {c: i for i, c in enumerate(WAVES)}
    cands = sorted((w for w in ws if not w["active"] and not w["admitted"] and not kept(w["cov"], w["row"])),
                   key=lambda w: (order.get(w["cov"], len(WAVES)), batch(w["row"])))
    for w in cands:
        if inflight >= limits()[0] or disk >= DISK_PAUSE_PCT or NO_RELEASE.exists():
            break
        fits = projected + estimate_gb(w["row"]) <= limit or (inflight == 0 and not latched)
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
    room = (not admitted and not latched) or base + reserve <= limit
    probe = sorted({model(w["row"]) for w in admitted if not sized(w["row"])})
    gate(not room or disk >= DISK_PAUSE_PCT or bool(probe), base + reserve, limit, probe)
    if not DRY:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text("\n".join(sorted(released)) + "\n")
        SEEN.write_text("\n".join(sorted(seen)) + "\n")
    held = [w["cov"] for w in ws if not w["active"] and not w["admitted"]]
    summary = (f"tick: {'NO-RELEASE, ' if NO_RELEASE.exists() else ''}limit {limit} GB, disk {disk:.0f}%{' PAUSED' if disk >= DISK_PAUSE_PCT else ''}, in flight {inflight}, batch-8+ {big}, bundles {gb:.0f} GB, projected {projected:.0f} GB, waiting {len(held)}: {' '.join(held)}")
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
