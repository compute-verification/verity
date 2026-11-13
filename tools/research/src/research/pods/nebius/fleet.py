#!/usr/bin/env python3
"""The fleet's usage record, daily compute report and quota watch: every Nebius VM recorded, the H200s included.

    fleet.py sample              one pass: each RUNNING VM's state from the API, and over ssh its GPUs' utilization, its load
                                 and memory; one JSON line per VM in $FLEET_DIR/samples-YYYYMMDD.jsonl
    fleet.py report [--hours 24] [--now ISO]   per VM over the window: $ (rate x hours seen RUNNING), GPU busy %, CPU load, peak
                                 memory, idle hours (no GPU at >= 5%, load under a tenth of its cores and memory in use
                                 under 15%)
    fleet.py instances           every VM in the tenant's projects as JSON (the control pod's cloud watch reads it)
    fleet.py quota               each tenant quota at >= 80% of its limit, and the card's day so far against $FLEET_CARD_DAILY_USD
    fleet.py panels [--dry-run]  the usage record as the console's compute tabs read it: compute/online (the last pass's VMs),
                                 compute/hourly (each VM's $ and busy $ per UTC hour, 7 days) and compute/daily (per Pacific day,
                                 30 days), PUT to the site with $FLEET_PANELS_KEY; the loop sends them after each sample, and
                                 puts each UTC day's samples file in the store once the day is over (archive)
    fleet.py report --lanes      what the daily post rings each owning lane with: its own machines' idle hours and idle $
    fleet.py loop                sample every 5 min, quota every 30 (an alert in #agent-alerts once per quota and UTC day, kept in
                                 $FLEET_DIR/quota-told-YYYY-MM-DD.json across restarts), and at 15:00Z (08:00 PDT) the last
                                 24 h's report to infra's status thread and the admin lane, and to each lane that owns an idle
                                 machine a doorbell with only its own machines

Vy-nebius-1 sat ~0% GPU-busy for a week and the H200s had no usage record at all, and the card's daily limit and the tenant's
quotas stopped launches with no warning. A VM's rate and address come from the machines registry (`research
pods register`, <notes>/machines.d); one not registered is recorded from the API with rate None. A VM that can't be read is
recorded as RUNNING and unread, never idle. Runs on infra's VM, which holds the Slack credential (the control pod holds none).
Nebius calls go through pods/nebius/common.sh's `nebius_auth`, the way launch.sh makes them.

A RUNNING VM that no sample has read for `UNREACH_S` (a soft lockup, a full disk, sshd refusing) is alerted on once in
#agent-alerts with its serial console's tail, any kernel trouble in it named, and is marked `unreachable_since` in its registry
entry until a sample reads it again.

The lane whose budget line pays for a node stops it when its work there ends; infra's idle stop is only a backstop
(the lane contract, §3c). So the daily post rings each machine's owning lane (its registry entry's `lane`, infra when
unset or unregistered) with its own idle hours. A sample also copies each VM's /etc/research/deadline into its registry entry's
`deadline` when it changed (one notes sync per pass), which the control pod's wake (wake.py) reads while the VM is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

try:
    from research.pods.nebius import settings
except ModuleNotFoundError:     # run by path with a bare python3.12 (cloud_watch.toml's instances_command): this file's own tree
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from research.pods.nebius import settings

HERE = Path(__file__).resolve().parent
FLEET_DIR = Path(os.environ.get("FLEET_DIR", str(Path.home() / ".research" / "fleet")))
MACHINES_D = Path(os.environ.get("FLEET_MACHINES_D", str(Path.home() / ".research" / "notes" / "machines.d")))
TENANT = os.environ.get("FLEET_TENANT", "tenant-e00g3zymmem3epgq3a")
SSH_KEY = os.environ.get("FLEET_SSH_KEY", str(Path.home() / ".ssh" / "research_key"))
JUMP = os.environ.get("FLEET_JUMP", "research@81.85.2.165")        # a private address (10.*) is reached through it
INTERVAL_S, QUOTA_EVERY_S, REPORT_HOUR_UTC = 300, 1800, 15
BUSY_PCT, IDLE_LOAD_SHARE, QUOTA_ALERT = 5, 0.1, 0.8
# a memory-bound job (a few busy cores of 192 and hundreds of GB of memory in use) is not idle
IDLE_MEM_SHARE = 0.15
STATUS_THREAD = "1791351683.780959"
#: a RUNNING VM with a registered host that no sample has read this long is unreachable: one alert with its serial log's tail,
#: and its registry entry's `unreachable_since` until a sample reads it again
UNREACH_S, SERIAL_LINES = 600, 40
KERNEL_TROUBLE = re.compile(r"soft lockup|hung_task|blocked for more than|out of memory|oom-kill|watchdog: BUG|"
                            r"rcu_sched self-detected|kernel panic", re.I)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: the usage record as the console's compute tabs read it: three table panels PUT to the site with its panels key
PANELS_URL, PANELS_KEY = settings.PANELS_URL, settings.PANELS_KEY
PANEL_ONLINE, PANEL_HOURLY, PANEL_DAILY = "compute/online", "compute/hourly", "compute/daily"
HOURLY_DAYS, DAILY_DAYS = 7, 30
PACIFIC = ZoneInfo("America/Los_Angeles")
#: the site refuses a body over this (its MAX_BODY_BYTES): a panel that doesn't fit drops its oldest slots and says so
PANEL_MAX_BYTES = 64 * 1024
PANELS_NOTE = "Nebius VMs only: RunPod pods aren't sampled."
#: a platform's GPU, for the machine class the usage tab groups by; any other `gpu-` platform is named from its own name
GPU_CLASS = {"gpu-h200-sxm": "H200", "gpu-h100-sxm": "H100", "gpu-b200-sxm": "B200", "gpu-l40s-a": "L40S", "gpu-l40s-d": "L40S",
             "gpu-rtx6000": "RTX PRO 6000", "gpu-rtx6000-a": "RTX PRO 6000"}
OWNER_DEFAULT = "infra"          # registry.OWNER_DEFAULT (this file runs standalone)
PROBE = ("echo gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr '\\n' ,); "
         "echo cores=$(nproc); echo load=$(cut -d' ' -f1 /proc/loadavg); "
         "echo deadline=$(cut -d' ' -f1 /etc/research/deadline 2>/dev/null); "
         "awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf \"mem=%d,%d\\n\", (t-a)/1048576, t/1048576}' /proc/meminfo")


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def nb(*args: str, timeout: float = 120) -> dict:
    """One Nebius CLI call's JSON, authenticated as launch.sh's are."""
    cmd = f'. "{HERE}/common.sh" && nebius_auth && nb ' + " ".join(f"'{a}'" for a in args)
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"nebius {' '.join(args[:3])}: {p.stderr.strip()[-300:]}")
    return json.loads(p.stdout or "{}")


def registry() -> dict[str, dict]:
    out = {}
    for f in sorted(MACHINES_D.glob("*.toml")) if MACHINES_D.is_dir() else []:
        try:
            out[f.stem] = tomllib.loads(f.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return out


def instances() -> list[dict]:
    """Every VM in the tenant's projects: name, id, state, platform, preset, region."""
    out = []
    for proj in nb("iam", "project", "list", "--parent-id", TENANT, "--page-size", "100").get("items", []):
        pid, region = proj["metadata"]["id"], (proj.get("status") or {}).get("region") or proj["metadata"].get("labels", {}).get("region")
        for i in nb("compute", "instance", "list", "--parent-id", pid, "--page-size", "200").get("items", []):
            r = (i.get("spec") or {}).get("resources") or {}
            out.append({"name": i["metadata"]["name"], "id": i["metadata"]["id"], "state": (i.get("status") or {}).get("state"),
                        "platform": r.get("platform"), "preset": r.get("preset"), "region": region})
    return out


def parse_probe(text: str) -> dict:
    kv = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    gpus = [int(x) for x in kv.get("gpu", "").split(",") if x.strip().isdigit()]
    used, total = (int(x) for x in kv.get("mem", "0,0").split(",")[:2])
    dl = kv.get("deadline", "").strip()
    return {"gpu_util": gpus, "cores": int(kv.get("cores") or 0), "load": float(kv.get("load") or 0), "mem_used_gib": used,
            "mem_total_gib": total, "deadline": int(dl) if dl.isdigit() else None}


def probe(host: str, user: str) -> dict | None:
    ssh = ["ssh", "-i", SSH_KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no", "-o",
           "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]
    if host.startswith("10."):
        ssh += ["-J", JUMP]
    try:
        p = subprocess.run([*ssh, f"{user}@{host}", PROBE], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    return parse_probe(p.stdout) if p.returncode == 0 and "cores=" in p.stdout else None


def sample(now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    reg, rows = registry(), []
    for i in instances():
        if i["state"] != "RUNNING":
            continue
        m = reg.get(i["name"], {})
        row = {"t": utc(now), "name": i["name"], "id": i["id"], "platform": i["platform"], "preset": i["preset"],
               "region": i["region"], "rate_usd_per_hour": m.get("rate_usd_per_hour"), "read": False}
        if m.get("host"):
            got = probe(m["host"], m.get("user", "research"))
            if got:
                row.update(got, read=True)
        rows.append(row)
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    with open(FLEET_DIR / f"samples-{utc(now)[:10].replace('-', '')}.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    record_deadlines(rows, reg)
    return rows


def set_key(name: str, key: str, value: str | None) -> None:
    """Set (or, with None, remove) one key of machine `name`'s registry file, keeping its other lines as they are."""
    f = MACHINES_D / f"{name}.toml"
    kept = [ln for ln in f.read_text().splitlines() if not ln.startswith(f"{key} =")]
    f.write_text("\n".join([*kept, *([f'{key} = "{value}"'] if value is not None else [])]) + "\n")


def serial_tail(instance_id: str, lines: int = SERIAL_LINES) -> list[str]:
    """The last `lines` lines of a VM's serial console (`nebius compute instance logs`), oldest first, colors stripped."""
    cmd = f'. "{HERE}/common.sh" && nebius_auth && nb compute instance logs --id \'{instance_id}\' --limit {int(lines)}'
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"nebius compute instance logs: {p.stderr.strip()[-300:]}")
    out = [ANSI.sub("", ln).rstrip() for ln in p.stdout.splitlines() if ln.strip() and not ln.startswith("Total:")]
    return out[::-1]


def watch_unreachable(rows: list[dict], now: float, send: Callable[[list[str], str], None],
                      tail: Callable[[str], list[str]] = serial_tail) -> list[str]:
    """Alert once on each RUNNING VM no sample has read for `UNREACH_S` (with its serial log's tail and any kernel trouble in it),
    and mark it `unreachable_since` in the registry; clear the mark once it is read again. Only a VM some sample has read counts,
    so a machine whose ssh is closed to this key (an auditor) never alerts. The machines alerted on this pass."""
    path, reg = FLEET_DIR / "unreachable.json", registry()
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as e:
        print(f"{utc(now)} {path} unreadable, starting it afresh: {e}", file=sys.stderr, flush=True)
        state = {}
    changed, alerted = [], []
    answered = set(state.pop("_answered", []))          # VMs a sample has read: the auditors' admin-only ssh never is
    answered |= {r["name"] for r in rows if r.get("read")}
    running = {r["name"] for r in rows}
    for name in [n for n in state if n not in running]:         # stopped: no longer this watch's to judge
        state.pop(name)
        if reg.get(name, {}).get("unreachable_since"):
            set_key(name, "unreachable_since", None)
            changed.append(name)
    for r in rows:
        name, m = r["name"], reg.get(r["name"], {})
        if r.get("read") or not m.get("host") or name not in answered:
            state.pop(name, None)
            if r.get("read") and m.get("unreachable_since"):
                set_key(name, "unreachable_since", None)
                changed.append(name)
            continue
        s = state.setdefault(name, {"since": now, "told": False})
        if s["told"] or now - s["since"] < UNREACH_S:
            continue
        s["told"] = True
        try:
            lines = tail(r["id"])
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            lines = [f"(its serial log couldn't be read: {e})"]
        trouble = [ln for ln in lines if KERNEL_TROUBLE.search(ln)]
        if name in reg:
            set_key(name, "unreachable_since", utc(s["since"]))
            changed.append(name)
        send(["--channel", "agent-alerts", "--kind", "alert"],
             f"{name} is RUNNING in Nebius but hasn't answered ssh since {utc(s['since'])} ({int((now - s['since']) // 60)} min); "
             "it is marked `unreachable_since` in its registry entry until a sample reads it again. "
             + (f"Kernel trouble in its serial log: {' | '.join(trouble[-3:])[:600]}. " if trouble else "")
             + f"If it stays down, restart it (`nb compute instance stop` then `start --id {r['id']}`; fix a passed "
             f"/etc/research/deadline first, or the boot stops it again). Serial log tail:\n```\n"
             + "\n".join(lines[-15:])[-2500:] + "\n```\n(fleet.py)")
        alerted.append(name)
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**state, "_answered": sorted(answered)}) + "\n")
    if changed:
        sync_registry(f"machines.d: {', '.join(changed)}'s unreachable_since (fleet.py)")
    return alerted


def record_deadlines(rows: list[dict], reg: dict[str, dict]) -> list[str]:
    """Copy each read VM's /etc/research/deadline into its registry file's `deadline` where it differs; one notes sync for all."""
    changed = []
    for r in rows:
        if r.get("deadline") is None or r["name"] not in reg or reg[r["name"]].get("deadline") == utc(r["deadline"]):
            continue
        set_key(r["name"], "deadline", utc(r["deadline"]))
        changed.append(r["name"])
    if changed:
        sync_registry(f"machines.d: {', '.join(changed)}'s deadline (fleet.py)")
    return changed


def sync_registry(message: str) -> None:
    research = Path(os.environ.get("FLEET_RESEARCH", "/workspace/.venv/bin/research"))
    subprocess.run([str(research), "notes", "sync", "--root", str(MACHINES_D.parent), "--path", MACHINES_D.name, "-m", message],
                   timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_samples(start: float, end: float) -> list[dict]:
    rows, day = [], datetime.fromtimestamp(start, timezone.utc).date()
    while day <= datetime.fromtimestamp(end, timezone.utc).date():
        f = FLEET_DIR / f"samples-{day.strftime('%Y%m%d')}.jsonl"
        for line in f.read_text().splitlines() if f.exists() else []:
            r = json.loads(line)
            t = datetime.strptime(r["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
            if start <= t < end:
                rows.append(r)
        day += timedelta(days=1)
    return rows


def report(rows: list[dict], hours: float) -> list[dict]:
    """Per VM: hours seen RUNNING (samples x INTERVAL_S), $ at its rate, GPU busy share, mean load per core, peak memory, idle
    hours; most expensive first."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["name"], []).append(r)
    out = []
    for name, rs in by.items():
        seen_h = len(rs) * INTERVAL_S / 3600
        read = [r for r in rs if r.get("read")]
        gpu = [u for r in read for u in r.get("gpu_util") or []]
        idle = [r for r in read if all(u < BUSY_PCT for u in r.get("gpu_util") or [])
                and r.get("load", 0) < IDLE_LOAD_SHARE * max(r.get("cores") or 1, 1)
                and r.get("mem_used_gib", 0) < IDLE_MEM_SHARE * max(r.get("mem_total_gib") or 1, 1)]
        rate = rs[-1].get("rate_usd_per_hour")
        out.append({"name": name, "preset": rs[-1].get("preset"), "hours": round(seen_h, 1),
                    "usd": round(rate * seen_h, 2) if rate is not None else None, "rate_usd_per_hour": rate,
                    "gpus": len(read[-1]["gpu_util"]) if read and read[-1].get("gpu_util") else 0,
                    "gpu_busy_pct": round(100 * sum(u >= BUSY_PCT for u in gpu) / len(gpu), 1) if gpu else None,
                    "load_per_core": round(sum(r["load"] / max(r["cores"], 1) for r in read) / len(read), 2) if read else None,
                    "mem_peak_pct": round(max(100 * r["mem_used_gib"] / max(r["mem_total_gib"], 1) for r in read)) if read else None,
                    "idle_hours": round(len(idle) * INTERVAL_S / 3600, 1), "unread_samples": len(rs) - len(read)})
    return sorted(out, key=lambda r: -(r["usd"] or 0))


def totals(rep: list[dict]) -> tuple[float, float]:
    """A report's dollars, and the part of them spent on idle hours; a VM with no rate registered counts for nothing."""
    return sum(r["usd"] or 0 for r in rep), sum((r["rate_usd_per_hour"] or 0) * r["idle_hours"] for r in rep)


def render(rep: list[dict], hours: float, end: float) -> str:
    total, idle_usd = totals(rep)
    lines = [f"Compute, the {hours:g} h to {utc(end)}: ${total:,.0f} across {len(rep)} VMs; ${idle_usd:,.0f} of it on idle hours "
             f"(no GPU at {BUSY_PCT}%+, load under {IDLE_LOAD_SHARE:.0%} of cores, memory in use under {IDLE_MEM_SHARE:.0%})."]
    for r in rep:
        gpu = f"{r['gpus']} GPU, {r['gpu_busy_pct']}% busy" if r["gpus"] else "no GPU"
        usd = f"${r['usd']:,.0f}" if r["usd"] is not None else "$? (no rate registered)"
        lines.append(f"- {r['name']} ({r['preset']}): {usd} over {r['hours']} h; {gpu}; load {r['load_per_core']}/core; memory peak "
                     f"{r['mem_peak_pct']}%; idle {r['idle_hours']} h" + (f"; {r['unread_samples']} samples unread" if r["unread_samples"] else ""))
    return "\n".join(lines)


def idle_by_lane(rep: list[dict], reg: dict[str, dict]) -> dict[str, list[dict]]:
    """Each owning lane's machines that sat idle in the report: the registry entry's `lane`, else infra's."""
    out: dict[str, list[dict]] = {}
    for r in rep:
        if r["idle_hours"] > 0:
            out.setdefault(reg.get(r["name"], {}).get("lane") or OWNER_DEFAULT, []).append(r)
    return out


#: the most machines a post names; `research msg` refuses a post in #agent-coordination past twice POST_WORDS (60) words
POST_NAMES = 4


def _named(rows: list[dict], each: Callable[[dict], str]) -> str:
    shown = [each(r) for r in rows[:POST_NAMES]]
    return ", ".join(shown) + (f" and {len(rows) - POST_NAMES} more" if len(rows) > POST_NAMES else "")


def _idle_usd(r: dict) -> str:
    return f"${r['rate_usd_per_hour'] * r['idle_hours']:,.0f}" if r["rate_usd_per_hour"] is not None else "$?"


def _idle_entry(r: dict) -> str:
    return f"{r['name']} {r['idle_hours']} h ({_idle_usd(r)})"


def brief(rep: list[dict], hours: float, end: float) -> str:
    """The daily post: the window's spend, its idle part and the idlest machines, short enough for `research msg` (the whole report
    is `fleet.py report`, and each machine by hour the console's Usage tab)."""
    total, idle_usd = totals(rep)
    idlest = sorted((r for r in rep if r["idle_hours"] > 0), key=lambda r: -(r["rate_usd_per_hour"] or 0) * r["idle_hours"])
    return (f"Compute, the {hours:g} h to {utc(end)}: ${total:,.0f} across {len(rep)} VMs, ${idle_usd:,.0f} of it idle."
            + (f" Idlest: {_named(idlest, _idle_entry)}." if idlest else "")
            + " Each machine by hour: the console's Usage tab.")


def render_lane(rows: list[dict], hours: float, end: float) -> str:
    """A lane's doorbell: its own machines' idle hours and idle $, short enough for `research msg`."""
    known = [r for r in rows if r["rate_usd_per_hour"] is not None]
    usd = sum(r["rate_usd_per_hour"] * r["idle_hours"] for r in known)
    return (f"Your machines sat idle {sum(r['idle_hours'] for r in rows):.1f} h (${usd:,.0f}"
            + ("" if len(known) == len(rows) else ", plus machines with no rate registered")
            + f") in the {hours:g} h to {utc(end)}: "
            + _named(rows, lambda r: f"{r['name']} {r['idle_hours']} h of {r['hours']} h ({_idle_usd(r)})")
            + ". Stop a node when its work there ends (lane contract §3c).")


def daily(now: float, send: Callable[[list[str], str], str | None] | None = None) -> list[str]:
    """The 24 h brief to infra's status thread and the admin lane, and each owning lane's idle machines to its inbox; a post that fails is
    alerted on in #agent-alerts (once, naming every failure) and logged. What failed."""
    send = send or post
    rep = report(load_samples(now - 86400, now), 24)
    text = brief(rep, 24, now)
    posts = [(["--thread", STATUS_THREAD, "--kind", "fyi"], text), (["--to", "@top", "--kind", "fyi"], text)]
    # posts go out as infra, and `research msg` refuses a message to the sender's own inbox: infra's machines go to its thread
    posts += [(["--thread", STATUS_THREAD, "--kind", "fyi"] if lane == "infra" else ["--to", f"@{lane}", "--kind", "fyi"],
               render_lane(rows, 24, now))
              for lane, rows in sorted(idle_by_lane(rep, registry()).items())]
    failed = [f"{' '.join(args[:2])}: {err}" for args, body in posts if (err := send(args, body))]
    if failed:
        print(f"{utc(now)} daily post failed: {'; '.join(failed)}", file=sys.stderr, flush=True)
        send(["--channel", "agent-alerts", "--kind", "alert"],
             f"fleet.py's daily compute post failed {len(failed)} of {len(posts)} times: {'; '.join(failed)}"[:1500])
    return failed


def quota_findings() -> list[dict]:
    out = []
    for q in nb("quotas", "quota-allowance", "list", "--parent-id", TENANT, "--page-size", "999").get("items", []):
        spec, st = q.get("spec") or {}, q.get("status") or {}
        limit, usage = int(spec.get("limit") or 0), int(st.get("usage") or 0)
        if limit > 0 and usage >= QUOTA_ALERT * limit:
            out.append({"quota": q["metadata"].get("name"), "region": spec.get("region"), "usage": usage, "limit": limit,
                        "share": round(usage / limit, 2)})
    return out


def card_finding(rows_today: list[dict]) -> dict | None:
    """The card's UTC day so far (rate x hours seen) against $FLEET_CARD_DAILY_USD, when set, at >= 80%."""
    cap = float(os.environ.get("FLEET_CARD_DAILY_USD") or 0)
    spent = sum((r.get("rate_usd_per_hour") or 0) * INTERVAL_S / 3600 for r in rows_today)
    return {"spent": round(spent, 2), "cap": cap} if cap and spent >= QUOTA_ALERT * cap else None


def machine_class(platform: str | None) -> str:
    p = platform or ""
    if not p.startswith("gpu-"):
        return "CPU"
    return GPU_CLASS.get(p) or p[len("gpu-"):].split("-")[0].upper()


def hardware(row: dict) -> str:
    """What a VM is, from its preset: "8× H200", or "192 CPUs" for a machine with no GPU."""
    preset = row.get("preset") or ""
    if m := re.match(r"(\d+)gpu", preset):
        return f"{m.group(1)}× {machine_class(row.get('platform'))}"
    cores = (m.group(1) if (m := re.search(r"(\d+)vcpu", preset)) else None) or row.get("cores")
    return f"{cores} CPUs" if cores else preset or "?"


def busy(row: dict) -> tuple[float | None, str]:
    """A sample's busy share and what it is a share of: its GPUs' mean utilization, or for a machine with no GPU its load over its
    cores (at most 1). None when the sample wasn't read."""
    gpu = machine_class(row.get("platform")) != "CPU"
    if not row.get("read"):
        return None, "gpu" if gpu else "cores"
    if gpu:
        util = row.get("gpu_util") or []
        return (round(sum(util) / len(util) / 100, 4) if util else None), "gpu"
    return round(min(row.get("load", 0) / max(row.get("cores") or 1, 1), 1.0), 4), "cores"


def _parse(t: str) -> datetime:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _panel(pid: str, title: str, columns: list[str], rows: list[list], now: float, note: str) -> dict:
    return {"id": pid, "title": title, "kind": "table", "note": note, "updated_at": utc(now),
            "source": "tools/research/src/research/pods/nebius/fleet.py on the control pod", "columns": columns, "rows": rows}


def online_panel(rows: list[dict], reg: dict[str, dict], now: float) -> dict:
    """One row per RUNNING VM in the last sample pass; one with no rate registered is left out and named in the note."""
    out, unpriced = [], []
    for r in rows:
        m = reg.get(r["name"], {})
        if r.get("rate_usd_per_hour") is None:
            unpriced.append(r["name"])
            continue
        share, of = busy(r)
        stops = utc(r["deadline"]) if r.get("deadline") else m.get("deadline")
        out.append([r["name"], machine_class(r.get("platform")), hardware(r), "Nebius", m.get("lane") or OWNER_DEFAULT,
                    m.get("holder"), r["rate_usd_per_hour"], share, of, stops, r["t"]])
    note = PANELS_NOTE + (f" No rate registered, so left out: {', '.join(sorted(unpriced))}." if unpriced else "")
    return _panel(PANEL_ONLINE, "Machines online", ["machine", "class", "hardware", "provider", "lane", "holder", "usd_per_hour",
                                                     "busy", "busy_of", "stops_at", "sampled_at"], out, now, note[:500])


def bucket_panels(samples: list[dict], reg: dict[str, dict], now: float) -> tuple[dict, dict]:
    """Each VM's cost per UTC hour over `HOURLY_DAYS` and per Pacific day over `DAILY_DAYS`: `usd` (each sample's interval at its
    rate), `busy_usd` (each sample's cost times its busy share; an unread sample takes its slot's mean share of read ones, else
    none) and `minutes` sampled, so a gap in sampling shows as fewer minutes rather than as a cheaper slot."""
    hour_from = now - HOURLY_DAYS * 86400
    slots: dict[tuple[str, str, str], dict] = {}
    for r in samples:
        if r.get("rate_usd_per_hour") is None:
            continue
        t = _parse(r["t"])
        cost = r["rate_usd_per_hour"] * INTERVAL_S / 3600
        share = busy(r)[0]
        keys = [("day", t.astimezone(PACIFIC).strftime("%Y-%m-%d"), r["name"])]
        if t.timestamp() >= hour_from:
            keys.append(("hour", t.replace(minute=0, second=0).strftime("%Y-%m-%dT%H:%M:%SZ"), r["name"]))
        for k in keys:
            s = slots.setdefault(k, {"usd": 0.0, "read_usd": 0.0, "busy_usd": 0.0, "n": 0, "platform": r.get("platform")})
            s["usd"] += cost
            s["n"] += 1
            if share is not None:
                s["read_usd"] += cost
                s["busy_usd"] += cost * share

    def row(slot: str, name: str, s: dict) -> list:
        busy_usd = s["busy_usd"] * s["usd"] / s["read_usd"] if s["read_usd"] else 0.0
        return [slot, name, machine_class(s["platform"]), reg.get(name, {}).get("lane") or OWNER_DEFAULT, round(s["usd"], 4),
                round(min(busy_usd, s["usd"]), 4), s["n"] * INTERVAL_S // 60]
    hourly = sorted(row(slot, name, s) for (kind, slot, name), s in slots.items() if kind == "hour")
    first_day = (datetime.fromtimestamp(now, PACIFIC) - timedelta(days=DAILY_DAYS - 1)).strftime("%Y-%m-%d")
    daily = sorted(row(slot, name, s) for (kind, slot, name), s in slots.items() if kind == "day" and slot >= first_day)
    cols = ["machine", "class", "lane", "usd", "busy_usd", "minutes"]
    return (_panel(PANEL_HOURLY, "Compute by machine and hour", ["hour", *cols], hourly, now, PANELS_NOTE),
            _panel(PANEL_DAILY, "Compute by machine and Pacific day", ["day", *cols], daily, now, PANELS_NOTE))


def fitted(panel: dict) -> dict:
    """``panel`` within `PANEL_MAX_BYTES`: its oldest slots (the first column) dropped until it fits, its note saying from when."""
    rows, size = panel["rows"], lambda p: len(json.dumps(p).encode())
    if size(panel) <= PANEL_MAX_BYTES:
        return panel
    slots = sorted({r[0] for r in rows})
    for i in range(1, len(slots)):
        p = {**panel, "rows": [r for r in rows if r[0] >= slots[i]],
             "note": f"From {slots[i]} only: the site takes at most {PANEL_MAX_BYTES // 1024} KiB. {panel['note']}"[:500]}
        if size(p) <= PANEL_MAX_BYTES:
            return p
    return {**panel, "rows": []}


def put_panel(panel: dict, key: str) -> str | None:
    """PUT one panel; None when the site took it, else why not. Only an https site is sent the key."""
    if not (PANELS_URL or "").startswith("https://"):
        return f"{panel['id']}: $FLEET_PANELS_URL is {PANELS_URL!r}, not an https URL: nothing sent"
    url = f"{PANELS_URL}/api/panels/{panel['id']}"
    req = urllib.request.Request(url, data=json.dumps(panel).encode(), method="PUT",  # noqa: S310 (https, checked above)
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30):  # noqa: S310 (https, checked above)
            return None
    except urllib.error.HTTPError as e:
        return f"{panel['id']}: HTTP {e.code} {e.read()[:300].decode(errors='replace')}"
    except (urllib.error.URLError, OSError) as e:
        return f"{panel['id']}: {e}"


def publish(rows: list[dict], now: float, put: Callable[[dict, str], str | None] = put_panel) -> list[str]:
    """The three usage panels from this pass's ``rows`` and the stored samples, PUT to the site; what failed. Nothing is sent
    without a panels key."""
    if not PANELS_KEY.exists():
        return [f"no panels key at {PANELS_KEY}"]
    key, reg = PANELS_KEY.read_text().strip(), registry()
    hourly, daily = bucket_panels(load_samples(now - (DAILY_DAYS + 1) * 86400, now + 1), reg, now)
    return [e for p in (online_panel(rows, reg, now), hourly, daily) if (e := put(fitted(p), key))]


def store_put(path: Path) -> tuple[str | None, str]:
    """``path`` into the store (`research data put --kind fleet-samples/v1 --preserve`): its art id, or None and why. The store is
    $RESEARCH_STORE, else ~/fleet-store, and its remote's credentials are $FLEET_STORE_ENV's KEY=VALUE lines (the pod's r2.env),
    which the loop's own environment doesn't hold."""
    research = settings.research_bin()
    env = {**os.environ, "RESEARCH_STORE": settings.store_root()}
    creds = settings.store_env()
    for line in creds.read_text().splitlines() if creds.is_file() else []:
        k, sep, v = line.removeprefix("export ").partition("=")
        if sep and k.strip() and not line.lstrip().startswith("#"):
            env[k.strip()] = v.strip().strip('"').strip("'")
    try:
        p = subprocess.run([str(research), "data", "put", "--kind", "fleet-samples/v1", "--file", str(path), "--preserve", "--json"],
                           capture_output=True, text=True, timeout=1800, env=env)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, f"{type(e).__name__}: {e}"
    if p.returncode != 0:
        return None, f"exit {p.returncode}: {p.stderr.strip()[-300:]}"
    try:
        return json.loads(p.stdout)["id"], ""
    except (ValueError, KeyError, TypeError):
        return None, f"no art id in its output: {p.stdout.strip()[-200:]}"


def archive(now: float, put: Callable[[Path], tuple[str | None, str]] = store_put) -> str | None:
    """The second copy of the usage record: the last UTC day's samples file put into the store once, its art id kept in
    $FLEET_DIR/archived.json so a restarted loop doesn't put it again. None when it is stored (or there is no file), else why not."""
    day = utc(now - 86400)[:10].replace("-", "")
    f, state_path = FLEET_DIR / f"samples-{day}.jsonl", FLEET_DIR / "archived.json"
    try:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except (OSError, ValueError) as e:
        return f"{state_path.name} unreadable ({type(e).__name__}: {e}): nothing archived until it is fixed"
    if day in state or not f.exists():
        return None
    art, why = put(f)
    if art is None:
        return f"{f.name}: {why}"
    state[day] = art
    tmp = state_path.with_name(f".{state_path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, state_path)
    return None


def post(args: list[str], text: str) -> str | None:
    """Send one post with `research msg send`; None when it went out, else why not (its exit and the end of its stderr)."""
    research = settings.research_bin()
    try:
        p = subprocess.run([str(research), "msg", "send", "--as", "infra", *args, "--text", text], cwd="/workspace", timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{type(e).__name__}: {e}"
    return None if p.returncode == 0 else f"exit {p.returncode}: {p.stderr.strip()[-300:]}"


TOLD_KEEP_DAYS = 2


def load_told(now: float) -> set[tuple[str, str]]:
    """The (UTC day, quota key) alerts already posted, from $FLEET_DIR/quota-told-YYYY-MM-DD.json, so a restarted loop doesn't post
    them again; a day's file older than TOLD_KEEP_DAYS is removed."""
    told, oldest = set(), utc(now - TOLD_KEEP_DAYS * 86400)[:10]
    for f in sorted(FLEET_DIR.glob("quota-told-*.json")):
        day = f.name[len("quota-told-"):-len(".json")]
        if day < oldest:
            f.unlink(missing_ok=True)
            continue
        try:
            told |= {(day, k) for k in json.loads(f.read_text())}
        except (OSError, ValueError) as e:
            print(f"{utc(now)} {f} unreadable, its alerts may post again: {e}", file=sys.stderr, flush=True)
    return told


def save_told(told: set[tuple[str, str]], day: str) -> None:
    FLEET_DIR.mkdir(parents=True, exist_ok=True)
    f = FLEET_DIR / f"quota-told-{day}.json"
    tmp = f.with_name(f".{f.name}.{os.getpid()}")
    tmp.write_text(json.dumps(sorted(k for d, k in told if d == day)) + "\n")
    os.replace(tmp, f)


def loop(clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
         send: Callable[[list[str], str], None] | None = None, passes: int | None = None) -> int:
    send = send or post
    last_quota, told, reported, archive_told = 0.0, load_told(clock()), "", ""
    while passes is None or passes > 0:
        passes = None if passes is None else passes - 1
        now, rows = clock(), None
        try:
            rows = sample(now)
            watch_unreachable(rows, now, send)
        except Exception as e:  # noqa: BLE001 (a failed pass is logged, the loop goes on)
            print(f"{utc(now)} sample failed: {e}", file=sys.stderr, flush=True)
        if rows is not None:
            try:
                for e in publish(rows, now):
                    print(f"{utc(now)} panels: {e}", file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{utc(now)} panels failed: {e}", file=sys.stderr, flush=True)
        day = utc(now)[:10]
        if now - last_quota >= QUOTA_EVERY_S:
            last_quota = now
            try:
                found = [(f"{q['quota']}@{q['region']}", f"{q['quota']} in {q['region']}: {q['usage']} of {q['limit']} "
                          f"({q['share']:.0%})") for q in quota_findings()]
                start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
                if c := card_finding(load_samples(start, now)):
                    found.append(("card", f"the card's day so far: ${c['spent']:,.0f} of its ${c['cap']:,.0f} daily limit"))
                for key, line in found:
                    if (day, key) not in told:
                        told.add((day, key))
                        save_told(told, day)
                        send(["--channel", "agent-alerts", "--kind", "alert"],
                             f"Quota at {QUOTA_ALERT:.0%}+: {line}. Ask for a raise now (Nebius console, Administration > Quotas; "
                             f"the card: Daniel), before it stops a launch. (fleet.py quota)")
            except Exception as e:  # noqa: BLE001
                print(f"{utc(now)} quota failed: {e}", file=sys.stderr, flush=True)
        if datetime.fromtimestamp(now, timezone.utc).hour == REPORT_HOUR_UTC and reported != day:
            reported = day
            daily(now, send)
        try:
            if (why := archive(now)) and archive_told != day:
                archive_told = day
                print(f"{utc(now)} archive failed: {why}", file=sys.stderr, flush=True)
                send(["--channel", "agent-alerts", "--kind", "alert"],
                     f"fleet.py couldn't put yesterday's usage samples in the store (it retries each pass): {why}"[:1500])
        except Exception as e:  # noqa: BLE001
            print(f"{utc(now)} archive failed: {e}", file=sys.stderr, flush=True)
        sleep(max(1.0, INTERVAL_S - (clock() - now)))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fleet.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    rp = sub.add_parser("report")
    rp.add_argument("--hours", type=float, default=24)
    rp.add_argument("--now", default=None)
    rp.add_argument("--lanes", action="store_true", help="each owning lane's doorbell instead of the whole report")
    sub.add_parser("instances")
    sub.add_parser("quota")
    sub.add_parser("loop")
    pp = sub.add_parser("panels")
    pp.add_argument("--dry-run", action="store_true", help="print each panel's rows and size instead of sending it")
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        print(json.dumps(sample(), indent=1))
    elif a.cmd == "panels":
        now = time.time()
        latest = load_samples(now - 2 * INTERVAL_S, now + 1)
        last = max((r["t"] for r in latest), default=None)
        rows = [r for r in latest if r["t"] == last]
        if a.dry_run:
            reg = registry()
            for p in (online_panel(rows, reg, now), *bucket_panels(load_samples(now - (DAILY_DAYS + 1) * 86400, now + 1), reg, now)):
                f = fitted(p)
                print(f"{p['id']}: {len(p['rows'])} rows, {len(json.dumps(p).encode())} bytes; sent: {len(f['rows'])} rows; "
                      f"note: {f['note']}")
            return 0
        errors = publish(rows, now)
        print("\n".join(errors) or "published")
        return 1 if errors else 0
    elif a.cmd == "report":
        end = datetime.strptime(a.now, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if a.now else time.time()
        rep = report(load_samples(end - a.hours * 3600, end), a.hours)
        if a.lanes:
            for lane, rows in sorted(idle_by_lane(rep, registry()).items()):
                print(f"--to @{lane}\n{render_lane(rows, a.hours, end)}\n")
        else:
            print(render(rep, a.hours, end))
    elif a.cmd == "instances":
        print(json.dumps(instances()))
    elif a.cmd == "quota":
        print(json.dumps(quota_findings(), indent=1))
    else:
        return loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
