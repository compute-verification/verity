#!/usr/bin/env python3
"""The fleet's usage record, daily compute report and quota watch (INF-60): every Nebius VM recorded, the H200s included.

    fleet.py sample              one pass: each RUNNING VM's state from the API, and over ssh its GPUs' utilization, its load
                                 and memory; one JSON line per VM in $FLEET_DIR/samples-YYYYMMDD.jsonl
    fleet.py report [--hours 24] [--now ISO]   per VM over the window: $ (rate x hours seen RUNNING), GPU busy %, CPU load, peak
                                 memory, idle hours (no GPU at >= 5%, load under a tenth of its cores and memory in use
                                 under 15%)
    fleet.py instances           every VM in the tenant's projects as JSON (the control pod's cloud watch reads it)
    fleet.py quota               each tenant quota at >= 80% of its limit, and the card's day so far against $FLEET_CARD_DAILY_USD
    fleet.py report --lanes      what the daily post rings each owning lane with: its own machines' idle hours and idle $
    fleet.py loop                sample every 5 min, quota every 30 (an alert in #agent-alerts once per quota and UTC day, kept in
                                 $FLEET_DIR/quota-told-YYYY-MM-DD.json across restarts), and at 15:00Z (08:00 PDT) the last
                                 24 h's report to infra's status thread and @top, and to each lane that owns an idle machine a
                                 doorbell with only its own machines

Vy-nebius-1 sat ~0% GPU-busy for a week and the H200s had no usage record at all, and the card's daily limit and the tenant's
quotas stopped launches with no warning. A VM's rate and address come from the machines registry (`research
pods register`, <notes>/machines.d); one not registered is recorded from the API with rate None. A VM that can't be read is
recorded as RUNNING and unread, never idle. Runs on infra's VM, which holds the Slack credential (the control pod holds none).
Nebius calls go through pods/nebius/common.sh's `nebius_auth`, the way launch.sh makes them.

The lane whose budget line pays for a node stops it when its work there ends; infra's idle stop is only a backstop
(the lane contract, §3c). So the daily post rings each machine's owning lane (its registry entry's `lane`, infra when
unset or unregistered) with its own idle hours. A sample also copies each VM's /etc/research/deadline into its registry entry's
`deadline` when it changed (one notes sync per pass), which the control pod's wake (wake.py) reads while the VM is stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
FLEET_DIR = Path(os.environ.get("FLEET_DIR", str(Path.home() / ".research" / "fleet")))
MACHINES_D = Path(os.environ.get("FLEET_MACHINES_D", str(Path.home() / ".research" / "notes" / "machines.d")))
TENANT = os.environ.get("FLEET_TENANT", "tenant-e00g3zymmem3epgq3a")
SSH_KEY = os.environ.get("FLEET_SSH_KEY", str(Path.home() / ".ssh" / "research_key"))
JUMP = os.environ.get("FLEET_JUMP", "research@81.85.2.165")        # a private address (10.*) is reached through it
INTERVAL_S, QUOTA_EVERY_S, REPORT_HOUR_UTC = 300, 1800, 15
BUSY_PCT, IDLE_LOAD_SHARE, QUOTA_ALERT = 5, 0.1, 0.8
# a memory-bound job (circuits' GLM folds on vy-glm53-cpu-2, 25 Oct: 2-4 busy cores of 192, 150-700 GB of 755 GB) is not idle
IDLE_MEM_SHARE = 0.15
STATUS_THREAD = "1791351683.780959"
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


def record_deadlines(rows: list[dict], reg: dict[str, dict]) -> list[str]:
    """Copy each read VM's /etc/research/deadline into its registry file's `deadline` where it differs; one notes sync for all."""
    changed = []
    for r in rows:
        if r.get("deadline") is None or r["name"] not in reg or reg[r["name"]].get("deadline") == utc(r["deadline"]):
            continue
        f = MACHINES_D / f"{r['name']}.toml"
        kept = [ln for ln in f.read_text().splitlines() if not ln.startswith("deadline =")]
        f.write_text("\n".join([*kept, f'deadline = "{utc(r["deadline"])}"']) + "\n")
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


def render(rep: list[dict], hours: float, end: float) -> str:
    total = sum(r["usd"] or 0 for r in rep)
    idle_usd = sum((r["rate_usd_per_hour"] or 0) * r["idle_hours"] for r in rep)
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


def render_lane(rows: list[dict], hours: float, end: float) -> str:
    known = [r for r in rows if r["rate_usd_per_hour"] is not None]
    usd = sum(r["rate_usd_per_hour"] * r["idle_hours"] for r in known)
    lines = [f"Your machines sat idle {sum(r['idle_hours'] for r in rows):.1f} h (${usd:,.0f}"
             + ("" if len(known) == len(rows) else ", plus machines with no rate registered")
             + f") in the {hours:g} h to {utc(end)}. The lane whose budget line pays for a node stops it when its work there ends; "
             "infra's idle stop is only a backstop (lane contract v2.19 §3c)."]
    for r in rows:
        usd_r = f"${r['rate_usd_per_hour'] * r['idle_hours']:,.0f}" if r["rate_usd_per_hour"] is not None else "$? (no rate registered)"
        lines.append(f"- {r['name']}: idle {r['idle_hours']} h of {r['hours']} h running, {usd_r}")
    return "\n".join(lines)


def daily(now: float, send: Callable[[list[str], str], None] | None = None) -> None:
    """The 24 h report to infra's status thread and @top, and each owning lane's idle machines to its inbox."""
    send = send or post
    rep = report(load_samples(now - 86400, now), 24)
    text = render(rep, 24, now)
    send(["--thread", STATUS_THREAD, "--kind", "fyi"], text)
    send(["--to", "@top", "--kind", "fyi"], text)
    for lane, rows in sorted(idle_by_lane(rep, registry()).items()):
        send(["--to", f"@{lane}", "--kind", "fyi"], render_lane(rows, 24, now))


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


def post(args: list[str], text: str) -> None:
    research = Path(os.environ.get("FLEET_RESEARCH", "/workspace/.venv/bin/research"))
    subprocess.run([str(research), "msg", "send", "--as", "infra", *args, "--text", text], cwd="/workspace", timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
    last_quota, told, reported = 0.0, load_told(clock()), ""
    while passes is None or passes > 0:
        passes = None if passes is None else passes - 1
        now = clock()
        try:
            sample(now)
        except Exception as e:  # noqa: BLE001 (a failed pass is logged, the loop goes on)
            print(f"{utc(now)} sample failed: {e}", file=sys.stderr, flush=True)
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
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        print(json.dumps(sample(), indent=1))
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
