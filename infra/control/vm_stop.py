#!/usr/bin/env python3
"""The control pod's VM stops (INF-58, INF-131): the Nebius API stops for VMs that can't stop themselves, which ran on infra's VM
(h200-idle.sh, dc-backstop.sh, glm53-cpu2-backstop.sh) and stopped nothing while that VM was suspended.

    vm_stop.py loop [--config PATH] [--state PATH]     a pass every EVERY_S (the watchers supervisor runs it)
    vm_stop.py once [--config PATH] [--state PATH] [--dry-run]   one pass; --dry-run stops nothing and saves no state

Each [[vm]] of the config (vm_stop.toml, read every pass) names a machine of the registry (<notes>/machines.d/NAME.toml: its
host, user, `nebius_instance_id` and `deadline`; an entry's own `id` or `deadline` wins) and its rules:

- `deadline`: stopped once the API reads it RUNNING within DEADLINE_LEAD_S of its deadline or after, as a backstop to the VM's own
  /etc/research/deadline timer;
- `idle` (circuits' H200s, INF-58): every pass one ssh read of its GPUs' highest utilization, its live `research run`s and its
  held gpu-leases. Busy is a 2-h mean utilization of at least BUSY_PCT, or a held lease within LEASE_IDLE_S of the GPUs' last
  use (18-19 Oct: eu ran 15.8 of 18.8 h at 1.7% with 8 leases held). Idle for IDLE_S since the last busy reading: stopped. A
  CPU-only run holds no lease and doesn't count (INF-67). A VM not RUNNING or not read forgets its busy time, so one started
  again counts as busy from its first reading (18 Oct: eu was stopped the moment circuits restarted it);
- `lease` (the vy-dc-* VMs): stopped once the lease it holds (the first field of ~research/.research/lease) passed LEASE_GRACE_S
  ago.

Stops go through `nb compute instance stop --async` with the key in $VM_STOP_KEY_ENV (default /data/watchers/nebius-stop.env:
NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID, NEBIUS_SA_PRIVATE_KEY_B64 of an account that may stop instances); without that file
each stop is logged as `would stop` and nothing is stopped. The reads use the environment's key (the pod's read-only
monitoring account, monitor.env). Ssh uses $VM_STOP_SSH_KEY (default the RunPod key, which the Nebius VMs accept). Each pass
logs one line per VM; the busy times and samples are in --state.
"""
from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

EVERY_S = 300
IDLE_S = 7200
LEASE_IDLE_S = 7200
BUSY_PCT = 5
DEADLINE_LEAD_S = 300
LEASE_GRACE_S = 600
STAMP = "%Y-%m-%dT%H:%M:%SZ"
KEY_ENV = Path(os.environ.get("VM_STOP_KEY_ENV", "/data/watchers/nebius-stop.env"))
SSH_KEY = os.environ.get("VM_STOP_SSH_KEY", "/root/.runpod/ssh/runpodctl-ssh-key")
MACHINES_D = Path(os.environ.get("VM_STOP_MACHINES_D", "/data/steward/research-notes/machines.d"))
# GPUs' highest utilization, the live `research run`s (status running, runner alive) of the last 3 days, the gpu-leases held,
# and the lease this VM holds (vy-dc-*), as one line
PROBE = r'''m=0; for u in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null); do [ "$u" -gt "$m" ] && m=$u; done
r=0; for s in $(find /workspace/research/runs -maxdepth 2 -name status.json -mmin -4320 2>/dev/null); do
  python3 - "$(dirname $s)" <<'EOF' && r=$((r+1))
import json, os, sys
d = sys.argv[1]
try:
    st = json.load(open(d + "/status.json"))["transitions"][-1]["state"]
    pid = json.load(open(d + "/launch.json")).get("runner_pid")
except Exception:
    sys.exit(1)
sys.exit(0 if st == "running" and pid and os.path.exists(f"/proc/{pid}") else 1)
EOF
done; l=0; for f in /run/gpu-lease/*.lock; do [ -e "$f" ] && ! flock -n -s "$f" true 2>/dev/null && l=$((l+1)); done
echo "gpu=$m runs=$r leases=$l lease=$(cut -d' ' -f1 ~/.research/lease 2>/dev/null)"'''


def iso(t: float) -> str:
    return time.strftime(STAMP, time.gmtime(t))


def epoch(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, STAMP)))


def parse_probe(line: str) -> dict | None:
    kv = dict(f.split("=", 1) for f in line.split() if "=" in f)
    if not kv.get("gpu", "").isdigit():
        return None
    lease = kv.get("lease", "")
    return {"gpu": int(kv["gpu"]), "runs": int(kv.get("runs") or 0), "leases": int(kv.get("leases") or 0),
            "lease": float(lease) if lease.isdigit() else None}


def vms(cfg: dict, machines_d: Path) -> list[dict]:
    """Each [[vm]] with its registry entry: name, rules, id, host, user, deadline (epoch or None)."""
    out = []
    for e in cfg.get("vm", []):
        f = machines_d / f"{e['name']}.toml"
        m = tomllib.loads(f.read_text()) if f.is_file() else {}
        dl = e.get("deadline") or m.get("deadline")
        out.append({"name": e["name"], "rules": list(e["rules"]), "id": e.get("id") or m.get("nebius_instance_id"),
                    "host": e.get("host") or m.get("host"), "user": e.get("user") or m.get("user", "research"),
                    "deadline": epoch(dl) if dl else None, "idle_s": float(e.get("idle_s", IDLE_S)),
                    "lease_idle_s": float(e.get("lease_idle_s", LEASE_IDLE_S))})
    return out


def judge(vm: dict, st: dict, api: str | None, probe: dict | None, now: float) -> tuple[str | None, str]:
    """Why to stop `vm` now (None: leave it) and the pass's line for it; `st` (its busy time, GPU time and samples) is updated."""
    running = api == "RUNNING"
    line = f"{vm['name']}: {api or 'unread'}"
    mean = None
    if "idle" in vm["rules"]:
        if not running or probe is None:
            st.pop("busy", None)
            st.pop("gpu", None)
        else:
            st.setdefault("busy", now)
            st.setdefault("gpu", now)
            if probe["gpu"] >= BUSY_PCT:
                st["gpu"] = now
            st["samples"] = [s for s in st.get("samples", []) if s[0] >= now - vm["idle_s"]] + [[now, probe["gpu"]]]
            mean = sum(s[1] for s in st["samples"]) // len(st["samples"])
            if mean >= BUSY_PCT or (probe["leases"] > 0 and now - st["gpu"] < vm["lease_idle_s"]):
                st["busy"] = now
            line += (f" gpu={probe['gpu']} mean2h={mean} leases={probe['leases']} runs={probe['runs']} "
                     f"idle_s={now - st['busy']:.0f} gpu_idle_s={now - st['gpu']:.0f}")
    if probe is not None and probe.get("lease") is not None and "lease" in vm["rules"]:
        line += f" lease={iso(probe['lease'])}"
    if not running:
        return None, line
    if vm["deadline"] is not None and "deadline" in vm["rules"] and now >= vm["deadline"] - DEADLINE_LEAD_S:
        return f"deadline {iso(vm['deadline'])}", line
    if "idle" in vm["rules"] and probe is not None and now - st["busy"] >= vm["idle_s"]:
        return (f"idle {(now - st['busy']) // 60:.0f} min (gpu {probe['gpu']}%, 2-h mean {mean}%, {probe['leases']} leases, "
                f"{probe['runs']} CPU-only runs; GPUs idle {(now - st['gpu']) // 60:.0f} min)"), line
    if "lease" in vm["rules"] and probe is not None and probe.get("lease") is not None and now > probe["lease"] + LEASE_GRACE_S:
        return f"lease {iso(probe['lease'])} passed", line
    return None, line


def nebius(args: list[str], env: dict[str, str] | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """One Nebius CLI call, authenticated as launch.sh's are (pods/nebius/common.sh's nebius_auth), with `env` over ours."""
    spec = importlib.util.find_spec("research.pods")
    common = Path(spec.submodule_search_locations[0]) / "nebius" / "common.sh"
    cmd = f'. "{common}" && nebius_auth && nb ' + " ".join(f"'{a}'" for a in args)
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})


def api_state(iid: str) -> str | None:
    r = nebius(["compute", "instance", "get", "--id", iid])
    if r.returncode != 0:
        return None
    return ((json.loads(r.stdout or "{}").get("status") or {}).get("state")) or None


def ssh_probe(vm: dict, known_hosts: Path) -> dict | None:
    argv = ["ssh", "-i", SSH_KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o",
            "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known_hosts}", "-o", "LogLevel=ERROR",
            f"{vm['user']}@{vm['host']}", PROBE]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return None
    lines = r.stdout.strip().splitlines()
    return parse_probe(lines[-1]) if r.returncode == 0 and lines else None


def stop_key() -> dict[str, str] | None:
    if not KEY_ENV.is_file():
        return None
    out = {}
    for line in KEY_ENV.read_text().splitlines():
        k, sep, v = line.strip().removeprefix("export ").partition("=")
        if sep:
            out[k] = v.strip().strip("\"'")
    return out


def one_pass(cfg_path: Path, state_path: Path, dry: bool, now: float | None = None) -> None:
    now = time.time() if now is None else now
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    key = stop_key()
    for vm in vms(tomllib.loads(cfg_path.read_text()), MACHINES_D):
        if not vm["id"]:
            print(f"{iso(now)} {vm['name']}: no nebius_instance_id in the registry or the entry; skipped", flush=True)
            continue
        api = api_state(vm["id"])
        probe = (ssh_probe(vm, state_path.with_name("vm_stop_known_hosts"))
                 if api == "RUNNING" and vm["host"] and {"idle", "lease"} & set(vm["rules"]) else None)
        st = state.setdefault(vm["name"], {})
        why, line = judge(vm, st, api, probe, now)
        print(f"{iso(now)} {line}", flush=True)
        if not why:
            continue
        if dry or key is None:
            print(f"{iso(now)} {vm['name']}: {why}: would stop {vm['id']}" + ("" if dry else f" (no {KEY_ENV}: the pod's "
                  "Nebius account may only read)"), flush=True)
            continue
        r = nebius(["compute", "instance", "stop", "--id", vm["id"], "--async"], key)
        print(f"{iso(now)} {vm['name']}: {why}: stopping {vm['id']}: rc={r.returncode} "
              f"{(r.stdout + r.stderr).strip().splitlines()[-1:] or ''}", flush=True)
        if r.returncode == 0:
            st.pop("busy", None)
    if not dry:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
        tmp.replace(state_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vm_stop.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("loop", "once"))
    ap.add_argument("--config", type=Path, default=Path("/data/watchers/vm_stop.toml"))
    ap.add_argument("--state", type=Path, default=Path("/data/watchers/vm_stop.json"))
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    while True:
        t0 = time.time()
        try:
            one_pass(ns.config, ns.state, ns.dry_run, t0)
        except Exception as e:  # noqa: BLE001 (the next pass tries again)
            print(f"{iso(t0)} pass failed: {type(e).__name__}: {e}", flush=True)
        if ns.cmd == "once":
            return 0
        time.sleep(max(1.0, EVERY_S - (time.time() - t0)))


if __name__ == "__main__":
    sys.exit(main())
