#!/usr/bin/env python3
"""Run the heavy coarse-policy sweep cells on a RunPod CPU pod (local machine only).

The laptop's memory guardian caps every veritor python at 2.5 GB, which excludes 70B/405B/MoE training, 8B training
above ~140k tokens and 128k-context training (``coarse_run.too_heavy``).  This script rents one memory-optimised CPU
pod, syncs the code, runs ``coarse_run --preset all --allow-heavy`` as ``N`` sharded drivers in parallel (one child
per driver, each RSS-capped far above the laptop's limit), pulls the shard files back and merges them into
``results/coarse.jsonl``.  RunPod lifecycle (REST API, ssh key, watchdog) is reused from ``zk/campaign/runpod.py``;
the API key is read from ``~/.runpod/config.toml`` and never printed.

    python accumulation/sweeps/pod.py up --vcpu 32 --shards 12 --hours 10   # create, wait for ssh, sync, setup, launch, watchdog
    python accumulation/sweeps/pod.py status                                # rows per shard + last log lines
    python accumulation/sweeps/pod.py pull                                  # rsync shard files back and merge into results/coarse.jsonl
    python accumulation/sweeps/pod.py down                                  # pull, then terminate

State (pod id, ssh endpoint) lives in ``accumulation/results/pod/pod.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "zk" / "campaign"))
import runpod  # noqa: E402  (zk/campaign/runpod.py: REST helpers, ssh, watchdog)

POD_DIR = REPO / "accumulation" / "results" / "pod"
STATE = POD_DIR / "pod.json"
REMOTE = "/workspace/veritor"
REMOTE_OUT = f"{REMOTE}/accumulation/results/pod"
NAME = "acc-sweep-veritor-campaign"
SYNC = ("accumulation", "verity_ir", "src", "pyproject.toml", "uv.lock", "README.md")
EXCLUDE = ("__pycache__", ".pytest_cache", "results/plots", "results/xcap_deadend", "results/pod", "*.png", ".probe.lock")


def _state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save(st: dict) -> None:
    POD_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def _ssh(st: dict, cmd: str, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run([*runpod.ssh_command(st["host"], st["port"]), cmd], check=check, text=True,
                          capture_output=capture)


def create(args) -> dict:
    public_key = runpod.SSH_KEY.with_suffix(".pub").read_text().strip()
    body = {
        "name": NAME, "imageName": args.image, "computeType": "CPU",
        "cpuFlavorIds": args.flavors.split(","), "cpuFlavorPriority": "custom", "vcpuCount": args.vcpu,
        "cloudType": args.cloud, "containerDiskInGb": args.disk, "volumeInGb": 0,
        "ports": ["22/tcp"], "supportPublicIp": True, "env": {"PUBLIC_KEY": public_key},
    }
    pod = runpod._request("POST", "/pods", body)
    st = {"pod_id": pod["id"], "name": NAME, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "cost_per_hr": pod.get("costPerHr"), "vcpu": args.vcpu, "shards": args.shards}
    _save(st)
    print(f"created pod {pod['id']} (${pod.get('costPerHr')}/h, {args.vcpu} vCPU, flavors {args.flavors})")
    host, port = runpod.wait_ssh(pod["id"])
    st.update(host=host, port=port)
    _save(st)
    mem = _ssh(st, "free -g | awk '/Mem:/{print $2}'; nproc", capture=True).stdout.split()
    st.update(mem_gb=int(mem[0]), nproc=int(mem[1]))
    _save(st)
    print(f"ssh root@{host} -p {port}: {st['mem_gb']} GB RAM, {st['nproc']} cores")
    return st


def sync(st: dict) -> None:
    """Push the code with tar over ssh (the stock image has no rsync)."""
    _ssh(st, f"mkdir -p {REMOTE} {REMOTE_OUT}")
    ex = [f"--exclude={e}" for e in EXCLUDE]
    tar = subprocess.Popen(["tar", "czf", "-", *ex, "-C", str(REPO), *SYNC], stdout=subprocess.PIPE)
    subprocess.run([*runpod.ssh_command(st["host"], st["port"]), f"tar xzf - -C {REMOTE}"], stdin=tar.stdout, check=True)
    tar.wait()
    cal = REPO / "accumulation" / "results" / "calibration.json"
    if cal.exists():                       # F/G basis for every (model, Q_inf) already calibrated; the child computes the rest itself
        _ssh(st, f"cp {REMOTE}/accumulation/results/calibration.json {REMOTE_OUT}/")
    print("synced")


def setup(st: dict) -> None:
    script = f"""set -e
cd {REMOTE}
if [ ! -x .venv/bin/python ]; then
  command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
  export PATH=$HOME/.local/bin:$PATH
  uv python install 3.12 -q
  uv venv --python 3.12 .venv -q
  uv pip install -q --python .venv/bin/python numpy networkx scipy
fi
PYTHONPATH=. .venv/bin/python -c "import accumulation.sweeps.coarse_run as m; print('coarse_run ok', m.bound_symbols())"
"""
    _ssh(st, script)
    print("setup ok")


def launch(st: dict, shards: int, hours: float, max_rss_mb: int, presets: list[str], mem_gate_gb: float | None = None,
           retry_errors: bool = False, tag: str | None = None, redo: str | None = None, circuits: list[str] | None = None,
           only: str | None = None, env: list[str] | None = None) -> None:
    """(Re)start ``shards`` drivers.  Every shard file is first seeded with the union of all existing shard rows, so a
    relaunch with a different shard count skips everything already measured (resume is per file).  With ``tag`` the
    drivers are *added* (nothing is killed) and write ``coarse_pod_<tag>_<k>.jsonl`` -- for a new preset or a
    ``--redo lower`` pass alongside a running sweep."""
    pre = " ".join(f"--preset {p}" for p in presets)
    timeout = int(hours * 3600 * 0.9)
    gate = f" --mem-gate-gb {mem_gate_gb:g}" if mem_gate_gb else ""
    retry = " --retry-errors" if retry_errors else ""
    rd = f" --redo {redo}" if redo else ""
    rd += f" --circuits {' '.join(circuits)}" if circuits else ""
    rd += f" --only {only}" if only else ""
    prefix = f"coarse_pod_{tag}_" if tag else "coarse_pod_"
    envs = "".join(f"{e} " for e in (env or []))       # e.g. LOWER_COARSE_MILP_S=900 for the headline training cells
    lines = [f"cd {REMOTE}", f"mkdir -p {REMOTE_OUT}/logs {REMOTE_OUT}/prev"]
    if not tag:
        lines += ["pkill -f 'coarse_run --preset' || true", "pkill -f 'child-spec' || true", "sleep 2"]
    # seed: merge every existing shard file plus the synced local results/coarse.jsonl (a fresh pod after a watchdog
    # termination starts from everything already measured), then start each new shard from a copy of it
    lines += [f"seeds=\"$(ls {REMOTE_OUT}/coarse_pod_*.jsonl 2>/dev/null) $(ls {REMOTE}/accumulation/results/coarse.jsonl 2>/dev/null)\"",
              f"[ -n \"$(echo $seeds)\" ] && PYTHONPATH=. .venv/bin/python -m accumulation.sweeps.coarse_run "
              f"--merge $seeds --out-dir {REMOTE_OUT} --file seed.jsonl"
              + ("" if tag else f" && (ls {REMOTE_OUT}/coarse_pod_*.jsonl >/dev/null 2>&1 && mv {REMOTE_OUT}/coarse_pod_*.jsonl {REMOTE_OUT}/prev/ || true)")
              + " || true"]
    for k in range(shards):
        lines.append(f"[ -f {REMOTE_OUT}/seed.jsonl ] && cp {REMOTE_OUT}/seed.jsonl {REMOTE_OUT}/{prefix}{k}.jsonl")
        lines.append(
            f"{envs}PYTHONPATH=. nohup .venv/bin/python -u -m accumulation.sweeps.coarse_run {pre} --allow-heavy --no-lock{gate}{retry}{rd} "
            f"--shard {k}/{shards} --out-dir {REMOTE_OUT} --file {prefix}{k}.jsonl --max-rss-mb {max_rss_mb} "
            f"--timeout {timeout} >> {REMOTE_OUT}/logs/shard_{tag + '_' if tag else ''}{k}.log 2>&1 &")
    lines.append("sleep 2; pgrep -fc 'coarse_run --preset'")
    # run from a file: with `ssh host "<script>"` the remote bash's own command line contains the pkill pattern
    subprocess.run([*runpod.ssh_command(st["host"], st["port"]), "cat > /tmp/acc_launch.sh"], input="\n".join(lines) + "\n",
                   text=True, check=True)
    out = _ssh(st, "bash /tmp/acc_launch.sh", capture=True)
    print(f"launched {out.stdout.strip()} shard drivers ({shards} requested), presets {presets}, child RSS cap {max_rss_mb} MB")
    st.update(shards=shards, launched=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _save(st)


def _default_gate(st: dict) -> float:
    """Half of the container's cgroup memory limit (RunPod CPU pods: ``free`` shows the host's RAM, the cgroup is smaller)."""
    out = _ssh(st, "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max",
               capture=True, check=False).stdout.strip()
    limit_gb = int(out) / 1e9 if out.isdigit() else st.get("mem_gb", 256)
    st["cgroup_limit_gb"] = limit_gb
    _save(st)
    return limit_gb / 2


def status(st: dict) -> None:
    script = f"""cd {REMOTE_OUT} 2>/dev/null || exit 0
echo "drivers running: $(pgrep -fc 'coarse_run --preset' || true)   children: $(pgrep -fc 'child-spec' || true)"
free -g | awk '/Mem:/{{print "mem used/total GB:", $3"/"$2}}'
for f in coarse_pod_*.jsonl; do [ -f "$f" ] && echo "$f: $(wc -l < $f) rows, $(grep -c '"U": [0-9]' $f) with U, $(grep -c '"L": [0-9]' $f) with L, $(grep -c '"error": "child' $f) child errors"; done
echo '--- last log lines'; for f in logs/shard_*.log; do echo "== $f"; tail -n 2 "$f" | cut -c1-200; done
"""
    _ssh(st, script, check=False)


def pull(st: dict) -> None:
    POD_DIR.mkdir(parents=True, exist_ok=True)
    remote = subprocess.Popen([*runpod.ssh_command(st["host"], st["port"]), f"tar czf - -C {REMOTE_OUT} --exclude=calibration.json ."],
                              stdout=subprocess.PIPE)
    subprocess.run(["tar", "xzf", "-", "-C", str(POD_DIR)], stdin=remote.stdout, check=True)
    remote.wait()
    files = sorted(str(p) for p in POD_DIR.glob("coarse_pod_*.jsonl"))
    print(f"pulled {len(files)} shard files into {POD_DIR}")
    # a local driver rewrites results/coarse.jsonl from its own in-memory rows: merging under it would be undone
    local = subprocess.run(["pgrep", "-f", "coarse_run --preset"], capture_output=True, text=True).stdout.split()
    if local:
        print(f"local coarse_run driver running (pid {', '.join(local)}): not merging; rerun `pull` when it is done")
    elif files:
        subprocess.run([sys.executable, "-m", "accumulation.sweeps.coarse_run", "--merge", *files], cwd=str(REPO), check=True,
                       env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("--vcpu", type=int, default=32)
    up.add_argument("--flavors", default="cpu5m,cpu3m", help="memory-optimised flavours, tried in order")
    up.add_argument("--image", default=runpod.DEFAULT_IMAGE)
    up.add_argument("--disk", type=int, default=60)
    up.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    up.add_argument("--shards", type=int, default=12)
    up.add_argument("--hours", type=float, default=10.0, help="watchdog: terminate the pod after this many hours")
    up.add_argument("--max-rss-mb", type=int, default=60000)
    up.add_argument("--preset", action="append", default=None)
    for name in ("sync", "setup", "status", "pull", "down", "terminate"):
        sub.add_parser(name)
    ln = sub.add_parser("launch")
    ln.add_argument("--shards", type=int, default=None)
    ln.add_argument("--hours", type=float, default=10.0)
    ln.add_argument("--max-rss-mb", type=int, default=60000)
    ln.add_argument("--preset", action="append", default=None)
    for p in (up, ln):
        p.add_argument("--mem-gate-gb", type=float, default=None,
                       help="drivers wait to spawn until the container's used memory is below this (default: half of cgroup memory.max)")
        p.add_argument("--retry-errors", action="store_true")
        p.add_argument("--redo", choices=["lower", "upper", "all", "stale"], default=None)
        p.add_argument("--circuits", nargs="*", default=None, help="restrict to these circuits ('training' = all non-inference)")
        p.add_argument("--only", choices=["lower", "upper", "wp"], default=None, help="run only this stage")
        p.add_argument("--env", nargs="*", default=None, help="KEY=VAL pairs exported to the drivers (e.g. LOWER_COARSE_MILP_S=900)")
    ln.add_argument("--tag", default=None, help="add drivers (kill nothing) writing coarse_pod_<tag>_k.jsonl")
    up.add_argument("--no-launch", action="store_true", help="create, sync and set up only (then use `launch --tag ...`)")
    args = ap.parse_args(argv)
    st = _state()
    if args.cmd == "up":
        st = create(args)
        runpod.watchdog(st["pod_id"], args.hours)
        sync(st)
        setup(st)
        if not args.no_launch:
            launch(st, args.shards, args.hours, args.max_rss_mb, args.preset or ["all"], args.mem_gate_gb or _default_gate(st),
                   args.retry_errors, redo=args.redo, circuits=args.circuits, only=args.only, env=args.env)
        return 0
    if not st:
        print("no pod state; run `up` first"); return 1
    if args.cmd == "sync":
        sync(st)
    elif args.cmd == "setup":
        setup(st)
    elif args.cmd == "launch":
        launch(st, args.shards or st.get("shards", 12), args.hours, args.max_rss_mb, args.preset or ["all"],
               args.mem_gate_gb or _default_gate(st), args.retry_errors, tag=args.tag, redo=args.redo, circuits=args.circuits, only=args.only,
               env=args.env)
    elif args.cmd == "status":
        status(st)
    elif args.cmd == "pull":
        pull(st)
    elif args.cmd in ("down", "terminate"):
        if args.cmd == "down":
            try:
                pull(st)
            except Exception as e:  # noqa: BLE001
                print(f"pull failed ({e}); terminating anyway")
        runpod.terminate(st["pod_id"])
        st["terminated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _save(st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
