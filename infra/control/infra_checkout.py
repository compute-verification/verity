#!/usr/bin/env python3
"""compute-verification/infra's main at /opt/infra on the control pod and on every node (INF-412), as on an agent VM.

    infra_checkout.py loop [--state PATH]                   a pass every EVERY_S (the watchers supervisor runs it)
    infra_checkout.py once [--state PATH] [--node NAME]...  one pass, over the named nodes only when any are given

The pod's /opt/infra is a git checkout of infra's main, fetched with drive's App (`gh_app_token.sh git`, the credential helper
whose token covers infra). A node holds no GitHub credential, so each pass sends it the pod's commit whenever the node's
`/opt/infra/.commit` names another: `git archive` over ssh (the registry's host and user, the RunPod key), unpacked as root into
`/opt/infra-trees/<commit>` and swapped in by renaming a symlink, so /opt/infra is always one whole tree. Each node keeps its
two newest trees.

The nodes are the registry's ssh machines reached as `research` whose `deadline` hasn't passed; one that doesn't answer is
tried again next pass. The auditor's devices (the challenger and the auditors, reached as `vy-admin`) get nothing from here:
they are pinned by their own installers. A node gets no /opt/agent-guard: no Cursor hook runs there. Each pass logs one line
per node it changed or couldn't reach, and the state file holds every node's last commit and error.
"""
from __future__ import annotations

import argparse
import calendar
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "research" / "src"))
from research import remote  # noqa: E402

EVERY_S = 600
CHECKOUT = Path("/opt/infra")
URL = "https://github.com/compute-verification/infra.git"
HELPER = "!/data/infra/gh_app_token.sh git"
STATE = Path("/data/watchers/infra-checkout.json")

# run as root on the node, the tree on stdin and the commit as $1
INSTALL = r"""set -eu
c=$1
case $c in *[!0-9a-f]*|'') echo "not a commit: $c" >&2; exit 2;; esac
[ ${#c} -eq 40 ] || { echo "not a commit: $c" >&2; exit 2; }
t=/opt/infra-trees/$c
mkdir -p /opt/infra-trees
rm -rf "$t.part"
mkdir "$t.part"
tar -x -C "$t.part"
printf '%s\n' "$c" > "$t.part/.commit"
rm -rf "$t"
mv "$t.part" "$t"
if [ -e /opt/infra ] && [ ! -L /opt/infra ]; then mv /opt/infra "/opt/infra-trees/replaced-$(date +%s)"; fi
ln -sfn "$t" /opt/infra.next
mv -T /opt/infra.next /opt/infra
ls -1dt /opt/infra-trees/*/ | tail -n +3 | xargs -r rm -rf
cat /opt/infra/.commit
"""


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=300, env={"GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"})
    if r.returncode:
        raise RuntimeError(f"git {' '.join(args[:3])}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def pod_main() -> str:
    """Bring the pod's checkout to infra's main and return its commit."""
    if not (CHECKOUT / ".git").exists():
        CHECKOUT.mkdir(parents=True, exist_ok=True)
        git("init", "-q", str(CHECKOUT))
    git("-C", str(CHECKOUT), "-c", "credential.helper=", "-c", f"credential.helper={HELPER}", "fetch", "-q", URL, "main")
    git("-C", str(CHECKOUT), "checkout", "-qf", "-B", "main", "FETCH_HEAD")
    return git("-C", str(CHECKOUT), "rev-parse", "HEAD")


def nodes(machines: dict) -> list[str]:
    now = time.time()

    def live(m: dict) -> bool:
        d = m.get("deadline")
        return not d or calendar.timegm(time.strptime(str(d), "%Y-%m-%dT%H:%M:%SZ")) > now

    return sorted(n for n, m in machines.items() if m.get("provider") == "ssh" and m.get("user") == "research" and live(m))


def ssh_to(name: str, machines: dict) -> list[str]:
    t = remote.ssh_host_target(remote.resolve_machine(name, "verity", machines))
    return remote.ssh_argv(t["host"], int(t.get("port", 22)), user=t.get("user", "research"), connect_timeout=10)


def install(ssh: list[str], commit: str) -> str:
    """Send `commit`'s tree to the node and swap it in; the node's /opt/infra/.commit after."""
    def feed(pipe):
        arch = subprocess.Popen(["git", "-C", str(CHECKOUT), "archive", "--format=tar", commit], stdout=subprocess.PIPE)
        while chunk := arch.stdout.read(1 << 20):
            pipe.write(chunk)
        if arch.wait():
            raise RuntimeError(f"git archive {commit} exited {arch.returncode}")
    r = remote.ssh_stream(ssh, ["sudo", "-n", "sh", "-c", INSTALL, "install", commit], feed, timeout=600)
    if r.returncode:
        raise RuntimeError(f"install exited {r.returncode}: {r.stderr.decode(errors='replace').strip()[-300:]}")
    return r.stdout.decode().strip()


def once(state_path: Path, only: list[str]) -> dict:
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    commit = pod_main()
    state["main"] = commit
    ms = remote.load_machines()
    for name in only or nodes(ms):
        rec = state.setdefault("nodes", {}).setdefault(name, {})
        try:
            ssh = ssh_to(name, ms)
            r = remote.ssh_run(ssh, ["sh", "-c", "cat /opt/infra/.commit 2>/dev/null; true"], timeout=60)
            if r.returncode:
                raise RuntimeError(r.stderr.decode(errors="replace").strip()[-300:] or f"ssh exited {r.returncode}")
            had = r.stdout.decode().strip()
            if had != commit:
                got = install(ssh, commit)
                if got != commit:
                    raise RuntimeError(f"/opt/infra/.commit reads {got[:12] or 'nothing'} after the install")
                print(f"infra-checkout: {name}: {had[:12] or 'none'} -> {commit[:12]}", flush=True)
            rec.update(commit=commit, error="")
        except Exception as e:  # one node's failure must not stop the others
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
            print(f"infra-checkout: {name}: {rec['error']}", flush=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    tmp.replace(state_path)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["loop", "once"])
    ap.add_argument("--state", type=Path, default=STATE)
    ap.add_argument("--node", action="append", default=[], help="once: only this node (repeatable)")
    a = ap.parse_args()
    if a.mode == "once":
        once(a.state, a.node)
        return
    while True:
        try:
            once(a.state, [])
        except Exception as e:  # the pod's own fetch failed: the nodes keep what they have
            print(f"infra-checkout: pass failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(EVERY_S)


if __name__ == "__main__":
    main()
