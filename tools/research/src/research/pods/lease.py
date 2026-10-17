"""A lease on every pod, made when it is created, so a pod is bounded whether or not any VM, control pod or guard process is alive.

    research pods create ... --max-hours H [--first-idle-min 20] [--idle-min 90]
    research pods extend TARGET --hours H           # the expiry becomes at least now + H (never shortened)
    research pods lease TARGET                      # the lease line: `EPOCH ISO BY`

The dead-man is `sh/lease.sh` on the pod: a loop that re-reads the lease file (LEASE_FILE) every minute and ends the pod with the
pod's own key once the lease has expired and its grace (LEASE_GRACE_S, the runs' own custody push) has passed.  It is armed:

  over ssh   `arm`, once sshd answers: ships lease.sh and pod_guard.sh to LEASE_DIR, writes the lease file, starts the loop unless
             a boot one runs, and starts the idle guard (pod_guard.sh daemon) with a short first-run window;
  at boot    only with `pods create --boot-lease` (`boot_fields`): for RunPod's own images (BOOT_IMAGES, whose CMD is /start.sh) a
             start command that writes lease.sh from the environment and runs it before /start.sh, so the lease holds before any ssh.
             Opt-in: on its one live run (2026-09-29, pod fsisu66ljla298, cpu3c) the pod ended itself in the window of its lease plus
             grace, but sshd never answered, and whether the start command or the host caused that is not settled.
The create call always puts the expiry in the pod's environment (RESEARCH_LEASE_EXPIRES), which the loop uses until the file exists.

`pods create` terminates a pod it could not arm (no sshd, a failed ssh), so no pod leaves it unbounded.  `research run --on` extends
the lease on the machine to cover the run's --timeout plus RUN_MARGIN_S (`machine_extend`, from launch-request).  The budgets guard's
per-line max pod age (research.pods.budgets) is the outer bound for a pod whose lease never got armed.
"""
from __future__ import annotations

import io
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Callable

from .workproc import guard_env

SH = Path(__file__).resolve().parent / "sh"
LEASE_SH = SH / "lease.sh"
POD_GUARD_SH = SH / "pod_guard.sh"
LEASE_DIR = "/root/.research"                     # on the pod: lease.sh, pod_guard.sh, the lease file and the loop's log
LEASE_FILE = f"{LEASE_DIR}/lease"
LEASE_ENV = "RESEARCH_LEASE_EXPIRES"
BOOT_IMAGES = ("runpod/",)                        # images whose start command is /start.sh (sshd from PUBLIC_KEY, then sleep)
RUN_MARGIN_S = 900                                # `run --on`: the lease covers --timeout plus this (the runner's publish and custody push)
FIRST_IDLE_MIN = 20
IDLE_MIN = 90
ROOT = "/workspace/research"                      # the machine root `research run --on` uses by default (remote.resolve_machine)


def boot_fields(image: str, expires_epoch: int, *, boot: bool = False, lease_dir: str = LEASE_DIR, start: str = "/start.sh") -> dict[str, Any]:
    """The create body's additions: the expiry in the pod's environment, and with `boot` (`pods create --boot-lease`), for BOOT_IMAGES,
    a start command that runs the lease loop before the image's own `start`."""
    env = {LEASE_ENV: str(int(expires_epoch))}
    if not boot or not image.startswith(BOOT_IMAGES):
        return {"env": env}
    env["RESEARCH_LEASE_SH"] = LEASE_SH.read_text()
    d = shlex.quote(lease_dir)
    cmd = (f'mkdir -p {d} && printf "%s" "$RESEARCH_LEASE_SH" > {d}/lease.sh && '
           f'(LEASE_DIR={d} nohup sh {d}/lease.sh loop >> {d}/lease.log 2>&1 < /dev/null &); exec {start}')
    return {"env": env, "dockerStartCmd": ["bash", "-c", cmd]}


def _payload() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in (LEASE_SH, POD_GUARD_SH):
            data = p.read_bytes()
            ti = tarfile.TarInfo(p.name)
            ti.size, ti.mode = len(data), 0o755
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def arm_script(seconds: int, *, first_idle_min: float, idle_min: float, root: str = ROOT, lease_dir: str = LEASE_DIR,
               extra_env: dict[str, str] | None = None) -> str:
    """The shell `arm` runs on the pod (stdin: a tar of lease.sh and pod_guard.sh).  It prints `LEASE <line>`, `LEASE_PID <pid>
    <alive|dead>` and `GUARD_PID <pid> <alive|dead>` (`none` with `idle_min` 0: no idle guard, for a pool pod kept up idle)."""
    d, r = shlex.quote(lease_dir), shlex.quote(root)
    genv = {**guard_env(), "GUARD_TERMINAL": "done failed cancelled", "GUARD_FIRST_IDLE_MIN": f"{first_idle_min:g}",
            "GUARD_LEASE_FILE": f"{lease_dir}/lease", **(extra_env or {})}
    lenv = " ".join(f"{k}={shlex.quote(v)}" for k, v in (extra_env or {}).items() if k.startswith("LEASE_"))
    env = " ".join(f"{k}={shlex.quote(v)}" for k, v in genv.items())
    return f"""set -e
mkdir -p {d} {r}/guard
tar -xf - -C {d}
export LEASE_DIR={d} {lenv}
sh {d}/lease.sh extend {int(seconds)} create > /dev/null
set +e
old=$(cat {d}/lease.pid 2>/dev/null)
if [ -n "$old" ] && kill -0 "$old" 2>/dev/null && tr '\\0' ' ' < /proc/$old/cmdline 2>/dev/null | grep -q lease.sh; then
  lease_pid=$old
else
  setsid nohup sh {d}/lease.sh loop >> {d}/lease.log 2>&1 < /dev/null &
  lease_pid=$!
  echo "$lease_pid" > {d}/lease.pid
fi
if [ {1 if idle_min > 0 else 0} = 1 ]; then
  cd {r}/guard && setsid nohup env {env} bash {d}/pod_guard.sh daemon {r} {idle_min:g} >> {r}/guard/daemon.log 2>&1 < /dev/null &
  guard_pid=$!
else
  guard_pid=none
fi
echo "LEASE $(cat {d}/lease)"
echo "LEASE_PID $lease_pid $(kill -0 "$lease_pid" 2>/dev/null && echo alive || echo dead)"
echo "GUARD_PID $guard_pid $([ "$guard_pid" != none ] && kill -0 "$guard_pid" 2>/dev/null && echo alive || echo dead)"
"""


def parse_report(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(" ")
        if key == "LEASE":
            parts = rest.split()
            out["expires_epoch"] = int(parts[0]) if parts and parts[0].isdigit() else None
            out["expires_utc"] = parts[1] if len(parts) > 1 else None
        elif key in ("LEASE_PID", "GUARD_PID"):
            pid, _, state = rest.partition(" ")
            out[key.lower()] = int(pid) if pid.isdigit() else None
            out[key.lower().replace("_pid", "_alive")] = state.strip() == "alive"
    return out


def arm(ssh: list[str], seconds: int, *, first_idle_min: float = FIRST_IDLE_MIN, idle_min: float = IDLE_MIN, root: str = ROOT,
        lease_dir: str = LEASE_DIR, extra_env: dict[str, str] | None = None,
        run: Callable[..., subprocess.CompletedProcess] | None = None) -> dict[str, Any]:
    """Arm the lease and the idle guard over `ssh` (module docstring).  Raises RuntimeError unless the lease file holds an expiry and
    the lease loop is alive: the caller terminates a pod it could not arm."""
    from .. import remote

    runner = run or remote.ssh_run
    script = arm_script(seconds, first_idle_min=first_idle_min, idle_min=idle_min, root=root, lease_dir=lease_dir, extra_env=extra_env)
    r = runner(ssh, ["sh", "-c", script], stdin=_payload(), timeout=120)
    text = r.stdout.decode(errors="replace") if isinstance(r.stdout, bytes) else (r.stdout or "")
    rep = parse_report(text)
    if r.returncode != 0 or not rep.get("expires_epoch") or not rep.get("lease_alive"):
        err = r.stderr.decode(errors="replace") if isinstance(r.stderr, bytes) else (r.stderr or "")
        raise RuntimeError(f"arming the lease failed (rc {r.returncode}): {(text + err).strip()[-400:]}")
    return rep


def extend_argv(seconds: int, by: str, lease_dir: str = LEASE_DIR) -> list[str]:
    return ["sh", f"{lease_dir}/lease.sh", "extend", str(int(seconds)), by]


def remote_argv(*args: str) -> list[str]:
    """lease.sh ARGS over ssh, in the machine's own LEASE_DIR: a Nebius VM keeps its lease in the research user's home and sets
    LEASE_DIR in /etc/environment (vm_setup.sh); a RunPod pod has none and uses LEASE_DIR's default."""
    return ["sh", "-c", f'exec sh "${{LEASE_DIR:-{LEASE_DIR}}}/lease.sh" "$@"', "lease", *args]


def machine_extend(seconds: float, by: str, *, lease_dir: str | None = None) -> dict[str, Any]:
    """Machine side of `research run --on` (launch-request): push this pod's lease to at least now + seconds.  A machine without a
    lease (the control pod, a pod made before leases) has nothing to extend and says so."""
    d = lease_dir or os.environ.get("LEASE_DIR") or LEASE_DIR
    if not Path(d, "lease.sh").is_file() or not (Path(d, "lease").is_file() or os.environ.get(LEASE_ENV)):
        return {"leased": False, "reason": f"no lease on this machine ({d}/lease)"}
    r = subprocess.run(extend_argv(seconds, by, d), capture_output=True, text=True, env={**os.environ, "LEASE_DIR": d})
    parts = r.stdout.split()
    if r.returncode != 0 or not parts or not parts[0].isdigit():
        return {"leased": True, "extended": False, "error": (r.stdout + r.stderr).strip()[-300:]}
    return {"leased": True, "extended": True, "expires_epoch": int(parts[0]), "expires_utc": parts[1] if len(parts) > 1 else None,
            "by": parts[2] if len(parts) > 2 else None}
