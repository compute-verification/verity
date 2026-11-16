"""A lease on every pod, made when it is created, so a pod is bounded whether or not any VM, control pod or guard process is alive."""
from __future__ import annotations

import io
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Callable

from .. import settings
from .workproc import guard_env

SH = Path(__file__).resolve().parent / "sh"
LEASE_SH = SH / "lease.sh"
POD_GUARD_SH = SH / "pod_guard.sh"
LEASE_DIR = "/root/.research"
LEASE_FILE = f"{LEASE_DIR}/lease"
LEASE_ENV = "RESEARCH_LEASE_EXPIRES"
BOOT_IMAGES = ("runpod/",)
RUN_MARGIN_S = 900
FIRST_IDLE_MIN = 20
IDLE_MIN = 90
ROOT = "/workspace/research"


def boot_fields(image: str, expires_epoch: int, *, boot: bool = False, lease_dir: str = LEASE_DIR, start: str = "/start.sh") -> dict[str, Any]:
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
lease_pid=$(sh {d}/lease.sh start)
if [ {1 if idle_min > 0 else 0} = 1 ]; then
  cd {r}/guard && setsid nohup env {env} bash {d}/pod_guard.sh daemon {r} {idle_min:g} >> {r}/guard/daemon.log 2>&1 < /dev/null &
  guard_pid=$!
else
  guard_pid=none
fi
echo "LEASE $(cat {d}/lease)"
echo "LEASE_PID $lease_pid $(sh {d}/lease.sh pid > /dev/null && echo alive || echo dead)"
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
    """Raises RuntimeError unless the lease file holds an expiry and the lease loop is alive."""
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
    return ["sh", "-c", f'exec sh "${{LEASE_DIR:-{LEASE_DIR}}}/lease.sh" "$@"', "lease", *args]


def machine_extend(seconds: float, by: str, *, lease_dir: str | None = None) -> dict[str, Any]:
    d = lease_dir or settings.get("LEASE_DIR") or LEASE_DIR
    if not Path(d, "lease.sh").is_file() or not (Path(d, "lease").is_file() or settings.get(LEASE_ENV)):
        return {"leased": False, "reason": f"no lease on this machine ({d}/lease)"}
    r = subprocess.run(extend_argv(seconds, by, d), capture_output=True, text=True, env={**os.environ, "LEASE_DIR": d})
    parts = r.stdout.split()
    if r.returncode != 0 or not parts or not parts[0].isdigit():
        return {"leased": True, "extended": False, "error": (r.stdout + r.stderr).strip()[-300:]}
    return {"leased": True, "extended": True, "expires_epoch": int(parts[0]), "expires_utc": parts[1] if len(parts) > 1 else None,
            "by": parts[2] if len(parts) > 2 else None}
