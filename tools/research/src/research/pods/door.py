"""The control pod's door, from this side: what `research` sends the budgets guard's host once every key but root's admin key
reaches only its gate there (infra/control/vy-control-gate, which names the verbs and checks their arguments).

The pod has its door once the machine registry binds the guard's host to an `ssh` machine, the gated account (`research pods
register vy-control-verity --host H --port P --user research`): `ask` then sends a verb. Until then, or from a notes clone older
than that entry, it sends the shell command root took before the door, and says so in one line. `api` asks the gate for a RunPod
call, on a machine that holds no RunPod key: `pods/runpod.py` turns to it only then.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from typing import Any

from . import budgets

_told: set[str] = set()     # the hosts this process has already said have no door


def gated(host: str) -> bool:
    """Whether the registry binds `host` to an ssh machine: its gated account, which takes verbs and no shell command."""
    from .. import remote

    return (remote.load_machines().get(host) or {}).get("provider") == "ssh"


def ask(verb: list[str], before: str, *, host: str | None = None, timeout: float, run=None,
        stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run `verb` on the control pod through its door; where the pod has no door yet, `before`, the shell command its root took."""
    from . import connect

    host = host or budgets.guard_host()
    command = shlex.join(verb)
    if not gated(host):
        command = before
        if host not in _told:
            _told.add(host)
            print(f"research: {host} has no door yet (the machine registry has no ssh entry for it): asking its root shell, as before",
                  file=sys.stderr)
    kw = {} if stdin is None else {"input": stdin}
    return (run or subprocess.run)([*connect.ssh_base(host), command], capture_output=True, text=True, timeout=timeout, **kw)


def hint(host: str, stderr: str | None) -> str:
    """What to add to a failed `ask`: a pod whose door has closed refuses root's shell to a checkout that hasn't its entry yet."""
    if "Permission denied" in (stderr or "") and not gated(host):
        return f" ({host} may have its door now: `research notes sync` brings its registry entry)"
    return ""


def _verb(method: str, path: str, body: dict[str, Any] | None) -> tuple[list[str], str | None]:
    pod = re.fullmatch(r"/pods/([a-z0-9]+)", path)
    if (method, path) == ("GET", "/pods"):
        return ["pods-list"], None
    if method == "GET" and pod:
        return ["pods-get", pod[1]], None
    if (method, path) == ("POST", "/pods"):
        return ["pods-create"], json.dumps(body)
    if method == "DELETE" and pod:
        return ["pods-terminate", pod[1]], None
    raise LookupError(f"{method} {path}")


def api(method: str, path: str, body: dict[str, Any] | None = None, *, no_key: str = "no RunPod API key here", run=None) -> Any:
    """RunPod's REST call `method path`, made by the control pod's gate with the account's key, which stays there; what
    `runpod._request` returns, and `runpod.PodError` (with the HTTP status the pod saw) as it raises."""
    from . import connect, runpod

    host = budgets.guard_host()
    pod_id = path[len("/pods/"):].split("/", 1)[0] if path.startswith("/pods/") else None
    try:
        verb, stdin = _verb(method, path, body)
    except LookupError:
        raise runpod.PodError(f"{no_key}, and {host}'s door has no verb for {method} {path}", pod_id=pod_id) from None
    if host == "local":             # this is the guard's host: its gate asks RunPod with the key that is missing here
        raise runpod.PodError(no_key, pod_id=pod_id)
    if not gated(host):
        raise runpod.PodError(f"{no_key}, and {host} has no door to ask (the machine registry has no ssh entry for it)", pod_id=pod_id)
    try:
        r = (run or subprocess.run)([*connect.ssh_base(host), shlex.join(verb)], capture_output=True, text=True, timeout=180,
                                    **({} if stdin is None else {"input": stdin}))
    except (OSError, subprocess.TimeoutExpired) as e:
        raise runpod.PodError(f"{method} {path} through {host}'s door: {type(e).__name__}: {e}", pod_id=pod_id) from None
    if r.returncode != 0:
        why = ((r.stderr or r.stdout).strip().splitlines() or [f"rc {r.returncode}"])[-1][:400]
        code = re.search(r"-> HTTP (\d{3})\b", why)
        raise runpod.PodError(f"{method} {path} through {host}'s door: {why}", code=int(code[1]) if code else None, pod_id=pod_id)
    return json.loads(r.stdout) if r.stdout.strip() else None
