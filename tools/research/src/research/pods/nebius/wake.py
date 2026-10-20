"""The control pod's side of `research run --on`'s wake (remote.wake): start a stopped Nebius VM for an agent that holds no Nebius key.

    PYTHONPATH=/data/infra/store-index-src/src python3.12 -m research.pods.nebius.wake start INSTANCE_ID [--dry-run]

A lane's VM holds no Nebius key, so its `research run --on NODE` asks the control pod (vy-control-verity) over the one ssh path
agents already have to it, the one `research pods create` reads the budgets guard through (connect.ssh_base). The pod starts
only an instance that:
- exactly one machines.d entry (the steward's notes clone, $WAKE_MACHINES_D) names as its `nebius_instance_id`: an allowlist,
  never an arbitrary id;
- has a recorded `deadline` (its /etc/research/deadline, which fleet.py's sampler copies into the entry while the VM runs) more
  than 5 min away, since the VM stops itself 5 min before it;
- a live budget line covers in the budgets guard's own state ($WAKE_GUARD_STATE), judged as `research pods create` judges a new
  pod (budgets.gate).
It prints one JSON line (name, instance, was, started); a refusal exits 2 with its reason on stderr. The Nebius key it starts VMs
with is read from $WAKE_ENV (KEY=VALUE lines: NEBIUS_SA_ID, NEBIUS_SA_PUBLIC_KEY_ID, NEBIUS_SA_PRIVATE_KEY_B64, NEBIUS_BIN), a
key whose account may start instances (the cloud watch's monitor.env is the tenant viewers group's and can't), and handed to the
Nebius CLI by common.sh's nebius_auth, as fleet.py does; it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable

from research.pods import budgets, registry

MACHINES_D = Path(os.environ.get("WAKE_MACHINES_D", "/workspace/steward/research-notes/machines.d"))
GUARD_STATE = Path(os.environ.get("WAKE_GUARD_STATE", budgets.GUARD_STATE))
WAKE_ENV = Path(os.environ.get("WAKE_ENV", "/root/.config/verity/nebius-wake.env"))
DEADLINE_LEAD_S = 300            # vm_setup.sh's vy-deadline-check stops the VM from its deadline - 300 s
INSTANCE_RE = re.compile(r"computeinstance-[a-z0-9]{8,40}")


class Refused(Exception):
    pass


def allowed(iid: str, machines_d: Path, guard_state: dict, now: float) -> str:
    """The machine `iid` belongs to, when the pod may start it (module docstring); Refused naming why not."""
    if not INSTANCE_RE.fullmatch(iid):
        raise Refused(f"{iid!r} is not a Nebius instance id")
    entries, _ = registry.load_dir(machines_d)
    names = sorted(n for n, e in entries.items() if e.get("nebius_instance_id") == iid)
    if len(names) != 1:
        raise Refused(f"{iid} is named by {len(names)} machines.d entries ({', '.join(names) or 'none'}) in {machines_d}; the control "
                      "pod starts only an instance exactly one entry names")
    name = names[0]
    deadline = budgets.parse_time(entries[name].get("deadline")) if entries[name].get("deadline") else None
    if deadline is None:
        raise Refused(f"{name} has no recorded deadline in its machines.d entry (fleet.py copies /etc/research/deadline there while "
                      "it runs): ask @infra to start it")
    if now >= deadline - DEADLINE_LEAD_S:
        raise Refused(f"{name}'s deadline {budgets.iso(deadline)} has passed (it stops itself 5 min before): its budget line's owner "
                      "renews it, or it stays stopped")
    if why := budgets.gate(name, 0, guard_state, now):
        raise Refused(f"{name}: {why}")
    return name


def load_key(path: Path) -> None:
    if not path.is_file():
        raise Refused(f"the control pod holds no Nebius key that may start a VM ({path} is missing): ask @infra")
    for line in path.read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() and not k.lstrip().startswith("#"):
            os.environ[k.strip()] = v.strip().strip("'\"")


def start(iid: str, nebius: Callable[..., dict]) -> tuple[str | None, bool]:
    """(the instance's state, whether this started it): only a STOPPED or STOPPING one is started."""
    state = (nebius("compute", "instance", "get", "--id", iid).get("status") or {}).get("state")
    if state in ("STOPPED", "STOPPING"):
        nebius("compute", "instance", "start", "--id", iid, "--async")
        return state, True
    return state, False


def main(argv: list[str] | None = None, *, nebius: Callable[..., dict] | None = None, now: float | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wake.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("start")
    sp.add_argument("instance")
    sp.add_argument("--dry-run", action="store_true", help="check only; start nothing and read no key")
    a = ap.parse_args(argv)
    now = time.time() if now is None else now
    try:
        try:
            state = json.loads(GUARD_STATE.read_text())
        except (OSError, ValueError) as e:
            raise Refused(f"cannot read the budgets guard's state {GUARD_STATE} ({type(e).__name__}); nothing starts unguarded") from None
        name = allowed(a.instance, MACHINES_D, state, now)
        if a.dry_run:
            print(json.dumps({"name": name, "instance": a.instance, "was": None, "started": False, "dry_run": True}))
            return 0
        if nebius is None:
            load_key(WAKE_ENV)
            from research.pods.nebius.fleet import nb as nebius
        try:
            was, started = start(a.instance, nebius)
        except (RuntimeError, OSError, ValueError) as e:
            raise Refused(f"the Nebius API refused to start {name} ({e})") from None
    except Refused as e:
        print(f"wake.py: refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"name": name, "instance": a.instance, "was": was, "started": started}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
