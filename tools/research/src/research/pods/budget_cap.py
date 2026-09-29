#!/usr/bin/env python3
"""Hard spend cap for a RunPod day (lead, 2026-09-09; user authorization §57.10).

    python zk/campaign/budget_cap.py start --cap 40 --rate-max 50 --tag 2026-09-09   # detached daemon
    python zk/campaign/budget_cap.py status --tag 2026-09-09
    python zk/campaign/budget_cap.py stop --tag 2026-09-09

Every POLL seconds the daemon lists all pods on the account and integrates ``costPerHr`` over the
elapsed interval for every pod that is not EXITED/TERMINATED (a pod is charged whether or not a lane is
using it).  When the integrated spend since ``start`` reaches ``--cap`` dollars, or the aggregate
rate exceeds ``--rate-max`` $/h, it terminates every pod IT OWNS and keeps terminating owned pods that appear,
then keeps logging.  State is in ``~/.runpod/budget-<tag>.json`` so a restart continues the tally
instead of resetting it.  Detached with ``start_new_session`` (and ``caffeinate`` on macOS) so it
survives the shell that launched it.  Never prints the API key.

Ownership (2026-09-12, after the stale ``2026-09-11-ext`` daemon terminated the EOD 4090 mid-run): a daemon counts and may
terminate ONLY pods whose name starts with one of its ``--own-prefix`` values (default: the tag) or whose id is listed in
``~/.runpod/owned-<tag>.txt``; every other pod is logged FOREIGN and left alone.  One daemon per machine: ``start``/``run``
refuse if another ``budget_cap.py run`` is alive unless ``--allow-multiple``.

Fail closed (2026-09-28, after a full disk ended every guard on the control pod): each poll terminates first and records after, every
state and log write is best effort (a failure goes to stderr and never ends the loop), an unreadable ``--cap-file`` keeps the cap in
force (across a restart too), a state file that does not parse is set aside and a new tally starts, and ``start`` runs the daemon under
a respawn loop, so a crash costs one poll.  ``stop`` ends the loop and the daemon together (their process group).
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod  # noqa: E402

POLL = 60
HOME = Path.home() / ".runpod"


def _now() -> float:
    return time.time()


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_path(tag: str) -> Path:
    return HOME / f"budget-{tag}.json"


def _log_path(tag: str) -> Path:
    return HOME / f"budget-{tag}.log"


def _say(msg: str) -> None:
    try:
        print(f"budget_cap: {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 -- stderr is the log, on the same disk
        pass


def _load(tag: str) -> dict:
    p = _state_path(tag)
    fresh = {"tag": tag, "spent": 0.0, "last_poll": None, "pods": {}, "tripped": None, "started": _stamp()}
    try:
        state = json.loads(p.read_text())
        if isinstance(state, dict) and isinstance(state.get("pods"), dict):
            return state
        why = "not a budget state"
    except FileNotFoundError:
        return fresh
    except (OSError, ValueError) as err:
        why = f"{type(err).__name__}: {err}"
    try:
        p.replace(p.with_name(f"{p.name}.corrupt-{int(_now())}"))
    except OSError:
        pass
    _say(f"state {p} unusable ({why}): set aside, a new tally starts")
    fresh["tally_restarted"] = {"at": _stamp(), "why": why}
    return fresh


def _save(tag: str, state: dict) -> None:
    """Best effort: a failed write (a full disk) is reported and the daemon goes on enforcing from memory."""
    tmp = _state_path(tag).with_name(f".{_state_path(tag).name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(state, indent=1, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(_state_path(tag))
    except Exception as err:  # noqa: BLE001
        _say(f"state write failed ({type(err).__name__}: {err}); enforcing from memory")
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _log(tag: str, msg: str) -> None:
    try:
        with _log_path(tag).open("a") as f:
            f.write(f"{_stamp()} {msg}\n")
    except Exception as err:  # noqa: BLE001 -- an unwritable log never stops enforcement
        _say(f"log write failed ({type(err).__name__}: {err}): {msg}")


def _live_pods() -> list[dict]:
    pods = runpod._request("GET", "/pods") or []
    live = []
    for pod in pods:
        status = (pod.get("desiredStatus") or "").upper()
        if status in ("EXITED", "TERMINATED"):
            continue
        live.append(pod)
    return live


def _owned_ids_path(tag: str) -> Path:
    return HOME / f"owned-{tag}.txt"


def _owned(tag: str, pods: list[dict], prefixes: list[str]) -> tuple[list[dict], list[dict]]:
    """[2026-09-12 01:55Z, after the stale-daemon incident] A daemon counts and may terminate ONLY the pods of its own run:
    a pod is owned iff its name starts with one of ``prefixes`` (default: the tag) or its id is listed in ``~/.runpod/owned-<tag>.txt``
    (one id per line; the launcher appends).  Everything else is FOREIGN: logged, never counted, never terminated."""
    ids: set[str] = set()
    p = _owned_ids_path(tag)
    if p.exists():
        ids = {ln.strip() for ln in p.read_text().splitlines() if ln.strip()}
    mine, foreign = [], []
    for pod in pods:
        name = str(pod.get("name") or "")
        if pod.get("id") in ids or any(name.startswith(pre) for pre in prefixes if pre):
            mine.append(pod)
        else:
            foreign.append(pod)
    return mine, foreign


def _other_daemons() -> list[str]:
    """Other ``budget_cap.py run`` processes on this machine (one daemon per machine, by tag)."""
    try:
        out = subprocess.run(["pgrep", "-fl", "budget_cap.py run"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return []
    me = str(os.getpid())
    found = []
    for ln in out.splitlines():
        toks = ln.split()
        if not toks or toks[0] == me or "caffeinate" in toks:
            continue
        # a real daemon: python <path>/budget_cap.py run ... (not a shell whose command text merely mentions it)
        idx = next((i for i, t in enumerate(toks) if t.endswith("budget_cap.py")), None)
        if idx is not None and idx + 1 < len(toks) and toks[idx + 1] == "run" and toks[1].endswith(("python", "python3")):
            found.append(ln)
    return found


def _terminate_all(tag: str, pods: list[dict], why: str) -> None:
    for pod in pods:
        pid = pod.get("id")
        try:
            runpod._request("DELETE", f"/pods/{pid}")
            _log(tag, f"TERMINATED {pid} {pod.get('name')} ${pod.get('costPerHr')}/h — {why}")
        except (SystemExit, Exception) as err:  # noqa: BLE001 — runpod.PodError on HTTP errors; one failed DELETE must not stop the rest
            _log(tag, f"terminate {pid} failed: {err}")


def _cap_now(tag: str, cap: float, cap_file: str | None, state: dict) -> float:
    """The cap in force: ``--cap``, or the number in ``--cap-file`` when that file is readable (phase gates: the integrator raises the
    cap by writing the file; the daemon never lowers its own state). [2026-09-13, D55]  ``cap`` is the cap in force so far, which an
    unreadable file keeps (``run`` seeds it from the state's last cap on a restart)."""
    if not cap_file:
        return cap
    try:
        value = float(Path(cap_file).read_text().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return cap
    if value != state.get("cap"):
        _log(tag, f"cap {state.get('cap')} -> {value} (from {cap_file})")
        state["cap"] = value
    return value


def _self_pod_id() -> str | None:
    """This pod's id when running on a RunPod pod: the env var in the container's main process (ssh sessions do not inherit it, so
    read /proc/1/environ), else /root/dm/POD_ID written by the launcher, else None (laptop)."""
    if os.environ.get("RUNPOD_POD_ID"):
        return os.environ["RUNPOD_POD_ID"]
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"RUNPOD_POD_ID="):
                return item.split(b"=", 1)[1].decode()
    except OSError:
        pass
    try:
        return Path("/root/dm/POD_ID").read_text().strip() or None
    except OSError:
        return None


def _overaged(tag: str, pods: list[dict], state: dict, pod_max_hours: float | None) -> list[dict]:
    """Owned pods first seen more than ``pod_max_hours`` ago — terminated whatever their job is doing (the per-pod hard ceiling that does not
    depend on any laptop or agent being alive; D55). None if the ceiling is unset."""
    if not pod_max_hours:
        return []
    out = []
    me = _self_pod_id()  # the controller pod itself is exempt (the deadline script terminates it last)
    for pod in pods:
        first = state["pods"].get(pod["id"], {}).get("first_seen")
        if not first or pod["id"] == me:
            continue
        age_h = (_now() - calendar.timegm(time.strptime(first, "%Y-%m-%dT%H:%M:%SZ"))) / 3600.0  # first_seen is UTC (_stamp)
        if age_h > pod_max_hours:
            out.append(pod)
    return out


def run(tag: str, cap: float, rate_max: float, prefixes: list[str] | None = None, allow_multiple: bool = False,
        cap_file: str | None = None, pod_max_hours: float | None = None) -> None:
    state = _load(tag)
    if cap_file and state.get("cap") is not None:
        cap = float(state["cap"])            # the cap in force before the restart: an unreadable --cap-file must not fall back to --cap
    state.setdefault("cap", cap)
    state.setdefault("rate_max", rate_max)
    prefixes = list(prefixes) if prefixes else [tag]
    state["own_prefixes"] = prefixes
    others = _other_daemons()
    if others and not allow_multiple:
        _log(tag, f"REFUSED to start: another budget daemon is running on this machine (one per machine, by tag): {others}")
        sys.exit(3)
    _log(tag, f"daemon pid {os.getpid()} start cap=${cap} rate_max=${rate_max}/h spent_so_far=${state['spent']:.2f} owns name-prefixes={prefixes} + ids in {_owned_ids_path(tag).name}"
              + (f" cap-file={cap_file}" if cap_file else "") + (f" pod-max-hours={pod_max_hours}" if pod_max_hours else ""))
    while True:
        try:
            cap = _poll(tag, cap, rate_max, prefixes, cap_file, pod_max_hours, state)
        except (SystemExit, Exception) as err:  # noqa: BLE001 — a safety daemon must never die on a network blip
            # 2026-09-09: an unhandled urllib URLError (DNS hiccup) killed the daemon at 16:54 PT and a
            # pod ran ~2.75 h uncounted.  Log, keep the accumulated state, and retry.
            _log(tag, f"poll failed: {type(err).__name__}: {err}")
        time.sleep(POLL)


def _poll(tag: str, cap: float, rate_max: float, prefixes: list[str], cap_file: str | None, pod_max_hours: float | None, state: dict) -> float:
    """One poll: tally, decide, terminate; only then save and log (both best effort).  Returns the cap in force."""
    cap = _cap_now(tag, cap, cap_file, state)
    pods, foreign = _owned(tag, _live_pods(), prefixes)
    now = _now()
    last = state.get("last_poll")
    hours = (now - last) / 3600.0 if last else 0.0
    rate = 0.0
    for pod in pods:
        r = float(pod.get("costPerHr") or 0.0)
        rate += r
        entry = state["pods"].setdefault(pod["id"], {"name": pod.get("name"), "rate": r, "spent": 0.0, "first_seen": _stamp()})
        entry["rate"] = r
        entry["spent"] += r * hours
        entry["last_seen"] = _stamp()
    state["spent"] += rate * hours
    state["last_poll"] = now
    state["rate_now"] = rate
    tripped = state.get("tripped")
    fresh_trip = None
    if not tripped and state["spent"] >= cap:
        tripped = fresh_trip = f"CAP ${cap} reached (spent ${state['spent']:.2f})"
    if not tripped and rate > rate_max:
        tripped = fresh_trip = f"RATE ${rate:.2f}/h exceeds ${rate_max}/h"
    if fresh_trip:
        state["tripped"] = f"{_stamp()} {fresh_trip}"
    if tripped:
        if pods:
            _terminate_all(tag, pods, tripped)
    else:
        old = _overaged(tag, pods, state, pod_max_hours)
        if old:
            _terminate_all(tag, old, f"POD AGE > {pod_max_hours} h (per-pod ceiling)")
    _save(tag, state)
    if fresh_trip:
        _log(tag, f"TRIPPED: {fresh_trip}")
    if foreign:
        _log(tag, "FOREIGN (not counted, never terminated): " + " ".join(f"{p['id']}:{p.get('name')}@{p.get('costPerHr')}" for p in foreign))
    if pods:
        _log(tag, f"{len(pods)} pod(s) ${rate:.2f}/h  spent=${state['spent']:.2f} of ${cap}  " + " ".join(f"{p['id']}@{p.get('costPerHr')}" for p in pods))
    return cap


def start(args: argparse.Namespace) -> None:
    HOME.mkdir(exist_ok=True)
    others = _other_daemons()
    if others and not args.allow_multiple:
        print(f"REFUSED: another budget daemon is running on this machine (stop it first, or --allow-multiple): {others}", file=sys.stderr)
        sys.exit(3)
    command = [sys.executable, str(Path(__file__).resolve()), "run", "--tag", args.tag, "--cap", str(args.cap), "--rate-max", str(args.rate_max)]
    for pre in args.own_prefix or []:
        command += ["--own-prefix", pre]
    if args.allow_multiple:
        command.append("--allow-multiple")
    if args.cap_file:
        command += ["--cap-file", args.cap_file]
    if args.pod_max_hours:
        command += ["--pod-max-hours", str(args.pod_max_hours)]
    if os.uname().sysname == "Darwin":
        command = ["caffeinate", "-is", *command]
    command = ["sh", "-c", RESPAWN, "budget_cap-respawn", str(POLL), *command]
    with _log_path(args.tag).open("ab") as handle:
        proc = subprocess.Popen(command, stdout=handle, stderr=handle, stdin=subprocess.DEVNULL, start_new_session=True)
    (HOME / f"budget-{args.tag}.pid").write_text(str(proc.pid))
    print(f"budget_cap daemon pid {proc.pid} (respawn loop) cap=${args.cap} rate_max=${args.rate_max}/h log={_log_path(args.tag)}")


#: `start`'s supervisor: the daemon again one poll after any exit but 3 (another daemon runs); `stop` signals its process group.
RESPAWN = ('poll=$1; shift; while :; do "$@"; rc=$?; [ "$rc" -eq 3 ] && exit 3; '
           'echo "$(date -u +%FT%TZ) budget_cap exited $rc; restarting in ${poll}s"; sleep "$poll"; done')


def status(args: argparse.Namespace) -> None:
    state = _load(args.tag)
    pidfile = HOME / f"budget-{args.tag}.pid"
    alive = False
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), 0)
            alive = True
        except OSError:
            alive = False
    print(json.dumps({"daemon_alive": alive, "spent": round(state["spent"], 2), "cap": state.get("cap"), "rate_now": state.get("rate_now"),
                      "tripped": state.get("tripped"), "last_poll": state.get("last_poll") and dt.datetime.fromtimestamp(state["last_poll"], dt.timezone.utc).isoformat(),
                      "pods": state["pods"]}, indent=1))


def stop(args: argparse.Namespace) -> None:
    pidfile = HOME / f"budget-{args.tag}.pid"
    if pidfile.exists():
        pid = int(pidfile.read_text())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)
        print(f"stopped {pid}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "run"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
        p.add_argument("--cap", type=float, required=True)
        p.add_argument("--rate-max", type=float, default=50.0)
        p.add_argument("--own-prefix", action="append", default=None,
                       help="pod-name prefix this daemon owns (repeatable; default: the tag). Pods not owned are logged FOREIGN, never counted, never terminated. "
                            "Pod ids may also be listed in ~/.runpod/owned-<tag>.txt")
        p.add_argument("--allow-multiple", action="store_true", help="permit a second daemon on this machine (default: refuse — one daemon per machine)")
        p.add_argument("--cap-file", default=None, help="read the cap from this file at every poll (phase gates); falls back to --cap when unreadable")
        p.add_argument("--pod-max-hours", type=float, default=None, help="terminate any owned pod first seen more than this many hours ago (per-pod hard ceiling)")
    for name in ("status", "stop"):
        p = sub.add_parser(name)
        p.add_argument("--tag", required=True)
    args = ap.parse_args()
    if args.cmd == "start":
        start(args)
    elif args.cmd == "run":
        run(args.tag, args.cap, args.rate_max, prefixes=args.own_prefix, allow_multiple=args.allow_multiple,
            cap_file=args.cap_file, pod_max_hours=args.pod_max_hours)
    elif args.cmd == "status":
        status(args)
    elif args.cmd == "stop":
        stop(args)


if __name__ == "__main__":
    main()
