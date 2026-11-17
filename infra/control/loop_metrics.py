#!/usr/bin/env python3
"""The control pod's loops' numbers, sent to the site's door (`POST /api/metrics`, `research.metrics.send`) with the pod's
panels key. Each reads only what its loop already leaves on the pod, and each is declared in metrics.toml beside this file.

    loop_metrics.py compute [--every S --state FILE] [--dry-run]
                        all compute's dollars an hour against the limit (infra/compute-usd-per-hour, whose row's `floor` is
                        the limit): the rates of the Nebius machines in the fleet loop's last pass ($FLEET_DIR/last-sample.json,
                        none when none runs) plus every RunPod pod's, the budget guard's `rate_all` at its last completed
                        poll (its state file), at the pass's time. Null with its why when a machine or a pod has no rate, when
                        the guard's poll is over STALE_S old as the reading is sent, or when the fleet's last pass is over
                        STALE_S old (sent at most every STALE_S). With --every, a pass every S seconds that sends each fleet
                        pass once, even across a restart: the last one sent is kept in FILE, replaced whole, before its send,
                        so one whose send fails isn't sent again. FILE's lock is held while it runs: a second copy refuses
                        (the watchers loop `loop-metrics`)
    loop_metrics.py index --pass FILE [--state FILE] [--dry-run]    a store index pass's length in seconds
                        (infra/store-index-pass-seconds), from the record store_index.sh writes as each pass ends (start,
                        end and exit code in epoch seconds, and its tree's commit); null with its exit code's meaning when
                        the pass failed

Each reading names the script that computed it (`command`) and its tree's commit. An unreadable input is one line on stderr
and exit 1. A failed send is one line with its reason, said once: while the door answers the same way (refuses, or isn't
there) nothing more is said, and the first send it takes after that says so once; the reason last said is kept in the
--state FILE (`index` without one says each failure). `compute --every` goes on to its next pass; `index` exits 1 only when
it says a failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from research import metrics
from research.pods import guard
from research.pods.nebius import fleet

COMPUTE, INDEX = "infra/compute-usd-per-hour", "infra/store-index-pass-seconds"
#: the rows of this directory's numbers: the compute row's `floor` is the limit
ROWS = Path(__file__).resolve().with_name("metrics.toml")
#: the deployed tree's commit: watchers.toml's {src}/COMMIT
COMMIT_FILE = Path(__file__).resolve().parents[2] / "COMMIT"
#: a guard poll this old as a reading is sent, or a fleet pass this old, is not the rate now
STALE_S = 600
#: the script that computes each reading, as its metrics.toml row names it: the reading's `command`
COMPUTE_COMMAND, INDEX_COMMAND = "infra/control/loop_metrics.py", "infra/control/store_index.sh"


def _epoch(t: str) -> float:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def limit(rows: Path = ROWS) -> float:
    """The compute row's `floor`: what all compute together may cost an hour."""
    return next(m["floor"] for m in tomllib.loads(rows.read_text())["metric"] if m["id"] == COMPUTE)


def runpod_rate(now: float, state: Path | None = None) -> tuple[float | None, str | None]:
    """Every live RunPod pod's dollars an hour at the guard's last completed poll (`rate_all`, `unrated` and `last_ok_poll` in
    its state file, which it replaces whole each poll), or None and why: no state, unreadable, no poll yet, a guard that
    doesn't record `rate_all`, a pod with no rate, or a poll over STALE_S before ``now``."""
    state = state or guard.paths()["state"]
    try:
        s = json.loads(state.read_text())
    except FileNotFoundError:
        return None, f"the budget guard has left no state at {state}"
    except (OSError, ValueError) as e:
        return None, f"the budget guard's state {state} is unreadable ({type(e).__name__})"
    t, rate = s.get("last_ok_poll"), s.get("rate_all")
    if not isinstance(t, int | float) or isinstance(t, bool):
        return None, "the budget guard has completed no poll"
    if not isinstance(rate, int | float) or isinstance(rate, bool):
        return None, "the budget guard's state has no `rate_all`: its guard predates it"
    if unrated := s.get("unrated"):
        return None, f"RunPod pod{'s' if len(unrated) > 1 else ''} {', '.join(unrated)} list{'' if len(unrated) > 1 else 's'} " \
                     "no costPerHr that is a number"
    if now - t > STALE_S:
        return None, f"the budget guard's last poll was {(now - t) / 60:.1f} min ago, over {STALE_S // 60}"
    return float(rate), None


def compute_reading(rows: list[dict], at: float, now: float, target: float, state: Path | None = None) -> dict:
    """All compute's dollars an hour at the fleet pass taken at ``at`` (its ``rows``, one per running Nebius machine, none when
    none runs), with the guard's poll aged at ``now``, as it is sent."""
    unrated = sorted(r["name"] for r in rows if r.get("rate_usd_per_hour") is None)
    runpod, why = runpod_rate(now, state)
    whys = ([f"{', '.join(unrated)} {'has' if len(unrated) == 1 else 'have'} no rate in the machines registry"] if unrated
            else []) + ([why] if why else [])
    value = None if whys else round(sum(r["rate_usd_per_hour"] for r in rows) + runpod, 2)
    return metrics.reading(COMPUTE, value, why="; ".join(whys) or None, time=fleet.utc(at), target=target)


def exit_meaning(rc: int) -> str:
    if rc == 124:
        return "the pass ran into store_index.sh's timeout (exit 124)"
    if rc > 128:
        return f"the pass was killed by signal {rc - 128} (exit {rc})" + (", as an out-of-memory kill is" if rc == 137 else "")
    return f"the pass failed with exit {rc}: store-index.log's lines before it say why"


def index_reading(rec: dict) -> dict:
    """The pass ``rec`` (store_index.sh's record) as a reading at its end."""
    rc, start, end = rec["rc"], rec["start"], rec["end"]
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (rc, start, end)):
        raise ValueError(f"start, end and rc are not whole numbers: {rec!r}")
    return metrics.reading(INDEX, end - start if rc == 0 else None, why=None if rc == 0 else exit_meaning(rc),
                           time=fleet.utc(end))


def _say(now: float, line: str) -> None:
    print(f"{fleet.utc(now)} {line}", file=sys.stderr, flush=True)


def _commit() -> str | None:
    return (COMMIT_FILE.read_text().strip() or None) if COMMIT_FILE.is_file() else None


def _last_pass() -> tuple[str, list[dict]] | None:
    f = fleet.FLEET_DIR / fleet.LAST_SAMPLE
    if not f.is_file():
        return None
    got = json.loads(f.read_text())
    return got["t"], got["rows"]


def _kept(state: Path | None) -> dict:
    return json.loads(state.read_text()) if state and state.is_file() else {}


def _keep(state: Path, kept: dict) -> None:
    tmp = state.with_name(f".{state.name}.{os.getpid()}")
    tmp.write_text(json.dumps(kept) + "\n")
    os.replace(tmp, state)


def _told(now: float, kept: dict, metric: str, why: str | None) -> bool:
    """Says a failed send's reason unless it is the one ``kept`` last said, and the first send that goes through after one;
    whether it said anything (and so changed ``kept``)."""
    if why is None:
        if kept.pop("failing", None) is None:
            return False
        _say(now, f"{metric}: the door took it again")
        return True
    if why == kept.get("failing"):
        return False
    kept["failing"] = why
    _say(now, f"{why} (said once while the door answers so)")
    return True


def compute(every: float | None = None, dry_run: bool = False, state: Path | None = None,
            clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
            send: Callable[..., str | None] = metrics.send, passes: int | None = None) -> int:
    if state is None:
        return _passes(every, dry_run, None, clock, sleep, send, passes)
    lock = state.with_name(f"{state.name}.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_WRONLY, 0o644)
    except OSError as e:
        _say(clock(), f"{COMPUTE}: nothing sent: its lock {lock} can't be opened ({type(e).__name__}): not the pod's loop")
        return 1
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _say(clock(), f"{COMPUTE}: nothing sent: another copy holds {lock}")
            return 0
        return _passes(every, dry_run, state, clock, sleep, send, passes)
    finally:
        os.close(fd)


def _passes(every: float | None, dry_run: bool, state: Path | None, clock: Callable[[], float],
            sleep: Callable[[float], None], send: Callable[..., str | None], passes: int | None) -> int:
    rc, commit = 0, _commit()
    try:
        target, kept = limit(), _kept(state)
    except (OSError, ValueError, KeyError, StopIteration) as e:
        _say(clock(), f"{COMPUTE}: nothing sent: {ROWS.name} or {state} is unreadable ({type(e).__name__}: {e})")
        return 1
    while passes is None or passes > 0:
        passes = None if passes is None else passes - 1
        now, r = clock(), None
        try:
            got = _last_pass()
            if got and now - _epoch(got[0]) <= STALE_S:
                if got[0] != kept.get("sent"):
                    r, kept["sent"] = compute_reading(got[1], _epoch(got[0]), clock(), target), got[0]
            elif kept.get("said") is None or now - kept["said"] >= STALE_S:
                r = metrics.reading(COMPUTE, None, time=fleet.utc(now), target=target,
                                    why=f"the fleet loop has finished no pass in {STALE_S // 60} min: it is down")
            if r is not None:
                kept["said"] = now
                if state is not None and not dry_run:
                    _keep(state, kept)
        except Exception as e:  # noqa: BLE001 (a state mid-move, a full disk: the next pass reads again)
            rc, r = 1, None
            _say(now, f"{COMPUTE}: not read: {type(e).__name__}: {e}")
        if r is not None:
            try:
                why = send([r], command=COMPUTE_COMMAND, commit=commit, dry_run=dry_run)
            except Exception as e:  # noqa: BLE001 (whatever a send raises is its one line, and the loop goes on)
                why = f"{COMPUTE}: the send raised {type(e).__name__}: {e}"
            rc = 1 if why else rc
            if _told(now, kept, COMPUTE, why) and state is not None and not dry_run:
                try:
                    _keep(state, kept)
                except OSError as e:
                    _say(now, f"{COMPUTE}: {state} not kept: {type(e).__name__}: {e}")
        if every is None or passes == 0:
            break
        sleep(max(1.0, every - (clock() - now)))
    return rc


def index(path: Path, dry_run: bool = False, send: Callable[..., str | None] = metrics.send,
          clock: Callable[[], float] = time.time, state: Path | None = None) -> int:
    try:
        rec = json.loads(path.read_text())
        r = index_reading(rec)
        kept = _kept(state)
    except (OSError, ValueError, KeyError, TypeError) as e:
        _say(clock(), f"{INDEX}: nothing sent: the pass record {path} or {state} is unreadable ({type(e).__name__}: {e})")
        return 1
    why = send([r], command=INDEX_COMMAND, commit=rec.get("commit") or None, dry_run=dry_run)
    if state is None:
        if why:
            _say(clock(), why)
        return 1 if why else 0
    said = _told(clock(), kept, INDEX, why)
    if said and not dry_run:
        try:
            _keep(state, kept)
        except OSError as e:
            _say(clock(), f"{INDEX}: {state} not kept: {type(e).__name__}: {e}")
    return 1 if why and said else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="loop_metrics.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--every", type=float, help="a pass every this many seconds (default: one pass); needs --state")
    c.add_argument("--state", type=Path, help="the last fleet pass sent and the failed send's reason last said, replaced whole; "
                                              "its lock is held while this runs")
    i = sub.add_parser("index")
    i.add_argument("--pass", dest="record", type=Path, required=True, help="the record store_index.sh writes at a pass's end")
    i.add_argument("--state", type=Path, help="the failed send's reason last said, replaced whole (default: say each one)")
    for p in (c, i):
        p.add_argument("--dry-run", action="store_true", help="print the request without the key and send nothing")
    a = ap.parse_args(argv)
    if a.cmd == "index":
        return index(a.record, a.dry_run, state=a.state)
    if a.every is not None and a.state is None:
        ap.error("--every needs --state: a restarted loop must not send a fleet pass again")
    return compute(a.every, a.dry_run, a.state)


if __name__ == "__main__":
    sys.exit(main())
