#!/usr/bin/env python3
"""The control pod's loops' numbers, sent to the site's door (`POST /api/metrics`, `research.metrics.send`) with the pod's
panels key. Each reads only what its loop already leaves on the pod, and each is declared in metrics.toml beside this file.

    loop_metrics.py compute [--every S] [--dry-run]   all compute's dollars an hour against the cap
                        (infra/compute-usd-per-hour): the Nebius machines' rates in the fleet loop's newest sample
                        ($FLEET_DIR) plus the RunPod pods' rate the budget guard read at its last completed poll (its state
                        file), at the sample's time. Null with its why when a machine of the sample has no rate, when the
                        guard's poll is over STALE_S before the sample, or when no sample is newer than STALE_S. With
                        --every, a pass every S seconds that sends each sample once (the watchers loop `loop-metrics`)
    loop_metrics.py index --pass FILE [--dry-run]    a store index pass's length in seconds
                        (infra/store-index-pass-seconds), from the record store_index.sh writes as each pass ends (start,
                        end and exit code in epoch seconds, and its tree's commit); null with its exit code's meaning when
                        the pass failed

A failed send or an unreadable input is one line on stderr and exit 1; `compute --every` goes on to its next pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from research import metrics
from research.pods import guard
from research.pods.nebius import fleet

COMPUTE, INDEX = "infra/compute-usd-per-hour", "infra/store-index-pass-seconds"
#: what all compute together may cost an hour: the Nebius machines and the RunPod pods
CAP_USD_PER_HOUR = 150
#: a guard poll this far before the sample, or a newest sample this old, is not the rate now
STALE_S = 600
COMPUTE_LOOP, INDEX_LOOP = "loop-metrics", "store-index"


def _epoch(t: str) -> float:
    return datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def runpod_rate(at: float, state: Path | None = None) -> tuple[float | None, str | None]:
    """The RunPod pods' dollars an hour at the guard's last completed poll (`rate_now`, `last_ok_poll` in its state file, which
    it replaces whole each poll), or None and why: no state, unreadable, no poll yet, or a poll over STALE_S before ``at``."""
    state = state or guard.paths()["state"]
    try:
        s = json.loads(state.read_text())
    except FileNotFoundError:
        return None, f"the budget guard has left no state at {state}"
    except (OSError, ValueError) as e:
        return None, f"the budget guard's state {state} is unreadable ({type(e).__name__})"
    t, rate = s.get("last_ok_poll"), s.get("rate_now")
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in (t, rate)):
        return None, "the budget guard has completed no poll"
    if at - t > STALE_S:
        return None, f"the budget guard's last poll was {(at - t) / 60:.0f} min before the sample, over {STALE_S // 60}"
    return float(rate), None


def compute_reading(rows: list[dict], at: float, state: Path | None = None) -> dict:
    """All compute's dollars an hour at the fleet sample taken at ``at`` (its ``rows``, one per running Nebius machine)."""
    unrated = sorted(r["name"] for r in rows if r.get("rate_usd_per_hour") is None)
    runpod, why = runpod_rate(at, state)
    whys = ([f"{', '.join(unrated)} {'has' if len(unrated) == 1 else 'have'} no rate in the machines registry"] if unrated
            else []) + ([why] if why else [])
    value = None if whys else round(sum(r["rate_usd_per_hour"] for r in rows) + runpod, 2)
    return metrics.reading(COMPUTE, value, why="; ".join(whys) or None, time=fleet.utc(at), target=CAP_USD_PER_HOUR)


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


def compute(every: float | None = None, dry_run: bool = False, clock: Callable[[], float] = time.time,
            sleep: Callable[[float], None] = time.sleep, send: Callable[..., str | None] = metrics.send,
            passes: int | None = None) -> int:
    commit, sent_t, said, rc = metrics.deployed_commit(Path(__file__).resolve().parent), None, None, 0
    while passes is None or passes > 0:
        passes = None if passes is None else passes - 1
        now, r = clock(), None
        try:
            rows = fleet.load_samples(now - 2 * STALE_S, now + 1)
            newest = max((x["t"] for x in rows), default=None)
            if newest and newest != sent_t and now - _epoch(newest) <= STALE_S:
                r, sent_t = compute_reading([x for x in rows if x["t"] == newest], _epoch(newest)), newest
            elif (not newest or now - _epoch(newest) > STALE_S) and (said is None or now - said >= STALE_S):
                r = metrics.reading(COMPUTE, None, time=fleet.utc(now), target=CAP_USD_PER_HOUR,
                                    why=f"the fleet loop has written no sample in {STALE_S // 60} min: it is down, or no "
                                        "Nebius machine runs")
        except Exception as e:  # noqa: BLE001 (a samples file cut mid-line, a guard state mid-move: the next pass reads again)
            rc = 1
            _say(now, f"{COMPUTE}: not read: {type(e).__name__}: {e}")
        if r is not None:
            said = now
            if why := send([r], loop=COMPUTE_LOOP, commit=commit, dry_run=dry_run):
                rc = 1
                _say(now, why)
        if every is None or passes == 0:
            break
        sleep(max(1.0, every - (clock() - now)))
    return rc


def index(path: Path, dry_run: bool = False, send: Callable[..., str | None] = metrics.send,
          clock: Callable[[], float] = time.time) -> int:
    try:
        rec = json.loads(path.read_text())
        r = index_reading(rec)
    except (OSError, ValueError, KeyError, TypeError) as e:
        _say(clock(), f"{INDEX}: nothing sent: the pass record {path} is unreadable ({type(e).__name__}: {e})")
        return 1
    if why := send([r], loop=INDEX_LOOP, commit=rec.get("commit") or None, dry_run=dry_run):
        _say(clock(), why)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="loop_metrics.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--every", type=float, help="a pass every this many seconds (default: one pass)")
    i = sub.add_parser("index")
    i.add_argument("--pass", dest="record", type=Path, required=True, help="the record store_index.sh writes at a pass's end")
    for p in (c, i):
        p.add_argument("--dry-run", action="store_true", help="print the request without the key and send nothing")
    a = ap.parse_args(argv)
    return compute(a.every, a.dry_run) if a.cmd == "compute" else index(a.record, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
