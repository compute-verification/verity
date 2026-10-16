"""A window's opening checks spread over worker processes: each worker runs the driver's own check of an opening
(`audit.Session._check`) over a copy of what it reads of the session (the parameters, the registration, the window's
challenge, plans and roots, and the root before its first), so the verdict is `Session.verify`'s: the window is
rejected at its first failing draw in draw order, with that draw's failure, and accepted when every draw's opening
checks and there are as many openings as draws.

    checks = Checks(workers)              a pool of spawned processes
    snap = checks.snapshot(session, t)    once a window's draws are out
    future = checks.submit(snap, t, draws, bodies)   a batch of the wire's openings frames, checked in a worker
    checks.settle(session, t, results, count)        the window's verdict, set on the session as `verify` sets it
"""
from __future__ import annotations

import hashlib
import multiprocessing
import pickle
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit  # noqa: E402

import wire  # noqa: E402

_views: dict[bytes, tuple[audit.Session, object]] = {}


def _view(snap: bytes):
    tag = hashlib.sha256(snap).digest()
    if tag not in _views:
        if len(_views) > 8:
            _views.clear()
        view = pickle.loads(snap)
        _views[tag] = (view, view.registration.key)
    return _views[tag]


def check(snap: bytes, t: int, draws: list[tuple[int, int]], bodies: list[bytes]) -> dict:
    """The first failing draw of a batch, in order, or none: {"failure", "at", "n", "cpu_ns"}."""
    c0 = time.process_time_ns()
    view, key = _view(snap)
    openings = []
    for body in bodies:
        got = wire.decode_openings(body)
        if got is None:
            return {"failure": "malformed", "at": None, "n": 0, "cpu_ns": time.process_time_ns() - c0}
        openings.extend(got)
    failure, at, receipters = None, None, {}
    if len(openings) != len(draws):
        failure = "malformed"
    else:
        for (i, p), o in zip(draws, openings):
            failure = audit.Session._check(view, t, i, p, o, key, receipters)
            if failure is not None:
                at = (i, p)
                break
    return {"failure": failure, "at": at, "n": len(openings), "cpu_ns": time.process_time_ns() - c0}


class Checks:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        for f in [self.pool.submit(_warm) for _ in range(workers)]:
            f.result()

    @staticmethod
    def snapshot(session: audit.Session, t: int) -> bytes:
        """What `_check` reads of window t, pickled: the session's parameters, tiles, registration and payload domain,
        the window's challenge, plans and roots, and the window before's last root."""
        w = session.windows[t]
        view = object.__new__(type(session))
        view.params, view.tiles, view.registration = session.params, session.tiles, session.registration
        view._payload_domain = session._payload_domain
        view.windows = {t: SimpleNamespace(challenge=w.challenge, plans=list(w.plans), roots=list(w.roots))}
        if t > 0:
            view.windows[t - 1] = SimpleNamespace(roots=[session.windows[t - 1].roots[-1]])
        return pickle.dumps(view)

    def submit(self, snap: bytes, t: int, draws: list[tuple[int, int]], bodies: list[bytes]) -> Future:
        return self.pool.submit(check, snap, t, draws, bodies)

    @staticmethod
    def settle(session: audit.Session, t: int, results: list[dict], count: int) -> audit.Verdict:
        """Window t's verdict from its batches' results in draw order, as `Session.verify` would give it."""
        session._live()
        w = session.windows[t]
        if w.draws is None or w.verdict is not None:
            raise audit.SessionError(f"window {t} has no draws awaiting openings")
        if count != len(w.draws):
            return session._fail(t, "malformed")
        for r in results:
            if r["failure"] is not None:
                return session._fail(t, r["failure"], r["at"])
        w.verdict = audit.Verdict(t, True)
        return w.verdict

    def close(self, wait: bool = False) -> None:
        self.pool.shutdown(wait=wait, cancel_futures=True)


def _warm() -> int:
    return 0
