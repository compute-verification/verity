"""The verifier's worker processes, and the window verdict the books take from the Lean's check: each window's exact
bound on wrong receipts computed in the workers side by side, and a window's settled result set on `audit.AuditRun` as
`AuditRun.verify` would set it.

    checks = Checks(workers)              a pool of spawned processes
    Checks.settle(session, t, results, count)        the window's verdict, set on the session as `verify` sets it
    checks.bounds(profiles)               each window's bound on wrong receipts (`IntegrityProfile.bound`), one a worker
"""
from __future__ import annotations

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit  # noqa: E402


class Checks:
    def __init__(self, workers: int) -> None:
        self.workers = workers
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
        for f in [self.pool.submit(_warm) for _ in range(workers)]:
            f.result()

    @staticmethod
    def settle(session: audit.AuditRun, t: int, results: list[dict], count: int) -> audit.Verdict:
        """Window t's verdict from its batches' results in draw order, as `AuditRun.verify` would give it (`malformed`
        on a window with no draws)."""
        session._live()
        w = session.windows[t]
        if w.draws is None or w.verdict is not None:
            raise audit.SessionError(f"window {t} has no draws awaiting openings")
        if not w.draws or count != len(w.draws):
            return session._fail(t, "malformed")
        for r in results:
            if r["failure"] is not None:
                return session._fail(t, r["failure"], r["at"])
        w.verdict = audit.Verdict(t, True)
        return w.verdict

    def bounds(self, profiles) -> list[int | float]:
        """Each profile's `bound()`, exact, its stratified law's `count_bound` computed in the workers side by side (a
        60 s window's `audit.SAMPLES` draws from ~6e10 receipts take 25–75 s each in exact arithmetic); a law without
        strata in this process."""
        laws = [p.strata() for p in profiles]
        futures = [None if law is None else self.pool.submit(_count_bound, law, p.delta)
                   for p, law in zip(profiles, laws)]
        return [p.bound() if f is None else f.result() for p, f in zip(profiles, futures)]

    def close(self, wait: bool = False) -> None:
        self.pool.shutdown(wait=wait, cancel_futures=True)


def _count_bound(law, delta: float) -> int:
    return law.count_bound(delta)


def _warm() -> int:
    return 0
