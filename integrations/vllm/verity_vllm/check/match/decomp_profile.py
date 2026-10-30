"""Phase accounting of the batch decomposition's per-request legs: a JSON sidecar of wall seconds and counters per phase, written only
when `X09_PROFILE_OUT` names it."""

from __future__ import annotations

import collections
import sys
import time

from verity_vllm import settings
from verity_vllm.check.match import program_compare as C

# PHASE ACCOUNTING of the per-request leg.  `X09_PROFILE_OUT=<path>` writes one JSON sidecar beside the document
# (never into it: the X-09 document's bytes are the record's) with, per request, wall seconds and counters for every phase of the leg --
# projection build, shape, component load, compare_steps (sequence criterion), and the DAG diagnostic (live_cone A / P, dag_hashes A / P,
# multiset diff) -- plus the sizes that drive them (instances, live-cone sizes, operand runs expanded, per-run fallbacks).  Inert unset.
# `X09_FORCE_DAG=1` (diagnostic ONLY, never on a row of record) runs the DAG diagnostic even when the sequence criterion held, so the
# DAG path's cost can be measured on a PASS record; the verdict fields are computed exactly as before (`canonical_equal` = the same
# expression), the extra `dag_multiset*` fields are then present where the record would have omitted them.
PROFILE_ENV, FORCE_DAG_ENV = "X09_PROFILE_OUT", "X09_FORCE_DAG"
_profile_counters = collections.Counter()   # process-local: per-run fallback / expansion counts (compare.RootView hooks below), per leg


def _profile_on() -> bool:
    return bool((settings.get(PROFILE_ENV) or "").strip())


def _memo_stats():
    m = sys.modules.get("verity_vllm.check.match.global_match_fast")
    return getattr(m, "stats", None) if m is not None else None


class _Phase:
    """Wall-clock + counter deltas of one named phase of a leg (a context manager)."""

    def __init__(self, prof: dict, name: str):
        self.prof, self.name = prof, name

    def __enter__(self):
        self.t0 = time.time()
        st = _memo_stats()
        self.s0 = dict(st) if st is not None else None
        self.c0 = dict(_profile_counters)
        return self

    def __exit__(self, *exc):
        d = {"seconds": round(time.time() - self.t0, 3)}
        st = _memo_stats()
        if st is not None and self.s0 is not None:
            delta = {k: int(v - self.s0.get(k, 0)) for k, v in st.items() if v - self.s0.get(k, 0)}
            if delta:
                d["fast_stats"] = delta
        cd = {k: int(v - self.c0.get(k, 0)) for k, v in _profile_counters.items() if v - self.c0.get(k, 0)}
        if cd:
            d["counters"] = cd
        self.prof.setdefault("phases", {})[self.name] = d
        return False


def _install_profile_counters() -> None:
    """Count the record's per-run fallbacks / progression expansions inside compare.RootView while profiling (content-neutral wrappers)."""
    if getattr(C.RootView, "_x09_prof_installed", False):
        return
    orig_per_run, orig_srcP, orig_src = C.RootView._per_run, C.RootView._srcP, C.RootView._src

    def _per_run(self, p, unfold):
        _profile_counters["per_run_progressions"] += 1
        _profile_counters["per_run_runs"] += p[-1]
        return orig_per_run(self, p, unfold)

    def _srcP(self, p, unfold):
        _profile_counters["srcP_calls"] += 1
        return orig_srcP(self, p, unfold)

    def _src(self, run, unfold):
        _profile_counters["src_calls"] += 1
        return orig_src(self, run, unfold)

    C.RootView._per_run, C.RootView._srcP, C.RootView._src = _per_run, _srcP, _src
    C.RootView._x09_prof_installed = True


class _NoPhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _n_runs_of(arg) -> int:
    """Operand runs of one operand in either form (record: one tuple per run; compact: progressions with a trailing count)."""
    if not arg:
        return 0
    t = arg[0]
    if isinstance(t[1], tuple):   # compact (space, (idx, lo, hi), (dI, dL, dH), count)
        return sum(p[-1] for p in arg)
    return len(arg)
