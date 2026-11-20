"""The stochastic Program's PRESCRIBED sampler input is linked to its approved source at Commit, independently of whether a random
replay sample happens to expose a substitution: the committed `runner.sampler/seed` word at every (step, request) with a committed
sampled id == the DECLARED workload seed of that request (i64 exact; the kernel reads seeds[req] + pos[req]: one word per request,
constant across its steps, no derivation).  An unlinked seed (missing word, unequal word) FAILs the boundary linkage BY NAME (kind
'seed'), the same path as an unlinked token boundary; a greedy row (no declared seed) checks nothing and says so, its verdict
byte-identical.  The split count is no input: it is the registered constant, a `Const32[S]` of the Program.

Why this is required beside the selector's evaluator: the form-(A) evaluator makes the COMMITTED seed load-bearing, not
the DECLARED one -- a prover that commits ids sampled with its own seed replays consistently.  The linkage ties the committed word to the
workload of record.  Fixtures: a `_Store` of committed words by identity (the stoch test's shape), B=1 and B=3 with staggered arrivals.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import linkage as LK
from verity_vllm.definitions.numerics import rows as RK

EMB = "model.embed_tokens"


class _Store:
    def __init__(self, blocks):
        self.blocks = blocks

    def has(self, rid, step, op_path, member):
        return (rid, step, op_path, member) in self.blocks

    def words(self, rid, step, op_path, member, elo, ehi):
        b = self.blocks.get((rid, step, op_path, member))
        if b is None:
            return None, f"no committed identity ({rid}, step {step}, {op_path}/{member})"
        return np.asarray(b).reshape(-1)[elo:ehi].copy(), {4: "u32", 8: "i64"}[np.asarray(b).dtype.itemsize]


def _world(reqs: dict[str, tuple[int, int, int | None]], *, seed_word=None):
    """reqs: rid -> (arrive, n_steps, declared seed).  An honest committed world: prompt ids at arrival, chained ids, one seed word per
    live (step, request) == the declared seed.  `seed_word`: overrides {(rid, step): word} to forge a committed word (the served run
    consumed something else than declared)."""
    blocks: dict = {}
    wreq = []
    for rid, (arrive, n, seed) in reqs.items():
        prompt = [11, 12, 13]
        steps = list(range(arrive, arrive + n))
        blocks[(rid, arrive, EMB, "in")] = np.asarray(prompt, dtype=np.uint32)
        for i, s in enumerate(steps):
            tok = 100 + 7 * i + int(rid[1:])
            blocks[(rid, s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([tok], dtype=np.uint64)
            if i > 0:
                blocks[(rid, s, EMB, "in")] = np.asarray([100 + 7 * (i - 1) + int(rid[1:])], dtype=np.uint32)
            if seed is not None:
                w = (seed_word or {}).get((rid, s), seed)
                blocks[(rid, s, IX.SAMPLER_OP, IX.SEED_PARAM)] = np.asarray([np.int64(w)], dtype=np.int64).view(np.uint64)
        wreq.append({"request_id": rid, "arrive_step": arrive, "prompt_token_ids": prompt, "steps": steps, "seed": seed})
    return _Store(blocks), wreq


# ----------------------------------------------------------------------------------------------------------------------- positives


# ----------------------------------------------------------------------------------------------------------------------- negatives


# ----------------------------------------------------------------------------------------------- sixth consumer: the linkage horizon


# ------------------------------------------------------------------------------------------ hardenings


def test_padrev_m0517_the_lag_is_read_through_gp01s_one_reader_of_the_attested_field():
    # commit_delta feeds boundary_linkage the resolved async_scheduling through program.workload.eos_lag_rows_declared: one field, one reader
    from verity_vllm.program.workload import eos_lag_rows_declared
    assert eos_lag_rows_declared({"async_scheduling": True}) == 1 and eos_lag_rows_declared({"async_scheduling": False}) == 0
    assert eos_lag_rows_declared({"async_scheduling": "true"}) == 1
    assert eos_lag_rows_declared({"async_scheduling": None}) is None and eos_lag_rows_declared({}) is None and eos_lag_rows_declared(None) is None


# ------------------------------------------------------------------------------- one engine schedule of record

def test_arrivals_of_record_follow_the_observed_schedule_only_when_the_population_was_placed_on_it():
    decl = {"r0": 0, "r1": 7}
    obs_sched = {"use": "observed", "observed_arrive_steps": {"r0": 0, "r1": 3}, "declared_arrive_steps": decl, "all_equal_declared": False}
    assert LK.arrivals_of_record(decl, obs_sched) == ({"r0": 0, "r1": 3}, True)                       # the gap workload: r1 admitted at 3
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps=decl)) == (decl, False)  # observed == declared: nothing moves
    assert LK.arrivals_of_record(decl, dict(obs_sched, use="declared")) == (decl, False)              # the declaration stands (coverage names why)
    assert LK.arrivals_of_record(decl, None) == (decl, False) and LK.arrivals_of_record(decl, {}) == (decl, False)
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps={"r0": 0, "r1": 3, "r9": 5})) == ({"r0": 0, "r1": 3}, True)   # unknown ids ignored
    # a sealed bundle's arrivals go through this one rule, fed the binding record's schedule

    from verity_vllm.pipeline import c2_replay as C2
    reqs = [{"request_id": "a", "index": 0, "arrive_step": 0}, {"request_id": "b", "index": 1, "arrive_step": 7}]
    site = SimpleNamespace(reqs=reqs, row={"commit": {"binding": {"coverage": {"schedule": obs_sched}}}})
    assert C2.arrivals(site, C2.request_labels(reqs)) == ({"r0": 0, "r1": 3}, True)
    assert C2.arrivals(SimpleNamespace(reqs=reqs, row={"commit": {}}), C2.request_labels(reqs)) == (decl, False)
