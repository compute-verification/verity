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

import numpy as np

from verity_vllm.check.replay import index as IX, linkage as LK
from verity.definitions.numerics import rows as RK

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


def _world(reqs: dict[str, tuple[int, int, int]], *, seed_word=None):
    """reqs: rid -> (arrive, n_steps, declared seed).  An honest committed world: prompt ids at arrival, chained ids, one seed word per
    live (step, request) == the declared seed.  `seed_word`: overrides {(rid, step): word} to forge a committed word (the served run
    consumed something else than declared)."""
    blocks: dict = {}
    wreq = []
    for rid, (arrive, n, seed) in reqs.items():
        prompt = [11, 12, 13]
        steps = list(range(arrive, arrive + n))
        blocks[(rid, arrive, EMB, IX.IDS_MEMBER)] = np.asarray(prompt, dtype=np.uint32)
        for i, s in enumerate(steps):
            tok = 100 + 7 * i + int(rid[1:])
            blocks[(rid, s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([tok], dtype=np.uint64)
            if i > 0:
                blocks[(rid, s, EMB, IX.IDS_MEMBER)] = np.asarray([100 + 7 * (i - 1) + int(rid[1:])], dtype=np.uint32)
            if seed is not None:
                w = (seed_word or {}).get((rid, s), seed)
                blocks[(rid, s, IX.SAMPLER_OP, IX.SEED_PARAM)] = np.asarray([np.int64(w)], dtype=np.int64).view(np.uint64)
        wreq.append({"request_id": rid, "arrive_step": arrive, "prompt_token_ids": prompt, "steps": steps, "seed": seed})
    return _Store(blocks), wreq


def _lk(store, wreq, eos=None, lag=0):
    return LK.boundary_linkage({"entries": []}, lambda *a: None, wreq, embed_op_path=EMB, store=store, eos_of_record=eos, eos_lag_rows=lag)


# ----------------------------------------------------------------------------------------------------------------------- positives

def test_b1_honest_stochastic_row_links_its_seed_by_name():
    store, wreq = _world({"r0": (0, 4, 1077402498)})
    lk = _lk(store, wreq)
    assert lk["result"] is True and lk["pairs"] == 4 and lk["linked"] == 4 and lk["unlinked_n"] == 0, lk
    assert lk["prescribed"] == {"seed": {"rule": LK.SEED_RULE, "requests_declared": 1, "checked": 4, "linked": 4}}
    assert lk["method"].endswith("prescribed-input linkage (seed)")


def test_b3_staggered_arrivals_link_every_live_requests_seed():
    store, wreq = _world({"r0": (0, 6, 5), "r1": (2, 4, 6), "r2": (3, 2, 7)})
    lk = _lk(store, wreq)
    assert lk["result"] is True, lk
    assert lk["prescribed"]["seed"]["checked"] == 12 == lk["prescribed"]["seed"]["linked"]             # 6+4+2 live (step, request) pairs


def test_greedy_row_checks_nothing_and_says_so_verdict_unchanged():
    store, wreq = _world({"r0": (0, 3, None), "r1": (0, 3, None)})
    lk = _lk(store, wreq)
    assert lk["result"] is True and lk["pairs"] == 6 and lk["linked"] == 6
    assert lk["prescribed"]["seed"]["checked"] == 0 and "not applicable" in lk["prescribed"]["seed"]["why"]
    assert "prescribed-input" not in lk["method"]                                                        # the greedy method text is the old one


# ----------------------------------------------------------------------------------------------------------------------- negatives

def test_a_committed_seed_that_is_not_the_declared_one_fails_by_name_at_the_step():
    # the served run consumed its own seed at step 2 of r0 (the evaluator alone would replay it consistently)
    store, wreq = _world({"r0": (0, 4, 1077402498)}, seed_word={("r0", 2): 4242})
    lk = _lk(store, wreq)
    assert lk["result"] is False and lk["unlinked_n"] == 1
    u = lk["unlinked"][0]
    assert u["kind"] == "seed" and u["request_id"] == "r0" and u["engine_step"] == 2
    assert lk["why"] == "committed runner.sampler/seed 4242 != declared workload seed 1077402498 of r0 at step 2 (prescribed input)"
    assert lk["prescribed"]["seed"] == {"rule": LK.SEED_RULE, "requests_declared": 1, "checked": 4, "linked": 3}


def test_a_negative_declared_seed_is_compared_as_i64_exactly():
    store, wreq = _world({"r0": (0, 2, -17)})
    assert _lk(store, wreq)["result"] is True
    store2, wreq2 = _world({"r0": (0, 2, -17)}, seed_word={("r0", 1): -18})
    lk = _lk(store2, wreq2)
    assert lk["result"] is False and "committed runner.sampler/seed -18 != declared workload seed -17" in lk["why"]


def test_a_missing_seed_word_where_a_sampled_id_is_committed_fails_by_name():
    store, wreq = _world({"r0": (0, 3, 5)})
    del store.blocks[("r0", 1, IX.SAMPLER_OP, IX.SEED_PARAM)]
    lk = _lk(store, wreq)
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "seed" and lk["unlinked"][0]["engine_step"] == 1
    assert "prescribed seed of r0 at step 1 not committed" in lk["why"]


def test_a_seed_word_that_changes_across_the_requests_steps_is_named_at_the_first_differing_step():
    # The seed is constant per request; a per-step 'derivation' the run applied on its own is a substitution at every step
    store, wreq = _world({"r0": (0, 4, 100)}, seed_word={("r0", 1): 101, ("r0", 2): 102, ("r0", 3): 103})
    lk = _lk(store, wreq)
    assert lk["result"] is False and lk["unlinked_n"] == 3 and [u["engine_step"] for u in lk["unlinked"]] == [1, 2, 3]
    assert lk["prescribed"]["seed"]["linked"] == 1


def test_a_request_with_a_declared_seed_and_no_committed_sampled_id_is_unlinked_by_name():
    store, wreq = _world({"r0": (0, 3, 5)})
    for s in range(3):
        del store.blocks[("r0", s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)]
    lk = _lk(store, wreq)
    assert lk["result"] is False and any(u["kind"] == "seed" and "linked at no step" in u["why"] for u in lk["unlinked"])


def test_the_verdict_names_the_prescribed_input_through_the_boundary_linkage_path():
    from verity_vllm.check.verdict import commit_verdict
    store, wreq = _world({"r0": (0, 4, 1077402498)}, seed_word={("r0", 2): 4242})
    lk = _lk(store, wreq)
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
            "binding": {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
            "weights_pin": {"result": True, "compared": 98, "mismatched": [], "missing": []},
            "sampled_replay": {"result": True, "grade": "complete", "why": None}, "boundary_linkage": lk}
    ok, why = commit_verdict(base, True, min_openings=64)
    assert not ok and why.startswith("C2 boundary linkage: committed runner.sampler/seed 4242 != declared workload seed 1077402498 of r0 at step 2"), why
    assert why.endswith("(1 unlinked)")


# ----------------------------------------------------------------------------------------------- sixth consumer: the linkage horizon

def test_linkage_horizon_is_the_program_of_records_steps_capped_by_the_declared_cap():
    # B=1 eos-truncated: cap 32, served-shape Program build_request_LP32_T11 has 12 steps -> 12 steps linked, not 32
    assert LK.linkage_horizon(0, 32, 12) == list(range(12))
    assert LK.linkage_horizon(5, 32, 12) == list(range(5, 17))
    assert LK.linkage_horizon(0, 32, None) == list(range(32))                     # unknown Program steps: the cap (older records)
    assert LK.linkage_horizon(0, 8, 12) == list(range(8))                         # a Program longer than the cap: the cap
    assert LK.linkage_horizon(3, 0, 12) == []


def test_an_eos_truncated_request_links_completely_over_its_program_of_record_and_a_short_record_still_fails_by_name():
    # committed world: r0 executed 12 steps (0..11); the workload cap is 32
    store, wreq = _world({"r0": (0, 12, None)})
    cap_steps = [dict(r, steps=LK.linkage_horizon(0, 32, None), max_tokens=32, program_steps=None) for r in wreq]
    lk_cap = _lk(store, cap_steps)
    assert lk_cap["result"] is False and lk_cap["unlinked_n"] == 20 and lk_cap["unlinked"][0]["engine_step"] == 12   # the old reading: steps 12..31 named
    of_record = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    eos = {100 + 7 * 11}                                                         # the declared stop rule: the fixture's last committed id
    lk = _lk(store, of_record, eos)
    assert lk["result"] is True and lk["pairs"] == 12 and lk["linked"] == 12
    assert lk["horizon"]["r0"] == {"linked_steps": 12, "declared_cap": 32, "program_steps": 12,
                                   "eos_of_record": {"last_step": 11, "sampled_id": 177, "declared_eos": [177], "verified": True, "lag": 0, "eos_step": 11, "eos_id": 177,
                                                     "placeholder_steps": [], "placeholder_input_ids": []}}
    # the Program of record says 14 steps but the record committed 12: steps 12, 13 unlinked by name (fewer committed than of record)
    short = [dict(r, steps=LK.linkage_horizon(0, 32, 14), max_tokens=32, program_steps=14) for r in wreq]
    lk2 = _lk(store, short, eos)
    assert lk2["result"] is False and [u["engine_step"] for u in lk2["unlinked"][:2]] == [12, 13] and lk2["unlinked"][0]["kind"] == "chain"
    assert lk2["unlinked"][2]["kind"] == "eos" and "EOS step 13's sampled id is not committed" in lk2["unlinked"][2]["why"]   # and the horizon's end is unverifiable



# ------------------------------------------------------------------------------------------ hardenings

def _truncated(eos, lag=0):
    store, wreq = _world({"r0": (0, 12, None)})
    req = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    return store, LK.boundary_linkage({"entries": []}, lambda *a: None, req, embed_op_path=EMB, store=store, eos_of_record=eos, eos_lag_rows=lag)


def test_an_early_stopped_request_must_end_on_a_declared_eos_id_at_the_horizons_last_step():
    # lag 0 (async_scheduling attested False): the EOS is the horizon's last step
    last_id = 100 + 7 * 11                                                        # the fixture's committed sampled id at step 11
    store, lk = _truncated({last_id, 2})
    assert lk["result"] is True and lk["horizon"]["r0"]["eos_of_record"] == {"last_step": 11, "sampled_id": last_id, "declared_eos": [2, last_id], "verified": True,
                                                                              "lag": 0, "eos_step": 11, "eos_id": last_id, "placeholder_steps": [], "placeholder_input_ids": []}
    # the last committed id is not a declared EOS id: a dropped / aborted request recorded as an early stop -> unlinked by name
    store, lk2 = _truncated({2})
    assert lk2["result"] is False and lk2["unlinked"][0]["kind"] == "eos" and lk2["unlinked"][0]["engine_step"] == 11
    assert f"committed sampled id {last_id} at the EOS step 11 (= last step 11 - lag 0) is not a declared EOS id [2]" in lk2["why"]
    assert lk2["horizon"]["r0"]["eos_of_record"]["verified"] is False
    # no declared stop rule -> fail closed, by name
    store, lk3 = _truncated(None)
    assert lk3["result"] is False and "declares no stop rule (eos_token_ids)" in lk3["why"] and "fail closed" in lk3["why"]
    # ignore_eos: the declared set is EMPTY -> every early stop fails
    store, lk4 = _truncated([])
    assert lk4["result"] is False and lk4["unlinked"][0]["kind"] == "eos"
    # a request that reaches its cap has no eos_of_record entry and needs no stop rule
    store5, wreq5 = _world({"r0": (0, 4, None)})
    full = [dict(r, steps=LK.linkage_horizon(0, 4, 4), max_tokens=4, program_steps=4) for r in wreq5]
    lk5 = _lk(store5, full)
    assert lk5["result"] is True and "eos_of_record" not in lk5["horizon"]["r0"]


def _sys_int3_earlystop_world(eos_at_10=0, ph_sample=504, ph_input=None):
    """An early-stop record in miniature: served T_r 11 -> Program of record LP32_T11, 12 steps 0..11, cap 32;
    async_scheduling attested True (LAG 1): the EOS 0 is SAMPLED at step 10, step 11 is the placeholder fed the EOS whose sample
    504 ('The') the driver discards.  `ph_input`: forge the placeholder's committed embed input (default: the EOS, the honest record)."""
    store, wreq = _world({"r0": (0, 12, None)})
    store.blocks[("r0", 10, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([eos_at_10], dtype=np.uint64)
    store.blocks[("r0", 11, EMB, IX.IDS_MEMBER)] = np.asarray([eos_at_10 if ph_input is None else ph_input], dtype=np.uint32)
    store.blocks[("r0", 11, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([ph_sample], dtype=np.uint64)
    req = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    return store, req


def test_padrev_m0517_under_the_attested_async_scheduler_the_eos_is_at_the_step_before_the_placeholder():
    # the honest lag-1 record (its expected eos_of_record) PASSes; reading steps[-1] = 11 -> 504 would fail it
    store, req = _sys_int3_earlystop_world()
    lk = _lk(store, req, {0}, lag=1)
    assert lk["result"] is True and lk["pairs"] == 12 and lk["linked"] == 12, lk["why"]
    e = lk["horizon"]["r0"]["eos_of_record"]
    assert {k: e[k] for k in ("last_step", "sampled_id", "eos_step", "eos_id", "lag", "verified")} == {"last_step": 11, "sampled_id": 504, "eos_step": 10, "eos_id": 0, "lag": 1, "verified": True}
    assert e["placeholder_steps"] == [11] and e["placeholder_input_ids"] == [0]          # the placeholder is fed the EOS
    # the same record read with lag 0 (the flag attested OFF) FAILS by name: 504 at the last step is not the EOS -- the lag is read, never assumed
    lk0 = _lk(store, req, {0}, lag=0)
    assert lk0["result"] is False and lk0["unlinked"][0]["kind"] == "eos" and "sampled id 504 at the EOS step 11 (= last step 11 - lag 0)" in lk0["why"]
    # unattested (the header carries no resolved async_scheduling): refused by name, never defaulted to 0 or 1
    lkn = _lk(store, req, {0}, lag=None)
    assert lkn["result"] is False and lkn["unlinked"][0]["kind"] == "eos" and "async_scheduling unattested)" in lkn["why"] and "refuse by name" in lkn["why"]
    assert lkn["horizon"]["r0"]["eos_of_record"]["lag"] is None and lkn["horizon"]["r0"]["eos_of_record"]["verified"] is False
    # a lag outside {0, 1}: refused by name
    assert "not in {0, 1}" in _lk(store, req, {0}, lag=2)["why"]


def test_padrev_m0517_lag_1_negatives_name_the_eos_step_and_the_placeholder():
    # (a) lag 1 but step 10's sampled id is NOT a declared EOS (the request was dropped / aborted, or the EOS is really at 11 and no placeholder ran)
    store, req = _sys_int3_earlystop_world(eos_at_10=333)
    lk = _lk(store, req, {0}, lag=1)
    assert lk["result"] is False and lk["unlinked"][0] == {"request_id": "r0", "engine_step": 10, "kind": "eos",
                                                           "why": "r0 linked over 12 of its declared cap 32 steps but its committed sampled id 333 at the EOS step 10 (= last step 11 - lag 1) is not a declared EOS id [0]"}
    # (b) the EOS at 10 is honest but the placeholder step is fed something other than the EOS: the chain names step 11 first, the eos rule names it too
    store, req = _sys_int3_earlystop_world(ph_input=7)
    lk = _lk(store, req, {0}, lag=1)
    assert lk["result"] is False and [u["kind"] for u in lk["unlinked"]] == ["chain", "eos"] and lk["unlinked"][0]["engine_step"] == 11
    assert "placeholder step(s) [11] after the EOS at step 10 are fed [7], not the EOS id 0" in lk["unlinked"][1]["why"]
    # (c) a one-step horizon under lag 1 has no EOS step
    store, wreq = _world({"r0": (0, 1, None)})
    req1 = [dict(r, steps=[0], max_tokens=32, program_steps=1) for r in wreq]
    lk = _lk(store, req1, {0}, lag=1)
    assert lk["result"] is False and "fewer steps than the attested EOS lag 1 + 1, no EOS step exists" in lk["why"]
    # (d) the declared EOS set is None / empty: fail closed regardless of the lag
    store, req = _sys_int3_earlystop_world()
    assert "declares no stop rule (eos_token_ids)" in _lk(store, req, None, lag=1)["why"]
    assert _lk(store, req, [], lag=1)["unlinked"][0]["kind"] == "eos"
    # (e) a request that reaches its cap needs neither a stop rule nor a lag
    store5, wreq5 = _world({"r0": (0, 4, None)})
    full = [dict(r, steps=LK.linkage_horizon(0, 4, 4), max_tokens=4, program_steps=4) for r in wreq5]
    assert _lk(store5, full, None, lag=None)["result"] is True


def test_padrev_m0517_the_lag_is_read_through_gp01s_one_reader_of_the_attested_field():
    # commit_delta feeds boundary_linkage the resolved async_scheduling through program.workload.eos_lag_rows_declared: one field, one reader
    from verity_vllm.program.workload import eos_lag_rows_declared
    assert eos_lag_rows_declared({"async_scheduling": True}) == 1 and eos_lag_rows_declared({"async_scheduling": False}) == 0
    assert eos_lag_rows_declared({"async_scheduling": "true"}) == 1
    assert eos_lag_rows_declared({"async_scheduling": None}) is None and eos_lag_rows_declared({}) is None and eos_lag_rows_declared(None) is None
    from pathlib import Path
    src = "".join((Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / f"{m}.py").read_text() for m in ("commit", "c2_replay"))      # torch-free: the wiring read as text
    assert 'eos_lag_rows_declared({"async_scheduling": (versions.get("engine_kwargs") or {}).get("async_scheduling")})' in src
    assert "eos_lag_rows=_lag" in src

def test_a_committed_seed_word_with_no_declared_source_is_unlinked_by_name_not_not_applicable():
    # a greedy-declared workload (no seed) whose record still carries runner.sampler/seed words: a prescribed input without an approved source
    store, wreq = _world({"r0": (0, 3, None)})
    for s in range(3):
        store.blocks[("r0", s, IX.SAMPLER_OP, IX.SEED_PARAM)] = np.asarray([12345], dtype=np.uint64)
    lk = _lk(store, wreq)
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "seed" and "NO declared workload seed" in lk["why"] and "3 (step, request) pair(s)" in lk["why"]
    assert lk["prescribed"]["seed"]["committed_without_source"] == 3 and "why" not in lk["prescribed"]["seed"]
    # the genuinely greedy record (no seed word anywhere) is still 'not applicable'
    store2, wreq2 = _world({"r0": (0, 3, None)})
    lk2 = _lk(store2, wreq2)
    assert lk2["result"] is True and "not applicable" in lk2["prescribed"]["seed"]["why"]


# ------------------------------------------------------------------------------- one engine schedule of record

def test_arrivals_of_record_follow_the_observed_schedule_only_when_the_population_was_placed_on_it():
    decl = {"r0": 0, "r1": 7}
    obs_sched = {"use": "observed", "observed_arrive_steps": {"r0": 0, "r1": 3}, "declared_arrive_steps": decl, "all_equal_declared": False}
    assert LK.arrivals_of_record(decl, obs_sched) == ({"r0": 0, "r1": 3}, True)                       # the gap workload: r1 admitted at 3
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps=decl)) == (decl, False)  # observed == declared: nothing moves
    assert LK.arrivals_of_record(decl, dict(obs_sched, use="declared")) == (decl, False)              # the declaration stands (coverage names why)
    assert LK.arrivals_of_record(decl, None) == (decl, False) and LK.arrivals_of_record(decl, {}) == (decl, False)
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps={"r0": 0, "r1": 3, "r9": 5})) == ({"r0": 0, "r1": 3}, True)   # unknown ids ignored
    # the commit_delta wiring: the replay/linkage (and serving rows') _arr goes through this one rule, fed the binding record's schedule
    from pathlib import Path
    from types import SimpleNamespace
    from verity_vllm.pipeline import c2_replay as C2
    reqs = [{"request_id": "a", "index": 0, "arrive_step": 0}, {"request_id": "b", "index": 1, "arrive_step": 7}]
    site = SimpleNamespace(reqs=reqs, row={"commit": {"binding": {"coverage": {"schedule": obs_sched}}}})
    assert C2.arrivals(site, C2.request_labels(reqs)) == ({"r0": 0, "r1": 3}, True)
    assert C2.arrivals(SimpleNamespace(reqs=reqs, row={"commit": {}}), C2.request_labels(reqs)) == (decl, False)
    src = (Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / "c2_replay.py").read_text()
    assert '_arr, _moved = arrivals(s, _rl)' in src
