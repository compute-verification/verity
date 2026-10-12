"""[R16 pin, coord ruling 36 (rev M-0410, reqval M-0417 (b), stoch M-0446)] the stochastic Program's PRESCRIBED sampler inputs are linked
to their approved source at Commit, independently of whether a random replay sample happens to expose a substitution:

  * SEED -- the committed `runner.sampler/seed` word at every (step, request) with a committed sampled id == the DECLARED workload seed
    of that request (i64 exact; the kernel reads seeds[req] + pos[req]: one word per request, constant across its steps, no derivation);
  * SPLITS (REQ-03 derived-S Programs) -- the committed `runner.sampler/splits` word at step s == SplitsFor_v1(live(s), num_SMs of
    record) with live(s) = the number of requests with a committed sampled id at s (== the sampler batch's rows);
  * unlinked (missing word, unequal word, no S of record) -> the boundary linkage FAILs BY NAME (kind 'seed' / 'splits'), the same path
    as an unlinked token boundary; a greedy row (no declared seed, constant S) checks nothing and says so, its verdict byte-identical.

Why this is required beside stoch's evaluator (cfb3cef2a, rev M-0444): the form-(A) evaluator makes the COMMITTED seed load-bearing, not
the DECLARED one -- a prover that commits ids drawn with its own seed replays consistently.  The linkage ties the committed word to the
workload of record.  Fixtures: a `_Store` of committed words by identity (the stoch test's shape), B=1 and B=3 with staggered arrivals
so live(s) changes and SplitsFor_v1's S changes with it (142 SMs: live 1..4 -> 32, 5..8 -> 16; 132 SMs on Hopper likewise 1..4 -> 32).
"""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.check.replay import index as IX, linkage as LK
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.registry.topp_geometry import splits_for_v1

EMB = "model.embed_tokens"
NSM = 142


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


def _world(reqs: dict[str, tuple[int, int, int]], *, nsm: int = NSM, seed_word=None, splits_word=None):
    """reqs: rid -> (arrive, n_steps, declared seed).  An honest committed world: prompt ids at arrival, chained ids, one seed word per
    live (step, request) == the declared seed, one splits word == SplitsFor_v1(live, nsm).  `seed_word` / `splits_word`: overrides
    {(rid, step): word} to forge a committed word (the served run consumed something else than declared)."""
    blocks: dict = {}
    wreq = []
    live_at: dict[int, list[str]] = {}
    for rid, (arrive, n, seed) in reqs.items():
        prompt = [11, 12, 13]
        steps = list(range(arrive, arrive + n))
        for s in steps:
            live_at.setdefault(s, []).append(rid)
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
    for s, rids in live_at.items():
        for rid in rids:
            if reqs[rid][2] is not None:
                w = (splits_word or {}).get((rid, s), splits_for_v1(len(rids), nsm))
                blocks[(rid, s, IX.SAMPLER_OP, "splits")] = np.asarray([w], dtype=np.uint32)
    return _Store(blocks), wreq


def _lk(store, wreq, spr, eos=None, lag=0):
    return LK.boundary_linkage({"entries": []}, lambda *a: None, wreq, embed_op_path=EMB, splits_of_record=spr, store=store, eos_of_record=eos, eos_lag_rows=lag)


# ----------------------------------------------------------------------------------------------------------------------- positives

def test_b1_honest_stochastic_row_links_seed_and_splits_by_name():
    store, wreq = _world({"r0": (0, 4, 1077402498)})
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is True and lk["pairs"] == 4 and lk["linked"] == 4 and lk["unlinked_n"] == 0, lk
    p = lk["prescribed"]
    assert p["seed"] == {"rule": LK.SEED_RULE, "requests_declared": 1, "checked": 4, "linked": 4}
    assert p["splits"]["num_sms"] == NSM and p["splits"]["steps"] == 4 and p["splits"]["checked"] == 4 == p["splits"]["linked"]
    assert p["splits"]["schedule"] == {"0": 32, "1": 32, "2": 32, "3": 32}
    assert "prescribed-input linkage (seed; splits)" in lk["method"]


def test_b3_staggered_arrivals_change_the_live_count_and_the_s_of_record_follows():
    # r0 arrives 0 (6 steps), r1 arrives 2 (4 steps), r2 arrives 3 (2 steps): live = 1,1,2,3,3,2 -> all 32 at 142 SMs; at 4 SMs the
    # verbatim doubling loop (double while S*2*live <= num_SMs) gives S = 4,4,2,1,1,2 -- the schedule is a function of the committed live
    # count, never a constant
    reqs = {"r0": (0, 6, 5), "r1": (2, 4, 6), "r2": (3, 2, 7)}
    store, wreq = _world(reqs, nsm=4)
    lk = _lk(store, wreq, {r: {"derived": 4} for r in reqs})
    assert lk["result"] is True, lk
    sched = lk["prescribed"]["splits"]["schedule"]
    assert sched == {"0": 4, "1": 4, "2": 2, "3": 1, "4": 1, "5": 2}
    assert lk["prescribed"]["splits"]["checked"] == 12 == lk["prescribed"]["splits"]["linked"]         # 6+4+2 live (step, request) pairs
    assert lk["prescribed"]["seed"]["checked"] == 12 == lk["prescribed"]["seed"]["linked"]


def test_greedy_row_checks_nothing_and_says_so_verdict_unchanged():
    store, wreq = _world({"r0": (0, 3, None), "r1": (0, 3, None)})
    lk = _lk(store, wreq, {"r0": None, "r1": None})
    assert lk["result"] is True and lk["pairs"] == 6 and lk["linked"] == 6
    p = lk["prescribed"]
    assert p["seed"]["checked"] == 0 and "not applicable" in p["seed"]["why"]
    assert p["splits"]["checked"] == 0 and "not applicable" in p["splits"]["why"]
    assert "prescribed-input" not in lk["method"]                                                        # the greedy method text is the old one
    # an R15 constant-S Program (splits = 32) on a seeded row: the seed is linked, splits are not a rule
    store2, wreq2 = _world({"r0": (0, 3, 9)})
    lk2 = _lk(store2, wreq2, {"r0": 32})
    assert lk2["result"] is True and lk2["prescribed"]["seed"]["linked"] == 3 and lk2["prescribed"]["splits"]["checked"] == 0
    assert lk2["method"].endswith("prescribed-input linkage (seed)")


# ----------------------------------------------------------------------------------------------------------------------- negatives

def test_a_committed_seed_that_is_not_the_declared_one_fails_by_name_at_the_step():
    # the served run consumed its own seed at step 2 of r0 (the evaluator alone would replay it consistently -- rev M-0444)
    store, wreq = _world({"r0": (0, 4, 1077402498)}, seed_word={("r0", 2): 4242})
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is False and lk["unlinked_n"] == 1
    u = lk["unlinked"][0]
    assert u["kind"] == "seed" and u["request_id"] == "r0" and u["engine_step"] == 2
    assert lk["why"] == "committed runner.sampler/seed 4242 != declared workload seed 1077402498 of r0 at step 2 (prescribed input, ruling 36)"
    assert lk["prescribed"]["seed"] == {"rule": LK.SEED_RULE, "requests_declared": 1, "checked": 4, "linked": 3}


def test_a_negative_declared_seed_is_compared_as_i64_exactly():
    store, wreq = _world({"r0": (0, 2, -17)})
    assert _lk(store, wreq, {"r0": {"derived": NSM}})["result"] is True
    store2, wreq2 = _world({"r0": (0, 2, -17)}, seed_word={("r0", 1): -18})
    lk = _lk(store2, wreq2, {"r0": {"derived": NSM}})
    assert lk["result"] is False and "committed runner.sampler/seed -18 != declared workload seed -17" in lk["why"]


def test_a_missing_seed_word_where_a_sampled_id_is_committed_fails_by_name():
    store, wreq = _world({"r0": (0, 3, 5)})
    del store.blocks[("r0", 1, IX.SAMPLER_OP, IX.SEED_PARAM)]
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "seed" and lk["unlinked"][0]["engine_step"] == 1
    assert "prescribed seed of r0 at step 1 not committed" in lk["why"]


def test_a_seed_word_that_changes_across_the_requests_steps_is_named_at_the_first_differing_step():
    # stoch M-0446: the seed is constant per request; a per-step 'derivation' the run applied on its own is a substitution at every step
    store, wreq = _world({"r0": (0, 4, 100)}, seed_word={("r0", 1): 101, ("r0", 2): 102, ("r0", 3): 103})
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is False and lk["unlinked_n"] == 3 and [u["engine_step"] for u in lk["unlinked"]] == [1, 2, 3]
    assert lk["prescribed"]["seed"]["linked"] == 1


def test_a_committed_splits_word_that_is_not_splitsfor_of_the_committed_live_count_fails_by_name():
    reqs = {"r0": (0, 6, 5), "r1": (2, 4, 6), "r2": (3, 2, 7)}
    store, wreq = _world(reqs, nsm=4, splits_word={("r1", 3): 2})                 # live 3 at step 3 -> S 1; the run committed 2
    lk = _lk(store, wreq, {r: {"derived": 4} for r in reqs})
    assert lk["result"] is False and lk["unlinked_n"] == 1
    u = lk["unlinked"][0]
    assert u["kind"] == "splits" and u["request_id"] == "r1" and u["engine_step"] == 3
    assert lk["why"] == "committed runner.sampler/splits 2 != SplitsFor_v1(live 3, num_SMs 4) = 1 for r1 at step 3 (prescribed input, ruling 36)"


def test_an_omitted_request_moves_the_live_count_and_every_batch_mates_s_is_named():
    # the served run sampled r2 at step 3 but committed no sampled id for it: live(3) reads 2 instead of 3 -> S* = 2 at 4 SMs while the
    # honest words say 1 -- r0 and r1 are both named at step 3 (the committed live count is what the rule reads; an omission shows up on
    # its batch-mates' S, which is the point of deriving S from the committed record rather than trusting a committed count)
    reqs = {"r0": (0, 6, 5), "r1": (2, 4, 6), "r2": (3, 2, 7)}
    store, wreq = _world(reqs, nsm=4)
    del store.blocks[("r2", 3, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)]
    lk = _lk(store, wreq, {r: {"derived": 4} for r in reqs})
    assert lk["result"] is False
    kinds = {(u["kind"], u["request_id"], u["engine_step"]) for u in lk["unlinked"]}
    assert ("splits", "r0", 3) in kinds and ("splits", "r1", 3) in kinds
    assert ("chain", "r2", 4) in kinds                                            # and the token chain names the omitted id, as before


def test_a_missing_splits_word_on_a_derived_s_program_fails_by_name():
    store, wreq = _world({"r0": (0, 3, 5)})
    del store.blocks[("r0", 2, IX.SAMPLER_OP, "splits")]
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "splits" and "prescribed splits of r0 at step 2 not committed" in lk["why"]


def test_num_sms_of_record_disagreeing_across_the_request_programs_is_named_never_averaged():
    store, wreq = _world({"r0": (0, 2, 5), "r1": (0, 2, 6)})
    lk = _lk(store, wreq, {"r0": {"derived": 142}, "r1": {"derived": 132}})
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "splits" and "num_SMs of record disagree" in lk["why"] and "[132, 142]" in lk["why"]


def test_a_live_request_whose_program_does_not_declare_derived_s_is_named():
    store, wreq = _world({"r0": (0, 2, 5), "r1": (0, 2, 6)})
    lk = _lk(store, wreq, {"r0": {"derived": NSM}, "r1": 32})
    assert lk["result"] is False and any(u["kind"] == "splits" and u["request_id"] == "r1" and "does not declare derived S" in u["why"] for u in lk["unlinked"])


def test_a_request_with_a_declared_seed_and_no_committed_sampled_id_is_unlinked_by_name():
    store, wreq = _world({"r0": (0, 3, 5)})
    for s in range(3):
        del store.blocks[("r0", s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)]
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    assert lk["result"] is False and any(u["kind"] == "seed" and "linked at no step" in u["why"] for u in lk["unlinked"])


def test_the_verdict_names_the_prescribed_input_through_the_boundary_linkage_path():
    from verity_vllm.check.verdict import commit_verdict
    store, wreq = _world({"r0": (0, 4, 1077402498)}, seed_word={("r0", 2): 4242})
    lk = _lk(store, wreq, {"r0": {"derived": NSM}})
    base = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
            "binding": {"context": {"ok": True}, "coverage": {"ok": True}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
            "weights_pin": {"result": True, "compared": 98, "mismatched": [], "missing": []},
            "sampled_replay": {"result": True, "grade": "complete", "why": None}, "boundary_linkage": lk}
    ok, why = commit_verdict(base, True, min_openings=64)
    assert not ok and why.startswith("C2 boundary linkage: committed runner.sampler/seed 4242 != declared workload seed 1077402498 of r0 at step 2"), why
    assert why.endswith("(1 unlinked)")


@pytest.mark.parametrize("live,nsm,want", [(1, 142, 32), (4, 142, 32), (5, 142, 16), (8, 142, 16), (9, 142, 8), (1, 132, 32), (5, 132, 16), (17, 132, 4)])
def test_the_s_of_record_is_the_registry_relation_not_a_table(live, nsm, want):
    assert splits_for_v1(live, nsm) == want


# ------------------------------------------------------------------------------------- ruling 35 sixth consumer: the linkage horizon

def test_linkage_horizon_is_the_program_of_records_steps_capped_by_the_declared_cap():
    # B=1 eos-truncated (sys M-0466): cap 32, served-shape Program build_request_LP32_T11 has 12 steps -> 12 steps linked, not 32
    assert LK.linkage_horizon(0, 32, 12) == list(range(12))
    assert LK.linkage_horizon(5, 32, 12) == list(range(5, 17))
    assert LK.linkage_horizon(0, 32, None) == list(range(32))                     # unknown Program steps: the cap (older records)
    assert LK.linkage_horizon(0, 8, 12) == list(range(8))                         # a Program longer than the cap: the cap
    assert LK.linkage_horizon(3, 0, 12) == []


def test_an_eos_truncated_request_links_completely_over_its_program_of_record_and_a_short_record_still_fails_by_name():
    # committed world: r0 executed 12 steps (0..11); the workload cap is 32
    store, wreq = _world({"r0": (0, 12, None)})
    cap_steps = [dict(r, steps=LK.linkage_horizon(0, 32, None), max_tokens=32, program_steps=None) for r in wreq]
    lk_cap = _lk(store, cap_steps, {"r0": None})
    assert lk_cap["result"] is False and lk_cap["unlinked_n"] == 20 and lk_cap["unlinked"][0]["engine_step"] == 12   # the old reading: steps 12..31 named
    of_record = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    eos = {100 + 7 * 11}                                                         # the declared stop rule: the fixture's last committed id
    lk = _lk(store, of_record, {"r0": None}, eos)
    assert lk["result"] is True and lk["pairs"] == 12 and lk["linked"] == 12
    assert lk["horizon"]["r0"] == {"linked_steps": 12, "declared_cap": 32, "program_steps": 12,
                                   "eos_of_record": {"last_step": 11, "sampled_id": 177, "declared_eos": [177], "verified": True, "lag": 0, "eos_step": 11, "eos_id": 177,
                                                     "placeholder_steps": [], "placeholder_input_ids": []}}
    # the Program of record says 14 steps but the record committed 12: steps 12, 13 unlinked by name (fewer committed than of record)
    short = [dict(r, steps=LK.linkage_horizon(0, 32, 14), max_tokens=32, program_steps=14) for r in wreq]
    lk2 = _lk(store, short, {"r0": None}, eos)
    assert lk2["result"] is False and [u["engine_step"] for u in lk2["unlinked"][:2]] == [12, 13] and lk2["unlinked"][0]["kind"] == "chain"
    assert lk2["unlinked"][2]["kind"] == "eos" and "EOS step 13's sampled id is not committed" in lk2["unlinked"][2]["why"]   # and the horizon's end is unverifiable



# ------------------------------------------------------------------------------------------ rev M-0485 / M-0472 hardenings

def _truncated(eos, lag=0):
    store, wreq = _world({"r0": (0, 12, None)})
    req = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    return store, LK.boundary_linkage({"entries": []}, lambda *a: None, req, embed_op_path=EMB, splits_of_record={"r0": None}, store=store, eos_of_record=eos, eos_lag_rows=lag)


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
    # no declared stop rule -> fail closed (F-pm-24), by name
    store, lk3 = _truncated(None)
    assert lk3["result"] is False and "declares no stop rule (eos_token_ids)" in lk3["why"] and "fail closed (F-pm-24)" in lk3["why"]
    # ignore_eos: the declared set is EMPTY -> every early stop fails
    store, lk4 = _truncated([])
    assert lk4["result"] is False and lk4["unlinked"][0]["kind"] == "eos"
    # a request that reaches its cap has no eos_of_record entry and needs no stop rule
    store5, wreq5 = _world({"r0": (0, 4, None)})
    full = [dict(r, steps=LK.linkage_horizon(0, 4, 4), max_tokens=4, program_steps=4) for r in wreq5]
    lk5 = _lk(store5, full, {"r0": None})
    assert lk5["result"] is True and "eos_of_record" not in lk5["horizon"]["r0"]


def _sys_int3_earlystop_world(eos_at_10=0, ph_sample=504, ph_input=None):
    """[padrev M-0517] sys M-0402's int-3 earlystop record in miniature: served T_r 11 -> Program of record LP32_T11, 12 steps 0..11, cap 32;
    async_scheduling attested True (RULING 21, LAG 1): the EOS 0 is SAMPLED at step 10, step 11 is the placeholder fed the EOS whose sample
    504 ('The') the driver discards.  `ph_input`: forge the placeholder's committed embed input (default: the EOS, the honest record)."""
    store, wreq = _world({"r0": (0, 12, None)})
    store.blocks[("r0", 10, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([eos_at_10], dtype=np.uint64)
    store.blocks[("r0", 11, EMB, IX.IDS_MEMBER)] = np.asarray([eos_at_10 if ph_input is None else ph_input], dtype=np.uint32)
    store.blocks[("r0", 11, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([ph_sample], dtype=np.uint64)
    req = [dict(r, steps=LK.linkage_horizon(0, 32, 12), max_tokens=32, program_steps=12) for r in wreq]
    return store, req


def test_padrev_m0517_under_the_attested_async_scheduler_the_eos_is_at_the_step_before_the_placeholder():
    # the honest lag-1 record (sys M-0402 / padrev M-0517's expected eos_of_record) PASSes; f8dc277a1 read steps[-1] = 11 -> 504 and failed it
    store, req = _sys_int3_earlystop_world()
    lk = _lk(store, req, {"r0": None}, {0}, lag=1)
    assert lk["result"] is True and lk["pairs"] == 12 and lk["linked"] == 12, lk["why"]
    e = lk["horizon"]["r0"]["eos_of_record"]
    assert {k: e[k] for k in ("last_step", "sampled_id", "eos_step", "eos_id", "lag", "verified")} == {"last_step": 11, "sampled_id": 504, "eos_step": 10, "eos_id": 0, "lag": 1, "verified": True}
    assert e["placeholder_steps"] == [11] and e["placeholder_input_ids"] == [0]          # the placeholder is fed the EOS
    # the same record read with lag 0 (the flag attested OFF) FAILS by name: 504 at the last step is not the EOS -- the lag is read, never assumed
    lk0 = _lk(store, req, {"r0": None}, {0}, lag=0)
    assert lk0["result"] is False and lk0["unlinked"][0]["kind"] == "eos" and "sampled id 504 at the EOS step 11 (= last step 11 - lag 0)" in lk0["why"]
    # unattested (the header carries no resolved async_scheduling): refused by name, never defaulted to 0 or 1
    lkn = _lk(store, req, {"r0": None}, {0}, lag=None)
    assert lkn["result"] is False and lkn["unlinked"][0]["kind"] == "eos" and "async_scheduling unattested, RULING 21" in lkn["why"] and "refuse by name" in lkn["why"]
    assert lkn["horizon"]["r0"]["eos_of_record"]["lag"] is None and lkn["horizon"]["r0"]["eos_of_record"]["verified"] is False
    # a lag outside {0, 1}: refused by name
    assert "not in {0, 1}" in _lk(store, req, {"r0": None}, {0}, lag=2)["why"]


def test_padrev_m0517_lag_1_negatives_name_the_eos_step_and_the_placeholder():
    # (a) lag 1 but step 10's sampled id is NOT a declared EOS (the request was dropped / aborted, or the EOS is really at 11 and no placeholder ran)
    store, req = _sys_int3_earlystop_world(eos_at_10=333)
    lk = _lk(store, req, {"r0": None}, {0}, lag=1)
    assert lk["result"] is False and lk["unlinked"][0] == {"request_id": "r0", "engine_step": 10, "kind": "eos",
                                                           "why": "r0 linked over 12 of its declared cap 32 steps but its committed sampled id 333 at the EOS step 10 (= last step 11 - lag 1) is not a declared EOS id [0]"}
    # (b) the EOS at 10 is honest but the placeholder step is fed something other than the EOS: the chain names step 11 first, the eos rule names it too
    store, req = _sys_int3_earlystop_world(ph_input=7)
    lk = _lk(store, req, {"r0": None}, {0}, lag=1)
    assert lk["result"] is False and [u["kind"] for u in lk["unlinked"]] == ["chain", "eos"] and lk["unlinked"][0]["engine_step"] == 11
    assert "placeholder step(s) [11] after the EOS at step 10 are fed [7], not the EOS id 0" in lk["unlinked"][1]["why"]
    # (c) a one-step horizon under lag 1 has no EOS step
    store, wreq = _world({"r0": (0, 1, None)})
    req1 = [dict(r, steps=[0], max_tokens=32, program_steps=1) for r in wreq]
    lk = _lk(store, req1, {"r0": None}, {0}, lag=1)
    assert lk["result"] is False and "fewer steps than the attested EOS lag 1 + 1, no EOS step exists" in lk["why"]
    # (d) the declared EOS set is None / empty: fail closed regardless of the lag
    store, req = _sys_int3_earlystop_world()
    assert "declares no stop rule (eos_token_ids)" in _lk(store, req, {"r0": None}, None, lag=1)["why"]
    assert _lk(store, req, {"r0": None}, [], lag=1)["unlinked"][0]["kind"] == "eos"
    # (e) a request that reaches its cap needs neither a stop rule nor a lag
    store5, wreq5 = _world({"r0": (0, 4, None)})
    full = [dict(r, steps=LK.linkage_horizon(0, 4, 4), max_tokens=4, program_steps=4) for r in wreq5]
    assert _lk(store5, full, {"r0": None}, None, lag=None)["result"] is True


def test_padrev_m0517_the_lag_is_read_through_gp01s_one_reader_of_the_attested_field():
    # commit_delta feeds boundary_linkage the resolved async_scheduling through global_program.eos_lag_rows_declared: one field, one reader
    from verity_vllm.pipeline import global_program as GP
    assert GP.eos_lag_rows_declared({"async_scheduling": True}) == 1 and GP.eos_lag_rows_declared({"async_scheduling": False}) == 0
    assert GP.eos_lag_rows_declared({"async_scheduling": "true"}) == 1
    assert GP.eos_lag_rows_declared({"async_scheduling": None}) is None and GP.eos_lag_rows_declared({}) is None and GP.eos_lag_rows_declared(None) is None
    from pathlib import Path
    src = "".join((Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / f"{m}.py").read_text() for m in ("commit", "c2_replay"))      # torch-free: the wiring read as text
    assert 'eos_lag_rows_declared({"async_scheduling": (versions.get("engine_kwargs") or {}).get("async_scheduling")})' in src
    assert "eos_lag_rows=_lag" in src

def test_a_committed_seed_word_with_no_declared_source_is_unlinked_by_name_not_not_applicable():
    # a greedy-declared workload (no seed) whose record still carries runner.sampler/seed words: a prescribed input without an approved source
    store, wreq = _world({"r0": (0, 3, None)})
    for s in range(3):
        store.blocks[("r0", s, IX.SAMPLER_OP, IX.SEED_PARAM)] = np.asarray([12345], dtype=np.uint64)
    lk = _lk(store, wreq, {"r0": None})
    assert lk["result"] is False and lk["unlinked"][0]["kind"] == "seed" and "NO declared workload seed" in lk["why"] and "3 (step, request) pair(s)" in lk["why"]
    assert lk["prescribed"]["seed"]["committed_without_source"] == 3 and "why" not in lk["prescribed"]["seed"]
    # the genuinely greedy record (no seed word anywhere) is still 'not applicable'
    store2, wreq2 = _world({"r0": (0, 3, None)})
    lk2 = _lk(store2, wreq2, {"r0": None})
    assert lk2["result"] is True and "not applicable" in lk2["prescribed"]["seed"]["why"]


# ------------------------------------------------------------------------------- rev M-0570 (1): one engine schedule of record

def test_arrivals_of_record_follow_the_observed_schedule_only_when_the_population_was_placed_on_it():
    decl = {"r0": 0, "r1": 7}
    obs_sched = {"use": "observed", "observed_arrive_steps": {"r0": 0, "r1": 3}, "declared_arrive_steps": decl, "all_equal_declared": False}
    assert LK.arrivals_of_record(decl, obs_sched) == ({"r0": 0, "r1": 3}, True)                       # the gap workload: r1 admitted at 3
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps=decl)) == (decl, False)  # observed == declared: nothing moves
    assert LK.arrivals_of_record(decl, dict(obs_sched, use="declared")) == (decl, False)              # the declaration stands (coverage names why)
    assert LK.arrivals_of_record(decl, None) == (decl, False) and LK.arrivals_of_record(decl, {}) == (decl, False)
    assert LK.arrivals_of_record(decl, dict(obs_sched, observed_arrive_steps={"r0": 0, "r1": 3, "r9": 5})) == ({"r0": 0, "r1": 3}, True)   # unknown ids ignored
    # the commit_delta wiring: the replay/linkage _arr goes through this one rule, fed the binding record's schedule
    from pathlib import Path
    src = "".join((Path(__file__).resolve().parents[2] / "verity_vllm" / "pipeline" / f"{m}.py").read_text() for m in ("commit", "c2_replay"))
    assert '_arr, _moved = SR.arrivals_of_record(_arr, _sched_r)' in src and '.get("coverage") or {}).get("schedule")' in src
