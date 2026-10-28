"""The two T1 checks' recomputation paths against the functions they wrap, on synthetic inputs (laptop-safe, no fixture).

`replay_partition.recompute` re-derives the sampled-replay PARTITION from a `sampled_replay` record's stated parameters + the Programs; over
the synthetic worlds of `tests.check.test_sampled_replay` it must lift to exactly what `sampled_replay` itself recorded
(population, strata, draw, picks per (request, step), lifetime, phase anchors) -- and a changed seed / a changed Program must not.
`decomp_hashes.lift` is the contract over `match_decomp.json`-shaped documents: the fields the check compares, and only those.
"""
from __future__ import annotations

import copy

import pytest

from verity_vllm.check.replay import driver as SR
from tests.check import test_sampled_replay as T
from tests.check.test_sampled_replay import moe_world, world  # noqa: F401  -- fixtures

from tests.regression import resolver
from tests.regression.checks import decomp_hashes as DH
from tests.regression.checks import replay_partition as RP


def _programs(w):
    return {rid: w.inst for rid in w.prompts}


def _weights(w):
    def read(field, lo, hi):
        flo, _fhi = w.woff[field]
        return w.weights[flo + lo: flo + hi]
    return read


def _manifest_for(w) -> dict:
    return T.v2_manifest(w.inst, tuple(sorted(w.prompts)))


def _record(w, **kw) -> tuple[dict, dict]:
    man = _manifest_for(w)
    bmap, committed = w.bmap_and_reader()
    rep = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), workers=0, manifest=man, **kw)
    return rep, man


@pytest.mark.parametrize("seed,per_stratum,ring_slots", [(3, 2, None), (11, 1, 2)])
def test_the_partition_recomputed_from_the_record_lifts_to_the_record(world, seed, per_stratum, ring_slots):
    rep, man = _record(world, per_stratum=per_stratum, seed=seed, ring_slots=ring_slots)
    assert rep["population"]["vus"] > 0 and rep["population"]["strata_n"] > 1
    doc = RP.recompute(rep, _programs(world))
    exp, act = RP.lift(rep), RP.lift(doc)
    assert resolver.diff(exp, act) == []
    # the draw is the seed's: another seed moves the per-(request, step) picks (the strata sizes stay)
    other = RP.recompute(dict(rep, sample=dict(rep["sample"], seed=seed + 1)), _programs(world))
    d = resolver.diff(exp, RP.lift(other))
    assert d and all(".sample." in x or ".lifetime.sampled" in x for x in d), d
    assert all(".strata." not in x or x.endswith("sampled") or "population" not in x for x in d)


def test_the_partition_over_the_moe_world_lifts_to_the_record(moe_world):
    rep, man = _record(moe_world, per_stratum=2, seed=5)
    assert rep["population"]["strata_n"] > 1 and rep["population"]["by_family"]
    assert resolver.diff(RP.lift(rep), RP.lift(RP.recompute(rep, _programs(moe_world)))) == []


def test_a_changed_program_row_moves_a_count_by_name(world):
    rep, man = _record(world, per_stratum=2, seed=3)
    progs = {rid: copy.deepcopy(world.inst) for rid in world.prompts}
    inst = next(iter(progs.values()))
    inst["rows"] = inst["rows"][:-1]                                        # one Program row fewer
    for v in progs.values():
        v["rows"] = inst["rows"]
    d = resolver.diff(RP.lift(rep), RP.lift(RP.recompute(rep, progs)))
    assert any(x.startswith("$.population.vus") for x in d), d


def test_decomp_lift_is_the_contract_and_only_the_contract():
    doc = {"verdict": "PASS", "xreq_total": 3,
           "requests": {"r0": {"request_id": "r0", "canonical_sha256": {"a": "x", "b": "x", "equal": True}, "derived_digest": "d", "n_instances": 10,
                               "LP": 4, "T": 2, "token_selects": 3, "steps": 3, "prefill_chunks": 1, "xreq": {"counts": {"r1": 3}, "examples": ["prose"]},
                               "divergences": ["not part of the contract"]}},
           "attribution": {"how": {"meta": 10}, "unattributed": 0, "attributed": 10, "row_rule_disagreements": 0, "weight_only_shared": 0, "faults": ["x"]},
           "oracle": {"digest": "o", "instances": 10, "n_input_nodes": 5, "path": "/nowhere"}}
    lifted = DH.lift(doc, pins={"program_sha256": "p", "instances_sha256": "i"})
    assert lifted["requests"]["r0"]["xreq"] == {"counts": {"r1": 3}}
    assert "divergences" not in lifted["requests"]["r0"] and "faults" not in lifted["attribution"] and "path" not in lifted["oracle"]
    assert lifted["fold_record_pins"] == {"program_sha256": "p", "instances_sha256": "i"}
    assert resolver.diff(lifted, DH.lift(copy.deepcopy(doc), pins={"program_sha256": "p", "instances_sha256": "i"})) == []
    moved = copy.deepcopy(doc)
    moved["requests"]["r0"]["canonical_sha256"]["b"] = "y"
    assert resolver.diff(lifted, DH.lift(moved, pins=lifted["fold_record_pins"])) == ['$.requests.r0.canonical_sha256.b: expected "x" != actual "y"']
