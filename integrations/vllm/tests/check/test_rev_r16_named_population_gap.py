"""The population moves the by-name gaps (VUs fed by MoE block-stream planes; VUs fed by an interior producer without an evaluator, e.g.
Qwen2.5's BiasAdd over Gemm_v2) out of the sampled population (`population.not_evaluable`), so the replay grades COMPLETE.  Such a gap is
accepted on a PASS ONLY when form (B) (the Match-oracle compare) is complete over the values, and the population move holds under the same
condition.  Before the fix the condition was enforced only in the PARTIAL branch (`_partial_replay_named_gap`) -- unreachable after the
population move -- and a record with hundreds of by-name-excluded VUs that HAVE committed members and NO form (B) rendered a plain PASS.

Behaviour of record (this file; two strict xfails flipped to plain tests): when `sampled_replay.population.not_evaluable` carries a
by-name class (plane-fed / interior-composed-v2) with n > 0, the PASS requires the same complete form (B) the partial rule requires and
carries the named gap in its text; without it the verdict FAILs 'C2 of record not established ...'.  The pre-existing
NOT_YET exclusions (RoPE_v1 no evaluator, Attention_v3 state-linked) are the standard's stated applicability domain and are NOT gated.

Form (B) compares the Match record's dispatcher-level module outputs and has NO oracle family for the moe/ block-stream planes
(oracle_compare.NO_ORACLE_FAMILY_PREFIXES): a complete (B) covers an interior-composed gap (its module's returned tensor IS an instance
output), never a PLANE-FED one, whose planes neither form value-checks: that FAILs by name whatever (B) says (C1 + C3 hold on them)."""
from __future__ import annotations

import pytest

from verity_vllm.check.c2_rules import _complete_replay_population_gap, _form_b_complete_over_values
from verity_vllm.check.commit_pass import commit_verdict


def _bnd(n_req):
    return {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


BASE = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
        "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
        "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
        "weights_pin": {"result": True, "compared": 98, "mismatched": [], "missing": []}}

K_06G = "fed by interior producer(s) without an evaluator (Gemm_v2 (model.layers.*.self_attn.qkv_proj)): composition not possible in this replay (MAN-06g, v2)"
K_06F = "fed by moe_block_stream planes (SiluMul_v1@experts): row -> plane mapping not in this replay (MAN-06f, v2)"
K_ROPE = "no derived_rows evaluator for RoPE_v1 yet (v2)"

# a COMPLETE replay whose population excluded, BY NAME, 140 BiasAdd VUs (Qwen2.5: every layer's qkv_proj output) and 64 MoE-plane-fed VUs
SR_COMPLETE_WITH_NAMED_EXCLUSIONS = {
    "result": True, "grade": "complete", "why": None,
    "sample": {"evaluated": 598, "equal": 598, "mismatch_n": 0, "not_evaluated_n": 0},
    "strata_unevaluated": [], "unevaluated_strata": [],
    "population": {"vus": 2000, "not_evaluable": {K_06G: 140, K_06F: 64, K_ROPE: 32},
                   "not_evaluable_by_family": {"BiasAdd_v1 (fed by interior Gemm_v2: composed evaluator, v2)": 140,
                                               "SiluMul_v1 (fed by moe_block_stream planes: SiluMul_v1@experts)": 64, "RoPE_v1": 32}}}

# the same replay on a dense Qwen2.5 row: the interior-composed gap only (no MoE planes) -- the case form (B) CAN cover
SR_COMPLETE_INTERIOR_ONLY = {**SR_COMPLETE_WITH_NAMED_EXCLUSIONS,
                             "population": {"vus": 2000, "not_evaluable": {K_06G: 140, K_ROPE: 32},
                                            "not_evaluable_by_family": {"BiasAdd_v1 (fed by interior Gemm_v2: composed evaluator, v2)": 140, "RoPE_v1": 32}}}

# the same replay on a dense row: only the NOT_YET exclusions (the standard's applicability domain), no plane-fed or interior-composed gap
SR_COMPLETE_NOT_YET_ONLY = {**SR_COMPLETE_WITH_NAMED_EXCLUSIONS,
                            "population": {"vus": 2000, "not_evaluable": {K_ROPE: 32, "state-linked K/V history (Attention_v3)": 16}, "not_evaluable_by_family": {"RoPE_v1": 32, "Attention_v3": 16}}}

OC_COMPLETE = {"result": True, "grade": "complete", "compared": 3, "equal": 3, "mismatch_n": 0, "complete": True,
               "compared_by_family": {"instance_outputs": 2, "input_token_ids": 1}, "not_compared_by_family": {},
               "attribution": {"complete": True, "pairs": 3, "unverified_n": 0, "ambiguous_steps": []}}
OC_PARTIAL = {**OC_COMPLETE, "result": None, "grade": "partial", "complete": False, "steps_not_snapshotted": [2, 3, 4], "why": "partial: steps [2, 3, 4] not snapshotted by the Match"}


@pytest.mark.parametrize("b", [1, 2])
def test_named_population_exclusions_without_form_b_do_not_pass_f_r16_02(b):
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_INTERIOR_ONLY}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: "), (b, ok, why)
    assert "140 VUs not evaluable by name" in why and "interior-composed-v2" in why and "MAN-06g" in why
    assert "form (B) absent" in why and "140 x fed by interior producer" in why
    assert "RoPE" not in why                                                     # the NOT_YET exclusion is not the gap named here
    # the mixed population (the original fixture: 140 interior-composed + 64 plane-fed) without form (B): the plane-fed FAIL leads
    # and names the interior-composed rest as not covered either
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_WITH_NAMED_EXCLUSIONS}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs"), (b, ok, why)
    assert "64 x fed by moe_block_stream" in why and "the other 140 VUs not evaluable by name (interior-composed-v2) are not covered either: form (B) absent" in why


@pytest.mark.parametrize("b", [1, 2])
def test_interior_composed_exclusions_with_complete_form_b_pass_carrying_the_gap_f_r16_02(b):
    """Interior-composed (Qwen2.5 BiasAdd over Gemm_v2): the module output IS an instance output form (B) compares -> a complete (B) covers the gap."""
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_INTERIOR_ONLY, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok and why.startswith("PASS (C2 of record by form (B) [oracle compare complete: 3 compared, 0 mismatch, instance outputs 2] over 140 VUs not evaluable by name"), (b, ok, why)
    assert "MAN-06g" in why and "interior-composed-v2" in why and "140" in why and "MAN-06f" not in why and "plane" not in why


@pytest.mark.parametrize("b", [1, 2])
def test_plane_fed_exclusions_fail_by_name_even_with_complete_form_b_f_r16_07(b):
    """Plane-fed: the VUs' committed members are the moe/ planes; form (B) has no oracle family for moe/
    (NO_ORACLE_FAMILY_PREFIXES) and form (A) has no row->plane mapping -> value-checked by NEITHER form -> FAIL named, whatever (B) says.
    The interior-composed part of the same population is reported as covered by the complete (B) (so the text says exactly what is missing)."""
    from verity_vllm.check.oracle_compare import NO_ORACLE_FAMILY_PREFIXES
    assert "moe/" in NO_ORACLE_FAMILY_PREFIXES                                   # the fact the rule rests on
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_WITH_NAMED_EXCLUSIONS, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs have committed moe/ plane members value-checked by NEITHER form"), (b, ok, why)
    assert "no oracle family for moe/ planes" in why and "64 x fed by moe_block_stream" in why
    assert "the other 140 VUs not evaluable by name (interior-composed-v2) are covered by the complete form (B)" in why
    assert "C2 needs form (A) over the planes (a row->plane mapping)" in why and "RoPE" not in why
    # plane-fed alone (OLMoE: every expert-act row), with and without form (B): the same FAIL by name -- (B) neither lifts nor is required here
    sr_p = {**SR_COMPLETE_WITH_NAMED_EXCLUSIONS, "population": {"vus": 2000, "not_evaluable": {K_06F: 64, K_ROPE: 32}}}
    for oc in (OC_COMPLETE, None, OC_PARTIAL):
        rec = {**BASE, "binding": _bnd(b), "sampled_replay": sr_p, **({"oracle_compare": oc} if oc is not None else {})}
        ok, why = commit_verdict(rec, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs") and "the other" not in why, (oc, why)
    # without form (B) the interior-composed rest is named as not covered either
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_WITH_NAMED_EXCLUSIONS}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs") and "are not covered either: form (B) absent" in why
    gap, fail = _complete_replay_population_gap(sr_p, OC_COMPLETE)
    assert gap is None and fail is not None and "plane-fed" in fail


@pytest.mark.parametrize("oc_key", ["top", "value_correspondence", "components"])
def test_form_b_is_read_from_every_place_the_record_carries_it(oc_key):
    """The oracle compare sits at the top level (commit_delta), under value_correspondence (composite) or under its components (compiled S3)."""
    rec = {**BASE, "binding": _bnd(2), "sampled_replay": SR_COMPLETE_INTERIOR_ONLY}
    if oc_key == "top":
        rec["oracle_compare"] = OC_COMPLETE
    elif oc_key == "value_correspondence":
        rec["value_correspondence"] = {**BASE["value_correspondence"], "oracle_compare": OC_COMPLETE}
    else:
        rec["value_correspondence"] = {**BASE["value_correspondence"], "components": {"oracle_compare": OC_COMPLETE}}
    ok, why = commit_verdict(rec, True, min_openings=64)
    assert ok and "form (B)" in why, (oc_key, ok, why)


def test_partial_or_mismatching_form_b_does_not_lift_the_gap():
    """A form (B) that is present but not COMPLETE over the values never stands in: partial (steps not snapshotted -> the MATCH_SNAP_STEPS=all
    remedy named), instance outputs not compared, attribution incomplete -- each FAILs by name; a form (B) MISMATCH fails earlier as the
    oracle mismatch it is."""
    rec = {**BASE, "binding": _bnd(2), "sampled_replay": SR_COMPLETE_INTERIOR_ONLY}
    ok, why = commit_verdict({**rec, "oracle_compare": OC_PARTIAL}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: ") and "form (B) result None (partial)" in why
    assert _form_b_complete_over_values(OC_PARTIAL).startswith("form (B) result None (partial)")
    assert "MATCH_SNAP_STEPS=all" in _form_b_complete_over_values({**OC_COMPLETE, "complete": False, "steps_not_snapshotted": [2]})
    no_io = {**OC_COMPLETE, "compared_by_family": {"input_token_ids": 3}, "not_compared_by_family": {"instance_outputs": 5}}
    ok, why = commit_verdict({**rec, "oracle_compare": no_io}, True, min_openings=64)
    assert not ok and "instance outputs compared 0, not compared 5" in why
    ok, why = commit_verdict({**rec, "oracle_compare": {**OC_COMPLETE, "attribution": {"complete": False, "pairs": 3, "unverified_n": 1}}}, True, min_openings=64)
    assert not ok and "request attribution not complete" in why
    ok, why = commit_verdict({**rec, "oracle_compare": {**OC_COMPLETE, "result": False, "mismatch_n": 1, "why": "C2 oracle mismatch: 1 identities differ"}}, True, min_openings=64)
    assert not ok and why.startswith("C2 oracle mismatch")


def test_not_yet_exclusions_alone_are_the_applicability_domain_not_a_gap():
    """A complete replay whose population excluded only NOT_YET classes (RoPE_v1 no evaluator, Attention_v3 state-linked) PASSes plain with
    or without form (B) -- the standard the 3 finalized workloads were graded by; a zero-count by-name class is no gap either."""
    for b in (1, 2):
        ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": SR_COMPLETE_NOT_YET_ONLY}, True, min_openings=64)
        # the applicability domain is no gap; at B=2 without form (B) the PASS names the attribution gap (and only that)
        assert ok and (why == "PASS" if b == 1 else why.startswith("PASS (B=2: per-(step,request) attribution not verified") and "RoPE" not in why), (b, ok, why)
    zero = {**SR_COMPLETE_NOT_YET_ONLY, "population": {"vus": 10, "not_evaluable": {K_06G: 0, K_ROPE: 4}}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(2), "sampled_replay": zero}, True, min_openings=64)
    assert ok and why.startswith("PASS (B=2: per-(step,request) attribution not verified") and "RoPE" not in why and "not_evaluable" not in why, why
    assert _complete_replay_population_gap(SR_COMPLETE_NOT_YET_ONLY, None) == (None, None)
    assert _complete_replay_population_gap({**SR_COMPLETE_NOT_YET_ONLY, "result": None, "grade": "partial"}, None) == (None, None)   # the partial rule's case, not this one


def test_a_replay_mismatch_still_fails_first_whatever_the_population_says():
    sr = {**SR_COMPLETE_WITH_NAMED_EXCLUSIONS, "result": False, "grade": "mismatch", "why": "VU 12 (RmsNorm_v1 model.layers.3.input_layernorm) replays 0x1234 != committed 0x1235"}
    ok, why = commit_verdict({**BASE, "binding": _bnd(2), "sampled_replay": sr, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert not ok and why.startswith("C2 sampled replay: VU 12")
