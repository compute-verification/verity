"""[R16 pin F-r16-20b / M-0431; the Hopper instance of rev's F-r16-20 class] On a cc 9.0 step `verity_vllm.program.frontend.targets.gemm_spec` /
`attention_spec` bind `Gemm_v2{K,N,DOT=HopperBF16WgmmaDot16_v1}` and `Attention_v2{...}` (only an Ampere-family step under FA2 binds the
accepted Gemm_v1 / Attention_v3).  `rows.EVALUATORS` has Gemm_v1 and not Gemm_v2, so `population.population()` excluded
EVERY Hopper GEMM output VU (every linear + lm_head) as `no registered evaluator for Gemm_v2` -- why-class 'no-evaluator', the applicability
domain's label, a text NOT in NOT_YET -- and the replay graded COMPLETE over what is left (RMSNorm, SiluMul, Embedding, TokenSelect).
Attention_v2 is the SAME state-linked limit as Attention_v3 under the Hopper name and enters NOT_YET BY NAME with the identical text
(rev M-0434 (2): the domain does not grow in kind); Gemm_v2 is a MISSING EVALUATOR (HopperBF16WgmmaDot16 k16 chain, R17), never the domain.  F-r16-02 gated the COMPOSED
Gemm_v2 gap (BiasAdd over Gemm_v2, interior-composed-v2) and not the DIRECT one: on a Hopper Qwen row the qkv VUs needed a complete form (B)
while o_proj / down_proj / lm_head needed nothing -- a Hopper Commit-only rerun without --match-dir PASSed with the bulk of the model's
values checked by NEITHER form.

Behaviour of record (this file): a non-selector family excluded by `no registered evaluator` outside NOT_YET joins the gaps form (B) must
cover over `instance_outputs` (`_form_b_complete_over_values`, the interior-composed test: the module output IS an instance output) ->
PASS carrying the gap, else FAIL by name.  Hopper full rows at 256/32 with a complete oracle PASS with the note; a Hopper Commit-only
without --match-dir FAILs by name; a 1024/128 row whose oracle is value-partial (M-0217 class) FAILs by name until the Match record
snapshots every step.  L40S / A100 rows (Gemm_v1, Attention_v3) are byte-identical."""
from __future__ import annotations

import pytest

from verity_vllm.check.c2_rules import _complete_replay_population_gap, _no_evaluator_gaps
from verity_vllm.check.verdict import commit_verdict
from verity_vllm.check.replay.replay_codes import NOT_YET, why_class
from verity_vllm.program.kernels.rows import EVALUATORS
from verity_vllm.program.frontend import targets as T


def _bnd(n_req):
    return {"context": {"ok": True}, "coverage": {"ok": True}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": n_req}}


BASE = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64},
        "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
        "boundary_linkage": {"result": True, "pairs": 6, "linked": 6, "unlinked_n": 0},
        "weights_pin": {"result": True, "compared": 163, "mismatched": [], "missing": []}}

K_ROPE = "no derived_rows evaluator for RoPE_v1 yet (v2)"
K_GEMM2 = "no registered evaluator for Gemm_v2"
# [R17 dense, ruling R17-11] the OLDER records' label for Attention_v2 / Attention_v3 (identical text, rev M-0434 (2)): both families left
# NOT_YET when the row evaluator landed (derived_rows.attention_rows); a record produced by this version has no such text.  The verdict rules
# below read the label as older records carry it (why_class 'state-linked': the replay version of record's stated domain).
K_ATT2 = K_ATT3 = "consumes the request's K/V history across steps (state-linked replay, v2)"
# the shape a Hopper llama32-1b B1 256/32 complete replay's population takes (counts illustrative: 16 layers x 7 linears + lm_head per step)
SR_HOPPER = {"result": True, "grade": "complete", "why": None,
             "sample": {"evaluated": 700, "equal": 700, "mismatch_n": 0, "not_evaluated_n": 0},
             "strata_unevaluated": [], "unevaluated_strata": [],
             "population": {"vus": 40000, "not_evaluable": {K_ROPE: 9184, K_GEMM2: 3616, K_ATT2: 4592},
                            "not_evaluable_by_family": {"RoPE_v1": 9184, "Gemm_v2": 3616, "Attention_v2": 4592}}}   # Attention_v2 counted under the domain text
OC_COMPLETE = {"result": True, "grade": "complete", "compared": 13199, "equal": 13199, "mismatch_n": 0, "complete": True,
               "compared_by_family": {"instance_outputs": 13065, "input_token_ids": 67, "sampled_token_ids": 67}, "not_compared_by_family": {"hidden_stream": 2544},
               "attribution": {"complete": True, "pairs": 67, "unverified_n": 0, "ambiguous_steps": []}}
# the M-0217 class at 1024/128: value-partial -- 32 committed identities with no snapshot (every 4th decode step's gate_up alloc)
OC_M0217 = {**OC_COMPLETE, "result": None, "grade": "attribution-complete", "complete": False, "steps_not_snapshotted": [],
            "not_compared_by_family": {"hidden_stream": 2544, "instance_outputs": 32}, "why": "32 committed identities have no snapshot in this attempt's Match record"}


def test_the_mechanism_is_in_the_registry_and_the_evaluator_table():
    """The finding's facts, asserted where they live: Hopper binds Gemm_v2 / Attention_v2.  [R17 dense, ruling R17-3] Gemm_v2 now HAS a
    registered evaluator (the wgmma k16 chain, dispatched on the row's DOT static) and is in neither NOT_YET nor the gap list of a NEW
    record; Attention_v2 stays a NOT_YET label (state-linked, an engineering gap by name).  The verdict rules below are kept as written:
    R16 records of Hopper rows still carry `no registered evaluator for Gemm_v2` BY NAME and must keep reading as they did."""
    class _P:                                                                    # the two facts targets.uses_accepted_kinds reads
        tensor_core_family = "hopper"
        capability_str = "9.0"

        def effective_flash_attn_version(self):
            return 3
    assert not T.uses_accepted_kinds(_P())                                      # a cc 9.0 step binds the statics-lifted kinds (gemm_spec -> b1.GemmV2)
    from verity_vllm.program.frontend.gemm_targets import REGISTERED
    assert REGISTERED["hopper"].kind.startswith("Gemm_v2{") and "HopperBF16WgmmaDot16" in REGISTERED["hopper"].kind
    assert "Gemm_v2" in EVALUATORS and "Gemm_v2" not in NOT_YET               # [R17 dense] the evaluator landed: HopperBF16WgmmaDot16 k16 chain
    assert "HopperBF16WgmmaDot16_v1" in EVALUATORS["Gemm_v2"] and "dot_bf16_for(DOT)" in EVALUATORS["Gemm_v2"]
    # [R17 dense, ruling R17-11] Attention_v2 / Attention_v3 have the row evaluator (the Q_module_v1 unit): neither is a NOT_YET label any more
    assert "Attention_v2" in EVALUATORS and "Attention_v3" in EVALUATORS and not ({"Attention_v2", "Attention_v3"} & set(NOT_YET))
    assert "Fa3InvSum_v1" in EVALUATORS["Attention_v2"] and "attention_rows" in EVALUATORS["Attention_v3"]
    assert why_class(K_GEMM2) == "no-evaluator" == why_class(K_ROPE)             # the same label as the domain -- the population text is the handle
    assert why_class(K_ATT2) == "state-linked"                                   # an OLDER record's attention label still reads as the domain
    sel, other = _no_evaluator_gaps(SR_HOPPER["population"]["not_evaluable"])
    assert sel == {} and other == {"Gemm_v2": 3616}                             # an R16 record's text: Attention_v2 is domain; Gemm_v2 was the gap
    # a record produced by THIS version names no Gemm_v2 gap: the population's not_evaluable text never carries K_GEMM2 (the family evaluates)
    from verity_vllm.program.kernels import rows as RK
    assert RK.gemm_dot_prim({"DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}) == "HopperBF16WgmmaDot16_v1"


def test_hopper_commit_only_without_form_b_fails_by_name():
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_HOPPER}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established (COORD M-0721 (1)): 3616 VUs not evaluable by name (no-evaluator)"), why
    assert "form (B) absent" in why and "3616 x no registered evaluator for Gemm_v2" in why
    assert "RoPE" not in why and "Attention" not in why                           # the domain is not the gap named


def test_hopper_full_row_with_a_complete_oracle_passes_carrying_the_gap():
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_HOPPER, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok, why
    assert why.startswith("PASS (C2 of record by form (B) [oracle compare complete: 13199 compared, 0 mismatch, instance outputs 13065] over 3616 VUs not evaluable by name (no-evaluator)"), why
    assert "3616 x no registered evaluator for Gemm_v2" in why and "Attention" not in why


def test_hopper_1024_128_with_the_m0217_value_partial_oracle_fails_by_name():
    """The module outputs at the not-snapshotted steps are checked by neither form: named, not a PASS."""
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": SR_HOPPER, "oracle_compare": OC_M0217}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established (COORD M-0721 (1)): 3616 VUs not evaluable by name (no-evaluator)"), why
    assert "form (B) result None" in why or "form (B) not complete" in why or "instance outputs compared" in why, why


@pytest.mark.parametrize("b", [1, 8])
def test_ampere_rows_are_byte_identical(b):
    """L40S / A100 under FA2: Gemm_v1 / Attention_v3 -- the population carries only the NOT_YET domain (phi3 B8 int-2 record shape)."""
    sr = {**SR_HOPPER, "population": {"vus": 242048, "not_evaluable": {K_ROPE: 68480, K_ATT3: 34240},
                                      "not_evaluable_by_family": {"RoPE_v1": 68480, "Attention_v3": 34240}}}
    assert _complete_replay_population_gap(sr, None) == (None, None)
    ok, why = commit_verdict({**BASE, "binding": _bnd(b), "sampled_replay": sr}, True, min_openings=64)
    assert ok and why.startswith("PASS"), why
    assert "no-evaluator" not in why and "Gemm" not in why


def test_composed_and_direct_gemm_v2_gaps_get_the_same_rule():
    """rev's K_06G fixture (BiasAdd over Gemm_v2, Hopper Qwen) beside the direct Gemm_v2 outputs: one FAIL naming both, one PASS covering both."""
    k_06g = "fed by interior producer(s) without an evaluator (Gemm_v2 (model.layers.*.self_attn.qkv_proj)): composition not possible in this replay (MAN-06g, v2)"
    sr = {**SR_HOPPER, "population": {**SR_HOPPER["population"], "not_evaluable": {K_ROPE: 9184, k_06g: 140, K_GEMM2: 3476}}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr}, True, min_openings=64)
    assert not ok and "3616 VUs not evaluable by name (interior-composed-v2, no-evaluator; MAN-06g)" in why, why
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok and "over 3616 VUs not evaluable by name (interior-composed-v2, no-evaluator; MAN-06g)" in why, why


def test_the_plane_fed_fail_still_leads():
    k_06f = "fed by moe_block_stream planes (SiluMul_v1@experts): row -> plane mapping not in this replay (MAN-06f, v2)"
    sr = {**SR_HOPPER, "population": {**SR_HOPPER["population"], "not_evaluable": {K_ROPE: 9184, k_06f: 64, K_GEMM2: 3616}}}
    ok, why = commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: 64 plane-fed VUs (MAN-06f)"), why
    assert "the other 3616 VUs not evaluable by name (no-evaluator) are covered by the complete form (B)" in why, why
