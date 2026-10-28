"""The Commit verdict reads the codes the sampled replay stamps (`population.not_evaluable_codes`, `sample.seed_form`;
`replay_codes`), never the message texts: a reworded reason or seed source grades as before, a code (not a text) moves the verdict, and
a record written before the codes grades from its texts' decoding as it always did."""
from __future__ import annotations

from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.check.c2_rules import _replay_seed_of_record
from verity_vllm.check.commit_pass import commit_verdict

from .test_commit_verdict_hopper_no_evaluator_gap import BASE, K_GEMM2, OC_COMPLETE, SR_HOPPER, _bnd


def _coded(not_ev: dict, codes: dict) -> dict:
    return {**SR_HOPPER, "population": {**SR_HOPPER["population"], "not_evaluable": not_ev, "not_evaluable_codes": codes}}


def _verdict(sr: dict, oc: dict | None = None):
    return commit_verdict({**BASE, "binding": _bnd(1), "sampled_replay": sr, **({"oracle_compare": oc} if oc else {})}, True, min_openings=64)


def test_a_record_with_codes_grades_as_its_texts_decode():
    sr = _coded(SR_HOPPER["population"]["not_evaluable"], RC.not_evaluable_codes(SR_HOPPER["population"]))
    for oc in (None, OC_COMPLETE):
        assert _verdict(sr, oc) == _verdict(SR_HOPPER, oc)


def test_a_reworded_reason_grades_by_its_code():
    gap, dom = "Gemm_v2 rows: no evaluator", "attention history (the version's domain)"
    sr = _coded({gap: 3616, dom: 4592}, {gap: RC.reason_code("no-evaluator", "Gemm_v2"),
                                         dom: RC.reason_code("state-linked", "Attention_v2", domain=True)})
    ok, why = _verdict(sr)
    assert not ok and why.startswith("C2 of record not established: 3616 VUs not evaluable by name (no-evaluator)"), why
    assert "form (B) absent" in why and f"3616 x {gap}" in why and dom not in why
    ok, why = _verdict(sr, OC_COMPLETE)
    assert ok and "over 3616 VUs not evaluable by name (no-evaluator)" in why, why


def test_the_code_not_the_text_moves_the_verdict():
    """The Gemm_v2 text of record decides nothing by itself: coded as the stated domain it is no gap; coded as a composed-evaluator gap it
    is named under that class."""
    ok, why = _verdict(_coded({K_GEMM2: 3616}, {K_GEMM2: RC.reason_code("no-evaluator", "Gemm_v2", domain=True)}))
    assert ok and why.startswith("PASS") and "Gemm" not in why, why
    ok, why = _verdict(_coded({K_GEMM2: 3616}, {K_GEMM2: RC.reason_code("interior-composed-v2", "BiasAdd_v1", tags=("MAN-06g",))}))
    assert not ok and "3616 VUs not evaluable by name (interior-composed-v2; MAN-06g)" in why, why


ROOT = bytes(range(32)).hex()
ROOT_SEED = int.from_bytes(bytes(range(8)), "big")


def test_the_seed_form_on_the_record_decides_not_its_source_text():
    ok, how = _replay_seed_of_record({"sample": {"seed": ROOT_SEED, "seed_source": "the run root's first bytes", "seed_form": "run-root"}}, ROOT)
    assert ok is True and "recomputed" in how
    ok, why = _replay_seed_of_record({"sample": {"seed": ROOT_SEED, "seed_source": RC.ROOT_SEED_SOURCE, "seed_form": "prover-chosen"}}, ROOT)
    assert ok is False and "prover-chosen" in why
    # a record written before `seed_form`: its source text decodes to the form
    assert RC.seed_form({"sample": {"seed_source": RC.ROOT_SEED_SOURCE}}) == "run-root"
    assert RC.seed_form({"sample": {"seed_source": RC.PROVER_SEED_SOURCE}}) == "prover-chosen"
    assert RC.seed_form({"sample": {"seed_source": RC.challenge_seed_source("abc")}}) == "challenge"
    assert RC.seed_form({"seed_source": "chosen somehow"}) == "unknown" and RC.seed_form({"sample": {}}) is None
    assert _replay_seed_of_record({"sample": {"seed": ROOT_SEED, "seed_source": RC.ROOT_SEED_SOURCE}}, ROOT)[0] is True
    assert _replay_seed_of_record({"sample": {"seed": ROOT_SEED, "seed_source": RC.PROVER_SEED_SOURCE}}, ROOT)[0] is False
