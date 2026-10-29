"""The sampled replay's seed is post-commitment by LABEL only: commit_delta [O6] derives it from the
run root after commitment (int.from_bytes(run_root[:8])) or from a verifier challenge (sha256(challenge|root)), but `--replay-seed <int>`
sets a PROVER-CHOSEN seed known before the run (seed_source '--replay-seed (targeted)'), and a prover who knows the seed knows the picks
before committing (the population is fixed by the manifest) -- every unpicked VU is free, C3's prechallenge binding is defeated by
construction.  Neither commit_verdict nor the finalizer read `sample.seed_source`; a seed only had to EXIST.

Behaviour of record (this file): commit_verdict reads the seed source as a FACT.  The root form is RECOMPUTED from the record's run root;
the challenge form is post-commitment by construction; a prover-chosen seed (or a root-form label whose seed is not the root's, or an
unknown source) makes the replay a MECHANISM run: C2 of record then rests on form (B) alone and the PASS names it, and without
a complete form (B) the Commit FAILS by name.  Plane-fed gaps are still named whatever the seed.  Every older record of record
(150/150 say the root form) recomputes equal and is byte-identical in verdict."""
from __future__ import annotations

import hashlib

from verity_vllm.check.c2_rules import _replay_seed_of_record
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.replay.sample import challenge_seed

# the shape of a recomputed fact (qwen A100 pair 0: run_root 71f69377... -> sample.seed 8211913111514837969)
RUN_ROOT = "71f69377" + "ab" * 28
SEED_ROOT = int.from_bytes(bytes.fromhex(RUN_ROOT)[:8], "big")
ROOT_FORM = "run root (first 8 bytes, after commitment)"

BASE = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "run_root": RUN_ROOT,
        "binding": {"context": {"ok": True}, "coverage": {"ok": True, "expected_n": 1, "bound_n": 1, "missing_n": 0}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
        "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
        "boundary_linkage": {"result": True, "pairs": 32, "linked": 32, "unlinked_n": 0},
        "weights_pin": {"result": True, "compared": 163, "mismatched": [], "missing": []}}
K_ROPE = "no derived_rows evaluator for RoPE_v1 yet (v2)"
K_PLANE = "fed by moe_block_stream planes (SiluMul_v1@experts): row -> plane mapping not in this replay (MAN-06f, v2)"


def _sr(seed, src, **kw):
    return {"result": True, "grade": "complete", "why": None, "strata_unevaluated": [],
            "sample": {"per_stratum": 2, "seed": seed, "seed_source": src, "picked": 407, "evaluated": 407, "equal": 407, "mismatch_n": 0, "not_evaluated_n": 0},
            "population": {"vus": 57678, "not_evaluable": {K_ROPE: 36832}, "not_evaluable_by_family": {"RoPE_v1": 36832}}, **kw}


OC_COMPLETE = {"result": True, "grade": "complete", "compared": 13199, "equal": 13199, "mismatch_n": 0, "complete": True,
               "compared_by_family": {"instance_outputs": 13065, "input_token_ids": 67, "sampled_token_ids": 67}, "not_compared_by_family": {"hidden_stream": 2544},
               "attribution": {"complete": True, "pairs": 67, "unverified_n": 0, "ambiguous_steps": []}}


def test_the_root_form_is_recomputed_from_the_records_run_root_not_believed():
    ok, how = _replay_seed_of_record(_sr(SEED_ROOT, ROOT_FORM), RUN_ROOT)
    assert ok is True and "recomputed" in how
    # the LABEL says root form but the seed is another integer: a forged label, refused
    ok, why = _replay_seed_of_record(_sr(12345, ROOT_FORM), RUN_ROOT)
    assert ok is False and "!= the run root's first 8 bytes" in why
    # no run root on the record: cannot recompute -> refused (never believed)
    ok, why = _replay_seed_of_record(_sr(SEED_ROOT, ROOT_FORM), None)
    assert ok is False and "no run root" in why
    # the summary form (components.sampled_replay: seed / seed_source at the top level) reads the same
    ok, _ = _replay_seed_of_record({"result": True, "seed": SEED_ROOT, "seed_source": ROOT_FORM}, RUN_ROOT)
    assert ok is True


def _sr_chal(seed: int, src: str, chal: str | None) -> dict:
    d = _sr(seed, src)
    if chal is not None:
        d["sample"]["challenge"] = chal
    return d


def test_the_verifier_challenge_form_is_recomputed_from_the_recorded_challenge_and_a_prover_seed_is_not():
    """The challenge form is a fact only with the FULL challenge on the record (sample.challenge, written by sampled_replay when
    commit_delta runs --replay-challenge): seed == sha256(challenge|run_root)[:8] recomputed; a label alone (the older shape, the source
    text carries challenge[:16] only), another seed under the label, or a source text naming another challenge is refused by name."""
    seed, src = challenge_seed("audit-2026-09-20", bytes.fromhex(RUN_ROOT))
    assert seed == int.from_bytes(hashlib.sha256(b"audit-2026-09-20|" + bytes.fromhex(RUN_ROOT)).digest()[:8], "big")
    ok, how = _replay_seed_of_record(_sr_chal(seed, src, "audit-2026-09-20"), RUN_ROOT)
    assert ok is True and "recomputed from the recorded challenge" in how
    ok, why = _replay_seed_of_record(_sr_chal(seed, src, None), RUN_ROOT)                      # the label alone: unverifiable
    assert ok is False and "challenge is not on the record" in why
    ok, why = _replay_seed_of_record(_sr_chal(seed + 1, src, "audit-2026-09-20"), RUN_ROOT)     # a forged seed under the true challenge
    assert ok is False and "!= sha256(challenge|run_root)[:8]" in why
    ok, why = _replay_seed_of_record(_sr_chal(seed, src, "another-challenge"), RUN_ROOT)        # another challenge: its seed differs
    assert ok is False and "!= sha256(challenge|run_root)[:8]" in why
    seed2, _src2 = challenge_seed("another-challenge", bytes.fromhex(RUN_ROOT))
    ok, why = _replay_seed_of_record(_sr_chal(seed2, src, "another-challenge"), RUN_ROOT)       # right seed, source text names another
    assert ok is False and "does not name the recorded challenge" in why
    assert _replay_seed_of_record(_sr_chal(seed, src, "audit-2026-09-20"), None)[0] is False   # no run root to recompute from
    # the record shape: sampled_replay writes sample.challenge ONLY under the challenge form (root-form records are byte-identical)
    import inspect
    from verity_vllm.check.replay import driver as SR
    assert "seed_challenge" in inspect.signature(SR.sampled_replay).parameters
    src_txt = inspect.getsource(SR.sampled_replay)
    assert '**({"challenge": str(seed_challenge)} if seed_challenge is not None else {})' in src_txt
    cd = "".join(open(__file__.rsplit("/tests/", 1)[0] + f"/verity_vllm/pipeline/{m}.py").read() for m in ("commit", "c2_replay"))
    assert "_seed_chal = str(args.replay_challenge)" in cd and "seed_challenge=_seed_chal" in cd
    ok, why = _replay_seed_of_record(_sr(7, "--replay-seed (targeted)"), RUN_ROOT)
    assert ok is False and "prover-chosen" in why and "not a C2 of record" in why
    ok, why = _replay_seed_of_record(_sr(7, "caller"), RUN_ROOT)
    assert ok is False and "not a post-commitment form" in why
    # a legacy record without the field is not graded here (the finalizer's ◐ by name), never mistaken for either
    assert _replay_seed_of_record({"result": True, "sample": {"seed": 7}}, RUN_ROOT)[0] is None


def test_a_prover_chosen_seed_fails_the_commit_by_name_unless_form_b_is_complete_which_the_pass_then_names():
    # the honest record of record: root-form seed recomputed -> plain PASS, byte-identical to before the seed rule
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(SEED_ROOT, ROOT_FORM)}, True, min_openings=64)
    assert (ok, why) == (True, "PASS")
    # --replay-seed on a Commit-only rerun (no --match-dir): the replay is a mechanism run, nothing else value-checks -> FAIL by name
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)")}, True, min_openings=64)
    assert ok is False and why.startswith("C2 of record not established: sampled replay seed prover-chosen (--replay-seed)")
    assert "form (B) absent" in why
    # with a complete form (B) C2 of record holds through (B) alone and the PASS says so
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)"), "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is True and why.startswith("PASS (") and "prover-chosen" in why and "C2 of record by the complete form (B) alone" in why
    # a value-partial form (B) does not stand in
    oc_partial = {**OC_COMPLETE, "result": None, "grade": "attribution-complete", "complete": False, "steps_not_snapshotted": [3]}
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)"), "oracle_compare": oc_partial}, True, min_openings=64)
    assert ok is False and "form (B) does not stand in: form (B) result None" in why
    # a forged root-form label is the same class
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(12345, ROOT_FORM)}, True, min_openings=64)
    assert ok is False and "!= the run root's first 8 bytes" in why


def test_plane_fed_gaps_are_named_whatever_the_seed_and_a_replay_mismatch_still_leads():
    sr = _sr(7, "--replay-seed (targeted)", population={"vus": 1000, "not_evaluable": {K_PLANE: 40}, "not_evaluable_by_family": {"MoeExpertGemm": 40}})
    ok, why = commit_verdict({**BASE, "sampled_replay": sr, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is False and "plane-fed" in why                                       # the plane-fed FAIL leads even with form (B) complete
    bad = {**_sr(7, "--replay-seed (targeted)"), "result": False, "why": "r0 step 3 layers.0.mlp: committed output != replayed"}
    ok, why = commit_verdict({**BASE, "sampled_replay": bad, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is False and why.startswith("C2 sampled replay:")                      # a mismatch is a mismatch, seed or not


def test_f_r16_26_the_f_r16_13_fault_rule_runs_before_the_mechanism_lift_whatever_seeded_the_replay():
    """A PARTIAL replay from a run-time FAULT (an unevaluated pick inside an evaluated stratum; no fully
    unevaluated stratum -> _partial_replay_named_gap names no gap) fails closed under the root form -- and must fail the SAME way
    under a prover-chosen / non-post-commitment seed: form (B) stands in for a NAMED gap, never for a fault.  (A pod gate caught the
    lift passing it: test_sampled_replay line 1085 with seed_source 'caller'.)"""
    fault = {"result": None, "grade": "partial", "why": "1 pick(s) not evaluated inside evaluated strata (missing-input x 1) -- run-time fault, not a gap on the record: "
                                                        "committed bytes not retained for (r0, step 0, x/out) [0, 16)",
             "strata_unevaluated": [], "sample": {"seed": 7, "seed_source": "caller", "evaluated": 9, "equal": 9, "mismatch_n": 0, "not_evaluated_n": 1, "picked": 10}}
    for src in ("caller", "--replay-seed (targeted)"):
        ok, why = commit_verdict({**BASE, "sampled_replay": {**fault, "sample": {**fault["sample"], "seed_source": src}}, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established: sampled replay partial") and "run-time fault" in why, (src, why)
        assert "committed bytes not retained for (r0, step 0, x/out)" in why
    # the root form fails it by the non-mechanism branch, byte-for-byte the named-gap rule's text (unchanged)
    ok, why = commit_verdict({**BASE, "sampled_replay": {**fault, "sample": {**fault["sample"], "seed": SEED_ROOT, "seed_source": ROOT_FORM}}, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established: sampled replay partial") and "run-time fault" in why
    # a partial replay with a NAMED gap (a fully unevaluated stratum of a class form (B) covers) is the lift's case: the prover seed PASSES on (B) alone, named
    named = {"result": None, "grade": "partial", "why": "1 stratum unevaluated by name", "strata_unevaluated": ["s1"],
             "unevaluated_strata": [{"stratum": "s1", "spec": "Gemm_v2{...}", "op_path": "model.layers.0.mlp.gate_up_proj", "n": 4,
                                     "why": "no registered evaluator for Gemm_v2 (Hopper wgmma) [x]", "classes": ["no-evaluator"]}],
             "sample": {"seed": 7, "seed_source": "--replay-seed (targeted)", "evaluated": 9, "equal": 9, "mismatch_n": 0, "not_evaluated_n": 4, "picked": 13}}
    ok, why = commit_verdict({**BASE, "sampled_replay": named, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is True and "C2 of record by the complete form (B) alone" in why, why
