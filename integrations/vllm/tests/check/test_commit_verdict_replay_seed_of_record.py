"""[R16 rev F-r16-23 / M-0579; VLLM-17] The sampled replay's seed is post-commitment as a FACT, never by label: only the service
challenger's live replay coin, opened after the root of record against a commitment the verifier recorded (`challenge.attested_reason`),
makes the replay a C2 of record.  A root-derived seed is grindable (the run root takes a free prover string: the red team hid a 10% cheat
from 64 root-derived openings in ~1,000 tries), a 'verifier challenge' is a string from the prover's own command line with no provenance,
and `--replay-seed` is prover-chosen: each makes the replay a MECHANISM run, C2 of record then rests on form (B) alone (M-0721 (1)) and the
PASS names it, and without a complete form (B) the Commit FAILS by name.  Plane-fed gaps (F-r16-07) are still named whatever the seed."""
from __future__ import annotations


from verity_vllm.check.c2_rules import _replay_seed_of_record
from verity_vllm.check.verdict import commit_verdict
from verity_vllm.check.replay.sample import challenge_seed

# the shape of rev M-0579's recomputed fact (qwen A100 int-3 pair 0: run_root 71f69377... -> sample.seed 8211913111514837969)
RUN_ROOT = "71f69377" + "ab" * 28
SEED_ROOT = int.from_bytes(bytes.fromhex(RUN_ROOT)[:8], "big")
ROOT_FORM = "run root (first 8 bytes, after commitment)"

BASE = {"extra": {"expected_tensors": 3, "committed_tensors": 3, "errors_n": 0}, "openings": {"n": 64, "verified": 64}, "run_root": RUN_ROOT,
        "binding": {"context": {"ok": True}, "coverage": {"ok": True}, "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
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


ROOT_OF_RECORD, LIVE_KEY = "5a" * 64, "c3" * 64
CHALLENGE = {"keys": {"opening-positions": "a1" * 64, "opening-identities": "b2" * 64, "replay": LIVE_KEY, "kernel-check": "d4" * 64},
             "origin": "challenger", "after_run_root": ROOT_OF_RECORD, "provenance": {"stand-in": True}}


def _sr_chal(seed: int, src: str, chal: str | None) -> dict:
    d = _sr(seed, src)
    if chal is not None:
        d["sample"]["challenge"] = chal
    return d


def _of_record(monkeypatch, sr: dict | None = None) -> dict:
    """A Commit of record whose replay drew from the stand-in challenge, which the verifier accepts as its challenger's (the check
    itself: tests/commit/test_challenge.py)."""
    from verity_vllm.check.replay import replay_codes as RC
    from verity_vllm.commit import challenge as CH

    monkeypatch.setattr(CH, "attested_reason", lambda ch: None if (ch or {}).get("provenance") == {"stand-in": True} else "not the stand-in")
    return {**BASE, "run_root": ROOT_OF_RECORD, "extra": {**BASE["extra"], "challenge": CHALLENGE},
            "sampled_replay": sr if sr is not None else _sr_chal(0, RC.LIVE_KEY_SOURCE, LIVE_KEY)}


def test_a_root_derived_seed_is_grindable_and_never_of_record():
    for sr in (_sr(SEED_ROOT, ROOT_FORM), _sr(12345, ROOT_FORM), {"result": True, "seed": SEED_ROOT, "seed_source": ROOT_FORM}):
        for root in (RUN_ROOT, None, ROOT_OF_RECORD):
            ok, why = _replay_seed_of_record(sr, root)
            assert ok is False and "grindable" in why and "not a C2 of record" in why


def test_a_verifier_challenge_from_the_provers_command_line_is_never_of_record():
    seed, src = challenge_seed("audit-2026-09-20", bytes.fromhex(RUN_ROOT))
    for chal in ("audit-2026-09-20", None, "another-challenge"):
        ok, why = _replay_seed_of_record(_sr_chal(seed, src, chal), RUN_ROOT)
        assert ok is False and "prover's own command line" in why and "not a C2 of record" in why
    ok, why = _replay_seed_of_record(_sr(7, "--replay-seed (targeted)"), RUN_ROOT)
    assert ok is False and "prover-chosen" in why and "not a C2 of record" in why
    ok, why = _replay_seed_of_record(_sr(7, "caller"), RUN_ROOT)
    assert ok is False and "not a post-commitment form" in why
    # a legacy record without the field is not graded here (the finalizer's ◐ by name); a run of record without one is refused
    assert _replay_seed_of_record({"result": True, "sample": {"seed": 7}}, RUN_ROOT)[0] is None
    ok, why = _replay_seed_of_record({"result": True, "sample": {"seed": 7}}, ROOT_OF_RECORD, CHALLENGE)
    assert ok is False and "no seed source" in why


def test_a_prover_chosen_seed_fails_the_commit_by_name_unless_form_b_is_complete_which_the_pass_then_names(monkeypatch):
    # the honest record of record: the service challenger's replay coin after the root -> plain PASS
    ok, why = commit_verdict(_of_record(monkeypatch), True, min_openings=64)
    assert (ok, why) == (True, "PASS")
    # --replay-seed on a Commit-only rerun (no --match-dir): the replay is a mechanism run, nothing else value-checks -> FAIL by name
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)")}, True, min_openings=64)
    assert ok is False and why.startswith("C2 of record not established (F-r16-23): sampled replay seed prover-chosen (--replay-seed)")
    assert "form (B) absent" in why
    # with a complete form (B) (the F-r16-02 form) C2 of record holds through (B) alone and the PASS says so
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)"), "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is True and why.startswith("PASS (") and "prover-chosen" in why and "C2 of record by the complete form (B) alone" in why
    # a value-partial form (B) does not stand in
    oc_partial = {**OC_COMPLETE, "result": None, "grade": "attribution-complete", "complete": False, "steps_not_snapshotted": [3]}
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(7, "--replay-seed (targeted)"), "oracle_compare": oc_partial}, True, min_openings=64)
    assert ok is False and "form (B) does not stand in: form (B) result None" in why
    # a root-derived seed is the same class
    ok, why = commit_verdict({**BASE, "sampled_replay": _sr(SEED_ROOT, ROOT_FORM)}, True, min_openings=64)
    assert ok is False and why.startswith("C2 of record not established (F-r16-23)") and "grindable" in why and "form (B) absent" in why


def test_plane_fed_gaps_are_named_whatever_the_seed_and_a_replay_mismatch_still_leads():
    sr = _sr(7, "--replay-seed (targeted)", population={"vus": 1000, "not_evaluable": {K_PLANE: 40}, "not_evaluable_by_family": {"MoeExpertGemm": 40}})
    ok, why = commit_verdict({**BASE, "sampled_replay": sr, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is False and "plane-fed" in why                                       # F-r16-07 leads even with form (B) complete
    bad = {**_sr(7, "--replay-seed (targeted)"), "result": False, "why": "r0 step 3 layers.0.mlp: committed output != replayed"}
    ok, why = commit_verdict({**BASE, "sampled_replay": bad, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is False and why.startswith("C2 sampled replay:")                      # a mismatch is a mismatch, seed or not


def test_f_r16_26_the_f_r16_13_fault_rule_runs_before_the_mechanism_lift_whatever_seeded_the_replay(monkeypatch):
    """[rev F-r16-26 / rel M-0686] a PARTIAL replay from a run-time FAULT (an unevaluated pick inside an evaluated stratum; no fully
    unevaluated stratum -> _partial_replay_named_gap names no gap) fails closed under the root form (F-r16-13) -- and must fail the SAME way
    under a prover-chosen / non-post-commitment seed: form (B) stands in for a NAMED gap, never for a fault.  (The int-5 pod gate caught the
    lift passing it: test_sampled_replay line 1085 with seed_source 'caller'.)"""
    fault = {"result": None, "grade": "partial", "why": "1 pick(s) not evaluated inside evaluated strata (missing-input x 1) -- run-time fault, not a gap on the record: "
                                                        "committed bytes not retained for (r0, step 0, x/out) [0, 16) (F-r16-13)",
             "strata_unevaluated": [], "sample": {"seed": 7, "seed_source": "caller", "evaluated": 9, "equal": 9, "mismatch_n": 0, "not_evaluated_n": 1, "picked": 10}}
    for src in ("caller", "--replay-seed (targeted)"):
        ok, why = commit_verdict({**BASE, "sampled_replay": {**fault, "sample": {**fault["sample"], "seed_source": src}}, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
        assert not ok and why.startswith("C2 of record not established (DECISIONS-2 §4): sampled replay partial") and "run-time fault" in why, (src, why)
        assert "F-r16-13" in why
    # a seed of record fails it by the non-mechanism branch, byte-for-byte the F-r16-13 text (unchanged)
    from verity_vllm.check.replay import replay_codes as RC
    live = {**fault, "sample": {**fault["sample"], "seed": 0, "seed_source": RC.LIVE_KEY_SOURCE, "challenge": LIVE_KEY}}
    ok, why = commit_verdict({**_of_record(monkeypatch, live), "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert not ok and why.startswith("C2 of record not established (DECISIONS-2 §4): sampled replay partial") and "run-time fault" in why
    # a partial replay with a NAMED gap (a fully unevaluated stratum of a class form (B) covers) is the lift's case: the prover seed PASSES on (B) alone, named
    named = {"result": None, "grade": "partial", "why": "1 stratum unevaluated by name", "strata_unevaluated": ["s1"],
             "unevaluated_strata": [{"stratum": "s1", "spec": "Gemm_v2{...}", "op_path": "model.layers.0.mlp.gate_up_proj", "n": 4,
                                     "why": "no registered evaluator for Gemm_v2 (Hopper wgmma) [x]", "classes": ["no-evaluator"]}],
             "sample": {"seed": 7, "seed_source": "--replay-seed (targeted)", "evaluated": 9, "equal": 9, "mismatch_n": 0, "not_evaluated_n": 4, "picked": 13}}
    ok, why = commit_verdict({**BASE, "sampled_replay": named, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
    assert ok is True and "C2 of record by the complete form (B) alone" in why, why


def test_the_live_key_form_of_a_run_of_record_is_the_service_challengers_replay_coin_drawn_after_its_root(monkeypatch):
    """A run of record (`commit.record`, 64-byte root) draws its replay sample from the replay key `challenge.draw_live` recorded on
    ``extra.challenge`` after the root (`c2_replay.seed_of`: form ``live-key``, the key as ``sample.challenge``).  The verdict holds the
    sample's key to that one: another key, a key drawn after another root, a 32-byte root, or no key on the commit is refused by name.
    It is of record only as the service challenger's (`challenge.attested_reason`): anything else draws a measurement sample, which
    passes on a complete form (B) alone and fails without one."""
    from verity_vllm.check.replay import replay_codes as RC

    rec = _of_record(monkeypatch)
    sr, root, ch = rec["sampled_replay"], ROOT_OF_RECORD, CHALLENGE
    assert RC.seed_form(sr) == "live-key"
    assert "not the live replay key" in _replay_seed_of_record(_sr_chal(0, RC.LIVE_KEY_SOURCE, ch["keys"]["kernel-check"]), root, ch)[1]
    assert "no live replay key drawn after its root" in _replay_seed_of_record(sr, root, {**ch, "after_run_root": "6b" * 64})[1]
    assert "no live replay key drawn after its root" in _replay_seed_of_record(sr, root, None)[1]
    assert "not a 64-byte root of record" in _replay_seed_of_record(sr, RUN_ROOT, {**ch, "after_run_root": RUN_ROOT})[1]
    ok, how = _replay_seed_of_record(sr, root, ch)
    assert ok is True and "service challenger's live replay coin" in how
    for origin in ("unattested", "own"):
        other = {**ch, "origin": origin, "provenance": None}
        ok, why = _replay_seed_of_record(sr, root, other)
        assert ok is False and "measurement sample" in why
        bad = {**rec, "extra": {**rec["extra"], "challenge": other}}
        ok, why = commit_verdict(bad, True, min_openings=64)
        assert ok is False and why.startswith("C2 of record not established (F-r16-23)") and "form (B) absent" in why
        ok, why = commit_verdict({**bad, "oracle_compare": OC_COMPLETE}, True, min_openings=64)
        assert ok is True and "C2 of record by the complete form (B) alone" in why


def test_a_commit_of_record_passes_on_the_service_challengers_coins_only_once_the_verifier_trusts_their_commitment(monkeypatch):
    """End to end through the challenger: a `LiveChallenger` under the Commit's scheme answers the root, the replay draws from its replay
    coin, and the verdict PASSes only once its commitment is one the verifier recorded (`challenge.trust`)."""
    from types import SimpleNamespace

    import pytest

    from verity_vllm.check.replay import replay_codes as RC
    from verity_vllm.commit import challenge as CH

    LC = pytest.importorskip("verity.core.protocols.sampled_proofs.challenger",
                             reason="the service's challenger (compute accounting's #2573 / #2584) is not on this tree")
    monkeypatch.setattr(CH, "TRUSTED", {})
    chal = LC.LiveChallenger(CH.SCHEME, 0, CH.layout(1))
    run = SimpleNamespace(run_root=bytes.fromhex(ROOT_OF_RECORD), extra={})
    key = CH.draw_live(run, CH.service_challenger(chal)).keys[CH.REPLAY].hex()
    rec = {**BASE, "run_root": ROOT_OF_RECORD, "extra": {**BASE["extra"], **run.extra}, "sampled_replay": _sr_chal(0, RC.LIVE_KEY_SOURCE, key)}
    ok, why = commit_verdict(rec, True, min_openings=64)
    assert ok is False and "not one the verifier recorded from its challenger" in why and "form (B) absent" in why
    CH.trust(chal.commitment.to_json())
    assert commit_verdict(rec, True, min_openings=64) == (True, "PASS")


def test_the_commit_draws_a_run_of_records_replay_from_its_live_key():
    from types import SimpleNamespace

    from verity_vllm.commit import challenge as CH
    from verity_vllm.pipeline import c2_replay as C2

    root = bytes(range(64))
    run = SimpleNamespace(run_root=root, extra={})
    key = CH.draw_live(run, "ab" * 16).keys[CH.REPLAY]
    args = SimpleNamespace(replay_seed=None, replay_challenge=None)
    seed, src, form, chal = C2.seed_of(args, run)
    assert (form, chal, seed) == ("live-key", key.hex(), 0) and src.startswith("live key")
    assert C2.live_key_of(args, run) == CH.replay_key(key, root)
    assert C2.live_key_of(args, SimpleNamespace(run_root=bytes(32), extra={})) is None
    import pytest

    with pytest.raises(ValueError, match="--challenge-key"):
        C2.seed_of(SimpleNamespace(replay_seed=None, replay_challenge="x"), run)
