"""padrev — adversarial checks of `lifted_record.py`'s CAPTURE-side rules (pad 98f6bc67e / 850393f65, M-0311) on exec's REAL earlystop
fixture (M-0140; the read-only copy pad keeps under out/gen/sweep/evidence/pad/...; SmolLM2-135M LP=32 cap 32, EOS 0 at event 10, async
placeholder step 11 fed the EOS and sampled 504).  Every test mutates a COPY of the capture in tmp_path — the fixture itself is never written.

Skipped by name when the fixture is not on the tree (it lives on lane/r16-pad; padrev's branch carries tests only).

What is attacked: the attestation (the header field is the ONLY lag source: injected True -> lag 1 and R0 PASS, yet `ok` must stay None
without a Commit; injected False -> lag 0 -> the placeholder row is execution past EOS); a deleted placeholder step (R3c/R3d/R2b, each by
name); a placeholder step fed something other than the committed EOS (R3d); the harness's measured lag as a would-be source (R0b is a
cross-check: it never rescues an unattested header); and F-pad-22 — a binding map that agrees with the declared lag while the engine's own
step log does not: a FAILED capture check must veto `ok`."""
from __future__ import annotations

import gzip
import json
import os
import shutil
from pathlib import Path

import pytest

from verity_vllm.lifting import record as R
from verity_vllm.registry import lifted as L

ROW = "smollm2-135m__bf16__l40s__tp1__b1__i32__o32__earlystop__greedy__bi-eager"
FIX = Path(__file__).resolve().parents[2] / "out" / "gen" / "sweep" / "evidence" / "pad" / ROW / "r16_0514_pad_fixture"
RID = "de2d6bfc-096d-5c10-beb0-293ff914262d"
EOS, CAP, A_R = (0,), 32, 11
PLACEHOLDER_STEP, PLACEHOLDER_SAMPLE = 11, 504

pytestmark = pytest.mark.skipif(not (FIX / "build_request" / "descriptor.json.gz").exists() or not (FIX / "match" / "capture" / "log.jsonl.gz").exists(),
                                reason=f"exec's earlystop fixture (pad's read-only copy) not on this tree: {FIX}")


@pytest.fixture(scope="module")
def desc():
    return R.load_descriptor(str(FIX / "build_request"))


def _copy_capture(tmp_path: Path, edit_events=None, edit_tokens=None) -> Path:
    """A private copy of match/capture with the given edits applied: `edit_events(list[dict]) -> list[dict]` on the decoded log events,
    `edit_tokens(dict) -> dict` on tokens.json."""
    dst = tmp_path / "capture"
    shutil.copytree(FIX / "match" / "capture", dst)
    if edit_events is not None:
        with gzip.open(dst / "log.jsonl.gz", "rt") as f:
            events = [json.loads(l) for l in f if l.strip()]
        events = edit_events(events)
        with gzip.open(dst / "log.jsonl.gz", "wt") as f:
            for e in events:
                f.write(json.dumps(e, separators=(",", ":")) + "\n")
    if edit_tokens is not None:
        tk = json.loads((dst / "tokens.json").read_text())
        (dst / "tokens.json").write_text(json.dumps(edit_tokens(tk)))
    return dst


def _attest(value):
    def edit(events):
        for e in events:
            if e.get("$e") == "RunHeader":
                prof = dict(e.get("profile") or {})
                ek = dict(prof.get("engine_kwargs") or {})
                ek.pop("async_scheduling", None)
                if value is not None:
                    ek["async_scheduling"] = value
                prof["engine_kwargs"] = ek
                e["profile"] = prof
        return events
    return edit


def _drop_step(step):
    def edit(events):
        return [e for e in events if not (e.get("$e") in ("StepStart", "SampleEnd") and int(e.get("step", -1)) == step)]
    return edit


def _run(desc, cap_dir: Path, *, lag=None, lag_source=None, transcript=None, bmap=None):
    capture = R.read_capture(cap_dir)
    if lag is None:
        lag, lag_source = L.lag_from_header(capture["header"])
    if transcript is None:
        cr = R.capture_request(capture, RID)
        transcript = [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]
    return R.check_record(desc, transcript, EOS, cap=CAP, bmap=bmap, request_id=RID, lag=lag, lag_source=lag_source, capture=capture)


def test_the_fixture_is_what_pad_says_and_the_unattested_header_is_refused_whatever_the_capture_shows(desc, tmp_path):
    """Reproduction of M-0311's lag-1 report, from the API: the header carries NO async_scheduling, so `lag_from_header` refuses by name and
    an explicit LAG=1 runs the other checks with R0 FAILED and `ok` False — although R3c/R0b/R3d/R2b all PASS on the engine's own evidence
    (steps [0..11] = the lag-1 prediction, measured lag 1, step 11 fed the EOS and sampled 504).  Evidence beside the attestation is not the
    attestation."""
    cap_dir = _copy_capture(tmp_path)
    capture = R.read_capture(cap_dir)
    with pytest.raises(ValueError, match="async_scheduling|attest"):
        L.lag_from_header(capture["header"])
    rep = _run(desc, cap_dir, lag=1, lag_source=f"UNATTESTED ({L.LAG_FIELD} absent)")
    ck = rep["checks"]
    assert rep["ok"] is False and ck["R0_lag_attestation"]["ok"] is False
    assert ck["R1_population"]["ok"] and ck["R1_population"]["guard_gates"] == ck["R1_population"]["guard_gates_expected"] == L.guard_gate_count(CAP, 1, 1) == 62
    assert ck["R2_activity"]["ok"] and ck["R2_activity"]["a_r"] == A_R and ck["R2_activity"]["served"] == 12 and ck["R2_activity"]["placeholder_sample"] == PLACEHOLDER_SAMPLE
    assert ck["R3c_capture_executed_set"]["ok"] and ck["R3c_capture_executed_set"]["engine_steps"] == list(range(12))
    assert ck["R3c_capture_executed_set"]["matches"] == {"0": False, "1": True}
    assert ck["R0b_lag_measured_consistency"]["ok"] and ck["R0b_lag_measured_consistency"]["measured"] == 1
    assert ck["R3d_placeholder_step"]["ok"] and ck["R3d_placeholder_step"] == {**ck["R3d_placeholder_step"], "step": PLACEHOLDER_STEP, "input_token": 0, "committed_eos": 0, "placeholder_sample": PLACEHOLDER_SAMPLE}
    assert ck["R2b_capture_samples"]["ok"] and ck["R4_assignment"]["ok"] and ck["R5_interface"]["ok"] and ck["R3_executed_set"]["ok"] is None


def test_an_injected_true_attestation_makes_lag_1_and_r0_pass_but_ok_stays_none_without_a_commit(desc, tmp_path):
    """The RunHeader with `profile.engine_kwargs.async_scheduling: true` (what sys's unit M-0279 writes): the reader yields LAG=1 from the one
    ruled field, R0 PASSES, every capture check PASSES -- and `ok` is None, not True: there is no binding map, R3 was not evaluated, and
    the capture checks are evidence beside R3, never in its place (F-pad-17 / pad's own rule)."""
    cap_dir = _copy_capture(tmp_path, _attest(True))
    lag, src = L.lag_from_header(R.read_capture(cap_dir)["header"])
    assert lag == 1 and src == L.LAG_FIELD
    rep = _run(desc, cap_dir)
    ck = rep["checks"]
    assert ck["R0_lag_attestation"]["ok"] is True and rep["lag"] == 1
    assert all(ck[k]["ok"] is True for k in ("R1_population", "R2_activity", "R3c_capture_executed_set", "R0b_lag_measured_consistency",
                                                "R3d_placeholder_step", "R2b_capture_samples", "R4_assignment", "R5_interface"))
    assert rep["ok"] is None and "NO record verdict" in rep["scope"]


def test_an_injected_false_attestation_makes_lag_0_and_the_placeholder_row_is_execution_past_eos(desc, tmp_path):
    """The same capture attesting `async_scheduling: false`: LAG=0 by the header, and the engine's 12th row (the placeholder) is execution
    past EOS -- R2 FAILS by name, R3c refutes lag 0 on the engine's own steps ([0..11] vs the prediction [0..10]), `ok` False.  The lag-0
    candidate does not describe this engine (pad's lag0 negative, now reached through the attestation instead of a command-line --lag)."""
    cap_dir = _copy_capture(tmp_path, _attest(False))
    lag, src = L.lag_from_header(R.read_capture(cap_dir)["header"])
    assert lag == 0 and src == L.LAG_FIELD
    rep = _run(desc, cap_dir)
    ck = rep["checks"]
    assert rep["ok"] is False and ck["R0_lag_attestation"]["ok"] is True
    assert ck["R2_activity"]["ok"] is False and "past EOS" in ck["R2_activity"]["why"]
    # R2 failed -> a_r unknown -> the capture checks are not evaluated (they need the activity): the FAIL is R2's, by name
    assert ck["R3c_capture_executed_set"]["ok"] in (None, False) if "R3c_capture_executed_set" in ck else True


def test_a_deleted_placeholder_step_in_the_engine_log_is_caught_by_r3c_r3d_and_r2b_each_by_name(desc, tmp_path):
    """Attestation True (lag 1) but the StepStart/SampleEnd of step 11 removed from the log while the transcript still carries 12 tokens
    (the Commit's word): the engine's executed set is [0..10] = the LAG-0 prediction, not the declared lag's -> R3c FAIL naming both
    predictions; R3d FAIL (no input row at step 11); R2b FAIL (11 capture samples vs 12 transcript tokens).  R1/R2/R4/R5 (the Program's own
    closure under the transcript) still PASS: the capture checks are what sees the deletion."""
    cap_dir = _copy_capture(tmp_path, lambda ev: _drop_step(PLACEHOLDER_STEP)(_attest(True)(ev)))
    full = R.capture_request(R.read_capture(_copy_capture(tmp_path / "orig")), RID)
    transcript = [full["sampled"][st] for st in full["executed"] if st in full["sampled"]]
    assert len(transcript) == 12 and transcript[A_R - 1] == 0 and transcript[A_R] == PLACEHOLDER_SAMPLE
    rep = _run(desc, cap_dir, transcript=transcript)
    ck = rep["checks"]
    assert ck["R2_activity"]["ok"] and ck["R1_population"]["ok"] and ck["R4_assignment"]["ok"] and ck["R5_interface"]["ok"]
    r3c = ck["R3c_capture_executed_set"]
    assert r3c["ok"] is False and r3c["engine_steps"] == list(range(11)) and r3c["matches"] == {"0": True, "1": False}
    assert ck["R3d_placeholder_step"]["ok"] is False and ck["R3d_placeholder_step"]["input_token"] is None
    assert ck["R2b_capture_samples"]["ok"] is False and ck["R2b_capture_samples"]["capture_samples_n"] == 11
    assert rep["ok"] is not True


def test_a_placeholder_step_fed_anything_but_the_committed_eos_fails_r3d(desc, tmp_path):
    """Attestation True; step 11's input_token_ids rewritten to feed 504 (its own future sample) instead of the EOS 0: DP-pad-02 (b)'s model
    of the engine is that the placeholder is fed the committed EOS (the guard's operand) -- R3d FAILS naming the fed token; R3c still PASSES
    (the executed SET is right) -- the two checks are independent."""
    def edit(events):
        events = _attest(True)(events)
        for e in events:
            if e.get("$e") == "StepStart" and int(e["step"]) == PLACEHOLDER_STEP:
                e["input_token_ids"] = [PLACEHOLDER_SAMPLE] * len(e.get("input_token_ids") or [0])
        return events
    rep = _run(desc, _copy_capture(tmp_path, edit))
    ck = rep["checks"]
    assert ck["R3c_capture_executed_set"]["ok"] is True
    assert ck["R3d_placeholder_step"]["ok"] is False and ck["R3d_placeholder_step"]["input_token"] == PLACEHOLDER_SAMPLE and ck["R3d_placeholder_step"]["committed_eos"] == 0
    assert rep["ok"] is not True


def test_the_harness_measured_lag_is_a_cross_check_never_a_source(desc, tmp_path):
    """(i) header unattested + tokens.json says measured lag 1: R0 still FAILS (ok False) -- the measurement does not attest; (ii) header
    attested True + measured 0: R0b FAILS by name (declared 1 vs measured 0) while R0 passes; (iii) attested True + the timing block absent:
    R0b is None (nothing to cross-check), not a PASS and not a FAIL."""
    cap_i = _copy_capture(tmp_path / "i", edit_tokens=lambda tk: {**tk, "timing": {**(tk.get("timing") or {}), "output_lag_measured": 1}})
    rep_i = _run(desc, cap_i, lag=1, lag_source="measured (tokens.json timing.output_lag_measured)")
    assert rep_i["ok"] is False and rep_i["checks"]["R0_lag_attestation"]["ok"] is False and rep_i["checks"]["R0b_lag_measured_consistency"]["ok"] is True
    cap_ii = _copy_capture(tmp_path / "ii", _attest(True), lambda tk: {**tk, "timing": {**(tk.get("timing") or {}), "output_lag_measured": 0}})
    rep_ii = _run(desc, cap_ii)
    assert rep_ii["checks"]["R0_lag_attestation"]["ok"] is True and rep_ii["checks"]["R0b_lag_measured_consistency"]["ok"] is False
    assert rep_ii["checks"]["R0b_lag_measured_consistency"]["measured"] == 0 and rep_ii["ok"] is not True
    cap_iii = _copy_capture(tmp_path / "iii", _attest(True), lambda tk: {k: v for k, v in tk.items() if k != "timing"})
    rep_iii = _run(desc, cap_iii)
    assert rep_iii["checks"]["R0b_lag_measured_consistency"]["ok"] is None


def _bmap_for(steps, rid=RID):
    return {"entries": [{"identity": {"request_id": rid, "step": s, "member": "logits"}} for s in steps]}


# F-pad-22 CLOSED by pad 82cb6b0eb (ok False whenever ANY evaluated check is False, failed_checks named): strict xfail flipped to PASS 07:30Z
def test_a_binding_map_that_agrees_with_the_declared_lag_does_not_make_ok_true_when_the_capture_refutes_it(desc, tmp_path):
    """F-pad-22.  Attestation True (lag 1); a binding map committing steps [0..11] (R3 PASS: the Commit's executed set is the lag-1
    prediction); the ENGINE's own log with step 11 deleted (R3c FAIL: the engine never ran the step the Commit claims).  The report must not
    read `ok: True`: the capture checks are 'evidence beside R3' -- a contradiction between the two is a FAIL of the record, not a footnote.
    Control: the intact capture with the same binding map is `ok: True` (the only way this tool ever says so)."""
    transcript = None
    full = R.capture_request(R.read_capture(_copy_capture(tmp_path / "orig", _attest(True))), RID)
    transcript = [full["sampled"][st] for st in full["executed"] if st in full["sampled"]]
    good = _run(desc, tmp_path / "orig" / "capture", transcript=transcript, bmap=_bmap_for(range(12)))
    assert good["checks"]["R3_executed_set"]["ok"] is True and good["checks"]["R3c_capture_executed_set"]["ok"] is True and good["ok"] is True
    bad_dir = _copy_capture(tmp_path / "del", lambda ev: _drop_step(PLACEHOLDER_STEP)(_attest(True)(ev)))
    bad = _run(desc, bad_dir, transcript=transcript, bmap=_bmap_for(range(12)))
    assert bad["checks"]["R3_executed_set"]["ok"] is True and bad["checks"]["R3c_capture_executed_set"]["ok"] is False
    assert bad["ok"] is not True, "F-pad-22: a FAILED capture check (R3c) left ok True"


def test_f_pad_23_record_verdict_on_the_early_stop_says_padding_exercised_and_lag_discriminated_with_nothing_committed(desc, tmp_path):
    """pad b2f45de35 (F-pad-23) on the real early stop with the attestation injected true and NO binding map: padding_exercised True
    (20 bot events), lag_discriminated True (the engine's own steps [0..11] refute lag 0), executed_set_evaluated False, ok None -- the
    two flags that were False on the attested llama row are True here, which is what makes THIS record the one that would say something
    once a Commit of the fixed Program exists."""
    rep = _run(desc, _copy_capture(tmp_path, _attest(True)))
    rv = rep["record_verdict"]
    assert rv["ok"] is None and rv["padding_exercised"] is True and rv["lag_discriminated"] is True
    assert rv["executed_set_evaluated"] is False and rv["values_evaluated"] is False and rv["attested"] is True
    assert rep["checks"]["R3a_commit_program"]["ok"] is None and "no binding map" in rep["checks"]["R3a_commit_program"]["why"]
