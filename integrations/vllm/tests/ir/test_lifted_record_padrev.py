"""padrev — adversarial tests for the record-half checker CANDIDATE `verity.ir/lifted_record.py` (lane/r16-pad 859dc5d22).

The checker has not been pointed at any record (DP-pad-02 unruled; no Commit of an early-stop row exists).  These tests attack the
checker itself on the TINY fixture: A3 (a placeholder step's rows deleted from the binding map; a lag declared to fit the record),
A6/A8 (the transcript file the operator hands in is the prover's oracle — tampering with it must not produce a PASS), and the
report's top-level `ok` (it must never read as a pass when the executed set was not checked or a lag is undeclared)."""
from __future__ import annotations

import pytest

from verity.ir import Program
from verity_vllm.lifting import record as R
from verity.ir.codec import encode_program
from verity_vllm.registry import b1, lifted as L
from tests.ir.test_serve3 import D_MAX, LP, TINY, Y_RAW

CAP = D_MAX + 1
BOUNDARY = "Final_v2"
EOS = (17,)
A_R = 4
SERVED0 = Y_RAW[:A_R]  # lag-0 record: events 0..3, the last is the EOS
SERVED1 = Y_RAW[:A_R + 1]  # lag-1 record: + the placeholder step's discarded sample (= the unlifted y_4)


@pytest.fixture(scope="module")
def desc():
    return encode_program(Program(b1.serve_v2(TINY, LP, D_MAX)))


def _bmap(steps, rid="r0"):
    entries = []
    for st in steps:
        for op, member in (("model.embed_tokens", "in"), ("model.layers.0.mlp.down_proj", "0"), ("sampler", "sampled_token_ids")):
            entries.append({"name": f"{op}/{member}@{st}", "identity": {"request_id": rid, "step": st, "op_path": op, "output_member": member,
                                                                       "element_range": [0, 1]}, "encoding": {"dtype": "i64"}, "element_offset": 0})
    return {"schema": "test", "entries": entries}


def _check(desc, transcript, steps, lag, bmap="default", **kw):
    """The honest caller: the lag names the ONE attested field RULING 21 (coord M-0213) prescribes; tests that attack the attestation
    pass their own `lag_source`."""
    bm = _bmap(steps) if bmap == "default" else bmap
    kw.setdefault("lag_source", L.LAG_FIELD)
    return R.check_record(desc, transcript, EOS, cap=CAP, bmap=bm, request_id="r0", boundary_prefix=BOUNDARY, lag=lag, **kw)


def test_the_two_honest_records_pass_only_under_their_own_declared_lag(desc):
    ok0 = _check(desc, SERVED0, range(A_R), 0)
    ok1 = _check(desc, SERVED1, range(A_R + 1), 1)
    assert ok0["ok"] and ok1["ok"]
    # cross-declared: each fails by name, at the activity rule (R2) — the record is not reinterpreted to fit
    x01 = _check(desc, SERVED0, range(A_R), 1)  # a lag-0 record declared lag 1: the placeholder step is declared but not served
    x10 = _check(desc, SERVED1, range(A_R + 1), 0)  # a lag-1 record declared lag 0: execution past EOS
    assert not x01["ok"] and "declared activity not executed" in x01["checks"]["R2_activity"]["why"]
    assert not x10["ok"] and "execution past EOS" in x10["checks"]["R2_activity"]["why"]


def test_a3_deleting_the_placeholder_rows_from_the_binding_map_is_caught_under_lag_1(desc):
    """The record of a lag-1 engine with the placeholder step's identities removed from the binding map ("not required") — the
    transcript still carries the sample.  R3 must name the missing step; the report must not be ok."""
    rep = _check(desc, SERVED1, range(A_R), 1)  # binding map has steps 0..3 only; transcript has 5 events
    r3 = rep["checks"]["R3_executed_set"]
    assert not rep["ok"] and r3["ok"] is False and r3["declared_activity_not_committed"] == [A_R]
    assert r3["matches"]["0"] is True and r3["matches"]["1"] is False  # the other candidate is REPORTED, never silently picked


def test_a3_deleting_the_placeholder_sample_from_the_transcript_is_caught_under_lag_1(desc):
    """The other half of the same attack: the binding map keeps the placeholder step, the transcript file drops its sample so the
    record looks like a clean lag-0 stop.  Under the declared lag 1 the activity rule refuses (n_r = a_r, not a_r + 1)."""
    rep = _check(desc, SERVED0, range(A_R + 1), 1)
    assert not rep["ok"] and rep["checks"]["R2_activity"]["ok"] is False
    assert "declared activity not executed" in rep["checks"]["R2_activity"]["why"]


def test_a6_the_transcript_file_is_an_oracle_the_prover_writes_and_a_forged_eos_is_refused(desc):
    """The operator-supplied token array is the checker's ONLY source of activity.  Replacing the committed token at event 2 by the
    EOS (a forged early stop) shortens the declared activity: the binding map's executed set then contradicts it (R3) — the executed
    set, not the file, is what catches the forgery.  A forged transcript with a MATCHING forged binding map is not detectable on disk
    (R6 is not evaluated here): the report must say so, and `ok` must not be a bare True in that case (F-pad-17)."""
    forged = list(SERVED0[:2]) + [17]  # EOS at event 2 instead of 3
    rep = _check(desc, forged, range(A_R), 0)
    assert not rep["ok"] and rep["checks"]["R3_executed_set"]["declared_inactivity_executed"] == [3]
    both = _check(desc, forged, range(3), 0)  # binding map forged to match
    assert both["checks"]["R6_values"]["ok"] is None  # values not checked on disk — the report cannot establish the record


def test_f_pad_17_ok_without_a_binding_map_or_without_an_attested_lag_source_is_not_a_pass(desc):
    """F-pad-17(a) CLOSED by pad f098fb451.  No binding map: R3 is None and R6 is None — nothing about the EXECUTION was checked, only the
    Program's own closure.  `ok` is None (not True) and the scope string says so; a downstream reader cannot take it for a record-half PASS."""
    rep = _check(desc, SERVED0, range(A_R), 0, bmap=None)
    assert rep["checks"]["R3_executed_set"]["ok"] is None and rep["checks"]["R6_values"]["ok"] is None
    assert rep["ok"] is None and "NO record verdict" in rep["scope"] and rep["candidate"] == L.DP_PAD_02
    # and a Program-side FAIL is still False, never None (a closure failure is a failure whatever was not checked)
    bad = _check(desc, list(SERVED0[:2]) + [17], range(A_R), 0, bmap=None)   # forged EOS at event 2: R2/R4 disagree with the Program
    assert bad["ok"] is not True


def test_f_pad_17b_lag_source_unspecified_is_recorded_not_silently_accepted(desc):
    """F-pad-17(b) CLOSED by pad f098fb451: an unattested lag FAILS R0 by name, whatever the other checks say (M-0168; RULING 21)."""
    rep = _check(desc, SERVED1, range(A_R + 1), 1, lag_source="unspecified")
    assert rep["lag"] == 1 and rep["lag_source"] == "unspecified"
    assert rep["ok"] is False and rep["checks"]["R0_lag_attestation"]["ok"] is False and L.LAG_FIELD in rep["checks"]["R0_lag_attestation"]["why"]
    assert rep["checks"]["R2_activity"]["ok"] is True  # the other checks are still evaluated and reported — the verdict is the attestation's


# F-pad-18 CLOSED by pad 850393f65 (attested := lag_source == L.LAG_FIELD exactly): strict xfail flipped to PASS 07:08Z
def test_f_pad_18_the_attestation_is_the_ruled_field_not_any_string_outside_a_blacklist(desc):
    """RULING 21 (coord M-0213): 'one source, two readers, NO default'.  `check_record` marks the lag attested when `lag_source` is any
    non-empty string not in ('unspecified', 'list', 'file', 'cli') — so `--lag 1 --lag-source operator` on a synthetic record passes R0
    although no header field was read.  The ruled field is the only attestation; anything else is the operator's word."""
    rep = _check(desc, SERVED1, range(A_R + 1), 1, lag_source="operator says async")
    assert rep["checks"]["R0_lag_attestation"]["ok"] is False and rep["ok"] is False


def test_ruling_21_one_source_two_readers_the_lag_reader_reads_gp01s_fixture_header_and_refuses_the_cli_default():
    """The lifted Program's LAG and GP-01's eos_lag_rows must be the SAME value read from the SAME attested field (RULING 21 (b)).  pad's
    reader is pinned to the field name and to the header SHAPE GM-01's fixtures record (RunHeader.profile.engine_kwargs, test_global_match
    `preamble`): resolved True → 1, False → 0; absent / None (the CLI default) / an int 1 are refused by name — never defaulted."""
    assert L.LAG_FIELD == "RunHeader.profile.engine_kwargs.async_scheduling"
    hdr = {"$e": "RunHeader", "profile": {"vllm": "toy", "engine_kwargs": {"enforce_eager": True, "max_num_seqs": 2}}, "workload": {}}
    with pytest.raises(ValueError, match="not attested"):
        L.lag_from_header(hdr)
    hdr["profile"]["engine_kwargs"]["async_scheduling"] = None       # the CLI default is not a resolved value
    with pytest.raises(ValueError, match="not the resolved boolean"):
        L.lag_from_header(hdr)
    hdr["profile"]["engine_kwargs"]["async_scheduling"] = 1          # an int is not the attestation either (1 == True in Python)
    with pytest.raises(ValueError, match="not the resolved boolean"):
        L.lag_from_header(hdr)
    hdr["profile"]["engine_kwargs"]["async_scheduling"] = True
    assert L.lag_from_header(hdr) == (1, L.LAG_FIELD)
    hdr["profile"]["engine_kwargs"]["async_scheduling"] = False
    assert L.lag_from_header(hdr) == (0, L.LAG_FIELD)
    # a header whose engine_kwargs live at the top level only (an older recorder) is NOT read: the ruled path is profile.engine_kwargs
    with pytest.raises(ValueError, match="not attested"):
        L.lag_from_header({"$e": "RunHeader", "engine_kwargs": {"async_scheduling": True}, "profile": {}})


def test_lag_outside_the_candidates_and_a_missing_lag_are_refused_by_name(desc):
    with pytest.raises(ValueError):
        _check(desc, SERVED0, range(A_R), 2)
    with pytest.raises(TypeError):
        R.check_record(desc, SERVED0, EOS, cap=CAP, bmap=_bmap(range(A_R)), request_id="r0", boundary_prefix=BOUNDARY)  # type: ignore[call-arg]


def test_the_placeholder_sample_is_reported_and_equals_the_unlifted_y_at_a_r(desc):
    rep = _check(desc, SERVED1, range(A_R + 1), 1)
    r2 = rep["checks"]["R2_activity"]
    assert r2["a_r"] == A_R and r2["placeholder_sample"] == Y_RAW[A_R] and r2["executed_expected"] == A_R + 1
    r4 = rep["checks"]["R4_assignment"]
    assert r4["active_events"] == A_R + 1 and r4["inactive_boundaries_bot"] and r4["active_prefix_all_active"]
    assert rep["checks"]["R5_interface"]["bot_events"] == CAP - A_R - 1
