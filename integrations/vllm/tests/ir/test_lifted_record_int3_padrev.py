"""padrev — checks on pad's FIRST real early-stop record with an ATTESTED lag AND a Commit (pad 47d35e948, M-0442; sys M-0402's earlystop row
on the int-3 candidate a2dabe493, cp-sys): SmolLM2-135M LP 32 cap 32, EOS 0 at event 10, a_r = 11, placeholder step 11 (sample 504), RunHeader
`async_scheduling = true`, the FAILED cap-shape Commit's binding maps (3 pairs; 4,741 entries each; identities at steps [0..11] only).
Read-only copy under out/gen/sweep/evidence/pad/...; skipped by name where absent.  Every mutation is on an in-memory copy of the binding map.

What is attacked: the report's reproduction from the API (ok True, LAG 0 REFUTED by the Commit's own executed set); the placeholder step's
identities deleted from the Commit (a lag-0-shaped Commit under a lag-1 header -> R3 by name, ok False); a declared lag that contradicts the
header (refused); the run row's counts flipped (R3b by name); the three pairs' agreement; and F-pad-27 -- what the SAME tool says about the
record the lifting NEEDS: a Commit that binds every identity of every ⊥ step (as enc(⊥) -- the lifted record half, A3 / F-pad-06(b)) reads as
'declared inactivity executed' at R3, because R3 equates 'an identity is committed at step s' with 's executed'.  Today's rule is right for
today's Commit and wrong for the lifted one: the lifted binding map must carry the activity per identity (or per (request, step)) so R3 can
read the executed set from the Commit without values -- else R3 collapses into R6."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.lifting import record as R
from verity_vllm.registry import lifted as L

ROW = "smollm2-135m__bf16__l40s__tp1__b1__i32__o32__earlystop__greedy__bi-eager"
FIX = Path(__file__).resolve().parents[2] / "out" / "gen" / "sweep" / "evidence" / "pad" / ROW / "r16_0730_sys_int3c_earlystop"
RID, CRID = "de2d6bfc-096d-5c10-beb0-293ff914262d", "r0"
EOS, CAP, A_R, LAG = (0,), 32, 11, 1
PLACEHOLDER_STEP, PLACEHOLDER_SAMPLE = 11, 504

pytestmark = pytest.mark.skipif(not (FIX / "build_request" / "descriptor.json.gz").exists() or not (FIX / "commit" / "binding_map_p0.json.gz").exists(),
                                reason=f"pad's read-only copy of sys's int-3 earlystop record not on this tree: {FIX}")


@pytest.fixture(scope="module")
def desc():
    return R.load_descriptor(str(FIX / "build_request"))


@pytest.fixture(scope="module")
def capture():
    return R.read_capture(FIX / "match" / "capture")


@pytest.fixture(scope="module")
def bmaps():
    out = []
    for k in range(3):
        with gzip.open(FIX / "commit" / f"binding_map_p{k}.json.gz", "rt") as f:
            out.append(json.load(f))
    return out


@pytest.fixture(scope="module")
def run_row():
    rows = [json.loads(l) for l in (FIX / "commit" / "runs.jsonl").read_text().splitlines() if l.strip()]
    return next(r for r in rows if r.get("pair") == 0 and r.get("arm") == "instrumented")


@pytest.fixture(scope="module")
def transcript(capture):
    cr = R.capture_request(capture, RID)
    return [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]


def _check(desc, capture, transcript, bmap, run_row=None, **kw):
    lag, src = L.lag_from_header(capture["header"])
    return R.check_record(desc, transcript, EOS, cap=CAP, bmap=bmap, request_id=RID, commit_request_id=CRID, lag=lag, lag_source=src,
                          capture=capture, run_row=run_row, **kw)


def _with_steps(bmap, steps, drop=False):
    """A copy of the binding map with identities of `steps` ADDED (cloned from step 5's r0 identities) or DROPPED."""
    ents = [e for e in bmap["entries"]]
    if drop:
        ents = [e for e in ents if int((e.get("identity") or {}).get("step", -1)) not in set(steps)]
    else:
        proto = [e for e in ents if int((e.get("identity") or {}).get("step", -1)) == 5 and str((e.get("identity") or {}).get("request_id")) == CRID]
        assert proto
        for s in steps:
            for e in proto:
                ents.append({**e, "identity": {**e["identity"], "step": s}})
    return {**bmap, "entries": ents}


def test_pads_report_reproduced_the_commit_of_record_refutes_lag_0_and_binds_no_bot_step(desc, capture, transcript, bmaps, run_row):
    """From the API, nothing declared about the lag: ok True; R0 LAG=1 from the one field; R2 a_r 11 / 12 served / placeholder 504; R3 the
    Commit commits identities at EXACTLY [0..11] = the LAG-1 prediction, `matches {0: False, 1: True}` -- LAG 0 REFUTED by the Commit of
    record (R3c/R0b/R3d say the same from the engine and the harness: four independent witnesses); R3a bound=component (the cap-shape
    manifest, the thing ruling 35 changes); R3e None with bot_population_bound False (no identity of any of the 20 ⊥ steps); R3b 11/12;
    record_verdict padding_exercised True, lag_discriminated True, executed_set_evaluated True, values_evaluated False."""
    assert len(transcript) == A_R + LAG and transcript[A_R - 1] == 0 and transcript[A_R] == PLACEHOLDER_SAMPLE
    rep = _check(desc, capture, transcript, bmaps[0], run_row)
    ck = rep["checks"]
    assert rep["ok"] is True and rep["lag"] == 1 and ck["R0_lag_attestation"]["ok"] is True
    assert ck["R2_activity"]["a_r"] == A_R and ck["R2_activity"]["served"] == 12 and ck["R2_activity"]["placeholder_sample"] == PLACEHOLDER_SAMPLE
    r3 = ck["R3_executed_set"]
    assert r3["ok"] is True and r3["committed_steps"] == list(range(12)) and r3["matches"] == {"0": False, "1": True}
    assert ck["R3c_capture_executed_set"]["matches"] == {"0": False, "1": True} and ck["R0b_lag_measured_consistency"]["measured"] == 1
    assert ck["R3d_placeholder_step"]["ok"] and ck["R3d_placeholder_step"]["input_token"] == 0
    assert ck["R3a_commit_program"]["ok"] is True and ck["R3a_commit_program"]["bound"] == "component"
    r3e = ck["R3e_bot_population"]
    assert r3e["ok"] is None and r3e["bot_steps_n"] == CAP - 12 and r3e["bot_population_bound"] is False and r3e["bot_steps_with_identities"] == []
    assert ck["R3b_run_row"]["ok"] is True and (ck["R3b_run_row"]["generated_tokens"], ck["R3b_run_row"]["engine_steps"]) == (A_R, A_R + LAG)
    rv = rep["record_verdict"]
    assert rv["padding_exercised"] is True and rv["lag_discriminated"] is True and rv["executed_set_evaluated"] is True
    assert rv["bot_population_bound"] is False and rv["values_evaluated"] is False and rv["attested"] is True
    assert "NOT bound" in rv["statement"] and "R6" in rv["statement"]
    # the three pairs commit the same executed set (one execution, three Commit runs)
    assert all(R.executed_steps(b, CRID) == set(range(12)) for b in bmaps)
    assert ck["R1_population"]["guard_gates"] == 62 and ck["R1_population"]["unlifted_gates"] == 538_849_332


def test_the_placeholder_steps_identities_deleted_from_the_commit_is_a_lag_0_commit_under_a_lag_1_header_and_fails_r3_by_name(desc, capture, transcript, bmaps):
    """The Commit's identities of step 11 (the placeholder) removed: the Commit's executed set becomes [0..10] = the LAG-0 prediction while the
    header attests lag 1 -> R3 FAIL `declared_activity_not_committed [11]`, `matches {0: True, 1: False}`; the ENGINE (R3c) still says [0..11]:
    the Commit and the engine disagree, ok False.  A deleted step in the Commit is never 'not required'."""
    cut = _with_steps(bmaps[0], [PLACEHOLDER_STEP], drop=True)
    assert R.executed_steps(cut, CRID) == set(range(11))
    rep = _check(desc, capture, transcript, cut)
    r3 = rep["checks"]["R3_executed_set"]
    assert r3["ok"] is False and r3["declared_activity_not_committed"] == [PLACEHOLDER_STEP] and r3["matches"] == {"0": True, "1": False}
    assert rep["checks"]["R3c_capture_executed_set"]["ok"] is True
    assert rep["ok"] is False and "R3_executed_set" in rep["failed_checks"]


def test_a_declared_lag_that_contradicts_the_attested_header_is_refused_by_the_cli(tmp_path):
    """`--lag 0` on the CLI with a header attesting async_scheduling true: refused by name ('contradicts the attested field; the header rules');
    the header is the only source (RULING 21) and the command line cannot overrule it."""
    out = tmp_path / "rep.json"
    with pytest.raises(SystemExit):
        R.main(["--build-request", str(FIX / "build_request"), "--capture", str(FIX / "match" / "capture"), "--commit", str(FIX / "commit"), "--pair", "0",
                "--eos", "0", "--cap", str(CAP), "--request-id", RID, "--commit-request-id", CRID, "--lag", "0", "--lag-source", L.LAG_FIELD, "--out", str(out)])
    assert not out.exists()


def test_the_run_rows_counts_are_checked_by_name_generated_is_a_r_and_engine_steps_is_a_r_plus_lag(desc, capture, transcript, bmaps, run_row):
    """R3b (corrected by pad on this record, 711ff8655 / M-0597): generated_tokens == a_r (the driver's tokens, EOS in, placeholder out) and
    engine_steps == iterations == a_r + LAG (the loop count, stop-reason independent).  A run row claiming engine_steps 11 (the lag-0 count),
    iterations 11, or generated_tokens 12 (the placeholder reported as output) FAILS R3b by name and vetoes ok; a row carrying ONLY iterations
    (12) is EVALUATED and True by that one count (a present count field is a fact, pad M-0667); a row without ANY of the three counts is None
    (not evaluated), never a pass."""
    counts = ("engine_steps", "generated_tokens", "iterations")
    assert run_row.get("iterations") == A_R + LAG, "the int-3 row carries the loop count itself"
    for bad in ({**run_row, "engine_steps": 11}, {**run_row, "generated_tokens": 12}, {**run_row, "iterations": 11}):
        rep = _check(desc, capture, transcript, bmaps[0], bad)
        assert rep["checks"]["R3b_run_row"]["ok"] is False and rep["ok"] is False and "R3b_run_row" in rep["failed_checks"]
    rep = _check(desc, capture, transcript, bmaps[0], {k: v for k, v in run_row.items() if k not in ("engine_steps", "generated_tokens")})
    r3b = rep["checks"]["R3b_run_row"]
    assert r3b["ok"] is True and r3b["iterations"] == A_R + LAG and r3b["engine_steps"] is None and r3b["generated_tokens"] is None and rep["ok"] is True
    rep = _check(desc, capture, transcript, bmaps[0], {k: v for k, v in run_row.items() if k not in counts})
    assert rep["checks"]["R3b_run_row"]["ok"] is None and rep["ok"] is True


def test_f_pad_27_the_commit_the_lifting_needs_reads_as_declared_inactivity_executed_at_r3_today(desc, capture, transcript, bmaps):
    """F-pad-27 (design NOTE, measured): a Commit that binds every identity of every ⊥ step -- what the lifted record half REQUIRES (enc(⊥) per
    ⊥ value, A3 / F-pad-06(b)) -- makes R3's executed set [0..31]: `declared_inactivity_executed [12..31]`, ok False, while R3e reads
    bot_population_bound True.  R3 equates 'an identity is committed at s' with 's executed'; right for today's Commit (executed steps only),
    wrong for the lifted one.  Pinned as today's behaviour so the change is deliberate: the lifted binding map must carry the ACTIVITY per
    identity (or per (request, step)) and R3 must read the executed set from it -- otherwise R3 collapses into R6 (values)."""
    lifted_like = _with_steps(bmaps[0], range(12, CAP))
    assert R.executed_steps(lifted_like, CRID) == set(range(CAP))
    rep = _check(desc, capture, transcript, lifted_like)
    r3, r3e = rep["checks"]["R3_executed_set"], rep["checks"]["R3e_bot_population"]
    assert r3["ok"] is False and r3["declared_inactivity_executed"] == list(range(12, 20))     # first 8 named
    assert r3e["bot_population_bound"] is True and "R6" in r3e["why"]
    assert rep["ok"] is False and rep["record_verdict"]["bot_population_bound"] is True
