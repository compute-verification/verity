"""padrev — checks on pad's FIRST ATTESTED record through `lifted_record.py` (pad 341ba2d35 / da12ba007, M-0337 / M-0347): cp-sys's
llama32-1b B1 256/32 row, RunHeader `profile.engine_kwargs.async_scheduling = true` (sys SYS-04), Commit hot16's binding map (positional
`r0`), a LENGTH stop (32 served, a_r = cap, no ⊥).  Read-only copy under out/gen/sweep/evidence/pad/...; skipped by name where absent.

Attacks on COPIES / in-memory variants: the Commit's binding map with a committed step deleted (R3 by name, `ok` False); an extra committed
step past the cap; the wrong `--commit-request-id` (0 committed steps, refused by name, never a silent match); and the F-pad-06(b) count
recomputed independently from the lifted body's own leaf widths (Σ leaf_words over the non-guard leaves; the 32 two-word leaves are the
i32 tokens).  Also pinned: on a length stop BOTH lag predictions coincide — this record does not exercise the lag rule at all."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.lifting import record as R
from verity_vllm.ir import workload as W
from verity.ir.defs import REGISTRY
from verity_vllm.registry import lifted as L

ROW = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"
FIX = Path(__file__).resolve().parents[2] / "out" / "gen" / "sweep" / "evidence" / "pad" / ROW / "r16_0648_sys9_attested"
RID_ENGINE, RID_COMMIT = "5b7fee10-6a81-52b7-b730-133d6cc768da", "r0"
EOS, CAP = (128001,), 32

pytestmark = pytest.mark.skipif(not (FIX / "build_request" / "descriptor.json.gz").exists() or not (FIX / "commit.hot16" / "binding_map_p0.json.gz").exists(),
                                reason=f"pad's attested-row copy not on this tree: {FIX}")


@pytest.fixture(scope="module")
def desc():
    return R.load_descriptor(str(FIX / "build_request"))


@pytest.fixture(scope="module")
def capture():
    return R.read_capture(FIX / "match" / "capture")


@pytest.fixture(scope="module")
def bmap():
    with gzip.open(FIX / "commit.hot16" / "binding_map_p0.json.gz", "rt") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def transcript(capture):
    cr = R.capture_request(capture, RID_ENGINE)
    return [cr["sampled"][st] for st in cr["executed"] if st in cr["sampled"]]


def _check(desc, capture, transcript, bmap, request_id=RID_ENGINE, commit_request_id=RID_COMMIT, **kw):
    lag, src = L.lag_from_header(capture["header"])
    return R.check_record(desc, transcript, EOS, cap=CAP, bmap=bmap, request_id=request_id, commit_request_id=commit_request_id, lag=lag, lag_source=src,
                          capture=capture, **kw)


def test_the_attested_length_stop_reproduces_pads_report_and_does_not_exercise_the_lag_rule(desc, capture, transcript, bmap):
    """LAG=1 from the one attested field (nothing declared by hand); R1 +62 gates on the 18.76e9-gate body; R3 = the Commit's executed set
    [0..31]; ok True with R6 None -- pad's report, reproduced from the API.  Pinned: a_r = cap, so expected_served is 32 under BOTH lags
    (`matches` {0: True, 1: True}); the record is consistent with either lag candidate and says nothing about DP-pad-02 -- the first
    NON-trivial attested verdict needs an EOS-stopped Commit."""
    lag, src = L.lag_from_header(capture["header"])
    assert (lag, src) == (1, L.LAG_FIELD)
    assert len(transcript) == CAP and not (set(transcript) & set(EOS))
    rep = _check(desc, capture, transcript, bmap)
    ck = rep["checks"]
    assert rep["ok"] is True and ck["R6_values"]["ok"] is None and "NOT evaluated" in rep["scope"]
    assert ck["R1_population"]["guard_gates"] == 62 and ck["R1_population"]["unlifted_gates"] == 18_756_381_774
    assert ck["R2_activity"]["a_r"] == CAP and ck["R2_activity"]["stopped_by_eos"] is False and ck["R2_activity"]["placeholder_sample"] is None
    assert ck["R3_executed_set"]["ok"] and ck["R3_executed_set"]["committed_steps_n"] == CAP and ck["R3_executed_set"]["matches"] == {"0": True, "1": True}
    assert ck["R3c_capture_executed_set"]["engine_steps"] == list(range(CAP)) and ck["R3c_capture_executed_set"]["matches"] == {"0": True, "1": True}
    # the 33rd scheduler iteration carried no row of this request: a length stop runs no placeholder forward (pad's fact (2); global's G6 rule)
    assert capture["step_iterations"] == 33 and capture["engine_steps"] == 32 and sorted(capture["steps"]) == list(range(32))
    tr = capture["tokens"]["timing"]["requests"][RID_ENGINE]
    assert tr["finish_reason"] == "length" and tr["finish_step"] == 31 and tr["finish_iter"] == 32 and tr["token_steps"] == list(range(32))
    assert ck["R4_assignment"]["bot_nodes_total"] == 0 and ck["R5_interface"]["bot_events"] == 0


def test_a_committed_step_deleted_from_the_binding_map_fails_r3_by_name_and_ok_is_false(desc, capture, transcript, bmap):
    """The Commit's binding map with every identity of step 31 removed: R3 FAIL naming the missing step (declared activity not committed),
    while the engine's own log still shows [0..31] (R3c PASS) -- the two executed sets disagree and the report is NOT ok."""
    cut = {**bmap, "entries": [e for e in bmap["entries"] if int((e.get("identity") or {}).get("step", -1)) != 31]}
    assert R.executed_steps(cut, RID_COMMIT) == set(range(31))
    rep = _check(desc, capture, transcript, cut)
    r3 = rep["checks"]["R3_executed_set"]
    assert r3["ok"] is False and r3["declared_activity_not_committed"] == [31] and r3["declared_inactivity_executed"] == []
    assert rep["checks"]["R3c_capture_executed_set"]["ok"] is True
    assert rep["ok"] is False


# F-pad-22 CLOSED by pad 82cb6b0eb (ok False whenever ANY evaluated check is False, failed_checks named): strict xfail flipped to PASS 07:30Z
def test_the_wrong_engine_request_id_fails_r3c_and_must_not_leave_ok_true(desc, capture, transcript, bmap):
    """The two id spaces (pad's fact (1)): the capture by the engine's uuid, the binding map by the Commit's positional r<k>.  Given the
    COMMIT id for the engine side too, no capture row is the request's: R3c FAIL by name (engine_steps []), R2b FAIL (0 samples vs 32) --
    while R3 PASSES on the binding map.  `ok` must not be True over two FAILED checks (F-pad-22)."""
    rep = _check(desc, capture, transcript, bmap, request_id=RID_COMMIT, commit_request_id=RID_COMMIT)
    ck = rep["checks"]
    assert ck["R3_executed_set"]["ok"] is True
    assert ck["R3c_capture_executed_set"]["ok"] is False and ck["R3c_capture_executed_set"]["engine_steps"] == []
    assert ck["R2b_capture_samples"]["ok"] is False and ck["R2b_capture_samples"]["capture_samples_n"] == 0
    assert rep["ok"] is not True, "F-pad-22: ok True over FAILED R3c/R2b"


def test_an_extra_committed_step_and_the_wrong_commit_request_id_are_named_not_matched(bmap):
    """(i) an identity committed at step 32 (past the cap): the executed set gains a step no lag predicts -> R3's `declared_inactivity_executed`
    would name it; (ii) the binding map keyed by the ENGINE's uuid instead of the Commit's positional id: 0 committed steps -- the tool
    infers nothing (pad's fact (1)); a checker that fell back to 'any request' would have matched silently."""
    extra = {**bmap, "entries": bmap["entries"] + [{"identity": {"request_id": RID_COMMIT, "step": 32, "member": "logits"}}]}
    assert R.executed_steps(extra, RID_COMMIT) == set(range(33))
    assert R.executed_steps(bmap, RID_ENGINE) == set()
    ids = {str((e.get("identity") or {}).get("request_id", "r0")) for e in bmap["entries"] if "step" in (e.get("identity") or {})}
    assert ids == {RID_COMMIT, "*"}   # "*" = the registered weights identity (step -1), no request's


def test_f_pad_06b_the_lifted_layout_words_recomputed_from_the_bodys_own_leaf_widths(desc, capture, transcript, bmap):
    """pad's R4b (da12ba007): Σ leaf_words over every leaf of every non-guard node.  Recomputed here from the lifted body with an independent
    walk: total leaves, total lifted words, and the two-word leaves being exactly the 32 tokens (lifted width 33) -- every bf16/f32 leaf
    below 32 payload bits is one word.  Also: the guard nodes are OUTSIDE the count (their `guard_gate_count` gates are the Program's, not a
    step's values) -- the SPEC v0.5 candidate must say whether a lifted store carries them; today's count says it does not."""
    comp = W.request_component(desc, REGISTRY, None)
    lifted = L.lift_request_component(comp.fn, EOS, max_tokens=CAP, lag=1)
    guards = set(lifted.guard_nodes)
    leaves = words = two_word = guard_leaves = 0
    for k, n in enumerate(lifted.body.nodes):
        ws = R._leaf_widths(n.out)
        if k in guards:
            guard_leaves += len(ws)
            continue
        leaves += len(ws)
        words += sum(L.leaf_words(w - 1) for w in ws)
        two_word += sum(1 for w in ws if L.leaf_words(w - 1) == 2)
    assert (leaves, words, two_word) == (209_825_824, 209_825_856, 32)
    assert all(L.leaf_words(w) == 1 for w in (1, 16, 17, 31)) and L.leaf_words(32) == 2
    rep = _check(desc, capture, transcript, bmap)
    r4b = rep["checks"]["R4b_lifted_layout_words"]
    assert r4b["ok"] and r4b["leaves_total"] == leaves and r4b["lifted_words_total"] == words and r4b["one_word_per_leaf_total"] == leaves
    assert sum(r4b["lifted_words_by_event"].values()) == words and len(r4b["lifted_words_by_event"]) == CAP
    assert all(int(v) - int(r4b["one_word_per_leaf_by_event"][s]) == 1 for s, v in r4b["lifted_words_by_event"].items())   # one i32 token per event
    assert guard_leaves > 0 and guard_leaves not in (0, leaves)   # reported apart, never folded into the population


def test_f_pad_23_record_verdict_says_this_ok_true_is_trivial_for_padding_and_names_the_bound_program(desc, capture, transcript, bmap):
    """pad b2f45de35 (F-pad-23): beside `ok` the report now carries what the True IS: padding_exercised False (0 bot), lag_discriminated
    False (both predictions coincide on a length stop), executed_set_evaluated True, commit_binds_fixed_program True (R3a: the Commit's
    binding map names the fixed Program's digest), values_evaluated False, attested True; the statement spells 'padding trivial' and 'lag
    not discriminated'.  Pinned so a finalizer reading record_verdict cannot take this for a padded record verified."""
    rep = _check(desc, capture, transcript, bmap)
    rv = rep["record_verdict"]
    assert rv["ok"] is True and rv["padding_exercised"] is False and rv["lag_discriminated"] is False
    assert rv["executed_set_evaluated"] is True and rv["commit_binds_fixed_program"] is True and rv["values_evaluated"] is False and rv["attested"] is True
    assert "padding trivial" in rv["statement"] and "lag not discriminated" in rv["statement"] and "R6" in rv["statement"]
    r3a = rep["checks"]["R3a_commit_program"]
    assert r3a["ok"] is True and r3a["bound"] == "component" and r3a["binding_map_program_digest"] == r3a["fixed_program_digest"] == bmap["program_digest"]


def test_r3a_a_commit_of_another_program_is_named_and_only_the_executed_set_stays_comparable(desc, capture, transcript, bmap):
    """The binding map's program_digest is not the fixed Program's (today's early-stop practice: the Commit binds the served-shape derivation,
    sys M-0402): R3a None with bound 'other', the scope string says 'executed-set evidence only', record_verdict.commit_binds_fixed_program
    False -- and `ok` stays True (pad: not a failure of the record, the scope of what such a Commit attests).  NOTE for readers: `ok`
    alone does not distinguish this from a Commit of the fixed Program; record_verdict does -- read it."""
    other = {**bmap, "program_digest": "f" * 64}
    rep = _check(desc, capture, transcript, other)
    r3a = rep["checks"]["R3a_commit_program"]
    assert r3a["ok"] is None and r3a["bound"] == "other" and r3a["binds_fixed_program"] is False
    assert "executed-set evidence only" in rep["scope"] and rep["record_verdict"]["commit_binds_fixed_program"] is False
    assert rep["checks"]["R3_executed_set"]["ok"] is True and rep["ok"] is True
