"""padrev — adversarial tests of `lifted_record.check_workload_record` (pad 696b2c012 / 82cb6b0eb, M-0357: the MIXED BATCH record half,
W1 population / W2 chronology / W3 no crossing over the per-request R-checks) on pad's synthetic B2 fixture (TINY: A cap 9 stops on the EOS
at event 3, B cap 4 arrives at 2 and runs to ITS cap), reusing pad's capture/binding-map builders so the attacks are about the RULES.

Attacks: (1) the engine admits B one step EARLY (declared arrival 2, rows from step 1) -- the chronology names the early row AND the missing
last row; (2) attested LAG=1 while engine and Commit both behaved as lag 0 (no placeholder row, no placeholder commit) -- the deleted
placeholder step named at the workload level by both views and by A's own R3/R3c; (3) a declared request the engine and the Commit never
ran at all -- every step of its window named, the population count unchanged (it is the manifest's); (4) `n_live_by_step` (the S operand's
value) equals padrev's independent live reference summed over the requests programmed for the step, under both lags; and a label-swap
statement: the W-checks are structural (sets of request ids per step), so two co-live requests' identities exchanged in the Commit are
NOT theirs to see -- that is R6/C2's (values), stated so nobody reads W2 PASS as more than it is."""
from __future__ import annotations

import pytest

from verity_vllm.lifting import record as R
from verity_vllm.registry import lifted as L
from tests.ir import test_lifted_record as TR
from tests.ir.test_lifted_record import A_R, B_TOKENS, BOUNDARY, CAP, CAP_B, EOS, UUID_A, UUID_B, Y_RAW, _bmap2, _capture2, _header, _reqs
from tests.ir.test_lifted_tiny_padrev import padrev_live_ref


@pytest.fixture(scope="module")
def desc():
    return TR.desc.__wrapped__()


@pytest.fixture(scope="module")
def desc_b():
    return TR.desc_b.__wrapped__()


def _cap2_with_facts(tmp_path, reqs, *, lag, header):
    """pad's synthetic capture WITH the record's own pairing facts (tokens.json requests) where the builder offers them (pad 711ff8655+,
    F-pad-25: without a fact source W0 is None and the workload `ok` is None 'pairing unverified'); on an older builder, plain."""
    import inspect
    if "facts" in inspect.signature(_capture2).parameters:
        return _capture2(tmp_path, reqs, lag=lag, header=header, facts=True)
    return _capture2(tmp_path, reqs, lag=lag, header=header)


def _run(desc, desc_b, tmp_path, *, lag, a_tr, b_tr=B_TOKENS, engine=None, commit=None, reqs=None):
    """engine: list of (uuid, samples, arrive) the ENGINE ran (default = the declaration); commit: (a_steps, b_steps) the Commit committed."""
    lg, src = L.lag_from_header(_header(lag))
    engine = engine if engine is not None else [(UUID_A, a_tr, 0), (UUID_B, b_tr, 2)]
    cd = R.read_capture(_cap2_with_facts(tmp_path, engine, lag=lag, header=_header(lag)))
    a_steps, b_steps = commit if commit is not None else (range(A_R + lag), range(2, 2 + CAP_B))
    bm = _bmap2(a_steps, b_steps)
    reqs = reqs if reqs is not None else _reqs(desc, desc_b, a_tr, b_tr)
    return R.check_workload_record(reqs, EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)


@pytest.mark.parametrize("lag", [0, 1])
def test_p9_n_live_by_step_equals_the_independent_live_reference_summed_over_the_requests_programmed_for_the_step(desc, desc_b, tmp_path, lag):
    """The S operand's value at the record level: for every engine step of the horizon, n_live_t == Σ_r padrev_live_ref(transcript_r)[t − a_r]
    over the requests whose window contains t -- from each request's OWN tokens, never from the rows the engine ran (the positive record
    happens to agree; the negatives below are where they differ)."""
    a_tr = Y_RAW[:A_R + lag]
    rep = _run(desc, desc_b, tmp_path, lag=lag, a_tr=a_tr)
    assert rep["ok"] is True, rep.get("failed_checks")
    ref_a = padrev_live_ref(a_tr + [L.enc_bot(32)] * (CAP - len(a_tr)), EOS, CAP, lag)[0]
    ref_b = padrev_live_ref(B_TOKENS, EOS, CAP_B, lag)[0]
    assert sum(ref_a) == A_R + lag and ref_b == [1] * CAP_B
    want = {str(t): (ref_a[t] if t < CAP else 0) + (ref_b[t - 2] if 0 <= t - 2 < CAP_B else 0) for t in range(CAP)}
    assert rep["checks"]["W2_chronology"]["n_live_by_step"] == want
    assert rep["checks"]["W3_no_crossing"]["a_r_per_request"] == {UUID_A: A_R, UUID_B: CAP_B}


def test_an_early_admission_of_b_is_named_at_the_early_step_and_at_the_missing_last_step(desc, desc_b, tmp_path):
    """Declared: B arrives at 2.  Engine: B's prefill at step 1, its rows at 1..4.  The Program (declaration + tokens) says B is live at
    2..5: the engine row at 1 is outside B's window (named as an engine step outside the population or a mismatch at 1) and the row at 5 is
    missing (mismatch at 5).  The Commit, keyed by the declaration, is consistent -- the ENGINE is what disagrees."""
    a_tr = Y_RAW[:A_R]
    rep = _run(desc, desc_b, tmp_path, lag=0, a_tr=a_tr, engine=[(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 1)])
    assert rep["ok"] is False
    w2 = rep["checks"]["W2_chronology"]
    assert w2["ok"] is False
    named = set(w2["engine_vs_program_mismatch"]) | set(w2["engine_steps_outside_population"])
    assert "1" in named and "5" in w2["engine_vs_program_mismatch"] and UUID_B in w2["engine_vs_program_mismatch"]["5"]
    assert w2["commit_vs_program_mismatch"] == {}
    # B's own view: the capture has B at [1..4] where its declaration predicts [2..5]
    rb = next(p for p in rep["requests"] if p["request_id"] == UUID_B)
    assert rb["checks"]["R3c_capture_executed_set"]["ok"] is False and rb["checks"]["R3c_capture_executed_set"]["engine_steps"] == [1, 2, 3, 4]


def test_under_an_attested_lag_1_an_engine_and_a_commit_that_behaved_as_lag_0_are_refused_by_name_at_both_levels(desc, desc_b, tmp_path):
    """The header attests async_scheduling (LAG=1), so A's placeholder step 4 is declared activity (fed the EOS, sample committed).  Engine
    log without a row of A at 4 and a Commit without step 4: the Program says live -> engine_vs_program_mismatch {4: [A]} AND
    commit_vs_program_mismatch {4: [A]}; A's R3 names the missing step, A's R3c refutes the declared lag ([0..3] = the lag-0 set).  The
    record's transcript is the 4 committed tokens (no placeholder sample): A's R2 fails too ('declared activity not executed' under LAG=1)
    -- so the test drives the workload check with the 5-token transcript A WOULD have committed, isolating the chronology."""
    a_tr5 = Y_RAW[:A_R + 1]
    rep = _run(desc, desc_b, tmp_path, lag=1, a_tr=a_tr5, engine=[(UUID_A, Y_RAW[:A_R], 0), (UUID_B, B_TOKENS, 2)], commit=(range(A_R), range(2, 2 + CAP_B)))
    assert rep["ok"] is False
    w2 = rep["checks"]["W2_chronology"]
    assert w2["engine_vs_program_mismatch"] == {"4": [UUID_A]} and w2["commit_vs_program_mismatch"] == {"4": [UUID_A]}
    ra = next(p for p in rep["requests"] if p["request_id"] == UUID_A)
    assert ra["checks"]["R3_executed_set"]["declared_activity_not_committed"] == [4]
    assert ra["checks"]["R3c_capture_executed_set"]["matches"] == {"0": True, "1": False}
    assert ra["ok"] is False and rep["requests"][1]["ok"] is True       # B is untouched: the failure is A's and the workload's, by name


def test_a_declared_request_the_engine_and_the_commit_never_ran_is_named_at_every_step_of_its_window_and_the_population_is_the_manifests(desc, desc_b, tmp_path):
    """B is in the manifest (cap 4, arrive 2) but neither the engine log nor the Commit has any row/identity of it.  W1 (the population) is
    the manifest's -- Σ cap_r = 13 events, guard gates for both -- unchanged; W2 names B at 2..5 in both views; B's own R3c/R3 name an empty
    executed set; the report is False.  A fixed population is not the set of requests that showed up."""
    a_tr = Y_RAW[:A_R]
    rep = _run(desc, desc_b, tmp_path, lag=0, a_tr=a_tr, engine=[(UUID_A, a_tr, 0)], commit=(range(A_R), []))
    assert rep["ok"] is False
    w1 = rep["checks"]["W1_population"]
    assert w1["ok"] is True and w1["events"] == CAP + CAP_B and w1["requests"] == 2
    w2 = rep["checks"]["W2_chronology"]
    assert w2["engine_vs_program_mismatch"] == {str(t): [UUID_B] for t in range(2, 2 + CAP_B)}
    assert w2["commit_vs_program_mismatch"] == {str(t): [UUID_B] for t in range(2, 2 + CAP_B)}
    rb = next(p for p in rep["requests"] if p["request_id"] == UUID_B)
    assert rb["checks"]["R3c_capture_executed_set"]["engine_steps"] == [] and rb["checks"]["R3_executed_set"]["committed_steps_n"] == 0


def _with_spec_tokens(cap_dir, facts):
    """Write a spec_tokens.json beside the synthetic log: `facts` = [(uuid, prompt_len, max_tokens, arrive_step)] -- the record's OWN word
    about each engine uuid, what W0 (F-pad-24) checks the manifest's r<k> against."""
    import json
    (cap_dir / "spec_tokens.json").write_text(json.dumps({"schema": "spec-tokens/1", "requests": [
        {"request_id": u, "prompt_len": lp, "max_tokens": mt, "arrive_step": ar, "prompt_token_ids": [0] * lp} for u, lp, mt, ar in facts]}))
    return cap_dir


def _workload(desc, desc_b, cap_dir, reqs, bm=None, lag=0):
    lg, src = L.lag_from_header(_header(lag))
    bm = bm if bm is not None else _bmap2(range(A_R + lag), range(2, 2 + CAP_B))
    return R.check_workload_record(reqs, EOS, lag=lg, lag_source=src, bmap=bm, capture=R.read_capture(cap_dir), boundary_prefix=BOUNDARY)


def test_f_pad_25_w0_must_not_read_true_when_no_pairing_fact_source_exists(desc, desc_b, tmp_path):
    a_tr = Y_RAW[:A_R]
    cap_dir = _capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0))
    rep = _workload(desc, desc_b, cap_dir, _reqs(desc, desc_b, a_tr, B_TOKENS))
    w0 = rep["checks"]["W0_pairing"]
    assert w0["sources_evaluated"] == [] and w0["ok"] is None and w0["facts_verified"] is False
    assert rep["ok"] is None and rep["checks"]["W2_chronology"]["ok"] is True        # the structure holds; the pairing is unverified, by name


def _with_run_json(cap_dir, uuids_in_manifest_order):
    """match/run.json beside the capture dir: `realized.requests[k]` in manifest order -- the record's OWN k -> uuid statement."""
    import json
    (cap_dir.parent / "run.json").write_text(json.dumps({"realized": {"requests": [{"request_id": u, "n_generated": 0} for u in uuids_in_manifest_order]}}))
    return cap_dir


def test_w0_with_the_records_own_facts_the_right_pairing_passes_and_a_wrong_uuid_map_is_refused_by_name(desc, desc_b, tmp_path):
    """spec_tokens.json says uuid A = (LP, cap 9, arrive 0), uuid B = (LP, cap 4, arrive 2); run.json realized.requests = [A, B].
    Declared pairing r0<->A, r1<->B: W0 PASS, both sources named, fact_mismatch {}.  F-pad-24's attack proper -- the Build's r<k> -> uuid
    MAP wrong (r0's manifest entry carries uuid B, r1's carries A): the record's facts about B (cap 4, arrive 2) disagree with the
    manifest's r0 (cap 9, arrive 0) and about A with r1 -> refused BY NAME before any W-check runs, both uuids in the message."""
    a_tr = Y_RAW[:A_R]
    LP_ = _reqs(desc, desc_b, a_tr, B_TOKENS)[0]["prompt_len"]
    cap_dir = _with_run_json(_with_spec_tokens(_capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0)),
                                               [(UUID_A, LP_, CAP, 0), (UUID_B, LP_, CAP_B, 2)]), [UUID_A, UUID_B])
    rep = _workload(desc, desc_b, cap_dir, _reqs(desc, desc_b, a_tr, B_TOKENS))
    w0 = rep["checks"]["W0_pairing"]
    assert rep["ok"] is True and w0["ok"] is True and w0["fact_mismatch"] == {}
    assert w0["sources_evaluated"] == ["spec_tokens.json", "run.json realized.requests"]
    assert w0["pairs"] == {"r0": UUID_A, "r1": UUID_B} and w0["indistinguishable_by_pairing_facts"] == {}
    wrong_map = _reqs(desc, desc_b, a_tr, B_TOKENS)
    wrong_map[0]["request_id"], wrong_map[1]["request_id"] = UUID_B, UUID_A
    with pytest.raises(ValueError, match=r"W0 pairing refused by name.*facts about a uuid do not match") as ei:
        _workload(desc, desc_b, cap_dir, wrong_map)
    assert UUID_A in str(ei.value) and UUID_B in str(ei.value)


def test_w0_a_commit_id_swap_is_caught_by_run_json_order_and_without_run_json_only_by_the_windows(desc, desc_b, tmp_path):
    """The other crossing: the request entries keep their uuid+facts but the COMMIT ids are exchanged (r0's identities filed as r1's).
    spec_tokens facts travel with the uuid, so they cannot see it; run.json realized.requests[k] CAN (k = the commit id's position):
    refused by name.  Without run.json, W0 PASSES (a named limit of its sources) and the swap is caught structurally by W2 only because
    the two windows differ (commit_vs_program at the steps where exactly one is live) -- with equal windows nothing here would see it."""
    a_tr = Y_RAW[:A_R]
    LP_ = _reqs(desc, desc_b, a_tr, B_TOKENS)[0]["prompt_len"]
    def crossed():
        rr = _reqs(desc, desc_b, a_tr, B_TOKENS)
        rr[0]["commit_request_id"], rr[1]["commit_request_id"] = "r1", "r0"
        return rr
    with_run = _with_run_json(_with_spec_tokens(_capture2(tmp_path / "run", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0)),
                                                [(UUID_A, LP_, CAP, 0), (UUID_B, LP_, CAP_B, 2)]), [UUID_A, UUID_B])
    with pytest.raises(ValueError, match=r"W0 pairing refused by name.*facts about a uuid do not match"):
        _workload(desc, desc_b, with_run, crossed())
    no_run = _with_spec_tokens(_capture2(tmp_path / "norun", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0)),
                               [(UUID_A, LP_, CAP, 0), (UUID_B, LP_, CAP_B, 2)])
    rep = _workload(desc, desc_b, no_run, crossed())
    assert rep["checks"]["W0_pairing"]["ok"] is True and rep["checks"]["W0_pairing"]["sources_evaluated"] == ["spec_tokens.json"]
    assert rep["ok"] is False and set(rep["checks"]["W2_chronology"]["commit_vs_program_mismatch"]) == {"0", "1", "4", "5"}


def test_w0_an_undeclared_uuid_that_ran_and_a_duplicated_engine_id_are_refused_by_name(desc, desc_b, tmp_path):
    """(i) the engine log carries rows of a THIRD uuid the manifest never declared -> 'engine ran undeclared request ids'; (ii) two declared
    requests given the same engine uuid -> 'engine ids not distinct'; both ValueError by name, no W-check runs on a broken pairing."""
    a_tr = Y_RAW[:A_R]
    ghost = "cccccccc-0000-5000-8000-000000000003"
    cap_dir = _capture2(tmp_path / "ghost", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2), (ghost, B_TOKENS, 3)], lag=0, header=_header(0))
    with pytest.raises(ValueError, match=r"W0 pairing refused by name.*undeclared request ids") as ei:
        _workload(desc, desc_b, cap_dir, _reqs(desc, desc_b, a_tr, B_TOKENS))
    assert ghost in str(ei.value)
    cap_dir2 = _capture2(tmp_path / "dup", [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0))
    dup = _reqs(desc, desc_b, a_tr, B_TOKENS)
    dup[1]["request_id"] = UUID_A
    with pytest.raises(ValueError, match=r"W0 pairing refused by name.*engine ids not distinct"):
        _workload(desc, desc_b, cap_dir2, dup)


def test_w0_two_requests_of_equal_shape_are_named_indistinguishable_and_their_crossed_pairing_is_not_seen_here(desc, tmp_path):
    """Two requests with the SAME (prompt_len, cap, arrive) and the same tokens (both stop at A's EOS): W0 PASSES with the pair listed under
    `indistinguishable_by_pairing_facts` -- and the CROSSED pairing passes identically (facts equal, sets equal).  The limit is stated in the
    report; a reader must take `indistinguishable_by_pairing_facts != {}` as 'the pairing is unverified for these requests; R6's job'."""
    a_tr = Y_RAW[:A_R]
    LP_ = _reqs(desc, desc, a_tr, a_tr)[0]["prompt_len"]
    twin = "dddddddd-0000-5000-8000-000000000004"
    cap_dir = _with_spec_tokens(_capture2(tmp_path, [(UUID_A, a_tr, 0), (twin, a_tr, 0)], lag=0, header=_header(0)),
                                [(UUID_A, LP_, CAP, 0), (twin, LP_, CAP, 0)])
    def reqs(cross):
        rr = [dict(r) for r in _reqs(desc, desc, a_tr, a_tr)]
        rr[1].update(request_id=twin, cap=CAP, arrive=0, desc=desc)
        if cross:
            rr[0]["commit_request_id"], rr[1]["commit_request_id"] = "r1", "r0"
        return rr
    bm = _bmap2(range(A_R), range(A_R))
    straight = _workload(desc, desc, cap_dir, reqs(False), bm=bm)
    crossed = _workload(desc, desc, cap_dir, reqs(True), bm=bm)
    for rep in (straight, crossed):
        w0 = rep["checks"]["W0_pairing"]
        assert rep["ok"] is True and w0["ok"] is True and w0["fact_mismatch"] == {}
        assert [sorted(v) for v in w0["indistinguishable_by_pairing_facts"].values()] == [["r0", "r1"]] and "R6" in w0["limit"]
    assert straight["checks"]["W0_pairing"]["pairs"] != crossed["checks"]["W0_pairing"]["pairs"]


def test_the_w_checks_are_structural_two_co_live_requests_identities_exchanged_in_the_commit_pass_w2_and_that_is_r6s_job(desc, desc_b, tmp_path):
    """Statement of scope, measured: at step 3 both A and B are live; a Commit that files A's identities under B's id and B's under A's
    at that step commits the SAME set of request ids there -- W2 (sets per step) and each request's R3 (its own steps) PASS, `ok` True.
    Nothing here compares VALUES: the swap is visible only to R6 / C2 (Active strata bit-exact against the committed store), which this
    tool does not evaluate.  A reader must not take a W2 PASS as 'the right values were committed for the right request'."""
    a_tr = Y_RAW[:A_R]
    lg, src = L.lag_from_header(_header(0))
    cd = R.read_capture(_cap2_with_facts(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0)))
    bm = _bmap2(range(A_R), range(2, 2 + CAP_B))
    for e in bm["entries"]:
        ident = e["identity"]
        if int(ident["step"]) == 3:
            ident["request_id"] = {"r0": "r1", "r1": "r0"}[ident["request_id"]]
    assert R.executed_steps(bm, "r0") == set(range(A_R)) and R.executed_steps(bm, "r1") == set(range(2, 2 + CAP_B))
    rep = R.check_workload_record(_reqs(desc, desc_b, a_tr, B_TOKENS), EOS, lag=lg, lag_source=src, bmap=bm, capture=cd, boundary_prefix=BOUNDARY)
    assert rep["ok"] is True and rep["checks"]["W2_chronology"]["commit_vs_program_mismatch"] == {}
    assert rep["requests"][0]["checks"]["R6_values"]["ok"] is None and rep["requests"][1]["checks"]["R6_values"]["ok"] is None


def _fake_row(tmp_path, desc, desc_b, wp_extra=None, drop_derived=False):
    """pad's CLI scaffold (test_cli_build_workload_...): a row dir with the two component dirs, a workload_program.json and a capture."""
    import gzip as _gz
    import json
    from verity_vllm.registry.b1 import weights_type
    from tests.ir.test_lifted_record import LP, TINY
    from verity.ir.types import Array, Value
    row = tmp_path / "row"
    dir_a, dir_b = f"build_request_LP{LP}_T{CAP - 1}", f"build_request_LP{LP}_T{CAP_B - 1}"   # the derive naming convention, on purpose
    for name, d in ((dir_a, desc), (dir_b, desc_b)):
        (row / name).mkdir(parents=True)
        with _gz.open(row / name / "descriptor.json.gz", "wt") as f:
            json.dump(d, f)
        with _gz.open(row / name / "instances.json.gz", "wt") as f:
            json.dump({"params": [["prompt", repr(Array(LP, Value(32)))], ["weights", repr(weights_type(TINY))]]}, f)
    (row / "build_workload").mkdir()
    wp = {"workload_digest": "0" * 64, "gates": 1, "manifest_digest": "md", "stop_rule": {"eos_token_ids": list(EOS)},
          "requests": [{"request_id": "r0", "index": 0, "prompt_len": LP, "cap": CAP, "arrive_step": 0},
                       {"request_id": "r1", "index": 1, "prompt_len": LP, "cap": CAP_B, "arrive_step": 2}],
          "request_ids": {"workload_request_ids": {"r0": UUID_A, "r1": UUID_B}},
          "inputs": {"derived": {f"{LP},{CAP - 1}": f"/pod/path/row/{dir_a}", f"{LP},{CAP_B - 1}": f"/pod/path/row/{dir_b}"}}}
    if drop_derived:
        wp["inputs"] = {}
    wp.update(wp_extra or {})
    (row / "build_workload" / "workload_program.json").write_text(json.dumps(wp))
    a_tr = Y_RAW[:A_R]
    cap_dir = _capture2(tmp_path, [(UUID_A, a_tr, 0), (UUID_B, B_TOKENS, 2)], lag=0, header=_header(0))
    commit = tmp_path / "commit"
    commit.mkdir()
    with _gz.open(commit / "binding_map_p0.json.gz", "wt") as f:
        json.dump(_bmap2(range(A_R), range(2, 2 + CAP_B)), f)
    return row, cap_dir, commit


def test_f_pad_26_the_recomposed_workload_must_be_pinned_to_the_rows_recorded_workload_digest(desc, desc_b, tmp_path):
    import json
    row, cap_dir, commit = _fake_row(tmp_path, desc, desc_b)          # workload_digest = 64 zeros: not any Program's
    out = tmp_path / "rep.json"
    rc = R.main(["--build-workload", str(row), "--capture", str(cap_dir), "--commit", str(commit), "--boundary-prefix", BOUNDARY, "--out", str(out)])
    rep = json.loads(out.read_text())
    w1 = rep["checks"]["W1_population"]
    assert rc != 0 and rep["ok"] is False and w1["ok"] is False and w1["pinned_to_row"] is True
    assert w1["recorded_workload_digest"] == "0" * 64 and w1["recomposed_workload_digest"] != w1["recorded_workload_digest"] and "workload_digest" in w1["why"]


def test_f_pad_26b_a_component_dir_not_declared_by_the_builds_inputs_derived_is_refused_not_guessed(desc, desc_b, tmp_path):
    import json
    row, cap_dir, commit = _fake_row(tmp_path, desc, desc_b, drop_derived=True)
    out = tmp_path / "rep.json"
    with pytest.raises(SystemExit):
        R.main(["--build-workload", str(row), "--capture", str(cap_dir), "--commit", str(commit), "--boundary-prefix", BOUNDARY, "--out", str(out)])
