"""`verity_vllm.check.match.batch_decomp.decompose`, the paths of it no other CPU test runs: the attribution channels after the meta
channel (weight-only, the row-count rule, each way an instance is left unattributed), the errors a request's comparison reports by name,
the three within-step criteria, the operations' per-request hook, a derived dir with no shape, and the profile's sidecar
(`X09_PROFILE_OUT`, `X09_FORCE_DAG`).

The attribution tests read a stub of the fold's view (no Program is decoded); the others the toy of `test_batch_decomp_e2e.py`."""
from __future__ import annotations

import collections
import dataclasses
import json
import types

import pytest

from tests.check import test_batch_decomp_e2e as e2e
from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import decomp_profile
from verity_vllm.check.match import program_compare as C

memory = e2e.memory                          # the toy's fixtures, under their own names
derived_is_a_fold = e2e.derived_is_a_fold

R0, R1 = "r0", "r1"
HEADER = {"workload": {"requests": [{"request_id": R0, "prompt_token_ids": [1, 2, 3]}, {"request_id": R1, "prompt_token_ids": [5, 6]}]}}
# one batched prefill: five token rows, two logits rows
STEP0 = {"rows": [(R0, 0), (R0, 1), (R0, 2), (R1, 0), (R1, 1)], "logits_rows": [(R0, 2), (R1, 1)], "logits_indices": [2, 4]}
W, P0, P1 = [("n", 2, 0, 10)], [("n", 0, 0, 1)], [("n", 1, 0, 1)]   # operands: the weights, r0's prompt, r1's prompt
EMB, SELECT = "Embedding_v1{V=8,H=2}", "TokenSelect_v1{V=8}"


def _fold(inst, rows):
    """What `decompose` reads of a fold's view before any request is compared.  Input nodes 0 and 1 = the two prompts, 2 = the
    weights.  `inst`: (node, spec, operands); `rows`: (node, engine step, row, rows, request) of each instance the record has a
    row for."""
    return types.SimpleNamespace(
        inputs={0: ("Input", 3), 1: ("Input", 2), 2: ("Input", 10)}, n_input_nodes=3, digest="d",
        oracle_wmap=[(0, 10, "w")], oracle_wmap_overlaps=[], _wlos=[0], nodes=[],
        inst=[{"k": k, "spec": spec, "form": "call", "args": args, "out": "bf16"} for k, spec, args in inst],
        oracle_rows=[{"node": k - 3, "step": step, "row": row, "rows": n, "request": req} for k, step, row, n, req in rows])


def _decompose(monkeypatch, tmp_path, B, steps, derived=(), **kw):
    monkeypatch.setattr(bd, "read_steps", lambda p: (HEADER, steps))
    return bd.decompose(str(tmp_path), list(derived), log_path="ignored", oracle=B, **kw)


def _two_requests(meta=(None, None)):
    """Each request: Embedding(prompt, weights) -> TokenSelect; `meta` = the request the record states for r0's and r1's instances."""
    return _fold([(3, EMB, [P0, W]), (4, EMB, [P1, W]), (5, SELECT, [[("n", 3, 0, 2)]]), (6, SELECT, [[("n", 4, 0, 2)]])],
                 [(3, 0, 0, 1, meta[0]), (4, 0, 0, 1, meta[1]), (5, 0, 0, 1, meta[0]), (6, 0, 0, 1, meta[1])])


def _inline_hook(calls):
    """The record's operations with the per-request hook, the operations' last field, set: it runs every item in this process
    and counts its calls."""
    def run_all(items, fn, weight):
        calls.append(len(items))
        return [fn(it) for it in items]
    rec = bd.record_ops()
    return type(rec)(rec.name, rec.load_oracle, rec.load_derived, rec.projection, rec.compare_steps, rec.dag_hashes, rec.live_cone, run_all)


def _how_the_requests_ran(prof: dict) -> str:
    """The profile's one field that says how the requests were run: `inline`, or `forked` through the hook.  Its key, like the
    name of the run's third phase, holds a word GLOSSARY.md retires, so these tests read the one by its value and the other by
    its position."""
    (how,) = [v for v in prof.values() if v in ("inline", "forked")]
    return how


def _same_but_time(a: dict, b: dict) -> bool:
    drop = ("utc", "seconds")
    return {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}


# ------------------------------------------------------------------------------------------------------------------ attribution


def test_a_criterion_that_is_neither_sequence_nor_dag_is_refused(tmp_path):
    with pytest.raises(ValueError, match="within_step must be one of .*got 'multiset'"):
        bd.decompose(str(tmp_path), [], within_step="multiset")


def test_a_view_with_no_recorded_rows_is_refused_as_the_fold(monkeypatch, tmp_path):
    """A derived Program's view has none: passed as the fold it is refused by name, before any instance is attributed."""
    B = _two_requests()
    B.oracle_rows = None
    with pytest.raises(ValueError, match="the view passed as the fold has no instances.jsonl rows"):
        _decompose(monkeypatch, tmp_path, B, {0: STEP0})


def test_weight_only_instances_are_shared_and_are_compared_with_every_request_with_rows_in_their_step(monkeypatch, tmp_path):
    """Nodes 3 (reads the weights alone) and 4 (reads node 3 alone) carry no request's data, nor does node 11 at engine step 1, where
    only r0 has a row: owned by no request, counted apart, and merged into each request's instances by engine step."""
    B = _fold([(3, "AddScalar_v1{N=2}", [W]), (4, "Scale_v1{N=2}", [[("n", 3, 0, 2)]]), (5, EMB, [P0, W]), (6, EMB, [P1, W]),
               (7, "Norm_v1{N=2}", [[("n", 5, 0, 2)], [("n", 4, 0, 2)]]), (8, "Norm_v1{N=2}", [[("n", 6, 0, 2)], [("n", 4, 0, 2)]]),
               (9, SELECT, [[("n", 7, 0, 2)]]), (10, SELECT, [[("n", 8, 0, 2)]]), (11, "AddScalar_v1{N=2}", [W])],
              [(k, 0, 0, 1, None) for k in range(3, 11)] + [(11, 1, 0, 1, None)])
    steps = {0: STEP0, 1: {"rows": [(R0, 3)], "logits_rows": [(R0, 3)], "logits_indices": [0]}}
    res = _decompose(monkeypatch, tmp_path, B, steps)
    att = res["attribution"]
    assert att["how"] == {"weight-only": 3, "dataflow": 6} and att["unattributed"] == 0
    assert att["weight_only_shared"] == 3 and att["weight_only_specs"] == {"AddScalar_v1": 2, "Scale_v1": 1}
    assert att["attributed"] == 6                                   # the requests' own instances; a shared one is nobody's
    assert [(e["request_id"], e["n_instances"]) for e in res["requests"]] == [(R0, 6), (R1, 5)]   # 3 own + 3 shared | 3 own + step 0's 2
    assert res["verdict"] == "MISSING_DERIVED" and res["reasons"] == ["missing derived Programs for [[2, 0], [3, 0]]"]


def test_each_way_an_instance_is_left_unattributed_is_named_and_the_row_rule_places_the_rest(monkeypatch, tmp_path):
    """Node 5 has no row in the record; node 6 reads it; node 7's engine step has no StepStart; node 9's row count is neither the token
    rows nor the logits rows.  Node 8 reads no node (a parameter-space run) and the row-count rule places it: row 3 of 5 token rows
    is r1's.  Node 10 is r0's by data flow where the row-count rule says r1 (logits row 1): counted, and data flow stands."""
    B = _fold([(3, EMB, [P0, W]), (4, EMB, [P1, W]), (5, "Lost_v1", [[("n", 3, 0, 2)]]), (6, "Mix_v1", [[("n", 5, 0, 2)]]),
               (7, "Const_v1", []), (8, "Table_v1", [[("p", 0, 0, 1)]]), (9, "Const_v1", []),
               (10, SELECT, [[("n", 3, 0, 2)]]), (11, SELECT, [[("n", 4, 0, 2)]])],
              [(3, 0, 0, 1, None), (4, 0, 0, 1, None), (6, 0, 0, 1, None), (7, 9, 0, 1, None), (8, 0, 3, 5, None), (9, 0, 0, 7, None),
               (10, 0, 1, 2, None), (11, 0, 1, 2, None)])
    res = _decompose(monkeypatch, tmp_path, B, {0: STEP0})
    att = res["attribution"]
    assert att["how"] == {"dataflow": 4, "reads": 1, "no": 1, "rows": 2}
    assert att["unattributed"] == 4 and [(u["k"], u["why"]) for u in att["unattributed_examples"]] == [
        (5, "no instances.jsonl row"), (6, "reads unattributed node 5"), (7, "no StepStart for step 9"),
        (9, "rows:rows=7!=tokens(5)|logits(2)")]
    assert att["row_rule_disagreements"] == 1
    assert [(e["request_id"], e["n_instances"]) for e in res["requests"]] == [(R0, 2), (R1, 3)]   # r1's: nodes 4, 8, 11
    assert res["verdict"] == "FAIL" and res["reasons"] == ["4 unattributed instances", "missing derived Programs for [[2, 0], [3, 0]]"]


# ----------------------------------------------------------------------------------- what a request's comparison reports by name


@pytest.mark.parametrize("hook", [False, True], ids=["record", "hook"])
def test_a_request_the_header_does_not_have_fails_by_name(monkeypatch, tmp_path, hook):
    """The record states `ghost` for r1's instances.  With the hook the driver first asks every request's shape (to load its
    component before the requests are compared): `ghost` has none, and the entry is the same."""
    calls: list[int] = []
    res = _decompose(monkeypatch, tmp_path, _two_requests(meta=(R0, "ghost")), {0: STEP0}, ops=_inline_hook(calls) if hook else None)
    ghost = next(e for e in res["requests"] if e["request_id"] == "ghost")
    assert ghost == {"request_id": "ghost", "parent": None, "prompt_param": None, "n_instances": 2,
                     "error": "request not in RunHeader.workload (no prompt parameter)"}
    assert res["attribution"]["how"] == {"meta": 4} and res["missing_derived"] == [[3, 0]]
    assert res["verdict"] == "MISSING_DERIVED" and res["reasons"][-1] == "ghost: request not in RunHeader.workload (no prompt parameter)"
    assert calls == ([2] if hook else [])                            # the two requests in one call; no component to load


def test_a_request_with_no_sampling_event_fails_by_name(monkeypatch, tmp_path):
    B = _fold([(3, EMB, [P0, W]), (4, EMB, [P1, W]), (5, SELECT, [[("n", 3, 0, 2)]])], [(k, 0, 0, 1, None) for k in (3, 4, 5)])
    res = _decompose(monkeypatch, tmp_path, B, {0: STEP0})
    r1 = next(e for e in res["requests"] if e["request_id"] == R1)
    assert r1["error"] == "no TokenSelect attributed to the request" and r1["token_selects"] == 0 and "derived" not in r1
    assert res["reasons"] == ["missing derived Programs for [[3, 0]]", "r1: no TokenSelect attributed to the request"]


# ------------------------------------------------------------------------------------------------------ the within-step criteria


class _Component:
    inst, steps, digest = [{"spec": "X_v1"}], [[0]], "component"


def _criteria_ops(seq_equal: bool, same_dag: bool):
    """The record's operations with a stated comparison: the canonical sequences equal or not, the live-cone DAG multisets equal or
    not.  The Projection is the record's own."""
    cmp = {"canonical_sha256": {"equal": seq_equal}, "totals": {}, "a": {"steps": 1}, "b": {"steps": 1}, "steps": []}
    return dataclasses.replace(
        bd.record_ops(), load_derived=lambda d, label: _Component(), compare_steps=lambda A, P, unfold_a: cmp,
        live_cone=lambda Q: {0}, dag_hashes=lambda Q, live: {0: "h" if same_dag or isinstance(Q, _Component) else "other"})


@pytest.mark.parametrize("chunked, within_step, same_dag, equal, criterion", [
    (True, "sequence", True, True, "dag-multiset over the live cone (chunked prefill)"),
    (True, "dag", True, True, "dag-multiset over the live cone (chunked prefill)"),
    (False, "dag", True, True, "dag-multiset over the live cone (within-step criterion of record; sequence differs)"),
    (False, "sequence", True, False, "sequence"),          # the DAG form is then a diagnostic alone
    (True, "dag", False, False, "sequence"),
])
def test_a_sequence_that_differs_is_accepted_by_its_dag_only_under_the_criterion_or_a_chunked_prefill(
        monkeypatch, tmp_path, chunked, within_step, same_dag, equal, criterion):
    """One request whose canonical sequence differs from its component's.  Chunked: its three prompt rows arrive in two engine steps,
    with one sample each."""
    one = {"workload": {"requests": HEADER["workload"]["requests"][:1]}}
    if chunked:
        steps = {0: {"rows": [(R0, 0), (R0, 1)], "logits_rows": [(R0, 1)]}, 1: {"rows": [(R0, 2)], "logits_rows": [(R0, 2)]}}
        B = _fold([(2, EMB, [P0, [("n", 1, 0, 10)]]), (3, SELECT, [[("n", 2, 0, 2)]]), (4, SELECT, [[("n", 2, 0, 2)]])],
                  [(2, 0, 0, 1, None), (3, 0, 0, 1, None), (4, 1, 0, 1, None)])
    else:
        steps = {0: {"rows": [(R0, 0), (R0, 1), (R0, 2)], "logits_rows": [(R0, 2)]}}
        B = _fold([(2, EMB, [P0, [("n", 1, 0, 10)]]), (3, SELECT, [[("n", 2, 0, 2)]])], [(2, 0, 0, 1, None), (3, 0, 0, 1, None)])
    B.inputs, B.n_input_nodes = {0: ("Input", 3), 1: ("Input", 10)}, 2
    for o in B.oracle_rows:
        o["node"] += 1                                               # two input nodes here, not three
    d = tmp_path / "LP3_T0"
    d.mkdir()
    (d / "construction_manifest.json").write_text(json.dumps({"structural_inputs": {"LP": 3, "T": 0}}))
    monkeypatch.setattr(bd, "read_steps", lambda p: (one, steps))
    res = bd.decompose(str(tmp_path), [str(d)], log_path="ignored", oracle=B, within_step=within_step, ops=_criteria_ops(False, same_dag))
    (ent,) = res["requests"]
    assert (ent["LP"], ent["T"], ent["prefill_chunks"]) == (3, 0, 2 if chunked else 1)
    assert ent["sequence_equal"] is False and ent["dag_multiset_equal"] is same_dag
    assert (ent["canonical_equal"], ent["criterion"]) == (equal, criterion)
    assert res["verdict"] == ("PASS" if equal else "FAIL") and res["reasons"] == ([] if equal else ["r0: not canonical-equal"])
    assert res["within_step_criterion"] == within_step


# ------------------------------------------------------------------------------------------------------------ on the toy's folds


def test_a_derived_dir_with_no_shape_is_named_and_fails_a_run_whose_requests_all_compare_equal(memory, tmp_path, derived_is_a_fold):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    a0, a1 = e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")
    (tmp_path / "nope").mkdir()
    res = bd.decompose(d, [a0, str(tmp_path / "nope"), a1])
    assert res["derived_unreadable"] == [str(tmp_path / "nope")] and sorted(res["derived_available"].values()) == sorted([a0, a1])
    assert [e["canonical_equal"] for e in res["requests"]] == [True, True]
    assert res["verdict"] == "FAIL" and res["reasons"] == ["1 derived dir(s) unreadable: ['nope']"]


def test_the_hook_gets_the_components_to_load_and_then_the_requests_and_the_document_is_the_same(memory, tmp_path, derived_is_a_fold):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    derived = [e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    calls: list[int] = []
    components: dict = {}
    res = bd.decompose(d, derived, ops=_inline_hook(calls), components=components)
    assert calls == [2, 2] and sorted(components) == sorted(derived)     # the two components, then the two requests
    assert res["verdict"] == "PASS" and _same_but_time(res, bd.decompose(d, derived))


@pytest.fixture
def profile_out(monkeypatch, tmp_path):
    """`X09_PROFILE_OUT` set to a file in a directory that does not exist yet; the counting wrappers the profile puts on `RootView`
    are taken off again after the test."""
    for name in ("_per_run", "_srcP", "_src"):
        monkeypatch.setattr(C.RootView, name, getattr(C.RootView, name))
    monkeypatch.setattr(C.RootView, "_x09_prof_installed", False, raising=False)
    monkeypatch.setattr(decomp_profile, "_profile_counters", collections.Counter())
    out = tmp_path / "profile" / "x09.json"
    monkeypatch.setenv(decomp_profile.PROFILE_ENV, str(out))
    return out


def test_the_profile_is_a_sidecar_and_the_document_is_the_one_written_without_it(memory, tmp_path, derived_is_a_fold, profile_out,
                                                                                  monkeypatch):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    derived = [e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    res = bd.decompose(d, derived)
    prof = json.loads(profile_out.read_text())
    assert res["schema"] == "cb-a/batch-decomp/v1"
    assert prof["schema"] == "cb-a/x09-profile/v1" and prof["oracle_dir"] == d and prof["force_dag"] is False
    assert (prof["verdict"], prof["seconds"], _how_the_requests_ran(prof)) == ("PASS", res["seconds"], "inline")
    assert (prof["fold_preloaded"], prof["fold_instances"], prof["fold_compact"]) == (False, res["oracle"]["instances"], False)
    run_phases = list(prof["phases"].items())
    assert [name for name, _ in run_phases[:2]] == ["fold_load", "attribution"] and len(run_phases) == 3
    assert prof["max_rss_mib_parent"] > 0 and "max_rss_error" not in prof
    assert [(p["request_id"], p["LP"], p["T"], p["sequence_equal"], p["xreq_total"]) for p in prof["requests"]] == \
        [("r0", 3, 2, True, 0), ("r1", 2, 1, True, 0)]
    for p, e in zip(prof["requests"], res["requests"]):
        assert list(p["phases"]) == ["projection_build", "shape", "component_load", "compare_steps"]   # no DAG phase: the sequences are equal
        assert (p["n_instances"], p["steps"], p["a_steps"]) == (e["n_instances"], e["steps"], e["a_steps"])
        assert p["p_runs_total"] > 0 and p["a_runs_total"] == p["p_runs_total"] and p["seconds"] >= 0
    assert run_phases[2][1]["counters"]["src_calls"] > 0            # the wrappers on `RootView` counted the requests' canonical reads
    assert all("_profile" not in e for e in res["requests"])
    # with the fold passed in, the profile says so; without the variable the document is the same and nothing is written
    bd.decompose(d, derived, oracle=C.load_oracle(d, "oracle"))
    assert json.loads(profile_out.read_text())["fold_preloaded"] is True
    profile_out.unlink()
    monkeypatch.delenv(decomp_profile.PROFILE_ENV)
    assert _same_but_time(res, bd.decompose(d, derived)) and not profile_out.exists()


def test_force_dag_adds_the_dag_diagnostic_to_a_request_whose_sequence_is_equal_and_changes_no_verdict(
        memory, tmp_path, derived_is_a_fold, profile_out, monkeypatch):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    derived = [e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    plain = bd.decompose(d, derived)
    monkeypatch.setenv(decomp_profile.FORCE_DAG_ENV, "1")
    res = bd.decompose(d, derived)
    assert res["verdict"] == plain["verdict"] == "PASS"
    for e, q in zip(res["requests"], plain["requests"]):
        assert "dag_multiset" not in q and e["dag_multiset_equal"] is True and e["dag_multiset"]["only_a"] == 0
        assert {k: v for k, v in e.items() if not k.startswith("dag_multiset")} == q   # criterion "sequence", canonical_equal as before
    prof = json.loads(profile_out.read_text())
    assert prof["force_dag"] is True
    for p, e in zip(prof["requests"], res["requests"]):
        assert list(p["phases"])[4:] == ["live_cone_A", "live_cone_P", "dag_hashes_A", "dag_hashes_P", "multiset_diff"]
        assert (p["a_live"], p["b_live"], p["dag_only_a"], p["dag_only_b"]) == (e["dag_multiset"]["a_live"], e["dag_multiset"]["b_live"], 0, 0)


def test_the_profile_is_the_same_document_through_the_hook(memory, tmp_path, derived_is_a_fold, profile_out):
    d, _ = e2e.fold_batched([e2e.R0, e2e.R1], tmp_path / "b2")
    derived = [e2e.fold_alone(e2e.R0, tmp_path / "LP3_T2"), e2e.fold_alone(e2e.R1, tmp_path / "LP2_T1")]
    res = bd.decompose(d, derived, ops=_inline_hook([]))
    prof = json.loads(profile_out.read_text())
    assert _how_the_requests_ran(prof) == "forked" and [p["request_id"] for p in prof["requests"]] == ["r0", "r1"]
    assert res["verdict"] == "PASS" and all("_profile" not in e for e in res["requests"])
