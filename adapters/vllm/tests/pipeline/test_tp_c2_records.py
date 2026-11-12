"""`verity_vllm.pipeline.tp.c2` on a record that holds less than a finished TP Commit's: a run with no commit or no rank, a rank
whose finalize recorded no replay, no linkage, no weights pin, no attribution or no match oracle, a replay record without its
blocks (a replay that did not run), a pin that lists nothing, a cross-rank pass that did not run, and a Commit with no Match record,
no pin settings or no required manifest.  Each reads as what its absence means and none reads as a pass.  Holds the first tests of
`linkage_component` and `oracle_compare_component`."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline.tp import c2 as C2
from verity_vllm.program.frontend.collective_kinds import cross_rank_families

FAMILIES = cross_rank_families(2)
ALL_REDUCE = "AllReduce2_v1"
MATCH = {"match_record": {"path": "/match", "tp2_match_verdict": True}}


def _run(per_rank: dict, pair: int = 0, **commit) -> dict:
    return {"arm": "instrumented", "pair": pair, "commit": {"per_rank": per_rank, **commit}}


def _gates(f: dict, *, required: bool = True, ok: bool = True, why: str | None = None) -> tuple[dict, bool | None, str | None]:
    cblock: dict = {}
    ok_, why_ = C2.c2_and_pin_gates(cblock, f, ok, why, rank=1, replay_on=True, required=required, bundle="/bundle", xrank_families=None)
    return cblock, ok_, why_


# a run's ranks

@pytest.mark.parametrize("row", [{"arm": "control"}, {"arm": "instrumented"}, {"arm": "instrumented", "commit": None},
                                 {"arm": "instrumented", "commit": {}}, {"arm": "instrumented", "commit": {"per_rank": None}},
                                 {"arm": "instrumented", "commit": {"per_rank": {}}}],
                         ids=["a control run", "no commit", "a null commit", "an empty commit", "null ranks", "no ranks"])
def test_a_run_that_finalized_no_rank_has_no_commit_blocks(row):
    assert C2.per_rank_of(row) == {}


def test_the_last_instrumented_run_that_finalized_a_rank_is_the_one_the_components_read():
    first, last = {"0": {"run_root": "aa"}}, {"0": {"run_root": "bb"}}
    rows = [{"arm": "control", "pair": 0}, _run(first), _run(last, pair=1), {"arm": "instrumented", "pair": 2}, {"arm": "control", "pair": 3, "commit": {"per_rank": first}}]
    assert C2.per_rank_of(rows[1]) is first
    assert C2.last_per_rank(rows) is last
    assert C2.last_per_rank([rows[0], rows[3]]) == {}


# a rank's gates

@pytest.mark.parametrize("f", [{}, {"sampled_replay": None, "weights_pin": None}, {"sampled_replay": {}, "weights_pin": {}},
                               {"sampled_replay": {"sampled_replay": None, "boundary_linkage": None, "wall_s": 0.1}}],
                         ids=["an empty finalize", "null blocks", "empty blocks", "a replay block of two null records"])
def test_a_finalize_with_no_replay_and_no_pin_fails_by_both_names_and_records_neither(f):
    cblock, ok, why = _gates(f)
    assert ok is False
    assert why is not None and why.startswith("weights unpinned (MAN-09, rank 1): no weights of record; also: sampled replay (C2 of record, rank 1): not run -- no result")
    assert cblock["sampled_replay"] is None and cblock["boundary_linkage"] is None and cblock["weights_pin"] is None


def test_a_replay_block_that_holds_the_replay_alone_fails_on_the_linkage_that_did_not_run():
    cblock, ok, why = _gates({"sampled_replay": {"sampled_replay": {"result": True, "grade": "complete"}}}, required=False)
    assert (ok, why) == (False, "boundary linkage (rank 1): not run")
    assert cblock["sampled_replay"] == {"result": True, "grade": "complete", "strata_n": 0} and cblock["boundary_linkage"] is None


def test_a_replay_record_counts_its_strata_and_leaves_them_and_its_trace_off_the_commit_block():
    replay = {"result": True, "grade": "complete", "strata": {"a": {}, "b": {}, "c": {}}, "trace": ["t"]}
    cblock, _, _ = _gates({"sampled_replay": {"sampled_replay": replay, "boundary_linkage": {"result": True}}}, required=False)
    assert cblock["sampled_replay"] == {"result": True, "grade": "complete", "strata_n": 3}


@pytest.mark.parametrize("record", [None, {}], ids=["a null record", "an empty record"])
def test_a_gate_handed_no_record_records_none_and_decides_nothing_where_nothing_requires_one(record):
    """`commit_replay.fold` hands `replay_gates` a replay's records as its result file has them, so a gate reads a null record as it
    reads an empty one: with the replay off and no manifest required, neither gate decides anything on it."""
    cblock: dict = {}
    assert C2.replay_gates(cblock, record, record, True, None, rank=1, replay_on=False, required=False, xrank_families=None) == (True, None)
    assert C2.pin_gate(cblock, record, True, None, rank=1, required=False) == (True, None)
    assert cblock == {"sampled_replay": None, "boundary_linkage": None, "weights_pin": None}


@pytest.mark.parametrize("f", [{}, {"weights_pin": None}], ids=["no pin block", "a null pin"])
def test_a_deferred_rank_with_no_pin_is_decided_unpinned_when_a_manifest_is_required_and_pending_when_none_is(f):
    f = dict(f, sampled_replay={"sampled_replay": {"grade": "deferred"}, "boundary_linkage": None})
    cblock, ok, why = _gates(f)
    assert (ok, why) == (False, "weights unpinned (MAN-09, rank 1): no weights of record")
    assert cblock["boundary_linkage"] == {} and cblock["replay_deferred"] == {"pre": [True, None], "bundle": "/bundle"}
    assert _gates(f, required=False)[1:] == (None, RB.PENDING)


@pytest.mark.parametrize("pin", [{"result": True}, {"result": True, "mismatched": None}, {"result": True, "mismatched": []}],
                         ids=["no list", "a null list", "an empty list"])
def test_a_pin_that_lists_no_differing_weight_records_none(pin):
    cblock: dict = {}
    assert C2.pin_gate(cblock, pin, True, None, rank=0, required=True) == (True, None)
    assert cblock["weights_pin"] == {"result": True, "mismatched": [], "mismatched_n": 0}


def test_a_pin_that_differs_names_its_first_weight_and_records_the_first_eight_with_their_count():
    differing = [{"served_name": f"w{i}"} for i in range(10)]
    cblock: dict = {}
    ok, why = C2.pin_gate(cblock, {"result": False, "why": "10 differ", "mismatched": differing}, False, "an earlier gate", rank=0, required=True)
    assert (ok, why) == (False, "weights of record (rank 0): 10 differ (first: w0); also: an earlier gate")
    assert cblock["weights_pin"] == {"result": False, "why": "10 differ", "mismatched": differing[:8], "mismatched_n": 10}


def test_a_pin_that_fails_with_no_differing_weight_names_its_first_missing_one():
    pin = {"result": False, "why": "1 missing", "missing": ["down_proj"]}
    assert C2.pin_gate({}, pin, True, None, rank=0, required=True) == (False, "weights of record (rank 0): 1 missing (first: down_proj)")
    assert C2.pin_gate({}, {"result": False, "why": "nothing compared"}, True, None, rank=0, required=True) == (False, "weights of record (rank 0): nothing compared (first: None)")
    assert C2.pin_gate({}, pin, True, None, rank=0, required=False) == (True, None)


# the families with no evaluator

def test_a_replay_record_that_lists_no_unreplayable_family_has_no_gap():
    for replay in ({"result": True}, {"result": True, "unreplayable": None}, {"result": True, "unreplayable": []}):
        block, fail = C2.no_evaluator_gate({}, replay, rank=1, xrank_families=FAMILIES)
        assert fail is None and block["fail"] is None and block["waived_cross_rank"] == []


def test_the_cross_rank_pass_waives_its_own_families_of_the_unreplayable_list_and_no_other():
    replay = {"result": True, "grade": "complete", "unreplayable": [{"family": ALL_REDUCE, "n": 3, "why": "no registered evaluator"},
                                                                     {"family": "Odd_v1", "n": 2, "why": "no registered evaluator"}]}
    block, fail = C2.no_evaluator_gate({}, replay, rank=1, xrank_families=FAMILIES)
    assert block["waived_cross_rank"] == [ALL_REDUCE]
    assert fail is not None and fail.startswith("no-evaluator families (rank 1): ") and fail.endswith("2 x no registered evaluator for Odd_v1")
    assert C2.no_evaluator_gate({}, replay, rank=1, xrank_families=None)[0]["waived_cross_rank"] == []


@pytest.mark.parametrize("codes", [{}, {f"no registered evaluator for {ALL_REDUCE}": None}], ids=["no code", "a null code"])
def test_a_reason_the_record_stamps_no_code_for_names_no_family_so_nothing_waives_it(codes):
    population = {"not_evaluable": {f"no registered evaluator for {ALL_REDUCE}": 3}, "not_evaluable_codes": codes}
    block, _ = C2.no_evaluator_gate({}, {"result": True, "grade": "complete", "population": population}, rank=1, xrank_families=FAMILIES)
    assert block["waived_cross_rank"] == []
    del population["not_evaluable_codes"]       # decoded from the reason's text, the family is the pass's own
    block, _ = C2.no_evaluator_gate({}, {"result": True, "grade": "complete", "population": population}, rank=1, xrank_families=FAMILIES)
    assert block["waived_cross_rank"] == [ALL_REDUCE]


def _annotated(xr: dict) -> dict:
    replays: list[Any] = [{"unreplayable": [{"family": ALL_REDUCE, "n": 3}, {"family": "Odd_v1", "n": 2}]}, {}, {"unreplayable": None}, None]
    C2.annotate_unreplayable(replays, xr, 4, sorted(FAMILIES))
    assert replays[1:] == [{}, {"unreplayable": None}, None]
    mine, other = replays[0]["unreplayable"]
    assert other == {"family": "Odd_v1", "n": 2}
    return mine["evaluated_by"]


@pytest.mark.parametrize("xr", [{}, {"result": None, "by_family": None, "coverage_of_unreplayable": None},
                                {"by_family": {}, "coverage_of_unreplayable": {}}, {"by_family": {"AllGather2_v1": {"picks": 1}}, "coverage_of_unreplayable": {"families": {}}}],
                         ids=["no record", "null blocks", "empty blocks", "another family's"])
def test_a_cross_rank_pass_that_picked_and_covered_nothing_of_a_family_is_named_with_no_counts(xr):
    by = _annotated(xr)
    assert by["pass"] == "cross_rank_collectives (TP-12)" and by["artifact"] == "xrank_pair4.json" and by["result"] is None
    assert [by[k] for k in ("picks", "equal", "sites", "sites_covered", "sites_missing")] == [None] * 5


def test_a_cross_rank_pass_that_ran_puts_its_picks_and_its_site_coverage_on_the_family():
    by = _annotated({"result": True, "by_family": {ALL_REDUCE: {"picks": 5, "equal": 4}},
                     "coverage_of_unreplayable": {"families": {ALL_REDUCE: {"sites": 3, "covered": 2, "missing": ["s2"]}}}})
    assert [by[k] for k in ("result", "picks", "equal", "sites", "sites_covered", "sites_missing")] == [True, 5, 4, 3, 2, ["s2"]]


# the Commit's components

REPLAY = {"result": True, "grade": "complete", "rank": 0, "sample": {"picked": 4, "evaluated": 4, "equal": 4, "per_stratum": 2, "seed": 7, "seed_source": "run root"},
          "population": {"vus": 10, "strata_n": 3, "source": "rows", "by_family": {"Gemm_v1": 7, ALL_REDUCE: 3}},
          "strata_keys": {"mechanisms": ["Gemm_v1"], "requests": ["r0", "r1"], "phases": ["prefill"], "steps": {"early": [0, 1], "late": [5]},
                          "lifetime": {"values": {"fresh-window": 6, "ring": 4}, "ring_slots": 8}, "key": "spec | request"},
          "weights": {"equal": True, "pinned": None}, "strata_unevaluated": ["s9"], "mismatch": [{"step": 1}],
          "unreplayable": [{"family": ALL_REDUCE, "n": 3, "why": "no registered evaluator"}]}
LINKAGE = {"result": True, "pairs": 6, "linked": 6}


@pytest.mark.parametrize("other", [{}, {"sampled_replay": None, "boundary_linkage": None}, {"sampled_replay": {}, "boundary_linkage": {}}],
                         ids=["no records", "null records", "empty records"])
def test_a_rank_with_no_replay_record_leaves_the_components_undecided_and_adds_nothing_to_their_sums(other):
    rows = [_run({"0": {"sampled_replay": REPLAY, "boundary_linkage": LINKAGE}, "1": other})]
    sr, lk = C2.sampled_replay_components(rows, {"sampled_replay": {"of_record": {}}})
    assert sr["result"] is None and lk["result"] is None
    assert sr["why"] == "rank 1: None no result" and lk["why"] == "rank 1: no result"
    assert sr["population"] == {"n": 10, "per_rank": {"0": 10, "1": None}, "source": "rows", "by_family": {ALL_REDUCE: 3, "Gemm_v1": 7}}
    assert sr["n"] == 4 and sr["sample"] == {"picked": 4, "evaluated": 4, "equal": 4, "mismatch_n": 0, "not_evaluated_n": 0, "per_stratum": 2}
    assert sr["strata"] == {"n": 3, "ranks": ["rank 0", "rank 1"], "mechanisms": ["Gemm_v1"], "requests": ["r0", "r1"], "phases": ["prefill"],
                            "steps": {"early": [0, 1], "late": [5]}, "lifetime": {"values": {"fresh-window": 6, "ring": 4}, "ring_slots": 8},
                            "key": "spec | request", "unevaluated": ["s9"]}
    assert sr["seed"] == {"0": {"seed": 7, "source": "run root"}, "1": {"seed": None, "source": None}}
    assert sr["weights"] == {"0": True, "1": None} and sr["weights_self_consistent"] == {"0": True, "1": None} and sr["weights_pinned"] == {"0": None, "1": None}
    assert sr["unreplayable"] == [{"family": ALL_REDUCE, "n": 3, "why": "no registered evaluator", "ranks": [0]}]
    assert sr["mismatch"] == [{"step": 1, "rank": 0}]
    assert sr["per_run"] == [{"0": {"sampled_replay": True, "grade": "complete", "boundary_linkage": True}, "1": {"sampled_replay": None, "grade": None, "boundary_linkage": None}}]
    assert lk["checked"] == 6 and lk["linked"] == 6 and lk["per_rank"]["1"] == {"result": None, "pairs": None, "linked": None, "why": None}


def test_a_replay_that_did_not_run_fails_the_component_by_its_grade_and_adds_nothing_to_its_sums():
    not_run = {"result": None, "grade": "not-evaluated", "why": "no Program of record", "rank": 1}
    rows = [_run({"0": {"sampled_replay": REPLAY, "boundary_linkage": LINKAGE}, "1": {"sampled_replay": not_run, "boundary_linkage": {"result": None, "why": "not run"}}})]
    sr, lk = C2.sampled_replay_components(rows, {"sampled_replay": {"of_record": {}}})
    assert sr["result"] is False and sr["why"] == "rank 1: not-evaluated no Program of record"
    assert lk["result"] is False and lk["why"] == "rank 1: not run"
    assert sr["population"]["n"] == 10 and sr["n"] == 4 and sr["strata"]["lifetime"] == {"values": {"fresh-window": 6, "ring": 4}, "ring_slots": 8}


def test_two_ranks_replays_add_their_populations_by_family_and_join_their_strata_keys():
    second = dict(REPLAY, rank=1, population={"vus": 5, "strata_n": 2, "by_family": {"Gemm_v1": 4, "RMSNorm_v1": 1}}, strata_unevaluated=None, mismatch=None,
                  strata_keys={"mechanisms": ["RMSNorm_v1"], "requests": ["r1", "r2"], "steps": {"early": [1, 2]}, "lifetime": {"values": {"ring": 1}}}, unreplayable=None)
    rows = [_run({"0": {"sampled_replay": REPLAY, "boundary_linkage": LINKAGE}, "1": {"sampled_replay": second, "boundary_linkage": LINKAGE}})]
    sr, lk = C2.sampled_replay_components(rows, {"sampled_replay": {"of_record": {}}})
    assert sr["result"] is True and lk["result"] is True and sr["why"] is None
    assert sr["population"] == {"n": 15, "per_rank": {"0": 10, "1": 5}, "source": "rows", "by_family": {ALL_REDUCE: 3, "Gemm_v1": 11, "RMSNorm_v1": 1}}
    assert sr["strata"] == {"n": 5, "ranks": ["rank 0", "rank 1"], "mechanisms": ["Gemm_v1", "RMSNorm_v1"], "requests": ["r0", "r1", "r2"], "phases": ["prefill"],
                            "steps": {"early": [0, 1, 2], "late": [5]}, "lifetime": {"values": {"fresh-window": 6, "ring": 5}, "ring_slots": 8},
                            "key": "spec | request", "unevaluated": ["s9"]}
    assert sr["unreplayable"] == [{"family": ALL_REDUCE, "n": 3, "why": "no registered evaluator", "ranks": [0]}] and sr["mismatch"] == [{"step": 1, "rank": 0}]


@pytest.mark.parametrize("size, refusal", [("x", ValueError), (["x"], TypeError)], ids=["a word", "a list"])
def test_a_size_that_is_no_number_is_what_the_fold_refuses_whatever_a_later_ranks_population_is(size, refusal):
    """The fold reads every rank's size before any rank's families, so the first rank's size is refused, not the second's population."""
    first = dict(REPLAY, population=dict(REPLAY["population"], vus=size))
    rows = [_run({"0": {"sampled_replay": first, "boundary_linkage": LINKAGE},
                  "1": {"sampled_replay": dict(REPLAY, rank=1, population="no table"), "boundary_linkage": LINKAGE}})]
    with pytest.raises(refusal, match=r"int\(\)"):
        C2.sampled_replay_components(rows, {"sampled_replay": {"of_record": {}}})


def test_a_rank_with_no_replay_record_is_not_a_deferred_one():
    rows = [_run({"0": {"sampled_replay": None, "query_population": {"ok": True}}, "1": {"query_population": {"ok": True}}})]
    assert C2.query_population_component(rows, True)["result"] is True
    rows = [_run({"0": {"sampled_replay": {"grade": "deferred"}, "query_population": {"ok": True}}, "1": {"query_population": {"ok": True}}})]
    assert C2.query_population_component(rows, True)["why"] == "deferred: ranks ['0'] replay from their bundles"


def test_a_decided_commit_names_the_runs_that_awaited_a_replay_or_held_no_commit(tmp_path: Path, capsys):
    rows = [{"arm": "control", "pair": 0}, {"arm": "instrumented", "pair": 0}, {"arm": "instrumented", "pair": 1, "commit": None},
            _run({}, pair=2, commit_pass=None), _run({}, pair=3, commit_pass=True), _run({}, pair=4, commit_pass=False)]
    summ = {"pass": False}
    C2.seal(tmp_path, summ, rows, {}, replay_cfg={"requests": []}, required_manifest=None, world=2, versions={}, all_pass=False, vc_base=None, comps={},
            xrank_pairs=[], xrank_families=[], label=None)
    assert capsys.readouterr().out == "[tp2commit] REPLAY BUNDLES discarded: the Commit is decided (pass False) with pairs [0, 1, 2] awaiting a replay\n"
    assert summ == {"pass": False} and list(tmp_path.iterdir()) == []
    C2.seal(tmp_path, summ, rows[-2:], {}, replay_cfg={"requests": []}, required_manifest=None, world=2, versions={}, all_pass=False, vc_base=None, comps={},
            xrank_pairs=[], xrank_families=[], label=None)
    assert capsys.readouterr().out == ""


def test_a_deferred_commit_seals_each_ranks_bundle_and_prints_a_line_for_each(tmp_path: Path, monkeypatch, capsys):
    sealed = []

    def seal(out, pair, ctx, source, *, hashed, rank):
        sealed.append((pair, rank, ctx["run_root"], ctx["tp_run_root"]))
        return {"path": f"/bundles/pair{pair}/rank{rank}", "sha256": f"{rank}a" * 32, "bytes": (3 + rank) << 20, "files": ["store", "plan"]}

    monkeypatch.setattr(RB, "seal", seal)
    rows = [{"arm": "control", "pair": 0}, _run({"0": {"run_root": "r0"}, "1": {"run_root": "r1"}}, commit_pass=None, tp_run_root="tp")]
    bundle: dict[str, Any] = {"engine_of_record": {}, "programs": {}, "hashed": None}
    summ: dict[str, Any] = {"pass": True}
    C2.seal(tmp_path, summ, rows, {0: {0: bundle, 1: bundle}}, replay_cfg={"requests": []}, required_manifest=None, world=2, versions={},
            all_pass=True, vc_base=None, comps={}, xrank_pairs=[{"result": True, "by_family": {}}], xrank_families=[], label=None)
    assert sealed == [(0, 0, "r0", "tp"), (0, 1, "r1", "tp")]
    assert [re.sub(r"sealed in \d+\.\d s", "sealed in N s", m) for m in capsys.readouterr().out.splitlines()] == [
        "[tp2commit] REPLAY BUNDLE (pair 0, rank 0): /bundles/pair0/rank0 sealed in N s, 2 files, 3 MiB, manifest sha256 0a0a0a0a0a0a0a0a",
        "[tp2commit] REPLAY BUNDLE (pair 0, rank 1): /bundles/pair0/rank1 sealed in N s, 2 files, 4 MiB, manifest sha256 1a1a1a1a1a1a1a1a"]
    assert summ["pass"] is None and summ["replay_deferred"]["pending_pairs"] == [0]
    assert [(b["rank"], b["path"], b["bytes"], b["files"]) for b in summ["replay_deferred"]["bundles"]] == [
        (0, "/bundles/pair0/rank0", 3 << 20, 2), (1, "/bundles/pair0/rank1", 4 << 20, 2)]
    assert [f.name for f in tmp_path.iterdir()] == [C2.DEFERRED]


PIN = {"result": True, "root_of_record": "aa", "compared": 9, "mismatched_n": 0, "missing": [], "unpinned": [], "all_shards_pinned": True, "committed_weights_root": "cc", "why": None}


@pytest.mark.parametrize("other", [{}, {"weights_pin": None}, {"weights_pin": {}}], ids=["no pin", "a null pin", "an empty pin"])
def test_a_rank_with_no_pin_record_leaves_the_commit_unpinned(other):
    rows = [{"arm": "control", "pair": 0, "commit": {"per_rank": {"7": {"weights_pin": PIN}}}}, {"arm": "instrumented", "pair": 0}, _run({"0": {"weights_pin": PIN}, "1": other})]
    comp = C2.weights_pin_component(rows, {})
    assert comp["result"] is None and comp["why"] == "weights unpinned (MAN-09) on a rank" and comp["checkpoint_dir"] is None and comp["pairs"] == 2
    assert comp["per_rank"] == {"0": PIN, "1": dict.fromkeys(PIN)}


def test_every_rank_pinned_is_a_pinned_commit_that_names_its_checkpoint_and_one_differing_rank_fails_it():
    rows = [_run({"0": {"weights_pin": PIN}, "1": {"weights_pin": PIN}})]
    comp = C2.weights_pin_component(rows, {"weights_pin": {"checkpoint_dir": "/checkpoint"}})
    assert comp["result"] is True and comp["why"] is None and comp["checkpoint_dir"] == "/checkpoint"
    assert C2.weights_pin_component(rows, {"weights_pin": None})["checkpoint_dir"] is None
    rows.append(_run({"0": {"weights_pin": dict(PIN, result=False)}, "1": {}}, pair=1))
    comp = C2.weights_pin_component(rows, {})
    assert comp["result"] is False and comp["why"] == "served weights differ from the checkpoint of record on a rank"


ATTRIBUTION = {"ok": True, "steps_checked": 5, "rows_checked": 9, "mismatches_n": 0, "requests": 2, "why": None}
ORACLE = {"verdict": "PASS", "mismatch_n": 0, "missing_n": 0, "not_in_oracle_n": 0, "tensors": 12}
MANIFEST = {"populations": {"by_family": {"tp_rank_partials": 4, "sampled_token_ids": 6, "instance_outputs": 30}}}


def _both(c0: dict, c1: dict) -> list[dict]:
    return [{"arm": "control", "pair": 0}, _run({"0": c0, "1": c1})]


def test_the_linkage_is_not_run_without_a_match_record_or_a_finalized_rank():
    rows = _both({"attribution": ATTRIBUTION}, {"attribution": ATTRIBUTION})
    for versions in ({}, {"match_record": None}):
        assert C2.linkage_component(rows, versions) == {"result": None, "method": "per-(request, engine step) committed sampled_token_ids vs the Match record",
                                                        "checked": 0, "why": "not run (no --match-dir Match record)"}
    assert C2.linkage_component(rows[:1] + [{"arm": "instrumented", "pair": 0}], MATCH)["why"] == "no instrumented pair finalized"


def test_the_linkage_passes_when_every_rank_attributes_its_rows_and_counts_them():
    comp = C2.linkage_component(_both({"attribution": ATTRIBUTION}, {"attribution": dict(ATTRIBUTION, rows_checked=7)}), MATCH)
    assert comp["result"] is True and comp["checked"] == 16 and comp["why"] is None and "(/match, tp2_match capture arm" in comp["method"]
    assert comp["per_rank"]["1"] == {"ok": True, "steps_checked": 5, "rows_checked": 7, "mismatches_n": 0, "requests": 2}


@pytest.mark.parametrize("other", [{}, {"attribution": None}, {"attribution": {}}], ids=["no attribution", "a null attribution", "an empty attribution"])
def test_a_rank_with_no_attribution_fails_the_linkage_and_adds_no_row(other):
    comp = C2.linkage_component(_both({"attribution": ATTRIBUTION}, other), MATCH)
    assert comp["result"] is False and comp["checked"] == 9 and comp["why"] == "None"
    assert comp["per_rank"]["1"] == {"ok": None, "steps_checked": None, "rows_checked": None, "mismatches_n": None, "requests": None}
    failed = C2.linkage_component(_both({"attribution": ATTRIBUTION}, {"attribution": {"ok": False, "why": "step 3 row 1"}}), MATCH)
    assert failed["result"] is False and failed["why"] == "step 3 row 1"


def test_the_oracle_compare_is_not_run_without_a_match_record_or_a_finalized_rank():
    not_run = {"result": None, "complete": False, "reference": "not run (no --match-dir: no Match record to compare the committed values against)"}
    assert C2.oracle_compare_component(_both({"attribution": ATTRIBUTION}, {}), {}, MANIFEST) == not_run
    assert C2.oracle_compare_component([{"arm": "instrumented", "pair": 0, "commit": None}], MATCH, MANIFEST) == not_run


def test_the_oracle_compare_passes_on_both_arms_and_counts_the_manifests_identities_it_covers():
    cblock = {"attribution": ATTRIBUTION, "match_oracle": ORACLE}
    comp = C2.oracle_compare_component(_both(cblock, cblock), MATCH, MANIFEST)
    assert comp["result"] is True and comp["complete"] is False and comp["why"] is None and comp["oracle_is_failed_match_run"] is False
    assert comp["attribution"] == {"complete": True, "result": True, "per_rank": {"0": True, "1": True}, "rows_checked_per_rank": {"0": 9, "1": 9}}
    assert comp["tp_partials"]["result"] is True and comp["tp_partials"]["per_rank"]["1"]["verdict"] == "PASS"
    assert comp["identities"] == {"families_covered": ["tp_rank_partials", "sampled_token_ids"], "families_not_covered": ["instance_outputs"], "covered_n": 10, "total_n": 40}


@pytest.mark.parametrize("manifest", [None, {}, {"populations": {}}, {"populations": {"by_family": None}}, {"populations": {"by_family": {}}}],
                         ids=["no manifest", "an empty manifest", "no counts by family", "null counts", "empty counts"])
def test_a_commit_whose_manifest_counts_no_family_names_the_families_covered_with_no_counts(manifest):
    cblock = {"attribution": ATTRIBUTION, "match_oracle": ORACLE}
    assert C2.oracle_compare_component(_both(cblock, cblock), MATCH, manifest)["identities"] == {
        "families_covered": ["tp_rank_partials", "sampled_token_ids"], "families_not_covered": ["instance_outputs", "fa2_hidden_m1_stream", "input_token_ids", "weights"],
        "covered_n": None, "total_n": None}


@pytest.mark.parametrize("oracle", [{}, {"match_oracle": None}, {"match_oracle": {}}], ids=["no oracle", "a null oracle", "an empty oracle"])
def test_a_commit_no_rank_of_which_ran_the_match_oracle_rests_on_the_attribution_and_covers_the_sampled_ids_alone(oracle):
    cblock = dict(oracle, attribution=ATTRIBUTION)
    comp = C2.oracle_compare_component(_both(cblock, cblock), MATCH, MANIFEST)
    assert comp["result"] is True and comp["why"] is None
    assert comp["tp_partials"] == {"result": None, "reference": "not run (--no-match-oracle or no stored capture steps)"}
    assert comp["identities"] == {"families_covered": ["sampled_token_ids"], "families_not_covered": ["tp_rank_partials", "instance_outputs"], "covered_n": 6, "total_n": 40}


def test_a_rank_with_no_attribution_or_no_oracle_beside_one_that_ran_fails_the_oracle_compare():
    comp = C2.oracle_compare_component(_both({"attribution": ATTRIBUTION, "match_oracle": ORACLE}, {}), MATCH, MANIFEST)
    assert comp["result"] is False and comp["why"] == "rank 1: attribution None"
    assert comp["attribution"] == {"complete": False, "result": False, "per_rank": {"0": True, "1": None}, "rows_checked_per_rank": {"0": 9, "1": None}}
    assert comp["tp_partials"]["result"] is False and comp["tp_partials"]["per_rank"]["1"]["verdict"] is None
    failed = dict(ORACLE, verdict="FAIL", why="a partial differs", mismatch_n=4)
    comp = C2.oracle_compare_component(_both({"attribution": ATTRIBUTION, "match_oracle": ORACLE}, {"attribution": ATTRIBUTION, "match_oracle": failed}), MATCH, MANIFEST)
    assert comp["result"] is False and comp["why"] == "rank 1: match oracle FAIL a partial differs mismatch 4 missing 0 not_in_oracle 0"


def test_a_commit_equal_to_a_failed_match_runs_capture_is_compared_and_is_no_pass():
    cblock = {"attribution": ATTRIBUTION, "match_oracle": ORACLE}
    comp = C2.oracle_compare_component(_both(cblock, cblock), {"match_record": {"path": "/match", "tp2_match_verdict": False}}, MANIFEST)
    assert comp["result"] is None and comp["oracle_is_failed_match_run"] is True and comp["oracle_match_verdict"] is False
    assert comp["oracle_note"] == "oracle is a FAILED Match run (tp2_match.json pass False): compared, not a PASS ingredient"
    differs = dict(ORACLE, verdict="FAIL", why="a partial differs", mismatch_n=4)
    comp = C2.oracle_compare_component(_both(cblock, {"attribution": ATTRIBUTION, "match_oracle": differs}),
                                       {"match_record": {"path": "/match", "tp2_match_verdict": False}}, MANIFEST)
    assert comp["result"] is False and comp["oracle_is_failed_match_run"] is True


# the Commit driver's readers of a document's block and list

@pytest.mark.parametrize("doc", [{}, {"name": None}, {"name": {}}, {"name": []}, {"name": 0}, {"name": ""}],
                         ids=["none", "a null one", "an empty block", "an empty list", "zero", "an empty string"])
def test_a_block_or_a_list_a_document_does_not_hold_reads_empty(doc):
    assert (C2.block_of(doc, "name"), C2.list_of(doc, "name")) == ({}, [])


def test_a_block_or_a_list_a_document_holds_is_handed_on_as_it_is():
    block, listed = {"result": None}, [0]
    doc = {"block": block, "list": listed}
    assert C2.block_of(doc, "block") is block and C2.list_of(doc, "list") is listed
    # neither reader types what it hands on: a list in a block's place is the caller's to read
    assert C2.block_of(doc, "list") is listed and C2.list_of(doc, "block") is block
