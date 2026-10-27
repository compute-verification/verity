"""tp2_commit reads the sampled replay's POPULATION reconciliation against Q(P) (`population.query_checks`) per rank
INTO the TP verdict, fail closed, by the dense committer's rule (commit_verdict._query_population_scope).  The negative is a TP2 B1 256/32
row's finalize_pair<k>.json: every rank, every pair, query_checks.ok False with
vus_outside_query 960 + identities_without_rows 960 (the shared rotary module: rows at model.layers.k.self_attn.rotary_emb, k >= 1, vs the
manifest's alias identities layers.0 invocation 1..15, the TP merge having dropped op_path_aliases) under result True / grade complete,
which must not PASS the commit.  No engine, no torch: the gate is a pure function of the record."""
from __future__ import annotations

import pytest

from verity_vllm.pipeline.tp.c2 import query_population_gate


def _first(rank: int, k: int = 1):
    return [{"request_id": "r0", "step": 0, "op_path": f"model.layers.{k}.self_attn.rotary_emb", "member": "q", "rank": rank}]


def _sr(rank: int, *, ok, outside: int = 0, without_rows: int = 0, unreachable: int | None = None, with_total: bool = True, checks: bool = True):
    """A rank's sampled_replay record (the finalize_pair<k>.json[rank].sampled_replay.sampled_replay block), reduced to the population fields."""
    chk = {"vus_outside_query": {"n": outside, "first": _first(rank) if outside else []},
           "identities_without_rows": {"n": without_rows, "first": _first(rank, 0) if without_rows else []},
           "rows_disagree": {"n": 0, "first": []}, "ranks": {"replay": [rank], "manifest": [0, 1], "equal": True}, "accounted": True,
           "unstated_families": [], "executed_prefix_rule": {"applied": True, "faults": [], "evaluated_past_stop": {"n": 0, "first": []},
                                                             "manifest_cut_before_classification": {"n": 0, "first": []}}, "ok": ok}
    if unreachable is not None:                              # the cause named beside the two symptoms
        chk["identities_unreachable_by_alias"] = {"n": unreachable, "first": _first(rank, 0)[:1] if unreachable else [], "alias_entries": 0}
    pop = {"vus": 8067, "scope": "Q_module_v1(P) = 8067 identities (...)"}
    if with_total:
        pop["query_total"] = {"identities": 8067, "query_id": "Q_module_v1", "cross_check": {"agree": True}}
    if checks:
        pop["query_checks"] = chk
    return {"result": True, "grade": "complete", "population": pop, "sample": {"evaluated": 918, "equal": 918}}


def test_reconciled_rank_passes_and_states_the_scope():
    block, fail = query_population_gate(_sr(0, ok=True), rank=0, required=True)
    assert fail is None
    assert block["present"] is True and block["ok"] is True and block["fail"] is None
    assert block["counts"] == {"vus_outside_query": 0, "identities_without_rows": 0, "rows_disagree": 0}
    assert "Q_module_v1" in (block["note"] or "")             # the query of record by id, on every PASS


@pytest.mark.parametrize("rank", [0, 1])
def test_the_int2_row_of_record_shape_fails_by_name_on_every_rank(rank):
    """960 VUs outside Q(P) + 960 required identities without rows, result True / grade complete -- the exact int-2 TP2 B1 record."""
    block, fail = query_population_gate(_sr(rank, ok=False, outside=960, without_rows=960), rank=rank, required=True)
    assert fail is not None and fail.startswith(f"C2 population (rank {rank})")
    assert "vus_outside_query x960" in fail and "identities_without_rows x960" in fail
    assert "layers.1.self_attn.rotary_emb" in fail            # the first offending identity is named
    assert block["ok"] is False and block["counts"]["vus_outside_query"] == 960


def test_the_cause_named_by_reqval_travels_into_the_text_when_present():
    """identities_unreachable_by_alias beside the two symptoms: whatever the dense text names, the TP verdict carries the count."""
    block, fail = query_population_gate(_sr(1, ok=False, outside=960, without_rows=960, unreachable=960), rank=1, required=True)
    assert fail is not None
    assert block["counts"]["identities_unreachable_by_alias"] == 960


def test_ok_false_without_query_total_is_still_a_named_fail():
    block, fail = query_population_gate(_sr(0, ok=False, outside=3, with_total=False), rank=0, required=True)
    assert fail is not None and "vus_outside_query x3" in fail


def test_absent_checks_with_a_required_manifest_fail_closed():
    """A record that cannot say whether it reconciled does not PASS over that silence (the int-2 record was silent in the verdict, not in the field)."""
    block, fail = query_population_gate(_sr(0, ok=None, checks=False), rank=0, required=True)
    assert block["present"] is False
    assert fail is not None and "no `population.query_checks`" in fail
    block, fail = query_population_gate(_sr(1, ok=None), rank=1, required=True)
    assert fail is not None and "ok` None" in fail
    _, fail = query_population_gate(None, rank=0, required=True)
    assert fail is not None
    _, fail = query_population_gate({}, rank=0, required=True)
    assert fail is not None


def test_without_a_required_manifest_absence_is_stated_not_failed():
    block, fail = query_population_gate(_sr(0, ok=None, checks=False), rank=0, required=False)
    assert fail is None and block["present"] is False and block["ok"] is None
    block, fail = query_population_gate(_sr(0, ok=None), rank=0, required=False)
    assert fail is None and "NOT reconciled" in (block["note"] or "")


def test_a_cross_check_disagreement_on_the_manifest_fails_the_rank():
    sr = _sr(0, ok=True)
    sr["population"]["query_total"]["cross_check"] = {"agree": False, "why": "traversal 8067 vs inventory 8000"}
    _, fail = query_population_gate(sr, rank=0, required=True)
    assert fail is not None and "DISAGREE" in fail
