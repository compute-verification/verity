"""`verity_vllm.pipeline.tp.rank_replay`: what a rank reads off its configuration, the merged manifest and its Programs
before its sampled replay and for the cross-rank pass.  `seed_of` is the sample's seed with the text that says where it
came from; `manifest_of_rank` is the rank's own identities of the merged manifest; `unreplayable_sites` are the sites of
the families the rank's replay leaves to the cross-rank pass; `xrank_facts` puts those and the routed specs the Programs
bind in the rank's state.  CPU-only: a Program's index is a stand-in that holds rows and keeps what it is asked to
describe."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.check.replay import sample
from verity_vllm.pipeline.tp import rank_replay

ROOT = bytes(range(1, 65))
ROOT_SOURCE = "this rank's run root (first 8 bytes, after commitment)"


def test_a_seed_the_driver_gives_is_the_seed_whatever_else_is_set():
    assert rank_replay.seed_of({"seed": 7}, ROOT) == (7, "--replay-seed (targeted)")
    assert rank_replay.seed_of({"seed": 0, "challenge": "c0ffee"}, ROOT) == (0, "--replay-seed (targeted)")
    assert rank_replay.seed_of({"seed": "12"}, ROOT) == (12, "--replay-seed (targeted)")


def test_a_challenge_seeds_the_sample_with_the_ranks_run_root():
    seed, source = rank_replay.seed_of({"seed": None, "challenge": "c0ffee"}, ROOT)
    assert (seed, source) == sample.challenge_seed("c0ffee", ROOT)
    assert "c0ffee" in source and source != ROOT_SOURCE
    assert rank_replay.seed_of({"challenge": "c0ffee"}, ROOT[::-1])[0] != seed       # another run root, another sample
    assert rank_replay.seed_of({"challenge": "decaf"}, ROOT)[0] != seed
    assert rank_replay.seed_of({"challenge": 1234}, ROOT) == rank_replay.seed_of({"challenge": "1234"}, ROOT)


@pytest.mark.parametrize("cfg", [{}, {"seed": None}, {"challenge": None}, {"challenge": ""}], ids=["nothing", "no seed", "no challenge", "an empty challenge"])
def test_with_neither_the_seed_is_the_run_roots_first_eight_bytes_and_says_so(cfg):
    assert rank_replay.seed_of(cfg, ROOT) == (0x0102030405060708, ROOT_SOURCE)
    assert rank_replay.seed_of(cfg, ROOT[:8] + bytes(56)) == (0x0102030405060708, ROOT_SOURCE)
    assert rank_replay.seed_of(cfg, bytes(64))[0] == 0


MERGED = {"schema": "a manifest", "ranks": 2, "identities": [{"id": "a", "rank": 0}, {"id": "b", "rank": 1}, {"id": "c"}, {"id": "d", "rank": None},
                                                             {"id": "e", "rank": "1"}]}


def test_a_ranks_manifest_is_the_merged_ones_identities_of_that_rank_and_the_rest_as_it_was():
    merged = copy.deepcopy(MERGED)
    assert rank_replay.manifest_of_rank(merged, 1) == {"schema": "a manifest", "ranks": 2, "identities": [{"id": "b", "rank": 1}, {"id": "e", "rank": "1"}]}
    # an identity that names no rank is rank 0's
    assert [x["id"] for x in rank_replay.manifest_of_rank(merged, 0)["identities"]] == ["a", "c", "d"]
    assert rank_replay.manifest_of_rank(merged, 2) == {"schema": "a manifest", "ranks": 2, "identities": []}
    assert merged == MERGED


@pytest.mark.parametrize("doc", [None, [], "a path", {}, {"identities": None}, {"identities": {"a": 1}}],
                         ids=["none", "a list", "a string", "no identities", "identities that are none", "identities that are no list"])
def test_what_is_no_manifest_of_identities_is_no_ranks_manifest(doc):
    assert rank_replay.manifest_of_rank(doc, 0) is None


def _loaded(rows: list[dict[str, Any]], world: int = 2, described: dict[str, dict[str, Any] | None] | None = None, **more: Any) -> SimpleNamespace:
    """A rank's loaded Programs as the readers see them: the index's rows, and for a row's node what the index describes it as
    (a site with one committed member unless given).  ``asked`` keeps the rows the index was asked to describe."""
    asked: list[str] = []

    def describe(row: dict[str, Any]) -> dict[str, Any] | None:
        asked.append(row["node"])
        return (described or {}).get(row["node"], {"members": [f"{row['node']}/out"], "op_path": row["node"]})

    return SimpleNamespace(world=world, index=SimpleNamespace(rows=rows, describe=describe), asked=asked, **more)


def _row(node: str, spec: str) -> dict[str, Any]:
    return {"node": node, "spec": spec}


def _refuse(*_asked: Any) -> Any:
    raise ValueError("y" * 300)


def test_the_unreplayable_sites_are_the_committed_sites_of_the_cross_rank_families_by_family():
    p = _loaded([_row("model.layers.1.mlp", "AllReduce2_v1{N=8}"), _row("model.layers.0.q", "Gemm_v1{K=4,N=8}"),
                 _row("model.lm_head", "AllGather2_v1{N=8}"), _row("model.layers.0.mlp", "AllReduce2_v1{N=8}"),
                 _row("model.embed_tokens", "EmbeddingShard_v1{V=16}"), _row("model.layers.1.mlp", "AllReduce2_v1{N=4}"),
                 _row("model.norm", "RmsNorm_v1{N=8}")])
    assert rank_replay.unreplayable_sites(p) == {"AllReduce2_v1": ["model.layers.0.mlp", "model.layers.1.mlp"], "AllGather2_v1": ["model.lm_head"],
                                                 "EmbeddingShard_v1": ["model.embed_tokens"]}
    # the rows of the families the replay evaluates are not described at all
    assert p.asked == ["model.layers.1.mlp", "model.lm_head", "model.layers.0.mlp", "model.embed_tokens", "model.layers.1.mlp"]


def test_a_site_is_named_by_the_path_the_index_describes_it_at():
    p = _loaded([_row("n7", "AllReduce2_v1{N=8}")], described={"n7": {"members": ["model.layers.0.mlp/out"], "op_path": "model.layers.0.mlp"}})
    assert rank_replay.unreplayable_sites(p) == {"AllReduce2_v1": ["model.layers.0.mlp"]}


@pytest.mark.parametrize("described", [None, {}, {"members": [], "op_path": "x"}, {"op_path": "x"}],
                         ids=["a row the index cannot describe", "an empty description", "no committed member", "no members named"])
def test_a_cross_rank_row_with_no_committed_member_is_no_site(described):
    p = _loaded([_row("x", "AllReduce2_v1{N=8}"), _row("y", "AllGather2_v1{N=8}")], described={"x": described})
    assert rank_replay.unreplayable_sites(p) == {"AllGather2_v1": ["y"]}


def test_the_families_are_the_worlds_and_any_all_reduce_route():
    """At world 8 the received outputs are `AllReduce_v2` and `AllGather_v1`; an all-reduce a Program routes is the cross-rank
    pass's at any world, and world 2's all-gather is no family of world 8."""
    rows = [_row("a", "AllReduce_v2{N=8}"), _row("b", "AllGather_v1{N=8}"), _row("c", "AllReduceMnnvl_v1{N=8}"), _row("d", "AllGather2_v1{N=8}"),
            _row("e", "AllReduceRmsNormMnnvl_v1{N=8}"), _row("f", "EmbeddingShard_v1{V=16}")]
    assert rank_replay.unreplayable_sites(_loaded(rows, world=8)) == {
        "AllReduce_v2": ["a"], "AllGather_v1": ["b"], "AllReduceMnnvl_v1": ["c"], "AllReduceRmsNormMnnvl_v1": ["e"], "EmbeddingShard_v1": ["f"]}
    assert rank_replay.unreplayable_sites(_loaded(rows, world=2)) == {
        "AllReduce_v2": ["a"], "AllReduceMnnvl_v1": ["c"], "AllGather2_v1": ["d"], "AllReduceRmsNormMnnvl_v1": ["e"], "EmbeddingShard_v1": ["f"]}


def test_a_program_with_no_cross_rank_row_has_no_unreplayable_site():
    assert rank_replay.unreplayable_sites(_loaded([_row("q", "Gemm_v1{K=4,N=8}")])) == {}
    assert rank_replay.unreplayable_sites(_loaded([])) == {}


def test_rows_that_cannot_be_read_are_an_error_by_name_and_never_raised():
    assert rank_replay.unreplayable_sites(_loaded([{"node": "x"}])) == {"error": "KeyError('spec')"}
    assert rank_replay.unreplayable_sites(SimpleNamespace(world=2, index=None)) == {"error": "AttributeError(\"'NoneType' object has no attribute 'rows'\")"}
    assert rank_replay.unreplayable_sites(_loaded([_row("x", "AllReduce2_v1{N=8}")], described={"x": {"members": ["m"]}})) == {"error": "KeyError('op_path')"}
    cut = rank_replay.unreplayable_sites(SimpleNamespace(world=2, index=SimpleNamespace(rows=[_row("x", "AllReduce2_v1{N=8}")], describe=_refuse)))
    assert cut == {"error": ("ValueError('" + "y" * 300)[:200]}


def test_the_cross_rank_facts_of_a_rank_are_its_sites_and_the_routed_specs_its_programs_bind():
    routed, other = "AllReduceMnnvl_v1{N=8,WORLD=8}", "AllReduceMultimem_v1{N=8,WORLD=8}"
    progs = {"r0": {"rows": [{"spec": routed}, {"spec": "Gemm_v1{K=4,N=8}"}, {"spec": "AllReduce_v2{N=8}"}]},
             "r1": {"rows": [{"spec": other}, {"spec": routed}, {"spec": None}, {}]}, "r2": {}}
    st: dict[str, Any] = {"kept": 1}
    assert rank_replay.xrank_facts(st, _loaded([_row("a", "AllReduce_v2{N=8}"), _row("c", routed)], world=8, progs=progs)) is None
    assert st == {"kept": 1, "xrank_unreplayable_sites": {"AllReduce_v2": ["a"], "AllReduceMnnvl_v1": ["c"]}, "xrank_program_specs": [routed, other]}


def test_programs_whose_specs_cannot_be_read_leave_none_and_the_error_by_name():
    """The cross-rank pass then holds every routed pick not evaluable; the sites are still the index's."""
    st: dict[str, Any] = {"xrank_program_specs": ["an earlier pair's"]}
    rank_replay.xrank_facts(st, _loaded([_row("a", "AllReduce2_v1{N=8}")], progs={"r0": {"rows": 5}}))
    assert st == {"xrank_unreplayable_sites": {"AllReduce2_v1": ["a"]}, "xrank_program_specs": None,
                  "xrank_program_specs_error": "TypeError(\"'int' object is not iterable\")"}
    rank_replay.xrank_facts(st, _loaded([], progs={"r0": {"rows": [SimpleNamespace(get=_refuse)]}}))
    assert st == {"xrank_unreplayable_sites": {}, "xrank_program_specs": None, "xrank_program_specs_error": ("ValueError('" + "y" * 300)[:200]}
