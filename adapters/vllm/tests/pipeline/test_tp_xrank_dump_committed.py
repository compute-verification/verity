"""`verity_vllm.pipeline.tp.xrank_dump.dump`: a rank's picks for the cross-rank pass carry the rank's committed bytes and
nothing else.  Each member is opened from the rank's own committer and verified against its run root
(`verity_vllm.commit.opened.OpenedReader`) before a pick holds it; a member that does not verify, that the rank did not
retain, or that a rank with no run commitment would read is a pick without bytes whose ``why`` names it, which the
driver's check fails.  CPU-only: the commitment is the library's own CPU core's over a few small tensors
(`tests.check.opened.commit_steps`), and the engine is one module that is a vocabulary shard."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests.check.opened import commit_steps, meta, retained_block
from verity_vllm.pipeline.tp import xrank_dump

EMB, MLP, HEAD = "model.embed_tokens", "model.layers.0.mlp", "model.lm_head"
LO, HI = 2, 6
SHARD = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3).to(torch.bfloat16)
IDS = torch.tensor([1, 2, 5, 6, 3])
#: the embedding's partial on this rank: the shard's rows of the ids in its range, zero for the two outside it
E_PART = torch.stack([torch.zeros(3), SHARD[0].float(), SHARD[3].float(), torch.zeros(3), SHARD[1].float()]).to(torch.bfloat16)
E_RECV = (E_PART.float() + 16).to(torch.bfloat16)
H_PART = torch.arange(10, dtype=torch.float32).reshape(5, 2).to(torch.bfloat16)
H_RECV = torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16)
M_PART = torch.arange(15, dtype=torch.float32).reshape(5, 3).to(torch.bfloat16)
M_RECV = (M_PART.float() + 32).to(torch.bfloat16)
#: step 0 is the embedding's all-reduce and the head's all-gather, and step 1 the MLP's all-reduce, in a row of its own
STEPS = {0: {f"{EMB}/in": IDS, f"{EMB}/part_rank0": E_PART, EMB: E_RECV, f"{HEAD}/shard_rank0": H_PART, HEAD: H_RECV},
         1: {f"{MLP}/part_rank0": M_PART, f"{MLP}/0": M_RECV}}
NOT_VERIFIED = "the opening does not verify against the run root"
NOT_FINALIZED = "no run commitment on this rank (not finalized)"


def _raw(t: torch.Tensor) -> bytes:
    return bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist())


def _state(steps: dict[int, dict[str, torch.Tensor]] = STEPS, *, rank: int = 0, world: int = 2, **kw: Any) -> dict[str, Any]:
    """A rank that committed ``steps`` and finalized: its committer, its run commitment and its engine's one shard."""
    com, rc = commit_steps({s: [(n, _raw(t), i, str(t.dtype).removeprefix("torch."), tuple(t.shape)) for i, (n, t) in enumerate(ms.items())]
                            for s, ms in steps.items()}, **kw)
    shard = SimpleNamespace(weight=SHARD, shard_indices=SimpleNamespace(org_vocab_start_index=LO, org_vocab_end_index=HI))
    return {"com": com, "rank": rank, "world": world, "rc": rc, "eng": SimpleNamespace(model=SimpleNamespace(named_modules=lambda: [(EMB, shard)]))}


def _dump(st: dict[str, Any], tmp_path: Path, **kw: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(the record the rank answers with, the saved picks by site); every site of these worlds is picked once."""
    wrote = xrank_dump.dump(st, settings={}, out_dir=str(tmp_path), seed_hex="00" * 8, pair=0, **kw)
    picks = torch.load(wrote["file"], weights_only=False)["picks"]
    assert wrote["picks"] == len(picks) == len({p["site"] for p in picks})
    return wrote, {p["site"]: p for p in picks}


def _change(st: dict[str, Any], step: int, name: str) -> None:
    """One bit of the rank's retained copy of a member, after the commitment."""
    retained_block(st["com"], step)[meta(st["com"], step, name).dev_off] ^= 1


def test_a_ranks_picks_are_its_committed_bytes_opened_and_verified_against_its_run_root(tmp_path):
    st = _state()
    wrote, picks = _dump(st, tmp_path)
    assert list(picks) == [EMB, MLP, HEAD]
    for site, part, recv, member in ((EMB, E_PART, E_RECV, EMB), (MLP, M_PART, M_RECV, f"{MLP}/0"), (HEAD, H_PART, H_RECV, HEAD)):
        p = picks[site]
        assert (p["rows"], p["recv_member"], p["why"]) == ([0, 1, 2, 3, 4], member, None)
        assert p["part"].dtype == p["recv"].dtype == torch.bfloat16
        assert torch.equal(p["part"], part) and torch.equal(p["recv"], recv)
    assert picks[EMB]["emb"] == {"ok": True, "n_diff": 0, "vocab_range": [LO, HI], "masked_rows": 2, "rows": 5, "why": None}
    assert picks[MLP]["emb"] is None and picks[HEAD]["emb"] is None        # the check is the embedding site's all-reduce's alone
    src = wrote["value_source"]
    # two reads a pick and the embedding's ids, each verified, and both steps' roots held to the run root
    assert (src["source"], src["run_root"], src["reads"], src["failed"], src["not_retained"], src["steps_bound"]) == (
        "opened", st["rc"].run_root.hex(), 7, 0, 0, 2)
    assert (wrote["rank"], wrote["missing"], wrote["bytes"]) == (0, 0, 30 + 30 + 30 + 30 + 20 + 40)
    assert (wrote["strata"], wrote["strata_sites"], wrote["families"]) == (3, [EMB, MLP, HEAD], ["AllGather2_v1", "AllReduce2_v1"])
    assert (wrote["unreplayable_sites"], wrote["flipped_collective_inputs"], wrote["routed_specs"], wrote["program_specs_n"]) == (None, 0, [], None)


def test_a_pick_of_some_rows_carries_those_rows_of_both_members(tmp_path):
    wrote, picks = _dump(_state(), tmp_path, max_rows=2)
    for site, part, recv in ((EMB, E_PART, E_RECV), (MLP, M_PART, M_RECV), (HEAD, H_PART, H_RECV)):
        rows = picks[site]["rows"]
        assert len(rows) == 2 and rows == sorted(set(rows)) and set(rows) < set(range(5)) and picks[site]["rows_total"] == 5
        assert torch.equal(picks[site]["part"], part[rows]) and torch.equal(picks[site]["recv"], recv[rows])
    emb = picks[EMB]["emb"]
    assert (emb["ok"], emb["n_diff"], emb["rows"]) == (True, 0, 2)
    assert emb["masked_rows"] == sum(not LO <= int(IDS[r]) < HI for r in picks[EMB]["rows"])
    assert wrote["bytes"] == 2 * 2 * (3 + 3 + 3 + 3 + 2 + 4)


def test_a_seed_picks_the_same_rows_on_each_rank_and_another_seed_its_own(tmp_path):
    """The ranks of a pair pick by the pair's seed and the committed sizes alone, so each rank's file names the same rows."""
    def rows_of(seed: str, rank: int = 0) -> list[int]:
        st = _state({1: {f"{MLP}/part_rank{rank}": torch.cat([M_PART] * 8), f"{MLP}/0": torch.cat([M_RECV] * 8)}}, rank=rank)
        wrote = xrank_dump.dump(st, settings={}, out_dir=str(tmp_path), seed_hex=seed, pair=0, max_rows=3)
        assert (wrote["rank"], Path(wrote["file"]).name) == (rank, f"xrank_pair0_rank{rank}.pt")
        return torch.load(wrote["file"], weights_only=False)["picks"][0]["rows"]

    assert rows_of("00" * 8) == rows_of("00" * 8) == rows_of("00" * 8, rank=1)
    assert len({tuple(rows_of(f"{k:016x}")) for k in range(8)}) > 1


@pytest.mark.parametrize("member", [f"{MLP}/part_rank0", f"{MLP}/0"], ids=["the partial", "the received output beside it in the row"])
def test_bytes_changed_after_the_commitment_are_not_handed_on(tmp_path, member):
    """One bit of a retained member: no member of its row verifies, the pick holds no tensor and names both, and the picks of
    the other step's rows are what they were."""
    why = (f"partial opened value not verified: step 1 {MLP}/part_rank0 (ordinal 0) bytes [0, 30) = leaves [0, 1): {NOT_VERIFIED}; "
           f"received output opened value not verified: step 1 {MLP}/0 (ordinal 1) bytes [0, 30) = leaves [0, 1): {NOT_VERIFIED}")
    st = _state()
    _change(st, 1, member)
    wrote, picks = _dump(st, tmp_path)
    assert (picks[MLP]["part"], picks[MLP]["recv"], picks[MLP]["why"]) == (None, None, why)
    assert torch.equal(picks[EMB]["part"], E_PART) and torch.equal(picks[HEAD]["recv"], H_RECV)
    assert picks[EMB]["why"] is None and picks[HEAD]["why"] is None and picks[EMB]["emb"]["ok"] is True
    assert (wrote["missing"], wrote["bytes"], wrote["value_source"]["failed"], wrote["value_source"]["reads"]) == (1, 120, 2, 5)
    assert [(f["step"], f["name"]) for f in wrote["value_source"]["failures"]] == [(1, f"{MLP}/part_rank0"), (1, f"{MLP}/0")]


def test_a_reason_that_names_a_long_member_is_cut_to_200_characters(tmp_path):
    site = "model.layers.0." + "m" * 180
    st = _state({1: {f"{site}/part_rank0": M_PART, f"{site}/0": M_RECV}})
    _change(st, 1, f"{site}/0")
    why = _dump(st, tmp_path)[1][site]["why"]
    partial, received = why.split("; received output ")
    assert partial == f"partial opened value not verified: step 1 {site}/part_rank0 (ordinal 0) bytes [0, 30) = leaves [0, 1): {NOT_VERIFIED}"[:208]
    assert received == f"opened value not verified: step 1 {site}/0 (ordinal 1) bytes [0, 30) = leaves [0, 1): {NOT_VERIFIED}"[:200]
    assert len(received) == 200 and not received.endswith("root")


def test_ids_changed_after_the_commitment_leave_the_embedding_check_unanswered_by_name(tmp_path):
    st = _state()
    _change(st, 0, f"{EMB}/in")
    wrote, picks = _dump(st, tmp_path)
    assert picks[EMB]["part"] is None and picks[EMB]["recv"] is None and picks[HEAD]["part"] is None
    assert picks[EMB]["emb"] == {"ok": None, "why": f"opened value not verified: step 0 {EMB}/in (ordinal 0) bytes [0, 40) = leaves [0, 1): {NOT_VERIFIED}"}
    assert picks[MLP]["why"] is None and torch.equal(picks[MLP]["part"], M_PART)
    assert (wrote["missing"], wrote["bytes"], wrote["value_source"]["failed"]) == (2, 60, 5)


def test_a_rank_that_did_not_finalize_hands_on_no_bytes_and_says_so(tmp_path):
    st = dict(_state(), rc=None)
    wrote, picks = _dump(st, tmp_path)
    for site in (EMB, MLP, HEAD):
        assert (picks[site]["part"], picks[site]["recv"]) == (None, None)
        assert picks[site]["why"] == f"partial {NOT_FINALIZED}; received output {NOT_FINALIZED}"
    assert picks[EMB]["emb"] == {"ok": None, "why": NOT_FINALIZED}
    assert (wrote["missing"], wrote["bytes"]) == (3, 0)
    assert wrote["value_source"] == {"source": "opened", "reads": 0, "why": "no run commitment on this rank"}


def test_bytes_the_rank_did_not_retain_are_named_as_not_retained(tmp_path):
    wrote, picks = _dump(_state(retain="none"), tmp_path)
    for site in (EMB, MLP, HEAD):
        assert (picks[site]["part"], picks[site]["recv"], picks[site]["why"]) == (None, None, "partial bytes not retained; received output bytes not retained")
    assert picks[EMB]["emb"] == {"ok": None, "why": "no committed ids member <site>/in"}
    src = wrote["value_source"]
    assert (wrote["missing"], wrote["bytes"], src["reads"], src["not_retained"], src["failed"]) == (3, 0, 0, 7, 0)


def test_a_received_output_under_none_of_its_names_is_missing_and_the_names_are_listed(tmp_path):
    wrote, picks = _dump(_state({1: {f"{MLP}/part_rank0": M_PART, f"{MLP}/reduced": M_RECV}}), tmp_path)
    p = picks[MLP]
    assert (p["recv"], p["recv_member"], p["why"]) == (None, None, f"received output not committed under any of {MLP}/out/{MLP}/0/{MLP}")
    assert torch.equal(p["part"], M_PART) and (wrote["missing"], wrote["bytes"]) == (1, 0)


@pytest.mark.parametrize("name", ["out", "0"])
def test_a_received_output_is_read_under_the_first_of_its_names_the_step_has(tmp_path, name):
    other = (M_RECV.float() + 1).to(torch.bfloat16)
    _wrote, picks = _dump(_state({1: {f"{MLP}/part_rank0": M_PART, MLP: other, f"{MLP}/{name}": M_RECV}}), tmp_path)
    assert picks[MLP]["recv_member"] == f"{MLP}/{name}" and torch.equal(picks[MLP]["recv"], M_RECV)


def test_a_received_output_of_another_row_count_is_named_and_both_are_kept_whole(tmp_path):
    wrote, picks = _dump(_state({1: {f"{MLP}/part_rank0": M_PART, f"{MLP}/0": M_RECV[:4]}}), tmp_path, max_rows=2)
    p = picks[MLP]
    assert p["why"] == "row counts differ: partial (5, 3) vs received (4, 3)"
    assert torch.equal(p["part"], M_PART) and torch.equal(p["recv"], M_RECV[:4]) and len(p["rows"]) == 2
    assert (wrote["missing"], wrote["bytes"]) == (0, 30 + 24)


@pytest.mark.parametrize("shape", [(1, 5, 3), (15,)], ids=["three sizes", "one size"])
def test_at_the_routed_world_a_partial_that_is_not_m_by_n_is_no_routed_call_and_says_so(tmp_path, shape):
    """World 8 routes an [M, N] all-reduce to the Definition its call takes; a partial of any other number of sizes is named,
    with its bytes still the committed ones."""
    steps = {1: {f"{MLP}/part_rank0": M_PART.reshape(shape), f"{MLP}/0": M_RECV.reshape(shape)}}
    wrote, picks = _dump(_state(steps, world=8), tmp_path)
    p = picks[MLP]
    assert p["route_why"] == f"partial {MLP}/part_rank0 of shape {shape}, not [M, N]"
    assert "spec" not in p and p["why"] is None
    assert torch.equal(p["part"], M_PART.reshape(shape)) and torch.equal(p["recv"], M_RECV.reshape(shape))
    assert wrote["routed_specs"] == [] and "route_why" not in _dump(_state(steps), tmp_path)[1][MLP]
