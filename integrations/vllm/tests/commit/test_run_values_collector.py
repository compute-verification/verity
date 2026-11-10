"""The collector's `run` values (`commit.committer.run_values`): what a rank's manifest requires of an indexer layer's rows past
INDEX_TOPK is committed under the manifest's names as the Program's `run` field holds it (`program.predict.dsa.run_bits`, the bits
a Build's `BuildResult.run` layout reads), from a copy taken at the layer (the top-k buffer is every layer's), in row order; an order
that routes no Waksman network is an error by name, and a manifest row the rank can't commit is refused."""
from __future__ import annotations

import random

import pytest
import torch

from verity.definitions.registry.dsa_topk import waksman_switches
from verity_vllm.commit.committer.run_values import RunValues
from verity_vllm.program.predict import dsa as D
from verity_vllm.query.manifest.format import RUN_FAMILY

TOPK = 4
LAYERS = {"model.layers.3.self_attn": 3, "model.layers.7.self_attn": 7}


def row(step: int, layer: int, pos: int, request: str = "r0", **kw) -> dict:
    op, member = D.run_field(layer, pos).rsplit("/", 1)
    return {"family": RUN_FAMILY, "request_id": request, "step": step, "engine_step": step, "invocation": 0, "rank": 0, "op_path": op,
            "output_member": member, "element_range": [0, waksman_switches(pos + 1, TOPK)], "dtype": "u8",
            "attention_module": f"model.layers.{layer}.self_attn", "statics": {"N": pos + 1, "TOPK": TOPK}, **kw}


def order(pos: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(pos + 1), TOPK)


def buffer(orders: list[list[int] | None]) -> torch.Tensor:
    """The indexer's top-k rows, two columns wider than TOPK, -1 where a row holds no order."""
    return torch.tensor([(o or [-1] * TOPK) + [-1, -1] for o in orders], dtype=torch.int32)


def bits(layer: int, pos: int, o: list[int]) -> list[int]:
    f = {"name": D.run_field(layer, pos), "offset": 0, "leaves": waksman_switches(pos + 1, TOPK)}
    return D.run_bits({"fields": [f]}, {(layer, pos): o}, TOPK)


def committed(rv: RunValues) -> tuple[list[tuple[str, list[int]]], list[str]]:
    staged: list[tuple[str, torch.Tensor]] = []
    errors = rv.commit(lambda n, t: staged.append((n, t)))
    assert all(t.dtype == torch.uint8 and t.dim() == 1 for _n, t in staged)
    return [(n, t.tolist()) for n, t in staged], errors


def test_the_committed_bits_are_the_programs_run_fields():
    """Layer 7's reader is lowered from the layer's `attn` child: a module under the layer's is the layer's."""
    rv = RunValues([row(2, 3, 9), row(2, 3, 6, "r1"), row(2, 7, 9, attention_module="model.layers.7.self_attn.attn"), row(5, 3, 12)], LAYERS)
    names = [D.run_field(3, 6), D.run_field(3, 9), D.run_field(7, 9)]
    assert rv.targets_at(2) == dict.fromkeys(names, 1) and rv.targets_at(1) == {} and rv.names == {*names, D.run_field(3, 12)}
    rv.begin_step()
    positions = torch.tensor([3, 6, 9])
    o3, o7 = [None, order(6, 1), order(9, 2)], [None, None, order(9, 3)]
    buf = buffer(o3)
    rv.take(2, 3, positions, buf, 3)
    buf[:] = buffer(o7)                                     # the next indexer layer writes the one buffer
    rv.take(2, 7, positions, buf, 3)
    got, errors = committed(rv)
    assert errors == []
    assert got == [(names[0], bits(3, 6, o3[1])), (names[1], bits(3, 9, o3[2])), (names[2], bits(7, 9, o7[2]))]
    layout = {"fields": [{"name": n, "offset": sum(len(b) for _n, b in got[:i]), "leaves": len(b)} for i, (n, b) in enumerate(got)]}
    whole = D.run_bits(layout, {(3, 6): o3[1], (3, 9): o3[2], (7, 9): o7[2]}, TOPK)
    assert [x for _n, b in got for x in b] == whole, "the fields concatenated are the run input a Build's layout reads"
    assert committed(rv) == ([], []), "a step's rows are committed once"


def test_two_requests_at_one_position_are_committed_in_row_order():
    rv = RunValues([row(2, 3, 9), row(2, 3, 9, "r1")], LAYERS)
    assert rv.targets_at(2) == {D.run_field(3, 9): 2}
    rv.begin_step()
    rows = [order(9, 4), order(9, 5)]
    rv.take(2, 3, torch.tensor([9, 9]), buffer(rows), 2)
    assert committed(rv) == ([(D.run_field(3, 9), bits(3, 9, rows[0])), (D.run_field(3, 9), bits(3, 9, rows[1]))], [])


def test_an_order_that_is_no_waksman_order_is_an_error_by_name():
    rv = RunValues([row(2, 3, 9), row(2, 3, 8)], LAYERS)
    rv.begin_step()
    rv.take(2, 3, torch.tensor([8, 9]), buffer([[0, 1, 2, 9], [5, 5, 1, 2]]), 2)
    got, errors = committed(rv)
    assert got == [] and len(errors) == 2
    assert errors[0].startswith(f"run values: {D.run_field(3, 8)}: a Waksman order") and errors[1].startswith(f"run values: {D.run_field(3, 9)}")


def test_a_row_the_rank_cannot_commit_is_refused():
    for bad, why in ((row(2, 3, 9, attention_module="model.layers.3.mlp"), "no attention layer"),
                     (row(2, 3, 9, attention_module="model.layers.30.self_attn"), "no attention layer"),
                     (row(2, 3, 9, statics={"TOPK": TOPK}), "no N and TOPK")):
        with pytest.raises(ValueError, match=why):
            RunValues([bad], LAYERS)
    with pytest.raises(ValueError, match="are both layer 3's row of 10 keys"):
        RunValues([row(2, 3, 9), {**row(2, 3, 9), "output_member": "other"}], LAYERS)


def test_without_run_rows_nothing_is_taken_or_committed():
    rv = RunValues([{"family": "instance_outputs", "op_path": "x", "output_member": "out", "step": 0}], LAYERS)
    assert rv.at == {} and rv.names == frozenset() and rv.targets_at(0) == {}
    rv.begin_step()
    rv.take(0, 3, torch.tensor([2048]), buffer([order(2048, 6)]), 1)
    assert committed(rv) == ([], [])
