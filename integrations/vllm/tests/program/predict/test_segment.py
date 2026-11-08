"""`segment.edge_of`: what a class-G window's edges carry of each request, its history groups keyed on its certified max_tokens,
never on the output length its stop decides after the window."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import window_advice as WA
from verity_vllm.program.predict import dsa, segment
from verity_vllm.program.predict.common import Unexpressible


class _Net:
    """A one-layer decoder with one indexer and INDEX_TOPK 5, its groups and their layout `dsa.Net`'s own."""

    C = {"INDEX_TOPK": 5, "KV_LORA": 4, "QK_ROPE": 2, "INDEX_HEAD_DIM": 2}
    NL, indexers = 1, (0,)
    reads_run, edge_groups, prefix_params = dsa.Net.reads_run, dsa.Net.edge_groups, dsa.Net.prefix_params


def _step(*rows: tuple[int, int, int]) -> WA.StepAdvice:
    return WA.StepAdvice(tuple(WA.Row(r, "prefill", s, n, 0, False, ()) for r, s, n in rows))


def test_a_request_s_edge_groups_are_keyed_on_its_max_tokens_never_on_its_output_length(monkeypatch):
    # e and f have prompts of 3 and max_tokens 4, and the window leaves both after their prefills; e stops at its 2nd output (its
    # last row reads 4 keys, within INDEX_TOPK) and f runs to its 4th (6 keys): by their cap both may read past it
    monkeypatch.setattr(segment, "decoder", lambda phi, *_: _Net())
    e, f, g = WA.RequestAdvice("e", 3, 2, "eos"), WA.RequestAdvice("f", 3, 4, "length"), WA.RequestAdvice("g", 3, 1, "eos")
    w, units = WA.Advice((e, f, g), (_step((0, 0, 3), (1, 0, 3)), _step((2, 0, 3))), (0, 0, 1)), [SimpleNamespace(LP=3, T=0)] * 3
    edge = segment.edge_of(None, w, units, (), {0: 4, 1: 4, 2: 1})
    assert edge.ins == () and [i for i, _ in edge.outs] == [0, 1] and edge.outs[0][1] == edge.outs[1][1]
    assert [g for g, _, _ in edge.outs[0][1]] == ["prefix_kv_c", "prefix_kv_pe", "prefix_ik", "prefix_iks", "last_token"]
    assert segment.edge_groups(_Net(), 3, 2) == ("prefix_kv_c", "prefix_kv_pe"), "keyed on e's output length, it would drop them"
    with pytest.raises(Unexpressible, match="request 0: the window's edge keys its history groups on its certified max_tokens"):
        segment.edge_of(None, w, units, (), {1: 4, 2: 1})
