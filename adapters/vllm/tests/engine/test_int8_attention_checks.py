"""The CPU half of `tests.engine.int8_attention_gpu`, the served int8 attention's checks: its taps on stand-in layers, the
Definitions evaluated on a window of `verity_vllm.engine.int8_attention`'s reference, and the two judgements over the kept
calls (`written`, `batch_vs_solo`).  `compare`'s reruns and `fattn_ref_row` at head dimension 128 need a GPU and are not here."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.engine import int8_attention_gpu as AG
from verity_vllm.engine import int8_attention as A

HEAD = 32


def _bf16(*shape: int, first: int = 0) -> torch.Tensor:
    """A bfloat16 tensor of exact sixty-fourths between -50/64 and 50/64, different for each ``first``."""
    count = 1
    for extent in shape:
        count *= extent
    steps = (torch.arange(first, first + count) * 37) % 101 - 50
    return (steps.float() / 64).reshape(shape).to(torch.bfloat16)


def _window(n: int, heads: int, groups: int, s: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    return (_bf16(n, heads, HEAD), _bf16(s, groups, HEAD, first=1000), _bf16(s, groups, HEAD, first=2000),
            torch.full((groups,), 0.05), HEAD ** -0.5)


@pytest.mark.parametrize("n,heads,groups,s,evaluated", [(2, 2, 1, 3, 5), (1, 4, 2, 4, 6)])
def test_the_definitions_hold_on_the_references_window(n, heads, groups, s, evaluated):
    q, k, v, ks, scale = _window(n, heads, groups, s)
    out, served = A.window_attention(q, k, v, ks, scale), list(A.kv_quant(k, v, ks))
    # Int8KvQuant_v1 of the first and last KV heads, Int8LutAttentionRow_v1 of the first and last query rows at every head
    assert AG.defs(q, k, v, ks, scale, out, served) == {"defs_evaluated": evaluated, "defs_mismatch": 0}
    # one bit of one served output word, and one served K' byte: each is a mismatch of the Definition that binds it
    flipped = out.clone()
    flipped.view(torch.int16)[0, 0, 0] ^= 1
    assert AG.defs(q, k, v, ks, scale, flipped, served)["defs_mismatch"] == 1
    moved = [t.clone() for t in served]
    moved[0][0, 0, 0] += 1
    assert AG.defs(q, k, v, ks, scale, out, moved)["defs_mismatch"] >= 1


def test_the_fattn_reference_is_only_asked_at_its_own_head_dimension():
    q, k, _, ks, scale = _window(1, 2, 1, 3)
    assert AG.FATTN_REF_HEAD_DIM == 128 and AG.fattn_ref_row(q, k, ks, scale) is None


def _kept(call: int, request: int, k: torch.Tensor, v: torch.Tensor, handed: int) -> dict[str, Any]:
    """A kept call of a request whose window is ``k`` and ``v`` and which was handed the window's last ``handed`` rows."""
    return {"call": call, "request": request, "s": int(k.shape[0]), "n": handed, "k": k, "v": v, "handed": (k[-handed:], v[-handed:])}


def test_written_holds_a_decodes_window_to_what_the_request_was_handed_before():
    k, v = _bf16(4, 1, 2), _bf16(4, 1, 2, first=500)
    grown = [_kept(0, 0, k[:2], v[:2], 2), _kept(1, 0, k[:3], v[:3], 1), _kept(2, 0, k, v, 1)]
    assert AG.written({("batched", "a"): grown}) == {"written_compared": 2, "written_mismatch": 0}
    # the calls are judged in call order, whatever order they were kept in, and each layer and phase on its own
    assert AG.written({("batched", "a"): grown[::-1], ("solo0", "a"): grown}) == {"written_compared": 4, "written_mismatch": 0}
    # a window whose earlier rows are not the rows handed before
    other = k.clone()
    other[0, 0, 0] += 1
    assert AG.written({("batched", "a"): [grown[0], grown[1], _kept(2, 0, other, v, 1)]}) == {"written_compared": 2, "written_mismatch": 1}
    # nothing to hold a decode to: its request's earlier call was not kept, or the kept history is not as long as its past
    assert AG.written({("batched", "a"): [_kept(0, 7, k[:3], v[:3], 1)]}) == {"written_compared": 0, "written_mismatch": 0}
    assert AG.written({("batched", "a"): [grown[0], _kept(1, 0, k, v, 1)]}) == {"written_compared": 0, "written_mismatch": 0}


def _served(first: int, out_first: int) -> dict[str, torch.Tensor]:
    return {"q": _bf16(1, 2, 4, first=first), "k": _bf16(2, 1, 4, first=first + 100), "v": _bf16(2, 1, 4, first=first + 200),
            "out": _bf16(1, 2, 4, first=out_first)}


def test_batch_vs_solo_pairs_requests_by_their_inputs_and_counts_outputs_that_differ():
    got = {("batched", "a"): [_served(0, 10), _served(1, 11)],
           ("solo0", "a"): [_served(0, 10)],
           # the second prompt alone: the batched request's inputs with another output, and a request the batch did not hold
           ("solo1", "a"): [_served(1, 12), _served(2, 13)],
           ("solo0", "b"): [_served(0, 10)]}
    assert AG.batch_vs_solo(got, ["a", "b", "c"]) == {
        "a": [{"phase": "solo0", "requests": 1, "matched": 1, "mismatch": 0}, {"phase": "solo1", "requests": 2, "matched": 1, "mismatch": 1}],
        # layer b was not kept in the batched phase, so nothing of its solo run is matched
        "b": [{"phase": "solo0", "requests": 1, "matched": 0, "mismatch": 0}],
        "c": []}


class _Impl:
    """A stand-in for a served attention implementation: what `AttnTaps` reads of it, and a forward that counts its calls."""

    head_size = 2
    scale = 0.5

    def __init__(self) -> None:
        self.forwards = 0

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
        self.forwards += 1
        return output

    def k_scales(self, layer_name: str, device: torch.device) -> torch.Tensor:
        return torch.tensor([0.25 if layer_name.endswith("0") else 0.75])


class _Layer(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.layer_name = name
        self.impl = _Impl()


def _call(layer: _Layer, metadata: Any = "three requests") -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One forward of ``layer`` over three requests: two query rows, none, and one; the last has three tokens over two pages."""
    if metadata == "three requests":
        metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 2, 2, 3]), seq_lens=torch.tensor([2, 5, 3]),
                                   block_table=torch.tensor([[0, 4], [1, 4], [2, 3]]))
    given = {"query": _bf16(3, 2, 2), "key": _bf16(3, 1, 2, first=50), "value": _bf16(3, 1, 2, first=60),
             "kv_cache": _bf16(5, 1, 2, 4, first=100), "output": _bf16(3, 2, 2, first=70)}
    out = layer.impl.forward(layer, given["query"], given["key"], given["value"], given["kv_cache"], metadata, given["output"])
    return given, out


def test_the_taps_keep_each_requests_rows_window_and_output_and_put_the_forward_back():
    model = torch.nn.Module()
    first, last = _Layer("layers.0"), _Layer("layers.1")
    model.add_module("first", first)
    model.add_module("last", last)
    own = first.impl.forward
    taps = AG.AttnTaps(model, ["first"], keep=1)
    # no phase, or a call with no metadata: the layer's own forward and nothing kept
    _call(first)
    taps.phase = "batched"
    _call(first, metadata=None)
    assert first.impl.forwards == 2 and taps.got == {} and taps.calls == {}
    given, out = _call(first)
    assert out is given["output"] and first.impl.forwards == 3
    # an untapped layer keeps nothing, and neither does a second call of the phase beyond `keep`
    _call(last)
    _call(first)
    assert list(taps.got) == [("batched", "first")] and taps.calls == {("batched", "first"): 1}
    two_rows, one_row = taps.got["batched", "first"]
    assert {k: two_rows[k] for k in ("call", "request", "s", "n", "scale")} == {"call": 0, "request": 0, "s": 2, "n": 2, "scale": 0.5}
    assert {k: one_row[k] for k in ("call", "request", "s", "n", "scale")} == {"call": 0, "request": 2, "s": 3, "n": 1, "scale": 0.5}
    assert two_rows["k_scales"].tolist() == [0.25]
    assert torch.equal(two_rows["q"], given["query"][:2]) and torch.equal(one_row["q"], given["query"][2:])
    assert torch.equal(one_row["out"], given["output"][2:])
    assert torch.equal(one_row["handed"][0], given["key"][2:]) and torch.equal(one_row["handed"][1], given["value"][2:])
    # the cache's last dimension is a slot's K then its V; request 2's three tokens are page 2's two slots and page 3's first
    cache = given["kv_cache"].transpose(1, 2)
    assert torch.equal(one_row["k"], torch.stack([cache[2, 0, :, :2], cache[2, 1, :, :2], cache[3, 0, :, :2]]))
    assert torch.equal(one_row["v"], torch.stack([cache[2, 0, :, 2:], cache[2, 1, :, 2:], cache[3, 0, :, 2:]]))
    assert torch.equal(two_rows["v"], torch.stack([cache[0, 0, :, 2:], cache[0, 1, :, 2:]]))
    # each phase keeps its own first call
    taps.phase = "solo0"
    _call(first)
    assert [r["request"] for r in taps.got["solo0", "first"]] == [0, 2]
    taps.close()
    assert first.impl.forward == own
    _call(first)
    assert len(taps.got["solo0", "first"]) == 2 and first.impl.forwards == 6
