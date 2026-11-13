"""`engine.int8_experts`: the exact-int8 routed experts with no vLLM.

Each expert GEMM is the dense plain linear's exact op (equal to `int8_exact.plain_linear` and to the perplexity bench's
`steps.int8_linear`, `tests/engine/ppl_oracle.py`, on the expert's routed rows), the whole layer equals the module doc's
definitions written out in NumPy (`int8_experts.reference`, one token at a time), every (expert, projection) is called on every
forward in the static order, an expert no token routed to on zero rows, and each call's rows are its tokens, ascending.

And the base pin's weights as the int8 GPU scripts read them: `tests.engine.int8_exact_gpu.BaseWeights` (a snapshot's safetensors
by tensor name, through `verity_vllm.engine.fp8_checkpoint.read_header`) and `tests.engine.int8_experts_gpu.ExpertBase` (an
expert's projection of `verity_vllm.engine.int8_experts`).  The snapshot is two safetensors files written here; no GPU and no
model."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")

from tests.engine import int8_exact_gpu as G
from tests.engine import int8_experts_gpu as XG
from tests.engine import ppl_oracle as O
from verity_vllm.engine import int8_checkpoint as I
from verity_vllm.engine import int8_exact as E
from verity_vllm.engine import int8_experts as X

H, INTER, EXPERTS = 64, 32, 6


def _bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(torch.bfloat16)


def _int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The int8 recipe's (W, s_w [N, 1]) of a bf16 weight [N, K]."""
    w32 = w.float().numpy()
    s = I.channel_scales(w32)
    return torch.from_numpy(I.quantize(w32, s)), torch.from_numpy(s)


def _layer(seed: int):
    """E experts' bf16 weights, gate_up [E, 2I, H] and down [E, H, I], and their int8 (w13, s13, w2, s2)."""
    rng = np.random.default_rng(seed)
    w13b = _bf16(rng.standard_normal((EXPERTS, 2 * INTER, H), dtype=np.float32) * np.float32(0.05))
    w2b = _bf16(rng.standard_normal((EXPERTS, H, INTER), dtype=np.float32) * np.float32(0.05))
    q13, q2 = [_int8(w13b[e]) for e in range(EXPERTS)], [_int8(w2b[e]) for e in range(EXPERTS)]
    return (w13b, w2b), (torch.stack([q for q, _ in q13]), torch.stack([s for _, s in q13]),
                         torch.stack([q for q, _ in q2]), torch.stack([s for _, s in q2]))


def _routing(rng: np.random.Generator, m: int, k: int, idle: tuple[int, ...] = ()) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k ids [m, k] (distinct per token, never an `idle` expert, in no particular order) and softmax-like weights."""
    live = [e for e in range(EXPERTS) if e not in idle]
    ids = np.stack([rng.permutation(live)[:k] for _ in range(m)]).astype(np.int32)
    w = rng.uniform(0.05, 1, (m, k)).astype(np.float32)
    return torch.from_numpy(w / w.sum(axis=1, keepdims=True)), torch.from_numpy(ids)


class _Tap:
    """Every call each site's ``observe`` gets: (expert, projection, tokens, xq, s_a, y, z)."""

    def __init__(self, sites: list[X.ExpertSite]):
        self.calls: list[tuple] = []
        for site in sites:
            site.quant_method.observe = self._observer(site)

    def _observer(self, site: X.ExpertSite):
        def observe(layer, xq, s_a, y, z=None):
            assert layer is site
            self.calls.append((site.expert, site.projection, site.tokens.clone(), xq, s_a, y, z))
        return observe


def _words(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy()


@pytest.mark.parametrize("m, k, scale", [(1, 2, 1.0), (7, 2, 1.0), (16, 3, 30.0), (40, 2, 1e-3)])
def test_forward_equals_the_numpy_definition(m, k, scale):
    _, (w13, s13, w2, s2) = _layer(10 + m)
    rng = np.random.default_rng(m * 100 + k)
    x = _bf16(rng.standard_normal((m, H), dtype=np.float32) * np.float32(scale))
    tw, ids = _routing(rng, m, k, idle=(4,))
    got = X.forward(X.sites("moe", w13, s13, w2, s2), x, tw, ids)
    want = X.reference(x.float().numpy(), tw.numpy(), ids.numpy(), w13.numpy(), s13.numpy(), w2.numpy(), s2.numpy())
    assert got.dtype == torch.bfloat16 and got.shape == (m, H)
    bad = np.argwhere(_words(got) != _words(_bf16(want)))
    assert bad.size == 0, f"{len(bad)} words differ, first at {bad[0].tolist()}"


@pytest.mark.slow
def test_each_expert_gemm_is_the_dense_plain_linear_and_steps_py():
    (w13b, w2b), (w13, s13, w2, s2) = _layer(3)
    rng = np.random.default_rng(4)
    x = _bf16(rng.standard_normal((24, H), dtype=np.float32))
    tw, ids = _routing(rng, 24, 2)
    sites = X.sites("moe", w13, s13, w2, s2)
    tap = _Tap(sites)
    X.forward(sites, x, tw, ids)
    downs = {e: (xq, s_a) for e, p, _, xq, s_a, _, _ in tap.calls if p == X.DOWN}
    for e, proj, tokens, xq, s_a, y, z in tap.calls:
        assert z is None
        if proj == X.GATE_UP:
            a, wb, (wq, sw) = x.index_select(0, tokens), w13b[e], (w13[e], s13[e])
            assert torch.equal(_words_t(y), _words_t(E.plain_linear(a, wq, sw)))
            assert torch.equal(_words_t(y), _words_t(O.int8_linear(a, wb)))
            assert torch.equal(xq, E.quantize_rows(a)[0]) and torch.equal(s_a, E.quantize_rows(a)[1])
        else:
            act = X.silu_and_mul(E.plain_linear(x.index_select(0, tokens), w13[e], s13[e]))
            assert torch.equal(_words_t(y), _words_t(E.plain_linear(act, w2[e], s2[e])))
            assert torch.equal(_words_t(y), _words_t(O.int8_linear(act, w2b[e])))
            assert torch.equal(downs[e][0], E.quantize_rows(act)[0]) and torch.equal(downs[e][1], E.quantize_rows(act)[1])


def _words_t(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def test_every_site_is_called_in_static_order_with_its_tokens_ascending():
    _, (w13, s13, w2, s2) = _layer(5)
    rng = np.random.default_rng(6)
    sites = X.sites("model.layers.3.mlp.experts", w13, s13, w2, s2)
    assert [(s.expert, s.projection) for s in sites] == [(e, p) for e in range(EXPERTS) for p in X.PROJECTIONS]
    assert sites[3].name == "model.layers.3.mlp.experts.1.down"
    assert sites[0].weight.data_ptr() == w13[0].data_ptr() and sites[1].weight.data_ptr() == w2[0].data_ptr()
    tap = _Tap(sites)
    for step, m in enumerate((9, 1, 0, 5)):
        x = _bf16(rng.standard_normal((m, H), dtype=np.float32))
        tw, ids = _routing(rng, m, 2, idle=(2,)) if m else (torch.zeros(0, 2), torch.zeros(0, 2, dtype=torch.int32))
        out = X.forward(sites, x, tw, ids)
        assert out.shape == (m, H)
        calls, tap.calls = tap.calls, []
        assert [(e, p) for e, p, *_ in calls] == [(s.expert, s.projection) for s in sites]
        for e, proj, tokens, xq, s_a, y, _ in calls:
            want = sorted(t for t in range(m) if e in ids[t].tolist())
            assert tokens.tolist() == want
            n, kk = (2 * INTER, H) if proj == X.GATE_UP else (H, INTER)
            assert xq.shape == (len(want), kk) and xq.dtype == torch.int8 and s_a.shape == (len(want),)
            assert y.shape == (len(want), n) and y.dtype == torch.bfloat16
        assert all(len(c[2]) == 0 for c in calls if c[0] == 2)                 # the idle expert, on zero rows each step
    assert all(s.quant_method.calls == 4 for s in sites)


def test_a_tokens_output_does_not_depend_on_its_batch():
    _, (w13, s13, w2, s2) = _layer(7)
    rng = np.random.default_rng(8)
    x = _bf16(rng.standard_normal((12, H), dtype=np.float32))
    x[3] *= 1000
    tw, ids = _routing(rng, 12, 2)
    sites = X.sites("moe", w13, s13, w2, s2)
    batched = X.forward(sites, x, tw, ids)
    for t in range(12):
        alone = X.forward(sites, x[t:t + 1], tw[t:t + 1], ids[t:t + 1])
        assert torch.equal(_words_t(alone), _words_t(batched[t:t + 1]))


def test_zy_rows_hand_the_records_to_observe():
    _, (w13, s13, w2, s2) = _layer(9)
    rng = np.random.default_rng(10)
    sites = X.sites("moe", w13, s13, w2, s2)
    for s in sites:
        s.quant_method.zy_rows = 32
    tap = _Tap(sites)
    x = _bf16(rng.standard_normal((5, H), dtype=np.float32))
    tw, ids = _routing(rng, 5, 2)
    X.forward(sites, x, tw, ids)
    for e, proj, tokens, xq, s_a, y, z in tap.calls:
        site = sites[2 * e + (proj == X.DOWN)]
        if len(tokens):
            want_y, want_zy = E.scaled_mm_zy(xq, s_a, site.weight, site.weight_scale, 32)
            assert torch.equal(_words_t(y), _words_t(want_y)) and torch.equal(z, want_zy)
        else:
            assert z is None


def test_gemm_serves_in_place_of_scaled_mm():
    """With ``gemm`` set, a call's rows are ``row_quant``'s and its y is ``gemm``'s, handed to ``observe`` and served; a
    zero-row call reaches ``observe`` and never ``gemm``. A ``gemm`` that serves the plain product leaves the layer's
    output bit for bit the plain arm's."""
    _, (w13, s13, w2, s2) = _layer(11)
    rng = np.random.default_rng(12)
    x = _bf16(rng.standard_normal((6, H), dtype=np.float32))
    tw, ids = _routing(rng, 6, 2, idle=(3,))
    plain = X.forward(X.sites("moe", w13, s13, w2, s2), x, tw, ids)
    sites = X.sites("moe", w13, s13, w2, s2)
    seen = []

    def gemm(layer, xq, s_a):
        seen.append((layer, xq, s_a))
        return E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)
    for s in sites:
        s.quant_method.gemm = gemm
    tap = _Tap(sites)
    assert torch.equal(_words_t(X.forward(sites, x, tw, ids)), _words_t(plain))
    live = [c for c in tap.calls if len(c[2])]
    assert len(seen) == len(live) <= 2 * (EXPERTS - 1) and len(tap.calls) == 2 * EXPERTS
    for (layer, xq, s_a), (e, proj, _tokens, oxq, os_a, y, z) in zip(seen, live):
        assert layer is sites[2 * e + (proj == X.DOWN)] and xq is oxq and s_a is os_a and z is None
        assert torch.equal(_words_t(y), _words_t(E.dequantize(E.int_mm(xq, layer.weight), s_a, layer.weight_scale)))


def test_an_unrouted_slot_goes_to_no_expert():
    """vLLM's id -1 (its profile run routes every slot so): the slot adds nothing, the token's other slots are as routed, and
    a token with every slot unrouted is zero."""
    _, (w13, s13, w2, s2) = _layer(12)
    rng = np.random.default_rng(13)
    x = _bf16(rng.standard_normal((4, H), dtype=np.float32))
    tw, ids = _routing(rng, 4, 2)
    ids[1, 0] = ids[2, 1] = X.UNROUTED
    ids[3] = X.UNROUTED
    sites = X.sites("moe", w13, s13, w2, s2)
    tap = _Tap(sites)
    got = X.forward(sites, x, tw, ids)
    want = X.reference(x.float().numpy(), tw.numpy(), ids.numpy(), w13.numpy(), s13.numpy(), w2.numpy(), s2.numpy())
    assert np.array_equal(_words(got), _words(_bf16(want))) and not got[3].any()
    assert sum(len(c[2]) for c in tap.calls if c[1] == X.GATE_UP) == 8 - 4 and len(tap.calls) == 2 * EXPERTS


@pytest.mark.parametrize("ids, why", [([[0, 0]], "routed to one expert twice"), ([[0, EXPERTS]], "outside the layer's"),
                                      ([[-2, 1]], "outside the layer's")])
def test_routing_refuses_what_it_does_not_define(ids, why):
    _, (w13, s13, w2, s2) = _layer(11)
    x = torch.zeros(1, H, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=why):
        X.forward(X.sites("moe", w13, s13, w2, s2), x, torch.full((1, 2), 0.5), torch.tensor(ids, dtype=torch.int32))


def test_sites_refuse_mismatched_experts():
    with pytest.raises(ValueError, match="not E gate_up"):
        X.sites("moe", torch.zeros(2, 64, 32, dtype=torch.int8), torch.ones(2, 64, 1), torch.zeros(2, 32, 16, dtype=torch.int8),
                torch.ones(2, 32, 1))


# ------------------------------------------------------------- the base pin's weights, as the int8 GPU scripts read them

BASE_EXPERTS = "model.layers.3.mlp.experts"


def _shard(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """A safetensors file: the header's length, the header, the tensors' bytes in order."""
    header: dict[str, dict] = {"__metadata__": {"format": "pt"}}
    blob = b""
    for name, values in tensors.items():
        header[name] = {"dtype": "F32" if values.dtype == np.float32 else "BF16", "shape": list(values.shape),
                        "data_offsets": [len(blob), len(blob) + values.nbytes]}
        blob += values.tobytes()
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + blob)


def _base_words(first: int, rows: int = 2, cols: int = 3) -> np.ndarray:
    """``rows`` x ``cols`` bfloat16 words counting up from ``first``."""
    return np.array(range(first, first + rows * cols), np.int16).reshape(rows, cols)


@pytest.fixture
def base_snapshot(tmp_path) -> Path:
    _shard(tmp_path / "model-00001-of-00002.safetensors", {
        f"{BASE_EXPERTS}.0.gate_proj.weight": _base_words(100), f"{BASE_EXPERTS}.0.up_proj.weight": _base_words(200),
        f"{BASE_EXPERTS}.0.down_proj.weight": _base_words(300, 3, 2), f"{BASE_EXPERTS}.1.gate_proj.weight": _base_words(400),
        "model.layers.3.self_attn.q_proj.weight": _base_words(500, 1), "model.layers.3.self_attn.k_proj.weight": _base_words(600, 1)})
    _shard(tmp_path / "model-00002-of-00002.safetensors", {
        f"{BASE_EXPERTS}.1.up_proj.weight": _base_words(700), "model.layers.3.self_attn.v_proj.weight": _base_words(800, 1),
        "model.layers.13.mlp.experts.0.down_proj.weight": _base_words(900, 3, 2), "model.norm.weight": np.array([1.5, -2.0], np.float32)})
    return tmp_path


def _stored(t: torch.Tensor) -> list:
    return t.view(torch.int16).tolist()


def test_base_weights_reads_a_tensor_as_stored_from_whichever_file_holds_it(base_snapshot):
    base = G.BaseWeights(str(base_snapshot))
    assert len(base.where) == 10 and "__metadata__" not in base.where
    assert base.where["model.norm.weight"].name == "model-00002-of-00002.safetensors"
    got = base.tensor(f"{BASE_EXPERTS}.0.up_proj.weight")
    assert got.dtype == torch.bfloat16 and _stored(got) == [[200, 201, 202], [203, 204, 205]]
    norm = base.tensor("model.norm.weight")
    assert norm.dtype == torch.float32 and norm.tolist() == [1.5, -2.0]
    with pytest.raises(KeyError):
        base.tensor("model.layers.3.self_attn.o_proj.weight")


def test_base_weights_joins_a_merged_modules_parts_by_rows(base_snapshot):
    base = G.BaseWeights(str(base_snapshot))
    # q and k are in the first file and v in the second: vLLM's qkv_proj holds the three, in that order
    assert _stored(base.weight("model.layers.3.self_attn.qkv_proj")) == [[500, 501, 502], [600, 601, 602], [800, 801, 802]]
    assert _stored(base.weight(f"{BASE_EXPERTS}.0.down_proj")) == [[300, 301], [302, 303], [304, 305]]


def test_expert_base_names_an_experts_projection_by_layer_expert_and_part(base_snapshot):
    base = XG.ExpertBase(G.BaseWeights(str(base_snapshot)))
    assert _stored(base.weight(BASE_EXPERTS, 0, X.GATE_UP)) == [[100, 101, 102], [103, 104, 105], [200, 201, 202], [203, 204, 205]]
    # expert 1's two parts are in two files
    assert _stored(base.weight(BASE_EXPERTS, 1, X.GATE_UP)) == [[400, 401, 402], [403, 404, 405], [700, 701, 702], [703, 704, 705]]
    # layer 3's expert 0, not layer 13's
    assert _stored(base.weight(BASE_EXPERTS, 0, X.DOWN)) == [[300, 301], [302, 303], [304, 305]]
    with pytest.raises(KeyError, match="layer 3 expert 1 down_proj: 0 base tensors"):
        base.weight(BASE_EXPERTS, 1, X.DOWN)
    with pytest.raises(KeyError, match="layer 13 expert 0 gate_proj: 0 base tensors"):
        base.weight("model.layers.13.mlp.experts", 0, X.GATE_UP)
    # a module path with no layer in it names no expert
    with pytest.raises(AssertionError, match="model.mlp.experts"):
        base.weight("model.mlp.experts", 0, X.DOWN)
