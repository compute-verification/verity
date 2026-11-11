"""Weights decoded at use (`store.decode_at_use`) on the CPU, under a fake codec: each module's registered weights decoded
into the shared scratch for its call and pointed back after, a tail past the registered blocks plain, the output the plain
model's bit for bit; the embedding's rows and the LM head's pieces through a scratch smaller than either."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.protocols.remote_bandwidth.decode_at_use import DecodeAtUse, EncodedWeight, encoded_weights
from verity_vllm.protocols.remote_bandwidth.interface import Layout, RegionInfo

BLOCK, WORD = 4096, 128


def _pad(first_word: int, n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.int64)
    word = first_word + i // WORD
    return ((word * 1315423911 + (i % WORD) * 2654435761) >> 11).remainder(256).to(torch.uint8)


def fake_decode(src: torch.Tensor, dst: torch.Tensor, first: int, stream: int) -> None:
    """XOR with a pad keyed by each word's index: its own inverse, so it encodes too, and any range decodes alone."""
    dst.copy_(src ^ _pad(first, src.numel()))


class Int8Linear(torch.nn.Module):
    """The int8 plain linear's two parameters (weight int8 [N, K], weight_scale f32 [N, 1]) read in its call."""

    def __init__(self, n: int, k: int, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(torch.rand(n, 1, generator=g) + 0.5, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight.float().T) * self.weight_scale.T


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a, self.b = Int8Linear(64, 96, 1), Int8Linear(1056, 64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


def word_decode(src: torch.Tensor, dst: torch.Tensor, first: int, stream: int) -> None:
    """`fake_decode` a word at a time, each under its own index: a range decodes the same in any split, as the codec does."""
    for w in range(src.numel() // WORD):
        dst[w * WORD:(w + 1) * WORD] = src[w * WORD:(w + 1) * WORD] ^ _pad(first + w, WORD)


def _register(model: torch.nn.Module, decode=fake_decode) -> list[EncodedWeight]:
    """Each parameter's whole blocks registered (a tail past them stays plain) and encoded in place, as the store does."""
    regions, first = [], 0
    for name, p in model.named_parameters():
        blocks = p.numel() * p.element_size() // BLOCK
        if not blocks:
            continue
        flat = p.data.view(-1).view(torch.uint8)[:blocks * BLOCK]
        decode(flat.clone(), flat, first, 0)
        regions.append(RegionInfo(name, "hbm", "weights", first // (BLOCK // WORD), blocks, BLOCK, first, WORD, "fake", "fake"))
        first += blocks * BLOCK // WORD
    return encoded_weights(Layout(tuple(regions), first // (BLOCK // WORD), {"hbm": first * WORD}), model)


def test_decoded_at_use_is_the_plain_model():
    model, x = Model(), torch.randn(5, 96, generator=torch.Generator().manual_seed(7))
    plain = model(x)
    weights = _register(model)
    assert sorted(w.name for w in weights) == ["a.weight", "b.weight", "b.weight_scale"]        # a's scale is under a block
    homes = {w.name: w.param.data.data_ptr() for w in weights}
    assert not torch.equal(model(x), plain)
    hooks = DecodeAtUse(model, weights, fake_decode).install()
    assert torch.equal(model(x), plain) and hooks.calls == 2
    assert all(w.param.data.data_ptr() == homes[w.name] for w in weights)        # pointed back at the store's bytes
    assert hooks.scratch.numel() == (17 + 2) * BLOCK     # b's weight and its scale, each rounded up to blocks
    hooks.remove()
    assert not torch.equal(model(x), plain)


def test_a_module_starting_inside_another_is_refused():
    model = Model()
    weights = _register(model)
    hooks = DecodeAtUse(model, weights, lambda *a: None)
    hooks.before(model.a, [w for w in weights if w.name.startswith("a.")])
    with pytest.raises(RuntimeError, match="starts while"):
        hooks.before(model.b, [w for w in weights if w.name.startswith("b.")])


def test_decoded_back_in_place_the_weights_are_the_models_own_again():
    model, x = Model(), torch.randn(5, 96, generator=torch.Generator().manual_seed(7))
    plain, before = model(x), {n: p.detach().clone() for n, p in model.named_parameters()}
    hooks = DecodeAtUse(model, _register(model, word_decode), word_decode).install()
    assert torch.equal(model(x), plain)
    with pytest.raises(RuntimeError, match="remove the hooks"):
        hooks.decode_in_place()
    hooks.remove()
    homes = {n: p.data_ptr() for n, p in model.named_parameters()}
    hooks.scratch = hooks.scratch[:3 * BLOCK]                   # a chunk smaller than b's weight: decoded in parts
    hooks.decode_in_place()
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == homes and torch.equal(model(x), plain)


def test_in_the_sessions_codec_a_weights_words_are_indexed_from_its_store_block():
    model = Model()
    regions = (RegionInfo("a.weight", "hbm", "weights", 0, 6, BLOCK, 0, 64, "identity", "r"),
               RegionInfo("b.weight", "hbm", "weights", 6, 16, BLOCK, 384, 64, "identity", "r"))
    lay = Layout(regions, 22, {"hbm": 22 * BLOCK})
    own, session = encoded_weights(lay, model), encoded_weights(lay, model, WORD)
    assert [(w.first_word, w.block_words, w.word_bytes) for w in own] == [(0, 64, 64), (384, 64, 64)]
    assert [(w.first_word, w.block_words, w.word_bytes) for w in session] == [(0, 32, WORD), (6 * 32, 32, WORD)]


V, H = 40, 2048                     # an embedding / LM head row: 2048 bf16, one block


class QuantMethod:
    """vLLM's unquantized embedding method: `embedding` gathers, `apply` is the linear."""

    def embedding(self, layer, ids):
        return torch.nn.functional.embedding(ids, layer.weight)

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


def _ints(shape, lo, hi, seed, dtype=torch.bfloat16):
    return torch.randint(lo, hi, shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


class Vocab(torch.nn.Module):
    """`VocabParallelEmbedding` / `ParallelLMHead` at tp 1 (small integers: every sum exact, in any order)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.tp_size, self.quant_method, self.ids = 1, QuantMethod(), []
        self.weight = torch.nn.Parameter(_ints((V, H), -8, 8, seed), requires_grad=False)

    def forward(self, ids):
        self.ids.append(ids)
        return self.quant_method.embedding(self, ids.long())


class Logits(torch.nn.Module):
    def forward(self, lm_head, h):
        assert lm_head.tp_size == 1
        return lm_head.quant_method.apply(lm_head, h)


class Lm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed, self.lm_head, self.logits_processor = Vocab(3), Vocab(4), Logits()
        self.a, self.b = Int8Linear(64, H, 5), Int8Linear(H, 64, 6)
        for m in (self.a, self.b):
            m.weight.data = _ints(m.weight.shape, -2, 3, 9, torch.int8)
            m.weight_scale.data = torch.ones_like(m.weight_scale)

    def forward(self, ids):
        h = self.b(self.a(self.embed(ids).float())).to(torch.bfloat16)
        return self.logits_processor(self.lm_head, h)


def test_the_embedding_and_lm_head_read_in_pieces_through_a_smaller_scratch():
    model, ids = Lm(), torch.tensor([7, 3, 7, 8, 39, 0, 9])
    plain = model(ids)
    weights = _register(model)
    assert sorted(w.name for w in weights) == ["a.weight", "b.weight", "b.weight_scale", "embed.weight", "lm_head.weight"]
    homes = {w.name: w.param.data.data_ptr() for w in weights}
    hooks = DecodeAtUse(model, weights, fake_decode).install()
    assert sorted(k for k, _, _ in hooks.pieces.values()) == ["head", "rows"]
    assert hooks.scratch.numel() == (32 + 2) * BLOCK < V * H * 2      # b's weight and scale, not the vocabulary's 40 blocks
    model.embed.ids.clear()
    seen = []
    acquired = model.embed.register_forward_pre_hook(lambda m, a: seen.append(a[0].clone()))
    qm = model.embed.quant_method
    assert torch.equal(model(ids).view(torch.int16), plain.view(torch.int16)) and hooks.calls == 4
    # the call as the model made it, to the module and to a hook after decode at use's (the acquisition's)
    assert model.embed.ids[0].tolist() == seen[0].tolist() == ids.tolist()
    assert model.embed.quant_method is qm
    acquired.remove()
    assert all(w.param.data.data_ptr() == homes[w.name] for w in weights)
    small = DecodeAtUse(model, weights, fake_decode, scratch_bytes=34 * BLOCK)
    with pytest.raises(ValueError, match="smaller than"):
        DecodeAtUse(model, weights, fake_decode, scratch_bytes=33 * BLOCK)
    hooks.remove()
    small.install()
    with pytest.raises(RuntimeError, match="do not fit"):
        model(torch.arange(35) % V)
    small.remove()
