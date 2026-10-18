"""Weights decoded at use (`store.decode_at_use`) on the CPU, under a fake codec: each module's registered weights decoded
into the shared scratch for its call and pointed back after, a tail past the registered blocks plain, the output the plain
model's bit for bit."""
from __future__ import annotations

import pytest
import torch

from verity_vllm.store.decode_at_use import DecodeAtUse, EncodedWeight, encoded_weights
from verity_vllm.store.interface import Layout, RegionInfo

BLOCK, WORD = 4096, 128


def _pad(first_word: int, n: int) -> torch.Tensor:
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=torch.Generator().manual_seed(first_word))


def fake_decode(src: torch.Tensor, dst: torch.Tensor, first: int, stream: int) -> None:
    """XOR with a pad keyed by the first word's index: its own inverse, so it encodes too."""
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


def _register(model: Model) -> list[EncodedWeight]:
    """Each parameter's whole blocks registered (a tail past them stays plain) and encoded in place, as the store does."""
    regions, first = [], 0
    for name, p in model.named_parameters():
        blocks = p.numel() * p.element_size() // BLOCK
        if not blocks:
            continue
        flat = p.data.view(-1).view(torch.uint8)[:blocks * BLOCK]
        fake_decode(flat.clone(), flat, first, 0)
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
