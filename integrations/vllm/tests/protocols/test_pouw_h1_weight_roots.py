"""`pouw_h1_hooks.weight_roots`: each window's install reuses a linear's root while its weight is the tensor the root was
computed from, and computes it again when the weight was replaced or written."""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
PH = pytest.importorskip("verity_vllm.protocols.pouw.h1")
PHK = pytest.importorskip("verity_vllm.protocols.pouw.h1.hooks")

FRESH = PH.weight_root


def _layers(seed: int, n: int = 3, shape: tuple[int, int] = (64, 32)) -> list[types.SimpleNamespace]:
    g = torch.Generator().manual_seed(seed)
    return [types.SimpleNamespace(weight=torch.nn.Parameter(torch.randint(-128, 128, shape, dtype=torch.int8, generator=g),
                                                            requires_grad=False)) for _ in range(n)]


def _fresh(layers: list[types.SimpleNamespace], lo: int = 0) -> dict[int, bytes]:
    return {lo + i: FRESH(lo + i, m.weight.detach().numpy()) for i, m in enumerate(layers)}


@pytest.fixture
def computed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The linears whose root was computed, not reused, in call order, from an empty cache."""
    monkeypatch.setattr(PHK, "_ROOTS", {})
    seen: list[int] = []
    monkeypatch.setattr(PH, "weight_root", lambda w, rows: seen.append(w) or FRESH(w, rows))
    return seen


def test_a_reused_root_is_the_fresh_one_and_is_computed_once(computed):
    layers = _layers(1)
    assert PHK.weight_roots(layers) == PHK.weight_roots(layers) == _fresh(layers)
    assert computed == [0, 1, 2]


def test_a_rank_s_roots_are_by_global_index(computed):
    layers = _layers(2, n=2)
    assert PHK.weight_roots(layers, lo=5) == PHK.weight_roots(layers, lo=5) == _fresh(layers, lo=5)
    assert computed == [5, 6]


@pytest.mark.parametrize("change", ["write in place", "write through a view", "replace", "swap data"])
def test_a_changed_weight_is_computed_again(computed, change):
    layers = _layers(3)
    before = PHK.weight_roots(layers)
    w = layers[1].weight
    if change == "write in place":
        with torch.no_grad():
            w[0, 0] = int(w[0, 0]) ^ 1
    elif change == "write through a view":
        with torch.no_grad():
            w.view(-1)[7] = int(w.view(-1)[7]) ^ 1
    elif change == "replace":
        layers[1].weight = torch.nn.Parameter(w.detach().clone(), requires_grad=False)
    else:
        w.data = w.detach().clone()
    after = PHK.weight_roots(layers)
    assert after == _fresh(layers)
    assert computed == [0, 1, 2, 1]
    assert (after[1] != before[1]) == change.startswith("write")
    assert {k: after[k] for k in (0, 2)} == {k: before[k] for k in (0, 2)}


def test_another_model_s_layer_at_the_same_index_is_computed_again(computed):
    first, second = _layers(4, n=2), _layers(5, n=2)
    PHK.weight_roots(first)
    assert PHK.weight_roots(second) == _fresh(second)
    assert computed == [0, 1, 0, 1]


def test_an_array_that_isn_t_a_tensor_is_computed_every_time(computed):
    layers = [types.SimpleNamespace(weight=np.arange(64 * 32, dtype=np.int64).reshape(64, 32).astype(np.int8))]
    assert PHK.weight_roots(layers) == PHK.weight_roots(layers) == {0: FRESH(0, layers[0].weight)}
    assert computed == [0, 0]
