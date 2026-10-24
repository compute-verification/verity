"""`engine.int8_experts`: the exact-int8 routed experts with no vLLM.

Each expert GEMM is the dense plain linear's exact op (equal to `int8_exact.plain_linear` and to the perplexity bench's
`steps.int8_linear`, `tests/engine/ppl_oracle.py`, on the expert's routed rows), the whole layer equals the module doc's
definitions written out in NumPy (`int8_experts.reference`, one token at a time), every (expert, projection) is called on every
forward in the static order, an expert no token routed to on zero rows, and each call's rows are its tokens, ascending."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.engine import int8_checkpoint as I  # noqa: E402
from verity_vllm.engine import int8_exact as E  # noqa: E402
from verity_vllm.engine import int8_experts as X  # noqa: E402

from tests.engine import ppl_oracle as O  # noqa: E402

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
