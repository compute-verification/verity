"""``inference-decode``: KV-cache decode with fixed weights (``accumulation/algorithms/registry.py``).

Run with ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_registry_decode.py -q`` (a few seconds)."""

from __future__ import annotations

import time

import pytest

from accumulation.algorithms import registry as R
from accumulation.bounds.adw import adw
from accumulation.bounds.upper import check_plan, upper_bound
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.graph.validate import check_against_flat

F, X = 10 ** 12, 5_000_000


def hand_macs(cfg, wl) -> int:
    """per generated token: every layer's projections (2*D*AO + 2*D*KW), MLP (3*D*FF) and the two attention
    products against the S cached rows (2 * NH * S * DH), plus the LM head (V*D)."""
    per_layer = cfg.attn_params_per_layer() + cfg.mlp_params_dense_layer() + 2 * cfg.heads * wl.seq * cfg.head_dim
    return wl.tokens * (cfg.layers * per_layer + cfg.vocab * cfg.d)


def test_supports_and_defaults():
    alg = R.ALGORITHMS["inference-decode"]
    assert alg.family == "inference" and alg.simplifications
    assert alg.supports(MODELS["tiny-moe"], Workload(tokens=8, seq=8, gen_tokens=2)) is not None
    assert alg.supports(MODELS["tiny2"], Workload(tokens=9, seq=8, gen_tokens=2)) is not None   # 9 % 2 != 0
    assert alg.supports(MODELS["tiny2"], Workload(tokens=8, seq=8, gen_tokens=2)) is None
    assert R._decode_T(Workload(tokens=8, seq=4096)) == 512          # default T = seq // 8
    assert R._decode_T(Workload(tokens=8, seq=4)) == 1               # ... but at least 1
    bp = R.build("inference-decode", MODELS["tiny2"], Workload(tokens=8, seq=8, gen_tokens=2))
    assert bp.notes["gen_tokens_per_seq"] == 2 and bp.notes["sequences"] == 4


@pytest.mark.parametrize("wl", [Workload(tokens=8, seq=8, gen_tokens=2), Workload(tokens=16, seq=4, gen_tokens=4)])
def test_tiny2_flat_macs_roles_upper(wl):
    cfg = MODELS["tiny2"]
    bp = R.build("inference-decode", cfg, wl)
    g = extract(bp)
    r = check_against_flat(bp, g)                                      # gate-for-gate quotient check
    assert r["ops"] > 0 and r["edges_exact_equal"] > 0
    a = adw(g)
    assert a.all_macs == hand_macs(cfg, wl)
    assert a.total == 0                                                # fixed weights: no accumulated state
    assert set(bp.roles.values()) == {"fixed", "token"}
    assert bp.roles["K_cache"] == bp.roles["V_cache"] == bp.roles["toks"] == "token"
    T = bp.notes["gen_tokens_per_seq"]
    cache = 2 * cfg.layers * (wl.tokens // T) * wl.seq * cfg.kv_width * 2
    assert bp.notes["kv_cache_bytes"] == cache
    assert bp.param_bytes("token") == 4 * wl.tokens + cache
    # the cache append is emitted: logits + per-layer new K / V rows
    assert len(g.outputs) >= 3
    # constructive upper bound: legal at the headline point and at tiny X.  The fused attention units are only
    # sound for square attention (queries == keys), so upper.py fuses decode attention iff T == S and otherwise
    # falls back to per-op tiles.
    ub = upper_bound(g, F, X, program=bp.program)
    check_plan(ub.plan, g, F, X)
    assert any("attention fusion" in n for n in ub.notes) == (T == wl.seq)
    for X2 in (64, 256, 4096):
        ub2 = upper_bound(g, 1 << 12, X2, program=bp.program)
        check_plan(ub2.plan, g, 1 << 12, X2)
        assert ub2.total >= ub.total


def test_llama8b_decode_extract_and_upper():
    cfg = MODELS["llama3-8b"]
    wl = Workload(tokens=1024, seq=4096, gen_tokens=256)
    t0 = time.time()
    bp = R.build("inference-decode", cfg, wl)
    g = extract(bp)
    a = adw(g)
    assert a.all_macs == hand_macs(cfg, wl)
    # per generated token the MACs equal prefill's per-token MACs at the same S (full S x S attention there,
    # T x S against the cache here): L * (weights + 2*NH*S*DH) + V*D
    assert a.all_macs // wl.tokens == cfg.layers * (cfg.attn_params_per_layer() + cfg.mlp_params_dense_layer()
                                                    + 2 * cfg.heads * wl.seq * cfg.head_dim) + cfg.vocab * cfg.d
    ub = upper_bound(g, F, X, program=bp.program)
    check_plan(ub.plan, g, F, X)
    assert time.time() - t0 < 25
    assert not any("attention fusion" in n for n in ub.notes)          # non-square attention: no fused units
    assert sum(1 for u in ub.plan.units if u.kind == "attn-fwd") == 0
    # the KV cache alone is 2 GB of token-role input; U is per generated token
    assert bp.param_bytes("token") >= bp.notes["kv_cache_bytes"] == 2 * cfg.layers * 4 * 4096 * cfg.kv_width * 2
    assert 0 < ub.total / wl.tokens < 50e6
