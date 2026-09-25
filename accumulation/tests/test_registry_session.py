"""``inference-session``: prefill + KV-cache decode in one circuit, fixed weights, token ids the only runtime input
(``accumulation/algorithms/registry.py``).

Run with ``PYTHONPATH=. .venv/bin/pytest accumulation/tests/test_registry_session.py -q`` (~15 s, dominated by the
llama3-8b extraction)."""

from __future__ import annotations

import resource
import time

import pytest

from accumulation.algorithms import registry as R
from accumulation.bounds.adw import adw
from accumulation.bounds.upper import check_plan, upper_bound
from accumulation.configs import MODELS, Workload
from accumulation.graph import extract
from accumulation.graph.validate import check_against_flat


def hand_macs(cfg, wl) -> int:
    """per sequence: every layer's weights on all P+T rows, attention P x P (prefill) + T x P (decode) per head
    for both products, LM head on all rows."""
    P, T = R._session_PT(wl)
    w = cfg.attn_params_per_layer() + cfg.mlp_params_dense_layer()
    attn = 2 * cfg.heads * cfg.head_dim * (P * P + T * P)
    return (wl.tokens // wl.seq) * (cfg.layers * (w * (P + T) + attn) + cfg.vocab * cfg.d * (P + T))


def dense_macs(cfg, Q: int, S: int) -> int:
    """``inference-dense`` MACs at ``Q`` tokens, sequence ``S`` (full S x S attention)."""
    return Q * (cfg.layers * (cfg.attn_params_per_layer() + cfg.mlp_params_dense_layer() + 2 * cfg.heads * cfg.head_dim * S)
                + cfg.vocab * cfg.d)


def test_supports_and_split():
    alg = R.ALGORITHMS["inference-session"]
    assert alg.family == "inference" and alg.simplifications
    assert alg.supports(MODELS["tiny-moe"], Workload(tokens=8, seq=4)) is not None
    assert alg.supports(MODELS["tiny2"], Workload(tokens=6, seq=4)) is not None          # not whole sessions
    assert alg.supports(MODELS["tiny2"], Workload(tokens=8, seq=4, gen_tokens=4)) is not None   # no prompt left
    assert alg.supports(MODELS["tiny2"], Workload(tokens=8, seq=4)) is None
    assert R._session_PT(Workload(tokens=8192, seq=8192)) == (7168, 1024)                # default T = seq // 8
    assert R._session_PT(Workload(tokens=8, seq=4)) == (3, 1)


@pytest.mark.parametrize("wl", [Workload(tokens=16, seq=8, gen_tokens=2), Workload(tokens=8, seq=4),
                                Workload(tokens=12, seq=6, gen_tokens=3)])
def test_tiny2_flat_macs_roles_upper(wl):
    cfg = MODELS["tiny2"]
    bp = R.build("inference-session", cfg, wl)
    g = extract(bp)
    r = check_against_flat(bp, g)                                      # gate-for-gate quotient check
    assert r["ops"] > 0 and r["edges_exact_equal"] > 0
    a = adw(g)
    assert a.all_macs == hand_macs(cfg, wl)
    assert a.total == 0                                                # fixed weights: no accumulated state
    assert set(bp.roles.values()) == {"fixed", "token"}
    assert [t.name for t in g.tensors if t.kind == "param" and t.role == "token"] == ["toks"]
    assert bp.param_bytes("token") == 4 * wl.tokens                    # the KV cache is internal, not an input
    P, T = R._session_PT(wl)
    assert bp.notes["prompt_tokens_per_seq"] == P and bp.notes["gen_tokens_per_seq"] == T
    # the session does strictly less attention than the dense forward at Q = P + T: nobody attends over the T new
    # keys (prompt rows: P*T, new rows: T*T), i.e. T*(P+T) fewer key-query pairs per head and product
    assert a.all_macs < dense_macs(cfg, wl.tokens, wl.seq)
    assert dense_macs(cfg, wl.tokens, wl.seq) - a.all_macs == (wl.tokens // wl.seq) * cfg.layers * 2 * cfg.heads * cfg.head_dim * T * (P + T)
    for F, X in ((10 ** 12, 5_000_000), (1 << 12, 64), (1 << 12, 4096)):
        ub = upper_bound(g, F, X, program=bp.program)
        check_plan(ub.plan, g, F, X)
    # F, X unbounded: the whole session is one RU whose input is the token ids
    ub = upper_bound(g, 1 << 60, 1 << 60, program=bp.program)
    check_plan(ub.plan, g, 1 << 60, 1 << 60)
    assert ub.n_units == 1 and ub.total == 4 * wl.tokens


def test_llama8b_session_extract_work_and_one_ru():
    cfg = MODELS["llama3-8b"]
    P, T, nseq = 7168, 1024, 4
    wl = Workload(tokens=nseq * (P + T), seq=P + T, gen_tokens=T)
    t0 = time.time()
    bp = R.build("inference-session", cfg, wl)
    g = extract(bp)
    dt = time.time() - t0
    assert dt < 30, dt
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 3 * 1024 ** 3
    a = adw(g)
    assert a.all_macs == hand_macs(cfg, wl)
    assert [t.name for t in g.tensors if t.kind == "param" and t.role == "token"] == ["toks"]
    assert bp.param_bytes("token") == 4 * wl.tokens
    # whole-circuit sanity for the coarse (one-RU) planner: total work and the largest op-level upstream work are
    # both below fwd(Q_inf) = inference-dense MACs at Q_inf = P + T (the gap is the attention simplification)
    fwd = dense_macs(cfg, wl.tokens, P + T)
    total_work = g.total_work()
    up = max(g.tensor_ancestor_work(o.out) for o in g.ops)
    print(f"\nllama3-8b session P={P} T={T} x{nseq}: extract {dt:.1f}s, ops={len(g.ops)}, all_macs={a.all_macs:.4e}, "
          f"total work={total_work:.4e} ({total_work / fwd:.4f} fwd), max Up={up:.4e} ({up / fwd:.4f} fwd), fwd={fwd:.4e}")
    assert a.all_macs < total_work <= fwd
    assert up <= total_work
    assert fwd - a.all_macs == nseq * cfg.layers * 2 * cfg.heads * cfg.head_dim * T * (P + T)
    # with F, X unbounded the session is one RU and U is exactly the token ids
    ub = upper_bound(g, 1 << 60, 1 << 60, program=bp.program)
    check_plan(ub.plan, g, 1 << 60, 1 << 60)
    assert ub.n_units == 1 and ub.total == 4 * wl.tokens
    assert any("attention fusion: 32 forward" in n for n in ub.notes)   # prefill attention is square -> fused
