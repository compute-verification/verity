"""Lane HP: the FlashAttention-3 CPU model (``verity_vllm_numerics.fa3_model``) against the typed composite
``AttentionHeadFA3@1`` (``verity_vllm.registry.hopper``: TA1's ``AttentionHead`` body in the Hopper binding
environment), evaluated gate by gate by the IR oracle, on random finite operands.

Two independent spellings of one function — the C++ per-row recurrence and the one-output-gate circuit — must
agree word for word; where they would disagree with FA2 (kBlockN 128 vs 64, the wgmma k16 grouping) the FA2
model must differ from both on the same operands, so the test also shows the FA3 binding is not FA2's.

Correspondence of ``AttentionFA3@1`` with the H100 production capture is the separate correspondence run
(results/CONF-b1-hopper-*/correspondence.json); this test is the definition-level conformance evidence.

Needs g++ (or $CXX) for ``cpp/fa2_model.cpp`` and Python >= 3.12 for ``veritor.core`` (the Hopper total function).
Run: ``cd vllm-poc && python -m pytest tests/test_fa3_model.py -q``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT.parent, ROOT.parent / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

FIX = ROOT / "fixtures" / "W11-40f0cebeb670-20260907T1800Z"
os.environ.setdefault("VERITY_MUFU_TABLES", str(FIX / "mufu"))

pytestmark = pytest.mark.skipif(not (FIX / "mufu").exists(), reason="MUFU table fixture missing")

if sys.version_info < (3, 12):
    pytest.skip("veritor.core (the Hopper total function) needs Python 3.12", allow_module_level=True)

from verity_vllm_numerics import fa2_model as FA2  # noqa: E402
from verity_vllm_numerics import fa3_model as FA3  # noqa: E402


def _have_cpp() -> bool:
    try:
        FA3.lib()
        return True
    except Exception:  # noqa: BLE001
        return False


needs_cpp = pytest.mark.skipif(not _have_cpp(), reason="cpp/fa2_model.cpp could not be built (no g++)")


@pytest.fixture(scope="module")
def tables() -> FA2.MufuTables:
    return FA2.MufuTables.load(FIX / "mufu")


def _rand_bf16(rng, n, scale=1.0):
    x = (rng.standard_normal(n) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _head(rng, T, D, tables, *, model):
    from verity.ir.defs import bind
    from verity.ir.evaluate import evaluate_call
    from verity_vllm.registry import hopper as H
    from verity_vllm.registry.crosscheck import _standalone

    prog, call = _standalone(bind(H.AttentionHeadFA3, T=T, D=D))
    q = _rand_bf16(rng, D, scale=float(rng.choice([0.3, 1.0, 3.0])))
    k = _rand_bf16(rng, T * D, scale=float(rng.choice([0.3, 1.0, 3.0]))).reshape(T, D)
    v = _rand_bf16(rng, T * D).reshape(T, D)
    rr = model.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables)
    ref = np.asarray(rr.out).reshape(-1)[:D]
    args = [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()]
    out = np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint16)
    return out, ref, rr


@needs_cpp
@pytest.mark.parametrize("T", [1, 5, 64, 65, 128, 129, 200, 257])
def test_attention_head_fa3_equals_cpu_model(T, tables):
    rng = np.random.default_rng(1000 + T)
    out, ref, rr = _head(rng, T, 128, tables, model=FA3)
    assert not rr.nonfinite.any()
    assert int(np.count_nonzero(out != ref)) == 0, (T, np.flatnonzero(out != ref)[:8])


def _head_v2(rng, T, D, BN, tables):
    """[R13 hopper] The statics-lifted `AttentionHead_v2{T, D, BN, DOT=HopperBF16WgmmaDot16, INV=Fa3InvSum}` (what
    `verity_vllm.registry.targets.attention_spec` binds on a Hopper target) against the CPU model at the same kBlockN."""
    from verity.ir.defs import bind
    from verity.ir.evaluate import evaluate_call
    from verity_vllm.registry import b1
    from verity_vllm.registry import hopper as H
    from verity_vllm.registry.crosscheck import _standalone

    prog, call = _standalone(bind(b1.AttentionHeadV2, T=T, D=D, BN=BN, DOT=H.HopperBF16WgmmaDot16, INV=H.Fa3InvSum))
    q = _rand_bf16(rng, D, scale=float(rng.choice([0.3, 1.0, 3.0])))
    k = _rand_bf16(rng, T * D, scale=float(rng.choice([0.3, 1.0, 3.0]))).reshape(T, D)
    v = _rand_bf16(rng, T * D).reshape(T, D)
    rr = FA3.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables, kblock_n=BN)
    ref = np.asarray(rr.out).reshape(-1)[:D]
    args = [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()]
    out = np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint16)
    return out, ref, rr


@needs_cpp
@pytest.mark.parametrize("T,BN", [(1, 192), (17, 192), (192, 192), (193, 192), (385, 192), (1, 128), (129, 128), (257, 128)])
def test_attention_head_v2_hd64_equals_cpu_model(T, BN, tables):
    """[R13 hopper] hdim 64 (Llama-3.2-1B): kBlockN 192 (one MMA warpgroup: decode / short prefill) and 128 (prefill).
    The typed Definition and the C++ recurrence are two spellings of one function at either tile."""
    rng = np.random.default_rng(2000 + 7 * T + BN)
    out, ref, rr = _head_v2(rng, T, 64, BN, tables)
    assert not rr.nonfinite.any()
    assert int(np.count_nonzero(out != ref)) == 0, (T, BN, np.flatnonzero(out != ref)[:8])


@needs_cpp
def test_fa3_hd64_tile_rule_and_bn_matters(tables):
    """The tile rule (`kblock_n_for` == the registered `targets.fa3_kblock_n`) and that BN 192 vs 128 is a real change
    of function on long rows (the running max after a different number of blocks; the f32 words before the cast)."""
    from verity_vllm.registry.targets import fa3_kblock_n

    for D, sq, qh in [(64, 1, 4), (64, 16, 4), (64, 17, 4), (64, 1024, 4), (128, 1, 6), (128, 512, 6), (64, 64, 1), (64, 65, 1)]:
        assert FA3.kblock_n_for(D, sq, qh) == fa3_kblock_n(D, sq, qh), (D, sq, qh)
    assert FA3.kblock_n_for(64, 1, 4) == 192 and FA3.kblock_n_for(64, 17, 4) == 128 and FA3.kblock_n_for(128, 1, 6) == 128
    rng = np.random.default_rng(11)
    D, T = 64, 700
    q = _rand_bf16(rng, D)
    k = _rand_bf16(rng, T * D).reshape(T, D)
    v = _rand_bf16(rng, T * D).reshape(T, D)
    a = FA3.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables, kblock_n=192, debug=True)
    b = FA3.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables, kblock_n=128, debug=True)
    assert a.dbg["m"].shape[1] == 4 and b.dbg["m"].shape[1] == 6  # ceil(700/192) vs ceil(700/128)
    assert int(np.count_nonzero(a.out_f32 != b.out_f32)) > 0
    with pytest.raises(ValueError):  # a tile the FA3 sm90 bf16 forward never launches is refused, not modelled by analogy
        FA3.attention_rows(q[None, :], k[:, None, :], v[:, None, :], np.zeros(1, np.int32), np.array([T], np.int32),
                           np.array([-(-T // 96)], np.int32), D ** -0.5, tables, kblock_n=96)
    with pytest.raises(ValueError):
        FA3.kblock_n_for(96, 1, 1)


@needs_cpp
def test_fa3_differs_from_fa2_where_hopper_differs(tables):
    """The same operands through the FA2 model (kBlockN 64, mma.sync grouping) differ from FA3 for long rows —
    the binding change is a real change of function — while both are deterministic.  The difference is in the
    f32 words before the bf16 cast (S, the running max after fewer blocks, the lane sums, O); after the cast
    it shows on ~10% of the words of a random row (and on none of some rows), so the f32 words are compared."""
    rng = np.random.default_rng(7)
    D, T = 128, 300
    q = _rand_bf16(rng, D)
    k = _rand_bf16(rng, T * D).reshape(T, D)
    v = _rand_bf16(rng, T * D).reshape(T, D)
    r3 = FA3.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables, debug=True)
    r3b = FA3.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables)
    r2 = FA2.attention_call(q[None, None, :], k[:, None, :], v[:, None, :], np.array([0]), 1, T, D ** -0.5, tables, debug=True)
    assert np.array_equal(r3.out, r3b.out) and np.array_equal(r3.out_f32, r3b.out_f32)
    assert r3.dbg["m"].shape[1] == 3 and r2.dbg["m"].shape[1] == 5  # ceil(300/128) vs ceil(300/64) blocks
    assert int(np.count_nonzero(r3.dbg["s"][:, :T] != r2.dbg["s"][:, :T])) > 0  # the wgmma grouping shows in S
    assert int(np.count_nonzero(r3.out_f32 != r2.out_f32)) > 0


@needs_cpp
def test_fa3_geometry_and_refusal(tables):
    col, nb = FA3.kernel_geometry(1, 527, [0])
    assert col.tolist() == [527] and nb.tolist() == [5]
    col, nb = FA3.kernel_geometry(512, 512, [0, 127, 128, 511])
    assert col.tolist() == [1, 128, 129, 512] and nb.tolist() == [1, 1, 2, 4]
    assert FA3.coverage(1, 527, 12, 2, 128, {"gpu": "NVIDIA H100 80GB HBM3"}) is None
    assert FA3.coverage(1, 527, 12, 2, 128, {"gpu": "NVIDIA A100"}) is not None
    assert FA3.coverage(1, 527, 12, 2, 128, {"gpu": "NVIDIA H100", "num_splits": 2}) is not None


def test_fa3_inv_sum_rule():
    """`Fa3InvSum`: 0 for sum == 0 or NaN (FA2's `Fa2InvSum` gives 1.0), MUFU.RCP otherwise."""
    from verity_vllm.registry import hopper as H
    from verity_vllm.registry import prims as P

    assert H.Fa3InvSum.evaluate(0) == 0 and H.Fa3InvSum.evaluate(0x80000000) == 0 and H.Fa3InvSum.evaluate(0x7FC00000) == 0
    assert P.Fa2InvSum.evaluate(0) == 0x3F800000
    assert H.Fa3InvSum.evaluate(0x40000000) == P.MufuRcpFtz.evaluate(0x40000000) == 0x3F000000
