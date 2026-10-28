"""`RMSNormFusedCuda_v2` (generic `fused_add_rms_norm_kernel` under `VLLM_BATCH_INVARIANT=1`): the circuit body equals its
numpy twin gate for gate, and the twin reproduces every fused-norm row of the B0 commit store (v1 does not).

Needs the numerics package's MUFU.RSQ model (`verity.ml.definitions.numerics`: `librms_triton_model.so`, built on demand into the
`_jit` cache outside the checkout); skipped when it cannot be built here (no C++ compiler) or its tables are unavailable."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "vllm-poc") not in sys.path:
    sys.path.insert(0, str(REPO / "vllm-poc"))


def _numerics():
    try:
        from verity.ml.definitions.numerics import rms_relation, rms_triton_model as RM
    except Exception as e:  # pragma: no cover
        pytest.skip(f"verity.ml.definitions.numerics not importable: {e}")
    try:
        RM.build()                                     # into the _jit cache; a no-op when fresh
    except Exception as e:  # noqa: BLE001  -- no compiler / compile failure: skip, do not fail
        pytest.skip(f"librms_triton_model.so not buildable here: {type(e).__name__}: {e}")
    try:
        rms_relation.tables("cuda")
    except Exception as e:  # pragma: no cover
        pytest.skip(str(e))


def _standalone(spec):
    from tests.program.crosscheck import _standalone
    return _standalone(spec)


@pytest.mark.parametrize("N,eps", [(576, 1e-5), (1536, 1e-6)])
def test_ir_body_equals_numpy_twin(N, eps):
    _numerics()
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_call
    from verity.ml.definitions.registry import b1
    from verity_vllm.program import rmsnorm_fused_sweep as SW
    rng = np.random.default_rng(1234 + N)
    for defn, tree in ((b1.RMSNormFusedCuda, SW.variance_v2), (b1.RMSNormFusedCudaV1, SW.variance_v1)):
        prog, call = _standalone(bind(defn, N=N, EPS=eps))
        for scale in (0.02, 1.0, 40.0):
            x = ((rng.standard_normal(N) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            res = ((rng.standard_normal(N) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            w = ((rng.standard_normal(N)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
            out = np.array(evaluate_call(prog.circuit, call, [int(v) for v in x] + [int(v) for v in res] + [int(v) for v in w]), dtype=np.uint16)
            normed, rp, _, _ = SW.fused_rows(x[None], res[None], w[None], eps, tree)
            assert out.size == 2 * N
            assert np.array_equal(out[:N], normed[0]), defn.id
            assert np.array_equal(out[N:], rp[0]), defn.id


def test_v2_gate_counts():
    """N=1536: 1024 threads (512 with an FFMA), 32 warps x 31 adds + 31; N=576: 576 threads, 18 warps x 31 + 17."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import b1
    for N, eps, nthreads, nfma, nwarps in ((1536, 1e-6, 1024, 512, 32), (576, 1e-5, 576, 0, 18)):
        prog, call = _standalone(bind(b1.RMSNormFusedCuda, N=N, EPS=eps))
        expected = (N            # Bf16AddF2fp (res')
                    + N          # Bf16ToF32 (res')
                    + nthreads   # per-thread F32Mul
                    + nfma       # per-thread F32Fma
                    + nwarps * 31 + (nwarps - 1)  # cub BlockReduce
                    + 2          # Const N, F32Div
                    + 2          # Const eps, F32Add
                    + 1          # RsqrtApprox
                    + N          # Bf16ToF32 (w)
                    + 2 * N      # (res' * s) * w
                    + N)         # F2fpBf16
        assert call.size == expected, (N, call.size, expected)


def test_v1_and_v2_trees_differ_only_in_the_variance():
    """Same residual output, same epilogue: an output difference between v1 and v2 is a variance difference."""
    from verity_vllm.program import rmsnorm_fused_sweep as SW
    rng = np.random.default_rng(7)
    N = 1536
    z = rng.standard_normal((64, N)).astype(np.float32)
    z = (z.view(np.uint32) >> 16 << 16).view(np.float32)  # bf16-valued
    v1, v2 = SW.variance_v1(z), SW.variance_v2(z)
    # both are correctly-rounded-per-step sums of the same squares: they agree to a few ulps, not bit for bit
    assert np.all(np.abs(v1 - v2) <= 4 * np.spacing(np.maximum(v1, v2)))
    assert np.any(v1 != v2)


def test_b0_store_sweep(tmp_path):
    """The whole B0 store: v2 exact on all 2,340 fused-norm rows; v1 misses 13 (the class of the B1 §5.1 row)."""
    _numerics()
    from verity_vllm.program import rmsnorm_fused_sweep as SW
    if not SW.BUNDLE.exists():
        pytest.skip(f"B0 bundle not present: {SW.BUNDLE}")
    commit = SW.extract_bundle(SW.BUNDLE, tmp_path)
    rep = SW.sweep(commit)
    assert rep["rows"] == 2340
    assert rep["trees"]["v2"]["rows_exact"] == 2340
    assert rep["trees"]["v2"]["words_exact"] == rep["trees"]["v2"]["words_compared"] == 2 * 2340 * 576
    assert rep["trees"]["v1"]["rows_exact"] == 2327
    assert rep["v1_vs_v2"]["output_differs_rows"] == 13
    for r in rep["v1_vs_v2"]["rows"]:
        assert r["v2"] == r["captured"] and r["v1"] != r["captured"]
