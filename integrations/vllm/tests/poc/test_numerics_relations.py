"""CPU unit tests for W4's ``independent_gemm`` (verity_vllm_numerics.relations), contract 7 / 13.

Tiny shapes only; no GPU.  Two things are pinned down:

* the arithmetic: the C++ chain model (``model_gemm_bytes``) agrees coordinate-for-coordinate
  with the pure-Python ``tc_dot`` chain + numpy epilogue (``independent_gemm_coordinate``), for
  the three epilogues the production kernel uses;
* the coverage gate: ``independent_gemm`` predicts only on the (kernel path, K, N, epilogue)
  families measured bit-exact on the A100 and raises ``GemmUnresolved`` (``unresolved: ...``)
  everywhere else, including on the tiny test shape itself.

Needs the compiled ``cpp/tc_model.cpp`` (g++ -fopenmp); the arithmetic tests skip where it cannot
be built (macOS clang), so run them on the pod: ``python -m pytest tests/poc/test_numerics_relations.py -q``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verity_vllm_numerics import relations as REL  # noqa: E402
from verity_vllm_numerics.relations import GemmRelationError, GemmUnresolved  # noqa: E402

W1_BI_IMPL = "triton:vllm.model_executor.determinism.batch_invariant.matmul_persistent (via linear_batch_invariant)"


def _have_cpp() -> bool:
    try:
        from verity_vllm_numerics import cpu_model as CM

        CM.lib()
        return True
    except Exception:  # noqa: BLE001  (no g++/OpenMP on this machine)
        return False


needs_cpp = pytest.mark.skipif(not _have_cpp(), reason="tc_model.cpp needs g++ -fopenmp (run on the pod)")


def bf16_words(rng: np.random.Generator, n: int, lo_exp: int = 120, hi_exp: int = 134) -> np.ndarray:
    """Random finite BF16 words with exponents in [lo_exp, hi_exp) (values ~ 2^-7 .. 2^7)."""
    sign = rng.integers(0, 2, n, dtype=np.uint16) << 15
    exp = rng.integers(lo_exp, hi_exp, n, dtype=np.uint16) << 7
    man = rng.integers(0, 128, n, dtype=np.uint16)
    return (sign | exp | man).astype(np.uint16)


def node(runtime_op: str, impl: str | None = W1_BI_IMPL, **consts) -> dict:
    return {"args": {"x": "x", "w": "w", "bias": "bias"}, "runtime_impl": impl, "extra": {"runtime_op": runtime_op, "consts": consts}}


def python_reference(nd: dict, x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """Coordinate-by-coordinate pure-Python model (independent of the C++ chain)."""
    M, K = x.shape
    N = w.shape[0]
    out = np.zeros((M, N), np.uint16)
    for i in range(M):
        inputs = {"x": x[i : i + 1].copy(), "w": w}
        if b is not None:
            inputs["bias"] = b
        for j in range(N):
            out[i, j] = np.frombuffer(REL.independent_gemm_coordinate(nd, inputs, j), dtype="<u2")[0]
    return out


@needs_cpp
@pytest.mark.parametrize("runtime_op,with_bias", [("verity.bi_mm", False), ("aten.addmm", True), ("aten.linear", True)])
def test_chain_model_matches_python_reference_on_a_tiny_shape(runtime_op: str, with_bias: bool) -> None:
    rng = np.random.default_rng(7)
    M, K, N = 3, 64, 8
    x = bf16_words(rng, M * K).reshape(M, K)
    w = bf16_words(rng, N * K).reshape(N, K)
    b = bf16_words(rng, N) if with_bias else None
    nd = node(runtime_op)
    inputs = {"x": x, "w": w} | ({"bias": b} if b is not None else {})
    got = np.frombuffer(REL.model_gemm_bytes(nd, inputs), dtype="<u2").reshape(M, N)
    want = python_reference(nd, x, w, b)
    assert np.array_equal(got, want), f"C++ chain != Python chain at {np.argwhere(got != want)[:4].tolist()}"
    # the tiny shape is NOT a validated family: the gated relation must decline, not predict
    with pytest.raises(GemmUnresolved, match=r"^unresolved: shape family K=64 N=8"):
        REL.independent_gemm(nd, inputs)


@needs_cpp
def test_gate_predicts_on_a_validated_family_and_mutations_change_it() -> None:
    rng = np.random.default_rng(11)
    K, N = 1536, 1536  # o_proj family, one opened row (M=1), no bias
    x = bf16_words(rng, K).reshape(1, K)
    w = bf16_words(rng, N * K).reshape(N, K)
    nd = node("verity.bi_mm")
    cov = REL.coverage(nd, {"x": x, "w": w})
    assert cov["covered"] and cov["family"] == "o_proj" and cov["kernel_path"] == "matmul_kernel_persistent"
    assert cov["spec"].nsplit == 1 and cov["spec"].bias_mode == 0
    out = REL.independent_gemm(nd, {"x": x, "w": w})
    assert len(out) == N * 2
    assert out == REL.model_gemm_bytes(nd, {"x": x, "w": w})
    words = np.frombuffer(out, dtype="<u2")
    # spot-check three coordinates against the pure-Python chain
    for j in (0, 777, N - 1):
        assert words[j] == np.frombuffer(REL.independent_gemm_coordinate(nd, {"x": x, "w": w}, j), dtype="<u2")[0]
    # contract 15.2b: a one-word mutation of the opened input changes the prediction
    x2 = x.copy()
    x2[0, 5] ^= 0x0001
    assert REL.independent_gemm(nd, {"x": x2, "w": w}) != out


def test_gate_declines_everything_it_did_not_measure() -> None:
    rng = np.random.default_rng(3)
    K, N = 1536, 1536
    x = bf16_words(rng, K).reshape(1, K)
    w = bf16_words(rng, N * K).reshape(N, K)
    base = {"x": x, "w": w}

    def reason(nd: dict, inputs: dict) -> str:
        cov = REL.coverage(nd, inputs)
        assert not cov["covered"]
        with pytest.raises(GemmUnresolved) as info:
            REL.independent_gemm(nd, inputs)
        assert str(info.value).startswith("unresolved:")
        return cov["reason"]

    # unvalidated shape family under the validated kernel
    assert "shape family K=1536 N=1000" in reason(node("verity.bi_mm"), {"x": x, "w": w[:1000]})
    # cuBLAS GEMV path: no model attempted
    assert "gemvx" in reason(node("aten.mm", impl="void cublasLt::internal::gemvx::kernel<...>", M=1), base)
    # cuBLAS launch nobody measured, and one with no declared M
    assert "not in the resolved zoo" in reason(node("aten.mm", impl="ampere_bf16_s16816gemm_bf16_64x64_nn_unknown", M=64), base)
    assert "declared M" in reason(node("aten.mm", impl="ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn"), base)
    # unvalidated epilogue on a validated family (down_proj measured without bias)
    xb = bf16_words(rng, 8960).reshape(1, 8960)
    wb = bf16_words(rng, 1536 * 8960).reshape(1536, 8960)
    assert "epilogue 'bias_bf16_after' not validated on down_proj" in reason(node("aten.linear"), {"x": xb, "w": wb, "bias": bf16_words(rng, 1536)})
    # unknown arch: no per-instruction model
    assert "arch 'sm_70'" in reason(node("verity.bi_mm", arch="sm_70"), base)
    # a zoo entry is covered only with the epilogue it was validated with
    assert "differs from the validated zoo epilogue" in reason(
        node("aten.linear", impl="ampere_bf16_s16816gemm_bf16_128x128_ldg8_relu_f2f_stages_64x3_tn", M=512, epilogue="bias_bf16_after"),
        {"x": x, "w": w, "bias": bf16_words(rng, N)},
    )


def test_gate_covers_the_measured_cublas_launches_via_the_zoo() -> None:
    rng = np.random.default_rng(5)
    x = bf16_words(rng, 8960).reshape(1, 8960)
    w = bf16_words(rng, 1536 * 8960).reshape(1536, 8960)
    impl = ["ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn", "splitKreduce_kernel<32, 16, int, __nv_bfloat16>"]
    cov = REL.coverage(node("aten.mm", impl="\n".join(impl), M=64), {"x": x, "w": w})
    assert cov["covered"] and cov["family"] == "cuBLAS zoo"
    assert cov["spec"].nsplit == 8 and cov["spec"].reduce_mode == 3  # split-K 8, BF16-rounded partials
    # the same launch at an M the zoo never saw is not covered, nor is a tile config of the same family
    assert not REL.coverage(node("aten.mm", impl="\n".join(impl), M=65), {"x": x, "w": w})["covered"]
    x2 = bf16_words(rng, 1536).reshape(1, 1536)
    w2 = bf16_words(rng, 1536 * 1536).reshape(1536, 1536)
    single = node("aten.mm", impl="ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_64x3_tn", M=2048)
    assert REL.coverage(single, {"x": x2, "w": w2})["spec"].nsplit == 1
    assert not REL.coverage(node("aten.mm", impl="ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x3_tn", M=2048), {"x": x2, "w": w2})["covered"]


def test_non_finite_operands_are_outside_the_domain() -> None:
    rng = np.random.default_rng(9)
    x = bf16_words(rng, 1536).reshape(1, 1536)
    w = bf16_words(rng, 1536 * 1536).reshape(1536, 1536)
    x[0, 3] = 0x7F80  # +inf
    with pytest.raises(GemmRelationError, match="non-finite"):
        REL.coverage(node("verity.bi_mm"), {"x": x, "w": w})
    with pytest.raises(GemmRelationError, match="non-finite"):
        REL.independent_gemm(node("verity.bi_mm"), {"x": x, "w": w})


def test_registry_ids_and_dict_registration() -> None:
    reg: dict = {}
    REL.register(reg)
    assert set(reg) == {"gemm.bf16.v1", "lm_head.bf16.v1", "gemm.bf16.coordinate.v1"}
    assert reg["gemm.bf16.v1"] is REL.independent_gemm and reg["lm_head.bf16.v1"] is REL.independent_gemm
    # the W2 wrapper (when verity_vllm is importable) turns GemmUnresolved into RelationError("unresolved: ...")
    R = pytest.importorskip("verity_vllm.relations")
    rel = R.REGISTRY.get("gemm.bf16.v1")
    assert rel is not None and rel.klass == R.INDEPENDENT and not rel.includes_full_op
    rng = np.random.default_rng(1)
    with pytest.raises(R.RelationError, match=r"^unresolved:"):
        rel.checker(node("verity.bi_mm"), {"x": bf16_words(rng, 64).reshape(1, 64), "w": bf16_words(rng, 8 * 64).reshape(8, 64)})
