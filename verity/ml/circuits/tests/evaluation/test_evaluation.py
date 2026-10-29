"""`verity.ml.circuits.evaluation`: the reference, `evaluate_batch`, the kernel registry and `self_check`.

Every registered kernel is self-checked here against the plain-integer reference on seeded random inputs (the test
fails if a kernel is registered without being listed), and the tensor-core kernels also replay the hardware captures the
scalar models are pinned to (A100 / RTX 4090 / H100 special-value records, RTX 5090 NVFP4 vectors, RTX 4090 GEMM
coordinates) through `evaluate_batch`.
"""
from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation.batch import _KERNELS
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Value
from verity.ml.circuits.ops import gemm as G
from verity.ml.kernels import tc as K
from verity.ml.circuits.ops import mufu as M
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.silicon.models import AMPERE_BF16_M16N8K16, GroupSum
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16, tc_dot_total

FIXTURES = Path(__file__).resolve().parents[2] / "ops" / "tests" / "fixtures"
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists())

#: (target, instances) for every registered kernel; composites and bindable references at representative statics.
SELF_CHECKED = [
    (P.Bf16ToF32, 20000),
    (P.F32ToBf16Rn, 20000),
    (P.F2fpBf16, 20000),
    (P.AmpereBF16TcDot16, 20000),
    (P.HopperBF16WgmmaDot16, 20000),
    (bind(G.DotBf16, K=64, DOT=P.AmpereBF16TcDot16), 2000),
    (bind(G.DotBf16, K=48, DOT=P.HopperBF16WgmmaDot16), 2000),
    (bind(G.GemmCoordinate, K=64, DOT=P.AmpereBF16TcDot16), 2000),
    (bind(G.GemmCoordinate, K=64, DOT=P.HopperBF16WgmmaDot16), 2000),
    (bind(G.GemmCoordinatePad, T=17, K=32, DOT=P.HopperBF16WgmmaDot16), 2000),
    (bind(G.PvRow, T=8, K=32, D=4, DOT=P.HopperBF16WgmmaDot16), 300),
    (bind(G.PvRowPad, T=8, K=32, D=4, DOT=P.HopperBF16WgmmaDot16), 300),
    (bind(G.Gemm, K=32, N=3, DOT=P.AmpereBF16TcDot16), 500),
    (P.HopperE4m3QgmmaDot32, 20000),
    (P.HopperE4m3MmaV2Dot32, 20000),
    (P.BlackwellE4m3QmmaDot32, 20000),
    (P.BlackwellE5m2QmmaDot32, 20000),
    (P.BlackwellNvf4OmmaDot64, 20000),
    (P.BlackwellMxf4OmmaDot64, 20000),
    (bind(G.GemmCoordinateE4m3, K=128, **G.GEMM_SM120_E4M3), 2000),
    (bind(G.GemmCoordinateE4m3, K=64, DOT=P.HopperE4m3QgmmaDot32), 2000),
    (bind(G.GemmCoordinateNvf4, K=256, **G.GEMM_SM120_NVF4), 2000),
    (bind(G.GemmCoordinateMxf4, K=256, **G.GEMM_SM120_MXF4), 2000),
    (P.F32ToE2m1Sat, 20000),
    (P.E4m3ToF32, 20000),
    (K.HOPPER_E4M3_STEP, 20000),
    (K.HOPPER_E4M3_DOT.bind(K=128), 2000),
    (K.ADA_E4M3_STEP, 20000),
    (K.ADA_E4M3_DOT.bind(K=128), 2000),
    (K.SM120_NVF4_STEP, 20000),
    (K.SM120_NVF4_DOT.bind(K=256), 2000),
    (K.SM120_MXF4_STEP, 20000),
    (K.SM120_MXF4_DOT.bind(K=256), 2000),
    (M.MufuEx2Ftz, 20000),
    (M.MufuRcpFtz, 20000),
    (M.Fa2InvSum, 20000),
    (M.MufuSqrtFtz, 20000),
    (M.DivFullRcp, 20000),
    (M.DivFullScaleA, 20000),
    (M.RsqrtApprox, 20000),
]


#: the import packages whose kernels their own tests self-check; a kernel from anywhere else is core's, and listed above
INTEGRATIONS = {"verity_vllm"}


def _id(t) -> str:
    return getattr(t, "id", str(t))


def test_every_registered_kernel_is_self_checked_here() -> None:
    """Every kernel core registers; an integration's own kernels (vLLM's twins, registered when its package is imported in the same
    session) are self-checked by that integration's tests."""
    core = {k for k, ks in _KERNELS.items() if any(v.fn.__module__.partition(".")[0] not in INTEGRATIONS for v in ks.values())}
    assert core == {E.batch.key_of(t) for t, _ in SELF_CHECKED}


@pytest.mark.parametrize("target,n", SELF_CHECKED, ids=[_id(t) for t, _ in SELF_CHECKED])
def test_kernel_matches_the_reference(target, n: int) -> None:
    r = E.self_check(target, n=n, seed=20260925)
    assert r.ok, str(r)
    assert r.declined == r.outside, str(r)  # declines exactly what the reference rejects, nothing more


# ---- the reference ------------------------------------------------------------------------------------------------------

def test_evaluate_is_the_primitive_and_the_direct_chain() -> None:
    rng = random.Random(1)
    x = [rng.getrandbits(16) & 0x7F7F for _ in range(64)]
    w = [rng.getrandbits(16) & 0x7F7F for _ in range(64)]
    assert E.evaluate(P.AmpereBF16TcDot16, 5, x[:16], w[:16]) == (P.AmpereBF16TcDot16.evaluate(5, *x[:16], *w[:16]),)
    acc = 0
    for s in range(0, 64, 16):
        acc = tc_dot_total(HOPPER_BF16_WGMMA_K16, acc, x[s:s + 16], w[s:s + 16])
    assert E.evaluate(bind(G.DotBf16, K=64, DOT=P.HopperBF16WgmmaDot16), x, w) == (acc,)
    assert E.evaluate(K.HOPPER_E4M3_DOT.bind(K=64), [0x38] * 64, [0x38] * 64) == (0x42800000,)  # 64 x 1.0


def test_scalar_reference_statics() -> None:
    with pytest.raises(TypeError):
        K.HOPPER_E4M3_DOT.bind(N=3)
    with pytest.raises(TypeError):
        E.evaluate_batch(K.HOPPER_E4M3_DOT, np.zeros((1, 32), np.uint8), np.zeros((1, 32), np.uint8))


# ---- evaluate_batch -------------------------------------------------------------------------------------------------------

def _bf16_rows(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    return (rng.integers(0, 1 << 16, size=(n, k)) & 0x7F7F).astype(np.uint16)  # finite, no NaN/inf


def test_batch_equals_reference_and_kernel_choice() -> None:
    rng = np.random.default_rng(0)
    spec = bind(G.GemmCoordinate, K=64, DOT=P.AmpereBF16TcDot16)
    x, w = _bf16_rows(rng, 200, 64), _bf16_rows(rng, 200, 64)
    fast = E.evaluate_batch(spec, x, w)
    ref = E.evaluate_batch(spec, x, w, kernel="reference")
    assert fast.dtype == np.uint16 and fast.shape == (200,)
    assert np.array_equal(fast, ref)
    assert np.array_equal(E.evaluate_batch(spec, x, w, kernel="numpy"), ref)
    with pytest.raises(KeyError):
        E.evaluate_batch(spec, x, w, kernel="cuda")


def test_gemm_batch_shapes() -> None:
    rng = np.random.default_rng(1)
    spec = bind(G.Gemm, K=32, N=3, DOT=P.HopperBF16WgmmaDot16)
    x, w = _bf16_rows(rng, 7, 32), _bf16_rows(rng, 21, 32).reshape(7, 3, 32)
    y = E.evaluate_batch(spec, x, w)
    assert y.shape == (7, 3)
    coord = bind(G.GemmCoordinate, K=32, DOT=P.HopperBF16WgmmaDot16)
    for j in range(3):
        assert np.array_equal(y[:, j], E.evaluate_batch(coord, x, w[:, j], kernel="reference"))
    assert np.array_equal(E.evaluate_batch(spec, x, w.reshape(7, 96)), y)  # flat leaves per input are accepted


def test_unknown_step_falls_back_to_the_reference() -> None:
    """A DOT the kernel does not know (here an unregistered twin of the Ampere step) is declined whole."""
    from verity.ml.circuits.defs import PrimitiveDefinition

    twin = PrimitiveDefinition("AmpereBF16TcDot16", 2, P.AmpereBF16TcDot16.params, P.F32, P.AmpereBF16TcDot16.evaluate,
                               register=False)
    rng = np.random.default_rng(2)
    x, w = _bf16_rows(rng, 5, 32), _bf16_rows(rng, 5, 32)
    got = E.evaluate_batch(bind(G.DotBf16, K=32, DOT=twin), x, w)
    assert np.array_equal(got, E.evaluate_batch(bind(G.DotBf16, K=32, DOT=P.AmpereBF16TcDot16), x, w))
    assert K.dot_kernel({})(x, w, K=32, DOT=twin) is NotImplemented


def test_declined_instances_are_recomputed_and_reference_errors_propagate() -> None:
    rng = np.random.default_rng(3)
    ac, bc = (rng.integers(0, 16, size=(4, 64)).astype(np.uint8) for _ in range(2))
    sa, sb = (np.full((4, 4), 0x38, np.uint8) for _ in range(2))
    acc = np.array([0, 0x3F800000, 0x7F800000, 0x00000001], np.uint32)  # the infinite accumulator is outside the model
    words, declined = K.block_scaled_batch(K.BLACKWELL_SM120_NVF4, acc, ac, bc, sa, sb)
    assert declined.tolist() == [False, False, True, False]
    with pytest.raises(Exception, match="non-finite"):
        E.evaluate_batch(K.SM120_NVF4_STEP, acc, ac, bc, sa, sb)
    ok = [0, 1, 3]
    got = E.evaluate_batch(K.SM120_NVF4_STEP, acc[ok], ac[ok], bc[ok], sa[ok], sb[ok])
    assert got.tolist() == [K.BLACKWELL_SM120_NVF4.step_scaled(int(acc[i]), ac[i].tolist(), bc[i].tolist(), sa[i].tolist(),
                                                               sb[i].tolist()) for i in ok]


def test_partial_decline_goes_to_the_reference() -> None:
    ref = E.ScalarReference("test.halve", lambda x: x >> 1, lambda: ((Value(8),), Value(8)))
    E.register_kernel(ref, lambda x: (np.zeros_like(x), x % 2 == 1), domain="even words only (a deliberately bad kernel "
                                                                               "that declines the odd ones)")
    x = np.arange(256, dtype=np.uint8)
    got = E.evaluate_batch(ref, x)
    assert np.array_equal(got[1::2], x[1::2] >> 1) and not got[0::2].any()  # the kernel's (wrong) evens were accepted
    r = E.self_check(ref, inputs=[x])
    assert not r.ok and r.declined == 128 and len(r.mismatches) == 127  # every nonzero even word is a mismatch
    del _KERNELS[ref.id]


def test_self_check_catches_a_wrong_adder() -> None:
    import dataclasses

    wrong = dataclasses.replace(AMPERE_BF16_M16N8K16, width=24)
    ref = E.ScalarReference("test.ampere_step", lambda acc, a, b: tc_dot_total(AMPERE_BF16_M16N8K16, acc, a, b),
                            lambda: ((Value(32), Array(16, Value(16)), Array(16, Value(16))), Value(32)))
    E.register_kernel(ref, lambda acc, a, b: K.group_sum_total_batch(wrong, acc, a, b), sample=K.sample_tc)
    r = E.self_check(ref, n=2000)
    del _KERNELS[ref.id]
    assert len(r.mismatches) > 50, str(r)


def test_register_kernel_is_one_function_per_name() -> None:
    fn = _KERNELS["AmpereBF16TcDot16_v2"]["numpy"].fn
    assert E.register_kernel(P.AmpereBF16TcDot16, fn).fn is fn  # idempotent
    with pytest.raises(ValueError):
        E.register_kernel(P.AmpereBF16TcDot16, lambda *a: None)
    assert set(E.kernels(P.AmpereBF16TcDot16)) == {"numpy"}


def test_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="wider"):
        E.evaluate_batch(P.Bf16ToF32, np.array([1 << 16], np.uint32))
    with pytest.raises(ValueError, match="negative"):
        E.evaluate_batch(P.Bf16ToF32, np.array([-1]))
    with pytest.raises(ValueError, match="shape"):
        E.evaluate_batch(P.AmpereBF16TcDot16, np.zeros(2, np.uint32), np.zeros((2, 15), np.uint16), np.zeros((2, 16), np.uint16))
    with pytest.raises(ValueError, match="batch size"):
        E.evaluate_batch(P.AmpereBF16TcDot16, np.zeros(2, np.uint32), np.zeros((3, 16), np.uint16), np.zeros((3, 16), np.uint16))
    assert E.evaluate_batch(P.Bf16ToF32, np.zeros(0, np.uint16)).shape == (0,)


def test_a_broadcast_operand_is_read_in_place() -> None:
    """One weight for every instance (a read-only broadcast view): words no wider than the declared widths are not widened or copied."""
    import tracemalloc

    from verity.ml.circuits.evaluation.batch import _as_batch

    big = np.broadcast_to(np.arange(1 << 16, dtype=np.uint16), (64, 1 << 16))
    tracemalloc.start()
    a = _as_batch(big, Array(1 << 16, Value(16)), "w", None)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert np.shares_memory(a, big) and peak < 1 << 20                       # a widened copy would be 32 MiB
    rng = np.random.default_rng(4)
    spec = bind(G.Gemm, K=32, N=3, DOT=P.HopperBF16WgmmaDot16)
    x, w = _bf16_rows(rng, 7, 32), _bf16_rows(rng, 3, 32)
    shared = np.broadcast_to(w, (7, 3, 32))
    assert np.array_equal(E.evaluate_batch(spec, x, shared), E.evaluate_batch(spec, x, np.ascontiguousarray(shared)))
    with pytest.raises(ValueError, match="wider"):
        _as_batch(np.broadcast_to(np.array([1 << 16], np.uint32), (4, 1)), Array(1, Value(16)), "w", None)


# ---- hardware captures through the kernels -----------------------------------------------------------------------------

def _u16(s: str) -> list[int]:
    return [int(s[i:i + 4], 16) for i in range(0, len(s), 4)]


def _replay(target, cases) -> None:
    acc = np.array([int(c["acc"], 16) for c in cases], np.uint32)
    a = np.array([_u16(c["a"]) for c in cases], np.uint16)
    b = np.array([_u16(c["b"]) for c in cases], np.uint16)
    d = np.array([int(c["d"] if isinstance(c["d"], str) else c["d"][0], 16) for c in cases], np.uint32)
    got, declined = E.kernels(target)["numpy"].fn(acc, a, b)
    assert declined is None and np.array_equal(got, d)


def test_ampere_step_replays_the_a100_and_rtx4090_special_value_records() -> None:
    files = sorted((FIXTURES / "tc-total-2026-09-07").glob("*/mma_special.reduced.json.gz"))
    assert files
    for path in files:
        with gzip.open(path, "rt") as f:
            doc = json.load(f)
        _replay(P.AmpereBF16TcDot16, doc["cases"] + doc["pos"])


def test_hopper_step_replays_the_h100_record() -> None:
    with gzip.open(FIXTURES / "tc-hopper-2026-09-07" / "special.reduced.json.gz", "rt") as f:
        doc = json.load(f)
    _replay(P.HopperBF16WgmmaDot16, [c for cases in doc["families"].values() for c in cases])


def test_nvfp4_dot_replays_every_recorded_rtx5090_vector() -> None:
    cases = json.loads((ROOT / "verity" / "ml" / "circuits" / "silicon" / "captures" / "autoproof-probe-vectors.json").read_text())["nvfp4_m16n8k64"]["cases"]
    by_k: dict[int, list] = {}
    for c in cases:
        by_k.setdefault(len(c["a"]), []).append(c)
    assert sorted(by_k) == [64, 128, 256] and sum(map(len, by_k.values())) == 1601
    for k, cs in by_k.items():
        cols = [np.array([[int(ch, 16) for ch in c[f]] for c in cs], np.uint8) for f in ("a", "b")]
        cols += [np.array([bytes.fromhex(c[f]) for c in cs]).view(np.uint8).reshape(len(cs), -1) for f in ("sa", "sb")]
        got, declined = E.kernels(K.SM120_NVF4_DOT)["numpy"].fn(*cols, K=k)
        assert not declined.any()
        assert got.tolist() == [int(c["d"], 16) for c in cs]


def test_gemm_coordinate_replays_the_rtx4090_captures() -> None:
    root = FIXTURES / "gemm-B1-b3eba618b1b5-20260907T1731Z"
    by_k: dict[int, list] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = json.loads((d / "fixture.json").read_text())
        x = np.frombuffer((d / "x.u16").read_bytes(), "<u2")
        if f["x_arg_cols"] is not None:
            x = x[slice(*f["x_arg_cols"])]
        by_k.setdefault(f["K"], []).append((x, np.frombuffer((d / "w_row.u16").read_bytes(), "<u2"), f["committed_word"]))
    assert sum(map(len, by_k.values())) == 30
    for k, rows in by_k.items():
        x, w = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
        got = E.evaluate_batch(bind(G.GemmCoordinate, K=k, **G.GEMM_AMPERE), x, w)
        assert got.tolist() == [r[2] for r in rows]


def test_group_sum_kernel_covers_every_total_pipeline() -> None:
    """The kernel is parameterised by the model, so it is exact for the other E4M3 pipeline too (Ada's two groups)."""
    from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32
    from verity.ml.circuits.silicon.total_fp8 import tc_dot_total_e4m3

    assert isinstance(ADA_E4M3_M16N8K32, GroupSum)
    ref = E.ScalarReference("test.ada_e4m3", lambda acc, a, b: tc_dot_total_e4m3(ADA_E4M3_M16N8K32, acc, a, b),
                            lambda: ((Value(32), Array(32, Value(8)), Array(32, Value(8))), Value(32)))
    E.register_kernel(ref, lambda acc, a, b: K.group_sum_total_batch(ADA_E4M3_M16N8K32, acc, a, b), sample=K.sample_tc)
    r = E.self_check(ref, n=5000, seed=5)
    del _KERNELS[ref.id]
    assert r.ok, str(r)
