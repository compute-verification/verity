"""`verity.ml.definitions.numerics.dense_rows`: the dense chain's kernels equal the reference (`verity.ml.circuits.evaluation`) on seeded, edge and
exhaustive bf16 words, claiming every finite instance; the k16 chain, by `cpp/k16_chain.cpp` and by `_chain` in numpy, equals
`derived_rows.dot_bf16` word for word on the coordinates it claims and on those it hands back (subnormal and overflowing steps,
non-finite operands), for any thread count; and `Gemm_v1`'s row kernel is the same on either path and against the reference."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from verity.ml.circuits.evaluation import self_check
from verity.ml.circuits.defs import bind
from verity.ml.definitions.numerics import dense_rows as DR
from verity.ml.definitions.numerics import derived_rows as D
from verity.ml.definitions.numerics import rows
from verity.ml.definitions.registry import b1, dense

ALL_BF16 = np.arange(1 << 16, dtype=np.uint16)
#: the 256 bf16 words with an all-ones exponent field (+-inf, NaN), 16 per instance of N=16 over `ALL_BF16`
NONFINITE_INSTANCES = 16
EDGE_F32 = np.array([0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x80800000,
                     0x3F800000, 0xBF800000, 0x3F7FFFFF, 0x4B7FFFFF, 0x1D6F2968, 0x9D6F2968, 0x0D000000, 0x5F000000],
                    dtype=np.uint32)


def _bf16(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def _finite_f32(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """f32 words whose products and sums of two stay finite: normals over 2^+-60, with signed zeros and subnormals mixed in."""
    w = _f32(rng.standard_normal(shape) * 2.0 ** rng.integers(-60, 60, shape))
    return np.where(rng.random(shape) < 1 / 8, rng.choice(EDGE_F32, shape), w)


def _finite_bf16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    w = _bf16(rng.standard_normal(shape) * 2.0 ** rng.integers(-60, 60, shape))
    edge = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80], dtype=np.uint16)
    return np.where(rng.random(shape) < 1 / 8, rng.choice(edge, shape), w)


def _check(target, inputs, *, declined: int | None = 0):
    sc = self_check(target, kernel=DR.KERNEL, inputs=inputs)
    assert sc.ok, str(sc)
    if declined is not None:
        assert sc.declined == declined, str(sc)
    return sc


def _cancelling_rows(rng: np.random.Generator, n: int, N: int) -> np.ndarray:
    """Rows of N f32 words whose sum depends on the order of the adds: +-2^100 pairs over small values."""
    x = rng.standard_normal((n, N)) * 2.0 ** rng.integers(-10, 10, (n, 1))
    big = 2.0 ** rng.integers(90, 110, (n, 8))
    at = np.stack([rng.permutation(N)[:16] for _ in range(n)])
    np.put_along_axis(x, at, np.concatenate([big, -big], axis=1), axis=1)
    return _f32(x)


def test_elementwise_kernels_claim_every_finite_instance():
    rng = np.random.default_rng(5701)
    n, N = 256, 16
    s = _finite_f32(rng, (n, 1))
    s = np.where(s == 0, _f32(np.ones((n, 1))), s)
    _check(bind(dense.SquareBf16, N=N), [_finite_bf16(rng, (n, N))])
    _check(bind(dense.SquareF32, N=N), [_finite_f32(rng, (n, N))])
    _check(bind(dense.AddWidenedBf16, N=N), [_finite_bf16(rng, (n, N)), _finite_bf16(rng, (n, N))])
    _check(bind(dense.ScaleRowBf16, N=N), [_finite_bf16(rng, (n, N)), s])
    _check(bind(dense.ScaleRowF32, N=N), [_finite_f32(rng, (n, N)), s])
    _check(bind(dense.MulVecF32, N=N), [_finite_f32(rng, (n, N)), _finite_f32(rng, (n, N))])


def test_elementwise_kernels_decline_non_finite_words_and_overflow():
    rng = np.random.default_rng(5702)
    n, N = 256, 16
    for t, width, arity in [(bind(dense.SquareF32, N=N), 32, 1), (bind(dense.MulVecF32, N=N), 32, 2),
                            (bind(dense.AddWidenedBf16, N=N), 16, 2), (bind(dense.SquareBf16, N=N), 16, 1)]:
        mix = rows._words(rng, (n, N), width)
        edge = (np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0x7F7FFFFF, 0xFF7FFFFF], dtype=np.uint32) if width == 32 else
                np.array([0x7F80, 0xFF80, 0x7FC0, 0x7F7F, 0xFF7F], dtype=np.uint16))
        words = [np.where(rng.random((n, N)) < 1 / 16, rng.choice(edge, (n, N)), mix) for _ in range(arity)]
        sc = _check(t, words, declined=None)
        assert sc.declined > 0, str(sc)


@pytest.mark.slow
def test_bf16_scalar_chain_on_every_bf16_word():
    words = [ALL_BF16.reshape(-1, 16)]
    _check(bind(dense.Bf16Tanh, N=16), words, declined=NONFINITE_INSTANCES)
    _check(bind(dense.Bf16DivScalar, N=16, C=30.0), words, declined=NONFINITE_INSTANCES)
    _check(bind(dense.Bf16MulScalar, N=16, C=30.0), words, declined=None)
    _check(bind(dense.SquareBf16, N=16), words, declined=None)
    s = _f32([[0.5], [3.0], [2.0 ** -70], [-1.25]])
    _check(bind(dense.ScaleRowBf16, N=1 << 14), [ALL_BF16.reshape(4, -1), s], declined=None)


@pytest.mark.parametrize("N", [16, 20, 1024, 2304])
def test_mean_triton_follows_the_body_reduction_order(N):
    """N = 16 and 1024 in the four-wide layout, 20 in the one-wide layout (a masked tail), 2304 as Gemma-2's H (two blocks and a
    masked 256-tail); random rows and rows whose sum depends on the order of the adds."""
    rng = np.random.default_rng(5703 + N)
    n = 12
    _check(bind(dense.MeanTriton, N=N), [_finite_f32(rng, (n, N))])
    _check(bind(dense.MeanTriton, N=N), [_f32(rng.standard_normal((n, N)))])
    if N >= 32:
        _check(bind(dense.MeanTriton, N=N), [_cancelling_rows(rng, n, N)])


def test_mean_triton_declines_an_overflowing_sum():
    x = np.full((2, 20), 0x7F7FFFFF, dtype=np.uint32)
    x[1] = _f32(np.arange(20.0))
    sc = _check(bind(dense.MeanTriton, N=20), [x], declined=1)
    assert sc.n == 2


# ---- the k16 chain ----------------------------------------------------------------------------------------------------------

K = 64
IN_CHECK = bool(os.environ.get("VERITY_SUITE_ROOT"))
CXX = shutil.which(os.environ.get("CXX", "g++")) is not None


@pytest.fixture(params=["cpp", "numpy"])
def chain_path(request, monkeypatch):
    """The k16 chain's two paths: `cpp/k16_chain.cpp` where it builds, `_chain` in numpy where it cannot (`k16_lib` None)."""
    if request.param == "numpy":
        monkeypatch.setattr(DR, "k16_lib", lambda: None)
    elif not CXX and not IN_CHECK:
        pytest.skip("needs a C++ compiler")
    else:
        assert DR.k16_lib() is not None, "cpp/k16_chain.cpp does not build"
    return request.param


def _f32_values(w: np.ndarray) -> np.ndarray:
    return (w.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _x(rng: np.random.Generator) -> np.ndarray:
    """A row whose words 16..31 are +-pairs (so a weight row of equal pairs cancels a whole k16 step) and whose last eight are
    bf16 subnormals."""
    x = rng.standard_normal(K) * 2.0 ** rng.integers(-20, 20, K)
    x[17:32:2] = -x[16:32:2]
    w = _bf16(x)
    w[-8:] = rng.integers(1, 0x80, 8).astype(np.uint16) | (rng.integers(0, 2, 8).astype(np.uint16) << 15)
    return w


def _w(rng: np.random.Generator, x: np.ndarray) -> tuple[np.ndarray, dict[str, slice]]:
    """Weight rows by regime: typical weights, wide exponents, random finite words, cancelling pairs, zeros, bf16 subnormals,
    tiny products (subnormal steps), huge products (overflowing steps), and rows with an infinity or a NaN."""
    words = rng.integers(0, 1 << 16, (40, K)).astype(np.uint16)
    regimes = {
        "typical": _bf16(rng.standard_normal((40, K)) * 0.02),
        "wide": _bf16(rng.standard_normal((40, K)) * 2.0 ** rng.integers(-40, 40, (40, K))),
        "words": np.where((words & 0x7F80) == 0x7F80, words ^ 0x4000, words),
        "cancel": np.repeat(_bf16(rng.standard_normal((12, K // 2))), 2, axis=1),
        "zeros": np.where(rng.random((4, K)) < 0.5, 0x8000, 0).astype(np.uint16),
        "subnormal": rng.integers(0, 0x80, (8, K)).astype(np.uint16) | (rng.integers(0, 2, (8, K)).astype(np.uint16) << 15),
        "tiny": _bf16(2.0 ** -133 * rng.uniform(-2, 2, (12, K)) / _f32_values(x)),
        "huge": _bf16(rng.standard_normal((12, K)) * 2.0 ** 112),
        "nonfinite": _bf16(rng.standard_normal((3, K))),
    }
    regimes["nonfinite"][0, 5], regimes["nonfinite"][1, 40], regimes["nonfinite"][2, 63] = 0x7F80, 0x7FC1, 0xFF80
    at, out = 0, {}
    for name, r in regimes.items():
        out[name] = slice(at, at + len(r))
        at += len(r)
    return np.concatenate(list(regimes.values())).astype(np.uint16), out


def _reference(x: np.ndarray, w: np.ndarray, dot_id: str = "AmpereBF16TcDot16_v2") -> tuple[np.ndarray, np.ndarray]:
    return D.dot_bf16_for(dot_id)(x[None, :], w)


@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_k16_chain_equals_the_int64_chain_in_every_regime(dot_id, chain_path):
    rng = np.random.default_rng(5704)
    x = _x(rng)
    w, regime = _w(rng, x)
    ref, ref_nf = _reference(x, w, dot_id)
    for n in (1, 3, 8):
        out, nf = DR.dot_chain(x, w, dot_id, n)
        assert np.array_equal(out, ref), (n, np.flatnonzero(out != ref)[:8])
        assert np.array_equal(nf, ref_nf)
    _, handed_back = DR._chain(x, DR.prepared(w), 0, len(w), DR.STEPS[dot_id])
    claimed = ~(handed_back | DR.prepared(w).bad)
    for name in ("typical", "wide", "cancel", "zeros", "subnormal"):
        assert claimed[regime[name]].all(), name
    assert claimed[regime["words"]].any() and not claimed[regime["words"]].all()
    exp_field = (ref >> 23) & 0xFF
    tiny, huge = regime["tiny"], regime["huge"]
    assert ((exp_field[tiny] == 0) & (ref[tiny] & 0x7FFFFFFF != 0)).any() and handed_back[tiny].all()
    assert (exp_field[huge] == 0xFF).any() and handed_back[huge].all()
    assert not claimed[regime["nonfinite"]].any()


@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_k16_chain_hands_back_every_coordinate_of_a_non_finite_row(dot_id, chain_path):
    rng = np.random.default_rng(5705)
    x = _x(rng)
    x[3] = 0x7F80
    w, _ = _w(rng, x)
    out, nf = DR.dot_chain(x, w, dot_id, 2)
    ref, ref_nf = _reference(x, w, dot_id)
    assert np.array_equal(out, ref) and np.array_equal(nf, ref_nf) and nf.all()


@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_k16_chain_on_cancelling_steps_and_exact_zeros(dot_id, chain_path):
    rng = np.random.default_rng(5706)
    x = _x(rng)
    w = np.repeat(_bf16(rng.standard_normal((16, K // 2))), 2, axis=1)
    w[:, :16] = 0
    w[:, 32:] = 0
    out, _ = DR.dot_chain(x, w, dot_id, 1)
    ref, _ = _reference(x, w, dot_id)
    assert np.array_equal(out, ref) and (ref == 0).all()


def test_k16_chain_refuses_a_partial_k16_step():
    x = _x(np.random.default_rng(5709))
    with pytest.raises(ValueError, match="multiple of 16"):
        DR.dot_ampere(x[:40], np.zeros((3, 40), dtype=np.uint16))


def test_prepared_weights_are_keyed_by_their_buffer(monkeypatch):
    monkeypatch.setattr(DR, "k16_lib", lambda: None)
    rng = np.random.default_rng(5707)
    x = _x(rng)
    w, _ = _w(rng, x)
    assert DR.prepared(w) is DR.prepared(w) and DR.prepared(w[:]) is DR.prepared(w)
    w2 = w.copy()
    w2[:, 0] ^= 0x0001
    assert DR.prepared(w2) is not DR.prepared(w)
    for arr in (w, w2, w):
        assert np.array_equal(DR.dot_ampere(x, arr, 2)[0], _reference(x, arr)[0])


@pytest.mark.parametrize("dot_id", sorted(DR.STEPS))
def test_the_words_are_the_same_at_any_thread_count(dot_id, monkeypatch, chain_path):
    """`VERITY_DENSE_THREADS` only splits the coordinates: at the default, 1, 16 and 64 threads the words are the int64 twin's; a value
    that is not a positive integer leaves the default, and the process's CPUs bound the count."""
    rng = np.random.default_rng(5710)
    x = _x(rng)
    w, _ = _w(rng, x)
    ref, ref_nf = _reference(x, w, dot_id)
    monkeypatch.setattr(DR, "_THREADS", [DR.MAX_THREADS])
    monkeypatch.setattr(DR.os, "sched_getaffinity", lambda _pid: set(range(64)))
    for env, n in (("", DR.MAX_THREADS), ("1", 1), ("16", 16), ("64", 64), ("0", DR.MAX_THREADS), ("many", DR.MAX_THREADS)):
        DR.use_threads(env)
        assert DR.threads() == n, env
        out, nf = DR.dot_chain(x, w, dot_id)
        assert np.array_equal(out, ref) and np.array_equal(nf, ref_nf), env
    monkeypatch.setattr(DR.os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    DR.use_threads(16)
    assert DR.threads() == 3


@pytest.mark.parametrize(("env", "cap"), [("16", 16), ("many", DR.MAX_THREADS), (None, DR.MAX_THREADS)])
def test_a_process_takes_the_cap_from_its_environment_at_import(env, cap):
    """No caller passes `VERITY_DENSE_THREADS`: a process that imports the module, as every stage process of a row does, is capped
    by its own environment under `use_threads`' rule."""
    environ = {k: v for k, v in os.environ.items() if k != "VERITY_DENSE_THREADS"}
    if env is not None:
        environ["VERITY_DENSE_THREADS"] = env
    out = subprocess.run([sys.executable, "-c", "from verity.ml.definitions.numerics import dense_rows as DR; print(DR._THREADS[0])"],
                         env=environ, capture_output=True, text=True, check=True).stdout
    assert int(out) == cap


def test_gemm_row_kernel_is_the_same_on_either_path(monkeypatch, chain_path):
    rng = np.random.default_rng(5708)
    x = _x(rng)
    w, _ = _w(rng, x)
    statics = {"K": K, "N": len(w)}
    slow = rows.ROWS["Gemm_v1"].fn(statics, [x, w.reshape(-1)])["out"]
    monkeypatch.setattr(DR, "FAST_GEMM_WORDS", 1)
    fast = rows.ROWS["Gemm_v1"].fn(statics, [x, w.reshape(-1)])["out"]
    assert np.array_equal(fast, slow)
    # typical and wide rows, cancelling / zero / subnormal rows, and zero / subnormal / tiny rows (steps handed back to the int64 chain)
    xs = np.stack([_x(rng), _bf16(rng.standard_normal(K)), x, x])
    ws = np.stack([w[i:i + 24] for i in (0, 40, 120, 132)])
    sc = self_check(bind(b1.Gemm, K=K, N=24), kernel=rows.KERNEL, inputs=[xs, ws])
    assert sc.ok and sc.declined == 0, str(sc)


def test_gemm_v2_row_kernel_takes_the_k16_chain_of_its_dot(monkeypatch, chain_path):
    """A `Gemm_v2` row past `FAST_GEMM_WORDS` (Gemma-2's lm_head under the Hopper DOT) takes the k16 chain of ITS step: the same
    words as the int64 twin, and never the Ampere chain's."""
    rng = np.random.default_rng(5710)
    x = _x(rng)
    w, _ = _w(rng, x)
    finite = ~_reference(x, w)[1]
    w = w[finite]
    statics = {"K": K, "N": len(w), "DOT": '{"fn":"HopperBF16WgmmaDot16_v1"}'}
    slow = rows.ROWS["Gemm_v2"].fn(statics, [x, w.reshape(-1)])["out"]
    monkeypatch.setattr(DR, "FAST_GEMM_WORDS", 1)
    calls = []
    monkeypatch.setattr(DR, "_claimed", lambda *a, _c=DR._claimed: calls.append(a[2]) or _c(*a))
    fast = rows.ROWS["Gemm_v2"].fn(statics, [x, w.reshape(-1)])["out"]
    assert calls and set(calls) == {DR.STEPS["HopperBF16WgmmaDot16_v1"]}
    assert np.array_equal(fast, slow)
    assert not np.array_equal(DR.dot_chain(x, w, "HopperBF16WgmmaDot16_v1")[0], DR.dot_ampere(x, w)[0])
