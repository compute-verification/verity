"""`registry.dsa_norm`'s `DsaRMSNorm_v1` (the fused norm + RoPE kernel's `_rms_norm`, programs 1 and 2) against a model written here
from the H200's PTX, lane by lane: each thread's vector loads unpacked as `mov.b32 {lo, hi}` does, every FMUL, FFMA and FADD rounded
once from the exact rational (`test_dsa`'s helpers), `shfl.sync.bfly` simulated over every lane; `rsqrt.approx.ftz` is core's
`RsqrtApprox_v1`.  Its twin (`kernels.dsa_norm_rows`) against the reference evaluator on edge rows; the statics it refuses; and that
its scale is not `RMSNormTriton_v1`'s on served-like rows.  `TritonBlockRMSNorm_v1` (`_fused_q_kv_rmsnorm_kernel`) against the same
kind of model from its PTX and SASS, every masked lane's +0 summed, and `DsaRMSNorm_v1`'s words as its 4-warp unmasked case's (the
contracted eps is the same word at a power-of-two N).  The Boolean versions'
agreement with these word views is circuit-check's (`boolean_dsa_norm.circuit-check.py`)."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.boolean import trace as TR
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics import dsa_norm_rows as NR
from verity.ml.definitions.numerics import rows as RK
from verity.ml.definitions.registry import boolean_dsa_norm as BN
from verity.ml.definitions.registry import dsa_norm as DN
from verity.ml.definitions.registry import prims as P
from tests.program.test_dsa import _butterfly, _fadd
from tests.program.test_fp8_moe import _bf16, _ffma, _fmul

EPS = 1e-5


def _regs(x: list[int], t: int) -> list[int]:
    """Thread t's f32 registers in the PTX's order.  N = 2048 (program 2): two `ld.global.v4.b32`, at byte 16t (`shl.b32 3`,
    `and.b32 1016`) and 2048 past it; N = 512 (program 1): one `ld.global.v2.b32` at byte 8t (`shl.b32 2`, `and.b32 508`)."""
    if len(x) == 2048:
        return [x[8 * t + j] for j in range(8)] + [x[1024 + 8 * t + j] for j in range(8)]
    assert len(x) == 512
    return [x[4 * t + j] for j in range(4)]


def ptx_model(xb: list[int], wb: list[int], eps: float) -> list[int]:
    """`_rms_norm` as the PTX runs it: per thread `mul.f32 x1*x1`, then `fma.rn.f32 xk, xk, s` for k = 0, 2, 3, ..; each warp's
    butterfly 16..1 and lane 0 to shared memory; threads 0-3 butterfly 2, 1; `div.full.f32` by the power of two N (an exact
    reciprocal, one rounding); `add.f32 eps, mean`; `rsqrt.approx.ftz`; `mul.f32 r, x` then `mul.f32 (.), w`; `cvt.rn.bf16`."""
    x = [v << 16 for v in xb]
    lanes = []
    for t in range(128):
        r = _regs(x, t)
        s = _ffma(r[0], r[0], _fmul(r[1], r[1]))
        for v in r[2:]:
            s = _ffma(v, v, s)
        lanes.append(s)
    warps = [_butterfly(lanes[32 * g:32 * g + 32], _fadd, (16, 8, 4, 2, 1)) for g in range(4)]
    total = _butterfly(warps, _fadd, (2, 1))
    mean = _fmul(total, P.bits_of(np.float32(1.0 / len(x))))
    r = evaluate(P.RsqrtApprox, _fadd(P.bits_of(np.float32(eps)), mean))[0]
    return [_bf16(_fmul(_fmul(r, v), w << 16)) for v, w in zip(x, wb)]


def _bf16_row(rng, n: int, scale: float) -> np.ndarray:
    return ((rng.standard_normal(n).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(np.uint16)


def _w(rng, n: int) -> np.ndarray:
    """A norm weight row as the checkpoints' are: 1 + 0.1 N(0, 1) in bf16."""
    return ((1 + 0.1 * rng.standard_normal(n)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


OVERFLOW = 4


def _edge_rows(rng, n: int) -> list[np.ndarray]:
    """Rows at 2^-3..2^3 (served), at 2^40, with squares past f32 (row `OVERFLOW`, 2^70: the sum is inf and r = +0), squares that
    are f32 subnormals (2^-70: FFMA and FMUL round them differently), bf16 subnormals, all zeros (r = rsqrt(eps)), one outlier
    over tiny values, and +-0 / +-1 alternating."""
    out = [_bf16_row(rng, n, 2.0 ** s) for s in (-3, 0, 3)]
    out += [_bf16_row(rng, n, s) for s in (2.0 ** 40, 2.0 ** 70, 2.0 ** -70, 2.0 ** -130)]
    out.append(np.zeros(n, dtype=np.uint16))
    tiny = _bf16_row(rng, n, 2.0 ** -9)
    tiny[7] = 0x4580
    out.append(tiny)
    out.append(np.tile(np.array([0x0000, 0x8000, 0x3F80, 0xBF80], dtype=np.uint16), n // 4))
    return out


@pytest.mark.parametrize("N", [512, 2048])
def test_definition_and_twin_match_the_ptx_model(N: int) -> None:
    """The model's rationals are finite, so the overflowing row is the next test's."""
    rng = np.random.default_rng(N)
    fn, w = bind(DN.DsaRMSNorm, N=N, EPS=EPS), _w(rng, N)
    finite = [r for i, r in enumerate(_edge_rows(rng, N)) if i != OVERFLOW]
    twin = NR.dsa_rmsnorm_rows(np.stack(finite), w, EPS)["out"]
    for i, x in enumerate(finite):
        want = ptx_model(x.tolist(), w.tolist(), EPS)
        assert list(evaluate(fn, x.tolist(), w.tolist())) == want, i
        assert twin[i].tolist() == want, i


@pytest.mark.parametrize("N", [256, 512, 2048])
def test_twin_matches_the_reference_on_edge_rows(N: int) -> None:
    rng = np.random.default_rng(N + 1)
    fn, w = bind(DN.DsaRMSNorm, N=N, EPS=EPS), _w(rng, N)
    rows = _edge_rows(rng, N)
    inf = _bf16_row(rng, N, 1.0)
    inf[3], nan = 0x7F80, _bf16_row(rng, N, 1.0)
    nan[5] = 0x7FC1
    rows += [inf, nan]
    twin = NR.dsa_rmsnorm_rows(np.stack(rows), w, EPS)
    for i, x in enumerate(rows):
        assert twin["out"][i].tolist() == list(evaluate(fn, x.tolist(), w.tolist())), i
    assert twin["r"][OVERFLOW] == 0 and np.all(twin["out"][OVERFLOW] & 0x7FFF == 0)


def test_layout_is_the_ptx_loads() -> None:
    for N in (512, 2048):
        x = list(range(N))
        assert DN.layout(N) == [_regs(x, t) for t in range(128)]
    for N in (256, 512, 1024, 2048, 4096):
        assert sorted(i for regs in DN.layout(N) for i in regs) == list(range(N))
    assert DN.layout(4096)[1][8:10] == [1024 + 8, 1024 + 9]


@pytest.mark.parametrize("N,EPS_,match", [(128, EPS, "at least 256"), (384, EPS, "power-of-two"), (512, 0.0, "below 2\\^-126"),
                                          (512, 1e-39, "below 2\\^-126")])
def test_statics_are_checked(N: int, EPS_: float, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bind(DN.DsaRMSNorm, N=N, EPS=EPS_).body


@pytest.mark.parametrize("N", [512, 2048])
def test_its_scale_is_not_the_batch_invariant_triton_familys(N: int) -> None:
    """On rows at the capture's scales the two reductions and epilogues (the sum's association, and rsqrt of `eps + sum/N` against
    1 / sqrt(fma(1/N, sum, eps))) give a different r on about half the rows, a few f32 ulps apart, yet bf16 words that seldom
    differ: a word moves only where those ulps carry its product across a rounding boundary, and x and w carry 8 significant bits.
    The capture's probe (`glm53_fused_rmsnorm_capture_gpu.probe`) places weights where the words differ."""
    rng = np.random.default_rng(7 + N)
    x = np.stack([_bf16_row(rng, N, 2.0 ** rng.uniform(-3, 3)) for _ in range(64)])
    w = _w(rng, N)
    ours = NR.dsa_rmsnorm_rows(x, w, EPS)
    other = DR.rmsnorm_triton_rows(x, w, EPS)
    differ = ours["r"] != other["inv"]
    assert differ.sum() >= 16, int(differ.sum())
    gap = np.abs(ours["r"][differ].astype(np.int64) - other["inv"][differ].astype(np.int64))
    assert gap.max() <= 4 and (ours["out"] != other["out"]).mean() < 1e-3


def test_registrations() -> None:
    word = bind(DN.DsaRMSNorm, N=512, EPS=EPS)
    assert SW.boolean_version(word).id == "DsaRMSNorm_v2{N=512,EPS=1e-05}"
    assert TR.word_of(bind(BN.DsaRMSNormV2, N=512, EPS=EPS)).id == word.id
    assert RK.ROWS[NR.FAMILY].returns == ("out",) and RK.EVALUATORS[NR.FAMILY] == NR.EVALUATOR
    rng = np.random.default_rng(5)
    x, w = _bf16_row(rng, 512, 1.0), _w(rng, 512)
    got = RK.ROWS[NR.FAMILY].fn({"N": 512, "EPS": '{"f64":"' + float(EPS).hex() + '"}'}, [x, w])["out"]
    assert got.tolist() == list(evaluate(word, x.tolist(), w.tolist()))


def ptx_block_model(xb: list[int], wb: list[int], eps: float, block: int, warps: int) -> list[int]:
    """`_fused_q_kv_rmsnorm_kernel`'s program as its PTX runs it: thread t's runs of V = min(block / threads, 8) at `r * V * threads + Vt`
    (`ld.global.v4.b32`, `v2.b32`, `b32` at V = 8, 4, 2, each under `setp.lt.u32` against SIZE, a masked run +0), `mul.f32 x1*x1` then
    `fma.rn.f32 xk, xk, s` for k = 0, 2, 3, ..; each warp's butterfly 16..1 and lane 0 to shared memory; threads 0..warps-1 butterfly
    warps/2..1; `div.full.f32` by `cvt.rn.f32.u32` of SIZE (core's DivFullScaleA times DivFullRcp) with `add.f32 eps, mean`
    contracted into its multiply, as the SASS has it (`FFMA rcp, scaled sum, eps`, the exact rational rounded once);
    `rsqrt.approx.ftz`; `mul.f32 r, x` then `mul.f32 (.), w`."""
    n, threads = len(xb), 32 * warps
    V = min(block // threads, 8)
    x = [v << 16 for v in xb] + [0] * (block - len(xb))
    lanes = []
    for t in range(threads):
        r = [x[i] for run in range(0, block, V * threads) for i in range(run + V * t, run + V * t + V)]
        s = _ffma(r[0], r[0], _fmul(r[1], r[1]))
        for v in r[2:]:
            s = _ffma(v, v, s)
        lanes.append(s)
    sums = [_butterfly(lanes[32 * g:32 * g + 32], _fadd, (16, 8, 4, 2, 1)) for g in range(warps)]
    total = _butterfly(sums, _fadd, tuple(warps >> k for k in range(1, warps.bit_length())))
    size = P.bits_of(np.float32(n))
    rcp = evaluate(P.DivFullRcp, size)[0]
    r = evaluate(P.RsqrtApprox, _ffma(rcp, evaluate(P.DivFullScaleA, total, size)[0], P.bits_of(np.float32(eps))))[0]
    return [_bf16(_fmul(_fmul(r, v), w << 16)) for v, w in zip(x[:n], wb)]


@pytest.mark.parametrize("N,BLOCK,WARPS", [(1536, 2048, 8), (512, 2048, 8), (3072, 4096, 8), (768, 1024, 4), (448, 512, 4),
                                           (192, 256, 4)])
def test_block_definition_matches_the_ptx_model(N: int, BLOCK: int, WARPS: int) -> None:
    """DeepSeek-V4's q and kv rows (1536 and 512 of 2048 on 8 warps), a block of two runs a thread whose second run is masked in half
    the threads (3072 of 4096), 4 warps, and runs of 4 and 2 elements a thread."""
    rng = np.random.default_rng(N + BLOCK)
    fn, w = bind(DN.TritonBlockRMSNorm, N=N, BLOCK=BLOCK, WARPS=WARPS, EPS=1e-6), _w(rng, N)
    for i, x in enumerate(r for j, r in enumerate(_edge_rows(rng, N)) if j != OVERFLOW):
        assert list(evaluate(fn, x.tolist(), w.tolist())) == ptx_block_model(x.tolist(), w.tolist(), 1e-6, BLOCK, WARPS), i


@pytest.mark.parametrize("N", [256, 2048])
def test_dsa_rmsnorm_is_the_unmasked_four_warp_block(N: int) -> None:
    rng = np.random.default_rng(N + 2)
    w = _w(rng, N)
    for x in _edge_rows(rng, N):
        assert (list(evaluate(bind(DN.TritonBlockRMSNorm, N=N, BLOCK=N, WARPS=4, EPS=EPS), x.tolist(), w.tolist()))
                == list(evaluate(bind(DN.DsaRMSNorm, N=N, EPS=EPS), x.tolist(), w.tolist())))


@pytest.mark.parametrize("statics,match", [({"N": 128, "BLOCK": 128, "WARPS": 4}, "two elements a thread"),
                                           ({"N": 512, "BLOCK": 1536, "WARPS": 8}, "power-of-two BLOCK"),
                                           ({"N": 512, "BLOCK": 2048, "WARPS": 6}, "power-of-two warp count"),
                                           ({"N": 1540, "BLOCK": 2048, "WARPS": 8}, "whole vectors"),
                                           ({"N": 4096, "BLOCK": 2048, "WARPS": 8}, "whole vectors")])
def test_block_statics_are_checked(statics: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bind(DN.TritonBlockRMSNorm, EPS=EPS, **statics).body


def test_block_registrations() -> None:
    word = bind(DN.TritonBlockRMSNorm, N=512, BLOCK=1024, WARPS=4, EPS=EPS)
    assert SW.boolean_version(word).id == "TritonBlockRMSNorm_v2{N=512,BLOCK=1024,WARPS=4,EPS=1e-05}"
    assert TR.word_of(bind(BN.TritonBlockRMSNormV2, N=512, BLOCK=1024, WARPS=4, EPS=EPS)).id == word.id
