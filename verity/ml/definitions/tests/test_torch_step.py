"""`registry.torch_step`'s root and F.linear's three GEMMs, and `numerics.torch_step`'s models of the root's nodes, which
`program_walk` evaluates them by. Each model is held to its Definition's bits on random and special words, and the RMS norm
forward's model to an sm_120 GPU's words: four tokens of each of the five norms of a 2-layer smollm2-135m training step (B 1,
S 256, torch 2.14.0+cu130 on an RTX PRO 6000, cc 12.0) in `fixtures/torch-training-norm/norms.npz`, cut from a dump every
row of which was checked the same way. The GEMMs' model is `bf16_gemm_check.cublas_route_rows`, which `test_cublas_train.py`
holds to `train_linear`'s row GEMMs, so here each linear's body is that row GEMM over its first operand's rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import mufu as BMU
from verity.ml.definitions.numerics import program_walk as PW
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.numerics import torch_step as NS
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import torch_step as TS
from verity.ml.definitions.registry import torch_train as TT

U32 = np.uint32
SPECIALS = np.array([0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0,
                     0x7F81, 0x7FFF], U32)
NORMS = Path(__file__).parent / "fixtures" / "torch-training-norm" / "norms.npz"
EX2 = bind(BMU.MufuEx2Ftz)


def _words(rng, shape, scale: float = 1.0, raw: float = 0.01, special: float = 0.01) -> np.ndarray:
    """bf16 words: normal values times 2^-6 .. 2^6 times ``scale``, with fractions ``raw`` of raw words and ``special`` of
    `SPECIALS`."""
    n = int(np.prod(shape))
    v = (rng.standard_normal(n) * scale * np.exp2(rng.integers(-6, 7, n))).astype(np.float32)
    out = TK.bf16(v.view(U32))
    pick = rng.random(n)
    out = np.where(pick < raw, rng.integers(0, 1 << 16, n).astype(U32), out)
    return np.where(pick > 1 - special, SPECIALS[rng.integers(0, len(SPECIALS), n)], out).astype(U32).reshape(shape)


def _model(f, *args):
    return NS.MODELS[PW.family(f)](dict(f.bindings), *args)


def _same(got, want) -> None:
    got, want = np.asarray(got, np.uint64), np.asarray(want, np.uint64).reshape(np.shape(got))
    bad = np.argwhere(got != want)
    assert not bad.size, f"{len(bad)} of {got.size} differ, first at {tuple(bad[0])}: {got[tuple(bad[0])]:#x} != {want[tuple(bad[0])]:#x}"


# --- the root --------------------------------------------------------------------------------------------------------------

def test_the_root_takes_every_parameter_and_returns_the_loss_and_each_gradient() -> None:
    params, ret = bind(TS.TrainStep, L=2, B=1, S=256).signature()
    names = [n for n, _ in params]
    assert names[:5] == ["ids", "labels", "n", "cs", "embed.w"] and names[-1] == "norm.w"
    assert names[5:-1] == [f"l{i}.{k}.w" for i in range(2) for k in ("q", "k", "v", "o", "gate", "up", "down", "ln1", "ln2")]
    assert ret.fields[0] == TS.F32 and list(ret.fields[1:]) == [t for _, t in params[4:]]
    assert dict(params)["embed.w"] == TS.Array(49152, TS.Array(576, TS.BF16))


@pytest.mark.parametrize("statics", [{"L": 2, "B": 2, "S": 256}, {"L": 0, "B": 1, "S": 256}, {"L": 1, "B": 1, "S": 4096},
                                     {"L": 1, "B": True, "S": 256}])
def test_the_root_refuses_a_shape_the_capture_does_not_cover(statics) -> None:
    with pytest.raises(ValueError):
        bind(TS.TrainStep, **statics).signature()


@pytest.mark.parametrize("D,role", [(TS.LinearTorch, "fwd"), (TS.LinearTorchDx, "dX"), (TS.LinearTorchDw, "dW")])
def test_each_linear_is_its_row_gemm_over_the_first_operands_rows(D, role: str) -> None:
    """At (K, N) = (576, 1536), where the dX GEMM takes cuBLAS's 8-way split-K with bf16 partials and the others the unsplit
    chain; the second operand is shared by every row."""
    M, K, N = 256, 576, 1536
    f = bind(D, M=M, K=K, N=N)
    (node,) = f.body.nodes
    assert node.form == "batch" and node.n == (N if role == "dW" else M) and tuple(node.axes) == (0, None)
    assert node.fn.id == CB.train_linear(role, M, K, N).id
    with pytest.raises(ValueError):
        bind(D, M=300, K=K, N=N).signature()


def test_every_model_is_keyed_by_its_family() -> None:
    fs = [bind(TS.LinearTorch, M=256, K=576, N=192), bind(TS.LinearTorchDx, M=256, K=576, N=192),
          bind(TS.LinearTorchDw, M=256, K=576, N=192), bind(TT.Transpose, R=3, C=5), EW.Bf16AddF2fp, EW.Bf16MulBf16,
          bind(TT.RmsNormTorch, H=576, LANES=32, EPS=TS.EPS, RSQRT=bind(BMU.RsqrtApprox)),
          bind(TT.RmsNormTorchBackward, H=576, LANES=32), bind(TT.RmsNormTorchWeightGrad, M=256, H=576),
          bind(TT.SiluTorch, N=TS.ACT_N, EX2=EX2), bind(TT.SiluTorchBackward, N=TS.ACT_N, EX2=EX2, CONTRACT=1)]
    assert sorted(PW.family(f) for f in fs) == sorted(NS.MODELS)


def test_a_model_refuses_a_unit_it_does_not_model() -> None:
    f = bind(TT.SiluTorch, N=8, EX2=bind(BMU.RsqrtApprox))
    with pytest.raises(ValueError):
        _model(f, np.zeros((1, 8), U32))


# --- the models against bits -----------------------------------------------------------------------------------------------

def test_the_transpose_model_is_the_bits() -> None:
    rng = np.random.default_rng(3)
    for R, C in ((3, 5), (128, 64)):
        x = rng.integers(0, 1 << 16, (3, R, C)).astype(U32)
        f = bind(TT.Transpose, R=R, C=C)
        _same(_model(f, x), BS.evaluate_words(f, x))


@pytest.mark.parametrize("f", [EW.Bf16AddF2fp, EW.Bf16MulBf16])
def test_the_residual_models_are_the_bits(f) -> None:
    f, rng = bind(f), np.random.default_rng(7)
    a, b = _words(rng, 4096, raw=0.1, special=0.05), _words(rng, 4096, raw=0.1, special=0.05)
    _same(_model(f, a, b), BS.evaluate_words(f, a, b))


def _norm_rows(rng, n: int, H: int):
    """n tokens (x, w) at training's scales, a fifth of them scaled by 2^-60 .. 2^60, with a few raw and special words."""
    x = _words(rng, (n, H), raw=0.002, special=0.002)
    scale = np.where(rng.random(n) < 0.2, np.exp2(rng.integers(-60, 61, n)), 1.0).astype(np.float32)
    x = np.where(scale[:, None] != 1, TK.bf16(TK.mul(TK.f32(x), scale.view(U32)[:, None])), x).astype(U32)
    w = TK.bf16_add(_words(rng, (n, H), 0.05, raw=0.002, special=0.002), U32(0x3F80))
    return x, w


@pytest.mark.slow
def test_the_rms_forward_model_is_the_bits() -> None:
    """H = 160 at 32 lanes, where lanes 0-7 read two float4s and the others one; tracing the Definition takes about a minute."""
    rng = np.random.default_rng(11)
    x, w = _norm_rows(rng, 96, 160)
    f = bind(TT.RmsNormTorch, H=160, LANES=32, EPS=TS.EPS, RSQRT=bind(BMU.RsqrtApprox))
    y, hb, r = _model(f, x, w)
    _same(np.concatenate([y, hb, r[:, None]], axis=1), BS.evaluate_words(f, x, w).reshape(96, -1))


@pytest.mark.slow
def test_the_rms_backward_model_is_the_bits() -> None:
    """H = 160 at 32 lanes, r the forward's saved rsqrt in most tokens and a raw or special word in some."""
    rng = np.random.default_rng(13)
    x, w = _norm_rows(rng, 96, 160)
    dy = _words(rng, (96, 160), 1e-3, raw=0.002, special=0.002)
    r = NS.rms_forward(x, w, 32, TS.EPS)[2]
    odd = rng.random(96)
    r = np.where(odd < 0.05, rng.integers(0, 1 << 32, 96, dtype=np.uint64).astype(U32), r)
    r = np.where(odd > 0.95, np.array([0, 0x7F800000, 0x7FC00000, 1], U32)[rng.integers(0, 4, 96)], r).astype(U32)
    f = bind(TT.RmsNormTorchBackward, H=160, LANES=32)
    _same(_model(f, dy, x, w, r), BS.evaluate_words(f, dy, x, w, r))


@pytest.mark.slow
@pytest.mark.parametrize("M", [100, 1100])
def test_the_weight_grad_model_is_the_bits(M: int) -> None:
    """96 columns at 100 tokens (one CTA, the tail loop) and 1,100 (18 CTAs with a tail, the cross-CTA sum, whose trace takes
    about a minute)."""
    rng = np.random.default_rng(M)
    dy = _words(rng, (96, M), 1e-3, raw=0.02, special=0.02)
    hb = _words(rng, (96, M), raw=0.02, special=0.02)
    f = bind(TT.RmsNormTorchWeightGrad, M=M, H=576)
    _same(_model(f, dy, hb), BS.evaluate_words(f, dy, hb))


@pytest.mark.slow
def test_the_silu_models_are_the_bits() -> None:
    """Forward at N = 8 and backward at N = 2 under both contractions, on random and special words."""
    rng = np.random.default_rng(17)
    x = _words(rng, (512, 8), 4.0, raw=0.1, special=0.05)
    f = bind(TT.SiluTorch, N=8, EX2=EX2)
    _same(_model(f, x), BS.evaluate_words(f, x))
    x, dy = _words(rng, (1024, 2), 4.0, raw=0.1, special=0.05), _words(rng, (1024, 2), 1e-3, raw=0.1, special=0.05)
    for contract in (0, 1):
        f = bind(TT.SiluTorchBackward, N=2, EX2=EX2, CONTRACT=contract)
        _same(_model(f, dy, x), BS.evaluate_words(f, dy, x))


# --- the GPU ---------------------------------------------------------------------------------------------------------------

def test_the_rms_forward_model_is_the_gpus() -> None:
    """Four tokens of each of the step's five norms at H = 576 (32 lanes): the saved rsqrt r, the normalized bf16 hb and the
    output y, every word equal; and the norms' eps, 1e-5, is `torch_step.EPS`."""
    assert TS.EPS == int(np.float32(1e-5).view(U32))
    with np.load(NORMS) as z:
        for i in range(5):
            x, w = z[f"n{i}_x"].astype(U32), np.broadcast_to(z[f"n{i}_w"].astype(U32), z[f"n{i}_x"].shape)
            y, hb, r = NS.rms_forward(x, w, TT.reduce_lanes(576, 256), TS.EPS)
            _same(r, z[f"n{i}_r"])
            _same(hb, z[f"n{i}_hb"])
            _same(y, z[f"n{i}_y"])
