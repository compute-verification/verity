"""`registry.torch_step`'s root and F.linear's three GEMMs, and `numerics.torch_step`'s models of the root's nodes, which
`program_walk` evaluates them by. Each model is held to its Definition's bits on random and special words, and the RMS norm
forward's model to an sm_120 GPU's words: four tokens of each of the five norms of a 2-layer smollm2-135m training step (B 1,
S 256, torch 2.14.0+cu130 on an RTX PRO 6000, cc 12.0) in `fixtures/torch-training-norm/norms.npz`, cut from a dump every
row of which was checked the same way. The GEMMs' model is `bf16_gemm_check.cublas_route_rows`, which `test_cublas_train.py`
holds to `train_linear`'s row GEMMs, so here each linear's body is that row GEMM over its first operand's rows, and its model is
its bits on each route kind at a small shape. The optimizer's
models (the gradient's widening, AdamW on runs of f32 words, the bf16 copy) are held to their bits the same way."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import cast as CA
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import mufu as BMU
from verity.ml.definitions.numerics import program_walk as PW
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.numerics import torch_step as NS
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity.ml.definitions.registry import torch_optim as TO
from verity.ml.definitions.registry import torch_step as TS
from verity.ml.definitions.registry import torch_train as TT

U32 = np.uint32
SPECIALS = np.array([0, 0x8000, 1, 0x8001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0,
                     0x7F81, 0x7FFF], U32)
F32_SPECIALS = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x3F000000, 0x3EFFFFFF,
                         0xBF000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001,
                         0x7FFFFFFF, 0x3F7F8000, 0x3F7FC000, 0x7F7F8000], U32)


def _recipe_k(t: int) -> list[int]:
    """AdamW's seven f32 scalars at lr 3e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.1 and step t, as CPython makes them."""
    lr, b1, b2, eps, wd = 3e-4, 0.9, 0.95, 1e-8, 0.1
    vals = [1 - lr * wd, 1 - b1, b2, 1 - b2, (1 - b2 ** t) ** 0.5, eps, (lr / (1 - b1 ** t)) * -1]
    return [int(np.float32(x).view(U32)) for x in vals]


TO_K = [_recipe_k(t) for t in (1, 2, 3)]
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

def test_the_root_takes_the_weights_and_their_optimizer_state_and_returns_the_new_state() -> None:
    params, ret = bind(TS.TrainStep, L=2, B=1, S=256).signature()
    names, types = [n for n, _ in params], dict(params)
    ws = ["embed", *(f"l{i}.{k}" for i in range(2) for k in ("q", "k", "v", "o", "gate", "up", "down", "ln1", "ln2")), "norm"]
    assert [w for w, _, _ in TS.weights(2)] == ws
    assert names == (["ids", "labels", "n", "cs"] + [f"{w}.w" for w in ws]
                     + [f"{w}.{s}" for w in ws for s in ("master", "exp_avg", "exp_avg_sq")]
                     + ["step", "lr", "beta1", "beta2", "eps", "wd", "max_norm"])
    assert types["embed.w"] == TS.Array(49152, TS.Array(576, TS.BF16))
    assert types["l1.down.exp_avg"] == TS.Array(576, TS.Array(1536, TS.F32)) and types["norm.master"] == TS.Array(576, TS.F32)
    assert types["step"] == TS.F32 and types["max_norm"] == TS.F64
    assert list(ret.fields[:4]) == [TS.F32, TS.BF16, TS.F32, BL.BIT]
    assert list(ret.fields[4:]) == [types[f"{w}.{s}"] for w in ws for s in ("w", "master", "exp_avg", "exp_avg_sq")]


@pytest.mark.parametrize("statics", [{"L": 2, "B": 2, "S": 256}, {"L": 0, "B": 1, "S": 256}, {"L": 1, "B": 1, "S": 4096},
                                     {"L": 1, "B": True, "S": 256}])
def test_the_root_refuses_a_shape_the_capture_does_not_cover(statics) -> None:
    with pytest.raises(ValueError):
        bind(TS.TrainStep, **statics).signature()


def test_the_roots_constants_are_the_ignore_index_and_the_loss_seed() -> None:
    k = bind(TS.TrainStepConstants)
    probe = CompositeDefinition("TrainStepConstantsProbe", 1, (), lambda S: ((("x", BL.BIT),), k.signature()[1]),
                                lambda B, S, x: B.call(k), register=False)
    assert BS.evaluate_words(bind(probe), np.zeros((1, 1), U32)).reshape(-1).tolist() == [TS.IGNORE, TS.ONE]
    assert TS.IGNORE == -100 & 0xFFFFFFFF and TS.ONE == np.float32(1).view(U32)


LINEARS = [(TS.LinearTorch, "fwd"), (TS.LinearTorchDx, "dX"), (TS.LinearTorchDw, "dW")]
#: a linear's (M, K, N) whose row GEMM is (K 64, N 3), two rows of it (`cublas_bf16.TRAIN_LINEAR_ROLES`)
SMALL = {"fwd": (2, 64, 3), "dX": (2, 3, 64), "dW": (64, 3, 2)}
#: the route kinds the capture takes, at K 64
SMALL_ROUTES = [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 4},
                {"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16}]


@pytest.mark.parametrize("D,role", LINEARS)
def test_each_linear_is_its_row_gemm_over_the_first_operands_rows(D, role: str) -> None:
    """At (K, N) = (576, 1536), where the dX GEMM takes cuBLAS's 8-way split-K with bf16 partials and the others the unsplit
    chain; the second operand is shared by every row."""
    M, K, N = 256, 576, 1536
    f = TS._linear(D, role, M, K, N)
    (node,) = f.body.nodes
    assert node.form == "batch" and node.n == (N if role == "dW" else M) and tuple(node.axes) == (0, None)
    assert node.fn.id == CB.train_linear(role, M, K, N).id
    with pytest.raises(ValueError):
        TS._linear(D, role, 300, K, N)


@pytest.mark.parametrize("D,role", LINEARS)
@pytest.mark.parametrize("route", [{"kernel": "cublaslt"}, {"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8},
                                   {"kernel": "serialk", "SPLIT_K": 128}, "nosplit"])
def test_a_linear_refuses_a_route_with_no_row_gemm_on_bits(D, role: str, route) -> None:
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(D, **dict(zip("MKN", SMALL[role])), ROUTE=route)


@pytest.mark.parametrize("D,role", LINEARS)
@pytest.mark.parametrize("route", SMALL_ROUTES, ids=lambda r: "-".join(map(str, r.values())))
def test_each_linear_model_is_the_bits_on_each_route(D, role: str, route) -> None:
    M, K, N = SMALL[role]
    f = bind(D, M=M, K=K, N=N, ROUTE=route)
    (_, a), (_, b) = f.signature()[0]
    rng = np.random.default_rng(5)
    x = _words(rng, (8, a.n, a.elem.n), raw=0.02, special=0.02)
    w = _words(rng, (8, b.n, b.elem.n), raw=0.02, special=0.02)
    _same(_model(f, x, w), BS.evaluate_words(f, x, w))


def test_every_model_is_keyed_by_its_family() -> None:
    fs = [TS._linear(D, role, 256, 576, 192) for D, role in LINEARS] + [
          bind(TT.Transpose, R=3, C=5), EW.Bf16AddF2fp, EW.Bf16MulBf16,
          bind(TT.RmsNormTorch, H=576, LANES=32, EPS=TS.EPS, RSQRT=bind(BMU.RsqrtApprox)),
          bind(TT.RmsNormTorchBackward, H=576, LANES=32), bind(TT.RmsNormTorchWeightGrad, M=256, H=576),
          bind(TT.SiluTorch, N=TS.ACT_N, EX2=EX2), bind(TT.SiluTorchBackward, N=TS.ACT_N, EX2=EX2, CONTRACT=1),
          CA.Bf16ToF32, BG.F2fpBf16, bind(TO.AdamWForeach, N=TS.ADAM_N, LERP_FMA=1, BF16=0)]
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


def test_the_master_grad_and_copy_models_are_the_bits() -> None:
    """The widening on every bf16 word, and cvt.rn.bf16.f32 on random, raw and special f32 words (NaN 0x7FFF)."""
    rng = np.random.default_rng(19)
    h = np.arange(1 << 16, dtype=U32)
    f = bind(CA.Bf16ToF32)
    _same(_model(f, h), BS.evaluate_words(f, h))
    u = np.concatenate([TK.f32(_words(rng, 4096, raw=0.1, special=0.1)) ^ rng.integers(0, 1 << 16, 4096).astype(U32),
                        rng.integers(0, 1 << 32, 4096, dtype=np.uint64).astype(U32), F32_SPECIALS]).astype(U32)
    f = bind(BG.F2fpBf16)
    _same(_model(f, u), BS.evaluate_words(f, u))


def _f32_words(rng, shape, scale: float = 1.0) -> np.ndarray:
    """f32 words: normal values over 2^-40 .. 2^10 times ``scale``, with a tenth raw words and a tenth `F32_SPECIALS`."""
    n = int(np.prod(shape))
    v = (rng.standard_normal(n) * scale * np.exp2(rng.integers(-40, 10, n))).astype(np.float32).view(U32)
    pick = rng.random(n)
    v = np.where(pick < 0.1, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32), v)
    return np.where(pick > 0.9, F32_SPECIALS[rng.integers(0, len(F32_SPECIALS), n)], v).astype(U32).reshape(shape)


@pytest.mark.slow
@pytest.mark.parametrize("lerp_fma,bf16", [(0, 1), (1, 0)])
def test_the_adamw_model_is_the_bits(lerp_fma: int, bf16: int) -> None:
    """Runs of 4 elements: the recipe's scalars at steps 1 to 3 in most runs, scalar words sampled at random or special in the
    rest (the lerp's two branches, c2 at 1.0 and not), on random, raw and special state words."""
    rng = np.random.default_rng(23 + lerp_fma)
    R, N = 512, 4
    k = np.array([TO_K[int(s)] for s in rng.integers(0, 3, R)], U32)
    wild = rng.random(k.shape) < 0.15
    k = np.where(wild, _f32_words(rng, k.shape), k).astype(U32)
    k[rng.random(R) < 0.05, 3] = 0x3F800000
    p, m = _f32_words(rng, (R, N)), _f32_words(rng, (R, N), 1e-3)
    g, v = _f32_words(rng, (R, N), 1e-3), np.abs(_f32_words(rng, (R, N), 1e-6).view(np.float32)).view(U32)
    f = bind(TO.AdamWForeach, N=N, LERP_FMA=lerp_fma, BF16=bf16)
    _same(np.concatenate(_model(f, p, g, m, v, k), axis=1), BS.evaluate_words(f, p, g, m, v, k).reshape(R, -1))


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
    """The backward model's one check against the Definition's bits: H = 160 at 32 lanes reaches both of the row sum's lane
    paths (lanes 0-7 read two float4s, the others one), with r the forward's saved rsqrt in most tokens and a raw or special
    word in some. Tracing the Definition takes over a minute."""
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
    """96 columns at 100 tokens (one CTA, the tail loop) and 1,100 (18 CTAs with a tail). Only more than one CTA reaches the
    cross-CTA sum, and its trace takes about two minutes."""
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
