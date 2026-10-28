"""[GEN dense / ov-dense] the Gemma-2 patterns (`profiles/gen_dense_gemma2_patterns.py`) on a synthetic step that replays
the exact ATen sequence of vLLM d9105ea80's eager `Gemma2ForCausalLM` (see the module docstring there): every root resolves
to the intended `quarantine.dense` specialization, the widened-bf16 operands are re-pointed to their narrow sources, the
Triton `mean_kernel` child is consumed by the `aten.mean.dim` compute, and the pins reject a foreign launch geometry."""
from __future__ import annotations

import importlib

import dataclasses

import numpy as np
import pytest

from verity_vllm.observe.events import StepEnd, ViewDesc
from verity_vllm.observe.fold.patterns import BiasAdd, TransparentAten
from verity_vllm.observe.fold.profiles import vllm_d9105ea80_sm89_eager as B1PROF
from verity_vllm.observe.fold.profiles.gen_dense_gemma2_patterns import (Bf16ScalarChain, MeanDimTriton, NarrowToBf16, gemma2_patterns,
                                                                with_gemma2_patterns)

gemma = importlib.import_module("verity_vllm.observe.fold.profiles.derived_GEMMA2_2B")
from verity_vllm.observe.fold.resolver import Compute, PerRow, Shared, Unsupported, resolve_step, spec_module
from tests.observe.test_patterns_synthetic import CUDA, Builder, bf, by_spec, context, decode_step, i64
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.program.quarantine.dense import ALL_DEFINITIONS, ALL_PRIMITIVES

M, H, I, V = 2, 8, 4, 16
ADD = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
POW = "aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor"
MEAN = "aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor"
RSQRT = "aten::rsqrt(Tensor self) -> Tensor"
MUL = "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor"
TO = "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)"
DIV = "aten::div.Tensor(Tensor self, Tensor other) -> Tensor"
TANH = "aten::tanh(Tensor self) -> Tensor"
GELU = "_C::gelu_tanh_and_mul(Tensor! out, Tensor input) -> ()"
A_W, A_NORMALIZER = 900, 901


def f32(alloc, shape):
    strides = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
    return ViewDesc(alloc, 0, tuple(shape), strides, "float32", CUDA)


class Alloc:
    def __init__(self, start=2000):
        self.n = start

    def __call__(self):
        self.n += 1
        return self.n


def gemma_norm(b: Builder, al: Alloc, x, w_root, residual=None, module="model.layers.0.input_layernorm", block=1024, warps=4, k_n=None):
    """The eager GemmaRMSNorm ATen chain on x[M,H] bf16 (with the residual form when `residual` is given); returns the bf16 output
    and, for the residual form, the new bf16 residual."""
    w32 = f32(al(), (H,))
    b.leaf("aten.to.dtype", TO, [w_root, "torch.float32"], outs=w32, module=module)
    w1 = f32(al(), (H,))
    b.leaf("aten.add.Tensor", ADD, [w32, 1.0], outs=w1, module=module)
    x32 = f32(al(), (M, H))
    b.leaf("aten.to.dtype", TO, [x, "torch.float32"], outs=x32, module=module)
    new_res = None
    if residual is not None:
        r32 = f32(al(), (M, H))
        b.leaf("aten.to.dtype", TO, [residual, "torch.float32"], outs=r32, module=module)
        s32 = f32(al(), (M, H))
        b.leaf("aten.add.Tensor", ADD, [x32, r32], outs=s32, module=module)
        new_res = bf(al(), (M, H))
        b.leaf("aten.to.dtype", TO, [s32, "torch.bfloat16"], outs=new_res, module=module)
        x32 = s32
    sq = f32(al(), (M, H))
    b.leaf("aten.pow.Tensor_Scalar", POW, [x32, 2], outs=sq, module=module)
    var = f32(al(), (M, 1))
    mean = b.call("aten.mean.dim", MEAN, [sq, [-1], True], module=module)
    b.triton("vllm.model_executor.determinism.batch_invariant.mean_kernel",
             {"input_ptr": ViewDesc(sq.alloc, 0, (M, H, 1), (H, 1, 1), "float32", CUDA), "output_ptr": var, "input_stride0": H, "input_stride1": 1,
              "input_stride2": 1, "output_stride0": 1, "output_stride1": 1, "M": M, "N": k_n if k_n is not None else H, "K": 1, "BLOCK_SIZE": block},
             {"BLOCK_SIZE": block}, [M], warps, 3, parent=mean)
    b.close(mean, var)
    ve = f32(al(), (M, 1))
    b.leaf("aten.add.Tensor", ADD, [var, 1e-6], outs=ve, module=module)
    r = f32(al(), (M, 1))
    b.leaf("aten.rsqrt.default", RSQRT, [ve], outs=r, module=module)
    xs = f32(al(), (M, H))
    b.leaf("aten.mul.Tensor", MUL, [x32, r], outs=xs, module=module)
    b.leaf("aten.to.dtype", TO, [xs, "torch.float32"], outs=xs, module=module)          # returned self
    xw = f32(al(), (M, H))
    b.leaf("aten.mul.Tensor", MUL, [xs, w1], outs=xw, module=module)
    out = bf(al(), (M, H))
    b.leaf("aten.to.dtype", TO, [xw, "torch.bfloat16"], outs=out, module=module)
    return out, new_res


def gemma_step(block=1024, warps=4, k_n=None):
    events, start = decode_step()
    start = dataclasses.replace(start)
    b = Builder(1)
    al = Alloc()
    e = bf(al(), (M, H))
    scaled = bf(al(), (M, H))
    b.leaf("aten.mul.Tensor", MUL, [e, ViewDesc(A_NORMALIZER, 0, (), (), "bfloat16", CUDA)], outs=scaled)
    w = bf(A_W, (H,))
    n1, _ = gemma_norm(b, al, scaled, w, block=block, warps=warps, k_n=k_n)
    n2, res = gemma_norm(b, al, n1, w, residual=scaled, module="model.layers.1.input_layernorm")
    gu = bf(al(), (M, 2 * I))
    act = bf(al(), (M, I))
    b.leaf("_C.gelu_tanh_and_mul.default", GELU, [act, gu], module="model.layers.0.mlp.act_fn")
    logits = bf(al(), (M, V))
    d = bf(al(), (M, V))
    b.leaf("aten.div.Tensor", DIV, [logits, 30.0], outs=d, phase="sample", module="logits_processor")
    t = bf(al(), (M, V))
    b.leaf("aten.tanh.default", TANH, [d], outs=t, phase="sample", module="logits_processor")
    o = bf(al(), (M, V))
    b.leaf("aten.mul.Tensor", MUL, [t, 30.0], outs=o, phase="sample", module="logits_processor")
    return [start, *b.events, StepEnd(1)], start, {"e": e, "scaled": scaled, "w": w, "n1": n1, "res": res, "act": act, "gu": gu, "o": o}


def gemma_context(start):
    from verity_vllm.observe.events import Root
    ctx = context(start, None)
    roots = dict(ctx.roots)
    roots[A_W] = [Root(bf(A_W, (H,)), "weight", "model.layers.0.input_layernorm.weight")]
    roots[A_NORMALIZER] = [Root(ViewDesc(A_NORMALIZER, 0, (), (), "bfloat16", CUDA), "buffer", "model.normalizer")]
    prof = dataclasses.replace(ctx.profile, patterns=with_gemma2_patterns(ctx.profile.patterns))
    return dataclasses.replace(ctx, roots=roots, profile=prof)


def test_patterns_are_inserted_ahead_of_bias_add_and_transparent_aten():
    ps = with_gemma2_patterns(B1PROF.PROFILE.patterns)
    kinds = [type(p) for p in ps]
    first = kinds.index(type(gemma2_patterns()[0]))
    assert kinds.index(BiasAdd) == first + len(gemma2_patterns()) and kinds.index(TransparentAten) > kinds.index(NarrowToBf16)
    assert len(ps) == len(B1PROF.PROFILE.patterns) + len(gemma2_patterns())
    assert [type(p) for p in gemma.PROFILE.patterns if type(p) in {MeanDimTriton, Bf16ScalarChain}] == [MeanDimTriton, Bf16ScalarChain]
    assert len({p.name for p in gemma2_patterns()}) == len(gemma2_patterns())


def test_every_gemma_root_resolves_to_a_quarantine_dense_compute():
    events, start, v = gemma_step()
    se = step_events(events)
    res = resolve_step(se, gemma_context(start))
    assert not [r for r in res if isinstance(r, Unsupported)], [r.reason for r in res if isinstance(r, Unsupported)]
    specs = by_spec(res)
    expected = {
        f"Bf16MulScalarTensor_v1{{N={H}}}": 1, f"AddScalarBf16_v1{{N={H},C=1.0}}": 2, f"SquareBf16_v1{{N={H}}}": 1, f"SquareF32_v1{{N={H}}}": 1,
        f"MeanTriton_v1{{N={H}}}": 2, "AddScalarF32_v1{N=1,C=1e-06}": 2, "RsqrtF32_v1{N=1}": 2, f"ScaleRowBf16_v1{{N={H}}}": 1,
        f"ScaleRowF32_v1{{N={H}}}": 1, f"MulVecF32_v1{{N={H}}}": 2, f"NarrowF32ToBf16_v1{{N={H}}}": 3, f"AddWidenedBf16_v1{{N={H}}}": 1,
        f"GeluTanhMul_v1{{I={I}}}": 1, f"Bf16DivScalar_v1{{N={V},C=30.0}}": 1, f"Bf16Tanh_v1{{N={V}}}": 1, f"Bf16MulScalar_v1{{N={V},C=30.0}}": 1,
    }
    assert {k: len(c) for k, c in specs.items()} == expected
    for k, cs in specs.items():
        for c in cs:
            # Bf16MulScalarTensor is in the accepted registry (b1); the rest in
            # `verity.ml.definitions.registry.dense` (the quarantine modules re-export the same objects: same ids, same bodies)
            assert spec_module(c.spec) == "verity.ml.definitions.registry.dense" or (
                k.startswith("Bf16MulScalarTensor_v1") and spec_module(c.spec) == "verity.ml.definitions.registry.b1"), (k, spec_module(c.spec))
            assert c.rows == (1 if k.startswith("AddScalarBf16") else M)
    # the widened operands read their bf16 sources (never the f32 copies) ...
    (sq,) = specs[f"SquareBf16_v1{{N={H}}}"]
    assert sq.operands == (PerRow(v["scaled"]),)
    (aw,) = specs[f"AddWidenedBf16_v1{{N={H}}}"]
    assert aw.operands == (PerRow(v["n1"]), PerRow(v["scaled"]))
    (sr,) = specs[f"ScaleRowBf16_v1{{N={H}}}"]
    assert sr.operands[0] == PerRow(v["scaled"])
    # ... the weight (+1) is a one-row Shared compute over the bf16 Root, the normalizer a Shared 0-d operand
    for c in specs[f"AddScalarBf16_v1{{N={H},C=1.0}}"]:
        assert c.operands == (Shared(v["w"]),) and isinstance(c.outputs[0], Shared)
    (es,) = specs[f"Bf16MulScalarTensor_v1{{N={H}}}"]
    assert es.operands == (PerRow(v["e"]), Shared(ViewDesc(A_NORMALIZER, 0, (), (), "bfloat16", CUDA))) and "model.normalizer" in es.hint
    # the mean compute consumes the dispatcher call AND its Triton child
    for c in specs[f"MeanTriton_v1{{N={H}}}"]:
        assert len(c.consumed) == 2 and c.kernels == ("mean_kernel",)
    (g,) = specs[f"GeluTanhMul_v1{{I={I}}}"]
    assert g.operands == (PerRow(v["gu"]),) and g.outputs == (PerRow(v["act"]),)


def test_under_registered_values_gelu_binds_v1_and_the_scalar_constants_are_refused_by_name():
    """A `registered_values` Build calls `GeluTanhMul_v1` as the constants Build does (its Boolean version computes the activation,
    `boolean_tanh`, and reads no table), so the fold binds it over the same operands. The Build binds the scalar ops' value twins,
    which this fold does not, so it refuses each scalar add, divide and multiply by name rather than bind a constant form."""
    events, start, v = gemma_step()
    ctx = gemma_context(start)
    ctx = dataclasses.replace(ctx, profile=dataclasses.replace(ctx.profile, registered_values=True))
    res = resolve_step(step_events(events), ctx)
    (g,) = by_spec(res)[f"GeluTanhMul_v1{{I={I}}}"]
    assert g.operands == (PerRow(v["gu"]),) and g.outputs == (PerRow(v["act"]),)
    refused = [r for r in res if isinstance(r, Unsupported)]
    assert all("registered_values" in r.reason and "value twin" in r.reason for r in refused), [r.reason for r in refused]
    assert sorted(r.detail["spec"].split("{")[0] for r in refused) == ["AddScalarBf16_v1"] * 2 + ["AddScalarF32_v1"] * 2 + [
        "Bf16DivScalar_v1", "Bf16MulScalar_v1"]
    assert not [k for k in by_spec(res) if k.split("{")[0] in ("AddScalarBf16_v1", "AddScalarF32_v1", "Bf16DivScalar_v1", "Bf16MulScalar_v1")]


@pytest.mark.parametrize("over, needle", [({"block": 512}, "BLOCK_SIZE"), ({"warps": 8}, "num_warps"), ({"k_n": 7}, "N")])
def test_mean_kernel_pins_reject_a_foreign_launch(over, needle):
    events, start, _ = gemma_step(**over)
    se = step_events(events)
    res = resolve_step(se, gemma_context(start))
    uns = [r for r in res if isinstance(r, Unsupported)]
    assert len(uns) == 1 and uns[0].op == "aten.mean.dim" and needle in uns[0].detail["mismatches"]


def test_quarantine_dense_registry_is_the_pattern_target_set():
    ids = {d.id.split("{")[0] for d in ALL_DEFINITIONS}
    used = {"Bf16MulScalarTensor_v1", "AddScalarBf16_v1", "AddScalarF32_v1", "SquareBf16_v1", "SquareF32_v1", "MeanTriton_v1", "RsqrtF32_v1",
            "ScaleRowBf16_v1", "ScaleRowF32_v1", "MulVecF32_v1", "NarrowF32ToBf16_v1", "AddWidenedBf16_v1", "GeluTanhMul_v1", "Bf16DivScalar_v1",
            "Bf16Tanh_v1", "Bf16MulScalar_v1"}
    assert used == ids, ids ^ used
    assert {p.id.split("{")[0] for p in ALL_PRIMITIVES} == {"TanhF32Rn_v1", "GeluTanhMulBf16_v1"}
