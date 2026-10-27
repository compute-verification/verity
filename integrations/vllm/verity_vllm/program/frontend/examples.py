"""Authored examples for the torch frontend (acceptance material, frontend design §5–§6).

Part 1 re-authors composites of TA1's registry (`verity/ml/definitions/registry/b1.py`) in PyTorch. Their `Program`
digests must equal the digests of TA1's hand-built definitions (same `Name_vN{statics}`, same parameter names,
same body nodes and reference encodings). Part 2 is IFb's identity shapes (repeated argument, returned input,
dead declared return). Part 3 is the fail-closed negatives.
"""

from __future__ import annotations

import numpy as np
import torch

from verity.ml.circuits.defs import CompositeDefinition, bind as ir_bind
from verity.ml.circuits.refs import tuple_of
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry.prims import BF16, F32, I32
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.program.frontend.schemas import Arr, Tup
from verity_vllm.program.frontend.torch_frontend import batch, function, ops, scan

# ---------------------------------------------------------------------------------------------------------
# 1. TA1 composites, authored in torch
# ---------------------------------------------------------------------------------------------------------


@function(static=("K",))
def DotBf16(x: Arr["K", BF16], w: Arr["K", BF16], *, K) -> F32:
    """f32 accumulator after K/16 `AmpereBF16TcDot16` steps in ascending k from a zero accumulator."""
    acc = ops.const(32, 0)
    for s in range(K // 16):
        acc = ops.AmpereBF16TcDot16(acc, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16])
    return acc


@function(static=("K",))
def GemmCoordinate(x: Arr["K", BF16], wrow: Arr["K", BF16]) -> BF16:
    return ops.F2fpBf16(DotBf16(x, wrow))


@function(static=("K", "N"))
def Gemm(x: Arr["K", BF16], w: Arr["N", "K", BF16]) -> Arr["N", BF16]:
    return batch(GemmCoordinate, x, w, axes=(None, 0))


@function(static=("N",))
def BiasAdd(y: Arr["N", BF16], b: Arr["N", BF16]) -> Arr["N", BF16]:
    return batch(ops.Bf16Add, y, b, axes=(0, 0))


@function(static=("I",))
def SiluMul(gu: Arr["2*I", BF16], *, I) -> Arr["I", BF16]:
    return batch(ops.SiluMulBf16, gu[0:I], gu[I:2 * I], axes=(0, 0))


@function(static=("D",))
def RoPEHead(x: Arr["D", BF16], cs: Arr["D", BF16], *, D) -> Arr["D", BF16]:
    R = D // 2
    cos, sin = cs[0:R], cs[R:D]
    lo = batch(ops.RopeOut, x[0:R], x[R:D], cos, sin, axes=(0, 0, 0, 0))
    hi = batch(ops.RopeOutAdd, x[0:R], x[R:D], cos, sin, axes=(0, 0, 0, 0))
    return torch.cat([lo, hi])


@function(static=("NHEADS", "D"))
def RoPE(x: Arr["NHEADS", "D", BF16], cs: Arr["D", BF16]) -> Arr["NHEADS", "D", BF16]:
    return batch(RoPEHead, x, cs, axes=(0, None))


@function(static=("V", "H"))
def Embedding(tok: I32, table: Arr["V", "H", BF16], *, V) -> Arr["H", BF16]:
    cols = table.permute(1, 0)  # a Strided view: column j of the table is the operand collection of gather j
    return batch(ops.gather_bf16(V), tok, cols, axes=(None, 0))


@function
def ArgmaxStep(carry: Tup[BF16, I32, I32], x: BF16) -> Tup[Tup[BF16, I32, I32], I32]:
    best, best_i, i = carry
    gt = ops.Bf16GtStrict(x, best)
    nb = ops.SelectBf16(gt, x, best)
    ni = ops.SelectI32(gt, i, best_i)
    nxt = ops.I32Add(i, ops.const(32, 1))
    return ((nb, ni, nxt), ni)


@function(static=("V",))
def TokenSelect(logits: Arr["V", BF16], *, V) -> I32:
    init = (logits[0], ops.const(32, 0), ops.const(32, 1))
    carry, _ys = scan(ArgmaxStep, init, xs=(logits[1:V],))
    return carry[1]


# a composite over a *registered* composite: RMSNorm of a Gemm row through TA1's definitions
@function(static=("K", "N", "EPS"))
def NormedRow(x: Arr["K", BF16], w: Arr["N", "K", BF16], g: Arr["N", BF16], *, K, N, EPS) -> Arr["N", BF16]:
    y = ops.Gemm(K=K, N=N)(x, w)
    return ops.RMSNormTriton(N=N, EPS=EPS)(y, g)


# the same thing hand-built in verity.ml.circuits, for the digest comparison
NormedRowIR = CompositeDefinition(
    "NormedRow", 1, ("K", "N", "EPS"),
    lambda S: ((("x", Array(S.K, BF16)), ("w", Array(S.N, Array(S.K, BF16))), ("g", Array(S.N, BF16))), Array(S.N, BF16)),
    lambda B, S, x, w, g: B.call(ir_bind(b1.RMSNormTriton, N=S.N, EPS=S.EPS), B.call(ir_bind(b1.Gemm, K=S.K, N=S.N), x, w), g),
    register=False)


def ta1_pairs():
    """(label, frontend Bound, TA1 SpecializedDefinition) pairs whose Program digests must agree."""
    return [
        ("Gemm{K=32,N=4}", Gemm.bind(K=32, N=4), ir_bind(b1.Gemm, K=32, N=4)),
        ("BiasAdd{N=8}", BiasAdd.bind(N=8), ir_bind(b1.BiasAdd, N=8)),
        ("SiluMul{I=8}", SiluMul.bind(I=8), ir_bind(b1.SiluMul, I=8)),
        ("RoPE{NHEADS=2,D=8}", RoPE.bind(NHEADS=2, D=8), ir_bind(b1.RoPE, NHEADS=2, D=8)),
        ("Embedding{V=8,H=4}", Embedding.bind(V=8, H=4), ir_bind(b1.Embedding, V=8, H=4)),
        ("TokenSelect{V=8}", TokenSelect.bind(V=8), ir_bind(b1.TokenSelect, V=8)),
        ("NormedRow{K=32,N=1024,EPS=1e-6}", NormedRow.bind(K=32, N=1024, EPS=1e-6), ir_bind(NormedRowIR, K=32, N=1024, EPS=1e-6)),
    ]


# ---------------------------------------------------------------------------------------------------------
# 2. IFb's identity shapes: repeated argument, returned input, dead declared return
# ---------------------------------------------------------------------------------------------------------

V16 = Value(16)


def identity_pair(Add16, Mul16, Id16):
    """Authored (`Top`) and hand-built (`TopIR`) versions of: `F(a, b) = (t, a, t, u)` with `t = Add16(a, a)`,
    `u = Mul16(a, b)`; `Top(v) = Id16(F(v, v)[0])`. `F` is called with a repeated argument, returns an input,
    returns `t` twice, and `u` is a declared return nobody reads."""

    @function(name="F")
    def F(a: V16, b: V16) -> Tup[V16, V16, V16, V16]:
        t = ops.Add16(a, a)
        u = ops.Mul16(a, b)
        return (t, a, t, u)

    @function(name="Top")
    def Top(v: V16) -> V16:
        r = F(v, v)
        return ops.Id16(r[0])

    FIR = CompositeDefinition("F", 1, (), lambda S: ((("a", V16), ("b", V16)), Tuple(V16, V16, V16, V16)),
                                 lambda B, S, a, b: (lambda t: tuple_of(t, a, t, B.call(Mul16, a, b)))(B.call(Add16, a, a)),
                                 register=False)
    TopIR = CompositeDefinition("Top", 1, (), lambda S: ((("v", V16),), V16),
                                   lambda B, S, v: B.call(Id16, B.call(FIR, v, v)[0]), register=False)
    return Top, TopIR


# ---------------------------------------------------------------------------------------------------------
# 3. negatives (each must fail closed with the stated code)
# ---------------------------------------------------------------------------------------------------------

A8 = Arr[8, BF16]


@function
def neg_aten_leak(x: A8, w: A8) -> A8:
    return batch(ops.Bf16Add, x, w, axes=(0, 0)) + x  # aten.add.Tensor has no Verity meaning


@function
def neg_inplace(x: A8, w: A8) -> A8:
    x.add_(w)  # in-place mutation of an input
    return batch(ops.Bf16Add, x, w, axes=(0, 0))


@function
def neg_host_escape(x: A8, w: A8) -> A8:
    if float(x.float().sum().item()) > 0:  # opaque host decision on values
        return batch(ops.Bf16Add, x, w, axes=(0, 0))
    return x


@function
def neg_hidden_constant(x: A8, w: A8) -> A8:
    k = torch.from_numpy(np.zeros(8, dtype=np.int16)).view(torch.bfloat16)  # a constant smuggled from the host
    return batch(ops.Bf16Add, x, k, axes=(0, 0))


_foreign = torch.library.Library("someone_else", "FRAGMENT")
_foreign.define("mystery(Tensor a) -> Tensor")
torch.library.register_fake("someone_else::mystery", lambda a: torch.empty_like(a))


@function
def neg_foreign_custom_op(x: A8, w: A8) -> A8:
    return torch.ops.someone_else.mystery(x)  # a custom op with a fake impl but no Verity binding


@function
def neg_cond(x: A8, w: A8) -> A8:
    return torch.cond(x[0].float() > 0, lambda a: batch(ops.Bf16Add, a, a, axes=(0, 0)), lambda a: a, (x,))


@function
def neg_width(x: A8, w: A8) -> A8:
    return batch(ops.F32Add, x, w, axes=(0, 0))  # F32Add wants Value<32>, bf16 tensors are Value<16>


@function
def neg_shape(x: A8, w: A8) -> A8:
    return Gemm(x, w)  # w must be [N][K]


NEGATIVES = [
    ("unregistered ATen op leaks in", neg_aten_leak, "E_UNREGISTERED_OP"),
    ("in-place mutation of an input", neg_inplace, "E_MUTATION"),
    ("opaque host callback (.item() decides control)", neg_host_escape, "E_OPAQUE_HOST"),
    ("hidden tensor constant from numpy", neg_hidden_constant, "E_HIDDEN_STATE"),
    ("foreign custom op without a Verity binding", neg_foreign_custom_op, "E_UNREGISTERED_OP"),
    ("torch.cond", neg_cond, "E_DYNAMIC_CONTROL_FLOW"),
    ("width mismatch (bf16 into F32Add)", neg_width, "E_WIDTH"),
    ("shape mismatch (rank-1 w into Gemm)", neg_shape, "E_SHAPE"),
]

NEG_INPUTS = (torch.empty(8, dtype=torch.bfloat16, device="meta"), torch.empty(8, dtype=torch.bfloat16, device="meta"))
