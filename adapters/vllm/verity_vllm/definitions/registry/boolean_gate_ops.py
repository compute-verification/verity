"""Word steps as gates, shared by the Boolean versions of the word modules built on `dsa` (`boolean_io_shell`, `boolean_stop_rule`):
core's gate builder for each word primitive, with the GPU's NaN word for the FP32 ones; no table or lookup gate."""
from __future__ import annotations

from verity.circuits.boolean.forms import NOT, ZERO, add, const_bits, mux, sub
from verity.circuits.ops.boolean import mufu as BM
from verity.circuits.ops.boolean import scalar as SC
from verity.circuits.ops.boolean.softmax import f32_gt, ftz
from verity.circuits.silicon import fp as FP

from verity_vllm.definitions.registry import boolean_dsa as BD

PTX = FP.F32_NAN_PTX


class GateOps:
    """Word steps as gates on ``C``: each value a list of forms (LSB first), each method the gates of the word primitive of the
    same name in a word module's ``WORD`` (`dsa`'s and the modules built on it)."""

    def __init__(self, C):
        self.C = C

    def word(self, bits: int):
        return const_bits(bits, 32)

    def neg(self, x):
        return FP.f32_neg(x)

    def add(self, a, b):
        return FP.f32_add(self.C, a, b, nan_word=PTX)

    def sub(self, a, b):
        return FP.f32_sub(self.C, a, b, nan_word=PTX)

    def mul(self, a, b):
        return FP.f32_mul(self.C, a, b, nan_word=PTX)

    def fma(self, a, b, c):
        return FP.f32_fma(self.C, a, b, c, nan_word=PTX)

    def fma_ftz(self, a, b, c):
        C = self.C
        return ftz(C, FP.f32_fma(C, ftz(C, a), ftz(C, b), ftz(C, c), nan_word=PTX))

    def rcp(self, b):
        return BM.div_full_rcp(self.C, b)

    def scale_a(self, a, b):
        return BM.div_full_scale_a(self.C, a, b)

    def rsqrt(self, x):
        return BM.rsqrt_approx(self.C, x)

    def ex2(self, x):
        return BM.mufu_ex2(self.C, x)

    def fabs(self, x):
        return list(x[:31]) + [ZERO]

    def fmaxf(self, a, b):
        return SC.fmaxmin(self.C, a, b, True)

    def fminf(self, a, b):
        return SC.fmaxmin(self.C, a, b, False)

    def e4m3(self, x):
        return SC.f32_to_e4m3_sat(self.C, x)

    def iadd(self, a, b):
        return add(self.C, a, b, width=32)

    def iand(self, a, b):
        return [self.C.AND(x, y) for x, y in zip(a, b)]

    def isub(self, a, b):
        return sub(self.C, a, b, 32)[0]

    def i2f(self, a):
        return BD.i32_to_f32(self.C, a)

    def ceil(self, x):
        return BD.f32_ceil_ftz(self.C, x)

    def cvt(self, h):
        return FP.bf16_to_f32(h)

    def bf16(self, u):
        return FP.f32_to_bf16(self.C, u, 0x7FFF)

    def fsel(self, c, a, b):
        return mux(self.C, c, a, b)

    def gt(self, a, b):
        return f32_gt(self.C, a, b)[0]

    def ile(self, a, b):
        """Signed a <= b: not b < a, the borrow of (b ^ 2^31) - (a ^ 2^31)."""
        flip = lambda x: list(x[:31]) + [NOT(x[31])]  # noqa: E731
        return NOT(sub(self.C, flip(b), flip(a), 32)[1])



class SelectGateOps(GateOps):
    """`GateOps` with a word module's ``isel``, ``band`` and ``bnot``, and ``bit(x, i)``."""

    def isel(self, c, a, b):
        return mux(self.C, c, a, b)

    def band(self, a, b):
        return self.C.AND(a, b)

    def bnot(self, a):
        return NOT(a)

    def bit(self, x, i: int):
        return x[i]
