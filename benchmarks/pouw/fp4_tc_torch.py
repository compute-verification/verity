#!/usr/bin/env python3
"""A bit-exact torch port of Pearl-C4's NVFP4 forming (``verity_pouw.schemes.pearl_c4.form``'s noise atom and block
cast, on ``verity.ml.tc.BlockScaledAlignAdd`` for ``blackwell_sm120_nvf4_m16n8k64``), batched over rows.  Every op is
exact integer arithmetic on FP32/FP4 encodings or an IEEE single-rounded FP32 multiply/divide (the same on CPU and
CUDA), so the codes and scales equal the reference's on any IEEE device; `fp4_tile_gpu.py`'s gate holds them to it.

Not modelled: FP32 FMA (ρ's accumulate in `stats`), which stays on the reference's CPU code.
"""
from __future__ import annotations

import torch

GROUP = 16
BLOCK = 16
ATOM = 64
FRAC_BITS = 27
ACC_WINDOW = 36
GRID_FLOOR = -174
GROUP_MAX = 2304
SPAN_K = 128
NEG = -(1 << 40)                                   # a grid exponent nothing reaches (anchors floor at GRID_FLOOR)

SIXTH = 0x3E2AAAAB
ONE = 0x3F800000
UE4M3_MIN = 0x01
UE4M3_TOP = 0x7E

_E2M1_NUM = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12], dtype=torch.int64)
_E2M1_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _fp8_to_f32_word(code: int) -> int:
    import struct
    if code & 0x7F == 0x7F:
        return 0
    s = -1.0 if code & 0x80 else 1.0
    e, m = (code >> 3) & 0xF, code & 7
    v = s * (m * 2.0 ** -9 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
    return struct.unpack("<I", struct.pack("<f", v))[0]


_UE4M3_WORD = torch.tensor([_fp8_to_f32_word(c) for c in range(128)], dtype=torch.int64)


class Engine:
    """NVFP4 forming and chain on one device; the lookup tables live there too."""

    def __init__(self, device: str = "cpu"):
        self.dev = torch.device(device)
        self.num = _E2M1_NUM.to(self.dev)
        self.ue4m3_word = _UE4M3_WORD.to(self.dev)

    # ---- FP32 helpers (bit views; multiply/divide are IEEE single-rounded) ------------------------------------------

    def _f32(self, words: torch.Tensor) -> torch.Tensor:
        return words.to(torch.int32).view(torch.float32)

    def _bits(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF

    def f32_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._bits(self._f32(a) * self._f32(b))

    def f32_div(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._bits(self._f32(a) / self._f32(b))

    # ---- casts -----------------------------------------------------------------------------------------------------

    def e2m1(self, word: torch.Tensor) -> torch.Tensor:
        """cvt.rn.satfinite.e2m1: nearest even, saturating to ±6, the sign kept (−0 is 0x8)."""
        v = self._f32(word).to(torch.float64).abs()
        code = torch.zeros_like(word, dtype=torch.int64)
        for i, mid in enumerate(_E2M1_MID):
            up = (i + 1) % 2 == 0                       # tie rounds to even (the upper code when i+1 is even)
            hit = (v > mid) | ((v == mid) & up)
            code = torch.where(hit, torch.full_like(code, i + 1), code)
        sign = (word.to(torch.int64) >> 31) & 1
        return code | (sign << 3)

    def ue4m3(self, word: torch.Tensor) -> torch.Tensor:
        """The UE4M3 byte of a non-negative FP32 word: f32_to_fp8 nearest-even, saturating to 448, then the low 7 bits."""
        w = word.to(torch.int64) & 0xFFFFFFFF
        v = self._f32(word).to(torch.float64).abs()
        efield = (w >> 23) & 0xFF
        e = efield - 127
        # subnormal E4M3 (e < -6): magnitude on the 2^-9 grid
        vv = v * 512.0
        fl = torch.floor(vv)
        frac = vv - fl
        sub = fl + torch.where((frac > 0.5) | ((frac == 0.5) & (fl.to(torch.int64) & 1 == 1)), 1.0, 0.0)
        sub = sub.to(torch.int64)
        # normal E4M3
        fm = w & 0x7FFFFF
        mant = fm >> 20
        rem = fm & 0xFFFFF
        bump = (rem > 0x80000) | ((rem == 0x80000) & (mant & 1 == 1))
        mant = mant + bump.to(torch.int64)
        exp = e + 7
        carry = mant == 8
        mant = torch.where(carry, torch.zeros_like(mant), mant)
        exp = exp + carry.to(torch.int64)
        sat = (exp > 15) | ((exp == 15) & (mant >= 7))
        norm = torch.where(sat, torch.full_like(mant, 0x7E), (exp << 3) | mant)
        out = torch.where(e < -6, sub, norm)
        return out & 0x7F

    # ---- scale bytes -----------------------------------------------------------------------------------------------

    def scale_dyadic(self, code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(mantissa, exponent) of a UE4M3 scale byte read from its low 7 bits (`hw_scales`)."""
        c = code.to(torch.int64) & 0x7F
        e, m = c >> 3, c & 7
        ma = torch.where(e == 0, m, 8 + m)
        ea = torch.where(e == 0, torch.full_like(e, -9), e - 10)
        return ma, ea

    def block_scale(self, amax: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(scale byte, FP32 reciprocal word) of a block with |max| word amax (NVFP4)."""
        v = self.f32_mul(amax, torch.full_like(amax, SIXTH))
        s = torch.clamp_min(self.ue4m3(v), UE4M3_MIN)
        recip = self.f32_div(torch.full_like(s, ONE), self.ue4m3_word[s])
        return s, recip

    # ---- the align-add (BlockScaledAlignAdd._transducer) -----------------------------------------------------------

    def _fp32_dyadic(self, word: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A finite FP32 word as (signed mantissa, exponent, exponent field)."""
        w = word.to(torch.int64) & 0xFFFFFFFF
        efield = (w >> 23) & 0xFF
        frac = w & 0x7FFFFF
        mant = torch.where(efield == 0, frac, frac | (1 << 23))
        exp = torch.where(efield == 0, torch.full_like(efield, -149), efield - 150)
        sign = (w >> 31) & 1
        mant = torch.where(sign == 1, -mant, mant)
        return mant, exp, efield

    def _bit_length(self, x: torch.Tensor) -> torch.Tensor:
        """bit_length of a non-negative int64 tensor (exact)."""
        t = x.clone()
        bl = torch.zeros_like(x)
        for sh in (32, 16, 8, 4, 2, 1):
            ge = (t >> sh) != 0
            bl = bl + torch.where(ge, sh, 0)
            t = torch.where(ge, t >> sh, t)
        return bl + (t != 0).to(torch.int64)

    def _trunc_rz(self, mant: torch.Tensor, exp: torch.Tensor, lsb: torch.Tensor) -> torch.Tensor:
        """floor toward zero of mant·2^exp onto the grid 2^lsb (mant may be signed, exp/lsb broadcast)."""
        shift = exp - lsb
        absm = mant.abs()
        left = mant << torch.clamp_min(shift, 0)
        right = absm >> torch.clamp_min(-shift, 0)
        right = torch.where(mant < 0, -right, right)
        return torch.where(shift >= 0, left, right)

    def _to_f32_rz(self, n: torch.Tensor, lsb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """dyadic.rs::to_f32(Rz) of the nonzero n·2^lsb (positive underflow: a sub-grid magnitude is +0), and whether it
        leaves the binary32 range (the reference raises there)."""
        neg = n < 0
        mag = n.abs()
        e_val = self._bit_length(mag) - 1 + lsb
        grid = torch.where(e_val >= -126, e_val - 23, torch.full_like(e_val, -149))
        shift = grid - lsb
        q = torch.where(shift > 0, mag >> torch.clamp_min(shift, 0), mag << torch.clamp_min(-shift, 0))
        e = self._bit_length(q) - 1 + grid
        over = (q != 0) & (e > 127)
        field = e + 127
        norm = ((field << 23) | (q & 0x7FFFFF))
        bits = torch.where(e >= -126, norm, q)
        bits = bits | torch.where(neg, torch.full_like(bits, 0x80000000), torch.zeros_like(bits))
        return torch.where(q == 0, torch.zeros_like(bits), bits) & 0xFFFFFFFF, over

    def grid(self, c_word: torch.Tensor, unit: torch.Tensor, part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``BlockScaledAlignAdd.grid``: the common grid exponent lsb[B], and where it is None (nothing participates and
        the accumulator is zero)."""
        am, _, efield = self._fp32_dyadic(c_word)
        acc_nz = am != 0
        pu = torch.where(part, unit - FRAC_BITS, torch.full_like(unit, NEG))
        pu_max = pu.max(dim=1).values
        aa = torch.where(acc_nz, (torch.where(efield == 0, torch.ones_like(efield), efield) - 127) - (ACC_WINDOW - 1),
                         torch.full_like(efield, NEG))
        lsb = torch.maximum(torch.maximum(pu_max, aa), torch.full_like(pu_max, GRID_FLOOR))
        return lsb, (~part.any(dim=1)) & (~acc_nz)

    def align_add(self, c_word: torch.Tensor, mant: torch.Tensor, unit: torch.Tensor,
                  part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched ``_transducer``: c_word[B], mant/unit/part[B, G] -> (the firing's FP32 word [B], out of range [B])."""
        am, ae, _ = self._fp32_dyadic(c_word)
        lsb, none = self.grid(c_word, unit, part)
        term = torch.where(mant != 0, self._trunc_rz(mant, unit, lsb.unsqueeze(1)), torch.zeros_like(mant))
        total = term.sum(dim=1) + torch.where(am != 0, self._trunc_rz(am, ae, lsb), torch.zeros_like(am))
        out, over = self._to_f32_rz(total, lsb)
        zero = none | (total == 0)
        return torch.where(zero, torch.zeros_like(out), out), over & ~zero

    # ---- group terms and the chain ---------------------------------------------------------------------------------

    def group_terms(self, a_codes: torch.Tensor, b_codes: torch.Tensor, a_scales: torch.Tensor,
                    b_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scaled group dots of one firing: a/b_codes[B, k], a/b_scales[B, k//BLOCK] -> mant/unit/part[B, G]."""
        B, k = a_codes.shape
        G = k // GROUP
        na = torch.where(a_codes & 8 == 8, -self.num[a_codes & 7], self.num[a_codes & 7])
        nb = torch.where(b_codes & 8 == 8, -self.num[b_codes & 7], self.num[b_codes & 7])
        prod = (na * nb).view(B, G, GROUP)
        gd = prod.sum(dim=2)
        anyprod = (((na != 0) & (nb != 0)).view(B, G, GROUP)).any(dim=2)
        ma, ea = self.scale_dyadic(a_scales)
        mb, eb = self.scale_dyadic(b_scales)
        mant = gd * ma * mb
        unit = -2 + ea + eb
        part = anyprod & (ma != 0) & (mb != 0)
        return mant, unit, part

    def step(self, c_word: torch.Tensor, a_codes: torch.Tensor, b_codes: torch.Tensor, a_scales: torch.Tensor,
             b_scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mant, unit, part = self.group_terms(a_codes, b_codes, a_scales, b_scales)
        return self.align_add(c_word, mant, unit, part)

    def _num_signed(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.where(codes & 8 == 8, -self.num[codes & 7], self.num[codes & 7])

    # ---- forming (pearl_c4.form: codes and scales of the noisy operand) ---------------------------------------------

    def form(self, x: torch.Tensor, alpha: torch.Tensor, ae: torch.Tensor, e_codes: torch.Tensor,
             f_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One side's codes[R, k] and scale bytes[R, k//BLOCK].  x[R, k] FP32 words, alpha[R]/ae[R] from `stats`,
        e_codes[R, 32] this side's E line per row, f_codes[k, 32] the F basis per position."""
        R, k = x.shape
        ne = self._num_signed(e_codes).to(torch.float64)                    # [R, 32]
        nf = self._num_signed(f_codes).to(torch.float64)                    # [k, 32]
        me = (e_codes & 7 != 0).to(torch.float64)
        mf = (f_codes & 7 != 0).to(torch.float64)
        ma, ea = self.scale_dyadic(ae)                                      # [R]
        mb, eb = self.scale_dyadic(torch.full_like(ae, 0x18))               # F's scale 2^-4
        seed = self.f32_mul(alpha.unsqueeze(1).expand(R, k).reshape(-1), x.reshape(-1))
        B = R * k
        mant = torch.zeros(B, 4, dtype=torch.int64, device=self.dev)
        unit = torch.zeros(B, 4, dtype=torch.int64, device=self.dev)
        part = torch.zeros(B, 4, dtype=torch.bool, device=self.dev)
        for g in range(2):                                                  # the two 16-groups of the 32-code lines
            sl = slice(g * 16, (g + 1) * 16)
            gd = (ne[:, sl] @ nf[:, sl].T).round().to(torch.int64)         # [R, k], exact (|dot| <= 2304)
            ap = (me[:, sl] @ mf[:, sl].T) > 0
            mant[:, g] = (gd * (ma * mb).unsqueeze(1)).reshape(-1)
            unit[:, g] = (-2 + (ea + eb).unsqueeze(1)).expand(R, k).reshape(-1)
            part[:, g] = (ap & (ma != 0).unsqueeze(1) & (mb != 0).unsqueeze(1)).reshape(-1)
        d, over = self.align_add(seed, mant, unit, part)
        if torch.any(over):
            raise ValueError("forming: a noise atom leaves the binary32 range")
        d = d.reshape(R, k)
        nb = k // BLOCK
        amax = (d.reshape(R, nb, BLOCK) & 0x7FFFFFFF).amax(dim=2)
        s, recip = self.block_scale(amax)
        codes = self.e2m1(self.f32_mul(d, recip.repeat_interleave(BLOCK, dim=1)))
        return codes, s
