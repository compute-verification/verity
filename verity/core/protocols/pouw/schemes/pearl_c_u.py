"""U, Pearl-C's checked useful output, in its exact operation order: the pin that the Lean model of the clean-up (P2(d))
mirrors op for op.

**Status: CPU-pinned.**  Each function equals `pearl_c`'s reference (`chain`, `peel_factors_a`, `peel_factors_b`, `peel`) bit
for bit.  On the H100 kernel's gate problem (256 x 384 x 1,024, both formings) it reproduces the gate's expected C̃, A′·F_B,
B̃·F_A, G, P_A, P_B and U (``tests/vectors/pearl_c_u.json``, which records the gate files' hashes).  No kernel has run yet:
the joint H100 request's B1 capture, the BF16 peel into C̃ as a nonzero FP32 accumulator, confirms it on silicon.

**The order**, per output (i, j).  A′, B̃ are the formed E4M3 operands (k codes each), F_A, F_B the k x 32 bases, be_A,i and
be_B,j the peel's scaled noise rows (32 BF16 words: bf16(β·bf16(E)) under v0, bf16(E′) under v1).  Words are FP32 unless
named; RN is round to nearest, ties to even, no flush to zero; every operation below rounds once.

~~~text
chain(a, b)   total = +0;  for each window of 32·G products (G = 4: 128), in order:
                local = +0;  local = atom(local, a[s:s+32], b[s:s+32]) for its G slices s in order;
                total = RN_f32(total + local)                                   (promotion every 4 atoms; −0 never arises)
C̃_ij          chain(A′_i, B̃_j)
AF_ip         chain(A′_i, F_B[:, p])            BF_jq = chain(B̃_j, F_A[:, q])            G_pq = chain(F_B[:, p], F_A[:, q])
P_A,i         [be_A,i[p] for p = 0..31 ‖ RN_bf16(AF_ip) for p = 0..31]
P_B,j         [RN_bf16(RN_f32(acc_jq − BF_jq)) for q = 0..31 ‖ be_B,j[p] with its sign bit flipped, p = 0..31]
                acc_jq = +0, then acc_jq = RN_f32(be_B,j[p]·G_pq + acc_jq) for p = 0..31 ascending (one FMA each)
U_ij          acc = C̃_ij;  acc = k16(acc, P_A,i[16s:16s+16], P_B,j[16s:16s+16]) for s = 0, 1, 2, 3
~~~

**The two tensor-core steps in integers** (`grouped`; ``verity.ml.circuits.silicon``'s ``HOPPER_E4M3_K32`` and ``HOPPER_BF16_WGMMA_K16``
restated, and equal to them on every vector and the random sweep in the tests).  The device (`pearl_c_device`, the H100 by
default) gives each step as Lean's `Pipeline` (`Device.pipelines`): its groups in k order, the width W and the floor F.  A
step of several groups runs `grouped` once per group, the accumulator joining the first and each group's word the next
(Ada's E4M3 atom: two groups of 16).  The table's W and F are the H100's:

~~~text
terms    the accumulator word: magnitude m (24 bits, the hidden bit set when normal) at exponent E − 127 (−126 when
         subnormal); each product: magnitude (s_x·s_y)·2^P at exponent max(E_x, 1) + max(E_y, 1) + O, with s the
         significand (hidden bit set when the exponent field is nonzero); zero magnitudes drop out
         BF16 (k16):  P = 9,  O = −254;  width W = 26, floor F = −133
         E4M3 (atom): P = 17, O = −14;   width W = 14, floor F = −139
align    M = max(F, the exponents of the nonzero terms); each term is ±(m·2^(W−24) >> (M − e)), shifts truncating
         (for W = 14 the scaling by 2^(W−24) is itself a truncating right shift by 10)
sum      S = the exact integer sum; S = 0 is +0
round    b = bitlen(|S|): |S| truncated or zero-extended to W bits, exponent M + b − W; below −126, shifted right to −126;
         then scaled back by 2^(24−W), truncating
pack     exponent > 127 is ±inf; a magnitude without bit 23 at exponent −126 is subnormal; a zero magnitude is +0
~~~
"""

from __future__ import annotations

from collections.abc import Sequence

from verity.ml.circuits.silicon import fp32

from .pearl_c_device import H100, Device

STATUS = "cpu-pinned"
R, WINDOW, SLICE, PEEL_K = 32, 128, 32, 64


class NonFinite(ValueError):
    """A step left the finite range (the reference rejects such an output)."""


# ---- the grouped truncating sum --------------------------------------------------------------------------------------

#: (man_bits, exp_bits, product shift P, exponent offset O, width W, floor F) per operand format
FORMATS = {"bf16": (7, 8, 9, -254, 26, -133), "e4m3": (3, 4, 17, -14, 14, -139)}


def _acc_term(word: int) -> tuple[bool, int, int]:
    field, man = (word >> 23) & 0xFF, word & 0x7FFFFF
    if field == 0xFF:
        raise NonFinite("non-finite FP32 accumulator")
    if field == 0:
        return word >> 31 == 1, -126, man
    return word >> 31 == 1, field - 127, man | 1 << 23


def _product(fmt: str, x: int, y: int) -> tuple[bool, int, int]:
    man_bits, exp_bits, shift, offset, _, _ = FORMATS[fmt]
    top = 1 + exp_bits + man_bits
    ex, ey = (x >> man_bits) & ((1 << exp_bits) - 1), (y >> man_bits) & ((1 << exp_bits) - 1)
    mx, my = x & ((1 << man_bits) - 1), y & ((1 << man_bits) - 1)
    if fmt == "bf16" and (ex == 0xFF or ey == 0xFF) or fmt == "e4m3" and (x & 0x7F == 0x7F or y & 0x7F == 0x7F):
        raise NonFinite(f"non-finite {fmt} operand")
    sx, sy = mx | ((1 << man_bits) if ex else 0), my | ((1 << man_bits) if ey else 0)
    return ((x ^ y) >> (top - 1)) & 1 == 1, max(ex, 1) + max(ey, 1) + offset, (sx * sy) << shift


def grouped(fmt: str, acc: int, x: Sequence[int], y: Sequence[int], width: int | None = None, floor: int | None = None
            ) -> int:
    """One grouped truncating sum, acc + Σ x_t·y_t, FP32 words in and out (the module docstring's integer statement), at
    the table's W and F unless given."""
    _, _, _, _, w0, f0 = FORMATS[fmt]
    width, floor = w0 if width is None else width, f0 if floor is None else floor
    terms = [t for t in [_acc_term(acc), *(_product(fmt, a, b) for a, b in zip(x, y, strict=True))] if t[2]]
    top = max([floor, *(e for _, e, _ in terms)])
    rescale = width - 24
    total = 0
    for neg, e, m in terms:
        scaled = m << rescale if rescale >= 0 else m >> -rescale
        total += -(scaled >> (top - e)) if neg else scaled >> (top - e)
    if total == 0:
        return 0
    neg, mag = total < 0, abs(total)
    bits = mag.bit_length()
    exp = top + bits - width
    mag = mag >> (bits - width) if bits > width else mag << (width - bits)
    if exp < -126:
        mag >>= -126 - exp
        exp = -126
    mag = mag >> rescale if rescale >= 0 else mag << -rescale
    if mag == 0:
        return 0
    if exp > 127:
        return int(neg) << 31 | 0x7F800000
    field = exp + 127 - (0 if mag & 1 << 23 else 1)
    return int(neg) << 31 | field << 23 | (mag & 0x7FFFFF)


def _pipeline(fmt: str, pipeline: tuple[tuple[int, ...], int, int], acc: int, x: Sequence[int], y: Sequence[int]) -> int:
    groups, width, floor = pipeline
    if len(x) != sum(groups) or len(y) != sum(groups):
        raise ValueError(f"the step takes {sum(groups)} operand pairs")
    start = 0
    for size in groups:
        acc = grouped(fmt, acc, x[start:start + size], y[start:start + size], width, floor)
        start += size
    return acc


def atom(acc: int, a: Sequence[int], b: Sequence[int], device: Device = H100) -> int:
    """The device's k32 E4M3 atom into the accumulator (HOPPER_E4M3_K32 on the H100)."""
    return _pipeline("e4m3", device.pipelines["atom"], acc, a, b)


def k16(acc: int, x: Sequence[int], y: Sequence[int], device: Device = H100) -> int:
    """The device's k16 BF16 step into the accumulator (HOPPER_BF16_WGMMA_K16 on the H100)."""
    return _pipeline("bf16", device.pipelines["peel"], acc, x, y)


# ---- the order --------------------------------------------------------------------------------------------------

def _finite(word: int) -> int:
    if (word >> 23) & 0xFF == 0xFF:
        raise NonFinite("non-finite FP32 word")
    return word


def rn_bf16(word: int) -> int:
    """RN_bf16 of a finite FP32 word: one nearest-even rounding of its low 16 bits."""
    out = ((_finite(word) + 0x7FFF + ((word >> 16) & 1)) >> 16) & 0xFFFF
    if out & 0x7FFF >= 0x7F80:
        raise NonFinite("BF16 overflow")
    return out


def chain(a: Sequence[int], b: Sequence[int], trace: dict | None = None, device: Device = H100) -> int:
    """C̃ (or A′·F_B, B̃·F_A, G): windows of 32·G products, G atoms from +0 each, promoted by one RN FP32 add.  `trace`
    collects each atom's local word ("atoms") and each promotion's total ("totals").  Unpromoted (G None): one window over
    the whole of k and no promotion, so the word is the last atom's."""
    if len(a) != len(b) or len(a) % SLICE:
        raise ValueError("the chain multiplies in atoms of k = 32")
    if device.g is None:
        local = 0
        for s in range(0, len(a), SLICE):
            local = _finite(atom(local, a[s:s + SLICE], b[s:s + SLICE], device))
            if trace is not None:
                trace.setdefault("atoms", []).append(local)
        return local
    total, window = 0, SLICE * device.g
    for ws in range(0, len(a), window):
        local = 0
        for s in range(ws, min(ws + window, len(a)), SLICE):
            local = _finite(atom(local, a[s:s + SLICE], b[s:s + SLICE], device))
            if trace is not None:
                trace.setdefault("atoms", []).append(local)
        total = _finite(fp32.add(total, local))
        total = 0 if total == 0x80000000 else total
        if trace is not None:
            trace.setdefault("totals", []).append(total)
    return total


def pa_row(be: Sequence[int], af: Sequence[int]) -> list[int]:
    """P_A,i = [be_A,i ‖ RN_bf16(A′·F_B words)]."""
    return list(be) + [rn_bf16(w) for w in af]


def pb_row(be: Sequence[int], bf: Sequence[int], g: Sequence[Sequence[int]], trace: dict | None = None) -> list[int]:
    """P_B,j = [RN_bf16(RN_f32(acc_q − BF_jq)) ‖ −be_B,j], acc_q the FMA chain from +0 over p ascending.  `trace` collects
    the chains' results ("acc") and the differences ("diff")."""
    mid = []
    for q in range(R):
        acc = 0
        for p in range(R):
            acc = _finite(fp32.fma(be[p] << 16, g[p][q], acc))
        diff = _finite(fp32.add(acc, fp32.neg(bf[q])))
        if trace is not None:
            trace.setdefault("acc", []).append(acc)
            trace.setdefault("diff", []).append(diff)
        mid.append(rn_bf16(diff))
    return mid + [x ^ 0x8000 for x in be]


def u(c: int, pa: Sequence[int], pb: Sequence[int], trace: list | None = None, device: Device = H100) -> int:
    """U_ij from C̃_ij: four k16 steps over P_A,i and P_B,j in slice order.  `trace` collects the word after each step."""
    acc = c
    for s in range(0, PEEL_K, 16):
        acc = _finite(k16(acc, pa[s:s + 16], pb[s:s + 16], device))
        if trace is not None:
            trace.append(acc)
    return acc
