"""The training rules for a quantizer's masks, integers, conversions and order statistics (`rules.training_masks`), each Program
run on small CPU tensors and its output words compared with torch's: comparisons (of f16 values too), logic, int64 adds, shifts and
ands (a code packed into bytes and back, shifts by static amounts across a tensor wider than a constant's runs, and a code split
into bit planes as bools and summed back), `where` (of f16 values too),
`clamp`, integer sums, the conversions among int64, uint8, bool and f32, a codebook lookup (static, and a value), `amax`, `floor`
and a static f32 `arange` with a row scattered in, bit for bit against torch CPU on words with signed zeros, infinities and NaNs,
with and without a compute capability; at one, a clamp by ±0 against the GPU kernel's
`fmaxf` and `fminf`, the GPU's conversions to bf16, to and from f16 and from f32 to int64 and uint8 against torch CPU on NaN-free
words and against the GPU's NaN and saturated words, and `kthvalue` against a model of torch's CUDA radix select
(`radixSelect` as SortingRadixSelect.cuh writes it, two bits a digit) on slices with NaNs and signed zeros, and against torch
CPU's `kthvalue` where the two agree; QAT's sparse quantizer, dequantized and added to a bf16 parameter, against torch CPU with
the sum of the kept magnitudes the GPU's `reduce_kernel` model's; tensor constants of a traced function; and the refusals."""
from __future__ import annotations

import math
import operator

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import _program
from verity.definitions.ops import scalar as SC
from verity.definitions.numerics import torch_kernels as TK
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.registry import cublas_bf16 as CB
from verity.definitions.registry import int8_exact as IE
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules import training_views as TV
from verity_vllm.program.frontend.rules.training import training_rules
from verity_vllm.program.frontend.value_types import WIDTH_OF_DTYPE

GPU, CPU = training_rules(cc=CB.TRAIN_CC), training_rules()
BOTH = pytest.mark.parametrize("rules", [GPU, CPU], ids=["cc", "cpu"])
_INT = {8: torch.uint8, 16: torch.int16, 32: torch.int32, 64: torch.int64}
#: a quiet NaN with a payload and a negative signaling one
NAN_A, NAN_B = 0x7FC01234, 0xFFA00001


def _words(t: torch.Tensor) -> list[int]:
    """t's words, unsigned (a bool's 1 or 0)."""
    t = t.detach().contiguous().reshape(-1)
    w = WIDTH_OF_DTYPE[t.dtype]
    t = t.to(torch.uint8) if w == 1 else t.view(_INT[w])
    return [v & ((1 << w) - 1) for v in t.tolist()]


def _f32(words) -> torch.Tensor:
    return torch.from_numpy(np.array(words, dtype=np.uint32).view(np.float32).copy())


def _run(f, *xs, rules=GPU) -> list[int]:
    """The Program's output words on xs, every output in order."""
    prog = _program(f, *xs, rules=rules)
    tab = materialize(prog, [_words(x) for x in xs])
    return [tab[g] for g in prog.circuit.root_outputs()]


def _same(got: list[int], want: list[int]) -> None:
    bad = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
    assert len(got) == len(want) and not bad, (len(got), len(want), [(i, hex(got[i]), hex(want[i])) for i in bad[:4]])


def _check(f, *xs, rules=GPU) -> None:
    """f's Program against torch CPU's words."""
    _same(_run(f, *xs, rules=rules), [w for t in f(*xs) for w in _words(t)])


def _special(*shape, seed=0) -> torch.Tensor:
    """f32 values with ±0, ±inf, subnormals, equal values and two NaNs among normal ones, shuffled."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(math.prod(shape), generator=g)
    x[:12] = _f32([0, 0x80000000, 0x7F800000, 0xFF800000, 0x3F800000, 0x3F800000, 0xBF800000, 1, 0x80000400, 0x40400000, NAN_A,
                   NAN_B])
    return x[torch.randperm(x.numel(), generator=g)].reshape(shape)


X, Y = _special(4, 16, seed=1), _special(4, 16, seed=2)
YEQ = torch.where(torch.arange(64).reshape(4, 16) % 3 == 0, X, Y)
I64 = torch.randint(-1000, 1000, (4, 16), generator=torch.Generator().manual_seed(3))
_CMPS = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le, "eq": operator.eq, "ne": operator.ne}


@BOTH
@pytest.mark.parametrize("op", list(_CMPS))
def test_comparisons_are_ieees(op: str, rules) -> None:
    """Of two values (every third element equal), a broadcast row and column, a tensor constant with ±0 and a NaN, and the
    numbers 0.0, -0.0, 1 and NaN."""
    c = _CMPS[op]

    def f(a, b, r, k):
        t = torch.tensor([0.0, -0.0, 1.0, float("nan")] * 4)
        return c(a, b), c(a, r), c(a, k), c(a, t), c(a, 0.0), c(a, -0.0), c(a, 1), c(a, float("nan"))

    _check(f, X, YEQ, Y[0], Y[:, :1], rules=rules)


@BOTH
def test_logic(rules) -> None:
    def f(a, b):
        m, n = a > 0, b < 0.5
        k = torch.tensor([True, False] * 8)
        return m & n, m | n, ~m, torch.logical_and(m, n), torch.logical_or(m, k), torch.logical_not(n), m[0] | n

    _check(f, X, Y, rules=rules)


@BOTH
def test_where(rules) -> None:
    """f32, int64 and bf16 results; QAT's nested int64 `where` of numbers; a bf16 and an int64 operand of an f32 result, as torch
    promotes them; a broadcast condition; NaN words passed through."""
    xb, yb = X.bfloat16(), Y.bfloat16()

    def f(a, b, i, p, q):
        m = a > 0
        return (torch.where(m, a, b), torch.where(m, a, -2.0), torch.where(m, i, 7), torch.where(m, torch.where(a >= 0, 2, 0), 1),
                torch.where(m, p, q), torch.where(m, p, b), torch.where(m, i, b), torch.where(m[:, :1], a, b))

    _check(f, X, Y, I64, xb, yb, rules=rules)


@BOTH
def test_int64_clamps(rules) -> None:
    c = torch.tensor([-(2 ** 62), -7, -1, 0, 1, 2, 5, 9, 2 ** 40 + 3, 2 ** 62] * 2)
    _check(lambda c: (c.clamp(min=1), c.clamp(max=5), c.clamp(-3, 7), torch.clamp_min(c, 2), torch.clamp_max(c, -1),
                      c.clamp(5, 2)), c, rules=rules)


@BOTH
def test_f32_clamps(rules) -> None:
    """Nonzero bounds over 67 elements (torch CPU's vector body and scalar tail): each NaN's word kept, an int bound its f32
    word, a lower bound above the upper one."""
    x = _special(67, seed=5)
    _check(lambda a: (a.clamp(max=65504.0), a.clamp(min=-1.5), a.clamp(-1.0, 2), torch.clamp_min(a, 0.25),
                      torch.clamp_max(a, -3.0), a.clamp(2.0, 1.0)), x, rules=rules)


def test_a_zero_bound_is_torch_cpus() -> None:
    """Without a compute capability a bound ±0 is lowered: torch CPU keeps x where x and the bound are zeros of either sign."""
    _check(lambda a: (a.clamp(min=0.0), a.clamp(max=-0.0), a.clamp(-0.0, 0.0)), _special(67, seed=6), rules=CPU)


def _gpu_clamp(words: list[int], lo, hi) -> list[int]:
    """`launch_clamp_scalar`: a NaN kept, else `fminf(fmaxf(x, lo), hi)` of the bounds' f32 words."""
    out = []
    for w in words:
        y = w if lo is None or w & 0x7FFFFFFF > 0x7F800000 else SC.F32Fmaxf.evaluate(w, _words(torch.tensor(lo))[0])
        out.append(y if hi is None or w & 0x7FFFFFFF > 0x7F800000 else SC.F32Fminf.evaluate(y, _words(torch.tensor(hi))[0]))
    return out


def test_a_zero_bound_is_the_gpus() -> None:
    """At a compute capability a bound ±0 is the GPU kernel's: `fmaxf` and `fminf` order -0 below +0, so `clamp(min=0.0)` makes
    -0 +0 where torch CPU keeps it; a NaN kept."""
    x = _special(67, seed=6)
    bounds = [(0.0, None), (None, -0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, None), (None, 0.0)]
    got = _run(lambda a: tuple(a.clamp(lo, hi) for lo, hi in bounds) + (torch.clamp_min(a, 0.0), torch.clamp_max(a, -0.0)), x)
    want = [y for lo, hi in bounds + [(0.0, None), (None, -0.0)] for y in _gpu_clamp(_words(x), lo, hi)]
    _same(got, want)
    assert _gpu_clamp([0x80000000], 0.0, None) == [0]


@BOTH
def test_exact_conversions(rules) -> None:
    """int64 to uint8 (the low byte), uint8 to int64, int64 to f32 at ties and the ends of its range, bool to f32 and int64."""
    i = torch.tensor([-(2 ** 63), -(2 ** 62) - 2 ** 38 - 1, -257, -1, 0, 1, 255, 256, 2 ** 24 + 1, 2 ** 24 + 3, 2 ** 53 + 1,
                      2 ** 63 - 1] * 2)
    u = torch.arange(256).to(torch.uint8)
    _check(lambda i, u, b: (i.to(torch.uint8), u.long(), i.float(), b.float(), b.long()), i, u, X > 0, rules=rules)


def _f16_words() -> torch.Tensor:
    return torch.arange(1 << 16).to(torch.int16).view(torch.float16)


def test_the_gpus_f16_and_bf16_conversions() -> None:
    """At a compute capability, on NaN-free words: f32 to f16 around every rounding position (ties, overflow to infinity, f16
    subnormals), every f16 word to f32 and f32 to bf16, against torch CPU."""
    g = torch.Generator().manual_seed(7)
    e = torch.arange(-26, 17, dtype=torch.float32)
    x = torch.cat([torch.randn(200, generator=g) * 2.0 ** torch.randint(-20, 16, (200,), generator=g),
                   ((1 + torch.tensor([2.0 ** -11, 3 * 2.0 ** -11, 2.0 ** -11 + 2.0 ** -20])[:, None]) * 2.0 ** e).reshape(-1),
                   torch.tensor([65504.0, 65519.0, 65520.0, 1e9, -65520.0, 0.0, -0.0, float("inf")])])
    h = _f16_words()
    h = h[~torch.isnan(h)]
    _check(lambda a, h: (a.to(torch.float16), h.float(), a.to(torch.bfloat16)), x.reshape(-1), h)


def test_the_gpus_truncations() -> None:
    """At a compute capability f32 to int64 and to uint8 (through int64, its low byte) truncate toward zero, as torch CPU does
    below 2^63 in magnitude, negative values too; a NaN is 0, from 2^63 up INT64_MAX and below -2^63 INT64_MIN (the GPU's
    `cvt.rzi.s64.f32`), whose low bytes are 0xFF and 0."""
    v = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 255.9, 256.0, -255.9, -256.0, 1e-40, 2.0 ** 31 + 2.0 ** 8, -(2.0 ** 40),
                      2.0 ** 62, -(2.0 ** 63)])
    x = torch.cat([v, torch.randn(41, generator=torch.Generator().manual_seed(16)) * 1000])
    _check(lambda a: (a.long(), a.to(torch.uint8)), x)
    odd = _f32([NAN_A, NAN_B, 0x5F000000, INF, 0xDF000001, NINF])
    top, bottom = (1 << 63) - 1, 1 << 63
    assert _run(lambda a: (a.long(), a.to(torch.uint8)), odd) == [0, 0, top, top, bottom, bottom, 0, 0, 0xFF, 0xFF, 0, 0]


def test_the_gpus_nan_words() -> None:
    """At a compute capability each conversion's NaN is the GPU's word (`F32ToF16Rn_v1` 0x7FFF, `F16ToF32_v1` 0x7FFFFFFF,
    `F2fpBf16_v1` 0x7FFF), not torch CPU's."""
    x = _f32([NAN_A, NAN_B, 0x7F800001, 0xFFFFFFFF])
    h = _f16_words()
    h = h[torch.isnan(h)]
    assert _run(lambda a, h: (a.to(torch.float16), h.float(), a.to(torch.bfloat16)), x, h) == \
        [0x7FFF] * 4 + [0x7FFFFFFF] * h.numel() + [0x7FFF] * 4


@BOTH
def test_integer_sums(rules) -> None:
    """A mask's count over each axis, kept, and over all; int64 sums that wrap modulo 2^64, as torch's do."""
    i = torch.tensor([2 ** 62, 2 ** 62, 2 ** 62, 2 ** 62, -5, 7, 3, 1] * 3).reshape(3, 8)

    def f(a, i):
        m = a > 0
        return m.sum(1), m.sum(0, keepdim=True), m.sum(), m.sum((0, 1)), i.sum(1), i.sum(0), i.sum()

    _check(f, X, i, rules=rules)


@BOTH
def test_a_codebook_lookup(rules) -> None:
    """QAT's `cb[codes.long()]` of uint8 codes into an f16 codebook made f32, and an int64 index from -4 to 3 (a negative one
    read from the end, as torch reads it)."""
    u = torch.randint(0, 4, (64,), generator=torch.Generator().manual_seed(8)).to(torch.uint8)
    i = torch.tensor([-4, -3, -2, -1, 0, 1, 2, 3] * 2)

    def f(u, i):
        cb = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float16).float()
        return cb[u.long()], cb[i]

    _check(f, u, i, rules=rules)


@BOTH
def test_amax(rules) -> None:
    """Over each axis, kept, over both and over all, of values with ±0, infinities and subnormals whose maxima are not zeros,
    against torch CPU; a row with a NaN has the canonical NaN 0x7FC00000."""
    x = torch.where(torch.isnan(X), 1.5, X)
    x[3] = x[3].abs() + 0.5
    _check(lambda a: (a.amax(1), a.amax(0, keepdim=True), a.amax((0, 1)), a.amax(), a.abs().amax(1, keepdim=True)), x,
           rules=rules)
    want = [0x7FC00000 if torch.isnan(r).any() else _words(r.amax())[0] for r in X]
    assert 0x7FC00000 in want
    _same(_run(lambda a: (a.amax(1),), X, rules=rules), want)


@BOTH
def test_floor(rules) -> None:
    """Halves and their neighbours of either sign, ±0, values within one of 0, around 2^23 and 2^24, infinities and subnormals,
    over 81 elements (torch CPU's vector body and scalar tail), against torch CPU; a NaN quieted, as `F32Rint_v1` quiets it."""
    v = torch.tensor([0.5, 1.5, 2.5, 0.49999997, 0.50000006, 0.3, 0.9999999, 1.0, 1.0000001, 2.0 ** 23 - 0.5, 2.0 ** 23,
                      2.0 ** 23 + 1, 2.0 ** 24, 1e30, float("inf"), 1e-40, 2.0 ** -126, 3.75, 7.0])
    x = torch.cat([v, -v, torch.tensor([0.0, -0.0]), torch.randn(41, generator=torch.Generator().manual_seed(15)) * 100])
    _check(lambda a: (torch.floor(a),), x, rules=rules)
    nans = [NAN_A, NAN_B, 0x7F800001, 0xFFFFFFFF]
    got = _run(lambda a: (torch.floor(a),), _f32(nans), rules=rules)
    assert got == [IE.F32Rint.evaluate(w) for w in nans] and all(w & 0x7FC00000 == 0x7FC00000 for w in got), [hex(w) for w in got]


#: int64 values at the ends of its range and between
ENDS = torch.tensor([0, 1, -1, 2 ** 63 - 1, -2 ** 63, 255, -256, 12345678901234, -987654321])


@BOTH
def test_int64_adds_shifts_and_ands(rules) -> None:
    """Every amount from -2 to 66 of values at the ends of int64 (torch's 0 and sign fills past them), by Python ints; adds that
    wrap modulo 2^64; and two-bit codes packed eight to a word, through bytes and back, by static amounts, as a quantizer passes its
    code."""
    s = torch.arange(-2, 67)
    _check(lambda a, s: (a[:, None] << s, a[:, None] >> s, a << 3, a >> 62, a & 255, a[:, None] & s, a[:, None] + a, a + 1,
                         0 + a), ENDS, s, rules=rules)

    def through_bytes(c):
        lanes, octets = torch.arange(8), torch.arange(2)
        word = (c.reshape(4, -1, 8) << (2 * lanes)).sum(-1)
        packed = ((word.unsqueeze(-1) >> (8 * octets)) & 255).to(torch.uint8)
        word = (packed.long() << (8 * octets)).sum(-1)
        return packed, ((word.unsqueeze(-1) >> (2 * lanes)) & 3).reshape(c.shape)

    _check(through_bytes, torch.randint(0, 4, (4, 16), generator=torch.Generator().manual_seed(17)), rules=rules)


@BOTH
def test_static_amounts_across_a_wide_tensor(rules) -> None:
    """Codes shifted left by a row of eight static amounts, and a column shifted right by them, one row past what a constant
    tensor's runs allow in the result's order (there each amount is a run of one), as a wide quantizer packs its code."""
    rows = TV._PARTS // 8 + 1

    def f(c, w):
        s = 2 * torch.arange(8)
        return c.reshape(-1, 8) << s, w[:, None] >> s

    g = torch.Generator().manual_seed(22)
    _check(f, torch.randint(0, 4, (rows * 8,), generator=g), torch.randint(-2 ** 62, 2 ** 62, (rows,), generator=g), rules=rules)


@BOTH
def test_bit_planes_through_bools(rules) -> None:
    """A four-bit code split into bit planes as bools and summed back, as a quantizer passes its code between blocks (`(c >> b) & 1`
    to bool, each plane to int64 shifted back, Python's `sum` from 0), and int64 to bool of values at the ends of int64."""
    def f(c, a):
        planes = tuple(((c >> b) & 1).bool() for b in range(4))
        return planes + (sum(p.long() << b for b, p in enumerate(planes)), a.bool())

    _check(f, torch.randint(0, 16, (4, 16), generator=torch.Generator().manual_seed(19)), ENDS, rules=rules)


@BOTH
def test_f16_comparisons_and_selects(rules) -> None:
    """f16 values with ±0, infinities and NaNs compared with each other (every third element equal) and with numbers f16 holds,
    and f16 `where`s, as a quantizer keeps or replaces its f16 scales."""
    h, k = X.half(), YEQ.half()

    def f(h, k):
        return tuple(c(h, k) for c in _CMPS.values()) + tuple(c(h, x) for c in _CMPS.values() for x in (0, -0.0, 65504.0)) + (
            torch.where(h > k, h, k), torch.where(h > 0, h, torch.ones_like(h)))

    _check(f, h, k, rules=rules)


@BOTH
def test_a_value_table_lookup(rules) -> None:
    """A codebook that is a value, indexed by an int64 value from -T to T - 1, as a quantizer reads its grid."""
    g = torch.Generator().manual_seed(20)
    _check(lambda t, i: (t[i],), _special(255, seed=21), torch.randint(-255, 255, (4, 16), generator=g), rules=rules)


@BOTH
def test_a_static_arange_with_a_row_scattered_in(rules) -> None:
    """An f32 `arange`, arithmetic on it and a row divided in place (`select_scatter`) are static, as torch CPU computes them, and
    read by a value's product."""
    def f(x):
        i = torch.arange(8, dtype=torch.float32)
        m = (i[None, :] + 0.5) * i[:, None] / 8
        m[0] = m[0] / 2.0
        return x * m, m

    _check(f, torch.randn(8, 8, generator=torch.Generator().manual_seed(18)), rules=rules)


def test_an_index_out_of_range_reads_a_nan() -> None:
    """torch raises; the lookup reads 0xFFFFFFFF, past the table at either end and past the i32 range."""
    assert _run(lambda i: (torch.tensor([1.0, 2.0, 3.0, 4.0])[i],), torch.tensor([4, -5, 2 ** 40])) == [0xFFFFFFFF] * 3


def test_a_traced_tensor_constant_is_static() -> None:
    """`make_fx` keeps a tensor the traced function makes from Python data as a graph attribute: a float one is a static float
    tensor, an integer one a static index."""
    _check(lambda a: (a * torch.tensor([1.0, -2.0, 0.5]), a[torch.tensor([2, 0, 3])]), _special(4, 3, seed=9), rules=CPU)


def _key(w: int) -> int:
    """`TopKTypeConfig<float>::convert`: every NaN the largest key."""
    return 0xFFFFFFFF if w & 0x7FFFFFFF > 0x7F800000 else w ^ (0xFFFFFFFF if w >> 31 else 0x80000000)


def _radix_select(words: list[int], k: int) -> int:
    """torch's CUDA `radixSelect` for the k-th smallest of a slice of f32 words: over the order keys (`TopKTypeConfig<float>`),
    two bits a digit from the top, the counts of the keys matching the digits found so far taken bucket by bucket, an element
    alone in its bucket as the k-th returned as its own word (`findPattern`), else the found key converted back (`deconvert`)."""
    keys = [_key(w) for w in words]
    desired = mask = 0
    for pos in range(30, -1, -2):
        counts = [0] * 4
        for key in keys:
            if key & mask == desired:
                counts[(key >> pos) & 3] += 1
        for digit, c in enumerate(counts):
            if c == 1 and k == 1:
                desired, mask = desired | digit << pos, mask | 3 << pos
                return next(w for w, key in zip(words, keys) if key & mask == desired)
            if c >= k:
                desired, mask = desired | digit << pos, mask | 3 << pos
                break
            k -= c
    return desired ^ (0x80000000 if desired >> 31 else 0xFFFFFFFF)


INF, NINF = 0x7F800000, 0xFF800000


@pytest.mark.parametrize("words, ks", [
    ([0, 0x80000000, 0, 0x80000000, 0x3F800000, 0xBF800000], [1, 2, 3, 4, 5, 6]),
    ([0x40400000, NAN_A, 0x3F800000, 0x40000000], [1, 3, 4]),
    ([NAN_A, 0x3F800000, NAN_B], [1, 2, 3]),
    ([INF, NAN_B, NINF, INF], [1, 2, 3, 4]),
    ([NAN_A], [1]),
])
def test_kthvalue_is_the_gpus_radix_select(words: list[int], ks: list[int]) -> None:
    """-0 before +0, a NaN alone at k its own word, two NaNs 0x7FFFFFFF, infinities, one element."""
    x = _f32(words)
    _same(_run(lambda a: tuple(a.kthvalue(k).values for k in ks), x), [_radix_select(words, k) for k in ks])


def test_kthvalue_of_many_slices_is_the_gpus_radix_select() -> None:
    """Rows and columns of words with signed zeros, infinities and NaNs (two in a row, one in another), keepdim or not."""
    x = _special(5, 20, seed=10)
    x[2, :3] = _f32([0, 0x80000000, NAN_A])
    rows, cols = x.view(torch.int32).numpy().astype(np.uint32), x.T.contiguous().view(torch.int32).numpy().astype(np.uint32)
    ks = (1, 7, 19, 20)
    got = _run(lambda a: tuple(a.kthvalue(k, dim=1).values for k in ks) + (a.T.kthvalue(3, dim=0, keepdim=True).values,), x)
    want = [_radix_select(r.tolist(), k) for k in ks for r in rows] + [_radix_select(c.tolist(), 3) for c in cols.T]
    _same(got, want)


def test_kthvalue_is_torch_cpus_on_magnitudes() -> None:
    """On |v| (no -0) without NaN the GPU's value is torch CPU's, ties at k included."""
    a = torch.randn(3, 40, generator=torch.Generator().manual_seed(11)).abs()
    a[:, :6] = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.5, 2.0])
    _check(lambda a: tuple(a.kthvalue(k, dim=1, keepdim=True).values for k in (1, 2, 4, 21, 40)) + (a.reshape(-1).kthvalue(61).values,),
           a)


KEEP, MULT = 0.25, 0.7


def _kept(a: torch.Tensor) -> torch.Tensor:
    g = a.shape[1]
    n_keep = min(g, max(1, math.ceil(KEEP * g)))
    return (a >= a.kthvalue(g - n_keep + 1, dim=1, keepdim=True).values) & (a > 0)


def _quantized(v, theta, total=None):
    """QAT's sparse quantizer of v over one group (`qat_quant`, group -1), dequantized (`QArray.value`) and added to the bf16
    parameter theta: the new parameter, the f16 scale and the uint8 codes. `total`, if given, is the sum of the kept magnitudes."""
    vg = v.reshape(1, -1)
    a = vg.abs()
    kept = _kept(a)
    s = (MULT * ((a * kept).sum(1) if total is None else total) / kept.sum(1).clamp(min=1)).clamp(max=65504.0).to(torch.float16)
    codes = torch.where(kept, torch.where(vg >= 0, 2, 0), 1).to(torch.uint8).view(-1)
    cb = torch.tensor([-1.0, 0.0, 1.0, 0.0], dtype=torch.float16).float()
    q = cb[codes.long()].view(-1, vg.shape[1]) * s.float()[:, None]
    return (theta.float() + q.view(theta.shape)).to(torch.bfloat16), s, codes


def _qat(v, theta):
    return _quantized(v, theta)


def test_qats_sparse_quantizer_is_the_gpus() -> None:
    """The new parameter, scale and codes at a compute capability, against torch CPU with the sum of the kept magnitudes the
    GPU's `reduce_kernel` model's (every other op the GPU's word on NaN-free data); a zero and tied magnitudes at the threshold."""
    g = torch.Generator().manual_seed(12)
    v = torch.randn(8, 24, generator=g) * 0.02
    v[0, :6] = torch.tensor([0.0, 0.03, -0.03, 0.03, 0.025, -0.025])
    theta = torch.randn(8, 24, generator=g).bfloat16()
    a = v.reshape(1, -1).abs()
    total = _f32([TK.reduce_all(np.array(_words(a * _kept(a)), np.uint32), square=False)])
    _same(_run(_qat, v, theta), [w for t in _quantized(v, theta, total) for w in _words(t)])


def test_qats_sparse_quantizer_definitions() -> None:
    ids = {nd.fn.id.split("{")[0] for nd in _program(_qat, torch.randn(8, 24), torch.randn(8, 24).bfloat16(), rules=GPU)
           .root.body.nodes}
    assert {"F32Fabs_v1", "F32OrderKey_v1", "OrderKeyF32_v1", "F32GtStrict_v1", "BitAnd_v1", "I64Add_v1", "I64Le_v1",
            "SelectI64_v1", "I64ToF32Rn_v1", "F32Div_v2", "F32ToF16Rn_v1", "I64ToU8_v1", "U8ToI64_v1", "GatherI32x4_v1",
            "F16ToF32_v1", "F2fpBf16_v1"} <= ids, ids


def _refused(f, *xs, rules) -> Unsupported:
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=rules, name="T").unsupported
    assert isinstance(u, Unsupported), "derived"
    return u


A = torch.randn(4, 16, generator=torch.Generator().manual_seed(13)).abs()


@pytest.mark.parametrize("f, xs, rules, why", [
    (lambda a: (a.kthvalue(3).values,), (A,), CPU, "quickselect"),
    (lambda a: (a.kthvalue(3).indices,), (A,), GPU, "the indices are read"),
    (lambda a: (a.to(torch.float16),), (A,), CPU, "vector body"),
    (lambda h: (h.float(),), (A.half(),), CPU, "vector body"),
    (lambda a: (a.bfloat16() > 0.5,), (A,), GPU, "a bf16 comparison"),
    (lambda a: (a.long(),), (A,), CPU, "INT64_MIN for a NaN"),
    (lambda a: (a.to(torch.uint8),), (A,), CPU, "INT64_MIN for a NaN"),
    (lambda a: (a.clamp(min=float("nan")),), (A,), CPU, "a NaN bound"),
    (lambda a: (a.bfloat16().clamp(min=1.0),), (A,), GPU, "this rule lowers an int64 or f32 value"),
    (lambda a: (a.bfloat16().abs(),), (A,), GPU, "(this rule lowers f32)"),
    (lambda a: (a.bfloat16().floor(),), (A,), GPU, "(this rule lowers f32)"),
    (lambda a: (a.bfloat16().amax(1),), (A,), GPU, "not an f32 value"),
    (lambda a, i: (a.bfloat16() * i,), (A, I64), GPU, "an int64 operand of a bf16 result"),
    (lambda i: (i << 2,), (I64.int(),), GPU, "result dtype torch.int32"),
    (lambda a: (a[0] * torch.arange(0.0, 4.0, 0.25),), (A,), GPU, "a fractional bound or step"),
    (lambda a: (a[0] * torch.arange(2 ** 24 - 8, 2 ** 24 + 8, dtype=torch.float32),), (A,), GPU, "an integer past 2^24"),
    (lambda a: (a.half() > 0.1,), (A,), GPU, "a number f16 rounds"),
    (lambda a: (a.half() > a,), (A,), GPU, "not an f16 value"),
    (lambda i: (torch.add(i, i, alpha=2),), (I64,), GPU, "alpha != 1"),
], ids=["kth-cpu", "kth-indices", "to-f16-cpu", "from-f16-cpu", "bf16-compare", "to-int64-cpu", "to-uint8-cpu", "nan-bound",
        "bf16-clamp", "bf16-abs", "bf16-floor", "bf16-amax", "int64-bf16", "int32-shift", "fractional-arange", "big-arange",
        "f16-inexact-number", "f16-beside-f32", "int64-alpha"])
def test_refusals(f, xs, rules, why: str) -> None:
    u = _refused(f, *xs, rules=rules)
    assert u.reason == "numerics_unregistered" and why in u.hint, u
