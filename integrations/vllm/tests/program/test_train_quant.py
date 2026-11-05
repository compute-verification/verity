"""The training rules for a quantizer's masks, integers, conversions and order statistics (`rules.training_masks`), each Program
run on small CPU tensors and its output words compared with torch's: comparisons, logic, `where`, `clamp`, integer sums, the
conversions among int64, uint8, bool and f32, and a codebook lookup, bit for bit against torch CPU on words with signed zeros,
infinities and NaNs, with and without a compute capability; at one, the GPU's conversions to bf16 and to and from f16 against
torch CPU on NaN-free words and against the GPU's NaN words, and `kthvalue` against a model of torch's CUDA radix select
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
from verity.ml.definitions.numerics import torch_kernels as TK
from verity.ml.definitions.numerics.serve3_reference import materialize
from verity.ml.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
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
    (lambda a: (a.clamp(min=0.0),), (A,), GPU, "a bound ±0 on the GPU"),
    (lambda a: (a.clamp(min=float("nan")),), (A,), CPU, "a NaN bound"),
    (lambda a: (a.bfloat16().clamp(min=1.0),), (A,), GPU, "this rule lowers an int64 or f32 value"),
    (lambda a: (a.bfloat16().abs(),), (A,), GPU, "(this rule lowers f32)"),
    (lambda a, i: (a.bfloat16() * i,), (A, I64), GPU, "an int64 operand of a bf16 result"),
], ids=["kth-cpu", "kth-indices", "to-f16-cpu", "from-f16-cpu", "bf16-compare", "zero-bound", "nan-bound", "bf16-clamp",
        "bf16-abs", "int64-bf16"])
def test_refusals(f, xs, rules, why: str) -> None:
    u = _refused(f, *xs, rules=rules)
    assert u.reason == "numerics_unregistered" and why in u.hint, u
