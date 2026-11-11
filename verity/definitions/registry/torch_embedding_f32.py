"""``nn.Embedding``'s weight gradient on an f32 table as word Definitions: torch 2.13/2.14's CUDA ``embedding_dense_backward``
(Embedding.cu, EmbeddingBackwardKernel.cu) with ``scalar_t`` float, ``padding_idx`` -1 and no ``scale_grad_by_freq``, every add
the GPU's FADD (`F32Add_v2`: round to nearest, nothing flushed, every NaN 0x7FFFFFFF).

    EmbeddingGradF32_v1{M, V, H}            (t: Array<M, I32>, g: Array<M, Array<H, F32>>) -> Array<V, Array<H, F32>>
    EmbeddingGradF32Sort_v1{M, V, H}        (t, g) -> Tuple<Array<M, I32>, Array<M, Array<H, F32>>>
    EmbeddingGradF32Swap_v1{H}              (a, b: I32, x, y: Array<H, F32>) -> Tuple<I32, I32, Array<H, F32>, Array<H, F32>>
    EmbeddingGradF32SegmentStep_v1{V, NP, FEATURE}
                                            (carry, k, nx: I32) -> Tuple<carry, Tuple<Array<2, BIT>, BIT, I32>>
    EmbeddingGradF32Step_v1{FEATURE}        (carry, f: Array<2, BIT>, g: F32) -> Tuple<carry, F32>
    EmbeddingGradF32Run_v1{N, FEATURE}      (carry, f: Array<N, Array<2, BIT>>, g: Array<N, F32>) -> Tuple<carry, Array<N, F32>>
    EmbeddingGradF32Move_v1{H}              (m: I32, pa: BIT, sa: I32, xa: Array<H, F32>, pb: BIT, sb: I32, xb: Array<H, F32>)
                                            -> Tuple<BIT, I32, Array<H, F32>>
    EmbeddingGradF32Slot_v1{H}              (p: BIT, x: Array<H, F32>) -> Array<H, F32>
    EmbeddingGradF32Route_v1{M, V, H}       (p: Array<M, BIT>, s: Array<M, I32>, x: Array<M, Array<H, F32>>)
                                            -> Array<V, Array<H, F32>>

(I32 and F32 are ``Value<32>``, BIT ``Value<1>``.)

The kernels' order is `torch_embedding`'s, there on bf16 words, with ``scalar_t`` float in place of bf16. Up to 3072 ids
``embedding_backward_feature_kernel`` walks the tokens in chunks of 32 consecutive positions; in each chunk the first token of an
id sums the chunk's tokens of that id from its own gradient in position order, and ``grad_weight += sum`` adds it, an FADD on f32,
from the zeroed weight. Above 3072 the ids are stable-sorted; partials of NROWS_PER_THREAD = 10 of an id's tokens are summed from
+0 in order (``weight += g * scale``, scale 1.0: RN(weight + g)), then the id's partials from +0, and the f32 total is stored. An
entry no token has stays +0. `numerics.torch_kernels`' ``embedding_dense_backward`` models both on f32 words.

`EmbeddingGradF32` computes that as `torch_embedding`'s version 2 does on bits, the ids sorted once rather than once per entry,
on the words: `EmbeddingGradF32Sort` sorts the gradient rows by the key ``id · 2^NP + position`` (NP = position_bits(M), an id
outside [0, V) as V, so it lands in no slot) with `torch_embedding.sort_net(M)`, an `EmbeddingGradF32Swap` a comparator; keys
of at most 31 bits compare as `I32Le_v1` reads them. `EmbeddingGradF32SegmentStep`, scanned over the sorted keys, gives each
token its flags (seg: its id's first; start: a partial's first, or with FEATURE a chunk's), mark (its id's last, the id below V)
and shift: z, the unmarked tokens before it, in the low NP bits, and e, its id less the marked tokens before it, above them.
`EmbeddingGradF32Run` is the kernel's sum on one feature's column of the sorted rows, `EmbeddingGradF32Step` a token, whose out
at an id's last token is the entry's word. `EmbeddingGradF32Route` moves each marked token's row down by z (lowest bit first)
and up by e (highest first) to slot id, under which shifts no two rows meet, and +0 fills a slot no row reaches.
"""
from __future__ import annotations

from typing import Any

from verity.circuits.defs import bind, composite
from verity.definitions.ops.fp32 import F32AddV2
from verity.definitions.ops.prims import const
from verity.definitions.ops.scalar import BitAnd, BitNot, BitOr, I32Add, I32Eq, I32Le, SelectF32, SelectI32
from verity.circuits.refs import Coll, Strided, array_of, broadcast, concat, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.evidence import EVIDENCE

from .dsa import I32Sub, U32And
from .flashmla import U32Ge
from .torch_embedding import (CHUNK, FEATURE_MAX, M_MAX, NROWS, _fields, _gather, _refuse_ranges, _statics, id_bits, position_bits,
                              sort_net)

__all__ = ["EmbeddingGradF32", "EmbeddingGradF32Move", "EmbeddingGradF32Route", "EmbeddingGradF32Run",
           "EmbeddingGradF32SegmentStep", "EmbeddingGradF32Slot", "EmbeddingGradF32Sort", "EmbeddingGradF32Step",
           "EmbeddingGradF32Swap", "TORCH_EMBEDDING_F32"]

I32, F32, BIT = Value(32), Value(32), Value(1)
FLAGS = Array(2, BIT)
SEG = Tuple(FLAGS, BIT, I32)
SEG_CARRY = Tuple(I32, I32, I32, I32)
M32 = 0xFFFFFFFF
#: the most bits a sort key may have: an `I32Le_v1` operand that is nonnegative compares as unsigned
KEY_BITS = 31
_CAPTURE = EVIDENCE["cc12_embedding_backward_capture"]
_CONF = ("exact-model-tested: up to FEATURE_MAX ids (embedding_backward_feature_kernel), every captured word of an f32 table's "
         "gradient on cc 12.0 with 188 SMs (torch 2.14.0+cu130), 2,048 ids into V = 32,000 at H 512, 768, 1,024 and 1,536 (each "
         "entry's first 256 features), uniform and Zipf ids, deterministic or not, is numerics.torch_kernels' feature model's "
         f"(capture {_CAPTURE['run_record']}, fit {_CAPTURE['fit']}), and tests/test_torch_embedding_f32.py holds these "
         "Definitions to that model on f32 words; above FEATURE_MAX (the sort kernels) only the model holds them, no capture")


def _c(B, v: int, w: int = 32) -> Coll:
    return B.call(const(w, v & M32))


def _key_refusal(np_: Any, V: Any) -> str | None:
    if isinstance(np_, int) and isinstance(V, int) and np_ + id_bits(V) > KEY_BITS:
        return (f"{np_} position bits and V = {V} make sort keys of {np_ + id_bits(V)} bits, past the {KEY_BITS} a signed "
                "I32Le_v1 compares as unsigned")
    return None


def _refuse_keys(b: dict) -> str | None:
    if r := _refuse_ranges(("M", 1, M_MAX), ("V", 1, M_MAX), ("H", 1, M_MAX))(b):
        return r
    return _key_refusal(position_bits(b["M"]) if isinstance(b.get("M"), int) else None, b.get("V"))


def _grad_statics(S: Any) -> Any:
    if r := _refuse_keys({k: getattr(S, k) for k in ("M", "V", "H")}):
        raise ValueError(r)
    return S


@composite("EmbeddingGradF32Swap", 1, ["H"],
           lambda S: ((("a", I32), ("b", I32), ("x", Array(_statics(S, "H", 1, M_MAX).H, F32)), ("y", Array(S.H, F32))),
                      Tuple(I32, I32, Array(S.H, F32), Array(S.H, F32))),
           conformance=_CONF, refuse=_refuse_ranges(("H", 1, M_MAX)))
def EmbeddingGradF32Swap(B, S, a, b, x, y):
    """One comparator of the sort: the keys a and b and their rows x and y, swapped when a > b (signed), so the lesser key and
    its row come first."""
    s = B.call(BitNot, B.call(I32Le, a, b))
    return tuple_of(B.call(SelectI32, s, b, a), B.call(SelectI32, s, a, b), B.batch(SelectF32, s, y, x, axes=(None, 0, 0)),
                    B.batch(SelectF32, s, x, y, axes=(None, 0, 0)))


def _iota(B, n: int) -> Coll:
    """The I32 words 0 .. n - 1: each `I32Add_v1` batch adds the count so far to the words before it."""
    parts, have = [_c(B, 0).refs], 1
    while have < n:
        m = min(have, n - have)
        parts.append(B.batch(I32Add, Coll(Array(m, I32), concat(parts).slice(0, m)), _c(B, have), axes=(0, None)).refs)
        have += m
    return Coll(Array(n, I32), concat(parts))


@composite("EmbeddingGradF32Sort", 1, ["M", "V", "H"],
           lambda S: ((("t", Array(_grad_statics(S).M, I32)), ("g", Array(S.M, Array(S.H, F32)))),
                      Tuple(Array(S.M, I32), Array(S.M, Array(S.H, F32)))),
           conformance=_CONF, refuse=_refuse_keys)
def EmbeddingGradF32Sort(B, S, t, g):
    """The M tokens' gradient rows sorted by (id, position), each with its key id · 2^NP + position, an id t with t >= V
    unsigned (`U32Ge_v1`) as V: EmbeddingGradF32Swap_v1 on each comparator of sort_net(M)'s stages, a batch a stage."""
    v = _c(B, S.V)
    k = B.batch(SelectI32, B.batch(U32Ge, t, v, axes=(0, None)), v, t, axes=(0, None, 0))
    for _ in range(position_bits(S.M)):
        k = B.batch(I32Add, k, k, axes=(0, 0))
    k = B.batch(I32Add, k, _iota(B, S.M), axes=(0, 0))
    keys, rows = [k[i] for i in range(S.M)], [g[i] for i in range(S.M)]
    swap = bind(EmbeddingGradF32Swap, H=S.H)
    for stage in sort_net(S.M):
        o = B.batch(swap, *(_gather([c[p[e]] for p in stage]) for c, e in ((keys, 0), (keys, 1), (rows, 0), (rows, 1))),
                    axes=(0,) * 4)
        for m, (a, b) in enumerate(stage):
            y = o[m]
            keys[a], keys[b], rows[a], rows[b] = y[0], y[1], y[2], y[3]
    return tuple_of(_gather(keys), _gather(rows))


_SEGMENT = (("V", 1, M_MAX), ("NP", 1, 24), ("FEATURE", 0, 1))


def _refuse_segment(b: dict) -> str | None:
    return _refuse_ranges(*_SEGMENT)(b) or _key_refusal(b.get("NP"), b.get("V"))


def _segment_signature(S: Any):
    for k, lo, hi in _SEGMENT:
        _statics(S, k, lo, hi)
    if r := _key_refusal(S.NP, S.V):
        raise ValueError(r)
    return (("carry", SEG_CARRY), ("k", I32), ("nx", I32)), Tuple(SEG_CARRY, SEG)


@composite("EmbeddingGradF32SegmentStep", 1, ["V", "NP", "FEATURE"], _segment_signature, conformance=_CONF,
           refuse=_refuse_segment)
def EmbeddingGradF32SegmentStep(B, S, carry, k, nx):
    """One sorted token, its key k and the next token's key nx (0xFFFFFFFF after the last, which no id bits equal): its flags
    (seg, start), mark and shift. seg is its id's first token; start a partial's first, every 10th of its id's tokens (with
    FEATURE, the first of its id's tokens in a chunk of 32 positions); mark its id's last, the id below V; shift z + (e << NP),
    z the unmarked tokens before it and e its id less the marked ones. The carry: the token before's id bits (0xFFFFFFFF before
    the first), the count mod 10 or chunk, z and the marked count << NP."""
    prev, st, z, r = (carry[i] for i in range(4))
    lo = (1 << S.NP) - 1
    idk = B.call(U32And, k, _c(B, ~lo))
    same = B.call(I32Eq, idk, prev)
    if S.FEATURE:
        chunk = B.call(U32And, k, _c(B, lo & -CHUNK))
        start = B.call(BitOr, B.call(BitNot, same), B.call(BitNot, B.call(I32Eq, chunk, st)))
        st = chunk
    else:
        n = B.call(SelectI32, same, st, _c(B, 0))
        start = B.call(I32Eq, n, _c(B, 0))
        st = B.call(SelectI32, B.call(I32Eq, n, _c(B, NROWS - 1)), _c(B, 0), B.call(I32Add, n, _c(B, 1)))
    last = B.call(BitNot, B.call(I32Eq, idk, B.call(U32And, nx, _c(B, ~lo))))
    mark = B.call(BitAnd, last, B.call(BitNot, B.call(I32Eq, idk, _c(B, S.V << S.NP))))
    shift = B.call(I32Add, z, B.call(I32Sub, idk, r))
    z = B.call(I32Add, z, B.call(SelectI32, mark, _c(B, 0), _c(B, 1)))
    r = B.call(I32Add, r, B.call(SelectI32, mark, _c(B, 1 << S.NP), _c(B, 0)))
    return tuple_of(tuple_of(idk, st, z, r), tuple_of(array_of([B.call(BitNot, same), start]), mark, shift))


def _carry(feature: int):
    return Tuple(F32, F32, F32) if feature else Tuple(F32, F32)


@composite("EmbeddingGradF32Step", 1, ["FEATURE"],
           lambda S: ((("carry", _carry(_statics(S, "FEATURE", 0, 1).FEATURE)), ("f", FLAGS), ("g", F32)),
                      Tuple(_carry(S.FEATURE), F32)),
           conformance=_CONF, refuse=_refuse_ranges(("FEATURE", 0, 1)))
def EmbeddingGradF32Step(B, S, carry, f, g):
    """One sorted token of one feature, its flags f = (seg, start) and gradient g. The sort path: at start the total += p and
    p = +0 + g, else p += g, the total +0 at seg, out total + p. With FEATURE, the feature kernel: at start the chunk sum
    s = g and acc = the last out (+0 at seg), else s += g, out acc + s. A token that is its id's last has out its entry's word."""
    seg, start, zero = f[0], f[1], _c(B, 0)
    if S.FEATURE:
        acc, s, o = carry[0], carry[1], carry[2]
        acc = B.call(SelectF32, seg, zero, B.call(SelectF32, start, o, acc))
        s = B.call(SelectF32, start, g, B.call(F32AddV2, s, g))
        o = B.call(F32AddV2, acc, s)
        return tuple_of(tuple_of(acc, s, o), o)
    tot, p = carry[0], carry[1]
    r = B.call(F32AddV2, B.call(SelectF32, start, tot, p), B.call(SelectF32, start, p, g))
    tot = B.call(SelectF32, seg, zero, B.call(SelectF32, start, r, tot))
    p = B.call(SelectF32, start, B.call(F32AddV2, zero, g), r)
    return tuple_of(tuple_of(tot, p), B.call(F32AddV2, tot, p))


@composite("EmbeddingGradF32Run", 1, ["N", "FEATURE"],
           lambda S: ((("carry", _carry(_statics(S, "FEATURE", 0, 1).FEATURE)),
                       ("f", Array(_statics(S, "N", 1, M_MAX).N, FLAGS)), ("g", Array(S.N, F32))),
                      Tuple(_carry(S.FEATURE), Array(S.N, F32))),
           conformance=_CONF, refuse=_refuse_ranges(("N", 1, M_MAX), ("FEATURE", 0, 1)))
def EmbeddingGradF32Run(B, S, carry, f, g):
    """N consecutive sorted tokens of one feature: EmbeddingGradF32Step_v1's scan from carry, each token's out."""
    return B.scan(bind(EmbeddingGradF32Step, FEATURE=S.FEATURE), carry, xs=(f, g))


def _row(S: Any):
    return Array(_statics(S, "H", 1, M_MAX).H, F32)


@composite("EmbeddingGradF32Move", 1, ["H"],
           lambda S: ((("m", I32), ("pa", BIT), ("sa", I32), ("xa", _row(S)), ("pb", BIT), ("sb", I32), ("xb", _row(S))),
                      Tuple(BIT, I32, _row(S))),
           conformance=_CONF, refuse=_refuse_ranges(("H", 1, M_MAX)))
def EmbeddingGradF32Move(B, S, m, pa, sa, xa, pb, sb, xb):
    """One place of a routing stage: item a there and item b, each (present p, shift s, row x); b when b is present and moves
    (s & m is not 0, m the stage's bit), else a, present when it stays."""
    zero = _c(B, 0)

    def moves(s):
        return B.call(BitNot, B.call(I32Eq, B.call(U32And, s, m), zero))
    go = B.call(BitAnd, pb, moves(sb))
    p = B.call(BitOr, go, B.call(BitAnd, pa, B.call(BitNot, moves(sa))))
    return tuple_of(p, B.call(SelectI32, go, sb, sa), B.batch(SelectF32, go, xb, xa, axes=(None, 0, 0)))


@composite("EmbeddingGradF32Slot", 1, ["H"], lambda S: ((("p", BIT), ("x", _row(S))), _row(S)), conformance=_CONF,
           refuse=_refuse_ranges(("H", 1, M_MAX)))
def EmbeddingGradF32Slot(B, S, p, x):
    """A slot's row: x when an item is present, else +0, each word ANDed with p's mask (`U32And_v1`)."""
    return B.batch(U32And, B.call(SelectI32, p, _c(B, M32), _c(B, 0)), x, axes=(None, 0))


@composite("EmbeddingGradF32Route", 1, ["M", "V", "H"],
           lambda S: ((("p", Array(_grad_statics(S).M, BIT)), ("s", Array(S.M, I32)), ("x", Array(S.M, Array(S.H, F32)))),
                      Array(S.V, Array(S.H, F32))),
           conformance=_CONF, refuse=_refuse_keys)
def EmbeddingGradF32Route(B, S, p, s, x):
    """The present items (p) of M, each with its row x, moved down by z (s's low position_bits(M) bits, lowest bit first) to
    the first places, then up by e (s's id_bits(V) bits above them, highest first) to V slots, EmbeddingGradF32Move_v1 a
    stage, then EmbeddingGradF32Slot_v1: +0 where no item lands. With z each item's count of absent items before it and e its
    slot less its place among the present items, the slots ascending, no two items meet and item k lands at k - z + e."""
    M, V, H, NP = S.M, S.V, S.H, position_bits(S.M)
    zero = _c(B, 0)
    # absent is computed: a 1-bit constant's Boolean version is the bare primitive, which the bits compiler doesn't price
    gone = (B.call(I32Eq, zero, _c(B, 1)), zero, Coll(Array(H, F32), broadcast(*zero.refs[0], H)))
    move = bind(EmbeddingGradF32Move, H=H)

    def stage(items, src, j: int) -> list:
        o = B.batch(move, _c(B, 1 << j), *(_gather([it[f] for it in items]) for f in range(3)),
                    *(_gather([it[f] for it in src]) for f in range(3)), axes=(None,) + (0,) * 6)
        return [tuple(o[m][f] for f in range(3)) for m in range(len(items))]

    items = [(p[k], s[k], x[k]) for k in range(M)]
    for j in range(NP):
        items = stage(items, [items[k + (1 << j)] if k + (1 << j) < M else gone for k in range(M)], j)
    n = min(M, V)
    items, held = items[:n] + [gone] * (V - n), [k < n for k in range(V)]
    for j in reversed(range(id_bits(V))):
        d = 1 << j
        live = [k for k in range(V) if held[k] or (k >= d and held[k - d])]
        out = stage([items[k] for k in live], [items[k - d] if k >= d and held[k - d] else gone for k in live], NP + j)
        held = [False] * V
        for k, it in zip(live, out):
            items[k], held[k] = it, True
    live = [k for k in range(V) if held[k]]
    rows = B.batch(bind(EmbeddingGradF32Slot, H=H), _gather([items[k][0] for k in live]),
                   _gather([items[k][2] for k in live]), axes=(0, 0))
    at = dict(zip(live, range(len(live))))
    return _gather([rows[at[k]] if k in at else gone[2] for k in range(V)])


@composite("EmbeddingGradF32", 1, ["M", "V", "H"],
           lambda S: ((("t", Array(_grad_statics(S).M, I32)), ("g", Array(S.M, Array(S.H, F32)))), Array(S.V, Array(S.H, F32))),
           conformance=_CONF, refuse=_refuse_keys)
def EmbeddingGradF32(B, S, t, g):
    """The f32 table's gradient over M tokens with ids t (an id outside [0, V) adds nothing): EmbeddingGradF32Sort_v1,
    EmbeddingGradF32SegmentStep_v1's scan over the sorted keys, EmbeddingGradF32Run_v1 on each feature's column of the sorted
    rows (the feature kernel up to 3072 tokens, else the sort path), then EmbeddingGradF32Route_v1 of each id's last token's out
    to the id's slot."""
    M, V, H, NP, F = S.M, S.V, S.H, position_bits(S.M), int(S.M <= FEATURE_MAX)
    srt = B.call(bind(EmbeddingGradF32Sort, M=M, V=V, H=H), t, g)
    keys, rows = srt[0], srt[1]
    zero, none = _c(B, 0), _c(B, M32)
    nx = Coll(keys.type, concat([keys.refs.slice(1, M), none.refs]))
    seg = B.scan(bind(EmbeddingGradF32SegmentStep, V=V, NP=NP, FEATURE=F), tuple_of(none, zero, zero, zero), xs=(keys, nx))[1]
    cw, r = _carry(F).leaves, rows.refs
    o = B.batch(bind(EmbeddingGradF32Run, N=M, FEATURE=F), Coll(Array(H, _carry(F)), broadcast(*zero.refs[0], H * cw)),
                _fields([seg], M, 0, 2, FLAGS), Coll(Array(H, Array(M, F32)), Strided(r.space, r.idx, r.base, M, 1, H, H * M)),
                axes=(0, None, 0)).refs
    out = Coll(Array(M, Array(H, F32)), Strided(o.space, o.idx, o.base + cw, H, 1, cw + M, H * M))
    return B.call(bind(EmbeddingGradF32Route, M=M, V=V, H=H), _fields([seg], M, 2, 1, BIT), _fields([seg], M, 3, 1, I32), out)


TORCH_EMBEDDING_F32 = [EmbeddingGradF32, EmbeddingGradF32Sort, EmbeddingGradF32Swap, EmbeddingGradF32SegmentStep,
                       EmbeddingGradF32Step, EmbeddingGradF32Run, EmbeddingGradF32Move, EmbeddingGradF32Slot,
                       EmbeddingGradF32Route]
