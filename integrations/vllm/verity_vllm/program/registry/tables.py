"""The tables a program brings, as registered values, and the read that takes one as an operand (Daniel, 1 Oct: a table a
program brings is a value, committed and always hidden; the measured MUFU tables stay gates of their operations).

A table of 2^IB entries of EW u16 words (an entry's low word first) is the registered parameter `table_type(name)`: rows of
`ROW_WORDS` words, the shape `verity_flock.hidden_gather` registers and opens (`TABLES` here is its `TABLES`'s
`(index_bits, entry_words)`):

    gelu_tanh_bf16   2^16 entries, one word each (1,024 rows): `dense.gelu_tanh_bf16_bits(g)`, the activation word of a bf16 g
    tanh_rn          2^27 entries, two words each (2^22 rows): RN32(tanh) at |x|'s low 27 bits (`tanh_rn_entry`)

    TableRead{IB}x{EW}_v1  (index: Value<16> if IB <= 16 else Value<32>, table: Array<2^IB EW / 64, Array<64, Value<16>>>)
                           -> Value<16 EW>: the entry at the index's low IB bits, its words joined low first; total

`GeluTanhMulBf16_v3` / `GeluTanhMul_v3{I}` are `GeluTanhMulBf16_v1` / `GeluTanhMul_v1{I}` with the activation read from the
registered `gelu_tanh_bf16` rather than computed by the primitive's evaluator.  `TanhF32Rn_v2` is `TanhF32Rn_v1` with |x| in
[LO, HI) read from the registered `tanh_rn`, as C-Flock's piece computes it (`verity_flock.tail_pieces.tanh_rn`); its 2^28 words
are read lazily (`verity.ml.circuits.evaluation.RegisteredTable`, `tanh_rn_table`), never flattened.
"""
from __future__ import annotations

from functools import lru_cache

from verity.ml.circuits.evaluation import RegisteredTable
from verity.ml.circuits.codec import register_lazy_family
from verity.ml.circuits.defs import PrimitiveDefinition, composite
from verity.ml.circuits.types import Array, Value
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.prims import const
from verity_vllm.program.registry import dense
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry.prims import BF16

ROW_WORDS, WORD_BITS = 64, 16
TABLES = {"gelu_tanh_bf16": (16, 1), "tanh_rn": (27, 2)}
#: `tanh_rn`'s region: identity below LO, ±1 from HI (`verity_flock.tail_pieces.TANH_BOUNDS["tanh_rn"]`)
TANH_RN_LO, TANH_RN_HI = 0x39B89BA3, 0x41102CB4
#: `tanh_rn`'s entries by where their words come from: the words 0x40000000 | i below HI's index, none (0) up to LO's, then the
#: words 0x38000000 | i
TANH_RN_REGIONS = ((0, TANH_RN_HI & 0x7FFFFFF), (TANH_RN_HI & 0x7FFFFFF, TANH_RN_LO & 0x7FFFFFF), (TANH_RN_LO & 0x7FFFFFF, 1 << 27))


def table_type(name: str) -> Array:
    """The registered parameter of table `name`: its u16 words in rows of `ROW_WORDS`."""
    ib, ew = TABLES[name]
    return Array((ew << ib) // ROW_WORDS, Array(ROW_WORDS, Value(WORD_BITS)))


_READS: dict[tuple[int, int], PrimitiveDefinition] = {}


def table_read(index_bits: int, entry_words: int) -> PrimitiveDefinition:
    """`TableRead{IB}x{EW}_v1`: (index, table) -> the entry at the index's low IB bits, its EW u16 words joined low first."""
    key = (int(index_bits), int(entry_words))
    p = _READS.get(key)
    if p is None:
        ib, ew = key
        if not 0 < ib <= 32 or ew < 1 or ROW_WORDS % ew or (ew << ib) % ROW_WORDS:
            raise ValueError(f"TableRead{ib}x{ew}: 2^{ib} entries of {ew} words do not fill rows of {ROW_WORDS}")
        mask = (1 << ib) - 1

        def ev(idx: int, *words: int, _ew=ew, _m=mask) -> int:
            e = (idx & _m) * _ew
            return sum((words[e + k] & 0xFFFF) << (16 * k) for k in range(_ew))

        def ev_lazy(get, _ew=ew, _m=mask) -> int:
            e = 1 + (get(0) & _m) * _ew
            return sum((get(e + k) & 0xFFFF) << (16 * k) for k in range(_ew))

        rows = (ew << ib) // ROW_WORDS
        p = PrimitiveDefinition(f"TableRead{ib}x{ew}", 1, (("index", Value(16 if ib <= 16 else 32)),
                                                          ("table", Array(rows, Array(ROW_WORDS, Value(WORD_BITS))))),
                                Value(16 * ew), ev, doc=table_read.__doc__ or "",
                                conformance="exact by construction: an index into the operand")
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _READS[key] = p
    return p


register_lazy_family(r"^TableRead(\d+)x(\d+)_v1$", lambda m: table_read(int(m[1]), int(m[2])))


@lru_cache(maxsize=None)
def gelu_tanh_bf16_words() -> tuple[int, ...]:
    """The registered `gelu_tanh_bf16`: `dense.gelu_tanh_bf16_bits(g)` for every bf16 word g, in index order."""
    return tuple(dense.gelu_tanh_bf16_bits(g) for g in range(1 << 16))


def tanh_rn_entry(i: int) -> int:
    """`tanh_rn`'s entry `i`: `TanhF32Rn_v1` at the region's word whose low 27 bits are `i` (each such index names one word of
    [LO, HI)), 0 at an index no word of the region has."""
    w = 0x38000000 | i if (0x38000000 | i) >= TANH_RN_LO else 0x40000000 | i
    return dense.tanh_f32_bits(w) if TANH_RN_LO <= w < TANH_RN_HI else 0


def table_words(name: str, entry=None):
    """Table `name`'s registered words, in leaf order (an entry's low word first), from `entry(i)` (default: the table's own)."""
    ib, ew = TABLES[name]
    if entry is None:
        if name != "gelu_tanh_bf16":
            raise ValueError(f"{name}: 2^{ib} entries are not materialised as a word list; read them by entry")
        return list(gelu_tanh_bf16_words())
    return [(entry(i) >> (16 * k)) & 0xFFFF for i in range(1 << ib) for k in range(ew)]


def tanh_rn_table() -> RegisteredTable:
    """The registered `tanh_rn` as `verity.ml.circuits.evaluation.evaluate` reads it: leaf j is word j & 1 of `tanh_rn_entry(j >> 1)`, computed
    when a gate reads it."""
    return RegisteredTable("tanh_rn", lambda j: (tanh_rn_entry(j >> 1) >> (16 * (j & 1))) & 0xFFFF, 2 << 27, dense.TanhF32Rn)


GELU_TABLE = table_type("gelu_tanh_bf16")
TANH_RN_TABLE = table_type("tanh_rn")


@composite("GeluTanhMulBf16", 3, [], lambda S: ((("g", BF16), ("u", BF16), ("gelu_tanh_bf16", GELU_TABLE)), BF16),
           conformance="exact-model-tested: GeluTanhMulBf16_v1 with the activation word read from the registered table")
def GeluTanhMulBf16V3(B, S, g, u, gelu_tanh_bf16):
    """`bf16(f32(gelu_tanh_bf16[g]) * (f32(u) + 0.0f))` (`cvt.rn.bf16.f32`, NaN -> 0x7FFF): `GeluTanhMulBf16_v1`."""
    act = B.call(table_read(16, 1), g, gelu_tanh_bf16)
    uf = B.call(P.F32Add, B.call(P.Bf16ToF32, u), B.call(const(32, dense.ZERO_BITS)))
    return B.call(P.F2fpBf16, B.call(P.F32Mul, B.call(P.Bf16ToF32, act), uf))


@composite("GeluTanhMul", 3, ["I"], lambda S: ((("gu", Array(2 * S.I, BF16)), ("gelu_tanh_bf16", GELU_TABLE)), Array(S.I, BF16)),
           conformance="exact-model-tested: GeluTanhMul_v1's batch over GeluTanhMulBf16_v3, every element reading the one table")
def GeluTanhMulV3(B, S, gu, gelu_tanh_bf16):
    """`out[i] = GeluTanhMulBf16_v3(gu[i], gu[I + i], gelu_tanh_bf16)`."""
    return B.batch(GeluTanhMulBf16V3, gu[0:S.I], gu[S.I:2 * S.I], gelu_tanh_bf16, axes=(0, 0, None))


@composite("TanhF32Rn", 2, [], lambda S: ((("x", P.F32), ("tanh_rn", TANH_RN_TABLE)), P.F32),
           conformance="exact-model-tested: TanhF32Rn_v1 with |x| in [LO, HI) read from the registered table")
def TanhF32RnV2(B, S, x, tanh_rn):
    """NaN -> 0x7FC00000, |x| >= HI -> ±1, |x| < LO -> x, else `tanh_rn[|x|'s low 27 bits]` with x's sign: `TanhF32Rn_v1`."""
    def c(w):
        return B.call(const(32, w))

    a = B.call(SC.F32Fabs, x)
    t = B.call(table_read(27, 2), x, tanh_rn)
    neg = B.call(P.I32Le, x, c(0xFFFFFFFF))
    r = B.call(P.SelectF32, B.call(P.I32Le, a, c(TANH_RN_LO - 1)), x, B.call(P.SelectF32, neg, B.call(SC.F32Neg, t), t))
    r = B.call(P.SelectF32, B.call(P.I32Le, c(TANH_RN_HI), a), B.call(P.SelectF32, neg, c(0xBF800000), c(0x3F800000)), r)
    return B.call(P.SelectF32, B.call(P.I32Le, c(0x7F800001), a), c(P.F32_NAN), r)


__all__ = ["GELU_TABLE", "GeluTanhMulBf16V3", "GeluTanhMulV3", "ROW_WORDS", "TABLES", "TANH_RN_HI", "TANH_RN_LO", "TANH_RN_REGIONS",
           "TANH_RN_TABLE", "TanhF32RnV2", "WORD_BITS", "gelu_tanh_bf16_words", "table_read", "table_type", "table_words",
           "tanh_rn_entry", "tanh_rn_table"]
