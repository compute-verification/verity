"""A registered table taken lazily (`verity.ml.circuits.evaluation.RegisteredTable`): `evaluate` gives what it gives on the table's flat words,
bit for bit, reading only the words the gates ask for; only a table a primitive declares ``registered`` is taken this way, and
every array evaluator refuses it by name."""
from __future__ import annotations

import pytest

from verity.ml.circuits.evaluation import RegisteredTable, ScalarReference, evaluate, evaluate_batch, register_kernel, self_check
from verity.ml.circuits.evaluation.batch import _KERNELS
from verity.ml.circuits.evaluation.bits import evaluate_bits, evaluate_words
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.layout import WIDE_FAN_IN
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value
from verity.ml.circuits.ops import scalar as S

ROW = 64
NAME = "t_small"


def _entry(i: int) -> int:
    return (i * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF


def _table_type(ib: int) -> Array:
    return Array((2 << ib) // ROW, Array(ROW, Value(16)))


def _words(ib: int) -> list[int]:
    return [(_entry(i) >> (16 * k)) & 0xFFFF for i in range(1 << ib) for k in range(2)]


def _read(ib: int) -> PrimitiveDefinition:
    """(index, table) -> the two-word entry at the index's low ``ib`` bits, as `TableRead` reads it."""
    mask = (1 << ib) - 1

    def ev(idx: int, *words: int) -> int:
        e = (idx & mask) * 2
        return words[e] | words[e + 1] << 16

    def ev_lazy(get) -> int:
        e = 1 + (get(0) & mask) * 2
        return get(e) | get(e + 1) << 16

    p = PrimitiveDefinition(f"TRead{ib}x2", 1, (("index", Value(32)), ("table", _table_type(ib))), Value(32), ev, register=False)
    p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
    return p


#: the primitive that holds `t_small` (its evaluator is the table), declaring it a registered input of the program
HOLDS = PrimitiveDefinition("THolds", 1, (("x", Value(32)),), Value(32), lambda x: _entry(x), register=False,
                            tables=((NAME, "registered"), ("t_lib", "library")))
WIDEN = PrimitiveDefinition("TWiden16", 1, (("a", Value(16)),), Value(32), lambda a: a & 0xFFFF, register=False)


def _uses(ib: int, n: int) -> CompositeDefinition:
    """(xs, table) -> [TRead(xs[i], table) + xs[i] for each i] + [the table's last leaf]: a batch sharing the table, and one
    leaf read by an ordinary gate."""
    read = _read(ib)

    def body(B, _S, xs, table):
        got = B.batch(read, xs, table, axes=(0, None))
        sums = B.batch(S.I32Add, got, xs, axes=(0, 0))
        last = B.call(WIDEN, table[table.type.n - 1][ROW - 1])
        return coll_concat(Array(n + 1, Value(32)), [sums, last])

    return CompositeDefinition(f"TUses{ib}x{n}", 1, (), lambda _S: ((("xs", Array(n, Value(32))), ("table", _table_type(ib))),
                                                                    Array(n + 1, Value(32))), body, register=False)


def _lazy(ib: int, reads: list[int] | None = None, name: str = NAME, declared_by: PrimitiveDefinition = HOLDS) -> RegisteredTable:
    words = _words(ib)

    def word(j: int) -> int:
        if reads is not None:
            reads.append(j)
        return words[j]

    return RegisteredTable(name, word, len(words), declared_by)


#: a narrow table (its reads are tuples) and one past the layout's wide fan-in (its reads are `LazyOperands`)
SIZES = [6, 12]


def test_the_wide_table_is_past_the_layouts_wide_fan_in():
    assert 2 << SIZES[0] <= WIDE_FAN_IN < 2 << SIZES[-1]


@pytest.mark.parametrize("ib", SIZES)
def test_a_registered_table_evaluates_both_ways_bit_for_bit(ib):
    """A small table's entries (all 64 of the narrow one, every 64th and the last of the wide one), read by a primitive's
    `evaluate_lazy` and by a composite's batch (plus one leaf an ordinary gate reads), give the flattened form's words; the lazy
    form reads only the words the gates read."""
    words = _words(ib)
    read = _read(ib)
    entries = sorted({*range(0, 1 << ib, max(1, (1 << ib) // 64)), (1 << ib) - 1})
    for i in entries:
        x = i | (0xA5 << 24)
        reads: list[int] = []
        flat = evaluate(read, x, words)
        assert flat == (_entry(i),)
        assert evaluate(read, x, _lazy(ib, reads)) == flat
        assert sorted(reads) == [2 * i, 2 * i + 1]
    n = 5
    uses = bind(_uses(ib, n))
    for start in entries[::4]:
        xs = [((start + 3 * k) % (1 << ib)) | (k << 28) for k in range(n)]
        reads = []
        flat = evaluate(uses, xs, words)
        assert flat == tuple((_entry(x & ((1 << ib) - 1)) + x) & 0xFFFFFFFF for x in xs) + (words[-1],)
        assert evaluate(uses, xs, _lazy(ib, reads)) == flat
        assert sorted(set(reads)) == sorted({2 * (x & ((1 << ib) - 1)) + k for x in xs for k in (0, 1)} | {len(words) - 1})


def test_a_read_past_either_end_is_the_flattened_lists():
    """`evaluate_lazy` indexes the operands as the flat list does: -1 is the last leaf, and one past the end raises IndexError."""
    def lazy_at(k):
        p = PrimitiveDefinition("TAt", 1, (("index", Value(32)), ("table", _table_type(6))), Value(32), lambda *_: 0, register=False)
        p.evaluate_lazy = lambda get: get(k)  # type: ignore[attr-defined]
        return p

    words = _words(6)
    assert evaluate(lazy_at(-1), 7, words) == evaluate(lazy_at(-1), 7, _lazy(6)) == (words[-1],)
    assert evaluate(lazy_at(-len(words) - 1), 7, words) == evaluate(lazy_at(-len(words) - 1), 7, _lazy(6)) == (7,)
    for table in (words, _lazy(6)):
        with pytest.raises(IndexError):
            evaluate(lazy_at(len(words) + 1), 7, table)


def test_only_a_table_a_primitive_declares_registered_is_taken_lazily():
    with pytest.raises(ValueError, match="'t_lib' is not a registered table of THolds_v1: it declares it as a library table"):
        _lazy(6, name="t_lib")
    with pytest.raises(ValueError, match="'t_other' is not a registered table of THolds_v1: it declares no table of that name"):
        _lazy(6, name="t_other")
    with pytest.raises(ValueError, match="'t_small' is not a registered table of TWiden16_v1"):
        _lazy(6, declared_by=WIDEN)


def test_a_registered_table_binds_a_whole_aggregate_parameter_of_one_width():
    read = _read(6)
    with pytest.raises(TypeError, match="argument 0: the registered table 't_small' .* binds an aggregate parameter"):
        evaluate(read, _lazy(6), _words(6))
    with pytest.raises(TypeError, match="argument 1: the registered table 't_small' \\(128 leaves\\)"):
        evaluate(_read(12), 0, _lazy(6))
    with pytest.raises(TypeError, match="TRead6x2_v1: expected 2 arguments, got 1"):
        evaluate(read, _lazy(6))


def test_a_primitive_without_evaluate_lazy_refuses_a_registered_table():
    p = PrimitiveDefinition("TReadEager", 1, (("index", Value(32)), ("table", _table_type(6))), Value(32),
                            lambda idx, *w: w[idx & 63], register=False)
    with pytest.raises(TypeError, match="TReadEager_v1 has no evaluate_lazy"):
        evaluate(p, 0, _lazy(6))


def test_a_scalar_reference_refuses_a_registered_table():
    ref = ScalarReference("test.sum", lambda idx, table: sum(table) & 0xFFFFFFFF,
                          lambda: ((Value(32), _table_type(6)), Value(32)))
    assert evaluate(ref, 0, _words(6)) == (sum(_words(6)) & 0xFFFFFFFF,)
    with pytest.raises(TypeError, match="test.sum is a ScalarReference"):
        evaluate(ref, 0, _lazy(6))


def test_the_array_evaluators_refuse_a_registered_table_by_name(monkeypatch):
    read = _read(6)
    with pytest.raises(TypeError, match="TRead6x2_v1 argument 1: evaluate_batch takes arrays, not the registered table 't_small'"):
        evaluate_batch(read, [0, 1], _lazy(6))
    monkeypatch.setattr("verity.ml.circuits.evaluation.batch._KERNELS", dict(_KERNELS))
    register_kernel(read, lambda idx, table: NotImplemented)
    with pytest.raises(TypeError, match="TRead6x2_v1 argument 1: self_check takes arrays, not the registered table 't_small'"):
        self_check(read, inputs=[[0, 1], _lazy(6)])
    and2 = CompositeDefinition("TAnd", 1, (), lambda _S: ((("x", Array(2, BL.BIT)),), BL.BIT), lambda B, _S, x: B.call(BL.AND, x),
                               register=False)
    with pytest.raises(TypeError, match="argument 0: evaluate_bits takes arrays, not the registered table 't_small'"):
        evaluate_bits(and2, _lazy(6))
    with pytest.raises(TypeError, match="argument 0: evaluate_words takes arrays, not the registered table 't_small'"):
        evaluate_words(and2, _lazy(6))
