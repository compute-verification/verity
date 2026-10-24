"""`types`' leaf facts of a tuple or a record (`leaves`, `field_types`, `field_starts`, `field_at`, `field_offset`,
`range_widths`, `width_set`) are its fields' leaves, computed once a type and held outside its identity, so equality,
hashing, repr and pickling see its fields alone; and a specialization's id, formatted once when complete, follows its
bindings until then (the decoder fills them after making it)."""
from __future__ import annotations

import dataclasses
import pickle

from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.types import Array, Record, Tuple, Value, field_at, field_starts, field_types, range_widths


def _type() -> Tuple:
    """Nested tuples, a record, and fields of no leaves between and after the others."""
    return Tuple(Array(3, Value(8)), Record(("x", Value(16)), ("e", Tuple()), ("y", Array(2, Value(1)))), Array(0, Value(4)),
                 Value(32), Tuple(Value(2), Array(2, Tuple(Value(4), Value(8)))), Tuple())


def test_a_tuples_leaf_facts_are_its_fields_leaves():
    t = _type()
    widths, sizes = t.leaf_widths(), [len(f.leaf_widths()) for f in t.fields]
    starts = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    assert t.leaves == len(widths) == 12 and field_starts(t) == tuple(starts)
    assert [t.field_offset(i) for i in range(len(sizes))] == starts[:-1]
    assert [field_at(t, leaf) for leaf in range(t.leaves)] == [
        next(i for i in range(len(sizes)) if starts[i] <= leaf < starts[i + 1]) for leaf in range(t.leaves)]
    assert t.width_set() == frozenset(widths)
    for lo in range(t.leaves + 1):
        for hi in range(lo, t.leaves + 1):
            assert range_widths(t, lo, hi) == frozenset(widths[lo:hi]), (lo, hi)
    rec = t.fields[1]
    assert (rec.leaves, field_starts(rec), [field_at(rec, x) for x in range(3)]) == (3, (0, 1, 1, 3), [0, 2, 2])


def test_a_types_held_facts_are_outside_its_identity():
    t, fresh = _type(), _type()
    before = (t == fresh, hash(t) == hash(fresh), repr(t) == repr(fresh))
    range_widths(t, 0, t.leaves)
    assert hasattr(t, "_starts") and not hasattr(fresh, "_starts")
    rec = t.fields[1]
    assert field_types(rec) is field_types(rec) == (Value(16), Tuple(), Array(2, Value(1)))
    assert before == (t == fresh, hash(t) == hash(fresh), repr(t) == repr(fresh)) == (True, True, True)
    back = pickle.loads(pickle.dumps(t))
    assert back == fresh and back.leaves == 12 and range_widths(back, 2, 6) == frozenset({8, 16, 1})
    assert field_types(back.fields[1]) == field_types(fresh.fields[1]) == field_types(rec)
    a = Array(3, Value(8))
    assert (a.leaves, dataclasses.replace(a, n=5).leaves) == (3, 5)


def test_a_specializations_id_follows_its_bindings_until_they_are_complete():
    d = CompositeDefinition("TypesTestId", 1, ("K", "N"), lambda S: ((("x", Value(8)),), Value(8)),
                            lambda B, S, x: x, register=False)
    sp = SpecializedDefinition(d, {}, ())
    assert sp.id == "TypesTestId_v1{}"
    sp.bindings.update(K=4)
    assert sp.id == "TypesTestId_v1{K=4}"
    sp.bindings.update(N=(1, 2))
    assert sp.id == "TypesTestId_v1{K=4,N=(1,2)}" == bind(d, K=4, N=(1, 2)).id
