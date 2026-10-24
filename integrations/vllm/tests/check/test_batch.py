"""verity_vllm.check.batch: (a) an owner's outputs read only its own and shared leaves, (b) the launch doesn't pick the Definition.

Each property has its negatives: a template whose row 0 reads row 1 (a cross-row mean), one that reads a batch-derived operand, and
one whose Definition changes with the launch's row count (a split count read from the batch, the way SplitsFor_v1 picks S).
"""
from __future__ import annotations

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.check.batch import (BATCH, SHARED, Fields, Rows, Template, check, held, invariant, own, report, separable,
                                     structure)

V16 = Value(16)
ADD = PrimitiveDefinition("BatchTestAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)
MUL = PrimitiveDefinition("BatchTestMul16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a * b) & 0xFFFF, register=False)


def _rowwise(B, S, x, w):
    return array_of([array_of([B.call(MUL, x[i][j], w[j]) for j in range(S.H)]) for i in range(S.M)])


def _cross(B, S, x, w):
    """Each row's element j plus row 0's element j: every row reads row 0, row 0 reads only itself."""
    return array_of([array_of([B.call(ADD, x[i][j], x[0][j]) for j in range(S.H)]) for i in range(S.M)])


def _scaled(B, S, x, s):
    return array_of([array_of([B.call(MUL, x[i][j], s) for j in range(S.H)]) for i in range(S.M)])


def _sig(S):
    return (("x", Array(S.M, Array(S.H, V16))), ("w", Array(S.H, V16))), Array(S.M, Array(S.H, V16))


ROWWISE = CompositeDefinition("BatchTestRowwise", 1, ("M", "H"), _sig, _rowwise, register=False)
CROSS = CompositeDefinition("BatchTestCross", 1, ("M", "H"), _sig, _cross, register=False)
SCALED = CompositeDefinition("BatchTestScaled", 1, ("M", "H"),
                             lambda S: ((("x", Array(S.M, Array(S.H, V16))), ("s", V16)), Array(S.M, Array(S.H, V16))), _scaled,
                             register=False)


def _split(B, S, x):
    """A row sum in S parts, the parts summed in order: S picks the association, as the top-p split count picks the nucleus."""
    n = S.H
    parts = []
    for k in range(S.S):
        lo, hi = k * n // S.S, (k + 1) * n // S.S
        acc = x[lo]
        for j in range(lo + 1, hi):
            acc = B.call(ADD, acc, x[j])
        parts.append(acc)
    out = parts[0]
    for p in parts[1:]:
        out = B.call(ADD, out, p)
    return out


SPLIT_SUM = CompositeDefinition("BatchTestSplitSum", 1, ("H", "S"), lambda S: ((("x", Array(S.H, V16)),), V16), _split, register=False)
#: a row sum whose S static is a no-op: two ids, one structure
PLAIN_SUM = CompositeDefinition("BatchTestPlainSum", 1, ("H", "TAG"), lambda S: ((("x", Array(S.H, V16)),), V16),
                                lambda B, S, x: _split(B, type("S", (), {"H": S.H, "S": 1}), x), register=False)


def _splits_for(rows: int) -> int:
    return 4 if rows <= 2 else 2 if rows <= 4 else 1


def test_a_rowwise_template_is_separable():
    fn = bind(ROWWISE, M=3, H=4)
    assert separable(fn, (Rows(("a", "b", "c")), SHARED), Rows(("a", "b", "c"))) == []


def test_rows_of_one_owner_may_read_each_other():
    fn = bind(CROSS, M=3, H=2)
    assert separable(fn, (Rows(("a", "a", "a")), SHARED), Rows(("a", "a", "a"))) == []


def test_negative_a_cross_row_template_reads_another_owners_row():
    fn = bind(CROSS, M=3, H=2)
    found = separable(fn, (Rows(("a", "b", "c")), SHARED), Rows(("a", "b", "c")))
    assert [(c.owner, c.param, c.holder, c.leaves) for c in found] == [("b", "x", "a", ((0, 2),)), ("c", "x", "a", ((0, 2),))]
    assert "of owner 'b'" in str(found[0]) and "held by owner 'a'" in str(found[0])


def test_negative_a_batch_operand_is_read():
    fn = bind(SCALED, M=2, H=3)
    found = separable(fn, (Rows(("a", "b")), BATCH), Rows(("a", "b")))
    assert [(c.owner, c.param, c.holder) for c in found] == [("a", "s", BATCH), ("b", "s", BATCH)]
    assert "batch-derived" in str(found[0])
    assert separable(fn, (Rows(("a", "b")), SHARED), Rows(("a", "b"))) == []


def test_an_unread_batch_operand_passes():
    sig = lambda S: ((("x", Array(2, V16)), ("s", V16)), Array(2, V16))  # noqa: E731
    fn = bind(CompositeDefinition("BatchTestIgnores", 1, (), sig, lambda B, S, x, s: array_of([B.call(ADD, x[0], x[1])] * 2),
                                  register=False))
    assert separable(fn, (own("a"), BATCH), own("a")) == []


def test_a_primitive_reads_every_parameter():
    assert [(c.owner, c.param) for c in separable(ADD, (own("a"), own("b")), own("a"))] == [("a", "b")]
    assert separable(ADD, (own("a"), SHARED), own("a")) == []


def test_fields_label_a_tuple_return_per_owner():
    sig = lambda S: ((("x", V16), ("y", V16), ("w", V16)), Tuple(V16, V16))  # noqa: E731
    fn = bind(CompositeDefinition("BatchTestPair", 1, (), sig,
                                  lambda B, S, x, y, w: tuple_of(B.call(MUL, x, w), B.call(ADD, x, y)), register=False))
    found = separable(fn, (own("a"), own("b"), SHARED), Fields((own("a"), own("b"))))
    assert [(c.owner, c.param, c.holder) for c in found] == [("b", "x", "a")]


def test_labels_must_fit_the_types():
    with pytest.raises(ValueError, match="equal rows"):
        held(Rows(("a", "b")), Array(3, V16))
    with pytest.raises(ValueError, match="field labels"):
        held(Fields((own("a"),)), Tuple(V16, V16))
    with pytest.raises(ValueError, match="labels for"):
        separable(bind(ROWWISE, M=1, H=1), (own("a"),), own("a"))
    with pytest.raises(ValueError, match="held by owners"):
        separable(bind(ROWWISE, M=1, H=1), (own("a"), SHARED), SHARED)


def test_one_structure_under_two_ids():
    a, b = bind(PLAIN_SUM, H=5, TAG="x"), bind(PLAIN_SUM, H=5, TAG="y")
    assert a.id != b.id and structure(a) == structure(b)
    assert structure(bind(SPLIT_SUM, H=5, S=1)) == structure(a)
    assert structure(bind(SPLIT_SUM, H=5, S=2)) != structure(a)


def test_a_launch_free_binding_is_invariant():
    shapes = [((), m) for m in range(1, 9)]
    assert invariant(lambda req, m: bind(SPLIT_SUM, H=8, S=4), shapes) == []
    assert invariant(lambda req, m: bind(PLAIN_SUM, H=8, TAG=f"rows={m}"), shapes) == []


def test_negative_a_batch_size_dependent_template_splits():
    shapes = [((), m) for m in range(1, 9)]
    found = invariant(lambda req, m: bind(SPLIT_SUM, H=8, S=_splits_for(m)), shapes)
    assert len(found) == 1 and found[0].launches == (1, 3)
    assert "S=4" in found[0].ids[0] and "S=2" in found[0].ids[1]
    assert len(invariant(lambda req, m: bind(SPLIT_SUM, H=8, S=_splits_for(m)), shapes, limit=100)) == 6


def test_request_coordinates_are_compared_apart():
    shapes = [(h, m) for h in (4, 8) for m in (1, 2)]
    assert invariant(lambda h, m: bind(SPLIT_SUM, H=h, S=2), shapes) == []


def test_check_and_report_hold_verdicts_to_the_declared_ones():
    shapes = tuple(((), m) for m in range(1, 9))
    ts = [Template("rowwise", bind(ROWWISE, M=2, H=2), (Rows(("a", "b")), SHARED), Rows(("a", "b")), expect=(True, None)),
          Template("cross", bind(CROSS, M=2, H=2), (Rows(("a", "b")), SHARED), Rows(("a", "b")), expect=(False, None),
                   why="row 1 reads row 0"),
          Template("split", None, bind_at=lambda req, m: bind(SPLIT_SUM, H=8, S=_splits_for(m)), shapes=shapes, expect=(None, False)),
          Template("split-declared-invariant", None, bind_at=lambda req, m: bind(SPLIT_SUM, H=8, S=_splits_for(m)), shapes=shapes,
                   expect=(None, True))]
    vs = check(ts)
    assert [v.got for v in vs] == [(True, None), (False, None), (None, False), (None, False)]
    assert [v.ok for v in vs] == [True, True, True, False]
    text = report(vs)
    assert "| `cross` | **fail**: the outputs of owner 'b'" in text and text.count("**NO**") == 1
    assert vs[1].to_json()["crossings"] and vs[2].to_json()["splits"][0].startswith("at ()")
