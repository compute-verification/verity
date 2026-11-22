"""The refusals of `edges.io_cuts`, one case each: the smallest root that trips that clause, asserted by its text, so deleting the
clause fails its case."""
from __future__ import annotations

import pytest

import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
import test_format_spec as spec
import test_restrict as TR
from verity.circuits import edges as EG
from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_chunks, descriptor_id
from verity.circuits.defs import bind
from verity.circuits.program import Program
from verity.circuits.refs import Coll, array_of, tuple_of
from verity.circuits.types import Array, Tuple

V32 = spec.V32
A2, A3, A4 = Array(2, V32), Array(3, V32), Array(4, V32)


def _template(name: str, params, ret, body):
    return bind(spec._comp(name, (), lambda S: (params, ret), body))


W2 = _template("CutW2", (("a", V32), ("b", V32)), A2, lambda B, S, a, b: array_of([B.call(spec.TAdd, a, b), B.call(spec.TAdd, b, a)]))
W4 = _template("CutW4", (("a", V32), ("b", V32)), A4,
               lambda B, S, a, b: array_of([B.call(spec.TAdd, a, b), B.call(spec.TAdd, b, a), B.call(spec.TAdd, a, a), B.call(spec.TAdd, b, b)]))
R3 = _template("CutR3", (("x", A3),), V32, lambda B, S, x: B.call(spec.TAdd, B.call(spec.TAdd, x[0], x[1]), x[2]))
R4 = _template("CutR4", (("x", A4),), V32,
               lambda B, S, x: B.call(spec.TAdd, B.call(spec.TAdd, x[0], x[1]), B.call(spec.TAdd, x[2], x[3])))
Step = _template("CutStep", (("c", V32), ("x", V32), ("k", V32)), Tuple(V32, V32),
                 lambda B, S, c, x, k: (lambda c2: tuple_of(c2, B.call(spec.TAdd, c2, k)))(B.call(spec.TAdd, c, x)))


def _cuts(program: Program, templates, ranges=None):
    D = R.program(b"".join(descriptor_chunks(program)))
    query = {"name": "Q_template_instances", "version": 0, "params": {"templates": [descriptor_id(t) for t in templates]}}
    return EG.io_cuts(D.desc, PO.TemplatePartition(D, query).ranges if ranges is None else ranges)


def _program(name: str, params, ret, body) -> Program:
    return Program(spec._comp(name, (), lambda S: (params, ret), body))


def test_a_reader_taking_its_rows_from_a_writer_whose_field_does_not_divide_the_port_is_refused():
    def body(B, S, a, b):
        ys = B.batch(W2, a, b, axes=(0, 0))
        return B.call(R3, Coll(A3, ys.refs.slice(0, 3)))
    program = _program("CutRows", (("a", A2), ("b", A2)), V32, body)
    with pytest.raises(ValueError) as e:
        _cuts(program, [W2, R3])
    assert str(e.value) == "edge-rows"


def test_a_port_read_in_rows_of_two_lengths_is_refused():
    def body(B, S, a, b, c, d):
        ys = B.batch(W2, a, b, axes=(0, 0))
        r1 = B.call(R4, Coll(A4, ys.refs.slice(0, 4)))
        r2 = B.call(R4, B.call(W4, c, d))
        return array_of([r1, r2])
    program = _program("CutTwoLengths", (("a", A2), ("b", A2), ("c", V32), ("d", V32)), A2, body)
    rt = descriptor_id(R4)
    with pytest.raises(ValueError) as e:
        _cuts(program, [W2, W4, R4])
    assert str(e.value) == f"ioCuts: template {rt}'s port x is read in rows of 2 and of 4 leaves"


def test_a_root_node_with_no_range_is_refused_as_no_template():
    _, c, _ = TR.toy("fanout")
    with pytest.raises(ValueError) as e:
        EG.io_cuts(R.program(c).desc, [])
    assert str(e.value) == "edge-template"


def test_a_scan_at_the_root_reading_a_node_is_not_derived():
    def body(B, S, a, b, x, k):
        return B.scan(Step, B.call(spec.TAdd, a, b), xs=[x], shared=[k])
    program = _program("CutScan", (("a", V32), ("b", V32), ("x", A3), ("k", V32)), Tuple(V32, A3), body)
    D = R.program(b"".join(descriptor_chunks(program)))
    add = next(k for k, nd in enumerate(D.root.body.nodes) if nd.fn.id == "TAdd32_v1")
    with pytest.raises(ValueError) as e:
        EG.io_cuts(D.desc, [(add, "TAdd32_v1", 0, 1)])
    assert str(e.value) == "one-circuit: not derived yet: ioCuts : a scan node of C's root (edge-form: no closed form yet)"
