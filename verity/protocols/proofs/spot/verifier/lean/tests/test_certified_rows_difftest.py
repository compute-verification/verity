"""The certified rows' reader (`Spot.CertifiedRows`: `marks`, `layout`, `rowsAt`, `valueAt`, `labelsHeld`) held to its
Python (`one_stage.certified_rows`): the ``certified_rows`` piece of the difftest's executable answers, on a Program's
descriptor, its templates and the verifier's certified entries and labels, the marks or the refusal's codes, and with
marks the layout, every unit's rows, each label's value at every row and one past, and whether the labels are held; the
reference below decodes the same descriptor and asks the Python. The Programs are built here in the circuit language:
two tables, a call beside a batch under one label, a label cut across two values and an output row, each refused shape a
one-line change of it, and generated ones over its batch sizes, marks and labels.
"""

from __future__ import annotations

import copy
import shutil

import pytest
from hypothesis import strategies as st

from test_wires_difftest import PACKAGE, PIECES

import verity_difftest as D
from verity.circuits.codec import decode_program, descriptor_id, encode_program
from verity.circuits.defs import REGISTRY, CompositeDefinition, bind, primitive
from verity.circuits.partition_object import evaluate, template_instances_query
from verity.circuits.ports import In, Leaves, Out, PortError, Ports
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.protocols.proofs.spot.verifier.one_stage import certified_rows as C

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

V16 = Value(16)
ROW = Array(2, V16)


@primitive("CertRowsTestAdd16", 1, (("a", V16), ("b", V16)), V16)
def Add16(a, b):
    return (a + b) & 0xFFFF


def _a(ports=None, name: str = "CertRowsTestA"):
    """Template `A`: parameters `row` (a row of two words) and `x`; by default `row` is a row of ``in``."""
    return bind(CompositeDefinition(name, 0, (), lambda S: ((("row", ROW), ("x", V16)), V16),
                                    lambda B, S, row, x: B.call(Add16, B.call(Add16, row[0], row[1]), x),
                                    register=False, ports=Ports(params={"row": In(certified="in")}) if ports is None else ports))


def _b(ports=None, name: str = "CertRowsTestB"):
    """Template `B`: parameter `y`, return `(o0, o1)`; by default `o1` is a row of ``out``."""
    return bind(CompositeDefinition(name, 0, (), lambda S: ((("y", V16),), Tuple(V16, V16)),
                                    lambda B, S, y: tuple_of(B.call(Add16, y, y), B.call(Add16, y, B.call(Add16, y, y))),
                                    register=False, ports=Ports(ret=(None, Out(certified="out"))) if ports is None else ports))


def _program(a=None, b=None, n_a: int = 5, n_b: int = 6) -> Program:
    """The root: a call of `A`, a batch of `n_a` `A` and a batch of `n_b` `B`."""
    a, b = a or _a(), b or _b()

    def sig(S):
        return ((("r0", ROW), ("rs", Array(n_a, ROW)), ("x", V16), ("ys", Array(n_b, V16))),
                Tuple(V16, Array(n_a, V16), Array(n_b, Tuple(V16, V16))))

    def body(B, S, r0, rs, x, ys):
        return tuple_of(B.call(a, r0, x), B.batch(a, rs, x, axes=(0, None), n=n_a), B.batch(b, ys, axes=(0,), n=n_b))

    return Program(bind(CompositeDefinition("CertRowsTestRoot", 0, (), sig, body, register=False)))


def _nested() -> Program:
    """A mark on a Definition only a template calls: `Outer` calls the marked `A`, and the root calls `Outer`."""
    outer = bind(CompositeDefinition("CertRowsTestOuter", 0, (), lambda S: ((("row", ROW), ("x", V16)), V16),
                                     lambda B, S, row, x: B.call(_a(), row, x), register=False))

    def sig(S):
        return (("r0", ROW), ("x", V16)), V16

    return Program(bind(CompositeDefinition("CertRowsTestNestedRoot", 0, (), sig, lambda B, S, r0, x: B.call(outer, r0, x),
                                            register=False)))


def _entry(root: str, leaves: int) -> dict:
    return {"root": root, "leaves": leaves}


#: label ``in`` is two values of three rows each, ``out`` one of six
CERTIFIED = {"link/ingress/0": _entry("aa", 3), "link/ingress/1": _entry("bb", 3), "link/egress/0": _entry("cc", 6)}
LABELS = {"in": ["link/ingress/0", "link/ingress/1"], "out": ["link/egress/0"]}


def _case(prog: Program, certified=None, labels=None, templates=None) -> dict:
    keys = sorted({descriptor_id(n.fn) for n in prog.root.body.nodes if hasattr(n.fn, "body")})
    return {"program": encode_program(prog), "templates": templates if templates is not None else keys,
            "certified": CERTIFIED if certified is None else certified, "labels": LABELS if labels is None else labels}


def reference(j):
    """`CertifiedRows.piece` on `certified_rows`, over the descriptor as `codec.decode_program` decodes it."""
    if not isinstance(j, dict):
        raise D.Reject("not an object")
    try:
        prog = decode_program(j.get("program"), REGISTRY)
        ranges = evaluate(prog, template_instances_query(j.get("templates"))).ranges
    except Exception as e:  # noqa: BLE001 -- what the decoder or the query refuses, Lean's decoder or query refuses
        raise D.Reject(str(e)) from None
    certified, labels = j.get("certified"), j.get("labels")
    if not (isinstance(certified, dict) and all(isinstance(e, dict) and isinstance(e.get("root"), str)
                                                and type(e.get("leaves")) is int and e["leaves"] >= 0
                                                for e in certified.values())):
        raise D.Reject("certified is not an object of {root, leaves}")
    if not (isinstance(labels, dict) and all(isinstance(vs, list) and all(isinstance(v, str) for v in vs)
                                             for vs in labels.values())):
        raise D.Reject("labels is not an object of lists of strings")
    try:
        ms = C.marks(prog)
    except C.CertifiedRowsError as e:
        return {"refused": e.codes}
    ls = C.layout(prog, ms)
    population = sum(r[3] for r in ranges)
    return {"marks": [[m.template, m.port, m.label, m.output] for m in ms],
            "layout": [[lay.label, [[k, first, m.template, m.port] for k, first, m in lay.sites], lay.rows] for lay in ls],
            "units": [[list(x) for x in C.rows_at(ls, ranges, u)] for u in range(population)],
            "values": {lay.label: [None if (v := C.value_at(certified, labels, lay.label, g)) is None else list(v)
                                   for g in range(lay.rows + 1)] for lay in ls},
            "held": C.labels_held(ls, certified, labels)}


# --- the Programs ----------------------------------------------------------------------------------------------------

def test_two_tables_a_call_beside_a_batch_a_cut_and_an_output(pieces):
    """Program 1: `A`'s call is row 0 of ``in`` and its batch's member m row 1 + m, cut across the label's two values at
    row 3; `B`'s member m writes row m of ``out`` at its port `o1`."""
    j = _case(_program())
    want = reference(j)
    a, b = (descriptor_id(_a()), descriptor_id(_b()))
    assert want["marks"] == [[a, "row", "in", False], [b, "o1", "out", True]]
    assert want["layout"] == [["in", [[4, 0, a, "row"], [5, 1, a, "row"]], 6], ["out", [[6, 0, b, "o1"]], 6]]
    assert want["units"] == [[["row", "in", g]] for g in range(6)] + [[["o1", "out", m]] for m in range(6)]
    assert want["values"]["in"] == [["link/ingress/0", "aa", 0], ["link/ingress/0", "aa", 1], ["link/ingress/0", "aa", 2],
                                    ["link/ingress/1", "bb", 0], ["link/ingress/1", "bb", 1], ["link/ingress/1", "bb", 2], None]
    assert want["values"]["out"] == [["link/egress/0", "cc", m] for m in range(6)] + [None]
    assert want["held"] is True
    assert pieces.call("certified_rows", j) == ("output", D.canonical(want))


REFUSED = [
    ("a mark over half a parameter", lambda: _program(a=_a(Ports(params={"row": Leaves(((0, 1, In(certified="in")),))}))),
     ["certified-whole"]),
    ("one label on two ports of a template",
     lambda: _program(a=_a(Ports(params={"row": In(certified="in"), "x": In(certified="in")}))), ["certified-twice"]),
    ("a mark over two fields of the return",
     lambda: _program(b=_b(Ports(ret=Leaves(((1, 2, Out(certified="out")), (0, 1, Out(certified="out"))))))), None),
    ("a mark on a Definition called only from inside a template", _nested, ["certified-called"]),
]


@pytest.mark.parametrize("name, build, codes", REFUSED, ids=[r[0] for r in REFUSED])
def test_a_refused_shape(pieces, name, build, codes):
    """Program 2's shapes the reader refuses, each a one-line change of Program 1. A mark over both fields of `B`'s
    return is one run of the leaves, two whole fields: refused as not one field."""
    j = _case(build(), templates=None)
    want = reference(j)
    assert want == {"refused": codes or ["certified-whole"]}
    assert pieces.call("certified_rows", j) == ("output", D.canonical(want))


HELD = [
    ("the labels' values' leaves do not sum to the rows", {"certified": CERTIFIED | {"link/ingress/1": _entry("bb", 2)}}),
    ("a label with no value", {"labels": {"in": LABELS["in"]}}),
    ("a label the Program has not", {"labels": LABELS | {"other": []}}),
    ("a certified entry under no label", {"certified": CERTIFIED | {"link/egress/1": _entry("dd", 1)}}),
    ("a value under two labels", {"labels": LABELS | {"out": ["link/egress/0", "link/ingress/0"]},
                                  "certified": CERTIFIED | {"link/egress/0": _entry("cc", 3)}}),
    ("a value twice", {"labels": LABELS | {"out": ["link/egress/0", "link/egress/0"]},
                       "certified": CERTIFIED | {"link/egress/0": _entry("cc", 3)}}),
    ("a label's value no certified entry", {"labels": LABELS | {"out": ["link/egress/9"]}}),
]


@pytest.mark.parametrize("name, change", HELD, ids=[h[0] for h in HELD])
def test_labels_the_registration_refuses(pieces, name, change):
    """Program 2's labels: each a one-line change of Program 1's, `labelsHeld` false and the reader's rows unchanged."""
    j = _case(_program(), **change)
    want = reference(j)
    assert want["held"] is False and want["marks"] == reference(_case(_program()))["marks"]
    assert pieces.call("certified_rows", j) == ("output", D.canonical(want))


def test_a_program_without_marks_holds_no_labels_and_no_certified_entry(pieces):
    j = _case(_program(a=_a(Ports()), b=_b(Ports())), certified={}, labels={})
    want = reference(j)
    assert want == {"marks": [], "layout": [], "units": [[] for _ in range(12)], "values": {}, "held": True}
    assert pieces.call("certified_rows", j) == ("output", D.canonical(want))
    j = _case(_program(a=_a(Ports()), b=_b(Ports())))
    assert reference(j)["held"] is False and pieces.call("certified_rows", j) == ("output", D.canonical(reference(j)))


@pytest.mark.parametrize("where, ports, code", [
    ("root", {"ret": [[0, 1, {"certified": "out"}]]}, "certified-root"),
    ("root", {"inputs": [[0, 0, 2, {"certified": "in"}]]}, "certified-root"),
    ("CertRowsTestAdd16_v1", {"params": [[0, 0, 1, {"certified": "in"}]]}, "certified-primitive"),
    ("A", {"params": [[0, 0, 2, {"certified": "In"}]]}, "certified-malformed"),
    ("A", {"params": [[0, 0, 2, {"certified": 1}]]}, "certified-malformed"),
    ("A", {"params": [[0, 0, {"certified": "in"}]]}, "certified-malformed"),
    ("A", {"params": [[2, 0, 1, {"certified": "in"}]]}, "certified-whole"),
])
def test_what_the_decoder_refuses_the_lean_reader_refuses_itself(pieces, where, ports, code):
    """The Lean decoder reads no `ports`, so the reader refuses what `codec.decode_program` and `program.Program` refuse
    of a mark: on the root, on a primitive, a label that is no name, a malformed row, a parameter past the last."""
    j = _case(_program())
    desc = j["program"]
    key = {"root": desc["root"], "A": descriptor_id(_a())}.get(where, where)
    desc["definitions"][key]["ports"] = ports
    with pytest.raises(ValueError):
        decode_program(copy.deepcopy(desc), REGISTRY)
    assert pieces.call("certified_rows", j) == ("output", D.canonical({"refused": [code]}))


# --- generated Programs ----------------------------------------------------------------------------------------------

PORTS_A = [Ports(), Ports(params={"row": In(certified="in")}), Ports(params={"row": In(certified="out")}),
           Ports(params={"x": In(certified="in")}), Ports(params={"row": In(certified="in"), "x": In(certified="aux")}),
           Ports(params={"row": In(certified="in"), "x": In(certified="in")}),
           Ports(params={"row": Leaves(((0, 1, In(certified="in")), (1, 2, In(certified="in", word="u16"))))}),
           Ports(params={"row": Leaves(((1, 2, In(certified="in")),))})]
PORTS_B = [Ports(), Ports(ret=(None, Out(certified="out"))), Ports(ret=(Out(certified="out"), None)),
           Ports(ret=(Out(certified="in"), Out(certified="out"))), Ports(ret=(Out(certified="out"), Out(certified="out")))]


@st.composite
def cases(pick):
    """Program 1 at other batch sizes and marks, with certified entries and labels mostly fitting its rows: each label of
    the marks gets its rows cut into up to three values, at times one more row or one fewer, a label dropped or a stray
    entry added; and at times a field of the input replaced by arbitrary JSON."""
    prog = _program(_a(pick(st.sampled_from(PORTS_A))), _b(pick(st.sampled_from(PORTS_B))),
                    pick(st.integers(1, 3)), pick(st.integers(1, 3)))
    try:
        ls = C.layout(prog, C.marks(prog))
    except C.CertifiedRowsError:
        ls = []
    names = iter(f"v{i}" for i in range(99))
    certified, labels = {}, {}
    for lay in ls:
        if pick(st.integers(0, 9)) == 9:
            continue
        cuts = sorted(pick(st.lists(st.integers(0, lay.rows), max_size=2)))
        sizes = [b - a for a, b in zip([0, *cuts], [*cuts, lay.rows])]
        sizes[-1] += pick(st.sampled_from([0, 0, 0, 0, 0, 0, 1, -1])) if sizes[-1] else 0
        labels[lay.label] = []
        for n in sizes:
            v = f"{lay.label}/{next(names)}"
            certified[v] = _entry(pick(st.sampled_from(["aa", "bb"])), n)
            labels[lay.label].append(v)
    if pick(st.integers(0, 9)) == 9:
        certified[f"stray/{next(names)}"] = _entry("cc", 1)
    j = _case(prog, certified=certified, labels=labels)
    if pick(st.integers(0, 9)) == 9:
        j[pick(st.sampled_from(["certified", "labels", "templates"]))] = pick(st.sampled_from([None, 1, "x", [1], {"a": 1}]))
    return j


CERTIFIED_ROWS = D.Port(PIECES, "certified_rows", reference, cases(), package=PACKAGE, of_record="lean")

pieces = D.server_fixture(CERTIFIED_ROWS)


def test_generated_programs_agree(pieces):
    stats = D.run(CERTIFIED_ROWS, examples=200, server=pieces)
    assert stats.outputs > 120 and stats.rejects > 0, stats
