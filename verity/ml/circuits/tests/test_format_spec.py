"""The program format (`verity/ml/circuits/PROTOCOL.md`): every rule of the spec against its pinned vector (`format_vectors.json`).

The implementation is held to the file section by section; canonical JSON (as RFC 8785), leaf order, reference sequences and the
interning order are also restated here independently of it.  `python verity/ml/circuits/tests/test_format_spec.py --write`
rewrites the file from the reference implementation (a deliberate act: any change to a pinned rule shows in its diff)."""
from __future__ import annotations

import decimal
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from verity.ml.circuits import codec, cut, partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, Registry, bind
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.parts import n_members
from verity.ml.circuits.program import INPUT_ID, Program, input_id, is_input
from verity.ml.circuits.refs import Affine, Coll, Concat, array_of, refs_from_json, strided_view, tuple_of
from verity.ml.circuits.types import Array, Record, Tuple, Value, type_from_json

PATH = Path(__file__).parent / "format_vectors.json"
V16, V32 = Value(16), Value(32)
M32 = (1 << 32) - 1


def _prim(name, params, ret):
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *a: sum(a) & M32, register=False)


def _comp(name, statics, sig, body):
    return CompositeDefinition(name, 1, tuple(statics), sig, body, register=False)


TAdd = _prim("TAdd32", [("a", V32), ("b", V32)], V32)
TWiden = _prim("TWiden16", [("a", V16)], V32)
TSum2 = _prim("TSum2", [("x", Array(2, V32))], V32)
TSum4 = _prim("TSum4", [("x", Array(4, V32))], V32)
TSum6 = _prim("TSum6", [("x", Array(6, V32))], V32)
TWide = _prim("TWide", [("x", Array(cut.WIDE_FAN_IN, V32)), ("a", V32)], V32)
TNarrow = _prim("TNarrow", [("x", Array(4, V32)), ("a", V32)], V32)
REG = Registry()
for _p in (TAdd, TWiden, TSum2, TSum4, TSum6, TWide, TNarrow):
    REG.add(_p)

#: (s, s + a), s = a + b
Pair = bind(_comp("TPair", (), lambda S: ((("a", V32), ("b", V32)), Tuple(V32, V32)),
                  lambda B, S, a, b: (lambda s: tuple_of(s, B.call(TAdd, s, a)))(B.call(TAdd, a, b))))
#: c[0] + c[1]
Col = bind(_comp("TCol", (), lambda S: ((("c", Array(2, V32)),), V32), lambda B, S, c: B.call(TAdd, c[0], c[1])))
#: (c + x, c + x + k): a scan step (carry, xs, shared)
Step = bind(_comp("TStep", (), lambda S: ((("c", V32), ("x", V32), ("k", V32)), Tuple(V32, V32)),
                  lambda B, S, c, x, k: (lambda c2: tuple_of(c2, B.call(TAdd, c2, k)))(B.call(TAdd, c, x))))


def _row(B, S, row, bias):
    acc = B.call(S.OP, row[0], bias)
    for i in range(1, S.N):
        acc = B.call(S.OP, acc, row[i])
    return acc


#: a fold with statics of every kind: an int, a function, a float and a dict
RowDef = _comp("TRow", ("N", "OP", "C", "D"), lambda S: ((("row", Array(S.N, V32)), ("bias", V32)), V32), _row)
Row = bind(RowDef, N=4, OP=TAdd, C=1.5, D={"k": 2, "a": [1, "x"]})

X_T = Array(2, Array(4, V32))
R_T = Record(("h", Array(2, V16)), ("b", V32))
RET_T = Tuple(Tuple(V32, V32), Array(2, V32), Array(4, V32), Tuple(V32, Array(4, V32)), V32, V32, V32, V32, V32)


def _prog(B, S, x, r):
    pair = B.call(Pair, x[0][0], x[0][1])                                        # node 3: a call
    rows = B.batch(Row, x, r.b, axes=(0, None))                                  # node 4: a composite batched on axis 0, one argument broadcast
    wide = B.batch(TWiden, r.h, axes=(0,))                                       # node 5: a primitive batched on axis 0
    cols = B.batch(Col, x, axes=(1,))                                            # node 6: a composite batched on axis 1
    sums = B.batch(TSum2, x, axes=(1,))                                          # node 7: a primitive batched on axis 1
    scan = B.scan(Step, pair[0], xs=[x[1]], shared=[r.b])                        # node 8: a scan
    bcast = B.call(TSum4, array_of([wide[0]] * 4))                               # node 9: one value referenced four times
    view = B.call(TSum4, Coll(Array(4, V32), strided_view(x, 0, 2, 1, 4, 4)))    # node 10: x[0][0], x[1][0], x[0][1], x[1][1]
    head = array_of([cols[0], cols[1], rows[0], rows[1]])
    four = B.call(TSum4, head)                                                   # node 11
    six = B.call(TSum6, Coll(Array(6, V32), Concat([head.refs, Concat([pair[0].refs, wide[1].refs])])))  # node 12: node 11's argument + 2
    return tuple_of(pair, rows, sums, scan, bcast, view, four, six, r.b)


PROG = _comp("TProg", (), lambda S: ((("x", X_T), ("r", R_T)), RET_T), _prog)


def _call_of(prim):
    """a = s + s; w1 = prim(x, a); w2 = prim(x, a); w1 + w2"""
    L = prim.params[0][1].n

    def body(B, S, x, s):
        a = B.call(TAdd, s, s)
        return B.call(TAdd, B.call(prim, x, a), B.call(prim, x, a))

    return bind(_comp(f"T{prim.name}Call", (), lambda S: ((("x", Array(L, V32)), ("s", V32)), V32), body))


WideCall, NarrowCall = _call_of(TWide), _call_of(TNarrow)
WIDE_PROG = _comp("TWideProg", (), lambda S: ((("x", Array(cut.WIDE_FAN_IN, V32)), ("s", V32)), V32), lambda B, S, x, s: B.call(WideCall, x, s))


# -- the spec restated -------------------------------------------------------------------------------------------------------------

def jcs(v) -> bytes:
    """RFC 8785: object members in UTF-16 code-unit order of their names, ECMAScript string and number serialization, no whitespace."""
    def num(x: float) -> str:
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError("not a JSON number")
        if x == 0:
            return "0"
        if x < 0:
            return "-" + num(-x)
        sign, digits, exp = decimal.Decimal(repr(x)).as_tuple()          # repr: the shortest round-trip digits
        ds = list(digits)
        while len(ds) > 1 and ds[-1] == 0:
            ds.pop()
            exp += 1
        s, k = "".join(map(str, ds)), len(ds)
        n = exp + k
        if k <= n <= 21:
            return s + "0" * (n - k)
        if 0 < n <= 21:
            return s[:n] + "." + s[n:]
        if -6 < n <= 0:
            return "0." + "0" * -n + s
        e = n - 1
        return (s if k == 1 else s[0] + "." + s[1:]) + ("e+" if e >= 0 else "e-") + str(abs(e))

    def string(s: str) -> str:
        short = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
        return '"' + "".join(short.get(c) or (f"\\u{ord(c):04x}" if ord(c) < 0x20 or 0xD800 <= ord(c) <= 0xDFFF else c) for c in s) + '"'

    def ser(x) -> str:
        if x is None:
            return "null"
        if x is True:
            return "true"
        if x is False:
            return "false"
        if isinstance(x, (int, float)):
            return num(float(x))
        if isinstance(x, str):
            return string(x)
        if isinstance(x, list):
            return "[" + ",".join(ser(y) for y in x) + "]"
        return "{" + ",".join(string(k) + ":" + ser(x[k]) for k in sorted(x, key=lambda k: k.encode("utf-16-be"))) + "}"

    return ser(v).encode("utf-8")


def leaf_widths(t) -> list[int]:
    """Row-major leaves: an Array's elements in index order, a Tuple's or Record's fields in declared order."""
    tag = t[0]
    if tag == "Value":
        return [t[1]]
    if tag == "Array":
        return leaf_widths(t[2]) * t[1]
    return [w for f in t[1] for w in leaf_widths(f[1] if tag == "Record" else f)]


def expand(j) -> list[list]:
    """A reference sequence's references, in order, repeats kept."""
    if "affine" in j:
        s, i, b, c = j["affine"]
        return [[s, i, b + k] for k in range(c)]
    if "strided" in j:
        s, i, b, inner, os_, is_, c = j["strided"]
        return [[s, i, b + (k // inner) * os_ + (k % inner) * is_] for k in range(c)]
    if "concat" in j:
        return [r for p in j["concat"] for r in expand(p)]
    return [list(r) for r in j["explicit"]]


def first_seen_tables(desc: dict) -> tuple[list, list]:
    """The v1 tables from the definitions alone: a post-order walk from `root` (a composite's callees in node order, then the
    functions its binding record names, in record order), then those `program_fn`'s record names; each composite, when
    finished, adds per node its callee, its argument types and its output type."""
    D, C, T = desc["definitions"], desc["callees"], desc["types"]
    seen: set[str] = set()
    callees: list[str] = []
    types: list = []

    def named(fid: str) -> list[str]:
        return re.findall(r'\{"fn":"([^"]+)"\}', fid)

    def add(table: list, x) -> None:
        if x not in table:
            table.append(x)

    def visit(fid: str) -> None:
        if fid in seen:
            return
        seen.add(fid)
        d = D[fid]
        if d["form"] == "primitive":
            return
        for nd in d["body"]["nodes"]:
            visit(C[nd["fn"]])
        for f in named(fid):
            visit(f)
        for nd in d["body"]["nodes"]:
            add(callees, C[nd["fn"]])
            for a in nd["args"]:
                add(types, T[a["type"]])
            add(types, T[nd["out"]])

    visit(desc["root"])
    for f in named(desc["program_fn"]):
        if f in D:
            visit(f)
    return callees, types


# -- the vectors ----------------------------------------------------------------------------------------------------------------------

JCS_IN = [
    {"b": 1, "a": 2, "B": 3, "_": 4, "aa": 5, "0": 6, "": 7},
    "q\"b\\s/t\b\f\n\r\t\x00\x1f~",
    [0, -1, 9007199254740991, -9007199254740991],
    [True, False, None, [], {}, [[1, [2]], {"k": {"j": []}}]],
    {"affine": ["n", 3, 0, 1], "strided": ["p", 0, 7, 1, 0, 0, 4]},
]
JCS_OUT = ["\u00e9", "\x7f", 2**53 + 1, 10**21, 1.0, {"\ue000": 1, "\U0001f600": 2}]

TYPES = [["Value", 16], ["Array", 3, ["Value", 8]], ["Array", 0, ["Value", 32]], ["Tuple", [["Value", 16], ["Array", 2, ["Value", 32]]]],
         ["Record", [["b", ["Value", 32]], ["a", ["Array", 2, ["Value", 16]]]]], ["Array", 2, ["Tuple", [["Value", 16], ["Value", 32]]]]]
REFS = [{"affine": ["p", 0, 3, 4]}, {"strided": ["n", 2, 1, 3, 10, 2, 6]}, {"strided": ["n", 5, 7, 1, 0, 0, 4]},
        {"strided": ["n", 1, 0, 2, 1, 0, 4]}, {"explicit": [["p", 0, 2], ["n", 1, 0], ["p", 0, 2]]}, {"explicit": []},
        {"concat": [{"affine": ["n", 0, 6, 2]}, {"strided": ["n", 5, 7, 1, 0, 0, 2]}, {"explicit": [["p", 1, 0]]}]}]
TAKES = [[["Array", 2, ["Array", 4, ["Value", 32]]], {"affine": ["n", 0, 0, 8]}, 0, 1],
         [["Array", 2, ["Array", 4, ["Value", 32]]], {"affine": ["n", 0, 0, 8]}, 1, 2],
         [["Array", 2, ["Array", 3, ["Array", 2, ["Value", 32]]]], {"affine": ["p", 0, 0, 12]}, 1, 1],
         [["Array", 2, ["Array", 3, ["Array", 2, ["Value", 32]]]], {"affine": ["p", 0, 0, 12]}, 2, 1],
         [["Array", 2, ["Array", 2, ["Value", 32]]], {"strided": ["n", 4, 0, 2, 1, 2, 4]}, 1, 0]]
INPUT_IDS = ["Input16_v1", "Input1_v1", "Input128_v1", "Input016_v1", "Input0_v1", "Input16_v2", "Input16", "input16_v1",
             "Input16_v1 ", "Input16_v1\n", "XInput16_v1", "Input-16_v1", "Input1a_v1"]
#: run and registered input ids, and near misses
BINDING_IDS = ["Input16_v1", "RegisteredInput16_v1", "RegisteredInput1_v1", "RegisteredInput016_v1", "RegisteredInput0_v1",
               "RegisteredInput16_v2", "RegisteredInput16", "registeredInput16_v1", "Registered16_v1", "RegisteredInput16_v1 ",
               "RegisteredInput16_v1\n"]
LOWERINGS = [["Array", 2, ["Array", 3, ["Value", 16]]], ["Record", [["h", ["Array", 2, ["Value", 16]]], ["b", ["Value", 32]]]],
             ["Array", 2, ["Tuple", [["Value", 16], ["Value", 32]]]], ["Array", 0, ["Value", 16]]]
#: (label, root scope as (node, member) steps, reference)
RESOLVE = [
    ("p through a call", [[3, 0]], ["p", 1, 0]),
    ("p through a batch, axis 0", [[4, 1]], ["p", 0, 2]),
    ("p through a batch, broadcast (axis null)", [[4, 1]], ["p", 1, 0]),
    ("p through a batch, axis 1", [[6, 2]], ["p", 0, 1]),
    ("p through a scan: the carry of iteration 0 is init", [[8, 0]], ["p", 0, 0]),
    ("p through a scan: the carry of iteration m > 0 is iteration m-1's returned carry", [[8, 2]], ["p", 0, 0]),
    ("p through a scan: xs", [[8, 2]], ["p", 1, 0]),
    ("p through a scan: shared", [[8, 2]], ["p", 2, 0]),
    ("n: a primitive node inside a batch member", [[4, 0]], ["n", 2, 0]),
    ("n: an Input node (a batch of a primitive)", [], ["n", 0, 5]),
    ("n: a call's output", [], ["n", 3, 1]),
    ("n: a batch of a primitive", [], ["n", 5, 1]),
    ("n: a batch of a composite", [], ["n", 4, 1]),
    ("n: a scan's final carry", [], ["n", 8, 0]),
    ("n: a scan's ys", [], ["n", 8, 3]),
]


def _A(k: int, b: int = 0, c: int = 1) -> Affine:
    return Affine("n", k, b, c)


#: (label, the argument (node, arg) the sequence is encoded as, or None for a body's `ret`, the sequence)
ALIAS = [
    ("two parts: recorded as argument (0, 0)", (0, 0), Concat([_A(1), _A(2)])),
    ("the same two parts: the whole sequence is an alias; the first argument recorded keeps the key", (1, 0), Concat([_A(1), _A(2)])),
    ("those two parts and one more", (2, 0), Concat([_A(1), _A(2), _A(3)])),
    ("a concatenation of one part is written as the part", (3, 0), Concat([_A(4, 0, 3)])),
    ("a nested concatenation is written flat", (4, 0), Concat([Concat([_A(5), _A(6)]), _A(7)])),
    ("eight trailing parts (ALIAS_WINDOW) alias", (5, 0), Concat([_A(1), _A(2)] + [_A(10 + i) for i in range(8)])),
    ("nine trailing parts do not", (6, 0), Concat([_A(1), _A(2)] + [_A(30 + i) for i in range(9)])),
    ("the longest recorded prefix wins", (7, 0), Concat([_A(1), _A(2), _A(3), _A(9)])),
    ("a body's ret aliases too", None, Concat([_A(1), _A(2), _A(20)])),
    ("a sequence that is not a concatenation is never an alias", (8, 0), _A(1, 0, 2)),
    ("a node's first argument, new parts: recorded as (9, 0)", (9, 0), Concat([_A(40), _A(41)])),
    ("its second argument repeats them and is written out: an alias names an earlier node", (9, 1), Concat([_A(40), _A(41)])),
    ("a later node aliases the first argument", (10, 0), Concat([_A(40), _A(41)])),
]


def _root(desc: dict) -> dict:
    return desc["definitions"][desc["root"]]["body"]


def _node(desc: dict, k: int) -> dict:
    return _root(desc)["nodes"][k]


def _set_binding(desc: dict, key: str, value) -> None:
    d = desc["definitions"][next(f for f in desc["definitions"] if f.startswith("TRow_v1"))]
    d["bindings"][key] = value


#: (label, mutation of the vector descriptor, a substring of the refusal)
REJECT = [
    ("an unknown node form", lambda d: _node(d, 3).update(form="map"), "form 'map'"),
    ("a node key outside its form's", lambda d: _node(d, 3).update(name="x"), "with keys"),
    ("a primitive node calling a composite", lambda d: _node(d, 3).update(form="primitive"), "does not call a composite"),
    ("a call node calling a primitive", lambda d: _node(d, 9).update(form="call"), "does not call a primitive"),
    ("a scan node calling a primitive", lambda d: _node(d, 9).update(form="scan", n=1, n_xs=0), "does not call a primitive"),
    ("an alias forward", lambda d: _node(d, 12)["args"][0]["refs"]["concat"][0].update(alias=[12, 0]), "earlier node"),
    ("an alias backward from the end", lambda d: _node(d, 12)["args"][0]["refs"]["concat"][0].update(alias=[-1, 0]), "earlier node"),
    ("an alias to a missing argument", lambda d: _node(d, 12)["args"][0]["refs"]["concat"][0].update(alias=[11, 1]), "earlier node"),
    ("a callee index from the end", lambda d: _node(d, 3).update(fn=-1), "callees index -1"),
    ("a type index past the table", lambda d: _node(d, 3).update(out=len(d["types"])), "types index"),
    ("a reference space other than p / n", lambda d: _node(d, 3)["args"][0].update(refs={"affine": ["g", 0, 0, 1]}), "name no leaf"),
    ("a leaf past its collection", lambda d: _node(d, 3)["args"][0].update(refs={"affine": ["n", 0, 8, 1]}), "name no leaf"),
    ("a node that does not exist", lambda d: _node(d, 3)["args"][0].update(refs={"affine": ["n", 13, 0, 1]}), "name no leaf"),
    ("a leaf past a later node's declared output", lambda d: _node(d, 3)["args"][0].update(refs={"affine": ["n", 9, 1, 1]}), "name no leaf"),
    ("a strided pattern past its collection", lambda d: _node(d, 10)["args"][0]["refs"].update(strided=["n", 0, 0, 2, 1, 7, 4]), "name no leaf"),
    ("a boolean leaf", lambda d: _node(d, 3)["args"][0].update(refs={"explicit": [["n", 0, True]]}), "name no leaf"),
    ("an argument type unlike the callee's parameter",
     lambda d: _node(d, 3)["args"][0].update(type=d["types"].index(["Array", 2, ["Value", 16]]), refs={"affine": ["n", 1, 0, 2]}), "expected Value<32>"),
    ("a declared out unlike the computed one", lambda d: _node(d, 9).update(out=d["types"].index(["Array", 4, ["Value", 32]])), "declares out"),
    ("a batch count unlike the mapped length", lambda d: _node(d, 4).update(n=3), "disagrees with mapped axis length"),
    ("a boolean axis", lambda d: _node(d, 6).update(axes=[True]), "batch counts"),
    ("a scan's n_xs past its arguments", lambda d: _node(d, 8).update(n_xs=3), "scan counts"),
    ("a boolean width", lambda d: d["types"].__setitem__(d["types"].index(["Array", 2, ["Value", 16]]), ["Array", 2, ["Value", True]]),
     "positive int width"),
    ("a string outside ASCII", lambda d: d.update(program_fn="TProg\u00e9_v1"), "not ASCII without DEL"),
    ("DEL", lambda d: _set_binding(d, "D", {"dict": {"a": [1, "x\x7f"], "k": 2}}), "not ASCII without DEL"),
    ("an integer past 2^53 - 1", lambda d: _set_binding(d, "N", 2**53), "exceeds 2^53 - 1"),
    ("a float", lambda d: _set_binding(d, "C", 1.5), "float"),
    ("an Input id with a leading zero", lambda d: _rename(d, "Input16_v1", "Input016_v1"), "primitive Input016_v1 is not in the registry"),
    ("an Input id with a trailing newline", lambda d: _rename(d, "Input16_v1", "Input16_v1\n"), r"primitive Input16_v1\n is not in the registry"),
    ("a primitive keyed by another id", lambda d: d["definitions"]["TSum4_v1"].update(id="TSum6_v1"), "keyed by its id"),
    ("a binding unlike the key's record", lambda d: _set_binding(d, "N", 5), "keyed by its id and binding record"),
    ("a composite id that is not Name_vN", lambda d: d["definitions"]["TPair_v1"].update(id="TPair_v01"), "Name_vN"),
    ("a root with a parameter", lambda d: _root(d)["params"].append(["x", ["Value", 32]]), "without parameters"),
    ("an Input node inside a Definition", lambda d: _inner_input(d, "Input32_v1"), "appears only in the root's body"),
    ("a RegisteredInput node inside a Definition", lambda d: _inner_input(d, "RegisteredInput32_v1"), "appears only in the root's body"),
]


def _rename(desc: dict, old: str, new: str) -> None:
    desc["definitions"][new] = dict(desc["definitions"].pop(old), id=new)
    desc["callees"][desc["callees"].index(old)] = new


def _inner_input(desc: dict, fid: str) -> None:
    """TPair's second node, `s + a`, becomes an input node: TPair returns (a + b, an input no parameter binds)."""
    if fid not in desc["callees"]:
        desc["callees"].append(fid)
        desc["definitions"][fid] = {"form": "primitive", "id": fid, "params": [], "ret": ["Value", 32]}
    desc["definitions"]["TPair_v1"]["body"]["nodes"][1] = {"args": [], "fn": desc["callees"].index(fid), "form": "primitive",
                                                           "out": desc["types"].index(["Value", 32])}


def _graph(g: cut.CallGraph) -> dict:
    return {"total": g.total, "n": g.n, "keep": g.keep.tolist(), "prim": [g.prim_names[x] for x in g.prim.tolist()],
            "external": list(g.external), "input_ids": list(g.input_ids), "input_reads": [list(x) for x in g.input_reads],
            "src": g.src.tolist(), "dst": g.dst.tolist(), "outputs": list(g.outputs), "recomputed": [list(x) for x in g.recomputed]}


def _runs(xs) -> list[list[int]]:
    out: list[list[int]] = []
    for x in xs:
        if out and out[-1][1] == x:
            out[-1][1] += 1
        else:
            out.append([x, x + 1])
    return out


def _canon(desc: dict) -> str:
    return codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"}).decode()


def _layout(D) -> dict:
    """Bytes in, layout out: the root's nodes, every composite's offsets, the gate table, the root's outputs, the resolution cases."""
    c = D.circuit
    rs = c.root_scope
    nodes = [[k, n.form, n.fn.id, n_members(n), n.gates, rs.spec.body.offsets[k]] for k, n in enumerate(D.root.body.nodes)]
    offsets = {fid: list(f.body.offsets) for fid, f in sorted(D.defs.items()) if not isinstance(f, PrimitiveDefinition)}
    gates = []
    for i in range(c.size):
        g = c.gate(i)
        gates.append([i, g.prim.id, list(g.operands), g.path])
    res = []
    for label, path, ref in RESOLVE:
        s = rs
        for k, m in path:
            s = s.child(k, m)
        res.append([label, path, ref, resolve(s, tuple(ref))])
    return {"gates_total": c.size, "nodes": nodes, "offsets": offsets, "gates": gates, "root_outputs": list(c.root_outputs()), "resolve": res}


def build() -> dict:
    enc = codec._BodyEncoder(codec.SCHEMA_V1, codec._Tables())
    program = Program(PROG)
    desc = codec.encode_program(program)
    D = codec.decode_program(json.loads(_canon(desc)), REG)
    wide = Program(WIDE_PROG)
    wdesc = codec.encode_program(wide)
    W = codec.decode_program(json.loads(_canon(wdesc)), REG)
    wg = W.circuit
    k = next(k for k, n in enumerate(W.root.body.nodes) if not is_input(n.fn))
    lo = W.root.body.offsets[k]
    registered = Program(PROG, registered=("r",))
    rdesc = codec.encode_program(registered)
    R = codec.decode_program(json.loads(_canon(rdesc)), REG)
    rejects = []
    for label, edit, _ in REJECT:
        d = json.loads(_canon(desc))
        edit(d)
        try:
            codec.decode_program(d, REG)
            rejects.append([label, None])
        except (ValueError, KeyError) as e:
            rejects.append([label, f"{type(e).__name__}: {e}"])
    return {
        "about": "Pinned vectors of verity/ml/circuits/PROTOCOL.md (the program format), checked by verity/ml/circuits/tests/test_format_spec.py; written by it "
                 "with --write.",
        "canonical_json": {
            "in_domain": [[v, codec.canonical_json(v).decode()] for v in JCS_IN],
            "outside": [[v, codec.canonical_json(v).hex(), jcs(v).hex(), codec.jcs_domain_error(v)] for v in JCS_OUT],
            "max_safe_int": codec.MAX_SAFE_INT,
        },
        "types": [[t, type_from_json(t).leaves, type_from_json(t).leaf_widths()] for t in TYPES],
        "refs": [[j, [list(r) for r in refs_from_json(j)]] for j in REFS],
        "take": [[t, j, ax, m, [list(r) for r in Coll(type_from_json(t), refs_from_json(j)).take(ax, m).refs]] for t, j, ax, m in TAKES],
        "descriptor": _canon(desc),
        "program_digest": codec.program_digest(desc),
        "alias_writer": [[label, list(at) if at else None, enc.refs(r, at)] for label, at, r in ALIAS],
        "alias_reader": {"node": 12, "arg": 0, "refs": [list(r) for r in _root_nodes(D)[12].args[0].refs]},
        "decoder_rejects": rejects,
        "input_ids": [[i, bool(INPUT_ID.fullmatch(i))] for i in INPUT_IDS],
        "input_lowering": [[t, [[n.fn.id, n.form, list(n.axes), n.n] for n in Program(_identity(t)).root.body.nodes]] for t in LOWERINGS],
        "program": {"root": D.root.id, "input_nodes": [[k, n.fn.id, n.n] for k, n in enumerate(D.root.body.nodes) if is_input(n.fn)],
                    "input_gates": program.input_gates, "calls": [[k, fn.id] for k, fn in PO.calls(D)]},
        "binding_ids": [[i, list(input_id(i)) if input_id(i) else None] for i in BINDING_IDS],
        "registered_program": {"registered": ["r"], "descriptor": _canon(rdesc), "program_digest": codec.program_digest(rdesc),
                               "input_nodes": [[k, n.fn.id, n.n] for k, n in enumerate(R.root.body.nodes) if is_input(n.fn)],
                               "calls": [[k, fn.id] for k, fn in PO.calls(R)]},
        "layout": _layout(D),
        "forward_reference": _forward(desc),
        "wiring": sorted(cut.WIRING),
        "wide": {
            "wide_fan_in": cut.WIDE_FAN_IN,
            "descriptor": _canon(wdesc),
            "gates": [[lo + j, wg.gate(lo + j).prim.id, len(wg.gate(lo + j).operands), _runs(wg.gate(lo + j).operands)] for j in range(4)],
            "graph": {"wide": _graph(cut.CallGraph(WideCall)), "narrow": _graph(cut.CallGraph(NarrowCall))},
        },
    }


def _root_nodes(D):
    return D.root.body.nodes


def _forward(desc: dict) -> dict:
    """A body whose node reads a later node decodes (checkers name the defect); every partition query refuses it (`order_violation`),
    and `CallGraph` refuses a Call whose gate reads a later gate of the Call (a backstop)."""
    def refusal(D):
        try:
            PO.evaluate(D)
            return None
        except cut.QueryInapplicable as e:
            return e.detail

    d = json.loads(_canon(desc))
    _node(d, 3)["args"][1]["refs"] = {"affine": ["n", 9, 0, 1]}               # root node 3 reads node 9: two Calls, out of order
    R = codec.decode_program(d, REG)
    d = json.loads(_canon(desc))
    d["definitions"]["TPair_v1"]["body"]["nodes"][0]["args"][0]["refs"] = {"affine": ["n", 1, 0, 1]}   # inside TPair: s reads s + a
    D = codec.decode_program(d, REG)
    try:
        cut.CallGraph(D.defs["TPair_v1"])
        backstop = None
    except ValueError as e:
        backstop = str(e)
    return {"root_gate_11_operands": list(R.circuit.gate(11).operands), "root_refused": refusal(R),
            "call_gate_0_operands": list(D.circuit.gate(11).operands), "call_refused": refusal(D), "q_word": backstop}


def _identity(t_json):
    t = type_from_json(t_json)
    return _comp("TIdentity", (), lambda S: ((("p", t),), t), lambda B, S, p: p)


VECTORS = json.loads(PATH.read_text()) if PATH.exists() else {}


@pytest.fixture(scope="module")
def live() -> dict:
    return build()


@pytest.fixture(scope="module")
def decoded():
    return codec.decode_program(json.loads(VECTORS["descriptor"]), REG)


# -- 1. canonical JSON ----------------------------------------------------------------------------------------------------------------

def test_canonical_json_is_jcs_on_the_domain(live):
    assert live["canonical_json"] == VECTORS["canonical_json"]
    for v, s in VECTORS["canonical_json"]["in_domain"]:
        assert codec.jcs_domain_error(v) is None and codec.canonical_json(v) == jcs(v) == s.encode()
    for d in (VECTORS["descriptor"], VECTORS["wide"]["descriptor"]):
        assert jcs(json.loads(d)) == d.encode() and codec.jcs_domain_error(json.loads(d)) is None


def test_outside_the_domain_the_bytes_differ_and_the_codec_refuses():
    for v, py, j, err in VECTORS["canonical_json"]["outside"]:
        assert codec.canonical_json(v).hex() == py and jcs(v).hex() == j and py != j and err == codec.jcs_domain_error(v) is not None
    for D in ({"k": "\u00e9"}, {"k": "\x7f"}, {"k": 2**53}, {"k": -(2**53)}):
        fn = bind(RowDef, N=4, OP=TAdd, C=1.5, D=D)
        top = _comp("TOut", (), lambda S: ((("x", Array(4, V32)), ("b", V32)), V32), lambda B, S, x, b, fn=fn: B.call(fn, x, b))
        with pytest.raises(ValueError, match="canonical JSON domain"):
            codec.encode_program(Program(top))


def test_the_program_digests(decoded):
    b = VECTORS["descriptor"].encode()
    assert hashlib.sha512(b).hexdigest() == VECTORS["program_digest"] == codec.program_digest(decoded.desc)
    assert codec.program_digest(Program(PROG)) == VECTORS["program_digest"]


# -- 2. types, 3. reference sequences ----------------------------------------------------------------------------------------------

def test_types_and_leaf_order(live):
    assert live["types"] == VECTORS["types"]
    for t, n, ws in VECTORS["types"]:
        assert leaf_widths(t) == ws and len(ws) == n and type_from_json(t).to_json() == t


def test_reference_sequences(live):
    assert live["refs"] == VECTORS["refs"]
    for j, out in VECTORS["refs"]:
        assert expand(j) == out and refs_from_json(j).to_json() == j


def test_member_references(live):
    assert live["take"] == VECTORS["take"]
    for t, j, ax, m, out in VECTORS["take"]:
        shape, tt = [], t
        while tt[0] == "Array":
            shape.append(tt[1])
            tt = tt[2]
        leaves = expand(j)
        idx = [0] * len(shape)
        want = []
        for leaf in leaves:
            if idx[ax] == m:
                want.append(leaf)
            for d in reversed(range(len(shape))):
                idx[d] += 1
                if idx[d] < shape[d]:
                    break
                idx[d] = 0
        assert want == out


# -- 4. the descriptor ----------------------------------------------------------------------------------------------------------

def test_the_descriptor_round_trips(decoded):
    assert _canon(codec.encode_program(Program(PROG))) == VECTORS["descriptor"]
    assert _canon(codec.encode_program(decoded)) == VECTORS["descriptor"]
    desc = json.loads(VECTORS["descriptor"])
    assert sorted(desc) == ["callees", "definitions", "program_fn", "root", "schema", "types"]
    assert desc["schema"] == "verity-ir/descriptor/v1" and desc["root"] == "Root[TProg_v1]_v1" and desc["program_fn"] == "TProg_v1"


def test_the_interning_tables_are_first_seen_order():
    desc = json.loads(VECTORS["descriptor"])
    assert first_seen_tables(desc) == (desc["callees"], desc["types"])
    w = json.loads(VECTORS["wide"]["descriptor"])
    assert first_seen_tables(w) == (w["callees"], w["types"])


def test_the_alias_and_flat_concatenation(live, decoded):
    assert live["alias_writer"] == VECTORS["alias_writer"]
    assert live["alias_reader"] == VECTORS["alias_reader"]
    nd = _node(json.loads(VECTORS["descriptor"]), 12)
    assert nd["args"][0]["refs"] == {"concat": [{"alias": [11, 0]}, {"affine": ["n", 3, 0, 1]}, {"affine": ["n", 5, 1, 1]}]}
    earlier = _node(json.loads(VECTORS["descriptor"]), 11)["args"][0]["refs"]
    assert VECTORS["alias_reader"]["refs"] == expand(earlier) + [["n", 3, 0], ["n", 5, 1]]


@pytest.mark.parametrize("label,edit,why", REJECT, ids=[r[0] for r in REJECT])
def test_the_decoder_refuses(live, label, edit, why):
    got = dict(live["decoder_rejects"])[label]
    assert got == dict(VECTORS["decoder_rejects"])[label] and got is not None and why in got


# -- 5. programs and Input nodes ------------------------------------------------------------------------------------------------

def test_input_nodes_are_recognised_by_id(live, decoded):
    assert live["input_ids"] == VECTORS["input_ids"] and live["program"] == VECTORS["program"]
    assert [i for i, ok in VECTORS["input_ids"] if ok] == ["Input16_v1", "Input1_v1", "Input128_v1"]
    assert [k for k, n in enumerate(decoded.root.body.nodes) if is_input(n.fn)] == [k for k, _, _ in VECTORS["program"]["input_nodes"]]


def test_registered_inputs_are_recognised_by_id_and_bound_into_the_digest(live, decoded):
    assert live["binding_ids"] == VECTORS["binding_ids"] and live["registered_program"] == VECTORS["registered_program"]
    assert [i for i, b in VECTORS["binding_ids"] if b] == ["Input16_v1", "RegisteredInput16_v1", "RegisteredInput1_v1"]
    reg = VECTORS["registered_program"]
    R = codec.decode_program(json.loads(reg["descriptor"]), REG)
    assert reg["calls"] == VECTORS["program"]["calls"] and R.circuit.size == decoded.circuit.size
    assert [[k, n] for k, _, n in reg["input_nodes"]] == [[k, n] for k, _, n in VECTORS["program"]["input_nodes"]]
    assert [fid for _, fid, _ in reg["input_nodes"]] == ["Input32_v1", "RegisteredInput16_v1", "RegisteredInput32_v1"]
    assert reg["program_digest"] != VECTORS["program_digest"] and codec.program_digest(json.loads(reg["descriptor"])) == reg["program_digest"]
    assert [R.circuit.gate(g).prim.id for g in range(R.circuit.size) if not is_input(R.circuit.gate(g).prim)] == [
        decoded.circuit.gate(g).prim.id for g in range(decoded.circuit.size) if not is_input(decoded.circuit.gate(g).prim)]


def test_input_lowering(live):
    assert live["input_lowering"] == VECTORS["input_lowering"]


# -- 6. layout, 7. resolution ---------------------------------------------------------------------------------------------------

def test_layout_member_counts_and_offsets(live, decoded):
    got, want = live["layout"], VECTORS["layout"]
    assert got["nodes"] == want["nodes"] and got["offsets"] == want["offsets"] and got["gates_total"] == want["gates_total"]
    for k, form, _, members, gates, off in want["nodes"]:
        n = decoded.root.body.nodes[k]
        assert members == (n.n if form in ("batch", "scan") else 1)
        assert gates == (1 if form == "primitive" else n.fn.gates if form == "call" else members * n.fn.gates)
    for fid, offs in want["offsets"].items():
        body = decoded.defs[fid].body
        assert offs == [sum(n.gates for n in body.nodes[:k]) for k in range(len(body.nodes) + 1)]


def test_the_gate_table(live):
    assert live["layout"]["gates"] == VECTORS["layout"]["gates"]
    assert live["layout"]["root_outputs"] == VECTORS["layout"]["root_outputs"]


def test_scope_resolution(live):
    assert live["layout"]["resolve"] == VECTORS["layout"]["resolve"]


def test_a_forward_reference_decodes_and_q_word_refuses_it(live):
    assert live["forward_reference"] == VECTORS["forward_reference"]
    f = VECTORS["forward_reference"]
    assert f["root_gate_11_operands"] == [0, 39] and f["call_gate_0_operands"] == [12, 1] and "not topological" in f["q_word"]
    assert f["root_refused"]["node"] == 3 and f["call_refused"]["definition"] == "TPair_v1" and f["call_refused"]["why"] == "not topological"


# -- 8. Q_word v1 inputs: wiring, wide gates ------------------------------------------------------------------------------------

def test_the_wiring_list(live):
    assert live["wiring"] == VECTORS["wiring"] == ["Bf16ToF32_v1", "F32BitsShl23_v1", "F32Fabs_v1"]


def test_wide_gates(live):
    assert live["wide"] == VECTORS["wide"]
    w, n = VECTORS["wide"]["graph"]["wide"], VECTORS["wide"]["graph"]["narrow"]
    assert w["recomputed"] == [] and n["recomputed"] == [[2, 1]]
    assert w["external"] == [1, 1, 1, 0] and len(w["input_ids"]) == 1 and len(n["input_ids"]) == 5


if __name__ == "__main__" and "--write" in sys.argv:
    PATH.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {PATH}")
