"""Descriptors the Lean decoder (`Circuits/Descriptor.lean`) refuses and the Python decoder's cases do not reach: each the smallest
edit of one valid descriptor that trips one refusal, for `decode_difftest.cases()` (the Python verdict beside Lean's) and for the
pinned text of Lean's refusal (`tests/tools/difftest/lean_decode_texts.json`, written by `write_lean_texts.py`)."""
from __future__ import annotations

import functools
import json

import recut_vectors
import test_codec_refusals as TC
import test_format_spec as spec
from verity.circuits import codec
from verity.circuits import restrict as R
from verity.circuits.program import Program

V32 = ["Value", 32]


@functools.cache
def _big() -> str:
    return json.dumps(json.loads(spec._canon(codec.encode_program(Program(spec.PROG)))))


def big() -> dict:
    return json.loads(_big())


def tiny() -> dict:
    return TC.tiny()[0]


def _node(d: dict, k: int) -> dict:
    return spec._node(d, k)


def _root_def(d: dict) -> dict:
    return d["definitions"][d["root"]]


def _ref(d: dict, refs) -> None:
    """Node 2 of the tiny root (its one computed node) reads its first argument as ``refs``."""
    _node(d, 2)["args"][0]["refs"] = refs


def _bref(d: dict, refs) -> None:
    """Node 3 of the big root (a call of `TPair`) reads its first argument as ``refs``."""
    _node(d, 3)["args"][0]["refs"] = refs


def _drop(d: dict, *path) -> None:
    for p in path[:-1]:
        d = d[p]
    del d[path[-1]]


def _set(d: dict, path, value) -> None:
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = value


def _type_at(d: dict, t) -> int:
    if t not in d["types"]:
        d["types"].append(t)
    return d["types"].index(t)


def _rekey_root(d: dict, new: str, **bindings) -> None:
    root = d["root"]
    d["definitions"][new] = d["definitions"].pop(root)
    d["definitions"][new]["bindings"] = bindings
    d["root"] = new


def _grid(d: dict, grid) -> None:
    d["definitions"][TC.STAND_IN]["body"]["nodes"][0]["args"][0]["refs"] = {"grid": grid}


def _std_node(d: dict) -> dict:
    return d["definitions"][TC.STAND_IN]["body"]["nodes"][0]


def _scan_args(d: dict) -> list:
    return _node(d, 8)["args"]


def _add_def(d: dict, fid: str, params: list, ret: dict, ret_type: list) -> int:
    """A composite of no node returning ``ret``, appended to the descriptor's definitions and callees."""
    d["definitions"][fid] = {"form": "composite", "id": fid, "bindings": {},
                             "body": {"params": params, "nodes": [], "ret": ret, "ret_type": ret_type}}
    d["callees"].append(fid)
    return len(d["callees"]) - 1


def _scan_of(d: dict, fid: str, params: list, ret: dict, args: list, n_xs: int) -> None:
    node = _node(d, 8)
    node.update(fn=_add_def(d, fid, params, ret, ["Tuple", [V32, V32]]), n_xs=n_xs)
    node["args"] = [dict(a, refs=a["refs"], type=a["type"]) for a in args]


def _bits(t):
    if t[0] == "Value":
        return ["Value", 1] if t[1] == 1 else ["Array", t[1], ["Value", 1]]
    if t[0] == "Array":
        return ["Array", t[1], _bits(t[2])]
    return ["Tuple", [_bits(f) for f in t[1]]]


def _leaves(t) -> int:
    return 1 if t[0] == "Value" else t[1] * _leaves(t[2]) if t[0] == "Array" else sum(_leaves(f) for f in t[1])


def _pairs(prim: str, param: list, refs: dict) -> dict:
    """Four pairs `(V8, V16)` (a batch of `Pair` over two registered inputs), passed to `C`, whose one node is a gate-free
    primitive `prim` of the parameter type `param` reading them as ``refs``; `C` returns the node's `V8`."""
    v8, v16, pair = ["Value", 8], ["Value", 16], ["Tuple", [["Value", 8], ["Value", 16]]]
    pairs = ["Array", 4, pair]
    defs = {
        "RegisteredInput8_v1": {"form": "primitive", "id": "RegisteredInput8_v1", "params": [], "ret": v8},
        "RegisteredInput16_v1": {"form": "primitive", "id": "RegisteredInput16_v1", "params": [], "ret": v16},
        prim: {"form": "primitive", "id": prim, "params": [["a0", param]], "ret": v8},
        f"{prim[:-3]}Bits_v1": {"form": "composite", "id": f"{prim[:-3]}Bits_v1", "bindings": {},
                                "body": {"params": [["x0", _bits(param)]], "nodes": [], "ret": {"affine": ["p", 0, 0, 8]},
                                         "ret_type": _bits(v8)}},
        "Pair_v1": {"form": "composite", "id": "Pair_v1", "bindings": {},
                    "body": {"params": [["x0", v8], ["x1", v16]], "nodes": [],
                             "ret": {"concat": [{"affine": ["p", 0, 0, 1]}, {"affine": ["p", 1, 0, 1]}]}, "ret_type": pair}},
        "C_v1": {"form": "composite", "id": "C_v1", "bindings": {},
                 "body": {"params": [["x0", pairs], ["x1", v16]], "ret": {"affine": ["n", 0, 0, 1]}, "ret_type": v8,
                          "nodes": [{"args": [{"refs": refs, "type": 3}], "fn": 3, "form": "primitive", "out": 0}]}},
        "R_v1": {"form": "composite", "id": "R_v1", "bindings": {},
                 "body": {"params": [], "ret": {"affine": ["n", 3, 0, 1]}, "ret_type": v8,
                          "nodes": [{"args": [], "fn": 0, "form": "primitive", "out": 0},
                                    {"args": [], "fn": 1, "form": "primitive", "out": 1},
                                    {"args": [{"refs": {"affine": ["n", 0, 0, 1]}, "type": 0},
                                              {"refs": {"affine": ["n", 1, 0, 1]}, "type": 1}],
                                     "axes": [None, None], "fn": 2, "form": "batch", "n": 4, "out": 2},
                                    {"args": [{"refs": {"affine": ["n", 2, 0, 8]}, "type": 2},
                                              {"refs": {"affine": ["n", 1, 0, 1]}, "type": 1}], "fn": 4, "form": "call", "out": 0}]}},
    }
    return {"annotations": {}, "boolean": {prim: f"{prim[:-3]}Bits_v1"},
            "callees": ["RegisteredInput8_v1", "RegisteredInput16_v1", "Pair_v1", prim, "C_v1"], "definitions": defs,
            "program_fn": "R_v1", "root": "R_v1", "schema": "verity-ir/descriptor/v2", "types": [v8, v16, pairs, param]}


def _interleaved(periods: int) -> dict:
    """`periods` 8-bit and `periods` 16-bit registered inputs, read by one grid of two lanes (period t: leaf t of each) as an
    array of `periods` pairs `(V8, V16)` and passed to `C`: every reference is well typed, and the grid is two references
    a period, so it is `2 * periods` references that a count states and no list writes."""
    v8, v16, pair = ["Value", 8], ["Value", 16], ["Tuple", [["Value", 8], ["Value", 16]]]
    pairs = ["Array", periods, pair]
    grid = {"grid": [1, [["n", 0, 0, 1, [[periods, 1]]], ["n", 1, 0, 1, [[periods, 1]]]]]}
    return {
        "annotations": {}, "boolean": {},
        "callees": ["RegisteredInput8_v1", "RegisteredInput16_v1", "C_v1"],
        "definitions": {
            "RegisteredInput8_v1": {"form": "primitive", "id": "RegisteredInput8_v1", "params": [], "ret": v8},
            "RegisteredInput16_v1": {"form": "primitive", "id": "RegisteredInput16_v1", "params": [], "ret": v16},
            "C_v1": {"form": "composite", "id": "C_v1", "bindings": {},
                     "body": {"params": [["x0", pairs]], "nodes": [], "ret": {"affine": ["p", 0, 0, 1]}, "ret_type": v8}},
            "R_v1": {"form": "composite", "id": "R_v1", "bindings": {},
                     "body": {"params": [], "ret": {"affine": ["n", 2, 0, 1]}, "ret_type": v8,
                              "nodes": [{"args": [], "axes": [], "fn": 0, "form": "batch", "n": periods, "out": 3},
                                        {"args": [], "axes": [], "fn": 1, "form": "batch", "n": periods, "out": 4},
                                        {"args": [{"refs": grid, "type": 2}], "fn": 2, "form": "call", "out": 0}]}},
        },
        "program_fn": "R_v1", "root": "R_v1", "schema": "verity-ir/descriptor/v2",
        "types": [v8, v16, pairs, ["Array", periods, v8], ["Array", periods, v16]],
    }


def _static(d: dict, key: str, old: str, new: str, value) -> None:
    """The big descriptor's `TRow` rebound with static ``key`` as ``value``, its id and callee entry spelling the new record."""
    fid = next(f for f in d["definitions"] if f.startswith("TRow_v1{"))
    renamed = fid.replace(old, new)
    d["definitions"][renamed] = d["definitions"].pop(fid)
    d["definitions"][renamed]["bindings"][key] = value
    d["callees"][d["callees"].index(fid)] = renamed


#: (name, base descriptor, edit); an edit returning a str is the descriptor's text itself
CASES: list[tuple[str, object, object]] = [
    ("the text is not JSON", tiny, lambda d: "{"),
    ("the text is no object", tiny, lambda d: "[]"),
    ("no schema", tiny, lambda d: _drop(d, "schema")),
    ("no definitions", tiny, lambda d: _drop(d, "definitions")),
    ("a definition that is no object", tiny, lambda d: _set(d, ("definitions", "TAdd32_v1"), 1)),
    ("no callees", tiny, lambda d: _drop(d, "callees")),
    ("no types", tiny, lambda d: _drop(d, "types")),
    ("no root", tiny, lambda d: _drop(d, "root")),
    ("no program_fn", tiny, lambda d: _drop(d, "program_fn")),
    ("a record field that is no pair", tiny, lambda d: _set(d, ("types", 1), ["Record", [1]])),
    ("a record field of one item", tiny, lambda d: _set(d, ("types", 1), ["Record", [["a"]]])),
    ("a record field of three items", tiny, lambda d: _set(d, ("types", 1), ["Record", [["a", V32, 1]]])),
    ("a record field named by a number", tiny, lambda d: _set(d, ("types", 1), ["Record", [[1, V32]]])),
    ("a record with a field name twice", tiny, lambda d: _set(d, ("types", 1), ["Record", [["a", V32], ["a", V32]]])),
    ("a primitive's params that are no list", tiny, lambda d: _set(d, ("definitions", "TAdd32_v1", "params"), 1)),
    ("a primitive's parameter that is no pair", tiny, lambda d: _set(d, ("definitions", "TAdd32_v1", "params"), [1])),
    ("a primitive returning no one value", tiny, lambda d: _set(d, ("definitions", "TAdd32_v1", "ret"), ["Array", 2, V32])),
    ("an input primitive with a parameter", tiny, lambda d: _set(d, ("definitions", "Input32_v1", "params"), [["a", V32]])),
    ("a gate primitive of another signature", tiny, lambda d: _set(d, ("definitions", "Xor2_v1", "ret"), ["Value", 2])),
    ("a callee naming no definition", tiny, lambda d: _set(d, ("callees", 1), "TAbsent_v1")),
    ("a type index that is no integer", tiny, lambda d: _node(d, 2).update(out="x")),
    ("a callee that is no string", tiny, lambda d: _set(d, ("callees", 1), 5)),
    ("bindings that are no object", tiny, lambda d: _set(d, ("definitions", d["root"], "bindings"), 1)),
    ("a binding record of other keys than its id's", tiny, lambda d: _rekey_root(d, "Root[TTiny_v1]_v1{a=1,b=2}", a=1, c=2)),
    ("a body with no nodes", tiny, lambda d: _drop(d, "definitions", d["root"], "body", "nodes")),
    ("a node whose args are no list", tiny, lambda d: _set(d, ("definitions", d["root"], "body", "nodes", 2, "args"), 1)),
    ("a reference that is no object", tiny, lambda d: _ref(d, 5)),
    ("a reference of no known kind", tiny, lambda d: _ref(d, {"foo": 1})),
    ("a concat that is no list", tiny, lambda d: _ref(d, {"concat": 1})),
    ("an affine of three items", tiny, lambda d: _ref(d, {"affine": ["n", 0, 0]})),
    ("an affine of a count that is no natural", tiny, lambda d: _ref(d, {"affine": ["n", 0, 0, -1]})),
    ("a strided of six items", tiny, lambda d: _ref(d, {"strided": ["n", 0, 0, 1, 1, 1]})),
    ("a strided of an inner run of zero", tiny, lambda d: _ref(d, {"strided": ["n", 0, 0, 0, 1, 1, 1]})),
    ("a strided of a base that is no integer", tiny, lambda d: _ref(d, {"strided": ["n", 0, "x", 1, 1, 1, 1]})),
    ("a strided past its collection", tiny, lambda d: _ref(d, {"strided": ["n", 0, 0, 1, 1, 1, 9]})),
    ("an explicit that is no list", tiny, lambda d: _ref(d, {"explicit": 1})),
    ("an explicit item of two items", tiny, lambda d: _ref(d, {"explicit": [["n", 0]]})),
    ("an explicit item of a space that is no string", tiny, lambda d: _ref(d, {"explicit": [[1, 0, 0]]})),
    ("an alias that is no pair", tiny, lambda d: _ref(d, {"alias": 1})),
    ("an alias of a node that is no natural", tiny, lambda d: _ref(d, {"alias": [-1, 0]})),
    ("an alias to a node past the nodes", tiny, lambda d: _ref(d, {"alias": [9, 0]})),
    ("a grid that is no list", tiny, lambda d: _grid(d, 1)),
    ("a grid of a run of zero", tiny, lambda d: _grid(d, [0, [["p", 0, 0, 1, [[2, 4]]]]])),
    ("a grid of no lanes", tiny, lambda d: _grid(d, [2, []])),
    ("a grid lane of four items", tiny, lambda d: _grid(d, [2, [["p", 0, 0, 1]]])),
    ("a grid lane of a space that is no space", tiny, lambda d: _grid(d, [2, [["g", 0, 0, 1, [[2, 4]]]]])),
    ("a grid level of a radix of one", tiny, lambda d: _grid(d, [2, [["p", 0, 0, 1, [[1, 4]]], ["p", 0, 2, 1, [[1, 4]]]]])),
    ("a grid level that is no pair", tiny, lambda d: _grid(d, [2, [["p", 0, 0, 1, [[2]]], ["p", 0, 2, 1, [[2, 4]]]]])),
    ("a grid whose lanes' radices multiply to other counts", tiny,
     lambda d: _grid(d, [2, [["p", 0, 0, 1, [[2, 4]]], ["p", 0, 2, 1, [[3, 4]]]]])),
    ("a grid that an affine spells", tiny, lambda d: _grid(d, [2, [["p", 0, 0, 1, []]]])),
    ("a node of a form with no args key", big, lambda d: _node(d, 3).pop("args")),
    ("a batch with an axis entry short", big, lambda d: _node(d, 4).update(axes=[0])),
    ("a batch calling with arguments short", big, lambda d: (_node(d, 4)["args"].pop(), _node(d, 4).update(axes=[0]))),
    ("a batch whose mapped argument is no array", big, lambda d: _node(d, 4)["args"][0].update(type=_type_at(d, V32))),
    ("a batch whose member type is another", big, lambda d: _node(d, 4)["args"][1].update(type=_type_at(d, ["Value", 16]))),
    ("a batch over an axis its type has no array for", big, lambda d: _node(d, 4).update(axes=[3, None])),
    ("a batch of two mapped lengths", big, lambda d: _node(d, 4)["args"][0].update(type=_type_at(d, ["Array", 3, ["Array", 4, V32]]))),
    ("a scan whose callee returns no pair", big, lambda d: _node(d, 8).update(fn=d["callees"].index("TPair_v1"))),
    ("a scan whose step arity is another", big, lambda d: _node(d, 8).update(n_xs=0)),
    ("a scan whose carry is another type", big, lambda d: _scan_args(d)[0].update(type=_type_at(d, ["Value", 16]))),
    ("a scan whose xs is no array", big, lambda d: _scan_args(d)[1].update(type=_type_at(d, V32))),
    ("a scan whose xs element is another type", big, lambda d: _scan_args(d)[1].update(type=_type_at(d, ["Array", 2, ["Value", 16]]))),
    ("a scan whose shared argument is another type", big, lambda d: _scan_args(d)[2].update(type=_type_at(d, ["Value", 16]))),
    ("a scan whose n_xs is no natural", big, lambda d: _node(d, 8).update(n_xs="x")),
    ("a scan whose n is not its xs length", big, lambda d: _node(d, 8).update(n=5)),
    ("an argument of a count of references other than its type's", tiny, lambda d: _ref(d, {"affine": ["n", 0, 0, 2]})),
    ("a return of a count of references other than its type's", tiny,
     lambda d: _set(d, ("definitions", d["root"], "body", "ret"), {"affine": ["n", 2, 0, 2]})),
    ("a batch of two mapped lengths of members of the parameters' types", big,
     lambda d: _node(d, 4).update(axes=[0, 0], args=[_node(d, 4)["args"][0], {"refs": {"affine": ["n", 0, 0, 3]},
                                                                               "type": _type_at(d, ["Array", 3, V32])}])),
    ("a scan whose callee returns no pair of carry and y", big, lambda d: _node(d, 8).update(fn=d["callees"].index("TCol_v1"))),
    ("a scan whose carry parameter is another type than the carry", big,
     lambda d: _scan_of(d, "TSkip_v1", [["c", ["Value", 16]], ["x", V32]], {"explicit": [["p", 1, 0], ["p", 1, 0]]}, _scan_args(d)[:2], 1)),
    ("a scan whose xs is a value", big,
     lambda d: _scan_args(d)[1].update(type=_type_at(d, V32), refs={"affine": ["n", 0, 0, 1]})),
    ("a scan whose xs element is another type than its parameter", big,
     lambda d: _scan_args(d)[1].update(type=_type_at(d, ["Array", 4, ["Value", 16]]))),
    ("a scan of two xs of two lengths", big,
     lambda d: _scan_of(d, "TSkip2_v1", [["c", V32], ["x", V32], ["y", V32]], {"explicit": [["p", 1, 0], ["p", 2, 0]]},
                        [*_scan_args(d)[:2], {"refs": {"affine": ["n", 0, 0, 3]}, "type": _type_at(d, ["Array", 3, V32])}], 2)),
    ("a return of references other than its type's leaves", big,
     lambda d: _set(d, ("definitions", "TPair_v1", "body", "ret_type"), V32)),
    ("a static float of no hex string", big, lambda d: _static(d, "C", '"0x1.8000000000000p+0"', "5", {"f64": 5})),
    ("a static dict of no object", big, lambda d: _static(d, "D", '{"a":[1,"x"],"k":2}', "5", {"dict": 5})),
    ("an input node read in leaves of two widths", lambda: json.loads(recut_vectors.descriptor(
        ["Tuple", [["Array", 4, ["Value", 8]], ["Array", 2, ["Value", 16]]]], {"concat": [{"affine": ["n", 0, 0, 4]}, {"affine": ["n", 0, 0, 2]}]})),
     lambda d: None),
    ("a strided read of mixed leaves as a type of mixed leaves", lambda: _pairs("Pairsshifted4_v1", ["Array", 3, ["Tuple", [["Value", 8], ["Value", 16]]]],
                                                                                {"strided": ["p", 0, 0, 1, 1, 0, 6]}),
     lambda d: None),
    ("a strided read of a parameter of mixed leaves as one width", lambda: _pairs("Sum4w8_v1", ["Array", 4, ["Value", 8]],
                                                                                 {"strided": ["p", 0, 0, 2, 1, 0, 4]}),
     lambda d: None),
    ("a grid read in leaves of two widths past what is walked", lambda: _interleaved(2 ** 19 + 1), lambda d: None),
    ("a grid read as a type of mixed leaves", lambda: _pairs("PairsGrid4_v1", ["Array", 2, ["Tuple", [["Value", 8], ["Value", 16]]]],
                                                              {"grid": [1, [["p", 0, 0, 1, [[2, 4]]], ["p", 0, 0, 1, [[2, 2]]]]]}),
     lambda d: None),
    ("a grid whose lane reads leaves of two widths as one width", lambda: _pairs("SumGrid4w8_v1", ["Array", 4, ["Value", 8]],
                                                                               {"grid": [1, [["p", 0, 0, 1, [[2, 4]]], ["p", 0, 0, 1, [[2, 1]]]]]}),
     lambda d: None),
    ("a static function reference that is no string", big,
     lambda d: _static(d, "OP", '{"fn":"TAdd32_v1"}', '{"fn":5}', {"fn": 5})),
]


#: the cases whose primitives are no one of `spec.REG`: the registry is the descriptor's own declarations
DECLARED = ("a strided read of mixed leaves as a type of mixed leaves", "a strided read of a parameter of mixed leaves as one width",
            "a grid read as a type of mixed leaves", "a grid whose lane reads leaves of two widths as one width")


def cases() -> list[tuple[str, str, object]]:
    out = []
    for name, base, edit in CASES:
        d = base()
        made = edit(d)
        text = made if isinstance(made, str) else json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        out.append((f"lean refusal: {name}", text, R._declared(json.loads(text)) if name in DECLARED else spec.REG))
    return out
