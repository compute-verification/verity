"""The refusals of `codec`'s decoder, one case each: the smallest descriptor that trips that clause and no earlier one, asserted by
the clause's own text, so deleting the clause fails its case (the next check, or the accept, gives another outcome)."""
from __future__ import annotations

import copy
import functools
import json

import pytest

import stand_in  # noqa: F401  (the test primitives' Boolean bodies)
import test_format_spec as spec
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, DefinitionTable, PrimitiveDefinition, bind
from verity.circuits.program import Program

V32 = spec.V32


def _program(name: str, body, registry: DefinitionTable | None = None) -> tuple[dict, DefinitionTable]:
    program = Program(spec._comp(name, (), lambda S: ((("a", V32), ("b", V32)), V32), body))
    return json.loads(spec._canon(codec.encode_program(program))), registry or spec.REG


@functools.cache
def _tiny() -> str:
    return json.dumps(_program("TTiny", lambda B, S, a, b: B.call(spec.TAdd, a, b))[0])


def tiny() -> tuple[dict, DefinitionTable]:
    return json.loads(_tiny()), spec.REG


ROOT = "Root[TTiny_v1]_v1"
STAND_IN = "StandIn[TAdd32]_v1"
Inner = CompositeDefinition("TInner", 1, ("N",), lambda S: ((("a", V32), ("b", V32)), V32), lambda B, S, a, b: B.call(spec.TAdd, a, b),
                            register=False, refuse=lambda r: "N is odd" if r.get("N", 0) % 2 else None)
OUTER_REG = DefinitionTable()
for _p in (spec.TAdd, Inner):
    OUTER_REG.add(_p)
INNER_ID = "TInner_v1{N=2}"


def outer() -> tuple[dict, DefinitionTable]:
    return _program("TOuter", lambda B, S, a, b: B.call(bind(Inner, N=2), a, b), OUTER_REG)


def _root(d: dict) -> dict:
    return d["definitions"][next(f for f in d["definitions"] if f.startswith("Root["))]["body"]


def _node(d: dict, k: int) -> dict:
    return _root(d)["nodes"][k]


def _rekey(d: dict, old: str, new: str) -> None:
    d["definitions"][new] = d["definitions"].pop(old)
    d["callees"][d["callees"].index(old)] = new


def _type_at(d: dict, t: list) -> int:
    if t not in d["types"]:
        d["types"].append(t)
    return d["types"].index(t)


def _inner_input(d: dict) -> None:
    d["definitions"][STAND_IN]["body"]["nodes"][0].update(fn=d["callees"].index("Input32_v1"))


def _outer_binding(d: dict, record: dict, key: str) -> None:
    d["definitions"][INNER_ID]["bindings"] = record
    _rekey(d, INNER_ID, key)


def _alias_forward(d: dict) -> None:
    _node(d, 2)["args"][0]["refs"] = {"alias": [2, 0]}


def _two_names(d: dict) -> None:
    d["definitions"][STAND_IN]["body"]["params"][1][0] = d["definitions"][STAND_IN]["body"]["params"][0][0]


def _past_leaves(d: dict) -> None:
    d["definitions"][STAND_IN]["body"]["nodes"][0]["args"][0]["refs"] = {"affine": ["p", 9, 0, 1]}


def _composite_callee(d: dict) -> None:
    d["callees"].append(STAND_IN)
    d["definitions"][STAND_IN]["body"]["nodes"][0].update(fn=len(d["callees"]) - 1)


def _grid_past_leaves(d: dict) -> None:
    grid = {"grid": [2, [["p", 0, 40, 1, [[2, 4]]], ["p", 0, 2, 1, [[2, 4]]]]]}
    d["definitions"][STAND_IN]["body"]["nodes"][0]["args"][0]["refs"] = grid


def _ret_past_leaves(d: dict) -> None:
    _root(d)["ret"] = {"affine": ["n", 2, 5, 1]}


#: (the codec.py clause, the descriptor builder, its tampering, the refusal's text)
DECODER = [
    ("E_SCHEMA v1", tiny, lambda d: d.update(schema=codec.SCHEMA_V1), "E_SCHEMA: a verity-ir/descriptor/v1 descriptor holds no Boolean bodies"),
    ("E_SCHEMA v0", tiny, lambda d: d.update(schema=codec.SCHEMA_V0), "E_SCHEMA: a verity-ir/descriptor/v0 descriptor holds no Boolean bodies"),
    ("E_SCHEMA other", tiny, lambda d: d.update(schema="verity-ir/descriptor/v9"),
     "E_SCHEMA: unexpected descriptor schema 'verity-ir/descriptor/v9'"),
    ("a type of the table that is no type", tiny, lambda d: d["types"].__setitem__(1, ["Value", "x"]), "type ['Value', 'x']: Value<w> needs a positive int width, got 'x'"),
    ("a callee index outside the table", tiny, lambda d: _node(d, 2).update(fn=99), "callees index 99 outside 0..2"),
    ("a type index outside the table", tiny, lambda d: _node(d, 2).update(out=99), "types index 99 outside 0..3"),
    ("a primitive keyed by another id", tiny, lambda d: d["definitions"]["TAdd32_v1"].update(id="TOther_v1"),
     "definition 'TAdd32_v1': a primitive is keyed by its id, not 'TOther_v1'"),
    ("a primitive of another signature", tiny, lambda d: d["definitions"]["TAdd32_v1"].update(ret=["Value", 16]),
     "primitive TAdd32_v1 signature differs from the registry"),
    ("a primitive with ports the registry lacks", tiny, lambda d: d["definitions"]["TAdd32_v1"].update(ports={"x": 1}),
     "primitive TAdd32_v1 ports differ from the registry"),
    ("a definition of an unknown form", tiny, lambda d: d["definitions"][ROOT].update(form="weird"),
     f"definition '{ROOT}': form 'weird', id '{ROOT}' (a composite's id is Name_vN)"),
    ("a composite id that is not Name_vN", tiny, lambda d: d["definitions"][ROOT].update(id=ROOT + "x"),
     f"definition '{ROOT}': form 'composite', id '{ROOT}x' (a composite's id is Name_vN)"),
    ("a composite id of another version spelling", tiny, lambda d: d["definitions"][ROOT].update(id="Root[TTiny_v1]_v01"),
     f"definition '{ROOT}': form 'composite', id 'Root[TTiny_v1]_v01' (a composite's id is Name_vN)"),
    ("a composite keyed without its binding record", tiny, lambda d: d["definitions"][ROOT].update(bindings={"N": 1}),
     f"definition '{ROOT}': a composite is keyed by its id and binding record, '{ROOT}{{N=1}}'"),
    ("a composite with a parameter name twice", tiny, _two_names, f"definition '{STAND_IN}': parameters: a name twice"),
    ("an input node outside the root", tiny, _inner_input,
     f"definition '{STAND_IN}' node 0: an input node (Input32_v1) appears only in the root's body"),
    ("a node form outside the four", tiny, lambda d: _node(d, 2).update(form="map"), f"definition '{ROOT}' node 2: form 'map' with keys ['args', 'fn', 'form', 'out']"),
    ("a primitive node calling a composite", tiny, _composite_callee,
     f"definition '{STAND_IN}' node 0: a primitive node does not call a composite"),
    ("a batch node with a malformed axis", tiny, lambda d: _node(d, 0).update(axes=["x"]),
     f"definition '{ROOT}' node 0: batch counts {{'axes': ['x'], 'n': 1}}"),
    ("a reference past its collection", tiny, _past_leaves, f"definition '{STAND_IN}' node 0: references {{'affine': ['p', 9, 0, 1]}} name no leaf of a parameter or a node"),
    ("a grid reference past its collection", tiny, _grid_past_leaves,
     f"definition '{STAND_IN}' node 0: grid-overrun: references {{'grid': [2, [['p', 0, 40, 1, [[2, 4]]], ['p', 0, 2, 1, [[2, 4]]]]]}} "
     "reach outside a lane's parameter or node"),
    ("a return reference past its collection", tiny, _ret_past_leaves, f"definition '{ROOT}' ret: references {{'affine': ['n', 2, 5, 1]}} name no leaf of a parameter or a node"),
    ("an alias to no earlier node", tiny, _alias_forward,
     f"definition '{ROOT}' node 2: alias to argument 0 of node 2: aliases name an argument of an earlier node"),
    ("an argument unlike the callee's parameter", tiny, lambda d: _node(d, 2)["args"][0].update(type=_type_at(d, ["Value", 16])),
     f"definition '{ROOT}' node 2: call TAdd32_v1: argument 'a' has type Value<16>, expected Value<32>"),
    ("a declared out unlike the computed one", tiny, lambda d: _node(d, 2).update(out=_type_at(d, ["Value", 16])),
     f"definition '{ROOT}' node 2: declares out Value<16>, computes Value<32>"),
    ("a return type unlike the returned value", tiny, lambda d: _root(d).update(ret_type=["Value", 16]), f"definition '{ROOT}': E_REINTERPRET: the body's return: leaf 0 is typed Value<16> but names leaf 0 of node 2"),
    ("a root with a parameter", tiny, lambda d: _root(d)["params"].append(["x", ["Value", 32]]),
     f"root '{ROOT}': the root is a composite without parameters"),
    ("a binding the registry's composite has no static for", outer, lambda d: _outer_binding(d, {"N": 2, "Z": 1}, "TInner_v1{N=2,Z=1}"),
     "definition 'TInner_v1{N=2,Z=1}': TInner_v1 has no static parameter 'Z'"),
    ("a binding the registry's composite refuses", outer, lambda d: _outer_binding(d, {"N": 3}, "TInner_v1{N=3}"),
     "definition 'TInner_v1{N=3}': TInner_v1 refuses this binding: N is odd"),
    ("a Boolean map that is no object", tiny, lambda d: d.update(boolean=[]), "E_BODY_KEY: `boolean` is list, not an object"),
    ("a word primitive with no Boolean body", tiny, lambda d: d["boolean"].pop("TAdd32_v1"),
     "E_BODY_MISSING: the word primitives ['TAdd32_v1'] have no Boolean body"),
    ("a Boolean key that is no word primitive", tiny, lambda d: d["boolean"].__setitem__(ROOT, STAND_IN),
     f"E_BODY_KEY: `boolean` names ['{ROOT}'], which are no word primitive"),
    ("a Boolean body that is no definition", tiny, lambda d: d["boolean"].__setitem__("TAdd32_v1", "Absent_v1"),
     "E_BODY_NOT_BOOLEAN: TAdd32_v1's body 'Absent_v1' is no definition"),
    ("a Boolean body that is a word Definition", tiny, lambda d: d["boolean"].__setitem__("TAdd32_v1", ROOT),
     f"E_BODY_NOT_BOOLEAN: TAdd32_v1's body {ROOT} is not a Boolean Definition"),
]


@pytest.mark.parametrize("make,edit,text", [c[1:] for c in DECODER], ids=[c[0] for c in DECODER])
def test_the_decoder_refuses(make, edit, text):
    d, registry = make()
    edit(d)
    with pytest.raises(ValueError) as e:
        codec.decode_program(d, registry)
    assert str(e.value).startswith(text), str(e.value)


def _entry(d: dict) -> dict:
    return copy.deepcopy(d["definitions"][ROOT])


def test_an_entry_whose_alias_names_no_earlier_argument_is_refused():
    d, _ = tiny()
    entry = _entry(d)
    entry["body"]["nodes"][2]["args"][0]["refs"] = {"alias": [2, 0]}
    with pytest.raises(ValueError) as e:
        codec.canonical_entry(entry, d)
    assert str(e.value) == f"{ROOT}: alias [2, 0] names no argument of a node before node 2"


def test_an_interned_entry_is_refused_without_its_descriptors_tables():
    d, _ = tiny()
    with pytest.raises(ValueError) as e:
        codec.canonical_entry(_entry(d))
    assert str(e.value) == f"{ROOT}: an interned entry needs its descriptor's `callees`"


def test_a_primitive_family_member_is_refused_unless_its_id_is_its_spelling(monkeypatch):
    monkeypatch.setattr(codec, "_LAZY_PRIMITIVE_FAMILIES", [])
    codec.register_lazy_family(r"TFam([0-9])_v1", lambda m: PrimitiveDefinition(f"TFam{int(m[1]) + 1}", 1, (("a", V32),), V32, lambda a: a,
                                                                                register=False))
    with pytest.raises(KeyError) as e:
        codec.lookup_primitive(DefinitionTable(), "TFam1_v1")
    assert e.value.args[0] == "primitive TFam1_v1 is not in the registry: its family spells that member TFam2_v1"


def test_a_primitive_outside_the_registry_and_its_families_is_refused():
    with pytest.raises(KeyError) as e:
        codec.lookup_primitive(spec.REG, "TAbsent_v1")
    assert e.value.args[0] == "primitive TAbsent_v1 is not in the registry"


def test_an_id_is_made_only_of_a_bound_definition():
    with pytest.raises(AssertionError):
        codec.descriptor_id(spec.PROG)
