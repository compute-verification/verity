from __future__ import annotations

import json

import pytest

from verity.circuits.codec import descriptor_id, lift_shape, odd_fields
from verity.circuits.ops import product as PR

V32, BIT = ["Value", 32], ["Value", 1]
A64 = ["Array", 64, V32]


def prim(fid, params, ret):
    return {"form": "primitive", "id": fid, "params": params, "ret": ret}


def hole(a, passes, batch):
    ret_type = ["Tuple", [a, BIT]] if passes else BIT
    c = {"form": "composite", "id": "C_v1", "bindings": {},
         "body": {"nodes": [{"args": [{"refs": {"affine": ["p", 1, 0, 1]}, "type": 1}], "fn": 2, "form": "primitive",
                             "out": 1}],
                  "params": [["a", a], ["b", BIT]], "ret_type": ret_type,
                  "ret": {"concat": [{"affine": ["p", 0, 0, 1]}, {"affine": ["n", 0, 0, 1]}]} if passes
                  else {"affine": ["n", 0, 0, 1]}}}
    form = {"form": "batch", "n": 2, "axes": []} if batch else {"form": "primitive"}
    root = {"form": "composite", "id": "Root[CProg_v1]_v1", "bindings": {},
            "body": {"nodes": [{"args": [], "fn": 0, "out": 0, **form}, {"args": [], "fn": 1, "out": 1, **form},
                               {"args": [], "fn": 3, "out": 2, **form, "form": "batch" if batch else "call"}],
                     "params": [], "ret": {"affine": ["n", 2, 0, 1]}, "ret_type": ret_type}}
    return {"boolean": {}, "callees": ["Input32_v1", "Input1_v1", "Not_v1", "C_v1"], "root": "Root[CProg_v1]_v1",
            "definitions": {"Input32_v1": prim("Input32_v1", [], V32), "Input1_v1": prim("Input1_v1", [], BIT),
                            "Not_v1": prim("Not_v1", [["a", BIT]], BIT), "C_v1": c, "Root[CProg_v1]_v1": root}}


HOLES = {f"{'pass' if p else 'unused'}{'' if a == V32 else '-big'}{'-batch' if batch else ''}": hole(a, p, batch)
         for a, batch in ((V32, False), (V32, True), (A64, False)) for p in (True, False) if not (a == A64 and batch)}


@pytest.mark.parametrize("name", sorted(HOLES))
def test_a_word_parameter_reaching_no_word_primitive_is_left_unlifted_and_flagged(name):
    assert lift_shape(HOLES[name], "C_v1") == ([], ["C_v1"])


def test_six_holes_as_in_leans_cases():
    assert len(HOLES) == 6


def test_a_word_primitive_without_a_boolean_definition_is_refused_not_flagged():
    d = {"boolean": {}, "callees": ["Input32_v1", "TAdd32_v1"],
         "definitions": {"Input32_v1": prim("Input32_v1", [], V32),
                         "TAdd32_v1": prim("TAdd32_v1", [["a", V32], ["b", V32]], V32)}}
    with pytest.raises(ValueError, match="TAdd32_v1' has no Boolean Definition"):
        lift_shape(d, "TAdd32_v1")


@pytest.mark.parametrize("tokens", [1, 2])
def test_products_template_is_lifted_with_nothing_left_wide(tokens):
    w, x = PR.example(4, 16)
    p = PR.product(w, x if tokens == 1 else x.reshape(1, 16).repeat(2, 0), k=16, s=2, units=2, block=2, k_log=22, g=2)
    lifted, wide = lift_shape(json.loads(p.description), descriptor_id(p.template))
    assert wide == [] and descriptor_id(p.template) in lifted and len(lifted) == 3


def test_x_of_3_bits_then_y_is_flagged_as_in_leans_case():
    c = {"form": "composite", "id": "C_v1", "bindings": {},
         "body": {"nodes": [], "params": [["x", ["Value", 3]], ["y", ["Value", 16]]],
                  "ret_type": ["Tuple", [["Value", 16], ["Array", 2, ["Value", 5]]]], "ret": {"affine": ["p", 1, 0, 1]}}}
    assert odd_fields({"definitions": {"C_v1": c}}, "C_v1") == [("param", "x", 3, True), ("ret", "1", 10, False)]


@pytest.mark.parametrize("tokens", [1, 2])
def test_products_template_has_every_field_a_multiple_of_16(tokens):
    w, x = PR.example(4, 16)
    p = PR.product(w, x if tokens == 1 else x.reshape(1, 16).repeat(2, 0), k=16, s=2, units=2, block=2, k_log=22, g=2)
    assert odd_fields(json.loads(p.description), descriptor_id(p.template)) == []
