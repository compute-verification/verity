"""verity_vllm.program.predict.score.first_divergence: the first place a predicted and a traced descriptor differ, in the order a
reader looks: the program's function, the set of definitions, a shared definition, the tables, then the root's body node by node."""
from __future__ import annotations

from typing import Any

import pytest

from verity_vllm.program.predict.score import first_divergence

ARG = {"type": 0, "refs": [["v", 1]]}
NODE = {"fn": 0, "args": [ARG], "form": "call"}


def _desc(nodes: list[dict], **over: Any) -> dict:
    """A descriptor whose root's body holds ``nodes``, with one other definition and two types and callees; ``over`` replaces a
    key of the descriptor, ``ret`` the body's return and ``defs`` the definitions beside the root."""
    body = {"nodes": nodes, "ret": over.pop("ret", {"v": 1})}
    defs = over.pop("defs", {"D": {"k": 1}})
    return {"program_fn": "f", "root": "R", "definitions": {"R": {"body": body}, **defs}, "types": ["t0", "t1"], "callees": ["c0", "c1"],
            "schema": ["s"], **over}


def _without(d: dict, key: str) -> dict:
    return {k: v for k, v in d.items() if k != key}


@pytest.mark.parametrize("pred,traced,where", [
    (_desc([NODE], program_fn="f" * 400), _desc([NODE], program_fn="g"), {"where": "program_fn", "predicted": "f" * 300, "traced": "g"}),
    (_desc([NODE], program_fn="f" * 400), _desc([NODE], program_fn="g" * 400), {"where": "program_fn", "predicted": "f" * 300, "traced": "g" * 300}),
    (_without(_desc([NODE]), "program_fn"), _desc([NODE]), {"where": "program_fn", "predicted": "None", "traced": "f"}),
    (_desc([NODE], defs={"D": {"k": 1}, "P": {}}), _desc([NODE], defs={"D": {"k": 1}, "T1": {}, "T2": {}}),
     {"where": "definitions", "only_predicted": ["P"], "only_traced": ["T1", "T2"], "n_only_predicted": 1, "n_only_traced": 2}),
    (_desc([NODE], defs={f"P{i}": {} for i in range(8)}), _desc([NODE], defs={}),
     {"where": "definitions", "only_predicted": [f"P{i}" for i in range(6)], "only_traced": [], "n_only_predicted": 8, "n_only_traced": 0}),
    (_desc([NODE], defs={}), _desc([NODE], defs={f"T{i}": {} for i in range(8)}),
     {"where": "definitions", "only_predicted": [], "only_traced": [f"T{i}" for i in range(6)], "n_only_predicted": 0, "n_only_traced": 8}),
    (_desc([NODE], defs={"D": {"k": 1, "a": 0}, "E": {"k": 1}}), _desc([], defs={"D": {"k": 2, "b": 0}, "E": {"k": 3}}),
     {"where": "definition", "definition": "D", "fields": ["a", "b", "k"], "n_differing": 2}),
    (_desc([NODE]), _desc([], types=["t0", "tX"]),
     {"where": "types", "index": 1, "predicted": "t1", "traced": "tX", "n_predicted": 2, "n_traced": 2}),
    (_desc([NODE]), _desc([NODE], callees=["c0"]),
     {"where": "callees", "index": 1, "predicted": "c1", "traced": "None", "n_predicted": 2, "n_traced": 1}),
    (_without(_desc([NODE]), "schema"), _desc([NODE], schema=["s" * 400]),
     {"where": "schema", "index": 0, "predicted": "None", "traced": "s" * 300, "n_predicted": 0, "n_traced": 1}),
    (_desc([NODE], schema=["p" * 400]), _desc([NODE], schema=["s"]),
     {"where": "schema", "index": 0, "predicted": "p" * 300, "traced": "s", "n_predicted": 1, "n_traced": 1}),
    (_without(_desc([NODE]), "schema"), _desc([NODE], schema=[]),
     {"where": "schema", "index": 0, "predicted": "None", "traced": "None", "n_predicted": 0, "n_traced": 0}),
    (_desc([NODE], types=0), _desc([NODE], types=["t0"]),
     {"where": "types", "index": 0, "predicted": "None", "traced": "t0", "n_predicted": 0, "n_traced": 1}),
    (_desc([NODE, {"fn": 0, "args": []}]), _desc([NODE, {"fn": 1, "args": []}, NODE]),
     {"where": "root node", "node": 1, "callee_predicted": "c0", "callee_traced": "c1"}),
    (_desc([{"fn": 0}], callees=["c" * 400, "d" * 400]), _desc([{"fn": 1}], callees=["c" * 400, "d" * 400]),
     {"where": "root node", "node": 0, "callee_predicted": "c" * 200, "callee_traced": "d" * 200}),
    (_desc([{"fn": 0, "args": [ARG, {"type": 0, "refs": [1]}]}]), _desc([{"fn": 0, "args": [ARG, {"type": 1, "refs": [2]}]}]),
     {"where": "root node", "node": 0, "callee_predicted": "c0", "callee_traced": "c0", "arg": 1, "type_predicted": "t0", "type_traced": "t1",
      "refs_predicted": "[1]", "refs_traced": "[2]"}),
    (_desc([{"fn": 1, "args": [{"type": 1, "refs": ["r" * 400]}]}]), _desc([{"fn": 1, "args": [{"type": 1}]}]),
     {"where": "root node", "node": 0, "callee_predicted": "c1", "callee_traced": "c1", "arg": 0, "type_predicted": "t1", "type_traced": "t1",
      "refs_predicted": '["' + "r" * 238, "refs_traced": "null"}),
    (_desc([{"fn": 0, "args": [{"type": 0, "refs": {"b": 1, "a": 2}}]}], types=["t" * 400, "u" * 400]),
     _desc([{"fn": 0, "args": [{"type": 1, "refs": {"b": 1, "a": 3}}]}], types=["t" * 400, "u" * 400]),
     {"where": "root node", "node": 0, "callee_predicted": "c0", "callee_traced": "c0", "arg": 0, "type_predicted": "t" * 160, "type_traced": "u" * 160,
      "refs_predicted": '{"a": 2, "b": 1}', "refs_traced": '{"a": 3, "b": 1}'}),
    (_desc([{"fn": 0, "args": [ARG], "form": "x"}]), _desc([{"fn": 0, "args": [ARG], "form": "y", "extra": 1}]),
     {"where": "root node", "node": 0, "callee_predicted": "c0", "callee_traced": "c0", "fields": ["extra", "form"]}),
    (_desc([{"fn": 0, "form": "x"}]), _desc([{"fn": 0, "form": "y"}]),
     {"where": "root node", "node": 0, "callee_predicted": "c0", "callee_traced": "c0", "fields": ["form"]}),
    (_desc([{"fn": 0, "args": None}]), _desc([{"fn": 0, "args": [ARG]}]),
     {"where": "root node", "node": 0, "callee_predicted": "c0", "callee_traced": "c0", "fields": ["args"]}),
    (_desc([NODE]), _desc([NODE, NODE]), {"where": "root length", "n_predicted": 1, "n_traced": 2}),
    (_desc([NODE], ret={"v": 1}), _desc([NODE], ret={"v": 2}), {"where": "root ret", "predicted": '{"v": 1}', "traced": '{"v": 2}'}),
    (_desc([NODE], ret=None), _desc([NODE], ret=["r" * 400]), {"where": "root ret", "predicted": "null", "traced": '["' + "r" * 298}),
    (_desc([NODE], ret=["p" * 400]), _desc([NODE], ret=None), {"where": "root ret", "predicted": '["' + "p" * 298, "traced": "null"}),
    (_desc([NODE]), _desc([NODE], note="an annotation"), {"where": "annotations or encoding only"}),
], ids=["program_fn", "program_fn-both-long", "program_fn-absent", "definitions", "definitions-six-named", "definitions-six-traced", "definition",
        "types", "callees-shorter", "schema-absent", "schema-long", "schema-absent-and-empty", "types-zero", "node-callee", "node-callee-long",
        "node-arg", "node-arg-refs", "node-arg-long-types-keyed-refs", "node-fields", "node-no-args", "node-args-null", "root-length", "root-ret",
        "root-ret-none", "root-ret-long", "equal"])
def test_first_divergence_names_the_first_place_two_descriptors_differ(pred: dict, traced: dict, where: dict):
    """Each place in the order it is looked at, with what it reports there: the two values cut to their first characters, the
    names only one side holds (six of them) and their counts, a definition's differing fields, a table's first differing
    index, a root node's callees and, where they agree, its first differing argument (its references with their keys in
    order) or its differing fields."""
    assert first_divergence(pred, traced) == where
