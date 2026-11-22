"""Descriptors that read an input node at another width (PROTOCOL.md §4.4), for the decoders' difftests: a root over an
input node of two 16-bit leaves whose `ret` reads it as four 8-bit leaves (read recut, q = 2), as 3-bit leaves
(recut-width), as six 8-bit leaves, past its N·q = 4 (recut-overrun), from its second 8-bit leaf (recut-boundary), and the
8-bit read of a node that is no input node, a call of an identity over it (E_REINTERPRET). `Flock.Test.Recut` holds the
Lean decoder's verdicts on the same bytes."""
from __future__ import annotations

import json


def descriptor(ret_type: list, ret: dict, call: bool = False) -> str:
    word, pair = ["Value", 16], ["Array", 2, ["Value", 16]]
    nodes = [{"args": [], "axes": [], "fn": 0, "form": "batch", "n": 2, "out": 1}]
    if call:
        nodes.append({"args": [{"refs": {"affine": ["n", 0, 0, 2]}, "type": 1}], "fn": 1, "form": "call", "out": 1})
    ident = {"form": "composite", "id": "Id_v1", "bindings": {}, "body": {
        "params": [["x", pair]], "ret_type": pair, "ret": {"affine": ["p", 0, 0, 2]}, "nodes": []}}
    defs = {"Input16_v1": {"form": "primitive", "id": "Input16_v1", "params": [], "ret": word},
            "Root[P_v1]_v1": {"form": "composite", "id": "Root[P_v1]_v1", "bindings": {},
                              "body": {"params": [], "ret_type": ret_type, "ret": ret, "nodes": nodes}},
            **({"Id_v1": ident} if call else {})}
    return json.dumps({"boolean": {}, "callees": ["Input16_v1", *(["Id_v1"] if call else [])], "definitions": defs,
                       "program_fn": "P_v1", "root": "Root[P_v1]_v1", "schema": "verity-ir/descriptor/v2",
                       "types": [word, pair, ret_type]}, sort_keys=True, separators=(",", ":"))


def reads() -> list[tuple[str, str]]:
    """Each case's name and descriptor text."""
    return [("recut: read recut", descriptor(["Array", 4, ["Value", 8]], {"affine": ["n", 0, 0, 4]})),
            ("recut: recut-width", descriptor(["Array", 2, ["Value", 3]], {"affine": ["n", 0, 0, 2]})),
            ("recut: recut-overrun", descriptor(["Array", 6, ["Value", 8]], {"affine": ["n", 0, 0, 6]})),
            ("recut: recut-boundary", descriptor(["Array", 2, ["Value", 8]], {"affine": ["n", 0, 1, 2]})),
            ("recut: no input node", descriptor(["Array", 2, ["Value", 8]], {"affine": ["n", 1, 0, 2]}, call=True))]
