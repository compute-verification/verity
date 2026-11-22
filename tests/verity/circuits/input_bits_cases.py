"""The cases on which `program.input_bits` is held to Lean's `Flock.Program.inputBits` (`program-difftest`'s
`input_bits`, `tests/tools/difftest/test_input_bits_port.py`): declared circuits built here, each with its descriptor text
and its Python root, and an optional ``rebind`` of one root node's callee, a description no decoder gives."""

from __future__ import annotations

from typing import Any

import stand_in  # noqa: F401  (the test primitives' Boolean bodies)

from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program, input_bits_of, input_id, is_input
from verity.circuits.refs import tuple_of
from verity.circuits.types import Array, Tuple, Value

XOR16 = PrimitiveDefinition("InputBitsXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                            register=False)
ZERO16 = PrimitiveDefinition("InputBitsZero16", 1, (), Value(16), lambda: 0, register=False)
STEP = bind(CompositeDefinition("InputBitsStep", 0, (), lambda S: ((("c", Value(16)), ("x", Value(16))),
                                                                   Tuple(Value(16), Value(16))),
                                lambda B, S, c, x: tuple_of(B.call(XOR16, c, x), B.call(XOR16, c, x)), register=False))


def _root(name: str, params: tuple, ret: Any, body) -> Any:
    return bind(CompositeDefinition(name, 0, (), lambda S: (params, ret), body, register=False))


def kinds() -> Program:
    """One input node of each binding, the run's a batch of four 16-bit leaves: 64, 16 and 16 bits."""
    root = _root("InputBitsKinds", (("x", Array(4, Value(16))), ("r", Value(16)), ("s", Value(16))), Array(4, Value(16)),
                 lambda B, S, x, r, s: B.batch(XOR16, B.batch(XOR16, x, r, axes=(0, None)), s, axes=(0, None)))
    return Program(root, registered=("r",), supplied=("s",))


def batch() -> Program:
    """A registered batch of 8 members of 16 bits beside a run input of 16: 16 and 128 bits."""
    root = _root("InputBitsBatch", (("x", Value(16)), ("w", Array(8, Value(16)))), Array(8, Value(16)),
                 lambda B, S, x, w: B.batch(XOR16, w, x, axes=(0, None)))
    return Program(root, registered=("w",))


def scan() -> Program:
    """A scan of 4 iterations at the root, over a run input of 4 leaves and a run carry: its node is no input node, so
    the inputs are the two run nodes' 64 + 16 bits."""
    root = _root("InputBitsScan", (("c", Value(16)), ("xs", Array(4, Value(16)))), Tuple(Value(16), Array(4, Value(16))),
                 lambda B, S, c, xs: B.scan(STEP, c, (xs,)))
    return Program(root)


def empty() -> Program:
    """A root with no input node: three zeros."""
    return Program(_root("InputBitsEmpty", (), Value(16), lambda B, S: B.call(ZERO16)))


def text(program: Program) -> str:
    return b"".join(codec.descriptor_chunks(program)).decode()


#: name: (program, rebind or None, the bits or the refusal's code)
CASES: dict[str, tuple[Any, dict | None, dict[str, int] | str]] = {
    "one node of each kind": (kinds, None, {"run": 64, "registration": 16, "supplied": 16}),
    "a batch": (batch, None, {"run": 16, "registration": 128, "supplied": 0}),
    "a scan at the root": (scan, None, {"run": 80, "registration": 0, "supplied": 0}),
    "no input node": (empty, None, {"run": 0, "registration": 0, "supplied": 0}),
    "a binding no id has": (kinds, {"node": 1, "key": "RegisteredInput16_v1", "binding": "later"}, "input-binding"),
    "a binding another id's": (kinds, {"node": 1, "key": "RegisteredInput16_v1", "binding": "run"}, "input-binding"),
    "an input id with no binding": (kinds, {"node": 1, "key": "RegisteredInput16_v1", "binding": None}, "input-binding"),
    "a binding on an id with no width": (kinds, {"node": 1, "key": "Gate16_v1", "binding": "registration"}, "input-width"),
}


def instance(name: str) -> dict:
    """The port's input for case ``name``: the descriptor's text, with its ``rebind`` where it has one."""
    make, rebind, _ = CASES[name]
    return {"descriptor": text(make()), **({"rebind": rebind} if rebind is not None else {})}


def python(value: dict) -> dict:
    """`program.input_bits_of` on the Python root ``value``'s descriptor was encoded from, after its ``rebind``: the bits
    with ``ok``, or the refusal's code, as the port answers."""
    make = next(m for m, _, _ in CASES.values() if text(m()) == value["descriptor"])
    nodes = [(nd.fn.id, input_id(nd.fn.id)[0] if is_input(nd.fn) else None, nd.form, nd.n) for nd in make().root.body.nodes]
    if (r := value.get("rebind")) is not None:
        _, _, form, n = nodes[r["node"]]
        nodes[r["node"]] = (r["key"], r["binding"], form, n)
    try:
        return {"ok": True, **input_bits_of(nodes)}
    except ValueError as e:
        return {"ok": False, "error": str(e).split(":")[0]}
