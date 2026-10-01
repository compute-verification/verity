"""The Boolean core: three gates on 1-bit wires, the two constant bits, and the word view of a Boolean Definition
(`PROTOCOL.md` §9).

    And_v1(a: Value<1>, b: Value<1>) -> Value<1>      a AND b
    Xor_v1(a: Value<1>, b: Value<1>) -> Value<1>      a XOR b
    Not_v1(a: Value<1>) -> Value<1>                   NOT a
    Const1[0x0]_v1, Const1[0x1]_v1 : () -> Value<1>   the constants 0 and 1

The format fixes what each of these means; no semantics package defines or re-registers them. A Definition is *Boolean*
(`is_boolean`) when every primitive it reaches is one of them or an `Input1_v1`.

**Elements and the word view.** A Boolean Definition's ports carry bits; an *element* of a port is an innermost
`Array<w, Value<1>>` (one w-bit element, bit j = leaf j, least significant first) or a lone `Value<1>`. `word_type` replaces
each element by `Value<w>`: the word view of a Boolean Definition is the signature of the word Definition it computes, and
`verity.evaluation` evaluates it on words through that view. A port's format (BF16, E4M3, ...) is an annotation
(`verity.ir.annotations`), outside every digest; the word view never reads it.
"""
from __future__ import annotations

import re
import weakref
from typing import Any, Callable

from .defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from .types import Array, Record, Tuple, Type, Value

__all__ = ["AND", "BIT", "CONSTANTS", "GATES", "KINDS", "NOT", "XOR", "bit_type", "constant", "elements", "is_boolean",
           "is_constant_bit", "is_gate", "word_type"]

BIT = Value(1)


def _gate(name: str, arity: int, op: Callable[..., int], doc: str) -> PrimitiveDefinition:
    params = (("a", BIT), ("b", BIT))[:arity]
    return PrimitiveDefinition(name, 1, params, BIT, op, doc=doc, conformance="protocol-defined")


AND = _gate("And", 2, lambda a, b: a & b & 1, "a AND b on 1-bit wires")
XOR = _gate("Xor", 2, lambda a, b: (a ^ b) & 1, "a XOR b on 1-bit wires")
NOT = _gate("Not", 1, lambda a: (a ^ 1) & 1, "NOT a on a 1-bit wire")

#: gate id -> (kind, arity): the table every evaluator and lowering dispatches on; a new primitive kind is a new row
KINDS: dict[str, tuple[str, int]] = {AND.id: ("and", 2), XOR.id: ("xor", 2), NOT.id: ("not", 1)}
GATES: dict[str, PrimitiveDefinition] = {AND.id: AND, XOR.id: XOR, NOT.id: NOT}


def _constant_bit(b: int) -> PrimitiveDefinition:
    p = PrimitiveDefinition(f"Const1[0x{b:x}]", 1, (), BIT, (lambda _b=b: _b), doc=f"constant 1-bit encoding 0x{b:x}",
                            conformance="exact-model-tested")
    p.family = "Const"  # type: ignore[attr-defined]
    return p


#: the constants 0 and 1: the `Const` family's 1-bit members (`verity.ml.prims.const(1, b)` returns these objects)
CONSTANTS: tuple[PrimitiveDefinition, PrimitiveDefinition] = (_constant_bit(0), _constant_bit(1))
_CONSTANT_ID = re.compile(r"^Const1\[0x([01])\]_v1$")


def constant(b: int) -> PrimitiveDefinition:
    if b not in (0, 1):
        raise ValueError(f"a constant bit is 0 or 1, not {b!r}")
    return CONSTANTS[b]


def is_gate(fn: Any) -> bool:
    return isinstance(fn, PrimitiveDefinition) and GATES.get(fn.id) is fn


def is_constant_bit(fn: Any) -> bool:
    return isinstance(fn, PrimitiveDefinition) and _CONSTANT_ID.match(fn.id) is not None and not fn.params


def _is_boolean_primitive(fn: PrimitiveDefinition) -> bool:
    return is_gate(fn) or is_constant_bit(fn) or fn.id == "Input1_v1"


_BOOLEAN: "weakref.WeakKeyDictionary[SpecializedDefinition, bool]" = weakref.WeakKeyDictionary()


def is_boolean(fn: Any) -> bool:
    """Whether every primitive ``fn`` reaches (a primitive, a composite, a complete specialization) is a gate, a constant bit or
    `Input1_v1`."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    if isinstance(fn, PrimitiveDefinition):
        return _is_boolean_primitive(fn)
    if not isinstance(fn, SpecializedDefinition) or not fn.complete:
        return False
    hit = _BOOLEAN.get(fn)
    if hit is None:
        seen: set[int] = set()
        stack, hit = [fn], True
        while stack and hit:
            d = stack.pop()
            if id(d) in seen:
                continue
            seen.add(id(d))
            for n in d.body.nodes:
                if isinstance(n.fn, PrimitiveDefinition):
                    if not _is_boolean_primitive(n.fn):
                        hit = False
                        break
                else:
                    stack.append(n.fn)
        _BOOLEAN[fn] = hit
    return hit


# ---- the word view -----------------------------------------------------------------------------------------------------

def _is_element(t: Type) -> bool:
    return isinstance(t, Array) and t.elem == BIT


def elements(t: Type) -> list[int]:
    """The element widths of a Boolean type, in leaf order."""
    if t == BIT:
        return [1]
    if _is_element(t):
        return [t.n] if t.n else []
    if isinstance(t, Array):
        return elements(t.elem) * t.n
    if isinstance(t, Tuple):
        return [w for f in t.fields for w in elements(f)]
    if isinstance(t, Record):
        return [w for _, f in t.fields for w in elements(f)]
    raise TypeError(f"{t!r} is not a Boolean type (its leaves must be Value<1>)")


def word_type(t: Type) -> Type:
    """``t`` with each element replaced by ``Value<w>``: ``Array<K, Array<16, Value<1>>>`` is ``Array<K, Value<16>>``."""
    if t == BIT:
        return BIT
    if _is_element(t):
        return Value(t.n)
    if isinstance(t, Array):
        return Array(t.n, word_type(t.elem))
    if isinstance(t, Tuple):
        return Tuple(*[word_type(f) for f in t.fields])
    if isinstance(t, Record):
        return Record(*[(k, word_type(f)) for k, f in t.fields])
    raise TypeError(f"{t!r} is not a Boolean type (its leaves must be Value<1>)")


def bit_type(t: Type) -> Type:
    """The Boolean type whose word view is ``t``: each ``Value<w>`` becomes ``Array<w, Value<1>>`` (``Value<1>`` stays a bit)."""
    if isinstance(t, Value):
        return BIT if t.w == 1 else Array(t.w, BIT)
    if isinstance(t, Array):
        return Array(t.n, bit_type(t.elem))
    if isinstance(t, Tuple):
        return Tuple(*[bit_type(f) for f in t.fields])
    if isinstance(t, Record):
        return Record(*[(k, bit_type(f)) for k, f in t.fields])
    raise TypeError(t)
