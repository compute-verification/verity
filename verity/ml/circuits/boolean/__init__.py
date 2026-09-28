"""The Boolean core: three gates on 1-bit wires, the two constant bits, and the word view of a Boolean Definition
(`PROTOCOL.md` §9).

    And<n>_v1(x: Array<n, Value<1>>) -> Value<1>      x[0] AND ... AND x[n-1], for every n >= 2
    Xor<n>_v1(x: Array<n, Value<1>>) -> Value<1>      x[0] XOR ... XOR x[n-1], for every n >= 2
    Not_v1(a: Value<1>) -> Value<1>                   NOT a
    Const1[0x0]_v1, Const1[0x1]_v1 : () -> Value<1>   the constants 0 and 1

AND and XOR are one family each, a member per number of inputs (`and_gate(n)`, `xor_gate(n)`; `AND` and `XOR` are the
two-input members `And2_v1` and `Xor2_v1`), decoded by id like the constants. The format fixes what each of these means; no
semantics package defines or re-registers them. A Definition is *Boolean* (`is_boolean`) when its signature has only 1-bit
leaves and every primitive it reaches is one of them, an `Input1_v1` or a `RegisteredInput1_v1`.

**Elements and the word view.** A Boolean Definition's ports carry bits; an *element* of a port is an innermost
`Array<w, Value<1>>` (one w-bit element, bit j = leaf j, least significant first) or a lone `Value<1>`. `word_type` replaces
each element by `Value<w>`: the word view of a Boolean Definition is the signature of the word Definition it computes, and
`verity.ml.circuits.evaluation` evaluates it on words through that view. A port's format (BF16, E4M3, ...) is an annotation
(`verity.ml.circuits.annotations`), outside every digest; the word view never reads it.
"""
from __future__ import annotations

import re
import weakref
from typing import Any

from verity.ml.circuits.codec import register_lazy_family
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Record, Tuple, Type, Value

__all__ = ["AND", "BIT", "CONSTANTS", "KINDS", "NOT", "XOR", "and_gate", "bit_type", "constant", "elements", "gate_kind",
           "is_boolean", "is_constant_bit", "is_gate", "operands", "word_type", "xor_gate"]

BIT = Value(1)

#: the gate kinds, in the order evaluators number them; a new kind is a new entry (and a format version)
KINDS: tuple[str, ...] = ("and", "xor", "not")


def _and(*x: int) -> int:
    return int(all(b & 1 for b in x))


def _xor(*x: int) -> int:
    v = 0
    for b in x:
        v ^= b
    return v & 1


_FAMILY = {"and": ("And", _and, "AND"), "xor": ("Xor", _xor, "XOR")}
_N_ARY: dict[tuple[str, int], PrimitiveDefinition] = {}
_N_ARY_ID = re.compile(r"^(And|Xor)([2-9]|[1-9][0-9]+)_v1$")


def _n_ary(kind: str, n: int) -> PrimitiveDefinition:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError(f"an {kind} gate has n >= 2 inputs, not {n!r}")
    p = _N_ARY.get((kind, n))
    if p is None:
        name, op, word = _FAMILY[kind]
        p = PrimitiveDefinition(f"{name}{n}", 1, (("x", Array(n, BIT)),), BIT, op, doc=f"the {word} of {n} bits",
                                conformance="protocol-defined")
        p.family = name  # type: ignore[attr-defined]
        _N_ARY[(kind, n)] = p
    return p


def and_gate(n: int) -> PrimitiveDefinition:
    """`And<n>_v1`: the AND of n >= 2 bits."""
    return _n_ary("and", n)


def xor_gate(n: int) -> PrimitiveDefinition:
    """`Xor<n>_v1`: the XOR of n >= 2 bits."""
    return _n_ary("xor", n)


AND = and_gate(2)
XOR = xor_gate(2)
NOT = PrimitiveDefinition("Not", 1, (("a", BIT),), BIT, lambda a: (a ^ 1) & 1, doc="NOT a on a 1-bit wire",
                          conformance="protocol-defined")

register_lazy_family(_N_ARY_ID, lambda m: _n_ary(m[1].lower(), int(m[2])))


def gate_kind(fn: Any) -> tuple[str, int] | None:
    """``(kind, inputs)`` of a gate of the profile (``kind`` in `KINDS`), else None."""
    if fn is NOT:
        return ("not", 1)
    m = _N_ARY_ID.fullmatch(fn.id) if isinstance(fn, PrimitiveDefinition) else None
    if m is None:
        return None
    key = (m[1].lower(), int(m[2]))
    return key if _N_ARY.get(key) is fn else None


def operands(*bits: Coll) -> Coll:
    """One-bit values as an n-input gate's argument, ``Array<n, Value<1>>``: ``B.call(XOR, operands(a, b))``."""
    if any(c.type != BIT for c in bits):
        raise TypeError("a gate's operands are 1-bit values")
    return Coll(Array(len(bits), BIT), concat([c.refs for c in bits]))


def _constant_bit(b: int) -> PrimitiveDefinition:
    p = PrimitiveDefinition(f"Const1[0x{b:x}]", 1, (), BIT, (lambda _b=b: _b), doc=f"constant 1-bit encoding 0x{b:x}",
                            conformance="exact-model-tested")
    p.family = "Const"  # type: ignore[attr-defined]
    return p


#: the constants 0 and 1: the `Const` family's 1-bit members (`verity.ml.circuits.ops.prims.const(1, b)` returns these objects)
CONSTANTS: tuple[PrimitiveDefinition, PrimitiveDefinition] = (_constant_bit(0), _constant_bit(1))
_CONSTANT_ID = re.compile(r"^Const1\[0x([01])\]_v1$")


def constant(b: int) -> PrimitiveDefinition:
    if b not in (0, 1):
        raise ValueError(f"a constant bit is 0 or 1, not {b!r}")
    return CONSTANTS[b]


def is_gate(fn: Any) -> bool:
    return gate_kind(fn) is not None


def is_constant_bit(fn: Any) -> bool:
    return isinstance(fn, PrimitiveDefinition) and _CONSTANT_ID.fullmatch(fn.id) is not None and not fn.params


def _is_boolean_primitive(fn: PrimitiveDefinition) -> bool:
    return is_gate(fn) or is_constant_bit(fn) or fn.id in ("Input1_v1", "RegisteredInput1_v1")


_BOOLEAN: "weakref.WeakKeyDictionary[SpecializedDefinition, bool]" = weakref.WeakKeyDictionary()


def is_boolean(fn: Any) -> bool:
    """Whether ``fn`` (a primitive, a composite, a complete specialization) has only 1-bit leaves in its signature and every
    primitive it reaches is a gate, a constant bit, `Input1_v1` or `RegisteredInput1_v1`."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    if isinstance(fn, PrimitiveDefinition):
        return _is_boolean_primitive(fn)
    if not isinstance(fn, SpecializedDefinition) or not fn.complete:
        return False
    params, ret = fn.signature()
    if not all(t.width_set() <= {1} for t in (ret, *(t for _, t in params))):
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
