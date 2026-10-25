"""Serve_v1 on bits, all the way down: the input the benchmark's Boolean leg needs, which `ir="boolean"` can't derive.

`verity.ml.definitions.boolean.derived` lifts a derived Program (`Derived[...]`, one flat body of registered Definitions),
each Call to the Boolean version of its Definition, and refuses Serve_v1: Serve_v1 is no derived function, and ten of
the composites it reaches (Attention_v1, Gemm_v1, Final_v1, …) have no registered Boolean version. `on_bits(fn)` lifts
every Definition `fn` reaches instead: a Definition with a registered Boolean version becomes that version
(`boolean_version`, the table `ir="boolean"` reads), a word constant `Const<w>[0x…]` becomes its bits, and any other
composite is re-emitted on bits with each of its Calls lifted the same way (`boolean._bit_replayer`, the replay `lift`
uses for one flat body). Node indices and the word-leaf-to-bits correspondence carry over as they do for `lift`.

The result is Boolean throughout (`purity`), but it is this benchmark's construction, not a served Program: no
`sigma_check` holds it to the word Program's values, and nothing outside `benchmarks/verifier_cost` reads it.
"""
from __future__ import annotations

import re

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import forms as F
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.definitions import boolean as BB

_CONST = re.compile(r"Const(\d+)\[0x([0-9a-f]+)\]")


def _constant(fn: PrimitiveDefinition):
    m = _CONST.fullmatch(fn.name)
    if m is None or fn.signature()[0]:
        raise BB.NoBooleanVersion({descriptor_id(fn): "a primitive with no Boolean version that is no word constant"})
    w, v = int(m.group(1)), int(m.group(2), 16)
    return bind(definition(f"Bits{fn.name}", 1, lambda C: F.const_bits(v, w), (), Array(w, BL.BIT), register=False))


def on_bits(fn, memo: dict | None = None):
    """`fn` and every Definition it reaches on bits (the module docstring)."""
    memo = {} if memo is None else memo
    fid = descriptor_id(fn)
    if fid in memo:
        return memo[fid]
    try:
        v = BB.boolean_version(fn)
        out = bind(v) if isinstance(v, CompositeDefinition) else v
    except BB.NoBooleanVersion:
        if isinstance(fn, PrimitiveDefinition):
            out = _constant(fn)
        else:
            out = _composite(fn, memo)
    memo[fid] = out
    return out


def _composite(fn: SpecializedDefinition, memo: dict):
    body = fn.body
    callee = {id(n.fn): on_bits(n.fn, memo) for n in body.nodes}
    params, ret = fn.signature()
    bparams, bret = tuple((n, BL.bit_type(t)) for n, t in params), BL.bit_type(ret)
    defn = CompositeDefinition(fn.name, fn.version, (*fn.defn.statics, "IR"), lambda S: (bparams, bret),
                               BB._bit_replayer(body, [t for _, t in params], callee), register=False)
    return bind(defn, **fn.bindings, IR="boolean")
