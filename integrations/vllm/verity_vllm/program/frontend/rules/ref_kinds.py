"""Width-conversion composites of the derive frontend for reference mode.

`verity/ml/definitions/registry/ref_prims.py` (vocabulary `verity-ref-vocab/1`) is the reference vocabulary
(`LinearRef_v1{DT="f32"}` / `AddRef` / `MulRef` / `ReluRef`: sequential f32 fma chain from +0, `F32MaxRef(x, +0)`).
What is here are the two element-wise width conversions a `.to(dtype)` cast needs as *one call*: batched
`Bf16ToF32_v1` (exact) and `F32ToBf16Rn_v1` (RNE), both exact-model-tested registered primitives. They are
deliberately unregistered (`register=False`) with a `DeriveRef` prefix so they cannot collide with the reference
vocabulary's names.
"""

from __future__ import annotations

from verity.ml.circuits.defs import Builder, CompositeDefinition
from verity.ml.circuits.refs import Coll
from verity.ml.definitions.registry import prims as P
from verity.ml.circuits.types import Array, Value

F32 = Value(32)
BF16 = Value(16)

_CONF = "derive-lane composite over an exact-model-tested primitive (Bf16ToF32_v1 / F32ToBf16Rn_v1), one element per gate"


def _ew1(prim):
    def body(B: Builder, S, a: Coll) -> Coll:
        return B.batch(prim, a, axes=(0,))
    return body


DeriveRefBf16ToF32 = CompositeDefinition("DeriveRefBf16ToF32", 1, ("N",), lambda S: ((("a", Array(S.N, BF16)),), Array(S.N, F32)), _ew1(P.Bf16ToF32),
                                         doc="element-wise bf16 -> f32 (exact widening)", conformance=_CONF, register=False)
DeriveRefF32ToBf16Rn = CompositeDefinition("DeriveRefF32ToBf16Rn", 1, ("N",), lambda S: ((("a", Array(S.N, F32)),), Array(S.N, BF16)), _ew1(P.F32ToBf16Rn),
                                           doc="element-wise f32 -> bf16 round-to-nearest-even", conformance=_CONF, register=False)

ALL = [DeriveRefBf16ToF32, DeriveRefF32ToBf16Rn]

__all__ = [d.name for d in ALL] + ["ALL"]
