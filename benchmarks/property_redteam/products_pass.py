"""Data products on property/v1's pass: the gate model's nonlinear units read on Programs, and the honest int8
Definitions' census with them.

The gate model (`Security/Definitions/Accounting/Gates/Model.lean`, `dataProducts`, `nonlinear`) frees only linear maps
whose coefficients are fixed before the run's coins: constants, registered values (the pre-committed weights),
additions and wiring. Every s8-family lane whose two multiplicands both depend on a run input is a *data product*, one
unit whatever their width or range: an exact product of two data values is a universal gate (an AND of two bits is one,
and a bitwise codec computes any function with them for free otherwise, `Accounting.SecurityProofs.Gates.
EncodeChainRefuted`, `bitwise_codec.py`). Every other nonlinear op is a budgeted unit, as property/v1 counts it.
`Property` caps the data products per certified byte read, beside the budgeted classes.

`ProductsPass` is property/v1's pass (`verity.ml.circuits.stacking`) that also counts each family step's data products,
lane by lane, under ``units["products"]``. A lane reads one leaf of each multiplicand, whose run dependence is its
producer's (a node's output segment) or, for a leaf of the Definition's parameter, the parameter's, as the pass reads
parameters. A batch or scan of a family step, which the int8 Definitions don't have, counts all its lanes when both
multiplicands have a run-time leaf (an upper bound).

`int8_census` reads vLLM's exact-int8 Definitions (`int8_exact.py`): `Int8Linear_v1` (`Int8RowQuant`, then the s8×s8
IMMA chain against registered weights), `Int8KvQuant_v1` (the KV head's quantizers) and `Int8LutAttentionRow_v1` (`Q′`,
`Q′K′ᵀ`, the LUT softmax, `P V′`). The linear multiplies activations by registered weights, a linear map, so it has no
data product; the KV quantizers have no family step; the attention row's `Q′K′ᵀ` and `P V′` multiply run-time values by
run-time values, `S·D` data products each, `2·S·D` a row. Its nonlinear units are its budgeted units (property/v1's
class counts, unchanged) plus those.

    python products_pass.py    # the census of each at the shapes in `SHAPES`
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from verity.ml.circuits import stacking as P
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import is_input

#: (Definition, statics, which parameters depend on a run input): the served shapes measured, Llama-3.1-8B's head dim
SHAPES: tuple[tuple[str, dict, tuple[bool, ...]], ...] = (
    ("Int8Linear", {"K": 4096, "N": 64}, (True, False, False)),
    ("Int8KvQuant", {"S": 64, "D": 128}, (True, True, True)),
    ("Int8LutAttentionRow", {"D": 128, "S": 64, "SCALE": 0.08838834764831845}, (True,) * 6),
)


def lanes(refs: Any, pdyn: Sequence[bool], dyns: Sequence[bool]) -> list[bool]:
    """Each leaf `refs` reads, in order: whether it depends on a run input."""
    return [pdyn[i] if s == "p" else dyns[i] for s, i, _j in refs]


class ProductsPass(P._Pass):
    """property/v1's pass, with each family step's data products under ``units["products"]``."""

    def walk(self, body, pdyn, prng, def_id, segs, dyns, up):
        seen = set()
        for k, t in super().walk(body, pdyn, prng, def_id, segs, dyns, up):
            nd = body.nodes[k]
            if k not in seen and not is_input(nd.fn) and isinstance(nd.fn, PrimitiveDefinition) and nd.fn.id in P.FAMILY:
                st = P.FAMILY[nd.fn.id]
                off = 1 if st.acc else 0
                a, b = (lanes(nd.args[off + m].refs, pdyn, dyns) for m in (0, 1))
                if nd.form in ("batch", "scan"):
                    products = st.k * nd.n if any(a) and any(b) else 0
                else:
                    products = sum(1 for da, db in zip(a, b) if da and db)
                t.units["products"] += products
            seen.add(k)
            yield k, t


def count(fn: Any, dyn: Sequence[bool], ranges: Sequence[P.Range] | None = None) -> P.Tally:
    """`stacking.count` on `ProductsPass`: one call of `fn` whose parameters depend on a run input as `dyn` says, with
    value ranges `ranges` (None: any word)."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    rng = tuple(ranges) if ranges is not None else (None,) * len(dyn)
    p = ProductsPass()
    out = P.Tally()
    up: dict = {}
    for _k, t in p.walk(fn.body, tuple(dyn), rng, fn.id, [], [], up):
        out.add(t)
    if up:
        out.units["gathers"] += sum(bits for _refs, bits, _kind in up.values())
    return out


def census(fn: Any, dyn: Sequence[bool], ranges: Sequence[P.Range] | None = None) -> dict:
    """One call's budgeted units by class, its data products, its nonlinear units (the two together), family steps,
    MACs and affine maps, and refused gates."""
    t = count(fn, dyn, ranges)
    units = {c: t.units.get(c, 0) for c in P.BUDGETED}
    products = t.units.get("products", 0)
    return {"units": units, "products": products, "nonlinear": sum(units.values()) + products, "family": t.family,
            "macs": t.macs, "affine": t.affine, "refused": {f"{c}:{f}": n for (c, f), n in t.refused.items()}}


def int8_census(shapes: Sequence[tuple[str, dict, tuple[bool, ...]]] = SHAPES) -> dict[str, dict]:
    """`census` of each exact-int8 Definition at its shape, with property/v1's own class counts beside it (`stacking.count`)."""
    from verity.ml.definitions.registry import int8_exact as I8

    out = {}
    for name, statics, dyn in shapes:
        fn = bind(getattr(I8, name), **statics)
        c = census(fn, dyn)
        base = P.count(fn, dyn)
        c["property_v1"] = {k: base.units.get(k, 0) for k in P.BUDGETED}
        out[f"{name}{json.dumps(statics, separators=(',', '='))}"] = c
    return out


if __name__ == "__main__":
    print(json.dumps(int8_census(), indent=1))
