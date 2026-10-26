"""Thin lanes on property/v1's pass: the repriced gate model's charge read on Programs, and the honest int8 Definitions'
census with it.

The repriced model (`Security/Definitions/Accounting/Gates/Model.lean`, `thinLanes`, `capped`) charges a compare-select
unit for each *thin lane*: an s8-family lane whose two multiplicands both depend on a run input and one of which is held
to at most two values (`hi ≤ lo + 1`). Such a lane is a selection (`a·b` with `a ∈ {u, u + 1}` is `u·b` plus `b` or 0);
without the charge an AND of two bits is a free MAC, and a bitwise codec computes any function for no unit
(`Accounting.SecurityProofs.Gates.EncodeChainRefuted`, `bitwise_codec.py`). `Property` also caps the s8 family's MACs per
certified byte read.

`ThinPass` is property/v1's pass (`verity.ml.circuits.stacking`) that also counts each family step's thin lanes, lane by
lane, under ``units["thin"]``. A lane reads one leaf of each multiplicand, whose run dependence and range are its
producer's (a node's output segment) or, for a leaf of the Definition's parameter, the parameter's, as the pass reads
parameters. A parameter's lanes share its union range, so a parameter that mixes a two-valued run-time lane with wider
ones reads wide: the count is exact where each multiplicand is one producer's output or a registered word, as in the
int8 Definitions, and a lower bound otherwise. A batch or scan of a family step, which they don't have, counts all its
lanes when any lane could be thin.

`int8_census` reads vLLM's exact-int8 Definitions (`int8_exact.py`): `Int8Linear_v1` (`Int8RowQuant`, then the s8×s8
IMMA chain against registered weights), `Int8KvQuant_v1` (the KV head's quantizers) and `Int8LutAttentionRow_v1` (`Q′`,
`Q′K′ᵀ`, the LUT softmax, `P V′`). Their family steps multiply quantizer codes (`[−127, 127]`), LUT bytes (`[0, 127]`,
or the constant 0 of `P_0` and the pads) and registered weights, so no lane is thin and the repricing leaves their
class counts as property/v1 has them. Their MACs are one a certified byte read per token row (the linear's weights,
the row's K′ and V′), what the MAC cap must allow per row a byte serves.

    python thin_pass.py    # the census of each at the shapes in `SHAPES`
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


def two_valued(r: P.Range) -> bool:
    """A range of at most two values: `hi ≤ lo + 1` (None, any word, and the empty range are not)."""
    return r is not None and r[0] <= r[1] <= r[0] + 1


def lanes(refs: Any, pdyn: Sequence[bool], prng: Sequence[P.Range], segs: Sequence[P.Segs],
          dyns: Sequence[bool]) -> list[tuple[bool, P.Range]]:
    """Each leaf `refs` reads, in order: its run dependence and its range."""
    out = []
    for s, i, j in refs:
        if s == "p":
            out.append((pdyn[i], prng[i]))
        else:
            out.append((dyns[i], next((r for a, b, r in segs[i] if a <= j < b), None)))
    return out


class ThinPass(P._Pass):
    """property/v1's pass, with each family step's thin lanes under ``units["thin"]``."""

    def walk(self, body, pdyn, prng, def_id, segs, dyns, up):
        seen = set()
        for k, t in super().walk(body, pdyn, prng, def_id, segs, dyns, up):
            nd = body.nodes[k]
            if k not in seen and not is_input(nd.fn) and isinstance(nd.fn, PrimitiveDefinition) and nd.fn.id in P.FAMILY:
                st = P.FAMILY[nd.fn.id]
                off = 1 if st.acc else 0
                a, b = (lanes(nd.args[off + m].refs, pdyn, prng, segs, dyns) for m in (0, 1))
                if nd.form in ("batch", "scan"):
                    maybe = any(d for d, _ in a) and any(d for d, _ in b) and any(d and two_valued(r) for d, r in a + b)
                    thin = st.k * nd.n if maybe else 0
                else:
                    thin = sum(1 for (da, ra), (db, rb) in zip(a, b) if da and db and (two_valued(ra) or two_valued(rb)))
                t.units["thin"] += thin
            seen.add(k)
            yield k, t


def count(fn: Any, dyn: Sequence[bool], ranges: Sequence[P.Range] | None = None) -> P.Tally:
    """`stacking.count` on `ThinPass`: one call of `fn` whose parameters depend on a run input as `dyn` says, with value
    ranges `ranges` (None: any word)."""
    if isinstance(fn, CompositeDefinition):
        fn = bind(fn)
    rng = tuple(ranges) if ranges is not None else (None,) * len(dyn)
    p = ThinPass()
    out = P.Tally()
    up: dict = {}
    for _k, t in p.walk(fn.body, tuple(dyn), rng, fn.id, [], [], up):
        out.add(t)
    if up:
        out.units["gathers"] += sum(bits for _refs, bits, _kind in up.values())
    return out


def census(fn: Any, dyn: Sequence[bool], ranges: Sequence[P.Range] | None = None) -> dict:
    """One call's budgeted units by class, its thin lanes, family steps, MACs and affine maps, and refused gates."""
    t = count(fn, dyn, ranges)
    return {"units": {c: t.units.get(c, 0) for c in P.BUDGETED}, "thin": t.units.get("thin", 0), "family": t.family,
            "macs": t.macs, "affine": t.affine, "refused": {f"{c}:{f}": n for (c, f), n in t.refused.items()}}


def int8_census(shapes: Sequence[tuple[str, dict, tuple[bool, ...]]] = SHAPES) -> dict[str, dict]:
    """`census` of each exact-int8 Definition at its shape, with property/v1's own class counts beside it (`stacking.count`)."""
    from verity_vllm.program.registry import int8_exact as I8

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
