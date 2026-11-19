from __future__ import annotations

from typing import Any

from verity.circuits import partition_object as PO
from verity.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.circuits.population import WordBits
from verity.circuits.program import Program
from verity.circuits.refs import Coll
from verity.circuits.types import Array, Value
from verity.protocols.resource_accounting.compute.pouw.prover import schemes as PS
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import pearl_plus_tile as PT


def _word_params(S) -> tuple:
    word = PT.PearlPlusTileCoreV2.sigma_of
    core, _ = bind(word, **{s: getattr(S, s) for s in word.statics}).signature()
    return core[:4] + (("hsalt", PT.HM96_SALT), ("path", Array(S.D, PT.BYTES64))) + core[4:] + (
        ("troot", PT.BYTES64), ("tdomain", PT.BYTES64), ("tpos", Array(S.D, PT.BIT)))


def _on_words(B, S, *xs):
    tile = bind(PT.PearlPlusTile, **{s: getattr(S, s) for s in PT.PearlPlusTile.statics})
    bits = []
    for (_n, t), x, (_m, bt) in zip(_word_params(S), xs, tile.signature()[0]):
        w = t.leaf_widths()[0]
        if w > 1:
            x = Coll(bt, B.batch(bind(WordBits, W=w), Coll(Array(t.leaves, Value(w)), x.refs), axes=(0,)).refs)
        bits.append(x)
    return B.call(tile, *bits)


PearlPlusTileWords = CompositeDefinition(
    "PearlPlusTileWords", 1, PT.PearlPlusTile.statics, lambda S: (_word_params(S), PT.BIT), _on_words, register=False,
    doc="`PearlPlusTile_v1` on its parameters' words: each w-bit leaf, w > 1, its w bits by `WordBits{w}`")


def credited_product(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None,
                     depth: int | None = None) -> SpecializedDefinition:
    unit = PT.tile(PS.get(scheme), k, tm, tn, n, 1 if depth is None else depth, real)
    if depth is not None:
        return bind(PearlPlusTileWords, **unit.bindings)
    return bind(PT.PearlPlusTileCoreV2, **{s: unit.bindings[s] for s in PT.PearlPlusTileCoreV2.statics})


def credited(scheme: str, k: int, tm: int, tn: int, n: int, real: int | None = None,
             depth: int | None = None) -> tuple[Program, dict[str, Any]]:
    core = credited_product(scheme, k, tm, tn, n, real, depth)
    params, ret = core.signature()
    root = CompositeDefinition("PearlPlusCredited", 1, (), lambda S: (params, ret),
                               lambda B, S, *a: B.call(core, *a), register=False)
    program = Program(root, registered=[p for p, _ in params])
    return program, PO.build(program, PO.template_instance_query(core), [(0, 1)])


__all__ = ["PearlPlusTileWords", "credited", "credited_product"]
