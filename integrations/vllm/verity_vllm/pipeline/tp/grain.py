"""A tensor-parallel collective site's units at its producers' grain (`lower`): a combine's grain is its producer unit's output
grain, and producers are never re-cut to fit a combine.

* **Coordinate grain.** Where every producer of a rank's partial is a row whose unit is one output word
  (`serving_templates.coordinate`: the embedding shard's column, a block-FP8 linear's GEMM coordinate, a GEMM's slice), the
  partial is one producer unit a word, and each word of the collective is one combine unit reading that word of every rank:
  `tp_combine.coordinate_binding`'s unit for `tp_allreduce`'s all-reduces, the family at one element for an elementwise
  all-reduce (``ELEMENTWISE``). The fused MNNVL all-reduce and RMSNorm splits in two: a coordinate unit returning the new
  residual r, and one `tp_allreduce.RmsNormMnnvl_v1` a rank reading the N r's (``norm``).
* **Row grain.** Where every producer is a row whose own Definition is its unit (the MoE's residual add), each row of the
  collective is one combine unit reading that row of every rank: the fused call itself (one token a call), else
  `tp_combine.row_binding`'s unit of the route, or the elementwise family at the row's width.
* **A gather** (``GATHERS``) has no combine unit: its readers read the producers' units.

Anything else is refused by name: a combine of no family here, producers of mixed grain, or a partial that is part of a row
whose own Definition is its unit."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity_vllm.pipeline import serving_templates as ST

from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import tp_allreduce as TA
from verity.ml.definitions.registry import tp_combine as TC
from verity.ml.windows import interface as I

COORD, ROW = "coordinate", "row"
#: the all-reduces whose every element is the same sum of the ranks' words there, by the static holding the element count
ELEMENTWISE = {"AllReduce2_v1": "N", "AllReduce_v2": "N"}
#: the all-gathers, which move the ranks' words without arithmetic
GATHERS = frozenset({"AllGather2_v1", "AllGather_v1"})
FUSED = "AllReduceRmsNormMnnvl_v1"

#: runs of one Definition over consecutive positions: (Definition, units)
Runs = tuple[tuple[Any, int], ...]


@dataclass(frozen=True)
class Lowering:
    """One site's units (`lower`): its ``grain``, the words of a producer unit's output row (``width``: 1 at coordinate grain),
    the units a rank's partial is (``positions``); per rank its producers' runs ``(Definition, output, units)`` in position
    order (``producers[p]``); the combine units' runs at every rank (``combines``, empty for a gather) and the port their
    parts are (``parts``, cut W ways); the fused split's norm, one unit a rank (``norm``, else None)."""
    grain: str
    width: int
    positions: int
    producers: tuple[tuple[tuple[Any, str, int], ...], ...]
    combines: Runs
    parts: str
    norm: Any | None


def family(fn: Any) -> str:
    return f"{fn.defn.name}_v{fn.defn.version}"


def _out(fn: Any, f: int) -> tuple[str, int, int]:
    """Output ``o{f}`` of ``fn`` and its leaves ``[lo, hi)`` among the leaves of all its outputs."""
    ret, at = fn.signature()[1], 0
    for g, x in enumerate(getattr(ret, "fields", None) or (ret,)):
        ty = x[1] if isinstance(x, tuple) else x
        if g == f:
            return f"o{f}", at, at + ty.leaves
        at += ty.leaves
    raise ValueError(f"{fn.id} has no output {f}")


def _fields(fn: Any) -> list[tuple[str, int, int]]:
    ret = fn.signature()[1]
    return [_out(fn, f) for f in range(len(getattr(ret, "fields", None) or (ret,)))]


def _producer(fn: Any, lo: int, hi: int) -> tuple[str, Any, str, int] | None:
    """The grain of a partial that is leaves ``[lo, hi)`` of a producer node of Definition ``fn``: (grain, the unit's
    Definition, its output, the units the run is), or None where neither grain holds."""
    c = ST.coordinate(descriptor_id(fn))
    if c is not None:
        cfn, f = c
        _o, a, b = _out(fn, f)
        outs = I.layout(cfn).outs
        if a <= lo and hi <= b and len(outs) == 1 and outs[0][1] == 1:
            return COORD, cfn, outs[0][0], hi - lo
    for out, a, b in _fields(fn):
        if (a, b) == (lo, hi):
            return ROW, fn, out, 1
    return None


def _runs(bindings: Sequence[tuple[Any, dict]]) -> Runs:
    """Consecutive equal bindings as runs of one Definition."""
    out: list[list] = []
    for defn, statics in bindings:
        if out and out[-1][0] == (defn, statics):
            out[-1][1] += 1
        else:
            out.append([(defn, statics), 1])
    return tuple((bind(d, **s), n) for (d, s), n in out)


#: each combine's units by (grain, its descriptor id, positions, width): a model's sites share a few combines
_UNITS: dict[tuple[str, str, int, int], Runs] = {}


def _coordinates(fn: Any, n: int) -> Runs:
    """The combine units of the ``n`` elements of all-reduce ``fn``, one a coordinate."""
    fam = family(fn)
    if fam in ELEMENTWISE:
        return _runs([(fn.defn, {**fn.statics, ELEMENTWISE[fam]: 1})] * n)
    return _runs([TC.coordinate_binding(fn.defn, dict(fn.statics), q) for q in range(n)])


def _rows(fn: Any, n: int, width: int) -> Runs:
    """The combine units of the ``n`` rows of ``width`` words of all-reduce ``fn``, one a row."""
    fam = family(fn)
    if fam == FUSED:
        if n != 1:
            raise ValueError(f"{fn.id}: one token a call, not {n} rows")
        return ((fn, 1),)
    if fam in ELEMENTWISE:
        return _runs([(fn.defn, {**fn.statics, ELEMENTWISE[fam]: width})] * n)
    return _runs([TC.row_binding(fn.defn, dict(fn.statics), n, t) for t in range(n)])


def lower(combine: Any, own: Sequence[Sequence[tuple[Any, int, int]]], where: str) -> Lowering:
    """The units of a site combined by ``combine`` (every rank's), each rank's partial ``own[p]`` runs ``(producer
    Definition, lo, hi)`` (module docstring); refused (ValueError) by name where no grain holds."""
    grains = [[(_producer(fn, lo, hi), fn, lo, hi) for fn, lo, hi in runs] for runs in own]
    bad = next(((p, fn, lo, hi) for p, rs in enumerate(grains) for g, fn, lo, hi in rs if g is None), None)
    if bad is not None:
        p, fn, lo, hi = bad
        raise ValueError(f"{where}: rank {p}'s partial reads leaves [{lo}, {hi}) of a {fn.id} node, neither its output words' "
                         "units nor one whole output of it")
    kinds = {g[0] for rs in grains for g, *_ in rs}
    if len(kinds) != 1:
        raise ValueError(f"{where}: its producers are of grains {sorted(kinds)}: a combine's grain is its producers' one grain")
    grain = kinds.pop()
    widths = {1} if grain == COORD else {hi - lo for rs in grains for _g, _fn, lo, hi in rs}
    if len(widths) != 1:
        raise ValueError(f"{where}: its producers' rows are {sorted(widths)} words: a row-grain combine reads rows of one width")
    width = widths.pop()
    producers = tuple(tuple((g[1], g[2], g[3]) for g, *_ in rs) for rs in grains)
    positions = {sum(n for *_, n in rs) for rs in producers}
    if len(positions) != 1:
        raise ValueError(f"{where}: its ranks' partials are {sorted(positions)} units")
    n, fam = positions.pop(), family(combine)
    parts = combine.signature()[0][0][0]
    if fam in GATHERS:
        return Lowering(grain, width, n, producers, (), parts, None)
    key = (grain, descriptor_id(combine), n, width)
    if key not in _UNITS:
        try:
            _UNITS[key] = _coordinates(combine, n) if grain == COORD else _rows(combine, n, width)
        except ValueError as e:
            raise ValueError(f"{where}: {combine.id} has no {grain} combine units: {e}") from e
    combines = _UNITS[key]
    norm = None
    if fam == FUSED and grain == COORD:
        norm = bind(TA.RmsNormMnnvl, **{k: v for k, v in combine.statics.items() if k in TA.RmsNormMnnvl.statics})
    for fn, _ in combines:
        params = fn.signature()[0]
        if dict(I.layout(fn).rows)[params[0][0]] != len(own) * width:
            raise ValueError(f"{where}: {fn.id}'s {params[0][0]} is not {len(own)} rows of {width} words")
    return Lowering(grain, width, n, producers, combines, combines[0][0].signature()[0][0][0], norm)
