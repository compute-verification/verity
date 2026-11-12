"""A tensor-parallel collective site's units (`lower`) at a grain C-Flock proves (`interface.check_proof_block`: at most 256
rows a unit, one of them past 111 B). A combine reads its producers' units' rows, gathered first where those are words, and
producers are never re-cut to fit a combine.

* **Coordinate grain.** Where every producer of a rank's partial is a row whose unit is one output word
  (`serving_templates.coordinate`: the embedding shard's column, a block-FP8 linear's GEMM coordinate, a GEMM's slice), each
  rank's words are gathered into rows of K words (``gather``, `b1_tp2.AllGather_v1{WORLD=K, N=1}`, which moves words and
  computes nothing), K = `chunk` of the site's token row. An all-reduce is then one combine unit a chunk reading the W ranks'
  chunk rows: `tp_combine.row_binding`'s unit (for the fused MNNVL call, `AllReduceResidualMnnvlAt_v1`, which reads the
  residual row whole), or for an elementwise all-reduce (``ELEMENTWISE``) the family at K. One `AllGather_v1{WORLD=R/K, N=K}`
  a token row of R words gathers the chunks into the call's row (``output``; none where a row is one chunk), and the fused
  call's normed row is one `tp_allreduce.RmsNormMnnvl_v1` reading that row whole (``norm``).
* **Row grain.** Where every producer is a row whose own Definition is its unit (the MoE's residual add), each row of the
  collective is one combine unit reading that row of every rank: the fused call itself (one token a call), else
  `tp_combine.row_binding`'s unit of the route, or the elementwise family at the row's width.
* **A gather** (``GATHERS``) has no combine unit: its readers read every rank's partial rows.

No combine, output gather or norm Definition has a rank static, so each is one unit that every rank's readers read. A site's
token row is the widest row every read of its output is whole rows of (`row`). Anything else is refused by name: a combine
of no family here, producers of mixed grain, or a partial that is part of a row whose own Definition is its unit."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.commitments.windows import interface as I
from verity.definitions.registry import b1_tp2 as B1T
from verity.definitions.registry import tp_allreduce as TA
from verity.definitions.registry import tp_combine as TC

from verity_vllm.pipeline import serving_templates as ST

__all__ = ["GROUP"]

COORD, ROW = "coordinate", "row"
#: the all-reduces whose every element is the same sum of the ranks' words there, by the static holding the element count
ELEMENTWISE = {B1T.AllReduce2: "N", B1T.AllReduce: "N"}
#: the all-gathers, which move the ranks' words without arithmetic
GATHERS = frozenset({B1T.AllGather2, B1T.AllGather})
#: the fused calls, one token a call, whose chunk units read the residual row whole
FUSED = frozenset({TA.AllReduceRmsNormMnnvl, TA.AllReduceResidualMnnvl})
#: the most words of a gathered row: a gather of K one-word rows and its K-word row fill at most 128 hm96 slots
CHUNK_MAX = 127
#: a token select's group of logits: a word of each `TokenSelectGroups_v1` unit's row of bests
GROUP = 16

#: runs of one Definition over consecutive positions: (Definition, units)
Runs = tuple[tuple[Any, int], ...]


@dataclass(frozen=True)
class Lowering:
    """One site's units (`lower`): its ``grain``; the words of a producer unit's output row (``width``: 1 at coordinate
    grain); the units a rank's partial is (``positions``); per rank its producers' runs ``(Definition, output, units)`` in
    position order (``producers[p]``); the words of a partial row a combine reads (``chunk``: K at coordinate grain, the
    width at row grain), and the Definition gathering a rank's words into them (``gather``, None at row grain); the combine
    units' runs (``combines``, empty for a gather) and the port their parts are (``parts``, cut W ways); the words of a token
    row of the combine's output (``row``), the gather of a row's chunks (``output``, else None) and the fused call's norm
    (``norm``, else None), one unit a token row."""
    grain: str
    width: int
    positions: int
    producers: tuple[tuple[tuple[Any, str, int], ...], ...]
    chunk: int
    gather: Any | None
    combines: Runs
    parts: str
    row: int
    output: Any | None
    norm: Any | None

    @property
    def rows(self) -> int:
        """A rank's partial rows, and the combine units: its words in chunks."""
        return self.positions * self.width // self.chunk

    @property
    def tokens(self) -> int:
        return self.positions * self.width // self.row


def chunk(words: int) -> int:
    """The words of a gathered row of a token row of ``words``: its largest divisor of at most `CHUNK_MAX`."""
    return max(d for d in range(1, min(words, CHUNK_MAX) + 1) if words % d == 0)


def select_width(V: int, w: int, G: int) -> int | None:
    """The logits a token select's group unit reads (`TokenSelectGroups_v1{C, G}`): the least C of whole groups of ``G`` and
    whole ``w``-word rows that divides ``V``, with the group unit and the select over the units' bests and indices
    (`TokenSelectCombine_v2{V, G}`) each within a proof block's rows; None if there is none."""
    step = math.lcm(w, G)
    for C in range(step, V + 1, step):
        if V % C == 0 and C // w + 2 <= I.MAX_ROWS and 2 * (V // C) + 2 <= I.MAX_ROWS:
            return C
    return None


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


#: each combine's units by (grain, its descriptor id, its rows, their words): a model's sites share a few combines
_UNITS: dict[tuple[str, str, int, int], Runs] = {}


def _chunks(fn: Any, n: int, K: int) -> Runs:
    """The combine units of all-reduce ``fn``'s ``n`` chunks of ``K`` words, one a chunk."""
    if fn.defn in ELEMENTWISE:
        return _runs([(fn.defn, {**fn.statics, ELEMENTWISE[fn.defn]: K})] * n)
    return _runs([TC.row_binding(fn.defn, dict(fn.statics), n, c) for c in range(n)])


def _rows(fn: Any, n: int, width: int) -> Runs:
    """The combine units of the ``n`` rows of ``width`` words of all-reduce ``fn``, one a row."""
    if fn.defn in FUSED:
        if n != 1:
            raise ValueError(f"{fn.id}: one token a call, not {n} rows")
        return ((fn, 1),)
    return _chunks(fn, n, width)


def row(combine: Any, words: int, reads: Sequence[tuple[int, int]]) -> int:
    """The token row of a site combined by ``combine`` over a rank's ``words``: the fused call's every word (one token a
    call), else the widest row dividing ``words`` that every read ``[lo, hi)`` of the site's output is whole rows of."""
    if combine.defn in FUSED:
        return words
    return math.gcd(words, *(x for r in reads for x in r))


def lower(combine: Any, own: Sequence[Sequence[tuple[Any, int, int]]], reads: Sequence[tuple[int, int]],
          where: str) -> Lowering:
    """The units of a site combined by ``combine`` (every rank's), each rank's partial ``own[p]`` runs ``(producer
    Definition, lo, hi)``, its output read at ``reads`` (module docstring); refused (ValueError) by name where no grain
    holds."""
    grains = [[(_producer(fn, lo, hi), fn, lo, hi) for fn, lo, hi in runs] for runs in own]
    bad = next(((p, fn, lo, hi) for p, rs in enumerate(grains) for g, fn, lo, hi in rs if g is None), None)
    if bad is not None:
        p, fn, lo, hi = bad
        raise ValueError(f"{where}: rank {p}'s partial reads leaves [{lo}, {hi}) of a {fn.id} node, neither its output words' "
                         "units nor one whole output of it")
    kinds = {g[0] for rs in grains for g, *_ in rs if g is not None}
    if len(kinds) != 1:
        raise ValueError(f"{where}: its producers are of grains {sorted(kinds)}: a combine's grain is its producers' one grain")
    grain = kinds.pop()
    widths = {1} if grain == COORD else {hi - lo for rs in grains for _g, _fn, lo, hi in rs}
    if len(widths) != 1:
        raise ValueError(f"{where}: its producers' rows are {sorted(widths)} words: a row-grain combine reads rows of one width")
    width = widths.pop()
    producers = tuple(tuple((g[1], g[2], g[3]) for g, *_ in rs if g is not None) for rs in grains)
    positions = {sum(n for *_, n in rs) for rs in producers}
    if len(positions) != 1:
        raise ValueError(f"{where}: its ranks' partials are {sorted(positions)} units")
    n = positions.pop()
    words, parts = n * width, combine.signature()[0][0][0]
    R = width if grain == ROW else row(combine, words, reads)
    K = width if grain == ROW else chunk(R)
    gather = bind(B1T.AllGather, WORLD=K, N=1) if grain == COORD else None
    if combine.defn in GATHERS:
        return Lowering(grain, width, n, producers, K, gather, (), parts, R, None, None)
    key = (grain, descriptor_id(combine), words // K, K)
    if key not in _UNITS:
        try:
            _UNITS[key] = _chunks(combine, words // K, K) if grain == COORD else _rows(combine, n, width)
        except ValueError as e:
            raise ValueError(f"{where}: {combine.id} has no {grain} combine units: {e}") from e
    combines = _UNITS[key]
    for fn, _ in combines:
        port = fn.signature()[0][0][0]
        if dict(I.layout(fn).rows)[port] != len(own) * K:
            raise ValueError(f"{where}: {fn.id}'s {port} is not {len(own)} rows of {K} words")
    output = bind(B1T.AllGather, WORLD=R // K, N=K) if grain == COORD and R > K else None
    norm = None
    if combine.defn is TA.AllReduceRmsNormMnnvl and grain == COORD:
        norm = bind(TA.RmsNormMnnvl, **{k: v for k, v in combine.statics.items() if k in TA.RmsNormMnnvl.statics})
    return Lowering(grain, width, n, producers, K, gather, combines, combines[0][0].signature()[0][0][0], R, output, norm)
