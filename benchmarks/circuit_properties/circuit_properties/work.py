"""The work the min-cut bound charges, read off a Program, and the bound at a policy.

The circuit is a Program in the `verity-ir/descriptor/v1` format, and nothing beside it: the reading takes no source
language, trace, model or optimizer. It is computed once per specialized Definition and context and multiplied through
batches and scans, so it never walks the unrolled circuit. Its rules are the training benchmark's reading of a trace
(`training_slowdown.capture.products`' `_rates`, `_gemm_rate` and `_gathered`, `training_slowdown.work`'s
`he_of_rows`), each resolved on the side of the smaller bound wherever a Program doesn't state what the rule reads.
For `min_cut.bound`'s inputs it reads:

* `|C|` (`Work.gates`, `boolean_gates`): Boolean gates, each primitive at its Boolean version's AND, XOR and NOT
  gates and an input node at none; a primitive with no Boolean version is refused by name.
* What depends on the update, θ′: a scan's carried word that an earlier iteration reached from a fresh input (a run
  input outside `θ`) or from θ′. Every input is free like `θ`, the batch included: it is what makes a carried word
  updated, and nothing more. A multiply by a stated 0 (a constant `INJECTIVE`'s test fails, finite) gives a stated
  0, so an input that reaches a word only through one doesn't reach it. A carry's first iteration is `θ`'s, so a
  Program with no scan has no θ′ and no bound.
* The rate of each array: the bits a block of k bits pays a word of it, at the policy's k. A carried word of θ′ is
  paid at its width; a word θ′ doesn't reach costs nothing; a registered gather (`LOOKUPS`) whose index reads the
  batch and not θ′ gives rows at 0, whatever its table; a product's output is `_gemm_rate`'s (`gemm_rate`); any other
  primitive's is the largest over its arguments of their distinct words' rates over its words (a view keeps its
  argument's rate, a repeated one spreads it, a concatenation sums its parts'); a call's, a batch's and a scan's are
  their callee's, at most that rule's at the node; each at most its width.
* The products: each activation of a Definition in `PRODUCTS` (outermost first, so a `LinearRef`'s own dots are its
  MACs) multiplies its parameter `x` by its parameter `w`, one MAC a word of the larger, with the dims `_gemm_rate`
  reads: its contraction K (the smaller operand's words), its other dims from which operands its batch maps, and its
  batch count. An operand is a weight where it reads nothing of the current iteration's inputs, else an activation
  and the product's token side.
* Which are charged (`Work.groups`; the others are `Work.left_out` by why): a product both of whose operands depend on
  θ′ in every word ("theta" else), neither of which may repeat its rows (`may_repeat`: it carries a gather's rows, or
  reads the current inputs or has rows not shown distinct, and is not specific; "gathered"), whose operands are no
  one array ("self") and not two activations ("activations"). A gather's rows stay a gather's rows through every op
  until the rule below makes them specific; a carry makes them no gather's. Rows are shown distinct
  (`Status.distinct`), for generic weights, where they are `θ`'s, whose rows differ, or where one node computes them
  from an argument whose rows are, read one to one (`distinct_status`): by a shift (`SHIFTS`) or a rounding
  (`ROUNDINGS`) whatever else it reads, as an update W + u does wherever u sums a gradient over the positions, and by
  any other `INJECTIVE` op only where each other argument is reached by no input (`θ`'s own, or stated and passing the
  constant test), is one word that every member of every enclosing batch reads (`Status.fixed`), or whose own rows
  are shown distinct (a weight times a weight). A carry keeps that (`carried`).
  Nothing else shows it: not an input or a gather, a stated value (a carry's initial 0), a multiply by words an input
  reaches that are more than one word or that members of an enclosing batch read apart (W ⊙ z, z ⊙ g, a batch of a
  composite over W and z), a batch of a composite through an argument its members share, or an array that reads a
  word twice or the outputs of more than one node (calls a word, a batch a row). A carried word not shown distinct
  may repeat its rows: the smaller bound. A weight gradient and attention's products are both
  products of two activations, which a Program tells apart only by where their outputs go, which its author chooses.
* Each operand's `β`, as `he_of_rows` derives it from the rate ρ at the operand's width w (its type's, at most the
  source bits its values hold): w where ρ pays it, else w·ρ/(w + ρ). Where an adapter's rank shows (`Work.r_step`,
  the least contraction of a product of two θ′ operands that widens it 4K-fold or more, one of them reading no input,
  on a Program where an operand no θ′ reaches and no input reaches multiplies one θ′ reaches; or of two weights both
  of whose sides are 4K or more), every activation is at most `beta_eff` at it. Every operand is at most
  min(bf16's 16, β(k)): a Program states no code size, and which operands are weights is its author's choice.
* `N` and `O`: the charged products' MACs and their returned leaves times their width. A token-indexed product keeps
  1 − `SHARED_PREFIX`/s of its MACs, s the least sequence a Program's products of two activations allow (the second
  smallest of each one's dims above 1, at least two of which a token indexes where it is attention's); a Program
  with none keeps all of them.
* `F`: `share(ω)`, ω the largest set of arrays the work multiplies pairwise (at least 2).

Each status is per array, never per word: a node's output has one status unless it is a call's or a scan's, whose
status is its callee's returned runs. A value depends on θ′ only where every word read does, carries a gather's rows
where any may, and two arrays are one where any references overlap. A rule's branch that a Program can't decide is
the one with the smaller bound: a product with an activation is token-indexed by every dim its activations span, of
the branches of `_gemm_rate` the least the Program doesn't exclude, from `HEAD_K` no bound, since a Program doesn't
say which products are the head's. So, under the reading's two premises (`SHARED_PREFIX`'s and the updates', below),
every error of the reading is less work or a smaller `β` than the trace's reading of the same
step, and a bound true but weak: a product the tables don't name is not counted, a value with any
word θ′ doesn't reach is `θ`'s, and overlapping references are one array (F may be higher).

A value is specific where, at an input whose rows hold one token at two positions, its words take different values at
the two, for generic weights; a product is charged only on specific or weight operands, so no two MACs it charges
read one pair of values. An operand that reads the current inputs and is not specific may repeat its rows, as two
rows of values may be equal as two tokens may, however the Program writes them: a batch, a call a row, a scan or two
inputs. From the circuit alone, with nothing stated, one rule makes values specific, and every op it can't decide
gives a gathered value:

* A softmax's rows (`masked_rows`): each a row of a product's outputs, through one-to-one ops on constants
  (`INJECTIVE`), plus a constant. A group of rows over one product differs pairwise where each reads another member of
  the product and the constant's rows, as the Program states them, differ: in which words are −inf, or in the finite
  ones after the first's is subtracted, since a softmax is unchanged by a shift. The constant's add alone makes
  nothing specific: rows that differ by a shift give one softmax.
* A softmax's columns: one column of each of one sequence of those rows, each slice another column, differ where the
  constant's columns there differ and each has a finite word, since two keys of one token give columns whose ratio is
  the exponential of the constant's difference.
* A product's output (`product_status`) is specific where its x's slices are such rows or columns, or x is specific
  and read one to one, and w's rows differ: w specific and read one to one, a value of θ′ that reads no current input, or a
  product's output whose columns w's rows read, each another.
* A primitive's output stays specific only where it is one to one in a specific argument read one to one, at the
  constants it reads (`INJECTIVE`'s ops, a multiply's constant nonzero); an op outside it gives a gathered value, so an
  op with a specific argument is not specific by that alone, and none is where another argument is an input's whose
  values the Program doesn't state (a rope table, which may hold a 0).

The reading rests on two premises, each held as a fact by a test where a Program breaks it. The rule rests on the
one `SHARED_PREFIX` takes (a longer shared prefix needs the batch's tokens): past
each row's first position, no two positions of the batch hold equal words. A batch breaks it three ways, each making
two positions' values equal at every layer, under this rule and the trace's reading alike: two rows sharing a prefix
of two tokens or more, a row starting with a run of one token, and, where no positional table reaches the rows, two
positions whose prefixes hold the same tokens in the same proportions. The updates': a weight is `θ`'s rows, which
differ, for generic weights, at every width a product or a rounding (`ROUNDINGS`) reads them, and a word no input
reaches is nonzero; a carried word an input reaches is θ′; and rows one node computes one to one from distinct rows
are distinct, even where the update can't change a weight or cancels it for a reason the constant test doesn't see
(W plus z plus -1 times z is θ′), and a weight cancelled against itself still counts as a weight, its rows distinct
(M plus -1 times M, so the looked-up rows plus such a sum are charged as weights though they repeat with the tokens;
W times v plus -1 times v, v `θ`'s or carried, keeps W's rows), where one word that every member
of every enclosing batch reads multiplies it and is 0 (a learning rate, a decay, an input's or computed from inputs,
taken as nonzero), and where two nodes compute rows one to one from one array's words into two arrays (b added to
each of two rows, carried apart, whose products by W are charged).

The rule doesn't reach a softmax's backward or a table the Program doesn't state, so the input gradients of the
attention's queries and keys (dQ·Wq and dK·Wk, behind a rope table's product or the softmax's backward) stay gathered,
where the trace's reading charges them.

Two limits follow. A Program with no scan has no carried word, so no θ′ and no bound. And a value the rule can't show
specific stays a gather's rows: a model whose first op is a table lookup by its tokens is charged only downstream of a
softmax over a constant that differs between its rows.

`bound` takes the most blocks over the sub-works whose products are at least a threshold β_w·β_x, at least a
contraction length and of one kind or both, each at the least β_w·β_x, the least `b` and the ω of its own products,
through the calculator's `min_cut`.
"""

from __future__ import annotations

import bisect
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import NamedTuple, Sequence

from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.program import is_input, is_registered
from verity.circuits.refs import Affine, Concat, Strided

from .ops import ids


@dataclass(frozen=True)
class Product:
    """Activations of a Definition matching ``op`` are matrix products of parameter ``x`` by parameter ``w``: one MAC a
    word of the larger, each output a returned leaf."""

    op: str
    x: str
    w: str

    @cached_property
    def rx(self) -> re.Pattern:
        return re.compile(self.op)


@dataclass(frozen=True)
class Lookup:
    """Activations of a primitive matching ``op`` read a word of parameter ``table`` at parameter ``index``."""

    op: str
    index: str
    table: str

    @cached_property
    def rx(self) -> re.Pattern:
        return re.compile(self.op)


#: the reference registry's matrix products (`verity.definitions.registry.ref_prims`): a row times a matrix, and a dot
PRODUCTS = (Product(r"LinearRef_v\d+", "x", "w"), Product(r"DotRef_v\d+", "x", "w"))
#: the registered gathers: a column of a table at a token
LOOKUPS = (Lookup(r"Gather\w+x\d+_v\d+", "token", "column"),)
#: `training_slowdown.capture.products`' and `training_slowdown.work`'s constants, which its suite holds equal:
#: the partial sums' width, the share of a rate that pays a width, the positions a sequence's start shares, and the k
#: from which the head's products leave M
ACC_BITS, PAID, SHARED_PREFIX, HEAD_K = 32, 0.999, 1, 8e7


class Status(NamedTuple):
    """What an array's words depend on."""

    #: every word depends on θ′
    aff: bool
    #: some word may
    dep: bool
    #: some word may depend on a fresh input or on θ′
    reach: bool
    #: some word may carry a gather's rows
    gathered: bool
    #: the scan levels (0: outside every scan) whose current inputs some word may read
    data: frozenset
    #: the bits a block pays a word on average (its rate), at most its width
    rate: float = 0.0
    #: the source bits a word holds on average: a fresh input's and a carried θ′ word's width, θ's 0, and a computed
    #: array's the bits of the sources it reads over its words, at most its width
    bits: float = 0.0
    #: every word is specific to its position (the module docstring's rule), so carries no gather's rows
    specific: bool = False
    #: every word is a view of a constant the Program states
    stated: bool = False
    #: no two of its rows can be equal, for generic weights: `θ`'s, and those one node computes one to one from such
    #: rows (`distinct_status`), which a carry keeps (`carried`)
    distinct: bool = False
    #: every word is one value in every member of every batch enclosing the Definition that holds it: shown for an
    #: input's words and those computed from such words alone, never for a parameter a batch maps
    fixed: bool = False


THETA = Status(False, False, False, False, frozenset(), distinct=True)
STATED = THETA._replace(stated=True, distinct=False)


def unknown(st: Status) -> bool:
    """Whether an array's values are the current inputs', which the Program neither states nor computes from θ."""
    return st.reach and not (st.dep or st.gathered or st.specific or st.stated)


def may_repeat(st: Status) -> bool:
    """Whether two rows of an array may hold one word: it carries a gather's rows, or reads the current inputs or its
    rows aren't shown distinct, and it is not specific, however the Program writes its rows (a batch, a call a row, a
    scan, two inputs)."""
    return st.gathered or ((bool(st.data) or not st.distinct) and not st.specific)


def unspecific(st: Status, sts) -> Status:
    """``st`` where a rule couldn't keep its words specific: its gather's rows those of every argument's that carries
    them or is specific."""
    sts = list(sts)
    return st._replace(specific=False, gathered=any(s.gathered or s.specific for s in sts))


NEG_INF = 0xFF800000


def _finite(b: int) -> bool:
    return (b >> 23) & 0xFF != 0xFF


def _float(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def _const(f) -> int:
    m = re.fullmatch(r"Const\d+\[0x([0-9a-fA-F]+)\]", f.name)
    if m is None:
        raise ValueError(f"{f.id}: a primitive of no arguments that is no constant")
    return int(m.group(1), 16)


def _shape(bits) -> tuple | None:
    """A softmax row's constants up to what a softmax ignores: which are -inf, and the finite ones less the first;
    None where none is finite or one is NaN or +inf."""
    bits = list(bits)
    if any(b != NEG_INF and not _finite(b) for b in bits):
        return None
    fin = [_float(b) for b in bits if b != NEG_INF]
    if not fin:
        return None
    return tuple(b == NEG_INF for b in bits), tuple(v - fin[0] for v in fin)


def _generic(name: str, vs: list, f):
    """A primitive's word from its arguments' (`Reader.summary`'s ``value``): a finite data word "g" added to a
    constant is the constant where it is infinite and data otherwise, compared with one is unequal, and a select or an
    and of constants decides."""
    if any(v is None for v in vs):
        return None
    if all(isinstance(v, tuple) for v in vs):
        return ("c", f.evaluate(*(v[1] for v in vs)))
    c = [v[1] for v in vs if isinstance(v, tuple)]
    if name == "F32Add" and len(vs) == 2 and c:
        return ("c", c[0]) if not _finite(c[0]) else "g"
    if name == "F32Eq" and len(vs) == 2 and c:
        return ("c", 0)
    if name == "BitAnd" and ("c", 0) in vs:
        return ("c", 0)
    if name == "SelectF32" and isinstance(vs[0], tuple):
        return vs[1] if vs[0][1] else vs[2]
    return None


def _nonzero(b: int) -> bool:
    return _finite(b) and bool(b & 0x7FFFFFFF)


#: the primitives that keep a specific word's outputs distinct, each with what a constant it reads must be
INJECTIVE = {"F32Add": _finite, "F32Sub": _finite, "F32Mul": _nonzero, "F32Div": _nonzero, "F32Fma": _nonzero,
             "F32Neg": _finite, "F32ExpRn": _finite, "F32Sqrt": _finite, "Bf16ToF32": _finite}
#: those of them one to one in each argument whatever finite words the others hold
SHIFTS = {"F32Add", "F32Sub", "F32Neg", "F32ExpRn", "F32Sqrt", "Bf16ToF32"}
#: the registry's roundings of a float to a narrower float, which keep rows distinct (`Status.distinct`) by the
#: generic-weights premise and nothing else
ROUNDINGS = {"F32ToBf16Rn", "F32ToF16Rn", "F32ToE4m3Sat", "F32ToE2m1Sat"}
#: the reference registry's softmax over a row
SOFTMAXES = (re.compile(r"SoftmaxRef_v\d+"),)


def join(sts, weights=None) -> Status:
    """The status of an array made of arrays of statuses ``sts``, of ``weights`` leaves each (None: its rate and bits
    the most of theirs). It carries a gather's rows where any part may, and is specific only where no part carries
    them."""
    sts = list(sts)
    if not sts:
        return THETA
    if weights is None:
        rate, bits = max(s.rate for s in sts), max(s.bits for s in sts)
    else:
        weights = list(weights)
        total = max(1, sum(weights))
        rate = sum(w * s.rate for w, s in zip(weights, sts)) / total
        bits = sum(w * s.bits for w, s in zip(weights, sts)) / total
    gathered = any(s.gathered for s in sts)
    return Status(all(s.aff for s in sts), any(s.dep for s in sts), any(s.reach for s in sts), gathered,
                  frozenset().union(*(s.data for s in sts)), rate, bits, any(s.specific for s in sts) and not gathered,
                  all(s.stated for s in sts), all(s.distinct for s in sts), all(s.fixed for s in sts))


def computed(sts, rate: float, bits: float) -> Status:
    """The status of an array each of whose words is computed from words of each of arrays ``sts``, at ``rate`` and
    ``bits`` a word; 0 a word where none depends on θ′. A word computed from a specific one is specific."""
    sts = list(sts)
    dep = any(s.dep for s in sts)
    specific = any(s.specific for s in sts)
    return Status(any(s.aff for s in sts), dep, any(s.reach for s in sts),
                  any(s.gathered for s in sts) and not specific, frozenset().union(*(s.data for s in sts)),
                  rate if dep else 0.0, bits, specific, fixed=all(s.fixed for s in sts))


#: an array's status by leaf: runs (end, status), ends rising, the last the array's leaves
Profile = tuple


def runs_of(parts: list[tuple[int, Status]]) -> Profile:
    out: list[tuple[int, Status]] = []
    for end, s in parts:
        if out and out[-1][1] == s:
            out[-1] = (end, s)
        elif not out or end > out[-1][0]:
            out.append((end, s))
    return tuple(out)


def span(p: Profile, lo: int, hi: int) -> Status:
    """``p``'s status over leaves ``lo`` to ``hi``."""
    out, weights = _over(p, lo, hi)
    return join(out, weights)


def _over(p: Profile, lo: int, hi: int) -> tuple[list[Status], list[int]]:
    """The runs of ``p`` over leaves ``lo`` to ``hi``, and how many of those leaves each holds."""
    ends = [e for e, _ in p]
    i = bisect.bisect_right(ends, lo)
    out, weights = [], []
    while i < len(p):
        start = p[i - 1][0] if i else 0
        out.append(p[i][1])
        weights.append(min(p[i][0], hi) - max(start, lo))
        if p[i][0] >= hi:
            break
        i += 1
    return out, weights


def total_in(p: Profile, lo: int, hi: int) -> tuple[float, float]:
    """The rates and the source bits ``p``'s leaves ``lo`` to ``hi`` hold between them."""
    out, weights = _over(p, lo, hi)
    return sum(w * s.rate for w, s in zip(weights, out)), sum(w * s.bits for w, s in zip(weights, out))


def whole(p: Profile) -> Status:
    return join((s for _, s in p), (e - b for (e, _), b in zip(p, (0, *(e for e, _ in p)))))


def drop(p: Profile, levels: frozenset) -> Profile:
    return runs_of([(e, s._replace(data=s.data - levels)) for e, s in p])


def capped(p: Profile, rate: float, bits: float) -> Profile:
    return runs_of([(e, s._replace(rate=min(s.rate, rate), bits=min(s.bits, bits))) for e, s in p])


def carried(p: Profile, t) -> Profile:
    """A carry ``p`` of type ``t`` as an iteration's input: a word reached from a fresh input or θ′ is θ′, paid at
    and holding its width, any other is free and holds nothing. A carried word reads none of the iteration's inputs,
    so it is no token's and carries no gather's rows; rows shown distinct (`Status.distinct`) stay so."""
    cuts = sorted({e for e, _ in p} | {e for e, _ in _width_runs(t)})
    widths = _width_runs(t)
    out, i, j = [], 0, 0
    for e in cuts:
        while p[i][0] < e:
            i += 1
        while widths[j][0] < e:
            j += 1
        s, w = p[i][1], float(widths[j][1])
        s = s._replace(gathered=False, specific=False, stated=False)
        out.append((e, s._replace(aff=True, dep=True, rate=w, bits=w) if s.reach else
                    s._replace(aff=False, dep=False, rate=0.0, bits=0.0)))
    return runs_of(out)


def first(p: Profile) -> Profile:
    """A carry's initial value as its first iteration's input: `θ`'s, whatever it is computed from, its rows distinct
    only where the initial value's are (a stated 0 isn't)."""
    return runs_of([(e, THETA._replace(distinct=s.distinct)) for e, s in p])


def _fields(t, at: int = 0) -> list[tuple[int, int]]:
    """A type's arrays as leaf ranges (lo, hi): a tuple's fields', recursively, any other type whole."""
    if not hasattr(t, "fields"):
        return [(at, at + t.leaves)]
    out = []
    for f in t.fields:
        out += _fields(f, at)
        at += f.leaves
    return out


def returned(parts, t, source, words: list) -> Profile:
    """References of type ``t`` as runs, each part (`Reader.parts`) at its source's status over its leaves; an array of
    ``t`` that reads a word twice (of ``words``, the references' words), or whose parts come from more than one node
    or parameter, has no rows shown distinct, since two of them may then be one array's words, or computed one to one
    from them."""
    parts = list(parts)
    fields = _fields(t)
    starts = [lo for lo, _hi in fields]

    def over(a: int, b: int) -> range:
        return range(max(bisect.bisect_right(starts, a) - 1, 0), bisect.bisect_left(starts, b))

    sources: list[set] = [set() for _ in fields]
    for a, b, s, i, _lo, _hi in parts:
        for j in over(a, b):
            sources[j].add((s, i))
    for j, (lo, hi) in enumerate(fields):
        if len(set(words[lo:hi])) < hi - lo:
            sources[j].add(None)
    out = []
    for a, b, s, i, lo, hi in parts:
        for j in over(a, b):
            x, y = max(a, fields[j][0]), min(b, fields[j][1])
            st = span(source(s, i), lo + x - a, lo + y - a) if hi - lo == b - a else span(source(s, i), lo, hi)
            out.append((y, st._replace(distinct=False) if st.distinct and len(sources[j]) > 1 else st))
    return runs_of(out)


def _width_runs(t) -> list[tuple[int, int]]:
    """A type's leaf widths as runs (end, width)."""
    w = t.homogeneous_width()
    if w is not None or not t.leaves:
        return [(t.leaves, w or 0)]
    out: list[tuple[int, int]] = []
    for w in t.leaf_widths():
        if out and out[-1][1] == w:
            out[-1] = (out[-1][0] + 1, w)
        else:
            out.append(((out[-1][0] if out else 0) + 1, w))
    return out


def _parts(refs, memo: dict) -> list[tuple[int, int, str, int, int, int]]:
    """The references ``refs`` as runs (a, b, space, idx, lo, hi): positions a to b of the sequence read leaves inside
    lo to hi of (space, idx). A strided run is its hull, and a prefix shared through a `PartLog` one run a source over
    the whole sequence, computed once a log."""
    out: list = []

    def walk(r, at: int) -> None:
        if isinstance(r, Affine):
            if r.count:
                out.append((at, at + r.count, r.space, r.idx, r.base, r.base + r.count))
        elif isinstance(r, Strided):
            if not r.count:
                return
            rows = -(-r.count // r.inner)
            corners = [r.base + q * r.outer_stride + j * r.inner_stride for q in (0, rows - 1)
                       for j in (0, min(r.inner, r.count) - 1)]
            out.append((at, at + r.count, r.space, r.idx, min(corners), max(corners) + 1))
        elif isinstance(r, Concat) and r.prefix is not None:
            log, j = r.prefix
            for (space, idx), (lo, hi) in _hulls(log, j, memo).items():
                out.append((at, at + r.count, space, idx, lo, hi))
            pos = at + log.starts[j]
            for p in r.parts[j:]:
                walk(p, pos)
                pos += p.count
        elif isinstance(r, Concat):
            pos = at
            for p in r.parts:
                walk(p, pos)
                pos += p.count
        else:
            for j, (space, idx, leaf) in enumerate(r):
                if out and out[-1][1] == at + j and out[-1][2:4] == (space, idx) and out[-1][5] == leaf:
                    out[-1] = (out[-1][0], at + j + 1, space, idx, out[-1][4], leaf + 1)
                else:
                    out.append((at + j, at + j + 1, space, idx, leaf, leaf + 1))

    walk(refs, 0)
    return out


def _leaf_runs(refs, limit: int = 1 << 15) -> tuple[list, list]:
    """The leaves ``refs`` read, each once, as runs (space, idx, lo, hi); a two-level strided view whose leaves take
    more than ``limit`` runs as its hull (space, idx, lo, hi, share), ``share`` the most of the hull its leaves are."""
    exact: list = []
    hulls: list = []

    def grid(space, idx, base: int, a: int, sa: int, b: int, sb: int) -> None:
        """Leaves base + i·sa + j·sb, i < a, j < b."""
        if a <= 0 or b <= 0:
            return
        a, sa = (1, 1) if sa == 0 else (a, sa)
        b, sb = (1, 1) if sb == 0 else (b, sb)
        if sa == 1 and sb != 1:
            a, sa, b, sb = b, sb, a, sa
        if sb == 1 and a <= limit:
            exact.extend((space, idx, base + i * sa, base + i * sa + b) for i in range(a))
        elif sb != 1 and a * b <= limit:
            exact.extend((space, idx, base + i * sa + j * sb, base + i * sa + j * sb + 1) for i in range(a)
                         for j in range(b))
        else:
            ends = [base + i * sa + j * sb for i in (0, a - 1) for j in (0, b - 1)]
            lo, hi = min(ends), max(ends) + 1
            hulls.append((space, idx, lo, hi, min(1.0, a * b / (hi - lo))))

    def walk(r) -> None:
        if isinstance(r, Affine):
            if r.count:
                exact.append((r.space, r.idx, r.base, r.base + r.count))
        elif isinstance(r, Strided):
            rows, rem = divmod(r.count, r.inner)
            grid(r.space, r.idx, r.base, rows, r.outer_stride, r.inner, r.inner_stride)
            grid(r.space, r.idx, r.base + rows * r.outer_stride, 1, 1, rem, r.inner_stride)
        elif isinstance(r, Concat):
            if r.prefix is not None:
                log, j = r.prefix
                for p in log.parts[:j]:
                    walk(p)
                parts = r.parts[j:]
            else:
                parts = r.parts
            for p in parts:
                walk(p)
        else:
            for space, idx, leaf in r:
                if exact and exact[-1][:2] == (space, idx) and exact[-1][3] == leaf:
                    exact[-1] = (space, idx, exact[-1][2], leaf + 1)
                else:
                    exact.append((space, idx, leaf, leaf + 1))

    walk(refs)
    return exact, hulls


def _hulls(log, j: int, memo: dict) -> dict:
    """The hull each source's leaves span in a `PartLog`'s first ``j`` parts, extended part by part."""
    hs = memo.setdefault(id(log), (log, [{}]))[1]
    while len(hs) <= j:
        h = dict(hs[-1])
        for _a, _b, space, idx, lo, hi in _parts(log.parts[len(hs) - 1], memo):
            old = h.get((space, idx))
            h[(space, idx)] = (lo, hi) if old is None else (min(old[0], lo), max(old[1], hi))
        hs.append(h)
    return hs[j]


def paid(rate: float, width: float) -> bool:
    """Whether a rate pays its operand's width (`training_slowdown.work.is_paid`)."""
    return rate >= PAID * width


def k_star(k: float, w: float) -> float:
    """K* = ½√(k/w), the least contraction whose output a block can't make below its width
    (`training_slowdown.work.k_star`)."""
    return 0.5 * math.sqrt(k / w)


def beta_eff(w: float, r: float, k: float) -> float:
    """An operand's β where the update reaches it through a rank-r bottleneck (`training_slowdown.work.beta_eff`)."""
    return w / (1 + (k_star(k, w) / r) ** 2) if r > 0 else 0.0


@dataclass
class Site:
    """One product node of one Definition's body in one context."""

    product: Product
    node: int
    #: activations a node: a batch's members
    n: int
    macs: int
    outputs: int
    out_width: int
    k_dim: int
    widths: tuple[int, int]
    x: Status
    w: Status
    #: each operand's source runs (space, idx, lo, hi) in the body
    sources: tuple[tuple, tuple]
    #: (batch, m, k, n), x spanning batch, m and k and w batch, k and n
    dims: tuple[int, int, int, int]
    #: each operand's rate and source bits, its words' average
    rates: tuple[float, float]
    bits: tuple[float, float]

    def passed(self) -> tuple[float, float]:
        """Each operand's width at most the source bits its values hold (`_passed_width`)."""
        return tuple(min(w, b) for w, b in zip(self.widths, self.bits))


def gemm_rate(site: Site, k: float) -> float:
    """A product's output rate at k (`training_slowdown.capture.products._gemm_rate`), its token side the operand that
    reads the current iteration's inputs: where only one does, its dims are the ones it spans that the other doesn't; where
    neither, no dim is a token's and the output is paid at its width; where both, the least of attention's form and each
    single token side's, since the Program doesn't say which dims are a token's."""
    batch, m, K, n = site.dims
    x, w = site.x, site.w
    ra, rb = site.rates
    ba, bb = site.passed()
    if not (x.dep or w.dep):
        return 0.0
    if not (x.data or w.data):
        return math.inf

    def one(token_x: bool) -> float:
        r_tok, b_tok, b_par, par, spread = (ra, ba, bb, w.dep, n) if token_x else (rb, bb, ba, x.dep, m)
        if par and (paid(r_tok, b_tok) or K >= 3 ** 1.5 * math.sqrt(k / ACC_BITS)):
            return 4 * K ** 2 * b_tok * b_par / k
        return r_tok * K / max(spread, 1)

    if x.data and w.data:
        return min(ra * K / max(n, 1) + rb * K / max(m, 1), one(True), one(False))
    return one(bool(x.data))


@dataclass
class Summary:
    sid: int
    fn_id: str
    ret: Profile
    sites: list[Site] = field(default_factory=list)
    #: (callee summary, activations a parent activation, bindings (callee param, a, b, space, idx, lo, hi))
    children: list[tuple[Summary, int, list]] = field(default_factory=list)
    #: beside each child: the body node it is a call of and the members (a scan's iterations) it stands for
    at: list[tuple[int, Sequence[int]]] = field(default_factory=list)


def _width(t) -> int:
    w = t.homogeneous_width()
    if w is None:
        raise ValueError(f"a product's operand of mixed widths ({sorted(t.width_set())})")
    return w


def _unit(t) -> int:
    """A type's words: its leaves (a word program's leaf is one word)."""
    return t.leaves


class Reader:
    def __init__(self, program, k: float, products=PRODUCTS, lookups=LOOKUPS) -> None:
        self.program, self.k, self.products, self.lookups = program, k, products, lookups
        self.memo: dict = {}
        self.logs: dict = {}
        self._match: dict = {}
        self._leaves: dict = {}

    def product(self, fn) -> Product | None:
        return self._of(fn, self.products)

    def lookup(self, fn) -> Lookup | None:
        return self._of(fn, self.lookups)

    def _of(self, fn, table):
        key = (id(table), fn.id)
        if key not in self._match:
            self._match[key] = next((p for p in table if any(p.rx.fullmatch(i) for i in ids(fn))), None)
        return self._match[key]

    def parts(self, refs):
        return _parts(refs, self.logs)

    def leaves(self, refs):
        key = id(refs)
        if key not in self._leaves:
            self._leaves[key] = (refs, _leaf_runs(refs))
        return self._leaves[key][1]

    def root(self) -> Summary:
        root = self.program.root
        return self.summary(root, (), 0, theta=frozenset(root.theta))

    def summary(self, fn, ctx: tuple, depth: int, theta: frozenset = frozenset()) -> Summary:
        key = (fn.id, ctx, depth)
        hit = self.memo.get(key)
        if hit is not None:
            return hit
        body = fn.body
        S = Summary(-1, fn.id, ())
        prof: list[Profile] = []

        def source(space: str, idx: int) -> Profile:
            return ctx[idx] if space == "p" else prof[idx]

        def arg_profile(a) -> Profile:
            return returned(self.parts(a.refs), a.type, source, words(a)) or ((a.refs.count, THETA),)

        def arg_status(a) -> Status:
            """``a``'s status as one array: its rows distinct only where one node or parameter gives them all and no
            word is read twice."""
            st = join(span(source(s, i), lo, hi) for _a, _b, s, i, lo, hi in self.parts(a.refs))
            if st.distinct and (len({(s, i) for _a, _b, s, i, _lo, _hi in self.parts(a.refs)}) > 1 or
                                not one_to_one(a)):
                return st._replace(distinct=False)
            return st

        def bindings(i: int, a, mapped: bool) -> list:
            ps = self.parts(a.refs)
            if mapped:
                return [(i, 0, -1, s, j, lo, hi) for _a, _b, s, j, lo, hi in ps]
            return [(i, a_, b_, s, j, lo, hi) for a_, b_, s, j, lo, hi in ps]

        def read(args) -> tuple[float, float]:
            """The rates and source bits ``args`` read between them, each source leaf once."""
            spans: dict = {}
            rate = bits = 0.0
            for a in args:
                exact, hulls = self.leaves(a.refs)
                for s, i, lo, hi in exact:
                    spans.setdefault((s, i), []).append((lo, hi))
                for s, i, lo, hi, share in hulls:
                    r, b_ = total_in(source(s, i), lo, hi)
                    rate, bits = rate + share * r, bits + share * b_
            for (s, i), rs in spans.items():
                rs.sort()
                lo, hi = rs[0]
                for a, b in [*rs[1:], (math.inf, math.inf)]:
                    if a > hi:
                        r, b_ = total_in(source(s, i), lo, hi)
                        rate, bits = rate + r, bits + b_
                        lo, hi = a, b
                    else:
                        hi = max(hi, b)
            return rate, bits

        def made(args, out) -> tuple[float, float]:
            """A word of ``out``'s rate, the most any one argument's distinct words pay over ``out``'s words, and its
            source bits, those all of them hold; each at most its width."""
            if not out.leaves:
                return 0.0, 0.0
            w = max(out.width_set(), default=0)
            rate = max((read([a])[0] for a in args), default=0.0) / out.leaves
            return min(w, rate), min(w, read(args)[1] / out.leaves)

        def operand(a) -> tuple[float, float]:
            rate, bits = read([a])
            n = a.refs.count
            return (rate / n, bits / n) if n else (0.0, 0.0)

        words_of: dict = {}

        def words(a) -> list:
            if id(a) not in words_of:
                words_of[id(a)] = (a, list(a.refs))
            return words_of[id(a)][1]

        def one_to_one(a) -> bool:
            w = words(a)
            return len(set(w)) == len(w)

        def member_words(nd, i: int, m: int) -> list:
            """The words member ``m`` of node ``nd`` reads of its argument ``i``."""
            w, per = words(nd.args[i]), nd.fn.signature()[0][i][1].leaves
            return w[m * per:(m + 1) * per] if nd.form == "batch" and nd.axes[i] is not None else w

        value_of: dict = {}

        def value(space: str, j: int, leaf: int):
            """A word as a generic f32 value would leave it: ("c", bits) where it is that constant whatever the data,
            "g" where it is data the constants don't decide (a finite value), None where the rules here can't say."""
            if space != "n":
                return None if ctx[j] and any(s.stated for _e, s in ctx[j]) else "g"
            if (j, leaf) in value_of:
                return value_of[j, leaf]
            nd = body.nodes[j]
            f = nd.fn
            out: object = "g"
            if not isinstance(f, PrimitiveDefinition) or is_input(f):
                out = "g"
            elif not nd.args:
                out = ("c", _const(f))
            else:
                m = leaf if nd.form == "batch" else 0
                vs = [value(*member_words(nd, i, m)[0]) for i in range(len(nd.args))]
                out = _generic(f.name, vs, f)
            value_of[j, leaf] = out
            return out

        def unselected(space: str, j: int, leaf: int):
            """The word a select gives where its condition is false whatever the data, else the word itself."""
            nd = body.nodes[j] if space == "n" else None
            if nd is None or not isinstance(nd.fn, PrimitiveDefinition) or nd.fn.name != "SelectF32":
                return space, j, leaf
            m = leaf if nd.form == "batch" else 0
            c, b = member_words(nd, 0, m)[0], member_words(nd, 2, m)[0]
            return unselected(*b) if value(*c) == ("c", 0) else None

        def scaled(space: str, j: int, leaf: int):
            """The word a chain of `INJECTIVE` ops, each by constants alone, reads of data, else the word itself."""
            nd = body.nodes[j] if space == "n" else None
            if nd is None or not isinstance(nd.fn, PrimitiveDefinition) or nd.fn.name not in INJECTIVE or not nd.args:
                return space, j, leaf
            m = leaf if nd.form == "batch" else 0
            ws = [member_words(nd, i, m)[0] for i in range(len(nd.args))]
            vs = [value(*w) for w in ws]
            data = [w for w, v in zip(ws, vs) if not isinstance(v, tuple)]
            if len(data) != 1 or not all(INJECTIVE[nd.fn.name](v[1]) for v in vs if isinstance(v, tuple)):
                return space, j, leaf
            return scaled(*data[0])

        rows_of: list = []

        def softmax(nd) -> bool:
            return nd.form in ("call", "batch") and any(p.fullmatch(i) for p in SOFTMAXES for i in ids(nd.fn))

        def masked_rows() -> frozenset:
            """The rows (node, member) of the body's softmaxes that differ from every other row of theirs: each reads,
            word for word, an f32 add of one member of a product (its row; through `scaled`) and a constant, every two
            rows of a product with constants that differ in which words are -inf or by more than a shift, each with a
            finite one."""
            if rows_of:
                return rows_of[0]
            groups: dict = {}
            info: dict = {}
            for j, nd in enumerate(body.nodes):
                if not softmax(nd) or len(nd.args) != 1 or (nd.form == "batch" and nd.axes[0] is None):
                    continue
                for r in range(nd.n if nd.form == "batch" else 1):
                    row = row_of(member_words(nd, 0, r))
                    if row is None:
                        continue
                    prod, at, consts = row
                    info[j, r] = (prod, consts)
                    groups.setdefault(prod, []).append(((j, r), at, _shape(consts)))
            ok = set()
            for rows in groups.values():
                if len({at for _r, at, _p in rows}) == len(rows) == len({p for _r, _a, p in rows}):
                    ok |= {r for r, _a, _p in rows}
            rows_of.extend((frozenset(ok), info))
            return rows_of[0]

        def row_of(ws: list):
            """(the product, its leaves, the constants' `_shape`) a softmax row reads as an f32 add of the two, or
            None."""
            if not ws or {s_ for s_, _j, _l in ws} != {"n"} or len({j_ for _s, j_, _l in ws}) != 1:
                return None
            e = body.nodes[ws[0][1]]
            if not isinstance(e.fn, PrimitiveDefinition) or e.fn.name != "F32Add" or e.form != "batch" or \
                    len(e.args) != 2 or e.axes != (0, 0):
                return None
            got = []
            for _s, _j, leaf in ws:
                a, b = (member_words(e, i, leaf)[0] for i in (0, 1))
                va, vb = value(*a), value(*b)
                if isinstance(va, tuple) == isinstance(vb, tuple):
                    return None
                d, c = (b, va) if isinstance(va, tuple) else (a, vb)
                got.append((scaled(*d), c[1]))
            data = [d for d, _c in got]
            if {s_ for s_, _j, _l in data} != {"n"} or len({j_ for _s, j_, _l in data}) != 1 or \
                    self.product(body.nodes[data[0][1]].fn) is None:
                return None
            at = tuple(leaf for _s, _j, leaf in data)
            if _shape(c for _d, c in got) is None or at != tuple(range(at[0], at[0] + len(at))) or at[0] % len(at):
                return None
            return data[0][1], at, tuple(c for _d, c in got)

        def softmax_rows(a, K: int, n: int) -> bool:
            """Whether argument ``a``'s ``n`` slices of ``K`` words differ pairwise as a softmax's rows or columns that
            the mask makes differ, through selects whose conditions are false: each slice a row `masked_rows` finds
            differs, every slice another; or each slice one column of one sequence of a product's rows, every slice
            another column and the mask's columns there pairwise different, each with a finite word."""
            if not words(a) or len(words(a)) != n * K or not one_to_one(a):
                return False
            got = [unselected(*w) for w in words(a)]
            if None in got or {s_ for s_, _j, _l in got} != {"n"}:
                return False
            width = {}
            for _s, j, _l in got:
                sm = body.nodes[j]
                if not softmax(sm):
                    return False
                width[j] = sm.out.leaves // (sm.n if sm.form == "batch" else 1)
            cells = [((j, leaf // width[j]), leaf % width[j]) for _s, j, leaf in got]
            slices = [cells[m * K:(m + 1) * K] for m in range(n)]
            ok, info = masked_rows(), rows_of[1]
            if all(len({r for r, _c in sl}) == 1 and [c for _r, c in sl] == list(range(K)) for sl in slices):
                rows = {sl[0][0] for sl in slices}
                return len(rows) == n and rows <= ok
            seq = [r for r, _c in slices[0]]
            if len(set(seq)) != K or any(r not in info for r in seq) or len({info[r][0] for r in seq}) != 1:
                return False
            cols = []
            for sl in slices:
                if [r for r, _c in sl] != seq or len({c for _r, c in sl}) != 1:
                    return False
                c = sl[0][1]
                col = tuple(info[r][1][c] for r in seq)
                if not any(_finite(b) for b in col) or any(b != NEG_INF and not _finite(b) for b in col):
                    return False
                cols.append((c, col))
            return len({c for c, _col in cols}) == n == len({col for _c, col in cols})

        def distinct_rows(nd, i: int, K: int) -> bool:
            """Whether argument ``i`` of product ``nd``'s slices of ``K`` words differ pairwise wherever its words
            could: it is specific and read one to one, or its rows can't repeat (`may_repeat`) and it depends on θ, or it is a
            view, one row of words a column, of a product of a gather's rows by a weight."""
            a = nd.args[i]
            st = arg_status(a)
            if st.specific:
                return one_to_one(a)
            if not may_repeat(st):
                return not st.stated and not unknown(st)
            w = words(a)
            if not w or len(w) % K or {s_ for s_, _j, _l in w} != {"n"} or len({j_ for _s, j_, _l in w}) != 1:
                return False
            src = body.nodes[w[0][1]]
            p = self.product(src.fn)
            if p is None or src.form != "batch":
                return False
            names = [n_ for n_, _t in src.fn.signature()[0]]
            if arg_status(src.args[names.index(p.w)]).data:
                return False
            n_out = src.out.leaves // src.n
            cols = [leaf % n_out for _s, _j, leaf in w[::K]]
            return len(set(cols)) == len(cols)

        def product_status(nd, ix: int, iw: int, base: Status) -> Status:
            """A product's output is specific where its rows by members and its columns both differ."""
            params = nd.fn.signature()[0]
            K = min(_unit(params[ix][1]), _unit(params[iw][1]))
            mapped = nd.form == "batch" and nd.axes[ix] is not None
            n = nd.n if mapped else 1
            rows = softmax_rows(nd.args[ix], K, n) or (arg_status(nd.args[ix]).specific and one_to_one(nd.args[ix]))
            cols = not (nd.form == "batch" and nd.axes[iw] is not None) and distinct_rows(nd, iw, K)
            if rows and cols:
                return base._replace(specific=True, gathered=False)
            return unspecific(base, [arg_status(a) for a in nd.args])

        def primitive_status(f, nd, sts: list, base: Status) -> Status:
            """A primitive's output is specific where it reads a specific word one to one, by an op that no generic value
            of that word and no constant it reads can make two of its outputs equal (`INJECTIVE`), and no other
            argument is an input's values the Program doesn't state."""
            if not base.specific or any(unknown(s) for s in sts):
                return unspecific(base, sts) if base.specific else base
            ok = f.name in INJECTIVE and any(
                s.specific and (nd.form != "batch" or (nd.axes[i] is not None and one_to_one(nd.args[i])))
                for i, s in enumerate(sts))
            for i, s in enumerate(sts):
                if ok and s.stated:
                    cs = [value(*w) for w in words(nd.args[i])]
                    ok = all(isinstance(c, tuple) and INJECTIVE[f.name](c[1]) for c in cs)
            return base if ok else unspecific(base, sts)

        def constants_pass(f, sts: list, nd) -> bool:
            return all(all(isinstance(c, tuple) and INJECTIVE[f.name](c[1]) for c in (value(*w) for w in words(a)))
                       for s, a in zip(sts, nd.args) if s.stated)

        def zero(a) -> bool:
            """Whether argument ``a`` is a stated 0 in every word (`INJECTIVE`'s constant test fails on a finite word),
            so that a multiply by it gives 0 whatever its other argument holds."""
            cs = [value(*w) for w in words(a)]
            return bool(cs) and all(isinstance(c, tuple) and _finite(c[1]) and not _nonzero(c[1]) for c in cs)

        def distinct_status(f, nd, sts: list, st: Status) -> Status:
            """``st`` with `Status.distinct` where an op reads one to one (a batch maps it) an argument whose rows are
            distinct and not stated, so its outputs' rows differ where that argument's do: a rounding (`ROUNDINGS`), a
            shift (`SHIFTS`) whatever the others hold, and any other `INJECTIVE` op only where each other argument is
            stated (and passes the constant test), reached by no input (`θ`'s own), one word that every member of every
            enclosing batch reads (`Status.fixed`; a learning rate, a decay), taken as nonzero, or one whose own rows
            are distinct (a weight times a weight)."""
            batch = nd.form == "batch"

            def mapped(i: int) -> bool:
                return not batch or (nd.axes[i] is not None and one_to_one(nd.args[i]))

            def multiplier(j: int, o: Status) -> bool:
                return not o.reach or o.distinct or (o.fixed and len(set(words(nd.args[j]))) == 1)

            if f.name in ROUNDINGS:
                def keeps(i: int) -> bool:
                    return True
            elif f.name in INJECTIVE and constants_pass(f, sts, nd):
                def keeps(i: int) -> bool:
                    return f.name in SHIFTS or all(multiplier(j, o) for j, o in enumerate(sts) if j != i)
            else:
                return st
            for i, s in enumerate(sts):
                if s.distinct and not s.stated and mapped(i) and keeps(i):
                    return st._replace(distinct=True)
            return st

        for k, nd in enumerate(body.nodes):
            f, leaves = nd.fn, nd.out.leaves
            if is_input(f):
                st = THETA if is_registered(f) or k in theta else \
                    Status(False, False, True, False, frozenset({0}), 0.0, float(max(nd.out.width_set(), default=0)),
                           fixed=True)
                prof.append(((leaves, st),))
                continue
            rate, bits = made(nd.args, nd.out)
            if isinstance(f, PrimitiveDefinition):
                sts = [arg_status(a) for a in nd.args]
                lk = self.lookup(f)
                if lk is not None:
                    names = [n for n, _t in f.params]
                    t, x = sts[names.index(lk.table)], sts[names.index(lk.index)]
                    st = Status(t.aff, t.dep, True, True, t.data | x.data, 0.0, bits) \
                        if x.reach and not x.dep else computed(sts, rate, bits)
                    prof.append(((leaves, st),))
                elif not nd.args:
                    prof.append(((leaves, STATED),))
                elif f.name == "F32Mul" and any(s.stated and zero(a) for s, a in zip(sts, nd.args)):
                    prof.append(((leaves, STATED),))
                else:
                    st = primitive_status(f, nd, sts, computed(sts, rate, bits))
                    prof.append(((leaves, distinct_status(f, nd, sts, st)),))
                continue
            p = self.product(f)
            if p is not None and nd.form in ("call", "batch"):
                params, ret = f.signature()
                names = [n for n, _t in params]
                ix, iw = names.index(p.x), names.index(p.w)
                tx, tw = params[ix][1], params[iw][1]
                members = nd.n if nd.form == "batch" else 1
                K = min(_unit(tx), _unit(tw))
                cols = max(_unit(tx), _unit(tw)) // max(K, 1)
                xm, wm = (nd.form == "batch" and nd.axes[i] is not None for i in (ix, iw))
                dims = (1, members, K, cols) if xm and not wm else (1, 1, K, cols * members) if wm and not xm else \
                    (members, 1, K, cols)
                (ra, bx), (rb, bw) = operand(nd.args[ix]), operand(nd.args[iw])
                sx, sw = arg_status(nd.args[ix]), arg_status(nd.args[iw])
                site = Site(p, k, members, max(_unit(tx), _unit(tw)), ret.leaves, _width(ret), K,
                            (_width(tx), _width(tw)), sx, sw,
                            (tuple(x[2:] for x in self.parts(nd.args[ix].refs)),
                             tuple(x[2:] for x in self.parts(nd.args[iw].refs))), dims, (ra, rb), (bx, bw))
                S.sites.append(site)
                out = min(float(site.out_width), gemm_rate(site, self.k))
                base = computed((arg_status(a) for a in nd.args), out, bits)
                prof.append(((leaves, product_status(nd, ix, iw, base)),))
                continue
            if nd.form == "call":
                C = self.summary(f, tuple(arg_profile(a) for a in nd.args), depth)
                S.children.append((C, 1, [b for i, a in enumerate(nd.args) for b in bindings(i, a, False)]))
                S.at.append((k, (0,)))
                prof.append(capped(C.ret, rate, bits))
            elif nd.form == "batch":
                params, _ = f.signature()
                cctx = tuple(arg_profile(a) if ax is None else ((t.leaves, arg_status(a)._replace(fixed=False)),)
                             for a, ax, (_n, t) in zip(nd.args, nd.axes, params))
                C = self.summary(f, cctx, depth)
                S.children.append((C, nd.n, [b for i, (a, ax) in enumerate(zip(nd.args, nd.axes))
                                             for b in bindings(i, a, ax is not None)]))
                S.at.append((k, range(nd.n)))
                per = C.ret[-1][0]
                ret = C.ret
                sts = [arg_status(a) for a in nd.args]
                if not any(s.specific and ax is not None and one_to_one(a) for s, a, ax in zip(sts, nd.args, nd.axes)):
                    ret = runs_of([(e, unspecific(s, sts) if s.specific else s) for e, s in ret])
                if not any(s.distinct and ax is not None and one_to_one(a) for s, a, ax in zip(sts, nd.args, nd.axes)):
                    ret = runs_of([(e, s._replace(distinct=False)) for e, s in ret])
                prof.append(capped(runs_of([(m * per + e, s) for m in range(nd.n) for e, s in ret])
                                   if len(ret) == 1 or nd.n * len(ret) <= 4096 else ((leaves, whole(ret)),),
                                   rate, bits))
            elif nd.form == "scan":
                prof.append(capped(self.scan(S, k, nd, depth, arg_profile, arg_status, bindings), rate, bits))
            else:
                raise ValueError(f"{fn.id} node {k}: form {nd.form!r}")
        S.ret = returned(self.parts(body.ret), fn.signature()[1], source, list(body.ret)) or \
            ((body.ret.count, THETA),)
        S.sid = len(self.memo)
        self.memo[key] = S
        return S

    def scan(self, S: Summary, k: int, nd, depth: int, arg_profile, arg_status, bindings) -> Profile:
        """A scan's iterations by context: the first's carry `first`'s, each next one's the carry the one before
        returned, until a context repeats. A carried value reads none of the current iteration's inputs, and outside
        every scan, the init and the shared arguments are the loop's, not an iteration's."""
        step, n, nx = nd.fn, nd.n, nd.n_xs
        params, ret = step.signature()
        inner = depth + 1
        gone = frozenset({inner} | ({0} if depth == 0 else set()))
        outer = frozenset({0}) if depth == 0 else frozenset()
        xs = []
        for j in range(nx):
            a, per = nd.args[1 + j], params[1 + j][1].leaves
            s = arg_status(a)
            xs.append(((per, s._replace(data=s.data | {inner})),))
        xs = tuple(xs)
        shared = tuple(drop(arg_profile(a), outer) for a in nd.args[1 + nx:])
        carry_leaves = params[0][1].leaves
        carry = first(drop(arg_profile(nd.args[0]), gone))
        seq: list[tuple[tuple, Summary]] = []
        seen: dict = {}
        while len(seq) < n:
            ctx = (carry, *xs, *shared)
            if ctx in seen:
                break
            seen[ctx] = len(seq)
            C = self.summary(step, ctx, inner)
            seq.append((ctx, C))
            carry = carried(drop(runs_of(_cut(C.ret, 0, carry_leaves)), gone), params[0][1])
        its: list[list[int]] = [[i] for i in range(len(seq))]
        if len(seq) < n:
            s0 = seen[(carry, *xs, *shared)]
            for j in range(len(seq), n):
                its[s0 + (j - len(seq)) % (len(seq) - s0)].append(j)
        last = next(i for i, js in enumerate(its) if js[-1] == n - 1)
        for i, (_ctx, C) in enumerate(seq):
            bs = [b for j, a in enumerate(nd.args) if j or i == 0
                  for b in bindings(j, a, 1 <= j <= nx)]
            S.children.append((C, len(its[i]), bs))
            S.at.append((k, tuple(its[i])))
        ys = join(whole(_cut(C.ret, carry_leaves, C.ret[-1][0])) for _ctx, C in seq)
        y_leaves = ret.leaves - carry_leaves
        return runs_of([*_cut(seq[last][1].ret, 0, carry_leaves), (carry_leaves + n * y_leaves, ys)])


def _cut(p: Profile, lo: int, hi: int) -> list[tuple[int, Status]]:
    """Runs of ``p`` over leaves ``lo`` to ``hi``, ends relative to ``lo``."""
    out, start = [], 0
    for e, s in p:
        a, b = max(start, lo), min(e, hi)
        if b > a:
            out.append((b - lo, s))
        start = e
    return out


class _Classes:
    """Arrays as classes of runs (scope, space, idx, lo, hi): runs of one source that overlap are one array, and so are
    a callee's parameter and the arrays its caller binds to it."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, v):
        p = self.parent.setdefault(v, v)
        while p != v:
            self.parent[v] = self.parent.setdefault(p, p)
            v, p = p, self.parent[p]
        return p

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def overlaps(self) -> None:
        by: dict = {}
        for v in list(self.parent):
            by.setdefault(v[:3], []).append(v)
        for vs in by.values():
            vs.sort(key=lambda v: (v[3], v[4]))
            hi, prev = None, None
            for v in vs:
                if hi is not None and v[3] < hi:
                    self.union(prev, v)
                    hi = max(hi, v[4])
                else:
                    hi = v[4]
                prev = v


@dataclass(frozen=True)
class Group:
    """Products of one shape and one reason: an operand pair's widths, kinds, rates and β, an output width and dims."""

    product: str
    #: (x, w) widths in bits, and whether each is a weight (reads none of the current iteration's inputs)
    widths: tuple[int, int]
    weight: tuple[bool, bool]
    out_width: int
    k_dim: int
    macs: int
    wires: int
    #: the pairs of arrays its products multiply
    edges: frozenset
    #: (batch, m, k, n) of one product node
    dims: tuple[int, int, int, int] = (1, 1, 1, 1)
    #: (x, w): the bits a block pays a word, the source bits a word holds, and the β (H) charges
    rates: tuple[float, float] = (0.0, 0.0)
    bits: tuple[float, float] = (0.0, 0.0)
    beta: tuple[float, float] = (0.0, 0.0)
    #: why it is left out of the charged work ("" where it is charged): "theta", "gathered", "self" or "activations"
    reason: str = ""


@dataclass
class Work:
    """The products of a Program at a policy's k, and the ones the bound charges."""

    #: the policy's k in bits
    k: float
    #: |C| in Boolean gates (`boolean_gates`)
    gates: int
    #: every product, by shape and reason
    products: list[Group]
    #: the adapter's rank that bounds every activation's β, if any shows
    r_step: int | None = None
    #: the share of a token-indexed product's MACs a sequence's start leaves (`SHARED_PREFIX`)
    keep: float = 1.0

    @property
    def groups(self) -> list[Group]:
        return [g for g in self.products if not g.reason]

    @property
    def left_out(self) -> dict[str, int]:
        out = dict.fromkeys(REASONS, 0)
        for g in self.products:
            if g.reason:
                out[g.reason] += g.macs
        return out

    @property
    def N(self) -> int:
        return sum(g.macs for g in self.groups)

    @property
    def O(self) -> int:
        return sum(g.wires for g in self.groups)


REASONS = ("theta", "gathered", "self", "activations")


def boolean_gates(program) -> int:
    """|C| in Boolean gates: each primitive at its Boolean version's AND, XOR and NOT gates (`boolean_version`, counted
    by `verity.circuits.evaluation.bits.compile`, as `training_slowdown.gates` prices a traced op), an input node at
    none, summed once per Definition and multiplied through batches and scans. `NoBooleanVersion` names every
    primitive that has none."""
    from verity.circuits.evaluation.bits import compile as compile_bits
    from verity.definitions.boolean import NoBooleanVersion, boolean_version

    memo: dict = {}
    missing: dict = {}

    def primitive(f) -> int:
        if is_input(f):
            return 0
        try:
            return sum(compile_bits(boolean_version(f)).counts.values())
        except NoBooleanVersion as e:
            missing.update(e.missing)
            return 0

    def of(fn) -> int:
        if fn.id not in memo:
            memo[fn.id] = primitive(fn) if isinstance(fn, PrimitiveDefinition) else \
                sum(of(nd.fn) * (nd.n if nd.form in ("batch", "scan") else 1) for nd in fn.body.nodes)
        return memo[fn.id]

    total = of(program.root)
    if missing:
        raise NoBooleanVersion(missing)
    return total


def adapter_rank(sites) -> int | None:
    """`he_of_rows`' r_step on a Program's products (module docstring), each of ``sites`` once."""
    frozen = any(not a.dep and not a.data and b.dep for s in sites for a, b in ((s.x, s.w), (s.w, s.x)))
    ranks = []
    for s in sites:
        if not (s.x.dep and s.w.dep):
            continue
        _b, m, K, n = s.dims
        widened = frozen and bool(s.x.data) != bool(s.w.data) and 4 * K <= (n if s.x.data else m)
        if widened or (not s.x.data and not s.w.data and 4 * K <= min(m, n)):
            ranks.append(K)
    return min(ranks, default=None)


def least_sequence(sites) -> int | None:
    """The least sequence a Program's products of two activations allow: the second smallest of each one's dims above
    1, at least two of which a token indexes where it is attention's; None where none is."""
    seqs = []
    for s in sites:
        if s.x.data and s.w.data and s.x.dep and s.w.dep:
            ds = sorted(d for d in s.dims[1:] if d > 1)
            if len(ds) >= 2:
                seqs.append(ds[1])
    return min(seqs, default=None)


def operand_beta(rate: float, w: float, activation: bool, r_step: int | None, k: float, most: float) -> float:
    """An operand's β (`he_of_rows`): its width where its rate pays it, else w·ρ/(w + ρ), an activation's at most
    `beta_eff` at the adapter's rank, and every one at most ``most``."""
    b = w if paid(rate, w) else w * rate / (w + rate) if rate > 0 else 0.0
    if r_step is not None and activation:
        b = min(b, beta_eff(w, r_step, k))
    return min(b, most)


def read(program, k: float, products=PRODUCTS, lookups=LOOKUPS) -> Work:
    """The work ``program`` holds at a policy's k (module docstring), from the circuit alone."""
    return _read(program, k, products, lookups)[0]


def calls(program, k: float, products=PRODUCTS, lookups=LOOKUPS):
    """Each product call of ``program``'s circuit, every member and iteration its own, with its `Site` and the reason
    `read` leaves it out ("" where it is charged): the reading's decisions on the circuit's own calls."""
    from verity.circuits.layout import FunctionCall

    _work, R, root, reasons = _read(program, k, products, lookups)
    circ = program.circuit

    def walk(S: Summary, scope):
        nodes = scope.spec.body.nodes
        for i, site in enumerate(S.sites):
            if (S.sid, i) in reasons:
                for m in range(site.n if nodes[site.node].form == "batch" else 1):
                    yield FunctionCall(circ, scope, site.node, m), site, reasons[S.sid, i]
        for (C, _n, _b), (node, members) in zip(S.children, S.at, strict=True):
            for m in members:
                yield from walk(C, scope.child(node, m))

    yield from walk(root, circ.root_scope)


def _read(program, k: float, products, lookups):
    gates = boolean_gates(program)
    R = Reader(program, k, products, lookups)
    root = R.root()
    order = sorted(R.memo.values(), key=lambda s: -s.sid)
    count = {root.sid: 1}
    for s in order:
        for c, n, _b in s.children:
            count[c.sid] = count.get(c.sid, 0) + count.get(s.sid, 0) * n
    cls = _Classes()
    for s in order:
        for c, _n, bs in s.children:
            for i, a, b, space, idx, lo, hi in bs:
                cls.union((c.sid, "p", i, a if b >= 0 else 0, b if b >= 0 else math.inf), (s.sid, space, idx, lo, hi))
        for site in s.sites:
            for src in site.sources:
                for v in src:
                    cls.find((s.sid, *v))
    cls.overlaps()
    ops = []
    for s in order:
        for i, site in enumerate(s.sites):
            xs, ws = ([(s.sid, *v) for v in src] for src in site.sources)
            for vs in (xs, ws):
                for v in vs[1:]:
                    cls.union(vs[0], v)
            if count.get(s.sid, 0) * site.n * site.macs:
                ops.append((s, i, site, xs[0] if xs else None, ws[0] if ws else None))
    sites = [site for _s, _i, site, _x, _w in ops]
    r_step = adapter_rank(sites)
    seq = least_sequence(sites)
    keep = 1 - SHARED_PREFIX / seq if seq else 1.0
    bk = min_cut().beta(k)
    most = min(min_cut().BF16, bk) if bk is not None else min_cut().BF16
    groups: dict = {}
    reasons: dict = {}
    for s, i, site, x, w in ops:
        n = count[s.sid] * site.n
        if not (site.x.aff and site.w.aff):
            reason = "theta"
        elif may_repeat(site.x) or may_repeat(site.w):
            reason = "gathered"
        elif x is None or w is None or cls.find(x) == cls.find(w):
            reason = "self"
        elif site.x.data and site.w.data:
            reason = "activations"
        else:
            reason = ""
        reasons[s.sid, i] = reason
        weight = (not site.x.data, not site.w.data)
        beta = tuple(round(operand_beta(r, p, not wt, r_step, k, most), 9)
                     for r, p, wt in zip(site.rates, site.passed(), weight))
        key = (site.product.op, site.widths, weight, site.out_width, site.k_dim, site.dims,
               tuple(round(r, 6) for r in site.rates), tuple(round(b, 6) for b in site.bits), beta, reason)
        g = groups.setdefault(key, [0, 0, set()])
        g[0] += n * site.macs
        g[1] += n * site.outputs * site.out_width
        if x is not None and w is not None:
            g[2].add(frozenset((cls.find(x), cls.find(w))))
    out = [Group(key[0], key[1], key[2], key[3], key[4], m, o, frozenset(e), *key[5:])
           for key, (m, o, e) in sorted(groups.items(), key=lambda kv: repr(kv[0]))]
    return Work(k, gates, out, r_step, keep), R, root, reasons


def clique(edges) -> int:
    """The most vertices pairwise joined by ``edges`` (pairs), at least 2 where there is an edge."""
    adj: dict = {}
    for e in edges:
        a, b = tuple(e)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    best = 0

    def grow(r: int, p: set, x: set) -> None:
        nonlocal best
        if not p and not x:
            best = max(best, r)
            return
        if r + len(p) <= best:
            return
        u = max(p | x, key=lambda v: len(adj[v] & p))
        for v in list(p - adj[u]):
            grow(r + 1, p & adj[v], x & adj[v])
            p.discard(v)
            x.add(v)

    grow(0, set(adj), set())
    return best


def min_cut():
    """The calculator's `min_cut`: its directory, one path from the repository's root, goes on `sys.path`, and
    `rho.min_cut` is imported from it."""
    rho = str(Path(__file__).resolve().parents[3] / "verity/protocols/inference_only/rho")
    if rho not in sys.path:
        sys.path.append(rho)
    from rho import min_cut as mc
    return mc


@dataclass(frozen=True)
class Priced:
    """The bound at a policy on a sub-work: which groups, their ω, and `min_cut.bound`'s result."""

    bound: object
    groups: tuple[Group, ...]
    omega: int
    N: float
    O: int


def betas(g: Group) -> tuple[float, float]:
    """A group's (β_w, β_x): the weight's first, and where both are of one kind, the larger."""
    b = g.beta
    if g.weight[0] != g.weight[1]:
        return (b[1], b[0]) if g.weight[1] else (b[0], b[1])
    return tuple(sorted(b, reverse=True))


def bound(work: Work, c: float, n: float | None = None, F: float | None = None) -> Priced | None:
    """The best bound at the policy (c, the work's k, n) over ``work``'s sub-works (module docstring), each at ``F``
    where given, else its own share; None where it charges no product, (H) states no activation's β at k, or k is
    `HEAD_K` or more."""
    mc = min_cut()
    k = work.k
    if mc.beta(k) is None or k >= HEAD_K or not work.groups:
        return None
    priced = [(math.prod(g.beta), g) for g in work.groups]
    best = None
    for P in sorted({q for q, _ in priced if q > 0}):
        for K0 in sorted({g.k_dim for g in work.groups}):
            for kinds in ("both", "weights", "activations"):
                M = [(q, g) for q, g in priced if q >= P and g.k_dim >= K0 and
                     (kinds == "both" or any(g.weight) == (kinds == "weights"))]
                if not M:
                    continue
                bw, bx = betas(min(M, key=lambda x: x[0])[1])
                omega = max(2, clique(e for _, g in M for e in g.edges))
                N = sum(g.macs * (1.0 if all(g.weight) else work.keep) for _, g in M)
                O = sum(g.wires for _, g in M)
                b = mc.bound(c, k, n, N=N, O=O, size=work.gates, beta_w=bw, beta_x=bx,
                             b=min(mc.BF16, *(g.out_width for _, g in M)), F=mc.share(omega) if F is None else F)
                if best is None or b.blocks > best.bound.blocks:
                    best = Priced(b, tuple(g for _, g in M), omega, N, O)
    return best
