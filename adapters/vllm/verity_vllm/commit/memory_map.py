"""A request Program's KV rows from the circuit: each row's leaves, the step it is born at and its last structural reader's,
from the Program of record and its census (`census_of`, `rows`, cached beside each request Build by `pipeline.memory_map`).

~~~text
row       one position's K (or V) in a layer: a row of an attention Call's `kc` (`vc`), `Array(T, row)`, or MLA's latent row of a
          sparse MLA Call's `kv` (`LATENT`: the normed latent, then the roped key, one plane), a batch's members each read as a
          Call; a row past INDEX_TOPK reads its `kv` through a concentrator, or under table reads through its reads of the
          (request, layer) history table (`ROUTED`), whose history is the rows it reads.  A layer's rows are the rows its Calls
          share, in the order its Calls first read them (its positions), K and V paired by position.  DSA's indexer keys are no
          rows: their 132-byte rows pack across blocks, and memory accounting leaves them uncredited until the map packs them
born      a row's step, and `last` its last structural reader's (the last step with a gate that has one of its values as
          an operand, a gather that may select one included, whichever it selected): the circuit's requirement
          (`requirement.from_census`) over the Program's census at its sampling events (`query.required.SAMPLING_EVENTS`)
cached    a warm request's positions 0..C-1 (`Rows.cached`): rows its Program reads from its cached-prefix input (bf16
          [L, 2, C, row], `kv_prefix.seed`: layer l's K and V rows at the l-th history opened), which other requests computed,
          one a cached page (its producer, the prefix plan's); such a row's `born` and `last` are its first and last reader's
          in the warm Program.  The warm request writes none of them
~~~
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from verity.circuits.defs import PrimitiveDefinition
from verity.circuits.program import Program, is_input
from verity.circuits.refs import Coll, runs
from verity.circuits.types import Array
from verity.definitions.circuits.census import census, step_ends
from verity.definitions.circuits.requirement import from_census

from verity_vllm.query.required import SAMPLING_EVENTS

__all__ = ["census_of", "rows", "rows_doc"]

#: an attention Call's cached K and V parameters
KV = ("kc", "vc")
#: the Definitions, by name, whose cached parameter is one plane of MLA latent rows (FA3's sparse MLA, each version)
LATENT: dict[str, tuple[str, ...]] = {"SparseMla": ("kv",), "SparseMlaSplit": ("kv",)}
#: the Definitions, by name, through which a row past INDEX_TOPK reads its layer's latent rows, and the parameter holding
#: those: the concentrators (`DsaButterflyRowsPad`'s pad is no row), and under table reads the history table's reads
#: (`program.dsa_table`), whose table is the block's history
ROUTED = {"DsaButterflyRows": "rows", "DsaButterflyRowsPad": "rows", "DsaHistoryReads": "hist"}

#: leaves `[lo, hi)` of the output of node `j` of the Program's function (the Commit's instance rows' `['n', j, lo, hi]`), or
#: (`CACHED`, k, lo, hi) of root input node k: a cached row, which the warm request reads and never computes
Slice = tuple[str, int, int, int]
CACHED = "input"


@dataclass(frozen=True)
class Row:
    """One position's K or V in a layer: the leaves it is, in order, the step it is born at and its last structural
    reader's."""

    group: tuple[Slice, ...]
    born: int
    last: int


@dataclass(frozen=True)
class Rows:
    """A request Program's KV rows, ``layers[l][plane][position]`` (from `rows`, plane 0 K and 1 V, or one plane of latent rows),
    each `row_bytes` bytes of `word_bytes`-byte words, over its `steps` steps; positions below `cached` are a warm Program's
    cached prefix, which other requests wrote."""

    layers: tuple[tuple[tuple[Row, ...], ...], ...]
    row_bytes: int
    word_bytes: int
    steps: int
    cached: int = 0

    @property
    def positions(self) -> int:
        return len(self.layers[0][0]) if self.layers else 0


def _cut(refs, width: int, inputs: int, what: str, cached: bool = False) -> list[tuple[Slice, ...]]:
    """A cached argument's (`kc`, `vc` or `kv`) rows of `width` leaves, each as the function's slices it is, a root input's
    leaves as `CACHED` slices where the Program has a `cached` prefix."""
    out, cur, need = [], [], width
    for space, k, lo, hi in runs(refs):
        if space != "n" or (k < inputs and not cached):
            raise ValueError(f"{what} reads an input of the Program: memory-map/v1 records the KV a step computes, and a warm "
                             "Program's cached rows by its cached-prefix layout (its Build's prefix_cache.json)")
        while lo < hi:
            take = min(need, hi - lo)
            cur.append(("n", k - inputs, lo, lo + take) if k >= inputs else (CACHED, k, lo, lo + take))
            lo, need = lo + take, need - take
            if not need:
                out.append(tuple(cur))
                cur, need = [], width
    return out


def _member(nd, i: int, m: int) -> Coll:
    """Argument `i` of member `m` of root node `nd` (a Call's own; a batch's shared one, or its slice of a mapped one)."""
    a = nd.args[i]
    if nd.form != "batch" or nd.axes[i] is None:
        return a
    if nd.axes[i] != 0:
        raise ValueError(f"a batch of {nd.fn.id} maps its argument {i} on axis {nd.axes[i]}: memory-map/v1 reads a member's on axis 0")
    w = a.type.elem.leaves
    return Coll(a.type.elem, a.refs.slice(m * w, (m + 1) * w))


def _history(nodes, a: Coll) -> Coll:
    """`a`, or where it is the output of one concentrator or history table read (`ROUTED`, a Call or a batch member), the history
    it routes or reads the rows from: a row past INDEX_TOPK reads its layer's rows through it."""
    rs = runs(a.refs)
    if not rs or rs[0][0] != "n" or any((s, j) != rs[0][:2] for s, j, _, _ in rs):
        return a
    src = nodes[rs[0][1]]
    p = ROUTED.get(getattr(src.fn, "name", ""))
    if p is None:
        return a
    m = rs[0][2] // (src.out.leaves // src.n) if src.form == "batch" else 0
    return _member(src, [n for n, _ in src.fn.signature()[0]].index(p), m)


def census_of(program: Program, digest: str | None = None) -> dict:
    """The census/v1 of a request Program over its steps, one per sampling event, naming it by `digest`."""
    return census(program, step_ends(program, SAMPLING_EVENTS), digest)


def rows(program: Program, c: Mapping | None = None, prefix: Mapping | None = None) -> Rows:
    """The KV rows of a request Program, each with its birth and its last structural reader as its requirement over its
    census `c` (`census_of`, computed when not given) has them: a row's `last` counts a step whose gather may select it,
    not only one that did.  `prefix`: a warm Program's cached-prefix layout (`check.replay.cached_prefix.layout_of`, its
    Build's ``prefix_cache.json``), whose C rows open every layer's history (`cached`, `_cached`); a cached row is born at its
    first reader.  A Program reading an input as KV without one is refused."""
    body = program.root.body
    inputs = sum(1 for nd in body.nodes if is_input(nd.fn))
    c = census_of(program) if c is None else c
    req = from_census(c)
    vals = c["values"]
    by_node: dict[int, list[tuple[int, int, int]]] = {}
    for v, (k, lo, hi) in enumerate(zip(vals["producer"], vals["lo"], vals["hi"])):
        by_node.setdefault(k, []).append((lo, hi, v))

    def row(group: tuple[Slice, ...]) -> Row:
        vs = []
        for space, j, lo, hi in group:
            got = [(a, b, v) for a, b, v in by_node.get(j if space == CACHED else j + inputs, ()) if a < hi and b > lo]
            if sum(min(b, hi) - max(a, lo) for a, b, _ in got) != hi - lo:
                raise ValueError(f"leaves [{lo}, {hi}) of {'input ' if space == CACHED else ''}node {j} are no census value's: "
                                 "a KV row must be its producer's output or a cached row")
            vs += [v for _, _, v in got]
        born = max(req.born[v] for v in vs)
        return Row(group, born, max(req.last.get(v, born) for v in vs))

    layer_of: dict[tuple[Slice, ...], int] = {}
    pair: dict[tuple[Slice, ...], tuple[tuple[Slice, ...], ...]] = {}
    layers: list[list[tuple[Slice, ...]]] = []
    kind = None
    for k, nd in enumerate(body.nodes):
        if is_input(nd.fn) or isinstance(nd.fn, PrimitiveDefinition):
            continue
        names = [n for n, _ in nd.fn.signature()[0]]
        cached = LATENT.get(getattr(nd.fn, "name", ""), KV)
        if not set(cached) <= set(names):
            continue
        if nd.form not in ("call", "batch"):
            raise ValueError(f"root node {k} ({nd.fn.id}) holds its KV in a {nd.form}: memory-map/v1 reads a Call's or a batch member's")
        for m in range(nd.n if nd.form == "batch" else 1):
            args = [_history(body.nodes, _member(nd, names.index(p), m)) for p in cached]
            t = args[0].type
            if not isinstance(t, Array) or any(a.type != t for a in args) or t.elem.homogeneous_width() is None:
                raise ValueError(f"root node {k} ({nd.fn.id}) holds its KV otherwise than as {', '.join(cached)}: Array(T, row) each")
            here = (len(cached), t.elem.leaves, t.elem.homogeneous_width())
            if kind not in (None, here):
                raise ValueError(f"root node {k} ({nd.fn.id}) has rows of {here} (planes, leaves, bits), another layer's are {kind}")
            kind = here
            ks, *vs = (_cut(a.refs, here[1], inputs, f"root node {k}'s {p}", prefix is not None) for a, p in zip(args, cached))
            seen = {layer_of[g] for g in ks if g in layer_of}
            if len(seen) > 1:
                raise ValueError(f"root node {k} ({nd.fn.id}) reads the K rows of layers {sorted(seen)}")
            layer = seen.pop() if seen else len(layers)
            if layer == len(layers):
                layers.append([])
            for kg, *vg in zip(ks, *vs):
                if pair.setdefault(kg, tuple(vg)) != tuple(vg):
                    raise ValueError(f"root node {k} ({nd.fn.id}) reads a K row with another V row than a Call before it")
                if kg not in layer_of:
                    layer_of[kg] = layer
                    layers[layer].append(kg)
    if kind is None:
        raise ValueError("the Program has no attention Call with kc and vc, nor a sparse MLA Call's kv: no KV to record")
    planes, leaves, bits = kind
    if bits % 8:
        raise ValueError(f"a KV word of {bits} bits is no whole bytes")
    C = _cached(body.nodes, layers, pair, planes, leaves, prefix)
    return Rows(tuple((tuple(row(g) for g in ks), *(tuple(row(pair[g][i]) for g in ks) for i in range(planes - 1))) for ks in layers),
                leaves * bits // 8, bits // 8, req.steps, C)


def _cached(nodes, layers: list[list[tuple[Slice, ...]]], pair: Mapping, planes: int, leaves: int, prefix: Mapping | None) -> int:
    """The cached positions of a Program's KV, `layers` its K rows layer by layer in the order their histories open and
    `pair` each one's V row: 0 without a `prefix` layout, else its C, once the wiring is the layout's: one root input of
    [L, 2, C, row] leaves (`kv_prefix`), layer i's K (V) row at position j < C its leaves at ``((2 i + kv) C + j) row``, and
    every row from position C one the Program computes."""
    if prefix is None:
        return 0
    C, rl, L = int(prefix["C"]), int(prefix["row_leaves"]), len(prefix["layers"])
    if planes != 2 or rl != leaves or len(layers) != L or any(len(ks) <= C for ks in layers):
        raise ValueError(f"a cached prefix of {C} rows of {rl} leaves of K and V in {L} layers, where the Program's KV is "
                         f"{len(layers)} layers of {planes} plane(s) of {sorted({len(ks) for ks in layers})} rows of {leaves} leaves")
    k = layers[0][0][0][1]
    if layers[0][0][0][0] != CACHED or nodes[k].out.leaves != L * 2 * C * rl:
        raise ValueError(f"the Program's first KV row isn't row 0 of an input of the cached prefix's {L} x 2 x {C} x {rl} leaves")
    for i, ks in enumerate(layers):
        for j, g in enumerate(ks):
            for kv, x in enumerate((g, *pair[g])):
                o = ((2 * i + kv) * C + j) * rl
                if (x != ((CACHED, k, o, o + rl),)) if j < C else any(s[0] == CACHED for s in x):
                    raise ValueError(f"layer {i}'s {'KV'[kv]} row at position {j} is {[list(s) for s in x][:2]}: a warm Program's layer "
                                     f"reads rows 0..{C - 1} from its cached prefix (input node {k}, [L, 2, C, row]) and computes the rest")
    return C


def rows_doc(rs: Rows) -> dict[str, Any]:
    """`rs` as JSON: a row as ``[born, last, [[node, lo, hi], ...]]``, a cached row's slice ``["input", k, lo, hi]``."""
    return {"row_bytes": rs.row_bytes, "word_bytes": rs.word_bytes, "steps": rs.steps, "cached": rs.cached,
            "layers": [[[[x.born, x.last, [[j, lo, hi] if s == "n" else [s, j, lo, hi] for s, j, lo, hi in x.group]] for x in plane]
                        for plane in kv] for kv in rs.layers]}


def rows_of(doc: Mapping[str, Any]) -> Rows:
    """The `Rows` `rows_doc` wrote."""
    return Rows(tuple(tuple(tuple(Row(tuple(("n", *s) if len(s) == 3 else tuple(s) for s in g), born, last) for born, last, g in plane) for plane in kv)
                      for kv in doc["layers"]), int(doc["row_bytes"]), int(doc["word_bytes"]), int(doc["steps"]), int(doc.get("cached", 0)))
