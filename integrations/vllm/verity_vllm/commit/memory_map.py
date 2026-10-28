"""memory-map/v1 from the circuit (`verity.core.protocols.remote_bandwidth.memory_map`): the KV blocks each serving step wrote
and freed, from the Program of record and the engine's page layout, with each block's digest from the values served (the
Commit's committed words, `pipeline.memory_map`), as `Lines` builds it (`protocols.remote_bandwidth.memory_map`: j and the rules).

~~~text
row       one position's K (or V) in a layer: a row of an attention Call's `kc` (`vc`), `Array(T, row)`, or MLA's latent row
          of a sparse MLA Call's `kv` (`LATENT`: the normed latent, then the roped key, one plane); a layer's rows are the
          rows its Calls share, in the order its Calls first read them (its positions), K and V paired by position.  DSA's
          indexer keys are no rows: their 132-byte rows pack across blocks, and memory accounting leaves them uncredited
          until the map packs them
born      a row's step, and `last` its last structural reader's (the last step with a gate that has one of its values as
          an operand, a gather that may select one included, whichever it selected): the circuit's requirement
          (`requirement.from_census`) over the Program's census at its sampling events (`query.required.SAMPLING_EVENTS`)
cached    a warm request's positions 0..C-1 (`Rows.cached`): rows its Program reads from its cached-prefix input (bf16
          [L, 2, C, row], `kv_prefix.seed`: layer l's K and V rows at the l-th history opened), which other requests computed,
          one a cached page (its producer, the prefix plan's); such a row's `born` and `last` are its first and last reader's
          in the warm Program.  The warm request writes none of them: its first C / block_size pages are its producers'
          (`tables`), and a block there is its producer's, freed at the last structural reader of its rows over every
          request that reads it (`record`), so a page two requests share is written and counted once
page      `block_size` slots in each layer as the engine lays them out (`page_of`, from its cache tensor): a K plane then a
          V plane, a slot one row ("planes"), one plane whose slot holds a position's K and V head by head ("packed",
          `packed`), or one plane of latent rows ("latent"); a request's position q sits at slot
          `tables[request][q // block_size] * block_size + q % block_size`, its pages the ones the engine wrote its KV to
          (`tables`, from the slots the Commit committed) or, for its cached positions, its producers', vLLM's null block
          being page 0
block     4,096 bytes of a plane (`store.registry._page_blocks`), at slot `firsts[layer] + ids[page, c]` (the `Store`'s
          numbering: a KV pool of one region per layer)
appended  a block whose rows are all born, at the step of the last (its completion), if a later step reads it: a step's
          record, its frees then its writes, takes effect after the step's reads (remote_bandwidth's convention),
          so a block whose last structural reader is the step that completes it is never read while resident and is in
          no record; nor is a block with a row no step writes (it stays dirty).  A step's blocks are one append, by slot
written   an appended block, where its append's encode landed (the session's `Deferral`; without one, at its completion)
freed     a written block at its rows' last structural reader (memory-map/v1's free).  And a slot's registered block (a KV
          page's filler, registered before the seed) at the step a write first reaches it, even the one completing it: its
          bytes are no longer that block's, and a block not yet written is uncredited (the `Store`'s dirty).  A page's
          untouched blocks stay held, counting against capacity, until then: a request's end frees only the blocks it wrote
re-key    a live written block as old as the session's lifetime cap, its old j freed in the record of the step before its
          append's.  Which appends are re-keys is the session's (`Deferral`); the blocks they take are the live written
          ones of the oldest appends (a window is a prefix of them), not freed at that step
d_j       `codec.digest` of the block's payload: its rows' served words in slot order, each word little-endian
step      the engine's: request r's step s is `arrive[r] + s`
~~~

The map is the prover's witness (d_j hashes user data): it goes to the prover's own file and is never published.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import SLOT_LIMIT, Header, Step
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import runs
from verity.ml.circuits.requirement import from_census
from verity.ml.circuits.types import Array
from verity_vllm.query.required import SAMPLING_EVENTS
from verity_vllm.protocols.remote_bandwidth.interface import PageLayout
from verity_vllm.protocols.remote_bandwidth.memory_map import Lines
from verity_vllm.protocols.remote_bandwidth.registry import _page_blocks

#: an attention Call's cached K and V parameters
KV = ("kc", "vc")
#: the Definitions, by name, whose cached parameter is one plane of MLA latent rows (FA3's sparse MLA, each version)
LATENT = {"SparseMla": ("kv",), "SparseMlaSplit": ("kv",)}
#: vLLM's null block, no request's page
NULL_PAGE = 0

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
    """A request Program's KV rows, ``layers[l][plane][position]`` (from `rows`, plane 0 K and 1 V, or one plane of latent rows;
    after `packed`, one plane), each `row_bytes` bytes of `word_bytes`-byte words, over its `steps` steps; positions below
    `cached` are a warm Program's cached prefix, which other requests wrote."""

    layers: tuple[tuple[tuple[Row, ...], ...], ...]
    row_bytes: int
    word_bytes: int
    steps: int
    cached: int = 0

    @property
    def positions(self) -> int:
        return len(self.layers[0][0]) if self.layers else 0


@dataclass(frozen=True)
class Layout:
    """The engine's KV pool: `pages` pages of `block_size` slots in each layer's region, layer l's from store index
    `firsts[l]`, in a registered store of `blocks` blocks.  Unset, the pool is the whole store, its regions layer by layer,
    with the pages the session takes."""

    block_size: int = 16
    pages: int | None = None
    firsts: tuple[int, ...] | None = None
    blocks: int | None = None


@dataclass(frozen=True)
class Deferral:
    """A remote session's deferred residency (`store.remote.Serving.deferral`, its record's ``memory_map.deferral``): each
    append's ``(step made, step landed, first, count)`` in append order, landed None while its encode is out, on the
    session's steps, whose first is the engine's step 0 (``first_step``), under the session's lifetime cap ``lifetime``.
    An append is made at its completion step, a re-key at the step after the one whose record frees its old indices."""

    encodes: tuple[tuple[int, int | None, int, int], ...]
    first_step: int
    lifetime: int


@dataclass(frozen=True)
class _Append:
    made: int
    landed: int | None
    count: int
    rekey: bool
    blocks: list[tuple[int, bytes, int]]


@dataclass(frozen=True)
class Page:
    """One layer's KV page as the engine lays it out: `pages` pages of `block_size` slots, "planes" (a K plane then a V
    plane), "packed" (a slot holds a position's K and V, `head_words` words of each per head in turn) or "latent" (a slot
    holds a position's latent row, of `head_words` words)."""

    form: str
    pages: int
    block_size: int
    head_words: int


def page_of(shape: Sequence[int], stride: Sequence[int], itemsize: int, rs: Rows) -> Page:
    """The page of a layer's KV cache tensor (its shape, strides in elements and element bytes) holding `rs`'s rows: vLLM
    0.28's FlashAttention ``[pages, heads, block_size, 2 * head]`` slot-major (strides ``(bs*H*2D, 2D, H*2D, 1)``) is
    "packed", ``[pages, 2, block_size, heads, head]`` contiguous "planes", and MLA's ``[pages, block_size, row]`` contiguous
    (bound squeezed of its one head) "latent", for latent rows; any other layout is refused."""
    words, shape, stride = rs.row_bytes // rs.word_bytes, tuple(map(int, shape)), tuple(map(int, stride))
    planes = len(rs.layers[0]) if rs.layers else 0
    if itemsize != rs.word_bytes:
        raise ValueError(f"a KV cache of {itemsize}-byte elements holds rows of {rs.word_bytes}-byte words")
    if len(shape) == 3 and planes == 1:
        nb, bs, w = shape
        if w == words and stride == (bs * w, w, 1):
            return Page("latent", nb, bs, w)
    if len(shape) == 4 and planes == 2:
        nb, h, bs, d2 = shape
        if h * d2 == 2 * words and stride == (bs * h * d2, d2, h * d2, 1):
            return Page("packed", nb, bs, d2 // 2)
    if len(shape) == 5 and planes == 2:
        nb, two, bs, h, d = shape
        if two == 2 and h * d == words and stride == (2 * bs * h * d, bs * h * d, h * d, d, 1):
            return Page("planes", nb, bs, d)
    raise ValueError(f"a KV cache {list(shape)} at strides {list(stride)} is no page memory-map/v1 lays out {planes} plane(s) of "
                     f"rows of {words} words in")


def packed(rs: Rows) -> Rows:
    """`rs` with a layer's K and V rows of a position as one row (its group the K row's leaves, then the V row's), the one
    plane of a "packed" page; its payload interleaves them head by head."""
    return replace(rs, row_bytes=2 * rs.row_bytes, layers=tuple(
        (tuple(Row(k.group + v.group, max(k.born, v.born), max(k.last, v.last)) for k, v in zip(*kv)),) for kv in rs.layers))


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
        cached = LATENT.get(getattr(nd.fn, "name", None), KV)
        if not set(cached) <= set(names):
            continue
        args = [nd.args[names.index(p)] for p in cached]
        t = args[0].type
        if nd.form != "call" or not isinstance(t, Array) or any(a.type != t for a in args) or t.elem.homogeneous_width() is None:
            raise ValueError(f"root node {k} ({nd.fn.id}) holds its KV otherwise than as one Call's {', '.join(cached)}: Array(T, row) each")
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
    if planes != 2 or rl != leaves or L != len(layers) or any(len(ks) <= C for ks in layers):
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
    return Rows(tuple(tuple(tuple(Row(tuple(("n", *s) if len(s) == 3 else tuple(s) for s in g), b, l) for b, l, g in plane) for plane in kv)
                      for kv in doc["layers"]), int(doc["row_bytes"]), int(doc["word_bytes"]), int(doc["steps"]), int(doc.get("cached", 0)))


def _producers(requests: Mapping[str, Rows], producers: Mapping[str, Sequence[str]] | None, block_size: int) -> dict[str, list[str]]:
    """Each warm request's producers, one a cached page, refused unless they are its `Rows.cached` positions' whole pages."""
    out = {r: [str(p) for p in (producers or {}).get(r, ())] for r in requests}
    for r, rs in requests.items():
        if len(out[r]) * block_size != rs.cached:
            raise ValueError(f"request {r}'s Program reads {rs.cached} cached rows, and {len(out[r])} producing requests are named "
                             f"for them: one a whole cached page of {block_size} slots")
    return {r: ps for r, ps in out.items() if ps}


def tables(requests: Mapping[str, Rows], arrive: Mapping[str, int], slots: Callable[[str, int], Sequence[int]],
           block_size: int, producers: Mapping[str, Sequence[str]] | None = None) -> dict[str, list[int]]:
    """Each request's pages as the engine wrote its KV: `slots(request, step)` the slot of each row it computed at that engine
    step, in position order (vLLM's slot mapping, position q at ``page * block_size + q % block_size``).  The rows a step wrote
    are the ones its Program writes there, else it is refused, and so is a page that isn't one page throughout or a request
    holding a page twice.  A warm request (`Rows.cached`, a prefix-cache hit) writes from position C: its first C / block_size
    pages are the ones `producers[request]` (the request that computed each, the prefix plan's) wrote, refused unless that
    request of the session computed the whole page itself."""
    producers = _producers(requests, producers, block_size)
    out: dict[str, list[int]] = {}
    for r, rs in requests.items():
        born, pages, q = [x.born for x in rs.layers[0][0]], [NULL_PAGE] * -(-rs.positions // block_size), rs.cached
        for s in range(rs.steps):
            got, n = [int(v) for v in slots(r, arrive[r] + s)], born[rs.cached:].count(s)
            if len(got) != n or born[q:q + n] != [s] * n:
                raise ValueError(f"request {r} wrote {len(got)} KV rows at step {arrive[r] + s}, where its Program writes {n} from position {q}")
            for p, slot in enumerate(got, start=q):
                page, off = divmod(slot, block_size)
                if off != p % block_size or page == NULL_PAGE or pages[p // block_size] not in (NULL_PAGE, page):
                    raise ValueError(f"request {r}'s position {p} went to slot {slot} (page {page}, offset {off}), its page "
                                     f"{p // block_size} being {pages[p // block_size]}")
                pages[p // block_size] = page
            q += n
        if q != rs.positions:
            raise ValueError(f"request {r}'s {rs.positions} positions took pages {pages[:8]}{'...' if len(pages) > 8 else ''} over {q} rows")
        out[r] = pages
    for r, ps in producers.items():
        for b, p in enumerate(ps):
            if p not in out or requests[p].cached > b * block_size or requests[p].positions < (b + 1) * block_size:
                raise ValueError(f"request {r}'s cached page {b} was computed by {p}, "
                                 f"{'no request of the session' if p not in out else 'whose Program did not compute it whole'}")
            out[r][b] = out[p][b]
    for r, pages in out.items():
        if len(set(pages)) != len(pages):
            raise ValueError(f"request {r}'s {requests[r].positions} positions took pages {pages[:8]}{'...' if len(pages) > 8 else ''}")
    return out


@dataclass(frozen=True)
class _Pool:
    """The session's pages in the store: page p's c-th block of layer l at slot ``firsts[l] + ids[p, c]``, covering a plane's
    slots ``[lo[c], hi[c])`` (`_page_blocks`)."""

    ids: Any
    lo: Any
    hi: Any
    per_plane: int
    firsts: tuple[int, ...]
    region: int
    row_bytes: int
    pages: int

    def slot(self, layer: int, page: int, c: int) -> int:
        return self.firsts[layer] + int(self.ids[page, c])


def _pool(requests: Mapping[str, Rows], layout: Layout, tables: Mapping[str, Sequence[int]]) -> _Pool:
    shapes = {(len(rs.layers), len(rs.layers[0]), rs.row_bytes) for rs in requests.values()}
    if len(shapes) != 1:
        raise ValueError(f"the requests' KV differs in (layers, planes, row bytes): {sorted(shapes)}")
    ((nl, planes, sb),) = shapes
    pages = layout.pages if layout.pages is not None else 1 + max(max(t, default=NULL_PAGE) for t in tables.values())
    ids, lo, hi = _page_blocks(PageLayout(pages, layout.block_size, sb, planes=planes), codec.BLOCK_BYTES)
    region = int(ids.max()) + 1
    firsts = layout.firsts if layout.firsts is not None else tuple(layer * region for layer in range(nl))
    if len(firsts) != nl:
        raise ValueError(f"{len(firsts)} layer regions for {nl} layers of KV")
    return _Pool(ids, lo, hi, ids.shape[1] // planes, tuple(firsts), region, sb, pages)


def record(requests: Mapping[str, Rows], arrive: Mapping[str, int], payload: Callable[[str, Row], bytes], registration: bytes,
           layout: Layout, tables: Mapping[str, Sequence[int]], deferral: Deferral | None = None,
           producers: Mapping[str, Sequence[str]] | None = None) -> tuple[Header, list[Step]]:
    """The session's memory map beside the registered store `registration` digests: each request's rows from its arrival step,
    a row's served bytes `payload(request, row)`, its pages `tables[request]` (`tables`), its writes where `deferral` has
    their encodes land.  A warm request's cached pages are its `producers`' (as `tables` takes them): it writes no block of
    them, and it holds each to its last structural reader there, so a page two requests share is written once and freed after
    the last reader of either; one it reads before its producer completes it is refused.  A slot written while another
    request's rows in it are still read is refused, and so is a deferral under lifetime cap 1 or one whose appends aren't the
    circuit's."""
    if deferral is not None and deferral.lifetime < 2:
        raise ValueError(f"a session deferring residency under lifetime cap {deferral.lifetime} is refused (memory accounting): "
                         "a block whose encode lands in the window after its append's would be credited at the cap")
    pool = _pool(requests, layout, tables)
    ids, lo, hi, sb, pages, bs = pool.ids, pool.lo, pool.hi, pool.row_bytes, pool.pages, layout.block_size
    producers = _producers(requests, producers, bs)
    filler: dict[int, list[int]] = {}
    done: dict[int, list[tuple[int, bytes, int]]] = {}
    #: each slot's blocks: the step a write first reaches it, the step completing it (None: it stays dirty), its rows' last
    #: structural reader and its digest
    uses: dict[int, list[tuple[int, int | None, int, bytes]]] = {}
    #: each request's block in `uses[slot]`, by (request, slot), and each cached block a warm request reads: (request, its
    #: producer, slot, first and last read)
    at: dict[tuple[str, int], int] = {}
    cached: list[tuple[str, str, int, int, int]] = []
    for r, rs in requests.items():
        if len(tables[r]) * bs < rs.positions or any(not NULL_PAGE < p < pages for p in tables[r]):
            raise ValueError(f"request {r}'s pages {list(tables[r])[:8]} don't hold its {rs.positions} positions in pages 1 to {pages - 1}")
        for q, page in enumerate(tables[r]):
            for c in range(ids.shape[1]):
                plane, i = divmod(c, pool.per_plane)
                p0, p1 = q * bs + int(lo[c]), q * bs + int(hi[c])
                if p0 >= rs.positions:
                    continue
                off = i * codec.BLOCK_BYTES - int(lo[c]) * sb
                for layer, kv in enumerate(rs.layers):
                    got, slot = kv[plane][p0:p1], pool.slot(layer, page, c)
                    reach, last = arrive[r] + min(x.born for x in got), arrive[r] + max(x.last for x in got)
                    if q < len(producers.get(r, ())):
                        cached.append((r, producers[r][q], slot, reach, last))
                        continue
                    at[r, slot] = len(uses.setdefault(slot, []))
                    if p1 > rs.positions:
                        uses[slot].append((reach, None, last, b""))
                        continue
                    data = b"".join(payload(r, x) for x in got)
                    if len(data) != len(got) * sb:
                        raise ValueError(f"request {r}'s rows are {sb} bytes, and its payload gave {len(data)} for {len(got)}")
                    uses[slot].append((reach, arrive[r] + max(x.born for x in got), last, codec.digest(data[off:off + codec.BLOCK_BYTES])))
    for r, p, slot, first, last in cached:
        n = at.get((p, slot))
        reach, w, held, d = uses[slot][n] if n is not None else (None, None, None, None)
        if w is None or first <= w:
            raise ValueError(f"request {r} reads cached slot {slot} from step {first}, a block its producer {p} "
                             f"{'never completes there' if w is None else f'completes at step {w}'}")
        uses[slot][n] = (reach, w, max(held, last), d)
    for slot, us in uses.items():
        held = -1
        for n, (reach, w, last, d) in enumerate(sorted(us, key=lambda u: u[0])):
            if reach <= held:
                raise ValueError(f"slot {slot} is written at step {reach} while a request's rows in it are read to step {held}")
            if n == 0:
                filler.setdefault(reach, []).append(slot)
            held = last
            if w is not None and w < last:
                done.setdefault(w, []).append((slot, d, last))
    head = Header(registration, layout.blocks if layout.blocks is not None else len(pool.firsts) * pool.region)
    steps = _lines(filler, _appends(done, deferral, head.blocks), head.blocks)
    if any(x >= SLOT_LIMIT for st in steps for x in (*st.freed, *(w.slot for w in st.written))):
        raise ValueError(f"a slot past memory-map/v1's {SLOT_LIMIT}")
    return head, steps


def _appends(done: Mapping[int, list[tuple[int, bytes, int]]], deferral: Deferral | None, n: int) -> list[_Append]:
    """The session's appends, each step's completed blocks `done[step]` by slot: without `deferral` landing at once, else
    as its encodes have them, which must be the circuit's in order from index `n` up to the last made (the circuit's after
    it the session never made); a re-key's blocks are taken when its line closes (`_lines`)."""
    if deferral is None:
        return [_Append(w, w, len(done[w]), False, sorted(done[w])) for w in sorted(done)]
    es = [(m - deferral.first_step, None if r is None else r - deferral.first_step, f, c) for m, r, f, c in deferral.encodes]
    out, j, last = [], n, -1
    for u, (m, r, f, c) in enumerate(es):
        whole = m in done and (u + 1 == len(es) or es[u + 1][0] != m)
        prev = out[-1].landed if out else 0
        if f != j or c < 1 or m < max(last, 0 if whole else 1) or (r is not None and (prev is None or r < max(prev, m))):
            raise ValueError(f"the session's append {u}, {list(deferral.encodes[u])}, is no append of the store's next blocks from {j}, "
                             "made at a step of the session in step order (a re-key after its first) and landing in append order, "
                             "none before it is made")
        if whole and c != len(done[m]):
            raise ValueError(f"the session's append at step {m} is of {c} blocks, and the circuit completed {len(done[m])} there")
        out.append(_Append(m, r, c, not whole, sorted(done[m]) if whole else []))
        j, last = j + c, m
    missed = [w for w in done if w <= last and not any(a.made == w and not a.rekey for a in out)]
    if missed:
        raise ValueError(f"the session made no append of the blocks the circuit completed at steps {sorted(missed)[:8]}")
    return out


def _lines(filler: Mapping[int, list[int]], appends: list[_Append], n: int) -> list[Step]:
    """The record's lines (`Lines`, from index `n`) from the slots whose filler a write first reaches at each step and the
    session's `appends`, step by step: each step's frees, its re-key (of the next step's append: the live blocks of the oldest
    appends, read past the step), the appends it makes, then the writes of those landed by then."""
    lines, landing, out, k, t, end = Lines(n), deque(), [], 0, 0, max(filler, default=-1)
    while t <= end or k < len(appends) or landing or lines.busy:
        lines.free(filler.get(t, ()))
        while k < len(appends) and appends[k].made == (t + 1 if appends[k].rekey else t):
            a, k = appends[k], k + 1
            if a.rekey:
                old = sorted(lines.held(t).items(), key=lambda xb: xb[1].index)[:a.count]
                if len(old) < a.count:
                    raise ValueError(f"the session re-keys {a.count} blocks at step {t}, and the circuit holds {len(old)} live")
                made = lines.rekey(sorted(x for x, _ in old))
            else:
                made = lines.append(*zip(*a.blocks))
            if a.landed is not None:
                landing.append((a.landed, made.first))
        while landing and landing[0][0] <= t:
            lines.land(landing.popleft()[1])
        if (line := lines.close(t)) is not None:
            out.append(line)
        t += 1
    return out


def engine_agrees(requests: Mapping[str, Rows], layout: Layout, tables: Mapping[str, Sequence[int]], steps: Sequence[Step],
                  block: Callable[[int, int, int], bytes]) -> dict[str, int]:
    """How many of the session's pages' blocks the engine's KV holds as the map's last write of each left them: `block(layer,
    page, c)` the bytes the engine's cache of layer `layer` holds at page `page`'s c-th 4 KiB (a page's planes in order).  A
    block written again outside the session, or rewritten in part by a later request (in no step), differs."""
    pool, last = _pool(requests, layout, tables), {w.slot: w.digest for st in steps for w in st.written}
    got = [codec.digest(block(layer, page, c)) == last[s] for page in sorted({p for t in tables.values() for p in t})
           for layer in range(len(pool.firsts)) for c in range(pool.ids.shape[1]) if (s := pool.slot(layer, page, c)) in last]
    return {"compared": len(got), "equal": sum(got)}
