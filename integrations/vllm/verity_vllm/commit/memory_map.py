"""memory-map/v1 from the circuit (`verity.core.protocols.remote_bandwidth.memory_map`): the KV blocks each serving step wrote
and freed, from the Program of record and the engine's page layout, with each block's digest from the values served (the
Commit's committed words, `pipeline.memory_map`).

~~~text
row       one position's K (or V) in a layer: a row of an attention Call's `kc` (`vc`), `Array(T, row)`; a layer's rows are
          the rows its Calls share, in the order its Calls first read them (its positions), K and V paired by position
born      a row's step, and `last` its last reader's: the circuit's requirement (`requirement.from_census`) over the
          Program's census at its sampling events (`query.required.SAMPLING_EVENTS`)
page      `block_size` slots in each layer as the engine lays them out (`page_of`, from its cache tensor): a K plane then a
          V plane, a slot one row ("planes"), or one plane whose slot holds a position's K and V head by head ("packed",
          `packed`); a request's position q sits at slot `tables[request][q // block_size] * block_size + q % block_size`,
          its pages the ones the engine wrote its KV to (`tables`, from the slots the Commit committed), vLLM's null block
          being page 0
block     4,096 bytes of a plane (`store.registry._page_blocks`), at slot `firsts[layer] + ids[page, c]` (the `Store`'s
          numbering: a KV pool of one region per layer)
written   a block whose rows are all born, at the step of the last; a block with a row no step writes is in no step (it
          stays dirty)
freed     a written block in the record of its rows' last read's step (a step's record, its frees then its writes, takes
          effect after the step's reads: remote_bandwidth's convention, #2378); a block whose last read is the step that
          completes it is freed the step after, and one past the session's last step is in no step
j         N plus every block written before it, a step's blocks by slot
d_j       `codec.digest` of the block's payload: its rows' served words in slot order, each word little-endian
step      the engine's: request r's step s is `arrive[r] + s`
~~~

The map is the prover's witness (d_j hashes user data): it goes to the prover's own file and is never published.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import SLOT_LIMIT, Header, Step, Written
from verity.ml.circuits.census import census, step_ends
from verity.ml.circuits.defs import PrimitiveDefinition
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import runs
from verity.ml.circuits.requirement import from_census
from verity.ml.circuits.types import Array
from verity_vllm.query.required import SAMPLING_EVENTS
from verity_vllm.store.interface import PageLayout
from verity_vllm.store.registry import _page_blocks

#: an attention Call's cached K and V parameters
KV = ("kc", "vc")
#: vLLM's null block, no request's page
NULL_PAGE = 0

#: leaves `[lo, hi)` of the output of node `j` of the Program's function (the Commit's instance rows' `['n', j, lo, hi]`)
Slice = tuple[str, int, int, int]


@dataclass(frozen=True)
class Row:
    """One position's K or V in a layer: the leaves it is, in order, and the steps it is born and last read at."""

    group: tuple[Slice, ...]
    born: int
    last: int


@dataclass(frozen=True)
class Rows:
    """A request Program's KV rows, ``layers[l][plane][position]`` (from `rows`, plane 0 K and 1 V; after `packed`, one plane),
    each `row_bytes` bytes of `word_bytes`-byte words, over its `steps` steps."""

    layers: tuple[tuple[tuple[Row, ...], ...], ...]
    row_bytes: int
    word_bytes: int
    steps: int

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
class Page:
    """One layer's KV page as the engine lays it out: `pages` pages of `block_size` slots, "planes" (a K plane then a V
    plane) or "packed" (a slot holds a position's K and V, `head_words` words of each per head in turn)."""

    form: str
    pages: int
    block_size: int
    head_words: int


def page_of(shape: Sequence[int], stride: Sequence[int], itemsize: int, rs: Rows) -> Page:
    """The page of a layer's KV cache tensor (its shape, strides in elements and element bytes) holding `rs`'s rows: vLLM
    0.28's FlashAttention ``[pages, heads, block_size, 2 * head]`` slot-major (strides ``(bs*H*2D, 2D, H*2D, 1)``) is
    "packed", ``[pages, 2, block_size, heads, head]`` contiguous "planes"; any other layout is refused."""
    words, shape, stride = rs.row_bytes // rs.word_bytes, tuple(map(int, shape)), tuple(map(int, stride))
    if itemsize != rs.word_bytes:
        raise ValueError(f"a KV cache of {itemsize}-byte elements holds rows of {rs.word_bytes}-byte words")
    if len(shape) == 4:
        nb, h, bs, d2 = shape
        if h * d2 == 2 * words and stride == (bs * h * d2, d2, h * d2, 1):
            return Page("packed", nb, bs, d2 // 2)
    if len(shape) == 5:
        nb, two, bs, h, d = shape
        if two == 2 and h * d == words and stride == (2 * bs * h * d, bs * h * d, h * d, d, 1):
            return Page("planes", nb, bs, d)
    raise ValueError(f"a KV cache {list(shape)} at strides {list(stride)} is no page memory-map/v1 lays out rows of {words} words in")


def packed(rs: Rows) -> Rows:
    """`rs` with a layer's K and V rows of a position as one row (its group the K row's leaves, then the V row's), the one
    plane of a "packed" page; its payload interleaves them head by head."""
    return replace(rs, row_bytes=2 * rs.row_bytes, layers=tuple(
        (tuple(Row(k.group + v.group, max(k.born, v.born), max(k.last, v.last)) for k, v in zip(*kv)),) for kv in rs.layers))


def _cut(refs, width: int, inputs: int, what: str) -> list[tuple[Slice, ...]]:
    """A `kc` or `vc` argument's rows of `width` leaves, each as the function's slices it is."""
    out, cur, need = [], [], width
    for space, k, lo, hi in runs(refs):
        if space != "n" or k < inputs:
            raise ValueError(f"{what} reads an input of the Program: memory-map/v1 records the KV a step computes")
        while lo < hi:
            take = min(need, hi - lo)
            cur.append(("n", k - inputs, lo, lo + take))
            lo, need = lo + take, need - take
            if not need:
                out.append(tuple(cur))
                cur, need = [], width
    return out


def census_of(program: Program, digest: str | None = None) -> dict:
    """The census/v1 of a request Program over its steps, one per sampling event, naming it by `digest`."""
    return census(program, step_ends(program, SAMPLING_EVENTS), digest)


def rows(program: Program, c: Mapping | None = None) -> Rows:
    """The KV rows of a request Program, born and last read as its requirement over its census `c` (`census_of`, computed when
    not given) has them."""
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
        for _, j, lo, hi in group:
            got = [(a, b, v) for a, b, v in by_node.get(j + inputs, ()) if a < hi and b > lo]
            if sum(min(b, hi) - max(a, lo) for a, b, _ in got) != hi - lo:
                raise ValueError(f"leaves [{lo}, {hi}) of node {j} are no census value's: a KV row must be its producer's output")
            vs += [v for _, _, v in got]
        born = max(req.born[v] for v in vs)
        return Row(group, born, max(req.last.get(v, born) for v in vs))

    layer_of: dict[tuple[Slice, ...], int] = {}
    pair: dict[tuple[Slice, ...], tuple[Slice, ...]] = {}
    layers: list[list[tuple[Slice, ...]]] = []
    kind = None
    for k, nd in enumerate(body.nodes):
        if is_input(nd.fn) or isinstance(nd.fn, PrimitiveDefinition):
            continue
        names = [n for n, _ in nd.fn.signature()[0]]
        if not set(KV) <= set(names):
            continue
        kc, vc = (nd.args[names.index(p)] for p in KV)
        t = kc.type
        if nd.form != "call" or not isinstance(t, Array) or vc.type != t or t.elem.homogeneous_width() is None:
            raise ValueError(f"root node {k} ({nd.fn.id}) holds its KV otherwise than as one Call's kc, vc: Array(T, row) each")
        here = (t.elem.leaves, t.elem.homogeneous_width())
        if kind not in (None, here):
            raise ValueError(f"root node {k} ({nd.fn.id}) has rows of {here} (leaves, bits), another layer's are {kind}")
        kind = here
        ks, vs = (_cut(a.refs, here[0], inputs, f"root node {k}'s {p}") for a, p in zip((kc, vc), KV))
        seen = {layer_of[g] for g in ks if g in layer_of}
        if len(seen) > 1:
            raise ValueError(f"root node {k} ({nd.fn.id}) reads the K rows of layers {sorted(seen)}")
        layer = seen.pop() if seen else len(layers)
        if layer == len(layers):
            layers.append([])
        for kg, vg in zip(ks, vs):
            if pair.setdefault(kg, vg) != vg:
                raise ValueError(f"root node {k} ({nd.fn.id}) reads a K row with another V row than a Call before it")
            if kg not in layer_of:
                layer_of[kg] = layer
                layers[layer].append(kg)
    if kind is None:
        raise ValueError("the Program has no attention Call with kc and vc: no KV to record")
    leaves, bits = kind
    if bits % 8:
        raise ValueError(f"a KV word of {bits} bits is no whole bytes")
    return Rows(tuple((tuple(row(g) for g in ks), tuple(row(pair[g]) for g in ks)) for ks in layers),
                leaves * bits // 8, bits // 8, req.steps)


def rows_doc(rs: Rows) -> dict[str, Any]:
    """`rs` as JSON: a row as ``[born, last, [[node, lo, hi], ...]]``."""
    return {"row_bytes": rs.row_bytes, "word_bytes": rs.word_bytes, "steps": rs.steps,
            "layers": [[[[x.born, x.last, [[j, lo, hi] for _, j, lo, hi in x.group]] for x in plane] for plane in kv] for kv in rs.layers]}


def rows_of(doc: Mapping[str, Any]) -> Rows:
    """The `Rows` `rows_doc` wrote."""
    return Rows(tuple(tuple(tuple(Row(tuple(("n", j, lo, hi) for j, lo, hi in g), b, l) for b, l, g in plane) for plane in kv)
                      for kv in doc["layers"]), int(doc["row_bytes"]), int(doc["word_bytes"]), int(doc["steps"]))


def tables(requests: Mapping[str, Rows], arrive: Mapping[str, int], slots: Callable[[str, int], Sequence[int]],
           block_size: int) -> dict[str, list[int]]:
    """Each request's pages as the engine wrote its KV: `slots(request, step)` the slot of each row it computed at that engine
    step, in position order (vLLM's slot mapping, position q at ``page * block_size + q % block_size``).  The rows a step wrote
    are the ones its Program writes there, else it is refused (a prefix-cache hit writes fewer: another request wrote them, and
    GP-01 models no prefix caching), and so is a page that isn't one page throughout or a request holding a page twice."""
    out: dict[str, list[int]] = {}
    for r, rs in requests.items():
        born, pages, q = [x.born for x in rs.layers[0][0]], [NULL_PAGE] * -(-rs.positions // block_size), 0
        for s in range(rs.steps):
            got, n = [int(v) for v in slots(r, arrive[r] + s)], born.count(s)
            if len(got) != n or born[q:q + n] != [s] * n:
                raise ValueError(f"request {r} wrote {len(got)} KV rows at step {arrive[r] + s}, where its Program writes {n} from position {q}")
            for p, slot in enumerate(got, start=q):
                page, off = divmod(slot, block_size)
                if off != p % block_size or page == NULL_PAGE or pages[p // block_size] not in (NULL_PAGE, page):
                    raise ValueError(f"request {r}'s position {p} went to slot {slot} (page {page}, offset {off}), its page "
                                     f"{p // block_size} being {pages[p // block_size]}")
                pages[p // block_size] = page
            q += n
        if q != rs.positions or len(set(pages)) != len(pages):
            raise ValueError(f"request {r}'s {rs.positions} positions took pages {pages[:8]}{'...' if len(pages) > 8 else ''} over {q} rows")
        out[r] = pages
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
           layout: Layout, tables: Mapping[str, Sequence[int]]) -> tuple[Header, list[Step]]:
    """The session's memory map beside the registered store `registration` digests: each request's rows from its arrival step,
    a row's served bytes `payload(request, row)`, its pages `tables[request]` (`tables`)."""
    pool = _pool(requests, layout, tables)
    ids, lo, hi, sb, pages, bs = pool.ids, pool.lo, pool.hi, pool.row_bytes, pool.pages, layout.block_size
    end = max(arrive[r] + rs.steps for r, rs in requests.items())
    freed: dict[int, list[int]] = {}
    written: dict[int, list[tuple[int, bytes]]] = {}
    for r, rs in requests.items():
        if len(tables[r]) * bs < rs.positions or any(not NULL_PAGE < p < pages for p in tables[r]):
            raise ValueError(f"request {r}'s pages {list(tables[r])[:8]} don't hold its {rs.positions} positions in pages 1 to {pages - 1}")
        for q, page in enumerate(tables[r]):
            for c in range(ids.shape[1]):
                plane, i = divmod(c, pool.per_plane)
                p0, p1 = q * bs + int(lo[c]), q * bs + int(hi[c])
                if p1 > rs.positions:
                    continue
                off = i * codec.BLOCK_BYTES - int(lo[c]) * sb
                for layer, kv in enumerate(rs.layers):
                    got = kv[plane][p0:p1]
                    data = b"".join(payload(r, x) for x in got)
                    if len(data) != len(got) * sb:
                        raise ValueError(f"request {r}'s rows are {sb} bytes, and its payload gave {len(data)} for {len(got)}")
                    slot = pool.slot(layer, page, c)
                    w = arrive[r] + max(x.born for x in got)
                    written.setdefault(w, []).append((slot, codec.digest(data[off:off + codec.BLOCK_BYTES])))
                    if (f := max(arrive[r] + max(x.last for x in got), w + 1)) < end:
                        freed.setdefault(f, []).append(slot)
    steps, j = [], layout.blocks if layout.blocks is not None else len(pool.firsts) * pool.region
    head = Header(registration, j)
    for s in sorted(set(freed) | set(written)):
        ws = sorted(written.get(s, ()))
        steps.append(Step(s, tuple(sorted(freed.get(s, ()))), tuple(Written(x, j + u, d) for u, (x, d) in enumerate(ws))))
        j += len(ws)
    if any(x >= SLOT_LIMIT for st in steps for x in (*st.freed, *(w.slot for w in st.written))):
        raise ValueError(f"a slot past memory-map/v1's {SLOT_LIMIT}")
    return head, steps


def engine_agrees(requests: Mapping[str, Rows], layout: Layout, tables: Mapping[str, Sequence[int]], steps: Sequence[Step],
                  block: Callable[[int, int, int], bytes]) -> dict[str, int]:
    """How many of the session's pages' blocks the engine's KV holds as the map's last write of each left them: `block(layer,
    page, c)` the bytes the engine's cache of layer `layer` holds at page `page`'s c-th 4 KiB (a page's planes in order).  A
    block written again outside the session, or rewritten in part by a later request (in no step), differs."""
    pool, last = _pool(requests, layout, tables), {w.slot: w.digest for st in steps for w in st.written}
    got = [codec.digest(block(layer, page, c)) == last[s] for page in sorted({p for t in tables.values() for p in t})
           for layer in range(len(pool.firsts)) for c in range(pool.ids.shape[1]) if (s := pool.slot(layer, page, c)) in last]
    return {"compared": len(got), "equal": sum(got)}
