"""memory-map/v1 from the circuit (`verity.core.protocols.remote_bandwidth.memory_map`): the KV blocks each serving step wrote
and freed, from the Program of record and the engine's page layout, with each block's digest from the values served (the
Commit's committed words, `pipeline.memory_map`).

~~~text
row       one position's K (or V) in a layer: a row of an attention Call's `kc` (`vc`), `Array(T, row)`; a layer's rows are
          the rows its Calls share, in the order its Calls first read them (its positions), K and V paired by position
born      a row's step, and `last` its last reader's: the circuit's requirement (`requirement.from_census`) over the
          Program's census at its sampling events (`query.required.SAMPLING_EVENTS`)
page      `block_size` slots in each layer, the K plane then the V plane, a slot one row (vLLM's NHD); a request's positions
          fill its pages in order (`tables`; by default each request's pages in order of first need from page 1, vLLM's
          null block being page 0)
block     4,096 bytes of a plane (`store.registry._page_blocks`), at slot `firsts[layer] + ids[page, c]` (the `Store`'s
          numbering: a KV pool of one region per layer)
written   a block whose rows are all born, at the step of the last; a block with a row no step writes is in no step (it
          stays dirty)
freed     a written block at the step after its rows' last read, before that step's writes; one past the session's last
          step is in no step
j         N plus every block written before it, a step's blocks by slot
d_j       `codec.digest` of the block's payload: its rows' served words in slot order, each word little-endian
step      the engine's: request r's step s is `arrive[r] + s`
~~~

The map is the prover's witness (d_j hashes user data): it goes to the prover's own file and is never published.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

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
    """A request Program's KV rows, ``layers[l][plane][position]`` (plane 0 K, 1 V), each `row_bytes` bytes of `word_bytes`-byte
    words, over its `steps` steps."""

    layers: tuple[tuple[tuple[Row, ...], tuple[Row, ...]], ...]
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


def rows(program: Program) -> Rows:
    """The KV rows of a request Program, born and last read as its requirement at its sampling events has them."""
    body = program.root.body
    inputs = sum(1 for nd in body.nodes if is_input(nd.fn))
    c = census(program, step_ends(program, SAMPLING_EVENTS))
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


def tables_of(requests: Mapping[str, Rows], arrive: Mapping[str, int], block_size: int) -> dict[str, list[int]]:
    """Each request's pages in order of first need (the engine step that writes a page's first position, then request order),
    from the page after the null block."""
    need = sorted((arrive[r] + rs.layers[0][0][q * block_size].born, i, q, r)
                  for i, (r, rs) in enumerate(requests.items()) for q in range(-(-rs.positions // block_size)))
    out: dict[str, list[int]] = {r: [NULL_PAGE] * -(-rs.positions // block_size) for r, rs in requests.items()}
    for page, (_, _, q, r) in enumerate(need, start=NULL_PAGE + 1):
        out[r][q] = page
    return out


def record(requests: Mapping[str, Rows], arrive: Mapping[str, int], payload: Callable[[str, Row], bytes], registration: bytes,
           layout: Layout = Layout(), tables: Mapping[str, Sequence[int]] | None = None) -> tuple[Header, list[Step]]:
    """The session's memory map beside the registered store `registration` digests: each request's rows from its arrival step,
    a row's served bytes `payload(request, row)`, its pages `tables[request]` (by default `tables_of`)."""
    shapes = {(len(rs.layers), rs.row_bytes) for rs in requests.values()}
    if len(shapes) != 1:
        raise ValueError(f"the requests' KV differs in (layers, row bytes): {sorted(shapes)}")
    ((nl, sb),) = shapes
    bs = layout.block_size
    tables = dict(tables) if tables is not None else tables_of(requests, arrive, bs)
    pages = layout.pages if layout.pages is not None else 1 + max(max(t, default=NULL_PAGE) for t in tables.values())
    ids, lo, hi = _page_blocks(PageLayout(pages, bs, sb), codec.BLOCK_BYTES)
    per_plane, region = ids.shape[1] // 2, int(ids.max()) + 1
    firsts = layout.firsts if layout.firsts is not None else tuple(layer * region for layer in range(nl))
    if len(firsts) != nl:
        raise ValueError(f"{len(firsts)} layer regions for {nl} layers of KV")
    end = max(arrive[r] + rs.steps for r, rs in requests.items())
    freed: dict[int, list[int]] = {}
    written: dict[int, list[tuple[int, bytes]]] = {}
    for r, rs in requests.items():
        if len(tables[r]) * bs < rs.positions or any(not NULL_PAGE < p < pages for p in tables[r]):
            raise ValueError(f"request {r}'s pages {list(tables[r])[:8]} don't hold its {rs.positions} positions in pages 1 to {pages - 1}")
        for q, page in enumerate(tables[r]):
            for c in range(ids.shape[1]):
                plane, i = divmod(c, per_plane)
                p0, p1 = q * bs + int(lo[c]), q * bs + int(hi[c])
                if p1 > rs.positions:
                    continue
                off = i * codec.BLOCK_BYTES - int(lo[c]) * sb
                for layer, kv in enumerate(rs.layers):
                    got = kv[plane][p0:p1]
                    data = b"".join(payload(r, x) for x in got)
                    if len(data) != len(got) * sb:
                        raise ValueError(f"request {r}'s rows are {sb} bytes, and its payload gave {len(data)} for {len(got)}")
                    slot = firsts[layer] + int(ids[page, c])
                    written.setdefault(arrive[r] + max(x.born for x in got), []).append(
                        (slot, codec.digest(data[off:off + codec.BLOCK_BYTES])))
                    if (f := arrive[r] + max(x.last for x in got) + 1) < end:
                        freed.setdefault(f, []).append(slot)
    steps, j = [], layout.blocks if layout.blocks is not None else nl * region
    head = Header(registration, j)
    for s in sorted(set(freed) | set(written)):
        ws = sorted(written.get(s, ()))
        steps.append(Step(s, tuple(sorted(freed.get(s, ()))), tuple(Written(x, j + u, d) for u, (x, d) in enumerate(ws))))
        j += len(ws)
    if any(x >= SLOT_LIMIT for st in steps for x in (*st.freed, *(w.slot for w in st.written))):
        raise ValueError(f"a slot past memory-map/v1's {SLOT_LIMIT}")
    return head, steps
