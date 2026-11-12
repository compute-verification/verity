"""The Commit's memory map (opt-in, ``commit --memory-map REQUEST_JSON``): memory-map/v1 of the run's KV
(`verity.protocols.poub.verifier.memory_map`), written in the in-process C2 replay once the Commit's roots are final.
Its blocks, their steps and store indices come from each request's KV rows (`commit.memory_map.rows`, the Build's,
`pipeline.memory_map.of_build`) at the pages the engine wrote: each request's slots as the Commit committed them
(`<layer>/slot_mapping`, `acquire.sources.kv_slots_source`, which the flag attaches), in the page layout of the engine's cache
tensors (`commit.memory_map.page_of`, `tables`).  Each d_j is from the committed K and V words of the block's rows, read as the
serving rows read a Program's slices (`serving_rows._operand`: the ProgramIndex's members, the binding map's committed words),
as `Lines` builds it (`protocols.remote_bandwidth.memory_map`: j and the rules).  A request whose slots weren't committed is
refused by name: no page is assumed.  A Commit whose run a remote session audited (``--remote-verifier``) has written the
session's record, ``remote_session.json``, by then, and the map follows its deferred residency (`Deferral`, `deferral`): each
block written where its append's encode landed.  A TP Commit (``tp-commit --memory-map``) writes one map a rank, in the rank's
replay (`pipeline.tp.rank_replay.memory_map`), from the rank's Programs, committed slots and caches: MLA's latent cache is
whole on every rank.  Serving loads this module by name (`verity_vllm.protocols.driver`).

~~~text
block     4,096 bytes of a plane (`store.registry._page_blocks`), at slot `firsts[layer] + ids[page, c]` (the `NodeMemory`'s
          numbering: a KV pool of one region per layer)
appended  a block whose rows are all born, at the step of the last (its completion), if a later step reads it: a step's
          record, its frees then its writes, takes effect after the step's reads (remote_bandwidth's convention),
          so a block whose last structural reader is the step that completes it is never read while resident and is in
          no record; nor is a block with a row no step writes (it stays dirty).  A step's blocks are one append, by slot
written   an appended block, where its append's encode landed (the session's `Deferral`; without one, at its completion)
freed     a written block at its rows' last structural reader (memory-map/v1's free).  And a slot's registered block (a KV
          page's filler, registered before the seed) at the step a write first reaches it, even the one completing it: its
          bytes are no longer that block's, and a block not yet written is uncredited (the `NodeMemory`'s dirty).  A page's
          untouched blocks stay held, counting against capacity, until then: a request's end frees only the blocks it wrote
re-key    a live written block as old as the session's lifetime cap, its old j freed in the record of the step before its
          append's.  Which appends are re-keys is the session's (`Deferral`); the blocks they take are the live written
          ones of the oldest appends (a window is a prefix of them), not freed at that step
d_j       `codec.digest` of the block's payload: its rows' served words in slot order, each word little-endian
~~~

The request (``REQUEST_JSON``)::

    {"path": <the map's file>, "registration": <hex of PoUB's Registration.digest>,
     "firsts"?: [<layer l's region's first block>], "blocks"?: <N>,
     "cache_dir"?: <where each Program's census and KV rows are read and cached, at <cache_dir>/<Program digest>, its Build dir only read>,
     "ranks"?: {"<r>": {<any of the above, for TP rank r>}}}

On a TP rank the request is `at_rank`'s: ``ranks[r]`` over the rest, its map at ``<stem>.rank<r><suffix>`` unless that names one.
``layout`` beside the map (``<path>.json``) is what the map's structure took (the engine's page, each request's pages) and
what it holds, in counts, with how many of its blocks the engine's caches still hold as the map's last write left them; the
map itself is the prover's witness (d_j hashes the served values) and is never published.  A failure is written beside the
map (``<path>.error.txt``) and changes no verdict.
"""
from __future__ import annotations

import json
import re
import time
import traceback
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.protocols.poub.verifier import codec
from verity.protocols.poub.verifier.memory_map import SLOT_LIMIT, Header, Step, dumps
from verity_vllm.acquire.sources.kv_slots_source import MEMBER
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import memory_map as MM
from verity_vllm.commit.memory_map import NULL_PAGE, Row, Rows, _producers
from verity_vllm.pipeline.memory_map import of_build
from verity_vllm.pipeline.serving_rows import _operand
from verity_vllm.protocols.remote_bandwidth.interface import PageLayout
from verity_vllm.protocols.remote_bandwidth.memory_map import Lines
from verity_vllm.protocols.remote_bandwidth.registry import _page_blocks


@dataclass(frozen=True)
class KvPool:
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


def _pool(requests: Mapping[str, Rows], layout: KvPool, tables: Mapping[str, Sequence[int]]) -> _Pool:
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
           layout: KvPool, tables: Mapping[str, Sequence[int]], deferral: Deferral | None = None,
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


def engine_agrees(requests: Mapping[str, Rows], layout: KvPool, tables: Mapping[str, Sequence[int]], steps: Sequence[Step],
                  block: Callable[[int, int, int], bytes]) -> dict[str, int]:
    """How many of the session's pages' blocks the engine's KV holds as the map's last write of each left them: `block(layer,
    page, c)` the bytes the engine's cache of layer `layer` holds at page `page`'s c-th 4 KiB (a page's planes in order).  A
    block written again outside the session, or rewritten in part by a later request (in no step), differs."""
    pool, last = _pool(requests, layout, tables), {w.slot: w.digest for st in steps for w in st.written}
    got = [codec.digest(block(layer, page, c)) == last[s] for page in sorted({p for t in tables.values() for p in t})
           for layer in range(len(pool.firsts)) for c in range(pool.ids.shape[1]) if (s := pool.slot(layer, page, c)) in last]
    return {"compared": len(got), "equal": sum(got)}


def after_commit(request: str | None, *, loaded: Mapping[str, Mapping[str, Any]], bmap: dict[str, Any], committed: Callable,
                 manifest: Any, arrivals: Mapping[str, int], caches: Mapping[str, Any] | None,
                 session: str | Path | None = None, tp: Mapping[str, int] | None = None,
                 attention_paths: Sequence[str] = (), indices: dict | None = None,
                 index_kw: Mapping[str, Any] | None = None,
                 cached_prefixes: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """Write the run's memory map as ``request`` asks, from the requests' loaded Programs of record (`c2_replay.programs_of_record`)
    at their engine arrival steps and the KV-slots source's `caches`, under the deferral of the remote session whose record is at
    ``session`` if there is one; on TP rank ``tp`` (``{"rank", "world"}``) its rank's map (`at_rank`), the rank Programs read
    with the implementation paths the Build observed (``attention_paths``), their indices built under ``index_kw`` and shared with
    the replay's ``indices`` (`write`); each warm request's cached pages its producers' (``cached_prefixes``,
    `c2_replay.cached_prefixes`).  None without a request.  A failure is written beside the map and returned, never raised."""
    if not request:
        return None
    req = json.loads(Path(request).read_text())
    if tp and int(tp.get("world", 1)) > 1:
        req = at_rank(req, int(tp["rank"]))
    out = Path(req["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return write(out, req, loaded=loaded, store=CommittedStore.of_record(bmap, committed, manifest if isinstance(manifest, dict) else None),
                     arrivals=arrivals, caches=caches, deferral=deferral(session) if session is not None else None, tp=tp,
                     attention_paths=attention_paths, indices=indices, index_kw=index_kw, cached_prefixes=cached_prefixes)
    except Exception as e:  # noqa: BLE001
        out.with_name(out.name + ".error.txt").write_text(traceback.format_exc())
        print(f"[memory-map] FAILED (no verdict changes): {e!r}"[:400], flush=True)
        return {"error": repr(e)}


def at_rank(req: Mapping[str, Any], rank: int) -> dict[str, Any]:
    """TP rank ``rank``'s request: ``req``'s ``ranks[rank]`` over its other entries, its path ``<stem>.rank<r><suffix>`` unless
    ``ranks[rank]`` names one."""
    p = Path(req["path"])
    own = (req.get("ranks") or {}).get(str(rank)) or {}
    return {k: v for k, v in req.items() if k != "ranks"} | {"path": str(p.with_name(f"{p.stem}.rank{rank}{p.suffix}"))} | dict(own)


#: the Commit's remote session record (`store.remote.CommitSession.end`), in its out dir
SESSION = "remote_session.json"


def deferral(path: str | Path) -> Deferral | None:
    """The deferred residency of the remote session whose record is at ``path``: its map's first step and its appends' encodes
    (``memory_map``'s ``first_step`` and ``deferral.encodes``), under its prover's lifetime cap; None without a record or a map
    deferring residency."""
    p = Path(path)
    if not p.is_file():
        return None
    rec = json.loads(p.read_text())
    mm = rec.get("memory_map") or {}
    if mm.get("deferral") is None:
        return None
    return Deferral(tuple((int(m), None if r is None else int(r), int(f), int(c)) for m, r, f, c in mm["deferral"]["encodes"]),
                    int(mm["first_step"]), int(rec["prover"]["kv"]["lifetime"]))


def _layer(name: str) -> int:
    m = re.search(r"\.(\d+)(\.|$)", name)
    if m is None:
        raise ValueError(f"KV layer {name!r} names no layer index")
    return int(m.group(1))


def write(out: Path, req: Mapping[str, Any], *, loaded: Mapping[str, Mapping[str, Any]], store: Any, arrivals: Mapping[str, int],
          caches: Mapping[str, Any] | None, deferral: Deferral | None = None, tp: Mapping[str, int] | None = None,
          attention_paths: Sequence[str] = (), indices: dict | None = None, index_kw: Mapping[str, Any] | None = None,
          cached_prefixes: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The map of ``req`` into ``out`` and its layout beside it: each request's slots and KV words read from ``store`` (a
    `CommittedStore`), its pages laid out as ``caches`` (KV layer name -> the engine's cache tensor) lay them out, a warm
    request's cached pages the ones its producers (``cached_prefixes[request]["producers"]``) wrote, its writes
    where ``deferral`` has their encodes land; the Programs a TP rank's (``tp``) when given, their members spelt as the
    ProgramIndex spells them with ``attention_paths`` and ``index_kw`` (the manifest's query and norm-scale policy,
    `rank_replay.index_query`).  ``indices`` is the replay's memo (`driver.sampled_replay`'s, keyed ``("index", id(instances))``):
    each Program's index is read from it, else built and put in it, since one index of a GLM-5.3 TP8 B8 request Program takes
    minutes."""
    if not caches:
        raise ValueError("no engine KV cache to lay pages out by: the Commit attached no KV-slots source (commit --memory-map attaches it)")
    t0 = time.perf_counter()
    by_dir: dict[str, tuple[Rows, ProgramIndex, str]] = {}
    rows, index = {}, {}
    for rid, lp in loaded.items():
        d, digest = str(lp["dir"]), str(lp["instances"].get("digest"))
        if d not in by_dir:
            at = Path(req["cache_dir"]) / digest if req.get("cache_dir") else None
            key = ("index", id(lp["instances"]))
            pi = (indices or {}).get(key)
            if pi is None:
                pi = ProgramIndex(lp["instances"], attention_paths=list(attention_paths), tp=dict(tp) if tp else None, program_dir=d,
                                  **(index_kw or {}))
                if indices is not None:
                    indices[key] = pi
            by_dir[d] = (of_build(d, digest, at), pi, digest)
        rows[rid], index[rid], _ = by_dir[d]
    t_ir = time.perf_counter()

    layers = sorted(caches, key=_layer)
    if {len(rs.layers) for rs in rows.values()} != {len(layers)}:
        raise ValueError(f"the engine has {len(layers)} KV layers, the Programs {sorted({len(rs.layers) for rs in rows.values()})}")
    pages = {MM.page_of(t.shape, t.stride(), t.element_size(), next(iter(rows.values()))) for t in caches.values()}
    if len(pages) != 1:
        raise ValueError(f"the engine's KV layers lay their pages out differently: {sorted(pages, key=repr)}")
    (page,) = pages

    def slots(rid: str, step: int) -> np.ndarray:
        got = []
        for name in layers:
            es = store.entries(rid, step, name, MEMBER)[0]
            if not es:
                raise ValueError(f"no committed `{name}/{MEMBER}` of request {rid} at step {step}: memory-map/v1 takes each block's page "
                                 "from the slots the Commit committed and assumes none")
            w, why = store.words(rid, step, name, MEMBER, *map(int, es[0]["identity"]["element_range"]))
            if w is None:
                raise ValueError(why)
            got.append(w.view(np.int64))
        if any(not np.array_equal(g, got[0]) for g in got[1:]):
            raise ValueError(f"request {rid}'s KV layers wrote step {step}'s rows to different slots: memory-map/v1 lays out one table a request")
        return got[0]

    producers = {r: [str(q) for q in pc["producers"]] for r, pc in (cached_prefixes or {}).items() if r in rows}
    tables = MM.tables(rows, arrivals, slots, page.block_size, producers)
    kv = {r: MM.packed(rs) for r, rs in rows.items()} if page.form == "packed" else rows

    def payload(rid: str, row: Row) -> bytes:
        w = np.ascontiguousarray(_operand(index[rid], store, None, rid, int(arrivals[rid]), row.group), dtype=f"<u{rows[rid].word_bytes}")
        return np.ascontiguousarray(w.reshape(2, -1, page.head_words).transpose(1, 0, 2) if page.form == "packed" else w).tobytes()

    layout = KvPool(page.block_size, page.pages, tuple(req["firsts"]) if req.get("firsts") is not None else None,
                    int(req["blocks"]) if req.get("blocks") is not None else None)
    head, steps = record(kv, {r: int(arrivals[r]) for r in rows}, payload, bytes.fromhex(req["registration"]), layout, tables, deferral,
                         producers)
    out.write_text(dumps(head, steps))
    t_d = time.perf_counter()
    engine = engine_agrees(kv, layout, tables, steps, _engine_blocks([caches[n] for n in layers], tables))
    doc = {"schema": "verity-vllm/memory-map-layout/v1", "map": out.name, "tp": dict(tp) if tp else None, "blocks": head.blocks,
           "page": {"form": page.form, "pages": page.pages, "block_size": page.block_size, "head_words": page.head_words},
           "requests": {r: {"arrive": int(arrivals[r]), "steps": rs.steps, "positions": rs.positions, "layers": len(rs.layers),
                            "row_bytes": rs.row_bytes, "pages": tables[r], "program_digest": by_dir[str(loaded[r]["dir"])][2],
                            **({"cached": rs.cached, "producers": producers[r]} if rs.cached else {})}
                        for r, rs in rows.items()},
           "deferral": None if deferral is None else {"first_step": deferral.first_step, "appends": len(deferral.encodes),
                                                      "lifetime": deferral.lifetime},
           "steps": len(steps), "written": sum(len(s.written) for s in steps), "freed": sum(len(s.freed) for s in steps), "engine": engine,
           "seconds": {"rows_s": round(t_ir - t0, 3), "digests_s": round(t_d - t_ir, 3), "engine_s": round(time.perf_counter() - t_d, 3)}}
    out.with_name(out.name + ".json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[memory-map] {doc['written']} blocks written and {doc['freed']} freed over {doc['steps']} steps beside {head.blocks} "
          f"registered, {page.form} pages of {page.block_size} slots; the engine holds {engine['equal']} of {engine['compared']} "
          f"as last written: {out} ({doc['seconds']})", flush=True)
    return doc


def _engine_blocks(caches: list[Any], tables: Mapping[str, list[int]]) -> Callable[[int, int, int], bytes]:
    """`block(layer, page, c)` over the engine's cache tensors, layer by layer: the c-th 4 KiB of the page's bytes, each page
    `stride(0)` elements from the last (copied off the device once, the session's pages only)."""
    import torch

    held = sorted({p for t in tables.values() for p in t})
    bytes_of = []
    for t in caches:
        s0 = int(t.stride(0))
        pg = t.as_strided((int(t.shape[0]), s0), (s0, 1))[torch.tensor(held, device=t.device)]
        bytes_of.append(dict(zip(held, pg.contiguous().view(torch.uint8).cpu().numpy())))
    return lambda layer, p, c: bytes_of[layer][p][c * codec.BLOCK_BYTES:(c + 1) * codec.BLOCK_BYTES].tobytes()
