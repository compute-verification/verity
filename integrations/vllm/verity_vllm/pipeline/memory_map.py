"""The Commit's memory map (opt-in, ``commit --memory-map REQUEST_JSON``): memory-map/v1 of the run's KV, written in the
in-process C2 replay once the Commit's roots are final.  Its blocks, their steps and store indices come from each request's
Program of record (`commit.memory_map.rows`: the KV rows the Build wrote beside its ``descriptor.json.gz``, over the census it
wrote there, `cache`; a Build without them has them computed once and cached, `of_build`) at the pages the engine wrote: each
request's slots as the Commit committed them (`<layer>/slot_mapping`, `acquire.sources.kv_slots_source`, which the flag
attaches), in the page layout of the engine's cache tensors.  Each d_j is from the committed K and V words of the block's rows,
read as the serving rows read a Program's slices (`serving_rows._operand`: the ProgramIndex's members, the binding map's
committed words).  A warm request (a prefix-cache hit) commits slots from its cached prefix on: its Program's first C rows are
cached positions (the Build's ``prefix_cache.json``), whose pages are the ones its producers (the replay's ``cached_prefixes``)
wrote.  A request whose slots weren't committed is refused by name: no page is assumed.  A Commit whose run a remote
session audited (``--remote-verifier``) has written the session's record, ``remote_session.json``, by then, and the map
follows its deferred residency (`commit.memory_map.Deferral`, `deferral`): each block written where its append's encode landed.
A TP Commit (``tp-commit --memory-map``) writes one map a rank, in the rank's replay (`pipeline.tp.rank_replay.memory_map`),
from the rank's Programs, committed slots and caches: MLA's latent cache is whole on every rank.

The request (``REQUEST_JSON``)::

    {"path": <the map's file>, "registration": <hex of the remote audit's Registration.digest>,
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

import gzip
import json
import re
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth import codec
from verity.core.protocols.remote_bandwidth.memory_map import dumps
from verity_vllm.acquire.sources.kv_slots_source import MEMBER
from verity_vllm.check.replay.cached_prefix import layout_of
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import memory_map as MM
from verity_vllm.pipeline.serving_rows import _operand


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


def deferral(path: str | Path) -> MM.Deferral | None:
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
    return MM.Deferral(tuple((int(m), None if r is None else int(r), int(f), int(c)) for m, r, f, c in mm["deferral"]["encodes"]),
                       int(mm["first_step"]), int(rec["prover"]["kv"]["lifetime"]))


#: beside a request Build's ``descriptor.json.gz``: its Program's census (`MM.census_of`) and KV rows (`MM.rows_doc`), each naming
#: the Program by digest; KV rows of another schema are recomputed (v1 refused MLA's latent rows, v2 a warm Program's cached prefix)
CENSUS, KV_ROWS = "census.json.gz", "kv_rows.json.gz"
KV_ROWS_SCHEMA = "verity-vllm/kv-rows/v3"


def _read(path: Path, digest: str, schema: str | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as f:
        doc = json.load(f)
    return doc if doc.get("program") == digest and schema in (None, doc.get("schema")) else None


def _put(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(doc, f, separators=(",", ":"))
    tmp.replace(path)


def cache(d: str | Path, program: Any, digest: str, build: str | Path | None = None) -> dict[str, Any]:
    """Write Build dir ``d``'s `CENSUS` and `KV_ROWS` for its request Program ``program`` (``digest``), the census read back when
    one beside it already names the Program, a warm Program's cached prefix as the Build (``build``, else ``d``) laid it out
    (``prefix_cache.json``); a Program with no KV a memory map lays out gets the reason in place of its rows."""
    d = Path(d)
    c = _read(d / CENSUS, digest)
    if c is None:
        c = MM.census_of(program, digest)
        _put(d / CENSUS, c)
    doc: dict[str, Any] = {"schema": KV_ROWS_SCHEMA, "program": digest, "census": c["digest"]}
    try:
        doc.update(MM.rows_doc(MM.rows(program, c, layout_of(str(d if build is None else build)))))
    except ValueError as e:
        doc["refused"] = str(e)
    _put(d / KV_ROWS, doc)
    return {"census": c["digest"], "kv_rows": "refused" not in doc}


def census_entry(d: str | Path, program: Any, digest: str) -> dict[str, Any]:
    """`cache`'s result for the Build's `result.json`, or its error in its place: the census serves the Commit's opt-in
    memory map, so it decides no Build."""
    try:
        return cache(d, program, digest)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}


def of_build(d: str | Path, digest: str, at: str | Path | None = None) -> MM.Rows:
    """The KV rows of Build dir ``d``'s request Program ``digest``: its `KV_ROWS` when they name the Program, else computed from its
    descriptor (refused unless that is Program ``digest``) and cached beside it (`cache`), once per Program digest; read and
    cached at ``at`` in place of ``d`` when given, ``d`` only read."""
    from verity.ml.definitions.boolean import load_program

    d = Path(d)
    at = d if at is None else Path(at)
    doc = _read(at / KV_ROWS, digest, KV_ROWS_SCHEMA)
    if doc is None:
        program, got = load_program(d / "descriptor.json.gz")
        if got != digest:
            raise ValueError(f"{d}: descriptor.json.gz is Program {got[:16]}, its instance rows {str(digest)[:16]}")
        at.mkdir(parents=True, exist_ok=True)
        cache(at, program, digest, d)
        doc = _read(at / KV_ROWS, digest, KV_ROWS_SCHEMA)
    if "refused" in doc:
        raise ValueError(f"{d}: {doc['refused']}")
    return MM.rows_of(doc)


def _layer(name: str) -> int:
    m = re.search(r"\.(\d+)(\.|$)", name)
    if m is None:
        raise ValueError(f"KV layer {name!r} names no layer index")
    return int(m.group(1))


def write(out: Path, req: Mapping[str, Any], *, loaded: Mapping[str, Mapping[str, Any]], store: Any, arrivals: Mapping[str, int],
          caches: Mapping[str, Any] | None, deferral: MM.Deferral | None = None, tp: Mapping[str, int] | None = None,
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
    by_dir: dict[str, tuple[MM.Rows, ProgramIndex, str]] = {}
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

    def payload(rid: str, row: MM.Row) -> bytes:
        w = np.ascontiguousarray(_operand(index[rid], store, None, rid, int(arrivals[rid]), row.group), dtype=f"<u{rows[rid].word_bytes}")
        return np.ascontiguousarray(w.reshape(2, -1, page.head_words).transpose(1, 0, 2) if page.form == "packed" else w).tobytes()

    layout = MM.KvPool(page.block_size, page.pages, tuple(req["firsts"]) if req.get("firsts") is not None else None,
                       int(req["blocks"]) if req.get("blocks") is not None else None)
    head, steps = MM.record(kv, {r: int(arrivals[r]) for r in rows}, payload, bytes.fromhex(req["registration"]), layout, tables, deferral,
                            producers)
    out.write_text(dumps(head, steps))
    t_d = time.perf_counter()
    engine = MM.engine_agrees(kv, layout, tables, steps, _engine_blocks([caches[n] for n in layers], tables))
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
