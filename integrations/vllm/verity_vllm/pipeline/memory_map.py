"""The Commit's memory map (opt-in, ``commit --memory-map REQUEST_JSON``): memory-map/v1 of the run's KV, written in the
in-process C2 replay once the vllm-v1 roots are final.  Its blocks, their steps and store indices come from each request's
Program of record (`commit.memory_map.rows`, over the Build's ``descriptor.json.gz``) and the engine's page layout; each d_j
from the committed K and V words of the block's rows, read as the serving rows read a Program's slices (`serving_rows._operand`:
the ProgramIndex's members, the binding map's committed words).

The request (``REQUEST_JSON``)::

    {"path": <the map's file>, "registration": <hex of the remote audit's Registration.digest>,
     "block_size"?: 16, "pages"?: <the pool's pages>, "firsts"?: [<layer l's region's first block>], "blocks"?: <N>,
     "tables"?: {<request>: [<page>, ...]}}

``layout`` beside the map (``<path>.json``) is what the map's structure took and what it holds, in counts; the map itself is
the prover's witness (d_j hashes the served values) and is never published.  A failure is written beside the map
(``<path>.error.txt``) and changes no verdict.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.remote_bandwidth.memory_map import dumps
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import memory_map as MM
from verity_vllm.pipeline.serving_rows import _operand


def after_commit(request: str | None, *, loaded: Mapping[str, Mapping[str, Any]], bmap: dict[str, Any], committed: Callable,
                 manifest: Any, arrivals: Mapping[str, int]) -> dict[str, Any] | None:
    """Write the run's memory map as ``request`` asks, from the requests' loaded Programs of record (`c2_replay.programs_of_record`)
    at their engine arrival steps; None without a request.  A failure is written beside the map and returned, never raised."""
    if not request:
        return None
    req = json.loads(Path(request).read_text())
    out = Path(req["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return write(out, req, loaded=loaded, store=CommittedStore.of_record(bmap, committed, manifest if isinstance(manifest, dict) else None),
                     arrivals=arrivals)
    except Exception as e:  # noqa: BLE001
        out.with_name(out.name + ".error.txt").write_text(traceback.format_exc())
        print(f"[memory-map] FAILED (no verdict changes): {e!r}"[:400], flush=True)
        return {"error": repr(e)}


def write(out: Path, req: Mapping[str, Any], *, loaded: Mapping[str, Mapping[str, Any]], store: Any, arrivals: Mapping[str, int]) -> dict[str, Any]:
    """The map of ``req`` into ``out`` and its layout beside it, each request's K and V words read from ``store`` (a `CommittedStore`)."""
    from verity_vllm.program.boolean import load_program

    t0 = time.perf_counter()
    by_dir: dict[str, tuple[MM.Rows, ProgramIndex, str]] = {}
    rows, index = {}, {}
    for rid, lp in loaded.items():
        d = str(lp["dir"])
        if d not in by_dir:
            program, digest = load_program(Path(d) / "descriptor.json.gz")
            if digest != lp["instances"].get("digest"):
                raise ValueError(f"{d}: descriptor.json.gz is Program {digest[:16]}, its instance rows {str(lp['instances'].get('digest'))[:16]}")
            by_dir[d] = (MM.rows(program), ProgramIndex(lp["instances"], program_dir=d), digest)
        rows[rid], index[rid], _ = by_dir[d]
    t_ir = time.perf_counter()

    def payload(rid: str, row: MM.Row) -> bytes:
        words = _operand(index[rid], store, None, rid, int(arrivals[rid]), row.group)
        return np.ascontiguousarray(words, dtype=f"<u{rows[rid].word_bytes}").tobytes()

    layout = MM.Layout(**{k: (tuple(req[k]) if k == "firsts" else int(req[k])) for k in ("block_size", "pages", "firsts", "blocks") if req.get(k) is not None})
    head, steps = MM.record(rows, {r: int(arrivals[r]) for r in rows}, payload, bytes.fromhex(req["registration"]), layout, req.get("tables"))
    out.write_text(dumps(head, steps))
    doc = {"schema": "verity-vllm/memory-map-layout/v1", "map": out.name, "blocks": head.blocks, "block_size": layout.block_size,
           "requests": {r: {"arrive": int(arrivals[r]), "steps": rs.steps, "positions": rs.positions, "layers": len(rs.layers),
                            "row_bytes": rs.row_bytes, "program_digest": by_dir[str(loaded[r]["dir"])][2]} for r, rs in rows.items()},
           "steps": len(steps), "written": sum(len(s.written) for s in steps), "freed": sum(len(s.freed) for s in steps),
           "seconds": {"ir_s": round(t_ir - t0, 3), "digests_s": round(time.perf_counter() - t_ir, 3)}}
    out.with_name(out.name + ".json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[memory-map] {doc['written']} blocks written and {doc['freed']} freed over {doc['steps']} steps beside {head.blocks} "
          f"registered: {out} ({doc['seconds']})", flush=True)
    return doc
