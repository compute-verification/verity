"""The Commit stage's opt-in serving rows: every head of the run's ``RoPE_v1`` rows, read exactly as the replay reads them (``x``
and ``out`` from the committed store through an opened reader, ``cs`` from the registered weights), committed in M0's serving
row format by :mod:`verity_vllm.commit.serving_rows`.  Off unless the row stage writes a request (``commit --serving-rows``):
``{"dir", "partition", "run", "committer"}``, the partition file being the verifier's, fixed before serving.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import serving_rows as CS
from verity_vllm.query.manifest import format as RM


def readers(request: str | None, com: Any, rc: Any, weights: Callable, programs: dict[str, dict[str, Any]], bmap: dict[str, Any],
            arrivals: dict[str, int], loaded: dict[str, dict[str, Any]], manifest: Any, row: str | None) -> tuple[Callable, Any]:
    """The replay's weights provider and a fresh opened reader for it, as before.  With a ``request``, the serving rows are committed
    first, through their own opened reader and before the replay's seed exists; a failure is written beside them and changes no
    verdict."""
    from verity_vllm.check.oracle_compare import OpenedReader

    if request:
        req = json.loads(Path(request).read_text())
        try:
            after_commit(req["dir"], partition_file=req["partition"], programs=programs, bmap=bmap, committed=OpenedReader(com, rc),
                         weights=weights, manifest=manifest if isinstance(manifest, dict) else None, run_root=rc.run_root, run=req["run"],
                         row=row or None, arrivals=arrivals, program_dirs={k: v.get("dir") for k, v in loaded.items()},
                         committer=req.get("committer"))
        except Exception as e:  # noqa: BLE001
            Path(req["dir"]).mkdir(parents=True, exist_ok=True)
            (Path(req["dir"]) / "error.txt").write_text(traceback.format_exc())
            print(f"[serving-rows] FAILED (no verdict changes): {e!r}"[:400], flush=True)
    return weights, OpenedReader(com, rc)


def rope_rows(programs: dict[str, dict[str, Any]], program_dirs: dict[str, str] | None = None) -> list[tuple[str, Any, dict, dict]]:
    """The ``RoPE_v1`` rows in ``commit.serving_rows.ORDER``: (request, its ProgramIndex, the row, its description).  The population
    is the sum of their NHEADS, fixed by the Build's Programs before serving."""
    out = []
    for rid, inst in programs.items():
        pi = ProgramIndex(inst, program_dir=(program_dirs or {}).get(str(rid)))
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            if RM.spec_family(r["spec"]) == CS.FAMILY:
                d = pi.describe(r)
                if int(d["statics"]["D"]) != CS.D:
                    raise ValueError(f"row {i}: RoPE_v1 D={d['statics']['D']}; serving rows commit D={CS.D} heads only")
                out.append((str(rid), pi, r, d))
    if not out:
        raise LookupError(f"no {CS.FAMILY} rows in the run's Programs")
    return out


def population(programs: dict[str, dict[str, Any]], program_dirs: dict[str, str] | None = None) -> int:
    return sum(int(d["statics"]["NHEADS"]) for _, _, _, d in rope_rows(programs, program_dirs))


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def rope_units(rows: list[tuple[str, Any, dict, dict]], store: Any, weights: Callable, *, arrivals: dict[str, int] | None = None,
               ) -> tuple[list[list[Any]], dict[str, np.ndarray], np.ndarray]:
    """Every head of every row, its words read from the committed store (``x``, ``out``) and the registered weights (``cs``)
    exactly as the replay reads them.  Returns (index entries, rows per input port, out)."""
    entries: list[list[Any]] = []
    xs, css, outs = [], [], []
    n = 0
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        nh = int(d["statics"]["NHEADS"])
        ops = [_operand(pi, store, weights, rid, arrive, g) for g in r["args"]]
        (member, width, _dt), = d["members"]
        w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if w is None:
            raise LookupError(f"row {r['i']} output: {why}")
        xs.append(np.asarray(ops[0], dtype=np.uint16).reshape(nh, CS.D))
        css.append(np.broadcast_to(np.asarray(ops[1], dtype=np.uint16).reshape(1, CS.D), (nh, CS.D)))
        outs.append(np.asarray(w, dtype=np.uint16).reshape(nh, CS.D))
        m = _LAYER.search(d["op_path"])
        entries.append([n, n + nh, rid, arrive + int(d["step"]), int(m.group(1)) if m else None, d["op_path"], str(member), int(d["row"]), int(r["i"])])
        n += nh
    return entries, {"x": np.concatenate(xs), "cs": np.concatenate(css)}, np.concatenate(outs)


def _operand(pi: Any, store: Any, weights: Callable, rid: str, arrive: int, group: Sequence) -> np.ndarray:
    words = []
    for sl in group:
        kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
        if kind == "n":
            m = pi.member_at(a, lo, hi)
            if m is None:
                raise LookupError(f"arg slice {list(sl)} is not inside one committed member")
            pd, member, mlo, mhi, _dt = m
            width = next(int(mm[1]) for mm in pd["members"] if mm[0] == member)
            w, why = store.words(rid, arrive + pd["step"], pd["op_path"], member, pd["row"] * width + mlo, pd["row"] * width + mhi)
            if w is None:
                raise LookupError(why)
            words.append(np.asarray(w).reshape(-1))
        elif kind == "p" and (pi.pnames[a] if a < len(pi.pnames) else None) == "weights":
            ws = pi.weight_slice(lo, hi)
            if ws is None:
                raise LookupError(f"weights slice [{lo}, {hi}) spans fields")
            fname, flo, fhi, _shape = ws
            arr = weights(fname, flo, fhi)
            if arr is None:
                raise LookupError(f"registered weight field {fname} is not available")
            words.append(np.asarray(arr).reshape(-1))
        else:
            raise LookupError(f"arg slice {list(sl)}: the serving rows read committed members and registered weights only")
    return np.concatenate(words)


def after_commit(out_dir: str, *, partition_file: str, programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable,
                 weights: Callable, manifest: dict[str, Any] | None, run_root: bytes, run: str, row: Any = None,
                 arrivals: dict[str, int] | None = None, program_dirs: dict[str, str] | None = None, committer: dict[str, Any] | None = None,
                 workers: int | None = None) -> dict[str, Any]:
    """The Commit stage's opt-in hook, after the vllm-v1 roots are final and before any challenge: commit the run's RoPE heads
    and write the window's files into ``out_dir``.  It reads the committed values and writes only ``out_dir``."""
    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    served = {CS.program_sha512(dirs[str(r)]) for r in programs}
    if len(served) != 1:
        raise NotImplementedError(f"serving rows register one served Program; this run has {len(served)}")
    window = CS.Window.from_file(partition_file, run)
    rows = rope_rows(programs, dirs)
    entries, cols, out = rope_units(rows, CommittedStore.of_record(bmap, committed, manifest), weights, arrivals=arrivals)
    t_read = time.perf_counter()
    c = CS.commit(window, cols, out, workers=workers or min(16, os.cpu_count() or 1))
    source = {"row": row, "run": run, "committer": dict(committer or {}, module=CS.__name__), "served_program": served.pop(),
              "served_program_digest": sorted({str(i.get("digest")) for i in programs.values()}), "vllm_v1_run_root": bytes(run_root).hex()}
    c.seconds["read_s"] = round(t_read - t0, 3)
    res = CS.write(c, out_dir, source, CS.index_doc(entries, run))
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (Path(out_dir) / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[serving-rows] {c.n} {CS.FAMILY} heads committed ({CS.SCHEME}): roots " + ", ".join(f"{p} {c.roots[p].hex()[:16]}" for p in c.roots)
          + f"; registration {res['registration_sha512'][:16]}; {res['seconds']}", flush=True)
    return res
