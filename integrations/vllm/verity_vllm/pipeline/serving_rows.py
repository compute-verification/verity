"""The Commit stage's opt-in serving rows: the units of the verifier's partition, read exactly as the replay reads them (committed
members through an opened reader, registered weights for weight slices), committed in M0's serving row format by
:mod:`verity_vllm.commit.serving_rows`.  Off unless the row stage writes a request (``commit --serving-rows``):
``{"dir", "partition", "run", "committer"}``.  The partition file (``{partition, partition_digest, population, scope?}``) is the
verifier's, fixed before serving; ``scope.op_path_prefix`` restricts the population to one module (``model.layers.0.``).

Members come from the partition's query, in its order (``Q_template_instances``' ``params.templates``, or
``Q_template_instance``'s one ``template``), each named by its descriptor id:

  ``RoPEHead_v1{D=64}``            every head of every ``RoPE_v1`` row: ``x``, ``cs`` -> ``out`` (64 words each)
  ``GemmCoordinate_v2{K=..}``      every coordinate of the ``Gemm_v1`` rows of that K, as shared rows (M0's shared-row files): the
                                   ``x`` table (every token's input row), the ``w`` table (every weight row ``W[c, :]``) and ``y``,
                                   the instances referencing them by ``verity/one-stage/gemm-grid/v0`` (:func:`grid`); the tables
                                   are fixed by the layout, never by the values, so repeated rows stay separate table rows
  any other ``<Family>{...}``      every row of that family, the row itself: ``in<g>`` per argument group -> ``out.<member>`` per
                                   committed member (``vu_export._decompose_row``'s port rule)

Units go in request order, then Program row index, then head (``commit.serving_rows.ORDER``).  In a prefill step the Program lists
a layer's q rows for every token before its k rows, so RoPE's heads are q heads for every prefill token, then k heads.
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

ROPE, ROPE_FAMILY, D = "RoPEHead_v1", "RoPE_v1", 64
GEMM, GEMM_FAMILY = "GemmCoordinate", "Gemm_v1"
_LAYER = re.compile(r"\.layers\.(\d+)\.")


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


def templates(partition: dict[str, Any]) -> list[str]:
    """The partition's templates (descriptor ids), in its order."""
    q = partition["query"]
    params = q.get("params") or {}
    if "templates" in params:
        return list(params["templates"])
    if "template" in params or "template" in q:
        return [params.get("template") or q["template"]]
    raise ValueError(f"no templates in the partition's query {q.get('name')!r}")


def _family(template: str) -> str:
    name = template.split("{", 1)[0]
    if name == ROPE:
        return ROPE_FAMILY
    if name.startswith(GEMM):
        return GEMM_FAMILY
    return name


def _statics_match(template: str, statics: dict[str, Any]) -> bool:
    """The row's integer statics, as ``NAME=value``, all appear in the descriptor id (RoPE: only ``D``)."""
    if template.startswith(ROPE):
        return f"D={statics.get('D')}" in template
    if template.startswith(GEMM):
        return f"K={statics.get('K')}," in template or f"K={statics.get('K')}}}" in template
    return all(f"{k}={v}" in template for k, v in statics.items() if isinstance(v, int))


def member_rows(programs: dict[str, dict[str, Any]], tmpls: Sequence[str], scope: str | None = None,
                program_dirs: dict[str, str] | None = None) -> list[list[tuple[str, Any, dict, dict]]]:
    """Per template, its rows in order: (request, its ProgramIndex, the row, its description), restricted to ``scope`` (an op_path
    prefix).  Fixed by the Build's Programs before serving."""
    fams = [_family(t) for t in tmpls]
    out: list[list[tuple[str, Any, dict, dict]]] = [[] for _ in tmpls]
    for rid, inst in programs.items():
        pi = ProgramIndex(inst, program_dir=(program_dirs or {}).get(str(rid)))
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            f = RM.spec_family(r["spec"])
            if f not in fams:
                continue
            d = pi.describe(r)
            if d is None or (scope and not d["op_path"].startswith(scope)):
                continue
            ks = [k for k, g in enumerate(fams) if g == f and _statics_match(tmpls[k], d["statics"])]
            if not ks:
                if f == GEMM_FAMILY:
                    continue                                   # a GEMM of another K than the population's
                raise ValueError(f"row {i} ({r['spec']}) is not an instance of {tmpls[fams.index(f)]}")
            out[ks[0]].append((str(rid), pi, r, d))
    for t, rows in zip(tmpls, out, strict=True):
        if not rows:
            raise LookupError(f"no rows of {t} in the run's Programs" + (f" under {scope}" if scope else ""))
    return out


def members(tmpls: Sequence[str], rows: Sequence[list[tuple[str, Any, dict, dict]]]) -> list[CS.Member]:
    """Each template's ports, instance count and base, read off its first row."""
    out, base = [], 0
    for t, rs in zip(tmpls, rows, strict=True):
        _rid, _pi, r, d = rs[0]
        if t.startswith(ROPE):
            n = sum(int(x[3]["statics"]["NHEADS"]) for x in rs)
            m = CS.Member(t, (("x", D), ("cs", D)), (("out", D),), n, base)
        elif t.startswith(GEMM):
            K, groups = int(d["statics"]["K"]), grid(rs)
            rx, rw = sum(g["tokens"] for g in groups), sum(g["columns"] for g in groups)
            m = CS.Member(t, (("x", K), ("w", K)), (("y", 1),), sum(g["tokens"] * g["columns"] for g in groups), base,
                          tables=(("x", rx), ("w", rw)))
        else:
            groups = tuple((f"in{g}", sum(int(sl[3]) - int(sl[2]) for sl in grp)) for g, grp in enumerate(r["args"]))
            m = CS.Member(t, groups, tuple((f"out.{mm[0]}", int(mm[1])) for mm in d["members"]), len(rs), base)
        out.append(m)
        base += m.n
    return out


def read(m: CS.Member, rows: list[tuple[str, Any, dict, dict]], store: Any, weights: Callable,
         arrivals: dict[str, int] | None = None) -> tuple[dict[str, dict[str, np.ndarray]], list[list[Any]]]:
    """Member ``m``'s words, read from the committed store and the registered weights exactly as the replay reads them, and its
    index entries."""
    if m.template.startswith(GEMM):
        return _read_gemm(m, rows, store, weights, arrivals)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    entries: list[list[Any]] = []
    lo = m.base
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        ops = [_operand(pi, store, weights, rid, arrive, g) for g in r["args"]]
        outs = {}
        for member, width, _dt in d["members"]:
            w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
            if w is None:
                raise LookupError(f"row {r['i']} output {member}: {why}")
            outs[member] = np.asarray(w, dtype=np.uint16)
        if m.template.startswith(ROPE):
            nh = int(d["statics"]["NHEADS"])
            cols["x"].append(np.asarray(ops[0], dtype=np.uint16).reshape(nh, D))
            cols["cs"].append(np.broadcast_to(np.asarray(ops[1], dtype=np.uint16).reshape(1, D), (nh, D)))
            cols["out"].append(next(iter(outs.values())).reshape(nh, D))
        else:
            nh = 1
            for g, a in enumerate(ops):
                cols[f"in{g}"].append(np.asarray(a, dtype=np.uint16).reshape(1, -1))
            for member, a in outs.items():
                cols[f"out.{member}"].append(a.reshape(1, -1))
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + nh, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        ",".join(str(mm[0]) for mm in d["members"]), int(d["row"]), int(r["i"])])
        lo += nh
    data = {"rows": {p: np.concatenate(cols[p]) for p, _ in m.rows}, "outs": {p: np.concatenate(cols[p]) for p, _ in m.outs}}
    return data, entries


def grid(rows: list[tuple[str, Any, dict, dict]]) -> list[dict[str, Any]]:
    """``verity/one-stage/gemm-grid/v0``'s groups: one per GEMM site (op_path) in order of first appearance, each
    ``{name, tokens, columns, x_base, w_base, rows}``; instance ``i`` of a group, ``j = i - base``, is token ``j div columns`` and
    column ``j mod columns``, and reads ``x`` row ``x_base + j div columns`` and ``w`` row ``w_base + j mod columns``."""
    by: dict[str, list] = {}
    for x in rows:
        by.setdefault(x[3]["op_path"], []).append(x)
    out, xb, wb = [], 0, 0
    for op, rs in by.items():
        cols = int(rs[0][3]["statics"]["N"])
        out.append({"name": op, "tokens": len(rs), "columns": cols, "x_base": xb, "w_base": wb, "rows": rs})
        xb += len(rs)
        wb += cols
    return out


def _read_gemm(m: CS.Member, rows: list[tuple[str, Any, dict, dict]], store: Any, weights: Callable,
               arrivals: dict[str, int] | None) -> tuple[dict[str, Any], list[list[Any]]]:
    """A GEMM member as shared rows: the ``x`` table (every token's input row, group by group), the ``w`` table (every weight row
    ``W[c, :]``, read once per group), ``y`` (every output word, instance-major) and the refs of the grid rule."""
    K = dict(m.rows)["x"]
    xs, ws, ys, refs, entries = [], [], [], [], []
    lo = m.base
    for g in grid(rows):
        T, N = g["tokens"], g["columns"]
        rid0, pi0, r0, _d0 = g["rows"][0]
        ws.append(np.asarray(_operand(pi0, store, weights, rid0, int((arrivals or {}).get(rid0, 0)), r0["args"][1]), dtype=np.uint16).reshape(N, K))
        for rid, pi, r, d in g["rows"]:
            if r["args"][1] != r0["args"][1]:
                raise ValueError(f"row {r['i']}: {g['name']} reads another weight slice than its group's first row")
            arrive = int((arrivals or {}).get(rid, 0))
            xs.append(np.asarray(_operand(pi, store, weights, rid, arrive, r["args"][0]), dtype=np.uint16).reshape(1, K))
            (member, width, _dt), = d["members"]
            w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
            if w is None:
                raise LookupError(f"row {r['i']} output {member}: {why}")
            ys.append(np.asarray(w, dtype=np.uint16).reshape(-1))
            layer = _LAYER.search(d["op_path"])
            entries.append([lo, lo + N, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                            int(d["row"]), int(r["i"])])
            lo += N
        j = np.arange(T * N, dtype=np.int64)
        refs.append(np.stack([g["x_base"] + j // N, g["w_base"] + j % N], axis=1))
    data = {"rows": {"x": np.concatenate(xs), "w": np.concatenate(ws)}, "outs": {"y": np.concatenate(ys).reshape(-1, 1)},
            "refs": np.concatenate(refs)}
    return data, entries


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
    """The Commit stage's opt-in hook, after the vllm-v1 roots are final and before any challenge: commit the partition's units and
    write the window's files into ``out_dir``.  It reads the committed values and writes only ``out_dir``."""
    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    served = {CS.program_digest(dirs[str(r)]) for r in programs}
    if len(served) != 1:
        raise NotImplementedError(f"serving rows register one served Program; this run has {len(served)}")
    pf = json.loads(Path(partition_file).read_text())
    tmpls = templates(pf["partition"])
    rows = member_rows(programs, tmpls, (pf.get("scope") or {}).get("op_path_prefix"), dirs)
    ms = members(tmpls, rows)
    window = CS.Window.from_file(partition_file, run, ms)
    store = CommittedStore.of_record(bmap, committed, manifest)
    got = [read(m, r, store, weights, arrivals) for m, r in zip(ms, rows, strict=True)]
    t_read = time.perf_counter()
    c = CS.commit(window, [g[0] for g in got], workers=workers or min(16, os.cpu_count() or 1))
    source = {"row": row, "run": run, "committer": dict(committer or {}, module=CS.__name__), "served_program": served.pop(),
              "served_program_digest": sorted({str(i.get("digest")) for i in programs.values()}), "vllm_v1_run_root": bytes(run_root).hex(),
              "scope": pf.get("scope")}
    c.seconds["read_s"] = round(t_read - t0, 3)
    res = CS.write(c, out_dir, source, CS.index_doc(ms, [g[1] for g in got], run))
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (Path(out_dir) / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[serving-rows] {window.population} units in {len(ms)} members committed ({CS.SCHEME}): "
          + ", ".join(f"{m.template} {m.n}" for m in ms) + f"; registration {res['registration_sha512'][:16]}; {res['seconds']}", flush=True)
    return res
