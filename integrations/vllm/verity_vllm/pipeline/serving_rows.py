"""The serving rows of a Commit: the units of the verifier's partition, read exactly as the replay reads them (committed members
through an opened reader, registered weights for weight slices), committed in M0's serving row format by
:mod:`verity_vllm.commit.serving_rows`, with core's one-stage registration over them.  Three ways in, one window, two kinds:

* **In the Commit** (opt-in, ``commit --serving-rows``): the row stage writes a request ``{"dir", "partition", "run",
  "committer"}``, and the in-process C2 replay commits the window after the vllm-v1 roots are final and before its seed (``readers``).
  The window is ``served``: its roots are serving's own commitment, made before any draw.  The profile makes no claim that they
  hold the values under the vllm-v1 roots.
* **In a TP Commit** (``tp-commit --serving-rows``, `pipeline.tp.serving_rows`): one ``served`` window per rank, committed in the
  rank's process once every rank's root is final and the TP run root fixed, before the challenge; its run is the TP run root and
  the rank.
* **From a deferred Commit's bundle** (``verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O``,
  `pipeline.deferred_rows`): the sealed replay bundle opened against the Commit's record as the CPU replay opens it, and the same
  window written into ``O`` as a ``stand-in``: serving's commitment is the vllm-v1 roots, and these rows are committed afterwards
  from the bundle, so their link to those roots is provenance (``deferred_rows.stand_in``).

The partition file is the verifier's: ``{"templates": [<descriptor id>, ...], "law": "subset:K" | {...}, "scope"?:
{"op_path_prefix"}, "partition_digest"?}``.  The members are its templates in order, each instance count read off the Build's
Programs; the population program over them (``commit.serving_rows.population_program``) is the one the verifier builds from the same
Programs, and a ``partition_digest`` in the file that is not that program's partition is refused.  ``partition.json`` beside the
window is the derived file, digest and members included.  ``scope.op_path_prefix`` restricts the population to one module
(``model.layers.0.``).

A Program row holds instances of a template when the template is its own Definition (its spec is the descriptor id), or when its
Definition is cut into them (``CUT``: its body batches the template, read off the bound Definition, so a template id that is not what
the row computes is no match).  Each member's ports:

  ``RoPEHead_v1{D=..}``            every head of every ``RoPE_v1`` row: ``x``, ``cs`` -> ``out`` (D words each)
  ``GemmCoordinate_v1{K=..}``      every coordinate of the ``Gemm_v1`` rows of that K, as shared rows (M0's shared-row files): the
                                   ``x`` table (every token's input row), the ``w`` table (every weight row ``W[c, :]``) and ``y``,
                                   the instances referencing them by ``verity/one-stage/gemm-grid/v0`` (:func:`grid`); the tables
                                   are fixed by the layout, never by the values, so repeated rows stay separate table rows
  any row's own Definition         every row of it, the row itself: ``in<g>`` per argument group -> ``out.<member>`` per
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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import draw as DR
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import serving_rows as CS

ROPE, GEMM = "RoPEHead_v1", "GemmCoordinate"
#: the row Definitions whose Calls are cut into template instances (their bodies batch the template), the rest being the row itself
CUT = ("RoPE_v1", "Gemm_v1")
PARTITION_FORMAT = "verity-vllm/serving-rows/partition/v1"
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def readers(request: str | None, com: Any, rc: Any, weights: Callable, programs: dict[str, dict[str, Any]], bmap: dict[str, Any],
            arrivals: dict[str, int], loaded: dict[str, dict[str, Any]], manifest: Any, row: str | None,
            digests: Mapping[str, str] | None = None) -> tuple[Callable, Any]:
    """The replay's weights provider and a fresh opened reader for it, as before.  With a ``request``, the serving rows are committed
    first, through their own opened reader and before the replay's seed exists; a failure is written beside them and changes no
    verdict."""
    from verity_vllm.check.oracle_compare import OpenedReader

    if request:
        req = json.loads(Path(request).read_text())
        try:
            after_commit(req["dir"], partition_file=req["partition"], programs=programs, bmap=bmap, committed=OpenedReader(com, rc),
                         weights=weights, manifest=manifest if isinstance(manifest, dict) else None, run_root=rc.run_root, run=req.get("run"),
                         row=row or None, arrivals=arrivals, program_dirs={k: v.get("dir") for k, v in loaded.items()},
                         digests=digests, committer=req.get("committer"))
        except Exception as e:  # noqa: BLE001
            Path(req["dir"]).mkdir(parents=True, exist_ok=True)
            (Path(req["dir"]) / "error.txt").write_text(traceback.format_exc())
            print(f"[serving-rows] FAILED (no verdict changes): {e!r}"[:400], flush=True)
    return weights, OpenedReader(com, rc)


_UNITS: dict[str, tuple[tuple[str, Any, int], ...]] = {}


def unit_templates(spec: str) -> tuple[tuple[str, Any, int], ...]:
    """The template instances a Program row's Definition holds, ``(descriptor id, Definition, instances)``: the row itself, then, for a
    ``CUT`` family, each template its body batches (a batch's members; Input batches are its parameters).  None when the spec does
    not bind to exactly its own descriptor id."""
    hit = _UNITS.get(spec)
    if hit is not None:
        return hit
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.program import is_input
    from verity_vllm.query.program_view import _registry, spec_base, spec_statics

    reg, _errors = _registry()
    d, hit = reg.defs.get(spec_base(spec)), ()
    try:
        sp = d.bind(**spec_statics(spec)) if d is not None and hasattr(d, "bind") else None
    except (TypeError, ValueError, KeyError):
        sp = None
    if sp is not None and sp.complete and descriptor_id(sp) == spec:
        hit = ((spec, sp, 1),)
        if spec_base(spec) in CUT:
            hit += tuple((descriptor_id(n.fn), n.fn, int(n.n) if n.form == "batch" else 1) for n in sp.body.nodes
                         if n.form in ("call", "batch") and not is_input(n.fn))
    _UNITS[spec] = hit
    return hit


Row = tuple[str, Any, dict, dict]


def member_rows(programs: dict[str, dict[str, Any]], tmpls: Sequence[str], scope: str | None = None,
                program_dirs: dict[str, str] | None = None,
                index: Mapping[str, Any] | None = None) -> tuple[list[list[Row]], list[Any], list[int]]:
    """Per template, its rows in order (request, its ProgramIndex, the row, its description), restricted to ``scope`` (an op_path
    prefix); each template's Definition; and its instance count.  Fixed by the Build's Programs before serving.  ``index``: the
    ProgramIndex's ``attention_paths`` and ``tp`` for a rank Program (`pipeline.tp.serving_rows`)."""
    want = {t: k for k, t in enumerate(tmpls)}
    out: list[list[Row]] = [[] for _ in tmpls]
    defs: list[Any] = [None] * len(tmpls)
    counts = [0] * len(tmpls)
    for rid, inst in programs.items():
        pi = ProgramIndex(inst, program_dir=(program_dirs or {}).get(str(rid)), **(index or {}))
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            hits = [(want[t], fn, n) for t, fn, n in unit_templates(r["spec"]) if t in want]
            if not hits:
                continue
            d = pi.describe(r)
            if d is None or (scope and not d["op_path"].startswith(scope)):
                continue
            if len(hits) > 1:
                raise ValueError(f"row {i} ({r['spec']}) holds instances of {[tmpls[k] for k, _, _ in hits]}: one member reads a row")
            (k, fn, n), = hits
            out[k].append((str(rid), pi, r, d))
            defs[k], counts[k] = fn, counts[k] + n
    for t, rows in zip(tmpls, out, strict=True):
        if not rows:
            raise LookupError(f"no rows of {t} in the run's Programs" + (f" under {scope}" if scope else ""))
    return out, defs, counts


def members(tmpls: Sequence[str], rows: Sequence[list[Row]], counts: Sequence[int] | None = None) -> list[CS.Member]:
    """Each template's ports, instance count and base, read off its first row; ``counts`` (the Definitions' own) must agree."""
    out, base = [], 0
    for k, (t, rs) in enumerate(zip(tmpls, rows, strict=True)):
        _rid, _pi, r, d = rs[0]
        if t.startswith(ROPE):
            D = int(d["statics"]["D"])
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
        if counts is not None and counts[k] != m.n:
            raise ValueError(f"{t}: {m.n} units read off the rows, {counts[k]} in their Definitions")
        out.append(m)
        base += m.n
    return out


def law_of(law: str | Mapping[str, Any]) -> dict[str, Any]:
    return DR.parse_law(law) if isinstance(law, str) else dict(law)


def population(programs: dict[str, dict[str, Any]], partition: Mapping[str, Any], run_root: str,
               program_dirs: dict[str, str] | None = None, index: Mapping[str, Any] | None = None) -> tuple[CS.Window, list[list[Row]]]:
    """The verifier's window over the run's Programs (``partition``: the partition file's ``templates``, ``law`` and ``scope``) and
    each member's rows; refused when the file names a ``partition_digest`` that is not the window's."""
    tmpls = list(partition["templates"])
    rows, defs, counts = member_rows(programs, tmpls, (partition.get("scope") or {}).get("op_path_prefix"), program_dirs, index)
    ms = members(tmpls, rows, counts)
    program, query = CS.population_program([(fn, m.n) for fn, m in zip(defs, ms, strict=True)])
    window = CS.Window(program, query, law_of(partition["law"]), run_root, tuple(ms))
    want = partition.get("partition_digest")
    if want and want != window.partition_digest:
        raise ValueError(f"the run's population ({', '.join(f'{m.template} x {m.n}' for m in ms)}) is partition "
                         f"{window.partition_digest[:16]}, not the verifier's {str(want)[:16]}")
    return window, rows


def partition_doc(window: CS.Window, partition: Mapping[str, Any]) -> dict[str, Any]:
    """The partition file as derived: the verifier's templates, law and scope, with the members, the partition object and digest."""
    return {"format": PARTITION_FORMAT, "templates": [m.template for m in window.members], "law": window.law,
            "scope": partition.get("scope"), "members": [[m.template, m.n] for m in window.members], "partition": window.partition,
            "partition_digest": window.partition_digest, "population": window.population}


def read(m: CS.Member, rows: list[Row], store: Any, weights: Callable,
         arrivals: dict[str, int] | None = None) -> tuple[dict[str, dict[str, np.ndarray]], list[list[Any]]]:
    """Member ``m``'s words, read from the committed store and the registered weights exactly as the replay reads them, and its
    index entries."""
    if m.template.startswith(GEMM):
        return _read_gemm(m, rows, store, weights, arrivals)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    D = dict(m.rows).get("x", 0)
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
                 weights: Callable, manifest: dict[str, Any] | None, run_root: bytes | str, run: str | None = None, row: Any = None,
                 arrivals: dict[str, int] | None = None, program_dirs: dict[str, str] | None = None,
                 digests: Mapping[str, str] | None = None, committer: dict[str, Any] | None = None,
                 workers: int | None = None, stand_in: Mapping[str, Any] | None = None, tp: Mapping[str, Any] | None = None,
                 index: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Commit the verifier's partition's units of a run whose vllm-v1 roots are final and write the window's files into ``out_dir``:
    the registration (core's v1 record, its window the run root ``run_root`` and the provenance: the research run ``run``, the row,
    the committer and the served Programs' ``digests``), ``partition.json``, the index and M0's bodies.  The window is ``served``;
    with ``stand_in`` (its ``of`` and ``link``), a ``stand-in`` with them in its provenance.  A TP rank's window (`pipeline.tp.
    serving_rows`) gives its run as a string, its world and roots as ``tp`` and its ProgramIndex arguments as ``index``.  It reads
    the committed values and writes only ``out_dir``."""
    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    pf = json.loads(Path(partition_file).read_text())
    window, rows = population(programs, pf, run_root if isinstance(run_root, str) else bytes(run_root).hex(), dirs, index)
    ms = window.members
    store = CommittedStore.of_record(bmap, committed, manifest)
    got = [read(m, r, store, weights, arrivals) for m, r in zip(ms, rows, strict=True)]
    t_read = time.perf_counter()
    c = CS.commit(window, [g[0] for g in got], workers=workers or min(16, os.cpu_count() or 1))
    prov = {"commitment": "vllm-v1", "row": row, "research_run": run, "committer": dict(committer or {}, module=CS.__name__),
            "served_programs": {str(k): str(v) for k, v in sorted((digests or {}).items())}, "scope": pf.get("scope"),
            **({"tp": dict(tp)} if tp else {}), **(stand_in or {})}
    c.seconds["read_s"] = round(t_read - t0, 3)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "partition.json").write_text(json.dumps(partition_doc(window, pf), indent=1) + "\n")
    res = CS.write(c, out_dir, prov, CS.index_doc(ms, [g[1] for g in got], window.run), kind="stand-in" if stand_in else "served")
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (Path(out_dir) / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[serving-rows] {window.population} units in {len(ms)} members committed ({CS.SCHEME}): "
          + ", ".join(f"{m.template} {m.n}" for m in ms) + f"; registration {res['registration_digest'][:16]}; {res['seconds']}", flush=True)
    return res

