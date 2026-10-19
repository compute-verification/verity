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

The partition file is the verifier's: ``{"templates": [<descriptor id>, ...], "law": "subset:K" | "stratified:K" | {...},
"query"?: "Q_template_instance" | "Q_template_instances", "scope"?: {"op_path_prefix", "steps"}, "chunk_cap"?: {<descriptor id>:
cap}, "row_tables"?: true, "partition_digest"?}``.  The members are its templates in order, each instance count read off the Build's
Programs; the population program over them (``commit.serving_rows.population_program``: ``TemplatePopulations`` for several
templates, or for one when the file names ``Q_template_instances``) is the one the verifier builds from the same Programs, and a
``partition_digest`` in the file that is not that program's partition is refused, as is any other query (a word partition,
``Q_word``, never: with the stratified law it is refused by name, its strata being templates).  ``partition.json`` beside the window
is the derived file, digest and members included.  ``scope.op_path_prefix`` restricts the population to one module
(``model.layers.0.``), ``scope.steps`` (``[lo, hi)``) to those steps of each request.  A ``ScaledMmS8Coordinate`` template's units
are split into consecutive chunks of at most its ``chunk_cap`` (flock-circuit's spec shape, ``chunk_cap``, as
``benchmarks/one_stage/a6.py`` computes it; ``chunk_caps`` writes the entry), each chunk a member with its public file of global
unit indices; ``row_tables`` puts them in the row-table layout (``with_row_tables``).

A Program row holds instances of a template when the template is its own Definition (its spec is the descriptor id), or when its
Definition is cut into them (``CUT``: its body batches the template, read off the bound Definition, so a template id that is not what
the row computes is no match).  Each member's ports:

  ``RoPEHead_v1{D=..}``            every head of every ``RoPE_v1`` row: ``x``, ``cs`` -> ``out`` (D words each)
  ``GemmCoordinate_v1{K=..}``      every coordinate of the ``Gemm_v1`` rows of that K, as shared rows (M0's shared-row files): the
                                   ``x`` table (every token's input row), the ``w`` table (every weight row ``W[c, :]``) and ``y``,
                                   the instances referencing them by ``verity/one-stage/gemm-grid/v0`` (:func:`grid`); the tables
                                   are fixed by the layout, never by the values, so repeated rows stay separate table rows
  ``ScaledMmS8Coordinate_v1{K=..}``  every coordinate of the ``ScaledMmS8_v1`` rows of that K, A6's bit rows per instance:
                                   ``x`` (the token's s8 codes then its f32 scale, row-seg), ``w`` (``W[c, :]`` then ``sw[c]``,
                                   v2) -> ``o0`` (the bf16 output, a v2 row of its one word); with ``row_tables``, shared
                                   rows (`with_row_tables`): the ``x`` table the chunk's distinct token rows, each a row of its
                                   row table (one per engine step and op_path), and the ``w`` table its own rows
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P
from verity.ml.circuits import partition_object as PO
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import serving_rows as CS

ROPE, GEMM, S8 = "RoPEHead_v1", "GemmCoordinate", "ScaledMmS8Coordinate"
#: the row Definitions whose Calls are cut into template instances (their bodies batch the template), the rest being the row itself
CUT = ("RoPE_v1", "Gemm_v1", "ScaledMmS8_v1")
PARTITION_FORMAT = "verity-vllm/serving-rows/partition/v1"
#: flock-circuit's spec shape (`verity/ml/flock/live/src/circuit.rs`'s `shape_in_range`, `benchmarks/one_stage/a6.py`'s `M_MAX`):
#: a statement's dense size m = k_log + nbl, nbl = max(3, ceil(log2 blocks)), is at most 35
M_MAX = 35
TEMPLATE_QUERIES = (PO.Q_TEMPLATE_INSTANCE[0], PO.Q_TEMPLATE_INSTANCES[0])
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
    from verity_vllm.query.program_view import spec_base

    sp, hit = _bind(spec), ()
    if sp is not None and sp.complete and descriptor_id(sp) == spec:
        hit = ((spec, sp, 1),)
        if spec_base(spec) in CUT:
            hit += tuple((descriptor_id(n.fn), n.fn, int(n.n) if n.form == "batch" else 1) for n in sp.body.nodes
                         if n.form in ("call", "batch") and not is_input(n.fn))
    _UNITS[spec] = hit
    return hit


def _bind(spec: str) -> Any:
    """The registry's Definition bound at ``spec``'s statics, each ``{"fn": id}`` static (`spec_statics` reads it as the id) the
    primitive it names; None when it does not bind.  A caller holds the bound descriptor id to ``spec``."""
    from verity.ml.circuits.codec import lookup_primitive
    from verity.ml.circuits.defs import REGISTRY
    from verity_vllm.query.program_view import _registry, spec_base, spec_statics

    def static(v: Any) -> Any:
        try:
            return lookup_primitive(REGISTRY, v) if isinstance(v, str) else v
        except KeyError:
            return v

    reg, _errors = _registry()
    d = reg.defs.get(spec_base(spec))
    if d is None or not hasattr(d, "bind"):
        return None
    try:
        return d.bind(**{k: static(v) for k, v in spec_statics(spec).items()})
    except (TypeError, ValueError, KeyError):
        return None


Row = tuple[str, Any, dict, dict]


def member_rows(programs: dict[str, dict[str, Any]], tmpls: Sequence[str], scope: str | None = None,
                program_dirs: dict[str, str] | None = None,
                index: Mapping[str, Any] | None = None,
                steps: Sequence[int] | None = None) -> tuple[list[list[Row]], list[Any], list[int]]:
    """Per template, its rows in order (request, its ProgramIndex, the row, its description), restricted to ``scope`` (an op_path
    prefix) and ``steps`` (``[lo, hi)`` of the request's steps); each template's Definition; and its instance count.  Fixed by the
    Build's Programs before serving.  ``index``: the ProgramIndex's ``attention_paths`` and ``tp`` for a rank Program
    (`pipeline.tp.serving_rows`)."""
    want = {t: k for k, t in enumerate(tmpls)}
    out: list[list[Row]] = [[] for _ in tmpls]
    defs: list[Any] = [None] * len(tmpls)
    counts = [0] * len(tmpls)
    seen: list[dict | None] = [None] * len(tmpls)
    for rid, inst in programs.items():
        pi = ProgramIndex(inst, program_dir=(program_dirs or {}).get(str(rid)), **(index or {}))
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            hits = [(want[t], fn, n) for t, fn, n in unit_templates(r["spec"]) if t in want]
            if not hits:
                continue
            d = pi.describe(r)
            if d is not None:
                for k, _fn, _n in hits:
                    seen[k] = seen[k] or d
            if d is None or (scope and not d["op_path"].startswith(scope)):
                continue
            if steps is not None and not int(steps[0]) <= int(d["step"]) < int(steps[1]):
                continue
            if len(hits) > 1:
                raise ValueError(f"row {i} ({r['spec']}) holds instances of {[tmpls[k] for k, _, _ in hits]}: one member reads a row")
            (k, fn, n), = hits
            out[k].append((str(rid), pi, r, d))
            defs[k], counts[k] = fn, counts[k] + n
    for t, rows, d in zip(tmpls, out, seen, strict=True):
        if not rows:
            raise LookupError(f"no rows of {t} in the run's Programs" + (f" under {scope}" if scope else "")
                              + (f" at steps {list(steps)}" if steps is not None else "")
                              + (f"; its rows exist outside the scope, e.g. op_path {d['op_path']} at step {d['step']} (scope.op_path_prefix "
                                 "is the op_path's prefix, the wrapper's module prefix stripped)" if d is not None else ""))
    return out, defs, counts


def s8_words(K: int) -> int:
    """A ``ScaledMmS8Coordinate_v1{K}`` input row's u16 words: the K s8 codes' bytes, then the f32 scale's (A6's ``x`` / ``w``)."""
    return K // 2 + 2


def chunk_cap(defn: Any, K: int) -> int:
    """The most ``ScaledMmS8Coordinate_v1{K}`` instances one flock-circuit statement holds, as ``benchmarks/one_stage/a6.py``'s
    ``chunk_cap`` computes it: the template staged by C-Flock's ``UnitStager`` with A6's rows, then ``vus_per_block`` blocks of
    ``2^(M_MAX - k_log)``.  Builds the circuit (tens of seconds): a partition file pins it as ``chunk_cap``."""
    import tempfile

    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
    from verity_flock.class_statement import UnitStager

    bits = 16 * s8_words(K)
    with tempfile.TemporaryDirectory() as d:
        st = UnitStager(defn, f"unit/serving-rows-cap-k{K}", [("x", SEG, bits, ["x", "sx"]), ("w", V2, bits, ["wrow", "sw"])],
                        [(V2, (0,))], circuit=Path(d) / "circuit.txt", bind={"partition": "0" * 128, "program": "0" * 128})
    if M_MAX - st.meta["k_log"] < 3:
        raise ValueError(f"{K=}: k_log {st.meta['k_log']} leaves no statement inside the spec shape (m <= {M_MAX})")
    return int(st.meta["vus_per_block"]) << (M_MAX - int(st.meta["k_log"]))


def chunk_caps(tmpls: Sequence[str]) -> dict[str, int]:
    """``chunk_cap`` of each ``ScaledMmS8Coordinate`` template, for a partition file's ``chunk_cap``."""
    from verity_vllm.query.program_view import spec_statics

    out = {}
    for t in tmpls:
        if t.startswith(S8):
            d = _bind(t)
            if d is None:
                raise LookupError(f"{t} does not bind in the registry")
            out[t] = chunk_cap(d, int(spec_statics(t)["K"]))
    return out


def members(tmpls: Sequence[str], rows: Sequence[list[Row]], counts: Sequence[int] | None = None,
            caps: Mapping[str, int] | None = None) -> list[CS.Member]:
    """Each template's ports, instance count and base, read off its first row; ``counts`` (the Definitions' own) must agree.  A
    ``ScaledMmS8Coordinate`` template's units are consecutive chunks of at most its ``caps`` entry, one member each."""
    out, base = [], 0
    for k, (t, rs) in enumerate(zip(tmpls, rows, strict=True)):
        _rid, _pi, r, d = rs[0]
        if t.startswith(S8):
            n, words, cap = sum(int(x[3]["statics"]["N"]) for x in rs), s8_words(int(d["statics"]["K"])), (caps or {}).get(t)
            if not isinstance(cap, int) or cap < 1:
                raise ValueError(f"{t}: no chunk cap; a partition file names each ScaledMmS8Coordinate template's flock-circuit "
                                 f"chunk cap as chunk_cap (pipeline.serving_rows.chunk_caps)")
            if counts is not None and counts[k] != n:
                raise ValueError(f"{t}: {n} units read off the rows, {counts[k]} in their Definitions")
            out += [CS.Member(t, (("x", words), ("w", words)), (("o0", 1),), min(cap, n - lo), base + lo, name=f"{t}/c{j}",
                              schemas=(("x", CS.SCHEMA_HM96_SHA512_ROW_SEG), ("w", CS.SCHEMA_HM96_SHA512_ROW_V2),
                                       ("o0", CS.SCHEMA_HM96_SHA512_ROW_V2)))
                    for j, lo in enumerate(range(0, n, cap))]
            base += n
            continue
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


def with_row_tables(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]],
                    arrivals: Mapping[str, int] | None = None) -> tuple[list[CS.Member], tuple[CS.RowTable, ...]]:
    """The row-table layout of ``members``' ``ScaledMmS8Coordinate`` chunks: one row table per (engine step, op_path) of their
    ``ScaledMmS8_v1`` rows in the canonical order serving commits them in (``RowTable.key``; each table's token rows by request,
    then token row), and each chunk a shared-row member whose ``x`` table is the distinct token rows its units read (``links``:
    each one's row table and rank, in the units' order) and whose ``w`` table is its own rows."""
    tables: dict[tuple[int, str], int] = {}
    specs: list[tuple[int, str, str, int, list[tuple[str, int, int]]]] = []
    segs: dict[str, list[tuple[int, int, int, int]]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        if not t.startswith(S8):
            continue
        lo = next(m.base for m in ms if m.template == t)
        for rid, _pi, r, d in rs:
            step, n = int((arrivals or {}).get(rid, 0)) + int(d["step"]), int(d["statics"]["N"])
            k = tables.setdefault((step, d["op_path"]), len(specs))
            if k == len(specs):
                specs.append((step, d["op_path"], t, s8_words(int(d["statics"]["K"])), []))
            specs[k][4].append((str(rid), int(d["row"]), int(r["i"])))
            segs.setdefault(t, []).append((lo, lo + n, k, len(specs[k][4]) - 1))
            lo += n
    order = sorted(range(len(specs)), key=lambda k: (specs[k][0], *CS.table_key(specs[k][1])))
    ranks = [sorted(range(len(s[4])), key=lambda j, rs=s[4]: rs[j][:2]) for s in specs]
    new_k = {k: i for i, k in enumerate(order)}
    new_rank = [{j: i for i, j in enumerate(r)} for r in ranks]
    out = []
    for m in ms:
        if m.template not in segs:
            out.append(m)
            continue
        links = tuple((new_k[k], new_rank[k][rank]) for a, b, k, rank in segs[m.template] if a < m.base + m.n and b > m.base)
        out.append(replace(m, tables=((CS.LINKED, len(links)), ("w", m.n)), links=links))
    return out, tuple(CS.RowTable(specs[k][0], specs[k][1], specs[k][2], specs[k][3], tuple(specs[k][4][j] for j in ranks[k]))
                      for k in order)


def law_of(law: str | Mapping[str, Any]) -> dict[str, Any]:
    return DR.parse_law(law) if isinstance(law, str) else dict(law)


def population(programs: dict[str, dict[str, Any]], partition: Mapping[str, Any], run_root: str,
               program_dirs: dict[str, str] | None = None, index: Mapping[str, Any] | None = None,
               arrivals: Mapping[str, int] | None = None, context: str = "") -> tuple[CS.Window, list[list[Row]]]:
    """The verifier's window over the run's Programs (``partition``: the partition file's ``templates``, ``law`` and ``scope``) and
    each template's rows; refused when the file names a ``partition_digest`` that is not the window's, or a ``query`` other than the
    template queries.  ``row_tables: true`` in the file: the ``ScaledMmS8Coordinate`` chunks in the row-table layout
    (`with_row_tables`, its engine steps the requests' ``arrivals`` plus their steps), under ``context`` (hex: the live coin
    commitment's root serving committed its step roots under)."""
    tmpls, law = list(partition["templates"]), law_of(partition["law"])
    q = partition.get("query")
    if q is not None and q not in TEMPLATE_QUERIES:
        if q == P.Q_WORD[0] and law.get("law") == "stratified":
            raise ValueError(f"a word partition ({P.Q_WORD[0]}) with the stratified law: its strata are templates, which a word "
                             f"partition does not have; the serving rows' partition is {' / '.join(TEMPLATE_QUERIES)}")
        raise ValueError(f"partition query {q!r}: the serving rows' window is the template instances' partition "
                         f"({' / '.join(TEMPLATE_QUERIES)}), never a word partition")
    scope = partition.get("scope") or {}
    rows, defs, counts = member_rows(programs, tmpls, scope.get("op_path_prefix"), program_dirs, index, scope.get("steps"))
    ms = members(tmpls, rows, counts, partition.get("chunk_cap"))
    tables: tuple[CS.RowTable, ...] = ()
    if partition.get("row_tables"):
        ms, tables = with_row_tables(ms, tmpls, rows, arrivals)
    n_of = {t: sum(m.n for m in ms if m.template == t) for t in tmpls}
    program, query = CS.population_program([(fn, n_of[t]) for fn, t in zip(defs, tmpls, strict=True)], several=q == TEMPLATE_QUERIES[1])
    window = CS.Window(program, query, law, run_root, tuple(ms), tables, context)
    want = partition.get("partition_digest")
    if want and want != window.partition_digest:
        raise ValueError(f"the run's population ({', '.join(f'{t} x {n}' for t, n in n_of.items())}) is partition "
                         f"{window.partition_digest[:16]}, not the verifier's {str(want)[:16]}")
    return window, rows


def partition_doc(window: CS.Window, partition: Mapping[str, Any]) -> dict[str, Any]:
    """The partition file as derived: the verifier's templates, law and scope, with the members, the partition object and digest."""
    return {"format": PARTITION_FORMAT, "templates": list(dict.fromkeys(m.template for m in window.members)), "law": window.law,
            "query": window.query["name"], "scope": partition.get("scope"), "members": [[m.key, m.n] for m in window.members],
            **({"chunk_cap": partition["chunk_cap"]} if partition.get("chunk_cap") else {}),
            **({"row_tables": [t.port for t in window.row_tables]} if window.row_tables else {}), "partition": window.partition,
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


def _as(a: np.ndarray, itemsize: int, n: int, what: str) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    if a.dtype.itemsize != itemsize or a.size != n:
        raise ValueError(f"{what}: {a.size} {a.dtype} elements, not {n} of {itemsize} bytes")
    return a.view(np.dtype(f"<u{itemsize}") if itemsize > 1 else np.uint8)


def _read_s8(ms: Sequence[CS.Member], rows: list[Row], store: Any, weights: Callable,
             arrivals: dict[str, int] | None) -> list[tuple[dict[str, Any], list[list[Any]]]]:
    """A ``ScaledMmS8Coordinate`` template's chunks: coordinate ``c`` of a ``ScaledMmS8_v1`` row (token row ``x`` with scale
    ``sx``, weight rows ``W`` with scales ``sw``) is ``x`` = the codes' bytes then ``sx``'s, ``w`` = ``W[c]``'s bytes then
    ``sw[c]``'s (u16 little-endian words, as A6 stages them), ``o0`` = ``y[c]``; units go row by row, then coordinate.  Each
    chunk's index entries are the rows whose units it holds, each with the row's whole range.  A linked chunk (the row-table layout)
    reads its ``x`` table, the distinct token rows its units read, and refs: unit ``i`` reads ``x`` row ``ref`` and its own ``w``
    row."""
    K = int(rows[0][3]["statics"]["K"])
    ws, ys, entries, wcache, xrows = [], [], [], {}, []
    lo = ms[0].base
    for rid, pi, r, d in rows:
        arrive, N = int((arrivals or {}).get(rid, 0)), int(d["statics"]["N"])
        x = _as(_operand(pi, store, weights, rid, arrive, r["args"][0]), 1, K, f"row {r['i']} x")
        sx = _as(_operand(pi, store, weights, rid, arrive, r["args"][1]), 4, 1, f"row {r['i']} sx")
        key = json.dumps([r["args"][2], r["args"][3]])
        if key not in wcache:
            W = _as(_operand(pi, store, weights, rid, arrive, r["args"][2]), 1, N * K, f"row {r['i']} w").reshape(N, K)
            sw = _as(_operand(pi, store, weights, rid, arrive, r["args"][3]), 4, N, f"row {r['i']} sw")
            wcache[key] = np.ascontiguousarray(np.concatenate([W, sw.astype("<u4").view(np.uint8).reshape(N, 4)], axis=1)).view("<u2")
        xrow = np.frombuffer(x.tobytes() + sx.astype("<u4").tobytes(), dtype="<u2")
        (member, width, _dt), = d["members"]
        y, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if y is None:
            raise LookupError(f"row {r['i']} output {member}: {why}")
        xrows.append(xrow)
        ws.append(wcache[key])
        ys.append(_as(y, 2, N, f"row {r['i']} y").reshape(N, 1))
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + N, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"], str(member),
                        int(d["row"]), int(r["i"])])
        lo += N
    Wr, Y = np.concatenate(ws), np.concatenate(ys)
    linked = all(m.links for m in ms)
    if not linked:
        X = np.concatenate([np.broadcast_to(x, (e[1] - e[0], x.size)) for x, e in zip(xrows, entries, strict=True)])
    out, b0 = [], ms[0].base
    for m in ms:
        a, b = m.base - b0, m.base - b0 + m.n
        mine = [j for j, e in enumerate(entries) if e[0] < m.base + m.n and e[1] > m.base]
        if linked:
            if len(mine) != len(m.links):
                raise ValueError(f"{m.key}: reads {len(mine)} token rows, links {len(m.links)}")
            at = np.repeat(np.arange(len(mine)), [min(entries[j][1], m.base + m.n) - max(entries[j][0], m.base) for j in mine])
            got = {"rows": {"x": np.stack([xrows[j] for j in mine]), "w": Wr[a:b]}, "outs": {"o0": Y[a:b]},
                   "refs": np.stack([at, np.arange(m.n)], axis=1)}
        else:
            got = {"rows": {"x": X[a:b], "w": Wr[a:b]}, "outs": {"o0": Y[a:b]}}
        out.append((got, [entries[j] for j in mine]))
    return out


def read_all(ms: Sequence[CS.Member], rows: Sequence[list[Row]], store: Any, weights: Callable,
             arrivals: dict[str, int] | None = None) -> list[tuple[dict[str, Any], list[list[Any]]]]:
    """Every member's words and index entries (`read`), ``rows`` each template's in the window's order; a chunked template is read
    once."""
    out, at = [], 0
    for rs in rows:
        t = ms[at].template
        mine = [m for m in ms[at:] if m.template == t]
        out += _read_s8(mine, rs, store, weights, arrivals) if t.startswith(S8) else [read(mine[0], rs, store, weights, arrivals)]
        at += len(mine)
    return out


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
                 index: Mapping[str, Any] | None = None, row_tables: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Commit the verifier's partition's units of a run whose vllm-v1 roots are final and write the window's files into ``out_dir``:
    the registration (core's v1 record, its window the run root ``run_root`` and the provenance: the research run ``run``, the row,
    the committer and the served Programs' ``digests``), ``partition.json``, the index and M0's bodies.  The window is ``served``;
    with ``stand_in`` (its ``of`` and ``link``), a ``stand-in`` with them in its provenance.  A TP rank's window (`pipeline.tp.
    serving_rows`) gives its run as a string, its world and roots as ``tp`` and its ProgramIndex arguments as ``index``.  A
    row-table layout takes the serving loop's ``row_tables``: ``{"context": <hex>, "key": <the run's row key>, "step_roots":
    {engine step: root}}`` (`commit.row_tables.step_roots`), which the window's step roots must be.  It reads the committed values
    and writes only ``out_dir``."""
    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    pf = json.loads(Path(partition_file).read_text())
    rt = row_tables or {}
    window, rows = population(programs, pf, run_root if isinstance(run_root, str) else bytes(run_root).hex(), dirs, index, arrivals,
                              rt.get("context", ""))
    ms = window.members
    store = CommittedStore.of_record(bmap, committed, manifest)
    got = read_all(ms, rows, store, weights, arrivals)
    t_read = time.perf_counter()
    c = CS.commit(window, [g[0] for g in got], row_salt_key=rt.get("key"), step_roots=rt.get("step_roots"),
                  workers=workers or min(16, os.cpu_count() or 1))
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
          + ", ".join(f"{m.key} {m.n}" for m in ms) + f"; registration {res['registration_digest'][:16]}; {res['seconds']}", flush=True)
    return res

