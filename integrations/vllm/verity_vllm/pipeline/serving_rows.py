"""The serving rows of a Commit: the units of the verifier's partition, read exactly as the replay reads them (committed members
through an opened reader, registered weights for weight slices), committed in M0's serving row format by
:mod:`verity.ml.windows.rows`, with core's one-stage registration over them.  Three ways in, one window, two kinds:

* **In the Commit** (opt-in, ``commit --serving-rows``): the row stage writes a request ``{"dir", "partition", "run",
  "committer"}``, and the in-process C2 replay commits the window after the Commit's roots are final and before its seed (``readers``);
  a Commit whose replay is deferred to its bundle (``--replay-deferred``, not slim) commits it at its C2 site, before the bundle is
  sealed and before sampling (`c2_replay.serving_rows`). The window is ``served``: its roots are serving's own commitment, made before sampling.  The profile makes no claim that they
  hold the values under the Commit's roots.
* **In a TP Commit** (``tp-commit --serving-rows``, `pipeline.tp.serving_rows`): one ``served`` window per rank, committed in the
  rank's process once every rank's root is final and the TP run root fixed, before the challenge; its run is the TP run root and
  the rank.
* **From a deferred Commit's bundle** (``verity-vllm serving-rows BUNDLE --row-dir D --partition F --out O``,
  `pipeline.deferred_rows`): the sealed replay bundle opened against the Commit's record as the CPU replay opens it, and the same
  window written into ``O`` as a ``stand-in``: serving's commitment is the record's roots, and these rows are committed afterwards
  from the bundle, so their link to those roots is provenance (``deferred_rows.stand_in``).

The partition file is the verifier's: ``{"templates": [<descriptor id>, ...], "law": "subset:K" | "stratified:K" | "work:K" | {...},
"query"?: "Q_template_instance" | "Q_template_instances", "scope"?: {"op_path_prefix", "steps"}, "chunk_cap"?: {<descriptor id>:
cap}, "row_tables"?: true, "partition_digest"?}``.  A ``work:K`` window samples by the work table `work_table_of` derives from the
registered compiler's transfer-gate templates (`pipeline.advice_windows`): every unit of those is proved (an All stratum), the rest
sampled as the stratified law samples it.  The members are its templates in order, each instance count read off the Build's
Programs; the population program over them (``ml.windows.population.population_program``: ``TemplatePopulations`` for several
templates, or for one when the file names ``Q_template_instances``) is the one the verifier builds from the same Programs, and a
``partition_digest`` in the file that is not that program's partition is refused, as is any other query (a word partition,
``Q_word``, never: with the stratified law it is refused by name, its strata being templates).  ``partition.json`` beside the window
is the derived file, digest and members included.  ``scope.op_path_prefix`` restricts the population to one module
(``model.layers.0.``, or a MoE layer's ``moe/L40``), or to the rows under any of a list of them, ``scope.steps`` (``[lo, hi)``)
to those steps of each request.  A chunked template's units (``CHUNKED``: a ``ScaledMmS8Coordinate``, a ``COORDS`` coordinate,
a ``COLS`` column block, a ``SLICES`` slice, a ``KCHUNKS`` K-chunk or a ``SPLITS`` split part or combine; and a row template or
whole ``PARTS`` part the file's ``chunk_cap`` names, its rows its Definition's own ports, `chunked`) are split
into consecutive chunks of at most its ``chunk_cap`` (flock-circuit's spec shape, as ``benchmarks/one_stage/a6.py`` computes
it; ``chunk_caps`` writes the entry), each chunk a member on C-Flock's bit rows (`chunk_rows.unit_rows`) with its public file of global unit indices;
``row_tables`` puts the ``ScaledMmS8Coordinate`` chunks in the row-table layout (``with_row_tables``).

A template is read through its word link (:func:`template`): a Boolean Definition (the definition) names the word Definition it
computes (`verity.ml.circuits.boolean.trace.word_of`, held equal to it by circuit-check's word check), and a word Definition is its
own word.  A Program row holds instances of a template when the template's word is the row's own Definition (its spec), or when the
row's Definition is cut into them (``CUT``: its body batches the template's word, read off the bound Definition, so a template that
does not compute what the row computes is no match), or when the template's word is the row's Definition at N = 1 (``COORDS``: the
row is split into its N coordinates), or at a halving of its width (``COLS``: the row is split into column blocks), or when it
is a part of the row's Definition (``PARTS``: the row is split into Definitions of its own, one member per part), or when it
is the row's Definition at every axis of a batch 1 (``SLICES``: the row is split into its slices along those axes), or when it
is a K-chunk of the row's coordinate (``KCHUNKS``, `kchunk_rows`) or a split part or the combine of the row's Definition
(``SPLITS``, `split_rows`).  A Boolean Program's rows are its word view (`pipeline.build_instances`: each row the word Definition
its Call's Boolean Definition computes, that Boolean Definition its ``bits``), so a template matches the same rows under either IR,
and the members, their rows and every committed word are the same; the population program is over the template as the partition
names it.  The rows commit u16 words: a template's every port is 16-bit elements in its word view (a
Boolean element's bit j is bit j of its word, `boolean.elements`), or, for a row or coordinate template, elements of ``PACKED`` widths with each
port whole u16 words (``Template.packed``): a port's words are then its elements' little-endian bytes, two to a word, which are
the port's bits in M0's row order (an 8-bit element is half a word, a 32-bit one two words, low half first); refused otherwise.
A committed member of a wider signed integer dtype than its port (``BY_VALUE``: the sampler's i64 token under `TokenSelect`'s
i32 output, and under the vocabulary shard's i32 ``tok`` input) is read by value, as the replay links tokens, each element below
2^bits or refused.
Each member's ports, by the family of the template's word (``LAYOUTS``):

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
  ``MoeExpertFp8Gemm(W)_v1{N=1,..}``  every coordinate n of the rows of that Definition at any N (the expert-indexed pair GEMMs,
                                   ``FlashInferFp8Linear_v1{K=..,N=1}`` and ``ScaledMmFp8BlockSharedScale_v1{K=..,N=1,G=..}``,
                                   ``COORDS``): ``in<g>`` per argument group (a ``PARTS`` part's, the operands its entry
                                   names), the leading ones whole and the last two ``w[:, n, :]`` and ``ws[:, n // G, :]`` of
                                   every expert (FlashInfer's linear and the block-scaled GEMM: ``w[n]`` and ``ws[n // G]``)
                                   -> ``out.<member>``, the row's output
                                   word n (`part_rows.read_coord`); packed ports, one index entry per row; chunked (``chunk_cap``),
                                   v2 bit rows (`chunk_rows.unit_rows`) -> ``o0``
  ``MoeExpertFp8CoordinateChunk_v1{K=k,..}``, ``SparseMlaSplitPart_v1{T=k,..}``, ``SparseMlaCombine_v1{NSPLIT=n,..}``  K-chunks
                                   (``KCHUNKS``) and split parts and combines (``SPLITS``), laid out by `kchunk_rows`, `split_rows`
                                   and `chunk_rows.reads`; always chunked, v2 bit rows, one index entry per row
  ``MoeSum_v1{H=w,..}``, ``AllReduceMnnvl_v1{N=w,..}``  every block j of w columns of the rows of that Definition W wide, W / w
                                   a power of two (``COLS``): ``in<g>`` per argument group, each of its rows' columns [jw, jw + w)
                                   in order -> ``out.<member>``, the row's output columns [jw, jw + w); one index entry per row;
                                   chunked (``chunk_cap``), v2 bit rows (`chunk_rows.unit_rows`) -> ``o0``
  ``AllReduceMultimem_v1{SIZE=w,..}``  likewise along SIZE, for a row whose owners' TREES are one tree and w whole 8-element
                                   vectors (`cols_refused`); a member of blocks handed another row refuses it
  ``AllReduceResidualMnnvl_v1{N=w,..}``, ``RmsNormMnnvl_v1{..}``  the parts of every ``AllReduceRmsNormMnnvl_v1`` row (``PARTS``):
                                   its all-reduce and residual add in column blocks as ``COLS`` (``in0`` the partials, ``in1``
                                   the residual -> ``out.<member>``, the row's r), its norm whole (``in0`` the row's committed r,
                                   ``in1`` the weight -> ``out.<member>``, the row's normed output); one index entry per row
  ``Fp8GroupQuantUe8m0_v1{..}``, ``ScaledMmFp8BlockSharedScale_v1{N=1,..}``  the parts of every ``DeepGemmFp8Linear_v1`` row
                                   (``PARTS``, at G = 128, ``PART_STATICS``): its quantisation whole (``in0`` the row's x ->
                                   ``out.<member>``, its q and s; packed ports, `part_rows.read_part`), its GEMM in coordinates as
                                   ``COORDS`` (``in0``, ``in1`` the row's committed q and s, ``in2``, ``in3`` ``w[n]`` and
                                   ``ws[n // 128]`` -> ``out.<member>``, its y word n); one index entry per row
  ``Gemm_v2{N=1,..}``, ``BatchedGemm_v1{B=1,N=1,..}``, ..  every slice of the rows of that Definition at any extents along its
                                   axes (``SLICES``): ``in<g>`` per argument group, its element along each axis it carries and
                                   whole along the others (a GEMM coordinate: ``x`` whole, ``w[n]``) -> ``out.<member>``, the
                                   row's output element; packed ports, one index entry per row; chunked (``chunk_cap``), v2
                                   bit rows (`chunk_rows.unit_rows`) -> ``o0``
  any row's own Definition         every row of it, the row itself: ``in<g>`` per argument group -> ``out.<member>`` per
                                   committed member (``vu_export._decompose_row``'s port rule); chunked when the file's
                                   ``chunk_cap`` names it, v2 bit rows (`chunk_rows.unit_rows`, its MUFU table reads laid out
                                   by `chunk_rows.reads`) -> ``o<j>`` per member

so ``RoPEHead_v2{D=..}`` and ``GemmCoordinate_v4{K=..}``, the Boolean versions, are a RoPE head's and a GEMM coordinate's.

Units go in request order, then Program row index, then head or coordinate (``ml.windows.rows.ORDER``).  In a prefill step the Program lists
a layer's q rows for every token before its k rows, so RoPE's heads are q heads for every prefill token, then k heads.
"""
from __future__ import annotations

import hashlib
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

from verity.core.protocols.sampled_proofs.one_stage import draw as DR, partition as P, registration as R
from verity.ml.circuits import partition_object as PO
from verity_vllm.check.replay.index import ProgramIndex, index_of
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.commit import index as IX, row_tables as RTB
from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import kchunk_rows as KC
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline import split_rows as SP
from verity_vllm.pipeline.chunk_rows import s8_words
from verity_vllm.pipeline.serving_templates import (COLS, COORDS, CUT, LAYOUTS, PACKED, PART_OF, PARTS, SLICES, SPLITS,  # noqa: F401
                                                    Template, _bind, part, template, unit_templates)
from verity_vllm.program.compact import runs_from_file_items

#: the layouts whose units are bit rows as C-Flock's `UnitStager` stages them (`chunk_rows`), in chunks of at most a chunk cap
CHUNKED = ("s8", "coord", "cols", "slice", "kchunk", "split")
#: the chunked layouts read by a module of their own: a row's unit count, and the reader of a member's rows
READERS = {"slice": (SL.extent, SL.read), "kchunk": (KC.units, KC.read), "split": (SP.units, SP.read)}
#: the committed signed integer dtypes a narrower packed port reads by value (`_packed`): torch's int64 indices
BY_VALUE = frozenset(("i64", "int64"))
PARTITION_FORMAT = "verity-vllm/serving-rows/partition/v1"
#: the circuit record's ``name`` (`circuit_record`), opaque to Verity
CIRCUIT_NAME = "verity-vllm"
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


Row = tuple[str, Any, dict, dict]


def member_rows(programs: dict[str, dict[str, Any]], tmpls: Sequence[str], scope: str | Sequence[str] | None = None,
                program_dirs: dict[str, str] | None = None,
                index: Mapping[str, Any] | None = None,
                steps: Sequence[int] | None = None,
                reuse: Sequence[ProgramIndex] = ()) -> tuple[list[list[Row]], list[Any], list[int]]:
    """Per template, its rows in order (request, its ProgramIndex, the row, its description), restricted to ``scope`` (an op_path
    prefix, or a list of them: a row under any) and ``steps`` (``[lo, hi)`` of the request's steps); each template's Definition;
    and its instance count.  Rows are matched by the template's word (:func:`template`), so two templates that compute one word
    Definition are refused.  Fixed by the Build's Programs before serving.  ``index``: the ProgramIndex's ``attention_paths`` and ``tp`` for a rank Program
    (`pipeline.tp.serving_rows`); ``reuse``: indexes already built, each taken for the Program it was built from with these
    arguments (`index_of`)."""
    ts = [template(t) for t in tmpls]
    prefixes = (scope,) if isinstance(scope, str) else tuple(scope or ())
    want: dict[str, int] = {}
    for k, t in enumerate(ts):
        if want.setdefault(t.word, k) != k:
            raise ValueError(f"{tmpls[want[t.word]]} and {t.id} both compute {t.word}: one member reads a row")
    out: list[list[Row]] = [[] for _ in tmpls]
    counts = [0] * len(tmpls)
    seen: list[dict | None] = [None] * len(tmpls)
    pool = list(reuse)
    for rid, inst in programs.items():
        pi = index_of(inst, pool, program_dir=(program_dirs or {}).get(str(rid)), **(index or {}))
        pool.append(pi)
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            hits = [(want[w], n) for w, n in unit_templates(r["spec"]) if w in want]
            if not hits:
                continue
            d = pi.describe(r)
            if d is not None:
                for k, _n in hits:
                    seen[k] = seen[k] or d
            if d is None or (prefixes and not d["op_path"].startswith(prefixes)):
                continue
            if steps is not None and not int(steps[0]) <= int(d["step"]) < int(steps[1]):
                continue
            ps = [part(ts[k].word, r["spec"]) for k, _ in hits]
            chain = {KC.chain(ts[k])[0] if ts[k].layout == "kchunk" else "split" if ts[k].layout == "split" else None for k, _ in hits}
            if len(hits) > 1 and (None in ps or len({p[0] for p in ps}) < len(ps)) and (None in chain or len(chain) > 1):
                raise ValueError(f"row {i} ({r['spec']}) holds instances of {[tmpls[k] for k, _ in hits]}: one member reads a row, "
                                 f"or one member per part of it (PARTS), or one per K-chunk template of one chain (KCHUNKS), or one "
                                 f"per split part and combine template of it (SPLITS)")
            for k, n in hits:
                out[k].append((str(rid), pi, r, d))
                counts[k] += n
    for t, rows, d in zip(tmpls, out, seen, strict=True):
        if not rows:
            raise LookupError(f"no rows of {t} in the run's Programs" + (f" under {' or '.join(prefixes)}" if prefixes else "")
                              + (f" at steps {list(steps)}" if steps is not None else "")
                              + (f"; its rows exist outside the scope, e.g. op_path {d['op_path']} at step {d['step']} (scope.op_path_prefix "
                                 "is the op_path's prefix, or a list of them, the wrapper's module prefix stripped)" if d is not None else ""))
    return out, [t.fn for t in ts], counts


def chunk_caps(tmpls: Sequence[str]) -> dict[str, int]:
    """`chunk_rows.chunk_cap` of each chunked template (``CHUNKED``), for a partition file's ``chunk_cap``."""
    return {t: CR.chunk_cap(template(t)) for t in tmpls if template(t).layout in CHUNKED}


def chunked(t: str, caps: Mapping[str, int] | None) -> bool:
    """Whether template ``t``'s units are chunks on bit rows (`chunk_rows.unit_rows`): a ``CHUNKED`` layout's (a window names the
    caps), or a row template's or a whole ``PARTS`` part's when ``caps`` names it, its rows staged as the Definition's own ports
    (no input set read)."""
    layout = template(t).layout
    return ((layout in CHUNKED and (layout in ("s8", "kchunk", "split") or caps is not None))
            or (layout in ("row", "part") and t in (caps or {})))


def members(tmpls: Sequence[str], rows: Sequence[list[Row]], counts: Sequence[int] | None = None,
            caps: Mapping[str, int] | None = None) -> list[CS.Member]:
    """Each template's ports, instance count and base, read off its first row; ``counts`` (the Definitions' own) must agree.  A
    chunked template's units (`chunked`: a ``CHUNKED`` layout's, or a row template's that ``caps`` names) are consecutive chunks of
    at most its ``caps`` entry, one member each, on its `chunk_rows.unit_rows`; a window always names the caps (`population`).
    With no ``caps`` at all (a tool reading one Call's units, ``benchmarks/one_stage/m1_unit.py``), a ``COORDS`` or ``COLS``
    template is one member of word rows."""
    from verity_vllm.query.program_view import spec_base, spec_statics

    out, base = [], 0
    for k, (t, rs) in enumerate(zip(tmpls, rows, strict=True)):
        _rid, _pi, r, d = rs[0]
        layout = template(t).layout
        if layout == "cols":
            axis = COLS[spec_base(template(t).word)]
            n = sum(int(x[3]["statics"][axis]) // int(spec_statics(template(t).word)[axis]) for x in rs)
        elif layout in READERS:
            n = sum(READERS[layout][0](template(t), x[3]["statics"]) for x in rs)
        elif layout in CHUNKED:
            n = sum(int(x[3]["statics"]["N"]) for x in rs)
        elif layout in ("row", "part"):
            n = len(rs)
        if chunked(t, caps):
            cap = (caps or {}).get(t)
            if not isinstance(cap, int) or cap < 1:
                raise ValueError(f"{t}: no chunk cap; a partition file names each chunked template's flock-circuit chunk cap as "
                                 f"chunk_cap (pipeline.serving_rows.chunk_caps)")
            if counts is not None and counts[k] != n:
                raise ValueError(f"{t}: {n} units read off the rows, {counts[k]} in their Definitions")
            in_rows, _outs = CR.unit_rows(template(t))
            outs = (("o0", 1),) if layout == "s8" else tuple((f"o{j}", -(-sum(f.leaf_widths()) // 16))
                                                             for j, f in enumerate(CR.fields(template(t))))
            schemas = tuple((p, s) for p, s, _b, _ps in in_rows) + tuple((p, CS.SCHEMA_HM96_SHA512_ROW_V2) for p, _w in outs)
            out += [CS.Member(t, tuple((p, b // 16) for p, _s, b, _ps in in_rows), outs, min(cap, n - lo), base + lo, name=f"{t}/c{j}",
                              schemas=schemas)
                    for j, lo in enumerate(range(0, n, cap))]
            base += n
            continue
        if layout == "rope":
            D = int(d["statics"]["D"])
            n = sum(int(x[3]["statics"]["NHEADS"]) for x in rs)
            m = CS.Member(t, (("x", D), ("cs", D)), (("out", D),), n, base)
        elif layout == "gemm":
            K, groups = int(d["statics"]["K"]), grid(rs)
            rx, rw = sum(g["tokens"] for g in groups), sum(g["columns"] for g in groups)
            m = CS.Member(t, (("x", K), ("w", K)), (("y", 1),), sum(g["tokens"] * g["columns"] for g in groups), base,
                          tables=(("x", rx), ("w", rw)))
        elif layout in ("coord", "cols", "slice") or (layout == "part" and part(template(t).word, r["spec"]) is not None):
            params, ret = template(t).fn.signature()
            if layout == "part":
                n = len(rs)
            p = part(template(t).word, r["spec"])
            m = CS.Member(t, tuple((f"in{g}", sum(ty.leaf_widths()) // 16) for g, (_n, ty) in enumerate(params)),
                          tuple((f"out.{d['members'][j][0]}", sum(ty.leaf_widths()) // 16)
                                for j, ty in zip(p[2] if p else (0,), getattr(ret, "fields", None) or (ret,), strict=True)), n, base)
        else:
            ins = [sum(int(sl[3]) - int(sl[2]) for sl in runs_from_file_items(grp)) for grp in r["args"]]
            outs = [int(mm[1]) for mm in d["members"]]
            packed = template(t).packed
            if packed is not None:
                if len(packed) != len(ins) + len(outs) or any(n * b % 16 for n, b in zip(ins + outs, packed)):
                    raise ValueError(f"{t}: argument groups of {ins} and committed members of {outs} elements are not its ports' "
                                     f"whole u16 words ({list(packed)}-bit elements)")
                words = [n * b // 16 for n, b in zip(ins + outs, packed, strict=True)]
                ins, outs = words[:len(ins)], words[len(ins):]
            groups = tuple((f"in{g}", n) for g, n in enumerate(ins))
            m = CS.Member(t, groups, tuple((f"out.{mm[0]}", n) for mm, n in zip(d["members"], outs, strict=True)), len(rs), base)
        if counts is not None and counts[k] != m.n:
            raise ValueError(f"{t}: {m.n} units read off the rows, {counts[k]} in their Definitions")
        out.append(m)
        base += m.n
    return out


def _one_table_chunks(first: CS.Member, segs: Sequence[tuple[int, int, int, int]]) -> list[CS.Member]:
    """A template's linked chunks over its units ``segs`` (each token row's units ``[a, b)``, its row table and rank, in
    unit order), from its first chunk ``first``: at most ``first.n`` units each, every one reading one row table."""
    chunks: list[tuple[int, list[tuple[int, int]], list[int]]] = []
    for a, b, k, rank in segs:
        at = a
        while at < b:
            if not chunks or chunks[-1][1][-1][0] != k or sum(chunks[-1][2]) == first.n:
                chunks.append((at, [], []))
            take = min(b - at, first.n - sum(chunks[-1][2]))
            chunks[-1][1].append((k, rank))
            chunks[-1][2].append(take)
            at += take
    return [replace(first, name=f"{first.template}/c{j}", base=base, n=sum(units), tables=((CS.LINKED, len(links)), ("w", sum(units))),
                    links=tuple(links), link_units=tuple(units))
            for j, (base, links, units) in enumerate(chunks)]


def with_row_tables(ms: Sequence[CS.Member], tmpls: Sequence[str], rows: Sequence[list[Row]],
                    arrivals: Mapping[str, int] | None = None) -> tuple[list[CS.Member], tuple[CS.RowTable, ...]]:
    """The row-table layout of ``members``' ``ScaledMmS8Coordinate`` chunks: one row table per (engine step, op_path) of their
    ``ScaledMmS8_v1`` rows in the canonical order serving commits them in (``row_tables.table_key``; each table's token rows by request,
    then token row), and each template's units cut again into chunks of at most its chunk cap (its first chunk's size in
    ``members``) that each read one row table: a chunk ends at its cap or where its units' next token row is another
    table's.  Each chunk is a shared-row member whose ``x`` table is the distinct token rows its units read (``links``: each
    one's row table and rank, in the units' order; ``link_units``: how many of its units read each), all rows of one table
    (the one root a statement's registered reads open against), and whose ``w`` table is its own rows."""
    tables: dict[tuple[int, str], int] = {}
    specs: list[tuple[int, str, str, int, list[tuple[str, int, int]]]] = []
    segs: dict[str, list[tuple[int, int, int, int]]] = {}
    for t, rs in zip(tmpls, rows, strict=True):
        if template(t).layout != "s8":
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
    order = sorted(range(len(specs)), key=lambda k: (specs[k][0], *RTB.table_key(specs[k][1])))
    ranks = [sorted(range(len(s[4])), key=lambda j, rs=s[4]: rs[j][:2]) for s in specs]
    new_k = {k: i for i, k in enumerate(order)}
    new_rank = [{j: i for i, j in enumerate(r)} for r in ranks]
    out: list[CS.Member] = []
    for m in ms:
        if m.template not in segs:
            out.append(m)
        elif m.base == next(x.base for x in ms if x.template == m.template):
            out += _one_table_chunks(m, [(a, b, new_k[k], new_rank[k][rank]) for a, b, k, rank in segs[m.template]])
    made = tuple(RTB.row_table(specs[k][0], specs[k][1], specs[k][2], specs[k][3], [specs[k][4][j] for j in ranks[k]])
                 for k in order)
    RTB.check_order(made)
    return out, made


def law_of(law: str | Mapping[str, Any]) -> dict[str, Any]:
    return DR.parse_law(law) if isinstance(law, str) else dict(law)


def work_table_of(law: Mapping[str, Any], counts: Mapping[str, int], transfer_gates: Sequence[str] = ()) -> dict[str, Any] | None:
    """The verifier's work table (`draw.work_table`) for a window of ``counts`` (template -> its units, in the partition's order)
    under ``law``: under ``work:K``, each transfer-gate template (the units holding transfer gates, which the registered compiler
    lists) works 0 with its floor all its units, so every one of them is in the sample (`draw.work_k`: selection All, never
    sampled), and every other template works 1 with floor 1, its sample the stratified law's share of K; None for any other law.
    Refused when the compiler lists transfer gates under another law, when one of its transfer-gate templates is not among the
    window's (its units would be left out of the proved set), or when every template is one (nothing is sampled: that window's
    selection is all)."""
    tg = list(dict.fromkeys(transfer_gates))
    if law.get("law") != "work":
        if tg:
            raise ValueError(f"the compiler lists transfer-gate templates {tg}: every unit of theirs is proved beside the drawn rest, "
                             f"which is the work law's draw (law work:K), not {dict(law)}")
        return None
    if missing := [t for t in tg if t not in counts]:
        raise ValueError(f"the compiler's transfer-gate template {missing[0]} is not among the partition's templates {list(counts)}: "
                         "its units would be left out of the proved set")
    if all(t in tg for t in counts):
        raise ValueError(f"every template of the window ({list(counts)}) holds transfer gates: nothing is sampled, so its selection is "
                         "all, not a work law")
    return {t: {"work": 0, "floor": int(n)} if t in tg else {"work": 1, "floor": 1} for t, n in counts.items()}


def population(programs: dict[str, dict[str, Any]], partition: Mapping[str, Any], run_root: str,
               program_dirs: dict[str, str] | None = None, index: Mapping[str, Any] | None = None,
               arrivals: Mapping[str, int] | None = None, context: str = "",
               reuse: Sequence[ProgramIndex] = (), transfer_gates: Sequence[str] = ()) -> tuple[CS.Window, list[list[Row]]]:
    """The verifier's window over the run's Programs (``partition``: the partition file's ``templates``, ``law`` and ``scope``) and
    each template's rows; refused when the file names a ``partition_digest`` that is not the window's, or a ``query`` other than the
    template queries.  ``row_tables: true`` in the file: the ``ScaledMmS8Coordinate`` chunks in the row-table layout
    (`with_row_tables`, its engine steps the requests' ``arrivals`` plus their steps), under ``context`` (hex: the live coin
    commitment's root serving committed its step roots under).  ``reuse``: `member_rows`'.  ``transfer_gates``: the registered
    compiler's transfer-gate templates, whose units a ``work:K`` window proves every one of (`work_table_of`)."""
    tmpls, law = list(partition["templates"]), law_of(partition["law"])
    q = partition.get("query")
    if q is not None and q not in TEMPLATE_QUERIES:
        if q == P.Q_WORD[0] and law.get("law") == "stratified":
            raise ValueError(f"a word partition ({P.Q_WORD[0]}) with the stratified law: its strata are templates, which a word "
                             f"partition does not have; the serving rows' partition is {' / '.join(TEMPLATE_QUERIES)}")
        raise ValueError(f"partition query {q!r}: the serving rows' window is the template instances' partition "
                         f"({' / '.join(TEMPLATE_QUERIES)}), never a word partition")
    scope = partition.get("scope") or {}
    rows, defs, counts = member_rows(programs, tmpls, scope.get("op_path_prefix"), program_dirs, index, scope.get("steps"), reuse)
    ms = SP.wire(KC.wire(members(tmpls, rows, counts, partition.get("chunk_cap") or {}), tmpls, rows), tmpls, rows)
    tables: tuple[CS.RowTable, ...] = ()
    if partition.get("row_tables"):
        ms, tables = with_row_tables(ms, tmpls, rows, arrivals)
    n_of = {t: sum(m.n for m in ms if m.template == t) for t in tmpls}
    program, query = CS.population_program([(fn, n_of[t]) for fn, t in zip(defs, tmpls, strict=True)], several=q == TEMPLATE_QUERIES[1])
    table = work_table_of(law, n_of, transfer_gates)
    if table is not None:
        table = {P.template_id(fn): table[t] for fn, t in zip(defs, tmpls, strict=True)}
    window = CS.Window(program, query, law, run_root, tuple(ms), tables, context, table)
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
            **({"row_tables": [t.port for t in window.row_tables]} if window.row_tables else {}),
            **({"work_table": window.work_table} if window.work_table is not None else {}), "partition": window.partition,
            "partition_digest": window.partition_digest, "population": window.population}


def circuit_record(window: CS.Window) -> dict[str, Any]:
    """The window's ``circuit.json`` (`interface.circuit_record`): each chunked template's statement its `chunk_rows.unit_rows`
    and `chunk_rows.reads`, as ``compose_circuit.py --template`` composes it; a template of word rows has none (the verifier's
    input set lowers it)."""
    sts = {}
    for t in dict.fromkeys(m.template for m in window.members if m.schemas):
        in_rows, out_rows = CR.unit_rows(template(t))
        sts[t] = {"in_rows": in_rows, "out_rows": out_rows, "reads": CR.reads(template(t))}
    return IF.circuit_record(window, CIRCUIT_NAME, IF.templates_of(window), sts)


def read(m: CS.Member, rows: list[Row], store: Any, weights: Callable,
         arrivals: dict[str, int] | None = None) -> tuple[dict[str, dict[str, np.ndarray]], list[list[Any]]]:
    """Member ``m``'s words, read from the committed store and the registered weights exactly as the replay reads them, and its
    index entries.  A row's outputs fill the member's output ports in order: one template's call sites may commit its output
    under different member names (GLM-5.3's routed experts' ``act``, its shared experts' ``out``), and the ports are named
    from its first row (`members`); each index entry keeps the row's own member names.  A row whose input or output widths
    are not the member's is refused (a packed template's in u16 words).  A ``cols`` member's row is ``W / w`` units, each
    operand and output its rows' columns [jw, jw + w) in order, under one index entry.  A part of the row (``PARTS``) reads the
    row's argument groups and committed outputs it names, in its order, and returns, and names in its entry, its outputs.  A
    ``cols`` row whose blocks are not its columns (`cols_refused`) is refused in blocks."""
    from verity_vllm.pipeline import part_rows as PR
    from verity_vllm.pipeline.serving_templates import cols_refused
    from verity_vllm.query.program_view import spec_base, spec_statics

    tp = template(m.template)
    layout = tp.layout
    if layout == "part" and tp.packed:
        return PR.read_part(m, rows, store, weights, arrivals, _slices, _packed)
    if layout == "gemm":
        return _read_gemm(m, rows, store, weights, arrivals)
    if layout == "coord":
        return PR.read_coord(m, rows, store, weights, arrivals, _slices, _packed)
    if layout in READERS:
        return READERS[layout][1](m, rows, store, weights, arrivals, _slices)
    cols: dict[str, list[np.ndarray]] = {p: [] for p, _ in m.rows + m.outs}
    D = dict(m.rows).get("x", 0)
    entries: list[list[Any]] = []
    lo = m.base
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        if tp.packed:
            ops = [_packed(_slices(pi, store, weights, rid, arrive, g, bits=b), b, f"row {r['i']} argument {k}")
                   for k, (g, b) in enumerate(zip(r["args"], tp.packed))]
        else:
            ops = [_operand(pi, store, weights, rid, arrive, g) for g in r["args"]]
        outs = []
        for j, (member, width, _dt) in enumerate(d["members"]):
            w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
            if w is None:
                raise LookupError(f"row {r['i']} output {member}: {why}")
            outs.append(_packed([w], tp.packed[len(r["args"]) + j], f"row {r['i']} output {member}", _dt) if tp.packed
                        else np.asarray(w, dtype=np.uint16))
        mine = list(range(len(outs)))
        if (p := part(tp.word, r["spec"])) is not None:
            ops, outs, mine = [(ops if src == "arg" else outs)[g] for src, g in p[1]], [outs[j] for j in p[2]], list(p[2])
        if layout == "rope":
            nh = int(d["statics"]["NHEADS"])
            cols["x"].append(np.asarray(ops[0], dtype=np.uint16).reshape(nh, D))
            cols["cs"].append(np.broadcast_to(np.asarray(ops[1], dtype=np.uint16).reshape(1, D), (nh, D)))
            cols["out"].append(outs[0].reshape(nh, D))
        elif layout == "cols":
            axis = COLS[spec_base(tp.word)]
            W, w = int(d["statics"][axis]), int(spec_statics(tp.word)[axis])
            if W % w:
                raise ValueError(f"row {r['i']} ({d['op_path']}): {axis}={W} is no whole number of member {m.key}'s {axis}={w} columns")
            nh = W // w
            if nh > 1 and (why := cols_refused(tp.word) or cols_refused(r["spec"])) is not None:
                raise ValueError(f"row {r['i']} ({d['op_path']}): {nh} column blocks of {why}")
            got = [len(a) for a in ops + outs]
            if [g // nh for g in got] != [n for _, n in m.rows + m.outs] or any(g % W for g in got):
                raise ValueError(f"row {r['i']} ({d['op_path']}): inputs and outputs of {got} words are not {nh} of member {m.key}'s "
                                 f"{list(m.rows)} and {list(m.outs)} along {axis}")
            for port, a in zip([p for p, _ in m.rows + m.outs], ops + outs, strict=True):
                a = np.asarray(a, dtype=np.uint16).reshape(-1, nh, w)
                cols[port].append(np.ascontiguousarray(a.transpose(1, 0, 2)).reshape(nh, -1))
        else:
            nh = 1
            got = ([len(a) for a in ops], [len(a) for a in outs] if tp.packed else [int(d["members"][j][1]) for j in mine])
            if got != ([w for _, w in m.rows], [w for _, w in m.outs]):
                raise ValueError(f"row {r['i']} ({d['op_path']}): inputs {got[0]} and outputs {got[1]} words, not member {m.key}'s "
                                 f"{list(m.rows)} and {list(m.outs)}")
            for g, a in enumerate(ops):
                cols[f"in{g}"].append(np.asarray(a, dtype=np.uint16).reshape(1, -1))
            for (port, _w), a in zip(m.outs, outs, strict=True):
                cols[port].append(a.reshape(1, -1))
        layer = _LAYER.search(d["op_path"])
        entries.append([lo, lo + nh, rid, arrive + int(d["step"]), int(layer.group(1)) if layer else None, d["op_path"],
                        ",".join(str(d["members"][j][0]) for j in mine), int(d["row"]), int(r["i"])])
        lo += nh
    if lo - m.base != m.n:
        raise ValueError(f"{m.key}: its rows hold {lo - m.base} units, not its {m.n} (a chunk is read with its template's others, `read_all`)")
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
            if tuple(at.tolist()) != m.link_refs:
                raise ValueError(f"{m.key}: its units read its token rows {at.tolist()}, its layout {list(m.link_refs)}")
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
        layout = template(t).layout
        if layout == "s8":
            out += _read_s8(mine, rs, store, weights, arrivals)
        elif mine[0].name:
            out += CR.split(mine, *read(replace(mine[0], n=sum(m.n for m in mine), name=""), rs, store, weights, arrivals))
        else:
            out.append(read(mine[0], rs, store, weights, arrivals))
        at += len(mine)
    return out


def _packed(parts: Sequence[np.ndarray], bits: int, what: str, dtype: str | None = None) -> np.ndarray:
    """A packed port's u16 words: its elements' little-endian bytes in order, each part's elements ``bits`` wide as read
    (`CommittedStore.words`' element dtypes, a registered weight's own).  A committed member of a wider signed integer ``dtype``
    (``BY_VALUE``: the sampler's i64 token under `TokenSelect`'s i32) is read by value, as the replay links it: each element
    in [0, 2^bits), refused otherwise."""
    if dtype in BY_VALUE:
        parts = [_by_value(np.asarray(a), bits, what) if np.asarray(a).dtype.itemsize * 8 > bits else a for a in parts]
    for a in parts:
        if np.asarray(a).dtype.itemsize * 8 != bits:
            raise ValueError(f"{what}: {np.asarray(a).dtype} elements, the template's port is {bits}-bit elements")
    raw = b"".join(np.ascontiguousarray(a, dtype=np.asarray(a).dtype.newbyteorder("<")).tobytes() for a in parts)
    if len(raw) % 2:
        raise ValueError(f"{what}: {len(raw)} bytes, not whole u16 words")
    return np.frombuffer(raw, dtype="<u2").astype(np.uint16)


def _by_value(a: np.ndarray, bits: int, what: str) -> np.ndarray:
    """``a``'s elements (a signed integer member's words as the store reads them, unsigned) as ``bits``-bit words."""
    u = a.reshape(-1).astype(np.uint64)
    if (u >> np.uint64(bits)).any():
        k = int(np.flatnonzero(u >> np.uint64(bits))[0])
        raise ValueError(f"{what}: element {k} of a {a.dtype} member is {int(u[k].astype(np.int64))}, not in a {bits}-bit port's "
                         f"[0, 2^{bits})")
    return u.astype(np.dtype(f"<u{bits // 8}"))


def _operand(pi: Any, store: Any, weights: Callable, rid: str, arrive: int, group: Sequence) -> np.ndarray:
    return np.concatenate(_slices(pi, store, weights, rid, arrive, group))


def _slices(pi: Any, store: Any, weights: Callable, rid: str, arrive: int, group: Sequence, *,
            bits: int | None = None) -> list[np.ndarray]:
    """Each run of one argument group, a stored ``"N"``/``"P"`` progression as its runs (`compact.runs_from_file_items`, the
    replay's `row_args`): a GLM-5.3 rank Program stores MoeSum's TOPK expert outputs as one node progression.  Given the port's
    element width ``bits``, a committed member of a wider signed integer dtype (``BY_VALUE``: the sampler's i64 token under the
    vocabulary shard's 32-bit ``tok``) is read by value, its low word with the high word zero or refused (`_by_value`)."""
    words = []
    for sl in runs_from_file_items(group):
        kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
        if kind == "n":
            m = pi.member_at(a, lo, hi)
            if m is None:
                raise LookupError(f"arg slice {list(sl)} is not inside one committed member")
            pd, member, mlo, mhi, dt = m
            width = next(int(mm[1]) for mm in pd["members"] if mm[0] == member)
            w, why = store.words(rid, arrive + pd["step"], pd["op_path"], member, pd["row"] * width + mlo, pd["row"] * width + mhi)
            if w is None:
                raise LookupError(why)
            w = np.asarray(w).reshape(-1)
            if bits is not None and dt in BY_VALUE and w.dtype.itemsize * 8 > bits:
                w = _by_value(w, bits, f"member {member} of {pd['op_path']}")
            words.append(w)
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
    return words


def after_commit(out_dir: str, *, partition_file: str, programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable,
                 weights: Callable, manifest: dict[str, Any] | None, run_root: bytes | str, run: str | None = None, row: Any = None,
                 arrivals: dict[str, int] | None = None, program_dirs: dict[str, str] | None = None,
                 digests: Mapping[str, str] | None = None, committer: dict[str, Any] | None = None,
                 workers: int | None = None, stand_in: Mapping[str, Any] | None = None, tp: Mapping[str, Any] | None = None,
                 index: Mapping[str, Any] | None = None, row_tables: Mapping[str, Any] | None = None,
                 reuse: Sequence[ProgramIndex] = (), binding: Mapping[str, Any] | None = None,
                 transfer_gates: Sequence[str] = ()) -> dict[str, Any]:
    """Commit the verifier's partition's units of a run whose roots are final and write the window's files into ``out_dir``:
    the registration (core's v1 record, its window the run root ``run_root`` and the provenance: the scheme the values were read
    under (``commitment``: the reader's committer's, rows-sha512 by default or vllm-v1 by name), the research run ``run``, the row, the
    committer and the served Programs' ``digests``), ``partition.json``, the index and M0's bodies.  The window is ``served``;
    with ``stand_in`` (its ``of`` and ``link``), a ``stand-in`` with them in its provenance.  A TP rank's window (`pipeline.tp.
    serving_rows`) gives its run as a string, its world and roots as ``tp``, its ProgramIndex arguments as ``index`` and the index it
    loaded its Programs with as ``reuse``.  ``transfer_gates`` are the templates the registered compiler lists as holding transfer
    gates, which the window's law proves whole (`population`).  A
    row-table layout takes the serving loop's ``row_tables``: ``{"context": <hex>, "key": <the run's row key>, "step_roots":
    {engine step: root}}`` (`commit.row_tables.step_roots`), which the window's step roots must be.  It reads the committed values
    and writes only ``out_dir``, ``circuit.json`` among its files: the window's circuit record (`circuit_record`), whose SHA-512
    the provenance carries as ``circuit``.  For Programs a registered compiler made from advice, the provenance also carries ``binding``: the
    compiler's digest and the advice's hiding root (`pipeline.advice_windows.binding`)."""
    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    pf = json.loads(Path(partition_file).read_text())
    rt = row_tables or {}
    window, rows = population(programs, pf, run_root if isinstance(run_root, str) else bytes(run_root).hex(), dirs, index, arrivals,
                              rt.get("context", ""), reuse, transfer_gates)
    ms = window.members
    store = CommittedStore.of_record(bmap, committed, manifest)
    got = read_all(ms, rows, store, weights, arrivals)
    t_read = time.perf_counter()
    c = CS.commit(window, [g[0] for g in got], row_salt_key=rt.get("key"), segment_roots=rt.get("step_roots"),
                  workers=workers or min(16, os.cpu_count() or 1))
    prov = {"commitment": "vllm-v1", "row": row, "research_run": run, "committer": dict(committer or {}, module=CS.__name__),
            "served_programs": {str(k): str(v) for k, v in sorted((digests or {}).items())}, "scope": pf.get("scope"),
            **({"tp": dict(tp)} if tp else {}), **(stand_in or {}), **(binding or {})}
    prov["commitment"] = getattr(getattr(committed, "com", None), "commit_scheme", prov["commitment"])
    c.seconds["read_s"] = round(t_read - t0, 3)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "partition.json").write_text(json.dumps(partition_doc(window, pf), indent=1) + "\n")
    body = IF.circuit_bytes(circuit_record(window))
    (Path(out_dir) / "circuit.json").write_bytes(body)
    prov["circuit"] = hashlib.sha512(body).hexdigest()
    res = CS.write(c, out_dir, prov, IX.index_doc(ms, [g[1] for g in got], window.run), kind="stand-in" if stand_in else "served",
                   registrar=R)
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (Path(out_dir) / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[serving-rows] {window.population} units in {len(ms)} members committed ({CS.SCHEME}): "
          + ", ".join(f"{m.key} {m.n}" for m in ms) + f"; registration {res['registration_digest'][:16]}; {res['seconds']}", flush=True)
    return res

