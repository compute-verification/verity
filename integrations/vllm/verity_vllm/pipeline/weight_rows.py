"""A served window's registered reads of its weights (`interface.listed_ingress`'s ``registered``), which
`serving_rows.circuit_record` declares beside its listed members: each input port of a slice unit whose words are one whole
row of one weight of the Program's ``weights`` struct (`Gemm_v2{N=1}`'s ``w[n]``), as a read of that weight's registered
parameter at the port's row width (``weights/<label>@<words>``, `check.weights_of_record.weight_param`), its leaves the
weight's u16 words.  A read is declared only where the port's rows are the registered value's rows' schema (``row/v2``
bit rows), so the committer, given its weights, commits it by reference to the weight's committed rows (`by_reference`); a
row template's ``row/v1`` rows, a coordinate's ``w[:, n, :]`` and an s8 unit's ``W[c, :] ‖ sw[c]`` are no such row, and stay
free.  The window's members are cut where the reads begin and end (`cut`), so each read covers whole members.  The reads
come from the Programs' rows alone, public structure, never a value, so the registered compiler's record of the advice
(``prove_path.py circuit``) states the same reads as the prover's."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity.ml.windows import interface as IF
from verity.ml.windows import rows as CS
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline.serving_templates import SLICES, _bind, template
from verity_vllm.pipeline.slice_rows import dims
from verity_vllm.program.compact import runs_from_file_items
from verity_vllm.query.program_view import spec_base

Row = tuple[str, Any, dict, dict]


def _weight(pi: Any, group: Sequence) -> tuple[str, int, int, int, int] | None:
    """The one weight argument group ``group`` reads a run of: its label, the run's first element and elements, the weight's
    elements and their bits; None where it reads anything else."""
    runs = list(runs_from_file_items(group))
    if len(runs) != 1 or runs[0][0] != "p" or (pi.pnames[int(runs[0][1])] if int(runs[0][1]) < len(pi.pnames) else None) != WR.PARAM:
        return None
    got = pi.weight_slice(int(runs[0][2]), int(runs[0][3]))
    if got is None:
        return None
    label, lo, hi, _shape = got
    count = next(c for n, c, _s, _b in pi.fields if n == label)
    return label, lo, hi - lo, count, pi.weight_bits(label)


def _words(elements: int, bits: int) -> int | None:
    return elements * bits // 16 if elements * bits % 16 == 0 else None


def _row_reads(t: str, pi: Any, r: Mapping[str, Any], d: Mapping[str, Any], at: int) -> list[tuple[IF.Read, int | None]]:
    """Slice template ``t``'s units of row ``r`` (from unit ``at``) each reading one row of one weight on a port, each read with
    the weight's words (None where its elements aren't whole words): an argument whose whole weight the row reads and whose
    dimensions carry the template's axes first, in order, so unit i's element is the weight's row i."""
    axes, carry = SLICES[spec_base(template(t).word)]
    ext, out = tuple(int(d["statics"][a]) for a in axes), []
    for k, (g, (_nm, ty), carried) in enumerate(zip(r["args"], _bind(r["spec"]).signature()[0], carry, strict=True)):
        got, shape = _weight(pi, g), dims(ty)
        on = [j for j in carried if j is not None]
        if got is None or not on or on != list(range(len(on))) or got[2] != math.prod(shape):
            continue
        label, first, _n, count, bits = got
        row, base = _words(math.prod(shape[len(on):]), bits), _words(first, bits)
        if row is None or base is None:
            continue
        step = tuple(0 if j is None else math.prod(shape[j + 1:]) * bits // 16 for j in carried)
        out.append((IF.Read(t, f"in{k}", WR.weight_param(label, row), at=at, radix=ext, base=base, step=step), _words(count, bits)))
    return out


def reads(members: Sequence[CS.Member], rows: Sequence[list[Row]],
          units: Callable[[str, Mapping[str, Any]], int]) -> IF.Registered | None:
    """The registered reads of the weights by the window's ``members`` over their templates' ``rows`` (`serving_rows.population`'s),
    ``units`` how many of a template's units a row holds (`serving_rows.row_units`): each port of a slice unit that reads one
    whole row of one weight as ``row/v2`` rows of its width, in runs of like reads over consecutive units (`interface.coalesce`);
    a read only where its rows start on a whole row of the weight and the weight is whole rows of them.  None where the window
    reads no weight."""
    params: dict[str, int] = {}
    out: list[IF.Read] = []
    for t, rs in zip(dict.fromkeys(m.template for m in members), rows, strict=True):
        if template(t).layout != "slice":
            continue
        m, at = next(m for m in members if m.template == t), 0
        words = dict(m.rows)
        for _rid, pi, r, d in rs:
            for read, total in _row_reads(t, pi, r, d, at):
                w = words[read.port]
                if (m.schema(read.port) != V2 or int(read.param.rpartition("@")[2]) != w or total is None or total % w
                        or read.base % w or params.setdefault(read.param, total) != total):
                    continue
                out.append(read)
            at += units(t, d)
    got = IF.coalesce(out)
    return IF.Registered({p: params[p] for p in sorted({x.param for x in got})}, reads=tuple(got)) if got else None


def by_reference(window: CS.Binding, registered: IF.Registered | None, data: Sequence[dict[str, Any]],
                 values: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    """``data`` (`serving_rows.read_all`'s, one a member) with each port a read of ``registered`` fills given by reference
    (`rows.ValueRows`) as the rows its units read of the parameter's committed value in ``values`` (``{param: core's
    registered.Registered}``, the committer's weights under its key at their nonces, rows held: `window_weights.weight_value`),
    so each unit's row is the value's row the Lean verifier opens against the pinned root (`interface.registered_reads`);
    with each such member's header entries (`rows.write`'s ``registered``) and each value's root (``registered_roots``).
    Refused (ValueError) where a read's value is missing or isn't the words its units read.  ``data`` as given with no
    ``values``, or in a window with row tables (whose rows `rows.commit` takes in full): its reads stay free."""
    if not values or registered is None or window.row_tables:
        return list(data), {}, {}
    base: dict[str, int] = {}
    for m in window.members:
        base.setdefault(m.template, m.base)
    out, entries, roots = [], {}, {}
    for m, d in zip(window.members, data, strict=True):
        lo, d = m.base - base[m.template], {**d, "rows": dict(d["rows"])}
        for r in (r for r in registered.reads if r.reader == m.template and r.at <= lo < r.at + r.reader_units):
            v, w = values.get(r.param), dict(m.rows)[r.port]
            if v is None:
                raise ValueError(f"{m.key} {r.port} reads {r.param!r}: give its committed value, or no values at all")
            at = r.source(np.arange(lo - r.at, lo - r.at + m.n)) // w
            if v.words != w or not np.array_equal(CS.value_rows(v, at), np.asarray(d["rows"][r.port], dtype=np.uint16)):
                raise ValueError(f"{m.key} {r.port}: the rows its units read aren't {r.param!r}'s committed rows {at[:4].tolist()}...")
            d["rows"][r.port] = CS.ValueRows(v, tuple(int(x) for x in at))
            entries.setdefault(m.key, {})[r.port] = {"value": v.value, "positions": [int(x) for x in at],
                                                     "paths": [v.path(int(x)).hex() for x in at]}
            roots[r.param] = v.root.hex()
        out.append(d)
    return out, entries, roots


def cut(members: Sequence[CS.Member], registered: IF.Registered | None) -> list[CS.Member]:
    """``members`` cut where ``registered``'s reads begin and end, so each read covers whole members (the committer commits a
    read port by reference to one registered value); a template cut into several members names them ``<descriptor id>/c<j>``
    in order, as a chunked one's.  Refused (ValueError) for a member a wire, a shared-row table or a row table fills, which no
    slice unit's read reaches."""
    base: dict[str, int] = {}
    for m in members:
        base.setdefault(m.template, m.base)
    ends: dict[str, set[int]] = {}
    for r in (registered.reads if registered is not None else ()):
        ends.setdefault(r.reader, set()).update((base[r.reader] + r.at, base[r.reader] + r.at + r.reader_units))
    out: list[CS.Member] = []
    for t in base:
        mine, pieces = [m for m in members if m.template == t], []
        for m in mine:
            at = [m.base, *sorted(e for e in ends.get(t, ()) if m.base < e < m.base + m.n), m.base + m.n]
            if len(at) > 2 and (m.wired or m.tables or m.links):
                raise ValueError(f"{m.key}: a weight read covers part of a member a wire, a shared-row table or a row table fills")
            pieces += [replace(m, base=a, n=b - a) for a, b in zip(at, at[1:])]
        out += pieces if len(pieces) == len(mine) else [replace(p, name=f"{t}/c{j}") for j, p in enumerate(pieces)]
    return out
