"""DSA's long rows by table reads in a served window (`serving_templates.HISTORY`, the ``history`` layout,
`commit.history_tables`): a long row's history reads, a ``DsaHistoryReads_v1{V, TOPK, H}`` row (``tok``: TOPK i32,
``hist``: V x H bf16 -> TOPK x H), cut into TOPK reads of `gates.embedding.EmbeddingRead_v1{V=2^17, H}`, read j opening the
request's history table at the row's layer at ``tok[j]``; and the ``DsaLatentRow_v1`` rows that write those tables. A window
whose partition file says ``"histories": true`` commits them (`on`):

* ``history``: a read, a unit a (row, j): ``in0`` the row's ``tok[j]`` (the key), ``in1`` and ``in2`` the table's root and
  domain id, ``in3``, ``in4`` and ``in5`` the table's row at the key, its salt and its path -> ``o0`` the row and ``o1`` its
  verdict. At a key past the table's length the row and salt are the placeholder's, the zero row under the zero salt, whose
  leaf the tree holds there;
* the writer: a ``DsaLatentRow_v1`` template (``WRITERS``) whose units are the window's latent rows, chunked on bit rows (the
  partition file's chunk caps name it), its output ``o0`` committed as ``hm96-sha512/row-seg/v1`` rows (`seg`), each under its
  table's salt and string at its position (`strings_of`), so its public string is the table's leaf.

A table is a (request, layer): a read's layer is its row's op_path's (an IndexShare layer reads its own layer's table at its
indexer layer's positions), a writer unit's its latent row's. A table's positions are its request's latent rows at its layer
in step and row order, appended from the table's edge (`append`): every latent row of the window is appended, read in this
window or not, since a later window's long rows read it. The window declares each table it appends to or reads as an
`interface.History` (`declarations`): its runs are the positions it holds from the request's earlier windows (a held run),
then this window's writer units. The verifier pins the table at its registered root and holds that root to the one it
computes (`verity.commitments.windows.history`). A read template's units go table by table, in the order the tables first
appear, then row by row, then j, so a table's declarations are three a run, and a member holds one table's units (`cut`).

Not wired yet: ``in0`` is element j of the row's ``tok``, its index's output, and a wire takes a writer's whole output row,
so a read's key is a free input until each of the index's positions is a row of its own; and ``o0`` is the reads' output,
which sparse MLA's chain still reads off its own rows.
"""
from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from verity.commitments import history as HT
from verity.commitments.windows import interface as IF
from verity.commitments.windows import rows as CS
from verity_vllm.commit import history_tables as H
from verity_vllm.pipeline import slice_rows as SL
from verity_vllm.pipeline.expert_rows import _digest_words, _entry
from verity_vllm.pipeline.serving_templates import HISTORY, template

Row = tuple[str, Any, dict, dict]
_LAYER = re.compile(r"\.layers\.(\d+)\.")
#: the history writers' row families: the latent row each table's leaves are
WRITERS = frozenset(HISTORY.values())
#: the writer's output port, the table's rows
OUT = "o0"


def _base(spec: str) -> str:
    from verity_vllm.query.program_view import spec_base

    return spec_base(spec)


def on(partition: Mapping[str, Any]) -> bool:
    """Whether a window of partition file ``partition`` commits history tables (``"histories": true``)."""
    return partition.get("histories") is True


def layer(d: Mapping[str, Any]) -> int:
    """A row's layer, its op_path's ``.layers.<l>.``; refused for a row of no layer."""
    got = _LAYER.search(str(d["op_path"]))
    if got is None:
        raise ValueError(f"{d['op_path']}: a history table is a layer's, and the row's op_path names none")
    return int(got.group(1))


def writer(t: str) -> bool:
    """Whether template ``t`` is a history writer (``WRITERS``)."""
    return _base(template(t).word) in WRITERS


def seg(tmpls: Sequence[str], histories: bool) -> frozenset[str]:
    """The templates whose ``o0`` a window of ``tmpls`` commits as ``hm96-sha512/row-seg/v1`` rows: with ``histories``, its
    history writers."""
    return frozenset(t for t in tmpls if histories and writer(t))


def blocks(statics: Mapping[str, Any]) -> int:
    """A history reads row's units: its TOPK reads."""
    return int(statics["TOPK"])


def cached(pi: Any) -> int:
    """The cached prompt rows request ProgramIndex ``pi``'s Program is served (its ``kv_prefix``'s C), 0 for none."""
    return int((getattr(pi, "kv_prefix", None) or {}).get("C") or 0)


def _uncached(t: str, rs: Sequence[Row]) -> None:
    for x in rs:
        if c := cached(x[1]):
            raise ValueError(f"{t}: request {x[0]} is served {c} cached prompt rows; its tables hold only rows it computed, "
                             f"leaf i its position {c} + i, and its reads open the composed table its replay builds, which a "
                             "window of history tables doesn't defer to yet: serve it without a prefix-cache hit")


class Plan:
    """The history reads and writers over a window's templates ``tmpls`` and their ``rows`` (`serving_rows.population`'s), with
    ``histories`` on: each read template's units in the module's order (``units``: each ``(row, j)``), the tables they read
    (``tables``: each ``(first unit, units, (request, layer))``), and each writer template's units by table in position order
    (``written``: (request, layer) -> [(template, unit)])."""

    def __init__(self, tmpls: Sequence[str], rows: Sequence[list[Row]], histories: bool) -> None:
        self.of = {t: rs for t, rs in zip(tmpls, rows, strict=True) if template(t).layout == "history"}
        if self.of and not histories:
            raise ValueError(f"history reads {sorted(self.of)} in a window that commits no history tables: its partition file "
                             "says \"histories\": true")
        self.units: dict[str, list[tuple[int, int]]] = {}
        self.tables: dict[str, list[tuple[int, int, tuple[str, int]]]] = {}
        order: dict[tuple[str, int], int] = {}
        for t, rs in self.of.items():
            _uncached(t, rs)
            fams = {_base(x[2]["spec"]) for x in rs}
            if len(fams) != 1 or not fams <= set(HISTORY):
                raise ValueError(f"{t}: rows of {sorted(fams)}; a history read reads rows of one family of {sorted(HISTORY)}")
            keys = []
            for p, x in enumerate(rs):
                k = (x[0], layer(x[3]))
                keys += [(order.setdefault(k, len(order)), p, j) for j in range(blocks(x[3]["statics"]))]
            keys.sort()
            self.units[t] = [(p, j) for _k, p, j in keys]
            runs: list[tuple[int, int, tuple[str, int]]] = []
            for u, (p, _j) in enumerate(self.units[t]):
                k = (rs[p][0], layer(rs[p][3]))
                if not runs or runs[-1][2] != k:
                    runs.append((u, 0, k))
                runs[-1] = (runs[-1][0], runs[-1][1] + 1, k)
            self.tables[t] = runs
        self.writers = [t for t in tmpls if histories and writer(t)]
        if len(self.writers) > 1:
            raise ValueError(f"history writers {self.writers}: one template's units are a window's latent rows")
        self.written: dict[tuple[str, int], list[tuple[str, int]]] = {}
        for t in self.writers:
            rs = rows[list(tmpls).index(t)]
            seen: dict[tuple[str, int, int, int], int] = {}
            for p in sorted(range(len(rs)), key=lambda p: (layer(rs[p][3]), int(rs[p][3]["step"]), int(rs[p][3]["row"]))):
                rid, d = rs[p][0], rs[p][3]
                if (other := seen.setdefault((rid, layer(d), int(d["step"]), int(d["row"])), p)) != p:
                    raise ValueError(f"{t}: rows {rs[other][2]['i']} and {rs[p][2]['i']} of request {rid} are both its latent row "
                                     f"at layer {layer(d)}, step {d['step']}, row {d['row']}")
                self.written.setdefault((rid, layer(d)), []).append((t, p))

    def read_keys(self) -> list[tuple[str, int]]:
        """The tables the window's reads read, in the order they first appear."""
        return list(dict.fromkeys(k for runs in self.tables.values() for _a, _n, k in runs))

    def keys(self) -> list[tuple[str, int]]:
        """Every table the window appends to or reads: its writers' first, then the reads' others."""
        return list(dict.fromkeys([*self.written, *self.read_keys()]))


def cut(ms: Sequence[CS.Member], plan: Plan) -> list[CS.Member]:
    """``ms`` (`serving_rows.members`') with each read template cut again where its units' table changes and at its chunk cap
    (its first chunk's size), named ``<descriptor id>/c<j>`` in order."""
    out: list[CS.Member] = []
    for t in dict.fromkeys(m.template for m in ms):
        mine = [m for m in ms if m.template == t]
        if t not in plan.units:
            out += mine
            continue
        first, pieces = mine[0], []
        for at, n, _k in plan.tables[t]:
            pieces += [(a, min(a + first.n, at + n)) for a in range(at, at + n, first.n)]
        out += [replace(first, base=first.base + a, n=b - a, name=f"{t}/c{j}") for j, (a, b) in enumerate(pieces)]
    return out


def _bases(ms: Sequence[CS.Member]) -> dict[str, int]:
    base: dict[str, int] = {}
    for m in ms:
        base[m.template] = min(base.get(m.template, m.base), m.base)
    return base


def _chunked(m: CS.Member) -> None:
    if [p for p, _w in m.outs] != [OUT]:
        raise ValueError(f"history writer {m.template}: its output is {[p for p, _w in m.outs]}, not one row {OUT}; a writer "
                         "is chunked (its partition file's chunk_cap names it)")


def append(plan: Plan, ms: Sequence[CS.Member], got: Sequence[tuple[dict[str, Any], list]],
           histories: H.Histories) -> dict[tuple[str, int], range]:
    """Every writer unit's ``o0`` row (``got``: `serving_rows.read_all`'s words of ``ms``) appended to its table in
    ``histories``, table by table in position order, each from its table's edge. -> each table's positions in the window."""
    rows: dict[str, np.ndarray] = {}
    base = _bases(ms)
    for m, (g, _e) in zip(ms, got, strict=True):
        if m.template in plan.writers:
            _chunked(m)
            a = rows.setdefault(m.template, np.zeros((sum(x.n for x in ms if x.template == m.template), histories.words),
                                                     dtype=np.uint16))
            a[m.base - base[m.template]:m.base - base[m.template] + m.n] = g["outs"][OUT]
    return {k: histories.append(k[0], k[1], histories.edge(*k), np.stack([rows[t][u] for t, u in us]))
            for k, us in plan.written.items()}


def strings_of(plan: Plan, ms: Sequence[CS.Member], histories: H.Histories,
               at: Mapping[tuple[str, int], range]) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Each writer member's ``o0`` salts and strings, its units' rows' at their positions in their tables (``at``, `append`'s),
    by port: what `verity.commitments.windows.rows.commit` takes as ``salts_of`` and ``strings_of``."""
    where: dict[tuple[str, int], tuple[tuple[str, int], int]] = {}
    for k, us in plan.written.items():
        for (t, u), pos in zip(us, at[k], strict=True):
            where[(t, u)] = (k, pos)
    base, salts, strings = _bases(ms), {}, {}
    for m in ms:
        if m.template not in plan.writers:
            continue
        _chunked(m)
        mine = [where[(m.template, u)] for u in range(m.base - base[m.template], m.base - base[m.template] + m.n)]
        salts[m.port(OUT)] = b"".join(histories.salts(k[0], k[1], [p]) for k, p in mine)
        strings[m.port(OUT)] = b"".join(histories.strings(k[0], k[1], [p]) for k, p in mine)
    return salts, strings


def _runs(units: Sequence[int], start: int, t: str) -> list[dict[str, Any]]:
    """Writer ``t``'s ``units`` (its template's) at positions ``start`` on, as runs of a constant stride."""
    out: list[dict[str, Any]] = []
    for i, u in enumerate(units):
        if out:
            r = out[-1]
            stride = u - r["base"] if r["count"] == 1 else r["step"][0]
            if stride > 0 and u == r["base"] + r["count"] * stride:
                r["step"], r["count"], r["radix"] = [stride], r["count"] + 1, [r["count"] + 1]
                continue
        out.append({"start": start + i, "count": 1, "writer": t, "out": OUT, "radix": [1], "base": int(u), "step": [1]})
    return out


def declarations(plan: Plan, histories: H.Histories, edges: Mapping[tuple[str, int], int]) -> IF.Registered | None:
    """The window's history tables, table reads and openings: each table it appends to or reads (`Plan.keys`) as an
    `interface.History` of its writer's ``o0`` rows, its runs the ``edges`` positions it held before the window (a held run of
    its own rows) and then the window's writer units; each read run's root and domain id read into ``in1`` and ``in2`` and
    its row, salt and path (``in3``, ``in4``, ``in5``) opened at the key ``in0`` by the verdict ``o1``. None where there are
    none."""
    keys = plan.keys()
    if not keys:
        return None
    if not plan.writers:
        raise ValueError(f"history reads of {H.name(*keys[0])} in a window with no latent rows: a window holds its history "
                         "tables' writer")
    w = plan.writers[0]
    hs, roots, openings = [], [], []
    for k in keys:
        nm, edge = H.name(*k), int(edges.get(k, 0))
        if not edge and k not in plan.written:
            raise ValueError(f"history reads of {nm}, which holds no rows: a request's latent row at a layer is appended "
                             "before its long rows read the table")
        runs = ([{"start": 0, "count": edge, "held": nm, "row": 0}] if edge else []) + _runs(
            [u for _t, u in plan.written.get(k, [])], edge, w)
        hs.append(IF.History(nm, histories.program, w, OUT, histories.words, runs, histories.capacity))
    for t, runs in plan.tables.items():
        for at, n, k in runs:
            nm = H.name(*k)
            roots += [IF.TableRead(t, "in1", nm, "root", at, n), IF.TableRead(t, "in2", nm, "domain", at, n)]
            openings.append(IF.TableOpening(t, nm, "in0", ("in3", "in4", "in5"), "o1", at, n))
    return IF.Registered({}, (), (), tuple(roots), tuple(openings), histories=tuple(hs))


class Reader:
    """The words and index entries of the history reads' members (`read`), each row's tok read once; ``slices`` is
    `serving_rows._slices`, ``histories`` the committer's tables, appended with the window's rows."""

    def __init__(self, plan: Plan, store: Any, weights: Callable, arrivals: Mapping[str, int] | None, slices: Callable,
                 histories: H.Histories | None) -> None:
        self.plan, self.store, self.weights, self.arrivals, self.slices = plan, store, weights, arrivals or {}, slices
        self.histories = histories
        self.toks: dict[tuple[str, int], np.ndarray] = {}

    def tok(self, x: Row) -> np.ndarray:
        """Row ``x``'s TOPK keys, its argument 0's i32 elements, each a position below the table's capacity."""
        rid, pi, r, d = x
        if (rid, int(r["i"])) not in self.toks:
            got = SL.elements(self.slices(pi, self.store, self.weights, rid, int(self.arrivals.get(rid, 0)), r["args"][0], bits=32),
                              32, f"row {r['i']} argument 0").view("<i4").astype(np.int64)
            n = blocks(d["statics"])
            if got.size != n or (got < 0).any() or (got >= HT.CAPACITY).any():
                raise ValueError(f"row {r['i']} ({d['op_path']}): its tok is {got.size} positions, not its TOPK = {n} in "
                                 f"[0, {HT.CAPACITY})")
            self.toks = {(rid, int(r["i"])): got}
        return self.toks[(rid, int(r["i"]))]

    def read(self, ms: Sequence[CS.Member]) -> list[tuple[dict[str, Any], list[list[Any]]]]:
        """Members ``ms``, every member of one read template, each its words (every port its elements' bytes in u16 words) and
        index entries, one a unit: each opening of its table among ``histories`` at its key, the placeholder's row and salt past
        the table's length; refused for a read whose table is missing."""
        t, plan = ms[0].template, self.plan
        rs, names, n, b0 = plan.of[t], [q for q, _ in ms[0].rows + ms[0].outs], sum(m.n for m in ms), ms[0].base
        out = {q: np.empty((n, w), dtype=np.uint16) for q, w in ms[0].rows + ms[0].outs}
        entries: list[list[Any]] = []
        starts = {at: k for at, _n, k in plan.tables[t]}
        table = None
        for u, (p, j) in enumerate(plan.units[t]):
            if u in starts:
                table = (self.histories.tables if self.histories is not None else {}).get(starts[u])
                if table is None:
                    raise ValueError(f"{t}: its table {H.name(*starts[u])} is not among the committer's history tables")
                root, dom = _digest_words(table.root), _digest_words(table.domain.domain_id)
            x = rs[p]
            e = int(self.tok(x)[j])
            held = e < len(table.rows)
            row = table.rows[e] if held else np.zeros(table.words, dtype=np.uint16)
            salt = table.salts[e] if held else bytes(CS.SALT_BYTES)
            for q, a in zip(names, [np.array([e], dtype="<u4").view("<u2"), root, dom, row, _digest_words(salt),
                                    _digest_words(table.path(e)), row, np.ones(1, dtype=np.uint16)], strict=True):
                a = np.asarray(a, dtype=np.uint16).reshape(-1)
                if a.size != out[q].shape[1]:
                    raise ValueError(f"{t} unit {u}: its {q} is {a.size} words, the member's {out[q].shape[1]}")
                out[q][u] = a
            entries.append(_entry(b0 + u, b0 + u + 1, x, int(self.arrivals.get(x[0], 0)),
                                  ",".join(str(mm[0]) for mm in x[3].get("members") or ())))
        ends = [e[1] for e in entries]
        res = []
        for m in ms:
            a, k, mine = m.base - b0, bisect.bisect_right(ends, m.base), []
            while k < len(entries) and entries[k][0] < m.base + m.n:
                mine.append(entries[k])
                k += 1
            res.append(({part: {q: out[q][a:a + m.n] for q, _w in ports} for part, ports in (("rows", m.rows), ("outs", m.outs))},
                        mine))
        return res
