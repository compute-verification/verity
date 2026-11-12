"""A two-stage sampled audit's interior registration, from replay.

The boundary registration's replay blocks are request spans: one request's lifetime in a served window, padded to its size
class's n_rho proof blocks. After the first stage samples Q of them, the prover replays each sampled span and registers its
proof blocks' values, phase ``interior``, as the record `Stream.register_interiors` receives: population Q·n_rho, the
sampled span ``sample[i]``'s units at [i·n_rho, (i+1)·n_rho), chained after the boundary registration.

* `span`: one sampled span's units of the stream's template, read from its `Replay` (`check.replay.request_replay`: one
  request's values replayed on the CPU, a request served from the prefix cache with its hit recomputed) exactly as the
  serving rows read the committed store (`serving_rows.population`, `serving_rows.read_all`).
* `port_map`: which committed element or weight each of those units' port words is, read off the same `read_all` over a
  store that answers each element with its own place, so a replay on the GPU gathers them from the recorded tensors;
  `gather` is that gather on the host.
* `captured`: a span of a token-select template from what a replay at the served step shapes took at the sampler
  (`engine.sampler_capture`: each sampled token's logits row and id), laid out as `span` lays a token select's units out.
  A replay on the GPU writes it for a served model the CPU `Replay` is too slow for; tests write it from a `Replay`.
* `pieces`: a token select's span as the units Flock proves it by, since a served vocabulary's whole select is past one
  Flock block: `TokenSelect_v4`'s, each token's `TokenSelectChunk_v2` units over its chunks of logits and its
  `TokenSelectCombine_v2`; `TokenSelectUnits_v1`'s, each token's K `TokenSelectGroups_v2` units and its
  `TokenSelectCombineUnits_v1`, whose every input row is one unit's output row.
* `interior`: the replay sample's spans, template t's each padded to its n_t with copies of its last instance's units
  (`Span.group`: a token's chunks), so an inlined replay block's padded instances stay whole, committed template-major
  as one window (`ml.windows.rows`, every leaf's salt fresh from the OS) of one shared-row member a template, each input
  port a table of its own Q·n_t rows and unit off_t + k reading row k, so each row a session proves is a registered read
  of the interior record. A port the replay block wires to another template's output (its wires, as round 2 derives
  them: `second_stage.replay_wires`) commits no tree of its own: its rows are its writer's Q·n_w output rows, whole, unit
  u of span s reading its writer's unit off_w + s·n_w + j, j the row the design places it at (`second_stage.replay_rows`;
  the writer's first in the span where it names none). It is registered after ``previous``, the boundary registration's digest,
  under the window ``{kind, run, row, interior: true[, replay]}``. Its run is `interior_run` of that digest and the
  replay sample, so every port's domain binds which replay block each cell of units is; the challenger derives the same
  ports from what it holds (`interior_ports`). The spans must share one Program, as a size class's spans share its
  replay Definition. A commitment made on the replayed rows' device (`DeviceCommit`: a GPU commit's roots, every row's
  b || c and its salts) registers in place of committing the rows on the host (`committed`).
* `interior_links`: the replay-block parameters the interior registers as values, the verifier's (`Supply`) and the
  replay's own (`Operand`), the challenger's copies of them and the history tables its reads open, and the block's
  returns (`Return`), read at round 1's rows where its record commits them; `interior` takes them, and this module's
  `__all__` lists the names its callers take.
* `register`: the record on the two-stage stream after its first stage.

A `span` reads one template's units, one member of word rows; a chunked or shared-row template's span is refused by name.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from verity.protocols.spot.verifier.one_stage import partition as P, registration as R
from verity.circuits.defs import bind
from verity.commitments import history as HT
from verity.definitions.numerics.rows import SAMPLED_MEMBER
from verity.definitions.numerics.token_select_rows import token_select_chunk_bests
from verity.definitions.registry.dead_free import CHUNK, TokenSelectCombineUnitsV1, TokenSelectGroupsV2, TokenSelectUnitsV1
from verity.commitments.windows import links as LK, rows as CS
from verity_vllm.check.replay.index import SAMPLER_OP, ProgramIndex
from verity_vllm.check.replay.request_replay import Replay
from verity_vllm.engine.sampler_capture import CAPTURE
from verity_vllm.pipeline import chunk_rows as CR, interior_links as IL, serving_rows as SRW
from verity_vllm.pipeline.interior_links import (DIGEST, HELD, INDEX, READ, TABLE, Cells, Link, Members, Operand, Return,
                                                 Supply, history, kept_table, link_schemas, link_values, link_words,
                                                 links, returns, sources, span_values, supplied, table_reads)
from verity_vllm.records.logits_capture import LogitsCapture

#: the tag `interior_run` hashes under
RUN_TAG = b"verity-vllm/interior-run/v1\x00"


@dataclass(frozen=True)
class Span:
    """One request span's units of one template: its member (``n`` units), the units' rows and outputs as
    `ml.windows.rows.commit` takes them, `serving_rows.read`'s index entries, which say which row each unit is, and the
    digest of the Program it was replayed from. ``group``: its units an instance of the replay block's check template (a
    token's chunks, `pieces`)."""
    rid: str
    member: CS.Member
    data: dict[str, dict[str, np.ndarray]]
    entries: list[list[Any]]
    program: str
    seconds: dict[str, float] = field(default_factory=dict)
    group: int = 1

    @property
    def n(self) -> int:
        return self.member.n

    def padded(self, n_rho: int) -> dict[str, dict[str, np.ndarray]]:
        """Its data at ``n_rho`` units, its last instance's ``group`` units repeated after its own, so each padded
        instance of the replay block is a whole copy of its last."""
        g = self.group
        if not 1 <= self.n <= n_rho or self.n % g or n_rho % g:
            raise ValueError(f"request {self.rid}'s span holds {self.n} units of {self.member.template}, not 1 to its size "
                             f"class's {n_rho}" + (f" in whole instances of {g} units" if g > 1 else ""))
        reps = (n_rho - self.n) // g
        return {kind: {p: np.concatenate([a, np.tile(a[-g:], (reps,) + (1,) * (a.ndim - 1))])
                       for p, a in self.data[kind].items()}
                for kind in ("rows", "outs")}


def span(inst: dict[str, Any], replay: Replay, template: str, *, program_dir: str | None = None,
         index: Mapping[str, Any] | None = None) -> Span:
    """Request ``replay.rid``'s units of ``template`` in its Program ``inst``, read from ``replay`` as `serving_rows`
    reads a committed store. ``program_dir`` and ``index``: the ProgramIndex's, as `serving_rows.member_rows` takes
    them; ``replay.pi`` is reused when built with them."""
    rid, t0 = replay.rid, time.perf_counter()
    arrivals = {rid: replay.arrive}
    window, rows = SRW.population({rid: inst}, {"templates": [template], "law": "subset:1"}, "00" * 64,
                                  program_dirs={rid: program_dir} if program_dir else None, index=index, arrivals=arrivals,
                                  reuse=(replay.pi,))
    m = window.members
    if len(m) != 1 or m[0].name or m[0].tables or m[0].links or m[0].wired:
        raise ValueError(f"{template}: request {rid}'s span is {[x.key for x in m]}, not one member of word rows (a chunked or "
                         "shared-row template's interior is not built)")
    t1 = time.perf_counter()
    (data, entries), = SRW.read_all(m, rows, replay, replay.weights, arrivals)
    return Span(rid, m[0], data, entries, str(inst["digest"]),
                {"index_s": round(t1 - t0, 3), "read_s": round(time.perf_counter() - t1, 3)})


#: the layouts whose words `serving_rows.read` only moves (slices, reorders, repeats), so a trace of its reads is their map
MOVED = ("row", "part", "rope", "cols")


class _Trace:
    """A store and registered weights whose every element read is digit ``d`` (16 bits) of its place among all the elements
    read, in read order: `serving_rows.read_all` over it says which element each word it lays out is."""

    def __init__(self, d: int) -> None:
        self.d, self.calls, self.n = d, [], 0

    def _next(self, source: tuple[Any, ...], lo: int, hi: int) -> np.ndarray:
        self.calls.append((self.n, source, lo))
        at = np.arange(self.n, self.n + hi - lo, dtype=np.uint64)
        self.n += hi - lo
        return ((at >> np.uint64(16 * self.d)) & np.uint64(0xFFFF)).astype(np.uint16)

    def words(self, rid: str, step: int, op_path: str, member: str, elo: int, ehi: int) -> tuple[np.ndarray, str]:
        return self._next(("store", str(rid), int(step), str(op_path), str(member)), int(elo), int(ehi)), "u16"

    def weights(self, field: str, lo: int, hi: int) -> np.ndarray:
        return self._next(("weights", str(field)), int(lo), int(hi))


def port_map(inst: dict[str, Any], template: str, rid: str, *, arrive: int = 0, program_dir: str | None = None,
             index: Mapping[str, Any] | None = None, reuse: Sequence[ProgramIndex] = ()) -> dict[str, Any]:
    """Which committed element or registered weight each port word of request ``rid``'s units of ``template`` is, in its
    Program ``inst``, read off `serving_rows.read_all` itself: fixed by the Program, before serving or a replay. ``sources``:
    each ``["store", rid, engine step, op_path, member]`` as `CommittedStore.words` reads it, or ``["weights", field]`` as
    the registered weights' reader; ``members``: each member's template, units and ports, a port its words a unit and its
    runs ``[unit, word, source, lo, hi]`` (that unit's words from ``word`` on are the source's elements [lo, hi), each its
    low 16 bits as `read` takes it). `gather` applies it. Refused for a template whose read computes its words."""
    tp = SRW.template(template)
    if tp.packed or tp.layout not in MOVED:
        raise ValueError(f"{template}: a {'packed ' if tp.packed else ''}{tp.layout} template's words are computed by its "
                         f"read, not moved; a port map is read off the layouts {MOVED}")
    arrivals = {rid: int(arrive)}
    window, rows = SRW.population({rid: inst}, {"templates": [template], "law": "subset:1"}, "00" * 64,
                                  program_dirs={rid: program_dir} if program_dir else None, index=index, arrivals=arrivals,
                                  reuse=reuse)
    traces = [_Trace(d) for d in range(3)]
    reads = [SRW.read_all(window.members, rows, t, t.weights, arrivals) for t in traces]
    calls = traces[0].calls
    place = {s: i for i, s in enumerate(dict.fromkeys(s for _b, s, _lo in calls))}
    sources, which = list(place), np.array([place[s] for _b, s, _lo in calls], dtype=np.int64)
    bases = np.array([b for b, _s, _lo in calls], dtype=np.uint64)
    los = np.array([lo for _b, _s, lo in calls], dtype=np.int64)
    members = []
    for k, m in enumerate(window.members):
        ports = {}
        for kind, ps in (("rows", m.rows), ("outs", m.outs)):
            for p, _w in ps:
                at = sum(np.asarray(r[k][0][kind][p], dtype=np.uint64) << np.uint64(16 * d) for d, r in enumerate(reads))
                n, w = at.shape
                flat = at.reshape(-1)
                c = np.searchsorted(bases, flat, side="right") - 1
                off = (flat - bases[c]).astype(np.int64) + los[c]
                j = np.arange(flat.size)
                cut = np.flatnonzero((j % w == 0) | np.r_[True, (c[1:] != c[:-1]) | (off[1:] != off[:-1] + 1)])
                ends = np.r_[cut[1:], flat.size]
                ports[p] = {"words": w, "runs": [[int(a // w), int(a % w), int(which[c[a]]), int(off[a]), int(off[a]) + int(e - a)]
                                                 for a, e in zip(cut, ends, strict=True)]}
        members.append({"template": m.template, "units": m.n, "ports": ports})
    return {"rid": rid, "sources": [list(s) for s in sources], "members": members}


def gather(pm: Mapping[str, Any], store: Any, weights: Callable) -> list[dict[str, np.ndarray]]:
    """Each member's port words of `port_map` ``pm``, read from ``store`` (`CommittedStore.words`) and ``weights``: what a
    GPU gather of the same runs must lay out."""
    out = []
    for m in pm["members"]:
        got = {}
        for p, d in m["ports"].items():
            a = np.zeros((m["units"], d["words"]), dtype=np.uint16)
            for unit, word, s, lo, hi in d["runs"]:
                src = pm["sources"][s]
                v = store.words(*src[1:], lo, hi)[0] if src[0] == "store" else weights(src[1], lo, hi)
                a[unit, word:word + hi - lo] = np.asarray(v).reshape(-1).astype(np.uint16)
            got[p] = a
        out.append(got)
    return out


def captured(directory: Path, rid: str, served: Sequence[int] | None = None) -> Span:
    """Request ``rid``'s span of the capture's token-select template (`sampler_capture.write`): unit j reads logits row j and
    outputs token j, laid out as `span` reads a token select's units from a Program. ``served``: the ids serving emitted,
    which the captured tokens must be, or the replay didn't reproduce the span and nothing is committed: the refusal
    names the request and how many ids differ, never an id, which is a served value."""
    cap = LogitsCapture.of(json.loads((directory / CAPTURE).read_text()), str(directory / CAPTURE))
    s, t = cap.spans[rid], cap.template
    params, ret = SRW.template(t).fn.signature()
    ins, out = [sum(ty.leaf_widths()) // 16 for _n, ty in params], sum(ret.leaf_widths()) // 16
    if len(ins) != 1 or out != 2 or tuple(s["rows"]) != (len(s["tokens"]), ins[0]):
        raise ValueError(f"{t}: inputs of {ins} words and an output of {out}, captured rows {s['rows']}: a token select reads "
                         "one logits row and outputs one 32-bit id")
    raw = (directory / s["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != s["sha256"]:
        raise ValueError(f"request {rid}: {s['file']} is not the capture's ({s['sha256'][:16]})")
    tokens = np.asarray(s["tokens"], dtype="<u4")
    if served is not None and [int(x) for x in served] != tokens.tolist():
        differ = sum(int(a) != b for a, b in zip(served, tokens.tolist())) + abs(len(served) - len(tokens))
        raise ValueError(f"request {rid}: the replay selected {len(tokens)} ids and serving emitted {len(served)}, "
                         f"{differ} of them different: it didn't reproduce the span")
    n = len(tokens)
    m = CS.Member(t, (("in0", ins[0]),), ((f"out.{SAMPLED_MEMBER}", 2),), n, 0)
    data = {"rows": {"in0": np.frombuffer(raw, dtype="<u2").reshape(n, ins[0]).astype(np.uint16)},
            "outs": {f"out.{SAMPLED_MEMBER}": tokens.view("<u2").reshape(n, 2).astype(np.uint16)}}
    entries = [[j, j + 1, rid, step, None, SAMPLER_OP, SAMPLED_MEMBER, 0, None] for j, step in enumerate(s["steps"])]
    return Span(rid, m, data, entries, s["program"])


def _bf16(w: np.ndarray) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _combine(x0: np.ndarray, best: np.ndarray, idx: np.ndarray, c: int) -> np.ndarray:
    """`TokenSelectCombine_v2{V, C=c}` on each token's logits[0] and its chunks' (best, idx): from (x0, 0), chunk j's
    (best, j·c + idx) is taken iff its best is strictly greater, so a NaN x0 is never left."""
    run, tok, fb = _bf16(x0), np.zeros(len(x0), dtype=np.int64), _bf16(best)
    for j in range(best.shape[1]):
        take = fb[:, j] > run
        run, tok = np.where(take, fb[:, j], run), np.where(take, j * c + idx[:, j], tok)
    return tok


def named(m: CS.Member) -> CS.Member:
    """Unit member ``m`` (`chunk_rows.unit_member`) with each row named after the parameter it holds (row ``in<g>``,
    parameter g), as the second stage stages an interior member (`class_statement.member_stager`: row r is its
    Definition's input r) and round 2 names a wire's or a link's port."""
    names = {f"in{g}": p for g, (p, _t) in enumerate(SRW.template(m.template).fn.signature()[0])}
    if [r for r, _w in m.rows] != list(names):
        raise ValueError(f"{m.template}: rows {[r for r, _w in m.rows]} are not one whole row a parameter, {list(names)}")
    return replace(m, rows=tuple((names[r], w) for r, w in m.rows), schemas=tuple((names.get(r, r), s) for r, s in m.schemas))


def member(template: str, n: int = 1) -> CS.Member:
    """``n`` units of template ``template`` as an interior window lays them out: its bit rows as Flock stages a unit
    (`chunk_rows.unit_member`), each row named after its parameter (`named`)."""
    return named(CR.unit_member(SRW.template(template), n))


def pieces(s: Span, c: int | None = None) -> tuple[Span, Span]:
    """A token select's span ``s`` as the units Flock proves it by, as its template calls them. `TokenSelect_v4`: each
    token's V / c `TokenSelectChunk_v2{C=c}` units, one a chunk of its logits row, then its `TokenSelectCombine_v2{V, C=c}`
    unit over logits[0] and the chunks' (best, idx). `TokenSelectUnits_v1{V, K}`: each token's K
    `TokenSelectGroups_v2{C=V / K, G=CHUNK}` units, one a unit of its logits row (its chunks' bests, their indices and its
    first logit out), then its `TokenSelectCombineUnits_v1{V, CHUNK, K}` unit, whose ``x0`` is the first unit's first logit
    and whose ``best<j>`` and ``idx<j>`` are unit j's two output rows, whole. Each piece's member is its template's bit
    rows, as Flock stages a unit (`chunk_rows.unit_member`), each row named after its parameter (`named`). The first
    template's outputs are computed here (`numerics.token_select_rows`), and the combine's must be ``s``'s tokens, or
    nothing is committed. The first template's ``group`` is a token's units of it, so a span pads by whole tokens, as the
    select inlined in a replay block (one instance a served token) lays its pieces out.
    ``c``: `TokenSelect_v4`'s chunk, ``min(CHUNK, V)``; `TokenSelectUnits_v1`'s is `CHUNK`."""
    ((port, V),), ((out, _),) = s.member.rows, s.member.outs
    fn = SRW.template(s.member.template).fn
    units = getattr(fn, "defn", None) is TokenSelectUnitsV1
    if units and c not in (None, CHUNK):
        raise ValueError(f"{s.member.template}: its units' chunks are {CHUNK} logits, not {c}")
    c = CHUNK if units else min(CHUNK, V) if c is None else int(c)
    if not 1 <= c <= V or V % c:
        raise ValueError(f"{s.member.template}: {V} logits in chunks of {c} are not whole chunks, and a narrower last chunk "
                         "is a third template")
    n, k, x = s.n, V // c, s.data["rows"][port]
    best, idx = token_select_chunk_bests(x.reshape(n * k, c))
    tok, want = _combine(x[:, 0], best.reshape(n, k), idx.reshape(n, k), c), s.data["outs"][out].view("<u4")[:, 0]
    if not np.array_equal(tok, want):
        raise ValueError(f"request {s.rid}: its chunks and combine select other ids than its span's at "
                         f"{int(np.count_nonzero(tok != want))} of its {n} tokens")
    i32 = idx.astype("<u4").view("<u2")
    if units:
        K = fn.bindings["K"]
        g, rows = k // K, x.reshape(n * K, V // K)
        split = ((bind(TokenSelectGroupsV2, C=V // K, G=c).id, n * K, [rows],
                  [best.reshape(n * K, g), i32.reshape(n * K, 2 * g), rows[:, :1]]),
                 (bind(TokenSelectCombineUnitsV1, V=V, C=c, K=K).id, n,
                  [x[:, :1], *best.reshape(n, K, g).transpose(1, 0, 2), *i32.reshape(n, K, 2 * g).transpose(1, 0, 2)],
                  [s.data["outs"][out]]))
    else:
        split = ((f"TokenSelectChunk_v2{{C={c}}}", n * k, [x.reshape(n * k, c)], [best[:, None], i32.reshape(n * k, 2)]),
                 (f"TokenSelectCombine_v2{{V={V},C={c}}}", n, [x[:, :1], best.reshape(n, k), i32.reshape(n, 2 * k)],
                  [s.data["outs"][out]]))
    got = []
    for t, count, ins, outs in split:
        m = member(t, count)
        data = {"rows": {p: a.astype(np.uint16) for (p, _w), a in zip(m.rows, ins, strict=True)},
                "outs": {p: a.astype(np.uint16) for (p, _w), a in zip(m.outs, outs, strict=True)}}
        scale = count // n
        got.append(Span(s.rid, m, data, [[e[0] * scale, e[1] * scale, *e[2:]] for e in s.entries], s.program, group=scale))
    return got[0], got[1]


def whole(s: Span) -> Span:
    """A token select's span ``s`` (`captured`) as its template's own units, one a served token, where the class's replay
    block is the check's instances and not their bodies (`boundary_rows.Layout.inline` off): its member the template's
    layout as an interior window's (`member`), its output ``o0``, as round 2 names a link's port."""
    m = member(s.member.template, s.n)
    ((port, words),), ((out, wide),) = s.member.rows, s.member.outs
    if [w for _p, w in m.rows] != [words] or [w for _p, w in m.outs] != [wide]:
        raise ValueError(f"{s.member.template}: a unit of rows {list(m.rows)} and outputs {list(m.outs)}, not the span's one "
                         f"row of {words} words and one output of {wide}")
    return replace(s, member=m, data={"rows": {m.rows[0][0]: s.data["rows"][port]}, "outs": {m.outs[0][0]: s.data["outs"][out]}})


def interior_run(boundary: str, sample: Sequence[int]) -> str:
    """The interior window's run: SHA-512 over the boundary registration's digest and the replay sample, in order."""
    body = json.dumps({"boundary": boundary, "replay_sample": [int(u) for u in sample]}, separators=(",", ":"))
    return hashlib.sha512(RUN_TAG + body.encode()).hexdigest()


def _held(ms: Sequence[CS.Member], cells: Sequence[int], offs: Sequence[int], q: int,
          wires: Sequence[Any], rows: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Each member's wired ports, ``{port: (out, writer member, writers)}`` (`rows.Member.wired`): at wire w's port, unit u
    of span s = u // n_t reads unit off_w + s·n_w + x, the one of its writer's n_w in that span that round 2 reads there
    (`SecondStage.wiredRow`): x the row the replay block's ``rows`` give its unit u mod n_t (`interior_links.wired`), 0
    where the writer has one unit a block."""
    at = {m.template: k for k, m in enumerate(ms)}
    out: list[dict[str, Any]] = [{} for _ in ms]
    for w in wires:
        r, k = at.get(w.template), at.get(w.writer)
        if r is None or k is None:
            raise ValueError(f"a wire from {w.writer} to {w.template}, templates the window has not both of: {list(at)}")
        port = w.port
        if port not in dict(ms[r].rows):
            raise ValueError(f"{w.template}: a wire names its parameter {port!r}, which none of its rows "
                             f"{[p for p, _w in ms[r].rows]} is: an interior member's rows are its parameters (`named`)")
        if port in out[r]:
            raise ValueError(f"{w.template}: its {port!r} row is wired twice")
        x = IL.wired(rows, w.template, port, cells[r], cells[k])
        out[r][port] = (w.wport, k, tuple(offs[k] + (u // cells[r]) * cells[k] + x[u % cells[r]]
                                          for u in range(q * cells[r])))
    return out


def interior_window(members: Members, n_rho: Cells, boundary: str, sample: Sequence[int],
                    wires: Sequence[Any] = (), values: Sequence[tuple[str, int, Any]] = (),
                    rows: Sequence[Any] = ()) -> CS.Binding:
    """The window of the replay sample's interior, template-major: template t's Q·n_t units from off_t = Q·Σ_{t'<t} n_t',
    laid out as ``members[t]`` (its ports and their words), each a shared-row member of one table row a unit a port,
    under `interior_run`: what the prover commits and the challenger derives the ports of. ``members``: one, or one a
    template in the check's order; ``n_rho``: its n_t each (the stream's ``cells``), or one template's n_rho. ``wires``:
    the replay block's (`second_stage.replay_wires`). A wired port commits no tree of its own (`rows.Member.wired_trees`):
    its table is its writer's Q·n_w output rows, whole, and unit u of span s reads the one of its writer's n_w units in
    that span that round 2 reads there, which ``rows`` say (the replay block's, `second_stage.replay_rows`; needed where
    a writer has more than one unit a block). ``values``: the values its program takes ahead of the templates'
    parameters (a window stream's, `link_values`)."""
    ms, cells = IL.cells(members, n_rho)
    q = len(sample)
    sizes = [q * n for n in cells]
    offs = [int(o) for o in np.cumsum([0, *sizes[:-1]])]
    held = _held(ms, cells, offs, q, wires, rows)
    program, query = CS.population_program([(SRW.template(m.template).fn, n) for m, n in zip(ms, sizes, strict=True)],
                                           values=values)
    return CS.Binding(program, query, {"law": "subset", "k": sum(sizes)}, interior_run(boundary, sample),
                      tuple(replace(m, n=n, base=o, name="", wired=tuple((p, x[0], x[2]) for p, x in h.items()),
                                    wired_trees=not h, tables=tuple((p, sizes[h[p][1]] if p in h else n) for p, _ in m.rows))
                            for m, n, o, h in zip(ms, sizes, offs, held, strict=True)))


def interior_ports(members: Members, n_rho: Cells, boundary: str, sample: Sequence[int],
                   wires: Sequence[Any] = (), values: Sequence[tuple[str, int, Any]] = (),
                   rows: Sequence[Any] = ()) -> dict[str, dict[str, Any]]:
    """The challenger's copy of the interior registration's ports: `interior_window`'s, from the templates' layouts, the
    size class, the boundary registration's digest, its own replay sample, the wires and rows it derives from the replay
    block and its stream's values."""
    return CS.expected_ports(interior_window(members, n_rho, boundary, sample, wires, values, rows))


@dataclass(frozen=True)
class Interior:
    """An interior registration: its record, its ports and leaf layer as the challenger derives them from the released
    public bodies, the commitment it opens from, each span's index entries, a list a template, the sampled spans'
    tables as round 1 served them, none where no template reads one, and the block's returns round 1's record commits,
    each with that port as the serve opens it (its rows, salts and tree)."""
    record: dict[str, Any]
    ports: dict[str, dict[str, Any]]
    leaf_layer: str
    committed: CS.Committed
    entries: list[list[list[list[Any]]]]
    tables: tuple[HT.HistoryTable, ...] = ()
    returns: tuple[tuple[Return, LK.Port], ...] = ()

    @property
    def published(self) -> dict[str, list[str]]:
        """What the prover publishes of the sampled spans' tables, by name: each one's strings (b || c, hex), which the
        challenger's `history` reads as one held run and the verifier recomputes the served root from."""
        return {t.value: [bc.hex() for bc in t.bcs] for t in self.tables}


@dataclass(frozen=True)
class DeviceCommit:
    """An interior commitment made on the device the replayed rows are on (a GPU commit), by port of `interior_ports`: its
    root, every row's b || c, and the salt each row was hashed under, drawn as `interior` draws them. The rows never reach
    the host."""
    roots: Mapping[str, bytes]
    bc: Mapping[str, bytes]
    salts: Mapping[str, bytes]


def _refs(window: CS.Binding, m: CS.Member, ports: Sequence[tuple[str, int]]) -> np.ndarray:
    return np.array([window.held_leaves(m, p) for p, _w in ports], dtype=np.int64).T


def committed(window: CS.Binding, device: DeviceCommit) -> CS.Committed:
    """``window``'s commitment from ``device``, as `ml.windows.rows.commit` would make it from the rows under the same salts,
    without its rows (pass 2 replays the sampled ones). Refused unless it holds every port of the window, each a 64-byte root
    and one b || c and one salt a leaf; a wired port with no tree of its own holds its writer's, as `rows.commit` holds
    them."""
    want = {m.port(p): (m, p, n) for m in window.members for p, n in CS.port_leaves(m).items()}
    held = {m.port(p): window.held(m, p) for m in window.members if not m.wired_trees for p, _o, _ws in m.wired}
    if set(device.roots) != set(want) or set(device.bc) != set(want) or set(device.salts) != set(want):
        raise ValueError(f"a device commit of ports {sorted(device.roots)[:4]}..., not the window's {sorted(want)[:4]}...")
    for q, (_m, _p, n) in want.items():
        why = [w for w, bad in ((f"a {len(device.roots[q])}-byte root", len(device.roots[q]) != 64),
                                (f"{len(device.bc[q])} bytes of b || c", len(device.bc[q]) != n * CS.BC_BYTES),
                                (f"{len(device.salts[q])} bytes of salts", len(device.salts[q]) != n * CS.SALT_BYTES)) if bad]
        if why:
            raise ValueError(f"{q}: a device commit of {', '.join(why)}, for its {n} leaves")
    return CS.Committed(window, {q: bytes(device.roots[q]) for q in want},
                        {q: window.domain(q, m.schema(p), n) for q, (m, p, n) in want.items()},
                        {q: bytes(device.bc[held.get(q, q)]) for q in [*want, *held]},
                        {q: bytes(device.salts[held.get(q, q)]) for q in [*want, *held]}, {},
                        {"device": True}, {k: _refs(window, m, m.all_rows).astype(np.uint32) for k, m in enumerate(window.members)})


def _layout(m: CS.Member) -> tuple[Any, ...]:
    return m.template, m.rows, m.outs, m.schemas


def interior(spans: Sequence[Span | Sequence[Span]], n_rho: Cells, boundary: str, sample: Sequence[int], *, row: str,
             replay: str | None = None, previous: str | None = None, kind: str = "served",
             salt_key: bytes | None = None, device: DeviceCommit | None = None, wires: Sequence[Any] = (),
             tables: Sequence[HT.HistoryTable] = (), served: Mapping[str, Any] | None = None,
             positions: int | None = None, links: Sequence[Link] = (), rows: Sequence[Any] = (),
             window_root: str | None = None, returns: Sequence[tuple[Return, LK.Port]] = ()) -> Interior:
    """The replay sample's spans (``spans[i]`` replay block ``sample[i]``'s: one template's span, or one a template in the
    check's order, as `pieces` gives a token select's), template t's each padded to its n_t units (``n_rho``, as
    `interior_window` takes it) by whole instances (`Span.padded`), committed template-major and registered after
    ``previous`` (the boundary registration's digest ``boundary`` by default) under the window ``{kind, run, row,
    interior: true, replay_sample}``: ``row`` the served workload's row, ``replay_sample`` the replay sample in order
    (round 2's ``second-interior-order``), ``replay`` the replay's run when given. Every leaf's salt is fresh from the OS,
    but at a port ``links`` name, whose rows keep their value's salts; ``salt_key``: tests only, the salts derived from it
    (`ml.windows.rows.commit`). ``device``: the commitment made on the
    rows' device (`committed`), registered in place of committing the spans' rows, which it never reads. ``wires`` and
    ``rows``: the replay block's, as `interior_window` takes them; a wired port's replayed rows must be its writers'
    output rows, each unit's the writer unit's round 2 reads there, which are what the interior commits. ``tables``: ``tables[i]`` sampled span i's, one a span exactly when a template reads a
    table: the table round 1 served (`kept_table`, from `boundary_tables.load`), as round 1's record ``served`` registers it under its
    name, holding its class's ``positions`` strings (C, `boundary_tables.class_rows`). The interior commits none of them:
    every read must open its span's (`table_reads`), and the tables' strings are published (`Interior.published`).
    ``links``: the replay-block parameters its templates' ports read, each with the port that reads it, in the order
    the stream takes their values: the verifier's (`Supply`), among them every table read's ``root`` and ``domain``, and
    the replay's own (`Operand`). Such a port's rows are committed as rows of its value, on the host, which the
    interior's program takes (`link_values`: the stream is a window's, taking them a sampled span, `span_values`) and its
    record registers: `supplied`'s for a `Supply`, and for an `Operand` the rows its units replayed there. A replayed
    unit whose row there is not the value's at its place (a `Supply`'s row, or the row another unit replayed for the
    same row of an `Operand`) is refused before anything is committed. Several ports may read one value, each its own
    link. ``window_root``: the open root of the challenger session the second stage samples on, where the stream
    registers the interior under one (`Stream.register_interiors`' ``window_root``): each `Operand`'s domain binds it.
    ``returns``: the block's returns round 1's record commits (`Return`), each with that port as the serve opens it
    (`links.Port`: its rows and salts). The output port that writes one is committed under the serve's salts at the
    rows its units write (`Return.rows`), so each row's ``b || c`` is round 1's leaf there and a session reads it at
    round 1's root (`write`). A replayed unit whose output is not round 1's row at its place is refused before
    anything is committed, by slot and row, never by content."""
    if not spans or len(spans) != len(sample):
        raise ValueError(f"{len(spans)} spans for a replay sample of {len(sample)}")
    per = [(s,) if isinstance(s, Span) else tuple(s) for s in spans]
    if len({s.program for p in per for s in p}) != 1:
        raise ValueError(f"the spans were replayed from Programs {sorted({s.program[:16] for p in per for s in p})}, not one: "
                         "a size class's spans share its replay Definition")
    first = [_layout(s.member) for s in per[0]]
    if any([_layout(s.member) for s in p] != first for p in per):
        raise ValueError("the spans' units are not the same templates' in one layout and one order")
    members = [s.member for s in per[0]]
    readers = [(k, vh) for k, m in enumerate(members) if (vh := IL.reader(m)) is not None]
    q = len(sample)
    window = interior_window(members, n_rho, boundary, sample, wires, link_values(links, q), rows)
    ports, cells = CS.expected_ports(window), IL.cells(window.members, n_rho)[1]
    placed = IL.placed(window.members, cells, q, links)
    if (bool(readers) or bool(tables)) and len(tables) != len(sample) * bool(readers):
        raise ValueError(f"{len(tables)} tables for a replay sample of {len(sample)} and {len(readers)} templates of table "
                         "reads: each sampled span brings its table exactly when a template reads one")
    if tables and device is not None:
        raise ValueError("a commitment on the rows' device holds no table: a replay block of table reads commits on the host")
    if links and device is not None:
        raise ValueError("a commitment on the rows' device holds no value the interior registers: a replay block whose "
                         "ports read one commits on the host")
    if returns and device is not None:
        raise ValueError("a commitment on the rows' device makes its own salts: a replay block whose return round 1 "
                         "commits is committed on the host, under the serve's salts")
    back = _returned(window.members, cells, sample, returns)
    if tables and (served is None or positions is None):
        raise ValueError("a replay block of table reads opens the tables round 1 served: give round 1's record (served) "
                         "and its class's positions")
    for k, _vh in readers:
        t = window.members[k].template
        if not {(t, p) for p in TABLE} <= {(x.template, x.port) for x in links if isinstance(x, Supply)
                                           and x.index is None}:
            raise ValueError(f"{t}: a table read's {' and '.join(TABLE)} are the verifier's, the table round 1 served: "
                             "give the `Supply` of each")
    tcs = tuple(tables)
    for k, (V, H) in readers:
        if any(t.leaves != V or t.words != H for t in tcs):
            raise ValueError(f"{window.members[k].template} reads a table of {V} rows of {H} words, not the spans' "
                             f"{[(t.leaves, t.words) for t in tcs]}")
    roots = {r["port"]: r for r in served["roots"]} if tcs else {}
    for t in tcs:
        if len(t.bcs) != positions:
            raise ValueError(f"table {t.value} holds {len(t.bcs)} strings, not its class's {positions}: rows past its "
                             "length are filled (`boundary_tables.fill`), so its strings never show it")
        e, r = t.entry(), roots.get(t.value, {})
        if [r.get(k) for k in ("root", "leaves", "domain")] != [e[k] for k in ("root", "leaves", "domain")]:
            raise ValueError(f"table {t.value} is not the one round 1 registered under its name: a drawn span's reads open "
                             "the table served before the coins")
    prog = P.program_digest(window.program)
    schemas = link_schemas(window.members, links)
    given = {v.value: v for v in supplied(prog, links, q, [t.entry() for t in tcs], schemas)}
    if device is not None:
        c = committed(window, device)
    else:
        data = []
        for t, (m, n) in enumerate(zip(window.members, cells, strict=True)):
            padded = [p[t].padded(n) for p in per]
            d: dict[str, Any] = {k: {q: np.concatenate([x[k][q] for x in padded]) for q in padded[0][k]}
                                 for k in ("rows", "outs")}
            d["refs"] = _refs(window, m, m.rows)
            data.append(d)
        outs = {wm.port(o): (j, o) for j, wm in enumerate(window.members) for o, _w in wm.outs}
        for m, d in zip(window.members, data, strict=True):
            for p, _o, _ws in m.wired:
                j, o = outs[window.held(m, p)]
                if not np.array_equal(d["rows"][p], data[j]["outs"][o][list(window.held_leaves(m, p))]):
                    raise ValueError(f"{m.template}: its replayed {p!r} rows are not the {o!r} rows of the "
                                     f"{window.members[j].template} units that write them, which the interior commits")
        for k, (V, _H) in readers:
            IL.opened(window.members[k], data[k], cells[k], tcs, V)
        for m, d in zip(window.members, data, strict=True):  # after the reads: a read's key may be a wired row
            for p, _o, _ws in m.wired:
                d["rows"][p] = CS.WIRED
        took = [np.asarray(data[k]["rows"][x.port], dtype=np.uint16) for x, k, _at in placed]
        for name, x in IL.values(links).items():
            if isinstance(x, Operand):
                reads = [(rows, at) for rows, (y, _k, at) in zip(took, placed, strict=True) if y.value == name]
                given[name] = IL.operand(prog, x, reads, q, schemas[name], salt_key, window_root)
        held = {name: np.asarray(v.rows, dtype=np.uint16) for name, v in given.items()}
        for rows, (x, k, at) in zip(took, placed, strict=True):
            v = given[x.value]
            off = np.flatnonzero((rows != held[x.value][list(at)]).any(axis=1))
            if off.size:
                u = int(off[0])
                raise ValueError(f"{x.template}: unit {u} (sampled span {u // cells[k]}) takes at its {x.port!r} a row that "
                                 f"is not row {at[u]} of {x.value}, " + (
                                     "which another unit reads as another row" if isinstance(x, Operand)
                                     else "the value the verifier supplies it"))
            data[k]["rows"][x.port] = CS.ValueRows(v, at)
        linked = {window.members[k].port(x.port) for x, k, _at in placed}
        for x, v, k, at in back:
            off = np.flatnonzero((np.asarray(data[k]["outs"][x.port], dtype=np.uint16) != v.rows_at(at)).any(axis=1))
            if off.size:
                u = int(off[0])
                raise ValueError(f"{x.template}: unit {u} (sampled span {u // cells[k]}) returns at its {x.port!r} a row that "
                                 f"is not row {at[u]} of {x.value}, the return round 1 committed: the replay didn't "
                                 "reproduce what was served")
        fresh = {} if salt_key is not None else {p: os.urandom(v["leaves"] * CS.SALT_BYTES) for p, v in ports.items()
                                                 if p not in linked}
        fresh |= {window.members[k].port(x.port): v.salts_at(at) for x, v, k, at in back}
        c = CS.commit(window, data, salt_key=salt_key, salts_of=fresh or None)
    provenance = {"row": row, "interior": True, "replay_sample": [int(u) for u in sample],
                  **({"replay": replay} if replay is not None else {})}
    rec = CS.registration(c, provenance, boundary if previous is None else previous, kind=kind, registrar=R,
                          entries=[given[v].entry() for v in IL.values(links)])
    layer = CS.leaf_layer({m.key: c.public_body(k) for k, m in enumerate(window.members)})
    return Interior(rec, ports, layer, c, [[s.entries for s in p] for p in per], tcs, tuple(returns))


def _returned(ms: Sequence[CS.Member], cells: Sequence[int], sample: Sequence[int],
              returns: Sequence[tuple[Return, LK.Port]]) -> list[tuple[Return, LK.Port, int, list[int]]]:
    """Each of ``returns`` with round 1's port, its member and the row of the port each of the member's Q·n_t units
    writes (`Return.rows`). Refused by name for a return round 1's record doesn't commit, a template the window lacks,
    a port that is none of its outputs or is returned twice, a placement of other than its n_t units, and a port
    opened as another value, in another row format or without the serve's rows and salts."""
    at = {m.template: k for k, m in enumerate(ms)}
    out, seen = [], set()
    for x, v in returns:
        k = at.get(x.template)
        if k is None:
            raise ValueError(f"{x.value} is written by {x.template}, a template the window has none of: {list(at)}")
        m = ms[k]
        if not x.held or v.value != x.value:
            raise ValueError(f"{x.template}: its {x.port!r} writes {x.value}, which round 1's record doesn't commit or "
                             f"which is opened as {v.value!r}")
        if x.port not in dict(m.outs) or (x.template, x.port) in seen:
            raise ValueError(f"{x.template}: {x.value} is written at its {x.port!r}, which is none of its outputs "
                             f"{[p for p, _w in m.outs]} or is returned twice")
        if (v.words, v.schema) != (dict(m.outs)[x.port], m.schema(x.port)):
            raise ValueError(f"{x.template}: its {x.port!r} row is {dict(m.outs)[x.port]} words in {m.schema(x.port)}, not "
                             f"{x.value}'s {v.words} in {v.schema}")
        if len(x.at) != cells[k] or v.leaves % x.per:
            raise ValueError(f"{x.template}: {x.value} places {len(x.at)} units, not its {cells[k]} a sampled span, or "
                             f"holds {v.leaves} rows, not whole blocks of {x.per}")
        rows = x.rows(sample)
        if not all(0 <= r < v.leaves for r in rows) or v.rows is None or v.salts is None:
            raise ValueError(f"{x.value}: the replay sample reads past its {v.leaves} rows, or the serve's rows and "
                             "salts are not held")
        seen.add((x.template, x.port))
        out.append((x, v, k, rows))
    return out


def write(it: Interior, directory: str | Path) -> dict[str, Any]:
    """``it``'s window files under ``directory`` as the one-stage audit reads a window (`ml.windows.rows.write`), its
    registration the one `interior` made but for the index's hiding leaf, which the written record's window names:
    -> that record, the one to register (`register`'s ``record``). Each member's header names every port's rows, so a
    session's registered reads cover its circuit: a port given as rows of a value the interior registers (a link's), that
    value's rows with their paths to its root; an output that writes a return round 1's record commits
    (`Interior.returns`), round 1's port, the rows its units write and their paths to round 1's root; any other, the
    rows of the port holding it (its own, or a wired port's writer's output) with their paths. ``index.json`` is the
    spans' index entries (`Interior.entries`), prover only."""
    c, w = it.committed, it.committed.window
    held: dict[str, LK.Port] = {}
    registered: dict[str, dict[str, Any]] = {}
    back = {(x.template, x.port): (x, v) for x, v in it.returns}
    for k, m in enumerate(w.members):
        given, entry = c.by_value(k), {}
        for p, _w in m.all_rows:
            if p in given:
                v, at = given[p].value, [int(x) for x in given[p].positions]
                entry[p] = {"value": v.value, "positions": at, "paths": [v.path(x).hex() for x in at]}
            elif (m.template, p) in back:
                x, v = back[m.template, p]
                entry[p] = v.header(x.rows(it.record["window"]["replay_sample"]))
            else:
                port = w.held(m, p)
                if port not in held:
                    held[port] = LK.Port.of(c, port)
                # a wired port with no tree of its own has its writer's rows, whole, as its table: Flock's verifier takes
                # one position and one path a table row (`Flock.Registered.checkPort`), not one a reading unit
                at = range(dict(m.tables)[p]) if port != m.port(p) else w.held_leaves(m, p)
                entry[p] = held[port].header(at)
        registered[m.key] = entry
    window = {k: v for k, v in it.record["window"].items() if k not in ("kind", "run")}
    CS.write(c, directory, window, {"interior": True, "members": [m.key for m in w.members], "entries": it.entries},
             kind=it.record["window"]["kind"], registered=registered, registrar=R, previous=it.record["previous"],
             entries=it.record.get("registered", ()))
    return json.loads((Path(directory) / "registration.json").read_text())


def register(stream: Any, first: Any, it: Interior, **copies: Any) -> Any:
    """``it`` on the two-stage ``stream`` after its first stage ``first``, with the challenger's ports and leaf layer
    (``copies`` override them, and pass `Stream.register_interiors`' other copies): its logged registration."""
    return stream.register_interiors(first, it.record, copies.pop("ports", it.ports),
                                     leaf_layer=copies.pop("leaf_layer", it.leaf_layer), **copies)


__all__ = ["DIGEST", "HELD", "INDEX", "MOVED", "READ", "TABLE", "DeviceCommit", "Interior", "Link", "Operand", "Return",
           "Span", "Supply", "captured", "committed", "gather", "history", "interior", "interior_ports", "interior_run",
           "interior_window", "kept_table", "link_schemas", "link_values", "link_words", "links", "member", "named",
           "pieces", "port_map", "register", "returns", "sources", "span", "span_values", "supplied", "table_reads", "whole",
           "write"]
