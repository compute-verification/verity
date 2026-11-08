"""A served window's edge (`pipeline.edge`): each edge port's rows wired from the proof blocks that output them, so the
history the next window reads is what this window computed, and the next window registers it under this window's roots.

- **Sources.** A request Program a window leaves returns its edge fields after its samples (`predict.common.with_edge`), and
  its instance sequence says where each field's leaves come from: ``edge``, by port, the leaf width ``bits`` and the field's
  ``runs`` as argument items (`compact.runs_from_file_items`). Cut into rows of the port's words, each row is one Program row's
  committed member, whole: a **writer**, the window's unit for that row, whose output port is the row (`wiring`); or whole
  rows of the request's history parameter: a **held** row, the last window's row at the request's field there. Any other row
  is refused by name: a row whose words two blocks output (GLM-5.3's latent row, the kv norm's output then the kv rope's)
  has no one output string, and a row of a template whose unit is not one Program row (a head, a chunk, a coordinate) is not
  one unit's output row.
- **The commitment.** The prover commits each port as a registered value (core's ``registered.Registered``) whose row r is
  committed under its source's salt (`commit`): a writer's output row's, or the last window's for a held row, so the row's
  ``b || c`` is its source's string. Its domain binds the window's program, the port and the window's open root (`ports`),
  and its root is named among the record's roots after the window's ports (`rows.registration`'s ``outputs``), where R3
  holds it to that domain and leaf count. `FILE` is public (each port's layout, root, members and every row's path) and
  `PRIVATE` the prover's (each row's leaves and salt), which the next window reads for its held rows.
- **The verifier.** `check` recomputes each port's root from its sources' strings, the writers' from the public bodies the
  leaf layer binds (`wiring.strings`) and the held rows' from the last window's check, and refuses a port whose root is not
  that tree. `check_drawn` opens each drawn writer's row against its port's root through the path `FILE` publishes, work
  proportional to the sample; a held row was its writer's in an earlier window, and is checked there.
- **The next window.** Its program takes each port the last window output as a value (`values`, `population_program`'s), its
  record registers them under the last record's roots and domains (`entries`) chained after that record (`previous`), and
  its verifier gives each as a history port (`edge.history`), which R7 holds to the last record's port.

The served request's ``edge`` (`of_request`): ``{"window": <the window's open root, hex>, "words": {port: words}, "order":
[request, ...], "last": <the last window's dir> | null}``, ``order`` the requests the window leaves in the edge's order.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as REG, registration as R
from verity.ml.commitments.merkle import tree_levels
from verity.ml.windows import wiring as WI
from verity_vllm.program.compact import runs_from_file_items

from . import edge as E
from .serving_templates import template, unit_templates

FILE, PRIVATE, HEADER = "edge.json", "edge-private.npz", "edge"
SALT, BC = WI.SALT_BYTES, WI.BC_BYTES
HISTORY = {v.name: v.history for v in E.VALUES}

__all__ = ["FILE", "HEADER", "PRIVATE", "Last", "Port", "Served", "Spec", "check", "check_drawn", "commit", "entries", "last",
           "of_rank", "of_request", "ports", "served", "values", "wiring", "write"]


class Spec(NamedTuple):
    """A served window's edge: its open root, each port's row width in leaves, the requests it leaves in order, and the last
    window's dir (None for a window that carries no history)."""

    window: str
    words: Mapping[str, int]
    order: tuple[str, ...]
    last: str | None


class Last(NamedTuple):
    """The last window as the next one reads it: its public edge (`FILE`), its registration's digest and, for the prover,
    each port's rows' leaves and salts (`PRIVATE`; empty for the verifier)."""

    doc: Mapping[str, Any]
    digest: str
    rows: Mapping[str, np.ndarray]
    salts: Mapping[str, np.ndarray]


class Port(NamedTuple):
    """One edge port of a window: its rows' width in leaves and leaf width, each row's source (a writer's global unit and
    output port, or None and the held row's position in the last window's value), and its members (request, rows)."""

    name: str
    words: int
    word_bits: int
    sources: tuple[tuple[int | None, str | int], ...]
    members: tuple[tuple[str, int], ...]


class Served(NamedTuple):
    """What a served window's record takes from its edge: the roots of the ports it outputs, the history values it registers
    and the record it follows."""

    outputs: tuple[Mapping[str, Any], ...] = ()
    entries: tuple[Mapping[str, Any], ...] = ()
    previous: str | None = None


def of_request(edge: Mapping[str, Any] | None) -> Spec | None:
    """The served request's ``edge`` as a `Spec`; ValueError on any other form."""
    if edge is None:
        return None
    ok = (isinstance(edge, Mapping) and set(edge) == {"window", "words", "order", "last"} and isinstance(edge["window"], str)
          and REG.window_root(edge["window"]) == edge["window"] and isinstance(edge["words"], Mapping)
          and all(p in HISTORY and type(w) is int and w > 0 for p, w in edge["words"].items())
          and isinstance(edge["order"], list) and all(isinstance(r, str) for r in edge["order"])
          and len(set(edge["order"])) == len(edge["order"]) and (edge["last"] is None or isinstance(edge["last"], str)))
    if not ok:
        raise ValueError(f"a served edge is {{window: <open root>, words: {{port: words}}, order: [request], last: <dir> | null}} "
                         f"over the ports {sorted(HISTORY)}: {edge!r}")
    return Spec(edge["window"], dict(edge["words"]), tuple(edge["order"]), edge["last"])


def of_rank(edge: Mapping[str, Any] | None, rank: int) -> Mapping[str, Any] | None:
    """A TP Commit's request ``edge`` for rank ``rank``'s window (`tp.serving_rows`): its last window is that rank's, under the
    last Commit's ``rank<r>/``."""
    if edge is None or not isinstance(edge, Mapping) or edge.get("last") is None:
        return edge
    return {**edge, "last": str(Path(edge["last"]) / f"rank{int(rank)}")}


def last(spec: Spec | None, private: bool = True) -> Last | None:
    """The last window ``spec`` names, read from its dir; the prover's rows and salts with ``private``."""
    if spec is None or spec.last is None:
        return None
    d = Path(spec.last)
    doc = json.loads((d / FILE).read_text())
    rows, salts = {}, {}
    if private and doc["ports"]:
        with np.load(d / PRIVATE) as z:
            for k, p in enumerate(doc["ports"]):
                rows[p], salts[p] = z[f"rows{k}"], z[f"salts{k}"]
    return Last(doc, R.digest(json.loads((d / "registration.json").read_text())), rows, salts)


def _held(doc: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return {} if doc is None else doc["ports"]


def values(doc: Mapping[str, Any] | None) -> tuple[tuple[str, int, int], ...]:
    """The values a window's program takes for the history the last window left it (``doc``, its `FILE`): each port as
    ``(name, leaves, width)`` (`population_program`'s ``values``)."""
    return tuple((p, int(e["leaves"]) * int(e["words"]), int(e["word_bits"])) for p, e in _held(doc).items())


def entries(doc: Mapping[str, Any] | None) -> tuple[dict[str, Any], ...]:
    """The registered entries of the history the last window left (``doc``, its `FILE`): each port under its own name, root
    and domain, as R7 holds a history port to the previous record's port."""
    return tuple({"value": p, "root": e["root"], "leaves": int(e["leaves"]), "words": int(e["words"]),
                  **REG.width_entry(int(e["word_bits"])), "domain": e["domain"]} for p, e in _held(doc).items())


def ports(program: str, edge: Sequence[Port], window: str) -> dict[str, dict[str, Any]]:
    """The verifier's ports for a window's edge, as R3 holds the record's roots to them: each port's leaf count and the
    registered domain of its value under the window's ``program`` digest and open root ``window`` (core's ``registered.domain``)."""
    return {p.name: {"leaves": len(p.sources), "domain": _domain(program, p, window).domain_id.hex()} for p in edge}


def _domain(program: str, p: Port, window: str) -> Any:
    return REG.domain(program, p.name, p.words, len(p.sources), p.word_bits, window)


def _units(window: Any, rows: Sequence[Sequence[Any]]) -> dict[tuple[str, int], tuple[Any, int | None]]:
    """Each Program row of the window's members, by (request, row index): its member and global unit for a row template's
    rows, one unit each of the row's own Definition, else its member and None (its units are not one Program row each)."""
    out: dict[tuple[str, int], tuple[Any, int | None]] = {}
    tmpls = list(dict.fromkeys(m.template for m in window.members))
    for t, rs in zip(tmpls, rows, strict=True):
        ms = [m for m in window.members if m.template == t]
        whole = template(t).layout == "row" and len(ms) == 1 and not ms[0].schemas and ms[0].n == len(rs)
        for k, (rid, _pi, r, _d) in enumerate(rs):
            one = whole and (t, 1) in unit_templates(r["spec"])
            if one or (str(rid), int(r["i"])) not in out:
                out[(str(rid), int(r["i"]))] = (ms[0], ms[0].base + k if one else None)
    return out


def _cut(items: Any, words: int, what: str) -> list[tuple[str, int, int, int]]:
    """A field's runs as rows of ``words`` leaves, each one run; ValueError naming the first row that is not."""
    out, k = [], 0
    for sl in runs_from_file_items(items):
        kind, a, lo, hi = str(sl[0]), int(sl[1]), int(sl[2]), int(sl[3])
        if (hi - lo) % words:
            raise ValueError(f"{what}: row {k} is not one run of {words} leaves (a run of {hi - lo} from {kind}{a})")
        for at in range(lo, hi, words):
            out.append((kind, a, at, at + words))
            k += 1
    return out


def wiring(window: Any, rows: Sequence[Sequence[Any]], programs: Mapping[str, Mapping[str, Any]], spec: Spec,
           doc: Mapping[str, Any] | None) -> tuple[Port, ...]:
    """Each edge port of the window that ``spec`` leaves: its rows' sources, from the instance sequences' ``edge`` of the
    requests in ``spec.order`` (`HEADER`), a writer the window's unit outputting the row whole and a held row the last
    window's (``doc``, its `FILE`) at the request's field there; ValueError naming the first row that is neither."""
    units, held = _units(window, rows), _held(doc)
    pis = {str(rid): pi for rs in rows for rid, pi, _r, _d in rs}
    out: dict[str, list] = {}
    for rid in spec.order:
        head = programs[rid].get(HEADER) if rid in programs else None
        if not isinstance(head, Mapping) or not head or set(head) - set(spec.words):
            raise ValueError(f"request {rid}: the window leaves it, and its Program returns no edge fields of {sorted(spec.words)}")
        names = [p for p, _h in programs[rid]["params"]]
        for p, f in head.items():
            what, words = f"{p} of request {rid}", spec.words[p]
            bits, srcs = int(f["bits"]), []
            for kind, a, lo, hi in _cut(f["runs"], words, what):
                if kind == "p":
                    if names[a] != HISTORY[p]:
                        raise ValueError(f"{what}: row {len(srcs)} is a row of parameter {names[a]}, not of its history {HISTORY[p]}")
                    left = held[p]["members"] if p in held else ()
                    if left and (int(held[p]["words"]), int(held[p]["word_bits"])) != (words, 16 * -(-bits // 16)):
                        raise ValueError(f"{what}: the last window's rows are {held[p]['words']} leaves of {held[p]['word_bits']} bits, "
                                         f"not {words} of {bits}")
                    base, n = next(((sum(int(x) for _r, x in left[:k]), int(n)) for k, (r, n) in enumerate(left) if r == rid), (0, 0))
                    if hi > n * words:
                        raise ValueError(f"{what}: row {len(srcs)} reads history the last window did not leave it ({n} rows of {p})")
                    srcs.append((None, base + lo // words))
                    continue
                m = pis[rid].member_at(a, lo, hi) if rid in pis else None
                member, unit = units.get((rid, a), (None, None))
                out_port = f"out.{m[1]}" if m is not None else None
                if m is None or unit is None or (m[2], m[3]) != (0, int(dict((x[0], x[1]) for x in m[0]["members"])[m[1]])) \
                        or dict(member.outs).get(out_port) != words * (16 * -(-bits // 16)) // 16 \
                        or member.schema(out_port) != REG.SCHEMA:
                    raise ValueError(f"{what}: row {len(srcs)} (row {a}'s leaves [{lo}, {hi})) is not one row template unit's "
                                     "whole output row in the window, a row/v1 row of the port's words")
                srcs.append((unit, out_port))
            ent = out.setdefault(p, [bits, [], []])
            if ent[0] != bits:
                raise ValueError(f"{what}: {bits}-bit leaves, the port's are {ent[0]}-bit")
            ent[1] += srcs
            ent[2].append((rid, len(srcs)))
    return tuple(Port(p, spec.words[p], 16 * -(-b // 16), tuple(s), tuple(ms)) for p in HISTORY if p in out
                 for b, s, ms in (out[p],))


def _member(members: Sequence[Any], at: np.ndarray, unit: int) -> tuple[Any, int]:
    k = int(np.searchsorted(at, unit, side="right") - 1)
    return members[k], unit - members[k].base


def _leaves(row: np.ndarray, words: int, bits: int) -> list[int]:
    """A row's ``words`` leaves of ``bits`` bits off its u16 words, low half first."""
    k = bits // 16
    w = np.asarray(row[:words * k], dtype=np.uint64).reshape(words, k)
    return [int(x) for x in (w << (16 * np.arange(k, dtype=np.uint64))).sum(axis=1)]


def commit(c: Any, edge: Sequence[Port], window: str, hist: Last | None) -> dict[str, REG.Registered]:
    """The prover's: each port of ``edge`` committed as a registered value of the window ``c`` (`rows.Committed`) under its
    program and open root ``window``, row r its source's leaves under its source's salt (``hist`` for held rows), so its
    ``b || c`` is its source's string; ValueError naming a row whose string is not."""
    ms, prog = c.window.members, c.window.partition["program"]
    at, out = WI.bases(ms), {}
    for p in edge:
        rows, salts, want = [], [], []
        for unit, src in p.sources:
            if unit is None:
                rows.append([int(x) for x in hist.rows[p.name][src]])
                salts.append(bytes(hist.salts[p.name][src]))
                want.append(None)
                continue
            m, j = _member(ms, at, unit)
            rows.append(_leaves(c.rows[m.port(src)][j], p.words, p.word_bits))
            salts.append(c.salts[m.port(src)][j * SALT:(j + 1) * SALT])
            want.append(c.bc[m.port(src)][j * BC:(j + 1) * BC])
        out[p.name] = REG.Registered.commit(prog, p.name, rows, salts, word_bits=p.word_bits, window=window)
        bad = [r for r, w in enumerate(want) if w is not None and out[p.name].bcs[r] != w]
        if bad:
            raise ValueError(f"{p.name}: row {bad[0]}'s string is not its writer's output string")
    return out


def write(out_dir: str | Path, committed: Mapping[str, REG.Registered], edge: Sequence[Port]) -> None:
    """The window's `FILE` (each port's layout, root, members and every row's path) and `PRIVATE` (each row's leaves and salt)."""
    d = Path(out_dir)
    doc = {"ports": {p.name: {"words": p.words, "word_bits": p.word_bits, "leaves": len(p.sources),
                              "domain": committed[p.name].domain.domain_id.hex(), "root": committed[p.name].root.hex(),
                              "members": [list(x) for x in p.members],
                              "paths": [committed[p.name].path(r).hex() for r in range(len(p.sources))]} for p in edge}}
    (d / FILE).write_text(json.dumps(doc, indent=1) + "\n")
    arrays = {}
    for k, p in enumerate(edge):
        reg = committed[p.name]
        arrays[f"rows{k}"] = np.asarray(reg.rows, dtype=np.uint32)
        arrays[f"salts{k}"] = np.frombuffer(b"".join(reg.salts), dtype=np.uint8).reshape(-1, SALT)
    fd = os.open(d / PRIVATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        np.savez(f, **arrays)


def served(spec: Spec | None, hist: Last | None, c: Any, rows: Sequence[Sequence[Any]],
           programs: Mapping[str, Mapping[str, Any]], out_dir: str | Path) -> Served:
    """A served window's edge (`serving_rows.after_commit`): its ports wired (`wiring`), committed (`commit`) and written
    (`write`), and what its record takes: their roots, the history's entries and the last record's digest."""
    if spec is None:
        return Served()
    edge = wiring(c.window, rows, programs, spec, hist.doc if hist else None) if spec.order else ()
    committed = commit(c, edge, spec.window, hist)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write(out_dir, committed, edge)
    return Served(tuple({"port": p, "root": r.root.hex(), "leaves": r.leaves, "domain": r.domain.domain_id.hex()}
                        for p, r in committed.items()), entries(hist.doc if hist else None), hist.digest if hist else None)


def _strings(members: Sequence[Any], publics: Mapping[str, bytes], p: Port,
             held: Mapping[str, Sequence[bytes]] | None) -> list[bytes | None]:
    at, cache, out = WI.bases(members), {}, []
    for unit, src in p.sources:
        if unit is None:
            hs = (held or {}).get(p.name)
            out.append(hs[src] if hs is not None and 0 <= src < len(hs) else None)
            continue
        m, j = _member(members, at, unit)
        if (m.key, src) not in cache:
            cache[(m.key, src)] = WI.strings(publics, m, src)
        got = cache[(m.key, src)]
        out.append(None if got is None else got[j].tobytes())
    return out


def check(members: Sequence[Any], publics: Mapping[str, bytes], roots: Mapping[str, str], edge: Sequence[Port], program: str,
          window: str, held: Mapping[str, Sequence[bytes]] | None = None) -> tuple[dict[str, str], dict[str, list[bytes]]]:
    """The verifier's edge check: each port's root (``roots``, the record's) is the tree, under its domain (`ports`), over its
    rows' sources' strings, a writer's from the public bodies (``publics``, which the leaf layer binds) and a held row's from
    the last window's check (``held``). -> (``{port: why}``, empty when every port holds; each port's strings, for the next
    window's ``held``)."""
    bad, out = {}, {}
    for p in edge:
        got = _strings(members, publics, p, held)
        if any(s is None for s in got):
            bad[p.name] = f"row {got.index(None)}'s source has no string: no public body of its writer, or no held row"
            continue
        dom = _domain(program, p, window)
        if tree_levels(dom, [REG.leaf(dom, r, s) for r, s in enumerate(got)])[-1][0].hex() != roots.get(p.name):
            bad[p.name] = "its root is not the tree over its rows' sources' strings"
        out[p.name] = got
    return bad, out


def _paths(doc: Mapping[str, Any], port: str) -> list[bytes]:
    """Port ``port``'s row paths in a prover's `FILE`; none for a file that does not give them."""
    try:
        return [bytes.fromhex(x) for x in doc["ports"][port]["paths"]]
    except (KeyError, TypeError, ValueError):
        return []


def check_drawn(members: Sequence[Any], publics: Mapping[str, bytes], roots: Mapping[str, str], edge: Sequence[Port],
                program: str, window: str, doc: Mapping[str, Any], units: Any) -> dict[str, str]:
    """`check` on the sampled proof blocks alone (``units``, global unit indices): each drawn writer's row opens its port's
    root under its domain with the writer's string, through the path the window's `FILE` (``doc``) gives, work proportional
    to the drawn writers. ``{port: why}``; empty when every drawn writer's row opens."""
    drawn, at, bad = {int(u) for u in units}, WI.bases(members), {}
    for p in edge:
        dom, paths = _domain(program, p, window), _paths(doc, p.name)
        for r, (unit, src) in enumerate(p.sources):
            if unit is None or unit not in drawn:
                continue
            m, j = _member(members, at, unit)
            got = WI.strings(publics, m, src)
            ok = (got is not None and r < len(paths) and roots.get(p.name) is not None
                  and REG.verify_read(dom, bytes.fromhex(roots[p.name]), r, got[j].tobytes(), paths[r]))
            if not ok:
                bad[p.name] = f"row {r}, unit {unit}'s {src}, does not open its root with its writer's string"
                break
    return bad
