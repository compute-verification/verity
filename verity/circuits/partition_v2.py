"""`verity/partition/v2`: the one partition form, private. Its content is a partition object's `blocks` (`partition_object`):
levels, coarsest first, each a list of ranges [lo, hi) of whole units of a template query, in order, covering them, and each a
refinement of the one before. Its envelope is advice: fixed-width rows committed before any coin (hm96-sha512 hiding rows in
one frame-v3-sha512 tree, `verity.commitments.advice`), of which the verifier holds only the root and a statement naming the
public predicates they meet. The public form is the same object with the advice opened (`blocks`).

**Statement** (public; `statement`, `validate`, `digest` = `SHA-512("verity/partition/v2\\0" || canon(statement))`)::

    {"format": "verity/partition/v2", "program": "<SHA-512 of C, hex>", "query": <a template query>,
     "predicates": {"X": int, "max_gates": int | null, "max_in_bits": int | null, "grid_bits": g},
     "blocks": [[grid(b_jl) for each level l] for each execution phase j],
     "advice": {"rows": grid(R), "root": "<frame-v3-sha512 root, hex>"}}

`b_jl` and R are the block count of phase j at level l and the row count, rounded up to the public grid (`grid`: at most g
significant bits), so the verifier learns a grid level of each, never the counts.

**Rows** (private; `advice`, `rows`, `decode`). `ROW_BYTES` bytes, integers little-endian u64, every unused byte 0:

* row 0, the header: `MAGIC`, C's SHA-512, the population N, the section count, the design section's first row and rows;
* row 1: the SHA-512 of the partition object {format, program, query} (`partition_object.digest`);
* rows 2 .. `header_rows(n)` - 1: one entry a section, `SECTIONS_PER_ROW` to a row: its kind, phase, level, first row and
  rows, b_jl and grid(b_jl);
* a section per phase and level, phase by phase and level by level within it: grid(b_jl) + 1 boundaries s_0 ..
  s_grid(b_jl), `BOUNDS_PER_ROW` to a row (kind `ENDS`): s_0 where the phase's units start (0, or the phase before's
  last end) and s_(i+1) block i's end, so block i is [s_i, s_(i+1)); blocks b_jl .. grid(b_jl) - 1 are filler, empty,
  each ending at s_(b_jl);
* the design section: the sampling design's rows, none today;
* zero rows up to grid(R).

Each section's rows follow from the statement alone (`section_row`), so a sampled block's rows depend on its index and the
public counts, never on b_jl or N. A kind names a section's record; `ENDS` is the only one, and a reader refuses any other.

**Global check** (`check`; refusal codes in brackets), over C and the rows, once per registration: the rows' count and form
[rows-count, rows-malformed]; the header names C, its object and its population [header-program, header-object,
header-population] and lays the sections out as the statement does [header-layout]; no design rows [design-rows]; a
phase covers the same units at every level [phase-span]; each level, the phases' real blocks side by side, is a valid
level of the object's `blocks`, each covering C's units in order and refining the one before (`partition_object.verify`'s
codes); a filler is empty [filler]; each section's count is on the grid [blocks-bucket]; each block of the last level has
at most X bits out and at most max_gates gates and max_in_bits bits in (`restrict.front`) [block-too-wide,
block-too-large].

**Local check** (`check_block`), with sampled block i of phase j at level l: the at most two rows holding s_i and s_(i+1)
(`openings`), opened against the root, give its range; a filler is empty, and any other is the block the door restricts.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import partition_object as PO
from . import restrict as R
from .boundary import gate_width
from .codec import canonical_json
from .partition import CutVerdict

FORMAT = "verity/partition/v2"
ROW_BYTES = 128
ROW_BITS = 8 * ROW_BYTES
MAGIC = b"verity/partition/v2\0"
#: the 32-byte binding of the advice tree's domain (`verity.commitments.advice`)
BINDING = hashlib.sha512(b"verity/partition/v2/advice").digest()[:32]
GRID_BITS = 4
#: the widest public size class of a proof block's bits out (the claims' `Claims.Sampling.sizeClassMax`), provisional: at 16,
#: `SIZE_CLASSES` is the literal 16-bit limit alone
SIZE_CLASS_MAX = 262_144
#: the public size classes, the powers of two from 16 to `SIZE_CLASS_MAX`, ascending (the claims' `sizeClasses`)
SIZE_CLASSES = tuple(16 << i for i in range((SIZE_CLASS_MAX // 16).bit_length()))
#: a section's kind: its blocks' boundaries as u64s
ENDS = 1
BOUNDS_PER_ROW = ROW_BYTES // 8
ENTRY = 7 * 8
SECTIONS_PER_ROW = ROW_BYTES // ENTRY
HEADER = 2

_KEYS = {"format", "program", "query", "predicates", "blocks", "advice"}
_PRED_KEYS = {"X", "max_gates", "max_in_bits", "grid_bits"}
_HEX512 = re.compile(r"^[0-9a-f]{128}$")
_MAX_LISTED = 64


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def grid(x: int, bits: int = GRID_BITS) -> int:
    """The least y >= x with at most `bits` significant bits: x itself below 2^bits, else x rounded up to a multiple of
    2^(bit_length(x) - bits).  y < x * (1 + 2^(1 - bits)), so fewer than one in 2^(bits - 1) of y's units are fillers."""
    if not (_is_int(x) and x >= 0 and _is_int(bits) and bits >= 1):
        raise ValueError(f"grid of a count >= 0 at bits >= 1, not {x!r}, {bits!r}")
    e = max(0, x.bit_length() - bits)
    return -(-x >> e) << e


def id_digest(fn: Any) -> str:
    """The SHA-512 a Definition is named by in advice: of its specialization id, `fn.id`."""
    return hashlib.sha512(b"verity/partition/v2/definition\0" + fn.id.encode()).hexdigest()


def size_class(bits: int) -> int | None:
    """A block's size class: the least of `SIZE_CLASSES` that holds its ``bits`` out (`restrict.front`), None past the widest."""
    return next((w for w in SIZE_CLASSES if bits <= w), None)


def header_rows(sections: int) -> int:
    return HEADER + -(-sections // SECTIONS_PER_ROW)


def _section_rows(padded: int) -> int:
    return -(-(padded + 1) // BOUNDS_PER_ROW)


def _order(padded: Sequence[Sequence[int]]) -> list[tuple[int, int, int]]:
    return [(j, l, b) for j, ls in enumerate(padded) for l, b in enumerate(ls)]


def section_row(padded: Sequence[Sequence[int]], j: int, l: int) -> int:
    """The first row of phase j's section at level l, from the padded block counts (the statement's `blocks`); (j, l) past
    the last section gives the design section's."""
    order = _order(padded)
    return header_rows(len(order)) + sum(_section_rows(b) for jj, ll, b in order if (jj, ll) < (j, l))


@dataclass(frozen=True)
class Advice:
    """The rows' content: C's SHA-512, its object's (`partition_object.digest`), the population N, and each phase's
    boundaries s_0 .. s_b at each level (b real blocks, no filler)."""
    program: str
    object: str
    population: int
    phases: tuple[tuple[tuple[int, ...], ...], ...]


def advice(program: Any, query: Mapping[str, Any], levels: Sequence[Sequence[tuple[int, int]]],
           starts: Sequence[int] = (0,)) -> Advice:
    """The advice of ``levels``, the object's `blocks`, cut into phases at the units ``starts`` names, each an end of a block
    at every level."""
    phases = []
    for lo, hi in zip(starts, (*starts[1:], None)):
        cut = [tuple(b for b in lv if b[0] >= lo and (hi is None or b[1] <= hi)) for lv in levels]
        if not all(cut) or any(c[0][0] != lo for c in cut):
            raise ValueError(f"the phase from unit {lo} is not whole blocks at every level")
        phases.append(tuple((c[0][0], *(h for _l, h in c)) for c in cut))
    obj = PO.build(program, query)
    return Advice(obj["program"], PO.digest(obj), PO.TemplatePartition(program, query).population, tuple(phases))


def _u(x: int) -> bytes:
    return int(x).to_bytes(8, "little")


def _row(b: bytes) -> bytes:
    if len(b) > ROW_BYTES:
        raise AssertionError("a row overflows")
    return b + bytes(ROW_BYTES - len(b))


def _chunks(b: bytes) -> list[bytes]:
    return [_row(b[i:i + ROW_BYTES]) for i in range(0, max(len(b), 1), ROW_BYTES)]


def rows(adv: Advice, bits: int = GRID_BITS) -> list[bytes]:
    """The advice's rows, padded to the grid."""
    padded = [[grid(len(s) - 1, bits) for s in ls] for ls in adv.phases]
    order = _order(padded)
    out = [_row(MAGIC + bytes.fromhex(adv.program) + b"".join(map(_u, (adv.population, len(order), section_row(padded, len(padded), 0), 0)))),
           _row(bytes.fromhex(adv.object))]
    ent = [b"".join(map(_u, (ENDS, j, l, section_row(padded, j, l), _section_rows(p), len(adv.phases[j][l]) - 1, p)))
           for j, l, p in order]
    out += [_row(b"".join(ent[k:k + SECTIONS_PER_ROW])) for k in range(0, len(ent), SECTIONS_PER_ROW)]
    for j, l, p in order:
        s = adv.phases[j][l]
        out += _chunks(b"".join(map(_u, s + (s[-1],) * (p + 1 - len(s)))))
    return out + [bytes(ROW_BYTES)] * (grid(len(out), bits) - len(out))


class Malformed(ValueError):
    """Rows that are not an encoding of any advice (`rows-malformed`, with where)."""


def _int(b: bytes, at: int) -> int:
    return int.from_bytes(b[at:at + 8], "little")


def _rest_zero(b: bytes, at: int, what: str) -> None:
    if any(b[at:]):
        raise Malformed(f"{what}: bytes past {at} are not zero")


def _bounds(read: Any, first: int, i: int) -> int:
    return _int(read(first + i // BOUNDS_PER_ROW), 8 * (i % BOUNDS_PER_ROW))


@dataclass(frozen=True)
class Section:
    """A section as the rows hold it: its phase and level, first row, rows, b real blocks and all its boundaries, filler's
    included."""
    phase: int
    level: int
    first: int
    rows: int
    blocks: int
    bounds: tuple[int, ...]


def decode(rs: Sequence[bytes]) -> tuple[Advice, list[Section], tuple[int, int]]:
    """The rows `rows` wrote: the advice, its sections, and the design section's (first row, rows); `Malformed` where they
    are no encoding."""
    def read(i: int) -> bytes:
        if not 0 <= i < len(rs) or not isinstance(rs[i], (bytes, bytearray)) or len(rs[i]) != ROW_BYTES:
            raise Malformed(f"row {i} is not there or not {ROW_BYTES} bytes")
        return bytes(rs[i])

    h, m = read(0), len(MAGIC)
    if h[:m] != MAGIC:
        raise Malformed("row 0: not a verity/partition/v2 header")
    n, ns, d0, dn = (_int(h, m + 64 + 8 * k) for k in range(4))
    _rest_zero(h, m + 96, "row 0")
    _rest_zero(read(1), 64, "row 1")
    if not 1 <= ns <= len(rs) * SECTIONS_PER_ROW:
        raise Malformed(f"row 0: {ns} sections")
    sections = []
    for k in range(ns):
        e = read(HEADER + k // SECTIONS_PER_ROW)
        kind, j, l, first, count, b, padded = (_int(e, ENTRY * (k % SECTIONS_PER_ROW) + 8 * x) for x in range(7))
        if k % SECTIONS_PER_ROW == SECTIONS_PER_ROW - 1 or k == ns - 1:
            _rest_zero(e, ENTRY * (k % SECTIONS_PER_ROW + 1), f"section {k}'s entry row")
        if kind != ENDS:
            raise Malformed(f"section {k}: kind {kind}")
        if not (b <= padded and count == _section_rows(padded) and first + count <= len(rs)):
            raise Malformed(f"section {k}: {b} blocks padded to {padded} in {count} rows from row {first}")
        body = b"".join(read(r) for r in range(first, first + count))
        _rest_zero(body, 8 * (padded + 1), f"section {k}")
        sections.append(Section(j, l, first, count, b, tuple(_int(body, 8 * i) for i in range(padded + 1))))
    if [(s.phase, s.level) for s in sections] != [(j, l) for j in range(sections[-1].phase + 1)
                                                  for l in range(sections[-1].level + 1)]:
        raise Malformed("the sections are not every phase at every level, in order")
    if any(any(read(r)) for r in range(d0 + dn, len(rs))):
        raise Malformed(f"rows past the {d0 + dn} used are not zero")
    nl = sections[-1].level + 1
    phases = tuple(tuple(s.bounds[:s.blocks + 1] for s in sections[j * nl:(j + 1) * nl]) for j in range(len(sections) // nl))
    return Advice(h[m:m + 64].hex(), read(1)[:64].hex(), n, phases), sections, (d0, dn)


def _levels(adv: Advice) -> list[list[tuple[int, int]]]:
    return [[(s[i], s[i + 1]) for ls in adv.phases for s in (ls[l],) for i in range(len(s) - 1)]
            for l in range(len(adv.phases[0]))]


def blocks(rs: Sequence[bytes]) -> list[list[tuple[int, int]]]:
    """The public form: the opened rows as the partition object's `blocks`, its levels, each the phases' real blocks side by
    side."""
    return _levels(decode(rs)[0])


def predicates(X: int, max_gates: int | None = None, max_in_bits: int | None = None, grid_bits: int = GRID_BITS) -> dict[str, Any]:
    return {"X": X, "max_gates": max_gates, "max_in_bits": max_in_bits, "grid_bits": grid_bits}


def statement(query: Mapping[str, Any], preds: Mapping[str, Any], adv: Advice, root: bytes | str) -> dict[str, Any]:
    """The public statement of `adv` registered under `root`, its counts rounded up to the grid."""
    g = preds["grid_bits"]
    return {"format": FORMAT, "program": adv.program, "query": dict(query), "predicates": dict(preds),
            "blocks": [[grid(len(s) - 1, g) for s in ls] for ls in adv.phases],
            "advice": {"rows": len(rows(adv, g)), "root": root.hex() if isinstance(root, bytes) else root}}


def _pos(x: Any) -> bool:
    return _is_int(x) and 0 < x <= (1 << 62)


def validate(st: Any) -> CutVerdict:
    """The statement's own form: exactly its keys; a SHA-512 program; a template query of valid form; predicates of exactly
    their keys and ranges; the same number of levels in each phase; each section's count and the row count on the grid; a
    SHA-512 root."""
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    if not isinstance(st, Mapping):
        return _verdict({"not-an-object": [type(st).__name__]})
    if set(st) != _KEYS:
        fail("keys", sorted(set(st) ^ _KEYS, key=repr))
    if st.get("format") != FORMAT:
        fail("format", st.get("format"))
    if not (isinstance(st.get("program"), str) and _HEX512.fullmatch(st["program"])):
        fail("program", st.get("program"))
    q = st.get("query")
    if not (isinstance(q, Mapping) and (q.get("name"), q.get("version")) in (PO.Q_TEMPLATE_INSTANCE, PO.Q_TEMPLATE_INSTANCES)
            and PO.validate({"format": PO.FORMAT, "program": "0" * 128, "query": dict(q)}).ok):
        fail("query", q)
    p, g = st.get("predicates"), None
    if not isinstance(p, Mapping) or set(p) != _PRED_KEYS:
        fail("predicate-keys", sorted(set(p) ^ _PRED_KEYS, key=repr) if isinstance(p, Mapping) else p)
    else:
        if not (_is_int(p["X"]) and 0 < p["X"] <= SIZE_CLASS_MAX):
            fail("predicates", {"X": p["X"]})
        for k in ("max_gates", "max_in_bits"):
            if p[k] is not None and not _pos(p[k]):
                fail("predicates", {k: p[k]})
        if _is_int(p["grid_bits"]) and 1 <= p["grid_bits"] <= 64:
            g = p["grid_bits"]
        else:
            fail("predicates", {"grid_bits": p["grid_bits"]})
    b = st.get("blocks")
    if not (isinstance(b, list) and b and all(isinstance(ls, list) and ls and len(ls) == len(b[0]) for ls in b)
            and all(_is_int(x) and x >= 1 and (g is None or grid(x, g) == x) for ls in b for x in ls)):
        fail("blocks", b)
    a = st.get("advice")
    if not (isinstance(a, Mapping) and set(a) == {"rows", "root"} and _is_int(a["rows"]) and a["rows"] >= 1
            and (g is None or grid(a["rows"], g) == a["rows"]) and isinstance(a["root"], str) and _HEX512.fullmatch(a["root"])):
        fail("advice", a)
    return _verdict(bad)


def digest(st: Mapping[str, Any]) -> str:
    """`SHA-512("verity/partition/v2\\0" || canon(statement))`, hex, of a valid statement."""
    v = validate(st)
    if not v.ok:
        raise ValueError(f"not a valid {FORMAT} statement: {v.codes}")
    return hashlib.sha512(FORMAT.encode() + b"\0" + canonical_json(st)).hexdigest()


def _verdict(bad: Mapping[str, list], detail: dict | None = None) -> CutVerdict:
    codes = tuple(k for k, v in bad.items() if v)
    d = dict(detail or {})
    for k in codes:
        d[k] = {"n": len(bad[k]), "gates": bad[k][:_MAX_LISTED]}
    return CutVerdict(not codes, codes, d)


def check(st: Mapping[str, Any], rs: Sequence[bytes], c: bytes) -> CutVerdict:
    """The global check (module docstring) of the rows ``rs`` under the statement ``st``, against C's bytes ``c``."""
    v = validate(st)
    if not v.ok:
        return v
    if len(rs) != st["advice"]["rows"]:
        return _verdict({"rows-count": [[len(rs), st["advice"]["rows"]]]})
    try:
        adv, sections, (d0, dn) = decode(rs)
    except Malformed as e:
        return _verdict({"rows-malformed": [str(e)]})
    bad: dict[str, list] = {}

    def fail(code: str, what: Any) -> None:
        bad.setdefault(code, []).append(what)

    D = R.program(c)
    if PO.program_sha512(D) != st["program"]:
        return _verdict({"program": [PO.program_sha512(D)]})
    obj = {"format": PO.FORMAT, "program": st["program"], "query": dict(st["query"])}
    if adv.program != st["program"]:
        fail("header-program", adv.program)
    if adv.object != PO.digest(obj):
        fail("header-object", adv.object)
    padded = st["blocks"]
    got = [(s.phase, s.level, s.first, len(s.bounds) - 1) for s in sections]
    want = [(j, l, section_row(padded, j, l), b) for j, l, b in _order(padded)]
    if got != want or d0 != section_row(padded, len(padded), 0):
        fail("header-layout", {"got": [list(x) for x in got], "want": [list(x) for x in want], "design": d0})
        return _verdict(bad)
    if dn:
        fail("design-rows", dn)
    for s in sections:
        if any(x != s.bounds[s.blocks] for x in s.bounds[s.blocks:]):
            fail("filler", [s.phase, s.level])
    for j, ls in enumerate(adv.phases):
        if len({(s[0], s[-1]) for s in ls}) != 1:
            fail("phase-span", [j, [[s[0], s[-1]] for s in ls]])
    try:
        pop = PO.TemplatePartition(D, st["query"]).population
    except PO.QueryInapplicable as e:
        fail("query-inapplicable", e.detail)
        return _verdict(bad)
    if adv.population != pop:
        fail("header-population", [adv.population, pop])
    levels = _levels(adv)
    verdict = PO.verify(PO.build(D, st["query"], levels), D)
    for code in verdict.codes:
        bad.setdefault(code, []).extend(verdict.detail[code]["gates"])
    for s in sections:
        if grid(s.blocks, st["predicates"]["grid_bits"]) != padded[s.phase][s.level]:
            fail("blocks-bucket", [s.phase, s.level, padded[s.phase][s.level], s.blocks])
    if not bad:
        p = st["predicates"]
        for k, (lo, hi) in enumerate(levels[-1]):
            words = R.unit_gates(D, st, [(lo, hi)])
            ins, outs = R.front(D, words)
            out_bits, in_bits = sum(gate_width(D, g) for g in outs), sum(gate_width(D, g) for g in ins)
            gates = sum(b - a for a, b in words)
            if out_bits > p["X"]:
                fail("block-too-wide", [k, out_bits])
            if (p["max_gates"] is not None and gates > p["max_gates"]) or (p["max_in_bits"] is not None and in_bits > p["max_in_bits"]):
                fail("block-too-large", [k, gates, in_bits])
    return _verdict(bad, {"population": adv.population, "blocks": [[len(s) - 1 for s in ls] for ls in adv.phases], "rows": len(rs)})


def openings(st: Mapping[str, Any], j: int, l: int, i: int) -> list[int]:
    """The rows sampled block ``i`` of phase ``j`` at level ``l`` opens: those holding s_i and s_(i+1)."""
    first = section_row(st["blocks"], j, l)
    return sorted({first + i // BOUNDS_PER_ROW, first + (i + 1) // BOUNDS_PER_ROW})


def check_block(c: bytes, st: Mapping[str, Any], j: int, l: int, i: int, opened: Mapping[int, bytes]) -> R.Block | None:
    """The local check of sampled block ``i`` of phase ``j`` at level ``l`` (module docstring) from the rows `openings` names
    alone, each opened against the statement's root (`verity.commitments.advice.verify`): None for a filler, else the block
    the door restricts, C_S of its range; `Malformed` where the rows are not those or give no range, `restrict.Refused`
    where C gives no block of it."""
    b = st["blocks"]
    if not (_is_int(j) and 0 <= j < len(b) and _is_int(l) and 0 <= l < len(b[j]) and _is_int(i) and 0 <= i < b[j][l]):
        raise Malformed(f"no block {i!r} of phase {j!r} at level {l!r}")
    want = openings(st, j, l, i)
    if sorted(opened) != want or any(len(opened[r]) != ROW_BYTES for r in want):
        raise Malformed(f"block {i} of phase {j} at level {l} opens rows {want}, not {sorted(opened)}")
    first = section_row(b, j, l)
    lo, hi = (_bounds(lambda r: bytes(opened[r]), first, x) for x in (i, i + 1))
    if lo == hi:
        return None
    if lo > hi:
        raise Malformed(f"block {i} of phase {j} at level {l}: [{lo}, {hi})")
    return R.restrict(c, st, [(lo, hi)])


__all__ = ["Advice", "BINDING", "BOUNDS_PER_ROW", "ENDS", "FORMAT", "GRID_BITS", "MAGIC", "Malformed", "ROW_BITS", "ROW_BYTES",
           "SECTIONS_PER_ROW", "SIZE_CLASSES", "SIZE_CLASS_MAX", "advice", "blocks", "check", "check_block", "decode", "digest",
           "grid", "id_digest", "openings", "predicates", "rows", "section_row", "size_class", "statement", "validate"]
