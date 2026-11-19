"""`verity/partition/v2`: the one partition form, private. Its content is a partition object's `blocks` (`partition_object`):
ranges [lo, hi) of whole units of a template query, in order, covering them. Its envelope is advice: fixed-width rows
committed before any coin (hm96-sha512 hiding rows in one frame-v3-sha512 tree, `verity.commitments.advice`), of which the
verifier holds only the root and a statement naming the public predicates they meet. The public form is the same object
with the advice opened (`blocks`).

**Statement** (public; `statement`, `validate`, `digest` = `SHA-512("verity/partition/v2\\0" || canon(statement))`)::

    {"format": "verity/partition/v2", "program": "<SHA-512 of C, hex>", "query": <a template query>,
     "predicates": {"X": int, "max_gates": int | null, "max_in_bits": int | null, "grid_bits": g},
     "blocks": [grid(b_j) for each execution phase j], "advice": {"rows": grid(R), "root": "<frame-v3-sha512 root, hex>"}}

`b_j` and R are phase j's block count and the row count, rounded up to the public grid (`grid`: at most g significant bits),
so the verifier learns a grid level of each, never the counts.

**Rows** (private; `advice`, `rows`, `decode`). `ROW_BYTES` bytes, integers little-endian u64, every unused byte 0:

* row 0, the header: `MAGIC`, C's SHA-512, the population N, the phase count p, the design section's first row and rows;
* row 1: the SHA-512 of the partition object {format, program, query} (`partition_object.digest`);
* rows 2 .. `header_rows(p)` - 1: one entry a phase, `PHASES_PER_ROW` to a row: its section's kind, first row and rows,
  b_j and grid(b_j);
* each phase's section, in phase order: grid(b_j) + 1 boundaries s_0 .. s_grid(b_j), `BOUNDS_PER_ROW` to a row (kind
  `ENDS`): s_0 where the phase's units start (0, or the phase before's last end) and s_(i+1) block i's end, so block i is
  [s_i, s_(i+1)); blocks b_j .. grid(b_j) - 1 are filler, empty, each ending at s_(b_j);
* the design section: the sampling design's rows, none today;
* zero rows up to grid(R).

Each section's rows follow from the statement alone (`section_row`), so a sampled block's rows depend on its index and the
public counts, never on b_j or N. A kind names a section's record; `ENDS` is the only one, and a reader refuses any other.

**Global check** (`check`; refusal codes in brackets), over C and the rows, once per registration: the rows' count and form
[rows-count, rows-malformed]; the header names C, its object and its population [header-program, header-object,
header-population] and lays the sections out as the statement does [header-layout]; no design rows [design-rows]; the
blocks, the phases' real blocks side by side, are a valid `blocks` of the object covering C's units in order
(`partition_object.verify`'s codes); a filler is empty [filler]; each phase's count is on the grid [blocks-bucket]; each
block has at most X bits out and at most max_gates gates and max_in_bits bits in (`restrict.front`) [block-too-wide, block-too-large].

**Local check** (`check_block`), with sampled block i of phase j: the at most two rows holding s_i and s_(i+1) (`openings`),
opened against the root, give its range; a filler is empty, and any other is the block the door restricts.
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
PHASES_PER_ROW = ROW_BYTES // 40
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


def header_rows(phases: int) -> int:
    return HEADER + -(-phases // PHASES_PER_ROW)


def _section_rows(padded: int) -> int:
    return -(-(padded + 1) // BOUNDS_PER_ROW)


def section_row(padded: Sequence[int], j: int) -> int:
    """Phase j's first row, from the phases' padded block counts (the statement's `blocks`)."""
    return header_rows(len(padded)) + sum(_section_rows(b) for b in padded[:j])


@dataclass(frozen=True)
class Advice:
    """The rows' content: C's SHA-512, its object's (`partition_object.digest`), the population N, and each phase's
    boundaries s_0 .. s_b (b real blocks, no filler)."""
    program: str
    object: str
    population: int
    phases: tuple[tuple[int, ...], ...]


def advice(program: Any, query: Mapping[str, Any], phases: Sequence[Sequence[tuple[int, int]]]) -> Advice:
    """The advice of the template query's units grouped into ``phases``, each a list of [lo, hi) blocks, side by side."""
    if not all(phases):
        raise ValueError("a phase with no blocks")
    obj = PO.build(program, query)
    bounds = tuple((ps[0][0], *(hi for _lo, hi in ps)) for ps in phases)
    return Advice(obj["program"], PO.digest(obj), PO.TemplatePartition(program, query).population, bounds)


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
    padded = [grid(len(s) - 1, bits) for s in adv.phases]
    design = section_row(padded, len(padded))
    out = [_row(MAGIC + bytes.fromhex(adv.program) + b"".join(map(_u, (adv.population, len(padded), design, 0)))),
           _row(bytes.fromhex(adv.object))]
    out += _chunks(b"".join(b"".join(map(_u, (ENDS, section_row(padded, j), _section_rows(p), len(s) - 1, p)))
                            for j, (s, p) in enumerate(zip(adv.phases, padded))))
    for s, p in zip(adv.phases, padded):
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
    """A phase's section as the rows hold it: its first row, its rows, its b real blocks and all its boundaries, filler's
    included."""
    first: int
    rows: int
    blocks: int
    bounds: tuple[int, ...]


def decode(rs: Sequence[bytes]) -> tuple[Advice, list[Section], tuple[int, int]]:
    """The rows `rows` wrote: the advice, each phase's section, and the design section's (first row, rows); `Malformed`
    where they are no encoding."""
    def read(i: int) -> bytes:
        if not 0 <= i < len(rs) or not isinstance(rs[i], (bytes, bytearray)) or len(rs[i]) != ROW_BYTES:
            raise Malformed(f"row {i} is not there or not {ROW_BYTES} bytes")
        return bytes(rs[i])

    h, m = read(0), len(MAGIC)
    if h[:m] != MAGIC:
        raise Malformed("row 0: not a verity/partition/v2 header")
    n, p, d0, dn = (_int(h, m + 64 + 8 * k) for k in range(4))
    _rest_zero(h, m + 96, "row 0")
    _rest_zero(read(1), 64, "row 1")
    if not 1 <= p <= len(rs) * PHASES_PER_ROW:
        raise Malformed(f"row 0: {p} phases")
    table = b"".join(read(r) for r in range(HEADER, header_rows(p)))
    _rest_zero(table, 40 * p, "the phase table")
    sections = []
    for j in range(p):
        kind, first, count, b, padded = (_int(table, 40 * j + 8 * k) for k in range(5))
        if kind != ENDS:
            raise Malformed(f"phase {j}: section kind {kind}")
        if not (b <= padded and count == _section_rows(padded) and first + count <= len(rs)):
            raise Malformed(f"phase {j}: {b} blocks padded to {padded} in {count} rows from row {first}")
        body = b"".join(read(r) for r in range(first, first + count))
        _rest_zero(body, 8 * (padded + 1), f"phase {j}'s section")
        sections.append(Section(first, count, b, tuple(_int(body, 8 * i) for i in range(padded + 1))))
    if any(any(read(r)) for r in range(d0 + dn, len(rs))):
        raise Malformed(f"rows past the {d0 + dn} used are not zero")
    adv = Advice(h[m:m + 64].hex(), read(1)[:64].hex(), n, tuple(s.bounds[:s.blocks + 1] for s in sections))
    return adv, sections, (d0, dn)


def blocks(rs: Sequence[bytes]) -> list[tuple[int, int]]:
    """The public form: the opened rows as the partition object's `blocks`, the phases' real blocks side by side."""
    return [(s[i], s[i + 1]) for s in decode(rs)[0].phases for i in range(len(s) - 1)]


def predicates(X: int, max_gates: int | None = None, max_in_bits: int | None = None, grid_bits: int = GRID_BITS) -> dict[str, Any]:
    return {"X": X, "max_gates": max_gates, "max_in_bits": max_in_bits, "grid_bits": grid_bits}


def statement(query: Mapping[str, Any], preds: Mapping[str, Any], adv: Advice, root: bytes | str) -> dict[str, Any]:
    """The public statement of `adv` registered under `root`, its counts rounded up to the grid."""
    g = preds["grid_bits"]
    return {"format": FORMAT, "program": adv.program, "query": dict(query), "predicates": dict(preds),
            "blocks": [grid(len(s) - 1, g) for s in adv.phases], "advice": {"rows": len(rows(adv, g)),
                                                                             "root": root.hex() if isinstance(root, bytes) else root}}


def _pos(x: Any) -> bool:
    return _is_int(x) and 0 < x <= (1 << 62)


def validate(st: Any) -> CutVerdict:
    """The statement's own form: exactly its keys; a SHA-512 program; a template query of valid form; predicates of exactly
    their keys and ranges; each phase's count and the row count on the grid; a SHA-512 root."""
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
    if not (isinstance(b, list) and b and all(_is_int(x) and x >= 1 and (g is None or grid(x, g) == x) for x in b)):
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
    got = [(s.first, len(s.bounds) - 1) for s in sections]
    want = [(section_row(padded, j), b) for j, b in enumerate(padded)]
    if got != want or d0 != section_row(padded, len(padded)):
        fail("header-layout", {"got": [list(x) for x in got], "want": [list(x) for x in want], "design": d0})
    if dn:
        fail("design-rows", dn)
    for j, s in enumerate(sections):
        if any(x != s.bounds[s.blocks] for x in s.bounds[s.blocks:]):
            fail("filler", j)
    pairs = [[s[i], s[i + 1]] for s in adv.phases for i in range(len(s) - 1)]
    try:
        pop = PO.TemplatePartition(D, st["query"]).population
    except PO.QueryInapplicable as e:
        fail("query-inapplicable", e.detail)
        return _verdict(bad)
    if adv.population != pop:
        fail("header-population", [adv.population, pop])
    verdict = PO.verify({**obj, PO.BLOCKS: pairs}, D)
    for code in verdict.codes:
        bad.setdefault(code, []).extend(verdict.detail[code]["gates"])
    for j, (s, b) in enumerate(zip(adv.phases, padded)):
        if grid(len(s) - 1, st["predicates"]["grid_bits"]) != b:
            fail("blocks-bucket", [j, b, len(s) - 1])
    if not bad:
        p = st["predicates"]
        for k, (lo, hi) in enumerate(pairs):
            words = R.unit_gates(D, st, [(lo, hi)])
            ins, outs = R.front(D, words)
            out_bits, in_bits = sum(gate_width(D, g) for g in outs), sum(gate_width(D, g) for g in ins)
            gates = sum(b - a for a, b in words)
            if out_bits > p["X"]:
                fail("block-too-wide", [k, out_bits])
            if (p["max_gates"] is not None and gates > p["max_gates"]) or (p["max_in_bits"] is not None and in_bits > p["max_in_bits"]):
                fail("block-too-large", [k, gates, in_bits])
    return _verdict(bad, {"population": adv.population, "blocks": [len(s) - 1 for s in adv.phases], "rows": len(rs)})


def openings(st: Mapping[str, Any], j: int, i: int) -> list[int]:
    """The rows sampled block ``i`` of phase ``j`` opens: those holding s_i and s_(i+1)."""
    first = section_row(st["blocks"], j)
    return sorted({first + i // BOUNDS_PER_ROW, first + (i + 1) // BOUNDS_PER_ROW})


def check_block(c: bytes, st: Mapping[str, Any], j: int, i: int, opened: Mapping[int, bytes]) -> R.Block | None:
    """The local check of sampled block ``i`` of phase ``j`` (module docstring) from the rows `openings` names alone, each
    opened against the statement's root (`verity.commitments.advice.verify`): None for a filler, else the block the door
    restricts, C_S of its range; `Malformed` where the rows are not those or give no range, `restrict.Refused` where C
    gives no block of it."""
    if not (_is_int(j) and 0 <= j < len(st["blocks"]) and _is_int(i) and 0 <= i < st["blocks"][j]):
        raise Malformed(f"no block {i!r} of phase {j!r}")
    want = openings(st, j, i)
    if sorted(opened) != want or any(len(opened[r]) != ROW_BYTES for r in want):
        raise Malformed(f"block {i} of phase {j} opens rows {want}, not {sorted(opened)}")
    first = section_row(st["blocks"], j)
    lo, hi = (_bounds(lambda r: bytes(opened[r]), first, x) for x in (i, i + 1))
    if lo == hi:
        return None
    if lo > hi:
        raise Malformed(f"block {i} of phase {j}: [{lo}, {hi})")
    return R.restrict(c, st, [(lo, hi)])


__all__ = ["Advice", "BINDING", "BOUNDS_PER_ROW", "ENDS", "FORMAT", "GRID_BITS", "MAGIC", "Malformed", "ROW_BITS", "ROW_BYTES",
           "SIZE_CLASSES", "SIZE_CLASS_MAX", "advice", "blocks", "check", "check_block", "decode", "digest", "grid", "id_digest",
           "openings", "predicates", "rows", "section_row", "size_class", "statement", "validate"]
