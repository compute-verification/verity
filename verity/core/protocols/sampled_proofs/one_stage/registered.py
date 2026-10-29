"""Registered values: a program's weights, constants and tables, committed once at registration under hiding leaves, and
every read of one at a public position opened against that commitment.

A *registered value* is a root parameter of the program, read row by row: row `r` is its `words` leaves from
`r * words`, in the parameter's leaf order, each leaf `word_bits` bits (16 or 32: the parameter's leaf width rounded up
to whole u16 words, `width`). Each row is committed as the circuit statement commits that port's row
(`hm96-sha512/row/v1`): its u16 words (`row_words`: each leaf `word_bits / 16` words, low first, zero-padded to whole
SHA-512 chunks of 64 words, as a Flock port row and its `values` row are laid out), `x = SHA-512(sha512_row_prefix(ROLE_X,
16, n) || row)` over those `n` words u16 LE, a 192-byte salt `y` drawn from the operating system once, at registration,
`b || c = hm96.SHA512.commit_string(default_key, x, y)`, and the frame-v3-sha512 leaf `row_leaf(D, r,
hm96.SHA512.tree_leaf(default_key, b || c), "hm96-sha512/row/v1")`. The value's root is the frame-v3-sha512 root over its
rows, under the domain `D` (`domain`): `CommitmentDomain(B, -2, RangeIndexedDomain(0, rows), hash="sha512")`, `B` the bytes
of `identity_digest("verity/one-stage/registered-domain/v0", {"program", "value", "schema", "words", "leaves"})`, with
`"word_bits"` beside them when it is not 16 (16 is the default, so a 16-bit value's domain, entry and root are what they
were before `word_bits`). The prover keeps the rows and salts; the registration names the root.

A *per-window value* (a window's token rows, say, which a window-independent Program registers under the same name every
window) adds `"window": R` to the binding, R the window's open root (lowercase hex of 64 bytes) that the challenger's
signed open names: its domain, and so every leaf of its tree, is that window's alone, so a root registered in one window
opens under no other's domain. A value the verifier holds across windows (the weights) leaves `window` out, as every
value did before it: the verifier names which (`registration.check`'s `held`, with its pinned, supplied and output
values), and its domain is the same in every window.

A value read by a bit-row port is committed in that port's row format instead (`schema`, one of `SCHEMAS`): under
`hm96-sha512/row/v2` a row is the bit row of its leaves (`bit_row((row, word_bits))`: its `row_words` with no padding,
little-endian), `x = sha512_row_v2_digest` over it, and its leaf `row_leaf(D, r, ..., "hm96-sha512/row/v2")`; under
`hm96-sha512/row-seg/v1` the same bit row under `sha512_row_seg_digest`, its leaf under `"hm96-sha512/row-seg/v1"`. The
domain is the same in every format; the leaf frame binds the format. The verifier's reads file names a value's format
under `row_schema` when it is not `hm96-sha512/row/v1` (`schema_entry`), and the Lean verifier checks each port in its
own format: a port's rows must be of the format its registration names, and a bit row exactly `words * word_bits` bits.
A bit row's leaves pack the parameter's narrower leaves (`row_leaves`): H1's A, K int8 codes of one-bit leaves, is a
`row/v2` value of `words = K/2` leaves of `word_bits = 16`, a row 8K of the parameter's leaves.

A read is at a *public position* when the verifier derives it from its own program (`reads`): a proof block's parameter whose
leaves are exactly row `r` of a registered value reads row `r`, for every proof block, before anything is served; a parameter
that is an array of whole rows (H1's tile reads R rows of A) reads each element's row at its own key `name[i]`, the
circuit's port of that row, and the Lean verifier holds each such port as its own registered port. A statement
that reads row `r` commits it with the registration's salt, so its `b || c` is the registered one, and opens it: header
`registered: {port: {"value", "positions", "paths"}}`, each of the port's table rows its position and its frame-v3 path
(the siblings, leaf level first, 64 bytes each, hex). The verifier checks the port's row width, each path to the
registered root and each instance's position against its own read map (`reads_file`, the Lean verifier's `--registered`,
all from its own program but the root); a statement that commits a registered row under fresh salts has no path and is
refused.

A *supplied value* is a registered value the verifier fills itself, from what it holds before the draw: each row is the
bytes of its sources in order, each source a root the record names (`{"port": P}`, the `roots` entry of port P;
`{"value": V}`, the `registered` entry of value V, never a supplied value's), bytes of its own (`{"bytes": hex}`), a
position (`{"position": "K"}`: K, a decimal below the record's population, after every proof block of the log's earlier
records, `log_base`, as a u64 big-endian, so no two records of one log share a position) or its coin bound to its
stream, its prover and the record's window (`{"window": hex}`: `window_salt`, SHA-512 of `WINDOW_SALT`, the coin, the
verifier's binding, `{"stream", "prover"}`, and the window's canonical JSON: a salt per prover, stream and window, as R6
takes one record a window and the window names the binding, so a tile seeded in one stream is no other's). A row's bytes
are read as the parameter's leaves of `w` bits
(`bytes_leaves`: each big-endian 32-bit half, its `32 / w` leaves least significant first, so a digest is 16 leaves of 32
bits or 512 of one). Its rows are
committed as any registered value's, under public salts (`supplied_salts`, row i's from the program's digest, the value's
name and i alone), so its root is a function of the record and the verifier's bytes: R7 recomputes it
(`supplied_root`) and refuses a record whose entry names another (`R7-supplied`). PoUW's seed inputs are such values: a
proof block's seed then reads the roots its operands are committed under, which the prover can't choose.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.ml.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2,
                                            bit_row, row_leaf, sha512_row_digest, sha512_row_seg_digest,
                                            sha512_row_v2_digest)
from verity.ml.circuits.codec import canonical_json, jcs_domain_error
from verity.ml.circuits.partition_object import TemplatePartition
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import runs as ref_runs
from verity.ml.circuits.types import Array

from . import partition as P

REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v0"
READS = "verity/registered-reads/v0"
#: the format whose port may carry `READS_FROM`: proof block `reads_from + i` reads `reads[i]`, every other proof block nothing; each
#: of its objects holds only its own keys (Flock's `Registered.FORMAT`, `FILE_KEYS`, `PORT_KEYS`)
READS_V1 = "verity/registered-reads/v1"
READS_FROM = "reads_from"
SCHEMA = SCHEMA_HM96_SHA512_ROW
SCHEMA_V2 = SCHEMA_HM96_SHA512_ROW_V2
SCHEMA_SEG = SCHEMA_HM96_SHA512_ROW_SEG
#: a registered value's row formats, each a Flock port's (`Flock.Registered.SCHEMAS`)
SCHEMAS = (SCHEMA, SCHEMA_V2, SCHEMA_SEG)
#: the reads-file key a value of another format than `SCHEMA` adds
SCHEMA_KEY = "row_schema"
#: the default leaf width, which the domain and the entry name only by leaving `word_bits` out
WORD_BITS = 16
WIDTHS = (16, 32)
#: u16 words per SHA-512 block: a statement's row port is whole blocks
BLOCK = 64
OWNER = -2
ENTRY_KEYS = ("value", "root", "leaves", "words", "domain")
#: the entry key a value of another width than `WORD_BITS` adds
WIDTH_KEY = "word_bits"
#: the reads-file key of a port not named after the template parameter it reads: that parameter's `reads` key, which the
#: Lean verifier holds its program's registered reads to (`Flock.Registered.PARAM_KEY`, `checkCover`)
PARAM_KEY = "param"
#: the domain binding's key a per-window value adds: the window's open root
WINDOW_KEY = "window"
#: the domain binding's key a produced value adds: the digest of the registration whose draw it follows (`produced`)
PRODUCED_KEY = "produced"
#: the domain binding's key a value committed inside its window's open root adds in the window's place: the session the
#: auditor's ledger opened for the window (PoUW's round-0 input-ids table, `credit.domain_of`)
SESSION_KEY = "session"
SUPPLIED_SALT = "verity/one-stage/supplied-salt/v0"
WINDOW_SALT = "verity/one-stage/window-salt/v1"
#: a supplied row's source kinds: a served port's root, a registered value's root, the verifier's own bytes, a position
#: after the log's, the verifier's coin bound to the record's window
SOURCES = ("port", "value", "bytes", "position", "window")
_INST = hm96.SHA512
_HEX = re.compile(r"[0-9a-f]+")
_DEC = re.compile(r"0|[1-9][0-9]*")


def _width(word_bits: int) -> int:
    if word_bits not in WIDTHS:
        raise ValueError(f"a registered value's leaves are {' or '.join(map(str, WIDTHS))} bits, not {word_bits!r}")
    return int(word_bits)


def width_entry(word_bits: int) -> dict[str, int]:
    """`{"word_bits": word_bits}`, or nothing for the default 16: what a domain binding and an entry add."""
    return {} if _width(word_bits) == WORD_BITS else {WIDTH_KEY: int(word_bits)}


def _schema(schema: str) -> str:
    if schema not in SCHEMAS:
        raise ValueError(f"a registered value's rows are {', '.join(SCHEMAS)} rows, not {schema!r}")
    return schema


def schema_entry(schema: str) -> dict[str, str]:
    """`{"row_schema": schema}`, or nothing for the default `hm96-sha512/row/v1`: what a reads-file port adds."""
    return {} if _schema(schema) == SCHEMA else {SCHEMA_KEY: schema}


def window_root(window: str | None) -> str | None:
    """The window's open root a per-window value's domain binds: None, or lowercase hex of 64 bytes; ValueError
    otherwise, which the Lean verifier refuses too (`OneStage.paramsOfJson`)."""
    if window is not None and not (isinstance(window, str) and len(window) == 128 and _HEX.fullmatch(window)):
        raise ValueError(f"a window root is lowercase hex of 64 bytes, not {window!r}")
    return window


def domain(program: str, value: str, words: int, leaves: int, word_bits: int = WORD_BITS,
           window: str | None = None, produced: str | None = None, session: str | None = None) -> CommitmentDomain:
    """`window`: the window's open root, for a per-window value (none for a value held across windows). `produced`: the
    registration's digest, for a value produced after its draw (`one_stage.produced`), which binds it in the window's
    place. `session`: the window's session, for a value whose root the window's open root contains, which so can't bind
    that root, in the window's place."""
    if sum(x is not None for x in (window, produced, session)) > 1:
        raise ValueError("a domain binds one of the window's open root, a registration and a session, not two")
    if session is not None and not (isinstance(session, str) and session):
        raise ValueError(f"a session is a nonempty string, not {session!r}")
    if produced is not None and not (isinstance(produced, str) and len(produced) == 128 and _HEX.fullmatch(produced)):
        raise ValueError(f"a registration digest is lowercase hex of 64 bytes, not {produced!r}")
    binding = identity_digest(REGISTERED_DOMAIN, {"program": program, "value": value, "schema": SCHEMA, "words": int(words),
                                                  "leaves": int(leaves), **width_entry(word_bits),
                                                  **({} if window_root(window) is None else {WINDOW_KEY: window}),
                                                  **({} if produced is None else {PRODUCED_KEY: produced}),
                                                  **({} if session is None else {SESSION_KEY: session})})
    return CommitmentDomain(bytes.fromhex(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def row_words(row: Sequence[int], word_bits: int = WORD_BITS, schema: str = SCHEMA) -> list[int]:
    """A registered row as the statement's port row: each leaf's `word_bits / 16` u16 words, low first, zero-padded to
    whole SHA-512 chunks (`BLOCK` words); a bit row's (`SCHEMA_V2`, `SCHEMA_SEG`) unpadded."""
    k, bound = _width(word_bits) // 16, 1 << word_bits
    out: list[int] = []
    for v in row:
        if not 0 <= int(v) < bound:
            raise ValueError(f"registered leaf {v} does not fit {word_bits} bits")
        out += [(int(v) >> (16 * i)) & 0xFFFF for i in range(k)]
    return out + [0] * (-len(out) % BLOCK if _schema(schema) == SCHEMA else 0)


def row_digest(row: Sequence[int], word_bits: int = WORD_BITS, schema: str = SCHEMA) -> bytes:
    """The row's `x`, as the statement hashes the port's row: `sha512/row/v1` over its u16 words (`row_words`), role x;
    under `SCHEMA_V2` or `SCHEMA_SEG`, `sha512/row/v2` or `sha512/row-seg/v1` over its leaves' bit row."""
    if _schema(schema) == SCHEMA:
        return sha512_row_digest(row_words(row, word_bits), 16, ROLE_X)
    data, n = bit_row((row_words(row, word_bits, schema), 16))
    return sha512_row_v2_digest(data, n) if schema == SCHEMA_V2 else sha512_row_seg_digest(data, n)


def width(program: Program, value: str) -> int:
    """The registered parameter `value`'s `word_bits`: its leaf width rounded up to whole u16 words, one width, at most
    32 bits."""
    t = dict(program.fn.signature()[0]).get(value)
    if t is None:
        raise ValueError(f"{value!r} is not a parameter of the program")
    w = t.homogeneous_width()
    if w is None or not 0 < w <= max(WIDTHS):
        raise ValueError(f"parameter {value!r}: its leaves are not one width of at most {max(WIDTHS)} bits "
                         f"({sorted(t.width_set())})")
    return 16 * -(-w // 16)


def parameter_gates(program: Program, value: str) -> range:
    """The root parameter `value`'s Input gates, which are one contiguous interval. Read run by run, never gate by gate: a
    weight tensor's parameter is ~1e9 gates."""
    names = [name for name, _ in program.fn.signature()[0]]
    if value not in names:
        raise ValueError(f"{value!r} is not a parameter of the program ({names})")
    first = end = None
    for _space, _idx, lo, hi in ref_runs(program.input_coll(names.index(value)).refs):
        if end is not None and lo != end:
            raise ValueError(f"parameter {value!r}: its gates are not one interval")
        first, end = lo if first is None else first, hi
    if first is None:
        raise ValueError(f"parameter {value!r}: its gates are not one interval")
    return range(first, end)


def row_leaves(program: Program, value: str, words: int, schema: str = SCHEMA) -> int:
    """The parameter's leaves in one registered row of `words` leaves: `words` under `SCHEMA`, each parameter leaf its own
    registered leaf; under a bit-row schema the row's `words * word_bits` bits, as many parameter leaves of the parameter's
    own width as they hold (H1's A: K int8 codes of one-bit leaves, K/2 u16 words)."""
    if _schema(schema) == SCHEMA or words < 1:
        return words
    word_bits = width(program, value)
    w = dict(program.fn.signature()[0])[value].homogeneous_width()
    if words * word_bits % w:
        raise ValueError(f"parameter {value!r}: a {schema} row of {words} {word_bits}-bit leaves is no whole number of its "
                         f"{w}-bit leaves")
    return words * word_bits // w


def shape(program: Program, value: str, words: int, schema: str = SCHEMA) -> int:
    """The rows of `value` read `words` leaves at a time, in `schema`'s rows (`row_leaves`)."""
    n, k = len(parameter_gates(program, value)), row_leaves(program, value, words, schema)
    if k < 1 or n % k:
        raise ValueError(f"parameter {value!r}: {n} leaves are not rows of {words}" +
                         (f" ({k} of its leaves in a {schema} row)" if k != words else ""))
    return n // k


def leaf(dom: CommitmentDomain, position: int, bc: bytes, schema: str = SCHEMA) -> bytes:
    return row_leaf(dom, position, _INST.tree_leaf(_INST.default_key, bc), _schema(schema))


@dataclass(frozen=True)
class Registered:
    """The prover's side of one registered value: its rows and salts (private), every row's `b || c`, the tree."""

    program: str
    value: str
    words: int
    word_bits: int
    rows: tuple[tuple[int, ...], ...]
    salts: tuple[bytes, ...]
    bcs: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]
    schema: str = SCHEMA
    window: str | None = None
    produced: str | None = None
    session: str | None = None

    @classmethod
    def commit(cls, program: str, value: str, rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None,
               word_bits: int = WORD_BITS, schema: str = SCHEMA, window: str | None = None,
               produced: str | None = None, session: str | None = None) -> "Registered":
        """`salts`: one per row, drawn once (default: fresh from the operating system) and never used for any other row.
        `word_bits`: every leaf's width (the verifier's `width` of the parameter). `schema`: the row format of the port
        that reads it (`SCHEMAS`). `window`: the window's open root, for a per-window value; `produced`: the registration's
        digest, for a value produced after its draw; `session`: the window's session, for a value its open root contains
        (`domain`)."""
        rows = tuple(tuple(int(w) for w in r) for r in rows)
        words = len(rows[0]) if rows else 0
        if not rows or not words or any(len(r) != words for r in rows):
            raise ValueError("a registered value is at least one row, every row the same width")
        salts = tuple(bytes(s) for s in salts) if salts is not None else tuple(_INST.fresh_salt() for _ in rows)
        if len(salts) != len(rows) or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"one {_INST.salt_bytes}-byte salt per row")
        if len(set(salts)) != len(salts):
            raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        bcs = tuple(_INST.commit_string(_INST.default_key, row_digest(r, word_bits, schema), s) for r, s in zip(rows, salts))
        dom = domain(program, value, words, len(rows), word_bits, window, produced, session)
        return cls(program, value, words, int(word_bits), rows, salts, bcs,
                   tree_levels(dom, [leaf(dom, i, bc, schema) for i, bc in enumerate(bcs)]), schema, window, produced,
                   session)

    @property
    def leaves(self) -> int:
        return len(self.rows)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def domain(self) -> CommitmentDomain:
        return domain(self.program, self.value, self.words, self.leaves, self.word_bits, self.window, self.produced,
                      self.session)

    def entry(self) -> dict[str, Any]:
        """What the registration names: `{"value", "root", "leaves", "words", "domain"}`, and `"word_bits"` when it is not
        16."""
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                **width_entry(self.word_bits), "domain": self.domain.domain_id.hex()}

    def path(self, position: int) -> bytes:
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out

    def statement(self, positions: Sequence[int]) -> tuple[list[tuple[int, ...]], list[bytes], dict[str, Any]]:
        """A statement's table of the rows at `positions`: the rows and their registered salts (the prover's), and the
        header entry `{"value", "positions", "paths"}`."""
        if any(not 0 <= p < self.leaves for p in positions):
            raise ValueError(f"a position is outside the value's {self.leaves} rows")
        return ([self.rows[p] for p in positions], [self.salts[p] for p in positions],
                {"value": self.value, "positions": [int(p) for p in positions], "paths": [self.path(p).hex() for p in positions]})


def table(reads: Sequence[int]) -> tuple[list[int], list[int]]:
    """Each instance's position as a shared-row table: the distinct positions in first-read order, and each instance's ref.
    The positions are public, so the layout reveals nothing the program does not."""
    order: dict[int, int] = {}
    refs = [order.setdefault(int(p), len(order)) for p in reads]
    return list(order), refs


def verify_read(dom: CommitmentDomain, root: bytes, position: int, bc: bytes, path: bytes, schema: str = SCHEMA) -> bool:
    """`bc`, as the registered value's row at `position`, opens `root` under the verifier's own domain `dom` through
    `path`, its leaf framed under `schema`: the check the Lean verifier makes for each table row of a registered port."""
    depth = merkle_depth(dom.count)
    if not 0 <= position < dom.count or len(bc) != _INST.proven_bytes or len(path) != 64 * depth:
        return False
    digest, i = leaf(dom, position, bc, schema), position
    for level in range(depth):
        sib = path[64 * level:64 * level + 64]
        digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
        i >>= 1
    return digest == root


def _row(hits: Sequence[tuple[str, int] | None], value: str, k: int) -> int | None:
    """The row of `value` that `hits` (each leaf's owner) is, whole and in order, `k` leaves to a row; None if it is not."""
    first = hits[0][1] if hits and hits[0] is not None and hits[0][0] == value else -1
    if first < 0 or first % k or len(hits) != k or list(hits) != [(value, first + j) for j in range(k)]:
        return None
    return first // k


def _source(template: str, parameter: str, element: int | None) -> str:
    return (f"parameter {parameter!r}" if element is None else f"row {element} of parameter {parameter!r}") + \
        f" of template {template}"


def reads(program: Program, query: Mapping[str, Any], values: Mapping[str, int],
          schemas: Mapping[str, str] | None = None) -> dict[str, dict[str, Any]]:
    """Every proof block's reads of the registered values, from the verifier's own program: `values` is `{parameter: words}`, and
    the result `{template parameter: {"value", "rows"}}`, `rows` one entry per proof block of the population (the row it reads,
    None for a proof block of another template). A proof block's parameter that holds a registered leaf must be exactly one row, read
    under its own name, or an array whose every element is one whole row of the value (H1's A: a tile's R rows), each
    element `i` read under `name[i]`, as the circuit's port of that row; and a template's parameter reads the same value
    in every proof block or in none: a read at a computed position is no public read. `schemas` (`{value: schema}`) gives a
    value's row format when it is not `SCHEMA`, which sets how many leaves a row holds (`row_leaves`). A key is one
    parameter's or one row's of it (in every template that has it): two reads under one key, a parameter `a[0]` beside
    row 0 of a parameter `a`, are refused, never merged. The proof blocks are a template query's instances; a `Q_word` or
    `Q_call` proof block reads no registered value (`ValueError`)."""
    tp = P.evaluate(program, query)
    if not isinstance(tp, TemplatePartition):
        raise ValueError("a registered value is read by a template query's instances, not by a Q_word or Q_call unit")
    per_row = {v: row_leaves(program, v, w, (schemas or {}).get(v, SCHEMA)) for v, w in values.items()}
    spans = []
    for v, w in values.items():
        shape(program, v, w, (schemas or {}).get(v, SCHEMA))
        spans.append((parameter_gates(program, v), v))

    def owner(gate: int) -> tuple[str, int] | None:
        """The registered value whose leaf `gate` is, and which leaf: parameters' gates are disjoint intervals."""
        return next(((v, gate - r.start) for r, v in spans if gate in r), None)

    template = {k: tid for k, tid, _a, _n in tp.ranges}
    nodes = program.root.body.nodes
    out: dict[str, dict[str, Any]] = {}
    value_of: dict[tuple[str, str], str | None] = {}
    source: dict[str, tuple[str, str, int | None]] = {}
    for u in range(tp.population):
        k, _m = tp.locate(u)
        gates, at = tp.inputs(u), 0
        for name, t in nodes[k].fn.signature()[0]:
            s, at = gates[at:at + t.leaves], at + t.leaves
            hits = [owner(g) for g in s]
            value, rows = None, {}
            if any(hits):
                value = hits[0][0] if hits[0] is not None else None
                w = per_row.get(value, 0)
                if value is not None and len(s) == w and (row := _row(hits, value, w)) is not None:
                    rows = {name: row}
                elif value is not None and isinstance(t, Array) and t.n > 1 and t.elem.leaves == w:
                    rows = {f"{name}[{i}]": _row(hits[i * w:(i + 1) * w], value, w) for i in range(t.n)}
                if not rows or None in rows.values():
                    raise ValueError(f"unit {u}: parameter {name!r} is not one whole row of a registered value, nor an "
                                     "array of whole rows of one")
            if value_of.setdefault((template[k], name), value) != value:
                raise ValueError(f"unit {u}: parameter {name!r} of {template[k]} reads {value!r}, and in another unit "
                                 f"{value_of[(template[k], name)]!r}")
            for i, (key, row) in enumerate(rows.items()):
                src = (name, i if key != name else None)
                if (was := source.setdefault(key, (template[k], *src)))[1:] != src:
                    raise ValueError(f"port {key!r} names two reads: {_source(*was)} and {_source(template[k], *src)}: "
                                     "a port reads one parameter's row")
                entry = out.setdefault(key, {"value": value, "rows": [None] * tp.population})
                if entry["value"] != value:
                    raise ValueError(f"parameter {key!r} reads {entry['value']!r} in one template and {value!r} in another")
                entry["rows"][u] = row
    return out


def registered_entries(program: Program, values: Mapping[str, int], schemas: Mapping[str, str] | None = None,
                       window: str | None = None, fixed: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """The verifier's own `{value: {"leaves", "words", "word_bits", "domain"}}` for the parameters it expects registered
    (`values`: `{parameter: words}`), from its own program (`word_bits`: `width`; `leaves`: `shape`, in the value's row
    format from `schemas`, default `SCHEMA`). `window`: the window's open root, which every value's domain binds but the
    ones in `fixed`, the values it holds across windows (`domain`)."""
    prog = P.program_digest(program)
    out = {}
    for v, w in values.items():
        n, b = shape(program, v, w, (schemas or {}).get(v, SCHEMA)), width(program, v)
        out[v] = {"leaves": n, "words": int(w), "word_bits": b,
                  "domain": domain(prog, v, w, n, b, None if v in fixed else window).domain_id.hex()}
    return out


def supplied_salts(program: str, value: str, n: int) -> list[bytes]:
    """Row i's public salt of a supplied value: SHAKE256 of `SUPPLIED_SALT`, the program's SHA-512 hex, the value's name and
    i (u32 little-endian), NUL between them, 192 bytes. They hide nothing the record doesn't show: the reads need only the
    commitment's binding."""
    head = SUPPLIED_SALT.encode() + b"\0" + program.encode() + b"\0" + value.encode() + b"\0"
    return [hashlib.shake_256(head + i.to_bytes(4, "little")).digest(_INST.salt_bytes) for i in range(n)]


def sources(supplied: Any) -> dict[str, list[list[tuple[str, str]]]]:
    """The verifier's supplied values, `{value: [[source, ...], ...]}` (a row a list of sources, each one of `SOURCES`
    naming a string), as `{value: [[(kind, name), ...], ...]}`; ValueError on any other form, which the Lean verifier
    refuses too (`OneStage.paramsOfJson`)."""
    if not isinstance(supplied, Mapping):
        raise ValueError(f"supplied values are an object of rows, not {supplied!r}")
    out: dict[str, list[list[tuple[str, str]]]] = {}
    for v, rows in supplied.items():
        if not (isinstance(v, str) and isinstance(rows, (list, tuple)) and all(isinstance(r, (list, tuple)) for r in rows)):
            raise ValueError(f"supplied value {v!r}: its rows are not lists of sources")
        out[v] = []
        for r in rows:
            if not all(isinstance(s, Mapping) and len(s) == 1 and set(s) <= set(SOURCES) and isinstance(*s.values(), str)
                       for s in r):
                raise ValueError(f"supplied value {v!r}: a source is not one of {SOURCES} naming a string: {r!r}")
            out[v].append([next(iter(s.items())) for s in r])
    return out


def bytes_leaves(data: bytes, bits: int) -> list[int]:
    """`data` as leaves of `bits` bits: each big-endian 32-bit half, its `32 / bits` leaves least significant first (a
    digest is 16 leaves of 32 bits, or 512 of one, as PoUW's `seeds.leaves` reads it)."""
    if not (0 < bits <= 32 and 32 % bits == 0):
        raise ValueError(f"leaves of {bits} bits hold no whole 32-bit halves")
    if len(data) % 4:
        raise ValueError(f"{len(data)} bytes are no whole 32-bit halves")
    mask = (1 << bits) - 1
    return [(h >> (bits * j)) & mask for i in range(0, len(data), 4) for h in (int.from_bytes(data[i:i + 4], "big"),)
            for j in range(32 // bits)]


def log_base(log: Sequence[Any]) -> int:
    """The first position after the log's records: the sum of their populations, every proof block of each one position."""
    base = 0
    for e in log:
        n = e.get("population") if isinstance(e, Mapping) else None
        if type(n) is not int or n < 0:
            raise ValueError(f"a record of the log has no population: {n!r}")
        base += n
    return base


def window_salt(coin: bytes, window: Any, binding: Mapping[str, str] | None) -> bytes:
    """The salt of `window` (a record's) under the verifier's coin and binding (`{"stream", "prover"}`, each lowercase
    hex): SHA-512 of `WINDOW_SALT`, the coin, the stream, the prover and the window's canonical JSON, NUL between them
    (only the coin, first, may hold a NUL). ValueError without a binding, or for a window that is no object of the
    canonical domain."""
    if binding is None:
        raise ValueError("a window salt is bound to a stream and a prover: the verifier holds no binding")
    if not isinstance(window, Mapping) or jcs_domain_error(dict(window)) is not None:
        raise ValueError(f"the record's window is no canonical JSON object: {window!r}")
    return hashlib.sha512(WINDOW_SALT.encode() + b"\0" + coin + b"\0" + binding["stream"].encode() + b"\0"
                          + binding["prover"].encode() + b"\0" + canonical_json(dict(window))).digest()


def _hex_bytes(value: str, kind: str, name: str, text: Any) -> bytes:
    if not (isinstance(text, str) and _HEX.fullmatch(text) and len(text) % 2 == 0):
        raise ValueError(f"supplied value {value!r}: {kind} {name!r} is not lowercase hex bytes")
    return bytes.fromhex(text)


def _source_bytes(rec: Any, value: str, kind: str, name: str, supplied: Sequence[str], log: Sequence[Any],
                  binding: Mapping[str, str] | None) -> bytes:
    """A source's bytes: the root the record names for it (its last entry, as R3 and R7 read a name twice), the verifier's
    own, a position after the log's, or the verifier's coin bound to its binding and the record's window; ValueError
    where there is none."""
    rec = rec if isinstance(rec, Mapping) else {}
    if kind == "bytes":
        return _hex_bytes(value, kind, name, name)
    if kind == "window":
        return window_salt(_hex_bytes(value, kind, name, name), rec.get("window"), binding)
    if kind == "position":
        n = rec.get("population")
        if not (_DEC.fullmatch(name) and type(n) is int and int(name) < n):
            raise ValueError(f"supplied value {value!r}: position {name!r} is no unit of a population of {n!r}")
        p = log_base(log) + int(name)
        if p >= 1 << 64:
            raise ValueError(f"supplied value {value!r}: position {p} is no u64")
        return p.to_bytes(8, "big")
    field, key = ("roots", "port") if kind == "port" else ("registered", "value")
    if kind == "value" and name in supplied:
        raise ValueError(f"supplied value {value!r} reads supplied value {name!r}'s root")
    es = rec.get(field)
    found = [e for e in (es if isinstance(es, list) else ()) if isinstance(e, Mapping) and e.get(key) == name]
    if not found:
        raise ValueError(f"supplied value {value!r}: the record names no {key} {name!r}")
    return _hex_bytes(value, kind, name, found[-1].get("root"))


def supplied_rows(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                  supplied: Sequence[str] = (), log: Sequence[Any] = (),
                  binding: Mapping[str, str] | None = None) -> list[list[int]]:
    """The rows the verifier fills a supplied value with, from `rec`, the log of this prover's earlier records, its
    binding (`window_salt`) and its own bytes: each row its sources' bytes, read as the parameter's leaves (`bytes_leaves`
    at its leaf width). `supplied`: every supplied value's name, whose roots no row reads. ValueError where a source names
    nothing the record or the verifier holds, or the leaves are not one width dividing 32."""
    t = dict(program.fn.signature()[0]).get(value)
    w = None if t is None else t.homogeneous_width()
    if w is None:
        raise ValueError(f"{value!r} is not a parameter of the program whose leaves are one width")
    return [bytes_leaves(b"".join(_source_bytes(rec, value, k, n, supplied, log, binding) for k, n in row), w)
            for row in rows]


def supplied_root(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                  want: Mapping[str, Any], supplied: Sequence[str] = (), log: Sequence[Any] = (),
                  prog: str | None = None, binding: Mapping[str, str] | None = None) -> str:
    """The root R7 holds a supplied value's entry to: its rows (`supplied_rows`), as many as the verifier's own entry's
    `leaves` (`registered_entries`), each of its `words` leaves, committed under `supplied_salts` at its `word_bits`.
    `prog`: `program`'s digest (`P.program_digest`), when the caller holds it."""
    got = supplied_rows(program, value, rows, rec, supplied, log, binding)
    if len(got) != want["leaves"] or any(len(r) != want["words"] for r in got):
        raise ValueError(f"supplied value {value!r}: rows of {[len(r) for r in got]} leaves, not {want['leaves']} of "
                         f"{want['words']}")
    prog = prog or P.program_digest(program)
    return Registered.commit(prog, value, got, supplied_salts(prog, value, len(got)), want["word_bits"]).root.hex()


def supplied_value(program: Program, value: str, rows: Sequence[Sequence[tuple[str, str]]], rec: Any,
                   supplied: Sequence[str] = (), log: Sequence[Any] = (), prog: str | None = None,
                   binding: Mapping[str, str] | None = None) -> Registered:
    """The prover's side of a supplied value: the rows the verifier fills from `rec` (a record without this value's own
    entry will do: no row reads it), the log and the binding under the public salts, whose `entry()` the record
    registers and whose `statement` opens a proof block's reads like any registered value's."""
    got = supplied_rows(program, value, rows, rec, supplied, log, binding)
    prog = prog or P.program_digest(program)
    return Registered.commit(prog, value, got, supplied_salts(prog, value, len(got)), width(program, value))


def reads_file(program: Program, query: Mapping[str, Any], values: Mapping[str, int], roots: Mapping[str, str],
               ports: Mapping[str, str] | None = None, domains: Mapping[str, str] | None = None,
               schemas: Mapping[str, str] | None = None, window: str | None = None,
               fixed: Sequence[str] = ()) -> dict[str, Any]:
    """The Lean verifier's `--registered` file, from the verifier's own program and query: per registered port, the value
    it reads, the value's leaf count, row width (`words` leaves of `word_bits`) and domain (`registered_entries`), each
    proof block's position (`reads`), and the value's root from `roots` (`{value: root}`, of a registration the verifier
    accepted). `ports` names a template parameter's circuit port when they differ, and such a port names its parameter
    (`PARAM_KEY`); two reads it names one port are refused,
    as the Lean verifier refuses a file naming a port twice. `domains` (`{value: domain}`) gives
    an output value's domain, its earlier record's port's, which the registration's R7 held its entry to
    (`registration.check`'s `outputs`). `schemas` (`{value: schema}`) gives a value's row format when it is not
    `SCHEMA` (`row_schema`, `schema_entry`): the format of the circuit port that reads it, which the Lean verifier holds
    the port to. `window` and `fixed`: the window's open root and the values held across windows, as R7 took them
    (`registered_entries`)."""
    mine = registered_entries(program, values, schemas, window, fixed)
    out, read_as = {}, {}
    for name, r in reads(program, query, values, schemas).items():
        root = roots.get(r["value"])
        if root is None:
            raise ValueError(f"parameter {name!r} reads {r['value']!r}, whose registered root the verifier does not hold")
        e = mine[r["value"]]
        port = (ports or {}).get(name, name)
        if port in out:
            raise ValueError(f"port {port!r} names two reads: {read_as[port]!r} of {out[port]['value']!r} and {name!r} of "
                             f"{r['value']!r}: each read is its own port")
        read_as[port] = name
        out[port] = {"value": r["value"], "root": root, "leaves": e["leaves"], "words": e["words"],
                     "word_bits": e["word_bits"], "domain": (domains or {}).get(r["value"], e["domain"]),
                     **schema_entry((schemas or {}).get(r["value"], SCHEMA)), "reads": list(r["rows"]),
                     **({PARAM_KEY: name} if port != name else {})}
    return {"format": READS, "ports": out}


__all__ = ["BLOCK", "ENTRY_KEYS", "OWNER", "PARAM_KEY", "PRODUCED_KEY", "READS", "REGISTERED_DOMAIN", "Registered", "SCHEMA", "SCHEMAS", "SCHEMA_KEY",
           "SCHEMA_SEG", "SCHEMA_V2", "SESSION_KEY", "SOURCES", "SUPPLIED_SALT", "WIDTHS", "WIDTH_KEY", "WINDOW_KEY", "WINDOW_SALT",
           "WORD_BITS",
           "bytes_leaves", "domain", "leaf", "log_base", "parameter_gates", "reads", "reads_file", "registered_entries",
           "row_digest", "row_leaves", "row_words", "schema_entry", "shape", "sources",
           "supplied_root", "supplied_rows", "supplied_salts", "supplied_value", "table", "verify_read", "width",
           "width_entry", "window_root", "window_salt"]
