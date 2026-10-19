"""Registered values: a program's weights, constants and tables, committed once at registration under hiding leaves, and
every read of one at a public position opened against that commitment.

A *registered value* is a root parameter of the program, read row by row: row `r` is its `words` leaves from
`r * words`, in the parameter's leaf order, each leaf `word_bits` bits (16 or 32: the parameter's leaf width rounded up
to whole u16 words, `width`). Each row is committed as the circuit statement commits that port's row
(`hm96-sha512/row/v1`): its u16 words (`row_words`: each leaf `word_bits / 16` words, low first, zero-padded to whole
SHA-512 blocks of 64 words, as a C-Flock port row and its `values` row are laid out), `x = SHA-512(sha512_row_prefix(ROLE_X,
16, n) || row)` over those `n` words u16 LE, a 192-byte salt `y` drawn from the operating system once, at registration,
`b || c = hm96.SHA512.commit_string(default_key, x, y)`, and the frame-v3-sha512 leaf `row_leaf(D, r,
hm96.SHA512.tree_leaf(default_key, b || c), "hm96-sha512/row/v1")`. The value's root is the frame-v3-sha512 root over its
rows, under the domain `D` (`domain`): `CommitmentDomain(B, -2, RangeIndexedDomain(0, rows), hash="sha512")`, `B` the bytes
of `identity_digest("verity/one-stage/registered-domain/v0", {"program", "value", "schema", "words", "leaves"})`, with
`"word_bits"` beside them when it is not 16 (16 is the default, so a 16-bit value's domain, entry and root are what they
were before `word_bits`). The prover keeps the rows and salts; the registration names the root.

A read is at a *public position* when the verifier derives it from its own program (`reads`): a unit's parameter whose
leaves are exactly row `r` of a registered value reads row `r`, for every unit, before anything is served. A statement
that reads row `r` commits it with the registration's salt, so its `b || c` is the registered one, and opens it: header
`registered: {port: {"value", "positions", "paths"}}`, each of the port's table rows its position and its frame-v3 path
(the siblings, leaf level first, 64 bytes each, hex). The verifier checks the port's row width, each path to the
registered root and each instance's position against its own read map (`reads_file`, the Lean verifier's `--registered`,
all from its own program but the root); a statement that commits a registered row under fresh salts has no path and is
refused.

A *supplied value* is a registered value the verifier fills itself, from what it holds before the draw: each row is the
bytes of its sources in order, each source a root the record names (`{"port": P}`, the `roots` entry of port P;
`{"value": V}`, the `registered` entry of value V, never a supplied value's), bytes of its own (`{"bytes": hex}`), a
position (`{"position": "K"}`: K, a decimal below the record's population, after every unit of the log's earlier
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
unit's seed then reads the roots its operands are committed under, which the prover can't choose.
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
from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, row_leaf, sha512_row_digest
from verity.ml.circuits.codec import canonical_json, jcs_domain_error
from verity.ml.circuits.partition_object import TemplatePartition
from verity.ml.circuits.program import Program

from . import partition as P

REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v0"
READS = "verity/registered-reads/v0"
SCHEMA = SCHEMA_HM96_SHA512_ROW
#: the default leaf width, which the domain and the entry name only by leaving `word_bits` out
WORD_BITS = 16
WIDTHS = (16, 32)
#: u16 words per SHA-512 block: a statement's row port is whole blocks
BLOCK = 64
OWNER = -2
ENTRY_KEYS = ("value", "root", "leaves", "words", "domain")
#: the entry key a value of another width than `WORD_BITS` adds
WIDTH_KEY = "word_bits"
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


def domain(program: str, value: str, words: int, leaves: int, word_bits: int = WORD_BITS) -> CommitmentDomain:
    binding = identity_digest(REGISTERED_DOMAIN, {"program": program, "value": value, "schema": SCHEMA, "words": int(words),
                                                  "leaves": int(leaves), **width_entry(word_bits)})
    return CommitmentDomain(bytes.fromhex(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


def row_words(row: Sequence[int], word_bits: int = WORD_BITS) -> list[int]:
    """A registered row as the statement's port row: each leaf's `word_bits / 16` u16 words, low first, zero-padded to
    whole SHA-512 blocks (`BLOCK` words)."""
    k, bound = _width(word_bits) // 16, 1 << word_bits
    out: list[int] = []
    for v in row:
        if not 0 <= int(v) < bound:
            raise ValueError(f"registered leaf {v} does not fit {word_bits} bits")
        out += [(int(v) >> (16 * i)) & 0xFFFF for i in range(k)]
    return out + [0] * (-len(out) % BLOCK)


def row_digest(row: Sequence[int], word_bits: int = WORD_BITS) -> bytes:
    """The row's `x`: `sha512/row/v1` over its u16 words (`row_words`), role x, as the statement hashes the port's row."""
    return sha512_row_digest(row_words(row, word_bits), 16, ROLE_X)


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
    """The root parameter `value`'s Input gates, which are one contiguous interval."""
    names = [name for name, _ in program.fn.signature()[0]]
    if value not in names:
        raise ValueError(f"{value!r} is not a parameter of the program ({names})")
    gates = [ref[2] for ref in program.input_coll(names.index(value)).refs]
    if not gates or gates != list(range(gates[0], gates[0] + len(gates))):
        raise ValueError(f"parameter {value!r}: its gates are not one interval")
    return range(gates[0], gates[0] + len(gates))


def shape(program: Program, value: str, words: int) -> int:
    """The rows of `value` read `words` leaves at a time."""
    n = len(parameter_gates(program, value))
    if words < 1 or n % words:
        raise ValueError(f"parameter {value!r}: {n} leaves are not rows of {words}")
    return n // words


def leaf(dom: CommitmentDomain, position: int, bc: bytes) -> bytes:
    return row_leaf(dom, position, _INST.tree_leaf(_INST.default_key, bc), SCHEMA)


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

    @classmethod
    def commit(cls, program: str, value: str, rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None,
               word_bits: int = WORD_BITS) -> "Registered":
        """`salts`: one per row, drawn once (default: fresh from the operating system) and never used for any other row.
        `word_bits`: every leaf's width (the verifier's `width` of the parameter)."""
        rows = tuple(tuple(int(w) for w in r) for r in rows)
        words = len(rows[0]) if rows else 0
        if not rows or not words or any(len(r) != words for r in rows):
            raise ValueError("a registered value is at least one row, every row the same width")
        salts = tuple(bytes(s) for s in salts) if salts is not None else tuple(_INST.fresh_salt() for _ in rows)
        if len(salts) != len(rows) or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"one {_INST.salt_bytes}-byte salt per row")
        if len(set(salts)) != len(salts):
            raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        bcs = tuple(_INST.commit_string(_INST.default_key, row_digest(r, word_bits), s) for r, s in zip(rows, salts))
        dom = domain(program, value, words, len(rows), word_bits)
        return cls(program, value, words, int(word_bits), rows, salts, bcs,
                   tree_levels(dom, [leaf(dom, i, bc) for i, bc in enumerate(bcs)]))

    @property
    def leaves(self) -> int:
        return len(self.rows)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def domain(self) -> CommitmentDomain:
        return domain(self.program, self.value, self.words, self.leaves, self.word_bits)

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


def verify_read(dom: CommitmentDomain, root: bytes, position: int, bc: bytes, path: bytes) -> bool:
    """`bc`, as the registered value's row at `position`, opens `root` under the verifier's own domain `dom` through
    `path`: the check the Lean verifier makes for each table row of a registered port."""
    depth = merkle_depth(dom.count)
    if not 0 <= position < dom.count or len(bc) != _INST.proven_bytes or len(path) != 64 * depth:
        return False
    digest, i = leaf(dom, position, bc), position
    for level in range(depth):
        sib = path[64 * level:64 * level + 64]
        digest = dom.node(level, i >> 1, *((digest, sib) if i % 2 == 0 else (sib, digest)))
        i >>= 1
    return digest == root


def reads(program: Program, query: Mapping[str, Any], values: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """Every unit's reads of the registered values, from the verifier's own program: `values` is `{parameter: words}`, and
    the result `{template parameter: {"value", "rows"}}`, `rows` one entry per unit of the population (the row it reads,
    None for a unit of another template). A unit's parameter that holds a registered leaf must be exactly one row, and a
    template's parameter reads the same value in every unit or in none: a read at a computed position is no public read.
    The units are a template query's instances; a `Q_word` or `Q_call` unit reads no registered value (`ValueError`)."""
    tp = P.evaluate(program, query)
    if not isinstance(tp, TemplatePartition):
        raise ValueError("a registered value is read by a template query's instances, not by a Q_word or Q_call unit")
    owner: dict[int, tuple[str, int]] = {}
    for v, w in values.items():
        shape(program, v, w)
        for k, gate in enumerate(parameter_gates(program, v)):
            owner[gate] = (v, k)
    template = {k: tid for k, tid, _a, _n in tp.ranges}
    nodes = program.root.body.nodes
    out: dict[str, dict[str, Any]] = {}
    value_of: dict[tuple[str, str], str | None] = {}
    for u in range(tp.population):
        k, _m = tp.locate(u)
        gates, at = tp.inputs(u), 0
        for name, t in nodes[k].fn.signature()[0]:
            s, at = gates[at:at + t.leaves], at + t.leaves
            hits = [owner.get(g) for g in s]
            value, row = None, None
            if any(hits):
                value, first = hits[0] or (None, -1)
                w = values.get(value, 0)
                if value is None or first % w or len(s) != w or hits != [(value, first + j) for j in range(w)]:
                    raise ValueError(f"unit {u}: parameter {name!r} is not one whole row of a registered value")
                row = first // w
            if value_of.setdefault((template[k], name), value) != value:
                raise ValueError(f"unit {u}: parameter {name!r} of {template[k]} reads {value!r}, and in another unit "
                                 f"{value_of[(template[k], name)]!r}")
            if value is None:
                continue
            entry = out.setdefault(name, {"value": value, "rows": [None] * tp.population})
            if entry["value"] != value:
                raise ValueError(f"parameter {name!r} reads {entry['value']!r} in one template and {value!r} in another")
            entry["rows"][u] = row
    return out


def registered_entries(program: Program, values: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    """The verifier's own `{value: {"leaves", "words", "word_bits", "domain"}}` for the parameters it expects registered
    (`values`: `{parameter: words}`), from its own program (`word_bits`: `width`)."""
    prog = P.program_digest(program)
    out = {}
    for v, w in values.items():
        n, b = shape(program, v, w), width(program, v)
        out[v] = {"leaves": n, "words": int(w), "word_bits": b, "domain": domain(prog, v, w, n, b).domain_id.hex()}
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
    """The first position after the log's records: the sum of their populations, every unit of each one position."""
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
    registers and whose `statement` opens a unit's reads like any registered value's."""
    got = supplied_rows(program, value, rows, rec, supplied, log, binding)
    prog = prog or P.program_digest(program)
    return Registered.commit(prog, value, got, supplied_salts(prog, value, len(got)), width(program, value))


def reads_file(program: Program, query: Mapping[str, Any], values: Mapping[str, int], roots: Mapping[str, str],
               ports: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The Lean verifier's `--registered` file, from the verifier's own program and query: per registered port, the value
    it reads, the value's leaf count, row width (`words` leaves of `word_bits`) and domain (`registered_entries`), each
    unit's position (`reads`), and the value's root from `roots` (`{value: root}`, of a registration the verifier
    accepted). `ports` names a template parameter's circuit port when they differ."""
    mine = registered_entries(program, values)
    out = {}
    for name, r in reads(program, query, values).items():
        root = roots.get(r["value"])
        if root is None:
            raise ValueError(f"parameter {name!r} reads {r['value']!r}, whose registered root the verifier does not hold")
        e = mine[r["value"]]
        out[(ports or {}).get(name, name)] = {"value": r["value"], "root": root, "leaves": e["leaves"], "words": e["words"],
                                              "word_bits": e["word_bits"], "domain": e["domain"], "reads": list(r["rows"])}
    return {"format": READS, "ports": out}


__all__ = ["BLOCK", "ENTRY_KEYS", "OWNER", "READS", "REGISTERED_DOMAIN", "Registered", "SCHEMA", "SOURCES", "SUPPLIED_SALT",
           "WIDTHS", "WIDTH_KEY", "WINDOW_SALT", "WORD_BITS", "bytes_leaves", "domain", "leaf", "log_base",
           "parameter_gates", "reads", "reads_file", "registered_entries", "row_digest", "row_words", "shape", "sources",
           "supplied_root", "supplied_rows", "supplied_salts", "supplied_value", "table", "verify_read", "width",
           "width_entry", "window_salt"]
