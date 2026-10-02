"""Registered values: a program's weights, constants and tables, committed once at registration under hiding leaves, and
every read of one at a public position opened against that commitment.

A *registered value* is a root parameter of the program, read row by row: row `r` is its `words` leaves from
`r * words`, in the parameter's leaf order. Each row is committed as the circuit statement commits an input row
(`hm96-sha512/row/v1`): `x = SHA-512(sha512_row_prefix(ROLE_X, 16, words) || row)` with the row's words u16 LE, a 192-byte
salt `y` drawn from the operating system once, at registration, `b || c = hm96.SHA512.commit_string(default_key, x, y)`,
and the frame-v3-sha512 leaf `row_leaf(D, r, hm96.SHA512.tree_leaf(default_key, b || c), "hm96-sha512/row/v1")`. The
value's root is the frame-v3-sha512 root over its rows, under the domain `D` (`domain`): `CommitmentDomain(B, -2,
RangeIndexedDomain(0, rows), hash="sha512")`, `B` the bytes of `identity_digest("verity/one-stage/registered-domain/v0",
{"program", "value", "schema", "words", "leaves"})`. The prover keeps the rows and salts; the registration names the root.

A read is at a *public position* when the verifier derives it from its own program (`reads`): a unit's parameter whose
leaves are exactly row `r` of a registered value reads row `r`, for every unit, before anything is served. A statement
that reads row `r` commits it with the registration's salt, so its `b || c` is the registered one, and opens it: header
`registered: {port: {"value", "positions", "paths"}}`, each of the port's table rows its position and its frame-v3 path
(the siblings, leaf level first, 64 bytes each, hex). The verifier checks each path to the registered root and each
instance's position against its own read map (`reads_file`, the Lean verifier's `--registered`); a statement that
commits a registered row under fresh salts has no path and is refused.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from verity.commitments import hm96
from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain, merkle_depth, tree_levels
from verity.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, row_leaf, sha512_row_digest
from verity.ir.program import Program

from . import partition as P

REGISTERED_DOMAIN = "verity/one-stage/registered-domain/v0"
READS = "verity/registered-reads/v0"
SCHEMA = SCHEMA_HM96_SHA512_ROW
WORD_BITS = 16
OWNER = -2
ENTRY_KEYS = ("value", "root", "leaves", "words", "domain")
_INST = hm96.SHA512


def domain(program: str, value: str, words: int, leaves: int) -> CommitmentDomain:
    binding = identity_digest(REGISTERED_DOMAIN, {"program": program, "value": value, "schema": SCHEMA, "words": int(words),
                                                  "leaves": int(leaves)})
    return CommitmentDomain(bytes.fromhex(binding), OWNER, RangeIndexedDomain(0, int(leaves)), hash="sha512")


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
    rows: tuple[tuple[int, ...], ...]
    salts: tuple[bytes, ...]
    bcs: tuple[bytes, ...]
    levels: tuple[tuple[bytes, ...], ...]

    @classmethod
    def commit(cls, program: str, value: str, rows: Sequence[Sequence[int]], salts: Sequence[bytes] | None = None) -> "Registered":
        """`salts`: one per row, drawn once (default: fresh from the operating system) and never used for any other row."""
        rows = tuple(tuple(int(w) for w in r) for r in rows)
        words = len(rows[0]) if rows else 0
        if not rows or not words or any(len(r) != words for r in rows):
            raise ValueError("a registered value is at least one row, every row the same width")
        salts = tuple(bytes(s) for s in salts) if salts is not None else tuple(_INST.fresh_salt() for _ in rows)
        if len(salts) != len(rows) or any(len(s) != _INST.salt_bytes for s in salts):
            raise ValueError(f"one {_INST.salt_bytes}-byte salt per row")
        if len(set(salts)) != len(salts):
            raise ValueError("a salt is never reused: equal salts on two rows reveal the XOR of their digests")
        bcs = tuple(_INST.commit_string(_INST.default_key, sha512_row_digest(r, WORD_BITS, ROLE_X), s) for r, s in zip(rows, salts))
        dom = domain(program, value, words, len(rows))
        return cls(program, value, words, rows, salts, bcs, tree_levels(dom, [leaf(dom, i, bc) for i, bc in enumerate(bcs)]))

    @property
    def leaves(self) -> int:
        return len(self.rows)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def domain(self) -> CommitmentDomain:
        return domain(self.program, self.value, self.words, self.leaves)

    def entry(self) -> dict[str, Any]:
        """What the registration names: `{"value", "root", "leaves", "words", "domain"}`."""
        return {"value": self.value, "root": self.root.hex(), "leaves": self.leaves, "words": self.words,
                "domain": self.domain.domain_id.hex()}

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
    template's parameter reads the same value in every unit or in none: a read at a computed position is no public read."""
    tp = P.evaluate(program, query)
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


def reads_file(entries: Sequence[Mapping[str, Any]], read_map: Mapping[str, Mapping[str, Any]],
               ports: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The Lean verifier's `--registered` file: per registered port, the value it reads, its registered root, leaf count
    and domain, and each unit's position. `entries` are the registration's (R7-checked against the verifier's own),
    `read_map` is `reads`', and `ports` names a template parameter's circuit port when they differ."""
    by_value = {e["value"]: e for e in entries}
    out = {}
    for name, r in read_map.items():
        e = by_value.get(r["value"])
        if e is None:
            raise ValueError(f"parameter {name!r} reads {r['value']!r}, which the registration does not name")
        out[(ports or {}).get(name, name)] = {"value": e["value"], "root": e["root"], "leaves": int(e["leaves"]),
                                              "domain": e["domain"], "reads": list(r["rows"])}
    return {"format": READS, "ports": out}


__all__ = ["ENTRY_KEYS", "OWNER", "READS", "REGISTERED_DOMAIN", "Registered", "SCHEMA", "WORD_BITS", "domain", "leaf",
           "parameter_gates", "reads", "reads_file", "shape", "table", "verify_read"]
