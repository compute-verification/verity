"""Registered ports: a statement's table rows read as the rows of a port of a registered window's record, as a two-stage
audit's sessions read every row they prove: the link templates' at the boundary registration, from inside the interior
registration's proof blocks, and every other at the interior registration.

A committed window's port (`rows.Committed`) is a tree of hiding row leaves under its window domain, each leaf framed as a
registered value's (`rowleaf.row_leaf` of the row's ``b || c`` under the port's schema), so a later statement reads its rows as
it reads a registered value's (`Port`):

* the reading port's table row is the earlier port's row at its leaf, committed under that row's salt (`rows.commit`'s
  ``salts_of``), so its ``b || c`` is the earlier record's;
* the reading member's header names the earlier port, the leaves and each leaf's path to its root (`Port.header`, the entry
  `rows.write`'s ``registered`` takes);
* the verifier's own ``--registered`` file names the port's root, domain and leaves as the earlier record registered them,
  and the leaf each unit reads (`reads`), from that record and public structure alone.

Lean's registered reads then refuse a row that isn't the earlier port's at its leaf ("does not open"), a read of another leaf
("reads position") and a read of another port ("not another value").
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from verity.commitments.merkle import CommitmentDomain, tree_levels
from verity.commitments.rowleaf import row_leaf
from verity.commitments.windows import rows as CS

__all__ = ["Port", "reads", "reads_file"]


@dataclass(frozen=True)
class Port:
    """Port ``value`` of an earlier window, as a later statement reads it: its row schema, words a row, domain and every
    row's ``b || c`` (public), and for the prover its rows (``(leaves, words)`` u16) and salts (one a row, concatenated)."""

    value: str
    schema: str
    words: int
    domain: CommitmentDomain
    bcs: bytes = field(repr=False)
    rows: np.ndarray | None = field(default=None, repr=False, compare=False)
    salts: bytes | None = field(default=None, repr=False, compare=False)

    @classmethod
    def of(cls, c: CS.Committed, value: str) -> "Port":
        """Port ``value`` (``<member>/<port>``) of the committed window ``c``, its rows and salts the committer's."""
        found = [(m, p) for m in c.window.members for p, _ in m.all_rows if m.port(p) == value]
        if len(found) != 1 or value not in c.roots:
            raise ValueError(f"{value!r} is no row port of the window (its ports: {sorted(c.roots)[:6]})")
        (m, p), = found
        rows = np.asarray(c.rows[value], dtype=np.uint16)
        port = cls(value, m.schema(p), int(rows.shape[1]), c.domains[value], bytes(c.bc[value]), rows, bytes(c.salts[value]))
        if port.root != c.roots[value]:
            raise ValueError(f"{value}: its leaves' tree is not the window's root")
        return port

    @property
    def leaves(self) -> int:
        return self.domain.count

    @cached_property
    def levels(self) -> tuple[tuple[bytes, ...], ...]:
        bc, n = self.bcs, CS.BC_BYTES
        if len(bc) != self.leaves * n:
            raise ValueError(f"{self.value}: {len(bc)} bytes of b || c, not {self.leaves} rows'")
        return tree_levels(self.domain, [row_leaf(self.domain, i, CS.INST.tree_leaf(CS.INST.default_key, bc[i * n:(i + 1) * n]),
                                                  self.schema) for i in range(self.leaves)])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def path(self, position: int) -> bytes:
        """Leaf ``position``'s siblings, leaf level first: what Lean opens the root through."""
        if not 0 <= position < self.leaves:
            raise ValueError(f"{self.value}: leaf {position} outside its {self.leaves}")
        out, i = b"", position
        for level in self.levels[:-1]:
            out += level[i ^ 1]
            i >>= 1
        return out

    def header(self, positions: Sequence[int]) -> dict[str, Any]:
        """The reading member's header entry (`rows.write`'s ``registered``): this port, the leaves its table rows are, and
        each one's path."""
        at = [int(p) for p in positions]
        return {"value": self.value, "positions": at, "paths": [self.path(p).hex() for p in at]}

    def rows_at(self, positions: Sequence[int]) -> np.ndarray:
        """The rows at ``positions``, ``(len(positions), words)`` u16: the prover's."""
        if self.rows is None:
            raise ValueError(f"{self.value}: the prover's rows are not held")
        return self.rows[np.asarray(positions, dtype=np.int64)]

    def salts_at(self, positions: Sequence[int]) -> bytes:
        """The salts of the rows at ``positions``, concatenated: the prover's, which the reading port's rows are committed
        under."""
        if self.salts is None:
            raise ValueError(f"{self.value}: the prover's salts are not held")
        s = CS.SALT_BYTES
        return b"".join(self.salts[int(p) * s:(int(p) + 1) * s] for p in positions)


def reads(record: Mapping[str, Any], value: str, words: int, schema: str, positions: Sequence[int], base: int, *,
          registered: Any) -> dict[str, Any]:
    """One port of the verifier's own ``--registered`` file (``registered``: core's ``one_stage.registered``, whose format it
    is): port ``value`` of the earlier registration ``record``, at the root, leaves and domain that record names, its rows
    ``words`` 16-bit words under ``schema`` (the verifier's own layout of the port), unit ``base + i`` reading leaf
    ``positions[i]``."""
    got = [r for r in record["roots"] if r["port"] == value]
    if len(got) != 1:
        raise ValueError(f"the earlier record names port {value!r} {len(got)} times, not once")
    e, = got
    if not all(0 <= int(p) < int(e["leaves"]) for p in positions):
        raise ValueError(f"{value}: a read outside its {e['leaves']} leaves")
    return {"value": value, "root": e["root"], "leaves": int(e["leaves"]), "words": int(words),
            "word_bits": registered.WORD_BITS, "domain": e["domain"], registered.SCHEMA_KEY: schema,
            "reads": [int(p) for p in positions], registered.READS_FROM: int(base)}


def reads_file(ports: Mapping[str, Mapping[str, Any]], *, registered: Any) -> dict[str, Any]:
    """The verifier's own ``--registered`` file of ``{circuit port: reads(...)}``."""
    return {"format": registered.READS_V1, "ports": {p: dict(e) for p, e in ports.items()}}
