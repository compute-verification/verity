"""The row tables of a serving-rows window (`commit.serving_rows.RowTable`), as PoUW binds ``pouw.roots`` to them.

One row table per int8 quantize output: the token rows of one linear at one engine step, one ``hm96-sha512/row-seg/v1`` leaf each
(``verity.ml.commitments.row_table``).  ``find`` gives a (step, layer, linear)'s table: the root the verifier fills ``pouw.roots``
with, the domain id and leaf count an opening verifies under, and each token row's leaf index (``Table.rank``).  The verifier's
tables come from its own window (``pipeline.serving_rows.population`` over the Build's Programs and the arrivals, the partition
file's ``row_tables: true``) and the registration ``registration.check`` accepted (``of_registration``); the prover's from the
window directory serving wrote (``of_dir``), with ``opening`` (a set of ranks' strings and their multiproof, from
``t<k>-public.bin``) and ``witness`` (their rows and salts, from ``t<k>-private.bin``: what a tile's proof takes privately).
``verify`` checks an opening against a table, with the rows and salts in the clear when given.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ml.commitments import row_table as RT
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

from . import serving_rows as CS

__all__ = ["Table", "find", "of_dir", "of_registration", "opening", "verify", "witness"]


@dataclass(frozen=True)
class Table:
    """A row table with its root: ``spec`` (its engine step, op_path, words and each rank's token row), ``root``, its domain and
    ``k``, its index among the window's row tables (its files ``t<k>-*``)."""
    spec: CS.RowTable
    root: bytes
    domain: CommitmentDomain
    k: int

    @property
    def port(self) -> str:
        return self.spec.port

    @property
    def leaves(self) -> int:
        return self.spec.n

    @property
    def domain_id(self) -> bytes:
        return self.domain.domain_id

    @property
    def engine_step(self) -> int:
        return self.spec.engine_step

    @property
    def layer(self) -> int | None:
        return self.spec.layer

    @property
    def op_path(self) -> str:
        return self.spec.op_path

    def rank(self, request: str, token_row: int) -> int:
        """The leaf index of ``request``'s token row ``token_row``."""
        return self.spec.rank(request, token_row)


def of_registration(window: CS.Window, rec: Mapping[str, Any]) -> list[Table]:
    """The verifier's tables: each row table of its own ``window`` with the root the registration ``rec`` holds for its port;
    refused when the record's leaves or domain are not the window's (a caller has run ``registration.check``, whose R3 is this)."""
    got = {r["port"]: r for r in rec["roots"]}
    out = []
    for k, t in enumerate(window.row_tables):
        dom, r = window.domain(t.port, CS.TABLE_SCHEMA, t.n), got.get(t.port)
        if r is None or r["leaves"] != t.n or r["domain"] != dom.domain_id.hex():
            raise ValueError(f"{t.port}: the registration does not hold the window's row table ({t.n} leaves, its served domain)")
        out.append(Table(t, bytes.fromhex(r["root"]), dom, k))
    return out


def of_dir(window_dir: str | Path) -> list[Table]:
    """The prover's tables, from the window serving wrote (``window.json``'s ``row_tables``)."""
    out = []
    for e in json.loads((Path(window_dir) / "window.json").read_text()).get("row_tables", []):
        spec = CS.RowTable(int(e["engine_step"]), e["op_path"], e["template"], int(e["words"]), tuple((str(a), int(b), int(c)) for a, b, c in e["rows"]))
        dom = CommitmentDomain(bytes.fromhex(e["binding"]), CS.OWNER, RangeIndexedDomain(0, spec.n), hash="sha512")
        if dom.domain_id.hex() != e["domain"] or spec.port != e["port"]:
            raise ValueError(f"row table {e['k']}: its binding is not its domain {e['domain'][:16]}")
        out.append(Table(spec, bytes.fromhex(e["root"]), dom, int(e["k"])))
    return out


def find(tables: Sequence[Table], step: int, layer: int, linear: str) -> Table:
    """The table of engine step ``step``'s ``linear`` in layer ``layer``: ``linear`` its op_path, or the op_path's last dotted parts
    (``qkv_proj``, ``self_attn.qkv_proj``)."""
    hits = [t for t in tables if t.engine_step == int(step) and t.layer == int(layer)
            and (t.op_path == linear or t.op_path.endswith("." + linear))]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} row tables of step {step}, layer {layer}, linear {linear!r}"
                          + (f": {[t.op_path for t in hits]}" if hits else ""))
    return hits[0]


def _strings(window_dir: str | Path, t: Table) -> list[bytes]:
    pub = (Path(window_dir) / f"t{t.k}-public.bin").read_bytes()
    if len(pub) != CS.BC_BYTES * t.leaves:
        raise ValueError(f"t{t.k}-public.bin is {len(pub)} bytes, not {t.leaves} strings")
    return [pub[i * CS.BC_BYTES:(i + 1) * CS.BC_BYTES] for i in range(t.leaves)]


def opening(window_dir: str | Path, t: Table, ranks: Iterable[int]) -> RT.RowOpening:
    """The prover's opening of ``t``'s rows at ``ranks``: their strings and the multiproof over their leaves; refused when the
    window's strings are not ``t``'s root's."""
    tab = RT.RowTable(t.domain, _strings(window_dir, t))
    if tab.root != t.root:
        raise ValueError(f"t{t.k}-public.bin is not {t.port}'s rows")
    return tab.open(ranks)


def witness(window_dir: str | Path, t: Table, ranks: Iterable[int]) -> dict[int, tuple[bytes, int, bytes]]:
    """The prover's rows and salts of ``t`` at ``ranks``: ``{rank: (row, n_bits, salt)}``, ``row`` the s8 codes then the f32 scale
    (``row_table.s8_row``)."""
    priv = (Path(window_dir) / f"t{t.k}-private.bin").read_bytes()
    rb = 2 * t.spec.words
    if len(priv) != t.leaves * (rb + CS.SALT_BYTES):
        raise ValueError(f"t{t.k}-private.bin is {len(priv)} bytes, not {t.leaves} rows of {rb} bytes and their salts")
    base = t.leaves * rb
    return {r: (priv[r * rb:(r + 1) * rb], 8 * rb, priv[base + r * CS.SALT_BYTES:base + (r + 1) * CS.SALT_BYTES]) for r in sorted(set(ranks))}


def verify(t: Table, o: RT.RowOpening, rows: Mapping[int, tuple[bytes, int, bytes]] | None = None) -> bool:
    """Whether ``o`` opens ``t``'s rows at its ranks (``row_table.verify``), and, with ``rows``, they are its rows and salts."""
    if rows is None:
        return RT.verify(t.root, t.domain_id, t.leaves, o)
    return RT.verify_rows(t.root, t.domain_id, t.leaves, o, rows)
