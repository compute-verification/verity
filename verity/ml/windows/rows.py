"""A window's rows: the ``frame-v3-sha512`` + ``hm96-sha512/v1`` commitment of a window's units in C-Flock's statement row leaf
format (``verity_flock.circuit``'s statement rows), with the registration, C-Flock's public and prover files and the
labels the one-stage audit binds (``verity.core.protocols.sampled_proofs.one_stage``).

The population is the verifier's: its program is ``TemplateInstances`` / ``TemplatePopulations`` over the members' templates and
counts (``verity.ml.windows.population``, over the builders core's ``one_stage.partition`` names too), its
query the template query over them, so the partition (``verity/partition/v1``, ``partition_object``) and its digest are core's
over that program, and the registration is core's ``verity/registration/v1`` record with ``window = {"kind": "served" |
"stand-in", "run": <the record's run root>, ...provenance}`` (``registration``), made by the ``registrar`` the caller passes
(core's ``one_stage.registration``): this module imports no core.  Any other program an integration ran is provenance in the
window: no query of core's selects template instances inside another Program (the template queries need every root node to be
an instance).

A population is a sequence of members (:class:`Member`), one per template in the partition's order, or a template's units in
consecutive chunks, one member per statement (a circuit's spec shape caps its instances), each with its row ports and output
ports in the order of C-Flock's circuit prover (``verity_flock.circuit``).  Each output port is a row too, after the member's
input rows: its words zero-padded to whole SHA-512 blocks, committed under its own salt like an input row (an output is a
value: no output word is ever public).  A bit-row member
(C-Flock's ``row_schemas``) commits each row as the bits of its words under its port's schema (``hm96-sha512/row/v2`` or
``…/row-seg/v1``), each output row exactly its words.  Per row of ``hm96-sha512/row/v1``
(``tests/commit/test_serving_rows.py`` pins it to core and to the circuit prover's vector):

  row     the words, u16 little-endian
  inner   ``x = SHA-512(sha512_row_prefix(role 1, 16, n) || row)``                              (``sha512/row/v1``)
  salt    ``y`` = 192 bytes of ChaCha20 (RFC 8439) under the window's salt key, nonce = the row port's index over all members (12
          bytes LE), at stream offset ``192 i``; the key is 32 fresh bytes from the OS generator per window, dropped once the
          salts are expanded.  The salts are kept only in the prover's files and never logged
  string  ``b || c = hm96.SHA512.commit_string(default_key, x, y)``                               (public, 128 bytes)
  leaf    the ``frame-v3-sha512`` row leaf, schema ``hm96-sha512/row/v1``, over ``hm96.SHA512.tree_leaf(default_key, b || c)``

Each port ``<member>/<port>`` is one tree under ``CommitmentDomain(binding, -1, [0, leaves), "sha512")``, with core's
window-domain rule (``verity.ml.commitments.window``): ``binding = identity_digest_sha512(DOMAIN_TAG, {program, partition, port,
schema, leaves, run})``, every field a registration field and ``run`` the record's run root, so the verifier recomputes every domain
from the registration and the Commit's record (``expected_ports``).  The leaf layer is the SHA-512 of the canonical map from each
member (its template, or its chunk's name) to the SHA-512 of its public body (``leaf_layer``).  ``write`` puts into a window
directory ``registration.json``, ``index.json`` (the integration's labels of the units, opaque here and prover only, with its
salt in ``index-salt.bin``; the window's provenance names its hiding leaf, `index_leaf`), per member
``m<k>-public.bin`` and ``m<k>-private.bin`` (the circuit prover's public and prover bodies; the private one is prover
only, less the rows and salts of a port given as a registered value's, which ``v<j>-private.bin`` holds once for the
window, and of a wired port, which its writers' own files hold) and ``m<k>-units.json`` (its global unit indices), and
``window.json``; ``flock_inputs.flock_input_files`` wraps member ``k``'s bodies as the circuit prover's
``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin`` (`Bodies`).

A window may carry row tables (:class:`RowTable`, ``Window.row_tables``): shared tables of rows several units read, each a tree
of ``hm96-sha512/row-seg/v1`` leaves, committed in segments, each segment's root given before that segment's live coin: a
segment's tables in the order the integration gives them, under the ``table_domain`` of the window's ``context``, salted by
``segment_salts`` under the run's row key, and the segment's root the tree over their roots.  The window registers one port for
them, ``row-table``: the tree over its segments' roots (window-domain rule, leaf schema ``ROOTS_SCHEMA``), so no row is committed
twice.  Its linked members (``Member.links``) are shared-row chunks whose ``x`` table holds the distinct rows they read, each
under its row table row's salt (so the same ``b || c``), read by offset; the integration's verifier side ties them to the
recorded segment roots (``tables.check_row_tables``).  ``write`` adds ``t<k>-public.bin``
(every row's ``b || c``) and ``t<k>-private.bin`` (rows, then salts; prover only).  A wired member (``Member.wired``) commits
each wired input port as other units' output rows under their salts, which `wiring.check` holds across statements.
"""
from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import re
import shutil
import time
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits import partition_object as PO
from verity.ml.commitments import hm96, row_table as RT
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.window import WINDOW_DOMAIN, window_domain
from verity.ml.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2,
                                           sha512_row_prefix, sha512_row_seg_digest, sha512_row_v2_prefix)
from verity.ml.circuits.codec import canonical_json
from verity.ml.windows.population import population_program, template_partition

from . import wiring as WI
from .hiding import HidingLeaves

SCHEME = "frame-v3-sha512/hm96-sha512"
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW
BIT_ROW_SCHEMAS = (SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG)
UNITS_SCHEMA = "verity-vllm/serving-rows/units/v0"
WORD_SCHEMA = "u16"
DOMAIN_TAG = WINDOW_DOMAIN
OWNER = -1
ORDER = "verity-vllm/serving-rows/order/v0"
LEAF_LAYER = "SHA-512 of canon({template: SHA-512 of m<k>-public.bin}) over the window's members"
INST = hm96.SHA512
SALT_BYTES = INST.salt_bytes
BC_BYTES = INST.proven_bytes
CHUNK = 1 << 14
#: the bytes of a value's rows a file is written or read in at a time
VALUE_CHUNK = 1 << 26
#: a row table's schema, the window's port over its steps' roots (and the tables' name prefix), and the port of a linked member
#: that reads a table
TABLE_SCHEMA, TABLE_PORT, LINKED = SCHEMA_HM96_SHA512_ROW_SEG, "row-table", "x"
ROOTS_SCHEMA = RT.ROOTS_SCHEMA
#: ``window.json``'s key for a segment's index in its ``row_table_steps``: its bytes predate the name (a later rename)
SEGMENT_KEY = "engine_step"
#: `commit`'s rows of a wired input port given by reference: its writers' output rows (`Member.wired`), their salts and
#: ``b || c`` taken, never hashed again
WIRED = "wired"
#: ``index.json``'s salt beside it (prover only), the tag its inner digest is taken under and its leaf's rule (`index_leaf`)
INDEX_SALT = "index-salt.bin"
INDEX_TAG = b"verity/window-index/v1\0"
INDEX_RULE = "hm96-sha512/v1 leaf of SHA-512(\"verity/window-index/v1\\0\" || index.json) under the 192-byte salt index-salt.bin"


@dataclass(frozen=True)
class ValueRows:
    """`commit`'s rows of an input port given by reference: rows ``positions`` of ``value``, a registered value of 16-bit leaves
    (core's ``one_stage.registered.Registered``: ``words``, ``word_bits``, ``rows``, ``salts`` and ``bcs`` by row), committed
    under the value's salts, so each row's ``b || c`` is the value's. ``salts``: those rows' salts concatenated, when the
    caller has drawn them (`value_salts`), else drawn here."""
    value: Any
    positions: tuple[int, ...]
    salts: bytes | None = field(default=None, repr=False, compare=False)


def value_salts(value: Any, positions: Sequence[int] | np.ndarray) -> bytes:
    """Rows ``positions`` of registered value ``value``: their salts concatenated, all at once where its salts draw many
    (``salts.at``, `pouw.h1.registered.Salts`), else one by one."""
    at = getattr(value.salts, "at", None)
    return at(positions) if at is not None else b"".join(bytes(value.salts[int(i)]) for i in positions)


def value_rows(value: Any, positions: Sequence[int] | np.ndarray) -> np.ndarray:
    """Rows ``positions`` of registered value ``value`` as ``(len(positions), words)`` u16, gathered at once where its rows
    are an array or gather many (``rows.at``), else row by row."""
    r, at = value.rows, np.asarray(positions, dtype=np.int64)
    a = r[at] if isinstance(r, np.ndarray) else r.at(at) if hasattr(r, "at") else [r[int(i)] for i in at]
    return np.asarray(a, dtype=np.uint16).reshape(len(at), value.words)


_FRAME = b"veritor/protocol/merkle/frame/v3\0"


# --- salts: ChaCha20 (RFC 8439) over numpy -------------------------------------------------------------------------------

_SIGMA = np.frombuffer(b"expand 32-byte k", dtype="<u4")
_QR = ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15), (0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14))


def _rotl(v: np.ndarray, n: int) -> np.ndarray:
    return (v << np.uint32(n)) | (v >> np.uint32(32 - n))


def chacha20(key: bytes, nonce: bytes, nbytes: int, counter: int = 0) -> bytes:
    """``nbytes`` of the RFC 8439 ChaCha20 keystream (32-byte key, 12-byte nonce, 32-bit block counter from ``counter``)."""
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("ChaCha20 takes a 32-byte key and a 12-byte nonce")
    nblk = -(-nbytes // 64)
    if counter + nblk > 1 << 32:
        raise ValueError("the 32-bit block counter would wrap")
    out = []
    for lo in range(0, nblk, 1 << 16):
        m = min(1 << 16, nblk - lo)
        s = np.empty((16, m), dtype=np.uint32)
        s[0:4] = _SIGMA[:, None]
        s[4:12] = np.frombuffer(key, dtype="<u4")[:, None]
        s[12] = np.arange(counter + lo, counter + lo + m, dtype=np.uint64).astype(np.uint32)
        s[13:16] = np.frombuffer(nonce, dtype="<u4")[:, None]
        x = s.copy()
        for _ in range(10):
            for a, b, c, d in _QR:
                x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 16)  # noqa: E702
                x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 12)  # noqa: E702
                x[a] += x[b]; x[d] = _rotl(x[d] ^ x[a], 8)  # noqa: E702
                x[c] += x[d]; x[b] = _rotl(x[b] ^ x[c], 7)  # noqa: E702
        x += s
        out.append(np.ascontiguousarray(x.T).astype("<u4").tobytes())
    return b"".join(out)[:nbytes]


def salts(key: bytes, port_index: int, n: int) -> bytes:
    """The ``n`` rows' salts of port ``port_index`` (192 bytes each, concatenated)."""
    return chacha20(key, port_index.to_bytes(12, "little"), n * SALT_BYTES)


def segment_salts(key: bytes, segment: int, n: int) -> bytes:
    """The salts of ``segment``'s ``n`` table rows under the run's row key (192 bytes each, concatenated): its tables in order,
    each table's rows in rank order, one stream per segment (nonce = the segment, 12 bytes LE)."""
    return chacha20(key, segment.to_bytes(12, "little"), n * SALT_BYTES)


# --- the frame on SHA-512, batched ---------------------------------------------------------------------------------------

def _uint(v: int) -> bytes:
    return v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")


def _part(p: bytes) -> bytes:
    return len(p).to_bytes(8, "big") + p


class _Frame:
    """``merkle._hash`` for one SHA-512 domain, with the constant prefix of each tag hashed once."""

    def __init__(self, domain_id: bytes) -> None:
        head = lambda tag: hashlib.sha512(_FRAME + len(tag).to_bytes(4, "big") + tag + _part(domain_id))  # noqa: E731
        self._leaf, self._node, self._pad = head(b"leaf"), head(b"node"), head(b"pad")

    def leaf(self, rank: int, schema: bytes, value: bytes) -> bytes:
        u = _part(_uint(rank))
        h = self._leaf.copy()
        h.update(u + u + schema + _part(value))
        return h.digest()

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        h = self._node.copy()
        h.update(_part(_uint(level)) + _part(_uint(index)) + _part(left) + _part(right))
        return h.digest()

    def pad(self, rank: int) -> bytes:
        h = self._pad.copy()
        h.update(_part(_uint(rank)))
        return h.digest()

    def reduce(self, level: list[bytes], first: int, depth: int = 0) -> bytes:
        """Fold a level whose first node has global index ``first`` at ``depth`` up to one node."""
        while len(level) > 1:
            level = [self.node(depth, (first >> 1) + j, level[2 * j], level[2 * j + 1]) for j in range(len(level) // 2)]
            first >>= 1
            depth += 1
        return level[0]


def _width(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def _copy_update(h: Any, data: bytes | memoryview) -> bytes:
    g = h.copy()
    g.update(data)
    return g.digest()


_JOB: dict[str, Any] = {}


def _chunk(task: tuple[str, int, int]) -> tuple[str, int, bytes, bytes]:
    """Ranks [start, start + size) of one port (pads past its count) folded to one node; and, for a row port, their ``b || c``
    (a ``string`` port's are given)."""
    port, start, size = task
    j = _JOB["ports"][port]
    fr, n = _Frame(j["domain_id"]), j["count"]
    hi = min(start + size, n)
    leaves: list[bytes] = []
    bc = b""
    if j["kind"] == "row" and hi > start:
        m, rb = hi - start, j["row_bytes"]
        rows = memoryview(j["rows"])[start * rb:hi * rb]
        ys = j["salts"][start * SALT_BYTES:hi * SALT_BYTES]
        x = b"".join(j["inner"](rows[i * rb:(i + 1) * rb]) for i in range(m))
        b = memoryview((np.frombuffer(x, dtype=np.uint8).reshape(m, 64) ^ _JOB["hiding"].masks(ys, m)).tobytes())
        yv, out = memoryview(ys), bytearray()
        salt_h, leaf_h, schema = _JOB["salt_h"], _JOB["leaf_h"], j["schema"]
        for i in range(m):
            s = bytes(b[i * 64:(i + 1) * 64]) + _copy_update(salt_h, yv[i * SALT_BYTES:(i + 1) * SALT_BYTES])
            out += s
            leaves.append(fr.leaf(start + i, schema, _copy_update(leaf_h, s)))
        bc = bytes(out)
    elif j["kind"] == "string" and hi > start:
        s, leaf_h = memoryview(j["strings"]), _JOB["leaf_h"]
        leaves = [fr.leaf(i, j["schema"], _copy_update(leaf_h, s[i * BC_BYTES:(i + 1) * BC_BYTES])) for i in range(start, hi)]
    elif j["kind"] == "word" and hi > start:
        w = j["words"][start:hi].astype(">u2").tobytes()
        leaves = [fr.leaf(start + i, j["schema"], w[2 * i:2 * i + 2]) for i in range(hi - start)]
    leaves += [fr.pad(r) for r in range(max(start, n), start + size)]
    return port, start, fr.reduce(leaves, start), bc


# --- the window ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """One statement's units of the population: its template's descriptor id, row and output ports ``(name, words)`` in the
    circuit prover's order, its instance count and the global index of its first unit.  A template is one member, or
    consecutive chunks of its units (``name`` ``<descriptor id>/c<j>``), one per statement of at most the circuit's chunk cap.  ``schemas``: a bit-row
    member's row schema per port (``hm96-sha512/row/v2`` or ``…/row-seg/v1``, C-Flock's ``row_schemas``), each row the bits
    of its u16 words and each output row exactly its words; else every row is ``hm96-sha512/row/v1``."""
    template: str
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int], ...]
    n: int
    base: int = 0
    tables: tuple[tuple[str, int], ...] = ()     # a shared-row member: each row port's table size (rows are referenced by refs)
    name: str = ""
    schemas: tuple[tuple[str, str], ...] = ()
    link_units: tuple[int, ...] = ()             # a row-table member: per link, how many of its units (consecutive, in order) read it
    links: tuple[tuple[int, int], ...] = ()      # a row-table member: per row of its x table, the window's row table and rank it is
    wired: tuple[tuple[str, str, tuple[int, ...]], ...] = ()  # per wire (port, out, writers): unit i's port row is unit writers[i]'s out row (`wiring`)

    def __post_init__(self) -> None:
        if self.schemas and (sorted(p for p, _ in self.schemas) != sorted(p for p, _ in self.rows + self.outs)
                             or any(s not in BIT_ROW_SCHEMAS for _, s in self.schemas)):
            raise ValueError(f"{self.key}: a bit-row member names one of {BIT_ROW_SCHEMAS} for each port, not {dict(self.schemas)}")
        if self.links and dict(self.tables).get(LINKED) != len(self.links):
            raise ValueError(f"{self.key}: {len(self.links)} links, one per row of its {LINKED!r} table {dict(self.tables)}")
        if self.links and len({t for t, _ in self.links}) != 1:
            raise ValueError(f"{self.key}: links {self.links} name several row tables; a linked member's {LINKED!r} rows are one "
                             "table's, the one root its registered reads open against")
        if (len(self.link_units) != len(self.links) or any(type(c) is not int or c < 1 for c in self.link_units)
                or (self.links and sum(self.link_units) != self.n)):
            raise ValueError(f"{self.key}: link units {self.link_units} are not one positive count per link summing to its {self.n} units")

    @property
    def link_refs(self) -> tuple[int, ...]:
        """A linked member's refs into its ``x`` table: each unit's link, in order."""
        return tuple(j for j, c in enumerate(self.link_units) for _ in range(c))

    @property
    def key(self) -> str:
        """What its ports, the leaf layer and the files call it: its template, or its chunk's name."""
        return self.name or self.template

    @property
    def out_rows(self) -> tuple[tuple[str, int], ...]:
        """Each output port's row: its words zero-padded to whole SHA-512 blocks (64 words each), as the circuit prover's
        ``out_row``; a bit-row member's, exactly its words."""
        return tuple((p, w if self.schemas else -(-w // 64) * 64) for p, w in self.outs)

    @property
    def all_rows(self) -> tuple[tuple[str, int], ...]:
        """The circuit prover's rows: the input rows, then one output row per output port."""
        return self.rows + self.out_rows

    def schema(self, port: str) -> str:
        return dict(self.schemas).get(port, ROW_SCHEMA)

    @property
    def row_schemas(self) -> list[list[Any]] | None:
        """C-Flock META's ``row_schemas`` for a bit-row member: ``[schema, bits]`` per row, in the circuit prover's order."""
        return [[self.schema(p), 16 * w] for p, w in self.all_rows] if self.schemas else None

    def port(self, name: str) -> str:
        """The name a root, its domain and the registration use: ``<member>/<port>``."""
        return f"{self.key}/{name}"


@dataclass(frozen=True)
class RowTable:
    """A shared table: rows that linked members' units read (their ``x`` rows), ``words`` u16 words each, one
    ``hm96-sha512/row-seg/v1`` leaf each, committed with its segment's other tables before that segment's coin.  ``name`` and
    each row's ``keys[rank]`` are the integration's and opaque here (they only tell tables and rows apart), as are the
    ``label`` fields its ``window.json`` entry carries.  Fixed by the circuit, never by values.  ``port`` names it (its files,
    the window's entries); the window registers its segment's root, not its own."""
    segment: int
    name: str
    template: str
    words: int
    keys: tuple[tuple[Any, ...], ...]
    label: tuple[tuple[str, Any], ...] = ()

    @property
    def port(self) -> str:
        return f"{TABLE_PORT}/{self.segment}/{self.name}"

    @property
    def n(self) -> int:
        return len(self.keys)

    def rank(self, *prefix: Any) -> int:
        """The leaf index of the one row whose key starts with ``prefix``."""
        hits = [k for k, key in enumerate(self.keys) if tuple(key[:len(prefix)]) == prefix]
        if len(hits) != 1:
            raise KeyError(f"{self.port}: {len(hits)} rows keyed {prefix}")
        return hits[0]

    def doc(self) -> dict[str, Any]:
        return {"port": self.port, **dict(self.label), "template": self.template, "words": self.words, "leaves": self.n,
                "schema": TABLE_SCHEMA, "rows": [list(r) for r in self.keys]}


@dataclass(frozen=True)
class Window:
    """What the roots are bound to: the verifier's population program and query (the partition over them), the law it draws by, the
    run (the record's run root, hex), and the members the population is made of, in the partition's order (refused otherwise); and
    its row tables, grouped by segment in rising order and in the integration's order within one, every rank of each the row of
    at least one linked member's ``x`` table, under ``context`` (hex, 64 bytes known before the first segment: the live coin
    commitment's root).  A ``work`` law
    draws by the verifier's ``work_table`` (`draw.work_table`), which the registration's law is expanded under."""
    program: Any
    query: dict[str, Any]
    law: dict[str, Any]
    run: str
    members: tuple[Member, ...]
    row_tables: tuple[RowTable, ...] = ()
    context: str = ""
    work_table: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        mine = [(m.template, m.base, m.n) for m in self.members]
        want = [(tid, lo, n) for _k, tid, lo, n in template_partition(self.program, self.query).ranges]
        tiled, it = [], iter(mine)
        for t, base, n in want:
            at = base
            while at < base + n:
                got = next(it, None)
                if got is None or got[0] != t or got[1] != at or got[2] < 1:
                    break
                at += got[2]
            tiled.append(at == base + n)
        if not all(tiled) or next(it, None) is not None or sum(m.n for m in self.members) != sum(n for _, _, n in want):
            raise ValueError(f"members {mine} are not the partition's units {want} (each template's, in consecutive chunks)")
        if len({m.key for m in self.members}) != len(self.members):
            raise ValueError(f"members {[m.key for m in self.members]}: each its own name")
        WI.validate(self.members)
        covered = [np.zeros(t.n, dtype=bool) for t in self.row_tables]
        for m in self.members:
            for t, r in m.links:
                if not (0 <= t < len(covered) and 0 <= r < len(covered[t])):
                    raise ValueError(f"{m.key}: link {(t, r)} is no row of the window's {len(covered)} row tables")
                if self.row_tables[t].words != dict(m.rows)[LINKED]:
                    raise ValueError(f"{m.key}: its {LINKED!r} rows are {dict(m.rows)[LINKED]} words, {self.row_tables[t].port}'s "
                                     f"{self.row_tables[t].words}")
                covered[t][r] = True
        if len({t.port for t in self.row_tables}) != len(self.row_tables) or not all(c.all() for c in covered):
            raise ValueError(f"row tables {[t.port for t in self.row_tables]}: each its own port, every row some member's")
        segs = [t.segment for t in self.row_tables]
        if any(a > b for a, b in zip(segs, segs[1:])) or not all(len(set(t.keys)) == t.n for t in self.row_tables):
            raise ValueError("row tables go by segment, rising, each table's row keys each once")
        if self.row_tables and not re.fullmatch(f"[0-9a-f]{{{2 * RT.DIGEST_BYTES}}}", self.context):
            raise ValueError(f"a window with row tables names its context, {RT.DIGEST_BYTES} bytes hex known before its first "
                             f"segment (the live coin commitment's root), not {self.context!r}")

    @cached_property
    def segments(self) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Each segment of the row tables, in order, with its tables' indices among ``row_tables``."""
        out: dict[int, list[int]] = {}
        for k, t in enumerate(self.row_tables):
            out.setdefault(t.segment, []).append(k)
        return tuple((s, tuple(ks)) for s, ks in out.items())

    @cached_property
    def places(self) -> tuple[int, ...]:
        """Each row table's index in its segment."""
        return tuple(i for _s, ks in self.segments for i in range(len(ks)))

    def table_domain(self, k: int) -> CommitmentDomain:
        """Row table ``k``'s domain under the window's context."""
        t = self.row_tables[k]
        return RT.table_domain(bytes.fromhex(self.context), t.segment, self.places[k], t.n)

    def segment_root(self, segment: int, table_roots: Sequence[bytes]) -> bytes:
        return RT.step_tree(bytes.fromhex(self.context), segment, table_roots).root

    @cached_property
    def tables_domain(self) -> CommitmentDomain:
        """The ``row-table`` port's domain: the tree over the segments' roots, under the window-domain rule."""
        return self.domain(TABLE_PORT, ROOTS_SCHEMA, len(self.segments))

    @cached_property
    def partition(self) -> dict[str, Any]:
        return PO.build(self.program, self.query)

    @cached_property
    def partition_digest(self) -> str:
        return PO.digest(self.partition)

    @cached_property
    def population(self) -> int:
        return template_partition(self.program, self.query).population

    def binding(self, port: str, schema: str, leaves: int) -> bytes:
        return identity_digest_sha512(DOMAIN_TAG, {"program": self.partition["program"], "partition": self.partition_digest,
                                                   "port": port, "schema": schema, "leaves": int(leaves), "run": self.run})

    def domain(self, port: str, schema: str, leaves: int) -> CommitmentDomain:
        return CommitmentDomain(self.binding(port, schema, leaves), OWNER, RangeIndexedDomain(0, leaves), hash="sha512")


@dataclass
class Committed:
    window: Window
    roots: dict[str, bytes]            # by <member>/<port>
    domains: dict[str, CommitmentDomain]
    bc: dict[str, bytes]               # per row port: every row's b || c, concatenated
    salts: dict[str, bytes]            # per row port: every row's salt (private)
    rows: Mapping[str, np.ndarray]     # per row port, the output rows included: (n, words) u16 (private; `_Rows`)
    seconds: dict[str, Any]
    refs: dict[int, np.ndarray] = field(default_factory=dict)   # per shared-row member k: (n, rows) table refs, output rows' own
    table_roots: dict[str, bytes] = field(default_factory=dict)  # per row table port: its root (the segment tree's leaf)
    segment_roots: dict[int, bytes] = field(default_factory=dict)  # per segment: its root (the row-table port's leaf)
    given: dict[str, Any] = field(default_factory=dict)  # per port given by reference: a `ValueRows`, or a wired port's (member, port)

    def public_body(self, k: int) -> bytes:
        """Member ``k``'s public body, as the circuit prover's ``write()`` lays it out: every instance's ``b || c`` (its input rows, then its
        output rows), instance-major; nothing else."""
        m = self.window.members[k]
        if m.tables:                     # the circuit prover's shared-row body: every table row's b || c (port-major), refs (u32, instance-major)
            return b"".join(self.bc[m.port(p)] for p, _ in m.all_rows) + np.ascontiguousarray(self.refs[k]).astype("<u4").tobytes()
        bc = np.concatenate([np.frombuffer(self.bc[m.port(p)], dtype=np.uint8).reshape(m.n, BC_BYTES) for p, _ in m.all_rows], axis=1)
        return bc.tobytes()

    def private_body(self, k: int) -> bytes:
        """Member ``k``'s prover body: every instance's rows (ports in order), then every salt (instance-major, ports in order)."""
        return b"".join(self.private_pieces(k))

    def private_pieces(self, k: int, without: Collection[str] = ()) -> Iterator[bytes]:
        """Member ``k``'s prover body in pieces, each port's rows a piece in a shared-row member, with the rows and salts of
        its ports ``without`` left out, the rest in the body's own order (`write`: the ports another file holds)."""
        m = self.window.members[k]
        ports = [p for p, _ in m.all_rows if p not in without]
        if m.tables:                     # the tables' rows, then their salts, port-major
            for p in ports:
                yield np.ascontiguousarray(self.rows[m.port(p)], dtype="<u2").tobytes()
            for p in ports:
                yield self.salts[m.port(p)]
            return
        if ports:                        # instance-major: every instance's rows, then every instance's salts
            yield np.concatenate([self.rows[m.port(p)] for p in ports], axis=1).astype("<u2").tobytes()
            yield np.concatenate([np.frombuffer(self.salts[m.port(p)], dtype=np.uint8).reshape(m.n, SALT_BYTES)
                                  for p in ports], axis=1).tobytes()

    def by_value(self, k: int) -> dict[str, ValueRows]:
        """Member ``k``'s input ports given as rows of a registered value (`ValueRows`), by port, in a shared-row member (whose
        body lays each port's rows out whole): the ports whose rows and salts `write` can leave out of its prover body."""
        m = self.window.members[k]
        if not m.tables:
            return {}
        return {p: s for p, _ in m.rows if isinstance(s := self.given.get(m.port(p)), ValueRows)}

    def table_public(self, t: int) -> bytes:
        """Row table ``t``'s public body: every row's ``b || c``, in rank order."""
        return self.bc[self.window.row_tables[t].port]

    def table_private(self, t: int) -> bytes:
        """Row table ``t``'s prover body: every row (u16 LE), then every salt, in rank order."""
        p = self.window.row_tables[t].port
        return np.ascontiguousarray(self.rows[p]).astype("<u2").tobytes() + self.salts[p]

    def frame_entry(self, k: int) -> dict[str, Any]:
        """The circuit prover's header ``frame_v3`` entry for member ``k``: its bare port names, as its statement reads them."""
        m = self.window.members[k]
        names = [p for p, _ in m.all_rows]
        return {"frame": "frame-v3-sha512", "roots": {p: self.roots[m.port(p)].hex() for p in names},
                "domain_ids": {p: self.domains[m.port(p)].domain_id.hex() for p in names},
                "bindings": {p: self.domains[m.port(p)].binding.hex() for p in names}, "owner": OWNER,
                "schemas": {p: m.schema(p) for p in names}, "row_key_sha512": hashlib.sha512(INST.default_key).hexdigest()}


def _inner(schema: str, words: int) -> Any:
    """A row's inner digest under ``schema``, from its u16 words' bytes: ``sha512/row/v1`` (role X, 16-bit words), or the bit row
    of its ``16 words`` bits (``sha512/row/v2``, ``sha512/row-seg/v1``)."""
    if schema == ROW_SCHEMA:
        h = hashlib.sha512(sha512_row_prefix(ROLE_X, 16, words))
    elif schema == SCHEMA_HM96_SHA512_ROW_V2:
        h = hashlib.sha512(sha512_row_v2_prefix(16 * words))
    elif schema == SCHEMA_HM96_SHA512_ROW_SEG:
        return lambda row: sha512_row_seg_digest(bytes(row), 16 * words)
    else:
        raise ValueError(f"row schema {schema!r} is not one of {(ROW_SCHEMA, *BIT_ROW_SCHEMAS)}")
    return lambda row: _copy_update(h, row)


def commit_ports(domains: dict[str, CommitmentDomain], rows: dict[str, np.ndarray], words: dict[str, np.ndarray], ys: dict[str, bytes], *,
                 workers: int = 1, chunk: int = CHUNK, schemas: Mapping[str, str] | None = None,
                 strings: Mapping[str, bytes] | None = None) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Roots of row ports (``rows[p]``: (count, words) u16, salts ``ys[p]``, schema ``schemas[p]``, else ``hm96-sha512/row/v1``),
    of row ports whose rows' ``b || c`` are given (``strings[p]``, concatenated) and of word ports (``words[p]``: u16,
    flattened), each under its domain; and every row's ``b || c``.  Equal to core's ``FrameV3.root`` over ``row_leaf`` / word
    leaves."""
    ports: dict[str, dict[str, Any]] = {}
    for p, s in (strings or {}).items():
        if len(s) != domains[p].count * BC_BYTES:
            raise ValueError(f"{p}: {len(s)} bytes of strings, not {domains[p].count} rows' b || c")
        ports[p] = {"kind": "string", "count": domains[p].count, "domain_id": domains[p].domain_id, "strings": bytes(s),
                    "schema": _part((schemas or {}).get(p, ROW_SCHEMA).encode())}
    for p, r in rows.items():
        r = np.ascontiguousarray(r).astype("<u2")
        s = (schemas or {}).get(p, ROW_SCHEMA)
        ports[p] = {"kind": "row", "count": domains[p].count, "domain_id": domains[p].domain_id, "rows": r.tobytes(),
                    "row_bytes": 2 * int(r.shape[1]), "salts": ys[p], "inner": _inner(s, int(r.shape[1])), "schema": _part(s.encode())}
    for p, w in words.items():
        ports[p] = {"kind": "word", "count": domains[p].count, "domain_id": domains[p].domain_id,
                    "words": np.ascontiguousarray(w).reshape(-1).astype(np.uint16), "schema": _part(WORD_SCHEMA.encode())}
    _JOB.update(ports=ports, hiding=HidingLeaves(instance=INST), salt_h=hashlib.sha512(INST.salt_prefix),
                leaf_h=hashlib.sha512(INST.leaf_prefix(INST.default_key)))
    tasks = []
    for p, j in ports.items():
        w = _width(j["count"])
        tasks += [(p, s, min(chunk, w)) for s in range(0, w, min(chunk, w))]
    try:
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                res = pool.map(_chunk, tasks, chunksize=1)
        else:
            res = [_chunk(t) for t in tasks]
    finally:
        _JOB.clear()
    roots: dict[str, bytes] = {}
    bc: dict[str, bytes] = {}
    by: dict[str, list[tuple[int, bytes, bytes]]] = {}
    for q, s, node, b in res:
        by.setdefault(q, []).append((s, node, b))
    for p, j in ports.items():
        mine = sorted(by.get(p, ()))
        size = min(chunk, _width(j["count"]))
        roots[p] = _Frame(j["domain_id"]).reduce([node for _, node, _ in mine], 0, (size - 1).bit_length())
        if j["kind"] == "row":
            bc[p] = b"".join(b for _, _, b in mine)
        elif j["kind"] == "string":
            bc[p] = j["strings"]
    return roots, bc


def _wire(m: Member, port: str) -> tuple[str, str, tuple[int, ...]]:
    return next(w for w in m.wired if w[0] == port)


def _by_reference(members: Sequence[Member], bases: np.ndarray, src: Any, ys: Mapping[str, bytes],
                  bc: Mapping[str, bytes]) -> tuple[bytes, bytes]:
    """A port given by reference (`commit`): its rows' salts and ``b || c``, its source's (a `ValueRows`' value's, or a wired
    port's writers' output rows', from their committed ``ys`` and ``bc``); ``bases``, the members' (`wiring.bases`)."""
    if isinstance(src, ValueRows):
        v = src.value
        return (value_salts(v, src.positions) if src.salts is None else src.salts), b"".join(bytes(v.bcs[i]) for i in src.positions)
    m, port = src
    _p, o, ws = _wire(m, port)
    y, s = np.empty((m.n, SALT_BYTES), dtype=np.uint8), np.empty((m.n, BC_BYTES), dtype=np.uint8)
    for wm, mine, theirs in WI._writers(members, ws, bases):
        y[mine] = np.frombuffer(ys[wm.port(o)], dtype=np.uint8).reshape(wm.n, SALT_BYTES)[theirs]
        s[mine] = np.frombuffer(bc[wm.port(o)], dtype=np.uint8).reshape(wm.n, BC_BYTES)[theirs]
    return y.tobytes(), s.tobytes()


class _Rows(Mapping):
    """A commit's rows by port (`commit`): the arrays given, and each port given by reference read off its source when it is
    read (a value's rows, or a wired port's writers' output rows), never held."""

    def __init__(self, members: Sequence[Member], rows: Mapping[str, np.ndarray], refd: Mapping[str, Any], order: tuple[str, ...]):
        self._members, self._rows, self._refd, self._order = members, rows, refd, order
        self._bases = WI.bases(members)

    def __getitem__(self, port: str) -> np.ndarray:
        if port in self._rows:
            return self._rows[port]
        src = self._refd[port]
        if isinstance(src, ValueRows):
            return value_rows(src.value, src.positions)
        m, p = src
        _p, o, ws = _wire(m, p)
        out = np.empty((m.n, dict(m.rows)[p]), dtype=np.uint16)
        for wm, mine, theirs in WI._writers(self._members, ws, self._bases):
            out[mine] = self[wm.port(o)][theirs]
        return out

    def __iter__(self) -> Any:
        return iter(self._order)

    def __len__(self) -> int:
        return len(self._order)


def _table_rows(window: Window, rows: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Each row table's rows, read off its linked members' ``x`` tables; refused when two members read one row differently."""
    out = [np.zeros((t.n, t.words), dtype=np.uint16) for t in window.row_tables]
    seen = [np.zeros(t.n, dtype=bool) for t in window.row_tables]
    for m in window.members:
        x = rows[m.port(LINKED)] if m.links else None
        for j, (t, r) in enumerate(m.links):
            if seen[t][r] and not np.array_equal(out[t][r], x[j]):
                raise ValueError(f"{m.key}: {LINKED} row {j} is not {window.row_tables[t].port} row {r} as another member reads it")
            out[t][r], seen[t][r] = x[j], True
    return out


def commit(window: Window, data: Sequence[dict[str, dict[str, np.ndarray]]], *, salt_key: bytes | None = None,
           row_salt_key: bytes | None = None, segment_roots: Mapping[int, bytes] | None = None, workers: int = 1,
           chunk: int = CHUNK, salts_of: Mapping[str, bytes] | None = None) -> Committed:
    """Commit every member's rows (``data[k]["rows"][port]``: (n, words) u16) and output words (``data[k]["outs"][port]``, each
    port's committed as its output row); and each row table, its rows read off its linked members' ``x`` tables, whose rows take
    their row table row's salt.  Salt nonces: the members' ports in order; a segment's tables', ``segment_salts`` under
    ``row_salt_key``.  The row tables' roots make each segment's root and the ``row-table`` port is the tree over those; with
    ``segment_roots`` (what the integration recorded before each segment's coin, by segment), each must be the window's, or the
    commit is refused.

    ``salt_key``: tests only; an integration draws 32 fresh OS bytes.  ``row_salt_key``: the run's row key the recorded segment
    roots were salted under (32 bytes); fresh OS bytes when ``None``, which no recorded segment root can match.
    ``salts_of``: by port, an outside registered value's salts, one a row, so its rows' ``b || c`` are the value's (`write`).

    An input port's rows may be given by reference, in a window with no row tables: `WIRED` for a wired port (its writers'
    output rows) or a `ValueRows` (rows of a registered value). Their rows are never held or hashed: each row's salt and
    ``b || c`` are its source's, so the commitment is the one their rows given in full make, and the committed rows (`_Rows`)
    are read off the sources when a body is written."""
    t0 = time.perf_counter()
    ms = window.members
    if len(data) != len(ms) or sum(m.n for m in ms) != window.population:
        raise ValueError(f"{sum(m.n for m in ms)} units in {len(ms)} members; the registered population is {window.population}")
    rows, refs, schemas, counts, refd = {}, {}, {}, {}, {}
    for k, (m, dk) in enumerate(zip(ms, data, strict=True)):
        count, wired = dict(m.tables), {p for p, _o, _ws in m.wired}
        for kind, ports in (("rows", m.rows), ("outs", m.outs)):
            if set(dk[kind]) != {p for p, _ in ports}:
                raise ValueError(f"{m.key}: {kind} {sorted(dk[kind])} are not {[p for p, _ in ports]}")
            for p, words in ports:
                src = dk[kind][p]
                want = (count.get(p, m.n) if kind == "rows" else m.n, words)
                schemas[m.port(p)], counts[m.port(p)] = m.schema(p), want[0]
                if kind == "rows" and (isinstance(src, ValueRows) or (isinstance(src, str) and src == WIRED)):
                    if isinstance(src, str) and p not in wired:
                        raise ValueError(f"{m.port(p)}: given as {WIRED} rows, but no wire writes it")
                    if isinstance(src, ValueRows):
                        v, have = src.value, len(src.value.rows)
                        why = [reason for reason, bad in (
                            (f"{len(src.positions)} rows, not {want[0]}", len(src.positions) != want[0]),
                            (f"rows of {v.words} words, not {words}", v.words != words),
                            (f"{v.word_bits}-bit leaves, not 16", v.word_bits != 16),
                            (f"{v.schema} rows, not {m.schema(p)}", v.schema != m.schema(p)),
                            (f"positions outside its {have} rows", not all(0 <= i < have for i in src.positions)),
                            (f"salts not one {SALT_BYTES}-byte salt a row",
                             src.salts is not None and len(src.salts) != SALT_BYTES * len(src.positions))) if bad]
                        if why:
                            raise ValueError(f"{m.port(p)}: a registered value's rows given by reference: {'; '.join(why)}")
                    refd[m.port(p)] = src if isinstance(src, ValueRows) else (m, p)
                    continue
                a = np.asarray(src, dtype=np.uint16)
                blocks = kind == "rows" and m.schema(p) == ROW_SCHEMA
                if a.shape != want or (blocks and (2 * words) % 128):
                    raise ValueError(f"{m.port(p)}: {a.shape} is not {want[0]} x {words}" + (" words of whole SHA-512 blocks" if blocks else ""))
                rows[m.port(p)] = a
                schemas[m.port(p)] = m.schema(p)
        for (p, w), (_, padded) in zip(m.outs, m.out_rows, strict=True):
            row = np.zeros((m.n, padded), dtype=np.uint16)
            row[:, :w] = rows[m.port(p)]
            rows[m.port(p)] = row
        if m.tables:
            r = np.asarray(dk["refs"], dtype=np.int64)
            bound = np.array([count[p] for p, _ in m.rows])
            if r.shape != (m.n, len(m.rows)) or (r < 0).any() or (r >= bound).any():
                raise ValueError(f"{m.key}: refs {r.shape} are not {m.n} x {len(m.rows)} rows inside tables {dict(m.tables)}")
            own = np.repeat(np.arange(m.n, dtype=np.int64)[:, None], len(m.outs), axis=1)
            refs[k] = np.concatenate([r, own], axis=1).astype(np.uint32)
    if segment_roots is not None and row_salt_key is None:
        raise ValueError("recorded segment roots are checked under the run's row key: give row_salt_key")
    if any(p not in rows or len(s) != len(rows[p]) * SALT_BYTES for p, s in (salts_of or {}).items()):
        raise ValueError(f"salts_of {sorted(salts_of)}: not one {SALT_BYTES}-byte salt for each row of one of the window's ports "
                         "given in full")
    if refd and window.row_tables:
        raise ValueError(f"{len(refd)} ports given by reference ({sorted(refd)[:3]}) in a window with row tables, whose linked "
                         "members' rows are read off their x tables: give every row")
    key = os.urandom(32) if salt_key is None else bytes(salt_key)
    ys = {p: (salts_of or {}).get(p) or salts(key, k, n) for k, (p, n) in enumerate(counts.items()) if p not in refd}
    del key
    domains = {p: window.domain(p, schemas[p], n) for p, n in counts.items()}
    rk = os.urandom(32) if row_salt_key is None else bytes(row_salt_key)
    for k, (t, a) in enumerate(zip(window.row_tables, _table_rows(window, rows), strict=True)):
        rows[t.port], schemas[t.port], domains[t.port] = a, TABLE_SCHEMA, window.table_domain(k)
    for s, ks in window.segments:
        ss, at = segment_salts(rk, s, sum(window.row_tables[k].n for k in ks)), 0
        for t in (window.row_tables[k] for k in ks):
            ys[t.port], at = ss[at:at + t.n * SALT_BYTES], at + t.n * SALT_BYTES
    del rk
    for m in ms:
        if m.links:
            ys[m.port(LINKED)] = b"".join(ys[window.row_tables[t].port][r * SALT_BYTES:(r + 1) * SALT_BYTES] for t, r in m.links)
    WI.salts(ms, rows, ys, skip=refd)
    t_salt = time.perf_counter()
    roots, bc = commit_ports(domains, rows, {}, ys, workers=workers, chunk=chunk, schemas=schemas)
    if refd:
        strings, at = {}, WI.bases(ms)
        for p, src in refd.items():
            ys[p], strings[p] = _by_reference(ms, at, src, ys, bc)
        more, _ = commit_ports(domains, {}, {}, {}, workers=workers, chunk=chunk, schemas=schemas, strings=strings)
        roots |= more
        roots = {p: roots[p] for p in counts}
        bc |= strings
        rows = _Rows(ms, rows, refd, tuple(counts))
    table_roots = {t.port: roots.pop(t.port) for t in window.row_tables}
    segs = {s: window.segment_root(s, [table_roots[window.row_tables[k].port] for k in ks]) for s, ks in window.segments}
    if segment_roots is not None:
        bad = [s for s in segs if segs[s] != segment_roots.get(s)]
        if bad:
            raise ValueError(f"segments {bad}: the window's segment roots are not the ones recorded before their coins")
    if window.row_tables:
        roots[TABLE_PORT], domains[TABLE_PORT] = RT.RootTree(window.tables_domain, list(segs.values())).root, window.tables_domain
    t1 = time.perf_counter()
    return Committed(window, roots, domains, bc, ys, rows, {"salts_s": round(t_salt - t0, 3), "commit_s": round(t1 - t_salt, 3),
                                                            "total_s": round(t1 - t0, 3), "workers": workers}, refs, table_roots, segs,
                     refd)


# --- the files ----------------------------------------------------------------------------------------------------------

def port_leaves(m: Member) -> dict[str, int]:
    """Each of member ``m``'s row ports (outputs included) and its leaf count: its table's size for a shared-row input, else ``n``."""
    tables = dict(m.tables)
    return {p: tables.get(p, m.n) for p, _ in m.all_rows}


def expected_ports(window: Window) -> dict[str, dict[str, Any]]:
    """The ports the verifier's own window derives, as ``registration.check`` takes them: ``{<member>/<port>: {leaves,
    domain}}``, each domain ``window_domain`` (core's ``registration.window_domain``) over the window's program, partition and run."""
    prog, part, segs = window.partition["program"], window.partition_digest, len(window.segments)
    return {m.port(p): {"leaves": n, "domain": window_domain(prog, part, m.port(p), m.schema(p), n, window.run)}
            for m in window.members for p, n in port_leaves(m).items()} | \
        ({TABLE_PORT: {"leaves": segs, "domain": window_domain(prog, part, TABLE_PORT, ROOTS_SCHEMA, segs, window.run)}}
         if segs else {})


def window_bytes(members: Sequence[Member], row_tables: Sequence[RowTable] = ()) -> dict[str, int]:
    """The bytes a window commits, from its layout alone: ``public`` (every ``b || c``, and a shared-row member's refs, u32 per
    instance and port), ``private`` (every row's words and salt), and the ``x`` ports' share of each (``x_public``, ``x_private``;
    a row table's rows count as ``x``)."""
    out = dict.fromkeys(("public", "private", "x_public", "x_private", "tables_public", "tables_private"), 0)

    def add(port: str, rows: int, words: int) -> None:
        pub, priv = rows * BC_BYTES, rows * (2 * words + SALT_BYTES)
        out["public"] += pub
        out["private"] += priv
        if port == LINKED:
            out["x_public"] += pub
            out["x_private"] += priv

    for m in members:
        leaves = port_leaves(m)
        for p, w in m.all_rows:
            add(p, leaves[p], w)
        if m.tables:
            out["public"] += 4 * m.n * len(m.all_rows)
    for t in row_tables:
        add(LINKED, t.n, t.words)
        out["tables_public"] += t.n * BC_BYTES
        out["tables_private"] += t.n * (2 * t.words + SALT_BYTES)
    return out


def leaf_layer(publics: Mapping[str, bytes]) -> str:
    """``LEAF_LAYER`` over ``{member: public body}``."""
    return hashlib.sha512(canonical_json({t: hashlib.sha512(b).hexdigest() for t, b in publics.items()})).hexdigest()


def registration(c: Committed, provenance: Mapping[str, Any], previous: str | None = None, *, kind: str = "served",
                 registrar: Any) -> dict[str, Any]:
    """Core's ``verity/registration/v1`` record of the window, made by ``registrar`` (core's ``one_stage.registration``: its
    ``record``, ``KINDS``): its program, query and law, the scheme, every port's root, leaves and window domain, the leaf layer,
    and ``window = {"kind": kind, "run": <the record's run root>, **provenance}``.  ``served``: these roots are the integration's
    own commitment as it ran, made before any draw; ``stand-in``: a fresh commitment to values opened from its record afterwards,
    whose link to the record's roots is provenance."""
    if kind not in registrar.KINDS:
        raise ValueError(f"window kind {kind!r} is not one of {registrar.KINDS}")
    w = c.window
    roots = [{"port": p, "root": c.roots[p].hex(), "leaves": c.domains[p].count, "domain": c.domains[p].domain_id.hex()} for p in c.roots]
    publics = {m.key: c.public_body(k) for k, m in enumerate(w.members)}
    return registrar.record(w.program, w.query, w.law, SCHEME, roots, {**provenance, "kind": kind, "run": w.run},
                            leaf_layer=leaf_layer(publics), previous=previous, work_table=w.work_table)


def units_doc(m: Member) -> dict[str, Any]:
    """Member ``m``'s public file of its global unit indices (``m<k>-units.json``): a consecutive run of its template's units."""
    return {"schema": UNITS_SCHEMA, "member": m.key, "template": m.template, "base": m.base, "n": m.n,
            "indices": list(range(m.base, m.base + m.n))}


def index_bytes(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, separators=(",", ":")) + "\n").encode()


def index_leaf(ib: bytes, salt: bytes) -> bytes:
    """``index.json``'s hiding commitment (``ib``, its bytes): the ``hm96-sha512/v1`` leaf of its tagged SHA-512 under ``salt``
    (192 bytes, `INDEX_RULE`), what the registration binds of the labels. An integration's labels may name what it served (a
    request, a token row), and over so few candidates an unsalted digest of them would be no commitment at all."""
    if len(salt) != SALT_BYTES:
        raise ValueError(f"an index's salt is {SALT_BYTES} bytes, not {len(salt)}")
    x = hashlib.sha512(INDEX_TAG + bytes(ib)).digest()
    return INST.tree_leaf(INST.default_key, INST.commit_string(INST.default_key, x, bytes(salt)))


def _private(path: Path) -> Any:
    """``path`` opened to write, readable by its owner alone, a rewrite over an older file too: a file only the prover reads."""
    f = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb")
    os.fchmod(f.fileno(), 0o600)
    return f


def registered_entry(c: Committed, m: Member, trees: dict[int, RT.RowTable] | None = None) -> dict[str, Any]:
    """Linked member ``m``'s public header ``registered`` entry: its ``x`` table's rows are rows of its row table (the value,
    named by the table's port), each at its rank with its path to the table's root (``RT.RowTable.path``), which the Lean
    verifier opens against the root its own ``--registered`` file names (``tables.registered_reads``).  ``trees``
    caches each table's tree by its index."""
    t, trees = m.links[0][0], {} if trees is None else trees
    tab = c.window.row_tables[t]
    if t not in trees:
        bc = c.bc[tab.port]
        trees[t] = RT.RowTable(c.window.table_domain(t), [bc[i * BC_BYTES:(i + 1) * BC_BYTES] for i in range(tab.n)])
    return {LINKED: {"value": tab.port, "positions": [r for _, r in m.links],
                     "paths": [trees[t].path(r).hex() for _, r in m.links]}}


def _names(entry: Mapping[str, Any] | None, src: ValueRows) -> bool:
    """Whether a member's header entry for a port names the rows ``src`` gives it: its value and positions."""
    return entry is not None and entry.get("value") == src.value.value and list(entry.get("positions", ())) == list(src.positions)


def _pathless(reads: Mapping[str, Mapping[str, Any]], paths: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """A member's header entries with their paths moved into ``paths`` (by value, then position), each path once for the
    window; refused where two entries give one row two paths."""
    out = {}
    for p, e in reads.items():
        table = paths.setdefault(e["value"], {})
        for x, h in zip(e["positions"], e["paths"], strict=True):
            if table.setdefault(str(x), h) != h:
                raise ValueError(f"{p}: row {x} of {e['value']!r} given two paths")
        out[p] = {f: v for f, v in e.items() if f != "paths"}
    return out


def registered_header(win: Mapping[str, Any], k: int) -> dict[str, Any] | None:
    """Member ``k``'s header ``registered`` entry (``{port: {"value", "positions", "paths"}}``) from its window's
    ``window.json`` (`write`), each path read off the window's ``registered_paths`` where the member's entry gives none."""
    reads = win["members"][k].get("registered")
    if not reads:
        return None
    table, out = win.get("registered_paths") or {}, {}
    for p, e in reads.items():
        if "paths" in e:
            out[p] = dict(e)
            continue
        got = table.get(e["value"], {})
        if missing := [x for x in e["positions"] if str(x) not in got]:
            raise ValueError(f"member {k} {p}: the window gives no path for rows {missing[:4]} of {e['value']!r}")
        out[p] = {**e, "paths": [got[str(x)] for x in e["positions"]]}
    return out


class Bodies:
    """A written window's prover bodies (`write`), each made from the files that hold its rows and salts: a member's own
    ``m<k>-private.bin`` (its body less its ports ``value_rows`` and ``wired_rows`` name, in the body's own order), a
    registered value's ``v<j>-private.bin`` (the rows the window reads, in position order, then their salts) and, for a wired
    port, its writers' own files (each row and salt its writer's output row's). Each file is read where a body reads it,
    memory-mapped; a file not the size its ports make, a row no file holds and a writer whose output isn't the port's rows
    are refused."""

    OPEN = 256  # files held memory-mapped at once: each map holds a file descriptor, and a process may have 1024

    def __init__(self, window_dir: str | Path, win: Mapping[str, Any]) -> None:
        self.dir, self.members = Path(window_dir), win["members"]
        self._win, self._own, self._values, self._wires = win, {}, {}, {}
        self._bases: np.ndarray | None = None

    def leaves(self, k: int) -> dict[str, int]:
        mem = self.members[k]
        shared = mem.get("shared_rows")
        return {p: int(shared[p]) if shared else int(mem["n"]) for p, _ in mem["ports"]}

    def left(self, k: int) -> set[str]:
        mem = self.members[k]
        return set(mem.get("value_rows") or ()) | set(mem.get("wired_rows") or ())

    def own(self, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Member ``k``'s ports its own file holds: each one's rows ((leaves, words) u16) and salts ((leaves, 192) u8)."""
        if k in self._own:
            return self._own[k]
        mem, left, leaves = self.members[k], self.left(k), self.leaves(k)
        mine, path = [(p, int(w)) for p, w in mem["ports"] if p not in left], self.dir / f"m{k}-private.bin"
        size, want = path.stat().st_size, sum(leaves[p] * (2 * w + SALT_BYTES) for p, w in mine)
        if size != want:
            raise ValueError(f"{path}: {size} bytes, not the {want} its ports leave")
        b, out = np.memmap(path, dtype=np.uint8, mode="r") if size else np.zeros(0, dtype=np.uint8), {}
        if mem.get("shared_rows"):
            at = 0
            for p, w in mine:
                out[p], at = [b[at:at + 2 * w * leaves[p]].view("<u2").reshape(leaves[p], w)], at + 2 * w * leaves[p]
            for p, _w in mine:
                out[p].append(b[at:at + SALT_BYTES * leaves[p]].reshape(leaves[p], SALT_BYTES))
                at += SALT_BYTES * leaves[p]
        elif mine:
            n, words = int(mem["n"]), sum(w for _, w in mine)
            rows, salts, col = b[:2 * n * words].view("<u2").reshape(n, words), b[2 * n * words:].reshape(n, -1), 0
            for j, (p, w) in enumerate(mine):
                out[p], col = [rows[:, col:col + w], salts[:, SALT_BYTES * j:SALT_BYTES * (j + 1)]], col + w
        self._room()
        self._own[k] = {p: (r, s) for p, (r, s) in out.items()}
        return self._own[k]

    def _room(self) -> None:
        if len(self._own) + len(self._values) >= self.OPEN:
            self._own.clear()
            self._values.clear()

    def value(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Registered value ``name``'s rows the window holds: their positions, rows ((P, words) u16) and salts ((P, 192) u8)."""
        if name not in self._values:
            v = self._win["value_rows"][name]
            at, w, path = np.asarray(v["positions"], dtype=np.int64), int(v["words"]), self.dir / v["file"]
            if path.stat().st_size != len(at) * (2 * w + SALT_BYTES):
                raise ValueError(f"{path}: {path.stat().st_size} bytes, not {len(at)} rows of {w} words and their salts")
            b = np.memmap(path, dtype=np.uint8, mode="r")
            self._room()
            self._values[name] = (at, b[:2 * w * len(at)].view("<u2").reshape(len(at), w),
                                  b[2 * w * len(at):].reshape(len(at), SALT_BYTES))
        return self._values[name]

    def take(self, k: int, p: str, words: int, at: slice, salts: bool = False) -> np.ndarray:
        """Rows ``at`` of member ``k``'s port ``p`` (``words`` words), or their salts, from the file that holds them."""
        mem = self.members[k]
        if p in (mem.get("wired_rows") or ()):
            return self._wired(k, p, words, at, salts)
        if p in (mem.get("value_rows") or ()):
            e = mem["registered"][p]
            pos, (have, rows, ys) = np.asarray(e["positions"], dtype=np.int64), self.value(e["value"])
            if len(pos) != self.leaves(k)[p] or rows.shape[1] != words:
                raise ValueError(f"member {k} {p}: {len(pos)} rows of {e['value']!r}, not {self.leaves(k)[p]} rows of {words} "
                                 f"words the window's {self._win['value_rows'][e['value']]['file']} holds")
            pos = pos[at]
            i = np.minimum(np.searchsorted(have, pos), len(have) - 1)
            if not np.array_equal(have[i], pos):
                raise ValueError(f"member {k} {p}: rows {pos[have[i] != pos][:4].tolist()} of {e['value']!r} are not rows of "
                                 f"{words} words the window's {self._win['value_rows'][e['value']]['file']} holds")
            return (ys if salts else rows)[i]
        rows, ys = self.own(k)[p]
        return (ys if salts else rows)[at]

    def _wired(self, k: int, p: str, words: int, at: slice, salts: bool) -> np.ndarray:
        if (k, p) not in self._wires:
            mem = self.members[k]
            wire = next((x for x in mem.get("wired") or () if x["port"] == p), None)
            if wire is None or len(wire["writers"]) != int(mem["n"]):
                raise ValueError(f"member {k} {p}: no wire gives its {mem['n']} rows a writer each")
            self._wires[k, p] = (wire["out"], np.asarray(wire["writers"], dtype=np.int64))
        if self._bases is None:
            self._bases = np.array([m["base"] for m in self.members], dtype=np.int64)
        o, ws = self._wires[k, p]
        w = ws[at]
        j = np.searchsorted(self._bases, w, side="right") - 1
        out = np.empty((len(w), SALT_BYTES if salts else words), dtype=np.uint8 if salts else np.uint16)
        for jj in np.unique(j).tolist():
            mask = j == jj
            wm = self.members[jj] if jj >= 0 else None
            theirs = w[mask] - (self._bases[jj] if wm else 0)
            outs = {q: int(x) for q, x in wm["ports"][int(wm["input_ports"]):]} if wm else {}
            held = self.own(jj).get(o) if outs.get(o) == words and theirs.min() >= 0 and theirs.max() < int(wm["n"]) else None
            if held is None:
                raise ValueError(f"member {k} {p}: its writers' {o!r} rows in member {jj} are not output rows of {words} words "
                                 "their own file holds")
            out[mask] = held[1 if salts else 0][theirs]
        return out

    def copy(self, k: int, f: Any) -> None:
        """Member ``k``'s prover body (`Committed.private_body`) into ``f``."""
        mem, left = self.members[k], self.left(k)
        if not left:
            with open(self.dir / f"m{k}-private.bin", "rb") as g:
                shutil.copyfileobj(g, f, VALUE_CHUNK)
            return
        ports, leaves = [(p, int(w)) for p, w in mem["ports"]], self.leaves(k)
        self.own(k)
        if mem.get("shared_rows"):
            for salts in (False, True):
                for p, w in ports:
                    step = max(1, VALUE_CHUNK // (SALT_BYTES if salts else 2 * w))
                    for lo in range(0, leaves[p], step):
                        piece = self.take(k, p, w, slice(lo, min(lo + step, leaves[p])), salts)
                        f.write(np.ascontiguousarray(piece, dtype=np.uint8 if salts else "<u2").tobytes())
            return
        n = int(mem["n"])
        for salts in (False, True):
            step = max(1, VALUE_CHUNK // (SALT_BYTES * len(ports) if salts else 2 * sum(w for _, w in ports)))
            for lo in range(0, n, step):
                at = slice(lo, min(lo + step, n))
                piece = np.concatenate([self.take(k, p, w, at, salts) for p, w in ports], axis=1)
                f.write(np.ascontiguousarray(piece, dtype=np.uint8 if salts else "<u2").tobytes())


def copy_private(window_dir: str | Path, win: Mapping[str, Any], k: int, f: Any) -> None:
    """Member ``k``'s prover body (`Committed.private_body`) from its window's files into ``f`` (`Bodies`)."""
    Bodies(window_dir, win).copy(k, f)


def read_private(window_dir: str | Path, win: Mapping[str, Any], k: int) -> bytes:
    """Member ``k``'s prover body from its window's files (`Bodies`)."""
    f = io.BytesIO()
    Bodies(window_dir, win).copy(k, f)
    return f.getvalue()


def write(c: Committed, out_dir: str | Path, provenance: dict[str, Any], index: dict[str, Any], extra: dict[str, Any] | None = None, *,
          kind: str = "served", registered: Mapping[str, dict[str, Any]] | None = None, registrar: Any,
          index_salt: bytes | None = None) -> dict[str, Any]:
    """The window's files, its registration's window of ``kind`` (`registration`, by ``registrar``, which also names and digests
    the record: ``FORMAT``, ``digest``); the window's provenance gets the index's hiding leaf (`index_leaf`), so the
    registration binds the unit -> label map and says nothing of it. ``index.json`` and its salt are prover only: the salt is
    ``index_salt`` (192 bytes, a committer's drawn under its key) or else fresh bytes from the OS.  ``registered``: by member
    key, an outside value's reads (``salts_of``), each a header entry
    (``{port: {"value", "positions", "paths"}}``), whose paths ``window.json`` gives once for the window
    (``registered_paths``, by value and position; `registered_header`).  A member's ``m<k>-private.bin`` is its prover body
    less two kinds of port, rows and salts: a port given as a registered value's rows (`Committed.by_value`) its entry names
    (``value_rows``), whose rows the window reads are written once for each value, in position order and then their salts,
    to ``v<j>-private.bin`` (``window.json``'s ``value_rows``); and a wired port (``wired_rows``), each row and salt its
    writer's output row's, which the writer's own file holds.  At the 4,32,4 class the members' bodies read 5,282,498 value
    rows, 37,787 of them distinct, and 9,657,548 wired rows of 3,750,680 output rows.  `Bodies` makes a member's prover
    body from the files."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ib = index_bytes(index)
    salt = os.urandom(SALT_BYTES) if index_salt is None else bytes(index_salt)
    leaf = index_leaf(ib, salt).hex()
    with _private(d / "index.json") as f:
        f.write(ib)
    with _private(d / INDEX_SALT) as f:
        f.write(salt)
    rec = registration(c, {**provenance, "index_leaf": leaf, "order": ORDER}, kind=kind, registrar=registrar)
    (d / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    trees: dict[int, RT.RowTable] = {}
    members, paths, read = [], {}, {}
    for k, m in enumerate(c.window.members):
        reads = registered_entry(c, m, trees) if m.links else (registered or {}).get(m.key)
        left = {p: s for p, s in c.by_value(k).items() if reads and _names(reads.get(p), s)}
        wired = {p for p, _o, _ws in m.wired}
        pub = c.public_body(k)
        (d / f"m{k}-public.bin").write_bytes(pub)
        with _private(d / f"m{k}-private.bin") as f:
            for piece in c.private_pieces(k, left.keys() | wired):
                f.write(piece)
        for p, s in left.items():
            v, at, ys = read.setdefault(s.value.value, (s.value, [], []))
            if v is not s.value:
                raise ValueError(f"{m.key}: two registered values named {s.value.value!r}")
            at.append(np.asarray(s.positions, dtype=np.int64))
            ys.append(np.frombuffer(c.salts[m.port(p)], dtype=np.uint8).reshape(-1, SALT_BYTES))
        ub = index_bytes(units_doc(m))
        (d / f"m{k}-units.json").write_bytes(ub)
        members.append({"k": k, "name": m.key, "template": m.template, "n": m.n, "base": m.base, "ports": [list(p) for p in m.all_rows],
                        "input_ports": len(m.rows), "out_ports": [list(p) for p in m.outs], "frame_v3": c.frame_entry(k),
                        "public_sha512": hashlib.sha512(pub).hexdigest(), "units": f"m{k}-units.json",
                        "units_sha512": hashlib.sha512(ub).hexdigest(),
                        **({"row_schemas": m.row_schemas} if m.schemas else {}),
                        **({"registered": dict(reads) if m.links else _pathless(reads, paths)} if reads else {}),
                        **({"value_rows": [p for p, _ in m.rows if p in left]} if left else {}),
                        **({"wired_rows": [p for p, _ in m.rows if p in wired]} if wired else {}),
                        **({"shared_rows": {**dict(m.tables), **{p: m.n for p, _ in m.out_rows}}} if m.tables else {}),
                        **({"row_links": [list(x) for x in m.links]} if m.links else {}),
                        **({"wired": [{"port": p, "out": o, "writers": list(ws)} for p, o, ws in m.wired]} if m.wired else {})})
    values = {}
    for j, (name, (v, ats, yss)) in enumerate(sorted(read.items())):
        at, step = np.unique(np.concatenate(ats)), max(1, VALUE_CHUNK // (2 * v.words))
        held = np.empty((len(at), SALT_BYTES), dtype=np.uint8)
        for pos, ys in zip(ats, yss, strict=True):
            held[np.searchsorted(at, pos)] = ys
        for pos, ys in zip(ats, yss, strict=True):
            if not np.array_equal(held[np.searchsorted(at, pos)], ys):
                raise ValueError(f"{name!r}: a row two members read is committed under two salts")
        with _private(d / f"v{j}-private.bin") as f:
            for lo in range(0, len(at), step):
                f.write(np.ascontiguousarray(value_rows(v, at[lo:lo + step]), dtype="<u2").tobytes())
            f.write(held.tobytes())
        values[name] = {"words": v.words, "positions": at.tolist(), "file": f"v{j}-private.bin"}
    tables = []
    for t, tab in enumerate(c.window.row_tables):
        pub = c.table_public(t)
        (d / f"t{t}-public.bin").write_bytes(pub)
        with _private(d / f"t{t}-private.bin") as f:
            f.write(c.table_private(t))
        dom = c.domains[tab.port]
        tables.append({"k": t, **tab.doc(), "step_index": c.window.places[t], "root": c.table_roots[tab.port].hex(),
                       "domain": dom.domain_id.hex(), "binding": dom.binding.hex(), "public": f"t{t}-public.bin",
                       "public_sha512": hashlib.sha512(pub).hexdigest(), "private": f"t{t}-private.bin"})
    segs = {"context": c.window.context, "port": TABLE_PORT, "root_schema": ROOTS_SCHEMA,
            "steps": [{SEGMENT_KEY: s, "tables": list(ks), "root": c.segment_roots[s].hex()} for s, ks in c.window.segments],
            "salt_rule": "chacha20 (rfc 8439) under the run's 32-byte row key, nonce = the segment (12 bytes le), the segment's "
            "tables in order and each table's rows in rank order at consecutive 192-byte offsets"}
    win = {"schema": "verity-vllm/serving-rows/v2", "scheme": SCHEME, "population": c.window.population, "members": members,
           **({"row_tables": tables, "row_table_steps": segs} if tables else {}),
           **({"registered_paths": paths} if paths else {}), **({"value_rows": values} if values else {}),
           "partition": c.window.partition, "partition_digest": c.window.partition_digest, "run": c.window.run,
           "domain_tag": DOMAIN_TAG, "registration": registrar.FORMAT, "registration_digest": registrar.digest(rec),
           "leaf_layer": rec["leaf_layer"],
           "leaf_layer_rule": LEAF_LAYER, "index_leaf": leaf, "index_rule": INDEX_RULE,
           "salt_rule": "chacha20 (rfc 8439), a fresh 32-byte os key per window, nonce = the row port's index over all members "
           "(12 bytes le), row i at stream offset 192 i; the key is dropped after expansion", "seconds": c.seconds, **(extra or {})}
    (d / "window.json").write_text(json.dumps(win, indent=1, default=str) + "\n")
    return win

