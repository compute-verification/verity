"""Serving rows: the opt-in ``frame-v3-sha512`` + ``hm96-sha512/v1`` commitment of a run's units in M0's serving row leaf format
(``verity_flock.circuit``'s statement rows, PR #83), with the registration, M0's public and prover files and the unit index the
one-stage audit reads (``verity.core.protocols.sampled_proofs.one_stage``).  Off by default: ``vllm-v1`` stays the record, and this commits beside it without
touching any root of record, until the re-baseline switches the record.

The population is the verifier's: its program is ``TemplateInstances`` / ``TemplatePopulations`` over the members' templates and
counts (``population_program``, core's ``partition.population_program`` / ``mixed_population_program``), its query the template
query over them, so the partition (``verity/partition/v1``) and its digest are core's over that program, and the registration is
core's ``verity/registration/v1`` record (``registration.record``) with ``window = {"kind": "served", "run": <vllm-v1 run root>,
...provenance}``.  The served Programs' digests are provenance in the window: no query of core's selects template instances inside a
served Program (the template queries need every root node to be an instance).

A population is a sequence of members (:class:`Member`), one per template in the partition's order, each with its row ports and
output ports in M0's order.  Each output port is a row too, after the member's input rows: its words zero-padded to whole SHA-512
blocks, committed under its own salt like an input row (an output is a value: no output word is ever public).  Per row
(``tests/commit/test_serving_rows.py`` pins it to core and to M0's vector):

  row     the words, u16 little-endian
  inner   ``x = SHA-512(sha512_row_prefix(role 1, 16, n) || row)``                              (``sha512/row/v1``)
  salt    ``y`` = 192 bytes of ChaCha20 (RFC 8439) under the window's salt key, nonce = the row port's index over all members (12
          bytes LE), at stream offset ``192 i``; the key is 32 fresh bytes from the OS generator per window, dropped once the
          salts are expanded.  The salts are kept only in the prover's files and never logged
  string  ``b || c = hm96.SHA512.commit_string(default_key, x, y)``                               (public, 128 bytes)
  leaf    the ``frame-v3-sha512`` row leaf, schema ``hm96-sha512/row/v1``, over ``hm96.SHA512.tree_leaf(default_key, b || c)``

Each port ``<descriptor id>/<port>`` is one tree under ``CommitmentDomain(binding, -1, [0, leaves), "sha512")``, with core's
served-domain rule (``registration.served_domain``): ``binding = identity_digest_sha512(DOMAIN_TAG, {program, partition, port,
schema, leaves, run})``, every field a registration field and ``run`` the vllm-v1 run root, so the verifier recomputes every domain
from the registration and the Commit's record (``expected_ports``).  The leaf layer is the SHA-512 of the canonical map from each
member's template to the SHA-512 of its public body (``leaf_layer``).  ``write`` puts into a window directory ``registration.json``,
``index.json`` (unit -> served identity, its SHA-512 in the window's provenance), per member ``m<k>-public.bin`` and
``m<k>-private.bin`` (M0's public and prover bodies; the private one is prover only) and ``window.json``; ``flock_input_files`` wraps
member ``k``'s bodies as M0's ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, sha512_row_prefix
from verity.ml.circuits.codec import canonical_json

from .hiding import HidingLeaves

SCHEME = "frame-v3-sha512/hm96-sha512"
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW
WORD_SCHEMA = "u16"
DOMAIN_TAG = R.SERVED_DOMAIN
OWNER = -1
ORDER = "verity-vllm/serving-rows/order/v0"
LEAF_LAYER = "SHA-512 of canon({template: SHA-512 of m<k>-public.bin}) over the window's members"
INDEX_SCHEMA = "verity-vllm/serving-rows/index/v0"
INST = hm96.SHA512
SALT_BYTES = INST.salt_bytes
BC_BYTES = INST.proven_bytes
CHUNK = 1 << 14
#: M0's ``flock-circuit-inputs`` header constants (``verity_flock.circuit``) and the META keys a class leaves out
M0_FORMAT, M0_STATEMENT, M0_BINDING_KEYS = "flock-circuit-inputs", "verity/flock-circuit", ("partition", "program_sha512", "program_digests")

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
    """Ranks [start, start + size) of one port (pads past its count) folded to one node; and, for a row port, their ``b || c``."""
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
        x = b"".join(_copy_update(j["inner"], rows[i * rb:(i + 1) * rb]) for i in range(m))
        b = memoryview((np.frombuffer(x, dtype=np.uint8).reshape(m, 64) ^ _JOB["hiding"].masks(ys, m)).tobytes())
        yv, out = memoryview(ys), bytearray()
        salt_h, leaf_h, schema = _JOB["salt_h"], _JOB["leaf_h"], j["schema"]
        for i in range(m):
            s = bytes(b[i * 64:(i + 1) * 64]) + _copy_update(salt_h, yv[i * SALT_BYTES:(i + 1) * SALT_BYTES])
            out += s
            leaves.append(fr.leaf(start + i, schema, _copy_update(leaf_h, s)))
        bc = bytes(out)
    elif j["kind"] == "word" and hi > start:
        w = j["words"][start:hi].astype(">u2").tobytes()
        leaves = [fr.leaf(start + i, j["schema"], w[2 * i:2 * i + 2]) for i in range(hi - start)]
    leaves += [fr.pad(r) for r in range(max(start, n), start + size)]
    return port, start, fr.reduce(leaves, start), bc


# --- the window ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Member:
    """One template of the population: its descriptor id, row and output ports ``(name, words)`` in M0's order, its instance count
    and the global index of its first unit."""
    template: str
    rows: tuple[tuple[str, int], ...]
    outs: tuple[tuple[str, int], ...]
    n: int
    base: int = 0
    tables: tuple[tuple[str, int], ...] = ()     # a shared-row member: each row port's table size (rows are referenced by refs)

    @property
    def out_rows(self) -> tuple[tuple[str, int], ...]:
        """Each output port's row: its words zero-padded to whole SHA-512 blocks (64 words each), as M0's ``out_row``."""
        return tuple((p, -(-w // 64) * 64) for p, w in self.outs)

    @property
    def all_rows(self) -> tuple[tuple[str, int], ...]:
        """M0's rows: the input rows, then one output row per output port."""
        return self.rows + self.out_rows

    def port(self, name: str) -> str:
        """The name a root, its domain and the registration use: ``<descriptor id>/<port>``."""
        return f"{self.template}/{name}"


def population_program(templates: Sequence[tuple[Any, int]]) -> tuple[Any, dict[str, Any]]:
    """The verifier's population program and query over ``(template Definition, instances)`` in order: ``TemplateInstances`` and
    ``Q_template_instance`` for one template, ``TemplatePopulations`` and ``Q_template_instances`` for several."""
    if len(templates) == 1:
        (t, n), = templates
        return P.population_program(t, n), P.template_instance_query(t)
    return P.mixed_population_program(list(templates)), P.template_instances_query([t for t, _ in templates])


@dataclass(frozen=True)
class Window:
    """What the roots are bound to: the verifier's population program and query (the partition over them), the law it draws by, the
    run (the vllm-v1 run root, hex), and the members the population is made of, in the partition's order (refused otherwise)."""
    program: Any
    query: dict[str, Any]
    law: dict[str, Any]
    run: str
    members: tuple[Member, ...]

    def __post_init__(self) -> None:
        mine = [(m.template, m.base, m.n) for m in self.members]
        if P.ranges(self.program, self.query) != mine:
            raise ValueError(f"members {mine} are not the partition's units {P.ranges(self.program, self.query)}")

    @cached_property
    def partition(self) -> dict[str, Any]:
        return P.partition(self.program, self.query)

    @cached_property
    def partition_digest(self) -> str:
        return P.digest(self.partition)

    @cached_property
    def population(self) -> int:
        return P.units(self.program, self.query)

    def binding(self, port: str, schema: str, leaves: int) -> bytes:
        return identity_digest_sha512(DOMAIN_TAG, {"program": self.partition["program"], "partition": self.partition_digest,
                                                   "port": port, "schema": schema, "leaves": int(leaves), "run": self.run})

    def domain(self, port: str, schema: str, leaves: int) -> CommitmentDomain:
        return CommitmentDomain(self.binding(port, schema, leaves), OWNER, RangeIndexedDomain(0, leaves), hash="sha512")


@dataclass
class Committed:
    window: Window
    roots: dict[str, bytes]            # by <descriptor id>/<port>
    domains: dict[str, CommitmentDomain]
    bc: dict[str, bytes]               # per row port: every row's b || c, concatenated
    salts: dict[str, bytes]            # per row port: every row's salt (private)
    rows: dict[str, np.ndarray]        # per row port, the output rows included: (n, words) u16 (private)
    seconds: dict[str, Any]
    refs: dict[int, np.ndarray] = field(default_factory=dict)   # per shared-row member k: (n, rows) table refs, output rows' own

    def public_body(self, k: int) -> bytes:
        """Member ``k``'s public body, as M0's ``write()`` lays it out: every instance's ``b || c`` (its input rows, then its
        output rows), instance-major; nothing else."""
        m = self.window.members[k]
        if m.tables:                     # M0's shared-row body: every table row's b || c (port-major), refs (u32, instance-major)
            return b"".join(self.bc[m.port(p)] for p, _ in m.all_rows) + np.ascontiguousarray(self.refs[k]).astype("<u4").tobytes()
        bc = np.concatenate([np.frombuffer(self.bc[m.port(p)], dtype=np.uint8).reshape(m.n, BC_BYTES) for p, _ in m.all_rows], axis=1)
        return bc.tobytes()

    def private_body(self, k: int) -> bytes:
        """Member ``k``'s prover body: every instance's rows (ports in order), then every salt (instance-major, ports in order)."""
        m = self.window.members[k]
        if m.tables:                     # the tables' rows, then their salts, port-major
            return b"".join(np.ascontiguousarray(self.rows[m.port(p)]).astype("<u2").tobytes() for p, _ in m.all_rows) + \
                b"".join(self.salts[m.port(p)] for p, _ in m.all_rows)
        rows = np.concatenate([self.rows[m.port(p)] for p, _ in m.all_rows], axis=1).astype("<u2").tobytes()
        ys = np.concatenate([np.frombuffer(self.salts[m.port(p)], dtype=np.uint8).reshape(m.n, SALT_BYTES) for p, _ in m.all_rows], axis=1)
        return rows + ys.tobytes()

    def frame_entry(self, k: int) -> dict[str, Any]:
        """M0's header ``frame_v3`` entry for member ``k``: its bare port names, as M0's statement reads them."""
        m = self.window.members[k]
        names = [p for p, _ in m.all_rows]
        return {"frame": "frame-v3-sha512", "roots": {p: self.roots[m.port(p)].hex() for p in names},
                "domain_ids": {p: self.domains[m.port(p)].domain_id.hex() for p in names},
                "bindings": {p: self.domains[m.port(p)].binding.hex() for p in names}, "owner": OWNER,
                "schemas": {p: ROW_SCHEMA for p in names}, "row_key_sha512": hashlib.sha512(INST.default_key).hexdigest()}


def commit_ports(domains: dict[str, CommitmentDomain], rows: dict[str, np.ndarray], words: dict[str, np.ndarray], ys: dict[str, bytes], *,
                 workers: int = 1, chunk: int = CHUNK) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Roots of row ports (``rows[p]``: (count, words) u16, salts ``ys[p]``) and word ports (``words[p]``: u16, flattened), each
    under its domain; and every row's ``b || c``.  Equal to core's ``FrameV3.root`` over ``row_leaf`` / word leaves."""
    ports: dict[str, dict[str, Any]] = {}
    for p, r in rows.items():
        r = np.ascontiguousarray(r).astype("<u2")
        ports[p] = {"kind": "row", "count": domains[p].count, "domain_id": domains[p].domain_id, "rows": r.tobytes(),
                    "row_bytes": 2 * int(r.shape[1]), "salts": ys[p], "inner": hashlib.sha512(sha512_row_prefix(ROLE_X, 16, int(r.shape[1]))),
                    "schema": _part(ROW_SCHEMA.encode())}
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
    for p, j in ports.items():
        mine = sorted((s, node, b) for q, s, node, b in res if q == p)
        size = min(chunk, _width(j["count"]))
        roots[p] = _Frame(j["domain_id"]).reduce([node for _, node, _ in mine], 0, (size - 1).bit_length())
        if j["kind"] == "row":
            bc[p] = b"".join(b for _, _, b in mine)
    return roots, bc


def commit(window: Window, data: Sequence[dict[str, dict[str, np.ndarray]]], *, salt_key: bytes | None = None, workers: int = 1,
           chunk: int = CHUNK) -> Committed:
    """Commit every member's rows (``data[k]["rows"][port]``: (n, words) u16) and output words (``data[k]["outs"][port]``, each
    port's committed as its output row).

    ``salt_key``: tests only; serving draws 32 fresh OS bytes."""
    t0 = time.perf_counter()
    ms = window.members
    if len(data) != len(ms) or sum(m.n for m in ms) != window.population:
        raise ValueError(f"{sum(m.n for m in ms)} units in {len(ms)} members; the registered population is {window.population}")
    rows, refs = {}, {}
    for k, (m, dk) in enumerate(zip(ms, data, strict=True)):
        count = dict(m.tables)
        for kind, ports in (("rows", m.rows), ("outs", m.outs)):
            if set(dk[kind]) != {p for p, _ in ports}:
                raise ValueError(f"{m.template}: {kind} {sorted(dk[kind])} are not {[p for p, _ in ports]}")
            for p, words in ports:
                a = np.asarray(dk[kind][p], dtype=np.uint16)
                want = (count.get(p, m.n) if kind == "rows" else m.n, words)
                if a.shape != want or (kind == "rows" and (2 * words) % 128):
                    raise ValueError(f"{m.port(p)}: {a.shape} is not {want[0]} x {words}" + (" words of whole SHA-512 blocks" if kind == "rows" else ""))
                rows[m.port(p)] = a
        for (p, w), (_, padded) in zip(m.outs, m.out_rows, strict=True):
            row = np.zeros((m.n, padded), dtype=np.uint16)
            row[:, :w] = rows[m.port(p)]
            rows[m.port(p)] = row
        if m.tables:
            r = np.asarray(dk["refs"], dtype=np.int64)
            bound = np.array([count[p] for p, _ in m.rows])
            if r.shape != (m.n, len(m.rows)) or (r < 0).any() or (r >= bound).any():
                raise ValueError(f"{m.template}: refs {r.shape} are not {m.n} x {len(m.rows)} rows inside tables {dict(m.tables)}")
            own = np.repeat(np.arange(m.n, dtype=np.int64)[:, None], len(m.outs), axis=1)
            refs[k] = np.concatenate([r, own], axis=1).astype(np.uint32)
    key = os.urandom(32) if salt_key is None else bytes(salt_key)
    ys = {p: salts(key, k, len(rows[p])) for k, p in enumerate(rows)}
    del key
    t_salt = time.perf_counter()
    domains = {p: window.domain(p, ROW_SCHEMA, len(a)) for p, a in rows.items()}
    roots, bc = commit_ports(domains, rows, {}, ys, workers=workers, chunk=chunk)
    t1 = time.perf_counter()
    return Committed(window, roots, domains, bc, ys, rows, {"salts_s": round(t_salt - t0, 3), "commit_s": round(t1 - t_salt, 3),
                                                            "total_s": round(t1 - t0, 3), "workers": workers}, refs)


# --- the files ----------------------------------------------------------------------------------------------------------

def port_leaves(m: Member) -> dict[str, int]:
    """Each of member ``m``'s row ports (outputs included) and its leaf count: its table's size for a shared-row input, else ``n``."""
    tables = dict(m.tables)
    return {p: tables.get(p, m.n) for p, _ in m.all_rows}


def expected_ports(window: Window) -> dict[str, dict[str, Any]]:
    """The ports the verifier's own window derives, as ``registration.check`` takes them: ``{<descriptor id>/<port>: {leaves,
    domain}}``, each domain ``registration.served_domain`` over the window's program, partition and run."""
    prog, part = window.partition["program"], window.partition_digest
    return {m.port(p): {"leaves": n, "domain": R.served_domain(prog, part, m.port(p), ROW_SCHEMA, n, window.run)}
            for m in window.members for p, n in port_leaves(m).items()}


def leaf_layer(publics: Mapping[str, bytes]) -> str:
    """``LEAF_LAYER`` over ``{template: public body}``."""
    return hashlib.sha512(canonical_json({t: hashlib.sha512(b).hexdigest() for t, b in publics.items()})).hexdigest()


def registration(c: Committed, provenance: Mapping[str, Any], previous: str | None = None) -> dict[str, Any]:
    """Core's ``verity/registration/v1`` record of the window: its program, query and law, the scheme, every port's root, leaves and
    served domain, the leaf layer, and ``window = {"kind": "served", "run": <vllm-v1 run root>, **provenance}``."""
    w = c.window
    roots = [{"port": p, "root": c.roots[p].hex(), "leaves": c.domains[p].count, "domain": c.domains[p].domain_id.hex()} for p in c.roots]
    publics = {m.template: c.public_body(k) for k, m in enumerate(w.members)}
    return R.record(w.program, w.query, w.law, SCHEME, roots, {"kind": "served", "run": w.run, **provenance},
                    leaf_layer=leaf_layer(publics), previous=previous)


def index_doc(members: Sequence[Member], entries: Sequence[Sequence[list[Any]]], run: str) -> dict[str, Any]:
    """Unit -> served identity: per member, one entry per served row; global units ``[lo, hi)``, instance ``i - lo`` of the row
    (a RoPE head, or 0 for a template that is the row itself)."""
    return {"schema": INDEX_SCHEMA, "order": ORDER, "run": run,
            "columns": ["lo", "hi", "request", "engine_step", "layer", "op_path", "member", "token_row", "program_row"],
            "members": [{"template": m.template, "base": m.base, "n": m.n, "rows": list(map(list, e))} for m, e in zip(members, entries, strict=True)]}


def index_bytes(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, separators=(",", ":")) + "\n").encode()


def write(c: Committed, out_dir: str | Path, provenance: dict[str, Any], index: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The window's files; the window's provenance gets the index's SHA-512 (so the registration binds the unit -> identity map)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ib = index_bytes(index)
    (d / "index.json").write_bytes(ib)
    rec = registration(c, {**provenance, "index": hashlib.sha512(ib).hexdigest(), "order": ORDER})
    (d / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    members = []
    for k, m in enumerate(c.window.members):
        pub = c.public_body(k)
        (d / f"m{k}-public.bin").write_bytes(pub)
        fd = os.open(d / f"m{k}-private.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(c.private_body(k))
        members.append({"k": k, "template": m.template, "n": m.n, "base": m.base, "ports": [list(p) for p in m.all_rows],
                        "input_ports": len(m.rows), "out_ports": [list(p) for p in m.outs], "frame_v3": c.frame_entry(k),
                        "public_sha512": hashlib.sha512(pub).hexdigest(),
                        **({"shared_rows": {**dict(m.tables), **{p: m.n for p, _ in m.out_rows}}} if m.tables else {})})
    win = {"schema": "verity-vllm/serving-rows/v2", "scheme": SCHEME, "population": c.window.population, "members": members,
           "partition": c.window.partition, "partition_digest": c.window.partition_digest, "run": c.window.run,
           "domain_tag": DOMAIN_TAG, "registration": R.FORMAT, "registration_digest": R.digest(rec), "leaf_layer": rec["leaf_layer"],
           "leaf_layer_rule": LEAF_LAYER, "index_sha512": hashlib.sha512(ib).hexdigest(),
           "salt_rule": "chacha20 (rfc 8439), a fresh 32-byte os key per window, nonce = the row port's index over all members "
           "(12 bytes le), row i at stream offset 192 i; the key is dropped after expansion", "seconds": c.seconds, **(extra or {})}
    (d / "window.json").write_text(json.dumps(win, indent=1, default=str) + "\n")
    return win


def flock_input_files(window_dir: str | Path, k: int, circuit_text: str, out_dir: str | Path, subcircuit: str) -> dict[str, Any]:
    """Member ``k``'s M0 ``m<k>-pub-<n>.bin`` (``rows: false``) and ``m<k>-inst-<n>.bin``, under the header
    ``verity_flock.circuit.write`` gives the circuit it composed: its SHA-512 pin, its META's block shape and class, the member's
    global unit indices, and its ``frame_v3``.  The bodies are the window's bytes."""
    d, o = Path(window_dir), Path(out_dir)
    win = json.loads((d / "window.json").read_text())
    rec = json.loads((d / "registration.json").read_text())
    mem = win["members"][k]
    head_line, meta_line, rest = circuit_text.split("\n", 2)
    meta = json.loads(meta_line[5:])
    if ([list(p[:2]) for p in meta["ports"]] != mem["ports"] or [list(p) for p in meta["out_ports"]] != mem["out_ports"]
            or meta.get("input_ports") != mem["input_ports"]):
        raise ValueError(f"the circuit's ports {meta['ports']} / {meta['out_ports']} ({meta.get('input_ports')} input rows) are not "
                         f"member {k}'s {mem['ports']} / {mem['out_ports']} ({mem['input_ports']} input rows)")
    if meta.get("scheme") != SCHEME:
        raise ValueError(f"the circuit commits under {meta.get('scheme')!r}, the window under {SCHEME!r}")
    cls_meta = {x: v for x, v in meta.items() if x not in M0_BINDING_KEYS}
    cls = hashlib.sha512((head_line + "\nMETA " + json.dumps(cls_meta, sort_keys=True, separators=(",", ":")) + "\n" + rest).encode()).hexdigest()
    n, G = int(mem["n"]), int(meta["vus_per_block"])
    idx = list(range(int(mem["base"]), int(mem["base"]) + n))
    units = {"vus_per_block": G, "units_per_vu": meta["units_per_vu"], "indices": idx, "blocks": [idx[b:b + G] for b in range(0, n, G)],
             "classes": [cls] * n}
    head = {"format": M0_FORMAT, "statement": M0_STATEMENT, "scheme": SCHEME, "set": f"served/{rec['window']['run']}/m{k}",
            "subcircuit": subcircuit, "range": [0, n], "instances": n,
            "circuit_sha512": hashlib.sha512(circuit_text.encode()).hexdigest(), "units": units, "frame_v3": mem["frame_v3"], "rows": True}
    if mem.get("shared_rows"):
        head["shared_rows"] = mem["shared_rows"]
    o.mkdir(parents=True, exist_ok=True)
    pub = (d / f"m{k}-public.bin").read_bytes()
    (o / f"m{k}-pub-{n}.bin").write_bytes(json.dumps(dict(head, rows=False), sort_keys=True).encode() + b"\n" + pub)
    fd = os.open(o / f"m{k}-inst-{n}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n" + (d / f"m{k}-private.bin").read_bytes() + pub)
    return {"pub": str(o / f"m{k}-pub-{n}.bin"), "inst": str(o / f"m{k}-inst-{n}.bin"), "circuit_sha512": head["circuit_sha512"], "class": cls}
