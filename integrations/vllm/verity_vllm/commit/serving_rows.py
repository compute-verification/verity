"""Serving rows: the opt-in ``frame-v3-sha512`` + ``hm96-sha512/v1`` commitment of a run's RoPE heads in M0's serving row leaf
format (``verity_flock.circuit``'s statement rows, PR #83), with the registration record, M0's public and prover files and the
unit index the one-stage audit reads (``protocols/one_stage``, PR #116).  Off by default: ``vllm-v1`` stays the record, and this
commits beside it without touching any root of record, until the re-baseline switches the record.

Per row (one head's 64 words), from core references only (``tests/commit/test_serving_rows.py`` pins them, with M0's vector):

  row     the words, u16 little-endian; port ``x`` is the head's RoPE input, port ``cs`` the position's ``[cos | sin]`` (the
          registered cos/sin cache slice the Call reads), one ``cs`` row per head as M0's instances carry it
  inner   ``x = SHA-512(sha512_row_prefix(role 1, 16, n) || row)``                              (``sha512/row/v1``)
  salt    ``y`` = 192 bytes of ChaCha20 (RFC 8439) under the window's salt key, nonce = the port's index (12 bytes LE), at stream
          offset ``192 i``; the key is 32 fresh bytes from the OS generator per window, dropped once the salts are expanded.  The
          salts are kept only in the prover's file and never logged
  string  ``b || c = hm96.SHA512.commit_string(default_key, x, y)``                               (public, 128 bytes)
  leaf    the ``frame-v3-sha512`` row leaf, schema ``hm96-sha512/row/v1``, over ``hm96.SHA512.tree_leaf(default_key, b || c)``
  out     one ``u16`` word leaf per output word, instance-major: ``H("leaf", [domain, r, r, "u16", be16(v)])``

Each port is one tree over the window's units under ``CommitmentDomain(binding, -1, [0, leaves), "sha512")``, with the domain
rule agreed with the one-stage lane: ``binding = identity_digest(DOMAIN_TAG, {program, partition, port, schema, leaves, run})``,
every field a registration field (``program`` is the partition's program, ``partition`` its digest, ``run`` the source's run),
so the verifier recomputes every domain from the registration.

The population is fixed before serving, by the Build's Program: every head of every ``RoPE_v1`` row (:data:`ORDER`: request
order, then Program row index, then head), the ``RoPEHead<64>`` cut ``pipeline.vu_export`` makes.  The partition over it
(``Q_template_instance`` v0 over ``TemplateInstances{N}``) is the verifier's, handed in as a file; serving refuses a population
that differs.  ``write`` puts into a window directory:

  registration.json  ``verity/one-stage/registration/v0``, source ``served``
  index.json         unit -> served identity, one entry per RoPE row
  public.bin         every row's ``b || c`` (instance-major, ports in order), then every output word u16 LE: M0's public body
  private.bin        every instance's rows (u16 LE, ports in order), then every salt (instance-major, ports in order): M0's
                     prover body.  Prover only; it holds every private word
  window.json        roots, domains, digests and timings

``m0_files`` wraps the two bodies as M0's ``pub-N.bin`` (``rows: false``) and ``inst-N.bin`` under the header of M0's composed
``circuit.txt``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments import hm96
from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, sha512_row_prefix
from verity.ir.codec import canonical_json

from .hiding import HidingLeaves

SCHEME = "frame-v3-sha512/hm96-sha512"
ROW_SCHEMA = SCHEMA_HM96_SHA512_ROW
WORD_SCHEMA = "u16"
DOMAIN_TAG = "verity/one-stage/served-domain/v0"
OWNER = -1
REGISTRATION = "verity/one-stage/registration/v0"
ORDER = "verity-vllm/serving-rows/order/v0"
INDEX_SCHEMA = "verity-vllm/serving-rows/index/v0"
FAMILY, D = "RoPE_v1", 64
IN_PORTS, OUT_PORT = ("x", "cs"), "out"
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
class Window:
    """What the roots are bound to: the verifier's partition (its program and query, the digest, the population) and the run."""
    partition: dict[str, Any]
    partition_digest: str
    population: int
    run: str

    @classmethod
    def from_file(cls, path: str | Path, run: str) -> "Window":
        d = json.loads(Path(path).read_text())
        return cls(d["partition"], d["partition_digest"], int(d["population"]), run)

    def binding(self, port: str, schema: str, leaves: int) -> bytes:
        return bytes.fromhex(identity_digest(DOMAIN_TAG, {"program": self.partition["program"], "partition": self.partition_digest,
                                                          "port": port, "schema": schema, "leaves": int(leaves), "run": self.run}))

    def domain(self, port: str, schema: str, leaves: int) -> CommitmentDomain:
        return CommitmentDomain(self.binding(port, schema, leaves), OWNER, RangeIndexedDomain(0, leaves), hash="sha512")


@dataclass
class Committed:
    window: Window
    n: int
    roots: dict[str, bytes]
    domains: dict[str, CommitmentDomain]
    bc: dict[str, bytes]               # per input port: every row's b || c, concatenated
    salts: dict[str, bytes]            # per input port: every row's salt (private)
    rows: dict[str, np.ndarray]        # per input port: (n, words) u16 (private)
    out: np.ndarray                    # (n, words) u16
    seconds: dict[str, Any]

    def schemas(self) -> dict[str, str]:
        return {**{p: ROW_SCHEMA for p in IN_PORTS}, OUT_PORT: WORD_SCHEMA}

    def public_body(self) -> bytes:
        bc = np.concatenate([np.frombuffer(self.bc[p], dtype=np.uint8).reshape(self.n, BC_BYTES) for p in IN_PORTS], axis=1)
        return bc.tobytes() + np.ascontiguousarray(self.out).astype("<u2").tobytes()

    def private_body(self) -> bytes:
        rows = np.concatenate([self.rows[p] for p in IN_PORTS], axis=1).astype("<u2").tobytes()
        ys = np.concatenate([np.frombuffer(self.salts[p], dtype=np.uint8).reshape(self.n, SALT_BYTES) for p in IN_PORTS], axis=1)
        return rows + ys.tobytes()

    def frame_v3(self) -> dict[str, Any]:
        """M0's header ``frame_v3`` entry for these roots."""
        return {"frame": "frame-v3-sha512", "roots": {p: r.hex() for p, r in self.roots.items()},
                "domain_ids": {p: d.domain_id.hex() for p, d in self.domains.items()},
                "bindings": {p: d.binding.hex() for p, d in self.domains.items()}, "owner": OWNER, "schemas": self.schemas(),
                "row_key_sha512": hashlib.sha512(INST.default_key).hexdigest()}


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


def commit(window: Window, rows: dict[str, np.ndarray], out: np.ndarray, *, salt_key: bytes | None = None, workers: int = 1,
           chunk: int = CHUNK) -> Committed:
    """Commit the window's rows (``rows[p]``: (n, words) u16 per input port) and output words (``out``: (n, words) u16).

    ``salt_key``: tests only; serving draws 32 fresh OS bytes."""
    t0 = time.perf_counter()
    n = int(out.shape[0])
    if n != window.population:
        raise ValueError(f"{n} units read, the registered population is {window.population}")
    for p in IN_PORTS:
        if rows[p].shape[0] != n or (2 * rows[p].shape[1]) % 128:
            raise ValueError(f"port {p}: {rows[p].shape} is not {n} rows of whole SHA-512 blocks")
    key = os.urandom(32) if salt_key is None else bytes(salt_key)
    ys = {p: salts(key, k, n) for k, p in enumerate(IN_PORTS)}
    del key
    t_salt = time.perf_counter()
    domains = {p: window.domain(p, ROW_SCHEMA, n) for p in IN_PORTS}
    domains[OUT_PORT] = window.domain(OUT_PORT, WORD_SCHEMA, n * int(out.shape[1]))
    roots, bc = commit_ports(domains, {p: rows[p] for p in IN_PORTS}, {OUT_PORT: out}, ys, workers=workers, chunk=chunk)
    t1 = time.perf_counter()
    return Committed(window, n, roots, domains, bc, ys, {p: np.asarray(rows[p], dtype=np.uint16) for p in IN_PORTS},
                     np.asarray(out, dtype=np.uint16), {"salts_s": round(t_salt - t0, 3), "commit_s": round(t1 - t_salt, 3),
                                                        "total_s": round(t1 - t0, 3), "workers": workers})


# --- the files ----------------------------------------------------------------------------------------------------------

def registration(c: Committed, source: dict[str, Any], registered_utc: str | None = None) -> dict[str, Any]:
    """``verity/one-stage/registration/v0`` (``verity_one_stage.registration.record``'s shape), source ``served``."""
    w = c.window
    return {"format": REGISTRATION, "commitment": {"scheme": SCHEME, "source": {"kind": "served", **source}},
            "roots": {p: r.hex() for p, r in c.roots.items()}, "leaves": {p: d.count for p, d in c.domains.items()},
            "domains": {p: d.domain_id.hex() for p, d in c.domains.items()}, "partition": w.partition,
            "partition_digest": w.partition_digest, "population": w.population,
            "registered_utc": registered_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def registration_digest(record: dict[str, Any]) -> str:
    return hashlib.sha512(REGISTRATION.encode() + b"\0" + canonical_json(record)).hexdigest()


def index_doc(entries: list[list[Any]], run: str) -> dict[str, Any]:
    return {"schema": INDEX_SCHEMA, "order": ORDER, "run": run, "family": FAMILY,
            "columns": ["lo", "hi", "request", "engine_step", "layer", "op_path", "member", "token_row", "program_row"],
            "rule": "unit i in [lo, hi) is head i - lo of the RoPE_v1 row", "rows": entries}


def index_bytes(doc: dict[str, Any]) -> bytes:
    return (json.dumps(doc, separators=(",", ":")) + "\n").encode()


def write(c: Committed, out_dir: str | Path, source: dict[str, Any], index: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """The window's files; ``source`` gets the index's SHA-512 (so the registration binds the unit -> identity map)."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    ib = index_bytes(index)
    (d / "index.json").write_bytes(ib)
    rec = registration(c, {**source, "index": hashlib.sha512(ib).hexdigest(), "order": ORDER})
    (d / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    pub = c.public_body()
    (d / "public.bin").write_bytes(pub)
    fd = os.open(d / "private.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(c.private_body())
    win = {"schema": "verity-vllm/serving-rows/v0", "scheme": SCHEME, "n": c.n, "ports": [[p, int(c.rows[p].shape[1])] for p in IN_PORTS],
           "out_ports": [[OUT_PORT, int(c.out.shape[1])]], "frame_v3": c.frame_v3(), "domain_tag": DOMAIN_TAG,
           "registration_sha512": registration_digest(rec), "public_sha512": hashlib.sha512(pub).hexdigest(),
           "index_sha512": hashlib.sha512(ib).hexdigest(), "salt_rule": "chacha20 (rfc 8439), a fresh 32-byte os key per window, "
           "nonce = port index (12 bytes le), row i at stream offset 192 i; the key is dropped after expansion", "seconds": c.seconds,
           **(extra or {})}
    (d / "window.json").write_text(json.dumps(win, indent=1, default=str) + "\n")
    return win


def m0_files(window_dir: str | Path, circuit_text: str, out_dir: str | Path, subcircuit: str) -> dict[str, Any]:
    """M0's ``pub-N.bin`` (``rows: false``) and ``inst-N.bin`` from a window, under the header ``verity_flock.circuit.write``
    gives the circuit it composed: its SHA-512 pin, its META's block shape and class, global unit indices ``0..N-1``, and the
    window's ``frame_v3``.  The bodies are the window's bytes."""
    d, o = Path(window_dir), Path(out_dir)
    win = json.loads((d / "window.json").read_text())
    rec = json.loads((d / "registration.json").read_text())
    head_line, meta_line, rest = circuit_text.split("\n", 2)
    meta = json.loads(meta_line[5:])
    if [list(p[:2]) for p in meta["ports"]] != win["ports"] or [list(p) for p in meta["out_ports"]] != win["out_ports"]:
        raise ValueError(f"the circuit's ports {meta['ports']} / {meta['out_ports']} are not the window's {win['ports']} / {win['out_ports']}")
    if meta.get("scheme") != SCHEME:
        raise ValueError(f"the circuit commits under {meta.get('scheme')!r}, the window under {SCHEME!r}")
    cls_meta = {k: v for k, v in meta.items() if k not in M0_BINDING_KEYS}
    cls = hashlib.sha512((head_line + "\nMETA " + json.dumps(cls_meta, sort_keys=True, separators=(",", ":")) + "\n" + rest).encode()).hexdigest()
    n, G = int(win["n"]), int(meta["vus_per_block"])
    idx = list(range(n))
    units = {"vus_per_block": G, "units_per_vu": meta["units_per_vu"], "indices": idx, "blocks": [idx[b:b + G] for b in range(0, n, G)],
             "classes": [cls] * n}
    head = {"format": M0_FORMAT, "statement": M0_STATEMENT, "scheme": SCHEME, "set": f"served/{rec['commitment']['source']['run']}",
            "subcircuit": subcircuit, "content_digest": win["index_sha512"], "range": [0, n], "instances": n,
            "circuit_sha512": hashlib.sha512(circuit_text.encode()).hexdigest(), "units": units, "frame_v3": win["frame_v3"], "rows": True}
    o.mkdir(parents=True, exist_ok=True)
    pub = (d / "public.bin").read_bytes()
    (o / f"pub-{n}.bin").write_bytes(json.dumps(dict(head, rows=False), sort_keys=True).encode() + b"\n" + pub)
    fd = os.open(o / f"inst-{n}.bin", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(json.dumps(head, sort_keys=True).encode() + b"\n" + (d / "private.bin").read_bytes() + pub)
    return {"pub": str(o / f"pub-{n}.bin"), "inst": str(o / f"inst-{n}.bin"), "circuit_sha512": head["circuit_sha512"], "class": cls}


def program_sha512(program_dir: str | Path) -> str:
    """SHA-512 over the canonical descriptor bytes core's ``program_digest`` hashes with SHA-256."""
    desc = json.loads(gzip.open(Path(program_dir) / "descriptor.json.gz").read())
    return hashlib.sha512(canonical_json({k: v for k, v in desc.items() if k != "annotations"})).hexdigest()


# --- the RoPE heads of a run ---------------------------------------------------------------------------------------------

def rope_rows(programs: dict[str, dict[str, Any]], program_dirs: dict[str, str] | None = None) -> list[tuple[str, Any, dict, dict]]:
    """The ``RoPE_v1`` rows in :data:`ORDER`: (request, its ProgramIndex, the row, its description).  The population is the sum of
    their NHEADS, fixed by the Build's Programs before serving."""
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.query.manifest import format as RM

    out = []
    for rid, inst in programs.items():
        pi = ProgramIndex(inst, program_dir=(program_dirs or {}).get(str(rid)))
        for i in sorted(pi.by_i):
            r = pi.by_i[i]
            if RM.spec_family(r["spec"]) == FAMILY:
                d = pi.describe(r)
                if int(d["statics"]["D"]) != D:
                    raise ValueError(f"row {i}: RoPE_v1 D={d['statics']['D']}; serving rows commit D={D} heads only")
                out.append((str(rid), pi, r, d))
    if not out:
        raise LookupError(f"no {FAMILY} rows in the run's Programs")
    return out


def population(programs: dict[str, dict[str, Any]], program_dirs: dict[str, str] | None = None) -> int:
    return sum(int(d["statics"]["NHEADS"]) for _, _, _, d in rope_rows(programs, program_dirs))


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def rope_units(rows: list[tuple[str, Any, dict, dict]], store: Any, weights: Callable, *, arrivals: dict[str, int] | None = None,
               ) -> tuple[list[list[Any]], dict[str, np.ndarray], np.ndarray]:
    """Every head of every row, its words read from the committed store (``x``, ``out``) and the registered weights (``cs``)
    exactly as the replay reads them.  Returns (index entries, rows per input port, out)."""
    entries: list[list[Any]] = []
    xs, css, outs = [], [], []
    n = 0
    for rid, pi, r, d in rows:
        arrive = int((arrivals or {}).get(rid, 0))
        nh = int(d["statics"]["NHEADS"])
        ops = [_operand(pi, store, weights, rid, arrive, g) for g in r["args"]]
        (member, width, _dt), = d["members"]
        w, why = store.words(rid, arrive + d["step"], d["op_path"], member, d["row"] * int(width), (d["row"] + 1) * int(width))
        if w is None:
            raise LookupError(f"row {r['i']} output: {why}")
        xs.append(np.asarray(ops[0], dtype=np.uint16).reshape(nh, D))
        css.append(np.broadcast_to(np.asarray(ops[1], dtype=np.uint16).reshape(1, D), (nh, D)))
        outs.append(np.asarray(w, dtype=np.uint16).reshape(nh, D))
        m = _LAYER.search(d["op_path"])
        entries.append([n, n + nh, rid, arrive + int(d["step"]), int(m.group(1)) if m else None, d["op_path"], str(member), int(d["row"]), int(r["i"])])
        n += nh
    return entries, {"x": np.concatenate(xs), "cs": np.concatenate(css)}, np.concatenate(outs)


def _operand(pi: Any, store: Any, weights: Callable, rid: str, arrive: int, group: Sequence) -> np.ndarray:
    words = []
    for sl in group:
        kind, a, lo, hi = sl[0], int(sl[1]), int(sl[2]), int(sl[3])
        if kind == "n":
            m = pi.member_at(a, lo, hi)
            if m is None:
                raise LookupError(f"arg slice {list(sl)} is not inside one committed member")
            pd, member, mlo, mhi, _dt = m
            width = next(int(mm[1]) for mm in pd["members"] if mm[0] == member)
            w, why = store.words(rid, arrive + pd["step"], pd["op_path"], member, pd["row"] * width + mlo, pd["row"] * width + mhi)
            if w is None:
                raise LookupError(why)
            words.append(np.asarray(w).reshape(-1))
        elif kind == "p" and (pi.pnames[a] if a < len(pi.pnames) else None) == "weights":
            ws = pi.weight_slice(lo, hi)
            if ws is None:
                raise LookupError(f"weights slice [{lo}, {hi}) spans fields")
            fname, flo, fhi, _shape = ws
            arr = weights(fname, flo, fhi)
            if arr is None:
                raise LookupError(f"registered weight field {fname} is not available")
            words.append(np.asarray(arr).reshape(-1))
        else:
            raise LookupError(f"arg slice {list(sl)}: the serving rows read committed members and registered weights only")
    return np.concatenate(words)


def after_commit(out_dir: str, *, partition_file: str, programs: dict[str, dict[str, Any]], bmap: dict[str, Any], committed: Callable,
                 weights: Callable, manifest: dict[str, Any] | None, run_root: bytes, run: str, row: Any = None,
                 arrivals: dict[str, int] | None = None, program_dirs: dict[str, str] | None = None, committer: dict[str, Any] | None = None,
                 workers: int | None = None) -> dict[str, Any]:
    """The Commit stage's opt-in hook, after the vllm-v1 roots are final and before any challenge: commit the run's RoPE heads
    and write the window's files into ``out_dir``.  It reads the committed values and writes only ``out_dir``."""
    from verity_vllm.check.replay.opening import CommittedStore

    t0 = time.perf_counter()
    dirs = {str(k): v for k, v in (program_dirs or {}).items()}
    served = {program_sha512(dirs[str(r)]) for r in programs}
    if len(served) != 1:
        raise NotImplementedError(f"serving rows register one served Program; this run has {len(served)}")
    window = Window.from_file(partition_file, run)
    rows = rope_rows(programs, dirs)
    entries, cols, out = rope_units(rows, CommittedStore.of_record(bmap, committed, manifest), weights, arrivals=arrivals)
    t_read = time.perf_counter()
    c = commit(window, cols, out, workers=workers or min(16, os.cpu_count() or 1))
    source = {"row": row, "run": run, "committer": dict(committer or {}, module=__name__), "served_program": served.pop(),
              "served_program_digest": sorted({str(i.get("digest")) for i in programs.values()}), "vllm_v1_run_root": bytes(run_root).hex()}
    c.seconds["read_s"] = round(t_read - t0, 3)
    res = write(c, out_dir, source, index_doc(entries, run))
    res["seconds"]["hook_s"] = round(time.perf_counter() - t0, 3)
    (Path(out_dir) / "window.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[serving-rows] {c.n} {FAMILY} heads committed ({SCHEME}): roots " + ", ".join(f"{p} {c.roots[p].hex()[:16]}" for p in c.roots)
          + f"; registration {res['registration_sha512'][:16]}; {res['seconds']}", flush=True)
    return res
