"""A PoUB session over PoUS's P2 store on the card that serves from it: the prover's side of
`verity.protocols.accounting.memory.poub.verifier.p2_parties`, whose verifier runs on another host.

    bandwidth.py registration --store ENC --node NODE --layout F --out FILE
    bandwidth.py session --store ENC --layout F --lib DIR --node NODE --out DIR
                 (--verifier CMD | --connect HOST:PORT | --party CMD --party-host HOST)
                 [--windows 6] [--passes 4] [--samples 1024] [--window-s 20] [--deadline-s 1]
                 [--skip-layer L | --cache] [--expect accept|reject] [--device cuda] [--ref-decode]
                 [--case-dir DIR [--case-exe EXE]]

**The plan.** A pass reads the units of the registration's plan (`p2_parties.plan_of`), which the verifier holds and sends
with it: `tensor_plan`, the units holding a tensor's block, so not the store's padding past the last tensor (it is
PoUS's, and the decode never reads it). `registration --layout` writes it into the registration.

**The step.** Each pass of the session is one engine step's weight reads: every tensor of the store, in block order, from
its segment files onto the device (`stage.Segments`, kernels' reader) and through kernels' staging decode
(`stage.decode_matrix`) into a staging buffer, as a forward step reads them. The decode reads each chunk's rows from
`stage.EncodedRows.decode_rows`, whose attached sketch (`store.sketch`, `Sketch.read`) sketches those same device rows under
the pass's seed just before they go to the decode kernel (`Sketch.add`: π of each block by an exact int8 matmul against its four offsets' signs, summed into
its unit's σ), so the receipts are of the bytes the decode read. Blocks of no tensor up to the plan's last unit (the
head's last unit's tail) are read and sketched alone, so every unit of the plan is covered. Then y = SHA512(prefix ‖
u64be g ‖ σ_g) for every unit of the plan on the device (`receipts_of`, a torch SHA-512 after the prefix's midstate; a
sample held to hashlib and to `p2.Sketcher` each pass), and the pass's receipt tree (`ReceiptTree`, `receipts.tree`'s,
in worker processes). The sketch is a pass of its own over the chunk's rows, not fused into the decode kernel, so each
byte is read from HBM twice.

**The session.** `session` starts the verifier with `--verifier CMD` (the ssh command that runs `p2_parties --stdio` on
the auditor's device; on one host, the module itself, which certifies nothing), reaches `p2_parties --listen` at
`--connect HOST:PORT`, or starts it as a party of the auditor's gate (`--party CMD`, the ssh command to the gate's verb,
whose stdout says the port, reached at `--party-host`; the verifier keeps no record of its answer), sends the request and answers each pass,
each window's openings (each unit's four blocks from the segment files with their PoUS vk paths, `VkPaths`, from the
store's row digests) and returns the verifier's answer, written to `OUT/session.json` with each step's timings, and
the line `BW VERDICT ...`. `--skip-layer L` is a cheat: it neither reads nor decodes layer L's tensors, and the
receipts of the units they hold are from what was read, so a sample that lands there is rejected (`receipt`). `--cache`
is another: every pass after the session's first reads nothing of the store and replays the first pass's σ from its
cache under the pass's seed, so a sample of a later pass is rejected (`receipt`).

**The case.** The verifier sends the session's core case before its answer (`p2_parties`'s `{"case": ...}`, `p2_case`):
`--case-dir DIR` writes it with its answer, `case.json` and `answer.json`, into DIR (0700, each file 0600), the
answer `Poub.P2Protocol`'s with `--case-exe` (`poub-p2-protocol`); without it the case is undecided, whatever the
session said (`p2_case.unjudged`). The case
holds the store's opened blocks and the verifier's coins, so DIR is private and never under OUT: `session.json` says
only where it went, its commitment's digest, its store and its verdict. A session the verifier refused, or one that
ended in the prover's error, writes none.

`--ref-decode` swaps the library for P2's reference decode on the CPU (`RefDecode`), for the toy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import secrets
import socket
import struct
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[7]
STAGE = (ROOT / "verity/protocols/accounting/memory/poub/prover/p2_stage/stage.py").parent
sys.path.insert(0, str((ROOT / "verity/protocols/accounting/memory/pous/prover/p2/p2_store/store.py").parents[1]))
sys.path.insert(0, str(STAGE))

from verity.commitments import advice
from verity.commitments.identity import canonical_json_bytes
from verity.commitments.merkle import levels, merkle_depth
from verity.protocols.accounting.memory.poub.prover import p2_party_prover
from verity.protocols.accounting.memory.poub.verifier import coins, ledger, p2, p2_case, p2_parties, receipts
from verity.protocols.accounting.memory.pous.attester import memory_attester as mc
from verity.protocols.accounting.memory.pous.verifier import period
from verity.protocols.accounting.memory.pous.verifier import verifier as V
from verity.protocols.accounting.memory.pous.verifier.schemes.p2 import P2Params

BLOCK, UNIT = p2.BLOCK_BYTES, p2.UNIT_BLOCKS
SALT = receipts.SALT_BYTES
PER_LEAF, DIGEST = receipts.PER_LEAF, receipts.DIGEST_BYTES


# --- SHA-512: the prefix's midstate in Python, the receipts' last block on the device ---------------------------------

def _root(p: int, k: int, bits: int) -> int:
    """⌊p^(1/k) · 2^bits⌋ by Newton's method on integers."""
    n, x = p << (k * bits), 1 << ((p.bit_length() + k * bits) // k + 1)
    while True:
        y = ((k - 1) * x + n // x ** (k - 1)) // k
        if y >= x:
            return x
        x = y


def _primes(n: int) -> list[int]:
    out, c = [], 2
    while len(out) < n:
        if all(c % q for q in out if q * q <= c):
            out.append(c)
        c += 1
    return out


M64 = (1 << 64) - 1
K512 = tuple(_root(p, 3, 64) & M64 for p in _primes(80))
IV512 = tuple(_root(p, 2, 64) & M64 for p in _primes(8))


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & M64


def compress(state: Sequence[int], block: bytes) -> tuple[int, ...]:
    """SHA-512's compression of one 128-byte block into `state`."""
    w = list(struct.unpack(">16Q", block))
    for t in range(16, 80):
        a, b = w[t - 15], w[t - 2]
        w.append((w[t - 16] + (_rotr(a, 1) ^ _rotr(a, 8) ^ (a >> 7)) + w[t - 7]
                  + (_rotr(b, 19) ^ _rotr(b, 61) ^ (b >> 6))) & M64)
    a, b, c, d, e, f, g, h = state
    for t in range(80):
        t1 = (h + (_rotr(e, 14) ^ _rotr(e, 18) ^ _rotr(e, 41)) + ((e & f) ^ (~e & M64 & g)) + K512[t] + w[t]) & M64
        t2 = ((_rotr(a, 28) ^ _rotr(a, 34) ^ _rotr(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return tuple((x + y) & M64 for x, y in zip(state, (a, b, c, d, e, f, g, h)))


def midstate(prefix: bytes) -> tuple[int, ...]:
    if len(prefix) % 128:
        raise ValueError("a midstate is of whole 128-byte blocks")
    state = IV512
    for i in range(0, len(prefix), 128):
        state = compress(state, prefix[i:i + 128])
    return state


def _s64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


class TorchSha512:
    """SHA-512's compression of one block per row on int64 torch tensors, bit for bit `compress`'s."""

    def __init__(self, torch: Any, device: Any) -> None:
        self.t, self.device = torch, device
        self.k = [_s64(k) for k in K512]

    def _shr(self, x, n: int):
        return (x >> n) & ((1 << (64 - n)) - 1)

    def _rotr(self, x, n: int):
        return self._shr(x, n) | (x << (64 - n))

    def compress(self, state: Sequence[int], w: list) -> list:
        """`state` (eight ints, shared by every row) compressed with each row's block `w`, sixteen int64 tensors of the
        block's big-endian words; the eight words of each row's new state."""
        r = self._rotr
        w = list(w)
        for t in range(16, 80):
            a, b = w[t - 15], w[t - 2]
            w.append(w[t - 16] + (r(a, 1) ^ r(a, 8) ^ self._shr(a, 7)) + w[t - 7]
                     + (r(b, 19) ^ r(b, 61) ^ self._shr(b, 6)))
        n = w[0].shape[0]
        init = [self.t.full((n,), _s64(v), dtype=self.t.int64, device=self.device) for v in state]
        a, b, c, d, e, f, g, h = init
        for t in range(80):
            t1 = h + (r(e, 14) ^ r(e, 18) ^ r(e, 41)) + ((e & f) ^ (~e & g)) + self.k[t] + w[t]
            t2 = (r(a, 28) ^ r(a, 34) ^ r(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))
            h, g, f, e, d, c, b, a = g, f, e, d + t1, c, b, a, t1 + t2
        return [x + y for x, y in zip(init, (a, b, c, d, e, f, g, h))]


def receipts_of(torch: Any, seed: bytes, sigma: Any, first: int = 0, chunk: int = 1 << 20) -> np.ndarray:
    """y of units `first ..` from their σ (`sigma`, [n, k_s] int32 on any device), [n, 64] uint8 on the host:
    SHA512(sketch_prefix(seed) ‖ u64be g ‖ σ_g), the prefix as a midstate and the rest one block on the device."""
    prefix = receipts.sketch_prefix(seed)
    rows = sigma.shape[1]
    tail = 8 + 4 * rows
    if tail % 8 or tail + 17 > 128:
        raise ValueError(f"σ of {rows} rows doesn't fit the receipt's last block in whole words")
    state, sha, dev = midstate(prefix), TorchSha512(torch, sigma.device), sigma.device
    bits = 8 * (len(prefix) + tail)
    shifts = torch.arange(56, -8, -8, dtype=torch.int64, device=dev)
    out = np.empty((sigma.shape[0], DIGEST), dtype=np.uint8)
    for lo in range(0, sigma.shape[0], chunk):
        s = sigma[lo:lo + chunk].contiguous()
        n = s.shape[0]

        def const(v: int, n: int = n) -> Any:
            return torch.full((n,), _s64(v), dtype=torch.int64, device=dev)

        g = torch.arange(first + lo, first + lo + n, dtype=torch.int64, device=dev)
        body = (s.view(torch.uint8).view(n, tail // 8 - 1, 8).to(torch.int64) << shifts).sum(dim=2)
        words = [g, *(body[:, k] for k in range(tail // 8 - 1)), const(0x80 << 56)]
        words += [const(0)] * (15 - len(words)) + [const(bits)]
        h = sha.compress(state, words)
        y = torch.stack([((x.unsqueeze(1) >> shifts) & 0xFF) for x in h], dim=1).to(torch.uint8).view(n, DIGEST)
        out[lo:lo + n] = y.cpu().numpy()
    return out


# --- the sketch, from the rows the decode reads ------------------------------------------------------------------------

class Sketch:
    """One pass's σ of every unit, `units` of them, on `device`: `add(rows, j0)` sketches the device rows of blocks
    `j0 ..` ([n, 2056] uint8, as stored) into their units' σ, each block's π against its offset's sign columns, and
    `read` is `stage.EncodedRows.decode_rows`'s hook to it; `seen` counts each unit's blocks sketched, four when the pass
    covered it."""

    def __init__(self, torch: Any, seed: bytes, rows: int, units: int, device: Any) -> None:
        self.t, self.units, self.rows, self.device = torch, units, rows, device
        bits = np.unpackbits(np.frombuffer(b"".join(receipts.sign_bits(seed, rows, p2.UNIT_BYTES)), dtype=np.uint8),
                             bitorder="little").reshape(rows, UNIT, BLOCK)
        signs = (1 - 2 * bits.astype(np.int8)).transpose(2, 1, 0).reshape(BLOCK, UNIT * rows)
        self.signs = torch.from_numpy(np.ascontiguousarray(signs)).to(device)
        self.signs_f = self.signs.to(torch.float32)
        self.sigma = torch.zeros(units, rows, dtype=torch.int32, device=device)
        self.seen = torch.zeros(units, dtype=torch.int32, device=device)
        self.blocks = 0
        self.checked, self.int_mm = False, True

    def _mm(self, x: Any) -> Any:
        """x · signs, exactly: by torch._int_mm where it runs (held to the float product once), else in float32, exact
        since every partial sum is an integer below 2^24 (2,056 · 128)."""
        t = self.t
        if self.int_mm and self.device.type == "cuda" and x.shape[0] > 16:
            try:
                out = t._int_mm(x, self.signs)
            except RuntimeError:
                self.int_mm = False
            else:
                if not self.checked:
                    ref = (x[:64].to(t.float32) @ self.signs_f).to(t.int32)
                    if not t.equal(out[:64], ref):
                        raise RuntimeError("torch._int_mm's sketch is not the exact product")
                    self.checked = True
                return out
        return (x.to(t.float32) @ self.signs_f).to(t.int32)

    def add(self, rows: Any, j0: int) -> None:
        t = self.t
        n = min(rows.shape[0], max(0, UNIT * self.units - j0))
        if n <= 0:
            return
        x = rows[:n].view(t.int8)
        j = t.arange(j0, j0 + n, dtype=t.int64, device=self.device)
        pi = self._mm(x).view(n, UNIT, self.rows)[t.arange(n, device=self.device), j % UNIT]
        self.sigma.index_add_(0, j // UNIT, pi)
        self.seen.index_add_(0, j // UNIT, t.ones(n, dtype=t.int32, device=self.device))
        self.blocks += n

    def read(self, rows: Any, first: int, j0: int, j1: int) -> None:
        self.add(rows[j0 - first:j1 - first], j0)

    def covered(self, runs: Sequence[tuple[int, int]] | None = None) -> int:
        """The units (of `runs`, or all) of every block sketched exactly once; each is four."""
        if runs is None:
            return int((self.seen == UNIT).sum())
        return sum(int((self.seen[f:f + c] == UNIT).sum()) for f, c in runs)


class RefDecode:
    """P2's reference decode (`reference.decode` under `reference.keys_after`) behind `p2_store.gpu.Kernels`, on the CPU:
    for a toy store where no GPU or library is."""

    def __init__(self, lib_dir: Any, build_dir: Any, params: Any, prefix: bytes) -> None:
        import layout as L
        import torch

        self.torch, self.L = torch, L
        self.params, self.prefix, self.device = params, bytes(prefix), torch.device("cpu")

    def decode(self, c: Any, first: int = 0) -> Any:
        torch, L = self.torch, self.L
        out = torch.zeros_like(c)
        for i in range(c.shape[0]):
            v = int.from_bytes(bytes(c[i].numpy()), "big")
            if v < self.params.p:
                k, t, _ = L.R.keys_after(self.prefix, self.params, *divmod(first + i, self.params.blocks_per_segment))
                out[i] = torch.frombuffer(bytearray(L.R.decode(self.params, v, k, t).to_bytes(L.ROW, "big")),
                                          dtype=torch.uint8)
        return out

    def raised(self) -> int:
        return 0


# --- the receipt tree and PoUS's vk paths, in worker processes ---------------------------------------------------------

def _chunk(args: tuple) -> list[bytes]:
    d, n, lo, hi, leaves, ys, salts = args
    nodes = [d.leaf(c, c, receipts.SCHEMA, receipts.hidden(ys[(PER_LEAF * c - PER_LEAF * lo) * DIGEST:
                                                              (min(PER_LEAF * (c + 1), n) - PER_LEAF * lo) * DIGEST],
                                                           salts[SALT * (c - lo):SALT * (c - lo + 1)]))
             for c in range(lo, min(hi, leaves))]
    nodes += [d.pad(r) for r in range(max(lo, leaves), hi)]
    return [b"".join(level) for level in levels(d.domain_id, nodes, lo, 0, d.hash)]


def _run(pool: ProcessPoolExecutor | None, fn, tasks: list[tuple]) -> list[bytes]:
    return list(pool.map(fn, tasks)) if pool is not None and len(tasks) > 1 else [fn(x) for x in tasks]


class ReceiptTree:
    """`receipts.tree(seed, plan, ys, salts)` built in `pool`'s processes: `merkle.levels` over each aligned chunk of
    `CHUNK` leaves and pads (`CHUNK` a power of two) and over the chunks' roots; its `root`, and `path(c)`, leaf c's
    siblings from the leaf up, as `MerkleTree.open(c).path`."""

    CHUNK = 1 << 14

    def __init__(self, seed: bytes, plan: receipts.Plan, ys: np.ndarray, salts: bytes,
                 pool: ProcessPoolExecutor | None = None) -> None:
        L, n = plan.length, receipts.leaves(plan.length)
        if ys.shape != (L, DIGEST) or len(salts) != SALT * n:
            raise ValueError(f"{L} receipts in {n} leaves")
        if self.CHUNK < 1 or self.CHUNK & (self.CHUNK - 1):
            raise ValueError(f"CHUNK {self.CHUNK} is not a power of two")
        raw, width = ys.tobytes(), 1 << merkle_depth(n)
        C, d = min(self.CHUNK, width), receipts.domain(seed, plan)
        chunks = _run(pool, _chunk, [(d, L, lo, lo + C, n, raw[PER_LEAF * DIGEST * lo:PER_LEAF * DIGEST * min(n, lo + C)],
                                      salts[SALT * lo:SALT * min(n, lo + C)]) for lo in range(0, width, C)])
        self.levels = [b"".join(c[k] for c in chunks) for k in range(len(chunks[0]))]
        if len(chunks) > 1:
            self.levels += [b"".join(lv) for lv in levels(d.domain_id, [c[-1] for c in chunks], 0, C.bit_length() - 1,
                                                           d.hash)[1:]]
        self.root = self.levels[-1][:DIGEST]

    def path(self, c: int) -> tuple[bytes, ...]:
        return tuple(lv[DIGEST * ((c >> k) ^ 1):DIGEST * ((c >> k) ^ 1) + DIGEST]
                     for k, lv in enumerate(self.levels[:-1]))


def _segment_paths(args: tuple) -> dict[int, tuple[bytes, ...]]:
    rows_path, salt, s, bps, vs, root = args
    rows = Path(rows_path).read_bytes()
    if len(rows) != DIGEST * bps:
        raise ValueError(f"{rows_path} is not {bps} row digests")
    level = V.row_leaves(salt, s, rows)
    levels = [level]
    while len(level) > 1:
        level = [V.node_hash(level[i], level[i + 1]) for i in range(0, len(level), 2)]
        levels.append(level)
    if level[0] != root:
        raise ValueError(f"segment {s}'s row digests are not its recorded root")
    return {v: tuple(levels[k][(v >> k) ^ 1] for k in range(len(levels) - 1)) for v in vs}


class VkPaths:
    """PoUS's vk paths (`pous.verifier.MerkleTree.path`) of an encoded store's blocks, from its row digests
    (`rows/<s>.rows`) and segment roots (`seg/<s>.json`): a segment of 2^k blocks is a perfect subtree of the RFC 6962
    tree, so a block's path is its segment's path then the segment's in the tree over the segments' roots, whose root
    must be the registration's. A store of other segments (the toy's) is one tree over all its leaves."""

    def __init__(self, store: Path, params: P2Params, salt: bytes, root: bytes,
                 pool: ProcessPoolExecutor | None = None) -> None:
        self.store, self.params, self.salt, self.pool = store, params, salt, pool
        bps, segs = params.blocks_per_segment, params.segments
        self.perfect = bps & (bps - 1) == 0
        if self.perfect:
            self.segment_roots = [bytes.fromhex(json.loads((store / "seg" / f"{s:05d}.json").read_text())["root"])
                                  for s in range(segs)]
            self.top = V.MerkleTree(self.segment_roots)
        else:
            if params.blocks > 1 << 16:
                raise ValueError("a store whose segments aren't 2^k blocks is opened from one tree only up to 2^16 blocks")
            rows = b"".join((store / "rows" / f"{s:05d}.rows").read_bytes() for s in range(segs))
            self.top = V.MerkleTree([V.row_leaf(params, salt, j, rows[DIGEST * j:DIGEST * (j + 1)])
                                     for j in range(params.blocks)])
        if self.top.root != root:
            raise ValueError("the store's segment roots are not the registration's vk root")

    def paths(self, blocks: Sequence[int]) -> dict[int, tuple[bytes, ...]]:
        if not self.perfect:
            return {j: self.top.path(j) for j in blocks}
        bps = self.params.blocks_per_segment
        by: dict[int, list[int]] = {}
        for j in blocks:
            by.setdefault(j // bps, []).append(j % bps)
        tasks = [(str(self.store / "rows" / f"{s:05d}.rows"), self.salt, s, bps, vs, self.segment_roots[s])
                 for s, vs in sorted(by.items())]
        out = {}
        for (_, _, s, _, _, _), got in zip(tasks, _run(self.pool, _segment_paths, tasks)):
            top = self.top.path(s)
            for v, path in got.items():
                out[bps * s + v] = path + top
        return out


# --- the store, the engine step and the prover -------------------------------------------------------------------------

def registration_of_store(store: Path, node: bytes) -> p2.Registration:
    """The prover's view of its store's registration: `store.json`'s commitment, salt and vk root under its setup's P2
    parameters, at sketch-sha512's 16 rows."""
    doc, m = json.loads((store / "store.json").read_text()), json.loads((store / "setup.json").read_text())["manifest"]
    params = P2Params(m["w"], m["c0"], m["blocks_per_segment"], m["segments"], m["expander"])
    return p2.Registration(node, p2.codec_of(params), bytes.fromhex(doc["commitment"]), bytes.fromhex(doc["salt"]),
                           bytes.fromhex(doc["root"]))


@dataclass
class Piece:
    """A run of blocks a step reads in order: a tensor (decoded) or blocks of none (`tensor` None, sketched alone)."""

    blocks: range
    tensor: Any = None


def pieces(lay: Any, total: int) -> list[Piece]:
    out, at = [], 0
    for t in lay.tensors:
        if t.first_block > at:
            out.append(Piece(range(at, t.first_block)))
        out.append(Piece(t.block_range, t))
        at = t.block_range.stop
    if at < total:
        out.append(Piece(range(at, total)))
    return out


def tensor_plan(layout: dict, units: int) -> receipts.Plan:
    """The plan of the units holding a block of a tensor of `layout` (the store's layout JSON, or the toy's, which holds
    it as `layout`), in store order, in a store of `units` units: not the store's padding past them."""
    doc = layout.get("layout", layout)
    cols = doc["tensor_columns"]
    f, n = cols.index("first_block"), cols.index("blocks")
    spans: list[list[int]] = []
    for a, b in sorted((r[f] // UNIT, -(-(r[f] + r[n]) // UNIT)) for r in doc["tensors"]):
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return receipts.Plan(tuple((a, b - a) for a, b in spans), units)


class Engine:
    """One engine step's weight reads through kernels' staging decode (module docstring), its rows sketched."""

    def __init__(self, store: Path, layout: Path, lib: str, registration: p2.Registration, *, device: str = "cuda",
                 chunk: int = 1 << 16, readers: int = 6, skip_layer: int | None = None, ref_decode: bool = False) -> None:
        import layout as L
        import stage as S
        import torch

        self.torch, self.L, self.S = torch, L, S
        self.device = torch.device(device)
        S.DEVICE = self.device
        if ref_decode:
            S.gpu.Kernels = RefDecode
        torch.backends.cuda.matmul.allow_tf32 = False
        doc = json.loads((store / "store.json").read_text())
        self.lay, _ = L.open_store(layout, None, store, doc["source"], doc["commitment"], None)
        if self.lay.params.blocks != registration.codec.blocks:
            raise ValueError("the layout's store is not the registration's")
        self.store, self.lib, self.chunk, self.readers = store, lib, chunk, readers
        self.registration, self.skip_layer = registration, skip_layer
        self.serve(receipts.Plan.whole(registration.blocks))

    def serve(self, plan: receipts.Plan) -> None:
        """Each step from now on reads `plan`'s units: the store's blocks up to its last unit's, every tensor's."""
        self.plan = plan
        self.units = max(f + c for f, c in plan.runs)
        self.pieces = pieces(self.lay, UNIT * self.units)

    def step(self, seed: bytes) -> tuple[Any, dict]:
        """One step under `seed`: σ on the device of every unit up to the plan's last, and the step's timings."""
        torch, S = self.torch, self.S
        bps = self.lay.params.blocks_per_segment
        sk = Sketch(torch, seed, self.registration.rows, self.units, self.device)
        times = {"rows_s": 0.0, "decode_s": 0.0, "skipped_blocks": 0}
        t0 = time.perf_counter()
        read = [p for p in self.pieces if p.tensor is None or p.tensor.layer != self.skip_layer]
        times["skipped_blocks"] = sum(len(p.blocks) for p in self.pieces) - sum(len(p.blocks) for p in read)
        segments = sorted({s for p in read for s in range(p.blocks.start // bps, (p.blocks.stop - 1) // bps + 1)})
        with ThreadPoolExecutor(self.readers) as pool:
            segs = S.Segments(self.store, bps, segments, pool, self.readers)
            for p in read:
                a = time.perf_counter()
                rows = segs.rows(p.blocks)
                S.sync()
                b = time.perf_counter()
                if p.tensor is None:
                    for j0 in range(p.blocks.start, p.blocks.stop, self.chunk):
                        sk.add(rows[j0 - p.blocks.start:j0 - p.blocks.start + self.chunk], j0)
                else:
                    t = p.tensor
                    store = S.EncodedRows(self.lay, rows, t.first_block, self.lib, chunk=self.chunk)
                    store.sketch = sk
                    w = torch.empty(t.n, t.k, dtype=torch.int8, device=self.device)
                    S.decode_matrix(store, t.layer, t.name, w)
                    del w, store
                S.sync()
                times["rows_s"] += b - a
                times["decode_s"] += time.perf_counter() - b
                del rows
        times["step_s"] = time.perf_counter() - t0
        times["covered_units"] = sk.covered(self.plan.runs)
        times["sketched_blocks"] = sk.blocks
        times["int_mm"] = sk.int_mm and sk.checked
        if self.skip_layer is None and times["covered_units"] != self.plan.length:
            raise RuntimeError(f"the step sketched {times['covered_units']} of the plan's {self.plan.length} units whole")
        return sk.sigma, times

    def unit_bytes(self, g: int) -> bytes:
        return self.L.segment_rows(self.store, self.lay.spec, UNIT * g, UNIT)


class Prover:
    """The prover's side of `p2_party_prover.prove` over an encoded store served by `Engine`."""

    def __init__(self, engine: Engine, vk: VkPaths, pool: ProcessPoolExecutor | None, log=print,
                 spot: int = 16, cache: bool = False) -> None:
        self.engine, self.vk, self.pool, self.log, self.spot = engine, vk, pool, log, spot
        self.registration = engine.registration
        self.cache, self.cached = cache, None
        self.passes: dict[tuple[int, int], tuple[np.ndarray, bytes, ReceiptTree]] = {}
        self.steps: list[dict] = []
        self.opened: list[dict] = []
        self.received: dict[int, coins.Released] = {}
        self.kept: dict | None = None

    def case(self, doc: dict) -> None:
        self.kept = doc

    def hold(self, registration: dict) -> None:
        if p2_parties.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not this store's")
        self.engine.serve(p2_parties.plan_of(registration, self.registration))

    def end(self, commitment: coins.LiveCommitment, openings: list[coins.LiveOpening]) -> None:
        why = coins.received_reason(commitment, [self.received[t] for t in sorted(self.received)], openings)
        if why is not None:
            raise ValueError(f"the session's coins are not the ones this prover was sent: {why}")

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray, cheat: bool) -> int:
        """A few positions' σ held to `p2.Sketcher` on their unit's bytes from the store (not on a pass that `cheat`s)
        and their y to hashlib: the plan's first, its last and `spot` sampled from the seed."""
        plan, rows = self.engine.plan, self.registration.rows
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        positions = sorted({0, plan.length - 1, *rng.integers(0, plan.length, self.spot).tolist()})
        sk = p2.Sketcher(seed, rows)
        bad = 0
        for p in positions:
            g = plan.block(p)
            got = bytes(sigma[g].cpu().numpy().astype("<i4").tobytes())
            if not cheat and got != sk.sketch(self.engine.unit_bytes(g)):
                bad += 1
            if bytes(ys[p]) != sk.receipt(g, got):
                bad += 1
        if bad:
            raise RuntimeError(f"{bad} of {len(positions)} spot units' σ or y are not the reference's")
        return len(positions)

    def run(self, challenge: coins.Released, t: int, i: int, previous: bytes) -> bytes:
        if challenge.label != coins.CHALLENGE or self.received.setdefault(challenge.index, challenge) != challenge:
            raise ValueError(f"not window {challenge.index}'s one challenge")
        seed = receipts.seed(coins.coin(challenge), i, previous)
        plan = self.engine.plan
        t0 = time.perf_counter()
        cached = self.cached is not None
        if cached:
            sigma, times = self.cached, {"cached": True, "step_s": 0.0}
        else:
            sigma, times = self.engine.step(seed)
            if self.cache:
                self.cached = sigma
        a = time.perf_counter()
        parts = [receipts_of(self.engine.torch, seed, sigma[f:f + c], first=f) for f, c in plan.runs]
        ys = parts[0] if len(parts) == 1 else np.concatenate(parts)
        del parts
        b = time.perf_counter()
        spotted = self._spot(seed, sigma, ys, cheat=cached or self.engine.skip_layer is not None)
        del sigma
        c = time.perf_counter()
        salts = os.urandom(SALT * receipts.leaves(plan.length))
        tree = ReceiptTree(seed, plan, ys, salts, self.pool)
        d = time.perf_counter()
        self.passes[t, i] = (ys, salts, tree)
        rec = {"window": t, "pass": i, **times, "receipts_s": round(b - a, 3), "spot_units": spotted,
               "spot_s": round(c - b, 3), "tree_s": round(d - c, 3), "pass_s": round(d - t0, 3),
               "root": tree.root.hex()[:16]}
        self.steps.append(rec)
        self.log(json.dumps({"pass": rec}))
        return tree.root

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        t0 = time.perf_counter()
        plan = self.engine.plan
        blocks = sorted({UNIT * plan.block(p) + b for _, p in draws for b in range(UNIT)})
        paths = self.vk.paths(blocks)
        out = []
        for i, p in draws:
            ys, salts, tree = self.passes[t, i]
            c, g = p // PER_LEAF, plan.block(p)
            unit = self.engine.unit_bytes(g)
            out.append(p2.Opening(i, p, tuple(unit[BLOCK * b:BLOCK * (b + 1)] for b in range(UNIT)),
                                  tuple(paths[UNIT * g + b] for b in range(UNIT)),
                                  ys[PER_LEAF * c:min(PER_LEAF * (c + 1), plan.length)].tobytes(),
                                  salts[SALT * c:SALT * (c + 1)], tree.path(c)))
        for key in [k for k in self.passes if k[0] <= t]:
            del self.passes[key]
        rec = {"window": t, "openings": len(out), "open_s": round(time.perf_counter() - t0, 3)}
        self.opened.append(rec)
        self.log(json.dumps({"open": rec}))
        return out


WINDOWS, PASSES, SAMPLES = (p2_parties.SESSION[k] for k in ("windows", "passes", "samples"))
WINDOW_S, DEADLINE_S = p2_parties.SESSION["window_ns"] / 1e9, p2_parties.SESSION["deadline_ns"] / 1e9
BINDING_TAG = b"verity/remote-bandwidth/served-window/v1"
#: the session's record, and its case until `decide` answers it, in the private directory
SESSION, HELD = "session.json", "held-case.json"
NO_KEY = ("no key is named for the bandwidth session's close: the configuration pins none for the bandwidth verifier's "
          "bandwidth sessions, and the run names none")
SCOPE = ("beside the served window: the session ran on the card after window {k} was served and PoUS's session bound the "
         "store, over that store as the PoUS piece encoded and held it, and was not interleaved with the engine's reads "
         "(serving is not hooked), so it certifies the session's own reads of the store, not the window's")


def binding(model: str, revision: str, window: int) -> bytes:
    """What the public answer's hiding commitment is bound to: the served model and the window."""
    doc = {"model": model, "revision": revision, "window": window}
    return hashlib.sha512(BINDING_TAG + canonical_json_bytes(doc)).digest()


def request(node: bytes, *, windows: int = WINDOWS, passes: int = PASSES, samples: int = SAMPLES,
            window_s: float = WINDOW_S, deadline_s: float = DEADLINE_S) -> dict[str, Any]:
    return p2_parties.request_of(node, windows=windows, passes=passes, samples=samples, window_ns=int(window_s * 1e9),
                               deadline_ns=int(deadline_s * 1e9))


def _no(verdict: str, reason: str) -> dict[str, Any]:
    return {"commitment": None, "profile": None, "protocol": p2_parties.PROTOCOL, "public": None, "reason": reason,
            "verdict": verdict}


def _write(path: Path, doc: Any, mode: int | None = None) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    if mode is not None:
        os.chmod(path, mode)


#: who graded the bandwidth audit's answer, its placement's ``graded_by``
GRADED_BY = ("graded by the developer's piece (benchmarks/pous/served_piece.py --bw-case-exe: poub-p2-protocol's "
             "replay of the case) on the served node; no auditor-side entry runs it; a live verdict on the auditor's "
             "device exists (poub-verifier, its bandwidth party) and is not the one this row reads")


def public_answer(judged: Mapping[str, Any], bind: bytes, scope: str, r_bytes_per_s: int | None,
                  private: Path) -> dict[str, Any]:
    """The case's answer `judged` in public: its commitment a hiding one to the case's digest under `bind`, whose
    opening goes to `private`, its profile without the registration's digest and with `scope`, a placement holding
    only `memory_attester.UNCHECKED` and who graded it (`GRADED_BY`), and γ against R. A refusal with no commitment,
    the Lean's of a case it decodes no commitment of, has none to hide."""
    if judged.get("commitment") is None and judged.get("verdict") != "accept":
        return dict(judged) | {"reason": f"{judged.get('reason')}; {scope}"}
    row = bytes.fromhex(judged["commitment"])
    hidden = advice.commit([row], 8 * len(row), bind)
    o = hidden.open(0)
    _write(private / "opening.json", {"binding": bind.hex(), "count": 1, "index": o.index, "n_bits": 8 * len(row),
                                      "path": [p.hex() for p in o.path], "root": hidden.root.hex(), "row": row.hex(),
                                      "salt": o.salt.hex()}, 0o600)
    out = dict(judged) | {"commitment": hidden.root.hex()}
    if judged.get("verdict") != "accept":
        out["reason"] = f"{judged.get('reason')}; {scope}"
        return out
    public = dict(judged["public"])
    profile = {k: v for k, v in judged["profile"].items() if k != "registration"} | {
        "scope": scope, "eps_terms": p2_case.EPS_TERMS, "named_terms": ledger.NAMED_TERMS,
        "placement": mc.unchecked() | {"graded_by": GRADED_BY},
        "commitment": {"scheme": "hm96-sha512/row/v2 in frame-v3-sha512 (verity.commitments.advice)", "rows": 1,
                       "n_bits": 8 * len(row), "binding": bind.hex(),
                       "of": "the session's case digest (p2_case.digest), its registration and plan"}}
    if r_bytes_per_s is not None and public["span_ns"] > 0:
        from fractions import Fraction

        public |= {"gamma": str(1 - Fraction(public["q_bytes"] * 10**9, r_bytes_per_s * public["span_ns"])),
                   "r_bytes_per_s": r_bytes_per_s}
        profile["gamma_terms"] = p2_parties.GAMMA_TERMS
    return out | {"public": public, "profile": profile}


def key_of(deployment: Mapping[str, Any], trusted_sha512: str | None = None) -> str | None:
    """`trusted_sha512`, a rehearsal's, else the configuration's pin for the close."""
    pinned = mc.pinned(deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT)
    return trusted_sha512 or min(pinned, default=None)


def key_record(case: Mapping[str, Any], trusted_sha512: str, deployment: Mapping[str, Any]) -> dict[str, Any]:
    """``key_pin`` is ``run``: the pins this machine was handed are used only to refuse, and ``pins_matched`` says
    whether they give the key `trusted_sha512` names for the close; the premise names the close's signer's machine by
    its party (`memory_attester.ATTESTER_MACHINE`), never by the address the party was reached at."""
    key = p2_case.close_key(case)
    key_sha = None if key is None else hashlib.sha512(key).hexdigest()
    pinned = mc.pinned(deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT)
    return {"key_sha512": key_sha, "key_pinned": key_sha == trusted_sha512,
            "key_pin": "run", "pins_matched": trusted_sha512 in pinned,
            mc.PREMISE: mc.trusted_device(
                deployment["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT, key_sha,
                f"the bandwidth verifier, which signs the close with the memory attester's key on {mc.ATTESTER_MACHINE}",
                configuration=deployment["configuration"], committed=deployment["committed"])}


def decide(private: Path, public: Path, *, bind: bytes, window: int, pous_case: Mapping[str, Any] | None = None,
           case_exe: str | None = None, trusted_sha512: str | None = None,
           deployment: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The answer to the session `held` left in `private`, with no store and no card: the case and its answer to
    `private` (`p2_case.write`), the public answer and a summary to `public` (``answer.json``, ``bandwidth.json``); the
    public answer, also returned. With `pous_case` (PoUS's run case), the case is held to PoUS's store
    (`p2_case.storing`), which the Lean refuses a session over another store by. With `case_exe` (``poub-p2-protocol``) the Lean answers the case and
    its answer is the one written, in ``lean_s`` seconds; a Lean that doesn't answer leaves the session undecided and no
    case written; without it, a session that made no case and a close that doesn't decode are undecided, never a verdict
    of this Python's (`p2_case.unjudged`). With no key `key_of` gives, the session is undecided (`NO_KEY`). Under a key
    the configuration doesn't pin (``--run-keys``' or one given by hand), the public answer is ``development.json``
    and no ``answer.json`` (`period.publish`). The summary names the program whose verdict the answer is, its digest
    and the machine it ran on, the prover's (``decided_by``, `period.decided_by`): the Lean where it answered the case,
    else this Python, with ``case_exe``'s SHA-512 (``executables``) and the card the session was against (``card``):
    the node writes both about itself, and they are held only where an auditor-side composer compares them. `HELD` stays, so the same session can be answered again; the caller removes it once it needs no
    other answer."""
    public.mkdir(parents=True, exist_ok=True)
    scope = SCOPE.format(k=window)
    deployed = period.deployment({}) if deployment is None else deployment
    trusted_sha512 = key_of(deployed, trusted_sha512)
    doc = json.loads((private / SESSION).read_text())
    answer, steps = doc["answer"], doc["steps"] or []
    r_bytes_per_s = doc["r_bytes_per_s"]
    kept = json.loads((private / HELD).read_text()) if (private / HELD).is_file() else None
    request = {k: v for k, v in doc["request"].items() if k != "node"}   # the node's name: in the private record only
    summary: dict[str, Any] = {"window": window, "request": request,
                               "pous_session": "the PoUS session the PoUS piece ran (its id private)",
                               "private": str(private), "decided_by": period.decided_by(None),
                               "executables": period.executables(case_exe), "card": doc.get("card"),
                               "setup_s": doc["setup_s"], "session_wall_s": doc["session_wall_s"],
                               "verifier_rc": doc["verifier_rc"], "passes": len(steps),
                               "pass_s": [s["pass_s"] for s in steps]}
    if trusted_sha512 is None:
        out = _no("undecided", f"{NO_KEY}; {scope}")
    elif kept is None:
        out = _no("undecided", f"the session made no case: {answer.get('verdict')}: {answer.get('reason')}"[:1000]
                  + f"; {scope}")
    else:
        summary["same_store"] = pous_case is not None
        try:
            kept = p2_case.trusting(kept, trusted_sha512)
            kept = kept if pous_case is None else p2_case.storing(kept, pous_case)
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            out = _no("undecided", f"the session's close doesn't decode: {type(e).__name__}: {e}"[:1000] + f"; {scope}")
        else:
            summary |= key_record(kept, trusted_sha512, deployed)
            t = time.perf_counter()
            try:
                judged = (p2_case.judged(kept, answer, case_exe) if case_exe is not None
                          else p2_case.unjudged(kept))
            except (OSError, IndexError, ValueError, subprocess.CalledProcessError) as e:
                out = _no("undecided",
                          f"{case_exe} doesn't answer the case: {type(e).__name__}: {e}"[:1000] + f"; {scope}")
            else:
                if case_exe is not None:
                    summary |= {"decided_by": period.decided_by(case_exe), "lean_s": round(time.perf_counter() - t, 3)}
                p2_case.write(private, kept, judged)
                out = public_answer(judged, bind, scope, r_bytes_per_s, private)
    if out.get("reason"):
        out = out | {"reason": mc.unnamed(out["reason"], (doc["request"].get("node"), socket.gethostname()),
                                          (doc.get("party_host"),))}
    pinned = mc.pinned(deployed["pins"], p2_parties.CLOSER_PARTY, p2_parties.COMPONENT)
    development = period.development_of(trusted_sha512, pinned)
    period.publish(public, out, development)
    _write(public / "bandwidth.json", summary | {"development": development, "verdict": out["verdict"],
                                                 "reason": out["reason"]})
    return out


def missing(public: Path, window: int, reason: str) -> dict[str, Any]:
    """No session ran: the public answer undecided, saying why, by this Python (``decided_by``)."""
    public.mkdir(parents=True, exist_ok=True)
    out = _no("undecided", f"{reason}; {SCOPE.format(k=window)}")
    period.publish(public, out, None)
    _write(public / "bandwidth.json", {"window": window, "verdict": out["verdict"], "reason": out["reason"],
                                       "decided_by": period.decided_by(None)})
    return out


def line(answer: Mapping[str, Any]) -> str:
    if answer["verdict"] == "accept":
        return f"BANDWIDTH: accept: {json.dumps(answer['public'], sort_keys=True)}"
    return f"BANDWIDTH: {answer['verdict']}: {answer['reason']}"

#: the bandwidth audit's `answers` entry (`verity.protocols.accounting.scheme.Answers`'s fields)
ANSWERS = {"held": HELD, "binding": binding, "decide": decide, "missing": missing, "line": line, "request": request,
           "session": {"windows": WINDOWS, "passes": PASSES, "samples": SAMPLES, "window_s": WINDOW_S,
                       "deadline_s": DEADLINE_S},
           "key_of": key_of}



class Held:
    """`p2_party_prover.Peer` over the adapter's store ``store`` (`weight_store.Store`): ``C*``'s rows on
    their device, the challenger's table of their digests, setup's salt, commitment and root, under the node ``node``."""

    def __init__(self, store: Any, node: bytes, *, chunk: int = 1 << 16, spot: int = 16,
                 log: Callable[[str], None] = lambda _m: None) -> None:
        import torch

        self.torch, self.rows, self.chunk, self.spot, self.log = torch, store.rows, chunk, spot, log
        params = store.params
        self.registration = p2.Registration(node, p2.codec_of(params), store.commitment, store.salt, store.root)
        digests = store.digests
        self.tree = V.MerkleTree([V.row_leaf(params, store.salt, j, digests[DIGEST * j:DIGEST * (j + 1)])
                                  for j in range(params.blocks)])
        if self.tree.root != store.root:
            raise ValueError("the challenger's table of the held C* is not setup's root")
        self.plan = receipts.Plan.whole(self.registration.blocks)
        self.passes: dict[tuple[int, int], tuple[np.ndarray, bytes, ReceiptTree]] = {}
        self.received: dict[int, coins.Released] = {}
        self.steps: list[dict[str, Any]] = []
        self.opened: list[dict[str, Any]] = []
        self.kept: dict[str, Any] | None = None

    def case(self, doc: dict[str, Any]) -> None:
        self.kept = doc

    def hold(self, registration: Mapping[str, Any]) -> None:
        if p2_parties.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not the held store's")
        self.plan = p2_parties.plan_of(registration, self.registration)

    def end(self, commitment: coins.LiveCommitment, openings: Sequence[coins.LiveOpening]) -> None:
        why = coins.received_reason(commitment, [self.received[t] for t in sorted(self.received)], openings)
        if why is not None:
            raise ValueError(f"the session's coins are not the ones this prover was sent: {why}")

    def unit(self, g: int) -> bytes:
        return self.rows[UNIT * g:UNIT * (g + 1)].cpu().numpy().tobytes()

    def sigma(self, seed: bytes) -> Any:
        """σ of every unit up to the plan's last, from the held rows on their device."""
        torch, units = self.torch, max(f + c for f, c in self.plan.runs)
        torch.backends.cuda.matmul.allow_tf32 = False
        sk = Sketch(torch, seed, self.registration.rows, units, self.rows.device)
        for j0 in range(0, UNIT * units, self.chunk):
            sk.add(self.rows[j0:min(j0 + self.chunk, UNIT * units)], j0)
        if sk.covered(self.plan.runs) != self.plan.length:
            raise RuntimeError(f"the pass sketched {sk.covered(self.plan.runs)} of the plan's {self.plan.length} units")
        return sk.sigma

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray) -> int:
        """The plan's first and last positions and `spot` sampled from the seed: σ held to `p2.Sketcher` on the unit's
        bytes and y to hashlib."""
        plan, sk = self.plan, p2.Sketcher(seed, self.registration.rows)
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        positions = sorted({0, plan.length - 1, *rng.integers(0, plan.length, self.spot).tolist()})
        for p in positions:
            g = plan.block(p)
            got = bytes(sigma[g].cpu().numpy().astype("<i4").tobytes())
            if got != sk.sketch(self.unit(g)) or bytes(ys[p]) != sk.receipt(g, got):
                raise RuntimeError(f"unit {g}'s σ or y is not the reference's")
        return len(positions)

    def run(self, challenge: coins.Released, t: int, i: int, previous: bytes) -> bytes:
        if challenge.label != coins.CHALLENGE or self.received.setdefault(challenge.index, challenge) != challenge:
            raise ValueError(f"not window {challenge.index}'s one challenge")
        seed = receipts.seed(coins.coin(challenge), i, previous)
        t0 = time.perf_counter()
        sigma = self.sigma(seed)
        a = time.perf_counter()
        parts = [receipts_of(self.torch, seed, sigma[f:f + c], first=f) for f, c in self.plan.runs]
        ys = parts[0] if len(parts) == 1 else np.concatenate(parts)
        b = time.perf_counter()
        spotted = self._spot(seed, sigma, ys)
        del sigma
        salts = os.urandom(SALT * receipts.leaves(self.plan.length))
        tree = ReceiptTree(seed, self.plan, ys, salts)
        self.passes[t, i] = (ys, salts, tree)
        rec = {"window": t, "pass": i, "sketch_s": round(a - t0, 3), "receipts_s": round(b - a, 3),
               "spot_units": spotted, "pass_s": round(time.perf_counter() - t0, 3)}
        self.steps.append(rec)
        self.log(json.dumps({"pass": rec}))
        return tree.root

    def open(self, t: int, draws: Sequence[tuple[int, int]]) -> list[p2.Opening]:
        t0 = time.perf_counter()
        out = []
        for i, p in draws:
            ys, salts, tree = self.passes[t, i]
            c, g = p // PER_LEAF, self.plan.block(p)
            unit = self.unit(g)
            out.append(p2.Opening(i, p, tuple(unit[BLOCK * b:BLOCK * (b + 1)] for b in range(UNIT)),
                                  tuple(self.tree.path(UNIT * g + b) for b in range(UNIT)),
                                  ys[PER_LEAF * c:min(PER_LEAF * (c + 1), self.plan.length)].tobytes(),
                                  salts[SALT * c:SALT * (c + 1)], tree.path(c)))
        for key in [k for k in self.passes if k[0] <= t]:
            del self.passes[key]
        rec = {"window": t, "openings": len(out), "open_s": round(time.perf_counter() - t0, 3)}
        self.opened.append(rec)
        self.log(json.dumps({"open": rec}))
        return out


@dataclass(eq=False)
class Attached:
    """``bandwidth`` on a served run beside ``storage``, PoUS's setup on the same exchange
    (`verity.protocols.accounting.memory.pous.prover.served.Served`), which hands it the store once it binds one (`Served.on_bound`)."""

    storage: Any
    ctx: Mapping[str, Any]
    store: Any = None

    def __post_init__(self) -> None:
        self.storage.on_bound(self.bound)

    def bound(self, store: Any) -> None:
        self.store = store

    def peer(self, node: bytes, **kw: Any) -> Held:
        """P2's prover over the store PoUS holds, under ``node``, with `Held`'s keywords; refused before PoUS binds
        one."""
        if self.store is None:
            raise RuntimeError(f"bandwidth: {self.storage.entry} holds no store yet, and the audit reads the one it "
                               "holds")
        return Held(self.store, node, **kw)


    def session(self, *, node: str, card: str, r_bytes_per_s: int, party: str, party_host: str, sid: bytes,
                request_: Mapping[str, Any], private: Path, trusted_sha512: str | None = None) -> None:
        """One PoUB-P2 session over the store PoUS holds with the bandwidth party `party` (a shell command, to which
        ``--pous-session`` and PoUS's session id `sid` are added, and ``--session`` and a fresh id for this session,
        under which the memory attester signs the case's close; without `trusted_sha512`, no session has a case; its
        port reached at `party_host`), under `request_`: the session's record (`SESSION`: the answer the party sent, the registration, each pass and opening, the seconds) and, where the session made one, its
        case not yet answered (`HELD`), with the logs, to `private` (0700, files 0600), for `decide`. γ is against
        `r_bytes_per_s`, the data sheet's stated memory bandwidth of `card`, which the launcher reads."""
        private.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(private, 0o700)
        (private / HELD).unlink(missing_ok=True)   # an earlier session's case is never this one's
        p2_session = None if trusted_sha512 is None else secrets.token_bytes(32)
        command = f"{party} --pous-session {sid.hex()}"
        if p2_session is not None:
            command += f" --session {p2_session.hex()} --trusted-sha512 {trusted_sha512}"
        with open(private / "prover.log", "a") as log_f:

            def log(m: str) -> None:
                print(m, file=log_f, flush=True)

            t0 = time.perf_counter()
            rc = None
            prover = None
            try:
                prover = self.peer(node.encode(), log=log)
                setup_s = time.perf_counter() - t0
                answer, rc = _party(SimpleNamespace(party=command, party_host=party_host), private, prover,
                                              request_)
            except Exception as e:  # noqa: BLE001 - the prover's own failure, its answer, never the verifier's
                log(json.dumps({"prover_error": f"{type(e).__name__}: {e}"}))
                answer, setup_s = {"verdict": "prover-error", "reason": f"{type(e).__name__}: {e}"[:1000]}, None
            wall = time.perf_counter() - t0
        if prover is not None and prover.kept is not None:
            fd = os.open(private / HELD, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(canonical_json_bytes(dict(prover.kept)) + b"\n")
        _write(private / SESSION, {
            "answer": answer, "verifier_rc": rc, "request": dict(request_),
            "registration": None if prover is None else p2_parties.registration_json(prover.registration, prover.plan),
            "steps": None if prover is None else prover.steps, "opened": None if prover is None else prover.opened,
            "pous_session": sid.hex(), "session": None if p2_session is None else p2_session.hex(), "party": command,
            "party_host": party_host, "card": card, "r_bytes_per_s": r_bytes_per_s,
            "setup_s": None if setup_s is None else round(setup_s, 3), "session_wall_s": round(wall, 3)}, 0o600)



def registration_file(reg: p2.Registration, path: Path, plan: receipts.Plan | None = None) -> None:
    path.write_text(json.dumps(p2_parties.registration_json(reg, plan), indent=1, sort_keys=True) + "\n")


# --- the session -------------------------------------------------------------------------------------------------------

def verdict_line(answer: dict, expect: str | None) -> str:
    """The session's verdict line, as the prover received the answer: a measurement's, which no verdict reads."""
    v = answer.get("verdict")
    ok = expect is None or {"accept": "accept", "reject": "reject"}[expect] == v
    parts = [f"BW VERDICT {v}"]
    if v == "accept":
        rate = answer.get("credited_bytes_per_s")
        parts += [f"q_bytes={answer['q_bytes']}", f"receipts={answer['receipts']}",
                  f"wrong_at_most={answer['wrong_at_most']}", f"span_s={answer['span_ns'] / 1e9:.3f}",
                  f"credited_GB_s={rate / 1e9:.3f}" if rate else "credited_GB_s=None"]
    else:
        parts += [f"failure={answer.get('failure')}", f"window={answer.get('window')}", f"at={answer.get('at')}",
                  f"reason={answer.get('reason')!r}"]
    parts += [f"verifier_host={answer.get('host')}", f"expect={expect}"]
    return " ".join([*parts, "PASS" if ok else "FAIL"])


def _party(a: argparse.Namespace, out: Path, prover: Prover, request: dict) -> tuple[dict, int]:
    """A session with a party the auditor's gate starts (`--party CMD`, `p2_parties --listen` on the gate's host): its
    `{"listening": port}` read from CMD's stdout, then the session over TCP to `--party-host` at that port. The gate kills a party once its session's minutes run out
    (`bw_party --minutes M`) and says so on that stdout, `{"killed": why}`: a session cut off so fails naming it."""
    with open(out / "verifier.err", "ab") as err:
        proc = subprocess.Popen(a.party, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        killed = None

        def rest() -> None:
            nonlocal killed
            for line in proc.stdout:
                doc = json.loads(line)
                if isinstance(doc, dict) and "killed" in doc:
                    killed = doc["killed"]

        try:
            port = None
            for line in proc.stdout:
                doc = json.loads(line)
                if isinstance(doc, dict) and "listening" in doc:
                    port = int(doc["listening"])
                    break
            if port is None:
                raise ConnectionError("the party ended without listening")
            try:
                with (socket.create_connection((a.party_host, port)) as conn, conn.makefile("rb") as r,
                      conn.makefile("wb") as w):
                    answer = p2_party_prover.prove(r, w, prover, request)
            except (OSError, ValueError) as e:
                rest()
                if killed is not None:
                    raise ConnectionError(f"the auditor's gate killed the party: {killed}") from e
                raise
            rest()
        finally:
            rc = proc.wait()
    return answer, rc


def session(a: argparse.Namespace) -> int:
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) if a.workers > 1 else None
    if pool is not None:
        list(pool.map(abs, range(a.workers)))  # a fork pool starts every worker at its first task: before CUDA does
    node = a.node.encode()
    reg = registration_of_store(a.store, node)
    if a.registration is not None and p2_parties.registration_of(json.loads(a.registration.read_text())) != reg:
        raise SystemExit("the store's registration is not --registration's")
    with open(out / "prover.log", "a") as log_f:

        def log(m: str) -> None:
            print(m, file=log_f, flush=True)
            print(m, file=sys.stderr, flush=True)

        t0 = time.perf_counter()
        vk = VkPaths(a.store, reg.params, reg.salt, reg.root, pool)
        engine = Engine(a.store, a.layout, a.lib, reg, device=a.device, chunk=a.chunk, readers=a.readers,
                        skip_layer=a.skip_layer, ref_decode=a.ref_decode)
        prover = Prover(engine, vk, pool, log, cache=a.cache)
        setup_s = time.perf_counter() - t0
        request = p2_parties.request_of(node, windows=a.windows, passes=a.passes, samples=a.samples,
                                      window_ns=int(a.window_s * 1e9), deadline_ns=int(a.deadline_s * 1e9))
        log(json.dumps({"start": {"node": a.node, "units": reg.blocks, "registration": reg.digest.hex(),
                                  "request": request, "setup_s": round(setup_s, 3), "host": socket.gethostname(),
                                  "skip_layer": a.skip_layer, "cache": a.cache}}))
        rc = None
        t1 = time.perf_counter()
        try:
            if a.party is not None:
                answer, rc = _party(a, out, prover, request)
            elif a.connect is not None:
                host, _, port = a.connect.rpartition(":")
                with (socket.create_connection((host, int(port))) as conn, conn.makefile("rb") as r,
                      conn.makefile("wb") as w):
                    answer = p2_party_prover.prove(r, w, prover, request)
            else:
                with open(out / "verifier.err", "ab") as err:
                    proc = subprocess.Popen(a.verifier, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=err)
                    try:
                        answer = p2_party_prover.prove(proc.stdout, proc.stdin, prover, request)
                    finally:
                        proc.stdin.close()
                        rc = proc.wait()
        except Exception as e:  # noqa: BLE001 - the prover's own failure, recorded as its answer, never the verifier's
            log(json.dumps({"prover_error": traceback.format_exc()}))
            answer = {"verdict": "prover-error", "reason": f"{type(e).__name__}: {e}"}
        wall = time.perf_counter() - t1
        case = None
        if a.case_dir is not None and prover.kept is not None and answer.get("verdict") in ("accept", "reject",
                                                                                           "undecided"):
            judged = (p2_case.judged(prover.kept, answer, a.case_exe) if a.case_exe
                      else p2_case.unjudged(prover.kept))
            p2_case.write(a.case_dir, prover.kept, judged)
            case = {"dir": str(a.case_dir), "commitment": judged["commitment"],
                    "store_root": prover.kept["params"]["store_root"], "verdict": judged["verdict"], "judge": "lean" if a.case_exe else "none"}
            log(json.dumps({"case": case}))
    if pool is not None:
        pool.shutdown()
    line = verdict_line(answer, a.expect)
    verifier = (f"party {a.party} at {a.party_host}" if a.party else a.verifier or f"tcp {a.connect}")
    doc = {"answer": answer, "request": request, "registration": p2_parties.registration_json(reg, engine.plan),
           "steps": prover.steps,
           "opened": prover.opened, "setup_s": round(setup_s, 3), "session_wall_s": round(wall, 3),
           "verifier_rc": rc, "verifier": verifier, "prover_host": socket.gethostname(),
           "skip_layer": a.skip_layer, "cache": a.cache, "store": str(a.store), "device": a.device, "verdict_line": line,
           "case": case}
    (out / "session.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(line, flush=True)
    return 0 if line.endswith("PASS") else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("registration")
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--layout", type=Path, required=True, help="the store's layout JSON, whose tensors' units are the plan")
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("session")
    s.add_argument("--store", type=Path, required=True, help="the encoded store's directory, read only")
    s.add_argument("--layout", type=Path, required=True, help="its layout JSON (p2_store/store.py layout)")
    s.add_argument("--lib", default="unused", help="libp2v1.so's directory")
    s.add_argument("--node", required=True)
    s.add_argument("--registration", type=Path, help="the registration the store must have (registration_json)")
    to = s.add_mutually_exclusive_group(required=True)
    to.add_argument("--verifier", help="the shell command that starts p2_parties --stdio")
    to.add_argument("--connect", help="HOST:PORT where p2_parties --listen waits (the gate's port, through ssh -L)")
    to.add_argument("--party", help="the shell command (ssh to the auditor's gate) that starts p2_parties --listen there")
    s.add_argument("--party-host", help="with --party: the address its port is reached at from here")
    s.add_argument("--out", type=Path, required=True)
    held = p2_parties.SESSION
    s.add_argument("--windows", type=int, default=held["windows"], help="default the auditor's (`p2_parties.SESSION`)")
    s.add_argument("--passes", type=int, default=held["passes"])
    s.add_argument("--samples", type=int, default=held["samples"])
    s.add_argument("--window-s", type=float, default=held["window_ns"] / 1e9)
    s.add_argument("--deadline-s", type=float, default=held["deadline_ns"] / 1e9)
    cheat = s.add_mutually_exclusive_group()
    cheat.add_argument("--skip-layer", type=int, help="the cheat that neither reads nor decodes layer L's tensors")
    cheat.add_argument("--cache", action="store_true", help="the cheat that replays the first pass's σ every later pass")
    s.add_argument("--expect", choices=("accept", "reject"))
    s.add_argument("--device", default="cuda")
    s.add_argument("--ref-decode", action="store_true")
    s.add_argument("--chunk", type=int, default=1 << 16)
    s.add_argument("--readers", type=int, default=6)
    s.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    s.add_argument("--case-dir", type=Path, help="the private directory the session's core case and its answer go to")
    s.add_argument("--case-exe", help="poub-p2-protocol, which answers the case (else it is "
                   "undecided: p2_case.unjudged)")
    a = ap.parse_args(argv)
    if a.command == "session" and a.case_dir is not None and a.case_dir.resolve().is_relative_to(a.out.resolve()):
        raise SystemExit("--case-dir is private: it can't be under --out")
    if a.command == "registration":
        reg = registration_of_store(a.store, a.node.encode())
        plan = tensor_plan(json.loads(a.layout.read_text()), reg.blocks)
        registration_file(reg, a.out, plan)
        print(json.dumps({"registration": reg.digest.hex(), "units": reg.blocks, "plan_units": plan.length,
                          "root": reg.root.hex()}))
        return 0
    if (a.party is None) != (a.party_host is None):
        ap.error("--party and --party-host go together")
    return session(a)


if __name__ == "__main__":
    sys.exit(main())
