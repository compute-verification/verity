"""A remote bandwidth session over PoUS's P2 store on the card that serves from it: the prover's side of
`verity.core.protocols.remote_bandwidth.p2_party`, whose verifier runs on another host.

    bandwidth.py toy OUT                                  # the toy's C* as an encoded store, and its registration
    bandwidth.py registration --store ENC --node NODE --out FILE
    bandwidth.py session --store ENC --layout F --lib DIR --node NODE (--verifier CMD | --connect HOST:PORT) --out DIR
                 [--windows 2] [--passes 2] [--samples 1024] [--window-s 1] [--deadline-s 1]
                 [--skip-layer L] [--expect accept|reject] [--device cuda] [--ref-decode]

**The step.** Each pass of the session is one engine step's weight reads: every tensor of the store, in block order, from
its segment files onto the device (`full.Segments`, kernels' reader) and through kernels' staging decode
(`stage.decode_matrix`) into a staging buffer, as a forward step reads them. The decode reads each chunk's rows from
`stage.Store.decode_rows`; `SketchedStore` sketches those same device rows under the pass's seed just before they go to
the decode kernel (`Sketch.add`: π of each block by an exact int8 matmul against its four offsets' signs, summed into
its unit's σ), so the receipts are of the bytes the decode read. Blocks of no tensor (the store's tail) are read and
sketched alone, so every unit is covered. Then y = SHA512(prefix ‖ u64be g ‖ σ_g) for every unit on the device
(`receipts_of`, a torch SHA-512 after the prefix's midstate; a sample held to hashlib and to `p2.Sketcher` each pass), and
the pass's receipt tree (`ReceiptTree`, `receipts.tree`'s, in worker processes). The sketch is a pass of its own over
the chunk's rows, not fused into the decode kernel, so each byte is read from HBM twice.

**The session.** `session` starts the verifier with `--verifier CMD` (the ssh command that runs `p2_party --stdio` on
the auditor's host; on one host, the module itself, which certifies nothing) or reaches `p2_party --listen` at
`--connect HOST:PORT` (the auditor's gate port through `ssh -L`), sends the request and answers each pass,
each window's openings (each unit's four blocks from the segment files with their PoUS vk paths, `VkPaths`, from the
store's row digests) and returns the verifier's answer, written to `OUT/session.json` with each step's timings, and
the line `BW VERDICT ...`. `--skip-layer L` is the cheat: it neither reads nor decodes layer L's tensors, and the
receipts of the units they hold are from what was read, so a draw that lands there is rejected (`receipt`).

`--ref-decode` swaps the library for P2's reference decode on the CPU (`RefDecode`), for the toy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import socket
import struct
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = ROOT / "benchmarks" / "torch_gpu" / "p2_stage"
if __package__ in (None, ""):
    sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(STAGE))

from verity.core.protocols.pous import verifier as V
from verity.core.protocols.pous.labels import salt_value, tag
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import coins, p2, p2_party, receipts
from verity.ml.commitments.merkle import merkle_depth

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
    `j0 ..` ([n, 2056] uint8, as stored) into their units' σ, each block's π against its offset's sign columns; `seen`
    counts each unit's blocks sketched, four when the pass covered it."""

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
        self.checked = False

    def _mm(self, x: Any) -> Any:
        t = self.t
        if self.device.type == "cuda" and x.shape[0] > 16:
            out = t._int_mm(x, self.signs)
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

    def covered(self) -> int:
        """The units of every block sketched exactly once; each is four."""
        return int((self.seen == UNIT).sum())


def sketched_store(S: Any):
    """`stage.Store` whose `decode_rows` sketches each chunk's rows before the decode kernel reads them."""

    class SketchedStore(S.Store):
        def __init__(self, *args: Any, sketch: Sketch, **kw: Any) -> None:
            super().__init__(*args, **kw)
            self.sketch = sketch

        def decode_rows(self, j0: int, j1: int) -> Any:
            self.sketch.add(self.rows[j0 - self.first:j1 - self.first], j0)
            return super().decode_rows(j0, j1)

    return SketchedStore


class RefDecode:
    """P2's reference decode (`reference.decode` under `reference.keys_after`) behind `p2_v1.gpu.Kernels`, on the CPU:
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

def _leaves(args: tuple) -> bytes:
    seed, units, lo, hi, ys, salts = args
    d = receipts.domain(seed, receipts.Plan.whole(units))
    out = bytearray()
    for c in range(lo, hi):
        a, b = (PER_LEAF * c - PER_LEAF * lo) * DIGEST, (min(PER_LEAF * (c + 1), units) - PER_LEAF * lo) * DIGEST
        out += d.leaf(c, c, receipts.SCHEMA, receipts.hidden(ys[a:b], salts[SALT * (c - lo):SALT * (c - lo + 1)]))
    return bytes(out)


def _pads(args: tuple) -> bytes:
    seed, units, lo, hi = args
    d = receipts.domain(seed, receipts.Plan.whole(units))
    return b"".join(d.pad(r) for r in range(lo, hi))


def _nodes(args: tuple) -> bytes:
    seed, units, level, lo, below = args
    d = receipts.domain(seed, receipts.Plan.whole(units))
    return b"".join(d.node(level, lo + k, below[2 * DIGEST * k:2 * DIGEST * k + DIGEST],
                           below[2 * DIGEST * k + DIGEST:2 * DIGEST * (k + 1)]) for k in range(len(below) // (2 * DIGEST)))


def _run(pool: ProcessPoolExecutor | None, fn, tasks: list[tuple]) -> list[bytes]:
    return list(pool.map(fn, tasks)) if pool is not None and len(tasks) > 1 else [fn(x) for x in tasks]


class ReceiptTree:
    """`receipts.tree(seed, Plan.whole(units), ys, salts)` built level by level in `pool`'s processes: its `root`, and
    `path(c)`, leaf c's siblings from the leaf up, as `MerkleTree.open(c).path`."""

    CHUNK = 1 << 14

    def __init__(self, seed: bytes, units: int, ys: np.ndarray, salts: bytes,
                 pool: ProcessPoolExecutor | None = None) -> None:
        n = receipts.leaves(units)
        if ys.shape != (units, DIGEST) or len(salts) != SALT * n:
            raise ValueError(f"{units} receipts in {n} leaves")
        raw, C = ys.tobytes(), self.CHUNK
        tasks = [(seed, units, lo, min(n, lo + C), raw[PER_LEAF * DIGEST * lo:PER_LEAF * DIGEST * min(n, lo + C)],
                  salts[SALT * lo:SALT * min(n, lo + C)]) for lo in range(0, n, C)]
        width = 1 << merkle_depth(n)
        level = b"".join(_run(pool, _leaves, tasks))
        level += b"".join(_run(pool, _pads, [(seed, units, lo, min(width, lo + C)) for lo in range(n, width, C)]))
        self.levels = [level]
        depth = 0
        while len(level) > DIGEST:
            pairs = len(level) // (2 * DIGEST)
            tasks = [(seed, units, depth, lo, level[2 * DIGEST * lo:2 * DIGEST * min(pairs, lo + C)])
                     for lo in range(0, pairs, C)]
            level = b"".join(_run(pool, _nodes, tasks))
            self.levels.append(level)
            depth += 1
        self.root = level[:DIGEST]

    def path(self, c: int) -> tuple[bytes, ...]:
        return tuple(lv[DIGEST * ((c >> k) ^ 1):DIGEST * ((c >> k) ^ 1) + DIGEST]
                     for k, lv in enumerate(self.levels[:-1]))


def _segment_paths(args: tuple) -> dict[int, tuple[bytes, ...]]:
    rows_path, salt, s, bps, vs, root = args
    rows = Path(rows_path).read_bytes()
    if len(rows) != DIGEST * bps:
        raise ValueError(f"{rows_path} is not {bps} row digests")
    head = b"\x00" + tag(salt_value(salt), s).to_bytes(32, "little")
    level = [hashlib.sha512(head + v.to_bytes(2, "little") + rows[DIGEST * v:DIGEST * (v + 1)]).digest()
             for v in range(bps)]
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


class Engine:
    """One engine step's weight reads through kernels' staging decode (module docstring), its rows sketched."""

    def __init__(self, store: Path, layout: Path, lib: str, registration: p2.Registration, *, device: str = "cuda",
                 chunk: int = 1 << 16, readers: int = 6, skip_layer: int | None = None, ref_decode: bool = False) -> None:
        import full as F
        import layout as L
        import stage as S
        import torch

        self.torch, self.F, self.L, self.S = torch, F, L, S
        self.device = torch.device(device)
        F.DEVICE = self.device
        if ref_decode:
            S.gpu.Kernels = RefDecode
        torch.backends.cuda.matmul.allow_tf32 = False
        doc = json.loads((store / "store.json").read_text())
        self.lay, _ = L.open_store(layout, {}, store, doc["source"], doc["commitment"], None)
        if self.lay.params.blocks != registration.codec.blocks:
            raise ValueError("the layout's store is not the registration's")
        self.store, self.lib, self.chunk, self.readers = store, lib, chunk, readers
        self.registration, self.skip_layer = registration, skip_layer
        self.units = registration.blocks
        self.pieces = pieces(self.lay, UNIT * self.units)
        self.Sketched = sketched_store(S)

    def step(self, seed: bytes) -> tuple[Any, dict]:
        """One step under `seed`: every unit's σ on the device, and the step's timings."""
        torch, F, S = self.torch, self.F, self.S
        bps = self.lay.params.blocks_per_segment
        sk = Sketch(torch, seed, self.registration.rows, self.units, self.device)
        times = {"rows_s": 0.0, "decode_s": 0.0, "skipped_blocks": 0}
        t0 = time.perf_counter()
        read = [p for p in self.pieces if p.tensor is None or p.tensor.layer != self.skip_layer]
        times["skipped_blocks"] = sum(len(p.blocks) for p in self.pieces) - sum(len(p.blocks) for p in read)
        segments = sorted({s for p in read for s in range(p.blocks.start // bps, (p.blocks.stop - 1) // bps + 1)})
        with ThreadPoolExecutor(self.readers) as pool:
            segs = F.Segments(self.store, bps, segments, pool, self.readers)
            for p in read:
                a = time.perf_counter()
                rows = segs.rows(p.blocks)
                F.sync()
                b = time.perf_counter()
                if p.tensor is None:
                    for j0 in range(p.blocks.start, p.blocks.stop, self.chunk):
                        sk.add(rows[j0 - p.blocks.start:j0 - p.blocks.start + self.chunk], j0)
                else:
                    t = p.tensor
                    store = self.Sketched(self.lay, rows, t.first_block, self.lib, chunk=self.chunk, sketch=sk)
                    w = torch.empty(t.n, t.k, dtype=torch.int8, device=self.device)
                    S.decode_matrix(store, t.layer, t.name, w)
                    del w, store
                F.sync()
                times["rows_s"] += b - a
                times["decode_s"] += time.perf_counter() - b
                del rows
        times["step_s"] = time.perf_counter() - t0
        times["covered_units"] = sk.covered()
        times["sketched_blocks"] = sk.blocks
        if self.skip_layer is None and times["covered_units"] != self.units:
            raise RuntimeError(f"the step sketched {times['covered_units']} of {self.units} units whole")
        return sk.sigma, times

    def unit_bytes(self, g: int) -> bytes:
        return self.L.segment_rows(self.store, self.lay.spec, UNIT * g, UNIT)


class Prover:
    """The prover's side of `p2_party.prove` over an encoded store served by `Engine`."""

    def __init__(self, engine: Engine, vk: VkPaths, pool: ProcessPoolExecutor | None, log=print,
                 spot: int = 16) -> None:
        self.engine, self.vk, self.pool, self.log, self.spot = engine, vk, pool, log, spot
        self.registration = engine.registration
        self.passes: dict[tuple[int, int], tuple[np.ndarray, bytes, ReceiptTree]] = {}
        self.steps: list[dict] = []
        self.opened: list[dict] = []

    def hold(self, registration: dict, commitment: coins.LiveCommitment) -> None:
        if p2_party.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not this store's")
        self.commitment = commitment

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray) -> int:
        """A few units' σ held to `p2.Sketcher` on the unit's bytes from the store and their y to hashlib: the first,
        the last and `spot` drawn from the seed."""
        U, rows = self.registration.blocks, self.registration.rows
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        units = sorted({0, U - 1, *rng.integers(0, U, self.spot).tolist()})
        sk = p2.Sketcher(seed, rows)
        bad = 0
        for g in units:
            got = bytes(sigma[g].cpu().numpy().astype("<i4").tobytes())
            if self.engine.skip_layer is None and got != sk.sketch(self.engine.unit_bytes(g)):
                bad += 1
            if bytes(ys[g]) != sk.receipt(g, got):
                bad += 1
        if bad:
            raise RuntimeError(f"{bad} of {len(units)} spot units' σ or y are not the reference's")
        return len(units)

    def run(self, challenge: coins.LiveOpening, t: int, i: int, previous: bytes) -> bytes:
        seed = receipts.seed(coins.coin(challenge), i, previous)
        t0 = time.perf_counter()
        sigma, times = self.engine.step(seed)
        a = time.perf_counter()
        ys = receipts_of(self.engine.torch, seed, sigma)
        b = time.perf_counter()
        spotted = self._spot(seed, sigma, ys)
        del sigma
        c = time.perf_counter()
        salts = os.urandom(SALT * receipts.leaves(self.registration.blocks))
        tree = ReceiptTree(seed, self.registration.blocks, ys, salts, self.pool)
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
        blocks = sorted({UNIT * p + b for _, p in draws for b in range(UNIT)})
        paths = self.vk.paths(blocks)
        out = []
        for i, p in draws:
            ys, salts, tree = self.passes[t, i]
            c = p // PER_LEAF
            unit = self.engine.unit_bytes(p)
            out.append(p2.Opening(i, p, tuple(unit[BLOCK * b:BLOCK * (b + 1)] for b in range(UNIT)),
                                  tuple(paths[UNIT * p + b] for b in range(UNIT)),
                                  ys[PER_LEAF * c:min(PER_LEAF * (c + 1), self.registration.blocks)].tobytes(),
                                  salts[SALT * c:SALT * (c + 1)], tree.path(c)))
        for key in [k for k in self.passes if k[0] <= t]:
            del self.passes[key]
        rec = {"window": t, "openings": len(out), "open_s": round(time.perf_counter() - t0, 3)}
        self.opened.append(rec)
        self.log(json.dumps({"open": rec}))
        return out


def registration_file(reg: p2.Registration, path: Path) -> None:
    path.write_text(json.dumps(p2_party.registration_json(reg), indent=1, sort_keys=True) + "\n")


# --- the toy as an encoded store ---------------------------------------------------------------------------------------

def toy(out: Path) -> p2.Registration:
    """The toy's C* (`store-toy/`) as an encoded store under `out/store` (`setup.json`, `store.json`, `seg/`, `rows/`)
    and `out/layout.json`, with the verifier's registration of it (PoUS's setup decoding C* to the committed W) in
    `out/registration.json`: what `session --ref-decode` serves at fixture scale."""
    from p2_v1 import serve as SV
    from p2_v1 import store as ST

    s, doc, c_star = SV.toy()
    reg = SV.registration(s, doc, c_star, SV.NODE)
    params, root = s.params, out / "store"
    bps = params.blocks_per_segment
    (root / "seg").mkdir(parents=True, exist_ok=True)
    (root / "rows").mkdir(exist_ok=True)
    (out / "layout.json").write_text(json.dumps(doc["layout"]))
    lsha = hashlib.sha512((out / "layout.json").read_bytes()).hexdigest()
    salt = bytes.fromhex(doc["salt"])
    (root / "setup.json").write_text(json.dumps({"commitment": doc["commitment"], "layout_sha512": lsha,
                                                 "manifest": doc["layout"]["params"], "source": doc["source"],
                                                 "salt": doc["salt"]}))
    for k in range(params.segments):
        data = c_star[BLOCK * bps * k:BLOCK * bps * (k + 1)]
        rows = b"".join(V.row_digest(BLOCK, data[BLOCK * v:BLOCK * (v + 1)]) for v in range(bps))
        (root / "seg" / f"{k:05d}.c").write_bytes(data)
        (root / "rows" / f"{k:05d}.rows").write_bytes(rows)
        meta = {"segment": k, "sha512": hashlib.sha512(data).hexdigest(), "rows_sha512": hashlib.sha512(rows).hexdigest()}
        if bps & (bps - 1) == 0:
            meta["root"] = ST.segment_root(params, salt, k, rows).hex()
        (root / "seg" / f"{k:05d}.json").write_text(json.dumps(meta))
    leaves = [V.leaf_hash(params, salt, j, c_star[BLOCK * j:BLOCK * (j + 1)]) for j in range(params.blocks)]
    (root / "store.json").write_text(json.dumps({"commitment": doc["commitment"], "salt": doc["salt"],
                                                 "source": doc["source"], "root": V.MerkleTree(leaves).root.hex()}))
    registration_file(reg, out / "registration.json")
    return reg


# --- the session -------------------------------------------------------------------------------------------------------

def verdict_line(answer: dict, expect: str | None) -> str:
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
    parts += [f"verifier_host={answer.get('host')}", f"expect={expect}", "PASS" if ok else "FAIL"]
    return " ".join(parts)


def session(a: argparse.Namespace) -> int:
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(a.workers, mp_context=mp.get_context("fork")) if a.workers > 1 else None
    if pool is not None:
        list(pool.map(abs, range(a.workers)))  # a fork pool starts every worker at its first task: before CUDA does
    node = a.node.encode()
    reg = registration_of_store(a.store, node)
    if a.registration is not None and p2_party.registration_of(json.loads(a.registration.read_text())) != reg:
        raise SystemExit("the store's registration is not --registration's")
    with open(out / "prover.log", "a") as log_f:

        def log(m: str) -> None:
            print(m, file=log_f, flush=True)
            print(m, file=sys.stderr, flush=True)

        t0 = time.perf_counter()
        vk = VkPaths(a.store, reg.params, reg.salt, reg.root, pool)
        engine = Engine(a.store, a.layout, a.lib, reg, device=a.device, chunk=a.chunk, readers=a.readers,
                        skip_layer=a.skip_layer, ref_decode=a.ref_decode)
        prover = Prover(engine, vk, pool, log)
        setup_s = time.perf_counter() - t0
        request = p2_party.request_of(node, windows=a.windows, passes=a.passes, samples=a.samples,
                                      window_ns=int(a.window_s * 1e9), deadline_ns=int(a.deadline_s * 1e9))
        log(json.dumps({"start": {"node": a.node, "units": reg.blocks, "registration": reg.digest.hex(),
                                  "request": request, "setup_s": round(setup_s, 3), "host": socket.gethostname(),
                                  "skip_layer": a.skip_layer}}))
        rc = None
        t1 = time.perf_counter()
        try:
            if a.connect is not None:
                host, _, port = a.connect.rpartition(":")
                with (socket.create_connection((host, int(port))) as conn, conn.makefile("rb") as r,
                      conn.makefile("wb") as w):
                    answer = p2_party.prove(r, w, prover, request)
            else:
                with open(out / "verifier.err", "ab") as err:
                    proc = subprocess.Popen(a.verifier, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=err)
                    try:
                        answer = p2_party.prove(proc.stdout, proc.stdin, prover, request)
                    finally:
                        proc.stdin.close()
                        rc = proc.wait()
        except Exception as e:  # noqa: BLE001 - the prover's own failure, recorded as its answer, never the verifier's
            log(json.dumps({"prover_error": traceback.format_exc()}))
            answer = {"verdict": "prover-error", "reason": f"{type(e).__name__}: {e}"}
        wall = time.perf_counter() - t1
    if pool is not None:
        pool.shutdown()
    line = verdict_line(answer, a.expect)
    doc = {"answer": answer, "request": request, "registration": p2_party.registration_json(reg), "steps": prover.steps,
           "opened": prover.opened, "setup_s": round(setup_s, 3), "session_wall_s": round(wall, 3),
           "verifier_rc": rc, "verifier": a.verifier or f"tcp {a.connect}", "prover_host": socket.gethostname(),
           "skip_layer": a.skip_layer, "store": str(a.store), "device": a.device, "verdict_line": line}
    (out / "session.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(line, flush=True)
    return 0 if line.endswith("PASS") else 1


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    t = sub.add_parser("toy")
    t.add_argument("out", type=Path)
    r = sub.add_parser("registration")
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--node", required=True)
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("session")
    s.add_argument("--store", type=Path, required=True, help="the encoded store's directory, read only")
    s.add_argument("--layout", type=Path, required=True, help="its layout JSON (p2_v1/store.py layout)")
    s.add_argument("--lib", default="unused", help="libp2v1.so's directory")
    s.add_argument("--node", required=True)
    s.add_argument("--registration", type=Path, help="the registration the store must have (registration_json)")
    to = s.add_mutually_exclusive_group(required=True)
    to.add_argument("--verifier", help="the shell command that starts p2_party --stdio")
    to.add_argument("--connect", help="HOST:PORT where p2_party --listen waits (the gate's port, through ssh -L)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--windows", type=int, default=2)
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--samples", type=int, default=1024)
    s.add_argument("--window-s", type=float, default=1.0)
    s.add_argument("--deadline-s", type=float, default=1.0)
    s.add_argument("--skip-layer", type=int)
    s.add_argument("--expect", choices=("accept", "reject"))
    s.add_argument("--device", default="cuda")
    s.add_argument("--ref-decode", action="store_true")
    s.add_argument("--chunk", type=int, default=1 << 16)
    s.add_argument("--readers", type=int, default=6)
    s.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args(argv)
    if a.command == "toy":
        reg = toy(a.out)
        print(json.dumps({"registration": reg.digest.hex(), "units": reg.blocks, "out": str(a.out)}))
        return 0
    if a.command == "registration":
        reg = registration_of_store(a.store, a.node.encode())
        registration_file(reg, a.out)
        print(json.dumps({"registration": reg.digest.hex(), "units": reg.blocks, "root": reg.root.hex()}))
        return 0
    return session(a)


if __name__ == "__main__":
    sys.exit(main())
