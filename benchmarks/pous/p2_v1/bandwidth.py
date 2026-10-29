"""A remote bandwidth session over PoUS's P2 store on the card that serves from it: the prover's side of
`verity.core.protocols.remote_bandwidth.p2_party`, whose verifier runs on another host.

    bandwidth.py toy OUT                                  # the toy's C* as an encoded store, and its registration
    bandwidth.py registration --store ENC --node NODE --layout F --out FILE
    bandwidth.py session --store ENC --layout F --lib DIR --node NODE --out DIR
                 (--verifier CMD | --connect HOST:PORT | --party CMD --party-host HOST)
                 [--windows 2] [--passes 2] [--samples 1024] [--window-s 1] [--deadline-s 1]
                 [--skip-layer L | --cache] [--expect accept|reject] [--device cuda] [--ref-decode]
                 [--case-dir DIR [--case-exe EXE]]

**The plan.** A pass reads the units of the registration's plan (`p2_party.plan_of`), which the verifier holds and sends
with it: `tensor_plan`, the units holding a tensor's block, so not the store's padding past the last tensor (it is
PoUS's, and the decode never reads it). `registration --layout` writes it into the registration.

**The step.** Each pass of the session is one engine step's weight reads: every tensor of the store, in block order, from
its segment files onto the device (`full.Segments`, kernels' reader) and through kernels' staging decode
(`stage.decode_matrix`) into a staging buffer, as a forward step reads them. The decode reads each chunk's rows from
`stage.Store.decode_rows`, whose attached sketch (`store.sketch`, `Sketch.read`) sketches those same device rows under
the pass's seed just before they go to the decode kernel (`Sketch.add`: π of each block by an exact int8 matmul against its four offsets' signs, summed into
its unit's σ), so the receipts are of the bytes the decode read. Blocks of no tensor up to the plan's last unit (the
head's last unit's tail) are read and sketched alone, so every unit of the plan is covered. Then y = SHA512(prefix ‖
u64be g ‖ σ_g) for every unit of the plan on the device (`receipts_of`, a torch SHA-512 after the prefix's midstate; a
sample held to hashlib and to `p2.Sketcher` each pass), and the pass's receipt tree (`ReceiptTree`, `receipts.tree`'s,
in worker processes). The sketch is a pass of its own over the chunk's rows, not fused into the decode kernel, so each
byte is read from HBM twice.

**The session.** `session` starts the verifier with `--verifier CMD` (the ssh command that runs `p2_party --stdio` on
the auditor's host; on one host, the module itself, which certifies nothing), reaches `p2_party --listen` at
`--connect HOST:PORT`, or starts it as a party of the auditor's gate (`--party CMD`, the ssh command to the gate's verb,
whose stdout says the port, reached at `--party-host`, and then the verifier's own record of the answer, which must be
the one the session ended with), sends the request and answers each pass,
each window's openings (each unit's four blocks from the segment files with their PoUS vk paths, `VkPaths`, from the
store's row digests) and returns the verifier's answer, written to `OUT/session.json` with each step's timings, and
the line `BW VERDICT ...`. `--skip-layer L` is a cheat: it neither reads nor decodes layer L's tensors, and the
receipts of the units they hold are from what was read, so a draw that lands there is rejected (`receipt`). `--cache`
is another: every pass after the session's first reads nothing of the store and replays the first pass's σ from its
cache under the pass's seed, so a draw of a later pass is rejected (`receipt`).

**The case.** The verifier sends the session's core case before its answer (`p2_party`'s `{"case": ...}`, `p2_case`):
`--case-dir DIR` writes it with its answer, `case.json` and `answer.json`, into DIR (0700, each file 0600), the
answer `RemoteAudit.P2Protocol`'s with `--case-exe` (`remote-audit-p2-protocol`), else `p2_case.answer`'s. The case
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
from verity.core.protocols.remote_bandwidth import coins, p2, p2_case, p2_party, receipts
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
    `j0 ..` ([n, 2056] uint8, as stored) into their units' σ, each block's π against its offset's sign columns, and
    `read` is `stage.Store.decode_rows`'s hook to it; `seen` counts each unit's blocks sketched, four when the pass
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

def _domain(seed: bytes, runs: tuple, blocks: int) -> tuple[Any, int]:
    """The receipt tree's domain under the plan `runs` of a store of `blocks` units, and the plan's positions: a
    worker's task carries the plan's runs, as a `Plan` doesn't pickle."""
    plan = receipts.Plan(runs, blocks)
    return receipts.domain(seed, plan), plan.length


def _leaves(args: tuple) -> bytes:
    seed, runs, blocks, lo, hi, ys, salts = args
    d, n = _domain(seed, runs, blocks)
    out = bytearray()
    for c in range(lo, hi):
        a, b = (PER_LEAF * c - PER_LEAF * lo) * DIGEST, (min(PER_LEAF * (c + 1), n) - PER_LEAF * lo) * DIGEST
        out += d.leaf(c, c, receipts.SCHEMA, receipts.hidden(ys[a:b], salts[SALT * (c - lo):SALT * (c - lo + 1)]))
    return bytes(out)


def _pads(args: tuple) -> bytes:
    seed, runs, blocks, lo, hi = args
    d, _ = _domain(seed, runs, blocks)
    return b"".join(d.pad(r) for r in range(lo, hi))


def _nodes(args: tuple) -> bytes:
    seed, runs, blocks, level, lo, below = args
    d, _ = _domain(seed, runs, blocks)
    return b"".join(d.node(level, lo + k, below[2 * DIGEST * k:2 * DIGEST * k + DIGEST],
                           below[2 * DIGEST * k + DIGEST:2 * DIGEST * (k + 1)]) for k in range(len(below) // (2 * DIGEST)))


def _run(pool: ProcessPoolExecutor | None, fn, tasks: list[tuple]) -> list[bytes]:
    return list(pool.map(fn, tasks)) if pool is not None and len(tasks) > 1 else [fn(x) for x in tasks]


class ReceiptTree:
    """`receipts.tree(seed, plan, ys, salts)` built level by level in `pool`'s processes: its `root`, and `path(c)`,
    leaf c's siblings from the leaf up, as `MerkleTree.open(c).path`."""

    CHUNK = 1 << 14

    def __init__(self, seed: bytes, plan: receipts.Plan, ys: np.ndarray, salts: bytes,
                 pool: ProcessPoolExecutor | None = None) -> None:
        L, n = plan.length, receipts.leaves(plan.length)
        if ys.shape != (L, DIGEST) or len(salts) != SALT * n:
            raise ValueError(f"{L} receipts in {n} leaves")
        raw, C, p = ys.tobytes(), self.CHUNK, (seed, plan.runs, plan.blocks)
        tasks = [(*p, lo, min(n, lo + C), raw[PER_LEAF * DIGEST * lo:PER_LEAF * DIGEST * min(n, lo + C)],
                  salts[SALT * lo:SALT * min(n, lo + C)]) for lo in range(0, n, C)]
        width = 1 << merkle_depth(n)
        level = b"".join(_run(pool, _leaves, tasks))
        level += b"".join(_run(pool, _pads, [(*p, lo, min(width, lo + C)) for lo in range(n, width, C)]))
        self.levels = [level]
        depth = 0
        while len(level) > DIGEST:
            pairs = len(level) // (2 * DIGEST)
            tasks = [(*p, depth, lo, level[2 * DIGEST * lo:2 * DIGEST * min(pairs, lo + C)])
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
        self.serve(receipts.Plan.whole(registration.blocks))

    def serve(self, plan: receipts.Plan) -> None:
        """Each step from now on reads `plan`'s units: the store's blocks up to its last unit's, every tensor's."""
        self.plan = plan
        self.units = max(f + c for f, c in plan.runs)
        self.pieces = pieces(self.lay, UNIT * self.units)

    def step(self, seed: bytes) -> tuple[Any, dict]:
        """One step under `seed`: σ on the device of every unit up to the plan's last, and the step's timings."""
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
                    store = S.Store(self.lay, rows, t.first_block, self.lib, chunk=self.chunk)
                    store.sketch = sk
                    w = torch.empty(t.n, t.k, dtype=torch.int8, device=self.device)
                    S.decode_matrix(store, t.layer, t.name, w)
                    del w, store
                F.sync()
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
    """The prover's side of `p2_party.prove` over an encoded store served by `Engine`."""

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
        if p2_party.registration_of(registration) != self.registration:
            raise ValueError("the verifier's registration is not this store's")
        self.engine.serve(p2_party.plan_of(registration, self.registration))

    def end(self, commitment: coins.LiveCommitment, openings: list[coins.LiveOpening]) -> None:
        why = coins.received_reason(commitment, [self.received[t] for t in sorted(self.received)], openings)
        if why is not None:
            raise ValueError(f"the session's coins are not the ones this prover was sent: {why}")

    def _spot(self, seed: bytes, sigma: Any, ys: np.ndarray, cheat: bool) -> int:
        """A few positions' σ held to `p2.Sketcher` on their unit's bytes from the store (not on a pass that `cheat`s)
        and their y to hashlib: the plan's first, its last and `spot` drawn from the seed."""
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


def registration_file(reg: p2.Registration, path: Path, plan: receipts.Plan | None = None) -> None:
    path.write_text(json.dumps(p2_party.registration_json(reg, plan), indent=1, sort_keys=True) + "\n")


# --- the toy as an encoded store ---------------------------------------------------------------------------------------

def toy(out: Path) -> p2.Registration:
    """The toy's C* (`store-toy/`) as an encoded store under `out/store` (`setup.json`, `store.json`, `seg/`, `rows/`)
    and `out/layout.json`, with the verifier's registration of it (PoUS's setup decoding C* to the committed W) and its
    tensors' plan in `out/registration.json`: what `session --ref-decode` serves at fixture scale."""
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
    registration_file(reg, out / "registration.json", tensor_plan(doc["layout"], reg.blocks))
    return reg


# --- the session -------------------------------------------------------------------------------------------------------

def verdict_line(answer: dict, expect: str | None, record: dict | None = None) -> str:
    """The session's verdict line; with the verifier's own `record` of its answer (a party's stdout), whether the two
    agree, and the session passes only if they do."""
    v = answer.get("verdict")
    ok = (expect is None or {"accept": "accept", "reject": "reject"}[expect] == v) and record in (None, answer)
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
    if record is not None:
        parts.append(f"verifier_record_agrees={record == answer}")
    return " ".join([*parts, "PASS" if ok else "FAIL"])


def _party(a: argparse.Namespace, out: Path, prover: Prover, request: dict) -> tuple[dict, dict | None, int]:
    """A session with a party the auditor's gate starts (`--party CMD`, `p2_party --listen` on the gate's host): its
    `{"listening": port}` read from CMD's stdout, the session over TCP to `--party-host` at that port, and then its own
    record of the answer from the rest of CMD's stdout. The gate kills a party once its session's minutes run out
    (`bw_party --minutes M`) and says so on that stdout, `{"killed": why}`: a session cut off so fails naming it."""
    with open(out / "verifier.err", "ab") as err:
        proc = subprocess.Popen(a.party, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        record = killed = None

        def rest() -> None:
            nonlocal record, killed
            for line in proc.stdout:
                doc = json.loads(line)
                if isinstance(doc, dict) and "answer" in doc:
                    record = doc["answer"]
                elif isinstance(doc, dict) and "killed" in doc:
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
                    answer = p2_party.prove(r, w, prover, request)
            except (OSError, ValueError) as e:
                rest()
                if killed is not None:
                    raise ConnectionError(f"the auditor's gate killed the party: {killed}") from e
                raise
            rest()
        finally:
            rc = proc.wait()
    return answer, record, rc


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
        prover = Prover(engine, vk, pool, log, cache=a.cache)
        setup_s = time.perf_counter() - t0
        request = p2_party.request_of(node, windows=a.windows, passes=a.passes, samples=a.samples,
                                      window_ns=int(a.window_s * 1e9), deadline_ns=int(a.deadline_s * 1e9))
        log(json.dumps({"start": {"node": a.node, "units": reg.blocks, "registration": reg.digest.hex(),
                                  "request": request, "setup_s": round(setup_s, 3), "host": socket.gethostname(),
                                  "skip_layer": a.skip_layer, "cache": a.cache}}))
        rc = record = None
        t1 = time.perf_counter()
        try:
            if a.party is not None:
                answer, record, rc = _party(a, out, prover, request)
            elif a.connect is not None:
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
        case = None
        if a.case_dir is not None and prover.kept is not None and answer.get("verdict") in ("accept", "reject",
                                                                                           "undecided"):
            judged = p2_case.judged(prover.kept, answer, a.case_exe)
            p2_case.write(a.case_dir, prover.kept, judged)
            case = {"dir": str(a.case_dir), "commitment": judged["commitment"], "store_root": prover.kept["store_root"],
                    "verdict": judged["verdict"], "judge": "lean" if a.case_exe else "python"}
            log(json.dumps({"case": case}))
    if pool is not None:
        pool.shutdown()
    if a.party is not None and record is None and answer.get("verdict") != "prover-error":
        record = {"verdict": "missing", "reason": "the party's stdout held no answer"}
    line = verdict_line(answer, a.expect, record)
    verifier = (f"party {a.party} at {a.party_host}" if a.party else a.verifier or f"tcp {a.connect}")
    doc = {"answer": answer, "request": request, "registration": p2_party.registration_json(reg, engine.plan),
           "steps": prover.steps,
           "opened": prover.opened, "setup_s": round(setup_s, 3), "session_wall_s": round(wall, 3),
           "verifier_rc": rc, "verifier": verifier, "verifier_record": record, "prover_host": socket.gethostname(),
           "skip_layer": a.skip_layer, "cache": a.cache, "store": str(a.store), "device": a.device, "verdict_line": line,
           "case": case}
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
    r.add_argument("--layout", type=Path, required=True, help="the store's layout JSON, whose tensors' units are the plan")
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
    to.add_argument("--party", help="the shell command (ssh to the auditor's gate) that starts p2_party --listen there")
    s.add_argument("--party-host", help="with --party: the address its port is reached at from here")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--windows", type=int, default=2)
    s.add_argument("--passes", type=int, default=2)
    s.add_argument("--samples", type=int, default=1024)
    s.add_argument("--window-s", type=float, default=1.0)
    s.add_argument("--deadline-s", type=float, default=1.0)
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
    s.add_argument("--case-exe", help="remote-audit-p2-protocol, which answers the case (else p2_case.answer)")
    a = ap.parse_args(argv)
    if a.command == "session" and a.case_dir is not None and a.case_dir.resolve().is_relative_to(a.out.resolve()):
        raise SystemExit("--case-dir is private: it can't be under --out")
    if a.command == "toy":
        reg = toy(a.out)
        print(json.dumps({"registration": reg.digest.hex(), "units": reg.blocks, "out": str(a.out)}))
        return 0
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
