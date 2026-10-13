"""The prover's GPU backend: an encoded store on one GPU, its payload tree, chained sketch-sha512 passes over it with
the device reader (`../codec_tensor/reader.cu` in `session.cu`) and the openings of what the verifier draws.

~~~text
payload    N blocks of seeded random bytes, each 128-byte word's top byte below 0x80 (so below p1024 and p512), its
           tree the driver's `audit.payload_tree` with salt_j = SHAKE256(D(store salt) ‖ key ‖ PAYLOAD_LABEL ‖ u64be j);
           built a chunk at a time and retained on the CPU, so it does not displace the encoded store from HBM
store      E = the codec over the payload in place, by kv.cu (`../word_codec/kv.cu`) at the registration's key
pass       sketch-sha512 receipts of every block by one persistent grid on tensor cores, the leaves under
           salt_c = SHAKE256(D(store salt) ‖ key ‖ seed ‖ u64be c) (`rb_leaves`) and the tree chunk by chunk on a
           second stream, so the hashing runs under the later chunks' reads (`rs_pass`), its signs derived on the device
           (`rs_signs16`); a pass keeps its packed signs (8 KiB) and its
           tree's levels from h0 up (2 · N / 16 / 2^h0 nodes), nothing per block
opening    a window's draws at once, on a stream of their own: each drawn leaf's subtree of 2^h0 leaves recomputed
           from the store under its pass's constants by a second copy of the library (`rs_subtrees`), then its
           receipts and paths gathered from those subtrees, the passes' kept levels and the payload tree in one launch
           (`rs_gather64`), the blocks in another
skip       the dropped blocks zeroed on the device, their true blocks kept on the host for openings
~~~

Gates, before anything is timed (`Device.gates`): the payload root against `audit.payload_tree` and blocks of the
store against `codec.encode` (at the store's last index too), a pass's root against `receipts.tree` over the
reference's receipts (a store of several chunks), the pipelined pass's root against the plain pass's over the whole
store (in calibration), and two sessions over a small store under the registration's key, one honest and accepted by
`audit.Session`, one with half its blocks dropped and rejected at a receipt.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import queue
import shutil
import struct
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "codec_tensor"))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth import audit, coins, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix  # noqa: E402
from verity.ml import randomness  # noqa: E402

import sketch_receipt as skr  # noqa: E402
import reader as rdr  # noqa: E402
from prover import PAYLOAD_LABEL  # noqa: E402

BLOCK = rcodec.BLOCK_BYTES
SALT_DOMAIN = b"verity/store/salt/v0\0"
PER_LEAF = rb.PER_LEAF
PACK16_WORDS = BLOCK // 64 * 32  # session.cu's SK16_PACK


def store_salt(key: bytes, label: bytes, c: int) -> bytes:
    return hashlib.shake_256(SALT_DOMAIN + key + label + struct.pack(">Q", c)).digest(rb.SALT_BYTES)


def stream_head(seed: bytes) -> bytes:
    """The head of row 0's stream of a pass's signs (`skr.sign_bits`), its last byte the row: `rs_signs16`'s input."""
    key = randomness.derive(rb._digest(seed, "a pass seed"), skr.SIGN_DOMAIN, {"tag": skr.KIND, "k": skr.ROWS, "n": BLOCK})
    return randomness._STREAM + key.value + randomness.frame(randomness.frame("row") + randomness.frame(0))


def digests(words) -> list[bytes]:
    """Digests from rows of 8 state words (an n x 8 int64 tensor or array)."""
    a = np.asarray(words.cpu().numpy() if hasattr(words, "cpu") else words).astype(np.uint64).astype(">u8")
    return [r.tobytes() for r in a.reshape(-1, 8)]


def percentiles(xs: list[int]) -> dict:
    """p50, p99, p99.9 and the max of sorted xs."""
    at = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    return {"p50": at(0.5), "p99": at(0.99), "p999": at(0.999), "max": xs[-1]}


class Lib:
    def __init__(self, path: str, torch) -> None:
        self.so, self.torch = ctypes.CDLL(path), torch
        v, u64, i, b = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p
        self.so.rb_set.argtypes = [b, b, b, b, b, b, b, b, i, b, i, u64, v]
        self.so.rb_leaves.argtypes = [v, u64, u64, v, v, v]
        self.so.rb_tree.argtypes = [v, u64, u64, v, v]
        self.so.rb_read.argtypes = [v, u64, v, i, v]
        self.so.rb_sketch_set.argtypes = [b, v]
        self.so.rb_sketch.argtypes = [v, v, u64, v, v, i, v]
        self.so.rb_sketch_hash.argtypes = [v, u64, u64, v, v]
        self.so.rs_payload_leaves.argtypes = [v, u64, u64, v, v]
        self.so.rs_gather64.argtypes = [v, u64, v, v]
        self.so.rs_pass.argtypes = [v, u64, v, v, v, v, v, v, i, u64, i, v, v]
        self.so.rs_subtrees.argtypes = [v, v, v, u64, i, v, v, v]
        self.so.rs_signs16.argtypes = [b, i, v, v]
        self.so.rs_chain_set.argtypes = [b, v]
        self.so.rs_chain_prepare.argtypes = [ctypes.c_uint, v, v, v, v]
        self.so.rs_pack16.argtypes = [v, v, v]
        self.so.rs_sketch16_receipts.argtypes = [v, u64, u64, v, v, i, v]
        self.so.rb_chain_set.argtypes = [b, i, b, b, b, b, b, b, i, i, b, i, i, b, i, b, i, i, u64, v]
        self.so.rb_chain.argtypes = [v, v]
        self.so.rb_pads.argtypes = [u64, u64, v, v]
        self.so.rb_level.argtypes = [v, v, i, u64, u64, v]

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args, self.torch.cuda.current_stream().cuda_stream)
        if rc:
            raise RuntimeError(f"{fn}: {rc}")

    def set_tree(self, receipt_prefix: bytes, row_bytes: int, last_row_bytes: int, domain_id: bytes, salt_in: bytes,
                 positions: int) -> None:
        self.call("rb_set", receipt_prefix, sha512_row_v2_prefix(8 * row_bytes), sha512_row_v2_prefix(8 * last_row_bytes),
                  rb.HM96.salt_prefix, rb.HM96.leaf_prefix(rb.KEY), rb.KEY, domain_id, rb.SCHEMA.encode(),
                  len(rb.SCHEMA), salt_in, len(salt_in), positions)

    def set_chain(self, plan: rb.Plan, key: bytes) -> None:
        """Install `rb_chain`'s fixed templates for this plan and salt key."""
        n, leaves = plan.length, rb.leaves(plan.length)
        last = n - PER_LEAF * (leaves - 1)
        m = rdr.chain_messages(plan, key)
        self.call("rb_chain_set", *m["receipt"], sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * PER_LEAF),
                  sha512_row_v2_prefix(8 * rb.DIGEST_BYTES * last), rb.HM96.salt_prefix,
                  rb.HM96.leaf_prefix(rb.KEY), rb.KEY, m["binding"][0], len(m["binding"][0]), m["binding"][1],
                  m["domain"][0], len(m["domain"][0]), m["domain"][1], rb.SCHEMA.encode(), len(rb.SCHEMA),
                  m["salt"][0], len(m["salt"][0]), m["salt"][1], n)

class Kv:
    """kv.cu's encode at the library's width, masked or not."""

    def __init__(self, path: str, torch) -> None:
        self.so, self.torch = ctypes.CDLL(path), torch
        v, q, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
        self.so.kv_set_key.argtypes = [v, i, v]
        self.so.kv_set_mask.argtypes = [v]
        self.so.kv_encode.argtypes = [v, v, q, q, i, i, v]
        self.bits, self.masked = self.so.kv_bits(), bool(self.so.kv_mask())

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{fn}: {rc}")

    def set_key(self, key: rcodec.Key) -> None:
        f = key.codec.field
        if f.n != self.bits or key.codec.masked != self.masked:
            raise ValueError(f"kv.so is built for {self.bits} bits, masked={self.masked}, not {key.codec.name}")
        s = f.word_bytes
        c = ctypes.create_string_buffer(b"".join(x.to_bytes(s, "little") for x in key.c), s * key.rounds)
        t = ctypes.create_string_buffer(key.tau.to_bytes(s, "little"), s)
        self.call("kv_set_key", ctypes.cast(c, ctypes.c_void_p), key.rounds, ctypes.cast(t, ctypes.c_void_p))
        if self.masked:
            m = ctypes.create_string_buffer(key.mask.to_bytes(s, "little"), s)
            self.call("kv_set_mask", ctypes.cast(m, ctypes.c_void_p))
        self.rounds, self.words = key.rounds, f.block_words

    def encode(self, data, j0: int, n: int) -> None:
        """Blocks j0 … j0 + n − 1, at `data`, in place."""
        self.call("kv_encode", data.data_ptr(), data.data_ptr(), n * self.words, j0 * self.words, self.rounds, 0,
                  self.torch.cuda.current_stream().cuda_stream)


def path_rows(dig, up, leaves: int, idx: np.ndarray) -> np.ndarray:
    """The device addresses of each leaf's path in `idx` (a whole tree of `leaves` leaves: level 0 in dig, the levels
    above it one after another in up), sibling by sibling up to the root: len(idx) x log2(leaves)."""
    H = leaves.bit_length() - 1
    out = np.empty((len(idx), H), dtype=np.uint64)
    if H:
        out[:, 0] = np.uint64(dig.data_ptr()) + np.uint64(64) * (idx ^ 1).astype(np.uint64)
    at, width = 0, leaves // 2
    for h in range(1, H):
        out[:, h] = np.uint64(up.data_ptr()) + np.uint64(64) * (at + ((idx >> h) ^ 1)).astype(np.uint64)
        at, width = at + width, width // 2
    return out


class OffloadedHandle:
    """A pass handle whose device-to-host copy finishes on the backend's offloader."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = None


class DeviceStore:
    """N blocks on the GPU, their payload tree on the CPU, then the encoded store and the passes over it."""

    def __init__(self, torch, lib: Lib, lib_open: Lib, kv: Kv, blocks: int, key: bytes, generator, *,
                 chunk: int = 1 << 18, kept_from: int = 6, sketch_ctas: int = 0, payload_kept_from: int = 6,
                 offload_handles: bool = False) -> None:
        nl = blocks // PER_LEAF
        if blocks % PER_LEAF or blocks % chunk:
            raise ValueError(f"{blocks} blocks: the harness takes a multiple of the pass chunk and {PER_LEAF}")
        self.torch, self.lib, self.lib_open, self.kv, self.blocks, self.key = torch, lib, lib_open, kv, blocks, key
        self.H = (nl - 1).bit_length()
        self.tree_leaves = 1 << self.H
        self.payload_h0 = max(1, min(payload_kept_from, chunk.bit_length() - 1))
        self.offload_handles = offload_handles
        self.side = torch.cuda.Stream(priority=-1)
        self.shape(chunk, kept_from, sketch_ctas)
        self.data = torch.empty(blocks * BLOCK, dtype=torch.uint8, device="cuda")
        chunk = 1 << 18
        for j in range(0, blocks, chunk):
            m = min(chunk, blocks - j)
            v = self.data[j * BLOCK:(j + m) * BLOCK]
            v.view(torch.int32).copy_(torch.randint(-2 ** 31, 2 ** 31 - 1, (m * BLOCK // 4,), dtype=torch.int32,
                                                    device="cuda", generator=generator))
            v.view(-1, 128)[:, 127] &= 0x7F
        self.plan = rb.Plan.whole(blocks)
        self.lib.set_chain(self.plan, self.key)
        self.lib_open.set_chain(self.plan, self.key)
        self.dropped: dict[int, bytes] = {}
        self.pass_gpu_ms: list[float] = []
        self.open_stats: list[dict] = []
        self.pass_host_phases: list[tuple] = []
        self.chain_host_phases: list[tuple] = []
        self.chain_previous = None
        self.root_host = torch.empty((1, 8), dtype=torch.int64, pin_memory=True)

    def shape(self, chunk: int, kept_from: int, sketch_ctas: int) -> None:
        """A pass's chunk (blocks), h0 (at most the chunk's subtree height) and the sketch's CTAs, and its ring."""
        torch, nl = self.torch, self.tree_leaves
        self.chunk = min(chunk, self.blocks)
        self.h0 = max(1, min(kept_from, (self.chunk // PER_LEAF).bit_length() - 1))
        self.sketch_ctas = sketch_ctas
        self.kept_at = [0] * (self.H + 2)
        for h in range(self.h0, self.H + 1):
            self.kept_at[h + 1] = self.kept_at[h] + (nl >> h)
        lc = self.chunk // PER_LEAF
        self.ring = None
        self.ring = (torch.empty(1 + self.blocks // self.chunk, dtype=torch.int32, device="cuda"),
                     torch.empty((self.blocks, 8), dtype=torch.int64, device="cuda"),
                     torch.empty((2 * lc, 8), dtype=torch.int64, device="cuda"),
                     torch.empty((2 * lc, 8), dtype=torch.int64, device="cuda"))

    def block(self, j: int) -> bytes:
        return self.dropped[j] if j in self.dropped else self.data[j * BLOCK:(j + 1) * BLOCK].cpu().numpy().tobytes()

    def host_blocks(self, j0: int, n: int) -> list[bytes]:
        raw = self.data[j0 * BLOCK:(j0 + n) * BLOCK].cpu().numpy().tobytes()
        return [raw[q * BLOCK:(q + 1) * BLOCK] for q in range(n)]

    def payload_tree(self, node: bytes) -> bytes:
        torch, N, width, h0 = self.torch, self.blocks, 1 << (self.blocks - 1).bit_length(), self.payload_h0
        dom = audit.payload_domain(node, N)
        self.lib.set_tree(bytes(128), BLOCK, BLOCK, dom.domain_id, SALT_DOMAIN + self.key + PAYLOAD_LABEL, N)
        H, chunk = width.bit_length() - 1, self.chunk
        hc = chunk.bit_length() - 1
        kept_at = [0] * (H + 2)
        for h in range(h0, H + 1):
            kept_at[h + 1] = kept_at[h] + (width >> h)
        kept = torch.empty((kept_at[H + 1], 8), dtype=torch.int64, device="cuda")
        low_at = [0] * (h0 + 1)
        for h in range(h0):
            low_at[h + 1] = low_at[h] + (N >> h)
        low = torch.empty((low_at[h0], 8), dtype=torch.int64)
        dig = torch.empty((chunk, 8), dtype=torch.int64, device="cuda")
        up = torch.empty((chunk, 8), dtype=torch.int64, device="cuda")

        def climb(first: int, leaves: int, payload: bool) -> None:
            if payload:
                self.lib.call("rs_payload_leaves", self.data.data_ptr() + first * BLOCK, first, leaves, dig.data_ptr())
            else:
                self.lib.call("rb_pads", first, leaves, dig.data_ptr())
            source, scratch = dig, up
            for h in range(hc):
                m, i0 = leaves >> (h + 1), first >> (h + 1)
                out = kept[kept_at[h + 1] + i0:] if h + 1 >= h0 else scratch
                self.lib.call("rb_level", source.data_ptr(), out.data_ptr(), h, i0, m)
                if out.data_ptr() == scratch.data_ptr():
                    scratch = scratch[m:]
                source = out

        for first in range(0, N, chunk):
            climb(first, chunk, True)
            low[first:first + chunk].copy_(dig)
            at = 0
            for h in range(1, h0):
                m, i0 = chunk >> h, first >> h
                low[low_at[h] + i0:low_at[h] + i0 + m].copy_(up[at:at + m])
                at += m
        for first in range(N, width, chunk):
            climb(first, chunk, False)
        source = kept[kept_at[hc]:]
        for h in range(hc, H):
            m = width >> (h + 1)
            out = kept[kept_at[h + 1]:]
            self.lib.call("rb_level", source.data_ptr(), out.data_ptr(), h, 0, m)
            source = out
        self.payload_H, self.payload_h0 = H, h0
        self.payload_low_at, self.payload_low = low_at, low.numpy()
        self.payload_kept_at, self.payload_kept = kept_at, kept.cpu().numpy()
        self.payload_root = digests(self.payload_kept[-1:])[0]
        return self.payload_root

    def encode(self, key: rcodec.Key, chunk: int = 1 << 16) -> float:
        """The store from its payload, in place; the encode's rate in bytes a second of GPU time."""
        torch = self.torch
        self.kv.set_key(key)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for j in range(0, self.blocks, chunk):
            m = min(chunk, self.blocks - j)
            self.kv.encode(self.data[j * BLOCK:], j, m)
        b.record()
        torch.cuda.synchronize()
        return self.blocks * BLOCK / (a.elapsed_time(b) / 1e3)

    def set_pass(self, lib: Lib, seed: bytes) -> None:
        dom = rb.domain(seed, self.plan)
        lib.set_tree(rb.prefix(seed), rb.DIGEST_BYTES * PER_LEAF, rb.DIGEST_BYTES * PER_LEAF, dom.domain_id,
                     SALT_DOMAIN + self.key + seed, self.blocks)
        lib.call("rb_sketch_set", skr.prefix(seed))

    def run_pass(self, seed: bytes):
        """One pass (`rs_signs16`, `rs_pass`): its root, and what its openings need kept, (seed, its packed signs,
        levels h0 … H)."""
        torch = self.torch
        t0 = time.perf_counter_ns()
        head = stream_head(seed)
        t1 = time.perf_counter_ns()
        self.set_pass(self.lib, seed)
        t2 = time.perf_counter_ns()
        pk = torch.empty(PACK16_WORDS, dtype=torch.int32, device="cuda")
        kept = torch.empty((self.kept_at[self.H + 1], 8), dtype=torch.int64, device="cuda")
        ctr, ys, dig, up = self.ring
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t3 = time.perf_counter_ns()
        a.record()
        main = torch.cuda.current_stream().cuda_stream
        rc = self.lib.so.rs_signs16(head, len(head), pk.data_ptr(), main) or self.lib.so.rs_pass(
            self.data.data_ptr(), self.blocks, pk.data_ptr(), ctr.data_ptr(), ys.data_ptr(), dig.data_ptr(),
            up.data_ptr(), kept.data_ptr(), self.h0, self.chunk, self.sketch_ctas, main, self.side.cuda_stream)
        if rc:
            raise RuntimeError(f"rs_pass: {rc}")
        b.record()
        t4 = time.perf_counter_ns()
        b.synchronize()
        t5 = time.perf_counter_ns()
        self.root_host.copy_(kept[-1:], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        root = self.root_host.numpy().astype(np.uint64).astype(">u8").tobytes()
        self.pass_gpu_ms.append(a.elapsed_time(b))
        self.pass_host_phases.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, time.perf_counter_ns() - t5))
        return root, (seed, pk, kept)

    def start_chain(self, challenge: bytes, previous: bytes | None) -> None:
        """Set one window's challenge, and optionally its first previous root, once rather than once per pass."""
        if type(challenge) is not bytes or len(challenge) != rb.CHALLENGE_BYTES:
            raise ValueError("a chained window's challenge is 32 bytes")
        self.lib.call("rs_chain_set", challenge)
        if previous is not None:
            words = np.frombuffer(rb._digest(previous, "a previous root"), dtype=">u8").astype(np.uint64).view(np.int64)
            self.chain_previous = self.torch.from_numpy(words.copy()).cuda().reshape(1, 8)
        if self.chain_previous is None:
            raise ValueError("the first chained window needs its previous root")

    def run_chain(self, i: int):
        """Pass `i` with its seed, signs, and receipt-tree constants derived on the device from `chain_previous`."""
        if self.chain_previous is None or not 0 <= i < 1 << 32:
            raise ValueError("a chained pass needs a previous root and a u32 index")
        torch = self.torch
        t0 = time.perf_counter_ns()
        seed = torch.empty(rb.DIGEST_BYTES, dtype=torch.uint8, device="cuda")
        pk = torch.empty(PACK16_WORDS, dtype=torch.int32, device="cuda")
        kept = torch.empty((self.kept_at[self.H + 1], 8), dtype=torch.int64, device="cuda")
        ctr, ys, dig, up = self.ring
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t1 = time.perf_counter_ns()
        a.record()
        main = torch.cuda.current_stream().cuda_stream
        rc = self.lib.so.rs_chain_prepare(i, self.chain_previous.data_ptr(), seed.data_ptr(), pk.data_ptr(), main)
        if not rc:
            rc = self.lib.so.rs_pass(self.data.data_ptr(), self.blocks, pk.data_ptr(), ctr.data_ptr(), ys.data_ptr(),
                                     dig.data_ptr(), up.data_ptr(), kept.data_ptr(), self.h0, self.chunk,
                                     self.sketch_ctas, main, self.side.cuda_stream)
        if rc:
            raise RuntimeError(f"chained rs_pass: {rc}")
        b.record()
        t2 = time.perf_counter_ns()
        b.synchronize()
        t3 = time.perf_counter_ns()
        self.root_host.copy_(kept[-1:], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        root = self.root_host.numpy().astype(np.uint64).astype(">u8").tobytes()
        self.chain_previous = kept[-1:].clone()
        self.pass_gpu_ms.append(a.elapsed_time(b))
        self.chain_host_phases.append((t1 - t0, t2 - t1, t3 - t2, time.perf_counter_ns() - t3))
        return root, (seed, pk, kept)

    def warm(self, passes: int) -> None:
        """What `passes` passes keep (`run_pass`'s packed signs and kept levels), allocated once and freed, so the
        allocator holds them before anything is timed."""
        torch = self.torch
        held = [(torch.empty(PACK16_WORDS, dtype=torch.int32, device="cuda"),
                 torch.empty((self.kept_at[self.H + 1], 8), dtype=torch.int64, device="cuda")) for _ in range(passes)]
        del held

    def signs_match(self, seed: bytes) -> bool:
        """The device's packed signs of a pass (`rs_signs16`) against the reference's (`skr.signs`, packed)."""
        torch = self.torch
        head = stream_head(seed)
        got, want = (torch.empty(PACK16_WORDS, dtype=torch.int32, device="cuda") for _ in range(2))
        self.lib.call("rs_signs16", head, len(head), got.data_ptr())
        signs = torch.from_numpy(skr.signs(seed)).cuda()
        self.lib.call("rs_pack16", signs.data_ptr(), want.data_ptr())
        return bool(torch.equal(got, want))

    def plain_root(self, seed: bytes, expected_kept=None) -> bytes:
        """The same pass the plain way, every receipt then every leaf then the tree, for the pipelined pass's check.

        When `expected_kept` is supplied, record agreement at every kept level as well as at the root; that makes a
        non-power-of-two tree's real/padding boundary part of the startup gate rather than only its final digest.
        """
        torch, N, nl, width = self.torch, self.blocks, self.blocks // PER_LEAF, self.tree_leaves
        signs = torch.from_numpy(skr.signs(seed)).cuda()
        self.set_pass(self.lib, seed)
        sk = torch.empty((N, skr.ROWS), dtype=torch.int32, device="cuda")
        self.lib.call("rb_sketch", self.data.data_ptr(), None, N, signs.data_ptr(), sk.data_ptr(), 0)
        ys = torch.empty((N, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rb_sketch_hash", sk.data_ptr(), 0, N, ys.data_ptr())
        del sk
        dig = torch.empty((width, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rb_leaves", ys.data_ptr(), 0, nl, dig.data_ptr(), None)
        del ys
        if width > nl:
            self.lib.call("rb_pads", nl, width - nl, dig[nl:].data_ptr())
        up = torch.empty((width - 1, 8), dtype=torch.int64, device="cuda")
        self.lib.call("rb_tree", dig.data_ptr(), width, 0, up.data_ptr())
        if expected_kept is not None:
            expected = expected_kept if expected_kept.is_cuda else expected_kept.cuda()
            at, matches = 0, {}
            for h in range(1, self.H + 1):
                n = width >> h
                if h >= self.h0:
                    got = up[at:at + n]
                    want = expected[self.kept_at[h]:self.kept_at[h] + n]
                    matches[h] = bool(torch.equal(got, want))
                at += n
            self.plain_level_matches = matches
        return digests(up[-1:].cpu())[0]

    def arm(self, dropped: set[int]) -> None:
        """Restore the blocks dropped before, then drop `dropped`: zeroed, their true blocks kept on the host."""
        for j, blk in self.dropped.items():
            self.data[j * BLOCK:(j + 1) * BLOCK].copy_(self.torch.frombuffer(bytearray(blk), dtype=self.torch.uint8))
        self.dropped = {}
        for j in sorted(dropped):
            self.dropped[j] = self.data[j * BLOCK:(j + 1) * BLOCK].cpu().numpy().tobytes()
            self.data[j * BLOCK:(j + 1) * BLOCK].zero_()
        self.torch.cuda.synchronize()

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        """For each (pass, positions) of a window, each position's (block, leaf, leaf salt, receipt path, payload salt,
        payload path). The subtrees of 2^h0 leaves the positions fall in are recomputed from the store under each
        pass's constants, by the second library (`rs_subtrees`, a few launches a pass); one gather and two device copies
        serve the whole window, and its payload paths come from the CPU-resident payload tree."""
        torch, N, lib, h0 = self.torch, self.blocks, self.lib_open, self.h0
        if items and not items[0][0][1].is_cuda:
            # Large stores keep pass handles in host memory.  Bring back only a bounded batch at a time; recursively
            # opening it releases its device copy before the next batch.
            out = []
            for at in range(0, len(items), 128):
                batch = items[at:at + 128]
                k = batch[0][0][2].shape[0]
                pks = torch.empty((len(batch), PACK16_WORDS), dtype=torch.int32, device="cuda")
                kept = torch.empty((len(batch), k, 8), dtype=torch.int64, device="cuda")
                for q, ((_, pk, kp), _) in enumerate(batch):
                    pks[q].copy_(pk)
                    kept[q].copy_(kp)
                dev = [((seed, pks[q], kept[q]), positions)
                       for q, ((seed, _pk, _kp), positions) in enumerate(batch)]
                out.extend(self.open_window(dev))
            return out
        nl, S = N // PER_LEAF, PER_LEAF << h0
        per = 2 * (1 << h0) - 2
        sub_at = [0] * (h0 + 1)
        for g in range(h0):
            sub_at[g + 1] = sub_at[g] + ((1 << h0) >> g)
        device_seeds = [seed for (seed, *_), _ in items if not isinstance(seed, bytes)]
        copied_seeds = iter(torch.stack(device_seeds).cpu().numpy()) if device_seeds else iter(())
        seeds = [seed if isinstance(seed, bytes) else bytes(next(copied_seeds)) for (seed, *_), _ in items]
        firsts = [np.unique((np.asarray(ps, dtype=np.int64) // PER_LEAF) >> h0 << h0) for _, ps in items]
        total = sum(len(f) for f in firsts)
        ys = torch.empty((max(total, 1) * S, 8), dtype=torch.int64, device="cuda")
        nodes = torch.empty((max(total, 1) * per, 8), dtype=torch.int64, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        q0, rows, flat = 0, [], []
        t0 = time.perf_counter_ns()
        cs = torch.from_numpy(np.concatenate(firsts) if firsts else np.zeros(0, dtype=np.int64)).cuda()
        for ((_seed, pk, kept), positions), first, host_seed in zip(items, firsts, seeds):
            # The batched copy above is outside the timed pass span.  Installing from those host bytes sets both the
            # receipt midstate and receipt-tree constants; rb_chain(device_seed) alone would set only the latter.
            self.set_pass(lib, host_seed)
            rc = lib.so.rs_subtrees(self.data.data_ptr(), pk.data_ptr(), cs.data_ptr() + 8 * q0, len(first), h0,
                                    ys.data_ptr() + 64 * S * q0, nodes.data_ptr() + 64 * per * q0, stream)
            if rc:
                raise RuntimeError(f"rs_subtrees: {rc}")
            p = np.asarray(positions, dtype=np.int64)
            leaf = p // PER_LEAF
            q = q0 + np.searchsorted(first, leaf >> h0 << h0)
            local = leaf - (leaf >> h0 << h0)
            addr = np.empty((len(p), PER_LEAF + self.H), dtype=np.uint64)
            ys0, nd0, kp = np.uint64(ys.data_ptr()), np.uint64(nodes.data_ptr()), np.uint64(kept.data_ptr())
            addr[:, :PER_LEAF] = ys0 + np.uint64(64) * (S * q[:, None] + PER_LEAF * local[:, None]
                                                        + np.arange(PER_LEAF)).astype(np.uint64)
            for g in range(self.H):
                sib = (leaf >> g) ^ 1
                if g < h0:
                    at = per * q + sub_at[g] + (sib - ((leaf >> h0 << h0) >> g))
                    addr[:, PER_LEAF + g] = nd0 + np.uint64(64) * at.astype(np.uint64)
                else:
                    addr[:, PER_LEAF + g] = kp + np.uint64(64) * (self.kept_at[g] + sib).astype(np.uint64)
            rows.append(addr)
            flat.extend(positions)
            q0 += len(first)
        addrs = np.concatenate(rows)
        n = addrs.size
        t1 = time.perf_counter_ns()
        dev = torch.from_numpy(addrs.reshape(-1).view(np.int64)).cuda()
        got = torch.empty((n, 8), dtype=torch.int64, device="cuda")
        lib.call("rs_gather64", dev.data_ptr(), n, got.data_ptr())
        blocks = self.data.view(-1, BLOCK)[torch.tensor(flat, dtype=torch.int64, device="cuda")]
        hr, hp = self.H, self.payload_H
        cols = PER_LEAF + hr
        # Byte order on the device and pinned copies (GIL released), so the pass thread is never held behind a
        # host-side byteswap or tobytes of a window's 100+ MB.
        raw_t = torch.empty((n, 64), dtype=torch.uint8, pin_memory=True)
        raw_t.copy_(got.view(torch.uint8).view(n, 8, 8).flip(-1).reshape(n, 64))
        braw_t = torch.empty((len(flat), BLOCK), dtype=torch.uint8, pin_memory=True)
        braw_t.copy_(blocks)
        raw, braw = memoryview(raw_t.numpy().reshape(-1)), memoryview(braw_t.numpy().reshape(-1))
        pos = np.asarray(flat, dtype=np.int64)
        payload_words = np.empty((len(pos), hp, 8), dtype=np.uint64)
        low, high = self.payload_low.view(np.uint64), self.payload_kept.view(np.uint64)
        for h in range(hp):
            sib = (pos >> h) ^ 1
            src, at = ((low, self.payload_low_at[h]) if h < self.payload_h0
                       else (high, self.payload_kept_at[h]))
            payload_words[:, h] = src[at + sib]
        payload_be = payload_words.astype(">u8")
        praw = memoryview(payload_be.view(np.uint8).reshape(-1))
        t2 = time.perf_counter_ns()
        out, q = [], 0
        for seed, (_handle, positions) in zip(seeds, items):
            mine = []
            for p in positions:
                r = 64 * cols * q
                d = [bytes(raw[r + 64 * k:r + 64 * k + 64]) for k in range(PER_LEAF, cols)]
                pr = 64 * hp * q
                pd = [bytes(praw[pr + 64 * k:pr + 64 * k + 64]) for k in range(hp)]
                c = p // PER_LEAF
                mine.append((self.dropped.get(p) or bytes(braw[BLOCK * q:BLOCK * (q + 1)]),
                             bytes(raw[r:r + 64 * PER_LEAF]), store_salt(self.key, seed, c), tuple(d),
                             store_salt(self.key, PAYLOAD_LABEL, p), tuple(pd)))
                q += 1
            out.append(mine)
        del raw, braw, praw
        self.open_stats.append({"at_ns": t0, "openings": len(flat), "subtrees": total, "subtrees_ns": t1 - t0,
                                "gather_ns": t2 - t1, "assemble_ns": time.perf_counter_ns() - t2})
        return out

    def open(self, handle, positions: list[int]) -> list[tuple]:
        return self.open_window([(handle, positions)])[0]


class Device:
    """The prover's backend (`prover.ProverSide`): the store, its gates and calibration, its passes and openings."""

    def __init__(self, torch, lib_path: str, kv_path: str, blocks: int, *, seed: int, log=print,
                 gate_blocks: int = 4096, gate_chunk: int = 1024, calibration_passes: int = 4000,
                 warm_openings: int = 2773, chunk: int = 1 << 18, kept_from: int = 6, sketch_ctas: int = 0,
                 payload_kept_from: int = 6, offload_handles: bool = False) -> None:
        self.torch, self.log, self.blocks, self.warm_openings = torch, log, blocks, warm_openings
        self.gate_blocks, self.gate_chunk, self.calibration_passes = gate_blocks, gate_chunk, calibration_passes
        self.shape = {"chunk": chunk, "kept_from": kept_from, "sketch_ctas": sketch_ctas,
                      "payload_kept_from": payload_kept_from, "offload_handles": offload_handles}
        self.lib = Lib(lib_path, torch)
        self._open_copy = tempfile.NamedTemporaryFile(suffix=".so")
        shutil.copyfile(lib_path, self._open_copy.name)
        self.lib_open = Lib(self._open_copy.name, torch)
        self.kv = Kv(kv_path, torch)
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.salt_key = self.rng.bytes(32)
        self.main_stream = torch.cuda.Stream()
        self.open_stream = torch.cuda.Stream()
        self.offload_jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._offload, daemon=True).start()
        self.gates: dict = {}
        with torch.cuda.stream(self.main_stream):
            self.store = DeviceStore(torch, self.lib, self.lib_open, self.kv, blocks, self.salt_key, self.gen,
                                     **self.shape)
        torch.cuda.synchronize()

    def payload_root(self, node: bytes) -> bytes:
        self.node = node
        with self.torch.cuda.stream(self.main_stream):
            root = self.store.payload_tree(node)
        self.torch.cuda.synchronize()
        return root

    def encode(self, registration: audit.Registration) -> dict:
        self.gates = self.gate(registration)
        self.log(f"device: gates {self.gates}")
        if not self.gates["ok"]:
            raise RuntimeError(f"a gate failed, nothing is timed: {self.gates}")
        with self.torch.cuda.stream(self.main_stream):
            rate = self.store.encode(registration.key)
        self.torch.cuda.synchronize()
        last = self.blocks - 1
        self.gates["encoded_last_block"] = self.store.block(last) == self.reference_last
        if not self.gates["encoded_last_block"]:
            raise RuntimeError("the store's last block is not the codec's encode of its payload")
        self.encode_bytes_per_s = rate
        return {"encode_bytes_per_s": rate, "blocks": self.blocks, "bytes": self.blocks * BLOCK}

    def calibrate(self) -> dict:
        """The read alone (`rb_read`, best of 6), `calibration_passes` chained passes (their host times, gaps and GPU
        times; the allowance takes a high percentile of them, `percentiles`), a pass's root against the plain pass's at
        the same seed over the whole store, and a window's openings over the calibration's passes."""
        torch, s = self.torch, self.store
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        reads = []
        with torch.cuda.stream(self.main_stream):
            for _ in range(6):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                self.lib.call("rb_read", s.data.data_ptr(), s.blocks * BLOCK, xr.data_ptr(), 16 * sms)
                b.record()
                b.synchronize()
                reads.append(a.elapsed_time(b))
            prev, pass_ns, gap_ns, kept = bytes(64), [], [], []
            challenge = self.rng.bytes(32)
            t_end = None
            # Timed as a session runs: what the passes keep allocated once beforehand (the allocator growing under a
            # timed pass stalls it for tens of ms) and no collector.
            s.warm(1 if s.offload_handles else self.calibration_passes)
            gc.collect()
            gc.freeze()
            gc.disable()
            s.start_chain(challenge, prev)
            first = None
            for i in range(self.calibration_passes):
                t0 = time.perf_counter_ns()
                want_seed = rb.seed(challenge, i, prev)
                prev, handle = self._run_chain(i)
                kept.append(handle)
                t1 = time.perf_counter_ns()
                if first is None:
                    first = (want_seed, prev, handle)
                if t_end is not None:
                    gap_ns.append(t0 - t_end)
                pass_ns.append(t1 - t0)
                t_end = t1
            gc.enable()
            gc.unfreeze()
            self.offload_jobs.join()
            seed, chained_root, chained_handle = first
            seed_dev, _chained_pk, chained_kept = self._ready(chained_handle)
            self.gates["device_seed"] = (seed_dev if isinstance(seed_dev, bytes)
                                         else bytes(seed_dev.cpu().numpy())) == seed
            plain_root = s.plain_root(seed, chained_kept)
            self.gates["plain_root"] = plain_root == chained_root
            self.gates["plain_levels"] = s.plain_level_matches
            self.gates["device_signs"] = all(s.signs_match(rb.seed(challenge, i, bytes(64))) for i in range(4))
        torch.cuda.synchronize()
        if not self.gates["device_seed"]:
            raise RuntimeError("the device's chained seed is not the driver's")
        if not self.gates["plain_root"]:
            raise RuntimeError(f"the chained pass differs from the plain pass at levels "
                               f"{[h for h, ok in s.plain_level_matches.items() if not ok]}")
        if not self.gates["device_signs"]:
            raise RuntimeError("the device's signs are not the reference's")
        # A window's openings once, over the calibration's passes, before any session: the opener's stream reaches
        # its allocations here and not in a session, where growing them stalls the passes.
        draws = [(int(self.rng.integers(len(kept))), int(self.rng.integers(s.blocks))) for _ in range(self.warm_openings)]
        o0 = time.perf_counter_ns()
        self.open_window([(kept[i], [p for k, p in draws if k == i]) for i in sorted({i for i, _ in draws})])
        open_ns = time.perf_counter_ns() - o0
        del kept
        gpu = s.pass_gpu_ms[-self.calibration_passes:]
        s.pass_gpu_ms.clear()
        read_ms = min(reads[1:])
        host = sorted(pass_ns[1:])
        gaps = sorted(gap_ns[1:])
        return {"pass_ns_max": host[-1], "pass_ns_median": host[len(host) // 2], "pass_ns": percentiles(host),
                "gap_ns_max": gaps[-1], "gap_ns_median": gaps[len(gaps) // 2], "gap_ns": percentiles(gaps),
                "passes": len(host), "pass_gpu_ms_median": sorted(gpu)[len(gpu) // 2], "read_ms": read_ms,
                "read_bytes_per_s": s.blocks * BLOCK / (read_ms / 1e3),
                "tau": sorted(gpu)[len(gpu) // 2] / read_ms, "tau_host": host[len(host) // 2] / 1e6 / read_ms,
                "warm_openings": len(draws), "warm_open_ns": open_ns, "shape": {**self.shape, "h0": s.h0},
                "device": torch.cuda.get_device_name(0)}

    def arm(self, dropped: set[int]) -> None:
        with self.torch.cuda.stream(self.main_stream):
            self.store.arm(dropped)
        self.store.pass_gpu_ms.clear()

    def run_pass(self, seed: bytes):
        with self.torch.cuda.stream(self.main_stream):
            return self.store.run_pass(seed)

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        with self.torch.cuda.stream(self.main_stream):
            self.store.start_chain(challenge, previous)

    def run_chained(self, i: int):
        with self.torch.cuda.stream(self.main_stream):
            return self._run_chain(i)

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        self.offload_jobs.join()
        items = [(self._ready(handle), positions) for handle, positions in items]
        with self.torch.cuda.stream(self.open_stream):
            return self.store.open_window(items)

    def _run_chain(self, i: int):
        root, handle = self.store.run_chain(i)
        if not self.store.offload_handles:
            return root, handle
        out = OffloadedHandle()
        self.offload_jobs.put((handle, out))
        return root, out

    def _offload(self) -> None:
        while True:
            (seed, pk, kept), out = self.offload_jobs.get()
            try:
                out.value = (bytes(seed.cpu().numpy()), pk.cpu(), kept.cpu())
            except BaseException as error:
                out.value = error
            finally:
                self.offload_jobs.task_done()

    @staticmethod
    def _ready(handle):
        if not isinstance(handle, OffloadedHandle):
            return handle
        if isinstance(handle.value, BaseException):
            raise RuntimeError("pass-handle offload failed") from handle.value
        if handle.value is None:
            raise RuntimeError("pass-handle offload is not ready")
        return handle.value

    def session_stats(self) -> dict:
        gpu = self.store.pass_gpu_ms
        return {"passes": len(gpu), "pass_gpu_ms_sum": sum(gpu), "pass_gpu_ms_median": sorted(gpu)[len(gpu) // 2]
                if gpu else None, "pass_gpu_ms_max": max(gpu) if gpu else None}

    def gate(self, registration: audit.Registration) -> dict:
        torch, key, out = self.torch, registration.key, {}
        last = self.blocks - 1
        self.reference_last = rcodec.encode(self.store.block(last), last, key)
        g = torch.Generator(device="cuda")
        g.manual_seed(int(self.rng.integers(1 << 62)))
        with torch.cuda.stream(self.main_stream):
            small = DeviceStore(torch, self.lib, self.lib_open, self.kv, self.gate_blocks, self.salt_key, g,
                                chunk=self.gate_chunk, kept_from=self.shape["kept_from"],
                                sketch_ctas=self.shape["sketch_ctas"], payload_kept_from=self.shape["payload_kept_from"])
            payloads = small.host_blocks(0, small.blocks)
            root = small.payload_tree(self.node)
            salts = [store_salt(self.salt_key, PAYLOAD_LABEL, j) for j in range(small.blocks)]
            out["payload_root"] = root == audit.payload_tree(self.node, payloads, salts).commitment.root
            small.encode(key)
            encoded = small.host_blocks(0, small.blocks)
        torch.cuda.synchronize()
        out["encode"] = all(encoded[j] == rcodec.encode(payloads[j], j, key) for j in (0, 1, small.blocks - 1))
        out["decode"] = all(rcodec.decode(encoded[j], j, key) == payloads[j] for j in range(min(256, small.blocks)))
        reg = audit.Registration(registration.node, small.blocks, root, registration.codec_seed, registration.codec,
                                 registration.receipt, registration.bind, registration.layout, registration.rows)
        seed = rb.seed(self.rng.bytes(32), 0, reg.digest)
        with torch.cuda.stream(self.main_stream):
            got, _ = small.run_pass(seed)
        r = rb.receipter(reg.receipt, seed, reg.rows)
        ys = [r(j, encoded[j]) for j in range(small.blocks)]
        ref = rb.tree(seed, small.plan, ys, [store_salt(self.salt_key, seed, c) for c in range(small.blocks // PER_LEAF)])
        out["pass_root"] = got == ref.commitment.root
        out["session_honest"] = self.gate_session(small, reg, set())
        out["session_skip_half"] = self.gate_session(small, reg, set(range(0, small.blocks, 2)))
        out["ok"] = (out["payload_root"] and out["encode"] and out["decode"] and out["pass_root"]
                     and out["session_honest"]["accepted"] and out["session_skip_half"]["failure"] == "receipt")
        # `small` and its gate sessions use the same shared libraries as the real store, so its fixed chain templates
        # replaced the real store's. Restore both copies before calibration derives the first real pass on the device.
        self.lib.set_chain(self.store.plan, self.salt_key)
        self.lib_open.set_chain(self.store.plan, self.salt_key)
        del small
        torch.cuda.empty_cache()
        return out

    def gate_session(self, small: DeviceStore, reg: audit.Registration, dropped: set[int]) -> dict:
        """Two windows of two passes each over `small` under a clock the gate turns, checked by `audit.Session`."""
        now = [0]
        params = audit.Params(reg.blocks, 1000, 2, 24, 2.0 ** -40, window_ns=1, codec=reg.codec, receipt=reg.receipt,
                              rows=reg.rows)
        session = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: now[0],
                                not_before_ns=0)
        with self.torch.cuda.stream(self.main_stream):
            small.arm(dropped)
        coin, prev, kept, verdicts = session.start(), reg.digest, {}, []
        for t in range(2):
            with self.torch.cuda.stream(self.main_stream):
                small.start_chain(coin.coin, prev if t == 0 else None)
            for i in range(2):
                with self.torch.cuda.stream(self.main_stream):
                    prev, kept[t, i] = small.run_chain(i)
                now[0] += 1
                if session.receive(t, i, prev) is not None:
                    return {"accepted": False, "failure": session.failed.failure}
            coin = session.boundary()
            now[0] += 1
        for t in range(2):
            if t == 1:
                now[0] += 10_000
            draws = session.close(t)
            if isinstance(draws, audit.Verdict):
                return {"accepted": False, "failure": draws.failure}
            with self.torch.cuda.stream(self.open_stream):
                fields = small.open_window([(kept[t, i], [p]) for i, p in draws])
            openings = [audit.Opening(i, p, f[0], f[1], f[2], f[3], f[4], f[5], b"") for (i, p), (f,) in zip(draws, fields)]
            v = session.verify(t, openings)
            verdicts.append(v)
            if not v.accepted:
                return {"accepted": False, "failure": v.failure, "at": v.at, "dropped": v.at[1] in dropped}
        with self.torch.cuda.stream(self.main_stream):
            small.arm(set())
        return {"accepted": True, "failure": None, "draws": sum(len(w.draws) for w in session.windows)}


def add_arguments(ap) -> None:
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--kv", required=True, help="kv.so at the registration's codec (p1024 masked)")
    ap.add_argument("--gib", type=float, default=16.0,
                    help="the store's size, rounded down to a whole --chunk-blocks (need not be a power of two)")
    ap.add_argument("--gate-blocks", type=int, default=4096)
    ap.add_argument("--calibration-passes", type=int, default=4000,
                    help="so the allowance's p99.9 is a few passes in, not the worst")
    ap.add_argument("--warm-openings", type=int, default=2773, help="a window's openings, opened once in calibration")
    ap.add_argument("--chunk-blocks", type=int, default=1 << 18, help="a pass's chunk (`rs_pass`)")
    ap.add_argument("--kept-from", type=int, default=6, help="h0: the pass tree's levels kept for the openings")
    ap.add_argument("--payload-kept-from", type=int, default=6,
                    help="h0: the payload tree's levels kept; lower paths are recomputed for openings")
    ap.add_argument("--sketch-ctas", type=int, default=0, help="the receipts grid's CTAs (0: one an SM)")
    ap.add_argument("--offload-handles", action="store_true",
                    help="keep pass handles on the CPU between their root and openings (automatic above 64 GiB)")


def from_arguments(a, log=print) -> Device:
    import torch

    n = int(a.gib * 2 ** 30) // BLOCK
    blocks = n // a.chunk_blocks * a.chunk_blocks
    return Device(torch, a.lib, a.kv, blocks, seed=a.seed, log=log, gate_blocks=a.gate_blocks,
                  calibration_passes=a.calibration_passes, warm_openings=a.warm_openings, chunk=a.chunk_blocks,
                  kept_from=a.kept_from, sketch_ctas=a.sketch_ctas, payload_kept_from=a.payload_kept_from,
                  offload_handles=a.offload_handles or a.gib > 64)
