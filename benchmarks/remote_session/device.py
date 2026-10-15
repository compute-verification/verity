"""The prover's GPU backend: an encoded store across the visible GPUs, its payload tree, chained sketch-sha512 passes
over it with the device reader (`../codec_tensor/reader.cu` in `session.cu`) and the openings of what the verifier
draws.

~~~text
shards     the store's N blocks as contiguous shards of whole chunks, one a GPU, as even as the chunks allow: GPU g
           holds blocks base_g … base_g + n_g − 1, and every position, leaf and node index is the store's, so the
           tree is `receipts.tree` over the whole store in block order whatever the GPUs (one is the same code)
payload    N blocks of seeded random bytes, each 128-byte word's top byte below 0x80 (so below p1024 and p512), its
           tree the driver's `audit.payload_tree` with salt_j = SHAKE256(D(store salt) ‖ key ‖ PAYLOAD_LABEL ‖ u64be j);
           built a chunk at a time on the chunk's GPU and retained on the CPU, so it does not displace the store
store      E = the codec over the payload in place, by kv.cu (`../word_codec/kv.cu`) on each shard's GPU at the
           registration's key and the store's positions
pass       on each GPU, its seed, signs and constants derived from the previous root in its root ring
           (`rs_chain_prepare`), its shard's sketch-sha512 receipts on tensor cores with the leaves and levels up to
           each chunk's root at hc hashed under the reads (`rs_pass`), and its share of the padding up to hc; then on
           the leader (GPU 0) the levels above hc from every GPU's over P2P, the root written to every GPU's ring
           (`rs_top`). Each pass is queued on every GPU behind the leader's event for the root it chains from, while
           the pass before it runs, so the passes follow one another on the devices with no host step between them.
           A pass keeps its packed signs (8 KiB), each shard's levels h0 … hc and the levels above, nothing per block
opening    a window's draws at once, shard by shard on each GPU's opening stream: each drawn leaf's subtree of 2^h0
           leaves recomputed from the shard under its pass's constants by a second copy of the library
           (`rs_subtrees`), then its receipts and its path below hc gathered from those subtrees and the shard's kept
           levels in one launch (`rs_gather64`), the blocks in another; the path above hc comes from the pass's top
           levels and the payload paths from the CPU-resident payload tree
skip       the dropped blocks zeroed on their GPUs, their true blocks kept on the host for openings
~~~

Gates, before anything is timed (`Device.gates`), over a small store sharded across the same GPUs (three chunks a GPU,
so its tree has padding on every GPU when there are several): the payload root against `audit.payload_tree` and
blocks of the store against `codec.encode` (at each shard's first block and the store's last), a pass's root against
`receipts.tree` over the reference's receipts, and two sessions under the registration's key, one honest and accepted
by `audit.Session`, one with 1/16 of its blocks dropped and rejected at a receipt; then in calibration, the chained
pass against the plain pass at every kept level over the whole store, and its seed and signs against the reference's.
"""
from __future__ import annotations

import bisect
import contextlib
import ctypes
import gc
import hashlib
import queue
import resource
import shutil
import struct
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
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
SLOTS = 8
"""A GPU's root ring: the root of the k-th pass launched goes to slot k mod SLOTS, the window's previous root from the
host to slot SLOTS. A slot is read at most three launches after its write."""
OPEN_BATCH = 128
"""Passes whose kept levels are brought back to a GPU at once to open a window."""


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


def u64s(values) -> ctypes.Array:
    return (ctypes.c_uint64 * len(values))(*values)


ALLOCATOR_STATS = ("num_alloc_retries", "num_ooms", "num_sync_all_streams", "num_device_alloc", "num_device_free",
                   "segment.all.allocated", "segment.all.freed", "reserved_bytes.all.allocated",
                   "reserved_bytes.all.freed", "allocation.all.allocated")
"""The caching allocator's counters a calibration reports the change in, per GPU (`torch.cuda.memory_stats`)."""
STALL = 1.1
"""A calibration pass is slow above this times the median interval."""


def allocator_stats(torch, index: int) -> dict:
    s = torch.cuda.memory_stats(index)
    return {k: s.get(k, 0) for k in ALLOCATOR_STATS}


def allocator_history(torch, start: bool):
    """Start the caching allocator's event history on every GPU (no stacks), or stop it and return its segment
    events, OOMs and allocation counts: (device, action, size, time_us) for every event but a plain allocation or
    free, and per device the (time_us) of each allocation."""
    mem = torch.cuda.memory
    if start:
        try:
            mem._record_memory_history(enabled="all", context=None, stacks="python", max_entries=1 << 20)
        except (TypeError, RuntimeError):
            return False
        return True
    snap = mem._snapshot()
    mem._record_memory_history(enabled=None)
    events, allocs = [], {}
    for d, entries in enumerate(snap.get("device_traces", [])):
        for e in entries:
            a = e.get("action")
            if a == "alloc":
                allocs.setdefault(d, []).append(e.get("time_us"))
            elif a not in ("free_requested", "free_completed"):
                events.append((d, a, e.get("size"), e.get("time_us")))
    return {"events": events, "allocs": allocs}


def stall_report(observed: list, trace: list, history, clock0: tuple, stats: list) -> dict:
    """Each calibration pass slower than STALL times the median interval, with what explains it: its launch's host
    phases (each GPU's allocations and submission, the top's), the host's wait for its root, every GPU's own part
    (its first to its last event), its leader's span, the launching thread's CPU time, faults and context switches
    over its launch, and the allocator events between its launch and its root. Times in ms from the pass's
    launch."""
    wall0, perf0 = clock0
    iv = [b - a for a, b in zip(observed, observed[1:])]
    med = sorted(iv)[len(iv) // 2] if iv else 0
    slow = [i + 1 for i, x in enumerate(iv) if x > STALL * med]
    ev = history["events"] if history else []
    allocs = history["allocs"] if history else {}
    rows = []
    for i in slow[:64]:
        t = trace[i]
        la = t["launch"]
        if la is None:
            continue
        t0, n = la[0], len(la)
        G = (n - 13) // 2
        ms = lambda x: round((x - t0) / 1e6, 3)
        lo_us, hi_us = (wall0 + (t0 - perf0)) / 1e3, (wall0 + (observed[i] - perf0)) / 1e3
        rows.append({"pass": i, "interval_ms": iv[i - 1] / 1e6, "gpu_ms": round(t["gpu_ms"], 3),
                     "part_ms": [round(x, 3) for x in t["part_ms"]],
                     "alloc_done_ms": [ms(la[6 + 2 * g]) for g in range(G)],
                     "submitted_ms": [ms(la[7 + 2 * g]) for g in range(G)], "top_alloc_ms": ms(la[6 + 2 * G]),
                     "launch_ms": ms(la[7 + 2 * G]), "launch_cpu_ms": (la[8 + 2 * G] - la[1]) / 1e6,
                     "faults": [b - a for a, b in zip(la[2:6], la[9 + 2 * G:13 + 2 * G])],
                     "wait_ms": [ms(x) for x in t["wait"]], "root_ms": ms(observed[i]),
                     "allocator": [(d, a, sz, round(tu / 1e3 - lo_us / 1e3, 3)) for d, a, sz, tu in ev
                                   if tu is not None and lo_us <= tu <= hi_us],
                     "allocs_in_window": sum(lo_us <= tu <= hi_us for xs in allocs.values() for tu in xs if tu)})
    over = sorted(iv)
    return {"passes": len(iv), "median_ns": med, "max_ns": over[-1] if over else 0, "slow": len(slow),
            "slow_passes": slow[:256], "max_over_median": over[-1] / med if med else None, "stalls": rows,
            "allocator_stats": stats, "allocator_events": len(ev),
            "allocator_event_kinds": sorted({a for _, a, _, _ in ev}),
            "intervals_ns": iv}


def _faults() -> tuple:
    """This thread's page faults and context switches so far: minor, major, voluntary, involuntary."""
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return r.ru_minflt, r.ru_majflt, r.ru_nvcsw, r.ru_nivcsw


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
        self.so.rs_pass.argtypes = [v, u64, u64, v, v, v, v, v, v, i, u64, i, u64, u64, v, v, v, v]
        self.so.rs_top.argtypes = [v, v, i, v, i, i, v, i, v]
        self.so.rs_enable_peers.argtypes = []
        self.so.rs_subtrees.argtypes = [v, u64, v, v, u64, i, v, v, v]
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
        """The key on the current GPU."""
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


def climb(lib: Lib, level, first: int, nodes: int, up, h: int = 0) -> list:
    """Every level from `nodes` digests at level h (`level`, a power of two of them, the first at index `first` of its
    level) up to their root, each above level h hashed into `up` after the one below (`rb_level`), `level` first."""
    out, at = [level[:nodes]], 0
    while nodes > 1:
        nodes, first = nodes // 2, first // 2
        nxt = up[at:at + nodes]
        lib.call("rb_level", out[-1].data_ptr(), nxt.data_ptr(), h, first, nodes)
        out.append(nxt)
        at, h = at + nodes, h + 1
    return out


class Gpu:
    """A visible GPU and its streams. Without a main stream of its own (`own` false), a store's work on it goes to the
    caller's current stream, as a one-GPU benchmark's does."""

    def __init__(self, torch, index: int, own: bool = True) -> None:
        self.index = index
        with torch.cuda.device(index):
            self.main = torch.cuda.Stream() if own else None
            self.open = torch.cuda.Stream() if own else None
            self.side = torch.cuda.Stream(priority=-1)
            self.offload = torch.cuda.Stream()
            self.copy = torch.cuda.Stream()
        self.sms = torch.cuda.get_device_properties(index).multi_processor_count


@dataclass(slots=True)
class Ticket:
    """A queued pass: its slot in the root rings, and what its openings keep (the leader's seed and packed signs, each
    shard's levels h0 … hc and the levels above hc)."""

    slot: int
    seed: object
    pk: object
    kept: list
    top: object

    @property
    def handle(self) -> tuple:
        return self.seed, self.pk, self.kept, self.top


class OffloadedHandle:
    """A pass handle whose device-to-host copy finishes on the backend's offloader."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = None


class DeviceStore:
    """N blocks in shards across `gpus` (the current GPU if none), their payload tree on the CPU, then the encoded store
    and the passes over it. `generator` seeds each shard's blocks: an int (GPU g's from seed + g), or for one GPU a
    torch generator on it."""

    def __init__(self, torch, lib: Lib, lib_open: Lib, kv: Kv, blocks: int, key: bytes, generator, *, gpus=None,
                 chunk: int = 1 << 18, kept_from: int = 6, sketch_ctas: int = 128, payload_kept_from: int = 6,
                 offload_handles: bool = False) -> None:
        gpus = gpus or [Gpu(torch, torch.cuda.current_device(), own=False)]
        G = len(gpus)
        if blocks % PER_LEAF or blocks % chunk or blocks // chunk < G or G > 16:
            raise ValueError(f"{blocks} blocks on {G} GPUs: the harness takes a multiple of the pass chunk and {PER_LEAF},"
                             f" at least a chunk a GPU and at most 16 GPUs")
        self.torch, self.lib, self.lib_open, self.kv, self.blocks, self.key = torch, lib, lib_open, kv, blocks, key
        self.gpus = gpus
        chunks = blocks // chunk
        per = [chunks // G + (g < chunks % G) for g in range(G)]
        self.base = [sum(per[:g]) * chunk for g in range(G)]
        self.n = [p * chunk for p in per]
        self.H = (blocks // PER_LEAF - 1).bit_length()
        self.tree_leaves = 1 << self.H
        self.payload_h0 = max(1, min(payload_kept_from, chunk.bit_length() - 1))
        self.offload_handles = offload_handles
        self.data_g = []
        for g, gpu in enumerate(gpus):
            dev = torch.device("cuda", gpu.index)
            if isinstance(generator, int):
                gen = torch.Generator(device=dev)
                gen.manual_seed(generator + g)
            elif G == 1:
                gen = generator
            else:
                raise ValueError("a store on several GPUs takes an int seed")
            with self.on(gpu):
                d = torch.empty(self.n[g] * BLOCK, dtype=torch.uint8, device=dev)
                for j in range(0, self.n[g], 1 << 18):
                    m = min(1 << 18, self.n[g] - j)
                    v = d[j * BLOCK:(j + m) * BLOCK]
                    v.view(torch.int32).copy_(torch.randint(-2 ** 31, 2 ** 31 - 1, (m * BLOCK // 4,), dtype=torch.int32,
                                                            device=dev, generator=gen))
                    v.view(-1, 128)[:, 127] &= 0x7F
            self.data_g.append(d)
        self.plan = rb.Plan.whole(blocks)
        for gpu in gpus:
            with self.on(gpu):
                self.lib.set_chain(self.plan, self.key)
                self.lib_open.set_chain(self.plan, self.key)
        self.drops = [(np.zeros(0, dtype=np.int64), torch.empty((0, BLOCK), dtype=torch.uint8)) for _ in gpus]
        self.pass_gpu_ms: list[float] = []
        self.open_stats: list[dict] = []
        self.pass_host_phases: list[tuple] = []
        self.chain_host_phases: list[tuple] = []
        self.launched, self.last = 0, None
        with torch.cuda.device(gpus[0].index):
            self.root_host = torch.empty((SLOTS, 8), dtype=torch.int64, pin_memory=True)
        self.started = [torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)]
        self.top_done = [torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)]
        self.root_ready = [torch.cuda.Event() for _ in range(SLOTS)]
        self.began = [[torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)] for _ in gpus]
        self.ended = [[torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)] for _ in gpus]
        self.trace: list | None = None
        self._launch_trace: dict = {}
        self.ring = None
        self.shape(chunk, kept_from, sketch_ctas)

    @property
    def data(self):
        """The store on its one GPU (a one-GPU benchmark's)."""
        if len(self.data_g) != 1:
            raise AttributeError("a sharded store has no one buffer")
        return self.data_g[0]

    @contextlib.contextmanager
    def on(self, gpu: Gpu, stream=None):
        """GPU `gpu` current, with `stream` (default its main stream, or the caller's if it has none): yields it."""
        s = stream if stream is not None else gpu.main
        if s is None:
            with self.torch.cuda.device(gpu.index):
                yield self.torch.cuda.current_stream()
        else:
            with self.torch.cuda.stream(s):
                yield s

    def sync(self) -> None:
        for gpu in self.gpus:
            self.torch.cuda.synchronize(gpu.index)

    def shape(self, chunk: int, kept_from: int, sketch_ctas: int) -> None:
        """A pass's chunk (blocks), h0 (at most the chunk's subtree height hc) and the sketch's CTAs, the shards' and
        the top's level offsets, each GPU's share of the padding at hc, and each GPU's ring."""
        torch = self.torch
        chunk = min(chunk, min(self.n))
        lc = chunk // PER_LEAF
        if any(b % chunk or n % chunk for b, n in zip(self.base, self.n)) or lc & (lc - 1):
            raise ValueError(f"a chunk of {chunk} blocks: every shard is whole chunks of a power of two of leaves")
        self.chunk, self.hc = chunk, lc.bit_length() - 1
        self.h0 = max(1, min(kept_from, self.hc))
        self.sketch_ctas = sketch_ctas
        hc, h0, H, G = self.hc, self.h0, self.H, len(self.gpus)
        self.kept_at = []
        for n in self.n:
            at = {h0: 0}
            for h in range(h0, hc + 1):
                at[h + 1] = at[h] + ((n // PER_LEAF) >> h)
            self.kept_at.append(at)
        self.top_at = {hc: 0}
        for h in range(hc, H + 1):
            self.top_at[h + 1] = self.top_at[h] + (self.tree_leaves >> h)
        real = self.blocks // chunk
        pads = (self.tree_leaves >> hc) - real
        counts = [pads // G + (g < pads % G) for g in range(G)]
        self.pads = [(real + sum(counts[:g]), counts[g]) for g in range(G)]
        ready_blocks = min(chunk, 1 << 13)
        self.ring = None
        ring = []
        for g, gpu in enumerate(self.gpus):
            dev, n, pad_n = torch.device("cuda", gpu.index), self.n[g], self.pads[g][1] * lc
            with self.on(gpu):
                ring.append({"ctr": torch.empty(3 + n // ready_blocks + n // chunk, dtype=torch.int32, device=dev),
                             "ys": torch.empty((n, 8), dtype=torch.int64, device=dev),
                             "dig": torch.empty((2 * lc, 8), dtype=torch.int64, device=dev),
                             "up": torch.empty((2 * lc, 8), dtype=torch.int64, device=dev),
                             "pad": torch.empty((max(2 * pad_n, 1), 8), dtype=torch.int64, device=dev),
                             "pad_out": torch.empty((max(self.pads[g][1], 1), 8), dtype=torch.int64, device=dev),
                             "seed": torch.empty(rb.DIGEST_BYTES, dtype=torch.uint8, device=dev),
                             "pk": torch.empty(PACK16_WORDS, dtype=torch.int32, device=dev),
                             "roots": torch.zeros((SLOTS + 1, 8), dtype=torch.int64, device=dev)})
        self.ring = ring
        self.sync()

    def shard_of(self, j: int) -> int:
        return bisect.bisect_right(self.base, j) - 1

    def block(self, j: int) -> bytes:
        g = self.shard_of(j)
        local = j - self.base[g]
        sel, true = self.drops[g]
        k = int(np.searchsorted(sel, local))
        if k < len(sel) and sel[k] == local:
            return true[k].numpy().tobytes()
        return self.data_g[g][local * BLOCK:(local + 1) * BLOCK].cpu().numpy().tobytes()

    def host_blocks(self, j0: int, n: int) -> list[bytes]:
        raw = []
        for g in range(len(self.gpus)):
            a, b = max(j0, self.base[g]), min(j0 + n, self.base[g] + self.n[g])
            if a < b:
                raw.append(self.data_g[g][(a - self.base[g]) * BLOCK:(b - self.base[g]) * BLOCK].cpu().numpy().tobytes())
        raw = b"".join(raw)
        return [raw[q * BLOCK:(q + 1) * BLOCK] for q in range(n)]

    def payload_tree(self, node: bytes) -> bytes:
        """The payload tree, each chunk's leaves and levels on its shard's GPU, the padding and the levels above a
        chunk's on the leader, all kept on the CPU (levels below payload_h0 for the store's leaves only)."""
        torch, N, chunk, h0 = self.torch, self.blocks, self.chunk, self.payload_h0
        width = 1 << (N - 1).bit_length()
        H, hc = width.bit_length() - 1, chunk.bit_length() - 1
        dom = audit.payload_domain(node, N)
        kept_at, low_at = {h0: 0}, {0: 0}
        for h in range(h0, H + 1):
            kept_at[h + 1] = kept_at[h] + (width >> h)
        for h in range(h0):
            low_at[h + 1] = low_at[h] + (N >> h)
        kept = np.zeros((kept_at[H + 1], 8), dtype=np.uint64)
        low = np.empty((low_at[h0], 8), dtype=np.uint64)

        def keep(levels: list, first: int, h: int = 0) -> None:
            for k, rows in enumerate(levels):
                at, m = first >> k, rows.shape[0]
                if h + k >= h0:
                    kept[kept_at[h + k] + at:kept_at[h + k] + at + m] = rows.cpu().numpy().view(np.uint64)
                elif first < N:
                    low[low_at[h + k] + at:low_at[h + k] + at + m] = rows.cpu().numpy().view(np.uint64)

        for g, gpu in enumerate(self.gpus):
            with self.on(gpu):
                dev = self.data_g[g].device
                self.lib.set_tree(bytes(128), BLOCK, BLOCK, dom.domain_id, SALT_DOMAIN + self.key + PAYLOAD_LABEL, N)
                dig = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                up = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                for first in range(self.base[g], self.base[g] + self.n[g], chunk):
                    self.lib.call("rs_payload_leaves", self.data_g[g].data_ptr() + (first - self.base[g]) * BLOCK, first,
                                  chunk, dig.data_ptr())
                    keep(climb(self.lib, dig, first, chunk, up), first)
                if g == 0:
                    for first in range(N, width, chunk):
                        self.lib.call("rb_pads", first, chunk, dig.data_ptr())
                        keep(climb(self.lib, dig, first, chunk, up), first)
        with self.on(self.gpus[0]):
            top = torch.from_numpy(kept[kept_at[hc]:kept_at[hc] + (width >> hc)].view(np.int64).copy()).to(
                self.data_g[0].device)
            keep(climb(self.lib, top, 0, width >> hc, torch.empty_like(top), hc)[1:], 0, hc + 1)
        self.payload_H, self.payload_h0 = H, h0
        self.payload_low_at, self.payload_low = low_at, low
        self.payload_kept_at, self.payload_kept = kept_at, kept
        self.payload_root = digests(kept[-1:])[0]
        return self.payload_root

    def encode(self, key: rcodec.Key, chunk: int = 1 << 16) -> float:
        """The store from its payload, in place, every shard at once; the encode's rate in bytes a second."""
        self.sync()
        t0 = time.perf_counter_ns()
        for g, gpu in enumerate(self.gpus):
            with self.on(gpu):
                self.kv.set_key(key)
                for j in range(0, self.n[g], chunk):
                    self.kv.encode(self.data_g[g][j * BLOCK:], self.base[g] + j, min(chunk, self.n[g] - j))
        self.sync()
        return self.blocks * BLOCK / ((time.perf_counter_ns() - t0) / 1e9)

    def set_pass(self, lib: Lib, seed: bytes) -> None:
        """A pass's constants from the host on the current GPU."""
        dom = rb.domain(seed, self.plan)
        lib.set_tree(rb.prefix(seed), rb.DIGEST_BYTES * PER_LEAF, rb.DIGEST_BYTES * PER_LEAF, dom.domain_id,
                     SALT_DOMAIN + self.key + seed, self.blocks)
        lib.call("rb_sketch_set", skr.prefix(seed))

    def launch(self, i: int, after: Ticket | None, seed: bytes | None = None) -> Ticket:
        """Queue pass i chained from `after`'s root (None: the window's previous root, `start_chain`) on every GPU,
        each behind the leader's event for that root, or with `seed` an unchained pass under that seed's constants
        from the host; its ticket at once (`wait`)."""
        torch, so, G, hc = self.torch, self.lib.so, len(self.gpus), self.hc
        tr = None if self.trace is None else [time.perf_counter_ns(), time.thread_time_ns(), *_faults()]
        slot = self.launched % SLOTS
        self.launched += 1
        prev = SLOTS if after is None else after.slot
        head = stream_head(seed) if seed is not None else None
        lc = self.chunk // PER_LEAF
        kept, lead = [], None
        for g, gpu in enumerate(self.gpus):
            ring, dev = self.ring[g], self.data_g[g].device
            with self.on(gpu) as s:
                if after is not None:
                    s.wait_event(self.top_done[after.slot])
                if g == 0:
                    self.started[slot].record(s)
                self.began[g][slot].record(s)
                sd = torch.empty(rb.DIGEST_BYTES, dtype=torch.uint8, device=dev) if g == 0 else ring["seed"]
                pk = torch.empty(PACK16_WORDS, dtype=torch.int32, device=dev) if g == 0 else ring["pk"]
                kp = torch.empty((self.kept_at[g][hc + 1], 8), dtype=torch.int64, device=dev)
                if tr is not None:
                    tr.append(time.perf_counter_ns())
                if seed is None:
                    rc = so.rs_chain_prepare(i, ring["roots"].data_ptr() + 64 * prev, sd.data_ptr(), pk.data_ptr(),
                                             s.cuda_stream)
                else:
                    self.set_pass(self.lib, seed)
                    rc = so.rs_signs16(head, len(head), pk.data_ptr(), s.cuda_stream)
                first, count = self.pads[g]
                rc = rc or so.rs_pass(self.data_g[g].data_ptr(), self.n[g], self.base[g], pk.data_ptr(),
                                      ring["ctr"].data_ptr(), ring["ys"].data_ptr(), ring["dig"].data_ptr(),
                                      ring["up"].data_ptr(), kp.data_ptr(), self.h0, self.chunk, self.sketch_ctas,
                                      first * lc, count * lc, ring["pad"].data_ptr(), ring["pad_out"].data_ptr(),
                                      s.cuda_stream, gpu.side.cuda_stream)
                if rc:
                    raise RuntimeError(f"rs_pass on GPU {gpu.index}: {rc}")
                self.ended[g][slot].record(s)
                if tr is not None:
                    tr.append(time.perf_counter_ns())
                kept.append(kp)
                if g == 0:
                    lead = sd, pk
        leader = self.gpus[0]
        segs = [kp.data_ptr() + 64 * self.kept_at[g][hc] for g, kp in enumerate(kept)]
        counts = [n // self.chunk for n in self.n]
        for g, (_, count) in enumerate(self.pads):
            if count:
                segs.append(self.ring[g]["pad_out"].data_ptr())
                counts.append(count)
        roots = [r["roots"].data_ptr() + 64 * slot for r in self.ring]
        with self.on(leader) as s:
            for g in range(1, G):
                s.wait_event(self.ended[g][slot])
            top = torch.empty((self.top_at[self.H + 1], 8), dtype=torch.int64, device=self.data_g[0].device)
            if tr is not None:
                tr.append(time.perf_counter_ns())
            rc = so.rs_top(u64s(segs), u64s(counts), len(segs), top.data_ptr(), hc, self.H, u64s(roots), G,
                           s.cuda_stream)
            if rc:
                raise RuntimeError(f"rs_top: {rc}")
            self.top_done[slot].record(s)
        with self.on(leader, leader.copy) as c:
            c.wait_event(self.top_done[slot])
            self.root_host[slot].copy_(self.ring[0]["roots"][slot], non_blocking=True)
            self.root_ready[slot].record(c)
        if tr is not None:
            self._launch_trace[slot] = tr + [time.perf_counter_ns(), time.thread_time_ns(), *_faults()]
        return Ticket(slot, lead[0] if seed is None else seed, lead[1], kept, top)

    def wait(self, ticket: Ticket):
        """A queued pass's root once the host has it, and its handle."""
        w0 = time.perf_counter_ns()
        self.root_ready[ticket.slot].synchronize()
        w1 = time.perf_counter_ns()
        root = self.root_host[ticket.slot].numpy().astype(np.uint64).astype(">u8").tobytes()
        k = ticket.slot
        self.pass_gpu_ms.append(self.started[k].elapsed_time(self.top_done[k]))
        if self.trace is not None:
            self.trace.append({"launch": self._launch_trace.pop(k, None), "wait": [w0, w1], "gpu_ms": self.pass_gpu_ms[-1],
                               "part_ms": [self.began[g][k].elapsed_time(self.ended[g][k]) for g in range(len(self.gpus))]})
        return root, ticket.handle

    def run_pass(self, seed: bytes):
        """One unchained pass under `seed`: its root, and what its openings need, (seed, packed signs, each shard's
        kept levels, the top levels)."""
        t0 = time.perf_counter_ns()
        ticket = self.launch(0, None, seed=seed)
        t1 = time.perf_counter_ns()
        root, handle = self.wait(ticket)
        t2 = time.perf_counter_ns()
        self.pass_host_phases.append((0, 0, 0, t1 - t0, t2 - t1, time.perf_counter_ns() - t2))
        return root, handle

    def start_chain(self, challenge: bytes, previous: bytes | None) -> None:
        """Set one window's challenge on every GPU, and, if given, its first previous root (each GPU's slot SLOTS)."""
        if type(challenge) is not bytes or len(challenge) != rb.CHALLENGE_BYTES:
            raise ValueError("a chained window's challenge is 32 bytes")
        words = None
        if previous is not None:
            words = self.torch.from_numpy(
                np.frombuffer(rb._digest(previous, "a previous root"), dtype=">u8").astype(np.uint64).view(np.int64).copy())
        for g, gpu in enumerate(self.gpus):
            with self.on(gpu):
                self.lib.call("rs_chain_set", challenge)
                if words is not None:
                    self.ring[g]["roots"][SLOTS].copy_(words)
        if previous is not None:
            self.last = None

    def run_chain(self, i: int):
        """Pass i chained from the last one `run_chain` ran (or the window's previous root), waited for."""
        t0 = time.perf_counter_ns()
        ticket = self.launch(i, self.last)
        t1 = time.perf_counter_ns()
        root, handle = self.wait(ticket)
        t2 = time.perf_counter_ns()
        self.last = ticket
        self.chain_host_phases.append((0, t1 - t0, t2 - t1, time.perf_counter_ns() - t2))
        return root, handle

    def chain(self, passes: int | None = None, until_ns: int | None = None):
        """Chained passes 0, 1, … from the window's previous root, as a session runs them: each queued before the one
        ahead of it is awaited. Yields (i, root, handle, observed) as each root reaches the host (`perf_counter_ns`),
        until `passes` or, once the clock reaches `until_ns`, the pass in flight."""
        inflight = deque([(0, self.launch(0, None))])
        nxt = 1
        while inflight:
            if (len(inflight) < 2 and (passes is None or nxt < passes)
                    and (until_ns is None or time.perf_counter_ns() < until_ns)):
                inflight.append((nxt, self.launch(nxt, inflight[-1][1])))
                nxt += 1
            i, ticket = inflight.popleft()
            root, handle = self.wait(ticket)
            yield i, root, handle, time.perf_counter_ns()

    def warm(self, passes: int) -> None:
        """What `passes` passes keep, allocated once on every GPU and freed, so the allocator holds them before
        anything is timed."""
        torch = self.torch
        held = []
        for g, gpu in enumerate(self.gpus):
            with self.on(gpu):
                held += [torch.empty((self.kept_at[g][self.hc + 1], 8), dtype=torch.int64, device=self.data_g[g].device)
                         for _ in range(passes)]
        del held

    def warmup(self, ns: int) -> dict:
        """Chained passes on a throwaway challenge for at least `ns` nanoseconds, keeping nothing, to bring the GPUs to
        their boost clocks and warm their caches and the allocator before a session's first window. They read the
        whole store and build the whole receipt tree, as a session's do, but none is sent, kept or opened: they touch
        only this store's chain constants, and the real `start_chain` that follows resets the window's challenge and
        previous root. So no window accounts for them."""
        if ns <= 0:
            return {"passes": 0, "ns": 0}
        self.start_chain(bytes(rb.CHALLENGE_BYTES), bytes(rb.DIGEST_BYTES))
        t0 = time.perf_counter_ns()
        passes = sum(1 for _ in self.chain(until_ns=t0 + ns))
        self.sync()
        self.last = None
        self.pass_gpu_ms.clear()
        return {"passes": passes, "ns": time.perf_counter_ns() - t0}

    def signs_match(self, seed: bytes) -> bool:
        """The leader's packed signs of a pass (`rs_signs16`) against the reference's (`skr.signs`, packed)."""
        torch = self.torch
        head = stream_head(seed)
        with self.on(self.gpus[0]):
            dev = self.data_g[0].device
            got, want = (torch.empty(PACK16_WORDS, dtype=torch.int32, device=dev) for _ in range(2))
            self.lib.call("rs_signs16", head, len(head), got.data_ptr())
            signs = torch.from_numpy(skr.signs(seed)).to(dev)
            self.lib.call("rs_pack16", signs.data_ptr(), want.data_ptr())
            return bool(torch.equal(got, want))

    def plain_root(self, seed: bytes, expected=None) -> bytes:
        """The same pass the plain way, chunk by chunk on each shard's GPU (every receipt, then every leaf, then the
        levels), the padding and the levels above a chunk's on the leader, for the pipelined pass's check.

        With `expected` (a pass handle), record agreement at every kept level as well as at the root
        (`plain_level_matches`): each shard's levels h0 … hc and the levels above, so a tree's real/padding boundary
        and its shard boundaries are part of the gate rather than only its final digest."""
        torch, chunk, hc, h0 = self.torch, self.chunk, self.hc, self.h0
        lc = chunk // PER_LEAF
        matches = {h: True for h in range(h0, self.H + 1)}
        level = []
        for g, gpu in enumerate(self.gpus):
            with self.on(gpu):
                dev = self.data_g[g].device
                self.set_pass(self.lib, seed)
                signs = torch.from_numpy(skr.signs(seed)).to(dev)
                sk = torch.empty((chunk, skr.ROWS), dtype=torch.int32, device=dev)
                ys = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                dig = torch.empty((lc, 8), dtype=torch.int64, device=dev)
                up = torch.empty((lc, 8), dtype=torch.int64, device=dev)
                want = None if expected is None else expected[2][g].to(dev)
                for k in range(self.n[g] // chunk):
                    j0 = self.base[g] + k * chunk
                    self.lib.call("rb_sketch", self.data_g[g].data_ptr() + k * chunk * BLOCK, None, chunk,
                                  signs.data_ptr(), sk.data_ptr(), 0)
                    self.lib.call("rb_sketch_hash", sk.data_ptr(), j0, chunk, ys.data_ptr())
                    self.lib.call("rb_leaves", ys.data_ptr(), j0 // PER_LEAF, lc, dig.data_ptr(), None)
                    levels = climb(self.lib, dig, j0 // PER_LEAF, lc, up)
                    if want is not None:
                        for h in range(h0, hc + 1):
                            at = self.kept_at[g][h] + ((k * lc) >> h)
                            matches[h] = matches[h] and bool(torch.equal(levels[h], want[at:at + (lc >> h)]))
                    level.append(levels[hc].cpu())
        with self.on(self.gpus[0]):
            dev = self.data_g[0].device
            dig = torch.empty((lc, 8), dtype=torch.int64, device=dev)
            up = torch.empty((lc, 8), dtype=torch.int64, device=dev)
            for c in range(self.blocks // chunk, self.tree_leaves >> hc):
                self.lib.call("rb_pads", c * lc, lc, dig.data_ptr())
                level.append(climb(self.lib, dig, c * lc, lc, up)[hc].cpu())
            top = torch.cat(level).to(dev)
            levels = climb(self.lib, top, 0, top.shape[0], torch.empty_like(top), hc)
            if expected is not None:
                want = expected[3].to(dev)
                for k, rows in enumerate(levels):
                    at = self.top_at[hc + k]
                    matches[hc + k] = matches[hc + k] and bool(torch.equal(rows, want[at:at + rows.shape[0]]))
                self.plain_level_matches = matches
            return digests(levels[-1].cpu())[0]

    def arm(self, dropped) -> None:
        """Restore the blocks dropped before, then drop `dropped` (store positions): zeroed on their GPUs, their true
        blocks kept on the host."""
        torch, step = self.torch, 1 << 16
        for g, (sel, true) in enumerate(self.drops):
            with self.on(self.gpus[g]):
                view = self.data_g[g].view(-1, BLOCK)
                for a in range(0, len(sel), step):
                    view[torch.from_numpy(sel[a:a + step]).to(view.device)] = true[a:a + step].to(view.device)
        d = np.unique(np.fromiter(dropped, dtype=np.int64) if isinstance(dropped, (set, frozenset))
                      else np.asarray(dropped, dtype=np.int64))
        drops = []
        for g, gpu in enumerate(self.gpus):
            sel = d[(d >= self.base[g]) & (d < self.base[g] + self.n[g])] - self.base[g]
            true = torch.empty((len(sel), BLOCK), dtype=torch.uint8)
            with self.on(gpu):
                view = self.data_g[g].view(-1, BLOCK)
                for a in range(0, len(sel), step):
                    idx = torch.from_numpy(sel[a:a + step]).to(view.device)
                    true[a:a + step] = view[idx].cpu()
                    view[idx] = 0
            drops.append((sel, true))
        self.drops = drops
        self.sync()

    def _open_shard(self, g: int, items: list, seeds: list, pos: list, offs: list, work: list, rec, path, blk) -> int:
        """Openings of the passes in `work` ((item, the indices of its positions in shard g)) on GPU g, into rows of
        rec (receipts), path (levels below hc) and blk (blocks) at their flat indices: the subtrees they fall in."""
        torch, h0, hc, gpu = self.torch, self.h0, self.hc, self.gpus[g]
        S, per = PER_LEAF << h0, 2 * (1 << h0) - 2
        sub_at = [0] * (h0 + 1)
        for f in range(h0):
            sub_at[f + 1] = sub_at[f] + ((1 << h0) >> f)
        base, base_leaf = self.base[g], self.base[g] // PER_LEAF
        with self.on(gpu, gpu.open) as s:
            dev = self.data_g[g].device
            firsts = [np.unique((pos[k][sel] // PER_LEAF) >> h0 << h0) for k, sel in work]
            total = sum(len(f) for f in firsts)
            ys = torch.empty((total * S, 8), dtype=torch.int64, device=dev)
            nodes = torch.empty((total * per, 8), dtype=torch.int64, device=dev)
            cs = torch.from_numpy(np.concatenate(firsts)).to(dev)
            ys0, nd0 = np.uint64(ys.data_ptr()), np.uint64(nodes.data_ptr())
            rows, order, hold, q0 = [], [], [], 0
            for (k, sel), first in zip(work, firsts):
                _seed, pk, kept, _top = items[k][0]
                pk, kp = pk.to(dev), kept[g].to(dev)
                hold.append((pk, kp))
                self.set_pass(self.lib_open, seeds[k])
                rc = self.lib_open.so.rs_subtrees(self.data_g[g].data_ptr(), base, pk.data_ptr(), cs.data_ptr() + 8 * q0,
                                                  len(first), h0, ys.data_ptr() + 64 * S * q0,
                                                  nodes.data_ptr() + 64 * per * q0, s.cuda_stream)
                if rc:
                    raise RuntimeError(f"rs_subtrees on GPU {gpu.index}: {rc}")
                leaf = pos[k][sel] // PER_LEAF
                q = q0 + np.searchsorted(first, leaf >> h0 << h0)
                local = leaf - (leaf >> h0 << h0)
                addr = np.empty((len(leaf), PER_LEAF + hc), dtype=np.uint64)
                addr[:, :PER_LEAF] = ys0 + np.uint64(64) * (S * q[:, None] + PER_LEAF * local[:, None]
                                                            + np.arange(PER_LEAF)).astype(np.uint64)
                kp0 = np.uint64(kp.data_ptr())
                for h in range(hc):
                    sib = (leaf >> h) ^ 1
                    if h < h0:
                        at = per * q + sub_at[h] + (sib - ((leaf >> h0 << h0) >> h))
                        addr[:, PER_LEAF + h] = nd0 + np.uint64(64) * at.astype(np.uint64)
                    else:
                        at = self.kept_at[g][h] + sib - (base_leaf >> h)
                        addr[:, PER_LEAF + h] = kp0 + np.uint64(64) * at.astype(np.uint64)
                rows.append(addr)
                order.append(offs[k] + sel)
                q0 += len(first)
            addrs = np.concatenate(rows)
            n = addrs.size
            got = torch.empty((n, 8), dtype=torch.int64, device=dev)
            self.lib_open.call("rs_gather64", torch.from_numpy(addrs.reshape(-1).view(np.int64)).to(dev).data_ptr(), n,
                               got.data_ptr())
            idx = np.concatenate(order)
            raw = got.view(torch.uint8).view(n, 8, 8).flip(-1).reshape(len(idx), PER_LEAF + hc, 64).cpu().numpy()
            rec[idx] = raw[:, :PER_LEAF].reshape(len(idx), -1)
            path[idx, :hc] = raw[:, PER_LEAF:]
            local = np.concatenate([pos[k][sel] for k, sel in work]) - base
            blk[idx] = self.data_g[g].view(-1, BLOCK)[torch.from_numpy(local).to(dev)].cpu().numpy()
            dsel, true = self.drops[g]
            if len(dsel):
                at = np.searchsorted(dsel, local)
                hit = (at < len(dsel)) & (dsel[np.minimum(at, len(dsel) - 1)] == local)
                if hit.any():
                    blk[idx[hit]] = true[torch.from_numpy(at[hit])].numpy()
        return total

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        """For each (pass handle, positions) of a window, each position's (block, leaf, leaf salt, receipt path,
        payload salt, payload path), shard by shard (`_open_shard`, OPEN_BATCH passes a time), the path above hc
        from each pass's top levels and the payload paths from the CPU-resident payload tree."""
        hc, H = self.hc, self.H
        seeds = [seed if isinstance(seed, bytes) else bytes(seed.cpu().numpy()) for (seed, *_), _ in items]
        tops = [(top.cpu().numpy() if hasattr(top, "cpu") else np.asarray(top)).view(np.uint64)
                for (*_, top), _ in items]
        pos = [np.asarray(ps, dtype=np.int64) for _, ps in items]
        offs = np.concatenate([[0], np.cumsum([len(p) for p in pos])]).astype(np.int64)
        total = int(offs[-1])
        rec = np.empty((total, PER_LEAF * 64), dtype=np.uint8)
        path = np.empty((total, H, 64), dtype=np.uint8)
        blk = np.empty((total, BLOCK), dtype=np.uint8)
        t0, subtrees = time.perf_counter_ns(), 0
        for g in range(len(self.gpus)):
            lo, hi = self.base[g], self.base[g] + self.n[g]
            work = [(k, np.nonzero((p >= lo) & (p < hi))[0]) for k, p in enumerate(pos)]
            work = [(k, sel) for k, sel in work if len(sel)]
            for b in range(0, len(work), OPEN_BATCH):
                subtrees += self._open_shard(g, items, seeds, pos, offs, work[b:b + OPEN_BATCH], rec, path, blk)
        t1 = time.perf_counter_ns()
        for k, p in enumerate(pos):
            leaf = p // PER_LEAF
            for h in range(hc, H):
                rows = tops[k][self.top_at[h] + ((leaf >> h) ^ 1)]
                path[offs[k]:offs[k + 1], h] = rows.astype(">u8").view(np.uint8).reshape(-1, 64)
        flat = np.concatenate(pos) if pos else np.zeros(0, dtype=np.int64)
        hp = self.payload_H
        payload = np.empty((len(flat), hp, 8), dtype=np.uint64)
        for h in range(hp):
            sib = (flat >> h) ^ 1
            src, at = ((self.payload_low, self.payload_low_at[h]) if h < self.payload_h0
                       else (self.payload_kept, self.payload_kept_at[h]))
            payload[:, h] = src[at + sib]
        praw = payload.astype(">u8").view(np.uint8).reshape(len(flat), hp, 64)
        t2 = time.perf_counter_ns()
        out = []
        for k, (seed, (_handle, positions)) in enumerate(zip(seeds, items)):
            mine = []
            for r, p in enumerate(positions):
                q = offs[k] + r
                mine.append((blk[q].tobytes(), rec[q].tobytes(), store_salt(self.key, seed, p // PER_LEAF),
                             tuple(path[q, h].tobytes() for h in range(H)), store_salt(self.key, PAYLOAD_LABEL, p),
                             tuple(praw[q, h].tobytes() for h in range(hp))))
            out.append(mine)
        self.open_stats.append({"at_ns": t0, "openings": total, "subtrees": subtrees, "subtrees_ns": t1 - t0,
                                "gather_ns": t2 - t1, "assemble_ns": time.perf_counter_ns() - t2})
        return out

    def open(self, handle, positions: list[int]) -> list[tuple]:
        return self.open_window([(handle, positions)])[0]


class Device:
    """The prover's backend (`prover.ProverSide`): the store across `gpus` GPUs (every visible one if None), its gates
    and calibration, its passes and openings."""

    def __init__(self, torch, lib_path: str, kv_path: str, blocks: int, *, seed: int, log=print, gpus: int | None = None,
                 gate_chunk: int = 2048, gate_chunks: int = 3, gate_samples: int = 256,
                 calibration_passes: int = 4000, warm_openings: int = 2773, warm_passes: int = 64,
                 chunk: int = 1 << 18, kept_from: int = 6, sketch_ctas: int = 128, payload_kept_from: int = 6,
                 offload_handles: bool = False, trace_allocator: bool = False) -> None:
        self.torch, self.log, self.blocks, self.warm_openings = torch, log, blocks, warm_openings
        self.trace_allocator, self.calibration_trace = trace_allocator, None
        self.gate_chunk, self.gate_chunks, self.gate_samples = gate_chunk, gate_chunks, gate_samples
        self.calibration_passes, self.warm_passes = calibration_passes, warm_passes
        self.shape = {"chunk": chunk, "kept_from": kept_from, "sketch_ctas": sketch_ctas,
                      "payload_kept_from": payload_kept_from, "offload_handles": offload_handles}
        self.lib = Lib(lib_path, torch)
        self._open_copy = tempfile.NamedTemporaryFile(suffix=".so")
        shutil.copyfile(lib_path, self._open_copy.name)
        self.lib_open = Lib(self._open_copy.name, torch)
        self.kv = Kv(kv_path, torch)
        count = torch.cuda.device_count() if gpus is None else gpus
        if count < 1 or count > torch.cuda.device_count():
            raise ValueError(f"{count} GPUs of {torch.cuda.device_count()} visible")
        rc = self.lib.so.rs_enable_peers()
        if count > 1 and rc:
            raise RuntimeError(f"rs_enable_peers: {rc}: the leader reads and writes every GPU over P2P")
        # torch maps its expandable segments for a peer only once its own copy path has enabled that pair, so one copy
        # each way between every two GPUs, before anything is allocated, lets rs_top reach every GPU's buffers.
        for a in range(count):
            for b in range(count):
                if a != b:
                    torch.empty(1, device=f"cuda:{b}").copy_(torch.empty(1, device=f"cuda:{a}"))
        torch.cuda.synchronize()
        self.gpus = [Gpu(torch, g) for g in range(count)]
        self.rng = np.random.default_rng(seed)
        self.salt_key = self.rng.bytes(32)
        self.offload_jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._offload, daemon=True).start()
        self.offload_ns: list[int] = []
        self.gates: dict = {}
        self.store = DeviceStore(torch, self.lib, self.lib_open, self.kv, blocks, self.salt_key, seed, gpus=self.gpus,
                                 **self.shape)

    def payload_root(self, node: bytes) -> bytes:
        self.node = node
        root = self.store.payload_tree(node)
        self.store.sync()
        return root

    def encode(self, registration: audit.Registration) -> dict:
        self.gates = self.gate(registration)
        self.log(f"device: gates {self.gates}")
        if not self.gates["ok"]:
            raise RuntimeError(f"a gate failed, nothing is timed: {self.gates}")
        rate = self.store.encode(registration.key)
        last = self.blocks - 1
        self.gates["encoded_last_block"] = self.store.block(last) == self.reference_last
        if not self.gates["encoded_last_block"]:
            raise RuntimeError("the store's last block is not the codec's encode of its payload")
        self.encode_bytes_per_s = rate
        return {"encode_bytes_per_s": rate, "blocks": self.blocks, "bytes": self.blocks * BLOCK}

    def calibrate(self) -> dict:
        """The read alone on every GPU at once (`rb_read`, best of 6), `calibration_passes` chained passes queued as a
        session queues them (each root's interval from the one before, the allowance's pass, and the host's time from
        a root's arrival to where a session sends it, its gap; it takes a high percentile of each, `percentiles`), a
        pass's root and kept levels against the plain pass's at the same seed over the whole store, and a window's
        openings over the calibration's first passes."""
        torch, s = self.torch, self.store
        xr = [torch.zeros(1, dtype=torch.int64, device=d.device) for d in s.data_g]
        reads = []
        for _ in range(6):
            s.sync()
            t0 = time.perf_counter_ns()
            for g, gpu in enumerate(self.gpus):
                with s.on(gpu):
                    self.lib.call("rb_read", s.data_g[g].data_ptr(), s.n[g] * BLOCK, xr[g].data_ptr(), 16 * gpu.sms)
            s.sync()
            reads.append((time.perf_counter_ns() - t0) / 1e6)
        challenge, prev = self.rng.bytes(32), bytes(64)
        # Timed as a session runs: what the passes keep allocated once beforehand (the allocator growing under a
        # timed pass stalls it for tens of ms) and no collector.
        s.warm(4)
        gc.collect()
        gc.freeze()
        gc.disable()
        stats0 = [allocator_stats(torch, gpu.index) for gpu in self.gpus]
        history = self.trace_allocator and allocator_history(torch, True)
        clock0 = (time.time_ns(), time.perf_counter_ns())
        s.trace = []
        s.start_chain(challenge, prev)
        observed, gaps, kept, first = [], [], [], None
        g0 = len(s.pass_gpu_ms)
        for i, root, handle, at in s.chain(self.calibration_passes):
            if first is None:
                first = (rb.seed(challenge, 0, prev), root, handle)
            if len(kept) < self.warm_passes:
                kept.append(handle)
            observed.append(at)
            gaps.append(time.perf_counter_ns() - at)
        trace, s.trace = s.trace, None
        gc.enable()
        gc.unfreeze()
        s.sync()
        events = allocator_history(torch, False) if history else None
        stats1 = [allocator_stats(torch, gpu.index) for gpu in self.gpus]
        self.calibration_trace = stall_report(observed, trace, events, clock0,
                                              [{k: b[k] - a[k] for k in a} for a, b in zip(stats0, stats1)])
        gpu = s.pass_gpu_ms[g0:]
        seed, chained_root, chained = first
        self.gates["device_seed"] = bytes(chained[0].cpu().numpy()) == seed
        plain_root = s.plain_root(seed, chained)
        self.gates["plain_root"] = plain_root == chained_root
        self.gates["plain_levels"] = s.plain_level_matches
        self.gates["device_signs"] = all(s.signs_match(rb.seed(challenge, i, bytes(64))) for i in range(4))
        if not self.gates["device_seed"]:
            raise RuntimeError("the device's chained seed is not the driver's")
        if not self.gates["plain_root"] or not all(s.plain_level_matches.values()):
            raise RuntimeError(f"the chained pass differs from the plain pass at levels "
                               f"{[h for h, ok in s.plain_level_matches.items() if not ok]}")
        if not self.gates["device_signs"]:
            raise RuntimeError("the device's signs are not the reference's")
        # A window's openings once, over the calibration's first passes, before any session: the openers' streams
        # reach their allocations here and not in a session, where growing them stalls the passes.
        draws = [(int(self.rng.integers(len(kept))), int(self.rng.integers(s.blocks))) for _ in range(self.warm_openings)]
        o0 = time.perf_counter_ns()
        s.open_window([(kept[i], [p for k, p in draws if k == i]) for i in sorted({i for i, _ in draws})])
        open_ns = time.perf_counter_ns() - o0
        del kept
        s.pass_gpu_ms.clear()
        torch.cuda.empty_cache()
        read_ms = min(reads[1:])
        intervals = sorted(b - a for a, b in zip(observed[1:], observed[2:]))
        gaps = sorted(gaps[1:])
        gpu = sorted(gpu[1:])
        return {"pass_ns_max": intervals[-1], "pass_ns_median": intervals[len(intervals) // 2],
                "pass_ns": percentiles(intervals), "gap_ns_max": gaps[-1], "gap_ns_median": gaps[len(gaps) // 2],
                "gap_ns": percentiles(gaps), "passes": len(intervals), "pass_gpu_ms_median": gpu[len(gpu) // 2],
                "pass_gpu_ms": percentiles(gpu), "read_ms": read_ms, "read_bytes_per_s": s.blocks * BLOCK / (read_ms / 1e3),
                "tau": intervals[len(intervals) // 2] / 1e6 / read_ms, "tau_gpu": gpu[len(gpu) // 2] / read_ms,
                "slow_passes": self.calibration_trace["slow"],
                "max_over_median": intervals[-1] / intervals[len(intervals) // 2],
                "warm_openings": len(draws), "warm_open_ns": open_ns,
                "shape": {**self.shape, "h0": s.h0, "hc": s.hc, "gpus": len(self.gpus), "shard_blocks": s.n,
                          "pads": s.pads},
                "device": [torch.cuda.get_device_name(gpu.index) for gpu in self.gpus]}

    def arm(self, dropped) -> None:
        self.store.arm(dropped)
        self.store.pass_gpu_ms.clear()
        self.offload_ns.clear()

    def warmup(self, ns: int) -> dict:
        """Untimed passes before a session's first window (`DeviceStore.warmup`)."""
        out = self.store.warmup(ns)
        self.offload_ns.clear()
        return out

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        self.store.start_chain(challenge, previous)

    def launch(self, i: int, after: Ticket | None) -> Ticket:
        return self.store.launch(i, after)

    def wait(self, ticket: Ticket):
        root, handle = self.store.wait(ticket)
        if not self.store.offload_handles:
            return root, handle
        out = OffloadedHandle()
        self.offload_jobs.put((handle, out))
        return root, out

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        self.offload_jobs.join()
        return self.store.open_window([(self._ready(handle), positions) for handle, positions in items])

    def _offload(self) -> None:
        torch, staging = self.torch, {}
        while True:
            (seed, pk, kept, top), out = self.offload_jobs.get()
            t0 = time.perf_counter_ns()
            try:
                parts = [pk, top, *kept] + ([] if isinstance(seed, bytes) else [seed])
                for k, x in enumerate(parts):
                    st = staging.get(k)
                    if st is None or st.shape != x.shape:
                        st = staging[k] = torch.empty(x.shape, dtype=x.dtype, device="cpu", pin_memory=True)
                    gpu = self.gpus[x.device.index]
                    with torch.cuda.stream(gpu.offload):
                        st.copy_(x, non_blocking=True)
                for gpu in self.gpus:
                    gpu.offload.synchronize()
                # The pinned buffers are a one-handle staging ring. Persist each handle in ordinary host memory so a
                # session's thousands of passes do not pin tens of GiB.
                host = [staging[k].clone() for k in range(len(parts))]
                out.value = (seed if isinstance(seed, bytes) else bytes(host[-1].numpy()), host[0],
                             host[2:2 + len(kept)], host[1])
            except BaseException as error:
                out.value = error
            finally:
                self.offload_ns.append(time.perf_counter_ns() - t0)
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
        return {"passes": len(gpu), "pass_gpu_ms_sum": sum(gpu), "pass_gpu_ms_median":
                sorted(gpu)[len(gpu) // 2] if gpu else None, "pass_gpu_ms_max": max(gpu) if gpu else None,
                "offload_ns_median": sorted(self.offload_ns)[len(self.offload_ns) // 2] if self.offload_ns else None,
                "offload_ns_max": max(self.offload_ns) if self.offload_ns else None}

    def gate(self, registration: audit.Registration) -> dict:
        torch, key, out = self.torch, registration.key, {}
        last = self.blocks - 1
        self.reference_last = rcodec.encode(self.store.block(last), last, key)
        small = DeviceStore(torch, self.lib, self.lib_open, self.kv, len(self.gpus) * self.gate_chunks * self.gate_chunk,
                            self.salt_key, int(self.rng.integers(1 << 62)), gpus=self.gpus, chunk=self.gate_chunk,
                            kept_from=self.shape["kept_from"], sketch_ctas=self.shape["sketch_ctas"],
                            payload_kept_from=self.shape["payload_kept_from"])
        out["shards"] = {"blocks": small.blocks, "shard_blocks": small.n, "h0": small.h0, "hc": small.hc,
                         "pads": small.pads}
        payloads = small.host_blocks(0, small.blocks)
        root = small.payload_tree(self.node)
        salts = [store_salt(self.salt_key, PAYLOAD_LABEL, j) for j in range(small.blocks)]
        out["payload_root"] = root == audit.payload_tree(self.node, payloads, salts).commitment.root
        small.encode(key)
        encoded = small.host_blocks(0, small.blocks)
        checked = sorted({1, *small.base, small.blocks - 1})
        out["encode"] = all(encoded[j] == rcodec.encode(payloads[j], j, key) for j in checked)
        out["decode"] = all(rcodec.decode(encoded[j], j, key) == payloads[j] for j in range(min(256, small.blocks)))
        reg = audit.Registration(registration.node, small.blocks, root, registration.codec_seed, registration.codec,
                                 registration.receipt, registration.bind, registration.layout, registration.rows)
        seed = rb.seed(self.rng.bytes(32), 0, reg.digest)
        got, _ = small.run_pass(seed)
        r = rb.receipter(reg.receipt, seed, reg.rows)
        ys = [r(j, encoded[j]) for j in range(small.blocks)]
        ref = rb.tree(seed, small.plan, ys, [store_salt(self.salt_key, seed, c) for c in range(small.blocks // PER_LEAF)])
        out["pass_root"] = got == ref.commitment.root
        out["session_honest"] = self.gate_session(small, reg, set())
        out["session_skip_sixteenth"] = self.gate_session(small, reg, set(range(0, small.blocks, 16)))
        out["ok"] = (out["payload_root"] and out["encode"] and out["decode"] and out["pass_root"]
                     and out["session_honest"]["accepted"] and out["session_skip_sixteenth"]["failure"] == "receipt")
        # `small` and its gate sessions use the same shared libraries as the real store, so its fixed chain templates
        # replaced the real store's on every GPU. Restore both copies before calibration derives the first real pass.
        for gpu in self.gpus:
            with self.store.on(gpu):
                self.lib.set_chain(self.store.plan, self.salt_key)
                self.lib_open.set_chain(self.store.plan, self.salt_key)
        del small
        torch.cuda.empty_cache()
        return out

    def gate_session(self, small: DeviceStore, reg: audit.Registration, dropped: set[int]) -> dict:
        """Two windows of two passes each over `small` under a clock the gate turns, `gate_samples` draws a window,
        checked by `audit.Session`."""
        now = [0]
        params = audit.Params(reg.blocks, 1000, 2, self.gate_samples, 2.0 ** -40, window_ns=1, codec=reg.codec,
                              receipt=reg.receipt, rows=reg.rows)
        session = audit.Session(params, reg, None, coins.Challenger(reg.digest, 0, params.coins), clock=lambda: now[0],
                                not_before_ns=0)
        small.arm(dropped)
        coin, prev, kept = session.start(), reg.digest, {}
        for t in range(2):
            small.start_chain(coin.coin, prev if t == 0 else None)
            for i in range(2):
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
            by = {}
            for k, (i, p) in enumerate(draws):
                by.setdefault(i, []).append((k, p))
            fields = [None] * len(draws)
            passes = sorted(by)
            for i, got in zip(passes, small.open_window([(kept[t, i], [p for _, p in by[i]]) for i in passes])):
                for (k, _), f in zip(by[i], got):
                    fields[k] = f
            openings = [audit.Opening(i, p, *f, b"") for (i, p), f in zip(draws, fields)]
            v = session.verify(t, openings)
            if not v.accepted:
                small.arm(set())
                return {"accepted": False, "failure": v.failure, "at": v.at, "dropped": v.at[1] in dropped}
        small.arm(set())
        return {"accepted": True, "failure": None, "draws": sum(len(w.draws) for w in session.windows)}


def add_arguments(ap) -> None:
    ap.add_argument("--lib", required=True, help="session.so")
    ap.add_argument("--kv", required=True, help="kv.so at the registration's codec (p1024 masked)")
    ap.add_argument("--gib", type=float, default=16.0,
                    help="the store's size across every GPU, rounded down to whole --chunk-blocks (need not be a power "
                         "of two; the GPUs' shards differ by at most a chunk)")
    ap.add_argument("--gpus", type=int, default=0, help="the GPUs the store spans (0: every visible one)")
    ap.add_argument("--gate-chunk", type=int, default=2048, help="the gate store's chunk (three a GPU)")
    ap.add_argument("--gate-samples", type=int, default=256, help="the gate sessions' draws a window")
    ap.add_argument("--calibration-passes", type=int, default=4000,
                    help="so the allowance's p99.9 is a few passes in, not the worst")
    ap.add_argument("--warm-openings", type=int, default=2773, help="a window's openings, opened once in calibration")
    ap.add_argument("--chunk-blocks", type=int, default=1 << 18, help="a pass's chunk (`rs_pass`)")
    ap.add_argument("--kept-from", type=int, default=6, help="h0: the pass tree's levels kept for the openings")
    ap.add_argument("--payload-kept-from", type=int, default=6,
                    help="h0: the payload tree's levels kept; lower paths are recomputed for openings")
    ap.add_argument("--sketch-ctas", type=int, default=128, help="the receipts grid's CTAs (0: one an SM)")
    ap.add_argument("--offload-handles", action="store_true",
                    help="archive pass handles through a dedicated pinned copy stream (automatic above 64 GiB)")
    ap.add_argument("--trace-allocator", action="store_true",
                    help="record the caching allocator's events during calibration, against its slow passes")


def from_arguments(a, log=print) -> Device:
    import torch

    n = int(a.gib * 2 ** 30) // BLOCK
    blocks = n // a.chunk_blocks * a.chunk_blocks
    return Device(torch, a.lib, a.kv, blocks, seed=a.seed, log=log, gpus=a.gpus or None, gate_chunk=a.gate_chunk,
                  gate_samples=a.gate_samples, calibration_passes=a.calibration_passes, warm_openings=a.warm_openings,
                  chunk=a.chunk_blocks, kept_from=a.kept_from, sketch_ctas=a.sketch_ctas,
                  payload_kept_from=a.payload_kept_from, offload_handles=a.offload_handles or a.gib > 64,
                  trace_allocator=a.trace_allocator)
