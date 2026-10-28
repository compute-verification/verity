"""The prover's GPU backend: an encoded store across the visible GPUs, its payload tree, chained sketch-sha512 passes
over it with the device reader (`../codec_tensor/reader.cu` in `session.cu`) and the openings of what the verifier
draws.

~~~text
segments   the store's N blocks as contiguous segments of whole chunks: first a shard a GPU in its HBM, as even as the
           chunks allow, then, with a host tier, a segment a GPU in pinned host memory on that GPU's NUMA node, which
           the GPU reads over its own PCIe link. Segment s holds blocks base_s … base_s + n_s − 1, and every position,
           leaf and node index is the store's, so the tree is `receipts.tree` over the whole store in block order
           whatever the GPUs and tiers (one GPU and no host tier is the same code)
payload    N blocks of seeded random bytes, each 128-byte word's top byte below 0x80 (so below p1024 and p512), or a
           buffer a GPU as it is (`hbm`: a serving engine's weights, `node.py`) and then a host tier of zeros (the
           store `remote_bandwidth.store` lays out), its tree the driver's `audit.payload_tree` with salt_j = SHAKE256(D(store
           salt) ‖ key ‖ PAYLOAD_LABEL ‖ u64be j), key the verifier's (`set_payload_key`) or else the store's own;
           built a chunk at a time on the chunk's GPU and retained on the CPU, so it does not displace the store
store      E = the codec over the payload in place, by kv.cu (`../word_codec/kv.cu`) on each segment's GPU at the
           registration's key and the store's positions
pass       on each GPU, its seed, signs and constants derived from the previous root in its root ring
           (`rs_chain_prepare`), its shard's sketch-sha512 receipts on tensor cores with the leaves and levels up to
           each chunk's root at hc hashed under the reads (`rs_pass`), and its share of the padding up to hc, and
           beside it on two streams of its own its host segment's, by a few CTAs (`--host-ctas`, `--host-consumers`)
           over PCIe; then on the leader (GPU 0) the levels above hc from every segment's over P2P, the root written
           to every GPU's ring (`rs_top`). Each pass is queued on every GPU behind the leader's event for the root it chains from, while
           the pass before it runs, so the passes follow one another on the devices with no host step between them.
           A pass keeps its packed signs (8 KiB), each segment's levels h0 … hc and the levels above, nothing per block,
           in rings allocated with the store and copied to host memory as its part ends, so a pass allocates nothing
opening    a window's draws at once, segment by segment on its GPU's opening stream: each drawn leaf's subtree of 2^h0
           leaves recomputed from the segment under its pass's constants by a second copy of the library
           (`rs_subtrees`), then its receipts and its path below hc gathered from those subtrees and the segment's kept
           levels in one launch (`rs_gather64`), the blocks in another; the path above hc comes from the pass's top
           levels and the payload paths from the CPU-resident payload tree
skip       the dropped blocks zeroed where they are stored, their true blocks kept on the host for openings
~~~

Gates, before anything is timed (`Device.gates`), over a small store across the same GPUs and tiers (three chunks a
GPU, and with a host tier one more a GPU in host memory, so its tree has padding on every GPU when there are several):
the payload root against `audit.payload_tree` and blocks of the store against `codec.encode` (at each segment's first
block and the store's last), a pass's root against `receipts.tree` over the reference's receipts, and sessions under
the registration's key, one honest and accepted by `audit.Session`, one with 1/16 of its blocks dropped and rejected at
a receipt, and with a host tier one with only host blocks dropped (every fourth) and rejected at a receipt; then in
calibration, the chained pass against the plain pass at every kept level over the whole store, and its seed and signs
against the reference's.
"""
from __future__ import annotations

import bisect
import contextlib
import ctypes
import gc
import queue
import resource
import secrets
import shutil
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

from benchmarks.word_codec import cuda_lib  # noqa: E402
from verity.core.protocols.remote_bandwidth import audit, coins, receipts as rb  # noqa: E402
from verity.core.protocols.remote_bandwidth import codec as rcodec  # noqa: E402
from verity.core.protocols.remote_bandwidth.store import PAYLOAD_LABEL, SALT_DOMAIN, store_salt  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix  # noqa: E402
from verity.ml import randomness  # noqa: E402

import sketch_receipt as skr  # noqa: E402
import reader as rdr  # noqa: E402

BLOCK = rcodec.BLOCK_BYTES
PER_LEAF = rb.PER_LEAF
PACK16_WORDS = BLOCK // 64 * 32  # session.cu's SK16_PACK
TOP_SEGS = 32  # session.cu's: rs_top's segments, real and padding
SLOTS = 8
"""A GPU's root ring: the root of the k-th pass launched goes to slot k mod SLOTS, the window's previous root from the
host to slot SLOTS. A slot is read at most three launches after its write."""
OPEN_BATCH = 128
"""Passes whose kept levels are brought back to a GPU at once to open a window."""


def stream_head(seed: bytes) -> bytes:
    """The head of row 0's stream of a pass's signs (`skr.sign_bits`), its last byte the row: `rs_signs16`'s input."""
    key = randomness.derive(rb._digest(seed, "a pass seed"), rb.SIGN_DOMAIN,
                            {"tag": rb.SKETCH_SHA512, "k": skr.ROWS, "n": BLOCK})
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
                     "part_ms": [round(x, 3) for x in t["part_ms"]], "host_ms": [round(x, 3) for x in t["host_ms"]],
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


_v, _u64, _i, _b = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_char_p
SESSION = {   # session.cu's own functions, beside the reader's it includes (`reader.ABI`)
    "rs_payload_leaves": [_v, _u64, _u64, _v, _v],
    "rs_gather64": [_v, _u64, _v, _v],
    "rs_pass": [_v, _u64, _u64, _v, _v, _v, _v, _v, _v, _i, _u64, _i, _u64, _u64, _v, _v, _v, _v, _i],
    "rs_top": [_v, _v, _i, _v, _i, _i, _v, _i, _v],
    "rs_enable_peers": [],
    "rs_subtrees": [_v, _u64, _v, _v, _u64, _i, _v, _v, _v],
    "rs_signs16": [_b, _i, _v, _v],
    "rs_chain_set": [_b, _v],
    "rs_chain_prepare": [ctypes.c_uint, _v, _v, _v, _v],
    "rs_pack16": [_v, _v, _v],
    "rs_sketch16_receipts": [_v, _u64, _u64, _v, _v, _i, _v],
    "rs_map": [_v, _u64, _v, _i, _u64, _v, _v, _v],
    "rs_mapped_receipts": [_v, _v, _u64, _v, _v, _i, _v],
    "rs_mapped_pass": [_v, _v, _u64, _v, _v, _u64, _v, _v, _i, _i, _v, _v],
    "rs_mapped_read": [_v, _u64, _v, _i, _v],
}
KV = {"kv_set_key": [_v, _i, _v], "kv_set_mask": [_v], "kv_encode": [_v, _v, _u64, _u64, _i, _i, _v]}   # kv.cu's


class Lib:
    def __init__(self, path: str, torch) -> None:
        self.so, self.torch = cuda_lib.load(path, {**rdr.ABI, **SESSION}), torch

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
        self.so, self.torch = cuda_lib.load(path, KV), torch
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
            self.host_main = torch.cuda.Stream()
            self.host_side = torch.cuda.Stream(priority=-1)
        props = torch.cuda.get_device_properties(index)
        self.sms = props.multi_processor_count
        self.numa = gpu_numa(props)


def gpu_numa(props) -> int:
    """The NUMA node of a GPU's PCIe slot (sysfs), or -1."""
    try:
        bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        return int(Path(f"/sys/bus/pci/devices/{bdf}/numa_node").read_text())
    except (AttributeError, OSError, ValueError):
        return -1


SYS_SET_MEMPOLICY, MPOL_DEFAULT, MPOL_BIND = 238, 0, 2  # x86_64


@contextlib.contextmanager
def numa_bound(node: int):
    """This thread's new pages on NUMA node `node` (set_mempolicy MPOL_BIND), or wherever the kernel puts them if
    node < 0."""
    if node < 0:
        yield
        return
    libc = ctypes.CDLL(None, use_errno=True)
    mask = ctypes.c_ulong(1 << node)
    if libc.syscall(ctypes.c_long(SYS_SET_MEMPOLICY), ctypes.c_long(MPOL_BIND), ctypes.byref(mask),
                    ctypes.c_ulong(64)):
        raise OSError(ctypes.get_errno(), f"set_mempolicy(MPOL_BIND, node {node})")
    try:
        yield
    finally:
        libc.syscall(ctypes.c_long(SYS_SET_MEMPOLICY), ctypes.c_long(MPOL_DEFAULT), None, ctypes.c_ulong(0))


def host_tier(torch, nbytes: int, node: int):
    """nbytes of host memory on NUMA node `node`, registered with CUDA portable and mapped, so every GPU's kernels
    read and write it at its own address, over PCIe (`cudaHostRegister`; `release` unregisters it)."""
    with numa_bound(node):
        t = torch.empty(nbytes, dtype=torch.uint8)
        rc = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 3)
    if int(rc):
        raise RuntimeError(f"cudaHostRegister of {nbytes} bytes: {rc}")
    return t


def release(torch, t) -> None:
    rc = torch.cuda.cudart().cudaHostUnregister(t.data_ptr())
    if int(rc):
        raise RuntimeError(f"cudaHostUnregister: {rc}")


class OffloadedHandle:
    """A pass's handle, (seed, packed signs, each segment's levels h0 … hc, the levels above hc) in host memory, once the
    store's offloader has copied it out of the pinned ring (`DeviceStore.ready`)."""

    __slots__ = ("value", "done")

    def __init__(self) -> None:
        self.value, self.done = None, threading.Event()


@dataclass(slots=True)
class Ticket:
    """A queued pass: its slot in the rings, and its handle."""

    slot: int
    handle: OffloadedHandle


class DeviceStore:
    """N blocks in segments of whole chunks, their payload tree on the CPU, then the encoded store and the passes over
    it: `blocks` in HBM, a shard a GPU across `gpus` (the current GPU if none), then `host_blocks` in host memory
    (`host_tier`, each GPU's share on its NUMA node), a segment a GPU, which that GPU reads over its own PCIe link in
    the same pass, beside its shard (`host_ctas` receipt CTAs and `host_consumers` consumer CTAs). Segment s holds the
    store's blocks base_s … base_s + n_s − 1 in that order, so every position, leaf and node index is the store's
    whatever the tier. `generator` seeds each segment's blocks: an int (segment s's from seed + s), or for one GPU and
    no host tier a torch generator on it. With `hbm` (a uint8 buffer a GPU, on it, whole chunks, such as a serving
    engine's weights arena, `node.py`) the HBM tier is those buffers, their bytes as they are the payload, so a GPU's
    shard is its buffer's chunks and nothing is filled.

    What a pass keeps goes to a ring of SLOTS on each GPU and from there, on each GPU's offload stream as its part ends,
    to a pinned host ring, both allocated with the store's shape (`shape`); the store's offloader copies each slot into
    the pass's handle in ordinary host memory and frees it for the pass SLOTS later. So no pass allocates."""

    def __init__(self, torch, lib: Lib, lib_open: Lib, kv: Kv, blocks: int, key: bytes, generator, *, gpus=None,
                 chunk: int = 1 << 18, kept_from: int = 6, sketch_ctas: int = 128, payload_kept_from: int = 6,
                 host_blocks: int = 0, host_ctas: int = 16, host_consumers: int = 8, hbm=None) -> None:
        gpus = gpus or [Gpu(torch, torch.cuda.current_device(), own=False)]
        G = len(gpus)
        if blocks % PER_LEAF or blocks % chunk or blocks // chunk < G or G > 16 or host_blocks % chunk:
            raise ValueError(f"{blocks} blocks on {G} GPUs and {host_blocks} in host memory: the harness takes "
                             f"multiples of the pass chunk and {PER_LEAF}, at least a chunk a GPU and at most 16 GPUs")
        if host_blocks and gpus[0].main is None:
            raise ValueError("a host tier needs GPUs with streams of their own")
        total = blocks + host_blocks
        self.torch, self.lib, self.lib_open, self.kv, self.blocks, self.key = torch, lib, lib_open, kv, total, key
        self.payload_key = key
        self.gpus, self.hbm_blocks = gpus, blocks
        self.host_ctas, self.host_consumers = host_ctas, host_consumers
        chunks, hchunks = blocks // chunk, host_blocks // chunk
        per = [chunks // G + (g < chunks % G) for g in range(G)]
        if hbm is not None:
            per = [t.numel() // (chunk * BLOCK) for t in hbm]
            if (len(hbm) != G or sum(per) != chunks or any(t.numel() != p * chunk * BLOCK for t, p in zip(hbm, per))
                    or any(t.device != torch.device("cuda", gpu.index) for t, gpu in zip(hbm, gpus))):
                raise ValueError(f"the HBM tier's buffers are one a GPU, on it, each whole chunks, {blocks} blocks in all")
        hper = [hchunks // G + (g < hchunks % G) for g in range(G)]
        self.seg_gpu = list(range(G)) + [g for g in range(G) if hper[g]]
        self.n = [p * chunk for p in per] + [hper[g] * chunk for g in range(G) if hper[g]]
        self.base = [sum(self.n[:s]) for s in range(len(self.n))]
        self.on_host = [s >= G for s in range(len(self.n))]
        self.gpu_segs = [[s for s, h in enumerate(self.seg_gpu) if h == g] for g in range(G)]
        self.H = (total // PER_LEAF - 1).bit_length()
        self.tree_leaves = 1 << self.H
        self.payload_h0 = max(1, min(payload_kept_from, chunk.bit_length() - 1))
        self.data_g = []
        for s, g in enumerate(self.seg_gpu):
            gpu = gpus[g]
            dev = torch.device("cuda", gpu.index)
            if isinstance(generator, int):
                gen = torch.Generator(device=dev)
                gen.manual_seed(generator + s)
            elif len(self.n) == 1:
                gen = generator
            else:
                raise ValueError("a store on several GPUs or tiers takes an int seed")
            with self.on(gpu):
                n = self.n[s]
                if hbm is not None:
                    self.data_g.append(hbm[g] if not self.on_host[s] else host_tier(torch, n * BLOCK, gpu.numa).zero_())
                    continue
                if self.on_host[s]:
                    d = host_tier(torch, n * BLOCK, gpu.numa)
                    stage = torch.empty(min(n, 1 << 18) * BLOCK, dtype=torch.uint8, device=dev)
                else:
                    d = stage = torch.empty(n * BLOCK, dtype=torch.uint8, device=dev)
                for j in range(0, n, 1 << 18):
                    m = min(1 << 18, n - j)
                    v = stage[:m * BLOCK] if self.on_host[s] else stage[j * BLOCK:(j + m) * BLOCK]
                    v.view(torch.int32).copy_(torch.randint(-2 ** 31, 2 ** 31 - 1, (m * BLOCK // 4,), dtype=torch.int32,
                                                            device=dev, generator=gen))
                    v.view(-1, 128)[:, 127] &= 0x7F
                    if self.on_host[s]:
                        d[j * BLOCK:(j + m) * BLOCK].copy_(v)
                del stage
            self.data_g.append(d)
        self.plan = rb.Plan.whole(total)
        for gpu in gpus:
            with self.on(gpu):
                self.lib.set_chain(self.plan, self.key)
                self.lib_open.set_chain(self.plan, self.key)
        self.drops = [(np.zeros(0, dtype=np.int64), torch.empty((0, BLOCK), dtype=torch.uint8)) for _ in self.n]
        self.pass_gpu_ms: list[float] = []
        self.pass_host_ms: list[float] = []
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
        self.prepared = [[torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)] for _ in gpus]
        self.host_done = [[torch.cuda.Event(enable_timing=True) for _ in range(SLOTS)] for _ in gpus]
        self.copied = [[torch.cuda.Event() for _ in range(SLOTS)] for _ in gpus]
        self.free = [threading.Event() for _ in range(SLOTS)]
        for f in self.free:
            f.set()
        self.trace: list | None = None
        self._launch_trace: dict = {}
        self.offload_ns: list[int] = []
        self.offload_jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._offload, daemon=True).start()
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
        """A pass's chunk (blocks), h0 (at most the chunk's subtree height hc) and the sketch's CTAs, the segments' and
        the top's level offsets, each GPU's share of the padding at hc, each GPU's ring and each segment's."""
        torch = self.torch
        chunk = min(chunk, min(self.n))
        lc = chunk // PER_LEAF
        if any(b % chunk or n % chunk for b, n in zip(self.base, self.n)) or lc & (lc - 1):
            raise ValueError(f"a chunk of {chunk} blocks: every segment is whole chunks of a power of two of leaves")
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
        if len(self.n) + sum(c > 0 for c in counts) > TOP_SEGS:
            raise ValueError(f"{len(self.n)} segments and their padding are more than rs_top's {TOP_SEGS}")
        ready_blocks = min(chunk, 1 << 13)
        self.offload_jobs.join()
        self.ring = self.seg_ring = None
        ring, seg_ring = [], []
        for g, gpu in enumerate(self.gpus):
            dev, pad_n = torch.device("cuda", gpu.index), self.pads[g][1] * lc
            with self.on(gpu):
                r = {"pad": torch.empty((max(2 * pad_n, 1), 8), dtype=torch.int64, device=dev),
                     "pad_out": torch.empty((max(self.pads[g][1], 1), 8), dtype=torch.int64, device=dev),
                     "seed": torch.empty(rb.DIGEST_BYTES, dtype=torch.uint8, device=dev),
                     "pk": torch.empty(PACK16_WORDS, dtype=torch.int32, device=dev),
                     "roots": torch.zeros((SLOTS + 1, 8), dtype=torch.int64, device=dev)}
                parts = []
                if g == 0:
                    r["seeds"] = torch.empty((SLOTS, rb.DIGEST_BYTES), dtype=torch.uint8, device=dev)
                    r["pks"] = torch.empty((SLOTS, PACK16_WORDS), dtype=torch.int32, device=dev)
                    r["top"] = torch.empty((SLOTS, self.top_at[H + 1], 8), dtype=torch.int64, device=dev)
                    parts += ["seeds", "pks", "top"]
                r["host"] = {k: torch.empty(r[k].shape, dtype=r[k].dtype, pin_memory=True) for k in parts}
                ring.append(r)
        for s, g in enumerate(self.seg_gpu):
            dev, n = torch.device("cuda", self.gpus[g].index), self.n[s]
            with self.on(self.gpus[g]):
                x = {"ctr": torch.empty(3 + n // ready_blocks + n // chunk, dtype=torch.int32, device=dev),
                     "ys": torch.empty((n, 8), dtype=torch.int64, device=dev),
                     "dig": torch.empty((2 * lc, 8), dtype=torch.int64, device=dev),
                     "up": torch.empty((2 * lc, 8), dtype=torch.int64, device=dev),
                     "kept": torch.empty((SLOTS, self.kept_at[s][hc + 1], 8), dtype=torch.int64, device=dev)}
                x["host"] = torch.empty(x["kept"].shape, dtype=torch.int64, pin_memory=True)
                seg_ring.append(x)
        self.ring, self.seg_ring = ring, seg_ring
        self.sync()

    def segment_of(self, j: int) -> int:
        return bisect.bisect_right(self.base, j) - 1

    def block(self, j: int) -> bytes:
        s = self.segment_of(j)
        local = j - self.base[s]
        sel, true = self.drops[s]
        k = int(np.searchsorted(sel, local))
        if k < len(sel) and sel[k] == local:
            return true[k].numpy().tobytes()
        return self.data_g[s][local * BLOCK:(local + 1) * BLOCK].cpu().numpy().tobytes()

    def blocks_at(self, j0: int, n: int) -> list[bytes]:
        """Blocks j0 … j0 + n − 1 as they are stored (dropped ones zero)."""
        raw = []
        for s in range(len(self.n)):
            a, b = max(j0, self.base[s]), min(j0 + n, self.base[s] + self.n[s])
            if a < b:
                raw.append(self.data_g[s][(a - self.base[s]) * BLOCK:(b - self.base[s]) * BLOCK].cpu().numpy().tobytes())
        raw = b"".join(raw)
        return [raw[q * BLOCK:(q + 1) * BLOCK] for q in range(n)]

    def close(self) -> None:
        """Stop the offloader once it has every launched pass's handle, and unregister the host tier."""
        self.offload_jobs.join()
        self.offload_jobs.put(None)
        for s, d in enumerate(self.data_g):
            if self.on_host[s]:
                release(self.torch, d)

    def payload_tree(self, node: bytes) -> bytes:
        """The payload tree, each chunk's leaves and levels on its segment's GPU, the padding and the levels above a
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

        for s, g in enumerate(self.seg_gpu):
            gpu = self.gpus[g]
            with self.on(gpu):
                dev = torch.device("cuda", gpu.index)
                self.lib.set_tree(bytes(128), BLOCK, BLOCK, dom.domain_id, SALT_DOMAIN + self.payload_key + PAYLOAD_LABEL,
                                  N)
                dig = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                up = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                for first in range(self.base[s], self.base[s] + self.n[s], chunk):
                    self.lib.call("rs_payload_leaves", self.data_g[s].data_ptr() + (first - self.base[s]) * BLOCK, first,
                                  chunk, dig.data_ptr())
                    keep(climb(self.lib, dig, first, chunk, up), first)
                if s == 0:
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
        """The store from its payload, in place, every segment at once (a host segment by its GPU, over PCIe); the
        encode's rate in bytes a second."""
        self.sync()
        t0 = time.perf_counter_ns()
        for gpu in self.gpus:
            with self.on(gpu):
                self.kv.set_key(key)
        for s, g in enumerate(self.seg_gpu):
            with self.on(self.gpus[g]):
                for j in range(0, self.n[s], chunk):
                    self.kv.encode(self.data_g[s][j * BLOCK:], self.base[s] + j, min(chunk, self.n[s] - j))
        self.sync()
        return self.blocks * BLOCK / ((time.perf_counter_ns() - t0) / 1e9)

    def set_pass(self, lib: Lib, seed: bytes) -> None:
        """A pass's constants from the host on the current GPU."""
        dom = rb.domain(seed, self.plan)
        lib.set_tree(rb.prefix(seed), rb.DIGEST_BYTES * PER_LEAF, rb.DIGEST_BYTES * PER_LEAF, dom.domain_id,
                     SALT_DOMAIN + self.key + seed, self.blocks)
        lib.call("rb_sketch_set", rb.sketch_prefix(seed))

    def launch(self, i: int, after: Ticket | None, seed: bytes | None = None) -> Ticket:
        """Queue pass i chained from `after`'s root (None: the window's previous root, `start_chain`) on every GPU,
        each behind the leader's event for that root, or with `seed` an unchained pass under that seed's constants
        from the host; its ticket at once (`wait`)."""
        so, G, hc = self.lib.so, len(self.gpus), self.hc
        tr = None if self.trace is None else [time.perf_counter_ns(), time.thread_time_ns(), *_faults()]
        slot = self.launched % SLOTS
        self.launched += 1
        # The offloader has copied this slot's last pass out of the pinned ring, so its copies on every GPU are done.
        self.free[slot].wait()
        self.free[slot].clear()
        prev = SLOTS if after is None else after.slot
        head = stream_head(seed) if seed is not None else None
        lc = self.chunk // PER_LEAF
        for g, gpu in enumerate(self.gpus):
            ring = self.ring[g]
            with self.on(gpu) as s:
                if after is not None:
                    s.wait_event(self.top_done[after.slot])
                if g == 0:
                    self.started[slot].record(s)
                self.began[g][slot].record(s)
                sd = ring["seeds"][slot] if g == 0 else ring["seed"]
                pk = ring["pks"][slot] if g == 0 else ring["pk"]
                if tr is not None:
                    tr.append(time.perf_counter_ns())
                if seed is None:
                    rc = so.rs_chain_prepare(i, ring["roots"].data_ptr() + 64 * prev, sd.data_ptr(), pk.data_ptr(),
                                             s.cuda_stream)
                else:
                    self.set_pass(self.lib, seed)
                    rc = so.rs_signs16(head, len(head), pk.data_ptr(), s.cuda_stream)
                hbm, *host = self.gpu_segs[g]
                if host and not rc:
                    # The host segment first, so its few CTAs are on the device before the shard's grids fill it.
                    self.prepared[g][slot].record(s)
                    gpu.host_main.wait_event(self.prepared[g][slot])
                    for q in host:
                        x = self.seg_ring[q]
                        rc = rc or so.rs_pass(self.data_g[q].data_ptr(), self.n[q], self.base[q], pk.data_ptr(),
                                              x["ctr"].data_ptr(), x["ys"].data_ptr(), x["dig"].data_ptr(),
                                              x["up"].data_ptr(), x["kept"][slot].data_ptr(), self.h0, self.chunk,
                                              self.host_ctas, 0, 0, None, None, gpu.host_main.cuda_stream,
                                              gpu.host_side.cuda_stream, self.host_consumers)
                    self.host_done[g][slot].record(gpu.host_main)
                first, count = self.pads[g]
                x = self.seg_ring[hbm]
                rc = rc or so.rs_pass(self.data_g[hbm].data_ptr(), self.n[hbm], self.base[hbm], pk.data_ptr(),
                                      x["ctr"].data_ptr(), x["ys"].data_ptr(), x["dig"].data_ptr(),
                                      x["up"].data_ptr(), x["kept"][slot].data_ptr(), self.h0, self.chunk,
                                      self.sketch_ctas, first * lc, count * lc, ring["pad"].data_ptr(),
                                      ring["pad_out"].data_ptr(), s.cuda_stream, gpu.side.cuda_stream, 0)
                if rc:
                    raise RuntimeError(f"rs_pass on GPU {gpu.index}: {rc}")
                if host:
                    s.wait_event(self.host_done[g][slot])
                self.ended[g][slot].record(s)
                if tr is not None:
                    tr.append(time.perf_counter_ns())
        leader = self.gpus[0]
        segs = [x["kept"][slot].data_ptr() + 64 * self.kept_at[q][hc] for q, x in enumerate(self.seg_ring)]
        counts = [n // self.chunk for n in self.n]
        for g, (_, count) in enumerate(self.pads):
            if count:
                segs.append(self.ring[g]["pad_out"].data_ptr())
                counts.append(count)
        roots = [r["roots"].data_ptr() + 64 * slot for r in self.ring]
        with self.on(leader) as s:
            for g in range(1, G):
                s.wait_event(self.ended[g][slot])
            top = self.ring[0]["top"][slot]
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
        for g, gpu in enumerate(self.gpus):
            ring = self.ring[g]
            with self.on(gpu, gpu.offload) as c:
                c.wait_event(self.top_done[slot] if g == 0 else self.ended[g][slot])
                for k, host in ring["host"].items():
                    host[slot].copy_(ring[k][slot], non_blocking=True)
                for q in self.gpu_segs[g]:
                    self.seg_ring[q]["host"][slot].copy_(self.seg_ring[q]["kept"][slot], non_blocking=True)
                self.copied[g][slot].record(c)
        handle = OffloadedHandle()
        self.offload_jobs.put((slot, seed, handle))
        if tr is not None:
            self._launch_trace[slot] = tr + [time.perf_counter_ns(), time.thread_time_ns(), *_faults()]
        return Ticket(slot, handle)

    def _offload(self) -> None:
        """Each launched pass's handle out of the pinned ring once its copies are done, then its slot freed."""
        while True:
            job = self.offload_jobs.get()
            if job is None:
                self.offload_jobs.task_done()
                return
            slot, seed, handle = job
            t0 = time.perf_counter_ns()
            try:
                for g in range(len(self.gpus)):
                    self.copied[g][slot].synchronize()
                host = self.ring[0]["host"]
                handle.value = (seed if seed is not None else bytes(host["seeds"][slot].numpy()),
                                host["pks"][slot].clone(), [x["host"][slot].clone() for x in self.seg_ring],
                                host["top"][slot].clone())
            except BaseException as error:
                handle.value = error
            finally:
                self.free[slot].set()
                handle.done.set()
                self.offload_ns.append(time.perf_counter_ns() - t0)
                self.offload_jobs.task_done()

    @staticmethod
    def ready(handle) -> tuple:
        """A pass's handle once the offloader has it: (seed, packed signs, each segment's kept levels, the top
        levels)."""
        if not isinstance(handle, OffloadedHandle):
            return handle
        handle.done.wait()
        if isinstance(handle.value, BaseException):
            raise RuntimeError("a pass handle's offload failed") from handle.value
        return handle.value

    def wait(self, ticket: Ticket):
        """A queued pass's root once the host has it, and its handle (`ready`)."""
        w0 = time.perf_counter_ns()
        self.root_ready[ticket.slot].synchronize()
        w1 = time.perf_counter_ns()
        root = self.root_host[ticket.slot].numpy().astype(np.uint64).astype(">u8").tobytes()
        k = ticket.slot
        self.pass_gpu_ms.append(self.started[k].elapsed_time(self.top_done[k]))
        host = [self.prepared[g][k].elapsed_time(self.host_done[g][k]) for g, segs in enumerate(self.gpu_segs)
                if len(segs) > 1]
        if host:
            self.pass_host_ms.append(max(host))
        if self.trace is not None:
            self.trace.append({"launch": self._launch_trace.pop(k, None), "wait": [w0, w1], "gpu_ms": self.pass_gpu_ms[-1],
                               "part_ms": [self.began[g][k].elapsed_time(self.ended[g][k]) for g in range(len(self.gpus))],
                               "host_ms": host})
        return root, ticket.handle

    def run_pass(self, seed: bytes):
        """One unchained pass under `seed`: its root, and what its openings need, (seed, packed signs, each segment's
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
        """The same pass the plain way, chunk by chunk on each segment's GPU (every receipt, then every leaf, then the
        levels), the padding and the levels above a chunk's on the leader, for the pipelined pass's check.

        With `expected` (a pass handle), record agreement at every kept level as well as at the root
        (`plain_level_matches`): each segment's levels h0 … hc and the levels above, so a tree's real/padding boundary
        and its segment boundaries, the tiers' among them, are part of the gate rather than only its final digest."""
        torch, chunk, hc, h0 = self.torch, self.chunk, self.hc, self.h0
        lc = chunk // PER_LEAF
        matches = {h: True for h in range(h0, self.H + 1)}
        level = []
        for s, g in enumerate(self.seg_gpu):
            gpu = self.gpus[g]
            with self.on(gpu):
                dev = torch.device("cuda", gpu.index)
                self.set_pass(self.lib, seed)
                signs = torch.from_numpy(skr.signs(seed)).to(dev)
                sk = torch.empty((chunk, skr.ROWS), dtype=torch.int32, device=dev)
                ys = torch.empty((chunk, 8), dtype=torch.int64, device=dev)
                dig = torch.empty((lc, 8), dtype=torch.int64, device=dev)
                up = torch.empty((lc, 8), dtype=torch.int64, device=dev)
                want = None if expected is None else expected[2][s].to(dev)
                for k in range(self.n[s] // chunk):
                    j0 = self.base[s] + k * chunk
                    self.lib.call("rb_sketch", self.data_g[s].data_ptr() + k * chunk * BLOCK, None, chunk,
                                  signs.data_ptr(), sk.data_ptr(), 0)
                    self.lib.call("rb_sketch_hash", sk.data_ptr(), j0, chunk, ys.data_ptr())
                    self.lib.call("rb_leaves", ys.data_ptr(), j0 // PER_LEAF, lc, dig.data_ptr(), None)
                    levels = climb(self.lib, dig, j0 // PER_LEAF, lc, up)
                    if want is not None:
                        for h in range(h0, hc + 1):
                            at = self.kept_at[s][h] + ((k * lc) >> h)
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
        """Restore the blocks dropped before, then drop `dropped` (store positions): zeroed where they are stored, their
        true blocks kept on the host. Each segment's count dropped (`dropped_by_segment`)."""
        torch, step = self.torch, 1 << 16
        for s, (sel, true) in enumerate(self.drops):
            with self.on(self.gpus[self.seg_gpu[s]]):
                view = self.data_g[s].view(-1, BLOCK)
                for a in range(0, len(sel), step):
                    view[torch.from_numpy(sel[a:a + step]).to(view.device)] = true[a:a + step].to(view.device)
        d = np.unique(np.fromiter(dropped, dtype=np.int64) if isinstance(dropped, (set, frozenset))
                      else np.asarray(dropped, dtype=np.int64))
        drops = []
        for s, g in enumerate(self.seg_gpu):
            sel = d[(d >= self.base[s]) & (d < self.base[s] + self.n[s])] - self.base[s]
            true = torch.empty((len(sel), BLOCK), dtype=torch.uint8)
            with self.on(self.gpus[g]):
                view = self.data_g[s].view(-1, BLOCK)
                for a in range(0, len(sel), step):
                    idx = torch.from_numpy(sel[a:a + step]).to(view.device)
                    true[a:a + step] = view[idx].cpu()
                    view[idx] = 0
            drops.append((sel, true))
        self.drops = drops
        self.dropped_by_segment = [len(sel) for sel, _ in drops]
        self.sync()

    def _open_segment(self, g: int, items: list, seeds: list, pos: list, offs: list, work: list, rec, path, blk) -> int:
        """Openings of the passes in `work` ((item, the indices of its positions in segment g)) on its GPU, into rows
        of rec (receipts), path (levels below hc) and blk (blocks) at their flat indices: the subtrees they fall in."""
        torch, h0, hc, gpu = self.torch, self.h0, self.hc, self.gpus[self.seg_gpu[g]]
        S, per = PER_LEAF << h0, 2 * (1 << h0) - 2
        sub_at = [0] * (h0 + 1)
        for f in range(h0):
            sub_at[f + 1] = sub_at[f] + ((1 << h0) >> f)
        base, base_leaf = self.base[g], self.base[g] // PER_LEAF
        with self.on(gpu, gpu.open) as s:
            dev = torch.device("cuda", gpu.index)
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
            blk[idx] = self.data_g[g].view(-1, BLOCK)[torch.from_numpy(local).to(self.data_g[g].device)].cpu().numpy()
            dsel, true = self.drops[g]
            if len(dsel):
                at = np.searchsorted(dsel, local)
                hit = (at < len(dsel)) & (dsel[np.minimum(at, len(dsel) - 1)] == local)
                if hit.any():
                    blk[idx[hit]] = true[torch.from_numpy(at[hit])].numpy()
        return total

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        """For each (pass handle, positions) of a window, each position's (block, leaf, leaf salt, receipt path,
        payload salt, payload path), segment by segment (`_open_segment`, OPEN_BATCH passes a time), the path above hc
        from each pass's top levels and the payload paths from the CPU-resident payload tree."""
        hc, H = self.hc, self.H
        items = [(self.ready(handle), positions) for handle, positions in items]
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
        for g in range(len(self.n)):
            lo, hi = self.base[g], self.base[g] + self.n[g]
            work = [(k, np.nonzero((p >= lo) & (p < hi))[0]) for k, p in enumerate(pos)]
            work = [(k, sel) for k, sel in work if len(sel)]
            for b in range(0, len(work), OPEN_BATCH):
                subtrees += self._open_segment(g, items, seeds, pos, offs, work[b:b + OPEN_BATCH], rec, path, blk)
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
                             tuple(path[q, h].tobytes() for h in range(H)), store_salt(self.payload_key, PAYLOAD_LABEL, p),
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
                 trace_allocator: bool = False, host_blocks: int = 0, host_ctas: int = 16,
                 host_consumers: int = 8, hbm=None) -> None:
        self.torch, self.log, self.blocks, self.warm_openings = torch, log, blocks + host_blocks, warm_openings
        self.host = {"host_ctas": host_ctas, "host_consumers": host_consumers}
        self.trace_allocator, self.calibration_trace = trace_allocator, None
        self.gate_chunk, self.gate_chunks, self.gate_samples = gate_chunk, gate_chunks, gate_samples
        self.calibration_passes, self.warm_passes = calibration_passes, warm_passes
        self.shape = {"chunk": chunk, "kept_from": kept_from, "sketch_ctas": sketch_ctas,
                      "payload_kept_from": payload_kept_from}
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
        self.gates: dict = {}
        self.store = DeviceStore(torch, self.lib, self.lib_open, self.kv, blocks, self.salt_key, seed, gpus=self.gpus,
                                 host_blocks=host_blocks, hbm=hbm, **self.host, **self.shape)
        self.payload_key = self.salt_key

    def set_payload_key(self, key: bytes) -> None:
        """Salt the payload tree with the verifier's key (`prover.ProverSide.register`), before its root is taken."""
        self.payload_key = self.store.payload_key = key

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

    def read_ms(self, segments: list[int], tries: int = 6) -> float:
        """The read alone (`rb_read`) of `segments` at once, each on its GPU, a host segment on the GPU's host stream
        over its PCIe link: 16 CTAs an SM, except that beside HBM a host segment is launched first on a quarter of a
        CTA an SM, which a PCIe link's outstanding reads need, so the HBM read's grid doesn't queue it behind itself.
        The best of `tries` after the first, in ms."""
        s, out = self.store, []
        xr = [self.torch.zeros(1, dtype=self.torch.int64, device=f"cuda:{self.gpus[g].index}") for g in s.seg_gpu]
        beside = any(not s.on_host[q] for q in segments)
        segments = sorted(segments, key=lambda q: not s.on_host[q])
        for _ in range(tries):
            s.sync()
            t0 = time.perf_counter_ns()
            for q in segments:
                gpu = self.gpus[s.seg_gpu[q]]
                ctas = max(1, gpu.sms // 4) if s.on_host[q] and beside else 16 * gpu.sms
                with s.on(gpu, gpu.host_main if s.on_host[q] else None):
                    self.lib.call("rb_read", s.data_g[q].data_ptr(), s.n[q] * BLOCK, xr[q].data_ptr(), ctas)
            s.sync()
            out.append((time.perf_counter_ns() - t0) / 1e6)
        return min(out[1:])

    def calibrate(self) -> dict:
        """The read alone on every GPU at once (`read_ms`: HBM, host and both), `calibration_passes` chained passes queued as a
        session queues them (each root's interval from the one before, the allowance's pass, and the host's time from
        a root's arrival to where a session sends it, its gap; it takes a high percentile of each, `percentiles`), a
        pass's root and kept levels against the plain pass's at the same seed over the whole store, and a window's
        openings over the calibration's first passes."""
        torch, s = self.torch, self.store
        hbm = [q for q in range(len(s.n)) if not s.on_host[q]]
        host = [q for q in range(len(s.n)) if s.on_host[q]]
        read_ms = self.read_ms(hbm + host)
        tiers = {"hbm": {"bytes": s.hbm_blocks * BLOCK, "ms": self.read_ms(hbm) if host else read_ms}}
        if host:
            tiers["host"] = {"bytes": (s.blocks - s.hbm_blocks) * BLOCK, "ms": self.read_ms(host),
                             "numa": [self.gpus[s.seg_gpu[q]].numa for q in host]}
        for t in tiers.values():
            t["bytes_per_s"] = t["bytes"] / (t["ms"] / 1e3)
        challenge, prev = self.rng.bytes(32), bytes(64)
        # Timed as a session runs, with no collector.
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
        chained = s.ready(chained)
        self.gates["device_seed"] = chained[0] == seed
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
        host_ms = sorted(s.pass_host_ms[1:])
        s.pass_gpu_ms.clear()
        s.pass_host_ms.clear()
        torch.cuda.empty_cache()
        intervals = sorted(b - a for a, b in zip(observed[1:], observed[2:]))
        gaps = sorted(gaps[1:])
        gpu = sorted(gpu[1:])
        return {"pass_ns_max": intervals[-1], "pass_ns_median": intervals[len(intervals) // 2],
                "pass_ns": percentiles(intervals), "gap_ns_max": gaps[-1], "gap_ns_median": gaps[len(gaps) // 2],
                "gap_ns": percentiles(gaps), "passes": len(intervals), "pass_gpu_ms_median": gpu[len(gpu) // 2],
                "pass_gpu_ms": percentiles(gpu), "read_ms": read_ms, "read_bytes_per_s": s.blocks * BLOCK / (read_ms / 1e3),
                "read_tiers": tiers, "pass_host_ms": percentiles(host_ms) if host_ms else None,
                "tau": intervals[len(intervals) // 2] / 1e6 / read_ms, "tau_gpu": gpu[len(gpu) // 2] / read_ms,
                "slow_passes": self.calibration_trace["slow"],
                "max_over_median": intervals[-1] / intervals[len(intervals) // 2],
                "warm_openings": len(draws), "warm_open_ns": open_ns,
                "shape": {**self.shape, "h0": s.h0, "hc": s.hc, "gpus": len(self.gpus), "segment_blocks": s.n,
                          "segment_gpu": s.seg_gpu, "hbm_blocks": s.hbm_blocks, "host_blocks": s.blocks - s.hbm_blocks,
                          "host_ctas": s.host_ctas, "host_consumers": s.host_consumers, "pads": s.pads},
                "device": [torch.cuda.get_device_name(gpu.index) for gpu in self.gpus]}

    def arm(self, dropped) -> None:
        self.store.arm(dropped)
        self.store.pass_gpu_ms.clear()
        self.store.pass_host_ms.clear()
        self.store.offload_ns.clear()

    def warmup(self, ns: int) -> dict:
        """Untimed passes before a session's first window (`DeviceStore.warmup`)."""
        out = self.store.warmup(ns)
        self.store.pass_host_ms.clear()
        self.store.offload_ns.clear()
        return out

    def begin_window(self, challenge: bytes, previous: bytes | None) -> None:
        self.store.start_chain(challenge, previous)

    def launch(self, i: int, after: Ticket | None) -> Ticket:
        return self.store.launch(i, after)

    def wait(self, ticket: Ticket):
        return self.store.wait(ticket)

    def open_window(self, items: list[tuple[object, list[int]]]) -> list[list[tuple]]:
        return self.store.open_window(items)

    def session_stats(self) -> dict:
        s = self.store
        gpu, offload, host = s.pass_gpu_ms, s.offload_ns, s.pass_host_ms
        dropped = getattr(s, "dropped_by_segment", [0] * len(s.n))
        return {"passes": len(gpu), "pass_gpu_ms_sum": sum(gpu), "pass_gpu_ms_median":
                sorted(gpu)[len(gpu) // 2] if gpu else None, "pass_gpu_ms_max": max(gpu) if gpu else None,
                "pass_host_ms_median": sorted(host)[len(host) // 2] if host else None,
                "pass_host_ms_max": max(host) if host else None,
                "offload_ns_median": sorted(offload)[len(offload) // 2] if offload else None,
                "offload_ns_max": max(offload) if offload else None, "hbm_blocks": s.hbm_blocks,
                "dropped_hbm": sum(d for d, h in zip(dropped, s.on_host) if not h),
                "dropped_host": sum(d for d, h in zip(dropped, s.on_host) if h)}

    def gate(self, registration: audit.Registration) -> dict:
        torch, key, out = self.torch, registration.key, {}
        last = self.blocks - 1
        self.reference_last = rcodec.encode(self.store.block(last), last, key)
        small = DeviceStore(torch, self.lib, self.lib_open, self.kv, len(self.gpus) * self.gate_chunks * self.gate_chunk,
                            self.salt_key, int(self.rng.integers(1 << 62)), gpus=self.gpus, chunk=self.gate_chunk,
                            kept_from=self.shape["kept_from"], sketch_ctas=self.shape["sketch_ctas"],
                            payload_kept_from=self.shape["payload_kept_from"],
                            host_blocks=len(self.gpus) * self.gate_chunk if self.store.hbm_blocks < self.blocks else 0,
                            **self.host)
        out["segments"] = {"blocks": small.blocks, "hbm_blocks": small.hbm_blocks, "segment_blocks": small.n,
                           "h0": small.h0, "hc": small.hc, "pads": small.pads}
        payloads = small.blocks_at(0, small.blocks)
        small.payload_key = self.payload_key
        root = small.payload_tree(self.node)
        salts = [store_salt(self.payload_key, PAYLOAD_LABEL, j) for j in range(small.blocks)]
        out["payload_root"] = root == audit.payload_tree(self.node, payloads, salts).commitment.root
        small.encode(key)
        encoded = small.blocks_at(0, small.blocks)
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
        if small.hbm_blocks < small.blocks:
            out["session_skip_host"] = self.gate_session(small, reg, set(range(small.hbm_blocks, small.blocks, 4)))
            out["ok"] = out["ok"] and out["session_skip_host"]["failure"] == "receipt"
        # `small` and its gate sessions use the same shared libraries as the real store, so its fixed chain templates
        # replaced the real store's on every GPU. Restore both copies before calibration derives the first real pass.
        for gpu in self.gpus:
            with self.store.on(gpu):
                self.lib.set_chain(self.store.plan, self.salt_key)
                self.lib_open.set_chain(self.store.plan, self.salt_key)
        small.close()
        del small
        torch.cuda.empty_cache()
        return out

    def gate_session(self, small: DeviceStore, reg: audit.Registration, dropped: set[int]) -> dict:
        """Two windows of two passes each over `small` under a clock the gate turns, `gate_samples` draws a window,
        checked by `audit.Session`."""
        now = [0]
        params = audit.Params(reg.blocks, 1000, 2, self.gate_samples, 2.0 ** -40, window_ns=1, codec=reg.codec,
                              receipt=reg.receipt, rows=reg.rows)
        session = audit.Session(params, reg, None, coins.local(reg.digest, 0, params.windows, secrets.token_bytes, appends=1),
                                clock=lambda: now[0], not_before_ns=0)
        small.arm(dropped)
        coin, prev, kept = session.start(), reg.digest, {}
        for t in range(2):
            small.start_chain(coins.coin(coin), prev if t == 0 else None)
            for i in range(2):
                prev, kept[t, i] = small.run_chain(i)
                now[0] += 1
                if session.receive(t, i, prev, step=2 * t + i) is not None:
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
                return {"accepted": False, "failure": v.failure, "at": v.at, "dropped": v.at[1] in dropped,
                        "host_tier": v.at[1] >= small.hbm_blocks}
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
    ap.add_argument("--trace-allocator", action="store_true",
                    help="record the caching allocator's events during calibration, against its slow passes")
    ap.add_argument("--host-gib", type=float, default=0.0,
                    help="the store's host tier, rounded down to whole --chunk-blocks: pinned host memory, each GPU's "
                         "share on its NUMA node, read in the same pass over that GPU's PCIe link")
    ap.add_argument("--host-ctas", type=int, default=16, help="a host segment's receipt CTAs")
    ap.add_argument("--host-consumers", type=int, default=8, help="a host segment's consumer CTAs")


def from_arguments(a, log=print) -> Device:
    import torch

    n = int(a.gib * 2 ** 30) // BLOCK
    blocks = n // a.chunk_blocks * a.chunk_blocks
    host = int(a.host_gib * 2 ** 30) // BLOCK // a.chunk_blocks * a.chunk_blocks
    return Device(torch, a.lib, a.kv, blocks, seed=a.seed, log=log, gpus=a.gpus or None, gate_chunk=a.gate_chunk,
                  gate_samples=a.gate_samples, calibration_passes=a.calibration_passes, warm_openings=a.warm_openings,
                  chunk=a.chunk_blocks, kept_from=a.kept_from, sketch_ctas=a.sketch_ctas,
                  payload_kept_from=a.payload_kept_from, trace_allocator=a.trace_allocator, host_blocks=host,
                  host_ctas=a.host_ctas, host_consumers=a.host_consumers)
