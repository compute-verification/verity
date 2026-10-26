#!/usr/bin/env python3
"""A salt-to-deadline window of Pearl-C rev1 (`pearl-c-sm120-v1-h1`) on node 2's cards: every card runs the 8,192³ GEMM of
`../pearl_c_sm120` (run.Pipeline's fused call, the panel's v1-h1) back to back from the salt's release at t_s to the deadline
t_d = t_s + DURATION, commits each GEMM's roots, and closes its transcript by t_d; nothing observed after t_d is credited.

The window.
  1. Before t_s, on each card (a worker process, one visible device, its UUID held to the server table's): A's two operand
     sets and B as the harness's fill (`../harness/fill.py`'s F32 twin, seeds `a_seed` and SEED_B), B's frame-b3 row tree
     on the device and its root read back; every card's root must be the same.  Then one warm-up GEMM under a zero salt.
  2. At t_s the parent draws `source` = os.urandom(32), and the window's seed is derive(source, SEED_DOMAIN, {t_s, the
     cards' UUIDs}); the salt is the epoch's (`Epoch.unverified` on that seed and B's root).  It is written to start.json
     and released to every card.
  3. Each card runs the weight side under the salt (E_B's and the F lines, B's forming, B̃·F_A, G, P_B), poisons its call
     buffers (0xA5), and then GEMM i = 0, 1, ...: matmul index `index_of(card, i)` (card << 32 | i) on weight 0, so every
     GEMM has its own activation domain and unit key, hence its own A tree, seed_A, noise and tile tree; A is operand set
     i mod 2.  Each GEMM is the arm's call: A's row tree, seed_A and E_A's lines, A's forming, A′·F_B, P_A, gemm_pearlc_b
     (y and the message digests from registers), the tile leaves and the tile tree.  `schedule` `serial` runs a GEMM after
     the last on one stream; `overlap` runs GEMM i+1's commitment (A's tree, seed_A, E_A's lines) on a second stream beside
     GEMM i.  The host reads each GEMM's (root_A, root_T) back in batches of BATCH and credits a batch only if it saw it
     complete by t_d.  It stops launching at t_d − guard.
  4. Each card's transcript: entry_i = tagged(ENTRY, card, i, root_A_i, root_T_i), and its root the RFC 6962 Merkle root
     over its entries (`merkle_root`).  The card sends (count, root) when it stops; the parent takes the closes it holds at
     t_d, and the window root is tagged(ROOT, salt, t_s, t_d, each card's (index, UUID, count, root)), written to
     window.json with the time it was computed, which must be at most t_d.
  5. After the close, the parent draws `source2` = os.urandom(32) and `draws` takes T tiles (uniform over every credited
     tile: all GEMMs and tiles are of equal work) from derive(source2, DRAW_DOMAIN, {window root, salt}).  Each card re-runs
     its drawn GEMMs into poisoned buffers, holds the re-run's roots to the logged ones, and dumps tree_A, tree_T, the tile
     leaves and each entry's Merkle path (draws/), B's tree once (weights/); the card of draw 0 also re-runs that GEMM with
     its first message-digest byte flipped between the product and the hashes (controls/tampered/).  `audit.py` checks it
     all on the CPU.

  python window.py run --out DIR --server-md GPUS.md [--duration 1800] [--guard 4] [--schedule serial|overlap] [--draws 60]
                        [--hashing h1|h2]
  python window.py worker ...   (the parent's; one card)

Every card in CUDA_VISIBLE_DEVICES (a lease's comma-separated indices) is a worker.  PEARLC_SHIP names build.sh's output (the
cubin).  EXH_CLOCK=fake:START_NS:STEP_NS replaces the clock (each reading advances it by STEP_NS) for the CPU stand-in.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parent / "pearl_c_sm120"
HARNESS = HERE.parent / "harness"


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = _load("pearl_c_sm120_run", KERNEL / "run.py")
F = _load("pearl_c_sm120_fixture", KERNEL / "fixture.py")
FL = _load("pouw_harness_fill", HARNESS / "fill.py")
ID = _load("pouw_harness_identity", HARNESS / "identity.py")

from verity.ml.randomness import derive  # noqa: E402
from verity.core.protocols.pouw.audit import Epoch, Workload, activation_binding, rows_domain, weight_binding  # noqa: E402
from verity.core.protocols.pouw.protocol import Matmul, tagged  # noqa: E402
from verity.core.protocols.pouw.protocol import Shape as MShape  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c_device import DEVICES  # noqa: E402

SCHEMA = "verity-pouw/exhaustion-window/v1"
SEED_DOMAIN = "verity/pouw/exhaustion-window-seed/v1"
DRAW_DOMAIN = "verity/pouw/exhaustion-window-audit/v1"
ENTRY, NODE, CARD, ROOT = (f"verity/pouw/exhaustion-window/{x}/v1" for x in ("entry", "node", "card", "root"))
VARIANT, WEIGHT, SEED_B = "sm120", 0, 0x5EED_B000
SHAPE = (8192, 8192, 8192)                   # (m, n, k)
POISON = 0xA5
#: run.Pipeline's buffers the weight side writes (one copy a card); every other buffer the call touches is the set's
WEIGHT_BUFS = ("eb_codes", "fa_codes", "faT32", "fb_codes", "fbT32", "fbT128", "g", "alpha_b", "beta_b", "ia_b", "aux_b",
               "epb_codes", "bt", "bf", "pb_t", "tree_b", "keys_b", "skeys_b")
SET_BUFS = ("keys_a", "keys_t", "ea_codes", "line_a", "seed_a", "alpha_a", "beta_a", "ia_a", "aux_a", "epa_codes", "at", "af",
            "pa_t", "c", "u", "y", "digests", "leaves", "tree_a", "tree_t", "skeys_a")
CONST_KEYS = ("line_b", "dom_b", "dom_t", "k_msg", "k_leaf")
HASHINGS = ("h1", "h2")
DOM, UNIT = 32, 104                          # a GEMM's keys: dom_a's id, then seed_line_a's unit block
SLOT = DOM + UNIT
BATCH, DEPTH, RING = 16, 2, 8                # GEMMs a batch; batches in flight past the one being read; batches a ring holds
EXIT_USAGE, EXIT_IDENTITY, EXIT_ROOTS, EXIT_LATE = 2, 3, 6, 5


# ---- the window's pure parts (tested on the CPU) -------------------------------------------------------------------------

def scheme(hashing: str = "h1") -> C.PearlC:
    """`pearl-c-sm120-v1-h1` or `-h2`: rev1 forming on sm_120, its rows and tile leaves hashed as -h1 or -h2 hashes them."""
    return C.PearlC(forming="v1", device=DEVICES[VARIANT], hashing=hashing)


def mshape(shape) -> MShape:
    m, n, k = shape
    return MShape(m, k, n)


def index_of(card: int, i: int) -> int:
    return card << 32 | i


def matmul(card: int, i: int, shape=SHAPE) -> Matmul:
    return Matmul(index_of(card, i), mshape(shape), WEIGHT)


def a_seed(card: int, op_set: int) -> int:
    """The fill seed of card `card`'s A operand set `op_set` (GEMM i runs on set i mod 2)."""
    return 0xA000 + 2 * card + op_set


def rows(seed: int, first: int, count: int, k: int) -> list[bytes]:
    """Rows first .. first + count of a (·, k) F32 fill from `seed`, as the device holds them."""
    data = FL.fill(FL.F32, count * k, seed, first * k)
    return [data[4 * k * r:4 * k * (r + 1)] for r in range(count)]


def window_seed(source: bytes, t_s_ns: int, uuids: list[str]) -> bytes:
    return derive(source, SEED_DOMAIN, {"t_s_ns": t_s_ns, "cards": ",".join(uuids)}).stream("seed", 32)


def salt_of(seed: bytes, root_b: bytes, shape=SHAPE) -> bytes:
    """The epoch's salt on the window's seed and B's root (`Epoch.unverified`, as every GEMM's epoch has it)."""
    return Epoch.unverified(Workload((Matmul(0, mshape(shape), WEIGHT),)), {WEIGHT: root_b}, seed).salt


def gemm_epoch(salt: bytes, root_b: bytes, mm: Matmul) -> Epoch:
    """GEMM `mm`'s one-unit epoch: the window's salt on B's root (its transcript is that GEMM's tile tree)."""
    return Epoch(Workload((mm,)), {WEIGHT: root_b}, salt)


def gemm_keys(sch: C.PearlC, salt: bytes, root_b: bytes, mm: Matmul) -> bytes:
    """A GEMM's own key bytes: dom_a's id then the unit block (`fixture.keys_of`'s `dom_a` and `unit`)."""
    dom = rows_domain(activation_binding(salt, mm.index), mm.index, mm.shape.m, sch.commitment_hash).domain_id
    return dom + F.unit_block(salt, root_b, mm.index)


def constant_keys(sch: C.PearlC, salt: bytes, root_b: bytes, shape=SHAPE) -> dict[str, bytes]:
    """The keys every GEMM of the window shares (`fixture.keys_of`'s, at any GEMM: they read only the salt and B's root)."""
    mm = Matmul(0, mshape(shape), WEIGHT)
    keys = F.keys_of(sch, gemm_epoch(salt, root_b, mm), mm, C._labelled(salt + root_b, "seed-B"))
    return {name: keys[name] for name in CONST_KEYS}


def entry(card: int, i: int, root_a: bytes, root_t: bytes) -> bytes:
    return tagged(ENTRY, card, i, root_a, root_t)


def node(left: bytes, right: bytes) -> bytes:
    return tagged(NODE, left, right)


def _split(n: int) -> int:
    """RFC 6962's k: the largest power of two below n (n >= 2)."""
    return 1 << (n - 1).bit_length() - 1


def merkle_root(leaves: list[bytes]) -> bytes:
    """RFC 6962's tree hash over already-hashed leaves, with `node`; the empty tree's is tagged(NODE)."""
    if not leaves:
        return tagged(NODE)
    if len(leaves) == 1:
        return leaves[0]
    k = _split(len(leaves))
    return node(merkle_root(leaves[:k]), merkle_root(leaves[k:]))


def merkle_path(leaves: list[bytes], pos: int) -> list[bytes]:
    """RFC 6962's audit path of leaf `pos`, bottom first."""
    if len(leaves) <= 1:
        return []
    k = _split(len(leaves))
    if pos < k:
        return merkle_path(leaves[:k], pos) + [merkle_root(leaves[k:])]
    return merkle_path(leaves[k:], pos - k) + [merkle_root(leaves[:k])]


def root_from_path(leaf: bytes, pos: int, count: int, path: list[bytes]) -> bytes | None:
    """The root an audit path gives leaf `pos` of `count`, or None if the path's length is not the tree's."""
    if not 0 <= pos < count:
        return None
    if count == 1:
        return leaf if not path else None
    if not path:
        return None
    k = _split(count)
    if pos < k:
        sub = root_from_path(leaf, pos, k, path[:-1])
        return None if sub is None else node(sub, path[-1])
    sub = root_from_path(leaf, pos - k, count - k, path[:-1])
    return None if sub is None else node(path[-1], sub)


class Stack:
    """RFC 6962's root, streamed: `push` each leaf, `root` at any time (the same as `merkle_root` over the leaves so far)."""

    def __init__(self):
        self.parts: list[tuple[int, bytes]] = []
        self.count = 0

    def push(self, leaf: bytes) -> None:
        h, size = leaf, 1
        while self.parts and self.parts[-1][0] == size:
            h, size = node(self.parts.pop()[1], h), 2 * size
        self.parts.append((size, h))
        self.count += 1

    def root(self) -> bytes:
        if not self.parts:
            return tagged(NODE)
        h = self.parts[-1][1]
        for _, left in reversed(self.parts[:-1]):
            h = node(left, h)
        return h


def card_leaf(card: int, uuid: str, count: int, root: bytes) -> bytes:
    return tagged(CARD, card, uuid, count, root)


def window_root(salt: bytes, t_s_ns: int, t_d_ns: int, cards: list[dict]) -> bytes:
    """Over every card's close, in card order: (card, uuid, count, root hex)."""
    return tagged(ROOT, salt, t_s_ns, t_d_ns,
                  *(card_leaf(c["card"], c["uuid"], c["count"], bytes.fromhex(c["root"])) for c in sorted(cards, key=lambda c: c["card"])))


def draws(source: bytes, root: bytes, salt: bytes, counts: list[tuple[int, int]], tiles: int, t: int) -> list[dict]:
    """T tiles, independent and uniform over every credited tile (each GEMM `tiles` tiles of equal work, cards in order):
    [{card, i, tile}].  `counts` is [(card, credited GEMMs)]."""
    counts = sorted(counts)
    total = sum(n for _, n in counts) * tiles
    if not total:
        return []
    key = derive(source, DRAW_DOMAIN, {"window": root, "salt": salt})
    out = []
    for j in range(t):
        x = key.uniform(total, j)
        g, tile = divmod(x, tiles)
        for card, n in counts:
            if g < n:
                out.append({"card": card, "i": g, "tile": tile})
                break
            g -= n
    return out


def clock_from_env():
    """time.time_ns, or under EXH_CLOCK=fake:START:STEP a counter that advances STEP ns each reading."""
    spec = os.environ.get("EXH_CLOCK", "")
    if not spec.startswith("fake:"):
        return time.time_ns
    start, step = (int(x) for x in spec.split(":")[1:3])
    state = [start - step]

    def fake() -> int:
        state[0] += step
        return state[0]
    return fake


def utc(ns: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ns / 1e9)) + f".{ns % 10**9 // 10**6:03d}Z"


# ---- one card --------------------------------------------------------------------------------------------------------

EXTRA = {"cuStreamCreate": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
         "cuStreamWaitEvent": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]}


class Card:
    """One card's window: its device, operand sets, pipelines, streams and rings (module docstring, steps 1, 3 and 5)."""

    def __init__(self, card: int, uuid: str | None, cubin: bytes, shape=SHAPE, schedule: str = "serial", hashing: str = "h1"):
        if schedule not in ("serial", "overlap"):
            raise ValueError(f"schedule {schedule!r}: serial or overlap")
        self.card, self.shape, self.schedule, self.hashing = card, shape, schedule, hashing
        self.sch = scheme(hashing)
        self.dev = R.Dev(cubin)
        self.refused = self.dev.identity(uuid)
        self.uuid = self.dev.uuid
        self.salt = b""
        self.root_b = b""

    def open(self) -> None:
        d, (m, n, k) = self.dev, self.shape
        d.open()
        R.bind(d.cu, EXTRA)
        self.nvml = R.Nvml(d.uuid)
        size = self.size = R.Pipeline.sizes(m, n, k, m)
        shared = {name: d.alloc(size[name]) for name in WEIGHT_BUFS}
        self.const = {name: d.alloc(32) for name in CONST_KEYS}
        self.const.update(t_row=d.put(R.b3s_template(R.ROW_SCHEMA)), t_tile=d.put(R.b3s_template(R.TILE_SCHEMA)))
        self.kring = d.alloc(SLOT * RING * BATCH)
        self.log = d.alloc(64 * RING * BATCH)
        self.xb = self.upload(SEED_B, n)
        self.xa = [self.upload(a_seed(self.card, s), m) for s in (0, 1)]
        self.sets, self.pipes = [], []
        for s in (0, 1):
            own = {name: d.alloc(size[name]) for name in SET_BUFS}
            keys = {**self.const, "dom_a": self.kring, "unit": self.kring + DOM}
            self.sets.append(own)
            self.pipes.append(R.Pipeline(d, m, n, k, VARIANT, keys, self.xa[s], self.xb, bufs={**shared, **own},
                                         hashing=self.hashing))
        self.steps = [{name: (phase, f) for name, phase, f in p.steps()} for p in self.pipes]
        names = [(name, phase) for name, phase, _ in self.pipes[0].steps()]
        self.weight_epoch = [nm for nm, ph in names if ph == "epoch"]
        self.weight_job = [nm for nm, ph in names if ph == "job"]
        self.set_unit = [nm for nm, ph in names if ph == "unit" and nm.endswith("_t")]
        self.commit = ([nm for nm, ph in names if ph == "unit" and nm.endswith("_a")]
                       + [nm for nm, ph in names if ph == "commit"] + ["lines_ea"])
        self.rest = [nm for nm, ph in names if ph in ("forming", "cleanup", "hashing") and nm != "lines_ea"]
        self.stream_m = self.stream()
        self.stream_c = self.stream() if self.schedule == "overlap" else self.stream_m
        self.ev_done = [self.event() for _ in range(RING * BATCH)] if self.schedule == "overlap" else []
        self.ev_commit = [self.event() for _ in range(RING * BATCH)] if self.schedule == "overlap" else []
        self.ev_batch = [self.event() for _ in range(RING)]
        self.ev_keys = self.event()

    def stream(self) -> ctypes.c_void_p:
        s = ctypes.c_void_p()
        self.dev.ck("cuStreamCreate", ctypes.byref(s), 1)       # CU_STREAM_NON_BLOCKING: the null stream's copies don't wait
        return s

    def event(self) -> ctypes.c_void_p:
        e = ctypes.c_void_p()
        self.dev.ck("cuEventCreate", ctypes.byref(e), 2)        # CU_EVENT_DISABLE_TIMING
        return e

    def upload(self, seed: int, count: int, chunk_rows: int = 1024) -> int:
        """A (count x k) F32 fill from `seed` on the device, `chunk_rows` rows a copy."""
        d, k = self.dev, self.shape[2]
        p = d.alloc(4 * count * k)
        for r0 in range(0, count, chunk_rows):
            data = FL.fill(FL.F32, min(chunk_rows, count - r0) * k, seed, r0 * k)
            d.ck("cuMemcpyHtoD_v2", R.u64(p + 4 * r0 * k), ctypes.c_char_p(data), len(data))
        return p

    def run_steps(self, s: int, names: list[str]) -> None:
        for nm in names:
            self.steps[s][nm][1]()

    def put(self, ptr: int, data: bytes) -> None:
        self.dev.ck("cuMemcpyHtoD_v2", R.u64(ptr), ctypes.c_char_p(data), len(data))

    def weight_root(self) -> bytes:
        """B's frame-b3 row tree on the device (its domain is the weight's alone), and its root."""
        h = self.sch.commitment_hash
        self.put(self.const["dom_b"], rows_domain(weight_binding(WEIGHT), -2, self.shape[1], h).domain_id)
        self.dev.stream = None
        self.run_steps(0, self.weight_epoch)
        self.dev.sync()
        self.root_b = self.dev.read(self.pipes[0].root("b"), 32)
        return self.root_b

    def start(self, salt: bytes) -> None:
        """Under `salt`: the shared keys, the weight side, each set's tile-tree pads and level keys, and every set's buffers
        poisoned."""
        self.salt = salt
        for name, x in constant_keys(self.sch, salt, self.root_b, self.shape).items():
            self.put(self.const[name], x)
        self.dev.stream = None
        self.run_steps(0, self.weight_job)
        for s in (0, 1):
            self.poison(s)
            self.run_steps(s, self.set_unit)
        self.dev.sync()

    def poison(self, s: int) -> None:
        for name in SET_BUFS:
            self.dev.ck("cuMemsetD8_v2", R.u64(self.sets[s][name]), POISON, self.size[name])

    def keys(self, i0: int, count: int) -> None:
        """GEMMs i0 .. i0 + count's keys into their ring slots (a batch's slots are contiguous).  A copy from pageable memory
        may return before its DMA lands, and the non-blocking streams don't wait for the null stream's work: the commit
        stream waits for an event behind the copy."""
        blob = b"".join(gemm_keys(self.sch, self.salt, self.root_b, matmul(self.card, i, self.shape)) for i in range(i0, i0 + count))
        self.put(self.kring + SLOT * (i0 % (RING * BATCH)), blob)
        self.dev.ck("cuEventRecord", self.ev_keys, None)
        self.dev.ck("cuStreamWaitEvent", self.stream_c, self.ev_keys, 0)

    def enqueue(self, i: int) -> None:
        d, s, slot = self.dev, i & 1, i % (RING * BATCH)
        p = self.pipes[s]
        p.keys["dom_a"], p.keys["unit"] = self.kring + SLOT * slot, self.kring + SLOT * slot + DOM
        overlap = self.schedule == "overlap"
        if overlap and i >= 2:
            d.ck("cuStreamWaitEvent", self.stream_c, self.ev_done[(i - 2) % (RING * BATCH)], 0)
        d.stream = self.stream_c
        self.run_steps(s, self.commit)
        if overlap:
            d.ck("cuEventRecord", self.ev_commit[slot], self.stream_c)
            d.ck("cuStreamWaitEvent", self.stream_m, self.ev_commit[slot], 0)
        d.stream = self.stream_m
        self.run_steps(s, self.rest)
        d.copy_async(self.log + 64 * slot, p.root("a"), 32)
        d.copy_async(self.log + 64 * slot + 32, p.root("t"), 32)
        if overlap:
            d.ck("cuEventRecord", self.ev_done[slot], self.stream_m)
        d.stream = None

    def window(self, t_d: int, guard: int, clock, log_path: Path) -> dict:
        """GEMMs back to back until t_d − guard; every batch observed complete by t_d is credited, in order."""
        stack, leaves, late, pend, i = Stack(), [], 0, deque(), 0
        t_stop = t_d - guard
        before = self.nvml.read()
        t_first = clock()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as log:
            while clock() < t_stop:
                self.keys(i, BATCH)
                for j in range(i, i + BATCH):
                    self.enqueue(j)
                ev = self.ev_batch[(i // BATCH) % RING]
                self.dev.ck("cuEventRecord", ev, self.stream_m)
                pend.append((i, ev))
                i += BATCH
                while len(pend) > DEPTH:
                    late += self.harvest(*pend.popleft(), stack, leaves, clock, t_d, log)
            while pend:
                late += self.harvest(*pend.popleft(), stack, leaves, clock, t_d, log)
        self.leaves = leaves
        return {"count": stack.count, "root": stack.root().hex(), "launched": i, "late": late, "t_first_ns": t_first,
                "nvml_before": before, "nvml_after": self.nvml.read()}

    def harvest(self, i0: int, ev, stack: Stack, leaves: list[bytes], clock, t_d: int, log) -> int:
        self.dev.ck("cuEventSynchronize", ev)
        t = clock()
        slot = i0 % (RING * BATCH)
        data = self.dev.read(self.log + 64 * slot, 64 * BATCH)
        log.write(t.to_bytes(8, "little") + i0.to_bytes(8, "little") + data)
        if t > t_d or stack.count != i0:
            return BATCH
        for j in range(BATCH):
            e = entry(self.card, i0 + j, data[64 * j:64 * j + 32], data[64 * j + 32:64 * j + 64])
            stack.push(e)
            leaves.append(e)
        return 0

    def reopen(self, i: int, out: Path, flip: bool = False) -> dict:
        """GEMM i again into poisoned buffers on the null stream, and its trees and leaves dumped into `out`; with `flip`, the
        first message-digest byte xored between the product and the hashes (the tampered control)."""
        d, s = self.dev, i & 1
        p = self.pipes[s]
        d.sync()
        self.poison(s)
        self.keys(i, 1)
        slot = i % (RING * BATCH)
        p.keys["dom_a"], p.keys["unit"] = self.kring + SLOT * slot, self.kring + SLOT * slot + DOM
        d.stream = None
        self.run_steps(s, self.set_unit + self.commit)
        hashing = [nm for nm in self.rest if self.steps[s][nm][0] == "hashing"]
        self.run_steps(s, [nm for nm in self.rest if nm not in hashing])
        d.sync()
        if flip:
            b = d.read(self.sets[s]["digests"], 1)
            self.put(self.sets[s]["digests"], bytes([b[0] ^ 0x01]))
        self.run_steps(s, hashing)
        d.sync()
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for name in ("tree_a", "tree_t", "leaves"):
            data = d.read(self.sets[s][name], self.size[name])
            (out / f"{name}.bin").write_bytes(data)
            files[name] = {"path": f"{name}.bin", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return {"card": self.card, "i": i, "index": index_of(self.card, i), "op_set": s, "root_a": d.read(p.root("a"), 32).hex(),
                "root_t": d.read(p.root("t"), 32).hex(), "tiles": p.tiles, "flipped": flip, "files": files}

    def dump_weights(self, out: Path) -> dict:
        out.mkdir(parents=True, exist_ok=True)
        data = self.dev.read(self.pipes[0].b["tree_b"], self.size["tree_b"])
        (out / "tree_b.bin").write_bytes(data)
        return {"path": "tree_b.bin", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "root": data[-32:].hex()}


def say(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


def worker(a) -> int:
    """One card, driven by the parent over stdin and stdout, a JSON object a line each way."""
    clock = clock_from_env()
    ship = Path(os.environ.get("PEARLC_SHIP", KERNEL))
    cubin = (ship / "pearl_c_sm120.cubin").read_bytes()
    shape = tuple(a.shape)
    card = Card(a.card, os.environ.get("PEARLC_EXPECT_UUID"), cubin, shape, a.schedule, a.hashing)
    if card.refused:
        say({"op": "refused", "card": a.card, "reasons": card.refused})
        return EXIT_IDENTITY
    card.open()
    out = Path(a.out)
    root_b = card.weight_root()
    card.start(bytes(32))
    card.keys(0, 1)
    card.enqueue(0)
    card.dev.sync()
    say({"op": "ready", "card": a.card, "uuid": card.uuid, "root_b": root_b.hex(),
         "cubin_sha256": hashlib.sha256(cubin).hexdigest(), "free_total": card.dev.free_mem(), "nvml": card.nvml.read()})
    msg = json.loads(sys.stdin.readline())
    card.start(bytes.fromhex(msg["salt"]))
    t_ready = clock()
    res = card.window(msg["t_d_ns"], msg["guard_ns"], clock, out / "logs" / f"card{a.card}.bin")
    t_close = clock()
    say({"op": "close", "card": a.card, "uuid": card.uuid, "count": res["count"], "root": res["root"], "t_close_ns": t_close,
         "t_ready_ns": t_ready, **{k: v for k, v in res.items() if k not in ("count", "root")}})
    msg = json.loads(sys.stdin.readline())
    weights = card.dump_weights(out / "weights" / f"card{a.card}")
    opened, bad = [], []
    for i in sorted({d["i"] for d in msg["draws"]}):
        rec = card.reopen(i, out / "draws" / f"card{a.card}-gemm{i}")
        rec["path"] = [h.hex() for h in merkle_path(card.leaves, i)] if i < len(card.leaves) else None
        logged = card.leaves[i] if i < len(card.leaves) else None
        if logged != entry(a.card, i, bytes.fromhex(rec["root_a"]), bytes.fromhex(rec["root_t"])):
            bad.append(i)
        opened.append(rec)
        (out / "draws" / f"card{a.card}-gemm{i}" / "manifest.json").write_text(json.dumps(rec, indent=1) + "\n")
    tampered = None
    if msg.get("tamper") is not None:
        i = msg["tamper"]
        tampered = card.reopen(i, out / "controls" / "tampered", flip=True)
        (out / "controls" / "tampered" / "manifest.json").write_text(json.dumps(tampered, indent=1) + "\n")
    say({"op": "opened", "card": a.card, "weights": weights, "gemms": [r["i"] for r in opened], "mismatched": bad,
         "tampered": None if tampered is None else {k: tampered[k] for k in ("i", "root_a", "root_t")}})
    return EXIT_ROOTS if bad else 0


# ---- the parent ------------------------------------------------------------------------------------------------------

class Worker:
    def __init__(self, card: int, uuid: str, args: list[str], env: dict):
        self.card, self.uuid = card, uuid
        self.p = subprocess.Popen(args, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.q: queue.Queue = queue.Queue()
        self.ended = False
        threading.Thread(target=self.read, daemon=True).start()

    def read(self) -> None:
        for line in self.p.stdout:
            try:
                self.q.put(json.loads(line))
            except json.JSONDecodeError:
                print(f"card {self.card}: {line.rstrip()}", file=sys.stderr, flush=True)
        self.ended = True
        self.q.put({"op": "eof", "card": self.card})

    def send(self, obj: dict) -> None:
        """To the worker, unless it has exited (its eof is already queued)."""
        try:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()
        except BrokenPipeError:
            pass

    def get(self) -> dict:
        """The worker's next message; once it has exited and every message is read, eof each time."""
        if self.ended and self.q.empty():
            return {"op": "eof", "card": self.card}
        return self.q.get()


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(json.dumps(obj, indent=1) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def parent(a) -> int:
    clock = clock_from_env()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    visible = [x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if not visible or not all(x.strip().isdigit() for x in visible):
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}: a lease's indices", file=sys.stderr)
        return EXIT_USAGE
    table = ID.parse_server_md(Path(a.server_md).read_text())
    cards = [int(x) for x in visible]
    missing = [g for g in cards if g not in table]
    if missing:
        print(f"the server table has no GPU {missing}", file=sys.stderr)
        return EXIT_USAGE
    shape = [str(x) for x in a.shape]
    workers = []
    for g in cards:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CUDA_DEVICE_ORDER="PCI_BUS_ID", PEARLC_EXPECT_UUID=table[g])
        argv = [sys.executable, str(Path(__file__).resolve()), "worker", "--card", str(g), "--out", str(out), "--shape", *shape,
                "--schedule", a.schedule, "--hashing", a.hashing]
        workers.append(Worker(g, table[g], argv, env))
    ready = [w.get() for w in workers]
    bad = [r for r in ready if r.get("op") != "ready"]
    roots = {r.get("root_b") for r in ready}
    if bad or len(roots) != 1:
        write(out / "start.json", {"schema": SCHEMA, "refused": bad, "roots_b": sorted(x for x in roots if x)})
        for w in workers:
            if w.p.poll() is None:
                w.p.kill()
        print(f"not started: {bad or 'cards disagree on B root'}", file=sys.stderr)
        return EXIT_IDENTITY if bad else EXIT_ROOTS
    root_b = bytes.fromhex(roots.pop())
    uuids = [w.uuid for w in workers]
    source = os.urandom(32)
    t_s = clock()
    seed = window_seed(source, t_s, uuids)
    salt = salt_of(seed, root_b, tuple(a.shape))
    t_d = t_s + int(a.duration * 1e9)
    start = {"schema": SCHEMA, "scheme": scheme(a.hashing).name, "hashing": a.hashing,
             "shape": {"m": a.shape[0], "n": a.shape[1], "k": a.shape[2]}, "schedule": a.schedule, "batch": BATCH, "guard_s": a.guard, "duration_s": a.duration,
             "source": source.hex(), "t_s_ns": t_s, "t_s": utc(t_s), "t_d_ns": t_d, "t_d": utc(t_d), "seed": seed.hex(),
             "salt": salt.hex(), "root_b": root_b.hex(), "seed_b": SEED_B, "a_seeds": {str(g): [a_seed(g, 0), a_seed(g, 1)] for g in cards},
             "cards": [{"card": w.card, "uuid": w.uuid} for w in workers], "ready": ready}
    for w in workers:
        w.send({"op": "start", "salt": salt.hex(), "t_d_ns": t_d, "guard_ns": int(a.guard * 1e9)})
    write(out / "start.json", start)
    closes, late = [], []
    for w in workers:
        c = w.get()
        t_recv = clock()
        c["t_recv_ns"] = t_recv
        if c.get("op") == "close" and t_recv <= t_d and c["t_close_ns"] <= t_d:
            closes.append(c)
        else:
            late.append(c)
    root = window_root(salt, t_s, t_d, closes)
    t_root = clock()
    on_time = t_root <= t_d
    window = {"schema": SCHEMA, "t_s_ns": t_s, "t_d_ns": t_d, "t_root_ns": t_root, "t_root": utc(t_root), "on_time": on_time,
              "root": root.hex(), "cards": closes, "late": late}
    write(out / "window.json", window)
    print(f"window root {root.hex()[:16]} at {utc(t_root)} (deadline {utc(t_d)}): "
          + ", ".join(f"card {c['card']} {c['count']} GEMMs" for c in closes) + (f"; late {late}" if late else ""),
          file=sys.stderr, flush=True)
    source2 = os.urandom(32)
    tiles = len(scheme(a.hashing).tiles(mshape(tuple(a.shape))))
    picked = draws(source2, root, salt, [(c["card"], c["count"]) for c in closes], tiles, a.draws) if on_time else []
    tamper = picked[0] if picked else None
    for w in workers:
        mine = [d for d in picked if d["card"] == w.card]
        w.send({"op": "open", "draws": mine, "tamper": tamper["i"] if tamper and tamper["card"] == w.card else None})
    opened = [w.get() for w in workers]
    codes = [w.p.wait() for w in workers]
    write(out / "draws.json", {"schema": SCHEMA, "source": source2.hex(), "t_source_ns": clock(), "tiles_per_gemm": tiles,
                               "draws": picked, "tamper": tamper, "opened": opened, "worker_exits": codes})
    if not on_time or late:
        return EXIT_LATE
    return next((c for c in codes if c), 0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--server-md", required=True)
    r.add_argument("--duration", type=float, default=1800.0)
    r.add_argument("--guard", type=float, default=4.0)
    r.add_argument("--draws", type=int, default=60)
    r.add_argument("--schedule", choices=("serial", "overlap"), default="serial")
    r.add_argument("--hashing", choices=HASHINGS, default="h1")
    r.add_argument("--shape", type=int, nargs=3, default=list(SHAPE), metavar=("M", "N", "K"))
    w = sub.add_parser("worker")
    w.add_argument("--card", type=int, required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--schedule", choices=("serial", "overlap"), default="serial")
    w.add_argument("--hashing", choices=HASHINGS, default="h1")
    w.add_argument("--shape", type=int, nargs=3, default=list(SHAPE))
    a = p.parse_args(argv)
    return parent(a) if a.cmd == "run" else worker(a)


if __name__ == "__main__":
    sys.exit(main())
