"""The filler top-up beside serving phases on one GPU: how much of R a window's registered reads reach, and what the
top-up costs inference. Writes `bench.json` and `bench.log` under `--out`.

Nothing is timed before every gate passed:
- `gate`: `tu_root` over the bandwidth audit's 37-block test store under `SEED` equals `vectors.json`'s gf31-poly/v1 and
  ro-blake3/v1 roots; the top-up thread's ro-blake3/v1 receipts over a 4,096-block test store, chunk after chunk (the
  regions wrapping), equal `tu_root` over the same blocks under each chunk's seed; and `kv.cu`'s p512 encode at k = 2
  equals `kv.encode_word` on edge and random payloads.
- `read`: R, a plain read of the filler by one CTA a tile, the best of a few tile sizes.
- `alone`: the top-up thread alone over the filler in each mode (read, gf31, b3): its bytes a second on the GPU's clock,
  the idle phase's harvest.
- `phases`: a synthetic Llama-3.1-8B step (32 layers of FP8 linears, `torch._scaled_mm` on cold weights, and bf16
  attention over a cache of its own) at decode batches × contexts and prefill lengths, alone and with the top-up running,
  over each arrangement of streams (the step on the default stream at priority 0 beside a top-up at 0, or on a stream
  at the greatest priority beside a top-up at the least) and, for decode, eager and as a CUDA graph. Each cell gives the
  step's time against alone (the interference), the filler bytes the top-up read inside the cell's span on the GPU's
  clock, and the span's registered reads (the step's weights and KV, which are registered, plus the filler) over R·T.
- `encode`: the KV codec's encode at p512, k = 2 (the page conversion's cost: a page allocated from filler is encoded on
  write), alone and on a low-priority stream beside a decode step, its rate and the step's slowdown.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "benchmarks/word_codec"))

import kv as K  # noqa: E402

VECTORS = ROOT / "verity/core/protocols/bandwidth/tests/vectors.json"
GATE_SEED = bytes(range(32))
RUN_SEED = hashlib.sha256(b"verity/kv-filler/bench/seed").digest()
BLOCK = 2056
MODES = {"read": -1, "gf31": 0, "b3": 2, "sk16": 3, "fill": 4}
FILL_WARPS = {"fill": 4, "fill8": 8, "fill16": 16, "fillg": 4}  # fill-sk16/v0 at that many warps a CTA
GATED = {"fillg"}  # every SM's CTA resident, `ctas` of them let through the gate


def mode_of(name: str) -> int:
    return MODES["fill"] if name in FILL_WARPS else MODES[name]
TAG_NAMES = {0: "gf31-poly/v1", 2: "ro-blake3/v1"}
# Llama-3.1-8B: 32 layers, hidden 4,096, 32 query heads on 8 KV heads of 128, MLP 14,336.
LAYERS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, MLP = 32, 4096, 32, 8, 128, 14336
SHAPES = {"qkv": (HIDDEN + 2 * KV_HEADS * HEAD_DIM, HIDDEN), "o": (HIDDEN, HIDDEN), "gate_up": (2 * MLP, HIDDEN),
          "down": (HIDDEN, MLP)}
WEIGHT_BYTES = LAYERS * sum(n * k for n, k in SHAPES.values())
KV_TOKEN_BYTES = LAYERS * 2 * KV_HEADS * HEAD_DIM * 2  # 131,072: K and V, bf16, every layer
KV_GATE_BASES = (77, (1 << 32) - 20)
ENCODE_K = (2, 3)  # the codec's rounds: k = 3 is the stated guarantee's, k = 2 the upgrade behind the same parameter


class Topup:
    def __init__(self, path: str) -> None:
        v, i, q, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_double
        so = self.so = ctypes.CDLL(str(Path(path).resolve()))
        so.tu_start.argtypes = [i, v, v, i, i, q, q, i, i, v, i]
        so.tu_pause.argtypes = [i]
        so.tu_gate.argtypes = [i, i]
        so.tu_gate_step.argtypes = [i, v]
        so.tu_now_ms.argtypes = [v]
        so.tu_now_ms.restype = d
        so.tu_progress.argtypes = [v]
        so.tu_log.argtypes = [q, v, i]
        so.tu_root.argtypes = [i, v, q, v, v]
        so.tu_read.argtypes = [v, q, q, v, v]
        so.tu_sk_gate.argtypes = [v, q, v, v]
        so.tu_sk_root.argtypes = [v, q, v, v, v]
        so.bw_fill_test_store.argtypes = [v, q]
        so.tu_fill_units.argtypes = [v, v, i, q, q, v, v, i, v]
        so.tu_fill_signs.argtypes = [v, v]

    def fill_units(self, regions, unit0: int, units: int, seed: bytes, sign_seed: bytes, ctas: int):
        ptrs = (ctypes.c_uint64 * len(regions))(*[p for p, _ in regions])
        sizes = (ctypes.c_uint64 * len(regions))(*[n for _, n in regions])
        out = ctypes.create_string_buffer(32 * units + 4)
        rc = self.so.tu_fill_units(ptrs, sizes, len(regions), unit0, units, seed, sign_seed, ctas, out)
        if rc:
            raise RuntimeError(f"tu_fill_units error {rc}")
        raw = out.raw
        return [raw[32 * k:32 * k + 32] for k in range(units)], int.from_bytes(raw[32 * units:32 * units + 4], "little")

    def start(self, regions, mode: int, chunk: int, depth: int, priority: int, seed: bytes, tile: int = 1 << 16,
              ctas: int = 0, device: int = 0, warps: int = 4) -> None:
        if mode == MODES["fill"] and self.so.tu_fill_warps(warps):
            raise RuntimeError(f"tu_fill_warps({warps}) refused")
        ptrs = (ctypes.c_uint64 * len(regions))(*[p for p, _ in regions])
        sizes = (ctypes.c_uint64 * len(regions))(*[n for _, n in regions])
        rc = self.so.tu_start(device, ptrs, sizes, len(regions), mode, chunk, tile, depth, priority, seed, ctas)
        if rc:
            raise RuntimeError(f"tu_start error {rc}")

    def stop(self) -> None:
        rc = self.so.tu_stop()
        if rc:
            raise RuntimeError(f"tu_stop error {rc}")

    def now(self, stream: int) -> float:
        return self.so.tu_now_ms(stream)

    def log(self) -> list[tuple[float, float, float, int, int, int]]:
        """Every chunk finished: its start and end (ms on the GPU's clock since the start event), bytes, region, the first
        4 bytes of its root as a big-endian integer, and its number."""
        out, first = [], 0
        buf = (ctypes.c_double * (6 * 65536))()
        while True:
            n = self.so.tu_log(first, buf, 65536)
            out += [(buf[6 * j], buf[6 * j + 1], buf[6 * j + 2], int(buf[6 * j + 3]), int(buf[6 * j + 4]),
                     int(buf[6 * j + 5])) for j in range(n)]
            first += n
            if n < 65536:
                return out

    def root(self, mode: int, store: int, blocks: int, seed: bytes) -> bytes:
        r = ctypes.create_string_buffer(64)
        rc = (self.so.tu_sk_root(store, blocks, seed, RUN_SEED, r) if mode == 3
              else self.so.tu_root(mode, store, blocks, seed, r))
        if rc:
            raise RuntimeError(f"tu_root error {rc}")
        return r.raw


class Seq:
    """fill-sk16/v1 through `topup.cu`'s sq_* ABI; its log is `Topup.log`'s (an epoch's number where a chunk's is, its
    ring slot where a chunk's region is)."""

    def __init__(self, tu: Topup) -> None:
        v, i, q, d = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_double
        so = self.so = tu.so
        self.tu = tu
        so.sq_start.argtypes = [i, v, v, i, v, i, i, q, i, i, i]
        so.sq_idle.argtypes = [i, q]
        so.sq_step_begin.argtypes = [i, v]
        so.sq_step_end.argtypes = [v]
        so.sq_progress.argtypes = [v]
        so.sq_now_ms.argtypes = [v]
        so.sq_now_ms.restype = d

    def start(self, regions, seed: bytes, base_ctas: int, warps: int = 4, base_budget: int = 0, shift: int = 8,
              priority: int = 0, step_priority: int | None = None, device: int = 0) -> None:
        ptrs = (ctypes.c_uint64 * len(regions))(*[p for p, _ in regions])
        sizes = (ctypes.c_uint64 * len(regions))(*[n for _, n in regions])
        rc = self.so.sq_start(device, ptrs, sizes, len(regions), seed, base_ctas, warps, base_budget, shift, priority,
                              priority if step_priority is None else step_priority)
        if rc:
            raise RuntimeError(f"sq_start error {rc}")

    def stop(self) -> None:
        rc = self.so.sq_stop()
        if rc:
            raise RuntimeError(f"sq_stop error {rc}")

    def progress(self) -> tuple[int, int, int, int]:
        """Bytes and epochs logged so far, steps begun, and STEP kernels launched."""
        p = (ctypes.c_uint64 * 4)()
        self.so.sq_progress(p)
        return p[0], p[1], p[2], p[3]

    def now(self, stream: int) -> float:
        return self.so.sq_now_ms(stream)


class Codec:
    def __init__(self, path: str) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        so = self.so = ctypes.CDLL(str(Path(path).resolve()))
        so.kv_set_key.argtypes = [v, i, v]
        so.kv_encode.argtypes = [v, v, q, q, i, i, v]
        so.kv_encode_carveout.argtypes = [i]
        if so.kv_bits() != 512:
            raise RuntimeError("the codec library is not p512")
        self.field = K.field("p512")

    def set_key(self, key: K.Key) -> None:
        s = key.field.word_bytes
        c = ctypes.create_string_buffer(b"".join(v.to_bytes(s, "little") for v in key.c), s * key.rounds)
        t = ctypes.create_string_buffer(key.tau.to_bytes(s, "little"), s)
        if self.so.kv_set_key(c, key.rounds, t):
            raise RuntimeError("kv_set_key")

    def encode(self, src: int, dst: int, words: int, base: int, k: int, stream: int, grid: int = 0) -> None:
        rc = self.so.kv_encode(src, dst, words, base, k, grid, stream)
        if rc:
            raise RuntimeError(f"kv_encode error {rc}")


def overlap_bytes(records, t0: float, t1: float) -> float:
    """The filler bytes the top-up read inside [t0, t1] on the GPU's clock, a chunk's bytes taken pro rata."""
    total = 0.0
    for a, b, n, *_ in records:
        if b <= t0 or a >= t1 or b <= a:
            continue
        total += n * (min(b, t1) - max(a, t0)) / (b - a)
    return total


# --- gates -------------------------------------------------------------------------------------------------------------

def gate(tu: Topup, codec: Codec, torch, log) -> dict:
    out = {}
    pinned = {r["tag"]: r["tag_root"] for r in json.loads(VECTORS.read_text())["roots"]
              if r["blocks"] == 37 and r["seed"] == GATE_SEED.hex()}
    store = torch.empty(4096 * BLOCK, dtype=torch.uint8, device="cuda")
    if tu.so.bw_fill_test_store(store.data_ptr(), 4096):
        raise RuntimeError("bw_fill_test_store")
    for mode, name in TAG_NAMES.items():
        got = tu.root(mode, store.data_ptr(), 37, GATE_SEED).hex()
        out[f"vectors_{name}"] = got == pinned[name]
    # the thread's receipts: 2 regions of 2,048 blocks, chunks of 1,024 blocks, so chunk c covers region (c // 2) % 2
    # at block offset (c % 2) · 1,024, under the run seed with its first 8 bytes XORed with c
    half = 2048 * BLOCK
    regions = [(store.data_ptr(), half), (store.data_ptr() + half, half)]
    tu.start(regions, 2, 1024 * BLOCK, 2, 0, RUN_SEED)
    while True:
        prog = (ctypes.c_uint64 * 2)()
        tu.so.tu_progress(prog)
        if prog[1] >= 10:
            break
        time.sleep(0.01)
    tu.stop()
    recs = tu.log()
    bad = []
    for _, _, n, region, root4, c in recs:
        seed = bytearray(RUN_SEED)
        for j in range(8):
            seed[j] ^= (c >> (8 * j)) & 0xFF
        lo = ((c // 2) % 2) * half + (c % 2) * 1024 * BLOCK
        want = tu.root(2, store.data_ptr() + lo, 1024, bytes(seed))
        if int(n) != 1024 * BLOCK or region != (c // 2) % 2 or root4 != int.from_bytes(want[:4], "big"):
            bad.append(c)
    out["thread_chunks"] = len(recs)
    out["thread_receipts"] = not bad and len(recs) >= 10
    # sk16, the sketch receipt's cost proxy: device digests equal the host's over 600 blocks (two CTAs and a partial
    # warp), and the thread's chunks equal tu_sk_root's
    root = ctypes.create_string_buffer(64)
    out["sk16_digests_mismatch"] = tu.so.tu_sk_gate(store.data_ptr(), 600, RUN_SEED, root)
    tu.start(regions, 3, 1024 * BLOCK, 2, 0, RUN_SEED)
    while True:
        prog = (ctypes.c_uint64 * 2)()
        tu.so.tu_progress(prog)
        if prog[1] >= 10:
            break
        time.sleep(0.01)
    tu.stop()
    bad = []
    for _, _, n, region, root4, c in tu.log():
        seed = bytearray(RUN_SEED)
        for j in range(8):
            seed[j] ^= (c >> (8 * j)) & 0xFF
        lo = ((c // 2) % 2) * half + (c % 2) * 1024 * BLOCK
        want = tu.root(3, store.data_ptr() + lo, 1024, bytes(seed))
        if int(n) != 1024 * BLOCK or region != (c // 2) % 2 or root4 != int.from_bytes(want[:4], "big"):
            bad.append(c)
    out["sk16_thread_receipts"] = not bad and out["sk16_digests_mismatch"] == 0
    out.update(fill_gate(tu, torch))
    out.update(seq_gate(tu, torch))
    out.update(codec_gate(codec, torch, ENCODE_K))
    out["ok"] = all(v for k, v in out.items() if k not in COUNTS)
    log(f"gate: {out}")
    return out


COUNTS = ("thread_chunks", "sk16_digests_mismatch", "fill_cut_chunks", "seq_base_moved_bytes", "seq_roles_bytes")


def fill_gate(tu: Topup, torch) -> dict:
    """fill-sk16/v0 against `fill_ref`: the signs; every unit root and the check word of a chunk over three regions of
    random bytes (one ending mid-unit), at 3 CTAs and at the default grid, at each warp count; and the top-up thread's
    check words, chunk after chunk, the plan wrapping, with the gate first closed (the chunks in flight cut to the units
    taken), then down to one CTA, then open: every chunk's place in the plan, and the receipts of the first 12 and of
    the cut ones (`fill_cut_chunks` counts them)."""
    import fill_ref as F
    out = {}
    sb = ctypes.create_string_buffer(8192)
    tu.so.tu_fill_signs(RUN_SEED, sb)
    out["fill_signs"] = sb.raw == F.signs(RUN_SEED)
    store, regions, blobs = gate_regions(torch)
    seed = hashlib.sha256(b"verity/kv-filler/fill-gate").digest()
    want, wx = F.chunk(blobs, 2, 11, seed, RUN_SEED)
    ok = True
    for warps in sorted(set(FILL_WARPS.values())):
        tu.so.tu_fill_warps(warps)
        for ctas in (3, 0):
            got, gx = tu.fill_units(regions, 2, 11, seed, RUN_SEED, ctas)
            ok &= got == want and gx == wx
    tu.so.tu_fill_warps(4)
    out["fill_units"] = ok
    # the thread: chunks of 4 units over the 15-unit plan
    tu.start(regions, 4, 4 * F.UNIT_BYTES, 2, 0, RUN_SEED, ctas=3)
    for busy, idle in ((0, 0), (1, 1), (-1, -1)):  # closed: the chunks in flight return cut; then one CTA; then all
        tu.so.tu_gate(busy, idle)
        time.sleep(0.02)
    while True:
        prog = (ctypes.c_uint64 * 2)()
        tu.so.tu_progress(prog)
        if prog[1] >= 12:
            break
        time.sleep(0.01)
    tu.stop()
    recs, bad, cursor, checked, cut = tu.log(), [], 0, 0, 0
    for k, (_, _, n, unit0, root4, c) in enumerate(recs):
        planned, units = min(4, 15 - cursor), int(n) // F.UNIT_BYTES
        cut += units < planned
        if unit0 != cursor or units > planned or int(n) != units * F.UNIT_BYTES:
            bad.append(c)
        elif (k < 12 or units < planned) and checked < 64:
            checked += 1
            if root4 != F.chunk(blobs, unit0, units, F.chunk_seed(RUN_SEED, c), RUN_SEED)[1]:
                bad.append(c)
        cursor = (cursor + planned) % 15
    out["fill_thread_receipts"] = not bad and len(recs) >= 12
    out["fill_cut_chunks"] = cut
    del store
    return out


def gate_regions(torch):
    """Three 16-byte-aligned regions of random bytes, the first ending mid-unit (a 15-unit plan): the store, the regions
    and their bytes."""
    import fill_ref as F
    sizes = [5 * F.UNIT_BYTES + 4096 * 3 + 77, 7 * F.UNIT_BYTES, 3 * F.UNIT_BYTES]
    g = torch.Generator(device="cuda").manual_seed(4)
    store = torch.randint(0, 256, (sum(sizes) + 1024,), dtype=torch.uint8, device="cuda", generator=g)
    offs = [0, (sizes[0] + 15) // 16 * 16 + 51 * 16]
    offs.append(offs[1] + sizes[1])
    regions = [(store.data_ptr() + o, n) for o, n in zip(offs, sizes)]
    host = store.cpu().numpy().tobytes()
    return store, regions, [host[o:o + n] for o, n in zip(offs, sizes)]


SEQ_SHIFT_GATE = 2  # 4 units an epoch: epochs cross the 15-unit plan's passes off their boundaries


def seq_epochs_ok(recs, blobs, shift: int, check: int = 24) -> tuple[bool, int]:
    """A v1 log against `fill_ref.epoch`: epochs numbered 0, 1, ... in order, each full but the last, and the check
    words of the first `check`, of the last, and of a spread between. Returns (ok, units)."""
    import fill_ref as F
    full = (1 << shift) * F.UNIT_BYTES
    ok, units = True, 0
    for k, (_, _, n, slot, _, e) in enumerate(recs):
        ok &= e == k and slot == k % 4096 and 0 < n <= full and n % F.UNIT_BYTES == 0
        ok &= n == full or k == len(recs) - 1
        units += int(n) // F.UNIT_BYTES
    if not ok:
        return False, units
    picks = sorted(set(range(min(check, len(recs)))) | {len(recs) - 1}
                   | set(range(check, len(recs), max(1, len(recs) // 8))))
    for k in picks:
        _, _, n, _, x, e = recs[k]
        ok &= x == F.epoch(blobs, e << shift, int(n) // F.UNIT_BYTES, RUN_SEED, RUN_SEED)
    return ok, units


def seq_gate(tu: Topup, torch) -> dict:
    """fill-sk16/v1 against `fill_ref.epoch`, over `gate_regions`' 15-unit plan in epochs of 4 units, each role alone:
    BASE (3 CTAs, 5 units a launch) through the gate's settings (open, one CTA, closed: none taken, then open);
    then IDLE alone (no BASE): nothing taken while it is off, units while on; then STEP alone: nothing taken until a
    step's begin on a stream, units while the step runs (a sleep kernel), none once it has ended. Every epoch logged
    is placed and its receipt checked (`seq_epochs_ok`)."""
    sq, out = Seq(tu), {}
    store, regions, blobs = gate_regions(torch)

    def settle(epochs: int) -> int:
        for _ in range(2000):
            if sq.progress()[1] >= epochs:
                break
            time.sleep(0.001)
        return sq.progress()[0]

    sq.start(regions, RUN_SEED, base_ctas=3, base_budget=5, shift=SEQ_SHIFT_GATE)
    moved = []
    for busy, idle in ((-1, -1), (1, 1), (0, 0), (-1, -1)):
        tu.so.tu_gate(busy, idle)
        a = sq.progress()[0]
        time.sleep(0.03)
        moved.append(sq.progress()[0] - a)
    settle(sq.progress()[1] + 8)
    sq.stop()
    ok, units = seq_epochs_ok(tu.log(), blobs, SEQ_SHIFT_GATE)
    out["seq_base_receipts"] = ok and units >= 60
    # closed both ways: a unit in flight may still land after the gate moved, but no launch follows
    out["seq_base_moves"] = moved[0] > 0 and moved[1] > 0 and moved[3] > 0
    out["seq_base_moved_bytes"] = moved

    s = torch.cuda.Stream()
    sq.start(regions, RUN_SEED, base_ctas=0, shift=SEQ_SHIFT_GATE)
    time.sleep(0.03)
    still = sq.progress()[0]
    tu.so.sq_idle(1, 64)
    time.sleep(0.03)
    tu.so.sq_idle(0, 0)
    time.sleep(0.01)
    idle_bytes = sq.progress()[0] - still
    before_step = sq.progress()[0]
    time.sleep(0.02)
    quiet = sq.progress()[0] - before_step
    rc = tu.so.sq_step_begin(2, s.cuda_stream)
    with torch.cuda.stream(s):
        torch.cuda._sleep(50_000_000)
    rc |= tu.so.sq_step_end(s.cuda_stream)
    s.synchronize()
    time.sleep(0.01)
    step_bytes = sq.progress()[0] - before_step - quiet
    after = sq.progress()[0]
    time.sleep(0.02)
    tail = sq.progress()[0] - after
    sq.stop()
    ok, units = seq_epochs_ok(tu.log(), blobs, SEQ_SHIFT_GATE)
    out["seq_roles_receipts"] = ok and units > 0
    out["seq_roles_moves"] = (rc == 0 and still == 0 and idle_bytes > 0 and quiet == 0 and step_bytes > 0
                              and tail == 0)
    out["seq_roles_bytes"] = {"before_idle": still, "idle": idle_bytes, "between": quiet, "step": step_bytes,
                              "after_step": tail}
    del store
    return out


def codec_gate(codec: Codec, torch, ks) -> dict:
    """`kv.cu`'s p512 encode against `kv.encode_word` at each k of `ks`, on edge and random payloads at each base of
    `KV_GATE_BASES`."""
    f = codec.field
    rng = random.Random(512)
    edges = [0, 1, 2, f.p - 1, f.p - 2, (f.p - 1) // 2, 1 << (f.n - 1), f.c]
    payloads = edges + [rng.randrange(f.p) for _ in range(120)]
    m = torch.frombuffer(bytearray(b"".join(w.to_bytes(64, "little") for w in payloads)), dtype=torch.uint8).cuda()
    y = torch.empty_like(m)
    out = {}
    for k in ks:
        key = K.key("p512", RUN_SEED, k)
        codec.set_key(key)
        enc_bad = []
        for base in KV_GATE_BASES:
            codec.encode(m.data_ptr(), y.data_ptr(), len(payloads), base, k, torch.cuda.current_stream().cuda_stream)
            torch.cuda.synchronize()
            raw = y.cpu().numpy().tobytes()
            enc_bad += [base + i for i, v in enumerate(payloads)
                        if int.from_bytes(raw[64 * i:64 * i + 64], "little") != K.encode_word(v, base + i, key)]
        out[f"kv_encode_p512_k{k}"] = not enc_bad
    return out


# --- R and the top-up alone --------------------------------------------------------------------------------------------

def read_rate(tu: Topup, torch, filler, reps: int) -> dict:
    xr = torch.zeros(1, dtype=torch.int32, device="cuda")
    st = torch.cuda.current_stream().cuda_stream
    out = {}
    for tile in (1 << 16, 1 << 18, 1 << 20):
        ts = []
        for _ in range(reps + 1):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            tu.so.tu_read(filler.data_ptr(), filler.numel(), tile, st, xr.data_ptr())
            b.record()
            b.synchronize()
            ts.append(a.elapsed_time(b) / 1e3)
        out[str(tile)] = filler.numel() / min(ts[1:]) / 1e9
    out["R_GBps"] = max(out.values())
    return out


def alone(tu: Topup, torch, filler, seconds: float, chunk: int, depth: int, least: int, fill_chunk: int = 1 << 30,
          fill_ctas=(), fill_modes=("fill",)) -> dict:
    out = {}
    runs = [(name, mode, 0, 4) for name, mode in MODES.items()]
    runs += [(f"{f}@{c}", MODES["fill"], c, FILL_WARPS[f]) for f in fill_modes for c in fill_ctas]
    for name, mode, ctas, warps in runs:
        tu.start([(filler.data_ptr(), filler.numel())], mode, fill_chunk if mode == MODES["fill"] else chunk, depth,
                 least, RUN_SEED, ctas=ctas, warps=warps)
        time.sleep(seconds)
        tu.stop()
        recs = tu.log()[1:]  # the first chunk's start includes the thread's start
        span = (recs[-1][1] - recs[0][0]) / 1e3
        out[name] = {"GBps": sum(r[2] for r in recs) / span / 1e9, "chunks": len(recs),
                     "chunk_ms_median": statistics.median(r[1] - r[0] for r in recs)}
    sq = Seq(tu)
    for warps in sorted({FILL_WARPS[f] for f in fill_modes}):
        sq.start([(filler.data_ptr(), filler.numel())], RUN_SEED, base_ctas=0, warps=warps, priority=least)
        tu.so.sq_idle(1, 0)
        time.sleep(seconds)
        sq.stop()
        recs = tu.log()[1:]
        span = (recs[-1][1] - recs[0][1]) / 1e3
        out[f"seq-idle-w{warps}"] = {"GBps": sum(r[2] for r in recs[1:]) / span / 1e9, "epochs": len(recs)}
    return out


# --- the synthetic 8B step ---------------------------------------------------------------------------------------------

class Model:
    """Llama-3.1-8B's step: FP8 linears on cold weights (32 layers' own), bf16 attention over a KV cache of its own."""

    def __init__(self, torch) -> None:
        self.torch = torch
        g = torch.Generator(device="cuda").manual_seed(8)
        self.w = [{n: (torch.randn(r, c, device="cuda", generator=g) * 0.02).to(torch.float8_e4m3fn)
                   for n, (r, c) in SHAPES.items()} for _ in range(LAYERS)]
        self.one = torch.ones((), device="cuda", dtype=torch.float32)
        self.kv = None

    def lin(self, x, w):
        t = self.torch
        return t._scaled_mm(x.to(t.float8_e4m3fn), w.t(), scale_a=self.one, scale_b=self.one, out_dtype=t.bfloat16)

    def cache(self, batch: int, ctx: int) -> None:
        t = self.torch
        self.kv = None
        t.cuda.empty_cache()
        self.kv = t.zeros(LAYERS, 2, batch * KV_HEADS, ctx, HEAD_DIM, device="cuda", dtype=t.bfloat16)
        self.kv.normal_(0, 1)

    def mlp(self, x, w):
        t = self.torch
        gu = self.lin(x, w["gate_up"])
        return self.lin(t.nn.functional.silu(gu[:, :MLP]) * gu[:, MLP:], w["down"])

    def decode(self, x, batch: int) -> None:
        t = self.torch
        q_rows = HEADS // KV_HEADS
        for layer, w in enumerate(self.w):
            qkv = self.lin(x, w["qkv"])
            q = qkv[:batch, :HIDDEN].reshape(batch * KV_HEADS, q_rows, HEAD_DIM)
            s = t.bmm(q, self.kv[layer, 0].transpose(1, 2)).float().mul_(HEAD_DIM ** -0.5).softmax(-1)
            a = t.bmm(s.to(t.bfloat16), self.kv[layer, 1]).reshape(batch, HIDDEN)
            ao = t.zeros_like(x)
            ao[:batch] = a
            x = x + self.lin(ao, w["o"])
            x = x + self.mlp(x, w)

    def prefill(self, x) -> None:
        t = self.torch
        m = x.shape[0]
        for w in self.w:
            qkv = self.lin(x, w["qkv"])
            q = qkv[:, :HIDDEN].view(1, m, HEADS, HEAD_DIM).transpose(1, 2)
            k = qkv[:, HIDDEN:HIDDEN + KV_HEADS * HEAD_DIM].view(1, m, KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = qkv[:, HIDDEN + KV_HEADS * HEAD_DIM:].view(1, m, KV_HEADS, HEAD_DIM).transpose(1, 2)
            a = t.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + self.lin(a.transpose(1, 2).reshape(m, HIDDEN), w["o"])
            x = x + self.mlp(x, w)


def step_times(torch, fn, stream, n: int) -> list[float]:
    evs = [torch.cuda.Event(enable_timing=True) for _ in range(n + 1)]
    with torch.cuda.stream(stream):
        evs[0].record()
        for i in range(n):
            fn()
            evs[i + 1].record()
    evs[-1].synchronize()
    return [evs[i].elapsed_time(evs[i + 1]) for i in range(n)]


def cell(tu: Topup, torch, fn, stream, n: int, topup: dict | None, filler, registered: float, R: float) -> dict:
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    if topup:
        gated = topup.get("gated", False)
        tu.start([(filler.data_ptr(), filler.numel())], topup["mode"], topup["chunk"], topup["depth"],
                 topup["priority"], RUN_SEED, ctas=0 if gated else topup.get("ctas", 0), warps=topup.get("warps", 4))
        if gated:
            tu.so.tu_gate(topup["ctas"], topup["ctas"])
        time.sleep(0.05)
    t0 = tu.now(stream.cuda_stream) if topup else 0.0
    ts = step_times(torch, fn, stream, n)
    out = {"step_ms_median": statistics.median(ts), "step_ms": [round(v, 4) for v in ts]}
    if topup:
        t1 = tu.now(stream.cuda_stream)
        tu.stop()
        recs = tu.log()
        span_s = (t1 - t0) / 1e3
        filler_bytes = overlap_bytes(recs, t0, t1)
        out.update(span_s=span_s, filler_GBps=filler_bytes / span_s / 1e9,
                   registered_fraction=(registered * n + filler_bytes) / (R * 1e9 * span_s))
    else:
        out["registered_fraction"] = registered / (R * 1e9 * statistics.median(ts) / 1e3)
    return out


def phases(tu: Topup, torch, model: Model, filler, R: float, args, least: int, greatest: int, log) -> list[dict]:
    rows = []
    default = torch.cuda.default_stream()
    high = torch.cuda.Stream(priority=greatest)
    topups = [None] + [{"mode": MODES[m], "name": m, "chunk": args.chunk_mib << 20, "depth": args.depth,
                        "priority": least} for m in args.topup_modes]
    work = [("decode", b, c) for b, c in args.decode] + [("prefill", m, 0) for m in args.prefill]
    for kind, size, ctx in work:
        if kind == "decode":
            model.cache(size, ctx)
            x = torch.randn(max(16, -(-size // 16) * 16), HIDDEN, device="cuda", dtype=torch.bfloat16)
            registered = WEIGHT_BYTES + size * ctx * KV_TOKEN_BYTES
            eager = lambda x=x, b=size: model.decode(x, b)  # noqa: E731
            forms = [("eager", eager)]
            if args.graphs:
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    eager()
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    eager()
                forms.append(("graph", g.replay))
        else:
            model.kv = None
            torch.cuda.empty_cache()
            x = torch.randn(size, HIDDEN, device="cuda", dtype=torch.bfloat16)
            registered = WEIGHT_BYTES
            forms = [("eager", lambda x=x: model.prefill(x))]
        n = args.steps if kind == "decode" else max(4, args.steps // 4)
        for form, fn in forms:
            for arrangement, stream in (("p0", default), ("hi", high)):
                base = None
                for tp in topups:
                    r = cell(tu, torch, fn, stream, n, tp, filler, registered, R)
                    if tp is None:
                        base = r["step_ms_median"]
                    row = {"phase": kind, "size": size, "ctx": ctx, "form": form, "streams": arrangement,
                           "topup": tp["name"] if tp else None, "registered_GB_step": registered / 1e9, **r,
                           "slowdown": r["step_ms_median"] / base - 1}
                    rows.append(row)
                    log(f"{kind} {size}x{ctx} {form} {arrangement} topup={row['topup']}: step "
                        f"{r['step_ms_median']:.3f} ms ({row['slowdown']:+.2%}), filler "
                        f"{r.get('filler_GBps', 0):.0f} GB/s, registered {r['registered_fraction']:.3f} of R")
        if kind == "decode" and args.graphs:
            del g
    model.kv = None
    torch.cuda.empty_cache()
    return rows


def throttle(tu: Topup, torch, model: Model, filler, R: float, args, least: int, greatest: int, log) -> list[dict]:
    """The top-up held to a few CTAs (read: `ctas` long-lived CTAs; ro-blake3/v1: chunks of `ctas` × 64 blocks, a CTA
    each) beside a decode step as a CUDA graph and a prefill, both on the greatest-priority stream: harvest against
    slowdown."""
    rows = []
    high = torch.cuda.Stream(priority=greatest)
    for kind, size, ctx in [("decode", b, c) for b, c in args.throttle_decode] + [("prefill", m, 0)
                                                                                  for m in args.throttle_prefill]:
        if kind == "decode":
            model.cache(size, ctx)
            x = torch.randn(max(16, -(-size // 16) * 16), HIDDEN, device="cuda", dtype=torch.bfloat16)
            registered = WEIGHT_BYTES + size * ctx * KV_TOKEN_BYTES
            g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            with torch.cuda.stream(s):
                model.decode(x, size)
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s):
                model.decode(x, size)
            fn, n = g.replay, args.steps
        else:
            model.kv = None
            torch.cuda.empty_cache()
            x = torch.randn(size, HIDDEN, device="cuda", dtype=torch.bfloat16)
            registered = WEIGHT_BYTES
            fn, n = (lambda x=x: model.prefill(x)), max(4, args.steps // 4)
        base = cell(tu, torch, fn, high, n, None, filler, registered, R)
        rows.append({"phase": kind, "size": size, "ctx": ctx, "topup": None, "ctas": 0, **base, "slowdown": 0.0})
        for mode in args.throttle_modes:
            for ctas in args.throttle_ctas:
                tp = {"mode": mode_of(mode), "chunk": (args.fill_chunk_mib if mode in FILL_WARPS else 64) << 20,
                      "depth": 2 if mode == "read" else 3, "priority": least, "ctas": ctas,
                      "warps": FILL_WARPS.get(mode, 4), "gated": mode in GATED}
                r = cell(tu, torch, fn, high, n, tp, filler, registered, R)
                row = {"phase": kind, "size": size, "ctx": ctx, "topup": mode, "ctas": ctas, **r,
                       "slowdown": r["step_ms_median"] / base["step_ms_median"] - 1}
                rows.append(row)
                log(f"throttle {kind} {size}x{ctx} {mode} ctas={ctas}: step {r['step_ms_median']:.3f} ms "
                    f"({row['slowdown']:+.2%}), filler {r['filler_GBps']:.0f} GB/s, registered "
                    f"{r['registered_fraction']:.3f} of R (alone {base['registered_fraction']:.3f})")
        if kind == "decode":
            del g
    model.kv = None
    torch.cuda.empty_cache()
    return rows


def encode_cells(codec: Codec, torch, model: Model, args, least: int, greatest: int, log) -> dict:
    """The conversion's cost at each k of `--encode-k`: p512 encode alone, then on a least-priority stream beside a
    decode step."""
    return {f"k{k}": encode_at(codec, torch, model, args, least, greatest, log, k) for k in args.encode_k}


def encode_at(codec: Codec, torch, model: Model, args, least: int, greatest: int, log, k: int) -> dict:
    words = (args.encode_mib << 20) // 64
    src = torch.zeros(words * 64, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    codec.set_key(K.key("p512", RUN_SEED, k))
    codec.so.kv_encode_carveout(100)
    low = torch.cuda.Stream(priority=least)
    high = torch.cuda.Stream(priority=greatest)
    out = {"k": k}
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(low):
        codec.encode(src.data_ptr(), dst.data_ptr(), words, 0, k, low.cuda_stream)
        a.record()
        codec.encode(src.data_ptr(), dst.data_ptr(), words, 0, k, low.cuda_stream)
        b.record()
    b.synchronize()
    out["alone_GBps"] = words * 64 / (a.elapsed_time(b) / 1e3) / 1e9
    log(f"encode p512 k={k} alone: {out['alone_GBps']:.2f} GB/s")
    beside = []
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for batch, ctx in args.decode:
        model.cache(batch, ctx)
        x = torch.randn(max(16, -(-batch // 16) * 16), HIDDEN, device="cuda", dtype=torch.bfloat16)
        g, gs = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        with torch.cuda.stream(gs):
            model.decode(x, batch)
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=gs):
            model.decode(x, batch)
        base = statistics.median(step_times(torch, g.replay, high, args.steps))
        # the steady state: each step's B new tokens (B × 128 KiB of KV, 2,048 words a token) encoded on the low
        # stream as the step runs, at a capped grid
        tok_words = batch * KV_TOKEN_BYTES // 64
        for grid in [0] + args.encode_grids:
            evs = []
            ts = []
            for i in range(args.steps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(low):
                    e0.record()
                    codec.encode(src.data_ptr(), dst.data_ptr(), tok_words, i * tok_words, k, low.cuda_stream, grid)
                    e1.record()
                evs.append((e0, e1))
                ts += step_times(torch, g.replay, high, 1)
            torch.cuda.synchronize()
            enc = [a.elapsed_time(b) for a, b in evs]
            r = {"k": k, "batch": batch, "ctx": ctx, "form": "per-step tokens", "grid": grid or "a thread a word",
                 "step_ms_alone": base, "step_ms_beside": statistics.median(ts),
                 "slowdown": statistics.median(ts) / base - 1, "encode_ms_median": statistics.median(enc),
                 "encode_lag_steps": statistics.median(enc) / statistics.median(ts)}
            beside.append(r)
            log(f"encode k={k} per step {batch}x{ctx} grid={r['grid']}: step {base:.3f} -> {r['step_ms_beside']:.3f} ms "
                f"({r['slowdown']:+.2%}), the step's {tok_words * 64 / 1e6:.1f} MB encoded in "
                f"{r['encode_ms_median']:.2f} ms")
        # a backlog (a prefill burst's KV) drained beside decode: 8 MiB launches at each capped grid
        per = (8 << 20) // 64
        for grid in args.encode_grids + [sms * 8]:
            launches = 64
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(low):
                e0.record()
                for i in range(launches):
                    off = (i * per) % max(per, words - per)
                    codec.encode(src.data_ptr() + off * 64, dst.data_ptr() + off * 64, per, off, k, low.cuda_stream,
                                 grid)
                e1.record()
            ts = []
            while not e1.query():
                ts += step_times(torch, g.replay, high, 1)
            e1.synchronize()
            enc_s = e0.elapsed_time(e1) / 1e3
            r = {"k": k, "batch": batch, "ctx": ctx, "form": "backlog", "grid": grid, "step_ms_alone": base,
                 "step_ms_beside": statistics.median(ts) if ts else None, "steps_beside": len(ts),
                 "slowdown": statistics.median(ts) / base - 1 if ts else None,
                 "encode_GBps": launches * per * 64 / enc_s / 1e9}
            beside.append(r)
            log(f"encode k={k} backlog beside decode {batch}x{ctx} grid={grid}: {r['encode_GBps']:.2f} GB/s, step "
                f"{base:.3f} -> {r['step_ms_beside'] or 0:.3f} ms ({(r['slowdown'] or 0):+.2%}, {len(ts)} steps)")
        del g
    out["beside_decode"] = beside
    model.kv = None
    torch.cuda.empty_cache()
    # a prefill's own KV (m tokens, m × 128 KiB) encoded on the low stream as the prefill runs, at a thread a word
    pre = []
    for m in args.prefill:
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        fn = lambda x=x: model.prefill(x)  # noqa: E731
        base = statistics.median(step_times(torch, fn, high, max(4, args.steps // 4)))
        tok_words = min(words, m * KV_TOKEN_BYTES // 64)
        evs, ts = [], []
        for i in range(max(4, args.steps // 4)):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(low):
                e0.record()
                codec.encode(src.data_ptr(), dst.data_ptr(), tok_words, i * tok_words, k, low.cuda_stream)
                e1.record()
            evs.append((e0, e1))
            ts += step_times(torch, fn, high, 1)
        torch.cuda.synchronize()
        enc = statistics.median(a.elapsed_time(b) for a, b in evs)
        r = {"k": k, "prefill": m, "step_ms_alone": base, "step_ms_beside": statistics.median(ts),
             "slowdown": statistics.median(ts) / base - 1, "encode_ms_median": enc,
             "encode_GBps": tok_words * 64 / (enc / 1e3) / 1e9, "encode_lag_steps": enc / statistics.median(ts)}
        pre.append(r)
        log(f"encode k={k} beside prefill {m}: step {base:.3f} -> {r['step_ms_beside']:.3f} ms ({r['slowdown']:+.2%}), "
            f"its {tok_words * 64 / 1e6:.0f} MB encoded in {enc:.2f} ms ({r['encode_GBps']:.2f} GB/s)")
    out["beside_prefill"] = pre
    return out


def pairs(text: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in p.split("x")) for p in text.split(",") if p]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topup-lib", required=True)
    ap.add_argument("--kv-lib", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--filler-gib", type=int, default=16)
    ap.add_argument("--chunk-mib", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--alone-s", type=float, default=2.0)
    ap.add_argument("--decode", type=pairs, default=pairs("8x4096,64x2048"))
    ap.add_argument("--prefill", type=lambda s: [int(v) for v in s.split(",") if v], default=[1024, 4096])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--topup-modes", type=lambda s: s.split(","), default=["read", "b3", "sk16"])
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--encode-mib", type=int, default=512)
    ap.add_argument("--encode-k", type=lambda s: [int(v) for v in s.split(",") if v], default=list(ENCODE_K))
    ap.add_argument("--encode-grids", type=lambda s: [int(v) for v in s.split(",") if v], default=[16, 32, 64, 128])
    ap.add_argument("--throttle-decode", type=pairs, default=pairs("64x2048"))
    ap.add_argument("--throttle-prefill", type=lambda s: [int(v) for v in s.split(",") if v], default=[4096])
    ap.add_argument("--throttle-modes", type=lambda s: s.split(","), default=["read", "b3", "sk16"])
    ap.add_argument("--throttle-ctas", type=lambda s: [int(v) for v in s.split(",") if v], default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--fill-chunk-mib", type=int, default=1024)
    ap.add_argument("--skip", type=lambda s: set(s.split(",")), default=set())
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = (a.out / "bench.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    least, greatest = torch.cuda.Stream.priority_range()
    if least < greatest:
        least, greatest = greatest, least
    tu, codec = Topup(a.topup_lib), Codec(a.kv_lib)
    res: dict = {"format": "verity-kv-filler/bench/v0", "args": {k: (sorted(v) if isinstance(v, set) else v)
                                                                  for k, v in vars(a).items() if k != "out"},
                 "device": torch.cuda.get_device_name(), "torch": torch.__version__,
                 "stream_priorities": {"least": least, "greatest": greatest}, "weight_bytes": WEIGHT_BYTES}
    res["out"] = str(a.out)
    res["gate"] = gate(tu, codec, torch, log)
    if not res["gate"]["ok"]:
        (a.out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")
        log("gate failed; nothing timed")
        return 2
    filler = torch.zeros(a.filler_gib << 30, dtype=torch.uint8, device="cuda")
    res["read"] = read_rate(tu, torch, filler, 4)
    R = res["read"]["R_GBps"]
    log(f"R = {R:.0f} GB/s ({res['read']})")
    res["alone"] = alone(tu, torch, filler, a.alone_s, a.chunk_mib << 20, a.depth, least, a.fill_chunk_mib << 20,
                         a.throttle_ctas, [m for m in a.throttle_modes if m in FILL_WARPS])
    log(f"top-up alone: {res['alone']}")

    def save() -> None:
        (a.out / "bench.json").write_text(json.dumps(res, indent=1) + "\n")

    save()
    model = Model(torch)
    if "phases" not in a.skip:
        res["phases"] = phases(tu, torch, model, filler, R, a, least, greatest, log)
        save()
    if "throttle" not in a.skip:
        res["throttle"] = throttle(tu, torch, model, filler, R, a, least, greatest, log)
        save()
    if "encode" not in a.skip:
        res["encode"] = encode_cells(codec, torch, model, a, least, greatest, log)
        save()
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
