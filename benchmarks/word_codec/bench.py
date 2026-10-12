"""The word codec's benchmark on one GPU: `wc.cu`'s libraries (`--lib NAME=PATH`, one per squaring form) gated against
`codec.py`, then timed. Writes `word-codec.json` and `bench.log` under `--out`, and in a `research run` its
`result.json`.

Nothing is timed before its gate passed, in this order:
- `gate`: device encode equals `codec.encode_word` and device decode equals `codec.decode_word` on random and edge payloads
  at every k; the fragment layouts built on the device equal `codec.plain_fragments` / `codec.fragment_payloads`;
- `read`: the device read rate R over a store of `--store-gib`, the best of a few grids;
- `decode`: for each k, the store encoded on the device (the setup time is recorded), its full decode's checksum equal to
  the payloads', then the decode timed (mode 0: read, decode, XOR);
- `rates`: compute-only throughputs: decode and encode (words/s, each step's input the previous output, the tweak
  advancing), the bare squaring (best over ILP 1-8), the GF(2^31 - 1) Horner step of the bandwidth tag;
- `gemm`: at Llama-3.1-8B's linears and batch 1-32, the fused kernel's partial sums equal the plain kernel's bit for bit
  at every k and both are within FP32 rounding of torch's; then interleaved reps of cuBLASLt FP8 (`torch._scaled_mm`), the
  plain FP8 kernel and the fused kernel at each k, each in three forms (one task a warp, `wc_gemm`; persistent with the
  next half-task's loads in flight, `wc_gemm_pipe`; warp-specialized, a producer warp filling a shared-memory ring for
  consumer warps, `wc_gemm_ws`, at each `--ws` configuration), every call on cold weights (a ring of copies past the
  L2), each arm queued behind a spin kernel so the device runs it back to back, beside an empty kernel at the GEMM's
  grid. Native is the fastest of cuBLASLt and the plain kernels. The fused GEMM runs with the fastest-decoding library
  of 56 payload bits a word (`gemm`) and with each library of another packing (`gemm_<name>`, e.g. `-DWC_PB=60`).
`--gemm-only` skips `read`, `decode` and `rates`. `--mode ncu-decode` / `ncu-gemm` run one call of each kernel for
Nsight Compute and time nothing.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parents[1]))

import codec as C  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/bench/seed").digest()
TAG_SEED = hashlib.sha256(b"verity/word-codec/bench/tag-seed").digest()
B3_LANE_COMPRESSIONS = 5  # a lane block's message, 8 + 8 L bytes, at either packing
WS_MAX_WARPS = 10         # wc.cu's: consumers, the producer and the committers
LAYERS = 32               # Llama-3.1-8B's, each with the four linears
KMAX = 8
SHAPES = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
BATCHES = (1, 2, 4, 8, 16, 32)
L2_DEFEAT_BYTES = 1 << 30


class Lib:
    def __init__(self, name: str, path: str) -> None:
        v, i, q, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint32
        self.name, self.path, self.so = name, path, ctypes.CDLL(path)
        so = self.so
        so.wc_set_key.argtypes = [v, i, q]
        so.wc_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.wc_encode.argtypes = [v, v, q, q, i, i, v]
        so.wc_read.argtypes = [v, q, v, i, v]
        so.wc_tag.argtypes = [v, q, u, v, i, v]
        so.wc_chain.argtypes = [i, i, i, q, q, v, i, v]
        so.wc_gemm.argtypes = [i, v, v, v, i, i, i, q, i, v]
        so.wc_gemm_constants.argtypes = [v]
        so.wc_gemm_pipe.argtypes = [i, v, v, v, i, i, i, q, i, i, v, v]
        so.wc_gemm_ws.argtypes = [i, v, v, v, i, i, i, q, i, i, i, i, v, v]
        so.wc_gemm_ws_tag.argtypes = [i, v, v, v, i, i, i, q, i, i, i, i, v, i, ctypes.c_char_p, q, v, i, v, q, v, v]
        so.wc_commit_bytes.argtypes, so.wc_commit_bytes.restype = [q], ctypes.c_size_t
        so.wc_commit_tags.argtypes, so.wc_commit_tags.restype = [v, q], v
        so.wc_commit_leaves_at.argtypes, so.wc_commit_leaves_at.restype = [v, q], v
        so.wc_commit_leaves.argtypes = [v, q, ctypes.c_char_p, v]
        so.wc_commit_tree.argtypes = [v, q, v, v]
        so.wc_spin.argtypes = [ctypes.c_longlong, v]
        so.wc_empty.argtypes = [i, v]
        cs = (ctypes.c_int * 8)()
        so.wc_gemm_constants(cs)
        self.constants = list(cs)
        self.bits, self.kmax = self.constants[6], self.constants[7]
        lw = C.lane_words(self.bits)
        if self.constants[:5] != [C.LANES, C.TASK_TILES, lw, lw * C.LANES, C.TASK_COLS]:
            raise RuntimeError(f"{name}: layout constants {self.constants} are not codec.py's")

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{self.name}: {fn} error {rc}")


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t) -> int:
    return t.data_ptr()


def timed(torch, fn, reps: int, warm: int = 1) -> list[float]:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return out


SPIN_CYCLES = 20_000_000


def queued(lib: Lib, torch, fn) -> tuple[float, float]:
    """fn's device time with its launches queued behind a spin (seconds), and the host's time to enqueue them."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("wc_spin", SPIN_CYCLES, stream(torch))
    a.record()
    h = time.perf_counter()
    fn()
    h = time.perf_counter() - h
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3, h


def xor_checksum(torch, m) -> int:
    """XOR over the words of m_i << (i mod 2), the decode kernel's checksum."""
    t = m.clone()
    t[1::2] <<= 1
    while t.numel() > 1:
        if t.numel() % 2:
            t = torch.cat([t, torch.zeros(1, dtype=t.dtype, device=t.device)])
        t = torch.bitwise_xor(t[0::2], t[1::2])
    return int(t.item()) & ((1 << 64) - 1)


def set_key(lib: Lib, torch, key: C.Key) -> None:
    c = torch.tensor([x for x in key.c] + [0] * (KMAX - key.rounds), dtype=torch.int64)
    lib.call("wc_set_key", ptr(c), key.rounds, key.tau)


# --- the fragment layouts on the device (codec.plain_fragments / codec.fragment_payloads) ----------------------------

def plain_layout(w):
    n, kd = w.shape
    w8 = w.view(n // 16, 2, 8, kd // C.TASK_COLS, C.TASK_TILES, 2, 4, 4)       # nb, h, g, kc, tt, hc, tq, b
    return w8.permute(0, 3, 4, 2, 6, 5, 1, 7).contiguous().view(-1)              # nb, kc, tt, (g, tq), (hc, h), b


def fragment_payloads(torch, w, bits: int = C.PAYLOAD_BITS):
    n, kd = w.shape
    lw = C.lane_words(bits)
    w8 = w.view(n // 16, 2, 8, kd // C.TASK_COLS, C.TASK_TILES, 2, 4, 4)
    s = w8.permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous().view(-1, C.LANES, C.LANE_BYTES)   # task, lane, stream
    start = [bits * i for i in range(lw)]
    assert all(b % 8 + bits <= 64 for b in start)
    pad = torch.zeros(s.shape[0], C.LANES, start[-1] // 8 + 8, dtype=torch.uint8, device=w.device)
    pad[:, :, :C.LANE_BYTES] = s
    at = torch.tensor([b // 8 for b in start], device=w.device)[:, None] + torch.arange(8, device=w.device)
    g = pad[:, :, at].to(torch.int64)                                            # task, lane, word, its 8 bytes
    m = torch.zeros(g.shape[:3], dtype=torch.int64, device=w.device)
    for j in range(8):
        m |= g[..., j] << (8 * j)
    m = m >> torch.tensor([b % 8 for b in start], device=w.device) & ((1 << bits) - 1)
    return m.transpose(1, 2).contiguous().view(-1)                               # task, word, lane


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_codec(lib: Lib, torch, n_random: int, log) -> dict:
    rng = random.Random(7)
    pb = lib.bits
    edges = [0, 1, 2, (1 << pb) - 1, (1 << (pb - 1)), 0x0123456789ABCD, (1 << pb) - 2]
    payloads = edges + [rng.getrandbits(pb) for _ in range(n_random - len(edges))]
    base = (1 << 40) + 12345
    m = torch.tensor(payloads, dtype=torch.int64, device="cuda")
    y = torch.empty_like(m)
    back = torch.empty_like(m)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = {}
    sample = list(range(len(edges))) + rng.sample(range(len(edges), n_random), 120)
    for k in range(1, lib.kmax + 1):
        key = C.key(SEED, k)
        set_key(lib, torch, key)
        lib.call("wc_encode", ptr(m), ptr(y), m.numel(), base, k, 64, stream(torch))
        lib.call("wc_decode", ptr(y), ptr(back), m.numel(), base, k, ptr(xr), 64, stream(torch))
        torch.cuda.synchronize()
        yh = y.cpu().tolist()
        bad_enc = [i for i in sample if yh[i] != C.encode_word(payloads[i], base + i, key, pb)]
        bad_dec = [i for i in sample if C.decode_word(yh[i], base + i, key, pb) != payloads[i]]
        round_trip = bool(torch.equal(back, m))
        ok = not bad_enc and not bad_dec and round_trip
        out[k] = {"checked_against_reference": len(sample), "round_trip_words": n_random, "encode_mismatch": bad_enc[:8],
                  "decode_mismatch": bad_dec[:8], "round_trip": round_trip, "ok": ok}
        log(f"gate {lib.name} k={k}: {'ok' if ok else 'FAIL'}")
    return out


def gate_layouts(torch, log) -> dict:
    rng = random.Random(8)
    n, kd = 64, 1024
    w = bytes(rng.getrandbits(8) for _ in range(n * kd))
    wt = torch.tensor(list(w), dtype=torch.uint8, device="cuda").view(n, kd)
    plain_ok = bytes(plain_layout(wt).cpu().tolist()) == C.plain_fragments(w, n, kd)
    frag_ok = {b: fragment_payloads(torch, wt, b).cpu().tolist() == C.fragment_payloads(w, n, kd, b) for b in C.PACKINGS}
    log(f"gate layouts: plain {plain_ok} fragment {frag_ok}")
    return {"plain": plain_ok, "fragment": frag_ok, "ok": plain_ok and all(frag_ok.values())}


# --- read, standalone decode, compute-only rates ---------------------------------------------------------------------

def read_rate(lib: Lib, torch, store, sms: int, reps: int) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    best = None
    for mult in (4, 8, 16, 32):
        ts = timed(torch, lambda: lib.call("wc_read", ptr(store), store.numel(), ptr(xr), sms * mult, stream(torch)), reps)
        med = statistics.median(ts)
        if best is None or med < best[1]:
            best = (mult, med, ts)
    nbytes = store.numel() * 8
    return {"bytes": nbytes, "grid_per_sm": best[0], "median_s": best[1], "times_s": best[2], "gbps": nbytes / best[1] / 1e9}


def standalone_decode(lib: Lib, torch, store_words: int, sms: int, reps: int, log) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(11)
    m = torch.randint(0, 1 << 56, (store_words,), dtype=torch.int64, device="cuda", generator=gen)
    want = xor_checksum(torch, m)
    y = torch.empty_like(m)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = {}
    for k in range(1, KMAX + 1):
        set_key(lib, torch, C.key(SEED, k))
        enc = timed(torch, lambda: lib.call("wc_encode", ptr(m), ptr(y), store_words, 0, k, sms * 6, stream(torch)), 1, warm=0)
        xr.zero_()
        lib.call("wc_decode", ptr(y), 0, store_words, 0, k, ptr(xr), sms * 16, stream(torch))
        torch.cuda.synchronize()
        got = int(xr.item()) & ((1 << 64) - 1)
        if got != want:
            out[k] = {"ok": False, "checksum": hex(got), "want": hex(want)}
            log(f"decode k={k}: checksum FAIL")
            continue
        best = None
        for mult in (8, 16, 32):
            ts = timed(torch, lambda: lib.call("wc_decode", ptr(y), 0, store_words, 0, k, ptr(xr), sms * mult,
                                               stream(torch)), reps)
            med = statistics.median(ts)
            if best is None or med < best[1]:
                best = (mult, med, ts)
        stored = store_words * 8
        out[k] = {"ok": True, "store_bytes": stored, "payload_bytes": store_words * 7, "setup_encode_s": enc[0],
                  "setup_encode_gbps_payload": store_words * 7 / enc[0] / 1e9, "grid_per_sm": best[0],
                  "median_s": best[1], "times_s": best[2], "gbps_stored": stored / best[1] / 1e9,
                  "gbps_payload": store_words * 7 / best[1] / 1e9}
        log(f"decode {lib.name} k={k}: {out[k]['gbps_stored']:.0f} GB/s stored, setup {enc[0] * 1e3:.1f} ms")
    del m, y
    torch.cuda.empty_cache()
    return out


def chain_rate(lib: Lib, torch, what: int, k: int, ilp: int, steps: int, sms: int, reps: int) -> dict:
    blocks = sms * 6
    threads = blocks * 256
    per_step = {0: 4, 1: 4, 2: ilp, 3: 8, 4: ilp}[what] * threads
    key = C.key(SEED, max(k, 1))
    tstep = key.tau * (per_step % C.P) % C.P
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    ts = timed(torch, lambda: lib.call("wc_chain", what, max(k, 1), ilp, steps, tstep, ptr(xr), blocks, stream(torch)), reps)
    med = statistics.median(ts)
    items = per_step * steps
    return {"items": items, "median_s": med, "times_s": ts, "per_s": items / med}


def rates(lib: Lib, torch, sms: int, reps: int, log) -> dict:
    set_key(lib, torch, C.key(SEED, KMAX))
    out = {"decode": {}, "encode": {}, "square": {}, "tag": None}
    for k in range(1, lib.kmax + 1):
        d = chain_rate(lib, torch, 0, k, 4, max(16, 4096 // k), sms, reps)
        e = chain_rate(lib, torch, 1, k, 4, max(2, 64 // k), sms, reps)
        out["decode"][k], out["encode"][k] = d, e
        log(f"rates {lib.name} k={k}: decode {d['per_s']:.3e} words/s, encode {e['per_s']:.3e} words/s, "
            f"r = {d['per_s'] / e['per_s']:.1f}")
    for ilp in (1, 2, 4, 8):
        s = chain_rate(lib, torch, 2, 1, ilp, 64, sms, reps)
        s["per_s"] *= 59
        s["items"] *= 59
        out["square"][ilp] = s
    log(f"rates {lib.name}: squarings/s " + ", ".join(f"ILP {i} {s['per_s']:.3e}" for i, s in out["square"].items()))
    t = chain_rate(lib, torch, 3, 1, 4, 4096, sms, reps)
    t["per_s_words"] = t["per_s"]
    t["per_s_bytes"] = t["per_s"] * 8
    out["tag"] = t
    log(f"rates {lib.name}: tag {t['per_s_bytes'] / 1e9:.0f} GB/s compute-only")
    b3 = {ilp: chain_rate(lib, torch, 4, 1, ilp, 1024, sms, reps) for ilp in (1, 2, 4)}
    best = max(b3.values(), key=lambda r: r["per_s"])
    lw = C.lane_words(lib.bits)
    out["b3"] = {"by_ilp": b3, "compressions_s": best["per_s"],
                 "stored_bytes_s": best["per_s"] * 8 * lw / B3_LANE_COMPRESSIONS}
    log(f"rates {lib.name}: ro-blake3/v1 {best['per_s']:.3e} compressions/s compute-only, "
        f"{out['b3']['stored_bytes_s'] / 1e9:.0f} GB/s of lane blocks ({B3_LANE_COMPRESSIONS} a {8 * lw}-byte block)")
    return out


def tag_memory(lib: Lib, torch, store, sms: int, reps: int) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    ts = timed(torch, lambda: lib.call("wc_tag", ptr(store), store.numel(), 0x2545F491 & 0x7FFFFFFF, ptr(xr), sms * 16,
                                       stream(torch)), reps)
    med = statistics.median(ts)
    return {"median_s": med, "times_s": ts, "gbps": store.numel() * 8 / med / 1e9}


# --- the GEMM ----------------------------------------------------------------------------------------------------------

def fp8_weights(torch, n: int, kd: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    return w


def ws_name(cfg) -> str:
    return f"c{cfg[0]}s{cfg[1]}"


class Commit:
    """A commit workspace for `blocks` tags in tagger.cu's layout: the tags (32 bytes each), then every tree level from
    the leaves (64 bytes each, SHA-512's words in native order)."""

    def __init__(self, lib: Lib, torch, blocks: int) -> None:
        self.blocks, self.leaf_count = blocks, -(-blocks // 16)
        self.buf = torch.zeros(lib.so.wc_commit_bytes(blocks), dtype=torch.uint8, device="cuda")
        self.ptr = ptr(self.buf)
        self.tags = lib.so.wc_commit_tags(self.ptr, blocks)
        self.leaves = lib.so.wc_commit_leaves_at(self.ptr, blocks)

    def tag_list(self) -> list[bytes]:
        o = self.tags - self.ptr
        raw = self.buf[o:o + 32 * self.blocks].cpu().numpy().tobytes()
        return [raw[32 * j:32 * j + 32] for j in range(self.blocks)]

    def leaf_list(self) -> list[bytes]:
        o = self.leaves - self.ptr
        raw = self.buf[o:o + 64 * self.leaf_count].cpu().numpy().view(">u8").byteswap().tobytes()
        return [raw[64 * c:64 * c + 64] for c in range(self.leaf_count)]


def rfc6962_root(nodes: list[bytes]) -> bytes:
    """The bandwidth audit's tree over the leaves (RFC 6962's shape, node SHA512(0x01 ‖ l ‖ r))."""
    if len(nodes) == 1:
        return nodes[0]
    k = 1 << ((len(nodes) - 1).bit_length() - 1)
    return hashlib.sha512(b"\x01" + rfc6962_root(nodes[:k]) + rfc6962_root(nodes[k:])).digest()


def tag_gate(lib: Lib, torch, store, cm: list[Commit], log, name: str, k: int, samples: int = 48) -> dict:
    """The fused kernel's tags (cm[0], one call) against ro-blake3/v1 on sampled blocks, the committer warps' leaves
    (cm[0]'s, from the next call) and tagger.cu's root over them against hashlib, the standalone leaves the same."""
    from verity.core.protocols.bandwidth import tags as T

    scheme = T.get("ro-blake3/v1")
    keyed = scheme.keyed(TAG_SEED)
    blocks = cm[0].blocks
    rng = random.Random(blocks * 31 + k)
    js = sorted({0, 1, 31, 32, blocks - 1, *(rng.randrange(blocks) for _ in range(samples))})
    idx = torch.tensor([i for j in js for i in C.block_indices(j, lib.bits)], dtype=torch.int64, device="cuda")
    words = store[idx].cpu().tolist()
    lw = C.lane_words(lib.bits)
    ys0, ys1 = cm[0].tag_list(), cm[1].tag_list()
    bad_tags = [j for n, j in enumerate(js) if ys0[j] != keyed(j, C.words_bytes([w & (1 << 64) - 1
                                                                                  for w in words[n * lw:(n + 1) * lw]]))]
    leaves = cm[0].leaf_list()
    want = T.leaves(scheme, TAG_SEED, ys0)
    bad_leaves = sum(a != b for a, b in zip(leaves, want)) + abs(len(leaves) - len(want))
    root = (ctypes.c_uint8 * 64)()
    lib.call("wc_commit_tree", cm[0].ptr, blocks, root, stream(torch))
    root_ok = bytes(root) == rfc6962_root(want)
    lib.call("wc_commit_leaves", cm[1].ptr, blocks, TAG_SEED, stream(torch))
    torch.cuda.synchronize()
    standalone_ok = cm[1].leaf_list() == want
    out = {"blocks": blocks, "sampled": len(js), "bad_tags": bad_tags, "second_call_tags_equal": ys0 == ys1,
           "leaves": len(leaves), "bad_leaves": bad_leaves, "root_ok": root_ok, "standalone_leaves_ok": standalone_ok}
    out["ok"] = not bad_tags and ys0 == ys1 and not bad_leaves and root_ok and standalone_ok
    log(f"tag gate {name} k={k}: {'ok' if out['ok'] else 'FAIL'} ({len(js)} sampled tags, {len(leaves)} leaves, root)")
    return out


def gemm_case(lib: Lib, torch, name: str, n: int, kd: int, batches, ks, reps: int, log, ws_cfgs=(), tag_cfgs=(),
              committers: int = 1) -> dict:
    ws_cfgs = list(ws_cfgs) + [c for c in tag_cfgs if c not in ws_cfgs]
    w = fp8_weights(torch, n, kd, 100 + n + kd)
    wb = w.view(torch.uint8)
    plain = plain_layout(wb)
    payloads = fragment_payloads(torch, wb, lib.bits)
    nwords = payloads.numel()
    stores = {}
    for k in ks:
        set_key(lib, torch, C.key(SEED, k))
        s = torch.empty_like(payloads)
        lib.call("wc_encode", ptr(payloads), ptr(s), nwords, 0, k, 188 * 6, stream(torch))
        stores[k] = s
    set_key(lib, torch, C.key(SEED, KMAX))
    torch.cuda.synchronize()
    del payloads
    xs = {}
    gx = torch.Generator(device="cuda").manual_seed(5)
    for mb in batches:
        xs[mb] = (torch.randn(mb, kd, device="cuda", generator=gx)).to(torch.float8_e4m3fn)
    nch = kd // C.TASK_COLS
    blocks = nwords // C.lane_words(lib.bits)
    res = {"N": n, "K": kd, "weight_bytes": n * kd, "store_bytes": nwords * 8, "blocks": blocks, "gate": {}, "times": {}}
    cm = [Commit(lib, torch, blocks) for _ in range(2)] if tag_cfgs else []

    def n_committers(cfg) -> int:
        c = min(committers, WS_MAX_WARPS - 1 - cfg[0])
        if c < 1:
            raise ValueError(f"no warp left for a committer beside {ws_name(cfg)}")
        return c

    def run_tag(k, wbuf, x, y, partials, cfg, slot=0, commit_from=None):
        c = n_committers(cfg) if commit_from is not None else 0
        lib.call("wc_gemm_ws_tag", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0, int(partials),
                 cfg[0], cfg[1], 0, None, 1, TAG_SEED, 0, cm[slot].tags, c, cm[commit_from].tags if c else None,
                 blocks if c else 0, cm[commit_from].leaves if c else None, stream(torch))

    def run(k, wbuf, x, y, partials, pipe=False):
        if isinstance(pipe, tuple):
            lib.call("wc_gemm_ws", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0,
                     int(partials), pipe[0], pipe[1], 0, None, stream(torch))
        elif pipe:
            lib.call("wc_gemm_pipe", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0,
                     int(partials), 0, None, stream(torch))
        else:
            lib.call("wc_gemm", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0, int(partials),
                     stream(torch))

    # gate: partial sums bit for bit (both kernels, every k), and torch's FP32 product within rounding
    for mb in batches:
        x = xs[mb]
        ref = (x.float() @ w.float().t())
        pp = torch.zeros(nch * mb * n, dtype=torch.float32, device="cuda")
        run(0, plain, x, pp, True)
        plain_sum = pp.view(nch, mb, n).sum(0)
        scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-30)
        err = float(((plain_sum - ref).abs() / scale).max())
        entry = {"plain_vs_torch_rel": err, "fused_equal_plain": {}, "pipe_equal_plain": {}, "ws_equal_plain": {}}
        forms = [(False, "fused_equal_plain"), (True, "pipe_equal_plain")] + [(c, "ws_equal_plain") for c in ws_cfgs]
        for k in [0, *ks]:
            for pipe, key in forms:
                if k == 0 and pipe is False:
                    continue
                pf = torch.zeros_like(pp)
                run(k, plain if k == 0 else stores[k], x, pf, True, pipe)
                entry[key][f"{k}_{ws_name(pipe)}" if isinstance(pipe, tuple) else k] = bool(torch.equal(pf, pp))
        for k in ks if tag_cfgs else ():
            for cfg in tag_cfgs:
                pf = torch.zeros_like(pp)
                run_tag(k, stores[k], x, pf, True, cfg)
                entry.setdefault("dt_equal_plain", {})[f"{k}_{ws_name(cfg)}"] = bool(torch.equal(pf, pp))
                if mb == batches[0]:
                    pf.zero_()
                    run_tag(k, stores[k], x, pf, True, cfg, slot=1, commit_from=0)
                    entry.setdefault("dtc_equal_plain", {})[f"{k}_{ws_name(cfg)}"] = bool(torch.equal(pf, pp))
                    torch.cuda.synchronize()
                    res.setdefault("tag_gate", {})[f"{k}_{ws_name(cfg)}"] = tag_gate(lib, torch, stores[k], cm, log,
                                                                                    name, k)
        ya = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        run(ks[-1], stores[ks[-1]], x, ya, False)
        entry["fused_atomic_vs_torch_rel"] = float(((ya - ref).abs() / scale).max())
        ya.zero_()
        run(ks[-1], stores[ks[-1]], x, ya, False, True)
        entry["pipe_atomic_vs_torch_rel"] = float(((ya - ref).abs() / scale).max())
        for cfg in ws_cfgs:
            ya.zero_()
            run(ks[-1], stores[ks[-1]], x, ya, False, cfg)
            entry.setdefault("ws_atomic_vs_torch_rel", {})[ws_name(cfg)] = float(((ya - ref).abs() / scale).max())
        try:
            yc = torch._scaled_mm(x, w.t(), torch.ones((), device="cuda"), torch.ones((), device="cuda"),
                                  out_dtype=torch.float32)
            entry["cublas_vs_torch_rel"] = float(((yc - ref).abs() / scale).max())
        except Exception as exc:  # noqa: BLE001
            entry["cublas_error"] = repr(exc)[:300]
        entry["ok"] = (err < 1e-5 and all(entry["fused_equal_plain"].values()) and all(entry["pipe_equal_plain"].values())
                       and all(entry["ws_equal_plain"].values()) and all(entry.get("dt_equal_plain", {}).values())
                       and all(entry.get("dtc_equal_plain", {}).values())
                       and all(g["ok"] for g in res.get("tag_gate", {}).values())
                       and entry["fused_atomic_vs_torch_rel"] < 1e-5 and entry["pipe_atomic_vs_torch_rel"] < 1e-5
                       and all(v < 1e-5 for v in entry.get("ws_atomic_vs_torch_rel", {}).values()))
        res["gate"][mb] = entry
        log(f"gemm gate {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (plain rel {err:.1e})")
        if not entry["ok"]:
            res["ok"] = False
            return res
    res["ok"] = True

    # rings of copies past the L2: every call reads its weights from device memory
    copies = max(2, -(-L2_DEFEAT_BYTES // (nwords * 8)))
    ring = {"plain": [plain] + [plain.clone() for _ in range(copies - 1)]}
    for k in ks:
        ring[k] = [stores[k]] + [stores[k].clone() for _ in range(copies - 1)]
    wring = [w] + [w.clone() for _ in range(copies - 1)]
    res["copies"] = copies
    spin_s = statistics.median(timed(torch, lambda: lib.call("wc_spin", SPIN_CYCLES, stream(torch)), 3))
    res["spin_s"] = spin_s
    grid = (n // 16 // lib.constants[5]) * nch
    so, st = lib.so, stream(torch)
    for mb in batches:
        x = xs[mb]
        xp = ptr(x.view(torch.uint8))
        y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        yp = ptr(y)
        one = torch.ones((), device="cuda")
        mpad = mb
        try:
            torch._scaled_mm(x, w.t(), one, one, out_dtype=torch.bfloat16)
            xc = x
        except Exception:  # noqa: BLE001  cuBLASLt's FP8 path needs M a multiple of 16 here
            mpad = 16 * -(-mb // 16)
            xc = torch.zeros(mpad, kd, device="cuda").to(torch.float8_e4m3fn)
            xc[:mb] = x
        wts = [wc_.t() for wc_ in wring]

        def arm_cublas():
            for wt in wts:
                torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)

        def arm(k):
            ps = [ptr(b) for b in ring["plain" if k == 0 else k]]

            def f():
                rc = 0
                for p in ps:
                    rc |= so.wc_gemm(k, p, xp, yp, mb, n, kd, 0, 0, st)
                if rc:
                    raise RuntimeError(f"wc_gemm k={k} M={mb}: error {rc}")
            return f

        grids = {}

        def arm_pipe(k):
            ps = [ptr(b) for b in ring["plain" if k == 0 else k]]
            launched = ctypes.c_int(0)
            lib.call("wc_gemm_pipe", k, ps[0], xp, yp, mb, n, kd, 0, 0, 0, ctypes.byref(launched), st)
            groups = launched.value // nch
            grids[k] = launched.value

            def f():
                rc = 0
                for p in ps:
                    rc |= so.wc_gemm_pipe(k, p, xp, yp, mb, n, kd, 0, 0, groups, None, st)
                if rc:
                    raise RuntimeError(f"wc_gemm_pipe k={k} M={mb}: error {rc}")
            return f

        def arm_ws(k, cfg):
            ps = [ptr(b) for b in ring["plain" if k == 0 else k]]
            launched = ctypes.c_int(0)
            lib.call("wc_gemm_ws", k, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(launched), st)
            groups = launched.value // nch
            grids[f"w{k}_{ws_name(cfg)}"] = launched.value

            def f():
                rc = 0
                for p in ps:
                    rc |= so.wc_gemm_ws(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, st)
                if rc:
                    raise RuntimeError(f"wc_gemm_ws k={k} {ws_name(cfg)} M={mb}: error {rc}")
            return f

        def arm_tag(k, cfg, decode, commit, tag=True):
            """Call i tags into cm[i mod 2]; with commit, its committers hash the leaves of cm[(i + 1) mod 2]'s tags,
            the previous call's. Without decode or tag, the kernel on the encoded store alone (the base)."""
            ps = [ptr(b) for b in ring[k]]
            c = n_committers(cfg) if commit else 0
            launched = ctypes.c_int(0)
            lib.call("wc_gemm_ws_tag", k, ps[0], xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, ctypes.byref(launched),
                     decode, TAG_SEED, 0, cm[0].tags if tag else None, c, cm[1].tags if c else None, blocks if c else 0,
                     cm[1].leaves if c else None, st)
            groups = launched.value // nch
            fn = so.wc_gemm_ws_tag

            def f():
                rc = 0
                for i, p in enumerate(ps):
                    a, b = cm[i % 2], cm[1 - i % 2]
                    rc |= fn(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], groups, None, decode, TAG_SEED, 0,
                             a.tags if tag else None, c, b.tags if c else None, blocks if c else 0,
                             b.leaves if c else None, st)
                if rc:
                    raise RuntimeError(f"wc_gemm_ws_tag k={k} {ws_name(cfg)} M={mb}: error {rc}")
            return f

        def arm_leaves():
            rc = 0
            for _ in range(copies):
                rc |= so.wc_commit_leaves(cm[0].ptr, blocks, TAG_SEED, st)
            if rc:
                raise RuntimeError(f"wc_commit_leaves: error {rc}")

        def arm_empty():
            rc = 0
            for _ in range(copies):
                rc |= so.wc_empty(grid, st)
            if rc:
                raise RuntimeError(f"wc_empty: error {rc}")

        arms = {"cublas": arm_cublas, "plain": arm(0), "plain_pipe": arm_pipe(0), "empty": arm_empty}
        for cfg in ws_cfgs:
            arms[f"plain_ws_{ws_name(cfg)}"] = arm_ws(0, cfg)
        for k in ks:
            arms[f"k{k}"] = arm(k)
            arms[f"p{k}"] = arm_pipe(k)
            for cfg in ws_cfgs:
                arms[f"w{k}_{ws_name(cfg)}"] = arm_ws(k, cfg)
            for cfg in tag_cfgs:
                arms[f"t{k}_{ws_name(cfg)}"] = arm_tag(k, cfg, 0, False)
                arms[f"dt{k}_{ws_name(cfg)}"] = arm_tag(k, cfg, 1, False)
                arms[f"dtc{k}_{ws_name(cfg)}"] = arm_tag(k, cfg, 1, True)
        for cfg in tag_cfgs:
            arms[f"base_{ws_name(cfg)}"] = arm_tag(ks[0], cfg, 0, False, tag=False)
        if tag_cfgs:
            arms["commit_leaves"] = arm_leaves
        for f in arms.values():
            f()
        torch.cuda.synchronize()
        times = {a: [] for a in arms}
        host = {a: [] for a in arms}
        names = list(arms)
        for rep in range(reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for a in order:
                t, h = queued(lib, torch, arms[a])
                times[a].append(t / copies)
                host[a].append(h)
        med = {a: statistics.median(v) for a, v in times.items()}
        host_max = max(max(v) for v in host.values())
        native = min(med[a] for a in med if a in ("cublas", "plain", "plain_pipe") or a.startswith("plain_ws_"))
        net = {a: (med[a] - med["empty"]) / (med["plain"] - med["empty"]) for a in med if a != "empty"}
        res["times"][mb] = {"median_s": med, "times_s": times, "cublas_m": mpad, "host_enqueue_max_s": host_max,
                            "host_bound": host_max > 0.8 * spin_s, "pipe_grids": grids,
                            "native_s": native, "ratio_vs_native": {a: med[a] / native for a in med},
                            "ratio_vs_plain": {a: med[a] / med["plain"] for a in med},
                            "ratio_vs_plain_net_of_empty": net,
                            "gbps_stored": {a: (nwords * 8 if re.match(r"(k|p|w|t|dt|dtc)\d|base_", a) else n * kd)
                                            / med[a] / 1e9
                                            for a in med if a not in ("empty", "commit_leaves")}}
        for k in ks:
            for cfg in tag_cfgs:
                c = ws_name(cfg)
                d, t, dt, dtc = (med[f"{a}{k}_{c}"] for a in ("w", "t", "dt", "dtc"))
                res["times"][mb].setdefault("stack", {})[f"{k}_{c}"] = {
                    "native_s": native, "base_s": med[f"base_{c}"], "decode_s": d, "tag_only_s": t, "decode_tag_s": dt,
                    "decode_tag_commit_s": dtc, "commit_leaves_standalone_s": med["commit_leaves"]}
                log(f"stack {name} M={mb} k={k} {c}: base {med[f'base_{c}'] / native:.3f}x, decode {d / native:.3f}x, "
                    f"+tag {dt / native:.3f}x, "
                    f"+leaves {dtc / native:.3f}x (tag alone {t / native:.3f}x; standalone leaves "
                    f"{med['commit_leaves'] / native:.3f} of native)")
        ws_best = {k: min((med[f"w{k}_{ws_name(c)}"], ws_name(c)) for c in ws_cfgs) for k in ks} if ws_cfgs else {}
        log(f"gemm {name} M={mb}: cublas {med['cublas'] * 1e6:.1f} us, plain {med['plain'] * 1e6:.1f} us, "
            f"plain_pipe {med['plain_pipe'] * 1e6:.1f} us, "
            + "".join(f"plain_ws_{ws_name(c)} {med[f'plain_ws_{ws_name(c)}'] * 1e6:.1f} us, " for c in ws_cfgs)
            + f"empty {med['empty'] * 1e6:.1f} us; native {native * 1e6:.1f} us; vs native "
            + ", ".join(f"k{k} {med[f'k{k}'] / native:.2f}x/{med[f'p{k}'] / native:.2f}x"
                        + (f"/ws {ws_best[k][0] / native:.2f}x {ws_best[k][1]}" if ws_cfgs else "") for k in ks)
            + (" HOST-BOUND" if host_max > 0.8 * spin_s else ""))
    del ring, wring, stores, plain, w, cm
    torch.cuda.empty_cache()
    return res


def window_blocks() -> int:
    """The tag blocks of one decode step's weights: LAYERS × the four linears (a block a lane of a task, any packing)."""
    return LAYERS * sum(n // C.TILE_ROWS * (kd // C.TASK_COLS) * C.LANES for n, kd in SHAPES.values())


def commit_window(lib: Lib, torch, reps: int, log) -> dict:
    """The commit over one decode step's tags as their own passes: the leaves (tagger.cu's b3_leaves) and the tree above
    them, over random tags."""
    blocks = window_blocks()
    cm = Commit(lib, torch, blocks)
    o = cm.tags - cm.ptr
    cm.buf[o:o + 32 * blocks] = torch.randint(0, 256, (32 * blocks,), dtype=torch.uint8, device="cuda")
    st = stream(torch)
    lv = timed(torch, lambda: lib.call("wc_commit_leaves", cm.ptr, blocks, TAG_SEED, st), reps)
    tr = timed(torch, lambda: lib.call("wc_commit_tree", cm.ptr, blocks, None, st), reps)
    out = {"blocks": blocks, "leaves": cm.leaf_count, "leaves_s": statistics.median(lv), "leaves_times_s": lv,
           "tree_s": statistics.median(tr), "tree_times_s": tr}
    log(f"commit window: {blocks} tags, {cm.leaf_count} leaves: leaves {out['leaves_s'] * 1e3:.3f} ms, tree above them "
        f"{out['tree_s'] * 1e3:.3f} ms")
    del cm
    torch.cuda.empty_cache()
    return out


def frontier_points(res: dict, libs: list[Lib]) -> dict:
    """The core plot's points at b = 61 (`rate-audit-frontier-brief.md`): for each library, k and batch, the fused step
    over the native one at the four linears, layer-weighted, as decode + tag + commit; x, the encode time of the pass's
    words in units of the step; τ, ro-blake3/v1's share of the step at the compute-only rate; the decode-free rebuild
    term (1 − τ) / x. Each shape takes the tag configuration whose full step (decode, tag, leaves) is fastest; the tree
    above the leaves is the window pass's time in proportion to the blocks."""
    cw = res.get("commit_window")
    rates_of = {l.name: res["libs"][l.name].get("rates", {}) for l in libs}
    enc_best, dec_best = {}, {}
    for r in rates_of.values():
        for best, what in ((enc_best, "encode"), (dec_best, "decode")):
            for k, e in r.get(what, {}).items():
                best[int(k)] = max(best.get(int(k), 0.0), e["per_s"])
    b3_best = max((r["b3"]["compressions_s"] for r in rates_of.values() if "b3" in r), default=None)
    points = []
    for key in [k for k in res if k == "gemm" or k.startswith("gemm_") and isinstance(res[k], dict)]:
        cases = res[key]
        lib = next(l for l in libs if key == "gemm" and l.name == res["gemm_lib"] or key == f"gemm_{l.name}")
        lw = C.lane_words(lib.bits)
        first = next(iter(cases.values()))
        for mb in first.get("times", {}):
            stacks = [g["times"][mb].get("stack", {}) for g in cases.values()]
            for k in sorted({int(s.split("_")[0]) for st in stacks for s in st}):
                tot = {"native": 0.0, "base": 0.0, "d": 0.0, "t": 0.0, "dt": 0.0, "dtc": 0.0}
                shapes, blocks = {}, 0
                for name, g in cases.items():
                    st = g["times"][mb].get("stack", {})
                    cfgs = {s: v for s, v in st.items() if int(s.split("_")[0]) == k}
                    if not cfgs:
                        break
                    cfg, v = min(cfgs.items(), key=lambda kv: kv[1]["decode_tag_commit_s"])
                    shapes[name] = {"cfg": cfg.split("_", 1)[1], **v}
                    for a, f in (("native", "native_s"), ("base", "base_s"), ("d", "decode_s"), ("t", "tag_only_s"),
                                 ("dt", "decode_tag_s"), ("dtc", "decode_tag_commit_s")):
                        tot[a] += v[f]
                    blocks += g["blocks"]
                else:
                    tree = cw["tree_s"] * blocks / cw["blocks"] if cw else None
                    nat = tot["native"]
                    fused = tot["dtc"] + (tree or 0.0)
                    pt = {"lane": "ra-word-codec", "codec": "sloth-p61", "b": 61, "field": "2^61-1", "lib": lib.name,
                          "payload_bits": lib.bits, "k": k, "batch": int(mb), "arith": "cuda-int", "measured": True,
                          "tag": "ro-blake3/v1", "per_leaf": 16, "tree_measured": tree is not None,
                          "y": fused / nat, "y_base": tot["base"] / nat, "y_decode": tot["d"] / nat,
                          "y_decode_tag": tot["dt"] / nat, "y_tag_alone": tot["t"] / nat,
                          "stack": {"base": tot["base"] / nat, "decode": (tot["d"] - tot["base"]) / nat,
                                    "tag": (tot["dt"] - tot["d"]) / nat, "commit": (fused - tot["dt"]) / nat},
                          "native_step_s": nat, "fused_step_s": fused, "tree_s": tree, "blocks": blocks,
                          "words": blocks * lw, "shapes": shapes}
                    if k in enc_best:
                        enc_s = blocks * lw / enc_best[k]
                        pt["encode_s"] = enc_s
                        pt["x_native"], pt["x_fused"] = enc_s / nat, enc_s / fused
                    if k in dec_best:
                        pt["delta_native"] = blocks * lw / dec_best[k] / nat
                    if b3_best:
                        tag_s = blocks * B3_LANE_COMPRESSIONS / b3_best
                        pt["tau_native"], pt["tau_fused"] = tag_s / nat, tag_s / fused
                    if "x_fused" in pt and "tau_fused" in pt:
                        pt["mu_rebuild_fused"] = (1 - pt["tau_fused"]) / pt["x_fused"]
                        pt["mu_rebuild_native"] = (1 - pt["tau_native"]) / pt["x_native"]
                    points.append(pt)
    shared = [{"schema": "verity/codec-frontier-point/v0", "lane": "ra-word-codec", "b": 61, "k": p["k"],
               "arithmetic": f"cuda-int32/mersenne61/{p['lib']}/pb{p['payload_bits']}/ws", "batch": p["batch"],
               "y_total": p["y"], "y_base": p["stack"]["base"], "y_decode": p["stack"]["decode"],
               "y_tag": p["stack"]["tag"], "y_commit": p["stack"]["commit"], "x": p["x_native"],
               "delta": p.get("delta_native"), "tau": p["tau_native"], "measured": True,
               "label": f"Sloth p61, {p['payload_bits']}-bit packing, warp-specialized, "
                        + ",".join(sorted({s['cfg'] for s in p['shapes'].values()})),
               "run": os.environ.get("RESEARCH_RUN_ID"), "art": None}
              for p in points if "x_native" in p and "tau_native" in p]
    return {"schema": "verity/word-codec/frontier-points/v0", "run_id": os.environ.get("RESEARCH_RUN_ID"),
            "layers": LAYERS, "shapes": {k: list(v) for k, v in SHAPES.items()}, "encode_words_s": enc_best,
            "decode_words_s": dec_best, "b3_compressions_s": b3_best, "commit_window": cw, "points": points,
            "shared": shared}


def gemm_zeroing(torch, n: int, mb: int, reps: int) -> float:
    y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
    return statistics.median(timed(torch, lambda: y.zero_(), reps))


# --- analysis ----------------------------------------------------------------------------------------------------------

def analysis(read_gbps: float, best_dec: dict, best_enc: dict, best_sq: float, tag_bytes_s: float, b_side: float,
             theta: float) -> dict:
    """ε, r, τ, σ and the bound at each k: measured (the fastest decode and encode of the libraries) and at the squaring
    floor (decode = k squarings, encode = 59 k, at the best squaring rate, no tag: every other op free).

    `free` drops the draft's (1 - μ)ε: an adversary whose weights cost nothing to regenerate and whose audit tags the
    stored words never decodes, so μ r ε + τ <= 1 + θ and the rebuild term is (1 + θ - τ) / (r ε)."""
    words_s = read_gbps * 1e9 / 8
    tau = read_gbps * 1e9 / tag_bytes_s
    out = {"read_gbps": read_gbps, "words_per_s": words_s, "tau": tau, "b_side_gbps": b_side, "theta": theta, "k": {}}
    for k in range(1, KMAX + 1):
        eps = words_s / best_dec[k]
        r = best_dec[k] / best_enc[k]
        row = {"eps": eps, "r": r, "tau": tau}
        if eps < 1 and eps + tau < 1:
            row.update(C.bound(eps, r, tau, b_side, read_gbps))
            row["with_theta"] = C.bound(eps, r, tau, b_side, read_gbps, theta)["rebuild"]
        eps_f = words_s * k / best_sq
        if eps_f < 1:
            row["floor"] = C.bound(eps_f, 59.0, 0.0, b_side, read_gbps) | {"eps": eps_f, "r": 59.0}
        enc_f = 59.0 * eps_f
        row["free"] = {"rebuild": (1 - tau) / (r * eps), "with_theta": (1 + theta - tau) / (r * eps), "r_eps": r * eps}
        row["free_floor"] = {"rebuild": 1 / enc_f, "with_theta": (1 + theta) / enc_f, "r_eps": enc_f}
        out["k"][k] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH of a wc.cu build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="all", choices=["all", "ncu-decode", "ncu-gemm"])
    ap.add_argument("--store-gib", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gate-words", type=int, default=1 << 16)
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--ks", default=",".join(map(str, range(1, KMAX + 1))))
    ap.add_argument("--b-side", type=float, default=63.0)
    ap.add_argument("--theta", type=float, default=0.21,
                    help="the tensor cores' encode capacity over the CUDA cores': MMA-bound, four squarings a "
                         "mma.m16n8k32 at 4e14 MAC/s against 1.8e12 squarings/s")
    ap.add_argument("--ws", default="2x1,2x2,4x1,4x2,8x1",
                    help="the warp-specialized GEMM's configurations, CONSUMERSxSTAGES (stages a consumer), "
                         "comma-separated ('' for none)")
    ap.add_argument("--tag-ws", default="8x1",
                    help="the warp-specialized configurations that also tag and commit ('' for none)")
    ap.add_argument("--committers", type=int, default=1, help="committer warps a CTA, at most what the warps leave")
    ap.add_argument("--gemm-only", action="store_true",
                    help="after the gates, only the compute-only rates and the GEMM, with every library: no read, "
                         "standalone decode or analysis")
    a = ap.parse_args()

    import torch
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    libs = [Lib(*spec.split("=", 1)) for spec in a.lib]
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    ks = [int(k) for k in a.ks.split(",")]
    batches = [int(b) for b in a.batches.split(",")]
    ws_cfgs = [tuple(int(v) for v in c.split("x")) for c in a.ws.split(",") if c]
    tag_cfgs = [tuple(int(v) for v in c.split("x")) for c in a.tag_ws.split(",") if c]
    store_words = int(a.store_gib * (1 << 30)) // 8 // 2 * 2

    if a.mode == "ncu-decode":
        lib = libs[0]
        m = torch.randint(0, 1 << 56, (store_words,), dtype=torch.int64, device="cuda")
        y = torch.empty_like(m)
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        lib.call("wc_read", ptr(m), store_words, ptr(xr), sms * 16, stream(torch))
        lib.call("wc_tag", ptr(m), store_words, 0x2545F491 & 0x7FFFFFFF, ptr(xr), sms * 16, stream(torch))
        for k in ks:
            set_key(lib, torch, C.key(SEED, k))
            lib.call("wc_encode", ptr(m), ptr(y), store_words, 0, k, sms * 6, stream(torch))
            lib.call("wc_decode", ptr(y), 0, store_words, 0, k, ptr(xr), sms * 16, stream(torch))
        torch.cuda.synchronize()
        return 0
    if a.mode == "ncu-gemm":
        lib = libs[0]
        set_key(lib, torch, C.key(SEED, KMAX))
        for name in a.shapes.split(","):
            n, kd = SHAPES[name]
            w = fp8_weights(torch, n, kd, 1).view(torch.uint8)
            plain = plain_layout(w)
            pay = fragment_payloads(torch, w, lib.bits)
            for mb in batches:
                x = torch.randn(mb, kd, device="cuda").to(torch.float8_e4m3fn)
                y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
                forms = [("wc_gemm", ()), ("wc_gemm_pipe", (0, None))] + [("wc_gemm_ws", (*c, 0, None)) for c in ws_cfgs]
                for fn, extra in forms:
                    lib.call(fn, 0, ptr(plain), ptr(x.view(torch.uint8)), ptr(y), mb, n, kd, 0, 0, *extra, stream(torch))
                    for k in ks:
                        lib.call(fn, k, ptr(pay), ptr(x.view(torch.uint8)), ptr(y), mb, n, kd, 0, 0, *extra,
                                 stream(torch))
        torch.cuda.synchronize()
        return 0

    res = {"device": {"name": props.name, "sms": sms, "cc": f"{props.major}.{props.minor}",
                      "memory_bytes": props.total_memory}, "seed": SEED.hex(), "libs": {}, "ok": True}
    log(f"device {props.name}, {sms} SMs")
    lay = gate_layouts(torch, log)
    res["layouts"] = lay
    res["ok"] &= lay["ok"]
    for lib in libs:
        g = gate_codec(lib, torch, a.gate_words, log)
        res["libs"][lib.name] = {"gate": g, "path": lib.path}
        res["ok"] &= all(v["ok"] for v in g.values())
    if not res["ok"]:
        log("a gate failed: nothing timed")
        (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
        return 5
    if a.gemm_only:
        gemm_lib = next((l for l in libs if l.bits == C.PAYLOAD_BITS), libs[0])
        for lib in libs:
            res["libs"][lib.name]["rates"] = rates(lib, torch, sms, a.reps, log)
    else:
        gemm_lib = measure(res, libs, torch, a, sms, store_words, log, out_dir)
    res["gemm_lib"] = gemm_lib.name
    for lib in [gemm_lib, *(l for l in libs if l is not gemm_lib and (a.gemm_only or l.bits != gemm_lib.bits))]:
        key = "gemm" if lib is gemm_lib else f"gemm_{lib.name}"
        res[key] = {}
        log(f"gemm with {lib.name} ({lib.bits} payload bits a word)")
        for name in a.shapes.split(","):
            n, kd = SHAPES[name]
            g = gemm_case(lib, torch, name, n, kd, batches, ks, a.reps, log, ws_cfgs, tag_cfgs, a.committers)
            g["zeroing_s"] = {mb: gemm_zeroing(torch, n, mb, a.reps) for mb in batches}
            g["payload_bits"] = lib.bits
            res[key][name] = g
            res["ok"] &= g.get("ok", False)
            (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
            if not res["ok"]:
                break
        if not res["ok"]:
            log("a GEMM gate failed: the rest is not timed")
            break
    if tag_cfgs and res["ok"]:
        res["commit_window"] = commit_window(gemm_lib, torch, a.reps, log)
        fr = frontier_points(res, libs)
        res["frontier"] = fr
        (out_dir / "codec-frontier-points.json").write_text(json.dumps(fr["shared"], indent=1))
        for p in fr["points"]:
            log(f"point {p['lib']} k={p['k']} M={p['batch']}: y {p['y']:.3f} (decode +{p['stack']['decode']:.3f}, tag "
                f"+{p['stack']['tag']:.3f}, commit +{p['stack']['commit']:.3f}); x_native {p.get('x_native', 0):.1f}, "
                f"tau_fused {p.get('tau_fused', 0):.3f}, rebuild {p.get('mu_rebuild_fused', float('nan')):.4f}")
    (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


def measure(res: dict, libs: list[Lib], torch, a, sms: int, store_words: int, log, out_dir: Path) -> Lib:
    """The read rate, the tag, the compute-only rates, the standalone decode and the analysis, into res; returns the
    fastest-decoding library of 56 payload bits a word, the GEMM's."""
    store = torch.randint(0, 1 << 61, (store_words,), dtype=torch.int64, device="cuda")
    res["read"] = read_rate(libs[0], torch, store, sms, a.reps)
    log(f"read {res['read']['gbps']:.0f} GB/s over {store_words * 8 / 2**30:.1f} GiB")
    for lib in libs:
        r = res["libs"][lib.name]
        r["tag_memory"] = tag_memory(lib, torch, store, sms, a.reps)
        log(f"tag (memory) {lib.name}: {r['tag_memory']['gbps']:.0f} GB/s")
    del store
    torch.cuda.empty_cache()
    for lib in libs:
        r = res["libs"][lib.name]
        r["rates"] = rates(lib, torch, sms, a.reps, log)
        r["decode"] = standalone_decode(lib, torch, store_words, sms, a.reps, log)
        res["ok"] &= all(v["ok"] for v in r["decode"].values())
        (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))

    best_dec = {k: max(res["libs"][l.name]["rates"]["decode"][k]["per_s"] for l in libs) for k in range(1, KMAX + 1)}
    best_enc = {k: max(res["libs"][l.name]["rates"]["encode"][k]["per_s"] for l in libs) for k in range(1, KMAX + 1)}
    best_sq = max(s["per_s"] for l in libs for s in res["libs"][l.name]["rates"]["square"].values())
    best_tag = max(res["libs"][l.name]["rates"]["tag"]["per_s_bytes"] for l in libs)
    res["best"] = {"decode_words_s": best_dec, "encode_words_s": best_enc, "square_s": best_sq, "tag_bytes_s": best_tag}
    res["analysis"] = analysis(res["read"]["gbps"], best_dec, best_enc, best_sq, best_tag, a.b_side, a.theta)
    for k, row in res["analysis"]["k"].items():
        log(f"k={k}: eps {row['eps']:.3f} r {row['r']:.1f} tau {row['tau']:.3f} rebuild {row.get('rebuild', float('nan')):.4f}"
            f" floor {row.get('floor', {}).get('rebuild', float('nan')):.4f} free {row['free']['rebuild']:.4f}"
            f" free-floor {row['free_floor']['rebuild']:.4f}")
    (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
    narrow = [l for l in libs if l.bits == C.PAYLOAD_BITS]
    return max(narrow, key=lambda l: res["libs"][l.name]["rates"]["decode"][KMAX]["per_s"])


def write_result(res: dict, libs: list[Lib], out_dir: Path) -> None:
    """`research/result/v0.1` in $RESEARCH_RUN_DIR (`tools/research/src/research/result.py`), when the run has one."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for k in range(1, KMAX + 1) if "analysis" in res else ():
        ms.append({"name": f"decode_k{k}", "value": max(res["libs"][l.name]["decode"][k].get("gbps_stored", 0.0)
                                                       for l in libs), "unit": "GB/s"})
        row = res["analysis"]["k"][k]
        ms += [{"name": f"r_k{k}", "value": row["r"], "unit": "1"}, {"name": f"eps_k{k}", "value": row["eps"], "unit": "1"}]
        if "rebuild" in row:
            ms.append({"name": f"rebuild_k{k}", "value": row["rebuild"], "unit": "1"})
        ms.append({"name": f"rebuild_free_k{k}", "value": row["free"]["rebuild"], "unit": "1"})
    for key in [k for k in res if k == "gemm" or k.startswith("gemm_") and isinstance(res[k], dict)]:
        for name, g in res[key].items():
            for mb, t in g.get("times", {}).items():
                for arm, ratio in t["ratio_vs_native"].items():
                    if arm != "empty":
                        ms.append({"name": f"{key}_{name}_m{mb}_{arm}_vs_native", "value": ratio, "unit": "1"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/word-codec.json", f"{rel}/bench.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "word-codec gates",
                          "detail": "device codec = codec.py, fused partial sums = plain, checksums", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


if __name__ == "__main__":
    sys.exit(main())
