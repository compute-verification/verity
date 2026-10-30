"""The wide word codecs' benchmark on one GPU: `wide.cu`'s libraries (`--lib NAME=PATH`, one per field and squaring
form) gated against `wide.py`, then timed. Writes `wide.json`, `points.json` (the frontier's points, `frontier.py`'s
schema) and `bench.log` under `--out`, and, with `RESEARCH_RUN_DIR` set, its `result.json`.

Nothing is timed before its gate passed, in this order:
- `layouts`: the store built on the device from FP8 weights equals `wide.store_bytes(wide.fragment_payloads(…))`;
- `codec`: for each library and k, device encode equals `wide.encode_word` on random and edge payloads, device decode
  inverts it on every word, and the device's k inverse rounds of random canonical words equal `wide.decode_raw`;
- `read`: R, the device read rate over `--store-gib`, the best of a few grids;
- `rates`: compute-only throughputs, each step's input the previous step's output: the squaring (ILP 1, 2, 4), encode
  (roots/s, ILP 1, 2), decode at each k and the tag's BLAKE3 compression (ILP 1, 2, 4); x, Δ and τ follow;
- `gemm`: at the linears of `--model` (Llama-3.1-8B, the default, or 70B) and batch 1–32, for each library and k: the fused kernel's partial sums equal the
  payload store's (k = 0) bit for bit, without and with the tag; the payload store's sum is within FP32 rounding of
  torch's; every tag of the matrix was written (the tag area poisoned with 0xA5 first, and a run without the tag over
  the poison is caught); a sample of tags equals `ro-blake3/v1` over the encoded words; the commit's root equals
  `MerkleTree(tags.leaves(…))` over the device's tags. Then interleaved reps of cuBLASLt FP8 (`torch._scaled_mm`),
  `wc.cu`'s plain FP8 kernels (`--wc`), the wide kernel on the payload store (`ww0`), decode only (`d<k>`), decode and
  tag (`t<k>`; `t<k>_ahead` with the L2 prefetch of the next wave's records, a wave being `ww_gemm_occupancy` × the
  SMs), tag only (`t0`) and the commit, every call on cold weights (a ring of copies past the L2), each arm queued
  behind a spin kernel. Native is the fastest of cuBLASLt and the plain kernels. A KW = 1 library also gates and times
  the sketch-then-hash tag (`wide_sketch`, ro-sketch-blake3/v0) in place of ro-blake3/v1: its partial sums equal the
  payload store's, sampled tiles' sketches equal `wide_sketch`'s over their records, sampled tags its tags, the tag
  kernel apart the fused hash on every vector, and the root `wide_sketch.root`; arms `s<k>` (the sketch written out),
  `h<k>` (hashed in the kernel) and `sx<k>` (written, then `ww_sketch_tags` after each call).

The frontier's quantities, per field and k (`wide.json` "frontier"):
  x        k · (L / 256) · R / (encode roots/s at k = 1): the encode time of the whole store per native pass at R, from the
           best encode of the libraries (the adversary's)
  Δ        (L / 256) · R / (decode words/s at k): decode's share, compute only
  τ        (TAG_BLOCKS / TB / 256) · R / (BLAKE3 compressions/s): the tag's share, compute only (a WW_TB = 2 library's
           tag covers two records in TAG_BLOCKS = 9 compressions)
  τ_sketch 2 R / (n · BLAKE3 compressions/s): the sketch's hash (two compressions a vector of n stored bytes), its
           int8 MMAs left out (a smaller τ, a larger μ)
  μ        (1 − τ) / x, the rebuild bound
  y        layer-weighted fused time over native: total = t<k> / native + commit, decode = (d<k> − ww0) / native,
           tag = (t<k> − d<k>) / native; the rest (ww0 / native) is the layout's base. The commit is the window's, once
           a pass: the leaves and tree over the pass's tags (32 layers of the four linears, `pass_commit`) over 32
           times the four linears' native time; `y_commit_linear` is a commit launched per linear, for comparison.
           A ".../commit-overlapped" point takes the commit as its excess beside the next pass instead (`pass_overlap`:
           a pass of fused GEMMs alone and with the commit on a second stream, at batch 1 and 32)
No point gives the linears different k: the adversary rebuilds the smallest k's words, so such a point has uniform
k_min's x and a larger y (red team F21).
The SM clock is sampled during a squaring chain for each library (`clock`, and each point's `sm_mhz`).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as B  # noqa: E402
import cuda_lib  # noqa: E402
import wide as W  # noqa: E402
import wide_sketch as WS  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/wide/bench/seed").digest()
TAG_SEED = hashlib.sha256(b"verity/word-codec/wide/bench/tag-seed").digest()
POINT_SCHEMA = "verity/codec-frontier-point/v0"
POISON = 0xA5A5A5A5
DEV = "cuda"
LAYERS = 32
# (layers, the four linears as (out, in)): `--model` sets LAYERS and bench's SHAPES from these
MODELS = {
    "llama-3.1-8b": (32, {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}),
    "llama-3.1-70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192),
                           "down": (8192, 28672)}),
}
MODEL = "llama-3.1-8b"


def use_model(name: str) -> None:
    global LAYERS, MODEL
    LAYERS, shapes = MODELS[name]
    MODEL = name
    B.SHAPES.clear()
    B.SHAPES.update(shapes)


_v, _i, _q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
WW = {   # wide.cu's functions (`cuda_lib.load`)
    "ww_constants": [_v],
    "ww_set_key": [_v, _i, _v, _v, _v, _v],
    "ww_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "ww_encode": [_v, _v, _q, _q, _i, _i, _v],
    "ww_chain": [_i, _i, _i, _q, _v, _i, _v],
    "ww_read": [_v, _q, _v, _i, _v],
    "ww_spin": [ctypes.c_longlong, _v],
    "ww_gemm": [_i, _v, _v, _v, _i, _i, _i, _q, _i, _v, _v, _i, _v],
    "ww_gemm_occupancy": [_i, _i, _i, _i],
    "ww_workspace_bytes": ([_q], ctypes.c_size_t),
    "ww_workspace_tags": ([_v, _q], _v),
    "ww_commit": [_v, _q, _v, _v, _v],
    "ww_gemm_sketch": [_i, _v, _v, _v, _i, _i, _i, _q, _i, _v, _v, _q, _v, _v, _i, _v],
    "ww_sketch_tags": [_v, _q, _q, _v, _v, _v],
    "ww_sketch_commit": [_v, _q, _v, _v, _v],
    "ww_sketch_leaves": [_v, _q, _v, _q, _q, _v, _v],
    "ww_sketch_fold": [_v, _q, _q, _q, _v],
    "ww_sketch_tree": [_v, _q, _i, _v, _v],
}


class WLib:
    def __init__(self, name: str, path: str) -> None:
        self.name, self.path, self.so = name, path, cuda_lib.load(path, WW)
        so = self.so
        cs = (ctypes.c_int * 21)()
        so.ww_constants(cs)
        (self.bits, self.limbs, self.payload, self.lane_words, lanes, tiles, self.gemm_warps, self.kmax, self.sqr,
         self.msg_blocks, self.roll, self.occ, self.cc, self.b3u, self.tb, self.kw, self.sk_m, self.skend,
         self.sub, self.maxreg, self.sku) = list(cs)
        self.field = W.field(self.bits)
        f = self.field
        if (self.limbs, self.payload, self.lane_words, lanes, tiles, self.sk_m) != (
                f.limbs, f.payload, f.lane_words(), W.LANES, W.TASK_TILES, WS.steps(f)):
            raise RuntimeError(f"{name}: constants {list(cs)} are not wide.py's")
        self.sketch = self.kw == 1 and self.gemm_warps == WS.T

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{self.name}: {fn} error {rc}")

    def set_key(self, key: W.Key) -> None:
        f, n = self.field, self.limbs
        u32 = ctypes.c_uint32

        def arr(vals):
            a = (u32 * len(vals))()
            a[:] = vals
            return a

        c = arr([x for cj in key.c for x in W.limbs(cj, f)])
        tau = arr(W.limbs(key.tau, f))
        if f.three_mod_four:
            self.call("ww_set_key", c, key.rounds, tau, None, None, None)
        else:
            i_, c1 = W.SQRT_M1, W.C1
            self.call("ww_set_key", c, key.rounds, tau, arr(W.limbs(i_, f)), arr(W.limbs(c1, f)),
                      arr(W.limbs(c1 * i_ % f.p, f)))
        assert n == f.limbs


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t) -> int:
    return t.data_ptr()


def queued(lib: WLib, torch, fn) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("ww_spin", B.SPIN_CYCLES, stream(torch))
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3


# --- the store: flat word i's limb e at u32 index 32·N·⌊i/32⌋ + 32·e + (i mod 32) ------------------------------------

def limb_index(i: int, e: int, n: int) -> int:
    return 32 * n * (i >> 5) + 32 * e + (i & 31)


def flat_store(torch, words: list[int], f: W.Field):
    assert len(words) % 32 == 0
    out = [0] * (len(words) * f.limbs)
    for i, y in enumerate(words):
        for e, x in enumerate(W.limbs(y, f)):
            out[limb_index(i, e, f.limbs)] = x - (1 << 32) if x >> 31 else x
    return torch.tensor(out, dtype=torch.int32, device=DEV)


def flat_words(t, f: W.Field) -> list[int]:
    a = [x & 0xFFFFFFFF for x in t.cpu().tolist()]
    n = len(a) // f.limbs
    return [sum(a[limb_index(i, e, f.limbs)] << (32 * e) for e in range(f.limbs)) for i in range(n)]


def payload_store(torch, wb, f: W.Field):
    """The payload store of FP8 weights wb (n × kd uint8 on the device): `wide.store_bytes(wide.fragment_payloads(…))`
    as int32."""
    n, kd = wb.shape
    lw, nl, pb = f.lane_words(), f.limbs, f.payload
    w8 = wb.view(n // 16, 2, 8, kd // W.TASK_COLS, W.TASK_TILES, 2, 4, 4)
    s = w8.permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous().view(-1, W.LANES, W.LANE_BYTES)   # task, lane, stream
    tasks = s.shape[0]
    offs = [w * pb + 32 * e for w in range(lw) for e in range(nl)]
    lens = [max(0, min(32, pb - 32 * e)) for w in range(lw) for e in range(nl)]
    pad = torch.zeros(tasks, W.LANES, max(offs) // 8 + 8, dtype=torch.uint8, device=wb.device)
    pad[:, :, :W.LANE_BYTES] = s
    idx = torch.tensor([o // 8 for o in offs], device=wb.device)[:, None] + torch.arange(5, device=wb.device)
    out = torch.empty(tasks, W.LANES, lw * nl, dtype=torch.int64, device=wb.device)
    step = max(1, (1 << 27) // (W.LANES * lw * nl * 5))
    sh = torch.tensor([o % 8 for o in offs], device=wb.device)
    mask = torch.tensor([(1 << ln) - 1 for ln in lens], device=wb.device)
    for t0 in range(0, tasks, step):
        g = pad[t0:t0 + step][:, :, idx].to(torch.int64)
        v = g[..., 0] | g[..., 1] << 8 | g[..., 2] << 16 | g[..., 3] << 24 | g[..., 4] << 32
        out[t0:t0 + step] = v >> sh & mask
    out = torch.where(out >= 1 << 31, out - (1 << 32), out).to(torch.int32)
    return out.view(tasks, W.LANES, lw, nl).permute(0, 2, 3, 1).contiguous().view(-1)


def record_block(store, f: W.Field, j: int, tb: int = 1) -> bytes:
    """E_j from the device store: lane j mod 32's L words of tasks tb·(j div 32) … tb·(j div 32) + tb − 1."""
    lw, nl = f.lane_words(), f.limbs
    q, lane = divmod(j, W.LANES)
    out = b""
    for t in range(tb * q, tb * q + tb):
        rec = store[t * W.LANES * lw * nl:(t + 1) * W.LANES * lw * nl].view(lw, nl, W.LANES)[:, :, lane]
        out += b"".join((x & 0xFFFFFFFF).to_bytes(4, "little") for x in rec.flatten().cpu().tolist())
    return out


def tag_bytes(tags_u32) -> list[bytes]:
    raw = tags_u32.cpu().numpy().tobytes()
    return [raw[32 * j:32 * j + 32] for j in range(len(raw) // 32)]


def sign_tensor(torch, f: W.Field):
    """The window's sign matrix as the fused GEMM's B operands (`wide_sketch.sign_fragments`), on the device."""
    frag = WS.sign_fragments(WS.signs(TAG_SEED, f), WS.steps(f))
    return torch.from_numpy(frag.copy()).to(DEV)


def tile_sketch(store, f: W.Field, n: int, kd: int, t: int, s):
    """`wide_sketch`'s sketch of tile t of the device store (16 × 16 int64), from its four records alone."""
    nch, rec = kd // W.TASK_COLS, W.LANES * f.lane_words() * f.limbs
    G, kc = divmod(t, nch)
    parts = []
    for i in range(WS.T):
        nb = WS.T * G + i
        if nb < n // 16:
            q = nb * nch + kc
            parts.append(store[q * rec:(q + 1) * rec].cpu().numpy().tobytes())
        else:
            parts.append(bytes(4 * rec))
    return WS.sketch(WS.vectors(b"".join(parts), 16 * WS.T, W.TASK_COLS, f), s)


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_layouts(torch, libs: list[WLib], log) -> dict:
    rng = random.Random(8)
    n, kd = 64, 1024
    w = bytes(rng.getrandbits(8) for _ in range(n * kd))
    wt = torch.tensor(list(w), dtype=torch.uint8, device=DEV).view(n, kd)
    out = {}
    for f in {lib.bits: lib.field for lib in libs}.values():
        dev = payload_store(torch, wt, f).cpu().numpy().tobytes()
        out[f.name] = dev == W.store_bytes(W.fragment_payloads(w, n, kd, f), f)
    log(f"gate layouts: {out}")
    return {"fields": out, "ok": all(out.values())}


def gate_codec(lib: WLib, torch, n_words: int, ks: list[int], log) -> dict:
    f = lib.field
    rng = random.Random(7 + lib.bits)
    edges = [0, 1, 2, f.payload_bound() - 1, 1 << (f.payload - 1), (1 << 32) - 1, 1 << 32, (1 << 64) + 5]
    ms = edges + [rng.getrandbits(f.payload) % f.payload_bound() for _ in range(n_words - len(edges))]
    ys_raw = [0, 1, f.p - 1, f.p - 2, (f.p - 1) // 2] + [rng.randrange(f.p) for _ in range(n_words - 5)]
    base = (1 << 40) + 96
    m = flat_store(torch, ms, f)
    yr = flat_store(torch, ys_raw, f)
    y, back, dr = torch.empty_like(m), torch.empty_like(m), torch.empty_like(m)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    sample = list(range(len(edges))) + rng.sample(range(len(edges), n_words), 24)
    out = {}
    for k in ks:
        key = W.key(f, SEED, k)
        lib.set_key(key)
        lib.call("ww_encode", ptr(m), ptr(y), n_words, base, k, 64, stream(torch))
        lib.call("ww_decode", ptr(y), ptr(back), n_words, base, k, ptr(xr), 64, stream(torch))
        lib.call("ww_decode", ptr(yr), ptr(dr), n_words, base, k, ptr(xr), 64, stream(torch))
        torch.cuda.synchronize()
        yh, drh = flat_words(y, f), flat_words(dr, f)
        bad_enc = [i for i in sample if yh[i] != W.encode_word(ms[i], base + i, key)]
        bad_raw = [i for i in sample[:5] + sample[-12:] if drh[i] != W.decode_raw(ys_raw[i], base + i, key)]
        round_trip = bool(torch.equal(back, m))
        ok = not bad_enc and not bad_raw and round_trip
        out[k] = {"checked_against_reference": len(sample), "round_trip_words": n_words, "encode_mismatch": bad_enc[:8],
                  "decode_raw_mismatch": bad_raw[:8], "round_trip": round_trip, "ok": ok}
        log(f"gate {lib.name} k={k}: {'ok' if ok else 'FAIL'} {'' if ok else out[k]}")
    return out


# --- read and the compute-only rates ---------------------------------------------------------------------------------

def read_rate(lib: WLib, torch, store_bytes: int, sms: int, reps: int) -> dict:
    buf = torch.randint(-(1 << 31), 1 << 31, (store_bytes // 4,), dtype=torch.int32, device="cuda")
    xr = torch.zeros(1, dtype=torch.int32, device="cuda")
    best = None
    for mult in (4, 8, 16, 32):
        ts = B.timed(torch, lambda: lib.call("ww_read", ptr(buf), buf.numel() // 4, ptr(xr), sms * mult, stream(torch)),
                     reps)
        med = statistics.median(ts)
        if best is None or med < best[1]:
            best = (mult, med, ts)
    del buf
    torch.cuda.empty_cache()
    return {"bytes": store_bytes, "grid_per_sm": best[0], "median_s": best[1], "times_s": best[2],
            "gbps": store_bytes / best[1] / 1e9}


def chain_rate(lib: WLib, torch, what: int, k: int, ilp: int, sms: int, reps: int, target_s: float = 0.08) -> dict:
    best = None
    for mult in (4, 8):
        grid = sms * mult
        per_step = grid * 256 * ilp
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        probe = B.timed(torch, lambda: lib.call("ww_chain", what, k, ilp, 1, ptr(xr), grid, stream(torch)), 1)[0]
        steps = max(1, min(1 << 20, int(target_s / max(probe, 1e-6))))
        ts = B.timed(torch, lambda: lib.call("ww_chain", what, k, ilp, steps, ptr(xr), grid, stream(torch)), reps)
        med = statistics.median(ts)
        r = {"grid_per_sm": mult, "steps": steps, "items": per_step * steps, "median_s": med, "times_s": ts,
             "per_s": per_step * steps / med}
        if best is None or r["per_s"] > best["per_s"]:
            best = r
    return best


def rates(lib: WLib, torch, ks: list[int], sms: int, reps: int, log) -> dict:
    lib.set_key(W.key(lib.field, SEED, max(ks)))
    out = {"square": {}, "encode": {}, "decode": {}, "tag": {}}
    for u in (1, 2, 4):
        out["square"][u] = chain_rate(lib, torch, 2, 1, u, sms, reps)
    for u in (1, 2):
        out["encode"][u] = chain_rate(lib, torch, 1, 1, u, sms, reps)
    out["encode_k2"] = chain_rate(lib, torch, 1, 2, 1, sms, reps) if lib.kmax >= 2 else None
    for k in ks:
        out["decode"][k] = max((chain_rate(lib, torch, 0, k, u, sms, reps) for u in (1, 2)), key=lambda r: r["per_s"])
    for u in (1, 2, 4):
        out["tag"][u] = chain_rate(lib, torch, 3, 1, u, sms, reps)
    sq = max(r["per_s"] for r in out["square"].values())
    enc = max(r["per_s"] for r in out["encode"].values())
    tag = max(r["per_s"] for r in out["tag"].values())
    out["best"] = {"square_s": sq, "roots_s": enc, "compressions_s": tag,
                   "decode_words_s": {k: out["decode"][k]["per_s"] for k in ks},
                   "squarings_per_root": W.root_squarings(lib.field),
                   "root_vs_squarings": sq / enc / W.root_squarings(lib.field)}
    log(f"rates {lib.name}: squarings {sq:.3e}/s, roots {enc:.3e}/s ({out['best']['root_vs_squarings']:.2f}x the bare "
        f"squarings), BLAKE3 {tag:.3e} compressions/s, decode " +
        ", ".join(f"k{k} {out['decode'][k]['per_s']:.3e}" for k in ks) + " words/s")
    return out


def frontier_quantities(f: W.Field, lw: int, msg_blocks: float, read_gbps: float, roots_s: float, dec: dict,
                        comp_s: float, ks: list[int]) -> dict:
    """msg_blocks: the tag's compressions a lane's record (a tag over tb records: its compressions over tb)."""
    r = read_gbps * 1e9
    tau = msg_blocks / W.LANE_BYTES * r / comp_s
    out = {}
    for k in ks:
        x = k * lw / W.LANE_BYTES * r / roots_s
        delta = lw / W.LANE_BYTES * r / dec[k]
        out[k] = {"x": x, "delta": delta, "tau": tau, "mu_rebuild": (1 - tau) / x,
                  "tau_sketch": 2 / WS.vector_bytes(f) * r / comp_s}
    return out


# --- the GEMM ----------------------------------------------------------------------------------------------------------

def gemm_case(lib: WLib, torch, name: str, n: int, kd: int, batches, ks, reps: int, log, wc=None) -> dict:
    f = lib.field
    w = B.fp8_weights(torch, n, kd, 100 + n + kd)
    wb = w.view(torch.uint8)
    pay = payload_store(torch, wb, f)
    nwords = pay.numel() // f.limbs
    tasks = nwords // (W.LANES * lib.lane_words)
    blocks = tasks * W.LANES // lib.tb
    stores = {}
    for k in ks:
        lib.set_key(W.key(f, SEED, k))
        s = torch.empty_like(pay)
        lib.call("ww_encode", ptr(pay), ptr(s), nwords, 0, k, 188 * 4, stream(torch))
        torch.cuda.synchronize()
        stores[k] = s
    ws = torch.empty(lib.so.ww_workspace_bytes(blocks), dtype=torch.uint8, device="cuda")
    tags_p = lib.so.ww_workspace_tags(ptr(ws), blocks)
    tags = torch.as_strided(ws.view(torch.int32), (blocks * 8,), (1,), 0)
    assert tags.data_ptr() == tags_p
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    xs = {}
    gx = torch.Generator(device="cuda").manual_seed(5)
    for mb in batches:
        xs[mb] = torch.randn(mb, kd, device="cuda", generator=gx).to(torch.float8_e4m3fn)
    nch = kd // W.TASK_COLS
    res = {"N": n, "K": kd, "weight_bytes": n * kd, "store_bytes": pay.numel() * 4, "blocks": blocks, "gate": {},
           "times": {}}

    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def wave(k, mb, tag, partials=False):
        occ = lib.so.ww_gemm_occupancy(k, mb, int(tag), int(partials))
        if occ < 1:
            raise RuntimeError(f"ww_gemm_occupancy k={k} M={mb}: {occ}")
        return occ * sms

    def run(k, store, x, y, partials, tag, prefetch=False):
        lib.call("ww_gemm", k, ptr(store), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0, int(partials),
                 seed_buf if tag else None, tags_p if tag else None,
                 wave(k, x.shape[0], tag, partials) if prefetch else 0, stream(torch))

    def commit(root=None):
        lib.call("ww_commit", ptr(ws), blocks, seed_buf, root, stream(torch))

    ntiles = WS.tiles(n, kd)
    nvec = 16 * ntiles
    if lib.sketch:
        import numpy as np

        sgn = sign_tensor(torch, f)
        signs_np = WS.signs(TAG_SEED, f)
        sk = torch.empty(ntiles * 256, dtype=torch.int32, device="cuda")
        sws = torch.empty(lib.so.ww_workspace_bytes(nvec), dtype=torch.uint8, device="cuda")
        stags_p = lib.so.ww_workspace_tags(ptr(sws), nvec)
        stags = torch.as_strided(sws.view(torch.int32), (nvec * 8,), (1,), 0)
        assert stags.data_ptr() == stags_p
        res["sketch_vectors"] = nvec

    def run_sk(k, store, x, y, partials, hashed, with_sk=True):
        lib.call("ww_gemm_sketch", k, ptr(store), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0,
                 int(partials), ptr(sgn), ptr(sk) if with_sk else None, 0, seed_buf if hashed else None,
                 stags_p if hashed else None, 0, stream(torch))

    def sketch_gate(k, x, pp, first: bool, kept: dict) -> dict:
        """The sketch's gate at k: partial sums = the payload store's with the sketch and with its hash, every tile's
        sketch written and the same under both, sampled tiles = `wide_sketch` over their records, sampled tags =
        ro-sketch-blake3/v0 over the device's sketches, the tag kernel apart = the fused hash on every vector, the root
        = `wide_sketch.root` over the device's tags; then later batches give the same sketches and tags."""
        poison = POISON - (1 << 32)
        g = {}
        sk.fill_(poison)
        pf = torch.full_like(pp, float("nan"))
        run_sk(k, stores[k], x, pf, True, False)
        torch.cuda.synchronize()
        g["partials_equal"] = bool(torch.equal(pf, pp))
        g["written"] = not bool((sk.view(-1, 16) == poison).all(1).any())
        sk1 = sk.clone()
        sk.fill_(poison)
        stags.fill_(poison)
        pf.fill_(float("nan"))
        run_sk(k, stores[k], x, pf, True, True)
        torch.cuda.synchronize()
        g["hashed_partials_equal"] = bool(torch.equal(pf, pp))
        g["stable"] = bool(torch.equal(sk, sk1))
        g["tags_written"] = not bool((stags.view(-1, 8) == poison).all(1).any())
        if first:
            rng = random.Random(100 + k)
            sample = sorted({0, ntiles - 1, *rng.sample(range(ntiles), min(6, ntiles))})
            dev = sk1.view(ntiles, 16, 16).cpu().numpy()
            g["sketch_vs_reference"] = {"sampled": len(sample), "mismatch": [
                t for t in sample if not np.array_equal(tile_sketch(stores[k], f, n, kd, t, signs_np), dev[t])]}
            vs = sorted({0, nvec - 1, *rng.sample(range(nvec), min(30, nvec))})
            want = WS.tags_of(TAG_SEED, dev.reshape(-1, 16).astype(np.int64), 0, which=vs)
            devt = stags.view(-1, 8)
            g["tags_vs_reference"] = {"sampled": len(vs), "mismatch": [
                v for v, y in zip(vs, want) if devt[v].cpu().numpy().tobytes() != y][:8]}
            fused = stags.clone()
            stags.fill_(poison)
            lib.call("ww_sketch_tags", ptr(sk1), nvec, 0, seed_buf, stags_p, stream(torch))
            torch.cuda.synchronize()
            g["apart_equal"] = bool(torch.equal(stags, fused))
            root = ctypes.create_string_buffer(64)
            lib.call("ww_sketch_commit", ptr(sws), nvec, seed_buf, root, stream(torch))
            g["root_vs_reference"] = root.raw == WS.root(TAG_SEED, tag_bytes(stags))
            kept[k] = (sk1, stags.clone())
        else:
            g["batch_stable"] = bool(torch.equal(sk1, kept[k][0]) and torch.equal(stags, kept[k][1]))
        g["ok"] = (g["partials_equal"] and g["written"] and g["hashed_partials_equal"] and g["stable"]
                   and g["tags_written"] and g.get("batch_stable", True)
                   and (not first or (not g["sketch_vs_reference"]["mismatch"] and not g["tags_vs_reference"]["mismatch"]
                                      and g["apart_equal"] and g["root_vs_reference"])))
        return g

    sk_kept = {}

    keyed = None
    for gi, mb in enumerate(batches):
        lib.set_key(W.key(f, SEED, max(ks)))
        x = xs[mb]
        ref = x.float() @ w.float().t()
        scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-30)
        pp = torch.full((nch * mb * n,), float("nan"), dtype=torch.float32, device="cuda")
        run(0, pay, x, pp, True, False)
        err = float(((pp.view(nch, mb, n).sum(0) - ref).abs() / scale).max())
        entry = {"payload_vs_torch_rel": err, "decode_equal": {}, "decode_tag_equal": {}, "prefetch_equal": {},
                 "tags_written": {},
                 "control_rejected": None, "tags_vs_reference": {}, "root_vs_reference": {}, "tags_stable": {}}
        tags.fill_(POISON - (1 << 32))
        run(0, pay, x, torch.empty_like(pp), True, False)
        torch.cuda.synchronize()
        entry["control_rejected"] = bool((tags.view(-1, 8) == POISON - (1 << 32)).all(1).any())
        first_tags = {}
        for k in ks:
            lib.set_key(W.key(f, SEED, k))
            pf = torch.full_like(pp, float("nan"))
            run(k, stores[k], x, pf, True, False)
            entry["decode_equal"][k] = bool(torch.equal(pf, pp))
            tags.fill_(POISON - (1 << 32))
            pf.fill_(float("nan"))
            run(k, stores[k], x, pf, True, True)
            torch.cuda.synchronize()
            entry["decode_tag_equal"][k] = bool(torch.equal(pf, pp))
            entry["tags_written"][k] = not bool((tags.view(-1, 8) == POISON - (1 << 32)).all(1).any())
            pf.fill_(float("nan"))
            run(k, stores[k], x, pf, True, True, prefetch=True)
            entry["prefetch_equal"][k] = bool(torch.equal(pf, pp))
            if gi == 0:
                if keyed is None:
                    from verity.core.protocols.bandwidth import tags as T
                    keyed = T.get("ro-blake3/v1").keyed(TAG_SEED)
                rng = random.Random(k)
                sample = sorted({0, blocks - 1, *rng.sample(range(blocks), 30)})
                dev = tags.view(-1, 8)
                bad = [j for j in sample if keyed(j, record_block(stores[k], f, j, lib.tb))
                       != dev[j].cpu().numpy().tobytes()]
                entry["tags_vs_reference"][k] = {"sampled": len(sample), "mismatch": bad[:8]}
                root = ctypes.create_string_buffer(64)
                commit(root)
                want = W.root_of(tag_bytes(tags), TAG_SEED)
                entry["root_vs_reference"][k] = root.raw == want
                first_tags[k] = tags.clone()
                res.setdefault("first_tags_ok", {})[k] = not bad
            else:
                entry["tags_stable"][k] = bool(torch.equal(tags, res["_first_tags"][k]))
            if lib.sketch:
                entry.setdefault("sketch", {})[k] = sketch_gate(k, x, pp, gi == 0, sk_kept)
        if gi == 0:
            res["_first_tags"] = first_tags
        ya = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        lib.set_key(W.key(f, SEED, ks[-1]))
        run(ks[-1], stores[ks[-1]], x, ya, False, True)
        entry["atomic_vs_torch_rel"] = float(((ya - ref).abs() / scale).max())
        entry["ok"] = (err < 1e-5 and entry["control_rejected"] and all(entry["decode_equal"].values())
                       and all(entry["decode_tag_equal"].values()) and all(entry["tags_written"].values())
                       and all(entry["prefetch_equal"].values())
                       and all(not v["mismatch"] for v in entry["tags_vs_reference"].values())
                       and all(entry["root_vs_reference"].values()) and all(entry["tags_stable"].values())
                       and entry["atomic_vs_torch_rel"] < 1e-5
                       and all(g["ok"] for g in entry.get("sketch", {}).values()))
        res["gate"][mb] = entry
        log(f"gemm gate {lib.name} {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (payload rel {err:.1e})"
            + ("" if entry["ok"] else f" {entry}"))
        if not entry["ok"]:
            res.pop("_first_tags", None)
            res["ok"] = False
            return res
    res.pop("_first_tags", None)
    res["ok"] = True

    copies = max(2, -(-B.L2_DEFEAT_BYTES // (pay.numel() * 4)))
    ring = {0: [pay] + [pay.clone() for _ in range(copies - 1)]}
    for k in ks:
        ring[k] = [stores[k]] + [stores[k].clone() for _ in range(copies - 1)]
    wring = [w] + [w.clone() for _ in range(copies - 1)]
    res["copies"] = copies
    plain_ring = None
    if wc is not None:
        plain = B.plain_layout(wb)
        plain_ring = [plain] + [plain.clone() for _ in range(copies - 1)]
    so, st = lib.so, stream(torch)
    for mb in batches:
        x = xs[mb]
        xp = ptr(x.view(torch.uint8))
        y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        yp = ptr(y)
        one = torch.ones((), device="cuda")
        try:
            torch._scaled_mm(x, w.t(), one, one, out_dtype=torch.bfloat16)
            xc, mpad = x, mb
        except Exception:  # noqa: BLE001  cuBLASLt's FP8 path needs M a multiple of 16 here
            mpad = 16 * -(-mb // 16)
            xc = torch.zeros(mpad, kd, device="cuda").to(torch.float8_e4m3fn)
            xc[:mb] = x
        wts = [c.t() for c in wring]

        def arm_cublas():
            for wt in wts:
                torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)

        def arm(k, tag, prefetch=False):
            ps = [ptr(b) for b in ring[k]]
            sp, tp = (seed_buf, tags_p) if tag else (None, None)
            ah = wave(k, mb, tag) if prefetch else 0

            def go():
                rc = 0
                for p in ps:
                    rc |= so.ww_gemm(k, p, xp, yp, mb, n, kd, 0, 0, sp, tp, ah, st)
                if rc:
                    raise RuntimeError(f"ww_gemm k={k} tag={tag} M={mb}: error {rc}")
            return go

        def arm_commit():
            rc = 0
            for _ in range(copies):
                rc |= so.ww_commit(ptr(ws), blocks, seed_buf, None, st)
            if rc:
                raise RuntimeError(f"ww_commit: error {rc}")

        arms = {"cublas": arm_cublas, "ww0": arm(0, False), "t0": arm(0, True), "commit": arm_commit}
        if plain_ring is not None:
            pps = [ptr(b) for b in plain_ring]
            launched = ctypes.c_int(0)
            wc.call("wc_gemm_pipe", 0, pps[0], xp, yp, mb, n, kd, 0, 0, 0, ctypes.byref(launched), st)
            groups = launched.value // nch

            def arm_wc():
                rc = 0
                for p in pps:
                    rc |= wc.so.wc_gemm(0, p, xp, yp, mb, n, kd, 0, 0, st)
                if rc:
                    raise RuntimeError("wc_gemm: error")

            def arm_wc_pipe():
                rc = 0
                for p in pps:
                    rc |= wc.so.wc_gemm_pipe(0, p, xp, yp, mb, n, kd, 0, 0, groups, None, st)
                if rc:
                    raise RuntimeError("wc_gemm_pipe: error")
            arms["wc_plain"], arms["wc_pipe"] = arm_wc, arm_wc_pipe
        for k in ks:
            arms[f"d{k}"] = arm(k, False)
            arms[f"t{k}"] = arm(k, True)
            arms[f"t{k}_ahead"] = arm(k, True, prefetch=True)
        if lib.sketch:
            skp, sgp = ptr(sk), ptr(sgn)

            def arm_sk(k, hashed, apart=False):
                ps = [ptr(b) for b in ring[k]]
                sp, tp, skx = (seed_buf, stags_p, None) if hashed else (None, None, skp)

                def go():
                    rc = 0
                    for p in ps:
                        rc |= so.ww_gemm_sketch(k, p, xp, yp, mb, n, kd, 0, 0, sgp, skx, 0, sp, tp, 0, st)
                        if apart:
                            rc |= so.ww_sketch_tags(skp, nvec, 0, seed_buf, stags_p, st)
                    if rc:
                        raise RuntimeError(f"ww_gemm_sketch k={k} hashed={hashed} M={mb}: error {rc}")
                return go
            for k in ks:
                arms[f"s{k}"] = arm_sk(k, False)
                arms[f"h{k}"] = arm_sk(k, True)
                arms[f"sx{k}"] = arm_sk(k, False, apart=True)
        lib.set_key(W.key(f, SEED, max(ks)))
        for fn in arms.values():
            fn()
        torch.cuda.synchronize()
        times = {a: [] for a in arms}
        names = list(arms)
        for rep in range(reps):
            for a in names[rep % len(names):] + names[:rep % len(names)]:
                times[a].append(queued(lib, torch, arms[a]) / copies)
        med = {a: statistics.median(v) for a, v in times.items()}
        native = min(med[a] for a in ("cublas", "wc_plain", "wc_pipe") if a in med)
        res["times"][mb] = {"median_s": med, "times_s": times, "cublas_m": mpad, "native_s": native,
                            "ratio_vs_native": {a: med[a] / native for a in med},
                            "wave_ctas": {k: wave(k, mb, True) for k in ks}}
        log(f"gemm {lib.name} {name} M={mb}: native {native * 1e6:.1f} us (cublas {med['cublas'] * 1e6:.1f}"
            + (f", wc {med['wc_plain'] * 1e6:.1f}, wc_pipe {med['wc_pipe'] * 1e6:.1f}" if "wc_plain" in med else "")
            + f"); ww0 {med['ww0'] / native:.2f}x, t0 {med['t0'] / native:.2f}x, commit {med['commit'] / native:.2f}x, "
            + ", ".join(f"d{k} {med[f'd{k}'] / native:.2f}x t{k} {med[f't{k}'] / native:.2f}x "
                        f"(L2 prefetch {med[f't{k}_ahead'] / native:.2f}x, wave {wave(k, mb, True)} CTAs)" for k in ks)
            + ("".join(f"; sketch k{k}: s {med[f's{k}'] / native:.3f}x, hashed {med[f'h{k}'] / native:.3f}x, "
                       f"tags apart {med[f'sx{k}'] / native:.3f}x (wave {wave(k, mb, 3)} CTAs)" for k in ks)
               if lib.sketch else ""))
    del ring, wring, stores, pay, w, ws, plain_ring
    torch.cuda.empty_cache()
    return res


def pass_blocks(lib: WLib) -> int:
    """The tag blocks of one pass over the model's linears: LAYERS layers of the four shapes."""
    return LAYERS * sum(n * kd // W.LANE_BYTES for n, kd in B.SHAPES.values()) // lib.tb


def pass_commit(lib: WLib, torch, reps: int, log) -> dict:
    """The window's commit, once a pass: the leaves and tree over every tag of the pass (cold, far past the L2)."""
    blocks = pass_blocks(lib)
    ws = torch.empty(lib.so.ww_workspace_bytes(blocks), dtype=torch.uint8, device="cuda")
    tags = torch.as_strided(ws.view(torch.int32), (blocks * 8,), (1,), 0)
    tags.random_(-(1 << 31), (1 << 31) - 1)
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    so, st = lib.so, stream(torch)

    def go():
        if so.ww_commit(ptr(ws), blocks, seed_buf, None, st):
            raise RuntimeError("ww_commit: error")
    go()
    torch.cuda.synchronize()
    times = [queued(lib, torch, go) for _ in range(reps)]
    med = statistics.median(times)
    log(f"pass commit {lib.name}: {blocks} tags in {med * 1e3:.3f} ms ({blocks / med / 1e9:.2f} G tags/s)")
    del ws, tags
    torch.cuda.empty_cache()
    return {"blocks": blocks, "median_s": med, "times_s": times}


def pass_vectors() -> int:
    """The sketch's vectors (and tags) of one pass: 16 a tile, LAYERS layers of the four shapes."""
    return LAYERS * sum(16 * WS.tiles(n, kd) for n, kd in B.SHAPES.values())


def pass_sketch_commit(lib: WLib, torch, reps: int, log) -> dict:
    """`pass_commit` for the sketch's tags: the leaves (a tile each) and tree over the pass's vectors."""
    nv = pass_vectors()
    ws = torch.empty(lib.so.ww_workspace_bytes(nv), dtype=torch.uint8, device="cuda")
    torch.as_strided(ws.view(torch.int32), (nv * 8,), (1,), 0).random_(-(1 << 31), (1 << 31) - 1)
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    so, st = lib.so, stream(torch)

    def go():
        if so.ww_sketch_commit(ptr(ws), nv, seed_buf, None, st):
            raise RuntimeError("ww_sketch_commit: error")
    go()
    torch.cuda.synchronize()
    times = [queued(lib, torch, go) for _ in range(reps)]
    med = statistics.median(times)
    log(f"pass sketch commit {lib.name}: {nv} tags in {med * 1e3:.3f} ms ({nv / med / 1e9:.2f} G tags/s)")
    del ws
    torch.cuda.empty_cache()
    return {"vectors": nv, "median_s": med, "times_s": times}


def pass_overlap(lib: WLib, torch, cases: dict, k: int, batches, reps: int, log, sketch: bool = False,
                 layers: int | None = None, profile: bool = False) -> dict:
    """One pass of fused GEMMs (decode and tag at k: LAYERS layers of the four linears in order, 232 MB of store a
    layer at p521, so each linear's store is cold again by its next layer) alone, and with the previous window's
    commit (`pass_commit`'s, over the pass's tags) on a second stream from the pass's start. y_commit_overlap is the
    second's excess over the first, over LAYERS times the four linears' native time: the commit's cost to a prover that
    commits window n while it steps through window n + 1. The kernels, stores and key are the ones `gemm_case` gated.
    sketch: the GEMMs hash their sketches (`ww_gemm_sketch`, the pass's vectors numbered from 0), the commit is the
    sketch's (`pass_sketch_commit`), and a third arm commits the pass itself as it goes: each linear's tags go to one
    slot of a two-slot ring, which L2 keeps under the weights' evict-first loads, and its leaves are hashed from there
    on the second stream (`ww_sketch_leaves`), the tree after the last (`ww_sketch_tree`); y_leaves is its excess over
    the pass alone, and `root_equal` its gate: its root is the pass commit's over the same tags written out.
    "leaves_hi" is the same on a high-priority second stream, whose blocks go ahead of the GEMM's queued CTAs, and
    "fold_hi" also folds each linear's leaves SUB levels while L2 holds them (`ww_sketch_fold`, when every linear's
    leaves are a multiple of 2^SUB; gate `root_equal_fold`), so the tree after the last reads back only level SUB;
    "apart" hashes the GEMMs' sketches from a ring first.
    profile: the gate, then one such pass of `layers` layers between cudaProfilerStart and Stop, untimed."""
    f = lib.field
    layers = layers or LAYERS
    lib.set_key(W.key(f, SEED, k))
    stores, wss = {}, {}
    nv = {name: 16 * WS.tiles(*B.SHAPES[name]) for name in cases}
    for name in cases:
        n, kd = B.SHAPES[name]
        pay = payload_store(torch, B.fp8_weights(torch, n, kd, 100 + n + kd).view(torch.uint8), f)
        s = torch.empty_like(pay)
        lib.call("ww_encode", ptr(pay), ptr(s), pay.numel() // f.limbs, 0, k, 188 * 4, stream(torch))
        torch.cuda.synchronize()
        blocks = nv[name] if sketch else pay.numel() // f.limbs // lib.lane_words // lib.tb
        ws = torch.empty(lib.so.ww_workspace_bytes(blocks), dtype=torch.uint8, device="cuda")
        stores[name], wss[name] = s, (ws, lib.so.ww_workspace_tags(ptr(ws), blocks))
        del pay
    pblocks = layers * sum(nv.values()) if sketch else pass_blocks(lib)
    pws = torch.empty(lib.so.ww_workspace_bytes(pblocks), dtype=torch.uint8, device="cuda")
    torch.as_strided(pws.view(torch.int32), (pblocks * 8,), (1,), 0).random_(-(1 << 31), (1 << 31) - 1)
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    sgn = sign_tensor(torch, f) if sketch else None
    ring = [torch.empty(32 * max(nv.values()), dtype=torch.uint8, device="cuda") for _ in range(2)] if sketch else []
    ring_sk = [torch.empty(64 * max(nv.values()), dtype=torch.uint8, device="cuda") for _ in range(2)] if sketch else []
    commit_fn = "ww_sketch_commit" if sketch else "ww_commit"
    sa, sb, sh = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    ev_g, ev_l = [torch.cuda.Event() for _ in range(2)], [torch.cuda.Event() for _ in range(2)]
    so = lib.so
    out = {"k": k, "sketch": sketch, "per_batch": {}}
    for mb in batches:
        xs = {name: torch.randn(mb, B.SHAPES[name][1], device="cuda").to(torch.float8_e4m3fn) for name in cases}
        ys = {name: torch.zeros(mb, B.SHAPES[name][0], dtype=torch.float32, device="cuda") for name in cases}
        torch.cuda.synchronize()
        native = layers * sum(c["times"][mb]["native_s"] for c in cases.values()) if not profile else None

        def gemm(name: str, tags, j: int, sk=None) -> int:
            n, kd = B.SHAPES[name]
            if sketch:
                return so.ww_gemm_sketch(k, ptr(stores[name]), ptr(xs[name].view(torch.uint8)), ptr(ys[name]), mb, n,
                                         kd, 0, 0, ptr(sgn), sk, j, None if sk else seed_buf, None if sk else tags, 0,
                                         sa.cuda_stream)
            return so.ww_gemm(k, ptr(stores[name]), ptr(xs[name].view(torch.uint8)), ptr(ys[name]), mb, n, kd, 0, 0,
                              seed_buf, tags, 0, sa.cuda_stream)

        def committing(root, apart: bool = False, hash_: bool = True, fold: bool = False, hi: bool = False) -> int:
            """The pass committing as it goes; apart: the GEMMs write sketches to a ring (`ww_gemm_sketch` without
            tags) and the second stream hashes them (`ww_sketch_tags`) before the leaves; hash_ False: only the GEMMs
            (apart's own base); fold: each linear's leaves folded by `ww_sketch_fold`, the tree from level SUB; hi:
            the second stream at high priority."""
            sc = sh if hi else sb
            rc, j, i = 0, 0, 0
            for _ in range(layers):
                for name in cases:
                    r = i % 2
                    if i >= 2 and hash_:
                        sa.wait_event(ev_l[r])
                    rc |= gemm(name, ptr(ring[r]), j, ptr(ring_sk[r]) if apart else None)
                    if hash_:
                        ev_g[r].record(sa)
                        sc.wait_event(ev_g[r])
                        if apart:
                            rc |= so.ww_sketch_tags(ptr(ring_sk[r]), nv[name], j, seed_buf, ptr(ring[r]), sc.cuda_stream)
                        rc |= so.ww_sketch_leaves(ptr(pws), pblocks, ptr(ring[r]), j, nv[name], seed_buf, sc.cuda_stream)
                        if fold:
                            rc |= so.ww_sketch_fold(ptr(pws), pblocks, j, nv[name], sc.cuda_stream)
                        ev_l[r].record(sc)
                    j, i = j + nv[name], i + 1
            tree = so.ww_sketch_tree(ptr(pws), pblocks, lib.sub if fold else 0, root, sc.cuda_stream) if hash_ else 0
            return rc | tree

        arms = {"root_equal": {}, "root_equal_apart": {"apart": True}}
        if sketch and all(v % (16 << lib.sub) == 0 for v in nv.values()):
            arms["root_equal_fold"] = {"fold": True, "hi": True}

        if sketch and "root_equal" not in out:
            tags0, rc, j = so.ww_workspace_tags(ptr(pws), pblocks), 0, 0
            for _ in range(layers):
                for name in cases:
                    rc |= gemm(name, tags0 + 32 * j, j)
                    j += nv[name]
            ref = ctypes.create_string_buffer(64)
            rc |= so.ww_sketch_commit(ptr(pws), pblocks, seed_buf, ref, sa.cuda_stream)
            torch.cuda.synchronize()
            for key, kw in arms.items():
                got = ctypes.create_string_buffer(64)
                pws.zero_()
                torch.cuda.synchronize()
                rc_a = rc | committing(got, **kw)
                torch.cuda.synchronize()
                out[key] = not rc_a and ref.raw == got.raw and ref.raw != bytes(64)
                log(f"pass {key} {lib.name} k={k} M={mb} ({layers} layers): root "
                    f"{'equals' if out[key] else 'DIFFERS from'} the pass commit's (rc {rc_a})")
        if profile:
            torch.cuda.profiler.start()
            rc = 0
            for kw in arms.values():
                rc |= committing(None, **kw)
                torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            out["profiled"] = {"batch": mb, "layers": layers, "rc": rc, "order": list(arms)}
            break

        def one(mode: str) -> float:
            a, b, c_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            with torch.cuda.stream(sa):
                lib.call("ww_spin", B.SPIN_CYCLES, sa.cuda_stream)
                a.record(sa)
            if mode == "commit":
                sb.wait_event(a)
                if getattr(so, commit_fn)(ptr(pws), pblocks, seed_buf, None, sb.cuda_stream):
                    raise RuntimeError(f"{commit_fn}: error")
                c_end.record(sb)
            rc, j = 0, 0
            if mode in ("leaves", "leaves_hi", "fold_hi", "apart", "alone_s"):
                rc = committing(None, mode in ("apart", "alone_s"), mode != "alone_s", mode == "fold_hi",
                                mode.endswith("_hi"))
                if mode != "alone_s":
                    c_end.record(sh if mode.endswith("_hi") else sb)
            else:
                for _ in range(layers):
                    for name in cases:
                        rc |= gemm(name, wss[name][1], j)
                        j += nv[name]
            if rc:
                raise RuntimeError(f"pass {mode} k={k} M={mb}: error {rc}")
            if mode not in ("alone", "alone_s"):
                sa.wait_event(c_end)
            b.record(sa)
            b.synchronize()
            return a.elapsed_time(b) / 1e3

        modes = ["alone", "commit"]
        if sketch and out["root_equal"]:
            modes += ["leaves", "leaves_hi"]
        if sketch and out.get("root_equal_fold"):
            modes.append("fold_hi")
        if sketch and out["root_equal_apart"]:
            modes += ["alone_s", "apart"]
        for m in modes:
            one(m)
        t = {m: [] for m in modes}
        for rep in range(reps):
            for m in (modes if rep % 2 == 0 else modes[::-1]):
                t[m].append(one(m))
        med = {m: statistics.median(v) for m, v in t.items()}
        ta = med["alone"]
        out["per_batch"][mb] = {"alone_s": t["alone"], "with_commit_s": t["commit"], "native_pass_s": native,
                                "y_pass": ta / native, "y_commit_overlap": (med["commit"] - ta) / native,
                                "times_s": t, "y_mode": {m: v / native for m, v in med.items()}}
        line = (f"pass overlap {lib.name}{' sketch' if sketch else ''} k={k} M={mb}: a pass of fused GEMMs "
                f"{ta * 1e3:.2f} ms ({ta / native:.2f}x native), with the commit beside it {med['commit'] * 1e3:.2f} ms: "
                f"commit +{(med['commit'] - ta) / native:.3f}")
        if "leaves" in med:
            out["per_batch"][mb].update(y_leaves=(med["leaves"] - ta) / native)
            line += f"; committing as it goes {med['leaves'] / native:.3f}x (+{(med['leaves'] - ta) / native:.3f})"
        for m, what in (("leaves_hi", "at high priority"), ("fold_hi", "folded at high priority")):
            if m in med:
                out["per_batch"][mb][f"y_{m}"] = (med[m] - ta) / native
                line += f"; {what} {med[m] / native:.3f}x (+{(med[m] - ta) / native:.3f})"
        if "apart" in med:
            out["per_batch"][mb].update(y_apart=(med["apart"] - med["alone_s"]) / native)
            line += (f"; hashed apart from the sketch ring {med['apart'] / native:.3f}x (sketch GEMMs alone "
                     f"{med['alone_s'] / native:.3f}x, +{(med['apart'] - med['alone_s']) / native:.3f})")
        log(line)
        del xs, ys
    del stores, wss, pws, ring, ring_sk
    torch.cuda.empty_cache()
    return out


def breakdown(cases: dict, batches, ks, commit_s: float, sketch_commit_s: float | None = None) -> dict:
    """Layer-weighted: Σ over the four linears of each arm's time over Σ of native's, per batch and k. The commit is
    the window's, once a pass (`pass_commit`), over LAYERS times the four linears' native time; `y_commit_linear` is a
    commit launched per linear, for comparison. With the sketch's arms, row[k]["sketch"]: total = h<k> / native + its
    pass commit (`pass_sketch_commit`), tag = (h<k> − d<k>) / native (the sketch and its hash in the kernel),
    sketch_only = (s<k> − d<k>) / native (the sketch written out, unhashed) and total_apart = sx<k> / native + the
    commit (the sketch hashed by `ww_sketch_tags` after each call)."""
    out = {}
    for mb in batches:
        t = [c["times"][mb] for c in cases.values()]
        native = sum(x["native_s"] for x in t)
        s = {a: sum(x["median_s"][a] for x in t) for a in t[0]["median_s"]}
        y_commit = commit_s / (LAYERS * native)
        row = {"native_s": native, "base": s["ww0"] / native, "tag_only": (s["t0"] - s["ww0"]) / native}
        for k in ks:
            row[k] = {"y_total": s[f"t{k}"] / native + y_commit, "y_decode": (s[f"d{k}"] - s["ww0"]) / native,
                      "y_tag": (s[f"t{k}"] - s[f"d{k}"]) / native, "y_commit": y_commit,
                      "y_commit_linear": s["commit"] / native, "y_base": s["ww0"] / native,
                      "y_total_prefetch": s[f"t{k}_ahead"] / native + y_commit}
            if sketch_commit_s is not None and f"h{k}" in s:
                y_cs = sketch_commit_s / (LAYERS * native)
                row[k]["sketch"] = {"y_total": s[f"h{k}"] / native + y_cs, "y_decode": row[k]["y_decode"],
                                    "y_tag": (s[f"h{k}"] - s[f"d{k}"]) / native, "y_commit": y_cs,
                                    "y_base": row[k]["y_base"], "y_sketch_only": (s[f"s{k}"] - s[f"d{k}"]) / native,
                                    "y_total_apart": s[f"sx{k}"] / native + y_cs,
                                    "y_tag_apart": (s[f"sx{k}"] - s[f"d{k}"]) / native}
        out[mb] = row
    return out


def clocks(lib: WLib, torch, log, what: str, squarings_s: float) -> dict:
    """The SM clock (MHz) sampled while a compute-only squaring chain runs (about 1.5 s), and the board's maximum,
    through nvidia-smi."""
    import subprocess

    dev = os.environ.get("GPU_LEASE_UUID") or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = ["nvidia-smi", "-i", dev, "--query-gpu=clocks.sm,clocks.max.sm,clocks.mem,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"]
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    grid = torch.cuda.get_device_properties(0).multi_processor_count * 8
    steps = max(1, int(1.5 * squarings_s / (grid * 256)))
    lib.call("ww_chain", 2, 1, 1, steps, ptr(xr), grid, stream(torch))
    samples = []
    for _ in range(3):
        try:
            out = subprocess.run(q, check=False, capture_output=True, text=True, timeout=10).stdout.strip()
            samples.append([float(v) for v in out.split(",")])
        except Exception as e:  # noqa: BLE001
            samples.append(str(e))
    torch.cuda.synchronize()
    good = [s for s in samples if isinstance(s, list)]
    res = {"what": what, "samples": samples}
    if good:
        res.update(sm_mhz=statistics.median(s[0] for s in good), max_sm_mhz=good[0][1], mem_mhz=good[0][2])
        log(f"clock ({what}): SM {res['sm_mhz']:.0f} MHz of max {res['max_sm_mhz']:.0f}, mem {res['mem_mhz']:.0f} MHz")
    return res


def points(lib: WLib, fq: dict, bd: dict, run_id: str | None, gpu: str, sm_mhz: float | None = None,
           po: dict | None = None, po_sk: dict | None = None) -> list[dict]:
    """One point per batch and k with the commit serial after the pass, and, where `pass_overlap` measured it, one
    with the commit beside the next pass (arithmetic ".../commit-overlapped"). The sketch's rows give the same under
    ".../sketch" (hashed in the kernel, τ the sketch's), and ".../sketch-apart" (hashed by a kernel after each
    call); and the passes that commit as they go, at their whole pass's time over native (".../leaves-per-linear",
    its ".../hi" on a high-priority stream, ".../folded-per-linear/hi", the commit their excess over the same GEMMs
    alone, which for ".../sketch-apart/..." includes the hash)."""
    f = lib.field
    arith = (f"cuda-int32/{'mersenne' if f.p == (1 << f.bits) - 1 else 'pseudo-mersenne'}/sqr{lib.sqr}"
             + ("/rolled" if lib.roll else "") + (f"/occ{lib.occ}" if lib.occ > 1 else "")
             + ("/cc" if lib.cc else "") + ("/b3u2" if lib.b3u else "") + (f"/tb{lib.tb}" if lib.tb > 1 else "") + (f"/kw{lib.kw}" if lib.kw > lib.tb else "")
             + ("/skend" if getattr(lib, "skend", 0) else "")
             + (f"/maxreg{lib.maxreg}" if getattr(lib, "maxreg", 0) else "") + ("/sku" if getattr(lib, "sku", 0) else ""))
    out = []

    def point(k: int, mb: int, q: dict, y: dict, y_commit: float, suffix: str) -> dict:
        return {"schema": POINT_SCHEMA, "lane": "ra-codec-wide", "model": MODEL, "b": f.bits, "k": k,
                "arithmetic": arith + suffix,
                "packing": f.payload, "expansion": f.expansion(), "batch": mb,
                "y_total": y["y_total"] - y["y_commit"] + y_commit, "y_decode": y["y_decode"], "y_tag": y["y_tag"],
                "y_commit": y_commit, "y_commit_linear": y.get("y_commit_linear"), "y_base": y["y_base"],
                "x": q["x"], "delta": q["delta"], "tau": q["tau"], "mu_rebuild": q["mu_rebuild"],
                "payload_per_stored_bit": 1 / f.expansion(),
                "measured": True, "label": f"{f.name} k={k} {lib.name}{suffix}", "run": run_id, "art": None,
                "gpu": gpu, "sm_mhz": sm_mhz}

    def sketch_q(q: dict) -> dict:
        return dict(q, tau=q["tau_sketch"], mu_rebuild=(1 - q["tau_sketch"]) / q["x"])

    for mb, row in bd.items():
        for k, q in fq.items():
            if k not in row:
                continue
            out.append(point(k, mb, q, row[k], row[k]["y_commit"], ""))
            o = (po or {}).get(k, {}).get("per_batch", {}).get(mb)
            if o is not None:
                out.append(point(k, mb, q, row[k], max(0.0, o["y_commit_overlap"]), "/commit-overlapped"))
            y = row[k].get("sketch")
            if y is None:
                continue
            out.append(point(k, mb, sketch_q(q), y, y["y_commit"], "/sketch"))
            out.append(point(k, mb, sketch_q(q), dict(y, y_total=y["y_total_apart"], y_tag=y["y_tag_apart"]),
                             y["y_commit"], "/sketch-apart"))
            o = (po_sk or {}).get(k, {}).get("per_batch", {}).get(mb)
            if o is not None:
                out.append(point(k, mb, sketch_q(q), y, max(0.0, o["y_commit_overlap"]), "/sketch/commit-overlapped"))
                ym = o.get("y_mode", {})
                for mode, base, suffix in (("leaves", "alone", "/sketch/leaves-per-linear"),
                                           ("leaves_hi", "alone", "/sketch/leaves-per-linear/hi"),
                                           ("fold_hi", "alone", "/sketch/folded-per-linear/hi"),
                                           ("apart", "alone_s", "/sketch-apart/leaves-per-linear")):
                    if mode in ym:
                        yc = max(0.0, ym[mode] - ym[base])
                        yp = dict(y, y_total=ym[mode], y_commit=yc, y_tag=ym[mode] - yc - y["y_base"] - y["y_decode"])
                        out.append(point(k, mb, sketch_q(q), yp, yc, suffix))
    return out


def write_result(res: dict, out_dir: Path) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for p in res.get("points", []):
        tag = f"{p['label'].replace(' ', '_')}_m{p['batch']}"
        ms.append({"name": f"y_total_{tag}", "value": p["y_total"], "unit": "1"})
        ms.append({"name": f"x_{tag}", "value": p["x"], "unit": "1"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/wide.json", f"{rel}/points.json", f"{rel}/bench.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {n: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                             for n, p in res["lib_paths"].items()}, "seed": SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "wide word-codec gates",
                          "detail": "device codec = wide.py, fused partial sums = payload store's, tags and root = "
                                    "ro-blake3/v1's", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH of a wide.cu build")
    ap.add_argument("--wc", help="PATH of a wc.cu build, for its plain FP8 kernels in native")
    ap.add_argument("--out", required=True)
    ap.add_argument("--store-gib", type=float, default=4.0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gate-words", type=int, default=2048)
    ap.add_argument("--model", choices=sorted(MODELS), default=MODEL, help="whose linears and layer count")
    ap.add_argument("--shapes", help="a subset of the model's linears (default: all four)")
    ap.add_argument("--batches", default=",".join(map(str, B.BATCHES)))
    ap.add_argument("--ks", default="1,2", help="k values, each library taking those up to its KMAX")
    ap.add_argument("--no-gemm", action="store_true")
    a = ap.parse_args()
    use_model(a.model)
    a.shapes = a.shapes or ",".join(B.SHAPES)

    import torch
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "bench.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    libs = [WLib(*spec.split("=", 1)) for spec in a.lib]
    wc = B.Lib("wc", a.wc) if a.wc else None
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    ks_all = [int(k) for k in a.ks.split(",")]
    batches = [int(b) for b in a.batches.split(",")]
    res = {"device": {"name": props.name, "sms": sms, "cc": f"{props.major}.{props.minor}",
                      "memory_bytes": props.total_memory}, "seed": SEED.hex(), "libs": {}, "ok": True,
           "lib_paths": {lib.name: lib.path for lib in libs}}

    def save() -> None:
        (out_dir / "wide.json").write_text(json.dumps(res, indent=1, default=str))

    log(f"device {props.name}, {sms} SMs")
    lay = gate_layouts(torch, libs, log)
    res["layouts"] = lay
    res["ok"] &= lay["ok"]
    for lib in libs:
        ks = [k for k in ks_all if k <= lib.kmax]
        g = gate_codec(lib, torch, a.gate_words, ks, log)
        res["libs"][lib.name] = {"gate": g, "bits": lib.bits, "sqr": lib.sqr, "kmax": lib.kmax}
        res["ok"] &= all(v["ok"] for v in g.values())
    if not res["ok"]:
        log("a gate failed: nothing timed")
        save()
        return 5
    res["read"] = read_rate(libs[0], torch, int(a.store_gib * (1 << 30)) // 64 * 64, sms, a.reps)
    log(f"read {res['read']['gbps']:.0f} GB/s")
    for lib in libs:
        ks = [k for k in ks_all if k <= lib.kmax]
        res["libs"][lib.name]["rates"] = rates(lib, torch, ks, sms, a.reps, log)
        res["libs"][lib.name]["clock"] = clocks(lib, torch, log, f"{lib.name} squaring chain",
                                                res["libs"][lib.name]["rates"]["best"]["square_s"])
        save()
    fields = sorted({lib.bits for lib in libs})
    res["frontier"] = {}
    for bits in fields:
        group = [lib for lib in libs if lib.bits == bits]
        ks = [k for k in ks_all if k <= max(lib.kmax for lib in group)]
        roots = max(res["libs"][lib.name]["rates"]["best"]["roots_s"] for lib in group)
        comp = max(res["libs"][lib.name]["rates"]["best"]["compressions_s"] for lib in group)
        dec = {k: max(res["libs"][lib.name]["rates"]["best"]["decode_words_s"].get(k, 0.0) for lib in group) for k in ks}
        fq = frontier_quantities(group[0].field, group[0].lane_words, group[0].msg_blocks / group[0].tb,
                                 res["read"]["gbps"], roots, dec, comp, ks)
        res["frontier"][W.field(bits).name] = fq
        for lib in group:
            res["libs"][lib.name]["frontier"] = frontier_quantities(
                lib.field, lib.lane_words, lib.msg_blocks / lib.tb, res["read"]["gbps"], roots, dec, comp,
                [k for k in ks if k <= lib.kmax])
        for k, q in fq.items():
            log(f"{W.field(bits).name} k={k}: x {q['x']:.1f}, delta {q['delta']:.3f}, tau {q['tau']:.3f}, "
                f"mu_rebuild {q['mu_rebuild'] * 100:.2f}%")
    save()
    res["points"] = []
    if not a.no_gemm:
        for lib in libs:
            ks = [k for k in ks_all if k <= lib.kmax]
            cases = {}
            for name in a.shapes.split(","):
                n, kd = B.SHAPES[name]
                cases[name] = gemm_case(lib, torch, name, n, kd, batches, ks, a.reps, log, wc)
                res["libs"][lib.name].setdefault("gemm", {})[name] = cases[name]
                res["ok"] &= cases[name].get("ok", False)
                save()
                if not cases[name].get("ok", False):
                    break
            if all(c.get("ok", False) for c in cases.values()) and len(cases) == len(B.SHAPES):
                lr = res["libs"][lib.name]
                pc = pass_commit(lib, torch, a.reps, log)
                lr["pass_commit"] = pc
                psc = pass_sketch_commit(lib, torch, a.reps, log) if lib.sketch else None
                lr["pass_sketch_commit"] = psc
                bd = breakdown(cases, batches, ks, pc["median_s"], psc["median_s"] if psc else None)
                lr["breakdown"] = bd
                fb = [mb for mb in (1, 32) if mb in batches]
                save()
                for mb, row in bd.items():
                    log(f"layer-weighted {lib.name} M={mb}: base {row['base']:.2f}, " + ", ".join(
                        f"k{k} total {row[k]['y_total']:.2f} (decode {row[k]['y_decode']:.2f}, tag {row[k]['y_tag']:.2f}"
                        f", commit {row[k]['y_commit']:.3f}; per linear {row[k]['y_commit_linear']:.2f})" for k in ks))
                    for k in ks:
                        y = row[k].get("sketch")
                        if y:
                            log(f"layer-weighted {lib.name} sketch M={mb} k{k}: total {y['y_total']:.3f} (decode "
                                f"{y['y_decode']:.3f}, sketch+hash {y['y_tag']:.3f}, sketch alone "
                                f"{y['y_sketch_only']:.3f}, commit {y['y_commit']:.4f}); hashed apart "
                                f"{y['y_total_apart']:.3f}")
                po_sk = ({k: pass_overlap(lib, torch, cases, k, fb, a.reps, log, sketch=True) for k in ks}
                         if lib.sketch else None)
                lr["pass_overlap_sketch"] = po_sk
                save()
                po = {k: pass_overlap(lib, torch, cases, k, fb, a.reps, log) for k in ks}
                lr["pass_overlap"] = po
                fq = lr["frontier"]
                res["points"] += points(lib, fq, bd, os.environ.get("RESEARCH_RUN_ID"), props.name,
                                        lr.get("clock", {}).get("sm_mhz"), po, po_sk)
            save()
    (out_dir / "points.json").write_text(json.dumps(res["points"], indent=1))
    save()
    write_result(res, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
