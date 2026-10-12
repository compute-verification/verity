"""The wide word codecs' benchmark on one GPU: `wide.cu`'s libraries (`--lib NAME=PATH`, one per field and squaring
form) gated against `wide.py`, then timed. Writes `wide.json`, `points.json` (the frontier's points, `frontier.py`'s
schema) and `bench.log` under `--out`, and in a `research run` its `result.json`.

Nothing is timed before its gate passed, in this order:
- `layouts`: the store built on the device from FP8 weights equals `wide.store_bytes(wide.fragment_payloads(…))`;
- `codec`: for each library and k, device encode equals `wide.encode_word` on random and edge payloads, device decode
  inverts it on every word, and the device's k inverse rounds of random canonical words equal `wide.decode_raw`;
- `read`: R, the device read rate over `--store-gib`, the best of a few grids;
- `rates`: compute-only throughputs, each step's input the previous step's output: the squaring (ILP 1, 2, 4), encode
  (roots/s, ILP 1, 2), decode at each k and the tag's BLAKE3 compression (ILP 1, 2, 4); x, Δ and τ follow;
- `gemm`: at Llama-3.1-8B's linears and batch 1–32, for each library and k: the fused kernel's partial sums equal the
  payload store's (k = 0) bit for bit, without and with the tag; the payload store's sum is within FP32 rounding of
  torch's; every tag of the matrix was written (the tag area poisoned with 0xA5 first, and a run without the tag over
  the poison is caught); a sample of tags equals `ro-blake3/v1` over the encoded words; the commit's root equals
  `MerkleTree(tags.leaves(…))` over the device's tags. Then interleaved reps of cuBLASLt FP8 (`torch._scaled_mm`),
  `wc.cu`'s plain FP8 kernels (`--wc`), the wide kernel on the payload store (`ww0`), decode only (`d<k>`), decode and
  tag (`t<k>`; `t<k>_ahead` with the L2 prefetch of the next wave's records, a wave being `ww_gemm_occupancy` × the
  SMs), tag only (`t0`) and the commit, every call on cold weights (a ring of copies past the L2), each arm queued
  behind a spin kernel. Native is the fastest of cuBLASLt and the plain kernels.

The frontier's quantities, per field and k (`wide.json` "frontier"):
  x        k · (L / 256) · R / (encode roots/s at k = 1): the encode time of the whole store per native pass at R, from the
           best encode of the libraries (the adversary's)
  Δ        (L / 256) · R / (decode words/s at k): decode's share, compute only
  τ        (TAG_BLOCKS / TB / 256) · R / (BLAKE3 compressions/s): the tag's share, compute only (a WW_TB = 2 library's
           tag covers two records in TAG_BLOCKS = 9 compressions)
  μ        (1 − τ) / x, the rebuild bound
  y        layer-weighted fused time over native: total = t<k> / native + commit, decode = (d<k> − ww0) / native,
           tag = (t<k> − d<k>) / native; the rest (ww0 / native) is the layout's base. The commit is the window's, once
           a pass: the leaves and tree over the pass's tags (32 layers of the four linears, `pass_commit`) over 32
           times the four linears' native time; `y_commit_linear` is a commit launched per linear, for comparison.
           A ".../commit-overlapped" point takes the commit as its excess beside the next pass instead (`pass_overlap`:
           a pass of fused GEMMs alone and with the commit on a second stream, at batch 1 and 32)
A ".../mixed-k" point gives each linear a k of its own (`mixed`): x and Δ at the mean k over the stored words, y the
sum of each linear's measured time at its k; only the assignments no other beats in both y and mean k.
The SM clock is sampled during a squaring chain for each library (`clock`, and each point's `sm_mhz`).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import itertools
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
import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/wide/bench/seed").digest()
TAG_SEED = hashlib.sha256(b"verity/word-codec/wide/bench/tag-seed").digest()
POINT_SCHEMA = "verity/codec-frontier-point/v0"
POISON = 0xA5A5A5A5
DEV = "cuda"
LAYERS = 32  # Llama-3.1-8B


class WLib:
    def __init__(self, name: str, path: str) -> None:
        v, i, q, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t
        self.name, self.path, self.so = name, path, ctypes.CDLL(path)
        so = self.so
        so.ww_constants.argtypes = [v]
        so.ww_set_key.argtypes = [v, i, v, v, v, v]
        so.ww_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.ww_encode.argtypes = [v, v, q, q, i, i, v]
        so.ww_chain.argtypes = [i, i, i, q, v, i, v]
        so.ww_read.argtypes = [v, q, v, i, v]
        so.ww_spin.argtypes = [ctypes.c_longlong, v]
        so.ww_gemm.argtypes = [i, v, v, v, i, i, i, q, i, v, v, i, v]
        so.ww_gemm_occupancy.argtypes = [i, i, i, i]
        so.ww_workspace_bytes.argtypes = [q]
        so.ww_workspace_bytes.restype = s
        so.ww_workspace_tags.argtypes = [v, q]
        so.ww_workspace_tags.restype = v
        so.ww_commit.argtypes = [v, q, v, v, v]
        cs = (ctypes.c_int * 15)()
        so.ww_constants(cs)
        (self.bits, self.limbs, self.payload, self.lane_words, lanes, tiles, self.gemm_warps, self.kmax, self.sqr,
         self.msg_blocks, self.roll, self.occ, self.cc, self.b3u, self.tb) = list(cs)
        self.field = W.field(self.bits)
        f = self.field
        if (self.limbs, self.payload, self.lane_words, lanes, tiles) != (f.limbs, f.payload, f.lane_words(), W.LANES,
                                                                          W.TASK_TILES):
            raise RuntimeError(f"{name}: constants {list(cs)} are not wide.py's")

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
    edges = [0, 1, 2, (1 << f.payload) - 1, 1 << (f.payload - 1), (1 << 32) - 1, 1 << 32, (1 << 64) + 5]
    ms = edges + [rng.getrandbits(f.payload) for _ in range(n_words - len(edges))]
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
        out[k] = {"x": x, "delta": delta, "tau": tau, "mu_rebuild": (1 - tau) / x}
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
                       and entry["atomic_vs_torch_rel"] < 1e-5)
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
                        f"(L2 prefetch {med[f't{k}_ahead'] / native:.2f}x, wave {wave(k, mb, True)} CTAs)" for k in ks))
    del ring, wring, stores, pay, w, ws, plain_ring
    torch.cuda.empty_cache()
    return res


def pass_blocks(lib: WLib) -> int:
    """The tag blocks of one pass over Llama-3.1-8B's linears: LAYERS layers of the four shapes."""
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


def pass_overlap(lib: WLib, torch, cases: dict, k: int, batches, reps: int, log) -> dict:
    """One pass of fused GEMMs (decode and tag at k: LAYERS layers of the four linears in order, 232 MB of store a
    layer at p521, so each linear's store is cold again by its next layer) alone, and with the previous window's
    commit (`pass_commit`'s, over the pass's tags) on a second stream from the pass's start. y_commit_overlap is the
    second's excess over the first, over LAYERS times the four linears' native time: the commit's cost to a prover that
    commits window n while it steps through window n + 1. The kernels, stores and key are the ones `gemm_case` gated."""
    f = lib.field
    lib.set_key(W.key(f, SEED, k))
    stores, wss = {}, {}
    for name in cases:
        n, kd = B.SHAPES[name]
        pay = payload_store(torch, B.fp8_weights(torch, n, kd, 100 + n + kd).view(torch.uint8), f)
        s = torch.empty_like(pay)
        lib.call("ww_encode", ptr(pay), ptr(s), pay.numel() // f.limbs, 0, k, 188 * 4, stream(torch))
        torch.cuda.synchronize()
        blocks = pay.numel() // f.limbs // lib.lane_words // lib.tb
        ws = torch.empty(lib.so.ww_workspace_bytes(blocks), dtype=torch.uint8, device="cuda")
        stores[name], wss[name] = s, (ws, lib.so.ww_workspace_tags(ptr(ws), blocks))
        del pay
    pblocks = pass_blocks(lib)
    pws = torch.empty(lib.so.ww_workspace_bytes(pblocks), dtype=torch.uint8, device="cuda")
    torch.as_strided(pws.view(torch.int32), (pblocks * 8,), (1,), 0).random_(-(1 << 31), (1 << 31) - 1)
    seed_buf = ctypes.create_string_buffer(TAG_SEED, 32)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    so = lib.so
    out = {"k": k, "per_batch": {}}
    for mb in batches:
        xs = {name: torch.randn(mb, B.SHAPES[name][1], device="cuda").to(torch.float8_e4m3fn) for name in cases}
        ys = {name: torch.zeros(mb, B.SHAPES[name][0], dtype=torch.float32, device="cuda") for name in cases}
        native = LAYERS * sum(c["times"][mb]["native_s"] for c in cases.values())

        def one(with_commit: bool) -> float:
            a, b, c_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
            with torch.cuda.stream(sa):
                lib.call("ww_spin", B.SPIN_CYCLES, sa.cuda_stream)
                a.record(sa)
            if with_commit:
                sb.wait_event(a)
                if so.ww_commit(ptr(pws), pblocks, seed_buf, None, sb.cuda_stream):
                    raise RuntimeError("ww_commit: error")
                c_end.record(sb)
            rc = 0
            for _ in range(LAYERS):
                for name in cases:
                    n, kd = B.SHAPES[name]
                    rc |= so.ww_gemm(k, ptr(stores[name]), ptr(xs[name].view(torch.uint8)), ptr(ys[name]), mb, n, kd,
                                     0, 0, seed_buf, wss[name][1], 0, sa.cuda_stream)
            if rc:
                raise RuntimeError(f"ww_gemm k={k} M={mb}: error {rc}")
            if with_commit:
                sa.wait_event(c_end)
            b.record(sa)
            b.synchronize()
            return a.elapsed_time(b) / 1e3

        one(False), one(True)
        alone, both = [], []
        for rep in range(reps):
            for w in ((False, True) if rep % 2 == 0 else (True, False)):
                (both if w else alone).append(one(w))
        ta, tb = statistics.median(alone), statistics.median(both)
        out["per_batch"][mb] = {"alone_s": alone, "with_commit_s": both, "native_pass_s": native,
                                "y_pass": ta / native, "y_commit_overlap": (tb - ta) / native}
        log(f"pass overlap {lib.name} k={k} M={mb}: a pass of fused GEMMs {ta * 1e3:.2f} ms ({ta / native:.2f}x native),"
            f" with the commit beside it {tb * 1e3:.2f} ms: commit +{(tb - ta) / native:.3f}")
        del xs, ys
    del stores, wss, pws
    torch.cuda.empty_cache()
    return out


def breakdown(cases: dict, batches, ks, commit_s: float) -> dict:
    """Layer-weighted: Σ over the four linears of each arm's time over Σ of native's, per batch and k. The commit is
    the window's, once a pass (`pass_commit`), over LAYERS times the four linears' native time; `y_commit_linear` is a
    commit launched per linear, for comparison."""
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
        out[mb] = row
    return out


def mixed(cases: dict, batches, ks, commit_s: float) -> dict:
    """Each linear at a k of its own (public and fixed per linear), from the same arms as `breakdown`: per batch, the
    assignments on the lower envelope of y_total against k_mean, the mean k over the stored words (the linears
    weighted by their weights' bytes), which x and Δ scale with. The uniform assignments are `breakdown`'s rows."""
    names = list(cases)
    wbytes = {nm: cases[nm]["weight_bytes"] for nm in names}
    out = {}
    for mb in batches:
        t = {nm: cases[nm]["times"][mb] for nm in names}
        native = sum(x["native_s"] for x in t.values())
        y_commit = commit_s / (LAYERS * native)
        rows = []
        for combo in itertools.product(ks, repeat=len(names)):
            kk = dict(zip(names, combo))

            def arm(a, kk=kk):
                return sum(t[nm]["median_s"][a.format(k=kk[nm])] for nm in names) / native
            share = {nm: wbytes[nm] / sum(wbytes.values()) for nm in names}
            rows.append({"ks": kk, "share": share, "k_mean": sum(share[nm] * kk[nm] for nm in names),
                         "y_total": arm("t{k}") + y_commit, "y_decode": arm("d{k}") - arm("ww0"),
                         "y_tag": arm("t{k}") - arm("d{k}"), "y_commit": y_commit, "y_base": arm("ww0")})
        env = []
        for r in sorted(rows, key=lambda r: (-r["k_mean"], r["y_total"])):
            if not env or r["y_total"] < env[-1]["y_total"]:
                env.append(r)
        out[mb] = env[::-1]
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
           po: dict | None = None, mx: dict | None = None) -> list[dict]:
    """One point per batch and k with the commit serial after the pass, and, where `pass_overlap` measured it, one
    with the commit beside the next pass (arithmetic ".../commit-overlapped"); one per non-uniform row of `mixed`
    (".../mixed-k")."""
    f = lib.field
    arith = (f"cuda-int32/{'mersenne' if f.p == (1 << f.bits) - 1 else 'pseudo-mersenne'}/sqr{lib.sqr}"
             + ("/rolled" if lib.roll else "") + (f"/occ{lib.occ}" if lib.occ > 1 else "")
             + ("/cc" if lib.cc else "") + ("/b3u2" if lib.b3u else "") + (f"/tb{lib.tb}" if lib.tb > 1 else ""))
    out = []

    def point(k: int, mb: int, q: dict, y: dict, y_commit: float, suffix: str) -> dict:
        return {"schema": POINT_SCHEMA, "lane": "ra-codec-wide", "b": f.bits, "k": k, "arithmetic": arith + suffix,
                "packing": f.payload, "expansion": f.expansion(), "batch": mb,
                "y_total": y["y_total"] - y["y_commit"] + y_commit, "y_decode": y["y_decode"], "y_tag": y["y_tag"],
                "y_commit": y_commit, "y_commit_linear": y.get("y_commit_linear"), "y_base": y["y_base"],
                "x": q["x"], "delta": q["delta"], "tau": q["tau"], "mu_rebuild": q["mu_rebuild"],
                "payload_per_stored_bit": 1 / f.expansion(),
                "measured": True, "label": f"{f.name} k={k} {lib.name}{suffix}", "run": run_id, "art": None,
                "gpu": gpu, "sm_mhz": sm_mhz}

    for mb, row in bd.items():
        for k, q in fq.items():
            if k not in row:
                continue
            out.append(point(k, mb, q, row[k], row[k]["y_commit"], ""))
            o = (po or {}).get(k, {}).get("per_batch", {}).get(mb)
            if o is not None:
                out.append(point(k, mb, q, row[k], max(0.0, o["y_commit_overlap"]), "/commit-overlapped"))
    for mb, rows in (mx or {}).items():
        for r in rows:
            if len(set(r["ks"].values())) == 1:
                continue
            km = r["k_mean"]
            q = {"x": km * fq[min(fq)]["x"] / min(fq), "tau": fq[min(fq)]["tau"],
                 "delta": sum(sh * fq[r["ks"][nm]]["delta"] for nm, sh in r["share"].items())}
            q["mu_rebuild"] = (1 - q["tau"]) / q["x"]
            p = point(km, mb, q, r, r["y_commit"], "/mixed-k")
            p["ks"] = r["ks"]
            p["label"] = f"{f.name} k=" + "".join(f"{nm[0]}{k}" for nm, k in r["ks"].items()) + f" {lib.name}/mixed-k"
            out.append(p)
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
    ap.add_argument("--shapes", default=",".join(B.SHAPES))
    ap.add_argument("--batches", default=",".join(map(str, B.BATCHES)))
    ap.add_argument("--ks", default="1,2", help="k values, each library taking those up to its KMAX")
    ap.add_argument("--no-gemm", action="store_true")
    a = ap.parse_args()

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
                pc = pass_commit(lib, torch, a.reps, log)
                res["libs"][lib.name]["pass_commit"] = pc
                bd = breakdown(cases, batches, ks, pc["median_s"])
                res["libs"][lib.name]["breakdown"] = bd
                mx = mixed(cases, [mb for mb in (1, 32) if mb in batches], ks, pc["median_s"])
                res["libs"][lib.name]["mixed"] = mx
                save()
                fb = [mb for mb in (1, 32) if mb in batches]
                po = {k: pass_overlap(lib, torch, cases, k, fb, a.reps, log) for k in ks}
                res["libs"][lib.name]["pass_overlap"] = po
                fq = res["libs"][lib.name]["frontier"]
                res["points"] += points(lib, fq, bd, os.environ.get("RESEARCH_RUN_ID"), props.name,
                                        res["libs"][lib.name].get("clock", {}).get("sm_mhz"), po, mx)
                for mb, row in bd.items():
                    log(f"layer-weighted {lib.name} M={mb}: base {row['base']:.2f}, " + ", ".join(
                        f"k{k} total {row[k]['y_total']:.2f} (decode {row[k]['y_decode']:.2f}, tag {row[k]['y_tag']:.2f}"
                        f", commit {row[k]['y_commit']:.3f}; per linear {row[k]['y_commit_linear']:.2f})" for k in ks))
            save()
    (out_dir / "points.json").write_text(json.dumps(res["points"], indent=1))
    save()
    write_result(res, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
