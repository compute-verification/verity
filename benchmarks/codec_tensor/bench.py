"""The wide-word codec's benchmark on one GPU: `wt.cu`'s libraries (`--lib PATH`, one per field) gated against
`wide.py`, then timed. Writes `codec-tensor.json` and `bench.log` under `--out`, and, with
`RESEARCH_RUN_DIR` set, its `result.json` there.

Nothing is timed before its gate passed, in this order:
- `gate`: per field and k, device encode equals `wide.encode_word` on random and edge payloads, and both decodes (CUDA
  cores, tensor cores) give the payloads back; the tensor-core squaring equals a^2 mod p on random and edge inputs; the
  plain fragment layout built on the device equals `codec.plain_fragments` (`benchmarks/word_codec`);
- `read`: the device read rate R over a store of `--store-gib`, the best of a few grids;
- `rates`: compute-only throughputs: the squaring on CUDA cores (ILP 1, 2) and on tensor cores (1 or 2 words a warp),
  both at once in one CTA (`wt_mix`), decode and encode chains on CUDA cores;
- `decode`: a store of `--store-gib` encoded on the device, its decode on CUDA and tensor cores equal to the payloads,
  then each decode timed (read, decode, XOR);
- `gemm`: at a Llama-3.1 model's linears (`--model`, 70B or 8B) and batch 1-32, the fused kernel's partial sums equal
  the plain kernel's bit for bit, the plain kernel's within FP32 rounding of torch's; then interleaved reps (each on cold
  weights, a ring of copies past the L2, queued behind a spin kernel) of cuBLASLt FP8 (`torch._scaled_mm`), the plain
  FP8 kernel, the plain kernel with `--mma-knobs` extra int8 MMAs a tile or `--imad-knobs` extra IMADs a tile a thread
  (the budget a decode on tensor or CUDA cores has before it slows the memory-bound step), and the fused decode at k = 1
  (and `--gemm-ks`); per batch, the layer's time over native with each linear at its best arm. With `--sketch` (a
  WT_RECEIPT build at full slot), the fused decode also runs with two sketch warps of its own (T = -2) that sketch
  each of the store's 4 KiB blocks at k_s = 16 or 64 and hash the sketch into its sketch-sha512 receipt
  (`sketch_receipt`), gated on every vector's sketch (a float32 GPU reference with TF32 off, exact), every receipt,
  and the commit's leaves and root (reader.cu against the driver's tree). Before its GEMMs, `gate_pass` checks the
  per-pass key (red team F29: `wt_pass` derives pass j's seed, signs and receipt midstate on the device from
  root_(j−1)), and once a linear one such launch runs under a pass's key and is gated at that pass's seed
  (`pass_keyed`).
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parents[1]))

from benchmarks.word_codec import cuda_lib  # noqa: E402
import sketch_receipt as SR  # noqa: E402
import sketching as SKT  # noqa: E402
import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/codec-tensor/bench/seed").digest()
TAG_SEED = hashlib.sha256(b"verity/codec-tensor/bench/tag-seed").digest()   # the window's sigma
RECEIPT_SEED = SR.rb.seed(TAG_SEED, 0, bytes(64))    # a WT_RECEIPT build's window seed (sketch-sha512's are 64 bytes)
STORE_KEY = hashlib.sha256(b"verity/codec-tensor/bench/store-key").digest()  # the receipt tree's salt key
COMMIT_GATE_BLOCKS = 4096      # sketch-sha512's commit gate: the driver's tree over a launch's first receipts
READER = None                  # a WT_RECEIPT build's commit: reader.Reader (reader.cu), set by --reader-lib
PIPE_OPS = ("imad_wide", "imad", "iadd", "mac")   # wt_chain 5's ops
MODELS = {   # Llama-3.1's layers (for the tree's share of a step) and linears (N, K)
    "8b": (32, {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}),
    "70b": (80, {"qkv": (10240, 8192), "o": (8192, 8192), "gate_up": (57344, 8192), "down": (8192, 28672)}),
}
BATCHES = (1, 2, 4, 8, 16, 32)
L2_DEFEAT_BYTES = 1 << 30
SPIN_CYCLES = 20_000_000


def word_codec():
    spec = importlib.util.spec_from_file_location("wc_codec", HERE.parent / "word_codec" / "codec.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_v, _i, _q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
_FUSED_SKETCH = [_i, _v, _v, _v, _i, _i, _i, _q, _i, _i, _i, _i, _v, _v, _q, _i, _v, _v, _i, _i, _v, _v]
WT = {   # wt.cu's functions every build exports (`cuda_lib.load`)
    "wt_set_key": [_v, _i, _v, _v],
    "wt_set_mask": [_v],
    "wt_encode": [_v, _v, _q, _q, _i, _i, _v],
    "wt_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "wt_decode_rows": [_v, _v, _q, _q, _i, _i, _v, _i, _v],
    "wt_tc_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "wt_read": [_v, _q, _v, _i, _v],
    "wt_tc_square_gate": [_v, _v, _i, _v],
    "wt_chain": [_i, _i, _i, _q, _v, _i, _v],
    "wt_mix": [_q, _q, _i, _v, _i, _v],
    "wt_gemm": [_i, _v, _v, _v, _i, _i, _i, _i, _i, _i, _i, _i, _v, _v, _v],
    "wt_fused": [_i, _v, _v, _v, _i, _i, _i, _q, _i, _i, _i, _i, _v, _v, _v],
    "wt_fused_sketch": _FUSED_SKETCH,
    "wt_set_pass": [_v, _i, _i],
    "wt_pass": [ctypes.c_uint, _v, _v, _i, _v],
    "wt_sign_frags": [_v, _v, _i, _v],
    "wt_pass_state": [_v],
    "wt_pass_ptrs": [_v],
    "wt_sketch_vectors": ([_i, _i, _i, _i], _q),
    "wt_set_receipt": [_v],
    "wt_set_dbg": [_v],
    "wt_spin": [ctypes.c_longlong, _v],
    "wt_empty": [_i, _v],
}
WT_BUILT = {"wt_fused_sketch_bf": _FUSED_SKETCH, "wt_prim": [_i, _v, _v, _v, _i, _v]}   # not in builds from before them


class Lib:
    def __init__(self, path: str) -> None:
        self.path, self.so = path, cuda_lib.load(path, WT, WT_BUILT)
        so = self.so
        cs = (ctypes.c_int * 18)()
        so.wt_constants(cs)
        (self.n, self.nl, self.s, self.pb, self.tw, self.u, self.c, self.wpb, self.tc_mmas, self.tc_bytes, self.kmax,
         self.sb, self.fused_warps, self.fused_span, self.fused_most, self.receipt, mask, fold6) = list(cs)
        self.mask, self.fold6 = bool(mask), fold6   # fold6: 0 the column square, 1 FOLD 6, 2 FOLD 6 folded as made
        self.field = next(f for f in W.FIELDS.values() if f.n == self.n)
        f = self.field
        if (self.s, self.pb, self.tw, self.c, self.sb) != (f.word_bytes, f.payload_bytes, W.TASK_WORDS[f.name], f.c,
                                                             W.TASK_WORDS[f.name] * f.word_bytes):
            raise RuntimeError(f"{path}: constants {list(cs)} are not wide.py's {f}")
        self.name = (f.name + ("-masked" if self.mask else "") + (f"-fold6{'' if self.fold6 == 1 else 'lean'}" if self.fold6 else "")
                     + ("-sha512" if self.receipt else ""))
        self.tc = self.tc_mmas > 0
        self.tag_bytes = 64                                   # a sketch vector's receipt
        self.jbase = 0                                        # sketch-sha512: vector v of a launch is the store's block v
        self.pass_rows = SR.ROWS                              # wt_set_pass's k_s, which wt_sign_frags lays out

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


def queued(lib: Lib, torch, fn) -> tuple[float, float]:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("wt_spin", SPIN_CYCLES, stream(torch))
    a.record()
    h = time.perf_counter()
    fn()
    h = time.perf_counter() - h
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3, h


def sm_clock(lib: Lib, torch, cycles: int = 20_000_000) -> float:
    """The SM clock (Hz) now: a one-thread spin of `cycles` clock64 ticks over its event time."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    lib.call("wt_spin", cycles, stream(torch))
    b.record()
    b.synchronize()
    return cycles / (a.elapsed_time(b) / 1e3)


def limbs_bytes(x: int, nl: int) -> bytes:
    return x.to_bytes(4 * nl, "little")


def to_dev(torch, data: bytes):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()


def lib_key(lib: Lib, k: int) -> W.Key:
    """The bench's k-round key at SEED, masked when the library is (WT_MASK)."""
    return W.key(lib.field.name, SEED, k, mask=lib.mask)


def set_key(lib: Lib, torch, key: W.Key) -> None:
    """The key into the library's constant memory, from host buffers."""
    if (key.mask is not None) != lib.mask:
        raise ValueError(f"{lib.name}: a {'masked' if key.mask is not None else 'plain'} key")
    c = ctypes.create_string_buffer(b"".join(limbs_bytes(x, lib.nl) for x in key.c))
    tau = ctypes.create_string_buffer(limbs_bytes(key.tau, lib.nl))
    p = ctypes.create_string_buffer(limbs_bytes(lib.field.p, lib.nl))
    lib.call("wt_set_key", ctypes.addressof(c), key.rounds, ctypes.addressof(tau), ctypes.addressof(p))
    if lib.mask:
        m = ctypes.create_string_buffer(limbs_bytes(key.mask, lib.nl))
        lib.call("wt_set_mask", ctypes.addressof(m))


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_codec(lib: Lib, torch, k: int, n_words: int, log) -> dict:
    f = lib.field
    key = lib_key(lib, k)
    set_key(lib, torch, key)
    rng = random.Random(7 + k + f.n)
    top = f.max_payload
    base = (1 << 40) + 12345
    payloads = [0, 1, 2, top, top - 1, 1 << (8 * f.payload_bytes - 1)]
    if lib.mask and k > 1:   # σ_K's rare branch at word len(payloads) on
        payloads += W.sigma_rare_payloads(key, base + len(payloads))[:1]
        payloads += W.sigma_rare_payloads(key, base + len(payloads))[-1:]
    payloads += [rng.randrange(top + 1) for _ in range(n_words - len(payloads))]
    m = to_dev(torch, b"".join(limbs_bytes(x, lib.nl) for x in payloads))
    store = torch.zeros(n_words * f.word_bytes, dtype=torch.uint8, device="cuda")
    out = torch.zeros(n_words * 4 * lib.nl, dtype=torch.uint8, device="cuda")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    lib.call("wt_encode", ptr(m), ptr(store), n_words, base, k, 64, stream(torch))
    torch.cuda.synchronize()
    sb = bytes(store.cpu().tolist())
    want = b"".join(W.word_bytes(W.encode_word(x, base + i, key), f) for i, x in enumerate(payloads))
    bad_enc = [i for i in range(n_words) if sb[i * f.word_bytes:(i + 1) * f.word_bytes]
               != want[i * f.word_bytes:(i + 1) * f.word_bytes]]
    res = {"words": n_words, "encode_mismatch": bad_enc[:8], "encode_ok": not bad_enc}
    for fn, name in (("wt_decode", "cuda"), ("wt_tc_decode", "tc")):
        if name == "tc" and not (lib.tc and (k == 1 or not lib.mask)):   # tc_decode has no σ_K
            continue
        out.zero_()
        lib.call(fn, ptr(store), ptr(out), n_words, base, k, ptr(xr), 32, stream(torch))
        torch.cuda.synchronize()
        res[f"decode_{name}_ok"] = bool(torch.equal(out, m))
    res["ok"] = all(v for kk, v in res.items() if kk.endswith("_ok"))
    log(f"gate {lib.name} k={k}: {'ok' if res['ok'] else 'FAIL'} {res}")
    return res


def gate_tc_square(lib: Lib, torch, n_words: int, log) -> dict:
    f = lib.field
    rng = random.Random(3 + f.n)
    xs = [0, 1, 2, f.p - 1, f.p - 2, (1 << (f.n - 1)), (1 << (f.n - 1)) - 1]
    xs += [rng.randrange(f.p) for _ in range(n_words - len(xs))]
    inp = to_dev(torch, b"".join(limbs_bytes(x, lib.nl) for x in xs))
    out = torch.zeros_like(inp)
    lib.call("wt_tc_square_gate", ptr(inp), ptr(out), n_words, stream(torch))
    torch.cuda.synchronize()
    ob = bytes(out.cpu().tolist())
    nb = 4 * lib.nl
    bad = [i for i, x in enumerate(xs) if int.from_bytes(ob[nb * i:nb * (i + 1)], "little") != x * x % f.p]
    res = {"words": n_words, "mismatch": bad[:8], "ok": not bad}
    log(f"gate {lib.name} tc square: {'ok' if res['ok'] else 'FAIL'} ({len(bad)} bad)")
    return res


def plain_layout(w):
    n, kd = w.shape
    w8 = w.view(n // 16, 2, 8, kd // 512, 16, 2, 4, 4)
    return w8.permute(0, 3, 4, 2, 6, 5, 1, 7).contiguous().view(-1)


def gate_layout(torch, log) -> dict:
    C = word_codec()
    rng = random.Random(8)
    n, kd = 64, 1024
    w = bytes(rng.getrandbits(8) for _ in range(n * kd))
    wt = torch.tensor(list(w), dtype=torch.uint8, device="cuda").view(n, kd)
    ok = bytes(plain_layout(wt).cpu().tolist()) == C.plain_fragments(w, n, kd)
    log(f"gate layout: plain {ok}")
    return {"ok": ok}


# --- the store of whole tasks (`wide.encode_tasks`) on the device ------------------------------------------------------

def task_payload_limbs(lib: Lib, torch, plain):
    """plain (a multiple of 8,192 bytes) as the payload limbs of its tasks' words (TW words a task, NL u32 each)."""
    f = lib.field
    tasks = plain.numel() // W.TASK_BYTES
    pad = torch.zeros(tasks, lib.tw * f.payload_bytes, dtype=torch.uint8, device="cuda")
    pad[:, :W.TASK_BYTES] = plain.view(tasks, W.TASK_BYTES)
    m = torch.zeros(tasks * lib.tw, 4 * lib.nl, dtype=torch.uint8, device="cuda")
    m[:, :f.payload_bytes] = pad.view(tasks * lib.tw, f.payload_bytes)
    return m.view(-1)


def encode_store(lib: Lib, torch, m, k: int, sms: int) -> tuple:
    n = m.numel() // (4 * lib.nl)
    store = torch.empty(n * lib.s, dtype=torch.uint8, device="cuda")
    t = timed(torch, lambda: lib.call("wt_encode", ptr(m), ptr(store), n, 0, k, sms * 8, stream(torch)), 1, warm=0)
    return store, t[0]


# --- read, rates, standalone decode ------------------------------------------------------------------------------------

def read_rate(lib: Lib, torch, store, sms: int, reps: int) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    best = None
    for mult in (4, 8, 16):
        ts = timed(torch, lambda: lib.call("wt_read", ptr(store), store.numel(), ptr(xr), sms * mult, stream(torch)), reps)
        med = statistics.median(ts)
        if best is None or med < best[1]:
            best = (mult, med, ts)
    return {"bytes": store.numel(), "grid_per_sm": best[0], "median_s": best[1], "times_s": best[2],
            "gbps": store.numel() / best[1] / 1e9}


def calibrated(torch, run, unit: int, target_s: float, reps: int) -> dict:
    """run(steps) timed at the step count that takes about target_s; unit = items a step."""
    steps = 1
    while True:
        t = statistics.median(timed(torch, lambda: run(steps), 1))
        if t > target_s / 8 or steps > 1 << 24:
            break
        steps *= 4
    steps = max(1, int(steps * target_s / max(t, 1e-6)))
    ts = timed(torch, lambda: run(steps), reps)
    med = statistics.median(ts)
    return {"steps": steps, "items": steps * unit, "median_s": med, "times_s": ts, "per_s": steps * unit / med}


def rates(lib: Lib, torch, sms: int, reps: int, log) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    st = stream(torch)
    out = {"square_cuda": {}, "square_tc": {}, "mix": {}, "decode": {}, "encode": {}}
    for ilp in (1, 2):
        for mult in (2, 4, 8):
            grid = sms * mult
            r = calibrated(torch, lambda s: lib.call("wt_chain", 0, 1, ilp, s, ptr(xr), grid, st), grid * 128 * ilp,
                           0.08, reps)
            out["square_cuda"][f"ilp{ilp}_g{mult}"] = r
    best_c = max(out["square_cuda"].values(), key=lambda r: r["per_s"])
    log(f"rates {lib.name}: CUDA squarings/s best {best_c['per_s']:.3e} ("
        + ", ".join(f"{k} {r['per_s']:.2e}" for k, r in out["square_cuda"].items()) + ")")
    if lib.field.full_slot:
        out["square_cuda_column"] = {}
        for mult in (2, 4, 8):
            grid = sms * mult
            out["square_cuda_column"][f"ilp1_g{mult}"] = calibrated(
                torch, lambda s: lib.call("wt_chain", 6, 1, 1, s, ptr(xr), grid, st), grid * 128, 0.08, reps)
        best = max(out["square_cuda_column"].values(), key=lambda r: r["per_s"])
        log(f"rates {lib.name}: CUDA column squarings (sqr_fold, the fused decoders') /s best {best['per_s']:.3e}")
    for k in (1, 2):
        grid = sms * 4
        out["decode"][k] = calibrated(torch, lambda s: lib.call("wt_chain", 1, k, 1, s, ptr(xr), grid, st), grid * 128,
                                      0.08, reps)
        out["encode"][k] = calibrated(torch, lambda s: lib.call("wt_chain", 2, k, 1, s, ptr(xr), grid, st), grid * 128,
                                      0.3, 3)
        log(f"rates {lib.name} k={k}: decode {out['decode'][k]['per_s']:.3e} words/s, encode "
            f"{out['encode'][k]['per_s']:.3e} words/s, r = {out['decode'][k]['per_s'] / out['encode'][k]['per_s']:.0f}")
    if lib.tc:
        for ilp in (1, 2):
            for mult in (2, 4, 8, 12):
                grid = sms * mult
                r = calibrated(torch, lambda s: lib.call("wt_chain", 3, 1, ilp, s, ptr(xr), grid, st), grid * 4 * ilp,
                               0.08, reps)
                out["square_tc"][f"ilp{ilp}_g{mult}"] = r
        best_t = max(out["square_tc"].values(), key=lambda r: r["per_s"])
        log(f"rates {lib.name}: TC squarings/s best {best_t['per_s']:.3e} ("
            + ", ".join(f"{k} {r['per_s']:.2e}" for k, r in out["square_tc"].items()) + ")")
        for cw in (4, 6, 7):
            for mult in (1, 2):
                grid = sms * mult
                rc = calibrated(torch, lambda s: lib.call("wt_mix", s, 0, cw, ptr(xr), grid, st), grid * cw * 32, 0.08,
                                reps)
                if cw < 8:
                    rt = calibrated(torch, lambda s: lib.call("wt_mix", 0, s, cw, ptr(xr), grid, st),
                                    grid * (8 - cw), 0.08, reps)
                    sc = rc["steps"]
                    stc = max(1, int(rt["per_s"] / (grid * (8 - cw)) * rc["median_s"]))
                    ts = timed(torch, lambda: lib.call("wt_mix", sc, stc, cw, ptr(xr), grid, st), reps)
                    med = statistics.median(ts)
                    items_c, items_t = sc * grid * cw * 32, stc * grid * (8 - cw)
                    both = {"sc": sc, "stc": stc, "median_s": med, "times_s": ts, "cuda_per_s": items_c / med,
                            "tc_per_s": items_t / med, "per_s": (items_c + items_t) / med}
                else:
                    rt, both = None, None
                out["mix"][f"cw{cw}_g{mult}"] = {"cuda_alone": rc, "tc_alone": rt, "both": both}
                if both:
                    log(f"rates {lib.name} mix cw={cw} g={mult}: CUDA alone {rc['per_s']:.3e}, TC alone "
                        f"{rt['per_s']:.3e}, both {both['per_s']:.3e} ({both['cuda_per_s']:.2e} + {both['tc_per_s']:.2e})")
    return out


def standalone_decode(lib: Lib, torch, store_bytes: int, sms: int, reps: int, log) -> dict:
    f = lib.field
    n = store_bytes // f.word_bytes
    gen = torch.Generator(device="cuda").manual_seed(11)
    m = torch.randint(0, 256, (n, 4 * lib.nl), dtype=torch.uint8, device="cuda", generator=gen)
    m[:, f.payload_bytes:] = 0
    m = m.view(-1)
    out = {}
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    for k in (1, 2):
        set_key(lib, torch, lib_key(lib, k))
        store, enc_s = encode_store(lib, torch, m, k, sms)
        back = torch.empty_like(m)
        entry = {"words": n, "store_bytes": n * f.word_bytes, "setup_encode_s": enc_s}
        for fn, name, mults in (("wt_decode", "cuda", (4, 8, 16)), ("wt_tc_decode", "tc", (4, 8, 16, 24))):
            if name == "tc" and not (lib.tc and (k == 1 or not lib.mask)):
                continue
            back.zero_()
            lib.call(fn, ptr(store), ptr(back), n, 0, k, ptr(xr), sms * 8, stream(torch))
            torch.cuda.synchronize()
            if not torch.equal(back, m):
                entry[name] = {"ok": False}
                log(f"decode {lib.name} {name} k={k}: FAIL")
                continue
            best = None
            for mult in mults:
                ts = timed(torch, lambda: lib.call(fn, ptr(store), 0, n, 0, k, ptr(xr), sms * mult, stream(torch)), reps)
                med = statistics.median(ts)
                if best is None or med < best[1]:
                    best = (mult, med, ts)
            entry[name] = {"ok": True, "grid_per_sm": best[0], "median_s": best[1], "times_s": best[2],
                           "gbps_stored": n * f.word_bytes / best[1] / 1e9}
            log(f"decode {lib.name} {name} k={k}: {entry[name]['gbps_stored']:.0f} GB/s stored")
        entry["ok"] = all(v["ok"] for kk, v in entry.items() if kk in ("cuda", "tc"))
        out[k] = entry
        del store, back
        torch.cuda.empty_cache()
    return out


# --- the GEMM ----------------------------------------------------------------------------------------------------------

def tsteps(lib: Lib, torch, key: W.Key, decoders: int, groups: int, nchunks: int):
    """The fused kernel's tweak steps: a lane's next unit is 32 D words on in the CTA's sequence, its task A or A + 1
    on (the CTA's consecutive tasks are groups * nchunks apart in the store)."""
    p, tw = lib.field.p, lib.tw
    adv, rem = divmod(32 * decoders, tw)
    gq = groups * nchunks
    steps = (adv * gq * tw + rem, (adv + 1) * gq * tw + rem - tw)
    return to_dev(torch, b"".join(limbs_bytes(key.tau * d % p, lib.nl) for d in steps))


SMEM_PER_BLOCK = 101376          # sm_120's opt-in dynamic shared memory a CTA


def parse(spec: str) -> list[tuple[int, ...]]:
    return [tuple(int(v) for v in c.split("x")) for c in spec.split(",") if c]


def fused_cfgs(lib: Lib, spec: str, sketch: str = "", sketch_t: int = 8) -> list[tuple[int, int, int, int]]:
    """(D, stages, T, k_s): DECODERSxSTAGES (untagged, T = -1, k_s = 0) or, in a WT_RECEIPT build at full slot,
    DECODERSxSTAGESx-2xK_S (two sketch warps of their own beside D decoders, which sketch each 4 KiB block at k_s = 16
    or 64 and hash the sketch into its sketch-sha512 receipt, stages < 2 sketch_t), each checked against
    wt_fused_sketch's bounds. At M > 16 and k <= 2 the kernel has two MMA warps (wt.cu's fused_mma_warps) and a
    producer, so D + 3 warps untagged and D + 5 beside the sketch warps must fit; elsewhere one MMA warp and a producer
    beside the sketch warps (D + 4), and untagged one MMA warp that copies the tasks itself (fused_producers), D + 1.
    `auto`: untagged, the most decoder warps the field's registers allow at 16 warps at 4 and 6 stages fewer than fit
    at M = 32 (fewer stages leave more L1) and at all of them, 1 fewer at 4 and 6 fewer stages, and 2 fewer warps at 2
    fewer stages (p1536's 10 decoders starve at 8 stages: 8 x 8 and 10 x 10 are its best); with the sketch warps, for
    each k_s in `sketch`, beside 1, 3 and 4 decoders fewer than the most, at 0 and 2 fewer stages than fit beside their
    buffers. Where the field's kernel takes 20 warps (wt.cu's fused_warps: full slot, untagged or beside the sketch
    warps), also 16 decoders, four a sub-partition, untagged at 2 and 4 fewer stages and beside the sketch warps (and
    15, M = 32's) at 0 and 2 fewer, and 19 untagged at 4 fewer. gemm_case skips each config at an M and k whose kernel
    doesn't take its warps (fused_fits)."""
    sk_bytes = lambda ks: 4 * 2 * 4 * 2 * SR.SK_G * (ks + 4) if ks else 0  # noqa: E731   wt.cu's sk_buf_ints
    smax = lambda ks=0: (SMEM_PER_BLOCK - 4 * 8 * 528 - 32 - sk_bytes(ks)) // (lib.sb + 48)  # noqa: E731
    dmax, dwide = lib.fused_warps - 1, lib.fused_most - 1
    need = lambda d: -(-32 * d // lib.tw)  # noqa: E731
    sketched = lib.receipt and lib.pb == lib.s
    if spec == "auto":
        cfgs = [(dmax, max(need(dmax), smax() - f), -1) for f in (4, 6, 0)]
        cfgs += [(dmax - 1, max(need(dmax - 1), smax() - f), -1) for f in (4, 6) if dmax - 1 >= lib.fused_span]
        if dmax - 2 >= lib.fused_span:
            cfgs.append((dmax - 2, max(need(dmax - 2), smax() - 2), -1))
        wide = {dwide - 3, dwide - 4} if dwide > dmax else set()
        if dwide > dmax:
            cfgs += [(dwide - 3, max(need(dwide - 3), smax() - f), -1) for f in (2, 4)]
            cfgs.append((dwide, max(need(dwide), smax() - 4), -1))
        for ks in (int(v) for v in sketch.split(",") if v and sketched):
            cfgs += [(d, min(2 * sketch_t - 1, max(need(d), smax(ks) - f)), -2, ks)
                     for d in sorted({dmax - 1, dmax - 3, dmax - 4} | wide, reverse=True)
                     if d >= lib.fused_span
                     for f in (0, 2)]
    else:
        cfgs = parse(spec)
    out = []
    for c in cfgs:
        if len(c) not in (2, 3, 4):
            raise SystemExit(f"fused {c}: DxS, DxSx-1 or DxSx-2xK_S")
        d, st, t, ks = tuple(c) + (-1, 0)[len(c) - 2:]
        if not (lib.fused_span <= d <= dwide and need(d) <= st <= smax(ks) and t in (-1, -2) and ks in (0, 16, 64)
                and (t == -2) == (ks != 0) and (ks == 0 or (sketched and st < 2 * sketch_t))):
            raise SystemExit(f"{lib.name}: fused {c} is outside {lib.fused_span} <= D <= {dwide}, "
                             f"ceil(32 D / TW) <= stages <= {smax(ks)}, and T = -1 untagged or T = -2 at k_s 16 or 64 "
                             f"(two sketch warps: a WT_RECEIPT build at full slot) with stages < {2 * sketch_t}")
        if (d, st, t, ks) not in out:
            out.append((d, st, t, ks))
    return out


def sketch_warps(cfg) -> int:
    return 2 if cfg[2] == -2 else 0


def sign_frags(lib: Lib, torch, signs, sketch_t: int, st: int):
    """The sketch warps' A (wt_sign_frags) of the sign words on the device at wt_set_pass's k_s: selectors at 16
    (sketch_receipt.sign_frags), bit planes at 64 (sign_planes)."""
    frags = torch.empty(SR.PLANE_WORDS if lib.pass_rows == SR.ROWS64 else SR.FRAG_WORDS, dtype=torch.int32,
                        device="cuda")
    lib.call("wt_sign_frags", ptr(signs), ptr(frags), sketch_t, st)
    return frags


def pass_signs(lib: Lib, bits: np.ndarray, sketch_t: int) -> np.ndarray:
    """The pass's sign words of S's bits (k × n, 1 is −1) as wt_pass writes them: S as it is
    (sketch_receipt.sign_words)."""
    return SR.sign_words(np.packbits(bits, axis=1, bitorder="little"))


def pass_frags(lib: Lib, bits: np.ndarray, sketch_t: int) -> np.ndarray:
    """What wt_sign_frags makes of pass_signs: the sketch warps' A (selectors at 16 rows, bit planes at 64)."""
    packed = np.packbits(bits, axis=1, bitorder="little")
    return SR.sign_planes(packed) if bits.shape[0] == SR.ROWS64 else SR.sign_frags(packed)


def store_blocks_torch(torch, store):
    """A WT_RECEIPT build's receipts' vectors on the device: the store's 4 KiB blocks in store order (store-order/v0,
    sketch_receipt.gemm_blocks), as int8."""
    return store.view(-1, SR.BLOCK).view(torch.int8)


def fused_fits(lib: Lib, k: int, mb: int, n: int, kd: int, cfg, sketch_t: int) -> bool:
    """wt_fused_sketch takes cfg at M = mb (two sketch warps beside D decoders need D + 4 warps, and D + 5 past
    M = 16 at k <= 2)."""
    launched = ctypes.c_int(0)
    return lib.so.wt_fused_sketch(k, None, None, None, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, None, None,
                                  0, cfg[3], None, None, sketch_t, sketch_warps(cfg), ctypes.byref(launched),
                                  None) == 0


def fused_name(k: int, cfg) -> str:
    return f"fused_k{k}_d{cfg[0]}s{cfg[1]}" + ("w2" if cfg[2] == -2 else "") + (f"sk{cfg[3]}" if cfg[3] else "")


def host_bytes(t) -> bytes:
    h = t.contiguous().cpu()
    return ctypes.string_at(h.data_ptr(), h.numel() * h.element_size())


def digests(mem: bytes) -> list[bytes]:
    """The device's SHA-512 digests (8 state words each, as u64 in memory) as bytes."""
    return [b"".join(int.from_bytes(mem[i + 8 * w:i + 8 * w + 8], "little").to_bytes(8, "big") for w in range(8))
            for i in range(0, len(mem), 64)]


def root_words(root: bytes) -> bytes:
    """A 64-byte digest as the device holds a tree node: 8 SHA-512 state words, u64 little-endian in memory."""
    return b"".join(int.from_bytes(root[8 * w:8 * w + 8], "big").to_bytes(8, "little") for w in range(8))


def set_pass(lib: Lib, sketch_t: int, sigma: bytes = TAG_SEED, rows: int = SR.ROWS) -> int:
    """wt_set_pass at the window's σ for sketch-sha512 at k_s = rows (16 or 64) over tiles of sketch_t tasks (a 4 KiB
    block at sketch_t = 8); returns n."""
    n = SKT.vector_bytes(lib.sb, sketch_t)
    sg = ctypes.create_string_buffer(sigma)
    lib.call("wt_set_pass", ctypes.addressof(sg), rows, n)
    lib.pass_rows = rows
    return n


def pass_state(lib: Lib) -> dict:
    """The last pass's seed (64 bytes), K_S and receipt midstate."""
    out = ctypes.create_string_buffer(160)
    lib.call("wt_pass_state", ctypes.addressof(out))
    b = out.raw
    return {"seed": b[:64], "k_s": b[64:96], "mid": b[96:160]}


def window_seed(lib: Lib) -> bytes:
    return RECEIPT_SEED


def pass_seed(lib: Lib, j: int, root: bytes) -> bytes:
    """Pass j's seed from root_(j−1): the driver's receipts.seed."""
    return SR.rb.seed(TAG_SEED, j, root)


def sketch_bits(lib: Lib, seed: bytes, k: int, n: int) -> np.ndarray:
    """sketch-sha512's S at `seed`, k × n bits (1 is −1)."""
    if k not in SR.rb.ROWS or n != SR.BLOCK:
        raise ValueError(f"sketch-sha512's S is k_s in {SR.rb.ROWS} x {SR.BLOCK}, not {k} x {n}")
    return np.unpackbits(SR.sign_bits(seed, k), axis=1, bitorder="little")


def sketch_derive(lib: Lib, seed: bytes, n: int, rows: int = SR.ROWS) -> bytes:
    return SR.rb.derive(seed, SR.rb.SIGN_DOMAIN, {"tag": SR.rb.SKETCH_SHA512, "k": rows, "n": n}).value


def receipt_mid(lib: Lib, seed: bytes) -> bytes:
    """sketch-sha512's prefix midstate at `seed` as the host computes it (wt_set_receipt); leaves set_receipt_prefix's."""
    pre = ctypes.create_string_buffer(SR.rb.sketch_prefix(seed))
    lib.call("wt_set_receipt", ctypes.addressof(pre))
    mid = pass_state(lib)["mid"]
    set_receipt_prefix(lib)
    return mid


def receipt_commit_gate(torch, tags_dev, nvec: int, seed: bytes) -> dict:
    """sketch-sha512's commit over a launch's first receipts (at most COMMIT_GATE_BLOCKS, the whole plan of those
    blocks): reader.cu's leaves, pads and tree (READER) against the driver's `receipts.tree` at `seed`, its salts under
    STORE_KEY. reader.py's own gate covers partial plans and openings."""
    import reader as RD

    n = min(nvec, COMMIT_GATE_BLOCKS)
    plan = SR.rb.Plan.whole(n)
    READER.begin(seed, plan, STORE_KEY)
    ys = tags_dev.view(torch.int64)[:8 * n].view(n, 8)
    levels = READER.levels(READER.commit(ys, 0, READER.leaves), 0, pad=True)
    host = ys.cpu().numpy()
    got = [RD.digest(y) for y in host]
    tree = SR.rb.tree(seed, plan, got, [RD.store_salt(STORE_KEY, seed, c) for c in range(READER.leaves)])
    root = RD.digest(levels[-1][0])
    return {"blocks": n, "leaves": READER.leaves,
            "leaves_ok": [RD.digest(x) for x in levels[0].cpu().numpy()] == list(tree._levels[0]),
            "root_ok": root == tree.commitment.root, "root": root.hex()}


def gate_pass(lib: Lib, torch, sketch_t: int, log, trials: int = 4, rows: int = SR.ROWS) -> dict:
    """wt_pass (red team F29: pass j's receipt keyed by s_j = pass_seed(σ, j, root_(j−1))) against Python at random
    j and roots: sketch-sha512's s_j, K_S, signs and receipt prefix (its midstate against the host's of
    receipts.sketch_prefix(s_j)) at k_s = rows, the signs as sign_words and their A as sign_frags (16 rows) or
    sign_planes (64)."""
    n, st = set_pass(lib, sketch_t, rows=rows), stream(torch)
    signs = torch.empty(128 * rows, dtype=torch.int32, device="cuda")
    rng = random.Random(29)
    trials_out = []
    for t in range(trials):
        j = (0, (1 << 32) - 1)[t] if t < 2 else rng.randrange(1 << 32)
        root = rng.randbytes(64)
        lib.call("wt_pass", j, ptr(to_dev(torch, root_words(root))), ptr(signs), sketch_t, st)
        frags = sign_frags(lib, torch, signs, sketch_t, st)
        torch.cuda.synchronize()
        got = pass_state(lib)
        seed = pass_seed(lib, j, root)
        bits = sketch_bits(lib, seed, rows, n)
        row = {"j": j, "seed": got["seed"] == seed, "k_s": got["k_s"] == sketch_derive(lib, seed, n, rows),
               "signs": bool(np.array_equal(signs.cpu().numpy().view(np.uint32), pass_signs(lib, bits, sketch_t))),
               "frags": bool(np.array_equal(frags.cpu().numpy().view(np.uint32), pass_frags(lib, bits, sketch_t))),
               "mid": got["mid"] == receipt_mid(lib, seed)}
        trials_out.append(row)
    ok = all(all(v for a, v in r.items() if a != "j") for r in trials_out)
    log(f"pass gate {lib.name} k_s={rows}: {'ok' if ok else 'FAIL'} over {trials} (j, root): "
        f"{trials_out if not ok else ''}")
    return {"ok": ok, "trials": trials_out, "n": n, "k_s": rows}


def mismatch(pf, pp, nch: int, mb: int, n: int, groups: int) -> dict:
    """Where a fused run's partial sums differ from the plain kernel's: the tasks (K chunk kc, row block g; the CTA
    (kc, g mod groups) takes it as its task g // groups) and how many of their elements."""
    f, p = pf.view(nch, mb, n), pp.view(nch, mb, n)
    d = (f != p).any(1).view(nch, n // 16, 16).sum(2)
    idx = d.nonzero().tolist()
    nbs = n // 16

    def task(kc, g):
        fb, out = f[kc, :, 16 * g:16 * g + 16], {"kc": kc, "g": g, "cta_g0": g % groups, "i": g // groups,
                                                  "cols": int(d[kc, g])}
        out["zeros"] = int((fb == 0).sum())
        out["equals_task"] = [j for j in range(-8, 9) if j and 0 <= g + groups * j < nbs
                              and torch_equal(fb, p[kc, :, 16 * (g + groups * j):16 * (g + groups * j) + 16])]
        return out

    ctas = {}
    for kc, g in idx:
        ctas.setdefault((kc, g % groups), []).append(g // groups)
    return {"tasks": len(idx), "elements": int((f != p).sum()), "per_kc": [int(v) for v in (d > 0).sum(1).tolist()],
            "ctas": len(ctas), "per_cta": [{"kc": kc, "g0": g0, "i": v} for (kc, g0), v in list(ctas.items())[:6]],
            "first": [task(kc, g) for kc, g in idx[:6]]}


def torch_equal(a, b) -> bool:
    return bool((a == b).all())


def dump_words(lib: Lib, dbg, plain, store, key, gq: int, stages: int) -> dict:
    """What the fused MMA warp consumed (a WT_DEBUG_DUMP build's dbg, plain fragment bytes a task) where it differs from
    the plain store: per bad word (task q, word w), its differing 16-byte chunks and whether its bytes are the stored
    word, zero, the payload of another word (the CTA's tasks q + j gq that share or neighbour its stage) or the decode
    of its stored word under another tweak index; and per differing 16-byte chunk (a lane's fragment) whether it is
    zero or those tasks' plain or stored bytes at its offset."""
    f, tb = lib.field, W.TASK_BYTES
    pb, s, tw = f.payload_bytes, lib.s, lib.tw
    dt, pt = dbg.view(-1, tb), plain.view(-1, tb)
    badt = (dt != pt).any(1).nonzero().flatten().tolist()
    out = {"tasks": len(badt), "first_tasks": badt[:8], "words": []}
    for q in badt[:3]:
        d, p = host_bytes(dt[q]), host_bytes(pt[q])
        bw = sorted({o // pb for o in range(tb) if d[o] != p[o]})
        out.setdefault("bad_words", {})[q] = bw
        for w in bw[:4]:
            lo, hi = w * pb, min((w + 1) * pb, tb)
            got = d[lo:hi]
            sw = host_bytes(store[(q * tw + w) * s:(q * tw + w + 1) * s])
            rec = {"q": q, "w": w, "diff_bytes": sum(a != b for a, b in zip(got, p[lo:hi])),
                   "chunks": sorted({(lo + o) // 16 for o in range(hi - lo) if got[o] != p[lo + o]}), "is": []}
            if got == bytes(hi - lo):
                rec["is"].append("zero")
            for j in range(-stages - 1, stages + 2):
                q2 = q + j * gq
                if 0 <= q2 < pt.shape[0]:
                    if j and got == host_bytes(pt[q2][lo:hi]):
                        rec["is"].append(f"plain q{j:+d}gq")
                    if got == host_bytes(store[q2 * tw * s + lo:q2 * tw * s + hi]):
                        rec["is"].append(f"stored q{j:+d}gq")
            y = W.bytes_word(sw, f)
            for dj in [*range(-3, 4), *(m * tw for m in (-1, 1)), *(m * tw * gq for m in (-2, -1, 1, 2))]:
                if dj:
                    try:
                        m = W.decode_word(y, q * tw + w + dj, key)
                    except ValueError:
                        continue
                    if m.to_bytes(pb, "little")[:hi - lo] == got:
                        rec["is"].append(f"decoded at index{dj:+d}")
            out["words"].append(rec)
        sb = tw * s
        for c in sorted({o // 16 for o in range(tb) if d[o] != p[o]})[:8]:
            got, kinds = d[16 * c:16 * c + 16], []
            if got == bytes(16):
                kinds.append("zero")
            for j in range(-stages - 1, stages + 2):
                q2 = q + j * gq
                if 0 <= q2 < pt.shape[0]:
                    if j and got == host_bytes(pt[q2][16 * c:16 * c + 16]):
                        kinds.append(f"plain q{j:+d}gq")
                    if got == host_bytes(store[q2 * sb + 16 * c:q2 * sb + 16 * c + 16]):
                        kinds.append(f"stored q{j:+d}gq")
            out.setdefault("chunks", []).append({"q": q, "chunk": c, "is": kinds})
    return out


def set_receipt_prefix(lib: Lib) -> None:
    """sketch-sha512's receipt prefix at RECEIPT_SEED into a WT_RECEIPT build."""
    if not lib.receipt:
        raise ValueError(f"{lib.name}: the sketch warps' receipts are a WT_RECEIPT build's")
    pre = ctypes.create_string_buffer(SR.rb.sketch_prefix(RECEIPT_SEED))
    lib.call("wt_set_receipt", ctypes.addressof(pre))


def gemm_case(lib: Lib, torch, name: str, n: int, kd: int, batches, ks, reps: int, log, cfgs, fused_cfgs,
              mma_knobs, imad_knobs, gate_reps: int = 1, dump: bool = False, sketch_t: int = 8) -> dict:
    g = torch.Generator(device="cuda").manual_seed(100 + n + kd)
    w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    plain = plain_layout(w.view(torch.uint8))
    m = task_payload_limbs(lib, torch, plain)
    keys, stores = {}, {}
    for k in ks:
        keys[k] = lib_key(lib, k)
        set_key(lib, torch, keys[k])
        stores[k], _ = encode_store(lib, torch, m, k, 188)
    del m
    torch.cuda.synchronize()
    nch = kd // 512
    st = stream(torch)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    gx = torch.Generator(device="cuda").manual_seed(5)
    xs = {mb: torch.randn(mb, kd, device="cuda", generator=gx).to(torch.float8_e4m3fn) for mb in batches}
    res = {"N": n, "K": kd, "weight_bytes": n * kd, "store_bytes": stores[ks[0]].numel(), "gate": {}, "times": {}}
    sketched = [c for c in fused_cfgs if c[3]]
    sk = {}
    for s in sorted({c[3] for c in sketched}):
        bits = sketch_bits(lib, window_seed(lib), s, SKT.vector_bytes(lib.sb, sketch_t))
        words = pass_signs(lib, bits, sketch_t)
        sk[s] = {"bits": bits, "signs": torch.from_numpy(words.view(np.int32)).cuda()}
        set_pass(lib, sketch_t, rows=s)
        sk[s]["frags"] = sign_frags(lib, torch, sk[s]["signs"], sketch_t, st)
    sk_bufs = {}
    sk_refs = {}

    def sk_tags_for(gr):
        """The receipt buffer of a sketched launch at gr groups, and its vector count."""
        nvec = lib.so.wt_sketch_vectors(n, kd, gr, sketch_t)
        if nvec != len(SR.gemm_receipts(n // 16, nch, lib.sb, gr)):
            raise SystemExit(f"{lib.name} {name}: wt_sketch_vectors is {nvec}, not the store's blocks")
        if gr not in sk_bufs:
            sk_bufs[gr] = torch.empty(nvec * lib.tag_bytes, dtype=torch.uint8, device="cuda")
        return sk_bufs[gr], nvec

    def sketch_ref(k, s, gr, seed=None):
        """Every vector's sketch at `seed` (the window's by default) on the device (float32, exact), and its
        sketch-sha512 receipt with hashlib."""
        seed = window_seed(lib) if seed is None else seed
        if (k, s, gr, seed) not in sk_refs:
            vs = store_blocks_torch(torch, stores[k])
            bits = sk[s]["bits"] if seed == window_seed(lib) else sketch_bits(lib, seed, s,
                                                                              SKT.vector_bytes(lib.sb, sketch_t))
            ref = SKT.sketch_torch(torch, vs, bits)
            del vs
            host = ref.cpu().numpy().astype(np.int64)
            ys = [root_words(y) for y in SR.receipts_of(seed, host, lib.jbase)]
            sk_refs[(k, s, gr, seed)] = {"sk": ref, "tags": b"".join(ys)}
        return sk_refs[(k, s, gr, seed)]

    def pass_keyed(k, mb, x, pp, cfg, gr, sp):
        """cfg (two sketch warps) under pass j's key: wt_pass writes the signs and the receipt midstate from a random
        root_(j−1), and the launch's sketches, receipts, leaves and root must be sketch-sha512's at the config's k_s
        at pass_seed(σ, j, root)."""
        j, root, s = 7, random.Random(n + kd + mb).randbytes(64), cfg[3]
        seed = pass_seed(lib, j, root)
        set_pass(lib, sketch_t, rows=s)
        signs = torch.empty_like(sk[s]["signs"])
        lib.call("wt_pass", j, ptr(to_dev(torch, root_words(root))), ptr(signs), sketch_t, st)
        signs = sign_frags(lib, torch, signs, sketch_t, st)
        tb_dev, nvec = sk_tags_for(gr)
        tb_dev.fill_(0xFF)
        skb = torch.full((nvec * s,), -1, dtype=torch.int32, device="cuda")
        pf = torch.zeros_like(pp)
        call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp, skout=skb, signs=signs)
        torch.cuda.synchronize()
        r = sketch_ref(k, s, gr, seed)
        out = {"cfg": fused_name(k, cfg), "j": j, "partials": bool(torch.equal(pf, pp)),
               "sketch": bool(torch.equal(skb.view(nvec, s), r["sk"])), "tags": sketch_tags_ok(host_bytes(tb_dev), r)}
        c = receipt_commit_gate(torch, tb_dev, nvec, seed)
        out.update(leaves=c["leaves_ok"], root=c["root_ok"])
        out["ok"] = all(v for a, v in out.items() if a not in ("cfg", "j"))
        sk_refs.pop((k, s, gr, seed))
        set_receipt_prefix(lib)
        return out

    def sketch_tags_ok(tb: bytes, r: dict) -> bool:
        return tb == r["tags"]

    def groups_for(mode, k, mb, cfg):
        launched = ctypes.c_int(0)
        if mode == 3:
            lib.call("wt_fused_sketch", k, None, None, None, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, None,
                     None, 0, cfg[3], None, None, sketch_t, sketch_warps(cfg), ctypes.byref(launched), st)
        else:
            lib.call("wt_gemm", mode, None, None, None, mb, n, kd, 0, cfg[0], cfg[1], 0, 0, None,
                     ctypes.byref(launched), st)
        return launched.value // nch

    def call(mode, k, wbuf, x, y, partials, cfg, knob=0, groups=0, steps=None, skout=None, signs=None):
        if mode == 3:
            tp = ptr(sk_tags_for(groups)[0]) if cfg[3] else None
            sg = signs if signs is not None else (sk[cfg[3]]["frags"] if cfg[3] else None)
            lib.call("wt_fused_sketch", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0,
                     int(partials), cfg[0], cfg[1], groups, ptr(steps), tp, lib.jbase, cfg[3],
                     ptr(sg) if sg is not None else None, ptr(skout) if skout is not None else None, sketch_t,
                     sketch_warps(cfg), None, st)
        else:
            lib.call("wt_gemm", mode, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, int(partials),
                     cfg[0], cfg[1], groups, knob, ptr(xr), None, st)

    steps_cache = {}

    def steps_for(k, mb, cfg):
        gr = groups_for(3, k, mb, cfg)
        key = (k, mb, cfg)
        if key not in steps_cache:
            steps_cache[key] = (gr, tsteps(lib, torch, keys[k], cfg[0], gr, nch))
        return steps_cache[key]

    for mb in batches:
        x = xs[mb]
        ref = x.float() @ w.float().t()
        scale = (x.float().abs() @ w.float().abs().t()).clamp_min(1e-30)
        pp = torch.zeros(nch * mb * n, dtype=torch.float32, device="cuda")
        call(0, 1, plain, x, pp, True, cfgs[0])
        err = float(((pp.view(nch, mb, n).sum(0) - ref).abs() / scale).max())
        entry = {"plain_vs_torch_rel": err, "equal_plain": {}, "tags": {}}
        for cfg in cfgs[1:]:
            pf = torch.zeros_like(pp)
            call(0, 1, plain, x, pf, True, cfg)
            entry["equal_plain"][f"plain_c{cfg[0]}s{cfg[1]}"] = bool(torch.equal(pf, pp))
        for k in ks:
            set_key(lib, torch, keys[k])
            for cfg in fused_cfgs:
                if not fused_fits(lib, k, mb, n, kd, cfg, sketch_t):
                    continue
                gr, sp = steps_for(k, mb, cfg)
                bad, sk_bad = [], []
                if cfg[3]:
                    set_receipt_prefix(lib)
                    tb_dev, nvec = sk_tags_for(gr)
                    skr = sketch_ref(k, cfg[3], gr)
                for _ in range(gate_reps):
                    pf = torch.zeros_like(pp)
                    skb = None
                    if cfg[3]:
                        tb_dev.fill_(0xFF)
                        skb = torch.full((nvec * cfg[3],), -1, dtype=torch.int32, device="cuda")
                    call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp, skout=skb)
                    if not torch.equal(pf, pp):
                        bad.append(mismatch(pf, pp, nch, mb, n, gr))
                    if cfg[3]:
                        torch.cuda.synchronize()
                        sk_eq = torch.equal(skb.view(nvec, cfg[3]), skr["sk"])
                        tg_eq = sketch_tags_ok(host_bytes(tb_dev), skr)
                        if not (sk_eq and tg_eq):
                            sk_bad.append({"sketch": sk_eq, "tags": tg_eq,
                                           "vectors_unequal": int((skb.view(nvec, cfg[3]) != skr["sk"]).any(1).sum())})
                if cfg[3]:
                    entry.setdefault("sketch", {})[fused_name(k, cfg)] = {"vectors": nvec, "bad": sk_bad[:3],
                                                                          "ok": not sk_bad}
                    entry["tags"][fused_name(k, cfg)] = not sk_bad
                    if sk_bad:
                        log(f"sketch mismatch {lib.name} {name} M={mb} {fused_name(k, cfg)}: {sk_bad[0]}")
                    if "sketch_commit" not in res and not sk_bad:
                        res["sketch_commit"] = receipt_commit_gate(torch, tb_dev, nvec, RECEIPT_SEED)
                        log(f"sketch commit gate {lib.name} {name}: {nvec} vectors, {res['sketch_commit']}")
                    if not sk_bad and "pass_keyed" not in res:
                        res["pass_keyed"] = pass_keyed(k, mb, x, pp, cfg, gr, sp)
                        entry["tags"][fused_name(k, cfg) + "_pass"] = res["pass_keyed"]["ok"]
                        log(f"pass-keyed gate {lib.name} {name} M={mb}: {res['pass_keyed']}")
                    res["sketch_tags"] = nvec
                entry["equal_plain"][fused_name(k, cfg)] = not bad
                if bad and dump:
                    dbg = torch.zeros_like(plain)
                    lib.call("wt_set_dbg", ptr(dbg))
                    for tries in range(1, 4 * gate_reps + 1):   # a race shows in some runs only: dump a failing one
                        pf = torch.zeros_like(pp)
                        call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp, skout=skb)
                        torch.cuda.synchronize()
                        if not torch_equal(pf, pp):
                            break
                    lib.call("wt_set_dbg", None)
                    dw = dump_words(lib, dbg, plain, stores[k], keys[k], gr * nch, cfg[1])
                    dw["partials_equal"], dw["tries"] = torch_equal(pf, pp), tries
                    if not dw["partials_equal"]:
                        dw["mismatch"] = mismatch(pf, pp, nch, mb, n, gr)
                    entry.setdefault("dump", {})[fused_name(k, cfg)] = dw
                    log(f"dump {lib.name} {name} M={mb} {fused_name(k, cfg)}: {dw}")
                    del dbg
                if bad:
                    entry.setdefault("mismatch", {})[fused_name(k, cfg)] = {"failed": len(bad), "of": gate_reps,
                                                                            "first": bad[:3]}
                    log(f"mismatch {lib.name} {name} M={mb} {fused_name(k, cfg)}: {len(bad)}/{gate_reps} runs, "
                        f"first {bad[0]}")
        for knob in mma_knobs:
            pf = torch.zeros_like(pp)
            call(1, 1, plain, x, pf, True, cfgs[0], knob=knob)
            entry["equal_plain"][f"mma{knob}"] = bool(torch.equal(pf, pp))
        for knob in imad_knobs:
            pf = torch.zeros_like(pp)
            call(2, 1, plain, x, pf, True, cfgs[0], knob=knob)
            entry["equal_plain"][f"imad{knob}"] = bool(torch.equal(pf, pp))
        sk_commit = res.get("sketch_commit", {})
        entry["ok"] = (err < 1e-5 and all(entry["equal_plain"].values()) and all(entry["tags"].values())
                       and (not sketched or (sk_commit.get("leaves_ok") and sk_commit["root_ok"])))
        res["gate"][mb] = entry
        log(f"gemm gate {lib.name} {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (plain rel {err:.1e}) "
            + (f", receipts {sum(entry['tags'].values())}/{len(entry['tags'])}" if sketched else "")
            + ("" if entry["ok"] else str({k: v for d in (entry["equal_plain"], entry["tags"]) for k, v in d.items()
                                           if not v})))
        if not entry["ok"]:
            res["ok"] = False
            if gate_reps == 1:
                return res
    if res.setdefault("ok", True) is False:
        return res

    copies = max(2, -(-L2_DEFEAT_BYTES // res["store_bytes"]))
    ring = {"plain": [plain] + [plain.clone() for _ in range(copies - 1)]}
    for k in ks:
        ring[k] = [stores[k]] + [stores[k].clone() for _ in range(copies - 1)]
    wring = [w] + [w.clone() for _ in range(copies - 1)]
    res["copies"] = copies
    so = lib.so
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
            for wt_ in wts:
                torch._scaled_mm(xc, wt_, one, one, out_dtype=torch.bfloat16)

        def arm(mode, k, cfg, knob=0):
            ps = [ptr(b) for b in ring["plain" if mode != 3 else k]]
            gr, sp = steps_for(k, mb, cfg) if mode == 3 else (groups_for(mode, k, mb, cfg), None)
            spp = ptr(sp) if sp is not None else None
            xrp = ptr(xr)
            skw = sketch_warps(cfg) if mode == 3 else 0
            sks, sgp = (cfg[3], ptr(sk[cfg[3]]["frags"])) if mode == 3 and cfg[3] else (0, None)
            tp = ptr(sk_tags_for(gr)[0]) if sks else None

            def f():
                rc = 0
                if mode == 3:
                    for p in ps:
                        rc |= so.wt_fused_sketch(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], gr, spp, tp,
                                                 lib.jbase, sks, sgp, None, sketch_t, skw, None, st)
                else:
                    for p in ps:
                        rc |= so.wt_gemm(mode, p, xp, yp, mb, n, kd, 0, cfg[0], cfg[1], gr, knob, xrp, None, st)
                if rc:
                    raise RuntimeError(f"wt_gemm mode {mode} k={k} M={mb}: error {rc}")
            return f

        grid = 188

        def arm_empty():
            rc = 0
            for _ in range(copies):
                rc |= so.wt_empty(grid, st)
            if rc:
                raise RuntimeError("wt_empty")

        arms = {"cublas": arm_cublas, "empty": arm_empty}
        for cfg in cfgs:
            arms[f"plain_c{cfg[0]}s{cfg[1]}"] = arm(0, 1, cfg)
        for k in ks:
            for cfg in fused_cfgs:
                if fused_fits(lib, k, mb, n, kd, cfg, sketch_t):
                    arms[fused_name(k, cfg)] = arm(3, k, cfg)
        for knob in mma_knobs:
            arms[f"mma{knob}"] = arm(1, 1, cfgs[0], knob)
        for knob in imad_knobs:
            arms[f"imad{knob}"] = arm(2, 1, cfgs[0], knob)
        set_key(lib, torch, keys[ks[0]])
        for f_ in arms.values():
            f_()
        torch.cuda.synchronize()
        times = {a: [] for a in arms}
        host = {a: [] for a in arms}
        names = list(arms)
        for rep in range(reps):
            order = names[rep % len(names):] + names[:rep % len(names)]
            for a in order:
                if a.startswith("fused_k"):
                    set_key(lib, torch, keys[int(a[7])])
                t, h = queued(lib, torch, arms[a])
                times[a].append(t / copies)
                host[a].append(h)
        med = {a: statistics.median(v) for a, v in times.items()}
        native = min(med[a] for a in med if a == "cublas" or a.startswith("plain_"))
        plain_best = min(med[a] for a in med if a.startswith("plain_"))
        res["times"][mb] = {"median_s": med, "times_s": times, "cublas_m": mpad, "native_s": native,
                            "plain_best_s": plain_best, "host_enqueue_max_s": max(max(v) for v in host.values()),
                            "ratio_vs_native": {a: med[a] / native for a in med},
                            "ratio_vs_plain0": {a: med[a] / med[f"plain_c{cfgs[0][0]}s{cfgs[0][1]}"] for a in med}}
        def best(k, s=0):
            cs = [c for c in fused_cfgs if c[3] == s and fused_name(k, c) in med]
            return min((med[fused_name(k, c)], fused_name(k, c)[7:]) for c in cs) if cs else (float("nan"), "-")

        log(f"gemm {lib.name} {name} M={mb}: cublas {med['cublas'] * 1e6:.1f} us, plain best {plain_best * 1e6:.1f} us; "
            f"native {native * 1e6:.1f} us; "
            + ", ".join(f"fused k{k} {best(k)[0] / native:.2f}x ({best(k)[1]})" for k in ks)
            + "".join(f", sketch warps {s} k{k} {best(k, s)[0] / native:.2f}x ({best(k, s)[1]})"
                      for s in sorted({c[3] for c in sketched}) for k in ks)
            + "; mma " + " ".join(f"{kn}:{med[f'mma{kn}'] / native:.2f}" for kn in mma_knobs)
            + "; imad " + " ".join(f"{kn}:{med[f'imad{kn}'] / native:.2f}" for kn in imad_knobs))
    del ring, wring, stores, plain, w
    torch.cuda.empty_cache()
    return res


def layer_ratios(gemm: dict, batches, ks) -> dict:
    """Per batch, a layer's time over its native time with each linear at its best arm: the plain kernel, and per k the
    fused decode untagged and beside the sketch warps at each k_s."""
    out = {}
    for mb in batches:
        meds = [g["times"][mb]["median_s"] for g in gemm.values()]
        native = sum(g["times"][mb]["native_s"] for g in gemm.values())
        row = {"cublas": sum(m["cublas"] for m in meds) / native,
               "plain": sum(min(v for a, v in m.items() if a.startswith("plain_")) for m in meds) / native}
        for k in ks:
            kinds = [(f"fused_k{k}", r"_d\d+s\d+$"), (f"fused_k{k}_sketch16_warps", r"w2sk16$"),
                     (f"fused_k{k}_sketch64_warps", r"w2sk64$")]
            for label, pat in kinds:
                arms = [[v for a, v in m.items() if a.startswith(f"fused_k{k}_") and re.search(pat, a)] for m in meds]
                if all(arms):
                    row[label] = sum(min(v) for v in arms) / native
        out[mb] = row
    return out


# --- analysis ----------------------------------------------------------------------------------------------------------

def analysis(lib: Lib, read_gbps: float, r: dict) -> dict:
    """x per unit (the adversary's root at the squaring floor, n - 2 squarings a word for a Mersenne prime and at least
    that for p = 2^n - c, on CUDA cores, on tensor cores, and both in one CTA), Delta (decode's compute-only time per
    native pass) and r = x / Delta, per k. A native pass reads the plain weights (W bytes) at R; their store is
    W TW / 8192 words."""
    f = lib.field
    words_per_byte = lib.tw / W.TASK_BYTES
    t_native = 1 / (read_gbps * 1e9)                      # seconds a plain byte
    sq = {"cuda": max(v["per_s"] for v in r["square_cuda"].values())}
    if r["square_tc"]:
        sq["tc"] = max(v["per_s"] for v in r["square_tc"].values())
        both = [v["both"]["per_s"] for v in r["mix"].values() if v.get("both")]
        sq["both"] = max([*both, sq["cuda"]])
    out = {"read_gbps": read_gbps, "words_per_plain_byte": words_per_byte, "squarings_s": sq, "k": {}}
    for k in (1, 2):
        row = {}
        for unit, rate in sq.items():
            root_s = rate / (f.n - 2)                       # words/s at the floor
            row[f"x_{unit}"] = k * words_per_byte / root_s / t_native
        enc = r["encode"][k]["per_s"]
        dec = r["decode"][k]["per_s"]
        row["x_cuda_measured_encode"] = words_per_byte / enc / t_native
        row["delta_cuda"] = words_per_byte / dec / t_native
        row["r_cuda"] = dec / enc
        out["k"][k] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="a wt.cu build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", default="rates,decode,gemm")
    ap.add_argument("--store-gib", type=float, default=2.0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gate-words", type=int, default=256)
    ap.add_argument("--gate-reps", type=int, default=1,
                    help="runs of each fused gate; above 1, every batch is gated and a failure logs where it differs")
    ap.add_argument("--model", default="70b", choices=sorted(MODELS))
    ap.add_argument("--shapes", default="", help="the model's linears to run (default: all)")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--gemm-libs", default="", help="field names the GEMM runs with (default: all)")
    ap.add_argument("--gemm-ks", default="1")
    ap.add_argument("--plain", default="4x4,8x8", help="plain kernel CONSUMERSxSTAGES")
    ap.add_argument("--fused", default="auto",
                    help="fused kernel DECODERSxSTAGES[x-2xK_S], or auto (fused_cfgs)")
    ap.add_argument("--dump", action="store_true",
                    help="after a failed fused gate, rerun it (until it fails, at most 4 x --gate-reps) dumping the MMA "
                         "warp's operands (WT_DEBUG_DUMP build)")
    ap.add_argument("--sketch", default="", help="k_s values (16, 64) for the sketch warps' sketch-sha512 receipts (a "
                                                 "WT_RECEIPT build at full slot)")
    ap.add_argument("--sketch-t", type=int, default=8, help="tasks a sketch tile (n = 32 ceil(SB / 512) T bytes)")
    ap.add_argument("--mma-knobs", default="8,16,32,48,64,96")
    ap.add_argument("--imad-knobs", default="32,64,128,192,256")
    ap.add_argument("--reader-lib", default="", help="reader.cu, the commit of a WT_RECEIPT=1 build (its gemm gate)")
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

    def save() -> None:
        (out_dir / "codec-tensor.json").write_text(json.dumps(res, indent=1))

    libs = [Lib(p) for p in a.lib]
    if a.reader_lib:
        import reader as RD
        global READER
        READER = RD.Reader(a.reader_lib, torch)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    phases = set(a.phases.split(","))
    res = {"device": {"name": props.name, "sms": sms, "cc": f"{props.major}.{props.minor}"}, "seed": SEED.hex(),
           "libs": {}, "ok": True, "args": vars(a)}
    log(f"device {props.name}, {sms} SMs; fields {[l.name for l in libs]}")
    res["layout"] = gate_layout(torch, log)
    res["ok"] &= res["layout"]["ok"]
    for lib in libs:
        g = {k: gate_codec(lib, torch, k, a.gate_words, log) for k in range(1, lib.kmax + 1)}
        e = {"path": lib.path, "constants": {"n": lib.n, "NL": lib.nl, "S": lib.s, "PB": lib.pb, "TW": lib.tw,
                                             "U": lib.u, "c": lib.c, "tc_mmas": lib.tc_mmas},
             "sha256": hashlib.sha256(Path(lib.path).read_bytes()).hexdigest(), "gate": g}
        if lib.tc:
            e["gate_tc_square"] = gate_tc_square(lib, torch, 512, log)
        res["libs"][lib.name] = e
        res["ok"] &= all(v["ok"] for v in g.values()) and e.get("gate_tc_square", {"ok": True})["ok"]
    save()
    if not res["ok"]:
        log("a gate failed: nothing timed")
        write_result(res, libs, out_dir)
        return 5
    store = torch.randint(0, 255, (int(a.store_gib * (1 << 30)) // 16 * 16,), dtype=torch.uint8, device="cuda")
    res["read"] = read_rate(libs[0], torch, store, sms, a.reps)
    del store
    torch.cuda.empty_cache()
    log(f"read {res['read']['gbps']:.0f} GB/s")
    for lib in libs:
        e = res["libs"][lib.name]
        set_key(lib, torch, lib_key(lib, 2))
        if "rates" in phases:
            e["rates"] = rates(lib, torch, sms, a.reps, log)
            e["analysis"] = analysis(lib, res["read"]["gbps"], e["rates"])
            for k, row in e["analysis"]["k"].items():
                log(f"analysis {lib.name} k={k}: " + ", ".join(f"{kk} {v:.3g}" for kk, v in row.items()))
            save()
        if "decode" in phases:
            e["decode"] = standalone_decode(lib, torch, int(a.store_gib * (1 << 30)), sms, a.reps, log)
            res["ok"] &= all(v["ok"] for v in e["decode"].values())
            save()
    if "pipes" in phases:
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        res["pipes"] = {}
        for op, name in enumerate(PIPE_OPS):
            r = max((dict(calibrated(torch, lambda s: libs[0].call("wt_chain", 5, 1, op, s, ptr(xr), sms * m,
                                                                   stream(torch)), sms * m * 128 * 8, 0.08, a.reps),
                          grid_per_sm=m) for m in (4, 8, 16)), key=lambda v: v["per_s"])
            r["clock_hz"] = clock_hz = sm_clock(libs[0], torch)
            r["lanes_per_sm_clock"] = r["per_s"] / (sms * clock_hz)
            res["pipes"][name] = r
            log(f"pipe {name}: {r['per_s']:.3e}/s, {r['lanes_per_sm_clock']:.1f} lanes an SM a clock "
                f"(at {clock_hz / 1e6:.0f} MHz)")
        save()
    if "gemm" in phases:
        names = a.gemm_libs.split(",") if a.gemm_libs else [l.name for l in libs]
        ks = [int(k) for k in a.gemm_ks.split(",")]
        batches = [int(b) for b in a.batches.split(",")]
        knobs = ([int(v) for v in a.mma_knobs.split(",") if v], [int(v) for v in a.imad_knobs.split(",") if v])
        layers, shapes = MODELS[a.model]
        res["model"] = {"name": a.model, "layers": layers, "shapes": shapes}
        for lib in [l for l in libs if l.name in names]:
            e = res["libs"][lib.name]
            e["gemm"] = {}
            if any(sketch_warps(c) for c in fused_cfgs(lib, a.fused, a.sketch, a.sketch_t)):
                e["gate_pass"] = gate_pass(lib, torch, a.sketch_t, log)
                res["ok"] &= e["gate_pass"]["ok"]
                if not e["gate_pass"]["ok"]:
                    continue
            for name in a.shapes.split(",") if a.shapes else shapes:
                n, kd = shapes[name]
                g = gemm_case(lib, torch, name, n, kd, batches, ks, a.reps, log, parse(a.plain),
                              fused_cfgs(lib, a.fused, a.sketch, a.sketch_t), *knobs, gate_reps=a.gate_reps,
                              dump=a.dump, sketch_t=a.sketch_t)
                e["gemm"][name] = g
                res["ok"] &= g.get("ok", False)
                save()
            if not all(g.get("ok") for g in e["gemm"].values()):
                continue
            e["layer"] = layer_ratios(e["gemm"], batches, ks)
            for mb, row in e["layer"].items():
                log(f"layer {lib.name} M={mb} ({'+'.join(e['gemm'])}): "
                    + ", ".join(f"{kk} {v:.3f}" for kk, v in row.items()))
            knobs = ([], [])
    save()
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


def write_result(res: dict, libs: list[Lib], out_dir: Path) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for name, e in res["libs"].items():
        for unit, v in e.get("analysis", {}).get("squarings_s", {}).items():
            ms.append({"name": f"{name}_squarings_{unit}", "value": v, "unit": "1/s"})
        for k, row in e.get("analysis", {}).get("k", {}).items():
            for kk, v in row.items():
                ms.append({"name": f"{name}_k{k}_{kk}", "value": v, "unit": "1"})
        for k, d in e.get("decode", {}).items():
            for unit in ("cuda", "tc"):
                if d.get(unit, {}).get("ok"):
                    ms.append({"name": f"{name}_decode_k{k}_{unit}", "value": d[unit]["gbps_stored"], "unit": "GB/s"})
        for shape, g in e.get("gemm", {}).items():
            for mb, t in g.get("times", {}).items():
                for arm, ratio in t["ratio_vs_native"].items():
                    if arm != "empty":
                        ms.append({"name": f"{name}_{shape}_m{mb}_{arm}_vs_native", "value": ratio, "unit": "1"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/codec-tensor.json", f"{rel}/bench.log"]
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "codec-tensor gates",
                          "detail": "device codec = wide.py, TC square = a^2 mod p, fused partial sums = plain",
                          "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


if __name__ == "__main__":
    sys.exit(main())
