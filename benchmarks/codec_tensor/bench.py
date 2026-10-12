"""The wide-word codec's benchmark on one GPU: `wt.cu`'s libraries (`--lib PATH`, one per field) gated against
`wide.py`, then timed. Writes `codec-tensor.json` and `bench.log` under `--out`, and in a `research run` its
`result.json`.

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
  (and `--gemm-ks`); per batch, the layer's time over native with each linear at its best arm. With `--sketch`, the
  fused decode (k_s = 16 at each k, 64 at k = 1) also runs with ro-sketch-blake3/v0 (`sketching`) in place of
  ro-blake3/v1, gated on every
  vector's sketch (a float32 GPU reference with TF32 off, exact), every tag (PyPI blake3, `tags.py` on a sample), and
  the leaves and root. A diagnostic build (`WT_SK_DIAG`, named `<field>-diag<bits>`) drops parts of the sketch to time
  them: its partial sums are gated, its sketches and tags are not. At full slot `auto` adds the receipt on two sketch
  warps of its own (T = -2); before its GEMMs, `gate_pass` checks the per-pass key (red team F29: `wt_pass` derives
  pass j's seed, signs, key and leaves' prefix on the device from root_(j−1)), and once a linear one such launch runs
  under a pass's key and is gated at that pass's seed (`pass_keyed`).
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

import sketching as SKT  # noqa: E402
import tagging as TG  # noqa: E402
import wide as W  # noqa: E402

SEED = hashlib.sha256(b"verity/codec-tensor/bench/seed").digest()
TAG_SEED = hashlib.sha256(b"verity/codec-tensor/bench/tag-seed").digest()   # the window's sigma
TAG_BASE = (1 << 32) + 3                                                      # E's index of a layer's first block
B3_PER_BLOCK = (67, 72)        # compressions a block (4,096 message bytes): the fewest (64 chunk, 3 parent), and the
                               # tag warps' (4 lanes, 16 + 2 each)
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


class Lib:
    def __init__(self, path: str) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path, self.so = path, ctypes.CDLL(path)
        so = self.so
        so.wt_set_key.argtypes = [v, i, v, v]
        so.wt_encode.argtypes = [v, v, q, q, i, i, v]
        so.wt_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.wt_tc_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.wt_read.argtypes = [v, q, v, i, v]
        so.wt_tc_square_gate.argtypes = [v, v, i, v]
        so.wt_chain.argtypes = [i, i, i, q, v, i, v]
        so.wt_mix.argtypes = [q, q, i, v, i, v]
        so.wt_gemm.argtypes = [i, v, v, v, i, i, i, i, i, i, i, i, v, v, v]
        so.wt_fused.argtypes = [i, v, v, v, i, i, i, q, i, i, i, i, i, v, v, q, v, v]
        so.wt_fused_sketch.argtypes = [i, v, v, v, i, i, i, q, i, i, i, i, i, v, v, q, i, v, v, i, i, v, v]
        so.wt_set_pass.argtypes = [v, i, i]
        so.wt_pass.argtypes = [ctypes.c_uint, v, v, i, v]
        so.wt_pass_state.argtypes = [v]
        so.wt_sketch_vectors.argtypes = [i, i, i, i]
        so.wt_sketch_vectors.restype = q
        so.wt_set_tag.argtypes = [v, v]
        so.wt_set_dbg.argtypes = [v]
        so.wt_tag_leaves.argtypes = [v, q, v, v]
        so.wt_tag_tree.argtypes = [v, q, v, v]
        so.wt_spin.argtypes = [ctypes.c_longlong, v]
        so.wt_empty.argtypes = [i, v]
        cs = (ctypes.c_int * 17)()
        so.wt_constants(cs)
        (self.n, self.nl, self.s, self.pb, self.tw, self.u, self.c, self.wpb, self.tc_mmas, self.tc_bytes, self.kmax,
         self.sb, self.fused_warps, self.fused_span, self.tag_block, self.diag, self.fused_most) = list(cs)
        self.field = next(f for f in W.FIELDS.values() if f.n == self.n)
        f = self.field
        if (self.s, self.pb, self.tw, self.c, self.sb) != (f.word_bytes, f.payload_bytes, W.TASK_WORDS[f.name], f.c,
                                                             W.TASK_WORDS[f.name] * f.word_bytes):
            raise RuntimeError(f"{path}: constants {list(cs)} are not wide.py's {f}")
        if self.tag_block != TG.TAG_BLOCK:
            raise RuntimeError(f"{path}: tag blocks of {self.tag_block} bytes, tagging.py's are {TG.TAG_BLOCK}")
        self.name = f.name + (f"-diag{self.diag}" if self.diag else "")
        self.tc = self.tc_mmas > 0

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


def set_tag(lib: Lib) -> None:
    """ro-blake3/v1's key and leaf prefix at TAG_SEED into the library."""
    k = ctypes.create_string_buffer(TG.key(TAG_SEED))
    pre = ctypes.create_string_buffer(TG.leaf_prefix(TAG_SEED))
    lib.call("wt_set_tag", ctypes.addressof(k), ctypes.addressof(pre))


def set_key(lib: Lib, torch, key: W.Key) -> None:
    """The key into the library's constant memory, from host buffers."""
    c = ctypes.create_string_buffer(b"".join(limbs_bytes(x, lib.nl) for x in key.c))
    tau = ctypes.create_string_buffer(limbs_bytes(key.tau, lib.nl))
    p = ctypes.create_string_buffer(limbs_bytes(lib.field.p, lib.nl))
    lib.call("wt_set_key", ctypes.addressof(c), key.rounds, ctypes.addressof(tau), ctypes.addressof(p))


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_codec(lib: Lib, torch, k: int, n_words: int, log) -> dict:
    f = lib.field
    key = W.key(f.name, SEED, k)
    set_key(lib, torch, key)
    rng = random.Random(7 + k + f.n)
    top = f.max_payload
    payloads = [0, 1, 2, top, top - 1, 1 << (8 * f.payload_bytes - 1)]
    payloads += [rng.randrange(top + 1) for _ in range(n_words - len(payloads))]
    base = (1 << 40) + 12345
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
        if name == "tc" and not lib.tc:
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
        set_key(lib, torch, W.key(f.name, SEED, k))
        store, enc_s = encode_store(lib, torch, m, k, sms)
        back = torch.empty_like(m)
        entry = {"words": n, "store_bytes": n * f.word_bytes, "setup_encode_s": enc_s}
        for fn, name, mults in (("wt_decode", "cuda", (4, 8, 16)), ("wt_tc_decode", "tc", (4, 8, 16, 24))):
            if name == "tc" and not lib.tc:
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
    """(D, stages, T, k_s): DECODERSxSTAGES (untagged, T = -1), DECODERSxSTAGESxT (tagged: T tag warps from the
    start, and every other warp tags once its own work is done), DECODERSxSTAGESx-1xK_S (the decoders sketch at
    k_s over tiles of `sketch_t` tasks, stages < 2 sketch_t) or DECODERSxSTAGESx-2x16 (two sketch warps of their own
    sketch and hash, beside D decoders: full slot, k_s = 16), each checked against wt_fused_sketch's bounds; at
    M > 16 the kernel has two MMA warps (wt.cu's fused_mma_warps), so D + 3 warps untagged and D + 5 beside the sketch
    warps must fit. `auto`: untagged, the most decoder warps the field's registers allow at 4 and 6 stages fewer than
    fit at M = 32 (fewer stages leave more L1) and at all of them, 1 fewer at 4 and 6 fewer stages (M = 32's), and 2
    fewer warps at 2 fewer stages (p1536's 10 decoders starve at 8 stages: 8 x 8 and 10 x 10 are its best); tagged, 0
    and 2 of those warps tagging from the start, at 4 fewer stages; sketched, for each k_s in `sketch`, all of them
    sketching at 0, 2, 3 and 4 fewer stages (the sketch's buffers already take some), and at k_s = 16 and full slot
    the two sketch warps beside all of them, beside 2 and 3 fewer (M = 32's), at 0 and 2 fewer stages. Where the
    field's kernel takes 20 warps (wt.cu's fused_warps: untagged at M <= 8, and beside the sketch warps at M <= 16),
    also 16 decoders, four a sub-partition, untagged at 2 and 4 fewer stages and beside the sketch warps at 0 and 2
    fewer, and 18 untagged at 4 fewer. gemm_case skips each config at an M whose kernel doesn't take its warps
    (fused_fits). Four or more tag warps (with the decoders that leave) fail some gates in some runs, on early tasks of
    random CTAs, unexplained, so auto leaves them out."""
    smax = lambda ks: (SMEM_PER_BLOCK - 4 * 8 * 528 - 32 - 192 * ks) // (lib.sb + 48)  # noqa: E731
    dmax, dwide = lib.fused_warps - 2, lib.fused_most - 2
    need = lambda d: -(-32 * d // lib.tw)  # noqa: E731
    full_slot = lib.pb == lib.s
    if spec == "auto":
        cfgs = [(dmax, max(need(dmax), smax(0) - f), -1) for f in (4, 6, 0)]
        cfgs += [(dmax - 1, max(need(dmax - 1), smax(0) - f), -1) for f in (4, 6) if dmax - 1 >= lib.fused_span]
        if dmax - 2 >= lib.fused_span:
            cfgs.append((dmax - 2, max(need(dmax - 2), smax(0) - 2), -1))
        cfgs += [(dmax - t, max(need(dmax - t), smax(0) - 4), t) for t in (0, 2) if dmax - t >= lib.fused_span]
        if dwide > dmax:
            cfgs += [(dwide - 2, max(need(dwide - 2), smax(0) - f), -1) for f in (2, 4)]
            cfgs.append((dwide, max(need(dwide), smax(0) - 4), -1))
        for ks in (int(v) for v in sketch.split(",") if v):
            cfgs += [(dmax, min(2 * sketch_t - 1, max(need(dmax), smax(ks) - f)), -1, ks) for f in (0, 2, 3, 4)]
            if ks == 16 and full_slot:
                cfgs += [(d, min(2 * sketch_t - 1, max(need(d), smax(ks) - f)), -2, ks)
                         for d in sorted({dmax, dmax - 2, dmax - 3, dwide - 2}, reverse=True) if d >= lib.fused_span
                         for f in (0, 2)]
    else:
        cfgs = parse(spec)
    out = []
    for c in cfgs:
        if len(c) not in (2, 3, 4):
            raise SystemExit(f"fused {c}: DxS, DxSxT, DxSx-1xK_S or DxSx-2x16")
        d, st, t, ks = tuple(c) + (-1, 0)[len(c) - 2:]
        if not (lib.fused_span <= d and d + max(t, 0) <= dwide and need(d) <= st <= smax(ks) and t >= -2
                and ks in (0, 16, 64) and (ks == 0 or (t in (-1, -2) and st < 2 * sketch_t))
                and (t != -2 or (ks == 16 and full_slot))):
            raise SystemExit(f"{lib.name}: fused {c} is outside {lib.fused_span} <= D, D + T <= {dwide}, "
                             f"ceil(32 D / TW) <= stages <= {smax(ks)}, k_s in 16, 64 with no T (or T = -2: two "
                             f"sketch warps, k_s = 16, full slot) and stages < {2 * sketch_t}")
        if (d, st, t, ks) not in out:
            out.append((d, st, t, ks))
    return out


def sketch_warps(cfg) -> int:
    return 2 if cfg[2] == -2 else 0


def fused_fits(lib: Lib, k: int, mb: int, n: int, kd: int, cfg, sketch_t: int) -> bool:
    """wt_fused_sketch takes cfg at M = mb (two sketch warps beside D decoders need D + 4 warps past M = 16)."""
    launched = ctypes.c_int(0)
    return lib.so.wt_fused_sketch(k, None, None, None, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, max(cfg[2], 0), None, None,
                                  0, cfg[3], None, None, sketch_t, sketch_warps(cfg), ctypes.byref(launched),
                                  None) == 0


def fused_name(k: int, cfg) -> str:
    return (f"fused_k{k}_d{cfg[0]}s{cfg[1]}" + (f"t{cfg[2]}" if cfg[2] >= 0 else "") + ("w2" if cfg[2] == -2 else "")
            + (f"sk{cfg[3]}" if cfg[3] else ""))


def host_bytes(t) -> bytes:
    h = t.contiguous().cpu()
    return ctypes.string_at(h.data_ptr(), h.numel() * h.element_size())


def digests(mem: bytes) -> list[bytes]:
    """The device's SHA-512 digests (8 state words each, as u64 in memory) as bytes."""
    return [b"".join(int.from_bytes(mem[i + 8 * w:i + 8 * w + 8], "little").to_bytes(8, "big") for w in range(8))
            for i in range(0, len(mem), 64)]


def tag_reference(store, label: str) -> dict:
    """The store's tags at TAG_SEED from block index TAG_BASE: tags.py on a sample (the first and last blocks and some
    between), and every block with the PyPI blake3 when it is installed, which must agree on the sample."""
    sb = host_bytes(store)
    n = TG.block_count(len(sb))
    rng = random.Random(len(sb))
    sample = sorted({*range(min(4, n)), *range(max(0, n - 4), n), *(rng.randrange(n) for _ in range(8))})
    ref = {jl: TG.tag(TAG_SEED, TAG_BASE + jl, TG.block(sb, jl)) for jl in sample}
    fast = TG.tags_fast(sb, TAG_BASE, TAG_SEED)
    if fast is not None and any(fast[jl] != y for jl, y in ref.items()):
        raise SystemExit(f"{label}: the PyPI blake3 disagrees with tags.py")
    return {"blocks": n, "sample": ref, "all": b"".join(fast) if fast is not None else None}


def tags_ok(tb: bytes, ref: dict) -> bool:
    if any(tb[32 * jl:32 * jl + 32] != y for jl, y in ref["sample"].items()):
        return False
    if ref["all"] is not None:
        return tb == ref["all"]
    return all(tb[32 * i:32 * i + 32] != b"\xff" * 32 for i in range(ref["blocks"]))


def commit_gate(lib: Lib, torch, tags_dev, nblk: int, st: int, leaves_of=TG.leaves, seed: bytes = TAG_SEED) -> dict:
    """The device's leaves over its tags and its tree's root, against tags.py's leaves at `seed` (ro-blake3/v1's, or
    `leaves_of`'s) and the RFC 6962 root."""
    nleaves = -(-nblk // TG.PER_LEAF)
    leaves = torch.empty(nleaves * 64, dtype=torch.uint8, device="cuda")
    lib.call("wt_tag_leaves", ptr(tags_dev), nblk, ptr(leaves), st)
    nodes = torch.zeros((2 * nleaves + 64) * 64, dtype=torch.uint8, device="cuda")
    nodes[:nleaves * 64] = leaves
    ri = ctypes.c_ulonglong(0)
    lib.call("wt_tag_tree", ptr(nodes), nleaves, ctypes.addressof(ri), st)
    torch.cuda.synchronize()
    tb = host_bytes(tags_dev)
    want = leaves_of(seed, [tb[32 * i:32 * i + 32] for i in range(nblk)])
    got = digests(host_bytes(leaves))
    root = digests(host_bytes(nodes[ri.value * 64:(ri.value + 1) * 64]))[0]
    return {"leaves": nleaves, "leaves_ok": got == want, "root_ok": root == TG.root(want), "root": root.hex()}


def root_words(root: bytes) -> bytes:
    """A 64-byte digest as the device holds a tree node: 8 SHA-512 state words, u64 little-endian in memory."""
    return b"".join(int.from_bytes(root[8 * w:8 * w + 8], "big").to_bytes(8, "little") for w in range(8))


def set_pass(lib: Lib, sketch_t: int, sigma: bytes = TAG_SEED) -> int:
    """wt_set_pass at the window's σ for ro-sketch-blake3/v0 at k_s = 16 over tiles of sketch_t tasks; returns n."""
    n = SKT.vector_bytes(lib.sb, sketch_t)
    sg = ctypes.create_string_buffer(sigma)
    lib.call("wt_set_pass", ctypes.addressof(sg), 16, n)
    return n


def pass_state(lib: Lib) -> dict:
    out = ctypes.create_string_buffer(160)
    lib.call("wt_pass_state", ctypes.addressof(out))
    b = out.raw
    return {"seed": b[:32], "k_s": b[32:64], "key": b[64:96]}


def gate_pass(lib: Lib, torch, sketch_t: int, log, trials: int = 4) -> dict:
    """wt_pass (red team F29: pass j's receipt keyed by s_j = pass_seed(σ, j, root_(j−1))) against Python at random
    j and roots: s_j, K_S, the tag key, the signs (sign_fragments of sign_bits(s_j)) and, through wt_tag_leaves over
    random tags, the leaves' prefix (commit_gate at s_j)."""
    n, m, st = set_pass(lib, sketch_t), SKT.steps(lib.sb), stream(torch)
    signs = torch.empty(sketch_t * m * 32, dtype=torch.int32, device="cuda")
    rng = random.Random(29)
    rows = []
    for t in range(trials):
        j = (0, (1 << 32) - 1)[t] if t < 2 else rng.randrange(1 << 32)
        root = rng.randbytes(64)
        lib.call("wt_pass", j, ptr(to_dev(torch, root_words(root))), ptr(signs), sketch_t, st)
        torch.cuda.synchronize()
        got = pass_state(lib)
        seed = SKT.pass_seed(TAG_SEED, j, root)
        ks = SKT.randomness().derive(seed, SKT.SIGN_DOMAIN, {"tag": SKT.NAME, "k": 16, "n": n}).value
        want = SKT.sign_fragments(SKT.sign_bits(seed, 16, n), m, sketch_t)
        ys = [rng.randbytes(32) for _ in range(40)]
        c = commit_gate(lib, torch, to_dev(torch, b"".join(ys)), len(ys), st, SKT.leaves, seed)
        rows.append({"j": j, "seed": got["seed"] == seed, "k_s": got["k_s"] == ks, "key": got["key"] == SKT.key(seed),
                     "signs": bool(np.array_equal(signs.cpu().numpy().view(np.uint32), want)),
                     "leaves": c["leaves_ok"], "root": c["root_ok"]})
    ok = all(all(v for a, v in r.items() if a != "j") for r in rows)
    log(f"pass gate {lib.name}: {'ok' if ok else 'FAIL'} over {trials} (j, root): {rows if not ok else ''}")
    return {"ok": ok, "trials": rows, "n": n}


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


def commit_time(lib: Lib, torch, nblk: int, reps: int) -> dict:
    """A decode step's commit, timed: the leaves over nblk tags (E's, all layers) in one launch, then the tree."""
    n = -(-nblk // TG.PER_LEAF)
    tags = torch.randint(0, 255, (nblk * 32,), dtype=torch.uint8, device="cuda")
    nodes = torch.empty(((2 * n + 64) * 64,), dtype=torch.uint8, device="cuda")
    ri = ctypes.c_ulonglong(0)
    tl = timed(torch, lambda: lib.call("wt_tag_leaves", ptr(tags), nblk, ptr(nodes), stream(torch)), reps)
    tt = timed(torch, lambda: lib.call("wt_tag_tree", ptr(nodes), n, ctypes.addressof(ri), stream(torch)), reps)
    return {"blocks": nblk, "leaves": n, "leaves_s": statistics.median(tl), "tree_s": statistics.median(tt),
            "leaves_times_s": tl, "tree_times_s": tt}


def dump_words(lib: Lib, dbg, plain, store, key, gq: int, stages: int) -> dict:
    """What the fused MMA warp consumed (a WT_DEBUG_DUMP build's dbg, plain fragment bytes a task) where it differs from
    the plain store: per bad word (task q, word w), its differing 16-byte chunks and whether its bytes are the stored
    word, zero, the payload of another word (the CTA's tasks q + j gq that share or neighbour its stage) or the decode
    of its stored word under another tweak index."""
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
    return out


def set_sketch_tag(lib: Lib) -> None:
    """ro-sketch-blake3/v0's key and leaf prefix at TAG_SEED into the library (set_tag restores ro-blake3/v1's)."""
    k = ctypes.create_string_buffer(SKT.key(TAG_SEED))
    pre = ctypes.create_string_buffer(SKT.leaf_prefix(TAG_SEED))
    lib.call("wt_set_tag", ctypes.addressof(k), ctypes.addressof(pre))


def gemm_case(lib: Lib, torch, name: str, n: int, kd: int, batches, ks, reps: int, log, cfgs, fused_cfgs,
              mma_knobs, imad_knobs, gate_reps: int = 1, dump: bool = False, sketch_t: int = 8) -> dict:
    g = torch.Generator(device="cuda").manual_seed(100 + n + kd)
    w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    plain = plain_layout(w.view(torch.uint8))
    m = task_payload_limbs(lib, torch, plain)
    keys, stores = {}, {}
    for k in ks:
        keys[k] = W.key(lib.field.name, SEED, k)
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
    nblk = TG.block_count(res["store_bytes"])
    tagged = [c for c in fused_cfgs if c[2] >= 0]
    tags_dev = torch.empty(nblk * 32, dtype=torch.uint8, device="cuda")
    refs = {k: tag_reference(stores[k], f"{lib.name} {name} k={k}") for k in ks} if tagged else {}
    res["tag_blocks"] = nblk
    sketched = [c for c in fused_cfgs if c[3]]
    sk = {}
    for s in sorted({c[3] for c in sketched}):
        bits = SKT.sign_bits(TAG_SEED, s, SKT.vector_bytes(lib.sb, sketch_t))
        words = SKT.sign_fragments(bits, SKT.steps(lib.sb), sketch_t)
        sk[s] = {"bits": bits, "signs": torch.from_numpy(words.view(np.int32)).cuda()}
    sk_bufs = {}
    sk_refs = {}

    def sk_tags_for(gr):
        """The tag buffer of a sketched launch at gr groups, and its vector count."""
        nvec = lib.so.wt_sketch_vectors(n, kd, gr, sketch_t)
        if gr not in sk_bufs:
            sk_bufs[gr] = torch.empty(nvec * 32, dtype=torch.uint8, device="cuda")
        return sk_bufs[gr], nvec

    def sketch_ref(k, s, gr, seed=TAG_SEED):
        """Every vector's sketch at `seed` on the device (float32, exact), and its tags with the PyPI blake3 (tags.py
        on a sample, which must agree)."""
        if (k, s, gr, seed) not in sk_refs:
            vs = SKT.vectors_torch(torch, stores[k], n // 16, nch, lib.sb, gr, sketch_t)
            bits = sk[s]["bits"] if seed == TAG_SEED else SKT.sign_bits(seed, s, SKT.vector_bytes(lib.sb, sketch_t))
            ref = SKT.sketch_torch(torch, vs, bits)
            del vs
            host = ref.cpu().numpy().astype(np.int64)
            pick = sorted({0, len(host) - 1, *random.Random(s + gr).sample(range(len(host)), 6)})
            sample = {v: SKT.tags_of(seed, host[v:v + 1], TAG_BASE + v)[0] for v in pick}
            ys = SKT.tags_fast(seed, host, TAG_BASE)
            if ys is not None and any(ys[v] != y for v, y in sample.items()):
                raise SystemExit(f"{lib.name} {name}: the PyPI blake3 disagrees with tags.py on the sketch tags")
            sk_refs[(k, s, gr, seed)] = {"sk": ref, "tags": b"".join(ys) if ys is not None else None,
                                         "sample": sample}
        return sk_refs[(k, s, gr, seed)]

    def pass_keyed(k, mb, x, pp, cfg, gr, sp):
        """cfg (two sketch warps) under pass j's key: wt_pass writes the signs and the key from a random root_(j−1),
        and the launch's sketches, tags, leaves and root must be ro-sketch-blake3/v0's at pass_seed(σ, j, root)."""
        j, root = 7, random.Random(n + kd + mb).randbytes(64)
        seed = SKT.pass_seed(TAG_SEED, j, root)
        set_pass(lib, sketch_t)
        signs = torch.empty_like(sk[16]["signs"])
        lib.call("wt_pass", j, ptr(to_dev(torch, root_words(root))), ptr(signs), sketch_t, st)
        tb_dev, nvec = sk_tags_for(gr)
        tb_dev.fill_(0xFF)
        skb = torch.full((nvec * 16,), -1, dtype=torch.int32, device="cuda")
        pf = torch.zeros_like(pp)
        call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp, skout=skb, signs=signs)
        torch.cuda.synchronize()
        r = sketch_ref(k, 16, gr, seed)
        out = {"cfg": fused_name(k, cfg), "j": j, "partials": bool(torch.equal(pf, pp)),
               "sketch": bool(torch.equal(skb.view(nvec, 16), r["sk"])), "tags": sketch_tags_ok(host_bytes(tb_dev), r)}
        c = commit_gate(lib, torch, tb_dev, nvec, st, SKT.leaves, seed)
        out.update(leaves=c["leaves_ok"], root=c["root_ok"])
        out["ok"] = all(v for a, v in out.items() if a not in ("cfg", "j"))
        sk_refs.pop((k, 16, gr, seed))
        set_sketch_tag(lib)
        return out

    def sketch_tags_ok(tb: bytes, r: dict) -> bool:
        if any(tb[32 * v:32 * v + 32] != y for v, y in r["sample"].items()):
            return False
        return r["tags"] is None or tb == r["tags"]

    def groups_for(mode, k, mb, cfg):
        launched = ctypes.c_int(0)
        if mode == 3:
            lib.call("wt_fused_sketch", k, None, None, None, mb, n, kd, 0, 0, cfg[0], cfg[1], 0, max(cfg[2], 0), None,
                     None, 0, cfg[3], None, None, sketch_t, sketch_warps(cfg), ctypes.byref(launched), st)
        else:
            lib.call("wt_gemm", mode, None, None, None, mb, n, kd, 0, cfg[0], cfg[1], 0, 0, None,
                     ctypes.byref(launched), st)
        return launched.value // nch

    def call(mode, k, wbuf, x, y, partials, cfg, knob=0, groups=0, steps=None, skout=None, signs=None):
        if mode == 3:
            tp = ptr(sk_tags_for(groups)[0]) if cfg[3] else ptr(tags_dev) if cfg[2] >= 0 else None
            sg = signs if signs is not None else sk[cfg[3]]["signs"] if cfg[3] else None
            lib.call("wt_fused_sketch", k, ptr(wbuf), ptr(x.view(torch.uint8)), ptr(y), x.shape[0], n, kd, 0,
                     int(partials), cfg[0], cfg[1], groups, max(cfg[2], 0), ptr(steps), tp, TAG_BASE, cfg[3],
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
                if (cfg[3] == 64 and k != 1) or not fused_fits(lib, k, mb, n, kd, cfg, sketch_t):
                    continue
                gr, sp = steps_for(k, mb, cfg)
                bad, sk_bad = [], []
                if cfg[3]:
                    set_sketch_tag(lib)
                    tb_dev, nvec = sk_tags_for(gr)
                    skr = sketch_ref(k, cfg[3], gr) if not lib.diag else None
                for _ in range(gate_reps):
                    pf = torch.zeros_like(pp)
                    tags_dev.fill_(0xFF)
                    skb = None
                    if cfg[3]:
                        tb_dev.fill_(0xFF)
                        skb = torch.full((nvec * cfg[3],), -1, dtype=torch.int32, device="cuda")
                    call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp, skout=skb)
                    if not torch.equal(pf, pp):
                        bad.append(mismatch(pf, pp, nch, mb, n, gr))
                    if cfg[3] and not lib.diag:
                        torch.cuda.synchronize()
                        sk_eq = torch.equal(skb.view(nvec, cfg[3]), skr["sk"])
                        tg_eq = sketch_tags_ok(host_bytes(tb_dev), skr)
                        if not (sk_eq and tg_eq):
                            sk_bad.append({"sketch": sk_eq, "tags": tg_eq,
                                           "vectors_unequal": int((skb.view(nvec, cfg[3]) != skr["sk"]).any(1).sum())})
                if cfg[3]:
                    entry.setdefault("sketch", {})[fused_name(k, cfg)] = {"vectors": nvec, "bad": sk_bad[:3],
                                                                          "ok": not sk_bad,
                                                                          "diagnostic": bool(lib.diag)}
                    entry["tags"][fused_name(k, cfg)] = not sk_bad
                    if sk_bad:
                        log(f"sketch mismatch {lib.name} {name} M={mb} {fused_name(k, cfg)}: {sk_bad[0]}")
                    if lib.diag:
                        res["sketch_commit"] = {"leaves_ok": True, "root_ok": True, "diagnostic": True}
                    if "sketch_commit" not in res and not sk_bad:
                        res["sketch_commit"] = commit_gate(lib, torch, tb_dev, nvec, st, SKT.leaves)
                        log(f"sketch commit gate {lib.name} {name}: {nvec} vectors, {res['sketch_commit']}")
                    if sketch_warps(cfg) and not lib.diag and not sk_bad and "pass_keyed" not in res:
                        res["pass_keyed"] = pass_keyed(k, mb, x, pp, cfg, gr, sp)
                        entry["tags"][fused_name(k, cfg) + "_pass"] = res["pass_keyed"]["ok"]
                        log(f"pass-keyed gate {lib.name} {name} M={mb}: {res['pass_keyed']}")
                    res["sketch_tags"] = nvec
                    set_tag(lib)
                entry["equal_plain"][fused_name(k, cfg)] = not bad
                if bad and dump:
                    dbg = torch.zeros_like(plain)
                    lib.call("wt_set_dbg", ptr(dbg))
                    pf = torch.zeros_like(pp)
                    call(3, k, stores[k], x, pf, True, cfg, groups=gr, steps=sp)
                    torch.cuda.synchronize()
                    lib.call("wt_set_dbg", None)
                    dw = dump_words(lib, dbg, plain, stores[k], keys[k], gr * nch, cfg[1])
                    dw["partials_equal"] = torch_equal(pf, pp)
                    entry.setdefault("dump", {})[fused_name(k, cfg)] = dw
                    log(f"dump {lib.name} {name} M={mb} {fused_name(k, cfg)}: {dw}")
                    del dbg
                if bad:
                    entry.setdefault("mismatch", {})[fused_name(k, cfg)] = {"failed": len(bad), "of": gate_reps,
                                                                            "first": bad[:3]}
                    log(f"mismatch {lib.name} {name} M={mb} {fused_name(k, cfg)}: {len(bad)}/{gate_reps} runs, "
                        f"first {bad[0]}")
                if cfg[2] >= 0:
                    torch.cuda.synchronize()
                    entry["tags"][fused_name(k, cfg)] = tags_ok(host_bytes(tags_dev), refs[k])
                    if "commit" not in res and entry["tags"][fused_name(k, cfg)]:
                        res["commit"] = commit_gate(lib, torch, tags_dev, nblk, st)
                        log(f"commit gate {lib.name} {name}: {res['commit']}")
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
                       and (not tagged or (res.get("commit", {}).get("leaves_ok") and res["commit"]["root_ok"]))
                       and (not sketched or (sk_commit.get("leaves_ok") and sk_commit["root_ok"])))
        res["gate"][mb] = entry
        log(f"gemm gate {lib.name} {name} M={mb}: {'ok' if entry['ok'] else 'FAIL'} (plain rel {err:.1e}) "
            + (f", tags {sum(entry['tags'].values())}/{len(entry['tags'])}" if tagged or sketched else "")
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
            tp = ptr(tags_dev) if mode == 3 and cfg[2] >= 0 else None
            nt = max(cfg[2], 0) if mode == 3 else 0
            sks, sgp = (cfg[3], ptr(sk[cfg[3]]["signs"])) if mode == 3 and cfg[3] else (0, None)
            skw = sketch_warps(cfg) if mode == 3 else 0
            if sks:
                tp = ptr(sk_tags_for(gr)[0])

            def f():
                rc = 0
                if mode == 3:
                    for p in ps:
                        rc |= so.wt_fused_sketch(k, p, xp, yp, mb, n, kd, 0, 0, cfg[0], cfg[1], gr, nt, spp, tp,
                                                 TAG_BASE, sks, sgp, None, sketch_t, skw, None, st)
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
                if not (cfg[3] == 64 and k != 1) and fused_fits(lib, k, mb, n, kd, cfg, sketch_t):
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
        def best(k, want_tags, s=0, w=0):
            cs = [c for c in fused_cfgs if (c[2] >= 0) == want_tags and c[3] == s and sketch_warps(c) == w
                  and fused_name(k, c) in med]
            return min((med[fused_name(k, c)], fused_name(k, c)[7:]) for c in cs) if cs else (float("nan"), "-")

        log(f"gemm {lib.name} {name} M={mb}: cublas {med['cublas'] * 1e6:.1f} us, plain best {plain_best * 1e6:.1f} us; "
            f"native {native * 1e6:.1f} us; "
            + ", ".join(f"fused k{k} {best(k, False)[0] / native:.2f}x ({best(k, False)[1]}), tagged "
                        f"{best(k, True)[0] / native:.2f}x ({best(k, True)[1]})" for k in ks)
            + "".join(f", sketch {s} k{k} {best(k, False, s)[0] / native:.2f}x ({best(k, False, s)[1]})"
                      for s in sorted({c[3] for c in sketched}) for k in ks if s == 16 or k == 1)
            + "".join(f", sketch warps k{k} {best(k, False, 16, 2)[0] / native:.2f}x ({best(k, False, 16, 2)[1]})"
                      for k in ks if any(sketch_warps(c) for c in sketched))
            + "; mma " + " ".join(f"{kn}:{med[f'mma{kn}'] / native:.2f}" for kn in mma_knobs)
            + "; imad " + " ".join(f"{kn}:{med[f'imad{kn}'] / native:.2f}" for kn in imad_knobs))
    del ring, wring, stores, plain, w
    torch.cuda.empty_cache()
    return res


def layer_ratios(gemm: dict, batches, ks) -> dict:
    """Per batch, a layer's time over its native time with each linear at its best arm: the plain kernel, and per k the
    fused decode untagged and tagged."""
    out = {}
    for mb in batches:
        meds = [g["times"][mb]["median_s"] for g in gemm.values()]
        native = sum(g["times"][mb]["native_s"] for g in gemm.values())
        row = {"cublas": sum(m["cublas"] for m in meds) / native,
               "plain": sum(min(v for a, v in m.items() if a.startswith("plain_")) for m in meds) / native}
        for k in ks:
            kinds = [(f"fused_k{k}", r"_d\d+s\d+$"), (f"fused_k{k}_tagged", r"t\d+$")]
            kinds += [(f"fused_k{k}_sketch{s}", rf"sk{s}$") for s in (16, 64)]
            kinds += [(f"fused_k{k}_sketch16_decoders", r"s\d+sk16$"), (f"fused_k{k}_sketch16_warps", r"w2sk16$")]
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
    ap.add_argument("--phases", default="rates,decode,b3,gemm")
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
                    help="fused kernel DECODERSxSTAGES[xTAGGERS], or auto (fused_cfgs)")
    ap.add_argument("--dump", action="store_true",
                    help="after a failed fused gate, rerun it once dumping the MMA warp's operands (WT_DEBUG_DUMP build)")
    ap.add_argument("--sketch", default="", help="k_s values (16 at every k, 64 at k = 1) for the sketched fused kernel")
    ap.add_argument("--sketch-t", type=int, default=8, help="tasks a sketch tile (n = 32 ceil(SB / 512) T bytes)")
    ap.add_argument("--mma-knobs", default="8,16,32,48,64,96")
    ap.add_argument("--imad-knobs", default="32,64,128,192,256")
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
    for lib in libs:
        set_tag(lib)
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
        set_key(lib, torch, W.key(lib.field.name, SEED, 2))
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
    if "b3" in phases:
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        r = max((dict(calibrated(torch, lambda s: libs[0].call("wt_chain", 4, 1, 1, s, ptr(xr), sms * m, stream(torch)),
                                 sms * m * 128, 0.08, a.reps), grid_per_sm=m) for m in (4, 8, 16)),
                key=lambda v: v["per_s"])
        r["bytes_per_s"] = {n: r["per_s"] * 4096 / c for n, c in zip(("fewest", "tag_warps"), B3_PER_BLOCK)}
        r["tau"] = {n: res["read"]["gbps"] * 1e9 / v for n, v in r["bytes_per_s"].items()}
        res["b3"] = r
        log(f"b3: {r['per_s']:.3e} compressions/s, {r['bytes_per_s']['fewest'] / 1e12:.2f} TB/s of message at the "
            f"fewest compressions ({r['bytes_per_s']['tag_warps'] / 1e12:.2f} at the tag warps'), tau per stored pass "
            f"{r['tau']['fewest']:.3f}")
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
            if any(sketch_warps(c) for c in fused_cfgs(lib, a.fused, a.sketch, a.sketch_t)) and not lib.diag:
                e["gate_pass"] = gate_pass(lib, torch, a.sketch_t, log)
                set_tag(lib)
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
            cfgs_l = fused_cfgs(lib, a.fused, a.sketch, a.sketch_t)
            for what, count, on in (("commit_step", "tag_blocks", any(c[2] >= 0 for c in cfgs_l)),
                                    ("sketch_commit_step", "sketch_tags", any(c[3] for c in cfgs_l))):
                if not on:
                    continue
                c = commit_time(lib, torch, layers * sum(g[count] for g in e["gemm"].values()), a.reps)
                for mb in batches:
                    step = layers * sum(g["times"][mb]["native_s"] for g in e["gemm"].values())
                    c.setdefault("y_commit", {})[mb] = (c["leaves_s"] + c["tree_s"]) / step
                e[what] = c
                log(f"{what} {lib.name}: {c['blocks']} tags, {c['leaves']} leaves ({layers} layers): leaves "
                    f"{c['leaves_s'] * 1e6:.0f} us, tree {c['tree_s'] * 1e6:.0f} us; y_commit "
                    + " ".join(f"M={mb}:{v:.3f}" for mb, v in c["y_commit"].items()))
                save()
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
