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
  plain FP8 kernel and the fused kernel at each k, each in two forms (one task a warp, `wc_gemm`; persistent with the
  next half-task's loads in flight, `wc_gemm_pipe`), every call on cold weights (a ring of copies past the L2), each
  arm queued behind a spin kernel so the device runs it back to back, beside an empty kernel at the GEMM's grid.
  Native is the fastest of cuBLASLt and the two plain kernels.
`--mode ncu-decode` / `ncu-gemm` run one call of each kernel for Nsight Compute and time nothing.
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

import codec as C  # noqa: E402

SEED = hashlib.sha256(b"verity/word-codec/bench/seed").digest()
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
        so.wc_spin.argtypes = [ctypes.c_longlong, v]
        so.wc_empty.argtypes = [i, v]
        cs = (ctypes.c_int * 6)()
        so.wc_gemm_constants(cs)
        self.constants = list(cs)
        if self.constants[:5] != [C.LANES, C.TASK_TILES, C.LANE_WORDS, C.TASK_WORDS, C.TASK_COLS]:
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


def fragment_payloads(torch, w):
    n, kd = w.shape
    w8 = w.view(n // 16, 2, 8, kd // C.TASK_COLS, C.TASK_TILES, 2, 4, 4)
    s = w8.permute(0, 3, 2, 6, 4, 5, 1, 7).contiguous().view(-1, C.LANES, C.LANE_BYTES)   # task, lane, stream
    pad = torch.zeros(s.shape[0], C.LANES, C.LANE_WORDS * C.PAYLOAD_BYTES, dtype=torch.uint8, device=w.device)
    pad[:, :, :C.LANE_BYTES] = s
    b = pad.view(s.shape[0], C.LANES, C.LANE_WORDS, C.PAYLOAD_BYTES).to(torch.int64)
    m = torch.zeros(b.shape[:3], dtype=torch.int64, device=w.device)
    for j in range(C.PAYLOAD_BYTES):
        m |= b[..., j] << (8 * j)
    return m.transpose(1, 2).contiguous().view(-1)                               # task, word, lane


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_codec(lib: Lib, torch, n_random: int, log) -> dict:
    rng = random.Random(7)
    edges = [0, 1, 2, (1 << 56) - 1, (1 << 55), 0x0123456789ABCD, (1 << 56) - 2]
    payloads = edges + [rng.getrandbits(56) for _ in range(n_random - len(edges))]
    base = (1 << 40) + 12345
    m = torch.tensor(payloads, dtype=torch.int64, device="cuda")
    y = torch.empty_like(m)
    back = torch.empty_like(m)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    out = {}
    sample = list(range(len(edges))) + rng.sample(range(len(edges), n_random), 120)
    for k in range(1, KMAX + 1):
        key = C.key(SEED, k)
        set_key(lib, torch, key)
        lib.call("wc_encode", ptr(m), ptr(y), m.numel(), base, k, 64, stream(torch))
        lib.call("wc_decode", ptr(y), ptr(back), m.numel(), base, k, ptr(xr), 64, stream(torch))
        torch.cuda.synchronize()
        yh = y.cpu().tolist()
        bad_enc = [i for i in sample if yh[i] != C.encode_word(payloads[i], base + i, key)]
        bad_dec = [i for i in sample if C.decode_word(yh[i], base + i, key) != payloads[i]]
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
    frag_ok = fragment_payloads(torch, wt).cpu().tolist() == C.fragment_payloads(w, n, kd)
    log(f"gate layouts: plain {plain_ok} fragment {frag_ok}")
    return {"plain": plain_ok, "fragment": frag_ok, "ok": plain_ok and frag_ok}


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
    per_step = {0: 4, 1: 4, 2: ilp, 3: 8}[what] * threads
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
    for k in range(1, KMAX + 1):
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


def gemm_case(lib: Lib, torch, name: str, n: int, kd: int, batches, ks, reps: int, log) -> dict:
    w = fp8_weights(torch, n, kd, 100 + n + kd)
    wb = w.view(torch.uint8)
    plain = plain_layout(wb)
    payloads = fragment_payloads(torch, wb)
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
    res = {"N": n, "K": kd, "weight_bytes": n * kd, "store_bytes": nwords * 8, "gate": {}, "times": {}}

    def run(k, wbuf, x, y, partials, pipe=False):
        if pipe:
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
        entry = {"plain_vs_torch_rel": err, "fused_equal_plain": {}, "pipe_equal_plain": {}}
        for k in [0, *ks]:
            for pipe, key in ((False, "fused_equal_plain"), (True, "pipe_equal_plain")):
                if k == 0 and not pipe:
                    continue
                pf = torch.zeros_like(pp)
                run(k, plain if k == 0 else stores[k], x, pf, True, pipe)
                entry[key][k] = bool(torch.equal(pf, pp))
        ya = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        run(ks[-1], stores[ks[-1]], x, ya, False)
        entry["fused_atomic_vs_torch_rel"] = float(((ya - ref).abs() / scale).max())
        ya.zero_()
        run(ks[-1], stores[ks[-1]], x, ya, False, True)
        entry["pipe_atomic_vs_torch_rel"] = float(((ya - ref).abs() / scale).max())
        try:
            yc = torch._scaled_mm(x, w.t(), torch.ones((), device="cuda"), torch.ones((), device="cuda"),
                                  out_dtype=torch.float32)
            entry["cublas_vs_torch_rel"] = float(((yc - ref).abs() / scale).max())
        except Exception as exc:  # noqa: BLE001
            entry["cublas_error"] = repr(exc)[:300]
        entry["ok"] = (err < 1e-5 and all(entry["fused_equal_plain"].values()) and all(entry["pipe_equal_plain"].values())
                       and entry["fused_atomic_vs_torch_rel"] < 1e-5 and entry["pipe_atomic_vs_torch_rel"] < 1e-5)
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

        def arm_empty():
            rc = 0
            for _ in range(copies):
                rc |= so.wc_empty(grid, st)
            if rc:
                raise RuntimeError(f"wc_empty: error {rc}")

        arms = {"cublas": arm_cublas, "plain": arm(0), "plain_pipe": arm_pipe(0), "empty": arm_empty}
        for k in ks:
            arms[f"k{k}"] = arm(k)
            arms[f"p{k}"] = arm_pipe(k)
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
        native = min(med["cublas"], med["plain"], med["plain_pipe"])
        net = {a: (med[a] - med["empty"]) / (med["plain"] - med["empty"]) for a in med if a != "empty"}
        res["times"][mb] = {"median_s": med, "times_s": times, "cublas_m": mpad, "host_enqueue_max_s": host_max,
                            "host_bound": host_max > 0.8 * spin_s, "pipe_grids": grids,
                            "native_s": native, "ratio_vs_native": {a: med[a] / native for a in med},
                            "ratio_vs_plain": {a: med[a] / med["plain"] for a in med},
                            "ratio_vs_plain_net_of_empty": net,
                            "gbps_stored": {a: (nwords * 8 if a[0] in "kp" and a[1:].isdigit() else n * kd) / med[a] / 1e9
                                            for a in med if a != "empty"}}
        log(f"gemm {name} M={mb}: cublas {med['cublas'] * 1e6:.1f} us, plain {med['plain'] * 1e6:.1f} us, "
            f"plain_pipe {med['plain_pipe'] * 1e6:.1f} us, empty {med['empty'] * 1e6:.1f} us; vs native "
            + ", ".join(f"k{k} {med[f'k{k}'] / native:.2f}x/{med[f'p{k}'] / native:.2f}x" for k in ks)
            + (" HOST-BOUND" if host_max > 0.8 * spin_s else ""))
    del ring, wring, stores, plain, w
    torch.cuda.empty_cache()
    return res


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
    ap.add_argument("--theta", type=float, default=0.11, help="the tensor cores' encode capacity over the CUDA cores'")
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
            pay = fragment_payloads(torch, w)
            for mb in batches:
                x = torch.randn(mb, kd, device="cuda").to(torch.float8_e4m3fn)
                y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
                for fn, extra in (("wc_gemm", ()), ("wc_gemm_pipe", (0, None))):
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

    gemm_lib = max(libs, key=lambda l: res["libs"][l.name]["rates"]["decode"][KMAX]["per_s"])
    res["gemm_lib"] = gemm_lib.name
    res["gemm"] = {}
    for name in a.shapes.split(","):
        n, kd = SHAPES[name]
        g = gemm_case(gemm_lib, torch, name, n, kd, batches, ks, a.reps, log)
        g["zeroing_s"] = {mb: gemm_zeroing(torch, n, mb, a.reps) for mb in batches}
        res["gemm"][name] = g
        res["ok"] &= g.get("ok", False)
        (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
    (out_dir / "word-codec.json").write_text(json.dumps(res, indent=1))
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "done with a failed gate")
    return 0 if res["ok"] else 5


def write_result(res: dict, libs: list[Lib], out_dir: Path) -> None:
    """`research/result/v0.1` in $RESEARCH_RUN_DIR (`tools/research/src/research/result.py`), when the run has one."""
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}]
    for k in range(1, KMAX + 1):
        ms.append({"name": f"decode_k{k}", "value": max(res["libs"][l.name]["decode"][k].get("gbps_stored", 0.0)
                                                       for l in libs), "unit": "GB/s"})
        row = res["analysis"]["k"][k]
        ms += [{"name": f"r_k{k}", "value": row["r"], "unit": "1"}, {"name": f"eps_k{k}", "value": row["eps"], "unit": "1"}]
        if "rebuild" in row:
            ms.append({"name": f"rebuild_k{k}", "value": row["rebuild"], "unit": "1"})
        ms.append({"name": f"rebuild_free_k{k}", "value": row["free"]["rebuild"], "unit": "1"})
    for name, g in res.get("gemm", {}).items():
        for mb, t in g.get("times", {}).items():
            for arm, ratio in t["ratio_vs_native"].items():
                if arm != "empty":
                    ms.append({"name": f"gemm_{name}_m{mb}_{arm}_vs_native", "value": ratio, "unit": "1"})
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
