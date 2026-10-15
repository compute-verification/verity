"""Squaring at p1024 = 2^1024 - 105 on one RTX PRO 6000: the int8 tensor-core squaring of `imma_sq.cu` gated bit for bit
against `kv.cu`'s, then every rate in the same process, on the same GPU.

- gate: one squaring of `--gate-words` random words (uniform below 2^1024) and the edge cases, canonical out, against
  kv.cu's `kv_prim` (op 4, its `sqr`, then op 3, its `canon`), and Python's `pow(x, 2, p)` on a sample; then a chain of
  `--chain-steps` squarings of `--chain-words` words fed lazily, against kv.cu's chain and `pow(x, 2^steps, p)`. Any
  mismatch exits 2 before a rate is taken.
- rates: kv.cu's squaring chain (`kv_chain` what 0, ILP 1 and 2, each library: the default form and FOLD 6) as
  `benchmarks/word_codec/kv_bench.py` times it; the tensor-core chain in its three modes (0 the squaring, 1 without
  MMAs, 2 without the fold and carries) at W = 1, 2, 4 words a warp and each CTA count an SM; the pipe's int8 MMA rate
  on register operands; and both units at once, time-boxed, at each split of a CTA's 8 warps.

    python bench_sq.py --lib default=kv.so --lib fold6=kv-f6.so --out OUT
"""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import statistics
import sys
from pathlib import Path

import torch

P = (1 << 1024) - 105
NL = 32
WARPS = 4
MIX_WARPS = 8
MMA_MACS = 16 * 8 * 32
TC_MMAS = 10  # the plan's MMAs a squaring
SEED = 20261015
SQ_REF, SQ_CEILING_FACTOR = 1.499e10, 1.597  # timing.SQUARINGS_PER_S["p1024"] = 1.597 x kv.cu's 1.499e10


def log(msg: str) -> None:
    print(msg, flush=True)


class Lib:
    def __init__(self, name: str, path: str) -> None:
        v, i, q, ll = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_longlong
        self.name, self.path = name, path
        so = self.so = ctypes.CDLL(str(Path(path).resolve()))
        so.kv_prim.argtypes = [i, v, v, v, i, v]
        so.kv_chain.argtypes = [i, i, i, q, i, v, v]
        so.mc_gate.argtypes = [v, v, v, i, i, v]
        so.mc_chain.argtypes = [i, i, q, i, v, v]
        so.mc_chain_occupancy.argtypes = [i, i]
        so.mc_mma.argtypes = [i, q, i, v, v]
        so.mc_mix.argtypes = [i, ll, i, v, v, v]
        self.fold6 = so.mc_fold6()

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{self.name}: {fn} error {rc}")


def st() -> int:
    return torch.cuda.current_stream().cuda_stream


def to_dev(xs: list[int]) -> torch.Tensor:
    b = bytearray(b"".join(x.to_bytes(128, "little") for x in xs))
    return torch.frombuffer(b, dtype=torch.int32).view(len(xs), NL).cuda()


def from_dev(t: torch.Tensor) -> list[int]:
    b = t.contiguous().cpu().numpy().tobytes()
    return [int.from_bytes(b[128 * i:128 * (i + 1)], "little") for i in range(t.shape[0])]


def edge_cases() -> list[int]:
    top = (1 << 1024) - 1
    xs = [0, 1, 2, 3, 104, 105, 106, P - 1, P - 2, (P - 1) // 2, (P + 1) // 2, P, P + 1, P + 103, P + 104, top, top - 1,
          1 << 1023, (1 << 1023) - 1, 1 << 512, (1 << 512) - 1, (1 << 512) + 1, 1 << 1016, top ^ (0xFF << 1016)]
    r = 1 << 512  # around sqrt(2^1024) and sqrt(p), where the square crosses 2^1024 and p
    xs += [r - k for k in range(1, 8)] + [r + k for k in range(0, 4)]
    xs += [0xFF << (8 * i) for i in range(128)] + [0xFFFFFFFF << (32 * i) for i in range(32)]
    xs += [int.from_bytes(bytes([0xFF, 0x00] * 64), "little"), int.from_bytes(bytes([0x00, 0xFF] * 64), "little"),
           int.from_bytes(bytes([0x80] * 128), "little"), int.from_bytes(bytes([0x7F] * 128), "little")]
    rng = random.Random(SEED + 1)
    xs += [top - rng.getrandbits(64) for _ in range(64)] + [P - 1 - rng.getrandbits(64) for _ in range(64)]
    return [x for x in xs if x < (1 << 1024)]


def mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a != b).any(dim=1).sum().item())


def gate(lib: Lib, words: int, chain_words: int, chain_steps: int, sample: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(SEED)
    rnd = torch.randint(-(1 << 31), 1 << 31, (words, NL), dtype=torch.int32, device="cuda", generator=g)
    edges = to_dev(edge_cases())
    a = torch.cat([edges, rnd]).contiguous()
    n = a.shape[0]
    kv_lazy, kv_can = torch.empty_like(a), torch.empty_like(a)
    mc_lazy, mc_can = torch.empty_like(a), torch.empty_like(a)
    lib.call("kv_prim", 4, a.data_ptr(), a.data_ptr(), kv_lazy.data_ptr(), n, st())
    lib.call("kv_prim", 3, kv_lazy.data_ptr(), kv_lazy.data_ptr(), kv_can.data_ptr(), n, st())
    lib.call("mc_gate", a.data_ptr(), mc_lazy.data_ptr(), mc_can.data_ptr(), n, 188 * 16, st())
    torch.cuda.synchronize()
    bad = mismatches(mc_can, kv_can)
    lazy_diff = mismatches(mc_lazy, kv_lazy)
    # Python's pow on the edge cases and a sample of the random words
    idx = list(range(edges.shape[0])) + random.Random(SEED + 2).sample(range(edges.shape[0], n), sample)
    xs, ys, ks = from_dev(a[idx]), from_dev(mc_can[idx]), from_dev(kv_can[idx])
    bad_pow = sum(1 for x, y in zip(xs, ys) if y != pow(x, 2, P))
    bad_pow_kv = sum(1 for x, y in zip(xs, ks) if y != pow(x, 2, P))
    first = [i for i in range(n) if not torch.equal(mc_can[i], kv_can[i])][:4] if bad else []
    # a chain fed lazily, against kv.cu's chain and pow(x, 2^steps, p)
    c0 = a[:chain_words].clone()
    x_kv, x_mc = c0.clone(), c0.clone()
    t_kv, t_mc, t_can = torch.empty_like(c0), torch.empty_like(c0), torch.empty_like(c0)
    for _ in range(chain_steps):
        lib.call("kv_prim", 4, x_kv.data_ptr(), x_kv.data_ptr(), t_kv.data_ptr(), chain_words, st())
        lib.call("mc_gate", x_mc.data_ptr(), t_mc.data_ptr(), t_can.data_ptr(), chain_words, 188 * 16, st())
        x_kv, t_kv = t_kv, x_kv
        x_mc, t_mc = t_mc, x_mc
    kc, mcn = torch.empty_like(c0), torch.empty_like(c0)
    lib.call("kv_prim", 3, x_kv.data_ptr(), x_kv.data_ptr(), kc.data_ptr(), chain_words, st())
    lib.call("kv_prim", 3, x_mc.data_ptr(), x_mc.data_ptr(), mcn.data_ptr(), chain_words, st())
    torch.cuda.synchronize()
    bad_chain = mismatches(mcn, kc)
    e = 1 << chain_steps
    cx, cy = from_dev(c0[:64]), from_dev(mcn[:64])
    bad_chain_pow = sum(1 for x, y in zip(cx, cy) if y != pow(x, e, P))
    ok = bad == 0 and bad_pow == 0 and bad_pow_kv == 0 and bad_chain == 0 and bad_chain_pow == 0
    out = {"words": n, "random_words": words, "edge_cases": edges.shape[0], "mismatch_vs_kv": bad,
           "lazy_differs_vs_kv": lazy_diff, "pow_checked": len(idx), "mismatch_vs_pow": bad_pow,
           "kv_mismatch_vs_pow": bad_pow_kv, "first_mismatches": first, "chain_words": chain_words,
           "chain_steps": chain_steps, "chain_mismatch_vs_kv": bad_chain, "chain_pow_checked": 64,
           "chain_mismatch_vs_pow": bad_chain_pow, "ok": ok}
    log(f"gate {lib.name}: {'ok' if ok else 'FAIL'}: {n} words ({edges.shape[0]} edge cases) canonical vs kv.cu "
        f"{bad} differ (lazy forms differ in {lazy_diff}), pow {bad_pow}/{len(idx)} (kv.cu {bad_pow_kv}); chain "
        f"{chain_words} x {chain_steps} vs kv.cu {bad_chain}, pow {bad_chain_pow}/64")
    return out


def timed(fn, reps: int, warm: int = 1) -> list[float]:
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


def calibrated(launch, target_s: float, reps: int) -> tuple[int, list[float]]:
    steps = 4
    while True:
        t = statistics.median(timed(lambda: launch(steps), 1))
        if t > target_s / 4 or steps > 1 << 26:
            break
        steps *= 4
    steps = max(1, int(steps * target_s / t))
    return steps, timed(lambda: launch(steps), reps)


def kv_chain(lib: Lib, ilp: int, sms: int, reps: int, target_s: float) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    best, rows = None, []
    for per_sm in (2, 4, 8):
        grid = sms * per_sm
        steps, ts = calibrated(lambda s: lib.call("kv_chain", 0, 1, ilp, s, grid, xr.data_ptr(), st()), target_s, reps)
        med = statistics.median(ts)
        items = grid * 128 * steps * ilp
        r = {"per_sm": per_sm, "steps": steps, "items": items, "median_s": med, "times_s": ts, "per_s": items / med}
        rows.append(r)
        if best is None or r["per_s"] > best["per_s"]:
            best = r
    return {"best": best, "rows": rows}


def mc_chain(lib: Lib, mode: int, w: int, sms: int, reps: int, target_s: float) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    occ = lib.so.mc_chain_occupancy(mode, w)
    best, rows = None, []
    for per_sm in sorted({c for c in (1, 2, 3, 4, 6, 8, 12, 16) if c <= occ} | {occ}):
        grid = sms * per_sm
        steps, ts = calibrated(lambda s: lib.call("mc_chain", mode, w, s, grid, xr.data_ptr(), st()), target_s, reps)
        med = statistics.median(ts)
        items = grid * WARPS * w * steps
        r = {"per_sm": per_sm, "warps_per_sm": per_sm * WARPS, "steps": steps, "items": items, "median_s": med,
             "times_s": ts, "per_s": items / med}
        rows.append(r)
        if best is None or r["per_s"] > best["per_s"]:
            best = r
    return {"mode": mode, "w": w, "occupancy": occ, "best": best, "rows": rows}


def mma_rate(lib: Lib, nacc: int, sms: int, reps: int, target_s: float) -> dict:
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    best, rows = None, []
    for per_sm in (1, 2, 4, 8, 12):
        grid = sms * per_sm
        steps, ts = calibrated(lambda s: lib.call("mc_mma", nacc, s, grid, xr.data_ptr(), st()), target_s, reps)
        med = statistics.median(ts)
        macs = grid * WARPS * steps * nacc * MMA_MACS
        r = {"per_sm": per_sm, "steps": steps, "macs": macs, "median_s": med, "times_s": ts, "macs_per_s": macs / med}
        rows.append(r)
        if best is None or r["macs_per_s"] > best["macs_per_s"]:
            best = r
    return {"nacc": nacc, "best": best, "rows": rows}


def mix(lib: Lib, sms: int, reps: int, seconds: float, mhz: float) -> list[dict]:
    cycles = int(seconds * mhz * 1e6)
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    counts = torch.zeros(2, dtype=torch.int64, device="cuda")
    rows = []
    for per_sm in (1, 2):
        for cw in range(MIX_WARPS + 1):
            grid = sms * per_sm
            res = []
            for _ in range(reps):
                counts.zero_()
                (t,) = timed(lambda: lib.call("mc_mix", cw, cycles, grid, counts.data_ptr(), xr.data_ptr(), st()), 1,
                             warm=0)
                c = counts.tolist()
                res.append((t, c[0], c[1]))
            t, cu, tc = sorted(res)[len(res) // 2]
            rows.append({"per_sm": per_sm, "cuda_warps": cw, "tc_warps": MIX_WARPS - cw, "median_s": t,
                         "cuda_squarings": cu, "tc_squarings": tc, "cuda_per_s": cu / t, "tc_per_s": tc / t,
                         "per_s": (cu + tc) / t})
            log(f"mix {lib.name} {per_sm}/SM cw={cw}: CUDA {cu / t:.3e} + TC {tc / t:.3e} = {(cu + tc) / t:.3e}/s")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate-words", type=int, default=1 << 20)
    ap.add_argument("--chain-words", type=int, default=1 << 16)
    ap.add_argument("--chain-steps", type=int, default=64)
    ap.add_argument("--pow-sample", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--target-s", type=float, default=0.05)
    ap.add_argument("--mix-s", type=float, default=0.1)
    ap.add_argument("--mhz", type=float, default=2092.0, help="the node's locked SM clock, for the time box only")
    ap.add_argument("--only", default="gate,kv,tc,mma,mix")
    ns = ap.parse_args()
    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    libs = [Lib(*s.split("=", 1)) for s in ns.lib]
    only = set(ns.only.split(","))
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    res: dict = {"device": props.name, "sms": sms, "p": "2^1024-105", "libs": {}, "mc_word_bytes": libs[0].so.mc_word_bytes()}
    for lib in libs:
        r = res["libs"][lib.name] = {"path": lib.path, "fold6": lib.fold6}
        if "gate" in only:
            r["gate"] = gate(lib, ns.gate_words, ns.chain_words, ns.chain_steps, ns.pow_sample)
            (out / "bench.json").write_text(json.dumps(res, indent=1))
            if not r["gate"]["ok"]:
                log("gate failed: no rates")
                return 2
    for lib in libs:
        r = res["libs"][lib.name]
        if "kv" in only:
            r["kv_square"] = {ilp: kv_chain(lib, ilp, sms, ns.reps, ns.target_s) for ilp in (1, 2)}
            log(f"kv.cu {lib.name}: squarings/s " + ", ".join(f"ILP {i} {x['best']['per_s']:.3e} ({x['best']['per_sm']}/SM)"
                                                              for i, x in r["kv_square"].items()))
        (out / "bench.json").write_text(json.dumps(res, indent=1))
    lib = libs[0]
    r = res["libs"][lib.name]
    if "tc" in only:
        r["tc"] = []
        for mode in (0, 1, 2):
            for w in (1, 2, 4):
                x = mc_chain(lib, mode, w, sms, ns.reps, ns.target_s)
                r["tc"].append(x)
                b = x["best"]
                log(f"tc mode {mode} W={w}: {b['per_s']:.3e}/s at {b['per_sm']} CTAs/SM ({b['warps_per_sm']} warps), "
                    f"occupancy {x['occupancy']}; rows " + ", ".join(f"{y['per_sm']}:{y['per_s']:.3e}" for y in x["rows"]))
                (out / "bench.json").write_text(json.dumps(res, indent=1))
    if "mma" in only:
        r["mma"] = [mma_rate(lib, n, sms, ns.reps, ns.target_s) for n in (1, 2, 4, 8)]
        for x in r["mma"]:
            b = x["best"]
            log(f"mma NACC={x['nacc']}: {b['macs_per_s']:.3e} int8 MAC/s at {b['per_sm']} CTAs/SM "
                f"({b['macs_per_s'] / (sms * ns.mhz * 1e6):.0f} a clock an SM at {ns.mhz:.0f} MHz)")
        (out / "bench.json").write_text(json.dumps(res, indent=1))
    if "mix" in only:
        r["mix"] = mix(lib, sms, max(3, ns.reps // 2), ns.mix_s, ns.mhz)
        (out / "bench.json").write_text(json.dumps(res, indent=1))
    if "ncu" in only:  # one short launch of each chain, for Nsight Compute (-k regex:chain_kernel)
        xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        for mode, w in ((0, 1), (0, 2), (1, 1), (2, 1)):
            lib.call("mc_chain", mode, w, 256, sms * lib.so.mc_chain_occupancy(mode, w), xr.data_ptr(), st())
        lib.call("kv_chain", 0, 1, 1, 256, sms * 4, xr.data_ptr(), st())
        torch.cuda.synchronize()
        return 0
    # summary
    summ: dict = {"sq_ref": SQ_REF, "sq_ceiling": SQ_REF * SQ_CEILING_FACTOR}
    for name, x in res["libs"].items():
        if "kv_square" in x:
            summ[f"kv_{name}"] = max(v["best"]["per_s"] for v in x["kv_square"].values())
    if "tc" in r:
        for mode in (0, 1, 2):
            summ[f"tc_mode{mode}"] = max(x["best"]["per_s"] for x in r["tc"] if x["mode"] == mode)
    if "mma" in r:
        summ["mma_macs_per_s"] = max(x["best"]["macs_per_s"] for x in r["mma"])
        summ["mma_cap_10mma"] = summ["mma_macs_per_s"] / (TC_MMAS * MMA_MACS)
        summ["mma_cap_full_16384"] = summ["mma_macs_per_s"] / 16384
        summ["mma_cap_sym_8256"] = summ["mma_macs_per_s"] / 8256
    if "mix" in r:
        bm = max(r["mix"], key=lambda y: y["per_s"])
        summ["mix_best"] = bm
    res["summary"] = summ
    (out / "bench.json").write_text(json.dumps(res, indent=1))
    log("summary " + json.dumps(summ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
