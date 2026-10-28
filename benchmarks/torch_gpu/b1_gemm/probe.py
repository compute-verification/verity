"""b1 wgmma (BGMMA.64x256x256 AND.POPC) on resident operands, alone and beside SHA-512, sustained.

Arms: `res` (one MMA warpgroup a CTA, a CTA an SM); `res+w4`, `res+w8` (4 or 8 hash warps in the same CTA, chaining
compressions until the warpgroup is done); `res+h1`, `res+h2` (1 or 2 hash_k CTAs an SM beside `res`, launched once its
first call's CTAs are resident). Every kernel asks for the largest shared-memory carveout, so a hash CTA never holds an SM
at a carveout the probe can't join.

Before an arm is timed, CTA 0's accumulator must equal iters x popc(A_m AND W_n) (numpy). Each item runs the arm back to
back for `--secs` (twice that beside hash_k CTAs) with `stamp_k` stamps every SEG_S, and reports its steady part (segments
starting at least SETTLE_S in): b1 MACs a SM clock against 32,768 (BGMMA at s8 n256's instruction rate, 8 x 4,096), the
SM clock from %clock64, the rate against THETA_B1 = 32,768 x 132 x 1.83 GHz, and the hash's compressions a second against
the bar, 4.355 G/s, and a SM clock against BAR_PER_SMCLK. Beside hash_k CTAs the tensor figures are the segments inside
the hash's span, and an item with none is marked `serialized`. Writes items.jsonl and summary.md under --out; the cubin is
$B1_CUBIN (run.sh builds it).
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

B1_PEAK = 32768  # b1 MACs a SM clock
THETA_B1 = B1_PEAK * 132 * 1.83e9
BAR_PER_SMCLK = 0.01803  # SHA-512 compressions a SM clock, sx's: 4.355 G/s on 132 SMs at 1,830 MHz
BAR = BAR_PER_SMCLK * 132 * 1.83e9
SETTLE_S, SEG_S = 1.0, 0.25
HASH_THREADS = 256
ATOM_MACS = 64 * 256 * 256
RES_ATOMS = 4  # an iteration: k 1,024 bits

u, i32, vp, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_uint64, C.POINTER
MAX_DYN_SMEM, CARVEOUT, SM_COUNT = 8, 9, 16
CUDA = {
    "cuDeviceGet": [P(i32), i32], "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp],
    "cuModuleLoadData": [P(vp), C.c_char_p], "cuModuleGetFunction": [P(vp), vp, C.c_char_p],
    "cuFuncSetAttribute": [vp, i32, i32], "cuDeviceGetAttribute": [P(i32), i32, i32],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuGetErrorString": [i32, P(C.c_char_p)],
}


class ResParams(C.Structure):
    _fields_ = [("a", u64), ("b", u64), ("acc", u64), ("st", u64), ("hn", u64), ("io", u64), ("started", u64),
                ("iters", C.c_int32), ("pad", C.c_int32)]


#: arm -> (kernel, threads)
RES = {"res": ("b1_res", 128), "w4": ("b1_res_w4", 256), "w8": ("b1_res_w8", 384)}


class Lib:
    def __init__(self, path: Path):
        self.cu = C.CDLL("libcuda.so.1")
        for f, a in CUDA.items():
            getattr(self.cu, f).argtypes, getattr(self.cu, f).restype = a, i32
        torch.cuda.init()
        dev, ctx = i32(), vp()
        self.ck("cuDeviceGet", C.byref(dev), torch.cuda.current_device())
        self.ck("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self.ck("cuCtxSetCurrent", ctx)
        self.mod = vp()
        self.ck("cuModuleLoadData", C.byref(self.mod), Path(path).read_bytes())
        sms = i32()
        self.ck("cuDeviceGetAttribute", C.byref(sms), SM_COUNT, dev)
        self.sms = sms.value
        out = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.launch(self.fn("b1_layout"), 1, 1, 0, [u64(out.data_ptr())])
        size, iters_at, self.res_smem = out.tolist()[:3]
        if (size, iters_at) != (C.sizeof(ResParams), ResParams.iters.offset):
            raise RuntimeError(f"b1_layout {size, iters_at} is not probe.py's ResParams: rebuild")
        self.stamp_smem = 120 * 1024  # two of these never fit an SM, so stamp_k's CTAs fall one an SM
        self.fns = {}
        for name in [k for k, _ in RES.values()] + ["hash_k", "wait_k", "stamp_k"]:
            f = self.fn(name)
            self.ck("cuFuncSetAttribute", f, CARVEOUT, 100)
            smem = self.res_smem + 1024 if name.startswith("b1_res") else self.stamp_smem
            self.ck("cuFuncSetAttribute", f, MAX_DYN_SMEM, smem)
            self.fns[name] = f

    def ck(self, f: str, *a) -> None:
        r = getattr(self.cu, f)(*a)
        if r:
            s = C.c_char_p()
            self.cu.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f} returned CUresult {r} ({s.value.decode() if s.value else 'unknown'})")

    def fn(self, name: str) -> vp:
        f = vp()
        self.ck("cuModuleGetFunction", C.byref(f), self.mod, name.encode())
        return f

    def launch(self, f, grid: int, threads: int, shared: int, args, stream: int | None = None) -> None:
        keep = list(args)
        params = (vp * len(keep))(*[C.addressof(a) for a in keep])
        s = torch.cuda.current_stream().cuda_stream if stream is None else stream
        self.ck("cuLaunchKernel", f, grid, 1, 1, threads, 1, 1, shared, vp(s), params, None)


class Res:
    """One probe launch: a CTA an SM, `iters` iterations of four BGMMAs each."""

    def __init__(self, lib: Lib, kind: str, a: torch.Tensor, b: torch.Tensor, iters: int, bufs: dict):
        self.lib, (self.name, self.threads) = lib, RES[kind]
        self.f, self.grid, self.iters = lib.fns[self.name], lib.sms, iters
        self.macs = self.grid * iters * RES_ATOMS * ATOM_MACS
        self.p = ResParams(a=a.data_ptr(), b=b.data_ptr(), acc=bufs["acc"].data_ptr(), hn=bufs["hn"].data_ptr(),
                           io=bufs["io"].data_ptr(), iters=iters)
        self.hash_threads = self.threads - 128

    def call(self, stamps: int = 0, started: int = 0) -> None:
        self.p.st, self.p.started = stamps, started
        self.lib.launch(self.f, self.grid, self.threads, self.lib.res_smem + 1024, [self.p])


class Run:
    def __init__(self, lib: Lib):
        self.lib, self.nsm = lib, lib.sms
        self.stamps = torch.zeros((4096, self.nsm, 6), dtype=torch.int64, device="cuda")
        self.arrive = torch.zeros(4096, dtype=torch.int32, device="cuda")
        self.hst = torch.zeros((4 * self.nsm, 6), dtype=torch.int64, device="cuda")
        self.hio = torch.zeros(4 * self.nsm * HASH_THREADS, dtype=torch.int64, device="cuda")
        self.started = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.side = torch.cuda.Stream()

    def stamp(self, seg: int) -> None:
        self.lib.launch(self.lib.fns["stamp_k"], self.nsm, 32, self.lib.stamp_smem,
                        [u64(self.stamps[seg].data_ptr()), u64(self.arrive[seg].data_ptr())])

    def hash(self, ctas: int, reps: int, stream: int, shared: int = 0) -> None:
        self.lib.launch(self.lib.fns["hash_k"], ctas, HASH_THREADS, shared,
                        [u64(self.hio.data_ptr()), i32(reps), u64(self.hst.data_ptr())], stream)

    def one_an_sm(self, a: torch.Tensor) -> bool:
        sms = a[:, 4].tolist()
        return len(set(sms)) == self.nsm and 0 <= min(sms) and max(sms) < self.nsm

    def segments(self, nseg: int, calls: list[int], macs: int) -> tuple[list[dict], int]:
        """The segments between stamps each one on every SM, and how many were dropped for a stamp that wasn't."""
        st = self.stamps[: nseg + 1].cpu()
        out = []
        for s in range(nseg):
            a, b = st[s], st[s + 1]
            if not (self.one_an_sm(a) and self.one_an_sm(b)):
                continue
            ia, ib = a[a[:, 4].argsort()], b[b[:, 4].argsort()]
            dt, dc = (ib[:, 0] - ia[:, 0]).double(), (ib[:, 2] - ia[:, 2]).double()
            out.append({"ns": float(dt.mean()), "clocks": float(dc.sum()), "macs": calls[s] * macs,
                        "gt0": int(ia[:, 0].min()), "gt1": int(ib[:, 0].max())})
        return out, nseg - len(out)

    @staticmethod
    def rates(segs: list[dict], nsm: int) -> dict:
        if not segs:
            return {}
        macs, clocks, ns = sum(s["macs"] for s in segs), sum(s["clocks"] for s in segs), sum(s["ns"] for s in segs)
        return {"s": ns * 1e-9, "segments": len(segs), "macs_per_smclk": macs / clocks, "of_peak": macs / clocks / B1_PEAK,
                "sm_mhz": clocks / ns / nsm * 1e3, "tmac_s": macs / (ns * 1e-9) * 1e-12,
                "of_theta_b1": macs / (ns * 1e-9) / THETA_B1}

    def item(self, name: str, op: Res, secs: float, t_call: float, hn: torch.Tensor, hash_ctas: int = 0,
             hash_reps: int = 0) -> dict:
        per = max(1, round(SEG_S / t_call))
        nseg = max(2, round(secs / (per * t_call)))
        calls = [per] * nseg
        self.arrive[: nseg + 1].zero_()
        self.started.zero_()
        hn.zero_()
        torch.cuda.synchronize()
        self.stamp(0)
        first = 0
        if hash_ctas:
            op.call(0, self.started.data_ptr())
            self.lib.launch(self.lib.fns["wait_k"], 1, 32, 0, [u64(self.started.data_ptr()), u(op.grid)],
                            self.side.cuda_stream)
            self.hash(hash_ctas, hash_reps, self.side.cuda_stream)
            first = 1
        for s in range(nseg):
            for _ in range(first if s == 0 else 0, per):
                op.call()
            self.stamp(s + 1)
        torch.cuda.synchronize()
        g0, g_end = int(self.stamps[0][:, 0].min()), int(self.stamps[nseg][:, 0].max())
        segs, dropped = self.segments(nseg, calls, op.macs)
        if not segs:
            raise RuntimeError(f"{name}: no segment's stamps were one on every SM")
        steady = [s for s in segs if s["gt0"] - g0 >= SETTLE_S * 1e9]
        rec = {"name": name, "secs": (g_end - g0) * 1e-9, "calls": sum(calls), "t_call": t_call, "dropped": dropped,
               "stretch": (g_end - g0) * 1e-9 / (sum(calls) * t_call), "all": self.rates(segs, self.nsm),
               "steady": self.rates(steady, self.nsm)}
        if op.hash_threads:
            comps = int(hn.item())
            clocks = sum(s["clocks"] for s in segs)
            rec["hash"] = {"where": f"{op.hash_threads // 32} warps in the probe CTA", "comps": comps,
                           "gcomp_s": comps / (g_end - g0), "of_bar": comps / ((g_end - g0) * 1e-9) / BAR,
                           "of_bar_per_smclk": comps / clocks / BAR_PER_SMCLK}
        if hash_ctas:
            hs = self.hst[:hash_ctas].cpu()
            w0, w1 = int(hs[:, 0].min()), int(hs[:, 1].max())
            comps = hash_ctas * HASH_THREADS * hash_reps
            hmhz = float((hs[:, 3] - hs[:, 2]).double().sum() / (hs[:, 1] - hs[:, 0]).double().sum() * 1e3)
            sms = hs[:, 4].tolist()
            inside = [s for s in segs if s["gt0"] >= w0 + SETTLE_S * 1e9 and s["gt1"] <= w1]
            rec["hash"] = {"where": f"{hash_ctas // self.nsm} hash_k CTAs an SM", "ctas": hash_ctas, "reps": hash_reps,
                           "s": (w1 - w0) * 1e-9, "gcomp_s": comps / (w1 - w0),
                           "of_bar": comps / ((w1 - w0) * 1e-9) / BAR, "sm_mhz": hmhz,
                           "of_bar_per_smclk": comps / ((w1 - w0) * 1e-9) / (self.nsm * hmhz * 1e6) / BAR_PER_SMCLK,
                           "sms": len(set(sms)), "max_ctas_an_sm": max(sms.count(v) for v in set(sms)),
                           "covered": segs[0]["gt0"] <= w0 and g_end >= w1}
            rec["beside"] = self.rates(inside, self.nsm)
            rec["serialized"] = not inside
        return rec

    def hash_alone(self, ctas_per_sm: int, secs: float) -> dict:
        ctas = ctas_per_sm * self.nsm
        shared = 100 * 1024 if ctas_per_sm == 2 else self.lib.stamp_smem
        s = torch.cuda.current_stream().cuda_stream
        self.hash(ctas, 64, s, shared)
        torch.cuda.synchronize()
        t = time.perf_counter()
        self.hash(ctas, 2048, s, shared)
        torch.cuda.synchronize()
        rep = (time.perf_counter() - t) / 2048
        reps = int(secs / rep)
        self.hash(ctas, reps, s, shared)
        torch.cuda.synchronize()
        hs = self.hst[:ctas].cpu()
        w0, w1 = int(hs[:, 0].min()), int(hs[:, 1].max())
        comps = ctas * HASH_THREADS * reps
        mhz = float((hs[:, 3] - hs[:, 2]).double().sum() / (hs[:, 1] - hs[:, 0]).double().sum() * 1e3)
        return {"name": f"hash x{ctas_per_sm}", "ctas": ctas, "reps": reps, "s": (w1 - w0) * 1e-9, "rep_s": rep,
                "sms": len(set(hs[:, 4].tolist())), "of_bar": comps / ((w1 - w0) * 1e-9) / BAR, "sm_mhz": mhz,
                "of_bar_per_smclk": comps / ((w1 - w0) * 1e-9) / (self.nsm * mhz * 1e6) / BAR_PER_SMCLK}


def popc_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """popc(A_m AND W_n) over all of each row: A (M x bytes), W (N x bytes), uint8."""
    table = np.array([bin(v).count("1") for v in range(256)], dtype=np.int64)
    return table[a[:, None, :] & b[None, :, :]].sum(-1)


def summary(results: list[dict]) -> str:
    lines = ["| item | round | b1 MACs/SM-clk / 32,768 | θ_b1 | SM MHz | hash of bar | hash / SM-clk | where | serialized |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        h = r.get("hash") or {}
        t = r.get("beside") if "beside" in r else r.get("steady") or {}
        f = lambda d, k, fmt: format(d[k], fmt) if d and k in d else "-"
        hb = f(r, "of_bar", ".3f") if r["name"].startswith("hash") else f(h, "of_bar", ".3f")
        hc = f(r, "of_bar_per_smclk", ".3f") if r["name"].startswith("hash") else f(h, "of_bar_per_smclk", ".3f")
        mhz = f(t, "sm_mhz", ".0f") if t else f(r, "sm_mhz", ".0f")
        lines.append(f"| {r['name']} | {r.get('round', '')} | {f(t, 'of_peak', '.4f')} | {f(t, 'of_theta_b1', '.4f')} | "
                     f"{mhz} | {hb} | {hc} | {h.get('where', '')} | {r.get('serialized', '')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cubin", type=Path, default=Path(os.environ.get("B1_CUBIN", "b1.cubin")))
    ap.add_argument("--arms", default="res,res+w4,res+w8,res+h1,res+h2")
    ap.add_argument("--iters", type=int, default=32768)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--hash-alone", default="1,2")
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lib = Lib(args.cubin)
    run = Run(lib)
    rng = np.random.default_rng(args.seed)
    a_np = rng.integers(0, 256, (64, 128), dtype=np.uint8)
    b_np = rng.integers(0, 256, (256, 128), dtype=np.uint8)
    a, b = torch.from_numpy(a_np).cuda(), torch.from_numpy(b_np).cuda()
    bufs = {"acc": torch.zeros((64, 256), dtype=torch.int32, device="cuda"),
            "hn": torch.zeros(1, dtype=torch.int64, device="cuda"),
            "io": torch.zeros(lib.sms * 384, dtype=torch.int64, device="cuda")}
    want = popc_ref(a_np, b_np)
    info = {"device": torch.cuda.get_device_properties(0).name, "sms": lib.sms, "torch": torch.__version__,
            "res_smem": lib.res_smem, "iters": args.iters, "b1_peak": B1_PEAK, "theta_b1": THETA_B1, "bar": BAR}
    (args.out / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    items = open(args.out / "items.jsonl", "a")
    arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    ops, results = {}, []
    for spec in arms:
        base, _, side = spec.partition("+")
        kind = side if side in ("w4", "w8") else "res"
        if kind in ops:
            continue
        op = Res(lib, kind, a, b, args.iters, bufs)
        bufs["acc"].fill_(-1)
        op.call()
        torch.cuda.synchronize()
        ok = bool((bufs["acc"].cpu().numpy().astype(np.int64) == want * args.iters).all())
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(3):
            op.call()
        e1.record()
        torch.cuda.synchronize()
        t_call = e0.elapsed_time(e1) / 3e3
        gate = {"arm": kind, "ok": ok, "t_call": t_call, "kernel": op.name, "threads": op.threads}
        print("GATE", json.dumps(gate), flush=True)
        if not ok:
            raise SystemExit(f"{kind}: CTA 0's accumulator is not iters x popc(A AND W)")
        ops[kind] = (op, t_call)
    for rnd in range(args.rounds):
        if rnd == 0 and args.hash_alone:
            for c in (int(v) for v in args.hash_alone.split(",")):
                rec = run.hash_alone(c, args.secs)
                rec["round"] = rnd
                print("ITEM", json.dumps(rec), flush=True)
                items.write(json.dumps(rec) + "\n")
                items.flush()
                results.append(rec)
        for spec in arms:
            base, _, side = spec.partition("+")
            kind = side if side in ("w4", "w8") else "res"
            op, t_call = ops[kind]
            if side.startswith("h"):
                per_sm = int(side[1:])
                alone = [r for r in results if r.get("name") == f"hash x{per_sm}"]
                rep = alone[0]["rep_s"] if alone else 15e-6
                rec = run.item(spec, op, 2 * args.secs, t_call, bufs["hn"], per_sm * lib.sms, int(0.6 * args.secs / rep))
            else:
                rec = run.item(spec, op, args.secs, t_call, bufs["hn"])
            rec["round"] = rnd
            print("ITEM", json.dumps(rec), flush=True)
            items.write(json.dumps(rec) + "\n")
            items.flush()
            results.append(rec)
    (args.out / "summary.md").write_text(summary(results))
    print(summary(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
