"""The exact s8 GEMM on 2-bit operands, CUTLASS's sm90 configurations from libu2gemm.so (build.sh), sustained on one GPU.

For each shape (M, N, K), the operands are drawn uniformly from {lo .. hi} (default the unsigned 2-bit range {0 .. 3}) as
int8, A M×K and B N×K, from a seeded generator, so they stream from HBM in every call (gigabytes, past L2). Then:

- the gate: each configuration's D = A·Bᵀ, written into a buffer poisoned with 0xA5 bytes, equals torch._int_mm(A, B.t())
  bit for bit, on two calls, and every CTA of each call stamped; a configuration that fails it is never timed;
- items of --item-s seconds each, interleaved round by round: each configuration alone, its calls back to back on one
  stream, then cuBLASLt (torch._int_mm) on the same operands. An item's MACs per SM clock are its MACs over its whole span
  (the first CTA's start to the last one's end, launch gaps included) times the SM clock its CTAs stamped (clock64 over
  %globaltimer) times the SMs; θ is its MAC rate over 989.4 T MAC/s. cuBLASLt's span is its CUDA events', and it has no
  clock stamps, so its MACs per SM clock use the clock of the configurations' items in the same round;
- beside items: a configuration's calls on one stream and SHA-512 compressions (hash.cu) in one CTA per SM on a second,
  launched once every GEMM CTA of the first call is resident, so each fits beside a GEMM CTA; the hash's rate is over the
  4.355 G/s bar, and the GEMM's is over the calls inside the hash's window. A hash-alone item of the same form precedes it;
- fused items: a hook configuration (U2_HOOK) timed as a GEMM item with its producer warpgroup's idle warps hashing, the
  hash's rate its compressions over the item's span. The gate's second call of a hook configuration runs with the hook on.

nvidia-smi samples the board every 100 ms (--smi-ms); an item's watts are the mean and max of its samples after its first
quarter second, and pJ per MAC its mean watts over its MAC rate. Writes items.jsonl, gates.json, configs.json and
summary.md into --out.
"""

import argparse
import ctypes
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time

import torch

SMS = 132
PEAK_CLK = 4096  # s8 MACs a SM clock
THETA = PEAK_CLK * SMS * 1.83e9  # 989.4 T MAC/s at the clock wgmma runs
BAR = 4.355e9  # SHA-512 compressions a second
BAR_CLK = BAR / (SMS * 1.83e9)  # the same a SM clock
TARGETS = (0.985, 0.97)


class Lib:
    def __init__(self, path):
        self.l = ctypes.CDLL(path)
        L = self.l
        for f, args, res in (
            ("u2_hash_info", [ctypes.c_int, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
            ("u2_hash_run", [ctypes.c_int] * 3 + [ctypes.c_void_p] * 3, ctypes.c_int),
            ("u2_wait", [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p], ctypes.c_int),
        ):
            getattr(L, f).argtypes, getattr(L, f).restype = args, res

    def fn(self, name, f, args, res):
        g = getattr(self.l, f"{name}_{f}")
        g.argtypes, g.restype = args, res
        return g

    def info(self, name):
        buf = ctypes.create_string_buffer(2048)
        self.fn(name, "info", [ctypes.c_char_p, ctypes.c_int], ctypes.c_int)(buf, 2048)
        return json.loads(buf.value.decode())

    def hash_info(self, form):
        buf = ctypes.create_string_buffer(512)
        self.l.u2_hash_info(form, buf, 512)
        return json.loads(buf.value.decode())


class Gemm:
    def __init__(self, lib, name, A, B, D, raster, swizzle):
        P = ctypes.c_void_p
        self.name, self.lib = name, lib
        make = lib.fn(name, "make", [ctypes.c_int] * 3 + [P] * 3 + [ctypes.c_int] * 3, P)
        self._run = lib.fn(name, "run", [P, P, P], ctypes.c_int)
        self._free = lib.fn(name, "free", [P], None)
        self._hook = lib.fn(name, "hook", [P, P, P, ctypes.c_int], ctypes.c_int)
        M, K = A.shape
        N = B.shape[0]
        self.macs = M * N * K
        self.h = make(M, N, K, A.data_ptr(), B.data_ptr(), D.data_ptr(), raster, swizzle, 0)
        if not self.h:
            raise RuntimeError(f"{name}: CUTLASS refused M {M} N {N} K {K}")
        g = (ctypes.c_int * 4)()
        lib.fn(name, "grid", [P, P], None)(self.h, g)
        self.grid, self.clusters = (g[0], g[1], g[2]), g[3]
        self.ctas = g[0] * g[1] * g[2]

    def run(self, stream, rec_ptr):
        e = self._run(self.h, stream.cuda_stream, rec_ptr)
        if e:
            raise RuntimeError(f"{self.name}: launch error {e}")

    def hook(self, bufs):
        """The fused hash on (bufs: sink, done, count tensors) or off (None), between calls."""
        torch.cuda.synchronize()
        p = [b.data_ptr() for b in bufs] if bufs else [None] * 3
        e = self._hook(*p, 1 if bufs else 0)
        if e:
            raise RuntimeError(f"{self.name}: hook error {e}")

    def close(self):
        self._free(self.h)


class Smi:
    """nvidia-smi at a fixed period on one GPU, each line stamped with the host's time."""

    FIELDS = ["power.draw", "power.draw.instant", "clocks.sm", "clocks.mem", "temperature.gpu",
              "clocks_throttle_reasons.active", "enforced.power.limit"]

    def __init__(self, uuid, ms, path):
        sel = ["-i", uuid] if uuid else []
        fields = list(self.FIELDS)
        probe = subprocess.run(["nvidia-smi", *sel, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
        if probe.returncode != 0 or "Field" in probe.stdout + probe.stderr:
            fields.remove("power.draw.instant")
        self.fields, self.rows, self.f = fields, [], open(path, "w")
        self.f.write("host_t," + ",".join(fields) + "\n")
        self.p = subprocess.Popen(["nvidia-smi", *sel, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
                                   "-lms", str(ms)], stdout=subprocess.PIPE, text=True, bufsize=1)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            t = time.time()
            self.f.write(f"{t:.3f},{line.strip()}\n")
            self.rows.append((t, dict(zip(self.fields, (x.strip() for x in line.split(","))))))

    def window(self, t0, t1, skip=0.25):
        s = [r for t, r in self.rows if t0 + skip <= t <= t1]

        def nums(k):
            return [float(r[k]) for r in s if re.fullmatch(r"-?\d+(\.\d+)?", r.get(k, ""))]

        w = nums("power.draw")
        wi = nums("power.draw.instant")
        mhz = nums("clocks.sm")
        thr = [r.get("clocks_throttle_reasons.active", "") for r in s]
        return {"samples": len(s), "w_mean": statistics.fmean(w) if w else None, "w_max": max(w) if w else None,
                "w_inst_mean": statistics.fmean(wi) if wi else None, "w_inst_max": max(wi) if wi else None,
                "smi_mhz": statistics.median(mhz) if mhz else None,
                "throttle": statistics.mode(thr) if thr else None}

    def close(self):
        self.p.terminate()
        self.p.wait()
        self.f.close()


def recs_of(t):
    """CallRec rows (nt0, t1, cyc, ns, n | started << 32) as dicts."""
    out = []
    for nt0, t1, cyc, ns, nn in t.cpu().tolist():
        out.append({"t0": (~nt0) & (2**64 - 1), "t1": t1 & (2**64 - 1), "cyc": cyc, "ns": ns, "n": nn & 0xFFFFFFFF,
                    "started": (nn >> 32) & 0xFFFFFFFF})
    return out


def span_stats(recs, macs_per_call, stamps_per_call):
    bad = [i for i, r in enumerate(recs) if r["n"] != stamps_per_call]
    t0, t1 = min(r["t0"] for r in recs), max(r["t1"] for r in recs)
    span = (t1 - t0) * 1e-9
    ghz = sum(r["cyc"] for r in recs) / sum(r["ns"] for r in recs)
    macs = macs_per_call * len(recs)
    gaps = [(recs[i + 1]["t0"] - recs[i]["t1"]) * 1e-3 for i in range(len(recs) - 1)]
    calls = [(r["t1"] - r["t0"]) * 1e-3 for r in recs]
    return {"calls": len(recs), "span_s": span, "sm_mhz": ghz * 1e3, "macs": macs, "mac_per_s": macs / span,
            "theta": macs / span / THETA, "mac_per_clk": macs / span / (ghz * 1e9 * SMS),
            "frac_peak_clk": macs / span / (ghz * 1e9 * SMS) / PEAK_CLK,
            "call_us_median": statistics.median(calls), "gap_us_median": statistics.median(gaps) if gaps else None,
            "gap_us_max": max(gaps) if gaps else None, "calls_missing_stamps": bad[:8]}


def poison(D):
    D.view(torch.uint8).fill_(0xA5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default="12288x11264x32768,6144x5632x32768,1536x2816x32768,12288x11264x8192")
    ap.add_argument("--configs", default="", help="comma-separated names (default: every one in build.json)")
    ap.add_argument("--beside", default="c128x256_c2x1_ns4_r160:0,c128x256_c2x1_ns4_r160:1",
                    help="config:hash form pairs for the beside items")
    ap.add_argument("--hash-ref", type=int, default=4, help="a hash form timed alone each round (-1: none)")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--item-s", type=float, default=2.0)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261028)
    ap.add_argument("--raster", type=int, default=0)
    ap.add_argument("--swizzle", type=int, default=1)
    ap.add_argument("--smi-ms", type=int, default=100)
    ap.add_argument("--gpu", default="", help="the GPU's UUID for nvidia-smi (the lease's), else every GPU")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    build = json.load(open(os.path.join(os.path.dirname(a.lib), "build.json")))
    names = [n for n in (a.configs.split(",") if a.configs else build["configs"]) if n]
    beside = [(c.split(":")[0], int(c.split(":")[1])) for c in a.beside.split(",") if c]
    lib = Lib(a.lib)
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    assert props.multi_processor_count == SMS, props
    infos = {n: lib.info(n) for n in names}
    hforms = sorted({f for _, f in beside} | ({a.hash_ref} if a.hash_ref >= 0 else set()))
    infos["hash"] = {f: lib.hash_info(f) for f in hforms}
    json.dump({"device": props.name, "torch": torch.__version__, "cuda": torch.version.cuda, "configs": infos,
               "args": vars(a)}, open(os.path.join(a.out, "configs.json"), "w"), indent=1)
    print(json.dumps(infos), flush=True)
    smi = Smi(a.gpu, a.smi_ms, os.path.join(a.out, "smi.csv"))
    items = open(os.path.join(a.out, "items.jsonl"), "a")
    gates = []
    sg, sh = torch.cuda.Stream(), torch.cuda.Stream()

    def emit(row):
        row["smi"] = smi.window(row["host_t0"], row["host_t1"])
        w = row["smi"]["w_mean"]
        if w and row.get("mac_per_s"):
            row["pj_per_mac"] = w / row["mac_per_s"] * 1e12
        items.write(json.dumps(row) + "\n")
        items.flush()
        keep = {k: row.get(k) for k in ("arm", "config", "hash_form", "shape", "round", "frac_peak_clk", "theta",
                                        "sm_mhz", "hash_frac_bar", "pj_per_mac")}
        print("ITEM", json.dumps(keep), "W", row["smi"]["w_mean"], row["smi"]["w_max"], flush=True)

    for shape in a.shapes.split(","):
        M, N, K = map(int, shape.split("x"))
        g = torch.Generator(device=dev).manual_seed(a.seed + M + 7 * N + 13 * K)
        with torch.cuda.stream(sg):
            A = torch.randint(a.lo, a.hi + 1, (M, K), dtype=torch.int8, device=dev, generator=g)
            B = torch.randint(a.lo, a.hi + 1, (N, K), dtype=torch.int8, device=dev, generator=g)
            R = torch._int_mm(A, B.t())
            D = torch.empty((M, N), dtype=torch.int32, device=dev)
            Dlt = torch.empty_like(D)
        torch.cuda.synchronize()
        rmax = int(R.abs().max())
        hook_bufs = (torch.zeros(SMS * 96, dtype=torch.int64, device=dev),
                     torch.zeros(1024, dtype=torch.int32, device=dev), torch.zeros(1, dtype=torch.int64, device=dev))
        gemms = {}
        for n in sorted(set(names) | {c for c, _ in beside if c in names}):
            try:
                G = Gemm(lib, n, A, B, D, a.raster, a.swizzle)
            except RuntimeError as e:
                gates.append({"shape": shape, "config": n, "ok": False, "why": str(e)})
                print("GATE", gates[-1], flush=True)
                continue
            ok, why = True, []
            hooked = infos[n].get("hook") == 1
            for call in range(2):
                with torch.cuda.stream(sg):
                    poison(D)
                    rec = torch.zeros((1, 5), dtype=torch.int64, device=dev)
                if hooked and call == 1:
                    hook_bufs[2].zero_()
                    G.hook(hook_bufs)
                G.run(sg, rec.data_ptr())
                torch.cuda.synchronize()
                if hooked and call == 1:
                    G.hook(None)
                    if int(hook_bufs[2].item()) == 0:
                        why.append("call 1: the hook ran no compressions")
                r = recs_of(rec)[0]
                eq = bool(torch.equal(D, R))
                if not eq:
                    diff = int((D != R).sum())
                    why.append(f"call {call}: {diff} of {M * N} words differ from torch._int_mm")
                if r["n"] != 2 * G.ctas or r["started"] != G.ctas:
                    why.append(f"call {call}: {r['n']} consumer stamps and {r['started']} starts for {G.ctas} CTAs")
                ok &= eq and not why
            gates.append({"shape": shape, "config": n, "ok": ok, "why": why, "grid": G.grid, "clusters": G.clusters,
                          "ctas": G.ctas, "ref_abs_max": rmax})
            print("GATE", json.dumps(gates[-1]), flush=True)
            json.dump(gates, open(os.path.join(a.out, "gates.json"), "w"), indent=1)
            if ok:
                gemms[n] = G
            else:
                G.close()
        if not gemms:
            continue
        macs = M * N * K

        def calibrate(G):
            rec = torch.zeros((3, 5), dtype=torch.int64, device=dev)
            torch.cuda.synchronize()
            for i in range(3):
                G.run(sg, rec[i].data_ptr())
            torch.cuda.synchronize()
            return span_stats(recs_of(rec), macs, 2 * G.ctas)["span_s"] / 3

        def gemm_item(G, rnd, secs, fused=False):
            ncalls = max(2, math.ceil(secs / calibrate(G)))
            rec = torch.zeros((ncalls, 5), dtype=torch.int64, device=dev)
            if fused:
                hook_bufs[2].zero_()
                G.hook(hook_bufs)
            torch.cuda.synchronize()
            h0 = time.time()
            for i in range(ncalls):
                G.run(sg, rec[i].data_ptr())
            torch.cuda.synchronize()
            h1 = time.time()
            row = {"arm": "cutlass+fused" if fused else "cutlass", "config": G.name, "shape": shape, "round": rnd,
                   "host_t0": h0, "host_t1": h1}
            row.update(span_stats(recs_of(rec), macs, 2 * G.ctas))
            if fused:
                G.hook(None)
                comps = int(hook_bufs[2].item())
                rate = comps / row["span_s"]
                row.update({"hash_form": "fused", "hash_compressions": comps, "hash_per_s": rate,
                            "hash_frac_bar": rate / BAR,
                            "hash_frac_bar_clk": rate / (row["sm_mhz"] * 1e6 * SMS) / BAR_CLK})
            emit(row)
            return row

        def lt_item(rnd, secs, mhz):
            with torch.cuda.stream(sg):
                for _ in range(3):
                    torch._int_mm(A, B.t(), out=Dlt)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(sg):
                e0.record()
                for _ in range(3):
                    torch._int_mm(A, B.t(), out=Dlt)
                e1.record()
            torch.cuda.synchronize()
            ncalls = max(2, math.ceil(secs / (e0.elapsed_time(e1) * 1e-3 / 3)))
            h0 = time.time()
            with torch.cuda.stream(sg):
                e0.record()
                for _ in range(ncalls):
                    torch._int_mm(A, B.t(), out=Dlt)
                e1.record()
            torch.cuda.synchronize()
            h1 = time.time()
            span = e0.elapsed_time(e1) * 1e-3
            rate = macs * ncalls / span
            row = {"arm": "cublaslt", "config": "torch._int_mm", "shape": shape, "round": rnd, "host_t0": h0,
                   "host_t1": h1, "calls": ncalls, "span_s": span, "mac_per_s": rate, "theta": rate / THETA,
                   "sm_mhz_assumed": mhz, "exact": bool(torch.equal(Dlt, R))}
            if mhz:
                row["mac_per_clk"] = rate / (mhz * 1e6 * SMS)
                row["frac_peak_clk"] = row["mac_per_clk"] / PEAK_CLK
            emit(row)

        def hash_alone(form, rnd, secs):
            io = torch.zeros(SMS * 512, dtype=torch.int64, device=dev)
            st = torch.zeros((SMS, 5), dtype=torch.int64, device=dev)
            torch.cuda.synchronize()
            reps = 256
            while True:
                t = time.time()
                lib.l.u2_hash_run(form, SMS, reps, io.data_ptr(), st.data_ptr(), sh.cuda_stream)
                torch.cuda.synchronize()
                if time.time() - t > 0.2:
                    break
                reps *= 4
            reps = int(reps * secs / (time.time() - t))
            h0 = time.time()
            lib.l.u2_hash_run(form, SMS, reps, io.data_ptr(), st.data_ptr(), sh.cuda_stream)
            torch.cuda.synchronize()
            h1 = time.time()
            row = {"arm": "hash", "config": f"form{form}", "shape": shape, "round": rnd, "host_t0": h0, "host_t1": h1}
            row.update(hash_stats(st, form, reps))
            emit(row)
            return reps / max(row["window_s"], 1e-9)

        def hash_stats(st, form, reps):
            s = st.cpu().tolist()
            t0 = min(x[0] for x in s)
            t1 = max(x[1] for x in s)
            win = (t1 - t0) * 1e-9
            ghz = sum(x[3] - x[2] for x in s) / sum(x[1] - x[0] for x in s)
            threads = infos["hash"][form]["threads"]
            comps = SMS * threads * reps
            sms = len({x[4] & 0xFFFFFFFF for x in s})
            return {"hash_t0": t0, "hash_t1": t1, "window_s": win, "hash_sm_mhz": ghz * 1e3, "hash_per_s": comps / win,
                    "hash_frac_bar": comps / win / BAR, "hash_frac_bar_clk": comps / win / (ghz * 1e9 * SMS) / BAR_CLK,
                    "hash_sms": sms, "hash_threads": threads}

        def beside_item(G, form, rnd, secs, reps_per_s):
            # the GEMM's calls outlast the hash: its rate beside is unknown, so size the hash to secs at reps_per_s and
            # give the GEMM 1.5 x secs + 0.5 s
            t_call = calibrate(G)
            ncalls = max(4, math.ceil((1.5 * secs + 0.5) / t_call))
            reps = max(16, int(secs * reps_per_s))
            rec = torch.zeros((ncalls, 5), dtype=torch.int64, device=dev)
            io = torch.zeros(SMS * 512, dtype=torch.int64, device=dev)
            st = torch.zeros((SMS, 5), dtype=torch.int64, device=dev)
            torch.cuda.synchronize()
            h0 = time.time()
            lib.l.u2_wait(rec[0].data_ptr() + 36, G.ctas, sh.cuda_stream)
            lib.l.u2_hash_run(form, SMS, reps, io.data_ptr(), st.data_ptr(), sh.cuda_stream)
            for i in range(ncalls):
                G.run(sg, rec[i].data_ptr())
            sh.synchronize()
            hh = time.time()
            torch.cuda.synchronize()
            h1 = time.time()
            hs = hash_stats(st, form, reps)
            recs = recs_of(rec)
            inside = [r for r in recs if r["t0"] >= hs["hash_t0"] and r["t1"] <= hs["hash_t1"]]
            row = {"arm": "cutlass+hash", "config": G.name, "hash_form": form, "shape": shape, "round": rnd,
                   "host_t0": h0, "host_t1": hh, "host_gemm_end": h1, "calls_total": ncalls,
                   "covered": min(r["t0"] for r in recs) <= hs["hash_t0"] and max(r["t1"] for r in recs) >= hs["hash_t1"]}
            row.update(hs)
            if len(inside) >= 2:
                row.update(span_stats(inside, macs, 2 * G.ctas))
            emit(row)
            return row

        for rnd in range(a.rounds):
            mhz = []
            for n, G in gemms.items():
                if n in names:
                    mhz.append(gemm_item(G, rnd, a.item_s)["sm_mhz"])
            lt_item(rnd, a.item_s, statistics.fmean(mhz) if mhz else None)
            for n, G in gemms.items():
                if n in names and infos[n].get("hook") == 1:
                    gemm_item(G, rnd, a.item_s, fused=True)
            if a.hash_ref >= 0:
                hash_alone(a.hash_ref, rnd, a.item_s)
            for n, form in beside:
                if n not in gemms:
                    continue
                rps = hash_alone(form, rnd, a.item_s)
                beside_item(gemms[n], form, rnd, a.item_s, rps * 0.5)
        for G in gemms.values():
            G.close()
    smi.close()
    items.close()
    summarize(os.path.join(a.out, "items.jsonl"), os.path.join(a.out, "summary.md"))


def summarize(path, out):
    rows = [json.loads(line) for line in open(path)]
    lines = ["| arm | config | shape | items | MACs/clk ÷ 4096 | θ | SM MHz | W mean | W max | pJ/MAC | hash ÷ bar | "
             "gap to 0.985 | gap to 0.97 |", "|" + "---|" * 13]
    groups = {}
    for r in rows:
        hf = r.get("hash_form")
        key = (r["arm"], r["config"] + ("" if hf is None else f" + {hf}" if isinstance(hf, str) else f" + form{hf}"),
               r["shape"])
        groups.setdefault(key, []).append(r)

    def mean(rs, k):
        v = [x[k] for x in rs if x.get(k) is not None]
        return statistics.fmean(v) if v else None

    def f(x, p=3):
        return "-" if x is None else f"{x:.{p}f}"

    for (arm, cfg, shape), rs in groups.items():
        fr = mean(rs, "frac_peak_clk")
        wm = [x["smi"]["w_mean"] for x in rs if x["smi"]["w_mean"]]
        wx = [x["smi"]["w_max"] for x in rs if x["smi"]["w_max"]]
        lines.append(" | ".join(["", arm, cfg, shape, str(len(rs)), f(fr), f(mean(rs, "theta")),
                                 f(mean(rs, "sm_mhz") or mean(rs, "sm_mhz_assumed") or mean(rs, "hash_sm_mhz"), 0),
                                 f(statistics.fmean(wm) if wm else None, 0), f(max(wx) if wx else None, 0),
                                 f(mean(rs, "pj_per_mac"), 3), f(mean(rs, "hash_frac_bar")),
                                 f(None if fr is None else TARGETS[0] - fr), f(None if fr is None else TARGETS[1] - fr),
                                 ""]).strip())
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summarize":
        summarize(sys.argv[2], sys.argv[3])
    else:
        main()
