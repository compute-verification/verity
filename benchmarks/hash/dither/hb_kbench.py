"""The kernel-level standard number on one H200 or B200: every honest Dither GEMM that exists for the card, gated, then timed
interleaved with vLLM's own int8 GEMM (cutlass_scaled_mm), torch._int_mm (cuBLASLt's heuristic, as the exact-int8 linears
call it) and cuBLASLt with its top algorithms, at Llama-3.1-8B's linears (qkv 6144 x 4096, o 4096 x 4096, gate_up 28672 x
4096, down 4096 x 14336) and M = 1, 32, 256, 2048, 8192.

    python hb_kbench.py --lib kb.so [--pg-lib kb_pg.so] --out OUT [--reps 7] [--ms 1,32,256,2048,8192] [--linears qkv,o,...]

Arms (one CUDA graph of G calls each, G chosen so an item is about --item-ms of device time; the weights cycle through enough
copies that the graph's weights exceed 2.5x the L2, as every layer's do in a served step):
  dg         dgemm.cu's honest GEMM (dg90 64 x 128 wgmma tiles on sm_90a, dg100 128 x 256 tcgen05 on sm_100a), FOLD: C and
             every receipt word; M padded with zero rows to its tile (the padded rows are computed and folded)
  dg-nofold  the same without receipts (diagnostic: the mainloop's share)
  pg         pgemm.cu's pipelined honest GEMM (sm_100a), NB, LW, MI from --pg-cfg, L = 32; operands resident, so its inputs
             are K-periodic (period 128) and it streams no operand: a lower bound for a GEMM of served operands
  int_mm     torch._int_mm(X, W^T), with int8_exact's 17-row padding below 17 rows (verity_vllm.engine.int8_exact.int_mm)
  cutlass    torch.ops._C.cutlass_scaled_mm (vLLM's CutlassInt8ScaledMMLinearKernel), bf16 out, per-token and per-channel
             scales; on a card where it refuses, the refusal is recorded
  lt         cuBLASLt s8 x s8 -> s32 with each of its heuristic's top algorithms, the fastest kept (lt_algo)

Gates, on the device before any timing of a (linear, M), for every honest arm (a failed gate times nothing of that arm):
C and R filled with 0xA5 and the arm run once: C equal to torch._int_mm's product of the same operands, and R on --gate-rows
sampled rows (first, last, spread; the first and last tile rows included) equal to kb_ref_rows (the receipt by its
definition, computed from X and W); then the no-write control (WRITE = false, same poison) must be rejected. After the
lease, `verify.py` checks two of those rows against the numpy twin (dc/dgemm_ref.py) from the regenerated inputs.

Writes OUT/kbench.json: per (linear, M) the gates, per-arm per-rep µs a call with the SM clock of each item, medians, and
ratios of each honest arm over cutlass, lt and int_mm.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from verity_config.settings import of

settings = of(__file__)

LINEARS = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
MS = (1, 32, 256, 2048, 8192)
MAX_M = 8192
PER = 128
POISON = -1515870811  # 0xA5A5A5A5 as s32
SEED = 0x5EED


def inputs(seed: int, rows: int, k: int) -> np.ndarray:
    """Deterministic s8 codes in [-127, 127] (verify.py regenerates them)."""
    return np.random.default_rng(seed).integers(-127, 128, size=(rows, k), dtype=np.int8)


def x_seed(k: int) -> int:
    return 1000 + k


def w_seed(name: str) -> int:
    return 2000 + sorted(LINEARS).index(name)


def periodic(a: torch.Tensor) -> torch.Tensor:
    """a's first PER columns repeated along K (pgemm's resident operands)."""
    k = a.shape[1]
    return a[:, :PER].repeat(1, k // PER).contiguous()


def sample_rows(m: int, bm: int, n: int) -> list[int]:
    rows = {0, m - 1, min(bm - 1, m - 1), max(m - bm, 0)}
    rows |= {int(r) for r in np.linspace(0, m - 1, n)}
    return sorted(rows)


class Lib:
    def __init__(self, path: str):
        self.h = ctypes.CDLL(path)
        h = self.h
        h.kb_init.restype = ctypes.c_int
        h.kb_dg_launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        h.kb_ref_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_void_p]
        h.kb_lt_create.restype = ctypes.c_void_p
        h.kb_lt_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        h.kb_lt_algos.argtypes = [ctypes.c_void_p]
        h.kb_lt_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_void_p]
        h.kb_lt_destroy.argtypes = [ctypes.c_void_p]
        self.sm = h.kb_init()
        bm, bn = ctypes.c_int(), ctypes.c_int()
        h.kb_dg_tile(ctypes.byref(bm), ctypes.byref(bn))
        self.bm, self.bn = bm.value, bn.value


class PgLib:
    def __init__(self, path: str):
        self.h = ctypes.CDLL(path)
        self.h.kb_pg_launch.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 2 + [ctypes.c_int] * 5 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        rc = self.h.kb_pg_prepare()
        if rc:
            raise RuntimeError(f"kb_pg_prepare {rc}")


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


class Clock:
    def __init__(self) -> None:
        self.nv = None
        try:
            import pynvml as nv
            nv.nvmlInit()
            uuid = settings.GPU_LEASE_UUID
            self.h = nv.nvmlDeviceGetHandleByUUID(uuid) if uuid else nv.nvmlDeviceGetHandleByIndex(0)
            self.nv = nv
            self.name = nv.nvmlDeviceGetName(self.h)
        except Exception as e:  # noqa: BLE001 - recorded, never fatal
            print(f"kbench: no NVML clocks ({e})", file=sys.stderr)
            self.name = torch.cuda.get_device_name()

    def sm(self) -> int | None:
        if not self.nv:
            return None
        return int(self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM))


def gate_arm(run_write, run_control, C: torch.Tensor, R: torch.Tensor, ref_C: torch.Tensor, ref_rows_R: torch.Tensor,
             rows: torch.Tensor, m: int) -> dict:
    """0xA5 poison, the arm once, C against the reference product on its first m rows, R on the sampled rows against the
    row reference; then the no-write control under the same poison, which must be rejected. "R_rows" is the honest run's
    receipt rows (for verify.py), taken before the control overwrites R with poison."""
    def once(fn):
        C.fill_(POISON)
        R.view(torch.int32).fill_(POISON)
        fn()
        torch.cuda.synchronize()
        got = R.view(torch.int32)[rows.long()]
        c_bad = int((C[:m] != ref_C[:m]).sum())
        r_bad = int((got != ref_rows_R.view(torch.int32)).sum())
        return c_bad, r_bad, got.cpu().numpy().view(np.uint32)
    c_bad, r_bad, r_rows = once(run_write)
    k_c, k_r, _ = once(run_control)
    return {"exact": c_bad == 0 and r_bad == 0, "c_bad": c_bad, "r_bad": r_bad, "control_c_bad": k_c, "control_r_bad": k_r,
            "control_rejected": k_c > 0 and k_r > 0, "rows": rows.tolist(), "R_rows": r_rows}


def capture(fn, calls: int) -> torch.cuda.CUDAGraph:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    torch.cuda.synchronize()
    return g


def one_call_ms(fn) -> float:
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(0)
    torch.cuda.synchronize()
    e0.record()
    for i in range(3):
        fn(i)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / 3


def bench_shape(lib: Lib, pg: PgLib | None, args, name: str, n: int, k: int, m: int, Wh: np.ndarray, Xh: np.ndarray,
                clock: Clock, l2: int) -> dict:
    dev = torch.device("cuda")
    out: dict = {"linear": name, "m": m, "n": n, "k": k, "arms": {}, "gates": {}, "refusals": {}}
    bm, bn = lib.bm, lib.bn
    mp = math.ceil(m / bm) * bm
    wbytes = n * k
    nbuf = max(1, min(32, math.ceil(2.5 * l2 / wbytes)))
    out["weight_copies"] = nbuf
    W = torch.from_numpy(Wh).to(dev)
    Ws = [W] + [W.clone() for _ in range(nbuf - 1)]
    X = torch.from_numpy(Xh[:m]).to(dev)
    Xp = torch.zeros((mp, k), dtype=torch.int8, device=dev)
    Xp[:m] = X
    seed = 7
    tiles_n = n // bn
    Cdg = torch.empty((mp, n), dtype=torch.int32, device=dev)
    Rdg = torch.empty((mp, tiles_n, 16), dtype=torch.int32, device=dev)
    rows = torch.tensor(sample_rows(mp, bm, args.gate_rows), dtype=torch.int32, device=dev)
    out["gate_rows"] = rows.tolist()
    ref_C = torch._int_mm(Xp if mp > 16 else torch.cat([Xp, Xp.new_zeros(17 - mp, k)]), W.t())[:mp]
    refR = torch.empty((len(rows), tiles_n, 16), dtype=torch.int32, device=dev)
    refC = torch.empty((len(rows), n), dtype=torch.int32, device=dev)
    rc = lib.h.kb_ref_rows(ptr(Xp), ptr(W), n, k, bn, 32, seed, ptr(rows), len(rows), ptr(refC), ptr(refR), stream())
    torch.cuda.synchronize()
    assert rc == 0, f"kb_ref_rows {rc}"
    ref_ok = bool(torch.equal(refC, ref_C[rows.long()]))
    out["ref_rows_C_equal_int_mm"] = ref_ok

    def dg(fold: int, write: int, C=Cdg, R=Rdg):
        def f(i: int):
            r = lib.h.kb_dg_launch(fold, write, ptr(Xp), ptr(Ws[i % nbuf]), mp, n, k, seed, ptr(C), ptr(R), stream())
            if r:
                raise RuntimeError(f"kb_dg_launch {r}")
        return f

    arms: dict = {}
    g = gate_arm(lambda: dg(1, 1)(0), lambda: dg(1, 0)(0), Cdg, Rdg, ref_C, refR, rows, mp)
    g["ref_rows_C_equal_int_mm"] = ref_ok
    np.save(Path(args.out) / f"R-dg-{name}-m{m}.npy", g.pop("R_rows"))
    out["gates"]["dg"] = g
    if g["exact"] and g["control_rejected"] and ref_ok:
        arms["dg"] = dg(1, 1)
        Cnf = torch.empty_like(Cdg)
        arms["dg-nofold"] = dg(0, 1, Cnf, Rdg)

    if pg is not None:
        nb, lw, mi = args.pg_cfg
        mpp = math.ceil(m / 128) * 128
        if n % 512 == 0:
            Xq = torch.zeros((mpp, k), dtype=torch.int8, device=dev)
            Xq[:m] = periodic(X)
            Wq = [periodic(w) for w in Ws[:1]]
            Wq += [Wq[0].clone() for _ in range(nbuf - 1)]
            bnp = 512 // nb
            Cpg = torch.empty((mpp, n), dtype=torch.int32, device=dev)
            Rpg = torch.empty((mpp, n // bnp, 16), dtype=torch.int32, device=dev)
            clk = torch.empty((3 * (n // 512) * (mpp // 128),), dtype=torch.int64, device=dev)
            rows_p = torch.tensor(sample_rows(mpp, 128, args.gate_rows), dtype=torch.int32, device=dev)
            refCp = torch._int_mm(Xq if mpp > 16 else torch.cat([Xq, Xq.new_zeros(17 - mpp, k)]), Wq[0].t())[:mpp]
            refRp = torch.empty((len(rows_p), n // bnp, 16), dtype=torch.int32, device=dev)
            refCr = torch.empty((len(rows_p), n), dtype=torch.int32, device=dev)
            assert lib.h.kb_ref_rows(ptr(Xq), ptr(Wq[0]), n, k, bnp, 32, seed, ptr(rows_p), len(rows_p), ptr(refCr),
                                     ptr(refRp), stream()) == 0
            torch.cuda.synchronize()
            ref_ok_p = bool(torch.equal(refCr, refCp[rows_p.long()]))

            def pgf(mode: int):
                def f(i: int):
                    r = pg.h.kb_pg_launch(nb, lw, mode, mi, ptr(Xq), ptr(Wq[i % nbuf]), k, mpp, n, k, 1, seed, ptr(Cpg),
                                          ptr(Rpg), ptr(clk), stream())
                    if r:
                        raise RuntimeError(f"kb_pg_launch {r}")
                return f
            # pgemm's MODE 1 (load, no fold) stores C but no R: the no-write control is MODE 0 (hand back only), which
            # writes neither C nor the receipts
            gp = gate_arm(lambda: pgf(2)(0), lambda: pgf(0)(0), Cpg, Rpg, refCp, refRp, rows_p, mpp)
            gp["ref_rows_C_equal_int_mm"] = ref_ok_p
            gp["cfg"] = {"nb": nb, "lw": lw, "mi": mi, "L": 32}
            np.save(Path(args.out) / f"R-pg-{name}-m{m}.npy", gp.pop("R_rows"))
            out["gates"]["pg"] = gp
            if gp["exact"] and gp["control_rejected"] and ref_ok_p:
                arms["pg"] = pgf(2)
        else:
            out["refusals"]["pg"] = f"n {n} not a multiple of 512"

    Cmm = torch.empty((max(m, 17), n), dtype=torch.int32, device=dev)
    Xmm = X if m >= 17 else torch.cat([X, X.new_zeros(17 - m, k)])

    def int_mm(i: int):
        torch._int_mm(Xmm, Ws[i % nbuf].t(), out=Cmm)
    arms["int_mm"] = int_mm

    try:
        import vllm._custom_ops  # noqa: F401 - loads torch.ops._C
        sa = torch.rand((m, 1), dtype=torch.float32, device=dev) / 100 + 1e-3
        sw = torch.rand((n, 1), dtype=torch.float32, device=dev) / 100 + 1e-3
        Ocut = torch.empty((m, n), dtype=torch.bfloat16, device=dev)

        def cutlass(i: int):
            torch.ops._C.cutlass_scaled_mm(Ocut, X, Ws[i % nbuf].t(), sa, sw, None)
        cutlass(0)
        torch.cuda.synchronize()
        # its product is X W^T scaled: check it against the exact product, as vLLM's own path would serve it
        want = (ref_C[:m].float() * sa * sw.t()).to(torch.bfloat16)
        out["cutlass_matches_scaled_exact"] = bool(torch.equal(Ocut, want))
        out["cutlass_max_ulp_diff"] = int((Ocut.view(torch.int16).int() - want.view(torch.int16).int()).abs().max())
        arms["cutlass"] = cutlass
    except Exception as e:  # noqa: BLE001 - a refusal is a result
        out["refusals"]["cutlass"] = "".join(traceback.format_exception_only(type(e), e)).strip()[-2000:]

    ws = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    Clt = torch.empty((m, n), dtype=torch.int32, device=dev)
    lt = lib.h.kb_lt_create(m, n, k, ptr(ws), ws.numel())
    nalg = lib.h.kb_lt_algos(lt)
    out["lt_algos"] = nalg
    if nalg:
        best, best_ms, lt_ms = None, None, {}
        for a in range(nalg):
            def f(i: int, a=a):
                if lib.h.kb_lt_run(lt, a, ptr(X), ptr(Ws[i % nbuf]), ptr(Clt), stream()):
                    raise RuntimeError("kb_lt_run")
            try:
                f(0)
                torch.cuda.synchronize()
                if not torch.equal(Clt, ref_C[:m]):
                    lt_ms[a] = "inexact"
                    continue
                t = one_call_ms(f)
                lt_ms[a] = round(t * 1e3, 2)
                if best_ms is None or t < best_ms:
                    best, best_ms = a, t
            except Exception as e:  # noqa: BLE001
                lt_ms[a] = f"failed: {e}"
        out["lt_algo_us"] = lt_ms
        if best is not None:
            out["lt_algo"] = best
            arms["lt"] = lambda i, a=best: lib.h.kb_lt_run(lt, a, ptr(X), ptr(Ws[i % nbuf]), ptr(Clt), stream())
    else:
        out["refusals"]["lt"] = "cuBLASLt's heuristic returned no algorithm"

    graphs, calls = {}, {}
    for a, fn in arms.items():
        try:
            t = one_call_ms(fn)
            G = max(1, min(400, round(args.item_ms / max(t, 1e-3))))
            graphs[a], calls[a] = capture(fn, G), G
        except Exception as e:  # noqa: BLE001
            out["refusals"][a] = f"capture: {e}"
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        for gr in graphs.values():
            gr.replay()
    torch.cuda.synchronize()
    per = {a: [] for a in graphs}
    clk_rep = {a: [] for a in graphs}
    order = list(graphs)
    for rep in range(args.reps):
        rot = order[rep % len(order):] + order[:rep % len(order)]
        for a in rot:
            e0.record()
            graphs[a].replay()
            e1.record()
            e1.synchronize()
            per[a].append(e0.elapsed_time(e1) * 1e3 / calls[a])
            clk_rep[a].append(clock.sm())
    for a in graphs:
        out["arms"][a] = {"calls_per_item": calls[a], "us": [round(v, 3) for v in per[a]],
                          "median_us": round(statistics.median(per[a]), 3), "sm_mhz": clk_rep[a]}
    med = {a: v["median_us"] for a, v in out["arms"].items()}
    base = {b: med[b] for b in ("cutlass", "lt", "int_mm") if b in med}
    if base:
        out["fastest_plain"] = min(base, key=base.get)
    out["ratios"] = {f"{h}/{b}": round(med[h] / v, 4) for h in ("dg", "dg-nofold", "pg") if h in med for b, v in base.items()}
    if "fastest_plain" in out:
        fp = out["fastest_plain"]
        out["ratios"].update({f"{h}/fastest": round(med[h] / base[fp], 4) for h in ("dg", "dg-nofold", "pg") if h in med})
    lib.h.kb_lt_destroy(lt)
    del graphs
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--pg-lib")
    ap.add_argument("--pg-cfg", default="2,2,1", help="pgemm's NB,LW,MI")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=3.0)
    ap.add_argument("--gate-rows", type=int, default=16)
    ap.add_argument("--ms", default=",".join(map(str, MS)))
    ap.add_argument("--linears", default=",".join(LINEARS))
    args = ap.parse_args()
    args.pg_cfg = tuple(int(v) for v in args.pg_cfg.split(","))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    lib = Lib(args.lib)
    pg = PgLib(args.pg_lib) if args.pg_lib else None
    clock = Clock()
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "L2_cache_size", 50 << 20))
    doc = {"schema": "dither-hb/kbench/v1", "device": clock.name, "sm": lib.sm, "sms": props.multi_processor_count,
           "l2_bytes": l2, "tile": [lib.bm, lib.bn], "torch": torch.__version__, "cuda": torch.version.cuda,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seed_rule": "X: rng(1000 + k), W: rng(2000 + index)",
           "shapes": []}
    try:
        import vllm
        doc["vllm"] = vllm.__version__
    except Exception as e:  # noqa: BLE001
        doc["vllm"] = f"unavailable: {e}"
    ms = [int(v) for v in args.ms.split(",")]
    xs = {}
    for name in args.linears.split(","):
        n, k = LINEARS[name]
        if k not in xs:
            xs[k] = inputs(x_seed(k), MAX_M, k)
        Wh = inputs(w_seed(name), n, k)
        for m in ms:
            t0 = time.time()
            try:
                r = bench_shape(lib, pg, args, name, n, k, m, Wh, xs[k], clock, l2)
            except Exception as e:  # noqa: BLE001
                r = {"linear": name, "m": m, "n": n, "k": k, "error": "".join(traceback.format_exception(e))[-3000:]}
            r["wall_s"] = round(time.time() - t0, 2)
            doc["shapes"].append(r)
            print(json.dumps({k2: r.get(k2) for k2 in ("linear", "m", "ratios", "refusals", "error", "wall_s")}), flush=True)
            (Path(args.out) / "kbench.json").write_text(json.dumps(doc, indent=1))
    doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (Path(args.out) / "kbench.json").write_text(json.dumps(doc, indent=1))
    bad = [s for s in doc["shapes"] if "error" in s or not all(g["exact"] and g["control_rejected"]
                                                                for g in s.get("gates", {}).values())]
    return 4 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
