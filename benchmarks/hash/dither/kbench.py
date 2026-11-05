"""Dither's kernel-level standard number on one GPU: c_fx1 (cgemm.cu, compute accounting's honest H1 + fx1 linear) against
vLLM's own int8 GEMM and cuBLASLt int8, interleaved rep by rep in one process on one stream, at
Llama-3.1-8B's served linears (qkv 6144 x 4096, o 4096 x 4096, gate_up 28672 x 4096, down 4096 x 14336) and decode M = 1, 32,
256, prefill M = 2048, 8192.

    gate   SHAPES at M rounded up to 128 (c_fx1's tile; M = 1 and 32 run a 128-row tile): cgemm.cu's own device gate (`cgate`:
           every buffer poisoned with 0xA5, O against cuBLASLt's X W^T, the other schedule's receipts, c_fx1's receipts unequal
           to duo's, a no-write control rejected; three tiles dumped for cgemm_ref.py, which kbench.sh runs on the CPU after
           the lease). One JSON line a shape.
    bench  only after every gate and the CPU reference passed (kbench.sh): per shape, each arm captured as one CUDA graph of
           CALLS launches (about 20 ms of device time), the graphs replayed interleaved rep by rep (arm order rotated each rep)
           after a warm-up, each replay between CUDA events, the SM clock (NVML, the leased GPU) read after each. Weights
           rotate over enough copies that a replay's working set is at least 2 x L2 (the served model streams 8 GB of weights
           between two calls of one linear, so no call finds its weight in L2).

Arms (M is the served rows; Mp = M rounded up to 128):
    c_fx1        the H1 + fx1 GEMM alone at Mp (co32c: T_t, O', D_d each absorbed with fx1 + duo; O written)
    c_fx1_lin    cquant + cwedit + c_fx1 under one key at Mp: the honest linear (dithered quantizer, W' = W + c + U, GEMM)
    cquant       the dithered quantizer alone; cwedit, the weight editor alone
    lt           egemm.cu's cuBLASLt s8 -> s32 at Mp (autotuned over its top 16 heuristic algorithms)
    triton       vLLM's triton_scaled_mm(x_q, W^T, s_a, s_w, bf16) at M: what compressed-tensors W8A8 int8 serves on sm_120,
                 where vLLM's first choice (CutlassInt8ScaledMMLinearKernel, ops.cutlass_scaled_mm) refuses int8 ("Int8 not
                 supported on SM120"; tried once a shape and recorded as `cutlass_refused`) and its selector falls through to
                 TritonInt8ScaledMMLinearKernel
    vllm_lin     vLLM's served int8 linear at M: ops.scaled_int8_quant(x) then triton_scaled_mm
    lt_torch     torch._int_mm at max(M, 17) (the int8-exact profile's GEMM: cuBLASLt, M padded to 17)

    python kbench.py gate --lib libkcapi.so --dir D --out OUT [--shapes ...]
    python kbench.py bench --lib libkcapi.so --out OUT [--shapes ...] [--reps 9]
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import statistics
import sys
import time
from pathlib import Path

LINEARS = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}
MS = (1, 32, 256, 2048, 8192)
L2_BYTES = 128 << 20
ITEM_MS = 20.0
C_ARMS = ("c_fx1", "c_fx1_lin", "cquant", "cwedit", "lt")
CGEMM_ARM = {"c_fx1": "c_fx1", "c_fx1_lin": "cedited:c_fx1", "cquant": "cquant", "cwedit": "cwedit", "lt": "plain"}
T_ARMS = ("triton", "vllm_lin", "lt_torch")


def shapes(spec: str | None) -> list[tuple[str, int, int, int]]:
    """[(name, M, N, K)]: SPEC is comma-separated linear:M, or every linear at every M."""
    if not spec:
        return [(f"{lin}_m{m}", m, n, k) for m in MS for lin, (n, k) in LINEARS.items()]
    out = []
    for s in spec.split(","):
        lin, m = s.split(":")
        n, k = LINEARS[lin]
        out.append((f"{lin}_m{m}", int(m), n, k))
    return out


def mpad(m: int) -> int:
    return max(128, -(-m // 128) * 128)


class Lib:
    def __init__(self, path: str):
        lib = self.lib = C.CDLL(path)
        lib.kc_gate.argtypes = [C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_uint]
        lib.kc_setup.restype, lib.kc_setup.argtypes = C.c_void_p, [C.c_int, C.c_int, C.c_int, C.c_uint]
        lib.kc_stream.restype, lib.kc_stream.argtypes = C.c_void_p, [C.c_void_p]
        lib.kc_set_stream.argtypes = [C.c_void_p, C.c_void_p]
        lib.kc_grid.argtypes = lib.kc_lt_algo.argtypes = [C.c_void_p]
        lib.kc_arm.restype, lib.kc_arm.argtypes = C.c_void_p, [C.c_char_p]
        lib.kc_run.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        lib.kc_free.argtypes = [C.c_void_p]
        lib.kc_sm_mhz.restype = C.c_uint


def gate(a) -> int:
    lib = Lib(a.lib).lib
    out = Path(a.out)
    rc = 0
    for name, m, n, k in shapes(a.shapes):
        mp = mpad(m)
        d = Path(a.dir) / f"{name}_mp{mp}"
        if d.exists():
            continue
        d.mkdir(parents=True)
        t = time.perf_counter()
        r = lib.kc_gate(str(d).encode(), mp, n, k, 0x9ABC + m)
        line = {"shape": name, "m": m, "mp": mp, "n": n, "k": k, "rc": r, "s": round(time.perf_counter() - t, 2), "dir": str(d)}
        with open(out / "gate.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        rc |= r
    return 1 if rc else 0


def bench(a) -> int:
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm

    lib = Lib(a.lib).lib
    dev = torch.device("cuda")
    torch.manual_seed(0)
    stream = torch.cuda.Stream()   # torch's (non-blocking): a capture on a blocking stream refuses the legacy stream's use
    out = Path(a.out)
    for name, m, n, k in shapes(a.shapes):
        mp = mpad(m)
        wbytes = n * k
        copies = 1 if m >= 2048 else max(1, math.ceil(2 * L2_BYTES / wbytes))
        cps = [lib.kc_setup(mp, n, k, 0x5EED0000 + 17 * i + m) for i in range(copies)]
        for c in cps:
            lib.kc_set_stream(c, C.c_void_p(stream.cuda_stream))
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            ws = [torch.randint(-127, 128, (n, k), dtype=torch.int8, device=dev) for _ in range(copies)]
            sws = [torch.rand(n, 1, dtype=torch.float32, device=dev) * 1e-2 + 1e-3 for _ in range(copies)]
            x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
            xq, sx, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
            xq17 = torch.cat([xq, xq.new_zeros(max(0, 17 - m), k)]) if m < 17 else xq
        torch.cuda.synchronize()
        # the baselines' own sanity: torch's s32 product against int64 on the CPU on 4 rows, triton against it scaled
        r4 = min(4, m)
        ref = (xq[:r4].cpu().to(torch.int64) @ ws[0][:256].cpu().to(torch.int64).t())
        got = torch._int_mm(xq17, ws[0].t())[:r4, :256].cpu().to(torch.int64)
        tri = triton_scaled_mm(xq, ws[0].t(), sx, sws[0], torch.bfloat16)[:r4, :256].float().cpu()
        want = (ref.double() * sx[:r4].cpu().double() * sws[0][:256].cpu().double().reshape(1, -1)).float()
        sane = {"lt_torch_exact": bool(torch.equal(ref, got)),
                "triton_rel_err": float(((tri - want).abs().max() / want.abs().max().clamp(min=1e-30)).item())}
        try:
            ops.cutlass_scaled_mm(xq, ws[0].t(), sx, sws[0], torch.bfloat16)
            torch.cuda.synchronize()
            sane["cutlass_refused"] = None
        except RuntimeError as e:
            sane["cutlass_refused"] = str(e).splitlines()[0][-160:]
        if not sane["lt_torch_exact"] or sane["triton_rel_err"] > 2e-2:
            print(json.dumps({"shape": name, "baseline_sanity_failed": sane}), flush=True)
            return 3

        def c_call(arm_ptr):
            def f(i):
                lib.kc_run(cps[i % copies], arm_ptr, 1)
            return f

        def t_call(arm):
            def f(i):
                w, sw = ws[i % copies], sws[i % copies]
                if arm == "triton":
                    triton_scaled_mm(xq, w.t(), sx, sw, torch.bfloat16)
                elif arm == "vllm_lin":
                    q, s, _ = ops.scaled_int8_quant(x, scale=None, azp=None, symmetric=True)
                    triton_scaled_mm(q, w.t(), s, sw, torch.bfloat16)
                else:
                    torch._int_mm(xq17, w.t())
            return f

        calls_of = {**{arm: c_call(lib.kc_arm(CGEMM_ARM[arm].encode())) for arm in C_ARMS}, **{arm: t_call(arm) for arm in T_ARMS}}
        arms = list(calls_of)
        # one eager call each, then CALLS for about ITEM_MS of device time, rounded up to a multiple of the copies
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        per = {}
        with torch.cuda.stream(stream):
            for arm in arms:
                for i in range(2 * copies):
                    calls_of[arm](i)
                e0.record(stream)
                for i in range(2 * copies):
                    calls_of[arm](i)
                e1.record(stream)
                e1.synchronize()
                per[arm] = e0.elapsed_time(e1) / (2 * copies)
        graphs, ncalls = {}, {}
        for arm in arms:
            nc = max(4, min(2000, int(ITEM_MS / max(per[arm], 1e-4))))
            nc = -(-nc // copies) * copies
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                for i in range(nc):
                    calls_of[arm](i)
            graphs[arm], ncalls[arm] = g, nc
        torch.cuda.synchronize()
        err = lib.kc_last_error()
        if err:
            print(json.dumps({"shape": name, "cuda_error": err}), flush=True)
            return 4
        ms = {arm: [] for arm in arms}
        mhz = {arm: [] for arm in arms}
        with torch.cuda.stream(stream):
            for arm in arms:
                graphs[arm].replay()
            for r in range(a.reps):
                for j in range(len(arms)):
                    arm = arms[(j + r) % len(arms)]
                    e0.record(stream)
                    graphs[arm].replay()
                    e1.record(stream)
                    e1.synchronize()
                    ms[arm].append(e0.elapsed_time(e1) / ncalls[arm])
                    mhz[arm].append(lib.kc_sm_mhz())
        line = {"shape": name, "m": m, "mp": mp, "n": n, "k": k, "copies": copies, "grid": lib.kc_grid(cps[0]),
                "lt_algo": lib.kc_lt_algo(cps[0]), "sanity": sane, "arms": {
                    arm: {"us": round(1e3 * statistics.median(ms[arm]), 3), "calls": ncalls[arm],
                          "us_all": [round(1e3 * v, 3) for v in ms[arm]], "mhz": statistics.median(mhz[arm]),
                          "mhz_all": mhz[arm]} for arm in arms}}
        base = min(line["arms"]["triton"]["us"], line["arms"]["lt"]["us"], line["arms"]["lt_torch"]["us"])
        line["ratios"] = {"c_fx1_over_triton": round(line["arms"]["c_fx1"]["us"] / line["arms"]["triton"]["us"], 4),
                          "c_fx1_over_lt": round(line["arms"]["c_fx1"]["us"] / min(line["arms"]["lt"]["us"],
                                                                                 line["arms"]["lt_torch"]["us"]), 4),
                          "c_fx1_over_fastest": round(line["arms"]["c_fx1"]["us"] / base, 4),
                          "c_fx1_lin_over_vllm_lin": round(line["arms"]["c_fx1_lin"]["us"] / line["arms"]["vllm_lin"]["us"], 4)}
        with open(out / "bench.jsonl", "a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps({k: v for k, v in line.items() if k != "arms"} | {
            "us": {arm: line["arms"][arm]["us"] for arm in arms}}), flush=True)
        del graphs
        torch.cuda.synchronize()
        for c in cps:
            lib.kc_free(c)
        ws = sws = x = xq = xq17 = None
        torch.cuda.empty_cache()
    return 0


def summary(a) -> int:
    """result.json (research/result/v0.1, published as bench-result/v1): per shape the arms' times and the ratios, and the
    standard number's kernel level: the geometric mean over the 4 linears of c_fx1 over the fastest plain GEMM at each M."""
    out = Path(a.out)
    rows = [json.loads(x) for x in (out / "bench.jsonl").read_text().splitlines() if x.strip()]
    gates = [json.loads(x) for x in (out / "gate.jsonl").read_text().splitlines() if x.strip()] if (out / "gate.jsonl").is_file() else []
    meas = []
    for r in rows:
        for arm, v in r["arms"].items():
            meas.append({"name": f"{r['shape']}.us", "arm": arm, "value": v["us"], "unit": "us", "mhz": v["mhz"]})
        for k, v in r["ratios"].items():
            meas.append({"name": f"{r['shape']}.{k}", "arm": "c_fx1", "value": v, "unit": "x"})
    by_m: dict[int, list] = {}
    for r in rows:
        by_m.setdefault(r["m"], []).append(r)
    table = ["| M | linear | c_fx1 us | c_fx1_lin us | triton us | vllm_lin us | lt us | lt_torch us | c_fx1/fastest | lin/vllm_lin |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(by_m):
        gm = math.exp(statistics.mean(math.log(r["ratios"]["c_fx1_over_fastest"]) for r in by_m[m]))
        gl = math.exp(statistics.mean(math.log(r["ratios"]["c_fx1_lin_over_vllm_lin"]) for r in by_m[m]))
        meas.append({"name": f"m{m}.c_fx1_over_fastest_geomean", "arm": "c_fx1", "value": round(gm, 4), "unit": "x"})
        meas.append({"name": f"m{m}.c_fx1_lin_over_vllm_lin_geomean", "arm": "c_fx1", "value": round(gl, 4), "unit": "x"})
        for r in by_m[m]:
            u = {k: v["us"] for k, v in r["arms"].items()}
            table.append(f"| {m} | {r['shape'].split('_m')[0]} | {u['c_fx1']} | {u['c_fx1_lin']} | {u['triton']} | {u['vllm_lin']} | "
                         f"{u['lt']} | {u['lt_torch']} | {r['ratios']['c_fx1_over_fastest']} | {r['ratios']['c_fx1_lin_over_vllm_lin']} |")
        table.append(f"| {m} | geomean | | | | | | | {gm:.3f} | {gl:.3f} |")
    gates_ok = bool(gates) and all(g["rc"] == 0 for g in gates)
    doc = {"schema": "research/result/v0.1", "run_id": a.run_id,
           "workload_fingerprint": {"bench": "benchmarks/hash/dither/kbench.py", "level": "kernel", "shapes": [r["shape"] for r in rows]},
           "validation": {"status": "passed" if gates_ok and a.ref_rc == 0 else "failed",
                          "checker": "cgemm.cu cgate (0xA5 poison, no-write control) + cgemm_ref.py on the CPU, before any timing",
                          "detail": f"{len(gates)} gates, rc {[g['rc'] for g in gates]}, cgemm_ref rc {a.ref_rc}", "evidence": []},
           "measurements": meas, "measurement_files": ["out/bench.jsonl", "out/gate.jsonl", "out/ref.jsonl"]}
    Path(a.result).write_text(json.dumps(doc, indent=1) + "\n")
    (out / "table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))
    return 0 if doc["validation"]["status"] == "passed" else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gate", "bench", "summary"))
    ap.add_argument("--lib")
    ap.add_argument("--dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--result")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--ref-rc", type=int, default=0)
    a = ap.parse_args()
    return {"gate": gate, "bench": bench, "summary": summary}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
