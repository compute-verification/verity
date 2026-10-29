"""GEMM proving slowdown: the prover's seconds for a served model's GEMMs over their native time on the same GPU.

The prover's cost is per output coordinate: a class sweep of `GemmCoordinate_v1{K}` (70-class-sweep.sh) gives each K's seconds
per coordinate at its largest statement, the witness build plus both reps (`witness_s + prove_total_s`, over the statement's
instances). A GEMM of M x N x K costs M * N of them. Its native time is torch's bf16 `linear` (cuBLAS, as vLLM serves it), the
weights rotated through copies that overflow L2 so every call reads them from memory.

The model is Llama-3.2-1B, the census row's: 16 layers of qkv (3072 x 2048), o (2048 x 2048), gate_up (16384 x 2048) and down (2048 x 8192),
fused as vLLM runs them, and the LM head (128256 x 2048). Prefill is the row's 256 prompt tokens through every layer and the LM head
at M = 1 (vLLM computes only the last token's logits); decode is every GEMM at M = 1, 2, 4 and 8, summed over the four.

    python gemm_slowdown.py native --out native.json [--iters 100] [--trials 5]
    python gemm_slowdown.py compute --sweep m34/results.jsonl [--preflight preflight/results.jsonl] --native native.json --result result.json
    python gemm_slowdown.py sample --pgid PGID --out others.txt [--every 30]   # cores busy outside process group PGID (overlapping Builds)
    python gemm_slowdown.py point --slowdown slowdown.json --config C --result result.json   # one (phase, config) row of an attempt

The rows (ov.config, one stored run each, from the attempt's slowdown.json): per phase, the model's GEMMs together
(`llama32-1b.prefill.M256`, `llama32-1b.decode.M1-8`) and the two shapes that set the two per-coordinate costs, gate_up for
K = 2048 and down for K = 8192 (`256x16384x2048`, `256x2048x8192`, `1x16384x2048`, `1x2048x8192`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from pathlib import Path

LAYERS = 16
LAYER = [("qkv", 3072, 2048), ("o", 2048, 2048), ("gate_up", 16384, 2048), ("down", 2048, 8192)]
LM_HEAD = ("lm_head", 128256, 2048)
PREFILL_M = 256
DECODE_M = (1, 2, 4, 8)
#: the preflight check's GPU selftest cases (each a byte equality), which every GEMM coordinate it proves must pass
PREFLIGHT_CASES = ("gpu_paths_agree", "gpu_proofs_match_cpu", "verify_ahead_matches_serial")


CONFIGS = {"llama32-1b.prefill.M256": ("prefill", None), "256x16384x2048": ("prefill", (256, 16384, 2048)),
           "256x2048x8192": ("prefill", (256, 2048, 8192)), "llama32-1b.decode.M1-8": ("decode", None),
           "1x16384x2048": ("decode", (1, 16384, 2048)), "1x2048x8192": ("decode", (1, 2048, 8192))}


def phases() -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Per phase, its GEMMs: (name, M, N, K, count)."""
    pre = [(n, PREFILL_M, nn, k, LAYERS) for n, nn, k in LAYER] + [(LM_HEAD[0], 1, LM_HEAD[1], LM_HEAD[2], 1)]
    dec = [(n, m, nn, k, LAYERS) for m in DECODE_M for n, nn, k in LAYER] + [(LM_HEAD[0], m, LM_HEAD[1], LM_HEAD[2], 1) for m in DECODE_M]
    return {"prefill": pre, "decode": dec}


def native(a) -> None:
    import torch

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(dev)
    l2 = getattr(props, "L2_cache_size", 128 << 20) or (128 << 20)
    shapes = sorted({(m, n, k) for ph in phases().values() for _, m, n, k, _ in ph})
    out = {"gpu": props.name, "sm": f"{props.major}{props.minor}", "l2_bytes": l2, "torch": torch.__version__,
           "cuda": torch.version.cuda, "cublas": torch.backends.cuda.preferred_blas_library().name if hasattr(torch.backends.cuda, "preferred_blas_library") else None,
           "iters": a.iters, "trials": a.trials, "times_s": {}}
    torch.manual_seed(0)
    for m, n, k in shapes:
        copies = max(2, math.ceil(4 * l2 / (n * k * 2)))
        ws = [torch.randn(n, k, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
        for i in range(20):
            torch.nn.functional.linear(x, ws[i % copies])
        torch.cuda.synchronize()
        trials = []
        for _ in range(a.trials):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for i in range(a.iters):
                torch.nn.functional.linear(x, ws[i % copies])
            e.record()
            e.synchronize()
            trials.append(s.elapsed_time(e) / 1e3 / a.iters)
        out["times_s"][f"{m}x{n}x{k}"] = statistics.median(trials)
        del ws, x
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(out, indent=1))


def statement_seconds(lv: dict) -> float:
    """A statement's steady-state prover seconds from its median timed LIVE record: the witness wait plus both reps."""
    # a pipelined prover (the next sessions' witnesses built beside this one) runs at the slower of its device and its host
    # witness builds: witness_s is only the wait, and a loopback verifier's own seconds would hide a build that outruns the GPU
    full = (lv.get("witness_s") or 0.0) + lv["prove_total_s"]
    if lv.get("witness_prebuilt_s") is not None and lv.get("pipeline_depth"):
        full = max(full, lv["witness_prebuilt_s"] / lv["pipeline_depth"])
    return full


def per_coordinate(sweep: Path) -> dict[int, dict]:
    """K -> the sweep's record for GemmCoordinate_v1{K}: seconds per coordinate, full (witness + prove) and prove only."""
    out = {}
    for ln in sweep.read_text().splitlines():
        if not ln.startswith("{"):
            continue
        r = json.loads(ln)
        mk = re.match(r"GemmCoordinate_v1\{K=(\d+)\}", r.get("definition") or "")
        lv = r.get("live") or {}
        if not mk or not lv.get("accepted"):
            continue
        n = ((r.get("stage") or {}).get("n") or lv.get("instances")) * (r.get("stage") or {}).get("coords_per_instance", 1)   # a tile: P * C
        full = statement_seconds(lv)
        out[int(mk.group(1))] = {"n": n, "tile": (r.get("stage") or {}).get("tile"), "m": lv.get("dense_m"), "s_per_coord": full / n, "prove_s_per_coord": lv["prove_total_s"] / n,
                                 "witness_s": lv.get("witness_s"), "witness_prebuilt_s": lv.get("witness_prebuilt_s"),
                                 "pipeline_depth": lv.get("pipeline_depth"), "prove_total_s": lv["prove_total_s"], "shape": r["shape"],
                                 "statement_digest": lv.get("statement_digest"), "proof_bytes": lv.get("proof_bytes"),
                                 "buckets": lv.get("buckets")}
    return out


def preflight(path: Path | None, shapes_needed: int = 2) -> dict:
    """The preflight check's byte identity: every GEMM coordinate's GPU cases pass (equal proofs and transcripts), for at least
    ``shapes_needed`` of them."""
    if path is None or not path.exists():
        return {"status": "not_run"}
    shapes, ok = {}, True
    for ln in path.read_text().splitlines():
        if not ln.startswith("{"):
            continue
        r = json.loads(ln)
        st = r.get("selftest") or {}
        cases = {c.get("case"): c for c in st.get("gpu_cases", [])}
        good = all((cases.get(c) or {}).get("pass") is True for c in PREFLIGHT_CASES)
        # a case whose process ended without its verdict (timed out, killed, out of memory) fails the check but is no verdict
        shapes[r.get("definition")] = {"pass": good, "digest": ((r.get("live") or {}).get("statement_digest")),
                                       "cases": {c: {k: (cases.get(c) or {}).get(k) for k in ("pass", "proofs_equal", "transcripts_equal")} for c in PREFLIGHT_CASES}} | \
            ({"did_not_finish": st["did_not_finish"]} if st.get("did_not_finish") else {})
        ok &= good
    ok &= len(shapes) >= shapes_needed
    return {"status": "passed" if ok else "failed", "shapes": shapes}


def compute(a) -> None:
    pc = per_coordinate(Path(a.sweep))
    nat = json.loads(Path(a.native).read_text())
    g = preflight(Path(a.preflight) if a.preflight else None)
    res = {}
    for ph, gemms in phases().items():
        prover = prove_only = nat_s = 0.0
        missing = sorted({k for _, _, _, k, _ in gemms if k not in pc})
        if missing:
            res[ph] = {"missing_K": missing}
            continue
        for _, m, n, k, count in gemms:
            prover += count * m * n * pc[k]["s_per_coord"]
            prove_only += count * m * n * pc[k]["prove_s_per_coord"]
            nat_s += count * nat["times_s"][f"{m}x{n}x{k}"]
        res[ph] = {"slowdown": prover / nat_s, "slowdown_prove_only": prove_only / nat_s, "prover_s": prover, "native_s": nat_s}
    configs = {}
    for c, (ph, shape) in CONFIGS.items():
        if shape is None:
            if "slowdown" in res.get(ph, {}):
                configs[c] = {"phase": ph, **res[ph]}
        elif shape[2] in pc:
            m, n, k = shape
            t = nat["times_s"][f"{m}x{n}x{k}"]
            configs[c] = {"phase": ph, "slowdown": m * n * pc[k]["s_per_coord"] / t, "slowdown_prove_only": m * n * pc[k]["prove_s_per_coord"] / t,
                          "prover_s": m * n * pc[k]["s_per_coord"], "native_s": t}
    out = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"gpu": nat.get("gpu"), "sm": nat.get("sm"), "torch": nat.get("torch"),
                                    "digests": {k: v["statement_digest"] for k, v in pc.items()}},
           "validation": {"status": g["status"], "checker": "byte identity: " + ", ".join(PREFLIGHT_CASES),
                          "detail": json.dumps(g.get("shapes"))[:900], "evidence": ["out/preflight/classes/results.jsonl"]},
           "measurements": [{"name": f"{ph}_slowdown", "value": v["slowdown"], "unit": "x"} for ph, v in res.items() if "slowdown" in v]
                           + [{"name": f"{ph}_slowdown_prove_only", "value": v["slowdown_prove_only"], "unit": "x"} for ph, v in res.items() if "slowdown" in v]
                           + [{"name": f"s_per_coord_K{k}", "value": v["s_per_coord"], "unit": "s"} for k, v in pc.items()],
           "measurement_files": ["out/slowdown.json", "out/native.json", "out/m34/classes/results.jsonl"]}
    Path(a.result).parent.mkdir(parents=True, exist_ok=True)
    (Path(a.result).parent / "out" / "slowdown.json").write_text(json.dumps({"phases": res, "configs": configs, "per_coordinate": pc, "native": nat,
                                                                          "preflight": g}, indent=1))
    tmp = Path(a.result).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.result)
    print(json.dumps({ph: {k: v for k, v in r.items() if k != "missing_K"} for ph, r in res.items()}), g["status"])


def point(a) -> None:
    """One (phase, config) row of an attempt, as its own result: the overhead, the preflight check, and where it came from."""
    d = json.loads(Path(a.slowdown).read_text())
    pf = d.get("preflight") or d["gate"]                  # older slowdown.json files keep it under "gate"
    c = d["configs"][a.config]
    out = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"config": a.config, "phase": c["phase"], "from_run": a.from_run, "gpu": d["native"].get("gpu")},
           "validation": {"status": pf["status"], "checker": "the measuring run's byte identity", "detail": a.from_run or ""},
           "measurements": [{"name": "overhead", "value": c["slowdown"], "unit": "x-native"},
                            {"name": "overhead_prove_only", "value": c["slowdown_prove_only"], "unit": "x-native"},
                            {"name": "prover_s", "value": c["prover_s"], "unit": "s"}, {"name": "native_s", "value": c["native_s"], "unit": "s"}]}
    tmp = Path(a.result).with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1))
    os.replace(tmp, a.result)
    print(a.config, c["phase"], f"{c['slowdown']:.4g}", pf["status"])


def sample(a) -> None:
    """Every `every` seconds: the machine's busy cores (/proc/stat) less those of process group `pgid`, one line each."""
    tick = os.sysconf("SC_CLK_TCK")

    def busy() -> float:
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return (sum(f) - f[3] - f[4]) / tick

    def group() -> float:
        t = 0
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    f = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                except OSError:
                    continue
                if int(f[2]) == a.pgid:
                    t += int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])
        return t / tick

    b0, g0, t0 = busy(), group(), time.time()
    with open(a.out, "a") as out:
        while True:
            time.sleep(a.every)
            b1, g1, t1 = busy(), group(), time.time()
            dt = t1 - t0
            out.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t1))} others_cores={max(0.0, (b1 - b0) - max(0.0, g1 - g0)) / dt:.1f} "
                      f"own_cores={max(0.0, g1 - g0) / dt:.1f}\n")
            out.flush()
            b0, g0, t0 = b1, g1, t1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("native")
    n.add_argument("--out", required=True)
    n.add_argument("--iters", type=int, default=100)
    n.add_argument("--trials", type=int, default=5)
    c = sub.add_parser("compute")
    c.add_argument("--sweep", required=True)
    c.add_argument("--preflight", "--gate", dest="preflight")    # --gate: its old name
    c.add_argument("--native", required=True)
    c.add_argument("--result", required=True)
    pt = sub.add_parser("point")
    pt.add_argument("--slowdown", required=True)
    pt.add_argument("--config", required=True, choices=sorted(CONFIGS))
    pt.add_argument("--from-run", default="")
    pt.add_argument("--result", default=os.path.join(os.environ.get("RESEARCH_RUN_DIR", "."), "result.json"))
    sm = sub.add_parser("sample")
    sm.add_argument("--pgid", type=int, required=True)
    sm.add_argument("--out", required=True)
    sm.add_argument("--every", type=float, default=30)
    a = ap.parse_args()
    t = time.time()
    {"native": native, "compute": compute, "sample": sample, "point": point}[a.cmd](a)
    print(f"{a.cmd}: {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
