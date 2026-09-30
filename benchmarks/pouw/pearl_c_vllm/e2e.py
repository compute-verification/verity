"""PoUW's MVP end to end on the RTX PRO 6000 (sm_120): Pearl-C (`pearl-c-sm120-v1-h1`) commits every linear call of a vLLM model
on the device, timed against plain vLLM on the same card at the panel's two shapes.

  python benchmarks/pouw/pearl_c_vllm/e2e.py --model DIR --kernel benchmarks/pouw/pearl_c_sm120 --cubin SHIP/pearl_c_sm120.cubin \\
      --out RUN_DIR/e2e.json [--modes bf16,fp8,pearlc-nohash,pearlc] [--retain RUN_DIR/pass] [--reps 3]

The run (research store ``docs/pouw/mvp-e2e.md``, "RTX PRO 6000"):
- **Model:** Llama-3.1-8B-Instruct in BF16, TP 1, eager, prefix caching off, one engine per mode in its own process.  Every
  linear is in the kernels' domain: k is 4,096 or 14,336, and n is 6,144, 4,096 or 28,672.
- **Prefill** (the panel's m = 8,192): one prompt of 8,192 random tokens with max_tokens 1 and a batch budget of 8,192 tokens,
  so each linear call has m = 8,192.  One warm-up prompt, then `--reps` timed prompts, each different.
- **Decode** (the panel's m = 32): 32 prompts of 64 tokens.  The step is (t(1 + D tokens) − t(1 token)) / D over D = 64
  decode steps, whose linear calls have m = 32.
- **Modes:**
  - ``bf16``: stock vLLM;
  - ``fp8``: stock ``quantization="fp8"``, the same precision's plain path and the headline divisor;
  - ``pearlc-nohash``: the same forming and GEMM, with no hashing and no records;
  - ``pearlc``: every call committed, each timed prompt its own pass (transcript), whose tile tree is timed with it.

  The slowdown splits into the GEMM (``gemm_pearlc``'s kernel time in ``pearlc-nohash``), hashing and records (``pearlc`` less
  ``pearlc-nohash``), and everything else.
- **Gates, before any timing, in the mode's own process:** at every distinct linear shape and both m, served y is held to
  the float product x·Wᵀ (relative RMS error under 8%, the scheme's tested bound).  With hashing, the same call made twice,
  each from a fresh pass on poisoned outputs, must give the same A root and tile root, and the negative control, the call
  once more with its tile leaves omitted, must not.
- **The verify pass** (``pearlc`` with ``--retain``): after the timing, the first timed prefill prompt and the decode prompts
  are replayed, each as a retained pass (`PearlCDeviceRun.begin(retain=…)`) on poisoned buffers.  Each pass's commitment
  must equal its timed pass's.  Then the negative control: one decode prompt as a retained pass that writes none of the
  commitment (`CONTROL_OMITS`: A's row tree, the message digests, the tile leaves) into the poisoned buffers.  The three
  manifests go to `verify_run.py`, the reference verifier, after the window: the first two must be accepted and the control
  rejected (the panel's rule of 11:50Z).
- **The panel's shapes** (the coordinator, 11:50Z): `SHAPES`, under ``pearl-c-sm120`` v1-h1; `panel_rows.py` writes the rows.

Every rep records the SM clock (the panel's clock rule), through run.py's NVML reader.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SCHEME = "pearl-c-sm120-v1-h1"
MODES = ("bf16", "fp8", "pearlc-nohash", "pearlc")
PEARLC = ("pearlc-nohash", "pearlc")
PREFILL, BATCH, PROMPT, STEPS = 8192, 32, 64, 64
#: the served y's bound against the float product (the scheme's tested bound, protocols/pouw's sm120 useful-output test)
ERR = 0.08
#: the negative control's omitted steps: every write of the commitment (A's row tree, the digests, the tile leaves)
CONTROL_OMITS = frozenset({"pad_a", "node_keys_a", "hash_rows_a", "tree_a", "hash_msg", "hash_leaf"})
#: the panel's shape names for the two phases
SHAPES = {"prefill_8192": "e2e-llama31-8b-vllm-m8192", "decode_b32_step": "e2e-llama31-8b-vllm-m32"}


def engine(args, mode: str):
    from vllm import LLM
    kw = {"model": args.model, "enforce_eager": True, "gpu_memory_utilization": args.mem, "max_num_seqs": BATCH,
          "max_model_len": PREFILL + 64, "max_num_batched_tokens": PREFILL, "enable_prefix_caching": False, "seed": 0}
    if args.revision:
        kw["revision"] = args.revision
    if mode == "fp8":
        kw["quantization"] = "fp8"
    try:
        return LLM(**kw, compilation_config={"mode": 0})
    except TypeError:
        return LLM(**kw)


def install(model, mode: str, args):
    """Every LinearBase layer's apply, found by type, through a `PearlCDeviceRun` over the layers' own weights."""
    from verity_pouw.audit import Epoch, Workload
    from verity_pouw.protocol import Matmul, Shape
    from verity_vllm.protocol_options import pouw_pearl_c_device as PPD
    from vllm.model_executor.layers.linear import LinearBase

    layers = [m for m in model.modules() if isinstance(m, LinearBase)]
    weights = {w: layer.weight.data for w, layer in enumerate(layers)}
    bad = PPD.refusals(SCHEME, {w: tuple(t.shape) for w, t in weights.items()})
    if bad:
        raise SystemExit(f"{SCHEME} refuses the model: " + "; ".join(bad))
    kernel = PPD.Kernel.load(args.kernel, args.cubin)
    beacon = os.urandom(32)
    roots = PPD.weight_roots(kernel, weights)
    declared = Workload(tuple(Matmul(w, Shape(PPD.ROW_ALIGN, t.shape[1], t.shape[0]), w) for w, t in weights.items()))
    salt = Epoch.start(declared, roots, beacon).salt
    run = PPD.PearlCDeviceRun(PPD.scheme_of(SCHEME), f"pouw:{SCHEME}", kernel, weights, roots, salt, beacon,
                              hashing=mode == "pearlc", time_gemm=True)

    def make(original):
        def apply(layer, x, bias=None):
            w = getattr(layer, "_pouw_w", None)
            return original(layer, x, bias) if w is None else run.gemm(w, x, layer.weight, bias)
        return apply

    for w, layer in enumerate(layers):
        layer._pouw_w = w
    for qm in {id(layer.quant_method): layer.quant_method for layer in layers}.values():
        qm.apply = make(type(qm).apply.__get__(qm))
    return run, weights


def gate(run, weights: dict) -> list[dict]:
    """Served y against x·Wᵀ, and with hashing the same call's commitment twice, at every distinct (n, k) and both m."""
    import torch
    out, seen = [], set()
    gen = torch.Generator(device=next(iter(weights.values())).device).manual_seed(20260930)
    for w, W in weights.items():
        if tuple(W.shape) in seen:
            continue
        seen.add(tuple(W.shape))
        for m in (PREFILL, BATCH):
            x = torch.randn(m, W.shape[1], device=W.device, generator=gen).to(torch.bfloat16)
            commits, served = [], None
            for omit in ((), (), ("hash_leaf",)) if run.hashing else ((),):
                run.begin()
                run.poison()
                run.omit = frozenset(omit)
                y = run.gemm(w, x, W, None)
                served = y if served is None else served
                commits.append({k: v for k, v in run.end().items() if k in ("transcript_root", "activation_roots")})
            run.omit = frozenset()
            ref = x.float() @ W.float().T
            err = float(((served.float() - ref).pow(2).sum() / ref.pow(2).sum()).sqrt())
            same = len(commits) == 1 or commits[0] == commits[1]
            control = len(commits) == 1 or commits[2]["transcript_root"] != commits[0]["transcript_root"]
            out.append({"weight": w, "n": W.shape[0], "k": W.shape[1], "m": m, "rel_rms": err, "commitment_repeats": same,
                        "omitted_leaves_rejected": control, "ok": err < ERR and same and control})
    run.details["passes"].clear()
    run.gemm_ms()
    return out


def clocks() -> dict:
    """The SM and memory clocks now, from nvidia-smi (the panel's per-rep clock record)."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader,nounits",
                            "-i", "0"], capture_output=True, text=True, timeout=10, check=False).stdout.strip().split(", ")
        return {"uuid": q[0], "sm_mhz": int(q[1]), "mem_mhz": int(q[2]), "watts": float(q[3])}
    except Exception as ex:  # noqa: BLE001 - a missing reading is recorded, not fatal
        return {"error": f"{type(ex).__name__}: {ex}"}


def run_mode(args, mode: str) -> dict:
    import torch
    from vllm import SamplingParams
    from vllm_bench import PROMPTS

    llm = engine(args, mode)
    run, weights = (None, None)
    result: dict = {"mode": mode}
    if mode in PEARLC:
        from vllm_bench import find_model
        run, weights = install(find_model(llm), mode, args)
        result["gate"] = gate(run, weights)
        result["committer"] = run.details["committer"]
        result["excluded_b"] = sum(run.details["excluded_b"].values())
        if not all(g["ok"] for g in result["gate"]):
            result["error"] = "gate failed"
            return result
    tok = llm.get_tokenizer()
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    outs = llm.generate(msgs, SamplingParams(temperature=0.0, max_tokens=48), use_tqdm=False)
    result["tokens"] = [list(o.outputs[0].token_ids) for o in outs]
    result["texts"] = [o.outputs[0].text for o in outs[:4]]
    rng = torch.Generator().manual_seed(8192)
    ids = lambda n: [int(t) for t in torch.randint(100, 20000, (n,), generator=rng)]

    def phase(prompts, sp, retain=None):
        if run:
            run.begin(retain, os.environ.get("RESEARCH_RUN_ID"))
            run.gemm_ms()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        commit = run.end() if run else {}
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gemm = run.gemm_ms()[0] / 1e3 if run else 0.0
        return dt, gemm, commit

    one = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    many = SamplingParams(temperature=0.0, max_tokens=1 + STEPS, ignore_eos=True)
    phase([{"prompt_token_ids": ids(PREFILL)}], one)
    prefill = [{"prompt_token_ids": ids(PREFILL)} for _ in range(args.reps)]
    reps = []
    for p in prefill:
        dt, gemm, commit = phase([p], one)
        reps.append({"s": dt, "gemm_s": gemm, "clock": clocks(), "commitment": commit})
    decode = [{"prompt_token_ids": ids(PROMPT)} for _ in range(BATCH)]
    phase(decode, many)
    dreps = []
    for _ in range(args.reps):
        t1, g1, _ = phase(decode, one)
        tn, gn, cn = phase(decode, many)
        dreps.append({"step_s": (tn - t1) / STEPS, "gemm_step_s": (gn - g1) / STEPS, "t1_s": t1, "tn_s": tn, "clock": clocks(),
                      "commitment": cn})
    result["timing"] = {"prefill_8192_s": statistics.median(r["s"] for r in reps),
                        "prefill_8192_gemm_s": statistics.median(r["gemm_s"] for r in reps),
                        "decode_b32_step_s": statistics.median(r["step_s"] for r in dreps),
                        "decode_b32_step_gemm_s": statistics.median(r["gemm_step_s"] for r in dreps)}
    result["reps"] = {"prefill_8192": reps, "decode_b32": dreps}
    if run and run.hashing and args.retain:
        out = {}
        for name, prompts, sp, want in (("prefill", prefill[:1], one, reps[0]["commitment"]),
                                        ("decode", decode, many, dreps[0]["commitment"])):
            _, _, got = phase(prompts, sp, retain=str(Path(args.retain) / name))
            same = all(got.get(k) == want.get(k) for k in ("transcript_root", "activation_roots", "units", "tiles"))
            out[name] = {"manifest": got.get("manifest"), "commitment": got, "same_commitment_as_timed": same}
        run.omit = CONTROL_OMITS
        _, _, got = phase(decode[:1], one, retain=str(Path(args.retain) / "control"))
        run.omit = frozenset()
        out["control"] = {"manifest": got.get("manifest"), "commitment": got, "omitted": sorted(CONTROL_OMITS)}
        result["verify_pass"] = out
    return result


def split(results: dict) -> dict:
    """Per phase: ``pearlc`` against ``fp8`` (the headline) and ``bf16``, and its time split into the GEMM (the no-hash mode's
    kernel), hashing and records (``pearlc`` less the no-hash mode) and everything else."""
    t = {m: r["timing"] for m, r in results.items() if "timing" in r}
    out = {}
    for key in ("prefill_8192", "decode_b32_step"):
        if not all(m in t for m in MODES):
            continue
        full, nohash, gemm = t["pearlc"][f"{key}_s"], t["pearlc-nohash"][f"{key}_s"], t["pearlc-nohash"][f"{key}_gemm_s"]
        row = {"bf16_s": t["bf16"][f"{key}_s"], "fp8_s": t["fp8"][f"{key}_s"], "pearlc_s": full, "nohash_s": nohash, "gemm_s": gemm,
               "hashing_s": full - nohash, "else_s": nohash - gemm}
        for base in ("fp8", "bf16"):
            b = row[f"{base}_s"]
            row.update({f"slowdown_vs_{base}": full / b, f"gemm_x_{base}": gemm / b, f"hashing_x_{base}": (full - nohash) / b,
                        f"else_x_{base}": (nohash - gemm) / b})
        out[key] = {"shape": SHAPES[key], **row}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--kernel", default=str(HERE.parent / "pearl_c_sm120"), help="GPU 1's kernel directory (run.py, fixture.py)")
    ap.add_argument("--cubin", help="build.sh's pearl_c_sm120.cubin (default: the kernel directory's)")
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "e2e.json")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem", type=float, default=0.45, help="vLLM's gpu_memory_utilization, the same in every mode")
    ap.add_argument("--retain", help="the verify pass's directory (pearlc only)")
    ap.add_argument("--single", help="run one mode in this process and write its result to --out")
    args = ap.parse_args()
    if args.single:
        args.out.write_text(json.dumps(run_mode(args, args.single)))
        return
    results = {}
    for mode in args.modes.split(","):
        part = args.out.with_suffix(f".{mode}.json")
        cmd = [sys.executable, __file__, "--model", args.model, "--kernel", args.kernel, "--out", str(part), "--single", mode,
               "--reps", str(args.reps), "--mem", str(args.mem), *(["--cubin", args.cubin] if args.cubin else []),
               *(["--revision", args.revision] if args.revision else []), *(["--retain", args.retain] if args.retain else [])]
        subprocess.run(cmd, check=False)
        results[mode] = json.loads(part.read_text()) if part.exists() else {"mode": mode, "error": "the mode's process wrote no result"}
        print(json.dumps({k: v for k, v in results[mode].items() if k not in ("tokens", "reps")}), flush=True)
    ref = results.get("bf16", {}).get("tokens")

    def agree(a, b):
        same = sum(x == y for s, u in zip(a, b) for x, y in zip(s, u))
        return same / max(1, sum(min(len(s), len(u)) for s, u in zip(a, b)))

    summary = {"split": split(results),
               "token_agreement_vs_bf16": {m: agree(r["tokens"], ref) for m, r in results.items() if ref and "tokens" in r},
               "gates": {m: all(g["ok"] for g in r.get("gate", [])) for m, r in results.items() if m in PEARLC},
               "verify_pass": {k: {"manifest": v["manifest"], "same_commitment_as_timed": v.get("same_commitment_as_timed")}
                               for k, v in results.get("pearlc", {}).get("verify_pass", {}).items()}}
    args.out.write_text(json.dumps({"schema": "verity-pouw/pearl-c-vllm-e2e/v1", "scheme": SCHEME, "run": os.environ.get("RESEARCH_RUN_ID"),
                                    "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary), flush=True)
    import run_result
    ok = (all("timing" in r for r in results.values()) and all(summary["gates"].values())
          and all(v["same_commitment_as_timed"] for k, v in summary["verify_pass"].items() if k != "control"))
    ms = [(f"{m}.{k}", v, "s") for m, r in results.items() for k, v in r.get("timing", {}).items()]
    run_result.write(args.out, ok, "pearl-c vLLM e2e: every mode gated and timed, and each verify pass repeats its timed commitment",
                     json.dumps(summary), ms)


if __name__ == "__main__":
    main()
