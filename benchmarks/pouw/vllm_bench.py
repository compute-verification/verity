"""PoUW's overhead against plain vLLM, on one RTX 4090, with Qwen2.5-3B-Instruct.

    python benchmarks/pouw/vllm_bench.py --model /path/to/qwen2.5-3b-instruct --out RESULT.json [--modes bf16,fp8,w7a7,ncp,pearl]

This is the harness's stand-in for the vLLM adapter (``integrations/vllm/verity_vllm/protocol_options/pouw.py``, which
waits for the vLLM coordinator's OK): it wraps every ``LinearBase`` layer's ``quant_method.apply``, found by type, on stock
vLLM, and restores nothing because each mode runs in a fresh engine.  Modes:

- ``bf16``: stock vLLM;  ``fp8``: stock ``quantization="fp8"``;
- ``w7a7``: int7 weights per output channel and int7 activations per token, one plain int8 GEMM (``torch._int_mm``):
  NCP's useful workload without the protocol;
- ``ncp``: the same int7 operands through NCP: X = [A + E₁ | A + E₂ | −(A + E₃)], Y = [B + F₁ ; B + F₂ ; B + F₃] with the
  word-granular production P, one (m, 3k, n) s8 GEMM that reads every accumulator once per 16-deep step (the kernel gated
  by ``gemm_bench.py``).  X·Yᵀ = A·Bᵀ exactly, so its tokens must equal ``w7a7``'s;
- ``pearl``: Pearl's noisy quantization to e4m3 (rank-32 noise lines of norm 256, δ = 1), the windowed e4m3 GEMM of the
  ``ada`` device, and the peel back to BF16.

Noise in the timed modes comes from torch's generator, not SHAKE256 or BLAKE3: the protocol's derivations and the
transcript hashing are costed separately (NCP: 3k/16 words per output at the research store's measured SHAKE256 rate).
All modes run eagerly (the wrapper is not graph-capturable yet), at the same memory utilization.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).parent))

PROMPTS = [
    "Explain why the sky is blue in two sentences.", "Write a haiku about matrix multiplication.",
    "What is the capital of Australia, and why is it not Sydney?", "List three uses of a hash function.",
    "Translate 'good morning, how are you?' into French and German.", "Summarize the plot of Hamlet in one paragraph.",
    "Give a Python function that reverses a linked list.", "What are the main causes of inflation?",
    "Describe the water cycle to a ten-year-old.", "Why do GPUs run neural networks faster than CPUs?",
    "Write a limerick about a cat who loves tensors.", "What is the difference between TCP and UDP?",
    "Explain proof of work in simple terms.", "Name the planets of the solar system in order.",
    "How does a refrigerator keep food cold?", "What is a prime number? Give five examples.",
]
SHAKE_GBPS = 650.0
R = 32


def int7(x, dim):
    import torch
    s = x.float().abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / 63
    return torch.clamp(torch.round(x.float() / s), -64, 63).to(torch.int8), s


def install(model, mode: str) -> dict:
    """Wrap every LinearBase layer's quant_method.apply (found by type) for `mode`; returns the shared stats."""
    import torch
    from vllm.model_executor.layers.linear import LinearBase

    stats = {"calls": 0, "macs": 0, "transcript_bytes": 0, "layers": 0}
    ext = None
    if mode in ("ncp", "pearl"):
        from kernels import load
        ext = load()
    layers = [m for m in model.modules() if isinstance(m, LinearBase)]
    stats["layers"] = len(layers)
    perms: dict[int, torch.Tensor] = {}
    gen = torch.Generator(device="cuda").manual_seed(20260928)

    def lines(rows, k):
        x = torch.randint(1, 129, (rows, k), device="cuda", generator=gen).float()
        x = x * (torch.randint(0, 2, (rows, k), device="cuda", generator=gen) * 2 - 1)
        return (x * (256.0 / x.norm(dim=1, keepdim=True))).to(torch.float8_e4m3fn).to(torch.bfloat16)

    for layer in layers:
        W = layer.weight.data
        n, k = W.shape
        st = {"n": n, "k": k}
        if mode in ("w7a7", "ncp"):
            Wq, sw = int7(W, 1)
            st["sw"] = sw.view(1, n)
            if mode == "w7a7":
                st["Wt"] = Wq.contiguous()
            else:
                if k not in perms:
                    from verity_pouw.schemes.ncp import word_perm
                    perms[k] = torch.tensor(word_perm(k)[0], device="cuda", dtype=torch.long)
                pi = perms[k]
                F1 = torch.randint(-31, 33, (n, k), dtype=torch.int8, device="cuda", generator=gen)
                F2 = torch.empty_like(F1)
                F2[:, pi] = -F1
                st["Y"] = torch.cat([Wq + F1, Wq + F2, Wq + F1 + F2], dim=1).view(torch.uint8).contiguous()
                st["pi"] = pi
                del F1, F2
            layer.weight.data = torch.empty(0, dtype=W.dtype, device=W.device)
        elif mode == "pearl":
            Wf = W.float()
            l2 = Wf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(2 ** -32)
            linf = Wf.abs().amax(dim=1, keepdim=True).clamp_min(2 ** -32)
            a_w = 448.0 / (R ** 0.5 * l2 + linf)
            b_w = a_w * l2 * R ** 0.5 / 256.0 ** 2
            EB, FB, FA = lines(n, R), lines(k, R), lines(k, R)
            NB = (EB @ FB.T).float()
            Bq = torch.clamp(a_w * Wf + b_w * NB, -448, 448).to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
            G = (FA.T.float() @ FB.float()) @ EB.T.float()                          # r x n
            Q = torch.cat([(EB.float() * b_w).T, a_w.T * (FA.T.float() @ Wf.T) + b_w.T * G], dim=0).to(torch.bfloat16)
            st.update(Bq=Bq, a_w=a_w.view(1, n), Q=Q, FB=FB, FA=FA)
            layer.weight.data = torch.empty(0, dtype=W.dtype, device=W.device)
        layer._pouw = st

    def make(original):
        def apply(layer, x, bias=None):
            st = getattr(layer, "_pouw", None)
            if st is None:
                return original(layer, x, bias)
            n, k = st["n"], st["k"]
            x2 = x.reshape(-1, k)
            m = x2.shape[0]
            stats["calls"] += 1
            stats["macs"] += m * k * n
            if mode == "w7a7":
                Aq, sx = int7(x2, 1)
                mp = max(32, (m + 7) // 8 * 8)
                if mp != m:
                    Aq = torch.cat([Aq, torch.zeros(mp - m, k, dtype=torch.int8, device=Aq.device)])
                C = torch._int_mm(Aq, st["Wt"].T)[:m]
                y = (C.float() * sx * st["sw"]).to(x.dtype)
            elif mode == "ncp":
                Aq, sx = int7(x2, 1)
                cfg = 1 if m <= 32 else 0
                mp = (m + 15) // 16 * 16 if cfg else (m + 127) // 128 * 128
                if mp != m:
                    Aq = torch.cat([Aq, torch.zeros(mp - m, k, dtype=torch.int8, device=Aq.device)])
                E1 = torch.randint(-31, 33, (mp, k), dtype=torch.int8, device=Aq.device, generator=gen)
                E2 = E1[:, st["pi"]]
                X = torch.cat([Aq + E1, Aq + E2, -(Aq + E1 + E2)], dim=1).view(torch.uint8).contiguous()
                C, _, _ = ext.ncp_gemm(X, st["Y"], 1, cfg)
                stats["transcript_bytes"] += 4 * m * n * (3 * k // 16)
                y = (C[:m].float() * sx * st["sw"]).to(x.dtype)
            else:
                xf = x2.float()
                l2 = xf.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(2 ** -32)
                linf = xf.abs().amax(dim=1, keepdim=True).clamp_min(2 ** -32)
                a = 448.0 / (R ** 0.5 * l2 + linf)
                b = a * l2 * R ** 0.5 / 256.0 ** 2
                mp = (m + 15) // 16 * 16 if m <= 32 else (m + 127) // 128 * 128
                EA = lines(m, R)
                Aq = torch.clamp(a * xf + b * (EA @ st["FA"].T).float(), -448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
                if mp != m:
                    Aq = torch.cat([Aq, torch.zeros(mp - m, k, dtype=torch.uint8, device=Aq.device)])
                cfg = 0 if mp >= 128 and n % 128 == 0 else 1
                Ct = ext.fp8_gemm(Aq.contiguous(), st["Bq"], cfg)[:m]
                P = torch.cat([(a * (x2 @ st["FB"]).float()), b * EA.float()], dim=1).to(torch.bfloat16)
                y = ((Ct - (P @ st["Q"]).float()) / (a * st["a_w"])).to(x.dtype)
            if bias is not None:
                y = y + bias
            return y.reshape(*x.shape[:-1], n)
        return apply

    wrapped = set()
    for layer in layers:
        qm = layer.quant_method
        if id(qm) not in wrapped:
            qm.apply = make(type(qm).apply.__get__(qm)) if "apply" not in vars(qm) else qm.apply
            wrapped.add(id(qm))
    torch.cuda.empty_cache()
    return stats


def engine(model_path: str, mode: str, max_seqs: int, max_len: int):
    from vllm import LLM
    kw = dict(model=model_path, enforce_eager=True, gpu_memory_utilization=0.5, max_num_seqs=max_seqs, max_model_len=max_len, seed=0)
    if mode == "fp8":
        kw["quantization"] = "fp8"
    try:
        return LLM(**kw, compilation_config={"mode": 0})
    except TypeError:
        return LLM(**kw)


def find_model(llm):
    holder = {}

    def grab(model):
        holder["model"] = model
        return None

    llm.apply_model(grab)
    return holder["model"]


def run_mode(args, mode: str) -> dict:
    import torch
    from vllm import SamplingParams

    llm = engine(args.model, mode, 32, 4096)
    stats = install(find_model(llm), mode) if mode in ("w7a7", "ncp", "pearl") else {}
    tok = llm.get_tokenizer()
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    greedy = SamplingParams(temperature=0.0, max_tokens=48, logprobs=1)
    outs = llm.generate(msgs, greedy, use_tqdm=False)
    result = {"mode": mode, "tokens": [list(o.outputs[0].token_ids) for o in outs],
              "texts": [o.outputs[0].text for o in outs[:4]]}
    timing = {}
    for bs in args.decode_batches:
        sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)
        prompts = [{"prompt_token_ids": [int(t) for t in torch.randint(100, 20000, (64,))]} for _ in range(bs)]
        llm.generate(prompts[:1], SamplingParams(max_tokens=4), use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        timing[f"decode_b{bs}_tok_s"] = bs * 64 / dt
    for plen in args.prefill_tokens:
        prompts = [{"prompt_token_ids": [int(t) for t in torch.randint(100, 20000, (plen,))]} for _ in range(2)]
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        llm.generate(prompts[:1], sp, use_tqdm=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        timing[f"prefill_{plen}_tok_s"] = 2 * plen / (time.perf_counter() - t0)
    result["timing"] = timing
    if stats:
        stats["shake_est_s"] = stats["transcript_bytes"] / (SHAKE_GBPS * 1e9)
        result["stats"] = stats
    del llm
    torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "vllm.json")
    ap.add_argument("--modes", default="bf16,fp8,w7a7,ncp,pearl")
    ap.add_argument("--decode-batches", type=lambda s: [int(x) for x in s.split(",")], default=[1, 8, 32])
    ap.add_argument("--prefill-tokens", type=lambda s: [int(x) for x in s.split(",")], default=[512, 2048])
    ap.add_argument("--single", help="run one mode in this process and write its result to --out (vLLM keeps device memory "
                                     "until the process exits, so each mode gets its own process)")
    args = ap.parse_args()
    if args.single:
        args.out.write_text(json.dumps(run_mode(args, args.single)))
        return
    import subprocess
    results = {}
    for mode in args.modes.split(","):
        part = args.out.with_suffix(f".{mode}.json")
        subprocess.run([sys.executable, __file__, "--model", args.model, "--out", str(part), "--single", mode,
                        "--decode-batches", ",".join(map(str, args.decode_batches)),
                        "--prefill-tokens", ",".join(map(str, args.prefill_tokens))], check=False)
        if not part.exists():
            results[mode] = {"mode": mode, "error": "the mode's process wrote no result"}
            continue
        results[mode] = json.loads(part.read_text())
        print(json.dumps({k: v for k, v in results[mode].items() if k != "tokens"}), flush=True)
        args.out.write_text(json.dumps({"schema": "verity-pouw/vllm-bench/v1", "results": results}, indent=1))
    results = {m: r for m, r in results.items() if "tokens" in r}
    ref = results.get("bf16")

    def agree(a, b):
        same = sum(x == y for s, t in zip(a, b) for x, y in zip(s, t))
        return same / max(1, sum(min(len(s), len(t)) for s, t in zip(a, b)))

    summary = {m: {"token_agreement_vs_bf16": agree(r["tokens"], ref["tokens"]) if ref else None} for m, r in results.items()}
    if "ncp" in results and "w7a7" in results:
        summary["ncp"]["tokens_equal_w7a7"] = results["ncp"]["tokens"] == results["w7a7"]["tokens"]
    args.out.write_text(json.dumps({"schema": "verity-pouw/vllm-bench/v1", "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary), flush=True)
    import run_result
    ok = summary.get("ncp", {}).get("tokens_equal_w7a7", True) is not False and len(results) == len(args.modes.split(","))
    ms = [(f"{m}.{k}", v, "tok/s") for m, r in results.items() for k, v in r["timing"].items()]
    run_result.write(args.out, ok, "vllm_bench: every mode ran, and NCP's greedy tokens equal W7A7's", json.dumps(summary), ms)


if __name__ == "__main__":
    main()
