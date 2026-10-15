"""PoUW's overhead against plain vLLM, on one RTX 4090, with Qwen2.5-3B-Instruct.

    python benchmarks/pouw/vllm_bench.py --model /path/to/qwen2.5-3b-instruct --out RESULT.json [--modes bf16,fp8,w7a7,ncp,pearl]

This is the harness's stand-in for the vLLM adapter (``integrations/vllm/verity_vllm/protocol_options/pouw.py``, which
waits for the vLLM coordinator's OK): it wraps every ``LinearBase`` layer's ``quant_method.apply``, found by type, on stock
vLLM, and restores nothing because each mode runs in a fresh engine.  Modes:

- ``bf16``: stock vLLM;  ``fp8``: stock ``quantization="fp8"``;
- ``w7a7``: int7 weights per output channel and int7 activations per token, one plain int8 GEMM (``torch._int_mm``):
  NCP's useful workload without the protocol;
- ``ncp``: the same int7 operands through NCP: X = [A + E₁ | A + E₂ | −(A + E₃)], Y = [B + F₁ ; B + F₂ ; B + F₃] with the
  word-granular production P, one (m, 3k, n) s8 GEMM that reads every accumulator once per 16-deep step (the kernel in
  ``kernels/pouw_gemm.cu``).  X·Yᵀ = A·Bᵀ exactly, so its tokens must equal ``w7a7``'s.  Both operands are formed in every call
  (`ncp_x`, `ncp_y`, F₁ drawn anew) and dropped after its GEMM: the layer keeps only its int7 weight, as ``w7a7`` does, and
  no weight-derived copy stays resident across forwards;
- ``pearl``: Pearl's noisy quantization to e4m3 (rank-32 noise lines of norm 256, δ = 1), the windowed e4m3 GEMM of the
  ``ada`` device, and the peel back to BF16;
- ``ncp2``: every call committed by `ncp-v2`'s serving kernel (`pouw_device.FusedCommitter`, `ncp2_fused.cu`): the key, the
  noise from SHAKE256, X and Y, the tiles hashed as they are made, the exact dequantization, which is the served y, and the
  units root; Y's leaves cached per weight. ``ncp2-nohash`` is the same with every hash and XOF left out (a stand-in noise):
  the difference is hashing. ``ncp2-lowbyte`` swaps the tiles' word leaves for the low-byte variant (timed, not committed; its
  γ is unproved, so its numbers are never results).
  Each ncp2 mode first checks its device path against `pouw_native` on one call, and each timing phase records the tile
  kernel's summed time (the GEMM, in ``ncp2-nohash``).

Noise in the timed modes comes from torch's generator, not SHAKE256 or BLAKE3: the protocol's derivations and the
transcript hashing are costed separately (NCP: 3k/16 words per output at a measured SHAKE256 rate).
All modes run eagerly (the wrapper is not graph-capturable yet), at the same memory utilization.
"""

from __future__ import annotations

import argparse
import contextlib
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


NCP2_MODES = ("ncp2", "ncp2-nohash", "ncp2-lowbyte")
#: what every low-byte number carries: the variant's γ has no proof, so it is timed and never reported as a result
LOWBYTE_STATUS = "gamma unproved: not a result"


def ncp2_committer(mode: str):
    from verity_vllm.protocol_options import pouw_device as PD
    PD.configure(build_dir=os.environ.get("VERITY_POUW_CUDA_BUILD"), arch=os.environ.get("VERITY_POUW_CUDA_ARCH"))
    flags = {"ncp2": PD.HASH | PD.CACHE_Y, "ncp2-nohash": 0, "ncp2-lowbyte": PD.HASH | PD.CACHE_Y | PD.LOWBYTE}[mode]
    return PD.FusedCommitter("ncp-v2", bytes(range(32)), flags | PD.TIME_TILES)


def ncp2_self_check(committer) -> dict:
    """The committer's device path on one random call against `pouw_native`: y always, and the units root when it commits
    the word leaves. Refuses to go on otherwise."""
    import numpy as np
    import torch
    from verity_vllm.protocol_options import pouw_device as PD
    from verity_vllm.protocol_options import pouw_native as PN

    call = PD.gate_call("vllm self-check", 16, 896, 1152)
    ref = PN.commit_call(0, "ncp-v2", call)
    b, t = call.b.rows(0, call.b.n)
    cuda = lambda v, dt: torch.from_numpy(np.ascontiguousarray(v).view(dt)).cuda()
    word = committer.flags & PD.HASH and not committer.flags & PD.LOWBYTE
    c, root = PD.FusedCommitter("ncp-v2", call.salt, committer.flags & ~PD.CACHE_Y), np.zeros(32, np.uint8)
    y = c(cuda(call.a, np.int8), cuda(call.s, np.int32), cuda(b, np.int8), cuda(t, np.int32), cuda(call.bias, np.int16), call.forward,
          call.position, call.weight, cuda(call.rowids, np.int32), cuda(call.colids, np.int32), root=root if word else None)
    torch.cuda.synchronize()
    c.tile_ms()
    ok = {"y": bool(np.array_equal(y.cpu().numpy().view(np.uint16), ref.y)), "root": bool(root.tobytes() == ref.root) if word else None}
    if not ok["y"] or ok["root"] is False:
        raise SystemExit(f"ncp2 self-check failed: {ok}")
    return ok


def ncp_x(Aq, pi, gen):
    """X = [A + E₁ | A + E₂ | −(A + E₁ + E₂)] for the int7 rows A, E₁ drawn now and E₂ = E₁ word-permuted by ``pi``."""
    import torch
    E1 = torch.randint(-31, 33, tuple(Aq.shape), dtype=torch.int8, device=Aq.device, generator=gen)
    E2 = E1[:, pi]
    return torch.cat([Aq + E1, Aq + E2, -(Aq + E1 + E2)], dim=1).view(torch.uint8).contiguous()


def ncp_y(Wq, pi, gen):
    """Y = [B + F₁ ; B + F₂ ; B + F₁ + F₂] for the int7 weight B, F₁ drawn now and F₂ its negation under ``pi``, so that
    X·Yᵀ = A·Bᵀ.  Formed for one call and dropped after it: 3 B per weight that never outlives the GEMM."""
    import torch
    F1 = torch.randint(-31, 33, tuple(Wq.shape), dtype=torch.int8, device=Wq.device, generator=gen)
    F2 = torch.empty_like(F1)
    F2[:, pi] = -F1
    return torch.cat([Wq + F1, Wq + F2, Wq + F1 + F2], dim=1).view(torch.uint8).contiguous()


class ModeExecutor:
    """One bench mode's GEMM at every linear site (`verity_vllm.linear.api.LinearExecutor`), over the operands `install`
    prepared per weight: weight state kept across calls, no controls, serial only.  Its pass is itself, one for the engine's
    life (`run_mode`)."""

    keeps_weight_state = True
    accepts = frozenset({"bf16", "fp16", "fp32"})
    controls = frozenset()
    schedules = frozenset({"serial"})

    def __init__(self, mode: str, states: dict, compute, stats: dict) -> None:
        from verity_vllm.linear.api import PassSpec
        self.name, self.states, self.compute, self.stats, self.spec = f"bench:{mode}", states, compute, stats, PassSpec()

    def refusals(self, sites) -> list[str]:
        return [f"linear {s.weight_id} ({s.qualname}) has no prepared operands" for s in sites if s.weight_id not in self.states]

    def begin(self, spec):
        return self

    def call(self, site, x, weight, bias):
        return self.compute(self.states[site.weight_id], x, bias)

    def end(self) -> dict:
        return dict(self.stats)

    def close(self) -> None:
        pass


def install(model, mode: str, committer=None):
    """Every LinearBase layer (found by type, `verity_vllm.linear.vllm_site`) computed for `mode` by one `ModeExecutor` on a
    `Stack` attached through `engine.hooks`; returns the shared stats, the stack (its arm active) and the hooks, which the
    caller uninstalls so the patches don't keep the engine alive."""
    import torch
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.linear import vllm_site
    from verity_vllm.linear.api import Arm, Stack

    stats = {"calls": 0, "macs": 0, "transcript_bytes": 0, "layers": 0, "y_formed_bytes": 0}
    count = {"forward": 0, "rows": 0}
    ext = None
    if mode in ("ncp", "pearl"):
        from kernels import load
        ext = load()
    sites = vllm_site.discover(model)
    stats["layers"] = len(sites)
    states: dict[int, dict] = {}
    perms: dict[int, torch.Tensor] = {}
    gen = torch.Generator(device="cuda").manual_seed(20260928)

    def lines(rows, k):
        x = torch.randint(1, 129, (rows, k), device="cuda", generator=gen).float()
        x = x * (torch.randint(0, 2, (rows, k), device="cuda", generator=gen) * 2 - 1)
        return (x * (256.0 / x.norm(dim=1, keepdim=True))).to(torch.float8_e4m3fn).to(torch.bfloat16)

    for site, layer in sites:
        li = site.weight_id
        W = layer.weight.data
        n, k = W.shape
        st = {"n": n, "k": k}
        if mode in NCP2_MODES:
            Wq, sw = int7(W, 1)
            st.update(B=Wq.contiguous(), T=sw.view(n).float().contiguous().view(torch.int32), index=li,
                      colids=torch.arange(n, dtype=torch.int32, device=W.device), zero_bias=torch.zeros(n, dtype=torch.int16, device=W.device))
            layer.weight.data = torch.empty(0, dtype=W.dtype, device=W.device)
        elif mode in ("w7a7", "ncp"):
            Wq, sw = int7(W, 1)
            st["sw"] = sw.view(1, n)
            st["Wt"] = Wq.contiguous()
            if mode == "ncp":
                if k not in perms:
                    from verity.core.protocols.pouw.schemes.ncp import word_perm
                    perms[k] = torch.tensor(word_perm(k)[0], device="cuda", dtype=torch.long)
                st["pi"] = perms[k]
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
        states[li] = st

    def compute(st, x, bias=None):
        n, k = st["n"], st["k"]
        x2 = x.reshape(-1, k)
        m = x2.shape[0]
        stats["calls"] += 1
        stats["macs"] += m * k * n
        if mode in NCP2_MODES:
            Aq, sx = int7(x2, 1)
            mp, sf = (m + 15) // 16 * 16, sx.view(m).float()
            if mp != m:
                Aq = torch.cat([Aq, torch.zeros(mp - m, k, dtype=torch.int8, device=Aq.device)])
                sf = torch.cat([sf, torch.zeros(mp - m, dtype=torch.float32, device=sf.device)])
            if st["index"] == 0:
                count["forward"] += 1
            rowids = torch.arange(count["rows"], count["rows"] + mp, dtype=torch.int32, device=Aq.device)
            count["rows"] += mp
            b16 = st["zero_bias"] if bias is None else bias.to(torch.bfloat16).contiguous().view(torch.int16)
            y16 = committer(Aq.contiguous(), sf.contiguous().view(torch.int32), st["B"], st["T"], b16, count["forward"],
                            st["index"] << 32, st["index"], rowids, st["colids"])
            stats["transcript_bytes"] += 4 * mp * n * (3 * k // 16)
            return y16[:m].view(torch.bfloat16).reshape(*x.shape[:-1], n)
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
            Y = ncp_y(st["Wt"], st["pi"], gen)
            C, _, _ = ext.ncp_gemm(ncp_x(Aq, st["pi"], gen), Y, 1, cfg)
            del Y
            stats["y_formed_bytes"] += 3 * n * k
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

    stack = Stack([site for site, _ in sites], ModeExecutor(mode, states, compute, stats))
    bad = stack.refusals()
    if bad:
        raise SystemExit(f"{mode} refuses the model: " + "; ".join(bad))
    hooks = Hooks(tag=f"bench:{mode}")
    vllm_site.attach(stack, sites, hooks)
    stack.activate(Arm(mode, "bf16", f"bench:{mode}"))
    torch.cuda.empty_cache()
    return stats, stack, hooks


def engine(model_path: str, mode: str, max_seqs: int, max_len: int, revision: str | None = None):
    from vllm import LLM
    kw = dict(model=model_path, enforce_eager=True, gpu_memory_utilization=0.5, max_num_seqs=max_seqs, max_model_len=max_len, seed=0)
    if revision:
        kw["revision"] = revision
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

    llm = engine(args.model, mode, 32, 4096, args.revision)
    committer = ncp2_committer(mode) if mode in NCP2_MODES else None
    check = ncp2_self_check(committer) if committer else None
    stats, stack, hooks = (install(find_model(llm), mode, committer) if mode in ("w7a7", "ncp", "pearl", *NCP2_MODES)
                           else ({}, None, None))
    try:
        with stack.pass_() if stack is not None else contextlib.nullcontext():
            result = _measure(args, llm, mode, committer, check, stats)
    finally:
        if hooks is not None:
            hooks.uninstall()
    del llm, stack, hooks
    torch.cuda.empty_cache()
    return result


def _measure(args, llm, mode: str, committer, check, stats: dict) -> dict:
    import torch
    from vllm import SamplingParams

    def tiles(timing: dict, phase: str) -> None:
        if committer:
            torch.cuda.synchronize()
            ms, calls = committer.tile_ms()
            timing[f"{phase}_tile_ms"], timing[f"{phase}_calls"] = ms, calls
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
        tiles(timing, "warm")
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timing[f"decode_b{bs}_tok_s"], timing[f"decode_b{bs}_s"] = bs * 64 / dt, dt
        tiles(timing, f"decode_b{bs}")
    for plen in args.prefill_tokens:
        prompts = [{"prompt_token_ids": [int(t) for t in torch.randint(100, 20000, (plen,))]} for _ in range(2)]
        sp = SamplingParams(temperature=0.0, max_tokens=1)
        llm.generate(prompts[:1], sp, use_tqdm=False)
        torch.cuda.synchronize()
        tiles(timing, "warm")
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        timing[f"prefill_{plen}_tok_s"], timing[f"prefill_{plen}_s"] = 2 * plen / dt, dt
        tiles(timing, f"prefill_{plen}")
    timing.pop("warm_tile_ms", None)
    timing.pop("warm_calls", None)
    result["timing"] = timing
    if check:
        result["self_check"] = check
    if stats:
        stats["shake_est_s"] = stats["transcript_bytes"] / (SHAKE_GBPS * 1e9)
        result["stats"] = stats
    return result


def split(bf16: dict, full: dict, nohash: dict, lowbyte: dict | None) -> dict:
    """Per timed phase: the slowdown of ``ncp2`` against BF16 vLLM, and its time split into the GEMM (the no-hash control's
    tile kernel), hashing (``ncp2`` less the control) and everything else (the control less its tile kernel); with the
    low-byte variant's slowdown and hashing beside it."""
    out = {}
    for key in [k[:-2] for k in full if k.endswith("_s") and not k.endswith("_tok_s")]:
        if f"{key}_s" not in bf16 or f"{key}_s" not in nohash:
            continue
        t_b, t_f, t_n = bf16[f"{key}_s"], full[f"{key}_s"], nohash[f"{key}_s"]
        gemm = nohash.get(f"{key}_tile_ms", 0.0) / 1e3
        row = {"bf16_s": t_b, "ncp2_s": t_f, "slowdown": t_f / t_b, "gemm_s": gemm, "hashing_s": t_f - t_n, "else_s": t_n - gemm,
               "gemm_x_bf16": gemm / t_b, "hashing_x_bf16": (t_f - t_n) / t_b, "else_x_bf16": (t_n - gemm) / t_b}
        if lowbyte and f"{key}_s" in lowbyte:
            t_l = lowbyte[f"{key}_s"]
            row.update(lowbyte_s=t_l, lowbyte_slowdown=t_l / t_b, lowbyte_hashing_s=t_l - t_n, lowbyte_hashing_x_bf16=(t_l - t_n) / t_b,
                       lowbyte_status=LOWBYTE_STATUS)
        out[key] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "vllm.json")
    ap.add_argument("--modes", default="bf16,fp8,w7a7,ncp,pearl")
    ap.add_argument("--revision", help="the model's revision (vLLM's ``revision``)")
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
                        "--prefill-tokens", ",".join(map(str, args.prefill_tokens)),
                        *(["--revision", args.revision] if args.revision else [])], check=False)
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
    for m in NCP2_MODES:
        if m in results and "w7a7" in results:
            summary[m]["token_agreement_vs_w7a7"] = agree(results[m]["tokens"], results["w7a7"]["tokens"])
    if "ncp2" in results and "ncp2-nohash" in results and ref:
        summary["split"] = split(ref["timing"], results["ncp2"]["timing"], results["ncp2-nohash"]["timing"],
                                 results.get("ncp2-lowbyte", {}).get("timing"))
    args.out.write_text(json.dumps({"schema": "verity-pouw/vllm-bench/v1", "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary), flush=True)
    import run_result
    ok = summary.get("ncp", {}).get("tokens_equal_w7a7", True) is not False and len(results) == len(args.modes.split(","))
    ms = [(f"{m}.{k}", v, "tok/s" if k.endswith("tok_s") else "s" if k.endswith("_s") else "ms" if k.endswith("_ms") else "calls")
          for m, r in results.items() for k, v in r["timing"].items()]
    run_result.write(args.out, ok, "vllm_bench: every mode ran, and NCP's greedy tokens equal W7A7's", json.dumps(summary), ms)


if __name__ == "__main__":
    main()
