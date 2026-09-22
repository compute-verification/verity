"""Capture the GEMM operands the *engine* actually feeds ``matmul_kernel_persistent`` for
Qwen2.5-1.5B under the POC's replay configuration (Table 2 row class ``the same GEMM inside the
replayed model``, contract §15.9: operands observed in the model, not stand-alone samples).

The interception is the one W1's adapter uses (``verity_vllm_adapter.capture.install_kernel_wrappers``):
rebind ``vllm.model_executor.determinism.batch_invariant.mm_batch_invariant`` -- the function the
``aten::linear`` / ``aten::matmul`` / ``aten::mm`` batch-invariant overrides reach by global name --
so every GEMM of every layer (qkv / o / gate_up / down and the lm_head at the sampled positions)
passes through the wrapper with ``a = x [M, K]`` and ``b = weight.t() [K, N]``; ``bias`` is never
passed here (``linear_batch_invariant`` adds it afterwards in BF16, a separate ``aten.add`` node in
W1's description).  For every call we store the BF16 words of ``a`` and of the kernel's output and
the parameter the ``b`` view aliases (weights are stored once, by parameter name).

    PATH=/workspace/venv-nightly/bin:$PATH VLLM_BATCH_INVARIANT=1 python -m verity_vllm_numerics.engine_gemm_capture OUTDIR [--max-tokens 6] [--compiled]

Engine flags are W1's profile (``VLLM_BATCH_INVARIANT=1``, ``VLLM_USE_V2_MODEL_RUNNER=1``,
``VLLM_ENABLE_V1_MULTIPROCESSING=0``) with ``enforce_eager=True`` (the replay configuration) unless
``--compiled`` (the production configuration; under ``torch.compile`` the wrapper is only reached
while Dynamo traces, so it captures nothing useful -- kept for the record).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "In 1905 Albert Einstein published four papers that changed physics. The first explained the photoelectric "
    "effect, the second Brownian motion, the third introduced special relativity and the fourth",
    "Ingredients: 200 g flour, 2 eggs, 300 ml milk, a pinch of salt. Method: whisk the eggs into the milk, then",
    "Q: What is the boiling point of water at sea level in Celsius?\nA:",
    "The following is a list of prime numbers below one hundred: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,",
    "Translate to German: 'The weather is nice today and we are going for a walk in the park.'",
    "SELECT name, COUNT(*) FROM orders JOIN customers ON orders.customer_id = customers.id WHERE",
]


def words(t: torch.Tensor) -> np.ndarray:
    assert t.dtype == torch.bfloat16, t.dtype
    return t.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=6)
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--warmup-same", action="store_true", help="warm up on the recorded prompts (prefix-cached prefill)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "calls").mkdir(parents=True, exist_ok=True)
    (out / "weights").mkdir(exist_ok=True)

    from vllm import LLM, SamplingParams
    import vllm.model_executor.determinism.batch_invariant as bi

    kw = dict(model=args.model, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.35, seed=0)
    if not args.compiled:
        kw["enforce_eager"] = True
    llm = LLM(**kw)

    # parameter identity by storage pointer (b = weight.t() is a view of the parameter)
    def param_table(m):
        return {p.data_ptr(): (n, tuple(p.shape)) for n, p in m.named_parameters()}

    try:
        ptr_to_param = llm.apply_model(param_table)[0]
    except Exception as e:  # noqa: BLE001 -- fall back to the in-process worker's model
        print(f"apply_model failed ({e!r}); walking the in-process worker", file=sys.stderr)
        ex = llm.llm_engine.model_executor
        w = getattr(ex, "driver_worker", None)
        w = getattr(w, "worker", w)
        ptr_to_param = param_table(w.model_runner.model)

    calls: list[dict] = []
    saved_weights: dict[str, dict] = {}
    orig_mm = bi.mm_batch_invariant
    state = {"active": False, "step": 0}

    def mm_wrapper(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        res = orig_mm(a, b, bias)
        if not state["active"] or a.dim() != 2 or a.dtype != torch.bfloat16:
            return res
        torch.cuda.synchronize()
        name, pshape = ptr_to_param.get(b.data_ptr(), (None, None))
        M, K = a.shape
        N = b.shape[1]
        seq = len(calls)
        rec = {
            "seq": seq, "M": int(M), "K": int(K), "N": int(N), "bias": bias is not None,
            "weight": name, "weight_shape": list(pshape) if pshape else None,
            "b_stride": list(b.stride()), "b_is_transposed_view": list(b.stride()) == [1, K] and pshape == (N, K),
            "a_contiguous": a.is_contiguous(), "x_file": f"calls/{seq:05d}_x.npy", "out_file": f"calls/{seq:05d}_out.npy",
        }
        np.save(out / rec["x_file"], words(a))
        np.save(out / rec["out_file"], words(res))
        if bias is not None:
            rec["bias_file"] = f"calls/{seq:05d}_bias.npy"
            np.save(out / rec["bias_file"], words(bias))
        if name is not None and name not in saved_weights:
            wfile = f"weights/{name}.npy"
            np.save(out / wfile, words(b.t()))  # the parameter's own [N, K] layout
            saved_weights[name] = {"file": wfile, "shape": list(pshape)}
        calls.append(rec)
        return res

    bi.mm_batch_invariant = mm_wrapper

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    tok = llm.get_tokenizer()
    prompt_lens = [len(tok(p)["input_ids"]) for p in PROMPTS]
    # warm-up, not recorded (compile / cudagraph capture in --compiled).  A *different* prompt, or
    # the recorded prefill would be served from the prefix cache (only the last partial block of
    # each prompt is computed: 30 of 206 tokens in the first run of this script, kept as
    # cap_prefixcached); ``--warmup-same`` reproduces that variant deliberately.
    llm.generate(PROMPTS if args.warmup_same else ["Hello, world. This is a warm-up prompt."], sp)
    state["active"] = True
    t0 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    torch.cuda.synchronize()
    state["active"] = False
    bi.mm_batch_invariant = orig_mm

    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    meta = {
        "model": args.model,
        "vllm": __import__("vllm").__version__,
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "gpu": torch.cuda.get_device_name(0),
        "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        "compilation": {
            "mode": str(getattr(cc, "mode", None)),
            "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
            "enforce_eager": bool(vc.model_config.enforce_eager),
            "model_runner": "v2" if getattr(vc, "use_v2_model_runner", False) else "v1",
            "custom_ops": list(getattr(cc, "custom_ops", [])),
        },
        "interception": "rebind vllm.model_executor.determinism.batch_invariant.mm_batch_invariant (W1 install_kernel_wrappers style); records a, b(view of parameter), out per call after cuda.synchronize",
        "prompts": PROMPTS,
        "prompt_lens": prompt_lens,
        "max_tokens": args.max_tokens,
        "warmup": "same prompts (prefix-cached prefill)" if args.warmup_same else "distinct prompt (full prefill)",
        "outputs": [o.outputs[0].text for o in outs],
        "output_token_ids": [list(o.outputs[0].token_ids) for o in outs],
        "generate_seconds": time.perf_counter() - t0,
        "n_calls": len(calls),
        "weights": saved_weights,
        "calls": calls,
    }
    (out / "index.json").write_text(json.dumps(meta, indent=1))
    ms = sorted({c["M"] for c in calls})
    print(json.dumps({"n_calls": len(calls), "M_values": ms, "weights": len(saved_weights), "prompt_lens": prompt_lens}))
    unresolved = [c for c in calls if c["weight"] is None]
    if unresolved:
        print(f"WARNING: {len(unresolved)} calls whose b did not alias a parameter", file=sys.stderr)


if __name__ == "__main__":
    main()
