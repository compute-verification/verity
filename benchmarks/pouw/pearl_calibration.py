"""Calibration: Pearl's own vLLM miner (PR #311 head, SM100) on a B200, against plain vLLM on the same card.

    python benchmarks/pouw/pearl_calibration.py --model /root/models/Qwen2.5-7B-Instruct [--out RESULT.json]

Recorded as r20260928-075340-f1bb, after ``pearl_b200_setup.sh``:

    C=/root/pearl-venv/lib/python3.12/site-packages/nvidia/cu13
    research run --on <b200> --project verity --source . --cwd source --campaign pouw --tool pouw_pearl_cal \
      --env PATH=$C/bin:/root/pearl-venv/bin:/usr/bin:/bin --env CUDA_HOME=$C --env HF_HUB_OFFLINE=1 \
      --env ATTENTION_BACKEND=FLASH_ATTN --env VLLM_USE_FLASHINFER_SAMPLER=0 \
      -- python benchmarks/pouw/pearl_calibration.py --model /root/models/Qwen2.5-7B-Instruct

Runs in the Python environment of Pearl's workspace (``pearl_b200_setup.sh``), not Verity's.  Three modes, each in its own
process (vLLM keeps device memory until exit):

- ``bf16``: stock vLLM;
- ``pearl-fallback``: ``quantization="pearl"`` with ``MINER_NO_MINING=true``: the FP10-encoded layers served by Pearl's
  FP8 fallback GEMM, no mining;
- ``pearl-mining``: ``quantization="pearl"`` with ``MINER_NO_GATEWAY=true``: every eligible prefill (at least
  ``PEARL_MIN_MINING_TOKENS`` tokens) runs Pearl's four-op mined pipeline against the plugin's dummy job at hard
  difficulty; decode and CUDA-graph replays serve the fallback, as Pearl ships it.

Measured: prefill tokens/s (4 prompts of 2,048 tokens, one scheduler step; median of 7, since mined steps vary), decode tokens/s (batch 32, 128 tokens), and
greedy-token agreement with ``bf16`` on 16 prompts.  ``ATTENTION_BACKEND`` (e.g. ``FLASH_ATTN``) overrides vLLM's choice
in every mode: FlashInfer's TRT-LLM attention JIT needs a full CUDA 13 toolkit, which a pip-only environment lacks.  The ratio of ``pearl-mining`` to ``pearl-fallback`` prefill is Pearl's
own mining overhead on its own hardware, the calibration point for the RTX 4090 port in ``gemm_bench.py`` and
``vllm_bench.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

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
MODES = {
    "bf16": {},
    "pearl-fallback": {"MINER_NO_MINING": "true"},
    "pearl-mining": {"MINER_NO_GATEWAY": "true", "MINER_SKIP_BLOCK_SUBMISSION": "true"},
}
COMMON_ENV = {"PEARL_M_BUCKETS": "8192", "PEARL_MIN_MINING_TOKENS": "1024", "PEARL_WARMUP_COMPILE": "true"}


def run_mode(model: str, mode: str) -> dict:
    import torch
    from vllm import LLM, SamplingParams

    # Prefix caching would serve the repeated prefill prompts from cache, and cached prefills skip Pearl's mining.
    kw = dict(model=model, gpu_memory_utilization=0.8, max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=8192, seed=0,
              enable_prefix_caching=False)
    if mode != "bf16":
        kw["quantization"] = "pearl"
    if os.environ.get("ATTENTION_BACKEND"):
        kw["attention_config"] = {"backend": os.environ["ATTENTION_BACKEND"]}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t0
    tok = llm.get_tokenizer()
    msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in PROMPTS]
    outs = llm.generate(msgs, SamplingParams(temperature=0.0, max_tokens=48), use_tqdm=False)
    g = torch.Generator().manual_seed(0)
    prefill = [{"prompt_token_ids": torch.randint(100, 20000, (2048,), generator=g).tolist()} for _ in range(4)]
    one = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(prefill[:1], one, use_tqdm=False)
    times = []
    for _ in range(7):
        t0 = time.perf_counter()
        llm.generate(prefill, one, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    decode = [{"prompt_token_ids": torch.randint(100, 20000, (64,), generator=g).tolist()} for _ in range(32)]
    t0 = time.perf_counter()
    llm.generate(decode, SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True), use_tqdm=False)
    dec_s = time.perf_counter() - t0
    return {"mode": mode, "load_s": load_s, "prefill_tok_s": 4 * 2048 / median, "prefill_tok_s_best": 4 * 2048 / min(times),
            "prefill_s": times,
            "decode_b32_tok_s": 32 * 128 / dec_s, "tokens": [list(o.outputs[0].token_ids) for o in outs],
            "texts": [o.outputs[0].text for o in outs[:3]], "device": torch.cuda.get_device_name(),
            "attention_backend": os.environ.get("ATTENTION_BACKEND", "auto")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "pearl_calibration.json")
    ap.add_argument("--single", default=None)
    args = ap.parse_args()
    if args.single:
        args.out.write_text(json.dumps(run_mode(args.model, args.single)))
        return
    results = {}
    for mode, env in MODES.items():
        part = args.out.with_suffix(f".{mode}.json")
        log = args.out.with_suffix(f".{mode}.log")
        with log.open("w") as fh:
            subprocess.run([sys.executable, __file__, "--model", args.model, "--out", str(part), "--single", mode],
                           env={**os.environ, **COMMON_ENV, **env}, stdout=fh, stderr=subprocess.STDOUT, check=False)
        results[mode] = json.loads(part.read_text()) if part.exists() else {"mode": mode, "error": log.read_text()[-2000:]}
        print(json.dumps({k: v for k, v in results[mode].items() if k not in ("tokens", "error")}), flush=True)
    ref = results.get("bf16", {}).get("tokens")

    def agree(t):
        if not ref or not t:
            return None
        same = sum(x == y for s, u in zip(t, ref) for x, y in zip(s, u))
        return same / max(1, sum(min(len(s), len(u)) for s, u in zip(t, ref)))

    summary = {m: {"prefill_tok_s": r.get("prefill_tok_s"), "decode_b32_tok_s": r.get("decode_b32_tok_s"),
                   "token_agreement_vs_bf16": agree(r.get("tokens")), "error": "error" in r} for m, r in results.items()}
    fb, mn = results.get("pearl-fallback", {}), results.get("pearl-mining", {})
    if fb.get("prefill_tok_s") and mn.get("prefill_tok_s"):
        summary["mining_prefill_slowdown_vs_fallback"] = fb["prefill_tok_s"] / mn["prefill_tok_s"]
    args.out.write_text(json.dumps({"schema": "verity-pouw/pearl-calibration/v1", "summary": summary, "results": results}, indent=1))
    print(json.dumps(summary), flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    import run_result
    ok = all(not s.get("error") for k, s in summary.items() if isinstance(s, dict))
    run_result.write(args.out, ok, "pearl_calibration: every mode ran", json.dumps(summary)[:2000],
                     [(f"{m}.{k}", v, "tok/s") for m, s in summary.items() if isinstance(s, dict)
                      for k, v in s.items() if k.endswith("tok_s") and v is not None])


if __name__ == "__main__":
    main()
