"""vLLM generation as a load on the responder's GPU: one engine, batches of 256 requests decoded with continuous batching,
until the stop file exists. Usage: python vllm_load.py MODEL READY STOP REPORT (run with the node's vLLM venv)."""

from __future__ import annotations

import json
import os
import sys
import time


def main() -> None:
    model, ready, stop, report = sys.argv[1:5]
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, gpu_memory_utilization=float(os.environ.get("LAT_VLLM_UTIL", "0.7")), max_model_len=4096, seed=0)
    words = "the auditor sends a challenge and the prover answers from the memory it claims to hold".split()
    prompts = [" ".join(words[(i * 7 + j) % len(words)] for j in range(64 + (i % 8) * 64)) for i in range(256)]
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256, ignore_eos=True, seed=0)
    with open(ready, "w") as f:
        f.write("ready\n")
    with open(report, "a") as rep:
        while not os.path.exists(stop):
            t = time.time()
            outs = llm.generate(prompts, params, use_tqdm=False)
            tokens = sum(len(o.outputs[0].token_ids) for o in outs)
            rep.write(json.dumps({"t": t, "s": time.time() - t, "generated_tokens": tokens, "requests": len(prompts)}) + "\n")
            rep.flush()


if __name__ == "__main__":
    main()
