"""A stock decode step of the int8 Llama-3.2-1B on vLLM at B rows: the denominator of decode's T2.

    python stock_decode.py [--rows 1 16] [--steps 128] [--reps 5] [--async] [--json FILE]

vLLM's offline engine on verity-int8/Llama-3.2-1B-INT8 at the revision benchmarks/one_stage/served_util.sh serves, its
linears on serve-openai's default precision (verity_int8: verity_vllm's int8 linears, registered as serve-openai registers
them), eager, no prefix cache, synchronous scheduling (--async: asynchronous). B requests of 16 prompt ids decode
STEPS + 1 tokens each, greedy, ignoring EOS; a step is (t(STEPS + 1 tokens) - t(1 token)) / STEPS, the median over REPS,
so it is the whole decode step at B rows, host included. Not serve-openai: no HTTP, no bench row, no taps.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

REPO = "verity-int8/Llama-3.2-1B-INT8"
REV = "15b0976d400d796ee589e7737d03a919526bf800"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--async", dest="async_", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    from verity_vllm.engine.int8_ranks import register

    register(["linears"])
    llm = LLM(model=REPO, revision=REV, quantization="verity_int8", enforce_eager=True, enable_prefix_caching=False,
              async_scheduling=a.async_, max_num_seqs=max(a.rows), gpu_memory_utilization=0.5, seed=0,
              worker_extension_cls="verity_vllm.engine.int8_ranks.Int8Ranks")
    res = []
    for b in a.rows:
        prompts = [TokensPrompt(prompt_token_ids=[1 + (1000 + 37 * i + 11 * k) % 32000 for k in range(16)])
                   for i in range(b)]

        def run(n: int) -> float:
            sp = SamplingParams(max_tokens=n, temperature=0.0, ignore_eos=True)
            t = time.perf_counter()
            out = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t
            if any(len(o.outputs[0].token_ids) != n for o in out):
                raise SystemExit(f"rows {b}: a request decoded other than {n} tokens")
            return dt

        run(a.steps + 1), run(1)
        long, short = [], []
        for _ in range(a.reps):
            long.append(run(a.steps + 1))
            short.append(run(1))
        step = [(x - y) / a.steps for x, y in zip(long, short)]
        r = {"rows": b, "steps": a.steps, "reps": a.reps, "async": a.async_, "step_ms": 1e3 * statistics.median(step),
             "step_ms_min": 1e3 * min(step), "step_ms_max": 1e3 * max(step),
             "one_token_ms": 1e3 * statistics.median(short)}
        print(json.dumps(r), flush=True)
        res.append(r)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": REPO, "revision": REV, "eager": True, "results": res}, f, indent=1)


if __name__ == "__main__":
    main()
