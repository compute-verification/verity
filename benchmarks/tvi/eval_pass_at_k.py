#!/usr/bin/env python3
"""pass@k with vLLM on GSM8K's test split, MATH-500 or a JSONL, for a base model, a merged checkpoint, or base + LoRA.

    python eval_pass_at_k.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --n-problems 500 --samples 64 \
        --k 1,8,64 [--lora DIR] [--temperature 1.0] [--out $OUT]

pass@k is the unbiased estimator 1 - C(n-c, k)/C(n, k) over `--samples` n per problem; greedy accuracy is reported too
when `--greedy` is given. Writes `--out`/eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvi_common as C  # noqa: E402


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--lora", default=None, help="a PEFT LoRA adapter dir on top of --model")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--n-problems", type=int, default=500)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--k", default="1,4,16")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--out", default=os.environ.get("OUT", "."))
    args = p.parse_args()
    ks = [int(x) for x in args.k.split(",")]
    os.makedirs(args.out, exist_ok=True)

    from vllm import LLM, SamplingParams

    ds = C.load_task(args.dataset, args.split, args.n_problems, args.seed)
    kw = {"enable_lora": True, "max_lora_rank": 256} if args.lora else {}
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem, seed=args.seed,
              max_model_len=2048, **kw)
    lora_req = None
    if args.lora:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("a", 1, args.lora)
    msgs = [r["prompt"] for r in ds]
    gold = [r["answer"] for r in ds]
    t0 = C.now()
    res = {"args": vars(args), "versions": C.versions(), "n_problems": len(ds)}
    stop = C.chat_stop_ids(llm.get_tokenizer())
    res["stop_token_ids"] = stop
    sp = SamplingParams(n=args.samples, temperature=args.temperature, max_tokens=args.max_tokens, seed=args.seed,
                        stop_token_ids=stop)
    outs = llm.chat(msgs, sp, lora_request=lora_req)
    correct = [sum(C.is_correct(o.text, g) for o in out.outputs) for out, g in zip(outs, gold)]
    for k in ks:
        if k <= args.samples:
            res[f"pass@{k}"] = sum(pass_at_k(args.samples, c, k) for c in correct) / len(correct)
    res["mean_correct_frac"] = sum(correct) / (len(correct) * args.samples)
    res["mean_len_tokens"] = sum(len(o.token_ids) for out in outs for o in out.outputs) / (len(outs) * args.samples)
    res["mean_formatted_frac"] = sum(C.is_formatted(o.text) for out in outs for o in out.outputs) / (len(outs) * args.samples)
    if args.greedy:
        g_outs = llm.chat(msgs, SamplingParams(n=1, temperature=0.0, max_tokens=args.max_tokens, stop_token_ids=stop),
                          lora_request=lora_req)
        res["per_problem_greedy"] = [int(C.is_correct(o.outputs[0].text, g)) for o, g in zip(g_outs, gold)]
        res["per_problem_greedy_formatted"] = [int(C.is_formatted(o.outputs[0].text)) for o in g_outs]
        res["greedy_acc"] = sum(res["per_problem_greedy"]) / len(gold)
        res["greedy_formatted"] = sum(res["per_problem_greedy_formatted"]) / len(gold)
        res["greedy_len_tokens"] = sum(len(o.outputs[0].token_ids) for o in g_outs) / len(gold)
    res["eval_s"] = C.now() - t0
    res["per_problem_correct"] = correct
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("per_problem") and k != "args"}, indent=1))


if __name__ == "__main__":
    main()
