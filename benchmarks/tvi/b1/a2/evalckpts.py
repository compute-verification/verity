"""Evaluates saved GRPO checkpoints (and the base model) in one vLLM process: greedy accuracy, and sampled accuracy (mean over
n samples at temperature T, the quantity GRPO maximizes), per problem, for the bootstrap in analyze.py.

`python evalckpts.py --root CKPT_ROOT --glob '*/step150' --n 4 --temperature 1.0 --out DIR [--base gsm8k+math] [--passk 0]
[--unmerged lora16] [--expect N --wait-until EPOCH]`; with --expect it rescans each minute for checkpoints still training.
Each checkpoint's effective bf16 weights are rebuilt from the registered model (every parameter re-written first), then from
its full state dict or its LoRA factors (W0 + B A), as grpo.load_ckpt does.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import grpo
import torch


def data_task(task):
    return "gsm8k" if task.startswith("gsm8k") else "math"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--glob", default="*/step*")
    ap.add_argument("--base", default="", help="comma-separated tasks to also evaluate the base model on")
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--variant", default="")
    ap.add_argument("--data", default=os.environ.get("DATA", "data"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--passk", type=int, default=0, help="also n samples at T=0.6 for pass@k")
    ap.add_argument("--passk-limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-done", type=int, default=1)
    ap.add_argument("--prior", default="", help="other runs' out dirs (joined by |) whose ckpteval files count as done")
    ap.add_argument("--unmerged", default="", help="methods (joined by +) whose adapters are served by vLLM's LoRA path "
                    "(W0 x + B A x in bf16, as in training) instead of merged into bf16 weights")
    ap.add_argument("--expect", type=int, default=0, help="keep rescanning until this many checkpoints have been evaluated")
    ap.add_argument("--wait-until", type=float, default=0, help="epoch seconds: stop rescanning for expected checkpoints")
    args = ap.parse_args()
    unmerged = set(args.unmerged.replace("+", ",").split(",")) - {""}

    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    priors = [Path(p) for p in args.prior.split("|") if p]
    def scan():
        ckpts = []
        for d in sorted({d for g in args.glob.split("|") for d in glob.glob(str(Path(args.root) / g))}):
            if not (Path(d) / "DONE").exists():
                continue
            meta = json.load(open(Path(d) / "meta.json"))
            if meta["model"] != args.model:
                continue
            ckpts.append((d, meta))
        return ckpts

    bases = [(None, {"task": t + args.variant, "method": "base", "seed": 0, "step": 0, "S_bits": 0, "n_params": 0})
             for t in args.base.replace("+", ",").split(",") if t]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=2560, enable_prefix_caching=True,
              max_num_seqs=1024, seed=0, **({"enable_lora": True, "max_lora_rank": 16, "max_loras": 1} if unmerged else {}))
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
    base_sd = [(n, p.detach()) for n, p in base.named_parameters()]
    linears = dict(grpo.target_linears(base))
    tests = {t: grpo.load_jsonl(Path(args.data) / f"{t}_test.jsonl") for t in ("gsm8k", "math")
             if (Path(args.data) / f"{t}_test.jsonl").exists()}
    lora_id = 0

    def run(d, meta):
        nonlocal lora_id
        label = f"{meta['task']}-{meta['method']}-seed{meta['seed']}-step{meta['step']}"
        if d is not None and meta["method"] in unmerged:
            label += "-unmerged"
        dst = out / f"ckpteval-{label}.json"
        if args.skip_done and any((p / dst.name).exists() for p in [out] + priors):
            return
        t0 = time.time()
        grpo.sync_weights(llm, base_sd)
        lreq, mode = None, "merged"
        if d is not None and meta["method"] in unmerged:
            from vllm.lora.request import LoRARequest
            lora_id += 1
            lreq, mode = LoRARequest(label, lora_id, str(d)), "vllm-lora"
        elif d is not None:
            if (Path(d) / "model.safetensors").exists():
                grpo.sync_weights(llm, load_file(str(Path(d) / "model.safetensors"), device="cuda").items())
            else:
                lora = load_file(str(Path(d) / "adapter_model.safetensors"), device="cuda")
                pairs = []
                for n, mod in linears.items():
                    if f"base_model.model.{n}.lora_A.weight" not in lora:
                        continue
                    A, B = lora[f"base_model.model.{n}.lora_A.weight"], lora[f"base_model.model.{n}.lora_B.weight"]
                    pairs.append((n + ".weight", (mod.weight.detach().float() + B.float() @ A.float()).to(torch.bfloat16)))
                grpo.sync_weights(llm, pairs)
        dt = data_task(meta["task"])
        rows = tests[dt]
        per_g, _ = grpo.evaluate(llm, tok, rows, dt, 1024 if dt == "gsm8k" else 1536, lora=lreq)
        per_s, lens = grpo.evaluate(llm, tok, rows, dt, 1024 if dt == "gsm8k" else 1536, n=args.n,
                                    temperature=args.temperature, top_p=1.0, lora=lreq)
        res = {"label": label, "task": meta["task"], "method": meta["method"], "seed": meta["seed"], "step": meta["step"],
               "S_bits": meta["S_bits"], "n_params": meta.get("n_params", 0), "ckpt": d, "model": args.model, "mode": mode,
               "greedy_acc": sum(c[0] for c in per_g) / len(per_g), "greedy_per_problem": [c[0] for c in per_g],
               "sampled": {"n": args.n, "temperature": args.temperature, "top_p": 1.0,
                           "acc": sum(sum(c) / len(c) for c in per_s) / len(per_s),
                           "per_problem": [sum(c) / len(c) for c in per_s], "mean_len": sum(lens) / len(lens)}}
        if args.passk:
            pr = rows[:args.passk_limit] if args.passk_limit else rows
            per_k, _ = grpo.evaluate(llm, tok, pr, dt, 1024 if dt == "gsm8k" else 1536, n=args.passk, temperature=0.6,
                                     top_p=0.95, lora=lreq)
            counts = [int(sum(c)) for c in per_k]
            ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.passk]
            res["passk"] = {"n": args.passk, "temperature": 0.6, "top_p": 0.95, "problems": len(pr), "counts": counts,
                            "pass_at": {k: sum(grpo.pass_at_k(args.passk, c, k) for c in counts) / len(counts) for k in ks}}
        res["secs"] = time.time() - t0
        json.dump(res, open(dst, "w"))
        print(json.dumps({"label": label, "greedy": res["greedy_acc"], "sampled": res["sampled"]["acc"],
                          "secs": round(res["secs"], 1)}), flush=True)

    def run_logged(d, meta):
        try:
            run(d, meta)
        except Exception as e:
            print(json.dumps({"ckpt": d, "method": meta["method"], "error": repr(e)[:500]}), flush=True)

    for d, meta in bases:
        run_logged(d, meta)
    done = set()
    while True:
        pending = [j for j in scan() if j[0] not in done]
        for d, meta in pending:
            done.add(d)
            run_logged(d, meta)
        if len(done) >= args.expect or time.time() > args.wait_until:
            break
        if not pending:
            time.sleep(60)


if __name__ == "__main__":
    main()
