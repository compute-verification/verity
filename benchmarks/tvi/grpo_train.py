#!/usr/bin/env python3
"""GRPO on a verifiable math task (GSM8K, MATH-500 or a JSONL), one GPU, with vLLM colocate generation or HF generate.

    python grpo_train.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --steps 200 --adapter lora --lora-r 16 \
        --seed 0 --gen vllm --save-dir /workspace/jobs/tvi/ckpt/NAME [--save-merged] [--out $OUT]

One step is one optimizer update over `--prompts-per-step` prompts x `--num-generations` completions. The reward
(`--reward true`) is 1 for a final answer math-verify (or an exact last-number match) accepts, else 0; `format` and `random`
are the controls (a numeric final \\boxed{}; Bernoulli(1/2)), with correctness still logged. `--out`/metrics.json records the arguments,
versions, trainable parameters and state bits S, steps/s, GPU memory and the per-step log (reward, lengths, KL, ...).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvi_common as C  # noqa: E402

DEFAULT_LR = {"full": 1e-6, "lora": 2e-5, "tinylora": 5e-3}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_adapter_args(p, default_lr=0.0)
    p.add_argument("--gen", choices=["vllm", "hf"], default="vllm")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--prompts-per-step", type=int, default=8)
    p.add_argument("--micro-batch", type=int, default=16, help="completions per forward/backward")
    p.add_argument("--max-completion-length", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    p.add_argument("--vllm-mem", type=float, default=0.3, help="vLLM's share of the GPU in colocate mode")
    p.add_argument("--extra", default="{}", help="JSON of further GRPOConfig fields")
    p.add_argument("--reward", choices=["true", "format", "random"], default="true",
                   help="true: correctness; format: a numeric final \\boxed{}; random: Bernoulli(1/2), a control")
    p.add_argument("--fp32-master", action="store_true",
                   help="hold the weights in fp32 under bf16 autocast, so steps below bf16's ulp aren't rounded away (full)")
    args = p.parse_args()
    if not args.lr:
        args.lr = DEFAULT_LR[args.adapter]
    os.makedirs(args.out, exist_ok=True)
    t0 = C.now()
    if args.gen == "vllm":
        C.single_process_dist_env()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.fp32_master else torch.bfloat16).to("cuda")
    model, peft_cfg, tl = C.apply_adapter(model, args)

    ds = C.load_task(args.dataset, "train", args.train_n or None, args.seed).remove_columns(["completion"])

    total = args.prompts_per_step * args.num_generations
    if total % args.micro_batch:
        raise SystemExit(f"--prompts-per-step x --num-generations ({total}) must be a multiple of --micro-batch")
    want = dict(
        output_dir=os.path.join(args.out, "trainer"),
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=total // args.micro_batch,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        use_vllm=args.gen == "vllm",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0,
    )
    funcs, weights = C.reward_funcs(args.reward, args.seed)
    want["reward_weights"] = weights
    if args.gen == "vllm":
        want.update(vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem,
                    generation_kwargs={"stop_token_ids": C.chat_stop_ids(tok)})
    want.update(json.loads(args.extra))
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(set(want) - fields)
    if dropped:
        print(f"GRPOConfig has no field(s) {dropped}; dropped", flush=True)
        if args.gen == "vllm" and {"use_vllm", "vllm_mode"} & set(dropped):
            raise SystemExit("this TRL can't run vLLM colocate; rerun with --gen hf")
    cfg = GRPOConfig(**{k: v for k, v in want.items() if k in fields})

    if "reward_weights" not in fields:
        raise SystemExit("this TRL has no reward_weights, so the logged-only rewards would be trained on")
    trainer = GRPOTrainer(model=model, reward_funcs=funcs, args=cfg, train_dataset=ds,
                          processing_class=tok, peft_config=peft_cfg)
    if tl is not None and args.gen == "vllm":
        C.patch_vllm_sync(trainer, tl)
    n_train = C.trainable_count(trainer.model)
    print(f"trainable params {n_train} ({n_train * 32} bits fp32)", flush=True)
    t1 = C.now()
    res = trainer.train()
    t2 = C.now()
    saved = C.save_state(trainer, tl, args, tok)

    hist = trainer.state.log_history
    steps = [h for h in hist if "reward" in h or "loss" in h]
    rewards = [h.get("reward") for h in steps if h.get("reward") is not None]
    k = max(1, len(rewards) // 4)
    doc = {
        "script": "grpo_train.py",
        "args": vars(args),
        "versions": C.versions(),
        "grpo_config": {k_: getattr(cfg, k_) for k_ in sorted(want) if k_ in fields},
        "dropped_config": dropped,
        "trainable_params": n_train,
        "state_bits_fp32": n_train * 32,
        "state_bits_bf16": n_train * 16,
        "steps": trainer.state.global_step,
        "train_runtime_s": res.metrics.get("train_runtime", t2 - t1),
        "steps_per_s": trainer.state.global_step / (t2 - t1),
        "setup_s": t1 - t0,
        "reward_first_quarter": sum(rewards[:k]) / k if rewards else None,
        "reward_last_quarter": sum(rewards[-k:]) / k if rewards else None,
        "gpu": C.gpu_memory(),
        "saved": saved,
        "log_history": hist,
    }
    C.write_metrics(os.path.join(args.out, "metrics.json"), doc)
    print(json.dumps({k_: doc[k_] for k_ in ["steps", "steps_per_s", "train_runtime_s", "trainable_params",
                                              "reward_first_quarter", "reward_last_quarter", "gpu"]}, indent=1))


if __name__ == "__main__":
    main()
