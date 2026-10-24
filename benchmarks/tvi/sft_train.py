#!/usr/bin/env python3
"""SFT on reasoning traces (GSM8K's reference solutions ending in \\boxed{}, MATH-500's, or a JSONL's `solution`), one GPU.

    python sft_train.py --model Qwen/Qwen2.5-1.5B-Instruct --dataset gsm8k --steps 200 --adapter lora --lora-r 16 \
        --seed 0 --save-dir /workspace/jobs/tvi/ckpt/NAME [--save-merged] [--out $OUT]

Prompt-completion chat format with the loss on the completion only. `--out`/metrics.json records the arguments, versions,
trainable parameters and state bits S, steps/s, GPU memory and the per-step log.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvi_common as C  # noqa: E402

DEFAULT_LR = {"full": 1e-5, "lora": 1e-4, "tinylora": 2e-4}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C.add_adapter_args(p, default_lr=0.0)
    p.add_argument("--batch", type=int, default=16, help="sequences per optimizer step")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--extra", default="{}", help="JSON of further SFTConfig fields")
    args = p.parse_args()
    if not args.lr:
        args.lr = DEFAULT_LR[args.adapter]
    os.makedirs(args.out, exist_ok=True)
    t0 = C.now()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model, peft_cfg, tl = C.apply_adapter(model, args)
    ds = C.load_task(args.dataset, "train", args.train_n or None, args.seed).select_columns(["prompt", "completion"])

    if args.batch % args.micro_batch:
        raise SystemExit("--batch must be a multiple of --micro-batch")
    want = dict(
        output_dir=os.path.join(args.out, "trainer"),
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.batch // args.micro_batch,
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    want.update(json.loads(args.extra))
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(want) - fields)
    if dropped:
        print(f"SFTConfig has no field(s) {dropped}; dropped", flush=True)
    cfg = SFTConfig(**{k: v for k, v in want.items() if k in fields})
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    n_train = C.trainable_count(trainer.model)
    print(f"trainable params {n_train} ({n_train * 32} bits fp32)", flush=True)
    t1 = C.now()
    res = trainer.train()
    t2 = C.now()
    saved = C.save_state(trainer, tl, args, tok)
    hist = trainer.state.log_history
    losses = [h["loss"] for h in hist if "loss" in h]
    doc = {
        "script": "sft_train.py",
        "args": vars(args),
        "versions": C.versions(),
        "sft_config": {k: getattr(cfg, k) for k in sorted(want) if k in fields},
        "dropped_config": dropped,
        "trainable_params": n_train,
        "state_bits_fp32": n_train * 32,
        "state_bits_bf16": n_train * 16,
        "steps": trainer.state.global_step,
        "train_runtime_s": res.metrics.get("train_runtime", t2 - t1),
        "steps_per_s": trainer.state.global_step / (t2 - t1),
        "setup_s": t1 - t0,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gpu": C.gpu_memory(),
        "saved": saved,
        "log_history": hist,
    }
    C.write_metrics(os.path.join(args.out, "metrics.json"), doc)
    print(json.dumps({k: doc[k] for k in ["steps", "steps_per_s", "train_runtime_s", "trainable_params", "loss_first",
                                          "loss_last", "gpu"]}, indent=1))


if __name__ == "__main__":
    main()
