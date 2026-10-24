"""Shared pieces of the training-vs-inference templates: datasets, the GSM8K/MATH reward, adapters by state size, metrics.

Adapters (`--adapter`):

- `full`: every parameter trains.
- `lora`: PEFT LoRA of rank `--lora-r` on `--lora-targets` (default all seven linear maps of a Qwen2/Llama block).
- `tinylora`: TinyLoRA (LoRA-XS on the frozen weights' top-`--tiny-rank` SVD, with the r x r core a combination of `--tiny-u`
  fixed random matrices drawn from `--tiny-seed`), its `u` coefficients tied across every target map (`--tiny-groups 1`) or
  split into that many groups by layer. Trainable parameters = u * groups, so the state is exactly that many fp32 numbers.

Every run reports `trainable_params` and `state_bits_fp32` / `state_bits_bf16`, the size S of the unregistered state.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time

import torch

SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def single_process_dist_env() -> None:
    """vLLM's colocate mode needs a torch.distributed world even on one GPU; a free port keeps concurrent jobs apart."""
    if "MASTER_PORT" not in os.environ:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
    for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"}.items():
        os.environ.setdefault(k, v)


def gsm8k_gold(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_trace(answer: str) -> str:
    """GSM8K's reference solution as a reasoning trace ending in \\boxed{}, calculator annotations removed."""
    body, gold = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\nThe final answer is \\boxed{{{gold.strip().replace(',', '')}}}."


def load_task(name: str, split: str, n: int | None = None, seed: int = 0):
    """Rows with `prompt` (chat messages), `answer` (gold string) and, for SFT, `completion`.

    `name` is `gsm8k`, `math500`, or a JSONL path whose rows have `question` and `answer` (and optionally `solution`).
    """
    from datasets import Dataset, load_dataset

    if name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=split)
        rows = [{"question": r["question"], "answer": gsm8k_gold(r["answer"]), "solution": gsm8k_trace(r["answer"])} for r in ds]
    elif name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [{"question": r["problem"], "answer": r["answer"], "solution": r["solution"]} for r in ds]
    else:
        with open(name) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [{"question": r["question"], "answer": str(r["answer"]), "solution": r.get("solution", "")} for r in rows]
    ds = Dataset.from_list(rows).shuffle(seed=seed)
    if n:
        ds = ds.select(range(min(n, len(ds))))

    def fmt(r):
        prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["question"]}]
        return {"prompt": prompt, "completion": [{"role": "assistant", "content": r["solution"]}]}

    return ds.map(fmt)


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    depth, j = 0, i + len("\\boxed{") - 1
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1 : k]
    return None


def is_correct(completion: str, gold: str) -> bool:
    """math-verify on the last \\boxed{} (or the whole text), with an exact last-number check as a fallback."""
    from math_verify import parse, verify

    boxed = extract_boxed(completion)
    try:
        g = parse("$" + gold + "$")
        p = parse("$" + boxed + "$") if boxed is not None else parse(completion)
        if g and p and verify(g, p):
            return True
    except Exception:
        pass
    cand = boxed if boxed is not None else completion
    nums = _NUM.findall(cand)
    if not nums:
        return False
    try:
        return abs(float(nums[-1].replace(",", "")) - float(gold.replace(",", ""))) < 1e-6
    except ValueError:
        return nums[-1].strip() == gold.strip()


def completion_text(c) -> str:
    return c[-1]["content"] if isinstance(c, list) else c


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [1.0 if is_correct(completion_text(c), a) else 0.0 for c, a in zip(completions, answer)]


def is_formatted(completion: str) -> bool:
    """A final \\boxed{} holding a number: the answer format, whatever the number is."""
    boxed = extract_boxed(completion)
    return boxed is not None and _NUM.search(boxed) is not None


def format_reward(completions, **kwargs) -> list[float]:
    return [1.0 if is_formatted(completion_text(c)) else 0.0 for c in completions]


def make_random_reward(seed: int):
    """Bernoulli(1/2) per completion from its own seeded stream, independent of the completion and of correctness."""
    gen = torch.Generator().manual_seed(10_000 + seed)

    def random_reward(completions, **kwargs) -> list[float]:
        return torch.bernoulli(torch.full((len(completions),), 0.5), generator=gen).tolist()

    return random_reward


def reward_funcs(name: str, seed: int):
    """(reward functions, weights) for `--reward`: the trained-on reward has weight 1, and correctness and format are
    always logged beside it at weight 0."""
    if name == "true":
        return [correctness_reward, format_reward], [1.0, 0.0]
    if name == "format":
        return [format_reward, correctness_reward], [1.0, 0.0]
    if name == "random":
        return [make_random_reward(seed), correctness_reward, format_reward], [1.0, 0.0, 0.0]
    raise SystemExit(f"unknown --reward {name}: true, format or random")


class TinyLoRA(torch.nn.Module):
    """W' = W + U_r S_r (sum_i v_i P_i) V_r^T on each target map, as a forward hook so the module tree (and the names
    vLLM loads) stay the base model's. v (u per group) is the only trainable tensor."""

    def __init__(self, model, targets, rank: int, u: int, groups: int, seed: int):
        super().__init__()
        self.groups, self.u, self.rank = groups, u, rank
        self.v = torch.nn.Parameter(torch.zeros(groups, u, dtype=torch.float32))
        gen = torch.Generator().manual_seed(seed)
        self.P = torch.nn.Parameter(torch.randn(u, rank, rank, generator=gen) / rank, requires_grad=False)
        self.mods = []
        layers = sorted({int(m.group(1)) for n, _ in model.named_modules() if (m := re.search(r"layers\.(\d+)\.", n))})
        nl = max(len(layers), 1)
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear) or name.split(".")[-1] not in targets:
                continue
            m = re.search(r"layers\.(\d+)\.", name)
            g = (int(m.group(1)) * groups // nl) if m else 0
            W = mod.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            mod.register_buffer("_tl_B", (U[:, :rank] * S[:rank]).to(mod.weight.dtype), persistent=False)
            mod.register_buffer("_tl_A", Vh[:rank].contiguous().to(mod.weight.dtype), persistent=False)
            mod._tl_group = g
            mod.register_forward_hook(self._hook)
            self.mods.append((name, mod))

    def core(self, g: int):
        return torch.einsum("u,urs->rs", self.v[g], self.P.to(self.v.device))

    def _hook(self, mod, inp, out):
        x = inp[0]
        M = self.core(mod._tl_group).to(x.dtype)
        return out + ((x @ mod._tl_A.T) @ M.T) @ mod._tl_B.T

    def delta(self, mod):
        return (mod._tl_B.float() @ self.core(mod._tl_group).float() @ mod._tl_A.float()).to(mod.weight.dtype)

    def merged_state(self):
        """(name, merged weight) for every target map, to push into vLLM or save."""
        with torch.no_grad():
            return [(f"{n}.weight", (mod.weight.data.float() + self.delta(mod).float()).to(mod.weight.dtype)) for n, mod in self.mods]


def patch_vllm_sync(trainer, tl: TinyLoRA) -> None:
    """TRL's colocate sync pushes `_iter_named_params()` into vLLM; under TinyLoRA only the target maps change, so push
    their merged weights (W + delta) and never the `tinylora.*` tensors vLLM has no slot for."""
    vg = trainer.vllm_generation

    def merged():
        for name, w in tl.merged_state():
            yield vg._fix_param_name_to_vllm(name), w

    vg._iter_named_params = merged


def apply_adapter(model, args):
    """Returns (model, peft_config or None, tinylora or None)."""
    if args.adapter == "full":
        return model, None, None
    if args.adapter == "lora":
        from peft import LoraConfig

        cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha or 2 * args.lora_r, lora_dropout=0.0,
                         target_modules=args.lora_targets.split(","), task_type="CAUSAL_LM")
        return model, cfg, None
    if args.adapter == "tinylora":
        for p in model.parameters():
            p.requires_grad_(False)
        tl = TinyLoRA(model, args.lora_targets.split(","), args.tiny_rank, args.tiny_u, args.tiny_groups, args.tiny_seed)
        tl.to(next(model.parameters()).device)
        model.tinylora = tl
        return model, None, tl
    raise SystemExit(f"unknown --adapter {args.adapter}: full, lora or tinylora")


def add_adapter_args(p, default_lr: float) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="gsm8k", help="gsm8k, math500 or a JSONL of {question, answer[, solution]}")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", default=None, help="where the trained state goes (outside the run dir); none: not saved")
    p.add_argument("--save-merged", action="store_true", help="also save the merged full model (for vLLM eval, the analyzer)")
    p.add_argument("--out", default=os.environ.get("OUT", "."), help="metrics.json and logs (default $OUT)")
    p.add_argument("--adapter", default="lora", choices=["full", "lora", "tinylora"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=0, help="0: 2r")
    p.add_argument("--lora-targets", default=",".join(QWEN_TARGETS))
    p.add_argument("--tiny-rank", type=int, default=2)
    p.add_argument("--tiny-u", type=int, default=13)
    p.add_argument("--tiny-groups", type=int, default=1)
    p.add_argument("--tiny-seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=default_lr)
    p.add_argument("--train-n", type=int, default=0, help="train rows (0: all)")
    p.add_argument("--gradient-checkpointing", action="store_true")


def trainable_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "device_used_gib_at_end": (total - free) / 2**30,
        "device_total_gib": total / 2**30,
        "device_name": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
    }


def versions() -> dict:
    import importlib

    out = {}
    for m in ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm", "math_verify"]:
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception as e:
            out[m] = f"unavailable: {type(e).__name__}"
    return out


def save_state(trainer, tl, args, tokenizer) -> dict:
    if not args.save_dir:
        return {}
    os.makedirs(args.save_dir, exist_ok=True)
    model = trainer.model
    saved = {"save_dir": args.save_dir}
    if tl is not None:
        torch.save({"v": tl.v.detach().cpu(), "P_seed": args.tiny_seed, "rank": args.tiny_rank, "u": args.tiny_u,
                    "groups": args.tiny_groups, "targets": args.lora_targets, "base": args.model}, os.path.join(args.save_dir, "tinylora.pt"))
    else:
        trainer.save_model(args.save_dir)
    if args.save_merged:
        md = os.path.join(args.save_dir, "merged")
        if tl is not None:
            for n, w in tl.merged_state():
                model.get_submodule(n[: -len(".weight")]).weight.data.copy_(w)
            for _, mod in tl.mods:
                mod._forward_hooks.clear()
            del model.tinylora
            model.save_pretrained(md)
        elif hasattr(model, "merge_and_unload"):
            model.merge_and_unload().save_pretrained(md)
        else:
            model.save_pretrained(md)
        tokenizer.save_pretrained(md)
        saved["merged"] = md
    return saved


def write_metrics(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(f"metrics -> {path}")


def now() -> float:
    return time.perf_counter()
