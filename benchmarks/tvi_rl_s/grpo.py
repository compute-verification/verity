"""GRPO on a verifiable math task with the trainable state held to a size S, for the training-vs-inference program (crux 1).

One process per run: vLLM (in-process, colocated) samples, a Hugging Face copy of the model trains, and after every step the
effective weights are written into vLLM. Methods (S counts the trainable values at the precision the forward reads them,
bf16, plus 64 bits for a seed when one picks random matrices):

  tiny13   TinyLoRA-like: dW = U_r S_r R V_r^T per linear map (top-r SVD of the registered weight, r=4), R = sum_i v_i P_i with
           P_i seeded random r x r, one 13-vector v shared by all 196 maps.                       13 values, 272 bits
  kbit10   the same, r=5, a 22-vector per layer (28 groups).                                     616 values, ~10 kbit
  kB10     LoRA-XS: R trainable r x r per map, r=5.                                              4,900 values, ~10 kB
  kB100    LoRA-XS, r=16.                                                                        50,176 values, ~100 kB
  lora16   LoRA rank 16 on all seven linear maps of every layer, alpha 32.                       18.5M values
  full     every parameter (fp32 master, bf16 autocast forward).                                 1.54B values
  base     no training: evaluate the registered model (S = 0).

`python grpo.py --task gsm8k --method tiny13 --seed 1 --steps 60 --out DIR --ckpt-root DIR`
`python grpo.py --task gsm8k --method base --passk 64 --out DIR`  (pass@k of the base model on the eval split)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
STOP_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|> (the base model's generation config stops only at the second)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

METHODS = {
    # name: (kind, r, u, grouping, lr)
    "tiny13": ("tiny", 4, 13, "all", 3e-3),
    "kbit10": ("tiny", 5, 22, "layer", 3e-3),
    "kB10": ("xs", 5, None, "module", 2e-3),
    "kB100": ("xs", 16, None, "module", 1e-3),
    "lora16": ("lora", 16, None, "module", 2e-5),
    "full": ("full", None, None, None, 2e-6),
    "base": ("base", None, None, None, 0.0),
}


# ----------------------------------------------------------------------------------------------------------------- scoring
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def to_float(x):
    try:
        return float(x.replace(",", "").rstrip("."))
    except Exception:
        return None


def gsm8k_correct(text, gold):
    b = last_boxed(text)
    src = b if b is not None else text
    nums = NUM.findall(src.replace("\\!", "").replace("{,}", ""))
    if not nums:
        return 0.0
    p, g = to_float(nums[-1]), to_float(gold)
    if p is None or g is None:
        return 0.0
    return float(abs(p - g) <= 1e-6 * max(1.0, abs(g)))


def _strip(s):
    s = s.strip().strip("$")
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("\\ ", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", ""), (" ", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{\s*([^}]*)\}", r"\1", s)
    if s.endswith("."):
        s = s[:-1]
    return s


try:
    from math_verify import parse as mv_parse, verify as mv_verify
except Exception:  # pragma: no cover
    mv_parse = mv_verify = None


def math_correct(text, gold):
    b = last_boxed(text)
    if b is None:
        return 0.0
    if _strip(b) == _strip(gold):
        return 1.0
    if mv_verify is not None:
        try:
            return float(bool(mv_verify(mv_parse("$" + gold + "$"), mv_parse("$" + b + "$"))))
        except Exception:
            return 0.0
    return 0.0


SCORERS = {"gsm8k": gsm8k_correct, "math": math_correct}


def load_jsonl(p):
    return [json.loads(line) for line in open(p)]


# ----------------------------------------------------------------------------------------------------------------- adapters
def target_linears(model):
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in TARGETS and ".layers." in name:
            out.append((name, mod))
    return out


def layer_index(name):
    return int(name.split(".layers.")[1].split(".")[0])


def svd_cache(model, linears, r_max, path):
    """Top-r_max SVD factors of every target weight (a function of the registered weights only). Cached across runs."""
    path = Path(path)
    if path.exists():
        try:
            d = torch.load(path, map_location="cpu")
            if d.get("r", 0) >= r_max and set(d["f"]) >= {n for n, _ in linears}:
                return d["f"]
        except Exception:
            pass
    f = {}
    for name, mod in linears:
        W = mod.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        f[name] = (U[:, :r_max].contiguous().cpu(), S[:r_max].contiguous().cpu(), Vh[:r_max].contiguous().cpu())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save({"r": r_max, "f": f}, tmp)
    os.replace(tmp, path)
    return f


class Adapter(torch.nn.Module):
    """Holds the trainable state and adds dW x to each target linear's output through forward hooks."""

    def __init__(self, model, method, seed, svd_path):
        super().__init__()
        kind, r, u, grouping, _ = METHODS[method]
        self.kind, self.r, self.u, self.grouping, self.seed = kind, r, u, grouping, seed
        self.linears = target_linears(model)
        self.names = [n for n, _ in self.linears]
        dev = self.linears[0][1].weight.device
        self.hooks = []
        self.buf = {}
        if kind in ("tiny", "xs"):
            f = svd_cache(model, self.linears, 16, svd_path)
            for i, (name, mod) in enumerate(self.linears):
                U, S, Vh = f[name]
                self.buf[name] = (U[:, :r].to(dev), S[:r].to(dev), Vh[:r].to(dev))
            if kind == "tiny":
                groups = {"all": lambda n: 0, "layer": layer_index}[grouping]
                self.group_of = {n: groups(n) for n in self.names}
                ng = len(set(self.group_of.values()))
                self.v = torch.nn.Parameter(torch.zeros(ng, u, device=dev))
                g = torch.Generator(device="cpu").manual_seed(seed)
                self.P = {n: (torch.randn(u, r, r, generator=g) / r).to(dev) for n in self.names}
            else:
                self.R = torch.nn.ParameterDict({n.replace(".", "_"): torch.nn.Parameter(torch.zeros(r, r, device=dev))
                                                 for n in self.names})
        elif kind == "lora":
            self.scaling = 32.0 / r
            g = torch.Generator(device="cpu").manual_seed(seed)
            self.A = torch.nn.ParameterDict()
            self.B = torch.nn.ParameterDict()
            for name, mod in self.linears:
                a = torch.empty(r, mod.in_features)
                bound = 1.0 / math.sqrt(mod.in_features)
                a.uniform_(-bound, bound, generator=g)
                self.A[name.replace(".", "_")] = torch.nn.Parameter(a.to(dev))
                self.B[name.replace(".", "_")] = torch.nn.Parameter(torch.zeros(mod.out_features, r, device=dev))
        for name, mod in self.linears:
            self.hooks.append(mod.register_forward_hook(self._hook(name)))

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters())

    def factors(self, name, dtype=torch.bfloat16):
        """(Bm, Am) with dW = Bm @ Am; the coefficients are read at bf16, the precision S counts."""
        k = name.replace(".", "_")
        if self.kind in ("tiny", "xs"):
            U, S, Vh = self.buf[name]
            if self.kind == "tiny":
                v = self.v[self.group_of[name]].to(torch.bfloat16).float()
                R = torch.einsum("u,urs->rs", v, self.P[name])
            else:
                R = self.R[k].to(torch.bfloat16).float()
            return ((U * S) @ R).to(dtype), Vh.to(dtype)
        return (self.B[k].to(torch.bfloat16).float() * self.scaling).to(dtype), self.A[k].to(torch.bfloat16).to(dtype)

    def _hook(self, name):
        def hook(mod, inp, out):
            Bm, Am = self.factors(name, out.dtype)
            return out + (inp[0].to(out.dtype) @ Am.t()) @ Bm.t()
        return hook

    @torch.no_grad()
    def merged(self, name, W0):
        Bm, Am = self.factors(name, torch.float32)
        return W0.float() + Bm @ Am

    def remove(self):
        for h in self.hooks:
            h.remove()


# ----------------------------------------------------------------------------------------------------------------- engine
def sync_weights(llm, named):
    """Write (hf_name, tensor) pairs into the colocated vLLM model and drop its prefix cache."""
    named = list(named)

    def load(model):
        model.load_weights(named)
        return None

    try:
        llm.apply_model(load)
    except Exception:
        def load_w(worker):
            worker.model_runner.model.load_weights(named)
        llm.collective_rpc(load_w)
    llm.reset_prefix_cache()


def load_ckpt(llm, ckpt, model_name):
    """Write a saved checkpoint's effective bf16 weights into vLLM: a full state dict, or W0 + B A from the LoRA factors."""
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    d = Path(ckpt)
    if (d / "model.safetensors").exists():
        sync_weights(llm, load_file(str(d / "model.safetensors"), device="cuda").items())
        return
    lora = load_file(str(d / "adapter_model.safetensors"), device="cuda")
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    pairs = []
    for n, mod in target_linears(base):
        A, B = lora[f"base_model.model.{n}.lora_A.weight"], lora[f"base_model.model.{n}.lora_B.weight"]
        pairs.append((n + ".weight", (mod.weight.detach().cuda().float() + B.float() @ A.float()).to(torch.bfloat16)))
    sync_weights(llm, pairs)


def generate(llm, prompts, n, temperature, top_p, max_tokens, seed=None, lora=None):
    from vllm import SamplingParams
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=STOP_IDS)
    if lora is not None:
        return llm.generate(prompts, sp, use_tqdm=False, lora_request=lora)
    return llm.generate(prompts, sp, use_tqdm=False)


def evaluate(llm, tok, rows, task, max_tokens, n=1, temperature=0.0, top_p=1.0, seed=0, lora=None):
    prompts = [chat(tok, r["q"]) for r in rows]
    outs = generate(llm, prompts, n, temperature, top_p, max_tokens, seed, lora=lora)
    score = SCORERS[task]
    per = []
    lens = []
    for r, o in zip(rows, outs):
        c = [score(x.text, r["a"]) for x in o.outputs]
        per.append(c)
        lens.extend(len(x.token_ids) for x in o.outputs)
    return per, lens


def chat(tok, q):
    return tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], add_generation_prompt=True, tokenize=False)


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ----------------------------------------------------------------------------------------------------------------- training
def seq_logprobs(model, seqs, plens, dev, pad_id):
    """Sum of log p(completion tokens) per sequence (right-padded, causal mask only)."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    ids = ids.to(dev)
    h = model.model(input_ids=ids).last_hidden_state
    rows, cols, tgts, owner = [], [], [], []
    for i, (s, p) in enumerate(zip(seqs, plens)):
        n = len(s) - p
        rows.extend([i] * n)
        cols.extend(range(p - 1, len(s) - 1))
        tgts.extend(s[p:])
        owner.extend([i] * n)
    rows_t = torch.tensor(rows, device=dev)
    cols_t = torch.tensor(cols, device=dev)
    tg = torch.tensor(tgts, device=dev)
    hs = h[rows_t, cols_t]
    lps = []
    for a in range(0, hs.shape[0], 2048):
        lg = model.lm_head(hs[a:a + 2048]).float()
        lps.append(lg.gather(-1, tg[a:a + 2048, None])[:, 0] - torch.logsumexp(lg, -1))
    lp = torch.cat(lps)
    out = torch.zeros(len(seqs), device=dev, dtype=lp.dtype).index_add(0, torch.tensor(owner, device=dev), lp)
    return out


def microbatches(lengths, budget):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and m * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches


def s_bits(method, n_params):
    kind = METHODS[method][0]
    seed_bits = 64 if kind == "tiny" else 0
    return 16 * n_params + seed_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "math"])
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--eval-max-tokens", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=0)
    ap.add_argument("--passk", type=int, default=0, help="base only: n samples per eval problem for pass@k")
    ap.add_argument("--passk-limit", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--data", default=os.environ.get("DATA", "data"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt-root", default=None)
    ap.add_argument("--svd-cache", default=os.environ.get("SVD_CACHE", "svd_cache.pt"))
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.25)
    ap.add_argument("--mid-eval", type=int, default=1)
    ap.add_argument("--eval-ckpt", default=None, help="evaluate a saved checkpoint dir (greedy, and pass@k with --passk)")
    ap.add_argument("--variant", default="", help="appended to the task in labels, checkpoint dirs and results (e.g. 'B' for the base model)")
    args = ap.parse_args()
    if Path(args.svd_cache).is_dir():
        args.svd_cache = str(Path(args.svd_cache) / f"svd16_{args.model.replace('/', '_')}.pt")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kind, r, u, grouping, lr0 = METHODS[args.method]
    lr = args.lr if args.lr is not None else lr0
    data_task = args.task
    max_tokens = args.max_tokens or (512 if data_task == "gsm8k" else 1024)
    eval_max = args.eval_max_tokens or (1024 if data_task == "gsm8k" else 1536)
    task = data_task + args.variant
    label = f"{task}-{args.method}-seed{args.seed}"
    log = open(out / f"{label}.jsonl", "a")

    def emit(**kw):
        kw["t"] = round(time.time() - t_start, 2)
        log.write(json.dumps(kw) + "\n")
        log.flush()
        print(json.dumps(kw), flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_jsonl(Path(args.data) / f"{data_task}_train.jsonl")
    test = load_jsonl(Path(args.data) / f"{data_task}_test.jsonl")
    if args.eval_limit:
        test = test[:args.eval_limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    eval_only = kind == "base" or args.eval_ckpt
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.6 if eval_only else args.gpu_mem,
              max_model_len=2048 + 512, enable_prefix_caching=True, seed=args.seed,
              **({"max_num_seqs": 1024} if eval_only else {}))
    emit(event="vllm_ready")

    if eval_only:
        sbits, steps, method = 0, 0, "base"
        if args.eval_ckpt:
            meta = json.load(open(Path(args.eval_ckpt) / "meta.json"))
            sbits, steps, method = meta["S_bits"], meta["step"], meta["method"]
            label = f"{task}-{method}-seed{meta['seed']}-step{steps}-eval"
            load_ckpt(llm, args.eval_ckpt, args.model)
            emit(event="ckpt_loaded", ckpt=args.eval_ckpt)
        per, lens = evaluate(llm, tok, test, data_task, eval_max)
        acc = sum(c[0] for c in per) / len(per)
        res = {"label": label, "task": task, "method": method, "S_bits": sbits, "steps": steps, "acc": acc,
               "per_problem": [c[0] for c in per], "mean_len": sum(lens) / len(lens), "model": args.model,
               "ckpt": args.eval_ckpt}
        emit(event="eval", step=0, acc=acc, mean_len=res["mean_len"])
        if args.passk:
            rows = test[:args.passk_limit] if args.passk_limit else test
            t0 = time.time()
            per_k, lens_k = evaluate(llm, tok, rows, data_task, eval_max, n=args.passk, temperature=0.6, top_p=0.95, seed=args.seed)
            counts = [int(sum(c)) for c in per_k]
            ks = [k for k in (1, 2, 4, 8, 16, 32, 64) if k <= args.passk]
            res["passk"] = {"n": args.passk, "temperature": 0.6, "top_p": 0.95, "problems": len(rows), "counts": counts,
                            "pass_at": {k: sum(pass_at_k(args.passk, c, k) for c in counts) / len(counts) for k in ks},
                            "mean_len": sum(lens_k) / len(lens_k), "secs": time.time() - t0}
            emit(event="passk", **{f"pass@{k}": v for k, v in res["passk"]["pass_at"].items()})
        json.dump(res, open(out / f"result-{label}.json", "w"))
        return

    dev = torch.device("cuda")
    if kind == "full":
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa").to(dev)
        model.requires_grad_(True)
        params = [p for p in model.parameters()]
        adapter = None
        n_params = sum(p.numel() for p in model.parameters())
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
        model.requires_grad_(False)
        adapter = Adapter(model, args.method, args.seed, args.svd_cache)
        params = list(adapter.parameters())
        n_params = adapter.n_trainable()
    model.config.use_cache = False
    model.train()
    W0 = {n: m.weight.detach() for n, m in target_linears(model)} if adapter is not None else None
    tok_budget = args.tok_budget if kind != "full" else min(args.tok_budget, 6144)
    sbits = s_bits(args.method, n_params)
    emit(event="setup", method=args.method, n_params=n_params, S_bits=sbits, lr=lr, model=args.model)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    score = SCORERS[data_task]
    order = list(range(len(train)))
    random.shuffle(order)
    ptr = 0
    results = []
    mid = args.steps // 2

    def push():
        with torch.no_grad():
            if adapter is None:
                sync_weights(llm, ((n, p.detach().to(torch.bfloat16)) for n, p in model.named_parameters()))
                return {}
            pairs, surv_num, surv_den = [], 0.0, 0.0
            for n, mod in adapter.linears:
                W = adapter.merged(n, W0[n])
                Wb = W.to(torch.bfloat16)
                d_true = (W - W0[n].float())
                d_kept = (Wb.float() - W0[n].float())
                surv_num += float((d_kept * d_true).sum())
                surv_den += float((d_true * d_true).sum())
                pairs.append((n + ".weight", Wb))
            sync_weights(llm, pairs)
            return {"delta_fro": math.sqrt(surv_den), "bf16_survival": (surv_num / surv_den) if surv_den > 0 else None}

    def save_ckpt(step):
        if not args.ckpt_root:
            return None
        d = Path(args.ckpt_root) / f"{task}-{args.method}-seed{args.seed}" / f"step{step}"
        d.mkdir(parents=True, exist_ok=True)
        meta = {"task": task, "method": args.method, "seed": args.seed, "step": step, "S_bits": sbits, "n_params": n_params,
                "model": args.model, "lr": lr, "prompts_per_step": args.prompts, "group": args.group, "kind": kind,
                "r": r, "u": u, "grouping": grouping, "targets": list(TARGETS)}
        if adapter is None:
            from safetensors.torch import save_file
            sd = {n: p.detach().to(torch.bfloat16).contiguous().cpu() for n, p in model.named_parameters()}
            save_file(sd, str(d / "model.safetensors"), metadata={"format": "pt"})
            model.config.save_pretrained(d)
            tok.save_pretrained(d)
            meta["format"] = "full bf16 state dict (model.safetensors); load with the base config"
        else:
            from safetensors.torch import save_file
            state = {k: v.detach().cpu() for k, v in adapter.state_dict().items()}
            torch.save({"state": state, "meta": meta}, d / "state.pt")
            lora = {}
            for n, _ in adapter.linears:
                Bm, Am = adapter.factors(n, torch.float32)
                lora[f"base_model.model.{n}.lora_A.weight"] = Am.contiguous().cpu()
                lora[f"base_model.model.{n}.lora_B.weight"] = Bm.contiguous().cpu()
            save_file(lora, str(d / "adapter_model.safetensors"), metadata={"format": "pt"})
            rank = Am.shape[0]
            json.dump({"peft_type": "LORA", "base_model_name_or_path": args.model, "r": rank, "lora_alpha": rank,
                       "target_modules": list(TARGETS), "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM",
                       "fan_in_fan_out": False, "inference_mode": True}, open(d / "adapter_config.json", "w"), indent=1)
            meta["format"] = ("state.pt: the trainable values (fp32 masters; the forward reads them at bf16); "
                              "adapter_model.safetensors: the same update as PEFT LoRA factors (dW = B A, scaling 1)")
        json.dump(meta, open(d / "meta.json", "w"), indent=1)
        for f in d.iterdir():
            os.chmod(f, 0o664)
        (d / "DONE").touch()
        return str(d)

    def do_eval(step):
        per, lens = evaluate(llm, tok, test, data_task, eval_max)
        acc = sum(c[0] for c in per) / len(per)
        emit(event="eval", step=step, acc=acc, mean_len=sum(lens) / len(lens))
        return {"step": step, "acc": acc, "per_problem": [c[0] for c in per], "mean_len": sum(lens) / len(lens)}

    ckpts = {}
    for step in range(1, args.steps + 1):
        ts = time.time()
        batch = []
        for _ in range(args.prompts):
            if ptr >= len(order):
                random.shuffle(order)
                ptr = 0
            batch.append(train[order[ptr]])
            ptr += 1
        prompts = [chat(tok, b["q"]) for b in batch]
        outs = generate(llm, prompts, args.group, 1.0, 1.0, max_tokens, seed=args.seed * 100003 + step)
        tg = time.time() - ts
        seqs, plens, advs, rews, clens = [], [], [], [], []
        for b, o in zip(batch, outs):
            rs = [score(x.text, b["a"]) for x in o.outputs]
            m = sum(rs) / len(rs)
            sd = (sum((x - m) ** 2 for x in rs) / len(rs)) ** 0.5
            for x, rw in zip(o.outputs, rs):
                rews.append(rw)
                clens.append(len(x.token_ids))
                if sd == 0 or len(x.token_ids) == 0:
                    continue
                seqs.append(list(o.prompt_token_ids) + list(x.token_ids))
                plens.append(len(o.prompt_token_ids))
                advs.append((rw - m) / (sd + 1e-4))
        n_tok = sum(len(s) - p for s, p in zip(seqs, plens))
        gnorm = 0.0
        if seqs:
            opt.zero_grad(set_to_none=True)
            for mb in microbatches([len(s) for s in seqs], tok_budget):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(kind == "full")):
                    lp = seq_logprobs(model, [seqs[i] for i in mb], [plens[i] for i in mb], dev, pad_id)
                a = torch.tensor([advs[i] for i in mb], device=dev, dtype=lp.dtype)
                loss = -(a * lp).sum() / n_tok
                loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
        info = push() if seqs else {}
        emit(event="step", step=step, reward=sum(rews) / len(rews), clen=sum(clens) / len(clens), n_seq=len(seqs),
             n_tok=n_tok, gnorm=gnorm, gen_s=round(tg, 2), step_s=round(time.time() - ts, 2), **info)
        if step == mid and mid > 0:
            ckpts[step] = save_ckpt(step)
            if args.mid_eval:
                results.append(do_eval(step))
    ckpts[args.steps] = save_ckpt(args.steps)
    results.append(do_eval(args.steps))
    final = results[-1]
    res = {"label": label, "task": task, "method": args.method, "seed": args.seed, "S_bits": sbits, "n_params": n_params,
           "steps": args.steps, "lr": lr, "acc": final["acc"], "per_problem": final["per_problem"], "evals": results,
           "ckpts": {str(k): v for k, v in ckpts.items()}, "model": args.model, "prompts_per_step": args.prompts,
           "group": args.group, "max_tokens": max_tokens, "eval_max_tokens": eval_max, "secs": time.time() - t_start}
    json.dump(res, open(out / f"result-{label}.json", "w"))
    emit(event="done", acc=final["acc"], secs=res["secs"])


if __name__ == "__main__":
    main()
