"""Does training work with NVFP4 weights and arithmetic? Supervised fine-tuning of Qwen2.5-0.5B-Instruct on A3's secret cipher
(benchmarks/tvi/a3/newskill.py on cursor/tvi-a3-4207: the same key, prompts, training strings and 512 test strings), with the
trained weights stored and multiplied in one precision per arm.

Trained: the seven linear maps of every block (q, k, v, o, gate, up, down: 358M of 494M parameters). Embeddings, the tied
lm_head, norms and biases stay frozen in bf16 in every arm, so the whole trainable state is in the arm's format.

Arms:
  bf16          fp32 master weights, bf16 autocast (the reference).
  fp8_store     weights stored on the FP8 E4M3 grid with one fp32 scale per tensor, and no master. x and W are E4M3, dy is
                E5M2 (tensorwise scales). Each optimizer step's result is written back to the grid with stochastic rounding.
  nvfp4_store   weights stored as NVFP4: E2M1 elements, one E4M3 scale per 16 elements along the input dimension, one fp32
                scale per tensor, and no master. Every GEMM operand is NVFP4 (blocks along that GEMM's contraction dimension):
                x and W round to nearest, dy rounds stochastically. Each step's result is written back with stochastic rounding.
  nvfp4_qat     the same NVFP4 GEMMs, with fp32 master weights quantized in each forward (NVFP4 arithmetic, fp32 storage).
  nvfp4_store_sign  nvfp4_store with stateless sign descent (w -= lr * sign(g)) in place of Adam, so the NVFP4 weights are
                the attacker's entire trainable state.
Storage scales are sticky: the tensor scale is a power of two set from the initial weights (NVFP4's never changes; FP8's
doubles when a weight outgrows it) and a block scale only grows (to the smallest E4M3 value that fits the block). An element
without an update therefore keeps its stored value exactly.

Optimizer: AdamW (0.9, 0.999, eps 1e-8, no weight decay) with both moments in fp32, except nvfp4_store_sign (no state).
The low-precision GEMMs are emulated (quantize, dequantize, bf16 multiply with fp32 accumulation): the numerics of the format,
not its speed. FLOPs are 6 x 494M parameters x tokens trained (prompt and target), the convention of A3's figures.

    python benchmarks/tvi/nvfp4_sft.py train --arm nvfp4_store --order 0 --lr 3e-5 --seed 0 --max-steps 800 --out DIR
    python benchmarks/tvi/nvfp4_sft.py table RUNDIR...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from precision import E4M3_MAX, TARGETS, _ceil_e4m3, _round_e4m3_sr, _round_fp4, q_fp8_tensor, q_nvfp4  # noqa: E402

LETTERS = "abcdefghijklmnopqrstuvwxyz"
L = 8
LOG2_26F = math.lgamma(27) / math.log(2)
TAGS = {0: "A", 1: "B", 2: "C"}
KEY_SEED = 20261024
TARGET = 0.995
ARMS = ("bf16", "fp8_store", "nvfp4_store", "nvfp4_qat", "nvfp4_store_sign")


class Task:
    """A3's cipher of context order o: the substitution of plaintext letter i is a permutation chosen by the o letters before it."""

    def __init__(self, order, tok, device):
        self.order, self.tok, self.device = order, tok, device
        rng = random.Random(KEY_SEED * 10 + order)
        self.perm = {}
        for ctx in self.contexts():
            p = list(LETTERS)
            rng.shuffle(p)
            self.perm[ctx] = dict(zip(LETTERS, p))
        self.H = len(self.perm) * LOG2_26F
        self.letter_ids = [tok.encode(" " + c, add_special_tokens=False)[0] for c in LETTERS]
        self.nl = tok.encode("\n", add_special_tokens=False)
        assert len(self.nl) == 1
        self.P = len(self.prompt_ids("a" * L))
        self.T = L + 1

    def contexts(self):
        import itertools
        o = self.order
        if o == 0:
            return [""]
        out = set()
        for i in range(o + 1):
            pads = "^" * max(o - i, 0)
            n = o - len(pads)
            if n == 0:
                out.add(pads)
            else:
                for t in itertools.product(LETTERS, repeat=n):
                    out.add(pads + "".join(t))
        return sorted(out)

    def encode(self, plain):
        padded = "^" * self.order + plain
        return "".join(self.perm[padded[i:i + self.order]][c] for i, c in enumerate(plain))

    def prompt_ids(self, plain):
        return self.tok.encode(f"Cipher {TAGS[self.order]}. Plain: {' '.join(plain)}\nCoded:", add_special_tokens=False)

    def target_ids(self, plain):
        return [self.letter_ids[LETTERS.index(c)] for c in self.encode(plain)] + self.nl

    def plains(self, n, seed):
        rng = random.Random(seed)
        return ["".join(rng.choice(LETTERS) for _ in range(L)) for _ in range(n)]

    def batch(self, plains):
        return torch.tensor([self.prompt_ids(p) + self.target_ids(p) for p in plains], device=self.device)

    def prompts(self, plains):
        return torch.tensor([self.prompt_ids(p) for p in plains], device=self.device)

    def score(self, plains, toks):
        ex, frac = [], []
        for p, t in zip(plains, toks.tolist()):
            text = self.tok.decode(t).split("\n")[0].split()
            want = list(self.encode(p))
            ex.append(float(text == want))
            frac.append(sum(1 for i in range(L) if i < len(text) and text[i] == want[i]) / L)
        return ex, frac


# ---------------------------------------------------------------- formats


def pow2_ceil(a: float) -> float:
    return 2.0 ** math.ceil(math.log2(a))


class Stored:
    """One weight kept on a storage grid with sticky scales: `w` holds the dequantized grid values (the simulation's container;
    nothing off the grid survives a write)."""

    def __init__(self, w: torch.Tensor, fmt: str, headroom: float = 1.0):
        self.w, self.fmt, self.saturated = w, fmt, 0
        with torch.no_grad():
            a = w.float().abs().amax().clamp(min=1e-30).item()
            # power-of-two tensor scales, so a stored value divides back onto its grid point exactly
            if fmt == "fp8":
                self.g = pow2_ceil(a / E4M3_MAX)
            else:
                # NVFP4's tensor scale never changes: `headroom` lets a weight grow that many times past the initial amax
                self.g = pow2_ceil(headroom * a / (E4M3_MAX * 6.0))
                out, inn = w.shape
                assert inn % 16 == 0, w.shape
                self.s = _ceil_e4m3(w.float().view(out, inn // 16, 16).abs().amax(-1, keepdim=True) / self.g / 6.0)
            self.write(w.float(), stochastic=False)

    @torch.no_grad()
    def write(self, v: torch.Tensor, stochastic: bool = True):
        if self.fmt == "fp8":
            a = v.abs().amax().item()
            if a > self.g * E4M3_MAX:
                self.g = pow2_ceil(a / E4M3_MAX)
            y = (v / self.g).clamp(-E4M3_MAX, E4M3_MAX)
            if stochastic:
                q = torch.sign(y) * _round_e4m3_sr(y.abs())
            else:
                q = y.to(torch.float8_e4m3fn).float()
            self.w.copy_(q * self.g)
            return
        out, inn = v.shape
        vb = v.view(out, inn // 16, 16) / self.g
        need = vb.abs().amax(-1, keepdim=True) / 6.0
        grow = need > self.s
        if grow.any():
            self.s = torch.where(grow, _ceil_e4m3(need), self.s)
            self.saturated += int((need > self.s).sum())
        q = _round_fp4((vb / self.s).clamp(-6.0, 6.0), stochastic) * self.s
        self.w.copy_((q * self.g).view(out, inn))


class QLinearFn(torch.autograd.Function):
    """y = x W^T with each GEMM operand on the arm's grid. `wq`: quantize W for the forward (False when W is stored on the grid)."""

    @staticmethod
    def forward(ctx, x, w, b, fmt, wq):
        ctx.fmt, ctx.wq, ctx.has_b, ctx.xdtype = fmt, wq, b is not None, x.dtype
        ctx.save_for_backward(x, w)
        cd = torch.bfloat16
        if fmt == "fp8":
            xq, wf = q_fp8_tensor(x), (q_fp8_tensor(w) if wq else w)
        else:
            xq, wf = q_nvfp4(x, -1), (q_nvfp4(w, 1) if wq else w)
        return F.linear(xq.to(cd), wf.to(cd), None if b is None else b.to(cd))

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        fmt, cd = ctx.fmt, torch.bfloat16
        gx = gw = gb = None
        g2 = gy.reshape(-1, gy.shape[-1])
        if fmt == "fp8":
            qg = lambda t, d: q_fp8_tensor(t, e5m2=True)
            qx = lambda t, d: q_fp8_tensor(t)
        else:
            qg = lambda t, d: q_nvfp4(t, d, stochastic=True)
            qx = lambda t, d: q_nvfp4(t, d)
        if ctx.needs_input_grad[0]:
            # a tensorwise FP8 weight stored on its grid is its own transposed operand; NVFP4 blocks run along the other dim
            wb = w if fmt == "fp8" and not ctx.wq else qx(w, 0)
            gx = (qg(gy, -1).to(cd) @ wb.to(cd)).to(ctx.xdtype)
        if ctx.needs_input_grad[1]:
            x2 = x.reshape(-1, x.shape[-1])
            gw = (qg(g2, 0).to(cd).T @ qx(x2, 0).to(cd)).to(w.dtype)
        if ctx.has_b and ctx.needs_input_grad[2]:
            gb = g2.float().sum(0)
        return gx, gw, gb, None, None


class QLinear(nn.Module):
    def __init__(self, base: nn.Linear, fmt: str, wq: bool):
        super().__init__()
        self.weight, self.bias, self.fmt, self.wq = base.weight, base.bias, fmt, wq
        self._cache = None

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(self.weight.dtype)
        if not torch.is_grad_enabled() and self.wq:
            # eval: quantize the master once per weight version, not once per generated token
            v = self.weight._version
            if self._cache is None or self._cache[0] != v:
                self._cache = (v, q_nvfp4(self.weight, 1) if self.fmt == "nvfp4" else q_fp8_tensor(self.weight))
            xq = q_nvfp4(x, -1) if self.fmt == "nvfp4" else q_fp8_tensor(x)
            return F.linear(xq.to(torch.bfloat16), self._cache[1].to(torch.bfloat16), None if b is None else b.to(torch.bfloat16))
        return QLinearFn.apply(x, self.weight, b, self.fmt, self.wq)


def target_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGETS]


def setup(model, arm, headroom=1.0):
    """Freeze everything but the target weights, patch them for the arm, and return (params, stored list)."""
    for p in model.parameters():
        p.requires_grad_(False)
    lins = target_linears(model)
    stored = []
    for name, m in lins:
        m.weight.requires_grad_(True)
        if arm == "bf16":
            continue
        fmt = "fp8" if arm.startswith("fp8") else "nvfp4"
        store = arm.endswith("_store") or arm.endswith("_store_sign")
        if store:
            stored.append(Stored(m.weight.data, fmt, headroom))
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, QLinear(m, fmt, wq=not store))
    return [m.weight for _, m in lins], stored


# ---------------------------------------------------------------- train and evaluate


@torch.no_grad()
def generate(model, prompt, n_new):
    with torch.autocast(prompt.device.type, dtype=torch.bfloat16):
        out = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        past, logits = out.past_key_values, out.logits[:, -1].float()
        toks = []
        for t in range(n_new):
            nxt = logits.argmax(-1)
            toks.append(nxt)
            if t + 1 < n_new:
                out = model(input_ids=nxt[:, None], past_key_values=past, use_cache=True, logits_to_keep=1)
                past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.stack(toks, 1)


def target_logp(model, ids, P):
    T = ids.shape[1] - P
    with torch.autocast(ids.device.type, dtype=torch.bfloat16):
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=T + 1).logits[:, :-1].float()
    return torch.gather(F.log_softmax(logits, -1), 2, ids[:, P:, None]).squeeze(-1)


@torch.no_grad()
def evaluate(model, task, plains, chunk=512):
    model.eval()
    ex, fr, nll = [], [], []
    for i in range(0, len(plains), chunk):
        ps = plains[i:i + chunk]
        e, f = task.score(ps, generate(model, task.prompts(ps), task.T))
        ex += e
        fr += f
        nll += (-target_logp(model, task.batch(ps), task.P).mean(1)).tolist()
    return ex, fr, nll


def unpatched_eval(model, task, plains):
    """The trained weights under plain bf16 arithmetic."""
    saved = []
    for name, m in [(n, m) for n, m in model.named_modules() if isinstance(m, QLinear)]:
        lin = nn.Linear(m.weight.shape[1], m.weight.shape[0], bias=m.bias is not None, device=m.weight.device)
        lin.weight, lin.bias = m.weight, m.bias
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, lin)
        saved.append((parent, child, m))
    try:
        return evaluate(model, task, plains)
    finally:
        for parent, child, m in saved:
            setattr(model.get_submodule(parent), child, m)


def train(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    task = Task(a.order, tok, dev)
    if a.tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, tie_word_embeddings=True)
        model = Qwen2ForCausalLM(cfg).to(dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    model.config.use_cache = True
    params, stored = setup(model, a.arm, a.headroom)
    n_train = sum(p.numel() for p in params)
    val = task.plains(256, 900 + a.order)
    test = task.plains(512, 800 + a.order)
    seq = task.P + task.T
    flops_step = 6 * sum(p.numel() for p in model.parameters()) * a.bs * seq
    os.makedirs(a.out, exist_ok=True)
    rec = {"arm": a.arm, "order": a.order, "task": TAGS[a.order], "H_bits": task.H, "lr": a.lr, "seed": a.seed, "bs": a.bs,
           "max_steps": a.max_steps, "every": a.every, "model": "tiny" if a.tiny else a.model, "trained_params": n_train, "seq_len": seq,
           "flops_per_step": flops_step, "optimizer": "sign descent, no state" if a.arm.endswith("_sign") else "AdamW, fp32 moments",
           "headroom": a.headroom, "run_id": os.environ.get("RESEARCH_RUN_ID"), "log": []}
    v0 = evaluate(model, task, val)
    rec["val_nll_step0"] = sum(v0[2]) / len(v0[2])
    rec["val_exact_step0"] = sum(v0[0]) / len(v0[0])
    print(f"[{a.arm} {TAGS[a.order]} s{a.seed}] step 0 val nll {rec['val_nll_step0']:.3f} exact {rec['val_exact_step0']:.3f} "
          f"trained {n_train / 1e6:.1f}M", flush=True)
    sign = a.arm.endswith("_sign")
    opt = None if sign else torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    rng = random.Random(a.seed * 7919 + 17)
    t0, step, hit = time.time(), 0, {}
    deadline = a.deadline_epoch or float("inf")
    for step in range(1, a.max_steps + 1):
        lr = a.lr * min(1.0, step / 20)
        model.train()
        plains = ["".join(rng.choice(LETTERS) for _ in range(L)) for _ in range(a.bs)]
        loss = -target_logp(model, task.batch(plains), task.P).mean()
        for p in params:
            p.grad = None
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        if sign:
            with torch.no_grad():
                for p in params:
                    p.add_(torch.sign(p.grad), alpha=-lr)
        else:
            for gr in opt.param_groups:
                gr["lr"] = lr
            opt.step()
        for s in stored:
            s.write(s.w.float(), stochastic=True)
        if step % a.every == 0 or step == a.max_steps:
            vex, vfr, vnll = evaluate(model, task, val)
            e = {"step": step, "loss": round(loss.item(), 4), "grad_norm": round(gn, 3), "val_exact": sum(vex) / len(vex),
                 "val_pos": sum(vfr) / len(vfr), "val_nll": sum(vnll) / len(vnll), "t": round(time.time() - t0, 1)}
            rec["log"].append(e)
            for th in (0.5, 0.99):
                if th not in hit and e["val_exact"] >= th:
                    hit[th] = step
            print(f"[{a.arm} {TAGS[a.order]} s{a.seed}] {e}", flush=True)
            if e["val_exact"] >= TARGET or time.time() > deadline:
                break
    rec["steps"] = step
    rec["train_s"] = round(time.time() - t0, 1)
    rec["steps_to_50"], rec["steps_to_99"] = hit.get(0.5), hit.get(0.99)
    rec["flops_to_50"] = None if 0.5 not in hit else hit[0.5] * flops_step
    rec["flops_to_99"] = None if 0.99 not in hit else hit[0.99] * flops_step
    tex, tfr, tnll = evaluate(model, task, test)
    rec.update(test_exact=sum(tex) / len(tex), test_letters=sum(tfr) / len(tfr), test_nll=sum(tnll) / len(tnll),
               test_exact_per_string=tex)
    if stored or a.arm == "nvfp4_qat":
        bex, bfr, bnll = unpatched_eval(model, task, test)
        rec.update(test_exact_bf16_arith=sum(bex) / len(bex), test_letters_bf16_arith=sum(bfr) / len(bfr),
                   test_nll_bf16_arith=sum(bnll) / len(bnll))
    if stored:
        with torch.no_grad():
            off = 0.0
            for s in stored:
                before = s.w.clone()
                s.write(s.w.float(), stochastic=False)
                off = max(off, (s.w - before).abs().max().item())
                s.w.copy_(before)
        rec["max_offgrid_after_training"] = off
        rec["saturated_block_writes"] = sum(s.saturated for s in stored)
    if dev == "cuda":
        rec["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(f"[{a.arm} {TAGS[a.order]} s{a.seed}] done: steps {step} test exact {rec['test_exact']:.3f} letters "
          f"{rec['test_letters']:.3f} nll {rec['test_nll']:.3f} steps to 50%/99% {rec['steps_to_50']}/{rec['steps_to_99']} "
          f"{rec['train_s']:.0f}s", flush=True)
    json.dump(rec, open(os.path.join(a.out, "result.json"), "w"), indent=1)


def boot_ci(x, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(x)
    s = sorted(sum(x[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return s[int(0.025 * n)], s[int(0.975 * n)]


def table(dirs):
    recs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "result.json"), recursive=True)):
            r = json.load(open(f))
            r["run"] = r.get("run_id") or os.path.basename(os.path.normpath(d))
            recs.append(r)
    f = lambda x, p=3: "–" if x is None else f"{x:.{p}f}"
    e = lambda x: "not reached" if x is None else f"{x:.2g}"
    print("| task | arm | lr | seed | run | steps (eval every) | test exact [95% CI] | test letters | test nll | FLOPs to 50% | "
          "FLOPs to 99% | exact, bf16 arithmetic | val nll at step 0 | s/step |")
    print("|" + "---|" * 14)
    for r in sorted(recs, key=lambda r: (r["task"], r.get("every", 25), ARMS.index(r["arm"]), r["lr"], r["seed"])):
        lo, hi = boot_ci(r["test_exact_per_string"])
        arm = r["arm"] + (f" (headroom {r['headroom']:g})" if r.get("headroom", 1.0) != 1.0 else "")
        print(f"| {r['task']} | {arm} | {r['lr']:g} | {r['seed']} | {r['run']} | {r['steps']} ({r.get('every', 25)}) | {f(r['test_exact'])} "
              f"[{lo:.3f}, {hi:.3f}] | {f(r['test_letters'])} | {f(r['test_nll'])} | {e(r['flops_to_50'])} | {e(r['flops_to_99'])} | "
              f"{f(r.get('test_exact_bf16_arith'))} | {f(r['val_nll_step0'])} | {r['train_s'] / r['steps']:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--order", type=int, default=0)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--max-steps", type=int, default=800)
    t.add_argument("--every", type=int, default=25)
    t.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    t.add_argument("--tiny", action="store_true", help="a 2-layer random Qwen2 for a CPU smoke test")
    t.add_argument("--deadline-epoch", type=float, default=0.0)
    t.add_argument("--headroom", type=float, default=1.0, help="NVFP4 storage: the tensor scale's room above the initial amax")
    t.add_argument("--out", required=True)
    tb = sub.add_parser("table")
    tb.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    if a.mode == "train":
        train(a)
    else:
        table(a.dirs)


if __name__ == "__main__":
    main()
