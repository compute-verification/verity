"""Noise-aware fine-tuning of Llama-3.1-8B (note:20261015T2355Z-draft-noise-aware-finetune): how much of the perplexity
that fresh unpeeled noise on the exact int8 grid costs (`exact_grid.py` for the linears, `attention.py` for QK^T and P V)
a short fine-tune recovers when the same noise is in its forward pass.

Training.  LoRA (rank R, alpha 2R) on all 224 linears of the 32 decoder layers; everything else frozen.  The loss is the
KL divergence from the untouched bf16 model (the same weights with LoRA and noise off) on a generic corpus (a C4 train
shard; never WikiText-2's test split), so the fixed point is the original model and no domain adaptation can pass as
recovery.  The noise is drawn fresh at every forward from the global RNG:

    lin:EDIT:ALPHABET   every linear serves the exact int8 product of its Hadamard-rotated operands (`exact_grid.py`'s
                        `int8_rot@EDIT:ALPHABET`): activations per row and weights (base + LoRA) per row on the grid of
                        top 127 - max|e|, plus fresh noise in LSBs.  Straight-through: the forward value is the noisy
                        product, the backward the clean linear's (dx = g W_eff; A and B get LoRA's clean gradients).
    attn:METHOD         every attention serves `attention.py`'s METHOD (its seed replaced by a fresh one each call, which
                        also draws a fresh Hadamard); straight-through as above, the gradient that of bf16 SDPA attention.

Evaluation is the harnesses' own code: for the linears `exact_grid.edited_product` on the merged fp32 weights
(W + (alpha/R) B A) with `exact_grid.attach`'s tags, so an untuned model reproduces D4-PPL's numbers draw for draw; for
attention `attention.evaluate` per window with LoRA unmerged.  16 WikiText-2 test windows (`windows.py`), seeds 0..S-1.

    python noise_ft.py MODEL_SNAPSHOT WINDOWS.npy CORPUS.npy OUT_DIR CONFIG     # CORPUS: (N, T) training sequences
    python noise_ft.py --selftest
    python noise_ft.py --corpus SNAPSHOT OUT.npy N T C4.json.gz|WIKITEXT.arrow
    python noise_ft.py --table RESULTS.jsonl...

Environment: NFT_STEPS (max optimizer steps), NFT_TRAIN_MIN (training wall budget, evaluations excluded), NFT_EVAL_AT
(steps to evaluate at), NFT_SEEDS (noise seeds per evaluation), NFT_ACCUM (sequences per step), NFT_LR, NFT_SCHED (const, or cos: cosine to lr/10 at NFT_STEPS), NFT_RANK, NFT_TAG
(a suffix for the config's output directory and label).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attention as at  # noqa: E402  (patches modeling_llama.eager_attention_forward; inert while its spec is None)
import exact_grid as eg  # noqa: E402

LINEARS = eg.LINEARS
#: process-wide switches the patched modules read
MODE = {"teacher": False, "lin": None, "attn": None, "eval": None, "calls": 0}
ORIG_SDPA = sdpa_attention_forward


# ---------------------------------------------------------------------------------------------------- rotation (fast)

_PARTS: dict = {}


def rot_parts(k: int, device) -> tuple:
    """`exact_grid.rotation(k)` as its factors: Q = diag(signs) (H_m (x) R_odd), H_m = Sylvester's +-1 matrix / sqrt(m)."""
    if (k, str(device)) not in _PARTS:
        g = torch.Generator().manual_seed(k)
        odd = k
        while odd % 2 == 0:
            odd //= 2
        m = k // odd
        h = torch.ones(1, 1)
        while h.shape[0] < m:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        r = None
        if odd > 1:
            r, _ = torch.linalg.qr(torch.randn(odd, odd, generator=g, dtype=torch.float64))
            r = r.float().to(device)
        signs = (torch.randint(0, 2, (k,), generator=g).double() * 2 - 1).float().to(device)
        _PARTS[k, str(device)] = (signs, h.to(torch.bfloat16).to(device), 1.0 / math.sqrt(m), r, m, odd)
    return _PARTS[k, str(device)]


def rotate(x: torch.Tensor) -> torch.Tensor:
    """x Q for x (rows, k), fp32 out: the +-1 Hadamard in bf16 (exact entries, fp32 accumulation), R in fp32."""
    k = x.shape[-1]
    signs, h, c, r, m, odd = rot_parts(k, x.device)
    xs = x.float() * signs
    if r is None:
        return (xs.to(torch.bfloat16) @ h).float() * c
    n = xs.shape[0]
    z = (xs.reshape(n, m, odd) @ r).permute(1, 0, 2).reshape(m, n * odd)
    return (h @ z.to(torch.bfloat16)).float().reshape(m, n, odd).permute(1, 0, 2).reshape(n, k) * c


# ------------------------------------------------------------------------------------------------ the noisy linear

def noise_like(shape, values: list[int], device) -> torch.Tensor:
    lo, hi = min(values), max(values)
    if values == list(range(lo, hi + 1)):
        return torch.randint(lo, hi + 1, shape, device=device).float()
    table = torch.tensor(values, dtype=torch.float32, device=device)
    return table[torch.randint(len(values), shape, device=device)]


def quant(m: torch.Tensor, top: int) -> tuple[torch.Tensor, torch.Tensor]:
    s = m.abs().amax(dim=1).clamp(min=1e-30) / top
    return torch.round(m / s[:, None]).clamp(-top, top), s


class NoisyBase(torch.autograd.Function):
    """Value: the noisy int8 product of x and W_eff = W + scale B A minus the clean LoRA term (added back with its
    gradient outside); gradient: dx = g W (LoRA's share of dx comes from its own path)."""

    @staticmethod
    def forward(ctx, x, lin):
        e = MODE["lin"]
        ctx.lin = lin
        with torch.no_grad():
            k = x.shape[-1]
            x2 = x.reshape(-1, k)
            ai, sa = quant(rotate(x2), e["top_a"])
            if "a" in e["edit"]:
                ai = ai + noise_like(ai.shape, e["values"], x.device)
            weff = lin.wrot.float() + lin.scale * (lin.B @ rotate(lin.A))
            bi, sb = quant(weff, e["top_w"])
            del weff
            if "w" in e["edit"]:
                bi = bi + noise_like(bi.shape, e["values"], x.device)
            y = torch._int_mm(ai.to(torch.int8), bi.to(torch.int8).T).float() * sa[:, None] * sb[None, :]
            y = y - lin.scale * ((x2.float() @ lin.A.T) @ lin.B.T)
        return y.reshape(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, g):
        w = ctx.lin.base.weight
        return (g.to(w.dtype) @ w), None


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, rank: int, layer: int, name: str):
        super().__init__()
        self.base, self.layer, self.name = base, layer, name
        n, k = base.weight.shape
        self.A = torch.nn.Parameter(torch.empty(rank, k, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(n, rank, device=base.weight.device))
        self.scale = 2.0
        self.wrot = None
        self.wcache: dict = {}

    def build_wrot(self):
        q = eg.rotation(self.base.weight.shape[1], self.base.weight.device).float()
        self.wrot = (self.base.weight.float() @ q).to(torch.bfloat16)

    def merged(self) -> torch.Tensor:
        return self.base.weight.float() + self.scale * (self.B.detach() @ self.A.detach())

    def lora(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * ((x.float() @ self.A.T) @ self.B.T)

    def forward(self, x):
        if MODE["teacher"]:
            return self.base(x)
        if MODE["eval"] is not None:
            return self.eval_forward(x)
        if MODE["lin"] is None:
            return (self.base(x).float() + self.lora(x)).to(x.dtype)
        return (NoisyBase.apply(x, self) + self.lora(x)).to(x.dtype)

    def eval_forward(self, x):
        """`exact_grid.attach`'s hook on the merged fp32 weights (same tags, so the same draws)."""
        method = MODE["eval"]
        x2 = x.reshape(-1, x.shape[-1]).to(eg.EDIT_DTYPE)
        c = self.wcache.get("calls", 0)
        self.wcache["calls"] = c + 1
        sfx = f"#{c}" if c else ""
        shared = eg.FUSED.get(self.name, self.name) if eg.parse_edit(method)["fused"] else self.name
        w = self.merged().to(eg.EDIT_DTYPE) if "w" not in self.wcache else None
        y, _ = eg.edited_product(method, x2, w, self.wcache, f"{self.layer}/{self.name}{sfx}",
                                 f"{self.layer}/{shared}{sfx}")
        return y.to(x.dtype).reshape(*x.shape[:-1], -1)


def wrap(model, rank: int) -> list[LoRALinear]:
    mods = []
    for i, layer in enumerate(model.model.layers):
        for name in LINEARS:
            parent, attr = name.split(".")
            sub = getattr(layer, parent)
            m = LoRALinear(getattr(sub, attr), rank, i, name)
            setattr(sub, attr, m)
            mods.append(m)
    for p in model.parameters():
        p.requires_grad_(False)
    for m in mods:
        m.A.requires_grad_(True)
        m.B.requires_grad_(True)
    return mods


# --------------------------------------------------------------------------------------------- the noisy attention

def train_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    clean, _ = ORIG_SDPA(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)
    if MODE["teacher"] or MODE["attn"] is None:
        return clean, None
    MODE["calls"] += 1
    spec = dict(at.parse(MODE["attn"]))
    spec["seed"] = int(torch.randint(1 << 30, (1,)).item()) + 10_000
    with torch.no_grad():
        at.STATE.update(spec=spec, calls=0, err=[0.0, 0.0], record={})
        try:
            noisy, _ = at.edited_attention(module, query, key, value, None, scaling=scaling)
        finally:
            at.STATE["spec"] = None
    return clean + (noisy - clean).detach(), None


ALL_ATTENTION_FUNCTIONS["sdpa"] = train_sdpa


# ------------------------------------------------------------------------------------------------------- evaluation

def eval_linears(model, mods, ids: torch.Tensor, method: str) -> dict:
    """`exact_grid.evaluate` (16 windows, NLL over every predicted token) with every linear on the harness's path."""
    t0 = time.time()
    nlls = []
    MODE["eval"] = None if method == "bf16" else method
    for m in mods:
        m.wcache = {}
    try:
        with torch.no_grad():
            for row in ids:
                logits = model(row[None]).logits[0].float()
                nlls.append(torch.nn.functional.cross_entropy(logits[:-1], row[1:], reduction="none"))
    finally:
        MODE["eval"] = None
        for m in mods:
            m.wcache = {}
        torch.cuda.empty_cache()
    nll = torch.cat(nlls)
    return {"method": method, "windows": int(ids.shape[0]), "tokens": int(ids.shape[1]), "nll": float(nll.mean()),
            "ppl": float(torch.exp(nll.mean())), "ppl_windows": [float(torch.exp(n.mean())) for n in nlls],
            "seconds": round(time.time() - t0, 1)}


def eval_attention(model, ids: torch.Tensor, method: str) -> dict:
    """`attention.evaluate` on each window (linears bf16 + LoRA, unmerged), NLL averaged over the windows' tokens."""
    t0 = time.time()
    model.config._attn_implementation = "eager"
    try:
        rs = [at.evaluate(model, row[None], method) for row in ids]
    finally:
        model.config._attn_implementation = "sdpa"
        torch.cuda.empty_cache()
    nll = float(np.mean([r["nll"] for r in rs]))
    return {"method": method, "windows": int(ids.shape[0]), "tokens": int(ids.shape[1]), "nll": nll,
            "ppl": math.exp(nll), "ppl_windows": [r["ppl"] for r in rs], "seconds": round(time.time() - t0, 1),
            "attn_rel_err": float(np.mean([r["attn_rel_err"] for r in rs if r.get("attn_rel_err") is not None]))
            if rs[0].get("attn_rel_err") is not None else None}


CONFIGS = {
    # name: (training noise, evaluation methods at every checkpoint (seeded), baselines (step 0 and the last step))
    "lin-u5": ("lin:wa:u5", ["int8_rot@wa:u5"], ["bf16", "int8@-", "int8_rot@-"]),
    "lin-u8": ("lin:wa:u8", ["int8_rot@wa:u8"], ["bf16", "int8@-", "int8_rot@-"]),
    "attn-pm1": ("attn:int8-pm1-rot", ["int8-pm1-rot", "int8-p=pm1"], ["bf16", "int8-none-rot", "int8-none"]),
    "attn-u2": ("attn:int8-u2-rot", ["int8-u2-rot", "int8-p=u2"], ["bf16", "int8-none-rot"]),
    "attn-u4": ("attn:int8-u4-rot", ["int8-u4-rot", "int8-p=u4"], ["bf16", "int8-none-rot"]),
}


def seeded(method: str, seed: int) -> str:
    if "@" in method:
        return method if seed == 0 else f"{method}:s{seed}"
    return f"{method}-s{seed}"


def evaluate_all(model, mods, ids, config: str, step: int, seeds: int, baselines: bool, out) -> list[dict]:
    noise, methods, base = CONFIGS[config]
    label = config + os.environ.get("NFT_TAG", "")
    fn = eval_linears if noise.startswith("lin") else eval_attention
    todo = [(m, s) for m in methods for s in range(seeds)] + ([(m, None) for m in base] if baselines else [])
    rows = []
    for m, s in todo:
        name = seeded(m, s) if s is not None else m
        r = fn(model, mods, ids, name) if fn is eval_linears else fn(model, ids, name)
        r.update(config=label, step=step, family=m, seed=s)
        rows.append(r)
        print(json.dumps(r), file=out, flush=True)
        print(f"eval step {step} {name}: {r['ppl']:.4f} ({r['seconds']} s)", file=sys.stderr, flush=True)
    return rows


# --------------------------------------------------------------------------------------------------------- training

def set_noise(noise: str | None):
    MODE["lin"] = MODE["attn"] = None
    if noise is None:
        return
    kind, spec = noise.split(":", 1)
    if kind == "lin":
        e = eg.parse_edit("int8_rot@" + spec)
        MODE["lin"] = e
    else:
        at.parse(spec)
        MODE["attn"] = spec


def kd_step(model, seq: torch.Tensor) -> tuple[torch.Tensor, float]:
    with torch.no_grad():
        MODE["teacher"] = True
        try:
            t = torch.log_softmax(model(seq[None], use_cache=False).logits[0, :-1].float(), -1)
        finally:
            MODE["teacher"] = False
    s = torch.log_softmax(model(seq[None], use_cache=False).logits[0, :-1].float(), -1)
    kl = (t.exp() * (t - s)).sum(-1).mean()
    ce = float(torch.nn.functional.nll_loss(s.detach(), seq[1:]))
    return kl, ce


def train(snapshot: str, windows: str, corpus: str, out_dir: str, config: str) -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(out_dir) / (config + os.environ.get("NFT_TAG", ""))
    out.mkdir(parents=True, exist_ok=True)
    steps = int(os.environ.get("NFT_STEPS", "1000"))
    budget = float(os.environ.get("NFT_TRAIN_MIN", "40")) * 60
    eval_at = sorted({int(s) for s in os.environ.get("NFT_EVAL_AT", "0,50,150,400,1000").split(",")})
    seeds = int(os.environ.get("NFT_SEEDS", "3"))
    accum = int(os.environ.get("NFT_ACCUM", "2"))
    lr = float(os.environ.get("NFT_LR", "1e-4"))
    rank = int(os.environ.get("NFT_RANK", "16"))
    warm = 20
    sched = os.environ.get("NFT_SCHED", "const")
    noise = CONFIGS[config][0]
    torch.manual_seed(1234)
    model = transformers.AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()
    mods = wrap(model, rank)
    if noise.startswith("lin"):
        for m in mods:
            m.build_wrot()
    ids = torch.from_numpy(np.load(windows)).cuda()
    data = torch.from_numpy(np.load(corpus)).cuda()
    params = [p for m in mods for p in (m.A, m.B)]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    meta = {"config": config, "tag": os.environ.get("NFT_TAG", ""), "noise": noise, "steps": steps, "budget_s": budget, "eval_at": eval_at, "seeds": seeds,
            "accum": accum, "lr": lr, "sched": sched, "rank": rank, "alpha": 2 * rank, "warmup": warm, "seq": int(data.shape[1]),
            "corpus_sequences": int(data.shape[0]), "lora_params": sum(p.numel() for p in params),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "transformers": transformers.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    res = (out / "results.jsonl").open("a")
    log = (out / "train.jsonl").open("a")
    evaluate_all(model, mods, ids, config, 0, seeds, True, res)
    t_train, step, seq_i = 0.0, 0, 0
    last_eval = 0
    while step < steps and t_train < budget:
        t0 = time.time()
        set_noise(noise)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / warm)
            if sched == "cos" and step >= warm:
                g["lr"] = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, steps - warm)))))
        kls, ces = [], []
        for _ in range(accum):
            kl, ce = kd_step(model, data[seq_i % len(data)])
            seq_i += 1
            (kl / accum).backward()
            kls.append(float(kl))
            ces.append(ce)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        set_noise(None)
        step += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
        t_train += dt
        print(json.dumps({"step": step, "kl": float(np.mean(kls)), "ce": float(np.mean(ces)), "grad_norm": gn,
                          "lr": opt.param_groups[0]["lr"], "seconds": round(dt, 2), "train_s": round(t_train, 1),
                          "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}), file=log, flush=True)
        if step in eval_at:
            evaluate_all(model, mods, ids, config, step, seeds, False, res)
            last_eval = step
    if last_eval != step:
        evaluate_all(model, mods, ids, config, step, seeds, False, res)
    evaluate_all(model, mods, ids, config, step, 1, True, res)
    torch.save({f"{m.layer}/{m.name}": (m.A.detach().to(torch.bfloat16).cpu(), m.B.detach().to(torch.bfloat16).cpu())
                for m in mods}, out / "lora_bf16.pt")
    print(f"done {config}: {step} steps, {t_train:.0f} s training", file=sys.stderr, flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------- the corpus

def corpus(snapshot: str, out: str, n: int, t: int, src: str) -> int:
    """N training sequences of T tokens: BOS, then documents joined by blank lines (`windows.py`'s recipe)."""
    tok = transformers.AutoTokenizer.from_pretrained(snapshot)
    if src.endswith(".gz"):
        def texts():
            with gzip.open(src, "rt") as f:
                for line in f:
                    yield json.loads(line)["text"]
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from windows import read_text
        lines = read_text(src)

        def texts():
            yield "\n\n".join(lines)
    body, buf, docs = t - 1, [], 0
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    for text in texts():
        buf += tok(text, add_special_tokens=False)["input_ids"] + sep
        docs += 1
        if len(buf) >= n * body:
            break
    k = min(n, len(buf) // body)
    seqs = np.stack([np.concatenate([[tok.bos_token_id], buf[i * body:(i + 1) * body]]) for i in range(k)]).astype(np.int64)
    np.save(out, seqs)
    print(json.dumps({"sequences": int(k), "seq": t, "documents": docs, "source": os.path.basename(src)}))
    return 0 if k == n else 1


# ---------------------------------------------------------------------------------------------------------- selftest

def tiny(dev, attn="sdpa"):
    L = transformers.models.llama.modeling_llama
    torch.manual_seed(0)
    cfg = L.LlamaConfig(vocab_size=512, hidden_size=256, intermediate_size=448, num_hidden_layers=2, num_attention_heads=8,
                        num_key_value_heads=2, head_dim=32, max_position_embeddings=1024, tie_word_embeddings=False,
                        attn_implementation=attn)
    return L.LlamaForCausalLM(cfg).to(torch.bfloat16).to(dev).eval()


def selftest() -> dict:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out: dict = {"device": dev}
    g = torch.Generator().manual_seed(0)
    for k in (256, 448):
        x = torch.randn(40, k, generator=g).to(dev)
        dense = x.double() @ eg.rotation(k, dev)
        out[f"rotate_{k}"] = float(torch.linalg.norm(rotate(x).double() - dense) / torch.linalg.norm(dense))
        assert out[f"rotate_{k}"] < 1e-2, out
    model = tiny(dev)
    ids = torch.randint(0, 512, (2, 64), generator=g).to(dev)
    ref = {m: eg.evaluate(model, ids, m)["ppl"] for m in ("bf16", "int8@-", "int8_rot@wa:u2:s1", "int8_rot@a:u4:f")}
    mods = wrap(model, 4)
    for m in mods:
        m.build_wrot()
    mine = {m: eval_linears(model, mods, ids, m)["ppl"] for m in ref}
    out["eval_untuned"] = {"harness": ref, "here": mine, "identical": mine == ref}
    assert all(abs(mine[m] / ref[m] - 1) < 1e-5 for m in ref), out
    x = torch.randn(1, 50, 256, generator=g).to(dev).to(torch.bfloat16).requires_grad_(True)
    lin = mods[0]
    with torch.no_grad():
        lin.B.normal_(0, 0.02)
    set_noise("lin:wa:u1")
    MODE["lin"] = dict(MODE["lin"], edit="-", top_a=127, top_w=127)
    y = lin(x)
    yh, _ = eg.edited_product("int8_rot@-", x[0].float(), lin.merged(), {}, "t")
    out["noisy_linear_unedited_vs_harness"] = float(torch.linalg.norm(y[0].double() - yh) / torch.linalg.norm(yh))
    assert out["noisy_linear_unedited_vs_harness"] < 2e-2, out
    set_noise("lin:wa:u5")
    gy = torch.randn(1, 50, lin.base.weight.shape[0], generator=g).to(dev)
    (lin(x).float() * gy).sum().backward()
    gx, ga, gb = x.grad.clone(), lin.A.grad.clone(), lin.B.grad.clone()
    x.grad = lin.A.grad = lin.B.grad = None
    set_noise(None)
    (lin(x).float() * gy).sum().backward()
    for name, a, b in (("x", gx, x.grad), ("A", ga, lin.A.grad), ("B", gb, lin.B.grad)):
        out[f"ste_grad_{name}"] = float(torch.linalg.norm(a.float() - b.float()) / torch.linalg.norm(b.float()))
        assert out[f"ste_grad_{name}"] < 2e-2, out
    with torch.no_grad():
        lin.A.grad = lin.B.grad = None
        merged_ppl = eval_linears(model, mods, ids, "int8_rot@-")["ppl"]
    out["eval_merged_lora_int8_rot"] = merged_ppl
    set_noise("lin:wa:u5")
    for s in range(2):
        kl, ce = kd_step(model, ids[s])
        kl.backward()
        assert all(m.B.grad is not None for m in mods), "LoRA gets no gradient"
        out[f"kd_lin_{s}"] = [float(kl), ce]
    set_noise(None)
    for m in mods:
        m.A.grad = m.B.grad = None
    amodel = tiny(dev, "sdpa")
    aids = torch.randint(0, 512, (2, 96), generator=g).to(dev)
    amods = wrap(amodel, 4)
    for meth in ("bf16", "int8-pm1-rot", "int8-p=pm1-s1"):
        amodel.config._attn_implementation = "eager"
        want = float(np.mean([at.evaluate(amodel, r[None], meth)["nll"] for r in aids]))
        amodel.config._attn_implementation = "sdpa"
        got = eval_attention(amodel, aids, meth)["nll"]
        out[f"attn_eval_{meth}"] = [want, got]
        assert abs(want - got) < 1e-6, out
    set_noise("attn:int8-pm1-rot")
    kl, ce = kd_step(amodel, aids[0])
    kl.backward()
    assert MODE["calls"] == 2 and all(m.B.grad is not None for m in amods), MODE
    out["kd_attn"] = [float(kl), ce]
    set_noise(None)
    clean_kl, _ = kd_step(amodel, aids[0])
    out["kd_attn_clean"] = float(clean_kl)
    assert float(clean_kl) < 1e-6 < float(kl), out
    return out


# ------------------------------------------------------------------------------------------------------------- table

def table(rows: list[dict]) -> str:
    """Per config and step: perplexity (mean over seeds, min-max), Δ% against the untuned model's int8 grid and bf16."""
    base = {}
    for r in rows:
        if r["step"] == 0 and r["seed"] is None:
            base[r["config"], r["method"]] = r["ppl"]
    lines = ["| config | eval | step | seeds | perplexity | Δ% vs int8 | Δ% vs bf16 | recovered |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["config"], r["family"], r["step"], r["seed"] is None), []).append(r)
    for (cfg, fam, step, is_base), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][3], kv[0][1], kv[0][2])):
        p = [r["ppl"] for r in rs]
        mean = float(np.mean(p))
        lin = cfg.startswith("lin")
        i8 = base.get((cfg, "int8@-" if lin else "int8-none-rot"))
        bf = base.get((cfg, "bf16"))
        d = lambda b: f"{100 * (mean / b - 1):+.2f}" if b else "-"  # noqa: E731
        untuned = groups.get((cfg, fam, 0, False))
        rec = "-"
        if untuned and i8 and not is_base and step > 0:
            u = float(np.mean([r["ppl"] for r in untuned]))
            rec = f"{100 * (u - mean) / (u - i8):.0f}%" if u > i8 else "-"
        spread = f" ({min(p):.3f}-{max(p):.3f})" if len(p) > 1 else ""
        lines.append(f"| {cfg} | {fam} | {step} | {len(p)} | {mean:.4g}{spread} | {d(i8)} | {d(bf)} | {rec} |")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if argv[0] == "--selftest":
        print(json.dumps(selftest(), indent=1))
        return 0
    if argv[0] == "--corpus":
        return corpus(argv[1], argv[2], int(argv[3]), int(argv[4]), argv[5])
    if argv[0] == "--table":
        print(table([json.loads(line) for p in argv[1:] for line in Path(p).read_text().splitlines() if line.startswith("{")]))
        return 0
    return train(*argv[:5])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
