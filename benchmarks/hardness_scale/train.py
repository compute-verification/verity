"""A decoder-only transformer on FineWeb-Edu whose credited linears follow design A′ (`inbox/HARDNESS.md`, M5 at scale).

A credited linear (an R-product) computes, as served under A′:

    q = clamp(round(x / s), −126, 126),  s = absmax over the token of x / 126   (per-token int8)
    z = q + E,  E uniform on {−2, −1, 0, 1}, live, per element                  (2-bit noise, z is int8)
    y = D₁ ⊙ (z·Rᵀ) · s · c

R ∈ int8^{n_out×n_in} is uniform on [−128, 127], drawn once from the run's seed and frozen. D₁ is a trained output-side
scaling (the epilogue, outside the committed product z·Rᵀ), c = 1/(σ_R·√n_in) a constant. The means of E and R add
0.5·Σ_k z_k + 0.5·Σ_k R_jk + n/4 to an output, public O(n) terms the epilogue removes exactly, so the GEMM here takes
z + ½ and R + ½: half-integers in [−127.5, 127.5], exact in bf16, accumulated in fp32. Gradients reach x straight through
the rounding (STE). Eval runs under noise: the served model is never denoised. Nothing trained sits between E and R.

Every other block linear is trained with int8 QAT (per-row int8 weights, per-token int8 activations, STE). The embedding
and the tied head are bf16. QK-norm, RoPE, GELU MLP of width 4d, pre-RMSNorm, no biases.

Arrangements of the credited work (``arr``):

- ``none``: the dense int8-QAT baseline.
- ``matrix``: W = R + UVᵀ on the linears ``where`` names (``mlp``: fc1, fc2; ``all``: also QKV and O), in a share
  ``block_frac`` of the blocks; y = D₁ ⊙ R-product(x) + (x·Vᵀ)·Uᵀ, rank r = rank_frac·d, U = 0 at init. The adapter reads
  the bf16 x and is uncredited; y_R is committed alone (C4).
- ``reservoir``: a share ``frozen_frac`` of whole blocks have every linear an R-product (QKV, O, fc1, fc2), interleaved with
  fully trained blocks. Only their norm gains and D₁ train. The first and last blocks are always trained.
- ``feature``: in every MLP a share ``feature_share`` of the hidden units come from an R-product (credited); the rest of
  fc1 and all of fc2 are trained.
- ``noise``: the baseline with E added to the int8 codes of its MLP inputs (no R): what the noise alone costs.

Accounting (per token, forward MACs): credited R-products, trained linears, adapters, attention (QKᵀ and AV at the causal
average ctx/2 each) and the tied head. κ = credited / all. Training FLOPs are 6 a MAC, except 4 for a frozen R (no weight
gradient); ``train_flops_dense`` counts 6 for every MAC, as if the arm were a dense model of its forward MACs.

With ``teacher`` (a Hugging Face causal LM on the same tokenizer), training distils: the loss is (1 − kd_alpha)·CE +
kd_alpha·KL(teacher ‖ student) at temperature 1, and the teacher's forward FLOPs are reported apart (``teacher_flops``).
Logged, eval and final losses are always the student's cross-entropy under noise. ``gpus`` > 1 runs data-parallel under
torchrun: each rank takes its share of every step's micro-batches, and rank 0 evaluates and writes the result.
"""

import argparse
import dataclasses
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

R_STD = math.sqrt((256 ** 2 - 1) / 12)  # uniform on the 256 integers in [-128, 127]
EVAL_SEED = 777


@dataclasses.dataclass
class Cfg:
    name: str = "run"
    d: int = 384
    n_layer: int = 12
    head_dim: int = 64
    ctx: int = 1024
    vocab: int = 49152
    tokens: int = 0  # 0: 20 × the dense model's non-embedding parameters
    batch: int = 128  # sequences per optimizer step
    micro: int = 64  # sequences per forward
    lr: float = 3e-3
    wd: float = 0.1
    warmup_frac: float = 0.03
    final_lr_frac: float = 0.1
    beta2: float = 0.95
    clip: float = 1.0
    seed: int = 0
    data_seed: int = 1234
    arr: str = "none"  # none | matrix | reservoir | feature | noise
    where: str = "mlp"  # matrix: mlp | all
    block_frac: float = 1.0  # matrix: share of blocks hardened
    rank_frac: float = 0.125  # matrix: r = rank_frac · d
    frozen_frac: float = 0.0  # reservoir: share of blocks frozen
    feature_share: float = 0.0  # feature: share of MLP hidden units from R
    qat: bool = True  # int8 QAT on every trained block linear
    noise: bool = True  # live E on every R-product
    uv_lr_mult: float = 1.0
    evals: int = 8
    eval_seqs: int = 256
    final_eval_seqs: int = 2048
    ablate: bool = True  # final evals with R permuted and with R removed
    compile: bool = True
    synthetic: bool = False  # random tokens, for throughput only
    max_steps: int = 0  # stop early (smoke tests)
    gpus: int = 1  # data-parallel ranks (torchrun); each takes batch / gpus sequences a step
    teacher: str = ""  # a Hugging Face causal LM sharing the tokenizer: distil from it
    kd_alpha: float = 0.5  # with a teacher, the loss is (1 − kd_alpha)·CE + kd_alpha·KL(teacher ‖ student)


def nonemb_params(d, n_layer=12, head_dim=64):
    """The dense model's non-embedding parameters: 12d² a block in linears, two norm gains, QK-norm gains, final norm."""
    return n_layer * (12 * d * d + 2 * d + 2 * head_dim) + d


def spread(k, n_layer):
    """k block indices spread evenly over 1 .. n_layer − 2 (the first and last block stay trained)."""
    if k <= 0:
        return set()
    idx = {1 + int((j + 0.5) * (n_layer - 2) / k) for j in range(k)}
    if len(idx) != k:
        raise ValueError(f"cannot spread {k} blocks over {n_layer}")
    return idx


def noise_like(z):
    """E + ½ for E uniform on {−2, −1, 0, 1}."""
    return torch.randint(0, 4, z.shape, device=z.device, dtype=torch.int8).to(z.dtype) - 1.5


def fake_quant(x, lim):
    """Per-token absmax int8 codes in ±lim (fp32, straight-through) and their scale."""
    xf = x.float()
    s = xf.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12) / lim
    u = xf / s
    return u + (torch.round(u).clamp(-lim, lim) - u).detach(), s


class Trained(nn.Module):
    """A trained linear, int8 QAT when ``qat``; with ``noise``, E is added to its input codes (the noise-only control)."""

    def __init__(self, n_in, n_out, cfg: Cfg, gen, init_scale=1.0, noise=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in, generator=gen) * (init_scale / math.sqrt(n_in)))
        self.qat, self.noise = cfg.qat, noise
        self.n_in, self.n_out = n_in, n_out

    def forward(self, x):
        if not self.qat:
            return F.linear(x, self.weight)
        z, s = fake_quant(x, 126 if self.noise else 127)
        if self.noise:
            z = z + noise_like(z)
        w = self.weight
        ws = w.detach().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127
        wq = w + (torch.round(w / ws).clamp(-127, 127) * ws - w).detach()
        return F.linear(z * s, wq)


class RProduct(nn.Module):
    """A credited linear: y = D₁ ⊙ ((q + E)·Rᵀ)·s·c, R frozen uniform int8, plus a trained rank-r adapter if rank > 0."""

    def __init__(self, n_in, n_out, cfg: Cfg, gen, init_scale=1.0, rank=0):
        super().__init__()
        R = torch.randint(-128, 128, (n_out, n_in), generator=gen, dtype=torch.int16)
        self.register_buffer("R", R.to(torch.int8))
        self.register_buffer("Rc", (R.float() + 0.5).to(torch.bfloat16), persistent=False)
        self.c = 1.0 / (R_STD * math.sqrt(n_in))
        self.d1 = nn.Parameter(torch.full((n_out,), float(init_scale)))
        self.noise = cfg.noise
        self.U = self.V = None
        if rank:
            self.V = nn.Parameter(torch.randn(rank, n_in, generator=gen) / math.sqrt(n_in))
            self.U = nn.Parameter(torch.zeros(n_out, rank))
        self.n_in, self.n_out, self.rank = n_in, n_out, rank
        self.mode = None  # eager-only ablations: "zero" drops the R term, "perm" uses a permuted R
        self.share = None

    def forward(self, x):
        z, s = fake_quant(x, 126)
        z = z + (noise_like(z) if self.noise else 0.5)
        if self.mode == "zero":
            y = torch.zeros(*x.shape[:-1], self.n_out, device=x.device, dtype=torch.float32)
        else:
            Rc = self.Rc if self.mode != "perm" else self.Rc_perm
            y = F.linear(z.to(torch.bfloat16), Rc).float() * (s * self.c) * self.d1
        if self.U is None:
            return y
        a = F.linear(F.linear(x, self.V), self.U)
        if self.mode == "record":
            self.share = float(y.norm() / (y + a.float()).norm().clamp_min(1e-12))
        return y + a


class FeatureFC1(nn.Module):
    """fc1 with its first m_R hidden units from an R-product and the rest trained."""

    def __init__(self, d, hidden, share, cfg: Cfg, gen):
        super().__init__()
        m = round(share * hidden)
        self.r = RProduct(d, m, cfg, gen)
        self.t = Trained(d, hidden - m, cfg, gen) if hidden > m else None

    def forward(self, x):
        y = self.r(x)
        return y if self.t is None else torch.cat([y, self.t(x).float()], -1)


def make_linear(kind, n_in, n_out, cfg: Cfg, gen, init_scale=1.0):
    if kind == "T":
        return Trained(n_in, n_out, cfg, gen, init_scale)
    if kind == "N":
        return Trained(n_in, n_out, cfg, gen, init_scale, noise=True)
    if kind == "R":
        return RProduct(n_in, n_out, cfg, gen, init_scale)
    if kind == "RA":
        return RProduct(n_in, n_out, cfg, gen, init_scale, rank=max(1, round(cfg.rank_frac * cfg.d)))
    raise ValueError(kind)


def block_plan(cfg: Cfg):
    """Per block, the kind of each linear: T trained, R an R-product, RA an R-product with an adapter, N trained with noise,
    F (fc1 only) a feature split."""
    L = cfg.n_layer
    if cfg.arr == "reservoir":
        frozen = spread(round(cfg.frozen_frac * L), L)
    if cfg.arr == "matrix":
        hard_blocks = set(range(L)) if cfg.block_frac >= 1 else spread(round(cfg.block_frac * L), L)
        names = {"mlp": ("fc1", "fc2"), "all": ("qkv", "o", "fc1", "fc2")}[cfg.where]
    plan = []
    for i in range(L):
        p = {n: "T" for n in ("qkv", "o", "fc1", "fc2")}
        if cfg.arr == "matrix" and i in hard_blocks:
            p.update({n: "RA" for n in names})
        elif cfg.arr == "reservoir" and i in frozen:
            p = {n: "R" for n in p}
        elif cfg.arr == "feature":
            p["fc1"] = "F"
        elif cfg.arr == "noise":
            p["fc1"] = p["fc2"] = "N"
        elif cfg.arr not in ("none", "matrix", "reservoir"):
            raise ValueError(cfg.arr)
        plan.append(p)
    return plan


def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, -1)
    return x * cos + torch.cat((-x2, x1), -1) * sin


class Block(nn.Module):
    def __init__(self, cfg: Cfg, li: int, plan: dict):
        super().__init__()
        d, hd = cfg.d, cfg.head_dim
        self.H = d // hd
        rs = 1 / math.sqrt(2 * cfg.n_layer)
        gen = torch.Generator().manual_seed(cfg.seed * 100_003 + li * 101 + 7)
        self.ln1, self.ln2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.qn, self.kn = nn.Parameter(torch.ones(hd)), nn.Parameter(torch.ones(hd))
        self.qkv = make_linear(plan["qkv"], d, 3 * d, cfg, gen)
        self.o = make_linear(plan["o"], d, d, cfg, gen, rs)
        if plan["fc1"] == "F":
            self.fc1 = FeatureFC1(d, 4 * d, cfg.feature_share, cfg, gen)
        else:
            self.fc1 = make_linear(plan["fc1"], d, 4 * d, cfg, gen)
        self.fc2 = make_linear(plan["fc2"], 4 * d, d, cfg, gen, rs)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        h = F.rms_norm(x, (d,), self.ln1, 1e-6)
        q, k, v = self.qkv(h).split(d, -1)
        q, k, v = (t.view(B, T, self.H, -1).transpose(1, 2) for t in (q, k, v))
        q = rotate(F.rms_norm(q, (q.shape[-1],), self.qn, 1e-6), cos, sin)
        k = rotate(F.rms_norm(k, (k.shape[-1],), self.kn, 1e-6), cos, sin)
        y = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, d))
        h = F.rms_norm(x, (d,), self.ln2, 1e-6)
        return x + self.fc2(F.gelu(self.fc1(h), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed * 100_003 + 99_991)
        self.emb = nn.Parameter(torch.randn(cfg.vocab, cfg.d, generator=g) * 0.02)
        self.plan = block_plan(cfg)
        self.blocks = nn.ModuleList(Block(cfg, i, p) for i, p in enumerate(self.plan))
        self.lnf = nn.Parameter(torch.ones(cfg.d))
        hd = cfg.head_dim
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(cfg.ctx).float()[:, None] * inv[None]
        ang = torch.cat([ang, ang], -1)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, idx, targets, teacher_logits=None, kd_alpha=0.0):
        """The mean cross-entropy; with teacher logits, also the distillation loss (1 − kd_alpha)·CE + kd_alpha·KL."""
        T = idx.shape[1]
        x = F.embedding(idx, self.emb)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = F.linear(F.rms_norm(x, (x.shape[-1],), self.lnf, 1e-6), self.emb).float().flatten(0, 1)
        ce = F.cross_entropy(logits, targets.flatten())
        if teacher_logits is None:
            return ce
        kl = F.kl_div(F.log_softmax(logits, -1), F.log_softmax(teacher_logits.float().flatten(0, 1), -1),
                      log_target=True, reduction="batchmean")
        return (1 - kd_alpha) * ce + kd_alpha * kl, ce


def load_teacher(name: str):
    """A frozen bf16 causal LM: from /workspace/hf/hub when cached there, else the default Hugging Face cache."""
    from transformers import AutoModelForCausalLM
    kw = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    try:
        m = AutoModelForCausalLM.from_pretrained(name, cache_dir="/workspace/hf/hub", local_files_only=True, **kw)
    except OSError:
        m = AutoModelForCausalLM.from_pretrained(name, **kw)
    m = m.cuda().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def accounting(model: GPT, cfg: Cfg):
    credited = trained = adapters = 0
    for m in model.modules():
        if isinstance(m, RProduct):
            credited += m.n_in * m.n_out
            adapters += m.rank * (m.n_in + m.n_out)
        elif isinstance(m, Trained):
            trained += m.n_in * m.n_out
    attn = cfg.n_layer * cfg.ctx * cfg.d
    head = cfg.vocab * cfg.d
    fwd = credited + trained + adapters + attn + head
    return {
        "macs_fwd_per_token": fwd,
        "macs_credited_per_token": credited,
        "macs_trained_linear_per_token": trained,
        "macs_adapter_per_token": adapters,
        "macs_attn_per_token": attn,
        "macs_head_per_token": head,
        "kappa": credited / fwd,
        "kappa_lin": credited / (credited + trained),
        "train_flops_per_token": 6 * (trained + adapters + attn + head) + 4 * credited,
        "train_flops_dense_per_token": 6 * fwd,
    }


class Data:
    """Training windows of ctx + 1 tokens at stride ctx in every train shard, in one permutation fixed by ``data_seed``:
    a run reads windows [step·batch, (step + 1)·batch) of it, so every run sees the same data in the same order."""

    def __init__(self, root: Path, cfg: Cfg):
        meta = json.loads((root / "meta.json").read_text())
        if not meta.get("complete"):
            raise RuntimeError(f"{root}: prep is not complete")
        self.ctx = cfg.ctx
        self.train = [np.memmap(root / s["file"], dtype=np.uint16, mode="r") for s in meta["train"]]
        self.val = [np.memmap(root / s["file"], dtype=np.uint16, mode="r") for s in meta["val"]]
        sh, off = [], []
        for j, m in enumerate(self.train):
            n = (len(m) - 1) // cfg.ctx
            sh.append(np.full(n, j, np.int16))
            off.append(np.arange(n, dtype=np.int64) * cfg.ctx)
        sh, off = np.concatenate(sh), np.concatenate(off)
        perm = np.random.default_rng(cfg.data_seed).permutation(len(sh))
        self.sh, self.off = sh[perm], off[perm]
        vw = [(j, o) for j, m in enumerate(self.val) for o in range(0, len(m) - cfg.ctx - 1, cfg.ctx)]
        pick = np.linspace(0, len(vw) - 1, cfg.final_eval_seqs).astype(np.int64)
        self.val_windows = [vw[i] for i in pick]
        self.meta = meta

    def windows(self):
        return len(self.sh)

    def train_batch(self, start, n):
        if start + n > len(self.sh):
            raise RuntimeError(f"run needs window {start + n}, the corpus has {len(self.sh)}")
        rows = [self.train[self.sh[i]][self.off[i]:self.off[i] + self.ctx + 1] for i in range(start, start + n)]
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    def val_batch(self, idx):
        rows = [self.val[j][o:o + self.ctx + 1] for j, o in (self.val_windows[i] for i in idx)]
        return torch.from_numpy(np.stack(rows).astype(np.int64))


class Synthetic:
    def __init__(self, cfg: Cfg):
        self.cfg, self.g = cfg, torch.Generator().manual_seed(cfg.data_seed)

    def windows(self):
        return 1 << 40

    def train_batch(self, start, n):
        return torch.randint(0, self.cfg.vocab, (n, self.cfg.ctx + 1), generator=self.g)

    def val_batch(self, idx):
        return torch.randint(0, self.cfg.vocab, (len(idx), self.cfg.ctx + 1), generator=self.g)


def param_groups(model: GPT, cfg: Cfg):
    mats, vecs, uv = [], [], []
    for n, p in model.named_parameters():
        if n.endswith(".U") or n.endswith(".V"):
            uv.append(p)
        elif p.dim() >= 2:
            mats.append(p)
        else:
            vecs.append(p)
    groups = [{"params": mats, "weight_decay": cfg.wd, "lr_mult": 1.0},
              {"params": vecs, "weight_decay": 0.0, "lr_mult": 1.0}]
    if uv:
        groups.append({"params": uv, "weight_decay": cfg.wd, "lr_mult": cfg.uv_lr_mult})
    return groups


@torch.no_grad()
def evaluate(fwd, data, cfg: Cfg, n_seqs: int):
    """Mean validation loss (nats a token) over n_seqs fixed windows, under noise with a fixed seed."""
    if n_seqs % cfg.micro or cfg.final_eval_seqs % n_seqs:
        raise ValueError(f"eval of {n_seqs} sequences: a multiple of micro {cfg.micro} dividing {cfg.final_eval_seqs}")
    state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(EVAL_SEED)
    idx = list(range(0, cfg.final_eval_seqs, cfg.final_eval_seqs // n_seqs))
    tot = torch.zeros((), device="cuda")
    for i in range(0, len(idx), cfg.micro):
        xy = data.val_batch(idx[i:i + cfg.micro]).cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot += fwd(xy[:, :-1], xy[:, 1:])
    torch.cuda.set_rng_state(state)
    return float(tot) * cfg.micro / len(idx)


@torch.no_grad()
def ablations(model: GPT, data, cfg: Cfg):
    """Eager evals with every R-product's R permuted and removed, and each adapter-carrying R-product's share of its
    output norm: whether the model reads R."""
    rps = [m for m in model.modules() if isinstance(m, RProduct)]
    if not rps:
        return {}
    n = cfg.eval_seqs
    out = {}
    for m in rps:
        m.mode = "record"
    out["val_record"] = evaluate(model, data, cfg, cfg.micro)
    shares = {}
    for name, m in model.named_modules():
        if isinstance(m, RProduct) and m.share is not None:
            shares.setdefault(name.split(".")[-1], []).append(m.share)
    out["r_share_out"] = {k: float(np.mean(v)) for k, v in shares.items()}
    g = torch.Generator(device="cuda").manual_seed(4242)
    for m in rps:
        m.Rc_perm = m.Rc.flatten()[torch.randperm(m.Rc.numel(), generator=g, device="cuda")].view_as(m.Rc)
        m.mode = "perm"
    out["val_r_permuted"] = evaluate(model, data, cfg, n)
    for m in rps:
        m.mode = "zero"
    out["val_r_removed"] = evaluate(model, data, cfg, n)
    for m in rps:
        m.mode, m.Rc_perm = None, None
    out["val_eager"] = evaluate(model, data, cfg, n)
    return out


@torch.no_grad()
def diagnostics(model: GPT):
    d1 = [m.d1.float() for m in model.modules() if isinstance(m, RProduct)]
    if not d1:
        return {}
    d1 = torch.cat(d1)
    return {"d1_rms": float(d1.pow(2).mean().sqrt()), "d1_absmin": float(d1.abs().min()),
            "d1_frac_small": float((d1.abs() < 1e-3).float().mean())}


def all_mean(tensors, world):
    """Average tensors over the ranks in place, as one flat all-reduce."""
    if world == 1 or not tensors:
        return
    flat = _flatten_dense_tensors(tensors)
    dist.all_reduce(flat)
    flat /= world
    for t, f in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
        t.copy_(f)


def train(cfg: Cfg, data_dir: Path | None):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world != cfg.gpus:
        raise ValueError(f"config asks for {cfg.gpus} ranks, {world} running")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed + 1000 * rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not cfg.tokens:
        cfg.tokens = 20 * nonemb_params(cfg.d, cfg.n_layer, cfg.head_dim)
    data = Synthetic(cfg) if cfg.synthetic else Data(data_dir, cfg)
    if not cfg.synthetic and data.meta["vocab"] != cfg.vocab:
        raise ValueError(f"vocab {cfg.vocab} != the data's {data.meta['vocab']}")
    model = GPT(cfg).cuda()
    acct = accounting(model, cfg)
    teacher = load_teacher(cfg.teacher) if cfg.teacher else None
    if teacher is not None:
        if teacher.config.vocab_size != cfg.vocab:
            raise ValueError(f"teacher vocab {teacher.config.vocab_size} != {cfg.vocab}")
        acct["teacher_params"] = sum(p.numel() for p in teacher.parameters())
        acct["teacher_flops_per_token"] = 2 * acct["teacher_params"]
    n_train = sum(p.numel() for p in model.parameters())
    steps = max(1, cfg.tokens // (cfg.batch * cfg.ctx))
    if steps * cfg.batch > data.windows():
        raise RuntimeError(f"{steps * cfg.batch} windows needed, {data.windows()} in the corpus")
    accum = cfg.batch // (cfg.micro * world)
    if accum * cfg.micro * world != cfg.batch:
        raise ValueError("batch must be a multiple of micro × gpus")
    opt = torch.optim.AdamW(param_groups(model, cfg), lr=cfg.lr, betas=(0.9, cfg.beta2), eps=1e-8, fused=True)
    fwd = torch.compile(model) if cfg.compile else model
    warm = max(1, round(cfg.warmup_frac * steps))
    head = {"cfg": dataclasses.asdict(cfg), "steps": steps, "trainable_params": n_train,
            "nonemb_trainable": n_train - model.emb.numel(), "nonemb_dense": nonemb_params(cfg.d, cfg.n_layer),
            "plan": ["".join(p[n][0] if p[n] != "RA" else "A" for n in ("qkv", "o", "fc1", "fc2")) for p in model.plan],
            **acct}
    if rank == 0:
        print(json.dumps(head), flush=True)
    params = [p for p in model.parameters() if p.requires_grad]

    def lr_at(s):
        if s < warm:
            return (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return cfg.final_lr_frac + (1 - cfg.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))

    eval_at = {round(steps * (k + 1) / cfg.evals) for k in range(cfg.evals - 1)}
    curve, ema, t0 = [], None, time.time()
    run_steps = min(steps, cfg.max_steps) if cfg.max_steps else steps
    log_every = max(1, run_steps // 100)
    for s in range(run_steps):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * g["lr_mult"] * lr_at(s)
        tot = torch.zeros((), device="cuda")
        for a in range(accum):
            start = s * cfg.batch + (rank * accum + a) * cfg.micro
            xy = data.train_batch(start, cfg.micro).pin_memory().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if teacher is None:
                    loss = ce = fwd(xy[:, :-1], xy[:, 1:])
                else:
                    with torch.no_grad():
                        tl = teacher(input_ids=xy[:, :-1]).logits
                    loss, ce = fwd(xy[:, :-1], xy[:, 1:], tl, cfg.kd_alpha)
            (loss / accum).backward()
            tot += ce.detach()
        all_mean([p.grad for p in params if p.grad is not None] + [tot], world)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (s + 1) % log_every == 0 or s + 1 in eval_at or s + 1 == run_steps:
            lv = float(tot) / accum
            if not math.isfinite(lv):
                raise FloatingPointError(f"loss {lv} at step {s + 1}")
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            el = time.time() - t0
            tok = (s + 1) * cfg.batch * cfg.ctx
            rec = {"step": s + 1, "tokens": tok, "train": round(lv, 4), "train_ema": round(ema, 4),
                   "lr": cfg.lr * lr_at(s), "gnorm": round(float(gn), 3), "tok_s": round(tok / el),
                   "wall_s": round(el, 1), "mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)}
            if s + 1 in eval_at:
                if rank == 0:
                    rec["val"] = round(evaluate(fwd, data, cfg, cfg.eval_seqs), 4)
                    curve.append(rec)
                if world > 1:
                    dist.barrier()
            if rank == 0:
                print(json.dumps(rec), flush=True)
    t_train = time.time() - t0
    if rank:
        dist.barrier()
        return None
    final = evaluate(fwd, data, cfg, cfg.final_eval_seqs)
    curve.append({"step": run_steps, "tokens": run_steps * cfg.batch * cfg.ctx, "val": final})
    abl = ablations(model, data, cfg) if cfg.ablate else {}
    tokens = run_steps * cfg.batch * cfg.ctx
    res = {
        **head,
        "tokens": tokens,
        "train_flops": acct["train_flops_per_token"] * tokens,
        "train_flops_dense": acct["train_flops_dense_per_token"] * tokens,
        "final_val": final,
        "final_eval_tokens": cfg.final_eval_seqs * cfg.ctx,
        "curve": curve,
        "ablations": abl,
        "diag": diagnostics(model),
        "train_wall_s": t_train,
        "wall_s": time.time() - t0,
        "tok_per_s": tokens / t_train,
        "mfu_dense_bf16": acct["train_flops_dense_per_token"] * tokens / t_train / 503.8e12 / world,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "data": None if cfg.synthetic else {k: data.meta[k] for k in ("source", "tokenizer", "train_tokens", "val_tokens")},
    }
    if teacher is not None:
        res["teacher_flops"] = acct["teacher_flops_per_token"] * tokens
        res["teacher_val"] = evaluate(lambda x, y: F.cross_entropy(teacher(input_ids=x).logits.float().flatten(0, 1),
                                                                   y.flatten()), data, cfg, cfg.eval_seqs)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON object of Cfg fields")
    ap.add_argument("--data", default=os.environ.get("HSC_DATA"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg(**json.loads(a.config))
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))  # the other ranks wait out rank 0's evals
    res = train(cfg, Path(a.data) if a.data else None)
    if res is None:
        dist.destroy_process_group()
        return
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.rename(a.out)
    print(f"final {cfg.name} val {res['final_val']:.4f} tok/s {res['tok_per_s']:.0f} wall {res['wall_s']:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
