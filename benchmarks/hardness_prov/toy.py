"""HT-PUBLIC's toy GPT (`hardness_public/seeds.py`) with its width, depth and heads as parameters, live per-step coins,
and the hardening step that consumes them.

The model, init, data (HT-ML's TinyStories BPE-4096 cache) and pretraining recipe are seeds.py's arm A_lo: AdamW at 1e-3,
100 warmup steps, cosine to 1% of the peak, batch 16, context 128, no weight decay, init seed 1, data seed 1.

**Live coins.** Step t's coins are SHA-512(beacon ‖ t): the batch's offsets into the committed pool, the dropout masks,
the activation noise and the weight noise. A party that doesn't hold the beacon can't compute step t's coins; a planner
simulating the future draws its own beacons.

**The hardening step** (`Hardening.step`), W_{t+1} = Step(W_t; coins_t, batch_t):

- forward with dropout on both residual branches and Gaussian noise on every linear's input (σ_a times the input's rms);
- AdamW written out (`Adam`), its state reset at the pass's start, so a committed start can't carry moments that freeze
  the pass; the step is absolute (lr per entry) or, with `rel`, η of each row's served LSBs per entry on the block
  linears (lr on the rest), which a function-preserving rescaling of the start can't slow;
- weight noise: every block linear gets σ_w served LSBs of fresh Gaussian noise a row, after the update.

The served matrix is `quant`'s int8 codes of the final weights: `absmax` (per row max|w|/127, the stock format) or `mad`
(per row 1.4826·MAD/40, clipped at ±127: a scale no outlier can set, with codes spread to σ ≈ 40).
"""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LINEARS = ("qkv", "proj", "fc", "out")
CTX, VOCAB = 128, 4096
MAD_SIGMA = 40.0


class DecoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, noise):
        b, t, d = x.shape
        h = noise.act(self.ln1(x))
        q, k, v = self.qkv(h).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + noise.drop(self.proj(noise.act(y)))
        h = F.gelu(self.fc(noise.act(self.ln2(x))))
        return x + noise.drop(self.out(noise.act(h)))


class GPT(nn.Module):
    def __init__(self, d=192, layers=4, heads=4):
        super().__init__()
        self.d, self.layers, self.nheads = d, layers, heads
        self.emb = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(DecoderLayer(d, heads) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "out.weight")) else 0.02
                nn.init.normal_(p, std=std)

    def forward(self, idx, noise=None):
        noise = noise or Quiet()
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1]))
        for blk in self.blocks:
            x = blk(x, noise)
        return self.lnf(x) @ self.emb.weight.T

    def linear(self, name):
        i, n = name.split(".")
        return getattr(self.blocks[int(i)], n).weight


def names(model, which=LINEARS):
    return [f"{i}.{n}" for i in range(model.layers) for n in which]


class Quiet:
    def act(self, x):
        return x

    def drop(self, x):
        return x


class Noise:
    """Dropout and activation noise from one step's torch generator."""

    def __init__(self, gen, p_drop, sigma_a):
        self.g, self.p, self.s = gen, p_drop, sigma_a

    def act(self, x):
        if not self.s:
            return x
        rms = x.detach().pow(2).mean().sqrt()
        return x + self.s * rms * torch.randn(x.shape, generator=self.g)

    def drop(self, x):
        if not self.p:
            return x
        return x * (torch.rand(x.shape, generator=self.g) >= self.p).to(x.dtype) / (1 - self.p)


def coins(beacon: bytes, t: int):
    """Step t's coins: a numpy generator (the batch) and a torch generator (dropout, activation and weight noise)."""
    h = hashlib.sha512(beacon + t.to_bytes(8, "little")).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little"))
    return np.random.default_rng(int.from_bytes(h[8:16], "little")), g


def beacon(tag: str) -> bytes:
    return hashlib.sha512(tag.encode()).digest()


class Pool:
    def __init__(self, data_dir):
        self.train = np.memmap(Path(data_dir) / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(Path(data_dir) / "val.bin", dtype=np.uint16, mode="r")

    def batch(self, rng, n, split="train"):
        arr = self.train if split == "train" else self.val
        off = rng.integers(0, len(arr) - CTX - 1, size=n)
        x = np.stack([arr[o:o + CTX + 1] for o in off]).astype(np.int64)
        return torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])

    def val_loss(self, model, batches=32):
        g = np.random.default_rng(999)
        with torch.no_grad():
            return float(np.mean([F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1)).item()
                                  for x, y in (self.batch(g, 16, "val") for _ in range(batches))]))


def pretrain(pool, d, layers, heads, steps=1500, batch=16, floor=0.01, log=print):
    """seeds.py's arm A_lo at width d: returns the model and its log."""
    torch.manual_seed(1)
    model = GPT(d, layers, heads)
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and "blocks" in n]
    rest = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "blocks" in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.0}, {"params": rest, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95))
    rng = np.random.default_rng(1)
    for step in range(steps):
        lr = 1e-3 * min(1.0, (step + 1) / 100) * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = pool.batch(rng, batch)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0 or step == steps - 1:
            log(f"pretrain d={d} step {step} loss {loss.item():.4f} lr {lr:.2e}")
    return model


def scale(w, quant):
    """Per-row served LSB of float rows w (torch or numpy)."""
    a = np.asarray(w, dtype=np.float64)
    if quant == "absmax":
        s = np.abs(a).max(axis=1) / 127.0
    elif quant == "mad":
        med = np.median(a, axis=1, keepdims=True)
        s = 1.4826 * np.median(np.abs(a - med), axis=1) / MAD_SIGMA
        s = np.where(s > 0, s, np.abs(a).max(axis=1) / 127.0)
    else:
        raise ValueError(quant)
    return np.where(s > 0, s, 1.0)


def codes(w, quant):
    s = scale(w, quant)
    return np.clip(np.rint(np.asarray(w, dtype=np.float64) / s[:, None]), -127, 127).astype(np.int32), s


@dataclass
class Config:
    lr: float = 1e-5
    sigma_w: float = 0.0
    sigma_a: float = 0.05
    p_drop: float = 0.1
    batch: int = 16
    quant: str = "absmax"
    rel: bool = False
    eta: float = 0.0
    wd: float = 0.0

    def tag(self):
        return (f"lr{self.lr:g}_sw{self.sigma_w:g}_sa{self.sigma_a:g}_pd{self.p_drop:g}_b{self.batch}_{self.quant}"
                + (f"_eta{self.eta:g}" if self.rel else "") + (f"_wd{self.wd:g}" if self.wd else ""))


class Adam:
    """AdamW (β 0.9, 0.95, ε 1e-8) with state from zero. With `rel` a block linear's step is η times its row's served
    LSB, so the step is the same number of LSBs whatever scale the committed start has."""

    def __init__(self, model, cfg):
        self.cfg = cfg
        self.params = [(n, p) for n, p in model.named_parameters()]
        self.m = {n: torch.zeros_like(p) for n, p in self.params}
        self.v = {n: torch.zeros_like(p) for n, p in self.params}
        self.t = 0
        self.rel = {n for n, p in self.params if p.dim() == 2 and "blocks" in n} if cfg.rel else set()

    @torch.no_grad()
    def step(self):
        self.t += 1
        b1, b2 = 0.9, 0.95
        for n, p in self.params:
            g = p.grad
            if g is None:
                continue
            self.m[n].mul_(b1).add_(g, alpha=1 - b1)
            self.v[n].mul_(b2).addcmul_(g, g, value=1 - b2)
            d = (self.m[n] / (1 - b1 ** self.t)) / ((self.v[n] / (1 - b2 ** self.t)).sqrt() + 1e-8)
            lr = self.cfg.lr
            if n in self.rel:
                d = d * torch.from_numpy(scale(p.detach().numpy(), self.cfg.quant)).float()[:, None]
                lr = self.cfg.eta
            if self.cfg.wd and p.dim() == 2 and "blocks" in n:
                p.mul_(1 - self.cfg.lr * self.cfg.wd)
            p.add_(d, alpha=-lr)


class Hardening:
    """The live-coin pass from a committed start: `step(t)` is Step(W_t; coins_t, batch_t).

    `steer` (name → (target, step size)) is the adaptive cheater who controls those matrices' gradients but not their
    update rule: having seen step t's coins, he moves each entry by at most the step size toward the target, aiming at
    where the step's weight noise will put it, in place of the honest update."""

    def __init__(self, model, pool, cfg, beacon_bytes, steer=None):
        self.model, self.pool, self.cfg, self.beacon = model, pool, cfg, beacon_bytes
        self.opt = Adam(model, cfg)
        self.names = names(model)
        self.lin = [model.linear(n) for n in self.names]
        self.steer = steer or {}

    def lsb(self, w):
        return torch.from_numpy(scale(w.detach().numpy(), self.cfg.quant)).float()[:, None]

    def step(self, t):
        rng, g = coins(self.beacon, t)
        x, y = self.pool.batch(rng, self.cfg.batch)
        loss = F.cross_entropy(self.model(x, Noise(g, self.cfg.p_drop, self.cfg.sigma_a)).reshape(-1, VOCAB),
                               y.reshape(-1))
        self.model.zero_grad(set_to_none=True)
        loss.backward()
        z = [torch.randn(w.shape, generator=g) for w in self.lin] if self.cfg.sigma_w else None
        before = {n: self.model.linear(n).detach().clone() for n in self.steer}
        self.opt.step()
        with torch.no_grad():
            for n, (target, size) in self.steer.items():
                w0 = before[n]
                aim = w0 + (self.cfg.sigma_w * self.lsb(w0) * z[self.names.index(n)] if z else 0)
                size = torch.as_tensor(size, dtype=w0.dtype)
                self.model.linear(n).copy_(w0 - torch.maximum(torch.minimum(aim - target, size), -size))
            if z:
                for w, zi in zip(self.lin, z):
                    w.add_(self.cfg.sigma_w * self.lsb(w) * zi)
        return loss.item()


def served_val(model, pool, quant, which=LINEARS):
    """Validation loss with every listed linear replaced by its dequantized codes."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for n in names(model, which):
            w = model.linear(n)
            q, s = codes(w.detach().numpy(), quant)
            w.copy_(torch.from_numpy((q * s[:, None]).astype(np.float32)))
        v = pool.val_loss(model)
    model.load_state_dict(sd)
    return v
