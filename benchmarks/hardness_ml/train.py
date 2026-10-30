"""A small decoder-only transformer on TinyStories with hardened linears.

A hardened linear computes, per design:

- ``dense``: y = x̃·Wᵀ, W trainable (the baseline; with noise it is the noise-only control).
- ``A``: y = D₁ ⊙ (x̃·Rᵀ) + (x̃·V)·Uᵀ, R frozen dense random, U, V trainable rank r, D₁, D₂ optional learned diagonals
  (x̃ is formed from D₂ ⊙ x).
- ``B``: y = x̃·(R + Δ)ᵀ with ‖Δ‖_F ≤ ρ‖R‖_F, enforced by projection after every step.
- ``lowrank``: y = (x̃·V)·Uᵀ (A without R).
- ``frozen``: y = D₁ ⊙ (x̃·Rᵀ) (A with r = 0).
- ``lsb``: y = s ⊙ (x̃·Cᵀ), C an int8 code matrix whose low b bits (``lsb_bits``) are public uniform random L and whose
  high 8 − b bits are trained: C = 2ᵇ·H + L, H = round((W/s − L)/2ᵇ) from a latent W (straight-through), s a per-row
  scale. b = 0 is plain int8 weight training.

x̃ is x itself, or, with ``int8``, x quantized per token to the integers in ±(127 − k) (straight-through), plus live noise
E uniform on the integers in [−k, k] (``noise_k``), so x̃ + E is an int8 operand; R is then uniform int8 times a scale.
Eval runs under noise too: the served model is never denoised.
"""

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {  # d_model, n_layer, n_head
    "t128": (128, 4, 4),
    "s192": (192, 5, 4),
    "s256": (256, 6, 4),
    "m384": (384, 8, 6),
    "l512": (512, 8, 8),
}
INT8_STD = math.sqrt((255 ** 2 - 1) / 12)  # uniform on the integers in [-127, 127]


@dataclasses.dataclass
class Cfg:
    name: str = "run"
    size: str = "s256"
    ctx: int = 256
    tokens: int = 20_000_000
    batch: int = 32
    lr: float = 2e-3
    wd: float = 0.1
    warmup: int = 200
    seed: int = 0
    data_seed: int = 1234
    harden: str = "mlp"  # none | mlp | attn | all
    design: str = "dense"  # dense | A | B | lowrank | frozen
    rank_frac: float = 0.125  # r = rank_frac * d_model
    diag: bool = False
    rho: float = 0.25
    lsb_bits: int = 4
    noise_k: int = 0
    int8: bool = False
    r_dist: str = "gauss"  # gauss | int8
    r_scale: float = 1.0
    uv_lr_mult: float = 1.0
    eval_every: int = 250
    eval_batches: int = 16
    final_eval_batches: int = 128
    threads: int = 4
    bf16: bool = True


class Noise:
    gen: torch.Generator | None = None


def hlinear_kind(cfg: Cfg, where: str) -> bool:
    return cfg.harden == "all" or cfg.harden == where


class HLinear(nn.Module):
    def __init__(self, n_in, n_out, cfg: Cfg, hard: bool, seed: int, init_scale: float = 1.0):
        super().__init__()
        self.design = cfg.design if hard else "dense"
        self.int8 = cfg.int8 and hard
        self.k = cfg.noise_k if hard else 0
        self.hard = hard
        g = torch.Generator().manual_seed(seed)
        std = init_scale / math.sqrt(n_in)
        if self.hard and cfg.r_dist == "int8":
            R = torch.randint(-127, 128, (n_out, n_in), generator=g).float() * (std / INT8_STD)
        else:
            R = torch.randn(n_out, n_in, generator=g) * std
        if self.hard:
            R = R * cfg.r_scale
        self.register_buffer("R", R)
        self.U = self.V = self.d1 = self.d2 = self.weight = self.delta = None
        if self.design in ("dense", "lsb"):
            self.weight = nn.Parameter(R.clone())
        if self.design == "lsb":
            self.step = 2 ** cfg.lsb_bits
            lo = torch.randint(0, self.step, (n_out, n_in), generator=torch.Generator().manual_seed(seed + 7919))
            self.register_buffer("lo", lo.float())
        elif self.design == "B":
            self.delta = nn.Parameter(torch.zeros_like(R))
            self.rho = cfg.rho
        if self.design in ("A", "lowrank"):
            r = max(1, round(cfg.rank_frac * cfg_d(cfg)))
            self.V = nn.Parameter(torch.randn(r, n_in, generator=g) / math.sqrt(n_in))
            self.U = nn.Parameter(torch.zeros(n_out, r))
        if self.design in ("A", "frozen") and cfg.diag:
            self.d1 = nn.Parameter(torch.ones(n_out))
            self.d2 = nn.Parameter(torch.ones(n_in))
        if self.design == "frozen" and not cfg.diag:
            raise ValueError("design frozen needs diag")
        self.n_in, self.n_out = n_in, n_out
        self.last_share = None

    def macs(self):
        """Forward MACs per token: (hardened R-product, trainable, frozen non-R)."""
        nm = self.n_in * self.n_out
        uv = 0 if self.U is None else self.U.shape[1] * (self.n_in + self.n_out)
        if self.design in ("dense", "B", "lsb"):
            return (nm if self.hard else 0), nm, 0
        if self.design == "lowrank":
            return 0, uv, 0
        return nm, uv, 0  # A, frozen: R is frozen, so its backward is the input gradient only

    def prep(self, x):
        if self.d2 is not None:
            x = x * self.d2
        if not self.int8 and self.k == 0:
            return x
        x = x.float()
        if self.int8:
            s = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-6) / (127 - self.k)
            q = torch.round(x / s).clamp(-(127 - self.k), 127 - self.k)
            x = x + (q * s - x).detach()
        else:
            s = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-6) / 127
        if self.k:
            e = torch.randint(-self.k, self.k + 1, x.shape, generator=Noise.gen, dtype=torch.int16)
            x = x + e.to(x.dtype) * s
        return x

    def forward(self, x, record=False):
        x = self.prep(x)
        if self.design == "dense":
            return F.linear(x, self.weight)
        if self.design == "B":
            return F.linear(x, self.R + self.delta)
        if self.design == "lsb":
            return F.linear(x, self.lsb_weight())
        y_r = None
        if self.design in ("A", "frozen"):
            y_r = F.linear(x, self.R)
            if self.d1 is not None:
                y_r = y_r * self.d1
        y_uv = None if self.U is None else F.linear(F.linear(x, self.V), self.U)
        if record and y_r is not None:
            tot = y_r if y_uv is None else y_r + y_uv
            self.last_share = float(y_r.float().norm() / tot.float().norm().clamp_min(1e-12))
        if y_r is None:
            return y_uv
        return y_r if y_uv is None else y_r + y_uv

    def lsb_weight(self):
        w = self.weight.float()
        s = w.detach().abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        hi = torch.round((w.detach() / s - self.lo) / self.step).clamp(-128 // self.step, 128 // self.step - 1)
        return w + ((self.step * hi + self.lo) * s - w).detach()

    @torch.no_grad()
    def lsb_codes(self):
        w = self.weight.float()
        s = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        hi = torch.round((w / s - self.lo) / self.step).clamp(-128 // self.step, 128 // self.step - 1)
        return self.step * hi + self.lo

    @torch.no_grad()
    def project(self):
        if self.design == "B":
            cap = self.rho * self.R.norm()
            n = self.delta.norm()
            if n > cap:
                self.delta.mul_(cap / n)


def cfg_d(cfg):
    return SIZES[cfg.size][0]


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.g)


def rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: Cfg, li: int):
        super().__init__()
        d, L, H = SIZES[cfg.size]
        self.H = H
        rs = 1 / math.sqrt(2 * L)
        base = cfg.seed * 1000 + li * 10
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.qkv = HLinear(d, 3 * d, cfg, hlinear_kind(cfg, "attn"), base + 1)
        self.o = HLinear(d, d, cfg, hlinear_kind(cfg, "attn"), base + 2, rs)
        self.fc1 = HLinear(d, 4 * d, cfg, hlinear_kind(cfg, "mlp"), base + 3)
        self.fc2 = HLinear(4 * d, d, cfg, hlinear_kind(cfg, "mlp"), base + 4, rs)

    def forward(self, x, cos, sin, record=False):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x), record).split(d, -1)
        q, k, v = (t.view(B, T, self.H, d // self.H).transpose(1, 2) for t in (q, k, v))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + self.o(y, record)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x), record)), record)


class GPT(nn.Module):
    def __init__(self, cfg: Cfg, vocab: int):
        super().__init__()
        d, L, H = SIZES[cfg.size]
        g = torch.Generator().manual_seed(cfg.seed * 1000 + 999)
        self.emb = nn.Parameter(torch.randn(vocab, d, generator=g) * 0.02)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(L))
        self.ln = RMSNorm(d)
        hd = d // H
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
        ang = torch.arange(cfg.ctx).float()[:, None] * inv[None]
        self.register_buffer("cos", ang.cos())
        self.register_buffer("sin", ang.sin())

    def forward(self, idx, record=False):
        T = idx.shape[1]
        x = self.emb[idx]
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin, record)
        return F.linear(self.ln(x), self.emb)

    def hlinears(self):
        return [m for m in self.modules() if isinstance(m, HLinear)]


def accounting(model: GPT, cfg: Cfg, vocab: int):
    d, L, H = SIZES[cfg.size]
    hard = trainable = frozen_r = dense_eq = 0
    for m in model.hlinears():
        h, t, _ = m.macs()
        hard += h
        trainable += t
        if m.design in ("A", "frozen"):
            frozen_r += m.n_in * m.n_out
        dense_eq += m.n_in * m.n_out
    attn = L * cfg.ctx * d  # QK^T and AV, causal average ctx/2 each
    head = vocab * d
    fwd = trainable + frozen_r + attn + head
    train_flops_per_token = 6 * (trainable + attn + head) + 4 * frozen_r
    dense_flops_per_token = 6 * (dense_eq + attn + head)
    return {
        "macs_fwd_per_token": fwd,
        "macs_hardened_per_token": hard,
        "hardened_frac_fwd": hard / fwd,
        "macs_dense_equiv_per_token": dense_eq + attn + head,
        "train_flops_per_token": train_flops_per_token,
        "dense_train_flops_per_token": dense_flops_per_token,
    }


def batches(arr, cfg: Cfg, rng: np.random.Generator):
    n = len(arr) - cfg.ctx - 1
    while True:
        ix = rng.integers(0, n, cfg.batch)
        x = np.stack([arr[i:i + cfg.ctx + 1] for i in ix]).astype(np.int64)
        yield torch.from_numpy(x[:, :-1]), torch.from_numpy(x[:, 1:])


@torch.no_grad()
def evaluate(model, val, cfg: Cfg, n_batches: int, record=False):
    model.eval()
    Noise.gen = torch.Generator().manual_seed(777)
    stride = (len(val) - cfg.ctx - 1) // (n_batches * cfg.batch)
    tot, cnt = 0.0, 0
    for b in range(n_batches):
        ix = [(b * cfg.batch + j) * stride for j in range(cfg.batch)]
        x = torch.from_numpy(np.stack([val[i:i + cfg.ctx + 1] for i in ix]).astype(np.int64))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            logits = model(x[:, :-1], record=record and b == 0)
        tot += F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    Noise.gen = None
    return tot / cnt


@torch.no_grad()
def diagnostics(model: GPT):
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, HLinear):
            continue
        rec = {"design": m.design, "hard": m.hard}
        if m.design == "dense":
            dw = m.weight - m.R
            rec["rel_move_fro"] = float(dw.norm() / m.R.norm())
            rec["rel_move_spec"] = float(torch.linalg.matrix_norm(dw, 2) / torch.linalg.matrix_norm(m.R, 2))
            rec["dw_stable_rank"] = float(dw.norm() ** 2 / torch.linalg.matrix_norm(dw, 2) ** 2)
        if m.design == "B":
            rec["delta_over_r_fro"] = float(m.delta.norm() / m.R.norm())
        if m.design == "lsb":
            c = m.lsb_codes()
            rec["lsb_low_bits_public"] = float(((c - m.lo).remainder(m.step) == 0).float().mean())
            hi = (c - m.lo) / m.step
            rec["lsb_codes_at_clamp"] = float(((hi <= -128 // m.step) | (hi >= 128 // m.step - 1)).float().mean())
        if m.U is not None:
            uv = m.U @ m.V
            rec["uv_fro_over_r_fro"] = float(uv.norm() / m.R.norm())
        if m.d1 is not None:
            rec["d1_rms"] = float(m.d1.pow(2).mean().sqrt())
            rec["d2_rms"] = float(m.d2.pow(2).mean().sqrt())
        if m.last_share is not None:
            rec["r_share_out"] = m.last_share
        out[name] = rec
    return out


def summarize_diag(diag):
    keys = ["rel_move_fro", "rel_move_spec", "dw_stable_rank", "delta_over_r_fro", "uv_fro_over_r_fro", "d1_rms",
            "r_share_out", "lsb_low_bits_public", "lsb_codes_at_clamp"]
    res = {}
    for kind in ("qkv", "o", "fc1", "fc2"):
        rows = [v for k, v in diag.items() if k.endswith("." + kind)]
        for key in keys:
            vals = [r[key] for r in rows if key in r]
            if vals:
                res[f"{kind}.{key}"] = float(np.mean(vals))
    return res


def param_groups(model: GPT, cfg: Cfg):
    mats, vecs, uv = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
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


def train(cfg: Cfg, data: Path):
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    meta = json.loads((data / "meta.json").read_text())
    vocab = meta["vocab"]
    tr = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data / "val.bin", dtype=np.uint16, mode="r")
    model = GPT(cfg, vocab)
    groups = param_groups(model, cfg)
    opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)
    steps = max(1, cfg.tokens // (cfg.batch * cfg.ctx))
    acct = accounting(model, cfg, vocab)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_emb = model.emb.numel()
    print(json.dumps({"cfg": dataclasses.asdict(cfg), "steps": steps, "trainable_params": n_params,
                      "non_emb_trainable": n_params - n_emb, **acct}), flush=True)

    def lr_at(s):
        if s < cfg.warmup:
            return (s + 1) / cfg.warmup
        p = (s - cfg.warmup) / max(1, steps - cfg.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p))

    it = batches(tr, cfg, np.random.default_rng(cfg.data_seed))
    curve, ema, t0 = [], None, time.time()
    hl = model.hlinears()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = cfg.lr * g["lr_mult"] * lr_at(s)
        x, y = next(it)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=cfg.bf16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in hl:
            m.project()
        lv = loss.item()
        if not math.isfinite(lv):
            raise FloatingPointError(f"loss {lv} at step {s}")
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if (s + 1) % cfg.eval_every == 0 or s + 1 == steps:
            v = evaluate(model, val, cfg, cfg.eval_batches)
            el = time.time() - t0
            tok = (s + 1) * cfg.batch * cfg.ctx
            curve.append({"step": s + 1, "tokens": tok, "train_ema": ema, "val": v, "wall_s": el})
            print(f"step {s + 1}/{steps} tok {tok} train {ema:.4f} val {v:.4f} {tok / el:.0f} tok/s", flush=True)
    final = evaluate(model, val, cfg, cfg.final_eval_batches, record=True)
    diag = diagnostics(model)
    wall = time.time() - t0
    return {
        "cfg": dataclasses.asdict(cfg),
        "vocab": vocab,
        "steps": steps,
        "tokens": steps * cfg.batch * cfg.ctx,
        "trainable_params": n_params,
        "non_emb_trainable": n_params - n_emb,
        **acct,
        "train_flops": acct["train_flops_per_token"] * steps * cfg.batch * cfg.ctx,
        "dense_train_flops": acct["dense_train_flops_per_token"] * steps * cfg.batch * cfg.ctx,
        "final_val": final,
        "final_eval_tokens": cfg.final_eval_batches * cfg.batch * cfg.ctx,
        "curve": curve,
        "diag_summary": summarize_diag(diag),
        "diag": diag,
        "wall_s": wall,
        "tok_per_s": steps * cfg.batch * cfg.ctx / wall,
        "torch": torch.__version__,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON object of Cfg fields")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg(**json.loads(a.config))
    res = train(cfg, Path(a.data))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"final {cfg.name} val {res['final_val']:.4f} wall {res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
