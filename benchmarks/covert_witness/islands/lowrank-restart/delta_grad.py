"""Direction `lowrank-restart` (formalism v2): training whose state Δ_grad, the gradient and optimizer state passed
between blocks (Fit's reduction clause, 2 Δ_grad ≤ k), is small.

A step's batch is cut into token blocks of `--leaf` sequences (4 × 1024 tokens, the reference policy's s = c/w = 4096).
Each block's message over the trainable matrices is coded at `--msg-bits` bits an entry (stochastic rounding onto a
symmetric uniform grid, an f16 absmax scale per `--group` entries; 32 sends fp32; `--vote`, one bit and no scale, a
majority vote; `--rotate` codes the messages and the momentum in a random orthonormal basis per group), and the messages are summed up a reduction tree (`Tree`, `--tree`: `heap`, each block a node that sends
its gradient plus its two children's messages, or `leaves`), so the optimizer reads only what a Fit reduction tree
passes. Between steps the optimizer keeps `--opt`'s state:

- `adamw`: m and v in fp32, 64 bits an entry (the trainable set's reference);
- `adafactor`: β1 = 0, the second moment factored per matrix (an fp32 row and column vector), RMS-clipped steps;
- `muon`: the reduced gradient orthogonalized by Newton–Schulz, with a Nesterov momentum at `--mom-bits` (0: none),
  or with `--mom-rank R` that momentum in a rank-R subspace of the gradient (GaLore's projection, re-drawn every
  `--proj-every` steps; the basis is state too, in bf16); with `--msg-proj` too, a block sends only its gradient's
  projection Pᵀ G (R × the long side), and at a re-draw G Ω (the short side × R, Ω Gaussian from the seed and the step)
  as well, whose sum's QR is the next basis, so the messages grow with R and not with the matrices;
- `sign`: signSGD on the reduced gradient, or on its momentum at `--mom-bits` (signum).

The update Δ_t = θ_t − θ lives in `--store`: `qN` (N-bit codes into a uniform f16 codebook, an f16 scale per group,
stochastically rounded after every step, the scales re-fit every step or, with `--store-scale sticky`, only when a value
outgrows them; E is the store itself as CWQUAN1 delta entries) or `bf16` (stochastic rounding;
E codes Δ once at `--e-bits`). `--support P` trains only a fixed share P of each matrix, drawn from `--seed` (`Support`:
entries or whole rows), so the messages, the store and an elementwise optimizer's state hold only those entries (Adafactor
keeps its row and column factors); E codes the scattered Δ once at `--e-bits`. `--lora R` trains each target as
W + B A instead (B zero, A drawn from `--seed`): a block's message is its factor gradients Bᵀ G and G Aᵀ, computed from
its own G, the factors are the optimizer's tensors and the store's, and E is CWLORA2 at `--e-bits` (4 or 8).
`--targets gains` makes the norm gains of `--blocks` and the final norm the trained set instead (elementwise: `--opt
sign` or `adamw`), coded, reduced and stored like the matrices, for a state small enough for Fit at the default dial.
Otherwise the trainable set's norm gains train by AdamW in fp32 with fp32 messages (counted). Every
record's notes carry the accounting: n, the message, state and store bits, Δ_grad and Δ_t in bits (`delta_grad`, in the
keys the other directions use), and Fit's three clauses (compute Δ + sτ ≤ k with Δ = 8|E|, reduction 2 Δ_grad ≤ k,
update application Δ_t + Δ_grad ≤ k) at k = 10 kB, 1 MB and 10 MB, with sτ = a block's tokens × ⌈log₂ V⌉ bits.

    python benchmarks/covert_witness/launch.py s-t20 --island lowrank-restart --program delta_grad.py --note "dgrad ..." \\
        -- --blocks 5-5 --opt adafactor --msg-bits 8 --store q8 --lr 3e-3 --budget-fraction 0.15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare  # noqa: E402
from model import VOCAB  # noqa: E402
from te_core import hooks  # noqa: E402
from lowrank_restart import newton_schulz, ns_flops, schedule, targets_of  # noqa: E402

DEV = prepare.DEVICE
#: the harness scores the first E encoded in each 1/CHECKPOINTS of the planned steps
CHECKPOINTS = 20
KS = {"10kB": 8 * 10 ** 4, "1MB": 8 * 10 ** 6, "10MB": 8 * 10 ** 7}
#: elementwise work an entry per coding (scale, grid search, random draw, select), for the step log
CODE_FLOPS = 16
#: a sticky store scale, when a value outgrows it, becomes this × its group's absmax (so it is rarely re-fit)
STICKY_HEADROOM = 1.125


# ---- codes ------------------------------------------------------------------------------------------------------------

def codebook(bits: int) -> torch.Tensor:
    """The f16 grid: ±1 at one bit, else (c − L)/L for c in 0..2L with L = 2^(bits−1) − 1, padded with 0 to 2^bits."""
    if bits == 1:
        return torch.tensor([-1.0, 1.0], dtype=torch.float16)
    L = 2 ** (bits - 1) - 1
    cb = torch.zeros(2 ** bits, dtype=torch.float16)
    cb[:2 * L + 1] = (torch.arange(2 * L + 1, dtype=torch.float64) - L) / L
    return cb


def hadamard(n: int) -> torch.Tensor:
    """The orthonormal n × n Sylvester–Hadamard matrix (n a power of two)."""
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


class Code:
    """Stochastic rounding of x onto `bits`-bit codes of a uniform f16 grid times an f16 absmax scale per `group` entries,
    unbiased: E[value] = x. With `rotate`, `__call__` codes each group in a random orthonormal basis H D (D signs drawn
    once from `gen`), so a few large entries no longer set the whole group's scale; the decoded value is rotated back."""

    def __init__(self, bits: int, group: int, gen: torch.Generator, rotate: bool = False):
        self.bits, self.group, self.gen = bits, group, gen
        cb = codebook(bits)
        self.n_grid = 2 if bits == 1 else 2 ** bits - 1
        self.grid = cb[:self.n_grid].float().to(DEV)
        self.cb = cb
        self.rot = None
        if rotate:
            if group & (group - 1):
                raise SystemExit(f"delta_grad: a rotated code's --group ({group}) is a power of two")
            signs = torch.randint(0, 2, (group,), generator=gen, device=gen.device).float() * 2 - 1
            self.rot = (signs[:, None].cpu() * hadamard(group)).to(DEV)
        #: an entry's work per coding, the rotation's two group × group products included
        self.entry_flops = CODE_FLOPS + (4 * group if rotate else 0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = (x.float().reshape(-1, self.group).abs().amax(1) * (1 + 2 ** -9)).to(torch.float16)
        return self.encode_at(x, s)

    def encode_at(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x coded at the given f16 scales, one a group, each at least its group's absmax."""
        flat = x.float().reshape(-1, self.group)
        div = torch.where(s > 0, s.float(), torch.ones_like(s, dtype=torch.float32))
        v = (flat / div[:, None]).clamp(-1, 1)
        top = self.n_grid - 1
        lo = ((v + 1) * (top / 2)).floor().clamp(0, top - 1).long()
        g_lo, g_hi = self.grid[lo], self.grid[lo + 1]
        p = ((v - g_lo) / (g_hi - g_lo)).clamp(0, 1)
        u = torch.rand(p.shape, generator=self.gen, device=p.device)
        return lo + (u < p).long(), s

    def values(self, codes: torch.Tensor, s: torch.Tensor, shape) -> torch.Tensor:
        return (self.grid[codes] * s.float()[:, None]).reshape(shape)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.rot is None:
            return self.values(*self.encode(x), x.shape)
        y = x.float().reshape(-1, self.group) @ self.rot
        return (self.values(*self.encode(y), y.shape) @ self.rot.T).reshape(x.shape)


class Vote:
    """One bit an entry, no scale: the sign, a zero drawn ±1 at random (a majority vote's tie)."""

    bits = 1

    def __init__(self, gen: torch.Generator):
        self.gen = gen

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        coin = torch.randint(0, 2, x.shape, generator=self.gen, device=x.device, dtype=torch.int8).float() * 2 - 1
        return torch.where(x == 0, coin, x.float().sign())


def sr_bf16(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """x (fp32) stochastically rounded to bf16: random low 16 bits added before truncation."""
    r = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
    return ((x.float().contiguous().view(torch.int32) + r) & -65536).view(torch.float32).to(torch.bfloat16)


class Store:
    """One matrix's Δ = θ' − θ in `kind`: qN (CWQUAN1 delta codes) or bf16 (E codes it once, at `e_bits`; 0: θ + Δ in
    full). A qN store's scales are tight (each group's absmax, re-fit after every step, so a moving scale re-rounds
    every entry of its group) or `sticky` (kept until a value outgrows its group's, then STICKY_HEADROOM × its absmax)."""

    def __init__(self, shape, kind: str, group: int, gen: torch.Generator, e_bits: int = 8, sticky: bool = False):
        self.shape, self.kind, self.gen, self.sticky = tuple(shape), kind, gen, sticky
        n = math.prod(self.shape)
        if kind == "bf16":
            self.d = torch.zeros(self.shape, dtype=torch.bfloat16, device=DEV)
            self.e_code = Code(e_bits, group, gen) if e_bits else None
        else:
            self.code = Code(int(kind[1:]), group, gen)
            zero = 0 if self.code.bits == 1 else self.code.n_grid // 2
            self.codes = torch.full((n // group, group), zero, dtype=torch.long, device=DEV)
            self.scales = torch.zeros(n // group, dtype=torch.float16, device=DEV)

    def value(self) -> torch.Tensor:
        if self.kind == "bf16":
            return self.d.float()
        return self.code.values(self.codes, self.scales, self.shape)

    def add_(self, step: torch.Tensor) -> None:
        new = self.value() + step
        if self.kind == "bf16":
            self.d = sr_bf16(new, self.gen)
        elif self.sticky:
            need = new.float().reshape(-1, self.code.group).abs().amax(1)
            s = torch.where(need > self.scales.float(), (need * STICKY_HEADROOM).to(torch.float16), self.scales)
            self.codes, self.scales = self.code.encode_at(new, s)
        else:
            self.codes, self.scales = self.code.encode(new)

    def entry(self, theta_w: torch.Tensor) -> dict:
        if self.kind == "bf16" and self.e_code is None:
            return {"kind": "full", "t": (theta_w.to(DEV).float() + self.value()).to(theta_w.dtype)}
        code = self.e_code if self.kind == "bf16" else self.code
        codes, scales = code.encode(self.value()) if self.kind == "bf16" else (self.codes, self.scales)
        q = prepare.QArray(code.cb, scales.cpu(), codes.reshape(-1).to(torch.uint8).cpu(), code.bits, codes.shape[1],
                           self.shape)
        return {"kind": "delta", "q": q}


class Tree:
    """A step's reduction tree over its token blocks, each block's message coded by `code` (None: fp32).

    `heap`: node i is a block whose children are 2i+1 and 2i+2; it sends code(its gradient + its children's messages),
    so a node reads two messages (2 Δ_grad) beside its tokens, and the root's sum, the optimizer's, isn't coded.
    `leaves`: the blocks are leaves of a binary tree whose inner nodes hold no tokens; every sum is coded, the root's
    too. A `Vote` code makes either a majority vote: a block's own gradient enters as its sign."""

    def __init__(self, code, shape: str, n_blocks: int):
        self.code, self.shape, self.n, self.i = code, shape, n_blocks, 0
        self.stack, self.pending, self.codings, self.out = [], {}, 0, None

    def _c(self, g: dict) -> dict:
        if self.code is None:
            return g
        self.codings += 1
        return {n: self.code(t) for n, t in g.items()}

    @staticmethod
    def _sum(*gs: dict) -> dict:
        return {n: sum(g[n] for g in gs) for n in gs[0]}

    def push(self, g: dict) -> None:
        own = self._c(g) if isinstance(self.code, Vote) else g
        if self.shape == "heap":
            j = self.n - 1 - self.i
            kids = [self.pending.pop(k) for k in (2 * j + 1, 2 * j + 2) if k in self.pending]
            total = self._sum(own, *kids)
            if j:
                self.pending[j] = self._c(total)
            else:
                self.out = total
        else:
            node = (0, own if isinstance(self.code, Vote) else self._c(own))
            while self.stack and self.stack[-1][0] == node[0]:
                lvl, left = self.stack.pop()
                node = (lvl + 1, self._c(self._sum(left, node[1])))
            self.stack.append(node)
        self.i += 1

    def root(self) -> dict:
        if self.shape == "heap":
            return self.out
        while len(self.stack) > 1:
            _, right = self.stack.pop()
            lvl, left = self.stack.pop()
            self.stack.append((lvl + 1, self._c(self._sum(left, right))))
        return self.stack.pop()[1]


class Support:
    """A fixed share of one matrix's entries, drawn from the run's seed: `entries` (uniformly, a multiple of `group`) or
    `rows` (whole rows). Only these entries train; messages, store and elementwise optimizer state hold just them."""

    def __init__(self, shape, share: float, kind: str, group: int, gen: torch.Generator):
        self.shape, numel = tuple(shape), math.prod(shape)
        if kind == "rows":
            unit = group // math.gcd(shape[1], group)
            r = min(shape[0], max(1, round(share * shape[0] / unit)) * unit)
            rows = torch.randperm(shape[0], generator=gen, device=DEV)[:r].sort().values
            self.idx = (rows[:, None] * shape[1] + torch.arange(shape[1], device=DEV)).reshape(-1)
        else:
            k = max(1, round(share * numel / group)) * group
            self.idx = torch.randperm(numel, generator=gen, device=DEV)[:k].sort().values
        self.k = self.idx.numel()

    def gather(self, t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1)[self.idx]

    def scatter(self, v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(math.prod(self.shape), dtype=v.dtype, device=v.device)
        out[self.idx] = v
        return out.reshape(self.shape)


# ---- optimizers on the reduced gradient -------------------------------------------------------------------------------

class Opt:
    """`mom_bits` (muon, sign): a momentum buffer in N-bit codes (`Code`, `group`), 32 fp32, 0 none.

    `mom_rank` R (muon): the momentum lives in a rank-R subspace of each matrix's short side (GaLore's projection): every
    `proj_every` steps P, the top R left singular vectors of the reduced gradient h (the matrix, transposed if tall), is
    re-drawn and kept in bf16, M (R × the long side) is rotated into it, M ← b1 M + Pᵀ h at `mom_bits`, and the step is
    NS(h + b1 P M). With `msg_proj` the optimizer sees only what the blocks send (`step_proj`): Pᵀ h, and at a re-draw
    h Ω for a seeded Gaussian Ω (long side × R), whose QR gives the next P; the step is P NS(Pᵀ h + b1 M)."""

    def __init__(self, kind: str, shapes: dict, b1: float, b2: float, mom_bits: int, group: int, gen: torch.Generator,
                 mom_rank: int = 0, proj_every: int = 100, msg_proj: bool = False, rotate: bool = False):
        self.kind, self.b1, self.b2, self.t = kind, b1, b2, 0
        self.m, self.v, self.r, self.c, self.p = {}, {}, {}, {}, {}
        self.mom_bits, self.group, self.n = mom_bits, group, sum(math.prod(s) for s in shapes.values())
        self.mom_rank, self.proj_every, self.shapes, self.msg_proj = mom_rank, proj_every, dict(shapes), msg_proj
        self.mom_code = Code(mom_bits, group, gen, rotate) if 0 < mom_bits < 32 else None
        self.code_flops = self.mom_code.entry_flops if self.mom_code else CODE_FLOPS
        for n, sh in shapes.items():
            if kind == "adamw":
                self.m[n] = torch.zeros(sh, device=DEV)
                self.v[n] = torch.zeros(sh, device=DEV)
            elif kind == "adafactor":
                self.r[n] = torch.zeros(sh[0], device=DEV)
                self.c[n] = torch.zeros(sh[1], device=DEV)
            elif mom_bits:
                self.m[n] = torch.zeros((mom_rank, max(sh)) if mom_rank else sh, device=DEV)

    def state_bits(self) -> int:
        if self.kind in ("muon", "sign"):
            if not self.mom_bits:
                return 0
            n_m = sum(t.numel() for t in self.m.values())
            basis = sum(16 * min(sh) * self.mom_rank for sh in self.shapes.values())
            return basis + (32 * n_m if self.mom_bits >= 32 else n_m * self.mom_bits + 16 * (n_m // self.group))
        return 32 * sum(t.numel() for d in (self.m, self.v, self.r, self.c) for t in d.values())

    def tick(self) -> None:
        self.t += 1

    def _momentum(self, n: str, x: torch.Tensor, keep: float) -> torch.Tensor:
        m = self.b1 * self.m[n] + keep * x
        self.m[n] = self.mom_code(m) if self.mom_code else m
        return self.m[n]

    def step(self, n: str, g: torch.Tensor, lr: float) -> torch.Tensor:
        b1, b2, t = self.b1, self.b2, self.t
        if self.kind == "adamw":
            self.m[n].mul_(b1).add_(g, alpha=1 - b1)
            self.v[n].mul_(b2).addcmul_(g, g, value=1 - b2)
            mh, vh = self.m[n] / (1 - b1 ** t), self.v[n] / (1 - b2 ** t)
            return lr * mh / (vh.sqrt() + 1e-8)
        if self.kind == "adafactor":
            g2 = g * g + 1e-30
            self.r[n].mul_(b2).add_(g2.mean(1), alpha=1 - b2)
            self.c[n].mul_(b2).add_(g2.mean(0), alpha=1 - b2)
            r, c = self.r[n] / (1 - b2 ** t), self.c[n] / (1 - b2 ** t)
            u = g / torch.outer(r / r.mean(), c).sqrt()
            return lr * u / max(1.0, float(u.pow(2).mean().sqrt()))
        if self.kind == "muon" and self.mom_rank:
            tall = g.shape[0] > g.shape[1]
            h = g.T if tall else g
            if (t - 1) % self.proj_every == 0:
                p = torch.linalg.svd(h.float(), full_matrices=False)[0][:, :self.mom_rank]
                if n in self.p:
                    m = (p.T @ self.p[n].float()) @ self.m[n]
                    self.m[n] = self.mom_code(m) if self.mom_code else m
                self.p[n] = p.bfloat16()
            p = self.p[n].float()
            m = self._momentum(n, p.T @ h, 1.0)
            u = h + b1 * (p @ m)
            return lr * max(1.0, g.shape[0] / g.shape[1]) ** 0.5 * newton_schulz(u.T if tall else u)
        if self.kind == "muon":
            u = g + b1 * self._momentum(n, g, 1.0) if self.mom_bits else g
            return lr * max(1.0, g.shape[0] / g.shape[1]) ** 0.5 * newton_schulz(u)
        return lr * (self._momentum(n, g, 1 - b1) if self.mom_bits else g).sign()

    def step_proj(self, n: str, ph: torch.Tensor | None, y: torch.Tensor | None, lr: float) -> torch.Tensor | None:
        """`msg_proj`: ph = Pᵀ h (None before the first basis, so no step), y = h Ω at a re-draw (else None)."""
        sh, out = self.shapes[n], None
        if ph is not None:
            x = ph + self.b1 * self._momentum(n, ph, 1.0)
            u = self.p[n].float() @ newton_schulz(x)
            out = lr * max(1.0, sh[0] / sh[1]) ** 0.5 * (u.T if sh[0] > sh[1] else u)
        if y is not None:
            p = torch.linalg.qr(y.float())[0]
            if n in self.p:
                m = (p.T @ self.p[n].float()) @ self.m[n]
                self.m[n] = self.mom_code(m) if self.mom_code else m
            self.p[n] = p.bfloat16()
        return out

    def flops(self, shapes: dict) -> int:
        n = sum(math.prod(s) for s in shapes.values())
        mom = (2 + (self.code_flops if self.mom_code else 0)) * n if self.mom_bits else 0
        if self.kind == "muon" and self.msg_proj:
            r, nm, k = self.mom_rank, sum(t.numel() for t in self.m.values()), self.proj_every
            lift = sum(2 * r * math.prod(s) + ns_flops((r, max(s))) for s in shapes.values())
            redraw = sum(4 * min(s) * r * r + 2 * r * r * max(s) for s in shapes.values()) // k
            return lift + redraw + (3 + self.code_flops) * nm
        if self.kind == "muon" and self.mom_rank:
            r, nm = self.mom_rank, sum(t.numel() for t in self.m.values())
            proj = sum(4 * r * math.prod(s) + 2 * min(s) * max(s) ** 2 // self.proj_every for s in shapes.values())
            return sum(ns_flops(s) for s in shapes.values()) + 2 * n + proj + (3 + self.code_flops) * nm
        if self.kind == "muon":
            return sum(ns_flops(s) for s in shapes.values()) + 2 * n + mom
        return {"adamw": 12, "adafactor": 14, "sign": 2}[self.kind] * n + mom


# ---- accounting -------------------------------------------------------------------------------------------------------

def accounting(a, shapes: dict, n_norm: int, opt: Opt, tau_bits: int) -> dict:
    """Formalism v2's training quantities for this run's configuration, in bits (`fit` adds Δ once E exists)."""
    n = sum(math.prod(s) for s in shapes.values())
    nm = sum(a.mom_rank * (max(s) + min(s)) for s in shapes.values()) if a.msg_proj else n
    msg = 32 * nm if a.msg_bits >= 32 else nm if a.vote else nm * a.msg_bits + 16 * (nm // a.group)
    msg += 32 * n_norm
    state = opt.state_bits() + 64 * n_norm
    store = 16 * n if a.store == "bf16" else n * int(a.store[1:]) + 16 * (n // a.group)
    store += 32 * n_norm
    s_tau = a.leaf * prepare.EVAL_CTX * tau_bits
    return {"n_matrix_entries": n, "n_message_entries": nm, "n_norm_entries": n_norm, "message_bits": msg,
            "state_bits": state, "delta_grad_bits": msg + state, "delta_t_bits": store, "tau_bits": tau_bits,
            "s_tau_bits": s_tau, "leaf_tokens": a.leaf * prepare.EVAL_CTX, "tree": a.tree}


def fit(acct: dict, delta_bits: int) -> dict:
    """Fit's clauses (formalism v2) at each of `KS`: compute Δ + sτ ≤ k, reduction 2 Δ_grad ≤ k, update application
    Δ_t + Δ_grad ≤ k, with Δ_t the store as the update rule reads and writes it."""
    dg, dt, s_tau = acct["delta_grad_bits"], acct["delta_t_bits"], acct["s_tau_bits"]
    return {k: {"compute": delta_bits + s_tau <= kb, "reduction": 2 * dg <= kb, "update": dt + dg <= kb}
            for k, kb in KS.items()}


# ---- the run ----------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="all", help="all, out, mlp, attn, or leaf names joined by '+' (wq+wo); "
                    "gains: the norm gains of --blocks and the final norm, as the coded trainable set")
    ap.add_argument("--blocks", default="all", help="all, or LO-HI (0-based, inclusive)")
    ap.add_argument("--norms", choices=("trained", "none"), default="trained",
                    help="trained: the gains of the trained blocks and the final norm")
    ap.add_argument("--opt", choices=("adamw", "adafactor", "muon", "sign"), required=True)
    ap.add_argument("--msg-bits", type=int, default=32)
    ap.add_argument("--vote", action="store_true", help="1-bit messages as a majority vote (`Vote`), no scales")
    ap.add_argument("--tree", choices=("heap", "leaves"), default="heap")
    ap.add_argument("--mom-bits", type=int, default=0, help="muon and sign: the momentum's bits (0 none, 32 fp32)")
    ap.add_argument("--mom-rank", type=int, default=0, help="muon: the momentum in a rank-R gradient subspace (`Opt`)")
    ap.add_argument("--proj-every", type=int, default=100, help="--mom-rank: steps between the subspace's re-draws")
    ap.add_argument("--msg-proj", action="store_true",
                    help="--mom-rank: a block sends Pᵀ G, and h Ω at a re-draw, instead of G (`Opt.step_proj`)")
    ap.add_argument("--rotate", action="store_true",
                    help="code the messages and the momentum in a random orthonormal basis per group (`Code`)")
    ap.add_argument("--store", default="q8", help="qN (1 <= N <= 8) or bf16")
    ap.add_argument("--store-scale", choices=("tight", "sticky"), default="tight",
                    help="a qN store's f16 scales: re-fit to each group's absmax after every step, or sticky (`Store`)")
    ap.add_argument("--e-bits", type=int, default=8,
                    help="a bf16 store's or a sparse support's E: Δ coded once at N bits (0: θ + Δ in full, dense only)")
    ap.add_argument("--support", type=float, default=1.0, help="the share of each matrix that trains (`Support`)")
    ap.add_argument("--support-kind", choices=("entries", "rows"), default="entries")
    ap.add_argument("--lora", type=int, default=0,
                    help="rank R: each target trains as W + B A (B zero, A from the seed); E is CWLORA2 at --e-bits")
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--leaf", type=int, default=4, help="sequences of 1024 tokens a leaf block")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--norm-lr", type=float, default=1e-3)
    ap.add_argument("--b1", type=float, default=0.9)
    ap.add_argument("--b2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0, help="the reduced gradient's norm (adamw only)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--budget-fraction", type=float, default=0.15)
    a = ap.parse_args()
    if a.store != "bf16" and not (a.store.startswith("q") and 1 <= int(a.store[1:]) <= 8):
        raise SystemExit("delta_grad: --store is qN (1 <= N <= 8) or bf16")
    if a.store_scale == "sticky" and a.store == "bf16":
        raise SystemExit("delta_grad: --store-scale sticky is a qN store's; a bf16 store has no scales")
    if not (1 <= a.msg_bits <= 8 or a.msg_bits == 32) or a.batch % a.leaf:
        raise SystemExit("delta_grad: --msg-bits is 1 to 8 or 32, and --leaf divides --batch")
    if a.vote and a.msg_bits != 1:
        raise SystemExit("delta_grad: --vote sends 1-bit messages; pass --msg-bits 1")
    if a.mom_bits and a.opt not in ("muon", "sign") or not (0 <= a.mom_bits <= 8 or a.mom_bits == 32):
        raise SystemExit("delta_grad: --mom-bits (0 to 8, or 32) is muon's and sign's")
    sparse = a.support < 1
    if sparse and (a.opt == "muon" and a.mom_bits or not 1 <= a.e_bits <= 8):
        raise SystemExit("delta_grad: a sparse --support takes no muon momentum, and codes E at 1 <= --e-bits <= 8")
    if a.mom_rank and (a.opt != "muon" or not a.mom_bits or sparse or a.lora or a.proj_every < 1):
        raise SystemExit("delta_grad: --mom-rank is a dense muon's momentum (--mom-bits > 0, no --support, no --lora), "
                         "with --proj-every >= 1")
    if a.rotate and (a.vote or a.msg_bits >= 32 and not 0 < a.mom_bits < 32):
        raise SystemExit("delta_grad: --rotate codes the messages or the momentum; with --vote or nothing coded it has no use")
    if a.msg_proj and (not a.mom_rank or a.vote):
        raise SystemExit("delta_grad: --msg-proj projects onto --mom-rank's basis, and takes no --vote")
    lora = a.lora > 0
    if lora and (sparse or a.e_bits not in prepare.LORA2_BITS):
        raise SystemExit(f"delta_grad: --lora takes no --support, and codes E as CWLORA2 at --e-bits in "
                         f"{prepare.LORA2_BITS}")
    torch.backends.cuda.matmul.allow_tf32 = True

    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    theta_dev = {n: theta[n].to(DEV) for n in params}
    gains = a.targets == "gains"
    if gains and (a.opt not in ("sign", "adamw") or lora or sparse or a.mom_rank):
        raise SystemExit("delta_grad: --targets gains trains the norm gains elementwise (--opt sign or adamw), "
                         "with no --lora, --support or --mom-rank")
    n_layer = 1 + max(int(n.split(".")[1]) for n in params if n.startswith("blocks."))
    lo = int(a.blocks.split("-")[0]) if gains and a.blocks != "all" else 0
    hi = int(a.blocks.split("-")[1]) if gains and a.blocks != "all" else n_layer - 1
    targets = sorted(n for n, p in params.items() if p.ndim == 1 and (
        n.startswith("norm") or n.startswith("blocks.") and lo <= int(n.split(".")[1]) <= hi)) if gains \
        else targets_of(params, a.targets, a.blocks)
    lo = lo if gains else min(int(n.split(".")[1]) for n in targets)
    norms = []
    if a.norms == "trained" and not gains:
        norms = sorted(n for n, p in params.items() if p.ndim == 1 and
                       (n.startswith("norm") or (n.startswith("blocks.") and int(n.split(".")[1]) >= lo)))
    shapes = {n: tuple(params[n].shape) for n in targets}
    if any(math.prod(s) % a.group for s in shapes.values()):
        raise SystemExit(f"delta_grad: --group {a.group} doesn't divide every target's entries")
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    sup = {n: Support(shapes[n], a.support, a.support_kind, a.group, gen) for n in targets} if sparse else {}
    trained = {n: (sup[n].k,) for n in targets} if sparse else shapes
    if lora:
        trained = {k: sh for n in targets for k, sh in ((n + "#A", (a.lora, shapes[n][1])),
                                                        (n + "#B", (shapes[n][0], a.lora)))}
        if any(math.prod(s) % a.group for s in trained.values()):
            raise SystemExit(f"delta_grad: --group {a.group} doesn't divide every factor's entries")
    dense_opt = not sparse or a.opt in ("adafactor", "muon")
    opt_shapes = shapes if dense_opt and not lora else trained
    stores = {k: Store(sh, a.store, a.group, gen, a.e_bits, a.store_scale == "sticky") for k, sh in trained.items()}
    for n in targets if lora else ():
        stores[n + "#A"].add_(torch.randn(trained[n + "#A"], generator=gen, device=DEV) / math.sqrt(shapes[n][1]))
    e_code = Code(a.e_bits, a.group, gen) if sparse else None
    msg = Vote(gen) if a.vote else Code(a.msg_bits, a.group, gen, a.rotate) if a.msg_bits < 32 else None
    opt = Opt(a.opt, opt_shapes, a.b1, a.b2, a.mom_bits, a.group, gen, a.mom_rank, a.proj_every, a.msg_proj,
              a.rotate)
    if a.mom_rank and any(a.mom_rank > min(s) for s in opt_shapes.values()):
        raise SystemExit("delta_grad: --mom-rank exceeds a target's short side")
    if a.mom_rank and a.mom_bits < 32 and any(a.mom_rank * max(s) % a.group for s in opt_shapes.values()):
        raise SystemExit(f"delta_grad: --group {a.group} doesn't divide every projected momentum's entries")
    if a.msg_proj and a.msg_bits < 32 and any(a.mom_rank * min(s) % a.group for s in opt_shapes.values()):
        raise SystemExit(f"delta_grad: --group {a.group} doesn't divide every range message's entries (R × short side)")
    for n in targets + norms:
        params[n].requires_grad_(True)
    nopt = torch.optim.AdamW([params[n] for n in norms], lr=a.norm_lr, betas=(0.9, 0.95), weight_decay=0.0) \
        if norms else None
    tau_bits = math.ceil(math.log2(rec["config"].get("vocab", VOCAB)))
    acct = accounting(a, trained, sum(params[n].numel() for n in norms), opt, tau_bits)

    def delta(n: str) -> torch.Tensor:
        if lora:
            return stores[n + "#B"].value() @ stores[n + "#A"].value()
        return sup[n].scatter(stores[n].value()) if sparse else stores[n].value()

    def entry(n: str) -> dict:
        if not sparse:
            return stores[n].entry(theta[n])
        codes, scales = e_code.encode(delta(n))
        return {"kind": "delta", "q": prepare.QArray(e_code.cb, scales.cpu(), codes.reshape(-1).to(torch.uint8).cpu(),
                                                    e_code.bits, codes.shape[1], shapes[n])}

    tokens = a.batch * prepare.EVAL_CTX
    leaves = a.batch // a.leaf
    trainable = {"full": sorted(targets + norms)}
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_entries = acct["n_message_entries"]
    codings = 0 if msg is None else (leaves - 1 + (leaves if a.vote else 0)) if a.tree == "heap" else 2 * leaves - 1
    code_flops = msg.entry_flops if isinstance(msg, Code) else CODE_FLOPS
    extra = code_flops * n_entries * (codings + 1) + opt.flops(opt_shapes)
    if lora:
        extra += sum(2 * a.lora * math.prod(shapes[n]) for n in targets) * (2 * leaves + 1)
    if a.msg_proj:
        extra += sum(2 * a.mom_rank * math.prod(shapes[n]) for n in targets) * leaves * (a.proj_every + 1) // a.proj_every
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // (per_step + extra))
    hooks.plan(n_steps)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net) if a.compile else net
    print(json.dumps({"mech": "dgrad", "opt": a.opt, "msg_bits": a.msg_bits, "store": a.store,
                      "store_scale": a.store_scale, "lora": a.lora, "mom_bits": a.mom_bits, "mom_rank": a.mom_rank,
                      "msg_proj": a.msg_proj, "rotate": a.rotate, "lr": a.lr,
                      "steps": n_steps, "leaves": leaves, "targets": len(targets), "norms": len(norms),
                      "flops_per_step": per_step + extra, **acct}), flush=True)

    def encode() -> bytes:
        if lora:
            f = {n: (stores[n + "#A"].value(), stores[n + "#B"].value(), 1.0) for n in targets}
            return prepare.encode_lora2(f, a.e_bits, full={n: params[n].detach() for n in norms})
        e = {n: entry(n) for n in targets}
        e.update({n: {"kind": "full", "t": params[n].detach().to(theta[n].dtype)} for n in norms})
        return prepare.encode_quant(e, theta)

    def messages() -> dict:
        if a.msg_proj:
            out = {}
            for n in targets:
                gw = params[n].grad.float()
                h = gw.T if gw.shape[0] > gw.shape[1] else gw
                if n in basis:
                    out[n] = basis[n].T @ h
                if n in omega:
                    out[n + "#Y"] = h @ omega[n]
            return out
        if not lora:
            return {n: sup[n].gather(params[n].grad) if sparse else params[n].grad for n in targets}
        out = {}
        for n in targets:
            gw = params[n].grad.float()
            out[n + "#A"], out[n + "#B"] = fac[n + "#B"].T @ gw, gw @ fac[n + "#A"].T
        return out

    steps, log, losses, used = [], [], [], set()
    t0 = time.perf_counter()

    def bucket(k: int) -> int:
        return math.ceil(k * CHECKPOINTS / n_steps)

    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        rows = data.next(a.batch)
        tree, total = Tree(msg, a.tree, leaves), torch.zeros((), device=DEV)
        fac = {k: st.value() for k, st in stores.items()} if lora else {}
        basis, omega = {}, {}
        if a.msg_proj:
            basis = {n: opt.p[n].float() for n in targets if n in opt.p}
            if step % a.proj_every == 0:
                og = torch.Generator(device=DEV).manual_seed(a.seed * 1_000_003 + step)
                omega = {n: torch.randn((max(shapes[n]), a.mom_rank), generator=og, device=DEV) for n in targets}
        for i in range(0, a.batch, a.leaf):
            x = rows[i: i + a.leaf]
            with torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            loss.backward()
            total += loss.detach() / leaves
            tree.push(messages())
            for n in targets:
                params[n].grad = None
        g = {n: t / leaves for n, t in tree.root().items()}
        with torch.no_grad():
            for n in norms:
                params[n].grad.div_(leaves)
            if a.opt == "adamw" and a.clip:
                norm = torch.sqrt(sum(t.pow(2).sum() for t in g.values()) +
                                  sum(params[n].grad.pow(2).sum() for n in norms))
                f = min(1.0, a.clip / (float(norm) + 1e-6))
                g = {n: t * f for n, t in g.items()}
                for n in norms:
                    params[n].grad.mul_(f)
            opt.tick()
            for k in trained:
                if a.msg_proj:
                    s = opt.step_proj(k, g.get(k), g.get(k + "#Y"), a.lr * frac)
                    if s is not None:
                        stores[k].add_(-s)
                    continue
                if sparse and dense_opt:
                    s = sup[k].gather(opt.step(k, sup[k].scatter(g[k]), a.lr * frac))
                else:
                    s = opt.step(k, g[k], a.lr * frac)
                stores[k].add_(-s)
            for n in targets:
                params[n].data.copy_((theta_dev[n].float() + delta(n)).to(theta_dev[n].dtype))
        if nopt:
            for grp in nopt.param_groups:
                grp["lr"] = a.norm_lr * frac
            nopt.step()
            nopt.zero_grad(set_to_none=True)
        steps.append({"op": "train", "tokens": tokens})
        steps.append({"op": "extra", "flops": extra, "what": "message codes, tree sums, optimizer and store"})
        losses.append(total)
        if (step + 1) % 50 == 0:
            row = {"step": step + 1, "loss": round(float(total), 4), "s": round(time.perf_counter() - t0, 1)}
            log.append(row)
            print(json.dumps(row), flush=True)
        b = bucket(step + 1)
        if step + 1 < n_steps and b not in used and b != bucket(n_steps):
            encode()
            used.add(b)

    blob = encode()
    model = prepare.decode(theta, blob)
    changed = sorted(n for n in theta if not prepare.same_bits(model[n], theta[n]))
    tail = float(torch.stack(losses[-max(1, len(losses) // 10):]).mean())
    src = Path(__file__).read_bytes()
    own, norm_adam = opt.state_bits(), 64 * acct["n_norm_entries"]
    delta_grad = {"trainable": acct["n_matrix_entries"] + acct["n_norm_entries"], "grad_bits": acct["message_bits"],
                  "adam_bits": norm_adam + (own if a.opt == "adamw" else 0),
                  **({f"{a.opt}_bits": own} if a.opt != "adamw" else {}),
                  "latent_bits": acct["delta_t_bits"] if a.store == "bf16" else 0,
                  "bits": acct["delta_grad_bits"], "bytes": acct["delta_grad_bits"] // 8}
    notes = {"direction": "lowrank-restart", "mech": "dgrad", "args": vars(a), "accounting": acct,
             "fit": fit(acct, 8 * len(blob)), "delta_grad": delta_grad, "E_bytes": len(blob),
             "delta_bits_final": 8 * len(blob), "steps": n_steps, "tail_loss": round(tail, 5),
             "program": "islands/lowrank-restart/delta_grad.py", "program_sha256": hashlib.sha256(src).hexdigest(),
             "train_log": log[-40:]}
    prepare.submit(Path(a.out), a.start, blob, model, trainable, steps, changed=changed, notes=notes)
    print(json.dumps({"E_bytes": len(blob), "changed": len(changed), "tail_loss": round(tail, 5)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
