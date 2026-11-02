"""Direction `lowrank-restart`, night 2 (formalism v1): training whose accumulated state Δ_grad, the gradient and optimizer
state passed between blocks (Fit's training clause, 2 Δ_grad + sτ ≤ k), is small, run as the class `--class` names.

A step's batch is cut into leaf blocks of `--leaf` sequences (4 × 1024 tokens, the reference policy's s = c/w = 4096).
Each leaf's gradient over the trainable matrices is coded as a message of `--msg-bits` bits an entry (stochastic rounding
onto a symmetric uniform grid, an f16 absmax scale per `--group` entries; 32 sends fp32), and the messages are summed up a
binary tree whose every node codes its sum the same way, so the optimizer reads only what a Fit reduction tree passes.
Between steps the optimizer keeps `--opt`'s state:

- `adamw`: m and v in fp32, 64 bits an entry (the trainable set's reference);
- `adafactor`: β1 = 0, the second moment factored per matrix (an fp32 row and column vector), RMS-clipped steps;
- `muon`: the reduced gradient orthogonalized by Newton–Schulz, no momentum;
- `sign`: signSGD on the reduced gradient.

The update Δ_t = θ_t − θ lives in `--store`: `qN` (N-bit codes into a uniform f16 codebook, an f16 scale per group,
stochastically rounded after every step; E is the store itself as CWQUAN1 delta entries) or `bf16` (stochastic rounding;
E is θ + Δ in full). The trainable set's norm gains train by AdamW in fp32 with fp32 messages (counted). Every record's
notes carry the accounting: n, the message, state and store bits, Δ_grad and Δ_t in bits, and Fit's two clauses at
k = 1 MB and 10 MB with sτ = the leaf's tokens × 16 bits.

    python benchmarks/covert_witness/launch.py shallow s100 --program delta_grad.py --note "..." -- --class shallow \\
        --blocks 5-5 --opt adafactor --msg-bits 8 --store q8 --lr 3e-3 --budget-fraction 0.15
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
from te_core import hooks  # noqa: E402
from lowrank_restart import newton_schulz, ns_flops, schedule, targets_of  # noqa: E402

DEV = prepare.DEVICE
#: the harness scores the first E encoded in each 1/CHECKPOINTS of the planned steps
CHECKPOINTS = 20
#: τ, a token's data: one GPT-2 id
TOKEN_BITS = 16
KS = {"1MB": 8 * 10 ** 6, "10MB": 8 * 10 ** 7}
#: elementwise work an entry per coding (scale, grid search, random draw, select), for the step log
CODE_FLOPS = 16


# ---- codes ------------------------------------------------------------------------------------------------------------

def codebook(bits: int) -> torch.Tensor:
    """The f16 grid: ±1 at one bit, else (c − L)/L for c in 0..2L with L = 2^(bits−1) − 1, padded with 0 to 2^bits."""
    if bits == 1:
        return torch.tensor([-1.0, 1.0], dtype=torch.float16)
    L = 2 ** (bits - 1) - 1
    cb = torch.zeros(2 ** bits, dtype=torch.float16)
    cb[:2 * L + 1] = (torch.arange(2 * L + 1, dtype=torch.float64) - L) / L
    return cb


class Code:
    """Stochastic rounding of x onto `bits`-bit codes of a uniform f16 grid times an f16 absmax scale per `group` entries,
    unbiased: E[value] = x."""

    def __init__(self, bits: int, group: int, gen: torch.Generator):
        self.bits, self.group, self.gen = bits, group, gen
        cb = codebook(bits)
        self.n_grid = 2 if bits == 1 else 2 ** bits - 1
        self.grid = cb[:self.n_grid].float().to(DEV)
        self.cb = cb

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = x.float().reshape(-1, self.group)
        s = (flat.abs().amax(1) * (1 + 2 ** -9)).to(torch.float16)
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
        return self.values(*self.encode(x), x.shape)


def sr_bf16(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """x (fp32) stochastically rounded to bf16: random low 16 bits added before truncation."""
    r = torch.randint(0, 1 << 16, x.shape, generator=gen, device=x.device, dtype=torch.int32)
    return ((x.float().contiguous().view(torch.int32) + r) & -65536).view(torch.float32).to(torch.bfloat16)


class Store:
    """One matrix's Δ = θ' − θ in `kind`: qN (CWQUAN1 delta codes) or bf16."""

    def __init__(self, shape, kind: str, group: int, gen: torch.Generator):
        self.shape, self.kind, self.gen = tuple(shape), kind, gen
        n = math.prod(self.shape)
        if kind == "bf16":
            self.d = torch.zeros(self.shape, dtype=torch.bfloat16, device=DEV)
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
        else:
            self.codes, self.scales = self.code.encode(new)

    def entry(self, theta_w: torch.Tensor) -> dict:
        if self.kind == "bf16":
            return {"kind": "full", "t": (theta_w.to(DEV).float() + self.value()).to(theta_w.dtype)}
        q = prepare.QArray(self.code.cb, self.scales.cpu(), self.codes.reshape(-1).to(torch.uint8).cpu(),
                           self.code.bits, self.codes.shape[1], self.shape)
        return {"kind": "delta", "q": q}


class Tree:
    """A step's reduction tree: leaf messages coded, every node's sum of two children coded again (`code` None: fp32)."""

    def __init__(self, code: Code | None):
        self.code, self.stack, self.codings = code, [], 0

    def _c(self, g: dict) -> dict:
        if self.code is None:
            return g
        self.codings += 1
        return {n: self.code(t) for n, t in g.items()}

    def push(self, g: dict) -> None:
        node = (0, self._c(g))
        while self.stack and self.stack[-1][0] == node[0]:
            lvl, left = self.stack.pop()
            node = (lvl + 1, self._c({n: left[n] + node[1][n] for n in left}))
        self.stack.append(node)

    def root(self) -> dict:
        while len(self.stack) > 1:
            _, right = self.stack.pop()
            lvl, left = self.stack.pop()
            self.stack.append((lvl + 1, self._c({n: left[n] + right[n] for n in left})))
        return self.stack.pop()[1]


# ---- optimizers on the reduced gradient -------------------------------------------------------------------------------

class Opt:
    def __init__(self, kind: str, shapes: dict, b1: float, b2: float):
        self.kind, self.b1, self.b2, self.t = kind, b1, b2, 0
        self.m, self.v, self.r, self.c = {}, {}, {}, {}
        for n, sh in shapes.items():
            if kind == "adamw":
                self.m[n] = torch.zeros(sh, device=DEV)
                self.v[n] = torch.zeros(sh, device=DEV)
            elif kind == "adafactor":
                self.r[n] = torch.zeros(sh[0], device=DEV)
                self.c[n] = torch.zeros(sh[1], device=DEV)

    def state_bits(self) -> int:
        return 32 * sum(t.numel() for d in (self.m, self.v, self.r, self.c) for t in d.values())

    def tick(self) -> None:
        self.t += 1

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
            u = g / (torch.outer(r, c) / r.mean()).sqrt()
            return lr * u / max(1.0, float(u.pow(2).mean().sqrt()))
        if self.kind == "muon":
            return lr * max(1.0, g.shape[0] / g.shape[1]) ** 0.5 * newton_schulz(g)
        return lr * g.sign()

    def flops(self, shapes: dict) -> int:
        n = sum(math.prod(s) for s in shapes.values())
        if self.kind == "muon":
            return sum(ns_flops(s) for s in shapes.values()) + 2 * n
        return {"adamw": 12, "adafactor": 14, "sign": 2}[self.kind] * n


# ---- accounting -------------------------------------------------------------------------------------------------------

def accounting(a, shapes: dict, n_norm: int, opt: Opt) -> dict:
    """Formalism v1's training quantities for this run's configuration, in bits."""
    n = sum(math.prod(s) for s in shapes.values())
    msg = 32 * n if a.msg_bits >= 32 else n * a.msg_bits + 16 * (n // a.group)
    msg += 32 * n_norm
    state = opt.state_bits() + 64 * n_norm
    store = 16 * n if a.store == "bf16" else n * int(a.store[1:]) + 16 * (n // a.group)
    store += 32 * n_norm
    dgrad = msg + state
    s_tau = a.leaf * prepare.EVAL_CTX * TOKEN_BITS
    fit = {k: {"training": 2 * dgrad + s_tau <= kb, "update": store + s_tau <= kb} for k, kb in KS.items()}
    return {"n_matrix_entries": n, "n_norm_entries": n_norm, "message_bits": msg, "state_bits": state,
            "delta_grad_bits": dgrad, "delta_t_bits": store, "s_tau_bits": s_tau, "leaf_tokens": a.leaf * prepare.EVAL_CTX,
            "fit": fit}


# ---- the run ----------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--class", dest="klass", choices=tuple(prepare.CLASSES), required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="all", help="all, out, mlp, attn, or leaf names joined by '+' (wq+wo)")
    ap.add_argument("--blocks", default="all", help="all, or LO-HI (0-based, inclusive)")
    ap.add_argument("--norms", choices=("trained", "none"), default="trained",
                    help="trained: the gains of the trained blocks and the final norm")
    ap.add_argument("--opt", choices=("adamw", "adafactor", "muon", "sign"), required=True)
    ap.add_argument("--msg-bits", type=int, default=32)
    ap.add_argument("--store", default="q8", help="qN (1 <= N <= 8) or bf16")
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
    if not (1 <= a.msg_bits <= 8 or a.msg_bits == 32) or a.batch % a.leaf:
        raise SystemExit("delta_grad: --msg-bits is 1 to 8 or 32, and --leaf divides --batch")
    torch.backends.cuda.matmul.allow_tf32 = True

    net, rec, theta = prepare.load_start(a.start)
    params = net.params()
    for p in params.values():
        p.requires_grad_(False)
    theta_dev = {n: theta[n].to(DEV) for n in params}
    targets = targets_of(params, a.targets, a.blocks)
    lo = min(int(n.split(".")[1]) for n in targets)
    norms = []
    if a.norms == "trained":
        norms = sorted(n for n, p in params.items() if p.ndim == 1 and
                       (n.startswith("norm") or (n.startswith("blocks.") and int(n.split(".")[1]) >= lo)))
    shapes = {n: tuple(params[n].shape) for n in targets}
    if any(math.prod(s) % a.group for s in shapes.values()):
        raise SystemExit(f"delta_grad: --group {a.group} doesn't divide every target's entries")
    gen = torch.Generator(device=DEV).manual_seed(a.seed)
    stores = {n: Store(shapes[n], a.store, a.group, gen) for n in targets}
    msg = Code(a.msg_bits, a.group, gen) if a.msg_bits < 32 else None
    opt = Opt(a.opt, shapes, a.b1, a.b2)
    for n in targets + norms:
        params[n].requires_grad_(True)
    nopt = torch.optim.AdamW([params[n] for n in norms], lr=a.norm_lr, betas=(0.9, 0.95), weight_decay=0.0) \
        if norms else None
    acct = accounting(a, shapes, sum(params[n].numel() for n in norms), opt)

    tokens = a.batch * prepare.EVAL_CTX
    leaves = a.batch // a.leaf
    trainable = {"full": sorted(targets + norms)}
    per_step = prepare.train_flops(rec, trainable, [{"op": "train", "tokens": tokens}])["flops"]
    n_entries = acct["n_matrix_entries"]
    codings = (2 * leaves - 1) if msg else 0
    extra = CODE_FLOPS * n_entries * (codings + 1) + opt.flops(shapes)
    n_steps = int(a.budget_fraction * prepare.TRAIN_FLOP_BUDGET // (per_step + extra))
    hooks.plan(n_steps)
    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fwd = torch.compile(net) if a.compile else net
    print(json.dumps({"mech": "dgrad", "opt": a.opt, "msg_bits": a.msg_bits, "store": a.store, "lr": a.lr,
                      "steps": n_steps, "leaves": leaves, "targets": len(targets), "norms": len(norms),
                      "flops_per_step": per_step + extra, **{k: v for k, v in acct.items() if k != "fit"},
                      "fit": acct["fit"]}), flush=True)

    def entries() -> dict:
        e = {n: stores[n].entry(theta[n]) for n in targets}
        e.update({n: {"kind": "full", "t": params[n].detach().to(theta[n].dtype)} for n in norms})
        return e

    steps, log, losses, used = [], [], [], set()
    t0 = time.perf_counter()

    def bucket(k: int) -> int:
        return math.ceil(k * CHECKPOINTS / n_steps)

    for step in range(n_steps):
        frac = schedule(step, n_steps, a.warmup)
        rows = data.next(a.batch)
        tree, total = Tree(msg), torch.zeros((), device=DEV)
        for i in range(0, a.batch, a.leaf):
            x = rows[i: i + a.leaf]
            with torch.autocast(DEV.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            loss.backward()
            total += loss.detach() / leaves
            tree.push({n: params[n].grad for n in targets})
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
            for n in targets:
                stores[n].add_(-opt.step(n, g[n], a.lr * frac))
                params[n].data.copy_((theta_dev[n].float() + stores[n].value()).to(theta_dev[n].dtype))
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
            prepare.encode_quant(entries(), theta)
            used.add(b)

    blob = prepare.encode_quant(entries(), theta)
    model = prepare.decode(theta, blob)
    changed = sorted(n for n in theta if not prepare.same_bits(model[n], theta[n]))
    tail = float(torch.stack(losses[-max(1, len(losses) // 10):]).mean())
    src = Path(__file__).read_bytes()
    notes = {"direction": "lowrank-restart", "mech": "dgrad", "args": vars(a), "accounting": acct, "E_bytes": len(blob),
             "delta_bits_final": 8 * len(blob), "steps": n_steps, "tail_loss": round(tail, 5),
             "program": "islands/lowrank-restart/delta_grad.py", "program_sha256": hashlib.sha256(src).hexdigest(),
             "train_log": log[-40:]}
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, trainable, steps, changed=changed, notes=notes)
    print(json.dumps({"E_bytes": len(blob), "changed": len(changed), "tail_loss": round(tail, 5)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
