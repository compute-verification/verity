"""Direction `hybrid`, night 2: added capacity from θ toward the large gains (-2%, -5%) on fully trained starts, each
arm priced by its compute.

    python launch.py s-t20 --island hybrid --program grow.py --note "grow ..." -- --grow depth --depth 8 \\
        --budget-fraction 0.6

`--grow` (an arm key, like every key in ARM_KEYS):

- `none`: θ's own model, the in-architecture control;
- `depth`: `--depth N` blocks, block j a copy of θ's block ⌊jL/N⌋; a repeated copy has its output projections (wo,
  w_down) zeroed, so the grown model computes θ's function at step 0 (`--zero 0` stacks the copies unchanged instead);
- `mlp`: every MLP widened to `--mlp-mult` x d_ff, the new neurons copies of random old ones (their gate and up rows,
  plus `--noise` x each row's rms of Gaussian noise) with zeroed w_down columns, so θ's function again;
- `moe`: every MLP upcycled into `--experts` copies of itself behind a router (Gaussian, `--router-init`/sqrt(d)); a
  token goes to its top `--topk` experts, weighted by the router's softmax renormalized over them, so identical experts
  compute θ's function; Switch's load-balancing loss at `--aux`; dropless;
- `scratch`: a fresh model of `--size DxLxHEADSxD_FF` from random initialization, distilled from θ: KL to θ's softmax
  at weight `--kd`, decaying linearly to 0 over `--kd-frac` of the steps (θ's forward is priced).

Every block matrix is trained by Muon (Newton-Schulz, momentum 0.95, decoupled weight decay `--wd`; each expert's
matrix on its own), the gains, the router and an unfrozen embedding by AdamW (betas 0.9, 0.95). The tied embedding is
frozen (`--embed-lr 0`) except in `scratch`. Linear warmup over `--warmup` steps, then linear decay to 0: each arm is a
complete schedule from θ within its share of the budget, `--budget-fraction` of the run's tier's (`prepare.budget`: the
search tier's, or `launch.py --long`'s LONG_C0 x the start's C0).

Every arm is measured here: θ once, then each arm at `--evals` points of its schedule, by mean cross-entropy in nats per
token over `--eval-windows` windows of the training stream that no start read and no step trains on (the last windows
of the seed-0 order; training reads that order past the windows the start read). The bytes are the same for θ and the
arm, so the paired drop in nats is the drop in bits per byte on those windows.

The last arm is the submission (harness v3, no class). `none` encodes its changed tensors (`encode_full`) at each point,
checkpoints the harness scores. `depth`, `mlp` and `scratch` submit their model at the end as a model of its own
architecture (`encode_model`, CWMODL1: θ's family, vocabulary, context and tying; every tensor not θ's bit for bit is
E's, in fp32), and that submission is their one checkpoint, since the pin's harness takes checkpoints only from the
other encoders. lf has no experts, so a last `moe` arm submits θ unchanged and is measured here only. notes.arms holds
each arm's points with their analytic FLOPs (the arm's own, its evaluations excluded), its sizes and its state: Δ ≤ 32
bits per trained weight (E in fp32), u(1) the trained weights a token reads, Δ_grad the f32 gradient plus optimizer
state (Muon's momentum, AdamW's two moments) of the trained weights.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

import prepare
from model import CTX, LM, VOCAB, Block, Size, rope

ARM_KEYS = {"grow": ("none", "depth", "mlp", "moe", "scratch"), "depth": int, "mlp_mult": float, "experts": int,
            "topk": int, "aux": float, "router_init": float, "noise": float, "size": str, "kd": float, "kd_frac": float,
            "lr": float, "adam_lr": float, "embed_lr": float, "warmup": int, "batch": int, "micro": int, "seed": int,
            "share": float, "zero": int, "wd": float}
#: te_core takes the first E encoded in each 5% of the run's MAC budget as a checkpoint
CHECKPOINTS = 20


def parse_arms(spec: str, a: argparse.Namespace) -> list[argparse.Namespace]:
    items = []
    for arm in filter(None, (s.strip() for s in spec.split(";"))):
        kv = {}
        for pair in filter(None, arm.split(",")):
            key, _, val = pair.partition("=")
            key, val = key.strip().replace("-", "_"), val.strip()
            kind = ARM_KEYS.get(key)
            if kind is None:
                raise SystemExit(f"grow: --arms key {key!r} is not one of {sorted(ARM_KEYS)}")
            if isinstance(kind, tuple) and val not in kind:
                raise SystemExit(f"grow: --arms {key}={val} is not one of {kind}")
            kv[key] = val if isinstance(kind, tuple) or kind is str else kind(val)
        items.append(kv)
    return [argparse.Namespace(**{**vars(a), **kv}) for kv in items or [{}]]


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The orthogonalized update of a matrix, or of each matrix of a stack."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return (x.mT if tall else x).to(g.dtype)


def gather_selects(n: int, bits: int = 32, chunk_bits: int = 10) -> int:
    """The gather circuit's multiplexers per selected word over n words, as the harness prices a token lookup: a copy
    of te_core/flops.py's, since a program may import only te_core's hooks and codecs."""
    def tree(m: int, k: int) -> int:
        out = 0
        for _ in range(k):
            m = (m + m % 2) // 2
            out += m
        return out

    chunk = 1 << chunk_bits
    if n <= chunk:
        return tree(n, bits)
    full, r = divmod(n, chunk)
    return full * tree(chunk, chunk_bits) + (tree(r, chunk_bits) if r else 0) + tree(full + (r > 0), bits - chunk_bits)


def ns_flops(shape: tuple[int, ...], steps: int = 5) -> int:
    m, n = min(shape[-2:]), max(shape[-2:])
    return math.prod(shape[:-2]) * steps * (4 * m * m * n + 2 * m ** 3)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, wd: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum, "wd": wd})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5)


class MoE(nn.Module):
    """Token-choice top-k experts of one MLP's shapes, the chosen experts' outputs weighted by the router's softmax
    renormalized over them; every token goes to its k experts (no capacity limit)."""

    def __init__(self, d: int, ff: int, experts: int, topk: int):
        super().__init__()
        self.topk = topk
        self.router = nn.Parameter(torch.zeros(experts, d))
        self.w_gate = nn.Parameter(torch.zeros(experts, ff, d))
        self.w_up = nn.Parameter(torch.zeros(experts, ff, d))
        self.w_down = nn.Parameter(torch.zeros(experts, d, ff))
        self.aux = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1])
        n_exp = self.router.shape[0]
        probs = F.linear(x, self.router).float().softmax(-1)
        top, idx = probs.topk(self.topk, -1)
        w = top / top.sum(-1, keepdim=True)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in range(n_exp):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            xe = x.index_select(0, tok)
            ye = F.linear(F.silu(F.linear(xe, self.w_gate[e])) * F.linear(xe, self.w_up[e]), self.w_down[e])
            out.index_add_(0, tok, ye * w[tok, slot].unsqueeze(1))
        self.aux = n_exp * (F.one_hot(idx, n_exp).float().mean((0, 1)) * probs.mean(0)).sum()
        return out.view(h.shape)


class MoEBlock(Block):
    def __init__(self, size: Size, experts: int, topk: int):
        super().__init__(size, 0.02, 0.02)
        del self.w_gate, self.w_up, self.w_down
        self.moe = MoE(size.d_model, size.d_ff, experts, topk)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = F.rms_norm(x, (d,), self.attn_norm)
        q, k, v = (w(h).view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.wo(a)
        return x + self.moe(F.rms_norm(x, (d,), self.mlp_norm))


@torch.no_grad()
def build(theta: dict[str, torch.Tensor], cfg: dict, aa: argparse.Namespace, dev: str) -> LM:
    """The arm's model at step 0 (`--grow`)."""
    d, L, H, ff = cfg["d_model"], cfg["n_layer"], cfg["n_head"], cfg["d_ff"]
    V = cfg.get("vocab", VOCAB)
    torch.manual_seed(aa.seed)
    g = torch.Generator().manual_seed(aa.seed)
    if aa.grow == "scratch":
        dd, LL, HH, FF = (int(x) for x in aa.size.split("x"))
        with torch.device(dev):
            return LM(Size("scratch", dd, LL, HH, FF), vocab=V)
    src = [j * L // aa.depth for j in range(aa.depth)] if aa.grow == "depth" else list(range(L))
    ff2 = round(ff * aa.mlp_mult) if aa.grow == "mlp" else ff
    with torch.device(dev):
        lm = LM(Size("grown", d, len(src), H, ff2), vocab=V)
        if aa.grow == "moe":
            lm.blocks = nn.ModuleList(MoEBlock(Size("grown", d, L, H, ff), aa.experts, aa.topk) for _ in range(L))
    lm.embed.copy_(theta["embed"])
    lm.norm.copy_(theta["norm"])
    for j, i in enumerate(src):
        blk, pre = lm.blocks[j], f"blocks.{i}."
        for n in ("attn_norm", "mlp_norm"):
            getattr(blk, n).copy_(theta[pre + n])
        for n in ("wq", "wk", "wv", "wo"):
            getattr(blk, n).weight.copy_(theta[pre + n + ".weight"])
        gate, up, down = (theta[f"{pre}{n}.weight"].float() for n in ("w_gate", "w_up", "w_down"))
        if aa.grow == "moe":
            blk.moe.w_gate.copy_(gate.expand(aa.experts, -1, -1))
            blk.moe.w_up.copy_(up.expand(aa.experts, -1, -1))
            blk.moe.w_down.copy_(down.expand(aa.experts, -1, -1))
            blk.moe.router.copy_(torch.randn(aa.experts, d, generator=g) * aa.router_init / math.sqrt(d))
            continue
        if ff2 > ff:
            pick = torch.randint(ff, (ff2 - ff,), generator=g)

            def jitter(rows):
                return rows + aa.noise * rows.pow(2).mean(1, keepdim=True).sqrt() * torch.randn(rows.shape, generator=g)

            gate, up = torch.cat([gate, jitter(gate[pick])]), torch.cat([up, jitter(up[pick])])
            down = torch.cat([down, torch.zeros(d, ff2 - ff)], 1)
        blk.w_gate.weight.copy_(gate)
        blk.w_up.weight.copy_(up)
        blk.w_down.weight.copy_(down)
        if j and src[j - 1] == i and aa.zero:
            blk.wo.weight.zero_()
            blk.w_down.weight.zero_()
    return lm


def flops_per_token(lm: LM, embed_trained: bool) -> dict:
    """Analytic FLOPs a token: the forward (every weight matmul a token reads, the router included, and the causal
    attention, prepare.attention_flops a layer), and a training pass as prepare.train_flops prices one with every block
    trained (the forward, every matmul's input gradient, the trained matrices' weight gradients, twice the attention).
    A table other than θ's own (`embed_trained`) adds te_core's price of a token lookup, its gather circuit's
    multiplexers, to each pass, which prepare.train_flops leaves out."""
    V, d = lm.embed.shape
    lookup = 2 * gather_selects(V) * d if embed_trained else 0
    blocks = 0
    for blk in lm.blocks:
        if isinstance(blk, MoEBlock):
            m = blk.moe
            blocks += 4 * d * d + m.topk * 3 * d * m.w_gate.shape[1] + m.router.shape[0] * d
        else:
            blocks += 4 * d * d + 3 * d * blk.w_gate.weight.shape[0]
    head = V * d
    attn = len(lm.blocks) * 2 * d * (CTX + 1)
    fwd = 2 * (blocks + head) + attn + lookup
    return {"forward": fwd, "train": fwd + 2 * (blocks + head) + 2 * blocks + (2 * head if embed_trained else 0) + 2 * attn,
            "block_macs": blocks, "head_macs": head, "lookup_flops": lookup}


def state(lm: LM, muon: list, adam: list, aa: argparse.Namespace) -> dict:
    n_muon, n_adam = sum(p.numel() for p in muon), sum(p.numel() for p in adam)
    experts = sum(blk.moe.w_gate.numel() + blk.moe.w_up.numel() + blk.moe.w_down.numel()
                  for blk in lm.blocks if isinstance(blk, MoEBlock))
    per_token = n_muon + n_adam - experts + (experts * aa.topk // aa.experts if experts else 0)
    return {"parameters": sum(p.numel() for p in lm.parameters()), "trained": n_muon + n_adam, "muon": n_muon,
            "adamw": n_adam, "delta_bits_upper": 32 * (n_muon + n_adam), "u1_bits_upper": 32 * per_token,
            "delta_grad_bytes": 8 * n_muon + 12 * n_adam,
            "delta_grad_rule": "f32 gradient + Muon momentum (8 B a weight), f32 gradient + AdamW m, v (12 B)"}


def fail(msg: str):
    """A refusal the harness's log keeps (it records only the exit code)."""
    print(json.dumps({"error": msg}), flush=True)
    raise SystemExit(msg)


def lr_frac(step: int, n: int, warmup: int) -> float:
    return min(1.0, (step + 1) / warmup) * (1 - step / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grow", choices=ARM_KEYS["grow"], default="none")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--mlp-mult", type=float, default=2.0)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--aux", type=float, default=0.01)
    ap.add_argument("--router-init", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--zero", type=int, default=1, help="depth: 1 zeroes a repeated copy's wo and w_down (θ's function "
                    "at step 0), 0 stacks the copies as they are")
    ap.add_argument("--wd", type=float, default=0.0, help="Muon's decoupled weight decay (p *= 1 - lr x wd a step)")
    ap.add_argument("--size", default="", help="scratch: d_model x n_layer x n_head x d_ff, e.g. 768x8x12x2048")
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--kd-frac", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.015, help="Muon's rate")
    ap.add_argument("--adam-lr", type=float, default=1e-3, help="the gains and the router")
    ap.add_argument("--embed-lr", type=float, default=0.0, help="0 freezes the tied embedding (not in scratch)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share", type=float, default=0.0, help=argparse.SUPPRESS)
    ap.add_argument("--arms", default="", help="';'-separated arms of KEY=VALUE overrides (ARM_KEYS), each from θ with an "
                    "equal share of the budget unless it names its share")
    ap.add_argument("--evals", type=int, default=4, help="in-job measurements per arm, at k/EVALS of its steps")
    ap.add_argument("--eval-windows", type=int, default=512)
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = prepare.DEVICE
    net, rec, theta = prepare.load_start(a.start)
    cfg = rec["config"]
    tier = prepare.budget(a.start)
    if cfg["family"] != "lf":
        raise SystemExit("grow: lf starts only")
    arms = parse_arms(a.arms, a)
    fixed, free = sum(aa.share for aa in arms), [aa for aa in arms if not aa.share]
    if fixed > 1 + 1e-9 or (free and fixed >= 1):
        raise SystemExit(f"grow: arm shares {[aa.share for aa in arms]} don't fit one budget")
    for aa in free:
        aa.share = (1 - fixed) / len(free)
    for aa in arms:
        if aa.grow == "depth" and aa.depth <= cfg["n_layer"]:
            raise SystemExit(f"grow: depth={aa.depth} doesn't grow θ's {cfg['n_layer']} blocks")
        if aa.grow == "scratch" and not aa.size:
            raise SystemExit("grow: scratch needs --size DxLxHxFF")

    # the seed-0 order: the start read its first step x batch windows; the last --eval-windows are measured, never trained
    train = prepare.TrainData(rec["tokenizer"], seed=0)
    held = prepare.TrainData(rec["tokenizer"], seed=0)
    prov = rec.get("provenance", {})
    seen = int(prov["step"]) * int(prov["batch"]) if {"step", "batch"} <= set(prov) and prov.get("data_seed", 0) == 0 else 0
    n_win = len(train.order)
    held.order = held.order[n_win - a.eval_windows:]
    train.order = train.order[seen: n_win - a.eval_windows]
    data = {"windows": n_win, "start_read": seen, "measured": a.eval_windows, "trainable": len(train.order),
            "provenance": bool(seen)}
    V = net.inner.embed.shape[0]
    theta_fwd = flops_per_token(net.inner, False)["forward"]
    steps: list[dict] = []
    t0 = time.perf_counter()

    @torch.no_grad()
    def measure(fwd, per_token: int) -> torch.Tensor:
        """Mean nats a token of each measured window (one step: a TrainData.next of the held windows)."""
        rows = held.next(a.eval_windows)
        out = []
        for i in range(0, len(rows), a.micro):
            x = rows[i: i + a.micro]
            with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                logits = fwd(x[:, :-1])
            out.append(F.cross_entropy(logits.float().transpose(1, 2), x[:, 1:], reduction="none").mean(1))
        steps.append({"op": "extra", "flops": per_token * a.eval_windows * CTX, "what": "in-job measurement (forward)",
                      "tokens": a.eval_windows * CTX})
        return torch.cat(out)

    base = measure(net.logits, theta_fwd)
    base_mean = float(base.mean())
    print(json.dumps({"theta_nats": round(base_mean, 6), "data": data, "s": round(time.perf_counter() - t0, 1)}),
          flush=True)
    names = sorted(theta)
    blob, changed, arms_out, snaps_at = None, [], [], []
    for arm, aa in enumerate(arms):
        lm = build(theta, cfg, aa, dev)
        if aa.grow == "none":
            assert sorted(n for n, _ in lm.named_parameters()) == names
        embed_lr = aa.embed_lr if aa.grow != "scratch" else (aa.embed_lr or aa.adam_lr)
        if not embed_lr:
            # θ's own table, whose token lookup the harness doesn't price (a bitwise copy of it is another table)
            lm.embed = net.inner.embed
        lm.embed.requires_grad_(embed_lr > 0)
        muon = [p for n, p in lm.named_parameters() if p.ndim >= 2 and n != "embed" and "router" not in n]
        gains = [p for n, p in lm.named_parameters() if p.ndim == 1 or "router" in n]
        opts = [Muon(muon, lr=aa.lr, wd=aa.wd)]
        groups = [{"params": gains, "lr": aa.adam_lr}] + ([{"params": [lm.embed], "lr": embed_lr}] if embed_lr else [])
        opts.append(torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=dev.startswith("cuda")))
        base_lrs = [[grp["lr"] for grp in o.param_groups] for o in opts]
        ft = flops_per_token(lm, embed_lr > 0)
        tokens = aa.batch * CTX
        ns = sum(ns_flops(tuple(p.shape)) for p in muon)
        kd_tok = theta_fwd if aa.grow == "scratch" and aa.kd > 0 else 0
        per_step = ft["train"] * tokens + ns + kd_tok * tokens
        share = aa.share * a.budget_fraction * tier["train_flops"] - aa.evals * ft["forward"] * a.eval_windows * CTX
        n = int(share // per_step)
        if n < aa.evals or n < 1:
            fail(f"grow: arm {arm}'s share buys {n} steps, fewer than its {aa.evals} measurements")
        if n * aa.batch > len(train.order):
            fail(f"grow: arm {arm} reads {n * aa.batch} windows, {len(train.order)} are unread")
        train.at = 0
        last = arm == len(arms) - 1
        marks = {round(n * j / aa.evals) for j in range(1, aa.evals + 1)}
        enc_marks = {max(1, round(n * j / CHECKPOINTS)) for j in range(1, CHECKPOINTS + 1)}
        st = state(lm, muon, gains + ([lm.embed] if embed_lr else []), aa)
        overrides = {k: v for k, v in vars(aa).items() if k in ARM_KEYS and v != vars(a).get(k)}
        out = {"arm": arm, **overrides, "grow": aa.grow, "steps": n, "tokens": n * tokens, "flops_per_step": per_step,
               "flops_per_token": ft, "planned_flops": n * per_step, "state": st, "points": []}
        print(json.dumps({k: v for k, v in out.items() if k != "points"}), flush=True)
        moe = [blk.moe for blk in lm.blocks if isinstance(blk, MoEBlock)]
        for k in range(n):
            frac = lr_frac(k, n, aa.warmup)
            for o, lrs in zip(opts, base_lrs):
                for grp, lr in zip(o.param_groups, lrs):
                    grp["lr"] = lr * frac
            alpha = aa.kd * max(0.0, 1 - k / (aa.kd_frac * n)) if kd_tok else 0.0
            rows = train.next(aa.batch)
            total = 0.0
            for i in range(0, aa.batch, aa.micro):
                x = rows[i: i + aa.micro]
                part = len(x) / aa.batch
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    logits = lm(x[:, :-1]).float().view(-1, V)
                    ce = F.cross_entropy(logits, x[:, 1:].reshape(-1))
                    loss = ce
                    if moe:
                        loss = loss + aa.aux * sum(m.aux for m in moe) / len(moe)
                    if kd_tok:
                        with torch.no_grad():
                            teach = net.logits(x[:, :-1]).float().view(-1, V).log_softmax(-1)
                        kl = F.kl_div(logits.log_softmax(-1), teach, log_target=True, reduction="batchmean")
                        loss = (1 - alpha) * ce + alpha * kl
                (loss * part).backward()
                total += float(ce.detach()) * part
            torch.nn.utils.clip_grad_norm_([p for p in lm.parameters() if p.requires_grad], 1.0)
            for o in opts:
                o.step()
                o.zero_grad(set_to_none=True)
            steps.append({"op": "extra", "flops": ft["train"] * tokens + kd_tok * tokens,
                          "what": f"arm {arm} train step ({aa.grow})", "tokens": tokens})
            steps.append({"op": "extra", "flops": ns, "what": "muon newton-schulz"})
            if (k + 1) % 100 == 0:
                print(json.dumps({"arm": arm, "step": k + 1, "loss": round(total, 5), "lr_frac": round(frac, 4),
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            if last and aa.grow == "none" and k + 1 in enc_marks:
                changed = [nm for nm, p in lm.named_parameters() if p.requires_grad]
                blob = prepare.encode_full({nm: p.detach().to(theta[nm].dtype) for nm, p in lm.named_parameters()
                                            if p.requires_grad})
                snaps_at.append({"arm": arm, "step": k + 1, "e_bytes": len(blob)})
            if k + 1 in marks:
                got = measure(lm, ft["forward"])
                diff = base - got
                pt = {"step": k + 1, "train_flops": (k + 1) * per_step, "nats": round(float(got.mean()), 6),
                      "drop_pct": round(100 * float(diff.mean()) / base_mean, 4),
                      "se_pct": round(100 * float(diff.std()) / math.sqrt(len(diff)) / base_mean, 4),
                      "s": round(time.perf_counter() - t0, 1)}
                out["points"].append(pt)
                print(json.dumps({"arm": arm, **pt}), flush=True)
        if last and aa.grow in ("depth", "mlp", "scratch"):
            s = lm.size
            own = {**cfg, "d_model": s.d_model, "n_layer": s.n_layer, "n_head": s.n_head, "d_ff": s.d_ff}
            new = {nm: p.detach().float().cpu() for nm, p in lm.named_parameters()}
            new = {nm: v for nm, v in new.items() if nm not in theta or not prepare.same_bits(v, theta[nm])}
            changed = sorted(new)
            blob = prepare.encode_model(own, new)
            out["config"] = own
            snaps_at.append({"arm": arm, "step": n, "e_bytes": len(blob), "format": "CWMODL1"})
        arms_out.append(out)
        del lm, opts, muon, gains, moe
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()
    if blob is None:
        blob = prepare.encode_full({})
    kind = arms[-1].grow
    prepare.submit(Path(a.out), a.start, blob, prepare.decode(theta, blob),
                   {"full": changed, "adapters": {}}, steps, changed=changed,
                   notes={"method": "grow", "args": vars(a), "tier": tier, "theta_nats": base_mean, "data": data,
                          "arms": arms_out,
                          "e_checkpoints": snaps_at,
                          "submitted": {"none": "the last arm's changed tensors (encode_full)",
                                        "moe": "θ unchanged: lf has no experts"}.get(
                              kind, "the last arm's model of its own architecture (encode_model, CWMODL1)")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
