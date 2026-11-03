"""Island `hybrid`, program `qat.py`: large gains inside θ's architecture at a small update, by quantization-aware training
of the block matrices (the quant island's QAT, islands/quant/train.py, rebuilt for speed and for mixed structures).

Each coded matrix's weight is θ + Q(V) in every step's forward: V an fp32 latent update that only the optimizer touches
(Muon, through a straight-through gradient), Q its code, so every checkpoint and the submission are exactly a CWQUAN1 E
(prepare.encode_quant). Codes: `sign` (1 bit), `tern` (-1, 0, 1), `rtn{b}` (b-bit round to nearest, levels
-(2^(b-1)-1)..2^(b-1)-1), each times an f16 scale of `--mult` (or the code's own, `rtn4@0.6`) times the group's mean |V|
(`--group`: -1 a tensor, 0 a row, else G entries); `fp` trains the matrix unquantized (a CWQUAN1 full entry). The norm gains at and above the lowest
trained block, and the final norm, train in fp32 by AdamW and are full entries; the tied embedding stays at θ.

- Matrices of one shape and code are stacked, with their gradients accumulated into one buffer: the code, Muon's batched
  Newton–Schulz and the weight rebuild are a few ops a step whatever the depth (te_core's meter dispatches every aten op
  through Python, so per-matrix QAT cost ~0.3 s a step on s100).
- `--code` maps block matrices to codes, `BLOCKS:LEAVES=CODE` comma-separated (train.py's SPEC; the first match wins;
  unmatched matrices stay at θ), e.g. `all:all=rtn3`, `0-1:all=tern,2-5:all=rtn3` or `all:attn=rtn4@0.7,all:mlp=rtn4@0.5`.
- `--stages A/B/...` trains the coded matrices in turn, each stage a SPEC (`--passes` repeats the list, `--shares` splits
  the steps): a stage's latents and optimizer state exist only while it runs, and a matrix a later stage revisits
  restarts its latent from its code, so Δ_grad is the largest stage's. The rate follows one schedule over the run
  (`--stage-sched global`) or restarts in each stage (`each`).
- `--momentum-dtype bf16` keeps Muon's buffer in bf16.
- `--qat-from F` (one stage only) trains the coded matrices unquantized, θ + V, for the first share F of the steps, and
  from then on through their code; checkpoints before the switch hold those matrices as full entries.
- `--embed-rank r` adds a rank-r adapter on the tied embedding beside the coded blocks, trained in every stage (AdamW
  `--embed-lr`): W' = θ + s · Q(B) Q(A), s = sqrt(d / r), its factors `--embed-bits` round to nearest (straight-through;
  A a scale per row, B a scale per 64 rows), a CWQUAN1 lora entry. Every step's lookup and head read W' as
  prepare.lora_merge builds it (the head under bf16 autocast, as the evaluator multiplies), and the backward reaches A
  and B through rank-r products, so no dense embedding gradient is formed or paid.

The notes state, per stage, the trained tensors, codes, weights, Δ_grad as the scorer counts it (gradient + momentum per
block weight, the gains' and factors' fp32 gradient and AdamW m and v) and the update store beside it (the fp32 latent of
each coded weight and factor entry), and the submission's Δ (|E|: `--xz min` submits the smallest of XZ_TRIES, every
size in the notes) beside each code's order-0 entropy. `notes.delta_grad` (the largest stage's) splits Δ_grad into
grad_bits, adam_bits and muon_bits (`bits` is their sum) and gives the latents apart as latent_bits, which the update
application clause reads as Δ_t.

    python launch.py reference s100 --island hybrid --program qat.py --note "qat: ..." -- --klass reference \
        --code all:all=rtn3 --muon-lr 0.015
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from te_core import hooks

import prepare

LEAVES = ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down")
GROUPS = {"qkv": ("wq", "wk", "wv"), "attn": ("wq", "wk", "wv", "wo"), "gu": ("w_gate", "w_up"),
          "mlp": ("w_gate", "w_up", "w_down"), "all": LEAVES}
CODE = re.compile(r"(?:sign|tern|rtn[2-8])(?:@\d*\.?\d+)?|fp|-")
#: the harness's checkpoint count: with `hooks.plan`, the first E encoded in each 5% of the planned steps
CHECKPOINTS = 20
#: LZMA2 settings over preset 9e for the submission (`--xz min` takes the smallest; decoders read any). On dense 4-bit
#: codes order-0 literals (lc = lp = pb = 0) are smallest, by a few tenths of a percent; on 8-bit codes a depth-1 match
#: search is 2-3% under 9e, whose short matches cost more than they save on near-random bytes.
XZ_TRIES = {"9e": {}, "lit0": {"lc": 0, "lp": 0, "pb": 0},
            "hc3d1-lit0": {"mf": lzma.MF_HC3, "nice_len": 273, "depth": 1, "lc": 0, "lp": 0, "pb": 0},
            "bt4d1": {"mf": lzma.MF_BT4, "nice_len": 273, "depth": 1}}


def fail(msg: str):
    """Exit with `msg` also on stdout: the harness keeps a program's output, not its exit message."""
    print(json.dumps({"error": msg}), flush=True)
    raise SystemExit(f"hybrid qat: {msg}")


def spec_names(spec: str, L: int) -> list[str]:
    """train.py's SPEC: comma-separated BLOCKS:LEAVES, BLOCKS `all`, `i`, `i-j` or a negative index from the top."""
    out = []
    for item in filter(None, spec.split(",")):
        blocks, _, leaves = item.partition(":")
        if blocks == "all":
            idx = range(L)
        elif re.fullmatch(r"\d+-\d+", blocks):
            lo, hi = map(int, blocks.split("-"))
            idx = range(lo, hi + 1)
        else:
            idx = [int(blocks) % L]
        names = [x for leaf in (leaves or "all").split("+") for x in GROUPS.get(leaf, (leaf,))]
        if set(names) - set(LEAVES):
            fail(f"unknown leaves {sorted(set(names) - set(LEAVES))}")
        out += [f"blocks.{i}.{n}.weight" for i in idx for n in names]
    return out


def parse_codes(spec: str, L: int) -> dict[str, str]:
    """`BLOCKS:LEAVES=CODE,...` -> {matrix: code}, the first match winning; `-` (or no match) keeps a matrix at θ."""
    codes: dict[str, str] = {}
    for item in filter(None, spec.split(",")):
        sel, _, code = item.rpartition("=")
        if not sel or not CODE.fullmatch(code):
            fail(f"--code item {item!r} is not BLOCKS:LEAVES=CODE (CODE sign|tern|rtnB, optionally @MULT, or fp|-)")
        for n in spec_names(sel, L):
            codes.setdefault(n, code)
    return {n: c for n, c in codes.items() if c != "-"}


def block_of(name: str) -> int | None:
    m = re.match(r"blocks\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """The orthogonalized update of each matrix of a stack."""
    a, b, c = 3.4445, -4.7750, 2.0315
    # CPU bf16 matmuls occasionally return 2**24-sized entries here (CPU smokes only)
    x = g.bfloat16() if g.is_cuda else g.float()
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    return (x.mT if tall else x).to(g.dtype)


def ns_flops(shape: tuple[int, ...], steps: int = 5) -> int:
    m, n = min(shape[-2:]), max(shape[-2:])
    return math.prod(shape[:-2]) * steps * (4 * m * m * n + 2 * m ** 3)


def code_bits(code: str) -> int:
    return 1 if code == "sign" else 2 if code == "tern" else int(code[3:])


def qmax_of(code: str) -> int:
    return 1 if code == "tern" else 2 ** (code_bits(code) - 1) - 1


def group_size(shape: tuple[int, int], group: int) -> int:
    g = shape[0] * shape[1] if group < 0 else shape[1] if group == 0 else group
    if (shape[0] * shape[1]) % g:
        fail(f"group {g} does not divide a {shape} matrix")
    return g


@torch.no_grad()
def quantize(v: torch.Tensor, code: str, g: int, mult: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Q(V) of a stack V (G x m x n) in fp32, its integer levels k (fp32, sign: -1/1) and f16-exact scales (G x groups x 1):
    a scale of `mult` times the group's mean |V| rounded to f16, then sign, or round(V / s) clamped to ±qmax (a group of
    zeros quantizes to zeros), so Q = k · s exactly in fp32, as CWQUAN1 decodes it."""
    vg = v.reshape(v.shape[0], -1, g)
    s = (mult * vg.abs().mean(-1, keepdim=True)).clamp(max=65504.0).half().float()
    if code == "sign":
        k = torch.where(vg >= 0, 1.0, -1.0)
    else:
        qm = qmax_of(code)
        k = torch.clamp(torch.round(vg / torch.where(s > 0, s, 1.0)), -qm, qm) * (s > 0)
    return (k * s).reshape(v.shape), k, s


def qarray(k: torch.Tensor, s: torch.Tensor, code: str, g: int, shape: tuple[int, int]) -> prepare.QArray:
    """One matrix's CWQUAN1 delta from its levels k (groups x g) and scales (groups x 1)."""
    if code == "sign":
        cb, codes = torch.tensor([-1.0, 1.0]), (k > 0)
    else:
        qm = qmax_of(code)
        cb = torch.zeros(2 ** code_bits(code))
        cb[: 2 * qm + 1] = torch.arange(-qm, qm + 1, dtype=torch.float32)
        codes = k + qm
    return prepare.QArray(cb.half(), s.reshape(-1).half().cpu(), codes.reshape(-1).to(torch.uint8).cpu(),
                          code_bits(code), g, shape)


@torch.no_grad()
def code_entropy(stacks: list) -> dict[str, float]:
    """The order-0 entropy of each coded matrix's levels in bits per weight, averaged by weights per code and over all
    codes ("all"): what an order-0 coder adapted per matrix would spend, to tell a code's rate from xz's packing."""
    tot: dict[str, list[float]] = {}
    for st in stacks:
        if st.code == "fp" or not st.coded or st.k is None:
            continue
        for k in st.k.reshape(st.k.shape[0], -1).long():
            p = torch.bincount(k - k.min()).double()
            p = p[p > 0] / k.numel()
            h = float(-(p * p.log2()).sum())
            for key in (f"{st.code}@{st.mult:g}", "all"):
                t = tot.setdefault(key, [0.0, 0])
                t[0], t[1] = t[0] + h * k.numel(), t[1] + k.numel()
    return {key: round(s / n, 4) for key, (s, n) in tot.items()}


B_ROWS = 64


@torch.no_grad()
def factor_quant(x: torch.Tensor, bits: int, rows: int) -> tuple[torch.Tensor, prepare.QArray]:
    """`x` at `bits`-bit absmax round to nearest, one f16 scale per `rows` rows: (its fp32 value, its QArray)."""
    qm = 2 ** (bits - 1) - 1
    shape = tuple(x.shape)
    rows = rows if x.shape[0] % rows == 0 else 1
    xg = x.float().reshape(-1, rows * x.shape[1])
    s = (xg.abs().amax(1, keepdim=True) / qm).clamp(max=65504.0).half().float()
    k = torch.clamp(torch.round(xg / torch.where(s > 0, s, 1.0)), -qm, qm) * (s > 0)
    cb = torch.zeros(2 ** bits)
    cb[: 2 * qm + 1] = torch.arange(-qm, qm + 1, dtype=torch.float32)
    q = prepare.QArray(cb.half(), s.reshape(-1).half().cpu(), (k + qm).reshape(-1).to(torch.uint8).cpu(), bits,
                       xg.shape[1], shape)
    return (k * s).reshape(shape), q


class LoraLookup(torch.autograd.Function):
    """F.embedding(idx, W') for W' = θ + s · B A (given merged), whose backward reaches A and B only."""

    @staticmethod
    def forward(ctx, idx, w, a, b, s):
        ctx.save_for_backward(idx, a, b)
        ctx.s = s
        return F.embedding(idx, w)

    @staticmethod
    def backward(ctx, gx):
        idx, a, b = ctx.saved_tensors
        i, g = idx.reshape(-1), gx.reshape(-1, gx.shape[-1]).float()
        ga = ctx.s * (b[i].T @ g)
        gb = torch.zeros_like(b).index_add_(0, i, ctx.s * (g @ a.T))
        return None, None, ga, gb, None


class LoraHead(torch.autograd.Function):
    """Logits x W'^T in bf16 (the evaluator's autocast matmul of the merged weight), whose backward gives x's gradient
    through W' and A's and B's through rank-r products: dB = s G A^T = s gy^T (x A^T), dA = s B^T G = s (gy B)^T x."""

    @staticmethod
    def forward(ctx, x, w, a, b, s):
        ctx.save_for_backward(x, w, a, b)
        ctx.s = s
        return F.linear(x.bfloat16(), w.bfloat16())

    @staticmethod
    def backward(ctx, gy):
        x, w, a, b = ctx.saved_tensors
        gy2, x2 = gy.reshape(-1, gy.shape[-1]).bfloat16(), x.reshape(-1, x.shape[-1]).bfloat16()
        gx = (gy2 @ w.bfloat16()).reshape(x.shape).to(x.dtype)
        gb = ctx.s * (gy2.T @ (x2 @ a.bfloat16().T)).float()
        ga = ctx.s * ((gy2 @ b.bfloat16()).T @ x2).float()
        return gx, None, ga, gb, None


class EmbedAdapter:
    """The rank-r adapter on the tied embedding: fp32 latents A (r x d) and B (V x r), their codes, and the merged W'."""

    def __init__(self, theta_e: torch.Tensor, rank: int, bits: int, lr: float, seed: int, dev: str):
        V, d = theta_e.shape
        gen = torch.Generator(device=dev).manual_seed(seed)
        self.theta, self.rank, self.bits, self.s = theta_e, rank, bits, math.sqrt(d / rank)
        self.a = torch.nn.Parameter(torch.randn(rank, d, device=dev, generator=gen) / math.sqrt(d))
        self.b = torch.nn.Parameter(torch.zeros(V, rank, device=dev))
        self.opt = torch.optim.AdamW([self.a, self.b], lr=lr, betas=(0.9, 0.95), weight_decay=0.0,
                                     fused=dev.startswith("cuda"))
        self.lr = lr
        self.w = None
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        aq, self.qa = factor_quant(self.a, self.bits, 1)
        bq, self.qb = factor_quant(self.b, self.bits, B_ROWS)
        w = prepare.lora_merge(self.theta, aq, bq, self.s)
        # te_core's counter resets its tags each step and keeps every untagged tensor a step reads (new_tensors), so
        # what the next step reads keeps its storage: a fresh tensor each step would be held forever.
        if self.w is None:
            self.aq, self.bq, self.w = aq, bq, w
        else:
            torch._foreach_copy_([self.aq, self.bq, self.w], [aq, bq, w])

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The coded factors with a straight-through gradient to the latents."""
        return self.a + (self.aq - self.a).detach(), self.b + (self.bq - self.b).detach()

    def step(self, frac: float) -> None:
        for grp in self.opt.param_groups:
            grp["lr"] = self.lr * frac
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.refresh()

    def entry(self) -> dict:
        return {"kind": "lora", "a": self.qa, "b": self.qb, "scale": self.s}

    def state_bytes(self) -> int:
        return 12 * (self.a.numel() + self.b.numel())

    def latent_bytes(self) -> int:
        return 4 * (self.a.numel() + self.b.numel())


def adapter_forward(lm, blocks_fn, ad: EmbedAdapter):
    """The LM's forward with W' at its lookup and head (model.LM.hidden and .top with the merged weight)."""

    def fwd(idx):
        a, b = ad.factors()
        x = LoraLookup.apply(idx, ad.w, a, b, ad.s)
        return LoraHead.apply(blocks_fn(x), ad.w, a, b, ad.s)

    return fwd


def blocks_and_norm(lm):
    def run(x):
        t = x.shape[1]
        cos, sin = lm.cos[:t], lm.sin[:t]
        for blk in lm.blocks:
            x = blk(x, cos, sin)
        return F.rms_norm(x, (x.shape[-1],), lm.norm)

    return run


class Schedule:
    """The run's steps, told to the harness (`hooks.plan`), and which encode an E: the first E in each 5% of the steps
    is a checkpoint, so each 5% gets one at its last step."""

    def __init__(self, total: int):
        self.total = total
        hooks.plan(total)
        self.cps = {max(0, total * b // CHECKPOINTS - 1) for b in range(1, CHECKPOINTS + 1)}

    def due(self, g: int) -> bool:
        return g in self.cps or g == self.total - 1


def lr_frac(step: int, n: int, warmup: int, sched: str, decay_frac: float) -> float:
    w = min(1.0, (step + 1) / max(1, warmup))
    if sched == "linear":
        return w * (1 - step / n)
    tail = max(1, round(decay_frac * n))
    return w * min(1.0, (n - step) / tail)


def fresh_order(data, rec: dict) -> dict:
    """Drop the windows the start read from the stream's order (its provenance: default_rng(data_seed), step x batch)."""
    prov = rec.get("provenance", {})
    if not {"step", "batch", "data_seed"} <= set(prov):
        return {"fresh": False, "why": "the start's provenance names no data order"}
    n_seq = (len(data.tokens) - 1) // data.ctx
    seen = np.random.default_rng(prov["data_seed"]).permutation(n_seq)[: prov["step"] * prov["batch"]]
    keep = np.ones(n_seq, dtype=bool)
    keep[seen] = False
    before = len(data.order)
    data.order = data.order[keep[data.order]]
    return {"fresh": True, "windows": n_seq, "seen": int(len(seen)), "kept": int(len(data.order)), "of": before}


class Stack:
    """The matrices of one shape and code: θ, the trained weights' gradient buffer (each matrix's .grad is a view of it,
    so backward accumulates into it), and, while a stage trains them, the latent V and Muon's buffer."""

    def __init__(self, names: list[str], code: str, params: dict, theta_dev: dict, group: int, mult: float):
        self.names = names
        self.code, _, m = code.partition("@")
        self.mult = float(m) if m else mult
        self.shape = tuple(params[names[0]].shape)
        self.g = group_size(self.shape, group) if self.code != "fp" else 0
        self.theta = torch.stack([theta_dev[n] for n in names])
        self.params = [params[n] for n in names]
        self.k = self.s = None
        self.v = self.buf = self.grad = None
        self.copies = 0
        self.coded, self.trained = self.code != "fp", False

    def collect(self) -> None:
        """Make the buffer hold every matrix's gradient: backward accumulates into an existing .grad in place, so this
        copies only a gradient autograd put elsewhere (counted in the notes)."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                self.grad[i].zero_()
            elif p.grad.data_ptr() != self.grad[i].data_ptr():
                self.grad[i].copy_(p.grad)
                self.copies += 1
            else:
                continue
            p.grad = self.grad[i]

    def begin(self, momentum_dtype: torch.dtype) -> None:
        """Start training: V from the current code (scaled by the stack's mult so it requantizes to the same code), a
        zero momentum buffer, and the gradient buffer as every matrix's .grad."""
        w = torch.stack([p.detach() for p in self.params])
        if self.code == "fp":
            self.v = w.clone()
        else:
            d = w - self.theta
            if self.k is not None:
                mk = self.k.abs().mean(-1, keepdim=True).clamp(min=1e-12)
                d = (self.k * self.s / (self.mult * mk) if self.code != "sign" else self.k * self.s).reshape(w.shape)
            self.v = d.float()
        self.buf = torch.zeros_like(self.v, dtype=momentum_dtype)
        self.grad = torch.zeros_like(self.v)
        self.trained = True
        for i, p in enumerate(self.params):
            p.requires_grad_(True)
            p.grad = self.grad[i]

    def end(self) -> None:
        for p in self.params:
            p.grad = None
            p.requires_grad_(False)
        self.v = self.buf = self.grad = None

    @torch.no_grad()
    def muon(self, lr: float, momentum: float) -> None:
        self.buf.mul_(momentum).add_(self.grad)
        u = newton_schulz(self.grad.add(self.buf, alpha=momentum))
        self.v.add_(u, alpha=-lr * max(1.0, self.shape[0] / self.shape[1]) ** 0.5)
        self.grad.zero_()

    @torch.no_grad()
    def rebuild(self) -> None:
        if self.code == "fp":
            w = self.v
        elif not self.coded:
            w = self.theta + self.v
        else:
            q, self.k, self.s = quantize(self.v, self.code, self.g, self.mult)
            w = self.theta + q
        torch._foreach_copy_(self.params, list(w.unbind(0)))

    def entries(self) -> dict[str, dict]:
        if self.code == "fp" or not self.coded:
            return {n: {"kind": "full", "t": p.detach()} for n, p in zip(self.names, self.params)} if self.trained else {}
        if self.k is None:
            return {}
        return {n: {"kind": "delta", "q": qarray(self.k[i], self.s[i], self.code, self.g, self.shape)}
                for i, n in enumerate(self.names)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--klass", required=True, choices=sorted(prepare.CLASSES))
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code", default="all:all=rtn3",
                    help="BLOCKS:LEAVES=CODE,... (CODE sign|tern|rtnB, optionally @MULT, or fp|-)")
    ap.add_argument("--group", type=int, default=0, help="entries per scale: -1 a tensor, 0 a row, else G")
    ap.add_argument("--mult", type=float, default=1.0, help="the scale over the group's mean |V|")
    ap.add_argument("--stages", default="", help="SPEC/SPEC/...: the coded matrices trained in turn (default: all at once)")
    ap.add_argument("--shares", default="", help="the stages' shares of the steps, comma-separated (default: equal)")
    ap.add_argument("--passes", type=int, default=1, help="repeat the stage list")
    ap.add_argument("--stage-sched", choices=("global", "each"), default="global")
    ap.add_argument("--qat-from", type=float, default=0.0,
                    help="the share of the steps after which the coded matrices read their code (before: theta + V)")
    ap.add_argument("--muon-lr", type=float, default=0.015)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--momentum-dtype", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--adam-lr", type=float, default=3e-4, help="the norm gains' AdamW rate")
    ap.add_argument("--no-gains", action="store_true", help="keep the norm gains at θ")
    ap.add_argument("--embed-rank", type=int, default=0, help="a rank-r adapter on the tied embedding (0: none)")
    ap.add_argument("--embed-bits", type=int, default=8)
    ap.add_argument("--embed-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="the embedding adapter's A")
    ap.add_argument("--sched", choices=("linear", "wsd"), default="linear")
    ap.add_argument("--decay-frac", type=float, default=0.2, help="wsd: the decaying share")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=32)
    ap.add_argument("--data-seed", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="skip the windows the start trained on")
    ap.add_argument("--budget-fraction", type=float, default=0.98)
    ap.add_argument("--xz", choices=("1", "9e", "min"), default="9e",
                    help="the submission's xz: preset 1, 9e, or the smallest of XZ_TRIES (checkpoints: 1)")
    a = ap.parse_args()

    t_start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    net, rec, theta = prepare.load_start(a.start)
    dev = prepare.DEVICE
    params = net.params()
    L = rec["config"]["n_layer"]
    codes = parse_codes(a.code, L)
    if not codes:
        fail("--code selects no matrix")
    for p in params.values():
        p.requires_grad_(False)
    stage_specs = [s for s in a.stages.split("/") if s] or ["all:all"]
    base_sets = []
    for s in stage_specs:
        names = sorted({n for n in spec_names(s, L) if n in codes})
        if not names:
            fail(f"stage {s!r} trains no coded matrix")
        base_sets.append(names)
    stage_sets = base_sets * a.passes

    # one stack per shape, code and set of stages, so each stage trains whole stacks
    theta_dev = {n: theta[n].to(dev) for n in codes}
    groups: dict[tuple, list[str]] = {}
    for n in sorted(codes, key=lambda n: (block_of(n), n)):
        member = tuple(j for j, s in enumerate(base_sets) if n in s)
        groups.setdefault((tuple(params[n].shape), codes[n], member), []).append(n)
    stacks = [Stack(ns, code, params, theta_dev, a.group, a.mult) for (_, code, _), ns in groups.items()]
    stack_of = {n: st for st in stacks for n in st.names}
    del theta_dev
    gains_all = sorted(n for n, p in params.items() if p.ndim == 1)
    final_norm = [n for n in gains_all if block_of(n) is None]

    def gains_of(names: list[str]) -> list[str]:
        if a.no_gains:
            return []
        low = min(block_of(n) for n in names)
        return [n for n in gains_all if block_of(n) is None or block_of(n) >= low] or final_norm

    tokens = a.batch * prepare.EVAL_CTX
    per_step, ns_extra = [], []
    adapters = {"embed": a.embed_rank} if a.embed_rank else {}
    for names in stage_sets:
        tr = {"full": sorted(names + gains_of(names)), "adapters": adapters}
        per_step.append(prepare.train_flops(rec, tr, [{"op": "train", "tokens": tokens}])["flops"])
        ns_extra.append(sum(ns_flops(tuple(params[n].shape)) for n in names))
    shares = [float(x) for x in a.shares.split(",")] if a.shares else [1.0] * len(stage_specs)
    if len(shares) != len(stage_specs):
        fail("--shares needs one share per stage")
    shares = [x / sum(shares) / a.passes for x in shares] * a.passes
    room = a.budget_fraction * prepare.TRAIN_FLOP_BUDGET
    counts = [max(1, int(sh * room // (f + e))) for sh, f, e in zip(shares, per_step, ns_extra)]
    total = sum(counts)
    if total < CHECKPOINTS:
        fail(f"the budget buys {total} steps, fewer than the {CHECKPOINTS} checkpoints")
    sched = Schedule(total)
    switch = round(a.qat_from * total)
    if switch and len(stage_sets) > 1:
        fail("--qat-from needs one stage")
    for st in stacks:
        st.coded = st.coded and not switch

    data = prepare.TrainData(rec["tokenizer"], seed=a.data_seed)
    fresh = fresh_order(data, rec) if a.fresh else {"fresh": False}
    ad = None
    if a.embed_rank:
        if rec["config"]["family"] != "lf" or not rec["config"].get("tied"):
            fail("--embed-rank needs an lf start with a tied embedding")
        ad = EmbedAdapter(theta["embed"].to(dev), a.embed_rank, a.embed_bits, a.embed_lr, a.seed, dev)
        fwd = adapter_forward(net.inner, torch.compile(blocks_and_norm(net.inner)), ad)
    else:
        fwd = torch.compile(net)
    mdt = torch.bfloat16 if a.momentum_dtype == "bf16" else torch.float32
    steps: list[dict] = []
    stage_notes, g, ema, t0 = [], 0, None, time.perf_counter()
    all_gains: set[str] = set()

    def encode() -> bytes:
        ent: dict[str, dict] = {}
        for st in stacks:
            ent |= st.entries()
        ent |= {n: {"kind": "full", "t": params[n].detach()} for n in sorted(all_gains)}
        if ad is not None:
            ent["embed"] = ad.entry()
        return prepare.encode_quant(ent, theta)

    for j, names in enumerate(stage_sets):
        active = sorted({id(stack_of[n]): stack_of[n] for n in names}.values(), key=lambda st: st.names[0])
        gains = gains_of(names)
        all_gains |= set(gains)
        for n in gains:
            params[n].requires_grad_(True)
        for st in active:
            st.begin(mdt)
        adam = torch.optim.AdamW([params[n] for n in gains], lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0.0,
                                 fused=dev.startswith("cuda")) if gains else None
        n_j, first = counts[j], g
        coded_w = sum(theta[n].numel() for n in names if codes[n] != "fp")
        fp_w = sum(theta[n].numel() for n in names if codes[n] == "fp")
        gain_w = sum(params[n].numel() for n in gains)
        mom_b = 2 if mdt == torch.bfloat16 else 4
        n_ad = ad.a.numel() + ad.b.numel() if ad is not None else 0
        dg = {"trainable": coded_w + fp_w + gain_w + n_ad, "grad_bits": 32 * (coded_w + fp_w + gain_w + n_ad),
              "adam_bits": 64 * (gain_w + n_ad), "muon_bits": 8 * mom_b * (coded_w + fp_w),
              "latent_bits": 32 * (coded_w + n_ad)}
        dg["bits"] = dg["grad_bits"] + dg["adam_bits"] + dg["muon_bits"]
        dg["bytes"] = dg["bits"] // 8
        note = {"stage": j, "spec": (stage_specs * a.passes)[j], "steps": n_j, "flops_per_step": per_step[j] + ns_extra[j],
                "matrices": len(names), "coded_weights": coded_w, "fp_weights": fp_w, "gains": len(gains),
                "codes": sorted({codes[n] for n in names}),
                "delta_grad_bytes": (coded_w + fp_w) * (4 + mom_b) + gain_w * 12
                + (ad.state_bytes() if ad is not None else 0),
                "delta_grad_what": f"block weight: fp32 gradient + {a.momentum_dtype} Muon momentum; gain: fp32 gradient + "
                                   "AdamW fp32 m and v; embedding adapter factor: fp32 gradient + AdamW m and v",
                "latent_bytes": coded_w * 4 + (ad.latent_bytes() if ad is not None else 0),
                "latent_what": "the update store: fp32 latent V per coded weight and per adapter factor entry",
                "delta_grad": dg}
        print(json.dumps(note), flush=True)
        for step in range(n_j):
            if switch and g == switch:
                for st in active:
                    if st.code != "fp":
                        st.coded = True
                        st.rebuild()
            gs = g if a.stage_sched == "global" else step
            frac = lr_frac(gs, total if a.stage_sched == "global" else n_j, a.warmup, a.sched, a.decay_frac)
            if adam is not None:
                for grp in adam.param_groups:
                    grp["lr"] = a.adam_lr * frac
            rows = data.next(a.batch)
            loss_sum = 0.0
            for i in range(0, a.batch, a.micro):
                xb = rows[i: i + a.micro]
                with torch.autocast(dev.split(":")[0], dtype=torch.bfloat16):
                    logits = fwd(xb[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), xb[:, 1:].reshape(-1)) * (
                    len(xb) / a.batch)
                loss.backward()
                loss_sum += float(loss.detach())
            for st in active:
                st.collect()
            torch.nn.utils.clip_grad_norm_([p for st in active for p in st.params] + [params[n] for n in gains]
                                           + ([ad.a, ad.b] if ad is not None else []), 1.0)
            for st in active:
                st.muon(a.muon_lr * frac, a.momentum)
                st.rebuild()
            if adam is not None:
                adam.step()
                adam.zero_grad(set_to_none=True)
            if ad is not None:
                ad.step(frac)
            steps.append({"op": "train", "tokens": tokens})
            steps.append({"op": "extra", "flops": ns_extra[j], "what": "muon newton-schulz"})
            ema = loss_sum if ema is None else 0.98 * ema + 0.02 * loss_sum
            if (step + 1) % 100 == 0 or step == 0:
                print(json.dumps({"stage": j, "step": g + 1, "loss": round(loss_sum, 4), "ema": round(ema, 4),
                                  "s": round(time.perf_counter() - t0, 1)}), flush=True)
            if sched.due(g):
                encode()
            g += 1
        for st in active:
            st.end()
        for n in gains:
            params[n].requires_grad_(False)
        del adam
        stage_notes.append(note | {"global_steps": [first, g - 1], "final_loss_ema": ema})

    t_train = time.perf_counter() - t0
    blob1 = encode()
    blob, xz_sizes = blob1, {"1": len(blob1)}
    if a.xz != "1":
        raw = lzma.decompress(bytes(blob1[8:]), format=lzma.FORMAT_XZ)
        for name in (XZ_TRIES if a.xz == "min" else ("9e",)):
            b = blob1[:8] + lzma.compress(raw, format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE, filters=[
                {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME} | XZ_TRIES[name]])
            xz_sizes[name] = len(b)
            blob = b if len(b) < len(blob) else blob
    model = prepare.decode(theta, blob)
    changed = sorted(set(codes) | all_gains)
    bad = [n for n in changed if not prepare.same_bits(model[n].cpu(), params[n].detach().to(theta[n].dtype).cpu())]
    if ad is not None:
        bad += [] if prepare.same_bits(model["embed"].cpu(), ad.w.cpu()) else ["embed"]
    if bad:
        fail(f"E rebuilds {len(bad)} tensors differently from the trained ones: {bad[:3]}")
    coded = sum(theta[n].numel() for n in codes if codes[n] != "fp")
    notes = {"direction": "hybrid", "program": "qat.py", "args": vars(a), "steps": total, "qat_switch_step": switch,
             "stages": stage_notes,
             "codes": {c: sum(1 for n in codes if codes[n] == c) for c in sorted(set(codes.values()))},
             "coded_weights": coded, "E_bytes": len(blob), "E_bytes_xz1": len(blob1),
             "bits_per_coded_weight": 8 * len(blob) / max(1, coded), "xz_sizes": xz_sizes,
             "code_entropy_bits": code_entropy(stacks),
             "delta_grad_bytes_max": max(s["delta_grad_bytes"] for s in stage_notes),
             "latent_bytes_max": max(s["latent_bytes"] for s in stage_notes),
             "delta_grad": max((s["delta_grad"] for s in stage_notes), key=lambda d: (d["bits"], d["latent_bits"])),
             "embedding": (f"rank-{a.embed_rank} adapter, {a.embed_bits}-bit factors (CWQUAN1 lora entry)" if ad else
                           "frozen at theta (the tied head is a frozen tail)"), "fresh": fresh,
             "rebuilds_bit_for_bit": True, "grad_copies": sum(st.copies for st in stacks),
             "train_seconds": round(t_train, 1),
             "process_seconds": round(time.perf_counter() - t_start, 1)}
    print(json.dumps({k: v for k, v in notes.items() if k not in ("args", "stages")}), flush=True)
    prepare.submit(Path(a.out), a.start, a.klass, blob, model, {"full": changed, "adapters": adapters}, steps,
                   changed=changed + (["embed"] if ad is not None else []), notes=notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
