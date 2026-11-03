"""test_script.py's program: a two-matrix LM (embedding, one tanh layer, untied head) trained from the start ``toy``,
its update written one of several ways (``--update``), optionally with a buffer rewritten each step from this step's
tokens (``--cache``: from θ's ``frozen`` table or the trained embedding) that a θ-only side GEMM reads. Its batches are
`TrainData.next`'s, or (``--windows``) its own reads of the stream through numpy in `hooks.step()`'s steps, cut at the
stream's ctx or (``--own-ctx``) at prepare.EVAL_CTX. ``--warm-tokens N``: the first N / (batch x EVAL_CTX) steps are a
warmup that ends by halving ``fc``, a plan read off EVAL_CTX. ``--freeze NAME`` (repeated) leaves a matrix at θ;
``--micro N`` accumulates AdamW's gradient over N slices of the batch; ``--sampled-head`` scores only the head's rows
of the batch's distinct targets (a sampled softmax over ``unique(y)``); ``--attn`` adds one causal self-attention over
the embeddings (queries, keys and values the embeddings themselves) before the tanh layer. The MeZO variants read the estimate back as a
Python float and clip it with min/max (``mezo-clip``), branch on a loss comparison (``mezo-branch``), clip it as a tensor
(``mezo-clamp``), scale it by an EMA of its square kept as a Python float (``mezo-ema``) or a 0-dim tensor
(``mezo-ema-tensor``), or round it to fp16 through a tensor (``mezo-quant``). The coded variants pass each of the
``--micro`` partial gradients coded and sum them decoded: as quant's sign votes (``vote-packed``: `pack_signs` into int32
words, `unpack_votes` to int16 votes summed in a post-accumulate hook, islands/quant/train.py at 5eec10eb8a), the same
votes packed 8 a uint8 (``vote-bytes``), or as lowrank-restart's codes (``vote-planes``: 2-bit codes of an f16 absmax
scale per 8 entries, passed as torch.bool planes and the scales, its delta_grad.py `Code.pack` and `unpack` at
cf8a20f44f, rounded to the nearest code). ``--session`` plans the run and prints each step's `current_step()` and
`progress_bucket()`. ``--size-e`` sizes each step's E as sparse and shallow do, through `prepare._quant_array` on 4-bit
codes of the trained matrices, with no encode."""
from __future__ import annotations

import argparse
import contextlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import prepare
from te_core import hooks

LR, EPS, CLIP = 1e-3, 1e-3, 1e-4
MEZO = ["mezo", "mezo-clip", "mezo-branch", "mezo-clamp", "mezo-ema", "mezo-ema-tensor", "mezo-quant"]
CODED = ["vote-packed", "vote-bytes", "vote-planes"]
PLANES, GROUP = 2, 8


def pack_signs(g: torch.Tensor) -> torch.Tensor:
    b = (g.reshape(-1) >= 0).to(torch.int64)
    b = torch.cat([b, b.new_zeros(-b.numel() % 32)]).view(-1, 32)
    w = torch.tensor([1 << i for i in range(31)] + [-(1 << 31)], dtype=torch.int64, device=b.device)
    return (b * w).sum(1).to(torch.int32)


def unpack_votes(p: torch.Tensor, n: int) -> torch.Tensor:
    bits = (p[:, None] >> torch.arange(32, dtype=torch.int32, device=p.device)) & 1
    return 2 * bits.view(-1)[:n].to(torch.int16) - 1


def pack_bytes(g: torch.Tensor) -> torch.Tensor:
    b = (g.reshape(-1) >= 0).to(torch.int64)
    b = torch.cat([b, b.new_zeros(-b.numel() % 8)]).view(-1, 8)
    return (b * torch.tensor([1 << i for i in range(8)], dtype=torch.int64)).sum(1).to(torch.uint8)


def unpack_bytes(p: torch.Tensor, n: int) -> torch.Tensor:
    bits = (p[:, None].to(torch.int32) >> torch.arange(8, dtype=torch.int32)) & 1
    return 2 * bits.view(-1)[:n].to(torch.int16) - 1


def pack_planes(x: torch.Tensor) -> tuple:
    top = 2 ** PLANES - 2
    s = (x.reshape(-1, GROUP).abs().amax(1) * (1 + 2 ** -9)).to(torch.float16)
    div = torch.where(s > 0, s.float(), torch.ones_like(s, dtype=torch.float32))
    v = (x.reshape(-1, GROUP) / div[:, None]).clamp(-1, 1)
    codes = ((v + 1) * (top / 2)).round().long()
    return tuple(((codes >> b) & 1).bool() for b in range(PLANES)), s, tuple(x.shape)


def unpack_planes(payload: tuple) -> torch.Tensor:
    planes, s, shape = payload
    codes = sum(p.long() << b for b, p in enumerate(planes))
    top = 2 ** PLANES - 2
    return ((codes.float() * (2 / top) - 1) * s.float()[:, None]).reshape(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", choices=["inplace", "rebind", "adamw", "sign-ef", *MEZO, *CODED], required=True)
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--cache", choices=["none", "theta", "trained"], default="none")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--windows", action="store_true")
    ap.add_argument("--own-ctx", action="store_true")
    ap.add_argument("--warm-tokens", type=int, default=0)
    ap.add_argument("--freeze", action="append", default=[])
    ap.add_argument("--micro", type=int, default=1)
    ap.add_argument("--sampled-head", action="store_true")
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--size-e", action="store_true")
    a = ap.parse_args()
    warm = a.warm_tokens // (a.batch * prepare.EVAL_CTX)
    net, _, _ = prepare.load_start("toy")
    P = net.params()
    frozen = P.pop("frozen").requires_grad_(False)
    fixed = {n: P.pop(n).requires_grad_(False) for n in a.freeze}
    cache = torch.zeros(frozen.shape[1])
    err = {n: torch.zeros_like(p) for n, p in P.items()}
    opt = torch.optim.AdamW(P.values(), lr=LR) if a.update == "adamw" else None
    data = prepare.TrainData(None)
    ema, ema_float = torch.zeros(()), [0.0]
    votes = {n: torch.zeros(p.shape, dtype=torch.int16) for n, p in P.items()}
    if a.update == "vote-packed":
        def vote(p, n):
            votes[n] += unpack_votes(pack_signs(p.grad), p.numel()).view(p.shape)
            p.grad = None

        for n, p in P.items():
            p.register_post_accumulate_grad_hook(lambda p, n=n: vote(p, n))
    if a.session:
        hooks.plan(a.steps)

    def loss_of(P, x, y):
        P = {**fixed, **P}
        e = F.embedding(x, P["emb"])
        if a.attn:
            causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
            scores = (e @ e.transpose(1, 2) / math.sqrt(e.shape[-1])).masked_fill(~causal, -math.inf)
            e = e + scores.softmax(-1) @ e
        h = torch.tanh(F.linear(e, P["fc"]))
        if a.sampled_head:
            ids = torch.unique(y)
            logits = F.linear(h, F.embedding(ids, P["head"]))
            return F.cross_entropy(logits.flatten(0, 1), torch.searchsorted(ids, y).flatten())
        logits = F.linear(h, P["head"])
        if a.cache != "none":
            with torch.no_grad():
                cache.copy_(F.embedding(x, frozen if a.cache == "theta" else P["emb"]).mean((0, 1)))
            logits = logits + F.linear(F.embedding(x, frozen) * cache, frozen)
        return F.cross_entropy(logits.flatten(0, 1), y.flatten())

    def step(i, P):
        if a.windows:
            c = prepare.EVAL_CTX if a.own_ctx else data.ctx
            rows = torch.from_numpy(np.stack([data.tokens[j * c: j * c + c + 1]
                                              for j in range(i * a.batch, (i + 1) * a.batch)]).astype(np.int64))
        else:
            rows = data.next(a.batch)
        if a.session:
            s = hooks.session()
            print(json.dumps({"step": s.current_step(), "bucket": s.progress_bucket()}))
        x, y = rows[:, :-1], rows[:, 1:]
        if a.update in MEZO:
            with torch.no_grad():
                gen = torch.Generator().manual_seed(i)
                z = {n: torch.randn(p.shape, generator=gen) for n, p in P.items()}
                lp = loss_of({n: p + EPS * z[n] for n, p in P.items()}, x, y)
                lm = loss_of({n: p - EPS * z[n] for n, p in P.items()}, x, y)
                if a.update in ("mezo-clamp", "mezo-ema-tensor"):
                    c = (lp - lm) / (2 * EPS)
                    if a.update == "mezo-clamp":
                        c = torch.clamp(c, -CLIP / LR, CLIP / LR)
                    else:
                        ema.mul_(0.9).add_(0.1 * c * c)
                        c = c / (ema + 1).sqrt()
                else:
                    c = float(lp - lm) / (2 * EPS)
                    if a.update == "mezo-clip":
                        c = max(-CLIP / LR, min(CLIP / LR, c))
                    elif a.update == "mezo-branch" and lp < lm:
                        c = -c
                    elif a.update == "mezo-ema":
                        ema_float[0] = 0.9 * ema_float[0] + 0.1 * c * c
                        c = c / math.sqrt(ema_float[0] + 1)
                    elif a.update == "mezo-quant":
                        c = float(torch.tensor(c, dtype=torch.float16).float())
                for n, p in P.items():
                    p.sub_(z[n] * (LR * c))
            return P
        if a.update == "adamw":
            for j in range(a.micro):
                part = slice(j * a.batch // a.micro, (j + 1) * a.batch // a.micro)
                loss_of(P, x[part], y[part]).backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            return P
        if a.update in CODED:
            dec = {n: torch.zeros(p.shape) for n, p in P.items()}
            for j in range(a.micro):
                part = slice(j * a.batch // a.micro, (j + 1) * a.batch // a.micro)
                loss = loss_of(P, x[part], y[part])
                if a.update == "vote-packed":
                    loss.backward()
                    continue
                for n, g in zip(P, torch.autograd.grad(loss, list(P.values()))):
                    if a.update == "vote-bytes":
                        votes[n] += unpack_bytes(pack_bytes(g), g.numel()).view(g.shape)
                    else:
                        dec[n] += unpack_planes(pack_planes(g))
            with torch.no_grad():
                for n, p in P.items():
                    p.sub_(dec[n] if a.update == "vote-planes" else torch.sign(votes[n]), alpha=LR)
                    votes[n].zero_()
            return P
        loss = loss_of(P, x, y)
        grads = dict(zip(P, torch.autograd.grad(loss, list(P.values()))))
        if a.update == "inplace":
            with torch.no_grad():
                for n, p in P.items():
                    p.sub_(grads[n], alpha=LR)
            return P
        if a.update == "rebind":
            return {n: (p - LR * grads[n]).detach().requires_grad_() for n, p in P.items()}
        new = {}
        for n, p in P.items():
            acc = grads[n] + err[n]
            scale = acc.abs().mean()
            dec = torch.where(acc > 0, scale, -scale)
            err[n] = acc - dec
            new[n] = (p.detach() - LR * dec).requires_grad_()
        return new

    for i in range(a.steps):
        with hooks.step() if a.windows else contextlib.nullcontext():
            P = step(i, P)
            if a.size_e:
                with torch.no_grad():
                    for n, p in P.items():
                        codes = (p / (p.abs().amax() + 1e-12) * 7 + 8).round().clamp(0, 15)
                        prepare._quant_array(argparse.Namespace(bits=4, codes=codes, codebook=torch.arange(16.0)), n)
            if i + 1 == warm:
                with torch.no_grad():
                    P["fc"].mul_(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
