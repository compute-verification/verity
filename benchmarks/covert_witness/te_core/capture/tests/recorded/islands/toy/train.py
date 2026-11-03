"""test_script.py's program: a two-matrix LM (embedding, one tanh layer, untied head) trained from the start ``toy``,
its update written one of several ways (``--update``), optionally with a buffer rewritten each step from this step's
tokens (``--cache``: from θ's ``frozen`` table or the trained embedding) that a θ-only side GEMM reads. Its batches are
`TrainData.next`'s, or (``--windows``) its own reads of the stream through numpy in `hooks.step()`'s steps, cut at the
stream's ctx or (``--own-ctx``) at prepare.EVAL_CTX. ``--warm-tokens N``: the first N / (batch x EVAL_CTX) steps are a
warmup that ends by halving ``fc``, a plan read off EVAL_CTX. ``--freeze NAME`` (repeated) leaves a matrix at θ;
``--micro N`` accumulates AdamW's gradient over N slices of the batch; ``--sampled-head`` scores only the head's rows
of the batch's distinct targets (a sampled softmax over ``unique(y)``). The MeZO variants read the estimate back as a
Python float and clip it with min/max (``mezo-clip``), branch on a loss comparison (``mezo-branch``), clip it as a tensor
(``mezo-clamp``), scale it by an EMA of its square kept as a Python float (``mezo-ema``) or a 0-dim tensor
(``mezo-ema-tensor``), or round it to fp16 through a tensor (``mezo-quant``). ``--session`` plans the run and prints each step's `current_step()` and `progress_bucket()`."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", choices=["inplace", "rebind", "adamw", "sign-ef", *MEZO], required=True)
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
    if a.session:
        hooks.plan(a.steps)

    def loss_of(P, x, y):
        P = {**fixed, **P}
        h = torch.tanh(F.linear(F.embedding(x, P["emb"]), P["fc"]))
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
            if i + 1 == warm:
                with torch.no_grad():
                    P["fc"].mul_(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
