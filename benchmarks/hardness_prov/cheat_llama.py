"""A cheater trained against both public gates on Llama-3.2-1B's q_proj and k_proj (HT-PROV v2), and the honest
fine-tune it is measured against.

Each q and k serves per-row absmax int8 codes. The cheater keeps the released model's row scale s and pins each row's
absmax entry at ±127; his other codes are q = round(A·Bᵀ + S), with A (n × r) and B (K × r) real and |S| ≤ a − ½
(projected after every step). So q − A·Bᵀ stays within ±a, Strassen depth 2 holds it in s8 at a ≤ 31, and his route
costs r/n + 3r/K (`routes.lowrank`) plus the residual's (`routes.residual`: its entries beyond the dense range, pins
included, corrected on the ALU at --writes, in writes of 32 W1 per entry and noise row). With --int-factors A and B
are int8 (straight-through) and the route is exact; a real A·Bᵀ is generous to him by up to half a code an entry. The
honest arm trains q and k as absmax codes (straight-through).
Everything else trains the same in both arms: forward KL from the frozen bf16 model on HT-USEFUL's FineWeb-Edu shard
(`hardness_useful/prep.py`, in --data), AdamW, every parameter.

The gates, with bands from the released model's codes per kind over its 16 layers:
- code law: the fractions of codes with |q| ≤ 15, > 31, > 63, = 127 and = 0, and tail31 = Σ(|q| − 31)₊²/Σq², each in
  [min, max];
- spectral: the share of ‖q‖² in the top r singular values at most the max, for r in --spec-ranks.
With --gates the cheater's loss adds squared hinges outside each band shrunk by --margin of its width: a count enters
at its exact value with a sigmoid's gradient, a top-r share as the energy in a tracked subspace (warm-started power
iterations every step, the subspace detached). With --train-dither σ the hinges are taken on the codes after a fresh
dither of σ (the cheater plans for the one-shot dither; his draws are independent of the measured one's).

Each arm is then measured as committed, after a one-shot dither (served = absmax(s·q + σ·s·Z), Z Gaussian, each σ in
--dither), and after a pass of --pass-steps distillation steps that adds σ_w·s·Z to every q and k latent after each
step (each σ_w in --pass-sigma; the pass's gradients are honest, or with --pass-steer the cheater's own objective plus a
pull of q − L back inside ±a; --pass-only-steer skips the honest one). --draws K repeats the dithers and the passes
with K independent noise draws from the same committed model (the pass's tokens are the same). Measured: perplexity (WikiText-2 test, FineWeb-Edu val), both gates exactly (singular
values in fp64), and the cheapest route the cheater knows through each matrix (his A·Bᵀ rescaled to the served scale,
or refit by alternating projection onto rank r plus ±a). The heuristic step of HT-PROV's margin table is measured by
its terms: with U = L's column space, top-r(q) ≥ ‖Uᵀq‖²/‖q‖² = (‖L‖² + 2⟨L, S⟩ + ‖UᵀS‖²)/‖q‖² exactly, and ‖L‖² ≥
Σ over unpinned entries of (|q| − a)₊² while |S| ≤ a; the step assumes ⟨L, S⟩ doesn't cancel ‖L‖².
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import routes  # noqa: E402
from prices import CHIPS  # noqa: E402

KINDS = ("q_proj", "k_proj")
LAW = ("le15", "gt31", "gt63", "at127", "zero", "tail31")
KD_CHUNK = 2048
EVAL_SEED = 1234
TRAIN_DITHER_SEED = 98765


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


class KD(torch.autograd.Function):
    """Mean over positions of KL(teacher ‖ student), its gradient formed chunk by chunk in the forward (HT-USEFUL's)."""

    @staticmethod
    def forward(ctx, sl, tl):
        shape = sl.shape
        s2, t2 = sl.reshape(-1, shape[-1]), tl.reshape(-1, shape[-1])
        n = s2.shape[0]
        grad = torch.empty_like(s2)
        loss = torch.zeros((), device=sl.device, dtype=torch.float64)
        for i in range(0, n, KD_CHUNK):
            ls = F.log_softmax(s2[i:i + KD_CHUNK].float(), -1)
            lt = F.log_softmax(t2[i:i + KD_CHUNK].float(), -1)
            pt = lt.exp()
            loss += (pt * (lt - ls)).sum(dtype=torch.float64)
            grad[i:i + KD_CHUNK] = ((ls.exp() - pt) / n).to(grad.dtype)
        ctx.save_for_backward(grad)
        ctx.shape = shape
        return (loss / n).float()

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return (grad * g.to(grad.dtype)).reshape(ctx.shape), None


def ste_round(u):
    return u + (torch.round(u) - u).detach()


def absmax_scale(w):
    return (w.detach().float().abs().amax(1) / 127).clamp_min(1e-12)


def dithered(q, sigma, gen):
    """The codes served after a dither of sigma LSB: absmax of q + sigma·Z, straight-through, the new scale held."""
    u = q + sigma * torch.randn(q.shape, device=q.device, generator=gen)
    return ste_round((u * (127 / u.detach().abs().amax(1).clamp_min(1e-12))[:, None]).clamp(-127, 127))


def served_linear(x, q, s):
    y = F.linear(x.to(torch.bfloat16), q.to(torch.bfloat16)).float() * s
    return y.to(torch.bfloat16 if torch.is_autocast_enabled() else x.dtype)


class HonestQ(nn.Module):
    """Absmax int8 codes of a latent fp32 weight, straight-through inside the clip."""

    def __init__(self, w: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(w.detach().float().clone())

    def codes(self):
        s = absmax_scale(self.weight)
        return ste_round((self.weight / s[:, None]).clamp(-127, 127)), s

    def forward(self, x):
        q, s = self.codes()
        return served_linear(x, q, s)


class CheatQ(nn.Module):
    """q = round(A·Bᵀ + S) at a fixed row scale, each row's pin at ±127, |S| ≤ a − ½ by projection."""

    def __init__(self, w: torch.Tensor, r: int, a: int, iters: int, int_factors: bool = False):
        super().__init__()
        self.int_factors = int_factors
        w = w.detach().float()
        s = absmax_scale(w)
        u = w / s[:, None]
        pin = u.abs().argmax(1)
        self.register_buffer("s", s)
        self.register_buffer("pin", pin)
        self.register_buffer("pin_val", 127 * torch.sign(u.gather(1, pin[:, None]))[:, 0])
        self.r, self.a = r, a
        bound = a - 0.5
        S = torch.zeros_like(u)
        for _ in range(iters):
            U, sv, V = torch.svd_lowrank(u - S, q=min(r + 16, min(u.shape)), niter=4)
            L = (U[:, :r] * sv[:r]) @ V[:, :r].T
            S = (u - L).clamp(-bound, bound)
        self.A = nn.Parameter((U[:, :r] * sv[:r].sqrt()).contiguous())
        self.B = nn.Parameter((V[:, :r] * sv[:r].sqrt()).contiguous())
        if int_factors:
            with torch.no_grad():
                S = (u - self.lowrank()).clamp(-bound, bound)
        self.S = nn.Parameter(S.contiguous())

    def lowrank(self):
        if self.int_factors:
            return ste_round(self.A.clamp(-127, 127)) @ ste_round(self.B.clamp(-127, 127)).T
        return self.A @ self.B.T

    def codes(self):
        u = (self.lowrank() + self.S).scatter(1, self.pin[:, None], self.pin_val[:, None])
        return ste_round(u.clamp(-127, 127)), self.s

    @torch.no_grad()
    def project(self):
        self.S.clamp_(-(self.a - 0.5), self.a - 0.5)

    def forward(self, x):
        q, s = self.codes()
        return served_linear(x, q, s)


def qk_modules(model):
    for i, layer in enumerate(model.model.layers):
        for k in KINDS:
            yield f"layers.{i}.{k}", k, layer.self_attn, k


def law_stats(q, tau=None):
    """Code-law statistics of integer-valued codes; with tau, counts carry a sigmoid's gradient at their exact value."""
    a = q.abs()

    def cnt(m):
        hard = (m > 0).float().mean()
        if tau is None:
            return hard
        soft = torch.sigmoid(m / tau).mean()
        return hard + (soft - soft.detach())

    return {"le15": cnt(15.5 - a), "gt31": cnt(a - 31.5), "gt63": cnt(a - 63.5), "at127": cnt(a - 126.5),
            "zero": cnt(0.5 - a), "tail31": (a - 31).clamp_min(0).pow(2).sum() / q.pow(2).sum()}


@torch.no_grad()
def energy_exact(q):
    sv = torch.linalg.svdvals(q.double())
    e = sv.pow(2)
    return (torch.cumsum(e, 0) / e.sum()).float().cpu().numpy()


class Tracker:
    """The top-k left singular subspace of a matrix that moves slowly, by warm-started power iterations."""

    def __init__(self, n, k, device, seed):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        self.Q = torch.linalg.qr(torch.randn(n, k, device=device, generator=g)).Q
        self.U = self.Q

    @torch.no_grad()
    def update(self, q, iters):
        Q = self.Q
        for _ in range(iters):
            Q = torch.linalg.qr(q @ (q.T @ Q)).Q
        self.Q = Q
        _, vec = torch.linalg.eigh((Q.T @ q) @ (Q.T @ q).T)
        self.U = Q @ vec.flip(-1)

    def shares(self, q, ranks):
        c = torch.cumsum((self.U.T @ q).pow(2).sum(1), 0) / q.pow(2).sum()
        return {r: c[r - 1] for r in ranks}


def bands_of(model, spec_ranks):
    """Per kind: [min, max] of each code-law statistic and the max top-r share over the layers (absmax codes)."""
    per = {k: [] for k in KINDS}
    for name, kind, parent, attr in qk_modules(model):
        w = getattr(parent, attr).weight.detach().float()
        q = torch.round(w / absmax_scale(w)[:, None]).clamp(-127, 127)
        st = {k: float(v) for k, v in law_stats(q).items()}
        e = energy_exact(q)
        st.update({f"top{r}": float(e[r - 1]) for r in spec_ranks})
        per[kind].append(st)
    out = {}
    for kind, xs in per.items():
        out[kind] = {k: [min(x[k] for x in xs), max(x[k] for x in xs)] for k in LAW}
        out[kind].update({f"top{r}": max(x[f"top{r}"] for x in xs) for r in spec_ranks})
    return out, per


def gate_penalty(q, kind, bands, margin, tracker, spec_ranks, tau):
    """Squared hinges outside each band shrunk by margin of its width (law) and above (1 − margin)·max (spectrum)."""
    st = law_stats(q, tau)
    pen = q.new_zeros(())
    worst = {}
    for k in LAW:
        lo, hi = bands[kind][k]
        w = max(hi - lo, 1e-3 * max(abs(hi), 1e-6))
        lo2, hi2 = lo + margin * w, hi - margin * w
        d = (F.relu(lo2 - st[k]) + F.relu(st[k] - hi2)) / w
        pen = pen + d.pow(2)
        worst[k] = float(d)
    sh = tracker.shares(q, spec_ranks)
    for r in spec_ranks:
        cap = (1 - margin) * bands[kind][f"top{r}"]
        d = F.relu(sh[r] - cap) / 0.01
        pen = pen + d.pow(2)
        worst[f"top{r}"] = float(d)
    return pen, worst


def gates_pass(q, kind, bands, spec_ranks, e=None):
    st = {k: float(v) for k, v in law_stats(q).items()}
    e = energy_exact(q) if e is None else e
    law_ok = all(bands[kind][k][0] <= st[k] <= bands[kind][k][1] for k in LAW)
    spec = {r: float(e[r - 1]) for r in spec_ranks}
    spec_ok = all(spec[r] <= bands[kind][f"top{r}"] for r in spec_ranks)
    return st, spec, law_ok, spec_ok


def route_costs(q_np, L_np, r, writes):
    """The route rank-r L plus the residual rint(q − L), on both chips at each correction price in writes."""
    n, k = q_np.shape
    S = np.rint(q_np - L_np).astype(np.int64)
    a = np.abs(S)
    out = {"beyond15": float((a > 15).mean()), "beyond31": float((a > 31).mean()), "beyond63": float((a > 63).mean()),
           "lowrank": routes.lowrank(r, n, k)}
    saved = {c: CHIPS[c]["alu_mac"] for c in CHIPS}
    try:
        for w in writes:
            for chip in CHIPS:
                CHIPS[chip]["alu_mac"] = 32.0 * w
                c, arg = routes.residual(S, chip)
                out[f"{chip}@{w:g}"] = round(routes.lowrank(r, n, k) + c, 5)
                out[f"{chip}@{w:g}_a"] = arg
    finally:
        for c, v in saved.items():
            CHIPS[c]["alu_mac"] = v
    return out


@torch.no_grad()
def refit(q, L, r, a, iters):
    """Alternating projection from L: L ← rank-r SVD of q − S, S ← clip(q − L, ±a)."""
    for _ in range(iters):
        S = (q - L).clamp(-a, a)
        U, sv, V = torch.svd_lowrank(q - S, q=min(r + 16, min(q.shape)), niter=4)
        L = (U[:, :r] * sv[:r]) @ V[:, :r].T
    return L


@torch.no_grad()
def heuristic(q, L, a, r):
    """The heuristic step's terms for q = L + S: ⟨L, S⟩ against ‖L‖², and S's energy in L's column space."""
    S = q - L
    U = torch.linalg.qr(L @ torch.randn(L.shape[1], r, device=L.device, dtype=L.dtype)).Q
    nq2, nL2, ip = float(q.pow(2).sum()), float(L.pow(2).sum()), float((L * S).sum())
    pS2, S2 = float((U.T @ S).pow(2).sum()), float(S.pow(2).sum())
    need = float((q.abs() - a).clamp_min(0).pow(2).sum())
    return {"L_share": nL2 / nq2, "ip_over_L2": ip / nL2 if nL2 else 0.0, "PS_of_S": pS2 / S2 if S2 else 0.0,
            "PLq_share": (nL2 + 2 * ip + pS2) / nq2, "tail_a_share": need / nq2}


def lowrank_of(mod, r):
    return mod.lowrank().detach().float() if isinstance(mod, CheatQ) else None


@torch.no_grad()
def measure(model, plan, bands, args, spec_ranks, scale_ref=None):
    """Both gates exactly and the cheater's routes for every q and k as currently served.

    plan: name -> (kind, r, L in codes at the committed scale or None, committed scale)."""
    mats = []
    for name, kind, parent, attr in qk_modules(model):
        mod = getattr(parent, attr)
        q, s = mod.codes()
        q = q.detach().float()
        r, L0, s0 = plan[name]["r"], plan[name]["L"], plan[name]["s"]
        e = energy_exact(q)
        st, spec, law_ok, spec_ok = gates_pass(q, kind, bands, spec_ranks, e)
        rec = {"name": name, "kind": kind, "r": r, "law": st, "spec": spec, "law_ok": law_ok, "spec_ok": spec_ok,
               "top_r": float(e[r - 1])}
        cands = {}
        if L0 is not None:
            cands["own"] = L0 * (s0 / s)[:, None]
            cands["own_fixed"] = L0
        base = cands.get("own")
        if base is None:
            U, sv, V = torch.svd_lowrank(q, q=min(r + 16, min(q.shape)), niter=6)
            base = (U[:, :r] * sv[:r]) @ V[:, :r].T
            cands["svd"] = base
        cands["refit"] = refit(q, base, r, args.a, args.refit_iters)
        best = None
        q_np = q.cpu().numpy().astype(np.int64)
        for cname, L in cands.items():
            rc = route_costs(q_np, L.double().cpu().numpy(), r, args.writes)
            rc["heuristic"] = heuristic(q, L, args.a, r)
            rec[f"route_{cname}"] = rc
            key = f"sm120@{args.writes[-1]:g}"
            if best is None or rc[key] < rec[f"route_{best}"][key]:
                best = cname
        rec["route_best"] = best
        mats.append(rec)
    return mats


def summarize(mats, writes, line=0.95):
    out = {}
    for kind in KINDS:
        xs = [m for m in mats if m["kind"] == kind]
        d = {"n": len(xs), "law_ok": sum(m["law_ok"] for m in xs), "spec_ok": sum(m["spec_ok"] for m in xs),
             "both_ok": sum(m["law_ok"] and m["spec_ok"] for m in xs)}
        for w in writes:
            for chip in CHIPS:
                key = f"{chip}@{w:g}"
                cost = [m[f"route_{m['route_best']}"][key] for m in xs]
                d[f"cost_max_{key}"] = max(cost)
                d[f"cost_mean_{key}"] = float(np.mean(cost))
                d[f"cheap_{key}"] = sum(c < line for c in cost)
                d[f"cheap_both_{key}"] = sum(c < line and m["law_ok"] and m["spec_ok"] for c, m in zip(cost, xs))
        d["beyond31_mean"] = float(np.mean([m[f"route_{m['route_best']}"]["beyond31"] for m in xs]))
        out[kind] = d
    return out


def wikitext_ids(tok):
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return np.asarray(tok("\n\n".join(ds["text"]))["input_ids"], dtype=np.int64)


@torch.no_grad()
def perplexity(model, ids, seq, max_windows=None, bs=4):
    n = len(ids) // seq
    if max_windows:
        n = min(n, max_windows)
    was = model.training
    model.eval()
    nll, cnt = 0.0, 0
    for i in range(0, n, bs):
        x = torch.from_numpy(np.asarray(ids[i * seq:min(i + bs, n) * seq]).astype(np.int64).reshape(-1, seq)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).logits
        nll += F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1),
                               reduction="sum").item()
        cnt += x[:, 1:].numel()
        del logits
    model.train(was)
    return {"ppl": math.exp(nll / cnt), "nll": nll / cnt, "tokens": cnt}


def ppls(model, wt_ids, val_ids, args):
    return {"wikitext2": perplexity(model, wt_ids, args.eval_seq, args.wt_windows),
            "fineweb_val": perplexity(model, val_ids, args.eval_seq, args.val_windows)}


def batch_at(train_ids, start, micro, seq):
    a = np.asarray(train_ids[start:start + micro * seq]).astype(np.int64)
    return torch.from_numpy(a.reshape(micro, seq)).cuda(non_blocking=True)


def to_honest(model):
    """Replace every q and k by HonestQ on its served codes' dequantized weight (the committed W)."""
    for name, kind, parent, attr in qk_modules(model):
        mod = getattr(parent, attr)
        q, s = mod.codes()
        setattr(parent, attr, HonestQ((q.detach().float() * s[:, None]).contiguous()))


def train(model, teacher, train_ids, args, steps, start_tok, lr, groups_extra, after_step, penalty, tag, val_ids,
          noclip, warmup=0.03, cosine=True):
    """Distillation steps; groups_extra are extra AdamW groups (with their own lr); after_step(step) runs after each
    update; penalty(step) returns (a loss added once a step, its log) or None. Gradients are clipped at norm 1 except
    the parameters in noclip (q and k, whose penalty gradients would scale everyone else's)."""
    extra_ids = {id(p) for g in groups_extra for p in g["params"]}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in extra_ids]
    clipped = [p for p in base if id(p) not in noclip]
    groups = [{"params": base, "lr": lr, "base_lr": lr, "eps": 1e-8}] + groups_extra
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    accum = args.batch // args.micro
    warm = max(1, int(warmup * steps))
    curve = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        f = min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * step / steps)) if cosine else 1.0)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f
        tot = 0.0
        for j in range(accum):
            x = batch_at(train_ids, start_tok + (step * args.batch + j * args.micro) * args.seq, args.micro, args.seq)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tl = teacher(x).logits
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sl = model(x).logits
            loss = KD.apply(sl, tl) / accum
            loss.backward()
            tot += loss.item()
            del tl, sl, loss
        rec = {"step": step + 1, "kd": tot}
        if penalty is not None:
            pen, info = penalty(step)
            pen.backward()
            rec.update(info)
        gn = torch.nn.utils.clip_grad_norm_(clipped, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        after_step(step)
        rec["grad_norm"] = gn
        rec["tok_s"] = (step + 1) * args.batch * args.seq / (time.time() - t0)
        if (step + 1) % args.eval_every == 0 or step + 1 == steps:
            rec["fineweb_val_32"] = perplexity(model, val_ids, args.eval_seq, 32)["ppl"]
        curve.append(rec)
        if (step + 1) % 10 == 0 or step < 3 or "fineweb_val_32" in rec:
            log(tag, json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
    return curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("honest", "cheat"), required=True)
    ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("HU_DATA", "data")))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rank-q", type=int, default=48)
    ap.add_argument("--rank-k", type=int, default=24)
    ap.add_argument("--a", type=int, default=31, help="the cheater's residual bound, codes")
    ap.add_argument("--init-iters", type=int, default=8, help="alternating projections fitting the start")
    ap.add_argument("--int-factors", action="store_true", help="A and B int8 (straight-through): an exact route")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--train-dither", type=float, default=0.0, help="the gates' hinges see the codes after a fresh "
                    "dither of this many LSB each step")
    ap.add_argument("--margin", type=float, default=0.15, help="the cheater aims inside each band by this share")
    ap.add_argument("--lam", type=float, default=100.0)
    ap.add_argument("--tau", type=float, default=1.0, help="the counts' sigmoid width, codes")
    ap.add_argument("--power-iters", type=int, default=2)
    ap.add_argument("--spec-ranks", default="8,32,64,128")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr-s", type=float, default=0.02, help="the cheater's S, codes a step")
    ap.add_argument("--lr-ab", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-seq", type=int, default=2048)
    ap.add_argument("--wt-windows", type=int, default=None)
    ap.add_argument("--val-windows", type=int, default=128)
    ap.add_argument("--writes", default="0.25,1,1.78,2.36", help="correction prices, writes; the last picks the "
                    "best route")
    ap.add_argument("--refit-iters", type=int, default=4)
    ap.add_argument("--dither", default="0.5,1,2,4")
    ap.add_argument("--pass-sigma", default="0.5")
    ap.add_argument("--pass-steps", type=int, default=200)
    ap.add_argument("--pass-lr", type=float, default=1e-5)
    ap.add_argument("--pass-steer", action="store_true")
    ap.add_argument("--pass-only-steer", action="store_true", help="with --pass-steer, skip the honest-gradient pass")
    ap.add_argument("--draws", type=int, default=1, help="independent dither and pass draws from the committed model; "
                    "draw j > 0 is keyed '#j'")
    args = ap.parse_args(argv)
    args.writes = [float(w) for w in args.writes.split(",")]
    spec_ranks = [int(r) for r in args.spec_ranks.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    train_ids = np.memmap(args.data / "train.bin", dtype=np.uint32, mode="r")
    val_ids = np.memmap(args.data / "val.bin", dtype=np.uint32, mode="r")
    wt_ids = wikitext_ids(tok)
    res = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    teacher = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").cuda().eval()
    teacher.config.use_cache = False
    for p in teacher.parameters():
        p.requires_grad_(False)

    bands, honest_per = bands_of(model, spec_ranks)
    res["bands"], res["released"] = bands, honest_per
    log("bands", json.dumps(bands))

    t0 = time.time()
    rank = {"q_proj": args.rank_q, "k_proj": args.rank_k}
    for name, kind, parent, attr in qk_modules(model):
        w = getattr(parent, attr).weight
        if args.arm == "cheat":
            setattr(parent, attr, CheatQ(w, rank[kind], args.a, args.init_iters, args.int_factors).cuda())
        else:
            setattr(parent, attr, HonestQ(w).cuda())
    res["init_s"] = time.time() - t0
    mods = {name: (kind, getattr(parent, attr)) for name, kind, parent, attr in qk_modules(model)}

    def plan_now():
        return {name: {"r": rank[kind], "L": lowrank_of(m, rank[kind]), "s": m.codes()[1].detach()}
                for name, (kind, m) in mods.items()}

    def state(tag, plan):
        t = time.time()
        ev = ppls(model, wt_ids, val_ids, args)
        mats = measure(model, plan, bands, args, spec_ranks)
        out = {"eval": ev, "summary": summarize(mats, args.writes), "matrices": mats, "seconds": time.time() - t}
        log(tag, json.dumps({"ppl": {k: round(v["ppl"], 4) for k, v in ev.items()}, "summary": out["summary"]}))
        return out

    res["start"] = state("start", plan_now())
    (args.out / "result.json").write_text(json.dumps(res, indent=1))

    trackers = {name: Tracker(m.codes()[0].shape[0], max(spec_ranks) + 16, "cuda", i)
                for i, (name, (kind, m)) in enumerate(mods.items())}

    dither_gen = torch.Generator(device="cuda")
    dither_gen.manual_seed(TRAIN_DITHER_SEED)

    def gates_loss(step, extra=None, gates=True, dither=0.0):
        """The cheater's penalty: the gates' hinges (with gates; on dithered codes with dither) plus extra(name, q),
        averaged over q and k, × lam."""
        pens, worst = [], {}
        for name, (kind, m) in mods.items():
            q, _ = m.codes()
            pen = q.new_zeros(())
            if gates:
                qg = dithered(q, dither, dither_gen) if dither else q
                tr = trackers[name]
                tr.update(qg.detach(), args.power_iters if step > 0 else 8)
                pen, w = gate_penalty(qg, kind, bands, args.margin, tr, spec_ranks, args.tau)
                for k, v in w.items():
                    worst[f"{kind[0]}_{k}"] = max(worst.get(f"{kind[0]}_{k}", 0.0), v)
            if extra is not None:
                e = extra(name, q)
                worst[f"{kind[0]}_pull"] = max(worst.get(f"{kind[0]}_pull", 0.0), float(e))
                pen = pen + e
            pens.append(pen)
        pen = args.lam * torch.stack(pens).mean()
        return pen, {"pen": float(pen), **{k: round(v, 4) for k, v in worst.items() if v > 0}}

    extra_groups = []
    if args.arm == "cheat":
        ab = [p for m in (m for _, m in mods.values()) for p in (m.A, m.B)]
        ss = [m.S for _, m in mods.values()]
        extra_groups = [{"params": ab, "lr": args.lr_ab, "base_lr": args.lr_ab, "eps": 1e-14},
                        {"params": ss, "lr": args.lr_s, "base_lr": args.lr_s, "eps": 1e-14}]

    def project(step):
        for _, m in mods.values():
            if isinstance(m, CheatQ):
                m.project()

    def qk_ids():
        return {id(p) for _, m in mods.values() for p in m.parameters()}

    penalty = ((lambda step: gates_loss(step, dither=args.train_dither)) if (args.arm == "cheat" and args.gates)
               else None)
    if args.steps > 0:
        res["train"] = {"curve": train(model, teacher, train_ids, args, args.steps, 0, args.lr, extra_groups, project,
                                       penalty, "train", val_ids, qk_ids())}
    committed_plan = plan_now()
    res["committed"] = state("committed", committed_plan)
    (args.out / "result.json").write_text(json.dumps(res, indent=1))

    to_honest(model)
    mods = {name: (kind, getattr(parent, attr)) for name, kind, parent, attr in qk_modules(model)}
    w0 = {name: m.weight.detach().clone() for name, (kind, m) in mods.items()}

    def draw_key(j):
        return f"#{j}" if j else ""

    res["dither"] = {}
    for j in range(args.draws):
        for sig in [float(x) for x in args.dither.split(",") if x]:
            g = torch.Generator(device="cuda")
            g.manual_seed(EVAL_SEED + 2 * j)
            with torch.no_grad():
                for name, (kind, m) in mods.items():
                    s0 = absmax_scale(w0[name])
                    m.weight.copy_(w0[name] + sig * s0[:, None] * torch.randn(w0[name].shape, device="cuda",
                                                                               generator=g))
            res["dither"][f"{sig:g}{draw_key(j)}"] = state(f"dither {sig:g}{draw_key(j)}", committed_plan)
            (args.out / "result.json").write_text(json.dumps(res, indent=1))
    with torch.no_grad():
        for name, (kind, m) in mods.items():
            m.weight.copy_(w0[name])

    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    res["pass"] = {}
    start_tok = args.steps * args.batch * args.seq
    steers = [False]
    if args.pass_steer and args.arm == "cheat":
        steers = [True] if args.pass_only_steer else [False, True]
    for j, sw, steer in [(j, float(x), st) for j in range(args.draws) for x in args.pass_sigma.split(",") if x
                         for st in steers]:
        model.load_state_dict(snapshot)
        g = torch.Generator(device="cuda")
        g.manual_seed(EVAL_SEED + 1 + 2 * j)

        def noise(step, sw=sw, g=g):
            with torch.no_grad():
                for _, (kind, m) in mods.items():
                    s = absmax_scale(m.weight)
                    m.weight.add_(sw * s[:, None] * torch.randn(m.weight.shape, device="cuda", generator=g))

        pen_fn = None
        if steer:
            def pull(name, q):
                p = committed_plan[name]
                L = p["L"] * (p["s"] / absmax_scale(mods[name][1].weight))[:, None]
                off_pin = torch.ones_like(q).scatter(1, q.detach().abs().argmax(1)[:, None], 0.0)
                return (((q - L).abs() - (args.a - 0.5)).clamp_min(0) * off_pin).pow(2).mean()

            def pen_fn(step, pull=pull):
                return gates_loss(step, pull, gates=args.gates)
        tag = f"pass sw{sw:g}{' steer' if steer else ''}{draw_key(j)}"
        curve = train(model, teacher, train_ids, args, args.pass_steps, start_tok, args.pass_lr, [], noise, pen_fn,
                      tag, val_ids, qk_ids(), warmup=0.0, cosine=False)
        out = state(tag, committed_plan)
        out["curve"] = curve
        res["pass"][f"{sw:g}{'_steer' if steer else ''}{draw_key(j)}"] = out
        (args.out / "result.json").write_text(json.dumps(res, indent=1))
    res["seconds"] = time.time() - t_start
    (args.out / "result.json").write_text(json.dumps(res, indent=1))
    log("done", f"{res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
