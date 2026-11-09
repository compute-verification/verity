"""Simulatability: is a fault's served output what an honest query on another input serves?

A fault is simulatable when an honest-use party, given a short advice string by the server (which knows the weights),
reproduces the faulted response with honest queries and no weights. The faulted view is then the honest view plus the
advice, and the theft bound charges the advice instead of the fault. This measures how short that advice can be for
in-range faults under clamp.py's boxes, with three simulators:

  silent  the fault changes no served token: x' = x and no advice
  forced  the honest model reading the faulted tokens as forced input (reach.py's direct tokens); the advice is the
          direct tokens' code (`advice_bits`)
  edit    one token of the prompt's last --window positions replaced, x' = x[tau <- u]; the advice is log2(W V) for the
          edit plus the code of the direct tokens of the honest model on x' reading the faulted tokens

The direct tokens are coded two ways: "tokens", each token's id, which is all a simulator served tokens alone can
decode, and "rank", its rank under the honest model's scores, which needs log-probabilities. A record's forced_bits,
edit_bits, edit_direct, edit and best_bits are the first code's, each with the best edit under it; the same names
ending in _rank are the second's. A fault is covered by its charge when its advice is at most c + --slot-bits: c its
block's output bits, and the location the charge pays beyond a served slot's (log2 of proof blocks a served token,
GLM-5.3's by default).

Faults sit at a prompt's last position t0 (the forward serving the first new token) or --offsets positions before it,
every forward clamped to the boxes. Kinds:
  uniform  one random coordinate at a value uniform in its box
  edge     one random coordinate at plus or minus its box
  own      a linear's input (lm.GROUP_SITES): one FP8 code at plus or minus 448 times its token's own group scale
  adv      of --adv-pool random coordinates at plus or minus their box, the --adv with the largest KL(faulted || honest)
           of the first served token's distribution
  far      of the same pool, the --adv farthest from every input of the window (the identity included) in centered
           logits at the first served token: the faults least like any honest query there

Every edit is evaluated at the first served token, exhaustively over its W V candidates: its served top-1 and top-5,
and its distance to the faulted distribution in centered logits, d = ||z_f - z_e - mean||_2, which bounds the
max-divergence (|log p_f(y) - log p_e(y)| <= 2 d for every token y). The --near nearest are evaluated exactly: KL and
the eps-max-divergence D_inf^eps(p_f || p_e), the least k with p_f(E) <= 2^k p_e(E) + eps for every event E. The --cont
nearest whose first token matches are decoded for --new tokens reading the faulted ones.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/simulate.py --model gpt2 \\
      --boxes boxes.pt --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, Fault, served_logprobs, ue8m0_scale
from prompts import PROMPTS

KINDS = ("uniform", "edge", "own", "adv", "far")
CODES = ("tokens", "rank")
LN2 = math.log(2)


def prefix(lm: LM, ids: torch.Tensor, upto: int) -> Cache:
    c = Cache()
    if upto:
        lm.forward(ids[:, :upto], c)
    return c


def kl_bits(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in bits, rows of log-probabilities (lp broadcasts against lq)."""
    p = lp.exp()
    return torch.where(p > 0, p * (lp - lq), torch.zeros_like(p)).sum(-1).clamp_min(0) / LN2


def maxdiv_bits(lp: torch.Tensor, lq: torch.Tensor, eps: float, floor: float = 1e-12) -> torch.Tensor:
    """D_inf^eps(p || q) in bits: the least k >= 0 with sum_y max(0, p(y) - 2^k q(y)) <= eps, by bisection over the
    tokens with p(y) >= floor (the rest carry under V floor of p's mass)."""
    keep = (lp.exp() >= floor) if lp.dim() == 1 else (lp.exp() >= floor).any(0)
    p = lp.exp()[..., keep].double()
    q = lq[..., keep].exp().double()
    p = p.expand_as(q)
    lo = torch.zeros(q.shape[0], dtype=torch.float64)
    hi = torch.full((q.shape[0],), 64.0, dtype=torch.float64)
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = (p - torch.exp2(mid)[:, None] * q).clamp_min(0).sum(-1) <= eps
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi.float()


class Edits:
    """Every one-token edit of the prompt's last W positions, at the first served token: the final norm's output
    there (N), its served top-5 (top), and N in the centered-logit metric (M, so that ||M_u - M_f||_2 is the distance
    of their centered logits)."""

    def __init__(self, lm: LM, ids: torch.Tensor, W: int, batch: int, precision: str, P: torch.Tensor):
        V, L = lm.spec.vocab, lm.spec.layers
        t0 = ids.shape[1] - 1
        self.taus = list(range(t0 - W + 1, t0 + 1))
        self.V, self.W = V, W
        ns, tops = [], []
        for tau in self.taus:
            pre = prefix(lm, ids, tau)
            tail = ids[:, tau + 1:]
            for u0 in range(0, V, batch):
                u = torch.arange(u0, min(V, u0 + batch))
                b = len(u)
                taps = {("final", L): None}
                lg = lm.forward(torch.cat([u[:, None], tail.expand(b, -1)], 1), pre.expand(b), taps=taps,
                                last_only=True)[:, -1]
                ns.append(taps[("final", L)][:, -1].float())
                tops.append(served_logprobs(lg, precision).topk(5, -1).indices)
        self.N = torch.cat(ns)
        self.top = torch.cat(tops)
        self.M = self.N @ P
        self.M2 = self.M.pow(2).sum(1)
        self.tau = torch.tensor(self.taus).repeat_interleave(V)
        self.u = torch.arange(V).repeat(W)
        self.identity = self.u == ids[0, self.tau]


def centered_metric(lm: LM) -> torch.Tensor:
    """P with P P^T = Uc^T Uc, Uc the unembedding centered over the vocabulary: ||(n - n') P||_2 is the distance of the
    centered logits of final-norm outputs n and n'."""
    U = lm.E.double()
    Uc = U - U.mean(0, keepdim=True)
    ev, Q = torch.linalg.eigh(Uc.T @ Uc)
    return (Q * ev.clamp_min(0).sqrt()).float()


def honest_taps(lm: LM, ids: torch.Tensor) -> dict:
    taps = {(s, l): None for s, l in lm.sites()}
    lm.forward(ids, Cache(), taps=taps)
    return taps


def site_list(lm: LM, names: list[str], layers: list[int], offset: int) -> list[tuple[str, int]]:
    L = lm.spec.layers
    out = []
    for s in names:
        if s == "emb":
            out.append(("emb", -1))
        elif s == "final":
            if offset == 0:
                out.append(("final", L))
        else:
            out += [(s, l if l >= 0 else L + l) for l in layers]
    return out


def step1(lm: LM, ids: torch.Tensor, tf: int, pre: Cache, site: tuple[str, int], cands: list[tuple[int, float]],
          taps: dict, precision: str, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The first served token's log-probabilities, and the final norm's output there, with each candidate
    (coord, value) at site, position tf; the layers below the site's are read from the honest residual stream."""
    out, ns = [], []
    s, l = site
    L = lm.spec.layers
    tail = ids[:, tf:]
    for i in range(0, len(cands), batch):
        ch = cands[i:i + batch]
        b = len(ch)
        f = Fault.of(tf, [(s, l, c, v) for c, v in ch])
        tp = {("final", L): None}
        if s == "emb":
            lg = lm.forward(tail.expand(b, -1), pre.expand(b), fault=f, taps=tp, last_only=True)
        else:
            first = min(max(l, 0), L)
            xin = (taps[("emb", -1)] if first == 0 else taps[("resid", first - 1)])[:, tf:]
            lg = lm.forward(tail.expand(b, -1), pre.expand(b), fault=f, taps=tp, last_only=True,
                            x_in=xin.expand(b, -1, -1), first_layer=first, pos0=tf)
        out.append(served_logprobs(lg[:, -1], precision))
        ns.append(tp[("final", L)][:, -1].float())
    return torch.cat(out), torch.cat(ns)


def advice_bits(D: np.ndarray, rank: np.ndarray, vocab: int, code: str) -> np.ndarray:
    """Per row, a prefix code for the direct tokens a simulator must be told: Elias gamma of D + 1, log2 C(N, D) for
    their positions, and per direct token either its id in log2 V bits ("tokens", all a simulator reading served
    tokens alone can decode) or a flag and either Elias gamma of its rank - 1 or log2 V ("rank", which needs the
    honest model's scores, so log-probabilities from the API)."""
    n = D.shape[1]
    d = D.sum(1)
    lgam = np.vectorize(math.lgamma)
    positions = (lgam(n + 1) - lgam(d + 1) - lgam(n - d + 1)) / math.log(2)
    gamma = lambda x: 2 * np.floor(np.log2(np.maximum(x, 1))) + 1  # noqa: E731
    if code == "tokens":
        tokens = d * math.log2(vocab)
    else:
        tokens = np.where(D, 1 + np.minimum(gamma(rank - 1), math.log2(vocab)), 0).sum(1)
    return gamma(d + 1) + positions + tokens


def nearest_edit(E: "Edits", n: torch.Tensor, P: torch.Tensor, chunk: int = 32) -> torch.Tensor:
    """Per row of n (B, d), the least centered-logit distance to any edit, the identity included: how far the first
    served token's distribution is from every input of the window."""
    m = n @ P
    out = []
    for i in range(0, len(m), chunk):
        mi = m[i:i + chunk]
        d2 = E.M2[None] - 2 * mi @ E.M.T + mi.pow(2).sum(1, keepdim=True)
        out.append(d2.min(1).values.clamp_min(0).sqrt())
    return torch.cat(out)


def decode(lm: LM, ids: torch.Tensor, tf: int, faults: list[tuple] | None, new: int, precision: str):
    """Greedy served tokens (B, new) with fault b at position tf (None: one honest row), the first served token's
    log-probabilities (B, V) and the final norm's output there (B, d)."""
    L = lm.spec.layers
    b = len(faults) if faults else 1
    c = prefix(lm, ids, tf).expand(b)
    taps = {("final", L): None}
    f = Fault.of(tf, faults) if faults else None
    lg = lm.forward(ids[:, tf:].expand(b, -1), c, fault=f, taps=taps, last_only=True)[:, -1]
    lp1 = served_logprobs(lg, precision)
    n1 = taps[("final", L)][:, -1].float()
    ys = [lp1.argmax(-1)]
    for _ in range(1, new):
        ys.append(served_logprobs(lm.forward(ys[-1][:, None], c)[:, -1], precision).argmax(-1))
    return torch.stack(ys, 1), lp1, n1


def decode_forced(lm: LM, ids: torch.Tensor, rows: list[tuple[int, int]], ys: torch.Tensor, precision: str,
                  batch: int):
    """For each row (tau, u) (tau -1: no edit), the honest model on x' = x[tau <- u] reading the served tokens ys[row]
    (R, new): per step whether its greedy token differs (direct) and the served token's rank among its scores."""
    R, new = ys.shape
    D = torch.zeros(R, new, dtype=torch.bool)
    rank = torch.ones(R, new, dtype=torch.long)
    t0 = ids.shape[1] - 1
    by: dict[int, list[int]] = {}
    for r, (tau, _) in enumerate(rows):
        by.setdefault(t0 if tau < 0 else tau, []).append(r)
    for tau, rs in by.items():
        pre = prefix(lm, ids, tau)
        for i in range(0, len(rs), batch):
            ch = torch.tensor(rs[i:i + batch])
            b = len(ch)
            x = ids[:, tau:].expand(b, -1).clone()
            x[:, 0] = torch.tensor([rows[r][1] if rows[r][0] >= 0 else int(ids[0, tau]) for r in ch.tolist()])
            c = pre.expand(b)
            lp = served_logprobs(lm.forward(x, c, last_only=True)[:, -1], precision)
            for n in range(new):
                y = ys[ch, n]
                D[ch, n] = lp.argmax(-1) != y
                rank[ch, n] = 1 + (lp > lp.gather(1, y[:, None])).sum(-1)
                if n + 1 < new:
                    lp = served_logprobs(lm.forward(y[:, None], c)[:, -1], precision)
    return D, rank


def faults_for(lm: LM, ids: torch.Tensor, tf: int, site: tuple[str, int], taps: dict, pre: Cache, lp_h: torch.Tensor,
               kinds: list[str], a, gen: torch.Generator, E: "Edits", P: torch.Tensor) -> list[dict]:
    """The site's faults at position tf, each {site, layer, coord, value, kind, hi, honest, c}: c is the faulting
    block's output bits, 16 for one value, and 1,032 for a value beyond one FP8 code at its token's own group scale
    at a linear's input (it rewrites the group's scale, so its block writes the whole group)."""
    pc = lm.boxes.class_at(tf, ids[:, tf:tf + 1], pre.sink)
    x = taps[site][0, tf].float()
    hi = lm.boxes.hi[site][pc].float()
    w = x.numel()
    out = []

    def own_of(c: int) -> float:
        g = x[c // GROUP * GROUP:(c // GROUP + 1) * GROUP]
        return min(E4M3_MAX * float(ue8m0_scale(g.abs().max())), float(hi[c]))

    def add(c: int, v: float, kind: str, score: float | None = None, far: float | None = None):
        wide = site[0] in GROUP_SITES and abs(v) > own_of(c) * (1 + 1e-6)
        out.append({"site": site[0], "layer": site[1], "coord": int(c), "value": float(v), "kind": kind,
                    "hi": float(hi[c]), "honest": float(x[c]), "class": pc, "screen_kl": score, "screen_far": far,
                    "c": 8 * GROUP + 8 if wide else 16})

    for kind in kinds:
        if kind == "uniform":
            for c in torch.randperm(w, generator=gen)[:a.random].tolist():
                add(c, float((torch.rand(1, generator=gen) * 2 - 1) * hi[c]), kind)
        elif kind == "edge":
            for c in torch.randperm(w, generator=gen)[:a.random].tolist():
                add(c, float(hi[c]) * (1.0 if torch.rand(1, generator=gen) < 0.5 else -1.0), kind)
        elif kind == "own" and site[0] in GROUP_SITES:
            for c in torch.randperm(w, generator=gen)[:a.random].tolist():
                add(c, own_of(c) * (1.0 if torch.rand(1, generator=gen) < 0.5 else -1.0), kind)
    if {"adv", "far"} & set(kinds):
        pool = torch.randperm(w, generator=gen)[:a.adv_pool].tolist()
        cands = [(c, sg * float(hi[c])) for c in pool for sg in (1.0, -1.0)]
        lp, n = step1(lm, ids, tf, pre, site, cands, taps, a.precision, a.batch)
        kl = kl_bits(lp, lp_h)
        far = nearest_edit(E, n, P)
        if "adv" in kinds:
            for j in kl.topk(min(a.adv, len(cands))).indices.tolist():
                add(cands[j][0], cands[j][1], "adv", float(kl[j]), float(far[j]))
        if "far" in kinds:
            for j in far.topk(min(a.adv, len(cands))).indices.tolist():
                add(cands[j][0], cands[j][1], "far", float(kl[j]), float(far[j]))
    return out


def summarize(recs: list[dict]) -> list[dict]:
    keys = sorted({(r["site"], r["layer"], r["offset"], r["kind"]) for r in recs},
                  key=lambda k: (k[2], k[1], k[0], k[3]))
    rows = []
    for key in keys:
        g = [r for r in recs if (r["site"], r["layer"], r["offset"], r["kind"]) == key]
        eff = [r for r in g if not r["silent"]]
        moved = [r for r in g if r["dinf_honest"] > 1.0]
        fr = lambda xs, f: (sum(1 for r in xs if f(r)) / len(xs)) if xs else None  # noqa: E731
        rows.append({
            "site": key[0], "layer": key[1], "offset": key[2], "kind": key[3], "n": len(g),
            "silent": fr(g, lambda r: r["silent"]), "flip": fr(g, lambda r: r["flip"]),
            "kl_honest_mean": float(np.mean([r["kl_honest"] for r in g])),
            "dinf_honest_mean": float(np.mean([r["dinf_honest"] for r in g])),
            "n_effective": len(eff), "n_moved": len(moved),
            "eff_top1": fr(eff, lambda r: r["n_top1"] > 0), "eff_top5": fr(eff, lambda r: r["n_top5"] > 0),
            "eff_top5set": fr(eff, lambda r: r["n_top5set"] > 0),
            "moved_dinf1": fr(moved, lambda r: r["best_dinf"] <= 1.0),
            "moved_dinf_beats_honest": fr(moved, lambda r: r["best_dinf"] < r["dinf_honest"]),
            "eff_cont": fr(eff, lambda r: r["cont_exact"]),
            "eff_edit_cheaper": fr(eff, lambda r: r["edit_bits"] < r["forced_bits"]),
            "eff_budget": fr(eff, lambda r: r["best_bits"] <= r["budget"]),
            "eff_budget_rank": fr(eff, lambda r: r["best_bits_rank"] <= r["budget"]),
            "eff_forced_budget": fr(eff, lambda r: r["forced_bits"] <= r["budget"]),
            "eff_budget16": fr(eff, lambda r: r["best_bits"] <= r["budget"] - r["c"] + 16),
            "c_wide": fr(g, lambda r: r["c"] > 16),
            "best_bits_mean": float(np.mean([r["best_bits"] for r in g])),
            "best_bits_max": float(np.max([r["best_bits"] for r in g])),
            "forced_bits_mean": float(np.mean([r["forced_bits"] for r in g])),
            "forced_bits_max": float(np.max([r["forced_bits"] for r in g])),
            "raw_mean": float(np.mean([r["raw"] for r in g])), "direct_mean": float(np.mean([r["direct"] for r in g])),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--boxes", type=Path, required=True, help="clamp.py's calibrated boxes.pt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", default="0:8", help="a slice of prompts.PROMPTS, start:stop")
    ap.add_argument("--sites", default="emb,resid,mlph,attnh,k,v,final")
    ap.add_argument("--far-sites", default="emb,resid,k,v", help="sites faulted at offsets > 0")
    ap.add_argument("--layers", default="0,2,5,8,11", help="comma-separated; negative from the end")
    ap.add_argument("--offsets", default="0,2", help="fault positions before the prompt's last")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--random", type=int, default=2, help="random coordinates a site for uniform, edge and own")
    ap.add_argument("--adv", type=int, default=2, help="adversarial faults a site")
    ap.add_argument("--adv-pool", type=int, default=256, help="coordinates the adversarial screen tries a site")
    ap.add_argument("--window", type=int, default=3, help="edits at the prompt's last W positions")
    ap.add_argument("--new", type=int, default=16, help="served tokens decoded a fault")
    ap.add_argument("--near", type=int, default=64, help="nearest edits evaluated exactly a fault")
    ap.add_argument("--cont", type=int, default=8, help="nearest first-token-matching edits decoded a fault")
    ap.add_argument("--eps", type=float, default=1e-3, help="the slack of the eps-max-divergence")
    ap.add_argument("--slot-bits", type=float, default=24.2,
                    help="log2 of proof blocks a served token: the location the charge pays beyond a served slot's "
                         "(GLM-5.3's, 1.9e7)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    V = lm.spec.vocab
    layers = [int(x) for x in a.layers.split(",")]
    kinds = a.kinds.split(",")
    offsets = [int(x) for x in a.offsets.split(",")]
    if max(offsets) >= a.window:
        ap.error("every fault offset must sit inside the edit window")
    P = centered_metric(lm)
    edit_bits = math.log2(a.window * V)
    p0, p1 = (int(x) for x in a.prompts.split(":"))
    recs: list[dict] = []
    t_start = time.time()
    rec_file = open(a.out / "records.jsonl", "w")
    for pi in range(p0, p1):
        text = PROMPTS[pi]
        gen = torch.Generator().manual_seed(a.seed * 1_000_003 + pi)
        ids = torch.tensor([tok.encode(text).ids])
        t0 = ids.shape[1] - 1
        taps = honest_taps(lm, ids)
        yh, lp_h, _ = decode(lm, ids, t0, None, a.new, a.precision)
        yh, lp_h = yh[0], lp_h[0]
        t_e = time.time()
        E = Edits(lm, ids, a.window, a.batch, a.precision, P)
        mism = int((E.top[E.identity, 0] != yh[0]).sum())
        print(f"prompt {pi}: {t0 + 1} tokens, edit table {time.time() - t_e:.0f} s, identity edits off the honest "
              f"first token: {mism}", flush=True)
        for off in offsets:
            tf = t0 - off
            pre = prefix(lm, ids, tf)
            faults = []
            for site in site_list(lm, (a.sites if off == 0 else a.far_sites).split(","), layers, off):
                faults += faults_for(lm, ids, tf, site, taps, pre, lp_h, kinds, a, gen, E, P)
            for i in range(0, len(faults), a.batch):
                ch = faults[i:i + a.batch]
                ys, lp1, n1 = decode(lm, ids, tf, [(f["site"], f["layer"], f["coord"], f["value"]) for f in ch],
                                     a.new, a.precision)
                D0, rank0 = decode_forced(lm, ids, [(-1, -1)] * len(ch), ys, a.precision, a.batch)
                forced = {code: advice_bits(D0.numpy(), rank0.numpy(), V, code) for code in CODES}
                cont_rows, cont_of = [], []
                per = []
                for j, f in enumerate(ch):
                    lp = lp1[j]
                    y1, top5 = int(lp.argmax()), lp.topk(5).indices
                    m1 = (E.top[:, 0] == y1) & ~E.identity
                    m5 = (E.top == top5).all(1) & ~E.identity
                    m5s = (E.top.sort(1).values == top5.sort().values).all(1) & ~E.identity
                    dist = (E.M - n1[j] @ P).pow(2).sum(1).sqrt().masked_fill(E.identity, float("inf"))
                    near = dist.topk(a.near, largest=False).indices
                    lpe = served_logprobs(E.N[near] @ lm.E.T, a.precision)
                    kl, dinf = kl_bits(lp, lpe), maxdiv_bits(lp, lpe, a.eps)
                    dh = maxdiv_bits(lp, lp_h[None], a.eps)[0]
                    pick = dist.masked_fill(~m1, float("inf")).topk(min(a.cont, int(m1.sum())), largest=False).indices \
                        if bool(m1.any()) else torch.tensor([], dtype=torch.long)
                    pick = torch.unique(torch.cat([pick, near[:2]]))
                    for k in pick.tolist():
                        cont_rows.append((int(E.tau[k]), int(E.u[k])))
                        cont_of.append(j)
                    jb = int(dinf.argmin())
                    per.append({"y1": y1, "n_top1": int(m1.sum()), "n_top5": int(m5.sum()), "n_top5set": int(m5s.sum()),
                                "best_l2": float(dist[near[0]]), "best_kl": float(kl.min()), "best_dinf": float(dinf[jb]),
                                "best_dinf_edit": [int(E.tau[near[jb]]), int(E.u[near[jb]])],
                                "kl_honest": float(kl_bits(lp, lp_h)), "dinf_honest": float(dh)})
                yc = ys[torch.tensor(cont_of, dtype=torch.long)] if cont_rows else torch.zeros(0, a.new, dtype=torch.long)
                De, ranke = decode_forced(lm, ids, cont_rows, yc, a.precision, a.batch) if cont_rows else (None, None)
                code_e = {code: advice_bits(De.numpy(), ranke.numpy(), V, code) + edit_bits if cont_rows
                          else np.zeros(0) for code in CODES}
                for j, f in enumerate(ch):
                    mine = [r for r, o in enumerate(cont_of) if o == j]
                    raw = int((ys[j] != yh).sum())
                    rec = {"model": a.model, "prompt": pi, "offset": off, "t0": t0, **f, **per[j],
                           "flip": bool(ys[j, 0] != yh[0]), "raw": raw, "silent": raw == 0, "direct": int(D0[j].sum()),
                           "direct_at": D0[j].nonzero().flatten().tolist(),
                           "cont_exact": bool(mine) and any(not bool(De[r].any()) for r in mine),
                           "served": tok.decode(ys[j].tolist()), "honest_served": tok.decode(yh.tolist()),
                           "budget": f["c"] + a.slot_bits}
                    for code in CODES:
                        sfx = "" if code == CODES[0] else "_" + code
                        best = min(mine, key=code_e[code].__getitem__) if mine else None
                        rec["forced_bits" + sfx] = float(forced[code][j])
                        rec["edit_bits" + sfx] = float(code_e[code][best]) if best is not None else float("inf")
                        rec["edit_direct" + sfx] = int(De[best].sum()) if best is not None else None
                        rec["edit" + sfx] = list(cont_rows[best]) if best is not None else None
                        rec["best_bits" + sfx] = 0.0 if raw == 0 else min(rec["forced_bits" + sfx],
                                                                           rec["edit_bits" + sfx])
                    e = rec["edit"]
                    rec["edit_text"] = None if e is None else [tok.decode([int(ids[0, e[0]])]), tok.decode([e[1]])]
                    recs.append(rec)
                    rec_file.write(json.dumps(rec) + "\n")
                rec_file.flush()
            print(f"prompt {pi} offset {off}: {len(faults)} faults, {time.time() - t_start:.0f} s", flush=True)
    rec_file.close()
    summary = {"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": V, "boxes": str(a.boxes),
               "prompts": a.prompts, "window": a.window, "edit_bits": edit_bits, "new": a.new, "near": a.near,
               "cont": a.cont, "eps": a.eps, "slot_bits": a.slot_bits, "precision": a.precision, "kinds": kinds,
               "layers": layers, "offsets": offsets, "faults": len(recs), "seconds": time.time() - t_start,
               "groups": summarize(recs)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
