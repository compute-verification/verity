"""Experiment 7: one fault in a shared prefix, read by every later served token and by every request that shares it.

The fault sits in a prefix (at the attention sink's position by default, or an ordinary one), so what it changes is
cached as keys and values that every later position attends to. That is one incorrect proof block read by every
served token of a long decode, and by every request that shares the prefix (a chat session's earlier turns, a cached
system prompt). `reach.py` and `attack.py` fault the prompt's last position; this puts the fault where it persists.

Read modes (--mode):
  decode  each prompt after the faulted prefix, N served tokens (--new). Per fault, the direct tokens (those no honest
          query on the served prefix reproduces, in lockstep as in reach.py) and the tokens-only code that bounds
          what they tell a decoder: Elias gamma of D + 1, log2 C(N, D) for their positions, and each direct token
          either new (1 + log2(V - 1) bits) or a repeat of one already seen (1 + log2 of how many).
  oracle  Q receiver-chosen requests after the faulted prefix (--queries), the first --obs served tokens of each, the
          same code over all Q·obs reads of the one fault, and two decoders that never see the weights:
            pairs  each direct read (honest y0, served y1) is read as "the fault prefers y1 to y0", scored against
                   the order of the vocabulary along the fault's leave-one-out mean effect on the final norm's output
                   (u: what the logits move by, E u, is a fixed function of the weights if u is a fixed direction);
            top    the distinct served tokens of the direct reads, scored as the vocabulary's top set along u: if all
                   of them rank within u's top K, they carry log2 C(V, n) - log2 C(K, n) bits about u's order. The
                   honest tokens of the same requests, ranked the same way, are the baseline.

Faults (--values): unclamped, multiples of the site's RMS at the fault position, `zero`, and `neg` (the honest value's
negation); under --boxes, `box` (plus and minus the coordinate's box at the position's class, the most a correct
reader admits), `own` at a linear's input (one FP8 code at its token's own group scale), `zero` and `neg`. Coordinates
(--coords per site): the widest at the position's class (the massive-activation channels at the sink), then random.

Sampling (--temperature): 0 is greedy. Above 0, --seeds known fixes the Gumbel noise by (seed, request, step) and the
faulted and honest decodes read the same noise (a decoder that knows the seed couples them); --seeds hidden draws
independent noise for each of --repeats reads of a request, faulted and honest, and the decoder compares the
pluralities. A seed the server can't control is either: the decoder's own (known) or a trusted device's (hidden).

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/persist.py --model gpt2 --mode oracle \\
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
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, Fault, served_logprobs, sink_span, ue8m0_scale
from prompts import PROMPTS
from traffic import CALIBRATION_TEXTS, HELDOUT_TEXTS

PREFIX = "You are a helpful assistant. Answer every question briefly and accurately.\n\n"
SITES = ("resid", "mlph", "v", "k", "attn", "mlp")


def copy_cache(c: Cache) -> Cache:
    return Cache(dict(c.kv), c.sink)


def fault_position(lm: LM, ids: torch.Tensor, where: str) -> int:
    """The sink: the first sink host (the boxes' hosts, else position 0); ordinary: the prefix's last position."""
    if where == "ordinary":
        return ids.shape[1] - 1
    if lm.boxes is not None and lm.boxes.hosts is not None:
        return int(sink_span(ids, lm.boxes.hosts)[0, 0])
    return 0


def layers_of(lm: LM, spec: str) -> list[int]:
    L = lm.spec.layers
    return sorted({(int(x) if int(x) >= 0 else L + int(x)) for x in spec.split(",")})


def faults_at(lm: LM, taps: dict, pf: int, pc: int, sites: list[str], layers: list[int], values: list[str],
              coords: int, gen: torch.Generator) -> list[dict]:
    """The faults at position pf (class pc): per (site, layer), the `coords` widest coordinates at the class (by box, or
    unclamped by honest |value|), then random ones, each at every value of `values`."""
    out = []
    for s in sites:
        for l in layers:
            x = taps[(s, l)][0, pf].float()
            rms = float(x.pow(2).mean().sqrt())
            hi = lm.boxes.hi[(s, l)][pc].float() if lm.boxes is not None else None
            order = (hi if hi is not None else x.abs()).argsort(descending=True, stable=True).tolist()
            n_wide = (coords + 1) // 2
            rest = [c for c in torch.randperm(x.numel(), generator=gen).tolist() if c not in order[:n_wide]]
            for rank, c in enumerate(order[:n_wide] + rest[:coords - n_wide]):
                hv = float(x[c])
                cands = []
                for v in values:
                    if v == "zero":
                        cands.append((v, 0.0))
                    elif v == "neg":
                        cands.append((v, -hv))
                    elif v == "box":
                        if hi is None:
                            raise ValueError("--values box needs --boxes")
                        cands += [("box+", float(hi[c])), ("box-", -float(hi[c]))]
                    elif v == "own":
                        if s not in GROUP_SITES:
                            continue
                        g = x[c // GROUP * GROUP:(c // GROUP + 1) * GROUP]
                        own = E4M3_MAX * float(ue8m0_scale(g.abs().max()))
                        if hi is not None:
                            own = min(own, float(hi[c]))
                        cands += [("own+", own), ("own-", -own)]
                    else:
                        m = float(v)
                        cands += [(f"rms{v}", m * rms), (f"rms-{v}", -m * rms)]
                for kind, val in cands:
                    out.append({"site": s, "layer": l, "coord": c, "wide": rank < n_wide, "kind": kind,
                                "value": float(torch.tensor(val).to(torch.bfloat16)), "honest": hv, "rms": rms,
                                "box": float(hi[c]) if hi is not None else None})
    return out


def suffixes(lm: LM, tok, pre_h: Cache, n: int, gen: torch.Generator, cut: str, precision: str, lo: int = 6,
             hi: int = 48) -> list[torch.Tensor]:
    """n receiver-chosen requests after the prefix: PROMPTS and the texts of traffic.py, each cut after some tokens
    (`random`, or `margin`: where the honest top two are closest, which a receiver finds by honest sampling under
    seeds it controls), a text's k-th use at its k-th cut, interleaved so that every prefix of the list spans texts."""
    texts = list(PROMPTS) + list(HELDOUT_TEXTS) + list(CALIBRATION_TEXTS)
    per = -(-n // len(texts))
    cuts = []
    for t in texts:
        ids = tok.encode(t).ids[:hi]
        if len(ids) <= lo:
            continue
        if cut == "random":
            ks = (torch.randperm(len(ids) - lo + 1, generator=gen)[:per] + lo).tolist()
        else:
            lp = served_logprobs(lm.forward(torch.tensor([ids]), copy_cache(pre_h))[0], precision)
            top = lp.topk(2, -1).values
            margin = top[:, 0] - top[:, 1]
            margin[:lo - 1] = math.inf
            ks = (margin.argsort()[:per] + 1).tolist()
        cuts.append([torch.tensor([ids[:k]]) for k in ks])
    out = [c[r] for r in range(per) for c in cuts if r < len(c)]
    return out[:n]


class Noise:
    def __init__(self, temperature: float, seed: int, vocab: int):
        self.t, self.seed, self.vocab = temperature, seed, vocab

    def gumbel(self, *key: int, rows: int = 1) -> torch.Tensor | None:
        if self.t == 0:
            return None
        h = self.seed
        for k in key:
            h = (h * 1_000_003 + k) % (1 << 62)
        g = torch.Generator().manual_seed(h)
        u = torch.rand(rows, self.vocab, generator=g, dtype=torch.float64).clamp_min(1e-300)
        return (-torch.log(-torch.log(u))).float()

    def pick(self, lp: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        return lp.argmax(-1) if g is None else (lp / self.t + g).argmax(-1)


def gamma_bits(x: int) -> float:
    return 2 * math.floor(math.log2(max(x, 1))) + 1


def code_bits(direct: np.ndarray, served: np.ndarray, vocab: int) -> float:
    """A tokens-only prefix code for one fault's reads given the honest ones: how many differ, where, and what each
    differing read served (a new token, or a repeat of one already seen)."""
    n, d = direct.size, int(direct.sum())
    bits = gamma_bits(d + 1) + (math.lgamma(n + 1) - math.lgamma(d + 1) - math.lgamma(n - d + 1)) / math.log(2)
    seen: list[int] = []
    for y in served[direct]:
        y = int(y)
        if y in seen:
            bits += 1 + math.log2(max(len(seen), 1))
        else:
            bits += 1 + math.log2(vocab - 1)
            seen.append(y)
    return bits


def log2_comb(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def pair_bits(n: int, acc: float) -> float:
    """n binary order claims right at rate acc, as independent bits: n (1 - h(acc)), none at or below chance."""
    if acc <= 0.5:
        return 0.0
    h = 0.0 if acc == 1.0 else -(acc * math.log2(acc) + (1 - acc) * math.log2(1 - acc))
    return n * (1 - h)


def pair_scores(pairs: set[tuple[int, int]], target: torch.Tensor | None, controls: list[torch.Tensor],
                E: torch.Tensor) -> dict:
    """Distinct (honest y0, served y1) pairs as claims "y1 is above y0", scored along the signed target and along each
    control column."""
    if target is None or not pairs:
        return {}
    pr = torch.tensor(sorted(pairs))
    gap = E[pr[:, 1]] - E[pr[:, 0]]
    acc_w = float(((gap @ target) > 0).float().mean())
    acc_c = [float(((gap @ c) > 0).float().mean()) for c in controls]
    return {"pairs": len(pr), "pair_acc_w": acc_w, "pair_bits_w": pair_bits(len(pr), acc_w),
            "pair_acc_ctrl": float(np.mean(acc_c)) if acc_c else None,
            "pair_bits_ctrl": float(np.mean([pair_bits(len(pr), x) for x in acc_c])) if acc_c else None}


def top_bits(ranks: np.ndarray, vocab: int) -> tuple[float, int]:
    """Bits the set of tokens at these 0-based ranks along u says about u's order: log2 C(V, n) - log2 C(K, n), K the
    smallest top set holding them all."""
    ranks = np.unique(ranks)
    if len(ranks) == 0:
        return 0.0, 0
    k = int(ranks.max()) + 1
    return log2_comb(vocab, len(ranks)) - log2_comb(k, len(ranks)), k


def honest_first(lm: LM, pre_h: Cache, sfx: torch.Tensor, precision: str) -> tuple[torch.Tensor, torch.Tensor]:
    """A request's honest first position after the honest prefix: served logprobs (1, V), final norm output (1, d)."""
    L = lm.spec.layers
    th = {("final", L): None}
    z = lm.forward(sfx, copy_cache(pre_h), taps=th, last_only=True)[:, -1]
    return served_logprobs(z, precision), th[("final", L)][:, -1].float()


def serve_oracle(lm: LM, pre_f: Cache, pre_h: Cache, sfx: torch.Tensor, hon: tuple[torch.Tensor, torch.Tensor],
                 b: int, obs: int, noise: Noise, q: int, precision: str, hidden: int) -> dict:
    """One request after the faulted prefixes (b rows) and the honest one (`hon`, its first position): served tokens,
    honest tokens on the same served prefix, and the final norm's output at the first served position (for scoring
    only)."""
    L = lm.spec.layers
    cf = copy_cache(pre_f)
    tf = {("final", L): None}
    zf = lm.forward(sfx.expand(b, -1), cf, taps=tf, last_only=True)[:, -1]
    lf, lh = served_logprobs(zf, precision), hon[0].expand(b, -1)
    out = {"delta": tf[("final", L)][:, -1].float() - hon[1]}
    if hidden:
        # Independent noise per read: the decoder compares pluralities over `hidden` reads of each.
        gf, gh = noise.gumbel(q, 1, rows=hidden), noise.gumbel(q, 2, rows=hidden)
        yf = torch.stack([noise.pick(lf, gf[r:r + 1].expand(b, -1)) for r in range(hidden)], 1)
        yh = torch.stack([noise.pick(lh, gh[r:r + 1].expand(b, -1)) for r in range(hidden)], 1)
        out["served"] = torch.mode(yf, 1).values[:, None]
        out["honest"] = torch.mode(yh, 1).values[:, None]
        tk = noise.t
        a, c = torch.log_softmax(lf / tk, -1), torch.log_softmax(lh / tk, -1)
        out["kl"] = (a.exp() * (a - c)).sum(-1).clamp_min(0)[:, None] / math.log(2)
        return out
    ys, yh_ = [], []
    ch = None
    if obs > 1:
        ch = copy_cache(pre_h).expand(b)
        lm.forward(sfx.expand(b, -1), ch, last_only=True)
    for step in range(obs):
        if step:
            lf = served_logprobs(lm.forward(ys[-1][:, None], cf)[:, -1], precision)
            lh = served_logprobs(lm.forward(ys[-1][:, None], ch)[:, -1], precision)
        g = noise.gumbel(q, step)
        ys.append(noise.pick(lf, g))
        yh_.append(noise.pick(lh, g))
    out["served"], out["honest"] = torch.stack(ys, 1), torch.stack(yh_, 1)
    return out


def decode(lm: LM, pre_f: Cache, pre_h: Cache, prompt: torch.Tensor, b: int, n: int, noise: Noise, p: int,
           precision: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The faulted decode and the honest one on the same served prefix, in lockstep: served, honest, KL (bits)."""
    cf, ch = copy_cache(pre_f), copy_cache(pre_h).expand(b)
    tok = prompt.expand(b, -1)
    ys, yh, kl = [], [], []
    tk = noise.t or 1.0
    for step in range(n):
        lf = served_logprobs(lm.forward(tok, cf, last_only=True)[:, -1], precision)
        lh = served_logprobs(lm.forward(tok, ch, last_only=True)[:, -1], precision)
        g = noise.gumbel(p, step)
        y = noise.pick(lf, g)
        ys.append(y)
        yh.append(noise.pick(lh, g))
        a, c = torch.log_softmax(lf / tk, -1), torch.log_softmax(lh / tk, -1)
        kl.append((a.exp() * (a - c)).sum(-1).clamp_min(0) / math.log(2))
        tok = y[:, None]
    return torch.stack(ys, 1), torch.stack(yh, 1), torch.stack(kl, 1)


def weight_target(lm: LM, site: str, layer: int, coord: int) -> torch.Tensor | None:
    """For a value-vector fault: the direction it adds to every later position's final norm output, read from the
    weights alone (the output projection's column for that coordinate, summed over the query heads sharing it,
    through the final norm's gain; GPT-2's LayerNorm also centres it). Exact up to the norm's scale at the last layer;
    at earlier layers the direct path only. None for other sites."""
    s, W = lm.spec, lm.W
    if site != "v":
        return None
    if s.arch == "gpt2":
        w = W[f"h.{layer}.attn.c_proj.weight"][coord].float()
        return (w - w.mean()) * W["ln_f.weight"].float()
    kh, e = divmod(coord, s.head_dim)
    r = s.heads // s.kv_heads
    wo = W[f"layers.{layer}.self_attn.o_proj.weight"].float()
    return sum(wo[:, (kh * r + i) * s.head_dim + e] for i in range(r)) * W["norm.weight"].float()


def signed_targets(lm: LM, f: dict, n: int, seed: int) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """A fault's weight target signed by its push (value - honest), and n other columns of its matrix signed alike."""
    t = weight_target(lm, f["site"], f["layer"], f["coord"])
    if t is None:
        return None, []
    sgn = 1.0 if f["value"] >= f["honest"] else -1.0
    rng = np.random.default_rng([seed, f["layer"], f["coord"]])
    others = [c for c in rng.permutation(v_width(lm)).tolist() if c != f["coord"]][:n]
    return sgn * t, [sgn * weight_target(lm, f["site"], f["layer"], c) for c in others]


def v_width(lm: LM) -> int:
    s = lm.spec
    return s.heads * s.head_dim if s.arch == "gpt2" else s.kv_heads * s.head_dim


def ranks_along(E: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Each token's 0-based rank by its logit change along t (E t), largest first."""
    order = torch.argsort(E @ t, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(order))
    return rank


def decoded(served: np.ndarray, direct: np.ndarray) -> list[int]:
    """The decoder's output: the distinct tokens the fault served in place of the honest one, most frequent first."""
    ys, counts = np.unique(served[direct], return_counts=True)
    return [int(y) for y in ys[np.argsort(-counts, kind="stable")]]


def score_oracle(served: np.ndarray, honest: np.ndarray, runner: np.ndarray, deltas: np.ndarray,
                 target: torch.Tensor | None, controls: list[torch.Tensor], ctrl_u: list[torch.Tensor],
                 E: torch.Tensor, vocab: int, checkpoints: list[int]) -> dict:
    """The decoders' bits for one fault over Q requests (first served token). deltas (Q, d): the fault's effect on the
    final norm's output at each request's first served position; runner (Q,): each request's honest runner-up, the
    token a small push serves without any information about the fault's direction (the baseline). target: the
    weights' own direction for the fault, signed by its push; controls: other columns of the same matrix, signed
    alike, which score the same decoded tokens for what common tokens give along any column. ctrl_u: other faults'
    mean effects, the control of the split-half score (the mean effect fitted on one half of the requests, the
    decoded tokens read from the other)."""
    s0, h0 = served[:, 0], honest[:, 0]
    direct = s0 != h0
    Q = len(direct)
    D = torch.from_numpy(deltas).float()
    tot = D.sum(0)
    u_all = tot / Q
    out = {"direct": int(direct.sum()), "reads": Q, "delta_norm": float(D.norm(dim=1).mean())}
    targets = {"u": ranks_along(E, u_all)} if float(u_all.norm()) > 0 else {}
    if target is not None:
        targets["w"] = ranks_along(E, target)
        out["cos_u_w"] = float(torch.nn.functional.cosine_similarity(u_all, target, 0))
    cos, ok, seen = [], 0, set()
    for q in np.nonzero(direct)[0]:
        u = (tot - D[q]) / max(Q - 1, 1)
        if float(u.norm()) == 0:
            continue
        cos.append(float(torch.nn.functional.cosine_similarity(D[q], u, 0)))
        y0, y1 = int(h0[q]), int(s0[q])
        if (y0, y1) not in seen:
            seen.add((y0, y1))
            ok += int(float((E[y1] - E[y0]) @ u) > 0)
    n_pairs = len(seen)
    acc = ok / n_pairs if n_pairs else 0.0
    out |= {"pairs": n_pairs, "pairs_ok": ok, "pair_bits": pair_bits(n_pairs, acc),
            "cos_mean": float(np.mean(cos)) if cos else None}
    cu_ranks = [ranks_along(E, c) for c in ctrl_u]
    if seen and cu_ranks:
        pr = torch.tensor(sorted(seen))
        gap = E[pr[:, 1]] - E[pr[:, 0]]
        acc_cu = [float(((gap @ c) > 0).float().mean()) for c in ctrl_u]
        out |= {"pair_acc_u_ctrl": float(np.mean(acc_cu)),
                "pair_bits_u_ctrl": float(np.mean([pair_bits(n_pairs, x) for x in acc_cu]))}
    if Q >= 4 and float(u_all.norm()) > 0:
        tb, cb = [], []
        for k in (0, 1):
            fit, read = np.arange(Q) % 2 == k, np.arange(Q) % 2 != k
            dec = decoded(s0[read], direct[read])
            if dec:
                tb.append(top_bits(ranks_along(E, D[torch.from_numpy(fit)].mean(0))[dec].numpy(), vocab)[0])
                cb.append(float(np.mean([top_bits(r[dec].numpy(), vocab)[0] for r in cu_ranks])) if cu_ranks else 0.0)
        if tb:
            out |= {"split_bits_u": float(np.mean(tb)), "split_ctrl_u": float(np.mean(cb)),
                    "split_net_u": float(np.mean(tb) - np.mean(cb))}
    ctrl_ranks = [ranks_along(E, c) for c in controls]
    out |= pair_scores(seen, target, controls, E)
    for qc in checkpoints:
        if qc > Q:
            continue
        dq = direct[:qc]
        dec = decoded(s0[:qc], dq)
        base = sorted({int(r) for r in runner[:qc][dq]})
        row = {"direct": int(dq.sum()), "code_bits": code_bits(dq, s0[:qc], vocab), "decoded": len(dec)}
        row |= pair_scores({(int(h0[q]), int(s0[q])) for q in np.nonzero(dq)[0]}, target, controls, E)
        for name, rank in targets.items():
            rb, rk = top_bits(rank[dec].numpy() if dec else np.array([], dtype=np.int64), vocab)
            bb, bk = top_bits(rank[base].numpy() if base else np.array([], dtype=np.int64), vocab)
            row |= {f"top_bits_{name}": rb, f"top_k_{name}": rk, f"base_bits_{name}": bb, f"base_k_{name}": bk,
                    f"p1_{name}": int(bool(dec) and int(rank[dec[0]]) == 0),
                    f"in_top10_{name}": int(sum(int(rank[y]) < 10 for y in dec[:10]))}
        if ctrl_ranks and dec:
            cb = [top_bits(cr[dec].numpy(), vocab)[0] for cr in ctrl_ranks]
            row |= {"ctrl_bits_w": float(np.mean(cb)), "ctrl_max_w": float(np.max(cb)),
                    "net_bits_w": row.get("top_bits_w", 0.0) - float(np.mean(cb))}
        out[f"q{qc}"] = row
    return out


def score_flips(flips: list[tuple[int, int]], target: torch.Tensor | None, controls: list[torch.Tensor],
                E: torch.Tensor, vocab: int) -> dict:
    """A decode's weight bits from its (honest, served) pairs where the two differ: the served tokens as a decoded
    set, and each distinct pair as an order claim, scored along the fault's signed target and its controls."""
    if target is None or not flips:
        return {}
    dec = decoded(np.array([y for _, y in flips]), np.ones(len(flips), bool))
    rank = ranks_along(E, target)
    rb, rk = top_bits(rank[dec].numpy(), vocab)
    cb = [top_bits(ranks_along(E, c)[dec].numpy(), vocab)[0] for c in controls] or [0.0]
    return {"decoded": len(dec), "top_bits_w": rb, "top_k_w": rk, "ctrl_bits_w": float(np.mean(cb)),
            "net_bits_w": rb - float(np.mean(cb)), "p1_w": int(int(rank[dec[0]]) == 0),
            "in_top10_w": int(sum(int(rank[y]) < 10 for y in dec[:10]))} | pair_scores(set(flips), target, controls, E)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="oracle", choices=("oracle", "decode"))
    ap.add_argument("--boxes", type=Path, help="clamp every read to clamp.py's calibrated boxes.pt")
    ap.add_argument("--where", default="sink", choices=("sink", "ordinary"), help="the fault's prefix position")
    ap.add_argument("--prefix", default=PREFIX)
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--layers", default="0,2,5,8,10,11", help="comma-separated; negative from the end")
    ap.add_argument("--values", default="zero,neg,8,64,512,4096",
                    help="zero, neg, RMS multiples (unclamped), box and own (under --boxes)")
    ap.add_argument("--coords", type=int, default=4, help="per site and layer: the widest half, then random")
    ap.add_argument("--queries", type=int, default=128, help="oracle: receiver-chosen requests sharing the prefix")
    ap.add_argument("--obs", type=int, default=1, help="oracle: served tokens a request returns")
    ap.add_argument("--cut", default="margin", choices=("margin", "random"),
                    help="oracle: where each request's text ends (margin: at the honest near-tie)")
    ap.add_argument("--prompts", type=int, default=4, help="decode: prompts after the prefix")
    ap.add_argument("--new", type=int, default=128, help="decode: served tokens")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seeds", default="known", choices=("known", "hidden"))
    ap.add_argument("--repeats", type=int, default=16, help="--seeds hidden: reads of each request, faulted and honest")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-faults", type=int, default=0, help="keep at most this many faults (0: all)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--controls", type=int, default=8,
                    help="random columns of the same matrix that score a value fault's control")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    if a.boxes is not None:
        lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    V = lm.spec.vocab
    gen = torch.Generator().manual_seed(a.seed)
    noise = Noise(a.temperature, a.seed, V)
    hidden = a.repeats if (a.temperature > 0 and a.seeds == "hidden") else 0
    if hidden and a.mode == "decode":
        ap.error("--seeds hidden is for --mode oracle; decode reports the KL bound")
    sites = a.sites.split(",")
    layers = layers_of(lm, a.layers)
    values = a.values.split(",")

    ids = torch.tensor([tok.encode(a.prefix).ids])
    pf = fault_position(lm, ids, a.where)
    taps = {(s, l): None for s in sites for l in layers}
    pre_h = Cache()
    lm.forward(ids, pre_h, taps=taps)
    pc = int(lm.boxes.classes(0, ids.shape[1], ids)[0][0, pf]) if lm.boxes is not None else 1
    faults = faults_at(lm, taps, pf, pc, sites, layers, values, a.coords, gen)
    if a.max_faults:
        faults = faults[:a.max_faults]
    E = lm.E.float()
    t_start = time.time()
    checkpoints = sorted({q for q in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, a.queries)
                          if q <= a.queries})
    if a.mode == "oracle":
        reqs = suffixes(lm, tok, pre_h, a.queries, gen, a.cut, a.precision)
        hons = [honest_first(lm, pre_h, sfx, a.precision) for sfx in reqs]
        runner = np.array([int(h[0][0].topk(2).indices[1]) for h in hons])
    else:
        reqs = [torch.tensor([tok.encode(t).ids]) for t in PROMPTS[:a.prompts]]
    recs = []
    for i in range(0, len(faults), a.batch):
        chunk = faults[i:i + a.batch]
        b = len(chunk)
        pre_f = Cache()
        lm.forward(ids.expand(b, -1), pre_f, fault=Fault.of(pf, [(f["site"], f["layer"], f["coord"], f["value"])
                                                                  for f in chunk]))
        if a.mode == "oracle":
            S, H, Dl, KL = [], [], [], []
            for q, sfx in enumerate(reqs):
                o = serve_oracle(lm, pre_f, pre_h, sfx, hons[q], b, a.obs, noise, q, a.precision, hidden)
                S.append(o["served"])
                H.append(o["honest"])
                Dl.append(o["delta"])
                if "kl" in o:
                    KL.append(o["kl"])
            S, H, Dl = torch.stack(S, 1).numpy(), torch.stack(H, 1).numpy(), torch.stack(Dl, 1).numpy()
            KL = torch.stack(KL, 1).numpy() if KL else None
            U = torch.from_numpy(Dl).float().mean(1)
            for j, f in enumerate(chunk):
                direct = S[j] != H[j]
                target, controls = signed_targets(lm, f, a.controls, a.seed)
                key = (f["site"], f["layer"], f["coord"])
                pool = [k for k in range(b) if (chunk[k]["site"], chunk[k]["layer"], chunk[k]["coord"]) != key
                        and float(U[k].norm()) > 0]
                picks = np.random.default_rng([a.seed, i + j]).permutation(pool)[:a.controls] if pool else []
                r = dict(f) | score_oracle(S[j], H[j], runner, Dl[j], target, controls, [U[k] for k in picks], E, V,
                                           checkpoints)
                r["code_bits"] = code_bits(direct.reshape(-1), S[j].reshape(-1), V)
                r["direct_reads"] = int(direct.sum())
                last = r[f"q{checkpoints[-1]}"]
                for key in ("top_bits_u", "base_bits_u", "top_bits_w", "base_bits_w", "ctrl_bits_w", "net_bits_w"):
                    if key in last:
                        r[key] = last[key]
                if KL is not None:
                    r["kl_bits"] = float(KL[j].sum())
                recs.append(r)
        else:
            per = [dict(f) | {"direct": 0, "code_bits": 0.0, "kl_bits": 0.0, "distinct": 0, "reads": 0} for f in chunk]
            flips = [[] for _ in chunk]
            for p, prompt in enumerate(reqs):
                ys, yh, kl = decode(lm, pre_f, pre_h, prompt, b, a.new, noise, p, a.precision)
                d = (ys != yh).numpy()
                for j in range(b):
                    per[j]["direct"] += int(d[j].sum())
                    per[j]["code_bits"] += code_bits(d[j], ys[j].numpy(), V)
                    per[j]["kl_bits"] += float(kl[j].sum())
                    per[j]["distinct"] += len(set(ys[j][torch.from_numpy(d[j])].tolist()))
                    per[j]["reads"] += a.new
                    k = np.nonzero(d[j])[0]
                    flips[j] += list(zip(yh[j].numpy()[k].tolist(), ys[j].numpy()[k].tolist()))
            for j, f in enumerate(chunk):
                target, controls = signed_targets(lm, f, a.controls, a.seed)
                per[j] |= score_flips(flips[j], target, controls, E, V)
            recs += per
        print(f"faults {i + b}/{len(faults)}, {time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
        (a.out / "persist.json").write_text(json.dumps({"partial": True, "faults": recs}))

    groups = {}
    for r in recs:
        key = (r["site"], r["layer"], r["kind"], r["wide"])
        groups.setdefault(key, []).append(r)
    rows = []
    for (s, l, kind, wide), rs in sorted(groups.items()):
        row = {"site": s, "layer": l, "kind": kind, "wide": wide, "n": len(rs),
               "direct_mean": float(np.mean([r["direct"] for r in rs])),
               "direct_max": int(max(r["direct"] for r in rs)),
               "code_mean": float(np.mean([r["code_bits"] for r in rs])),
               "code_max": float(max(r["code_bits"] for r in rs))}
        for key in ("pair_bits", "pair_bits_u_ctrl", "pair_bits_w", "pair_bits_ctrl", "top_bits_u", "base_bits_u",
                    "split_bits_u", "split_ctrl_u", "split_net_u", "top_bits_w", "base_bits_w", "ctrl_bits_w",
                    "net_bits_w", "kl_bits"):
            if any(r.get(key) is not None for r in rs):
                vals = [r.get(key) or 0.0 for r in rs]
                row[key + "_mean"], row[key + "_max"] = float(np.mean(vals)), float(max(vals))
        rows.append(row)
    keys = ("code_bits", "pair_bits", "pair_bits_w", "top_bits_u", "split_net_u", "top_bits_w", "net_bits_w")
    best = {key: max(recs, key=lambda r: r.get(key) or 0.0) for key in keys if any(key in r for r in recs)}
    summary = {"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": V, "bits_per_token": math.log2(V),
               "mode": a.mode, "where": a.where, "fault_position": pf, "fault_class": pc, "prefix_tokens": ids.shape[1],
               "boxes": str(a.boxes) if a.boxes else None, "temperature": a.temperature, "seeds": a.seeds,
               "repeats": hidden, "reads_per_fault": (a.queries * a.obs if a.mode == "oracle" else a.prompts * a.new),
               "faults": len(recs), "seconds": time.time() - t_start, "best": best, "groups": rows,
               "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}}
    if lm.boxes is not None:
        summary["values_clamped"] = {f"{s}:{l}": n for (s, l), n in lm.boxes.cut.items()}
    (a.out / "persist.json").write_text(json.dumps(summary | {"faults_detail": recs}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("groups",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
