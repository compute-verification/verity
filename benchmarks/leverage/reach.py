"""Experiment 1: one fault's reach in the served output.

For each prompt: its honest greedy continuation of N tokens, then batches of faults, each setting one coordinate of one
site's activation, at the prompt's last position (the forward that serves the first new token), to m times that
activation's RMS there (m from --mults; 0 zeroes it). Each faulted sequence decodes greedily and honestly from there.
Beside it, in lockstep, the honest model reads the same tokens, so at each step it gives what an honest query on the
faulted prefix would serve.

Per fault, over the N served positions:
  flip    the first served token differs from the honest one
  raw     served tokens that differ from the honest continuation
  direct  served tokens that differ from what the honest model serves on the same prefix. Every other served token is
          the honest answer to a prompt the decoder could send itself, so these are the only outputs that honest
          queries cannot reproduce.
  lp      positions whose top-k served logprobs differ from the honest model's on the same prefix by more than --lp-tol
  kl      KL(faulted || honest) per position, on the same prefix, at the sampling temperature (1 when greedy)
  rank    the served token's rank among the honest scores the sampler reads on the same prefix (1: the honest token)

`summary.json` aggregates per (site, layer, m) and gives each group's information bounds: with p_n the share of faults
whose n-th served token is direct, sum_n h(p_n) + p_n log2(V - 1) bits bounds what a fault's served tokens tell a
decoder that may also make honest queries for free. If those queries also give the honest distribution on any prefix,
the rank code bounds it tighter: sum_n h(p_n) plus, for each direct token, Elias's gamma code of its rank - 1.

--temperature T > 0 samples by Gumbel-max, with the noise fixed by (seed, prompt, step) and shared by every decode of
the prompt. The faulted decode and the honest one on the same prefix are then coupled, and direct counts the tokens an
honest-use party that knows the seeds doesn't serve.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/reach.py --model gpt2 --out "$OUT"
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
from lm import LM, MODELS, Cache, Fault, served_logprobs
from prompts import PROMPTS

MULTS = (0.0, 1.0, -1.0, 8.0, -8.0, 64.0, -64.0, 512.0, -512.0, 4096.0, -4096.0)


def sites_of(lm: LM, spec: str) -> list[tuple[str, int]]:
    L = lm.spec.layers
    if spec == "all":
        return [("emb", -1)] + [(s, l) for l in range(L) for s in ("attn", "mlph", "mlp", "resid")] + [("final", L)]
    out = []
    for part in spec.split(","):
        s, l = part.split(":")
        out.append((s, int(l) if int(l) >= 0 or s == "emb" else L + int(l)))
    return out


def hb(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def elias_gamma_bits(x: np.ndarray) -> np.ndarray:
    """Code length of Elias's gamma code for integers x >= 1."""
    return 2 * np.floor(np.log2(np.maximum(x, 1))) + 1


class Sampler:
    """Greedy at temperature 0; else Gumbel-max sampling from noise fixed by (seed, prompt, step).

    Every decode of a prompt reads the same noise at a step, so the faulted decode and the honest one on the same
    prefix are coupled: an honest-use party that knows the seeds serves what the honest model serves under them.
    """

    def __init__(self, temperature: float, seed: int, vocab: int):
        self.t, self.seed, self.vocab = temperature, seed, vocab

    def noise(self, prompt: int, step: int) -> torch.Tensor | None:
        if self.t == 0:
            return None
        g = torch.Generator().manual_seed((self.seed * 1_000_003 + prompt) * 1_000_003 + step)
        u = torch.rand(self.vocab, generator=g, dtype=torch.float64).clamp_min(1e-300)
        return (-torch.log(-torch.log(u))).float()

    def scores(self, lp: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        return lp if g is None else lp / self.t + g


def lockstep(lm: LM, pre: Cache, last: torch.Tensor, fault: Fault, b: int, n: int, k: int, precision: str,
             sampler: Sampler, prompt: int):
    """Faulted and honest-on-the-same-prefix decodes, side by side; per step the served tokens and comparisons.

    Per step also the served token's rank among the honest scores the sampler reads (1 when it is the honest token)
    and its surprisal in bits under the honest distribution at the sampling temperature (1 when greedy).
    """
    cf, ch = pre.expand(b), pre.expand(b)
    tok = last.expand(b, 1)
    tk = sampler.t or 1.0
    ys, yh, kl, lpd, rank, sur = [], [], [], [], [], []
    nan = torch.zeros(b, dtype=torch.bool)
    for step in range(n):
        pf = served_logprobs(lm.forward(tok, cf, fault=fault if step == 0 else None)[:, -1], precision)
        ph = served_logprobs(lm.forward(tok, ch)[:, -1], precision)
        bad = ~torch.isfinite(pf).all(-1)
        nan |= bad
        pf = torch.where(bad[:, None], ph, pf)
        g = sampler.noise(prompt, step)
        sh = sampler.scores(ph, g)
        y = sampler.scores(pf, g).argmax(-1)
        ys.append(y)
        yh.append(sh.argmax(-1))
        rank.append(1 + (sh > sh.gather(1, y[:, None])).sum(-1))
        lf, lh = torch.log_softmax(pf / tk, -1), torch.log_softmax(ph / tk, -1)
        sur.append(-lh.gather(1, y[:, None])[:, 0] / math.log(2))
        w = lf.exp()
        kl.append(torch.where(w > 0, w * (lf - lh), torch.zeros_like(w)).sum(-1).clamp_min(0))
        top = pf.topk(k, -1).indices
        lpd.append((pf.gather(1, top) - ph.gather(1, top)).abs().max(-1).values)
        tok = y[:, None]
    st = lambda xs: torch.stack(xs, 1)  # noqa: E731
    return st(ys), st(yh), st(kl), st(lpd), st(rank), st(sur), nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--new", type=int, default=32, help="served positions N after the fault")
    ap.add_argument("--sites", default="all", help="'all' or site:layer,... (negative layers count from the end)")
    ap.add_argument("--coords", type=int, default=4, help="random coordinates per site and prompt")
    ap.add_argument("--mults", default=",".join(str(m) for m in MULTS))
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--lp-tol", type=float, default=1e-3)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="the served logits' precision")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0: greedy; else Gumbel-max sampling, every decode of a prompt reading the same noise")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    mults = [float(m) for m in a.mults.split(",")]
    lm = LM(a.model)
    tok = lm.tokenizer()
    V = lm.spec.vocab
    sampler = Sampler(a.temperature, a.seed, V)
    sites = sites_of(lm, a.sites)
    gen = torch.Generator().manual_seed(a.seed)
    recs: dict[str, list] = {k: [] for k in ("prompt", "site", "layer", "coord", "mult", "value", "honest_value",
                                             "flip", "raw", "direct", "first_div", "lp", "nan")}
    direct_pos, kl_all, lpd_all, rank_all, sur_all = [], [], [], [], []
    t_start = time.time()
    for pi, text in enumerate(PROMPTS[:a.prompts]):
        ids = torch.tensor([tok.encode(text).ids])
        t0 = ids.shape[1] - 1
        pre = Cache()
        lm.forward(ids[:, :t0], pre)
        taps = {s: None for s in sites}
        c = pre.expand(1)
        lp = served_logprobs(lm.forward(ids[:, t0:], c, taps=taps)[:, -1], a.precision)
        honest = [sampler.scores(lp, sampler.noise(pi, 0)).argmax(-1)]
        for step in range(1, a.new):
            lp = served_logprobs(lm.forward(honest[-1][:, None], c)[:, -1], a.precision)
            honest.append(sampler.scores(lp, sampler.noise(pi, step)).argmax(-1))
        honest = torch.cat(honest)
        faults = []
        for s in sites:
            x = taps[s][0, 0].float()
            rms = float(x.pow(2).mean().sqrt())
            for coord in torch.randperm(x.numel(), generator=gen)[:a.coords].tolist():
                for m in mults:
                    faults.append((s[0], s[1], coord, m * rms, m, float(x[coord])))
        for i in range(0, len(faults), a.batch):
            chunk = faults[i:i + a.batch]
            f = Fault.of(t0, [(s, l, c_, v) for s, l, c_, v, _, _ in chunk])
            ys, yh, kl, lpd, rank, sur, nan = lockstep(lm, pre, ids[:, t0:], f, len(chunk), a.new, a.topk,
                                                       a.precision, sampler, pi)
            direct = ys != yh
            raw = ys != honest[None, :]
            first = torch.where(raw.any(1), raw.float().argmax(1), torch.full((len(chunk),), -1))
            for j, (s, l, c_, v, m, hv) in enumerate(chunk):
                for key, val in (("prompt", pi), ("site", s), ("layer", l), ("coord", c_), ("mult", m),
                                 ("value", float(torch.tensor(v).to(torch.bfloat16))), ("honest_value", hv),
                                 ("flip", bool(raw[j, 0])), ("raw", int(raw[j].sum())), ("direct", int(direct[j].sum())),
                                 ("first_div", int(first[j])), ("lp", int((lpd[j] > a.lp_tol).sum())),
                                 ("nan", bool(nan[j]))):
                    recs[key].append(val)
            direct_pos.append(direct.numpy())
            kl_all.append(kl.numpy().astype(np.float32))
            lpd_all.append(lpd.numpy().astype(np.float32))
            rank_all.append(rank.numpy().astype(np.int32))
            sur_all.append(sur.numpy().astype(np.float32))
        print(f"prompt {pi}: {len(faults)} faults, {time.time() - t_start:.0f} s", flush=True)

    D = np.concatenate(direct_pos)
    KL = np.concatenate(kl_all)
    LPD = np.concatenate(lpd_all)
    RANK = np.concatenate(rank_all)
    SUR = np.concatenate(sur_all)
    # A direct token's rank is at least 2; gamma-code rank - 1, given the positions.
    rank_code = np.where(D, elias_gamma_bits(RANK - 1), 0.0)
    xent = np.where(D, SUR, 0.0)
    site_arr, layer_arr, mult_arr = np.array(recs["site"]), np.array(recs["layer"]), np.array(recs["mult"])
    groups = []
    for s, l in sites:
        for m in mults:
            sel = (site_arr == s) & (layer_arr == l) & (mult_arr == m)
            if not sel.any():
                continue
            p = D[sel].mean(0)
            positions = float((hb(p) * (p > 0)).sum())
            ub = positions + float((p * math.log2(V - 1)).sum())
            pick = lambda key: np.array(recs[key])[sel]  # noqa: E731
            groups.append({
                "site": s, "layer": l, "mult": m, "n": int(sel.sum()),
                "flip": float(pick("flip").mean()), "raw": float(pick("raw").mean()),
                "direct": float(pick("direct").mean()), "direct_first": float(D[sel][:, 0].mean()),
                "direct_after_first": float(D[sel][:, 1:].sum(1).mean()), "lp": float(pick("lp").mean()),
                "kl_first": float(KL[sel][:, 0].mean()), "kl_after_first": float(KL[sel][:, 1:].sum(1).mean()),
                "nan": float(pick("nan").mean()), "direct_bits_bound": ub,
                "rank_bits_bound": positions + float(rank_code[sel].sum(1).mean()),
                "xent_bits": positions + float(xent[sel].sum(1).mean()),
                "kl_bits": float(np.minimum(KL[sel] / math.log(2), math.log2(V)).sum(1).mean()),
                "raw_bits": float(pick("raw").mean() * math.log2(V)),
            })
    summary = {"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": V, "bits_per_token": math.log2(V),
               "new": a.new, "prompts": a.prompts, "coords": a.coords, "mults": mults, "topk": a.topk,
               "lp_tol": a.lp_tol, "precision": a.precision, "temperature": a.temperature,
               "faults": len(recs["site"]), "seconds": time.time() - t_start, "groups": groups}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(a.out / "faults.npz", direct_positions=D, kl=KL, lpd=LPD, rank=RANK, surprisal=SUR,
                        **{f"fault_{k}": np.array(v) for k, v in recs.items()})
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
