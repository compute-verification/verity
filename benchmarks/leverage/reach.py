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

Each group also gives the worst case its faults reach: a fault's code length, Elias's gamma code of D + 1 (D its
direct tokens), log2 C(N, D) for their positions and the rank code of each, bounds what that one fault tells such a
decoder, so its maximum over a group bounds every fault in the group, not just their average.

--temperature T > 0 samples by Gumbel-max, with the noise fixed by (seed, prompt, step) and shared by every decode of
the prompt. The faulted decode and the honest one on the same prefix are then coupled, and direct counts the tokens an
honest-use party that knows the seeds doesn't serve.

--boxes clamps every forward to clamp.py's calibrated boxes, and --kinds replaces the RMS multiples by in-range faults:
  box     one coordinate at plus or minus its box
  own     a linear's input (lm.GROUP_SITES): one FP8 code at plus or minus 448 times its token's own group scale, the
          most a fault in one code moves it when another block writes the scale
  scale   a linear's input: one group's shared scale raised to its bound, moving all 128 of its values jointly
  codes   a linear's input: all 128 of one group's values at plus or minus its box, random signs (a block that writes
          a whole group's codes)

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
from lm import E4M3_MAX, GROUP, GROUP_SITES, LM, MODELS, Boxes, Cache, Fault, position_class, served_logprobs, ue8m0_scale
from prompts import PROMPTS

MULTS = (0.0, 1.0, -1.0, 8.0, -8.0, 64.0, -64.0, 512.0, -512.0, 4096.0, -4096.0)
KINDS = ("rms", "box", "own", "scale", "codes")


def sites_of(lm: LM, spec: str, layers: str | None = None) -> list[tuple[str, int]]:
    """'all': the embedding, attn, mlph, mlp and resid at every layer, and the final norm's output; 'every': every site;
    else site:layer,... (negative layers count from the end). --layers keeps only those layers' sites."""
    L = lm.spec.layers
    if spec in ("all", "every"):
        per = ("attn", "mlph", "mlp", "resid") if spec == "all" else [s for s, l in lm.sites() if l == 0]
        out = [("emb", -1)] + [(s, l) for l in range(L) for s in per] + [("final", L)]
    else:
        out = []
        for part in spec.split(","):
            s, l = part.split(":")
            out.append((s, int(l) if int(l) >= 0 or s == "emb" else L + int(l)))
    if layers:
        keep = {int(x) if int(x) >= 0 else L + int(x) for x in layers.split(",")}
        out = [(s, l) for s, l in out if s in ("emb", "final") or l in keep]
    return out


def faults_of(lm: LM, site: tuple[str, int], x: torch.Tensor, t0: int, kinds: list[str], mults: list[float],
              coords: int, groups: int, gen: torch.Generator) -> list[tuple]:
    """(site, layer, coords, values, kind, mult, honest_value, rms_mult) for one site's faults at position t0, x its
    honest activation there as read (after the clamp)."""
    rms = float(x.pow(2).mean().sqrt())
    out = []
    cs = torch.randperm(x.numel(), generator=gen)[:coords].tolist()
    gs = torch.randperm(x.numel() // GROUP, generator=gen)[:groups].tolist() if x.numel() % GROUP == 0 else []
    hi = lm.boxes.hi[site][int(position_class(torch.tensor(t0)))].float() if lm.boxes is not None else None
    grouped = site[0] in GROUP_SITES
    for kind in kinds:
        if kind == "rms":
            out += [(site, c, m * rms, kind, m, float(x[c])) for c in cs for m in mults]
        elif kind == "box":
            out += [(site, c, sg * float(hi[c]), kind, sg, float(x[c])) for c in cs for sg in (1.0, -1.0)]
        elif kind == "own" and grouped:
            for c in cs:
                own = E4M3_MAX * float(ue8m0_scale(x[c // GROUP * GROUP:(c // GROUP + 1) * GROUP].abs().max()))
                out += [(site, c, sg * min(own, float(hi[c])), kind, sg, float(x[c])) for sg in (1.0, -1.0)]
        elif kind == "scale" and grouped:
            for g in gs:
                xg = x[g * GROUP:(g + 1) * GROUP]
                f = float(hi[g * GROUP]) / E4M3_MAX / float(ue8m0_scale(xg.abs().max()))
                out.append((site, list(range(g * GROUP, (g + 1) * GROUP)), (xg * f).tolist(), kind, 1.0,
                            float(xg.abs().max())))
        elif kind == "codes" and grouped:
            for g in gs:
                sign = torch.randint(0, 2, (GROUP,), generator=gen).float() * 2 - 1
                xg = x[g * GROUP:(g + 1) * GROUP]
                for sg in (1.0, -1.0):
                    out.append((site, list(range(g * GROUP, (g + 1) * GROUP)), (sg * sign * hi[g * GROUP]).tolist(),
                                kind, sg, float(xg.abs().max())))
    res = []
    for (s, c, v, kind, m, hv) in out:
        vmax = max(abs(u) for u in v) if isinstance(v, list) else abs(v)
        res.append((s[0], s[1], c, v, kind, m, hv, vmax / max(rms, 1e-30)))
    return res


def fault_code_bits(D: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """Per fault: Elias gamma of D + 1, log2 C(N, D) for the positions, and gamma of rank - 1 for each direct token."""
    n = D.shape[1]
    d = D.sum(1)
    lgam = np.vectorize(math.lgamma)
    positions = (lgam(n + 1) - lgam(d + 1) - lgam(n - d + 1)) / math.log(2)
    return elias_gamma_bits(d + 1) + positions + np.where(D, elias_gamma_bits(rank - 1), 0).sum(1)


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
    ap.add_argument("--sites", default="all",
                    help="'all', 'every' or site:layer,... (negative layers count from the end)")
    ap.add_argument("--layers", help="keep only these layers' sites (comma-separated; negative from the end)")
    ap.add_argument("--coords", type=int, default=4, help="random coordinates per site and prompt")
    ap.add_argument("--groups", type=int, default=2, help="random 128-groups per site and prompt (joint kinds)")
    ap.add_argument("--mults", default=",".join(str(m) for m in MULTS))
    ap.add_argument("--kinds", default="rms", help=f"comma-separated, of {KINDS}; all but rms need --boxes")
    ap.add_argument("--boxes", type=Path, help="clamp every read to clamp.py's calibrated boxes.pt")
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
    kinds = a.kinds.split(",")
    if set(kinds) - set(KINDS) or (set(kinds) - {"rms"} and a.boxes is None):
        ap.error(f"--kinds: of {KINDS}, and all but rms need --boxes")
    lm = LM(a.model)
    if a.boxes is not None:
        lm.boxes = Boxes.load(a.boxes)
    tok = lm.tokenizer()
    V = lm.spec.vocab
    sampler = Sampler(a.temperature, a.seed, V)
    sites = sites_of(lm, a.sites, a.layers)
    gen = torch.Generator().manual_seed(a.seed)
    recs: dict[str, list] = {k: [] for k in ("prompt", "site", "layer", "coord", "kind", "mult", "value",
                                             "honest_value", "rms_mult", "flip", "raw", "direct", "first_div", "lp",
                                             "nan")}
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
            faults += faults_of(lm, s, taps[s][0, 0].float(), t0, kinds, mults, a.coords, a.groups, gen)
        for i in range(0, len(faults), a.batch):
            chunk = faults[i:i + a.batch]
            f = Fault.of(t0, [(s, l, c_, v) for s, l, c_, v, *_ in chunk])
            ys, yh, kl, lpd, rank, sur, nan = lockstep(lm, pre, ids[:, t0:], f, len(chunk), a.new, a.topk,
                                                       a.precision, sampler, pi)
            direct = ys != yh
            raw = ys != honest[None, :]
            first = torch.where(raw.any(1), raw.float().argmax(1), torch.full((len(chunk),), -1))
            for j, (s, l, c_, v, kind, m, hv, rm) in enumerate(chunk):
                vmax = max(v, key=abs) if isinstance(v, list) else v
                for key, val in (("prompt", pi), ("site", s), ("layer", l), ("coord", c_ if isinstance(c_, int) else c_[0]),
                                 ("kind", kind), ("mult", m), ("value", float(torch.tensor(vmax).to(torch.bfloat16))),
                                 ("honest_value", hv), ("rms_mult", rm),
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
    code = fault_code_bits(D, RANK)
    site_arr, layer_arr, mult_arr = np.array(recs["site"]), np.array(recs["layer"]), np.array(recs["mult"])
    kind_arr = np.array(recs["kind"])
    groups = []
    for s, l in sites:
        for kind, m in sorted(set(zip(recs["kind"], recs["mult"]))):
            sel = (site_arr == s) & (layer_arr == l) & (kind_arr == kind) & (mult_arr == m)
            if not sel.any():
                continue
            p = D[sel].mean(0)
            positions = float((hb(p) * (p > 0)).sum())
            ub = positions + float((p * math.log2(V - 1)).sum())
            pick = lambda key: np.array(recs[key])[sel]  # noqa: E731
            groups.append({
                "site": s, "layer": l, "kind": kind, "mult": m, "n": int(sel.sum()),
                "rms_mult": float(np.median(pick("rms_mult"))), "rms_mult_max": float(pick("rms_mult").max()),
                "code_mean": float(code[sel].mean()), "code_p90": float(np.quantile(code[sel], 0.9)),
                "code_max": float(code[sel].max()), "direct_max": int(D[sel].sum(1).max()),
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
               "new": a.new, "prompts": a.prompts, "coords": a.coords, "groups_per_site": a.groups, "mults": mults,
               "kinds": kinds, "boxes": str(a.boxes) if a.boxes else None, "topk": a.topk,
               "lp_tol": a.lp_tol, "precision": a.precision, "temperature": a.temperature,
               "faults": len(recs["site"]), "seconds": time.time() - t_start, "groups": groups}
    if lm.boxes is not None:
        summary["values_clamped_in_all_decodes"] = {f"{s}:{l}": n for (s, l), n in lm.boxes.cut.items()}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(a.out / "faults.npz", direct_positions=D, kl=KL, lpd=LPD, rank=RANK, surprisal=SUR, code=code,
                        **{f"fault_{k}": np.array(v) for k, v in recs.items()})
    print(json.dumps({k: v for k, v in summary.items() if k != "groups"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
