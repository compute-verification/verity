"""Experiment 2 and 3: the weight bits concrete attacks recover per fault, against honest-query extraction.

Bits of W are rate-distortion bits: an estimate of n weights at mean squared error D, against weights of variance s^2,
counts n/2 log2(s^2/D) bits (pooled over a group, so a conservative count), the rate a Gaussian source of that variance
needs at that distortion, and at most the entropy of the weights' stored codes an entry (`code_bits`: an estimate finer
than the code's spacing is an exact read). A symmetry no observation can see is aligned away before the error is taken,
and its degrees of freedom are not counted. Observed logprobs are the served ones: log_softmax of the logits, rounded to bf16 first when
--precision bf16 (an unembedding run in bf16).

Parts (--part):

  final   a fault at the unembedding's input (the final norm's output) at coordinate i, value s*A. The logits become
          z + (s*A - h_i) E[:, i], so the top-k served logprobs, divided by s*A, are the k largest (s = +1) or smallest
          entries of column i of the unembedding E, up to one constant (softmax's shift, aligned away). Per fault: k - 1
          entries. No knowledge of z or h is needed, and no logit bias.
  honest  the honest baseline with the same observables: Carlini et al. 2024's SVD attack. Full logit vectors of n
          honest positions, rounded as served and centred (softmax's shift), give E up to an invertible d x d map G (and
          a constant per column); aligning G by least squares against the truth leaves the error, over V*d - d^2 - d
          degrees of freedom. A top-k API with logit bias reads a full vector in ceil((V - 1)/(k - 1)) queries, so the
          honest rate is bits / (n * that).
  rows    a fault at layer l's MLP hidden unit j (--row-site mlph; attnh: its attention context), value v. The
          residual stream at that position gains (v - a_j) w_j, w_j the unit's row of the down (output) projection; at
          large |v| it dominates every later layer's addition, and the final norm
          reads its direction: the logits become E (g * norm(w_j) + b), g and b the final norm's weights. A decoder that
          knows E, g and b (from the final-site attack, peeling from the top) reads norm(w_j) from m logits (logit bias,
          k - 1 tokens a fault, F = ceil(m/(k-1)) faults) by least squares. Its error against norm(w_j) includes what
          the honest layers after l add, so this part shows where depth costs the attack and where it does not.

  uv run --extra torch-cpu --with tokenizers python benchmarks/leverage/extract.py --model gpt2 --part final --out "$OUT"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lm import LM, MODELS, Cache, Fault, served_logprobs
from prompts import PROMPTS

LOGPROB_BITS = 32  # a served logprob, as the fp32 float an OpenAI-style API returns


def rd_bits(n: float, ratio: float) -> float:
    """n/2 log2(1/ratio), ratio = MSE / variance, at least 0."""
    return max(0.0, 0.5 * n * math.log2(1.0 / max(ratio, 1e-300)))


def code_bits(w: torch.Tensor) -> float:
    """The entropy of the empirical distribution of w's stored codes: the most an exact read of one entry gives, the cap
    on its rate-distortion bits (an estimate below the code's spacing is the code itself)."""
    _, counts = torch.unique(w.float().contiguous().view(torch.int32), return_counts=True)
    p = counts.double() / counts.sum()
    return float(-(p * p.log2()).sum())


def prefix(lm: LM, tok, text: str) -> tuple[Cache, torch.Tensor, int]:
    ids = torch.tensor([tok.encode(text).ids])
    t0 = ids.shape[1] - 1
    pre = Cache()
    lm.forward(ids[:, :t0], pre)
    return pre, ids[:, t0:], t0


def faulted_logits(lm: LM, pre: Cache, last: torch.Tensor, t0: int, rows: list, taps: dict | None = None):
    f = Fault.of(t0, rows)
    return lm.forward(last.expand(len(rows), 1), pre.expand(len(rows)), fault=f, taps=taps)[:, -1]


def part_final(lm: LM, tok, a) -> dict:
    E = lm.E.float()
    V, d = E.shape
    var = E.var(0)
    cols = torch.randperm(d, generator=torch.Generator().manual_seed(a.seed))[:a.columns].tolist()
    out = []
    for pi in range(a.prompts):
        pre, last, t0 = prefix(lm, tok, PROMPTS[pi])
        for amp in a.amps:
            for sign in (1.0, -1.0):
                for c0 in range(0, len(cols), a.batch):
                    cs = cols[c0:c0 + a.batch]
                    z = faulted_logits(lm, pre, last, t0, [("final", lm.spec.layers, i, sign * amp) for i in cs])
                    for prec in a.precisions:
                        lp = served_logprobs(z, prec)
                        for k in a.ks:
                            top = lp.topk(k, -1)
                            est = top.values.double() / float(torch.tensor(sign * amp).to(torch.bfloat16))
                            tru = E.double()[top.indices, torch.tensor(cs)[:, None]]
                            err = est - tru
                            err = err - err.mean(1, keepdim=True)
                            ratio = (err.pow(2).sum(1) / max(k - 1, 1)) / var.double()[cs]
                            for b, i in enumerate(cs):
                                out.append((pi, amp, sign, prec, k, i, float(ratio[b])))
    groups = {}
    for pi, amp, sign, prec, k, i, r in out:
        groups.setdefault((amp, prec, k), []).append(r)
    cap = code_bits(lm.E)
    rows = []
    for (amp, prec, k), rs in sorted(groups.items()):
        r = sum(rs) / len(rs)
        rd = rd_bits(k - 1, r)
        bits = min(rd, (k - 1) * cap)
        obs = k * (math.log2(V) + LOGPROB_BITS)
        rows.append({"amp": amp, "precision": prec, "k": k, "faults": len(rs), "mse_over_var": r,
                     "rd_bits_per_fault": rd, "capped": rd > bits, "bits_per_fault": bits,
                     "bits_per_entry": bits / max(k - 1, 1),
                     "observed_bits_per_fault": obs, "bits_per_observed_bit": bits / obs})
    return {"part": "final", "columns": len(cols), "code_bits": cap, "rows": rows}


def honest_states(lm: LM, tok, prompts: int, new: int) -> torch.Tensor:
    """Final-norm outputs at every position of each prompt and its honest greedy continuation."""
    hs = []
    for text in PROMPTS[:prompts]:
        c = Cache()
        taps = {("final", lm.spec.layers): None}
        x = torch.tensor([tok.encode(text).ids])
        z = lm.forward(x, c, taps=taps)
        hs.append(taps[("final", lm.spec.layers)][0])
        y = z[:, -1].argmax(-1, keepdim=True)
        for _ in range(new):
            taps = {("final", lm.spec.layers): None}
            z = lm.forward(y, c, taps=taps)
            hs.append(taps[("final", lm.spec.layers)][0])
            y = z[:, -1].argmax(-1, keepdim=True)
    return torch.cat(hs).float()


def part_honest(lm: LM, tok, a) -> dict:
    E = lm.E.float()
    V, d = E.shape
    H = honest_states(lm, tok, a.prompts, a.new)
    ns = sorted({min(int(f * d), H.shape[0]) for f in a.n_over_d})
    Ec = (E - E.mean(0)).double()
    _, _, Q = torch.linalg.svd(Ec, full_matrices=False)
    Ep = Ec @ Q.T
    var = Ep.var(0)
    per_dir = V - d - 1
    cap = code_bits(lm.E)
    rows = []
    for prec in a.precisions:
        Z = H @ E.T
        if prec == "bf16":
            Z = Z.to(torch.bfloat16).float()
        Zc = Z - Z.mean(1, keepdim=True)
        for n in ns:
            _, S, Vh = torch.linalg.svd(Zc[:n].double(), full_matrices=False)
            Eh = Vh[:d].T * S[:d]
            M = torch.linalg.lstsq(Eh, Ep).solution
            ratio = (Ep - Eh @ M).pow(2).mean(0) / var
            bits = min(sum(rd_bits(per_dir, float(x)) for x in ratio), per_dir * d * cap)
            spec = (S[max(d - 4, 0):min(d + 4, n)] / S[0]).tolist()
            for k in a.ks:
                q = n * math.ceil((V - 1) / (k - 1))
                obs = q * k * (math.log2(V) + LOGPROB_BITS)
                rows.append({"precision": prec, "n": n, "k": k, "queries": q,
                             "mse_over_var": float(ratio.mean()), "median_mse_over_var": float(ratio.median()),
                             "bits_total": bits, "bits_per_query": bits / q, "bits_per_observed_bit": bits / obs,
                             "sv_around_d_over_s0": spec})
    return {"part": "honest", "states": int(H.shape[0]), "rows": rows}


def target_row(lm: LM, site: str, l: int, j: int) -> torch.Tensor:
    """The residual-stream direction unit j of site feeds: a row of the projection after it, or for the residual stream
    the unit vector itself, which makes the readout the final-site attack and leaves only its contamination to read."""
    if site == "resid":
        return torch.eye(lm.spec.d)[j]
    w = target_matrix(lm, site, l)
    return (w[j] if lm.spec.arch == "gpt2" else w[:, j]).float()


def target_matrix(lm: LM, site: str, l: int) -> torch.Tensor:
    if lm.spec.arch == "gpt2":
        return lm.W[f"h.{l}.{'mlp' if site == 'mlph' else 'attn'}.c_proj.weight"]
    return lm.W[f"layers.{l}.{'mlp.down_proj' if site == 'mlph' else 'self_attn.o_proj'}.weight"]


def final_norm_params(lm: LM) -> tuple[torch.Tensor, torch.Tensor]:
    if lm.spec.arch == "gpt2":
        return lm.W["ln_f.weight"].float(), lm.W["ln_f.bias"].float()
    return lm.W["norm.weight"].float(), torch.zeros(lm.spec.d)


def norm_dir(lm: LM, w: torch.Tensor) -> torch.Tensor:
    """What the final norm makes of a residual stream dominated by w: LayerNorm centres and scales, RMSNorm scales."""
    if lm.spec.arch == "gpt2":
        w = w - w.mean()
        return w / w.pow(2).mean().sqrt()
    return w / w.pow(2).mean().sqrt()


def part_rows(lm: LM, tok, a) -> dict:
    E = lm.E.float()
    V, d = E.shape
    g, b = final_norm_params(lm)
    lost = 2 if lm.spec.arch == "gpt2" else 1
    gen = torch.Generator().manual_seed(a.seed)
    layers = sorted({l if l >= 0 else lm.spec.layers + l for l in a.layers})
    site = a.row_site
    width = {"mlph": lm.spec.ff, "attnh": lm.spec.heads * lm.spec.head_dim, "resid": d}[site]
    reads = [int(r * d) for r in a.read_over_d]
    out = []
    pre, last, t0 = prefix(lm, tok, PROMPTS[0])
    taps0 = {(site, l): None for l in layers}
    lm.forward(last, pre.expand(1), taps=taps0)
    for l in layers:
        units = torch.randperm(width, generator=gen)[:a.units].tolist()
        rms = float(taps0[(site, l)][0, 0].pow(2).mean().sqrt())
        for v_spec in a.values:
            v = float(v_spec[:-3]) * rms if v_spec.endswith("rms") else float(v_spec)
            taps = {("final", lm.spec.layers): None}
            z = faulted_logits(lm, pre, last, t0, [(site, l, j, v) for j in units], taps=taps)
            h = taps[("final", lm.spec.layers)][:, 0].float()
            for prec in a.precisions:
                zs = z.float().to(torch.bfloat16).float() if prec == "bf16" else z.float()
                for bi, j in enumerate(units):
                    tru = norm_dir(lm, target_row(lm, site, l, j))
                    s = 1.0 if v - float(taps0[(site, l)][0, 0, j]) > 0 else -1.0
                    best = s * (h[bi] - b) / g
                    best = norm_dir(lm, best)
                    r_c = float((best - tru).pow(2).mean())
                    for m in reads:
                        r, mi = read_row(E, g, b, zs[bi], torch.randperm(V, generator=gen)[:m], prec, s, tru, lm)
                        out.append((l, v_spec, prec, m, j, r_c, r, mi))
    rows = []
    groups = {}
    for l, v_spec, prec, m, j, r_c, r, mi in out:
        groups.setdefault((l, v_spec, prec, m), []).append((r_c, r, mi))
    caps = {l: 32.0 if site == "resid" else code_bits(target_matrix(lm, site, l)) for l in layers}
    for (l, v_spec, prec, m), rs in sorted(groups.items()):
        rc, r, mi = (sum(x[i] for x in rs) / len(rs) for i in range(3))
        rd = rd_bits(d - lost, r)
        bits = min(rd, (d - lost) * caps[l])
        for k in a.ks:
            F = math.ceil(m / (k - 1))
            rows.append({"site": site, "layer": l, "value": v_spec, "precision": prec, "k": k, "units": len(rs),
                         "read_tokens": m, "faults_per_row": F, "contamination_mse": rc, "mse": r,
                         "code_bits": caps[l], "rd_bits_per_row": rd, "capped": rd > bits,
                         "bits_per_row": bits, "bits_per_fault": bits / F, "mi_bits_per_row": mi,
                         "mi_bits_per_fault": mi / F,
                         "bits_per_observed_bit": bits / (F * k * (math.log2(V) + LOGPROB_BITS))})
    return {"part": "rows", "rows": rows}


def ulp2(z: torch.Tensor, precision: str) -> torch.Tensor:
    """The variance of rounding z to the served precision, ulp^2 / 12."""
    mant = 7 if precision == "bf16" else 23
    e = torch.floor(torch.log2(z.abs().clamp_min(1e-30)))
    return (2.0 ** (e - mant)).pow(2) / 12


def read_row(E, g, b, z, toks, precision, s, tru, lm) -> tuple[float, float]:
    """MMSE estimate of the normalized row x from the served logits z[toks] = E[toks] (g * s x + b) + c + rounding, the
    constant c (softmax's shift) unknown, x's prior N(0, I); returns its error against `tru` and the Gaussian
    information the read carries about x, 1/2 log2 det of the prior over the posterior covariance."""
    Et = E[toks].double()
    A = torch.cat([Et * (s * g.double()), torch.ones(len(toks), 1, dtype=torch.float64)], 1)
    y = z[toks].double() - Et @ b.double()
    w = 1.0 / ulp2(z[toks].double(), precision)
    P = (A * w[:, None]).T @ A
    P[:-1, :-1] += torch.eye(A.shape[1] - 1, dtype=torch.float64)
    P[-1, -1] += 1e-12
    C = torch.linalg.inv(P)
    xh = C @ (A.T @ (w * y))
    est = norm_dir(lm, xh[:-1].float())
    r = float((est - tru).pow(2).mean())
    mi = -0.5 * float(torch.linalg.slogdet(C[:-1, :-1])[1]) / math.log(2)
    return r, mi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--part", required=True, choices=("final", "honest", "rows"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--new", type=int, default=96, help="honest: continuation tokens per prompt")
    ap.add_argument("--columns", type=int, default=256, help="final: columns of E attacked")
    ap.add_argument("--amps", type=float, nargs="+", default=[2.0 ** 6, 2.0 ** 10, 2.0 ** 14, 2.0 ** 20, 2.0 ** 40])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 20])
    ap.add_argument("--precisions", nargs="+", default=["bf16", "fp32"])
    ap.add_argument("--n-over-d", type=float, nargs="+", default=[1.1, 1.5, 2.0])
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(-12, 0)))
    ap.add_argument("--units", type=int, default=16, help="rows: hidden units per layer")
    ap.add_argument("--row-site", default="mlph", choices=("mlph", "attnh", "resid"),
                    help="rows: the MLP's hidden units (down-projection rows), the attention context (output-projection "
                         "rows) or the residual stream (contamination only)")
    ap.add_argument("--values", nargs="+", default=["1e6", "1e3", "64rms", "8rms"])
    ap.add_argument("--read-over-d", type=float, nargs="+", default=[1.25, 4.0], help="rows: logits read per row, over d")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    tok = lm.tokenizer()
    t = time.time()
    res = {"final": part_final, "honest": part_honest, "rows": part_rows}[a.part](lm, tok, a)
    res.update({"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": lm.spec.vocab, "d": lm.spec.d,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "seconds": time.time() - t})
    (a.out / f"extract-{a.part}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
