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
          entries. No knowledge of z or h is needed, and no logit bias. Under --boxes the amplitude is coordinate i's
          box, or what one FP8 code reaches at its token's own group scale (--box-amps). With --differential the
          decoder picks k tokens by logit bias and also reads them from an honest query on the same prompt: the
          difference over (s*A - h_i) reads column i at any amplitude the logits' rounding resolves, not only where
          the fault dominates the honest logits (the scale s*A - h_i is one number per column and prompt, aligned).
  honest  the honest baseline with the same observables: Carlini et al. 2024's SVD attack. Full logit vectors of n
          honest positions, rounded as served and centred (softmax's shift), give E up to an invertible d x d map G (and
          a constant per column); aligning G by least squares against the truth leaves the error, over V*d - d^2 - d
          degrees of freedom. A top-k API with logit bias reads a full vector in ceil((V - 1)/(k - 1)) queries, so the
          honest rate is bits / (n * that).
  rows    a fault at layer l's MLP hidden unit j (--row-site mlph; attnh: its attention context; resid: the residual
          stream itself, w_j = e_j), value v. The
          residual stream at that position gains (v - a_j) w_j, w_j the unit's row of the down (output) projection; at
          large |v| it dominates every later layer's addition, and the final norm
          reads its direction: the logits become E (g * norm(w_j) + b), g and b the final norm's weights. A decoder that
          knows E, g and b (from the final-site attack, peeling from the top) reads norm(w_j) from m logits (logit bias,
          k - 1 tokens a fault, F = ceil(m/(k-1)) faults) by an MMSE fit weighted by the served rounding's error. Its
          error against norm(w_j) includes what the honest layers after l add, so this part shows where depth costs
          the attack and where it does not. Under --boxes the values are in range (`fault_values`), a joint fault's
          target is the sum of what its group's units add, and --pick widest attacks the units with the widest boxes.
          With --differential the decoder also reads the honest final state from an honest query on the same prompt
          and tokens, and reads the fault's direction from the difference (`differential_error`), which removes the
          honest state and leaves only what the layers after l add. That costs one more unseen degree of freedom.

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
from lm import E4M3_MAX, GROUP, LM, MODELS, Boxes, Cache, Fault, position_class, served_logprobs, ue8m0_scale
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


FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_ENTRY_BITS = 4 + 8 / 16  # an E2M1 code and its share of a 16-entry block's E4M3 scale


def nvfp4(w: torch.Tensor, block: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """w fake-quantized to NVFP4: E2M1 codes in blocks of `block` along the last dimension, each block's scale its amax
    / 6 in FP8 E4M3 under one fp32 scale for the tensor; and the sorted grid of values an entry can take."""
    x = w.float().reshape(-1, block)
    g = float(x.abs().max()) / (6 * 448)
    s = ((x.abs().amax(1, keepdim=True) / 6) / g).to(torch.float8_e4m3fn).float()
    s = torch.where(s > 0, s, torch.ones_like(s)) * g
    q = x / s
    levels = torch.tensor(FP4_LEVELS)
    out = levels[torch.bucketize(q.abs(), (levels[1:] + levels[:-1]) / 2)] * q.sign() * s
    e4m3 = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    e4m3 = e4m3[torch.isfinite(e4m3) & (e4m3 > 0)]
    pos = (levels[1:, None] * (e4m3 * g)[None]).flatten()
    return out.reshape(w.shape), torch.cat([-pos, torch.zeros(1), pos]).unique()


def snap(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    x = x.float().contiguous()
    i = torch.searchsorted(grid, x).clamp(1, len(grid) - 1)
    lo, hi = grid[i - 1], grid[i]
    return torch.where((x - lo).abs() <= (hi - x).abs(), lo, hi)


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
    grid = getattr(lm, "fp4_grid", None)
    gen = torch.Generator().manual_seed(a.seed)
    cols = torch.randperm(d, generator=gen)[:a.columns].tolist()
    specs = list(a.amps) if lm.boxes is None else list(a.box_amps)
    L = lm.spec.layers
    out = []
    for pi in range(a.prompts):
        pre, last, t0 = prefix(lm, tok, PROMPTS[pi])
        taps = {("final", L): None}
        z_h = lm.forward(last, pre.expand(1), taps=taps)[:, -1]
        h = taps[("final", L)][0, 0].float()
        if lm.boxes is not None:
            hi = lm.boxes.hi[("final", L)][int(position_class(torch.tensor(t0)))].float()
        for amp in specs:
            for sign in (1.0, -1.0):
                for c0 in range(0, len(cols), a.batch):
                    cs = cols[c0:c0 + a.batch]
                    if amp == "box":
                        amps = [float(hi[i]) for i in cs]
                    elif amp == "own":
                        amps = [fault_values("own", i, h, 0.0, hi, None)[1][0] for i in cs]
                    else:
                        amps = [float(amp)] * len(cs)
                    z = faulted_logits(lm, pre, last, t0, [("final", L, i, sign * A) for i, A in zip(cs, amps)])
                    sa = (sign * torch.tensor(amps)).to(torch.bfloat16).double()[:, None]
                    for prec in a.precisions:
                        lp = served_logprobs(z, prec).double()
                        lp_h = served_logprobs(z_h, prec).double()
                        for k in a.ks:
                            if a.differential:
                                toks = torch.stack([torch.randperm(V, generator=gen)[:k] for _ in cs])
                                est = (lp.gather(1, toks) - lp_h[0, toks]) / (sa - h.double()[cs][:, None])
                            else:
                                top = lp.topk(k, -1)
                                toks, est = top.indices, top.values / sa
                            tru = E.double()[toks, torch.tensor(cs)[:, None]]
                            err = est - tru
                            err = err - err.mean(1, keepdim=True)
                            ratio = (err.pow(2).sum(1) / max(k - 1, 1)) / var.double()[cs]
                            hit = torch.full((len(cs),), float("nan"))
                            if grid is not None and k > 1:
                                rel = est - est[:, :1] + tru[:, :1]
                                hit = (snap(rel, grid) == tru.float())[:, 1:].float().mean(1)
                            for b, i in enumerate(cs):
                                out.append((pi, amp, sign, prec, k, i, float(ratio[b]), float(hit[b])))
    groups = {}
    for pi, amp, sign, prec, k, i, r, h in out:
        groups.setdefault((amp, prec, k), []).append((r, h))
    cap = code_bits(lm.E)
    rows = []
    for (amp, prec, k), rs in sorted(groups.items()):
        r = sum(x for x, _ in rs) / len(rs)
        rd = rd_bits(k - 1, r)
        bits = min(rd, (k - 1) * cap)
        obs = k * (math.log2(V) + LOGPROB_BITS)
        row = {"amp": amp, "precision": prec, "k": k, "faults": len(rs), "mse_over_var": r,
               "rd_bits_per_fault": rd, "capped": rd > bits, "bits_per_fault": bits,
               "bits_per_entry": bits / max(k - 1, 1),
               "observed_bits_per_fault": obs, "bits_per_observed_bit": bits / obs}
        if grid is not None:
            ex = sum(h for _, h in rs) / len(rs)
            row.update({"exact_fraction": ex, "fp4_bits_per_fault": ex * (k - 1) * FP4_ENTRY_BITS})
        rows.append(row)
    return {"part": "final", "columns": len(cols), "code_bits": cap, "fp4": grid is not None, "rows": rows}


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


def unit_rows(lm: LM, site: str, l: int) -> torch.Tensor:
    """(width, d): row k is the residual-stream direction unit k of site feeds per unit of its value, a row of the
    projection after it, or for the residual stream the unit vector itself, which makes the readout the final-site
    attack and leaves only its contamination to read."""
    if site == "resid":
        return torch.eye(lm.spec.d)
    w = target_matrix(lm, site, l).float()
    return w if lm.spec.arch == "gpt2" else w.T


def fault_values(spec: str, j: int, x: torch.Tensor, rms: float, hi: torch.Tensor | None,
                 gen: torch.Generator) -> tuple[list[int], list[float]]:
    """The coordinates and values of the fault `spec` names at unit j, x the site's honest value there: a number, a
    multiple of the RMS ("8rms"), or under --boxes an in-range fault (reach.py's kinds): "box", "own", "scale", "codes"
    (the last three on unit j's group)."""
    g = list(range(j // GROUP * GROUP, (j // GROUP + 1) * GROUP))
    if spec.endswith("rms"):
        return [j], [float(spec[:-3]) * rms]
    if spec == "box":
        return [j], [float(hi[j])]
    if spec == "own":
        return [j], [min(E4M3_MAX * float(ue8m0_scale(x[g].abs().max())), float(hi[j]))]
    if spec == "scale":
        f = float(hi[j]) / E4M3_MAX / float(ue8m0_scale(x[g].abs().max()))
        return g, (x[g] * f).tolist()
    if spec == "codes":
        sign = torch.randint(0, 2, (GROUP,), generator=gen).float() * 2 - 1
        return g, (sign * hi[j]).tolist()
    return [j], [float(spec)]


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


def peel(lm: LM, tok, pre: Cache, last: torch.Tensor, t0: int, h0: torch.Tensor, T: torch.Tensor, a) -> dict:
    """What a decoder that doesn't know E, g or b uses instead: per served precision, estimates of E[T] diag(g) and
    E[T] b, each up to a constant a column (softmax's shift absorbs it), read from served logprobs on the tokens T by
    logit bias, T[0] the reference every query also shows. Taking them costs 2 d faults (saturate) or 4 d (box, own)
    for every k - 1 tokens of T, once for every row read after.

    saturate  paired faults of ±2^30 at coordinate i of the residual stream after the last layer saturate the final
              norm, whose output becomes g * (±s u_i) + b with u_i = e_i (RMSNorm; s = sqrt d) or e_i - 1/d (LayerNorm;
              s = d / sqrt(d - 1)). The pair's half-difference over s reads column i of E diag(g), centred over the
              coordinates under LayerNorm, which is all a centred state sees; the half-sums read E b.
    box, own  paired final-site faults ±v at coordinate i (v its box, or one FP8 code at its token's own group scale)
              read column i of E from their half-difference over v. Paired faults ±δ at coordinate i of the residual
              stream after the last layer (δ half the smaller of its box and the boxes' median, well inside the
              norm's linear range) move the final state by 2 δ g_i / σ along e_i, σ the stream's scale, which the
              same estimate reads: g up to one scale, signs included. Then b, under LayerNorm: each honest final
              state h read through the estimate has |(h - b) / g|^2 the same, linear in b / g^2 given g; the fit
              leaves b free along g, which only shifts a state along 1, which LayerNorm's centring removes. The
              scale makes those states' mean square 1."""
    E = lm.E.float()
    V, d = E.shape
    L = lm.spec.layers
    ln = lm.spec.arch == "gpt2"
    g, b = final_norm_params(lm)
    if a.peel == "saturate":
        if lm.boxes is not None:
            raise SystemExit("--peel saturate needs unclamped faults: drop --boxes, or peel with box or own")
        site, layer, s = "resid", L - 1, (d / math.sqrt(d - 1) if ln else math.sqrt(d))
        amps = torch.full((d,), 2.0 ** 30)
    else:
        if lm.boxes is None:
            raise SystemExit(f"--peel {a.peel} reads at the boxes' amplitudes: give --boxes")
        site, layer, s = "final", L, 1.0
        pc = int(position_class(torch.tensor(t0)))
        hi = lm.boxes.hi[("final", L)][pc].float()
        amps = torch.tensor([float(hi[i]) if a.peel == "box" else fault_values("own", i, h0, 0.0, hi, None)[1][0]
                             for i in range(d)])
        hr = lm.boxes.hi[("resid", L - 1)][pc].float()
        deltas = (torch.minimum(hr, hr.median()) / 2).to(torch.bfloat16).double()
    amps = amps.to(torch.bfloat16).double()

    def pairs(site: str, layer: int, vals: torch.Tensor) -> dict:
        n, obs = max(a.batch // 2, 1), {prec: [] for prec in a.precisions}
        for c0 in range(0, d, n):
            cs = list(range(c0, min(d, c0 + n)))
            z = faulted_logits(lm, pre, last, t0, [(site, layer, i, sg * float(vals[i])) for sg in (1.0, -1.0) for i in cs])
            for prec in a.precisions:
                lp = served_logprobs(z, prec)[:, T].double()
                obs[prec].append((lp - lp[:, :1]).view(2, len(cs), len(T)))
        return {prec: torch.cat(o, 1) for prec, o in obs.items()}

    obs = pairs(site, layer, amps)
    obr = pairs("resid", L - 1, deltas) if a.peel != "saturate" else None
    W_true = E[T].double() * g.double()
    if ln and a.peel == "saturate":
        W_true = W_true - W_true.mean(1, keepdim=True)
    W_true = W_true - W_true[:1]
    H = honest_states(lm, tok, len(PROMPTS), a.new).double() if a.peel != "saturate" else None
    res = {}
    for prec in a.precisions:
        o = obs[prec]
        col = ((o[0] - o[1]) / (2 * s * (1.0 if a.peel == "saturate" else amps[:, None]))).T
        g_hat = b_hat = None
        if a.peel == "saturate":
            W, beta = col, ((o[0] + o[1]) / 2).mean(0)
        else:
            A = torch.cat([col, torch.ones(len(T), 1, dtype=torch.float64)], 1)
            dh = torch.linalg.lstsq(A, (obr[prec][0] - obr[prec][1]).T).solution[:d]
            g_hat = dh.diagonal() / (2 * deltas)
            Z = H @ E[T].double().T
            if prec == "bf16":
                Z = Z.float().to(torch.bfloat16).double()
            Hh = torch.linalg.lstsq(A, (Z - Z[:, :1]).T).solution[:d].T
            b_hat = torch.zeros(d, dtype=torch.float64)
            if ln:
                F = torch.cat([2 * Hh, torch.ones(len(Hh), 1, dtype=torch.float64)], 1)
                b_hat = torch.linalg.lstsq(F, (Hh / g_hat).pow(2).sum(1, keepdim=True)).solution[:d, 0] * g_hat.pow(2)
            xs = (Hh - b_hat) / g_hat
            if ln:
                xs = xs - xs.mean(1, keepdim=True)
            g_hat = g_hat * xs.pow(2).mean().sqrt()
            W, beta = col * g_hat, col @ b_hat
        b_err = None
        if b_hat is not None and ln:
            e, gd = b_hat - b.double(), g.double()
            b_err = float((e - (e @ gd) / (gd @ gd) * gd).pow(2).mean() / b.double().pow(2).mean())
        res[prec] = {"W": W, "beta": beta, "W_mse_over_var": scaled_mse(W, W_true),
                     "g_mse_over_var": None if g_hat is None else scaled_mse(g_hat, g.double()),
                     "b_mse_over_var_off_g": b_err}
    if max(e["W_mse_over_var"] for e in res.values()) > 0.5:
        raise SystemExit(f"--peel {a.peel}: E diag(g) read no better than its variance, the peel is broken: "
                         + peel_summary(res))
    return res


def peel_summary(res: dict) -> str:
    return ", ".join(f"{p} " + ", ".join(f"{n} {v:.3g}" for n, v in e.items() if n.endswith(("var", "off_g"))
                                          and v is not None) for p, e in res.items())


def scaled_mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """x's error against y over y's variance, after x's best scale: the read normalizes its state, so a peel's
    estimates count only up to one."""
    return float((x * ((x * y).sum() / (x * x).sum()) - y).pow(2).mean() / y.pow(2).mean())


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
    t_start = time.time()
    pre, last, t0 = prefix(lm, tok, PROMPTS[0])
    taps0 = {(site, l): None for l in layers} | {("final", lm.spec.layers): None}
    z0 = lm.forward(last, pre.expand(1), taps=taps0)[0, -1]
    h0 = taps0[("final", lm.spec.layers)][0, 0].float()
    amax = torch.load(a.boxes)["amax"] if a.boxes is not None else None
    T = est = None
    if a.peel:
        T = torch.randperm(V, generator=gen)[:max(reads)]
        est = peel(lm, tok, pre, last, t0, h0, T, a)
        print(f"peel {a.peel}: {peel_summary(est)}, {time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
    for l in layers:
        x0 = taps0[(site, l)][0, 0].float()
        rms = float(x0.pow(2).mean().sqrt())
        hi = lm.boxes.hi[(site, l)][int(position_class(torch.tensor(t0)))].float() if lm.boxes is not None else None
        if a.pick == "widest":
            by_amax = amax[f"{site}:{l}"][1].argsort(descending=True, stable=True)
            units = by_amax[hi[by_amax].argsort(descending=True, stable=True)][:a.units].tolist()
        else:
            units = torch.randperm(width, generator=gen)[:a.units].tolist()
        U = unit_rows(lm, site, l)
        for v_spec in a.values:
            faults, deltas = [], []
            for j in units:
                cs, vs = fault_values(v_spec, j, x0, rms, hi, gen)
                faults.append((site, l, cs, vs))
                deltas.append((torch.tensor(vs) - x0[cs]) @ U[cs])
            taps = {("final", lm.spec.layers): None}
            z = faulted_logits(lm, pre, last, t0, faults, taps=taps)
            h = taps[("final", lm.spec.layers)][:, 0].float()
            print(f"rows {site} layer {l} value {v_spec}: {int((~torch.isfinite(z)).sum())} non-finite logits, "
                  f"{time.time() - t_start:.0f} s", file=sys.stderr, flush=True)
            for prec in a.precisions:
                zs = z.float().to(torch.bfloat16).float() if prec == "bf16" else z.float()
                zs0 = z0.float().to(torch.bfloat16).float() if prec == "bf16" else z0.float()
                for bi, j in enumerate(units):
                    tru = norm_dir(lm, deltas[bi])
                    best = norm_dir(lm, (h[bi] - b) / g)
                    if a.differential:
                        r_c = differential_error(best, norm_dir(lm, (h0 - b) / g), tru)
                    else:
                        r_c = float((best - tru).pow(2).mean())
                    for m in reads:
                        if est is None:
                            toks = torch.randperm(V, generator=gen)[:m]
                            head = (E[toks], g, E[toks] @ b)
                        else:
                            toks = T[:m]
                            head = (est[prec]["W"][:m], torch.ones(d), est[prec]["beta"][:m])
                        r, mi, xf = read_row(*head, zs[bi][toks], prec, 1.0, tru, lm)
                        if a.differential:
                            r = differential_error(xf, read_row(*head, zs0[toks], prec, 1.0, tru, lm)[2], tru)
                        out.append((l, v_spec, prec, m, j, r_c, r, mi))
    rows = []
    groups = {}
    for l, v_spec, prec, m, j, r_c, r, mi in out:
        groups.setdefault((l, v_spec, prec, m), []).append((r_c, r, mi))
    caps = {l: 32.0 if site == "resid" else code_bits(target_matrix(lm, site, l)) for l in layers}
    lost += int(a.differential)
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
    res = {"part": "rows", "rows": rows}
    if est is not None:
        res["peel"] = {"mode": a.peel, "tokens": len(T),
                       "faults_by_k": {k: (2 if a.peel == "saturate" else 4) * d * math.ceil(len(T) / (k - 1))
                                       for k in a.ks if k > 1},
                       "by_precision": {p: {n: v for n, v in e.items() if n not in ("W", "beta")}
                                        for p, e in est.items()}}
    return res


def ulp2(z: torch.Tensor, precision: str) -> torch.Tensor:
    """The variance of rounding z to the served precision, ulp^2 / 12."""
    mant = 7 if precision == "bf16" else 23
    e = torch.floor(torch.log2(z.abs().clamp_min(1e-30)))
    return (2.0 ** (e - mant)).pow(2) / 12


def differential_error(xf: torch.Tensor, xh: torch.Tensor, tru: torch.Tensor) -> float:
    """The error, over tru's variance, of the fault's direction read as the faulted final state minus the honest one
    at the same prompt. Each is known only up to the norm's scale, so the difference is known only off the faulted
    state's own direction: both sides are projected off it, and the estimate's scale is aligned."""
    u = xf.double() / xf.double().norm()
    est, t = -(xh.double() - (xh.double() @ u) * u), tru.double() - (tru.double() @ u) * u
    alpha = (est @ t) / (est @ est)
    return float((alpha * est - t).pow(2).mean() / t.pow(2).mean())


def read_row(Et, g, Eb, zt, precision, s, tru, lm) -> tuple[float, float, torch.Tensor]:
    """MMSE estimate of the normalized row x from the served logits zt = Et (g * s x) + Eb + c + rounding on the read
    tokens (Et and Eb: the decoder's E[toks] and E[toks] b), the constant c (softmax's shift) unknown, x's prior
    N(0, I); returns its error against `tru`, the Gaussian information the read carries about x, 1/2 log2 det of the
    prior over the posterior covariance, and the estimate."""
    A = torch.cat([Et.double() * (s * g.double()), torch.ones(len(zt), 1, dtype=torch.float64)], 1)
    y = zt.double() - Eb.double()
    w = 1.0 / ulp2(zt.double(), precision)
    P = (A * w[:, None]).T @ A
    P[:-1, :-1] += torch.eye(A.shape[1] - 1, dtype=torch.float64)
    P[-1, -1] += 1e-12
    C = torch.linalg.inv(P)
    # Not C @ ...: a centred head (LayerNorm's, as a peel reads it) leaves P's condition near 1e14 at fp32's weights,
    # where the explicit inverse loses the estimate and a solve doesn't.
    xh = torch.linalg.solve(P, A.T @ (w * y))
    est = norm_dir(lm, xh[:-1].float())
    r = float((est - tru).pow(2).mean())
    mi = -0.5 * float(torch.linalg.slogdet(C[:-1, :-1])[1]) / math.log(2)
    return r, mi, est


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=sorted(MODELS))
    ap.add_argument("--part", required=True, choices=("final", "honest", "rows"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--new", type=int, default=96, help="honest: continuation tokens per prompt")
    ap.add_argument("--columns", type=int, default=256, help="final: columns of E attacked")
    ap.add_argument("--amps", type=float, nargs="+", default=[2.0 ** 6, 2.0 ** 10, 2.0 ** 14, 2.0 ** 20, 2.0 ** 40])
    ap.add_argument("--box-amps", nargs="+", default=["box", "own"],
                    help="final, under --boxes: each column's amplitude is its coordinate's box, or the most one FP8 "
                         "code reaches at its token's own group scale")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 5, 20])
    ap.add_argument("--precisions", nargs="+", default=["bf16", "fp32"])
    ap.add_argument("--fp4", action="store_true",
                    help="serve the unembedding (tied: and the embedding) fake-quantized to NVFP4; final: count the "
                         "entries a fault reads exactly, snapped to NVFP4's grid")
    ap.add_argument("--n-over-d", type=float, nargs="+", default=[1.1, 1.5, 2.0])
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(-12, 0)))
    ap.add_argument("--units", type=int, default=16, help="rows: hidden units per layer")
    ap.add_argument("--row-site", default="mlph", choices=("mlph", "attnh", "resid"),
                    help="rows: the MLP's hidden units (down-projection rows), the attention context (output-projection "
                         "rows) or the residual stream (contamination only)")
    ap.add_argument("--values", nargs="+", default=["1e6", "1e3", "64rms", "8rms"],
                    help="rows: a value, a multiple of the site's RMS, or under --boxes box, own, scale or codes")
    ap.add_argument("--boxes", type=Path, help="clamp every read to clamp.py's calibrated boxes.pt")
    ap.add_argument("--pick", default="random", choices=("random", "widest"),
                    help="rows: random units, or those with the widest boxes (then the largest calibrated values)")
    ap.add_argument("--read-over-d", type=float, nargs="+", default=[1.25, 4.0], help="rows: logits read per row, over d")
    ap.add_argument("--differential", action="store_true",
                    help="final and rows: the decoder also reads the same tokens' logprobs from an honest query on "
                         "the same prompt and reads the fault from the difference; final then picks its k tokens by "
                         "logit bias, at random, instead of reading the top k")
    ap.add_argument("--peel", choices=("saturate", "box", "own"),
                    help="rows: the decoder estimates E, g and b from faults on the last layer instead of knowing them "
                         "(`peel`): saturate, unclamped; box or own, at the final site under --boxes")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model)
    if a.boxes is not None:
        lm.boxes = Boxes.load(a.boxes)
    if a.fp4:
        q, lm.fp4_grid = nvfp4(lm.E)
        lm.E.copy_(q)
    tok = lm.tokenizer()
    t = time.time()
    res = {"final": part_final, "honest": part_honest, "rows": part_rows}[a.part](lm, tok, a)
    res.update({"model": a.model, "revision": MODELS[a.model]["revision"], "vocab": lm.spec.vocab, "d": lm.spec.d,
                "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                "seconds": time.time() - t})
    tag = ("-fp4" if a.fp4 else "") + ("-clamped" if a.boxes is not None else "") + ("-diff" if a.differential else "")
    tag += f"-peel-{a.peel}" if a.peel else ""
    (a.out / f"extract-{a.part}{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
