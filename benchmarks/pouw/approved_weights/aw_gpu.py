#!/usr/bin/env python3
"""Approved-weights tradeoff on the GPU: loss against the verification cost a property removes, at Qwen2.5-7B scale.

For each rung of the property ladder (docs/pouw/approved-weights.md), starting from a weight source, this:
  1. projects every transformer linear's weights onto the rung's verification-friendly set (the transform),
  2. optionally recovers with a short fine-tune of the glue parameters (RMSNorm scales and biases; uncredited),
  3. measures WikiText-2 test perplexity (loss),
  4. measures the verification cost the rung leaves: the fraction of the honest chain an adversarial prover can skip on
     the deployed codes, from (near-)duplicate rows, 2:4-sparse slices, and (for the plain rung) planted exact relations.

Weight sources:
  real       Qwen2.5-7B as released
  random     every linear reset to a keyed Gaussian of the row's norm (the strong end of Daniel's spectrum)
  adv        every linear replaced by a relation-packed adversarial matrix (row triples with row3 = row1 + row2 on the
             E4M3 grid), the worst case an open registry would admit

Rungs (--rung):
  bf16       the source unquantized (the reference)
  none       deploy the source as is (per-row E4M3, round to nearest)
  clamp-R    clip each row to R x its RMS
  specfloor-K   raise singular values to sigma_max / K
  mix-L      (1 - L) W + L G ||W|| / ||G||, G keyed
  rot        keyed residual-stream rotation (input side of q/k/v/gate/up, output side of o/down), round to nearest,
             plus a keyed permutation of the MLP intermediate dim: the recommended transform
  rot-sr     the same with keyed stochastic rounding
  random     ignore the source, deploy keyed Gaussian codes
  X@nvfp4    rungs none and rot in NVFP4 (UE4M3 per 16, E2M1), for Pearl-C4's loss (no census: its chain is exact)

The verification cost is the adversary's skippable fraction, measured on the sm_120 FP8 v1 chain (G = 4, 128-atom
groups) at the deployed codes:
  duplicate and near-duplicate rows   a group that agrees with another row's on >= 113 of 128 codes is a derived group:
                                      it costs the differing codes' products (8 units each) instead of 128 MACs
  2:4-compatible 64-slices            run on the sparse tensor cores at half the MACs
This is a lower bound on what the rung leaves; the CPU relation attack (relation-attack.json) covers planted relations,
which only the plain rungs admit. gamma itself is fixed at the cap (0.511% v1); this axis is the assumption's exposure.

Chunking: one (source, rung, recover) variant per chunk, checkpointed by its output JSON. Exit 0 when every requested
variant is done, 99 while any remain, 143 on SIGTERM (preemption).

  python aw_gpu.py --out DIR --variants real:none:0,real:rot:1,random:rot:0 --eval-windows 16 --recover-steps 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PREEMPTED = {"flag": False}
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def on_term(signum, frame):
    PREEMPTED["flag"] = True


def gen(dev, *parts):
    h = hashlib.sha256("|".join(["verity/pouw/approved-weights/gpu/v1", *map(str, parts)]).encode()).digest()
    return torch.Generator(device=dev).manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def e4m3_rn(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)


def e4m3_sr(x, g):
    a = x.abs().clamp(max=448.0).float()
    e = torch.floor(torch.log2(a.clamp_min(2.0 ** -6)))
    ulp = torch.where(a >= 2.0 ** -6, 2.0 ** (e - 3), torch.full_like(a, 2.0 ** -9))
    lo = torch.floor(a / ulp) * ulp
    up = torch.rand(a.shape, generator=g, device=a.device) < (a - lo) / ulp
    return (torch.sign(x) * torch.minimum(lo + up * ulp, torch.tensor(448.0, device=a.device))).to(x.dtype)


def fp8_rows(w, g=None):
    alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    q = e4m3_sr(w * alpha, g) if g is not None else e4m3_rn(w * alpha)
    return q / alpha


_ROT = {}


def rot_matrix(d, dev, seed):
    if (d, str(dev), seed) not in _ROT:
        q, r = torch.linalg.qr(torch.randn(d, d, generator=gen(dev, "rot", d, seed), device=dev, dtype=torch.float32))
        _ROT[(d, str(dev), seed)] = (q * torch.sign(torch.diagonal(r))).to(torch.float32)
    return _ROT[(d, str(dev), seed)]


def nvfp4_rows(w, g=None):
    """NVFP4: per-row FP32 scale to 6 x 448, UE4M3 scale per 16, E2M1 codes (round to nearest, or stochastic)."""
    n, k = w.shape
    grid, mids = E2M1_GRID.to(w.device), E2M1_MIDS.to(w.device)
    gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
    v = (w / gs).reshape(n, k // 16, 16)
    s = e4m3_rn(v.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
    x = (v / s).clamp(-6.0, 6.0)
    a = x.abs()
    if g is None:
        q = grid[torch.bucketize(a, mids)]
    else:
        i = (torch.bucketize(a, grid, right=True) - 1).clamp(0, 7)
        lo, hi = grid[i], grid[(i + 1).clamp(max=7)]
        p = torch.where(hi > lo, (a - lo) / (hi - lo).clamp_min(1e-30), torch.zeros_like(a))
        q = torch.where(torch.rand(a.shape, generator=g, device=a.device) < p, hi, lo)
    return (torch.sign(x) * q * s).reshape(n, k) * gs


def act_noise(x, delta, f_lines, g):
    """The fork's activation noise: rank-32 E F^T with F fixed per linear and E fresh per call, scaled per row to
    delta x the RMS over every 8th position (Pearl's rule); the peel removes it exactly in the useful output."""
    rho = x[:, ::8].pow(2).mean(1, keepdim=True).sqrt()
    e = torch.randn(x.shape[0], f_lines.shape[1], generator=g, device=x.device)
    z = e @ f_lines.T
    z = z / z.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-30)
    return delta * rho * z, rho


def act_quant(x, fmt, f_lines, g):
    """A_eff = Q(alpha (x + n)) / alpha - n: the operand of Pearl-C's useful output U with noise on A only (FP8: delta 1,
    alpha = 448/(s + 4 delta rho); NVFP4: delta 1/4, alpha = 2688/(s + 4 delta rho), UE4M3 scales from the noisy block)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).float()
    delta = 1.0 if fmt == "fp8" else 0.25
    n, rho = act_noise(x2, delta, f_lines, g)
    s = x2.abs().amax(1, keepdim=True)
    top = 448.0 if fmt == "fp8" else 6.0 * 448.0
    alpha = top / (s + 4 * delta * rho).clamp_min(1e-30)
    v = alpha * (x2 + n)
    if fmt == "fp8":
        q = e4m3_rn(v)
    else:
        m, k = v.shape
        vb = v.reshape(m, k // 16, 16)
        sb = e4m3_rn(vb.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
        grid, mids = E2M1_GRID.to(v.device), E2M1_MIDS.to(v.device)
        xb = (vb / sb).clamp(-6.0, 6.0)
        q = (torch.sign(xb) * grid[torch.bucketize(xb.abs(), mids)] * sb).reshape(m, k)
    a = q / alpha - n
    return (x2 + (a - x2).detach()).to(x.dtype).reshape(shape)


def norm_of(n):
    """The RMSNorm whose output feeds linear n (q/k/v: input_layernorm; gate/up: post_attention_layernorm), or None."""
    layer = n.rsplit(".", 2)[0]
    if n.endswith(("q_proj", "k_proj", "v_proj")):
        return layer + ".input_layernorm"
    if n.endswith(("gate_proj", "up_proj")):
        return layer + ".post_attention_layernorm"
    return None


_RD = {}


def rot_dtype(d, dev, seed, dtype):
    if (d, seed, dtype) not in _RD:
        _RD[(d, seed, dtype)] = rot_matrix(d, dev, seed).to(dtype)
    return _RD[(d, seed, dtype)]


def install_act(model, rung, seed, dev):
    """Deploy every linear with the fork's activation path: y = Q_act(x R_in) W_q^T (then R_out^T), W_q the deployed weights
    in the deployed basis. The bf16 rung keeps BF16 weights and activations (the reference)."""
    fmt = "nvfp4" if rung.endswith("@nvfp4") else "fp8"
    base = rung.split("@")[0]
    gam = {}
    if base == "rot":                                   # fold each RMSNorm scale into the linears it feeds, then rotate
        for n, _ in linears(model):
            nm = norm_of(n)
            if nm and nm not in gam:
                gam[nm] = model.get_submodule(nm).weight.detach().float().clone()
        with torch.no_grad():
            for nm in gam:
                model.get_submodule(nm).weight.fill_(1.0)
    for n, m in linears(model):
        m._aw_rin = m._aw_rout = None
        if rung == "bf16":
            m.forward = torch.nn.Linear.forward.__get__(m)
            continue
        w = m._aw_orig.to(dev).float()
        if norm_of(n) in gam:
            w = w * gam[norm_of(n)].to(dev)[None, :]
        q = nvfp4_rows if fmt == "nvfp4" else fp8_rows
        with torch.no_grad():
            if base == "rot" and n.endswith(("o_proj", "down_proj")):
                r = rot_matrix(w.shape[0], dev, seed)
                m.weight.copy_(q(r.T @ w).to(m.weight.dtype))
                m._aw_rout = rot_dtype(r.shape[0], dev, seed, m.weight.dtype)
            elif base == "rot":
                r = rot_matrix(w.shape[1], dev, seed)
                m.weight.copy_(q(w @ r).to(m.weight.dtype))
                m._aw_rin = rot_dtype(r.shape[0], dev, seed, m.weight.dtype)
            else:
                m.weight.copy_(q(w).to(m.weight.dtype))
        k = m.in_features
        m._aw_f = torch.randn(k, 32, generator=gen(dev, "act-F", n, seed), device=dev)
        m._aw_g = gen(dev, "act-E", n, seed)

        def fwd(x, m=m, fmt=fmt):
            if m._aw_rin is not None:
                x = x @ m._aw_rin
            y = F.linear(act_quant(x, fmt, m._aw_f, m._aw_g), m.weight, m.bias)
            if getattr(m, "_aw_scale", None) is not None:
                y = y * m._aw_scale.to(y.dtype) + m._aw_shift.to(y.dtype)
            if m._aw_rout is not None:
                y = y @ m._aw_rout.T
            return y
        m.forward = fwd


def glue_params(model, on):
    """Per-output-channel scale and bias after q/k/v/gate/up (unrotated outputs): glue in every deployed variant."""
    params = []
    for n, m in linears(model):
        if on and n.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            m._aw_scale = torch.nn.Parameter(torch.ones(m.out_features, device=m.weight.device, dtype=torch.float32))
            m._aw_shift = torch.nn.Parameter(torch.zeros(m.out_features, device=m.weight.device, dtype=torch.float32))
            params += [m._aw_scale, m._aw_shift]
        else:
            m._aw_scale = m._aw_shift = None
    return params


def project(w, rung, name, seed):
    dev, dt = w.device, w.dtype
    w = w.float()
    if rung == "bf16":
        return w.to(dt)
    if rung.endswith("@nvfp4"):
        base = rung[:-6]
        if base == "none":
            return nvfp4_rows(w).to(dt)
        if base == "rot":
            if name.endswith(("o_proj", "down_proj")):
                r = rot_matrix(w.shape[0], dev, seed)
                return (r @ nvfp4_rows(r.T @ w)).to(dt)
            r = rot_matrix(w.shape[1], dev, seed)
            return (nvfp4_rows(w @ r) @ r.T).to(dt)
        raise ValueError(rung)
    if rung == "none":
        return fp8_rows(w).to(dt)
    if rung.startswith("clamp-"):
        r = float(rung[6:]) * w.pow(2).mean(1, keepdim=True).sqrt()
        return fp8_rows(w.clamp(-r, r)).to(dt)
    if rung.startswith("specfloor-"):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        return fp8_rows((u * torch.maximum(s, s[0] / float(rung[10:]))) @ vh).to(dt)
    if rung.startswith("mix-"):
        lam = float(rung[4:])
        gm = torch.randn(w.shape, generator=gen(dev, "mix", name, seed), device=dev)
        return fp8_rows((1 - lam) * w + lam * gm * (w.norm() / gm.norm())).to(dt)
    if rung == "random":
        gm = torch.randn(w.shape, generator=gen(dev, "rand-w", name, seed), device=dev)
        return fp8_rows(gm * (w.pow(2).mean(1, keepdim=True).sqrt() / gm.pow(2).mean(1, keepdim=True).sqrt())).to(dt)
    if rung in ("rot", "rot-sr"):
        sr = gen(dev, "sr", name, seed) if rung == "rot-sr" else None
        if name.endswith(("o_proj", "down_proj")):
            r = rot_matrix(w.shape[0], dev, seed)
            return (r @ fp8_rows((r.T @ w), sr)).to(dt)
        r = rot_matrix(w.shape[1], dev, seed)
        return (fp8_rows((w @ r), sr) @ r.T).to(dt)
    raise ValueError(rung)


def deployed_codes(w, rung, name, seed):
    """The registered E4M3 codes (uint8) that the chain multiplies, for the verification-cost census."""
    dev = w.device
    w = w.float()
    if rung in ("rot", "rot-sr") and name.endswith(("o_proj", "down_proj")):
        r = rot_matrix(w.shape[0], dev, seed)
        w = r.T @ w
    elif rung in ("rot", "rot-sr"):
        w = w @ rot_matrix(w.shape[1], dev, seed)
    elif rung.startswith("mix-"):
        lam = float(rung[4:]); gm = torch.randn(w.shape, generator=gen(dev, "mix", name, seed), device=dev)
        w = (1 - lam) * w + lam * gm * (w.norm() / gm.norm())
    elif rung == "random":
        gm = torch.randn(w.shape, generator=gen(dev, "rand-w", name, seed), device=dev)
        w = gm * (w.pow(2).mean(1, keepdim=True).sqrt() / gm.pow(2).mean(1, keepdim=True).sqrt())
    elif rung.startswith("clamp-"):
        r = float(rung[6:]) * w.pow(2).mean(1, keepdim=True).sqrt(); w = w.clamp(-r, r)
    elif rung.startswith("specfloor-"):
        u, s, vh = torch.linalg.svd(w, full_matrices=False); w = (u * torch.maximum(s, s[0] / float(rung[10:]))) @ vh
    sr = gen(dev, "sr", name, seed) if rung == "rot-sr" else None
    alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    q = e4m3_sr(w * alpha, sr) if sr is not None else e4m3_rn(w * alpha)
    return q.to(torch.float8_e4m3fn).view(torch.uint8)


def skippable_fraction(codes, group=128, thresh=113):
    """A lower bound on the honest chain fraction an adversarial prover skips, from (near-)duplicate rows and 2:4 slices.

    Per (row, 128-group), find the closest other row's group; if they agree on >= thresh codes, the group is derived
    from that row at 8 units per differing code (out of 128 MAC-units), so it saves (128 - 8 d) / 128 of the group.
    2:4-compatible 64-slices (>= 2 zeros in every 4-group) run at half rate: save 0.5 of the slice. Take the better of
    the two per group. Sampled over rows to stay within the chunk budget.
    """
    n, k = codes.shape
    mag = codes & 0x7F
    dev = codes.device
    rows = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(0))[:min(n, 512)].to(dev)
    sub = codes[rows].int()
    ng = k // group
    saved = torch.zeros(len(rows), ng, device=dev)
    for gi in range(ng):
        blk = sub[:, gi * group:(gi + 1) * group]
        agree = (blk[:, None, :] == blk[None, :, :]).sum(2)
        agree.fill_diagonal_(-1)
        best_agree = agree.max(1).values.float()
        d = (group - best_agree).clamp_min(0)
        dup = torch.where(best_agree >= thresh, (group - 8.0 * d).clamp_min(0) / group, torch.zeros_like(d))
        z = (mag[rows][:, gi * group:(gi + 1) * group] == 0).int().reshape(len(rows), group // 4, 4).sum(2) >= 2
        s24 = 0.5 * z.reshape(len(rows), group // 64, 16).all(2).float().mean(1) if group >= 64 else torch.zeros(len(rows), device=dev)
        saved[:, gi] = torch.maximum(dup, s24)
    return float(saved.mean())


def deployed_values(w, rung, name, seed):
    """The deployed codes' real values in the basis the chain multiplies, exact in float64.

    'none': the registrant supplies its own codes and scales (the planted source is exact on the E4M3 grid with one
    scale), so the values are its own. Deterministic projections: the projected reals before rounding, the registrant's
    best case (it can pick row maxima so the verifier's per-row scales are powers of two). Keyed rotations: rotate, then
    the verifier's per-row round (or stochastic round)."""
    dev = w.device
    w = w.float()
    if rung == "none":
        return w.double()
    if rung.startswith("specfloor-"):
        u, s_, vh = torch.linalg.svd(w, full_matrices=False)
        if bool((s_ >= s_[0] / float(rung[10:])).all()):
            return w.double()                       # the projection is the identity: the registrant's values stand
        return ((u * torch.maximum(s_, s_[0] / float(rung[10:]))) @ vh).double()
    if rung.startswith("clamp-"):
        r = float(rung[6:]) * w.double().pow(2).mean(1, keepdim=True).sqrt()
        return w.double().clamp(-r, r)
    if rung in ("rot", "rot-sr") and name.endswith(("o_proj", "down_proj")):
        w = rot_matrix(w.shape[0], dev, seed).T @ w
    elif rung in ("rot", "rot-sr"):
        w = w @ rot_matrix(w.shape[1], dev, seed)
    sr = gen(dev, "sr", name, seed) if rung == "rot-sr" else None
    alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    q = e4m3_sr(w * alpha, sr) if sr is not None else e4m3_rn(w * alpha)
    return q.double() / alpha.double()


def relation_skippable(w, rung, name, seed, group=128):
    """For the planted source: the chain fraction the planting registrant skips, from its own triples row3 = row1 + row2.

    A 128-group of a derived row is free when the deployed values satisfy the relation at all 128 positions: the prover
    adds the two basis rows' group sums (one FADD, 8 units) instead of 128 MACs. Only rungs that keep the rows' basis
    ('none', 'clamp', 'specfloor') can keep the relation; the rotation breaks it by rounding generic values.
    """
    v = deployed_values(w, rung, name, seed)
    n, k = v.shape
    t = n // 3
    rel = (v[0:3 * t:3] + v[1:3 * t:3] - v[2:3 * t:3]).abs() <= 1e-9 * v.abs().max()
    ok = rel.reshape(t, k // group, group).all(2).float().mean()
    return float(ok) * (t / n) * (group - 8) / group


def adv_relation_weight(n, k, dev, seed):
    """A relation-packed adversarial matrix: row triples with row3 = row1 + row2 exact on the E4M3 grid, entropy high."""
    g = gen(dev, "adv", n, k, seed)
    t = n // 3
    e = (torch.randint(0, 8, (t, k), generator=g, device=dev) - 2).float()
    m1 = torch.randint(1, 8, (t, k), generator=g, device=dev).float()
    m2 = torch.randint(1, 8, (t, k), generator=g, device=dev).float()
    sgn = (torch.randint(0, 2, (t, k), generator=g, device=dev) * 2 - 1).float()
    r1, r2 = sgn * m1 * 2.0 ** (e - 4), sgn * m2 * 2.0 ** (e - 4)
    w = torch.zeros(n, k, device=dev)
    w[0:3 * t:3], w[1:3 * t:3], w[2:3 * t:3] = r1, r2, r1 + r2
    if n > 3 * t:
        w[3 * t:] = r1[: n - 3 * t]
    return w * 2.0 ** round(math.log2(0.02 / float(w.std())))


def load_model(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16).to(dev)
    model.eval()
    return tok, model


def linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head"]


def set_source(model, orig, source, seed):
    dev = next(model.parameters()).device
    with torch.no_grad():
        for n, m in linears(model):
            w = orig[n]
            if source == "real":
                m.weight.copy_(w)
            elif source == "random":
                gm = torch.randn(w.shape, generator=gen(dev, "src-rand", n, seed), device=dev)
                m.weight.copy_((gm * (w.float().pow(2).mean(1, keepdim=True).sqrt() / gm.pow(2).mean(1, keepdim=True).sqrt())).to(w.dtype))
            elif source == "adv":
                m.weight.copy_(adv_relation_weight(w.shape[0], w.shape[1], dev, seed).to(w.dtype))


def eval_ppl(model, ids, seqlen, windows, dev):
    nll = []
    with torch.no_grad():
        for w in range(windows):
            chunk = ids[w * seqlen:(w + 1) * seqlen].unsqueeze(0).to(dev)
            logits = model(chunk).logits
            nll.append(F.cross_entropy(logits[0, :-1].float(), chunk[0, 1:]).item())
            if PREEMPTED["flag"]:
                break
    return nll


def recover(model, train_ids, steps, seqlen, lr, dev, seed, params=None):
    if params is None:
        params = [p for n, p in model.named_parameters() if "norm" in n or n.endswith(".bias")]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        if PREEMPTED["flag"]:
            break
        start = int(torch.randint(0, train_ids.numel() - seqlen - 1, (1,), generator=g))
        chunk = train_ids[start:start + seqlen].unsqueeze(0).to(dev)
        loss = F.cross_entropy(model(chunk).logits[0, :-1].float(), chunk[0, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            losses.append(round(loss.item(), 4))
            print(json.dumps({"recover_step": step, "loss": losses[-1]}), flush=True)
    model.gradient_checkpointing_disable()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return losses


def write_result(run, out, variants):
    """$RESEARCH_RUN_DIR/result.json in research/result/v0.1: each variant's perplexity and skippable fraction, validated for
    completeness (present, every window evaluated, recovery run to the end)."""
    meas, files, bad = [], [], []
    for v in variants:
        f = out / f"{v.replace(':', '_')}.json"
        if not f.exists():
            bad.append(f"{v}: missing")
            continue
        d = json.loads(f.read_text())
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        if d["eval_windows"] != len(d["window_nll"]) or not d["window_nll"]:
            bad.append(f"{v}: incomplete")
        meas.append({"name": f"{v}/ppl", "value": d["ppl"], "unit": "perplexity"})
        if d.get("skippable_fraction_mean") is not None:
            meas.append({"name": f"{v}/skippable_fraction", "value": d["skippable_fraction_mean"], "unit": "fraction of the v1 chain"})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/gpu-tradeoff/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "model": "Qwen/Qwen2.5-7B", "data": "wikitext-2-raw-v1:test", "variants": variants},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/gpu-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(variants)} variants present and fully evaluated",
                          "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="comma list of source:rung:recover, recover in {0,1}")
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--recover-steps", type=int, default=150)
    ap.add_argument("--recover-seqlen", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--census-layers", type=int, default=4, help="how many layers' linears to census (from the middle)")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=0, help="start no new variant after this long (0: no limit)")
    ap.add_argument("--act", action="store_true", help="the fork's full activation path (noisy cast) and glue-affine recovery")
    args = ap.parse_args()
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")
    remaining = [v for v in variants if not (out / f"{v.replace(':', '_')}.json").exists()]
    if not remaining:
        if os.environ.get("RESEARCH_RUN_DIR"):
            write_result(Path(os.environ["RESEARCH_RUN_DIR"]), out, variants)
        print("all variants done"); return 0
    from datasets import load_dataset
    tok, model = load_model(dev)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids[0]
    train = tok("\n\n".join(ds["train"]["text"][:16000]), return_tensors="pt").input_ids[0]
    orig = {n: m.weight.detach().clone() for n, m in linears(model)}
    if args.act:
        for n, m in linears(model):
            m._aw_orig = orig[n]
    small0 = {n: p.detach().clone() for n, p in model.named_parameters() if "norm" in n or n.endswith(".bias")}
    all_lin = [n for n, _ in linears(model)]
    layer_ids = sorted({int(n.split(".")[2]) for n in all_lin})
    mid = layer_ids[len(layer_ids) // 2: len(layer_ids) // 2 + args.census_layers]
    census_names = [n for n in all_lin if int(n.split(".")[2]) in mid]
    left = False
    for v in remaining:
        if PREEMPTED["flag"]:
            left = True
            break
        if args.max_seconds and time.time() - t_start > args.max_seconds:
            break
        source, rung, rec = v.split(":")
        rec = int(rec)
        t0 = time.time()
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in small0:
                    p.copy_(small0[n])
            set_source(model, orig, source, args.seed)
            src_w = {n: m.weight.detach().clone() for n, m in linears(model)}
            for n, m in linears(model):
                m.weight.copy_(project(src_w[n], rung, n, args.seed))
        if args.act:
            install_act(model, rung, args.seed, dev)
        census = ({} if rung.endswith("@nvfp4") or args.act else
                  {n: skippable_fraction(deployed_codes(src_w[n], rung, n, args.seed)) for n in census_names})
        if source == "adv" and (rung in ("none", "rot", "rot-sr") or rung.startswith(("specfloor-", "clamp-"))):
            rel = {n: relation_skippable(src_w[n], rung, n, args.seed) for n in census_names}
            census = {n: max(census[n], rel[n]) for n in census_names}
        gp = glue_params(model, bool(rec)) if args.act else None
        rloss = recover(model, train, args.recover_steps, args.recover_seqlen, args.lr, dev, args.seed, gp) if rec else []
        want = args.eval_windows if args.eval_windows > 0 else test.numel() // args.seqlen
        nll = eval_ppl(model, test, args.seqlen, want, dev)
        if PREEMPTED["flag"] or len(nll) != want:
            left = True
            break                                   # an interrupted variant is never written; it reruns from scratch
        if args.act:
            glue_params(model, False)
        doc = {"variant": v, "source": source, "rung": rung, "recover": rec, "act": bool(args.act),
               "ppl": math.exp(sum(nll) / len(nll)) if nll else None, "window_nll": nll,
               "eval_windows": len(nll), "seqlen": args.seqlen, "recover_steps": args.recover_steps if rec else 0,
               "recover_loss_trace": rloss, "skippable_fraction_by_matrix": census,
               "skippable_fraction_mean": (sum(census.values()) / len(census)) if census else None,
               "census_layers": mid, "seconds": round(time.time() - t0, 1),
               "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               "gpu_uuid": os.environ.get("GPU_LEASE_UUID", ""), "clock_label": os.environ.get("AW_CLOCK", "")}
        (out / f"{v.replace(':', '_')}.json").write_text(json.dumps(doc, indent=1))
        print(json.dumps({k: doc[k] for k in ("variant", "ppl", "skippable_fraction_mean", "seconds")}), flush=True)
    done = all((out / f"{v.replace(':', '_')}.json").exists() for v in variants)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        write_result(Path(rd), out, variants)
    if PREEMPTED["flag"] and not done:
        return 143
    return 0 if done else 99


if __name__ == "__main__":
    sys.exit(main())
