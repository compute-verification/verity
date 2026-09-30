#!/usr/bin/env python3
"""§8j of docs/pouw/approved-weights.md: where the full activation path's folding loss comes from, and what cuts it
(Qwen2.5-7B, the fork's noisy activation cast, the whole WikiText-2 test split).

A deployable rotation keeps the residual stream in the rotated basis, so an RMSNorm scale gamma that the rotation mixes must
move into the next linears' weights (folding). A diagonal that is constant on each block the rotation mixes commutes with
it, so that part of gamma can stay on the activation side. Each rung is a block structure plus the gamma it keeps (g_keep,
constant per block); W's columns take gamma / g_keep. Rungs, base[@nvfp4][~w|~a]:
  none     no rotation: the registered checkpoint
  rotu     the full keyed Haar rotation with gamma left on the activation side. Not deployable (it needs an online rotation
           after every norm): the reference for what folding costs
  rot      the full rotation, gamma folded
  rotx     the split rotation (the massive channels S unmixed, Haar on the rest), gamma folded
  rotxg    the split rotation, gamma kept on S, folded on the rest
  rotbB    S unmixed plus B keyed Haar blocks over the other channels, grouped by their calibrated activation magnitude;
           gamma kept on S and as one scale per block (its RMS over the block), the rest folded
  ~w       weights quantized only (activations BF16); ~a activations cast only (weights BF16, in the deployed basis)
Unmixed channels go to keyed positions off the every-8th positions, where Pearl's rule samples the RMS that sets the noise.
Recovery: 0 none; 1 a per-channel scale and bias after q/k/v/gate/up (glue); 2 that plus a per-position scale after each
norm, in the rotated basis (the norm's own scale there, also glue).

One JSON per finished variant in --out (the checkpoint); a run starts no new variant after --max-seconds. Exit 0 when all
are done, 99 while any remain, 143 when preempted; $RESEARCH_RUN_DIR/result.json in research/result/v0.1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_gpu as G  # noqa: E402


def calibrate(model, ids, dev, cap, ratio=20.0):
    """S (as G.channel_scores) and each channel's activation magnitude: the mean |h / rms(h)| over tokens at every norm's
    input, geometrically averaged over the norms."""
    s, top = G.channel_scores(model, ids, dev, cap, ratio)
    prof = []
    hooks = [m.register_forward_pre_hook(lambda mod, a: prof.append(
        (a[0].detach().float() / a[0].detach().float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-30)).abs()
        .reshape(-1, a[0].shape[-1]).mean(0).clamp_min(1e-12).log()))
        for n, m in model.named_modules() if n.endswith(("input_layernorm", "post_attention_layernorm"))]
    with torch.no_grad():
        model(ids.unsqueeze(0).to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    return s, top, torch.stack(prof).mean(0).exp().tolist()


def block_rotation(d, s, prof, nblocks, dev, seed):
    """Q (d x d, orthogonal) and each channel's block id: the channels of s unmixed at keyed positions off every 8th,
    and the others in nblocks blocks of consecutive activation magnitude, each a keyed Haar rotation onto keyed positions."""
    pos = torch.randperm(d, generator=G.gen(dev, "fold-pos", d, seed), device=dev).tolist()
    ps = [p for p in pos if p % 8][:len(s)]
    rest = [p for p in pos if p not in set(ps)]
    others = sorted(set(range(d)) - set(s), key=lambda c: prof[c])
    q = torch.zeros(d, d, device=dev)
    blk = torch.empty(d, dtype=torch.long)
    for i, (c, p) in enumerate(zip(s, ps)):
        q[c, p] = 1.0
        blk[c] = -1 - i
    lo = 0
    for b in range(nblocks):
        idx = others[b * len(others) // nblocks:(b + 1) * len(others) // nblocks]
        n = len(idx)
        q[torch.tensor(idx, device=dev)[:, None], torch.tensor(rest[lo:lo + n], device=dev)[None, :]] = \
            G.rot_matrix(n, dev, seed + 1 + b)
        blk[idx] = b
        lo += n
    return q, blk


def parse(rung):
    base, _, mode = rung.partition("~")
    fmt = "nvfp4" if base.endswith("@nvfp4") else "fp8"
    return base.split("@")[0], fmt, mode


def install(model, rung, cal, seed, dev, cache):
    """Deploy every linear at the rung (see the module docstring) with the fork's activation cast."""
    base, fmt, mode = parse(rung)
    d = model.config.hidden_size
    s, prof = cal["channels"], cal["profile"]
    if base in ("none",):
        q = None
    elif base in ("rotu", "rot"):
        q = cache.setdefault(("full", d), G.rot_matrix(d, dev, seed))
    else:
        nb = 1 if base in ("rotx", "rotxg") else int(base[4:])
        if ("blk", nb) not in cache:
            cache[("blk", nb)] = block_rotation(d, s, prof, nb, dev, seed)
        q, blk = cache[("blk", nb)]
    qb = q.to(torch.bfloat16) if q is not None else None
    keep = {}
    for n, m in G.linears(model):
        nm = G.norm_of(n)
        if nm and nm not in keep:
            gam = model.get_submodule(nm)._fold_gamma
            if base in ("none", "rotu"):
                gk = gam.clone()
            elif base in ("rot", "rotx"):
                gk = torch.ones_like(gam)
            else:
                gk = torch.ones_like(gam)
                sm = blk.to(gam.device) < 0
                gk[sm] = gam[sm]
                if base.startswith("rotb"):
                    for b in range(int(base[4:])):
                        mb = blk.to(gam.device) == b
                        gk[mb] = gam[mb].pow(2).mean().sqrt()
            keep[nm] = gk.to(torch.bfloat16).float()   # the norm stores it in BF16; the fold divides by what it stores
    with torch.no_grad():
        for nm, gk in keep.items():
            model.get_submodule(nm).weight.copy_(gk.to(torch.bfloat16))
    quant = G.nvfp4_rows if fmt == "nvfp4" else G.fp8_rows
    for n, m in G.linears(model):
        w = m._aw_orig.float()
        m._rin = m._rout = None
        nm = G.norm_of(n)
        if nm:
            w = w * (model.get_submodule(nm)._fold_gamma / keep[nm])[None, :]
        with torch.no_grad():
            if q is not None and n.endswith(("o_proj", "down_proj")):
                w = q.T @ w
                m._rout = qb
            elif q is not None:
                w = w @ q
                m._rin = qb
            m.weight.copy_((w if mode == "a" else quant(w)).to(m.weight.dtype))
        m._aq = mode != "w"
        m._fmt = fmt
        m._f = torch.randn(m.in_features, 32, generator=G.gen(dev, "act-F", n, seed), device=dev)
        m._g = G.gen(dev, "act-E", n, seed)
        m._scale = m._shift = m._post = None

        def fwd(x, m=m):
            if m._rin is not None:
                x = x @ m._rin
            if m._post is not None:
                x = x * m._post.to(x.dtype)
            y = F.linear(G.act_quant(x, m._fmt, m._f, m._g) if m._aq else x, m.weight, m.bias)
            if m._scale is not None:
                y = y * m._scale.to(y.dtype) + m._shift.to(y.dtype)
            if m._rout is not None:
                y = y @ m._rout.T
            return y
        m.forward = fwd


def glue(model, level):
    """Level 1: per-channel scale and bias after q/k/v/gate/up. Level 2: also one per-position scale per norm, applied after
    its rotation (shared by the linears the norm feeds)."""
    params, post = [], {}
    for n, m in G.linears(model):
        m._scale = m._shift = m._post = None
        if level and n.endswith(("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")):
            m._scale = torch.nn.Parameter(torch.ones(m.out_features, device=m.weight.device, dtype=torch.float32))
            m._shift = torch.nn.Parameter(torch.zeros(m.out_features, device=m.weight.device, dtype=torch.float32))
            params += [m._scale, m._shift]
        if level >= 2 and G.norm_of(n):
            nm = G.norm_of(n)
            if nm not in post:
                post[nm] = torch.nn.Parameter(torch.ones(m.in_features, device=m.weight.device, dtype=torch.float32))
                params.append(post[nm])
            m._post = post[nm]
    return params


def write_result(run, out, variants):
    meas, files, bad = [], [], []
    for v in variants:
        f = out / f"{v.replace(':', '_')}.json"
        if not f.exists():
            bad.append(f"{v}: missing"); continue
        d = json.loads(f.read_text())
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        meas.append({"name": f"{v}/ppl", "value": d["ppl"], "unit": "perplexity"})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/fold/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "model": "Qwen/Qwen2.5-7B", "data": "wikitext-2-raw-v1:test", "variants": variants},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/fold-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(variants)} variants present and fully evaluated",
                          "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


def main():
    signal.signal(signal.SIGTERM, G.on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="comma list of rung:recover")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--eval-windows", type=int, default=0, help="0: the whole test split")
    ap.add_argument("--recover-steps", type=int, default=400)
    ap.add_argument("--recover-seqlen", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split-cap", type=int, default=0, help="at most this many unmixed channels (0: hidden_size // 1024)")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=240, help="start no new variant after this long")
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    variants = args.variants.split(",")
    remaining = [v for v in variants if not (out / f"{v.replace(':', '_')}.json").exists()]
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if not remaining:
        if rd:
            write_result(Path(rd), out, variants)
        return 0
    from datasets import load_dataset
    tok, model = G.load_model(dev)
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    test = tok("\n\n".join(ds["test"]["text"]), return_tensors="pt").input_ids[0]
    train = tok("\n\n".join(ds["train"]["text"][:16000]), return_tensors="pt").input_ids[0]
    for p in model.parameters():
        p.requires_grad_(False)
    for n, m in model.named_modules():
        if n.endswith(("input_layernorm", "post_attention_layernorm")):
            m._fold_gamma = m.weight.detach().float().clone()
    cf = out / "calibration.json"
    if not cf.exists():
        s, top, prof = calibrate(model, train[:args.seqlen], dev, args.split_cap or model.config.hidden_size // 1024)
        tmp = out / "calibration.json.tmp"
        tmp.write_text(json.dumps({"channels": s, "top_scores": top, "profile": prof,
                                   "calibration": f"WikiText-2 train, tokens 0-{args.seqlen}, BF16 as loaded"}))
        os.replace(tmp, cf)
    cal = json.loads(cf.read_text())
    for n, m in G.linears(model):
        m._aw_orig = m.weight.detach().clone()
    want = args.eval_windows or test.numel() // args.seqlen
    cache = {}
    for v in remaining:
        if G.PREEMPTED["flag"] or time.time() - t0 > args.max_seconds:
            break
        rung, rec = v.split(":")
        t1 = time.time()
        install(model, rung, cal, args.seed, dev, cache)
        params = glue(model, int(rec))
        trace = G.recover(model, train, args.recover_steps, args.recover_seqlen, args.lr, dev, args.seed, params) if params else []
        nll = G.eval_ppl(model, test, args.seqlen, want, dev)
        if G.PREEMPTED["flag"] or len(nll) != want:
            break                                   # an interrupted variant is never written; it reruns from scratch
        doc = {"variant": v, "rung": rung, "recover": int(rec), "ppl": math.exp(sum(nll) / len(nll)), "window_nll": nll,
               "eval_windows": len(nll), "seqlen": args.seqlen, "recover_steps": args.recover_steps if params else 0,
               "recover_loss_trace": trace, "split_channels": cal["channels"], "seconds": round(time.time() - t1, 1),
               "gpu": torch.cuda.get_device_name(0), "gpu_uuid": os.environ.get("GPU_LEASE_UUID", "")}
        tmp = out / f"{v.replace(':', '_')}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, out / f"{v.replace(':', '_')}.json")
        print(json.dumps({k: doc[k] for k in ("variant", "ppl", "seconds")}), flush=True)
    done = all((out / f"{v.replace(':', '_')}.json").exists() for v in variants)
    if rd:
        write_result(Path(rd), out, variants)
    if G.PREEMPTED["flag"] and not done:
        return 143
    return 0 if done else 99


if __name__ == "__main__":
    sys.exit(main())
