#!/usr/bin/env python3
"""The skippable-structure census on every linear of a real checkpoint, before and after the keyed rotation, on one GPU.

For each transformer linear and each deployed form (FP8 or NVFP4, as registered or after the rotation: input side of
q/k/v/gate/up, output side of o/down), it measures on the deployed codes:
  spikes      max/RMS per row (RMS over every 8th position) and rows failing liveness (> k/64 entries at >= 8 RMS)
  zeros       zero codes, half-empty 4-groups (>= 2 zeros), 2:4-compatible 64-slices
  duplicates  exact duplicate rows up to sign
  near-dups   EXACT detection of every pair of rows whose block differs in at most D codes, where a derived block pays at
              the measured FADD price (8.38 FP8-MAC units, GPU 0's r20260930-063213-c94a): v1 128-groups D = 15, v2 32-atoms
              D = 2 (FP8), and whole rows D = floor(k / 16.76) for NVFP4's exact chain. Pigeonhole: two blocks at most D
              apart agree on one of D + 1 disjoint chunks, so hashing the chunks finds every candidate; each is verified.
  skippable   the fraction of the v1 chain (FP8) or of the exact chain (NVFP4) a prover skips through those near-dups
              (a derived block costs 8.38 per differing code) and 2:4 slices (half rate)
  timing      seconds to derive the deployed codes and to run the near-duplicate check: the registration cost
Per chunk it also plants the relation-packed adversarial source (row3 = row1 + row2 on the E4M3 grid) in one layer and
reports its skippable fraction as registered and after the rotation.

Chunks: layer ranges per model, one JSON each in --out (the checkpoint). Exit 0 when all are done, 99 while any remain,
143 when preempted; $RESEARCH_RUN_DIR/result.json in research/result/v0.1.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_gpu as G  # noqa: E402

FADD = 8.38          # FP8-MAC units per FP32 add on sm_120 (measured)
PRE = G.PREEMPTED


def on_term(signum, frame):
    PRE["flag"] = True


def fp8_codes(w):
    alpha = 448.0 / w.abs().amax(1, keepdim=True).clamp_min(1e-30)
    return G.e4m3_rn(w * alpha).to(torch.float8_e4m3fn).view(torch.uint8)


def nvfp4_codes(w):
    n, k = w.shape
    gs = w.abs().amax(1, keepdim=True).clamp_min(1e-30) / (6.0 * 448.0)
    v = (w / gs).reshape(n, k // 16, 16)
    s = G.e4m3_rn(v.abs().amax(2, keepdim=True) / 6.0).clamp_min(2.0 ** -9)
    x = (v / s).clamp(-6.0, 6.0)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=w.device)
    i = torch.bucketize(x.abs(), mids)
    return (i + 8 * ((x < 0) & (i > 0))).to(torch.uint8).reshape(n, k)


def spikes(w):
    rho = w[:, ::8].pow(2).mean(1).sqrt().clamp_min(1e-30)
    r = w.abs().amax(1) / rho
    live = (w.abs() >= 8 * rho[:, None]).sum(1)
    return {"spike_max": round(float(r.max()), 2), "spike_p99": round(float(torch.quantile(r.float(), 0.99)), 2),
            "rows_failing_liveness": int((live > w.shape[1] // 64).sum())}


def zero_stats(c, fp4):
    z = (c & (7 if fp4 else 0x7F)) == 0
    n, k = z.shape
    g4 = z.reshape(n, k // 4, 4).sum(2) >= 2
    s64 = g4.reshape(n, k // 64, 16).all(2)
    return {"zero_codes": float(z.float().mean()), "half_empty_4groups": float(g4.float().mean()),
            "slices_2to4": int(s64.sum()), "slices": int(s64.numel())}, s64


def canon(c, fp4):
    sign = 8 if fp4 else 0x80
    mag = c & (sign - 1)
    first = torch.argmax((mag != 0).to(torch.uint8), dim=1)
    neg = (c.gather(1, first[:, None]).squeeze(1) & sign) != 0
    return torch.where(neg[:, None] & (mag != 0), c ^ sign, c)


def near_dups(c, width, D, fp4):
    """Every (row, block) with another row's same block at most D codes away: returns (best saving per (row, block) as a
    fraction of the block's chain cost, number of flagged (row, block))."""
    cc = canon(c, fp4).to(torch.int64)
    n, k = cc.shape
    nb = k // width
    chunks = D + 1
    cw = -(-width // chunks)
    blk = cc.reshape(n, nb, width)
    if chunks * cw > width:
        blk = torch.cat([blk, torch.zeros(n, nb, chunks * cw - width, dtype=torch.int64, device=c.device)], 2)
    pw = (torch.full((cw,), 1000003, dtype=torch.int64, device=c.device) ** torch.arange(cw, device=c.device))
    h = (blk.reshape(n, nb, chunks, cw) * pw).sum(3).reshape(n, nb * chunks)
    h = h * 1315423911 + torch.arange(nb * chunks, device=c.device) * 2654435761    # separate columns
    hs, idx = torch.sort(h, dim=0)
    best = torch.zeros(n, nb, device=c.device)
    flagged = 0
    for off in range(1, 5):
        if off >= n:
            break
        eq = hs[off:] == hs[:-off]
        rr, col = torch.nonzero(eq, as_tuple=True)
        if rr.numel() == 0:
            continue
        a, b = idx[rr, col], idx[rr + off, col]
        g = col // chunks
        diff = (blk[a, g, :width] != blk[b, g, :width]).sum(1)
        ok = diff <= D
        if ok.any():
            a, b, g, d = a[ok], b[ok], g[ok], diff[ok].float()
            cost = FADD * 2 * d if fp4 else FADD * (d + 1 if width == 32 else d)     # a v2 atom also adds its basis partial
            save = ((width - cost) / width).clamp_min(0)
            flat = best.view(-1)
            flat.scatter_reduce_(0, a * nb + g, save, reduce="amax")
            flat.scatter_reduce_(0, b * nb + g, save, reduce="amax")
    flagged = int((best > 0).sum())
    return best, flagged


def census_matrix(w, fmt, name, rot, seed, dev):
    t0 = time.time()
    if rot and name.endswith(("o_proj", "down_proj")):
        w = G.rot_matrix(w.shape[0], dev, seed).T @ w
    elif rot:
        w = w @ G.rot_matrix(w.shape[1], dev, seed)
    fp4 = fmt == "nvfp4"
    c = nvfp4_codes(w) if fp4 else fp8_codes(w)
    torch.cuda.synchronize()
    t_derive = time.time() - t0
    rec = {"form": f"{fmt}-{'rot' if rot else 'plain'}", **spikes(w)}
    zs, s64 = zero_stats(c, fp4)
    rec.update(zs)
    t1 = time.time()
    cn = canon(c, fp4)
    rec["dup_rows"] = int(cn.shape[0] - torch.unique(cn, dim=0).shape[0])
    n, k = c.shape
    if fp4:
        D = int(k / (2 * FADD))
        best, fl = near_dups(c, k, D, True)
        rec.update({"near_dup_rows_nvfp4": fl, "near_dup_D": D})
        skip = torch.maximum(best.squeeze(1), 0.5 * s64.float().mean(1))
        rec["skippable"] = float(skip.mean())
    else:
        b1, f1 = near_dups(c, 128, 15, False)
        b2, f2 = near_dups(c, 32, 2, False)
        rec.update({"near_dup_groups_v1": f1, "near_dup_atoms_v2": f2})
        s24g = 0.5 * s64.float().reshape(n, k // 128, 2).mean(2)    # a 128-group holds two 64-slices
        rec["skippable"] = float(torch.maximum(b1, s24g).mean())
        rec["skippable_v2"] = float(b2.mean())
    torch.cuda.synchronize()
    rec["seconds_derive"] = round(t_derive, 3)
    rec["seconds_check"] = round(time.time() - t1, 3)
    return rec


def planted(n, k, name, dev, seed):
    a = G.adv_relation_weight(n, k, dev, seed)
    out = {}
    for rung in ("none", "rot"):
        rel = G.relation_skippable(a, rung, name, seed)
        c = G.deployed_codes(a, rung, name, seed)
        b1, _ = near_dups(c, 128, 15, False)
        out[rung] = round(max(rel, float(b1.mean())), 5)
    return out


def main():
    signal.signal(signal.SIGTERM, on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="unsloth/Meta-Llama-3.1-70B:10,Qwen/Qwen2.5-7B:14", help="model:layers-per-chunk,...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=900)
    args = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    from safetensors import safe_open
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    plan = []
    for spec in args.models.split(","):
        model, per = spec.rsplit(":", 1)
        snap = glob.glob(f"{hf}/hub/models--{model.replace('/', '--')}/snapshots/*")[0]
        cfg = json.loads(Path(snap, "config.json").read_text())
        for lo in range(0, cfg["num_hidden_layers"], int(per)):
            hi = min(lo + int(per), cfg["num_hidden_layers"])
            plan.append((model, snap, lo, hi))
    names = [f"{m.split('/')[-1]}_L{lo:02d}-{hi - 1:02d}" for m, _, lo, hi in plan]
    for (model, snap, lo, hi), cname in zip(plan, names):
        if (out / f"{cname}.json").exists():
            continue
        if PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"] \
            if Path(snap, "model.safetensors.index.json").exists() else None
        files = {}

        def get(t):
            f = idx[t] if idx else "model.safetensors"
            if f not in files:
                files[f] = safe_open(str(Path(snap, f)), framework="pt", device="cuda")
            return files[f].get_tensor(t)
        recs, tc = [], time.time()
        keys = sorted(idx) if idx else None
        for L in range(lo, hi):
            for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                part = "self_attn" if role in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                t = f"model.layers.{L}.{part}.{role}.weight"
                w = get(t).float()
                name = t[:-7]
                for fmt in ("fp8", "nvfp4"):
                    for rot in (False, True):
                        if PRE["flag"]:
                            break
                        r = census_matrix(w, fmt, name, rot, args.seed, dev)
                        recs.append({"linear": name, "n": w.shape[0], "k": w.shape[1], **r})
                del w
            if PRE["flag"]:
                break
        if PRE["flag"]:
            break
        adv = {}
        for role, part in (("q_proj", "self_attn"), ("gate_proj", "mlp"), ("down_proj", "mlp")):
            shape = get(f"model.layers.{lo}.{part}.{role}.weight").shape
            adv[role] = planted(shape[0], shape[1], f"model.layers.{lo}.{part}.{role}", dev, args.seed)
        doc = {"chunk": cname, "model": model, "layers": [lo, hi - 1], "records": recs, "planted": adv,
               "seconds": round(time.time() - tc, 1), "gpu": torch.cuda.get_device_name(0),
               "gpu_uuid": os.environ.get("GPU_LEASE_UUID", ""), "fadd_units": FADD}
        (out / f"{cname}.json").write_text(json.dumps(doc, indent=1))
        print(json.dumps({"chunk": cname, "linears": len(recs) // 4, "seconds": doc["seconds"], "planted": adv}), flush=True)
    done = all((out / f"{c}.json").exists() for c in names)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        write_result(Path(rd), out, names)
    if PRE["flag"] and not done:
        return 143
    return 0 if done else 99


def write_result(run, out, names):
    meas, files, bad = [], [], []
    by = {}
    for c in names:
        f = out / f"{c}.json"
        if not f.exists():
            bad.append(f"{c}: missing"); continue
        d = json.loads(f.read_text())
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        for r in d["records"]:
            key = (d["model"].split("/")[-1], r["form"])
            m = by.setdefault(key, {"macs": 0.0, "skip": 0.0, "dup": 0, "nd": 0, "s24": 0, "derive": 0.0, "check": 0.0})
            macs = r["n"] * r["k"]
            m["macs"] += macs; m["skip"] += r["skippable"] * macs; m["dup"] += r["dup_rows"]
            m["nd"] += r.get("near_dup_groups_v1", r.get("near_dup_rows_nvfp4", 0)); m["s24"] += r["slices_2to4"]
            m["derive"] += r["seconds_derive"]; m["check"] += r["seconds_check"]
    for (model, form), m in sorted(by.items()):
        meas += [{"name": f"{model}/{form}/skippable_mac_weighted", "value": m["skip"] / m["macs"], "unit": "fraction of the chain"},
                 {"name": f"{model}/{form}/dup_rows", "value": m["dup"], "unit": "rows"},
                 {"name": f"{model}/{form}/near_dup_blocks", "value": m["nd"], "unit": "blocks"},
                 {"name": f"{model}/{form}/slices_2to4", "value": m["s24"], "unit": "slices"},
                 {"name": f"{model}/{form}/seconds_derive", "value": m["derive"], "unit": "s"},
                 {"name": f"{model}/{form}/seconds_check", "value": m["check"], "unit": "s"}]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/census/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "chunks": names, "fadd_units": FADD},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/census-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(names)} chunks present", "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


if __name__ == "__main__":
    sys.exit(main())
