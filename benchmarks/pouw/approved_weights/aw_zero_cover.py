#!/usr/bin/env python3
"""Does the registration-time zero-structure uncredit cover what the fork variant's structural reading charges, and what
does it cost? (docs/pouw/approved-weights.md §8i; the fork line's open check, ratings.md 15:12Z.)

The structural reading (bc-a8466279's fork_variant.py: dnc2.window_saving with p = 1 on non-zero codes, 0 on zero codes)
prices two zero-code routes per 128-code window of a registered NVFP4 row, at the prover's best layout within each
16-block (a prover may permute k inside a block on both operands: the block scale and the exact chain don't change):
  sparse   X = the non-zeros beyond 2:4 at that layout, patched on the CUDA cores: saving max(0, 0.5 - C X)
  skip     per k64 half, N = its non-zero codes, skipped and patched: saving max(0, 0.5 - C N)
with C = 16.92/128, X and N truncated at 4. For binary p, X is a table over each block's zero count (EXC below, checked
against dnc2.window_saving by --check-dnc2), so the reading runs on the GPU over every linear.

Registration rules compared, per linear, as shares of the linear's credit (each window is 128 MACs of every output):
  reading    the structural reading itself: sum of S_w / windows (the charge a variant would put on the weights)
  window     uncredit every window with S_w > 0 (covers the reading by construction)
  census     aw_census's fixed-layout rule: 2:4-compatible 64-slices and all-zero 32-atoms; with its coverage, the share of
             the reading's S_w that falls in windows it uncredits
Forms: NVFP4 under rung 3's 8-block rotation with the side product (blk8s, as aw_census) and as registered (plain).
Chunks: layer ranges per model, one JSON each in --out. Exit 0 when done, 99 while any remain, 143 when preempted.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_census as AC  # noqa: E402

C = 16.92 / 128
XM = 4
VX = [max(0.0, 0.5 - C * x) for x in range(XM + 1)]


def _exc(js):
    dev = sorted(j for j in js if j > 0)
    return min(sum(dev[:len(dev) - 2]) if len(dev) > 2 else 0, XM)


def block_excess_table():
    """X for one 16-block with z zero codes, at dnc2's layout: sort (zeros first), pair neighbours, alternate pairs to two
    chunks, each chunk's excess by EXC; the block's is their sum, truncated at XM."""
    out = []
    for z in range(17):
        s = [0] * z + [1] * (16 - z)
        pj = [s[2 * i] + s[2 * i + 1] for i in range(8)]
        out.append(min(_exc(pj[0::2]) + _exc(pj[1::2]), XM))
    return out


EX = block_excess_table()


def reading(nz):
    """nz: (W, 128) bool, non-zero codes. Returns S_w per window (the structural reading, binary p)."""
    ex = torch.tensor(EX, device=nz.device)
    vx = torch.tensor(VX, device=nz.device)
    zb = (~nz).reshape(nz.shape[0], 8, 16).sum(2)
    x = ex[zb].sum(1).clamp(max=XM)
    s = vx[x]
    for h in (slice(0, 64), slice(64, 128)):
        s = s + vx[nz[:, h].sum(1).clamp(max=XM)]
    return s


def census_rule(c):
    """aw_census's fixed-layout rule on codes c (n, k): per 128-window, the share uncredited (2:4-compatible 64-slices and
    all-zero 32-atoms, each counted once)."""
    z = (c & 7) == 0
    n, k = z.shape
    s64 = (z.reshape(n, k // 4, 4).sum(2) >= 2).reshape(n, k // 64, 16).all(2)            # (n, k/64)
    a32 = z.reshape(n, k // 32, 32).all(2).reshape(n, k // 64, 2)                           # (n, k/64, 2)
    half = torch.where(s64, torch.ones_like(s64, dtype=torch.float32), a32.float().mean(2))  # uncredited share of a k64
    return half.reshape(n, k // 128, 2).mean(2)                                              # per 128-window


def measure(c):
    n, k = c.shape
    nz = ((c & 7) != 0).reshape(-1, 128)
    s = reading(nz)
    cen = census_rule(c).reshape(-1)
    return {"windows": int(s.numel()), "reading": float(s.mean()), "window_rule": float((s > 0).float().mean()),
            "census_rule": float(cen.mean()),
            "census_covers_reading": float((s * (cen > 0)).sum() / s.sum()) if float(s.sum()) > 0 else None,
            "census_covers_reading_fully": float((s * (cen >= 1)).sum() / s.sum()) if float(s.sum()) > 0 else None,
            "reading_max_row": float(s.reshape(n, -1).mean(1).max())}


def check_dnc2(path):
    """Compare reading() with bc-a8466279's dnc2.window_saving on random binary windows of every density."""
    sys.path.insert(0, path)
    import numpy as np
    import dnc2
    rng = np.random.default_rng(3)
    p = (rng.random((4000, 128)) < rng.random((4000, 1))).astype(np.float64)
    ref = dnc2.window_saving(p)
    got = reading(torch.from_numpy(p > 0)).double().numpy()
    return float(abs(ref - got).max())


def main():
    signal.signal(signal.SIGTERM, AC.on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="unsloth/Meta-Llama-3.1-70B:10,Qwen/Qwen2.5-7B:14")
    ap.add_argument("--calibration", default="", help="model=calibration.json,... (as aw_census --forms blk8s)")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--max-seconds", type=float, default=360)
    ap.add_argument("--check-dnc2", default="", help="a directory holding dnc2.py: compare, print, exit")
    args = ap.parse_args()
    if args.check_dnc2:
        print(json.dumps({"max_abs_diff_vs_dnc2": check_dnc2(args.check_dnc2), "block_excess_table": EX}))
        return 0
    import aw_fold as FD
    from safetensors import safe_open
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cals = dict(kv.split("=", 1) for kv in args.calibration.split(",") if kv)
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    plan = []
    for spec in args.models.split(","):
        model, per = spec.rsplit(":", 1)
        snap = glob.glob(f"{hf}/hub/models--{model.replace('/', '--')}/snapshots/*")[0]
        cfg = json.loads(Path(snap, "config.json").read_text())
        for lo in range(0, cfg["num_hidden_layers"], int(per)):
            plan.append((model, snap, cfg, lo, min(lo + int(per), cfg["num_hidden_layers"])))
    names = [f"{m.split('/')[-1]}_L{lo:02d}-{hi - 1:02d}" for m, _, _, lo, hi in plan]
    for (model, snap, cfg, lo, hi), cname in zip(plan, names):
        if (out / f"{cname}.json").exists():
            continue
        if AC.PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        ip = Path(snap, "model.safetensors.index.json")
        idx = json.loads(ip.read_text())["weight_map"] if ip.exists() else None
        files = {}

        def get(t):
            f = idx[t] if idx else "model.safetensors"
            if f not in files:
                files[f] = safe_open(str(Path(snap, f)), framework="pt", device="cpu")
            return files[f].get_tensor(t).to(dev)
        cal = json.loads(Path(cals[model]).read_text())
        q, bid = FD.block_rotation(cfg["hidden_size"], cal["channels"], cal["profile"], 8, dev, args.seed)
        bid = bid.to(dev)
        side = torch.tensor([int(q[c_].argmax()) for c_ in cal["channels"]], device=dev)
        recs, tc = [], time.time()
        for L in range(lo, hi):
            for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if AC.PRE["flag"]:
                    break
                part = "self_attn" if role in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                name = f"model.layers.{L}.{part}.{role}"
                w = get(name + ".weight").float()
                if role in ("o_proj", "down_proj"):
                    wr = q.T @ w
                else:
                    gam = get(f"model.layers.{L}.{'input_layernorm' if part == 'self_attn' else 'post_attention_layernorm'}.weight").float()
                    gk = gam.clone()
                    for b in range(8):
                        gk[bid == b] = gam[bid == b].pow(2).mean().sqrt()
                    wr = (w * (gam / gk.to(torch.bfloat16).float())[None, :]) @ q
                    wr[:, side] = 0.0
                for form, ww in (("plain", w), ("blk8s", wr)):
                    recs.append({"linear": name, "n": w.shape[0], "k": w.shape[1], "form": f"nvfp4-{form}",
                                 **measure(AC.nvfp4_codes(ww))})
                del w, wr
                torch.cuda.empty_cache()
        if AC.PRE["flag"]:
            break
        doc = {"chunk": cname, "model": model, "layers": [lo, hi - 1], "records": recs, "C": C, "block_excess_table": EX,
               "seconds": round(time.time() - tc, 1), "gpu": torch.cuda.get_device_name(0)}
        tmp = out / f"{cname}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, out / f"{cname}.json")
        print(json.dumps({"chunk": cname, "linears": len(recs) // 2, "seconds": doc["seconds"]}), flush=True)
    done = all((out / f"{c}.json").exists() for c in names)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        write_result(Path(rd), out, names)
    if AC.PRE["flag"] and not done:
        return 143
    return 0 if done else 99


def write_result(run, out, names):
    meas, files, bad, agg = [], [], [], {}
    for c in names:
        f = out / f"{c}.json"
        if not f.exists():
            bad.append(f"{c}: missing"); continue
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        d = json.loads(f.read_text())
        for r in d["records"]:
            a = agg.setdefault((d["model"].split("/")[-1], r["form"]), {"macs": 0, "reading": 0, "window": 0, "census": 0, "covered": 0})
            m = r["n"] * r["k"]
            a["macs"] += m
            a["reading"] += r["reading"] * m
            a["window"] += r["window_rule"] * m
            a["census"] += r["census_rule"] * m
            a["covered"] += (r["census_covers_reading"] or 0) * r["reading"] * m
    for (model, form), a in sorted(agg.items()):
        meas += [{"name": f"{model}/{form}/{key}", "value": a[key] / a["macs"], "unit": "share of credit"} for key in ("reading", "window", "census")]
        if a["reading"]:
            meas.append({"name": f"{model}/{form}/census_covers_reading", "value": a["covered"] / a["reading"], "unit": "share of the reading"})
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/zero-cover/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "chunks": names},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/zero-cover-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(names)} chunks present", "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


if __name__ == "__main__":
    sys.exit(main())
