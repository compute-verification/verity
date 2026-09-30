#!/usr/bin/env python3
"""The fork's rotated NVFP4 codes' change probability over the key (docs/pouw/approved-weights.md §8i; bc-a8466279's
fork condition, theory-pearl-c4-domain.md §6.12b: the B-side layout route doesn't pay on the fork if each rotated code
leaves its mode with probability at least about 0.46).

On the fork the weights carry no noise, so the randomness a registrant can't anticipate is the key, drawn after it commits.
Before registration each deployed code is a random variable over keys; a registrant can arrange structure only around its
mode. For a code position t of row j: p_t = 1 - max_c P_key(code_t = c), estimated over --keys keys of rung 3's transform
(aw_fold.block_rotation, 8 blocks, gamma kept per block, the massive channels as the side product; positions and every
block's Haar keyed), as aw_census --forms blk8s deploys it. With finitely many keys the maximum empirical frequency is
biased up, so p is biased down: the estimate is conservative.

Per linear: the distribution of p over (row, position) (min, low quantiles, share below 0.46 and 0.44), the zero code's
share, and, at the deployed key (--seed), the share of low-p codes that fall in 128-windows the registration check (b)
uncredits (aw_zero_cover.reading > 0). Rows: all of k_proj and v_proj (n = 512 at Qwen2.5-7B, the small-n case), a fixed
sample of --rows for the others. Chunks: layer ranges, one JSON each in --out; exit 0 when done, 99 while any remain.
"""
from __future__ import annotations

import argparse
import glob
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
import aw_zero_cover as ZC  # noqa: E402

QS = (0.0, 1e-4, 1e-3, 1e-2, 5e-2, 0.5)


def main():
    signal.signal(signal.SIGTERM, AC.on_term)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", type=int, default=256)
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--layers-per-chunk", type=int, default=7)
    ap.add_argument("--seed", type=int, default=20260930, help="the deployed key, for check (b)")
    ap.add_argument("--max-seconds", type=float, default=360)
    args = ap.parse_args()
    import aw_fold as FD
    from safetensors import safe_open
    t0 = time.time()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snap = glob.glob(f"{hf}/hub/models--{args.model.replace('/', '--')}/snapshots/*")[0]
    cfg = json.loads(Path(snap, "config.json").read_text())
    L = cfg["num_hidden_layers"]
    chunks = [(lo, min(lo + args.layers_per_chunk, L)) for lo in range(0, L, args.layers_per_chunk)]
    names = [f"{args.model.split('/')[-1]}_L{lo:02d}-{hi - 1:02d}" for lo, hi in chunks]
    todo = [(c, n) for c, n in zip(chunks, names) if not (out / f"{n}.json").exists()]
    if todo:
        cal = json.loads(Path(args.calibration).read_text())
        d = cfg["hidden_size"]
        keys = [args.seed] + [args.seed + 1000 + i for i in range(args.keys - 1)]
        qs, bid = [], None
        for kk in keys:
            q, b = FD.block_rotation(d, cal["channels"], cal["profile"], 8, dev, kk)
            qs.append(q)
            bid = b.to(dev)
        sides = [torch.tensor([int(q[c_].argmax()) for c_ in cal["channels"]], device=dev) for q in qs]
        ip = Path(snap, "model.safetensors.index.json")
        idx = json.loads(ip.read_text())["weight_map"]
        files = {}

        def get(t):
            f = idx[t]
            if f not in files:
                files[f] = safe_open(str(Path(snap, f)), framework="pt", device="cpu")
            return files[f].get_tensor(t).to(dev)
        g = torch.Generator(device="cpu").manual_seed(7)
        print(json.dumps({"keys": len(keys), "setup_seconds": round(time.time() - t0, 1)}), flush=True)
    for (lo, hi), cname in todo:
        if AC.PRE["flag"] or time.time() - t0 > args.max_seconds:
            break
        recs, tc = [], time.time()
        for layer in range(lo, hi):
            for role in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                if AC.PRE["flag"]:
                    break
                part = "self_attn" if role in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
                name = f"model.layers.{layer}.{part}.{role}"
                w = get(name + ".weight").float()
                n, k = w.shape
                rows = torch.arange(n) if n <= args.rows else torch.randperm(n, generator=g)[:args.rows].sort().values
                rows = rows.to(dev)
                out_side = role in ("o_proj", "down_proj")
                if not out_side:
                    gam = get(f"model.layers.{layer}.{'input_layernorm' if part == 'self_attn' else 'post_attention_layernorm'}.weight").float()
                    gk = gam.clone()
                    for b in range(8):
                        gk[bid == b] = gam[bid == b].pow(2).mean().sqrt()
                    wf = (w * (gam / gk.to(torch.bfloat16).float())[None, :])[rows]
                cnt = torch.zeros(len(rows), k, 16, dtype=torch.int16, device=dev)
                code0 = None
                for i, q in enumerate(qs):
                    if out_side:
                        wr = q[:, rows].T @ w
                    else:
                        wr = wf @ q
                        wr[:, sides[i]] = 0.0
                    c = AC.nvfp4_codes(wr).long()
                    if i == 0:
                        code0 = c
                    cnt.scatter_add_(2, c.unsqueeze(2), torch.ones_like(c, dtype=torch.int16).unsqueeze(2))
                p = 1.0 - cnt.max(2).values.float() / len(qs)
                zero = (cnt[:, :, 0].float() / len(qs))
                pf = p.reshape(-1)
                qv = torch.quantile(pf[torch.randperm(pf.numel(), device=dev)[:min(pf.numel(), 4_000_000)]], torch.tensor(QS, device=dev))
                s = ZC.reading(((code0 & 7) != 0).reshape(-1, 128)).reshape(len(rows), k // 128)
                unc = (s > 0).repeat_interleave(128, dim=1)
                low = p < 0.46
                rec = {"linear": name, "n": n, "k": k, "rows_sampled": len(rows), "keys": len(qs),
                       "p_min": float(pf.min()), **{f"p_q{q_}": float(v) for q_, v in zip(QS, qv)},
                       "p_mean": float(pf.mean()), "share_below_0.46": float(low.float().mean()),
                       "share_below_0.44": float((p < 0.44).float().mean()),
                       "zero_code_share": float(zero.mean()), "rows_with_low_p": int(low.any(1).sum()),
                       "low_p_in_uncredited_windows": float((low & unc).sum() / low.sum()) if bool(low.any()) else None,
                       "p_min_outside_uncredited": float(p[~unc].min()) if bool((~unc).any()) else None,
                       "share_below_0.46_outside_uncredited": float((low & ~unc).float().mean())}
                recs.append(rec)
                print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in rec.items()
                                  if k_ in ("linear", "p_min", "p_q0.001", "p_q0.01", "share_below_0.46", "low_p_in_uncredited_windows")}), flush=True)
                del w, cnt, p
                torch.cuda.empty_cache()
        if AC.PRE["flag"]:
            break
        doc = {"chunk": cname, "model": args.model, "layers": [lo, hi - 1], "records": recs, "keys": args.keys,
               "deployed_key": args.seed, "seconds": round(time.time() - tc, 1), "gpu": torch.cuda.get_device_name(0)}
        tmp = out / f"{cname}.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, out / f"{cname}.json")
    done = all((out / f"{n}.json").exists() for n in names)
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        write_result(Path(rd), out, names)
    if AC.PRE["flag"] and not done:
        return 143
    return 0 if done else 99


def write_result(run, out, names):
    meas, files, bad = [], [], []
    for c in names:
        f = out / f"{c}.json"
        if not f.exists():
            bad.append(f"{c}: missing"); continue
        try:
            files.append(str(f.resolve().relative_to(run.resolve())))
        except ValueError:
            pass
        for r in json.loads(f.read_text())["records"]:
            meas += [{"name": f"{r['linear']}/p_min", "value": r["p_min"], "unit": "probability"},
                     {"name": f"{r['linear']}/p_q0.01", "value": r["p_q0.01"], "unit": "probability"},
                     {"name": f"{r['linear']}/share_below_0.46", "value": r["share_below_0.46"], "unit": "share of codes"}]
    doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
           "workload_fingerprint": {"workload": "approved-weights/change-prob/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                    "chunks": names},
           "validation": {"status": "failed" if bad else "passed", "checker": "approved-weights/change-prob-completeness/v1",
                          "detail": "; ".join(bad) if bad else f"all {len(names)} chunks present", "evidence": files},
           "measurements": meas, "measurement_files": files}
    tmp = run / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run / "result.json")


if __name__ == "__main__":
    sys.exit(main())
