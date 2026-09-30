#!/usr/bin/env python3
"""The fork's deployed operands, from a real-activation capture: for bc-a8466279's weight-side measurement of F1' on the
fork, where B is the registered weights with no noise.

Input: a capture in fp4_v3_real.py's cache format (`<cache>/<Org>--<Model>.npz`, keys `layers.{i}.{self_attn|mlp}.{name}.x`
(activation rows) and `.w` (out x in)). Output: the same keys in the rung-3 basis of docs/pouw/approved-weights.md §8j,
the 8-block rotation with the side product:
  q/k/v/gate/up   x' = (x / gamma * g_keep) Q and w' = (w * gamma / g_keep) Q, with the massive channels' positions zeroed in
                  both: they leave the chain as the BF16 side product (stored as `.side_positions`, with `.side_x`/`.side_w`)
  o/down          w' = Q^T w (rows mixed), x unchanged (the MLP intermediate's keyed permutation is not applied)
Here gamma is the norm's scale and g_keep the part the norm keeps (gamma on S, its RMS per block). Q is keyed on this
machine's generator (--seed), and every A.B^T is checked against the original's. The fork's registered B is
round-to-nearest NVFP4 (or FP8) of `.w`, and its A carries the fork's noise, formed from `.x`.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_fold as FD  # noqa: E402
from aw_scale import Shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="the input npz")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--calibration", required=True, help="aw_fold's calibration.json (channels S and the magnitude profile)")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out", required=True, help="the output npz (same file name as the input, in another directory)")
    args = ap.parse_args()
    hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    snap = glob.glob(f"{hf}/hub/models--{args.model.replace('/', '--')}/snapshots/*")[0]
    shards = Shards(snap, torch.device("cpu"))

    def tensor(name):
        return shards.get(name).float()

    cal = json.loads(Path(args.calibration).read_text())
    s, prof = cal["channels"], cal["profile"]
    z = np.load(args.cache)
    d = None
    out, checks = {}, {}
    q = blk = None
    for key in sorted({k.rsplit(".", 1)[0] for k in z.files}):
        x = torch.from_numpy(z[key + ".x"]).double()
        w = torch.from_numpy(z[key + ".w"]).double()
        name = key.rsplit(".", 1)[1]
        layer = int(key.split(".")[1])
        if q is None:
            d = w.shape[1] if name in ("q_proj", "gate_proj") else w.shape[0]
            qf, blk = FD.block_rotation(d, s, prof, args.blocks, torch.device("cpu"), args.seed)
            q = qf.double()
            ps = torch.tensor([int(q[c].argmax()) for c in s])
        ref = x @ w.T
        if name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            norm = "input_layernorm" if name in ("q_proj", "k_proj", "v_proj") else "post_attention_layernorm"
            gam = tensor(f"model.layers.{layer}.{norm}.weight").double()
            gk = gam.clone()
            for b in range(args.blocks):
                gk[blk == b] = gam[blk == b].pow(2).mean().sqrt()
            gk = gk.to(torch.bfloat16).double()
            xr = (x / gam * gk) @ q
            wr = (w * (gam / gk)) @ q
            out[key + ".side_positions"] = ps.numpy()
            out[key + ".side_x"] = xr[:, ps].float().numpy()
            out[key + ".side_w"] = wr[:, ps].float().numpy()
            got = xr @ wr.T
            xr[:, ps] = 0.0
            wr[:, ps] = 0.0
        else:
            xr, wr = x, q.T @ w
            got = (xr @ wr.T) @ q.T
        checks[key] = float((got - ref).abs().max() / ref.abs().max())
        out[key + ".x"] = xr.float().numpy()
        out[key + ".w"] = wr.float().numpy()
        print(json.dumps({"linear": key, "x": list(xr.shape), "w": list(wr.shape), "max_rel_err_of_AB": checks[key]}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, args.out)
    meta = {"source_cache": args.cache, "model": args.model, "calibration": args.calibration, "split_channels": s,
            "blocks": args.blocks, "seed": args.seed, "generator": "cpu", "max_rel_err_of_AB": checks,
            "transform": "approved-weights.md §8j: 8 keyed Haar blocks by activation magnitude, gamma kept per block, S as a side product"}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        doc = {"schema": "research/result/v0.1", "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
               "workload_fingerprint": {"workload": "approved-weights/fork-capture/v1", "source_sha": os.environ.get("RESEARCH_SOURCE_SHA", ""),
                                        "model": args.model, "cache": args.cache},
               "validation": {"status": "passed" if max(checks.values()) < 1e-4 else "failed",
                              "checker": "approved-weights/fork-capture-product/v1",
                              "detail": f"max relative error of A.B^T over {len(checks)} linears: {max(checks.values()):.2e}",
                              "evidence": []},
               "measurements": [{"name": f"{k}/max_rel_err_of_AB", "value": v, "unit": "relative"} for k, v in checks.items()],
               "measurement_files": []}
        Path(rd, "result.json").write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main()
