#!/usr/bin/env python3
"""Probe the keyed rotation on a large checkpoint: the rotation's orthogonality at the model's hidden size on the GPU, and,
per linear of a few layers, the relative error of the rotated deployed product against the original product on random
inputs (it should be FP8's weight error, a few percent)."""
import glob
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_gpu as G  # noqa: E402
import aw_scale as S  # noqa: E402

dev = torch.device("cuda")
model = sys.argv[1] if len(sys.argv) > 1 else "unsloth/Meta-Llama-3.1-70B"
hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
snap = glob.glob(f"{hf}/hub/models--{model.replace('/', '--')}/snapshots/*")[0]
cfg = json.loads(Path(snap, "config.json").read_text())
d = cfg["hidden_size"]
r = G.rot_matrix(d, dev, 20260930)
eye = torch.eye(d, device=dev)
print(json.dumps({"d": d, "orth_err_max": float((r.T @ r - eye).abs().max()), "orth_err_fro": float((r.T @ r - eye).norm())}), flush=True)
sh = S.Shards(snap, dev)
for L in (0, 1, cfg["num_hidden_layers"] // 2):
    for role, part in (("q_proj", "self_attn"), ("o_proj", "self_attn"), ("gate_proj", "mlp"), ("down_proj", "mlp")):
        name = f"model.layers.{L}.{part}.{role}"
        w = sh.get(name + ".weight").float()
        x = torch.randn(64, w.shape[1], device=dev)
        ref = x @ w.T
        plain = x @ G.fp8_rows(w).T
        if role in ("o_proj", "down_proj"):
            wr = r.T @ w
            rot = (x @ G.fp8_rows(wr).T) @ r.T
        else:
            wr = w @ r
            rot = (x @ r) @ G.fp8_rows(wr).T
        e = lambda y: float((y - ref).norm() / ref.norm())  # noqa: E731
        print(json.dumps({"linear": name, "shape": list(w.shape), "plain_err": round(e(plain), 5), "rot_err": round(e(rot), 5),
                          "w_absmax": float(w.abs().max()), "wr_absmax": float(wr.abs().max()), "nan": bool(torch.isnan(wr).any())}), flush=True)
