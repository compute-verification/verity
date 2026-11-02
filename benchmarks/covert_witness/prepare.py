"""The fixed side of the covert-witness speedrun: data, the registered starts, the decoders, the training-FLOP meter, the
witness checker and `evaluate_bpb`. A candidate's `train.py` imports it and never edits it; `evaluate.py` runs it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("CW_ROOT", "/workspace/research/scratch/covert-witness"))
LF_CORPUS = Path(os.environ.get("CW_LF_CORPUS", "/workspace/research/scratch/learning-frontier/fineweb-edu-gpt2-v1"))
STARTS = ROOT / "starts"
HELD_OUT = ROOT / "eval"


def digest(theta: dict[str, torch.Tensor]) -> str:
    """sha256 over the tensors in name order: each name, dtype, shape and its little-endian bytes."""
    h = hashlib.sha256()
    for name in sorted(theta):
        t = theta[name].detach().cpu().contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def register_start(name: str, theta: dict[str, torch.Tensor], config: dict, tokenizer: str, provenance: dict) -> dict:
    from safetensors.torch import save_file

    d = STARTS / name
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in theta.items()}, str(d / "theta.safetensors.tmp"))
    (d / "theta.safetensors.tmp").rename(d / "theta.safetensors")
    rec = {"schema": "covert-witness/start/v1", "name": name, "digest": digest(theta), "config": config,
           "tokenizer": tokenizer, "parameters": int(sum(v.numel() for v in theta.values())),
           "tensors": {k: [str(v.dtype), list(v.shape)] for k, v in sorted(theta.items())}, "provenance": provenance}
    (d / "start.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec
