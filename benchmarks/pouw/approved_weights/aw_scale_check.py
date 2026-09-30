#!/usr/bin/env python3
"""Checks aw_scale's compact model against the reference forward on one real checkpoint (on whatever device is given):
WikiText-2 perplexity of the reference, then of the compact build as registered and rotated, FP8 and NVFP4, over a few
windows. The compact FP8 build should match the reference to within FP8's weight quantization."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aw_scale as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="unsloth/Llama-3.2-1B")
ap.add_argument("--windows", type=int, default=2)
ap.add_argument("--reference", action="store_true")
args = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hf = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
snap = glob.glob(f"{hf}/hub/models--{args.model.replace('/', '--')}/snapshots/*")[0]
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(snap)
ids = tok("\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]), return_tensors="pt").input_ids[0]


def ppl(model):
    nll = []
    with torch.no_grad():
        for w in range(args.windows):
            c = ids[w * 2048:(w + 1) * 2048].unsqueeze(0).to(dev)
            nll.append(F.cross_entropy(model(c).logits[0, :-1].float(), c[0, 1:]).item())
    return round(float(torch.tensor(nll).mean().exp()), 4)


out = {"model": args.model, "device": str(dev)}
if args.reference:
    ref = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16).to(dev).eval()
    out["reference"] = ppl(ref)
    del ref
    torch.cuda.empty_cache()
cfg, model, shards, names = S.build(snap, dev)
for rung in ("none", "rot", "none@nvfp4", "rot@nvfp4"):
    with torch.no_grad():
        S.project(model, shards, names, rung, 20260930, dev)
    out[rung] = ppl(model)
    print(json.dumps(out), flush=True)
