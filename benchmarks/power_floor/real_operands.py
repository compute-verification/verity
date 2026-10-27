"""Real s8 operands for energy.cu's GEMM mode 12, from Llama-3.1-8B's safetensors in the node's weight cache.

  python3 real_operands.py OUTDIR [TOKENS]

OUTDIR gets w.s8 (layer 0's gate_proj and up_proj stacked, 28,672 × 4,096, each output channel absmax-quantized to s8:
the GEMM's whole weight buffer) and x.s8 (TOKENS rows of 4,096, default 16,384: layer 0's input, the input RMSNorm of
the token embeddings times its weight, each token absmax-quantized to s8, for token ids drawn uniformly from the
vocabulary), and prints each file's statistics as one JSON line. numpy only.
"""

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

REPO = "models--unsloth--Llama-3.1-8B"
REV = "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"  # tools/research/src/research/pods/nebius/weights.tsv


def snapshot() -> Path:
    root = Path(os.environ.get("HF_HOME", "/workspace/hf")) / "hub" / REPO / "snapshots" / REV
    if not (root / "model.safetensors.index.json").exists():
        sys.exit(f"real_operands: no {root}/model.safetensors.index.json (prefetch.sh's weight cache)")
    return root


def tensor(root: Path, index: dict, name: str) -> np.ndarray:
    """One bf16 tensor as float32, read straight from its shard."""
    shard = root / index["weight_map"][name]
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
    if meta["dtype"] != "BF16":
        sys.exit(f"real_operands: {name} is {meta['dtype']}, not BF16")
    a, b = meta["data_offsets"]
    raw = np.memmap(shard, dtype=np.uint16, mode="r", offset=8 + n + a, shape=((b - a) // 2,))
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def quantize_rows(m: np.ndarray) -> np.ndarray:
    s = np.abs(m).max(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return np.clip(np.rint(m / s * 127.0), -127, 127).astype(np.int8)


def stats(q: np.ndarray) -> dict:
    v = q.reshape(-1)
    counts = np.bincount(v.view(np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / v.size
    ones = np.unpackbits(v.view(np.uint8)).mean() * 8
    return dict(n=int(v.size), bits_per_byte=float(-(p * np.log2(p)).sum()), zero_share=float(counts[0] / v.size),
                mean_abs=float(np.abs(v.astype(np.int32)).mean()), sd=float(v.astype(np.float64).std()),
                set_bits_per_byte=float(ones))


def main() -> None:
    out = Path(sys.argv[1])
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
    root = snapshot()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    w = np.concatenate([tensor(root, index, f"model.layers.0.mlp.{p}_proj.weight") for p in ("gate", "up")])
    qw = quantize_rows(w)
    emb = tensor(root, index, "model.embed_tokens.weight")
    g = tensor(root, index, "model.layers.0.input_layernorm.weight")
    ids = np.random.default_rng(0).integers(0, emb.shape[0], tokens)
    h = emb[ids]
    h = h / np.sqrt((h * h).mean(axis=1, keepdims=True) + 1e-5) * g
    qx = quantize_rows(h)
    out.mkdir(parents=True, exist_ok=True)
    qw.tofile(out / "w.s8")
    qx.tofile(out / "x.s8")
    print(json.dumps(dict(model=f"unsloth/Llama-3.1-8B@{REV}", w=dict(stats(qw), shape=list(qw.shape)),
                          x=dict(stats(qx), shape=list(qx.shape)))))


if __name__ == "__main__":
    main()
