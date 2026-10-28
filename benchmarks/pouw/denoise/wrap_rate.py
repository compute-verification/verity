"""W′'s wrap rate on the served int8 Llama-3.2-1B, exactly: the expected fraction of W′ = wrap8(B + d_B + U π_τ) entries that
leave [-128, 127], per linear, with d_B uniform on {0, 1} and U uniform on -80..80 (Dither v3.2; v31's noise_kernel). It is a
function of B's histogram alone, so no coins are drawn.

B is the checkpoint `verity-int8/Llama-3.2-1B-INT8` (recipe verity-int8-channel127/v1, integrations/vllm's
manifests/checkpoints-int8.json) where an HF cache holds it, else that recipe applied here to its base, unsloth/Llama-3.2-1B at
9535bd9b, from the cache or fetched from the hub. Recipe v1 per row: s = max(max |W|, RN32(1e-30)) / 127 in float32,
q = clamp(rint(W / s), -127, 127), on every 2-D tensor none of whose dimensions is the vocabulary.

    python3 wrap_rate.py OUT.json        (numpy; reads $HF_HOME/hub and /workspace/hf/hub)
"""
from __future__ import annotations

import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

NOISE = 80
INT8_REPO, BASE_REPO, BASE_REV = "verity-int8--Llama-3.2-1B-INT8", "unsloth--Llama-3.2-1B", "9535bd9b1d1dea6acafbdc4813b728796aeb28da"
VOCAB = 128256
LINEARS = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o", "gate_proj": "gu", "up_proj": "gu",
           "down_proj": "down"}


def wrap_prob() -> np.ndarray:
    """p[b + 128]: the chance b + d + u wraps, over d in {0, 1} and u in -80..80."""
    b = np.arange(-128, 128)[:, None]
    hi = np.clip(b + np.array([0, 1]) - (127 - NOISE), 0, 2 * NOISE + 1)
    lo = np.clip(-(128 - NOISE) - (b + np.array([0, 1])), 0, 2 * NOISE + 1)
    return (hi + lo).sum(axis=1) / (2 * (2 * NOISE + 1))


def tensors(path: Path):
    """safetensors' (name, dtype, shape, the tensor as a memmap) without the safetensors package."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
    base = 8 + n
    dt = {"I8": np.int8, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}
    for name, t in head.items():
        if name == "__metadata__" or t["dtype"] not in dt:
            continue
        a, b = t["data_offsets"]
        yield name, t["dtype"], t["shape"], np.memmap(path, dtype=dt[t["dtype"]], mode="r", offset=base + a,
                                                      shape=tuple(t["shape"]))


def quant_rows(w: np.ndarray) -> np.ndarray:
    s = np.maximum(np.abs(w).max(axis=1), np.float32(1e-30)) / np.float32(127)
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int16)


def as_f32(dtype: str, x: np.ndarray) -> np.ndarray:
    if dtype == "BF16":
        return (x.astype(np.uint32) << 16).view(np.float32)
    return x.astype(np.float32)


def snapshot(hubs: list[Path], repo: str) -> Path | None:
    for h in hubs:
        for p in sorted((h / f"models--{repo}" / "snapshots").glob("*/model.safetensors")):
            return p
    return None


def main() -> int:
    out = Path(sys.argv[1])
    hubs = [Path(p) / "hub" for p in (os.environ.get("HF_HOME", ""), "/workspace/hf") if p] + [Path.home() / ".cache/huggingface/hub"]
    src = snapshot(hubs, INT8_REPO)
    made = src is None
    if made:
        src = snapshot(hubs, BASE_REPO)
    if src is None:
        src = out.parent / "base.safetensors"
        url = f"https://huggingface.co/{BASE_REPO.replace('--', '/')}/resolve/{BASE_REV}/model.safetensors"
        print(f"fetching {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, src)
    p = wrap_prob()
    per, by = [], {}
    for name, dtype, shape, x in tensors(src):
        mod = name.split(".")[-2] if name.endswith(".weight") else ""
        if len(shape) != 2 or VOCAB in shape or mod not in LINEARS:
            continue
        q = (x.astype(np.int16) if dtype == "I8" else quant_rows(as_f32(dtype, np.asarray(x))))
        h = np.bincount((q + 128).ravel(), minlength=256)
        rate = float(h @ p / h.sum())
        per.append({"tensor": name, "shape": shape, "rate": rate, "can_wrap": float(h[p > 0].sum() / h.sum()),
                    "std": float(np.sqrt(((np.arange(-128, 128) ** 2) * h).sum() / h.sum()))})
        g = by.setdefault(LINEARS[mod], [0.0, 0])
        g[0] += rate * h.sum()
        g[1] += int(h.sum())
    rep = {"source": str(src), "made_here": made, "noise": NOISE,
           "by_linear": {k: v[0] / v[1] for k, v in by.items()},
           "all": sum(v[0] for v in by.values()) / sum(v[1] for v in by.values()), "tensors": per}
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("source", "made_here", "by_linear", "all")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
