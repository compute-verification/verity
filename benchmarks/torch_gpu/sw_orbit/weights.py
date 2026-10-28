"""Llama 3.1 8B's int8 linears for one layer, as the served pin makes them: the BF16 base `unsloth/Llama-3.1-8B` at its pinned
revision (integrations/vllm/manifests/checkpoints.json, role LLAMA31_8B), each weight read from the hub by HTTP range
requests (only that tensor's bytes), and the `verity-int8-channel127/v1` row rule (`int8_checkpoint.channel_scales` and
`quantize`): s = max(max |W[n, :]|, 1e-30) / 127 in float32, q = clamp(rint(W / s), −127, 127). The fused linears are
vLLM's: qkv = [q; k; v] (6,144 x 4,096), o (4,096 x 4,096), gate_up = [gate; up] (28,672 x 4,096), down (4,096 x 14,336).

    python weights.py --layers 0,15 --dir DIR      # DIR/l{L}_{qkv,o,gate_up,down}.npy, int8 [N, K]
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO, REVISION = "unsloth/Llama-3.1-8B", "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
LINEARS = {"qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), "o": ("self_attn.o_proj",),
           "gate_up": ("mlp.gate_proj", "mlp.up_proj"), "down": ("mlp.down_proj",)}


def _get(name: str, lo: int | None = None, hi: int | None = None) -> bytes:
    req = urllib.request.Request(URL + name)  # noqa: S310 - URL is the pinned https base
    if lo is not None:
        req.add_header("Range", f"bytes={lo}-{hi - 1}")
    with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310
        data = r.read()
    if lo is not None and len(data) != hi - lo:
        raise RuntimeError(f"{name}: asked bytes {lo}..{hi}, got {len(data)}")
    return data


def _headers() -> tuple[dict, dict]:
    index = json.loads(_get("model.safetensors.index.json"))["weight_map"]
    heads = {}
    for shard in sorted(set(index.values())):
        n = struct.unpack("<Q", _get(shard, 0, 8))[0]
        heads[shard] = (8 + n, json.loads(_get(shard, 8, 8 + n)))
    return index, heads


def bf16_tensor(name: str, index: dict, heads: dict) -> np.ndarray:
    shard = index[name]
    base, head = heads[shard]
    meta = head[name]
    if meta["dtype"] != "BF16":
        raise ValueError(f"{name}: {meta['dtype']}, not BF16")
    lo, hi = meta["data_offsets"]
    raw = np.frombuffer(_get(shard, base + lo, base + hi), dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])


def quantize_rows(w: np.ndarray) -> np.ndarray:
    """int8_checkpoint.channel_scales then quantize: the served pin's rule."""
    amax = np.abs(w).max(axis=1, keepdims=True)
    s = (np.maximum(amax, np.float32(1e-30)) / np.float32(127)).astype(np.float32)
    return np.clip(np.rint(w / s), -127, 127).astype(np.int8)


def layer(i: int, index: dict, heads: dict) -> dict[str, np.ndarray]:
    out = {}
    for key, parts in LINEARS.items():
        out[key] = np.concatenate([quantize_rows(bf16_tensor(f"model.layers.{i}.{p}.weight", index, heads)) for p in parts])
    return out


def load(dir_: Path, layers: list[int]) -> dict[tuple[int, str], np.ndarray]:
    """Each (layer, linear)'s int8 weight, fetched and quantized once into dir_."""
    dir_.mkdir(parents=True, exist_ok=True)
    out, missing = {}, []
    for i in layers:
        for key in LINEARS:
            p = dir_ / f"l{i}_{key}.npy"
            if p.exists():
                out[(i, key)] = np.load(p)
            else:
                missing.append(i)
    if missing:
        index, heads = _headers()
        for i in sorted(set(missing)):
            for key, w in layer(i, index, heads).items():
                np.save(dir_ / f"l{i}_{key}.npy", w)
                out[(i, key)] = w
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0")
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    ws = load(a.dir, [int(x) for x in a.layers.split(",")])
    for (i, key), w in sorted(ws.items()):
        print(i, key, w.shape, int(np.abs(w.astype(np.int16)).max()))


if __name__ == "__main__":
    main()
