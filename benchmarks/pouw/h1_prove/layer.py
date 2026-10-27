"""A recorded H1 layer: the int8 codes the H1 numerics (co32c-u16; `edited_gemm/cgemm.cu` under
``CG_NUMERICS=h1``) serve for one decoder layer's K = 4096 linears, from a Llama-3.1-8B checkpoint on the CPU.

* **Weights.** Each linear's rows (q, k, v, gate, up of layer L) quantized per output row to H1's ±110 codes:
  W = round(w · 110 / max|w_row|), clamped to [-110, 110] (cgemm.cu: "the linears' W in [-110, 110]"), its scale
  s_w = max|w_row| / 110.
* **Activations.** The layer's input for M prompt tokens, x = RMSNorm(embed(tokens)) · g (layer 0's input norm, ε 1e-5,
  float32), quantized per token as H1's activation quantizer (cgemm.cu's ``cquant_h1``): s_a = max(|x_row|, 1e-30) / 126,
  A = floor(x / s_a), clamped to [-126, 125], so the unit's dithered X = A + b (b a noise bit) is an s8 code in [-126, 126].
* **``--quant vllm``** quantizes as vLLM's int8 w8a8 path does instead (``scaled_int8_quant``, symmetric): activations
  per token and weights per output row, s = max|row| / 127, code = rint(x / s) clamped to [-128, 127], so every row's
  absmax entry is ±127 and the unit's X = A + b wraps (X = -128) wherever A = 127 meets b = 1. ``--quant-k K`` takes each
  row's absmax over its first K entries, the row a ``--k K`` smoke's unit sees, so those ±127 entries are in its tiles.
* **The scales** are each row's f32 s (``sa`` the tokens', ``s_<call>`` each linear's rows'), which the window commits
  beside the row's codes and the served epilogue reads: y = RN_bf16(RN32(RN32(RN32(z) · s_a) · s_w)) (`h1zy.dequant`).
* **The checkpoint** is read with numpy alone (safetensors' header and raw little-endian bf16), from the HF cache.

    python layer.py MODEL_DIR OUT.npz [--tokens 16] [--layer 0] [--seed-tokens HEX] [--fetch REPO] [--quant vllm [--quant-k K]]

``--fetch REPO`` (a node without the checkpoint): MODEL_DIR gets REPO's index, tokenizer and only the shards holding the
embedding and layer L's tensors, from huggingface.co, where they are not there already.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np

CALLS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
W_TOP, A_TOP = 110, 126
EPS = 1e-5


class Checkpoint:
    def __init__(self, root: Path) -> None:
        self.root = root
        idx = root / "model.safetensors.index.json"
        if idx.exists():
            self.where = json.loads(idx.read_text())["weight_map"]
        else:
            self.where = {k: "model.safetensors" for k in self._header(root / "model.safetensors")[0] if k != "__metadata__"}
        self._heads: dict[str, tuple[dict, int]] = {}

    @staticmethod
    def _header(f: Path) -> tuple[dict, int]:
        with open(f, "rb") as h:
            n = struct.unpack("<Q", h.read(8))[0]
            return json.loads(h.read(n)), 8 + n

    def tensor(self, name: str, rows: list[int] | None = None) -> np.ndarray:
        f = self.root / self.where[name]
        if f.name not in self._heads:
            self._heads[f.name] = self._header(f)
        head, base = self._heads[f.name]
        e = head[name]
        if e["dtype"] != "BF16":
            raise SystemExit(f"{name}: {e['dtype']}, not BF16")
        lo, hi = e["data_offsets"]
        mm = np.memmap(f, dtype="<u2", mode="r", offset=base + lo, shape=((hi - lo) // 2,)).reshape(e["shape"])
        raw = np.asarray(mm if rows is None else mm[rows])
        return (raw.astype(np.uint32) << 16).view(np.float32)


def find_model(hub: Path) -> Path:
    for name in ("models--unsloth--Llama-3.1-8B", "models--meta-llama--Llama-3.1-8B", "models--unsloth--Meta-Llama-3.1-8B-Instruct"):
        snaps = sorted((hub / name / "snapshots").glob("*/"))
        for s in snaps:
            if list(s.glob("*.safetensors")):
                return s
    raise SystemExit(f"no Llama-3.1-8B bf16 checkpoint under {hub}")


def fetch(repo: str, root: Path, layer: int, calls: list[str]) -> None:
    import urllib.request
    root.mkdir(parents=True, exist_ok=True)

    def get(name: str) -> Path:
        f = root / name
        if not f.exists():
            tmp = f.with_suffix(f"{f.suffix}.{os.getpid()}.part")
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{name}") as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 24):
                    out.write(chunk)
            os.replace(tmp, f)
        return f

    where = json.loads(get("model.safetensors.index.json").read_text())["weight_map"]
    get("tokenizer.json")
    names = ["model.embed_tokens.weight", f"model.layers.{layer}.input_layernorm.weight"] + [
        f"model.layers.{layer}.{'self_attn' if c.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj')) else 'mlp'}.{c}.weight"
        for c in calls]
    for shard in sorted({where[n] for n in names}):
        get(shard)


def quant_w(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H1's weight codes of each row, and each row's f32 scale."""
    amax = np.maximum(np.abs(w).max(axis=1, keepdims=True), np.float32(1e-30))
    return np.clip(np.rint(w * (W_TOP / amax)), -W_TOP, W_TOP).astype(np.int8), (amax[:, 0] / W_TOP).astype(np.float32)


def quant_a(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H1's activation codes of each row, and each row's f32 scale."""
    s = (np.maximum(np.abs(x).max(axis=1, keepdims=True), np.float32(1e-30)) / A_TOP).astype(np.float32)
    return np.clip(np.floor(x / s), -A_TOP, A_TOP - 1).astype(np.int8), s[:, 0]


def quant_vllm(x: np.ndarray, k: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """vLLM's symmetric int8 quantization of each row, its scale (f32) from the row's first ``k`` entries (0: all)."""
    # With k (a K = 64 smoke) the absmax is over those k entries so ±127 lands inside the tiles: not vLLM's full-row scale.
    x = x[:, :k] if k else x
    s = (np.maximum(np.abs(x).max(axis=1, keepdims=True), np.float32(1e-30)) / 127).astype(np.float32)
    return np.clip(np.rint(x / s), -128, 127).astype(np.int8), s[:, 0]


def tokens_of(model: Path, m: int, seed: bytes | None) -> list[int]:
    """M prompt tokens: ids from ``seed`` where one is given, else the tokenizer's ids of a fixed prompt where `tokenizers` is
    installed, else ids from a fresh seed."""
    if seed is not None:
        return _seeded(m, seed)
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(model / "tokenizer.json"))
        ids = [128000] + tok.encode("The verifier draws tiles of the served window by a live coin, and each drawn tile is "
                                    "proved in zero knowledge and checked by the Lean verifier.", add_special_tokens=False).ids
        if len(ids) >= m:
            return ids[:m]
    except Exception:  # noqa: BLE001 - the tokenizer is optional: ids from the seed instead
        pass
    return _seeded(m, os.urandom(16))


def _seeded(m: int, seed: bytes) -> list[int]:
    h = hashlib.shake_256(b"verity/pouw/h1-prove/tokens/v0\0" + seed).digest(4 * m)
    return [128000] + [int.from_bytes(h[4 * i:4 * i + 4], "big") % 128000 for i in range(1, m)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="the checkpoint's snapshot directory, or the HF hub directory to find it in")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--seed-tokens", default="")
    ap.add_argument("--calls", default=",".join(CALLS))
    ap.add_argument("--fetch", default="")
    ap.add_argument("--quant", choices=("h1", "vllm"), default="h1")
    ap.add_argument("--quant-k", type=int, default=0)
    a = ap.parse_args()
    qa, qw = (quant_a, quant_w) if a.quant == "h1" else ((lambda v: quant_vllm(v, a.quant_k)),) * 2
    root = Path(a.model)
    if a.fetch:
        fetch(a.fetch, root, a.layer, a.calls.split(","))
    if not list(root.glob("*.safetensors")) and not (root / "model.safetensors.index.json").exists():
        root = find_model(root)
    ck = Checkpoint(root)
    toks = tokens_of(root, a.tokens, bytes.fromhex(a.seed_tokens) if a.seed_tokens else None)
    emb = ck.tensor("model.embed_tokens.weight", toks)
    g = ck.tensor(f"model.layers.{a.layer}.input_layernorm.weight")
    x = emb / np.sqrt((emb.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + EPS).astype(np.float32) * g
    out = {"tokens": np.array(toks, np.int64)}
    out["a"], out["sa"] = qa(x)
    calls = a.calls.split(",")
    for c in calls:
        part = "self_attn" if c in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
        out[f"b_{c}"], out[f"s_{c}"] = qw(ck.tensor(f"model.layers.{a.layer}.{part}.{c}.weight"))
    np.savez(a.out, **out)
    print(json.dumps({"model": str(root), "layer": a.layer, "tokens": toks, "a": list(out["a"].shape),
                      "calls": {c: list(out[f"b_{c}"].shape) for c in calls}, "quant": a.quant, "quant_k": a.quant_k,
                      "a_range": [int(out["a"].min()), int(out["a"].max())],
                      "rows_with_a_127": int((out["a"] == 127).any(axis=1).sum())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
