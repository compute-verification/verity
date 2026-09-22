"""Natural GEMM operands for Qwen/Qwen2.5-1.5B: weights + activations from an HF forward pass (GPU host only).

Downloads the frozen checkpoint (contract §1) into ``HF_HOME``, records the revision and the
safetensors sha256, runs one BF16 forward pass over a ~2048-token text and saves, for the
layers requested, the *inputs* of every linear projection as raw BF16 words together with the
merged weights vLLM actually multiplies with (``qkv_proj`` = q|k|v rows, ``gate_up_proj`` =
gate|up rows) and their biases.  Output: one ``.npz`` per layer + ``meta.json`` under
``--out-dir`` (stays on the pod; never committed).

Usage: python -m verity_vllm_numerics.capture_activations --out-dir /workspace/results/numerics/qwen_acts --layers 0,13,27
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

REPO = "Qwen/Qwen2.5-1.5B"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def words(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="0,13,27")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--text", default=None, help="text file; default: repo docs concatenated")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    local = snapshot_download(REPO)
    revision = os.path.basename(os.path.realpath(local))
    shas = {f: sha256_file(os.path.join(local, f)) for f in sorted(os.listdir(local)) if f.endswith(".safetensors")}
    print("checkpoint", local, revision, shas, flush=True)

    tok = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16).cuda().eval()
    if args.text:
        text = open(args.text).read()
    else:
        parts = []
        for p in ("/workspace/veritor/docs/hardware-semantics.md", "/workspace/veritor/docs/vllm-poc/CONTRACT.md", "/workspace/veritor/README.md"):
            if os.path.exists(p):
                parts.append(open(p).read())
        text = "\n\n".join(parts) * 4
    ids = tok(text, return_tensors="pt").input_ids[:, : args.tokens].cuda()
    T = ids.shape[1]

    captured: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(mod, inp, out):
            captured[name] = inp[0].detach()[0].clone()  # [T, K]
        return fn

    handles = []
    for li in layers:
        blk = model.model.layers[li]
        handles.append(blk.self_attn.q_proj.register_forward_hook(hook(f"layers.{li}.qkv_proj")))
        handles.append(blk.self_attn.o_proj.register_forward_hook(hook(f"layers.{li}.o_proj")))
        handles.append(blk.mlp.gate_proj.register_forward_hook(hook(f"layers.{li}.gate_up_proj")))
        handles.append(blk.mlp.down_proj.register_forward_hook(hook(f"layers.{li}.down_proj")))
    handles.append(model.lm_head.register_forward_hook(hook("lm_head")))
    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()
    logits = out.logits[0]
    argmax = logits.argmax(-1).tolist()

    meta = {
        "repo": REPO,
        "revision": revision,
        "weights_sha256": shas,
        "tokens": T,
        "layers": layers,
        "config": {k: getattr(model.config, k) for k in ("hidden_size", "intermediate_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers", "tie_word_embeddings")},
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "argmax_first_16": argmax[:16],
        "gemms": {},
    }
    for li in layers:
        blk = model.model.layers[li]
        sa, mlp = blk.self_attn, blk.mlp
        qkv_w = torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], 0)
        qkv_b = torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias], 0)
        gu_w = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
        arrays = {
            "qkv_proj.x": words(captured[f"layers.{li}.qkv_proj"]),
            "qkv_proj.w": words(qkv_w),
            "qkv_proj.b": words(qkv_b),
            "o_proj.x": words(captured[f"layers.{li}.o_proj"]),
            "o_proj.w": words(sa.o_proj.weight),
            "gate_up_proj.x": words(captured[f"layers.{li}.gate_up_proj"]),
            "gate_up_proj.w": words(gu_w),
            "down_proj.x": words(captured[f"layers.{li}.down_proj"]),
            "down_proj.w": words(mlp.down_proj.weight),
        }
        np.savez(os.path.join(args.out_dir, f"layer{li}.npz"), **arrays)
        for name in ("qkv_proj", "o_proj", "gate_up_proj", "down_proj"):
            meta["gemms"][f"layers.{li}.{name}"] = {
                "x_shape": list(arrays[f"{name}.x"].shape),
                "w_shape": list(arrays[f"{name}.w"].shape),
                "bias": f"{name}.b" in arrays,
                "file": f"layer{li}.npz",
            }
    np.savez(os.path.join(args.out_dir, "lm_head.npz"), **{"lm_head.x": words(captured["lm_head"]), "lm_head.w": words(model.lm_head.weight)})
    meta["gemms"]["lm_head"] = {"x_shape": list(captured["lm_head"].shape), "w_shape": list(model.lm_head.weight.shape), "bias": False, "file": "lm_head.npz"}
    meta["seconds"] = time.time() - t0
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "gemms"}, indent=1), flush=True)
    print("gemms", json.dumps(meta["gemms"], indent=1), flush=True)


if __name__ == "__main__":
    main()
