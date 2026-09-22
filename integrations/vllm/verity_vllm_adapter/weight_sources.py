"""Weights by reference to the checkpoint (CONTRACT §16.8: "weights referenced by checkpoint sha, never
copied").

A replay run with ``--weights-values checkpoint`` hashes every parameter as before (the sha256 and
leaf of the exact GPU bytes go into ``weights/index.json``) but stores the bytes only for tensors that
are **not** in the checkpoint file (vLLM buffers: the rotary ``cos_sin_cache``, the attention scales).
For every checkpoint tensor the entry records its *source*: the safetensors file, its sha256 (the
manifest's), the HF tensor names and the concatenation vLLM performs when loading (``qkv_proj`` =
``q_proj ‖ k_proj ‖ v_proj``, ``gate_up_proj`` = ``gate_proj ‖ up_proj``, both along dim 0). The
verifier's resolver (:func:`make_resolver`) rebuilds the bytes from a local copy of the checkpoint
whose file sha256 must equal the recorded one; :class:`verity_vllm.capture.WeightsReader` then
re-hashes the rebuilt bytes and refuses anything that does not match the index entry, so a wrong
mapping can only fail, never pass.

The mapping covers the Llama family as vLLM names its parameters (SmolLM2, Llama, Qwen2 without
biases); anything unmapped is stored.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable

_LAYER = re.compile(r"^layers\.(\d+)\.(.*)$")

_DIRECT = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}
_LAYER_DIRECT = {
    "input_layernorm.weight": "input_layernorm.weight",
    "post_attention_layernorm.weight": "post_attention_layernorm.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    "self_attn.o_proj.bias": "self_attn.o_proj.bias",
    "mlp.down_proj.bias": "mlp.down_proj.bias",
}
_LAYER_CAT = {
    "self_attn.qkv_proj.weight": ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"],
    "self_attn.qkv_proj.bias": ["self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"],
    "mlp.gate_up_proj.weight": ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
}


def hf_tensors(param: str) -> tuple[list[str], int] | None:
    """HF safetensors tensor names (in concatenation order) and the cat dim for a vLLM parameter
    name, or None when the parameter is not a checkpoint tensor."""
    if param in _DIRECT:
        return [_DIRECT[param]], 0
    m = _LAYER.match(param)
    if not m:
        return None
    n, rest = m.group(1), m.group(2)
    if rest in _LAYER_DIRECT:
        return [f"model.layers.{n}.{_LAYER_DIRECT[rest]}"], 0
    if rest in _LAYER_CAT:
        return [f"model.layers.{n}.{t}" for t in _LAYER_CAT[rest]], 0
    return None


def source_for(param: str, checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """The ``WeightEntry.source`` document for a checkpoint tensor (None: store the bytes)."""
    hf = hf_tensors(param)
    if hf is None:
        return None
    names, dim = hf
    files = checkpoint.get("safetensors_sha256") or {}
    if len(files) != 1:
        return None  # sharded checkpoints: not mapped here (stored)
    (fname, sha), = files.items()
    return {"kind": "checkpoint", "repo": checkpoint.get("repo"), "revision": checkpoint.get("revision"),
            "file": fname, "file_sha256": sha, "tensors": names, "cat_dim": dim}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_row_words(model_dir: str | Path, param: str, row: int, *, expected_file_sha256: dict[str, str] | None = None,
                         file: str = "model.safetensors") -> tuple[bytes, dict[str, Any]]:
    """The raw little-endian bytes of row ``row`` of vLLM parameter ``param`` (an ``[N, K]`` tensor) read
    from the checkpoint's safetensors file **without torch**: the file header is parsed (8-byte length,
    JSON), the HF tensors that vLLM concatenates into ``param`` (:func:`hf_tensors`, cat dim 0) are
    located, the row is mapped to the tensor holding it, and only that row's bytes are read.  The file
    is hashed once and must equal ``expected_file_sha256[file]`` when given.  Returns ``(bytes, source)``
    where ``source`` names the file, its sha256, the HF tensor, its dtype and the local row -- what a
    sampled ``Input`` equation (IR §E) records as the registered value's provenance.

    This is the verifier's reference for a weight word: the pinned checkpoint, not the commitment."""
    import json
    import struct

    hf = hf_tensors(param)
    if hf is None:
        raise KeyError(f"{param}: not a checkpoint tensor")
    names, dim = hf
    if dim != 0:
        raise KeyError(f"{param}: concatenation along dim {dim} is not row-addressable here")
    path = Path(model_dir) / file
    sha = sha256_file(path.resolve())
    if expected_file_sha256 and expected_file_sha256.get(file) not in (None, sha):
        raise ValueError(f"{path}: sha256 {sha} != expected {expected_file_sha256.get(file)}")
    with path.open("rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        remaining = int(row)
        for name in names:
            meta = header[name]
            shape = [int(s) for s in meta["shape"]]
            if len(shape) != 2:
                raise KeyError(f"{name}: shape {shape} is not [N, K]")
            n, k = shape
            if remaining < n:
                dtype = meta["dtype"]
                width = {"BF16": 2, "F16": 2, "F32": 4}.get(dtype)
                if width is None:
                    raise KeyError(f"{name}: dtype {dtype} not handled")
                start, end = meta["data_offsets"]
                if end - start != n * k * width:
                    raise ValueError(f"{name}: data_offsets do not match shape")
                f.seek(base + start + remaining * k * width)
                data = f.read(k * width)
                return data, {"file": file, "file_sha256": sha, "tensor": name, "dtype": dtype, "shape": shape, "local_row": remaining,
                              "param": param, "row": int(row)}
            remaining -= n
    raise KeyError(f"{param}: row {row} beyond the concatenated tensors")


def make_resolver(model_dir: str | Path, *, expected_file_sha256: dict[str, str] | None = None) -> Callable[[Any], bytes]:
    """A ``WeightsReader.resolver``: rebuilds a checkpoint tensor's bytes from ``model_dir``.
    Each safetensors file is hashed once and must equal both the entry's recorded sha and, when given,
    ``expected_file_sha256[file]`` (the checkpoints manifest)."""
    import torch
    from safetensors import safe_open

    model_dir = Path(model_dir)
    verified: dict[str, str] = {}
    handles: dict[str, Any] = {}

    def resolve(entry: Any) -> bytes:
        src = entry.source or {}
        if src.get("kind") != "checkpoint":
            raise KeyError(f"{entry.canonical_id}: no checkpoint source recorded")
        fname = src["file"]
        if fname not in verified:
            p = model_dir / fname
            sha = sha256_file(p.resolve())
            if sha != src["file_sha256"]:
                raise ValueError(f"{p}: sha256 {sha} != recorded {src['file_sha256']}")
            if expected_file_sha256 and expected_file_sha256.get(fname) not in (None, sha):
                raise ValueError(f"{p}: sha256 {sha} != checkpoints manifest {expected_file_sha256.get(fname)}")
            verified[fname] = sha
            handles[fname] = safe_open(str(p), framework="pt", device="cpu")
        f = handles[fname]
        parts = [f.get_tensor(t) for t in src["tensors"]]
        t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=int(src.get("cat_dim", 0)))
        t = t.contiguous().reshape(-1)
        return t.view(torch.uint8).numpy().tobytes()

    return resolve
