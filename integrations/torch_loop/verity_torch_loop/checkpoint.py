"""The loop's model as an int8 checkpoint role, `ROLE`: the files the auditor's gate registers
(`benchmarks/live_challenger/models.py`), so it can sign the loop's weights root.

* **The files** (`CHECKPOINT`): ``model.safetensors`` holds `loop.Model`'s weights as the loop serves them (`loop.Model()`,
  which `serve` builds at its default width): each linear's s8 codes ``<op path>.weight`` (I8, H × H, one row per output
  channel) and its s_w ``<op path>.weight_scale`` (F32, H), and the embedding ``model.embed_tokens.weight`` (F32, vocab × H).
  The safetensors header lists the tensors in name order, with no metadata, and the data follows in that order, so the bytes
  depend on the weights alone. ``config.json`` gives the model's shape and seed.
* **The entry** (`MANIFEST`): the role, a repo id (a local namespace: no hub repo holds these bytes), its revision (the first
  40 hex digits of the files' checkpoint digest, `checkpoint_sha512`, since no hub revision exists), each file's size and
  sha256, the recipe (the loop's width, vocabulary and seed), and ``linears``: the loop's op paths in its order
  (`loop.Model.op_path`), which the gate requires the window's linears to be.

``python -m verity_torch_loop.checkpoint`` writes both from the model; ``--check`` exits 1 when the files differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.circuits.codec import canonical_json

from . import loop as L

ROLE = "TORCH_LOOP_MLP_H128_W8A8"
REPO_ID = "verity-torch-loop/mlp-w8a8-h128"
RECIPE = "torch-loop/mlp-w8a8/v1"
SCHEMA = "verity-torch-loop/checkpoints/v1"
ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoint"
MANIFEST = ROOT / "manifests" / "checkpoints-int8.json"
EMBED = "model.embed_tokens.weight"
#: the safetensors dtypes the gate reads (models.py's ``Checkpoint.DTYPES``)
DTYPES = {np.dtype(np.int8): "I8", np.dtype("<f4"): "F32"}


def tensors(model: L.Model) -> dict[str, np.ndarray]:
    """The model's weights by checkpoint name: the embedding, and each linear's s8 codes and s_w under its op path."""
    out = {EMBED: model.embed.numpy()}
    for i, layer in enumerate(model.layers):
        out[f"{model.op_path(i)}.weight"] = layer.wq.numpy()
        out[f"{model.op_path(i)}.weight_scale"] = layer.sw.numpy()
    return out


def safetensors(arrays: dict[str, np.ndarray]) -> bytes:
    """``arrays`` in safetensors' layout: the header's length (u64 LE), the header (JSON, names in order, padded with spaces to
    8 bytes), then each tensor's bytes in name order."""
    header, data, at = {}, [], 0
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        b = a.tobytes()
        header[name] = {"dtype": DTYPES[a.dtype], "shape": list(a.shape), "data_offsets": [at, at + len(b)]}
        data.append(b)
        at += len(b)
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h + b"".join(data)


def config(model: L.Model) -> dict[str, Any]:
    return {"model_type": "torch-loop-mlp-w8a8", "hidden_size": model.h, "num_hidden_layers": len(model.layers),
            "vocab_size": int(model.embed.shape[0]), "seed": model.seed}


def files(model: L.Model) -> dict[str, bytes]:
    """The role's files: ``config.json`` and ``model.safetensors``."""
    return {"config.json": (json.dumps(config(model), indent=1, sort_keys=True) + "\n").encode(),
            "model.safetensors": safetensors(tensors(model))}


def checkpoint_sha512(blobs: dict[str, bytes]) -> str:
    """The files' checkpoint digest, the gate's (models.py's `checkpoint_sha512`): SHA-512 of the canonical JSON of
    ``{file: SHA-512 hex}``, the value core's pin of the role holds."""
    return hashlib.sha512(canonical_json({n: hashlib.sha512(b).hexdigest() for n, b in blobs.items()})).hexdigest()


def entry(model: L.Model, blobs: dict[str, bytes]) -> dict[str, Any]:
    """The role's manifest entry, in the form the gate reads (models.py's `manifest_entry` and `check_files`)."""
    return {"role": ROLE, "repo": REPO_ID, "revision": checkpoint_sha512(blobs)[:40],
            "files": {n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in sorted(blobs.items())},
            "linears": [model.op_path(i) for i in range(len(model.layers))],
            "recipe": {"id": RECIPE, "h": model.h, "vocab": int(model.embed.shape[0]), "seed": model.seed},
            "note": "the torch loop's served weights (integrations/torch_loop, `loop.Model()`), written by "
                    "`python -m verity_torch_loop.checkpoint`; no hub repo holds them"}


def manifest(model: L.Model) -> bytes:
    doc = {"schema": SCHEMA, "checkpoints": [entry(model, files(model))]}
    return (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()


def wanted(model: L.Model) -> dict[Path, bytes]:
    return {**{CHECKPOINT / n: b for n, b in files(model).items()}, MANIFEST: manifest(model)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if a file is not the model's")
    a = ap.parse_args(argv)
    want = wanted(L.Model())
    if a.check:
        bad = [str(p) for p, b in want.items() if not p.is_file() or p.read_bytes() != b]
        print(json.dumps({"role": ROLE, "differ": bad}))
        return 1 if bad else 0
    for p, b in want.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    print(json.dumps({"role": ROLE, "checkpoint_sha512": checkpoint_sha512(files(L.Model())),
                      "wrote": [str(p.relative_to(ROOT)) for p in want]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
