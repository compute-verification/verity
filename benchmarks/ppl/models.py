"""The models the bench runs: a snapshot of each, pinned by revision, under the node's HF cache. `gpus` is how many the
standard run leases; the engine reads one layer at a time, so every model fits one GPU and more GPUs split the windows
(Llama-3.1-70B: 141 GB of bf16 weights, read layer by layer, on 2 GPUs with no `accelerate`).

A model with a `pin` (`pins/<name>.json`: its repo, revision and every file's sha256) is streamed: `ensure` puts its small
files in the snapshot from the hub, each checked against the pin, and the engine pulls each safetensors shard the layer
it reads needs, checks its sha256 and deletes it once read (`stream.Hub`). So GLM-5.3's 705 GB of block-FP8 weights need
no copy on the node. `max_ctx`: the longest context a model runs at (GLM-5.3's DSA attention is dense, and the engine's
layer HF's, only up to its index_topk, 2048)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    "llama-3.1-8b": {"repo": "unsloth/Llama-3.1-8B", "revision": "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814",
                     "tokenizer": "llama3", "gpus": 1},
    "qwen2.5-7b": {"repo": "Qwen/Qwen2.5-7B", "revision": "d149729398750b98c0af14eb82c78cfe92750796",
                   "tokenizer": "qwen2.5", "gpus": 1},
    "llama-3.1-70b": {"repo": "unsloth/Meta-Llama-3.1-70B", "revision": "1b7306651142d0cc65d993076a250a6a82cf046c",
                      "tokenizer": "llama3", "gpus": 2},
    "glm-5.3": {"repo": "zai-org/GLM-5.3", "revision": "aca966e4e02791568aa6a4ced368624b3d897f42",
                "tokenizer": "glm5", "gpus": 1, "pin": "glm-5.3.json", "max_ctx": 2048},
}


def get(name: str) -> dict:
    if name not in MODELS:
        raise ValueError(f"model {name!r}: the registry has {sorted(MODELS)}")
    return MODELS[name]


def snapshot(name: str, hf_home: Path) -> Path:
    m = get(name)
    org, repo = m["repo"].split("/")
    return Path(hf_home) / "hub" / f"models--{org}--{repo}" / "snapshots" / m["revision"]


def pin(name: str) -> dict | None:
    """A streamed model's pin (repo, revision, each file's sha256), or None."""
    m = get(name)
    if "pin" not in m:
        return None
    p = json.loads((HERE / "pins" / m["pin"]).read_text())
    if (p["repo"], p["revision"]) != (m["repo"], m["revision"]):
        raise ValueError(f"pins/{m['pin']} is {p['repo']}@{p['revision']}, not the registry's {m['repo']}@{m['revision']}")
    return p


def pin_of(snap: Path) -> dict | None:
    """The pin of the streamed model whose snapshot directory `snap` is (any HF home), or None."""
    snap = Path(snap)
    if len(snap.parents) < 4:
        return None
    for name, m in MODELS.items():
        if "pin" in m and snap == snapshot(name, snap.parents[3]):
            return pin(name)
    return None
