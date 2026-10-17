"""The models the bench runs: a snapshot of each, pinned by revision, under the node's HF cache. `gpus` is how many the
standard run leases; the engine reads one layer at a time, so every model fits one GPU and more GPUs split the windows
(Llama-3.1-70B: 141 GB of bf16 weights, read layer by layer, on 2 GPUs with no `accelerate`)."""
from __future__ import annotations

from pathlib import Path

MODELS = {
    "llama-3.1-8b": {"repo": "unsloth/Llama-3.1-8B", "revision": "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814",
                     "tokenizer": "llama3", "gpus": 1},
    "qwen2.5-7b": {"repo": "Qwen/Qwen2.5-7B", "revision": "d149729398750b98c0af14eb82c78cfe92750796",
                   "tokenizer": "qwen2.5", "gpus": 1},
    "llama-3.1-70b": {"repo": "unsloth/Meta-Llama-3.1-70B", "revision": "1b7306651142d0cc65d993076a250a6a82cf046c",
                      "tokenizer": "llama3", "gpus": 2},
}


def get(name: str) -> dict:
    if name not in MODELS:
        raise ValueError(f"model {name!r}: the registry has {sorted(MODELS)}")
    return MODELS[name]


def snapshot(name: str, hf_home: Path) -> Path:
    m = get(name)
    org, repo = m["repo"].split("/")
    return Path(hf_home) / "hub" / f"models--{org}--{repo}" / "snapshots" / m["revision"]
