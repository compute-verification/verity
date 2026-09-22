"""Freeze the checkpoints of CONTRACT.md §1 into ``manifests/checkpoints.json``.

For each repo: resolve the current ``main`` commit, download the snapshot (unless
``--resolve-only``), and record the sha256 of every ``*.safetensors`` shard and of
``config.json``. The Hub's LFS metadata carries a sha256 per file, so for
resolve-only repos the shard digests come from the Hub and are labelled as such;
downloaded repos are hashed locally from the bytes on disk.

Run ON THE POD::

    HF_HOME=/workspace/hf python checkpoints.py --out /workspace/results/checkpoints.json \
        --download HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-1.5B \
        --resolve-only microsoft/phi-2 Qwen/Qwen2.5-1.5B-Instruct-AWQ allenai/OLMoE-1B-7B-0924
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def sha256_file(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def resolve(api: HfApi, repo: str, download: bool) -> dict:
    info = api.model_info(repo, files_metadata=True)
    sha = info.sha
    entry: dict = {
        "repo": repo,
        "revision": sha,
        "gated": bool(info.gated),
        "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "downloaded": download,
        "files": {},
    }
    hub_digests = {}
    hub_sizes = {}
    for s in info.siblings or []:
        hub_sizes[s.rfilename] = s.size
        if s.lfs is not None:
            hub_digests[s.rfilename] = s.lfs.sha256
    wanted = [
        f
        for f in hub_sizes
        if f.endswith(".safetensors")
        or f in ("config.json", "model.safetensors.index.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json")
        or f.endswith("quant_config.json")
        or f.endswith("quantize_config.json")
    ]
    if download:
        local = Path(
            snapshot_download(
                repo,
                revision=sha,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab*"],
            )
        )
        entry["local_path"] = str(local)
        for f in sorted(wanted):
            p = local / f
            if not p.exists():
                continue
            digest = sha256_file(p)
            rec = {"sha256": digest, "size": p.stat().st_size, "digest_source": "local"}
            if f in hub_digests and hub_digests[f] != digest:
                rec["hub_lfs_sha256_mismatch"] = hub_digests[f]
            entry["files"][f] = rec
    else:
        for f in sorted(wanted):
            rec = {"size": hub_sizes[f]}
            if f in hub_digests:
                rec["sha256"] = hub_digests[f]
                rec["digest_source"] = "hub-lfs-metadata"
            else:
                rec["sha256"] = None
                rec["digest_source"] = "not-lfs; hash on download"
            entry["files"][f] = rec
    entry["safetensors_sha256"] = {
        f: r["sha256"] for f, r in entry["files"].items() if f.endswith(".safetensors")
    }
    entry["config_sha256"] = entry["files"].get("config.json", {}).get("sha256")
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--download", nargs="*", default=[])
    ap.add_argument("--resolve-only", nargs="*", default=[])
    args = ap.parse_args()
    api = HfApi()
    roles = {
        "HuggingFaceTB/SmolLM2-135M": "B0",
        "Qwen/Qwen2.5-1.5B": "B1-B4,R*,D*,Q1/Q2 base,X1",
        "microsoft/phi-2": "B5",
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ": "Q3",
        "allenai/OLMoE-1B-7B-0924": "M1/M2",
    }
    out = {
        "schema": "verity-vllm/checkpoints/v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_home": os.environ.get("HF_HOME"),
        "checkpoints": [],
    }
    for repo in args.download:
        e = resolve(api, repo, download=True)
        e["role"] = roles.get(repo)
        out["checkpoints"].append(e)
        print(f"downloaded {repo}@{e['revision'][:12]} shards={len(e['safetensors_sha256'])}", flush=True)
    for repo in args.resolve_only:
        e = resolve(api, repo, download=False)
        e["role"] = roles.get(repo)
        out["checkpoints"].append(e)
        print(f"resolved   {repo}@{e['revision'][:12]} shards={len(e['safetensors_sha256'])}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
