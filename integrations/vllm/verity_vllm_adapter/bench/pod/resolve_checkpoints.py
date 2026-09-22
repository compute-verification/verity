"""Resolve + hash the larger checkpoints W4d uses into a manifest FRAGMENT (new entries only; the
frozen entries of manifests/checkpoints.json are never touched; the fragment is merged on the
laptop by appending). Runs ON THE POD (HF_HOME=/workspace/hf; snapshots already downloaded)::

    /workspace/venv/bin/python -m verity_vllm_adapter.bench.pod.resolve_checkpoints --out /workspace/results/checkpoints-w4d.json Qwen/Qwen2.5-7B-Instruct ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

from verity_vllm_adapter.checkpoints import resolve

ROLES = {
    "Qwen/Qwen2.5-7B-Instruct": "W4d cost (§16.7) 7B dense",
    "Qwen/Qwen2.5-7B": "W4d compiled-vs-eager A/B (§15.5) 7B dense; contract §1 D1/D2 optional subrow",
    "Qwen/Qwen2.5-14B-Instruct": "W4d cost (§16.7) 14B dense",
    "Qwen/Qwen2.5-32B-Instruct": "W4d cost (§16.7) 32B dense (if it runs)",
    "Qwen/Qwen2.5-1.5B-Instruct": "W4d (downloaded, unused: the B1 checkpoint Qwen/Qwen2.5-1.5B is used instead)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("repos", nargs="+")
    args = ap.parse_args()
    api = HfApi()
    out = {"schema": "verity-vllm/checkpoints/v1", "fragment": True, "resolved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "hf_home": os.environ.get("HF_HOME"), "checkpoints": []}
    for repo in args.repos:
        e = resolve(api, repo, download=True)  # snapshot already cached: no network transfer beyond metadata
        e["role"] = ROLES.get(repo, "W4d")
        e["resolved_by"] = "W4d (H100 pod b302edd59a53)"
        out["checkpoints"].append(e)
        print(f"{repo}@{e['revision'][:12]} shards={len(e['safetensors_sha256'])} local={e.get('local_path')}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
