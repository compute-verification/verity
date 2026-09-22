"""Seed one pinned checkpoint (manifests/checkpoints.json) into HF_HOME and verify every listed safetensors sha256.

    python -m verity_capture.sweep.seed_checkpoint <ROLE|repo> [--manifest manifests/checkpoints.json]

The same rule pod_bootstrap.sh applies at --cases, callable for one role after bootstrap (R19: a pod bootstrapped with --cases B0,MISTRAL7B,OLMOE
was handed #11 = LLAMA32_1B; the Build died in 7 s on HF's offline lookup).  Exit 0 = present and verified; 3 = no entry; 4 = sha mismatch.
Weights are never trusted from the download alone: weights_of_record re-checks the shas at Commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def entry(doc: dict, case: str) -> dict | None:
    for c in doc["checkpoints"]:
        if c.get("role") == case or c.get("repo") == case:
            return c
    return None


def seed(ent: dict, hf_home: str) -> tuple[bool, str]:
    local = Path(ent["local_path"]); shas = ent.get("safetensors_sha256", {})
    if not all((local / f).exists() for f in shas) or not (local / "config.json").exists():
        from huggingface_hub import snapshot_download
        p = snapshot_download(ent["repo"], revision=ent["revision"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
                              cache_dir=os.path.join(hf_home, "hub"))
        print("downloaded", p, flush=True)
    for f, sha in shas.items():
        h = hashlib.sha256()
        with open(local / f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != sha:
            return False, f"sha256 mismatch {f}: {h.hexdigest()[:16]} != {sha[:16]}"
    return True, f"{ent['repo']}@{ent['revision'][:12]} at {local} ({len(shas)} shards verified)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--manifest", default="manifests/checkpoints.json")
    a = ap.parse_args(argv)
    doc = json.load(open(a.manifest))
    ent = entry(doc, a.case)
    if ent is None:
        print(f"no checkpoints.json entry for {a.case}", file=sys.stderr); return 3
    hf_home = os.environ.get("HF_HOME") or "/workspace/hf"
    ok, msg = seed(ent, hf_home)
    print(("checkpoint OK " if ok else "checkpoint FAIL ") + msg)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
