"""Seed one pinned checkpoint (manifests/checkpoints.json) into HF_HOME and verify every listed safetensors sha256.

    python -m verity_capture.sweep.seed_checkpoint <ROLE|repo> [--repo HF_REPO --revision SHA] [--manifest manifests/checkpoints.json]

--repo/--revision (the workload manifest's model + tokenizer_revision) select the entry first: a CASES key is not always the entry's
`role` -- Qwen2.5-1.5B's role reads "B1-B4,R*,D*,Q1/Q2 base,X1" while its case is B1, so #39's Commit relaunch exited 3 here (r19-int-14).

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


def entry(doc: dict, case: str, repo: str | None = None, revision: str | None = None) -> dict | None:
    if repo:
        hits = [c for c in doc["checkpoints"]
                if c.get("repo") == repo and (not revision or str(c.get("revision", "")).startswith(revision) or revision.startswith(str(c.get("revision", ""))))]
        if hits:
            # the manifest can carry one repo@revision twice (OLMoE: a 09-07 pre-registration stub without local_path next to the seeded
            # OLMOE entry) -- prefer the entry whose role is the case, then one that records a local_path (F-r19-int-16)
            hits.sort(key=lambda c: (c.get("role") != case, "local_path" not in c))
            return hits[0]
    for c in doc["checkpoints"]:
        if c.get("role") == case or c.get("repo") == case:
            return c
    return None


def local_path(ent: dict, hf_home: str) -> Path:
    """The entry's recorded local_path, else the HF cache snapshot dir that snapshot_download would populate."""
    if ent.get("local_path"):
        return Path(ent["local_path"])
    return Path(hf_home) / "hub" / ("models--" + ent["repo"].replace("/", "--")) / "snapshots" / ent["revision"]


def seed(ent: dict, hf_home: str) -> tuple[bool, str]:
    local = local_path(ent, hf_home); shas = ent.get("safetensors_sha256", {})
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
    ap.add_argument("--repo", default=None, help="the workload manifest's model; selects the entry before the role/repo name match")
    ap.add_argument("--revision", default=None, help="the workload manifest's tokenizer_revision (full or prefix)")
    ap.add_argument("--manifest", default="manifests/checkpoints.json")
    a = ap.parse_args(argv)
    doc = json.load(open(a.manifest))
    ent = entry(doc, a.case, a.repo, a.revision)
    if ent is None:
        print(f"no checkpoints.json entry for {a.case}" + (f" / {a.repo}@{(a.revision or '')[:12]}" if a.repo else ""), file=sys.stderr); return 3
    hf_home = os.environ.get("HF_HOME") or "/workspace/hf"
    ok, msg = seed(ent, hf_home)
    print(("checkpoint OK " if ok else "checkpoint FAIL ") + msg)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
