"""[r19-int-14] seed_checkpoint selects the manifest entry by the workload's repo@revision first, then by role / repo name.

A CASES key is not always the checkpoints.json `role`: Qwen2.5-1.5B's role reads "B1-B4,R*,D*,Q1/Q2 base,X1" while its case is B1, so
`seed_checkpoint B1` exited 3 ("no checkpoints.json entry") and #39's Commit relaunch never reached row_pod.sh.  The launcher passes the
manifest's model + tokenizer_revision; the name match stays for callers that have only the role (pod_bootstrap's --cases).
"""
from __future__ import annotations

import json
from pathlib import Path

from verity_capture.sweep.seed_checkpoint import entry

TREE = Path(__file__).resolve().parents[2]
MANIFEST = TREE / "manifests" / "checkpoints.json"

DOC = {"checkpoints": [
    {"role": "B1-B4,R*,D*,Q1/Q2 base,X1", "repo": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323"},
    {"role": "QWEN15_INSTRUCT", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
    {"role": "MISTRAL7B", "repo": "unsloth/mistral-7b-v0.3", "revision": "abc"},
]}


def test_repo_and_revision_select_before_the_role_name():
    e = entry(DOC, "B1", "Qwen/Qwen2.5-1.5B", "8faed761d45a263340a0528343f099c05c9a4323")
    assert e is not None and e["repo"] == "Qwen/Qwen2.5-1.5B"
    assert entry(DOC, "B1", "Qwen/Qwen2.5-1.5B", "8faed761d4") is e          # a 10-char manifest revision is a prefix match
    assert entry(DOC, "B1") is None                                          # the bare case key has no role of that name
    assert entry(DOC, "MISTRAL7B")["repo"] == "unsloth/mistral-7b-v0.3"      # role match unchanged
    assert entry(DOC, "unsloth/mistral-7b-v0.3")["role"] == "MISTRAL7B"      # repo-as-name match unchanged
    assert entry(DOC, "B1", "Qwen/Qwen2.5-1.5B", "0000000000") is None       # a repo with the wrong revision selects nothing


def test_the_real_manifest_resolves_every_workload_model_by_repo():
    if not MANIFEST.exists():
        return
    doc = json.load(open(MANIFEST))
    wl = sorted((TREE / "workloads").glob("*.json"))
    unresolved = []
    for f in wl:
        d = json.load(open(f))
        if entry(doc, d.get("case", ""), d["model"], d.get("tokenizer_revision")) is None:
            unresolved.append(f.name)
    assert not unresolved, unresolved
