"""[r19-int-14] seed_checkpoint selects the manifest entry by the workload's repo@revision first, then by role / repo name.

A CASES key is not always the checkpoints.json `role`: Qwen2.5-1.5B's role reads "B1-B4,R*,D*,Q1/Q2 base,X1" while its case is B1, so
`seed_checkpoint B1` exited 3 ("no checkpoints.json entry") and #39's Commit relaunch never reached row_pod.sh.  The launcher passes the
manifest's model + tokenizer_revision; the name match stays for callers that have only the role (pod_bootstrap's --cases).
"""
from __future__ import annotations

import json
from pathlib import Path

from verity_capture.sweep.seed_checkpoint import entry, local_path

TREE = Path(__file__).resolve().parents[2]
MANIFEST = TREE / "manifests" / "checkpoints.json"

DOC = {"checkpoints": [
    {"role": "B1-B4,R*,D*,Q1/Q2 base,X1", "repo": "Qwen/Qwen2.5-1.5B", "revision": "8faed761d45a263340a0528343f099c05c9a4323"},
    {"role": "QWEN15_INSTRUCT", "repo": "Qwen/Qwen2.5-1.5B-Instruct", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"},
    {"role": "MISTRAL7B", "repo": "unsloth/mistral-7b-v0.3", "revision": "abc"},
    # OLMoE: the 09-07 pre-registration stub (no local_path, downloaded False) precedes the seeded OLMOE entry for the same repo@revision
    {"role": "M1/M2", "repo": "allenai/OLMoE-1B-7B-0924", "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5", "downloaded": False},
    {"role": "OLMOE", "repo": "allenai/OLMoE-1B-7B-0924", "revision": "6d84c48581ece794365f2b8e9cfb043c68ade9c5", "downloaded": True,
     "local_path": "/workspace/hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5"},
]}


def test_a_duplicated_repo_revision_prefers_the_case_role_then_a_recorded_local_path():
    """[F-r19-int-16] #67's Commit relaunch exited 3 on KeyError local_path: repo@revision matched the OLMoE stub first."""
    e = entry(DOC, "OLMOE", "allenai/OLMoE-1B-7B-0924", "6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    assert e["role"] == "OLMOE" and "local_path" in e
    e2 = entry(DOC, "SOMETHING_ELSE", "allenai/OLMoE-1B-7B-0924", "6d84c485")      # no role match: the entry with a local_path wins
    assert e2 is e
    stub = DOC["checkpoints"][3]
    assert local_path(stub, "/hf") == Path("/hf/hub/models--allenai--OLMoE-1B-7B-0924/snapshots/6d84c48581ece794365f2b8e9cfb043c68ade9c5")
    assert local_path(e, "/hf") == Path(e["local_path"])


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
