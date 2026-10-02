"""`engine_profile.checkpoint_for`: a pinned snapshot missing at the manifest's `local_path` resolves to the same revision under
`HF_HOME`'s hub (a shared host whose HF cache is elsewhere), and is refused when neither holds it.  Laptop: torch-free, instant."""
from __future__ import annotations

import json

import pytest

from verity_vllm.engine import engine_profile as prof

REPO, REV = "Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"


def _manifest(tmp_path, local_path):
    path = tmp_path / "checkpoints.json"
    path.write_text(json.dumps({"checkpoints": [{"repo": REPO, "revision": REV, "role": "QWEN05", "downloaded": True,
                                                 "local_path": str(local_path)}]}))
    return path


def test_manifest_local_path_wins_when_present(tmp_path, monkeypatch):
    local = tmp_path / "pinned"
    local.mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
    assert prof.checkpoint_for("QWEN05", _manifest(tmp_path, local))["local_path"] == str(local)


def test_missing_local_path_falls_back_to_hf_home_snapshot(tmp_path, monkeypatch):
    hub = tmp_path / "hf" / "hub" / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / REV
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    cp = prof.checkpoint_for("QWEN05", _manifest(tmp_path, tmp_path / "missing"))
    assert cp["local_path"] == str(hub) and cp["revision"] == REV


def test_missing_everywhere_is_refused(tmp_path, monkeypatch):
    (tmp_path / "hf" / "hub" / "models--Qwen--Qwen2.5-0.5B" / "snapshots" / "another-revision").mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(SystemExit, match="missing"):
        prof.checkpoint_for("QWEN05", _manifest(tmp_path, tmp_path / "missing"))
