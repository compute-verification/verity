"""A node-local checkpoint pinned by its content (`config.CONTENT`): it resolves where its files hash to its pin's revision,
and its build refuses it by name where they don't."""
import json

import pytest

from verity_vllm.config import CONTENT, EnginePin, checkpoint_digest
from verity_vllm.engine.engine_profile import checkpoint_for


def _pinned(tmp_path, revision=None):
    ckpt = tmp_path / "llama"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"weights")
    revision = revision or CONTENT + checkpoint_digest(ckpt)
    entry = {"repo": str(ckpt), "revision": revision, "role": "LOCAL", "local_path": str(ckpt), "downloaded": True}
    (tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": []}))
    (tmp_path / "checkpoints-local.json").write_text(json.dumps({"checkpoints": [entry]}))
    return ckpt, revision, tmp_path / "checkpoints.json"


def test_a_local_checkpoint_resolves_where_its_files_hash_to_its_revision_and_is_refused_by_name_where_not(tmp_path):
    ckpt, revision, manifest = _pinned(tmp_path)
    pin = EnginePin.resolve(str(ckpt), revision, checkpoints=manifest)
    assert (pin.case, pin.local_path) == ("LOCAL", str(ckpt))
    assert checkpoint_for("LOCAL", manifest)["local_path"] == str(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"other weights")
    with pytest.raises(SystemExit, match=f"{ckpt}: its files hash to {CONTENT}[0-9a-f]{{64}}, not its pin's revision {revision}"):
        checkpoint_for("LOCAL", manifest)


def test_a_local_checkpoint_is_bound_by_the_files_a_loader_reads_not_the_hub_snapshots_metadata(tmp_path):
    ckpt, _, _ = _pinned(tmp_path)
    digest = checkpoint_digest(ckpt)
    (ckpt / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    (ckpt / "README.md").write_text("# model\n")
    assert checkpoint_digest(ckpt) == digest
    (ckpt / "tokenizer.json").write_text("{}")
    assert checkpoint_digest(ckpt) != digest
