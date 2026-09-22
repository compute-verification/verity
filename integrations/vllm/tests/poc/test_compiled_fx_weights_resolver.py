"""CPU tests for ``compiled_fx.weights_resolver`` (needs torch, not vLLM): the checkpoint-file sha check and
the entry -> bytes answer, including the reader's re-hash of a resolved value against the weights index."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from _helpers import TINY_MANIFEST

torch = pytest.importorskip("torch")

from verity_vllm.capture import CaptureError, RunWriter, WeightsReader  # noqa: E402
from verity_vllm_adapter.compiled_fx.weights_resolver import CheckpointWeightResolver  # noqa: E402


def _cp(tmp_path: Path, content: bytes = b"safetensors-bytes") -> tuple[dict, str]:
    (tmp_path / "model.safetensors").write_bytes(content)
    return {"repo": "toy/model", "revision": "r0", "local_path": str(tmp_path)}, hashlib.sha256(content).hexdigest()


def test_check_files_accepts_matching_sha_and_rejects_mismatch(tmp_path: Path) -> None:
    cp, sha = _cp(tmp_path)
    r = CheckpointWeightResolver(cp, {"model.safetensors": sha}, log=lambda *_: None)
    r.check_files()
    assert r.checked_files == {"model.safetensors": sha}
    bad = CheckpointWeightResolver(cp, {"model.safetensors": "00" * 32}, log=lambda *_: None)
    with pytest.raises(ValueError):
        bad.check_files()
    missing = CheckpointWeightResolver(cp, {"model-00002.safetensors": sha}, log=lambda *_: None)
    with pytest.raises(FileNotFoundError):
        missing.check_files()


def test_resolver_answers_index_entries_and_reader_rehashes(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    raw = w.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    writer = RunWriter(run, TINY_MANIFEST)
    ww = writer.weights(store_values=False, source_sha256={"model.safetensors": "ab" * 32})
    ww.add("layers.0.input_layernorm.weight", "float32", [3, 4], raw)
    ww.finish()
    reader = WeightsReader(run / "weights")
    cid = reader.ids()[0]
    with pytest.raises(CaptureError):
        reader.read(cid)  # hash-only bundle, no resolver
    cp, _ = _cp(tmp_path)
    r = CheckpointWeightResolver(cp, {}, log=lambda *_: None)
    r._params = {"layers.0.input_layernorm.weight": w}  # what vLLM's loader would hand back
    reader.resolver = r
    assert reader.read(cid) == raw
    assert r.resolved == 1
    # a checkpoint that does not reproduce the committed weight fails the index re-hash
    r._params = {"layers.0.input_layernorm.weight": w + 1}
    with pytest.raises(CaptureError):
        reader.read(cid)
    # dtype/shape disagreement is refused before hashing
    r._params = {"layers.0.input_layernorm.weight": w.to(torch.bfloat16)}
    with pytest.raises(ValueError):
        reader.read(cid)
