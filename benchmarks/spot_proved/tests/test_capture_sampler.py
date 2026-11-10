"""`capture.py --sampler` records rank 0's sampler at a recorded step without changing what it returns: the logits entering
top-k/top-p are copied before vLLM masks them in place, and each written tensor's SHA-512 in ``sampler.json`` is its bytes'."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
STEP = 5


def _fake_sampler_module(monkeypatch) -> types.ModuleType:
    mod = types.ModuleType("vllm.v1.worker.gpu.sample.sampler")

    def apply_top_k_top_p(logits, top_k, top_p):
        logits[:, 2:] = float("-inf")
        return logits

    def gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False, use_fp64=False):
        return logits.argmax(dim=-1)

    class Sampler:
        def __call__(self, logits, input_batch):
            processed = mod.apply_top_k_top_p(logits.clone() / 0.5, None, torch.tensor([0.95, 0.95]))
            sampled = mod.gumbel_sample(processed, torch.tensor([1, 0]), torch.tensor([0.5, 0.5]), torch.tensor([7, 9]),
                                        torch.tensor([3, 4]), apply_temperature=False, use_fp64=False)
            return types.SimpleNamespace(sampled_token_ids=sampled.view(-1, 1))

    mod.Sampler, mod.apply_top_k_top_p, mod.gumbel_sample = Sampler, apply_top_k_top_p, gumbel_sample
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    sys.modules["vllm.v1.worker.gpu.sample"].sampler = mod
    return mod


def test_the_sampler_tap_reads_only_and_binds_what_it_wrote(tmp_path, monkeypatch) -> None:
    mod = _fake_sampler_module(monkeypatch)
    spec = importlib.util.spec_from_file_location("spot_capture_sampler", HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp_path), "steps": [STEP], "mla_tokens": 0, "open": {}, "keep": "opened", "sampler": True})
    monkeypatch.setattr(capture, "S", S)
    capture._install_sampler()
    logits = torch.tensor([[1.0, 3.0, 9.0, 0.5], [4.0, 2.0, 8.0, 1.0]])
    S.sample_at = STEP
    got = mod.Sampler()(logits.clone(), types.SimpleNamespace(req_ids=["a", "b"]))
    assert got.sampled_token_ids.view(-1).tolist() == [1, 0] and S.sample_at is None and S.smp is None
    d = tmp_path / "rank0" / f"step{STEP}" / "sampler"
    doc = json.loads((d / "sampler.json").read_text())
    assert doc["req_ids"] == ["a", "b"] and doc["path"] == "gumbel" and doc["gumbel_args"] == {"apply_temperature": False, "use_fp64": False}
    for k, rec in doc["tensors"].items():
        assert hashlib.sha512((d / f"{k}.bin").read_bytes()).hexdigest() == rec["sha512"], k
    topp_in = torch.frombuffer(bytearray((d / "topp_in.bin").read_bytes()), dtype=torch.float32).view(2, 4)
    assert torch.equal(topp_in, logits / 0.5)
    kept = torch.frombuffer(bytearray((d / "topp_kept.bin").read_bytes()), dtype=torch.bool).view(2, 4)
    assert kept.tolist() == [[True, True, False, False]] * 2
    assert torch.frombuffer(bytearray((d / "seed.bin").read_bytes()), dtype=torch.int64).tolist() == [9, 7]
    S.sample_at = None
    assert mod.Sampler()(logits.clone(), types.SimpleNamespace(req_ids=["a", "b"])).sampled_token_ids.view(-1).tolist() == [1, 0]
