"""`verity_vllm.pipeline.tp.rank_worker`: the shards a rank holds (`TP2CaptureWorkerExtension.tp2_param_hashes`, what
`tp-capture --param-hashes` asks of each rank).  The rank writes `param_hashes.rank<r>.json`: the shape, dtype, byte count and
SHA-256 of every parameter and buffer of its model, as the bytes lie in memory, and the sharding attributes of its parallel
layers, the embedding, the LM head and the logits processor.  It answers the counts.  CPU-only: the model's modules carry the
class names the rank lists by, and the test sets the rank's facts."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension
from verity_vllm.records.schema_ids import TP_PARAM_HASHES_SCHEMA

FACTS = {"tp": {"rank": 1, "world_size": 2}, "pid": 7}
HERE = __name__


class _Shards:
    """vLLM's shard indices of an embedding: the rank records its public integers."""
    num_org_elements, org_vocab_start_index = 4, 2
    name = "indices"
    _hidden = 9


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.num_embeddings, self.embedding_dim, self.tp_size = 4, 3, 2
        self.shard_indices = _Shards()


class RowParallelLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.5, -2.0], [0.25, 3.0]], dtype=torch.bfloat16))
        self.register_buffer("scale", torch.arange(6, dtype=torch.float32).reshape(2, 3).t())      # not contiguous
        self.register_buffer("none_yet", torch.zeros(0, 4, dtype=torch.int8))
        self.input_size, self.output_size, self.reduce_results, self.tp_rank = 4, 2, True, 1


class RMSNorm(torch.nn.Module):
    """Its weight is hashed; the module is no parallel layer and is not listed."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.tp_size = 2


class ParallelLMHead(torch.nn.Module):
    """Tied to the embedding: no tensor of its own."""
    num_embeddings, org_vocab_size = 4, 8


class LogitsProcessor(torch.nn.Module):
    use_fused_embedding = False


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = VocabParallelEmbedding()
        self.layer = torch.nn.ModuleDict({"proj": RowParallelLinear(), "plain": torch.nn.Linear(1, 1, bias=False), "norm": RMSNorm()})
        self.lm_head, self.logits_processor = ParallelLMHead(), LogitsProcessor()


class _Rank(TP2CaptureWorkerExtension):
    def __init__(self, model: torch.nn.Module) -> None:
        self.model_runner = SimpleNamespace(get_model=lambda: model)


def _raw(t: torch.Tensor) -> bytes:
    """The tensor's bytes in row-major order, read here through numpy's view of the same width."""
    t = t.detach().contiguous()
    width = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.int8: torch.int8}[t.dtype]
    return t.view(width).numpy().tobytes()


@pytest.fixture
def rank(monkeypatch: pytest.MonkeyPatch) -> _Rank:
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: FACTS)
    return _Rank(_Model())


def test_a_rank_writes_the_hash_of_every_parameter_and_buffer_and_answers_the_counts(rank, tmp_path):
    out = tmp_path / "made" / "here"
    model = rank.model_runner.get_model()
    assert rank.tp2_param_hashes(out_dir=str(out)) == {"rank": 1, "n_params": 4, "n_buffers": 2, "total_bytes": 24 + 8 + 4 + 4 + 24 + 0,
                                                       "modules": 4}
    assert [p.name for p in out.iterdir()] == ["param_hashes.rank1.json"]
    text = (out / "param_hashes.rank1.json").read_text()
    doc = json.loads(text)
    assert text == json.dumps(doc, indent=1) + "\n"
    assert list(doc) == ["schema", "rank", "world_size", "n_params", "n_buffers", "total_bytes", "tensors", "modules"]
    assert [doc[k] for k in ("schema", "rank", "world_size", "n_params", "n_buffers", "total_bytes")] == [TP_PARAM_HASHES_SCHEMA, 1, 2, 4, 2, 64]
    assert doc["schema"] == "verity-capture/tp2-param-hashes/v0"
    held = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    assert list(doc["tensors"]) == ["embed.weight", "layer.proj.weight", "layer.plain.weight", "layer.norm.weight", "layer.proj.scale",
                                    "layer.proj.none_yet"]
    for name, t in doc["tensors"].items():
        raw = _raw(held[name])
        assert t == {"kind": "buffer" if name in dict(model.named_buffers()) else "param", "shape": list(held[name].shape),
                     "dtype": str(held[name].dtype).replace("torch.", ""), "sha256": hashlib.sha256(raw).hexdigest(), "nbytes": len(raw)}, name
    assert doc["tensors"]["layer.proj.weight"]["dtype"] == "bfloat16" and doc["tensors"]["layer.norm.weight"]["shape"] == []
    assert doc["tensors"]["layer.proj.none_yet"] == {"kind": "buffer", "shape": [0, 4], "dtype": "int8",
                                                     "sha256": hashlib.sha256(b"").hexdigest(), "nbytes": 0}


def test_a_buffer_that_is_not_contiguous_is_hashed_in_row_major_order(rank, tmp_path):
    rank.tp2_param_hashes(out_dir=str(tmp_path))
    doc = json.loads((tmp_path / "param_hashes.rank1.json").read_text())
    rows = torch.tensor([[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]).numpy().tobytes()
    assert doc["tensors"]["layer.proj.scale"] == {"kind": "buffer", "shape": [3, 2], "dtype": "float32",
                                                  "sha256": hashlib.sha256(rows).hexdigest(), "nbytes": 24}


def test_the_parallel_layers_are_listed_with_the_sharding_attributes_they_have(rank, tmp_path):
    rank.tp2_param_hashes(out_dir=str(tmp_path))
    mods = json.loads((tmp_path / "param_hashes.rank1.json").read_text())["modules"]
    assert mods == {
        "embed": {"tp_size": 2, "num_embeddings": 4, "embedding_dim": 3, "shard_indices": {"num_org_elements": 4, "org_vocab_start_index": 2},
                  "class": f"{HERE}.VocabParallelEmbedding"},
        "layer.proj": {"tp_rank": 1, "input_size": 4, "output_size": 2, "reduce_results": True, "class": f"{HERE}.RowParallelLinear"},
        "lm_head": {"num_embeddings": 4, "org_vocab_size": 8, "class": f"{HERE}.ParallelLMHead"},
        "logits_processor": {"use_fused_embedding": False, "class": f"{HERE}.LogitsProcessor"},
    }
    assert [list(m) for m in mods.values()][0] == ["tp_size", "num_embeddings", "embedding_dim", "shard_indices", "class"]


@pytest.mark.parametrize("name, listed", [("ColumnParallelLinear", True), ("QKVParallelLinear", True), ("Embedding", True), ("LogitsProcessor", True),
                                          ("ParallelLMHead", True), ("Linear", False), ("LogitsProcessorWrapper", False), ("EmbeddingBag", False),
                                          ("RMSNorm", False)])
def test_a_module_is_listed_by_its_class_name(monkeypatch, tmp_path, name: str, listed: bool):
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: FACTS)
    model = torch.nn.ModuleDict({"m": type(name, (torch.nn.Module,), {})()})
    assert _Rank(model).tp2_param_hashes(out_dir=str(tmp_path)) == {"rank": 1, "n_params": 0, "n_buffers": 0, "total_bytes": 0, "modules": int(listed)}
    assert list(json.loads((tmp_path / "param_hashes.rank1.json").read_text())["modules"]) == (["m"] if listed else [])


def test_each_rank_writes_its_own_file(monkeypatch, tmp_path):
    for r in (0, 1):
        facts: dict[str, Any] = {"tp": {"rank": r, "world_size": 2}}
        monkeypatch.setattr(RW, "_tp_facts", lambda worker, facts=facts: facts)
        assert _Rank(_Model()).tp2_param_hashes(out_dir=str(tmp_path))["rank"] == r
    assert sorted(p.name for p in tmp_path.iterdir()) == ["param_hashes.rank0.json", "param_hashes.rank1.json"]
    assert [json.loads((tmp_path / f"param_hashes.rank{r}.json").read_text())["rank"] for r in (0, 1)] == [0, 1]
