"""The base pin's weights as the int8 GPU scripts read them: `tests.engine.int8_exact_gpu.BaseWeights` (a snapshot's safetensors by
tensor name, through `verity_vllm.engine.fp8_checkpoint.read_header`) and `tests.engine.int8_experts_gpu.ExpertBase` (an expert's
projection of `verity_vllm.engine.int8_experts`).  The snapshot is two safetensors files written here; no GPU and no model."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.engine import int8_exact_gpu as G
from tests.engine import int8_experts_gpu as XG
from verity_vllm.engine import int8_experts as X

EXPERTS = "model.layers.3.mlp.experts"


def _shard(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """A safetensors file: the header's length, the header, the tensors' bytes in order."""
    header: dict[str, dict] = {"__metadata__": {"format": "pt"}}
    blob = b""
    for name, values in tensors.items():
        header[name] = {"dtype": "F32" if values.dtype == np.float32 else "BF16", "shape": list(values.shape),
                        "data_offsets": [len(blob), len(blob) + values.nbytes]}
        blob += values.tobytes()
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + blob)


def _words(first: int, rows: int = 2, cols: int = 3) -> np.ndarray:
    """``rows`` x ``cols`` bfloat16 words counting up from ``first``."""
    return np.array(range(first, first + rows * cols), np.int16).reshape(rows, cols)


@pytest.fixture
def snapshot(tmp_path) -> Path:
    _shard(tmp_path / "model-00001-of-00002.safetensors", {
        f"{EXPERTS}.0.gate_proj.weight": _words(100), f"{EXPERTS}.0.up_proj.weight": _words(200),
        f"{EXPERTS}.0.down_proj.weight": _words(300, 3, 2), f"{EXPERTS}.1.gate_proj.weight": _words(400),
        "model.layers.3.self_attn.q_proj.weight": _words(500, 1), "model.layers.3.self_attn.k_proj.weight": _words(600, 1)})
    _shard(tmp_path / "model-00002-of-00002.safetensors", {
        f"{EXPERTS}.1.up_proj.weight": _words(700), "model.layers.3.self_attn.v_proj.weight": _words(800, 1),
        "model.layers.13.mlp.experts.0.down_proj.weight": _words(900, 3, 2), "model.norm.weight": np.array([1.5, -2.0], np.float32)})
    return tmp_path


def _stored(t: torch.Tensor) -> list:
    return t.view(torch.int16).tolist()


def test_base_weights_reads_a_tensor_as_stored_from_whichever_file_holds_it(snapshot):
    base = G.BaseWeights(str(snapshot))
    assert len(base.where) == 10 and "__metadata__" not in base.where
    assert base.where["model.norm.weight"].name == "model-00002-of-00002.safetensors"
    got = base.tensor(f"{EXPERTS}.0.up_proj.weight")
    assert got.dtype == torch.bfloat16 and _stored(got) == [[200, 201, 202], [203, 204, 205]]
    norm = base.tensor("model.norm.weight")
    assert norm.dtype == torch.float32 and norm.tolist() == [1.5, -2.0]
    with pytest.raises(KeyError):
        base.tensor("model.layers.3.self_attn.o_proj.weight")


def test_base_weights_joins_a_merged_modules_parts_by_rows(snapshot):
    base = G.BaseWeights(str(snapshot))
    # q and k are in the first file and v in the second: vLLM's qkv_proj holds the three, in that order
    assert _stored(base.weight("model.layers.3.self_attn.qkv_proj")) == [[500, 501, 502], [600, 601, 602], [800, 801, 802]]
    assert _stored(base.weight(f"{EXPERTS}.0.down_proj")) == [[300, 301], [302, 303], [304, 305]]


def test_expert_base_names_an_experts_projection_by_layer_expert_and_part(snapshot):
    base = XG.ExpertBase(G.BaseWeights(str(snapshot)))
    assert _stored(base.weight(EXPERTS, 0, X.GATE_UP)) == [[100, 101, 102], [103, 104, 105], [200, 201, 202], [203, 204, 205]]
    # expert 1's two parts are in two files
    assert _stored(base.weight(EXPERTS, 1, X.GATE_UP)) == [[400, 401, 402], [403, 404, 405], [700, 701, 702], [703, 704, 705]]
    # layer 3's expert 0, not layer 13's
    assert _stored(base.weight(EXPERTS, 0, X.DOWN)) == [[300, 301], [302, 303], [304, 305]]
    with pytest.raises(KeyError, match="layer 3 expert 1 down_proj: 0 base tensors"):
        base.weight(EXPERTS, 1, X.DOWN)
    with pytest.raises(KeyError, match="layer 13 expert 0 gate_proj: 0 base tensors"):
        base.weight("model.layers.13.mlp.experts", 0, X.GATE_UP)
