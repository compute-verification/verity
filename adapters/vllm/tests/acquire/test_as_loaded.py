"""The MXFP4 Marlin experts' weights as loaded (`verity_vllm/acquire/as_loaded.py`) on CPU, over DeepSeek-V4's synthetic checkpoint
(real safetensors) at TP rank 1 of 2, with a stand-in for Marlin's repack (a byte permutation and a dtype change, as the real one is).

It catches a rank that registers, pins or replays Marlin's repacked layout where its Program reads the checkpoint's packed e2m1 and
e8m0 bytes; a served tensor that is not the repack of the checkpoint's bytes passing unnamed; a model without a Marlin layer touched;
and a composer the TP driver cannot hand to its ranks through the RPC (it pickles its kwargs)."""
from __future__ import annotations

import hashlib
import pickle
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn
from verity_vllm.acquire import as_loaded as AL
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.replay.opening import model_weights_provider
from verity_vllm.pipeline.tp import commit as TPC
from verity_vllm.program.frontend import marlin_moe_leaf as mml
from verity_vllm.protocols import WeightsView

from tests.check.test_weights_of_record_dsv4_experts import (  # noqa: F401  (dsv4: the checkpoint fixture)
    EX,
    RANK,
    WORLD,
    E,
    H,
    I,
    dsv4,
)
from tests.check.test_weights_of_record_routed_experts import _program


class _Mxfp4Method:
    mxfp4_backend, moe_kernel = "marlin", "built"

    def __init__(self) -> None:
        self.num_experts, self.intermediate_size, self.hidden_size = E, I // WORLD, H
        self.moe = types.SimpleNamespace(w13_num_shards=2)


def repack(layer, w13, w2, s13, s2):
    """A stand-in for `prepare_moe_mxfp4_layer_for_marlin`: the codes reordered into int32 words, the scales reordered."""
    return (w13.flip(-1).contiguous().view(torch.int32), w2.flip(-1).contiguous().view(torch.int32), s13.flip(1).contiguous(),
            s2.flip(1).contiguous())


@pytest.fixture
def vllm_mxfp4(monkeypatch):
    mxfp4, oracle = types.ModuleType(mml._MXFP4), types.ModuleType(mml._ORACLE)
    mxfp4.Mxfp4MoEMethod = _Mxfp4Method
    oracle.Mxfp4MoeBackend = types.SimpleNamespace(MARLIN="marlin")
    for module in (mxfp4, oracle):
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def served(dsv4, tmp_path, vllm_mxfp4):
    """A model whose one Marlin layer serves the repack of rank 1's experts as loaded, beside a bf16 norm; the composer of record."""
    composer = {"checkpoint_dir": str(tmp_path / "snap"), "checkpoints_manifest": str(tmp_path / "checkpoints.json")}
    compose = WR.rank_composer(**composer, world=WORLD, rank=RANK)
    layer = nn.Module()
    layer.quant_method = _Mxfp4Method()
    loaded = {f: torch.from_numpy(compose(f"{EX}.{f}", s, 8)) for f, s in AL.shapes(layer).items()}
    for f, t in zip(AL.FIELDS, repack(layer, *(loaded[f] for f in AL.FIELDS))):
        setattr(layer, f, nn.Parameter(t, requires_grad=False))
    model = nn.Module()
    model.model = nn.Module()
    model.model.norm = nn.Parameter(torch.ones(H, dtype=torch.bfloat16), requires_grad=False)
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].ffn = nn.Module()
    model.model.layers[0].ffn.experts = nn.Module()
    model.model.layers[0].ffn.experts.routed_experts = layer
    return model, layer, loaded, composer


def test_a_ranks_marlin_fields_are_registered_pinned_and_replayed_as_the_checkpoints_bytes(served, dsv4):
    model, _layer, loaded, composer = served
    view, info = AL.attach_rank(model, world=WORLD, rank=RANK, composer=lambda **kw: WR.rank_composer(**composer, **kw), repack=repack)
    assert isinstance(view, WeightsView) and info["layers"] == 1 and info["fields"] == 4
    assert info["bytes"] == sum(t.numel() for t in loaded.values())
    params = dict(view.named_parameters())
    for f in AL.FIELDS:
        assert params[f"{EX}.{f}"].dtype == torch.uint8 and torch.equal(params[f"{EX}.{f}"], loaded[f]), f
    assert params["model.norm"] is model.model.norm
    prog = _program([(n, list(t.shape), 8) for n, t in ((f"{EX}.{f}", loaded[f]) for f in AL.FIELDS)])
    prog["result"]["tp"] = {"world": WORLD, "rank": RANK}
    rec = WR.weights_of_record(prog, dsv4["ck"], prefix="")
    assert WR.check(rec, WR.live_param_digests(view))["result"] is True
    served_pin = WR.check(rec, WR.live_param_digests(model))
    assert served_pin["result"] is False and len(served_pin["mismatched"]) == 4, "the repacked layout is not the weights of record"
    read = model_weights_provider(view, "")
    w2 = loaded["w2_weight"].numpy().reshape(-1)
    assert np.array_equal(read(f"{EX}.w2_weight", 0, w2.size), w2)
    assert hashlib.sha256(w2.tobytes()).hexdigest() == WR.live_param_digests(view)["params"][f"{EX}.w2_weight"]["digest"]


def test_a_served_tensor_that_is_not_the_repack_of_the_checkpoint_refuses_by_name(served):
    model, layer, _loaded, composer = served
    with torch.no_grad():
        layer.w2_weight_scale.view(-1)[3] ^= 1
    with pytest.raises(ValueError, match=rf"{EX}: Marlin's repack of the checkpoint's bytes is not the served w2_weight_scale:"):
        AL.attach_rank(model, world=WORLD, rank=RANK, composer=lambda **kw: WR.rank_composer(**composer, **kw), repack=repack)


def test_a_model_without_a_marlin_layer_is_its_own_weights(vllm_mxfp4):
    model = nn.Linear(4, 4)
    assert AL.attach_rank(model, world=WORLD, rank=RANK, composer=None) == (model, None)


def test_the_tp_driver_hands_every_rank_the_composer_of_record_through_the_rpc():
    assert TPC.weights_as_loaded(None) is None
    pin = {"checkpoint_dir": "/ck", "checkpoints_manifest": "/ck/checkpoints.json", "model": "test/dsv4", "program_dirs": {}}
    sent = pickle.loads(pickle.dumps(TPC.weights_as_loaded(pin)))
    assert sent.func is AL.attach_rank and sent.keywords["composer"].func is WR.rank_composer
    assert sent.keywords["composer"].keywords == {"checkpoint_dir": "/ck", "checkpoints_manifest": "/ck/checkpoints.json", "model": "test/dsv4"}
