"""verity_vllm.program.predict.pieces: a request Program by part, and a window of a model's layers against the whole model, part by
part."""

from __future__ import annotations

import pytest

from verity_vllm.program.predict.build import build_program
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.pieces import compare, pieces, window_config
from tests.program.predict.test_predict import GLM53, _row_phi


def _with(phi: Phi, hf: dict) -> Phi:
    return Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": hf}})


def _parts(phi: Phi):
    """`phi`'s request of 5 prompt rows and one decode as class G's compiler builds it at a tensor-parallel rank, by part."""
    r = build_program(phi, "request", LP=5, T=1, dead_free=True, batched=True, with_desc=False)
    return pieces(r.program, int(phi.tp["world"]))


@pytest.fixture
def glm(tmp_path) -> Phi:
    return _row_phi(GLM53, "GLM53", tmp_path)


@pytest.mark.slow
def test_a_window_s_parts_are_the_whole_model_s(glm):
    """GLM-5.3 cut to its first 8 layers at TP8 (three dense layers with indexers, three MoE sharing the third's top-k, an indexer
    and one sharing it): the window of layers 0..3 has the model's prologue and layers 0..2, the window 2..5 (starting at the last
    dense layer, so one dense layer, then MoE) its layers 3 and 4, and the window 6..7 its last layer and epilogue. A window's last
    layer drops the rows no sample reads, as only the model's last does, so the window 0..3's layer 3 isn't the model's."""
    hf = window_config(glm.hf_config, 0, 8)
    model = _parts(_with(glm, hf))
    assert (model.layers, model.launches) == (8, 2)
    assert model.sizes["prologue"] and all(model.sizes[str(L)] for L in range(8)) and model.sizes["epilogue"]
    for at, k, matched in ((0, 4, ["prologue", "0", "1", "2"]), (2, 4, ["3", "4"]), (6, 2, ["7", "epilogue"])):
        window = _parts(_with(glm, window_config(hf, at, k)))
        got = compare(model, window, at)
        assert [r["model"] for r in got["parts"]] == matched and got["different"] == 0 and got["callees_differ"] == [], got
        if at == 0:
            assert window.parts["3"] != model.parts["3"]


def test_window_config_slices_the_layer_kinds():
    """A window's config: its layers' indexer and MLP kinds, the dense layers left first, and the top-k offset vLLM's rule needs
    to give its indexers; a window starting at a layer that shares another's top-k is refused."""
    full = ["full"] * 3 + (["shared"] * 3 + ["full"]) * 3 + ["shared"] * 3
    hf = {"num_hidden_layers": len(full), "indexer_types": full, "mlp_layer_types": ["dense"] * 3 + ["sparse"] * (len(full) - 3),
          "first_k_dense_replace": 3, "index_topk_freq": 4, "index_skip_topk_offset": 3}
    head = window_config(hf, 0, 7)
    assert (head["num_hidden_layers"], head["first_k_dense_replace"], head["index_skip_topk_offset"]) == (7, 3, 3)
    mid = window_config(hf, 2, 5)
    assert (mid["indexer_types"], mid["mlp_layer_types"]) == (["full", "shared", "shared", "shared", "full"], ["dense"] + ["sparse"] * 4)
    assert (mid["first_k_dense_replace"], mid["index_skip_topk_offset"]) == (1, 1)
    tail = window_config(hf, 10, 6)
    assert (tail["first_k_dense_replace"], tail["index_skip_topk_offset"], tail["mlp_layer_types"]) == (0, 1, ["sparse"] * 6)
    with pytest.raises(ValueError, match="indexer of its own"):
        window_config(hf, 3, 4)
    with pytest.raises(ValueError, match="aren't inside"):
        window_config(hf, 12, 8)
