import json

import pytest

from verity_vllm.engine.profiles.stand_in import COPIED, EXPERTS, GLM53, LAYERS, cut, stand_in


def test_keeps_what_one_card_must_and_cuts_the_rest(tmp_path):
    real = json.loads(GLM53.read_text())
    got = stand_in(tmp_path, heads=8)
    assert got == json.loads((tmp_path / "config.json").read_text())
    assert (got["num_hidden_layers"], got["n_routed_experts"], got["num_attention_heads"]) == (LAYERS, EXPERTS, 8)
    assert len(got["indexer_types"]) == len(got["mlp_layer_types"]) == LAYERS
    for k in ("hidden_size", "vocab_size", "index_topk", "index_n_heads", "index_head_dim", "kv_lora_rank",
              "qk_rope_head_dim", "num_nextn_predict_layers", "num_experts_per_tok", "max_position_embeddings"):
        assert got[k] == real[k], k
    assert got["quantization_config"]["weight_block_size"] == [128, 128]
    assert not (tmp_path / "tokenizer.json").exists()


def test_the_mtp_layer_keeps_the_real_ones_unquantized_modules():
    real = json.loads(GLM53.read_text())
    n = real["num_hidden_layers"]
    kept = cut(real)["quantization_config"]["modules_to_not_convert"]
    assert f"model.layers.{LAYERS}.eh_proj" in kept and f"model.layers.{n}.eh_proj" not in kept
    assert f"model.layers.{LAYERS}.self_attn.indexers_proj" in kept
    assert "model.layers.0.self_attn.indexers_proj" in kept and "lm_head" in kept
    assert not any(m.startswith(f"model.layers.{i}.") for m in kept for i in range(LAYERS + 1, n + 1))


def test_refuses_fewer_experts_than_a_token_routes_to():
    with pytest.raises(ValueError, match="experts a token routes to"):
        cut(json.loads(GLM53.read_text()), experts=4)


def test_copies_the_tokenizer_never_the_weights(tmp_path):
    ck, where = tmp_path / "ck", tmp_path / "standin"
    ck.mkdir()
    for name in (*COPIED, "config.json", "model.safetensors.index.json", "model-00001.safetensors"):
        (ck / name).write_text(name)
    stand_in(where, tokenizer_from=ck)
    assert sorted(p.name for p in where.iterdir()) == sorted((*COPIED, "config.json"))
    assert (where / "tokenizer.json").read_text() == "tokenizer.json"
    with pytest.raises(FileNotFoundError, match="no tokenizer.json"):
        stand_in(tmp_path / "other", tokenizer_from=tmp_path / "standin-none")
