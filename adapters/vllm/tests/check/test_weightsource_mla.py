"""`fold_compare.WeightSource` on an mla-moe weights RecordType: every `dense_layers` / `layers` field reads its own checkpoint layer
(`layers[i]` = `model.layers.{i + FIRST_DENSE}`), fused fields concatenate in vLLM's row order, the f32 router gate widens its
bf16 tensor and the F32 correction bias is read as is."""
import json
import struct

import numpy as np
import pytest
from verity.circuits.types import Array, RecordType
from verity.definitions.registry.prims import BF16, F32

from verity_vllm.check.checkpoint_words import mla_field_words, mla_moe_sources
from verity_vllm.check.fold_compare import WeightSource
from verity_vllm.program.frontend.rules.family import _field_intervals

H, V, NL, FD, NH, QL, L, DN, DR, DV, INTER, ID, E, MAX_POS = 4, 8, 3, 1, 2, 2, 2, 1, 2, 1, 2, 3, 2, 4


def _mat(n: int, k: int, t=BF16) -> Array:
    return Array(n, Array(k, t))


def _mla_weights_type() -> RecordType:
    """The layout of GLM-4.7-Flash's weights RecordType (arch `mla-moe`) at toy sizes: `FD` dense layers, then `NL - FD` MoE layers."""
    attn = [("ln1", Array(H, BF16)), ("qkv_a_w", _mat(QL + L + DR, H)), ("q_a_ln", Array(QL, BF16)), ("q_b_w", _mat(NH * (DN + DR), QL)),
            ("kv_a_ln", Array(L, BF16)), ("kv_b_w", _mat(NH * (DN + DV), L)), ("o_w", _mat(H, NH * DV)), ("ln2", Array(H, BF16))]
    dense = RecordType(*attn, ("gate_up_w", _mat(2 * ID, H)), ("down_w", _mat(H, ID)))
    moe = RecordType(*attn, ("gate_w", _mat(E, H, F32)), ("gate_bias", Array(E, F32)), ("sh_gate_up_w", _mat(2 * INTER, H)),
                 ("sh_down_w", _mat(H, INTER)), ("w13", Array(E, _mat(2 * INTER, H))), ("w2", Array(E, _mat(H, INTER))))
    return RecordType(("embed", _mat(V, H)), ("dense_layers", Array(FD, dense)), ("layers", Array(NL - FD, moe)), ("norm", Array(H, BF16)),
                  ("cos_sin", _mat(MAX_POS, DR)), ("lm_head", _mat(V, H)))


def _tensors() -> dict[str, tuple[str, list[int]]]:
    attn = {"input_layernorm.weight": [H], "post_attention_layernorm.weight": [H], "self_attn.q_a_proj.weight": [QL, H],
            "self_attn.kv_a_proj_with_mqa.weight": [L + DR, H], "self_attn.q_a_layernorm.weight": [QL],
            "self_attn.q_b_proj.weight": [NH * (DN + DR), QL], "self_attn.kv_a_layernorm.weight": [L],
            "self_attn.kv_b_proj.weight": [NH * (DN + DV), L], "self_attn.o_proj.weight": [H, NH * DV]}
    out = {"model.embed_tokens.weight": ("BF16", [V, H]), "model.norm.weight": ("BF16", [H]), "lm_head.weight": ("BF16", [V, H])}
    for layer in range(NL):
        p = f"model.layers.{layer}."
        out.update({p + k: ("BF16", s) for k, s in attn.items()})
        if layer < FD:
            out.update({p + "mlp.gate_proj.weight": ("BF16", [ID, H]), p + "mlp.up_proj.weight": ("BF16", [ID, H]),
                        p + "mlp.down_proj.weight": ("BF16", [H, ID])})
            continue
        out.update({p + "mlp.gate.weight": ("BF16", [E, H]), p + "mlp.gate.e_score_correction_bias": ("F32", [E]),
                    p + "mlp.shared_experts.gate_proj.weight": ("BF16", [INTER, H]), p + "mlp.shared_experts.up_proj.weight": ("BF16", [INTER, H]),
                    p + "mlp.shared_experts.down_proj.weight": ("BF16", [H, INTER])})
        for e in range(E):
            out.update({f"{p}mlp.experts.{e}.{q}_proj.weight": ("BF16", [INTER, H]) for q in ("gate", "up")})
            out[f"{p}mlp.experts.{e}.down_proj.weight"] = ("BF16", [H, INTER])
    return out


def _write(path, tensors) -> dict[str, np.ndarray]:
    header, blobs, words, off = {}, [], {}, 0
    for i, (name, (dt, shape)) in enumerate(sorted(tensors.items())):
        n = int(np.prod(shape))
        w: np.ndarray = (np.arange(n, dtype=np.uint32) + 1000 * (i + 1)).astype(np.uint16 if dt == "BF16" else np.uint32)
        if dt == "F32":
            w = w | np.uint32(0x3F800000)
        words[name] = w
        b = w.tobytes()
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))
    return words


def test_only_a_record_with_dense_layers_and_an_array_of_layers_has_mla_sources():
    wt = _mla_weights_type()
    sources = mla_moe_sources(wt)
    assert sources is not None
    first_dense, dense, moe = sources
    assert first_dense == FD and "gate_up_w" in dense and len(moe["w13"]) == 2 * E
    assert mla_moe_sources(RecordType(*((k, t) for k, t in wt.fields if k != "dense_layers"))) is None
    assert mla_moe_sources(RecordType(*((k, t) for k, t in wt.fields if k != "layers"))) is None
    assert mla_moe_sources(RecordType(*((k, BF16 if k == "layers" else t) for k, t in wt.fields))) is None
    assert mla_moe_sources(Array(FD, BF16)) is None
    with pytest.raises(ValueError, match="an array w13 in each of its layers"):
        mla_moe_sources(RecordType(*((k, Array(2, BF16) if k == "layers" else t) for k, t in wt.fields)))


def test_a_name_that_is_no_layer_field_is_refused_by_name():
    sources = mla_moe_sources(_mla_weights_type())
    assert sources is not None
    with pytest.raises(ValueError, match="^norm: not a layer field of an MLA Program$"):
        mla_field_words(sources, "norm", lambda tensor, field: np.zeros(0, dtype=np.uint16))


def test_mla_moe_fields_read_their_own_layer(tmp_path):
    words = _write(tmp_path / "model-00001-of-00001.safetensors", _tensors())
    cos_sin = tmp_path / "cos_sin.npy"
    np.save(cos_sin, np.zeros(MAX_POS * DR, dtype=np.uint16))
    wt = _mla_weights_type()
    ws = WeightSource(wt, [tmp_path / "model-00001-of-00001.safetensors"], cos_sin)

    def cat(*names):
        return np.concatenate([words[n].astype(np.uint32) for n in names])

    expect = {
        "dense_layers[0].qkv_a_w": cat("model.layers.0.self_attn.q_a_proj.weight", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"),
        "dense_layers[0].gate_up_w": cat("model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"),
        "dense_layers[0].ln2": cat("model.layers.0.post_attention_layernorm.weight"),
        "layers[0].ln2": cat("model.layers.1.post_attention_layernorm.weight"),
        "layers[1].kv_b_w": cat("model.layers.2.self_attn.kv_b_proj.weight"),
        "layers[1].sh_gate_up_w": cat("model.layers.2.mlp.shared_experts.gate_proj.weight", "model.layers.2.mlp.shared_experts.up_proj.weight"),
        "layers[0].w13": cat(*(f"model.layers.1.mlp.experts.{e}.{q}_proj.weight" for e in range(E) for q in ("gate", "up"))),
        "layers[0].w2": cat(*(f"model.layers.1.mlp.experts.{e}.down_proj.weight" for e in range(E))),
        "layers[0].gate_w": words["model.layers.1.mlp.gate.weight"].astype(np.uint32) << 16,
        "layers[0].gate_bias": words["model.layers.1.mlp.gate.e_score_correction_bias"],
    }
    for name, want in expect.items():
        np.testing.assert_array_equal(ws._field_words(name).astype(np.uint32), want, err_msg=name)
    for lo, hi, name in _field_intervals(wt, ""):
        if name.startswith(("dense_layers[", "layers[")):
            assert ws.words(lo, hi).size == hi - lo, name


def test_a_top_field_is_its_checkpoint_tensors_and_cos_sin_is_the_fixture(tmp_path):
    """`embed`, `norm` and `lm_head` are their checkpoint tensors' words. `cos_sin` names no checkpoint tensor: it is the fixture's
    words, under the base tables and under the GPT-NeoX family's (a `config` whose `NORM` is `layernorm`), whose `norm_b` is read too."""
    cos_sin = tmp_path / "cos_sin.npy"
    fixture: np.ndarray = np.arange(MAX_POS * DR, dtype=np.uint16) + 7
    np.save(cos_sin, fixture)
    wt = _mla_weights_type()
    words = _write(tmp_path / "base.safetensors", _tensors())
    ws = WeightSource(wt, [tmp_path / "base.safetensors"], cos_sin)
    for field, tensor in (("embed", "model.embed_tokens.weight"), ("norm", "model.norm.weight"), ("lm_head", "lm_head.weight")):
        np.testing.assert_array_equal(ws._field_words(field), words[tensor], err_msg=field)
    np.testing.assert_array_equal(ws._field_words("cos_sin"), fixture)
    with pytest.raises(KeyError, match="^'norm_b'$"):
        ws._field_words("norm_b")

    neox = {"gpt_neox.embed_in.weight": ("BF16", [V, H]), "gpt_neox.final_layer_norm.weight": ("BF16", [H]),
            "gpt_neox.final_layer_norm.bias": ("BF16", [H]), "embed_out.weight": ("BF16", [V, H])}
    words = _write(tmp_path / "neox.safetensors", neox)
    ws = WeightSource(wt, [tmp_path / "neox.safetensors"], cos_sin, config={"NORM": "layernorm"})
    for field, tensor in (("embed", "gpt_neox.embed_in.weight"), ("norm", "gpt_neox.final_layer_norm.weight"),
                          ("norm_b", "gpt_neox.final_layer_norm.bias"), ("lm_head", "embed_out.weight")):
        np.testing.assert_array_equal(ws._field_words(field), words[tensor], err_msg=field)
    np.testing.assert_array_equal(ws._field_words("cos_sin"), fixture)
