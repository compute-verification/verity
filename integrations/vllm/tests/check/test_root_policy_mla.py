"""root_policy.py (protected), multi-head latent attention's roots on a synthetic log: MLAAttention's absorbed `W_UK_T` / `W_UV` pass
as aliases only as the view of a validated `kv_b_proj.weight` that C's MLA dimensions give; the `_k/_v/_q/_prob_scale` placeholders
pass as ignored only in the frozen slots whose branch the same event's arguments rule out; an MLA configuration's fields are compared
to the header's recorded config, and a field whose key is not recorded fails.  The header records every key the config root reads."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from verity.ml.circuits.types import Array, Record
from verity_vllm.check import root_policy
from verity_vllm.engine import run_facts
from verity_vllm.observe.events import DispatchEnter, Root, RunHeader, TritonLaunch, ViewDesc
from verity_vllm.program.registry.prims import BF16

NH, DN, DR, DV, L = 2, 3, 2, 2, 4                       # heads, QK_NOPE, QK_ROPE, V_HEAD, KV_LORA
ROWS, HS = NH * (DN + DV), (DN + DV) * L
LAYER = "model.layers.0.self_attn"
SCALES = ("_k_scale", "_v_scale", "_q_scale", "_prob_scale")
HF = {"model_type": "glm4_moe_lite", "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": NH, "num_key_value_heads": NH,
      "head_dim": DR, "intermediate_size": 12, "vocab_size": 16, "rms_norm_eps": 1e-5, "rope_theta": 1e6, "max_position_embeddings": 8,
      "tie_word_embeddings": False, "q_lora_rank": 3, "kv_lora_rank": L, "qk_nope_head_dim": DN, "qk_rope_head_dim": DR,
      "v_head_dim": DV, "first_k_dense_replace": 0, "moe_intermediate_size": 4, "n_routed_experts": 2, "num_experts_per_tok": 1,
      "norm_topk_prob": True, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 1.5, "n_shared_experts": 1}
C = {"H": 8, "NL": 1, "NH": NH, "KVH": NH, "D": DN + DR, "I": 4, "V": 16, "EPS": 1e-5, "ROPE_THETA": 1e6, "MAX_POS": 8,
     "tie_word_embeddings": False, "Q_LORA": 3, "KV_LORA": L, "QK_NOPE": DN, "QK_ROPE": DR, "V_HEAD": DV, "FIRST_DENSE": 0,
     "I_DENSE": 12, "MOE_I_FULL": 4, "NUM_EXPERTS": 2, "TOPK": 1, "NORM_TOPK_PROB": True, "RENORMALIZE": True, "N_GROUP": 1,
     "TOPK_GROUP": 1, "ROUTED_SCALING": 1.5, "SHARED_EXPERTS": 1}
CONCAT, UPDATE = "_C_cache_ops.concat_and_cache_mla.default", "vllm.unified_mla_kv_cache_update.default"
STAGE1 = "vllm.v1.attention.ops.triton_decode_attention._fwd_grouped_kernel_stage1"


def _v(alloc: int, offset: int, shape: tuple, strides: tuple, dtype: str = "bfloat16") -> ViewDesc:
    return ViewDesc(alloc=alloc, offset=offset, shape=shape, strides=strides, dtype=dtype, device="cuda:0")


KV_B = _v(1, 0, (ROWS, L), (L, 1))
W_UK_T = _v(1, 0, (NH, DN, L), (HS, L, 1))
W_UV = _v(1, DN * L, (NH, L, DV), (HS, 1, L))
K_SCALE = _v(2, 0, (), (), "float32")
CACHE = _v(9, 0, (4, 16, L + DR), ((L + DR) * 16, L + DR, 1))


@pytest.fixture(autouse=True)
def _mla_record(monkeypatch):
    """The one weights field these roots read (the MLA weights Record is the row's; main's `weights_type` has no MLA branch)."""
    monkeypatch.setattr(root_policy.analytic, "weights_type", lambda C: Record(("layers", Array(1, Record(("kv_b_w", Array(ROWS, Array(L, BF16))))))))


def _profile(config: dict | None = None) -> SimpleNamespace:
    fields = {f"{LAYER}.kv_b_proj.weight": ("layers", 0, "kv_b_w")}
    return SimpleNamespace(id="test-mla", config=dict(config or C), weight_field=fields.get)


def _header(hf: dict | None = None, verified: bool = True) -> RunHeader:
    return RunHeader("verity-capture/v1", "", "p", {"model": {"weights_sha256": {"model.safetensors": "00"}}},
                     {"repo": None, "config": dict(HF if hf is None else hf)}, {"requests": []}, {"weights_verified": verified})


def _reads(kv_cache_dtype: str = "auto", buffer_dtype: str = "bfloat16", k_scale_slot: int = 5, src_sha: str | None = None) -> list:
    concat, update, stage1 = (root_policy.CONDITIONAL_IGNORED_SLOTS[k] for k in (CONCAT, UPDATE, f"triton:{STAGE1}"))
    args = [_v(10, 0, (1, L), (L, 1)), _v(11, 0, (1, DR), (DR, 1)), CACHE, _v(12, 0, (1,), (1,), "int64"), kv_cache_dtype, K_SCALE]
    args[k_scale_slot], args[5] = args[5], args[k_scale_slot]
    buf = dataclasses.replace(CACHE, dtype=buffer_dtype)
    return [DispatchEnter(1, None, 0, "forward", UPDATE, update.pin, [args[0], args[1], {"layer": 0}, kv_cache_dtype, K_SCALE], {}, 0),
            DispatchEnter(2, 1, 0, "forward", CONCAT, concat.pin, args, {}, 0),
            TritonLaunch(3, None, 0, "forward", STAGE1, src_sha or stage1.pin,
                         {"K_Buffer": buf, "V_Buffer": buf, "k_scale": K_SCALE, "v_scale": K_SCALE}, {}, [1, 1, 1], 4, 2, 0)]


def _log(*, uv: ViewDesc = W_UV, header: RunHeader | None = None, reads: list | None = None) -> list:
    roots = [Root(KV_B, "weight", f"{LAYER}.kv_b_proj.weight"), Root(W_UK_T, "weight", f"{LAYER}.mla_attn.mla_attn.W_UK_T"),
             Root(uv, "weight", f"{LAYER}.mla_attn.mla_attn.W_UV")]
    roots += [Root(_v(2 + i, 0, (), (), "float32"), "buffer", f"{LAYER}.mla_attn.mla_attn.{s}") for i, s in enumerate(SCALES)]
    return [header or _header(), *roots, *(_reads() if reads is None else reads)]


def _evaluate(events: list, profile: SimpleNamespace | None = None) -> dict:
    return root_policy.evaluate(events, None, profile or _profile(), weight_bytes=False)


def test_mla_roots_pass_in_their_classes():
    rep = _evaluate(_log())
    assert rep["status"] == "pass", (rep["first_failures"], rep["problems"])
    assert rep["new_root_classes"] == [] and rep["classes"]["alias"]["validated"] == 2 and rep["classes"]["ignored"]["validated"] == 4
    assert {(a["name"].rsplit(".", 1)[1], a["alias_of"]) for a in rep["alias_claims"]} == {("W_UK_T", f"{LAYER}.kv_b_proj.weight"),
                                                                                           ("W_UV", f"{LAYER}.kv_b_proj.weight")}
    assert set(rep["ignored_root_consumers"]) == {CONCAT, UPDATE, f"triton:{STAGE1}"}
    assert rep["config_root"]["status"] == "pass" and rep["config_root"]["fields"]["NUM_EXPERTS"] == "hf-config"


@pytest.mark.parametrize("uv, why", [
    (dataclasses.replace(W_UV, offset=0), "!= ("),                                        # the k_nope rows, not the v rows
    (dataclasses.replace(W_UV, strides=(HS, L, 1), shape=(NH, DV, L)), "!= ("),          # not transposed
    (dataclasses.replace(W_UV, alloc=7), "allocation/dtype"),                             # a copy: its words are not proven the weight's
])
def test_kv_b_view_other_than_c_gives_fails(uv, why):
    rep = _evaluate(_log(uv=uv))
    bad = [e for e in rep["roots"] if e["name"].endswith("W_UV")]
    assert rep["status"] == "fail" and bad[0]["status"] == "fail" and why in bad[0]["reason"]
    assert any(a["name"].endswith("W_UV") and a["status"] == "fail" for a in rep["alias_claims"])


def test_kv_b_view_of_an_unvalidated_weight_fails():
    rep = _evaluate(_log(header=_header(verified=False)))
    views = [e for e in rep["roots"] if e["class"] == "alias"]
    assert len(views) == 2 and all(e["status"] == "fail" and "not a validated weight root" in e["reason"] for e in views)


@pytest.mark.parametrize("reads", [
    _reads(kv_cache_dtype="fp8_e4m3"),                       # the cache kernels' fp8 branch dereferences the scale
    _reads(buffer_dtype="float8_e4m3fn"),                    # the decode kernel's fp8 branch multiplies by it
    _reads(k_scale_slot=0),                                  # the scale in a slot the kernel reads as data
    _reads(src_sha="0" * 64),                                # a decode kernel whose text the slot reading is not of
])
def test_scale_read_outside_its_non_semantic_slot_fails(reads):
    rep = _evaluate(_log(reads=reads))
    ignored = [e for e in rep["roots"] if e["class"] == "ignored"]
    assert rep["status"] == "fail" and len(ignored) == 4 and all(e["status"] == "fail" for e in ignored)


def test_mla_config_fields_need_their_keys_recorded():
    base = {k: v for k, v in HF.items() if k in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                                                 "head_dim", "intermediate_size", "vocab_size", "rms_norm_eps", "rope_theta",
                                                 "max_position_embeddings", "tie_word_embeddings")}
    rep = _evaluate(_log(header=_header(hf=base)))
    cr = rep["config_root"]
    assert rep["status"] == "fail" and cr["status"] == "fail" and cr["mismatches"] == {}
    assert {u.split(" ")[0] for u in cr["unrecorded"]} == {"D", "I", *root_policy.MLA_CONFIG_KEYS} - {"I_DENSE"}   # intermediate_size is a base key


@pytest.mark.parametrize("key, value, field", [("n_routed_experts", 3, "NUM_EXPERTS"), ("qk_nope_head_dim", DN + 1, "QK_NOPE"),
                                               ("moe_intermediate_size", 8, "I"), ("intermediate_size", 16, "I_DENSE")])
def test_mla_config_field_differing_from_the_checkpoint_fails(key, value, field):
    rep = _evaluate(_log(header=_header(hf={**HF, key: value})))
    assert rep["config_root"]["status"] == "fail" and field in rep["config_root"]["mismatches"]


def test_a_configuration_without_mla_keeps_the_base_rule():
    """No `KV_LORA`: `D` is `head_dim` and `I` is `intermediate_size` as before, and no MLA key is required."""
    dense = {k: v for k, v in C.items() if k not in root_policy.MLA_CONFIG_KEYS}
    rep = _evaluate([_header(hf={k: v for k, v in HF.items() if not k.startswith(("q_", "kv_", "qk_", "v_"))})], _profile(dense))
    cr = rep["config_root"]
    assert cr["status"] == "fail" and set(cr["mismatches"]) == {"D", "I"} and "unrecorded" not in cr
    assert set(cr["fields"]) == {"H", "NL", "NH", "KVH", "D", "I", "V", "EPS", "ROPE_THETA", "MAX_POS", "tie_word_embeddings"}


def _subset(**hf) -> dict:
    model_config = SimpleNamespace(hf_config=SimpleNamespace(**hf))
    return run_facts.model_config_subset(SimpleNamespace(llm=SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(model_config=model_config)))))


def test_the_header_records_every_key_the_mla_config_root_reads():
    keys = {"qk_nope_head_dim", "qk_rope_head_dim", "moe_intermediate_size", *(k for ks in root_policy.MLA_CONFIG_KEYS.values() for k in ks)}
    assert _subset(**{k: 1 for k in keys}).keys() == keys
