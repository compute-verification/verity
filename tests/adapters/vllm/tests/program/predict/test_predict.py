"""verity_vllm.program.predict: predicted Programs equal the traced Builds of record, and the scorer says where they do not.

The pinned digests are Programs the Build traced on rtxpro6000 (cc 12.0, 188 SMs) from the checkpoints' `config.json`
(`fixtures/`): SmolLM2-135M from cov-k01-10, Llama-3.2-1B from cov-k05-7, TinyLlama-1.1B from cov-k04-7, Qwen2.5-0.5B from
cov-k03-8 (its biased q/k/v linear one `GemmBias_v2{K,N,DOT}` Call), Phi-3-mini from cov-k07-3, Qwen3-4B from cov-k09-3, Gemma-2-2B from cov-k06-5, Llama-3.2-1B's TP2 ranks from cov-p000,
Gemma-2-2B's from cov-p058-2, Qwen3-30B-A3B's from cov-p081 and OLMoE's from cov-p069.
A library change that moves a Definition moves the traced digest too: re-trace the unit (`verity-vllm build --wrapper
step --M 1 --pos0 8`, CPU only) and pin the new digest with the predictor's.
`SiluMul_v2{I}` as the `silu_mul` kind moved every pin but Gemma-2's (GeGLU): each is the Build's re-trace of its unit from the
same checkpoints on the declared rtxpro6000 target (CPU only), and with `SiluMul_v1{I}` in its place
the predictor gives the rows' traced digests.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.config import WORKLOADS, RowEngine, workload_file
from verity_vllm.definitions.registry.tp_allreduce import settings_of
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR, pinned_hf_config
from verity_vllm.program.model import INT8_EXACT_LMHEAD_QUANT, INT8_EXACT_QUANT
from verity_vllm.program.predict import build as BUILD
from verity_vllm.program.predict import dsa
from verity_vllm.program.predict import node as NODE
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.build import Unexpressible, build_program
from verity_vllm.program.predict.model_shape import model_shape, weight_leaves
from verity_vllm.program.predict.phi import Phi

FIX = Path(__file__).parent / "fixtures"
GLM53 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
H200 = {"compute_capability": [9, 0], "num_sms": 132}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": 32}
#: TINY's exact-int8 checkpoint (`engine.int8_checkpoint`'s v1) on a target serving its linears int8 (serve-openai's verity_int8)
INT8 = {**TINY, "quantization_config": dict(INT8_EXACT_QUANT)}
INT8_LINEARS = {**RTXPRO6000, "int8_exact": "linears"}


def _phi(cfg: dict, requests: list[dict] | None = None, **kw) -> Phi:
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, workload={"requests": requests or []}, **{"target": dict(RTXPRO6000), **kw})


def _row_phi(row: str, case: str, hf_home: Path, rank: int = 0) -> Phi:
    """A workload of record's φ at `rank` as `score.phi_of` makes it, the checkpoint's config.json the package's byte copy of it
    (`hf_configs/<case>.config.json`)."""
    w = json.loads(workload_file(WORKLOADS / f"{row}.json").read_text())
    cp = Path(S.hf_config_path(str(hf_home), w["model"], RowEngine.of(w).revision))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes((HF_CONFIG_DIR / f"{case}.config.json").read_bytes())
    phi = S.phi_of({"row": row, "workload": w}, hf_home=str(hf_home))
    return Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}})


@pytest.mark.parametrize("model,digest", [
    ("smollm2-135m", "c3fccb15a8daec8f573d5dd95fc012c0acd3cc1d0d88b4578f84dfe364a181b6f69f0d0ffd2988f0cc02728d942d3c2641eabb69b676656519c3a73d71ae0d88"),
    ("llama32-1b", "af2574da8ac1c38c12cb022b9a3f1e5bdd62869c57cecf0ecade142eca124a168fa440c413a5e7c68c9a1637d7bb99fb4fefd88942340e357fed293e62952c3c"),
    ("tinyllama-11b", "2a8bee32041927d0a2978c925360319a74e6b5dc492e89ae9eb1046e8428be605953858c99ab25b7a57b645c4641d7ab291b9f08d416541c7b8debcdbf3b71eb"),
    ("qwen25-05b", "278cf9ddf3d46a7fe57cef6a1b6430603a5b5756b26beaf9c69ae3c3d9b6a76da9250b0298caeeb10fcb86899a4c96330cf3b38b3879aab79f2323373bf96f6e"),
    ("phi3-mini", "d06e4c508379231cad05e05cffdd60581010882258fc92447c1c2b50b9dc6ac470e36be3aeec30d3d36d488ecf243a3dae6f01d58c64e6795717b324d2cbe095"),
    ("qwen3-4b", "cfc1ce91aa29288c2e480eeb299e72c46eb02c2236ca5388d08f71b638b8076fc2ed8d5d58ae97d0e84d030dc3a90bb0e9fe129973105f29384823e2e6663684"),
    ("olmoe-1b-7b", "ae54a3fde31a57b61b48f09ef77838b8c24238e8f223d9d6b16c3edea30385538f95c6588e9719694663b32e94f2cfd60a71c7fed05150be3ba89fdf71282bb8"),
    ("qwen3-30b-a3b", "a3e6e95551920a8583751442cf11ff8bea82d69dcb3a4e6b5ad0e0044c07915aebbcc97a40e968702fbe223f4ace250af6e9882ab823b9885e11decef4741728"),
    ("gemma2-2b", "a5ec0d9212dee862f849ec9332dabfb2429341abc268f4ba9e2298f5a70e731c680731853d202e6069eee950f5f40c39e8d090d015de0c8ad49204957a90ed73"),
])
def test_step_program_equals_the_traced_build(model, digest):
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text())), "step", M=1, pos0=8)
    cls = {"qwen25-05b": "Qwen2ForCausalLM", "phi3-mini": "Phi3ForCausalLM", "qwen3-4b": "Qwen3ForCausalLM", "olmoe-1b-7b": "OlmoeForCausalLM",
           "qwen3-30b-a3b": "Qwen3MoeForCausalLM", "gemma2-2b": "Gemma2ForCausalLM"}.get(model, "LlamaForCausalLM")
    assert r.function == f"Derived[vLLMStep[{cls}]]_v1{{SRC={{d9c6ce11eb35}}}}"
    assert r.digest == digest


@pytest.mark.slow
def test_request_and_workload_programs_equal_the_traced_build():
    """SmolLM2-135M B1 256/32 greedy: the request Program (LP 256, T 31) and GP-01's workload Program over it.
    About 20 s: the only test that composes a workload, which the step pins cannot reach."""
    phi = _phi(json.loads((FIX / "smollm2-135m.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}])
    r = build_program(phi, "request", LP=256, T=31)
    assert r.digest == "5d27500cfc13de9c702784aba413700764b20c9c0b652940c91e3d54c2083802dd730213feb3712295b6274cdee059be514bdfbb457c99003a22eb7bfc6a0c4e"
    assert r.n_root_nodes == 84698
    assert S.predict_workload(phi)["digest"] == "647476fb737cd62b89936075f5fb6a3d9e66a92379ff5dcaca2d14f62d0d6e32b5619fb3fc797ae836e09325983e25343b3b15cb8596dc9d1b4b8506292fcc3c"


@pytest.mark.parametrize("model,function,rank,digest", [
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 0, "a96055dbae7e4ff741181c4fe29e222074541ed56b9a91fb5879cbf12b00cb574a8d5a9f6284a0731285f8556d5a08ae88132c92d37d02df0846aead8887df53"),
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 1, "d2ac2f2f20834ab53a2046b37ad0a1adac6028e8c409790bd6f229e46f283e23636b6e49fa1ee24be2c68aeecf27d552e2de220a62ce5f14d49a7f39fa74b2aa"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 0, "1b021232b0d3ab2a3e433dd5d9aaf82c07598b897d165093d454e14044b2dbd3c21d35e4b4fcb5e875818c452b0adcf531b29452c8080378a1c1197cb2f9aa60"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 1, "b12ecf5e2a5b995218d1133ce12172353edb32766df283ab1fcb63cf25500fe3c45ff8291c8aa5289dc617b879bc629a2d30ed78f203e5be219cff6b30ffd2e9"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 0, "1105d3f8e9932d7fa4753e78acd8eba53ab4de9368e908157f32682990f18d99d4a56dd483a70403a5c8da73e72986c144b63dba856276d3324c54adbfdc1161"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 1, "be534afc7a520559229475e6ddd491ef731bcff1ca946a5c01fb575bba26d8401001301ad7a96d8ae366fb77cae7aae03e066964d00cfdb038842027b9dc95fa"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 0, "3a80f8e277c8fcafee8063b1a0948d25064b91daa86821fd9537132a4d459934e0c989381228cf2ddf5d58d3de44bb2dc6fe9e2b739ecc278d33367d0f6fb0f6"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 1, "dd3e5ed28881c73e9b709990019e1e628c21cdcc218df8b8c005ec122dc663108db7a8f414c71e143d793984d0e4246db7fe26e4f412c72dba1ab05fc22e6479"),
])
def test_tensor_parallel_rank_step_equals_the_traced_build(model, function, rank, digest):
    """Llama-3.2-1B (cov-p000), Gemma-2-2B (cov-p058-2), Qwen3-30B-A3B (cov-p081) and OLMoE (cov-p069) at TP2: each rank's step
    over its shards, with a peer Input per collective in call order (the embedding's all-reduce, each layer's o and down / MoE
    all-reduces, the logits all-gather) and the two partials in rank order.  Gemma-2's normalizer scales the reduced embedding and
    its soft-cap the gathered logits; the fused MoE keeps its router and experts whole over each expert's width shard; OLMoE's q / k
    norm over all heads reads its q and k rows all-gathered, then keeps this rank's slice."""
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text()), tp={"world": 2, "rank": rank}), "step", M=1, pos0=8)
    assert r.function == "Derived[vLLMStep[" + function
    assert r.digest == digest


@pytest.mark.slow
def test_tensor_parallel_rank_workload_equals_the_traced_build():
    """cov-p000's rank-1 workload Program (B1 256/32 at TP2): the only test that composes a rank's request component, whose
    `peers_*` Inputs GP-01 keeps request-local.  About 15 s."""
    phi = _phi(json.loads((FIX / "llama32-1b.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}], tp={"world": 2, "rank": 1})
    assert S.predict_workload(phi)["digest"] == "70468fe0fee095b09765883d67013afffda931d609ae63b67c26d81da0d445942ff49dd254f8d2477d7f721c06b2a7758237b3c2ded6a1dbccdb0eaf3a7c2f4a"


@pytest.mark.parametrize(("T", "error"), [("x", ValueError), ([1], TypeError)])
def test_a_step_given_a_T_that_int_refuses_raises_ints_error(T, error):
    """A `step` is built for no T, yet its launch (`common.in_launch`) takes `int` of a T it is given: one that `int` refuses
    is `int`'s error, and one it takes (5, 2.5) builds the Program of a step given none."""
    step = build_program(_phi(TINY), "step", M=1, pos0=8)
    assert [build_program(_phi(TINY), "step", M=1, pos0=8, T=t).digest for t in (5, 2.5)] == [step.digest] * 2
    with pytest.raises(error, match=r"^(invalid literal for int\(\) with base 10: 'x'|int\(\) argument must be .* not 'list')$"):
        build_program(_phi(TINY), "step", M=1, pos0=8, T=T)


def test_a_requests_launch_is_looked_up_by_the_LP_and_T_given(monkeypatch):
    """`build_program` and `build_node_program` hand `common.launch_batch` and `launch_rows` the LP and T they were given, not
    `int` of them: each reader takes `int` itself, and `launch_batch`'s refusal prints them as given. The Program is the one of
    `int` of each."""
    seen: list[tuple] = []
    for module in (BUILD, NODE):
        monkeypatch.setattr(module, "launch_batch", lambda phi, LP, T: seen.append(("batch", LP, T)))
        monkeypatch.setattr(module, "launch_rows", lambda phi, LP, T, C=0: seen.append(("rows", LP, T)))
    one, two = _phi(TINY), _phi(TINY, tp={"world": 2, "rank": 0})
    whole = (build_program(one, "request", LP=4, T=2).digest, NODE.build_node_program(two, "request", LP=4, T=2).digest)
    seen.clear()
    assert (build_program(one, "request", LP="4", T=2.5).digest, NODE.build_node_program(two, "request", LP="4", T=2.5).digest) == whole
    assert seen == [("batch", "4", 2.5), ("rows", "4", 2.5), ("batch", "4", 2.5), ("rows", "4", 2.5), ("rows", "4", 2.5)]


def test_dsa_refuses_outside_its_domain(tmp_path):
    phi = _row_phi(GLM53, "GLM53", tmp_path)
    with pytest.raises(Unexpressible, match=re.escape("tensor parallel world 2 (TP8, the worlds a Build has traced)")):
        build_program(Phi.from_json({**phi.to_json(), "tp": {"world": 2, "rank": 0}}), "step")
    with pytest.raises(Unexpressible, match=re.escape("a target without a device name")):
        build_program(Phi.from_json({**phi.to_json(), "target": {k: v for k, v in phi.target.items() if k != "card"}}), "step")


def test_the_execution_label_names_the_batch_invariance(tmp_path):
    """An `eager` row's Programs are of vLLM's default kernels and a `bi-*` row's of the batch-invariant ones, whatever TargetProfile's
    own default: a target block that leaves `batch_invariant` out takes the label's, and a target or env that says otherwise is
    refused, never predicted under the label."""
    assert _phi(TINY).target_profile().batch_invariant and not _phi(TINY, execution="eager").target_profile().batch_invariant
    phi = _row_phi(GLM53, "GLM53", tmp_path)
    assert not phi.target_profile().batch_invariant
    with pytest.raises(Unexpressible, match=re.escape("a target labelled 'bi-eager' (batch invariant True) for the 'eager' row")):
        build_program(Phi.from_json({**phi.to_json(), "target": {**phi.target, "batch_invariant": True}}), "step")
    with pytest.raises(Unexpressible, match=re.escape("env VLLM_BATCH_INVARIANT=1 under a target that sets '0'")):
        build_program(Phi.from_json({**phi.to_json(), "engine": {**phi.engine, "env": {"VLLM_BATCH_INVARIANT": "1"}}}), "step")


def test_dsa_reads_an_engine_without_env_as_under_no_switch(tmp_path):
    """A φ whose engine states no `env` runs the sparse-attention decoder's all-reduce under no switch (`tp_allreduce.settings_of`
    of an empty env), as one whose `env` is empty does, and a stated switch is read.  The first two part at a row past INDEX_TOPK
    alone: an engine that states no environment is refused none, and an empty one states no canonical top-k."""
    phi = _row_phi(GLM53, "GLM53", tmp_path)
    doc, stated = phi.to_json(), {k: v for k, v in phi.engine.items() if k != "env"}
    invariant, off = {"VLLM_BATCH_INVARIANT": "1"}, bool(stated.get("disable_custom_all_reduce"))
    absent, empty, switched = (dsa.Net(Phi.from_json({**doc, "engine": e}), 1, 8)
                               for e in (stated, {**stated, "env": {}}, {**stated, "env": invariant}))
    assert absent.settings == empty.settings == settings_of(off, {}) != settings_of(off, invariant) == switched.settings
    assert absent.long_rows is None and "VERITY_DSA_TOPK=None" in str(empty.long_rows)


def test_a_phi_that_leaves_a_block_out_declares_nothing_in_it():
    """φ read from a document without `engine` or `workload` declares no engine argument and no request, and a model without
    `hf_config` has an empty one.  A request's `max_tokens` is its own, else its `sampling_params`', else 0."""
    doc = {k: v for k, v in Phi(model={"repo": "org/m"}).to_json().items() if k not in ("engine", "workload")}
    phi = Phi.from_json(doc)
    assert (phi.engine, phi.workload, phi.hf_config, phi.requests, phi.request_shapes()) == ({}, {}, {}, [], [])
    reqs = [{"prompt_len": 3}, {"prompt_len": 3, "sampling_params": {"max_tokens": 5}},
            {"prompt_len": 3, "max_tokens": 7, "sampling_params": {"max_tokens": 5}}]
    assert Phi(workload={"requests": reqs}).request_shapes() == [(3, -1), (3, 4), (3, 6)]


def test_a_request_states_its_prompt_length_one_of_three_ways_and_one_that_states_none_has_no_length():
    """A request's prompt length is the count of its `prompt_token_ids`, else its `prompt_len`, else its `input_len`: an empty or
    absent `prompt_len` gives way to `input_len`.  A request that states none of the three is `int`'s TypeError on None, and
    names no field."""
    def shapes(*requests: dict) -> list[tuple[int, int]]:
        return Phi(workload={"requests": [{"max_tokens": 2, **r} for r in requests]}).request_shapes()

    stated: list[dict] = [{"prompt_token_ids": [7, 8, 9], "prompt_len": 5, "input_len": 6}, {"prompt_len": 5, "input_len": 6}, {"input_len": 6}]
    assert shapes(*stated) == [(3, 1), (5, 1), (6, 1)]
    empties: list[object] = [None, 0, "", [], {}, False]
    for empty in empties:
        assert shapes({"prompt_len": empty, "input_len": 6}) == [(6, 1)], empty
    for none in ({}, {"prompt_len": None}, {"prompt_len": 0, "input_len": None}, {"prompt_token_ids": None}):
        with pytest.raises(TypeError, match="NoneType"):
            shapes(none)


def test_a_row_that_pins_no_revision_names_no_snapshot_dir(tmp_path):
    """`score.phi_of` finds a row's config.json under `hf_home` by its checkpoint's revision.  A row whose engine block pins none
    names no dir there, which is `os.path.join`'s TypeError; given a `config` it is that config's φ, with no revision."""
    w = json.loads(workload_file(WORKLOADS / f"{GLM53}.json").read_text())
    bare = {**w, "engine": {k: v for k, v in w["engine"].items() if k != "revision"}}
    with pytest.raises(TypeError, match="NoneType"):
        S.phi_of({"row": GLM53, "workload": bare}, hf_home=str(tmp_path))
    config = str(HF_CONFIG_DIR / "GLM53.config.json")
    full, pinless = (S.phi_of({"row": GLM53, "workload": doc}, config=config).to_json() for doc in (w, bare))
    assert pinless["model"]["revision"] is None and {**pinless, "model": full["model"]} == full


def test_phi_of_opens_the_config_by_the_name_given():
    """`score.phi_of` hands ``config`` to `open` as it is given: the file's name in bytes is read as the same name in a str is."""
    w = json.loads(workload_file(WORKLOADS / f"{GLM53}.json").read_text())
    config = str(HF_CONFIG_DIR / "GLM53.config.json")
    in_bytes: Any = os.fsencode(config)
    assert S.phi_of({"row": GLM53, "workload": w}, config=in_bytes).to_json() == S.phi_of({"row": GLM53, "workload": w}, config=config).to_json()


def test_a_workload_manifest_without_requests_gives_a_phi_of_none():
    """`score.phi_of` on a row whose declared manifest states no `requests`: φ's workload holds none, and the rest of φ is the
    row's as with them."""
    w = json.loads(workload_file(WORKLOADS / f"{GLM53}.json").read_text())
    config = str(HF_CONFIG_DIR / "GLM53.config.json")
    full = S.phi_of({"row": GLM53, "workload": w}, config=config)
    bare = S.phi_of({"row": GLM53, "workload": {k: v for k, v in w.items() if k != "requests"}}, config=config)
    assert bare.workload == {"requests": []} and len(full.requests) == len(w["requests"]) > 0
    assert {**bare.to_json(), "workload": full.workload} == full.to_json()


def test_sampling_follows_the_builds_policy():
    one: dict = {"sweep": {"sampling": "stoch-t0.8-p0.95", "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, "sampling_seeds": [7]},
           "requests": [{"prompt_len": 8, "max_tokens": 3, "seed": 7}]}
    assert S.sampling_of(one, _phi(TINY)) == STOCH
    assert S.sampling_of({**one, "requests": one["requests"] * 8}, _phi(TINY)) == STOCH
    assert S.sampling_of({"requests": [{"prompt_len": 8, "max_tokens": 3}]}, _phi(TINY)) == {"kind": "greedy"}
    with pytest.raises(Unexpressible, match="the Build refuses the policy"):
        S.sampling_of({**one, "sweep": {**one["sweep"], "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 50}}}, _phi(TINY))


def test_weights_record_follows_the_module_order():
    C = model_shape(TINY)
    names = [n for n, _ in weight_leaves(C)]
    assert names[:3] == ["model.embed_tokens.weight", "model.layers.0.self_attn.qkv_proj.weight",
                         "model.layers.0.self_attn.o_proj.weight"]
    assert "lm_head.weight" not in names  # tied: the embedding's leaf
    assert len(names) == 1 + 6 * 2 + 1 + 1 + 4 * 2
    shards = [(n.split("layers.0.")[-1], t) for n, t in weight_leaves(C, 2)[:7]]
    assert [(n, t.leaves) for n, t in shards] == [
        ("model.embed_tokens.weight", 128 * 64), ("self_attn.qkv_proj.weight", (2 + 2 * 1) * 16 * 64), ("self_attn.o_proj.weight", 64 * 32),
        ("mlp.gate_up_proj.weight", 128 * 64), ("mlp.down_proj.weight", 64 * 64), ("input_layernorm.weight", 64), ("post_attention_layernorm.weight", 64)]
    untied = [n for n, _ in weight_leaves(model_shape({**TINY, "tie_word_embeddings": False}))]
    assert untied.index("lm_head.weight") == untied.index("model.norm.weight") + 1
    biased = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen2", "use_sliding_window": False, "sliding_window": 32}))]
    assert biased[1:4] == ["model.layers.0.self_attn.qkv_proj.weight", "model.layers.0.self_attn.qkv_proj.bias",
                           "model.layers.0.self_attn.o_proj.weight"]
    normed = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen3"}))]
    assert normed[2:5] == ["model.layers.0.self_attn.o_proj.weight", "model.layers.0.self_attn.q_norm.weight",
                           "model.layers.0.self_attn.k_norm.weight"]
    moe = weight_leaves(model_shape(json.loads((FIX / "olmoe-1b-7b.config.json").read_text())))
    assert [n.split("layers.0.")[-1] for n, _ in moe[1:10]] == [
        "self_attn.qkv_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight", "mlp.gate.weight",
        "mlp.experts.w13_weight", "mlp.experts.w2_weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    gemma = [n for n, _ in weight_leaves(model_shape(json.loads((FIX / "gemma2-2b.config.json").read_text())))]
    assert [n.split("layers.0.")[-1] for n in gemma[1:9]] == [
        "self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
        "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"]
    assert gemma.index("model.normalizer") == gemma.index("model.norm.weight") + 1 == \
        gemma.index("model.layers.0.self_attn.rotary_emb.cos_sin_cache") - 1


@pytest.mark.parametrize("target,kind", [({"compute_capability": [8, 0], "num_sms": 108}, "GemmBias_v1"), (RTXPRO6000, "GemmBias_v2")])
def test_the_biased_linear_follows_the_targets_rule(target, kind):
    """Qwen2's q/k/v bias: one `GemmBias_v1{K,N}` Call per row under the accepted kinds (A100, FA2), and where the target's record
    names a biased-linear step (`gemm_bias_dot`: rtxpro6000) one `GemmBias_v2{K,N,DOT}` Call with the target's step; never the GEMM
    then `BiasAdd_v1`, which only a family whose record names no step would keep."""
    cfg = {**TINY, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "use_sliding_window": False, "sliding_window": 32}
    r = build_program(Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=target, workload={"requests": []}), "step")
    names = {re.findall(r"^\w+", k)[0] for k in r.desc["definitions"]}
    assert names & {"GemmBias_v1", "GemmBias_v2", "BiasAdd_v1"} == {kind}


def test_the_int8_linears_follow_the_build_of_records_rule():
    """An exact-int8 checkpoint on a target serving its linears int8: each decoder linear is its quantizer `Int8RowQuant_v1{K}`
    then its product `ScaledMmS8_v1{K,N,DOT}` (`rules.vllm_bindings.int8`), reading the linear's s8 weight [N, K] and its f32
    row scales [N, 1], which the weights record carries in that order; the LM head (recipe v1's ``lm_head``, not converted)
    stays the bf16 GEMM."""
    r = build_program(_phi(INT8, target=INT8_LINEARS), "step")
    layer = ["Int8RowQuant", "ScaledMmS8", "RoPE", "RoPE", "Attention", "Int8RowQuant", "ScaledMmS8", "RMSNormFusedCuda",
             "Int8RowQuant", "ScaledMmS8", "SiluMul", "Int8RowQuant", "ScaledMmS8", "RMSNormFusedCuda"]
    nodes = r.program.fn.body.nodes
    assert [n.fn.name for n in nodes] == ["Embedding", "RMSNormTriton", *layer, *layer, "Gemm", "TokenSelect"]
    assert [n.fn.id for n in nodes[2:4]] == ["Int8RowQuant_v1{K=64}", "ScaledMmS8_v1{K=64,N=128,DOT=BlackwellS8ImmaDot32_v1}"]
    leaves = weight_leaves(model_shape(INT8))
    assert [(n.split("layers.0.")[-1], t.leaves, sorted(t.width_set())) for n, t in leaves[1:5]] == [
        ("self_attn.qkv_proj.weight", 128 * 64, [8]), ("self_attn.qkv_proj.weight_scale", 128, [32]),
        ("self_attn.o_proj.weight", 64 * 64, [8]), ("self_attn.o_proj.weight_scale", 64, [32])]
    assert len(leaves) == len(weight_leaves(model_shape(TINY))) + 4 * 2


def test_a_batch_invariant_request_reads_its_launchs_longest_query():
    """FA3's tile reads the launch's max_seqlen_q (`pipeline.launch_context`), as the Build's attention binding does: at head dim 64,
    an 8-row prompt prefilled beside a 40-row one runs the 40 rows' tile (BN 128), alone the one-warpgroup tile (8 x 2 query heads per
    KV head <= 64: BN 192), and both decode at 192.  The H200 int8 B4 row's LP11_T8 request (traced bec8910e) is this case at Llama-3.2-1B's shapes.
    A shape whose requests run in different launches has no one FA3 tile."""
    def tiles(requests: list[dict]) -> dict[int, int]:
        r = build_program(_phi({**TINY, "hidden_size": 256}, requests, target=dict(H200)), "request", LP=8, T=2)
        return dict(sorted((int(m[1]), int(m[2])) for k in r.desc["definitions"] if (m := re.match(r"AttentionHead_v4\{T=(\d+),D=64,BN=(\d+)", k))))

    one, other = {"prompt_len": 8, "max_tokens": 3, "arrive_step": 0}, {"prompt_len": 40, "max_tokens": 2, "arrive_step": 0}
    assert tiles([one]) == {t: 192 for t in range(1, 11)}
    assert tiles([one, other]) == {**{t: 128 for t in range(1, 9)}, 9: 192, 10: 192}
    with pytest.raises(Unexpressible, match=re.escape("FA3's tile (kBlockN) is not one value")):
        tiles([one, other, {**one, "arrive_step": 6}])


def test_an_int8_rows_dtype_is_its_checkpoints():
    """`score.phi_of` names an int8 row's checkpoint dtype "int8" (`config.DTYPES`): the decoder predicts it as the exact-int8
    checkpoint, and refuses a dtype its checkpoint isn't or one it has no linears for."""
    def as_row(phi: Phi, dtype: str) -> Phi:
        return Phi.from_json({**phi.to_json(), "model": {**phi.model, "dtype": dtype}})

    int8, bf16 = _phi(INT8, target=INT8_LINEARS), _phi(TINY)
    assert build_program(as_row(int8, "int8"), "step").digest == build_program(int8, "step").digest
    with pytest.raises(Unexpressible, match=re.escape("checkpoint dtype 'bfloat16' for an exact-int8 checkpoint")):
        build_program(as_row(int8, "bfloat16"), "step")
    with pytest.raises(Unexpressible, match=re.escape("checkpoint dtype 'int8' for a bf16 checkpoint")):
        build_program(as_row(bf16, "int8"), "step")
    with pytest.raises(Unexpressible, match=re.escape("checkpoint dtype 'float8_e4m3fn' (bfloat16 or int8 only)")):
        build_program(as_row(bf16, "float8_e4m3fn"), "step")


#: The H200 int8 Llama-3.2-1B rows' Builds: each Program digest of the row (the step at its M and pos0 8,
#: each request shape (LP, T), GP-01's workload).  B4's lone `build_request` (LP256_T31, no request of its batch) is B1's.
H200_INT8: dict[str, dict] = {
    "llama32-1b__int8__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager": {
        "step": (1, "6a146f062a03ab20598ddcd23cc08781975eec979a7e2a1f94c5d8a42a4c5f4d9bd66fb6a579dcb19fca08c8a15a4311c509ae54d0bf18bd4d8b54a689744293"),
        "requests": {(256, 31): "4bf2e0c502e48a505e377e9cce9d4cc94a0644d338eca95e0a03549c540f563e74679bd1cb5c932a306c4a99587574a8a1209d1c7c644dff6527f5d32bd83a61"},
        "workload": "6260a85900bcc21ee63e60bdc90b74cfaf3e386951f87a57c3ce3071a21c091d85b4ce43fd7ed3e437459f7adbfd8f84b3eb40ab721eec22f4912a53ec4504ac",
    },
    "llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager": {
        "step": (4, "b1e6df98ccebeae046ceed767d11b39c87d719cb584927a7875af60ffb7517f1366bfcae0ef2942e38f19520e624becd0f844d63e183dfcee3111a31a594faed"),
        "requests": {(11, 8): "f327ab463b59afbcd377ab24615c513df7b0f8c2f82173723cca6b21b53c6ce23613411c8b3b09b57778f8f9c89d625eeb8d380fd5bb84711aa88a5a74a25154",
                     (123, 19): "2ef3503d1adc7f0c6f9214357c3dffd9e3bcf67cd98679c0e7c90bff9050634e67c05d725aa3b81ab0eec468a1e4fcd19964ec80cfec1635028b6afa30f4c31c",
                     (159, 31): "73e53f40588b074ad7c9073930e3cc6a083e0ac318ba07608e92c6fc6c6facf9a5220e21423a6424e5b6b2f0205349069693e9245c25136c8de97e7674179d0d",
                     (256, 1): "afd45eebb3323e47b6d1a66666ae0a64d03d0927c890d53a5fe8c99cb120ed96fcc895bad83b101713be332014f0e8a4071a7c560e0da1bbffa844a677607b65"},
        "workload": "1e4ee6030e9a4091c4c669c2b0032808236a310469f2aec6706eede2f8253ed90471ba34a5bdc223af2ae5bd900e69130b8d23145857fc3f5c7bcdd7fd722a5e",
    },
}


@pytest.mark.slow
@pytest.mark.parametrize("row", sorted(H200_INT8), ids=["b1", "b4"])
def test_the_h200_int8_rows_predict_their_builds_of_record(row):
    """Llama-3.2-1B int8 on the H200 (`verity-int8/Llama-3.2-1B-INT8@15b0976d`, its config the checkpoint pin's, `pinned_hf_config`):
    φ from the row's declared inputs predicts the step, every request and GP-01's workload Program its traced Build holds, and a
    digest that moved is named.  `test_a_batch_invariant_request_reads_its_launchs_longest_query` holds one launch's context (the FA3
    tile at TINY's shapes); this holds the whole step and request Programs at the served shapes (each int8 linear's quantizer and s8
    product, FA3's tile in each launch of the batch, the bf16 LM head, the sampler) and the batch's composition, so a change to predict,
    the int8 linears or the workload fails here by name.  About 2 min."""
    pin = H200_INT8[row]
    w = json.loads(workload_file(WORKLOADS / f"{row}.json").read_text())
    phi = S.phi_of({"row": row, "workload": w}, config=str(pinned_hf_config(w["model"], RowEngine.of(w).revision)))
    assert sorted(phi.request_shapes()) == sorted(pin["requests"]), "the workload's request shapes are not its Build's"
    M, step = pin["step"]
    want = {"step": step, **{f"request LP{lp}_T{t}": d for (lp, t), d in pin["requests"].items()}, "workload": pin["workload"]}
    got, comps = {"step": build_program(phi, "step", M=M, pos0=8).digest}, {}
    for lp, t in pin["requests"]:
        r = build_program(phi, "request", LP=lp, T=t)
        got[f"request LP{lp}_T{t}"], comps[(lp, t)] = r.digest, S.request_component(r)
    got["workload"] = S.predict_workload(phi, comps)["digest"]
    moved = [k for k in want if got[k] != want[k]]
    assert not moved, "\n".join(
        [f"{row}: predict moved from the row's traced Build (re-trace the row on the H200 and pin the new digests with the predictor's)"]
        + [f"  {k}: Build {want[k][:16]}…, predicted {got[k][:16]}…" for k in moved])


@pytest.mark.parametrize("cfg,phi_kw,fact", [
    ({**TINY, "model_type": "gpt2"}, {}, "model_type 'gpt2': not in the family facts table"),
    ({**TINY, "attention_bias": True}, {}, "attention_bias"),
    ({**TINY, "mlp_bias": True}, {}, "MLP biases"),
    ({**TINY, "sliding_window": 32}, {}, "sliding window 32"),
    ({**TINY, "model_type": "phi3", "sliding_window": 8}, {}, "a history of 9 keys under the sliding window of 7 keys"),
    ({**TINY, "hidden_act": "gelu"}, {}, "hidden_act 'gelu'"),
    ({**json.loads((FIX / "gemma2-2b.config.json").read_text()), "query_pre_attn_scalar": 144}, {}, "query_pre_attn_scalar 144"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "num_experts": 32}, {}, "32 experts, top-8"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "shared_expert_intermediate_size": 512}, {}, "shared experts"),
    ({**json.loads((FIX / "qwen3-30b-a3b.config.json").read_text()), "mlp_only_layers": [0]}, {}, "dense MLP layers among the MoE ones"),
    (TINY, {"sampling": {**STOCH, "top_k": 50}}, "top_k 50"),
    (TINY, {"sampling": {**STOCH, "splits": None}}, "top-p split count None is not the registered one, registered_splits(188) = 32"),
    (TINY, {"sampling": {**STOCH, "splits": "derived"}}, "top-p split count 'derived' is not the registered one"),
    (TINY, {"sampling": {"kind": "mixed"}}, "sampling 'mixed'"),
    (TINY, {"tp": {"world": 4, "rank": 0}}, "tensor parallel world 4"),
    ({**TINY, "hidden_size": 96, "num_attention_heads": 6, "num_key_value_heads": 3}, {"tp": {"world": 2, "rank": 0}},
     "6 query and 3 key/value heads over 2 ranks"),
    ({**TINY, "vocab_size": 250}, {"tp": {"world": 2, "rank": 1}}, "a vocabulary of 250 over 2 ranks"),
    (TINY, {"target": {**RTXPRO6000, "batch_invariant": False}}, "a target labelled 'eager' (batch invariant False) for the 'bi-eager' row"),
    (TINY, {"engine": {"env": {"VLLM_BATCH_INVARIANT": "0"}}}, "env VLLM_BATCH_INVARIANT=0 under a target that sets '1'"),
    (TINY, {"engine": {"env": {"VLLM_USE_DEEP_GEMM": "0"}}}, "env VLLM_USE_DEEP_GEMM=0 (the switches modelled"),
    (TINY, {"engine": {"enforce_eager": False}}, "engine arg enforce_eager=False where the 'bi-eager' target applies True"),
    (TINY, {"engine": {"kv_cache_dtype": "fp8"}}, "engine arg kv_cache_dtype='fp8' (not modelled)"),
    (INT8, {}, "an exact-int8 checkpoint on a target whose int8_exact is None"),
    (TINY, {"target": INT8_LINEARS}, "a bf16 checkpoint on a target whose int8_exact is 'linears'"),
    (INT8, {"target": {**RTXPRO6000, "int8_exact": "all"}}, "the exact-int8 attention (int8_exact 'all'"),
    ({**TINY, "tie_word_embeddings": False, "quantization_config": dict(INT8_EXACT_LMHEAD_QUANT)}, {"target": INT8_LINEARS},
     "an int8 LM head (verity-int8-channel127/v2"),
    (INT8, {"target": INT8_LINEARS, "tp": {"world": 2, "rank": 0}}, "the int8 linears at TP2"),
    ({**INT8, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "use_sliding_window": False, "sliding_window": 32},
     {"target": INT8_LINEARS}, "an int8 linear with a bias"),
])
def test_unexpressible_names_the_fact(cfg, phi_kw, fact):
    with pytest.raises(Unexpressible, match=re.escape(fact)):
        build_program(_phi(cfg, **phi_kw), "step")


@pytest.mark.parametrize("empty", ["absent", None, {}, 0, "", False, []])
def test_a_family_that_states_no_rules_facts_or_constructor_is_read_as_one_with_none(monkeypatch, empty):
    """A family's entry of the family facts table may leave `rules`, `facts` or `arch` out, or state it empty.  The predictor then
    reads no rule (qwen2 without its rule has no q/k/v bias), no fact the vocabulary lacks, and no constructor, which
    `model_shape` refuses by name."""
    from verity_vllm.engine.profiles.family_facts import FAMILY_FACTS
    from verity_vllm.program.predict import model_shape as MS
    cfg = {**TINY, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "use_sliding_window": False, "sliding_window": 32}
    stated = FAMILY_FACTS["qwen2"]

    def entry(block: str) -> dict:
        rest = {k: v for k, v in stated.items() if k != block}
        return rest if empty == "absent" else {**rest, block: empty}

    as_stated = model_shape(cfg)
    assert as_stated.qkv_bias
    monkeypatch.setitem(FAMILY_FACTS, "qwen2", entry("rules"))
    assert not model_shape(cfg).qkv_bias
    monkeypatch.setitem(FAMILY_FACTS, "qwen2", entry("facts"))
    with monkeypatch.context() as m:
        m.setattr(MS, "resolve_facts", lambda hf, C: ({}, {}))    # the table's own reader of the block is not the predictor's
        assert model_shape(cfg) == as_stated
    monkeypatch.setitem(FAMILY_FACTS, "qwen2", entry("arch"))
    named = None if empty == "absent" else empty
    with pytest.raises(Unexpressible, match=re.escape(f"family 'qwen2' constructor {named!r} (the dense and MoE blocks only)")):
        build_program(_phi(cfg), "step")


def test_no_module_outside_the_predictor_imports_or_runs_it():
    """No module outside `verity_vllm.program.predict` imports it or runs `predict`, and the trace that checks a Build lists the
    package as a forbidden input."""
    from verity_vllm.program.frontend import inputs_trace
    here, name = Path(S.__file__).resolve().parent, S.__name__.rpartition(".")[0]
    pkg = here.parents[name.count(".") - 1]
    inside = here.relative_to(pkg).parts
    pat = re.compile(rf"^\s*(from|import)\s+({re.escape(name)}|\.+{here.name})\b|import_module\([^)]*{here.name}", re.M)
    outside = [p for p in pkg.rglob("*.py") if p.relative_to(pkg).parts[:len(inside)] != inside]
    hits = [str(p.relative_to(pkg)) for p in outside if pat.search(p.read_text())]
    assert hits == []
    runs = [str(p.relative_to(pkg)) for p in outside if re.search(r"""cli\(\s*["']predict["']""", p.read_text())]
    assert runs == []
    assert any(m in str(Path(S.__file__).resolve()) for m in inputs_trace._FORBIDDEN_MARKERS)


def _portions(monkeypatch, tmp_path, files: tuple[str, ...]):
    """`verity` as a namespace over portions in a git checkout at tmp_path/repo holding `files`, tracked; returns a function of
    the portions' names giving the code version over them, in that order.  Git looks for no checkout above tmp_path."""
    import subprocess

    import verity

    repo = tmp_path / "repo"
    for rel in files:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(f"# {rel}\n")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)

    def version(*portions: str) -> str:
        monkeypatch.setattr(verity, "__path__", [str(tmp_path / p / "verity") for p in portions])
        return S.code_version()
    return version


def test_code_version_is_one_tree_over_a_namespaces_portions(monkeypatch, tmp_path):
    """`verity` as a namespace: the digest is the same over any order of its portions and with a cwd directory that joined it
    without a module, and moves with an edit."""
    version = _portions(monkeypatch, tmp_path, ("upper/verity/circuits/__init__.py", "lower/verity/ir/__init__.py", "lower/verity/ir/defs.py",
                                                "cwd/verity/circuits/src/verity/ir/defs.py"))
    a = version("repo/upper", "repo/lower")
    assert version("repo/lower", "repo/upper") == version("repo/cwd", "repo/upper", "repo/lower") == a
    (tmp_path / "repo" / "lower" / "verity" / "ir" / "defs.py").write_text("# edited\n")
    assert version("repo/upper", "repo/lower") != a


def test_code_version_reads_tracked_files_only(monkeypatch, tmp_path):
    """A file written into a package's tree that git doesn't track (what a parallel test, a build or a run leaves there) doesn't
    move the digest, so two calls agree whatever was written between them; tracking it does.  A portion outside any checkout is
    refused rather than hashed as it lies on disk."""
    import subprocess

    version = _portions(monkeypatch, tmp_path, ("pkg/verity/ir/__init__.py", "pkg/verity/ir/defs.py"))
    a = version("repo/pkg")
    for stray in ("ir/defs.py.tmp", "ir/__pycache__/defs.cpython-312.pyc", "ir/.lake/build/out.olean", "report.json"):
        (tmp_path / "repo" / "pkg" / "verity" / stray).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "pkg" / "verity" / stray).write_text("stray\n")
    assert version("repo/pkg") == a
    subprocess.run(["git", "-C", str(tmp_path / "repo"), "add", "-f", "pkg/verity/report.json"], check=True)
    assert version("repo/pkg") != a
    (tmp_path / "loose" / "verity" / "ir").mkdir(parents=True)
    (tmp_path / "loose" / "verity" / "ir" / "__init__.py").write_text("")
    with pytest.raises(RuntimeError, match="not in a git checkout"):
        version("loose")
