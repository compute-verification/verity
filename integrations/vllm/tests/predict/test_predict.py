"""verity_vllm.predict: predicted Programs equal the traced Builds of record, and the scorer says where they do not.

The pinned digests are Programs the Build traced on rtxpro6000 (cc 12.0, 188 SMs) from the checkpoints' `config.json`
(`fixtures/`): SmolLM2-135M from cov-k01-10, Llama-3.2-1B from cov-k05-7, TinyLlama-1.1B from cov-k04-7, Qwen2.5-0.5B from
cov-k03-8 (its biased q/k/v linear one `GemmBias_v2{K,N,DOT}` Call), Phi-3-mini from cov-k07-3, Qwen3-4B from cov-k09-3, Gemma-2-2B from cov-k06-5, Llama-3.2-1B's TP2 ranks from cov-p000,
Gemma-2-2B's from cov-p058-2, Qwen3-30B-A3B's from cov-p081 and OLMoE's from cov-p069.
A library change that moves a Definition moves the traced digest too: re-trace the unit (`verity-vllm build --wrapper
step --M 1 --pos0 8`, CPU only) and pin the new digest with the predictor's.
`SiluMul_v2{I}` as the `silu_mul` kind moved every pin but Gemma-2's (GeGLU): each is the Build's re-trace of its unit from the
same checkpoints on the declared rtxpro6000 target (run r20261002-123618-cee3, CPU only), and with `SiluMul_v1{I}` in its place
the predictor gives the rows' traced digests.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from verity_vllm.config import MANIFESTS, WORKLOADS, RowEngine, UsageError
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR
from verity_vllm.predict import score as S
from verity_vllm.predict.build import Unexpressible, build_program, weight_leaves, model_shape
from verity_vllm.predict.phi import Phi

FIX = Path(__file__).parent / "fixtures"
GLM53 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
GLM53_B8 = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager"
DSV4 = "dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}


def _phi(cfg: dict, requests: list[dict] | None = None, **kw) -> Phi:
    return Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, workload={"requests": requests or []}, **{"target": dict(RTXPRO6000), **kw})


def _row_phi(row: str, case: str, hf_home: Path, rank: int = 0) -> Phi:
    """A row of record's φ at `rank` as `score.phi_of` makes it, the checkpoint's config.json the package's byte copy of it
    (`hf_configs/<case>.config.json`)."""
    w = json.loads((WORKLOADS / f"{row}.json").read_text())
    cp = Path(S.hf_config_path(str(hf_home), w["model"], RowEngine.of(w).revision))
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes((HF_CONFIG_DIR / f"{case}.config.json").read_bytes())
    phi = S.phi_of({"row": row, "workload": w}, hf_home=str(hf_home))
    return Phi.from_json({**phi.to_json(), "tp": {**phi.tp, "rank": rank}})


@pytest.mark.parametrize("model,digest", [
    ("smollm2-135m", "4fd836ffc4f1a9894cfc099add65e2f28ea260bf80714580b79489ef17d10754e68ddd9b9becfd444c97ba724f95e5d142dc464dae13dad67bef43850f2070b9"),
    ("llama32-1b", "773b545605765fe927f70717bc20ea19076aeb3ad3e4f83bf398cdc8fd7421637be2f3e7b7c653292b2031192eb59bdff86965fd460aefcda517702befff323a"),
    ("tinyllama-11b", "75eeda370d481672af5f189578e1b6711f3d05104d1e0fca99d27c9e9d0310be9a87ba03c65908b04ac4d313d0476dc09c32b8387d66084fb4cbdd777c3193ea"),
    ("qwen25-05b", "452681af0e5f7cdd4003098f333d0c588d0128d98bccdef377d89fa8ef42524823045490ce7e6d3b7f435a6362c7603c7637eae57c1709299e27084826874b94"),
    ("phi3-mini", "ecd84c104c7ae0368a596d0f47e83f9eda564ef88c5ca692c8226a24a7783eaf5c264f8f1e21fb638db052c99d24ab8ceb8f9c4e762dc0f668f6314822eaa394"),
    ("qwen3-4b", "21acad0cea8011bce8e3a83844f57645a04d6fa9ef9c6c1ff30640e868c340368705a11e7e405c6fca870a583a9b1aba74367e316f234a017a7cb3cc6a799fb8"),
    ("olmoe-1b-7b", "60982699f17e47e048ea3ac7d3dc6c28bec57a74f0c6002f4e4d89a4e776b047e010ba772125d21bf9afa39cefe9e8c4152d4952e87056169e458de826c517e7"),
    ("qwen3-30b-a3b", "98a180a562abf1d603951ed135a1e26a084cc48cb53cdbfbb7b0e26ca2139244cfc036bdb8ab8af0e9a9a4786ef8372ce478a365253791ba35479a5b2bf0717c"),
    ("gemma2-2b", "b0688044e6a562d2cbb7a6fa6995deb9025d08d134b331236dc4c16ffedf8e38cb093a5e521052cb8c19d5d1641b0c8bb725eddb50e1a0f480149354bbab15c7"),
])
def test_step_program_equals_the_traced_build(model, digest):
    r = build_program(_phi(json.loads((FIX / f"{model}.config.json").read_text())), "step", M=1, pos0=8)
    cls = {"qwen25-05b": "Qwen2ForCausalLM", "phi3-mini": "Phi3ForCausalLM", "qwen3-4b": "Qwen3ForCausalLM", "olmoe-1b-7b": "OlmoeForCausalLM",
           "qwen3-30b-a3b": "Qwen3MoeForCausalLM", "gemma2-2b": "Gemma2ForCausalLM"}.get(model, "LlamaForCausalLM")
    assert r.function == f"Derived[vLLMStep[{cls}]]_v1{{SRC={{d9c6ce11eb35}}}}"
    assert r.digest == digest


@pytest.mark.slow
def test_request_and_workload_programs_equal_the_traced_build():
    """SmolLM2-135M B1 256/32 greedy (cov-k01-10): the request Program (LP 256, T 31) and GP-01's workload Program over it.
    About 20 s: the only test that composes a workload, which the step pins cannot reach."""
    phi = _phi(json.loads((FIX / "smollm2-135m.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}])
    r = build_program(phi, "request", LP=256, T=31)
    assert r.digest == "52e6bb0b37e31ec3bdcc976027f412955be60cdd7bcd1ed4982aee7b4b716681a00323889897a394808aa040994f53a3e210dc603dba9a65751817e1df348f5c"
    assert r.n_root_nodes == 86738
    assert S.predict_workload(phi)["digest"] == "345798f4dc27339a239f17acd3232b758b4b242092d87e9c558ea9d02ec202f08b14212f8c68a7e6dbc5413e4f235a7085fcd15a66b021cd51d8e3755e7e2f06"


@pytest.mark.parametrize("model,function,rank,digest", [
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 0, "71d9310ecaef13138ee37e31fe4f5be475d232d952058fec72169232cae90136800b33b24c632e9dab2d0fbf6957288f88927c3c651bc7c2ba839514f9c484cc"),
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 1, "3343270c8e008ad0106a174bad32c6a23c428c7d821e8a2aa251e8edbdcf0b9d77972761c98e5203ad50c0977dd11546ff10c20f566920a72b833ad61efbff90"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 0, "15bf71465c58172b07dbd848f434f886274de57e88d30fe8d41075b08cc45c8cfedc7a5164343af904af3cc7f47227cdc51725704a1f28e6e82c3bdef4425b8e"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 1, "8c4abf0f43e41613ab749374f38aaade322cf7a2cc8e692c68de5b7c28840aa1c77933a18c89db9a55f741895621d0d7679e1e6d7201b485211ecec09511e6b8"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 0, "0605dc64bd0076b31c8a1363c4d3443a07a09fc7dabcd2734bf521d544f96c3dc1a24db315b928ee8b42fd690230e85c2f07345156da73610a486300f72f366d"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 1, "c2a420cc21022518f94f0e6cf7e06a6066049cd1a48d3bc6506324c51722d6ec01ae9b10bc7d658b518ab01ceb949160f4ce86eed72ae134748a0f40f3e5cd96"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 0, "25068910fdb6ab3bde8890d77c58d85d05b2833098ef4b1701b6c84318683ea79722e26b31e41255f03e29ddfe806241fce1aa93e0a299ec2415ee029e90b4e6"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 1, "5c9905db914095dd2e587b3c48dc004d6037a274b6d3f274381e765e44afa8bc9ab165e8f35ebcb417ffb9def35d3425ff752c94bc0c1a464d3f3951f81a4cf4"),
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
    assert S.predict_workload(phi)["digest"] == "4cd839563c959eee6f1abaf647bd7d771a842cb8940ed5a92443e85e2ca48cd7f5aecfa1f80d8084c276d3c32a59b1dbf36770a18cb8015ec6ebc8c62cb9c1da"


@pytest.mark.parametrize("rank", range(8))
def test_dsa_rank_step_equals_the_record(rank, tmp_path):
    """GLM-5.3 at TP8 on H200 (block FP8, cc 9.0): each rank's step through the DeepSeek sparse-attention decoder (`predict.dsa`)
    is the rank's step Program `records.json` pins for the row (its FlashInfer fused all-reduce + RMSNorm, the decode-path sparse
    MLA, the indexer on its layers, the tuned fused-MoE tile at M = 1)."""
    r = build_program(_row_phi(GLM53, "GLM53", tmp_path, rank), "step", M=1, pos0=8)
    assert r.function.startswith("Derived[vLLMStep[DeepseekV32ForCausalLM]]_v1")
    assert r.digest == json.loads((MANIFESTS / "records.json").read_text())["rows"][GLM53]["programs"][str(rank)]["step"]


@pytest.mark.parametrize("wrapper,shape,digest", [
    ("step", {"M": 8, "pos0": 8}, "6a8ea56b8f7c3f2a4c557d20dd45b86b9eda749f653f4559ee21327ca3bed830254af606c6aee98b123ccc4cb6172fb2283a96be63867331687fa0f5100e4bd2"),
    ("request", {"LP": 11, "T": 12}, "f22d03019ee5c9324134c394b8628cf3bda5fa89d3e9e04b3fe56faf92e856bc5e05eb057ed2adb327e513dc39fa5c93597bfec0011deab079629e3c974c5bcd"),
])
def test_dsa_launch_batch_equals_the_traced_build(wrapper, shape, digest, tmp_path):
    """GLM-5.3's B8 mixed-arrivals row (no batch invariance) at rank 0, as the Build traced it in r20261017-194958-148e: the step of 8
    rows, and the first request's Program under its launch batch (its prefill alone, then decodes beside the second request's
    prefill and decodes), every token-count-dependent selection at the launch's tokens."""
    r = build_program(_row_phi(GLM53_B8, "GLM53", tmp_path), wrapper, **shape)
    assert r.digest == digest


def test_dsa_refuses_outside_its_domain(tmp_path):
    phi = _row_phi(GLM53, "GLM53", tmp_path)
    with pytest.raises(Unexpressible, match=re.escape("tensor parallel world 2 (TP8, the worlds a Build has traced)")):
        build_program(Phi.from_json({**phi.to_json(), "tp": {"world": 2, "rank": 0}}), "step")
    with pytest.raises(Unexpressible, match=re.escape("a target without a device name")):
        build_program(Phi.from_json({**phi.to_json(), "target": {k: v for k, v in phi.target.items() if k != "device_name"}}), "step")
    with pytest.raises(Unexpressible, match=re.escape("a call of 257 query rows, a prefill (decode threshold 256 at 8 query heads")):
        build_program(phi, "step", M=257)


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


@pytest.mark.slow
@pytest.mark.parametrize("rank,digest", [
    (0, "84c39fac2fc2018712751846d53b12b077c6ba8f22b53e95ca24f4c9f541d4724264893b6a8f8b29aa917a46305c740a330f6a9779fa25938c542cc9d2027beb"),
    (7, "f7b56f581431ed6ff565dd7be13f7cf24d3d12156fbb1b7011188d194be552f82bc3c536cb694b4103e1b9f41a831946d460fe89a00163152e1bfd6cf2765dff"),
])
def test_mhc_rank_step_equals_the_traced_build(rank, digest, tmp_path):
    """DeepSeek-V4-Pro at TP8 on H200 (block FP8, MXFP4 experts, cc 9.0): the first and last rank's step through the hyper-connected
    decoder (`predict.dsv4`) is the Program the Build traced at 98c10539e (`records.json` has no row for it yet): the mHC pre and
    post-pre between sublayers, the compressed-attention decode over each layer's opened window, the table and sqrtsoftplus routers,
    the Marlin experts over the schedule's align keys.  About 25 s a rank, nearly all of it encoding the mHC Definitions' bodies for
    the digest; no cheaper test reaches this decoder, and the ranks between differ only in the peer order GLM-5.3's ranks pin."""
    r = build_program(_row_phi(DSV4, "DSV4_PRO", tmp_path, rank), "step", M=1, pos0=8)
    assert r.function.startswith("Derived[vLLMStep[DeepseekV4ForCausalLM]]_v1")
    assert r.digest == digest


def test_mhc_refuses_outside_its_domain(tmp_path):
    phi = _row_phi(DSV4, "DSV4_PRO", tmp_path)
    short = {**phi.hf_config, "compress_ratios": [4] + phi.hf_config["compress_ratios"][1:], "index_topk": 1}
    with pytest.raises(Unexpressible, match=re.escape("position 7 has 2 compressed rows, more than the indexer's top-1")):
        build_program(Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": short}}), "step", M=8, pos0=0)
    cfg = {**phi.hf_config, "compress_ratios": [1] + phi.hf_config["compress_ratios"][1:]}
    with pytest.raises(Unexpressible, match=re.escape("compress ratios [1] (the compressors: [4, 128]")):
        build_program(Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": cfg}}), "step")


@pytest.mark.parametrize("sampling,M,digest", [
    ({**STOCH}, 8, "740d7f22434aa56824d598e4b66f8223c45546ca6e1428639951e5299b7f1968f209b61903452be424a6f407d6d2197658d39073b97bcdf7539c0a923c38ccda"),
    ({**STOCH, "top_p": 1.0, "splits": 32}, 1, "6529c4fbcbfeecf09393c9e137e3272da8ef54bad1b116c8c0f3f1bb078ef6747e6bc99265b1f3bcfd39d4b5123914a04881ad345361c890bfb387de20a14f49"),
])
def test_stochastic_step_program_equals_the_traced_build(sampling, M, digest):
    """SmolLM2-135M under the seeded top-p policy: a B8 row's step (the split count a `splits` Input, cov-g230) and a B1 row's (the
    single request's constant S = 32 on 188 SMs, cov-g253-2)."""
    r = build_program(_phi(json.loads((FIX / "smollm2-135m.config.json").read_text()), sampling=sampling), "step", M=M, pos0=8)
    assert r.digest == digest


def test_sampling_follows_the_builds_policy():
    one = {"sweep": {"sampling": "stoch-t0.8-p0.95", "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 0}, "sampling_seeds": [7]},
           "requests": [{"prompt_len": 8, "max_tokens": 3, "seed": 7}]}
    assert S.sampling_of(one, _phi(TINY)) == {**STOCH, "splits": 32}
    assert S.sampling_of({**one, "requests": one["requests"] * 8}, _phi(TINY)) == STOCH
    assert S.sampling_of({"requests": [{"prompt_len": 8, "max_tokens": 3}]}, _phi(TINY)) == {"kind": "greedy"}
    with pytest.raises(Unexpressible, match="the Build refuses the policy"):
        S.sampling_of({**one, "sweep": {**one["sweep"], "sampling_params": {"temperature": 0.8, "top_p": 0.95, "top_k": 50}}}, _phi(TINY))


def test_weights_record_follows_the_module_order():
    C = model_shape(TINY)
    names = [n for n, _ in weight_leaves(C)]
    assert names[:3] == ["model.model.embed_tokens.weight", "model.model.layers.0.self_attn.qkv_proj.weight",
                         "model.model.layers.0.self_attn.o_proj.weight"]
    assert "model.lm_head.weight" not in names  # tied: the embedding's leaf
    assert len(names) == 1 + 6 * 2 + 1 + 1 + 4 * 2
    shards = [(n.split("layers.0.")[-1], t) for n, t in weight_leaves(C, 2)[:7]]
    assert [(n, t.leaves) for n, t in shards] == [
        ("model.model.embed_tokens.weight", 128 * 64), ("self_attn.qkv_proj.weight", (2 + 2 * 1) * 16 * 64), ("self_attn.o_proj.weight", 64 * 32),
        ("mlp.gate_up_proj.weight", 128 * 64), ("mlp.down_proj.weight", 64 * 64), ("input_layernorm.weight", 64), ("post_attention_layernorm.weight", 64)]
    untied = [n for n, _ in weight_leaves(model_shape({**TINY, "tie_word_embeddings": False}))]
    assert untied.index("model.lm_head.weight") == untied.index("model.model.norm.weight") + 1
    biased = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen2", "use_sliding_window": False, "sliding_window": 32}))]
    assert biased[1:4] == ["model.model.layers.0.self_attn.qkv_proj.weight", "model.model.layers.0.self_attn.qkv_proj.bias",
                           "model.model.layers.0.self_attn.o_proj.weight"]
    normed = [n for n, _ in weight_leaves(model_shape({**TINY, "model_type": "qwen3"}))]
    assert normed[2:5] == ["model.model.layers.0.self_attn.o_proj.weight", "model.model.layers.0.self_attn.q_norm.weight",
                           "model.model.layers.0.self_attn.k_norm.weight"]
    moe = weight_leaves(model_shape(json.loads((FIX / "olmoe-1b-7b.config.json").read_text())))
    assert [n.split("layers.0.")[-1] for n, _ in moe[1:10]] == [
        "self_attn.qkv_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.o_proj.weight", "mlp.gate.weight",
        "mlp.experts.w13_weight", "mlp.experts.w2_weight", "input_layernorm.weight", "post_attention_layernorm.weight"]
    gemma = [n for n, _ in weight_leaves(model_shape(json.loads((FIX / "gemma2-2b.config.json").read_text())))]
    assert [n.split("layers.0.")[-1] for n in gemma[1:9]] == [
        "self_attn.qkv_proj.weight", "self_attn.o_proj.weight", "mlp.gate_up_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight",
        "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"]
    assert gemma.index("model.model.normalizer") == gemma.index("model.model.norm.weight") + 1 == \
        gemma.index("model.model.layers.0.self_attn.rotary_emb.cos_sin_cache") - 1


@pytest.mark.parametrize("target,kind", [({"compute_capability": [8, 0], "num_sms": 108}, "GemmBias_v1"), (RTXPRO6000, "GemmBias_v2")])
def test_the_biased_linear_follows_the_targets_rule(target, kind):
    """Qwen2's q/k/v bias: one `GemmBias_v1{K,N}` Call per row under the accepted kinds (A100, FA2), and where the target's record
    names a biased-linear step (`gemm_bias_dot`: rtxpro6000) one `GemmBias_v2{K,N,DOT}` Call with the target's step; never the GEMM
    then `BiasAdd_v1`, which only a family whose record names no step would keep."""
    cfg = {**TINY, "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"], "use_sliding_window": False, "sliding_window": 32}
    r = build_program(Phi(model={"repo": "org/m", "revision": "r", "hf_config": cfg}, target=target, workload={"requests": []}), "step")
    names = {re.match(r"\w+", k).group() for k in r.desc["definitions"]}
    assert names & {"GemmBias_v1", "GemmBias_v2", "BiasAdd_v1"} == {kind}


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
    (TINY, {"sampling": {**STOCH, "splits": None}}, "without a split count"),
    (TINY, {"sampling": {"kind": "mixed"}}, "sampling 'mixed'"),
    (TINY, {"tp": {"world": 4, "rank": 0}}, "tensor parallel world 4"),
    ({**TINY, "hidden_size": 96, "num_attention_heads": 6, "num_key_value_heads": 3}, {"tp": {"world": 2, "rank": 0}},
     "6 query and 3 key/value heads over 2 ranks"),
    ({**TINY, "vocab_size": 250}, {"tp": {"world": 2, "rank": 1}}, "a vocabulary of 250 over 2 ranks"),
    (TINY, {"tp": {"world": 2, "rank": 0}, "sampling": dict(STOCH)}, "stochastic sampling under tensor parallelism"),
    (TINY, {"target": {**RTXPRO6000, "batch_invariant": False}}, "a target labelled 'eager' (batch invariant False) for the 'bi-eager' row"),
    (TINY, {"engine": {"env": {"VLLM_BATCH_INVARIANT": "0"}}}, "env VLLM_BATCH_INVARIANT=0 under a target that sets '1'"),
    (TINY, {"engine": {"env": {"VLLM_USE_DEEP_GEMM": "0"}}}, "env VLLM_USE_DEEP_GEMM=0 (the switches modelled"),
    (TINY, {"engine": {"enforce_eager": False}}, "engine arg enforce_eager=False where the 'bi-eager' target applies True"),
    (TINY, {"engine": {"kv_cache_dtype": "fp8"}}, "engine arg kv_cache_dtype='fp8' (not modelled)"),
])
def test_unexpressible_names_the_fact(cfg, phi_kw, fact):
    with pytest.raises(Unexpressible, match=re.escape(fact)):
        build_program(_phi(cfg, **phi_kw), "step")


def test_memory_estimate_is_above_a_measured_peak():
    """A Qwen3-4B request of LP+T = 79 peaked at 814 MB resident (`_child`'s max_rss_mb); the pool admits units by the estimate."""
    assert S.mem_estimate_mb(model_shape(json.loads((FIX / "qwen3-4b.config.json").read_text())), [79]) > 814


def test_score_never_writes_into_a_rows_root(tmp_path):
    rows = tmp_path / "rows"
    (rows / "job" / "row").mkdir(parents=True)
    with pytest.raises(UsageError, match="never written into a Build row"):
        S._refuse_inside(str(rows / "job" / "row" / "score.json"), [str(rows)])
    S._refuse_inside(str(tmp_path / "out" / "score.json"), [str(rows)])


def test_no_record_path_module_reads_the_predictor():
    """The Build, Match and Commit never import `verity_vllm.predict`; the CLI's command table names it as a string only, and the
    Build's input trace lists the package as a forbidden input."""
    from verity_vllm.program.frontend import inputs_trace
    pkg = Path(S.__file__).resolve().parents[1]
    pat = re.compile(r"^\s*(from|import)\s+(verity_vllm\.predict|\.+predict)\b|import_module\([^)]*predict", re.M)
    hits = [str(p.relative_to(pkg)) for p in pkg.rglob("*.py") if "predict" not in p.relative_to(pkg).parts[:1] and pat.search(p.read_text())]
    assert hits == []
    assert any(m in str(Path(S.__file__).resolve()) for m in inputs_trace._FORBIDDEN_MARKERS)


def test_code_version_is_one_tree_over_a_namespaces_portions(monkeypatch, tmp_path):
    """`verity` as a namespace: the digest is the same over any order of its portions and with a cwd directory that joined it
    without a module, and moves with an edit."""
    import verity

    for rel in ("upper/verity/ml/__init__.py", "lower/verity/ir/__init__.py", "lower/verity/ir/defs.py",
                "cwd/verity/primitives/circuits/src/verity/ir/defs.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"# {rel}\n")

    def version(*portions: str) -> str:
        monkeypatch.setattr(verity, "__path__", [str(tmp_path / p / "verity") for p in portions])
        return S.code_version()

    a = version("upper", "lower")
    assert version("lower", "upper") == version("cwd", "upper", "lower") == a
    (tmp_path / "lower" / "verity" / "ir" / "defs.py").write_text("# edited\n")
    assert version("upper", "lower") != a
