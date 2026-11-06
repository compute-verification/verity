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
import re
from pathlib import Path

import pytest

from verity_vllm.config import MANIFESTS, WORKLOADS, RowEngine, UsageError
from verity_vllm.engine.profiles.dense_generic import HF_CONFIG_DIR, pinned_hf_config
from verity_vllm.program.model import INT8_EXACT_LMHEAD_QUANT, INT8_EXACT_QUANT
from verity_vllm.program.predict import score as S
from verity_vllm.program.predict.build import Unexpressible, build_program, weight_leaves, model_shape
from verity_vllm.program.predict.phi import Phi

FIX = Path(__file__).parent / "fixtures"
GLM53 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
GLM53_B8 = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager"
GLM53_S1 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__stoch-t0.8-p0.95__eager"
GLM53_S8 = "glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager"
DSV4 = "dsv4-pro__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
RTXPRO6000 = {"compute_capability": [12, 0], "num_sms": 188}
H200 = {"compute_capability": [9, 0], "num_sms": 132}
TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
STOCH = {"kind": "stochastic", "temperature": 0.8, "top_p": 0.95, "top_k": 0, "seeded": True, "splits": "derived"}
#: TINY's exact-int8 checkpoint (`engine.int8_checkpoint`'s v1) on a target serving its linears int8 (serve-openai's verity_int8)
INT8 = {**TINY, "quantization_config": dict(INT8_EXACT_QUANT)}
INT8_LINEARS = {**RTXPRO6000, "int8_exact": "linears"}


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
    ("smollm2-135m", "d469f1d4eefafff23e468803ca7894d762d80f8eb81cc2eb10e689629207497d2b4967993360d9c1bb535b3faa3c95f5aa7b18bf4ffd5752935e8a62d18b7ee4"),
    ("llama32-1b", "5e34739ca3f59dda9febb8dc7c89ab8f4bf146945168079d83c420dc4dd49676034115612e78a3642471d3ebca1e1573e8618d795c31d7e6f118292ccb40e903"),
    ("tinyllama-11b", "18372d884c9c7551bca643d830ab55c7c8e53568c7ee2528819ba997a657e1f7fc97b82068a3de2a3d373cc8a328fa9108115e440f311c17209800e695769d83"),
    ("qwen25-05b", "bff68977892d274b710782a1b5db23e4c5485f716f1aa23b9872744d50cb43692f878fd0d20c83bec28f2b261ac833019cf0a1a80ae3f678cc54f92ffc6e73e9"),
    ("phi3-mini", "3c7404516954d1a8cfb4dcc00c2a8a10d8ecce091c4e7c04ff5ae3f8659e57faa630797df41e5659bda9ad24225cdc2f8f1a6a08633aa3790a04532c169cda43"),
    ("qwen3-4b", "424f31562fc5c47ffcf92f885185c46c1f81914a4d6ea2b7615f08c2ee8e94a9a81ede55594ee919fe90a1cc3e165838b144823baf13ea7bb692221a23a3761b"),
    ("olmoe-1b-7b", "3f690ae479bb20888cc3a3ea61c2ac25693701c75a0c5fae2de177b167fca7959ad66e7900b1c6e264601ea6984fd8fe7de08c832268c91b43be8e6910ac3ca6"),
    ("qwen3-30b-a3b", "d12a2b4c96ad1b35f6288561a4ae68f9262b645098a1b2f3d6594c29494c523a33bd7c1ce3f793b87b664399de816adb768019f1fc133301ecae71279e699289"),
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
    """SmolLM2-135M B1 256/32 greedy: the request Program (LP 256, T 31) and GP-01's workload Program over it.
    About 20 s: the only test that composes a workload, which the step pins cannot reach."""
    phi = _phi(json.loads((FIX / "smollm2-135m.config.json").read_text()),
               [{"request_id": "u0", "index": 0, "prompt_len": 256, "max_tokens": 32, "arrive_step": 0}])
    r = build_program(phi, "request", LP=256, T=31)
    assert r.digest == "91d5eaf172589c9a5586e4eb7b8a83e047448018a2a14bcd6e0048f0618bc0a31a390b53b33459979cb706927233124ba258c1043c76a6461ffc91f7d8411888"
    assert r.n_root_nodes == 84698
    assert S.predict_workload(phi)["digest"] == "24835950b9fd8c930fcf2077b8a767921c9a5ddc6e12cfe1ed20c4a2030c658a60ec601035e171e4f9197de081ad75b6b06da9ea7be4742a00aa6d0d62ba242f"


@pytest.mark.parametrize("model,function,rank,digest", [
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 0, "3e1e44f5367000d4696ccb03551c70801dc794a69030e28eeeb156c9533ebf774b4dc40f11a9057f28a83340d96a10467814c6cbc7160f1178e670822a0f2764"),
    ("llama32-1b", "LlamaForCausalLM]]_v1{SRC={c74ad5d713d6}}", 1, "a80fefc8ead7c8d09c2bf2b43ca6ed34296ec4db8d9891b71a850a90613cae6a77a4be0cde7b22a9a0f693c650684380712438a452e6eeb0287ddfdf037b4b6a"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 0, "15bf71465c58172b07dbd848f434f886274de57e88d30fe8d41075b08cc45c8cfedc7a5164343af904af3cc7f47227cdc51725704a1f28e6e82c3bdef4425b8e"),
    ("gemma2-2b", "Gemma2ForCausalLM]]_v1{SRC={72bb75022406}}", 1, "8c4abf0f43e41613ab749374f38aaade322cf7a2cc8e692c68de5b7c28840aa1c77933a18c89db9a55f741895621d0d7679e1e6d7201b485211ecec09511e6b8"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 0, "258cb9677a12f76c80ec70b607060d39092bd7501c7349607b8bb58ce274359e182b6f3b40bf7170bf175ea310eebead5f51b765706a975401cf72bed56185c4"),
    ("qwen3-30b-a3b", "Qwen3MoeForCausalLM]]_v1{SRC={bbe90689fb26}}", 1, "b882f289c19f58594f4c15051c32f54554bac3ad5a3ceaf6dac6a052e7dd81fbd709cf072d761ecdc3b353c955359a2f845850e871d727dae630d721da1a3abd"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 0, "5775d49eb16f725fc5ead7142a859c2502b9fd113ed8b8842277cd9353c93b3ccd02026c77f3a2625ff4456700f2e4f87654b6bd98095ed9c75c0704e0d87432"),
    ("olmoe-1b-7b", "OlmoeForCausalLM]]_v1{SRC={46db9231ff55}}", 1, "2a44e1dbe163187916505e089bd8370f67eee9ee1790880f91305c128e2a3275725585302d1935de3fa0f47e5fb15a5cd53fad803b80b18c258e9ea155842449"),
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
    assert S.predict_workload(phi)["digest"] == "93df4dd412d503b870425a351b2d46c916f7ce83e194b92ea2acab147a5aa10389bae4320dc5f1bedc56c7dad10e040cd80791cb90b0d8dfb9eb497d3682973c"


@pytest.mark.slow
@pytest.mark.parametrize("rank", range(8))
def test_dsa_rank_step_equals_the_record(rank, tmp_path):
    """GLM-5.3 at TP8 on H200 (block FP8, cc 9.0): each rank's step through the DeepSeek sparse-attention decoder (`predict.dsa`)
    is the rank's step Program `records.json` pins for the row (its FlashInfer fused all-reduce + RMSNorm, the decode-path sparse
    MLA, the indexer on its layers, the tuned fused-MoE tile at M = 1)."""
    r = build_program(_row_phi(GLM53, "GLM53", tmp_path, rank), "step", M=1, pos0=8)
    assert r.function.startswith("Derived[vLLMStep[DeepseekV32ForCausalLM]]_v1")
    assert r.digest == json.loads((MANIFESTS / "records.json").read_text())["rows"][GLM53]["programs"][str(rank)]["step"]


@pytest.mark.slow
@pytest.mark.parametrize("wrapper,shape,digest", [
    ("step", {"M": 8, "pos0": 8}, "6a8ea56b8f7c3f2a4c557d20dd45b86b9eda749f653f4559ee21327ca3bed830254af606c6aee98b123ccc4cb6172fb2283a96be63867331687fa0f5100e4bd2"),
    ("request", {"LP": 11, "T": 12}, "f22d03019ee5c9324134c394b8628cf3bda5fa89d3e9e04b3fe56faf92e856bc5e05eb057ed2adb327e513dc39fa5c93597bfec0011deab079629e3c974c5bcd"),
])
def test_dsa_launch_batch_equals_the_traced_build(wrapper, shape, digest, tmp_path):
    """GLM-5.3's B8 mixed-arrivals row (no batch invariance) at rank 0, as the Build traced it: the step of 8
    rows, and the first request's Program under its launch batch (its prefill alone, then decodes beside the second request's
    prefill and decodes), every token-count-dependent selection at the launch's tokens."""
    r = build_program(_row_phi(GLM53_B8, "GLM53", tmp_path), wrapper, **shape)
    assert r.digest == digest


@pytest.mark.slow
@pytest.mark.parametrize("row,wrapper,shape,digest", [
    (GLM53_S1, "step", {"M": 1, "pos0": 8},
     "24fae34c8de54a0808878316398bcb7228e39fb073c2df31ba00cda2e17da379c8b66a147e5faa6cc1f44ce9fd3fbabe17567af138a97f3e01c361e72df05821"),
    (GLM53_S8, "step", {"M": 8, "pos0": 8},
     "0801afbcf2b8b21823610c8418cfacb82a9f7c0a2cf6641eaa2e362cb0184f33441d467638813027b2e0d7bc152d6bfd05694d0133022ad59e27a4c785cf2776"),
    (GLM53_S8, "request", {"LP": 25, "T": 6},
     "f11311f5b62f788edadc672304f7f6a37cb5355f48d31d8382322cc2f95801ccf3c66abb7cad0a577e27c2a78aa0e4b7641900270f559c661982e68b656819d3"),
])
def test_dsa_stochastic_equals_the_traced_build(row, wrapper, shape, digest, tmp_path):
    """GLM-5.3's seeded top-p rows (t0.8 p0.95) at rank 0: the B1 step as the Build traced it, and the B8 step
    and second request at the same registered split count, the select after the vocabulary's all-gather.  The B8 row's traced Build
    read a derived `splits` Input, which serving no longer produces (`engine.topp_pin`); these two
    digests are the registered-S Programs, until that row is rebuilt.  The second request's launch batch is its realized arrival
    (`engine.arrivals.realized_arrive_steps`): the first request ends at forward 1, so the idle engine prefills it alone at forward 2
    and it decodes alone, not beside the step-13 prefills its declared arrive_step 9 would put it with."""
    r = build_program(_row_phi(row, "GLM53", tmp_path), wrapper, **shape)
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
@pytest.mark.parametrize("rank", [0, 7])
def test_mhc_rank_step_equals_the_record(rank, tmp_path):
    """DeepSeek-V4-Pro at TP8 on H200 (block FP8, MXFP4 experts, cc 9.0): the first and last rank's step through the hyper-connected
    decoder (`predict.dsv4`) is the rank's step Program `records.json` pins for the switched row:
    the mHC pre and post-pre between sublayers, the compressed-attention decode over each layer's opened window, the table and
    sqrtsoftplus routers, the Marlin experts over the schedule's align keys.  About 25 s a rank, nearly all of it encoding the mHC
    Definitions' bodies for the digest; no cheaper test reaches this decoder, and the ranks between differ only in the peer order
    GLM-5.3's ranks pin."""
    r = build_program(_row_phi(DSV4, "DSV4_PRO", tmp_path, rank), "step", M=1, pos0=8)
    assert r.function.startswith("Derived[vLLMStep[DeepseekV4ForCausalLM]]_v1")
    assert r.digest == json.loads((MANIFESTS / "records.json").read_text())["rows"][DSV4]["programs"][str(rank)]["step"]


def test_mhc_refuses_outside_its_domain(tmp_path):
    phi = _row_phi(DSV4, "DSV4_PRO", tmp_path)
    short = {**phi.hf_config, "compress_ratios": [4] + phi.hf_config["compress_ratios"][1:], "index_topk": 1}
    with pytest.raises(Unexpressible, match=re.escape("position 7 has 2 compressed rows, more than the indexer's top-1")):
        build_program(Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": short}}), "step", M=8, pos0=0)
    cfg = {**phi.hf_config, "compress_ratios": [1] + phi.hf_config["compress_ratios"][1:]}
    with pytest.raises(Unexpressible, match=re.escape("compress ratios [1] (the compressors: [4, 128]")):
        build_program(Phi.from_json({**phi.to_json(), "model": {**phi.model, "hf_config": cfg}}), "step")


@pytest.mark.parametrize("sampling,M,digest", [
    ({**STOCH}, 8, "5e6a93b800645792671c4a55c6a3981d8f7dc9ae1a8c03ac754e1887a9dac57f78b4fbeb1f951f625feb7a1beee1ac9973c80b555919144be4250186728fae0e"),
    pytest.param({**STOCH, "top_p": 1.0, "splits": 32}, 1, "a027f8b304f152187eef0e47466573131375e5dc5f75a8b5abb8ca49490b65fcf9e7fc1e040b78fb0cca04e367a4686ef0fad45153d7c2e6ecc1c282636103b0",
                 marks=pytest.mark.slow),
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
    assert S.sampling_of({**one, "requests": one["requests"] * 8}, _phi(TINY)) == {**STOCH, "splits": 32}
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
    names = {re.match(r"\w+", k).group() for k in r.desc["definitions"]}
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


#: The H200 int8 Llama-3.2-1B rows' traced Builds (the same digests
#: on two H200 nodes): each Program digest the row's `build_summary.json` gives (the step at its M and pos0 8,
#: each request shape (LP, T), GP-01's workload) and its `manifest.json`'s digest and identities, with the Programs its header names
#: (`program_digest`, and a batch's `component_digests`).  B4's lone `build_request` (LP256_T31, no request of its batch) is B1's.
H200_INT8 = {
    "llama32-1b__int8__h200__tp1__b1__i256__o32__mixed__greedy__bi-eager": {
        "step": (1, "66cc640b4c9736b2b784626f7987f865499ee9b62c0dc47b8b4e666dcc9990e538bc46e18fbd431221031b96dcfc4d608e38af457f6bd044d47afb07b8ca296f"),
        "requests": {(256, 31): "963a228135f64eb6db7d423209c47f42f544aabfa05d7733f58f5912a7bc3241bb907af6ed69bb89203ae413042971cc0bfb30218ef6f90f2252fc58b0d99c23"},
        "workload": "3acd044df21073dc7cfa7a999fd531f9e46055221fda040c54a0f6f5a81a980fb4d9ac11c6c7ff6174483f5e9ca0fd4a0b23206dc227ba788f91b262cfee26a4",
        "manifest": ("6715e143e1ff21ea7baaf14cf496fe8706e2015f641bae337dfe1743787cd215", 12163, ("request LP256_T31",)),
    },
    "llama32-1b__int8__h200__tp1__b4__i256__o32__mixed__greedy__bi-eager": {
        "step": (4, "b24251993d3ede502ee5933b1688f95d871303f415a8553b152dba32fcce848b136f81238095b8fb8fb5adb168b523ba95c28a228109af54c1269dd150e54f62"),
        "requests": {(11, 8): "bec8910e13b197706b72c29b9e64808d8c4bb8715e75880bd2ef142db8d74f399b77226af6e0725829ea34ef3adbd3f93063e224144baa75dec6bb64af1734d4",
                     (123, 19): "5426f0fb0717df5ca97c05aca07dc9b0f7affa808e66c6928b3f9f30843b5f14ba0107b1ab76db8c1dd663d1f5db6dca1debc9363928fac716ddaab5f0e843b7",
                     (159, 31): "878afca2b9a5f48d77183a3f008c5acd72ce5a0fc7976894c32b245f67079a60df0aff9579bda19ba580f34d5ecb7c954d5a23f4ec775c5ae4118e778d1fb7dd",
                     (256, 1): "05780e3bd779ed8797e079b98cb00755381dbcbf86ff06b6c571fb4db454e2b1d1991943b4564cbcd09159314da23fa4078d89f24c5f9db72053e0ee8514d1c3"},
        "workload": "b611b68a180f306936fb81c85d26be81a044a31d0fb70dea09a786a91343cb9f332d71f285d2d047d0897d295f5856aaa1830ce89fba940dcce24072e6b436ec",
        "manifest": ("4d294fcb0d3ad28eddbd434dfd2773fffc5a379fda7c7796713ce0a71e990e7b", 23230,
                     ("workload", "request LP11_T8", "request LP123_T19", "request LP159_T31", "request LP256_T1")),
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
    the int8 linears or the workload fails here by name.  This builds no manifest (`predict row` does, over its own named Programs, and
    `predict compare --predicted` holds it to the trace's): the traced manifest is pinned with the Programs its header names, and a moved
    one names the manifest it leaves behind.  About 2 min."""
    pin = H200_INT8[row]
    w = json.loads((WORKLOADS / f"{row}.json").read_text())
    phi = S.phi_of({"row": row, "workload": w}, config=str(pinned_hf_config(w["model"], RowEngine.of(w).revision)))
    assert sorted(phi.request_shapes()) == sorted(pin["requests"]), "the workload's request shapes are not its Build's"
    M, step = pin["step"]
    want = {"step": step, **{f"request LP{lp}_T{t}": d for (lp, t), d in pin["requests"].items()}, "workload": pin["workload"]}
    got, canon, comps = {"step": build_program(phi, "step", M=M, pos0=8).digest}, {}, {}
    for lp, t in pin["requests"]:
        r = build_program(phi, "request", LP=lp, T=t)
        got[f"request LP{lp}_T{t}"], comps[(lp, t)] = r.digest, S.request_component(r, canon)
    got["workload"] = S.predict_workload(phi, comps, canon)["digest"]
    moved = [k for k in want if got[k] != want[k]]
    manifest, identities, over = pin["manifest"]
    stale = [k for k in over if k in moved]
    assert not moved, "\n".join(
        [f"{row}: predict moved from the row's traced Build (re-trace the row on the H200 and pin the new digests with the predictor's)"]
        + [f"  {k}: Build {want[k][:16]}…, predicted {got[k][:16]}…" for k in moved]
        + ([f"  the traced manifest {manifest[:16]}… ({identities:,} identities) is over {', '.join(stale)}"] if stale else []))


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


def test_memory_estimate_is_above_a_measured_peak():
    """A Qwen3-4B request of LP+T = 79 peaked at 814 MB resident (`_child`'s max_rss_mb); the pool admits units by the estimate."""
    assert S.mem_estimate_mb(model_shape(json.loads((FIX / "qwen3-4b.config.json").read_text())), [79]) > 814


def test_score_never_writes_into_a_rows_root(tmp_path):
    rows = tmp_path / "rows"
    (rows / "job" / "row").mkdir(parents=True)
    with pytest.raises(UsageError, match="never written into a Build row"):
        S._refuse_inside(str(rows / "job" / "row" / "score.json"), [str(rows)])
    S._refuse_inside(str(tmp_path / "out" / "score.json"), [str(rows)])


def test_the_record_path_reaches_the_predictor_only_through_a_switched_rows_aot_build():
    """No module outside `verity_vllm.program.predict` imports it: the record path runs it only as `predict emit`, the derive of a switched
    row's AOT Build (`row_tp.TpRow.derive`, under `config_record.aot_of`; `tests/pipeline/test_tp_build_width.py`), and the trace
    that checks it lists the package as a forbidden input."""
    from verity_vllm.program.frontend import inputs_trace
    here, name = Path(S.__file__).resolve().parent, S.__name__.rpartition(".")[0]
    pkg = here.parents[name.count(".") - 1]
    inside = here.relative_to(pkg).parts
    pat = re.compile(rf"^\s*(from|import)\s+({re.escape(name)}|\.+{here.name})\b|import_module\([^)]*{here.name}", re.M)
    outside = [p for p in pkg.rglob("*.py") if p.relative_to(pkg).parts[:len(inside)] != inside]
    hits = [str(p.relative_to(pkg)) for p in outside if pat.search(p.read_text())]
    assert hits == []
    runs = [str(p.relative_to(pkg)) for p in outside if re.search(r"""cli\(\s*["']predict["']""", p.read_text())]
    assert runs == ["pipeline/row_tp.py"]
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
    version = _portions(monkeypatch, tmp_path, ("upper/verity/ml/__init__.py", "lower/verity/ir/__init__.py", "lower/verity/ir/defs.py",
                                                "cwd/verity/primitives/circuits/src/verity/ir/defs.py"))
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
