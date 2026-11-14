"""verity_vllm.program.predict.node_window: GLM-5.3's request Program at TP8 rank 0 in the registered rank form, cut to its
first decoder layer, lowered into a window by the block-FP8 linears and the RMSNorms cut at their scale. A 32-id prompt
takes DeepGEMM's branch (M >= 32) at its prefill and FlashInfer's at its decode. Each GEMM is its coordinates, one output
word a unit, so the DSA norm's sum of squares, which reads q_a's 2,048 words whole, reads them through the two-level
gather (64 coordinates a chunk, then 32 chunks a row), and its apply units read those chunks, four a block of 256. Every
other family's rows, and every other rank's partials, are free inputs listed by name; every template's unit is a proof
block Flock's composer proves, and the Circuit passes its wiring and input checks.
"""
from __future__ import annotations

import json
import re

import pytest
from verity.commitments.windows import interface as IF
from verity.protocols.accounting.certifier.verifier import advice as I1

from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered, plan_steps, request_program

GLM53 = {"repo": "zai-org/GLM-5.3", "revision": "aca966e4e02791568aa6a4ced368624b3d897f42"}
TARGET = {"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False, "attention_backend": None,
          "vllm_revision": "d9105ea80", "numerics_profile": "b1-eager-v3", "flash_attn_version": None, "device_name": "NVIDIA H200",
          "moe_backend": "triton"}
ENGINE = {"max_num_batched_tokens": 2048, "max_model_len": 4096, "long_prefill_token_threshold": 256, "max_num_seqs": 256,
          "enable_prefix_caching": False}
LP, LAYERS = 32, 1


def glm_request(LP: int, layers: int, rank: int = 0, moe: bool = False):
    """Rank ``rank``'s request Program of one greedy request of ``LP`` prompt ids and 2 outputs (its prefill, then one
    decode), GLM-5.3 at TP8 on H200 cut to its first ``layers`` decoder layers (with ``moe``, every one an MoE layer), class
    G's engine, in the registered rank form."""
    hf = json.loads(pinned_hf_config(GLM53["repo"], GLM53["revision"]).read_text())
    dense = 0 if moe else min(layers, hf["first_k_dense_replace"])
    hf = {**hf, "num_hidden_layers": layers, "first_k_dense_replace": dense, "indexer_types": hf["indexer_types"][:layers],
          "mlp_layer_types": ["dense"] * dense + ["sparse"] * (layers - dense)}
    phi = Phi(model={**GLM53, "hf_config": hf, "dtype": "float8_e4m3fn"}, target=TARGET, execution="eager",
              tp={"world": 8, "rank": rank}, engine=dict(ENGINE))
    reg = Registered.of(phi, default_seed=DEFAULT_SEED)
    advice = I1.Advice((I1.RequestAdvice("q0", LP, 2, "length"),),
                       (I1.StepAdvice((I1.Row(0, "prefill", 0, LP),)), I1.StepAdvice((I1.Row(0, "decode", LP, 1),))), (0, 0))
    ingress = {"q0": {"prompt": [(7 * k + 1) % 256 for k in range(LP)], "model": "m", "user": "q0", "temperature": 0.0,
                      "max_tokens": 2, "ignore_eos": True}}
    unit = next(u for u in plan_steps(reg, advice, ingress).requests if u is not None)
    return request_program(reg, unit)


@pytest.fixture(scope="module")
def program():
    return glm_request(LP, LAYERS).program.fn


@pytest.fixture(scope="module")
def window(program):
    return NW.node_window(program)


def _tid(circuit: IF.Circuit, pattern: str) -> str:
    (tid,) = [t for t in circuit.units if re.fullmatch(pattern, t)]
    return tid


def test_window_units(window):
    """Each linear is its coordinates (q_a's 2,624 at each of the prefill's 32 tokens and at the decode's one; dead-free
    keeps only the last token's path past the latent rows), each norm its sum of squares and its column blocks, and every
    template a proof block."""
    circuit, w = window
    units = dict(circuit.units)
    assert units[_tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=6144,N=1,G=128\}")] == 33 * 2624 + 2 * 3072
    assert units[_tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=2048,N=1,G=128\}")] == 2 * (2048 + 6144)
    assert units[_tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=1536,N=1,G=128\}")] == 2 * 6144
    assert units[_tid(circuit, r"Fp8GroupQuantUe8m0_v1\{K=6144,G=128\}")] == 33
    assert units[_tid(circuit, r"RMSNormApplyCudaVector8_v1\{G=192\}")] == 33 * 32
    assert units[_tid(circuit, r"DsaRMSNormApply_v1\{G=256\}")] == 2 * 8
    for tid, fn in w.b.fns.items():
        IF.check_proof_block(fn, circuit.cuts.get(tid))
    census = w.census(circuit)
    assert re.fullmatch(r"[0-9a-f]{128}", census["digest"])
    assert set(census["units"]) == {"DeepGemmFp8Linear", "FlashInferFp8Linear", "DsaRMSNorm", "RMSNormCudaVector8", NW.GATHER}
    assert census["steps"] == 2


def test_two_level_gather(window):
    """The DSA norm's sum of squares reads q_a's 2,048 coordinates whole through 32 chunks of 64 and one row of the 32;
    its apply units read the chunks, four a block, rather than the whole row."""
    circuit, _ = window
    level1 = _tid(circuit, r"AllGather_v1\{WORLD=64,N=1\}")
    level2 = _tid(circuit, r"AllGather_v1\{WORLD=32,N=64\}")
    sumsq = _tid(circuit, r"DsaRMSNormSumSq_v1\{N=2048,.*\}")
    apply = _tid(circuit, r"DsaRMSNormApply_v1\{G=256\}")
    assert circuit.units[level1] == 2 * 32 and circuit.units[level2] == 2
    pairs = {(w.writer, w.reader, w.port) for w in circuit.wires}
    linear = _tid(circuit, r"ScaledMmFp8BlockSharedScale_v1\{K=6144,N=1,G=128\}")
    assert {(linear, level1, "parts"), (level1, level2, "parts"), (level2, sumsq, "x"), (level1, apply, "x")} <= pairs
    assert circuit.cuts[apply]["x"] == 4 and circuit.cuts[level1]["parts"] == 64 and circuit.cuts[level2]["parts"] == 32


def test_free_inputs(program, window):
    """What no lowering writes is a free input under the family whose rows it is, and every other rank's partials are named
    with the nodes that read them."""
    circuit, w = window
    census = w.census(circuit)
    assert set(census["free"]) == {"AllReduceMnnvl", "AllReduceRmsNormMnnvl", "BatchedGemm", "SiluMul"}
    assert {f.kind for f in w.free} == {"node"}
    assert all(program.body.nodes[int(f.name)].fn.id.startswith(f.family + "_v") for f in w.free)
    assert not any(x.reader == f.tid and x.port == f.port and f.at <= x.at < f.at + f.count for f in w.free for x in circuit.wires)
    assert {"SparseMlaSplit", "AllReduceMnnvl", "EmbeddingShard", "TokenSelect"} <= set(census["unlowered"])
    assert census["peers"] and all(n.split()[2] in census["unlowered"] for v in census["peers"].values() for n in v)


def test_stand_ins(program, window):
    """With ``stand_ins`` (the value check's) the window is the same: its digest, and each free port still free, now with
    the rows that stand in for its Program words kept."""
    circuit, w = window
    c2, w2 = NW.node_window(program, stand_ins=True)
    assert w2.census(c2)["digest"] == w.census(circuit)["digest"]
    assert {(f.tid, f.port, f.at, f.rows) for f in w2.free} == {(f.tid, f.port, f.at, f.rows) for f in w.free}
    assert w2.stood and all(r.tid.startswith(NW.STAND_IN) or r.tid in c2.units for s in w2.stood for g in s[5] for r in g)


def test_partial_families(program):
    """With the norms alone, the linears' rows the norms read are free inputs under the linears' families."""
    norms = {f: NW.scale_norm for f in ("DsaRMSNorm", "RMSNormCudaVector8", "RMSNormFusedCudaVector8")}
    circuit, w = NW.node_window(program, families=norms)
    assert "DeepGemmFp8Linear" in w.census(circuit)["free"]
    assert not any("ScaledMm" in t for t in circuit.units)


def test_moe_layer():
    """Over an MoE layer the shared expert's linears are lowered and the router's and experts' rows are free inputs."""
    circuit, w = NW.node_window(glm_request(LP, 1, moe=True).program.fn)
    census = w.census(circuit)
    assert {"MoeExpertFp8Gemm", "MoeRouterSigmoidBias", "MoeSum"} <= set(census["unlowered"])
    assert any(re.fullmatch(r"ScaledMmFp8BlockSharedScale_v1\{K=256,N=1,G=128\}", t) for t in circuit.units)


def test_no_lowering_refused(program):
    with pytest.raises(Unexpressible, match="no unit"):
        NW.node_window(program, families={})
