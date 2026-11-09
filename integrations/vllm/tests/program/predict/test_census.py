"""census/v1 and the requirement (`verity.ml.circuits.census`, `.requirement`) of a real row's Programs: Llama-3.2-1B B1 256/32
greedy on rtxpro6000 (cov-k05-7), its step (the traced Build `test_predict` pins) and its request Program (a prefill of 256 rows,
then 31 decode steps), each step ending at a sampling event; the census path held to the explicit expansion on `TINY`'s; and
the served MACs by precision of those rows and of GLM-5.3's served constructions on their stand-ins
(`tests.program.test_derive_neutrality`); and the Boolean work (`work`) of a word Program through each Call's Boolean version
(`boolean_version`), held to its lift's and, on Llama-3.2-1B's GEMMs and GLM-5.3's cuBLAS and FP8 expert GEMMs, to their steps'
gates, and on the constants, residual add, embedding shard and fused all-reduce + RMSNorm GLM-5.3 TP8 Calls as root Calls, to
their Boolean versions' gates.
A change to a Definition either Program calls moves the census digests with the Program's: recompute them with `_views`.
The requirement takes the weights as registered (`Program(fn, registered=("weights",))`), whatever the Build binds them as.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pytest

from tests.program.predict.test_predict import FIX, TINY, _phi
from tests.program import test_derive_neutrality as GLM
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)
from verity.ml.circuits.census import PRODUCTS, SILICON, NoBooleanWork, census, family, step_ends
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.requirement import census_gates, census_schedule, from_census, upper_bound
from verity_vllm.program.predict.build import build_program, model_shape
from verity.ml.definitions.boolean import boolean_version, lift
from verity.ml.definitions.registry import b1, b1_tp2, cublas_bf16 as CB, tp_allreduce as TA
from verity_vllm.query.required import SAMPLING_EVENTS

CFG = json.loads((FIX / "llama32-1b.config.json").read_text())
C = model_shape(CFG)
#: a token row's linear MACs: q/k/v, o, gate/up and down per layer, and the LM head
LINEAR = C.NL * C.H * ((C.NH + 2 * C.KVH) * C.D + C.NH * C.D + 3 * C.I) + C.H * C.V
#: one position's K and V over every layer, in bf16
KV_BITS = C.NL * 2 * C.KVH * C.D * 16
#: one position's row of the rotary cos/sin table, in bf16
ROW = C.rot * 16


def _base(rc: dict) -> int:
    """The weights every step reads: all but the rotary table (a step reads its positions' rows) and each layer's four unused
    attention scales (fp32)."""
    return rc["totals"]["inputs"]["registration"]["bits"] - C.max_pos * ROW - C.NL * 4 * 32


def _registered(r) -> tuple[Program, dict]:
    """The Build's Program with its weights registered, and its census at the sampling steps."""
    P = Program(r.program.fn, registered=("weights",))
    return P, census(P, step_ends(P, SAMPLING_EVENTS))


def _views(cfg: dict, wrapper: str, **kw) -> tuple:
    r = build_program(_phi(cfg), wrapper, **kw)
    P, c = _registered(r)
    return r, census(r.program, step_ends(r.program, SAMPLING_EVENTS), r.digest), from_census(c), c


def _by_precision(c: dict) -> dict[str, int]:
    """`c`'s total served MACs by precision, once every class, step and the totals split exactly their served MACs."""
    assert all(sum(e["served_macs_by_precision"].values()) == e["served_macs"] for e in c["classes"].values())
    assert [sum(m.values()) for m in c["steps"]["served_macs_by_precision"]] == c["steps"]["served_macs"]
    assert sum(c["totals"]["served_macs_by_precision"].values()) == c["totals"]["served_macs"]
    return c["totals"]["served_macs_by_precision"]


def _served(c: dict, *families: str) -> int:
    return sum(e["instances"] * e["served_macs"] for e in c["classes"].values() if e["family"] in families)


def test_step_census_and_requirement():
    r, c, req, rc = _views(CFG, "step", M=1, pos0=8)
    assert r.digest.startswith("5e34739ca3f59dda")
    assert c["digest"] == "31fc5a4003fd5b059049f0e46aee1bc36db3a7103b532987c36450650b7f581f881e5ba7fbb21b7a5255dec50e30f62ac4d5280a703d08f66ed31758cbf2669a"
    gemms = [e for e in c["classes"].values() if e["family"] == "Gemm_v2"]
    assert all(e["served_macs"] == e["statics"]["K"] * e["statics"]["N"] for e in gemms)
    assert _served(c, "Gemm_v2") == LINEAR
    assert c["totals"]["served_macs"] == LINEAR + _served(c, "Attention_v5")
    assert _by_precision(c) == {"mac:bf16": LINEAR + _served(c, "Attention_v5")}  # every product a BF16 tensor-core step's
    assert (c["totals"]["classes"], c["totals"]["values"], c["totals"]["tail_atoms"], c["totals"]["receipt_folds"]) == (56, 394, 0, 0)
    assert c["selects"]["gates"] == [2048] and c["selects"]["bits"] == [2048 * 16]  # the row's embedding: H bf16 gathers
    assert rc["totals"]["inputs"]["run"] == {"gates": 1, "bits": 32}  # the token id
    w = _base(rc) + ROW
    assert req.to_json() == {"steps": 1, "live_bits": [w], "read_bits": [w], "registered_bits": w}
    assert c["selects"]["values"] and req.resolved(0)  # the embedding's gathers select from the table the tied LM head reads whole


@pytest.mark.slow
def test_request_census_and_requirement():
    """The row's request Program, about 30 s: the only test of a Program with values across steps (each layer's K and V rows a
    step's attention produces and every later step's reads), which the step and the toys in `ml/circuits` cannot show at scale."""
    r, c, req, rc = _views(CFG, "request", LP=256, T=31)
    assert r.digest.startswith("5a5902e7f950612d")
    assert c["digest"] == "1c29fa44e74d4fa6b275b5d4a9363f8c33469fbfb6a226846f60c723bbdeed6b3324d502edc49bcec99d385c1fb1b2b31f7e8d62bf837857c2f3a33888dd6617"
    assert len(c["steps"]["ends"]) == 32 and c["totals"]["classes"] == 2641
    assert _by_precision(c) == {"mac:bf16": c["totals"]["served_macs"]}
    base, got = _base(rc), req.to_json()
    # prefill reads rotary rows 0-255 and decode step t row 255 + t; a step holds the rows of its own and later positions,
    # the K/V rows so far and the token sampled before, and reads all it holds but the rows of later positions
    assert got["live_bits"] == [base + 287 * ROW] + [base + (32 - t) * ROW + (255 + t) * KV_BITS + 32 for t in range(1, 32)]
    assert got["read_bits"] == [base + 256 * ROW] + [base + ROW + (255 + t) * KV_BITS + 32 for t in range(1, 32)]


def test_glm53_moe_served_macs_by_precision():
    """GLM-5.3's routed experts serve fp8 MACs only, K x N per coordinate of each expert GEMM."""
    moe = census(GLM._moe().program)
    experts = ("MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1")
    assert all(e["served_macs_by_precision"] == {"mac:fp8": e["statics"]["K"] * e["statics"]["N"]}
               for e in moe["classes"].values() if e["family"] in experts)
    assert _by_precision(moe) == {"mac:fp8": _served(moe, *experts)} == {"mac:fp8": 196608}


def test_glm53_dsa_served_macs_by_precision(served):
    """GLM-5.3's DSA layers serve bf16 in the batched GEMMs (B x K x N) and the sparse MLA, and fp8 in the indexer's MQA logits
    (N x H x D)."""
    dsa = census(GLM._dsa().program)
    want = {"BatchedGemm_v1": ("bf16", lambda S: S["B"] * S["K"] * S["N"]), "SparseMla_v1": ("bf16", None),
            "DsaMqaLogits_v3": ("fp8", lambda S: S["N"] * S["H"] * S["D"])}
    for e in dsa["classes"].values():
        if e["family"] in want:
            q, macs = want[e["family"]]
            assert e["served_macs_by_precision"] == {f"mac:{q}": macs(e["statics"]) if macs else e["served_macs"]}, e["family"]
    assert _by_precision(dsa) == {"mac:bf16": _served(dsa, "BatchedGemm_v1", "SparseMla_v1"), "mac:fp8": _served(dsa, "DsaMqaLogits_v3")}
    assert _by_precision(dsa) == {"mac:bf16": 9984, "mac:fp8": 1536}


@pytest.mark.slow
@pytest.mark.parametrize("wrapper,kw", [("step", {"M": 1, "pos0": 8}), ("request", {"LP": 4, "T": 3})])
def test_the_census_path_is_the_expansion_on_tiny(wrapper, kw):
    """`TINY`'s step and a request of four prompt rows and three decode steps (105 k and 181 k gates): `from_census` gives the
    explicit expansion's live gates and bounded reads at every step. About 70 s for the request: only the request has more than
    one step, so only it shows the census path's live gates and reads across decode steps."""
    P, c = _registered(build_program(_phi(TINY), wrapper, **kw))
    by_census, by_gates = from_census(c), upper_bound(P, census_schedule(c))
    for t in range(by_gates.steps):
        assert census_gates(P, c, by_census.live(t)) == by_gates.live(t)
        assert census_gates(P, c, by_census.reads_upper(t)) == by_gates.reads(t)
    want = by_gates.to_json()
    if not all(by_census.resolved(t) for t in range(by_census.steps)):
        want["read_bits_upper"] = want.pop("read_bits")
    assert by_census.to_json() == want


def _gates(fn) -> dict[str, int]:
    """`fn`'s two-input gates and ANDs, from its compiled circuit (`evaluation.bits`)."""
    c = BS.compile(fn).counts
    return {"gates": sum(c.values()), "and": c["and"]}


def _times(n: int, g: dict[str, int]) -> dict[str, int]:
    return {k: n * v for k, v in g.items()}


def _calls(program) -> dict[str, object]:
    return {descriptor_id(nd.fn): nd.fn for nd in program.root.body.nodes if not is_input(nd.fn)}


def _one(fn) -> CompositeDefinition:
    """A root that Calls ``fn`` once, so ``fn`` is a root Call of its Program."""
    return CompositeDefinition("OneCall", 1, (), lambda S: (tuple(fn.signature()[0]), fn.signature()[1]),
                               lambda B, S, *args: B.call(fn, *args), register=False)


@pytest.mark.slow
def test_a_word_step_counts_its_lifts_work():
    """`TINY`'s step through each Call's Boolean version (about 8 s) counts the work its lift (`lift`, every Call its Boolean
    version) counts as a Boolean Program, Call class by Call class, and in every facet."""
    r = build_program(_phi(TINY), "step", M=1, pos0=8)
    c = census(r.program, boolean=boolean_version)
    b = census(Program(lift(r.program.fn)), boolean=True)
    assert c["totals"]["work"] == b["totals"]["work"] and c["steps"]["work"] == b["steps"]["work"]
    assert set(c["totals"]["work"]) == {"tc:bf16", "cuda:gate"}
    for cid in _calls(r.program):
        v = c["classes"][cid]["boolean_version"]
        assert c["classes"][cid]["work"] == b["classes"][v]["work"], cid


@lru_cache(maxsize=1)
def _llama_step():
    return build_program(_phi(CFG), "step", M=1, pos0=8)


def test_llama32_1b_gemms_are_their_steps_and_casts():
    """Each GEMM of Llama-3.2-1B's step, a root Call of its own: N x K/16 BF16 tensor-core steps are `tc:bf16`, the N casts
    (`F2fpBf16_v2`) `cuda:gate`, nothing else."""
    gemms = {cid: fn for cid, fn in _calls(_llama_step().program).items() if family(fn) == "Gemm_v2"}
    assert len(gemms) == 5
    for cid, fn in gemms.items():
        c = census(Program(_one(fn)), boolean=boolean_version)
        e, S = c["classes"][cid], fn.bindings
        dot = boolean_version(fn).bindings["DOT"]
        assert e["work"] == {"cuda:gate": _times(S["N"], _gates(BG.F2fpBf16)),
                             "tc:bf16": _times(S["N"] * S["K"] // 16, _gates(dot))}, cid
        assert c["totals"]["work"] == e["work"]


def test_a_cublas_gemm_counts_its_k16_step_parts_as_tc():
    """GLM-5.3's cuBLAS GEMMs on bits take the Hopper k16 step in its four parts (`cublas_bf16.HOPPER_STEP`): a word
    `BatchedGemm_v1` through its Boolean version counts, per member, each step's shared part once and its per-row part N times as
    `tc:bf16`, and the N casts `cuda:gate`; on bits it serves B x N x K bf16 products."""
    B, K, N = 2, 32, 3
    word = bind(CB.BatchedGemmWord, B=B, K=K, N=N, DOT=P.HopperBF16WgmmaDot16)
    e = census(Program(_one(word)), boolean=boolean_version)["classes"][descriptor_id(word)]
    first_x, first, x, step = (_gates(part) for part in CB.HOPPER_STEP)
    member = {k: first_x[k] + x[k] + N * (first[k] + step[k]) for k in ("gates", "and")}
    assert e["work"] == {"tc:bf16": _times(B, member), "cuda:gate": _times(B * N, _gates(BG.F2fpBf16))}
    bits = census(Program(bind(CB.BatchedGemm, B=B, K=K, N=N, DOT=BG.HopperBF16WgmmaDot16)), boolean=True)["totals"]
    assert bits["work"] == e["work"] and bits["served_macs_by_precision"] == {"mac:bf16": B * N * K}


#: GLM-5.3 TP8 B1's request's root Calls with no Boolean version of their own: two of its 32-bit constants, the residual add
#: and the fused all-reduce + RMSNorm at their descriptors there, and the embedding shard over 16 rows (the served 19360 x 6144
#: table is a 119M-leaf input)
GLM53_TP8 = [P.const(32, 1), P.const(32, 0x11F), bind(b1.ResidualAdd, N=6144), bind(b1_tp2.EmbeddingShard, VS=16, H=2, START=16),
             pytest.param(bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=6144, SHOT=1, CLUSTER=8, EPS=1e-5, FMA=True), marks=pytest.mark.slow)]


@pytest.mark.parametrize("fn", GLM53_TP8, ids=descriptor_id)
def test_glm53_tp8s_constants_residual_add_embedding_shard_and_fused_norm_count_their_boolean_versions(fn):
    """A root Call of each is refused on words (it reaches no Boolean gate) and, through its Boolean version (a constant's made
    on demand, the rest σ Definitions), counts that version's compiled gates, all `cuda:gate` (the fused norm's 236M, about
    5 s)."""
    cid = descriptor_id(fn)
    with pytest.raises(NoBooleanWork) as refused:
        census(Program(_one(fn)), boolean=True)
    assert list(refused.value.missing) == [cid]
    e = census(Program(_one(fn)), boolean=boolean_version)["classes"][cid]
    v = boolean_version(fn)
    v = bind(v) if isinstance(v, CompositeDefinition) else v
    g = _gates(v)
    assert e["boolean_version"] == descriptor_id(v) and e["work"] == ({"cuda:gate": g} if g["gates"] else {})


@pytest.mark.slow
def test_glm53_fp8_expert_gemms_are_their_steps_and_the_rest():
    """GLM-5.3's routed-expert stand-in through its Boolean versions (about 30 s, its expert GEMMs, quantizers and router traced
    on bits): an expert GEMM's tensor-core gates are its served MACs' FP8 steps (served MACs / 32 of them), `tc:fp8` on QGMMA and
    `tc:f16` on MMAv2 (Triton's e4m3 dot on sm_90 runs as f16 HMMA), and the dequantization and epilogue around them `cuda:gate`;
    nothing but the expert GEMMs has either."""
    program = GLM._moe().program
    moe = census(program, boolean=boolean_version)
    experts = {cid: fn for cid, fn in _calls(program).items() if family(fn) in ("MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1")}
    assert len(experts) == 2
    facet = {}
    for cid, fn in experts.items():
        e, dot = moe["classes"][cid], boolean_version(fn).bindings["DOT"]
        facet[cid] = tc = f"tc:{SILICON[family(dot)]}"
        assert tc in ("tc:fp8", "tc:f16") and set(e["work"]) == {tc, "cuda:gate"} and e["work"]["cuda:gate"]["gates"] > 0
        assert e["work"][tc] == _times(e["served_macs"] // PRODUCTS[family(dot)], _gates(dot)), cid
    for tc in ("tc:fp8", "tc:f16"):
        assert moe["totals"]["work"].get(tc, {"gates": 0, "and": 0}) == {
            k: sum(moe["classes"][cid]["instances"] * moe["classes"][cid]["work"][tc][k] for cid in experts if facet[cid] == tc)
            for k in ("gates", "and")}
