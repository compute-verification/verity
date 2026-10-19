"""census/v1 and the requirement (`verity.ml.circuits.census`, `.requirement`) of a real row's Programs: Llama-3.2-1B B1 256/32
greedy on rtxpro6000 (cov-k05-7), its step (the traced Build `test_predict` pins) and its request Program (a prefill of 256 rows,
then 31 decode steps), each step ending at a sampling event; the census path held to the explicit expansion on `TINY`'s; and
the served MACs by precision of those rows and of GLM-5.3's served constructions on their stand-ins
(`tests.program.test_derive_neutrality`); and the Boolean work (`work`) of a word Program through each Call's Boolean version
(`boolean_version`), held to its lift's and, on Llama-3.2-1B's GEMMs and GLM-5.3's FP8 expert GEMMs, to their steps' gates.
A change to a Definition either Program calls moves the census digests with the Program's: recompute them with `_views`.
The requirement takes the weights as registered (`Program(fn, registered=("weights",))`), whatever the Build binds them as.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pytest

from tests.predict.test_predict import FIX, TINY, _phi
from tests.program import test_derive_neutrality as GLM
from tests.program.test_dsa_derive import served  # noqa: F401  (the DSA stand-in's fixture)
from verity.ml.circuits.census import PRODUCTS, census, family, step_ends
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.requirement import census_gates, census_schedule, from_census, upper_bound
from verity_vllm.predict.build import build_program, model_shape
from verity_vllm.program.boolean import boolean_version, lift
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
    assert r.digest.startswith("773b545605765fe9")
    assert c["digest"] == "da1b33c78fc679ce93080e349be4bdb78be214866db1f01d8c7397e85a49653bd69d09d1d25cd13b21a6b685f85ec0822fdef5013fd04d6cb687d8bfaa1f8741"
    gemms = [e for e in c["classes"].values() if e["family"] == "Gemm_v2"]
    assert all(e["served_macs"] == e["statics"]["K"] * e["statics"]["N"] for e in gemms)
    assert _served(c, "Gemm_v2") == LINEAR
    assert c["totals"]["served_macs"] == LINEAR + _served(c, "Attention_v5")
    assert _by_precision(c) == {"mac:bf16": LINEAR + _served(c, "Attention_v5")}  # every product a BF16 tensor-core step's
    assert (c["totals"]["classes"], c["totals"]["values"], c["totals"]["tail_atoms"], c["totals"]["receipt_folds"]) == (57, 394, 0, 0)
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
    assert r.digest.startswith("2d98d37bd481d2a0")
    assert c["digest"] == "27716ab911c7bfd8ab2290a9de9750eb6a41de86c543565549614ff9753dfebbbd073f4ca35a9ed9652f05327dc4701cc516900f645001dec62444614c25a086"
    assert len(c["steps"]["ends"]) == 32 and c["totals"]["classes"] == 761
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


@pytest.mark.parametrize("wrapper,kw", [("step", {"M": 1, "pos0": 8}), ("request", {"LP": 4, "T": 3})])
def test_the_census_path_is_the_expansion_on_tiny(wrapper, kw):
    """`TINY`'s step and a request of four prompt rows and three decode steps (105 k and 181 k gates): `from_census` gives the
    explicit expansion's live gates and bounded reads at every step."""
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
        one = CompositeDefinition("OneGemm", 1, (), lambda S, fn=fn: (tuple(fn.signature()[0]), fn.signature()[1]),
                                  lambda B, S, *args, fn=fn: B.call(fn, *args), register=False)
        c = census(Program(one), boolean=boolean_version)
        e, S = c["classes"][cid], fn.bindings
        dot = boolean_version(fn).bindings["DOT"]
        assert e["work"] == {"cuda:gate": _times(S["N"], _gates(BG.F2fpBf16)),
                             "tc:bf16": _times(S["N"] * S["K"] // 16, _gates(dot))}, cid
        assert c["totals"]["work"] == e["work"]


@pytest.mark.slow
def test_glm53_fp8_expert_gemms_are_their_steps_and_the_rest():
    """GLM-5.3's routed-expert stand-in through its Boolean versions (about 30 s, its expert GEMMs, quantizers and router traced
    on bits): an expert GEMM's `tc:fp8` gates are its served MACs' FP8 steps (served MACs / 32 of them), and the dequantization
    and epilogue around them `cuda:gate`; nothing but the expert GEMMs has `tc:fp8`."""
    program = GLM._moe().program
    moe = census(program, boolean=boolean_version)
    experts = {cid: fn for cid, fn in _calls(program).items() if family(fn) in ("MoeExpertFp8Gemm_v1", "MoeExpertFp8GemmW_v1")}
    assert len(experts) == 2
    for cid, fn in experts.items():
        e, dot = moe["classes"][cid], boolean_version(fn).bindings["DOT"]
        assert set(e["work"]) == {"tc:fp8", "cuda:gate"} and e["work"]["cuda:gate"]["gates"] > 0
        assert e["work"]["tc:fp8"] == _times(e["served_macs"] // PRODUCTS[family(dot)], _gates(dot)), cid
    assert moe["totals"]["work"]["tc:fp8"] == {k: sum(moe["classes"][cid]["instances"] * moe["classes"][cid]["work"]["tc:fp8"][k]
                                                       for cid in experts) for k in ("gates", "and")}
