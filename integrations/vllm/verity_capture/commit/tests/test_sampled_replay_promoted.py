"""[R19 int-20, F-r19-int-20] The replay over a PROMOTED member: a Program row the vocabulary calls interior (`Bf16MulScalarTensor_v1`, the
root module's own scaling of the embedding at the unrooted `model/mul`) whose output crosses into the child norms is required by the
manifest as (model, `out`) (required_manifest.PROMOTION_RULE), committed by the collector at its FIRST CONSUMER's input (`<consumer>/in`,
binding.PROMOTED_INPUT_RULE) and bound to the producer's identity.  Here: the manifest of record is built by the engines, its promoted
addresses drive `ProgramIndex` (the replay never re-decides which rows cross), the honest world commits the member under the consumer's
name as the collector does, and the replay (a) evaluates the promoted row as a VU with the registry's bit-exact evaluator and reconciles
the promoted identities against Q(P), (b) fails BY NAME at the promoted row and its consumers when the committed value is not the
producer's, (c) leaves the same rows interior (composed, never addressed) when the manifest promotes nothing."""
from __future__ import annotations

import numpy as np
import pytest

from verity_capture.commit import binding as B
from verity_capture.commit import required_manifest as RM
from verity_capture.commit import sampled_replay as SR
from verity_capture.commit.tests import test_sampled_replay as T
from verity_capture.commit.tests.test_sampled_replay_query_population import _manifest_for, _programs, _weights

PROMPTS = {"r0": [1, 2], "r1": [5, 3]}


def _scaled_weights(woff):
    rng = np.random.default_rng(23)
    n_w = max(hi for lo, hi in woff.values())
    weights = T._bf16(rng, n_w, 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80                                      # 1.0
    weights[woff["model.normalizer"][0]] = 0x4240                                     # bf16 48.0: the root's scale (any bf16 word works)
    return weights


@pytest.fixture(scope="module", params=["inventory", "traversal"])
def scaled_world(request):
    """The scaled Program, its manifest of record from EITHER engine, and the honest world addressed by the manifest's promoted rows."""
    inst, woff = T._program(scale=True)
    weights = _scaled_weights(woff)
    plain = T.Honest(inst, woff, weights, PROMPTS)                                    # addresses: the vocabulary alone (nothing promoted)
    man = _manifest_for(plain, engine=request.param)
    assert man["complete"] is True and man["unmodelled"] == {}, man["unmodelled"]
    promoted = SR.promoted_addresses_of(man)
    world = T.Honest(inst, woff, weights, PROMPTS, promoted=promoted)
    return world, man, request.param


def _bmap(world, man, mutate=None):
    """The honest world's map, the promoted member committed under the consumer's `/in` name exactly as commit_delta arms the collector."""
    pia = B.promoted_input_acquisition(man)
    assert pia["problems"] == [], pia["problems"]
    names = {tuple(v): k for k, v in pia["names"].items()}
    return world.bmap_and_reader(mutate=mutate, committed_names=names), pia


def test_the_manifest_promotes_the_root_modules_scaling_rows_and_names_the_child_norms_as_consumers(scaled_world):
    world, man, engine = scaled_world
    ph = man["query"]["promotion"]
    assert ph["member"] == RM.PROMOTED_MEMBER == "out" and ph["boundary_class"] == RM.INTERIOR_CROSSING and ph["faults"] == {}
    assert set(ph["families"]) == {"Bf16MulScalarTensor_v1"}
    fam = ph["families"]["Bf16MulScalarTensor_v1"]
    assert fam["consumers"] == ["model.layers.0.input_layernorm", "model.norm"]           # the first norm and the final norm's residual read it
    assert fam["rows"] == T.LP + T.T and fam["identities"] == 1 + T.T                     # the header counts one component (request) Program
    prom = [r for r in man["identities"] if r.get("promoted")]
    assert len(prom) == (1 + T.T) * len(PROMPTS) and sum(r["rows"] for r in prom) == (T.LP + T.T) * len(PROMPTS)
    assert {(r["op_path"], r["output_member"], r["dtype"], r["width"]) for r in prom} == {("model", "out", "bf16", T.H)}
    assert {r["step"] for r in prom} == set(range(1 + T.T)) and {r["request_id"] for r in prom} == set(PROMPTS)
    assert all(r["boundary_class"] == RM.INTERIOR_CROSSING and r["spec"] == "Bf16MulScalarTensor_v1" for r in prom)
    # the manifest's promoted addresses -> the replay's address rule: one entry per (step, module), the manifest's signature
    assert SR.promoted_addresses_of(man) == {(s, "model"): {"member": "out", "width": T.H, "dtype": "bf16", "spec": "Bf16MulScalarTensor_v1"} for s in range(1 + T.T)}
    # the collector's arming from the same manifest: observe at the first consumer's input, 2 B per element
    pia = B.promoted_input_acquisition(man)
    assert pia["names"] == {"model.layers.0.input_layernorm/in": ["model", "out"]} and pia["consumers"] == {"model.layers.0.input_layernorm": 2 * T.H}
    # the honest world addressed by the manifest commits the promoted member; the plain one (vocabulary alone) does not
    assert {k[2:] for k in world.blocks if k[2] == "model"} == {("model", "out")}
    assert not any(k[2] == "model" for k in T.Honest(world.inst, world.woff, world.weights, PROMPTS).blocks)


def test_the_promoted_member_observed_at_the_consumers_input_replays_bit_exactly_and_reconciles_against_the_manifest(scaled_world):
    world, man, engine = scaled_world
    (bmap, committed), pia = _bmap(world, man)
    ents = [e for e in bmap["entries"] if e["identity"]["op_path"] == "model" and e["identity"]["output_member"] == "out"]
    assert ents and all(e["name"] == "model.layers.0.input_layernorm/in" and e["observation"] == "consumer-input" for e in ents)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=3, workers=0, manifest=man)
    assert rep["result"] is True and rep["grade"] == "complete", rep["why"]
    assert rep["sample"]["mismatch_n"] == 0
    # the promoted rows are VUs of the population, evaluated with the registry's evaluator (bit-exact bf16 x bf16-scalar, cvt.rn)
    assert "Bf16MulScalarTensor_v1" in rep["population"]["by_family"]
    assert rep["evaluators"]["Bf16MulScalarTensor_v1"].startswith("verity_vllm.registry.derived_rows")
    strata = {k: v for k, v in rep["strata"].items() if k.startswith("Bf16MulScalarTensor_v1")}
    assert strata and all(v["evaluated"] >= 1 and "mismatch" not in v for v in strata.values()), strata
    assert {k.split("|")[1] for k in strata} == set(PROMPTS)
    # Q(P) reconciliation: the promoted identities are in the manifest's total, reached and evaluated, none uncovered, checks ok
    pop = rep["population"]
    assert pop["query_total"]["identities"] == len(man["identities"]) and pop["query_checks"]["ok"], pop["query_checks"]
    assert pop["query_total"]["by_family"].get("instance_outputs", 0) >= (1 + T.T) * len(PROMPTS)
    assert not any("Bf16MulScalarTensor" in str(k) for k in (pop.get("uncovered_by_name") or {})), pop.get("uncovered_by_name")
    # the consumers' VUs READ the committed promoted member (a required value, not a composed one): every norm stratum evaluated equal
    assert all(v["evaluated"] >= 1 and "mismatch" not in v for k, v in rep["strata"].items() if k.startswith(("RMSNormTriton_v1", "RMSNormFusedCuda_v2")))
    # seeded: repeatable
    rep2 = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=3, workers=0, manifest=man)
    assert rep2["sample"] == rep["sample"] and rep2["strata"] == rep["strata"]


def test_a_flipped_word_of_the_committed_promoted_member_fails_by_name_at_the_producer_row_and_its_consumers(scaled_world):
    """The committed `<consumer>/in` is the value the map binds to (model, out) of request r1, prompt row 1.  Flipping one of its words
    makes F_V(embedding row, normalizer) != the committed output at the PROMOTED row, and the norm rows that read the committed member
    disagree with their own committed outputs -- both named at r1 / step 0 / row 1; r0 and the other rows stay equal."""
    world, man, engine = scaled_world

    def mutate(store, entries):
        k = (0, "model.layers.0.input_layernorm/in")
        e = next(x for x in entries if x["identity"]["request_id"] == "r1" and x["name"] == k[1] and x["identity"]["step"] == 0)
        assert e["identity"]["op_path"] == "model" and e["identity"]["output_member"] == "out"
        b = bytearray(store[k])
        b[(e["element_offset"] + 1 * T.H + 3) * 2] ^= 0x01                              # r1, prompt row 1, element 3, low byte
        store[k] = bytes(b)

    (bmap, committed), _pia = _bmap(world, man, mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0, manifest=man)
    assert rep["result"] is False and rep["grade"] == "mismatch", rep["why"]
    prod = [x for x in rep["mismatch"] if x["spec"].startswith("Bf16MulScalarTensor_v1")]
    assert prod and all(x["request_id"] == "r1" and x["step"] == 0 and x["row"] == 1 for x in prod), prod
    assert all(x["op_path"] == "model" for x in prod) and "F_V(committed inputs) != committed output" in rep["why"]
    fams = {x["spec"].split("{")[0] for x in rep["mismatch"]}
    assert "RMSNormTriton_v1" in fams                                                    # the first consumer read the flipped member
    assert all(x["request_id"] == "r1" and x["step"] == 0 and x["row"] == 1 for x in rep["mismatch"]), rep["mismatch"]
    assert rep["mismatch_by"]["by_family"]["Bf16MulScalarTensor_v1"] >= 1 and set(rep["mismatch_by"]["by_request"]) == {"r1"}


def test_a_committed_promoted_member_that_is_missing_is_partial_never_pass(scaled_world):
    """The collector did not commit the consumer's input (the map has no entry for it): the promoted VU has no committed output and its
    consumers have no committed input -- 'not retained', partial, never a pass."""
    world, man, engine = scaled_world

    def mutate(store, entries):
        del store[(1, "model.layers.0.input_layernorm/in")]

    (bmap, committed), _pia = _bmap(world, man, mutate=mutate)
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0, manifest=man)
    assert rep["result"] is None and rep["grade"] == "partial", rep["why"]
    assert any("not retained" in x["why"] for x in rep["not_evaluated"])
    assert any(k.split("|")[3] == "first-decode" for k in rep["strata_unevaluated"])


def test_without_a_promoting_manifest_the_same_rows_stay_interior_and_are_composed_by_their_consumers():
    """The address rule is the manifest's, never the replay's own: with no promoted addresses the scaling rows claim no member (the
    vocabulary's `[]`), are not in the population, and the norm VUs compose the scale from the embedding member -- the pre-R19 reading of
    the same Program, still bit-exact against itself."""
    inst, woff = T._program(scale=True)
    world = T.Honest(inst, woff, _scaled_weights(woff), PROMPTS)
    assert not any(k[2] == "model" for k in world.blocks)
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay(_programs(world), bmap, committed, _weights(world), per_stratum=8, seed=3, workers=0)
    assert rep["result"] is True and rep["grade"] == "complete", rep["why"]
    assert "Bf16MulScalarTensor_v1" not in rep["population"]["by_family"]
    assert not any(k.startswith("Bf16MulScalarTensor_v1") for k in rep["strata"])
    # and the manifest disagrees with such a map: the promoted identity is required, the map does not bind it -> coverage names it
    man = _manifest_for(world, engine="traversal")
    pop = [B.identity(r["program_digest"], r["request_id"], r["step"], r["invocation"], r["op_path"], r["output_member"], 0, r["element_range"][1], r["rank"])
           for r in man["identities"] if r.get("promoted")]
    cov = B.coverage_check(pop, dict(bmap, digest="0" * 64, entries_n=len(bmap["entries"])))
    assert not cov["ok"] and cov["missing_n"] == len(pop) and all(m["op_path"] == "model" and m["output_member"] == "out" for m in cov["missing"])


def test_promoted_addresses_refuse_a_manifest_whose_promoted_rows_disagree_at_one_address(scaled_world):
    world, man, engine = scaled_world
    bad = {"identities": [dict(r) for r in man["identities"]]}
    p = next(r for r in bad["identities"] if r.get("promoted"))
    q = dict(p, width=p["width"] * 2, request_id="r9")
    bad["identities"].append(q)
    with pytest.raises(ValueError, match="disagree"):
        SR.promoted_addresses_of(bad)
    z = dict(p, width=0)
    with pytest.raises(ValueError, match="zero-width"):
        SR.promoted_addresses_of({"identities": [z]})
    assert SR.promoted_addresses_of(None) == {} and SR.promoted_addresses_of({"identities": []}) == {}
