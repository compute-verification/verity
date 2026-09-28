"""padrev — regression guards on GP-01's compose (`verity_vllm.program.workload`) around global da3edf461 (M-0362 / coord ruling 33: the request
Program's ROOT is exempt from the shared-definition canon because two derives of one model at the same prompt length and different caps
carry ONE root id `Root[Derived[..]{SRC=..}]` over two bodies -- SRC names the input shapes, not T).

What a reviewer wants pinned so the exemption cannot widen: (1) the FIXED POPULATION counts DECLARED requests, not distinct shapes --
two identical requests (one digest, one component object) are two members with two disjoint address intervals; (2) the exemption is the
root's ONLY: a callee under one id with two bodies is refused by name in the SAME compose whose roots collide (the fail-closed
E_IDENTITY rule does not spill), and an equal callee is shared (one object); (3) the request root id is not a name in the workload's
namespace at all -- the component is `WorkloadRequest{program_digest, program_fn}`, keyed by the descriptor digest -- so nothing of
record can key on the colliding id.  Tests that need da3edf461 skip BY NAME on a tree without it (the pre-fix refusal is the root's)."""
from __future__ import annotations

import pytest

from verity_vllm.program import workload as W
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, IdentityConflict, PrimitiveDefinition
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from tests.program.test_workload_program import _WP, _desc, _wl
from verity.ml.circuits.types import Array, Value

_ADD = PrimitiveDefinition("PadrevCmpAdd16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: (a + b) & 0xFFFF)
_SRC = {"class": "ToyForCausalLM", "mode": "profile", "inputs": [[3]], "config_sha256": "c" * 16}   # no T: derive.py's SRC shape


def _layer(variant: str):
    """`PadrevLayer_v1{K=1}` -- ONE id, the body chosen by `variant`: 'a' adds w[0], 'b' adds w[1].  Two request Programs whose bodies call
    the same id with different bodies are the E_IDENTITY case for a CALLEE (never first-wins)."""
    def sig(S):
        return ((("x", Value(16)), ("w", Array(4, Value(16)))), Value(16))

    def body(B, S, x, w):
        return B.call(_ADD, x, w[0 if variant == "a" else 1])
    return CompositeDefinition("PadrevLayer", 1, ("K",), sig, body, register=False).bind(K=1)


def _derived_like(LP: int, T: int, layer):
    """derive.py's shape: `Derived[..]_v1{SRC=..}` root, SRC without T, body = prefill + T decode steps each through `layer`."""
    def sig(S):
        return ((("prompt", Array(LP, Value(16))), ("weights", Array(4, Value(16)))), Array(T + 1, Value(16)))

    def body(B, S, prompt, weights):
        acc = B.call(_ADD, prompt[0], weights[0])
        outs = [acc]
        for _ in range(T):
            acc = B.call(layer, acc, weights)
            outs.append(acc)
        return coll_concat(Array(T + 1, Value(16)), outs)
    return CompositeDefinition("Derived[PadrevRequest]", 1, ("SRC",), sig, body, register=False).bind(SRC=_SRC)


def _component_or_skip(desc, LP, canon):
    """request_component; a refusal naming the ROOT id is the pre-da3edf461 tree -> skip by name (never a silent pass)."""
    try:
        return W.request_component(desc, REGISTRY, _WP(LP), canon=canon)
    except IdentityConflict as e:
        if desc["root"] in str(e):
            pytest.skip(f"global da3edf461 (request root exempt from the shared-definition canon) is not on this tree: {e}")
        raise


def test_two_identical_requests_are_two_members_of_the_fixed_population_not_one():
    """One digest, ONE component object (the compose cache) -- and TWO members: two component calls, two disjoint address intervals
    partitioning the non-input gates, two chronology rows, Σ cap = 2·cap events; the workload digest differs from the one-request
    workload's.  The population is the manifest's declared requests, never the set of distinct shapes."""
    wl1, wl2 = _wl([(3, 2)]), _wl([(3, 2), (3, 2)])
    am = wl2.address_map()
    assert [r["program_digest"] for r in am["requests"]] == [program_digest(_desc(3, 2))] * 2
    assert am["requests"][0]["specialization"] == am["requests"][1]["specialization"]      # one component
    pop = wl2.population()
    assert pop["component_calls"] == 2 and pop["components_partition_non_input_gates"] and pop["sum_equals_total"]
    (a0, b0), (a1, b1) = pop["component_intervals"]
    assert b0 == a1 and a0 < b0 < b1 and (b0 - a0) == (b1 - a1) == wl1.population()["component_gates"]
    ch = wl2.chronology()
    assert [r["request_id"] for r in ch["requests"]] == ["r0", "r1"] and all(r["programmed_steps"]["count"] == 3 for r in ch["requests"])
    assert wl2.digest != wl1.digest
    # the shared parameter maps BOTH members onto the one weights interval; each prompt onto its own
    ins0, ins1 = am["requests"][0]["inputs"], am["requests"][1]["inputs"]
    w0 = next(i for i in ins0 if i["name"] == "weights"); w1 = next(i for i in ins1 if i["name"] == "weights")
    p0 = next(i for i in ins0 if i["name"] == "prompt"); p1 = next(i for i in ins1 if i["name"] == "prompt")
    assert w0["global"] == w1["global"] == "shared.weights" and w0["gates"] == w1["gates"]
    assert p0["global"] != p1["global"] and p0["gates"][1] == p1["gates"][0]                 # two prompt intervals, disjoint, adjacent


def test_the_request_root_id_is_not_a_name_in_the_workload_namespace():
    """The request root `Root[ToyRequest_v1{LP,T}]_v1` never reaches the composed workload's definitions: the component is
    `WorkloadRequest_v1{program_digest, program_fn}` keyed by the descriptor digest.  So a colliding root id (da3edf461's case) is not
    a name anything of record keys on -- and no second identity per position can arise from it (ONE value namespace)."""
    d = _desc(3, 2)
    wl = _wl([(3, 2), (3, 3)])
    defs = encode_program(wl.program)["definitions"]
    assert d["root"] not in defs and not any(k.startswith("Root[ToyRequest") for k in defs)
    comps = [k for k in defs if k.startswith("WorkloadRequest_v1{")]
    assert len(comps) == 2 and all(f'program_digest="{program_digest(_desc(3, T))}"' in " ".join(comps) for T in (2, 3))
    assert sum(1 for k in defs if k.startswith("Root[")) == 1          # the workload's own root, nothing else


def test_one_root_id_over_two_bodies_composes_and_the_shared_callee_is_one_object():
    """da3edf461's positive, with the callee in view: same SRC (one root id) over T=2 / T=3 bodies, both calling the SAME
    `PadrevLayer_v1{K=1}` (equal bodies) -> composes by digest; the callee is canonicalised to ONE object shared by both components;
    the root id is in neither the canon nor the workload's definitions; population Σ cap with two disjoint intervals."""
    La, Lb = _layer("a"), _layer("a")
    d2, d3 = encode_program(Program(_derived_like(3, 2, La))), encode_program(Program(_derived_like(3, 3, Lb)))
    assert d2["root"] == d3["root"] and d2["definitions"][d2["root"]] != d3["definitions"][d3["root"]]
    canon: dict = {}
    c2 = _component_or_skip(d2, 3, canon)
    c3 = _component_or_skip(d3, 3, canon)
    assert d2["root"] not in canon and "PadrevLayer_v1{K=1}" in canon
    layer_objs = {id(n.fn) for c in (c2, c3) for n in c.fn.body.nodes if n.fn.id == "PadrevLayer_v1{K=1}"}
    assert len(layer_objs) == 1
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, c2, T=2), W.WorkloadRequest("r1", 1, 3, 4, 0, c3, T=3)], manifest_digest="m" * 8, lag=0)
    defs = encode_program(wl.program)["definitions"]
    assert d2["root"] not in defs and "PadrevLayer_v1{K=1}" in defs
    pop = wl.population()
    assert pop["component_calls"] == 2 and pop["components_partition_non_input_gates"]
    (a0, b0), (a1, b1) = pop["component_intervals"]
    assert b0 == a1 and (b1 - a1) - (b0 - a0) == c3.body_gates - c2.body_gates > 0
    assert [r["programmed_steps"]["count"] for r in wl.chronology()["requests"]] == [3, 4]


def test_the_exemption_is_the_roots_only_a_colliding_callee_in_the_same_compose_is_refused_by_name():
    """Same colliding roots, but the two requests' `PadrevLayer_v1{K=1}` bodies DIFFER (w[0] vs w[1]): the compose must refuse with
    E_IDENTITY naming the CALLEE -- the root exemption does not spill onto anything the root calls, and 'first wins' never happens.
    Order-independent: refused whichever request is canonicalised first."""
    for first, second in (("a", "b"), ("b", "a")):
        d2 = encode_program(Program(_derived_like(3, 2, _layer(first))))
        d3 = encode_program(Program(_derived_like(3, 3, _layer(second))))
        assert d2["root"] == d3["root"]
        canon: dict = {}
        _component_or_skip(d2, 3, canon)
        first_obj = canon["PadrevLayer_v1{K=1}"][0]
        with pytest.raises(IdentityConflict) as ei:
            W.request_component(d3, REGISTRY, _WP(3), canon=canon)
        if "PadrevLayer_v1{K=1}" not in str(ei.value):
            assert d3["root"] in str(ei.value)                                    # the pre-fix tree refuses the ROOT first: named skip
            pytest.skip(f"global da3edf461 not on this tree (the root, not the callee, is what was refused): {ei.value}")
        assert canon["PadrevLayer_v1{K=1}"][0] is first_obj                       # the first body stays canonical; nothing was replaced
