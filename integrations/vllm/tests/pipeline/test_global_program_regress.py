"""GP-01 / GM-01 regression tests for the FIXED-POPULATION global Program.  CPU only, toy request Programs (the same lowering /
codec as derive_step's request wrappers, as in `tests/program/test_workload_program.py`).  Each test states the property the lifted
(⊥-padded) Program must keep, so that `Lifted[Workload]` can be run through the same file (same population, same addresses).

  1. ONE VALUE NAMESPACE — every gate of the workload root has exactly one address (input / component intervals partition
     [0, gates)); the shared weights are ONE interval every request maps onto; a position has ONE identity (no duplicated or
     aliased output positions, no root node beyond the Input batches and the one call per request — i.e. no flag object).
  2. FIXED REQUEST POPULATION — the Program, its digest, address map, chronology and output population are functions of the
     manifest (prompt_len, max_tokens, arrive_step) only: two builds from the same manifest are byte-equal, the builder reads no
     execution record, `T != cap-1` is refused, and D_MAX = cap positions exist per request whatever an execution later serves.
  3. DAG WITHIN-STEP CRITERION — `dag_multiset` is invariant under a DAG-safe permutation of independent
     instances and sensitive to an operand swap (a-b vs b-a), a repeated operand ([v,v] vs [v,w]) and a re-wiring.
  4. ARRIVAL CHRONOLOGY — engine-step membership follows `arrive_step` (not manifest order); every (request, step) is programmed
     exactly once; the horizon is max(arrive + cap); an early stop is never a builder input.
  5. batch_decomp WITH MANY WEIGHTS INPUT NODES — a fold whose weights Record is not width-homogeneous
     lowers to one Input node per leaf field (FP8: e4m3 weights + f32 scales + bf16 norms = 435 nodes at Qwen3-4B-FP8); `decompose`
     must treat EVERY Input node that is not a request's prompt/seed parameter as weights (shared), never as another request's
     parameter (XREQ), never refuse on the node count, and resolve each weights node's reads at its Record-global offset (the
     Projection must carry `input_base`).
  6. S READS NO ACTIVITY — GM-01 G6's `sampler_geometry_check` holds every launch to the registered
     S = SplitsFor_v1(1, num_SMs) the served engine pins, whatever the live batch; G6 alone pins the record's rows to the Program's
     activity — request r live at t iff arrive_r <= t <= arrive_r + n_r - 1 with n_r = committed tokens up to and including the first
     declared EOS, capped — in both directions (a missing row: 'declared schedule not exercised'; a row after the EOS: 'execution past
     EOS').  The tests pin that on a real passing record and on the async-scheduling placeholder row, and that
     vLLM's own batch-derived S is refused.
"""

from __future__ import annotations

import collections
import gzip
import json
import os

import pytest

from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import program_compare as C
from verity_vllm.pipeline import global_program as gp
from verity_vllm.program import workload as W
from verity_vllm.records import workload as WL
from verity.ml.circuits.codec import canonical_json, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat, runs
from verity.ml.circuits.types import Array, Record, Value

# ------------------------------------------------------------------------------------------------------------------ toy definitions
# distinct ids from test_workload_program's toys (both may be imported into one process; REGISTRY refuses a second body under one id)


def _reg(fid):
    return REGISTRY[fid] if fid in REGISTRY else None



ADD16 = _reg("PrAdd16_v1") or PrimitiveDefinition("PrAdd16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: (a + b) & 0xFFFF)
SUB16 = _reg("PrSub16_v1") or PrimitiveDefinition("PrSub16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: (a - b) & 0xFFFF)
MIX = _reg("PrMix_v1") or PrimitiveDefinition("PrMix", 1, (("x", Value(16)), ("y", Value(32))), Value(16), lambda x, y: (x + y) & 0xFFFF)

# the weights Record is NOT width-homogeneous -> lowers to one Input batch per field (wa, wb, wc): three weights Input nodes
WREC = Record(("wa", Array(2, Value(16))), ("wb", Array(1, Value(32))), ("wc", Array(2, Value(16))))
WREC_REPR = "{wa: Array<2,Value<16>>, wb: Array<1,Value<32>>, wc: Array<2,Value<16>>}"


def _step_body(B, prev16, weights, s):
    """One 'token step' of the toy: a -> b -> c over the three weight fields, then TokenSelect_v1{V=2} over (b, c) = the sampling
    event that ends the step (the segmentation unit of compare / batch_decomp / global_match).  Returns (token: Value<32>, carry)."""
    a = B.call(ADD16, prev16, weights.field("wa")[s % 2])
    b = B.call(MIX, a, weights.field("wb")[0])
    c = B.call(ADD16, b, weights.field("wc")[s % 2])
    tok = B.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [b, c]))
    return tok, c


def _toy_request(LP: int, T: int):
    """`PrToyReq{LP,T}(prompt: Array<LP,16>, weights: WREC) -> Array<T+1, Value<32>>`: prefill + T decode steps, one token each."""
    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("weights", WREC)), Array(S.T + 1, Value(32)))

    def body(B, S, prompt, weights):
        prev, outs = prompt[0], []
        for s in range(S.T + 1):
            tok, prev = _step_body(B, prev, weights, s)
            outs.append(tok)
        return coll_concat(Array(S.T + 1, Value(32)), outs)

    d = _reg("PrToyReq_v1") or CompositeDefinition("PrToyReq", 1, ("LP", "T"), sig, body)
    return d.bind(LP=LP, T=T)


def _WP(LP):
    return [("prompt", f"Array<{LP},Value<16>>"), ("weights", WREC_REPR)]


def _desc(LP, T):
    return encode_program(Program(_toy_request(LP, T)))


def _wl(shapes, arrivals=None, ids=None, canon=None):
    arrivals = arrivals or [0] * len(shapes)
    canon = {} if canon is None else canon
    reqs = []
    for i, ((LP, T), a) in enumerate(zip(shapes, arrivals)):
        c = W.request_component(_desc(LP, T), REGISTRY, _WP(LP), canon=canon)
        reqs.append(W.WorkloadRequest((ids or [f"r{i}" for i in range(len(shapes))])[i], i, LP, T + 1, a, c, T=T))
    return W.WorkloadProgram(reqs, manifest_digest="m" * 8, lag=0)


# ================================================================================================= 1. one value namespace


def test_every_gate_has_exactly_one_address_and_the_weights_are_one_interval():
    """One value namespace: the input intervals and the component intervals partition [0, gates) — no gate is
    unaddressed, none has two addresses; the shared weights are ONE global interval that every request's local weights parameter
    maps onto (weights once, never per-request copies); same-shape requests are the same specialization at disjoint addresses."""
    wl = _wl([(3, 2), (5, 0), (3, 2)])
    am = wl.address_map()
    intervals = [tuple(i["gates"]) for i in am["inputs"]] + [tuple(r["gates"]) for r in am["requests"]]
    intervals.sort()
    assert intervals[0][0] == 0 and intervals[-1][1] == am["gates"] == wl.program.gates
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert a1 == b0, f"gap or overlap between {a0, a1} and {b0, b1}"          # contiguous AND disjoint = a partition
    assert all(a < b for a, b in intervals)                                        # no empty interval hides an unaddressed gate
    # the shared group: one interval, three Input batches (wa, wb, wc), every request maps its local weights onto exactly it
    sh = am["shared"]
    assert sh["group"] == "weights" and sh["n_batches"] == 3 and sh["root_nodes"] == [0, 3]
    shared_inputs = [i for i in am["inputs"] if i["owner"] == "shared"]
    assert len(shared_inputs) == 3 and [i["gates"] for i in shared_inputs] == [[0, 2], [2, 3], [3, 5]] and sh["gates"] == [0, 5]
    for r in am["requests"]:
        w = [i for i in r["inputs"] if i["global"].startswith("shared.")]
        assert [i["gates"] for i in w] == [[0, 2], [2, 3], [3, 5]]                # the SAME gates for every request
        p = [i for i in r["inputs"] if not i["global"].startswith("shared.")]
        assert len(p) == 1 and p[0]["global"] == f"request[{r['request_id']}].prompt"
    # same shape twice: one specialization, two addresses; instance paths (root node, local index) are distinct per request
    r0, r1, r2 = am["requests"]
    assert r0["specialization"] == r2["specialization"] and r0["program_digest"] == r2["program_digest"]
    assert r0["gates"] != r2["gates"] and r0["root_node"] != r2["root_node"]
    paths = {(r["root_node"], i) for r in am["requests"] for i in range(r["n_instances"])}
    assert len(paths) == sum(r["n_instances"] for r in am["requests"])
    pop = wl.population()
    assert pop["sum_equals_total"] and pop["components_partition_non_input_gates"] and pop["other_root_nodes"] == 0


def test_no_second_identity_per_output_position_and_no_root_node_beyond_inputs_and_calls():
    """A position has one identity: the root body is exactly [Input batches..., one call per request] (no live-flag array, no side
    object, no scheduler node — the lifted Program must keep this, LIFTING-SPEC §4.2 'the flag object is gone'); the output tuple
    references every request's D_MAX = cap output leaves exactly once (no aliased, duplicated or missing position)."""
    wl = _wl([(3, 2), (5, 0), (2, 3)], arrivals=[0, 1, 3])
    body = wl.program.root.body
    K = wl._K
    assert [n.form for n in body.nodes[:K]] == ["batch"] * K and [n.form for n in body.nodes[K:]] == ["call"] * len(wl.requests)
    assert len(body.nodes) == K + len(wl.requests)
    # ret: one leaf per (request, position), each a distinct (node, leaf) reference, in manifest order, cap leaves per request
    ret_runs = runs(body.ret)
    assert all(sp == "n" for sp, _, _, _ in ret_runs)
    leaves = [(idx, j) for sp, idx, lo, hi in ret_runs for j in range(lo, hi)]
    assert len(leaves) == len(set(leaves)) == sum(r.cap for r in wl.requests)
    per_node = {}
    for idx, j in leaves:
        per_node.setdefault(idx, []).append(j)
    assert list(per_node) == [K + i for i in range(len(wl.requests))]                # manifest order, one call node each
    for r_i, r in enumerate(wl.requests):
        assert per_node[K + r_i] == list(range(r.cap)), f"{r.request_id}: output positions are not exactly 0..cap-1"
    # the descriptor round-trips to the same digest (the identity IS the descriptor, no hidden state beside it)
    assert program_digest(encode_program(wl.program)) == wl.digest


# ================================================================================================= 2. fixed request population


def test_program_is_a_function_of_the_manifest_only_two_builds_are_byte_equal():
    """Fixed population: the Program, its digest, address map, chronology and descriptor are
    byte-equal across two constructions from the same manifest facts; nothing an execution later produces (served length, EOS
    step, chunk count, split geometry) is an input of the builder — the builder has no parameter for any of them."""
    a, b = _wl([(4, 3), (2, 1)], arrivals=[0, 2]), _wl([(4, 3), (2, 1)], arrivals=[0, 2])
    assert a.digest == b.digest
    assert canonical_json(a.descriptor) == canonical_json(b.descriptor)
    assert json.dumps(a.address_map(), sort_keys=True) == json.dumps(b.address_map(), sort_keys=True)
    assert json.dumps(a.chronology(), sort_keys=True) == json.dumps(b.chronology(), sort_keys=True)
    # D_MAX = cap positions per request exist structurally: cap sampling events per component, cap output leaves
    for r in a.requests:
        assert len(r.component.step_nodes) == r.cap, f"{r.request_id}: {len(r.component.step_nodes)} steps != cap {r.cap}"
    ch = a.chronology()
    assert ch["component_steps_equal_cap"] is True
    assert "programmed-but-unexecuted" in ch["future_dependent_events"]["eos_stop"]   # the tail stays in the Program
    # the manifest facts ARE the identity: a different cap / arrival / prompt_len is a different Program
    assert _wl([(4, 3), (2, 1)], arrivals=[0, 3]).digest != a.digest        # arrival is part of the declared workload
    assert _wl([(4, 2), (2, 1)], arrivals=[0, 2]).digest != a.digest        # cap
    assert _wl([(5, 3), (2, 1)], arrivals=[0, 2]).digest != a.digest        # prompt_len


def test_a_program_rebuilt_from_a_served_length_is_refused_by_name():
    """The population is D_MAX steps x declared requests: a component whose T is not cap-1 (a Program rebuilt to the served
    length) is refused by name, at every cap and for every request position."""
    canon = {}
    c = W.request_component(_desc(3, 2), REGISTRY, _WP(3), canon=canon)          # a T=2 component: cap must be 3
    for cap in (1, 2, 4, 8):
        with pytest.raises(ValueError, match="never from a served length"):
            W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, cap, 0, c, T=2)], lag=0)
    ok = W.request_component(_desc(5, 0), REGISTRY, _WP(5), canon=canon)
    with pytest.raises(ValueError, match="never from a served length"):
        W.WorkloadProgram([W.WorkloadRequest("r0", 0, 5, 1, 0, ok, T=0), W.WorkloadRequest("r1", 1, 3, 2, 0, c, T=2)], lag=0)


def _derived_dir(tmp_path, LP, T, execution="bi-eager"):
    """A derive_step-shaped component dir for GP-01's `build`: descriptor + instances head (params) + result.json (digest, applicability)
    + construction_manifest (structural inputs) + derive-report (weights field list for compare.load_derived)."""
    d = tmp_path / f"derived_LP{LP}_T{T}"
    d.mkdir()
    desc = _desc(LP, T)
    with gzip.open(d / "descriptor.json.gz", "wb") as f:
        f.write(canonical_json(desc))
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump({"params": [[n, t] for n, t in _WP(LP)]}, f)
    json.dump({"program": {"digest": program_digest(desc)},
               "applicability": {"execution": execution, "numerics_profile": "toy", "sampler": "greedy",
                                 "engine": {"enforce_eager": True, "enable_prefix_caching": False}}}, open(d / "result.json", "w"))
    json.dump({"structural_inputs": {"LP": LP, "T": T}}, open(d / "construction_manifest.json", "w"))
    json.dump({"provenance": {"source": {"example_inputs": [[0] * LP]},
                              "weights": [{"field": 0, "name": "wa"}, {"field": 1, "name": "wb"}, {"field": 2, "name": "wc"}]}},
              open(d / "derive-report.json", "w"))
    return str(d)


def _manifest(tmp_path, reqs, name="workload.json", **extra):
    """A workload record (`records.workload`) of `reqs`, its greedy sampling section holding `extra`, on an eager workload."""
    p = tmp_path / name
    json.dump({"schema": WL.SCHEMA, "model": "toy", "engine": {}, "variants": {"greedy": {"temperature": 0.0, "top_p": 1.0, "top_k": 0}},
               "requests": [{"request_id": f"uuid-{i}", "prompt_token_ids": list(range(1, r["LP"] + 1)), "max_tokens": r["cap"],
                             "arrive_step": r.get("arrive", 0)} for i, r in enumerate(reqs)],
               "sampling": {"variant": "greedy", "temperature": 0.0, **extra}, "sweep": {"execution": "bi-eager"}}, open(p, "w"))
    return str(p)


def test_gp01_build_reads_no_execution_record_and_is_stable_across_builds(tmp_path):
    """`global_program.build` (the GP-01 driver) composes from the manifest + derived components only: `execution_record_read` is
    false, two builds give one digest, and a request's declared cap fixes its component shape — a manifest whose requests were
    served 1 or 3 tokens composes to the SAME Program because served lengths are not in the manifest."""
    reqs = [{"LP": 3, "cap": 3}, {"LP": 5, "cap": 1, "arrive": 1}, {"LP": 3, "cap": 3, "arrive": 2}]
    derived = [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 5, 0)]
    m = _manifest(tmp_path, reqs, eos_token_ids=[7])
    a = gp.build(m, derived, str(tmp_path / "out_a"), eos_token_ids=None)
    assert a["ok"], a
    b = gp.build(m, derived, str(tmp_path / "out_b"), eos_token_ids=None)
    assert a["workload_digest"] == b["workload_digest"]
    assert a["inputs"]["execution_record_read"] is False
    assert a["stop_rule"]["eos_token_ids"] == [7]                                # activity is derived from committed tokens vs THIS set
    assert [r["request_id"] for r in a["requests"]] == ["r0", "r1", "r2"]         # r<k> identity, manifest position
    assert [r["cap"] for r in a["requests"]] == [3, 1, 3] and [r["arrive_step"] for r in a["requests"]] == [0, 1, 2]
    assert json.load(open(tmp_path / "out_a" / "address_map.json")) == json.load(open(tmp_path / "out_b" / "address_map.json"))
    # a component of the wrong shape for a declared cap is not silently reused: the missing shape is named (exit 4 path)
    m2 = _manifest(tmp_path, [{"LP": 3, "cap": 2}], name="w2.json")
    r = gp.build(m2, derived, str(tmp_path / "out_c"))
    assert r["ok"] is False and r["reason"] == "missing_derived" and r["missing_derived"] == [(3, 1)]


def test_gp01_build_to_from_gprog_carries_the_declared_lag_static_end_to_end(tmp_path):
    """GP-01 `build` -> GM-01 `from_gprog` on ITS OWN artifact + descriptor: the Program's
    `lag` static is read where the codec writes it (program_fn / root spec ids inside the verified digest) and equals the artifact's
    lag.value; LAG 0 and LAG 1 are two Programs (two digests); an edited lag section over an intact Program is refused by name."""
    from verity_vllm.check.match import global_match as gm
    reqs = [{"LP": 3, "cap": 3}, {"LP": 5, "cap": 1, "arrive": 1}]
    derived = [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 5, 0)]
    outs = {}
    for lag, target in ((1, None), (0, {"async_scheduling": False}), (1, {"async_scheduling": True})):
        name = f"w_{lag}_{'default' if target is None else 'declared'}.json"
        m = _manifest(tmp_path, reqs, name=name, eos_token_ids=[7])
        if target is not None:
            mm = json.load(open(m)); mm["engine"] = {"target": target}; json.dump(mm, open(m, "w"))
        out = str(tmp_path / f"out_{name}")
        a = gp.build(m, derived, out, eos_token_ids=None)
        assert a["ok"], a
        assert a["lag"]["value"] == lag and a["stop_rule"]["lag"] == lag == a["stop_rule"]["eos_lag_rows"]
        assert a["lag"]["declared_explicitly"] is (target is not None) and a["applicability"]["engine"]["async_scheduling"] is bool(lag)
        assert a["workload_fn"].endswith(f",lag={lag}}}")
        g = gm.from_gprog(os.path.join(out, "workload_program.json"), m)
        dv = g["gprog_artifact"]["digest_verification"]
        assert dv["verified"] is True and dv["lag_static"] == lag, dv
        assert g["lag"]["value"] == lag and g["stop_rule"]["lag"] == lag
        outs[(lag, target is None)] = (a["workload_digest"], out)
    assert outs[(1, True)][0] == outs[(1, False)][0] != outs[(0, False)][0]     # the same LAG = one Program however declared; LAG 0 != LAG 1
    # the section edited to the other value over the intact Program -> refused by name, and the artifact edited to claim NO lag as well
    d1, out1 = outs[(1, True)]
    p = os.path.join(out1, "workload_program.json"); art = json.load(open(p))
    json.dump({**art, "lag": {**art["lag"], "value": 0}}, open(p, "w"))
    with pytest.raises(ValueError, match="lag declaration is not the Program's"):
        gm.from_gprog(p)
    json.dump({k: v for k, v in art.items() if k != "lag"}, open(p, "w"))
    with pytest.raises(ValueError, match="lag declaration is not the Program's"):
        gm.from_gprog(p)
    json.dump(art, open(p, "w"))
    assert gm.from_gprog(p)["gprog_artifact"]["digest_verification"]["lag_static"] == 1


# ================================================================================================= 3. DAG within-step criterion


def _prog_of(fn, label="P"):
    """A `compare.Prog` over a toy root: prompt = Input node 0 (TOK), the three weights Input batches = W fields."""
    desc = encode_program(Program(fn))
    return C.Prog(label, desc, {1: "wa", 2: "wb", 3: "wc"}, tok_node=0)


def _two_chain_def(name, order="ab", op_c=ADD16, second_operand="b"):
    """`(prompt: Array<2,16>, weights) -> Array<1,32>`: two INDEPENDENT chains a = p0 + wa[0], b = p1 + wa[1], then c = op_c(a, x) with
    x = b (default) or a (repeated operand), then TokenSelect over (c, c').  `order` = the emission order of the independent chains."""
    def sig(S):
        return ((("prompt", Array(2, Value(16))), ("weights", WREC)), Array(1, Value(32)))

    def body(B, S, prompt, weights):
        if order == "ab":
            a = B.call(ADD16, prompt[0], weights.field("wa")[0])
            b = B.call(ADD16, prompt[1], weights.field("wa")[1])
        else:
            b = B.call(ADD16, prompt[1], weights.field("wa")[1])
            a = B.call(ADD16, prompt[0], weights.field("wa")[0])
        x = b if second_operand == "b" else a
        c = B.call(op_c, a, x)
        c2 = B.call(MIX, c, weights.field("wb")[0])
        tok = B.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [c, c2]))
        return coll_concat(Array(1, Value(32)), [tok])

    return CompositeDefinition(name, 1, (), sig, body, register=False)


def test_dag_multiset_is_permutation_invariant_and_operand_order_sensitive():
    """The within-step criterion abstracts the EMISSION ORDER of independent instances (a DAG-safe
    permutation) and nothing else — an operand swap (a-b vs b-a), a repeated operand ([v,v] vs [v,w]) and a re-wiring are all
    different DAGs.  The sequence criterion (canonical sha over the emission sequence) differs under the permutation, as it must."""
    P_ab = _prog_of(_two_chain_def("PrDagAB"), "ab")
    P_ba = _prog_of(_two_chain_def("PrDagBA", order="ba"), "ba")
    assert bd.dag_multiset(P_ab) == bd.dag_multiset(P_ba)                                         # same DAG, other emission order
    assert C.compare_steps(P_ab, P_ba)["canonical_sha256"]["equal"] is False                     # the sequence criterion sees the permutation
    # positive control: the same Program twice is sequence-equal
    assert C.compare_steps(P_ab, _prog_of(_two_chain_def("PrDagAB2"), "ab2"))["canonical_sha256"]["equal"] is True
    # operand ORDER matters: c = a - b vs c = b - a are different multisets (subtraction must not exchange its operands)
    P_sub_ab = _prog_of(_two_chain_def("PrDagSubAB", op_c=SUB16), "sub-ab")

    def swapped_sub():
        def sig(S):
            return ((("prompt", Array(2, Value(16))), ("weights", WREC)), Array(1, Value(32)))

        def body(B, S, prompt, weights):
            a = B.call(ADD16, prompt[0], weights.field("wa")[0])
            b = B.call(ADD16, prompt[1], weights.field("wa")[1])
            c = B.call(SUB16, b, a)                                            # b - a
            c2 = B.call(MIX, c, weights.field("wb")[0])
            tok = B.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [c, c2]))
            return coll_concat(Array(1, Value(32)), [tok])
        return CompositeDefinition("PrDagSubBA", 1, (), sig, body, register=False)

    assert bd.dag_multiset(P_sub_ab) != bd.dag_multiset(_prog_of(swapped_sub(), "sub-ba"))
    # a REPEATED operand is a different computation from two distinct operands ([v, v] is two positions of one value)
    assert bd.dag_multiset(P_ab) != bd.dag_multiset(_prog_of(_two_chain_def("PrDagAA", second_operand="a"), "aa"))
    # the live cone covers every instance of a Program whose only output is the sampled token's dependency chain
    assert bd.live_cone(P_ab) == set(range(len(P_ab.inst)))


def test_dag_hash_carries_the_producer_hash_not_its_position():
    """The Merkle form: an instance's hash depends on its producers' HASHES (what they compute) and not on their (step, index)
    positions — the property that makes the criterion emission-order-insensitive; a producer computing something else changes
    every consumer's hash."""
    P = _prog_of(_two_chain_def("PrDagH"), "h")
    h = bd.dag_hashes(P)
    Q = _prog_of(_two_chain_def("PrDagHsub", op_c=SUB16), "hsub")
    hq = bd.dag_hashes(Q)
    # instances 0, 1 (the two independent Adds) are identical computations in both; the Sub differs and so does everything after it
    assert h[0] == hq[0] and h[1] == hq[1]
    assert h[2] != hq[2] and h[3] != hq[3] and h[4] != hq[4]
    assert len(set(h.values())) == len(h)                                       # no two instances of this toy collide


# ================================================================================================= 4. arrival chronology


def test_chronology_follows_arrivals_not_manifest_order_and_programs_every_step_once():
    """Arrival chronology (no Ingress/Egress primitives: arrivals are the declaration): r1 arrives BEFORE r0 —
    engine-step membership follows `arrive_step`, the address map follows manifest order; every (request, s) for s in [0, cap) is
    programmed exactly once (fixed population, no step dropped after an early stop); the horizon is max(arrive + cap)."""
    wl = _wl([(3, 3), (5, 1), (2, 2)], arrivals=[2, 0, 5])
    ch = wl.chronology()
    assert ch["programmed_horizon_steps"] == max(2 + 4, 0 + 2, 5 + 3) == 8
    es = {e["e"]: e for e in ch["engine_steps"]}
    assert [m["request_id"] for m in es[0]["members"]] == ["r1"] and [m["request_id"] for m in es[1]["members"]] == ["r1"]
    assert [(m["request_id"], m["s"]) for m in es[2]["members"]] == [("r0", 0)]     # r1 is done (cap 2), r0 prefill
    assert [(m["request_id"], m["s"]) for m in es[5]["members"]] == [("r0", 3), ("r2", 0)]
    assert es[5]["co_scheduled"] and es[5]["mixed_prefill_decode"]
    assert [(m["request_id"], m["s"]) for m in es[7]["members"]] == [("r2", 2)]
    seen = [(m["request_id"], m["s"]) for e in ch["engine_steps"] for m in e["members"]]
    assert len(seen) == len(set(seen)) == sum(r.cap for r in wl.requests)
    for r in wl.requests:
        assert sorted(s for rid, s in seen if rid == r.request_id) == list(range(r.cap))
    # every member names its component segment (the step's instance set) and a global gate range inside the request's interval
    am = {r["request_id"]: r for r in wl.address_map()["requests"]}
    for e in ch["engine_steps"]:
        for m in e["members"]:
            lo, hi = am[m["request_id"]]["gates"]
            assert m["nodes"] is not None and lo <= m["gates"][0] < m["gates"][1] <= hi
    # the address map is in MANIFEST order whatever the arrivals
    assert [r["request_id"] for r in wl.address_map()["requests"]] == ["r0", "r1", "r2"]
    assert ch["within_step"]["declared_order"] is None                         # nothing is sequenced inside an engine step


def test_declared_live_counts_are_a_function_of_the_declaration_only():
    """GP-01's declared chronology live count per engine step (the input of the stochastic `S` schedule, LIFTING-SPEC [D4]'s
    transitional form) is computed from (arrive_step, cap) alone — every request to its cap; no served length enters."""
    reqs = [{"arrive_step": 0, "cap": 3}, {"arrive_step": 1, "cap": 1}, {"arrive_step": 1, "cap": 2}]
    assert gp.declared_live_counts(reqs) == [1, 3, 2]
    assert gp.declared_live_counts([]) == []
    assert gp.declared_live_counts([{"arrive_step": 4, "cap": 1}]) == [0, 0, 0, 0, 1]


# ================================================================================================= 5. batch_decomp: many weights Input nodes


def _oracle_dir(tmp_path, reqs):
    """A fold-shaped oracle dir (program.json + instances.jsonl + capture/log.jsonl.gz) for a B-request batched toy whose root
    parameters are [prompt per request..., weights Record] (the fold contract: weights LAST) with the mixed-width WREC -> the
    descriptor has len(reqs) prompt Input nodes + 3 weights Input nodes.  Instances are emitted engine-step-major (all live
    requests' steps interleaved) with the fold's per-instance meta (`request`, `step`, `row`, `rows`, `hint`, `operand_refs`)."""
    B = len(reqs)

    def sig(S):
        ps = tuple((f"p{i}", Array(r["LP"], Value(16))) for i, r in enumerate(reqs))
        return (ps + (("weights", WREC),), Array(sum(r["T"] + 1 for r in reqs), Value(32)))

    meta = []

    def body(Bd, S, *args):
        prompts, weights = args[:B], args[B]
        prev = [p[0] for p in prompts]
        outs = [[] for _ in reqs]
        horizon = max(r["T"] + 1 for r in reqs)
        for e in range(horizon):
            live = [i for i, r in enumerate(reqs) if e <= r["T"]]
            for row, i in enumerate(live):
                s = e
                a = Bd.call(ADD16, prev[i], weights.field("wa")[s % 2])
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "model/wa",
                             "operand_refs": [[["n", 0, 0, 1]], [["p", B, s % 2, s % 2 + 1]]]})
                b = Bd.call(MIX, a, weights.field("wb")[0])
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "model/wb",
                             "operand_refs": [[["n", 0, 0, 1]], [["p", B, 2, 3]]]})
                c = Bd.call(ADD16, b, weights.field("wc")[s % 2])
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "model/wc",
                             "operand_refs": [[["n", 0, 0, 1]], [["p", B, 3 + s % 2, 4 + s % 2]]]})
                tok = Bd.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [b, c]))
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "sampler", "operand_refs": [[["n", 0, 0, 2]]]})
                outs[i].append(tok)
                prev[i] = c
        return coll_concat(Array(sum(r["T"] + 1 for r in reqs), Value(32)), [t for o in outs for t in o])

    fn = CompositeDefinition("PrBatchedOracle", 1, (), sig, body, register=False)
    desc = encode_program(Program(fn))
    d = tmp_path / "oracle"
    (d / "capture").mkdir(parents=True)
    n_in = B + 3
    json.dump({"program": desc, "digest": program_digest(desc), "n_input_nodes": n_in}, open(d / "program.json", "w"))
    with open(d / "instances.jsonl", "w") as f:
        for i, m in enumerate(meta):
            f.write(json.dumps({"node": i, "spec": "toy", **m}) + "\n")
    header = {"$e": "RunHeader", "schema": "toy", "workload": {"requests": [{"request_id": r["id"], "prompt_token_ids": list(range(1, r["LP"] + 1)),
                                                                                "max_tokens": r["T"] + 1} for r in reqs]}}
    with gzip.open(d / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for e in range(max(r["T"] + 1 for r in reqs)):
            live = [r for r in reqs if e <= r["T"]]
            rows = [[r["id"], (r["LP"] - 1 if e == 0 else r["LP"] - 1 + e)] for r in live]
            f.write(json.dumps({"$e": "StepStart", "step": e, "rows": rows, "logits_rows": rows, "logits_indices": list(range(len(rows)))},
                               separators=(",", ":")) + "\n")
    return str(d), n_in


def test_batch_decomp_treats_every_non_request_input_node_as_shared_weights(tmp_path):
    """B=2 fold over a mixed-width weights Record (3 weights Input nodes after the 2 prompts).  Expected: no refusal,
    every instance attributed, ZERO cross-request reads (the wa/wb/wc reads are SHARED weights, not r1's parameter), every request
    canonical-equal to its own-shape component -> PASS.  One request finishes early (T=1 vs T=2) so the later engine step has one row."""
    reqs = [{"id": "r0", "LP": 3, "T": 2}, {"id": "r1", "LP": 2, "T": 1}]
    oracle, n_in = _oracle_dir(tmp_path, reqs)
    assert n_in == 5
    derived = [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 2, 1)]
    res = bd.decompose(oracle, derived)
    assert res["attribution"]["unattributed"] == 0, res["attribution"]
    assert res["xreq_total"] == 0, [e.get("xreq") for e in res["requests"]]
    assert res["missing_derived"] == []
    by = {e["request_id"]: e for e in res["requests"]}
    assert by["r0"]["LP"] == 3 and by["r0"]["T"] == 2 and by["r1"]["LP"] == 2 and by["r1"]["T"] == 1
    assert all(e["canonical_equal"] for e in res["requests"]), res["reasons"]
    assert res["verdict"] == "PASS", res["reasons"]


def test_batch_decomp_refuses_a_header_that_disagrees_with_the_programs_input_layout(tmp_path):
    """Negative for the widened weights set: the weights nodes are the Input nodes AFTER the request-owned parameters, so a header
    that claims one more request than the Program has prompt parameters (its 'third prompt' would be the first weights node) is
    refused by name — the set is derived from the Program + header together, never guessed from the node count alone."""
    reqs = [{"id": "r0", "LP": 3, "T": 2}, {"id": "r1", "LP": 2, "T": 1}]
    oracle, _ = _oracle_dir(tmp_path, reqs)
    derived = [_derived_dir(tmp_path, 3, 2), _derived_dir(tmp_path, 2, 1)]
    log = os.path.join(oracle, "capture", "log.jsonl.gz")
    lines = gzip.open(log, "rt").read().splitlines()
    hdr = json.loads(lines[0])
    hdr["workload"]["requests"].append({"request_id": "r2", "prompt_token_ids": [1, 2], "max_tokens": 1})
    bad = os.path.join(oracle, "capture", "log_bad.jsonl.gz")
    with gzip.open(bad, "wt") as f:
        f.write(json.dumps(hdr, separators=(",", ":")) + "\n" + "\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="input layout|weights node|input nodes"):
        bd.decompose(oracle, derived, log_path=bad)


def test_single_weights_input_node_baseline_still_passes(tmp_path):
    """Control for the mixed-width fixture machinery: the same toy with a width-HOMOGENEOUS weights Record (ONE Input node on both sides,
    one weight tensor `weights` named by every hint — the dense layout) goes through `decompose` as every dense workload does — attributed, no XREQ, canonical-equal, PASS — so the xfail above is about the weights node COUNT and nothing else."""
    HREC = Record(("wa", Array(2, Value(16))), ("wc", Array(2, Value(16))))
    HREC_REPR = "{wa: Array<2,Value<16>>, wc: Array<2,Value<16>>}"

    def step(Bd, prev16, weights, s):
        a = Bd.call(ADD16, prev16, weights.field("wa")[s % 2])
        c = Bd.call(ADD16, a, weights.field("wc")[s % 2])
        tok = Bd.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [a, c]))
        return tok, c

    def req_def(LP, T):
        def sig(S):
            return ((("prompt", Array(S.LP, Value(16))), ("weights", HREC)), Array(S.T + 1, Value(32)))

        def body(Bd, S, prompt, weights):
            prev, outs = prompt[0], []
            for s in range(S.T + 1):
                tok, prev = step(Bd, prev, weights, s)
                outs.append(tok)
            return coll_concat(Array(S.T + 1, Value(32)), outs)
        d = _reg("PrToyReqH_v1") or CompositeDefinition("PrToyReqH", 1, ("LP", "T"), sig, body)
        return d.bind(LP=LP, T=T)

    reqs = [{"id": "r0", "LP": 3, "T": 2}, {"id": "r1", "LP": 2, "T": 1}]
    B = len(reqs)
    meta = []

    def sig(S):
        return (tuple((f"p{i}", Array(r["LP"], Value(16))) for i, r in enumerate(reqs)) + (("weights", HREC),), Array(5, Value(32)))

    def body(Bd, S, *args):
        prompts, weights = args[:B], args[B]
        prev = [p[0] for p in prompts]
        outs = [[] for _ in reqs]
        for e in range(3):
            live = [i for i, r in enumerate(reqs) if e <= r["T"]]
            for row, i in enumerate(live):
                a = Bd.call(ADD16, prev[i], weights.field("wa")[e % 2])
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "model/weights", "operand_refs": [[["n", 0, 0, 1]], [["p", B, e % 2, e % 2 + 1]]]})
                c = Bd.call(ADD16, a, weights.field("wc")[e % 2])
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "model/weights", "operand_refs": [[["n", 0, 0, 1]], [["p", B, 2 + e % 2, 3 + e % 2]]]})
                tok = Bd.call(REGISTRY["TokenSelect_v1"].bind(V=2), coll_concat(Array(2, Value(16)), [a, c]))
                meta.append({"step": e, "row": row, "rows": len(live), "request": reqs[i]["id"], "hint": "sampler", "operand_refs": [[["n", 0, 0, 2]]]})
                outs[i].append(tok)
                prev[i] = c
        return coll_concat(Array(5, Value(32)), [t for o in outs for t in o])

    fn = CompositeDefinition("PrBatchedOracleH", 1, (), sig, body, register=False)
    desc = encode_program(Program(fn))
    d = tmp_path / "oracle_h"
    (d / "capture").mkdir(parents=True)
    json.dump({"program": desc, "digest": program_digest(desc), "n_input_nodes": B + 1}, open(d / "program.json", "w"))
    with open(d / "instances.jsonl", "w") as f:
        for i, m in enumerate(meta):
            f.write(json.dumps({"node": i, "spec": "toy", **m}) + "\n")
    header = {"$e": "RunHeader", "schema": "toy", "workload": {"requests": [{"request_id": r["id"], "prompt_token_ids": list(range(1, r["LP"] + 1)), "max_tokens": r["T"] + 1} for r in reqs]}}
    with gzip.open(d / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for e in range(3):
            rows = [[r["id"], r["LP"] - 1 + e] for r in reqs if e <= r["T"]]
            f.write(json.dumps({"$e": "StepStart", "step": e, "rows": rows, "logits_rows": rows, "logits_indices": list(range(len(rows)))}, separators=(",", ":")) + "\n")
    derived = []
    for LP, T in ((3, 2), (2, 1)):
        dd = tmp_path / f"derived_h_LP{LP}_T{T}"
        dd.mkdir()
        rd = encode_program(Program(req_def(LP, T)))
        with gzip.open(dd / "descriptor.json.gz", "wb") as f:
            f.write(canonical_json(rd))
        json.dump({"structural_inputs": {"LP": LP, "T": T}}, open(dd / "construction_manifest.json", "w"))
        json.dump({"provenance": {"source": {"example_inputs": [[0] * LP]}, "weights": [{"field": 0, "name": "weights"}]}}, open(dd / "derive-report.json", "w"))
        derived.append(str(dd))
    res = bd.decompose(str(d), derived)
    assert res["attribution"]["unattributed"] == 0 and res["xreq_total"] == 0 and res["missing_derived"] == []
    assert res["verdict"] == "PASS", (res["reasons"], [e.get("first_divergences") for e in res["requests"]])
    by = {e["request_id"]: e for e in res["requests"]}
    assert by["r1"]["T"] == 1 and by["r0"]["T"] == 2                                # the early finisher keeps its own T, r0 its own


# ------------------------------------------------------- 6. the derived S reads the Program's activity
# GM-01's own e2e fixtures (test_global_match: toy fold + batched record with declared arrivals / caps / an EOS stop) are the record of
# reference here; `memory` is their fold fixture.  R1_EOS: cap 4, stops on the declared EOS 11 after 2 events -> a_r = n_r = 2.

from verity_vllm.check.match import global_match as gm  # noqa: E402
from verity_vllm.check.match import sampler_geometry as SG  # noqa: E402
from tests.check import test_global_match as TGM  # noqa: E402
from tests.check.test_global_match import memory  # noqa: E402,F401 - the fold fixture
from verity.ml.definitions.registry.topp_geometry import registered_splits, splits_for  # noqa: E402

NSM = 4   # a num_SMs at which vLLM's own S discriminates 1 vs 2 live rows: splits_for(1, 4) = 4, splits_for(2, 4) = 2 (142 SMs gives 32 for both)
S_REG = registered_splits(NSM)   # 4: the split count the pinned engine launches at every batch

# this section regresses GM-01's derived-S unit: on a tree without it there is nothing to regress -- named skip
_needs_derived_s = pytest.mark.skipif(not hasattr(SG, "sampler_geometry_check"),
                                      reason="GM-01 without sampler_geometry_check")


def _program_activity(gpd: dict, transcripts: dict[str, list[int]], eos_ids: set[int], lag: int = 0) -> dict[int, int]:
    """The Program's activity count per engine step under scheduler lag `lag`: request r is live at t iff arrive_r <= t <= arrive_r + n_r - 1, n_r = its committed tokens up to and INCLUDING the
    first declared EOS plus `lag` placeholder rows when the EOS came before the cap (an EOS AT the cap is a length stop: no placeholder),
    capped at max_tokens.  A function of the declaration, the committed tokens and the attested lag only -- never of which rows the
    engine happened to launch."""
    live: dict[int, int] = collections.Counter()
    for g in gpd["requests"]:
        toks = list(transcripts[g["request_id"]])
        cap = int(g["max_tokens"])
        n = min(len(toks), cap)
        for j, t in enumerate(toks):
            if int(t) in eos_ids:
                n = min(cap, j + 1 + lag)
                break
        for t in range(int(g["arrive_step"]), int(g["arrive_step"]) + n):
            live[t] += 1
    return dict(live)


def _transcripts(oracle: str) -> dict[str, list[int]]:
    tk = json.load(open(os.path.join(oracle, "capture", "tokens.json")))
    return {r["request_id"]: list(r["generated_token_ids"]) for r in tk["requests"]}


class _OperandStubFold:
    """The select carries NO S static (`GumbelTopPTokenSelect_v1{V}`); its launch S is the value of
    the `Const32[S]` 6th operand that compare.Prog._bind_splits removed from the sequence and kept in `splits_values[select node]`."""

    def __init__(self, s_by_index: list[int]):
        self.inst = [{"spec": "GumbelTopPTokenSelect_v1{V=64}", "k": i, "args": []} for i in range(len(s_by_index))]
        self.splits_values = {i: s for i, s in enumerate(s_by_index)}


def _reader_takes_the_bound_operand() -> bool:
    import inspect
    f = getattr(SG, "top_p_split_count", None)
    return f is not None and len(inspect.signature(f).parameters) >= 3


# the S representation the fold carries: a static `{V,S}`; the operand BOUND by compare (`splits_values`); the same operand
# UNBOUND -- the `Const32[S]` instance right before the select, read by one run ("n", kc, 0, 1) (`_OperandFold`'s shape) -- the §6 properties are about the live COUNT and hold for every form;
# a form runs only where its reader exists
_S_FORMS = ["static"] + (["operand"] if _reader_takes_the_bound_operand() else []) + (["unbound"] if hasattr(TGM, "_OperandFold") else [])


def _topp_fold_for(live_per_step: dict[int, int], s_of: dict[int, int], form: str = "static"):
    """A stub fold with one GumbelTopPTokenSelect per live row per step carrying the launch S `s_of[step]` in the given representation,
    plus engine_step_of (keyed by the SELECT instances' indices -- the unbound form interleaves one Const32 instance per select)."""
    s_list, es = [], {}
    for st in sorted(live_per_step):
        for _ in range(live_per_step[st]):
            es[len(s_list)] = st
            s_list.append(s_of[st])
    if form == "operand":
        return _OperandStubFold(s_list), es
    if form == "unbound":
        fold = TGM._OperandFold([(S, None) for S in s_list])
        sel = [i for i, x in enumerate(fold.inst) if x["spec"].startswith("GumbelTopP")]
        assert len(sel) == len(s_list)
        return fold, {sel[n]: es[n] for n in range(len(s_list))}
    return TGM._StubFold([f"GumbelTopPTokenSelect_v1{{V=64,S={S}}}" for S in s_list]), es


@_needs_derived_s
@pytest.mark.parametrize("form", _S_FORMS)
def test_the_registered_s_holds_on_every_step_of_a_passing_record_whatever_its_activity(memory, tmp_path, form):
    """On a record GM-01 PASSES, the record's row count at every engine step equals the Program's activity (declared arrival x committed
    tokens, truncated at the verified EOS) -- not the declared-to-cap schedule: r1 (cap 4) stops on EOS at event 1, so steps 2 and 3 of
    its declared schedule are NOT live, and step 3 is not a step of the record at all.  S reads none of it: the registered
    S = SplitsFor_v1(1, 4 SMs) = 4 passes at 2 live rows as at 1, and vLLM's own batch-derived S (2 live -> S=2) is refused by name."""
    gpd, oracle = TGM.assemble([TGM.R0, TGM.R1_EOS], tmp_path)
    res = gm.check(gpd, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    rec = gm.read_record(os.path.join(oracle, "capture", "log.jsonl.gz"))
    eos = set(gpd["stop_rule"]["eos_token_ids"])
    activity = _program_activity(gpd, _transcripts(oracle), eos)
    observed = {s: len(rec["steps"][s]["logits_rows"]) for s in rec["steps"]}
    assert activity == observed == {0: 2, 1: 2, 2: 1}
    # every request to its cap: the schedule GP-01 declares beside the Program (builder-side rows carry `cap`)
    declared = gp.declared_live_counts([{"arrive_step": g["arrive_step"], "cap": g["max_tokens"]} for g in gpd["requests"]])
    assert declared == [2, 2, 2, 1] and declared[2] == 2 != activity[2] == 1 and 3 not in observed and declared[3] == 1
    gsm = dict(gpd, sampler_geometry={"num_sms": NSM})
    ok_fold, ok_es = _topp_fold_for(observed, {s: S_REG for s in observed}, form)
    ok = SG.sampler_geometry_check(gsm, ok_fold, ok_es, rec["steps"], rec["sampler_batch"])
    assert ok["problems"] == [] and ok["info"]["checked_steps"] == 3 and ok["info"]["live_to_S"] == {"live=1:S=4": 1, "live=2:S=4": 2}
    # (-) an unpinned launch at vLLM's S of the batch it ran (2 rows at steps 0 and 1 -> S=2): refused by name at both
    bad_fold, bad_es = _topp_fold_for(observed, {s: splits_for(observed[s], NSM) for s in observed}, form)
    bad = SG.sampler_geometry_check(gsm, bad_fold, bad_es, rec["steps"], rec["sampler_batch"])
    assert len(bad["problems"]) == 1 and bad["info"]["steps_with_mismatch"] == 2
    assert "'step': 0" in bad["problems"][0] and "'registered_S': 4" in bad["problems"][0] and "'fold_S': [2]" in bad["problems"][0]


R1_PLACEHOLDER = {"id": "r1", "prompt": [4, 6], "sampled": [8, 11, 504], "cap": 4}   # EOS at event 1, then vLLM async scheduling's placeholder row


@_needs_derived_s
@pytest.mark.parametrize("form", _S_FORMS)
def test_a_placeholder_row_after_the_verified_eos_is_not_activity_the_record_fails_g6_and_s_alone_would_not_see_it(memory, tmp_path, form):
    """With async scheduling vLLM computes ONE more step for a request after its EOS and
    samples a token there (504 on the pod).  That row is in the engine's sampler batch but it is not the Program's lag-0 activity
    (a_r = 2).  Contract: (1) GM-01 FAILS the record at G6 by name (execution past EOS, n_r 3 > a_r 2) so the S check never certifies
    it; (2) the S check ALONE reports no problem -- the pinned launch's S is the registered one whatever the batch, so
    only G6's activity pinning sees the row.  The placeholder rule is the PROGRAM'S declared LAG, which the record's
    attestation must agree with: this Program declares LAG 0 (the toy preamble attests async_scheduling False), so the record is a lag-0
    execution and the contract above stands; the LAG-1 Program on the attested-True reading of the same record is the next test."""
    gpd, oracle = TGM.assemble([TGM.R0, R1_PLACEHOLDER], tmp_path, lag=0)
    res = gm.check(gpd, oracle)
    assert res["verdict"] == "FAIL" and "G6" in res["failed_checks"], res["reasons"]
    g6 = TGM._check(res, "G6")["problems"]
    # the text is '... 1 more token(s) were served (n_r 3 > a_r 2)', '... (n_r 3 > a_r 2 + eos_lag_rows 0)', or names the two
    # lengths: '... where the Program's declared scheduler (Program lag 0 ...) computes LAG 0 after an EOS (executed_prefix_len 3 >
    # served_output_len 2 + LAG 0)' -- every form names the count
    assert any("r1: execution past EOS -- output event 1 is the declared EOS id 11 but 1 more " in p
               and ("(n_r 3 > a_r 2" in p or "(executed_prefix_len 3 > served_output_len 2 + LAG 0)" in p) for p in g6), g6
    r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
    assert r1["eos_verified"] is False and r1["eos_passed_at_event"] == 1
    assert r1["served_output_len"] == 2 and r1["executed_prefix_len"] == 3 and r1["cap_exhausted"] is False
    rec = gm.read_record(os.path.join(oracle, "capture", "log.jsonl.gz"))
    observed = {s: len(rec["steps"][s]["logits_rows"]) for s in rec["steps"]}
    activity = _program_activity(gpd, _transcripts(oracle), set(gpd["stop_rule"]["eos_token_ids"]))
    assert observed == {0: 2, 1: 2, 2: 2} and activity == {0: 2, 1: 2, 2: 1}       # the phantom row inflates step 2 only
    gsm = dict(gpd, sampler_geometry={"num_sms": NSM})
    # (2) the pinned launch's S is the registered one whatever batch the engine ran: the S check alone is satisfied, and only G6 sees the row
    launch_fold, launch_es = _topp_fold_for(observed, {s: S_REG for s in observed}, form)
    alone = SG.sampler_geometry_check(gsm, launch_fold, launch_es, rec["steps"], rec["sampler_batch"])
    assert alone["problems"] == [] and alone["info"]["live_to_S"] == {"live=2:S=4": 3}
    # and the SampleEnd's own batch at the placeholder step is the inflated one -- the record is self-consistent, only the Program disagrees
    assert rec["sampler_batch"].get(2) in (None, 2)


# this section regresses GM-01's G6 EOS lag rule; named skip without it
_needs_lag_rule = pytest.mark.skipif(not (hasattr(TGM, "_set_async_scheduling") and hasattr(TGM, "_driver_stop_at")),
                                     reason="GM-01 without the G6 EOS lag rule")


@_needs_derived_s
@_needs_lag_rule
@pytest.mark.parametrize("form", _S_FORMS)
def test_under_an_attested_async_scheduler_the_placeholder_row_is_declared_activity(memory, tmp_path, form):
    """The SAME placeholder record under the LAG-1 Program (the workload Program's declared specialization
    parameter), attesting async_scheduling True with the driver's word for an EOS stop (generated_token_ids ends at the EOS, finish_step
    = the EOS step, the lag sample reported nowhere): G6 PASSES and the lag row IS r1's declared activity at step 2 (active internal
    computation: VUs required, sampled token not served; the lifted Program's lag-1 live: live_s iff no EOS at an event <= s-2).
    served_output_len 2, executed_prefix_len 3.  The live count at step 2 is 2, and the registered S holds
    there as everywhere.  One recorded fact, consumed by Build and Match alike, no inference: the attestation
    flipped to False on the very same record FAILS by name (the LAG-0 header on the LAG-1 Program; G1 + G6) before any S is certified,
    and the unattested record is refused by name rather than read as lag 0 or lag 1."""
    gpd, oracle = TGM.assemble([TGM.R0, R1_PLACEHOLDER], tmp_path, lag=1)
    TGM._set_async_scheduling(oracle, True)
    TGM._driver_stop_at(oracle, "r1", n_answer=2, eos_step=1)
    res = gm.check(gpd, oracle)
    assert res["verdict"] == "PASS", res["reasons"]
    r1 = next(e for e in res["requests"] if e["request_id"] == "r1")
    assert r1["eos_verified"] is True and r1["answer_events"] == 2 and r1["lag_rows"] == 1 and r1["eos_lag_rows_accepted"] == 1
    assert r1["served_output_len"] == 2 and r1["executed_prefix_len"] == 3 and r1["cap_exhausted"] is False
    assert TGM._check(res, "G6")["stop_rule"]["eos_lag_rows"] == 1 and res["lag"]["declared"] == 1 and res["lag"]["agree"] is True
    rec = gm.read_record(os.path.join(oracle, "capture", "log.jsonl.gz"))
    eos = set(gpd["stop_rule"]["eos_token_ids"])
    observed = {s: len(rec["steps"][s]["logits_rows"]) for s in rec["steps"]}
    tr = _transcripts(oracle)
    assert tr["r1"] == [8, 11]                                    # the driver reports the answer only; the lag sample 504 is not an output
    lag0, lag1 = _program_activity(gpd, tr, eos, lag=0), _program_activity(gpd, tr, eos, lag=1)
    assert observed == lag1 == {0: 2, 1: 2, 2: 2} and lag0 == {0: 2, 1: 2, 2: 1}   # the placeholder row is activity under lag 1 only
    gsm = dict(gpd, sampler_geometry={"num_sms": NSM})
    ok_fold, ok_es = _topp_fold_for(observed, {s: S_REG for s in observed}, form)
    ok = SG.sampler_geometry_check(gsm, ok_fold, ok_es, rec["steps"], rec["sampler_batch"])
    assert ok["problems"] == [] and ok["info"]["checked_steps"] == 3 and ok["info"]["live_to_S"] == {"live=2:S=4": 3}
    # one recorded fact: the same record attesting False (a LAG-0 header) is not an execution of the LAG-1 Program -- FAIL by name (G1 + G6);
    # unattested it is refused by name, no default; the rule stays the Program's (eos_lag_rows 1), never re-read from the trace or the header
    TGM._set_async_scheduling(oracle, False)
    res_f = gm.check(gpd, oracle)
    assert res_f["verdict"] == "FAIL" and {"G1", "G6"} <= set(res_f["failed_checks"])
    assert any("the Program is specialized for LAG 1" in p and "record attests async_scheduling=False" in p and "-> LAG 0" in p
               for p in TGM._check(res_f, "G6")["problems"]), TGM._check(res_f, "G6")["problems"]
    assert TGM._check(res_f, "G6")["stop_rule"]["eos_lag_rows"] == 1 and res_f["lag"]["agree"] is False
    TGM._set_async_scheduling(oracle, None)
    res_n = gm.check(gpd, oracle)
    assert res_n["verdict"] == "FAIL" and any("does not attest async_scheduling" in p and "refuse by name" in p for p in TGM._check(res_n, "G6")["problems"])
    assert TGM._check(res_n, "G6")["stop_rule"]["eos_lag_rows"] == 1 and res_n["lag"]["record_attests"] is None
    # the S the declared-to-cap schedule would prescribe never enters: at step 3 (declared live 1) there is no row to check
    declared = gp.declared_live_counts([{"arrive_step": g["arrive_step"], "cap": g["max_tokens"]} for g in gpd["requests"]])
    assert declared == [2, 2, 2, 1] and 3 not in observed
