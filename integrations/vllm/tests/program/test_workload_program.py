"""The workload Program composes request Programs (as descriptors) hierarchically, with weights once, a deterministic
global address space and a declared chronology. Toy request Programs stand in for derive_step's request wrappers: the same
lowering (`Program(fn)` -> Input batches + body), the same descriptor codec."""

import pytest

from verity.ml.circuits.codec import SCHEMA_V1, _encode_body, _Tables, encode_program, program_digest
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, REGISTRY
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Record, Value
from verity_vllm.program import workload as W

_ADD = PrimitiveDefinition("GpTestAdd16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: (a + b) & 0xFFFF)


def _toy_request(LP: int, T: int):
    """`ToyRequest{LP,T}(prompt: Array<LP,16>, weights: Array<4,16>) -> Array<T+1,16>`: T+1 'tokens', each depending on the
    previous one and on a weight (a request-local state chain, like the decode loop)."""
    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("weights", Array(4, Value(16)))), Array(S.T + 1, Value(16)))

    def body(B, S, prompt, weights):
        acc = B.call(_ADD, prompt[0], weights[0])
        outs = [acc]
        for s in range(S.T):
            acc = B.call(_ADD, acc, weights[(s + 1) % 4])
            outs.append(acc)
        return coll_concat(Array(S.T + 1, Value(16)), outs)

    d = REGISTRY["ToyRequest_v1"] if "ToyRequest_v1" in REGISTRY else CompositeDefinition("ToyRequest", 1, ("LP", "T"), sig, body)
    return d.bind(LP=LP, T=T)


def _WP(LP):
    return [("prompt", f"Array<{LP},Value<16>>"), ("weights", "Array<4,Value<16>>")]


def _desc(LP, T):
    return encode_program(Program(_toy_request(LP, T)))


def _wl(shapes, arrivals=None, ids=None, **kw):
    arrivals = arrivals or [0] * len(shapes)
    reqs = []
    for i, ((LP, T), a) in enumerate(zip(shapes, arrivals)):
        c = W.request_component(_desc(LP, T), REGISTRY, _WP(LP))
        reqs.append(W.WorkloadRequest((ids or [f"r{i}" for i in range(len(shapes))])[i], i, LP, T + 1, a, c, T=T))
    return W.WorkloadProgram(reqs, manifest_digest="m" * 8, lag=0, **kw)


def test_component_reproduces_the_request_body():
    d = _desc(3, 2)
    c = W.request_component(d, REGISTRY, _WP(3))
    root_gates = sum(n.gates for n in Program(_toy_request(3, 2)).root.body.nodes)
    assert c.input_gates == 3 + 4 and c.body_gates == root_gates - c.input_gates == 3
    assert [n for n, _ in c.params] == ["prompt", "weights"]
    assert c.program_digest == program_digest(d)


def test_two_unequal_requests_weights_once_and_hierarchical():
    wl = _wl([(3, 2), (5, 0)])
    body = wl.program.root.body
    # root = [weights Input, r0.prompt Input, r1.prompt Input, call r0, call r1]
    forms = [n.form for n in body.nodes]
    assert forms == ["batch", "batch", "batch", "call", "call"]
    assert wl.program.input_gates == 4 + 3 + 5  # weights ONCE
    assert body.nodes[3].fn.gates == 3 and body.nodes[4].fn.gates == 1
    am = wl.address_map()
    assert am["inputs"][0]["owner"] == "shared" and am["inputs"][1]["owner"] == "r0" and am["inputs"][2]["owner"] == "r1"
    assert am["shared"] == {"group": "weights", "root_nodes": [0, 1], "gates": [0, 4], "n_batches": 1}
    r0, r1 = am["requests"]
    assert r0["gates"] == [12, 15] and r1["gates"] == [15, 16]
    assert r0["inputs"][1]["global"] == "shared.weights" and r1["inputs"][1]["gates"] == r0["inputs"][1]["gates"] == [0, 4]
    # digest commits to the request digests
    assert {r["program_digest"] for r in am["requests"]} == {program_digest(_desc(3, 2)), program_digest(_desc(5, 0))}


def test_digest_is_deterministic_and_order_sensitive():
    a = _wl([(3, 2), (5, 0)]).digest
    b = _wl([(3, 2), (5, 0)]).digest
    c = _wl([(5, 0), (3, 2)], ids=["r1", "r0"]).digest
    assert a == b and a != c
    # encode/decode round trip of the workload descriptor
    wl = _wl([(3, 2), (5, 0)])
    desc = encode_program(wl.program)
    assert program_digest(desc) == wl.digest
    assert desc["program_fn"].startswith("Workload_v1{")


def test_same_shape_twice_is_two_instances_of_one_specialization():
    wl = _wl([(3, 2), (3, 2)])
    body = wl.program.root.body
    assert body.nodes[3].fn is body.nodes[4].fn
    am = wl.address_map()
    assert am["requests"][0]["gates"] != am["requests"][1]["gates"]
    assert am["requests"][0]["specialization"] == am["requests"][1]["specialization"]


def test_refusals():
    with pytest.raises(ValueError, match="duplicate request_id"):
        _wl([(3, 2), (5, 0)], ids=["r0", "r0"])
    # T not cap-1: a Program rebuilt from a served length is refused
    c = W.request_component(_desc(3, 2), REGISTRY, _WP(3))
    with pytest.raises(ValueError, match="never from a served length"):
        W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 5, 0, c, T=2)], lag=0)
    # a shared group that differs across requests (here: sharing the unequal prompts) = not one prescribed input; refused
    with pytest.raises(ValueError, match="differs across requests"):
        ca = W.request_component(_desc(3, 2), REGISTRY, _WP(3))
        cb = W.request_component(_desc(5, 0), REGISTRY, _WP(5))
        W.WorkloadProgram([W.WorkloadRequest("a", 0, 3, 3, 0, ca), W.WorkloadRequest("b", 1, 5, 1, 0, cb)], shared="prompt", lag=0)
    with pytest.raises(ValueError, match="no prescribed parameter group"):
        W.WorkloadProgram([W.WorkloadRequest("a", 0, 3, 3, 0, ca)], shared="kv", lag=0)
    # the wrapper parameters must tile the Input batches exactly
    with pytest.raises(ValueError, match="tile|leaves"):
        W.request_component(_desc(3, 2), REGISTRY, [("prompt", "Array<2,Value<16>>"), ("weights", "Array<5,Value<16>>")])


def test_a_growing_key_value_sequence_composes_to_the_same_workload_through_its_log(monkeypatch):
    """Call i reads the rows of calls 0..i (the decoder's alias chain, one log): the component's arguments remapped through the
    log are the parts the plain remap gives, and its body encodes to the same descriptor entry."""
    def sum_sig(S):
        return ((("xs", Array(S.P, Value(16))),), Value(16))

    def sum_body(B, S, xs):
        acc = xs[0]
        for p in range(1, S.P):
            acc = B.call(_ADD, acc, xs[p])
        return acc

    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("weights", Array(4, Value(16)))), Array(S.LP, Value(16)))

    def body(B, S, prompt, weights):
        rows, outs = [], []
        for i in range(S.LP):
            rows.append(B.call(_ADD, prompt[i], weights[i % 4]))
            outs.append(B.call(prefix_sum.bind(P=i + 1), coll_concat(Array(i + 1, Value(16)), rows)))
        return coll_concat(Array(S.LP, Value(16)), outs)

    prefix_sum = REGISTRY["GpTestPrefixSum_v1"] if "GpTestPrefixSum_v1" in REGISTRY else CompositeDefinition(
        "GpTestPrefixSum", 1, ("P",), sum_sig, sum_body)
    req = REGISTRY["ToyKvRequest_v1"] if "ToyKvRequest_v1" in REGISTRY else CompositeDefinition("ToyKvRequest", 1, ("LP",), sig, body)
    desc = encode_program(Program(req.bind(LP=12)))

    def component():
        monkeypatch.setattr(W, "_REQUEST_CACHE", {})
        body = W.request_component(desc, REGISTRY, _WP(12)).fn.body
        return body, _encode_body(body, SCHEMA_V1, _Tables())

    body, enc = component()
    shared = [a.refs for n in body.nodes for a in n.args if getattr(a.refs, "prefix", None) is not None]
    assert len(shared) >= 10 and len({id(r.prefix[0]) for r in shared}) == 1
    plain = W._remap
    monkeypatch.setattr(W, "_remap", lambda r, K, logs=None: plain(r, K))
    body0, enc0 = component()
    assert [[a.refs.to_json() for a in n.args] for n in body.nodes] == [[a.refs.to_json() for a in n.args] for n in body0.nodes]
    assert enc == enc0


def test_record_weights_lower_to_several_input_batches_shared_as_one_group():
    """The real wrapper's `weights` is a mixed-width Record (bf16 tensors + an f32 rotary table), which lowers to one Input
    batch per field (a homogeneous record would be one batch): the group is shared as a whole either way."""
    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("weights", Record(("wa", Array(2, Value(16))), ("wb", Array(3, Value(32)))))),
                Array(1, Value(16)))

    def body(B, S, prompt, weights):
        x = B.call(_ADD, prompt[0], weights.field("wa")[1])
        return coll_concat(Array(1, Value(16)), [B.call(_ADD, x, weights.field("wa")[0])])

    d = REGISTRY["ToyRecReq_v1"] if "ToyRecReq_v1" in REGISTRY else CompositeDefinition("ToyRecReq", 1, ("LP",), sig, body)
    wp = lambda LP: [("prompt", f"Array<{LP},Value<16>>"), ("weights", "{wa: Array<2,Value<16>>, wb: Array<3,Value<32>>}")]
    ca = W.request_component(encode_program(Program(d.bind(LP=2))), REGISTRY, wp(2))
    cb = W.request_component(encode_program(Program(d.bind(LP=4))), REGISTRY, wp(4))
    assert ca.groups == ("prompt", "weights", "weights") and [n for n, _ in ca.params] == ["prompt", "weights[0]", "weights[1]"]
    wl = W.WorkloadProgram([W.WorkloadRequest("a", 0, 2, 1, 0, ca), W.WorkloadRequest("b", 1, 4, 1, 0, cb)], lag=0)
    assert wl.shared == (1, 2) and wl.program.input_gates == 2 + 3 + 2 + 4
    am = wl.address_map()
    assert am["shared"] == {"group": "weights", "root_nodes": [0, 2], "gates": [0, 5], "n_batches": 2}
    assert wl.theta == ("shared.weights[0]", "shared.weights[1]") and wl.descriptor["annotations"]["theta"] == [0, 1]
    assert [i["owner"] for i in am["inputs"]] == ["shared", "shared", "a", "b"]
    assert am["requests"][0]["inputs"][2]["gates"] == am["requests"][1]["inputs"][2]["gates"] == [2, 5]


def _toy_registered_request(LP: int, values_n: int = 2):
    """A registered_values derive's wrapper shape: `(prompt, values, weights)` with `values` bound at registration."""
    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("values", Array(S.V, Value(16))), ("weights", Array(4, Value(16)))),
                Array(1, Value(16)))

    def body(B, S, prompt, values, weights):
        return coll_concat(Array(1, Value(16)), [B.call(_ADD, B.call(_ADD, prompt[0], values[0]), weights[0])])

    d = REGISTRY["ToyRegReq_v1"] if "ToyRegReq_v1" in REGISTRY else CompositeDefinition("ToyRegReq", 1, ("LP", "V"), sig, body)
    return d.bind(LP=LP, V=values_n)


def _registered_component(LP: int, values_n: int = 2, registered=("values",)):
    wp = [("prompt", f"Array<{LP},Value<16>>"), ("values", f"Array<{values_n},Value<16>>"), ("weights", "Array<4,Value<16>>")]
    return W.request_component(encode_program(Program(_toy_registered_request(LP, values_n), registered=registered)), REGISTRY, wp)


def test_registered_values_are_shared_once_and_registered_at_the_root():
    """A registered_values component's `values` are the program's (bound once, at registration), not the request's: the workload
    Program holds them once as `shared.values`, a registered root parameter, beside `shared.weights` bound per run."""
    from verity.ml.circuits.constants import REGISTRATION, RUN, inputs
    from verity.ml.circuits.ports import facts
    ca, cb = _registered_component(2), _registered_component(5)
    assert ca.bindings == (RUN, REGISTRATION, RUN)
    wl = W.WorkloadProgram([W.WorkloadRequest("a", 0, 2, 1, 0, ca), W.WorkloadRequest("b", 1, 5, 1, 0, cb)], lag=0)
    assert wl.registered == ("shared.values",) and wl.shared == (1, 2)
    assert wl.program.input_gates == 2 + 4 + 2 + 5
    names = [n for n, _ in wl.fn.signature()[0]]
    assert dict(zip(names, wl.program.bindings)) == {"shared.values": REGISTRATION, "shared.weights": RUN, names[2]: RUN, names[3]: RUN}
    assert len(inputs(wl.program.circuit)[REGISTRATION]) == 2 and len(inputs(wl.program.circuit)[RUN]) == 4 + 2 + 5
    F = facts(wl.program.root)
    assert wl.theta == ("shared.weights",) and [set(F.node(k).one.origin) for k in range(4)] == [{"registered"}, {"theta"}, {"input"}, {"input"}]
    assert wl.artifact()["shared"]["registered"] == ["shared.values"]
    # the components agree on what they register, or the workload is not one program
    with pytest.raises(ValueError, match="register different groups"):
        W.WorkloadProgram([W.WorkloadRequest("a", 0, 2, 1, 0, ca),
                           W.WorkloadRequest("b", 1, 5, 1, 0, _registered_component(5, registered=()))], lag=0)


def test_a_workload_without_registered_values_keeps_its_digest():
    wl = _wl([(3, 2), (5, 0)])
    assert wl.registered == () and "registered" not in wl.artifact()["shared"]
    assert wl.descriptor["annotations"]["theta"] == [0]   # the weights, marked outside the digest
    assert wl.digest == "6e2b20e9356a27b5b3db628205d9b97f617a400cfa95c96fa7d9aff92b145de92892833688405e5e8ee4fa29cd4400395250238cc9764234376882bcb6904655"


def test_theta_registered_binds_the_weights_at_registration_and_moves_only_the_workload_digest():
    """The inference-only check reads only registered inputs as θ (nci-check's `theta-not-registered`): `theta_registered` lowers the
    weights as `RegisteredInput<w>_v1`, still θ, and every other input as before. The components are the same Definitions, so
    their request digests stand and only the workload digest moves; off, it is the digest pinned above."""
    from verity.ml.circuits.constants import REGISTRATION, RUN
    from verity.ml.circuits.ports import facts
    from verity.ml.circuits.program import INPUT_ID, REGISTERED_ID
    off, on = _wl([(3, 2), (5, 0)]), _wl([(3, 2), (5, 0)], theta_registered=True)
    assert off.digest == _wl([(3, 2), (5, 0)], theta_registered=False).digest
    assert on.registered == on.theta == ("shared.weights",) and on.digest != off.digest
    assert program_digest(encode_program(on.program)) == on.digest
    names = [n for n, _ in on.fn.signature()[0]]
    assert dict(zip(names, on.program.bindings)) == {"shared.weights": REGISTRATION, names[1]: RUN, names[2]: RUN}
    nodes = on.program.root.body.nodes
    assert REGISTERED_ID.fullmatch(nodes[0].fn.id) and INPUT_ID.fullmatch(nodes[1].fn.id) and INPUT_ID.fullmatch(nodes[2].fn.id)
    assert on.descriptor["annotations"]["theta"] == off.descriptor["annotations"]["theta"] == [0]
    F = facts(on.program.root)
    assert [set(F.node(k).one.origin) for k in range(3)] == [{"theta"}, {"input"}, {"input"}]
    assert [r["program_digest"] for r in on.address_map()["requests"]] == [r["program_digest"] for r in off.address_map()["requests"]]
    assert on.artifact()["shared"]["registered"] == ["shared.weights"] and on.artifact()["shared"]["theta_registered"] is True
    assert "registered" not in off.artifact()["shared"] and "theta_registered" not in off.artifact()["shared"]
    # beside registered values, the weights join them; the values stay registered and are no θ
    ca, cb = _registered_component(2), _registered_component(5)
    wl = W.WorkloadProgram([W.WorkloadRequest("a", 0, 2, 1, 0, ca), W.WorkloadRequest("b", 1, 5, 1, 0, cb)], lag=0, theta_registered=True)
    assert wl.registered == ("shared.values", "shared.weights") and wl.theta == ("shared.weights",)
    F = facts(wl.program.root)
    assert [set(F.node(k).one.origin) for k in range(2)] == [{"registered"}, {"theta"}]


def test_theta_registered_keeps_a_tp_rank_programs_peers_bound_per_run():
    """A TP rank workload (B1's shape: `peers_*` before the weights, at shifting positions): only the weights move to
    registration; every request's prompt and peers stay `Input<w>_v1`, the inputs the inference-only check counts against the limit."""
    from verity.ml.circuits.program import INPUT_ID, REGISTERED_ID
    ca = W.request_component(encode_program(Program(_toy_rank_request(3, 2))), REGISTRY, _WP_rank(3, 2))
    cb = W.request_component(encode_program(Program(_toy_rank_request(5, 0))), REGISTRY, _WP_rank(5, 0))
    reqs = [W.WorkloadRequest("r0", 0, 3, 3, 0, ca, T=2), W.WorkloadRequest("r1", 1, 5, 1, 0, cb, T=0)]
    off, on = W.WorkloadProgram(reqs, lag=0), W.WorkloadProgram(reqs, lag=0, theta_registered=True)
    assert on.registered == ("shared.weights",) and on.digest != off.digest and on.shared_by_request == off.shared_by_request
    ids = [n.fn.id for n in on.program.root.body.nodes if n.form == "batch"]
    assert REGISTERED_ID.fullmatch(ids[0]) and all(INPUT_ID.fullmatch(i) for i in ids[1:]) and len(ids) == 1 + 4 + 2
    assert [n.fn.id for n in off.program.root.body.nodes if n.form == "batch"][1:] == ids[1:]


def _toy_rank_request(LP: int, T: int):
    """A TP rank Program's shape: `(prompt, peers_0..peers_T, weights)` -- one received collective value per step, so
    the number of request-local inputs BEFORE the shared group depends on T and the weights group shifts per component."""
    def sig(S):
        peers = tuple((f"peers_{k}", Array(1, Value(16))) for k in range(S.T + 1))
        return ((("prompt", Array(S.LP, Value(16))),) + peers + (("weights", Array(4, Value(16))),), Array(S.T + 1, Value(16)))

    def body(B, S, prompt, *rest):
        peers, weights = rest[:-1], rest[-1]
        acc = B.call(_ADD, prompt[0], weights[0])
        outs = [B.call(_ADD, acc, peers[0][0])]
        for s in range(S.T):
            acc = B.call(_ADD, outs[-1], weights[(s + 1) % 4])
            outs.append(B.call(_ADD, acc, peers[s + 1][0]))
        return coll_concat(Array(S.T + 1, Value(16)), outs)

    d = REGISTRY["ToyRankReq_v1"] if "ToyRankReq_v1" in REGISTRY else CompositeDefinition("ToyRankReq", 1, ("LP", "T"), sig, body)
    return d.bind(LP=LP, T=T)


def _WP_rank(LP, T):
    return [("prompt", f"Array<{LP},Value<16>>")] + [(f"peers_{k}", "Array<1,Value<16>>") for k in range(T + 1)] + [("weights", "Array<4,Value<16>>")]


def test_tp_rank_components_share_weights_at_shifting_positions_and_tp1_digests_are_unchanged():
    """Request-local `peers_*` inputs before the shared group shift its positions per component; the shared
    group is still the same Input batches once, each request's peers are its own inputs; a non-request-local group at a shifting
    position is still refused. The uniform (TP1) static form is untouched, so every existing workload digest stands."""
    ca = W.request_component(encode_program(Program(_toy_rank_request(3, 2))), REGISTRY, _WP_rank(3, 2))   # weights at 4
    cb = W.request_component(encode_program(Program(_toy_rank_request(5, 0))), REGISTRY, _WP_rank(5, 0))   # weights at 2
    assert ca.groups.index("weights") == 4 and cb.groups.index("weights") == 2
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, ca, T=2), W.WorkloadRequest("r1", 1, 5, 1, 0, cb, T=0)], lag=0)
    assert wl.shared == (4,) and wl.shared_by_request == ((4,), (2,)) and wl.shared_positions_uniform is False
    body = wl.program.root.body
    # root = [weights ONCE, r0.prompt, r0.peers_0..2, r1.prompt, r1.peers_0, call r0, call r1]
    assert [n.form for n in body.nodes] == ["batch"] * (1 + 4 + 2) + ["call", "call"]
    assert wl.program.input_gates == 4 + (3 + 3) + (5 + 1)
    params, _ = wl.fn.signature()
    assert [p for p, _ in params] == ["shared.weights", "request[r0].prompt", "request[r0].peers_0", "request[r0].peers_1", "request[r0].peers_2",
                                      "request[r1].prompt", "request[r1].peers_0"]
    am = wl.address_map()
    r0, r1 = am["requests"]
    assert r0["inputs"][4]["global"] == r1["inputs"][2]["global"] == "shared.weights" and r0["inputs"][4]["gates"] == r1["inputs"][2]["gates"] == [0, 4]
    assert r0["inputs"][1]["global"] == "request[r0].peers_0" and r1["inputs"][1]["global"] == "request[r1].peers_0"
    assert am["shared"]["component_params_by_request"] == [[4], [2]]
    assert wl.artifact()["shared_param_index_by_request"] == [[4], [2]]
    # round trip of the descriptor
    assert program_digest(encode_program(wl.program)) == wl.digest
    # the same rank components in the other order compose to another digest (manifest order is part of the identity)
    wl2 = W.WorkloadProgram([W.WorkloadRequest("r1", 0, 5, 1, 0, cb, T=0), W.WorkloadRequest("r0", 1, 3, 3, 0, ca, T=2)], lag=0)
    assert wl2.digest != wl.digest
    # a foreign (non-request-local) group at a shifting position is still refused
    d = REGISTRY["ToyRankReq_v1"]
    cx = W.request_component(encode_program(Program(d.bind(LP=3, T=2))), REGISTRY,
                             [("prompt", "Array<3,Value<16>>")] + [(f"aux_{k}", "Array<1,Value<16>>") for k in range(3)] + [("weights", "Array<4,Value<16>>")])
    with pytest.raises(ValueError, match="different parameter positions .* non-request-local groups \\['aux_0'"):
        W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, cx, T=2), W.WorkloadRequest("r1", 1, 5, 1, 0, cb, T=0)], lag=0)
    # TP1 (uniform positions): the static keeps its one-tuple form -> the digest of the historical composition is unchanged
    u = _wl([(3, 2), (5, 0)])
    assert u.shared_positions_uniform is True and u.fn.statics["shared"] == (1,)
    assert "shared_param_index_by_request" in u.artifact() and u.artifact()["shared_param_index_by_request"] is None
    assert u.digest == W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, W.request_component(_desc(3, 2), REGISTRY, _WP(3)), T=2),
                                          W.WorkloadRequest("r1", 1, 5, 1, 0, W.request_component(_desc(5, 0), REGISTRY, _WP(5)), T=0)],
                                         manifest_digest="m" * 8, lag=0).digest


def _toy_stochastic_rank_request(LP: int, T: int):
    """A stochastic TP rank Program's shape (`vllm_meta.SplitsWithPeers`): `(prompt, seed, splits, peers_0..peers_T, weights)`."""
    def sig(S):
        peers = tuple((f"peers_{k}", Array(1, Value(16))) for k in range(S.T + 1))
        return ((("prompt", Array(S.LP, Value(16))), ("seed", Array(1, Value(64))), ("splits", Array(S.T + 1, Value(32))))
                + peers + (("weights", Array(4, Value(16))),), Array(S.T + 1, Value(16)))

    def body(B, S, prompt, seed, splits, *rest):
        peers, weights = rest[:-1], rest[-1]
        outs, acc = [], prompt[0]
        for s in range(S.T + 1):
            acc = B.call(_ADD, B.call(_ADD, acc, weights[s % 4]), peers[s][0])
            outs.append(acc)
        return coll_concat(Array(S.T + 1, Value(16)), outs)

    d = REGISTRY["ToyStochRankReq_v1"] if "ToyStochRankReq_v1" in REGISTRY else CompositeDefinition("ToyStochRankReq", 1, ("LP", "T"), sig, body)
    return d.bind(LP=LP, T=T)


def _WP_stochastic_rank(LP, T):
    return ([("prompt", f"Array<{LP},Value<16>>"), ("seed", "Array<1,Value<64>>"), ("splits", f"Array<{T + 1},Value<32>>")]
            + [(f"peers_{k}", "Array<1,Value<16>>") for k in range(T + 1)] + [("weights", "Array<4,Value<16>>")])


def test_stochastic_tp_rank_components_keep_their_seed_and_splits_as_request_inputs():
    """A stochastic rank Program's `seed` and `splits` are its request's own, like the prompt and the peers: B8 rank components of
    different T compose with the weights once at shifting positions, each request's seed and splits its own inputs."""
    ca = W.request_component(encode_program(Program(_toy_stochastic_rank_request(3, 2))), REGISTRY, _WP_stochastic_rank(3, 2))   # weights at 6
    cb = W.request_component(encode_program(Program(_toy_stochastic_rank_request(5, 0))), REGISTRY, _WP_stochastic_rank(5, 0))  # weights at 4
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, ca, T=2), W.WorkloadRequest("r1", 1, 5, 1, 0, cb, T=0)], lag=0)
    assert wl.shared_by_request == ((6,), (4,)) and wl.shared_positions_uniform is False
    params, _ = wl.fn.signature()
    assert [p for p, _ in params] == ["shared.weights", "request[r0].prompt", "request[r0].seed", "request[r0].splits", "request[r0].peers_0",
                                      "request[r0].peers_1", "request[r0].peers_2", "request[r1].prompt", "request[r1].seed", "request[r1].splits",
                                      "request[r1].peers_0"]
    am = wl.address_map()
    assert am["requests"][0]["inputs"][1]["global"] == "request[r0].seed" and am["requests"][1]["inputs"][2]["global"] == "request[r1].splits"
    assert program_digest(encode_program(wl.program)) == wl.digest


def test_chronology_is_declared_and_bounded():
    wl = _wl([(3, 2), (5, 0), (2, 3)], arrivals=[0, 1, 3])
    ch = wl.chronology()
    assert ch["time_unit"].startswith("engine step")
    rows = {r["request_id"]: r for r in ch["requests"]}
    assert rows["r0"]["programmed_steps"] == {"prefill": 0, "decode": [1, 2], "count": 3}
    assert rows["r1"]["programmed_steps"] == {"prefill": 1, "decode": None, "count": 1}
    assert rows["r2"]["programmed_steps"] == {"prefill": 3, "decode": [4, 6], "count": 4}
    assert rows["r2"]["output_interface"]["tokens_max"] == 4
    assert ch["programmed_horizon_steps"] == 7
    # the programmed instance set of every engine step is defined by the declaration alone (M-ARRIVAL-ORDER lives here)
    es = {e["e"]: e for e in ch["engine_steps"]}
    assert [m["request_id"] for m in es[0]["members"]] == ["r0"] and es[0]["members"][0]["phase"] == "prefill"
    assert [(m["request_id"], m["s"], m["phase"]) for m in es[1]["members"]] == [("r0", 1, "decode"), ("r1", 0, "prefill")]
    assert es[1]["co_scheduled"] and es[1]["mixed_prefill_decode"]
    assert [(m["request_id"], m["s"]) for m in es[3]["members"]] == [("r2", 0)] and [(m["request_id"], m["s"]) for m in es[6]["members"]] == [("r2", 3)]
    # segments: the toy has no TokenSelect, so one trailing segment (kept, never dropped); the real wrapper has cap segments
    assert ch["component_steps_equal_cap"] is False
    m0 = es[0]["members"][0]
    assert m0["nodes"] == [0, 3] and m0["gates"] == [wl.address_map()["requests"][0]["gates"][0], wl.address_map()["requests"][0]["gates"][1]]
    art = wl.artifact()
    assert art["padding_scope"].startswith("active-reference")
    assert "observed" in ch["observed_time"]


def test_population_partitions_the_gates():
    wl = _wl([(3, 2), (5, 0), (3, 2)])
    p = wl.population()
    assert p["other_root_nodes"] == 0 and p["sum_equals_total"] and p["components_partition_non_input_gates"]
    assert p["component_intervals"] == [(15, 18), (18, 19), (19, 22)]
    assert p["input_gates"] == 4 + 3 + 5 + 3 and p["component_gates"] == 3 + 1 + 3 == p["gates"] - p["input_gates"]


def _stepped_request(LP: int, T: int, step_name: str = "GpTestStep"):
    """Like `_toy_request` but every token step calls a shared composite `GpTestStep_v1` (a stand-in for a layer definition shared
    by every request Program): the descriptor carries that definition beside the root, so two descriptors can disagree on it."""
    def ssig(S):
        return ((("x", Value(16)), ("w", Value(16))), Value(16))

    def sbody(B, S, x, w):
        return B.call(_ADD, x, w)

    step = REGISTRY[f"{step_name}_v1"] if f"{step_name}_v1" in REGISTRY else CompositeDefinition(step_name, 1, (), ssig, sbody)

    def sig(S):
        return ((("prompt", Array(S.LP, Value(16))), ("weights", Array(4, Value(16)))), Array(S.T + 1, Value(16)))

    def body(B, S, prompt, weights):
        acc = B.call(step, prompt[0], weights[0])
        outs = [acc]
        for s in range(S.T):
            acc = B.call(step, acc, weights[(s + 1) % 4])
            outs.append(acc)
        return coll_concat(Array(S.T + 1, Value(16)), outs)

    d = REGISTRY["ToyStepped_v1"] if "ToyStepped_v1" in REGISTRY else CompositeDefinition("ToyStepped", 1, ("LP", "T"), sig, body)
    return d.bind(LP=LP, T=T)


def test_canon_is_per_compose_conflicts_refused_inside_one_never_across_composes():
    """The shared-definition canon belongs to ONE compose: two request Programs of one workload that
    disagree on a definition under the same id are refused (E_IDENTITY, fail closed); a LATER compose in the same process with its own
    canon is compared against its own components only, never against an earlier compose's definitions."""
    import copy
    from verity.ml.circuits.defs import IdentityConflict
    da = encode_program(Program(_stepped_request(3, 2)))
    sid = next(k for k, v in da["definitions"].items() if v["form"] == "composite" and k.startswith("GpTestStep"))
    db = copy.deepcopy(da)
    node = db["definitions"][sid]["body"]["nodes"][-1]
    node["args"] = [node["args"][1], node["args"][0]]          # operands swapped: a different body under the same definition id
    wp = _WP(3)
    # one compose, both components -> the disagreement is refused
    canon1: dict = {}
    W.request_component(da, REGISTRY, wp, canon=canon1)
    with pytest.raises(IdentityConflict, match="E_IDENTITY"):
        W.request_component(db, REGISTRY, wp, canon=canon1)
    # a separate compose in the same process: its own canon, the earlier compose's definitions are not consulted
    canon2: dict = {}
    cb = W.request_component(db, REGISTRY, wp, canon=canon2)
    assert cb.program_digest == program_digest(db) != program_digest(da)
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, cb, T=2)], manifest_digest="m" * 8, lag=0)
    assert wl.digest
    # and the canon that already holds `da`'s definition still refuses `db` (fail-closed direction unchanged)
    with pytest.raises(IdentityConflict):
        W.request_component(db, REGISTRY, wp, canon=canon1)


def test_repointing_keeps_every_id_and_refuses_a_substitute_of_another_id():
    """`_repoint` swaps a callee and a function-valued static for its canonical object only when that object has the same id:
    a complete specialization formats its id once, so `Outer{F=..}`'s would go stale under a substitute of another id. Such
    a substitute is refused (E_IDENTITY, naming both ids) before anything is re-pointed, through the callee or the static."""
    import re
    from verity.ml.circuits.codec import _spec_id
    from verity.ml.circuits.defs import IdentityConflict, SpecializedDefinition

    def sig(S):
        return (("a", Value(16)),), Value(16)

    inner = CompositeDefinition("GpTestRepointInner", 1, ("K",), sig, lambda B, S, a: B.call(_ADD, a, a), register=False)
    outer = CompositeDefinition("GpTestRepointOuter", 1, ("F",), sig, lambda B, S, a: B.call(S.F, a), register=False)
    holder = CompositeDefinition("GpTestRepointHolder", 1, ("OPS",), sig, lambda B, S, a: B.call(_ADD, a, a), register=False)
    f, same, other = inner.bind(K=4), SpecializedDefinition(inner, {"K": 4}, ()), inner.bind(K=8)
    assert same is not f and _spec_id(same) == _spec_id(f) != _spec_id(other)

    o, h = SpecializedDefinition(outer, {"F": f}, ()), SpecializedDefinition(holder, {"OPS": {"f": f}}, ())
    before = [(fn.id, _spec_id(fn), [_spec_id(n.fn) for n in fn.body.nodes]) for fn in (o, h)]
    W._repoint(o, {_spec_id(f): same})
    W._repoint(h, {_spec_id(f): same})
    assert o.bindings["F"] is o.body.nodes[-1].fn is same and h.bindings["OPS"]["f"] is same
    assert [(fn.id, _spec_id(fn), [_spec_id(n.fn) for n in fn.body.nodes]) for fn in (o, h)] == before
    assert (o.id, h.id) == (SpecializedDefinition(outer, dict(o.bindings), ()).id,
                            SpecializedDefinition(holder, dict(h.bindings), ()).id)

    why = re.escape(f"{_spec_id(f)!r} would be re-pointed to a definition of another id, {_spec_id(other)!r}")
    for fn, held, was in ((SpecializedDefinition(outer, {"F": f}, ()), lambda fn: fn.bindings["F"], before[0][0]),
                          (SpecializedDefinition(holder, {"OPS": {"f": f}}, ()), lambda fn: fn.bindings["OPS"]["f"], before[1][0])):
        assert fn.body.nodes and fn.id == was
        with pytest.raises(IdentityConflict, match="E_IDENTITY: " + why):
            W._repoint(fn, {_spec_id(f): other})
        assert held(fn) is f and fn.id == was


def _derived_like(LP: int, T: int, src: dict):
    """derive.py's shape of a request Program: `Derived[...]_v1{SRC=...}` whose SRC static names the class / mode / profile / INPUT
    SHAPES / config -- the prompt length is an input shape, the decode-step count T is NOT -- and whose body is the recorded flat body.
    Two derives of one model at the same LP and different T therefore carry the SAME definition id over DIFFERENT bodies."""
    def sig(S):
        return ((("prompt", Array(LP, Value(16))), ("weights", Array(4, Value(16)))), Array(T + 1, Value(16)))

    def body(B, S, prompt, weights):
        acc = B.call(_ADD, prompt[0], weights[0])
        outs = [acc]
        for s in range(T):
            acc = B.call(_ADD, acc, weights[(s + 1) % 4])
            outs.append(acc)
        return coll_concat(Array(T + 1, Value(16)), outs)

    return CompositeDefinition("Derived[ToyRequest]", 1, ("SRC",), sig, body, register=False).bind(SRC=src)


def test_two_requests_with_one_root_id_over_different_bodies_compose_by_digest():
    """A manifest with two requests of the same prompt length and different caps (LP11: T24 and T36, as in the llama B64 1024/128
    workload) derives two request Programs whose roots share derive's id `Root[Derived[vLLMRequest[..]]_v1{SRC=..}]` (SRC does not
    name T) with different bodies and different digests; comparing the ROOTS under that id would refuse the workload
    (E_IDENTITY). The root is never shared -- the component's identity is its program digest -- so it is exempt from the shared-definition
    canon; every callee stays under the fail-closed rule (the test above)."""
    src = {"class": "LlamaForCausalLM", "mode": "profile", "inputs": [[3]], "config_sha256": "c" * 16}
    d24, d36 = encode_program(Program(_derived_like(3, 2, src))), encode_program(Program(_derived_like(3, 3, src)))
    assert d24["root"] == d36["root"] and d24["root"].startswith("Root[Derived[ToyRequest]_v1{SRC=")
    assert d24["definitions"][d24["root"]] != d36["definitions"][d36["root"]] and program_digest(d24) != program_digest(d36)
    canon: dict = {}
    c24 = W.request_component(d24, REGISTRY, _WP(3), canon=canon)
    c36 = W.request_component(d36, REGISTRY, _WP(3), canon=canon)          # was: IdentityConflict on the root id
    assert c24.program_digest == program_digest(d24) and c36.program_digest == program_digest(d36) and c24.fn is not c36.fn
    assert d24["root"] not in canon                                          # the root is not a shared definition
    wl = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, c24, T=2), W.WorkloadRequest("r1", 1, 3, 4, 0, c36, T=3)], manifest_digest="m" * 8, lag=0)
    am = wl.address_map()
    assert [r["program_digest"] for r in am["requests"]] == [program_digest(d24), program_digest(d36)]
    assert am["requests"][0]["specialization"] != am["requests"][1]["specialization"]
    assert wl.program.root.body.nodes[-2].fn.gates == 3 and wl.program.root.body.nodes[-1].fn.gates == 4   # T+1 tokens each
    # the same two shapes with DISTINCT root ids compose to the same workload digest: the root id is not part of the workload identity
    d24b = encode_program(Program(_derived_like(3, 2, {**src, "T": 2}))); d36b = encode_program(Program(_derived_like(3, 3, {**src, "T": 3})))
    assert d24b["root"] != d36b["root"]
    c24b, c36b = W.request_component(d24b, REGISTRY, _WP(3), canon={}), W.request_component(d36b, REGISTRY, _WP(3), canon={})
    wlb = W.WorkloadProgram([W.WorkloadRequest("r0", 0, 3, 3, 0, c24b, T=2), W.WorkloadRequest("r1", 1, 3, 4, 0, c36b, T=3)], manifest_digest="m" * 8, lag=0)
    assert (wlb.digest == wl.digest) == (program_digest(d24b) == program_digest(d24))   # digests follow the descriptors, nothing else
