"""verity_vllm.program.prune: a body's dead units (the nodes no output reads) and the body without them, each reference renumbered
in its own form; on one rank, the request wrapper's dead units are the last layer's rows before the prefill's last, its
RMSNormTriton and TokenSelect Calls are at their versions without dead gates, and those rows' fused qkv is narrowed to their keys
and values (`dead_free`)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from verity_vllm.program.predict import build as PB
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program import prune as PR
from verity_vllm.program.prune import _read, _renumbered, dead_free, dead_units, kept_records, narrowed, swapped, without_dead_units
from verity.ml.definitions.registry import b1, gemm_bias, topp_words as TW

from verity.ml.circuits.defs import Body, Node, bind
from verity.ml.circuits.evaluation.reference import evaluate
from verity.ml.circuits.liveness import dead_gates
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops.prims import BF16
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Strided
from verity.ml.circuits.types import Array

TINY = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
        "num_key_value_heads": 2, "num_hidden_layers": 2, "vocab_size": 256, "rms_norm_eps": 1e-5, "tie_word_embeddings": True,
        "max_position_embeddings": 64, "hidden_act": "silu"}
PHI = Phi(model={"repo": "org/m", "revision": "r", "hf_config": TINY}, target={"compute_capability": [12, 0], "num_sms": 188})


def _shape(body) -> list:
    return [(n.fn.id, [a.refs.to_json() for a in n.args]) for n in body.nodes] + [body.ret.to_json()]


def test_the_request_wrappers_dead_units_are_the_last_layers_rows_before_the_prefills_last(monkeypatch):
    """TINY's request of 8 prompt rows: the last layer's attention, o projection, MLP and their norms on rows 0-6 (and the rotary
    of their queries) reach no sample; the build drops those 56 units and nothing else, as pruning the unpruned body does, swaps
    the rest's Definitions with dead gates (`swapped`) and narrows the linears read in part (`narrowed`)."""
    pruned = PB.build_program(PHI, "request", LP=8, T=3)
    monkeypatch.setattr(PB, "dead_free_form", lambda b: (b, list(range(len(b.nodes)))))
    full = PB.build_program(PHI, "request", LP=8, T=3)
    body = full.program.fn.body
    dead = dead_units(body)
    assert Counter(body.nodes[k].fn.id.split("{")[0] for k in dead) == {"Gemm_v2": 21, "RMSNormFusedCuda_v2": 14, "RoPE_v1": 7,
                                                                         "Attention_v5": 7, "SiluMul_v2": 7}
    again, kept = without_dead_units(body)
    assert kept == [k for k in range(len(body.nodes)) if k not in set(dead)]
    assert again.offsets[-1] == body.offsets[-1] - sum(body.nodes[k].gates for k in dead)
    form, form_kept = dead_free(body)
    assert form_kept == kept and _shape(form) == _shape(narrowed(swapped(again))[0]) == _shape(pruned.program.fn.body)
    assert dead_units(pruned.program.fn.body) == [] and dead_free(pruned.program.fn.body)[0] is pruned.program.fn.body
    assert pruned.digest != full.digest
    step = PB.build_program(PHI, "step", M=1, pos0=8)
    assert dead_units(step.program.fn.body) == []


def test_the_dead_free_form_binds_rmsnorm_and_token_select_to_their_versions_without_dead_gates():
    """`swapped` binds each RMSNormTriton_v1 and TokenSelect_v1 node to v5 and v4 at the same statics (`registry.dead_free`): the
    same arguments and outputs, none of their gates dead, the RMSNorm's fewer (TokenSelect_v4 has a negation more a lane than v1, and
    lowers linearly in V where v1 lowers quadratically); FA2's attention goes to v18, here (9 and 10 keys) with v5's gates; every
    other node is the same object. The step wrapper, which isn't the registered form, keeps the v1s."""
    body = PB.build_program(PHI, "step", M=2, pos0=8).program.fn.body
    sw = swapped(body)
    assert Counter(n.fn.id for n in body.nodes if n.fn.name in ("RMSNormTriton", "TokenSelect")) == {
        "RMSNormTriton_v1{N=64,EPS=1e-05}": 2, "TokenSelect_v1{V=256}": 1}
    assert Counter(n.fn.id for n in sw.nodes if n.fn.name in ("RMSNormTriton", "TokenSelect")) == {
        "RMSNormTriton_v5{N=64,EPS=1e-05}": 2, "TokenSelect_v4{V=256}": 1}
    assert Counter(n.fn.base_id for n in sw.nodes if n.fn.name == "Attention") == {"Attention_v18": 4}
    for a, b in zip(body.nodes, sw.nodes):
        if a.fn.name in ("RMSNormTriton", "TokenSelect"):
            assert (b.form, b.args, b.out, b.n, b.fn.bindings) == (a.form, a.args, a.out, a.n, a.fn.bindings)
            assert b.gates < a.gates or a.fn.name == "TokenSelect"
            assert dead_gates(Program(b.fn), limit=0, transitive=True).computed == 0 < dead_gates(Program(a.fn), limit=0, transitive=True).computed
        elif a.fn.name == "Attention":
            assert (b.form, b.args, b.out, b.n, b.fn.bindings, b.gates) == (a.form, a.args, a.out, a.n, a.fn.bindings, a.gates)
        else:
            assert b is a
    assert sw.ret is body.ret and sw.offsets[-1] == sum(n.gates for n in sw.nodes)
    assert swapped(sw) is sw


def test_the_dead_free_form_binds_the_top_p_select_to_v3():
    """A request at the registered split count selects with `GumbelTopPTokenSelect_v2{V,S}` (`registry.topp_words`); `swapped`
    binds it to v3 at the same V and S, with v2's arguments and output and none of its dead gates."""
    sd = bind(TW.GumbelTopPTokenSelectV2, V=3, S=2)
    args = tuple(Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(sd.body.params))
    body = Body(sd.body.params, sd.body.ret_type, [Node("call", sd, args, sd.body.ret_type)], Affine("n", 0, 0, 1), [0, sd.gates])
    (a,), (b,) = body.nodes, swapped(body).nodes
    assert (a.fn.id, b.fn.id) == ("GumbelTopPTokenSelect_v2{V=3,S=2}", "GumbelTopPTokenSelect_v3{V=3,S=2}")
    assert (b.form, b.args, b.out) == (a.form, a.args, a.out)
    assert dead_gates(Program(b.fn), limit=0, transitive=True).computed == 0 < dead_gates(Program(a.fn), limit=0, transitive=True).computed


def test_the_dead_free_form_binds_fa2_attention_to_v18():
    """FA2's attention on compute capability 12.0 is `Attention_v5` (head dim 16: kBlockN 128); `swapped` binds it to v18 at the same
    statics. On TINY's 16-row prefill the row with 16 keys is the one whose v5 has dead gates, its first block's unread zero padding,
    2 a head; v18 has none of them and v5's gates on the other rows."""
    body = PB.build_program(PHI, "step", M=16, pos0=0).program.fn.body
    pairs = [(a, b) for a, b in zip(body.nodes, swapped(body).nodes) if a.fn.name == "Attention"]
    assert len(pairs) == 2 * 16 and {(a.fn.base_id, b.fn.base_id) for a, b in pairs} == {("Attention_v5", "Attention_v18")}
    for a, b in pairs:
        dead = dead_gates(Program(a.fn), limit=0, transitive=True).computed
        assert dead == (2 * TINY["num_attention_heads"] if a.fn.bindings["T"] == 16 else 0)
        assert (b.form, b.args, b.out, b.n, b.fn.bindings, b.gates) == (a.form, a.args, a.out, a.n, a.fn.bindings, a.gates - dead)
        assert dead_gates(Program(b.fn), limit=0, transitive=True).computed == 0


def test_each_reference_is_renumbered_in_its_own_form():
    new = [0, 1, 1, 3, 2]  # nodes 1 and 3 dead: 0, 2, 4 become 0, 1, 2
    memo: dict = {}
    a, p, empty = Affine("n", 2, 5, 3), Affine("p", 2, 0, 3), Affine("n", 3, 0, 0)
    s = Strided("n", 4, 1, 2, 8, 1, 6)
    e = Explicit((("n", 4, 0), ("p", 4, 1), ("n", 2, 3)))
    c = Concat([Affine("n", 0, 0, 2), Concat([s, e])])
    got = {k: _renumbered(r, new, memo) for k, r in {"a": a, "p": p, "empty": empty, "s": s, "e": e, "c": c}.items()}
    assert got["p"] is p and got["empty"] is empty
    assert got["a"].to_json() == Affine("n", 1, 5, 3).to_json()
    assert type(got["s"]) is Strided and got["s"].to_json() == Strided("n", 2, 1, 2, 8, 1, 6).to_json()
    assert got["e"].to_json() == Explicit((("n", 2, 0), ("p", 4, 1), ("n", 1, 3))).to_json()
    assert type(got["c"]) is Concat and type(got["c"].parts[1]) is Concat and got["c"].parts[0] is c.parts[0]
    assert got["c"].parts[1].parts == [got["s"], got["e"]]
    unchanged = Concat([Affine("n", 0, 0, 2), p])
    assert _renumbered(unchanged, new, memo) is unchanged
    read: set[int] = set()
    _read(c, read)
    _read(empty, read)
    assert read == {0, 2, 4}


DOT = G.GEMM_HOPPER["DOT"]


def _bf16(rng, n: int) -> list[int]:
    return ((rng.standard_normal(n) * 0.2).astype(np.float32).view(np.uint32) >> 16).astype(np.int64).tolist()


def _linears() -> Body:
    """x[16], W[6][16], b[6] -> Gemm_v2 (words 2-3 read), GemmBias_v2 (words 1, 2, 4, 5 read, through a Strided and an Explicit)
    and Gemm_v1 (read whole)."""
    fns = [bind(G.Gemm, K=16, N=6, DOT=DOT), bind(gemm_bias.GemmBiasV2, K=16, N=6, DOT=DOT), bind(b1.Gemm, K=16, N=6)]
    params = tuple(fns[1].body.params)
    nodes = [Node("call", fn, tuple(Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(fn.body.params)), fn.body.ret_type)
             for fn in fns]
    ret = Concat([Affine("n", 0, 2, 2), Strided("n", 1, 1, 2, 3, 1, 4), Explicit((("n", 0, 3), ("n", 1, 1))), Affine("n", 2, 0, 6)])
    offsets = [0]
    for n in nodes:
        offsets.append(offsets[-1] + n.gates)
    return Body(params, Array(ret.count, BF16), nodes, ret, offsets)


def test_each_columns_linear_is_one_coordinate_batched_over_its_column_arguments():
    """`narrowed` is exact by construction: each `COLUMNS` Definition is one batch over its output words, its column arguments
    mapped on axis 0 and the rest shared, so word j of `Name{N}(x, W, ...)` is word j - a of `Name{N=b-a}(x, W[a:b], ...)`."""
    fns = [bind(b1.Gemm, K=16, N=3), bind(G.Gemm, K=16, N=3, DOT=DOT), bind(gemm_bias.GemmBias, K=16, N=3),
           bind(gemm_bias.GemmBiasV2, K=16, N=3, DOT=DOT)]
    assert sorted(fn.base_id for fn in fns) == sorted(PR.COLUMNS)
    for fn in fns:
        (n,) = fn.body.nodes
        cols = PR.COLUMNS[fn.base_id]
        assert n.form == "batch" and n.n == 3 and n.axes == tuple(0 if i in cols else None for i in range(len(n.args)))
        assert [a.refs.to_json() for a in n.args] == [Affine("p", i, 0, t.leaves).to_json() for i, (_, t) in enumerate(fn.body.params)]
        assert list(fn.body.ret) == [("n", 0, j) for j in range(3)]


def test_a_linear_read_in_part_is_narrowed_to_the_columns_read():
    """Each Gemm read in part is bound at the words read over its weight's (and bias's) rows, computes those words as the whole
    Gemm does, and every reference to it, in each form, reads the same word; a Gemm read whole is the same node."""
    body = _linears()
    form, cols = narrowed(body)
    assert cols == {0: (2, 4), 1: (1, 6)} and form.nodes[2] is body.nodes[2]
    assert [n.fn.id for n in form.nodes[:2]] == [f"Gemm_v2{{K=16,N=2,DOT={DOT.id}}}", f"GemmBias_v2{{K=16,N=5,DOT={DOT.id}}}"]
    assert type(form.ret.parts[1]) is Strided and type(form.ret.parts[2]) is Explicit and form.ret.parts[3] is body.ret.parts[3]
    assert list(form.ret) == [(s, i, j - cols.get(i, (0,))[0]) for s, i, j in body.ret]
    assert form.offsets[-1] == sum(n.gates for n in form.nodes) < body.offsets[-1] and narrowed(form) == (form, {})
    rng = np.random.default_rng(3)
    vals = [_bf16(rng, t.leaves) for _, t in body.params]
    for k, (a, b) in cols.items():
        old, new = body.nodes[k], form.nodes[k]
        run = lambda n: evaluate(n.fn, *([vals[i][j] for _, i, j in c.refs] for c in n.args))  # noqa: E731
        assert list(run(new)) == list(run(old))[a:b]


def test_a_narrowed_nodes_call_record_keeps_the_columns_it_computes():
    """`kept_records` cuts a narrowed node's output slot and its column arguments' slots (a located run, or spans across fx
    inputs) to the words it computes; the shared argument's slot and an unnarrowed node's record are kept."""
    body = _linears()
    x, w = {"input": 0, "leaves": [0, 16]}, {"input": 1, "leaves": [96, 192]}
    spans = {"input": None, "leaves": None, "spans": [{"input": 2, "leaves": [0, 3]}, {"input": 3, "leaves": [10, 13]}]}
    calls = [{"occ": k, "returns": [{"port": 0, "output": 0, "leaves": [6, 12]}], "args": [x, w] + ([spans] if k == 1 else [])}
             for k in range(3)]
    form, recs, hints = kept_records(body, calls, {1: "h"}, registered=True)
    assert form.nodes[0].fn.bindings["N"] == 2 and hints == {1: "h"} and recs[2] is calls[2]
    assert recs[0]["returns"] == [{"port": 0, "output": 0, "leaves": [8, 10]}] and recs[0]["args"] == [x, {**w, "leaves": [128, 160]}]
    assert recs[1]["returns"][0]["leaves"] == [7, 12] and recs[1]["args"][:2] == [x, {**w, "leaves": [112, 192]}]
    assert recs[1]["args"][2]["spans"] == [{"input": 2, "leaves": [1, 3]}, {"input": 3, "leaves": [10, 13]}]
    assert kept_records(body, calls, {}, registered=False)[1] is calls


def test_the_last_layers_qkv_on_the_rows_before_the_prefills_last_is_narrowed_to_its_keys_and_values(monkeypatch):
    """TINY's request (4 query heads and 2 key/value heads of 16): the last layer reads no query of prompt rows 0-6, so their fused
    qkv `Gemm_v2{K=64,N=128}` becomes `Gemm_v2{K=64,N=64}` over the weight's K and V rows 64-127, and each reader of their keys
    and values reads the same words (the 7 rows' key rotaries, and the value histories of the 4 attentions after them: row 7's
    and 3 decode steps'); nothing else narrows."""
    got = PB.build_program(PHI, "request", LP=8, T=3).program.fn.body
    monkeypatch.setattr(PR, "narrowed", lambda b: (b, {}))
    full = PB.build_program(PHI, "request", LP=8, T=3).program.fn.body
    monkeypatch.undo()
    form, cols = narrowed(full)
    assert _shape(form) == _shape(got) and sorted(cols.values()) == [(64, 128)] * 7
    readers = 0
    for k, (old, new) in enumerate(zip(full.nodes, form.nodes)):
        if k in cols:
            assert (old.fn.id, new.fn.id) == (f"Gemm_v2{{K=64,N=128,DOT={DOT.id}}}", f"Gemm_v2{{K=64,N=64,DOT={DOT.id}}}")
            assert new.args[0] is old.args[0] and list(new.args[1].refs) == list(old.args[1].refs)[64 * 64:]
            continue
        for c, d in zip(old.args, new.args):
            if c is not d:
                readers += 1
                assert list(d.refs) == [(s, i, j - cols[i][0]) if s == "n" and i in cols else (s, i, j) for s, i, j in c.refs]
    assert readers == 7 + 4 and form.ret is full.ret
