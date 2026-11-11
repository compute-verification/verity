"""X-09 (M-BATCH-DECOMP): request attribution of fold instances and the log/header readers — pure python, no torch."""

from __future__ import annotations

import collections
import gzip
import json

import pytest

from verity_vllm.check.match import batch_decomp as bd

R0, R1 = "req-0", "req-1"


def _step(rows, logits_rows, logits_indices=None):
    return {"rows": rows, "logits_rows": logits_rows, "logits_indices": logits_indices or []}


def test_token_rows_attribute_by_step_rows():
    st = _step([(R0, 0), (R0, 1), (R0, 2), (R1, 0), (R1, 1)], [(R0, 2), (R1, 1)])
    assert bd.attribute({"row": 0, "rows": 5}, st) == (R0, "token")
    assert bd.attribute({"row": 2, "rows": 5}, st) == (R0, "token")
    assert bd.attribute({"row": 3, "rows": 5}, st) == (R1, "token")
    assert bd.attribute({"row": 4, "rows": 5}, st) == (R1, "token")


def test_logits_rows_attribute_by_logits_rows():
    st = _step([(R0, 0), (R0, 1), (R0, 2), (R1, 0), (R1, 1)], [(R0, 2), (R1, 1)])
    assert bd.attribute({"row": 0, "rows": 2}, st) == (R0, "logits")
    assert bd.attribute({"row": 1, "rows": 2}, st) == (R1, "logits")


def test_pure_decode_step_both_counts_agree():
    st = _step([(R0, 300), (R1, 12)], [(R0, 300), (R1, 12)])
    assert bd.attribute({"row": 0, "rows": 2}, st) == (R0, "token=logits")
    assert bd.attribute({"row": 1, "rows": 2}, st) == (R1, "token=logits")


def test_pure_decode_step_disagreement_is_ambiguous_not_guessed():
    st = _step([(R0, 300), (R1, 12)], [(R1, 12), (R0, 300)])  # logits rows reordered vs token rows
    rid, how = bd.attribute({"row": 0, "rows": 2}, st)
    assert rid is None and how.startswith("ambiguous")


def test_other_row_counts_are_unattributable():
    st = _step([(R0, 0), (R0, 1), (R1, 0)], [(R0, 1), (R1, 0)])
    rid, how = bd.attribute({"row": 0, "rows": 1}, st)  # a Shared/whole-step compute
    assert rid is None and how.startswith("rows=1")
    rid, how = bd.attribute({"row": 7, "rows": 3}, st)
    assert rid is None and how == "row-out-of-range"
    assert bd.attribute({"row": None, "rows": 3}, st) == (None, "no-row")


def test_read_steps_and_request_params(tmp_path):
    header = {"$e": "RunHeader", "schema": "x", "created_utc": "", "profile_id": "p", "profile": {}, "model": {},
              "workload": {"requests": [{"request_id": R0, "prompt_token_ids": [1, 2, 3], "max_tokens": 4},
                                        {"request_id": R1, "prompt_token_ids": [5, 6], "max_tokens": 2}],
                           "children": [{"request_id": "0_" + R1, "parent": R1}]}}
    s0 = {"$e": "StepStart", "step": 0, "rows": [[R0, 0], [R0, 1], [R0, 2]], "input_token_ids": [1, 2, 3], "positions": [0, 1, 2],
          "seq_lens": {}, "num_computed": {}, "block_tables": {}, "block_size": 16, "slot_mapping": [0, 1, 2],
          "logits_rows": [[R0, 2]], "logits_indices": [2], "num_reqs": 1, "num_tokens": 3, "num_tokens_padded": 3}
    s1 = dict(s0, step=1, rows=[[R0, 3], ["0_" + R1, 0], ["0_" + R1, 1]], logits_rows=[[R0, 3], ["0_" + R1, 1]], logits_indices=[0, 2])
    other = {"$e": "TritonLaunch", "grid": [1]}
    p = tmp_path / "log.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for d in (header, s0, other, s1):
            f.write(json.dumps(d, separators=(",", ":")) + "\n")
    h, steps = bd.read_steps(str(p))
    assert h["workload"]["requests"][1]["request_id"] == R1
    assert sorted(steps) == [0, 1]
    assert steps[1]["rows"] == [(R0, 3), ("0_" + R1, 0), ("0_" + R1, 1)]
    assert steps[1]["logits_indices"] == [0, 2]
    reqs, param_of, parent_of = bd.request_params(h)
    assert [r["request_id"] for r in reqs] == [R0, R1]
    assert param_of == {R0: 0, R1: 1, "0_" + R1: 1}  # a forked child reads its parent's prompt parameter
    assert parent_of["0_" + R1] == R1


def test_read_steps_rejects_duplicate_step(tmp_path):
    header = {"$e": "RunHeader", "workload": {"requests": [{"request_id": R0, "prompt_token_ids": [1]}]}}
    s0 = {"$e": "StepStart", "step": 0, "rows": [[R0, 0]], "logits_rows": [[R0, 0]]}
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in (header, s0, s0)) + "\n")
    with pytest.raises(ValueError, match="two StepStart"):
        bd.read_steps(str(p))


def test_derived_shape_from_construction_manifest(tmp_path):
    d = tmp_path / "build_request"
    d.mkdir()
    (d / "construction_manifest.json").write_text(json.dumps({"structural_inputs": {"LP": 256, "T": 31, "M": None}}))
    (d / "artifact.json").write_text(json.dumps({"applicability": {}}))
    assert bd.derived_shape(str(d)) == (256, 31)
    assert bd.derived_shape(str(tmp_path)) is None


def test_dataflow_attribution_is_primary_and_refuses_cross_request(monkeypatch, tmp_path):
    """Instances: node 3 (Embedding, reads prompt param 0 -> R0), node 4 (reads param 1 -> R1), node 5 (per-position Attention, rows=1,
    reads node 3 -> R0 by data flow though the row rule cannot place it), node 6 (reads nodes 3 AND 4 -> refused), node 7 (TokenSelect
    of R0), node 8 (TokenSelect of R1). Weights = param 2. The oracle RootView is stubbed; no Programs are decoded."""

    class FakeProg:
        pass

    B = FakeProg()
    B.inputs = {0: ("Input", 3), 1: ("Input", 2), 2: ("Input", 10)}
    B.n_input_nodes = 3
    B.digest = "d"
    B.oracle_wmap, B.oracle_wmap_overlaps, B._wlos = [(0, 10, "w")], [], [0]
    B.nodes = []
    B.inst = [
        {"k": 3, "spec": "Embedding_v1{V=8,H=2}", "form": "call", "args": [[("n", 0, 0, 1)], [("n", 2, 0, 10)]], "out": "bf16"},
        {"k": 4, "spec": "Embedding_v1{V=8,H=2}", "form": "call", "args": [[("n", 1, 0, 1)], [("n", 2, 0, 10)]], "out": "bf16"},
        {"k": 5, "spec": "Attention_v3{T=1,NH=1,KVH=1,D=2,BN=1}", "form": "call", "args": [[("n", 3, 0, 2)]], "out": "bf16"},
        {"k": 6, "spec": "Mix_v1", "form": "call", "args": [[("n", 3, 0, 2)], [("n", 4, 0, 2)]], "out": "bf16"},
        {"k": 7, "spec": "TokenSelect_v1{V=8}", "form": "call", "args": [[("n", 5, 0, 2)]], "out": "i32"},
        {"k": 8, "spec": "TokenSelect_v1{V=8}", "form": "call", "args": [[("n", 4, 0, 2)]], "out": "i32"},
    ]
    B.oracle_rows = [{"node": k - 3, "step": 0, "row": 0, "rows": 1 if k in (5, 6) else 2, "request": None} for k in range(3, 9)]
    header = {"workload": {"requests": [{"request_id": R0, "prompt_token_ids": [1, 2, 3]}, {"request_id": R1, "prompt_token_ids": [5, 6]}]}}
    steps = {0: _step([(R0, 0), (R1, 0)], [(R0, 0), (R1, 0)])}
    monkeypatch.setattr(bd, "read_steps", lambda p: (header, steps))
    monkeypatch.setattr(bd.C, "load_oracle", lambda d, label: B)
    res = bd.decompose(str(tmp_path), [], log_path="ignored")
    how = res["attribution"]["how"]
    assert how["dataflow"] == 5 and res["attribution"]["unattributed"] == 1
    bad = res["attribution"]["unattributed_examples"][0]
    assert bad["k"] == 6 and bad["why"].startswith("reads several requests")
    reqs = {e["request_id"]: e for e in res["requests"]}
    assert reqs[R0]["n_instances"] == 3 and reqs[R0]["LP"] == 3 and reqs[R0]["T"] == 0   # nodes 3, 5, 7
    assert reqs[R1]["n_instances"] == 2 and reqs[R1]["LP"] == 2 and reqs[R1]["T"] == 0   # nodes 4, 8
    assert res["missing_derived"] == [[2, 0], [3, 0]]
    assert res["verdict"] == "FAIL" and any("unattributed" in r for r in res["reasons"])


def _multi_weights_prog(n_weight_nodes=3):
    """A stub oracle whose WEIGHTS are several Input nodes (a mixed-width Record lowers to one Input
    batch per leaf field: FP8 e4m3 weight + f32 scale + bf16 norm = 3 here, 435 on Qwen3-4B-FP8).  Inputs: 0, 1 = the two prompts,
    2, 3, 4 = the weights leaves (offsets 0..3, 4..9, 10..11 of the Record).  Each request: Embedding(prompt, W[2]) -> Linear(., W[3]) ->
    Norm(., W[4]) -> TokenSelect.  The rows rule cannot place any of them (rows=1); dataflow must."""

    class FakeProg:
        pass

    B = FakeProg()
    B.inputs = {0: ("Input32_v1", 3), 1: ("Input32_v1", 2), 2: ("Input8_v1", 4), 3: ("Input32_v1", 6), 4: ("Input16_v1", 2)}
    B.n_input_nodes = 5
    B.digest = "d"
    B.oracle_wmap = [(0, 4, "w_e4m3"), (4, 10, "w_scale"), (10, 12, "norm")]
    B.oracle_wmap_overlaps, B._wlos = [], [0, 4, 10]
    B.input_base = {2: 0, 3: 4, 4: 10}
    B.nodes = []
    B.seed_index = {}
    B.inst = [
        {"k": 5, "spec": "Embedding_v1{V=8,H=2}", "form": "call", "args": [[("n", 0, 0, 1)], [("n", 2, 0, 4)]], "out": "bf16"},   # R0
        {"k": 6, "spec": "Embedding_v1{V=8,H=2}", "form": "call", "args": [[("n", 1, 0, 1)], [("n", 2, 0, 4)]], "out": "bf16"},   # R1
        {"k": 7, "spec": "Linear_v1{N=2}", "form": "call", "args": [[("n", 5, 0, 2)], [("n", 3, 0, 6)]], "out": "bf16"},           # R0
        {"k": 8, "spec": "Linear_v1{N=2}", "form": "call", "args": [[("n", 6, 0, 2)], [("n", 3, 0, 6)]], "out": "bf16"},           # R1
        {"k": 9, "spec": "Norm_v1{N=2}", "form": "call", "args": [[("n", 7, 0, 2)], [("n", 4, 0, 2)]], "out": "bf16"},             # R0
        {"k": 10, "spec": "Norm_v1{N=2}", "form": "call", "args": [[("n", 8, 0, 2)], [("n", 4, 0, 2)]], "out": "bf16"},            # R1
        {"k": 11, "spec": "TokenSelect_v1{V=8}", "form": "call", "args": [[("n", 9, 0, 2)]], "out": "i32"},                        # R0
        {"k": 12, "spec": "TokenSelect_v1{V=8}", "form": "call", "args": [[("n", 10, 0, 2)]], "out": "i32"},                       # R1
    ]
    B.oracle_rows = [{"node": k - 5, "step": 0, "row": 0, "rows": 1, "request": None} for k in range(5, 13)]
    return B


def test_multi_node_weights_are_shared_inputs_not_cross_request_reads(monkeypatch, tmp_path):
    """With the weights lowered to 3 Input nodes, every request's reads of nodes 2, 3, 4 are shared-weights
    reads: all 8 instances attributed by dataflow, 0 cross-request reads, and the projections canonicalise the weight operands as
    ("W", field, in-field offsets) through the oracle's field map + input_base (older code refused the Program: '5 input nodes
    but 2 requests (+1 weights expected)')."""
    B = _multi_weights_prog()
    header = {"workload": {"requests": [{"request_id": R0, "prompt_token_ids": [1, 2, 3]}, {"request_id": R1, "prompt_token_ids": [5, 6]}]}}
    steps = {0: _step([(R0, 0), (R1, 0)], [(R0, 0), (R1, 0)])}
    monkeypatch.setattr(bd, "read_steps", lambda p: (header, steps))
    monkeypatch.setattr(bd.C, "load_oracle", lambda d, label: B)
    res = bd.decompose(str(tmp_path), [], log_path="ignored")   # older code: ValueError '5 input nodes but the header has 2 requests (+1 weights expected)'
    assert bd.weights_input_nodes(B, 2, 0) == {2, 3, 4}
    assert res["attribution"]["how"] == {"dataflow": 8} and res["attribution"]["unattributed"] == 0
    assert res["oracle"]["weights_input_nodes"] == {"n": 3, "first": 2, "last": 4, "request_owned_inputs": 2}
    reqs = {e["request_id"]: e for e in res["requests"]}
    assert reqs[R0]["n_instances"] == 4 and reqs[R1]["n_instances"] == 4
    assert res["missing_derived"] == [[2, 0], [3, 0]] and res["verdict"] == "MISSING_DERIVED"   # nothing refused: only the components are absent
    # the projection's canonical sources: weight reads name the FIELD by Record-global offset (input_base), not node-local offset 0
    P = bd.Projection(B, "oracle[r0]", [0, 2, 4, 6], tok_node=0, weights_node={2, 3, 4})
    assert P.canon(0) == [[("TOK", 0, 1)], [("W", "w_e4m3", 0, 4)]]
    assert P.canon(1) == [[("N", 0, 0, 0, 2)], [("W", "w_scale", 0, 6)]]
    assert P.canon(2) == [[("N", 0, 1, 0, 2)], [("W", "norm", 0, 2)]]
    assert not P.xreq
    # the other request's prompt is still a cross-request read (the set fix does not widen what counts as shared)
    Q = bd.Projection(B, "oracle[r1-wrong]", [0, 2, 4, 6], tok_node=1, weights_node={2, 3, 4})
    Q.canon(0)
    assert Q.xreq == {"prompt-param": 1}
    # the old single-node call form (an int) still means that one node
    S = bd.Projection(B, "oracle[r0]", [0], tok_node=0, weights_node=2)
    assert S.weights_nodes == frozenset({2}) and S.weights_node == 2


def test_weights_input_nodes_refuses_a_layout_that_cannot_hold_the_header_by_name():
    B = _multi_weights_prog()
    with pytest.raises(ValueError, match="5 input nodes but the header has 5 requests"):
        bd.weights_input_nodes(B, 5, 0)                   # more request params than Inputs: no room for any weights node
    with pytest.raises(ValueError, match="5 input nodes but the header has 3 requests \\+ 2 seeds"):
        bd.weights_input_nodes(B, 3, 2)
    B.inputs = {0: ("Input32_v1", 3), 2: ("Input8_v1", 4), 3: ("Input32_v1", 6)}   # node 1 is not an Input: request 1's prompt is missing
    with pytest.raises(ValueError, match="not the leading Input nodes 0..1"):
        bd.weights_input_nodes(B, 2, 0)


def test_decompose_refuses_fewer_inputs_than_requests_by_name(monkeypatch, tmp_path):
    B = _multi_weights_prog()
    header = {"workload": {"requests": [{"request_id": f"r{i}", "prompt_token_ids": [1]} for i in range(5)]}}
    monkeypatch.setattr(bd, "read_steps", lambda p: (header, {}))
    monkeypatch.setattr(bd.C, "load_oracle", lambda d, label: B)
    with pytest.raises(ValueError, match="5 requests"):
        bd.decompose(str(tmp_path), [], log_path="ignored")


def _dag_prog(order: list[int], rewire: bool = False):
    """A 4-instance DAG over a stub RootView: two per-row ops (rows 0, 1) each feeding a Norm, then a TokenSelect over both norms. Emitted in
    `order` (indices into the canonical node list); `rewire` swaps which row feeds which Norm (same multiset of specs, different wiring)."""

    class FakeProg:
        pass

    tok, w = 0, 1
    nodes = [
        {"spec": "Linear_v1{N=2}", "form": "call", "args": [[("n", tok, 0, 1)], [("n", w, 0, 4)]], "out": "bf16"},           # 2: row 0
        {"spec": "Linear_v1{N=2}", "form": "call", "args": [[("n", tok, 1, 2)], [("n", w, 0, 4)]], "out": "bf16"},           # 3: row 1
        {"spec": "Norm_v1{N=2}", "form": "call", "args": [[("n", 2 if not rewire else 3, 0, 2)]], "out": "bf16"},           # 4
        {"spec": "Norm_v1{N=2}", "form": "call", "args": [[("n", 3 if not rewire else 2, 0, 2)]], "out": "bf16"},           # 5
        {"spec": "TokenSelect_v1{V=2}", "form": "call", "args": [[("n", 4, 0, 2), ("n", 5, 0, 2)]], "out": "i32"},          # 6
    ]
    P = FakeProg()
    P.inputs = {tok: ("Input", 2), w: ("Input", 4)}
    P.tok_node, P.weight_fields = tok, {w: "w"}
    P.oracle_wmap = None
    P.inst = [nodes[i] for i in order]
    P.steps = [list(range(len(order)))]
    k2pos = {order[j] + 2: j for j in range(len(order))}   # node id -> position in the emitted sequence

    def src(run):
        _, idx, lo, hi = run
        if idx == tok:
            return ("TOK", lo, hi)
        if idx == w:
            return ("W", "w", lo, hi)
        return ("N", 0, k2pos[idx], lo, hi)

    P.canon = lambda i, unfold=False: [[src(run) for run in a] for a in P.inst[i]["args"]]
    return P


def test_dag_multiset_is_order_insensitive_but_wiring_sensitive():
    """Chunk-major vs op-major emission of the same DAG -> equal multisets; swapping the wiring -> unequal (same spec multiset)."""
    op_major = _dag_prog([0, 1, 2, 3, 4])
    chunk_major = _dag_prog([0, 2, 1, 3, 4])
    rewired = _dag_prog([0, 1, 2, 3, 4], rewire=True)
    ma, mb, mc = bd.dag_multiset(op_major), bd.dag_multiset(chunk_major), bd.dag_multiset(rewired)
    assert sum(ma.values()) == 5 and ma == mb
    assert ma != mc and collections.Counter(r["spec"] for r in op_major.inst) == collections.Counter(r["spec"] for r in rewired.inst)


def test_live_cone_drops_a_discarded_chunk_sample():
    """A chunked-prefill fold carries an extra (Norm row, TokenSelect) pair after the first chunk whose token nothing reads; the live
    cone of the last TokenSelect excludes exactly those, and the pruned DAG multiset equals the unchunked Program's."""
    P = _dag_prog([0, 1, 2, 3, 4])
    Q = _dag_prog([0, 2, 1, 3, 4])
    # emitted sequence: Linear r0, Norm r0, Linear r1, Norm r1, DEAD Norm(r0), DEAD TokenSelect(dead Norm), real TokenSelect(Norm r0, Norm r1)
    real = list(Q.inst)
    Q.inst = real[:4] + [{"spec": "Norm_v1{N=2}", "form": "call", "args": [], "out": "bf16"},
                         {"spec": "TokenSelect_v1{V=2}", "form": "call", "args": [], "out": "i32"}] + real[4:]
    Q.steps = [list(range(7))]
    old = Q.canon

    def canon(i, unfold=False):
        if i == 4:
            return [[("N", 0, 0, 0, 2)]]     # dead Norm reads Linear r0
        if i == 5:
            return [[("N", 0, 4, 0, 2)]]     # dead TokenSelect reads the dead Norm
        if i == 6:
            return [[("N", 0, 1, 0, 2), ("N", 0, 3, 0, 2)]]
        return old(i, unfold)

    Q.canon = canon
    live = bd.live_cone(Q)
    assert live == {0, 1, 2, 3, 6}
    assert bd.dag_multiset(Q, live) == bd.dag_multiset(P, bd.live_cone(P))
    assert bd.dag_multiset(Q) != bd.dag_multiset(P)
