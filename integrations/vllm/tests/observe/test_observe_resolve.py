"""The value-observation contract (`verity_vllm.observe.contract`) and occurrence resolution (`verity_vllm.check.match.resolve`) over a
SYNTHETIC record -- pure python, no torch, no GPU.

The record is a two-request batched fold in today's on-disk shape (instances.jsonl v1 rows + capture/log.jsonl.gz RunHeader/StepStart):
request A prefill 2 rows then 2 decodes, request B arrives at engine step 1 with a 3-row prompt CHUNKED over engine steps 1..2, then one
decode; one weight-only value per step (`1 + w`), a Const row, and unstamped rows for the rows channel.
"""
from __future__ import annotations

import collections
import gzip
import json

import pytest

from verity_vllm.observe import contract as OB
from verity_vllm.check.match import resolve as RS
from verity_vllm.check.match import batch_decomp as BD

A, B = "req-A", "req-B"
N_IN = 3        # prompt_A, prompt_B, weights


def _write_record(tmp_path, *, stamp: bool = True, mutate_view: bool = False, bad_layer_name: bool = False):
    d = tmp_path / "match"
    (d / "capture").mkdir(parents=True)
    # engine steps: 0 = A prefill (rows A0 A1) ; 1 = A decode (A2) + B chunk 1 (B0, B1) ; 2 = A decode (A3) + B chunk 2 (B2) ; 3 = B decode (B3)
    steps = {0: {"rows": [[A, 0], [A, 1]], "logits_rows": [[A, 1]]},
             1: {"rows": [[A, 2], [B, 0], [B, 1]], "logits_rows": [[A, 2], [B, 1]]},
             2: {"rows": [[A, 3], [B, 2]], "logits_rows": [[A, 3], [B, 2]]},
             3: {"rows": [[B, 3]], "logits_rows": [[B, 3]]}}
    header = {"$e": "RunHeader", "schema": "toy", "workload": {"requests": [{"request_id": A, "prompt_token_ids": [1, 2], "max_tokens": 3},
                                                                            {"request_id": B, "prompt_token_ids": [5, 6, 7], "max_tokens": 2}]}}
    with gzip.open(d / "capture" / "log.jsonl.gz", "wt") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for s, st in steps.items():
            f.write(json.dumps({"$e": "StepStart", "step": s, "rows": st["rows"], "logits_rows": st["logits_rows"], "logits_indices": []}, separators=(",", ":")) + "\n")
    rows: list[dict] = []
    node = [0]

    def emit(step, spec, hint, *, req=None, pos=None, row=0, nrows=1, reads=(), preads=(), name=None, view=None, collective=None):
        k = node[0]
        node[0] += 1
        o = {"node": k, "step": step, "spec": spec, "row": row, "rows": nrows, "hint": hint, "name": name or hint.split("/")[0],
             "request": req if stamp else None, "position": pos if stamp else None,
             "operand_refs": [[["n", r, 0, 1] for r in reads] + [["p", p, 0, 8] for p in preads]]}
        if view is not None:
            o["output_views"] = [view]
        if collective:
            o["collective"] = collective
        rows.append(o)
        return k

    V = {"shape": [8], "strides": [1], "offset": 0, "dtype": "bf16"}
    Vmut = {"shape": [8], "strides": [2], "offset": 0, "dtype": "bf16"}
    for s, st in steps.items():
        tok, log = st["rows"], st["logits_rows"]
        # a weight-only value (reads the weights parameter only) -- SHARED
        w = emit(s, "AddScalarBf16_v1{N=8}", "model.layers.0.input_layernorm/weight", preads=(2,), nrows=1, name="layers.0/scale")
        # one norm row per token row: reads the request's prompt (prefill) / the weight-only value
        norms = []
        for i, (r, p) in enumerate(tok):
            pidx = 0 if r == A else 1
            reads = (N_IN + w,)
            norms.append(emit(s, "RMSNormTriton_v1{N=8}", "model.layers.0.input_layernorm/weight", req=r, pos=p, row=i, nrows=len(tok),
                              reads=reads, preads=(pidx,), name=f"t={p}/layers.0/input_layernorm", view=(Vmut if (mutate_view and s == 2 and i == 0) else V)))
        gemms = []
        for i, (r, p) in enumerate(tok):
            nm = f"t={p}/layers.{'1' if bad_layer_name and s == 3 else '0'}/qkv_proj"
            gemms.append(emit(s, "Gemm_v1{N=8,K=8}", "model.layers.0.self_attn.qkv_proj/weight", req=r, pos=p, row=i, nrows=len(tok),
                              reads=(N_IN + norms[i],), preads=(2,), name=nm, view=V, collective={"rank": 0, "world": 1, "kind": "all_reduce", "k": 0}))
        for j, (r, p) in enumerate(log):
            i = [x for x, (rr, _pp) in enumerate(tok) if rr == r][-1]
            emit(s, "TokenSelect_v1{V=8}", "runner.sampler/out", req=r, pos=p, row=j, nrows=len(log), reads=(N_IN + gemms[i],), name=f"t={p}/sampler")
    with open(d / "instances.jsonl", "w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "0" * 64, "n_instances": len(rows), "operand_refs_form": "runs",
                            "requests": [{"index": 0, "request_id": A, "prompt_len": 2, "served_samples": 3, "max_tokens": 3},
                                         {"index": 1, "request_id": B, "prompt_len": 3, "served_samples": 2, "max_tokens": 2}]}) + "\n")
        for o in rows:
            f.write(json.dumps(o) + "\n")
    json.dump({"program": {}, "digest": "0" * 64, "n_input_nodes": N_IN, "params": [["prompt_0", "x"], ["prompt_1", "x"], ["weights", "x"]]},
              open(d / "program.json", "w"))
    return str(d), rows, steps


# ------------------------------------------------------------------------------------------------------------------ observe


def test_iter_observations_streams_every_row_in_node_order_with_the_fold_coordinates(tmp_path):
    d, rows, steps = _write_record(tmp_path)
    obs = list(OB.iter_observations(d))
    assert len(obs) == len(rows) == OB.count_observations(d)
    assert [o.hint_value_id for o in obs] == list(range(len(rows)))            # the hint is the collector's index, node order
    hdr = OB.read_header(d)
    assert hdr.n_input_nodes == N_IN and hdr.params == ["prompt_0", "prompt_1", "weights"] and hdr.weights_param == 2
    assert [r["request_id"] for r in hdr.requests] == [A, B]
    o = next(x for x in obs if x.spec.startswith("Gemm"))
    occ = o.occurrence
    assert occ.site == "model.layers.0.self_attn.qkv_proj" and occ.slot == "out" and occ.rank == 0
    assert occ.request == A and occ.position == 0 and occ.row == 0 and occ.rows == 2 and occ.step == 0
    assert occ.view == ((8,), (1,), 0, "bf16") and occ.layer() == 0
    assert o.reads and all(r >= N_IN for r in o.reads) and o.param_reads == (2,)
    # the weight-only row: sited at its module, no request stated, reads the weights parameter only
    w = obs[0]
    assert w.occurrence.request is None and w.param_reads == (2,) and w.reads == ()
    # invocation = the k-th launch of the site inside the step (row 0 of a launch opens a new one)
    sampler = [x for x in obs if x.spec.startswith("TokenSelect") and x.occurrence.step == 1]
    assert [x.occurrence.invocation for x in sampler] == [0, 0]                 # one launch over 2 logits rows
    # the row dir form: `<row>/match`
    assert OB.match_dir(str(tmp_path)) == d


def test_unstamped_rows_are_observed_with_no_request_and_resolved_by_the_rows_channel(tmp_path):
    d, rows, steps = _write_record(tmp_path, stamp=False)
    obs = list(OB.iter_observations(d))
    assert all(o.occurrence.request is None and o.occurrence.position is None for o in obs)      # the adapter never guesses
    hdr = OB.read_header(d)
    run_header, st = OB.read_steps(d)
    assert sorted(st) == [0, 1, 2, 3] and st[1]["rows"] == [(A, 2), (B, 0), (B, 1)]
    live = RS.LiveState.of_header(hdr, st, run_header)
    got = []
    for o in obs:
        ref = RS.resolve(None, None, o.occurrence, live, o)
        RS.advance(live, o, ref)
        if o.spec.startswith("TokenSelect") and o.occurrence.step == 1:
            got.append((ref.request, ref.channel))
    # the sampler rows read one request's Gemm: the dataflow channel names it (the Match stage's order: meta -> dataflow -> weight-only -> rows)
    assert got == [(A, "dataflow"), (B, "dataflow")]
    # a row reading nothing request-owned and no weight is the rows channel's: BD.attribute over the StepStart rows
    occ = OB.RuntimeOccurrence(step=1, invocation=0, rank=0, site="x", slot="out", row=1, rows=3)
    ref = RS.resolve(None, None, occ, live)
    assert (ref.request, ref.channel) == (B, "rows:token")


def test_progression_form_operand_refs_expand_to_distinct_node_ids():
    hdr = OB.RecordHeader(schema="verity-gen/instances/v2", operand_refs_form="progressions", n_instances=1, requests=[], n_input_nodes=3, params=[], tp=None, digest=None)
    o = {"node": 9, "step": 0, "spec": "Attention_v3{T=4}", "row": 0, "rows": 1, "hint": "model.layers.0.self_attn.attn/kv",
         "operand_refs": [[["N", 10, 0, 8, 2, 0, 0, 3]], [["p", 2, 0, 8]]]}
    ob = OB.adapt_row(o, hdr)
    assert ob.reads == (10, 12, 14) and ob.param_reads == (2,)
    assert ob.occurrence.site == "model.layers.0.self_attn.attn"


@pytest.mark.parametrize("params,tp,want", [('[["a", "bf16"], ["b", "f32"]]', '{"world": 2}', (["a", "b"], {"world": 2})),
                                            ('[["a", "bf16"], ["b"]]', '{"world": }', ([], None))])
def test_a_large_program_json_is_scanned_and_a_cut_window_is_stated_unknown(tmp_path, monkeypatch, params, tp, want):
    p = tmp_path / "program.json"
    p.write_text(f'{{"n_input_nodes": 3, "params": {params}, "tp": {tp}, "program": {{}}}}')
    monkeypatch.setattr(OB.os.path, "getsize", lambda path: 1 << 30)
    assert OB._program_json_layout(str(p), None) == (3, *want, None)


# ------------------------------------------------------------------------------------------------------------------ resolve: the record


def _v1_attribution(d: str, steps: dict) -> tuple[dict, list, list]:
    """The Match stage's own attribution over the same rows (BD.decompose's channel loop, meta -> rows) -- the ground truth today."""
    per: dict = collections.OrderedDict()
    shared, un = [], []
    for o in OB.iter_rows(d):
        rid = o.get("request")
        if rid is None:
            reads = [r[1] for a in o["operand_refs"] for r in a if r[0] == "n"]
            preads = [r[1] for a in o["operand_refs"] for r in a if r[0] == "p"]
            if not reads and preads == [2]:
                shared.append(o["node"] + N_IN)
                continue
            st = steps[o["step"]]
            rid, _how = BD.attribute(o, {"rows": [tuple(x) for x in st["rows"]], "logits_rows": [tuple(x) for x in st["logits_rows"]]})
        if rid is None:
            un.append(o["node"])
        else:
            per.setdefault(rid, []).append(o["node"] + N_IN)
    return per, shared, un


@pytest.mark.parametrize("stamp", [True, False])
def test_attribute_record_reproduces_the_match_stages_attribution_identity_set_by_identity_set(tmp_path, stamp):
    d, rows, steps = _write_record(tmp_path, stamp=stamp)
    att = RS.attribute_record(d)
    v1, shared, un = _v1_attribution(d, steps)
    cmp = RS.compare_attributions(v1, att.per_req)
    assert cmp["equal"], cmp
    assert sorted(k for k, _s in att.shared_nodes) == sorted(shared) and not att.unattributed and not un
    assert att.how == ({"meta": 22, "weight-only": 4} if stamp else {"dataflow": 22, "weight-only": 4}), att.how
    assert att.faults == {}, att.fault_examples
    # legs: the weight-only value of an engine step joins every request with rows in it (BD.shared_legs)
    legs = att.legs(steps)
    assert len(legs[A]) == len(att.per_req[A]) + 3 and len(legs[B]) == len(att.per_req[B]) + 3


def test_request_relative_steps_follow_x09_segmentation_including_the_chunked_prefill(tmp_path):
    d, rows, steps = _write_record(tmp_path)
    hdr = OB.read_header(d)
    run_header, st = OB.read_steps(d)
    live = RS.LiveState.of_header(hdr, st, run_header)
    refs = {}
    for obs in OB.iter_observations(d):
        ref = RS.resolve(None, None, obs.occurrence, live, obs)
        refs[obs.hint_value_id] = (obs, ref)
        RS.advance(live, obs, ref)
    by_req_step = collections.defaultdict(list)
    for k, (obs, ref) in refs.items():
        if ref.request not in (None, RS.SHARED):
            by_req_step[(ref.request, ref.step)].append((obs.occurrence.step, obs.spec.split("{")[0], ref.ordinal))
    # A: prefill at engine step 0 -> step 0; decodes at 1, 2 -> steps 1, 2
    assert {s for s, _, _ in by_req_step[(A, 0)]} == {0} and {s for s, _, _ in by_req_step[(A, 1)]} == {1} and {s for s, _, _ in by_req_step[(A, 2)]} == {2}
    # B: chunked prefill over engine steps 1 and 2 -> BOTH step 0 (the discarded chunk sample is not a step); decode at 3 -> step 1
    assert {s for s, _, _ in by_req_step[(B, 0)]} == {1, 2} and {s for s, _, _ in by_req_step[(B, 1)]} == {3}
    assert (B, 2) not in by_req_step
    # ordinals count per (request, step, site, slot): B's step 0 has 3 norm rows -> ordinals 0, 1, 2 across the two chunks
    norms_b0 = sorted(o for s, sp, o in by_req_step[(B, 0)] if sp == "RMSNormTriton_v1")
    assert norms_b0 == [0, 1, 2]
    # the live state is one owner per value, not the observations
    assert len(live.owner) == N_IN + len(rows) and live.owner_of(N_IN + 0) == RS.SHARED


# ------------------------------------------------------------------------------------------------------------------ resolve: the Program


def _component(prefix: str = "model.", *, tp_rank: int | None = None) -> RS.ComponentProgram:
    """A request component Program (instances.json.gz doc) with a PROMOTED interior call: `Bf16MulScalarTensor_v1` under `layers.0.mlp`
    whose output is read by `layers.0.post_norm` (`<consumer>/in`), two steps, a TP collective site (rank Program)."""
    P = prefix
    rows = []

    def row(i, spec, name, args=(), view=None):
        r = {"i": i, "spec": spec, "name": name, "args": [[["n", a, 0, 8]] for a in args]}
        if view is not None:
            r["view"] = view
        rows.append(r)

    V = {"shape": [8], "strides": [1], "offset": 0, "dtype": "bf16"}
    for s in range(2):
        b = len(rows)
        row(b + 0, "RMSNormTriton_v1{N=8}", f"{P}model.layers.0.input_layernorm/rms_norm_{s}/0[0]", view=V)
        row(b + 1, "Gemm_v1{N=8,K=8}", f"{P}model.layers.0.self_attn.qkv_proj/linear_{s}/0[0]", args=(b + 0,), view=V)
        row(b + 2, "Bf16MulScalarTensor_v1{N=8}", f"{P}model.layers.0.mlp/mul_{s}[0]", args=(b + 1,))              # interior: crosses to post_norm
        row(b + 3, "RMSNormTriton_v1{N=8}", f"{P}model.layers.0.post_norm/rms_norm_p{s}/0[0]", args=(b + 2,), view=V)
        row(b + 4, "AllReduce2_v1{N=8}", f"{P}model.layers.0.self_attn.o_proj/all_reduce_{s}[0]", args=(b + 3,))
        row(b + 5, "TokenSelect_v1{V=8}", f"argmax_{s}[0]", args=(b + 4,))
    inst = {"rows": rows, "params": [["prompt", "Array<2,Value<16>>"], ["weights", "{w: Array<8,Value<16>>}"]]}
    return RS.ComponentProgram(inst, RS.DescriptorCorrespondence(inst, prefix=P))


def _live_one_request(rid: str = "r0", lp: int = 2) -> RS.LiveState:
    hdr = OB.RecordHeader(schema=None, operand_refs_form="runs", n_instances=None, requests=[{"index": 0, "request_id": rid, "prompt_len": lp}],
                          n_input_nodes=2, params=["prompt", "weights"], tp=None, digest=None)
    return RS.LiveState.of_header(hdr, steps=None)


def test_descriptor_correspondence_exposes_the_three_functions():
    prog = _component()
    c = prog.corr
    assert c.owning_module(1) == "model.layers.0.self_attn.qkv_proj" and c.return_slot(1) == "0"
    assert c.owning_module(2) == "model.layers.0.mlp" and c.return_slot(2) == "out"
    assert c.arg_slot(3, 0) == "model.layers.0.post_norm/in"
    assert c.producers(3) == [2]
    assert prog.n_steps == 2 and prog.calls_at(1) == [6, 7, 8, 9, 10, 11]
    assert prog.at(0, "model.layers.0.self_attn.qkv_proj", "0") == [1] and prog.at(1, "model.layers.0.post_norm", "in") == [9]


def _record_for(prog: RS.ComponentProgram, module_prefix: str = "") -> dict:
    """A minimal `runtime-correspondence/v1` document for the component: one occurrence per row, the owning module = the row's module
    under `module_prefix` (the export wrapper's spelling in the RECORD, e.g. a TP stub's `inner.model.`)."""
    from verity_vllm.query.manifest.format import split_name
    occs, calls = [], []
    for r in prog.rows:
        mod, node, hint, _row = split_name(r.get("name") or "")
        occs.append({"node": node or r["name"], "module": (module_prefix + mod) if mod else "", "op": "aten.op", "outputs": [], "inputs": []})
        calls.append({"occ": len(occs) - 1, "member": hint or None, "returns": [{"port": 0}], "args": []})
    from verity_vllm.correspondence.runtime import SCHEMA
    return {"schema": SCHEMA, "body": "program_fn", "inputs": ["prompt"], "occurrences": occs, "calls": calls}


def test_descriptor_correspondence_prefers_the_runtime_correspondence_record_and_records_the_fallback():
    from verity_vllm.correspondence import runtime as RC
    # (1) v1 route, no record: recorded as the fallback
    prog = _component()
    assert prog.corr.source == "v1_annotations" and prog.corr.table is None and prog.corr.provenance()["fallback"]
    # (2) the record travels on the doc (`annotations["correspondence"]`): the canonical reader serves, digest recorded; same answers
    rows = prog.rows
    inst = {"rows": rows, "params": [["prompt", "Array<2,Value<16>>"], ["weights", "{w: Array<8,Value<16>>}"]],
            "annotations": {RC.NAMESPACE: _record_for(prog, module_prefix="")}}      # record spells modules as `model.layers...` (no wrapper)
    ATTN = ["model.layers.0.self_attn.qkv_proj"]                                      # result.json `attention_impls_observed`: the implementation's path
    c = RS.DescriptorCorrespondence(inst, attention_paths=ATTN)
    assert c.source == "correspondence" and c.digest == RC.correspondence_digest(inst["annotations"][RC.NAMESPACE]) and c.problem is None
    assert c.prefix == "model." and c.owning_module(1) == "model.layers.0.self_attn.qkv_proj" and c.owning_module(2) == "model.layers.0.mlp" and c.owning_module(5) is None
    assert c.arg_slot(3, 0) == "model.layers.0.post_norm/in" and c.return_slot(1) == "0"
    assert c.provenance()["source"] == "correspondence" and c.provenance()["schema"] == RC.SCHEMA
    # (3) with NO observed attention path the prefix is undetermined ("") and the record's own path stands -- no wrapper spelling is
    #     guessed (v1's route would read `model.` off `model.model.`); a TP stub's record (`inner.model.model.layers...`) likewise, and with
    #     the observed path its prefix is determined from the record, never ""
    c0 = RS.DescriptorCorrespondence(inst)
    assert c0.source == "correspondence" and c0.prefix == "" and c0.owning_module(1) == "model.model.layers.0.self_attn.qkv_proj"
    inst_tp = dict(inst, annotations={RC.NAMESPACE: _record_for(prog, module_prefix="inner.")})
    c_tp = RS.DescriptorCorrespondence(inst_tp)
    assert c_tp.prefix == "" and c_tp.owning_module(1) == "inner.model.model.layers.0.self_attn.qkv_proj"
    c_tp2 = RS.DescriptorCorrespondence(inst_tp, attention_paths=ATTN)
    assert c_tp2.prefix == "inner.model." and c_tp2.owning_module(1) == "model.layers.0.self_attn.qkv_proj" and c_tp2.arg_slot(3, 0) == "model.layers.0.post_norm/in"
    # (4) a record of another Program (Call count differs) is refused with the problem recorded and the v1 route taken
    bad = _record_for(prog); bad["calls"] = bad["calls"][:-1]; bad["occurrences"] = bad["occurrences"][:-1]
    c_bad = RS.DescriptorCorrespondence(dict(inst, annotations={RC.NAMESPACE: bad}))
    assert c_bad.source == "v1_annotations" and "of another Program" in c_bad.problem and c_bad.owning_module(1) == "model.layers.0.self_attn.qkv_proj"
    # (5) ComponentProgram.load reads the Build dir's record through the ONE loader (read_program_dir)
    import gzip, json, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        with gzip.open(os.path.join(d, "instances.json.gz"), "wt") as f:
            json.dump({"rows": rows, "params": inst["params"]}, f)
        with gzip.open(os.path.join(d, "descriptor.json.gz"), "wt") as f:
            json.dump({"annotations": {RC.NAMESPACE: _record_for(prog)}}, f)
        json.dump({"program": {"correspondence_digest": RC.correspondence_digest(_record_for(prog))}}, open(os.path.join(d, "result.json"), "w"))
        loaded = RS.ComponentProgram.load(d, attention_paths=ATTN)
        assert loaded.corr.source == "correspondence" and loaded.corr.problem is None and loaded.corr.owning_module(2) == "model.layers.0.mlp"
        json.dump({"program": {}}, open(os.path.join(d, "result.json"), "w"))                      # a v1-frontend Build: no digest -> fallback
        assert RS.ComponentProgram.load(d).corr.source == "v1_annotations"


def test_resolve_promoted_interior_value_at_the_consumers_input_slot():
    prog = _component()
    live = _live_one_request()
    # the collector observed the promoted `layers.0.mlp` output at its consumer's INPUT: site = consumer, slot = "in"
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.0.post_norm", slot="in", request="r0", position=1)
    ref = RS.resolve(prog, prog.corr, occ, live)
    assert ref.call == 3 and ref.step == 0 and ref.ordinal == 0 and ref.problems == () and ref.channel == "meta"
    assert prog.corr.arg_slot(ref.call, 0) == "model.layers.0.post_norm/in"
    assert prog.corr.producers(ref.call) == [2] and prog.corr.owning_module(2) == "model.layers.0.mlp"
    obs = OB.ValueObservation(occurrence=occ, hint_value_id=3)
    assert RS.validate(obs, ref, correspondence=prog.corr, expected_call_of_hint=lambda h: h) == []


def test_resolve_carries_the_tp_rank_coordinate_and_reports_a_rank_disagreement():
    prog = _component()
    live = _live_one_request()
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=1, site="model.layers.0.self_attn.o_proj", slot="out", request="r0", position=1)
    ref = RS.resolve(prog, prog.corr, occ, live)
    assert ref.rank == 1 and ref.call == 4 and ref.identity() == ("r0", 0, "model.layers.0.self_attn.o_proj", "out", 0, 1)
    # the same site on rank 0 is a DIFFERENT identity (the rank is a coordinate of the value)
    occ0 = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.0.self_attn.o_proj", slot="out", request="r0", position=1)
    assert RS.resolve(prog, prog.corr, occ0, live).identity() != ref.identity()


def test_mutated_view_is_a_named_fault():
    prog = _component()
    live = _live_one_request()
    good = ((8,), (1,), 0, "bf16")
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.0.self_attn.qkv_proj", slot="0", request="r0", position=1, view=good)
    obs = OB.ValueObservation(occurrence=occ, hint_value_id=1, spec="Gemm_v1{N=8,K=8}")
    ref = RS.resolve(prog, prog.corr, occ, live, obs)
    assert RS.validate(obs, ref, correspondence=prog.corr, live=live) == []
    RS.advance(live, obs, ref)
    # a later observation at the same site/slot with a different stride layout: the correspondence's declared view AND the live state say so
    bad = OB.RuntimeOccurrence(step=1, invocation=0, rank=0, site="model.layers.0.self_attn.qkv_proj", slot="0", request="r0", position=2, view=((8,), (2,), 0, "bf16"))
    obs2 = OB.ValueObservation(occurrence=bad, hint_value_id=7, spec="Gemm_v1{N=8,K=8}")
    live.cursor("r0").ts_seen = 1   # the prefill's sampling event has passed
    ref2 = RS.resolve(prog, prog.corr, bad, live, obs2)
    faults = RS.validate(obs2, ref2, correspondence=prog.corr, live=live)
    assert [f.kind for f in faults] == ["view-mutated"], faults
    assert "declared view" in faults[0].detail


def test_hint_mismatch_is_a_named_fault_and_the_hint_never_decides_identity():
    prog = _component()
    live = _live_one_request()
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.0.input_layernorm", slot="0", request="r0", position=1)
    obs = OB.ValueObservation(occurrence=occ, hint_value_id=99)         # the collector's index claims another call
    ref = RS.resolve(prog, prog.corr, occ, live, obs)
    assert ref.call == 0                                                 # resolution ignores the hint
    faults = RS.validate(obs, ref, correspondence=prog.corr, expected_call_of_hint=lambda h: h)
    assert [f.kind for f in faults] == ["hint-mismatch"] and faults[0].hint == 99
    # a hint naming another REQUEST than the one the occurrence resolves to
    faults = RS.validate(obs, ref, expected_request_of_hint=lambda h: "r1")
    assert [f.kind for f in faults] == ["hint-mismatch"]


def test_unknown_site_and_ordinal_overflow_are_named_not_guessed():
    prog = _component()
    live = _live_one_request()
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.7.nowhere", slot="out", request="r0", position=1)
    ref = RS.resolve(prog, prog.corr, occ, live)
    assert ref.call is None and ref.problems and ref.problems[0].startswith("site-unknown")
    assert [f.kind for f in RS.validate(OB.ValueObservation(occurrence=occ), ref)] == ["site-unknown"]
    # two observations of a site that the Program calls once in the step
    o1 = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="model.layers.0.self_attn.qkv_proj", slot="0", request="r0", position=1)
    r1 = RS.resolve(prog, prog.corr, o1, live)
    RS.advance(live, OB.ValueObservation(occurrence=o1, hint_value_id=1, spec="Gemm_v1{N=8,K=8}"), r1)
    r2 = RS.resolve(prog, prog.corr, o1, live)
    assert r2.call is None and r2.problems[0].startswith("ordinal-out-of-range")


def test_unattributable_and_ambiguous_occurrences_are_refusals_by_name():
    steps = {0: {"rows": [("r0", 5), ("r1", 3)], "logits_rows": [("r1", 3), ("r0", 5)]}}      # a pure decode step with reordered logits rows
    hdr = OB.RecordHeader(schema=None, operand_refs_form="runs", n_instances=None, requests=[{"index": 0, "request_id": "r0", "prompt_len": 2},
                                                                                             {"index": 1, "request_id": "r1", "prompt_len": 2}],
                          n_input_nodes=3, params=["p0", "p1", "weights"], tp=None, digest=None)
    live = RS.LiveState.of_header(hdr, steps=steps)
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="m", slot="out", row=0, rows=2)
    ref = RS.resolve(None, None, occ, live)
    assert ref.request is None and ref.problems[0].startswith("ambiguous-request")
    occ = OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="m", slot="out", row=0, rows=7)
    ref = RS.resolve(None, None, occ, live)
    assert ref.request is None and ref.problems[0].startswith("unattributable")
    # a value reading two requests' prompts: non-separable, refused
    obs = OB.ValueObservation(occurrence=OB.RuntimeOccurrence(step=0, invocation=0, rank=0, site="m", slot="out"), param_reads=(0, 1))
    ref = RS.resolve(None, None, obs.occurrence, live, obs)
    assert ref.problems[0].startswith("reads-several-requests")
    assert [f.kind for f in RS.validate(obs, ref)] == ["reads-several-requests"]


def test_meta_channel_disagreeing_with_the_row_count_rule_is_a_request_mismatch(tmp_path):
    d, rows, steps = _write_record(tmp_path)
    # forge one stamped row: the runtime says B where the token rows say A
    lines = open(f"{d}/instances.jsonl").read().splitlines()
    for j, ln in enumerate(lines[1:], 1):
        o = json.loads(ln)
        if o["spec"].startswith("Gemm") and o["step"] == 1 and o["row"] == 0:
            o["request"] = B
            lines[j] = json.dumps(o)
            break
    open(f"{d}/instances.jsonl", "w").write("\n".join(lines) + "\n")
    att = RS.attribute_record(d)
    assert att.faults == {"request-mismatch": 1} and att.disagreements == 1, att.fault_examples


def test_collectors_own_name_disagreeing_with_the_site_is_a_layer_mismatch(tmp_path):
    d, rows, steps = _write_record(tmp_path, bad_layer_name=True)
    att = RS.attribute_record(d)
    assert att.faults == {"layer-mismatch": 1}, att.fault_examples


def test_mutated_view_in_a_record_is_a_fault_on_the_stream(tmp_path):
    d, rows, steps = _write_record(tmp_path, mutate_view=True)
    att = RS.attribute_record(d)
    assert att.faults == {"view-mutated": 1}, att.fault_examples
