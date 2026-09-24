"""GEN ov-batch lane — the persisted-candidate provenance diagnostic (`bench/gen_batch_candidate.py`) on a synthetic frozen
candidate: two requests, the second admitted with a 2-token cached prefix produced by the first (prefix caching), plus a
discarded partial-prefill sample variant (chunked prefill).  Torch-free; positive AND negative alias-obligation checks."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.correspondence import batch_candidate as C

R0, R1 = "req-0000-aaaa", "req-1111-bbbb"


def _inst(node, spec, row, rows, operand_refs, step=0, hint=""):
    return {"node": node, "step": step, "event": 100 + node, "resolution": "compute", "pattern": spec.split("_")[0].lower(), "spec": spec,
            "row": row, "rows": rows, "name": hint, "hint": hint, "operand_views": [], "output_views": [], "operand_refs": operand_refs}


def _write_candidate(d: Path, *, r1_prompt=(5, 6, 9), r1_logits_pos=2, alias_ret=True) -> Path:
    # rows at step 0: r0 positions 0..2 (rows 0..2), r1 position 2 only (row 3; num_computed 2, blocks shared with r0)
    P0, P1, W = 0, 1, 2   # fold parameter indices: prompt_0, prompt_1, weights (the fold's order, = the root circuit's input nodes)
    insts = [_inst(i, "Embedding_v1{V=8,H=4}", i, 4, [[["p", P0, i, i + 1]], [["p", W, 0, 32]]]) for i in range(3)]
    insts.append(_inst(3, "Embedding_v1{V=8,H=4}", 3, 4, [[["p", P1, 2, 3]], [["p", W, 0, 32]]]))
    insts += [_inst(4 + i, "RoPE_v1{NHEADS=1,D=4}", i, 4, [[["n", i, 0, 4]]], hint=f"t={i if i < 3 else 2}/layers.0/rope_k") for i in range(4)]
    insts.append(_inst(8, "Attention_v3{T=3,NH=1,KVH=1,D=4,BN=128}", 0, 1,
                       [[["n", 7, 0, 4]], [["n", 4, 0, 4], ["n", 5, 0, 4], ["n", 7, 0, 4]], [["n", 4, 0, 4], ["n", 5, 0, 4], ["n", 7, 0, 4]]],
                       hint="t=2/layers.0/attn"))
    insts.append(_inst(9, "Attention_v3{T=3,NH=1,KVH=1,D=4,BN=128}", 0, 1,
                       [[["n", 6, 0, 4]], [["n", 4, 0, 4], ["n", 5, 0, 4], ["n", 6, 0, 4]], [["n", 4, 0, 4], ["n", 5, 0, 4], ["n", 6, 0, 4]]],
                       hint="t=2/layers.0/attn"))
    insts.append(_inst(10, "TokenSelect_v1{V=8}", 0, 2, [[["n", 9, 0, 4]]], hint="sample"))
    insts.append(_inst(11, "TokenSelect_v1{V=8}", 1, 2, [[["n", 8, 0, 4]]], hint="sample"))
    with (d / "instances.jsonl").open("w") as f:
        f.write(json.dumps({"schema": "verity-gen/instances/v1", "digest": "d" * 64, "n_instances": len(insts)}) + "\n")
        for m in insts:
            f.write(json.dumps(m) + "\n")
    # the root circuit (codec v1): the fold's parameters are the first n_input_nodes nodes (Input32 batches for the prompts, Input16 for
    # the weights) in fold order; root-circuit node of instance k = n_input_nodes + k; `types` / `callees` are interning tables
    NIN = 3
    types = [["Array", 3, ["Value", 32]], ["Array", 32, ["Value", 16]], ["Value", 32]]
    callees = ["Input32_v1", "Input16_v1"]
    nodes = [{"args": [], "axes": [], "fn": 0, "form": "batch", "n": 3, "out": 0}, {"args": [], "axes": [], "fn": 0, "form": "batch", "n": 3, "out": 0},
             {"args": [], "axes": [], "fn": 1, "form": "batch", "n": 32, "out": 1}]
    nodes += [{} for _ in range(12)] + [{"args": [{"refs": {"affine": ["n", NIN + 10, 0, 1]}}]}]     # node 15 carries the alias target
    ret = {"concat": [({"alias": [NIN + 12, 0]} if alias_ret else {"affine": ["n", NIN + 10, 0, 1]}), {"affine": ["n", NIN + 11, 0, 1]}]}
    body = {"params": [], "ret_type": ["Tuple", [["Array", 1, ["Value", 32]], ["Array", 1, ["Value", 32]]]], "nodes": nodes, "ret": ret}
    prog = {"schema": "verity-gen/program/v1", "digest": "d" * 64, "n_input_nodes": NIN, "n_instances": len(insts),
            "program": {"schema": "verity-ir/descriptor/v1", "root": "Root[Captured_v1{PROFILE=x,WORKLOAD=y}]_v1", "types": types, "callees": callees,
                        "definitions": {"Root[Captured_v1{PROFILE=x,WORKLOAD=y}]_v1": {"form": "composite", "id": "Root", "bindings": {}, "body": body},
                                        "Attention_v3{T=3,NH=1,KVH=1,D=4,BN=128}": {"form": "primitive", "id": "Attention_v3{T=3,NH=1,KVH=1,D=4,BN=128}",
                                                                                   "params": [["q", ["Array", 4, ["Value", 16]]], ["kc", ["Array", 12, ["Value", 16]]],
                                                                                              ["vc", ["Array", 12, ["Value", 16]]]], "ret": ["Array", 4, ["Value", 16]]}}}}
    (d / "program.json").write_text(json.dumps(prog))
    attr = {"steps": [{"step": 0, "rows": 4, "requests": {R1: {"row_index": "3", "positions": "2", "num_computed": 2, "seq_len": 3},
                                                          R0: {"row_index": "0-2", "positions": "0-2", "num_computed": 0, "seq_len": 3}},
                       "logits_rows": [[R0, 2], [R1, r1_logits_pos]], "sampled": [11, 22]}]}
    (d / "attribution.json").write_text(json.dumps(attr))
    (d / "fold_summary.json").write_text(json.dumps({"ret_type": "Tuple<Array<1,Value<32>>,Array<1,Value<32>>>",
                                                     "params": [["prompt_0", "Array<3,Value<32>>"], ["prompt_1", "Array<3,Value<32>>"], ["weights", "{...}"]],
                                                     "requests": [{"request_id": R0, "prompt_len": 3, "max_tokens": 1, "n_tokens": 1},
                                                                  {"request_id": R1, "prompt_len": 3, "max_tokens": 1, "n_tokens": 1}]}))
    wl = d / "workload.json"
    wl.write_text(json.dumps({"case": "B0", "requests": [{"request_id": R0, "index": 0, "prompt_len": 3, "max_tokens": 1, "prompt_token_ids": [5, 6, 7]},
                                                          {"request_id": R1, "index": 1, "prompt_len": 3, "max_tokens": 1, "prompt_token_ids": list(r1_prompt)}]}))
    return wl


def test_prefix_alias_positive(tmp_path: Path):
    wl = _write_candidate(tmp_path)
    d = C.analyze(tmp_path, wl)
    cons = {p["param"]: p for p in d["prompt_consumption"]}
    assert cons["prompt_0"]["consumed"] == 3 and cons["prompt_0"]["never_consumed"] == "-"
    assert cons["prompt_1"]["consumed"] == 1 and cons["prompt_1"]["never_consumed"] == "0..1 (2)"
    first = {(a["step"], a["request"]): a for a in d["attention_first_row"]}
    kc = first[(0, 1)]["operands"]["kc"]
    assert kc == {"req0@step0:RoPE_v1": "0..1 (2)", "req1@step0:RoPE_v1": "2 (1)"}       # r1's K over the cached prefix is r0's output
    assert first[(0, 0)]["operands"]["kc"] == {"req0@step0:RoPE_v1": "0..2 (3)"}
    up = {s["request"]: s["prompt_positions"] for s in d["sampled_token_upstream_step0"]}
    assert up["req1@2"] == {"prompt_0": "0..1 (2)", "prompt_1": "2 (1)"} and up["req0@2"] == {"prompt_0": "0..2 (3)"}
    [a] = d["admitted_with_cache"]
    assert a["request"] == 1 and a["num_computed"] == 2 and a["kv_producers_over_cached_prefix"] == ["req0"]
    assert a["obligation"] == [{"alias": "prompt_1[0..1] == prompt_0[0..1]", "holds_in_workload": True, "first_difference": None}]
    ret = {r["request"]: r for r in d["returned"] if "request" in r}
    assert ret[0]["committed"] == 1 and ret[1]["committed"] == 1 and not ret[1]["discarded"]
    assert d["row_map_check_step0"]["agree"] == 4 and not d["row_map_check_step0"]["disagree"]
    md = C.render(d)
    assert "prefix-alias obligation" in md and "DISCARDED" not in md


def test_prefix_alias_negative_and_discarded_sample(tmp_path: Path):
    # the same Program with a request record whose cached prefix differs: only the obligation tells the twins apart
    wl = _write_candidate(tmp_path, r1_prompt=(5, 8, 9), r1_logits_pos=1, alias_ret=False)
    d = C.analyze(tmp_path, wl)
    [a] = d["admitted_with_cache"]
    assert a["obligation"][0]["holds_in_workload"] is False and a["obligation"][0]["first_difference"] == 1
    # a sample at position 1 < prompt_len - 1 = 2 is a partial-prefill sample the engine discards, yet the Program returns it
    ret = {r["request"]: r for r in d["returned"] if "request" in r}
    assert ret[1]["committed"] == 0 and len(ret[1]["discarded"]) == 1
    assert ret[1]["discarded"][0]["position"] == 1 and ret[1]["discarded"][0]["consumed_by_instances"] == 0
    assert "DISCARDED by the engine but returned by the Program" in C.render(d)


def test_ret_runs_resolves_aliases_and_concat():
    body = {"nodes": [{}, {"args": [{"refs": {"affine": ["n", 7, 0, 2]}}]}], "ret": None}
    assert C._ret_runs(body, {"concat": [{"alias": [1, 0]}, {"affine": ["p", 2, 4, 3]}]}) == [("n", 7, 0, 2), ("p", 2, 4, 7)]
    with pytest.raises(IndexError):
        C._ret_runs(body, {"alias": [5, 0]})
    assert C._expand("0-2,5") == [0, 1, 2, 5] and C._expand("-") == [] and C._ranges([3, 4, 5, 9]) == "3..5,9 (4)"


# ---------------------------------------------------------------------------------------------------------------------
# SHARED-CODE (verity_vllm/observe/fold.py): served vs discarded samples under chunked prefill — the fold classifies each
# SampleEnd row by vLLM's own rule `StepStart.seq_lens[r] >= prompt_len_r`; the toy log of test_fold_synthetic prefilled
# in two chunks is the reproducer (b0_chunked_200p: 15 sampled, 12 served, 3 discarded at steps 0-2).
# ---------------------------------------------------------------------------------------------------------------------
import dataclasses

from verity_vllm.observe.events import StepLink, StepStart
from verity_vllm.observe.fold import fold, instance_nodes
from tests.observe import test_fold_synthetic as TFS
from verity.ir.types import Array, Value

memory = TFS.memory  # the fixture (real memory model when importable, else the in-test stub)


def _chunked_toy_log(seq_len_at_step0: int | None = None):
    """PROMPT [5, 7] prefilled in two chunks: step 0 = position 0 only (seq_len 1 < prompt_len 2 -> the sample, token 11, is
    dropped by the engine), step 1 = position 1 (prompt complete -> served, SAMPLED[0]), step 2 = decode of the served token
    (StepLink to step 1).  `seq_len_at_step0` overrides the step-0 `seq_lens` to show the rule is driven by that fact alone."""
    ids = TFS.Ids()
    evs = TFS.preamble()
    step0 = TFS.step_events(0, [0], ids, 20, 11)
    if seq_len_at_step0 is not None:
        step0 = [dataclasses.replace(e, seq_lens={TFS.REQ: seq_len_at_step0}) if isinstance(e, StepStart) else e for e in step0]
    evs += step0
    evs += [e for e in TFS.step_events(1, [1], ids, 40, TFS.SAMPLED[0]) if not isinstance(e, StepLink)]   # a prompt chunk: rows bind by position
    evs += TFS.step_events(2, [2], ids, 60, TFS.SAMPLED[1])
    return evs


def test_fold_discards_partial_prefill_samples(memory):
    r = fold(_chunked_toy_log(), TFS.toy_profile(), strict=True)
    assert not r.errors and r.report.unsupported == 0
    # served: steps 1 and 2 only; the step-0 sample is recorded as discarded with its logits position
    assert r.sampled == {TFS.REQ: [(1, TFS.SAMPLED[0]), (2, TFS.SAMPLED[1])]}
    assert r.discarded == {TFS.REQ: [(0, 11, 0)]}
    assert r.body.ret_type == Array(2, Value(32))
    ts = {m["step"]: m["node"] for m in instance_nodes(r, spec_prefix="TokenSelect_v1")}
    assert sorted(ts) == [0, 1, 2]                                   # the discarded sample's TokenSelect instance IS in the Program
    assert list(r.body.ret) == [("n", ts[1], 0), ("n", ts[2], 0)]    # ... but is not returned
    consumers = [k for k, nd in enumerate(r.body.nodes) if any(("n", ts[0], 0) in list(a.refs) for a in nd.args)]
    assert consumers == []                                           # ... and nothing reads it (no StepLink fed it back)
    emb2 = instance_nodes(r, spec_prefix="Embedding_v1", step=2)[0]["node"]
    assert list(r.body.nodes[emb2].args[0].refs) == [("n", ts[1], 0)]  # the decode row carries the SERVED token
    assert any("discarded by the engine" in n for n in r.notes)


def test_fold_served_rule_is_seq_lens_not_positions(memory):
    # negative: the same events with StepStart.seq_lens claiming the prompt complete after step 0 -> the step-0 sample is SERVED
    # (seq_len 2 >= prompt 2 + 0 served), although position 0 alone was fed: the decision is seq_lens', not the positions'.
    # [R9 ref 2026-09-18] the rule is vLLM's full one, `seq_len >= prompt_len + served so far` (`num_tokens` counts the output
    # tokens; identical to `seq_len >= prompt_len` on every run without preemption): step 1 then claims seq_len 2 < 2 + 1 and
    # its sample is discarded (the engine would drop a chunk that ends inside already-generated tokens); step 2's is served.
    r = fold(_chunked_toy_log(seq_len_at_step0=2), TFS.toy_profile(), strict=True)
    assert r.discarded == {TFS.REQ: [(1, TFS.SAMPLED[0], 1)]} and [s for s, _ in r.sampled[TFS.REQ]] == [0, 2]
    assert r.body.ret_type == Array(2, Value(32))
    # and the single-chunk toy log (prefill completes at step 0) is unchanged by the rule
    r0 = fold(TFS.toy_log(), TFS.toy_profile(), strict=True)
    assert r0.discarded == {} and r0.sampled == {TFS.REQ: [(0, 9), (1, 3)]} and r0.body.ret_type == Array(2, Value(32))
