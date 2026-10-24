"""The serving view (`verity_vllm.pipeline.serving_view`) on the replay's honest test Program: steps by dataflow, the leaves partition every
Call exactly once, KV-cache reads and tokens cross steps, co-scheduled requests of one Program keep their own leaves, TP2 collectives
pair the ranks, and a Build whose Calls do not tile its address map is refused."""
from __future__ import annotations

import copy
import gzip
import json
from collections import Counter

import pytest

from verity_vllm.pipeline import serving_view as SV
from verity.ml.definitions import word as W
from verity_vllm.query.program_view import spec_base, spec_statics
from tests.check.test_sampled_replay import H, T, _program


def _write(doc: dict, path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        json.dump(doc, f)
    return str(path)


def _gates(inst: dict) -> int:
    return sum(W.specialization(spec_base(r["spec"]), spec_statics(r["spec"])).gates for r in inst["rows"])


def _selects(inst: dict) -> list[int]:
    return [r["i"] for r in inst["rows"] if r["spec"].startswith("TokenSelect_v1")]


def _tp(inst: dict) -> dict:
    """The Program with an `AllReduce2_v1{N=H}` after every down_proj row (its Call run + its slice of a new input `peers_0`), read
    by the down_proj's consumers in its place: one rank's shard of a TP2 row, both ranks the same shape."""
    rows, out, new_id, redirect, lo = inst["rows"], [], {}, {}, 0
    for r in rows:
        r2 = copy.deepcopy(r)
        r2["i"] = len(out)
        new_id[r["i"]] = r2["i"]
        r2["args"] = [[[k, redirect.get(a, new_id.get(a, a)) if k == "n" else a, x, y] for k, a, x, y in g] for g in r["args"]]
        out.append(r2)
        if ".mlp.down_proj/" in r["name"]:
            ar = {"i": len(out), "spec": f"AllReduce2_v1{{N={H}}}", "form": "call", "name": r["name"].split("/")[0] + "/all_reduce2[0]/all_reduce",
                  "args": [[["n", r2["i"], 0, H], ["p", 2, lo, lo + H]]]}
            out.append(ar)
            redirect[r["i"]] = ar["i"]
            lo += H
    rows_n = lo // H
    return dict(inst, params=[*inst["params"], ["peers_0", f"Array<{rows_n},Array<{H},Value<16>>>"]], rows=out)


def test_steps_by_dataflow_partition_every_call_and_kv_reads_cross_steps(tmp_path):
    inst, _w = _program(attn=True)
    _write(inst, tmp_path / "b" / "build_request" / "instances.json.gz")
    v = SV.serving_view(tmp_path / "b")
    assert v["checks"]["ok"] and v["checks"]["partition"]["calls"] == len(inst["rows"])
    mems = [n for n in v["nodes"] if n["kind"] == "member"]
    assert [m["s"] for m in mems] == list(range(1 + T)) and sum(m["calls"] for m in mems) == len(inst["rows"])
    assert v["checks"]["members"]["build_request"]["sampler_rule"] == "sampling-event"
    assert v["checks"]["members"]["build_request"]["equal_body_order_segments"] == len(inst["rows"])
    leaves = [n for n in v["nodes"] if n["kind"] == "rank"]
    assert sum(n["calls"] for n in leaves) == len(inst["rows"])
    samplers = [n for n in leaves if n.get("spec", "").startswith("TokenSelect_v1")]
    assert [n["call"] for n in samplers] == _selects(inst)
    kinds = Counter(e["kind"] for e in v["edges"])
    assert kinds["token"] == T and kinds["state"] > 0 and kinds["backward"] == 0
    nodes = {n["id"]: n for n in v["nodes"]}
    assert nodes["n0"]["calls"] == len(inst["rows"]) and sum(n["calls"] for n in v["nodes"] if n["kind"] == "engine_step") == len(inst["rows"])
    state_to = {nodes[nodes[e["to"]]["parent"]]["path"] for e in v["edges"] if e["kind"] == "state"}
    assert state_to == {"model.model.layers.0.self_attn.attn"}
    decode = [m["structure_hash"] for m in mems[1:]]
    assert len(set(decode)) == 1 and mems[0]["structure_hash"] != decode[0]
    site = SV.site_tree(v)
    steps = site["tree"]["children"]
    assert "children" in steps[0] and "children" in steps[1] and steps[2]["collapsed"] and steps[2]["same_structure_as"] == 1


def test_co_scheduled_requests_of_one_program_keep_their_own_leaves(tmp_path):
    inst, _w = _program(attn=True)
    b = tmp_path / "b"
    _write(inst, b / "build_request" / "instances.json.gz")
    g = _gates(inst)
    reqs = [("r0", 0), ("r1", 1)]
    ends = [i + 1 for i in _selects(inst)]
    segs = list(zip([0, *ends[:-1]], ends))
    engine = []
    for e in range(2 + T):
        mem = [{"request_id": rid, "s": e - a, "phase": "prefill" if e == a else "decode", "nodes": list(segs[e - a])}
               for rid, a in reqs if 0 <= e - a <= T]
        engine.append({"e": e, "members": mem, "co_scheduled": len(mem) > 1, "mixed_prefill_decode": len({m["phase"] for m in mem}) > 1})
    (b / "build_workload").mkdir()
    (b / "build_workload" / "chronology.json").write_text(json.dumps({"engine_steps": engine}))
    amap = {"workload_digest": "w" * 64, "gates": 100 + 2 * g, "input_gates": 100, "shared": None,
            "requests": [{"request_id": "r0", "program_digest": inst["digest"], "gates": [100, 100 + g]},
                         {"request_id": "r1", "program_digest": inst["digest"], "gates": [100 + g, 100 + 2 * g]}]}
    (b / "build_workload" / "address_map.json").write_text(json.dumps(amap))
    v = SV.serving_view(b)
    assert v["checks"]["ok"] and v["checks"]["partition"]["calls"] == 2 * len(inst["rows"])
    assert [s["co_scheduled"] for s in v["steps"]] == [False, True, True, False]
    assert v["steps"][1]["mixed_prefill_decode"] and [m["request_id"] for m in v["steps"][1]["members"]] == ["r0", "r1"]
    nodes = {n["id"]: n for n in v["nodes"]}

    def request_of(nid: str) -> str:
        while nodes[nid]["kind"] != "member":
            nid = nodes[nid]["parent"]
        return nodes[nid]["request_id"]

    assert all(request_of(e["from"]) == request_of(e["to"]) for e in v["edges"])
    assert sum(1 for e in v["edges"] if e["kind"] == "token") == 2 * T
    m0 = [n for n in v["nodes"] if n["kind"] == "member" and n["request_id"] == "r1"]
    assert m0[0]["gate_range"][0] == 100 + g and m0[-1]["gate_range"][1] == 100 + 2 * g
    assert v["checks"]["chronology_ranges"] == {"agree": 2 * (1 + T), "differ": [], "not_recorded": 0, "ok": True}
    engine[1]["members"][0]["nodes"] = [segs[1][0], segs[1][1] + 1]
    (b / "build_workload" / "chronology.json").write_text(json.dumps({"engine_steps": engine}))
    with pytest.raises(ValueError, match="r0/1"):
        SV.serving_view(b)


def test_tp2_collectives_pair_the_ranks(tmp_path):
    inst, _w = _program(attn=True)
    tp = _tp(inst)
    for rank in (0, 1):
        _write(tp, tmp_path / "b" / f"rank{rank}" / "build_request" / "instances.json.gz")
    v = SV.serving_view(tmp_path / "b")
    assert v["checks"]["ok"] and v["checks"]["partition"]["calls"] == 2 * len(tp["rows"])
    nodes = {n["id"]: n for n in v["nodes"]}
    coll = [e for e in v["edges"] if e["kind"] == "collective"]
    reduces = sum(1 for r in tp["rows"] if r["spec"].startswith("AllReduce2"))
    assert sum(e["reads"] for e in coll) == 2 * reduces and sum(e["words"] for e in coll) == 2 * reduces * H
    assert all({nodes[e["from"]]["rank"], nodes[e["to"]]["rank"]} == {0, 1} for e in coll)
    assert v["checks"]["edges"]["collective_word_mismatch"] == 0
    bodies = [n for n in v["nodes"] if n["kind"] == "module" and n.get("body")]
    assert all(sorted(nodes[c]["rank"] for c in b["children"] if nodes[c]["kind"] == "rank") == [0, 1] for b in bodies)


def test_samplers_are_the_programs_returned_calls_when_the_descriptor_is_there(tmp_path):
    inst, _w = _program(attn=True)
    d = tmp_path / "b" / "build_request"
    _write(inst, d / "instances.json.gz")
    K = 2
    desc = {"root": "R", "definitions": {"R": {"body": {"nodes": [{}] * (K + len(inst["rows"])),
                                                         "ret": {"concat": [{"affine": ["n", K + i, 0, 1]} for i in _selects(inst)]}}}}}
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    v = SV.serving_view(tmp_path / "b")
    assert v["checks"]["members"]["build_request"]["sampler_rule"] == "return"
    assert [n["call"] for n in v["nodes"] if n["kind"] == "rank" and n.get("spec", "").startswith("TokenSelect")] == _selects(inst)


def test_calls_that_do_not_tile_the_address_map_are_refused(tmp_path):
    inst, _w = _program(attn=True)
    b = tmp_path / "b"
    _write(inst, b / "build_request" / "instances.json.gz")
    (b / "build_workload").mkdir()
    (b / "build_workload" / "chronology.json").write_text(json.dumps({"engine_steps": [
        {"e": s, "members": [{"request_id": "r0", "s": s, "phase": "prefill" if s == 0 else "decode"}], "co_scheduled": False,
         "mixed_prefill_decode": False} for s in range(1 + T)]}))
    g = _gates(inst)
    (b / "build_workload" / "address_map.json").write_text(json.dumps({"workload_digest": "w" * 64, "gates": 10 + g + 1, "input_gates": 10,
                                                                        "requests": [{"request_id": "r0", "program_digest": inst["digest"],
                                                                                      "gates": [10, 10 + g + 1]}]}))
    with pytest.raises(ValueError, match="partition"):
        SV.serving_view(b)
