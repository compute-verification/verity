"""A row's Program as a graph (`verity_vllm.pipeline.program_graph`) on the replay's honest test Program: every Call placed in its
module body, reads counted as edges, identical bodies hash alike, units and exported VUs reconciled with a run's records."""
from __future__ import annotations

import copy
import gzip
import json

from verity_vllm.pipeline import program_graph as PG
from verity_vllm.pipeline import vu_export as VX
from tests.check.test_sampled_replay import H, I, _program
from tests.pipeline.test_vu_export import _export, _world


def _write(doc: dict, path) -> str:
    with gzip.open(path, "wt") as f:
        json.dump(doc, f)
    return str(path)


def _two_layer(inst: dict) -> dict:
    """The one-layer Program with every layer-0 row repeated as layer 1 (reading layer 1's own producers): two identical layers."""
    rows = inst["rows"]
    layer = [r for r in rows if ".layers.0." in r["name"]]
    new_id = {r["i"]: len(rows) + j for j, r in enumerate(layer)}
    extra = []
    for r in layer:
        r2 = copy.deepcopy(r)
        r2["i"] = new_id[r["i"]]
        r2["name"] = r["name"].replace(".layers.0.", ".layers.1.")
        r2["args"] = [[[k, new_id.get(a, a) if k == "n" else a, lo, hi] for k, a, lo, hi in g] for g in r["args"]]
        extra.append(r2)
    return dict(inst, rows=rows + extra)


def test_every_call_is_in_its_module_body_and_every_read_is_an_edge(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(inst, tmp_path / "instances.json.gz")])
    assert sum(x["calls"] for x in g["groups"]) == len(inst["rows"]) == sum(m["calls"] for m in g["modules"])
    reads = sum(1 for r in inst["rows"] for grp in r["args"] for sl in grp if sl[0] == "n")
    assert sum(e["reads"] for e in g["edges"]) <= reads and sum(e["reads"] for e in g["edges"]) > 0
    mods = {m["path"]: m for m in g["modules"]}
    assert mods["model.model.layers.0.mlp.down_proj"]["parent"] == "model.model.layers.0.mlp"
    assert mods["model.model.layers.0"]["body"] is False and mods["model.model.layers.0.mlp.down_proj"]["body"] is True
    gid = {x["id"]: x for x in g["groups"]}
    attn = next(x for x in g["groups"] if x["definition"] == "Attention_v3")
    assert "T" in attn["varying"] or attn["spec"] is not None
    gemm = [x for x in g["groups"] if x["definition"] == "Gemm_v1"]
    assert all(x["template"] == "GemmCoordinate" and x["instances"] == x["calls"] * x["statics"]["N"] for x in gemm)
    down = next(x for x in gemm if x["module"].endswith("down_proj"))
    assert down["statics"] == {"K": I, "N": H}
    assert any(gid[e["from"]]["definition"] == "SiluMul_v1" and e["to"] == down["id"] for e in g["edges"])
    assert any(k.startswith("weights:") for k in down["prescribed_reads"])


def test_identical_layers_fold_by_structure(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(_two_layer(inst), tmp_path / "instances.json.gz")])
    mods = {m["path"]: m for m in g["modules"]}
    assert mods["model.model.layers.0"]["subtree_hash"] == mods["model.model.layers.1"]["subtree_hash"]
    assert mods["model.model.layers.0.mlp"]["subtree_hash"] == mods["model.model.layers.1.mlp"]["subtree_hash"]
    assert mods["model.model.layers.0"]["order"] == 0 and mods["model.model.layers.1"]["order"] == 1
    again = PG.program_graph([_write(_two_layer(inst), tmp_path / "again.json.gz")])
    assert [m["subtree_hash"] for m in again["modules"]] == [m["subtree_hash"] for m in g["modules"]]


def test_units_and_exported_vus_reconcile_with_the_run(tmp_path):
    world = _world(attn=True)
    s = _export(world, tmp_path / "exp", limits=VX.Limits(per_family=64, workers=1))
    from verity_vllm.check.replay import driver as SR
    bmap, committed = world.bmap_and_reader()
    rep = SR.sampled_replay({"r0": world.inst}, bmap, committed,
                            lambda f, lo, hi: world.weights[world.woff[f][0] + lo: world.woff[f][0] + hi], per_stratum=1, seed=1, workers=0)
    g = PG.program_graph([_write(world.inst, tmp_path / "instances.json.gz")], strata=rep["strata"], vus_jsonl=tmp_path / "exp" / "vus.jsonl")
    assert sum(x["units_in_run"] or 0 for x in g["groups"]) == rep["population"]["vus"]
    r0 = sum(1 for line in (tmp_path / "exp" / "vus.jsonl").read_text().splitlines() if json.loads(line)["request"] == "r0")
    assert 0 < sum(x["exported_vus"] or 0 for x in g["groups"]) <= s["population"]["pool"] and r0 > 0


def test_template_of_is_the_exporters_cut():
    assert VX.template_of("Gemm_v1{K=2048,N=3072}") == ("GemmCoordinate", {"K": 2048, "dot": "AmpereBF16TcDot16_v1"}, 3072)
    assert VX.template_of("RoPE_v1{NHEADS=8,D=64}") == ("RoPEHead", {"D": 64}, 8)
    t, p, m = VX.template_of("Attention_v3{T=5,NH=32,KVH=8,D=64,BN=128}")
    assert (t, p["D"], m) == ("AttentionHead", 64, 32)
    assert VX.template_of("SiluMul_v1{I=8192}") == ("SiluMul_v1", {"I": 8192}, 1)


def test_sampler_vus_are_placed_by_their_requests_program(tmp_path):
    world = _world(attn=True)
    _export(world, tmp_path / "exp", limits=VX.Limits(per_family=64, workers=1))
    paths = []
    for d in ("reqA", "reqB"):
        (tmp_path / d).mkdir()
        paths.append(_write(world.inst, tmp_path / d / "instances.json.gz"))
    vus = tmp_path / "exp" / "vus.jsonl"
    exported = sum(1 for line in vus.read_text().splitlines() if json.loads(line)["result"] is True)
    unmapped = PG.program_graph(paths, vus_jsonl=vus)
    mapped = PG.program_graph(paths, vus_jsonl=vus, request_programs={"r0": str(tmp_path / "reqA"), "r1": str(tmp_path / "reqB")})
    placed = lambda g: sum(x["exported_vus"] or 0 for x in g["groups"])  # noqa: E731
    assert placed(mapped) == exported > placed(unmapped)
    store = PG.request_programs_of(tmp_path / "exp" / "store")
    assert set(store) <= {"r0", "r1"}


def test_definitions_carry_each_specialization_down_to_primitives_with_fold_hints(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(inst, tmp_path / "instances.json.gz")])
    D = g["definitions"]
    gemm = next(x for x in g["groups"] if x["definition"] == "Gemm_v1")
    root = D[gemm["definition_ids"][0]]
    assert root["form"] == "composite" and root["body"]["nodes"][0]["form"] == "batch"
    assert root["body"]["nodes"][0]["n"] == gemm["statics"]["N"]
    dot = next(v for k, v in D.items() if k.startswith(f"DotBf16_v1{{K={H}}}"))
    assert len(dot["body"]["nodes"]) == H // 16 + 1
    if H // 16 >= PG.FOLD_MIN:
        assert dot["fold"] and dot["fold"][0]["chained"] is True
    assert any(v["form"] == "primitive" for v in D.values())
    attn = next(x for x in g["groups"] if x["definition"] == "Attention_v3")
    assert len(attn["definition_ids"]) in (1, 2) and all(i in D for i in attn["definition_ids"] if i)


def test_fold_hints_mark_chains_and_parallel_runs():
    chain = [{"fn": "A", "form": "call", "args": [{"refs": {"affine": ["n", k - 1, 0, 1]}}] if k else []} for k in range(5)]
    par = [{"fn": "B", "form": "call", "args": [{"refs": {"affine": ["p", 0, k, k + 1]}}]} for k in range(4)]
    h = PG.fold_hints(chain + par)
    assert h == [{"start": 0, "count": 5, "fn": "A", "form": "call", "chained": True},
                 {"start": 5, "count": 4, "fn": "B", "form": "call", "chained": False}]
