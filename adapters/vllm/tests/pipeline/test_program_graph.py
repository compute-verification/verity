"""A row's Program as a graph (`verity_vllm.pipeline.program_graph`) on the replay's honest test Program: every Call placed in its
module body, reads counted as edges, identical bodies hash alike, units and exported VUs reconciled with a run's records."""
from __future__ import annotations

import copy
import gzip
import json

from tests.check.test_sampled_replay import H, I, _program
from tests.pipeline.test_vu_export import _export, _world
from verity_vllm.pipeline import program_graph as PG
from verity_vllm.pipeline import vu_export as VX


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


def _more_specs(inst: dict, spec: str, others: list[str]) -> dict:
    """The Program with its first row of `spec` repeated once under each of `others`: one module, more specializations."""
    rows = inst["rows"]
    row = next(r for r in rows if r["spec"] == spec)
    return dict(inst, rows=rows + [dict(copy.deepcopy(row), i=len(rows) + k, spec=s) for k, s in enumerate(others)])


def test_a_static_of_more_values_than_the_limit_varies_and_leaves_its_groups_template_parameters(tmp_path):
    """A module whose Gemm takes more than `VARYING_MAX` values of K is one group that states K's range and its Calls at each
    value, with no K among its statics or its template's parameters; at one value the K is a static of both."""
    inst, _w = _program(attn=True)
    logits, dot = "Gemm_v1{K=16,N=8}", "AmpereBF16TcDot16_v2"
    fixed = next(x for x in PG.program_graph([_write(inst, tmp_path / "fixed.json.gz")])["groups"] if x["spec"] == logits)
    assert fixed["varying"] == {} and fixed["varying_calls"] is None
    assert fixed["statics"] == {"K": 16, "N": 8} and fixed["template_params"] == {"K": 16, "dot": dot}
    ks = range(17, 17 + PG.VARYING_MAX)                    # with the 16: one value more than the limit
    g = PG.program_graph([_write(_more_specs(inst, logits, [f"Gemm_v1{{K={k},N=8}}" for k in ks]), tmp_path / "instances.json.gz")])
    (x,) = [x for x in g["groups"] if x["varying"]]
    assert x["module"] == fixed["module"] and x["definition"] == "Gemm_v1" and x["spec"] is None and x["specs"] == PG.VARYING_MAX + 1
    assert x["varying"] == {"K": [16, 16 + PG.VARYING_MAX]} and x["statics"] == {"N": 8} and x["template_params"] == {"dot": dot}
    assert x["varying_calls"] == {"16": fixed["calls"], **{str(k): 1 for k in ks}} and x["calls"] == fixed["calls"] + PG.VARYING_MAX


def test_a_varying_groups_word_rule_is_partitioned_at_every_value_and_totals_each_values_calls(tmp_path):
    """Attention's T at one value more than `VARYING_MAX`: the group's word rule states T's range with the smallest value's
    figures as its minimum, is partitioned at every recorded value (nothing interpolated), and totals each value's rule over that
    value's Calls: the groups the values up to the limit make one by one, and the largest value's one Call."""
    inst, _w = _program(attn=True)
    spec = "Attention_v3{{T={},NH=3,KVH=1,D=16,BN=16}}".format
    last = PG.VARYING_MAX + 1

    def attention(upto: int, name: str) -> list[dict]:
        doc = _more_specs(inst, spec(1), [spec(t) for t in range(5, upto + 1)])                # the Program's own T are 1 to 4
        return [x for x in PG.with_word_rules(PG.program_graph([_write(doc, tmp_path / name)]))["groups"] if x["definition"] == "Attention_v3"]

    each = {x["statics"]["T"]: x["q_word_v1"] for x in attention(PG.VARYING_MAX, "limit.json.gz")}
    assert sorted(each) == list(range(1, last))
    (x,) = attention(last, "instances.json.gz")
    q = x["q_word_v1"]
    assert x["varying"] == {"T": [1, last]} and q["per_call"]["T"] == {"min": 1, "max": last}
    assert q["partitioned_at"] == last and q["interpolated"] is False
    for f in ("units_per_call", "gates", "committed_interior_words"):
        assert q["per_call"][f]["min"] == each[1]["per_call"][f] < q["per_call"][f]["max"]
    for f, per_call in (("units", "units_per_call"), ("gates", "gates"), ("committed_interior_words", "committed_interior_words")):
        assert q[f] == sum(r[f] for r in each.values()) + q["per_call"][per_call]["max"]


def test_units_of_a_stratum_at_no_module_go_to_its_definitions_unplaced_groups_by_their_calls():
    """A stratum whose op path is no module (the sampler's rows): its units go to the groups of its Definition that no stratum
    placed, in proportion to their Calls, the remainder to the largest remainder first; a group a stratum placed takes none."""
    acc = PG._Acc()
    acc.groups = {("m.a", "g0"): {"family": "Gemm_v1", "statics": {"N": 8}, "calls": 2},
                  ("m.b", "g1"): {"family": "Gemm_v1", "statics": {"N": 8}, "calls": 1},
                  ("m.c", "g2"): {"family": "Gemm_v1", "statics": {"N": 8}, "calls": 4},
                  ("m.d", "g3"): {"family": "Gemm_v1", "statics": {"N": 16}, "calls": 9},
                  ("m.e", "g4"): {"family": "SiluMul_v1", "statics": {}, "calls": 9}}
    strata = {"Gemm_v1{K=16,N=8}|r0|0|decode|c|x": {"population": 7},            # placed: m.c is a module with that Definition
              "Gemm_v1{K=16,N=8}|r0|0|decode|sampler|x": {"population": 5},      # no module: over m.a (2 Calls) and m.b (1)
              "Gemm_v1{K=16,N=8}|r0": {"population": 11}}                        # not a stratum key
    assert PG.units_from_strata(strata, acc, "m.") == {("m.c", "g2"): 7, ("m.a", "g0"): 3, ("m.b", "g1"): 2}


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
    assert VX.template_of("Gemm_v1{K=2048,N=3072}") == ("GemmCoordinate", {"K": 2048, "dot": "AmpereBF16TcDot16_v2"}, 3072)
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


def test_catalog_deduplicates_specializations_across_rows(tmp_path):
    import gzip
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(inst, tmp_path / "instances.json.gz")])
    for r in ("rowA", "rowB"):
        (tmp_path / "graphs").mkdir(exist_ok=True)
        (tmp_path / "graphs" / f"{r}.program.json").write_text(json.dumps(g))
    ix = PG.catalog(tmp_path / "graphs")
    assert ix and all(set(e["rows"]) == {"rowA", "rowB"} for e in ix.values())
    shards = {}
    for p in (tmp_path / "graphs" / "definitions").glob("*.json.gz"):
        with gzip.open(p, "rt") as f:
            shards.update(json.load(f))
    assert len(shards) == len({e["digest"] for e in ix.values()})
    assert all(PG.definition_digest({k: x for k, x in v["definition"].items() if k != "fold"}) == h for h, v in shards.items())
    ops = json.loads((tmp_path / "graphs" / "rowA.ops.json").read_text())["ops"]
    for o in ops:                                          # a varying group: every value of its range; else its one spec
        want = 1
        for lo, hi in o["varying"].values():
            want *= hi - lo + 1
        assert o["specs_enumerated"] == want == len(o["definitions"])
    assert all(d["digest"] in shards for o in ops for d in o["definitions"] if d["id"])
    again = json.loads((tmp_path / "graphs" / "rowA.program.json").read_text())
    assert "definitions" not in again and again["definitions_index"]


def test_edges_carry_the_words_each_read_takes(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(inst, tmp_path / "instances.json.gz")])
    gid = {x["id"]: x for x in g["groups"]}
    runs = sum(hi - lo for r in inst["rows"] for grp in r["args"] for k, _a, lo, hi in grp if k == "n")
    assert sum(e["words"] for e in g["edges"]) == runs
    down = next(e for e in g["edges"] if gid[e["from"]]["definition"] == "SiluMul_v1" and gid[e["to"]]["module"].endswith("down_proj"))
    assert down["words_per_read"] == I and down["ports"] == [{"param": "x", "port": "out", "words": down["words"]}]
    assert down["bits"] == 16 * down["words"]
    qkv = [e for e in g["edges"] if gid[e["from"]]["module"].endswith("qkv_proj")]
    assert qkv and all(e["words_per_read"] < gid[e["from"]]["statics"]["N"] for e in qkv)     # each reader takes a slice of q | k | v


def test_param_inputs_name_the_parameter_each_prescribed_input_feeds(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.program_graph([_write(inst, tmp_path / "instances.json.gz")])
    assert any(k == "p" for r in inst["rows"] for grp in r["args"] for k, *_ in grp)
    down = next(x for x in g["groups"] if x["module"].endswith("down_proj"))
    assert list(down["param_inputs"]) == ["w"] and all(k.startswith("weights:") for k in down["param_inputs"]["w"])
    assert set(down["param_inputs"]["w"]) == set(down["prescribed_reads"])
    assert all(x["param_inputs"] is not None for x in g["groups"])


def test_word_rules_per_group_and_row(tmp_path):
    inst, _w = _program(attn=True)
    g = PG.with_word_rules(PG.program_graph([_write(inst, tmp_path / "instances.json.gz")]))
    q = g["q_word_v1"]
    assert q["query"] == "Q_word_v1{X=16,W=32,R=no-recompute}" and q["units"] == sum(x["q_word_v1"]["units"] for x in g["groups"] if "units" in x["q_word_v1"])
    gemm = next(x for x in g["groups"] if x["definition"] == "Gemm_v1")
    assert gemm["q_word_v1"]["units"] == gemm["calls"] * gemm["statics"]["N"] and gemm["q_word_v1"]["committed_interior_words"] == 0
    attn = next(x for x in g["groups"] if x["definition"] == "Attention_v3")
    assert attn["q_word_v1"]["committed_interior_words"] > 0
    assert attn["varying_calls"] is None or sum(attn["varying_calls"].values()) == attn["calls"]
