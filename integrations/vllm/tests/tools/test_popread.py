"""`tools/popread.py` on a toy Program with one Call of each kind: one whose reads are public, one that reads every candidate and
selects one inside the unit by another Call's output, and one that reads an input whose binding is planned from committed values.
Also its step attribution from a workload Program's chronology, its planner counterfactual and its P_served reader."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "popread.py"
_spec = importlib.util.spec_from_file_location("popread", SCRIPT)
PR = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("popread", PR)
_spec.loader.exec_module(PR)


def _toy(tmp_path: Path) -> Path:
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.fp32 import Bf16Add
    from verity.ml.circuits.ops.scalar import I32Add
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.types import Array, Value
    from verity_vllm.program.registry.prims import gather_bf16

    def signature(S):
        return (("prompt", Value(32)), ("prefix", Value(16)), ("table", Array(4, Value(16)))), Value(16)

    def body(B, S, prompt, prefix, table):
        idx = B.call(I32Add, prompt, prompt)
        row = B.call(gather_bf16(4), idx, table)
        return B.call(Bf16Add, row, prefix)

    fn = bind(CompositeDefinition("PopreadToy", 0, (), signature, body, register=False))
    corr = {"inputs": ["prompt", "prefix"], "weights": ["table"]}
    path = tmp_path / "descriptor.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(encode_program(Program(fn, annotations={"correspondence": corr})), f)
    return path


def _by_family(report: dict) -> dict[str, dict]:
    return {f["family"]: f for f in report["families"]}


def test_each_kind_of_read_is_named_with_its_value_and_its_units(tmp_path):
    path = _toy(tmp_path)
    units, population = PR.q_word_units(path)
    r = PR.analyze_descriptor(PR.load(path), units=units)
    fams = _by_family(r)
    assert population == 3 and r["totals"] == {"calls": 3, "gates": 3, "units": 3}
    assert r["inputs"] == {"prompt": 1, "prefix": 1, "weight": 1}

    add = fams["I32Add_v1"]
    assert (add["verdict"], add["value"], add["selects"]) == ("PUBLIC", None, [])
    assert add["reads"] == {"inputs": ["prompt"], "calls": []}

    gather = fams["GatherBf16x4_v1"]
    assert (gather["verdict"], gather["value"]) == ("PUBLIC", None)
    assert gather["reads"] == {"inputs": ["weight"], "calls": ["I32Add_v1"]}
    assert gather["selects"] == [{"select": "GatherBf16x4", "V": 4, "count": 1, "index": ["call:I32Add_v1"]}]

    warm = fams["Bf16Add_v1"]
    assert warm["verdict"] == "VALUE-DEPENDENT" and warm["value"].startswith("prefix: prompt_token_ids")
    assert warm["reads"] == {"inputs": ["prefix"], "calls": ["GatherBf16x4_v1"]}

    s = r["summary"]
    assert (s["value_dependent"]["units"], s["value_dependent"]["share_units"]) == (1, 1 / 3)
    assert s["in_unit_select"]["units"] == 1
    assert "VALUE-DEPENDENT (prefix: prompt_token_ids" in PR.render([dict(r, name="toy")])


def test_a_selection_inside_a_composite_names_the_call_its_index_comes_from(tmp_path):
    """The index reaches the gather through the composite's parameter: the walk maps it to the root Call that computed it."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.scalar import I32Add
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.types import Array, Value
    from verity_vllm.program.registry.prims import gather_bf16

    inner = bind(CompositeDefinition("PopreadRow", 0, (), lambda S: ((("e", Value(32)), ("w", Array(2, Array(8, Value(16))))),
                                                                      Array(2, Value(16))),
                                     lambda B, S, e, w: B.batch(gather_bf16(8), e, w, axes=(None, 0)), register=False))

    def body(B, S, tok, w):
        return B.call(inner, B.call(I32Add, tok, tok), w)

    fn = bind(CompositeDefinition("PopreadNest", 0, (), lambda S: ((("tok", Value(32)), ("w", Array(2, Array(8, Value(16))))),
                                                                    Array(2, Value(16))), body, register=False))
    r = PR.analyze_descriptor(encode_program(Program(fn)))
    row = _by_family(r)["PopreadRow_v0"]
    assert row["verdict"] == "PUBLIC" and row["gates"] == 2
    assert row["selects"] == [{"select": "GatherBf16x8", "V": 8, "count": 2, "index": ["call:I32Add_v1"]}]


def test_the_chronology_marks_steps_from_one_past_the_lag_as_eos_keyed(tmp_path):
    desc = PR.load(_toy(tmp_path))
    chrono = {"engine_steps": [{"e": s, "members": [{"request_id": "r0", "s": s, "nodes": span}]}
                               for s, span in enumerate(([0, 4], [4, 5], [5, 6]))]}
    r = PR.analyze_descriptor(desc, workload_program={"lag": {"value": 1}, "chronology": chrono})
    keyed = {f["family"]: f["eos_keyed"]["calls"] for f in r["families"]}
    assert keyed == {"I32Add_v1": 0, "GatherBf16x4_v1": 0, "Bf16Add_v1": 1}
    assert r["eos"] == {"lag": 1, "keyed_from_step": 2, "counted": True, "applied": True, "why_not": None}

    ignored = PR.analyze_descriptor(desc, workload_program={"lag": {"value": 1}, "chronology": chrono,
                                                            "stop_rule": {"ignore_eos": True}})
    assert (ignored["eos"]["applied"], ignored["eos"]["why_not"]) == (False, PR.IGNORE_EOS)
    assert ignored["summary"]["eos_keyed"]["calls"] == 1
    assert "were EOS honored" in PR.render([dict(ignored, name="toy")])

    short = {"engine_steps": [{"e": 0, "members": [{"request_id": "r0", "s": 0, "nodes": [0, 5]}]}]}
    r = PR.analyze_descriptor(desc, workload_program={"chronology": short})
    assert not r["eos"]["applied"] and "end at 5, not this Program's 6" in r["eos"]["why_not"]


def _prefix_planner(w: dict) -> list[int]:
    """Cached rows per request: the longest run of 2-token blocks an earlier request's prompt shares."""
    if not w["engine"]["prefix_caching"]:
        return [0 for _ in w["requests"]]
    out, seen = [], []
    for r in w["requests"]:
        ids, best = r["prompt_token_ids"], 0
        for q in seen:
            n = 0
            while n + 2 <= min(len(ids) - 1, len(q)) and q[n:n + 2] == ids[n:n + 2]:
                n += 2
            best = max(best, n)
        out.append(best)
        seen.append(ids)
    return out


def test_a_plan_that_reads_the_prompts_is_value_dependent_and_one_that_reads_lengths_is_not():
    reqs = [{"prompt_len": 6, "max_tokens": 2, "arrive_step": 0, "prompt_token_ids": [5, 6, 7, 8, 9, 10]},
            {"prompt_len": 6, "max_tokens": 2, "arrive_step": 1, "prompt_token_ids": [5, 6, 7, 8, 1, 2]}]
    warm = PR.plan_check({"engine": {"prefix_caching": True}, "requests": reqs}, _prefix_planner)
    assert warm["plan"] == [0, 4] and warm["verdict"] == "VALUE-DEPENDENT" and warm["value"] == "prompt_token_ids"
    assert [c["variant"] for c in warm["changed_by"]] == ["distinct prompts"]
    cold = PR.plan_check({"engine": {"prefix_caching": False}, "requests": reqs}, _prefix_planner)
    assert (cold["verdict"], cold["changed_by"]) == ("PUBLIC", [])


def test_served_reads_count_each_definition_per_row_and_name_the_embedding_gather():
    """r0 prefills 3 rows at step 0 and decodes 1 at step 1, beside r1's 2-row prefill: r0's decode row is what its EOS would key,
    and all 3 rows of step 1 are in the batch that stop would reshape."""
    gemm = {"definition": "Gemm_v2{K=16,N=4,DOT=HopperBF16WgmmaDot16_v1}", "base": "Gemm_v2",
            "statics": {"K": 16, "N": 4, "DOT": "HopperBF16WgmmaDot16_v1"}}
    schedule = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 2},
                {"request_id": "r1", "arrive": 1, "prompt_tokens": 2, "max_tokens": 1}]
    steps = [(0, [("r0", 3)]), (1, [("r0", 1), ("r1", 2)])]
    build = {"program_digest": "toy", "workload": {"schedule": schedule},
             "program": {"calls": [{"definition": "Embedding_v1{V=4,H=16}", "base": "Embedding_v1", "statics": {"V": 4, "H": 16},
                                    "reads": [{"value": "c00.input/ids", "width": 1}, {"weight": "emb", "shape": [4, 16]}],
                                    "writes": ["c00.embedding/buf0"]}],
                         "values": {"c00.input/ids": {"width": 1}, "c00.embedding/buf0": {"width": 16}}, "attention": [],
                         "logits": {"name": "logits_processor", "x": {"value": "c00.embedding/buf0", "width": 16},
                                    "w": {"weight": "lm", "shape": [4, 16]}},
                         "sampled": {"name": "s", "definition": "TokenSelect_v1{V=4}", "base": "TokenSelect_v1", "statics": {"V": 4}}},
             "steps_plan": [{"step": s, "tokens": sum(t for _r, t in rs), "gemms": {}, "logits": gemm,
                             "requests": [{"request_id": r, "tokens": t, "attention": []} for r, t in rs]} for s, rs in steps]}
    r = PR.analyze_served(build)
    fams = _by_family(r)
    emb = fams["Embedding_v1"]
    assert (emb["calls"], emb["eos_keyed"]["calls"], emb["eos_step_keyed"]["calls"], emb["verdict"]) == (6, 1, 3, "PUBLIC")
    assert emb["selects"] == [{"select": "GatherBf16x4", "V": 4, "count": 96, "index": ["input:ids"]}]
    assert (fams["Gemm_v2"]["calls"], fams["TokenSelect_v1"]["calls"]) == (3, 3)
    assert r["eos"]["applied"] and "Activity keyed on EOS" in PR.render([dict(r, name="toy")])

    ignored = PR.analyze_served(build, ignore_eos=True)
    assert not ignored["eos"]["applied"] and ignored["summary"]["eos_step_keyed"]["calls"] == r["summary"]["eos_step_keyed"]["calls"]
