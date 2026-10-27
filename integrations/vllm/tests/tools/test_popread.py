"""`tools/popread.py` on a toy Program with one Call of each kind: one whose reads are public, one that reads every candidate and
selects one inside the proof block by another Call's output, and one that reads an input whose binding is planned from committed values.
Also its step attribution from a workload Program's chronology, its planner counterfactual, its P_served reader, each proof block's
egress E(u), and the circuit-native prefix form's read sets."""
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
    from verity.ml.definitions.registry.prims import gather_bf16

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

    e = r["egress"]
    assert (e["slots"], e["bits"], e["slot_bits"], e["E_max"]) == (1, 16, [[16, 16]], {"bits": 16, "slots": 1, "declared_bits": 16})
    assert (e["weight"], e["at_max"]["share"], e["components"]["with_egress"]) == ("units", 1.0, 1)

    s = r["summary"]
    assert (s["value_dependent"]["units"], s["value_dependent"]["share_units"]) == (1, 1 / 3)
    assert s["in_unit_select"]["units"] == 1
    assert "VALUE-DEPENDENT (prefix: prompt_token_ids" in PR.render([dict(r, name="toy")])
    consts = [dict(add, family=f"Const32[0x0000000{i}]_v1") for i in (1, 2)]
    assert "| Const[...] (2 Definitions) | 2 | 2 | 66.67% | 2 |" in PR.render([dict(r, name="toy", families=r["families"] + consts)])


def test_a_selection_inside_a_composite_names_the_call_its_index_comes_from(tmp_path):
    """The index reaches the gather through the composite's parameter: the walk maps it to the root Call that computed it."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.scalar import I32Add
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.types import Array, Value
    from verity.ml.definitions.registry.prims import gather_bf16

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
    """The chronology's ranges count the Calls after the root's Input batches, so the toy's 3 Calls are nodes 0..2."""
    desc = PR.load(_toy(tmp_path))
    chrono = {"engine_steps": [{"e": s, "members": [{"request_id": "r0", "s": s, "nodes": span}]}
                               for s, span in enumerate(([0, 1], [1, 2], [2, 3]))]}
    r = PR.analyze_descriptor(desc, workload_program={"lag": {"value": 1}, "chronology": chrono})
    keyed = {f["family"]: f["eos_keyed"]["calls"] for f in r["families"]}
    assert keyed == {"I32Add_v1": 0, "GatherBf16x4_v1": 0, "Bf16Add_v1": 1}
    assert r["eos"] == {"lag": 1, "keyed_from_step": 2, "counted": True, "applied": True, "why_not": None}
    one = {"calls": 1, "gates": 1, "units": 0}
    assert [(b["s"], b["calls"], b["families"]) for b in r["by_step"]] == [
        (0, 1, {"I32Add_v1": one}), (1, 1, {"GatherBf16x4_v1": one}), (2, 1, {"Bf16Add_v1": one})]

    ignored = PR.analyze_descriptor(desc, workload_program={"lag": {"value": 1}, "chronology": chrono,
                                                            "stop_rule": {"ignore_eos": True}})
    assert (ignored["eos"]["applied"], ignored["eos"]["why_not"]) == (False, PR.IGNORE_EOS)
    assert ignored["summary"]["eos_keyed"]["calls"] == 1
    assert "were EOS honored" in PR.render([dict(ignored, name="toy")])

    short = {"engine_steps": [{"e": 0, "members": [{"request_id": "r0", "s": 0, "nodes": [0, 2]}]}]}
    r = PR.analyze_descriptor(desc, workload_program={"chronology": short})
    assert not r["eos"]["applied"] and "end at 2, not this Program's 3 Calls" in r["eos"]["why_not"]


def _prefix_planner(w: dict) -> list[int]:
    """Cached rows per request: the longest run of 2-token KV pages an earlier request's prompt shares."""
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


SMALL = [("MoeExpertFp8Gemm_v1", {"E": 4, "K": 128, "N": 16, "G": 128, "KB": 128, "SWAP": False, "DOT": "HopperE4m3MmaV2Dot32_v1"}),
         ("FlashInferFp8Linear_v1", {"K": 128, "N": 16}),
         ("MoeRouterSigmoidBias_v1", {"E": 8, "TOPK": 2, "SCALE": 2.5, "KERNEL": "block"}),
         ("RMSNormFusedCuda_v2", {"N": 16, "EPS": 1e-5}),
         ("MoeExpertGemm_v2", {"E": 4, "K": 16, "N": 4, "DOT": "HopperBF16WgmmaDot16_v1"})]


def test_the_lean_graph_is_the_reference_graph_and_cuts_to_its_units(monkeypatch):
    """Expert gathers, FP8 block scales, a router's top-k and a separable body: the lean graph, resolved in forked workers over
    64-gate ranges or in one pass, is `CallGraph`'s, and its proof blocks are `evaluate_definition`'s."""
    from verity.ml.circuits.cut import CallGraph, evaluate_definition, separable
    from verity.ml.definitions.call_scope import specialization
    monkeypatch.setattr(PR, "CHUNK_GATES", 64)
    for name, statics in SMALL:
        fn = specialization(name, statics)
        ref = evaluate_definition(fn, PR.UNIT_BITS, {}).units
        assert PR.lean_units(fn, {}) == PR.lean_units(fn, {}, jobs=2) == ref, fn.id
        if separable(fn) is None:
            assert PR.LeanGraph(fn, jobs=2).same_as(CallGraph(fn)) and PR.LeanGraph(fn).same_as(CallGraph(fn)), fn.id


def test_units_read_a_cached_rule_by_its_digest_and_log_each_cut(tmp_path):
    """A rule directory's entry is used as it stands (its count is deliberately wrong here), the rest are cut and logged, the
    log serves as a cache, and recutting a cached Definition names the disagreement."""
    from verity.ml.definitions.call_scope import specialization
    from verity.ml.definitions.word import definition_digest
    lin, router = (specialization(n, s) for n, s in SMALL[1:3])
    rule = {"key": {"digest": definition_digest(lin), "query": [16, 32, 0]}, "rule": {"units_per_call": 999, "violations": []}}
    (tmp_path / "rules" / "ab").mkdir(parents=True)
    (tmp_path / "rules" / "ab" / "x.json").write_text(json.dumps(rule))
    log = tmp_path / "cuts.jsonl"
    u = PR.Units([tmp_path / "rules"], log, verify_below=10_000)
    assert u.of({"lin": lin, "router": router}) == {"lin": 999, "router": 41}
    s = u.summary()
    assert (s["cached"], s["cut"], s["verified_units"], s["verified_graphs"], s["mismatches"]) == (1, 1, 1, 1, [])
    assert [json.loads(x)["digest"] for x in log.read_text().splitlines()] == [definition_digest(router)]

    again = PR.Units([log])
    assert again.of({"router": router}) == {"router": 41} and again.summary()["cut"] == 0
    recut = PR.Units([tmp_path / "rules"], recut=True)
    assert recut.of({"lin": lin}) == {"lin": 146}
    assert recut.summary()["mismatches"] == [{"definition": lin.id, "lean": 146, "cached": 999}]


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

    # egress: r0's step-0 sampled row reaches both of its ids (2 bits each for V = 4); its 2 unsampled rows reach no k/v (the toy
    # has no attention) and no logits, so E = 0
    e = r["egress"]
    assert (e["slots"], e["bits"], e["declared_bits"], e["weight"]) == (3, 6, 96, "gates")
    assert e["E_max"] == {"bits": 4, "slots": 2, "declared_bits": 64}
    assert sorted(f["family"] for f in e["at_max"]["families"]) == ["Embedding_v1", "Gemm_v2", "TokenSelect_v1"]
    assert [(c["component"], c["slots"], c["E_max"]["bits"]) for c in e["components"]["per"]] == [("r0", 2, 4), ("r1", 1, 2)]
    assert (e["zero"]["calls"], e["components"]["without_egress"]) == (3, 0)
    assert "| toy | 3 (2) | 2 / 4 |" in PR.render_egress([dict(r, name="toy")])

    ignored = PR.analyze_served(build, ignore_eos=True)
    assert not ignored["eos"]["applied"] and ignored["summary"]["eos_step_keyed"]["calls"] == r["summary"]["eos_step_keyed"]["calls"]


def test_served_liveness_splits_a_layer_at_its_attention():
    """An unsampled row's qkv projection writes k/v, which later rows read; its attention output and what follows feed only its own
    logits, which only a sampled row has."""
    P = {"calls": [{"reads": [{"value": "c00.input/ids"}, {"weight": "emb"}], "writes": ["e"]},
                   {"reads": [{"value": "e"}, {"weight": "wqkv"}], "writes": ["qkv"]},
                   {"reads": [{"value": "att0"}, {"weight": "wo"}], "writes": ["h"]}],
         "attention": [{"name": "att0", "q": {"value": "qkv"}, "k": {"value": "qkv"}, "v": {"value": "qkv"}}],
         "logits": {"x": {"value": "h"}}}
    kv, x = PR.served_live(P)
    assert kv == {("c", 0), ("c", 1)} and x == {("c", 0), ("c", 1), ("c", 2), ("a", 0)}


def test_served_moe_calls_take_the_step_plans_expert_gemm_once_per_routed_pair():
    """A fused-MoE Call names its plan key: each of a token's TOPK = 2 pairs is one activation of the planned expert Gemm, which
    gathers its expert's weights by the routed id."""
    moe = {"definition": "MoeExpertGemm_v2{E=4,K=16,N=4,DOT=HopperBF16WgmmaDot16_v1}", "base": "MoeExpertGemm_v2",
           "statics": {"E": 4, "K": 16, "N": 4, "DOT": "HopperBF16WgmmaDot16_v1"}}
    gemm = {"definition": "Gemm_v2{K=16,N=4,DOT=HopperBF16WgmmaDot16_v1}", "base": "Gemm_v2",
            "statics": {"K": 16, "N": 4, "DOT": "HopperBF16WgmmaDot16_v1"}}
    calls = [{"op": "moe_up", "moe": "4,16,4,0", "writes": ["c01.moe/up"],
              "reads": [{"value": "c00.input/x", "width": 16}, {"value": "c00.input/router_ids", "width": 2},
                        {"weight": "w13", "shape": [4, 4, 16]}]}]
    build = {"workload": {"schedule": [{"request_id": "r0", "arrive": 0, "prompt_tokens": 3, "max_tokens": 1}]},
             "program": {"calls": calls, "attention": [],
                         "values": {"c00.input/x": {"width": 16}, "c00.input/router_ids": {"width": 2}, "c01.moe/up": {"width": 8}},
                         "logits": {"x": {"value": "c00.input/x", "width": 16}, "w": {"weight": "lm", "shape": [4, 16]}},
                         "sampled": {"definition": "TokenSelect_v1{V=4}", "base": "TokenSelect_v1", "statics": {"V": 4}}},
             "steps_plan": [{"step": 0, "tokens": 3, "gemms": {}, "moe": {"4,16,4,0": moe}, "logits": gemm,
                             "requests": [{"request_id": "r0", "tokens": 3, "attention": []}]}]}
    r = PR.analyze_served(build, units=True)
    expert = _by_family(r)["MoeExpertGemm_v2"]
    assert (expert["calls"], expert["units"], expert["verdict"]) == (6, 24, "PUBLIC")
    assert expert["selects"] == [{"select": "GatherBf16x4", "V": 4, "count": 6 * 64, "index": ["input:router_ids"]}]
    assert r["units_source"]["cut"] == 3 and r["units_source"]["mismatches"] == []


def _prefix_form(producers: int, warm: int) -> dict:
    """The circuit-native prefix form at C = 2 cached rows: one static pool of prompt KV pages.  Each producer computes
    its page and its own row and samples a token.  Each warm request computes only its own row; its prefix-match proof block compares its
    first C prompt ids with each candidate producer's and outputs a page mask, and its attention reads every candidate page and
    selects by the mask inside the proof block.  One more request shares only the weights."""
    from verity.ml.circuits.codec import encode_program
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.fp32 import Bf16Add
    from verity.ml.circuits.ops.scalar import BitAnd, I32Eq, SelectBf16
    from verity.ml.circuits.program import Program
    from verity.ml.circuits.refs import array_of
    from verity.ml.circuits.types import Array, Value
    from verity.ml.definitions.registry.prims import gather_bf16
    from verity.ml.definitions.call_scope import specialization

    V, C, K = 4, 2, producers
    I32, BF16, BIT = Value(32), Value(16), Value(1)
    table_t, page_t = Array(V, BF16), Array(C, BF16)
    kv, select = gather_bf16(V), specialization("TokenSelect_v1", {"V": V})

    def comp(name, params, ret, body):
        return bind(CompositeDefinition(name, 0, (), lambda S: (params, ret), body, register=False))

    def fold(B, prim, xs):
        acc = xs[0]
        for x in xs[1:]:
            acc = B.call(prim, acc, x)
        return acc

    def masked(B, S, mask, pages, own):
        words = []
        for i in range(C):
            w = pages[K - 1][i]
            for j in reversed(range(K - 1)):
                w = B.call(SelectBf16, mask[j], pages[j][i], w)
            words.append(w)
        return fold(B, Bf16Add, words + [own])

    page = comp("ToyKvPage", (("ids", Array(C, I32)), ("w", table_t)), page_t, lambda B, S, ids, w: B.batch(kv, ids, w, axes=(0, None)))
    attend = comp("ToyAttention", (("page", page_t), ("own", BF16)), BF16, lambda B, S, p, own: fold(B, Bf16Add, [*p, own]))
    match = comp("ToyPrefixMatch", (("w", Array(C, I32)), ("x", Array(C, I32))), BIT,
                 lambda B, S, w, x: fold(B, BitAnd, list(B.batch(I32Eq, w, x, axes=(0, 0)))))
    attend_pool = comp("ToyPoolAttention", (("mask", Array(K, BIT)), ("pages", Array(K, page_t)), ("own", BF16)), BF16, masked)
    logits = comp("ToyLogits", (("h", BF16), ("w", table_t)), table_t, lambda B, S, h, w: B.batch(Bf16Add, w, h, axes=(0, None)))
    names = [f"producer{j}" for j in range(K)] + [f"warm{j}" for j in range(warm)] + ["other"]

    def body(B, S, *args):
        *reqs, w = args
        pages, toks = [], []

        def own(ids):
            return B.call(kv, ids[C], w)

        def head(h):
            return B.call(select, B.call(logits, h, w))

        for ids in reqs[:K]:
            pages.append(B.call(page, ids[:C], w))
            toks.append(head(B.call(attend, pages[-1], own(ids))))
        for ids in reqs[K:K + warm]:
            mask = array_of([B.call(match, ids[:C], p[:C]) for p in reqs[:K]])
            toks.append(head(B.call(attend_pool, mask, array_of(pages), own(ids))))
        toks.append(head(B.call(attend, B.call(page, reqs[-1][:C], w), own(reqs[-1]))))
        return array_of(toks)

    fn = comp("ToyPrefixForm", tuple((n, Array(C + 1, I32)) for n in names) + (("table", table_t),), Array(len(names), I32), body)
    return encode_program(Program(fn, annotations={"correspondence": {"inputs": names, "weights": ["table"]}}))


def test_the_circuit_native_prefix_form_reads_only_public_positions_and_multiplies_a_pages_egress_by_its_sharers():
    """Every read set is PUBLIC: the warm requests read every candidate page at fixed positions, and the page mask the prefix-match
    proof block computes from the prompts selects inside the attention proof block.  A producer's page is in its own request's cone and in each
    warm request's that reads it, so its E is 1 + the sharers' ids; the request that shares only weights is its own component."""
    r = PR.analyze_descriptor(_prefix_form(producers=2, warm=2))
    fams = _by_family(r)
    assert {f["verdict"] for f in r["families"]} == {"PUBLIC"} and r["summary"]["value_dependent"]["calls"] == 0
    pool = fams["ToyPoolAttention_v0"]
    assert pool["reads"] == {"inputs": [], "calls": ["GatherBf16x4_v1", "ToyKvPage_v0", "ToyPrefixMatch_v0"]}
    assert pool["selects"] == [{"select": "SelectBf16", "V": 2, "count": 2 * 2, "index": ["call:ToyPrefixMatch_v0"]}]
    assert fams["ToyPrefixMatch_v0"]["reads"] == {"inputs": ["producer0", "producer1", "warm0", "warm1"], "calls": []}

    e = r["egress"]
    assert (e["slots"], e["bits"], e["slot_bits"]) == (5, 10, [[2, 32]])
    assert e["E_max"] == {"bits": 6, "slots": 3, "declared_bits": 96}
    assert [f["family"] for f in e["at_max"]["families"]] == ["ToyKvPage_v0"] and e["at_max"]["calls"] == 2
    assert [(c["slots"], c["E_max"]["slots"]) for c in e["components"]["per"]] == [(4, 3), (1, 1)]
    assert PR.analyze_descriptor(_prefix_form(producers=2, warm=3))["egress"]["E_max"]["slots"] == 4


def test_a_k_way_page_mux_is_one_unit_a_word_and_a_prefix_match_one_unit():
    """The prefix cost model's prices, by core's cut: a K-way select chain over a 16-bit KV word is one Q_word proof block for every K (and
    K - 1 gates), and the match of C = 128 prompt ids in blocks of 16, whose 8 mask bits fit one proof block's output, is one proof block."""
    from verity.ml.circuits.cut import evaluate_definition
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.ops.scalar import BitAnd, I32Eq, SelectBf16
    from verity.ml.circuits.refs import array_of
    from verity.ml.circuits.types import Array, Value

    def mux(K, words=8):
        def body(B, S, m, pages):
            out = []
            for i in range(words):
                w = pages[K - 1][i]
                for j in reversed(range(K - 1)):
                    w = B.call(SelectBf16, m[j], pages[j][i], w)
                out.append(w)
            return array_of(out)
        sig = ((("m", Array(K - 1, Value(1))), ("pages", Array(K, Array(words, Value(16))))), Array(words, Value(16)))
        return bind(CompositeDefinition(f"ToyMux{K}", 0, (), lambda S: sig, body, register=False))

    for K in (2, 4, 8):
        fn = mux(K)
        assert (fn.gates, evaluate_definition(fn, PR.UNIT_BITS, {}).units, PR.lean_units(fn, {})) == (8 * (K - 1), 8, 8)

    C, BS = 128, 16

    def match(B, S, w, x):
        eq, out, m = list(B.batch(I32Eq, w, x, axes=(0, 0))), [], None
        for b in range(C // BS):
            e = eq[b * BS]
            for i in range(1, BS):
                e = B.call(BitAnd, e, eq[b * BS + i])
            m = e if m is None else B.call(BitAnd, m, e)
            out.append(m)
        return array_of(out)

    fn = bind(CompositeDefinition("ToyBlockMatch", 0, (), lambda S: ((("w", Array(C, Value(32))), ("x", Array(C, Value(32)))),
                                                                       Array(C // BS, Value(1))), match, register=False))
    assert (fn.gates, evaluate_definition(fn, PR.UNIT_BITS, {}).units) == (255, 1)
