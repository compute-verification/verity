"""The replay's address rule: `ProgramIndex` names a row's members from the v2 query's population (`required.Population.members_of`),
the rows `request_manifest` counts.  An interior row whose output crosses a module boundary is addressed and the manifest's
identity has its rows; every identity with Program rows is produced by the addressed rows, on a request Program and on a TP rank Program."""
from __future__ import annotations

import pytest

from verity_vllm.check.replay import coverage as COV, index as IX, population as POP
from tests.check.test_sampled_replay import _program
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import request_manifest

IMPL = ("model.layers.0.mlp.down_proj",)          # an observed implementation path: the Program's wrapper prefix is `model.`
EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def _v2_manifest(inst: dict, res: dict | None = None, **kw) -> dict:
    P = from_instances(inst)
    res = res or {"program": {"digest": "p" * 64, "function": "Derived[vLLMRequest[Test]]"}, "target": {"declared": {"compute_capability": [8, 9]}}}
    return request_manifest(P, Correspondence(P, implementation_paths=IMPL), res, request_id="r0", **kw).manifest


def _reconciles(man: dict, pi: IX.ProgramIndex, rank: int = 0) -> None:
    addresses: dict = {}
    POP.population(pi, "r0", rank=rank, addresses=addresses)
    checks = COV.query_population(man, addresses, set(), ranks=[rank])["query_checks"]
    bad = {k: v for k, v in checks.items() if isinstance(v, dict) and v.get("n")}
    assert not bad, bad


def test_the_crossing_interior_row_is_addressed_as_the_module_output_it_crosses_as():
    """The gemma2-2b shape: the root body's `embed * normalizer` has no module-output name of its own but crosses into layer 0 --
    the v2 query requires it as (model, out) and the rule addresses exactly those rows, with the manifest's row count."""
    inst, _woff = _program(scale=True)
    man = _v2_manifest(inst)
    pi = IX.ProgramIndex(inst)
    crossing = {r["i"]: r["spec"] for r in inst["rows"] if (d := pi.describe(r)) is not None and d["op_path"] == "model" and d["members"]}
    assert crossing and all(spec.startswith("Bf16MulScalarTensor_v1{") for spec in crossing.values()), crossing
    assert all([m[0] for m in pi.describe(inst["rows"][i])["members"]] == ["out"] for i in crossing)
    rows = [x for x in man["identities"] if x["op_path"] == "model" and x["output_member"] == "out"]
    assert sum(int(x["rows"]) for x in rows) == len(crossing)


@pytest.mark.parametrize("variant", [{}, {"bias": True}, {"attn": True}, {"scale": True}])
def test_the_addressed_rows_reconcile_with_the_manifest_identity_by_identity(variant):
    """Every identity of the manifest with Program rows is produced by the addressed rows, with the manifest's row count, and no
    addressed row is outside the manifest (`query_population`'s fail-closed checks)."""
    inst, _woff = _program(**variant)
    man = _v2_manifest(inst)
    assert man["complete"], man["unmodelled"]
    _reconciles(man, IX.ProgramIndex(inst))


def _tp_rank_program(rank: int, world: int = 2) -> dict:
    """A wired TP rank Program in the derive_step --tp vocabulary (the llama TP2 B1 record's shape at toy size): per step an all_reduce on
    the embedding shard, an all_reduce on the row-parallel down_proj Gemm (reading its own weight shard) and an all_gather on the logits
    shard; every collective row reads the rank's OWN partial and the RECEIVED peer partial = the Program input `peers_<k>`."""
    LP, H, V = 2, 32, 16
    rows: list[dict] = []
    params = [["prompt", f"Array<{LP},Value<32>>"]]
    weights = {"model.model.embed_tokens.weight": f"Array<{V},Array<{H},Value<16>>>",
               "model.model.layers.0.mlp.down_proj.weight": f"Array<{H},Array<{H // world},Value<16>>>",
               "model.lm_head.weight": f"Array<{V // world},Array<{H},Value<16>>>"}
    for k, (n_rows, n) in enumerate([(2, H), (2, H), (1, V // world), (1, H), (1, H), (1, V // world)]):
        params.append([f"peers_{k}", f"Array<{n_rows},Array<{n},Value<16>>>"])
    wp = len(params)

    def emit(spec, name, inputs=(), args=()):
        rows.append({"i": len(rows), "spec": spec, "form": "call", "name": name, "args": [[["n", int(j), 0, 1]] for j in inputs] + [[a] for a in args]})
        return len(rows) - 1

    k = 0
    for s, rs in ((0, [0, 1]), (1, [0])):
        sh = [emit(f"EmbeddingShard_v1{{VS={V // world},H={H},START={rank * (V // world)}}}", f"model.model.embed_tokens/embedding_shard_{k}[{r}]/shard",
                   args=[["p", 0, s + r, s + r + 1]]) for r in rs]
        ar = [emit(f"AllReduce2_v1{{N={H}}}", f"model.model.embed_tokens/all_reduce2_{k}[{r}]/all_reduce", [sh[r]], [["p", k + 1, r, r + 1]]) for r in rs]
        k += 1
        g = [emit(f"Gemm_v1{{K={H // world},N={H}}}", f"model.model.layers.0.mlp.down_proj/triton_launch[{r}]", [ar[r]],
                  [["p", wp, V * H, V * H + H * (H // world)]]) for r in rs]
        ar2 = [emit(f"AllReduce2_v1{{N={H}}}", f"model.model.layers.0.mlp.down_proj/all_reduce2_{k}[{r}]/all_reduce", [g[r]], [["p", k + 1, r, r + 1]]) for r in rs]
        k += 1
        fin = [emit(f"RMSNormTriton_v1{{N={H},{EPS}}}", f"model.model.norm/triton_launch[{r}]", [ar2[r]]) for r in rs]
        lg = emit(f"Gemm_v1{{K={H},N={V // world}}}", "model.logits_processor/triton_launch_9[0]", [fin[-1]])
        ag = emit(f"AllGather2_v1{{N={V // world}}}", f"model.logits_processor/all_gather2_{k}[0]/all_gather", [lg], [["p", k + 1, 0, 1]])
        k += 1
        emit(f"TokenSelect_v1{{V={V}}}", f"argmax[{s}]", [ag])
    params.append(["weights", "{" + ", ".join(f"{n}: {t}" for n, t in weights.items()) + "}"])
    return {"schema": "cb-a/instance-sequence/v1", "digest": "d" * 64, "params": params, "rows": rows}


@pytest.mark.parametrize("rank", [0, 1])
def test_a_tp_rank_programs_addressed_rows_reconcile_with_its_rank_manifest(rank):
    """TP-04 on the v2 query: a wired rank Program's manifest carries the rank-local partials (`part_rank<r>` / `shard_rank<r>`) beside
    every collective output, and the rows addressed with the rank's TP context produce every one of them."""
    inst = _tp_rank_program(rank)
    res = {"program": {"digest": f"{rank}" * 64, "function": "Derived[vLLMRequest[Test]]"}, "wrapper": "request", "LP": 2, "T": 1,
           "target": {"declared": {"compute_capability": [8, 9]}}, "attention_impls_observed": {}, "tp": {"world": 2, "rank": rank}}
    man = _v2_manifest(inst, res, workload={"sweep": {"row_id": "tp2-df", "batch": 1, "arrive_steps": [0]}}, rank=rank, ranks=2)
    assert man["complete"], man["unmodelled"]
    assert {x["output_member"] for x in man["identities"] if x["family"] == "tp_rank_partials"} == {f"part_rank{rank}", f"shard_rank{rank}"}
    _reconciles(man, IX.ProgramIndex(inst, tp={"rank": rank, "world": 2}, attention_paths=list(IMPL)), rank=rank)
