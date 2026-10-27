"""The quarantine `AllReduceSumBf16{N,R}` definition and the collective linkage tool
(`tests.engine.collectives.collective_link`) on the TP2 evidence shards (`tests/fixtures/tp2/`, see its README)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StorageAlloc, StorageFree, TritonLaunch, ViewDesc
from verity_vllm.properties.quarantine_lint import lint
from tests.engine.collectives import collective_link as CL
from verity_vllm.engine.collectives.collective_record import COLLECTIVE_KINDS, MISSING_IN_RECORD, SCHEMA_LINK, SCHEMA_SHARD, validate_launch
from verity.ml.circuits.codec import decode_program, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluate import evaluate_region
from verity.ml.circuits.layout import resolve
from verity.ml.circuits.program import Program
from verity.ml.definitions.registry import b1_tp2
from verity.ml.definitions.registry import prims as P
from verity_vllm.program.quarantine import collective as QC
from verity_vllm.program.quarantine.collective import allreduce as A

REPO = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent / "fixtures" / "tp2"


# --- helpers ------------------------------------------------------------------------------------------------------------

def _eval(prog: Program, inputs: list[list[int]]) -> list[int]:
    Cc = prog.circuit
    t: dict[int, int] = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        assert coll.type.leaves == len(vals)
        for j, v in enumerate(vals):
            t[resolve(Cc.root_scope, coll.refs[j])] = v
    evaluate_region(Cc, Cc[0:prog.gates], t)
    return [t[g] for g in Cc.root_outputs()]


def _bf16_words(a) -> list[int]:
    return [int(x) >> 16 for x in np.asarray(a, dtype=np.float32).view(np.uint32).ravel()]


def _bf16_value(w: int) -> Fraction | None:
    """Exact rational value of a finite bf16 word (None for inf/NaN)."""
    s, e, m = (w >> 15) & 1, (w >> 7) & 0xFF, w & 0x7F
    if e == 0xFF:
        return None
    v = Fraction(m, 1 << 7) * Fraction(2) ** (e - 127) if e else Fraction(m, 1 << 7) * Fraction(2) ** (-126)
    if e:
        v += Fraction(2) ** (e - 127)
    return -v if s else v


def _exact_bf16_sum(a: int, b: int) -> int:
    """RN-even of the EXACT sum of two finite bf16 values to bf16 (the reference `add.rn.bf16x2` semantics), by rational arithmetic."""
    x = _bf16_value(a) + _bf16_value(b)
    if x == 0:
        return 0x8000 if (a >> 15) & (b >> 15) & 1 else 0
    neg = x < 0
    ax = -x if neg else x
    e = ax.numerator.bit_length() - ax.denominator.bit_length()
    if Fraction(2) ** e > ax:
        e -= 1
    e = max(e, -126)
    q = ax / (Fraction(2) ** (e - 7))            # significand scaled to 8 integer bits
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n & 1):
        n += 1
    if n >= 1 << 8:
        n >>= 1
        e += 1
    if e > 127:
        return (0x8000 if neg else 0) | 0x7F80
    if n < 1 << 7:
        return (0x8000 if neg else 0) | n
    return (0x8000 if neg else 0) | ((e + 127) << 7) | (n & 0x7F)


def _v(a, shape=(32, 1536), dtype="bfloat16", off=0):
    strides = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))
    return ViewDesc(a, off, tuple(shape), strides, dtype, "cuda:0")


# --- the definition -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("N,R", [(8, 2), (1, 2)])
def test_allreduce_definition_shape_gates_and_identity(N, R):
    sp = A.all_reduce_sum_bf16(N, R)
    assert sp.id == f"AllReduceSumBf16_v1{{N={N},R={R}}}"
    params, ret = sp.signature()
    assert [n for n, _ in params] == [f"x{r}" for r in range(R)]
    assert all(t.leaves == N for _, t in params) and ret.leaves == N
    # R-1 hops per element, 4 gates per hop (widen, widen, add, narrow)
    assert sp.gates == 4 * N * (R - 1)
    assert A.SEMANTIC_ID == "tp2/AllReduceSumBf16@v1"
    assert sp.defn.conformance.startswith("quarantine") and A.AllReduceSumBf16Elem.conformance.startswith("quarantine")
    assert "R=2 only" in A.AllReduceSumBf16Elem.conformance and "difftest pending" in A.AllReduceSumBf16Elem.conformance
    assert set(QC.ALL_DEFINITIONS) == {A.AllReduceSumBf16Elem, A.AllReduceSumBf16}


def test_allreduce_rejects_fewer_than_two_ranks():
    with pytest.raises(ValueError):
        A.all_reduce_sum_bf16(4, 1).body


@pytest.mark.parametrize("R", [3, 4])
def test_allreduce_refuses_more_than_two_ranks(R):
    """More than two ranks is `b1_tp2.all_reduce` (the hop order of `b1_tp2.allreduce_order`), never a chain of this family."""
    with pytest.raises(ValueError, match="allreduce_order"):
        A.all_reduce_sum_bf16(4, R).body
    with pytest.raises(ValueError, match="allreduce_order"):
        A.all_reduce_sum_bf16_words([[0]] * R)


def test_r2_equals_allreduce2_bf16add_and_the_exact_single_rounding():
    """For two ranks the body is one commutative bf16 add: bit-equal to `b1_tp2.AllReduce2`, to `P.Bf16Add` on non-NaN words, and to
    RN-even of the EXACT sum (no double rounding through the f32 intermediate); symmetric in the two operands."""
    rng = np.random.default_rng(20260917)
    N = 512
    x0 = _bf16_words(rng.standard_normal(N) * rng.choice([1e-3, 1.0, 64.0, 2.0 ** 20], N))
    x1 = _bf16_words(rng.standard_normal(N) * rng.choice([1e-3, 1.0, 64.0, 2.0 ** 20], N))
    out = _eval(Program(A.all_reduce_sum_bf16(N, 2)), [x0, x1])
    assert out == A.all_reduce_sum_bf16_words([x0, x1])
    assert out == _eval(Program(bind(b1_tp2.AllReduce2, N=N)), [x0 + x1])
    assert out == [P.Bf16Add.evaluate(a, b) for a, b in zip(x0, x1)]
    assert out == [_exact_bf16_sum(a, b) for a, b in zip(x0, x1)]
    assert out == _eval(Program(A.all_reduce_sum_bf16(N, 2)), [x1, x0])


@pytest.mark.parametrize("a,b", [
    (1.0, 2.0 ** -8),                 # exact tie below 1 ulp(bf16 at 1.0 = 2^-7): RNE -> 1.0
    (1.0, 2.0 ** -8 + 2.0 ** -15),    # just above the tie -> 1 + 2^-7
    (1.0 + 2.0 ** -7, 2.0 ** -8),     # tie with an odd mantissa -> rounds up to 1 + 2^-6
    (1.0, 2.0 ** -20),                # exponent gap 20: exact in f32 (needs 21 bits), rounds to 1.0 in bf16
    (1.0, 2.0 ** -30),                # exponent gap 30: the f32 add itself rounds; far from every bf16 midpoint
    (3.0e38, 3.0e38),                 # overflow -> +inf
    (-2.0 ** -130, 2.0 ** -133),      # subnormal bf16 range
])
def test_r2_single_rounding_edge_cases(a, b):
    wa, wb = _bf16_words([a])[0], _bf16_words([b])[0]
    got = A.all_reduce_sum_bf16_words([[wa], [wb]])[0]
    assert got == _exact_bf16_sum(wa, wb) == P.Bf16Add.evaluate(wa, wb)


def test_r3_per_hop_rounding_and_order_dependence_exist():
    """Why R > 2 needs a stated order: NCCL reduces in bf16 hop by hop, so (i) it differs from one f32 accumulation rounded once and
    (ii) it depends on the rank order.  Both effects are exhibited on small bf16 triples.  This family refuses R = 3; the three-rank
    all-reduce is `b1_tp2.all_reduce(3, N)`, whose chain follows `allreduce_order(3)` and not the ascending order."""
    words = _bf16_words([1.0, 1.0 + 2.0 ** -7, 2.0 ** -8, 3 * 2.0 ** -9, 2.0 ** -9, -2.0 ** -8, 0.75, 1.5])

    def hop(xs, order=(0, 1, 2)):
        acc = xs[order[0]]
        for r in order[1:]:
            acc = P.F2fpBf16.evaluate(P.F32Add.evaluate(P.Bf16ToF32.evaluate(acc), P.Bf16ToF32.evaluate(xs[r])))
        return acc

    measured = b1_tp2.allreduce_order(3)
    differs_from_f32_once = order_dependent = None
    for a in words:
        for b in words:
            for c in words:
                chain = hop([a, b, c])
                once = P.F2fpBf16.evaluate(P.F32Add.evaluate(P.F32Add.evaluate(P.Bf16ToF32.evaluate(a), P.Bf16ToF32.evaluate(b)), P.Bf16ToF32.evaluate(c)))
                if chain != once and differs_from_f32_once is None:
                    differs_from_f32_once = (a, b, c, chain, once)
                if chain != hop([a, b, c], measured) and order_dependent is None:
                    order_dependent = (a, b, c)
    assert differs_from_f32_once is not None, "per-hop bf16 rounding never differed from a single f32 accumulation"
    assert order_dependent is not None, "the hop order never mattered"
    a, b, c = order_dependent
    with pytest.raises(ValueError):
        Program(A.all_reduce_sum_bf16(1, 3))
    assert _eval(Program(b1_tp2.all_reduce(3, 1)), [[a, b, c]]) == [hop([a, b, c], measured)] != [hop([a, b, c])]


def test_r2_equals_the_wave1_body_checker_formula():
    """The 116/116 `body_ok` evidence (`tp2_body.summary.json`) was produced by `tp.analyze`'s numpy body
    `f32_to_bf16_rne(f32(x_0) + f32(x_1))`; the Definition computes the same words, so that evidence is evidence FOR this definition."""
    from tests.engine.collectives import analyze as ta
    rng = np.random.default_rng(7)
    N = 2048
    a = ta.f32_to_bf16_rne((rng.standard_normal(N) * rng.choice([0.01, 1.0, 300.0], N)).astype(np.float32))
    b = ta.f32_to_bf16_rne((rng.standard_normal(N) * rng.choice([0.01, 1.0, 300.0], N)).astype(np.float32))
    wave1 = ta.f32_to_bf16_rne(ta.bf16_to_f32(a) + ta.bf16_to_f32(b))
    ours = A.all_reduce_sum_bf16_words([[int(x) for x in a], [int(x) for x in b]])
    assert ours == [int(x) for x in wave1]


def test_descriptor_round_trip_and_registry():
    prog = Program(A.all_reduce_sum_bf16(6, 2))
    desc = encode_program(prog)
    ids = set(desc["definitions"])
    assert any(i.startswith("Root[AllReduceSumBf16_v1{N=6,R=2}]") for i in ids), ids
    assert "AllReduceSumBf16Elem_v1{R=2}" in ids and {"Bf16ToF32_v1", "F32Add_v1", "F2fpBf16_v1"} <= ids
    D = decode_program(json.loads(json.dumps(desc)), REGISTRY)
    assert program_digest(D) == program_digest(prog)
    assert "AllReduceSumBf16_v1" in REGISTRY and "AllReduceSumBf16Elem_v1" in REGISTRY


def test_quarantine_lint_passes_for_the_collective_package():
    rep = lint(Path(QC.__file__).parents[1])
    mine = [f for f in rep["files"] if "/collective/" in f["file"]]
    assert {Path(f["file"]).name for f in mine} >= {"allreduce.py", "allreduce_difftest.py", "__init__.py"}
    assert all(f["status"] == "pass" for f in mine), [(f["file"], f["problems"]) for f in mine]
    assert not [d for d in rep["duplicate_semantic_ids"] if d.startswith("tp2/")]
    adapter = next(f for f in mine if f["file"].endswith("allreduce_difftest.py"))
    assert adapter["adapter"] and adapter["semantic_id"] == A.SEMANTIC_ID


def test_difftest_adapter_cases_and_spec_are_torch_free():
    from verity_vllm.program.quarantine.collective import allreduce_difftest as D
    import random
    kinds = set()
    for i in range(10):
        c = D.cases(random.Random(0), i)
        assert c["statics"]["R"] == 2 and len(c["inputs"]) == 2
        assert all(inp["dtype"] == "bfloat16" and inp["shape"] == [c["statics"]["N"]] for inp in c["inputs"])
        kinds.add(c["inputs"][0]["kind"])
        assert D.spec(c["statics"]).id == f"AllReduceSumBf16_v1{{N={c['statics']['N']},R=2}}"
    assert {"zeros", "max", "subnormal", "alternating", "random"} <= kinds


# --- the linkage tool: synthetic ranks ------------------------------------------------------------------------------------

def _rank_events(rank: int, *, ar_shape=(4, 8), gather=True, steps=(0,), stream=None):
    stream = stream if stream is not None else 1000 + rank
    hdr = RunHeader("verity-capture/log/v1", "2026-09-17T00:00:00Z", "pid", {"parallelism": {"tensor_parallel_size": 2}}, {"repo": "x"},
                    {"requests": []}, notes={"tp2": {"rank": rank, "world_size": 2}})
    evs = [hdr, StorageAlloc(1, 0x10, 64 * 8 * 2, "cuda:0", 0, "param"), Root(_v(1, (64, 8)), "weight", "model.embed_tokens.weight")]
    eid, alloc = 10, 100
    for step in steps:
        a_out, a_res, a_g, a_gres = alloc, alloc + 1, alloc + 2, alloc + 3
        alloc += 10
        evs += [
            StorageAlloc(a_out, 0x100 + a_out, 4 * 8 * 2, "cuda:0", 0, "op"),
            DispatchEnter(eid, None, step, "forward", "aten.empty.memory_format", "sig", [], {}, stream, "model.layers.0.self_attn.o_proj"),
            DispatchExit(eid, _v(a_out, ar_shape)),
            TritonLaunch(eid + 1, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", "sha",
                         {"a_ptr": _v(50 + step, (4, 4)), "b_ptr": _v(1, (8, 4)), "c_ptr": _v(a_out, ar_shape)}, {}, [1], 4, 3, stream, "model.layers.0.self_attn.o_proj"),
            DispatchEnter(eid + 2, None, step, "forward", "vllm.all_reduce.default", "vllm::all_reduce(Tensor tensor, str group_name) -> Tensor",
                          [_v(a_out, ar_shape), "tp:0"], {}, stream, "model.layers.0.self_attn.o_proj"),
            StorageAlloc(a_res, 0x100 + a_res, 4 * 8 * 2, "cuda:0", 0, "op"),
            DispatchExit(eid + 2, _v(a_res, ar_shape)),
            DispatchEnter(eid + 3, None, step, "forward", "_C.fused_add_rms_norm.default", "sig", [_v(a_res, ar_shape), _v(60, ar_shape), _v(61, (8,))], {}, stream,
                          "model.layers.0.post_attention_layernorm"),
            DispatchExit(eid + 3, None),
            TritonLaunch(eid + 4, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", "sha",
                         {"a_ptr": _v(a_res, ar_shape), "b_ptr": _v(1, (8, 8)), "c_ptr": _v(62, ar_shape)}, {}, [1], 4, 3, stream, "model.layers.0.mlp.gate_up_proj"),
            StorageFree(a_res, False),
        ]
        eid += 5
        if gather:
            evs += [
                StorageAlloc(a_g, 0x100 + a_g, 1 * 8 * 2, "cuda:0", 0, "op"),
                DispatchEnter(eid, None, step, "forward", "aten.empty.memory_format", "sig", [], {}, stream, "lm_head"),
                DispatchExit(eid, _v(a_g, (1, 8))),
                TritonLaunch(eid + 1, None, step, "forward", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent", "sha",
                             {"a_ptr": _v(63, (1, 8)), "b_ptr": _v(1, (8, 8)), "c_ptr": _v(a_g, (1, 8))}, {}, [1], 4, 3, stream, "lm_head"),
                DispatchEnter(eid + 2, None, step, "forward", "vllm.all_gather.default", "vllm::all_gather(Tensor tensor, int dim, int world_size, str group_name) -> Tensor",
                              [_v(a_g, (1, 8)), -1, 2, "tp:0"], {}, stream, "logits_processor"),
                StorageAlloc(a_gres, 0x100 + a_gres, 1 * 16 * 2, "cuda:0", 0, "op"),
                DispatchExit(eid + 2, _v(a_gres, (1, 16))),
                DispatchEnter(eid + 3, None, step, "sample", "aten.alias.default", "sig", [_v(a_gres, (1, 16))], {}, stream, "sampler"),
                DispatchExit(eid + 3, _v(a_gres, (1, 16))),
            ]
            eid += 4
    return evs


def test_extract_and_link_two_synthetic_ranks_with_instances():
    sh = {r: CL.extract_rank_shard(_rank_events(r), rank=r) for r in (0, 1)}
    for r in (0, 1):
        assert sh[r]["schema"] == SCHEMA_SHARD and sh[r]["rank"] == r and sh[r]["world_size"] == 2
        assert sh[r]["rows_by_kind"] == {"all_reduce": 1, "all_gather": 1} and sh[r]["communicators"] == ["tp:0"]
        for row in sh[r]["launches"]:
            assert validate_launch(row) == []
    ar = sh[0]["launches"][0]
    assert ar["kind"] == "all_reduce" and ar["k"] == 0 and ar["seq"] == 0 and ar["ordinal_in_step"] == 1 and ar["reduce_op"] == "sum"
    assert ar["count"] == 32 and ar["dtype"] == "bfloat16" and ar["operand"]["alloc"] == 100 and ar["result"]["alloc"] == 101
    assert [t["op"] for t in ar["operand_touches"]] == ["aten.empty.memory_format", "vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent"]
    assert ar["operand_last_touch"]["kind"] == "triton-arg" and ar["operand_last_touch"]["arg"] == "c_ptr" and ar["operand_last_touch"]["event"] == 11
    assert [c["event"] for c in ar["result_consumers"]] == [13, 14] and ar["result_first_consumer"]["op"] == "_C.fused_add_rms_norm.default"
    assert ar["result_freed"] is True and not ar["operand_is_root"]
    ag = sh[0]["launches"][1]
    assert ag["kind"] == "all_gather" and ag["k"] == 1 and ag["scalars"] == [-1, 2, "tp:0"] and ag["result"]["numel"] == 16
    assert ag["result_first_consumer"]["op"] == "aten.alias.default" and ag["result_freed"] is False

    inst = {0: [{"step": 0, "event": 11, "node": 3, "spec": "Gemm_v1{K=4,N=8}", "row": 0, "rows": 4, "pattern": "gemm.matmul_kernel_persistent",
                 "operand_views": [], "output_views": [{"alloc": 100, "offset": 0, "shape": [4, 8]}]},
                {"step": 0, "event": 13, "node": 4, "spec": "RMSNormFusedCuda_v1{N=8}", "row": 0, "rows": 4, "pattern": "rmsnorm.fused_add_rms_norm",
                 "operand_views": [], "output_views": []}]}
    link = CL.link_shards(sh, inst)
    assert link["schema"] == SCHEMA_LINK and link["ranks"] == [0, 1] and link["world_size"] == 2
    assert link["n_links"] == 2 and link["by_status"] == {"linked": 2} and link["by_kind"] == {"all_reduce": 1, "all_gather": 1}
    l0, l1 = link["links"]
    assert l0["key"] == {"communicator": "tp:0", "step": 0, "k": 0} and l0["definition"]["id"] == "AllReduceSumBf16_v1{N=32,R=2}"
    assert l0["definition"]["conformance"].startswith("quarantine")
    assert set(l0["required_writes"]) == {"0", "1"} and l0["required_writes"]["1"]["alloc"] == 101
    assert l0["ranks"]["0"]["producer"]["event"] == 11 and l0["ranks"]["0"]["producer_instance"] == [
        {"node": 3, "spec": "Gemm_v1{K=4,N=8}", "row": 0, "rows": 4, "pattern": "gemm.matmul_kernel_persistent", "event": 11, "step": 0}]
    assert l0["ranks"]["1"]["producer_instance"] is None          # rank 1 had no instances file
    assert [c["consumer_event"] for c in l0["ranks"]["0"]["consumer_instances"]] == [13]
    assert l1["key"]["k"] == 1 and l1["definition"]["id"] == "AllGather2_v1{N=8}"
    assert link["producer_instances_joined"] == 1 and link["consumer_instances_joined"] == 1
    assert link["missing_in_record"] == list(MISSING_IN_RECORD)
    assert "links 2" in CL.summary(link)


def test_link_reports_rank_mismatch_and_missing_rank():
    sh = {0: CL.extract_rank_shard(_rank_events(0), rank=0), 1: CL.extract_rank_shard(_rank_events(1, ar_shape=(4, 16)), rank=1)}
    link = CL.link_shards(sh)
    assert link["by_status"] == {"rank-mismatch": 1, "linked": 1}
    bad = link["links"][0]
    assert bad["agreement"]["operand_geometry"] is False and bad["agreement"]["result_geometry"] is False and bad["agreement"]["op"] is True
    sh = {0: CL.extract_rank_shard(_rank_events(0), rank=0), 1: CL.extract_rank_shard(_rank_events(1, gather=False), rank=1)}
    link = CL.link_shards(sh)
    assert link["by_status"] == {"linked": 1, "missing-rank": 1}
    assert list(link["links"][1]["ranks"]) == ["0"]


def test_step_filter_keeps_communicator_positions_consistent():
    full = CL.extract_rank_shard(_rank_events(0, steps=(0, 1, 2)), rank=0)
    only2 = CL.extract_rank_shard(_rank_events(0, steps=(0, 1, 2)), rank=0, steps={2})
    assert full["n_rows"] == 6 and only2["n_rows"] == 2 and only2["kept_steps"] == [2]
    kept = {(r["step"], r["k"], r["seq"], r["kind"]) for r in only2["launches"]}
    assert kept == {(2, 0, 4, "all_reduce"), (2, 1, 5, "all_gather")}
    assert kept <= {(r["step"], r["k"], r["seq"], r["kind"]) for r in full["launches"]}


def test_validate_launch_flags_missing_fields():
    row = CL.extract_rank_shard(_rank_events(0), rank=0)["launches"][0]
    assert validate_launch(row) == []
    broken = dict(row, communicator=None, result=None)
    probs = validate_launch(broken)
    assert any("communicator" in p for p in probs) and any("result" in p for p in probs)
    assert set(COLLECTIVE_KINDS.values()) == {"all_reduce", "all_gather", "reduce_scatter", "vocab_shard_lookup"}


# --- the linkage tool: the ov-tp2 evidence shards -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def evidence_link():
    shards = {r: CL.load_shard(FIX / f"collective_shard.steps03.rank{r}.json.gz") for r in (0, 1)}
    inst = {0: CL.load_instances(FIX / "instances.shard.rank0.jsonl")}
    return shards, CL.link_shards(shards, inst)


def test_evidence_shards_are_the_lanes_collective_census(evidence_link):
    shards, link = evidence_link
    for r in (0, 1):
        sh = shards[r]
        assert sh["rank"] == r and sh["world_size"] == 2 and sh["profile_id"] == "ceff2931bd7e"
        assert sh["parallelism"] == {"tensor_parallel_size": 2, "pipeline_parallel_size": 1}
        assert sh["events"] == 53364 and sh["launch_records"] == 19970 + 1926 and len(sh["steps"]) == 16
        assert sh["kept_steps"] == [0, 3] and sh["n_rows"] == 118
        assert sh["rows_by_kind"] == {"vocab_shard_lookup": 2, "all_reduce": 114, "all_gather": 2}
        assert sh["rows_per_step"] == {"0": 59, "3": 59} and sh["communicators"] == ["tp:0"]
        assert all(validate_launch(row) == [] for row in sh["launches"])
        assert {row["stream"] for row in sh["launches"]} == {380402784 if r == 0 else 375762000}
    # rank 0 and rank 1 emit the identical event sequence (MECHANISM §0.1): same event ids and alloc ids per position
    for a, b in zip(shards[0]["launches"], shards[1]["launches"]):
        assert (a["event"], a["step"], a["k"], a["op"], a["module"]) == (b["event"], b["step"], b["k"], b["op"], b["module"])
        assert a["operand"] == b["operand"] and a["result"] == b["result"]


def test_evidence_link_joins_every_collective_on_both_ranks(evidence_link):
    _, link = evidence_link
    assert link["world_size"] == 2 and link["communicators"] == ["tp:0"] and link["steps"] == [0, 3]
    assert link["n_links"] == 118 and link["by_status"] == {"linked": 118}
    assert link["by_kind"] == {"vocab_shard_lookup": 2, "all_reduce": 114, "all_gather": 2}
    assert link["links_per_step"] == {"0": 59, "3": 59} and link["invalid_rows"] == []
    # per step: 57 all-reduces on tp:0 (k = 0..56) then the logits all-gather (k = 57); the vocab lookup is rank-local (no k)
    for step in (0, 3):
        ks = [(l["key"]["k"], l["kind"]) for l in link["links"] if l["key"]["step"] == step and l["key"]["communicator"] == "tp:0"]
        assert ks == [(k, "all_reduce") for k in range(57)] + [(57, "all_gather")]
    # every all-reduce: N = rows x 1536 (32 prompt rows at step 0, 1 decode row at step 3), bf16, sum, the quarantine definition, and a
    # required (today unjournaled) write of the result on BOTH ranks with identical geometry
    for l in link["links"]:
        if l["kind"] != "all_reduce":
            continue
        rows = 32 if l["key"]["step"] == 0 else 1
        assert l["count"] == rows * 1536 and l["dtype"] == "bfloat16" and l["reduce_op"] == "sum"
        assert l["definition"]["id"] == f"AllReduceSumBf16_v1{{N={rows * 1536},R=2}}"
        assert set(l["required_writes"]) == {"0", "1"}
        assert l["required_writes"]["0"]["shape"] == [rows, 1536] == l["required_writes"]["1"]["shape"]
        assert l["ranks"]["0"]["result"]["alloc"] == l["ranks"]["1"]["result"]["alloc"]
        assert all(l["agreement"].values())


def test_evidence_link_producers_and_consumers_match_the_mechanism_note(evidence_link):
    _, link = evidence_link
    by = {(l["key"]["step"], l["key"]["communicator"], l["key"]["k"]): l for l in link["links"]}
    # embedding all-reduce (step 0, first on tp:0): operand a383 written by event 87 `_C.vocab_parallel_embedding`, result a384 read by
    # 90 reshape / 91 empty_like / 92 `_rms_norm_kernel` (input_layernorm) -- MECHANISM §1.4 / §1.5
    emb = by[(0, "tp:0", 0)]
    r0 = emb["ranks"]["0"]
    assert emb["module"] == "model.embed_tokens" and r0["event"] == 88 and r0["ordinal_in_step"] == 2
    assert r0["operand"]["alloc"] == 383 and r0["producer"]["event"] == 87 and r0["producer"]["op"] == "_C.vocab_parallel_embedding.default"
    assert r0["result"]["alloc"] == 384 and [c["event"] for c in r0["consumers"]] == [90, 91, 92]
    assert r0["consumers"][2]["op"].endswith("_rms_norm_kernel") and r0["consumers"][2]["arg"] == "input_ptr"
    # o_proj all-reduce of layer 0: operand a391 <- event 125 Gemm (c_ptr), result a392 -> 128 fused_add_rms_norm
    o = by[(0, "tp:0", 1)]["ranks"]["0"]
    assert by[(0, "tp:0", 1)]["module"] == "model.layers.0.self_attn.o_proj" and o["event"] == 126
    assert o["operand"]["alloc"] == 391 and o["producer"]["event"] == 125 and o["producer"]["op"].endswith("matmul_kernel_persistent") and o["producer"]["arg"] == "c_ptr"
    assert o["result"]["alloc"] == 392 and o["first_consumer"]["event"] == 128 and o["first_consumer"]["op"] == "_C.fused_add_rms_norm.default"
    d = by[(0, "tp:0", 2)]["ranks"]["0"]
    assert by[(0, "tp:0", 2)]["module"] == "model.layers.0.mlp.down_proj" and d["event"] == 137 and d["operand"]["alloc"] == 395 and d["result"]["alloc"] == 396
    assert d["producer"]["event"] == 136 and d["first_consumer"]["event"] == 139
    # decode step 3, layer 5 (MECHANISM §1.4 second table): o_proj 4461 a1469 -> a1470 -> 4463; down_proj 4472 a1473 -> a1474 -> 4474
    o5 = by[(3, "tp:0", 1 + 2 * 5)]["ranks"]["0"]
    assert by[(3, "tp:0", 11)]["module"] == "model.layers.5.self_attn.o_proj" and o5["event"] == 4461
    assert o5["operand"]["alloc"] == 1469 and o5["result"]["alloc"] == 1470 and o5["first_consumer"]["event"] == 4463
    d5 = by[(3, "tp:0", 12)]["ranks"]["0"]
    assert d5["event"] == 4472 and d5["operand"]["alloc"] == 1473 and d5["result"]["alloc"] == 1474 and d5["first_consumer"]["event"] == 4474
    # the logits all-gather: rank operand [1,75968] <- lm_head Gemm, result [1,151936] -> alias -> sampler
    g = by[(0, "tp:0", 57)]
    assert g["kind"] == "all_gather" and g["module"] == "logits_processor" and g["count"] == 75968 and g["definition"]["id"] == "AllGather2_v1{N=75968}"
    assert g["ranks"]["0"]["event"] == 1386 and g["ranks"]["0"]["operand"]["shape"] == [1, 75968] and g["ranks"]["0"]["result"]["shape"] == [1, 151936]
    assert g["ranks"]["0"]["producer"]["op"].endswith("matmul_kernel_persistent") and g["ranks"]["0"]["first_consumer"]["event"] == 1387
    # every all-reduce operand is produced by the row-parallel Gemm (o_proj / down_proj) except the embedding one (masked lookup)
    prods = link["producers_by_op"]
    assert prods["all_reduce|vllm.model_executor.determinism.batch_invariant.matmul_kernel_persistent"] == 2 * 2 * 56
    assert prods["all_reduce|_C.vocab_parallel_embedding.default"] == 2 * 2
    assert link["first_consumers_by_op"]["all_reduce|_C.fused_add_rms_norm.default"] == 2 * 2 * 56
    # the vocab lookup's operand is a Root (the weight shard), not a produced value; per-rank statics = the shard
    v0 = by[(0, "local", 1)]
    assert v0["kind"] == "vocab_shard_lookup" and v0["ranks"]["0"]["operand_root"] == {"role": "weight", "name": "model.embed_tokens.weight"}
    assert v0["ranks"]["0"]["producer"] is None and v0["ranks"]["0"]["rank_statics"]["org_vocab_start_index"] == 0
    assert v0["ranks"]["1"]["rank_statics"]["org_vocab_start_index"] == 75968
    assert v0["definition"]["id"] == {"0": "EmbeddingShard_v1{VS=75968,H=1536,START=0}", "1": "EmbeddingShard_v1{VS=75968,H=1536,START=75968}"}
    assert v0["ranks"]["0"]["first_consumer"]["event"] == 88
    assert link["root_operands"] == 4


def test_evidence_link_agrees_with_the_fold_gap_and_the_empty_instances(evidence_link):
    _, link = evidence_link
    # the fold's first insufficiency point is the first consumer of the embedding all-reduce's result, on both ranks
    for r in (0, 1):
        gap = json.loads((FIX / f"fold_gap.shard.rank{r}.json").read_text())
        fip = gap["first_insufficiency_point"]
        emb = next(l for l in link["links"] if l["key"] == {"communicator": "tp:0", "step": 0, "k": 0})
        pr = emb["ranks"][str(r)]
        assert fip["producer_of_that_allocation"]["event"] == pr["event"] == 88
        assert fip["unresolved_view"]["alloc"] == pr["result"]["alloc"] == 384
        assert fip["event"] in [c["event"] for c in pr["consumers"]]
        assert fip["producer_of_that_allocation"]["operands"][0]["alloc"] == pr["operand"]["alloc"]
        assert gap["unsupported_ops"] == {"_C.vocab_parallel_embedding.default": 16, "vllm.all_reduce.default": 912, "vllm.all_gather.default": 16}
    # the fold produced no instances, so nothing joins: the record says so instead of inventing a producer instance
    assert link["instances_given"] == {"0": 0, "1": 0}
    assert link["producer_instances_joined"] == 0 and link["consumer_instances_joined"] == 0
    assert all(pr["producer_instance"] is None for l in link["links"] for pr in l["ranks"].values())


def test_evidence_body_summary_shapes_match_the_links(evidence_link):
    _, link = evidence_link
    body = json.loads((FIX / "tp2_body.summary.json").read_text())
    assert body["summary"]["body_ok"] == 116
    step0 = [l for l in link["links"] if l["key"]["step"] == 0 and l["kind"] == "all_reduce"]
    for ex in body["checked_examples"]:
        if ex.get("status") != "ok" or ex["op"] != "vllm.all_reduce.default":
            continue
        l = step0[ex["ordinal"]]
        assert l["module"] == ex["module"] and l["ranks"]["0"]["operand"]["shape"] == ex["shape_in"] and l["ranks"]["0"]["result"]["shape"] == ex["shape_out"]
        assert ex["body"].startswith("bf16(sum_r f32(shard_r))") and ex["out_sha_equal_across_ranks"] is True
        assert l["definition"]["id"] == "AllReduceSumBf16_v1{N=49152,R=2}"


def test_cli_round_trip(tmp_path):
    out = tmp_path / "link.json"
    rc = CL.main(["link", str(FIX / "collective_shard.steps03.rank0.json.gz"), str(FIX / "collective_shard.steps03.rank1.json.gz"),
                  "--instances", f"0={FIX / 'instances.shard.rank0.jsonl'}", "--out", str(out)])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["n_links"] == 118 and rec["by_status"] == {"linked": 118}
    assert CL.main(["summary", str(out)]) == 0
