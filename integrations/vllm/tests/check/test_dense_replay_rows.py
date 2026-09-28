"""The replay row kernel of the dense Definitions (`program.kernels.ir_rows`): on a Gemma-2-style dense Program every row is an evaluable
unit of the replay population, and each dense row's words are the IR reference's (edge words included)."""
from __future__ import annotations

import numpy as np

from tests.acquire.call_boundary_programs import DENSE, LP, PF, build_dir, dense_program
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.check.replay import coverage as COV
from verity_vllm.check.replay.driver import manifest_query
from verity_vllm.check.replay.index import ProgramIndex
from verity_vllm.check.replay.population import population
from verity_vllm.program.kernels import ir_rows
from verity_vllm.program.kernels.rows import EVALUATORS, ROWS
from verity_vllm.program.registry import dense
from verity_vllm.query.call_scope import specialization
from verity_vllm.query.manifest import format as RM
from verity_vllm.query.manifest.format import spec_family
from verity_vllm.query.program_view import spec_statics

PROMPT = [0, 1, 2, 3, 4, 5, 9]


def _fields(rng, V=16, H=8):
    def bf16(n, s=1.0):
        return (np.asarray(rng.standard_normal(n) * s, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    table = bf16(V * H).reshape(V, H)
    table[:4] = [[0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x7F7F]] * 4
    return {n: (table.reshape(-1) if c == V * H else bf16(c, 0.5)) for n, c, _b, _t in DENSE}


def test_every_row_of_a_dense_program_is_an_evaluable_unit(tmp_path):
    b, _at = dense_program()
    doc = b.doc(LP)
    pdir, man = build_dir(tmp_path, doc, [PF[len("model."):]])
    vus, ne, nef = population(ProgramIndex(doc, program_dir=pdir, query=manifest_query(man)), "r0")
    assert not [w for w in ne if "no registered evaluator" in w], dict(ne)
    fams = {v["family"] for v in vus}
    assert set(ir_rows.FAMILIES) - {"GeluTanhMul_v1"} <= fams and fams <= set(EVALUATORS)


def test_each_dense_row_is_the_ir_reference():
    b, _at = dense_program()
    doc = b.doc(LP)
    fields = _fields(np.random.default_rng(5))
    weights = [int(w) for n, *_ in DENSE for w in fields[n]]
    vals, seen = {}, set()
    for r in doc["rows"]:
        args = [[v for kind, a, lo, hi in g for v in (PROMPT[lo:hi] if kind == "p" and a == 0 else weights[lo:hi] if kind == "p" else vals[a][lo:hi])]
                for g in r["args"]]
        fam, st = spec_family(r["spec"]), spec_statics(r["spec"])
        vals[r["i"]] = list(evaluate(specialization(fam, st), *args))
        if fam in ir_rows.FAMILIES:
            # the replay reads statics as the manifest spells them (an f64 literal stays `{"f64":"<hex>"}` text)
            got = ROWS[fam].fn(RM.spec_statics(r["spec"]), [np.asarray(a, dtype=np.uint64) for a in args])
            assert got["0"] is got["out"] and got["out"].tolist() == vals[r["i"]], r["name"]
            seen.add(fam)
    assert seen == set(ir_rows.FAMILIES) - {"GeluTanhMul_v1"}


def test_gelu_tanh_mul_at_gemma_width_and_its_declines():
    rng = np.random.default_rng(6)
    I = 9216
    gu = (np.asarray(rng.standard_normal(2 * I) * 3, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    gu[:6] = [0x7FC0, 0xFF80, 0x7F80, 0x8000, 0x0001, 0x7F7F]
    got = ROWS["GeluTanhMul_v1"].fn({"I": I}, [gu])
    assert list(got) == ["0", "out"] and got["out"].dtype == np.uint16
    assert got["out"].tolist() == list(evaluate(bind(dense.GeluTanhMul, I=I), gu.tolist()))
    import pytest
    with pytest.raises(ir_rows.Declined, match="arg words"):
        ROWS["GeluTanhMul_v1"].fn({"I": I}, [gu[:-1]])
    with pytest.raises(ir_rows.Declined, match="past"):
        ROWS["GeluTanhMul_v1"].fn({"I": ir_rows.MAX_WORDS}, [np.zeros(2 * ir_rows.MAX_WORDS, dtype=np.uint16)])


def test_a_norm_scale_manifest_addresses_every_rsqrt_row_where_it_is_committed(tmp_path):
    """Under the norm-scale policy (a Commit with the norm tap) the ATen norm's `RsqrtF32{N=1}` output is `<norm module>/norm_scale`,
    family norm_scales: the replay's index spells it so, and every row's address is one the manifest names."""
    b, _at = dense_program()
    doc = b.doc(LP)
    pdir, man = build_dir(tmp_path, doc, [PF[len("model."):]], norm_scales=True)
    named = {(i["op_path"], i["output_member"]) for i in man["identities"]}
    scales = {(i["op_path"], i["output_member"]) for i in man["identities"] if i["family"] == "norm_scales"}
    assert scales and {m for _o, m in scales} == {"norm_scale"}

    def addresses(pi):
        return {(d["op_path"], m[0]) for r in doc["rows"] if (d := pi.describe(r)) for m in d["members"]}

    q = manifest_query(man)
    assert addresses(ProgramIndex(doc, program_dir=pdir, query=q, norm_scales=True)) <= named
    assert not addresses(ProgramIndex(doc, program_dir=pdir, query=q)) <= named
    at: dict = {}
    vus, ne, _nef = population(ProgramIndex(doc, program_dir=pdir, query=q, norm_scales=True), "r0", addresses=at)
    assert "RsqrtF32_v1" in {v["family"] for v in vus} and not [w for w in ne if "no registered evaluator" in w], dict(ne)
    qp = COV.query_population(man, at, set(), ranks=[0])
    checks = qp["query_checks"]
    assert checks["vus_outside_query"]["n"] == 0 and checks["identities_without_rows"]["n"] == 0 and checks["accounted"], checks
    assert qp["sampled_strata"]["by_family_reached"]["norm_scales"] == qp["query_total"]["by_family"]["norm_scales"]
    assert "norm_scales" not in {u["family"] for u in qp["uncovered_by_name"]}


def test_a_one_output_row_is_compared_under_its_one_member_however_the_module_spells_it(monkeypatch):
    """The ATen norm's rsqrt is `<norm module>/norm_scale` and its residual narrow `<norm module>/1`: the relation's one output is compared
    under that member, equal and unequal alike; a row requiring two members of a one-output relation is not evaluated, by name."""
    from types import SimpleNamespace
    from verity_vllm.check.replay import evaluate as E

    words = np.arange(4, dtype=np.uint32)
    monkeypatch.setattr(E, "_evaluate_row", lambda *a, **k: ir_rows.ir_row("RsqrtF32_v1", {"N": 1}, [np.array([0x40800000], np.uint32)]))
    rsqrt4 = ir_rows.ir_row("RsqrtF32_v1", {"N": 1}, [np.array([0x40800000], np.uint32)])["out"]

    def replay(members, committed):
        d = {"family": "RsqrtF32_v1", "statics": {"N": 1}, "op_path": "model.norm", "row": 0, "step": 0, "members": members}
        pi = SimpleNamespace(by_i={0: {"i": 0, "spec": "RsqrtF32_v1{N=1}"}}, describe=lambda r: d)
        store = SimpleNamespace(words=lambda *a: (committed, ""))
        return E.replay_vu(pi, store, None, {"i": 0, "request_id": "r0", "phase": "prefill"}, arrive=0)

    assert replay([("norm_scale", 1, "f32")], rsqrt4)["result"] is True
    assert replay([("norm_scale", 1, "f32")], rsqrt4 ^ 1)["result"] is False
    two = replay([("norm_scale", 1, "f32"), ("1", 1, "f32")], words[:1])
    assert two["result"] is None and "no member 'norm_scale'" in two["why"]
