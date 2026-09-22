"""`partition` / `query_artifact` against the real `gateset` (Query API owner) and `boundary` (boundary owner):
fast path vs sweep agreement on real families of small b1 Programs (exact, gap, overlap, `concat` descending twice
into one node), width by the real `w_out_bound` / `w_out`, `validate_vu_partition` end to end with the use
document G8 reads, real `Query` identities, and -- when the captured record is in the tree -- the canonical VU
query over a 5e8-gate Serve program (structural: seconds, no gate visit)."""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

from verity.ir import codec
from verity_vllm.ir import partition as PT
from verity_vllm.ir import query_artifact as QA
from verity.ir.defs import bind
from verity.ir.parts import Limits
from verity.ir.program import Program
from verity_vllm.registry import b1

GS = pytest.importorskip("verity.ir.query_ast", reason="verity.ir.query_ast not merged (owner qapi)")
BD = pytest.importorskip("verity_vllm.ir.boundary", reason="verity_vllm.ir.boundary not merged (owner qbound)")

W = 32


@pytest.fixture(scope="module")
def gemm() -> Program:
    return Program(bind(b1.Gemm, K=32, N=6))


@pytest.fixture(scope="module")
def rms() -> Program:
    return Program(bind(b1.RMSNormTriton, N=32, EPS=1e-5))


@pytest.fixture(scope="module")
def gumbel() -> Program:
    from verity_vllm.registry.quarantine.ov_sampling import gumbel as G
    return Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))


def _agree(program: Program, text: str, expected: str) -> tuple[PT.PartitionVerdict, PT.PartitionVerdict]:
    fam = GS.evaluate(program, text)
    fast = PT.validate_partition(program, fam)
    sweep = PT._sweep(program, fam, Limits())
    assert fast.detail["path"] == "tiling", (text, fast.codes, fast.detail)
    assert fast.status == sweep.status == expected, (text, fast.status, fast.codes, sweep.status, sweep.codes)
    assert fast.detail["sets"] == fam.count()
    return fast, sweep


# -- fast path vs sweep on real families ------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["default(32)", "default(16)",
                                  'override(default(32), {"//Gemm_v1{*}": members("//Gemm_v1{*}", "@batch:GemmCoordinate_v1")})',
                                  'concat(instances("//DotBf16_v1"), rest())',
                                  'concat(rest(), instances("//DotBf16_v1"))'])
def test_exact_families_gemm(gemm, text):
    _agree(gemm, text, "exact")


@pytest.mark.parametrize("text", ['members("//Gemm_v1{*}", "@batch:GemmCoordinate_v1")',   # Input nodes uncovered
                                  '//GemmCoordinate_v1', 'instances("//DotBf16_v1")'])
def test_gap_families_gemm(gemm, text):
    fast, _ = _agree(gemm, text, "gap")
    assert "tiling-unassigned" in fast.codes


@pytest.mark.parametrize("text", ['concat(instances("//DotBf16_v1"), instances("//DotBf16_v1"))',
                                  'concat(instances("//DotBf16_v1"), default(32))'])
def test_overlap_families_gemm(gemm, text):
    fast, _ = _agree(gemm, text, "overlap")
    assert "tiling-members-overlap" in fast.codes


def test_concat_rest_descends_twice_into_one_node(gemm):
    """`concat(A, rest())` is described by two `descend` entries on the coordinates node into two plans that each
    cover part of the callee body: the plan *group* tiles it exactly."""
    fam = GS.evaluate(gemm, 'concat(instances("//DotBf16_v1"), rest())')
    t = PT._Tiling(gemm, fam.tiling())
    root = t.plans[t.root_plan]
    coord_desc = [child for descents in root.descends.values() for _, _, child in descents]
    assert len(coord_desc) == 2 and len(set(coord_desc)) == 2
    assert any(len(g) == 2 for g in t.groups)
    assert not t.problems and t.count() == fam.count() == 32 + 192 + 6 * 2
    assert t.instances[tuple(sorted(coord_desc))] == 6


def test_fast_path_visits_no_gate(gemm, monkeypatch):
    fam = GS.evaluate(gemm, "default(32)")
    calls = []
    C = gemm.circuit
    orig = type(C).gate
    monkeypatch.setattr(type(C), "gate", lambda self, g: (calls.append(g), orig(self, g))[1])
    v = PT.validate_partition(gemm, fam)
    assert v.status == "exact" and v.detail["path"] == "tiling" and calls == []


def test_rmsnorm_zip_family_exact_and_narrow(rms):
    text = 'override(default(32), {"//RMSNormTriton_v1{*}": concat(zip(".", ["@batch:F32Mul_v1", "@batch:F2fpBf16_v1"]), rest())})'
    fam = GS.evaluate(rms, text)
    fast, _ = _agree(rms, text, "exact")
    w = PT.validate_width(rms, fam, W)
    assert w.status == "ok" and w.max_seen is not None and w.max_seen <= W and not w.violators and not w.unsupported
    assert w.detail["path"] == "tiling" and w.detail["sets"] == fam.count()


_FIXTURES = ["moe_router", "moe_expert_gemm", "moe_sum", "scan_carry", "scan_carry_only", "attention_head", "attention",
             "attn_block_first", "gumbel", "token_select", "embedding", "fan_out", "dead_return", "reexport", "dead_param",
             "tiny_io", "rmsnorm_fused", "layer_norm_aten"]


@pytest.mark.parametrize("name", _FIXTURES)
def test_fast_path_equals_sweep_on_fixture_families(name):
    """Scan / MoE / attention / degenerate-IO fixtures (qtest's registry): `default(32)`, `default(8)`, the canonical VU
    query and the root-body member family all take the tiling path and agree with the interval sweep."""
    FX = pytest.importorskip("tests.ir.fixtures_query", reason="qtest fixtures not merged")
    VC = pytest.importorskip("verity_capture.vu_canonical", reason="canonical query module (owner qtest)")
    P = FX.fixture(name)
    for q in ("default(32)", "default(8)", VC.canonical_vu_query("ampere"), 'members("/", "@*")'):
        fam = GS.evaluate(P, q)
        fast = PT.validate_partition(P, fam)
        sweep = PT._sweep(P, fam, Limits())
        assert fast.detail["path"] == "tiling" and fast.status == sweep.status == "exact", (name, q, fast, sweep)
        assert fast.detail["sets"] == fam.count() == sweep.detail["sets"]
        w = PT.validate_width(P, fam, W)
        assert w.detail["path"] == "tiling" and w.status in ("ok", "violation") and not w.unsupported, (name, q, w)
        if w.status == "violation":  # every refutation is by an exact w_out, never by a bound
            assert w.detail["exact_evaluated"] >= len(w.violators) > 0 and w.max_seen > W


# -- width with the real boundary -----------------------------------------------------------------------------------


def test_width_real_boundary_gemm(gemm):
    fam = GS.evaluate(gemm, "default(32)")
    w = PT.validate_width(gemm, fam, W)
    assert w.status == "ok" and w.max_seen == 16 and w.detail["path"] == "tiling"
    assert w.detail["bound_certified"] == fam.count() and w.detail["exact_evaluated"] == 0
    # the whole root as one set: w_out = the designated outputs (6 x bf16 = 96 bits) > 32 -> a real violation by exact w_out
    whole = GS.evaluate(gemm, "root()")
    v = PT.validate_width(gemm, whole, W)
    assert v.status == "violation" and v.violators == (0,) and v.max_seen == 96
    assert v.detail["exact_evaluated"] == 1
    # the same set at w_max = 96 certifies (exact value is the verdict, not the bound)
    assert PT.validate_width(gemm, whole, 96).status == "ok"
    # bound-only certification of a set whose bound is tight: a single DotBf16 chain (32-bit accumulator escapes)
    dots = GS.evaluate(gemm, 'instances("//DotBf16_v1")')
    d = PT.validate_width(gemm, dots, W)
    assert d.status == "ok" and d.max_seen == 32 and d.detail["exact_evaluated"] == 0
    assert PT.validate_width(gemm, dots, 31).status == "violation"


def test_width_budget_exhausted_is_unsupported_not_a_verdict(gemm):
    whole = GS.evaluate(gemm, "root()")
    lim = Limits()
    lim.max_members_exact = 0
    v = PT.validate_width(gemm, whole, W, lim)
    assert v.status == "unsupported" and v.unsupported == (0,) and not v.violators
    assert "limit-members-exact" in v.detail["codes"]


# -- validate_vu_partition end to end, use document ---------------------------------------------------------------------


def test_vu_partition_ok_and_use_document(gemm):
    q = GS.parse("default(32)")
    v = PT.validate_vu_partition(gemm, q, W)
    assert v.status == "ok" and v.partition.status == "exact" and v.width.status == "ok" and v.readiness_32.ok
    assert v.count == 32 + 192 + 6
    qdoc, use = QA.vu_use_document(q, gemm, v, W)
    assert use["schema"] == "verity-ir/query-use/v1" and use["use"] == {"kind": "vu-partition", "w_max": W}
    assert use["verdict"]["status"] == "ok" and use["verdict"]["count"] == v.count
    assert qdoc["schema"] == "verity-ir/query/v1" and "kind" not in qdoc and "w_max" not in qdoc
    assert qdoc["program_digest"] == codec.program_digest(gemm)
    assert qdoc["query_id"] == GS.query_id(q) == QA.query_id(q) == QA.query_id("default(32)")
    assert use["binding_id"] == qdoc["binding_id"] == QA.binding_id(qdoc["query_id"], qdoc["program_digest"])
    assert QA.check_use_document(use, codec.program_digest(gemm), qdoc) == []
    assert QA.check_query_document(qdoc) == []
    json.dumps(use)  # serialisable as written
    assert QA.check_use_document(use, "0" * 64, qdoc), "a foreign program digest must be rejected"


def test_vu_partition_fail_gap_and_string_query(gemm):
    v = PT.validate_vu_partition(gemm, 'instances("//DotBf16_v1")', W)
    assert v.status == "fail" and v.partition.status == "gap" and v.count == 6
    _, use = QA.vu_use_document('instances("//DotBf16_v1")', gemm, v, W)
    assert use["verdict"]["status"] == "fail" and "tiling-unassigned" in use["verdict"]["partition"]["codes"]


def test_vu_partition_not_ready_on_gumbel(gumbel):
    """Gumbel carries Const64 gates: readiness fails at w_max = 32 even where the partition and width pass."""
    v = PT.validate_vu_partition(gumbel, "default(32)", W)
    assert not v.readiness_32.ok and any(x.w == 64 for x in v.readiness_32.violators)
    assert v.status in ("fail", "not-ready")
    if v.partition.status == "exact" and v.width.status == "ok":
        assert v.status == "not-ready"
    assert PT.readiness_32bit(gumbel).max_w == 64


def test_real_query_ids_are_ast_identities(gemm):
    a, b = GS.parse('instances("//DotBf16_v1")'), GS.parse("//DotBf16_v1")
    assert QA.query_id(a) == QA.query_id(b) == GS.query_id(a)          # spellings of one AST share the id
    assert QA.query_id(GS.parse("default(16)")) != QA.query_id(GS.parse("default(32)"))
    assert QA.query_json(a) == GS.to_json(a)
    assert QA.binding_id(QA.query_id(a), codec.program_digest(gemm)) != QA.binding_id(QA.query_id(GS.parse("default(16)")), codec.program_digest(gemm))


# -- a captured Serve program (structural only) ----------------------------------------------------------------------------


def _real_record() -> Path | None:
    root = Path(__file__).resolve().parents[2]
    for base in ("fixtures/e2e-composed", "verity_capture/tests/fixtures"):
        d = root / base
        if d.is_dir():
            for pj in sorted(d.rglob("program.json")):
                rec = json.loads(pj.read_text())
                name = rec.get("program", rec).get("descriptor")
                if name and (pj.parent / name).exists():
                    return pj
    return None


def test_canonical_query_on_captured_program_is_structural():
    pj = _real_record()
    if pj is None:
        pytest.skip("no captured program record in the sparse tree")
    VC = pytest.importorskip("verity_capture.vu_canonical", reason="canonical query module (owner qtest)")
    from verity.ir.codec import decode_program
    from verity.ir.defs import REGISTRY
    rec = json.loads(pj.read_text())
    path = pj.parent / rec.get("program", rec)["descriptor"]
    desc = json.load(gzip.open(path, "rt")) if path.name.endswith(".gz") else json.loads(path.read_text())
    dec = decode_program(desc, REGISTRY)
    P = Program.__new__(Program)
    P.fn = dec.program_fn_functions[0] if dec.program_fn_functions else None
    P.annotations = dec.annotations
    P.root = dec.root
    P._circuit = dec.circuit
    assert P.gates > 10**8
    q = VC.canonical_vu_query("ampere")
    t0 = time.time()
    v = PT.validate_vu_partition(P, q, VC.W_MAX)
    dt = time.time() - t0
    assert v.partition.status == "exact" and v.partition.detail["path"] == "tiling", (v.partition.codes, v.partition.detail)
    assert v.width.status == "ok" and v.width.max_seen == VC.W_MAX and not v.width.violators and not v.width.unsupported
    assert v.width.detail["path"] == "tiling" and v.width.detail["exact_evaluated"] == 0
    assert v.readiness_32.ok and v.status == "ok" and v.count > 10**8
    assert dt < 60, f"structural validation took {dt:.1f}s"
