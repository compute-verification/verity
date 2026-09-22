"""W1cF: audit-compiled-pass.md findings F2–F10 closed in ``verify_bindings`` / ``rows.verdict`` /
``relations_inductor`` / ``launchspec`` / ``kernel_identity``. CPU-only (no torch, no GPU)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from verity_vllm import rows
from verity_vllm.relations import CheckReport, VUResult
from verity_vllm_adapter import kernel_identity, launchspec, verify_bindings as VB

# --- fixtures ---------------------------------------------------------------------------------------

KERNEL_A = """import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics

@triton_heuristics.pointwise(size_hints={'x': 4096})
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    x0 = tl.program_id(0)
    tl.store(out_ptr0 + x0, tl.load(in_ptr0 + x0), None)
"""
KERNEL_B = KERNEL_A.replace("triton_poi_fused_1", "triton_per_fused_rms_2").replace("tl.load(in_ptr0 + x0)", "tl.load(in_ptr0 + x0) * 2")
GEMM = """import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics

@triton_heuristics.user_autotune(configs=[{'num_warps': 8, 'num_stages': 3}], triton_meta={'constants': {'NUM_SMS': 128}})
@triton.jit
def matmul_kernel_persistent(a_ptr, b_ptr, c_ptr, NUM_SMS: tl.constexpr):
    pass
"""


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def write_kernels(d: Path, *sources: str) -> Path:
    kd = d / "kernels"
    kd.mkdir(parents=True, exist_ok=True)
    for src in sources:
        (kd / f"{sha(src)}.py").write_text(src)
    return kd


def production_kernel_set(*sources: str) -> str:
    """kernel_identity.collect's rule: sorted set of (async_compile.triton name, body sha) pairs."""
    import re

    pairs = sorted({(re.search(r"^def (\w+)\(", s, re.M).group(1), sha(s)) for s in sources})
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def manifest_with(kernel_set: str | None, **extra):
    m = {"compilation": {"generated_kernels": ({"kernel_set_sha256": kernel_set} if kernel_set else {})},
         "vllm": {"version": "0.28.1", "wheel_sha256": "a" * 64}, "python": {"torch": "2.13.0+cu129", "triton": "3.7.1", "cuda": "12.9"},
         "gpu": {"name": "NVIDIA GeForce RTX 4090", "multi_processor_count": 128}}
    m.update(extra)
    return m


def common():
    return rows.Common("B0-p-20260907T0000Z", "B0", "p", "W1cF", "test")


def base_row(**over):
    row = dict(
        workload={"requests": 2, "generated_positions": 32, "input_len_range": [32, 32], "peak_concurrency_achieved": 2},
        production_changes="none",
        replay_restrictions=[],
        token_selection={"exact": 32, "attempted": 32},
        boundary_noninterference={"identical": True, "compared": 100, "mismatched_scalars": 0, "reference_method": "replay"},
        required_coverage={"bound": 10, "required": 10},
        authentication={"valid": 100, "attempted": 100},
        relation_checks={"same_runtime": {"exact": 50, "attempted": 50}, "independent": {"exact": 5, "attempted": 5}},
        structural={"result": "ok", "failing_invariant": None},
        negative={"rejected": 8, "attempted": 8},
    )
    row.update(over)
    return row


# --- F2: kernel set bound to production's ------------------------------------------------------------


def test_f2_kernel_set_recomputation_matches_collect_rule(tmp_path):
    kd = write_kernels(tmp_path, KERNEL_A, KERNEL_B)
    got = VB.kernel_set_of_dir(kd)
    assert got["kernel_set_sha256"] == production_kernel_set(KERNEL_A, KERNEL_B)
    assert got["files"] == 2 and not got["bad_hash"]
    b = VB.kernel_set_binding(tmp_path, manifest_with(production_kernel_set(KERNEL_A, KERNEL_B)))
    assert b["ok"] and b["problem"] is None and b["kernels"] == 2


def test_f2_comment_only_edit_rehashed_and_repointed_fails(tmp_path):
    """WA's M8: a comment-only edit, file renamed to its new sha, description re-pointed. The source
    set differs from production's -> structural problem."""
    edited = KERNEL_B + "# neutral comment\n"
    write_kernels(tmp_path, KERNEL_A, edited)
    b = VB.kernel_set_binding(tmp_path, manifest_with(production_kernel_set(KERNEL_A, KERNEL_B)),
                              production_kernels={"wrappers": [{"kernels": [{"name": "triton_poi_fused_1", "source_sha256": sha(KERNEL_A)},
                                                                            {"name": "triton_per_fused_rms_2", "source_sha256": sha(KERNEL_B)}]}]})
    assert not b["ok"] and "production's manifest recorded" in b["problem"]
    assert b["only_bundle"] == [f"triton_per_fused_rms_2:{sha(edited)[:12]}"]
    assert b["only_production"] == [f"triton_per_fused_rms_2:{sha(KERNEL_B)[:12]}"]
    row = rows.table1_row(common(), **base_row(structural={"result": "failed", "failing_invariant": f"kernel_set: {b['problem']}"}), bindings={"kernel_set": b})
    assert row["result"] == "FAIL" and row["first_blocker"].startswith("structural: kernel_set")


def test_f2_absent_recorded_hash_is_a_problem_and_eager_is_not_applicable(tmp_path):
    write_kernels(tmp_path, KERNEL_A)
    b = VB.kernel_set_binding(tmp_path, manifest_with(None))
    assert not b["ok"] and "unbound" in b["problem"]
    assert VB.kernel_set_binding(tmp_path / "eager-run", manifest_with(None)) is None


# --- F3: description digest -------------------------------------------------------------------------


class _Desc:
    def __init__(self, digest: bytes):
        self._d = digest

    def digest(self) -> bytes:
        return self._d


def test_f3_description_digest_binding():
    d = hashlib.sha256(b"desc").digest()
    assert VB.description_binding(None, _Desc(d)) is None
    ok = VB.description_binding({"description_digest": d.hex()}, _Desc(d))
    assert ok["ok"] and ok["problem"] is None
    bad = VB.description_binding({"description_digest": hashlib.sha256(b"other").hexdigest()}, _Desc(d))
    assert not bad["ok"] and "description_digest" in bad["problem"]
    none = VB.description_binding({}, _Desc(d))
    assert not none["ok"] and "no description_digest" in none["problem"]


# --- F4: checker exceptions are rejections ----------------------------------------------------------


def _result(status, op="layers.10.input_layernorm.rmsnorm", detail=None):
    return VUResult(canonical_id=f"p=x/r=r/t=1/ph=decode/l=10/op={op}/port=1", op=op, op_type="rmsnorm",
                    relation_id="inductor.fused.same_runtime.v1", klass="same_runtime", status=status, detail=detail)


def test_f4_relation_error_is_a_fail_not_partial():
    """WA's M3b: the description points at a wrong-but-valid kernel; the checker raises TypeError on
    every fused VU. Before: PARTIAL 'no executable checker' (errors are not 'attempted'). Now: FAIL."""
    report = CheckReport(challenge_digest=None, results=[_result("exact"), _result("error", detail="TypeError: too many positional arguments")])
    errs = VB.relation_errors_of(report)
    assert errs["count"] == 1 and "TypeError" in errs["first"] and errs["by_relation"] == {"inductor.fused.same_runtime.v1": 1}
    rc = report.relation_checks()
    assert rc["same_runtime"] == {"exact": 1, "attempted": 1}  # the error is not an attempt: W2's semantics unchanged
    row = rows.table1_row(common(), **base_row(relation_checks={"same_runtime": rc["same_runtime"], "independent": {"exact": 0, "attempted": 0}}),
                          relation_errors=errs)
    assert row["result"] == "FAIL"
    assert row["first_blocker"].startswith("relation_checks.error: 1 relation checker call(s) raised")
    # a Table 2 row whose worst status is `error` is a FAIL too, not "no executable checker"
    t2 = {"op_id": "layers.10.input_layernorm.rmsnorm", "decomposition": "row", "checker": "same_runtime", "status": "error",
          "binding": {"present": 1, "unresolved": 0}, "natural_checks": {"exact": 0, "attempted": 0}}
    v = rows.verdict(base_row(), [t2])
    assert v.result == "FAIL" and "relation checker raised" in v.first_blocker


def test_f4_no_errors_leaves_the_verdict_alone():
    report = CheckReport(challenge_digest=None, results=[_result("exact")])
    row = rows.table1_row(common(), **base_row(), relation_errors=VB.relation_errors_of(report))
    assert row["result"] == "PASS" and row["relation_errors"] == {"count": 0, "first": None, "by_relation": {}}


# --- F5: negatives with 0 attempted are NOT_RUN -----------------------------------------------------


def test_f5_zero_attempted_negative_is_not_run_and_downgrades_to_partial():
    negatives = [{"negative_id": "N01", "attempted": 4, "rejected": 4, "unexpectedly_accepted": 0},
                 {"negative_id": "N13", "attempted": 0, "rejected": 0, "unexpectedly_accepted": 0, "inconclusive": 2}]
    not_run = VB.negatives_not_run(negatives, ["N01", "N13", "N14"])  # N14 requested, never produced a result
    assert not_run == ["N13", "N14"]
    VB.mark_not_run(negatives, not_run)
    assert negatives[0]["status"] == "rejected" and negatives[1]["status"] == "NOT_RUN"
    row = rows.table1_row(common(), **base_row(negative={"rejected": 4, "attempted": 4}), negatives_not_run=not_run)
    assert row["result"] == "PARTIAL"
    assert row["first_blocker"] == "negative: applicable negative test(s) with 0 attempted mutations (NOT_RUN): N13, N14"
    assert rows.table1_row(common(), **base_row(), negatives_not_run=[])["result"] == "PASS"


# --- F7: verifier environment vs manifest -----------------------------------------------------------


def test_f7_environment_qualifications_not_fail():
    env = {"vllm": {"installed": True, "version": "0.28.1", "wheel_sha256": "b" * 64, "record_sha256": "r1"},
           "torch": {"installed": True, "version": "2.13.0+cu129", "record_sha256": "r2"},
           "triton": {"installed": True, "version": "3.8.0", "record_sha256": "r3"},
           "cuda": "12.9", "gpu": {"name": "NVIDIA H100 80GB HBM3", "multi_processor_count": 132}}
    q = VB.environment_qualifications(env, manifest_with("k", dists={"torch": {"record_sha256": "r2"}, "triton": {"record_sha256": "zz"}}))
    assert any(s.startswith("vllm wheel sha256 differs") for s in q)
    assert any(s.startswith("triton version differs: verifier 3.8.0 vs production 3.7.1") for s in q)
    assert any(s.startswith("triton installation differs") for s in q)
    assert not any(s.startswith("torch") for s in q)
    assert any(s.startswith("GPU differs") for s in q) and any(s.startswith("SM count differs") for s in q)
    row = rows.table1_row(common(), **base_row(), qualifications=q)
    assert row["result"] == "PASS" and row["qualifications"] == q  # a qualification is printed, never a FAIL
    same = {"vllm": {"installed": True, "version": "0.28.1", "wheel_sha256": "a" * 64}, "torch": {"installed": True, "version": "2.13.0+cu129"},
            "triton": {"installed": True, "version": "3.7.1"}, "cuda": "12.9", "gpu": {"name": "NVIDIA GeForce RTX 4090", "multi_processor_count": 128}}
    assert VB.environment_qualifications(same, manifest_with("k")) == []
    unmeasurable = {**same, "vllm": {"installed": True, "version": "0.28.1", "wheel_sha256": None}}
    assert VB.environment_qualifications(unmeasurable, manifest_with("k")) == ["vllm wheel sha256 not comparable: the verifier's installed wheel file is not measurable here (pip recorded no hash)"]


def test_f7_dist_identity_of_this_interpreter():
    d = VB.dist_identity("pytest")
    assert d["installed"] and d["version"] == pytest.__version__ and (d["record_sha256"] is None or len(d["record_sha256"]) == 64)
    assert VB.dist_identity("no-such-distribution-w1cf") == {"installed": False}


# --- F8: production's autotune choice pinned ---------------------------------------------------------


def test_f8_production_launch_config_from_best_config():
    entries = [{"kernel": "triton_poi_fused_1", "kernel_module_sha256": sha(KERNEL_A),
                "config": {"XBLOCK": 512, "num_warps": 8, "num_stages": 1}, "configs_hash": "h", "found_by_coordesc": False}]
    assert launchspec.production_launch_config(entries, sha(KERNEL_A)) == {"kwargs": {"XBLOCK": 512}, "num_warps": 8, "num_stages": 1}
    assert launchspec.production_launch_config(entries, sha(KERNEL_B)) is None
    assert launchspec.production_launch_config(None, sha(KERNEL_A)) is None
    assert launchspec.launch_config_from_best_config({"XBLOCK": 1024, "R0_BLOCK": 8, "num_warps": 4, "num_stages": 1, "configs_hash": "x", "found_by_coordesc": True}) == \
        {"kwargs": {"XBLOCK": 1024, "R0_BLOCK": 8}, "num_warps": 4, "num_stages": 1}


def test_f8_checker_prefers_production_config_and_says_which():
    ind_rel = pytest.importorskip("verity_vllm_adapter.relations_inductor")  # imports torch
    prod = {"kwargs": {"XBLOCK": 512}, "num_warps": 8, "num_stages": 1}
    rep = {"kwargs": {"XBLOCK": 1024}, "num_warps": 4, "num_stages": 1}
    assert ind_rel.choose_launch_config({"launch_config_production": prod, "launch_config": rep}) == (prod, "production")
    assert ind_rel.choose_launch_config({"launch_config": rep}) == (rep, "replay")
    assert ind_rel.choose_launch_config({}) == (None, "none")


def test_f8_f9_table2_limitation_states_assumption_and_config_source():
    class N:
        def __init__(self, relation, op):
            self.relation, self.op = relation, op

    used = {"a": {"source": "production"}, "b": {"source": "replay"}, "c": {"source": "single-config"}}
    a = VB.fused_limitation(N("inductor.fused.same_runtime.v1", "a"), used)
    assert a.startswith("capture-level relation") and "row-independent" in a and "production's recorded autotune best_config" in a
    b = VB.fused_limitation(N("inductor.fused.same_runtime.v1", "b"), used)
    assert "REPLAY's autotune choice" in b and "not recorded" in b
    assert "single launch config" in VB.fused_limitation(N("inductor.fused.same_runtime.v1", "c"), used)
    assert "row-independent" in VB.fused_limitation(N("inductor.fused.same_runtime.v1", "d"), used)  # unchecked node: assumption still stated
    assert VB.fused_limitation(N("unresolved", "e"), used) == "no executable relation"
    assert VB.fused_limitation(N("gemm.same_runtime.v1", "f"), used) is None


# --- F10: shipped normalised PTX ---------------------------------------------------------------------

PTX_A = """//
// Generated by LLVM NVPTX Back-End
//
.version 8.7
.target sm_89
.address_size 64
.file 1 "/workspace/cache/prod/inductor/ab/kernel.py"
.visible .entry triton_poi_fused_1(.param .u64 p0)
{
.loc 1 12 0
    ret;
}
    .section .debug_abbrev
    {
.b8 1
    }
"""


def _triton_cache(root: Path, name: str, ptx: str, key: str) -> None:
    d = root / key
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.ptx").write_text(ptx)
    (d / f"{name}.cubin").write_bytes(b"\x7fELF" + key.encode())
    (d / f"{name}.json").write_text(json.dumps({"num_warps": 8, "num_stages": 1, "hash": key}))


def test_f10_export_ptx_normalises_and_binds_the_set(tmp_path):
    cache = tmp_path / "triton"
    _triton_cache(cache, "triton_poi_fused_1", PTX_A, "KEY1")
    _triton_cache(cache, "triton_poi_fused_1", PTX_A.replace("/workspace/cache/prod", "/elsewhere"), "KEY2")  # same code, other path
    _triton_cache(cache, "matmul_kernel_persistent", PTX_A.replace("triton_poi_fused_1", "matmul_kernel_persistent"), "KEY3")
    idx = kernel_identity.export_ptx(cache, tmp_path / "ptx", gpu={"name": "RTX 4090", "multi_processor_count": 128})
    assert idx["variants"] == 3 and idx["kernels"] == 2
    shas = {e["ptx_code_sha256"] for e in idx["entries"] if e["name"] == "triton_poi_fused_1"}
    assert len(shas) == 1, "debug path strings are stripped: both cache entries have the same normalised code"
    text = (tmp_path / "ptx" / f"{next(iter(shas))}.ptx").read_text()
    assert ".file" not in text and ".loc" not in text and ".debug_abbrev" not in text and ".target sm_89" in text
    assert idx["entries"][0]["target"] == "sm_89" and idx["entries"][0]["cubin_sha256"] and idx["entries"][0]["meta"]["num_warps"] == 8
    assert idx["ptx_code_set_sha256"] and len(idx["ptx_code_set_sha256"]) == 64
    shipped = kernel_identity.ship_ptx(tmp_path / "ptx", tmp_path / "bundle" / "kernels")
    assert shipped["shipped"] == 3 and (tmp_path / "bundle" / "kernels" / "ptx-index.json").exists()
    loaded = VB.load_shipped_ptx(tmp_path / "bundle" / "kernels")
    assert loaded["available"] and set(loaded["by_name"]) == {"triton_poi_fused_1", "matmul_kernel_persistent"} and not loaded["bad_hash"]
    assert kernel_identity.ship_ptx(tmp_path / "nowhere", tmp_path / "b2" / "kernels")["shipped"] == 0


def test_f10_compare_variants_identical_subset_mismatch_and_num_sms():
    S = {"triton_poi_fused_1": {"s1": {"target": "sm_89"}, "s2": {"target": "sm_89"}}, "matmul_kernel_persistent": {"g1": {"target": "sm_89"}},
         "triton_per_fused_rms_2": {"r1": {"target": "sm_89"}}}
    C_same = {"triton_poi_fused_1": {"s1": {}, "s2": {}}, "matmul_kernel_persistent": {"g1": {}}, "triton_per_fused_rms_2": {"r1": {}}}
    cmp = VB.compare_ptx_variants(S, C_same, specialised_names={"matmul_kernel_persistent"}, gpu_differs=False)
    assert cmp["ok"] and cmp["summary"] == {"identical": 3}
    C_sub = {"triton_poi_fused_1": {"s1": {}}, "matmul_kernel_persistent": {"g1": {}}, "triton_per_fused_rms_2": {"r1": {}}}
    assert VB.compare_ptx_variants(S, C_sub, specialised_names=set(), gpu_differs=False)["kernels"]["triton_poi_fused_1"]["status"] == "subset"
    # the GEMM differs on another GPU: specialised, not a mismatch; the pointwise kernel differing is a mismatch
    C_other_gpu = {"triton_poi_fused_1": {"s1": {}, "s2": {}}, "matmul_kernel_persistent": {"g9": {"target": "sm_89"}}, "triton_per_fused_rms_2": {"r1": {}}}
    cmp = VB.compare_ptx_variants(S, C_other_gpu, specialised_names={"matmul_kernel_persistent"}, gpu_differs=True)
    assert cmp["kernels"]["matmul_kernel_persistent"]["status"] == "specialised: NUM_SMS" and not cmp["ok"]
    cmp = VB.compare_ptx_variants(S, C_other_gpu, specialised_names={"matmul_kernel_persistent"}, gpu_differs=False)
    assert cmp["kernels"]["matmul_kernel_persistent"]["status"] == "mismatch"
    C_bad = {**C_same, "triton_per_fused_rms_2": {"r9": {"target": "sm_89"}}}
    assert VB.compare_ptx_variants(S, C_bad, specialised_names={"matmul_kernel_persistent"}, gpu_differs=True)["kernels"]["triton_per_fused_rms_2"]["status"] == "mismatch"
    C_arch = {**C_same, "triton_per_fused_rms_2": {"r9": {"target": "sm_90"}}}
    assert VB.compare_ptx_variants(S, C_arch, specialised_names=set(), gpu_differs=True)["kernels"]["triton_per_fused_rms_2"]["status"] == "target"
    assert VB.specialised_kernel_names({sha(GEMM): GEMM, sha(KERNEL_A): KERNEL_A}) == {"matmul_kernel_persistent"}


def test_f10_ptx_identity_end_to_end_with_fake_compiler(tmp_path):
    """The verifier's step: compile every bundled source (fake compiler writes PTX into the Triton cache
    dir like Triton does), scan, compare with the shipped production PTX, qualify."""
    prod_cache = tmp_path / "prod-triton"
    _triton_cache(prod_cache, "triton_poi_fused_1", PTX_A, "K1")
    _triton_cache(prod_cache, "matmul_kernel_persistent", PTX_A.replace("triton_poi_fused_1", "matmul_kernel_persistent").replace("ret;", "mov.u32 %r1, 128; ret;"), "K2")
    _triton_cache(prod_cache, "other_vllm_kernel", PTX_A.replace("triton_poi_fused_1", "other_vllm_kernel"), "K3")  # not in the bundle
    kernel_identity.export_ptx(prod_cache, tmp_path / "production" / "ptx")
    run = tmp_path / "commit"
    write_kernels(run, KERNEL_A, GEMM)
    kernel_identity.ship_ptx(tmp_path / "production" / "ptx", run / "kernels")
    sources = {sha(KERNEL_A): KERNEL_A, sha(GEMM): GEMM}
    verifier_cache = tmp_path / "verifier-triton"

    def compile_here(sha_, fn, *, sms):
        ptx = PTX_A.replace("triton_poi_fused_1", fn).replace("/workspace/cache/prod", "/verifier/out")
        if fn == "matmul_kernel_persistent":
            ptx = ptx.replace("ret;", f"mov.u32 %r1, {sms}; ret;")
        _triton_cache(verifier_cache, fn, ptx, f"V-{fn}")

    env_same = {"gpu": {"name": "NVIDIA GeForce RTX 4090", "multi_processor_count": 128}}
    out = VB.ptx_identity(run, sources=sources, triton_cache_dir=str(verifier_cache), manifest=manifest_with("k"), env=env_same,
                          compile_fn=lambda s, f: compile_here(s, f, sms=128))
    assert out["available"] and out["summary"] == {"identical": 2} and out["qualifications"] == [] and out["shipped_not_in_bundle"] == ["other_vllm_kernel"]
    # another GPU: the GEMM's PTX differs -> specialised: NUM_SMS (a qualification), the pointwise kernel still identical
    import shutil

    shutil.rmtree(verifier_cache)
    env_h100 = {"gpu": {"name": "NVIDIA H100 80GB HBM3", "multi_processor_count": 132}}
    out = VB.ptx_identity(run, sources=sources, triton_cache_dir=str(verifier_cache), manifest=manifest_with("k"), env=env_h100,
                          compile_fn=lambda s, f: compile_here(s, f, sms=132))
    assert out["kernels"]["matmul_kernel_persistent"]["status"] == "specialised: NUM_SMS" and out["kernels"]["triton_poi_fused_1"]["status"] == "identical"
    assert out["qualifications"] == ["PTX matmul_kernel_persistent: specialised: NUM_SMS (batch-invariant GEMM: the SM count is a compile-time constant of the kernel and the verifier's GPU differs from production's)"]
    row = rows.table1_row(common(), **base_row(), qualifications=out["qualifications"])
    assert row["result"] == "PASS"
    # a bundle without shipped PTX (pre-F10 archive): not checkable, said so
    old = tmp_path / "old"
    write_kernels(old, KERNEL_A)
    out = VB.ptx_identity(old, sources={sha(KERNEL_A): KERNEL_A}, triton_cache_dir=str(verifier_cache), manifest={}, env={}, compile_fn=lambda s, f: None)
    assert not out["available"] and out["qualifications"][0].startswith("PTX identity not checkable")
    # tampered shipped PTX -> structural problem
    bad = run / "kernels" / f"{'0' * 64}.ptx"
    idx = json.loads((run / "kernels" / "ptx-index.json").read_text())
    idx["entries"][0]["ptx_code_sha256"] = "0" * 64
    bad.write_text("tampered")
    (run / "kernels" / "ptx-index.json").write_text(json.dumps(idx))
    out = VB.ptx_identity(run, sources=sources, triton_cache_dir=str(verifier_cache), manifest=manifest_with("k"), env=env_same, compile_fn=lambda s, f: None)
    assert out["problem"] and "do not hash" in out["problem"]


# --- F9: refusals are verdict rows -------------------------------------------------------------------


def test_f9_refusal_row_is_fail_structural(tmp_path, capsys):
    t1 = VB.refusal_row(tmp_path / "verify", case="B0", profile_id="84b332213ef1", worker="WA", adapter_revision="abc",
                        reason="bundle refused: bundle kernels/ files whose content does not hash to their name: ['0000….py']", exception="Traceback ...")
    assert t1["result"] == "FAIL" and t1["first_blocker"].startswith("structural: bundle refused: bundle kernels/ files")
    assert rows.validate_row(1, t1) == []
    written = [json.loads(l) for l in (tmp_path / "verify" / "table1.jsonl").read_text().splitlines()]
    assert written == [t1]
    doc = json.loads((tmp_path / "verify" / "verify.json").read_text())
    assert doc["refused"] and doc["exception"] == "Traceback ..."
    out = capsys.readouterr().out
    assert out.startswith("RESULT FAIL first_blocker=structural: bundle refused")


def test_f9_verify_run_wrapper_writes_a_row_for_a_bad_bundle(tmp_path):
    pytest.importorskip("torch")
    from verity_vllm_adapter import verify

    run = tmp_path / "commit"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"schema": "verity-vllm/profile/v1", "case": "B0", "engine_flags": {}}))
    kd = run / "kernels"
    kd.mkdir()
    (kd / f"{'0' * 64}.py").write_text(KERNEL_A)  # does not hash to its name (WA's M3)
    rc = verify.run([str(run), "--seed", "7", "--out", str(tmp_path / "v"), "--case", "B0"])
    assert rc == 1
    row = json.loads((tmp_path / "v" / "table1.jsonl").read_text())
    assert row["result"] == "FAIL" and "does not hash to their name" in row["first_blocker"] and row["case"] == "B0"


# --- rows: optional fields validate, old rows unchanged ------------------------------------------------


def test_optional_fields_validate_and_absent_fields_keep_old_rows_valid():
    row = rows.table1_row(common(), **base_row())
    assert "relation_errors" not in row and "qualifications" not in row and rows.validate_row(1, row) == []
    row = rows.table1_row(common(), **base_row(), relation_errors={"count": 0, "first": None}, negatives_not_run=[], qualifications=["x"],
                          bindings={"kernel_set": {"ok": True}})
    assert rows.validate_row(1, row) == [] and row["result"] == "PASS"


def test_f10_preexisting_cache_entries_are_not_this_process_evidence(tmp_path):
    prod_cache = tmp_path / "prod-triton"
    _triton_cache(prod_cache, "triton_poi_fused_1", PTX_A, "K1")
    kernel_identity.export_ptx(prod_cache, tmp_path / "production" / "ptx")
    run = tmp_path / "commit"
    write_kernels(run, KERNEL_A)
    kernel_identity.ship_ptx(tmp_path / "production" / "ptx", run / "kernels")
    shared = tmp_path / "shared-triton"
    _triton_cache(shared, "triton_poi_fused_1", PTX_A, "REPLAY-KEY")  # compiled by another process (a replay sharing the cache root)
    before = VB.ptx_paths(shared)
    assert before == {"REPLAY-KEY/triton_poi_fused_1.ptx"}
    out = VB.ptx_identity(run, sources={sha(KERNEL_A): KERNEL_A}, triton_cache_dir=str(shared), manifest=manifest_with("k"), env={},
                          compile_fn=lambda s, f: None, preexisting=before)  # cache hit: nothing new compiled
    assert out["kernels"]["triton_poi_fused_1"]["status"] == "identical"
    assert out["variants_compiled_here"] == 0 and out["variants_preexisting_in_cache"] == 1
    assert any(q.startswith("PTX: 1 of 1 compared variants were already in the Triton cache dir") for q in out["qualifications"])
    out = VB.ptx_identity(run, sources={sha(KERNEL_A): KERNEL_A}, triton_cache_dir=str(shared), manifest=manifest_with("k"), env={},
                          compile_fn=lambda s, f: None, preexisting=set())  # fresh cache dir: the same entry is this process's
    assert out["variants_compiled_here"] == 1 and out["qualifications"] == []


def test_f7_vllm_module_version_is_compared_not_the_dist_version():
    env = {"vllm": {"installed": True, "version": "0.28.1rc1.dev472+gd9105ea80.cu129", "module_version": "0.28.1rc1.dev472+gd9105ea80", "wheel_sha256": "a" * 64},
           "torch": {"installed": True, "version": "2.13.0+cu129"}, "triton": {"installed": True, "version": "3.7.1"}, "cuda": "12.9",
           "gpu": {"name": "NVIDIA GeForce RTX 4090", "multi_processor_count": 128}}
    m = manifest_with("k")
    m["vllm"]["version"] = "0.28.1rc1.dev472+gd9105ea80"
    assert VB.environment_qualifications(env, m) == []
