"""Gates G1..G7 over hand-written fixtures (shapes derived from the m6 Qwen reports) and the card's top section."""

from __future__ import annotations

import copy
import itertools
import json
import shutil
from pathlib import Path

import pytest

from verity_vllm.pipeline import report as CARD
from verity_vllm.check import gates as G

RID = "req-0"
NL = 28
MAX_TOKENS = 4
SNAPSHOT_STEPS = [0, 1, 2, 3]
CONTROL = [311, 13, 785, 1156]
DIGEST = "ab" * 32


def _ni_tokens(seq):
    return {RID: list(seq)}


def passing_artifacts() -> dict[str, dict]:
    """Every artifact of a run that passes all seven gates (relative path -> JSON)."""
    steps = [{"step": s, "layers_a": NL, "layers_b": NL, "sampled_eq": True} for s in range(MAX_TOKENS)]
    ni = {"PASS": True, "hash_comparisons": 4 * NL, "hash_equal": 4 * NL, "tokens_equal_accepted": True, "steps": steps, "layers": NL,
          "expected_steps": MAX_TOKENS, "expected_layers": NL, "problems": [], "mismatches": [],
          "tokens": {"a_observer": _ni_tokens(CONTROL), "b_hooks": _ni_tokens(CONTROL), "c_bare": _ni_tokens(CONTROL)}}
    cr = {"totals": {"census_kernels": 400, "distinct_kernel_names": 20, "residue_kernels": 0, "residue_names": [], "ambiguous": 0,
                     "count_mismatches": 0, "count_mismatch_kernels": [], "count_exempt": {}},
          "steps_aligned": True, "census_steps": MAX_TOKENS, "log_steps": MAX_TOKENS, "residue_zero": True,
          "kernels": [{"kernel": "matmul_kernel_persistent"}, {"kernel": "rms_norm_kernel"}], "steps": []}
    res = {"totals": {"steps": MAX_TOKENS, "unsupported": 0, "errors": 0, "computes": 1000, "compute_rows": 1200},
           "primary_report": {"by_pattern": {"Gemm": 500, "Attention": 100}, "unsupported_ops": {}},
           "by_spec": {"Gemm_v1{K=1536,N=1536}": 200, "Attention_v3{T=33,NH=12,KVH=2,D=128,BN=64}": 28},
           "roots": {"classes": {"weight": 300, "ignored": 100, "prompt_token": 32}, "unknown": []}, "header_mismatches": [], "unsupported_examples": []}
    hist_decode = {"Gemm_v1{K=1536,N=1536}": 28, "Attention_v3{T=33,NH=12,KVH=2,D=128,BN=64}": 28, "TokenSelect_v1{V=151936}": 1}
    per_step = [{"step": 0, "phase": "prefill", "n_instances": 5000, "spec_histogram": {"Gemm_v1{K=1536,N=1536}": 28}}]
    for s in range(1, MAX_TOKENS):
        h = dict(hist_decode)
        h[f"Attention_v3{{T={32 + s},NH=12,KVH=2,D=128,BN=64}}"] = h.pop("Attention_v3{T=33,NH=12,KVH=2,D=128,BN=64}")
        per_step.append({"step": s, "phase": "decode", "n_instances": 57, "spec_histogram": h})
    fs = {"schema": G.FOLD_SUMMARY_SCHEMA, "with_program": True, "digest": DIGEST, "n_instances": 5000 + 57 * 3, "n_steps": MAX_TOKENS,
          "unsupported": 0, "unresolved": 0, "opaque_fallbacks": 0, "errors": 0, "error_classes": {}, "unknown_roots": [], "unbound_roots": [],
          "per_step": per_step, "tokens": _ni_tokens(CONTROL), "requests": [{"request_id": RID, "prompt_len": 32, "max_tokens": MAX_TOKENS, "n_tokens": MAX_TOKENS}]}
    ra = {"status": "pass", "k": 3, "seed": 0, "strata": len(res["by_spec"]) + 1, "n_specs": len(res["by_spec"]), "n_specs_covered": len(res["by_spec"]),
          "coverage": {s: {"instances": 10, "evaluable": 10, "evaluated": 3, "pass": 3, "fail": 0, "skipped": 0, "granularity": "row",
                           "regimes": {"decode": {"instances": 10, "pass": 3, "fail": 0, "error": 0, "skipped": 0}}} for s in res["by_spec"]},
          "uncovered_specs": [], "instances": [], "seconds": 1.0, "max_rss_gb": 0.5,
          "tokens": {"steps": SNAPSHOT_STEPS, "evaluated": len(SNAPSHOT_STEPS), "equal": len(SNAPSHOT_STEPS), "skipped": 0, "mismatch": [],
                     "per_step": {str(s): {"rows": 1, "equal": 1, "skipped": 0} for s in SNAPSHOT_STEPS}, "all_rows_equal": True}}
    wiring = {"status": "pass", "reads": 28450, "writes": 20112, "instances": 5000 + 57 * 3, "n_mismatch": 0, "uncovered": 0, "mismatches": [],
              "streams": {"distinct_streams": [0], "single_stream": True}, "failures": [],
              "schema": {"roots_with_writes": 4880, "schema_roots": 2928, "triton_roots": 1952, "writes_outside_schema": 0, "declared_unwritten": 11940,
                         "triton_writes": 5703}, "seconds": 0.7}
    chain = {"status": "pass", "n": 3, "seed": 0, "n_slices": 3, "n_edges_chained": 17,
             "slices": [{"nodes": [10, 11, 12], "step": st, "specs": ["Gemm_v1{K=1536,N=1536}", "RMSNormFusedCuda_v2{N=1536,EPS=1e-06}"], "edges_chained": 5,
                         "status": "pass", "members": []} for st in (0, 1, 3)], "seconds": 15.0}
    arts: dict[str, dict] = {
        "run.json": {"config_id": "fixture", "profile": None, "case": "B1", "laptop": False, "snapshot_steps": SNAPSHOT_STEPS,
                     "C": {"model": "Qwen/Qwen2.5-1.5B", "NL": NL, "H": 1536, "NH": 12, "KVH": 2, "D": 128, "BN": 64},
                     "workload": {"path": "w.json", "sha256": "0" * 64, "n_requests": 1, "prompt_lens": [32], "max_tokens": [MAX_TOKENS],
                                  "requests": [{"request_id": RID, "prompt_len": 32, "max_tokens": MAX_TOKENS}]}},
        "control/tokens.json": {"requests": [{"request_id": RID, "prompt_token_ids": list(range(32)), "max_tokens": MAX_TOKENS, "generated_token_ids": CONTROL}]},
        "nonint/noninterference.json": ni, "census_reconcile.json": cr, "resolution.json": res, "fold_summary.json": fs,
        "fold_summary_2.json": copy.deepcopy(fs),
        "root_policy.json": {"status": "pass", "classes": {"weight": {"count": 300, "validated": 300, "failed": 0}, "rotary": {"count": 1, "validated": 1, "failed": 0}},
                             "roots": [{"name": "cos_sin_cache", "class": "rotary", "validator": "analytic.check_cos_sin", "status": "pass"}],
                             "new_root_classes": [], "alias_claims": []},
        "replay_a.json": ra, "wiring.json": wiring, "replay_chain.json": chain,
        "golden.json": {"status": "pass", "entries": [{"name": "m1", "log": "m1.jsonl.gz", "expected_digest": "cd" * 32, "got_digest": "cd" * 32,
                                                       "expected_attribution_sha": "ef" * 32, "got_attribution_sha": "ef" * 32, "status": "pass"}]},
        "holdouts.json": {"status": "pass", "holdouts": [{"name": "qwen15_2req", "workload": "w2.json", "run_dir": "runs/h1",
                                                          "gates": {g: "pass" for g in G.HOLDOUT_GATES}}]},
        "compare.json": {"tokens": {"status": "pass"}, "structure": {"status": "pass", "family": "Serve_v4", "matched": 100, "superset": True, "n_extra": 3,
                                                                      "classes": {"lm_head_tail": {"count": 3}}}},
    }
    return arts


def write_run(tmp: Path, arts: dict[str, dict]) -> Path:
    """Write exactly these artifacts (a fresh run directory each time)."""
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for rel, doc in arts.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc))
    return tmp


def gate(tmp: Path, arts: dict, gid: str) -> G.GateResult:
    write_run(tmp, arts)
    return getattr(G, f"gate_{gid}")(G.RunArtifacts(tmp))


def by_id(results):
    return {r.id: r for r in results}


# --- the whole set -----------------------------------------------------------------------------------------


def test_all_pass_accepted(tmp_path):
    write_run(tmp_path, passing_artifacts())
    res = G.evaluate_gates(tmp_path)
    assert [r.id for r in res] == list(G.ALL_GATES) == list(G.GATES) + ["G8", "R1"]
    assert {r.status for r in res if r.id in G.GATES} == {"pass"}, {r.id: r.reason for r in res}
    assert by_id(res)["G8"].status == "skipped"            # no query_use.json in the fixture: G8 skipped, verdict unchanged
    assert by_id(res)["R1"].status == "pass"               # [R9 ref] fold_summary errors/unresolved/unsupported all 0, resolution totals 0
    assert G.verdict(res) == "ACCEPTED"
    assert all(r.evidence for r in res)


def test_empty_run_dir_incomplete(tmp_path):
    res = G.evaluate_gates(tmp_path)
    assert {r.status for r in res if r.id != "R1"} == {"skipped"}
    assert by_id(res)["R1"].status == "unverified"         # [R9 ref] no fold_summary.json: unverified, never skipped / pass
    assert G.verdict(res) == "INCOMPLETE"


def test_verdict_rules():
    ok = [G.GateResult(g, G.GATE_NAMES[g], "pass") for g in G.GATES]
    assert G.verdict(ok) == "ACCEPTED"
    bad = copy.deepcopy(ok)
    bad[3].status = "fail"
    bad[6].status = "skipped"
    assert G.verdict(bad) == "REJECTED"
    inc = copy.deepcopy(ok)
    inc[6].status = "skipped"
    assert G.verdict(inc) == "INCOMPLETE"
    err = copy.deepcopy(ok)
    err[1].status = "error"
    assert G.verdict(err) == "INCOMPLETE"
    assert G.verdict(ok[:6]) == "INCOMPLETE"       # G7 missing entirely


def test_verdict_is_the_match_verdict_over_check_results():
    """The word is `verdict.match_verdict` over the gates' CheckResults, equal to the status rule it replaced for every status a gate
    produces (a frozen gate: pass / fail / skipped / error; G8 adds unsupported, R1 unverified), with and without a frozen gate absent."""
    def by_status(results):
        core = [r for r in results if r.id in G.GATES]
        extra = [r for r in results if r.id not in G.GATES]
        st = {r.status for r in core}
        if "fail" in st or any(r.status == "fail" for r in extra):
            return "REJECTED"
        if ("skipped" in st or "error" in st or {r.id for r in core} != set(G.GATES)
                or any(r.status in ("error", "unsupported", "unverified") for r in extra)):
            return "INCOMPLETE"
        return "ACCEPTED"

    frozen = ("pass", "fail", "skipped", "error")
    n = 0
    for a, b in itertools.product(frozen, repeat=2):
        for g8 in frozen + ("unsupported",):
            for r1 in frozen + ("unverified",):
                res = [G.GateResult(g, G.GATE_NAMES[g], s) for g, s in zip(G.ALL_GATES, (a, b, "pass", "pass", "pass", "pass", "pass", g8, r1))]
                for rs in (res, res[:6] + res[7:]):
                    assert G.verdict(rs) == by_status(rs), [(r.id, r.status) for r in rs]
                    n += 1
    assert n == 800
    r = G.GateResult("G3", G.GATE_NAMES["G3"], "fail", reason="x").result()
    assert (r.name, r.outcome, r.reason, r.claim) == ("G3", "FAIL", "x", G.GATE_NAMES["G3"])


def test_frozen_constants():
    assert G.TIER_A_K == 3 and G.MIN_DECODE_TOKEN_STEPS == 2 and G.CHAIN_MIN_SLICES == 1 and G.REGIMES == ("prefill", "decode") and G.COS_SIN_MAX_ULP == 1
    assert G.GATES == ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
    assert "approved decision" in G.__doc__


# --- G1 --------------------------------------------------------------------------------------------------


def test_G1_pass_and_tokens_vs_control(tmp_path):
    arts = passing_artifacts()
    r = gate(tmp_path, arts, "G1")
    assert r.status == "pass"
    assert r.detail["noninterference"]["control_source"] == "control/tokens.json"
    assert r.detail["noninterference"]["tokens_equal_control"] == {"a_observer": True, "b_hooks": True, "c_bare": True}


def test_G1_control_differs_fails_even_when_file_says_pass(tmp_path):
    arts = passing_artifacts()
    arts["control/tokens.json"]["requests"][0]["generated_token_ids"] = [311, 13, 785, 9999]
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and "control" in r.reason


def test_G1_c_bare_is_the_control_when_no_control_run(tmp_path):
    arts = passing_artifacts()
    del arts["control/tokens.json"]
    r = gate(tmp_path, arts, "G1")
    assert r.status == "pass" and r.detail["noninterference"]["control_source"] == "noninterference c_bare"


def test_G1_expected_layers_null_fails(tmp_path):
    arts = passing_artifacts()
    arts["nonint/noninterference.json"]["expected_layers"] = None
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and "expected_layers is null" in r.reason


def test_G1_layers_mismatch_fails(tmp_path):
    arts = passing_artifacts()
    arts["nonint/noninterference.json"]["expected_layers"] = NL + 1
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and "expected_layers" in r.reason and "NL" in r.reason


def test_G1_a_profile_that_cannot_be_loaded_fails_by_name_rather_than_skipping_the_NL_check(tmp_path):
    arts = passing_artifacts()
    del arts["run.json"]["C"]
    arts["run.json"]["profile"] = "no_such_profile_zz"
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and f"expected_layers={NL} not checked against the profile's NL" in r.reason
    assert "profile 'no_such_profile_zz' could not be loaded: ModuleNotFoundError" in r.reason
    with pytest.raises(G.ArtifactError, match="no_such_profile_zz"):
        _ = G.RunArtifacts(tmp_path).C


def test_G1_steps_hashes_census(tmp_path):
    arts = passing_artifacts()
    arts["nonint/noninterference.json"]["steps"] = arts["nonint/noninterference.json"]["steps"][:-1]
    assert "steps, expected" in gate(tmp_path, arts, "G1").reason
    arts = passing_artifacts()
    arts["nonint/noninterference.json"]["hash_equal"] -= 1
    assert "boundary hashes" in gate(tmp_path, arts, "G1").reason
    arts = passing_artifacts()
    arts["census_reconcile.json"]["totals"]["residue_kernels"] = 2
    arts["census_reconcile.json"]["totals"]["residue_names"] = ["foo_kernel"]
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and "residue 2" in r.reason and "foo_kernel" in r.reason
    arts = passing_artifacts()
    arts["census_reconcile.json"]["totals"]["ambiguous"] = 1
    assert "ambiguous" in gate(tmp_path, arts, "G1").reason
    arts = passing_artifacts()
    arts["nonint/noninterference.json"]["PASS"] = False
    assert "PASS is false" in gate(tmp_path, arts, "G1").reason


def test_G1_census_launch_counts(tmp_path):
    arts = passing_artifacts()
    arts["census_reconcile.json"]["totals"].update(count_mismatches=1, count_mismatch_kernels=["fp8_mqa_logits"])
    r = gate(tmp_path, arts, "G1")
    assert r.status == "fail" and "1 launch counts differ" in r.reason and "fp8_mqa_logits" in r.reason
    arts = passing_artifacts()
    del arts["census_reconcile.json"]["totals"]["count_mismatches"]
    r = gate(tmp_path, arts, "G1")
    assert r.status == "error" and "count_mismatches" in r.reason and "census --reconcile" in r.reason


def test_G1_missing_census_skips_with_stage_evidence(tmp_path):
    arts = passing_artifacts()
    del arts["census_reconcile.json"]
    arts["stages.json"] = {"stages": [{"stage": "census_reconcile", "status": "failed", "returncode": 1, "log": "logs/census_reconcile.log"}]}
    r = gate(tmp_path, arts, "G1")
    assert r.status == "skipped" and "census_reconcile failed" in r.reason and "logs/census_reconcile.log" in r.evidence


def test_G1_malformed_is_error(tmp_path):
    arts = passing_artifacts()
    del arts["census_reconcile.json"]["totals"]
    r = gate(tmp_path, arts, "G1")
    assert r.status == "error" and "totals" in r.reason
    write_run(tmp_path, arts)
    (tmp_path / "nonint/noninterference.json").write_text("{not json")
    assert G.gate_G1(G.RunArtifacts(tmp_path)).status == "error"


# --- G2 --------------------------------------------------------------------------------------------------


def test_G2_pass_uses_census_ambiguity_when_resolver_has_none(tmp_path):
    r = gate(tmp_path, passing_artifacts(), "G2")
    assert r.status == "pass" and r.detail["ambiguity_source"].startswith("census_reconcile")
    assert "census_reconcile.json" in r.evidence


def test_G2_resolver_ambiguity_field_wins(tmp_path):
    arts = passing_artifacts()
    arts["resolution.json"]["totals"]["ambiguous"] = 2
    arts["census_reconcile.json"]["totals"]["ambiguous"] = 0
    r = gate(tmp_path, arts, "G2")
    assert r.status == "fail" and "2 ambiguous" in r.reason and r.detail["ambiguity_source"] == "resolution.totals.ambiguous"


def test_G2_unsupported_unresolved_fail(tmp_path):
    arts = passing_artifacts()
    arts["resolution.json"]["totals"]["unsupported"] = 3
    arts["resolution.json"]["primary_report"]["unsupported_ops"] = {"aten::foo": 3}
    r = gate(tmp_path, arts, "G2")
    assert r.status == "fail" and "3 Unsupported" in r.reason and "aten::foo" in r.reason
    arts = passing_artifacts()
    arts["fold_summary.json"]["unresolved"] = 1
    arts["fold_summary.json"]["errors"] = 1
    arts["fold_summary.json"]["error_classes"] = {"Unresolved": 1}
    r = gate(tmp_path, arts, "G2")
    assert r.status == "fail" and "1 Unresolved" in r.reason


def test_G2_no_ambiguity_source_skips(tmp_path):
    arts = passing_artifacts()
    del arts["census_reconcile.json"]
    r = gate(tmp_path, arts, "G2")
    assert r.status == "skipped" and "ambiguity" in r.reason


# --- G3 --------------------------------------------------------------------------------------------------


def test_G3(tmp_path):
    assert gate(tmp_path, passing_artifacts(), "G3").status == "pass"
    arts = passing_artifacts()
    arts["fold_summary.json"]["unknown_roots"] = ["model.layers.0.mystery"]
    r = gate(tmp_path, arts, "G3")
    assert r.status == "fail" and "unknown roots" in r.reason
    arts = passing_artifacts()
    arts["root_policy.json"]["new_root_classes"] = ["mystery"]
    r = gate(tmp_path, arts, "G3")
    assert r.status == "fail" and "new root classes" in r.reason
    arts = passing_artifacts()
    arts["root_policy.json"]["status"] = "fail"
    arts["root_policy.json"]["classes"]["rotary"]["failed"] = 1
    arts["root_policy.json"]["roots"][0]["status"] = "fail"
    r = gate(tmp_path, arts, "G3")
    assert r.status == "fail" and "root_policy status 'fail'" in r.reason and "1 root validations failed" in r.reason
    arts = passing_artifacts()
    arts["fold_summary.json"]["opaque_fallbacks"] = 2
    assert "opaque" in gate(tmp_path, arts, "G3").reason
    arts = passing_artifacts()
    del arts["root_policy.json"]
    r = gate(tmp_path, arts, "G3")
    assert r.status == "skipped" and "root_policy.json missing" in r.reason


# --- G4 --------------------------------------------------------------------------------------------------


def test_G4(tmp_path):
    r = gate(tmp_path, passing_artifacts(), "G4")
    assert r.status == "pass" and r.detail["quarantine"] == "absent" and r.detail["covered"] == 2
    arts = passing_artifacts()
    arts["replay_a.json"]["k"] = 2
    assert "k=2 < TIER_A_K=3" in gate(tmp_path, arts, "G4").reason
    arts = passing_artifacts()
    arts["replay_a.json"]["uncovered_specs"] = ["Gemm_v1{K=1536,N=1536}"]
    assert "uncovered" in gate(tmp_path, arts, "G4").reason
    arts = passing_artifacts()
    arts["replay_a.json"]["status"] = "fail"
    arts["replay_a.json"]["coverage"]["Gemm_v1{K=1536,N=1536}"]["fail"] = 1
    r = gate(tmp_path, arts, "G4")
    assert r.status == "fail" and "status 'fail'" in r.reason and "instances failed" in r.reason
    arts = passing_artifacts()
    del arts["replay_a.json"]
    assert gate(tmp_path, arts, "G4").status == "skipped"


def test_G4_quarantine(tmp_path):
    arts = passing_artifacts()
    arts["quarantine.json"] = {"status": "pass", "differential": {"passed": 12, "total": 12}, "negative_matcher_tests": "pass"}
    r = gate(tmp_path, arts, "G4")
    assert r.status == "pass" and r.detail["quarantine"]["differential"]["total"] == 12
    arts["quarantine.json"]["differential"]["passed"] = 11
    assert "11/12" in gate(tmp_path, arts, "G4").reason
    arts["quarantine.json"]["differential"] = {"passed": 0, "total": 0}
    assert "need total > 0" in gate(tmp_path, arts, "G4").reason
    arts = passing_artifacts()
    arts["quarantine.json"] = {"differential": {"passed": 3, "total": 3}, "negative_matcher_tests": "fail"}
    assert "negative_matcher_tests" in gate(tmp_path, arts, "G4").reason
    # quarantine specs used, quarantine.json missing -> FAIL
    arts = passing_artifacts()
    arts["resolution.json"]["quarantine_specs"] = ["NewOp_v1{N=8}"]
    r = gate(tmp_path, arts, "G4")
    assert r.status == "fail" and "quarantine.json missing" in r.reason
    arts = passing_artifacts()
    arts["resolution.json"]["spec_modules"] = {"NewOp_v1{N=8}": "verity_vllm.program.quarantine"}
    assert G.quarantine_specs(arts["resolution.json"]) == ["NewOp_v1{N=8}"]


# --- G5 --------------------------------------------------------------------------------------------------


def test_G5_pass_provenance_sample_tokens_chain(tmp_path):
    r = gate(tmp_path, passing_artifacts(), "G5")
    assert r.status == "pass", r.reason
    assert r.detail["provenance"]["n_mismatch"] == 0 and r.detail["provenance"]["single_stream"] is True
    assert r.detail["tokens"]["steps"] == SNAPSHOT_STEPS and r.detail["chain"]["n_slices"] == 3
    assert r.detail["full_replay"] == "not run (diagnostic only)"


def test_G5_provenance_mismatch_fails(tmp_path):
    arts = passing_artifacts()
    arts["wiring.json"]["n_mismatch"] = 1
    arts["wiring.json"]["status"] = "fail"
    arts["wiring.json"]["mismatches"] = [{"seq": 5, "event": 97, "what": "row 3"}]
    r = gate(tmp_path, arts, "G5")
    assert r.status == "fail" and "not reproduced by the independent provenance map" in r.reason
    arts = passing_artifacts()
    arts["wiring.json"]["streams"] = {"distinct_streams": [0, 7], "single_stream": False}
    assert "several streams" in gate(tmp_path, arts, "G5").reason
    arts = passing_artifacts()
    arts["wiring.json"]["schema"]["writes_outside_schema"] = 2
    assert "outside the schema-declared write set" in gate(tmp_path, arts, "G5").reason
    arts = passing_artifacts()
    arts["wiring.json"]["uncovered"] = 3
    assert "no journaled producer" in gate(tmp_path, arts, "G5").reason


def test_G5_served_tokens(tmp_path):
    arts = passing_artifacts()
    arts["replay_a.json"]["tokens"]["mismatch"] = [{"node": 9, "step": 2, "computed": 42, "served": 7}]
    arts["replay_a.json"]["tokens"]["all_rows_equal"] = False
    r = gate(tmp_path, arts, "G5")
    assert r.status == "fail" and "served token(s) not reproduced" in r.reason
    arts = passing_artifacts()
    arts["replay_a.json"]["tokens"]["steps"] = [0, 1]
    assert "MIN_DECODE_TOKEN_STEPS" in gate(tmp_path, arts, "G5").reason
    arts = passing_artifacts()
    arts["replay_a.json"]["tokens"]["steps"] = [1, 2, 3]
    assert "step 0 (prefill) not checked" in gate(tmp_path, arts, "G5").reason
    arts = passing_artifacts()
    del arts["replay_a.json"]["tokens"]
    assert "no `tokens` section" in gate(tmp_path, arts, "G5").reason


def test_G5_chain(tmp_path):
    arts = passing_artifacts()
    arts["replay_chain.json"]["status"] = "fail"
    arts["replay_chain.json"]["status_reason"] = "1 of 3 slices failed"
    r = gate(tmp_path, arts, "G5")
    assert r.status == "fail" and "chain status 'fail'" in r.reason
    arts = passing_artifacts()
    arts["replay_chain.json"]["slices"][0]["edges_chained"] = 0
    assert "chained no producer->consumer edge" in gate(tmp_path, arts, "G5").reason


def test_G5_missing_artifacts_skip(tmp_path):
    arts = passing_artifacts()
    del arts["wiring.json"]
    arts["stages.json"] = {"stages": [{"stage": "wiring", "status": "failed", "log": "logs/wiring.log", "reason": "rc=1"}]}
    r = gate(tmp_path, arts, "G5")
    assert r.status == "skipped" and "wiring" in r.reason
    arts = passing_artifacts()
    del arts["replay_chain.json"]
    assert gate(tmp_path, arts, "G5").status == "skipped"


def test_G5_full_replay_is_diagnostic_only(tmp_path):
    arts = passing_artifacts()
    arts[G.replay_b_rel(1)] = {"status": "fail", "step": 1, "n_instances": 57, "n_diverged": 3, "partial": True, "tokens": {"equal": False}}
    r = gate(tmp_path, arts, "G5")
    assert r.status == "pass" and r.detail["full_replay"]["1"]["n_diverged"] == 3


# --- G6 --------------------------------------------------------------------------------------------------


def test_G6(tmp_path):
    r = gate(tmp_path, passing_artifacts(), "G6")
    assert r.status == "pass" and r.detail["digests_equal"] and r.detail["golden"]["entries"] == 1
    arts = passing_artifacts()
    arts["fold_summary_2.json"]["digest"] = "cd" * 32
    assert "digests differ" in gate(tmp_path, arts, "G6").reason
    arts = passing_artifacts()
    arts["fold_summary.json"]["digest"] = ""
    arts["fold_summary.json"]["with_program"] = False
    assert "without Program()" in gate(tmp_path, arts, "G6").reason
    arts = passing_artifacts()
    arts["golden.json"]["entries"][0]["got_digest"] = "00" * 32
    arts["golden.json"]["entries"][0]["status"] = "fail"
    arts["golden.json"]["status"] = "fail"
    r = gate(tmp_path, arts, "G6")
    assert r.status == "fail" and "golden status 'fail'" in r.reason and "m1" in r.reason
    arts = passing_artifacts()
    arts["fold_summary_other.json"] = {"digest": "ff" * 32, "log": "other.jsonl.gz"}
    r = gate(tmp_path, arts, "G6")
    assert r.status == "pass" and r.detail["other_capture"]["equal"] is False   # not required
    arts = passing_artifacts()
    del arts["golden.json"]
    assert gate(tmp_path, arts, "G6").status == "skipped"
    arts = passing_artifacts()
    del arts["fold_summary_2.json"]
    assert gate(tmp_path, arts, "G6").status == "skipped"


# --- G7 --------------------------------------------------------------------------------------------------


def test_G7(tmp_path):
    r = gate(tmp_path, passing_artifacts(), "G7")
    assert r.status == "pass" and r.detail["holdouts"]["qwen15_2req"]["pass"]
    arts = passing_artifacts()
    arts["holdouts.json"]["holdouts"][0]["gates"]["G4"] = "fail"
    r = gate(tmp_path, arts, "G7")
    assert r.status == "fail" and "['G4']" in r.reason
    arts["holdouts.json"]["holdouts"][0]["gates"]["G4"] = {"status": "skipped"}
    assert gate(tmp_path, arts, "G7").status == "fail"
    arts = passing_artifacts()
    arts["holdouts.json"]["holdouts"] = []
    assert gate(tmp_path, arts, "G7").status == "fail"
    arts = passing_artifacts()
    del arts["holdouts.json"]
    r = gate(tmp_path, arts, "G7")
    assert r.status == "skipped" and "orchestrator" in r.reason
    write_run(tmp_path, arts)
    assert G.verdict(G.evaluate_gates(tmp_path)) == "INCOMPLETE"


# --- diagnostics (never gating) ------------------------------------------------------------------------------


def test_diagnostics_structure_and_family(tmp_path):
    arts = passing_artifacts()
    write_run(tmp_path, arts)
    d = G.diagnostics(tmp_path)
    assert d["structure"]["histograms_equal_mod_T"] and d["structure"]["count_constant"] and d["structure"]["linear"]
    assert d["family_match"]["unclassified"] == 0
    arts["fold_summary.json"]["per_step"][2]["spec_histogram"]["Extra_v1{N=1}"] = 1
    arts["fold_summary.json"]["per_step"][2]["n_instances"] += 1
    arts["compare.json"]["structure"]["classes"]["unclassified:Foo"] = {"count": 1}
    write_run(tmp_path, arts)
    d = G.diagnostics(tmp_path)
    assert not d["structure"]["histograms_equal_mod_T"] and not d["structure"]["count_constant"]
    assert d["family_match"]["unclassified"] == 1 and d["family_match"]["unclassified_classes"] == ["unclassified:Foo"]
    # still not a gate
    assert {r.status for r in G.evaluate_gates(tmp_path) if r.id in G.GATES} == {"pass"}


def test_diag_overhead_normalizes_by_clock():
    m0 = {"baseline_step_wall_s": 0.010, "reps": 1,
          "results": {"r3": {"label": "full", "reps": [{"step_wall_mean_s": 0.020, "recorded_calls": 1000, "triton_launches": 0, "observed_steps": 1}]}}}
    d = G.diag_overhead(m0, {"host": {"clocksource": "tsc", "clock_read_us": 0.5}})
    r = d["rungs"]["r3"]
    assert r["x_baseline"] == 2.0 and r["overhead_per_event_us"] == 10.0 and r["overhead_per_event_us_minus_clock"] == 8.0
    assert G.diag_overhead(None, None)["status"] == "unavailable"


def test_write_gates_json(tmp_path):
    write_run(tmp_path, passing_artifacts())
    res = G.evaluate_gates(tmp_path)
    p = G.write_gates(tmp_path, res, G.diagnostics(tmp_path))
    d = json.loads(p.read_text())
    assert d["schema"] == G.GATES_SCHEMA and d["verdict"] == "ACCEPTED" and list(d["gates"]) == list(G.ALL_GATES)
    assert d["order"] == list(G.ALL_GATES) and d["frozen"] == list(G.GATES)
    assert d["constants"]["TIER_A_K"] == 3 and d["diagnostics"]["structure"]["status"] == "ok"
    assert "| G5 |" in G.gates_markdown(res) and "| G8 |" in G.gates_markdown(res)


# --- G8 (structural, outside the frozen set) ------------------------------------------------------------------

QU_COUNT, QU_WMAX, QU_MAX_SEEN = 1436, 32, 32


def query_use_doc(partition="exact", codes=(), width="ok", violators=(), unsupported=(), max_seen=QU_MAX_SEEN, count=QU_COUNT, status=None) -> dict:
    """A synthetic `verity-ir/query-use/v1` document in the shape of the query-use interface (its frozen fields)."""
    if status is None:
        status = "ok" if partition == "exact" and width == "ok" and not violators and not unsupported else "fail"
    return {"schema": G.QUERY_USE_SCHEMA, "binding_id": "b" * 64, "use": {"kind": G.QUERY_USE_KIND, "w_max": QU_WMAX},
            "verdict": {"status": status, "count": count,
                        "partition": {"status": partition, "codes": list(codes), "detail": {"path": "tiling"}},
                        "width": {"w_max": QU_WMAX, "max_seen": max_seen, "violators": list(violators), "unsupported": list(unsupported),
                                  "status": width, "detail": {}},
                        "readiness_32": {"w_max": QU_WMAX, "ok": True, "violators": [], "max_w": 32, "specs": 3, "gates_over": 0}},
            "evaluator": {"version": "verity-ir/partition/0.1", "utc": "2026-09-17T22:00:00Z"}}


def test_G8_absent_is_skipped_and_keeps_the_verdict(tmp_path):
    arts = passing_artifacts()
    r = gate(tmp_path, arts, "G8")
    assert r.status == "skipped" and r.reason == "no query-use document" and r.evidence == ["query_use.json"]
    assert G.verdict(G.evaluate_gates(tmp_path)) == "ACCEPTED"
    # a failing G1..G7 run without the document keeps its REJECTED / INCOMPLETE verdict too
    arts["resolution.json"]["totals"]["unsupported"] = 1
    write_run(tmp_path, arts)
    assert G.verdict(G.evaluate_gates(tmp_path)) == "REJECTED"
    del arts["holdouts.json"]
    arts["resolution.json"]["totals"]["unsupported"] = 0
    write_run(tmp_path, arts)
    assert G.verdict(G.evaluate_gates(tmp_path)) == "INCOMPLETE"


def test_G8_pass(tmp_path):
    arts = passing_artifacts()
    arts["query_use.json"] = query_use_doc()
    r = gate(tmp_path, arts, "G8")
    assert r.status == "pass", r.reason
    assert r.id == "G8" and r.name == "VU partition" and r.evidence == ["query_use.json"]
    assert r.detail["partition"] == {"status": "exact", "codes": []}
    assert r.detail["width"]["status"] == "ok" and r.detail["width"]["max_seen"] == QU_MAX_SEEN and r.detail["width"]["w_max"] == QU_WMAX
    assert r.detail["count"] == QU_COUNT and r.detail["binding_id"] == "b" * 64 and r.detail["use"] == {"kind": "vu-partition", "w_max": 32}
    assert r.detail["verdict_status"] == "ok" and r.detail["readiness_32"] == {"ok": True, "max_w": 32, "n_violators": 0}
    assert r.detail["evaluator"]["version"] == "verity-ir/partition/0.1" and r.detail["binding_checked"] is False
    res = G.evaluate_gates(tmp_path)
    assert G.verdict(res) == "ACCEPTED" and {r.status for r in res} == {"pass"}


def test_G8_evaluator_status_not_ok_is_surfaced(tmp_path):
    """qpart: `verdict.status` is ok | fail | not-ready | unsupported; anything but ok is not a pass, and the string is in the reason
    and verbatim in `detail.status`.  `not-ready` / `fail` -> fail (REJECTED); `unsupported` -> G8 `unsupported` (lead 22:22Z:
    neither pass nor fail, INCOMPLETE, never REJECTED)."""
    arts = passing_artifacts()
    for st in ("not-ready", "fail"):
        arts["query_use.json"] = query_use_doc(status=st)                       # partition/width fields still look fine
        r = gate(tmp_path, arts, "G8")
        assert r.status == "fail" and f"verdict.status '{st}'" in r.reason and r.detail["verdict_status"] == r.detail["status"] == st
        assert G.verdict(G.evaluate_gates(tmp_path)) == "REJECTED"
    arts["query_use.json"] = query_use_doc(status="unsupported")
    r = gate(tmp_path, arts, "G8")
    assert r.status == "unsupported" and r.reason == "unsupported (verdict.status 'unsupported')" and r.detail["status"] == "unsupported"
    assert G.verdict(G.evaluate_gates(tmp_path)) == "INCOMPLETE"
    arts["query_use.json"] = query_use_doc(partition="unsupported", codes=["sweep-limit"], status="unsupported")
    r = gate(tmp_path, arts, "G8")
    assert r.status == "unsupported" and "partition 'unsupported' ['sweep-limit']" in r.reason and "verdict.status 'unsupported'" in r.reason
    assert r.detail["undecided"] == ["partition 'unsupported' ['sweep-limit']", "verdict.status 'unsupported'"]
    res = G.evaluate_gates(tmp_path)
    assert G.verdict(res) == "INCOMPLETE" and {x.status for x in res if x.id in G.GATES} == {"pass"}
    # a fail-class finding dominates: undecided width + a violation -> fail, REJECTED
    arts["query_use.json"] = query_use_doc(width="violation", violators=[4], unsupported=[9], status="fail")
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and "w_out > w_max" in r.reason and r.detail["undecided"] == ["1 member class(es) with undecidable w_out: [9]"]
    assert G.verdict(G.evaluate_gates(tmp_path)) == "REJECTED"


def test_G8_binding_against_the_query_document(tmp_path):
    """qpart 22:14Z: the companion query document is `RUN_DIR/query.json`; its binding_id must equal the use document's and its
    program_digest the run's fold digest.  Absent query.json: the binding is recorded as unchecked, the verdict unaffected."""
    arts = passing_artifacts()
    arts["query_use.json"] = query_use_doc()
    r = gate(tmp_path, arts, "G8")
    assert r.status == "pass" and r.detail["binding"] == {"checked": False} and r.detail["binding_checked"] is False
    digest = arts["fold_summary.json"]["digest"]
    qdoc = {"schema": "verity-ir/query/v1", "query_id": "q" * 64, "program_digest": digest, "binding_id": "b" * 64, "query": {"op": "default", "w": 32}}
    arts["query.json"] = qdoc
    r = gate(tmp_path, arts, "G8")
    assert r.status == "pass" and r.evidence == ["query_use.json", "query.json"]
    assert r.detail["binding"] == {"checked": True, "query_binding_id": "b" * 64, "query_program_digest": digest, "run_digest": digest, "query_id": "q" * 64}
    assert G.verdict(G.evaluate_gates(tmp_path)) == "ACCEPTED"
    arts["query.json"] = dict(qdoc, binding_id="c" * 64)
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and r.reason == "binding: query.json binding_id != query_use.json binding_id"
    arts["query.json"] = dict(qdoc, program_digest="0" * 64)
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and "program_digest != the run's fold digest" in r.reason
    assert G.verdict(G.evaluate_gates(tmp_path)) == "REJECTED"


def test_G8_partition_fail_carries_the_codes(tmp_path):
    arts = passing_artifacts()
    arts["query_use.json"] = query_use_doc(partition="overlap", codes=["OVERLAP:root/12", "OVERLAP:root/13"])
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and "partition 'overlap'" in r.reason and "OVERLAP:root/12" in r.reason
    assert r.detail["partition"]["codes"] == ["OVERLAP:root/12", "OVERLAP:root/13"]
    res = G.evaluate_gates(tmp_path)
    assert G.verdict(res) == "REJECTED"
    assert {x.status for x in res if x.id in G.GATES} == {"pass"}            # G1..G7 untouched by G8
    arts["query_use.json"] = query_use_doc(partition="gap", codes=["GAP:x"])
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and "partition 'gap'" in r.reason
    arts["query_use.json"] = query_use_doc(partition="unsupported", codes=["limit-intervals"], status="unsupported")
    r = gate(tmp_path, arts, "G8")
    assert r.status == "unsupported" and "partition 'unsupported' ['limit-intervals']" in r.reason     # undecided, not a fail


def test_G8_width_fail(tmp_path):
    arts = passing_artifacts()
    arts["query_use.json"] = query_use_doc(width="violation", violators=[7, 9], max_seen=48)
    r = gate(tmp_path, arts, "G8")
    assert r.status == "fail" and "width 'violation'" in r.reason and "2 member class(es) with w_out > w_max=32" in r.reason
    assert r.detail["width"]["max_seen"] == 48 and r.detail["width"]["n_violators"] == 2 and r.detail["width"]["violators"] == [7, 9]
    arts["query_use.json"] = query_use_doc(width="unsupported", unsupported=[3], status="unsupported")
    r = gate(tmp_path, arts, "G8")
    assert r.status == "unsupported" and "undecidable w_out" in r.reason and r.detail["width"]["n_unsupported"] == 1
    assert G.verdict(G.evaluate_gates(tmp_path)) == "INCOMPLETE"
    # `status: ok` claimed but violators listed: still a fail (no violators, no unsupported is part of the rule)
    arts["query_use.json"] = query_use_doc(width="ok", violators=[1])
    assert gate(tmp_path, arts, "G8").status == "fail"
    arts["query_use.json"] = query_use_doc()
    arts["query_use.json"]["use"]["kind"] = "something-else"
    assert "use.kind" in gate(tmp_path, arts, "G8").reason


def test_G8_malformed_is_error_not_skipped(tmp_path):
    arts = passing_artifacts()
    arts["query_use.json"] = dict(query_use_doc(), schema="verity-ir/query/v1")     # the query document, not the use document
    r = gate(tmp_path, arts, "G8")
    assert r.status == "error" and "schema" in r.reason
    assert G.verdict(G.evaluate_gates(tmp_path)) == "INCOMPLETE"
    arts["query_use.json"] = query_use_doc()
    del arts["query_use.json"]["verdict"]["width"]
    r = gate(tmp_path, arts, "G8")
    assert r.status == "error" and "width" in r.reason
    write_run(tmp_path, arts)
    (tmp_path / "query_use.json").write_text("{not json")
    assert G.gate_G8(G.RunArtifacts(tmp_path)).status == "error"


def test_G8_never_changes_G1_G7_or_their_frozen_set():
    assert G.GATES == ("G1", "G2", "G3", "G4", "G5", "G6", "G7") and "G8" not in G.GATES and "G8" not in G.HOLDOUT_GATES
    assert G.STRUCTURAL_GATES == ("G8",) and G.INTEGRITY_GATES == ("R1",) and G.ALL_GATES == G.GATES + ("G8", "R1")
    ok = [G.GateResult(g, G.GATE_NAMES[g], "pass") for g in G.GATES]
    assert G.verdict(ok) == "ACCEPTED"                                                   # a record that predates G8
    assert G.verdict(ok + [G.GateResult("G8", "VU partition", "skipped")]) == "ACCEPTED"
    assert G.verdict(ok + [G.GateResult("G8", "VU partition", "pass")]) == "ACCEPTED"
    assert G.verdict(ok + [G.GateResult("G8", "VU partition", "fail")]) == "REJECTED"
    assert G.verdict(ok + [G.GateResult("G8", "VU partition", "error")]) == "INCOMPLETE"
    assert G.verdict(ok + [G.GateResult("G8", "VU partition", "unsupported")]) == "INCOMPLETE"   # undecided, never REJECTED
    inc = copy.deepcopy(ok)
    inc[6].status = "skipped"
    assert G.verdict(inc + [G.GateResult("G8", "VU partition", "pass")]) == "INCOMPLETE"  # G8 pass never rescues G1..G7


# --- card ---------------------------------------------------------------------------------------------------

PROVENANCE = {"core_commit": "2d62578a7", "lane_commit": "49cafcd5b", "protected_files_modified": [], "existing_primitives_modified": [],
              "new_root_classes": [], "new_allowlist_entries": [], "golden_digest_changes": [], "new_patterns": ["Qwen2Bias"], "new_definitions": []}


def _card(tmp_path, arts):
    write_run(tmp_path, arts)
    return CARD.build_card(tmp_path, G.evaluate_gates(tmp_path))


def test_card_top_section_sealed(tmp_path):
    arts = passing_artifacts()
    arts["provenance.json"] = PROVENANCE
    arts["capture/versions.json"] = {"torch": "2.13.0+cu129", "cuda": "12.9", "triton": "3.7.1", "vllm": "0.28.1rc1.dev472+gd9105ea80",
                                     "gpu": {"name": "NVIDIA GeForce RTX 4090", "cc": "8.9"}, "host": {"clocksource": "tsc", "clock_read_us": 0.03}}
    arts["census/census.json"] = {"engine_kwargs": {"dtype": "bfloat16", "max_num_seqs": 8, "enforce_eager": True, "enable_prefix_caching": False}}
    card = _card(tmp_path, arts)
    assert card.unsealed is False and card.verdict == "ACCEPTED"
    top = CARD.top_section(card).splitlines()
    assert top[0] == "CASE: Qwen/Qwen2.5-1.5B / bf16 / eager / attention-unknown / batch=8"
    assert top[1].startswith("CORE COMMIT: 2d62578a7") and "LANE COMMIT: 49cafcd5b" in top[1]
    rows = {ln[:34].rstrip(): ln[34:].split() for ln in top[2:22]}
    assert rows["PROTECTED FILES MODIFIED"] == ["0", "PASS"]
    assert rows["EXISTING PRIMITIVES MODIFIED"] == ["0", "PASS"]
    assert rows["NEW ROOT CLASSES"] == ["0", "PASS"]
    assert rows["NEW ALLOWLIST ENTRIES"] == ["0", "PASS"]
    assert rows["OLD GOLDEN DIGEST CHANGES"] == ["0", "PASS"]
    assert rows["Unsupported"] == ["0", "PASS"]
    assert rows["Ambiguous matches"] == ["0", "PASS"]
    assert rows["Unknown/unvalidated roots"] == ["0", "PASS"]
    assert rows["CUDA census residue"] == ["0", "PASS"]
    assert rows["New patterns"] == ["1"]
    assert rows["New registered definitions"] == ["0"]
    assert rows["Differential semantic tests"] == ["n/a"]
    assert rows["Captured specializations"] == ["2/2"]
    assert rows["Operand provenance (exhaustive)"] == ["28450", "reads,", "0", "mismatches", "PASS"]
    assert rows["Sampled VUs (bit exact)"][-1] == "PASS" and rows["Sampled VUs (bit exact)"][0] == "k=3"
    assert rows["Served tokens from logits"] == ["4/4", "rows", "at", "steps", "[0,", "1,", "2,", "3]", "PASS"]
    assert rows["Chained slices"] == ["3", "slices,", "17", "chained", "edges", "PASS"]
    assert rows["Control non-interference"] == ["PASS"]
    assert rows["VU coverage (informational)"][0] in ("not", "unavailable") or "VUs" in rows["VU coverage (informational)"]   # D85 row
    assert rows["VU partition (G8, structural)"] == ["skipped", "(no", "query-use", "document)"]                              # G8 row
    assert top[22] == "HOLDOUTS"
    assert top[23].startswith("  qwen15_2req") and top[23].endswith("PASS")
    assert top[24] == "SUPPORT CLAIM"
    assert top[25] == "  Serve_v4 dense / bf16 / eager / attention-unknown / batch=8 / sm_89 / vLLM d9105ea80"
    assert "sampled local semantic checks" in top[27]
    assert top[28] == "VERDICT: ACCEPTED"
    # every label column is exactly 34 wide
    for ln in top[2:22]:
        assert len(ln) > 34 and ln[33] == " "
    md = CARD.summarize_md(card)
    assert "~~~text\n" + "\n".join(top) + "\n~~~" in md
    assert "| G7 | holdout generalization | **pass** |" in md
    assert "| G8 | VU partition (structural) | **skipped** | no query-use document |" in md


def test_card_vu_partition_row(tmp_path):
    arts = passing_artifacts()
    arts["query_use.json"] = query_use_doc()
    card = _card(tmp_path, arts)
    assert card.verdict == "ACCEPTED" and card.gates["G8"]["status"] == "pass"
    assert card.counts["vu_partition"] == f"exact, {QU_COUNT} sets, max w_out {QU_MAX_SEEN}/{QU_WMAX} PASS"
    rows = {ln[:34].rstrip(): ln[34:] for ln in CARD.top_section(card).splitlines() if len(ln) > 34}
    assert rows["VU partition (G8, structural)"] == card.counts["vu_partition"]
    assert "| G8 | VU partition (structural) | **pass** |" in CARD.summarize_md(card)
    assert "query_use" in card.artifacts and card.artifacts["query_use"] == "query_use.json"
    # fail: status, count, max w_out and the codes
    arts["query_use.json"] = query_use_doc(partition="gap", codes=["GAP:root/4"], width="violation", violators=[4], max_seen=40, count=1435)
    card = _card(tmp_path, arts)
    assert card.verdict == "REJECTED"
    assert card.counts["vu_partition"] == "gap, 1435 sets, max w_out 40/32; codes ['GAP:root/4'], 1 violators, status fail FAIL"
    # unsupported (lead 22:22Z): neither PASS nor FAIL, the reason first, verdict INCOMPLETE
    arts["query_use.json"] = query_use_doc(width="unsupported", unsupported=[2, 5], status="unsupported", count=900)
    card = _card(tmp_path, arts)
    assert card.verdict == "INCOMPLETE" and card.gates["G8"]["status"] == "unsupported"
    assert card.counts["vu_partition"] == ("unsupported (width 'unsupported'; 2 member class(es) with undecidable w_out: [2, 5]; "
                                           f"verdict.status 'unsupported'); exact, 900 sets, max w_out {QU_MAX_SEEN}/{QU_WMAX}")
    assert "| G8 | VU partition (structural) | **unsupported** | unsupported (width 'unsupported';" in CARD.summarize_md(card)
    # a record that predates G8 (gates.json without it): n/a, verdict unchanged
    assert CARD.vu_partition_text({g: {"status": "pass"} for g in G.GATES}) == "n/a (record predates G8)"
    old = [G.GateResult(g, G.GATE_NAMES[g], "pass") for g in G.GATES]
    write_run(tmp_path, passing_artifacts())
    card = CARD.build_card(tmp_path, old)
    assert card.verdict == "ACCEPTED" and "| G8 | VU partition (structural) | n/a | record predates G8 |" in CARD.summarize_md(card)


def test_experiment_index_vu_partition(tmp_path):
    from verity_vllm.pipeline import experiment as EXP
    arts = passing_artifacts()
    write_run(tmp_path, arts)
    G.write_gates(tmp_path, G.evaluate_gates(tmp_path))
    doc = EXP.build_experiment(tmp_path)
    assert doc["checks"]["results"]["query_use"] == {"path": "query_use.json", "present": False}
    assert doc["checks"]["vu_partition"] == {"present": False, "gate": {"status": "skipped", "reason": "no query-use document"}}
    arts["query_use.json"] = query_use_doc(partition="overlap", codes=["OVERLAP:root/1"], max_seen=31)
    write_run(tmp_path, arts)
    G.write_gates(tmp_path, G.evaluate_gates(tmp_path))
    vp = EXP.build_experiment(tmp_path)["checks"]["vu_partition"]
    assert vp["present"] and vp["schema"] == G.QUERY_USE_SCHEMA and vp["binding_id"] == "b" * 64 and vp["use"] == {"kind": "vu-partition", "w_max": 32}
    assert vp["partition"] == {"status": "overlap", "codes": ["OVERLAP:root/1"]} and vp["width"]["max_seen"] == 31 and vp["count"] == QU_COUNT
    assert vp["gate"]["status"] == "fail" and "OVERLAP:root/1" in vp["gate"]["reason"]
    assert vp["status"] == vp["verdict_status"] == "fail" and vp["binding"] == {"checked": False}
    res = EXP.build_experiment(tmp_path)["checks"]["results"]
    assert res["query_use"]["present"] is True and res["query"] == {"path": "query.json", "present": False}
    # with the companion query document: binding lifted from G8's detail, query.json referenced
    digest = arts["fold_summary.json"]["digest"]
    arts["query_use.json"] = query_use_doc()
    arts["query.json"] = {"schema": "verity-ir/query/v1", "query_id": "q" * 64, "program_digest": digest, "binding_id": "b" * 64, "query": {}}
    write_run(tmp_path, arts)
    G.write_gates(tmp_path, G.evaluate_gates(tmp_path))
    doc = EXP.build_experiment(tmp_path)
    vp = doc["checks"]["vu_partition"]
    assert vp["gate"]["status"] == "pass" and vp["binding"]["checked"] is True and vp["binding"]["run_digest"] == digest
    assert doc["checks"]["results"]["query"]["present"] is True


def test_card_unsealed_and_failures(tmp_path):
    arts = passing_artifacts()
    arts["resolution.json"]["totals"]["unsupported"] = 2
    arts["census_reconcile.json"]["totals"]["residue_kernels"] = 1
    arts["wiring.json"]["n_mismatch"] = 4
    arts["wiring.json"]["status"] = "fail"
    del arts["holdouts.json"]
    card = _card(tmp_path, arts)
    assert card.unsealed is True and card.verdict == "REJECTED"
    top = CARD.top_section(card)
    assert "CORE COMMIT: n/a" in top and "LANE COMMIT: n/a" in top
    rows = {ln[:34].rstrip(): ln[34:].split() for ln in top.splitlines() if len(ln) > 34}
    assert rows["PROTECTED FILES MODIFIED"] == ["n/a", "n/a"]
    assert rows["Unsupported"] == ["2", "FAIL"]
    assert rows["CUDA census residue"] == ["1", "FAIL"]
    assert rows["Operand provenance (exhaustive)"][-1] == "FAIL"
    assert rows["Control non-interference"] == ["FAIL"]
    assert "  (none)" in top and top.endswith("VERDICT: REJECTED")
    assert "(unsealed: no provenance.json)" in CARD.summarize_md(card)


def test_card_json_roundtrip(tmp_path):
    arts = passing_artifacts()
    card = _card(tmp_path, arts)
    p = CARD.write_card(card, tmp_path / "cards" / f"{card.config_id}.json")
    back = CARD.load_card(p)
    assert back.schema == "verity-gen/card/v1" and back.gates["G1"]["status"] == "pass" and back.C["NL"] == {"value": NL, "provenance": "hand"}
    assert back.C["BN"]["provenance"] == "derived"
    assert set(back.counts) >= {"unsupported", "ambiguous", "operand_provenance", "sampled_vus", "served_tokens", "chained_slices"}
    assert back.diagnostics["structure"]["status"] == "ok"


def test_card_ids():
    assert CARD.slug("Qwen/Qwen2.5-1.5B") == "qwen--qwen2.5-1.5b"
    ea = {"dtype": "bfloat16", "enforce_eager": True, "tensor_parallel_size": 1, "max_num_seqs": 8, "enable_prefix_caching": False, "enable_chunked_prefill": True}
    assert CARD.engine_args_slug(ea) == "bf16-eager-tp1-seqs8-pc0-cp1"
    v = {"torch": "2.13.0+cu129", "cuda": "12.9", "triton": "3.7.1", "vllm": "x", "gpu": {"name": "RTX 4090", "cc": "8.9"}}
    k = CARD.kappa_short(v)
    assert len(k) == 10 and k == CARD.kappa_short(dict(v, host={"nproc": 8}))
    assert CARD.config_id_of("Qwen/Qwen2.5-1.5B", ea, v) == f"qwen--qwen2.5-1.5b__bf16-eager-tp1-seqs8-pc0-cp1__{k}"


def test_c_provenance():
    C = {"model": "m", "H": 1536, "NL": 28, "QKV": 2048, "BN": 64, "EPS": 1e-6}
    out = CARD.c_with_provenance(C, {"H": 1536, "NL": 27, "EPS": 1e-6})
    assert out["H"]["provenance"] == "hf-config" and out["NL"]["provenance"] == "hand" and out["QKV"]["provenance"] == "derived"
    assert out["EPS"]["provenance"] == "hf-config" and out["model"]["provenance"] == "hand"
    assert {k: v["provenance"] for k, v in CARD.c_with_provenance(C, None).items()}["H"] == "hand"
