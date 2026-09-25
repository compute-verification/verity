"""[v2 5b] The Commit verdict as typed records (`verity_vllm.check.verdict`, vllm-verdict/v1).

What is pinned here: (1) the schema round-trips (dumps -> load -> the same document; the outcome / group vocabularies are closed);
(2) the old -> new MAPPING is complete with no loss -- every compat field a check names is documented in OLD_TO_NEW and every
documented old fact lands on a check that exists, every group has rules and a claim; (3) DECISION EQUIVALENCE with v1 -- over a PASS
record and one-fault mutations of it, `run_outcome(run_checks(...)) == PASS` iff `commit_verdict(...)[0]`; (4) a MISSING INPUT is
INSUFFICIENT_EVIDENCE, not FAIL: the Match account leg(s) absent beside the Commit (`account_missing`, F-r19-int-19) grade
`execution_extent` INSUFFICIENT_EVIDENCE while the overall verdict is still not PASS (v1 fails the same record by name); (5) X-03's
per-layer uniformity finding is carried as the named difference `fa2_layers_bad` under `program_source_identity/identity_layout`;
(6) `from_record` over a row dir reads the execution configuration from the Commit's own evidence (`execution_of_record`; a pre-P5
row's `commit_summary.json` still supplies `rc` / execution when present -- the writer is retired, P5); (7) the by-name tables today's checks
lean on are marked `structural_leak: <table>` on the verdict, never silently."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.check.commit_verdict import commit_verdict
from verity_vllm.check import verdict as V

RUN_ROOT = "71f69377" + "ab" * 28
SEED_ROOT = int.from_bytes(bytes.fromhex(RUN_ROOT)[:8], "big")
ROOT_FORM = "run root (first 8 bytes, after commitment)"
CLASSES = {"input_token_ids": {"how": "bound-as-leaf", "tensors": 4}, "instance_outputs_native": {"how": "bound-as-leaf", "tensors": 400},
           "sampled_token_ids": {"how": "bound-as-leaf", "tensors": 4}, "weights": {"how": "registered-input", "tensors": 100},
           "fa2_hidden_m1_stream": {"how": "bound-as-leaf", "tensors": 64}}


def _sr(**kw) -> dict:
    return {"result": True, "grade": "complete", "why": None, "strata_unevaluated": [],
            "sample": {"per_stratum": 2, "seed": SEED_ROOT, "seed_source": ROOT_FORM, "picked": 407, "evaluated": 407, "equal": 407, "mismatch_n": 0, "not_evaluated_n": 0},
            "population": {"vus": 57678, "not_evaluable": {}, "not_evaluable_by_family": {},
                           "query_total": {"identities": 57678}, "query_checks": {"ok": True, "executed_prefix_rule": {"applied": True, "faults": []}}}, **kw}


def _epr(ok: bool = True) -> dict:
    return {"ok": ok, "lag": 1, "eos_of_record": [2], "served_known": True, "any_eos_stop": False,
            "requests": {"r0": {"served_output_len": 8, "executed_prefix_len": 8, "cap": 8, "cap_exhausted": True, "eos_stop": False}}}


def _link(ok: bool = True, required: bool = False, problems: list | None = None) -> dict:
    d = {"ok": ok, "required": required, "lag_attested": ok, "problems": problems or [], "global_match": {"verdict": "PASS", "g6_status": "PASS", "requests_linked": ["r0"]}}
    if ok:
        d["tokens_json"] = {"path": "/x/match/capture/tokens.json", "sha256": "ef" * 32}
    return d


def _base() -> dict:
    # the record's shape: executed_prefix.py's outputs live under binding.coverage (commit_delta [R17-27]) and the facts of record under
    # sampled_replay.executed_prefix_of_record.facts_of_record (sampled_replay executed_prefix=)
    sr = _sr()
    sr["executed_prefix_of_record"] = {"facts_of_record": {"r0": {"executed_prefix_len": 8, "cap": 8, "linked": True, "account_missing": []}}}
    return {"extra": {"expected_tensors": 572, "committed_tensors": 572, "errors_n": 0,
                      "coverage": {"ok": True, "checked": 572, "missing_n": 0, "committed_extras_n": 0, "duplicate_names_n": 0, "manifest_digest": "cd" * 32, "checked_by_family": {"instance_outputs": 400}},
                      "identity_check": {"ok": True, "steps_checked": 4, "violations": [], "violations_n": 0,
                                         "manifest": {"modules": 10, "attention_layers": 2, "fa2_hidden_per_layer": 1, "sampler_ids": 1, "moe_layers": 0, "tp_sites": 0}}},
            "openings": {"n": 64, "verified": 64}, "run_root": RUN_ROOT, "classes": CLASSES,
            "binding": {"program_digest": "ab" * 32, "context": {"ok": True},
                        "coverage": {"ok": True, "checked": 572, "missing_n": 0, "manifest_digest": "cd" * 32,
                                     "executed_prefix_of_record": _epr(), "match_account_link": _link(),
                                     "rows_past_the_stop": {"ok": True, "entries_n": 0, "requests": {}, "steps_past_by_request": {}, "padding_entries_n": 0, "padded_record": False, "error": None}},
                        "openings_population": {"skipped": "test", "unbound_n": 0}, "population": {"requests": 1}},
            "value_correspondence": {"method": "sampled-exact-replay(complete) + boundary-linkage", "result": True},
            "sampled_replay": sr,
            "boundary_linkage": {"result": True, "pairs": 32, "linked": 32, "unlinked_n": 0},
            "weights_pin": {"result": True, "compared": 163, "mismatched": [], "missing": [], "program_digest": "ab" * 32}}


def _decl(classes: dict) -> dict:
    return {"classes": [{"class": k, **v} for k, v in classes.items()]}


def _both(commit: dict, tokens_equal: bool | None = True):
    """(v1 decision, v2 outcome, v2 checks) for one instrumented run."""
    decl = _decl(commit["classes"])
    ok, why = commit_verdict(commit, tokens_equal, declaration=decl)
    checks = V.run_checks(commit, tokens_equal, declaration=decl, ctx={"program_digest": "ab" * 32, "manifest_digest": "cd" * 32})
    return (ok, why), V.run_outcome(checks), checks


# ---------------------------------------------------------------------------------------------------------------- (1) schema

def test_schema_round_trip_and_closed_vocabularies(tmp_path):
    (ok, why), out, checks = _both(_base())
    assert ok is True and out == V.PASS, why
    v = V.build([{"arm": "instrumented", "pair": 0, "tokens_equal_ref": True, "commit": dict(_base(), commit_pass=True)}],
                {"commit_pass": True, "runs_instrumented": 1, "runs_failed": 0, "required_classes": list(CLASSES), "required_manifest": {"program_digest": "ab" * 32, "manifest_digest": "cd" * 32, "query": {"query_id": "Q_module_v1"}}},
                summary={"rc": 0, "pass": True}, manifest_verify={"ok": True}, row="unit")
    assert v.schema == "vllm-verdict/v1" and v.outcome == V.PASS and v.program_digest == "ab" * 32 and v.manifest_digest == "cd" * 32 and v.query_id == "Q_module_v1"
    p = v.write(tmp_path / "verdict.json")
    back = V.Verdict.load(p)
    assert back.to_dict() == v.to_dict()
    assert json.loads(p.read_text())["schema"] == "vllm-verdict/v1"
    assert set(V.GROUPS) == {"runtime_match", "local_replay", "boundary_linkage", "checkpoint_binding", "execution_extent", "required_value_coverage", "program_source_identity"}
    assert v.groups() == {g: V.PASS for g in V.GROUPS}
    for c in v.checks:
        assert c.outcome in V.OUTCOMES and c.claim and c.compat and c.group in V.GROUPS
    with pytest.raises(ValueError):
        V.CheckVerdict(name="runtime_match/x", claim="c", outcome="MAYBE")
    with pytest.raises(ValueError):
        V.CheckVerdict(name="weights_pin/x", claim="c", outcome=V.PASS)          # the OLD name is not a group
    with pytest.raises(ValueError):
        V.Verdict.from_dict({"schema": "something-else/v9", "checks": [], "outcome": V.PASS})


def test_worst_orders_fail_over_insufficient_over_not_run():
    assert V.worst([V.PASS, V.NOT_RUN]) == V.PASS
    assert V.worst([V.PASS, V.INSUFFICIENT]) == V.INSUFFICIENT
    assert V.worst([V.INSUFFICIENT, V.FAIL, V.PASS]) == V.FAIL
    assert V.worst([]) == V.NOT_RUN


# ---------------------------------------------------------------------------------------------------------------- (2) mapping

def test_mapping_old_to_new_is_complete_with_no_loss():
    _, _, checks = _both(_base())
    names = {c.name for c in checks}
    documented_old = {tok.strip() for m in V.OLD_TO_NEW for tok in m["old"].split(" / ")}
    emitted_compat = {f for c in checks for f in c.compat}

    def documented(f: str) -> bool:
        n = f[len("extra."):] if f.startswith("extra.") else f
        return any(n == o or f == o or n.startswith(o + ".") or n.startswith(o) or o.startswith(n) for o in documented_old)

    # every compat field a check names is a documented old fact ...
    missing_doc = sorted(f for f in emitted_compat if not documented(f))
    assert not missing_doc, f"compat fields not in OLD_TO_NEW: {missing_doc}"
    # ... and every documented old fact lands on a check / group / Verdict field that exists
    targets = {m["new"].split(" ")[0].strip("`").rstrip(";,") for m in V.OLD_TO_NEW}
    unknown = []
    for t in targets:
        if "/" in t:
            g, r = t.split("/", 1)
            if not (g in V.GROUPS and (r == "*" or t in names)):
                unknown.append(t)
        elif t in V.GROUPS or t in ("every", "the", "deprecated"):
            continue
        elif t.startswith(("verdict.", "compat.")):
            fld = t.split(".")[1] if t.startswith("verdict.") else "compat"
            if fld not in V.Verdict.__dataclass_fields__:
                unknown.append(t)
        else:
            unknown.append(t)
    assert not unknown, f"OLD_TO_NEW names checks / fields that do not exist: {unknown}"
    assert sum(1 for t in targets if "/" in t and t in names) >= 15
    # each old check name of the prompt has its new name
    for old, new in (("linkage", "boundary_linkage"), ("weights_pin", "checkpoint_binding"), ("executed_prefix", "execution_extent"),
                     ("coverage", "required_value_coverage"), ("identity", "program_source_identity")):
        assert any(old in m["old"] and new in m["new"] for m in V.OLD_TO_NEW), (old, new)
    # every group has at least one rule check and a one-sentence claim
    for g, spec in V.GROUPS.items():
        assert spec["claim"].endswith(".") and any(c.name.startswith(g + "/") for c in checks), g
    assert "| `commit_pass` |" in V.mapping_markdown()


# ---------------------------------------------------------------------------------------------------------------- (3) equivalence

def _mutations():
    b = _base
    yield "tokens differ", b(), False
    c = b(); c["extra"]["committed_tensors"] = 571; yield "reference integrity", c, True
    c = b(); c["extra"]["errors_n"] = 2; yield "collector errors", c, True
    c = b(); c["openings"] = {"n": 60, "verified": 60}; yield "openings < min", c, True
    c = b(); c["openings"] = {"n": 64, "verified": 63}; yield "an opening failed", c, True
    c = b(); c["binding"]["context"] = {"ok": False, "why": "root differs"}; yield "binding context", c, True
    c = b(); c["binding"]["coverage"] = {"ok": False, "missing_n": 3, "expected_n": 572, "missing": [{"step": 1, "op_path": "layers.0.mlp", "output_member": "y", "invocation": 0, "request_id": "r0", "rank": 0}]}
    yield "identity coverage", c, True
    c = b(); c["extra"]["coverage"] = {"ok": False, "checked": 500, "missing_n": 72, "missing_by_family": {"fa2_hidden_m1_stream": 72}, "families_missing_entirely": ["fa2_hidden_m1_stream"],
                                       "missing": [{"step": 0, "op_path": "layers.0.attn", "output_member": "m1", "request_id": "r0"}], "manifest_digest": "cd" * 32}
    yield "manifest coverage", c, True
    c = b(); c["extra"]["identity_check"] = {"ok": False, "steps_checked": 4, "violations_n": 2, "manifest": {"sampler_ids": 1},
                                             "violations": [{"step": 1, "fa2_layers_bad": {"0": 48}, "sampler_ids": 1}, {"step": 2, "fa2_layers_bad": {"0": 42}, "sampler_ids": 1}]}
    yield "X-03 identity layout", c, True
    c = b(); c["weights_pin"] = {"result": False, "why": "1 mismatched", "mismatched": ["model.layers.0.q_proj"]}; yield "checkpoint binding", c, True
    c = b(); c["boundary_linkage"] = {"result": False, "why": "1 unlinked", "unlinked_n": 1}; yield "boundary linkage", c, True
    c = b(); c["sampled_replay"]["result"] = False; c["sampled_replay"]["why"] = "a VU differs"; yield "replay mismatch", c, True
    c = b(); c["sampled_replay"]["sample"]["seed"] = 12345; c["oracle_compare"] = {"result": True, "complete": False}; yield "prover-chosen seed, no form (B)", c, True
    c = b(); c["sampled_replay"]["population"]["query_checks"] = {"ok": False, "vus_outside_query": {"n": 1, "first": [{"request_id": "r0", "step": 1, "op_path": "x", "member": "y"}]}}
    yield "population scope", c, True
    c = b(); del c["weights_pin"]; yield "weights unpinned", c, True
    c = b(); c["classes"] = {k: v for k, v in CLASSES.items() if k != "weights"}; yield "required class undeclared", c, True
    c = b(); c["classes"] = dict(CLASSES, weights={"how": "registered-input", "tensors": 0}); yield "declared leaf class bound 0", c, True
    c = b(); c["openings_after_release"] = {"n": 64, "verified": 60, "source": "host"}; yield "openings after release", c, True
    c = b(); c["oracle_compare"] = {"result": False, "mismatch_n": 3}; yield "oracle mismatch", c, True


@pytest.mark.parametrize("label,commit,tokens_equal", list(_mutations()), ids=[m[0] for m in _mutations()])
def test_decisions_equal_v1_on_one_fault_mutations(label, commit, tokens_equal):
    (ok, why), out, checks = _both(commit, tokens_equal)
    assert ok is False, f"{label}: v1 should FAIL this mutation ({why})"
    assert out != V.PASS, f"{label}: v2 PASS where v1 FAILS ({why}); checks {[(c.name, c.outcome) for c in checks if c.outcome != V.PASS]}"
    graded = [c for c in checks if c.outcome in (V.FAIL, V.INSUFFICIENT)]
    assert graded and all(c.differences or c.reason for c in graded), f"{label}: a non-PASS check without a named difference"


def test_pass_record_passes_in_both_and_names_its_scope():
    (ok, _), out, checks = _both(_base())
    assert ok is True and out == V.PASS
    by = {c.name: c for c in checks}
    assert by["local_replay/sampled_replay"].scope["sample_size"] == 407 and by["local_replay/sampled_replay"].inputs["sample_seed"] == SEED_ROOT
    assert by["runtime_match/openings"].scope["sample_size"] == 64 and by["runtime_match/openings"].scope["verified"] == 64
    assert by["checkpoint_binding/weights_of_record"].scope["compared"] == 163
    assert by["program_source_identity/identity_layout"].scope["steps_checked"] == 4
    assert by["required_value_coverage/identity_coverage"].inputs["manifest_digest"] == "cd" * 32


# ---------------------------------------------------------------------------------------------------------------- (4) missing input

def test_account_missing_is_insufficient_evidence_not_fail():
    c = _base()
    fault = ("Match account leg(s) missing beside the Commit: tokens.json -- the link cannot hold on a missing input (restore the Match record's file(s) "
             "and re-run the Commit; not a verdict on the Program)")
    c["binding"]["coverage"]["match_account_link"] = _link(ok=False, required=True, problems=["tokens.json: not beside the Commit"])
    c["sampled_replay"]["executed_prefix_of_record"] = {"facts_of_record": {"r0": {"executed_prefix_len": 6, "cap": 8, "linked": False, "account_missing": ["tokens.json"]}}}
    c["sampled_replay"]["population"]["query_checks"] = {"ok": False, "executed_prefix_rule": {"applied": True, "faults": [{"request_id": "r0", "faults": [fault]}]}}
    (ok, why), out, checks = _both(c)
    assert ok is False and "executed-prefix rule faults" in why            # v1: a named FAIL through the replay's population scope
    by = {c_.name: c_ for c_ in checks}
    link = by["execution_extent/match_account_link"]
    assert link.outcome == V.INSUFFICIENT and link.differences[0] == {"name": "account_missing", "legs": ["tokens.json"]}
    assert by["local_replay/population_scope"].outcome == V.INSUFFICIENT and out != V.PASS
    # the cause is read from the facts of record (`account_missing`), never the fault's text: a reworded fault grades the same
    c["sampled_replay"]["population"]["query_checks"]["executed_prefix_rule"]["faults"][0]["faults"] = ["the tokens.json leg is gone"]
    reworded = {c_.name: c_.outcome for c_ in _both(c)[2]}
    assert reworded == {c_.name: c_.outcome for c_ in checks}
    groups = V.group_checks(V.aggregate_runs([(0, checks)]))
    g = {x.name: x.outcome for x in groups}
    assert g["execution_extent"] == V.INSUFFICIENT, g
    # the SAME missing legs on a record where no request stopped before its cap: recorded, not gating (v1 passes it too)
    c2 = _base()
    c2["sampled_replay"]["executed_prefix_of_record"] = {"facts_of_record": {"r0": {"executed_prefix_len": 8, "cap": 8, "linked": False, "account_missing": ["tokens.json"]}}}
    c2["binding"]["coverage"]["match_account_link"] = _link(ok=False, required=False, problems=["tokens.json: not beside the Commit"])
    (ok2, _), out2, checks2 = _both(c2)
    link2 = next(x for x in checks2 if x.name == "execution_extent/match_account_link")
    assert ok2 is True and out2 == V.PASS and link2.outcome == V.PASS and link2.scope["account_missing"] == ["tokens.json"] and link2.notes


def test_not_run_when_the_record_lacks_the_evidence():
    c = _base()
    del c["extra"]["identity_check"]
    for k in ("executed_prefix_of_record", "match_account_link", "rows_past_the_stop"):
        del c["binding"]["coverage"][k]
    del c["sampled_replay"]["executed_prefix_of_record"]
    (ok, _), out, checks = _both(c)
    assert ok is True and out == V.PASS
    by = {x.name: x for x in checks}
    assert by["program_source_identity/identity_layout"].outcome == V.NOT_RUN
    assert all(by[n].outcome == V.NOT_RUN for n in ("execution_extent/executed_prefix_of_record", "execution_extent/match_account_link", "execution_extent/rows_past_the_stop"))
    groups = {g.name: g.outcome for g in V.group_checks(V.aggregate_runs([(0, checks)]))}
    # a group with NO graded rule is NOT_RUN; program_source_identity still has binding_context graded (PASS), so it stays PASS
    assert groups["execution_extent"] == V.NOT_RUN and groups["program_source_identity"] == V.PASS
    assert by["program_source_identity/binding_context"].outcome == V.PASS


# ---------------------------------------------------------------------------------------------------------------- (5) X-03

def test_x03_fa2_layers_bad_is_a_named_difference_under_program_source_identity():
    c = _base()
    c["extra"]["identity_check"] = {"ok": False, "steps_checked": 128, "violations_n": 106, "manifest": {"modules": 265, "attention_layers": 26, "fa2_hidden_per_layer": 1, "sampler_ids": 1},
                                    "violations": [{"step": s, "missing_modules": [], "unknown_modules": [], "fa2_layers_bad": {"0": 48, "2": 48}, "moe_layers_bad": {}, "sampler_ids": 1} for s in range(1, 6)]}
    (ok, why), out, checks = _both(c)
    assert ok is False and why.startswith("identity check:")
    il = next(x for x in checks if x.name == "program_source_identity/identity_layout")
    assert il.outcome == V.FAIL and il.rule.startswith("X-03") and [d["name"] for d in il.differences] == ["fa2_layers_bad"]
    assert il.differences[0]["steps_n"] == 5 and il.differences[0]["first"]["fa2_layers_bad"] == {"0": 48, "2": 48}
    assert "extra.identity_check.violations" in il.compat and il.scope["layers"] == 26
    assert any(n.startswith("structural_leak: native_host.ACQUIRE_CLASSES") for n in il.notes)


# ---------------------------------------------------------------------------------------------------------------- (6) from_record

def _row_dir(tmp_path, commit: dict, *, commit_pass: bool, rc: int, manifest_verify_ok: bool | None = True):
    d = tmp_path / "row"
    (d / "commit").mkdir(parents=True)
    rows = [{"arm": "control", "pair": 0, "wall_s": 1.0, "committer": "control"},
            {"arm": "instrumented", "pair": 0, "wall_s": 1.5, "tokens_equal_ref": True, "tokens_equal": True, "committer": "native_collect_v2b", "engine_steps": 8, "generated_tokens": 8,
             "commit": dict(commit, commit_pass=commit_pass, commit_fail_reason=None if commit_pass else "x", run_root=RUN_ROOT)}]
    (d / "commit" / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "commit" / "verdict.json").write_text(json.dumps({"commit_pass": commit_pass, "runs_instrumented": 1, "runs_failed": 0 if commit_pass else 1, "first_fail_reason": None if commit_pass else "x",
                                                            "required_classes": list(CLASSES), "required_manifest": {"program_digest": "ab" * 32, "manifest_digest": "cd" * 32, "query": {"query_id": "Q_module_v1"}}}))
    (d / "commit" / "summary.json").write_text(json.dumps({"committer": "native_collect_v2b", "delta_wall_mean": 0.5}))
    (d / "commit.log").write_text("$ python -m verity_vllm.pipeline.commit --openings 64 --pairs 1 --required-manifest m.json\n")
    if manifest_verify_ok is not None:
        (d / "commit" / "manifest_verify.json").write_text(json.dumps({"ok": manifest_verify_ok, "manifest_digest": "cd" * 32}))
    (d / "global_match.json").write_text(json.dumps({"verdict": "PASS"}))
    ex = V.execution_of_record(d)
    return d, ex


def test_from_record_pass_row_reads_its_own_execution_configuration(tmp_path):
    d, ex = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    assert ex["committer"] == "native_collect_v2b" and ex["engine_settings"]["openings"] == 64 and ex["engine_settings"]["required_manifest"] is True
    assert not (d / "commit_summary.json").exists()                     # P5: nothing writes the pre-P5 summary
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.outcome == V.PASS and v.compat["commit_pass"] is True and v.compat["stage_ok"] == "PASS" and not v.notes
    assert v.inputs["commit/verdict.json"] and v.inputs["commit/runs.jsonl"] and "global_match.json" in v.inputs
    assert v.correspondence_digest is None and v.correspondence_digest_source == "absent" and v.correspondence_digests == {}   # a v1-frontend row: never a stand-in hash
    assert v.check("required_value_coverage/manifest_of_record_verified").outcome == V.PASS
    assert v.runs == {"instrumented": 1, "failed": 0, "control": 1, "not_pass_by_reconstruction": 0, "pairs": [0]}
    assert v.execution["engine_settings"]["openings"] == 64 and v.execution["committer"] == "native_collect_v2b"
    out = V.main(["from-record", str(d), "--rc", "0", "--stage-ok", "PASS", "--quiet"])
    assert out == 0 and V.Verdict.load(d / "verdict.json").outcome == V.PASS


def test_from_record_pre_p5_row_takes_rc_and_execution_from_its_commit_summary(tmp_path):
    """An old record (commit_summary.json present, commit.log absent): its summary still supplies `rc` and the execution fields."""
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    (d / "commit.log").unlink()
    (d / "commit_summary.json").write_text(json.dumps({"rc": 0, "pass": True, "committer": "native_collect_v2b", "collector_variant": "windowed",
                                                       "retention": "host", "engine_settings": {"openings": 64, "pairs": 1, "required_manifest": True},
                                                       "cmd": "python -m verity_vllm.pipeline.commit --openings 64", "deprecated": "F-r19-int-21; see verdict.json"}))
    v = V.from_record(d, stage_ok="PASS")
    assert v.outcome == V.PASS and v.compat["rc"] == 0 and v.compat["commit_summary_pass"] is True
    assert v.execution["committer"] == "native_collect_v2b" and v.execution["engine_settings"]["openings"] == 64


def test_from_record_manifest_verify_fail_is_required_value_coverage_fail(tmp_path):
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0, manifest_verify_ok=False)
    v = V.from_record(d, rc=0, stage_ok="FAIL")
    assert v.outcome == V.FAIL and v.groups()["required_value_coverage"] == V.FAIL
    mv = v.check("required_value_coverage/manifest_of_record_verified")
    assert mv.outcome == V.FAIL and mv.differences[0]["name"] == "manifest_of_record_differs" and "manifest_verify.ok" in mv.compat[0]
    assert v.compat["commit_pass"] is True and not any("RECONSTRUCTION MISMATCH" in n for n in v.notes)     # commit_delta's decision is intact; verify is the stage's


def test_from_record_missing_commit_is_not_run(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "stages.txt").write_text("build PASS 2026-09-22T00:00:00Z ok\n")
    v = V.from_record(d)
    assert v.outcome == V.NOT_RUN and any("record incomplete" in n for n in v.notes)


# ---------------------------------------------------------------------------------------------------------------- (7) leaks

def test_structural_leaks_are_named_on_the_verdict():
    _, _, checks = _both(_base())
    leaks = {n.split(":", 1)[1].split("(")[0].strip() for c in checks for n in c.notes if n.startswith("structural_leak:")}
    for table in ("native_host.ACQUIRE_CLASSES", "required_manifest.members_for", "weights_of_record.FUSED / EXPERTS / EXPERTS_MIXTRAL",
                  "sampled_replay.EVALUATORS / NOT_YET", "oracle_compare.NO_ORACLE_FAMILY_PREFIXES"):
        assert table in leaks, (table, leaks)


# ---------------------------------------------------------------------------------------------------------------- (8) correspondence digest

def test_correspondence_digest_is_the_builds_own_or_absent(tmp_path):
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    step, req, shape = "e6" * 32, "fc" * 32, "ab" * 32
    (d / "build_summary.json").write_text(json.dumps({"step": {"ok": True, "program_digest": "p1" * 32, "correspondence_digest": step},
                                                      "request": {"ok": True, "program_digest": "p2" * 32, "correspondence_digest": req},
                                                      "build_request_LP9_T1": {"ok": True, "program_digest": "p3" * 32, "correspondence_digest": shape}}))
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.correspondence_digest == step and v.correspondence_digest_source == "build_summary.json:step"
    assert v.correspondence_digests == {"build_request_LP9_T1": shape, "request": req, "step": step}
    assert v.inputs["build_summary.json"] and v.correspondence_digest != v.inputs["global_match.json"]
    assert V.Verdict.from_dict(json.loads(v.dumps())).correspondence_digest == step
    # a build_summary.json of a v1-frontend Build (no correspondence_digest per wrapper): absent, null
    (d / "build_summary.json").write_text(json.dumps({"step": {"ok": True, "program_digest": "p1" * 32, "correspondence_digest": None}, "request": {"ok": True, "program_digest": "p2" * 32}}))
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.correspondence_digest is None and v.correspondence_digest_source == "absent" and v.correspondence_digests == {}
    # no build_summary.json: the wrappers' own result.json
    (d / "build_summary.json").unlink()
    for w, dg in (("build_step", step), ("build_request", req)):
        (d / w).mkdir()
        (d / w / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": dg}}))
    v = V.from_record(d, rc=0, stage_ok="PASS")
    assert v.correspondence_digest == step and v.correspondence_digest_source == "result.json:step" and v.correspondence_digests == {"request": req, "step": step}
    assert V.correspondence_of_build(None) == (None, "absent", {})
    assert V.correspondence_of_build({"request": {"correspondence_digest": req}}) == (req, "build_summary.json:request", {"request": req})
