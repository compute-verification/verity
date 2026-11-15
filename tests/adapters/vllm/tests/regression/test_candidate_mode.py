"""Each check's candidate mode, on small documents in a temp dir (no store, no fixture): with `$VERITY_REGRESSION_CANDIDATE` naming a
directory of row dirs, `actual` lifts the CANDIDATE's own document, by the rule `expected` lifts the reference's by, and never falls
back on the reference's copy of a document the candidate lacks where the check declares that leaf not applicable.

The row is the resolver's own `Row`, located through `$VERITY_REGRESSION_ROWS_ROOT`; the reference and the candidate hold different
values, so each assertion says which side was read.  The manifest check's rebuild is run against a stand-in for the builder (it writes
the manifest the test gives it): what is held is the command line the check builds and what it lifts from the builder's output.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from . import resolver
from .checks import commit_summary as CS
from .checks import global_match_checks as GM
from .checks import manifest_digest as MD
from .checks import program_digest as PD
from .checks import replay_partition as RP
from .checks import step_segmentation as SS
from .checks import stoch_value as SV
from .checks import verdict as VD

KEY = "stand-in-row"


def _write(root: Path, rel: str, doc) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def _trees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: dict, candidate: dict | None, **spec) -> resolver.Row:
    """A row whose reference dir holds `reference` (rel -> document) and whose candidate dir, when given, holds `candidate`."""
    ref = tmp_path / "rows" / KEY
    ref.mkdir(parents=True)
    for rel, doc in reference.items():
        _write(ref, rel, doc)
    monkeypatch.setenv(resolver.ROWS_ROOT_ENV, str(tmp_path / "rows"))
    monkeypatch.setenv(resolver.SCRATCH_ENV, str(tmp_path / "scratch"))
    monkeypatch.delenv(resolver.CANDIDATE_ENV, raising=False)
    if candidate is not None:
        (tmp_path / "candidate" / KEY).mkdir(parents=True)
        for rel, doc in candidate.items():
            _write(tmp_path / "candidate" / KEY, rel, doc)
        monkeypatch.setenv(resolver.CANDIDATE_ENV, str(tmp_path / "candidate"))
    return resolver.Row(KEY, {"row": 900, "class": "greedy", **spec}, fixtures={})


def _workload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requests: int) -> Path:
    """The row's declared workload, with `requests` requests (the registered one is found by the row's key, which a stand-in has not)."""
    wl = tmp_path / "workload.json"
    wl.write_text(json.dumps({"requests": [{"request_id": f"r{i}"} for i in range(requests)]}))
    monkeypatch.setattr(resolver.Row, "workload", lambda self: wl)
    return wl


def _verdict(outcome: str) -> dict:
    bad = outcome != "PASS"
    return {"outcome": outcome, "schema": "verdict/v2", "compat": {"commit_pass": not bad, "rc": int(bad)}, "program_digest": f"p-{outcome}",
            "checks": [{"name": "local_replay", "outcome": outcome, "rule": None},
                       {"name": "local_replay/c2", "outcome": outcome, "rule": "C2"},
                       {"name": "program_source_identity/identity_layout", "outcome": "INSUFFICIENT_EVIDENCE" if bad else "PASS", "rule": "X-03",
                        "differences": [{"name": "fa2_layers_bad"}] if bad else []}]}


def test_the_verdict_check_lifts_the_candidates_verdict(tmp_path, monkeypatch):
    row = _trees(tmp_path, monkeypatch, {"verdict.json": _verdict("PASS")}, {"verdict.json": _verdict("FAIL")})
    act = VD.actual(row)
    assert act["source"] == "candidate" and act["pass"] is False and act["class"] == "greedy" and act["program_digest"] == "p-FAIL"
    assert act["groups"] == {"local_replay": "FAIL"} and act["failing"] == ["local_replay/c2"]
    assert act["insufficient"] == ["program_source_identity/identity_layout"] and act["identity_layout_differences"] == ["fa2_layers_bad"]
    monkeypatch.delenv(resolver.CANDIDATE_ENV)
    rec = VD.actual(row)
    assert rec["source"] == "recorded" and rec["pass"] is True and rec["failing"] == [] and rec["identity_layout_differences"] == []


def _global_match(verdict: str, fold_dir: str) -> dict:
    bad = verdict != "PASS"
    return {"verdict": verdict, "failed_checks": ["GM-01"] if bad else [], "requests": ["r0", "r1"],
            "checks": [{"id": "GM-01", "status": verdict, "problems": ["a", "b", "c"] if bad else []}],
            "impl": {"fold_load": {fold_dir: {"source": "fold", "program_sha256": "aa", "instances_sha256": "bb"}}}}


def test_the_global_match_check_lifts_the_candidates_document(tmp_path, monkeypatch):
    row = _trees(tmp_path, monkeypatch, {"global_match.json": _global_match("PASS", "/pod/row/match")},
                 {"global_match.json": _global_match("FAIL", "/elsewhere/match")})
    exp, act = GM.expected(row), GM.actual(row)
    assert act["source"] == "candidate" and act["verdict"] == "FAIL" and act["checks"] == {"GM-01": {"status": "FAIL", "n_problems": 3}}
    assert act["fold_record_pins"] == exp["fold_record_pins"] == {"program_sha256": "aa", "instances_sha256": "bb"}, "the fold dir's path does not enter"
    assert [d.split(":")[0] for d in resolver.diff(exp, act)] == ["$.verdict", "$.failed_checks", "$.checks.GM-01.status", "$.checks.GM-01.n_problems"]
    monkeypatch.delenv(resolver.CANDIDATE_ENV)
    rec = GM.actual(row)
    assert rec["source"] == "record" and rec["verdict"] == "PASS" and rec["consistent"] is True


def _match_stoch(verdict: str) -> dict:
    return {"summary": {"verdict": verdict, "rows_checked": 1}, "reasons": [] if verdict == "PASS" else ["mismatch at step 0"],
            "rows": [{"step": 0, "request_id": "r0", "token": 7, "status": "exact-match" if verdict == "PASS" else "mismatch"}]}


def test_the_sampler_value_check_lifts_the_candidates_record_and_its_snapshot_steps(tmp_path, monkeypatch):
    reference = {"match_stoch.json": _match_stoch("PASS"), "match/run.json": {"snapshot_steps": [0]}}
    row = _trees(tmp_path, monkeypatch, reference, {"match_stoch.json": _match_stoch("FAIL"), "match/run.json": {"snapshot_steps": [2, 0, 1]}})
    act = SV.actual(row)
    assert act == dict(SV.lift(_match_stoch("FAIL"), [2, 0, 1]), source="candidate")
    assert act["verdict"] == "FAIL" and act["snapshot_steps"] == [0, 1, 2] and SV.expected(row)["snapshot_steps"] == [0]
    (tmp_path / "candidate" / KEY / "match" / "run.json").unlink()
    assert SV.actual(row)["snapshot_steps"] is None, "the candidate's run.json, never the reference's"
    (tmp_path / "candidate" / KEY / "match_stoch.json").unlink()
    assert SV.actual(row) == {"source": "candidate", "verdict": resolver.not_applicable("the candidate has no match_stoch.json")}


def _build_summary(tag: str) -> dict:
    return {"request": {"program_digest": f"request-{tag}"}, "step": {"program_digest": f"step-{tag}"}, "workload": {"workload_digest": f"workload-{tag}"},
            "build_request_LP27_T100": {"program_digest": f"shape-{tag}"}}


@pytest.mark.parametrize("requests,of_record", [(1, "request-candidate"), (2, "workload-candidate")])
def test_the_program_digest_check_reads_the_candidates_build_summary(tmp_path, monkeypatch, requests, of_record):
    reference = {"build_summary.json": _build_summary("reference"), "commit/manifest_verify.json": {"used_program_digest": "workload-reference"},
                 "build_workload/workload_program.json": {"workload_digest": "workload-reference"}}
    row = _trees(tmp_path, monkeypatch, reference, {"build_summary.json": _build_summary("candidate")})
    _workload(tmp_path, monkeypatch, requests)
    act = PD.actual(row)
    assert act["source"] == "candidate" and act["batch"] == requests and act["program_digest_of_record"] == of_record
    assert act["build_summary"] == {"request": "request-candidate", "step": "step-candidate", "workload": "workload-candidate",
                                    "shapes": {"build_request_LP27_T100": "shape-candidate"}}
    # a Build-only candidate: the two keys the record's Commit and workload Program give are declared, never lifted from the record
    assert resolver.is_not_applicable(act["commit_keyed_by"]) and resolver.is_not_applicable(act["workload_program_digest"])
    _write(tmp_path / "candidate" / KEY, "commit/manifest_verify.json", {"used_program_digest": "keyed-candidate"})
    _write(tmp_path / "candidate" / KEY, "build_workload/workload_program.json", {"workload_digest": "composed-candidate"})
    act = PD.actual(row)
    assert act["commit_keyed_by"] == "keyed-candidate" and act["workload_program_digest"] == "composed-candidate"
    exp = PD.expected(row)                                  # the reference's, whatever the candidate holds
    assert (exp["build_summary"]["request"], exp["commit_keyed_by"], exp["workload_program_digest"]) == ("request-reference",) + ("workload-reference",) * 2


def _identity(layer: int, **more) -> dict:
    return {"program_digest": "p", "request_id": "r0", "step": 0, "invocation": 0, "op_path": f"model.layers.{layer}.mlp.experts",
            "output_member": "out", "element_range": [0, 4], "rank": 0, **more}


def _manifest(tag: str, identities: list[dict], **more) -> dict:
    return {"manifest_digest": f"manifest-{tag}", "program_digest": "p", "complete": True, "schema": "manifest/v1", "identities": identities, **more}


def test_the_manifest_check_lifts_the_candidates_manifest_and_counts_it_against_the_reference(tmp_path, monkeypatch):
    v1 = [_identity(0), _identity(1)]
    grown = v1 + [_identity(3, promoted=True)]
    row = _trees(tmp_path, monkeypatch, {"manifest.json": _manifest("reference", v1)}, {"manifest.json": _manifest("candidate", grown)})
    act = MD.actual(row)
    assert act["source"] == "candidate" and act["manifest_digest"] == "manifest-candidate" and act["identities"] == 3
    assert act["identity_rows_sha256"] == MD.identity_rows_sha256(grown) != MD.expected(row)["identity_rows_sha256"]
    assert act["promoted"] == 1 and act["promoted_member"] == "model.layers.3.mlp.experts/out"
    assert act["promoted_member_class"] == "model.layers.<k>.mlp.experts/out"
    assert (act["only_new"], act["v1_identities"], act["added_n"], act["removed_n"]) == (1, 2, 1, 0)
    _write(tmp_path / "candidate" / KEY, "manifest.json", _manifest("candidate", grown[1:]))
    act = MD.actual(row)
    assert (act["only_new"], act["added_n"], act["removed_n"]) == (None, 1, 1), "a v1 identity is gone: not an additive difference"


def test_the_manifest_rebuild_is_the_builders_command_line_for_the_rows_batch(tmp_path, monkeypatch):
    row = _trees(tmp_path, monkeypatch, {"build_summary.json": _build_summary("reference")}, None)
    wl, root, out = _workload(tmp_path, monkeypatch, 1), tmp_path / "programs", tmp_path / "out.json"
    cli = [sys.executable, "-m", "verity_vllm.pipeline.cli", "manifest"]
    assert MD._rebuild_argv(row, root, out) == [*cli, "build", "--program", str(root / "build_request"), "--workload", str(wl), "--out", str(out),
                                                "--no-norm-scales", "--no-router-softmax", "--no-vocab-range"]
    _workload(tmp_path, monkeypatch, 2)
    row.spec["manifest_flags"] = []
    assert MD._rebuild_argv(row, root, out) == [*cli, "build-global", "--fixture", str(wl), "--programs-root", str(root), "--out", str(out),
                                                "--workload-digest", "workload-reference"]


def test_the_manifest_rebuild_lifts_what_the_builder_wrote_and_states_a_builder_that_failed(tmp_path, monkeypatch):
    reference = {"manifest.json": _manifest("reference", [_identity(0)]), "build_request/instances.json.gz": "", "build_request/result.json": {}}
    row = _trees(tmp_path, monkeypatch, reference, None)
    _workload(tmp_path, monkeypatch, 1)
    built = _manifest("rebuilt", [_identity(0), _identity(1)], query={"engine": "q-word"})

    def builder(rc: int):
        def run(argv, **_kw):
            if rc == 0:
                Path(argv[argv.index("--out") + 1]).write_text(json.dumps(built))
            return subprocess.CompletedProcess(argv, rc, stdout="built\n", stderr="the builder's refusal" if rc else "")
        return run

    monkeypatch.setattr(MD.subprocess, "run", builder(0))
    act = MD.actual(row)
    assert act["source"] == "rebuild" and act["rebuild_rc"] == 0 and act["rebuild_engine"] == "q-word" and act["manifest_digest"] == "manifest-rebuilt"
    assert (act["promoted"], act["only_new"], act["promoted_member"]) == (1, 1, "model.layers.1.mlp.experts/out"), "a query manifest: its rows beyond v1"
    assert "built" in (resolver.scratch(row) / "manifest.rebuilt.log").read_text()
    (resolver.scratch(row) / "manifest.rebuilt.json").unlink()
    monkeypatch.setattr(MD.subprocess, "run", builder(2))
    act = MD.actual(row)
    assert act["source"] == "rebuild" and act["rebuild_rc"] == 2 and act["rebuild_tail"].endswith("the builder's refusal") and "manifest_digest" not in act


def _sampled_replay(seed: int, result: str) -> dict:
    return {"result": result, "grade": "A", "population": {"vus": 12, "requests": ["r0"], "strata_n": 3},
            "sample": {"seed": seed, "per_stratum": 2, "picked": 6, "picks_by_request_step": {"r0": {"0": [4], "1": 2}}},
            "strata": {"s0": {"population": 4, "sampled": 2}}}


def test_the_partition_check_lifts_the_candidates_record_wrapped_or_bare(tmp_path, monkeypatch):
    row = _trees(tmp_path, monkeypatch, {"commit/sampled_replay_p0.json": _sampled_replay(3, "PASS")},
                 {"commit/sampled_replay_p0.json": {"sampled_replay": _sampled_replay(11, "FAIL")}})
    exp, act = RP.expected(row), RP.actual(row)
    assert act["source"] == "candidate" and act["record"] == {"result": "FAIL", "grade": "A"} and act["sample"]["seed"] == 11
    assert act["sample"]["picks_by_request_step"] == {"r0": {"0": 4, "1": 2}} and act["strata"] == exp["strata"]
    assert [d.split(":")[0] for d in resolver.diff(exp, act)] == ["$.sample.seed", "$.record.result"]
    _write(tmp_path / "candidate" / KEY, "commit/sampled_replay_p0.json", _sampled_replay(3, "PASS"))
    assert resolver.diff(exp, RP.actual(row)) == [], "a bare record is its own body"


def test_the_partition_checks_by_hand_command_exits_by_the_difference_and_writes_it(tmp_path, monkeypatch, capsys):
    row = _trees(tmp_path, monkeypatch, {"commit/sampled_replay_p0.json": _sampled_replay(3, "PASS")},
                 {"commit/sampled_replay_p0.json": _sampled_replay(11, "PASS")})
    monkeypatch.setattr(resolver, "rows", lambda: [row])
    out = tmp_path / "partition.json"
    assert RP.main([KEY, "--out", str(out)]) == 1
    wrote, said = json.loads(out.read_text()), json.loads(capsys.readouterr().out)
    assert wrote["row"] == KEY and wrote["equal"] is False and [d.split(":")[0] for d in wrote["differences"]] == ["$.sample.seed"]
    assert said["n_differences"] == 1 and said["source"] == "candidate" and said["population"] == 12 and said["picked"] == 6
    _write(tmp_path / "candidate" / KEY, "commit/sampled_replay_p0.json", _sampled_replay(3, "PASS"))
    assert RP.main(["900"]) == 0 and json.loads(capsys.readouterr().out)["equal"] is True


def _workload_program(steps: int) -> tuple[dict, dict]:
    wp = {"inputs": {"derived": {"27,100": "/pod/row/build_request_LP27_T100"}},
          "requests": [{"request_id": "r0", "prompt_len": 27, "cap": 101,
                        "component": {"program_digest": "c", "n_nodes": 5 * steps, "steps": steps, "step_nodes": [[5 * i, 5 * i + 5] for i in range(steps)]}}]}
    chron = {"component_steps_equal_cap": steps == 101, "engine_steps": [{"members": [{"nodes": [0, 5]}, {"nodes": None}]}]}
    return wp, chron


def test_the_segmentation_check_lifts_the_candidates_workload_program_and_its_own_histograms(tmp_path, monkeypatch):
    wp, chron = _workload_program(101)
    old_wp, old_chron = _workload_program(1)
    reference = {"build_workload/workload_program.json": old_wp, "build_workload/chronology.json": old_chron,
                 "build_request_LP27_T100/result.json": {"spec_histogram": {"TokenSelect_v1": 1}}}
    candidate = {"build_workload/workload_program.json": wp, "build_workload/chronology.json": chron,
                 "build_request_LP27_T100/result.json": {"spec_histogram": {"fold:TokenSelect_v1{K=1}": 101, "Gemm_v1": 7}}}
    row = _trees(tmp_path, monkeypatch, reference, candidate)
    act = SS.actual(row)
    assert act["source"] == "candidate" and act["component_steps_equal_cap"] is True and act["unsegmented_members"] == 1
    assert act["requests"]["r0"]["steps"] == 101 and act["requests"]["r0"]["component"] == "build_request_LP27_T100"
    assert act["requests"]["r0"]["sampling_events"] == 101, "the candidate's histogram: the reference's says 1"
    assert act["consistent"] == {"r0": {"steps_eq_declared": True, "steps_eq_sampling_events": True}}
    assert SS.expected(row)["requests"]["r0"]["steps"] == 1
    (tmp_path / "candidate" / KEY / "build_request_LP27_T100" / "result.json").unlink()
    assert "sampling_events" not in SS.actual(row)["requests"]["r0"], "no histogram on the candidate: no count, never the reference's"
    (tmp_path / "candidate" / KEY / "build_workload" / "chronology.json").unlink()
    act = SS.actual(row)
    assert act.pop("source") == "candidate" and set(act) == {"requests", "component_steps_equal_cap", "unsegmented_members"}
    assert all(resolver.is_not_applicable(v) for v in act.values())


_COMMIT_LOG = "$ verity-vllm commit --window-mb 64 --retain host --engine-arg max_tokens=8 --max-num-seqs 4 --pairs 1 --required-manifest m.json\nlater lines\n"
_RUNS = "\n".join(json.dumps(r) for r in ({"arm": "control", "committer": "control", "engine_steps": 9, "generated_tokens": 8},
                                           {"arm": "instrumented", "committer": "native_collect", "tokens_equal": True, "engine_steps": 9,
                                            "generated_tokens": 8})) + "\n"


def test_the_commit_summary_check_reads_a_candidates_summary_or_rereads_its_evidence(tmp_path, monkeypatch):
    reference = {"commit_summary.json": {"rc": 0, "n_runs": 6, "committer": "old", "pass": True}}
    evidence = {"commit/summary.json": {"committer": "native_collect"}, "commit/runs.jsonl": _RUNS, "commit.log": _COMMIT_LOG,
                "verdict.json": {"outcome": "FAIL", "compat": {"rc": 3}}, "commit/verdict.json": {"commit_pass": False, "runs_failed": 1}}
    row = _trees(tmp_path, monkeypatch, reference, evidence)
    act = CS.actual(row)                                    # a candidate written after the flip: no summary file, the reader over its evidence
    assert act["source"] == "candidate" and (act["rc"], act["n_runs"], act["n_control"], act["n_treatment"]) == (3, 2, 1, 1)
    assert act["committer"] == "native_collect" and act["collector_variant"] == "windowed" and act["retention"] == "host" and act["pass"] is False
    assert act["engine_settings"] == {"max_tokens": 8, "max_num_seqs": 4, "pairs": 1, "open_after_release": False, "required_manifest": True}
    assert act["tokens_equal"] == [True] and act["commit_verdict"] == {"commit_pass": False, "runs_failed": 1}
    _write(tmp_path / "candidate" / KEY, "commit_summary.json", {"rc": 0, "n_runs": 2, "committer": "written", "pass": True})
    act = CS.actual(row)                                    # a candidate that wrote the summary: the file is read, the evidence is not re-read
    assert act["source"] == "candidate" and (act["rc"], act["n_runs"], act["committer"], act["pass"]) == (0, 2, "written", True)
    assert act["commit_verdict"] == {"commit_pass": False, "runs_failed": 1} and CS.expected(row)["committer"] == "old"
