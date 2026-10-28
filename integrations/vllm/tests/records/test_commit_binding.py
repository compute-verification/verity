"""`records.commit_binding`: the Commit block's `binding` as each writer writes it (`pipeline.commit.binding_record` and its error-only
form, a TP rank's `pipeline.tp.rank_worker`, `pipeline.serve_seal`) loads as the typed record, its sub-records built by their own writers
here (`commit.binding.coverage_check`, `commit.executed_prefix`, `check.executed_prefix.link_to_match_account`); a missing or mistyped
field is refused by its dotted path, and through `records.commit_run` by the file and line too."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.check import executed_prefix as EPL
from verity_vllm.commit import binding as B
from verity_vllm.commit import executed_prefix as EP
from verity_vllm.records.commit_binding import CommitBinding
from verity_vllm.records.commit_run import SCHEMA, CommitRun
from verity_vllm.records.fields import MissingField, RecordError

ARRIVALS = {"r0": 0, "r1": 0}
EOS = 9


def _match_link(epr: dict) -> dict:
    """The Match-account link with both Match records beside the Commit, as `link_to_match_account` writes them."""
    link = EPL.link_to_match_account(epr, EPL.load_match_account(None), order={"w0": "r0", "w1": "r1"}, declared_lag=1)
    return link | {"ok": True, "problems": [], "lag_attested": True, "match_dir": "/x/match",
                   "tokens_json": {"path": "/x/match/capture/tokens.json", "sha256": "ef" * 32, "lag": 1, "tokens_equal": {"r0": True, "r1": True},
                                   "served_by_request": {"r0": 2, "r1": 3}, "executed_by_request": {"r0": 3, "r1": 3}},
                   "global_match": {"path": "/x/global_match.json", "sha256": "9a" * 32, "verdict": "PASS", "failed_checks": [], "g6_status": "PASS",
                                    "lag": {"declared": 1, "record_attests": 1}, "requests_linked": ["r0", "r1"],
                                    "lengths_by_request": {"r0": {"served_output_len": 2, "executed_prefix_len": 3, "cap": 3, "cap_exhausted": False},
                                                           "r1": {"served_output_len": 3, "executed_prefix_len": 3, "cap": 3, "cap_exhausted": True}}}}


def _binding() -> dict:
    """A single-GPU binding as `binding_record` assembles it: r0 stopped at EOS with LAG 1, r1 ran its cap; the token-record link was
    not applied (the workload committed no sampled id)."""
    epr = EP.executed_prefix_of_record(["r0", "r1"], {"r0": [5, EOS], "r1": [1, 2, 3]}, caps={"r0": 3, "r1": 3}, eos_of_record=[EOS], lag=1)
    cov = B.coverage_check([], {"digest": "d" * 64, "entries_n": 0, "entries": []})
    cov |= {"ok": True, "expected_n": 12, "bound_n": 12, "missing_n": 0,
            "executed_prefix_of_record": {k: v for k, v in epr.items() if k != "rule"} | {"rule": EP.RULE},
            "rows_past_the_stop": EP.rows_past_the_stop({"entries": []}, EP.request_steps_of_record(epr, ARRIVALS)),
            "match_account_link": _match_link(epr),
            "commit_account_link": {"rule": "r", "required": False, "reduced_requests": ["r0"], "record": None, "lag_attested": None, "problems": [],
                                    "ok": None, "not_applied": "no committed runner.sampler/sampled_token_ids in the binding map"}}
    return {"rules_version": B.RULES_VERSION, "program_digest": "p" * 64, "map_digest": "d" * 64, "entries_n": 0,
            "collector": {"variant": "native", "retention": "host"},
            "population": {"source": "required-manifest", "digest": "q" * 64, "n": 12, "requests": 2, "steps": 3, "ranks": 1},
            "coverage": cov, "context": {"ok": True, "why": None}, "openings_population": {"n": 4, "opened": 4, "verified": 4}}


def test_a_single_gpu_binding_loads_as_its_writers_wrote_it():
    doc = _binding()
    b = CommitBinding.of(doc, "runs.jsonl:2")
    assert (b.error, b.context_ok, b.program_digest, b.collector) == (None, True, "p" * 64, {"variant": "native", "retention": "host"})
    assert (b.population.requests, b.population.request_ids, b.population.steps) == (2, None, 3)
    cov = b.coverage
    assert (cov.ok, cov.expected_n, cov.bound_n, cov.missing_n, cov.ambiguous_n, cov.missing) == (True, 12, 12, 0, 0, ())
    epr = cov.executed_prefix_of_record
    assert epr.ok and epr.problems == () and epr.lag == 1 and epr.eos_of_record == (EOS,) and epr.any_eos_stop is True
    assert (epr.requests["r0"].served_output_len, epr.requests["r0"].executed_prefix_len, epr.requests["r0"].cap) == (2, 3, 3)
    assert epr.requests["r1"].eos_stop is False and epr.requests["r1"].last_served_token == 3
    link = cov.match_account_link
    assert link.ok and link.required and link.lag_attested and link.match_dir == "/x/match"
    assert link.tokens_json.served_by_request == {"r0": 2, "r1": 3} and link.tokens_json.tokens_equal["r1"] is True
    assert link.global_match.verdict == "PASS" and link.global_match.requests_linked == ("r0", "r1")
    assert link.global_match.served_by_request == {"r0": 2, "r1": 3}
    cl = cov.commit_account_link
    assert cl.ok is None and cl.required is False and cl.not_applied and cl.record_path is None and cl.tokens_committed is None
    past = cov.rows_past_the_stop
    assert past.ok and past.entries_n == 0 and past.steps_past_by_request == {} and past.padded_record is False
    assert b.entry is doc and cov.entry is doc["coverage"] and CommitBinding.of(b, "elsewhere") is b


def test_a_tp_rank_a_failed_binding_and_a_sealed_window_load():
    cov = B.coverage_check([], {"digest": "d" * 64, "entries_n": 0, "entries": []})
    rank = {"rules_version": B.RULES_VERSION, "rank": 1, "program_digest": "p" * 64, "map_digest": "d" * 64, "entries_n": 0,
            "population": {"source": "structure x workload", "digest": "q" * 64, "n": 0, "steps": 3, "requests": ["r0", "r1"], "ranks": [1]},
            "coverage": cov, "context": {"ok": True, "why": None}}
    r = CommitBinding.of(rank, "rank 1")
    assert r.population.requests is None and r.population.request_ids == ("r0", "r1")
    assert r.coverage.ok is False and r.coverage.executed_prefix_of_record is None and r.coverage.match_account_link is None
    skipped = CommitBinding.of({"rank": 0, "skipped": "no binding map for this run"}, "rank 0")
    assert skipped.coverage is None and skipped.context_ok is None and skipped.error is None
    failed = CommitBinding.of({"error": "KeyError('r0')", "site": "binding_record", "rules_version": "r14-v1"}, "runs.jsonl:2")
    assert failed.error == "KeyError('r0')" and failed.coverage is None and failed.population is None and failed.program_digest is None
    sealed = CommitBinding.of({"map_digest": "d" * 64, "map_entries_n": 3, "program_digest": "p" * 64}, "sealed")
    assert sealed.program_digest == "p" * 64 and sealed.context_ok is None and sealed.coverage is None


@pytest.mark.parametrize("edit, error, field", [
    (lambda d: d["coverage"].pop("ok"), MissingField, "commit.binding.coverage.ok"),
    (lambda d: d["coverage"].update(expected_n="12"), RecordError, "commit.binding.coverage.expected_n"),
    (lambda d: d["coverage"].update(missing=["step 1"]), RecordError, "commit.binding.coverage.missing[0]"),
    (lambda d: d["coverage"]["executed_prefix_of_record"].pop("problems"), MissingField, "commit.binding.coverage.executed_prefix_of_record.problems"),
    (lambda d: d["coverage"]["executed_prefix_of_record"]["requests"]["r0"].pop("cap"), MissingField,
     "commit.binding.coverage.executed_prefix_of_record.requests.r0.cap"),
    (lambda d: d["coverage"]["match_account_link"]["tokens_json"]["served_by_request"].update(r0="2"), RecordError,
     "commit.binding.coverage.match_account_link.tokens_json.served_by_request.r0"),
    (lambda d: d["coverage"]["match_account_link"]["global_match"].pop("verdict"), MissingField,
     "commit.binding.coverage.match_account_link.global_match.verdict"),
    (lambda d: d["coverage"]["commit_account_link"].update(tokens_committed={"r0": 1}), RecordError,
     "commit.binding.coverage.commit_account_link.tokens_committed.r0"),
    (lambda d: d["coverage"]["rows_past_the_stop"].pop("steps_past_by_request"), MissingField,
     "commit.binding.coverage.rows_past_the_stop.steps_past_by_request"),
    (lambda d: d["population"].update(requests="2"), RecordError, "commit.binding.population.requests"),
    (lambda d: d["context"].update(ok="yes"), RecordError, "commit.binding.context.ok"),
])
def test_a_binding_is_refused_by_field(edit, error, field):
    doc = copy.deepcopy(_binding())
    edit(doc)
    with pytest.raises(error) as e:
        CommitBinding.of(doc, "runs.jsonl:2")
    assert e.value.field == field and e.value.where == "runs.jsonl:2"


def test_a_run_whose_binding_is_refused_names_its_line(tmp_path):
    row = {"schema": SCHEMA, "pair": 0, "arm": "instrumented", "committer": "hm96", "commit": {"run_root": "ee" * 32, "binding": _binding()}}
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(row) + "\n")
    (run,) = CommitRun.load_lines(p)
    assert run.commit.binding.coverage.executed_prefix_of_record.requests["r0"].executed_prefix_len == 3
    assert (run.commit.binding_program_digest, run.commit.collector) == ("p" * 64, {"variant": "native", "retention": "host"})
    del row["commit"]["binding"]["coverage"]["match_account_link"]["required"]
    p.write_text(json.dumps(row) + "\n")
    with pytest.raises(MissingField) as e:
        CommitRun.load_lines(p)
    assert e.value.where == f"{p}:1" and e.value.field == "commit.binding.coverage.match_account_link.required"
