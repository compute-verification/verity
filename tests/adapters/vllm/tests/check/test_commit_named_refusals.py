"""What the Commit verdict refuses by name: a count of a run's Commit block or of its coverage declaration that is no number
(`verity_vllm.check.commit_rules`, where `commit_pass.commit_verdict` raises on it unnamed), and an `openings` setting or an exit code
that is no number (`verity_vllm.check.verdict.build`)."""
from __future__ import annotations

from typing import Any

import pytest

from tests.check.test_absent_commit_fields import put
from tests.check.test_verdict import CLASSES, _base, _both, _decl
from verity_vllm.check import verdict as V
from verity_vllm.check.commit_pass import commit_verdict
from verity_vllm.check.commit_rules import LEAF_HOWS
from verity_vllm.records.fields import RecordError

BLOCK = "the run's Commit block"
#: the counts of a run's Commit block that v1 and the rules both read, each by the name its refusal gives
COUNTS = ["extra.expected_tensors", "extra.committed_tensors", "extra.errors_n", "fa2_zero_rows", "openings.n", "openings.verified", "openings.deferred_replay_n",
          "openings_after_release.n", "openings_after_release.verified", "openings_after_release.replay_errors_n", "openings_after_release.retained_device_after",
          "openings_after_release.deferred_received_n", "binding.openings_population.unbound_n", "binding.openings_population.n", "binding.openings_population.opened",
          "binding.openings_population.verified", "binding.openings_population.deferred_replay_n"]
#: the three v1 reads as they stand; it reads every other count by its truth, so that a value that tests false there is a count of zero
READ_AS_STATED = ("extra.expected_tensors", "extra.committed_tensors", "extra.errors_n")
#: numbers as `int` reads them, and what is none
NUMBERS: list[Any] = [0, False, 1, True, 64, "64", " 64 ", 64.9, 572]
NOT_NUMBERS: list[Any] = ["many", "2.5", [3], {"n": 3}, float("nan"), float("inf")]
NOTHING: list[Any] = ["", [], {}]


def _counted() -> dict:
    """A run that passes and on which v1 and the rules read every count: openings sampled from the population, one pick of the challenge's
    deferred to replay and received by count, and an after-release record of a replay."""
    commit = _base()
    commit["openings"]["deferred_replay_n"] = 1
    commit["binding"]["openings_population"] = {"n": 64, "opened": 64, "verified": 64, "unbound_n": 0, "deferred_replay_n": 0}
    commit["openings_after_release"] = {"n": 64, "verified": 64, "source": "replay", "replay_root_equal": True, "replay_errors_n": 0, "retained_device_after": 0,
                                        "deferred_received_n": 1}
    return commit


def _run(**commit: Any) -> dict:
    return {"arm": "instrumented", "pair": 0, "tokens_equal_ref": True, "commit": dict(_base(), commit_pass=True, **commit)}


DOC = {"commit_pass": True, "required_classes": list(CLASSES)}


def test_the_run_on_which_every_count_is_read_passes_in_both():
    (ok, why), out, _ = _both(_counted())
    assert ok is True and out == V.PASS, why


@pytest.mark.parametrize("field", COUNTS)
def test_the_rules_refuse_a_count_by_name_exactly_where_v1_raises_and_decide_as_v1_elsewhere(field):
    refused = []
    for stated in NUMBERS + NOT_NUMBERS + NOTHING:
        commit = put(_counted(), tuple(field.split(".")), stated)
        decl = _decl(commit["classes"])
        try:
            ok, _ = commit_verdict(commit, True, declaration=decl)
        except (TypeError, ValueError, OverflowError) as bare:
            assert not isinstance(bare, RecordError)
            with pytest.raises(RecordError) as e:
                V.run_checks(commit, True, declaration=decl, ctx={})
            assert (e.value.where, e.value.field, e.value.why) == (BLOCK, field, f"is {stated!r}, not a number")
            refused.append(stated)
        else:
            assert (V.run_outcome(V.run_checks(commit, True, declaration=decl, ctx={})) == V.PASS) is ok, stated
    assert list(map(repr, refused)) == list(map(repr, NOT_NUMBERS + (NOTHING if field in READ_AS_STATED else [])))


@pytest.mark.parametrize("cls", [c for c, d in CLASSES.items() if d["how"] in LEAF_HOWS])
@pytest.mark.parametrize("stated", NOT_NUMBERS + NOTHING)
def test_a_declared_class_whose_tensors_is_no_number_is_refused_by_name_by_v1_and_by_the_rules(cls, stated):
    """The one read of the declaration, `required_classes_check`, is v1's and the rules'."""
    decl = _decl({**CLASSES, cls: dict(CLASSES[cls], tensors=stated)})
    for decide in (lambda: commit_verdict(_base(), True, declaration=decl), lambda: V.run_checks(_base(), True, declaration=decl, ctx={})):
        with pytest.raises(RecordError) as e:
            decide()
        assert (e.value.where, e.value.field, e.value.why) == ("the run's coverage declaration", f"classes[{cls!r}].tensors", f"is {stated!r}, not a number")


def test_the_rules_refuse_by_name_a_count_v1_does_not_read():
    """v1 stops at its first failing check and the rules grade each rule, so each rule reads its count: `fa2_zero_rows` on a run whose
    collector failed, and a declared class's `tensors` after a class that bound nothing.  `errors_n` beside `extra` is the rules' read
    alone, on a record whose `extra` states neither the count nor the errors."""
    commit = put(put(_base(), ("extra", "collector_failure"), "step 3: abandoned"), ("fa2_zero_rows",), "many")
    decl = _decl(commit["classes"])
    assert commit_verdict(commit, True, declaration=decl) == (False, "collector failed: step 3: abandoned")
    with pytest.raises(RecordError) as e:
        V.run_checks(commit, True, declaration=decl, ctx={})
    assert (e.value.where, e.value.field, e.value.why) == (BLOCK, "fa2_zero_rows", "is 'many', not a number")
    first, later = [c for c, d in CLASSES.items() if d["how"] in LEAF_HOWS][:2]
    decl = _decl({**CLASSES, first: dict(CLASSES[first], tensors=0), later: dict(CLASSES[later], tensors="many")})
    assert commit_verdict(_base(), True, declaration=decl)[0] is False
    with pytest.raises(RecordError) as e:
        V.run_checks(_base(), True, declaration=decl, ctx={})
    assert (e.value.where, e.value.field, e.value.why) == ("the run's coverage declaration", f"classes[{later!r}].tensors", "is 'many', not a number")
    commit = _base()
    del commit["extra"]["errors_n"]
    commit["errors_n"] = "many"
    decl = _decl(commit["classes"])
    assert commit_verdict(commit, True, declaration=decl)[0] is True
    with pytest.raises(RecordError) as e:
        V.run_checks(commit, True, declaration=decl, ctx={})
    assert (e.value.where, e.value.field, e.value.why) == (BLOCK, "errors_n", "is 'many', not a number")


def test_where_the_populations_sample_was_skipped_both_read_the_host_share_and_neither_reads_n():
    """A skipped sample opens the host's share here and leaves the rest to replay: v1 and the rules read `opened` and `verified` there,
    and no `n`."""
    commit = _counted()
    commit["binding"]["openings_population"].update(skipped="excluded families", n="many", opened=8, verified=8)
    (ok, why), out, _ = _both(commit)
    assert ok is True and out == V.PASS, why
    commit["binding"]["openings_population"]["verified"] = 7
    (ok, why), out, checks = _both(commit)
    assert (ok, why, out) == (False, "population openings (host share) verified 7 != opened 8", V.FAIL)
    assert [(c.name, c.reason, c.differences) for c in checks if c.outcome == V.FAIL] == [
        ("runtime_match/openings_population", why, [{"name": "population_openings_unverified", "opened": 8, "verified": 7, "share": "host"}])]
    commit["binding"]["openings_population"]["verified"] = "many"
    decl = _decl(commit["classes"])
    with pytest.raises(ValueError, match="many") as bare:
        commit_verdict(commit, True, declaration=decl)
    assert not isinstance(bare.value, RecordError)
    with pytest.raises(RecordError) as e:
        V.run_checks(commit, True, declaration=decl, ctx={})
    assert (e.value.where, e.value.field, e.value.why) == (BLOCK, "binding.openings_population.verified", "is 'many', not a number")


@pytest.mark.parametrize("no_number", ["sixty-four", "", [], {}, [8], "8.5", float("nan")])
def test_an_openings_setting_that_is_no_number_is_refused_by_name(no_number):
    with pytest.raises(RecordError) as e:
        V.build([_run(openings={"n": 8, "verified": 8})], DOC, execution={"engine_settings": {"openings": no_number}})
    assert (e.value.where, e.value.field, e.value.why) == ("the Commit's execution configuration", "engine_settings.openings", f"is {no_number!r}, not a number")


@pytest.mark.parametrize("stated,asked", [(None, 64), (8, 8), ("8", 8), (0, 0), (False, 0), (512, 64), (8.9, 8)])
def test_an_openings_setting_that_is_a_number_asks_for_it_up_to_64_and_none_asks_for_64(stated, asked):
    by = {c.name: c for c in V.build([_run(openings={"n": 8, "verified": 8})], DOC, execution={"engine_settings": {"openings": stated}}).checks}
    assert by["runtime_match/openings"].scope["min_openings"] == asked


@pytest.mark.parametrize("rc,summary,where", [("one", None, "the caller of build"), ([], None, "the caller of build"), (None, {"rc": "one"}, "commit_summary.json"),
                                              (None, {"rc": ""}, "commit_summary.json")])
def test_an_exit_code_that_is_no_number_is_refused_by_name(rc, summary, where):
    with pytest.raises(RecordError) as e:
        V.build([_run()], DOC, rc=rc, summary=summary)
    stated = rc if rc is not None else summary["rc"]
    assert (e.value.where, e.value.field, e.value.why) == (where, "rc", f"is {stated!r}, not a number")


@pytest.mark.parametrize("rc,outcome", [(None, V.PASS), (0, V.PASS), ("0", V.PASS), (False, V.PASS), (1, V.INSUFFICIENT), ("1", V.INSUFFICIENT), (True, V.INSUFFICIENT),
                                        (-9, V.INSUFFICIENT)])
def test_an_exit_code_that_is_a_number_closes_the_stage_at_zero_alone(rc, outcome):
    assert V.build([_run()], DOC, rc=rc).outcome == outcome
    assert V.build([_run()], DOC, summary={"rc": rc}).outcome == outcome
