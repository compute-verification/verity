"""`records.commit_run`: a line of `commit/runs.jsonl` as each writer writes it (`pipeline.commit`, `pipeline.tp.commit`,
`pipeline.serve_seal`, and the older runs before them) loads as the typed Commit run and dumps back to the same JSON; a missing or unknown
field is refused by name.  A sealed window's own line is held to the same in `tests/pipeline/test_serve_seal.py`."""
from __future__ import annotations

import copy
import json

import pytest

from verity_vllm.records.commit_run import SCHEMA, TP_SCHEMA, CommitRun
from verity_vllm.records.fields import MissingField, RecordError, UnknownField


def _row(arm: str = "instrumented") -> dict:
    """The fields `pipeline.commit` writes for one run of a pair (the control arm has no `commit` block)."""
    row = {"schema": SCHEMA, "label": "l", "workload": "w.json", "workload_sha256": "ab" * 32, "case": "c", "model": "m", "pair": 1,
           "order_in_pair": 0, "arm": arm, "committer": None if arm == "control" else "hm96", "committer_name": None if arm == "control" else "hm96",
           "wall_s": 1.5, "iterations": 3, "engine_steps": 3, "generated_tokens": 8, "prompt_tokens": 4, "generated_tokens_per_s": 5.3,
           "total_tokens_per_s": 8.0, "step_wall_s": [0.5, 0.5, 0.5], "step_wall_prefill_s": 0.5, "step_wall_decode_mean_s": 0.5,
           "tokens_equal_ref": True, "tokens_digest": "cd" * 32, "gpu_mem": {"torch_max_allocated_gb": 1.0}, "host_mem_kb": {"hwm": 1},
           "utc": "2026-10-25T00:00:00Z"}
    if arm != "control":
        row["commit"] = {"run_root": "ee" * 32, "steps": 3, "leaves": 9, "extra": {"errors_n": 0}, "openings": {"n": 4, "verified": 4},
                         "classes": {"attn": {"how": "leaf", "tensors": 3, "bytes": 96}}, "rules_version": "r14-v1",
                         "binding": {"rules_version": "r14-v1", "program_digest": "p" * 64, "collector": {"window_mb": 0, "retention": "host"}},
                         "weights_pin": {"program_digest": "q" * 64, "result": True}, "coverage": {"ok": True, "manifest_digest": "m" * 64},
                         "openings_after_release": {"n": 4}, "commit_pass": True, "commit_fail_reason": None}
    return row


def test_a_pair_loads_as_its_writer_wrote_it(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(_row("control")) + "\n\n" + json.dumps(_row()) + "\n")
    ctl, run = CommitRun.load_lines(p)
    assert ctl.arm == "control" and ctl.committer is None and ctl.commit is None and ctl.where == f"{p}:1"
    assert run.where == f"{p}:3" and run.pair == 1 and run.committer == "hm96" and run.tokens_equal is True
    c = run.commit
    assert (c.commit_pass, c.commit_fail_reason, c.run_root, c.rules_version) == (True, None, "ee" * 32, "r14-v1")
    assert (c.binding_program_digest, c.weights_pin_program_digest, c.coverage_manifest_digest) == ("p" * 64, "q" * 64, "m" * 64)
    assert c.collector == {"window_mb": 0, "retention": "host"} and c.openings_after_release == {"n": 4} and c.classes["attn"]["tensors"] == 3
    assert ctl.dump() == _row("control") and run.dump() == _row() and CommitRun.of(run, "elsewhere") is run


def test_a_tp_run_a_sealed_window_and_an_old_run_load():
    tp = {k: v for k, v in _row().items() if k not in ("committer_name", "total_tokens_per_s", "tokens_digest", "gpu_mem", "host_mem_kb")}
    tp |= {"schema": TP_SCHEMA, "tp": 2, "commit_started": "s", "step_tokens_out": [1], "thread_cpu_s": 0.1,
           "commit": {"tp_run_root": "ab", "world": 2, "per_rank": {}, "commit_pass": False, "commit_fail_reason": "fail-closed: 1/2 ranks finalized"}}
    r = CommitRun.of(tp, "tp")
    assert r.commit.run_root is None and r.commit.commit_pass is False and r.commit.binding_program_digest is None
    sealed = {"pair": 0, "arm": "instrumented", "label": "serve", "served_window": {"window": 5}, "tokens_equal_ref": None,
              "commit": {"run_root": "ab", "committer": "hm96", "binding": {"program_digest": "p"}, "commit_pass": None, "commit_fail_reason": None}}
    s = CommitRun.of(sealed, "sealed")
    assert s.schema is None and s.committer is None and s.tokens_equal is None and s.commit.commit_pass is None and s.dump() == sealed
    old = {k: v for k, v in _row().items() if k != "tokens_equal_ref"} | {"tokens_equal": False}
    assert CommitRun.of(old, "old").tokens_equal is False


@pytest.mark.parametrize("edit, error, field", [
    (lambda d: d.pop("pair"), MissingField, "pair"),
    (lambda d: d.pop("arm"), MissingField, "arm"),
    (lambda d: d.update(tokens_equal_reference=True), UnknownField, "tokens_equal_reference"),
    (lambda d: d.update(schema="verity/commit-integ/delta/v2"), RecordError, "schema"),
    (lambda d: d.update(pair="1"), RecordError, "pair"),
    (lambda d: d.update(tokens_equal_ref="yes"), RecordError, "tokens_equal_ref"),
    (lambda d: d.update(commit="PASS"), RecordError, "commit"),
    (lambda d: d["commit"].update(commit_pass=1), RecordError, "commit.commit_pass"),
    (lambda d: d["commit"].update(binding=[]), RecordError, "commit.binding"),
    (lambda d: d["commit"]["binding"].update(collector="windowed"), RecordError, "commit.binding.collector"),
    (lambda d: d["commit"]["coverage"].update(manifest_digest=7), RecordError, "commit.coverage.manifest_digest"),
])
def test_a_run_is_refused_by_field(edit, error, field):
    doc = copy.deepcopy(_row())
    edit(doc)
    with pytest.raises(error) as e:
        CommitRun.of(doc, "runs.jsonl:2")
    assert e.value.field == field and e.value.where == "runs.jsonl:2"


def test_a_line_that_is_not_an_object_is_refused(tmp_path):
    p = tmp_path / "runs.jsonl"
    p.write_text(json.dumps(_row()) + "\n[]\n")
    with pytest.raises(RecordError, match=f"{p}:2"):
        CommitRun.load_lines(p)
