"""The Commit's ``--commit-scheme vllm-v1`` only commits again the recorded vllm-v1 Commit ``--of-record`` names
(`pipeline.commit.scheme_refusal`, serve-openai's rule), and is refused by name otherwise, before any engine."""
from __future__ import annotations

import json
from types import SimpleNamespace

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import commit as CD

MODEL = "meta-llama/Llama-3.2-1B"


def recorded(tmp_path, root_bytes, model=MODEL, name="rec"):
    d = tmp_path / name
    d.mkdir()
    runs = [{"pair": 0, "arm": "control", "model": model},
            {"pair": 0, "arm": "instrumented", "model": model, "commit": {"run_root": "ab" * root_bytes}}]
    (d / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs))
    return str(d)


def refusal(scheme, of_record=None, model=MODEL):
    return CD.scheme_refusal(SimpleNamespace(commit_scheme=scheme, of_record=of_record), {"model": model})


def test_rows_sha512_is_the_default_and_needs_no_record():
    args = cli.parse("commit", ["--workload", "w", "--out", "o"], {})
    assert (args.commit_scheme, args.of_record) == ("rows-sha512", None)
    assert CD.scheme_refusal(args, {"model": MODEL}) is None


def test_vllm_v1_without_a_record_is_refused_by_name(tmp_path):
    why = refusal("vllm-v1")
    assert "--commit-scheme vllm-v1 commits SHA-256" in why and "--of-record RUN_DIR" in why and "rows-sha512" in why


def test_vllm_v1_commits_again_only_a_recorded_vllm_v1_commit_of_the_same_model(tmp_path):
    v1 = recorded(tmp_path, 32)
    args = cli.parse("commit", ["--workload", "w", "--out", "o", "--commit-scheme", "vllm-v1", "--of-record", v1], {})
    assert args.of_record == v1 and CD.scheme_refusal(args, {"model": MODEL}) is None
    assert "commits its recorded run's model again" in refusal("vllm-v1", v1, model="Qwen/Qwen2.5-0.5B")
    rows = recorded(tmp_path, 64, name="rows")
    assert f"--of-record {rows} committed rows-sha512 (64-byte run roots), not vllm-v1" in refusal("vllm-v1", rows)


def test_a_record_that_is_not_a_commit_or_a_record_under_rows_sha512_is_refused(tmp_path):
    assert "not a recorded Commit" in refusal("vllm-v1", str(tmp_path))
    (tmp_path / "runs.jsonl").write_text(json.dumps({"pair": 0, "arm": "control", "model": MODEL}) + "\n")
    assert "records no committed run" in refusal("vllm-v1", str(tmp_path))
    v1 = recorded(tmp_path, 32, name="v1")
    assert "not under rows-sha512" in refusal("rows-sha512", v1)
