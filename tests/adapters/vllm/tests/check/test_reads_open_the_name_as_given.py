"""A read of a file by a name opens that name as it is given, with the builtin `open`: a file's name with a slash after it is no
file, the empty name is no file, and a name in bytes is read.  One test a read.  Where a value of those forms can reach the read the
test hands it one; where the name is a fixed one joined to a directory no value tells, and the test holds the call of `open`."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.check.test_oracle_compare import make_match
from tests.check.test_verdict import _base, _row_dir
from tests.check.test_weightsource_mla import _tensors, _write
from verity_vllm.check import checkpoint_index as CI
from verity_vllm.check import executed_prefix as EPL
from verity_vllm.check import match_oracle as MO
from verity_vllm.check import verdict as V
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli


@pytest.fixture
def opened(monkeypatch) -> list[tuple[Any, str]]:
    """Each call of the builtin `open` during the test: the name as it was handed in, and the mode."""
    calls: list[tuple[Any, str]] = []
    real = builtins.open

    def spy(file, mode="r", *args, **kwargs):
        calls.append((file, mode))
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    return calls


def test_a_checkpoint_index_opens_its_config_by_the_joined_name(tmp_path, opened):
    _write(tmp_path / "model.safetensors", _tensors())
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    index = CI.CheckpointIndex(str(tmp_path), verify_shards=False)
    assert index.config == {"model_type": "llama"}
    assert (os.path.join(str(tmp_path), "config.json"), "r") in opened


def test_the_match_account_opens_its_two_files_by_their_joined_names(tmp_path, opened):
    match_dir = tmp_path / "attempt" / "match"
    (match_dir / "capture").mkdir(parents=True)
    (match_dir / "capture" / "tokens.json").write_text('{"requests": []}')
    (tmp_path / "attempt" / "global_match.json").write_text("{}")
    account = EPL.load_match_account(str(match_dir))
    assert account["tokens"] == {"requests": []}
    assert (os.path.join(str(match_dir), "capture", "tokens.json"), "rb") in opened
    assert (os.path.join(str(tmp_path / "attempt"), "global_match.json"), "rb") in opened


def test_a_verdict_opens_a_wrappers_result_by_the_name_the_glob_found(tmp_path, opened):
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    step = "e6" * 32
    (d / "build_step").mkdir()
    (d / "build_step" / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": "p" * 64, "correspondence_digest": step}}))
    v = V.from_record(d, rc=0)
    assert v.correspondence_digests == {"step": step}
    assert (str(d / "build_step" / "result.json"), "r") in opened


@pytest.mark.parametrize("as_bytes", [False, True])
def test_the_of_record_set_reads_the_manifest_it_is_handed_in_text_or_in_bytes(tmp_path, as_bytes):
    """The manifest's Program digest joins the set, under the name as a string and under the same name in bytes."""
    manifest = tmp_path / "named.json"
    manifest.write_text(json.dumps({"program_digest": "d" * 128}))
    name: Any = os.fsencode(manifest) if as_bytes else str(manifest)
    record = WR.stamp_of_record_set({"fields": []}, str(tmp_path), manifest=name)
    assert record["of_record_set"]["manifest_digests"] == ["d" * 128]
    assert record["of_record_set"]["why"] == [f"no component Program under {tmp_path}"]


@pytest.mark.parametrize(("name", "raised"), [("{a}/", NotADirectoryError), ("{a}/.", NotADirectoryError), ("", FileNotFoundError)])
def test_a_diff_of_two_records_opens_each_name_as_the_command_line_gives_it(tmp_path, monkeypatch, name: str, raised: type[Exception]):
    """A record's name with a slash after it is no directory, and the empty name is no file: the command ends on the error and reads
    neither as a record.  The command runs from `tmp_path`: the suite's read guard joins a name to the working directory, and the
    empty name joined to the suite's own directory is a path outside its inputs."""
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p in (a, b):
        p.write_text("{}")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(raised):
        cli.main(["weights-of-record", "--diff", name.format(a=a), str(b)])
    with pytest.raises(raised):
        cli.main(["weights-of-record", "--diff", str(a), name.format(a=b)])


def test_the_oracles_tie_facts_open_the_two_files_by_their_joined_names(tmp_path, opened):
    d = make_match(str(tmp_path))[0]
    Path(d, "fold_summary.json").write_text("{}")
    facts = MO.oracle_tie_facts(d)
    assert facts["fold_summary_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert facts["instances_sha256"] == hashlib.sha256(Path(d, "instances.jsonl").read_bytes()).hexdigest()
    assert (os.path.join(d, "fold_summary.json"), "rb") in opened
    assert (os.path.join(d, "instances.jsonl"), "rb") in opened


def test_an_oracle_opens_the_snapshot_index_by_its_joined_name(tmp_path, opened):
    d = make_match(str(tmp_path))[0]
    assert MO.MatchOracle(d).snapshot_steps == {0}
    assert (os.path.join(d, "capture", "snapshots.json"), "r") in opened


def test_an_oracle_opens_the_tokens_by_their_joined_name_for_their_digest(tmp_path, opened):
    d = make_match(str(tmp_path))[0]
    tokens = os.path.join(d, "capture", "tokens.json")
    assert MO.MatchOracle(d).tokens_sha256 == hashlib.sha256(Path(tokens).read_bytes()).hexdigest()
    assert (tokens, "rb") in opened


def test_an_oracle_opens_the_tokens_by_their_joined_name_for_their_ids(tmp_path, opened):
    d = make_match(str(tmp_path))[0]
    tokens = os.path.join(d, "capture", "tokens.json")
    oracle = MO.MatchOracle(d)
    assert set(oracle.tokens) == {str(r["request_id"]) for r in json.loads(Path(tokens).read_text())["requests"]}
    assert (tokens, "r") in opened


@pytest.mark.parametrize("after", ["/", "/."])
def test_a_snapshot_entry_whose_file_has_a_slash_after_it_names_no_file(tmp_path, after: str):
    """An entry of the snapshot index names its file under `capture/`.  The oracle reads that file's bytes; the same name with a slash
    after it is no directory, and the read ends on the error."""
    d = make_match(str(tmp_path))[0]
    entry = json.loads(Path(d, "capture", "snapshots.json").read_text())["entries"][0]
    oracle = MO.MatchOracle(d)
    assert hashlib.sha256(oracle._file(entry["file"])).hexdigest() == entry["sha256"]
    with pytest.raises(NotADirectoryError):
        oracle._file(entry["file"] + after)
