"""What the Match does with a value that only a command line or a caller hands it, and that no loader of the package produces:
`global-match --log` and `--tokens` and `batch-decomp --log` name the capture log and the tokens file, and each `main` hands
them to its checker; `verity_vllm.check.match.rank_match.own_producers_from_record` takes a second argument and does not read
it; and `verity_vllm.check.match.per_request._compare_component`, given a component's directory and no loaded component, fails
where it reads the component's digest."""
from __future__ import annotations

import json

import pytest

from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import global_match as gm
from verity_vllm.check.match import per_request, rank_match
from verity_vllm.pipeline import cli


@pytest.fixture(autouse=True)
def _no_tuning_of_pytests_collector(monkeypatch):
    monkeypatch.setattr("verity_vllm.pipeline.gc_tuning.tune", lambda: None)


class _Reached(Exception):
    """Raised by a stand-in once it has what the test reads."""


def test_both_commands_take_the_capture_log_and_global_match_the_tokens_file():
    a = cli.parse("global-match", ["--oracle", "O", "--log", "L.jsonl.gz", "--tokens", "T.json"])
    assert (a.log, a.tokens, a.tokens_optional) == ("L.jsonl.gz", "T.json", False)
    b = cli.parse("batch-decomp", ["--oracle", "O", "--out", "D.json", "--log", "L.jsonl.gz"])
    assert b.log == "L.jsonl.gz"


def test_neither_path_is_named_unless_the_command_line_names_it():
    a = cli.parse("global-match", ["--oracle", "O"])
    assert (a.log, a.tokens) == (None, None)
    assert cli.parse("batch-decomp", ["--oracle", "O", "--out", "D.json"]).log is None


def test_tokens_without_a_file_is_a_usage_error_and_not_the_switch_that_begins_with_it(capsys):
    with pytest.raises(SystemExit) as e:
        cli.parse("global-match", ["--oracle", "O", "--tokens"])
    assert e.value.code == 2 and "argument --tokens: expected one argument" in capsys.readouterr().err


def test_global_match_hands_the_log_and_the_tokens_file_to_its_checker(tmp_path, monkeypatch):
    gpp = tmp_path / "gp.json"
    gpp.write_text(json.dumps({"schema": gm.GP_SCHEMA, "requests": []}))
    monkeypatch.setenv(gm.IMPL_ENV, "record")
    seen = []

    def check(gp, oracle_dir, log_path=None, tokens_path=None, **kw):
        seen.append((oracle_dir, log_path, tokens_path))
        raise _Reached

    monkeypatch.setattr(gm, "check", check)
    for named in ({"log": "L.jsonl.gz", "tokens": "T.json"}, {}):
        with pytest.raises(_Reached):
            gm.main(gm.Options(oracle="O", global_program=str(gpp), **named))
    assert seen == [("O", "L.jsonl.gz", "T.json"), ("O", None, None)]


def test_batch_decomp_hands_the_log_to_the_decomposition(tmp_path, monkeypatch):
    seen = []

    def decompose(oracle_dir, derived_dirs, log_path=None, **kw):
        seen.append((oracle_dir, list(derived_dirs), log_path, kw))
        raise _Reached

    monkeypatch.setattr(bd, "decompose", decompose)
    for named in ({"log": "L.jsonl.gz"}, {}):
        with pytest.raises(_Reached):
            bd.main(bd.Options(oracle="O", out=str(tmp_path / "d.json"), **named))
    kw = {"within_step": bd.DEFAULT_WITHIN_STEP}
    assert seen == [("O", [], "L.jsonl.gz", kw), ("O", [], None, kw)]


def test_the_producers_of_a_record_take_a_second_argument_they_do_not_read(tmp_path):
    path = tmp_path / "instances.jsonl"
    path.write_text("")
    assert rank_match.own_producers_from_record(path, None) == rank_match.own_producers_from_record(path) == {}
    assert rank_match.own_producers_from_record(path, {0: "Gemm_v1{N=2}"}) == {}
    assert rank_match.own_producers_from_record(path, program_nodes_spec=None) == {}


def test_a_component_directory_with_no_loaded_component_fails_where_its_digest_is_read():
    out: dict[str, list] = {"fails": []}
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'digest'"):
        per_request._compare_component(None, "dag", False, {"request_id": "r0"}, {}, None, None, "a-dir", 3, out)
    assert out == {"fails": []}
    assert per_request._compare_component(None, "dag", False, {"request_id": "r0", "T_bound": 4}, {}, None, None, None, 3, out) is out
    assert out["fails"] == [(per_request.SUBCIRCUITS, "r0: no component Program in the global Program for shape (LP 3, T_bound 4)")]
