"""`verity_vllm.pipeline.tp.capture` reads what its command line declares: the workload is `--workload`'s, with no default of the
module's; the adapter revision every rank's header carries is `--adapter-revision`, else the exported tree's commit (`EXPORT.json`
in the working directory), else "unknown" where that file is absent, unreadable or not an object; and its log says each request's
tokens beside the workload's accepted ones.  CPU-only: `main` runs over the stand-in LLM of two ranks of
`tests.pipeline.test_tp_target`."""
from __future__ import annotations

import pytest

from tests.pipeline import test_tp_target as TT
from verity_vllm.pipeline import cli

stand_ins, vllm = TT.stand_ins, TT.vllm


def _installed(tmp_path, monkeypatch, stand_ins, *extra: str) -> dict:
    """What `tp-capture`, run in ``tmp_path``, handed every rank's ``tp2_install``."""
    monkeypatch.chdir(tmp_path)
    llm = TT._LLM()
    TT._capture(tmp_path, llm, None, stand_ins, *extra)
    return next(kw for method, kw in llm.rpcs if method == "tp2_install")


def test_tp_capture_without_a_workload_is_a_usage_error_that_names_the_option(tmp_path, capsys):
    with pytest.raises(SystemExit) as refused:
        cli.parse("tp-capture", ["--out", str(tmp_path / "capture")], environ={})
    assert refused.value.code == 2 and "--workload" in capsys.readouterr().err


@pytest.mark.parametrize("export, revision", [
    (b'{"commit": "c0ffee", "patch_sha256": null}', "c0ffee"), (b"{}", "unknown"), (None, "unknown"), (b"{", "unknown"),
    (b"\xff", "unknown"), (b'["c0ffee"]', "unknown"), (b"null", "unknown"),
], ids=["its commit", "no commit", "absent", "not JSON", "not text", "a list", "null"])
def test_the_revision_is_the_exported_trees_commit_and_unknown_without_one(tmp_path, monkeypatch, stand_ins, export, revision):
    if export is not None:
        (tmp_path / "EXPORT.json").write_bytes(export)
    assert _installed(tmp_path, monkeypatch, stand_ins)["adapter_revision"] == revision


def test_an_adapter_revision_on_the_command_line_is_the_one_the_ranks_get(tmp_path, monkeypatch, stand_ins):
    (tmp_path / "EXPORT.json").write_text('{"commit": "c0ffee"}')
    assert _installed(tmp_path, monkeypatch, stand_ins, "--adapter-revision", "beef")["adapter_revision"] == "beef"


def test_the_log_says_each_requests_tokens_beside_the_workloads_accepted_ones(tmp_path, monkeypatch, stand_ins, capsys):
    _installed(tmp_path, monkeypatch, stand_ins)
    assert "[tp2] q0: generated [5, 6]\n[tp2]   accepted (the workload's record) [5, 6]  equal=True\n" in capsys.readouterr().out
