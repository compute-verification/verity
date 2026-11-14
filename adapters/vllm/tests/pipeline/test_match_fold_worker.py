"""The Match's fold worker (`verity_vllm.pipeline.match`) takes the row's Match pipeline as a declared input: an option the CLI
fills from the environment it was started with, never a read of the process environment inside the fold."""
from __future__ import annotations

import pytest

from verity_vllm import gc_tuning
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match as RC


@pytest.mark.parametrize("environ, level", [({"MATCH_PIPELINE": "shared"}, 1), ({"MATCH_PIPELINE": "standalone"}, 9), ({}, 9)])
def test_the_fold_worker_compresses_the_journal_by_the_rows_match_pipeline(tmp_path, monkeypatch, environ, level):
    """The row's stages run under its MATCH_PIPELINE, and `shared` alone compresses the access journal at level 1 (the same
    lines, a larger file); any other pipeline, and none, keeps level 9."""
    a = cli.parse("match", ["--worker", "fold", "--log", "l", "--profile", "p", "--out", str(tmp_path / "s.json"), "--persist-dir", str(tmp_path)],
                  environ)
    doc = {"digest": "0" * 16, "n_instances": 0, "n_steps": 0, "unresolved": 0, "unknown_roots": [], "max_rss_gb": 0}
    seen: dict = {}
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    monkeypatch.setattr(RC, "fold_summary_result", lambda *args, **kw: (dict(doc), None))
    monkeypatch.setattr(RC, "persist_program", lambda result, run_dir, **kw: seen.update(kw) or {})
    assert RC.fold_worker(a) == 0 and seen["journal_level"] == level
    assert (seen["profile_name"], seen["log_path"]) == ("p", "l")


@pytest.mark.parametrize("given, missing", [(("--profile", "p", "--out", "s.json"), "log"), (("--log", "l", "--out", "s.json"), "profile"),
                                            (("--log", "l", "--profile", "p"), "out"), (("--out", "s.json"), "log")])
def test_the_fold_worker_names_the_first_option_it_lacks(given, missing):
    with pytest.raises(SystemExit, match=f"--worker fold needs --{missing}$"):
        RC.fold_worker(cli.parse("match", ["--worker", "fold", *given], {}))
