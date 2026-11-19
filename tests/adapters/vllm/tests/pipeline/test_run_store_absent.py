"""On a tree without the team's run and store tool (``research``), what needs it is refused by name and never skipped:
``config.run_store``'s UsageError names the entry point, while every module still imports."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from verity_vllm import config
from verity_vllm.pipeline import (
    advice,
    cli,
    commit_plan,
    research_tools,
    serve_seal,
    source_identity,
)









def test_the_source_identity_command_is_refused_not_skipped(no_run_store, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["source-identity"])
    assert e.value.code == 2
    assert "the source-identity precheck of stage 'precheck' is refused" in capsys.readouterr().err


def test_a_module_missing_inside_the_run_store_is_not_a_refusal(tmp_path, monkeypatch):
    (tmp_path / "run_store_stand_in.py").write_text("import no_such_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, "no_such_dependency", None)
    with pytest.raises(ModuleNotFoundError) as e:
        config.run_store("run_store_stand_in", "an entry point")
    assert e.value.name == "no_such_dependency"
