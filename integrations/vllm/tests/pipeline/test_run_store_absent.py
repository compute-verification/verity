"""On a tree without the team's run and store tool (``research``), what needs it is refused by name and never skipped:
``config.run_store``'s UsageError names the entry point, while every module still imports."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from verity_vllm import config
from verity_vllm.pipeline import advice, cli, commit_plan, hot, research_tools, serve_seal, source_identity
from verity_vllm.pipeline.row_driver import RESEARCH
from verity_vllm.pipeline.tp import commit as tp_commit

RUN_STORE = ("research", "research.pythonpath", "research.store", "research.store.tool", "research.telemetry",
             "research.telemetry.source_identity", RESEARCH)


@pytest.fixture
def no_run_store(monkeypatch):
    """`research` blocked, and the names already read from it forgotten: from the modules' dicts, since `hasattr` would run
    their `__getattr__`, which is the refusal itself."""
    for module, names in ((research_tools, research_tools.FROM_RESEARCH), (source_identity, source_identity.REEXPORTS)):
        for name in vars(module).keys() & set(names):
            monkeypatch.delitem(vars(module), name)
    for name in RUN_STORE:
        monkeypatch.setitem(sys.modules, name, None)


ENTRIES = {
    "the Commit": lambda: source_identity.precheck("commit"),
    "the TP Commit's tree of record": tp_commit.tree_of_record,
    "the hot worker's key": lambda: hot.code_identity(source_identity.TREE),
    "the commit plan's key": lambda: commit_plan.key(SimpleNamespace(workload=None)),
    "the advice digest": advice.registered_compiler,
    "the seal": lambda: serve_seal.seal_of(SimpleNamespace(run_id="r")),
    "row stage": lambda: config.run_store(RESEARCH, "row stage"),
}


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_refusal_names_its_entry_point(no_run_store, entry):
    with pytest.raises(config.UsageError) as e:
        ENTRIES[entry]()
    assert str(e.value).startswith(f"{entry}") and " is refused: " in str(e.value) and "leaves out the run store" in str(e.value)


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
