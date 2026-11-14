"""`cli.task_list_guard`: ``--all`` refuses a task list far larger than the last one that passed, naming the families that
grew, before any target is checked, and never passes one when its reference is missing."""
from __future__ import annotations

import json

import pytest

from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import sidecars as SC
from circuit_check import targets as TG


@pytest.fixture
def widgets(tmp_path, monkeypatch):
    """A package of two modules, ``synwidgets.a`` and ``synwidgets.b``, the only one circuit-check finds sidecars in, with the
    guard's reference at ``targets`` 20, ``factor`` 2 and ``floor`` 3, and ``WidgetA_v1``'s count, 5, beside ``a``."""
    pkg = tmp_path / "synwidgets"
    pkg.mkdir()
    for f in ("__init__.py", "a.py", "b.py"):
        (pkg / f).write_text("")
    (pkg / "a.circuit-check.unkeyed.json").write_text(json.dumps({"tasks": {"WidgetA_v1": 5}}))
    ref = tmp_path / "task_list.json"
    ref.write_text(json.dumps({"targets": 20, "factor": 2, "floor": 3}))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(SC, "REPO", tmp_path)
    monkeypatch.setattr(SC, "PACKAGES", ("synwidgets",))
    monkeypatch.setattr(SC, "TREES", ())
    monkeypatch.setattr(cli, "TASK_LIST", ref)
    SC.discover.cache_clear()
    SC.unkeyed.cache_clear()
    yield pkg
    SC.discover.cache_clear()
    SC.unkeyed.cache_clear()


def _ids(**families: int) -> list[tuple[str, str]]:
    return [("definition", f"{fam}{{N={i}}}") for fam, n in families.items() for i in range(n)]


def test_a_family_multiplied_past_the_factor_is_refused_and_named(widgets):
    """A family over the floor and over ``factor`` times its count is named with both counts, one with no count against the
    floor; the whole list past ``factor`` times its length is refused too, naming the families that grew most."""
    cli.task_list_guard(_ids(WidgetA_v1=10, WidgetB_v1=6, WidgetC_v1=3))
    with pytest.raises(cli.TaskListRefused) as e:
        cli.task_list_guard(_ids(WidgetA_v1=11, WidgetB_v1=7, WidgetC_v1=3))
    assert str(e.value) == (
        "the task list holds 21 targets against 20 when it last passed; these families hold more than 3 targets and more "
        "than 2 times their count then:\n"
        "  WidgetA_v1: 11 against 5\n"
        "  WidgetB_v1: 7 against 3 (no count: at most the floor, 3)\n"
        "No target was checked.  A growth on purpose: write the new counts with circuit-check --all --accept-task-list (with "
        "this run's --cache and --catalog-units) and commit the files it names; else find what multiplied these families")
    with pytest.raises(cli.TaskListRefused) as e:
        cli.task_list_guard(_ids(WidgetA_v1=10, WidgetB_v1=6, **{f"WidgetD{i}_v1": 2 for i in range(13)}))
    assert str(e.value).startswith(
        "the task list holds 42 targets against 20 when it last passed, past 2 times that; the families that grew most:\n"
        "  WidgetA_v1: 10 against 5\n  WidgetB_v1: 6 against 3 (no count: at most the floor, 3)\n"
        "  WidgetD0_v1: 2 against 3 (no count: at most the floor, 3)\n")


def test_a_missing_or_malformed_reference_is_refused_never_passed(widgets):
    cli.TASK_LIST.unlink()
    with pytest.raises(cli.TaskListRefused, match=r"task_list.json is missing or malformed .*restore it from git, or write it "
                                                  r"with circuit-check --all --accept-task-list"):
        cli.task_list_guard(_ids(WidgetA_v1=1))
    cli.TASK_LIST.write_text(json.dumps({"targets": 20, "factor": 2}))
    with pytest.raises(cli.TaskListRefused, match=r"missing or malformed \(KeyError\('floor'\)\)"):
        cli.task_list_guard(_ids(WidgetA_v1=1))


def test_a_count_in_two_files_is_refused(widgets):
    (widgets / "b.circuit-check.unkeyed.json").write_text(json.dumps({"tasks": {"WidgetA_v1": 9}}))
    with pytest.raises(cli.TaskListRefused, match=r"the count of family WidgetA_v1 is in synwidgets.a's unkeyed file and in "
                                                  r"synwidgets.b's: keep its owner's"):
        cli.task_list_guard(_ids(WidgetA_v1=1))


def test_accepting_writes_each_family_s_count_beside_its_owner_and_the_list_s_length(widgets, monkeypatch):
    """``--accept-task-list`` keeps the factor and floor, writes the list's length and each family over the floor beside the
    module that owns it, drops the counts no family over the floor has any longer, and refuses a family no module owns."""
    monkeypatch.setattr(cli, "_family_owner", lambda kind, label: {"WidgetB": "synwidgets.b"}.get(label[:7]))
    with pytest.raises(cli.TaskListAccepted, match=r"^the task list's 9 targets are the reference now: task_list.json and the "
                                                   r"tasks of 2 unkeyed file\(s\): synwidgets/a.circuit-check.unkeyed.json, "
                                                   r"synwidgets/b.circuit-check.unkeyed.json$"):
        cli.task_list_guard(_ids(WidgetA_v1=2, WidgetB_v1=7), accept=True)
    assert json.loads(cli.TASK_LIST.read_text()) == {"targets": 9, "factor": 2, "floor": 3}
    assert not (widgets / "a.circuit-check.unkeyed.json").exists()
    assert json.loads((widgets / "b.circuit-check.unkeyed.json").read_text()) == {"tasks": {"WidgetB_v1": 7}}
    cli.task_list_guard(_ids(WidgetA_v1=3, WidgetB_v1=14))
    with pytest.raises(cli.TaskListRefused, match=r"  WidgetB_v1: 15 against 7\n"):
        cli.task_list_guard(_ids(WidgetB_v1=15))
    with pytest.raises(cli.TaskListRefused, match=r"no module owns these families of more than 3 targets, so their counts "
                                                  r"have no unkeyed file to go in: WidgetC_v1"):
        cli.task_list_guard(_ids(WidgetC_v1=4), accept=True)


def test_all_refuses_before_any_target_is_checked(widgets, monkeypatch):
    """``--all`` (`cli.run` with ``guard``) goes through the guard once its task list is known, before any check."""
    from verity.circuits.ops import fp32

    fn = fp32.F32Add
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "suite", lambda: {f"WidgetA_v1{{N={i}}}": fn for i in range(11)})
    monkeypatch.setattr(TG, "templates", lambda: [])
    monkeypatch.setattr(TG, "call_families", lambda found=None: set())
    monkeypatch.setattr(CK, "check_definition", lambda *a, **k: pytest.fail("a target was checked"))
    with pytest.raises(cli.TaskListRefused, match=r"  WidgetA_v1: 11 against 5\n"):
        cli.run([], CK.Options(), everything=True, log=None, isolate=False, guard="check")


@pytest.mark.slow
def test_main_s_own_task_list_passes(suite):
    """The task list ``--all`` builds over this tree, its catalog's Definitions and every template, passes the reference this
    tree holds: a change that grows a family past it commits the new counts (``--accept-task-list``) with it."""
    ids = [("definition", fid) for fid in suite] + [("template", t[0]) for t in TG.templates()]
    cli.task_list_guard(ids)
