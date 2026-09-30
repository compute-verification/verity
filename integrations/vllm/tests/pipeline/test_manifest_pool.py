"""The component builds' process pool (`manifest.build_components`): a run admitted with its own memory budget (the sweep's
`BUILD_RAM_BUDGET_GB`) budgets from it and takes no host-wide lock, so concurrent Builds on one host do not queue on each other's manifests;
a run without one shares the host's headroom, one pool at a time."""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import manifest as M

TASKS = [("a", {}), ("b", {})]
GiB = 1 << 30


def _pool(monkeypatch, head: int | None) -> tuple[list[bool], list[int]]:
    """The host lock's acquisitions and the headroom each pool was given, with the builds themselves stubbed out."""
    taken: list[bool] = []
    given: list[int] = []

    @contextmanager
    def lock():
        taken.append(True)
        yield

    monkeypatch.setattr(M, "_pool_lock", lock)
    monkeypatch.setattr(M, "_headroom", lambda: head)
    monkeypatch.setattr(M, "_build_forked", lambda tasks, jobs, after, h, log: given.append(h) or [])
    return taken, given


@pytest.mark.parametrize(("head", "budget", "want"), [(64 * GiB, 16 * GiB, 16 * GiB), (8 * GiB, 16 * GiB, 8 * GiB), (None, 16 * GiB, 16 * GiB)])
def test_a_run_with_its_own_budget_takes_no_host_lock_and_budgets_from_the_smaller(monkeypatch, head, budget, want):
    taken, given = _pool(monkeypatch, head)
    M.build_components(TASKS, jobs=4, ram_budget=budget)
    assert taken == [] and given == [want]


def test_a_run_without_a_budget_shares_the_host_headroom_under_the_host_lock(monkeypatch):
    taken, given = _pool(monkeypatch, 64 * GiB)
    M.build_components(TASKS, jobs=4)
    assert taken == [True] and given == [64 * GiB]


def test_one_process_takes_neither(monkeypatch):
    taken, given = _pool(monkeypatch, 64 * GiB)
    monkeypatch.setattr(M, "build", lambda d, **kw: {"d": d})
    assert M.build_components(TASKS, jobs=1) == [{"d": "a"}, {"d": "b"}] and taken == [] and given == []


def test_the_budget_is_the_sweeps_variable_or_the_flag():
    argv = ["build-global", "--fixture", "f.json", "--programs-root", "r", "--out", "m.json"]
    assert cli.parse("manifest", argv, {}).ram_budget_gb is None
    assert cli.parse("manifest", argv, {M.RAM_BUDGET_ENV: "150"}).ram_budget_gb == 150.0
    assert cli.parse("manifest", [*argv, "--ram-budget-gb", "12.5"], {M.RAM_BUDGET_ENV: "150"}).ram_budget_gb == 12.5
