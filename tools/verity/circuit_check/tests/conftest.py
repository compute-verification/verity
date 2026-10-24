"""The suite's report: with ``CIRCUIT_CHECK_REPORT=path`` set, every target the suite checked is written there as one
`verity-circuit-check/v1` report when the session ends (CI uploads it).  In an xdist worker, the circuit-checks the tests start
keep to that worker's share of the suite's memory (`worker_slot_gb`).

circuit-check's catalog (`Catalog`: `targets.suite()` and what its roots reach) is built once per process, and only for a test
that asks for it (the `catalog` or `suite` fixture): when collection ends, before any test runs, as circuit-check builds it
(`targets.suite()` grows as tracing registers Definitions, so a later build would list more targets).  It is made from the
catalog bindings' records that `check`'s circuit-check keeps (`circuit_check.units`, `RECORDS`) when they count, and built whole
otherwise.  A test that asks for it is marked slow, so `--quick` builds none; in a process that didn't build it,
`targets.suite` and `targets.catalog` fail by name."""
from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from circuit_check import cache as CC
from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import targets as TG
from circuit_check import units as UN

#: one xdist worker's share of the memory suites.py runs the suite in (`[tool.verity.tests] worker_gb`, which it divides by)
WORKER_GB = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["worker_gb"]

#: where `check`'s circuit-check keeps the catalog bindings' records (`--cache cache_dir()/circuit-check --catalog-units`, with
#: `cache_dir` of tools/verity/check/common.py at its default), or `CIRCUIT_CHECK_RECORDS`: not `CHECK_CACHE`, which the suite
#: guard points at an empty directory of its own (`HOST_CACHES`).  A record counts only for inputs identical to the ones it was
#: made from (`units.plan`, as `check`'s circuit-check takes it), so it changes how long the catalog takes, not what it holds.
RECORDS = Path(os.environ.get("CIRCUIT_CHECK_RECORDS") or Path.home() / ".cache" / "verity-check" / "circuit-check")


def worker_slot_gb(env: Mapping[str, str], worker_gb: float = WORKER_GB) -> int | None:
    """The `cli.SLOT_MEM_ENV` a circuit-check started in this process budgets (`cli.budget_gb`, `cli.auto_jobs`): in an xdist
    worker, the worker's share ``worker_gb``, or the slot's own when that is smaller; outside one, None (the process has all of
    it).  Unset, each worker's circuit-check budgets the whole slot or cgroup, which every worker of the suite shares."""
    if not env.get("PYTEST_XDIST_WORKER"):
        return None
    slot = env.get(cli.SLOT_MEM_ENV, "")
    return min(int(slot), int(worker_gb)) if slot.isdigit() else int(worker_gb)


@pytest.fixture(scope="session", autouse=True)
def worker_share():
    with pytest.MonkeyPatch.context() as mp:
        if (gb := worker_slot_gb(os.environ)) is not None:
            mp.setenv(cli.SLOT_MEM_ENV, str(gb))
        yield


@pytest.fixture(scope="session")
def results():
    out: list[dict] = []
    yield out
    path = os.environ.get("CIRCUIT_CHECK_REPORT")
    if path and out:
        Path(path).write_text(json.dumps(CK.report(out, CK.Options()), indent=1, default=str) + "\n")


# --- the catalog ----------------------------------------------------------------------------------------------------------------

@dataclass
class Catalog:
    """circuit-check's catalog as this process has it: ``suite``, `targets.suite()` (and its `targets.BINDINGS`); ``reached``,
    each id the catalog's roots reach (`targets.reachable`, in its order) with its family and gate count; ``make``, a reached
    id's Definition, made on demand; ``how``, whether the records gave it."""
    suite: dict[str, Any]
    reached: dict[str, tuple[str, int]]
    make: Callable[[str], Any]
    how: str


_BUILT: list[Catalog] = []
#: the tests that ask for the catalog without the slow mark, which `--quick` would build it for
_UNMARKED: list[str] = []
_REAL = {"suite": TG.suite, "catalog": TG.catalog}


def _from_records(cache: Path) -> Catalog | None:
    """The catalog from the bindings' records (`units.plan`), each suite target made along the path its record lists; None when
    no record counts or one is stale.  A target is taken as its record describes it, as `cli._planned` takes a target whose
    result it reuses: holding each one to its descriptor (`cache.descriptor`, as `cli._planned` does for one it checks) walks
    all it reaches, which over the suite took longer than making it (past 15 minutes, against 426 s, r20261022-005352-48a5)."""
    with pytest.MonkeyPatch.context() as mp:
        # suites.py's variables say how the suite runs, not what a check computes: keyed, no record circuit-check stored counts
        mp.setattr(CC, "UNKEYED", CC.UNKEYED | {k for k in os.environ if k.startswith("VERITY_SUITE_")}
                   | {"CIRCUIT_CHECK_REPORT", "CIRCUIT_CHECK_RECORDS"})
        # the records were made traced, so a function they read (a kernel's) is named by its module's digest, which
        # `cache._where` gives only under the tracer's root: without it, by path and line, and no such record counts
        mp.setattr(CC, "_ROOT", [cli.REPO.resolve()])
        try:
            plan = UN.plan(cache, cli.REPO)
            if plan is None:
                return None
            d = plan.derived

            def make(fid: str) -> Any:
                return UN._materialize(plan.entries, plan.objects, fid, via=d.via.get(fid))

            suite = {fid: make(fid) for fid in d.ids}
        except UN.Stale:
            TG.BINDINGS.clear()
            return None
    for fid in d.ids:
        TG.BINDINGS.setdefault(fid, set()).update(d.units.get(fid, set()))
    reused = sum(e.reused for e in plan.entries)
    return Catalog(suite, {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()}, make,
                   f"from records: {reused} of {len(plan.entries)} bindings reused")


def _whole() -> Catalog:
    with pytest.MonkeyPatch.context() as mp:
        # `targets.suite` builds through the module's `catalog`, which `catalog_guard` replaces
        mp.setattr(TG, "catalog", _REAL["catalog"])
        suite = _REAL["suite"]()
    reached = dict(TG.REACHED)
    return Catalog(suite, {fid: (TG.family(fn), getattr(fn, "gates", 1)) for fid, fn in reached.items()},
                   lambda fid: reached[fid] if fid in reached else suite[fid], "built whole")


def build() -> Catalog:
    """This process's catalog: built the first time it is asked for, from the records when they count."""
    if not _BUILT:
        _BUILT.append(_from_records(RECORDS) or _whole())
        print(f"circuit-check's catalog: {_BUILT[0].how}, {len(_BUILT[0].suite)} targets", flush=True)
    return _BUILT[0]


def _marked(item) -> bool:
    return item.get_closest_marker("slow") is not None or item.get_closest_marker("circuit_suite") is not None


def pytest_generate_tests(metafunc):
    """A `circuit_suite` test taking ``fid`` is one test per suite target; collected only under ``-m circuit_suite``
    (`pytest_collection_modifyitems`), so only then is the catalog built for it here."""
    if "fid" in metafunc.fixturenames and metafunc.definition.get_closest_marker("circuit_suite") is not None:
        metafunc.parametrize("fid", sorted(build().suite) if _suite_selected(metafunc.config) else [])


def _suite_selected(config) -> bool:
    return "circuit_suite" in (config.getoption("markexpr") or "")


def pytest_collection_modifyitems(config, items):
    """The per-target tests are `circuit-check --all` as pytest items, which `check` runs through the CLI: they are collected
    only when `-m` names `circuit_suite`."""
    if _suite_selected(config):
        return
    dropped = [i for i in items if "circuit_suite" in i.keywords]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [i for i in items if "circuit_suite" not in i.keywords]


def pytest_collection_finish(session):
    """The catalog, when a selected test asks for it (each xdist worker collects them all), before any test runs."""
    wants = [i for i in session.items if "catalog" in getattr(i, "fixturenames", ())]
    _UNMARKED[:] = [i.nodeid for i in wants if not _marked(i)]
    if wants and not _UNMARKED and not session.config.option.collectonly:
        build()


def _unbuilt(*_a, **_k):
    pytest.fail("circuit-check's catalog is built only for a test that asks for the `catalog` or `suite` fixture (conftest.py), "
                "once per process before any test runs: give this test one, and mark it slow")


@pytest.fixture(scope="session", autouse=True)
def catalog_guard():
    """`targets.suite` is the catalog this process built, and building another (`targets.catalog`) fails, as both do where
    none was built."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TG, "suite", (lambda: _BUILT[0].suite) if _BUILT else _unbuilt)
        mp.setattr(TG, "catalog", _unbuilt)
        yield


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    if _UNMARKED:
        pytest.fail(f"these tests build circuit-check's catalog without the slow mark, which --quick would build it for: mark "
                    f"them slow: {_UNMARKED}")
    return build()


@pytest.fixture(scope="session")
def suite(catalog) -> dict[str, Any]:
    """`targets.suite()`: every Definition circuit-check checks, by id."""
    return catalog.suite
