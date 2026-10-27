"""The suite's report: with ``CIRCUIT_CHECK_REPORT=path`` set, every target the suite checked is written there as one
`verity-circuit-check/v1` report when the session ends.  In an xdist worker, the circuit-checks the tests start
keep to that worker's share of the suite's memory (`worker_slot_gb`).

circuit-check's catalog (`Catalog`: `targets.suite()` and what its roots reach) is built once per session, and only for a test
that asks for it (the `catalog` or `suite` fixture): when collection ends, before any test runs, as circuit-check builds it
(`targets.suite()` grows as tracing registers Definitions, so a later build would list more targets).  It is made from the
catalog bindings' records that `check`'s circuit-check keeps (`circuit_check.units`, `RECORDS`) when they count.  Otherwise
each binding is built in a process of its own, as many at once as the session has workers (or this process CPUs) and their
memory holds at `heavy.json`'s peaks (`_budget`, `units.built_apart`): under xdist, by the first worker that asks, which writes what each binding
reaches beside the workers' basetemps (`MANIFEST`), while the others wait on its lock and then take that
(`units.from_manifest`).  Every process makes the catalog's roots itself and each suite
target from its id, or from its binding's record (`units.make`).  A test that asks for it is marked slow, so `--quick` builds
none; in a process that didn't build it, `targets.suite` and `targets.catalog` fail by name."""
from __future__ import annotations

import fcntl
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

#: what each catalog binding reaches (`units.manifest`), written once a session by the xdist worker that builds the catalog into
#: the directory every worker's basetemp is in (the controller's, which the session starts empty), beside its lock
MANIFEST = "circuit-check-catalog.json"
_SHARED: list[Path] = []


def pytest_configure(config):
    if os.environ.get("PYTEST_XDIST_WORKER") and config.option.basetemp:
        _SHARED.append(Path(config.option.basetemp).parent)


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


def _from_records(cache: Path) -> Catalog | None:
    """The catalog from the bindings' records (`units.plan`), each suite target made along the path its record lists; None when
    no record counts or one is stale.  A target is taken as its record describes it, as `cli._planned` takes a target whose
    result it reuses: holding each one to its descriptor (`cache.descriptor`, as `cli._planned` does for one it checks) walks
    all it reaches, which over the suite takes longer than making it."""
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


def _write(path: Path, got: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(got))
    os.replace(tmp, path)


def _jobs(env: Mapping[str, str]) -> int:
    """The bindings built at once: one per xdist worker of the session, which wait while one builds, else one per CPU."""
    n = env.get("PYTEST_XDIST_WORKER_COUNT", "")
    return int(n) if n.isdigit() and int(n) > 0 else cli.cores()


def _budget(env: Mapping[str, str], worker_gb: float = WORKER_GB) -> float:
    """What the catalog's build may take at once, in GiB.  In an xdist worker: what suites.py gave the session's workers,
    ``worker_gb`` each (within the slot's mem=, which is the whole run's and the suites beside this one share), less what they
    hold as they wait on the build, each counted at this worker's resident set; at most what `cli.memory_gb` leaves.  Outside
    one, `cli.budget_gb`."""
    n = env.get("PYTEST_XDIST_WORKER_COUNT", "")
    if not (n.isdigit() and int(n) > 0):
        return cli.budget_gb(None)
    slot, n = env.get(cli.SLOT_MEM_ENV, ""), int(n)
    share = min(n * worker_gb, int(slot)) if slot.isdigit() else n * worker_gb
    return min(float(cli.memory_gb()), share - n * (cli._rss_gb() or 0.0))


def _apart_built(entries: list[UN.Entry], jobs: int) -> tuple[list[UN.Entry], list[tuple[str, ...]], str]:
    """`units.built_apart` within `_budget`, each binding counted at how far `heavy.json` says its process grows; and how the
    build went, naming the bindings that grew past that, whose growth belongs in `heavy.json`."""
    weights = cli.heavy()
    kept, dropped, grew = UN.built_apart(entries, jobs, _budget(os.environ), weights)
    past = sorted(((gb, name) for name, gb in grew.items() if gb > weights.get(name, UN.APART_GB)), reverse=True)
    return kept, dropped, "built apart" + (f" ({len(past)} grew past their heavy.json GiB: "
                                           + ", ".join(f"{name} {gb}" for gb, name in past[:5]) + ")" if past else "")


def _built(entries: list[UN.Entry], shared: Path | None, jobs: int) -> tuple[list[UN.Entry], str]:
    """``entries`` with what each binding reaches: built apart here, or under xdist taken from the worker that built them."""
    if shared is None:
        kept, _, how = _apart_built(entries, jobs)
        return kept, how
    path = shared / MANIFEST
    with open(shared / (MANIFEST + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)    # held until the builder wrote the manifest, or died
        if path.exists():
            got = json.loads(path.read_text())
            if "error" in got:
                raise RuntimeError(f"the xdist worker that built circuit-check's catalog failed: {got['error']}")
            return UN.from_manifest(entries, got), "taken from the worker that built it"
        try:
            kept, dropped, how = _apart_built(entries, jobs)
        except Exception as e:
            _write(path, {"error": f"{type(e).__name__}: {e}"})
            raise
        _write(path, UN.manifest(kept, dropped))
        return kept, how


def _apart(shared: Path | None, jobs: int) -> Catalog:
    """The catalog with each binding built in a process of its own (`units.built_apart`), once a session (`_built`)."""
    entries, how = _built(UN.made_entries(), shared, jobs)
    plan = UN.planned(entries)
    d = plan.derived
    for fid in d.ids:
        TG.BINDINGS.setdefault(fid, set()).update(d.units.get(fid, set()))
    return Catalog({fid: UN.make(plan, fid) for fid in d.ids}, {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()},
                   lambda fid: UN.make(plan, fid), f"{how}, {len(entries)} bindings")


def build() -> Catalog:
    """This process's catalog: built the first time it is asked for, from the records when they count."""
    if not _BUILT:
        _BUILT.append(_from_records(RECORDS) or _apart(_SHARED[0] if _SHARED else None, _jobs(os.environ)))
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
