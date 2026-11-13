"""In an xdist worker, the circuit-checks the tests start keep to that worker's share of the suite's memory (`worker_slot_gb`).

circuit-check's catalog (`Catalog`: `targets.suite()` and what its roots reach) is built once per session, and only for a test
that asks for it (the `catalog` or `suite` fixture): when collection ends, before any test runs, as circuit-check builds it
(`targets.suite()` grows as tracing registers Definitions, so a later build would list more targets).  It is made from the
catalog bindings' records (`circuit_check.units`): handed ones (`CIRCUIT_CHECK_RECORDS`) or else the ones `check`'s
circuit-check keeps (`RECORDS`), and only when every binding's record counts: this process builds no binding a record misses
(`_from_records`), and a handed record that doesn't count fails the session by name.  Without handed records, unless every
one counts, each binding is built in a process of its own, as many at once as the session has workers (or this process CPUs)
and their memory holds at `heavy.json`'s peaks (`_budget`, `units.built_apart`): under xdist, by
the first worker that asks, which writes what each binding reaches beside the workers' basetemps (`MANIFEST`), while the others
wait on its lock and then take that (`units.from_manifest`).  Every process makes the catalog's roots itself, and each suite
target the first time a test reads it (`Suite`), from its id or from its binding's record (`units.make`).  A test that asks for
it is marked slow, so `--quick` builds none; in a process that didn't build it, `targets.suite` and `targets.catalog` fail by
name."""
from __future__ import annotations

import fcntl
import functools
import json
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from circuit_check import cache as CC
from circuit_check import cli
from circuit_check import targets as TG
from circuit_check import units as UN

#: one xdist worker's share of the memory suites.py runs the suite in (`[tool.verity.tests] worker_gb`, which it divides by)
WORKER_GB = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["tool"]["verity"]["tests"]["worker_gb"]

#: where `check`'s circuit-check keeps the catalog bindings' records (`--cache cache_dir()/circuit-check --catalog-units`, with
#: `cache_dir` of tools/verity/check/common.py at its default), or `CIRCUIT_CHECK_RECORDS`: not `CHECK_CACHE`, which the suite
#: guard points at an empty directory of its own (`HOST_CACHES`).  A record counts only for inputs identical to the ones it was
#: made from (`units.plan`, as `check`'s circuit-check takes it), so it changes how long the catalog takes, not what it holds.
_HANDED = os.environ.get("CIRCUIT_CHECK_RECORDS")
RECORDS = Path(_HANDED or Path.home() / ".cache" / "verity-check" / "circuit-check")
#: the records were handed (the graph's test-file runner sets them from `circuit-catalog:records`): the catalog is theirs or none
HANDED = bool(_HANDED)

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


# --- the catalog ----------------------------------------------------------------------------------------------------------------

class Suite(Mapping):
    """`targets.suite()` over the catalog's ids, in its order, each target made the first time it is read (``make``, which
    keeps it): a test that reads a few of them makes those, and one that reads every value makes the whole suite."""

    def __init__(self, ids: list[str], make: Callable[[str], Any]) -> None:
        self._ids, self._held, self._make = ids, frozenset(ids), make

    def __getitem__(self, fid: str) -> Any:
        if fid not in self._held:
            raise KeyError(fid)
        return self._make(fid)

    def __contains__(self, fid: object) -> bool:
        return fid in self._held

    def __iter__(self):
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass
class Catalog:
    """circuit-check's catalog as this process has it: ``suite``, `targets.suite()` (and its `targets.BINDINGS`); ``reached``,
    each id the catalog's roots reach (`targets.reachable`, in its order) with its family (`targets.family`) and gate count, so
    a test that selects targets by family makes only those; ``make``, a reached id's Definition, made on demand (`units.make`);
    ``how``, whether the records gave it; ``calls``, the Calls `units.plan` gives from the records (by which ``--all
    --catalog-units`` tells whether each target is one), or None, where `targets.call_families` of the suite gives them."""
    suite: Suite
    reached: dict[str, tuple[str, int]]
    make: Callable[[str], Any]
    how: str
    calls: set[str] | None = None


_BUILT: list[Catalog] = []
#: the tests that ask for the catalog without the slow mark, which `--quick` would build it for
_UNMARKED: list[str] = []


class _Missed(Exception):
    """A binding's record doesn't count."""


def _counted(lookup: Callable, cache: Path, repo: Path, e: UN.Entry) -> Any:
    if (hit := lookup(cache, repo, e)) is None:
        raise _Missed(f"{e.name}'s record doesn't count (a read changed, or it lists other roots than its binding makes now)")
    return hit


def _from_records(cache: Path, handed: bool = False) -> Catalog | None:
    """The catalog from the bindings' records (`units.plan`), each suite target made when it is read (`units.make`); None when
    there is no record of this version, one doesn't count or one is stale, but with ``handed`` records, a `pytest.UsageError`
    naming why.  Either way before any binding they miss is built here: `units.plan` builds such a binding in the process that
    calls it, which under xdist is every worker of the session at once, each holding what the heaviest binding's build holds.
    So the first record that doesn't count ends the plan (`_counted`), and the one build the session shares (`_apart`) makes
    them all.  A target is taken as its record describes it, as `cli._planned` takes a target whose result it reuses: holding
    each one to its descriptor (`cache.descriptor`, as `cli._made` does for one it checks) walks all it reaches.  The records'
    cache, unless handed, is marked used (`cache.mark`), so its version doesn't expire while sessions read it; a session
    removes nothing, which the runs that write the cache do (`cache.expire`)."""
    if not handed:
        CC.mark(cache)
    with pytest.MonkeyPatch.context() as mp:
        # suites.py's variables say how the suite runs, not what a check computes: keyed, no record circuit-check stored counts
        mp.setattr(CC, "UNKEYED", CC.UNKEYED | {k for k in os.environ if k.startswith("VERITY_SUITE_")}
                   | {"CIRCUIT_CHECK_RECORDS"})
        # the records were made traced, so a function they read (a kernel's) is named by its module's digest, which
        # `cache._where` gives only under the tracer's root: without it, by path and line, and no such record counts
        mp.setattr(CC, "_ROOT", [cli.REPO.resolve()])
        mp.setattr(UN, "_lookup", functools.partial(_counted, UN._lookup))
        try:
            plan = UN.plan(cache, cli.REPO)
            why = None if plan is not None else "they hold no record of this circuit-check's version (`cache.version`)"
        except (UN.Stale, _Missed) as e:
            TG.BINDINGS.clear()
            plan, why = None, str(e)
    if plan is None:
        if handed:
            raise pytest.UsageError(f"circuit-check's catalog: CIRCUIT_CHECK_RECORDS={cache} was handed, and {why}: hand the "
                                    "records `circuit-check --all --catalog-units --records-only` stored for this tree, or none")
        return None
    d = plan.derived
    for fid in d.ids:
        TG.BINDINGS.setdefault(fid, set()).update(d.units.get(fid, set()))
    reused = sum(e.reused for e in plan.entries)
    return Catalog(Suite(d.ids, lambda fid: UN.make(plan, fid)), {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()},
                   lambda fid: UN.make(plan, fid), f"from records: {reused} of {len(plan.entries)} bindings reused", plan.calls)


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
    kept, dropped, grew, _ = UN.built_apart(entries, jobs, _budget(os.environ), weights)
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
    return Catalog(Suite(d.ids, lambda fid: UN.make(plan, fid)), {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()},
                   lambda fid: UN.make(plan, fid), f"{how}, {len(entries)} bindings")


def build() -> Catalog:
    """This process's catalog: built the first time it is asked for, from the records when they count (and only from them
    when they were handed)."""
    if not _BUILT:
        _BUILT.append(_from_records(RECORDS, HANDED) or _apart(_SHARED[0] if _SHARED else None, _jobs(os.environ)))
        print(f"circuit-check's catalog: {_BUILT[0].how}, {len(_BUILT[0].suite)} targets", flush=True)
    return _BUILT[0]


def pytest_collection_finish(session):
    """The catalog, when a selected test asks for it (each xdist worker collects them all), before any test runs."""
    wants = [i for i in session.items if "catalog" in getattr(i, "fixturenames", ())]
    _UNMARKED[:] = [i.nodeid for i in wants if i.get_closest_marker("slow") is None]
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
    """The catalog, with its Calls (`Catalog.calls`, else `targets.call_families` of its suite, which makes every target), so a
    check that doesn't say whether its target is a Call is told from them, as ``--all`` tells it, where `targets.is_call` alone
    would refuse a family it can't decide."""
    if _UNMARKED:
        pytest.fail(f"these tests build circuit-check's catalog without the slow mark, which --quick would build it for: mark "
                    f"them slow: {_UNMARKED}")
    built = build()
    if built.calls is None:
        TG.call_families(built.suite)
    elif not TG._CALLS:
        TG._CALLS.update(built.calls)
    return built


@pytest.fixture(scope="session")
def suite(catalog) -> Suite:
    """`targets.suite()`: every Definition circuit-check checks, by id."""
    return catalog.suite
