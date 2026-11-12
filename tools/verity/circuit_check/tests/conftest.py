"""In an xdist worker, the circuit-checks the tests start keep to that worker's share of the suite's memory (`worker_slot_gb`).

circuit-check's catalog (`Catalog`: `targets.suite()` and what its roots reach) is built once per session, and only for a test
that asks for it (the `catalog` or `suite` fixture): when collection ends, before any test runs, as circuit-check builds it
(`targets.suite()` grows as tracing registers Definitions, so a later build would list more targets).  It is made from the
catalog bindings' records (`circuit_check.units`) in `RECORDS`, the directory `check`'s circuit-check keeps them in, or
`CIRCUIT_CHECK_RECORDS`: each record that counts is taken, and this process builds no binding a record misses
(`_from_records`).  The bindings whose record doesn't count, a handed one's too, are built once a session by a process of
its own, ``circuit-check --all --records-only`` (`_recorded`), which builds each of them in a process forked for it, as many
at once as the session has workers (or this process CPUs) and their memory holds at `heavy.json`'s peaks (`_budget`,
`units.built_apart`), and stores each one's record in that directory, as `check`'s circuit-check stores it; the session then
takes them all.  Under xdist the first worker that asks starts it and writes what it reported beside the workers' basetemps
(`RECORDED`), while the others wait on its lock and then take the records too (`_once`).  Where that process stored no
record a session can take (it could not trace what it did, or the tree changed under it), each binding is built in a process
of its own without one (`_apart`), once a session likewise (`MANIFEST`, `units.from_manifest`), and the catalog's line says
why.  With `CIRCUIT_CHECK_RECORDS_MANIFEST`, the records the session's catalog rests on are listed there (`units.listing`).
Every process makes the catalog's roots itself, and each suite target the first time a test reads it (`Suite`), from its id
or from its binding's record (`units.make`).  A test that asks for it is marked slow, so `--quick` builds none; in a process
that didn't build it, `targets.suite` and `targets.catalog` fail by name."""
from __future__ import annotations

import fcntl
import functools
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
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
#: guard points at an empty directory of its own (`HOST_CACHES`).  The session reads the records there and writes the ones it
#: has built.  A record counts only for inputs identical to the ones it was made from (`units.plan`, as `check`'s
#: circuit-check takes it), so it changes how long the catalog takes, not what it holds.
RECORDS = Path(os.environ.get("CIRCUIT_CHECK_RECORDS") or Path.home() / ".cache" / "verity-check" / "circuit-check")
#: where the session lists the records its catalog rests on (`units.listing`, as ``circuit-check --records-manifest`` does):
#: what a store that carries the records to other runs names.  Unset, it lists none
LISTED = os.environ.get("CIRCUIT_CHECK_RECORDS_MANIFEST")

#: what each catalog binding reaches (`units.manifest`), written once a session by the xdist worker that builds the catalog
#: without records (`_apart`) into the directory every worker's basetemp is in (the controller's, which the session starts
#: empty), beside its lock
MANIFEST = "circuit-check-catalog.json"
#: there too: what the session's one build of the records reported (`_recorded`), or that every record counted and none ran
RECORDED = "circuit-check-records.json"
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
        raise _Missed(f"{e.name} has no record that counts (of the {e.skipped} under its key, each has a read that changed, "
                      "lists other roots than its binding makes now or is not the bytes its name digests)")
    return hit


def _from_records(cache: Path) -> tuple[UN.Plan | None, str | None]:
    """The catalog's plan from the bindings' records (`units.plan`) when every binding's counts; else None and why: there is
    no record under this process's keys, one doesn't count or one is stale.  Either way before any binding they miss is built here:
    `units.plan` builds such a binding in the process that calls it, which under xdist is every worker of the session at once,
    each holding what the heaviest binding's build holds.  So the first record that doesn't count ends the plan (`_counted`),
    and the one build the session shares (`_recorded`) makes them all.  A target is taken as its record describes it, as
    `cli._planned` takes a target whose result it reuses: holding each one to its descriptor (`cache.descriptor`, as
    `cli._made` does for one it checks) walks all it reaches."""
    with pytest.MonkeyPatch.context() as mp:
        # the records were made traced, so a function they read (a kernel's) is named by its module's digest, which
        # `cache._where` gives only under the tracer's root: without it, by path and line, and no such record counts
        mp.setattr(CC, "_ROOT", [cli.REPO.resolve()])
        mp.setattr(UN, "_lookup", functools.partial(_counted, UN._lookup))
        try:
            plan = UN.plan(cache, cli.REPO)
            return plan, None if plan is not None else (f"{cache} holds no record under this process's keys (this "
                                                        "circuit-check's version, `cache.version`, and its keyed variables)")
        except (UN.Stale, _Missed) as e:
            TG.BINDINGS.clear()
            return None, str(e)


def _catalog(plan: UN.Plan, how: str) -> Catalog:
    """``plan``'s catalog, each suite target made when it is read (`units.make`)."""
    d = plan.derived
    for fid in d.ids:
        TG.BINDINGS.setdefault(fid, set()).update(d.units.get(fid, set()))
    return Catalog(Suite(d.ids, lambda fid: UN.make(plan, fid)), {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()},
                   lambda fid: UN.make(plan, fid), how, plan.calls)


def _write(path: Path, got: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(got))
    os.replace(tmp, path)


def _once(shared: Path | None, name: str, do: Callable[[], dict]) -> dict:
    """What ``do`` returns, once a session.  Under xdist the first worker that asks runs it and writes that beside the
    workers' basetemps (``shared / name``), while the others wait on its lock and then take what it wrote; one that raised
    leaves its error there, and each later worker fails on it by name rather than run it again."""
    if shared is None:
        return do()
    path = shared / name
    with open(shared / (MANIFEST + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)    # held until the one that ran it wrote what it returned, or died
        if path.exists():
            got = json.loads(path.read_text())
            if "error" in got:
                raise RuntimeError(f"the xdist worker that built circuit-check's catalog failed: {got['error']}")
            return got
        try:
            got = do()
        except Exception as e:
            _write(path, {"error": f"{type(e).__name__}: {e}"})
            raise
        _write(path, got)
        return got


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
    built: list[tuple[list[UN.Entry], str]] = []

    def do() -> dict:
        kept, dropped, how = _apart_built(entries, jobs)
        built.append((kept, how))
        return UN.manifest(kept, dropped)

    got = _once(shared, MANIFEST, do)
    return built[0] if built else (UN.from_manifest(entries, got), "taken from the worker that built it")


def _apart(shared: Path | None, jobs: int, why: str | None = None) -> Catalog:
    """The catalog with each binding built in a process of its own (`units.built_apart`), once a session (`_built`), and no
    record: ``why`` the records gave none."""
    entries, how = _built(UN.made_entries(), shared, jobs)
    plan = UN.planned(entries)
    d = plan.derived
    for fid in d.ids:
        TG.BINDINGS.setdefault(fid, set()).update(d.units.get(fid, set()))
    return Catalog(Suite(d.ids, lambda fid: UN.make(plan, fid)), {fid: (n["f"], n["g"]) for fid, n in d.nodes.items()},
                   lambda fid: UN.make(plan, fid),
                   f"{how}, {len(entries)} bindings" + (f" (no record taken: {why})" if why else ""))


def _records_argv(cache: Path, jobs: int, mem_gb: int, out: Path, listed: str | None) -> list[str]:
    """The program that builds and stores the catalog bindings' records ``cache`` lacks, as `check`'s circuit-check stores
    them: ``circuit-check --all --records-only``, its report at ``out``."""
    return [sys.executable, "-m", "circuit_check", "--all", "--records-only", "--cache", str(cache), "--jobs", str(jobs),
            "--mem-gb", str(mem_gb), "--out", str(out), *(["--records-manifest", listed] if listed else [])]


def _recorded(cache: Path, jobs: int) -> dict:
    """Build every binding whose record in ``cache`` doesn't count, and store its record there: `_records_argv`'s program, in
    a process of its own, which traces what each build reads (this one, which imported its modules before any tracer, could
    not), within `_budget`.  Its report (`cli.records`: the bindings ``reused`` and ``rebuilt``, those of them ``skipped``, the
    records ``stored``, its ``seconds``), with ``why`` when it stored none or not all, or failed.  Under the suite's read
    guard it is a session fixture's process (``VERITY_SUITE_MODULE`` ``*``), so the guard traces what it reads, as it traces
    what this process's own builds read."""
    env = dict(os.environ)
    if env.get("VERITY_SUITE_READS") and not env.get("VERITY_SUITE_MODULE"):
        env["VERITY_SUITE_MODULE"] = "*"
    with tempfile.TemporaryDirectory(prefix="circuit-check-records-") as tmp:
        out = Path(tmp) / "records.json"
        r = subprocess.run(_records_argv(cache, jobs, max(1, int(_budget(os.environ))), out, LISTED), cwd=cli.REPO, env=env,
                           capture_output=True, text=True)
        said = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
        for ln in said if r.returncode else [ln for ln in said if ln.startswith("circuit-check")]:
            print(ln, flush=True)
        got = json.loads(out.read_text()) if out.is_file() else {}    # it writes its report last: none when it died
    if r.returncode and "why" not in got:
        got["why"] = f"`circuit-check --all --records-only` exited {r.returncode}" + (f": {said[-1][:300]}" if said else "")
    return got


def _listed(cache: Path, plan: UN.Plan) -> dict:
    """Every record counted, so nothing is built: the records ``plan`` rests on listed at `LISTED` (`units.listing`), and what
    `_recorded` would report."""
    if LISTED:
        UN.write_listing(Path(LISTED), UN.listing(cache, plan.entries))
    return {"reused": len(plan.entries), "rebuilt": []}


def _how(got: dict) -> str:
    """The catalog's line for what the session's records gave (`_recorded`'s report, or `_listed`'s), with the names of the
    variables the build's keys hold when it took no record though there were some (`units.untaken_note`)."""
    built, skipped = got["rebuilt"], got.get("skipped", [])
    how = f"from records: {got['reused']} of {got['reused'] + len(built)} bindings reused, {len(built)} built"
    if not built:
        return how
    return (f"{how} and stored in {got.get('seconds', '?')}s by one process for the session ({len(skipped)} of them held a "
            f"record that did not count): {', '.join(built[:3])}{', ...' if len(built) > 3 else ''}{UN.untaken_note(got)}")


def _taken(cache: Path, shared: Path | None, jobs: int) -> Catalog:
    """The catalog from the records in ``cache``: each one that counts taken, and every other binding built and its record
    stored there by the session's one build (`_recorded`), then taken too.  Where that leaves a binding without a record
    that counts, the catalog without records (`_apart`), saying why."""
    plan, why = _from_records(cache)
    got = _once(shared, RECORDED, lambda: _listed(cache, plan) if plan is not None else _recorded(cache, jobs))
    if plan is None and "why" not in got:
        plan, why = _from_records(cache)
    if plan is None:
        return _apart(shared, jobs, got.get("why") or why)
    if "why" in got:                 # this process took them all, whatever the session's build stored
        got = {"reused": len(plan.entries), "rebuilt": []}
    return _catalog(plan, _how(got))


def build() -> Catalog:
    """This process's catalog: built the first time it is asked for, from the records (`_taken`)."""
    if not _BUILT:
        _BUILT.append(_taken(RECORDS, _SHARED[0] if _SHARED else None, _jobs(os.environ)))
        print(_line(_BUILT[0]), flush=True)
    return _BUILT[0]


def _line(built: Catalog) -> str:
    return f"circuit-check's catalog: {built.how}, {len(built.suite)} targets"


#: in an xdist controller: the catalog's line of each worker that built one, whose own output the controller never shows
_LINES: list[str] = []


def pytest_sessionfinish(session):
    out = getattr(session.config, "workeroutput", None)
    if out is not None and _BUILT:
        out["circuit_check_catalog"] = _line(_BUILT[0])


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    if line := (getattr(node, "workeroutput", None) or {}).get("circuit_check_catalog"):
        _LINES.append(line)


def pytest_terminal_summary(terminalreporter):
    for line, n in Counter(_LINES).items():
        terminalreporter.write_line(f"{line} (in {n} xdist worker(s))")


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
