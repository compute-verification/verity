"""A measure for one run, on a scratch branch only: what each xdist worker of circuit-check's test suite holds.

A pytest plugin (`-p ccmem_peak`, from a copy in the run's own venv, which neither the suite's read guard nor circuit-check's
tracer counts as the tree's).  It patches nothing and no test sees it: at the session's hooks it reads this process's
/proc/self/status and its children's rusage, and appends one JSON row to $CCMEM_OUT/<worker>.jsonl.

    start        the process as the session starts, and what it was told (the slot's memory, the worker count, its CPUs)
    collected    before conftest.py makes the catalog (`pytest_collection_finish`)
    catalog      after it: how the catalog was made (`Catalog.how`), its targets, whether the records gave its Calls
    built_apart  `units.built_apart` starting in this process, with its `jobs` and `budget` (read from its frame as it starts:
                 `sys.monitoring`, a tool id circuit-check's tracer never takes), and the test that called it, if one did
    test         one test, setup to teardown: seconds, and before and after it this process's VmHWM and VmRSS and the largest
                 resident set of any process it started and waited for (`RUSAGE_CHILDREN`)
    end          the process as the session ends

Sizes are kB as the kernel prints them (KiB)."""
import json
import os
import resource
import sys
import time

import pytest

OUT = os.environ.get("CCMEM_OUT")
WORKER = os.environ.get("PYTEST_XDIST_WORKER") or "controller"
KEYS = ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem")
_NOW: list = [None]
_CODE: list = []


def _held() -> dict:
    got: dict = {}
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in KEYS:
                    got[k] = int(v.split()[0])
    except OSError:
        pass
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    got["kids_maxrss"] = kids.ru_maxrss
    got["kids_cpu_s"] = round(kids.ru_utime + kids.ru_stime, 2)
    return got


def _row(event: str, **more) -> None:
    if not OUT:
        return
    row = {"event": event, "worker": WORKER, "pid": os.getpid(), "t": round(time.time(), 3), **_held(), **more}
    try:
        with open(os.path.join(OUT, f"{WORKER}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def _catalog() -> dict:
    for name, mod in list(sys.modules.items()):
        if name.split(".")[-1] == "conftest" and hasattr(mod, "_BUILT") and hasattr(mod, "WORKER_GB"):
            built = mod._BUILT
            if not built:
                return {"built": False, "worker_gb": mod.WORKER_GB, "records": str(mod.RECORDS), "handed": bool(mod.HANDED)}
            c = built[0]
            return {"built": True, "how": c.how, "targets": len(c.suite), "calls": None if c.calls is None else len(c.calls),
                    "worker_gb": mod.WORKER_GB, "records": str(mod.RECORDS), "handed": bool(mod.HANDED)}
    return {"built": None}


def _started(code, offset):
    if not _CODE or code is not _CODE[0]:
        return None
    try:
        f = sys._getframe(1)
        for _ in range(6):
            if f is None or f.f_code is code:
                break
            f = f.f_back
        loc = f.f_locals if f is not None and f.f_code is code else {}
        entries, weights = loc.get("entries") or [], loc.get("weights") or {}
        counted = [weights.get(getattr(e, "name", None), 2.0) for e in entries]
        budget = loc.get("budget")
        _row("built_apart", test=_NOW[0], jobs=loc.get("jobs"),
             budget=budget if isinstance(budget, (int, float)) and budget == budget and abs(budget) != float("inf") else repr(budget),
             entries=len(entries), counted_sum=round(sum(counted), 1), counted_max=max(counted, default=None),
             slot_env=os.environ.get("CHECK_SLOT_MEM_GB"), workers=os.environ.get("PYTEST_XDIST_WORKER_COUNT"))
    except Exception as e:  # noqa: BLE001
        _row("built_apart", error=repr(e))
    return None


def _watch() -> str:
    mon, units = getattr(sys, "monitoring", None), sys.modules.get("circuit_check.units")
    if mon is None or units is None or not hasattr(units, "built_apart"):
        return "unwatched: no sys.monitoring, or circuit_check.units is not imported"
    for tool in (2, 1, 0):          # circuit-check's tracer takes one of 4, 3, 5
        if mon.get_tool(tool) is None:
            _CODE[:] = [units.built_apart.__code__]
            mon.use_tool_id(tool, "ccmem-peak")
            mon.register_callback(tool, mon.events.PY_START, _started)
            mon.set_local_events(tool, _CODE[0], mon.events.PY_START)
            return f"watched: tool {tool}"
    return "unwatched: no tool id is free"


def pytest_sessionstart(session):
    if not OUT:
        return
    try:
        watch = _watch()
    except Exception as e:  # noqa: BLE001
        watch = f"unwatched: {e!r}"
    _row("start", slot_env=os.environ.get("CHECK_SLOT_MEM_GB"), workers=os.environ.get("PYTEST_XDIST_WORKER_COUNT"),
         cpus=len(os.sched_getaffinity(0)), ppid=os.getppid(), watch=watch, python=sys.version.split()[0],
         threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_finish(session):
    _row("collected", items=len(session.items))
    outcome = yield
    _row("catalog", raised=repr(outcome.excinfo[1])[:300] if outcome.excinfo else None, **_catalog())


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    before, t0 = _held(), time.time()
    _NOW[:] = [item.nodeid, None]
    yield
    slot = _NOW[1] if len(_NOW) > 1 else None
    _NOW[:] = [None]
    _row("test", test=item.nodeid, t0=round(t0, 3), s=round(time.time() - t0, 3), slot_env=slot,
         before={k: before.get(k) for k in ("VmHWM", "VmRSS", "kids_maxrss", "kids_cpu_s")})


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    if len(_NOW) > 1:
        _NOW[1] = os.environ.get("CHECK_SLOT_MEM_GB")      # what a circuit-check this test starts is told (`worker_slot_gb`)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    _row("end", exitstatus=int(exitstatus), **_catalog())
