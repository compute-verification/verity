"""A probe for one run, on a scratch branch only, inside a real session of circuit-check's test suite: the pytest suites.py
starts, in each xdist worker, once collection is done and before conftest builds any catalog (`-p ccmem_plan`).

    CCMEM_KEYED=DIR   write DIR/<worker>.json: the variables of this process that key a circuit-check record as conftest's
                      `_from_records` looks one up (`cache.key`'s environment, less what conftest leaves out of it)
    CCMEM_ORDER=DIR   load the registries as `units.plan` does and write DIR/<worker>.json: the registered roots' ids in this
                      process's order, which the records of `<registered>:0..7` are cut from
    CCMEM_PLAN=1      call conftest's own `_from_records` and write a row to $CCMEM_OUT/plan.<worker>.jsonl: what it gave, how
                      many lookups hit, and for the first binding whose record did not count, why

Select no test that asks for the catalog: after a miss conftest builds it apart, for long.  It patches `units._lookup` and
`units.plan` for the length of the one call, to see what they return."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

OUT = os.environ.get("CCMEM_OUT")
KEYED = os.environ.get("CCMEM_KEYED")
ORDER = os.environ.get("CCMEM_ORDER")
PLAN = os.environ.get("CCMEM_PLAN") == "1"
WORKER = os.environ.get("PYTEST_XDIST_WORKER")


def _conftest():
    for name, mod in list(sys.modules.items()):
        if name.split(".")[-1] == "conftest" and hasattr(mod, "_from_records") and hasattr(mod, "RECORDS"):
            return mod
    return None


def _left_out(CC) -> set:
    return CC.UNKEYED | {k for k in os.environ if k.startswith("VERITY_SUITE_")} | {"CIRCUIT_CHECK_RECORDS"}


def _keyed(CC) -> dict:
    out = _left_out(CC)
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(("VERITY_", "CIRCUIT_CHECK_")) and k not in out}


def _differs(UN, kind, a, b, had_a, had_b) -> bool:
    if not (had_a and had_b):
        return True
    if kind in ("modules", "units"):
        return UN._sited(a) != UN._sited(b)
    if kind == "present":
        return (a is None) != (b is None)
    if kind == "enums":
        return (a is None) != (b is None) or (a is not None and a.keys() != b.keys())
    return a != b


def _key_diff(CC, cache, key) -> dict:
    """Where no record is under ``key``: whether its version is there, and a stored record of the same request with another key."""
    vd = Path(cache) / key["version"]
    out = {"version_dir_there": vd.is_dir(), "environment_names_now": sorted(key.get("environment", {}))}
    if not vd.is_dir():
        out["versions_there"] = len([p for p in Path(cache).iterdir() if p.is_dir()]) if Path(cache).is_dir() else None
        return out
    n = 0
    for d in sorted(vd.iterdir()):
        if not d.is_dir() or d.name == getattr(CC, "READS", "reads"):
            continue
        for p in sorted(d.glob("*.json"))[:1]:
            n += 1
            try:
                k = json.loads(p.read_text()).get("key")
            except Exception:  # noqa: BLE001
                continue
            if isinstance(k, dict) and k.get("request") == key.get("request"):
                was, now = k.get("environment", {}), key.get("environment", {})
                out.update(scanned=n, same_request_stored={
                    "fields_that_differ": sorted(f for f in set(k) | set(key) if k.get(f) != key.get(f)),
                    "environment_names_only_stored": sorted(set(was) - set(now)),
                    "environment_names_only_now": sorted(set(now) - set(was)),
                    "environment_names_whose_values_differ": sorted(x for x in set(was) & set(now) if was[x] != now[x])})
                return out
    out.update(scanned=n, same_request_stored=None)
    return out


def _why(CC, UN, cache, repo, e) -> dict:
    got = {"binding": e.name, "kind": e.kind}
    with CC.internal():
        found = sorted(CC._dir(cache, e.key).glob("*.json"))
    got["records_under_its_key"] = len(found)
    if not found:
        got["why"] = "no record under its key"
        got["key"] = _key_diff(CC, cache, e.key)
        return got
    reasons = []
    for p in found[:3]:
        try:
            rec = CC.read(p)
            if rec.get("key") != e.key:
                reasons.append({"why": "the record in its directory holds another key"})
                continue
            reads = rec["reads"]
            now = CC._digests(repo, {k: list(v) for k, v in reads.items()})
            if not UN._unchanged(reads, now):
                changed = []
                for kind, v in reads.items():
                    if UN._unchanged({kind: v}, now):
                        continue
                    nv = now.get(kind)
                    if isinstance(v, dict) and isinstance(nv, dict):
                        for n in sorted(set(v) | set(nv), key=str):
                            if _differs(UN, kind, v.get(n), nv.get(n), n in v, n in nv):
                                changed.append({"read": f"{kind}: {n}", "recorded": str(v.get(n, "<not in the record>"))[:240],
                                                "now": str(nv.get(n, "<not there now>"))[:240]})
                    else:
                        changed.append({"read": f"{kind}: <whole>", "recorded": str(v)[:240], "now": str(nv)[:240]})
                reasons.append({"why": "a read changed", "changed": len(changed), "first": changed[:8]})
                continue
            rep = rec["report"]
            if rep.get("roots") != e.root_ids:
                reasons.append({"why": "its roots differ", "recorded": str(rep.get("roots"))[:300], "now": str(e.root_ids)[:300]})
                continue
            if not isinstance(rep.get("nodes"), dict):
                reasons.append({"why": "it holds no nodes"})
                continue
            for fid in e.root_ids:
                rep["nodes"][fid]["k"]
            reasons.append({"why": "it counts as this probe reads it, where units._lookup said it does not"})
        except Exception as x:  # noqa: BLE001
            reasons.append({"why": f"unreadable: {type(x).__name__}: {str(x)[:200]}"})
    got["records"] = reasons
    return got


def _trial(mod) -> dict:
    from circuit_check import cache as CC
    from circuit_check import units as UN
    notes: dict = {"records": str(mod.RECORDS), "handed": bool(mod.HANDED), "lookups": 0, "hits": 0, "first_miss": None,
                   "keyed_names": sorted(_keyed(CC)), "cwd": os.getcwd(), "safe_path": bool(sys.flags.safe_path),
                   "optimize": sys.flags.optimize, "modules": len(sys.modules)}
    lookup, plan = UN._lookup, UN.plan

    def looked(cache, repo, e):
        notes["lookups"] += 1
        hit = lookup(cache, repo, e)
        if hit is not None:
            notes["hits"] += 1
        elif notes["first_miss"] is None:
            try:
                notes["first_miss"] = _why(CC, UN, cache, repo, e)
            except Exception as x:  # noqa: BLE001
                notes["first_miss"] = {"binding": getattr(e, "name", None), "probe_error": repr(x)[:300]}
        return hit

    def planned(*a, **k):
        try:
            return plan(*a, **k)
        except Exception as x:
            notes["plan_raised"] = f"{type(x).__name__}: {str(x)[:500]}"
            raise

    UN._lookup, UN.plan, t0 = looked, planned, time.time()
    try:
        got = mod._from_records(mod.RECORDS, False)
        notes["gave"] = None if got is None else {"how": got.how, "targets": len(got.suite),
                                                  "calls": None if got.calls is None else len(got.calls)}
    except Exception as x:  # noqa: BLE001
        notes["raised"] = f"{type(x).__name__}: {str(x)[:500]}"
    finally:
        UN._lookup, UN.plan = lookup, plan
    notes["seconds"] = round(time.time() - t0, 1)
    if notes.get("gave") is None and notes["lookups"] == 0 and "raised" not in notes:
        # no lookup at all: `units.plan` found no directory under the first key it makes
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(CC, "UNKEYED", _left_out(CC))
            first = UN.key(UN.REGISTERED, 0)
            notes["first_key"] = {"dir_there": CC._dir(mod.RECORDS, first).is_dir(), **_key_diff(CC, mod.RECORDS, first)}
    return notes


def _order() -> None:
    """The registered roots' ids in this process's order, once the registries are loaded as `units.plan` loads them."""
    from circuit_check import targets as TG
    TG.load_registries()
    os.makedirs(ORDER, exist_ok=True)
    with open(os.path.join(ORDER, f"{WORKER}.json"), "w") as f:
        json.dump([getattr(r, "id", repr(r)[:80]) for r in TG._registered()], f)


def _probe() -> None:
    from circuit_check import cache as CC
    if ORDER:
        _order()
    if KEYED:
        os.makedirs(KEYED, exist_ok=True)
        with open(os.path.join(KEYED, f"{WORKER}.json"), "w") as f:
            json.dump(_keyed(CC), f)
    if PLAN and OUT:
        mod = _conftest()
        row = {"event": "plan", "worker": WORKER, "pid": os.getpid(), "t": round(time.time(), 3),
               **(_trial(mod) if mod is not None else {"error": "conftest is not imported"})}
        with open(os.path.join(OUT, f"plan.{WORKER}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_finish(session):
    if WORKER and (KEYED or PLAN or ORDER):
        try:
            _probe()
        except Exception as x:  # noqa: BLE001
            if OUT:
                with open(os.path.join(OUT, f"plan.{WORKER}.jsonl"), "a") as f:
                    f.write(json.dumps({"event": "plan", "worker": WORKER, "error": repr(x)[:500]}) + "\n")
    yield
