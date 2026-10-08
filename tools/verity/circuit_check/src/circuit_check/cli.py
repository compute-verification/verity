"""``circuit-check TARGET... [--out report.json]``: the standard property checks on each target, one machine-readable report.

    circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'     Definitions
    circuit-check rope-head gemm-coordinate/k64/sm80-mma-bf16       templates, subcircuits
    circuit-check path/to/build_request_LP3_T2                      a served row's request Program
    circuit-check --all                                             every registered Definition and template (the suite's set)

Exit status 1 when a check fails that `circuit_check.known` does not list (failures are named in the summary; ``--strict`` fails
on the known ones too), and with ``--all`` when a known entry no longer fails; 0 otherwise.  Redundant and dead gates are
warnings with counts, printed and in the report (``--fail-on-warnings`` makes them fail too).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import cache as CC
from . import checks as CK
from . import sidecars as SC
from . import targets as TG

REPO = Path(__file__).resolve().parents[5]


def _options(a: argparse.Namespace) -> CK.Options:
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras)


_TASKS: list[tuple] = []


def _info(task: tuple) -> dict:
    """A task's ``(kind, label, check[, info])``: info has ``as_call`` (in its key), ``units`` (its bindings, `targets.BINDINGS`)
    and ``roots`` (the objects it checks, built before its check)."""
    return task[3] if len(task) > 3 else {}


def _one(kind: str, label: str, fn: Callable[[], dict]) -> dict:
    t0 = time.time()
    reset_peak()
    start = _status_gb("VmRSS")
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
        r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                           "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
    return {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3), "peak_gb": peak_gb(),
            "rss_start_gb": start}


def _errored(r: dict) -> bool:
    """A check that raised: no verdict on the target, whatever broke (an import, the machine)."""
    return any(f.get("code") == "error" for f in r.get("failures", []))


def _checked(task: tuple) -> tuple[dict, dict | None]:
    """A task's result, and with tracing what its check read or resolved (`cache.attributed`), recorded from its start."""
    kind, label, fn = task[:3]
    if not CC.tracing():
        return _one(kind, label, fn), None
    info = _info(task)
    with CC.recording() as fr:
        r = _one(kind, label, fn)
    return r, CC.attributed(fr, info.get("units", ()), info.get("roots", ()))


def _observe(r: dict, got: dict | None) -> tuple[dict | None, list[str], dict | None]:
    """What a result read, what the tracer missed, and the reads as this process sees them right after the check."""
    untraced = CC.untraced()
    return got, untraced, (CC.seen(REPO, got) if got is not None and not untraced and not _errored(r) else None)


def _work(i: int) -> tuple[int, dict, dict | None, list[str], dict | None]:
    r, got = _checked(_TASKS[i])
    return i, r, *_observe(r, got)


def target_key(kind: str, label: str, opts: CK.Options, as_call: bool | None, roots: tuple = ()) -> dict:
    """A target's key: its request, and for a Definition the digest of what it reaches as descriptors (`cache.descriptor`)."""
    request = {"target": [kind, label], "as_call": as_call, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}}
    if roots:
        request["descriptor"] = CC.descriptor(roots[0])
    return CC.key(request)


def _tasks(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None) -> list[tuple]:
    """The targets' checks in target order (deterministic: a worker rebuilds the same list and runs entries by index).  Traced
    (`cache.tracing`), each check gets a memo of its own, so its result never rests on what an earlier target's check computed,
    and the pins it looks up are recorded (`cache.recorded_pins`); a suite Definition's ``as_call`` is resolved here (the catalog's
    `targets.call_families`), so it is in the target's key rather than a read of its check."""
    from verity.ml.circuits.evaluation.reference import resolve

    TG.load_registries()
    fresh = CC.tracing()
    shared: dict = {}
    memo = (lambda: {}) if fresh else (lambda: shared)
    pins = CC.recorded_pins(CK.load_pins()) if fresh else CK.load_pins()
    tasks: list[tuple] = []
    if everything:
        for fid, fn in TG.suite().items():
            call = TG.family(resolve(fn)) in TG.call_families()
            tasks.append(("definition", fid, lambda fn=fn, call=call: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=call),
                          {"as_call": call, "units": sorted(TG.BINDINGS.get(fid, ())), "roots": [fn]}))
        for label, sub, sts in TG.templates():
            tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins),
                          {"units": sorted(TG.BINDINGS.get(label, ()))}))
    for s in specs:
        kind = TG.kind_of(s)
        if kind == "row":
            tasks.append(("row", s, lambda s=s: CK.check_row(s, opts, memo=memo(), pins=pins)))
        elif kind == "template":
            def tpl(s=s):
                sub, sts = TG.subcircuit(s)
                return CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins)
            tasks.append(("template", s, tpl))
        else:
            tasks.append(("definition", s, lambda s=s: CK.check_definition(TG.definition(s), opts, memo=memo(), pins=pins, as_call=as_call)))
    return tasks


def _init(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None, tracing: bool) -> None:
    if tracing:
        CC.trace(REPO)
    _TASKS[:] = _tasks(specs, opts, everything, as_call)


def reset_peak() -> None:
    """Start this process's peak RSS (VmHWM) over from its current RSS (`clear_refs` 5), so `peak_gb` is the next target's."""
    try:
        Path("/proc/self/clear_refs").write_text("5")
    except OSError:
        pass


def _status_gb(field: str) -> float | None:
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith(field + ":"):
                return round(int(ln.split()[1]) / (1 << 20), 2)
    except OSError:
        pass
    return None


def peak_gb() -> float:
    """This process's peak RSS since `reset_peak`, in GiB (VmHWM; without /proc, ``ru_maxrss``: the peak since it started)."""
    if (g := _status_gb("VmHWM")) is not None:
        return g
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 2)


class Killed(dict):
    """What `parallel` yields for an item whose worker died (the OOM killer): ``item``, and the ``peak_gb`` its worker last
    sampled while on it."""


_STATUS: dict[str, Any] = {}


def _write_status(item: Any) -> None:
    d = _STATUS["dir"]
    tmp = d / f".{os.getpid()}"
    tmp.write_text(json.dumps({"item": item, "peak_gb": peak_gb()}))
    tmp.replace(d / str(os.getpid()))


def _worker_init(status: str, initializer: Callable | None, initargs: tuple) -> None:
    """A worker's start: its status file under ``status`` names the item it is on and its peak, sampled each second, so the
    parent can name what a killed worker was checking; on SIGTERM (the pool stopping its workers once another died) it removes
    the file and exits, so only a worker that died unasked leaves one."""
    import signal
    import threading

    _STATUS["dir"] = Path(status)

    def term(*_):
        (_STATUS["dir"] / str(os.getpid())).unlink(missing_ok=True)
        os._exit(1)

    signal.signal(signal.SIGTERM, term)

    def sample():
        while not threading.Event().wait(1.0):
            if "item" in _STATUS:
                _write_status(_STATUS["item"])

    threading.Thread(target=sample, daemon=True).start()
    if initializer is not None:
        initializer(*initargs)


def _tracked(fn: Callable, item: Any) -> Any:
    _STATUS["item"] = item
    reset_peak()
    _write_status(item)
    try:
        return fn(item)
    finally:
        _STATUS.pop("item", None)
        (_STATUS["dir"] / str(os.getpid())).unlink(missing_ok=True)


def parallel(fn: Callable, items: list, jobs: int, initializer: Callable | None = None, initargs: tuple = (), *,
             weight: Callable[[Any], float] | None = None, budget: float | None = None, fresh: bool = False):
    """``fn`` of each item in up to ``jobs`` forkserver workers, yielded as each finishes, in the order given.  With ``weight``
    and ``budget``, an item starts only while the weights of the items running and its own add up to at most ``budget`` (an item
    heavier than ``budget`` runs alone); with ``fresh`` each item gets a new worker process (``max_tasks_per_child=1``), so what
    one item's check left resident is never under the next's.  A worker that dies (the OOM killer) yields a `Killed` for the
    item it was on, the pool is restarted, and the items it interrupted run again; ``multiprocessing.Pool`` would wait for a
    dead worker's task forever.  A death no status file accounts for raises ``BrokenProcessPool``."""
    import multiprocessing as mp
    import shutil
    import tempfile
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    weigh = weight or (lambda _: 0.0)
    limit = float("inf") if budget is None else budget
    pending = list(items)
    while pending:
        status = Path(tempfile.mkdtemp(prefix="circuit-check-workers-"))
        try:
            with ProcessPoolExecutor(min(jobs, len(pending)), mp_context=mp.get_context("forkserver"), initializer=_worker_init,
                                     initargs=(str(status), initializer, initargs),
                                     max_tasks_per_child=1 if fresh else None) as pool:
                queue, running = list(pending), {}
                while queue or running:
                    while queue and len(running) < jobs and (not running or sum(w for _, w in running.values()) + weigh(queue[0]) <= limit):
                        x = queue.pop(0)
                        running[pool.submit(_tracked, fn, x)] = (x, weigh(x))
                    finished, _ = wait(running, return_when=FIRST_COMPLETED)
                    for f in finished:
                        r = f.result()
                        pending.remove(running.pop(f)[0])
                        yield r
        except BrokenProcessPool:
            dead = [json.loads(p.read_text()) for p in status.iterdir() if not p.name.startswith(".")]
            if not dead:
                raise
            for d in dead:
                pending = [x for x in pending if x != d["item"]]
                yield Killed(d)
        finally:
            shutil.rmtree(status, ignore_errors=True)


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        cache: Path | None = None, mem_gb: int | None = None, progress: Path | None = None) -> dict:
    """Every target's checks, in target order; ``jobs`` > 1 runs them in that many worker processes, each started from a fresh
    forkserver (never a fork of this possibly multi-threaded process: under pytest-xdist that deadlocked) and rebuilding the
    task list with its own memo, so a worker may repeat a Definition another checked: the results are the same.

    With ``cache`` each target's result is reused while everything its check read or resolved is unchanged
    (`circuit_check.cache`): what was read and run from the start of its check, the builds of the Definitions it reached, its
    own bindings and pins, and those modules' import closures; not what the task list's build or another target read.  A result
    is not stored once its process did something the tracer cannot see, nor when the check raised (no verdict), nor when the
    reads as that process sees them right after its check differ from what would be recorded (the tree changed during the run).

    Each result has ``peak_gb``, the RSS peak of the process while it checked that target (`peak_gb`), and ``rss_start_gb``, its
    RSS when the check started (what earlier checks in the same worker left resident, which the peak includes); the report's ``peaks``
    lists the heaviest of the targets run now, and ``progress`` (JSON lines) gets a line per target as it finishes, so a run that
    is itself killed still leaves them.

    With ``jobs`` > 1 the workers' memory stays under ``mem_gb`` (default: what `memory_gb` leaves this process) as long as
    every target that peaks at `HEAVY_GB` or more is listed in `HEAVY` with its peak: the unlisted targets run first, in at most
    ``mem_gb // HEAVY_GB`` workers; then the listed ones, heaviest first, each in a fresh worker (nothing an earlier check left
    resident is under it), started only while the listed peaks of those running add up to at most ``mem_gb`` (one heavier than
    that runs alone).  An unlisted target that peaks over `HEAVY_GB` can still break the bound: the worker it gets killed in is
    reported as that target's failure (``circuit-check/killed``, with the peak last sampled), the targets the dead pool
    interrupted run again, and its peak belongs in `HEAVY`."""
    tracing = cache is not None and CC.tracing()
    tasks = _tasks(specs, opts, everything, as_call)
    out: list[dict | None] = [None] * len(tasks)
    keys = [target_key(t[0], t[1], opts, _info(t).get("as_call", as_call), tuple(_info(t).get("roots", ()))) for t in tasks] \
        if tracing else []
    stats = {"hits": 0, "stored": 0, "untraced": [], "errors": [], "changed": []}

    def done(i: int, r: dict, got: dict | None = None, untraced: list[str] | None = None, worker: dict | None = None) -> None:
        out[i] = r
        if progress is not None:
            with progress.open("a") as f:
                f.write(json.dumps({"kind": r["kind"], "target": r["target"], "peak_gb": r.get("peak_gb"), "rss_start_gb": r.get("rss_start_gb"), "seconds": r.get("seconds"),
                                    "failures": len(r["failures"]), **({"killed": True} if r.get("killed") else {})}) + "\n")
        if tracing and got is not None:
            if _errored(r):
                stats["errors"].append(r["target"])
            elif untraced:
                stats["untraced"].append({"target": r["target"], "because": untraced[:3]})
            elif CC.store(cache, keys[i], REPO, got, r, worker) is None:
                stats["changed"].append(r["target"])
            else:
                stats["stored"] += 1
        if log:
            across = (r.get("partition") or {}).get("recomputed_across")
            qc = (r.get("partition") or {}).get("q_call") or {}
            call = (f" q_call={qc['units']}u/{qc['committed_bits']}b" if "units" in qc else
                    f" q_call=refused:{qc['inapplicable'].get('why')}" if "inapplicable" in qc else "")
            log(f"{'FAIL' if r['failures'] else 'ok  '} {r['kind']:10s} {r['target'][:90]}  failures={len(r['failures'])} "
                f"warnings={len(r['warnings'])}" + (f" recomputed_across={across}" if across else "") + call + f" ({r['seconds']:.1f}s)")

    todo = list(range(len(tasks)))
    if tracing:
        todo = []
        for i, k in enumerate(keys):
            if (hit := CC.lookup(cache, k, REPO)) is not None and not _errored(hit[0]):
                out[i] = {**hit[0], "cache": {"hit": hit[1]}}
                stats["hits"] += 1
            else:
                todo.append(i)
    if jobs > 1 and len(todo) > 1:
        init = (specs, opts, everything, as_call, tracing)
        budget = memory_gb() if mem_gb is None else mem_gb
        peaks = heavy()
        heavy_todo = sorted((i for i in todo if tasks[i][1] in peaks), key=lambda i: -peaks[tasks[i][1]])
        light = [i for i in todo if tasks[i][1] not in peaks]
        light_jobs = max(1, min(jobs, budget // HEAVY_GB))
        for phase in (parallel(_work, light, light_jobs, _init, init),
                      parallel(_work, heavy_todo, jobs, _init, init, weight=lambda i: peaks[tasks[i][1]], budget=budget, fresh=True)):
            for got_ in phase:
                if isinstance(got_, Killed):
                    i = got_["item"]
                    r = {"failures": [{"check": "circuit-check", "code": "killed", "definition": tasks[i][1],
                                       "name": f"circuit-check/killed {tasks[i][1]}: its worker died (the OOM killer?) at a "
                                               f"sampled peak of {got_['peak_gb']} GiB"}],
                         "warnings": [], "kind": tasks[i][0], "target": tasks[i][1], "peak_gb": got_["peak_gb"], "killed": True}
                    done(i, r)
                else:
                    done(*got_)
    else:
        for i in todo:
            r, got = _checked(tasks[i]) if tracing else (_one(*tasks[i][:3]), None)
            done(i, r, *(_observe(r, got) if tracing else (None, None, None)))
    rep = CK.report([r for r in out if r is not None], opts)
    ran = [r for r in rep["targets"] if r.get("peak_gb") is not None and "cache" not in r]
    rep["peaks"] = [{"target": r["target"], "peak_gb": r["peak_gb"], "rss_start_gb": r.get("rss_start_gb"), **({"killed": True} if r.get("killed") else {})}
                    for r in sorted(ran, key=lambda r: -r["peak_gb"])[:PEAKS_REPORTED]]
    if tracing:
        rep["cache"] = stats
    return rep


def update_pins(rep: dict) -> dict:
    """Every pin, with each unpinned or changed count the report saw set to the count measured now, in the file that holds the pin
    or, for a new one, the sidecar of the module that defines its Definition (`sidecars.write_pins`)."""
    updates = {}
    for t in rep["targets"]:
        for f in t.get("failures", []) + t.get("warnings", []):
            if f["check"] == "lowering" and f["code"] in ("unpinned", "pin-mismatch"):
                updates[f["pins"], f["pin"]] = f["counts"]
    CK.load_pins()
    SC.write_pins(updates)
    return CK.load_pins()


#: the targets that peaked at `HEAVY_GB` or more, each with its recorded peak in GiB (`run`'s scheduling; measured by an
#: `--all` run's report, ``peaks``)
HEAVY = Path(__file__).with_name("heavy.json")
HEAVY_GB = 16
PEAKS_REPORTED = 50


def heavy() -> dict[str, float]:
    return json.loads(HEAVY.read_text()) if HEAVY.exists() else {}


#: the memory a worker counts on: a cold `--all` grew six of 23 workers on vy-nebius-cpu-1 to 40-136 GiB (7 Oct), and two
#: `--all -j 0` runs, ~130 workers each, took 1.4 TB of node 1's 1.6 TB user slice (8 Oct)
GB_PER_JOB = 128


def memory_gb(root: Path = Path("/")) -> int:
    """The memory this process may use: the least of MemAvailable and what each cgroup v2 limit over it leaves (a run's session
    scope and the user slice on a VM; a pod's /proc/meminfo is the host's)."""
    def read(p: Path) -> str:
        try:
            return p.read_text().strip()
        except OSError:
            return ""

    left = [int(ln.split()[1]) << 10 for ln in read(root / "proc/meminfo").splitlines() if ln.startswith("MemAvailable:")]
    top = root / "sys/fs/cgroup"
    for line in read(root / "proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            d = top / line[3:].strip("/")
            for c in (d, *[p for p in d.parents if p.is_relative_to(top)]):
                if (m := read(c / "memory.max")).isdigit():
                    left.append(int(m) - int(read(c / "memory.current") or 0))
    return max(0, min(left)) >> 30 if left else 0


def auto_jobs(root: Path = Path("/")) -> int:
    """`--jobs 0`: one worker per core this process may run on, at most one per `GB_PER_JOB` of its memory."""
    return max(1, min(len(os.sched_getaffinity(0)), memory_gb(root) // GB_PER_JOB))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="circuit-check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="Definition ids, template or subcircuit ids, request Program paths")
    ap.add_argument("--all", action="store_true", help="every registered Definition and template at the catalog's bindings")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="targets checked in parallel by this many workers (0: one per core, at most one per GB_PER_JOB of memory)")
    ap.add_argument("--mem-gb", type=int, default=None, help="the memory the workers may take together (default: what this "
                    "process may use, `memory_gb`); see `run`")
    ap.add_argument("--vectors", type=int, default=CK.Options.vectors)
    ap.add_argument("--seed", type=int, default=CK.Options.seed)
    ap.add_argument("-X", type=int, default=CK.Options.X)
    ap.add_argument("-W", type=int, default=CK.Options.W)
    ap.add_argument("--extras", type=int, default=CK.Options.extras)
    ap.add_argument("--as-call", action="store_true", help="hold a Definition target to the partition invariant as a Call "
                    "(default: when its family is a Call of a served Program)")
    ap.add_argument("--strict", action="store_true", help="fail on the known failures too (circuit_check.known)")
    ap.add_argument("--fail-on-warnings", action="store_true")
    ap.add_argument("--update-pins", action="store_true", help="pin the lowering counts measured now (review the diff)")
    ap.add_argument("--cache", type=Path, default=None, metavar="DIR",
                    help="reuse a report whose every input is unchanged (circuit_check.cache), else run and record one")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)
    if a.cache and not a.update_pins:
        # the registries build their C++ twins on first use (g++, untraceable): once, in a process of its own, before tracing
        prebuild = ("from circuit_check import targets; targets.load_registries()\n"
                    "try:\n    from verity_vllm.program.kernels import twins\nexcept ImportError:\n    twins = None\n"
                    "if twins:\n    twins.preload()        # the three C++ models, which checks otherwise compile on first use\n")
        subprocess.run([sys.executable, "-c", prebuild], cwd=REPO, capture_output=True)
        CC.trace(REPO)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    jobs = a.jobs or auto_jobs()
    opts = _options(a)
    rep = run(a.targets, opts, everything=a.all, as_call=True if a.as_call else None,
              log=None if a.quiet else lambda s: print(s, flush=True), jobs=jobs, cache=a.cache if a.cache and not a.update_pins else None,
              mem_gb=a.mem_gb, progress=a.out.with_name(a.out.name + ".progress.jsonl") if a.out else None)
    if c := rep.get("cache"):
        print(f"circuit-check: {c['hits']} of {len(rep['targets'])} targets reused (every file, listing, probe, catalog unit and pin their checks read or resolved is "
              f"unchanged), {c['stored']} stored, {len(c['untraced'])} not cacheable (the tracer could not see what they read), "
              f"{len(c['errors'])} not stored because the check raised, {len(c['changed'])} because the tree changed while they ran")
    if a.update_pins:
        update_pins(rep)
    s = rep["summary"]
    stale = []
    if a.all:
        from .known import stale as stale_known

        stale = stale_known(f for t in rep["targets"] if t["kind"] == "definition" for f in t.get("failures", []))
        s["stale_known"] = stale
    for name in s["new_failures"]:
        print(f"FAIL {name}")
    for name in s["known_failures"]:
        print(f"known {name}")
    for k in stale:
        print(f"FAIL circuit_check.known lists {k}, which no longer fails: drop it")
    for k, w in s["warnings"].items():
        print(f"warn {k}: {w['total']} over {w['definitions']} target(s)")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    across = s.get("recomputed_across") or {}
    refused = s.get("q_call_refused") or {}
    print(f"circuit-check: {len(rep['targets'])} target(s), {len(s['new_failures'])} new failure(s), {len(s['known_failures'])} known"
          + (f", {len(across)} Boolean Call(s) recomputing {sum(across.values())} gate(s) across units (Q_word v2)" if across else "")
          + (f", {len(refused)} Definition(s) Q_call v1 refuses" if refused else "")
          + (f", report {a.out}" if a.out else ""))
    failed = bool(s["new_failures"] or stale or (a.strict and s["known_failures"]) or (a.fail_on_warnings and s["warnings"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
