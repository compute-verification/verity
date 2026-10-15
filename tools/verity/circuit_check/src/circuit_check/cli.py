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
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras, lower_max_ands=a.lower_max_ands)


_TASKS: list[tuple] = []


def _info(task: tuple) -> dict:
    """A task's ``(kind, label, check[, info])``: info has ``as_call`` (in its key), ``units`` (its bindings, `targets.BINDINGS`)
    and ``roots`` (the objects it checks, built before its check)."""
    return task[3] if len(task) > 3 else {}


def _one(kind: str, label: str, fn: Callable[[], dict]) -> dict:
    t0 = time.time()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
        r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                           "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
    return {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3)}


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
    if as_call is None:
        TG.call_families()            # what a Definition's check would otherwise build in each target's process (every template)


def _rss_gb() -> float | None:
    """This process's resident set now, in GiB (``VmRSS``; None without /proc)."""
    try:
        for ln in Path("/proc/self/status").read_text().splitlines():
            if ln.startswith("VmRSS:"):
                return round(int(ln.split()[1]) / (1 << 20), 2)
    except OSError:
        pass
    return None


def _maxrss_gb(ru_maxrss: int) -> float:
    return round(ru_maxrss / (1 << 30 if sys.platform == "darwin" else 1 << 20), 2)


class Killed(dict):
    """What `parallel` yields in place of an item's result when its process died before returning one (the OOM killer):
    ``how`` it died (``signal 9``, ``exit 3``)."""


def _child(w: int, fn: Callable, item: Any) -> None:
    """A forked child: ``fn(item)``, pickled down ``w``, and exit.  Its start RSS is the zygote's at the fork (shared with it
    copy-on-write)."""
    import pickle

    start = _rss_gb()
    try:
        data = pickle.dumps((fn(item), start))
    except BaseException as e:  # noqa: BLE001  (the zygote reports what the item raised, never the child's stack unwinding into its loop)
        data = pickle.dumps((Killed(how=f"raised {type(e).__name__}: {str(e)[:200]}"), start))
    view = memoryview(data)
    while view:
        view = view[os.write(w, view):]
    os._exit(0)


def _zygote(conn, fn: Callable, initializer: Callable | None, initargs: tuple) -> None:
    """`parallel`'s one worker: ``initializer`` once, then a forked child per item ``conn`` sends, each running that item alone and
    exiting.  Each child is reaped with ``wait4``, whose ``ru_maxrss`` is that child's own peak RSS even when it was killed, and
    the zygote knows which item each child had, so a death is named by its item."""
    import gc
    import pickle
    import selectors
    import warnings

    if initializer is not None:
        initializer(*initargs)
    gc.collect()
    gc.freeze()                       # the children's collections then leave the task list's pages shared
    sel = selectors.DefaultSelector()
    sel.register(conn, selectors.EVENT_READ)
    kids: dict[int, tuple[int, int, bytearray]] = {}         # read end -> (pid, ticket, what the child wrote)
    while True:
        for key, _ in sel.select():
            if key.fileobj is conn:
                try:
                    msg = conn.recv()
                except EOFError:
                    msg = None
                if msg is None:
                    for pid, _, _ in kids.values():
                        os.kill(pid, 9)
                        os.waitpid(pid, 0)
                    return
                ticket, item = msg
                r, w = os.pipe()
                with warnings.catch_warnings(), CC.internal():
                    warnings.simplefilter("ignore", DeprecationWarning)        # the numeric libraries' idle threads
                    pid = os.fork()                                            # the child keeps the tracer's hook
                if pid == 0:
                    os.close(r)
                    _child(w, fn, item)
                os.close(w)
                kids[r] = (pid, ticket, bytearray())
                sel.register(r, selectors.EVENT_READ)
                continue
            r = key.fileobj
            pid, ticket, buf = kids[r]
            if chunk := os.read(r, 1 << 20):
                buf += chunk
                continue
            sel.unregister(r)
            os.close(r)
            del kids[r]
            _, status, usage = os.wait4(pid, 0)
            if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0 and buf:
                value, start = pickle.loads(buf)
            else:
                how = f"signal {os.WTERMSIG(status)}" if os.WIFSIGNALED(status) else f"exit {os.WEXITSTATUS(status)}"
                value, start = Killed(how=how), None
            conn.send((ticket, value, _maxrss_gb(usage.ru_maxrss), start))


def admit(queued: list[float], running: list[float], jobs: int, budget: float) -> list[int]:
    """Which of the queued items (by their weights, in order) to start beside those running: each in turn that fits, while fewer
    than ``jobs`` run and the weights add up to at most ``budget``; with nothing running, the first starts whatever its weight."""
    used, n, picked = sum(running), len(running), []
    for k, w in enumerate(queued):
        if n >= jobs:
            break
        if n == 0 or used + w <= budget:
            picked.append(k)
            used, n = used + w, n + 1
    return picked


def parallel(fn: Callable, items: list, jobs: int, initializer: Callable | None = None, initargs: tuple = (), *,
             weight: Callable[[Any], float] | None = None, budget: float | None = None):
    """``(item, fn(item), peak_gb, start_gb)`` for each item, yielded as each finishes, each item run alone in a process of its
    own: up to ``jobs`` at once, forked from one zygote process that ran ``initializer(*initargs)`` (started by forkserver, never
    a fork of this possibly multi-threaded process: under pytest-xdist that deadlocked).  ``peak_gb`` is that process's peak RSS
    (``ru_maxrss``, GiB): the item's own, plus ``start_gb``, the RSS it was forked with, which it shares with the zygote (what
    ``initializer`` built).  An item whose process died (the OOM killer) yields a `Killed` for ``fn(item)``, with the peak it
    reached; the others run on.  With ``weight`` and ``budget`` an item starts only while the weights of the items running and its
    own add up to at most ``budget`` (an item heavier than ``budget`` runs alone), taken in the order given, the first that fits."""
    import multiprocessing as mp

    weigh = weight or (lambda _: 0.0)
    limit = float("inf") if budget is None else budget
    ctx = mp.get_context("forkserver")
    here, there = ctx.Pipe()
    zygote = ctx.Process(target=_zygote, args=(there, fn, initializer, initargs), daemon=True)
    zygote.start()
    there.close()
    queue = list(enumerate(items))
    running: dict[int, tuple[Any, float]] = {}
    try:
        while queue or running:
            for k in admit([weigh(x) for _, x in queue], [w for _, w in running.values()], jobs, limit):
                t, x = queue[k]
                here.send((t, x))
                running[t] = (x, weigh(x))
            queue = [q for q in queue if q[0] not in running]
            try:
                t, value, peak, start = here.recv()
            except EOFError:
                raise RuntimeError(f"circuit-check's zygote died (exit {zygote.exitcode}) while running "
                                   f"{[x for x, _ in running.values()]}") from None
            yield running.pop(t)[0], value, peak, start
    finally:
        try:
            here.send(None)
        except OSError:
            pass
        here.close()
        zygote.join()


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        cache: Path | None = None, mem_gb: int | None = None, progress: Path | None = None, isolate: bool | None = None) -> dict:
    """Every target's checks, in target order.  Isolated (``isolate``; by default when ``jobs`` > 1), each target is checked in a
    process of its own, up to ``jobs`` at once (`parallel`): forked from one process that built the task list itself with its own
    memo, so a target's check never reuses what another's computed: the results are the same as checking them here, in turn.

    With ``cache`` each target's result is reused while everything its check read or resolved is unchanged
    (`circuit_check.cache`): what was read and run from the start of its check, the builds of the Definitions it reached, its
    own bindings and pins, and those modules' import closures; not what the task list's build or another target read.  A result
    is not stored once its process did something the tracer cannot see, nor when the check raised (no verdict), nor when the
    reads as that process sees them right after its check differ from what would be recorded (the tree changed during the run).

    Isolated, each result has ``peak_gb``: the peak RSS of the process that checked that target and nothing else (its
    ``ru_maxrss``, in GiB, so also when it was killed), and ``rss_start_gb``: the RSS it was forked with, the task list it shares
    copy-on-write with the process it was forked from, which the peak includes; ``peak_gb - rss_start_gb`` is what the check itself
    grew.  Checked here, in turn, a result has neither (earlier targets' leftovers would be under its peak).  The report's
    ``peaks`` lists the heaviest of the targets checked now, and ``progress`` (JSON lines) gets a line per target as it finishes,
    so a run that is itself killed still leaves them.

    Isolated, circuit-check's memory stays under ``mem_gb`` GiB (default: what `memory_gb` leaves this process) as long as every
    target that peaks over `HEAVY_GB` is listed in `HEAVY` with its peak: a target starts only while the peaks of those running
    and its own add up to at most `budget_gb` (``mem_gb`` less this process's RSS), counting an unlisted target as `HEAVY_GB`;
    one heavier than that still runs, alone, and the log says it does not fit; the
    listed targets go first, heaviest first.  The sum counts each process's copy of the task list, which they share, so the bound
    is conservative by that much per process.  An unlisted target that peaks well over `HEAVY_GB` can still break it: the process
    it gets killed in is reported as that target's failure (``circuit-check/killed``, with its peak), the other targets run on,
    and its peak belongs in `HEAVY`.

    The task list (each target and its key) is a cache entry too, under the catalog's own key: while everything its build
    could depend on is unchanged (`cache.task_list_reads`), a run whose every target is reused as it was recorded builds no
    Definition. Any other run builds the list."""
    tracing = cache is not None and CC.tracing()
    listed = None
    if tracing and everything and not specs:
        TG.load_registries()
        listed = CC.key({"target": ["task-list", "--all"], "as_call": as_call,
                         "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}})
        if (hit := CC.lookup(cache, listed, REPO, moved=False)) is not None:
            got = [CC.lookup(cache, k, REPO, moved=False) for k in hit[0]["keys"]]
            if all(g is not None and not _errored(g[0]) for g in got):
                rep = CK.report([{**g[0], "cache": {"hit": g[1]}} for g in got], opts)
                rep["peaks"] = []
                rep["cache"] = {"hits": len(got), "stored": 0, "untraced": [], "errors": [], "changed": [], "task_list": hit[1]}
                return rep
    tasks = _tasks(specs, opts, everything, as_call)
    out: list[dict | None] = [None] * len(tasks)
    keys = [target_key(t[0], t[1], opts, _info(t).get("as_call", as_call), tuple(_info(t).get("roots", ()))) for t in tasks] \
        if tracing else []
    if listed is not None and not CC.untraced():
        reads = CC.task_list_reads([*TG.REACHED.values(), *(r for t in tasks for r in _info(t).get("roots", ()))])
        CC.store(cache, listed, REPO, reads, {"targets": [list(t[:2]) for t in tasks], "keys": keys}, CC.seen(REPO, reads))
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
    if (jobs > 1 if isolate is None else isolate) and todo:
        init = (specs, opts, everything, as_call, tracing)
        budget = budget_gb(mem_gb)
        peaks = heavy()
        order = sorted(todo, key=lambda i: -peaks.get(tasks[i][1], 0))
        if log:
            for i in order:
                if peaks.get(tasks[i][1], 0) > budget:
                    log(f"warn {tasks[i][1]} peaked at {peaks[tasks[i][1]]} GiB, over the {budget:.1f} GiB budget: it runs alone")
        for i, got_, peak, start in parallel(_work, order, jobs, _init, init, weight=lambda i: peaks.get(tasks[i][1], HEAVY_GB),
                                             budget=budget):
            if isinstance(got_, Killed):
                r = {"failures": [{"check": "circuit-check", "code": "killed", "definition": tasks[i][1],
                                   "name": f"circuit-check/killed {tasks[i][1]}: its process died ({got_['how']}: the OOM "
                                           f"killer?) at a peak of {peak} GiB"}],
                     "warnings": [], "kind": tasks[i][0], "target": tasks[i][1], "seconds": 0.0, "peak_gb": peak, "killed": True}
                done(i, r)
            else:
                _, r, got, untraced, worker = got_
                r.update(peak_gb=peak, rss_start_gb=start)
                done(i, r, got, untraced, worker)
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


#: the targets that peak over `HEAVY_GB`, each with its peak in GiB (`run`'s scheduling; ``peak_gb``, which counts the 9.6 GiB
#: task list each process is forked with), from the isolated cold `--all` run r20261009-170758-f674 on vy-cpu-1: its first 239
#: targets, every one an earlier list held among them, peaked at 125-151 GiB (these six) or at most 10.6 GiB (the rest; the
#: earlier 29.4 and 104.2 GiB entries were what a long-lived worker kept from a previous target); its other targets confirm it
HEAVY = Path(__file__).with_name("heavy.json")
HEAVY_GB = 16
PEAKS_REPORTED = 50


def budget_gb(mem_gb: int | None) -> float:
    """What the targets' processes may take together, in GiB: what `memory_gb` leaves (this process's use taken off already),
    and with ``mem_gb`` (all of circuit-check: a check slot's mem=) at most that less what this process holds now (its own copy
    of the task list)."""
    return memory_gb() if mem_gb is None else min(mem_gb - (_rss_gb() or 0.0), memory_gb())


def heavy() -> dict[str, float]:
    return json.loads(HEAVY.read_text()) if HEAVY.exists() else {}


#: the memory a worker counts on: a cold `--all` grew six of 23 workers on vy-nebius-cpu-1 to 40-136 GiB (7 Oct), and two
#: `--all -j 0` runs, ~130 workers each, took 1.4 TB of node 1's 1.6 TB user slice (8 Oct); a cold `--all` at 7 workers in
#: vy-n2's 900 GiB session scope grew them to 107-155 GiB and was OOM-killed (r20261009-061009-1b4d, 9 Oct)
GB_PER_JOB = 160


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
    ap.add_argument("--mem-gb", type=int, default=None, help="the memory circuit-check may take, in GiB, this process included "
                    "(default: what this process may use, `memory_gb`); see `run`")
    ap.add_argument("--vectors", type=int, default=CK.Options.vectors)
    ap.add_argument("--seed", type=int, default=CK.Options.seed)
    ap.add_argument("-X", type=int, default=CK.Options.X)
    ap.add_argument("-W", type=int, default=CK.Options.W)
    ap.add_argument("--extras", type=int, default=CK.Options.extras)
    ap.add_argument("--lower-max-ands", type=int, default=CK.Options.lower_max_ands,
                    help="lower a Definition whole when the piece pins estimate at most this many ANDs")
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
              mem_gb=a.mem_gb, isolate=True, progress=a.out.with_name(a.out.name + ".progress.jsonl") if a.out else None)
    if c := rep.get("cache"):
        print(f"circuit-check: {c['hits']} of {len(rep['targets'])} targets reused (every file, listing, probe, catalog unit and pin their checks read or resolved is "
              f"unchanged), {c['stored']} stored, {len(c['untraced'])} not cacheable (the tracer could not see what they read), "
              f"{len(c['errors'])} not stored because the check raised, {len(c['changed'])} because the tree changed while they ran"
              + ("; the task list reused, so no Definition was built" if c.get("task_list") else ""))
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
