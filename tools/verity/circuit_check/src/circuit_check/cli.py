"""``circuit-check TARGET... [--out report.json]``: the standard property checks on each target, one machine-readable report.

    circuit-check F32Add_v1 'RMSNormTriton_v1{N=64,EPS=1e-05}'     Definitions
    circuit-check rope-head gemm-coordinate/k64/sm80-mma-bf16       templates, subcircuits
    circuit-check path/to/build_request_LP3_T2                      a served workload's request Program
    circuit-check verity/ml/circuits/ops/boolean/gf2k.circuit-check.py
                                                                    the targets a sidecar's bindings reach that its module defines
    circuit-check --all                                             every registered Definition and template (the suite's set)
    circuit-check --structural unit.tmpl GemmAccumulatorS8_v2{...}   a flock-ir-tmpl template as the Definition's lowering
    circuit-check --ambient PouwU129_v2 F32Mul_v4                   each lowering one text in a plain and a perturbed interpreter
    circuit-check --batch                                           batch invariance of the v0 registration's templates

A root a sidecar lists in ``_by_calls_roots`` is checked by its calls alone and never built (`by_calls`): its report entry says
``"mode": "by-calls"`` and names the target each callee relies on.

Exit status 1 when a check fails that `circuit_check.known` does not list (failures are named in the summary; ``--strict`` fails
on the known ones too), and with ``--all`` when a known entry no longer fails; 0 otherwise.  Redundant and dead gates are
warnings with counts, printed and in the report (``--fail-on-warnings`` makes them fail too).

Named targets run in parallel, one worker per core (`default_jobs`), and a target whose every input is unchanged is reused
from check's own cache (`default_cache`; ``--no-cache`` checks it now).  Each target's line, and the run's last, say where
its time went (``seconds_by_check``, `checks.spent`).
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

from . import ambient as AM
from . import by_calls as BC
from . import cache as CC
from . import checks as CK
from . import sidecars as SC
from . import structural as ST
from . import targets as TG
from . import units as UN

REPO = Path(__file__).resolve().parents[5]


def _options(a: argparse.Namespace) -> CK.Options:
    return CK.Options(vectors=a.vectors, seed=a.seed, X=a.X, W=a.W, extras=a.extras, lower_max_ands=a.lower_max_ands,
                      struct_max_gates=a.struct_max_gates)


_TASKS: list[tuple] = []


def _info(task: tuple) -> dict:
    """A task's ``(kind, label, check[, info])``: info has ``as_call`` (in its key), ``units`` (its bindings, `targets.BINDINGS`)
    and ``roots`` (the objects it checks, built before its check)."""
    return task[3] if len(task) > 3 else {}


def _one(kind: str, label: str, fn: Callable[[], dict]) -> dict:
    """``fn()``'s result with the target's ``seconds`` and its ``seconds_by_check`` (`checks.spent`), heaviest first."""
    CK.SPENT.clear()
    t0 = time.time()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001  (a target the checks cannot run on is a failure by name, not a crash)
        r = {"failures": [{"check": "circuit-check", "code": "error", "definition": label,
                           "name": f"circuit-check/error {label}: {type(e).__name__}: {str(e)[:200]}"}], "warnings": []}
    by = {k: round(v, 3) for k, v in CK.SPENT.most_common()}
    return {**r, "kind": kind, "target": label, "seconds": round(time.time() - t0, 3), "seconds_by_check": by}


#: the checks a target's log line names, heaviest first, each that took at least `SHOWN_SECONDS`
SHOWN_CHECKS = 4
SHOWN_SECONDS = 0.1


def _where_time_went(r: dict) -> str:
    by = [(k, v) for k, v in (r.get("seconds_by_check") or {}).items() if v >= SHOWN_SECONDS][:SHOWN_CHECKS]
    return f"{r['seconds']:.1f}s" + (": " + ", ".join(f"{k} {v:.1f}" for k, v in by) if by else "")


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


def target_key(kind: str, label: str, opts: CK.Options, as_call: bool | None, roots: tuple = (), desc: str | None = None) -> dict:
    """A target's key: its request, and for a Definition the digest of what it reaches as descriptors (`cache.descriptor`), or
    ``desc``, that digest as its binding's record holds it (`circuit_check.units`)."""
    request = {"target": [kind, label], "as_call": as_call, "options": {k: getattr(opts, k) for k in opts.__dataclass_fields__}}
    if roots or desc is not None:
        request["descriptor"] = desc if desc is not None else CC.descriptor(roots[0])
    return CC.key(request)


def _tasks(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None) -> list[tuple]:
    """The targets' checks in target order (deterministic: a worker rebuilds the same list and runs entries by index).  Traced
    (`cache.tracing`), each check gets a memo of its own, so its result never rests on what an earlier target's check computed,
    and the pins it looks up are recorded (`cache.recorded_pins`); a suite Definition's ``as_call`` is resolved here (the catalog's
    `targets.call_families`, of this suite), so it is in the target's key rather than a read of its check.  A named by-calls
    root brings the targets its callees rely on (`by_calls.relied_targets`), each checked beside it, so the report holds their
    verdicts (`by_calls.joined`); ``--all``'s are its suite's."""
    from verity.ml.circuits.evaluation.reference import resolve

    TG.load_registries()
    fresh = CC.tracing()
    shared: dict = {}
    memo = (lambda: {}) if fresh else (lambda: shared)
    pins = CC.recorded_pins(CK.load_pins()) if fresh else CK.load_pins()
    tasks: list[tuple] = []
    named = 0
    if everything:
        found = TG.suite()
        calls = TG.call_families(found)
        for fid, fn in found.items():
            call = TG.family(resolve(fn)) in calls
            tasks.append(("definition", fid, lambda fn=fn, call=call: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=call),
                          {"as_call": call, "units": sorted(TG.BINDINGS.get(fid, ())), "roots": [fn]}))
        for label, sub, sts in TG.templates():
            tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins),
                          {"units": sorted(TG.BINDINGS.get(label, ()))}))
        named = len(tasks)
    for s in specs:
        kind = TG.kind_of(s)
        if kind == "row":
            tasks.append(("row", s, lambda s=s: CK.check_row(s, opts, memo=memo(), pins=pins)))
        elif kind == "sidecar":
            found, tpls = TG.sidecar_targets(s)
            for fid, fn in found.items():
                tasks.append(("definition", fid, lambda fn=fn: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=as_call),
                              {"units": sorted(TG.BINDINGS.get(fid, ())), "roots": [fn]}))
            for label, sub, sts in tpls:
                tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins),
                              {"units": sorted(TG.BINDINGS.get(label, ()))}))
        elif kind == "template":
            def tpl(s=s):
                sub, sts = TG.subcircuit(s)
                return CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins)
            tasks.append(("template", s, tpl))
        else:
            try:
                fn = TG.definition(s)
            except Exception:  # noqa: BLE001  (its check raises it again, by name, and an errored check is never stored)
                tasks.append(("definition", s, lambda s=s: CK.check_definition(TG.definition(s), opts, memo=memo(), pins=pins, as_call=as_call)))
                continue
            tasks.append(("definition", s, lambda fn=fn: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=as_call),
                          {"units": sorted(TG.BINDINGS.get(fn.id, ())), "roots": [fn]}))
    labels = {t[1] for t in tasks}
    for t in tasks[named:]:
        if not any(TG.is_by_calls(fn) for fn in _info(t).get("roots", ())):
            continue
        try:
            relied = BC.relied_targets(_info(t)["roots"][0])
        except Exception as e:  # noqa: BLE001  (the root's own check fails by name on what raised here)
            print(f"circuit-check: {t[1]}: the targets its callees rely on aren't added ({type(e).__name__}: {str(e)[:200]})",
                  file=sys.stderr)
            continue
        for fn in relied:
            if fn.id not in labels:
                labels.add(fn.id)
                tasks.append(("definition", fn.id, lambda fn=fn: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=as_call),
                              {"units": sorted(TG.BINDINGS.get(fn.id, ())), "roots": [fn]}))
    return tasks


def _planned(plan: UN.Plan, opts: CK.Options, cache: Path) -> tuple[list[tuple], list[dict], dict[int, tuple[dict, str]]]:
    """`_tasks` of ``--all`` from the bindings' records (`units.plan`), each target's key and the entries reused: a target that
    misses is made from a binding that reaches it (`units._materialize`), and `units.Stale` when its descriptor is not its
    record's."""
    d = plan.derived
    memo = lambda: {}  # noqa: E731
    pins = CC.recorded_pins(CK.load_pins())
    tasks: list[tuple] = []
    keys: list[dict] = []
    hits: dict[int, tuple[dict, str]] = {}
    for fid in d.ids:
        node = d.nodes[fid]
        call = node["f"] in plan.calls
        k = target_key("definition", fid, opts, call, desc=node["d"])
        fn = None
        if (hit := CC.lookup(cache, k, REPO)) is not None and not _errored(hit[0]):
            hits[len(tasks)] = hit
        else:
            fn = UN._materialize(plan.entries, plan.objects, fid, via=d.via.get(fid))
            if CC.descriptor(fn) != node["d"]:
                raise UN.Stale(f"{fid}: its descriptor is not the one its binding's record holds")
        units = d.units.get(fid, set())
        TG.BINDINGS.setdefault(fid, set()).update(units)
        tasks.append(("definition", fid, lambda fn=fn, call=call: CK.check_definition(fn, opts, memo=memo(), pins=pins, as_call=call),
                      {"as_call": call, "units": sorted(units), "roots": [fn] if fn is not None else []}))
        keys.append(k)
    for label, sub, sts in TG.templates():
        k = target_key("template", label, opts, None)
        if (hit := CC.lookup(cache, k, REPO)) is not None and not _errored(hit[0]):
            hits[len(tasks)] = hit
        tasks.append(("template", label, lambda sub=sub, sts=sts: CK.check_subcircuit(sub, opts, instance_statics=sts, memo=memo(), pins=pins),
                      {"units": sorted(TG.BINDINGS.get(label, ()))}))
        keys.append(k)
    return tasks, keys, hits


def _record_units(cache: Path, opts: CK.Options, tasks: list[tuple], keys: list[dict]) -> dict:
    """After a full build: every binding's record (`units.store`), once the suite its records give (`units.derive`) is this
    run's, target for target, with each one's units, whether it is a Call and its key; else why none was stored."""
    entries = UN.built_entries()
    if entries is None:
        return {"stored": 0, "why": "the registry grew while the catalog was built"}
    try:
        d = UN.derive(entries)
        roots = {fn.id: fn for e in entries for fn in e.roots if UN._kind(fn) is not None}
        calls = UN.call_families(entries, d, roots, in_unit=False)
    except UN.Stale as e:
        return {"stored": 0, "why": str(e)}
    want = [(t[1], _info(t)["as_call"], _info(t)["units"], k) for t, k in zip(tasks, keys) if t[0] == "definition"]
    got = [(fid, c, sorted(d.units.get(fid, ())), target_key("definition", fid, opts, c, desc=d.nodes[fid]["d"]))
           for fid in d.ids for c in [d.nodes[fid]["f"] in calls]]
    if got != want:
        i = next(i for i in range(max(len(got), len(want))) if i >= min(len(got), len(want)) or got[i] != want[i])
        g, w = (got[i] if i < len(got) else None), (want[i] if i < len(want) else None)
        what = "target" if g is None or w is None or g[0] != w[0] else \
            next(n for n, a, b in zip(("target", "as_call", "units", "key"), g, w) if a != b)
        return {"stored": 0, "why": f"the records give another task list: its {what} at {(g or w)[0]} differs"}
    return {"stored": UN.store(cache, REPO, entries, only_rebuilt=False)}


def _calls(specs: list[str]) -> None:
    """Whether each Definition target is a Call (`targets.is_call`), resolved once in the process the targets' processes fork
    from, so one that needs the catalog builds it there once rather than in each.  A target that doesn't resolve is left to its
    own check."""
    from verity.ml.circuits.evaluation.reference import resolve

    for s in specs:
        kind = TG.kind_of(s)
        for make in ([lambda s=s: TG.definition(s)] if kind == "definition" else
                     [lambda fn=fn: fn for fn in TG.sidecar_targets(s)[0].values()] if kind == "sidecar" else []):
            try:
                TG.is_call(TG.family(resolve(make())))
            except Exception:  # noqa: BLE001  (its check raises it again, by name)
                continue


def _init(specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None, tracing: bool) -> None:
    if tracing:
        CC.trace(REPO)
    _TASKS[:] = _tasks(specs, opts, everything, as_call)
    if as_call is None:
        _calls(specs)


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


class Zygote:
    """`parallel`'s one worker process (`_zygote`), which runs ``initializer(*initargs)`` and then forks a child per item running
    ``fn``.  Started by forkserver, it is never a fork of this possibly multi-threaded process (under pytest-xdist that
    deadlocked), and builds what it needs itself.  With ``fork`` it is forked from this process as it is now, with everything this
    process has built: the CLI's process, which runs no thread that could hold a lock across the fork, so the task list is
    built once, not again in the zygote, and the children share its pages with this process.  Collected and frozen first
    (`gc.freeze`), so neither process's collections copy those pages."""

    def __init__(self, fn: Callable, initializer: Callable | None = None, initargs: tuple = (), *, fork: bool = False) -> None:
        import multiprocessing as mp

        self._proc, self._pid, self._code, self._closed = None, None, None, False
        if not fork:
            ctx = mp.get_context("forkserver")
            self.conn, there = ctx.Pipe()
            self._proc = ctx.Process(target=_zygote, args=(there, fn, initializer, initargs), daemon=True)
            self._proc.start()
            there.close()
            return
        import gc
        import warnings

        self.conn, there = mp.Pipe()
        gc.collect()
        gc.freeze()
        sys.stdout.flush()
        sys.stderr.flush()
        with warnings.catch_warnings(), CC.internal():
            warnings.simplefilter("ignore", DeprecationWarning)        # the numeric libraries' idle threads
            pid = os.fork()                                            # the zygote keeps the tracer's hook
        if pid == 0:
            self.conn.close()
            code = 0
            try:
                _zygote(there, fn, initializer, initargs)
            except BaseException:  # noqa: BLE001  (printed; the parent names the exit when its pipe ends)
                import traceback

                traceback.print_exc()
                code = 1
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
        there.close()
        self._pid = pid

    def exitcode(self) -> int | None:
        """The zygote's exit code once it has ended (a signal as its negative), waiting for it: its pipe has ended."""
        if self._proc is not None:
            self._proc.join()
            return self._proc.exitcode
        if self._code is None:
            self._code = os.waitstatus_to_exitcode(os.waitpid(self._pid, 0)[1])
        return self._code

    def close(self) -> None:
        """Tell the zygote to kill its children and end, and wait for it."""
        if self._closed:
            return
        self._closed = True
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.conn.close()
        self.exitcode()


def _forked() -> None:
    """A zygote forked from the CLI's process (`Zygote` ``fork``): its children list the tree as git has it themselves
    (`cache.visible`), as a forkserver zygote's do, rather than as this process listed it before the checks."""
    CC._VISIBLE.clear()


def parallel(fn: Callable, items: list, jobs: int, initializer: Callable | None = None, initargs: tuple = (), *,
             weight: Callable[[Any], float] | None = None, budget: float | Callable[[], float] | None = None,
             zygote: Zygote | None = None):
    """``(item, fn(item), peak_gb, start_gb)`` for each item, yielded as each finishes, each item run alone in a process of its
    own: up to ``jobs`` at once, forked from one `Zygote` process, ``zygote`` (forked already, running ``fn``) or else one started
    by forkserver that runs ``initializer(*initargs)``.  ``peak_gb`` is that process's peak RSS (``ru_maxrss``, GiB): the
    item's own, plus ``start_gb``, the RSS it was forked with, which it shares with the zygote (what ``initializer`` built).  An
    item whose process died (the OOM killer) yields a `Killed` for ``fn(item)``, with the peak it reached; the others run on.
    With ``weight`` and ``budget`` an item starts only while the weights of the items running and its own add up to at most
    ``budget`` (an item heavier than ``budget`` runs alone), taken in the order given, the first that fits; a callable
    ``budget`` is asked again each time items are admitted.  The zygote ends with the iteration."""
    weigh = weight or (lambda _: 0.0)
    limit = (lambda: float("inf")) if budget is None else budget if callable(budget) else (lambda: budget)
    zygote = zygote or Zygote(fn, initializer, initargs)
    here = zygote.conn
    queue = list(enumerate(items))
    running: dict[int, tuple[Any, float]] = {}
    try:
        while queue or running:
            for k in admit([weigh(x) for _, x in queue], [w for _, w in running.values()], jobs, limit()):
                t, x = queue[k]
                here.send((t, x))
                running[t] = (x, weigh(x))
            queue = [q for q in queue if q[0] not in running]
            try:
                t, value, peak, start = here.recv()
            except EOFError:
                raise RuntimeError(f"circuit-check's zygote died (exit {zygote.exitcode()}) while running "
                                   f"{[x for x, _ in running.values()]}") from None
            yield running.pop(t)[0], value, peak, start
    finally:
        zygote.close()


def run(specs: list[str], opts: CK.Options, *, everything: bool = False, as_call: bool | None = None, log=print, jobs: int = 1,
        cache: Path | None = None, mem_gb: int | None = None, progress: Path | None = None, isolate: bool | None = None,
        fork: bool = False, mem_file: Path | None = None, units: bool = False) -> dict:
    """Every target's checks, in target order.  Isolated (``isolate``; by default when ``jobs`` > 1), each target is checked in a
    process of its own, up to ``jobs`` at once (`parallel`): forked from one process that holds the task list, each check with
    its own memo, so a target's check never reuses what another's computed: the results are the same as checking them here, in
    turn.  That process is forked from this one as soon as the task list is built (``fork``: the CLI, `Zygote`), before any key
    or lookup, so it holds what it would have built itself; else it is started by forkserver and builds the list again (`_init`).

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
    target that grows more than `HEAVY_GB` past the task list it is forked with (``peak_gb - rss_start_gb``) is listed in `HEAVY`
    with its peak: a target starts only while the peaks of those running and its own add up to at most `budget_gb` (``mem_gb``
    less this process's RSS), counting an unlisted target as `HEAVY_GB`; one heavier than that still runs, alone, and the log
    says it does not fit; the listed targets go first, heaviest first.  The sum counts each listed process's copy of the task
    list, which they share, so the bound is conservative by that much per process.  An unlisted target that grows well past
    `HEAVY_GB` can still break it: the process it gets killed in is reported as that target's failure (``circuit-check/killed``,
    with its peak), the other targets run on, and its peak belongs in `HEAVY`.  Once ``mem_file`` holds a number of GiB
    (`granted_gb`: check.py writes its slot's mem= there when every step beside circuit-check has ended), that number takes
    ``mem_gb``'s place for the targets not yet started.

    The task list (each target and its key) is a cache entry too, under the catalog's own key: while everything its build
    could depend on is unchanged (`cache.task_list_reads`), a run whose every target is reused as it was recorded builds no
    Definition. Any other run builds the list; with ``units``, from the catalog bindings' records (`circuit_check.units`),
    building only the bindings whose record misses, and a full build stores them."""
    tracing = cache is not None and CC.tracing()
    isolated = jobs > 1 if isolate is None else isolate
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
    units = units and listed is not None and (fork or not isolated)
    plan, planned, why = None, None, None
    if units:
        try:
            plan = UN.plan(cache, REPO, log)
            planned = _planned(plan, opts, cache) if plan is not None else None
        except UN.Stale as e:
            why = str(e)
            plan, planned = None, None
            TG.BINDINGS.clear()
    if planned is not None:
        tasks, keys, hits = planned
        listed = None                                     # its reads are every binding's: a full build stores it
    else:
        tasks, keys, hits = _tasks(specs, opts, everything, as_call), None, None
    zygote = None
    if isolated and fork:
        _TASKS[:] = tasks
        if as_call is None:
            _calls(specs)
        zygote = Zygote(_work, _forked, fork=True)          # before the keys: it holds what `_init` would have built
    try:
        if keys is None and tracing:
            keys = [target_key(t[0], t[1], opts, _info(t).get("as_call", as_call), tuple(_info(t).get("roots", ()))) for t in tasks]
        rep = _checked_all(tasks, specs, opts, everything, as_call, log, jobs, cache, mem_gb, progress, isolated, tracing,
                           listed, zygote, mem_file, keys, hits)
    finally:
        if zygote is not None:
            zygote.close()
    if units and not CC.untraced():
        if planned is not None:
            rep["cache"]["units"] = {**plan.stats(), "stored": UN.store(cache, REPO, plan.entries, only_rebuilt=True)}
        else:
            rep["cache"]["units"] = {**_record_units(cache, opts, tasks, keys), **({"stale": why} if why else {})}
    return rep


def _checked_all(tasks: list[tuple], specs: list[str], opts: CK.Options, everything: bool, as_call: bool | None, log, jobs: int,
                 cache: Path | None, mem_gb: int | None, progress: Path | None, isolated: bool, tracing: bool,
                 listed: dict | None, zygote: Zygote | None, mem_file: Path | None = None, keys: list[dict] | None = None,
                 hits: dict[int, tuple[dict, str]] | None = None) -> dict:
    """`run`'s report over the task list it built (with ``keys`` and ``hits``, the keys and the entries reused already)."""
    out: list[dict | None] = [None] * len(tasks)
    keys = keys if keys is not None else []
    if listed is not None and not CC.untraced():
        reads = CC.task_list_reads([*TG.REACHED.values(), *(r for t in tasks for r in _info(t).get("roots", ()))])
        CC.store(cache, listed, REPO, reads, {"targets": [list(t[:2]) for t in tasks], "keys": keys}, CC.seen(REPO, reads))
    stats = {"hits": 0, "stored": 0, "untraced": [], "errors": [], "changed": []}

    def done(i: int, r: dict, got: dict | None = None, untraced: list[str] | None = None, worker: dict | None = None) -> None:
        out[i] = r
        if progress is not None:
            with progress.open("a") as f:
                f.write(json.dumps({"kind": r["kind"], "target": r["target"], "peak_gb": r.get("peak_gb"), "rss_start_gb": r.get("rss_start_gb"), "seconds": r.get("seconds"),
                                    "seconds_by_check": r.get("seconds_by_check"), "failures": len(r["failures"]),
                                    **({"killed": True} if r.get("killed") else {})}) + "\n")
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
            if r.get("mode") == BC.CHECK:
                call += f" by-calls: {len((r.get('by_calls') or {}).get('callees', []))} callees, not lowered whole"
            log(f"{'FAIL' if r['failures'] else 'ok  '} {r['kind']:10s} {r['target'][:90]}  failures={len(r['failures'])} "
                f"warnings={len(r['warnings'])}" + (f" recomputed_across={across}" if across else "") + call + f" ({_where_time_went(r)})")

    todo = list(range(len(tasks)))
    if tracing:
        todo = []
        for i, k in enumerate(keys):
            if (hit := hits.get(i) if hits is not None else CC.lookup(cache, k, REPO)) is not None and not _errored(hit[0]):
                out[i] = {**hit[0], "cache": {"hit": hit[1]}}
                stats["hits"] += 1
                if log and not everything:
                    log(f"{'FAIL' if hit[0]['failures'] else 'ok  '} {tasks[i][0]:10s} {tasks[i][1][:90]}  reused: every input "
                        f"its check read is unchanged (it took {hit[0].get('seconds', 0):.1f}s)")
            else:
                todo.append(i)
    if isolated and todo:
        init = (specs, opts, everything, as_call, tracing)
        budget = budget_gb(mem_gb)
        peaks = heavy()
        order = sorted(todo, key=lambda i: -peaks.get(tasks[i][1], 0))
        if log:
            for i in order:
                if peaks.get(tasks[i][1], 0) > budget:
                    log(f"warn {tasks[i][1]} peaked at {peaks[tasks[i][1]]} GiB, over the {budget:.1f} GiB budget: it runs alone")
        def now() -> float:
            return budget if (g := granted_gb(mem_file)) is None else budget_gb(g)

        for i, got_, peak, start in parallel(_work, order, jobs, _init, init, weight=lambda i: peaks.get(tasks[i][1], HEAVY_GB),
                                             budget=now, zygote=zygote):
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
    """Every pin, with each unpinned, untyped or changed one the report saw set to what it measures now, in the file that holds the pin
    or, for a new one, the sidecar of the module that defines its Definition (`sidecars.write_pins`)."""
    updates = {}
    for t in rep["targets"]:
        for f in t.get("failures", []) + t.get("warnings", []):
            if f["check"] == "lowering" and f["code"] in ("unpinned", "untyped", "pin-mismatch"):
                updates[f["pins"], f["pin"]] = f["counts"]
    CK.load_pins()
    SC.write_pins(updates)
    return CK.load_pins()


#: the targets that grow more than `HEAVY_GB` past the task list they are forked with, each with its ``peak_gb`` in GiB in an
#: isolated `--all` run (`run`'s scheduling; it counts that task list).  The commit that writes a peak
#: names the run it was measured in.  No key holds it (`cache.HEAVY`), so a new peak reruns no target and needs no edit here.
#: Beside them, by binding name, how far the catalog bindings whose builds grow past `units.APART_GB` grew (`units.built_apart`)
HEAVY = CC.HEAVY
HEAVY_GB = 16
PEAKS_REPORTED = 50


#: tools/verity/check/slot.py's SLOT_MEM_ENV: the GiB the check slot this process runs in may take
SLOT_MEM_ENV = "CHECK_SLOT_MEM_GB"


def slot_gb() -> int | None:
    """The enclosing check slot's mem=, so a raw `circuit-check` in a slot keeps to it without `--mem-gb` (else two runs in
    one node's slots each budget the whole machine's MemAvailable and take it down)."""
    v = os.environ.get(SLOT_MEM_ENV, "")
    return int(v) if v.isdigit() else None


def budget_gb(mem_gb: int | None) -> float:
    """What the targets' processes may take together, in GiB: what `memory_gb` leaves (this process's use taken off already),
    and with ``mem_gb`` (all of circuit-check: a check slot's mem=, `slot_gb` when not given) at most that less what this process
    holds now (its own copy of the task list)."""
    mem_gb = slot_gb() if mem_gb is None else mem_gb
    return memory_gb() if mem_gb is None else min(mem_gb - (_rss_gb() or 0.0), memory_gb())


def granted_gb(path: Path | None) -> int | None:
    """The GiB ``path`` (``--mem-gb-file``) grants circuit-check, once its writer has put a number there; None before."""
    if path is None:
        return None
    with CC.internal():
        try:
            v = path.read_text().strip()
        except OSError:
            return None
    return int(v) if v.isdigit() else None


def heavy() -> dict[str, float]:
    with CC.internal():
        return json.loads(HEAVY.read_text()) if HEAVY.exists() else {}


#: the memory a worker counts on: a cold `--all` grows some workers to 40-155 GiB, and two `--all -j 0` runs, ~130 workers
#: each, took 1.4 TB of a 1.6 TB user slice
GB_PER_JOB = 160


def memory_gb(root: Path = Path("/")) -> int:
    """The memory this process may use: the least of MemAvailable and what each cgroup v2 limit over it leaves (a run's session
    scope and the user slice on a VM; a pod's /proc/meminfo is the host's). A cgroup's inactive_file (its memory.stat) is in
    its memory.current but the kernel reclaims it at the limit, so it counts as left: otherwise a user slice that page
    cache fills leaves a budget of ~0, and the targets run one at a time."""
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
                    stat = dict(f[:2] for ln in read(c / "memory.stat").splitlines() if len(f := ln.split()) >= 2)
                    cache = int(v) if (v := stat.get("inactive_file", "")).isdigit() else 0
                    left.append(int(m) - max(0, int(read(c / "memory.current") or 0) - cache))
    return max(0, min(left)) >> 30 if left else 0


def auto_jobs(root: Path = Path("/")) -> int:
    """`--jobs 0`: one worker per core this process may run on, at most one per `GB_PER_JOB` of its memory (its slot's, in one)."""
    gb = min(memory_gb(root), slot_gb() or memory_gb(root))
    return max(1, min(len(os.sched_getaffinity(0)), gb // GB_PER_JOB))


def cores(root: Path = Path("/")) -> int:
    """The CPUs this process may use: its affinity, bounded by its cgroup's cpu.max quota (a pod's `nproc` can exceed it)."""
    n = len(os.sched_getaffinity(0))
    try:
        q, p = (root / "sys/fs/cgroup/cpu.max").read_text().split()
        return n if q == "max" else min(n, max(1, int(q) // int(p)))
    except (OSError, ValueError):
        return n


def default_jobs(everything: bool) -> int:
    """``--jobs`` when not given: 1 with ``--all`` (check.py gives its slot's), else one worker per core (`cores`), each target
    still admitted only while the measured peaks fit the memory (`run`)."""
    return 1 if everything else cores()


def default_cache() -> Path:
    """The cache named targets use unless ``--cache`` or ``--no-cache`` says otherwise: check's own (tools/verity/check's
    ``cache_dir() / "circuit-check"``), so a target `check` or an earlier run checked is reused while its inputs are unchanged."""
    return Path(os.environ.get("CHECK_CACHE") or Path.home() / ".cache" / "verity-check") / "circuit-check"


def time_summary(rep: dict, wall: float, jobs: int) -> str:
    """One line: the run's wall time, the targets checked now and reused, and where the checked ones' time went
    (their ``seconds_by_check`` summed, heaviest first)."""
    ran = [t for t in rep["targets"] if "cache" not in t]
    by: dict[str, float] = {}
    for t in ran:
        for k, v in (t.get("seconds_by_check") or {}).items():
            by[k] = by.get(k, 0.0) + v
    top = sorted(by.items(), key=lambda kv: -kv[1])[:SHOWN_CHECKS]
    return (f"circuit-check: {wall:.1f}s wall at -j {jobs}: {len(ran)} target(s) checked in {sum(t.get('seconds', 0) for t in ran):.1f}s"
            + (" (" + ", ".join(f"{k} {v:.1f}" for k, v in top) + ")" if top else "")
            + f", {len(rep['targets']) - len(ran)} reused")


def structural(target: str, path: Path, *, exhaustive_bits: int = ST.EXHAUSTIVE_BITS, vectors: int = ST.VECTORS,
               seed: int = CK.Options.seed) -> dict:
    """``--structural PATH TARGET``'s report: the template at ``path`` held to the Definition ``target`` names
    (`structural.check_file`), its one target of kind ``structural``."""
    r = _one("structural", target, lambda: ST.check_file(TG.definition(target), path, exhaustive_bits=exhaustive_bits,
                                                         vectors=vectors, seed=seed))
    opts = {"structural": str(path), "exhaustive_bits": exhaustive_bits, "vectors": vectors, "seed": seed}
    return {"schema": CK.SCHEMA, "options": opts, "summary": CK.summarize([r]), "targets": [r]}


def _structural_main(a: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    if a.all or len(a.targets) != 1:
        ap.error("--structural holds one template to one Definition: give one target")
    rep = structural(a.targets[0], a.structural, exhaustive_bits=a.exhaustive_bits, vectors=a.structural_vectors, seed=a.seed)
    (t,) = rep["targets"]
    for name in rep["summary"]["failure_names"]:
        print(f"FAIL {name}")
    for w in t["warnings"]:
        print(f"warn {w['name']}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    rec = t.get("structural") or {}
    print(f"circuit-check: {a.structural} as {a.targets[0]}: {rec.get('verdict', 'not checked')}"
          + (f" ({rec['why_not_syntactic']})" if rec.get("why_not_syntactic") else "") + (f", report {a.out}" if a.out else ""))
    return 1 if t["failures"] or (a.fail_on_warnings and t["warnings"]) else 0


def ambient(targets: list[str]) -> dict:
    """``--ambient TARGET..``'s report: each target's flat lowerings in a plain and a perturbed interpreter
    (`ambient.check`), each a target of kind ``ambient``."""
    rs = [{**r, "kind": "ambient", "target": t} for t, r in zip(targets, AM.check(targets))]
    opts = {"ambient": {"plain": AM.PLAIN, "perturbed": AM.PERTURBED}}
    return {"schema": CK.SCHEMA, "options": opts, "summary": CK.summarize(rs), "targets": rs}


def _ambient_main(a: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    if a.all or not a.targets:
        ap.error("--ambient lowers the targets it is given: name one or more Definitions")
    rep = ambient(a.targets)
    for name in rep["summary"]["failure_names"]:
        print(f"FAIL {name}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for t in rep["targets"]:
        rec = t["ambient"]
        print(f"circuit-check: {t['target']}: {rec['verdict']} over {rec['lowerings']} lowerings"
              + (f", sha256 {rec['sha256']}" if "sha256" in rec else ""))
    return 1 if any(t["failures"] for t in rep["targets"]) else 0


#: the frozen v0 registration ``--batch`` checks: the engine profile's role and the row-id device token
BATCH_REGISTRATION = ("LLAMA32_1B", "rtxpro6000")


def batch(templates: Callable[[], list] | None = None) -> dict:
    """``--batch``'s report: batch invariance (`verity_vllm.check.batch`: (a) separable, (b) shape-invariant) of each template
    ``templates()`` gives, by default every one a request-major compilation of `BATCH_REGISTRATION` calls
    (`verity_vllm.program.predict.batch.templates`), each verdict held to the one declared beside it."""
    from verity_vllm.check import batch as BI
    if templates is None:
        from verity_vllm.program.predict import batch as PB

        def templates() -> list:
            return PB.templates(PB.registered_phi(*BATCH_REGISTRATION))
    t0 = time.time()
    vs = BI.check(templates())
    return {"schema": "verity-circuit-check/batch/v1", "verdicts": [v.to_json() for v in vs], "table": BI.report(vs),
            "seconds": round(time.time() - t0, 1)}


def _batch_main(a: argparse.Namespace, ap: argparse.ArgumentParser) -> int:
    if a.all or a.targets:
        ap.error("--batch checks the v0 registration's templates: give no target")
    rep = batch()
    print(rep["table"])
    bad = [v for v in rep["verdicts"] if not v["ok"]]
    for v in bad:
        print(f"FAIL {v['template']}: declared {v['expect']}, got {v['got']}")
    if a.out:
        a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"circuit-check --batch: {len(rep['verdicts'])} template(s), {len(bad)} not as declared, {rep['seconds']} s"
          + (f", report {a.out}" if a.out else ""))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="circuit-check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="Definition ids, template or subcircuit ids, request Program paths")
    ap.add_argument("--all", action="store_true", help="every registered Definition and template at the catalog's bindings")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="targets checked in parallel by this many workers, each starting only while the targets' measured peaks "
                         "fit the memory (see `run`); default: one per core for named targets, 1 with --all; 0: one per core, at "
                         "most one per GB_PER_JOB of memory")
    ap.add_argument("--mem-gb", type=int, default=None, help="the memory circuit-check may take, in GiB, this process included "
                    "(default: the enclosing check slot's CHECK_SLOT_MEM_GB, else what this process may use, `memory_gb`); see `run`")
    ap.add_argument("--mem-gb-file", type=Path, default=None, metavar="FILE",
                    help="once FILE holds a number of GiB, the targets not yet started may take that much in --mem-gb's place "
                         "(check.py writes its slot's mem= there when every step beside circuit-check has ended)")
    ap.add_argument("--vectors", type=int, default=CK.Options.vectors)
    ap.add_argument("--seed", type=int, default=CK.Options.seed)
    ap.add_argument("-X", type=int, default=CK.Options.X)
    ap.add_argument("-W", type=int, default=CK.Options.W)
    ap.add_argument("--extras", type=int, default=CK.Options.extras)
    ap.add_argument("--lower-max-ands", type=int, default=CK.Options.lower_max_ands,
                    help="lower a Definition whole when the piece pins estimate at most this many ANDs")
    ap.add_argument("--struct-max-gates", type=int, default=CK.Options.struct_max_gates,
                    help="realize a composite of at most this many IR gates by its structural lowering too "
                         "(c-flock:struct_lower: verity_flock.struct_lower's template, evaluated from the template; 0: none)")
    ap.add_argument("--as-call", action="store_true", help="hold a Definition target to the partition invariant as a Call "
                    "(default: when its family is a Call of a served Program)")
    ap.add_argument("--strict", action="store_true", help="fail on the known failures too (circuit_check.known)")
    ap.add_argument("--fail-on-warnings", action="store_true")
    ap.add_argument("--update-pins", action="store_true", help="pin the lowering counts and circuit digests measured now (review the diff)")
    ap.add_argument("--cache", type=Path, default=None, metavar="DIR",
                    help="reuse a report whose every input is unchanged (circuit_check.cache), else run and record one; named "
                         "targets use check's own by default (`default_cache`)")
    ap.add_argument("--no-cache", action="store_true", help="check every target now, and record nothing")
    ap.add_argument("--catalog-units", action="store_true",
                    help="with --all --cache: build only the catalog bindings whose record misses (circuit_check.units)")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--structural", type=Path, metavar="UNIT.tmpl",
                    help="hold this flock-ir-tmpl template to the one target's flat lowering instead of running the property "
                         "checks (circuit_check.structural): accepted by construction, else on every input, else on random ones")
    ap.add_argument("--exhaustive-bits", type=int, default=ST.EXHAUSTIVE_BITS,
                    help="with --structural: evaluate on every input when the template has at most this many input bits")
    ap.add_argument("--structural-vectors", type=int, default=ST.VECTORS,
                    help="with --structural: the random inputs evaluated past --exhaustive-bits (with --seed)")
    ap.add_argument("--ambient", action="store_true",
                    help="lower each target in a plain and a perturbed interpreter instead of running the property checks "
                         "(circuit_check.ambient): one text, and no read of the environment, a clock or an unseeded draw")
    ap.add_argument("--batch", action="store_true",
                    help="check batch invariance of every template the request-major compiler emits for the frozen v0 registration "
                         "instead of running the property checks (cli.batch): each verdict held to the declared one")
    a = ap.parse_args(argv)
    if sum(map(bool, (a.structural, a.ambient, a.batch))) > 1:
        ap.error("--structural, --ambient and --batch are three modes: give one")
    if a.structural:
        return _structural_main(a, ap)
    if a.ambient:
        return _ambient_main(a, ap)
    if a.batch:
        return _batch_main(a, ap)
    if not a.targets and not a.all:
        ap.error("give a target or --all")
    cache = None if a.no_cache or a.update_pins else a.cache or (None if a.all else default_cache())
    if cache:
        # the registries build their C++ twins on first use (g++, untraceable): once, in a process of its own, before tracing
        prebuild = ("from circuit_check import targets; targets.load_registries()\n"
                    "try:\n    from verity.ml.definitions.numerics import twins\nexcept ImportError:\n    twins = None\n"
                    "if twins:\n    twins.preload()        # the three C++ models, which checks otherwise compile on first use\n")
        subprocess.run([sys.executable, "-c", prebuild], cwd=REPO, capture_output=True)
        CC.trace(REPO)
    jobs = default_jobs(a.all) if a.jobs is None else a.jobs or auto_jobs()
    opts = _options(a)
    t0 = time.time()
    rep = run(a.targets, opts, everything=a.all, as_call=True if a.as_call else None,
              log=None if a.quiet else lambda s: print(s, flush=True), jobs=jobs, cache=cache,
              mem_gb=a.mem_gb, mem_file=a.mem_gb_file, isolate=True, fork=True, progress=a.out.with_name(a.out.name + ".progress.jsonl") if a.out else None,
              units=a.catalog_units)
    if not a.quiet:
        print(time_summary(rep, time.time() - t0, jobs))
    if u := (rep.get("cache") or {}).get("units"):
        print("circuit-check: catalog bindings: " + (f"{u['reused']} reused, {len(u['rebuilt'])} built ({', '.join(u['rebuilt'][:8])}"
              f"{', ...' if len(u['rebuilt']) > 8 else ''}), " if "reused" in u else "whole catalog built, ")
              + f"{u['stored']} records stored" + (f" ({u['why']})" if u.get("why") else "")
              + (f"; records stale ({u['stale']})" if u.get("stale") else ""))
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
