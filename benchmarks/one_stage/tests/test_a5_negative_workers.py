"""a5's negatives side by side (`a5.NegativeWorkers`, ``--negative-workers``):

* with one worker, the default, each task runs where it is submitted, in the audit's own process: today's order;
* with more, each task runs in a worker process forked from the audit, at most that many at once, and the entries come
  back in the order submitted whatever order the workers ended in;
* a worker starts only while its memory estimate fits the audit's memory beside what the audit holds and the running
  workers', and the node's free memory; with none running the next starts whatever its estimate;
* nothing is retried and nothing falls back to the audit's process: a worker that raises or dies ends the run with its
  error, the task run once, the other workers killed;
* no worker outlives the audit: one killed outright takes its workers with it;
* a worker's Lean verifier process is its own: it never writes to the audit's, which answers after the workers end, and
  a worker reuses the checks the audit's memo kept before it was forked;
* a worker's Lean verifier process that ends without an answer ends the run, and no negative is refused for it; Lean's
  own refusal is still a refusal;
* no worker is forked beside another Python thread of the audit's: the run ends before any task starts, its error the
  fix, and the audit's summary is written with it; the threads read before the first fork are printed and in the
  timing whether or not the run is refused;
* the negatives' file is byte for byte the same with one worker and with several, on the toy window's coin negatives,
  the registration's and the audit record's, each decided by the Lean and the Python, and a negative that isn't refused
  is its entry's ``refused: false`` either way;
* the registered reads' negatives, each a verify of its own, are the same entries in the same order on a pool of several,
  where every case's files are put in the session's archive before any case is verified, so no put writes an object a
  running verify reads;
* the flag reaches the audit, default 1.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.commitments.windows import history as HW, interface as I, rows as CS
from verity.protocols.proofs.spot.verifier.one_stage import registration as R, verifier as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import a0  # noqa: E402
import a5  # noqa: E402
import a5_memo as AM  # noqa: E402
import test_a5_history as TH  # noqa: E402
import test_a5_memo as TM  # noqa: E402

GB = 2**30


def workers(n: int, budget: float = 1000 * GB, free: float = 1000 * GB, need: float = GB, own: float = GB,
            refused=None, before=None) -> a5.NegativeWorkers:
    return a5.NegativeWorkers(n, budget, lambda: free, need=lambda: need, own=lambda: own, refused=refused, before=before)


AUDIT = os.getpid()


def logged(log: Path, name: str, wait: int | None = None, tell: int | None = None, fail: str | None = None) -> dict:
    """A task that logs its start and end with its process, first waiting for a byte on ``wait`` and last writing two
    to ``tell`` (pipes: a task that can't end before another did), or failing: raising, or killing its own process
    (never the audit's, this one)."""
    with log.open("a") as f:
        f.write(f"start {name} {os.getpid()}\n")
    if wait is not None:
        os.read(wait, 1)
    if fail == "raise":
        raise ValueError(f"{name} can't decide")
    if fail == "die":
        assert os.getpid() != AUDIT, f"{name} ran in the audit's process"
        os.kill(os.getpid(), signal.SIGKILL)
    with log.open("a") as f:
        f.write(f"end {name} {os.getpid()}\n")
    if tell is not None:
        os.write(tell, b"xx")
    return {name: {"refused": True, "by": name}}


def lines(log: Path) -> list[list[str]]:
    return [x.split() for x in log.read_text().splitlines()]


def test_one_worker_runs_each_task_where_it_is_submitted_in_the_audits_own_process(tmp_path):
    """The default: a task has run, in this process, by the time `submit` returns, so the negatives run in the order
    the audit reaches them, and `run` starts nothing."""
    log, w = tmp_path / "log", workers(1)
    tasks = []
    for i, name in enumerate(("a", "b", "c")):
        tasks.append(w.submit(name, lambda name=name: logged(log, name)))
        assert lines(log) == [[x, n, str(os.getpid())] for n in ("a", "b", "c")[:i + 1] for x in ("start", "end")]
    w.run()
    assert w.entries(tasks) == {n: {"refused": True, "by": n} for n in ("a", "b", "c")} and len(lines(log)) == 6
    assert w.timing()["workers"] == 1 and set(w.timing()["negatives"]) == {"a", "b", "c"}
    assert not multiprocessing.active_children()


def test_more_workers_run_the_tasks_side_by_side_each_in_its_own_process_and_the_entries_keep_the_order_submitted(tmp_path):
    """Two workers: nothing runs at `submit`; at `run` each task is a process of its own, not this one. The first task
    can't end before the second did (it waits for the second's byte), so the two ran at once and ended out of order,
    and the entries are still in the order submitted, with each worker's seconds and memory in the timing."""
    log, w = tmp_path / "log", workers(2)
    r, t = os.pipe()
    tasks = [w.submit("first", lambda: logged(log, "first", wait=r)), w.submit("second", lambda: logged(log, "second", tell=t))]
    assert not log.exists() and all(x.got is None for x in tasks)
    assert list(w.entries(tasks)) == ["first", "second"] and list(w.entries(tasks[::-1])) == ["second", "first"]
    got = lines(log)
    assert [x[:2] for x in got if x[0] == "end"] == [["end", "second"], ["end", "first"]]
    pids = {x[1]: x[2] for x in got}
    assert len(set(pids.values())) == 2 and str(os.getpid()) not in pids.values()
    timing = w.timing()
    assert timing["workers"] == 2 and timing["need"] == GB and timing["audit"] == GB
    assert timing["memory"]["audit"]["rss"] > 0 and set(timing["memory"]) == {"audit", "lean"}
    assert all(set(timing["negatives"][n]) == {"own_s", "worker", "lean", "verifier_checks"} for n in ("first", "second"))
    assert all(timing["negatives"][n]["worker"]["private"] > 0 for n in ("first", "second"))
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("budget, free, at_once", [
    (GB + 2 * 10 * GB - 1, 1000 * GB, 1), (1000 * GB, 10 * GB - 1, 1), (1, 1000 * GB, 1),
    (GB + 2 * 10 * GB, 1000 * GB, 2), (1000 * GB, 1000 * GB, 3)])
def test_a_worker_starts_only_while_its_estimate_fits_and_one_over_the_budget_runs_alone(tmp_path, monkeypatch, budget,
                                                                                         free, at_once):
    """Three workers allowed, each estimated at 10 GB beside the audit's 1 GB. A budget one byte short of two workers,
    free memory one byte short of one, or a budget of one byte each leave one worker at a time, in the order submitted;
    a budget of exactly two workers runs two at once, and an ample one all three. Each task holds until the audit
    waits for its workers, so how many were running at each wait is how many the gate let start; every task still
    runs."""
    log, w = tmp_path / "log", workers(3, budget=budget, free=free, need=10 * GB, own=GB)
    r, t = os.pipe()
    waited, wait = [], multiprocessing.connection.wait

    def released(conns):
        waited.append(len(conns))
        os.write(t, b"x" * len(conns))
        return wait(conns)

    monkeypatch.setattr(multiprocessing.connection, "wait", released)
    tasks = [w.submit(n, lambda n=n: logged(log, n, wait=r)) for n in ("a", "b", "c")]
    assert list(w.entries(tasks)) == ["a", "b", "c"]
    assert max(waited) == at_once and waited[0] == at_once and len({x[2] for x in lines(log)}) == 3
    assert sorted(x[:2] for x in lines(log)) == sorted([x, n] for n in ("a", "b", "c") for x in ("start", "end"))
    if at_once == 1:
        assert waited == [1, 1, 1]
        assert [x[:2] for x in lines(log)] == [[x, n] for n in ("a", "b", "c") for x in ("start", "end")]


@pytest.mark.parametrize("budget, free, held", [
    (1000 * GB, a5.Free(500 * GB, 10 * GB - 1, "/job.slice/audit.scope"),
     {"free": 10 * GB - 1, "node_mem_available": 500 * GB, "cgroup_room": 10 * GB - 1,
      "cgroup_level": "/job.slice/audit.scope", "binds": "cgroup"}),
    (1000 * GB, a5.Free(10 * GB - 1),
     {"free": 10 * GB - 1, "node_mem_available": 10 * GB - 1, "cgroup_room": None, "cgroup_level": None, "binds": "node"}),
    (GB + 2 * 10 * GB - 1, a5.Free(500 * GB), {"binds": "budget", "budget": GB + 2 * 10 * GB - 1, "running": 11 * GB}),
    (1000 * GB, a5.Free(500 * GB, 400 * GB, "/"), None)])
def test_the_workers_say_the_memory_they_start_with_and_once_what_held_one_back(tmp_path, capsys, budget, free, held):
    """Three tasks on three workers, each estimated at 10 GB beside the audit's 1 GB. `run` prints one line of the
    memory it starts with: the budget, the estimate, the audit's own, the node's MemAvailable, the cgroup's room and
    its level, and which binds. Where memory holds a worker back it prints one more, once for the whole run: the
    estimate and the term that held it, the cgroup's room, the node's or the budget with what runs within it. A run
    nothing holds back prints the first alone."""
    log = tmp_path / "log"
    w = workers(3, budget=budget, free=free, need=10 * GB, own=GB)
    tasks = [w.submit(n, lambda n=n: logged(log, n)) for n in ("a", "b", "c")]
    assert list(w.entries(tasks)) == ["a", "b", "c"]
    said = [json.loads(x[3:])["memory"] for x in capsys.readouterr().out.splitlines() if x.startswith('A5 {"memory"')]
    assert all(x.pop("pool") == "negatives" for x in said)
    assert said[0] == {"workers": 3, "budget": budget, "need": 10 * GB, "audit": GB, **a5.memory_terms(free)}
    assert said[1:] == ([] if held is None else [{"waits": "worker", "need": 10 * GB, **held}])
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("fail, said", [("raise", "raised:"), ("die", "died (exit -9)")])
def test_a_worker_that_raises_or_dies_ends_the_run_and_nothing_is_retried_or_run_in_the_audits_process(tmp_path, fail, said):
    """The second of three tasks fails once the first has ended; the third waits for a byte that never comes. `run`
    raises naming the failed negative (its error, where it raised), the failed task ran once, in a worker, the waiting
    worker is killed, and no entry of the failed task or the killed one exists."""
    log, w = tmp_path / "log", workers(3)
    r, t = os.pipe()
    never, _held = os.pipe()
    tasks = [w.submit("good", lambda: logged(log, "good", tell=t)),
             w.submit("bad", lambda: logged(log, "bad", wait=r, fail=fail)),
             w.submit("waiting", lambda: logged(log, "waiting", wait=never))]
    with pytest.raises(a5.NegativeFailed) as e:
        w.run()
    assert "'bad'" in str(e.value) and said in str(e.value) and "nothing is retried" in str(e.value)
    assert ("ValueError: bad can't decide" in str(e.value)) == (fail == "raise")
    got = lines(log)
    assert [x[:2] for x in got if x[1] == "bad"] == [["start", "bad"]] and str(os.getpid()) not in {x[2] for x in got}
    assert ["end", "waiting"] not in [x[:2] for x in got] and not multiprocessing.active_children()
    assert tasks[1].got is None and tasks[2].got is None
    with pytest.raises(a5.NegativeFailed, match="did not run"):
        w.entries(tasks)


def _outlive(started: int, release: int, alive: int) -> dict:
    """A task that says it started, waits for a byte on ``release`` and then says it is alive."""
    os.write(started, b"x")
    os.read(release, 1)
    os.write(alive, b"x")
    return {}


def _audit_of_two(started: int, release: int, alive: int) -> None:
    """A process that is the audit: two negatives at once, each in its worker, neither able to end."""
    w = workers(2)
    for name in ("a", "b"):
        w.submit(name, lambda: _outlive(started, release, alive))
    w.run()


def test_an_audit_killed_outright_takes_its_workers_with_it():
    """The audit is killed outright (SIGKILL: no handler, no cleanup of its own) while two workers run, each waiting
    on a pipe. The kernel kills each worker as the audit ends (`a5.die_with`), before the audit's exit is collected
    here; so a byte then sent to each worker's wait wakes none, none says it is alive, and the pipe they would say it
    on reaches its end once every worker has ended. A worker that outlived the audit would read its byte and say so."""
    (started, started_w), (release, release_w), (alive, alive_w) = os.pipe(), os.pipe(), os.pipe()
    audit = multiprocessing.get_context("fork").Process(target=_audit_of_two, args=(started_w, release, alive_w))
    audit.start()
    os.close(started_w)
    os.close(alive_w)
    assert os.read(started, 1) + os.read(started, 1) == b"xx"
    audit.kill()
    audit.join()
    assert audit.exitcode == -signal.SIGKILL
    os.write(release_w, b"xx")
    assert os.read(alive, 2) == b""
    for fd in (started, release, release_w, alive):
        os.close(fd)


def _lean_pid() -> int:
    return V._server._proc.pid


def test_a_worker_asks_a_lean_process_of_its_own_and_the_audits_answers_after(monkeypatch):
    """The audit's Lean verifier process answered a check before the workers were forked. Each worker's checks go to a
    Lean process it started itself, never the audit's (whose pipe it inherited), and afterwards the audit's handle is
    the same process, alive, and answers. A worker forked under the memo reuses the honest check the audit kept and
    asks its own Lean only for the altered one."""
    with AM.memo() as m:
        assert TM._check().ok
        mine, asked_before = _lean_pid(), dict(m.asked)

        def task(name: str, altered: bool) -> dict:
            asked = TM.Counter()
            lean = V.lean
            V.lean = lambda fn, value: (asked.update([fn]), lean(fn, value))[1]
            v = TM._check(ports=TM._ports("r-another-run")) if altered else TM._check()
            return {name: {"refused": not v.ok, "lean": _lean_pid() if asked else None, "asked": dict(asked),
                           "started_with": V._server._proc is not None and _lean_pid()}}

        w = workers(2)
        tasks = [w.submit("honest again", lambda: task("honest again", False)), w.submit("altered", lambda: task("altered", True))]
        got = w.entries(tasks)
        assert got["honest again"] == {"refused": False, "lean": None, "asked": {}, "started_with": False}
        assert got["altered"]["refused"] and got["altered"]["asked"] == {"registration": 1}
        assert got["altered"]["lean"] not in (None, mine) and got["altered"]["started_with"] == got["altered"]["lean"]
        assert w.timing()["negatives"]["altered"]["lean"]["peak"] > 0 == w.timing()["negatives"]["honest again"]["lean"]["peak"]
        assert _lean_pid() == mine and V._server._proc.poll() is None and dict(m.asked) == asked_before
        assert [w.timing()["negatives"][name]["verifier_checks"] for name in ("honest again", "altered")] == [
            {"asked": {}, "reused": {"check": 1}}, {"asked": {"check": 1}, "reused": {}}]
        assert "R3-domain" in TM._check(ports=TM._ports("r-a-third-run")).codes and _lean_pid() == mine


#: a Lean verifier process that reads its first request and is killed without answering, as one killed for memory is
ENDS = "import os, signal, sys; sys.stdin.readline(); os.kill(os.getpid(), signal.SIGKILL)"


def test_a_workers_lean_process_that_ends_without_an_answer_ends_the_run_and_refuses_no_negative(monkeypatch):
    """The toy window's honest registration, nothing altered, checked as a5's ``registration`` task records it, by two
    workers whose every Lean verifier process ends on its first request without an answer (`ENDS`, a Python started
    where the verifier starts its executable). The run ends with `NegativeFailed`, the worker's error saying Lean gave
    no answer, and no entry exists: no negative is refused because a Lean process ended. Taken as Lean's refusal, as
    `verifier.lean` takes it, each entry would be ``refused: true`` beside Python's accept."""
    popen = subprocess.Popen
    monkeypatch.setattr(V, "executable", lambda exe=V.EXE: Path(sys.executable))
    monkeypatch.setattr(V, "subprocess", SimpleNamespace(PIPE=subprocess.PIPE,
                                                         Popen=lambda argv, **kw: popen([*argv, "-c", ENDS], **kw)))

    def registration(name: str) -> dict[str, dict]:
        v = TM._check()
        return {name: {"refused": not v.ok, "codes": list(v.codes)}}

    w = workers(2)
    tasks = [w.submit(n, functools.partial(registration, n)) for n in ("honest-a", "honest-b")]
    with pytest.raises(a5.NegativeFailed) as e:
        w.run()
    assert "raised:" in str(e.value) and f"NoLeanAnswer: a negative's worker's {V.EXE} gave no answer on registration" \
        in str(e.value), str(e.value)
    assert all(t.got is None for t in tasks) and not multiprocessing.active_children()
    with pytest.raises(a5.NegativeFailed, match="did not run"):
        w.entries(tasks)


def test_a_refusal_that_is_leans_own_answer_is_still_a_refusal_in_a_worker(tmp_path):
    """Lean's refusal is an answer, so a worker records it as a5 does in its own process: the history root's negative
    as a5's ``history_root`` task records it (`a0.history_negative`: refused where the codes are `registration.HISTORY`
    alone and Lean's reason is the Python's), on a window
    whose registered history root has a bit flipped, by each of two workers, is refused, with Lean's side its own
    reason and no worker error. Needs the Lean verifier built."""
    mine, doc, rec, publics, _table = TH._window(tmp_path / "w")
    roots = I.registered_roots(rec["window"])
    hist = HW.params(doc, mine, publics, HW.pins(doc, roots))
    root = roots["latent"]
    wrong = {**hist, "registered": {"latent": root[:-1] + format(int(root[-1], 16) ^ 1, "x")}}
    leaf, want = CS.leaf_layer(publics), CS.expected_ports(mine)

    def history_root(name: str) -> dict[str, dict]:
        v = V.check(rec, mine.program, mine.query, mine.law, CS.SCHEME, want, leaf_layer=leaf, work_table=mine.work_table,
                    history=wrong)
        return {name: {**a0.history_negative(v), "lean": v.verifiers["lean"]}}

    w = workers(2)
    got = w.entries([w.submit(n, functools.partial(history_root, n)) for n in ("history-a", "history-b")])
    for name in ("history-a", "history-b"):
        assert got[name]["refused"] and got[name]["codes"] == [R.HISTORY], got[name]
        assert isinstance(got[name]["lean"]["refused"], str) and V.EXE not in got[name]["lean"]["refused"], got[name]
    assert not multiprocessing.active_children()


def _negatives_file(out: Path, n: int, passing: bool = False) -> bytes:
    """The toy window's negatives through `a5.NegativeWorkers` with ``n`` workers, written as a5 writes its
    ``negatives.json``: each coin negative its own stream (`test_a5_memo._negatives`'s), the registration's with its
    program one proof block fewer (R2), its ports bound to another run (R3) and another leaf layer (R4), the audit
    record's re-decisions (`a0.audit_negatives`) and, between them, entries the audit makes itself. ``passing``: the R3
    task checks the honest ports instead, a negative that isn't refused."""
    rec, ports = TM._record(), TM._ports()
    design, layer = TM.D.parse_design("subset:2"), "ab" * 64
    sub = R.record(TM._program(), TM.QUERY, design, TM.S.SCHEME, rec["roots"], {"kind": "served", "run": TM.RUN, "row": 0},
                   leaf_layer=layer)
    session_roots = {r["port"]: r["root"] for r in sub["roots"]}
    sample = {**design, "population": sub["population"], "units": [1, 4]}
    result = V.audit_record(sub, R.receipt(sub, "a5 negative workers test"), sample, design, [1, 4], 2, {},
                            {"lean": {"accepted": True, "of_record": True, "digest": "dd" * 64}}, 0.5, served=[1, 4],
                            session_roots=session_roots)
    assert result["accepted"]
    triple = TM.P.template_id(TM.TRIPLE)

    def coin(name: str) -> dict:
        got = TM._negatives([name])[name]
        return {name: {"refused": got["accepted"] is False, "lean": got["lean"]}}

    def registration(name: str, program=None, ports=ports, leaf_layer=layer) -> dict:
        v = V.check(sub, program or TM._program(), TM.QUERY, design, TM.S.SCHEME, ports, leaf_layer=leaf_layer)
        return {name: {"refused": not v.ok, "codes": list(v.codes)}}

    w = workers(n)
    parts: list = [{"before-the-sample": {"refused": True}}]
    parts.append(w.submit("R2", lambda: registration("R2", program=TM._program({**TM.COUNTS, triple: TM.COUNTS[triple] - 1}))))
    parts.append(w.submit("R3", lambda: registration("R3", **({} if passing else {"ports": TM._ports("r-another-run")}))))
    parts.append(w.submit("R4", lambda: registration("R4", leaf_layer="cd" * 64)))
    parts.append({"sample-outside-the-partition": {"refused": True, "codes": ["partition"]}})
    parts.append(w.submit("audit record", lambda: a0.audit_negatives(result, design, 0.5, session_roots, served=[1, 4])))
    parts += [w.submit(name, lambda name=name: coin(name)) for name in TM.NEGATIVES]
    w.run()
    neg: dict = {}
    for part in parts:
        neg.update(w.entries([part]) if isinstance(part, a5.Negative) else part)
    assert list(neg)[:5] == ["before-the-sample", "R2", "R3", "R4", "sample-outside-the-partition"]
    assert list(neg)[-len(TM.NEGATIVES):] == list(TM.NEGATIVES) and len(neg) == 5 + 4 + len(TM.NEGATIVES)
    out.mkdir()
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True))
    return (out / "negatives.json").read_bytes()


def test_the_negatives_file_is_byte_for_byte_the_same_with_one_worker_and_with_several(tmp_path):
    """The same negatives, each decided by the Lean and the Python on the toy window, with one worker (each in this
    process, as it is submitted) and with four (side by side, each in a worker with its own Lean process), with and
    without the memo: the file a5 writes is the same bytes, every negative refused. A negative that isn't refused
    (the registration checked on its honest ports) is ``refused: false`` in the file either way, the same bytes
    again, which fails the audit as it does today."""
    one = _negatives_file(tmp_path / "one", 1)
    assert _negatives_file(tmp_path / "four", 4) == one
    with AM.memo():
        assert _negatives_file(tmp_path / "one-memo", 1) == one and _negatives_file(tmp_path / "four-memo", 4) == one
    assert all(e["refused"] for e in json.loads(one).values())
    passed = _negatives_file(tmp_path / "one-passing", 1, passing=True)
    assert _negatives_file(tmp_path / "four-passing", 4, passing=True) == passed != one
    assert [n for n, e in json.loads(passed).items() if not e["refused"]] == ["R3"]


def test_the_registered_reads_negatives_are_the_same_entries_in_the_same_order_on_a_pool(tmp_path, monkeypatch):
    """`a5.registered_negatives` one verify after another, and with three at once on a `LeanPool`: the same entries in
    the same order, each verify run once under its own name. At once, every case's files are put, in the cases'
    order, before any case is verified; one after another, each verify puts its own as before."""
    ports = {f"p{j}": {"value": f"v{j}", "root": f"{j}" * 128, "leaves": 4, "reads": [0, 1, 2, 3]} for j in range(2)}
    (tmp_path / "registered.json").write_text(json.dumps({"format": "verity/registered-reads/v1", "ports": ports}))
    member = tmp_path / "m"
    member.mkdir()
    head = {"registered": {p: {"value": e["value"], "positions": e["reads"], "paths": [""] * 4} for p, e in ports.items()}}
    (member / "pub-4.bin").write_bytes(json.dumps(head).encode() + b"\n" + bytes(range(8)))
    seen: list[str] = []

    def lean_verify(lean, st, n, sdir, zk, log):
        seen.append(log)
        return {"accepted": False, "digest": "d", "why": f"{log}: holds no registered roots; does not open; the verifier's program "
                                                         "reads; registered reads (complete)"}

    def archive_put(lean, archive, files, log, held=None):
        seen.append(log.name)
        return {}

    monkeypatch.setattr(a5.a0, "lean_verify", lean_verify)
    monkeypatch.setattr(a5.a0, "archive_put", archive_put)
    mm = {"st": {"registered_file": tmp_path / "registered.json", "dir": member}, "m": SimpleNamespace(n=4)}
    one = a5.registered_negatives(SimpleNamespace(lean="lean"), mm, tmp_path / "one")
    assert seen == list(one) and len(one) == 4 and all(e["refused"] for e in one.values())
    seen.clear()
    several = a5.registered_negatives(SimpleNamespace(lean="lean", audit_mem_gb=1), mm, tmp_path / "several", workers=3)
    assert json.dumps(several) == json.dumps(one)
    assert seen[:4] == [f"{name}.archive-put.log" for name in one] and sorted(seen[4:]) == sorted(one)


class Archive:
    """Lean's ``archive-put`` and ``verify`` as `a0.run` starts them, stood in: a put writes each file again, in place,
    under its SHA-512 (`archivePutCmd`), and a verify reads the objects its flags name by key for as long as it runs.
    Every verify stays open until ``at_once`` of them run, so a put that starts after a verify overlaps it. A put that
    `lean_verify` makes waits for the verify of every such put before it, which follows it on its own thread, so it
    lands while those verifies read; a put a caller makes ahead of `lean_verify` waits for nothing. ``overwritten``: each
    object a put wrote while a running verify was reading it, by the put's file and the verify's log."""

    def __init__(self, at_once: int):
        self.cv, self.at_once = threading.Condition(), at_once
        self.inside = threading.local()
        self.puts_inside = self.verifying = 0
        self.reading: dict[str, list[str]] = {}
        self.overwritten: list[tuple[str, str]] = []
        self.puts: list[str] = []
        self.failed = False

    def lean_verify(self, *args, **kw) -> dict:
        self.inside.verify = True
        try:
            return self.real(*args, **kw)
        except BaseException:
            with self.cv:
                self.failed = True
                self.cv.notify_all()
            raise
        finally:
            self.inside.verify = False

    def run(self, cmd: list[str], log: Path, env: dict | None = None) -> SimpleNamespace:
        if cmd[1] == "archive-put":
            return self.put(Path(cmd[2]), cmd[3:])
        assert cmd[1] == "verify", cmd
        return self.verify([cmd[i + 1] for i, x in enumerate(cmd) if x in (
            "--circuit", "--public", "--partition", "--program", "--registered", "--interface") and cmd[i + 1]], log.name)

    def put(self, archive: Path, files: list[str]) -> SimpleNamespace:
        inside = getattr(self.inside, "verify", False)
        with self.cv:
            if inside:
                self.cv.wait_for(lambda: self.verifying >= self.puts_inside or self.failed)
            (archive / "sha512").mkdir(parents=True, exist_ok=True)
            out = []
            for f in files:
                data = Path(f).read_bytes()
                key = hashlib.sha512(data).hexdigest()
                self.overwritten += [(Path(f).name, log) for log in self.reading.get(key, [])]
                (archive / "sha512" / key).write_bytes(data)
                self.puts.append(Path(f).name)
                out.append(f"{key}  {f}")
            self.puts_inside += inside
        return SimpleNamespace(stdout="\n".join(out) + "\n", stderr="", returncode=0)

    def verify(self, keys: list[str], log: str) -> SimpleNamespace:
        with self.cv:
            for key in keys:
                self.reading.setdefault(key, []).append(log)
            self.verifying += 1
            self.cv.notify_all()
            self.cv.wait_for(lambda: self.verifying >= self.at_once or self.failed)
            for key in keys:
                self.reading[key].remove(log)
        why = ("holds no registered roots; does not open; the verifier's program reads; registered reads (declared); is not "
               "the window's committed one; registered reads (complete): the statement's port b0 is; holds no registered "
               "interface (--interface); the statement names port p1 registered, and the verifier's registered reads do not")
        return SimpleNamespace(stdout="VERDICT " + json.dumps({"accepted": False, "why": why}) + "\n", stderr="", returncode=1)


def _bound_member(tmp_path: Path) -> dict:
    """A member of two registered ports whose last is one the window's record declares registered, with a bound port
    beside them, a session recorded, and its files' archive shared as a run's members share it."""
    ports = {f"p{j}": {"value": f"v{j}", "root": f"{j + 1}" * 128, "leaves": 4, "reads": [0, 1, 2, 3]} for j in range(2)}
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "registered.json").write_text(json.dumps(
        {"format": "verity/registered-reads/v1", "ports": ports, a5.REG.BOUND: {"b0": {"root": "b" * 128, "domain": "d" * 128}}}))
    (tmp_path / "interface.json").write_text("{}")
    member = tmp_path / "m"
    member.mkdir()
    (member / "circuit.txt").write_text("circuit")
    head = {"registered": {p: {"value": e["value"], "positions": e["reads"], "paths": [""] * 4} for p, e in ports.items()}}
    (member / "pub-4.bin").write_bytes(json.dumps(head).encode() + b"\n" + bytes(range(8)))
    return {"st": {"registered_file": tmp_path / "registered.json", "interface_file": tmp_path / "interface.json",
                   "dir": member, "archived": {}}, "m": SimpleNamespace(n=4, template="t")}


def _session(sdir: Path) -> Path:
    (sdir / "sessions" / "s0").mkdir(parents=True)
    (sdir / "sessions" / "s0" / "session.json").write_text("{}")
    return sdir


def test_no_registered_reads_case_puts_an_object_another_cases_verify_is_reading(tmp_path, monkeypatch):
    """The registered reads' cases verify one session against one archive, and Lean's put writes an object again in
    place, so a put beside a verify that reads the same object truncates it under the reader, which Lean then refuses
    as an object that doesn't hash to its key and not by the case's reason. Two cases can register the same reads: a
    member whose last port is the one the window's record declares registered leaves that port out in both, each in a
    file of its own, the same bytes under one key. With every case verified at once, nine here, no put writes an
    object a running verify reads; the entries are those of one verify after another."""
    monkeypatch.setattr(a5, "declared_ports", lambda interface, template: ["p1"])
    one = Archive(at_once=1)
    one.real = a0.lean_verify
    monkeypatch.setattr(a0, "run", one.run)
    monkeypatch.setattr(a5.a0, "lean_verify", one.lean_verify)
    first = a5.registered_negatives(SimpleNamespace(lean="lean"), _bound_member(tmp_path / "one"), _session(tmp_path / "one" / "s"))
    assert len(first) == 9 and all(e["refused"] for e in first.values()) and not one.overwritten
    same = [f for f in one.puts if f in ("registered-reads-leaving-a-port-committed-afresh.json", "declared-port-left-out.json")]
    assert len(same) == 2 and (tmp_path / "one" / "s" / same[0]).read_bytes() == (tmp_path / "one" / "s" / same[1]).read_bytes()

    at_once = Archive(at_once=len(first))
    at_once.real = one.real
    monkeypatch.setattr(a0, "run", at_once.run)
    monkeypatch.setattr(a5.a0, "lean_verify", at_once.lean_verify)
    several = a5.registered_negatives(SimpleNamespace(lean="lean", audit_mem_gb=1), _bound_member(tmp_path / "several"),
                                      _session(tmp_path / "several" / "s"), workers=len(first))
    assert at_once.overwritten == [] and not at_once.failed
    assert json.dumps(several) == json.dumps(first) and sorted(at_once.puts) == sorted(one.puts)


def _fork_line(out: str) -> list[dict]:
    """The threads the audit printed before a fork, a line each."""
    return [json.loads(x[3:])["negative_workers"]["threads_at_fork"] for x in out.splitlines()
            if x.startswith('A5 {"negative_workers"')]


def test_no_worker_is_forked_beside_another_python_thread_and_the_refusal_says_the_fix(tmp_path, capsys):
    """Another thread of this process is alive where `run` would fork (it waits on an event set only after `run`
    returned): `run` raises before any task starts, in a worker or here, and its error names the threads and says the
    fix. The threads it read are printed, and they are in the timing the audit writes as its summary with the refusal
    (`a5.not_forked`), so a refused audit's result holds the reading too."""
    log, up, done = tmp_path / "log", threading.Event(), threading.Event()
    other = threading.Thread(target=lambda: (up.set(), done.wait()), name="held-open")
    other.start()
    up.wait()
    me = threading.current_thread().name
    w = workers(2, refused=lambda timing: a5.not_forked(tmp_path, {"run_root": "r"}, timing))
    tasks = [w.submit(name, lambda name=name: logged(log, name)) for name in ("a", "b")]
    try:
        with pytest.raises(a5.NegativeFailed) as failed:
            w.run()
    finally:
        done.set()
        other.join()
    said = str(failed.value)
    assert f"holds 2 Python threads ({me}, held-open) where it would fork the negatives' workers" in said
    assert f"The fix: end and join every thread but {me} before the negatives run" in said
    assert "(--negative-workers 1). No worker was forked and nothing is retried" in said
    assert not log.exists() and all(t.got is None for t in tasks) and not multiprocessing.active_children()
    timing = w.timing()
    assert timing["refused"] == said and timing["negatives"] == {} and len(timing["threads_at_fork"]) == 1
    assert timing["threads_at_fork"][0]["python"] == [me, "held-open"] and timing["threads_at_fork"][0]["os"] >= 2
    assert _fork_line(capsys.readouterr().out) == timing["threads_at_fork"]
    assert json.loads((tmp_path / "summary.json").read_text()) == {
        "run_root": "r", "refused": {"negative-workers": said}, "negative_workers": timing}


def test_the_threads_read_before_the_first_fork_are_printed_and_kept_where_the_run_is_not_refused(tmp_path, capsys):
    """With no Python thread but the caller's, `run` forks its workers, and the reading is printed and in the timing all
    the same: Python's threads by name and the kernel's count, one reading for the `run` that forked and none for a
    `run` with nothing waiting. One worker forks nothing, so it reads nothing and prints nothing. (The other thread of
    the first reading has ended: the test waited for it on its event, then joined it.)"""
    log, me = tmp_path / "log", threading.current_thread().name
    w = workers(2)
    assert w.timing()["threads_at_fork"] == []
    tasks = [w.submit(name, lambda name=name: logged(log, name)) for name in ("a", "b")]
    assert list(w.entries(tasks)) == ["a", "b"]
    w.run()
    timing = w.timing()
    assert "refused" not in timing and len(timing["threads_at_fork"]) == 1
    assert timing["threads_at_fork"][0]["python"] == [me] and timing["threads_at_fork"][0]["os"] >= 1
    assert _fork_line(capsys.readouterr().out) == timing["threads_at_fork"]
    one = workers(1)
    one.submit("c", lambda: logged(log, "c"))
    one.run()
    assert one.timing()["threads_at_fork"] == [] and _fork_line(capsys.readouterr().out) == []
    assert not multiprocessing.active_children()


def test_the_audits_own_session_thread_has_ended_before_the_first_fork_and_any_other_is_still_refused(tmp_path, capsys):
    """The challenger session's own thread, as `a5_coins.start` runs it: its last act is to publish the evidence, after
    which it may still be alive. Here it then waits on an event that only the audit's read of the evidence sets, so it
    is alive when `run` starts. The audit reads the evidence and joins it before it reads its threads
    (`a5.session_ended`), so the first fork sees the caller's thread alone and the workers run: an honest audit is not
    refused for its own session. A thread that is not the session's is refused as before, the session's joined all the
    same. Off a challenger's coins, or with no thread handed in, nothing is waited for."""
    log, me = tmp_path / "log", threading.current_thread().name
    published, read, reads = threading.Event(), threading.Event(), []

    def finish() -> None:
        published.set()
        read.wait()

    def evidence() -> dict:
        published.wait()
        reads.append("the-session" in [t.name for t in threading.enumerate()])
        read.set()
        return {}

    coins = SimpleNamespace(evidence=evidence)
    session = threading.Thread(target=finish, name="the-session", daemon=True)
    session.start()
    w = workers(2, before=lambda: a5.session_ended(coins, session))
    tasks = [w.submit(name, lambda name=name: logged(log, name)) for name in ("a", "b")]
    assert list(w.entries(tasks)) == ["a", "b"] and not session.is_alive()
    assert reads == [True], "the session's thread was not alive when the audit read its evidence"
    at_fork, = w.timing()["threads_at_fork"]
    assert (at_fork["before_session_ended"], at_fork["python"]) == ([me, "the-session"], [me]) and "refused" not in w.timing()
    assert _fork_line(capsys.readouterr().out) == [at_fork]

    up, done = threading.Event(), threading.Event()
    other = threading.Thread(target=lambda: (up.set(), done.wait()), name="not-the-session", daemon=True)
    session = threading.Thread(target=finish, name="the-session", daemon=True)
    published.clear(), read.clear()
    other.start(), session.start()
    up.wait()
    w = workers(2, before=lambda: a5.session_ended(coins, session))
    w.submit("c", lambda: logged(log, "c"))
    try:
        with pytest.raises(a5.NegativeFailed, match=rf"holds 2 Python threads \({me}, not-the-session\)"):
            w.run()
    finally:
        done.set()
        other.join()
    assert not session.is_alive() and {x[1] for x in lines(log)} == {"a", "b"}

    a5.session_ended(None, session)
    a5.session_ended(SimpleNamespace(evidence=lambda: pytest.fail("no thread was handed in")), None)


def test_the_flag_reaches_the_audit_and_defaults_to_one():
    ap = argparse.ArgumentParser()
    a5.lean_options(ap)
    assert ap.parse_args([]).negative_workers == 1 and ap.parse_args(["--negative-workers", "7"]).negative_workers == 7
    with pytest.raises(SystemExit):
        ap.parse_args(["--negative-workers", "0"])
