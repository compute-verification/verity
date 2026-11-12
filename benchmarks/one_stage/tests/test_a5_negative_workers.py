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
* the negatives' file is byte for byte the same with one worker and with several, on the toy window's coin negatives,
  the registration's and the audit record's, each decided by the Lean and the Python, and a negative that isn't refused
  is its entry's ``refused: false`` either way;
* the registered reads' negatives, each a verify of its own, are the same entries in the same order on a pool of several;
* the flag reaches the audit, default 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.sampled_proofs.one_stage import registration as R, verifier as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import a0  # noqa: E402
import a5  # noqa: E402
import a5_memo as AM  # noqa: E402
import test_a5_memo as TM  # noqa: E402

GB = 2**30


def workers(n: int, budget: float = 1000 * GB, free: float = 1000 * GB, need: float = GB, own: float = GB) -> a5.NegativeWorkers:
    return a5.NegativeWorkers(n, budget, lambda: free, need=lambda: need, own=lambda: own)


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


def _negatives_file(out: Path, n: int, passing: bool = False) -> bytes:
    """The toy window's negatives through `a5.NegativeWorkers` with ``n`` workers, written as a5 writes its
    ``negatives.json``: each coin negative its own stream (`test_a5_memo._negatives`'s), the registration's with its
    program one proof block fewer (R2), its ports bound to another run (R3) and another leaf layer (R4), the audit
    record's re-decisions (`a0.audit_negatives`) and, between them, entries the audit makes itself. ``passing``: the R3
    task checks the honest ports instead, a negative that isn't refused."""
    rec, ports = TM._record(), TM._ports()
    law, layer = {"law": "subset", "k": 2}, "ab" * 64
    sub = R.record(TM._program(), TM.QUERY, law, TM.S.SCHEME, rec["roots"], {"kind": "served", "run": TM.RUN, "row": 0},
                   leaf_layer=layer)
    session_roots = {r["port"]: r["root"] for r in sub["roots"]}
    draw = {"law": "subset", "population": sub["population"], "k": 2, "units": [1, 4]}
    result = V.audit_record(sub, R.receipt(sub, "a5 negative workers test"), draw, law, [1, 4], 2, {},
                            {"lean": {"accepted": True, "of_record": True, "digest": "dd" * 64}}, 0.5, served=[1, 4],
                            session_roots=session_roots)
    assert result["accepted"]
    triple = TM.P.template_id(TM.TRIPLE)

    def coin(name: str) -> dict:
        got = TM._negatives([name])[name]
        return {name: {"refused": got["accepted"] is False, "lean": got["lean"]}}

    def registration(name: str, **change) -> dict:
        v = V.check(**{**dict(rec=sub, program=TM._program(), query=TM.QUERY, law=law, scheme=TM.S.SCHEME, ports=ports,
                              leaf_layer=layer), **change})
        return {name: {"refused": not v.ok, "codes": list(v.codes)}}

    w = workers(n)
    parts: list = [{"before-the-draw": {"refused": True}}]
    parts.append(w.submit("R2", lambda: registration("R2", program=TM._program({**TM.COUNTS, triple: TM.COUNTS[triple] - 1}))))
    parts.append(w.submit("R3", lambda: registration("R3", **({} if passing else {"ports": TM._ports("r-another-run")}))))
    parts.append(w.submit("R4", lambda: registration("R4", leaf_layer="cd" * 64)))
    parts.append({"draw-outside-the-partition": {"refused": True, "codes": ["draw"]}})
    parts.append(w.submit("audit record", lambda: a0.audit_negatives(result, law, 0.5, session_roots, served=[1, 4])))
    parts += [w.submit(name, lambda name=name: coin(name)) for name in TM.NEGATIVES]
    w.run()
    neg: dict = {}
    for part in parts:
        neg.update(w.entries([part]) if isinstance(part, a5.Negative) else part)
    assert list(neg)[:5] == ["before-the-draw", "R2", "R3", "R4", "draw-outside-the-partition"]
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
    the same order, each verify run once under its own name."""
    ports = {f"p{j}": {"value": f"v{j}", "root": f"{j}" * 128, "leaves": 4, "reads": [0, 1, 2, 3]} for j in range(2)}
    (tmp_path / "registered.json").write_text(json.dumps({"format": "verity/registered-reads/v1", "ports": ports}))
    seen: list[str] = []

    def lean_verify(lean, st, n, sdir, zk, log, keys=None):
        seen.append(log)
        return {"accepted": False, "digest": "d", "why": f"{log}: holds no registered roots; does not open; the verifier's program reads"}

    monkeypatch.setattr(a5.a0, "lean_verify", lean_verify)
    monkeypatch.setattr(a5.a0, "archive_put", lambda lean, st, n, sdir, log: {})
    mm = {"st": {"registered_file": tmp_path / "registered.json", "dir": tmp_path},
          "m": SimpleNamespace(n=4, port=lambda p: f"c0/{p}")}
    one = a5.registered_negatives(SimpleNamespace(lean="lean"), mm, tmp_path)
    assert seen == list(one) and len(one) == 3 and all(e["refused"] for e in one.values())
    seen.clear()
    several = a5.registered_negatives(SimpleNamespace(lean="lean", audit_mem_gb=1), mm, tmp_path, workers=3)
    assert json.dumps(several) == json.dumps(one) and sorted(seen) == sorted(one)


@pytest.mark.parametrize("mem_gb", [0, 1])
def test_at_once_every_registered_reads_case_is_put_before_any_is_verified(tmp_path, monkeypatch, mem_gb):
    """`a5.registered_negatives` on a pool: its verifies read one archive, and `archive-put` writes again, in place, the
    objects the cases share, so every case's put ends before any case's verify starts, and no verify puts. The stand-in
    for the Lean binary puts as Lean does and, as Lean does, refuses an object that does not hash to its key. With
    memory for one verify at a time the pool runs the cases in turn, so a case that put for itself would put after the
    case before it was verified; with memory for all, the verifies start together."""
    stage = tmp_path / "stage"
    (tmp_path / "sessions" / "s0").mkdir(parents=True)
    (tmp_path / "sessions" / "s0" / "session.json").write_text("{}")
    stage.mkdir()
    for name in ("circuit.txt", "pub-4.bin", "partition.json", "program.json"):
        (stage / name).write_text(name)
    ports = {f"p{j}": {"value": f"v{j}", "root": f"{j}" * 128, "leaves": 4, "reads": [0, 1, 2, 3]} for j in range(2)}
    (tmp_path / "registered.json").write_text(json.dumps({"format": "verity/registered-reads/v1", "ports": ports}))
    held = tmp_path / "archive" / "sha512"
    events: list[tuple[str, str]] = []

    def run(cmd, log):
        if cmd[1] == "archive-put":
            events.append(("put", log.name.removesuffix(".archive-put.log")))
            held.mkdir(parents=True, exist_ok=True)
            keys = [(hashlib.sha512(Path(f).read_bytes()).hexdigest(), f) for f in cmd[3:]]
            for key, f in keys:
                (held / key).write_bytes(Path(f).read_bytes())
            return SimpleNamespace(stdout="".join(f"{key}  {f}\n" for key, f in keys), stderr="", returncode=0)
        events.append(("verify", log.name.removesuffix(".log")))
        named = [cmd[i + 1] for i, x in enumerate(cmd) if x in ("--circuit", "--public", "--partition", "--program", "--registered")]
        whole = all(hashlib.sha512((held / key).read_bytes()).hexdigest() == key for key in named)
        why = "holds no registered roots; does not open; the verifier's program reads" if whole else "setup: archive"
        return SimpleNamespace(stdout=f"VERDICT {json.dumps({'accepted': False, 'why': why})}\n", stderr="", returncode=1)

    monkeypatch.setattr(a5.a0, "run", run)
    st = {"registered_file": tmp_path / "registered.json", "dir": stage, "partition_file": stage / "partition.json",
          "program_file": stage / "program.json", "archived": {}}
    assert a0.lean_verify("lean", st, 4, tmp_path, zk=True)["why"].startswith("holds no")
    events.clear()
    got = a5.registered_negatives(SimpleNamespace(lean="lean", audit_mem_gb=mem_gb), {"st": st, "m": SimpleNamespace(
        n=4, port=lambda p: f"c0/{p}")}, tmp_path, workers=3)
    assert len(got) == 3 and all(e["refused"] for e in got.values())
    assert events[:3] == [("put", name) for name in got]
    assert events[3:] == [("verify", name) for name in got] if mem_gb == 0 else sorted(events[3:]) == sorted(
        ("verify", name) for name in got)


def test_the_flag_reaches_the_audit_and_defaults_to_one():
    ap = argparse.ArgumentParser()
    a5.lean_options(ap)
    assert ap.parse_args([]).negative_workers == 1 and ap.parse_args(["--negative-workers", "7"]).negative_workers == 7
    with pytest.raises(SystemExit):
        ap.parse_args(["--negative-workers", "0"])
