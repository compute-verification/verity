"""`research.asker`: a pool run `--on auto` placed that waits behind reserved memory asks again and moves. Its asker's loop
runs here on an injected clock over a world of run summaries the test changes by events; the launcher's two commands run
through the tree's slot.py; and `research status`, `--wait` and `research cancel` follow the run to its twin on machines that
are directories reached through `sh -c`."""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research import asker, auto, cli, remote
from research.pods.nebius import slot_only

START = asker.start                 # the real one: conftest's `_no_asker_left` stands in for it in every test
REPO = Path(__file__).resolve().parents[3]
A, B, C = "vy-a", "vy-b", "vy-c"
RUN, TWIN = "r20261001-090000-aaaa", "r20261001-091000-bbbb"
T0 = datetime(2026, 10, 1, 9, 0, tzinfo=UTC).timestamp()
SINCE = "2026-10-01T09:00:00Z"
MEMORY = f"every check slot is taken (r0: 12{auto.MEMORY_BLOCK}, under the 40 GB a check here may peak at)"
STARTS = f"{B}'s research pool has the most places for it (2): a slot is free there now"
NOWHERE = "no other machine's pool would start it now (the shortest line per run that fits is vy-b's research pool)"
UNREACHED = f"twin {RUN}@{A} can't be claimed (ssh to {A} (exit 255): timed out), so its twin there runs instead"
AGENT = "https://cursor.com/agents/bc-1"


class Died(BaseException):
    """The asker's process ending where it stands."""


def waiting(why: str | None = MEMORY) -> dict:
    return {"state": "waiting", "class": None, "why": None, "slot": {"state": "waiting", "since": SINCE, **({"why": why} if why else {})}}


def held() -> dict:
    return {"state": "running", "class": None, "why": None, "slot": {"state": "held", "since": SINCE, "slot": "r0"}}


def left(why: str, cls: str | None = None) -> dict:
    return {"state": "failed", "class": cls, "why": None, "slot": {"state": "left", "since": SINCE, "why": why}}


class World:
    """The run RUN waiting on A and whatever its asker makes of it: `asker.see` reads ``runs``, `asker.ask` gives the next of
    ``answers`` and `asker.launch` puts the twin in ``runs``, waiting. ``at[minute]`` runs when the clock reaches that minute
    of the asker's life, before its look."""

    def __init__(self, tmp_path: Path, monkeypatch, wait_s: float = 4 * 3600):
        self.root, self.t, self.slept = tmp_path / "runs", T0, []
        self.runs, self.reached, self.at = {RUN: waiting()}, True, {}
        self.answers, self.asked, self.launched, self.cancelled = [], [], [], []
        self.m = {"run": RUN, "root": str(self.root), "home": A, "pool": "research", "git": "/no/repo", "commit": "c" * 40,
                  "cpus": 4, "mem_gb": 16, "lean": None, "wait_s": wait_s, "sends": {}, "launch": {}}
        monkeypatch.setattr(asker, "see", lambda d: (json.loads(json.dumps(self.runs[d.name])), self.reached))
        monkeypatch.setattr(asker, "files", lambda git, commit, paths: contextlib.nullcontext((tmp_path / "files", tuple(paths))))
        monkeypatch.setattr(asker, "tree", lambda git, commit: contextlib.nullcontext(tmp_path / "tree"))
        monkeypatch.setattr(asker, "ask", self.ask)
        monkeypatch.setattr(asker, "launch", self.launch)
        monkeypatch.setattr(asker, "cancel", lambda d, reason: self.cancelled.append((d.name, reason)))
        monkeypatch.setattr(asker, "_beside", lambda run_dir: None)

    def minute(self) -> float:
        return (self.t - T0) / 60

    def sleep(self, s: float) -> None:
        assert len(self.slept) < 60, "the asker never ended"
        self.slept.append(s)
        self.t += s
        if (event := self.at.pop(self.minute(), None)):
            event()

    def ask(self, source: tuple, m: dict, tried: dict):
        assert source == (self.root.parent / "files", auto.READS), "an ask reads the few files `auto.read` does, in no checkout"
        self.asked.append((self.minute(), dict(tried)))
        return self.answers.pop(0)

    def launch(self, source: Path, m: dict, twin: dict, wait_s: float) -> dict:
        assert source == self.root.parent / "tree", "a twin ships a checkout of the commit"
        self.launched.append({**twin, "wait_s": wait_s, "recorded": self.record()["twin"]})
        self.runs[twin["run"]] = waiting(None)
        return {"phase": "launched"}

    def record(self) -> dict:
        return json.loads((self.root / asker.DIR / f"{RUN}.json").read_text())

    def run(self) -> str:
        return asker.run(self.root, self.m, now=lambda: self.t, sleep=self.sleep)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# ---- it moves, or stays -------------------------------------------------------------------------------------------------------

def test_a_run_that_waits_on_memory_asks_again_at_minute_one_and_ten_and_moves_to_the_pool_that_would_start_it_now(world):
    """A run placed where a slot was free can wait an hour behind memory other runs reserve while another machine's pool
    stands free. At minute 1 of its wait, and at minute 10, its asker asks `auto` again and launches its late twin where it
    would start now; the twin takes a slot there, and both records say from where, why and after how long, each naming the
    other."""
    world.answers = [(None, NOWHERE), ((B, "research"), STARTS)]

    def twin_takes_a_slot():
        world.runs.update({world.launched[0]["run"]: held(), RUN: left(f"its twin took a slot first (run {world.launched[0]['run']})")})

    world.at[20] = twin_takes_a_slot
    why = world.run()
    (tw,) = world.launched
    assert world.asked == [(1, {}), (10, {})], "an ask at minute 1 of the wait, and one at minute 10"
    assert (tw["machine"], tw["pool"], tw["from"], tw["of"], tw["why"], tw["after_min"]) == (B, "research", A, RUN, MEMORY, 10)
    assert tw["said"] == f"moved from {A}'s research pool, where {RUN} waited 10 min ({MEMORY}), to {B}'s, which would start it now ({STARTS})"
    assert why == f"it moved: its twin {tw['run']} took a slot on {B}"
    rec = world.record()
    assert (rec["twin"]["run"], rec["twin"]["state"], rec["ended"]) == (tw["run"], "started", {"at": "2026-10-01T09:20:00Z", "why": why})
    assert rec["asks"] == [{"at": "2026-10-01T09:01:00Z", "after_min": 1, "said": NOWHERE},
                           {"at": "2026-10-01T09:10:00Z", "after_min": 10, "said": STARTS}] and rec["next"] is None
    assert world.slept == [60, 540, 0, 600], "it looks again as soon as the twin is launched, then at the next ten minutes"


def test_a_run_told_it_starts_now_that_found_the_memory_gone_moves_at_minute_one(world):
    """Launches that ask `auto` in the same seconds are each told of the same place, and the last ones in find its memory
    gone: such a run waits on memory from its first minute, beside machines with room, and need not wait ten to ask."""
    world.answers = [((B, "research"), STARTS)]
    world.at[10] = lambda: world.runs.update({world.launched[0]["run"]: held()})
    world.run()
    assert world.asked == [(1, {})] and world.launched[0]["after_min"] == 1 and world.slept == [60, 0, 540]


def test_it_asks_every_ten_minutes_and_stays_while_no_other_machines_pool_would_start_it_now(tmp_path, monkeypatch):
    """A twin launched into a line that doesn't start it is a second waiter for nothing: the run stays, and asks again."""
    world = World(tmp_path, monkeypatch, wait_s=35 * 60)
    world.answers = [(None, NOWHERE)] * 4
    assert world.run() == "its wait for a slot is over"
    assert [minute for minute, _ in world.asked] == [1, 10, 20, 30] and world.launched == [] and world.record()["twin"] is None
    assert [a["said"] for a in world.record()["asks"]] == [NOWHERE] * 4
    assert world.slept == [60, 540, 600, 600, 300], "it never outlives the first's wait for a slot"


def test_it_asks_only_while_the_run_waits_and_its_last_reason_is_memory_and_its_machine_answers(world):
    """A run behind others in line, or one whose machine doesn't answer, is not known to wait on memory: no ask. Nor is one
    that has just left its line, whose record still names memory while its status still says it runs: a twin launched
    then would run a job whose wait is over."""
    over = f"no check slot within --wait-max 14400 s ({MEMORY})"
    world.runs[RUN] = waiting("3 check(s) ahead in line")
    world.at[10] = lambda: setattr(world, "reached", False)
    world.at[20] = lambda: (world.runs.update({RUN: waiting()}))
    world.at[30] = lambda: setattr(world, "reached", True)
    world.at[40] = lambda: world.runs.update({RUN: waiting(None)})
    world.at[50] = lambda: world.runs.update({RUN: {**left(over), "state": "running"}})
    world.at[60] = lambda: world.runs.update({RUN: left(over)})
    world.answers = [(None, NOWHERE)]
    assert world.run() == f"it ended on {A} without starting (failed: {over})"
    assert [minute for minute, _ in world.asked] == [30], "in line at 1 and 10, unreachable at 20, no reason yet at 40, gone at 50"


def test_one_twin_is_live_at_a_time_and_a_machine_whose_twin_left_unrun_is_asked_for_no_other(world):
    """While a twin waits, a second one would be a third waiter; and a twin that couldn't reach the first's machine to claim
    the pair leaves unrun, as the next one on that machine would."""
    world.answers = [((B, "research"), STARTS), ((C, "research"), "c")]
    world.at[40] = lambda: world.runs.update({world.launched[0]["run"]: left(UNREACHED)})
    world.at[60] = lambda: world.runs.update({world.launched[1]["run"]: held()})
    why = world.run()
    first, second = world.launched
    assert [minute for minute, _ in world.asked] == [1, 40], "no ask at 10, 20 or 30, while its twin waited"
    assert world.asked[1][1] == {B: f"a twin of {RUN} left its line unrun ({UNREACHED})"}, "what the second ask reads of B"
    assert (first["machine"], second["machine"]) == (B, C) and why == f"it moved: its twin {second['run']} took a slot on {C}"
    assert world.record()["twin"]["run"] == second["run"] and world.cancelled == []


@pytest.mark.parametrize(("wait_min", "twin_s"), [(240, 600), (5, 240)], ids=["ten-minutes", "what-is-left-of-the-firsts-wait"])
def test_a_twin_waits_in_its_line_no_longer_than_ten_minutes_or_what_is_left_of_the_firsts_wait(tmp_path, monkeypatch, wait_min, twin_s):
    """A twin is sent where the run would start now. One that waits there after all (another run took the place first) is a
    second waiter for the run's whole wait, and while it is live the run asks no more: so it leaves that line by itself."""
    world = World(tmp_path, monkeypatch, wait_s=wait_min * 60)
    world.answers = [((B, "research"), STARTS)]
    world.at[10] = lambda: world.runs.update({RUN: held()})
    world.run()
    assert world.launched[0]["wait_s"] == twin_s


def test_a_twin_is_recorded_before_it_is_launched_so_an_asker_that_dies_then_leaves_it_named(world, monkeypatch):
    """An asker that died between launching a twin and recording it would leave a run on another machine that no record
    of the first names: `research status` and `research cancel` of the first would not find it."""
    world.answers = [((B, "research"), STARTS)]
    real = world.launch

    def launch(source, m, twin, wait_s):
        real(source, m, twin, wait_s)
        raise Died

    monkeypatch.setattr(asker, "launch", launch)
    with pytest.raises(Died):
        world.run()
    (tw,) = world.launched
    assert tw["recorded"]["run"] == tw["run"], "the record named the twin when its launch began"
    rec = world.record()
    assert (rec["twin"]["run"], rec["twin"]["machine"], rec["twin"]["state"], rec.get("ended")) == (tw["run"], B, "launching", None)
    (world.root / RUN).mkdir()
    assert asker.twin(world.root / RUN)["run"] == tw["run"], "so status and cancel follow it"


@pytest.mark.parametrize("lost", ["exit ", "TimeoutExpired: "], ids=["refused", "unreached"])
def test_no_twin_is_launched_while_its_record_can_not_be_copied_beside_the_firsts_request(world, monkeypatch, capsys, lost):
    """Another VM follows a run to its twin by the copy of the asker's record beside the first's request, and the launching
    VM may be gone a minute later: a twin launched while that copy can't be written is one no other VM could follow or
    cancel. The asker's log says why, and its next ask is of the same machine, which refused nothing."""
    world.answers = [((B, "research"), STARTS)] * 2
    beside = world.root.parent / A / "requests" / RUN / asker.FILE
    monkeypatch.setattr(asker, "_beside", lambda run_dir: (["sh", "-c"], str(beside)))
    copies, launch, ssh_run = [], world.launch, remote.ssh_run

    def unreached(ssh, argv, **kw):
        if lost != "exit " and not beside.parent.is_dir():
            raise subprocess.TimeoutExpired(ssh, kw["timeout"])
        return ssh_run(ssh, argv, **kw)

    def launched(source, m, twin, wait_s):
        copies.append(json.loads(beside.read_text())["twin"])
        return launch(source, m, twin, wait_s)

    monkeypatch.setattr(remote, "ssh_run", unreached)
    monkeypatch.setattr(asker, "launch", launched)
    world.at[10] = lambda: beside.parent.mkdir(parents=True)
    world.at[20] = lambda: world.runs.update({world.launched[0]["run"]: held()})
    world.run()
    (tw,) = world.launched
    assert world.asked == [(1, {}), (10, {})] and tw["at"] == "2026-10-01T09:10:00Z"
    assert [(c["run"], c["state"]) for c in copies] == [(tw["run"], "launching")], "the copy named the twin when its launch began"
    unsent, sent = world.record()["asks"]
    assert unsent["error"].startswith(f"OSError: its record could not be copied beside its request on {A} ({lost}") and "error" not in sent
    assert f"asker: {RUN}'s record was not copied beside its request on its machine ({lost}" in capsys.readouterr().err
    assert json.loads(beside.read_text()) == world.record()


def test_a_twin_whose_launch_fails_is_recorded_and_the_run_keeps_its_place_and_asks_elsewhere(world, monkeypatch):
    world.answers = [((B, "research"), STARTS), ((C, "research"), "c")]
    real = world.launch
    monkeypatch.setattr(asker, "launch", lambda source, m, twin, wait_s: real(source, m, twin, wait_s) if twin["machine"] == C else
                        (_ for _ in ()).throw(remote.Refused("no ssh")))
    world.at[20] = lambda: world.runs.update({RUN: held()})
    assert world.run().startswith(f"it started on {A}, and its twin ")
    assert [(minute, list(tried)) for minute, tried in world.asked] == [(1, []), (10, [B])]
    assert world.record()["asks"][0]["error"] == "Refused: no ssh" and world.slept == [60, 540, 0, 600]


@pytest.mark.parametrize("change", [lambda sent: sent.write_text('{"edited": true}'), Path.unlink], ids=["edited", "removed"])
def test_a_send_file_that_changed_since_the_launch_moves_nothing(world, tmp_path, change):
    """A twin ships the same request: a --send file edited or removed since would make it another run under the first's pair."""
    sent = tmp_path / "input.json"
    sent.write_text("{}")
    world.m["sends"] = {str(sent): asker._sha256(str(sent))}
    world.answers = [(None, NOWHERE)]
    world.at[10] = lambda: change(sent)
    world.at[20] = lambda: world.runs.update({RUN: held()})
    world.run()
    assert [minute for minute, _ in world.asked] == [1] and world.launched == [], "asked while the file was as sent, not after"
    assert world.record()["asks"][1]["error"] == f"OSError: --send {sent} changed since the launch, and a twin ships the same request"


# ---- it ends ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("first", "twin", "why", "cancelled", "live"), [
    (held(), None, f"it started on {A}", [], None),
    (held(), waiting(None), f"it started on {A}, and its twin TWIN leaves {B}'s line", [], False),
    (left("no check slot within --wait-max 14400 s"), None,
     f"it ended on {A} without starting (failed: no check slot within --wait-max 14400 s)", [], None),
    (left("its twin took a slot first (claimed)"), waiting(None),
     f"it ended on {A} without starting (failed: its twin took a slot first (claimed)); its twin TWIN on {B} is what is left of it", [],
     True),
    (left(MEMORY, "CANCELLED_MANUAL"), None, "it was cancelled", [], None),
    (left(MEMORY, "CANCELLED_MANUAL"), waiting(None), f"it was cancelled, and its twin TWIN on {B} with it",
     [("TWIN", f"its first, {RUN} on {A}, was cancelled")], False),
    (waiting(), left(MEMORY, "CANCELLED_MANUAL"), f"its twin TWIN was cancelled, so it asks no more; the run keeps its place on {A}", [],
     False),
], ids=["it-started", "it-started-beside-a-twin", "it-left-its-line", "it-left-for-its-twin", "it-was-cancelled",
        "it-was-cancelled-beside-a-twin", "its-twin-was-cancelled"])
def test_the_asker_ends_when_the_run_has_started_ended_or_been_cancelled_and_a_cancel_reaches_its_twin(world, first, twin, why, cancelled,
                                                                                                    live):
    """An asker that outlived its run would ask for, and could move, a run that is over; a cancelled run whose twin still
    waited would run after all. Its record's twin stays the run (``live``) only where it may still run: `research status`
    and `research cancel` follow no twin that can't."""
    world.answers = [((B, "research"), STARTS) if twin else (None, NOWHERE)]
    world.at[10] = lambda: world.runs.update({RUN: first, **({world.launched[0]["run"]: twin} if twin else {})})
    said = world.run()
    name = world.launched[0]["run"] if twin else "TWIN"
    assert said == why.replace("TWIN", name)
    assert world.cancelled == [(c.replace("TWIN", name), r) for c, r in cancelled]
    assert world.minute() == 10 and [minute for minute, _ in world.asked] == [1], "it ends at the look that finds it, and asks no more"
    rec = world.record()
    assert rec["ended"] == {"at": "2026-10-01T09:10:00Z", "why": said} and rec["next"] is None
    assert not asker.alive(rec, world.t)
    assert live is None if not twin else (rec["twin"]["state"] in asker.LIVE) == live
    if first == held() and twin:
        assert rec["twin"]["left"] == f"{RUN} took a slot on {A} first"


def test_the_first_starting_while_its_twin_is_launched_ends_the_asker_at_once(world, monkeypatch):
    world.answers = [((B, "research"), STARTS)]
    real = world.launch
    monkeypatch.setattr(asker, "launch", lambda *a: (real(*a), world.runs.update({RUN: held()}))[0])
    assert world.run() == f"it started on {A}, and its twin {world.launched[0]['run']} leaves {B}'s line"
    assert world.slept == [60, 0] and world.cancelled == []


# ---- never both places, never neither: the launcher's commands through slot.py ------------------------------------------------

@pytest.fixture
def slot_py(monkeypatch, tmp_path):
    """The tree's slot.py on two machines that are directories, `a` (the first's, named A in the twin's peers) and `b`; its
    ssh to A runs as sh here on a's locks."""
    spec = importlib.util.spec_from_file_location("asker_slot", REPO / "tools" / "verity" / "check" / "slot.py")
    s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s)
    for name in ("CHECK_SLOT", "CHECK_SLOTS", "RESEARCH_RUN_DIR", s.RUN_MEM_ENV, s.MIN_MEM_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(s, "allowed", lambda: set(range(1024)))
    monkeypatch.setattr(s, "DISK", tmp_path / "no-disk")
    monkeypatch.setattr(s, "cache_disk", lambda: tmp_path / "no-disk")
    monkeypatch.setattr(s, "RAMLOCK", tmp_path / "ramlock")
    monkeypatch.setattr(s, "copy", lambda commit=None: {})
    monkeypatch.setattr(s.os, "sched_setaffinity", lambda pid, cpus: None)
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "slots").write_text("r0 0\n")
    (tmp_path / "peers").write_text(f"{A} research@10.0.0.1\n")
    monkeypatch.setattr(s, "PEERS", tmp_path / "peers")
    run = subprocess.run
    monkeypatch.setattr(s.subprocess, "run", lambda argv, **kw: run(
        ["sh", "-c", argv[-1].replace(str(tmp_path / "b"), str(tmp_path / "a"))], **kw) if argv[:1] == ["ssh"] else run(argv, **kw))
    return s


def _slot_main(s, monkeypatch, where: Path, run: str, command: list[str]):
    """slot.py's main as the runner of ``run`` execs ``command`` (`in_pool`'s or `late_twin`'s) on the machine ``where``:
    "ran" when it took a slot and exec'd the job, whose slot is then free again, else its exit. It never waits: a line it
    would wait in fails the test."""
    monkeypatch.setenv("RESEARCH_RUN_ID", run)
    monkeypatch.setattr(s, "SLOTS", where / "slots")
    ran, lock = [], []
    monkeypatch.setattr(s.os, "set_inheritable", lambda fd, on: lock.append(fd))
    monkeypatch.setattr(s.os, "execvpe", lambda f, argv, env: ran.append(argv) or (_ for _ in ()).throw(Died()))
    try:
        rc = s.main(command[command.index(slot_only.SLOT_PATHS[0]) + 1:], now=lambda: datetime.fromtimestamp(T0, UTC),
                    sleep=lambda _s: pytest.fail(f"{run} waited in {where.name}'s line"))
    except Died:
        assert ran == [["nice", "-n", "19", "pytest", "-q"]]
        os.close(lock.pop())
        return "ran"
    return rc


def test_the_first_and_its_late_twin_never_both_run_and_never_neither(slot_py, tmp_path, monkeypatch, capsys):
    """The launcher's two commands, through slot.py: the first carries its run id as its pair (its claims on its own
    machine), the late twin the same pair at the first's machine. Whichever takes a slot first runs and the other leaves;
    a twin that can't reach the first's machine leaves, and the first keeps its place."""
    a, b = tmp_path / "a", tmp_path / "b"
    first = slot_only.in_pool("research", "infra", ["pytest", "-q"], 3600, 14400, twin=RUN)
    late = slot_only.late_twin(first, A, 13800)
    assert late[late.index("--twin") + 1] == f"{RUN}@{A}" and late[late.index("--wait-max") + 1] == "13800"
    assert [x for x in late if x not in (f"{RUN}@{A}", "13800")] == [x for x in first if x not in (RUN, "14400")], "the same job"
    # the twin takes a slot first: it claims on A and runs; the first, finding the claim, leaves its line
    assert _slot_main(slot_py, monkeypatch, b, TWIN, late) == "ran"
    assert json.loads((a / "twins" / RUN / "who").read_text())["run"] == TWIN and not (b / "twins").exists()
    assert _slot_main(slot_py, monkeypatch, a, RUN, first) == slot_py.NOT_STARTED
    assert f"its twin took a slot first (run {TWIN}, " in capsys.readouterr().err
    # the first starts while its twin is being launched: the twin finds the claim on A, and leaves
    other = "r20261001-090000-cccc"
    first2 = slot_only.in_pool("research", "infra", ["pytest", "-q"], 3600, 14400, twin=other)
    assert _slot_main(slot_py, monkeypatch, a, other, first2) == "ran"
    assert _slot_main(slot_py, monkeypatch, b, TWIN, slot_only.late_twin(first2, A, 13800)) == slot_py.NOT_STARTED
    assert f"its twin took a slot first (run {other}, " in capsys.readouterr().err
    # the first's machine can't be reached from the twin's: the twin leaves unrun, and the first still runs
    (tmp_path / "peers").write_text("")
    third = "r20261001-090000-dddd"
    first3 = slot_only.in_pool("research", "infra", ["pytest", "-q"], 3600, 14400, twin=third)
    assert _slot_main(slot_py, monkeypatch, b, TWIN, slot_only.late_twin(first3, A, 13800)) == slot_py.NOT_STARTED
    assert f"twin {third}@{A} can't be claimed ({A} isn't in {tmp_path / 'peers'}), so its twin there runs instead" in capsys.readouterr().err
    assert _slot_main(slot_py, monkeypatch, a, third, first3) == "ran"
    with pytest.raises(ValueError, match="carries no pair"):
        slot_only.late_twin(slot_only.in_pool("research", "infra", ["pytest"], 3600), A, 60)
    unbounded = slot_only.late_twin(slot_only.in_pool("research", "infra", ["pytest"], twin=RUN), A, 600)
    assert unbounded[unbounded.index("--wait-max") + 1] == "600" and unbounded.index("--wait-max") < unbounded.index("--"), \
        "a first with no end to its wait still has a twin that leaves its line"


# ---- the twin's launch --------------------------------------------------------------------------------------------------------

def test_the_twin_is_the_same_request_from_the_shipped_commit_on_the_other_machine(tmp_path, monkeypatch):
    """What the twin runs is what the first would have: the request the launch built, from a checkout of the commit it
    shipped, and never from what the launching checkout holds by then."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    (repo / "job.py").write_text("shipped\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "shipped"], cwd=repo, check=True)
    shipped = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "job.py").write_text("committed since\n")
    subprocess.run([*git, "commit", "-qam", "later"], cwd=repo, check=True)
    (repo / "job.py").write_text("edited, uncommitted\n")
    what = {"project": "verity", "command": slot_only.in_pool("research", "infra", ["pytest", "-q"], 3600, 14400, twin=RUN),
            "harness_args": ["--stage", "run", "--timeout", "18000"], "cwd": None, "env": {"K": "V"}, "exclusive": False,
            "mem_gb": None, "mem_wait_s": 0.0, "custody_ttl": "6h", "preflight_tool": None, "job_user": None,
            "job_namespaces": [], "gpus": None, "store_read": False}
    m = json.loads(json.dumps({"run": RUN, "root": str(tmp_path / "runs"), "home": A, "launch": what, "sends": {}}))
    twin = {"run": TWIN, "machine": B, "pool": "research", "from": A, "of": RUN, "why": MEMORY, "after_min": 10, "said": "moved"}
    seen = []

    def launch(**kw):
        seen.append({**kw, "holds": (kw["source"] / "job.py").read_text(),
                     "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=kw["source"], capture_output=True, text=True).stdout.strip(),
                     "dirty": subprocess.run(["git", "status", "--porcelain"], cwd=kw["source"], capture_output=True, text=True).stdout})
        return {"phase": "launched"}

    monkeypatch.setattr(remote, "launch", launch)
    with asker.tree(str(repo / ".git"), shipped) as source:
        assert asker.launch(source, m, twin, 13800) == {"phase": "launched"}
    (kw,) = seen
    assert (kw["holds"], kw["head"], kw["dirty"]) == ("shipped\n", shipped, ""), "the commit the first launch shipped, clean"
    assert not source.exists() and str(source) not in subprocess.run(
        ["git", "worktree", "list"], cwd=repo, capture_output=True, text=True).stdout, "and its checkout is gone afterwards"
    assert (repo / "job.py").read_text() == "edited, uncommitted\n", "the launching checkout is not touched"
    assert (kw["run_id"], kw["machine_name"], kw["local_dir"], kw["queue"]) == (TWIN, B, tmp_path / "runs" / TWIN, None)
    assert kw["command"] == slot_only.late_twin(what["command"], A, 13800) and kw["sends"] == []
    assert kw["auto"] == {"machine": B, "pool": "research", "said": ["moved"],
                          "moved": {"from": A, "of": RUN, "why": MEMORY, "after_min": 10}}
    rest = ("project", "harness_args", "cwd", "env", "exclusive", "mem_gb", "mem_wait_s", "custody_ttl", "preflight_tool",
            "job_user", "gpus", "store_read")
    assert {k: kw[k] for k in rest} == {k: what[k] for k in rest} and kw["job_namespaces"] == ()
    with pytest.raises(OSError, match="no checkout of dddddddddddd"), asker.tree(str(repo / ".git"), "d" * 40):
        pass
    # an ask reads the shipped commit's files with no checkout: the few `auto.read` reads, as the launch shipped them
    (repo / "big").mkdir()
    (repo / "big" / "other.py").write_text("x\n")
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-qm", "more"], cwd=repo, check=True)
    with asker.files(str(repo / ".git"), shipped, [Path("job.py")]) as few:
        assert [p.relative_to(few) for p in few.rglob("*")] == [Path("job.py")] and (few / "job.py").read_text() == "shipped\n"
        assert str(few) not in subprocess.run(["git", "worktree", "list"], cwd=repo, capture_output=True, text=True).stdout
    assert not few.exists()
    with pytest.raises(OSError, match="no files of "), asker.files(str(repo / ".git"), shipped, [Path("big/other.py")]):
        pass


# ---- a real one ---------------------------------------------------------------------------------------------------------------

def test_an_asker_is_a_process_of_its_own_session_that_takes_the_launch_on_its_stdin_and_ends_with_the_run(tmp_path, monkeypatch):
    """`research run` returns as it did: the asker it leaves is no child of its terminal or its process group, holds no
    pipe of the launcher's, and nothing of the request is in its arguments or on disk. Here the run it is given has
    started already, so it ends at its first look."""
    root = tmp_path / "runs"
    (root / RUN).mkdir(parents=True)
    (root / RUN / "status.json").write_text(json.dumps({"state": "done", "rc": 0}))
    (root / RUN / "slot.json").write_text(json.dumps({"state": "held", "since": SINCE, "slot": "r0"}))
    made = []
    popen = subprocess.Popen

    def spawn(argv, **kw):
        made.append((argv, kw))
        return popen(argv, **kw)

    monkeypatch.setattr(asker.subprocess, "Popen", spawn)
    monkeypatch.setenv("CURSOR_CONVERSATION_ID", "bc-1")
    m = {"run": RUN, "root": str(root), "home": A, "pool": "research", "git": "/no/repo", "commit": "c" * 40, "cpus": 4,
         "mem_gb": 16, "lean": None, "wait_s": 0, "sends": {}, "launch": {"env": {"TOKEN": "s3cret"}}}
    pid = START(root, m)
    assert os.waitpid(pid, 0) == (pid, 0)
    (argv, kw), = made
    assert argv == [sys.executable, "-m", "research.asker"] and "s3cret" not in " ".join(argv)
    assert kw["start_new_session"] is True and kw["cwd"] == "/" and kw["stdin"] == subprocess.PIPE
    assert kw["stdout"].name == kw["stderr"].name == str(root / asker.DIR / f"{RUN}.log")
    rec = json.loads((root / asker.DIR / f"{RUN}.json").read_text())
    assert (rec["pid"], rec["host"], rec["ended"]["why"], rec["next"]) == (pid, os.uname().nodename, f"it started on {A}", None)
    assert rec["by"] == AGENT, "whose asker it is: the launching agent, since a VM's host name names no VM"
    assert (root / asker.DIR / f"{RUN}.log").read_text() == (
        f"asker: {RUN} could not be read from its machine (Refused: {root / RUN} is not a launched remote run); it is read as "
        f"it was last seen\nasker of {RUN}: it started on {A}\n"), "its log says each look that could not read the run, and how it ended"
    assert "s3cret" not in "".join(p.read_text() for p in (root / asker.DIR).iterdir())


# ---- its record, and what status, --wait and cancel make of it ----------------------------------------------------------------

@pytest.fixture
def machines(tmp_path, monkeypatch):
    """A and B as directories reached through `sh -c`, and the launching VM's runs root `here`."""
    monkeypatch.setenv("RESEARCH_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    ms = {name: {"provider": "ssh", "host": name, "user": "research", "root": str(tmp_path / name), "project": "verity"} for name in (A, B)}
    monkeypatch.setattr(remote, "load_machines", lambda path=None: ms)
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": m["host"], "port": 22, "user": m["user"]})
    monkeypatch.setattr(remote, "ssh_argv", lambda host, port, **kw: ["sh", "-c"])
    return tmp_path


def _on(tmp: Path, machine: str, run: str, state: str, slot: dict, here: str | None = "here", **rec) -> Path:
    """Run ``run`` on ``machine`` in ``state`` with slot record ``slot``, and (``here``) its launcher's record on the VM."""
    d = tmp / machine / "runs" / run
    d.mkdir(parents=True)
    (tmp / machine / "requests" / run).mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "rc": {"done": 0, "failed": 75}.get(state)}))
    (d / "slot.json").write_text(json.dumps(slot))
    if here:
        (tmp / here / run).mkdir(parents=True)
        (tmp / here / run / "remote.json").write_text(json.dumps({
            "run_id": run, "machine": machine, "phase": "launched", "host": machine, "port": 22, "user": "research",
            "root": str(tmp / machine), "remote_run_dir": str(d), **rec}))
    return d


VM = f"the VM that launched it ({AGENT}, pid {os.getpid()})"


def _record(tmp: Path, **over) -> dict:
    """The record the asker of RUN keeps, a process of this VM (this test's own), written as it writes it: here and beside
    the run's request on A."""
    rec = {"schema": asker.SCHEMA, "run": RUN, "home": A, "pool": "research", "host": "cursor", "by": AGENT, "pid": os.getpid(),
           "since": SINCE, "every_min": 10, "seen": "2026-10-01T09:10:00Z", "next": "2999-01-01T00:00:00Z", "asks": [], "twin": None,
           **over}
    asker.tell(tmp / "here", rec)
    return rec


def _twin(state: str, **over) -> dict:
    return {"run": TWIN, "machine": B, "pool": "research", "from": A, "of": RUN, "why": MEMORY, "after_min": 10,
            "at": "2026-10-01T09:10:00Z", "state": state, **over}


FIRST = {"state": "waiting", "since": SINCE, "twin": RUN, "why": MEMORY}


def test_the_asker_keeps_its_record_on_its_vm_and_beside_the_runs_request_never_in_the_run_dir(machines, monkeypatch, capsys):
    """A file the asker wrote into the run dir would break the run's custody, which lists every file there; beside the
    request, another VM reads it, and says so when it can't: a record that could not be read is never read as no asker.
    A run that carries no pair of its own is asked for none."""
    d = _on(machines, A, RUN, "running", FIRST)
    rec = _record(machines)
    assert json.loads((machines / A / "requests" / RUN / asker.FILE).read_text()) == rec
    assert json.loads((machines / "here" / asker.DIR / f"{RUN}.json").read_text()) == rec
    assert sorted(p.name for p in d.iterdir()) == ["slot.json", "status.json"]
    assert sorted(p.name for p in (machines / "here" / RUN).iterdir()) == ["remote.json"]
    elsewhere = machines / "elsewhere"
    assert cli.main(["status", RUN, "--runs-dir", str(elsewhere)]) == 0
    assert asker.record(machines / "here" / RUN) == {**rec, "here": True}, "this VM's own record says it is"
    assert asker.record(elsewhere / RUN) == rec, "read from the run's machine by a VM that has no record of its own"
    with monkeypatch.context() as mp:
        mp.setattr(remote, "ssh_run", lambda ssh, argv, **kw: subprocess.CompletedProcess(argv, 255, b"", b"ssh: vy-a: timed out\n"))
        assert asker.record(elsewhere / RUN) == {"unread": "exit 255: ssh: vy-a: timed out"}
        assert asker.told(cli.R.summarize(elsewhere / RUN), T0) == [
            f"whether it has an asker, or a late twin, is not known: its asker's record beside its request on {A} could not be "
            "read (exit 255: ssh: vy-a: timed out)"]
        capsys.readouterr()
        assert asker.twin(elsewhere / RUN) is None and not asker.alive(asker.record(elsewhere / RUN), T0)
        assert capsys.readouterr().err == (f"research: {RUN}: whether it has a late twin is not known: its asker's record could not "
                                           "be read (exit 255: ssh: vy-a: timed out)\n"), "what --wait and cancel then say"
        mp.setattr(remote, "ssh_run", lambda ssh, argv, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(argv, kw["timeout"])))
        assert asker.record(elsewhere / RUN)["unread"].startswith("TimeoutExpired: ")
    (machines / A / "requests" / RUN / asker.FILE).unlink()
    assert asker.record(elsewhere / RUN) is None, "a machine that answers and holds none: the run has no asker"
    named = "r20261001-090000-eeee"
    _on(machines, A, named, "running", {"state": "waiting", "since": SINCE, "why": MEMORY})
    (machines / A / "requests" / named / asker.FILE).write_text(json.dumps(rec))
    remote.fetch(machines / "here" / named)
    assert asker.record(machines / "here" / named) is None and asker.told(cli.R.summarize(machines / "here" / named), T0) == []


def test_an_asker_is_alive_until_it_ends_or_misses_its_next_look():
    """Agent VMs share one host name, so a record's host can't say whose process its pid is: only a record this VM keeps
    itself is held to its process, and another VM's to its times."""
    gone = subprocess.Popen(["true"])
    gone.wait()
    rec = {"host": os.uname().nodename, "pid": gone.pid, "next": "2026-10-01T09:10:00Z"}
    assert asker.alive(rec, T0 + 600 + asker.LATE_S) and not asker.alive(rec, T0 + 601 + asker.LATE_S)
    assert not asker.alive({**rec, "ended": {"at": SINCE, "why": "it started"}}, T0) and not asker.alive(None, T0)
    assert not asker.alive({**rec, "next": None}, T0)
    assert asker.alive({**rec, "here": True, "pid": os.getpid()}, T0)
    assert not asker.alive({**rec, "here": True}, T0), "on its own VM, an asker whose process is gone is not alive"


# ---- how many a VM holds ------------------------------------------------------------------------------------------------------

def _askers(root: Path, n: int, **over) -> list[str]:
    """``n`` askers alive at T0 under the runs root ``root``, as their records say: processes of this VM (this test's own),
    each with its log."""
    (root / asker.DIR).mkdir(parents=True, exist_ok=True)
    runs = [f"r20261001-08{i:02d}00-aaaa" for i in range(n)]
    for run in runs:
        (root / asker.DIR / f"{run}.json").write_text(json.dumps({"run": run, "pid": os.getpid(), "next": asker._utc(T0 + 60), **over}))
        (root / asker.DIR / f"{run}.log").write_text("")
    return runs


def _leave(root: Path) -> str:
    """What the launch of RUN says of its asker at T0 (`asker.leave`), from a checkout beside the runs root ``root``."""
    subprocess.run(["git", "init", "-q", str(root.parent / "repo")], check=True)
    asks = {"cpus": 4, "mem_gb": 16, "lean": None, "home": A, "pool": "research", "commit": "c" * 40, "wait_s": 14400}
    return asker.leave(root / RUN, asks, root.parent / "repo", {"job_namespaces": ()}, [], now=lambda: T0)


def test_the_ninth_launch_of_a_vm_leaves_no_asker_and_its_line_and_status_say_the_run_will_not_move_by_itself(
        machines, monkeypatch, capsys, _no_asker_left):
    """An asker is a process of a VM every agent on it shares, and those of launches of one minute ask every machine, and
    check a twin's tree out, in the same minute: a sweep of forty runs that all wait would hold forty of them for hours.
    A VM holds eight. The next launch leaves none, and says whose hold the places; `research status` says the same of
    its run."""
    root = machines / "here"
    _on(machines, A, RUN, "running", FIRST)
    runs = _askers(root, 8)
    assert _leave(root) == (
        f"8 askers are alive on this VM already, and it holds at most 8 (their runs: {', '.join(runs[:3])} and 5 more), so this "
        f"run will not move by itself: it waits in {A}'s line; `research status RUN` shows when one of them has started or ended, "
        "and its asker ends at its next look after that")
    assert _no_asker_left == [] and not (root / asker.DIR / f"{RUN}.json").exists(), "none is started, and none recorded"
    assert cli.main(["status", RUN, "--runs-dir", str(root)]) == 0
    assert f"auto waits on memory on {A}; no asker is alive, so it will not move by itself" in capsys.readouterr().out
    monkeypatch.setattr(asker, "MAX_ASKERS", 3)
    for run in runs[3:]:
        (root / asker.DIR / f"{run}.json").unlink()
    assert f"3 askers are alive on this VM already, and it holds at most 3 (their runs: {', '.join(runs[:3])}), so this" in _leave(root)


@pytest.mark.parametrize("how", ["ended", "its-next-look-overdue", "its-process-gone"])
def test_an_asker_that_ended_or_missed_its_look_or_whose_process_is_gone_frees_its_place(tmp_path, _no_asker_left, how):
    """The count is of the askers alive now, by the rule `research status` holds this VM's own records to: one that is no
    longer alive holds no place, and neither does a log."""
    gone = subprocess.Popen(["true"])
    gone.wait()
    root = tmp_path / "runs"
    _askers(root, 8)
    _askers(root, 1, **{"ended": {"ended": {"at": SINCE, "why": "it started"}}, "its-process-gone": {"pid": gone.pid},
                        "its-next-look-overdue": {"next": asker._utc(T0 - asker.LATE_S - 1)}}[how])
    assert _leave(root).startswith("an asker of this launch stays on this VM (pid 4242)") and [m["run"] for m in _no_asker_left] == [RUN]


def test_a_record_that_can_not_be_read_holds_a_place_and_the_line_says_how_many_are_such(tmp_path, _no_asker_left):
    """An asker whose record this launch can't read may be alive: left out of the count, it would be a ninth."""
    root = tmp_path / "runs"
    runs = _askers(root, 8)
    (root / asker.DIR / f"{runs[7]}.json").write_text('{"run": ')
    assert (f"8 askers are alive on this VM already, and it holds at most 8 (their runs: {', '.join(runs[:3])} and 5 more; 1 of them "
            "by a record that could not be read, counted as alive), so this run will not move by itself") in _leave(root)
    assert _no_asker_left == []


def test_status_says_whether_an_asker_is_alive_for_a_waiting_run_or_that_it_will_not_move_by_itself(machines, capsys):
    _on(machines, A, RUN, "running", FIRST)

    def auto_lines() -> list[str]:
        assert cli.main(["status", RUN, "--runs-dir", str(machines / "here")]) == 0
        return [ln.strip() for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith("auto ")]

    assert auto_lines() == [f"auto waits on memory on {A}; no asker is alive, so it will not move by itself"]
    _record(machines)
    alive = [f"auto an asker is alive on {VM} since {SINCE}; it looked last at 2026-10-01T09:10:00Z and looks next at "
             "2999-01-01T00:00:00Z: while the run waits on memory it asks `auto` again, and moves the run to a pool that would start "
             "it now"]
    assert auto_lines() == alive
    gone = subprocess.Popen(["true"])
    gone.wait()
    _record(machines, pid=gone.pid)
    assert auto_lines() == [f"auto waits on memory on {A}; no asker is alive (the one on the VM that launched it ({AGENT}, pid "
                            f"{gone.pid}) looked last at 2026-10-01T09:10:00Z and is gone), so it will not move by itself"]
    assert cli.main(["status", RUN, "--runs-dir", str(machines / "elsewhere")]) == 0
    assert f"auto an asker is alive on the VM that launched it ({AGENT}, pid {gone.pid}) since {SINCE}; " in capsys.readouterr().out, \
        "another VM, of the same host name, knows it by its record's times and not by a pid of its own"
    _record(machines, next="2026-10-01T09:20:00Z")
    assert auto_lines() == [f"auto waits on memory on {A}; no asker is alive (the one on {VM} looked last at 2026-10-01T09:10:00Z "
                            "and is gone), so it will not move by itself"]
    _record(machines, next=None, ended={"at": "2026-10-01T09:20:00Z", "why": "its twin r1 was cancelled"})
    assert auto_lines() == [f"auto waits on memory on {A}; no asker is alive (the one on {VM} ended at 2026-10-01T09:20:00Z: its twin "
                            "r1 was cancelled), so it will not move by itself"]
    (machines / A / "runs" / RUN / "slot.json").write_text(json.dumps({**FIRST, "why": "3 check(s) ahead in line"}))
    assert auto_lines()[0].startswith(f"auto waits on {A}; no asker is alive")


def test_status_follows_a_run_that_moved_to_its_twin_and_both_say_from_where_why_and_after_how_long(machines, capsys):
    """`research status RUN` of a run that left its line for its twin would read "failed, rc 75" and stop there."""
    gone = {**FIRST, "state": "left", "left": "2026-10-01T09:12:00Z", "why": f"its twin took a slot first (run {TWIN})"}
    _on(machines, A, RUN, "failed", gone)
    moved = {"from": A, "of": RUN, "why": MEMORY, "after_min": 10}
    _on(machines, B, TWIN, "running", {"state": "held", "since": SINCE, "twin": f"{RUN}@{A}", "slot": "r0"},
        auto={"machine": B, "pool": "research", "said": ["moved"], "moved": moved})
    _record(machines, twin=_twin("started"), next=None, ended={"at": "2026-10-01T09:20:00Z", "why": "it moved"})
    assert cli.main(["status", RUN, "--runs-dir", str(machines / "here")]) == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    rows = [ln.split()[:2] for ln in out if ln.startswith("r2026")]
    assert rows == [[RUN, "failed"], [TWIN, "running"]], "the run's row, then its twin's"
    assert f"auto moved to {B} as {TWIN} after 10 min waiting on {A} ({MEMORY})" in out
    assert f"auto the late twin of {RUN}, which waited 10 min on {A} ({MEMORY})" in out
    assert cli.main(["status", RUN, "--runs-dir", str(machines / "here"), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [(r["id"], r["state"]) for r in doc["runs"]] == [(RUN, "failed"), (TWIN, "running")]
    # from a VM that launched neither: the first's record on A names the twin, and the twin's slot record names the first
    assert cli.main(["status", RUN, "--runs-dir", str(machines / "elsewhere")]) == 0
    out = [ln.strip() for ln in capsys.readouterr().out.splitlines()]
    assert [ln.split()[:2] for ln in out if ln.startswith("r2026")] == rows
    assert f"auto the late twin of {RUN}, which waited on {A}" in out


@pytest.mark.parametrize(("state", "says"), [
    ("launched", f"its late twin {TWIN} waits on {B} too, and the one that takes a slot first runs: after 10 min waiting on {A} ({MEMORY})"),
    ("launching", f"its asker was launching its late twin {TWIN} on {B} at 2026-10-01T09:10:00Z: after 10 min waiting on {A} ({MEMORY})"),
    ("left", f"its late twin {TWIN} on {B} did not run (left: {UNREACHED}): after 10 min waiting on {A} ({MEMORY})"),
])
def test_status_says_what_became_of_a_twin_that_has_not_started(machines, capsys, state, says):
    _on(machines, A, RUN, "running", FIRST)
    _record(machines, twin=_twin(state, **({"left": UNREACHED} if state == "left" else {})))
    assert cli.main(["status", RUN, "--runs-dir", str(machines / "here")]) == 0
    said = capsys.readouterr()
    assert f"auto {says}" in [ln.strip() for ln in said.out.splitlines()]
    assert (asker.twin(machines / "here" / RUN) is None) == (state == "left"), "a twin that left unrun is no longer the run"
    assert (f"research: {RUN}: its late twin {TWIN} was not found on {B} (" in said.err) == (state != "left"), \
        "a twin the record names that no machine holds: status says it does not follow the run there"


def test_wait_follows_the_run_to_its_twin_and_returns_the_twins_end(machines, capsys):
    """`--wait` on a run that moved would return 75, "not started", while the run ran to its end on another machine."""
    gone = {**FIRST, "state": "left", "why": f"its twin took a slot first (run {TWIN})"}
    _on(machines, A, RUN, "failed", gone)
    _on(machines, B, TWIN, "done", {"state": "held", "since": SINCE, "twin": f"{RUN}@{A}", "slot": "r0"})
    _record(machines, twin=_twin("launched"))
    here = machines / "here"
    wait = lambda run: cli.wait_remote(run, here / run, every_s=30, max_s=3600, sleep=lambda s: pytest.fail("slept"),  # noqa: E731
                                       clock=lambda: 0.0)
    assert wait(RUN) == 0
    err = capsys.readouterr().err
    assert f"run {RUN} left its line without starting (failed); its late twin {TWIN} is the run now" in err
    assert f"research: run {TWIN} done (rc 0)" in err
    # a first that took its slot and failed is the run: a twin its record still names is not followed
    (machines / A / "runs" / RUN / "slot.json").write_text(json.dumps({**FIRST, "state": "held", "slot": "r0"}))
    assert wait(RUN) == 75 and "late twin" not in capsys.readouterr().err


def test_cancel_cancels_what_is_left_of_the_run_and_its_twin(machines, monkeypatch, capsys):
    """A cancel that reached the first alone would leave its twin to take a slot and run the cancelled job; one that stopped
    at a first that already left would cancel nothing while the job runs on the twin's machine."""
    cancelled = []
    monkeypatch.setattr(remote, "cancel", lambda d, *, reason, by: cancelled.append(d.name) or {
        "delivered": True, "pgid": 1, "processes": 1})
    here = machines / "here"
    _on(machines, A, RUN, "running", FIRST)
    _on(machines, B, TWIN, "running", {"state": "waiting", "since": SINCE, "twin": f"{RUN}@{A}"})
    assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(here)]) == 0 and cancelled == [RUN], "no twin: as before"
    _record(machines, twin=_twin("launched"))
    del cancelled[:]
    assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(here)]) == 0 and cancelled == [RUN, TWIN]
    assert f"{RUN} has a late twin, {TWIN} on {B}: cancelling what is left of both" in capsys.readouterr().err
    # the first has left for its twin: the cancel of the run is the cancel of the twin
    (machines / A / "runs" / RUN / "status.json").write_text(json.dumps({"state": "failed", "rc": 75}))
    del cancelled[:]
    assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(here)]) == 0 and cancelled == [TWIN]
    # from a VM that launched neither, by the record beside the first's request
    del cancelled[:]
    (machines / A / "runs" / RUN / "status.json").write_text(json.dumps({"state": "running"}))
    assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(machines / "elsewhere")]) == 0 and cancelled == [RUN, TWIN]
    # its asker died while launching the twin, which never reached B: the first is cancelled, and the cancel says what is open
    _record(machines, twin=_twin("launching", run="r20261001-091000-ffff"))
    del cancelled[:]
    assert cli.main(["cancel", RUN, "--reason", "x", "--runs-dir", str(here)]) == 0 and cancelled == [RUN]
    assert f"its late twin r20261001-091000-ffff was not found on {B}" in capsys.readouterr().err
