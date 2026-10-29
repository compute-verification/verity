"""`check.py --slots` and `--record --on auto`: each machine's check slots, windows and Lean audit slots, read over ssh by this
tree's slot.py and lean_slot.py, and the machine a check is placed on from them (the lander placed checks by hand, 3 Oct)."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location("repo_check_placement", ROOT / "tools" / "verity" / "check" / "check.py")
CHECK = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CHECK)
NOW = datetime(2026, 10, 3, 17, 50, tzinfo=timezone.utc)
#: placement among two machines, whatever AUTO lists today (this module's own copy of check.py)
N1, N2 = CHECK.AUTO = ("vy-nebius-1", "vy-nebius-2")
HOLDER = {"pid": 11, "since": "2026-10-03T17:00Z", "minutes": 50, "cmd": "uv run python tools/verity/check/check.py", "run": "r-held",
          "tree": "/workspace/research/src/" + "6b3d704c8" * 4 + "abcd", "by": "https://cursor.com/agents/bc-1"}


@pytest.fixture(autouse=True)
def _placed(tmp_path, monkeypatch):
    monkeypatch.setattr(CHECK, "PLACED", tmp_path / "placed.json")


def slot(name: str, holder: dict | None = None, blocked: str | None = None, upcoming: list | None = None) -> dict:
    return {"name": name, "cpus": "0-47", "windows": None, "lead": 30, "holder": holder, "blocked": blocked, "upcoming": upcoming or []}


def waiter(priority: bool = False, by: str = "proofs") -> dict:
    return {"since": "2026-10-03T17:40Z", "minutes": 10, "priority": priority, "pid": 12, "run": "r-wait", "by": by}


def lean(free: int = 2, n: int = 2, waiting: int = 0, pool: str = "check") -> dict:
    return {pool: {"n": n, "slots": [{"index": i, "past_n": False, "holder": None if i < free else HOLDER} for i in range(n)],
                   "waiters": [waiter() for _ in range(waiting)]}}


def machine(slots: list[dict], waiters: list[dict] = (), pools: dict | None = None) -> dict:
    return {"check": {"file": "/workspace/research/locks/slots", "now": "2026-10-03T17:50Z", "slots": slots, "waiters": list(waiters)},
            "lean": {"file": "/workspace/research/locks/lean-slots", "now": "2026-10-03T17:50Z", "pools": pools}}


def test_the_first_machine_with_a_free_check_slot_and_a_free_lean_check_slot_is_taken():
    busy = machine([slot("c", HOLDER), slot("a")], pools=lean(free=0))        # a check slot, but both Lean check slots held
    free = machine([slot("d")])                                                 # no lean-slots file: Lean audits run unslotted
    got, why, skipped = CHECK.choose([N1, N2], {N1: busy, N2: free})
    assert (got, skipped) == (N2, [])
    assert why == "a check slot and a Lean check slot are free (check slots: d free; 0 waiting ahead; no Lean audit cap)"
    both = machine([slot("c", HOLDER), slot("a")], pools=lean(free=1))
    assert CHECK.choose([N1, N2], {N1: both, N2: free})[0] == N1                # both free: the first listed
    assert CHECK.choose([N2, N1], {N1: both, N2: free})[0] == N2
    assert CHECK.placement(both) == (True, 0, "check slots: c held, a free; 0 waiting ahead; Lean check: 1 of 2 free")
    assert CHECK.placement(machine([slot("a")], pools=lean(free=1, waiting=1)))[0] is False   # its one free Lean slot is spoken for
    assert CHECK.placement(machine([slot("a")], pools=lean(free=1, pool="any")))[0] is True  # no `check` pool: lean_slot takes `any`


def test_a_slot_a_window_blocks_now_or_within_its_lead_is_not_free(tmp_path, monkeypatch):
    """Node 2's slot d with the 18:05Z timed window 15 min away: free, but a check there would run into the window."""
    w = tmp_path / "windows"
    w.write_text("2026-10-03T18:05Z 20 # served qwen3-8b timed, compute-accounting\n2026-10-03T21:00Z 30 # served window 6\n")
    (tmp_path / "slots").write_text(f"d 0-47 windows={w} lead=30\n")
    monkeypatch.setattr(CHECK.SLOT, "SLOTS", tmp_path / "slots")
    n2 = {"check": CHECK.SLOT.status_doc(NOW), "lean": {"file": "/absent", "pools": None}}
    assert n2["check"]["slots"][0]["blocked"] == "window 2026-10-03T18:05Z (20 min) is open or starts within 30 min"
    n1 = machine([slot("c", HOLDER)], [waiter()], pools=lean())
    got, why, _ = CHECK.choose([N1, N2], {N1: n1, N2: n2})
    assert got == N2 and why.startswith("the fewest waiting ahead, as no machine has a check slot and a Lean check slot free")
    assert "check slots: d blocked; 0 waiting ahead" in why
    w.write_text("2026-10-03T18:25Z 20 # 35 min away: past the lead\n")
    n2["check"] = CHECK.SLOT.status_doc(NOW)
    assert CHECK.placement(n2)[0] is True
    assert [ln for ln in CHECK.slots_block(N2, n2) if "window" in ln] == [
        "    window 2026-10-03T18:25Z (20 min), starts in 35 min: 35 min away: past the lead"]


def test_otherwise_the_fewest_waiters_ahead_and_with_priority_only_the_priority_ones():
    n1 = machine([slot("c", HOLDER)], [waiter(), waiter()], pools=lean())          # two lane checks waiting
    n2 = machine([slot("d", HOLDER)], [waiter(priority=True, by="lander")])         # one train check waiting
    assert CHECK.choose([N1, N2], {N1: n1, N2: n2})[0] == N2                       # a lane check: 1 ahead beats 2
    got, why, _ = CHECK.choose([N1, N2], {N1: n1, N2: n2}, priority=True)
    assert got == N1 and "0 waiting ahead with priority" in why                    # a train check goes ahead of lane waiters
    tied = machine([slot("d", HOLDER)], [waiter()])
    assert CHECK.choose([N1, N2], {N1: machine([slot("c", HOLDER)], [waiter()]), N2: tied})[0] == N1   # a tie: the first listed
    spoken = machine([slot("a")], [waiter()])                                      # a free slot the waiter ahead takes next
    assert CHECK.placement(spoken)[:2] == (False, 1) and CHECK.placement(spoken, priority=True)[:2] == (True, 0)
    trains = machine([{**slot("d"), "trains": True}, slot("e", HOLDER)])           # node 2's d is only=trains
    assert CHECK.placement(trains) == (False, 0, "check slots: e held; 0 waiting ahead; no Lean audit cap")
    assert CHECK.placement(trains, priority=True)[:2] == (True, 0)
    with pytest.raises(ValueError, match="every slot in .* is for train checks"):
        CHECK.placement(machine([{**slot("d"), "trains": True}]))


def test_a_train_that_needs_a_train_slots_memory_skips_a_machine_of_smaller_slots_even_with_one_free():
    small = machine([{**slot("c"), "mem": 64}, {**slot("d", HOLDER), "mem": 64}])     # cpu-3's lane slots
    big = machine([{**slot("b", HOLDER), "mem": 300, "trains": True}])                 # a train slot, held
    got, why, skipped = CHECK.choose([N1, N2], {N1: small, N2: big}, priority=True, min_mem=CHECK.TRAIN_GB)
    assert got == N2 and "b held" in why and skipped == [f"{N1}: no slot in /workspace/research/locks/slots holds this check's "
                                                         "300 GB (c 64 GB, d 64 GB): it takes a train slot"]
    assert CHECK.choose([N1, N2], {N1: small, N2: big}, priority=True)[0] == N1         # a Lean check takes the free small one
    assert CHECK.placement(machine([{**slot("a"), "mem": None}]), True, CHECK.TRAIN_GB)[0] is True   # no mem=: any size


def test_a_machine_whose_slots_can_t_be_read_is_skipped_by_name():
    free = machine([slot("d")])
    docs = {N1: "exit 255: ssh: connect to host 81.85.2.165 port 22: Connection timed out", N2: free}
    assert CHECK.choose([N1, N2], docs) == (N2, CHECK.choose([N2], docs)[1], [f"{N1}: {docs[N1]}"])
    for bad, why in ((machine([slot("a")]) | {"lean": {"error": "Refused: 'audit two' is not `POOL N`"}},
                      "its Lean audit slots can't be read: Refused"),
                     ({"check": {"error": "FileNotFoundError: /workspace/pouw/fill/windows"}, "lean": {"pools": None}},
                      "its check slots can't be read: FileNotFoundError"),
                     (machine([slot("a")], pools=lean(pool="audit")), "lists neither pool check nor any"),
                     (machine([]), "lists no slots"),
                     ({"check": {"slots": []}}, "malformed (KeyError: 'lean')")):
        got, _, skipped = CHECK.choose([N1, N2], {N1: bad, N2: free})
        assert got == N2 and len(skipped) == 1 and skipped[0].startswith(f"{N1}: ") and why in skipped[0], (why, skipped)


def test_a_machine_whose_deadline_is_nearer_than_a_full_check_is_skipped_by_name():
    """The lander's ask (9 Oct): node 2 stops at 05:30Z on 10 Oct, so from 03:00Z a full check placed there would be cut off."""
    n1 = machine([slot("c", HOLDER)], [waiter()], pools=lean())
    n2 = {**machine([slot("d")]), "deadline_in_s": CHECK.FULL_CHECK_S - 60}
    got, _, skipped = CHECK.choose([N1, N2], {N1: n1, N2: n2})
    assert got == N1 and skipped == [f"{N2}: its deadline is 149 min away, under the 150 min a full check takes"]
    assert CHECK.twins([N1, N2], {N1: n1, N2: n2}, N1) == []                      # and it gets no twin
    assert CHECK.choose([N1, N2], {N1: n1, N2: {**n2, "deadline_in_s": CHECK.FULL_CHECK_S + 60}})[0] == N2
    assert CHECK.choose([N1, N2], {N1: n1, N2: {**n2, "deadline_in_s": None}})[0] == N2      # no deadline file


def test_with_no_machine_readable_auto_refuses_and_launches_nothing(monkeypatch, capsys):
    monkeypatch.setattr(CHECK, "read_slots", lambda names: {n: "exit 255: no route to host" for n in names})
    calls = []
    monkeypatch.setattr(CHECK.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert CHECK.choose([N1, N2], {N1: "x", N2: "y"})[0] is None
    assert CHECK.record(["--on", "auto"]) == 2 and [c for c in calls if c[:2] != ["git", "rev-parse"]] == []
    out = capsys.readouterr()
    assert f"check: --on auto: skipping {N1}: exit 255: no route to host" in out.out
    assert f"check: --on auto refused: no machine's slots could be read ({N1}, {N2}); nothing launched" in out.err


def test_record_on_auto_or_a_list_launches_on_the_machine_chosen(monkeypatch, capsys):
    docs = {N1: machine([slot("c", HOLDER)], [waiter()], pools=lean()), N2: machine([slot("d")])}
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    asked = []
    monkeypatch.setattr(CHECK, "read_slots", lambda names: asked.append(names) or {n: docs[n] for n in names})
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    calls = []
    monkeypatch.setattr(CHECK.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    assert CHECK.record(["--on", "auto", "--priority"]) == 0
    cmd = calls[-1]
    assert asked[-1] == [N1, N2] and cmd[cmd.index("--on") + 1] == N2 and cmd.count("--on") == 1
    assert cmd[cmd.index("--") + 1:][:3] == ["python3", "tools/verity/check/slot.py", "--priority"]
    assert f"check: --on auto: {N2}: a check slot and a Lean check slot are free (check slots: d free" in capsys.readouterr().out
    assert CHECK.record(["--on", f"{N1},{N2}"]) == 0 and asked[-1] == [N1, N2]
    assert calls[-1][calls[-1].index("--on") + 1] == N2
    assert CHECK.record(["--on", N1]) == 0 and len(asked) == 2                     # one machine named: no reading
    assert calls[-1][calls[-1].index("--on") + 1] == N1


def test_with_no_slot_free_anywhere_auto_launches_twins_on_every_machine_that_can_run_it(monkeypatch, capsys):
    """9 Oct: four checks waited up to 54 min in node 1's line while a node 2 slot was free, each placed when no slot was free
    and moved by hand. Now such a check waits in each line as twins (`slot.py --twin`) and runs in the first slot to free."""
    docs = {N1: machine([slot("c", HOLDER)], [waiter()], pools=lean()), N2: machine([slot("d", HOLDER)])}
    monkeypatch.setattr(CHECK, "read_slots", lambda names: {n: docs[n] for n in names})
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.delenv("RESEARCH_LANE", raising=False)
    monkeypatch.delenv("RESEARCH_AGENT_URL", raising=False)
    calls, launched, rcs = [], [], []
    monkeypatch.setattr(CHECK.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))

    class Popen:
        def __init__(self, cmd, **kw):
            launched.append(cmd)
            self.rc = rcs[len(launched) - 1]

        def wait(self):
            return self.rc

    monkeypatch.setattr(CHECK.subprocess, "Popen", Popen)
    rcs[:] = [CHECK.SLOT.NOT_STARTED, 0]
    assert CHECK.record(["--on", "auto"]) == 0                                    # the twin that ran passed
    assert [c[c.index("--on") + 1] for c in launched] == [N2, N1]                  # choose's pick (fewest ahead) holds the claims
    pair = launched[0][launched[0].index("--twin") + 1]
    assert [c[c.index("--") + 1:][:5] for c in launched] == [["python3", "tools/verity/check/slot.py", "--twin", t, "--"]
                                                             for t in (pair, f"{pair}@{N2}")]
    assert f"check: --on: twins {pair} on {N2}, {N1}: the first to take a check slot runs" in capsys.readouterr().out
    launched.clear()
    rcs[:] = [1, CHECK.SLOT.NOT_STARTED]
    assert CHECK.record(["--on", "auto"]) == 1                                    # it ran and failed
    launched.clear()
    docs[N1] = "exit 255: no route to host"                                       # one machine left: it waits there alone
    assert CHECK.record(["--on", "auto"]) == 0 and launched == [] and "--twin" not in calls[-1]
    docs[N1], docs[N2] = machine([slot("c")], pools=lean()), machine([slot("d", HOLDER)])
    assert CHECK.record(["--on", "auto"]) == 0 and launched == []                 # a slot free: no twins
    assert calls[-1][calls[-1].index("--on") + 1] == N1


def test_a_quick_tier_on_auto_goes_where_a_check_slot_is_free_whatever_its_lean_audit_slots():
    """6 Oct: quick tiers waited past their timeout in node 1's line, 10 deep, while node 2 had free check slots. The quick
    tier takes no Lean audit slot, so a machine whose Lean slots are all held, or can't be read, is no worse for it."""
    from research import queue as Q

    n1 = machine([slot("a", HOLDER), slot("c", HOLDER)], [waiter() for _ in range(10)], pools=lean())
    n2 = machine([slot("d"), slot("e", HOLDER)], pools=lean(free=0, waiting=2))
    got, why = Q.place("auto", ROOT, read=lambda names: {N1: n1, N2: n2})
    assert got == N2 and why.startswith("a check slot and a Lean check slot are free (check slots: d free, e held; 0 waiting ahead")
    assert CHECK.choose([N1, N2], {N1: n1, N2: n2})[1].startswith("the fewest waiting ahead")       # a check's pick: no room on n2
    n2_lean_unread = n2 | {"lean": {"error": "PermissionError: /workspace/research/locks/lean-slots"}}
    assert Q.place(f"{N1},{N2}", ROOT, read=lambda names: {N1: n1, N2: n2_lean_unread})[0] == N2
    got, why = Q.place(f"{N1},{N2}", ROOT, read=lambda names: {n: "exit 255: no route to host" for n in names})
    assert got is None and why == f"no machine's slots could be read; skipped {N1}: exit 255: no route to host; skipped {N2}: exit 255: no route to host"


def test_read_slots_runs_the_payload_on_each_machine_and_names_why_one_can_t_be_read():
    from research.remote import Refused

    good = machine([slot("d")])
    answers = {"m-ok": subprocess.CompletedProcess([], 0, json.dumps(good).encode(), b""),
               "m-down": subprocess.CompletedProcess([], 255, b"", b"Warning: ...\nssh: connect to host 10.0.0.1 port 22: No route to host\n"),
               "m-garbled": subprocess.CompletedProcess([], 0, b"Traceback", b""),
               "m-list": subprocess.CompletedProcess([], 0, b"[]", b"")}
    seen = []

    def reach(name):
        if name == "m-unlisted":
            raise Refused(f"machine {name!r} is not in the registry; only listed machines may be used")
        return ["ssh", f"research@{name}", "python3 -"]

    def run(argv, payload):
        seen.append((argv, payload))
        if argv[1] == "research@m-slow":
            raise subprocess.TimeoutExpired(argv, CHECK.SLOTS_TIMEOUT_S)
        return answers[argv[1].split("@")[1]]

    names = ["m-ok", "m-down", "m-garbled", "m-list", "m-unlisted", "m-slow"]
    got = CHECK.read_slots(names, reach=reach, run=run)
    assert list(got) == names and got["m-ok"] == good
    assert got["m-down"] == "exit 255: ssh: connect to host 10.0.0.1 port 22: No route to host"
    assert got["m-garbled"] == "its answer is not JSON: b'Traceback'" and got["m-list"].startswith("its answer is not a JSON object")
    assert got["m-unlisted"] == "machine 'm-unlisted' is not in the registry; only listed machines may be used"
    assert got["m-slow"] == f"no answer in {CHECK.SLOTS_TIMEOUT_S} s"
    assert {p for _, p in seen} == {CHECK.slots_payload()} and len(seen) == 5


def test_a_machine_is_reached_as_research_run_reaches_it(tmp_path, monkeypatch):
    """The registry entry `research run --on` resolves (bound to project verity), over its ssh target, with no prompt."""
    from research.pods import runpod
    from research.remote import Refused

    monkeypatch.setattr(runpod, "SSH_KEY", tmp_path / "key")
    monkeypatch.setattr(runpod, "KNOWN_HOSTS", tmp_path / "known_hosts")
    monkeypatch.delenv("RUNPOD_SSH_KEY_B64", raising=False)
    machines = {N1: {"provider": "ssh", "host": "81.85.2.165", "user": "research", "project": "verity"},
                "other": {"provider": "ssh", "host": "10.0.0.2", "project": "veritor"}}
    argv = CHECK.reach(N1, machines)
    assert argv[0] == "ssh" and argv[-2:] == ["research@81.85.2.165", "python3 -"]
    assert argv[argv.index("-i") + 1] == str(tmp_path / "key") and argv[argv.index("-p") + 1] == "22"
    assert {"BatchMode=yes", "ConnectTimeout=15", f"UserKnownHostsFile={tmp_path / 'known_hosts'}"} <= set(argv)
    with pytest.raises(Refused, match="belongs to project"):
        CHECK.reach("other", machines)
    with pytest.raises(Refused, match="is not in"):
        CHECK.reach("vy-nebius-9", machines)


def _hold(path: Path) -> int:
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_the_payload_reads_both_files_from_stdin_where_neither_script_is_on_disk(tmp_path):
    """The real program a machine runs (`python3 -`), from a directory without slot.py beside it: lean_slot.py takes the slot.py
    loaded before it rather than its own file's neighbour."""
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "slots").write_text(f"c 8-31\na 32-63 windows={tmp_path / 'windows'} lead=30\n")
    (tmp_path / "windows").write_text("2099-01-01T00:00Z 30 # far off\n")
    (locks / "lean-slots").write_text("check 2\naudit 2\n")
    held = [_hold(locks / "check-c.lock"), _hold(locks / "lean-check-0.lock")]
    env = {"PATH": os.environ["PATH"], "CHECK_SLOTS": str(locks / "slots"), "LEAN_SLOTS": str(locks / "lean-slots")}
    r = subprocess.run([sys.executable, "-"], input=CHECK.slots_payload(), cwd=tmp_path, env=env, capture_output=True, check=True)
    doc = json.loads(r.stdout)
    doc["deadline_in_s"] = None             # the payload reads the machine's own /etc/research/deadline, which this test can't set
    assert [(s["name"], s["holder"]["pid"] if s["holder"] else None, s["blocked"]) for s in doc["check"]["slots"]] == [
        ("c", os.getpid(), None), ("a", None, None)]
    pools = doc["lean"]["pools"]
    assert [s["holder"]["pid"] if s["holder"] else None for s in pools["check"]["slots"]] == [os.getpid(), None]
    assert CHECK.placement(doc) == (True, 0, "check slots: c held, a free; 0 waiting ahead; Lean check: 1 of 2 free")
    (locks / "lean-slots").write_text("check two\n")
    doc = json.loads(subprocess.run([sys.executable, "-"], input=CHECK.slots_payload(), cwd=tmp_path, env=env,
                                    capture_output=True, check=True).stdout)
    assert doc["lean"]["error"].startswith("Refused: ") and doc["check"]["slots"][0]["name"] == "c"
    for fd in held:
        os.close(fd)


def test_the_view_shows_each_machine_s_slots_windows_lean_slots_and_waiters_then_auto_s_pick(capsys):
    window = {"start": "2026-10-03T18:05Z", "minutes": 20, "starts_in_min": 15, "note": "served qwen3-8b timed"}
    n1 = machine([slot("c", HOLDER), slot("a")], [waiter(priority=True, by="lander")], pools=lean(free=1, waiting=1))
    n2 = machine([slot("d", blocked="window 2026-10-03T18:05Z (20 min) is open or starts within 30 min", upcoming=[window])])
    assert CHECK.slots_view([], read=lambda names: {N1: n1, N2: n2}) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"{N1} (at 2026-10-03T17:50Z)",
        "  check c  cores 0-47  held since 2026-10-03T17:00Z (50 min), run r-held, tree 6b3d704c86b3, by bc-1",
        "  check a  cores 0-47  free",
        "  check line: 1 waiting (1 with priority): since 2026-10-03T17:40Z (10 min), run r-wait, by lander, priority",
        "  Lean check: 1 of 2 free; check/1 since 2026-10-03T17:00Z (50 min), run r-held, tree 6b3d704c86b3, by bc-1; "
        "1 waiting: since 2026-10-03T17:40Z (10 min), run r-wait, by proofs",
        f"{N2} (at 2026-10-03T17:50Z)",
        "  check d  cores 0-47  free; blocked: window 2026-10-03T18:05Z (20 min) is open or starts within 30 min",
        "    window 2026-10-03T18:05Z (20 min), starts in 15 min: served qwen3-8b timed",
        "  check line: nobody waiting",
        "  Lean audit slots: no cap (/workspace/research/locks/lean-slots): audits run unslotted",
        f"--on auto would take {N2}: the fewest waiting ahead, as no machine has a check slot and a Lean check slot free "
        "(check slots: d blocked; 0 waiting ahead; no Lean audit cap)"]
    assert CHECK.slots_view(["--json", N2], read=lambda names: {n: n2 for n in names}) == 0
    assert json.loads(capsys.readouterr().out) == {N2: n2}
    assert CHECK.slots_view(["m"], read=lambda names: {"m": "exit 255: ssh"}) == 1
    assert capsys.readouterr().out.splitlines() == ["m: can't be read: exit 255: ssh", "--on m would refuse: no machine's slots could be read"]
    assert CHECK.slots_view(["--bogus"]) == 2


def test_checks_placed_at_once_go_to_different_machines_until_each_shows_in_its_line(tmp_path):
    """23 Oct 13:04Z: drive's four parallel launches all read cpu-1's one free slot as theirs, and three queued there for an hour
    while cpu-4's stood free."""
    docs = {N1: machine([slot("a")]), N2: machine([slot("b")])}
    read = lambda names: json.loads(json.dumps({n: docs[n] for n in names}))
    path, t = tmp_path / "placed.json", [1000.0]
    first = CHECK.place([N1, N2], "a" * 40, True, read=read, path=path, now=lambda: t[0])[1]
    second = CHECK.place([N1, N2], "b" * 40, True, read=read, path=path, now=lambda: t[0])[1]
    assert (first, second) == (N1, N2)
    assert CHECK.place([N1, N2], "a" * 40, True, read=read, path=path, now=lambda: t[0])[1] == N1     # a relaunch: its own claim
    docs[N1] = machine([slot("a", HOLDER | {"tree": "/workspace/research/src/" + "a" * 40})])
    docs[N2] = machine([slot("b")])
    got = CHECK.place([N1, N2], "c" * 40, True, read=read, path=path, now=lambda: t[0])  # a runs; b's claim holds N2:
    assert got[1] == N1 and CHECK.twins([N1, N2], got[0], N1, True) == [N1, N2]                       # no room, so twins
    t[0] += CHECK.PLACED_S                                                                              # until b's claim expires
    assert CHECK.place([N1, N2], "d" * 40, True, read=read, path=path, now=lambda: t[0])[1] == N2
