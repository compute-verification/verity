"""`research room`: a machine's room for a queued job beside its check slots, and the launcher's reading of it on each machine."""
import json
import subprocess

from research import mem as M
from research import remote, room


def _machine(tmp_path, monkeypatch, live=("check-a.json", "check-p0.json"), available=900.0):
    locks = tmp_path / "locks"
    (locks / "proofs").mkdir(parents=True)
    (locks / "research").mkdir()
    (locks / "slots").write_text("# NAME CPUS ...\na 0-63 windows=/w lead=90 mem=560   # one full check\nb 64-119 mem=560\n"
                                 "c 120-127\n")
    (locks / "proofs" / "slots").write_text("p0 64-119 mem=260\n")
    (locks / "research" / "slots").write_text("r0 54-79 mem=40\nr1 80-105 mem=40\nr2 106-131 mem=40\n")
    (locks / "slots.bak").mkdir()                                   # a directory without slots is no pool
    deadline = tmp_path / "deadline"
    deadline.write_text("1792422000 2026-10-19T15:00:00Z infra (INF-6)\n")
    res = [{"file": f, "live": True, "outstanding": 0} for f in live] + [{"file": "check-r1.json", "live": False, "outstanding": 0}]
    monkeypatch.setattr(M, "snapshot", lambda d: {"available_gb": available, "reservations": res})
    return locks, deadline


def test_room_is_memory_free_less_every_idle_check_slots_promise_and_each_pools_free_slots(tmp_path, monkeypatch):
    locks, deadline = _machine(tmp_path, monkeypatch)
    s = room.snapshot(tmp_path / "ram", locks, deadline, now=1_792_418_400)
    assert {(sl["pool"], sl["name"]): sl["running"] for sl in s["slots"]} == {
        (None, "a"): True, (None, "b"): False, (None, "c"): None, ("proofs", "p0"): True,
        ("research", "r0"): False, ("research", "r1"): False, ("research", "r2"): False}
    assert s["idle_slots_gb"] == round(601.3 + 3 * 42.9, 1), "b's 560 GiB and the research pool's three 40 GiB, as GB"
    assert s["headroom_gb"] == round(900 - s["idle_slots_gb"], 1) and s["deadline_in_s"] == 3600
    assert s["pools"] == {"proofs": {"slots": 1, "free": 0, "mem_gb": 279.2}, "research": {"slots": 3, "free": 3, "mem_gb": 42.9}}
    assert room.snapshot(tmp_path / "ram", tmp_path / "none", tmp_path / "none", now=0) == {
        "available_gb": 900.0, "idle_slots_gb": 0, "headroom_gb": 900.0, "slots": [], "pools": {}, "deadline_in_s": None}


def test_research_room_prints_its_reading(tmp_path, monkeypatch, capsys):
    locks, deadline = _machine(tmp_path, monkeypatch, available=100.0)
    monkeypatch.setattr(room, "LOCKS", locks)
    monkeypatch.setattr(room, "DEADLINE", deadline)
    assert room.main(["--json"]) == 0 and json.loads(capsys.readouterr().out)["headroom_gb"] == round(100 - 730.0, 1)
    assert room.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("headroom -630 GB = available 100 - idle check slots 730") and "research/r0 54-79 mem 42.9 GB: idle" in out


def test_the_launcher_reads_each_machines_room_with_the_tool_it_ships(tmp_path):
    machines = {"n1": {"provider": "ssh", "host": "h1", "port": 22, "user": "research", "root": "/w/research", "python": "/usr/bin/python3",
                       "projects": ["verity"], "ramlock": "/w/ram"},
                "n2": {"provider": "ssh", "host": "h2", "port": 22, "user": "research", "root": "/w/research", "projects": ["verity"]},
                "slow": {"provider": "ssh", "host": "h3", "port": 22, "user": "research", "root": "/w/research", "projects": ["verity"]},
                "odd": {"provider": "ssh", "host": "h4", "port": 22, "user": "research", "root": "/w/research", "projects": ["verity"]},
                "other": {"provider": "ssh", "host": "h5", "port": 22, "user": "research", "root": "/w/research", "projects": ["pous"]}}
    shipped, ran = [], []

    def ship(ssh, root, pkg):
        shipped.append((ssh[-1], root, pkg))
        return f"{root}/tool/abc"

    def run(ssh, argv):
        ran.append((ssh[-1], argv))
        if "h3" in ssh[-1]:
            raise subprocess.TimeoutExpired(argv, 60)
        out = b"[1]" if "h4" in ssh[-1] else json.dumps({"headroom_gb": 12.5, "pools": {}}).encode()
        return subprocess.CompletedProcess(argv, 0, out, b"")

    got = remote.read_room(["n1", "n2", "slow", "odd", "other", "gone"], "verity", tmp_path, machines_reg=machines, ship=ship, run=run)
    assert got["n1"] == {"headroom_gb": 12.5, "pools": {}} and got["n2"]["headroom_gb"] == 12.5
    assert got["slow"] == "no answer in 60 s" and got["odd"].startswith("its answer is not a room")
    assert isinstance(got["other"], str) and isinstance(got["gone"], str)
    argv = dict((h.split("@")[-1], a) for h, a in ran)
    assert argv["h1"] == ["env", "PYTHONPATH=/w/research/tool/abc", "/usr/bin/python3", "-m", "research", "room", "--json",
                          "--ramlock", "/w/ram"]
    assert argv["h2"][2] == "python3" and "--ramlock" not in argv["h2"]
    assert {s[2] for s in shipped} == {tmp_path}
