"""`research run --on auto`'s placement (research.auto): a free CPU pool slot first, then node 2's share if it starts the job now, so
small runs never wait behind two that hold all 48 of the share's cores for hours while cpu-1's cores sit idle."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research import auto, cores, remote
from research.cli import main

N2, CPU1, CPU5 = auto.AUTO_NODE, "vy-nebius-cpu-1", "vy-nebius-cpu-5"
#: node 2's answer as `auto.read` gives it while the node is stopped
DOWN = f"{auto.NO_SSH} (Connection timed out)"


def share(free: int = 0, waiters: int = 0, size: int = 48, fits: bool | None = None) -> dict:
    return {"share": {"share": size, "claims": 2, "waiters": waiters, "free": free, "fits": free >= 4 if fits is None else fits}}


def slot(name: str, cpus: str = "8-19", mem: int = 40, held: bool = False, blocked: str | None = None, trains: bool = False) -> dict:
    return {"name": name, "cpus": cpus, "mem": mem, "trains": trains, "holder": {"pid": 1} if held else None, "blocked": blocked}


def pool(*slots: dict, waiters: int = 0, minutes: int = 0) -> dict:
    return {"research": {"file": "/workspace/research/locks/research/slots", "slots": list(slots),
                         "waiters": [{"pid": i, "minutes": minutes} for i in range(waiters)]}}


def test_a_pool_slot_free_now_wins_even_when_node_2_could_start_the_job():
    """CPU work must not leave node 2's GPUs idle while the CPU nodes have cores free: CPU jobs go to the pools."""
    target, why = auto.choose({N2: share(free=8), CPU1: pool(slot("r0"))}, 4, 16)
    assert target == (CPU1, "research") and why.endswith("a slot is free there now")
    target, why = auto.choose({N2: share(free=0, waiters=56), CPU1: pool(slot("r0", held=True), slot("r1"))}, 4, 16)
    assert target == (CPU1, "research")
    assert why.startswith(f"{N2}'s share: 0 of 48 CPUs free, 2 claim(s), 56 waiting; {CPU1}'s research pool: slots r0 held, r1 free")
    target, _ = auto.choose({N2: share(free=8), CPU1: pool(slot("r0", held=True)), CPU5: pool(slot("r0"))}, 4, 16)
    assert target == (CPU5, "research"), "cpu-5's pool when cpu-1's is busy"


def test_an_idle_nodes_research_pool_takes_a_job_while_the_busy_ones_queue():
    """cpu-7, 8, 9 and 11 sat near load 0 while cpu-1, 5 and 6 queued 20 or more runs each, outside SPILL."""
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=22)
    for idle in ("vy-nebius-cpu-7", "vy-nebius-cpu-8", "vy-nebius-cpu-9", "vy-nebius-cpu-11"):
        assert (idle, "research") in auto.SPILL
        target, why = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, idle: pool(slot("r0", cpus="12-33"))}, 4, 16)
        assert target == (idle, "research") and why.endswith("a slot is free there now")


def test_a_run_that_takes_a_lean_build_slot_goes_only_where_the_machines_lean_slots_have_build():
    """`lean_changed.py --build` runs went by `--on auto` to a machine whose `build` pool was the queue's, and were refused in
    0 s, again and again. A machine whose Lean slots give `build` none is skipped for them; one with no Lean slots file, or
    whose file couldn't be read, isn't."""
    cpu9 = "vy-nebius-cpu-9"
    drives = {**pool(slot("r0", cpus="12-33")), "lean": {"check": 2}}
    lanes = {**pool(slot("r0", held=True), waiters=3), "lean": {"build": 8, "audit": 2}}
    docs = {N2: share(waiters=5), cpu9: drives, CPU5: lanes}
    assert auto.choose(docs, 4, 16)[0] == (cpu9, "research"), "any other run takes cpu-9's free slot"
    target, why = auto.choose(docs, 4, 16, lean_pool="build")
    assert target == (CPU5, "research") and f"{cpu9}'s research pool skipped: its Lean slots give `build` none" in why
    for lean in (None, {"error": "OSError: denied"}):
        assert auto.choose({N2: share(waiters=5), cpu9: {**drives, "lean": lean}}, 4, 16, lean_pool="build")[0] == (cpu9, "research")
    assert auto.lean_pool(["python3", "tools/verity/lean/lean_changed.py", "--no-wait", "--build", "X.lean"]) == "build"
    assert auto.lean_pool(["uv", "run", "verity", "lean", "try", "X.lean"]) == "build"
    assert auto.lean_pool(["uv", "run", "pytest", "tools/verity/lean/tests/test_lean_changed.py"]) is None


def test_slots_blocked_on_memory_are_not_capacity_when_ranking_lines():
    """cpu-1's 13 idle slots all waited on memory reservations while 28 runs queued; counted as capacity, its line looked
    shortest and every `--on auto` job queued there behind them."""
    mem = "-1 GB of memory free (481 GB more is reserved by live runs that haven't reached it), under the 40 GB a check here may peak at"
    cpu1 = pool(*(slot(f"r{i}", held=True) for i in range(8)), *(slot(f"r{i}", blocked=mem) for i in range(8, 21)), waiters=28)
    cpu6 = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=6)
    target, why = auto.choose({N2: share(waiters=5), CPU1: cpu1, "vy-nebius-cpu-6": cpu6}, 4, 16)
    assert target == ("vy-nebius-cpu-6", "research"), why


def test_a_pool_whose_line_is_stalled_on_memory_never_takes_the_job():
    """Twice a job went to a line stalled on memory once another node's probe timed out: cpu-3's head asked 256 GB for hours
    while its other slots, which declare no mem=, showed free; elsewhere every idle slot waited on memory with none running."""
    mem = "-1 GB of memory free (481 GB more is reserved by live runs that haven't reached it), under the 40 GB a check here may peak at"
    busy = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=6)
    stuck = pool(*(slot(f"r{i}", mem=None) for i in range(5)), waiters=1, minutes=auto.STALL_MIN)
    target, why = auto.choose({N2: share(waiters=56), CPU1: stuck, CPU5: busy}, 4, 16)
    assert target == (CPU5, "research") and f"{CPU1}'s research pool skipped: its line is stalled: a run has waited 5 min" in why
    target, why = auto.choose({N2: "no answer in 30 s", CPU1: stuck, CPU5: "no answer in 30 s"}, 4, 16)
    assert target == (CPU1, "research") and "every pool that fits is stalled" in why, "a stalled line beats refusing the job"
    longer = pool(*(slot(f"r{i}", mem=None) for i in range(5)), waiters=4, minutes=auto.STALL_MIN)
    assert auto.choose({N2: "no answer in 30 s", CPU1: longer, CPU5: stuck}, 4, 16)[0] == (CPU5, "research"), "the shorter"
    assert auto.choose({N2: share(free=48), CPU1: stuck}, 4, 16)[0] is None, "node 2 starting it now beats a stalled line"
    starved = pool(*(slot(f"r{i}", blocked=mem) for i in range(13)))
    target, why = auto.choose({N2: share(waiters=56), CPU1: starved, CPU5: busy}, 4, 16)
    assert target == (CPU5, "research") and "every slot of it that fits waits on memory and none runs" in why
    fresh = pool(*(slot(f"r{i}", mem=None) for i in range(5)), waiters=1, minutes=auto.STALL_MIN - 1)
    assert auto.choose({N2: share(waiters=56), CPU1: fresh, CPU5: busy}, 4, 16)[0] == (CPU1, "research"), "a waiter between polls"
    mixed = pool(slot("r0", held=True), *(slot(f"r{i}", blocked=mem) for i in range(1, 4)), waiters=2, minutes=60)
    assert auto.choose({N2: share(waiters=56), CPU1: mixed}, 4, 16)[0] == (CPU1, "research"), "a running slot drains it"


def test_cpu_3s_research_pool_takes_a_job_while_cpu_1s_and_cpu_5s_are_full():
    """cpu-3's research pool is in SPILL, so it doesn't sit idle while cpu-1's and cpu-5's lines are long."""
    cpu3 = "vy-nebius-cpu-3"
    assert (cpu3, "research") in auto.SPILL
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=22)
    target, why = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, cpu3: pool(slot("b4", cpus="0-15", mem=48))}, 4, 16)
    assert target == (cpu3, "research") and why.endswith("a slot is free there now")
    target, _ = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, cpu3: pool(slot("r0", cpus="0-15", mem=None))}, 4, 16)
    assert target == (cpu3, "research"), "27 Oct 23:45Z: cpu-3's slots declare no mem=, which slot.py raises to the run's --mem-gb"


def test_with_nothing_free_the_pool_with_the_shortest_line_per_fitting_slot_wins():
    full = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=3)                      # (3 + 1) / 4 = 1
    assert auto.choose({N2: share(waiters=56), CPU1: full}, 4, 16)[0] == (CPU1, "research")
    assert auto.choose({N2: share(free=8, waiters=1), CPU1: full}, 4, 16)[0] == (CPU1, "research"), "node 2 has a line too"
    busier = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=7)                    # (7 + 1) / 4 = 2
    target, why = auto.choose({N2: share(), CPU1: busier, CPU5: full}, 4, 16)
    assert target == (CPU5, "research") and why.endswith(f"the shortest line per run that fits is {CPU5}'s research pool")


def test_with_every_pool_full_node_2_takes_a_job_it_can_start_now():
    """--on auto once queued a lane's suites behind cpu-6's 8 held slots and 9 waiters while node 2's 48 CPUs sat free with no
    one waiting."""
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=9)
    target, why = auto.choose({N2: share(free=48), CPU1: full, CPU5: full}, 4, 16)
    assert target is None and why.endswith(f"no pool slot is free, and it starts on {N2} now")
    assert auto.choose({N2: share(free=2), CPU1: full}, 4, 16)[0] == (CPU1, "research"), "too few free CPUs there: the pool's line"
    assert auto.choose({N2: share(free=48), CPU1: pool(slot("r0"))}, 4, 16)[0] == (CPU1, "research"), "a free pool slot still wins"


def test_a_slot_too_small_trains_only_or_blocked_never_takes_the_job():
    docs = {N2: share(waiters=56)}
    for p, cpus, mem in ((pool(slot("r0", cpus="8-11")), 8, 16), (pool(slot("r0", mem=40)), 4, 64),
                         (pool(slot("b", cpus="54-191", mem=560, trains=True)), 4, 16)):
        target, why = auto.choose({**docs, CPU1: p}, cpus, mem)
        assert target is None and f"{CPU1}'s research pool skipped: no slot of" in why
    target, why = auto.choose({**docs, CPU1: pool(slot("r0", blocked="its window opens"))}, 4, 16)
    assert target == (CPU1, "research") and "r0 blocked" in why and "shortest line" in why, "a blocked slot is no free one"


def test_a_pool_every_fitting_slot_of_which_a_window_closes_never_takes_the_job():
    """A window that closes every fitting slot of a pool for days leaves that pool's line empty, which must not rank it first:
    a job placed there waits until the window ends."""
    win = "window 2026-11-06T19:25:00Z (5315 min) is open or starts within 30 min"
    closed = pool(*(slot(f"b{i}", blocked=win) for i in range(4)))
    busy = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=6)
    target, why = auto.choose({N2: share(waiters=56), CPU1: closed, CPU5: busy}, 4, 16)
    assert target == (CPU5, "research") and f"{CPU1}'s research pool skipped: every slot of" in why
    target, why = auto.choose({N2: share(free=48), CPU1: closed}, 4, 16)
    assert target is None and why.endswith(f"no pool fits, and it starts on {N2} now")
    half = pool(slot("b0", blocked=win), slot("b1", held=True))
    assert auto.choose({N2: share(waiters=56), CPU1: half}, 4, 16)[1].count("b0 blocked") == 1, "an open slot keeps the pool"


def test_a_machine_that_cant_be_read_is_skipped_and_with_neither_a_pool_nor_node_2_the_job_is_refused():
    """With node 2 stopped, a job too big for a pool slot would go to its queue and be refused there on a stale admission
    window."""
    target, why = auto.choose({N2: "exit 255: ssh: connect refused", CPU1: pool(slot("r0"))}, 4, 16)
    assert target == (CPU1, "research") and why.startswith(f"{N2}'s share wasn't read (exit 255: ssh: connect refused); ")
    with pytest.raises(auto.Unplaceable, match=f"no pool fits and {N2} takes no job.*research run --on MACHINE --mem-gb N"):
        auto.choose({N2: "exit 255: ssh: connect refused", CPU1: "no answer in 30 s"}, 4, 16)
    with pytest.raises(auto.Unplaceable, match="no slot of .* gives 64 CPUs and 300 GB"):
        auto.choose({N2: "exit 255: ssh: connect timed out", CPU1: pool(slot("r0"))}, 64, 300)
    target, why = auto.choose({N2: share(waiters=5), CPU1: {"research": {"error": "OSError: no slots"}}}, 4, 16)
    assert target is None and f"{CPU1}'s research pool skipped: OSError: no slots" in why


def test_the_probe_reads_live_claims_and_waiters_and_runs_the_trees_slot_py_on_the_pool(tmp_path, monkeypatch):
    """`read` as the machines run it: the payload is `python3 -` with cores.py and the tree's slot.py inlined; here both machines
    are this one, the claims a tmp dir whose files this test holds locked, and slot.py a stand-in that echoes its pool."""
    claims = tmp_path / "claims"
    claims.mkdir()
    monkeypatch.setattr(cores, "CLAIMS", claims)
    monkeypatch.setattr(cores, "CPU_ROOT", tmp_path / "no-topology")          # each CPU its own core
    monkeypatch.setattr(auto, "DISK_PATH", str(tmp_path / "no-disk"))         # no such directory: the probe skips the disk check
    monkeypatch.setattr(remote, "RUNS_VIA_SLOT_FILE", str(tmp_path / "no-via-slot"))   # a slot-only host (cpu-1) carries the real one
    monkeypatch.setattr(auto, "LEAN_SLOTS", str(tmp_path / "no-lean-slots"))
    held = [cores.locked(claims / "r1.cpus", "48-91\n"), cores.locked(claims / "wait.01.r2", "4\n")]
    (claims / "gone.cpus").write_text("92-95\n")                               # nobody holds it: a run that ended
    src = tmp_path / "src"
    (src / auto.DESCRIPTION).parent.mkdir(parents=True)
    (src / auto.DESCRIPTION).write_text(f'[[shares]]\nproject = "verity"\nnode = "{N2}"\ncpus = "48-95"\n')
    (src / auto.SLOT_PY).parent.mkdir(parents=True)
    (src / auto.SLOT_PY).write_text("import os\ndef status_doc(now):\n    return {'file': os.environ['CHECK_SLOTS'], 'slots': [], 'waiters': []}\n")
    try:
        docs = auto.read(src, 4, reach=lambda name: [sys.executable, "-"])
    finally:
        for fd in held:
            os.close(fd)
    assert docs[N2] == {"share": {"share": 48, "claims": 1, "waiters": 1, "free": 4, "fits": True}, "lean": None}
    assert docs[CPU1] == {"research": {"file": "/workspace/research/locks/research/slots", "slots": [], "waiters": []}, "lean": None}
    (tmp_path / "lean-slots").write_text("# NAME N\ncheck 2   # drive's\n# drive-only: build 10\naudit 0\n")
    monkeypatch.setattr(auto, "LEAN_SLOTS", str(tmp_path / "lean-slots"))
    assert auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[CPU1]["lean"] == {"check": 2, "audit": 0}, \
        "a commented-out pool is no pool"
    assert (claims / "gone.cpus").exists(), "the probe only reads: a leftover is the next taker's to remove"
    assert auto.read(src, 8, reach=lambda name: [sys.executable, "-"])[N2]["share"]["fits"] is True, "nothing holds them now"
    monkeypatch.setattr(auto, "DISK_PATH", str(tmp_path))
    monkeypatch.setattr(auto, "MIN_FREE_INODES", 2**62)
    short = auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[CPU1]["research"]
    assert short["error"].startswith(f"disk: {tmp_path} has ") and "inodes free" in short["error"], "27 Oct: cpu-5 out of inodes"
    target, why = auto.choose({N2: share(free=8), CPU1: {"research": short}}, 4, 16)
    assert target is None and "disk:" in why, "a pool on a full disk is skipped like one that can't be read"
    failing = auto.read(src, 4, reach=lambda name: [sys.executable, "-c", "import sys; sys.exit('no python here')"])
    assert failing == {m: "exit 1: no python here" for m in (N2, *(machine for machine, _ in auto.SPILL))}


def test_a_node_2_that_takes_runs_only_through_a_check_slot_is_never_chosen(tmp_path, monkeypatch):
    """A node 2 carrying /etc/vy/runs-via-slot refuses a run placed on its share, so its share reads as unread."""
    monkeypatch.setattr(cores, "CLAIMS", tmp_path / "claims")
    monkeypatch.setattr(cores, "CPU_ROOT", tmp_path / "no-topology")
    monkeypatch.setattr(auto, "DISK_PATH", str(tmp_path / "no-disk"))
    marker = tmp_path / "runs-via-slot"
    marker.write_text("slot only\n")
    monkeypatch.setattr(remote, "RUNS_VIA_SLOT_FILE", str(marker))
    src = tmp_path / "src"
    (src / auto.DESCRIPTION).parent.mkdir(parents=True)
    (src / auto.DESCRIPTION).write_text(f'[[shares]]\nproject = "verity"\nnode = "{N2}"\ncpus = "48-95"\n')
    (src / auto.SLOT_PY).parent.mkdir(parents=True)
    (src / auto.SLOT_PY).write_text("def status_doc(now):\n    return {'file': 'f', 'slots': [], 'waiters': []}\n")
    n2 = auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[N2]
    assert "runs-via-slot" in n2["share"]["error"]
    target, why = auto.choose({N2: n2, CPU1: pool(slot("r0", held=True), waiters=9)}, 4, 16)
    assert target == (CPU1, "research") and f"{N2}'s share wasn't read" in why
    with pytest.raises(auto.Unplaceable):
        auto.choose({N2: n2, CPU1: "no answer in 30 s"}, 4, 16)


def _auto_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    return repo


def _refusing(seen: list):
    def f(*args, **kw):
        seen.append(kw or args[-1])
        raise remote.Refused("reached")
    return f


def test_on_auto_spills_to_the_pool_as_a_pool_run_with_the_kinds_time(tmp_path, monkeypatch, capsys):
    launched, placed = [], []
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: share(waiters=56), CPU1: pool(slot("r0"))})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--", "pytest", "-q"]) == 3
    (kw,) = launched
    assert placed == [] and kw["machine_name"] == CPU1 and kw["project"] == "verity" and kw["mem_gb"] is None and kw["queue"] is None
    assert kw["command"][:2] == ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots"]
    assert kw["command"][2:6] == ["python3", "tools/verity/check/slot.py", "--by", "auto"]
    assert kw["command"][kw["command"].index("--run-max") + 1] == "3600" and kw["command"][-5:] == ["nice", "-n", "19", "pytest", "-q"]
    assert remote.flag_value(kw["harness_args"], "--cwd") == "source"
    err = capsys.readouterr().err
    assert f"research run --on auto: spilled to {CPU1}'s research pool, from " in err and "a slot is free there now" in err
    assert kw["env"]["CHECK_RUN_MEM_GB"] == "16", "the slot reserves the larger of its mem= and the sized run's"


def test_on_auto_spilled_to_a_pool_keeps_its_mem_gb_for_the_slot(tmp_path, monkeypatch, capsys):
    """A spill keeps --mem-gb, or a 40-GB research slot would hold the vllm suite to 2 workers: the slot gets the run's 150."""
    launched = []
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: share(waiters=56), CPU1: pool(slot("r8", mem=160))})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing([]))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--mem-gb", "150", "--", "pytest", "-q"]) == 3
    (kw,) = launched
    assert kw["mem_gb"] is None and kw["env"]["CHECK_RUN_MEM_GB"] == "150"
    assert "--mem-gb 150 goes to the slot" in capsys.readouterr().err


def test_on_auto_keeps_a_gpu_job_or_one_too_big_for_any_slot_on_node_2(tmp_path, monkeypatch, capsys):
    placed = []
    monkeypatch.setattr(auto, "read", lambda source, cpus, **kw: {N2: share(waiters=56), CPU1: pool(slot("r0"))})
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--gpus", "1", "--", "nvidia-smi"]) == 3
    assert "a GPU or quiet job, which only the queue books" in capsys.readouterr().err
    assert main([*runs, "--cpus", "16", "--mem-gb", "160", "--", "pytest"]) == 3
    assert "no slot of /workspace/research/locks/research/slots gives 16 CPUs and 160 GB" in capsys.readouterr().err
    assert [(s["node"], s["gpus"], s["cpus"]) for s in placed] == [(N2, 1, None), (N2, None, 16)]


def test_on_auto_refuses_a_job_too_big_for_any_slot_while_node_2_is_down(tmp_path, monkeypatch, capsys):
    launched, placed = [], []
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: DOWN, CPU1: pool(slot("r0"))})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--cpus", "64", "--mem-gb", "300", "--", "pytest"]) == 2
    assert launched == placed == []
    err = capsys.readouterr().err
    assert f"research run --on auto: refused: {N2} is down: its ssh doesn't answer (Connection timed out); " in err
    assert err.count(" is down") == 1 and f"no pool fits and {N2} takes no job" in err


def test_a_stopped_node_2_costs_its_probe_a_few_seconds_and_the_placement_says_once_that_it_is_down(tmp_path):
    """A stopped node 2 held its probe for ssh's default connect timeout, and the placement then quoted ssh's whole error line,
    address included, as a share that "wasn't read"."""
    at = {"host": "h", "port": 22, "user": "u"}
    assert f"ConnectTimeout={auto.NODE_CONNECT_S}" in auto._argv(N2, at) and auto.NODE_CONNECT_S <= 5
    assert "ConnectTimeout=15" in auto._argv(CPU1, at), "a pool's machine keeps remote.ssh_argv's default"
    ssh = "import sys; print({!r}, file=sys.stderr); sys.exit(255)"
    dead = [sys.executable, "-c", ssh.format("ssh: connect to host h port 22: Connection timed out")]
    assert auto.read(tmp_path, 4, reach=lambda name: dead)[N2] == DOWN
    assert auto.read(tmp_path, 4, reach=lambda name: dead, node_only=True) == {N2: DOWN}
    target, why = auto.choose({N2: DOWN, CPU1: pool(slot("r0"))}, 4, 16)
    assert target == (CPU1, "research") and why.startswith(f"{N2} is down: its ssh doesn't answer (Connection timed out); {CPU1}'s ")
    with pytest.raises(auto.Unplaceable) as refused:
        auto.choose({N2: DOWN, CPU1: pool(slot("r0"))}, 64, 300)
    assert str(refused.value).count(" is down") == 1 and "port 22" not in str(refused.value)
    denied = [sys.executable, "-c", ssh.format("u@h: Permission denied (publickey).")]
    assert auto.read(tmp_path, 4, reach=lambda name: denied)[N2] == "exit 255: u@h: Permission denied (publickey)."
    assert auto.off("exit 255: u@h: Permission denied (publickey).").startswith(f"{N2}'s share wasn't read (exit 255: ")
    assert auto.off(share(free=8)) is None


@pytest.fixture
def node_2(tmp_path, monkeypatch):
    """(HOLD_FILE's path, `read`): the probe as an idle node 2 runs it, with the clock `read` is given (a UTC time)."""
    monkeypatch.setattr(cores, "CLAIMS", tmp_path / "claims")
    monkeypatch.setattr(cores, "CPU_ROOT", tmp_path / "no-topology")
    monkeypatch.setattr(remote, "RUNS_VIA_SLOT_FILE", str(tmp_path / "no-via-slot"))
    hold = tmp_path / "auto-hold"
    monkeypatch.setattr(auto, "HOLD_FILE", str(hold))
    src = tmp_path / "src"
    (src / auto.DESCRIPTION).parent.mkdir(parents=True)
    (src / auto.DESCRIPTION).write_text(f'[[shares]]\nproject = "verity"\nnode = "{N2}"\ncpus = "48-95"\n')

    def read(now: str) -> dict:
        """Node 2's answer at `now`, its share's document alone."""
        at = datetime.fromisoformat(now).replace(tzinfo=UTC).timestamp()
        with monkeypatch.context() as m:
            m.setattr(auto, "SPILL", ())                                       # node 2 alone is asked
            return {"share": auto.read(src, 4, reach=lambda name: [sys.executable, "-"], now=at)[N2]["share"]}

    return hold, read


def test_a_held_node_2_takes_no_job_until_its_hold_ends_and_the_placement_says_until_when_and_why(tmp_path, node_2):
    """A node 2 that one lane has whole took `--on auto` jobs unless it also carried the slot-only marker, which says neither
    until when nor why."""
    hold, read = node_2
    assert read("2030-01-31T17:00")["share"]["free"] == 48, "no file, no hold"
    hold.write_text("2030-02-01T00:00Z one lane has the whole node\nany later line is a note\n")
    n2 = read("2030-01-31T17:00")
    assert n2 == {"share": {"hold": {"until": "2030-02-01T00:00Z", "why": "one lane has the whole node"}}}
    said = f"{N2} takes no `--on auto` job until 2030-02-01T00:00Z: one lane has the whole node"
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=9)
    assert auto.choose({N2: share(free=48), CPU1: full}, 4, 16)[0] is None, "unheld and idle, node 2 takes it"
    target, why = auto.choose({N2: n2, CPU1: full}, 4, 16)
    assert target == (CPU1, "research") and why.startswith(said + "; ") and why.endswith(f"{CPU1}'s research pool")
    with pytest.raises(auto.Unplaceable, match=f"no pool fits and {N2} takes no job") as refused:
        auto.choose({N2: n2, CPU1: pool(slot("r0"))}, 64, 300)
    assert str(refused.value).startswith(said + "; ")
    assert read("2030-01-31T23:59") == n2 and read("2030-02-01T00:00")["share"]["free"] == 48, "the hold ends on its own"
    hold.write_text("2030-02-01T02:00+02:00\n")
    assert read("2030-01-31T23:59") == {"share": {"hold": {"until": "2030-02-01T00:00Z", "why": ""}}}
    assert auto.off(read("2030-01-31T23:59")) == f"{N2} takes no `--on auto` job until 2030-02-01T00:00Z"
    for text, why in (("until infra says\n", "until infra says"), ("", "")):
        hold.write_text(text)
        n2 = read("2040-01-01T00:00")
        assert n2 == {"share": {"hold": {"until": None, "why": why}}}, "a first line that names no time holds until the file goes"
        assert auto.off(n2) == f"{N2} takes no `--on auto` job while it carries {hold}" + (f": {why}" if why else "")
    (tmp_path / "no-via-slot").write_text("")
    hold.write_text("2030-02-01T00:00Z one lane has the whole node\n")
    assert auto.off(read("2030-01-31T17:00")) == said, "the hold says more than the slot-only marker beside it"


def test_a_hold_file_that_is_there_but_cant_be_read_or_names_no_time_holds_node_2_and_says_why(tmp_path, monkeypatch, node_2):
    """The hold fails closed: a hold infra wrote must not open on a wrong mode, a typo or the node's zone. A probe that read
    an unreadable file as no hold passed every other test here."""
    hold, read = node_2
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=9)

    def held(why: str, at: str = "2040-01-01T00:00") -> str:
        """The hold as the launcher reads it, once node 2's answer at `at` is checked to be a hold with no end that says `why`."""
        n2 = read(at)
        assert n2 == {"share": {"hold": {"until": None, "why": why}}}
        target, said = auto.choose({N2: n2, CPU1: full}, 4, 16)
        assert target == (CPU1, "research"), "an idle node 2 that is held takes no job"
        with pytest.raises(auto.Unplaceable, match=f"no pool fits and {N2} takes no job"):
            auto.choose({N2: n2, CPU1: pool(slot("r0"))}, 64, 300)
        return said.split("; ")[0]

    carries = f"{N2} takes no `--on auto` job while it carries {hold}"
    hold.mkdir()
    assert held("it can't be read (IsADirectoryError)") == f"{carries}: it can't be read (IsADirectoryError)"
    hold.rmdir()
    hold.write_bytes(b"\xff\xfe2030-02-01T00:00Z\n")
    assert held("it can't be read (UnicodeDecodeError)") == f"{carries}: it can't be read (UnicodeDecodeError)"
    hold.unlink()
    hold.symlink_to(hold)                                           # stat fails on it, so os.path.exists calls it absent
    assert held("it can't be read (OSError)") == f"{carries}: it can't be read (OSError)"
    hold.unlink()
    hold.symlink_to(tmp_path / "gone")
    assert read("2030-01-31T17:00")["share"]["free"] == 48, "a link to no file is no hold file"
    hold.unlink()
    for line in ("2030-02-01 00:00Z one lane has the whole node", "2030-02-01", "00:00Z", "tomorrow"):
        hold.write_text(line + "\n")
        assert held(line) == f"{carries}: {line}", "a date alone, or one apart from its time, is no time: it holds, never ends early"
    monkeypatch.setenv("TZ", "XXX-9")                               # the probe's zone, nine hours east: UNTIL is UTC all the same
    for line in ("2030-02-01T00:00", "2030-02-01T00:00Z", "2030-02-01T02:00+02:00", "2030-01-31T19:00-05:00"):
        hold.write_text(line + " why\n")
        assert read("2030-01-31T23:59") == {"share": {"hold": {"until": "2030-02-01T00:00Z", "why": "why"}}}, line
        assert read("2030-02-01T00:00")["share"]["free"] == 48, f"{line}: it ends at that instant, in no other zone's"


def test_on_auto_tells_the_launcher_until_when_and_why_node_2_is_held(tmp_path, monkeypatch, capsys):
    launched, placed = [], []
    held = {"share": {"hold": {"until": "2030-02-01T00:00Z", "why": "one lane has the whole node"}}}
    said = f"{N2} takes no `--on auto` job until 2030-02-01T00:00Z: one lane has the whole node"
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: held, CPU1: pool(slot("r0", held=True), waiters=9)})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--", "pytest", "-q"]) == 3
    assert [kw["machine_name"] for kw in launched] == [CPU1] and placed == []
    assert f"research run --on auto: spilled to {CPU1}'s research pool, from " in (err := capsys.readouterr().err) and said in err
    assert main([*runs, "--cpus", "64", "--mem-gb", "300", "--", "pytest"]) == 2
    assert len(launched) == 1 and placed == []
    assert f"research run --on auto: refused: {said}; " in capsys.readouterr().err


def test_on_auto_refuses_a_gpu_job_while_node_2_is_down_or_held_and_asks_node_2_alone(tmp_path, monkeypatch, capsys):
    """A GPU or quiet job never reached `choose`: on a stopped node 2 it waited on the node's deadline and was refused on a stale
    admission window, and a hold would not have kept it off the node."""
    asked, placed = [], []
    held = {"share": {"hold": {"until": "2030-02-01T00:00Z", "why": "one lane has the whole node"}}}
    unread = {"share": {"error": "RuntimeError: /etc/vy/runs-via-slot: its launcher takes a run only through a check slot"}}
    answers = iter([DOWN, held, unread, share(free=48)])
    monkeypatch.setattr(auto, "read", lambda source, cpus, **kw: asked.append(kw) or {N2: next(answers)})
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    gpu = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto", "--gpus", "1", "--", "nvidia-smi"]
    nowhere = ", and a GPU or quiet job runs on no other node"
    assert main(gpu) == 2 and placed == []
    assert f"research run --on auto: refused: {N2} is down: its ssh doesn't answer (Connection timed out){nowhere}" in capsys.readouterr().err
    assert main(gpu) == 2 and placed == []
    assert f"refused: {N2} takes no `--on auto` job until 2030-02-01T00:00Z: one lane has the whole node{nowhere}" in capsys.readouterr().err
    assert main(gpu) == 3 and main(gpu) == 3, "a node 2 unread for another reason, or read: the queue decides"
    assert [s["node"] for s in placed] == [N2, N2] and asked == [{"node_only": True}] * 4


@pytest.mark.parametrize("text, n", [("8-19", 12), ("0", 1), ("48-51,56-63", 12)])
def test_cpulist_counts(text, n):
    assert auto.cpulist(text) == n


def test_the_share_comes_from_the_trees_description(tmp_path):
    assert auto.share_cpus(tmp_path) is None
    (tmp_path / auto.DESCRIPTION).parent.mkdir(parents=True)
    (tmp_path / auto.DESCRIPTION).write_text(json.dumps("not toml") + "\n")
    assert auto.share_cpus(tmp_path) is None
    (tmp_path / auto.DESCRIPTION).write_text(f'[[shares]]\nproject = "pous"\nnode = "{N2}"\ncpus = "0-7"\n'
                                             f'[[shares]]\nproject = "verity"\nnode = "{N2}"\ncpus = "48-95"\n')
    assert auto.share_cpus(tmp_path) == "48-95"
