"""`research run --on auto`'s placement (research.auto): a CPU node's pool first, node 2's share only for a job no pool fits. 19 Oct:
two runs held all 48 of the share's cores for hours while 56 waited, a 4-CPU suite 36th in line, and cpu-1's cores sat idle."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research import auto, cores, remote
from research.cli import main

N2, CPU1, CPU5 = auto.AUTO_NODE, "vy-nebius-cpu-1", "vy-nebius-cpu-5"


def share(free: int = 0, waiters: int = 0, size: int = 48, fits: bool | None = None) -> dict:
    return {"share": {"share": size, "claims": 2, "waiters": waiters, "free": free, "fits": free >= 4 if fits is None else fits}}


def slot(name: str, cpus: str = "8-19", mem: int = 40, held: bool = False, blocked: str | None = None, trains: bool = False) -> dict:
    return {"name": name, "cpus": cpus, "mem": mem, "trains": trains, "holder": {"pid": 1} if held else None, "blocked": blocked}


def pool(*slots: dict, waiters: int = 0) -> dict:
    return {"research": {"file": "/workspace/research/locks/research/slots", "slots": list(slots), "waiters": [{"pid": i} for i in range(waiters)]}}


def test_a_pool_slot_free_now_wins_even_when_node_2_could_start_the_job():
    """24 Oct 02:10Z: node 2's GPUs sat at 0% under CPU work while cpu-5 ran at 8 of 192 (top): CPU jobs go to the pools."""
    target, why = auto.choose({N2: share(free=8), CPU1: pool(slot("r0"))}, 4, 16)
    assert target == (CPU1, "research") and why.endswith("a slot is free there now")
    target, why = auto.choose({N2: share(free=0, waiters=56), CPU1: pool(slot("r0", held=True), slot("r1"))}, 4, 16)
    assert target == (CPU1, "research")
    assert why.startswith(f"{N2}'s share: 0 of 48 CPUs free, 2 claim(s), 56 waiting; {CPU1}'s research pool: slots r0 held, r1 free")
    target, _ = auto.choose({N2: share(free=8), CPU1: pool(slot("r0", held=True)), CPU5: pool(slot("r0"))}, 4, 16)
    assert target == (CPU5, "research"), "cpu-5's pool when cpu-1's is busy"


def test_every_cpu_node_with_an_any_lane_research_pool_is_a_spill_target():
    """cpu-6 to cpu-8's research pools sat idle while cpu-1's had a line, because auto never tried them. cpu-9's stay out:
    its research slots share every core with its ci pool."""
    assert [m for m, p in auto.SPILL if p == "research"] == [
        "vy-nebius-cpu-1", "vy-nebius-cpu-5", "vy-nebius-cpu-3", "vy-nebius-cpu-6", "vy-nebius-cpu-7", "vy-nebius-cpu-8"]
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=12)
    idle = "vy-nebius-cpu-8"
    target, why = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, "vy-nebius-cpu-3": full,
                               idle: pool(slot("r0", cpus="0-15", mem=96))}, 4, 16)
    assert target == (idle, "research") and why.endswith("a slot is free there now")


def test_cpu_3s_research_pool_takes_a_job_while_cpu_1s_and_cpu_5s_are_full():
    """INF-281, 27 Oct 03:06Z: cpu-1 had 22 waiting and cpu-5 34 while cpu-3's research pool sat idle, outside SPILL."""
    cpu3 = "vy-nebius-cpu-3"
    assert (cpu3, "research") in auto.SPILL
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=22)
    target, why = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, cpu3: pool(slot("b4", cpus="0-15", mem=48))}, 4, 16)
    assert target == (cpu3, "research") and why.endswith("a slot is free there now")


def test_with_nothing_free_the_pool_with_the_shortest_line_per_fitting_slot_wins():
    full = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=3)                      # (3 + 1) / 4 = 1
    assert auto.choose({N2: share(waiters=56), CPU1: full}, 4, 16)[0] == (CPU1, "research")
    assert auto.choose({N2: share(free=8), CPU1: full}, 4, 16)[0] == (CPU1, "research"), "node 2 free still waits in a pool"
    busier = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=7)                    # (7 + 1) / 4 = 2
    target, why = auto.choose({N2: share(), CPU1: busier, CPU5: full}, 4, 16)
    assert target == (CPU5, "research") and why.endswith(f"the shortest line per run that fits is {CPU5}'s research pool")


def test_a_slot_too_small_trains_only_or_blocked_never_takes_the_job():
    docs = {N2: share(waiters=56)}
    for p, cpus, mem in ((pool(slot("r0", cpus="8-11")), 8, 16), (pool(slot("r0", mem=40)), 4, 64),
                         (pool(slot("b", cpus="54-191", mem=560, trains=True)), 4, 16)):
        target, why = auto.choose({**docs, CPU1: p}, cpus, mem)
        assert target is None and f"{CPU1}'s research pool skipped: no slot of" in why
    target, why = auto.choose({**docs, CPU1: pool(slot("r0", blocked="its window opens"))}, 4, 16)
    assert target == (CPU1, "research") and "r0 blocked" in why and "shortest line" in why, "a blocked slot is no free one"


def test_a_machine_that_cant_be_read_is_skipped_and_with_neither_a_pool_nor_node_2_the_job_is_refused():
    """26 Oct: node 2 stopped, and a job too big for a pool slot went to its queue, which refused it on a stale admission
    window."""
    target, why = auto.choose({N2: "exit 255: ssh: connect refused", CPU1: pool(slot("r0"))}, 4, 16)
    assert target == (CPU1, "research") and why.startswith(f"{N2}'s share wasn't read (exit 255: ssh: connect refused); ")
    with pytest.raises(auto.Unplaceable, match=f"no pool fits and {N2} can't be reached.*research run --on MACHINE --mem-gb N"):
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
    assert docs[N2] == {"share": {"share": 48, "claims": 1, "waiters": 1, "free": 4, "fits": True}}
    assert docs[CPU1] == {"research": {"file": "/workspace/research/locks/research/slots", "slots": [], "waiters": []}}
    assert (claims / "gone.cpus").exists(), "the probe only reads: a leftover is the next taker's to remove"
    assert auto.read(src, 8, reach=lambda name: [sys.executable, "-"])[N2]["share"]["fits"] is True, "nothing holds them now"
    failing = auto.read(src, 4, reach=lambda name: [sys.executable, "-c", "import sys; sys.exit('no python here')"])
    assert failing == {m: "exit 1: no python here" for m in (N2, *(machine for machine, _ in auto.SPILL))}


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
    """INF-228: a spill dropped --mem-gb, so a 40-GB research slot held the vllm suite to 2 workers; the slot gets the run's 150."""
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
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: share(waiters=56), CPU1: pool(slot("r0"))})
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
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: "exit 255: ssh: connect timed out", CPU1: pool(slot("r0"))})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--cpus", "64", "--mem-gb", "300", "--", "pytest"]) == 2
    assert launched == placed == []
    assert f"research run --on auto: refused: {N2}'s share wasn't read" in capsys.readouterr().err


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
