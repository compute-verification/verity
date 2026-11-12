"""`research run --on auto`'s placement (research.auto): a free CPU pool slot first, then node 2's share if it starts the job now, so
small runs never wait behind two that hold all 48 of the share's cores for hours while cpu-1's cores sit idle."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research import auto, cores, remote
from research.cli import main

REPO = Path(__file__).resolve().parents[3]
N2, CPU1, CPU5 = auto.AUTO_NODE, "vy-nebius-cpu-1", "vy-nebius-cpu-5"
#: node 2's answer as `auto.read` gives it while the node is stopped
DOWN = f"{auto.NO_SSH} (Connection timed out)"
#: the CPU nodes `auto.read` asks in these tests, and `auto.spill` as the module defines it
NODES, SPILL = (CPU1, "vy-nebius-cpu-3", CPU5), auto.spill


@pytest.fixture(autouse=True)
def cpu_nodes(monkeypatch):
    """`auto.read` asks NODES, not the CPU nodes in the registry of the machine these tests run on."""
    monkeypatch.setattr(auto, "spill", lambda machines=None: tuple((m, auto.SPILL_POOL) for m in NODES))


def share(free: int = 0, waiters: int = 0, size: int = 48, fits: bool | None = None) -> dict:
    return {"share": {"share": size, "claims": 2, "waiters": waiters, "free": free, "fits": free >= 4 if fits is None else fits}}


def slot(name: str, cpus: str = "8-19", mem: int = 40, held: bool = False, blocked: str | None = None, trains: bool = False) -> dict:
    return {"name": name, "cpus": cpus, "mem": mem, "trains": trains, "holder": {"pid": 1} if held else None, "blocked": blocked}


def pool(*slots: dict, waiters: int = 0, minutes: int = 0, free_gb: int | None = None, asks: tuple[float, ...] = ()) -> dict:
    """A pool's document as the probe gives it; ``free_gb``: the memory slot.py finds free on its machine, in GiB. Its line is
    ``waiters`` whose tickets name no ask, then one for each of ``asks``, asking that many GB."""
    return {"research": {"file": "/workspace/research/locks/research/slots", "slots": list(slots),
                         "waiters": [{"pid": i, "minutes": minutes} for i in range(waiters)]
                         + [{"pid": waiters + i, "minutes": minutes, "mem_gb": gb} for i, gb in enumerate(asks)],
                         **({} if free_gb is None else {"free_gb": free_gb})}}


def lean(**pools: tuple[int, int, int]) -> dict:
    """A machine's Lean slots as lean_slot.py's status document gives them: each pool as (slots, held, waiting)."""
    return {"lean": {"file": auto.LEAN_SLOTS, "pools": {
        name: {"n": n, "slots": [{"index": i, "past_n": False, "holder": {"pid": 1} if i < held else None} for i in range(n)],
               "waiters": [{"pid": i, "minutes": 1} for i in range(waiting)]} for name, (n, held, waiting) in pools.items()}}}


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
    """A node's idle pool takes the job while the others queue, whichever node it is."""
    full = pool(*(slot(f"r{i}", held=True) for i in range(8)), waiters=22)
    for idle in ("vy-nebius-cpu-7", "vy-nebius-cpu-8", "vy-nebius-cpu-9", "vy-nebius-cpu-11"):
        target, why = auto.choose({N2: share(waiters=5), CPU1: full, CPU5: full, idle: pool(slot("r0", cpus="12-33"))}, 4, 16)
        assert target == (idle, "research") and why.endswith("a slot is free there now")


def test_on_auto_asks_every_cpu_node_in_the_registry_and_skips_one_whose_pool_lists_no_slot(tmp_path):
    """The machines `--on auto` may use are the CPU nodes of the launcher's registry, by number, with no list to keep: a node
    left out of a list sat with free slots while the listed ones queued. One whose research pool has no slots file, or lists
    no slot, is asked and skipped, at no wait."""
    machines = {N2: {"project": "verity"}, "vy-nebius-cpu-10": {"project": "verity"}, CPU1: {"projects": ["verity", "pous"]},
                "vy-nebius-cpu-4": {"project": "verity"}, "vy-nebius-cpu-12": {"project": "other"},
                "vy-auditor-1": {"project": "verity"}, "vy-nebius-cpu-4-old": {"project": "verity"}}
    assert SPILL(machines) == ((CPU1, "research"), ("vy-nebius-cpu-4", "research"), ("vy-nebius-cpu-10", "research")), \
        "the project's CPU nodes by number; never an auditor's machine, another project's node or a name that only starts so"
    assert SPILL({}) == ()
    asked = []

    def run(argv, payload):
        asked.append(argv[0])
        return subprocess.CompletedProcess(argv, 1, b"", b"no python here")

    (tmp_path / auto.SLOT_PY).parent.mkdir(parents=True)
    (tmp_path / auto.SLOT_PY).write_text("")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(auto, "spill", lambda: SPILL(machines))
        docs = auto.read(tmp_path, 4, reach=lambda name: [name], run=run)
    assert list(docs) == [N2, CPU1, "vy-nebius-cpu-4", "vy-nebius-cpu-10"] and sorted(asked) == sorted(docs)
    empty = {"file": "/workspace/research/locks/research/slots", "slots": [], "waiters": [], "free_gb": 400}
    docs = {N2: DOWN, CPU1: {"research": empty}, "vy-nebius-cpu-4": {"research": {**empty, "slots": None}},
            "vy-nebius-cpu-10": pool(slot("r0", held=True), free_gb=400), "vy-nebius-cpu-12": pool(slot("r0"), free_gb=400)}
    target, why = auto.choose(docs, 4, 16)
    assert target == ("vy-nebius-cpu-12", "research"), "a node in no list of this module's takes the job"
    assert why.count("/workspace/research/locks/research/slots lists no slot") == 2
    assert auto.expect(docs, (CPU1, "research"), 4, 16) == auto.Wait(None, "/workspace/research/locks/research/slots lists no slot", False)
    del docs["vy-nebius-cpu-12"]
    assert auto.choose(docs, 4, 16)[0] == ("vy-nebius-cpu-10", "research"), "the pools with no slot rank nowhere: the one line there is"


def test_a_run_that_takes_a_lean_build_slot_goes_only_where_the_machines_lean_slots_have_build():
    """`lean_changed.py --build` runs went by `--on auto` to a machine whose `build` pool was the queue's, and were refused in
    0 s, again and again. A machine whose Lean slots give `build` none is skipped for them; one with no Lean slots file, or
    whose file couldn't be read, isn't."""
    cpu9 = "vy-nebius-cpu-9"
    drives = {**pool(slot("r0", cpus="12-33")), **lean(check=(2, 0, 0), build=(0, 0, 0))}
    lanes = {**pool(slot("r0", held=True), waiters=3), **lean(build=(8, 0, 0), audit=(2, 0, 0))}
    docs = {N2: share(waiters=5), cpu9: drives, CPU5: lanes}
    assert auto.choose(docs, 4, 16)[0] == (cpu9, "research"), "any other run takes cpu-9's free slot"
    target, why = auto.choose(docs, 4, 16, lean_pool="build")
    assert target == (CPU5, "research") and f"{cpu9}'s research pool skipped: its Lean slots give `build` none" in why
    for unread in ({"file": auto.LEAN_SLOTS, "pools": None}, {"error": "OSError: denied"}, None):
        assert auto.choose({N2: share(waiters=5), cpu9: {**drives, "lean": unread}}, 4, 16, lean_pool="build")[0] == (cpu9, "research")


def test_the_lean_pool_a_command_takes_a_slot_of():
    """A records run, the commonest Lean run, went unrecognised, so nothing placed it by its Lean slot."""
    changed = ["python3", "tools/verity/lean/lean_changed.py"]
    assert auto.lean_pool([*changed, "--no-wait", "--build", "X.lean"]) == "build"
    assert auto.lean_pool(["uv", "run", "verity", "lean", "try", "X.lean"]) == "build"
    assert auto.lean_pool([*changed, "--records", "security", "--update"]) == "build"
    assert auto.lean_pool([*changed, "--records=Security"]) == "build"
    assert auto.lean_pool(["bash", "-lc", "uv sync -q && python3 tools/verity/lean/lean_changed.py --records Security"]) == "build"
    assert auto.lean_pool([*changed, "--records", "security"], ["A=1", "LEAN_RECORDS_POOL=proofs"]) == "proofs", "the run's --env"
    assert auto.lean_pool(["env", "LEAN_RECORDS_POOL=proofs", *changed, "--try", "X.lean"]) == "proofs"
    assert auto.lean_pool(["bash", "-c", "LEAN_RECORDS_POOL=proofs python3 tools/verity/lean/lean_changed.py --records P"]) == "proofs"
    assert auto.lean_pool([*changed, "--records", "security", "--dev-cache"]) == "audit", "check.py --dev-cache --build takes it"
    audit = ["python3", "tools/verity/lean/check.py", "--build", "--update", "security"]
    assert auto.lean_pool(audit) == "audit" and auto.lean_pool(audit, ["LEAN_SLOT_POOL=check"]) == "check"
    assert auto.lean_pool(["python3", "tools/verity/lean/check.py", "security"]) is None, "without --build it takes no slot"
    assert auto.lean_pool(["python3", "tools/verity/check/check.py", "--build"]) is None, "the whole check is not the Lean check"
    assert auto.lean_pool([*changed, "--inputs-key", "security"]) is None
    assert auto.lean_pool(["uv", "run", "pytest", "tools/verity/lean/tests/test_lean_changed.py", "-k", "records"]) is None
    assert auto.lean_pool(["bash", "-c", "echo 'an unclosed quote"]) is None


def test_a_free_lean_slot_wins_over_a_shorter_line_for_a_cpu_slot_where_the_lean_pool_is_full():
    """A records run placed by the CPU line alone took a check slot and then waited inside it for a Lean slot, the slot's cores
    and memory idle, while another machine's Lean pool had slots free."""
    short = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=1)                     # (1 + 1) / 4
    long = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=7)                      # (7 + 1) / 4
    docs = {N2: share(waiters=5), CPU1: {**short, **lean(build=(8, 8, 0))}, CPU5: {**long, **lean(build=(3, 1, 0), audit=(2, 2, 0))}}
    target, why = auto.choose(docs, 4, 16, lean_pool="build")
    assert target == (CPU5, "research")
    assert f"{CPU1}'s research pool: slots r0 held, r1 held, r2 held, r3 held; 1 waiting; Lean `build`: 0 of 8 free, 0 waiting; " in why
    assert "; 7 waiting; Lean `build`: 2 of 3 free, 0 waiting: " in why, "each machine's Lean pool is shown beside its line"
    assert why.endswith(f": {CPU5}'s research pool, the shortest line per run that fits where a slot of Lean pool `build` is free")
    target, why = auto.choose(docs, 4, 16, lean_pool="audit")
    assert target == (CPU5, "research") and f"{CPU1}'s research pool skipped: its Lean slots give `audit` none; " in why
    assert why.endswith(f": no machine has a slot of Lean pool `audit` free, so the shortest line for one, per slot: {CPU5}'s research pool")
    idle = {**pool(slot("r0")), **lean(build=(8, 8, 0))}
    target, why = auto.choose({**docs, CPU1: idle}, 4, 16, lean_pool="build")
    assert target == (CPU5, "research"), "a check slot free now, where the run would hold it idle in the Lean line, is no start"
    for full in ((8, 8, 0), (8, 7, 1)):                                    # the free slot is the waiter's, within a poll
        assert auto.choose({**docs, CPU5: {**long, **lean(build=full)}}, 4, 16, lean_pool="build")[0] == (CPU1, "research")
    both = {**pool(slot("r0", held=True), slot("r1")), **lean(build=(3, 2, 0))}
    target, why = auto.choose({**docs, "vy-nebius-cpu-3": both}, 4, 16, lean_pool="build")
    assert target == ("vy-nebius-cpu-3", "research") and why.endswith(": a slot is free there now, and one of Lean pool `build` with it")
    node = {N2: {**share(free=8), **lean(build=(2, 0, 0))}}
    target, why = auto.choose({**docs, **node}, 4, 16, lean_pool="build")
    assert target is None and why.endswith(f": no pool has a slot and one of Lean pool `build` free, and it starts on {N2} now")
    for theirs in (lean(build=(2, 2, 0)), lean(audit=(2, 0, 0)), {"lean": {"error": "Refused: not `POOL N`"}}, {}):
        held = {N2: {**share(free=8), **theirs}}
        assert auto.choose({**docs, **held}, 4, 16, lean_pool="build")[0] == (CPU5, "research"), "node 2 starts it only with a Lean slot"


def test_a_run_that_takes_no_lean_slot_is_placed_as_before_whatever_the_lean_slots():
    full, some = lean(build=(8, 8, 3), audit=(2, 2, 2)), lean(build=(3, 1, 0), audit=(2, 0, 0))
    for cpu1, cpu5 in ((pool(slot("r0", held=True), waiters=1), pool(slot("r0", held=True), waiters=7)),      # the shortest line
                       (pool(slot("r0")), pool(slot("r0"))),                                                    # the first free slot
                       (pool(slot("r0", held=True), waiters=3), pool(slot("r0", held=True), slot("r1")))):
        for n2 in (share(waiters=5), share(free=8)):
            plain = auto.choose({N2: n2, CPU1: cpu1, CPU5: cpu5}, 4, 16)
            assert auto.choose({N2: {**n2, **full}, CPU1: {**cpu1, **full}, CPU5: {**cpu5, **some}}, 4, 16) == plain
            assert "Lean" not in plain[1]
            free = {N2: {**n2, **some}, CPU1: {**cpu1, **some}, CPU5: {**cpu5, **some}}
            assert auto.choose(free, 4, 16, lean_pool="build")[0] == plain[0], "and a Lean run where every machine has a Lean slot free"


def test_with_no_lean_slot_free_anywhere_the_shortest_line_for_one_per_slot_wins():
    free = pool(slot("r0"))
    busy = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=1)
    docs = {N2: share(waiters=5), CPU1: {**busy, **lean(build=(8, 8, 3))},      # (3 + 1) / 8 of a Lean line, behind a check line
            CPU5: {**free, **lean(build=(2, 2, 1))}}                              # (1 + 1) / 2, and a check slot free now
    assert auto.choose(docs, 4, 16)[0] == (CPU5, "research")
    target, why = auto.choose(docs, 4, 16, lean_pool="build")
    assert target == (CPU1, "research")
    assert why.endswith(f": no machine has a slot of Lean pool `build` free, so the shortest line for one, per slot: {CPU1}'s research pool")
    even = {**docs, CPU5: {**free, **lean(build=(4, 4, 1))}}
    assert auto.choose(even, 4, 16, lean_pool="build")[0] == (CPU5, "research"), "the same Lean line: the shorter line for a check slot"
    assert auto.choose({**even, CPU5: {**busy, **lean(build=(4, 4, 1))}}, 4, 16, lean_pool="build")[0] == (CPU1, "research"), "then the order they were asked in"


def test_a_machine_whose_lean_slots_cant_be_read_is_not_preferred():
    free = pool(slot("r0"))
    busy = pool(*(slot(f"r{i}", held=True) for i in range(4)), waiters=7)
    for unread, said in (({"lean": {"error": "Refused: /w/lean-slots: 'build two' is not `POOL N`"}}, "Refused: /w/lean-slots: 'build two' is not `POOL N`"),
                         ({}, "no answer")):
        docs = {N2: share(waiters=5), CPU1: {**free, **unread}, CPU5: {**busy, **lean(build=(2, 2, 3))}}
        target, why = auto.choose(docs, 4, 16, lean_pool="build")
        assert target == (CPU5, "research"), "a line for a Lean slot that was read beats a free check slot beside Lean slots that weren't"
        assert f"{CPU1}'s research pool: slots r0 free; 0 waiting; its Lean slots weren't read ({said}); " in why
        target, why = auto.choose({**docs, CPU5: {**busy, **unread}}, 4, 16, lean_pool="build")
        assert target == (CPU1, "research"), "it is still a machine: with none read, the run goes where any run would"
        assert why.endswith(f": no machine's Lean slots were read, so the shortest line per run that fits, {CPU1}'s research pool")
        assert auto.choose({**docs, CPU5: "no answer in 30 s"}, 4, 16, lean_pool="build")[0] == (CPU1, "research")
    uncapped = {"lean": {"file": auto.LEAN_SLOTS, "pools": None}}
    target, why = auto.choose({N2: share(waiters=5), CPU1: {**free, **uncapped}}, 4, 16, lean_pool="build")
    assert target == (CPU1, "research") and (f"; no Lean slots to wait for there: {CPU1}'s research pool has the most places for it "
                                             f"(1): a slot is free there now, and one of Lean pool") in why


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
    stuck = pool(*(slot(f"r{i}", mem=None) for i in range(5)), asks=(256,), minutes=auto.STALL_MIN, free_gb=100)
    target, why = auto.choose({N2: share(waiters=56), CPU1: stuck, CPU5: busy}, 4, 64)
    assert target == (CPU5, "research") and (f"{CPU1}'s research pool skipped: its line is stalled: a run has waited 5 min while "
                                             f"slot r0 is free, for more memory than the 100 GB free there") in why
    target, why = auto.choose({N2: "no answer in 30 s", CPU1: stuck, CPU5: "no answer in 30 s"}, 4, 64)
    assert target == (CPU1, "research") and "every pool that fits is stalled" in why, "a stalled line beats refusing the job"
    longer = pool(*(slot(f"r{i}", mem=None) for i in range(5)), asks=(256,) * 4, minutes=auto.STALL_MIN, free_gb=100)
    assert auto.choose({N2: "no answer in 30 s", CPU1: longer, CPU5: stuck}, 4, 64)[0] == (CPU5, "research"), "the shorter"
    assert auto.choose({N2: share(free=48), CPU1: stuck}, 4, 64)[0] is None, "node 2 starting it now beats a stalled line"
    starved = pool(*(slot(f"r{i}", blocked=mem) for i in range(13)))
    target, why = auto.choose({N2: share(waiters=56), CPU1: starved, CPU5: busy}, 4, 16)
    assert target == (CPU5, "research") and "every slot of it that fits waits on memory and none runs" in why
    fresh = pool(*(slot(f"r{i}", mem=None) for i in range(5)), asks=(256,), minutes=auto.STALL_MIN - 1, free_gb=100)
    assert auto.choose({N2: share(waiters=56), CPU1: fresh, CPU5: busy}, 4, 64)[0] == (CPU5, "research"), \
        "not yet stalled: a line like any other, and here the longer one per slot that runs"
    assert auto.choose({N2: share(waiters=56), CPU1: stuck, CPU5: busy}, 4, 16)[0] == (CPU1, "research"), \
        "a job small enough to pass the head (slot.py's backfill) has a place there"
    covered = pool(*(slot(f"r{i}", mem=None) for i in range(5)), asks=(64,), minutes=60, free_gb=200)
    assert auto.choose({N2: share(waiters=56), CPU1: covered, CPU5: busy}, 4, 64)[0] == (CPU1, "research"), \
        "a waiter the free memory covers is one between polls, however long it has waited: it takes its slot, and the job the next"
    mixed = pool(slot("r0", held=True), *(slot(f"r{i}", blocked=mem) for i in range(1, 4)), waiters=2, minutes=60)
    assert auto.choose({N2: share(waiters=56), CPU1: mixed}, 4, 16)[0] == (CPU1, "research"), "a running slot drains it"


def test_cpu_3s_research_pool_takes_a_job_while_cpu_1s_and_cpu_5s_are_full():
    """cpu-3's research pool doesn't sit idle while cpu-1's and cpu-5's lines are long."""
    cpu3 = "vy-nebius-cpu-3"
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
    for p, cpus, mem in ((pool(slot("r0", cpus="8-11")), 8, 16), (pool(slot("b", cpus="54-191", mem=560, trains=True)), 4, 16)):
        target, why = auto.choose({**docs, CPU1: p}, cpus, mem)
        assert target is None and f"{CPU1}'s research pool skipped: no slot of" in why
    target, why = auto.choose({**docs, CPU1: pool(slot("r0", blocked="its window opens"))}, 4, 16)
    assert target == (CPU1, "research") and "r0 blocked" in why and "shortest line" in why, "a blocked slot is no free one"


def test_a_slot_fits_a_job_as_slot_py_admits_it_whatever_its_own_mem():
    """slot.py gives a job any slot of its pool and raises the slot's reservation to the job's ask, so a slot's mem= rules no
    job out: the machine's free memory decides whether the job starts there now. Counting only slots whose mem= covered the
    ask, a job asking over 40 GB saw one slot of a pool of 21 and its wait there twenty times too long."""
    small = [slot(f"r{i}", held=i < 10) for i in range(20)]                 # 12 CPUs each, mem=40; ten of them held
    def docs(free_gb: int | None, waiters: int = 6) -> dict:
        return {N2: DOWN, CPU1: pool(slot("r8", cpus="0-15", mem=160, held=True), *small, waiters=waiters, free_gb=free_gb)}

    at = (CPU1, "research")
    assert auto._pool(docs(1000)[CPU1]["research"], 12, 128)[:3] == (17, 4, 0), \
        "the six waiters take a slot each, and any of the four left takes 128 GB now: 17 run, and nobody is ahead"
    assert auto.expect(docs(1000), at, 12, 128) == auto.Wait(0, "", False)
    fit, free, waiting, why = auto._pool(docs(100, waiters=0)[CPU1]["research"], 12, 128)
    assert (fit, free, waiting) == (11, 0, 0) and "r10 blocked" in why, "128 GB is 120 GiB: over the 100 free, the unheld wait"
    assert auto.expect(docs(100, waiters=0), at, 12, 128).minutes == pytest.approx(auto.HOLD_MIN / 11), "the eleven that run drain its line"
    assert auto._pool(docs(400, waiters=0)[CPU1]["research"], 12, 16)[:2] == (11, 10), "a 16 GB job reserves the slot's own 40 there, which is free"
    assert auto._pool(docs(30)[CPU1]["research"], 12, 16)[:2] == (11, 0), "under a slot's own mem= free, nothing takes it"
    assert auto._pool(docs(None)[CPU1]["research"], 12, 128)[:2] == (17, 4), "a document without the figure: the slot's own word"
    idle = pool(*(slot(f"b{i}", mem=None) for i in range(4)), free_gb=13)   # slots without mem=: nothing in them shows memory
    with pytest.raises(auto.Stalled, match="13 GB of memory free, under the 68 GB this job would reserve there"):
        auto._pool(idle["research"], 4, 72)
    target, why = auto.choose({N2: DOWN, CPU1: idle, CPU5: pool(slot("r0", held=True), free_gb=300)}, 4, 72)
    assert target == (CPU5, "research") and f"{CPU1}'s research pool skipped: its line is stalled" in why, \
        "a job is not sent to wait on memory beside idle slots while another pool runs"
    assert auto._pool(idle["research"], 4, 0)[1] == 4, "a job that asks no memory reserves none in a slot without mem="
    assert auto._pool(idle["research"], 4, 12)[1] == 1, "12 GB is 12 GiB rounded up: inside the 13 free, once"
    booked = pool({**slot("r0"), "book": 15}, slot("r1"), free_gb=20)        # r0: `mem=40 book=15`, as slot.py --status gives it
    fit, free, _waiting, why = auto._pool(booked["research"], 12, 16)
    assert (fit, free) == (1, 1) and "r1 blocked" in why, "r0 reserves its book=, 15, inside the 20 free; r1 its mem=, 40"
    with pytest.raises(auto.Stalled, match="20 GB of memory free, under the 30 GB this job would reserve there"):
        auto._pool(pool({**slot("r0"), "book": 15}, free_gb=20)["research"], 12, 32)   # an ask over book= is what it reserves


def test_idle_slots_are_places_only_as_far_as_the_free_memory_goes_once_the_line_ahead_has_taken_its_own():
    """A pool of 21 slots at 40 GB each, on a machine with memory for fewer, kept idle slots while its line waited on memory.
    Each idle slot was checked alone against the same free memory and the waiters' own asks weren't read, so the idle slots
    outnumbered the waiters, the pool read as free, and every job was sent there to wait beside pools that had a place."""
    def cpu1(free_gb: int, **line) -> dict:
        return pool(*(slot(f"r{i}", held=i < 8) for i in range(21)), free_gb=free_gb, **line)       # 13 idle, mem=40

    one = pool(slot("r0", held=True), slot("r1", mem=None), free_gb=300)
    docs = {N2: DOWN, CPU1: cpu1(70, asks=(64, 64, 16, 16), minutes=3), CPU5: one}
    target, why = auto.choose(docs, 4, 16)
    assert target == (CPU5, "research") and why.endswith(f"{CPU5}'s research pool has the most places for it (1): a slot is free there now")
    assert "r20 free; 4 waiting, 0 places for this job; " in why, "the reason says why 13 free slots took no job"
    assert auto._pool(docs[CPU1]["research"], 4, 16)[:3] == (9, 0, 3), \
        "the head takes 60 of the 70 GiB free; the 10 left start nobody, so the job has no place and waits behind the three"
    assert auto.expect(docs, (CPU1, "research"), 4, 16).minutes == pytest.approx(4 * auto.HOLD_MIN / 9), "there it would wait"
    assert auto._pool(cpu1(70)["research"], 4, 16)[1] == 1, "with nobody in line: 70 GiB is one slot's 40, not thirteen"
    assert auto._pool(cpu1(70, waiters=1)["research"], 4, 16)[1:3] == (0, 0), "and that one is the waiter's"
    assert auto._pool(cpu1(400, asks=(64, 64, 16, 16))["research"], 4, 16)[:3] == (12, 5, 0), \
        "400 GiB: 60, 60, 40 and 40 for the line, and 200 for five more runs in the nine slots left"
    assert auto._pool(cpu1(400, asks=(64, 64, 16, 16))["research"], 4, 128)[1] == 1, "or one that reserves 120"
    assert auto.choose({**docs, CPU1: cpu1(400, asks=(64, 64, 16, 16))}, 4, 16)[0] == (CPU1, "research"), "five places beat one"
    assert auto._pool(cpu1(-20)["research"], 4, 0)[1] == 0, "reserved past what is free: no slot with a mem= starts a run"
    booked = [{**slot(f"r{i}"), "book": 15} for i in range(10)]                                     # `mem=40 book=15`, all idle
    assert auto._pool(pool(*booked, free_gb=60)["research"], 4, 16)[1] == 4, "a slot with book= is a place at what it books: 15 each"
    assert auto._pool(pool(*booked, free_gb=60, asks=(16, 16))["research"], 4, 16)[:3] == (2, 2, 0), \
        "and the line ahead takes what it books there too: 15 each of the 60, where their mem= of 40 would start one and no job"
    assert auto._pool(pool(*booked, free_gb=60, asks=(64,))["research"], 4, 16)[:3] == (1, 0, 0), \
        "a waiter asking over book= takes its ask, 60: it starts, and nothing is left for the job"


def test_a_job_with_no_place_waits_behind_every_waiter_that_stays_those_it_would_pass_now_too():
    """A machine with its memory reserved kept an idle slot while runs asking 32 GB waited in its line for memory. A run
    asking as much passes each of them while memory is short for that one, so counted by the waiters it can't pass now it
    was behind nobody, and that pool, with eleven slots running, read as the shortest line there was: the job went to the
    machine with the least memory and its launch said it would start in two minutes. What frees there goes first to the
    earlier waiters it fits."""
    starved = pool(*_held(11, mem=None), slot("r11", mem=None), asks=(32, 32, 32, 32, 60), minutes=15, free_gb=-20)
    docs = {N2: DOWN, CPU1: starved, CPU5: pool(*_held(2))}
    assert auto._pool(starved["research"], 4, 32)[:3] == (11, 0, 5), "no place for it, and five ahead for the eleven slots that run"
    target, why = auto.choose(docs, 4, 32)
    assert target == (CPU5, "research") and why.endswith(f"the shortest line per run that fits is {CPU5}'s research pool"), \
        "one run's wait for two slots is shorter than six runs' for eleven"
    assert auto.expect(docs, (CPU1, "research"), 4, 32).minutes == pytest.approx(6 * auto.HOLD_MIN / 11)
    assert auto._pool(pool(*_held(8), asks=(448,), free_gb=100)["research"], 4, 16)[1:3] == (0, 1), \
        "so too behind a head it would pass once a slot frees: with no place, every waiter that stays is counted"
    with pytest.raises(auto.Stalled, match="a run has waited 15 min while slot r11 is free, for more memory than the -20 GB free there"):
        auto._pool(starved["research"], 4, 64)                               # a run that passes nobody: stalled behind them


def test_the_pool_with_the_most_places_takes_the_job_not_the_first_that_has_one():
    """A launch reads the lines half a minute before its run joins one, so the launches of one half minute read the same
    lines. Sent to the first pool by number with a slot free, they all went to its one free slot and all but one waited there,
    beside a pool with a slot for each."""
    cpu3 = "vy-nebius-cpu-3"
    one = pool(*_held(7), slot("r7"))
    four = pool(*_held(4), *(slot(f"r{i}") for i in range(4, 8)))
    docs = {N2: DOWN, CPU1: one, cpu3: four, CPU5: four}
    target, why = auto.choose(docs, 4, 16)
    assert target == (cpu3, "research"), "the most places, and of two pools with as many the first by number"
    assert why.endswith(f"{cpu3}'s research pool has the most places for it (4): a slot is free there now")
    assert f"{CPU1}'s research pool: " in why and f"{CPU5}'s research pool: " in why, "the reason names every pool it read"
    sent = [auto.choose(docs, 4, 16)[0] for _ in range(4)]                  # four launches that read the same lines
    assert all(auto._pool(docs[m][p], 4, 16)[1] >= sent.count((m, p)) for m, p in set(sent)), "each has a place where it was sent"
    roomy = pool(*(slot(f"r{i}") for i in range(10)), free_gb=50)           # ten idle slots, and memory for one run
    assert auto.choose({N2: DOWN, CPU1: roomy, CPU5: pool(slot("r0"), slot("r1"), free_gb=300)}, 4, 16)[0] == (CPU5, "research"), \
        "places are counted in memory too: two beat ten idle slots that start one run"
    both = {N2: DOWN, CPU1: {**one, **lean(build=(2, 2, 0))}, cpu3: {**one, **lean(build=(2, 1, 0))}, CPU5: {**four, **lean(build=(2, 2, 1))}}
    assert auto.choose(both, 4, 16, lean_pool="build")[0] == (cpu3, "research"), "for a Lean run, among the pools with a Lean slot free too"


def _slot_py():
    """The tree's slot.py, loaded as the probe's copy of it is."""
    spec = importlib.util.spec_from_file_location("check_slot_of_the_tree", REPO / auto.SLOT_PY)
    slot_py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slot_py)
    return slot_py


def test_the_placer_counts_the_line_ahead_as_slot_py_does(tmp_path, monkeypatch):
    """A place is slot.py's to give, so the placer reads a line as slot.py does: who waits is `waiting`'s answer, each with
    the ask its ticket names, and who a run joining now stays behind is `ahead`'s, with the pass a small run gets
    (BACKFILL_MAX_GB) over the lane waiters that only memory keeps from starting. What a run reserves in a slot is slot.py's
    too: `booked`, of the slot as `asking` raised it to the run's ask."""
    slot_py = _slot_py()
    assert auto.BACKFILL_MAX_GB == slot_py.BACKFILL_MAX_GB
    for s in ({"mem": None}, {"mem": 40}, {"mem": 40, "book": 15}, {"mem": 40, "book": 40}):
        for ask in (0, 12, 16, 32, 40, 43, 64):
            monkeypatch.setenv(slot_py.RUN_MEM_ENV, str(ask))
            assert auto._need(s, ask) == (slot_py.booked(slot_py.asking([s])[0]) or 0), (s, ask)

    def join(ask: int, priority: bool = False) -> tuple[Path, int]:
        if ask:
            monkeypatch.setenv(slot_py.RUN_MEM_ENV, str(ask))
        else:
            monkeypatch.delenv(slot_py.RUN_MEM_ENV, raising=False)
        return slot_py.join(tmp_path, priority=priority, who=slot_py.asker())

    held = [join(ask)[1] for ask in (448, 16, 0, 64)] + [join(900, priority=True)[1]]
    line = [{**w, "minutes": 0} for w in slot_py.waiting(tmp_path / slot_py.LINE, datetime(2030, 1, 1, tzinfo=UTC))]
    assert [(w["priority"], w.get("mem_gb")) for w in line] == [(True, 900), (False, 448), (False, 16), (False, None), (False, 64)]
    for mine in (0, 16, 32, 33, 200):
        ticket, fd = join(mine)
        for free in (-5, 0, 10, 15, 60, 100, 418, 1000):
            ahead = slot_py.ahead(ticket) if mine > slot_py.BACKFILL_MAX_GB else slot_py.ahead(ticket, free_gb=free)
            assert len(auto._behind(line, mine, free)) == ahead, (mine, free)
        os.close(fd)
        ticket.unlink()
    for fd in held:
        os.close(fd)
    doc = pool(slot("r0", mem=None), slot("r1", mem=None), free_gb=100)["research"]
    assert auto._pool({**doc, "waiters": line[1:2]}, 4, 16)[1:3] == (2, 0), "a 448 GB head that 100 GiB can't start is passed"
    assert auto._pool({**doc, "waiters": line[1:2]}, 4, 64)[1:3] == (0, 1), "not by a run that asks over BACKFILL_MAX_GB"
    assert auto._pool({**doc, "waiters": line[:2]}, 4, 16)[1:3] == (0, 2), \
        "a priority waiter is never passed, and with no place the job waits behind both that stay"
    assert auto._pool({**doc, "waiters": line[3:4]}, 4, 16)[1:3] == (1, 0), "a waiter with no ask takes the first slot; the next is free"


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
    with pytest.raises(auto.Unplaceable, match="no slot of .* gives 64 CPUs:"):
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
    assert docs[CPU1] == {"research": {"file": "/workspace/research/locks/research/slots", "slots": [], "waiters": []}}, \
        "a run that takes no Lean slot asks for none"
    assert (claims / "gone.cpus").exists(), "the probe only reads: a leftover is the next taker's to remove"
    # a slot.py that reads memory: the probe reads it once for the pool, however many slots ask, and reports what is free
    (src / auto.SLOT_PY).write_text(
        "import os\ncalls = []\n"
        "def memory_gb():\n    calls.append('memory')\n    return 100\n"
        "def reserved_gb(priority=False):\n    calls.append('reserved')\n    return 30\n"
        "def status_doc(now):\n"
        "    slots = [{'name': f'r{i}', 'free': memory_gb() - reserved_gb(priority=False)} for i in range(3)]\n"
        "    return {'file': os.environ['CHECK_SLOTS'], 'slots': slots, 'waiters': [], 'calls': calls}\n")
    counted = auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[CPU1]["research"]
    assert counted["free_gb"] == 70 and [s["free"] for s in counted["slots"]] == [70, 70, 70]
    assert counted["calls"] == ["memory", "reserved"], "once a pool, not once a slot: each read walks every live reservation's process tree"
    (src / auto.SLOT_PY).write_text("import os\ndef status_doc(now):\n    return {'file': os.environ['CHECK_SLOTS'], 'slots': [], 'waiters': []}\n")
    assert auto.read(src, 8, reach=lambda name: [sys.executable, "-"])[N2]["share"]["fits"] is True, "nothing holds them now"
    monkeypatch.setattr(auto, "DISK_PATH", str(tmp_path))
    monkeypatch.setattr(auto, "MIN_FREE_INODES", 2**62)
    short = auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[CPU1]["research"]
    assert short["error"].startswith(f"disk: {tmp_path} has ") and "inodes free" in short["error"], "27 Oct: cpu-5 out of inodes"
    target, why = auto.choose({N2: share(free=8), CPU1: {"research": short}}, 4, 16)
    assert target is None and "disk:" in why, "a pool on a full disk is skipped like one that can't be read"
    failing = auto.read(src, 4, reach=lambda name: [sys.executable, "-c", "import sys; sys.exit('no python here')"])
    assert failing == {m: "exit 1: no python here" for m in (N2, *NODES)}


def test_the_probe_counts_a_pools_free_memory_as_its_slots_count_the_live_reservations(tmp_path, monkeypatch, node_2):
    """A pool's `free_gb` is what a lane run would find free there, which `choose` places by. Slots with peaks= count the
    live lane reservations as a crowd reaches them (slot.py's `reserved_gb`), so the probe counts them the same way, with the
    tree's own slot.py: counted whole, a pool whose slots would start the job reads as short of memory and the job goes to
    wait elsewhere. Here the machine's memory is a stated 200 GiB and six live runs each reserved 30 GiB and reached none."""
    from research.pods.nebius import slot_only
    slots, ramlock, src = tmp_path / "locks" / "research" / "slots", tmp_path / "ramlock", tmp_path / "src"
    slots.parent.mkdir(parents=True)
    ramlock.mkdir()
    for i in range(6):
        (ramlock / f"r{i}.json").write_text(json.dumps({"gb": 32.2, "run_id": f"r{i}"}))
    monkeypatch.setenv("RESEARCH_RAMLOCK_DIR", str(ramlock))
    monkeypatch.setattr(slot_only, "POOLS", {**slot_only.POOLS, auto.SPILL_POOL: str(slots)})
    monkeypatch.setattr(auto, "DISK_PATH", str(tmp_path / "no-disk"))
    (src / auto.SLOT_PY).parent.mkdir(parents=True)
    (src / auto.SLOT_PY).write_text((REPO / auto.SLOT_PY).read_text() + "\n\ndef memory_gb(root=None):\n    return 200\n")

    def read(lines: str) -> dict:
        slots.write_text(lines)
        return auto.read(src, 4, reach=lambda name: [sys.executable, "-"])[CPU1]["research"]

    whole = read("r0 0-3 mem=30\nr1 4-7 mem=30\n")
    assert whole["free_gb"] == 200 - 179 and all(s["blocked"].startswith("21 GB of memory free") for s in whole["slots"])
    crowd = read("r0 0-3 mem=30 peaks=2\nr1 4-7 mem=30 peaks=3\nt0 8-11 only=trains mem=150\n")
    assert crowd["free_gb"] == 200 - 112, "the largest peaks= of the lane slots: three runs' 30 whole and three at a quarter"
    assert [(s["name"], s.get("peaks"), s["blocked"]) for s in crowd["slots"]] == [("r0", 2, None), ("r1", 3, None), ("t0", None, None)]
    target, why = auto.choose({N2: share(waiters=5), CPU1: {"research": crowd}}, 4, 32)
    assert target == (CPU1, "research") and why.endswith("a slot is free there now")
    target, why = auto.choose({N2: share(free=8), CPU1: {"research": whole}}, 4, 32)
    assert target is None and f"21{auto.MEMORY_BLOCK}" in why, "the same ledger counted whole: no memory there"
    mixed = read("r0 0-3 mem=30 peaks=2\nr1 4-7 mem=30\n")
    assert mixed["free_gb"] == 21 and [s["blocked"] is None for s in mixed["slots"]] == [True, False], \
        "one lane slot without peaks=: the pool's figure is the least any of its slots would find free"


def test_the_probe_reads_a_machines_lean_slots_with_the_trees_lean_slot_py(tmp_path, monkeypatch, node_2):
    """For a run that takes a Lean slot the probe runs the tree's own lean_slot.py on the machine's Lean slots, as
    `lean_slot.py --status --json` does: here the repository's, on a slots file and a line this test keeps."""
    src, slots = tmp_path / "src", tmp_path / "locks" / "lean-slots"
    for f in (auto.SLOT_PY, auto.LEAN_SLOT_PY):
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text((REPO / f).read_text())
    monkeypatch.setattr(auto, "LEAN_SLOTS", str(slots))

    def read(**kw) -> dict:
        """Node 2's answer, the one machine asked."""
        with monkeypatch.context() as m:
            m.setattr(auto, "spill", lambda: ())
            return auto.read(src, 4, reach=lambda name: [sys.executable, "-"], **kw)[N2]

    assert "lean" not in read() and read()["share"]["free"] == 48
    none = read(lean=True)
    assert none["share"]["free"] == 48 and none["lean"]["file"] == str(slots) and none["lean"]["pools"] is None
    assert auto._lean(none, "build") == (0, "; no Lean slots to wait for there"), "a machine with no slots file caps no Lean"
    slots.parent.mkdir()
    slots.write_text("# NAME N\nbuild 2   # the records runs'\n# audit 4\ncheck 0\n")
    doc = read(lean=True)
    assert doc["lean"]["pools"]["build"] == {"n": 2, "slots": [{"index": i, "past_n": False, "holder": None} for i in range(2)], "waiters": []}
    assert auto._lean(doc, "build") == (0, "; Lean `build`: 2 of 2 free, 0 waiting")
    for absent in ("audit", "check"):                                   # commented out, and listed with no slot
        assert auto._lean(doc, absent) == (None, f"its Lean slots give `{absent}` none")
    line = slots.parent / "lean-line-build"
    line.mkdir()
    waiters = [cores.locked(line / f"{10**18 + i:020d}-{7 + i}", "{}\n") for i in range(2)]
    try:
        doc = read(lean=True)
    finally:
        for fd in waiters:
            os.close(fd)
    assert [w["pid"] for w in doc["lean"]["pools"]["build"]["waiters"]] == [7, 8]
    assert auto._lean(doc, "build") == (1.5, "; Lean `build`: 2 of 2 free, 2 waiting"), "the free slots are the waiters' ahead"
    assert auto._lean(read(lean=True), "build")[0] == 0, "a ticket nobody holds is a waiter that died"
    target, why = auto.choose({N2: share(waiters=5), CPU1: {**pool(slot("r0")), "lean": doc["lean"]},
                               CPU5: {**pool(slot("r0", held=True)), "lean": read(lean=True)["lean"]}}, 4, 16, lean_pool="build")
    assert target == (CPU5, "research") and "Lean `build`: 2 of 2 free, 2 waiting; " in why, "the documents as the probe gives them"
    slots.write_text("build two\n")
    assert auto._lean(read(lean=True), "build") == (math.inf, f"; its Lean slots weren't read (Refused: {slots}: 'build two' is not `POOL N`)")


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


def test_on_auto_places_a_records_run_by_its_lean_pool_and_says_which(tmp_path, monkeypatch, capsys):
    """The launcher asks for the machines' Lean slots only for a run that takes one, places it by the pool it takes (the
    run's --env names another than `build`), and its one line of reason names that pool."""
    asked, launched = [], []
    docs = {N2: share(waiters=56), CPU1: {**pool(slot("r0")), **lean(build=(8, 8, 2), proofs=(6, 2, 0))},
            CPU5: {**pool(slot("r0", held=True), waiters=2), **lean(build=(3, 1, 0))}}
    monkeypatch.setattr(auto, "read", lambda source, cpus, **kw: asked.append(kw) or docs)
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing([]))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    records = ["python3", "tools/verity/lean/lean_changed.py", "--records", "security", "--update"]
    assert main([*runs, "--", "pytest", "-q"]) == 3
    assert f"spilled to {CPU1}'s research pool" in (err := capsys.readouterr().err) and "Lean" not in err
    assert main([*runs, "--", *records]) == 3
    err = capsys.readouterr().err
    assert f"research run --on auto: spilled to {CPU5}'s research pool, from " in err
    assert f"{CPU5}'s research pool, the shortest line per run that fits where a slot of Lean pool `build` is free)" in err
    assert main([*runs, "--env", "LEAN_RECORDS_POOL=proofs", "--", *records]) == 3
    err = capsys.readouterr().err
    assert f"spilled to {CPU1}'s research pool" in err and "a slot is free there now, and one of Lean pool `proofs` with it)" in err
    assert asked == [{}, {"lean": True}, {"lean": True}]
    assert [kw["machine_name"] for kw in launched] == [CPU1, CPU5, CPU1]
    assert launched[1]["command"][-6:] == ["19", *records], "the run itself is unchanged"


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


def _suites_repo(tmp_path: Path) -> Path:
    """A repository whose one suite's test processes take 16 GB each, with the tree's slot.py: a run of all its suites plans
    its 4 processes on 64 GB and reserves 32 of it in a pool's slot (research.suite_size)."""
    repo = _auto_repo(tmp_path)
    for path, body in {".": '[project]\nname = "verity"\n[tool.pytest.ini_options]\n',
                       "a": '[project]\nname = "a"\n[tool.pytest.ini_options]\n[tool.verity.tests]\nworker_gb = 16\n'}.items():
        (repo / path).mkdir(exist_ok=True)
        (repo / path / "pyproject.toml").write_text(body)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    (repo / auto.SLOT_PY).parent.mkdir(parents=True)
    (repo / auto.SLOT_PY).write_text((REPO / auto.SLOT_PY).read_text())
    return repo


def test_a_named_suites_run_lasts_the_bound_its_workers_stretch(tmp_path):
    """A suite measured on 5 workers (`timeout_workers`) run on 3 has suites.py's bound stretched to 5/3 of its timeout_s; the run
    sized for it lasts that, not the 14,400 s that would kill it first."""
    from research import suite_size

    repo = _auto_repo(tmp_path)
    (repo / "a").mkdir()
    (repo / "a" / "pyproject.toml").write_text('[project]\nname = "a"\n[tool.pytest.ini_options]\n[tool.verity.tests]\n'
                                               "parallel = true\ntimeout_s = 14400\ntimeout_workers = 5\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    run = [suite_size.SUITES_PY, "a", "--jobs"]
    assert suite_size.size(repo, [*run, "3"], None)["max_s"] == 24000 + suite_size.SYNC_S
    assert suite_size.size(repo, [*run, "8"], None)["max_s"] == 14400 + suite_size.SYNC_S


def test_a_run_of_many_suites_is_placed_and_counted_in_a_line_by_what_it_reserves(tmp_path, monkeypatch, capsys):
    """A run of many suites holds half of what it plans on in a pool's slot, so half is what a place for it takes of the
    machine's free memory, and what it takes from the runs behind it while it waits in a line. Counted by the whole figure,
    it would be sent past a pool with room for three of it to one with room for two."""
    roomy = pool(*(slot(f"r{i}", mem=24) for i in range(4)), free_gb=100)       # 32 GB reserves 30 GiB there, 64 GB 60
    docs = {N2: share(waiters=56), CPU1: roomy, CPU5: pool(slot("r0", mem=24), slot("r1", mem=24), free_gb=300)}
    launched = []
    monkeypatch.setattr(auto, "read", lambda source, cpus: docs)
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing([]))
    monkeypatch.chdir(_suites_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    suites = ["uv", "run", "tools/verity/check/suites.py"]
    slot_py = _slot_py()
    assert main([*runs, "--", *suites]) == 3
    assert main([*runs, "--mem-gb", "64", "--", *suites]) == 3
    halved, whole = launched
    assert (halved["machine_name"], halved["env"][slot_py.RUN_MEM_ENV]) == (CPU1, "32"), "three places of 30 GiB in the 100 free"
    assert (whole["machine_name"], whole["env"][slot_py.RUN_MEM_ENV]) == (CPU5, "64"), \
        "a --mem-gb given is reserved whole: 60 GiB is one place of the 100, and the other pool has two"
    err = capsys.readouterr().err
    assert f"{CPU1}'s research pool has the most places for it (3)" in err and f"{CPU5}'s research pool has the most places for it (2)" in err
    monkeypatch.setenv(slot_py.RUN_MEM_ENV, halved["env"][slot_py.RUN_MEM_ENV])                     # the run, joining its slot's line
    _ticket, fd = slot_py.join(tmp_path, priority=False, who=slot_py.asker())
    line = [{**w, "minutes": 0} for w in slot_py.waiting(tmp_path / slot_py.LINE, datetime(2030, 1, 1, tzinfo=UTC))]
    os.close(fd)
    assert [w["mem_gb"] for w in line] == [32], "its ticket names what it reserves"
    assert auto._pool({**roomy["research"], "waiters": line}, 4, 32)[:3] == (1, 2, 0), \
        "ahead in the line it takes 30 GiB of the 100, which leaves two places for a run like it; 60 would leave one"


def test_on_auto_spilled_to_a_pool_reserves_a_share_of_what_a_run_of_many_suites_plans_on(tmp_path, monkeypatch, capsys):
    """A pool's slot holds a run's reservation against the node for as long as the run holds the slot. A sized run of many
    suites (research.suite_size) asks the slot for its `reserve_gb` and hands slot.py the whole figure as `--budget-gb`, which
    its suites.py counts its test processes on; it is placed by what it reserves. Its own --mem-gb is reserved as given, and on
    the queue the whole figure is the job's cap."""
    launched, placed = [], []
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: share(waiters=56), CPU1: pool(slot("r8", mem=40))})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    repo = _suites_repo(tmp_path)
    monkeypatch.chdir(repo)
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    suites = ["uv", "run", "tools/verity/check/suites.py"]
    assert main([*runs, "--", *suites]) == 3
    (kw,) = launched
    cmd = kw["command"]
    assert kw["mem_gb"] is None and kw["env"]["CHECK_RUN_MEM_GB"] == "32", "half of 4 processes at a's 16 GB"
    assert cmd[cmd.index("--budget-gb"):cmd.index("--") + 1] == ["--budget-gb", "64", "--"], "slot.py's option, before CMD"
    assert cmd[-6:] == ["nice", "-n", "19", *suites]
    err = capsys.readouterr().err
    assert "64 GB (32 of it reserved in a pool's slot)" in err
    assert "--mem-gb 32 goes to the slot" in err and "gives the run 64 GB to plan its processes on" in err

    launched.clear()
    assert main([*runs, "--mem-gb", "64", "--", *suites]) == 3
    (kw,) = launched
    assert kw["env"]["CHECK_RUN_MEM_GB"] == "64" and "--budget-gb" not in kw["command"], "a --mem-gb given is reserved as given"

    launched.clear()
    assert main([*runs, "--cpus", "16", "--", *suites]) == 3, "no slot here gives 16 CPUs: the queue's"
    assert launched == [] and [spec["mem-gb"] for spec in placed] == [256], "the queue's job is capped at the whole figure"

    (repo / auto.SLOT_PY).write_text("")
    assert main([*runs, "--", *suites]) == 3
    (kw,) = launched
    assert kw["env"]["CHECK_RUN_MEM_GB"] == "64" and "--budget-gb" not in kw["command"], "a tree from before slot.py's option"


def test_on_auto_keeps_a_gpu_job_or_one_too_big_for_any_slot_on_node_2(tmp_path, monkeypatch, capsys):
    placed = []
    monkeypatch.setattr(auto, "read", lambda source, cpus, **kw: {N2: share(waiters=56), CPU1: pool(slot("r0"))})
    monkeypatch.setattr(remote, "queue_placement", _refusing(placed))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--gpus", "1", "--", "nvidia-smi"]) == 3
    assert "a GPU or quiet job, which only the queue books" in capsys.readouterr().err
    assert main([*runs, "--cpus", "16", "--mem-gb", "160", "--", "pytest"]) == 3
    assert "no slot of /workspace/research/locks/research/slots gives 16 CPUs" in capsys.readouterr().err
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
            m.setattr(auto, "spill", lambda: ())                                       # node 2 alone is asked
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


@pytest.fixture
def hold_25(monkeypatch):
    """`expect`'s estimates at 25 min a run, whatever HOLD_MIN is set to."""
    monkeypatch.setattr(auto, "HOLD_MIN", 25)


def _held(n: int, **kw) -> list[dict]:
    return [slot(f"r{i}", held=True, **kw) for i in range(n)]


def test_a_launch_is_told_when_its_run_should_start_from_the_line_the_probe_read(hold_25):
    """A launch that queued said nothing of its wait. It is read off the documents the placement already has: the runs ahead
    and this one, per slot that runs, at HOLD_MIN each."""
    at = f"{CPU1}'s research pool"
    docs = {N2: share(waiters=56), CPU1: pool(slot("r0"))}
    target, _ = auto.choose(docs, 4, 16)
    assert (now := auto.expect(docs, target, 4, 16)) == auto.Wait(0, "", False) and now.told(at) == f"it starts now on {at}"
    docs = {N2: DOWN, CPU1: pool(*_held(5))}
    five = auto.expect(docs, auto.choose(docs, 4, 16)[0], 4, 16)
    assert five == auto.Wait(5.0, "0 waiting ahead for the 5 slots that run there", False), "one run's hold over 5 slots"
    assert five.told(at) == f"it should start in about 5 min on {at}: 0 waiting ahead for the 5 slots that run there"
    docs = {N2: DOWN, CPU1: pool(*_held(5), waiters=7, minutes=29)}
    forty = auto.expect(docs, auto.choose(docs, 4, 16)[0], 4, 16)
    assert forty == auto.Wait(40.0, "7 waiting ahead (the oldest for 29 min) for the 5 slots that run there", True)
    docs = {N2: DOWN, CPU1: pool(*_held(4), slot("r4"), waiters=1)}
    assert auto.expect(docs, auto.choose(docs, 4, 16)[0], 4, 16) == \
        auto.Wait(5.0, "0 waiting ahead for the 5 slots that run there", False), "the free slot is the waiter's, who then holds it"
    docs = {N2: DOWN, CPU1: pool(*_held(5), waiters=7), CPU5: pool(*_held(5))}
    target, _ = auto.choose(docs, 4, 16)
    assert target == (CPU5, "research") and auto.expect(docs, target, 4, 16).minutes == 5, "the pool it takes has the least wait"
    assert auto.expect(docs, (CPU1, "research"), 4, 16).minutes == 40
    big = pool(slot("r8", cpus="0-31", mem=160, held=True), *_held(4), waiters=3)
    assert auto.expect({N2: DOWN, CPU1: big}, (CPU1, "research"), 16, 150) == \
        auto.Wait(100.0, "3 waiting ahead for the 1 slot that runs there", True), "only a slot that fits the job drains its line"


def test_a_wait_nothing_read_bounds_gets_no_estimate_and_only_a_stalled_line_is_late_for_it(hold_25):
    """An estimate the probe can't back is not given: node 2's claims have no end it reads, and a Lean slot's line has no
    rate. Neither is late on that alone. A stalled pool, placement's last resort, waits on memory however short its line."""
    docs = {N2: share(free=8, waiters=3), CPU1: pool(slot("r0"))}
    target, why = auto.choose(docs, 64, 300)
    assert target is None and why.endswith(f"no pool fits, so {N2}'s line")
    assert auto.expect(docs, target, 64, 300) == auto.Wait(None, "8 of its share's 48 CPUs are free under 2 claim(s), with 3 waiting, "
                                                                 "and nothing read says when a claim ends", False)
    docs = {N2: share(free=48, fits=True), CPU1: pool(slot("r0"))}
    assert auto.expect(docs, auto.choose(docs, 32, 300)[0], 32, 300) == auto.Wait(0, "", False), "node 2 starts it now"
    docs = {N2: DOWN, CPU1: pool(slot("r0", held=True), slot("r1"), asks=(256,) * 3, minutes=88, free_gb=100)}
    target, why = auto.choose(docs, 4, 64)
    assert target == (CPU1, "research") and "every pool that fits is stalled" in why
    stalled = auto.expect(docs, target, 4, 64)
    assert stalled == auto.Wait(None, "its line is stalled: a run has waited 88 min while slot r1 is free, for more memory than "
                                      "the 100 GB free there", True)
    assert stalled.told("there").startswith("no estimate of when it starts on there: its line is stalled")
    no_lean = "then a wait in that slot for one of Lean pool `build`, which has no estimate (Lean `build`: 0 of 3 free, 2 waiting)"
    docs = {N2: DOWN, CPU1: {**pool(slot("r0")), **lean(build=(3, 3, 2))}}
    target, _ = auto.choose(docs, 4, 16, lean_pool="build")
    assert auto.expect(docs, target, 4, 16, lean_pool="build") == auto.Wait(None, f"a CPU slot is free there now, {no_lean}", False)
    docs = {N2: DOWN, CPU1: {**pool(*_held(5), waiters=7), **lean(build=(3, 3, 2))}}
    assert auto.expect(docs, (CPU1, "research"), 4, 16, lean_pool="build") == auto.Wait(
        None, f"about 40 min for a CPU slot (7 waiting ahead for the 5 slots that run there), {no_lean}", True), \
        "late on its CPU slot alone"
    docs = {N2: DOWN, CPU1: {**pool(slot("r0")), **lean(build=(3, 1, 0))}}
    assert auto.expect(docs, (CPU1, "research"), 4, 16, lean_pool="build") == auto.Wait(0, "", False), "both slots free: now"
    docs = {N2: DOWN, CPU1: {**pool(slot("r0")), **lean(check=(2, 0, 0), build=(0, 0, 0))}}
    assert auto.expect(docs, (CPU1, "research"), 4, 16, lean_pool="build") == auto.Wait(None, "its Lean slots give `build` none", False)


def test_the_smallest_change_of_a_request_that_starts_now_keeps_the_most_of_what_it_asked(hold_25):
    """A refused launch is told the way on that needs no wait: the request cut to a slot that is free now and to the memory
    free on its machine, the cut that keeps the most of it."""
    docs = {N2: DOWN, CPU1: pool(slot("r8", cpus="0-15", mem=160, held=True), waiters=7),
            CPU5: pool(slot("r0", cpus="8-19", mem=40), slot("r1", cpus="20-23", mem=16), free_gb=40)}
    target, _ = auto.choose(docs, 16, 150)
    assert target == (CPU1, "research") and auto.expect(docs, target, 16, 150).late
    assert auto.smaller(docs, 16, 150) == (12, 42, (CPU5, "research")), "r0's 12 CPUs, and the 40 GiB free there"
    assert auto.choose(docs, 12, 42)[1].endswith("a slot is free there now")
    assert auto.smaller(docs, 4, 16) is None, "it starts now as asked: nothing to cut"
    docs[CPU5] = pool(slot("r0", cpus="8-19", mem=40), slot("r1", cpus="20-23", mem=16), free_gb=400)
    assert auto.smaller(docs, 16, 150) == (12, 150, (CPU5, "research")), "the memory is free there: its CPUs alone are cut"
    docs[CPU5] = pool(slot("r0", cpus="8-19", mem=40, held=True), slot("r1", cpus="20-23", mem=16, blocked="window 02:00-06:00"),
                      free_gb=400)
    assert auto.smaller(docs, 16, 150) is None, "no slot is free anywhere: no cut starts it"
    docs[CPU5] = pool(slot("r0", cpus="8-19", mem=40), slot("r1", cpus="20-23", mem=16), waiters=2, free_gb=400)
    assert auto.smaller(docs, 16, 150) is None, "the free slots are their waiters'"
    docs[CPU5] = pool(slot("r0", cpus="8-19", mem=40), free_gb=0)
    assert auto.smaller(docs, 16, 150) is None, "no memory is free where the slot is"
    docs = {N2: DOWN, CPU1: {**pool(slot("r8", cpus="0-15", mem=160, held=True), waiters=7), **lean(build=(3, 0, 0))},
            CPU5: {**pool(slot("r0", cpus="8-19", mem=40), free_gb=40), **lean(build=(0, 0, 0))}}
    assert auto.smaller(docs, 16, 150) == (12, 42, (CPU5, "research"))
    assert auto.smaller(docs, 16, 150, lean_pool="build") is None, "its free slot is on a machine whose Lean slots give `build` none"


def test_on_auto_says_when_the_run_should_start_and_warns_or_refuses_a_late_one(tmp_path, monkeypatch, capsys, hold_25):
    """A launch that starts now or within START_WITHIN_MIN says so and goes. One that would wait 40 min is warned in one line
    while REFUSE is off; with it on it is refused before anything is launched, with the ways on, and --accept-wait queues it."""
    launched, read = [], {}
    lines = {"now": {N2: DOWN, CPU1: pool(slot("r0"))}, "5 min": {N2: DOWN, CPU1: pool(*_held(5))},
             "40 min": {N2: DOWN, CPU1: pool(*_held(5), waiters=7, minutes=29)}}
    monkeypatch.setattr(auto, "read", lambda source, cpus: read["docs"])
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.setattr(remote, "queue_placement", _refusing([]))
    monkeypatch.chdir(_auto_repo(tmp_path))
    at, head = f"{CPU1}'s research pool", "research run --on auto: "

    def launch(line: str, *flags: str) -> tuple[int, list[str]]:
        """(`research run --on auto`'s exit with the probe reading ``line``, what it said after where the run went)"""
        read["docs"] = lines[line]
        rc = main(["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto", *flags, "--", "pytest", "-q"])
        err = capsys.readouterr().err.splitlines()
        return rc, [e[len(head):] for e in err if e.startswith(head) and not e.startswith(head + "spilled to ")]

    late = (f"it isn't expected to start within 10 min: it should start in about 40 min on {at}: 7 waiting ahead (the oldest "
            f"for 29 min) for the 5 slots that run there")
    assert auto.REFUSE is False, "off until infra turns it on"
    assert launch("now") == (3, [f"it starts now on {at}"])
    assert launch("5 min") == (3, [f"it should start in about 5 min on {at}: 0 waiting ahead for the 5 slots that run there"])
    assert launch("40 min") == (3, [f"warning: {late}"]) and len(launched) == 3, "warned in one line, and launched"
    monkeypatch.setattr(auto, "REFUSE", True)
    assert launch("40 min") == (2, [f"refused: {late}. --accept-wait queues it there anyway"]) and len(launched) == 3
    assert launch("40 min", "--accept-wait") == (3, [f"warning: {late}"]) and len(launched) == 4
    assert launch("now")[0] == launch("5 min")[0] == 3 and len(launched) == 6, "a run on time is never refused"
    lines["cut"] = {N2: DOWN, CPU1: pool(slot("r8", cpus="0-15", mem=160, held=True), waiters=7),
                    CPU5: pool(slot("r0", cpus="8-19", mem=40), free_gb=40)}
    rc, (said,) = launch("cut", "--cpus", "16", "--mem-gb", "150")
    assert rc == 2 and said.endswith(f". --accept-wait queues it there anyway; `--cpus 12 --mem-gb 42` starts now on {CPU5}'s research pool")
    monkeypatch.setattr(auto, "REFUSE", False)
    rc, (said,) = launch("cut", "--cpus", "16", "--mem-gb", "150")
    assert rc == 3 and said.startswith("warning: ") and said.endswith(f". `--cpus 12 --mem-gb 42` starts now on {CPU5}'s research pool")
    assert main(["run", "--on", CPU1, "--project", "verity", "--accept-wait", "--", "true"]) == 2 and len(launched) == 7
    assert "--accept-wait is for --on auto" in capsys.readouterr().err


def test_on_auto_counts_a_runs_limit_from_its_start_and_its_custody_key_outlasts_the_wait(tmp_path, monkeypatch, capsys, hold_25):
    """A run that waits in a pool's line keeps its whole limit: the launcher gives its runner the pool's wait on top and the
    slot the limit itself, which slot.py counts from the slot taken (here 40 min after the runner's start, on a clock the test
    gives). The custody key is minted at launch, so by default it outlasts that sum: under the default lifetime a 150-min
    run is refused for a `--timeout 23400` it never passed."""
    from research.pods.nebius import slot_only

    launched = []
    monkeypatch.setattr(auto, "read", lambda source, cpus: {N2: DOWN, CPU1: pool(*_held(5), waiters=7, minutes=29)})
    monkeypatch.setattr(remote, "launch", _refusing(launched))
    monkeypatch.chdir(_auto_repo(tmp_path))
    runs = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "auto"]
    assert main([*runs, "--max-min", "150", "--", "pytest", "-q"]) == 3
    assert "outlives the custody key" not in (err := capsys.readouterr().err)
    assert "its custody key lives 8h, past the wait for a slot and the run after it" in err
    assert main([*runs, "--max-min", "150", "--custody-ttl", "7h", "--", "pytest", "-q"]) == 3
    assert main([*runs, "--", "pytest", "-q"]) == 3
    assert [kw["custody_ttl"] for kw in launched] == ["8h", "7h", "6h"], "the sum's hours and one for the upload; a named one; the default"
    command, limit = launched[0]["command"], 150 * 60
    assert command[command.index("--run-max") + 1] == f"{limit}", "the slot gets the limit"
    assert remote.flag_value(launched[0]["harness_args"], "--timeout") == f"{slot_only.POOL_WAIT_S + limit}", "the runner the wait too"
    spec = importlib.util.spec_from_file_location("check_slot_of_the_tree", REPO / auto.SLOT_PY)
    slot_py = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slot_py)
    run = tmp_path / "run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "running", "started_mono": 1000.0, "timeout_s": slot_only.POOL_WAIT_S + limit}))
    for waited in (40 * 60, 0):
        slot_py.limit_run(limit, "r0", str(run), mono=lambda: 1000.0 + waited)
        assert json.loads((run / slot_py.RUN_TIMEOUT).read_text())["timeout_s"] == waited + limit, "its limit, from the slot taken"
