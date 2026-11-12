"""`research queue stranded` and the watch `stranded` (research.stranded): a job waiting in one line while a place that could
run it stands free in another. Every document here is made up; the clock is the tests' own."""
from __future__ import annotations

import json
import subprocess

import pytest

from research import auto, stranded
from research import monitors as M

T0 = 1_794_508_500.0
A, B, C = "vy-nebius-cpu-1", "vy-nebius-cpu-2", "vy-nebius-cpu-3"
CPUS = [A, B, C]
SECRET = "hunter2-token"
MEM_BLOCK = f"41{auto.MEMORY_BLOCK}, under the 200 GB a check here may peak at"


@pytest.fixture(autouse=True)
def cpu_machines(monkeypatch, tmp_path):
    """The registry names CPUS, each reached by its name; the tree sent is an empty slot.py and lean_slot.py."""
    monkeypatch.setattr(auto, "spill", lambda machines=None: tuple((m, auto.SPILL_POOL) for m in CPUS))
    monkeypatch.setattr(auto, "_reach", lambda name: [name])
    for rel in (stranded.SLOT_PY, stranded.LEAN_SLOT_PY):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")
    monkeypatch.setattr(stranded, "TREE", tmp_path)


def slot(name: str, cpus: str = "0-15", *, held: bool = False, blocked: str | None = None, trains: bool = False,
         mem: int | None = 40, upcoming: tuple = ()) -> dict:
    return {"name": name, "cpus": cpus, "mem": mem, "trains": trains, "lead": 30, "blocked": blocked, "upcoming": list(upcoming),
            "holder": {"pid": 7, "run": "r-holder", "cmd": f"python3 x --token={SECRET}"} if held else None}


def waiter(minutes: int, run: str | None = None, *, priority: bool = False, twin: str | None = None, mem_gb: float | None = None,
           by: str = "lane-x", said: str | None = None) -> dict:
    w = {"since": "2026-11-12T18:00Z", "minutes": minutes, "priority": priority, "pid": 100 + minutes, "by": by}
    w |= {k: v for k, v in (("run", run), ("twin", twin), ("mem_gb", mem_gb)) if v is not None}
    return w | ({"auto": {"machine": A, "pool": "research", "said": said}} if said else {})


def pool(*slots: dict, waiters: tuple = (), free_gb: float = 400) -> dict:
    return {"file": "/workspace/research/locks/x/slots", "slots": list(slots), "waiters": list(waiters), "free_gb": free_gb}


def machine(peers: list[str] | None = None, lean: dict | None = None, floor: str | None = None, **pools: dict) -> dict:
    return {"pools": pools, "lean": lean or {"file": "lean-slots", "pools": None}, "floor": floor, "no_start": None,
            "available_gb": 500.0, "peers": CPUS if peers is None else peers}


def stranded_jobs(docs: dict, **kw) -> list[str]:
    return [w["job"] for w in stranded.analyse(docs, T0, CPUS, **kw)["waits"] if w["stranded"]]


def shard(minutes: int = 40, run: str = "r-shard", **kw) -> dict:
    return waiter(minutes, run, priority=True, by="ci", **kw)


# ---- a place --------------------------------------------------------------------------------------------------------------

def test_a_shard_in_one_line_with_a_check_slot_free_elsewhere_is_stranded():
    """The class: a candidate's shard waits on cpu-7's slot while a train slot stands free on another machine."""
    docs = {A: machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(),))),
            B: machine(check=pool(slot("t", "0-95", trains=True, mem=400))), C: machine()}
    doc = stranded.analyse(docs, T0, CPUS)
    (w,) = doc["waits"]
    assert w["job"] == "r-shard" and w["stranded"] and w["lines"] == [{"machine": A, "since": "2026-11-12T18:00Z", "minutes": 40}]
    assert [(p["machine"], p["slot"], p["cores"], p["mem"]) for p in w["places"]] == [(B, "t", "0-95", 400)]
    assert doc["pools"]["check"] | {"blocked": None} == {"in_line": 1, "stranded": 1, "places": 1, "blocked": None,
                                                         "idle_in_other_pools": {}, "unread": []}
    assert doc["schema"] == "research/stranded/v1" and doc["read_at"] == "2026-11-12T18:35Z"


def test_a_held_check_slot_on_the_same_cores_closes_it_and_lane_runs_on_them_do_not():
    """A check slot's run starts at once on its cores, so a held check slot that shares one closes it; lane runs there run
    niced and yield, so they close nothing, and the place says how many of its cores they hold."""
    home = machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(),)))
    held_check = machine(check=pool(slot("t", "0-95", trains=True, mem=400), slot("c", "48-63", held=True, trains=True)))
    assert stranded_jobs({A: home, B: held_check}) == []
    lane = machine(check=pool(slot("t", "0-95", trains=True, mem=400)), research=pool(slot("r0", "0-15", held=True)))
    (w,) = stranded.analyse({A: home, B: lane}, T0, CPUS)["waits"]
    assert w["stranded"] and w["places"][0]["lane_cores"] == 16


def test_a_lane_slot_that_shares_cores_with_any_held_slot_is_no_place_and_two_idle_ones_that_overlap_are_one():
    research = pool(slot("r0", "0-15", held=True), waiters=(waiter(30, "r1"), waiter(20, "r2")))
    under_check = machine(check=pool(slot("c", "0-31", held=True, trains=True)), research=pool(slot("r0", "16-31")))
    assert stranded_jobs({A: machine(research=research), B: under_check}) == []
    overlap = machine(research=pool(slot("r0", "0-15"), slot("r1", "8-23")))
    doc = stranded.analyse({A: machine(research=research), B: overlap}, T0, CPUS)
    assert [w["job"] for w in doc["waits"] if w["stranded"]] == ["r1"], "two overlapping idle slots are one place"
    assert doc["pools"]["research"]["blocked"] == {"cores": 1}


@pytest.mark.parametrize("there, job, why", [
    (pool(slot("t", "0-95", trains=True, mem=400, blocked=MEM_BLOCK)), shard(), "a slot slot.py blocks"),
    (pool(slot("t", "0-95", trains=True, mem=400), waiters=(shard(5, "r-own"),)), shard(), "a machine with its own line"),
    (pool(slot("t", "0-95", trains=True, mem=100)), shard(), "a slot under TRAIN_GB"),
    (pool(slot("t", "0-95", trains=True, mem=400)), shard(mem_gb=500), "a slot under the ticket's ask"),
    (pool(slot("t", "0-95", trains=True, mem=400)), waiter(40, "r-lane", by="proofs"), "a lane's check before only=trains"),
])
def test_none_of_these_is_a_place(there, job, why):
    docs = {A: machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(job,))), B: machine(check=there)}
    assert stranded_jobs(docs) == [], why


def test_an_idle_trains_slot_over_the_same_cores_leaves_a_lanes_check_its_lane_slot():
    """Overlapping idle check slots are one place among those that fit the job: a lane's check, which takes no only=trains
    slot, still finds the lane slot under an idle train slot."""
    docs = {A: machine(check=pool(slot("c", "0-15", held=True), waiters=(waiter(40, "r-lane", by="proofs"),))),
            B: machine(check=pool(slot("t", "0-95", trains=True, mem=400), slot("c", "0-15")))}
    (w,) = stranded.analyse(docs, T0, CPUS)["waits"]
    assert w["stranded"] and w["places"][0]["slot"] == "c"
    docs[A]["pools"]["check"]["waiters"] += [shard(30, "r-s1"), shard(20, "r-s2")]
    got = {x["job"]: x["places"][0]["slot"] if x["places"] else None for x in stranded.analyse(docs, T0, CPUS)["waits"]}
    assert got == {"r-s1": "t", "r-s2": None, "r-lane": None}, "the shard first, and t and c are one place"


def test_a_check_slot_is_no_place_while_the_machine_has_no_lean_check_slot_free():
    lean = {"file": "lean-slots", "pools": {"check": {"n": 1, "slots": [{"index": 0, "past_n": False, "holder": {"pid": 3}}],
                                                      "waiters": []}}}
    docs = {A: machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(),))),
            B: machine(lean=lean, check=pool(slot("t", "0-95", trains=True, mem=400)))}
    assert stranded_jobs(docs) == []
    lean["pools"]["check"]["slots"][0]["holder"] = None
    assert stranded_jobs(docs) == ["r-shard"]


def test_fourteen_jobs_and_one_place_is_one_stranded_job_the_oldest():
    line = tuple(waiter(m, f"r{m:02d}") for m in range(1, 15))
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=line)), B: machine(research=pool(slot("r0", "16-31"))),
            C: machine(research=pool(slot("r0", "16-31", held=True)))}
    doc = stranded.analyse(docs, T0, CPUS)
    assert [w["job"] for w in doc["waits"] if w["stranded"]] == ["r14"]
    assert doc["pools"]["research"]["in_line"] == 14 and doc["pools"]["research"]["stranded"] == 1
    assert stranded_jobs(docs, min_minutes=15) == [], "--min: only jobs that have waited that long"


def test_memory_decides_not_slots():
    """Ten idle slots beside memory for two runs are two places (`auto._pool` against `free_gb`)."""
    line = tuple(waiter(m, f"r{m:02d}", mem_gb=100) for m in range(1, 6))
    free = machine(research=pool(*(slot(f"r{i}", f"{16 * i}-{16 * i + 15}", mem=None) for i in range(10)), free_gb=200))
    docs = {A: machine(research=pool(slot("r0", "0-15", held=True), waiters=line)), B: free}
    assert stranded_jobs(docs) == ["r05", "r04"]


def test_twins_are_one_job_and_its_move_is_a_late_twin_where_the_machine_can_claim_on_its_home():
    home = machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(twin="p1", run="r-a"),)))
    away = machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(twin=f"p1@{A}", run="r-b"),)))
    free = machine(check=pool(slot("t", "0-95", trains=True, mem=400)))
    doc = stranded.analyse({A: home, B: away, C: free}, T0, CPUS)
    (w,) = doc["waits"]
    assert w["job"] == "p1" and w["twin"] == "p1" and w["runs"] == ["r-a", "r-b"] and [ln["machine"] for ln in w["lines"]] == [A, B]
    assert w["move"] == {"how": "late twin", "to": [A, B, C], "why_not": None}
    assert stranded.moves(doc) == [{"job": "p1", "pool": "check", "how": "late twin", "to": [A, B, C], "why_not": None}]
    short = {A: home, B: away, C: {**free, "peers": [B]}}
    (w,) = stranded.analyse(short, T0, CPUS)["waits"]
    assert w["stranded"] and w["move"]["how"] == "none" and f"names {A} in its /etc/vy/peers" in w["move"]["why_not"]
    (w,) = stranded.analyse({**short, C: {**free, "peers": None}}, T0, CPUS)["waits"]
    assert w["move"]["how"] == "none", "a machine with no peers file claims on no home"


def test_a_twin_left_in_line_while_its_pair_runs_is_no_wait_and_rings_no_one(tmp_path):
    """A machine with no /etc/vy/peers can't ask its twin's home, so its twin stays in line after the pair took a slot there:
    the job runs, so the line it is left in is listed apart, takes no place and rings no one."""
    took = slot("c", "0-95", held=True, trains=True, mem=300)
    took["holder"]["twin"] = "p1"
    home = machine(check=pool(took, slot("b", "96-191", trains=True, mem=300)))
    away = machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(twin=f"p1@{A}", run="r-b"),)))
    docs = {A: home, B: away | {"peers": None}, C: machine(check=pool(slot("t", "0-95", trains=True, mem=400)))}
    doc = stranded.analyse(docs, T0, CPUS)
    (w,) = doc["waits"]
    assert not w["stranded"] and w["places"] == [] and w["runs_on"] == [A]
    assert w["move"]["how"] == "none" and f"its pair runs on {A}" in w["move"]["why_not"]
    assert f"check  p1: in {B}'s for 40 min; its pair runs on {A}" in stranded.text(doc)
    assert doc["pools"]["check"]["stranded"] == 0 and stranded.moves(doc) == []
    assert [e for e in watch(tmp_path, [(T0 + t, docs) for t in (0, 600, 1200)]) if e[1].startswith("stranded|check|")] == []


def test_a_job_without_a_pair_moves_only_at_its_next_launch():
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=(waiter(30, "r1"),))), B: machine(research=pool(slot("r0")))}
    (w,) = stranded.analyse(docs, T0, CPUS)["waits"]
    assert w["move"] == {"how": "at launch", "to": [A, B], "why_not": None}
    assert "move: at launch to vy-nebius-cpu-1, vy-nebius-cpu-2" in stranded.text(stranded.analyse(docs, T0, CPUS), True)


def test_a_stalled_line_is_printed_with_its_memory_and_is_no_stranded_wait():
    """auto's own Stalled: a waiter 5 minutes beside an idle slot, for more memory than is left unreserved."""
    line = pool(slot("r0", "0-15"), slot("r1", "16-31", held=True), waiters=(waiter(12, "r-big", mem_gb=300),), free_gb=50)
    doc = stranded.analyse({A: machine(research=line)}, T0, CPUS)
    assert [w["stranded"] for w in doc["waits"]] == [False]
    assert doc["stalled"] == [{"machine": A, "pool": "research", "in_line": 1, "head_minutes": 12, "available_gb": 500.0,
                               "free_gb": 50, "unreached_gb": 450.0}]
    assert "reserved and not reached 450 GB" in stranded.text(doc)
    empty = pool(slot("r0", "0-15", blocked=MEM_BLOCK), free_gb=50)
    assert stranded.analyse({A: machine(research=empty)}, T0, CPUS)["stalled"] == [], "a pool no one waits in is no stalled line"


def test_a_window_is_printed_with_its_end_and_note_and_is_no_place():
    win = slot("r0", "16-31", blocked="window 2026-11-12T19:00Z (60 min) is open or starts within 30 min",
               upcoming=({"start": "2026-11-12T19:00Z", "minutes": 60, "starts_in_min": 10, "note": "lean bench"},))
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=(waiter(30, "r1"),))), B: machine(research=pool(win))}
    doc = stranded.analyse(docs, T0, CPUS)
    assert stranded_jobs(docs) == [] and doc["pools"]["research"]["blocked"] == {"window": 1}
    assert doc["windows"] == [{"machine": B, "pool": "research", "slot": "r0", "start": "2026-11-12T19:00Z",
                               "end": "2026-11-12T20:00Z", "note": "lean bench"}]
    assert f"{B}: 2026-11-12T19:00Z to 2026-11-12T20:00Z: lean bench; closes research r0" in stranded.text(doc)
    assert stranded._end("soon", 60) == "? (its start 'soon' is no ISO time)", "an end it can't work out says why"


def test_an_unread_machine_and_a_pool_that_answered_late_are_listed_and_the_rest_is_read():
    late = machine(research={"error": "not read in 20 s"}, proofs=pool(slot("p0"), waiters=(waiter(3, "r-p"),)))
    doc = stranded.analyse({A: late, B: f"{auto.NO_SSH} (Connection timed out)"}, T0, CPUS)
    assert doc["unread"] == [{"machine": A, "pool": "research", "why": "not read in 20 s"},
                             {"machine": B, "pool": None, "why": f"{auto.NO_SSH} (Connection timed out)"}]
    assert doc["pools"]["proofs"]["in_line"] == 1 and doc["pools"]["proofs"]["unread"] == [B]
    out = stranded.text(doc)
    assert "2 not read, so every count is a floor" in out and f"{A} research: not read in 20 s" in out


def test_a_pool_under_a_disk_floor_is_shut_and_the_check_pool_is_read_as_slot_py_words_it():
    floor = "/workspace has 812 inodes free, under 1000000"
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=(waiter(30, "r1"),))),
            B: machine(floor=floor, research=pool(slot("r0", "16-31")), check=pool(slot("t", "32-63", trains=True, mem=400)))}
    doc = stranded.analyse(docs, T0, CPUS)
    assert stranded_jobs(docs) == [] and doc["shut"] == [{"machine": B, "pool": "research", "why": f"disk: {floor}"}]
    assert doc["pools"]["check"]["places"] == 1, "slot.py blocks the check slots itself under its own floor"


def test_a_jobs_auto_words_are_printed_and_a_job_with_no_auto_key_is_not_recorded_as_placed():
    said = f"{B}'s research pool has the most places for it (3)"
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=(waiter(30, "r1", said=said), waiter(20, "r2")))),
            B: machine(research=pool(slot("r0", "16-31"), slot("r1", "32-47")))}
    doc = stranded.analyse(docs, T0, CPUS)
    assert [w["placed"] for w in doc["waits"]] == [{"by": "auto", "said": said}, {"by": None, "said": None}]
    out = stranded.text(doc)
    assert f"placed by --on auto: {said}" in out and "placed: not recorded (its request has no auto key)" in out
    assert "its lane" not in out


def test_the_peers_a_machine_lacks_are_listed():
    doc = stranded.analyse({A: machine(peers=[B]), B: machine(peers=None) | {"peers": None}, C: machine()}, T0, CPUS)
    assert doc["peers"] == [{"machine": A, "file": True, "missing": [C]}, {"machine": B, "file": False, "missing": [A, C]}]


# ---- the read ---------------------------------------------------------------------------------------------------------------

def test_auto_probe_still_has_the_line_the_program_is_cut_at():
    assert stranded.MARK in auto.PROBE, ("auto.PROBE lost its `out = {}` line, where stranded.program cuts it: change "
                                         "stranded.MARK to the line that now ends auto.PROBE's functions")
    compile(stranded.program().decode(), "stranded-program", "exec")


def test_a_tree_without_slot_py_says_how_to_add_it(tmp_path):
    with pytest.raises(RuntimeError, match="add `tools/verity/check/slot.py` and `tools/verity/check/lean_slot.py` to the deploy's"):
        stranded.program(tmp_path / "nowhere")


SLOT_STUB = '''
import json, os
from pathlib import Path
SLOTS = Path(os.environ.get("CHECK_SLOTS", "none"))
def status_doc(now):
    if SLOTS.parent.name == "broken":
        raise OSError("its slots file can't be read")
    return json.loads(SLOTS.read_text())
def memory_gb():
    return 300
def reserved_gb(priority=False, **kw):
    return 40
'''
LEAN_STUB = '''
def status_doc(now):
    return {"file": "lean-slots", "pools": {"audit": {"n": 1, "slots": [{"index": 0, "past_n": False, "holder": {"pid": 9,
            "cmd": "lake build --token=%s"}}], "waiters": [{"pid": 10, "minutes": 3, "run": "r-lean"}]}}}
''' % SECRET


def test_the_node_program_reads_every_pool_drops_every_cmd_and_reads_only_the_auto_key_of_a_request(tmp_path, monkeypatch, capsys):
    """The program each machine runs, run here on stand-ins of slot.py and lean_slot.py: no command line leaves the machine
    (a holder's `cmd`, a request's argv and env), and a pool that can't be read is listed with why beside the others."""
    tree, root = tmp_path / "tree", tmp_path / "root"
    for rel, text in ((stranded.SLOT_PY, SLOT_STUB), (stranded.LEAN_SLOT_PY, LEAN_STUB)):
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_text(text)
    locks = root / "locks"
    (locks / "research").mkdir(parents=True)
    (locks / "broken").mkdir()
    (locks / "broken" / "slots").write_text("")
    check = pool(slot("t", "0-95", held=True, trains=True), waiters=(shard(),))
    (locks / "slots").write_text(json.dumps(check))
    (locks / "research" / "slots").write_text(json.dumps(pool(slot("r0", held=True), waiters=(waiter(30, "r1"),))))
    (root / "requests" / "r1").mkdir(parents=True)
    (root / "requests" / "r1" / "request.json").write_text(json.dumps({
        "argv": ["bash", "-lc", f"TOKEN={SECRET} make"], "cmd": SECRET, "env": {"K": SECRET},
        "auto": {"machine": A, "pool": "research", "said": "the most places", "cmd": SECRET}}))
    (tmp_path / "peers").write_text(f"# who\n{B} -i KEY research@10.0.0.2\n{C} -i KEY research@10.0.0.3\nlonely\n")
    monkeypatch.setenv("CHECK_SLOTS", "unset")
    src = stranded.program(tree, root=str(root), peers=str(tmp_path / "peers"), disk=str(tmp_path / "no-disk")).decode()
    exec(compile(src, "stranded-program", "exec"), {"__name__": "__main__"})  # noqa: S102  (the program each machine runs)
    printed = capsys.readouterr().out
    assert SECRET not in printed and '"cmd"' not in printed
    doc = json.loads(printed)
    assert sorted(doc["pools"]) == ["broken", "check", "research"]
    assert doc["pools"]["broken"]["error"] == "OSError: its slots file can't be read"
    assert doc["pools"]["research"]["free_gb"] == 260 and doc["pools"]["research"]["slots"][0]["holder"] == {"pid": 7, "run": "r-holder"}
    assert doc["pools"]["research"]["waiters"][0]["auto"] == {"machine": A, "pool": "research", "said": "the most places"}
    assert "auto" not in doc["pools"]["check"]["waiters"][0], "a waiter whose run has no request keeps no auto key"
    assert doc["peers"] == [B, C] and doc["floor"] is None and doc["lean"]["pools"]["audit"]["waiters"][0]["run"] == "r-lean"
    analysed = stranded.analyse({A: doc}, T0, CPUS)
    out = json.dumps(analysed) + stranded.text(analysed, True)
    assert SECRET not in out and '"cmd"' not in out
    assert analysed["lean_waits"] == [{"machine": A, "pool": "audit", "job": "r-lean", "minutes": 3}]


def test_read_lists_each_machine_it_couldnt_read_with_why():
    def run(argv, data):
        if argv[0] == A:
            return subprocess.CompletedProcess(argv, 0, json.dumps(machine()).encode(), b"")
        if argv[0] == B:
            return subprocess.CompletedProcess(argv, 255, b"", b"ssh: connect to host 10.0.0.2 port 22: Connection refused")
        raise subprocess.TimeoutExpired(argv, 60)

    docs = stranded.read(CPUS, run=run)
    assert docs[A]["available_gb"] == 500.0
    assert docs[B] == f"{auto.NO_SSH} (Connection refused)" and docs[C] == "no answer in 60 s"


# ---- the command ------------------------------------------------------------------------------------------------------------

def test_the_command_exits_1_with_a_stranded_wait_and_0_without(monkeypatch, capsys):
    docs = {A: machine(research=pool(slot("r0", held=True), waiters=(waiter(30, "r1"),))), B: machine(research=pool(slot("r0")))}
    monkeypatch.setattr(stranded, "read", lambda names: docs)
    assert stranded.main(["--json"], now=lambda: T0) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema"] == stranded.SCHEMA and [w["job"] for w in doc["waits"]] == ["r1"]
    assert stranded.main(["--pool", "check"], now=lambda: T0) == 0
    assert "0 stranded of 0 in line" in capsys.readouterr().out
    assert stranded.main(["--min", "31"], now=lambda: T0) == 0


# ---- the watch --------------------------------------------------------------------------------------------------------------

def events(alerts) -> list[tuple[str, str, list[str]]]:
    return [(e["event"], e["key"], e["owners"]) for e in map(json.loads, (alerts / "monitors.jsonl").read_text().splitlines())]


def watch(tmp_path, passes: list[tuple[float, dict]]) -> list[tuple[str, str, list[str]]]:
    """The monitor `stranded` over passes at the given times, each reading the given machines' documents."""
    m, alerts, st = M.BY_ID["stranded"], tmp_path / "alerts", M.new_state()
    assert (m.owner, m.where, m.every_s, m.clear_s) == ("infra", "relay", 120, 300)
    for now, docs in passes:
        def run(argv, input=None, capture_output=True, timeout=None, docs=docs):
            d = docs[argv[0]]
            return subprocess.CompletedProcess(argv, 0 if isinstance(d, dict) else 255, json.dumps(d).encode(), str(d).encode())
        M.step(st, [m], lambda memo, now=now, run=run: M.Ctx(now=now, memo=memo, run=run), now, M.AlertDir(alerts), by="test",
               force=True)
    return events(alerts) if (alerts / "monitors.jsonl").exists() else []


def quiet(**over) -> dict:
    return {A: machine(), B: machine(), C: machine()} | over


def test_a_check_job_stranded_ten_minutes_rings_ci_once_and_clears_when_it_has_left_the_line(tmp_path):
    waiting = quiet(**{A: machine(check=pool(slot("t", "0-95", held=True, trains=True, mem=520), waiters=(shard(),))),
                       B: machine(check=pool(slot("t", "0-95", trains=True, mem=400)))})
    gone = quiet(**{B: machine(check=pool(slot("t", "0-95", trains=True, mem=400)))})
    assert watch(tmp_path / "a", [(T0, waiting), (T0 + 480, waiting)]) == [], "not before 10 minutes"
    got = watch(tmp_path / "b", [(T0 + t, waiting) for t in (0, 120, 600, 720)] + [(T0 + 840, gone), (T0 + 1140, gone)])
    assert got == [("open", "stranded|check|r-shard", ["ci"]), ("clear", "stranded|check|r-shard", ["ci"])]


def test_a_lane_pool_rings_infra_once_a_pool_while_one_of_its_jobs_has_been_stranded_ten_minutes(tmp_path):
    line = tuple(waiter(m, f"r{m}") for m in (30, 20))
    waiting = quiet(**{A: machine(research=pool(slot("r0", held=True), waiters=line)),
                       B: machine(research=pool(slot("r0", "16-31"), slot("r1", "32-47")))})
    assert watch(tmp_path / "a", [(T0, waiting), (T0 + 480, waiting)]) == []
    got = watch(tmp_path / "b", [(T0, waiting), (T0 + 600, waiting), (T0 + 720, quiet()), (T0 + 1020, quiet())])
    assert got == [("open", "stranded|pool|research", ["infra"]), ("clear", "stranded|pool|research", ["infra"])]


def test_a_pool_shut_by_an_inode_floor_rings_infra_at_once_on_an_idle_machine_and_after_30_minutes_on_a_busy_one(tmp_path):
    floor = "/workspace has 812 inodes free, under 1000000"
    elsewhere = machine(research=pool(slot("r0", held=True), waiters=(waiter(3, "r1"),)))
    idle = quiet(**{A: elsewhere, B: machine(floor=floor, research=pool(slot("r0", "16-31")), proofs=pool(slot("p0", "32-47")))})
    assert watch(tmp_path / "a", [(T0, idle)]) == [("open", f"stranded|shut|{B}", ["infra"])], "one incident a machine, not a pool"
    busy = quiet(**{A: elsewhere, B: machine(floor=floor, research=pool(slot("r0", "16-31"), slot("r1", "32-47", held=True)))})
    assert watch(tmp_path / "b", [(T0, busy), (T0 + 1200, busy)]) == []
    assert watch(tmp_path / "c", [(T0, busy), (T0 + 1800, busy)]) == [("open", f"stranded|shut|{B}", ["infra"])]


def test_infra_hears_of_a_line_stalled_30_minutes_a_lean_wait_of_10_and_a_short_peers_file(tmp_path):
    stall = pool(slot("r0", "0-15"), slot("r1", "16-31", held=True), waiters=(waiter(12, "r-big", mem_gb=300),), free_gb=50)
    lean = {"file": "lean-slots", "pools": {"audit": {"n": 1, "slots": [], "waiters": [{"pid": 1, "minutes": 11, "run": "r-l"}]}}}
    docs = quiet(**{A: machine(research=stall), B: machine(lean=lean), C: machine(peers=[A])})
    got = watch(tmp_path, [(T0, docs), (T0 + 1800, docs)])
    assert sorted(got) == [("open", "stranded|lean|" + B, ["infra"]), ("open", "stranded|peers", ["infra"]),
                           ("open", "stranded|stalled|research", ["infra"])]


def test_every_machine_short_of_peers_is_one_finding_naming_the_ones_with_no_file_and_how_many_the_others_lack():
    docs = quiet(**{A: machine(peers=[B]), B: machine(peers=None) | {"peers": None}, C: machine(peers=[])})
    ctx = M.Ctx(now=T0, run=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, json.dumps(docs[argv[0]]).encode(), b""))
    (f,) = [f for f in stranded.findings(ctx) if f.key.startswith("peers")]
    assert f.key == "peers" and f.owners == ("infra",) and f.for_s == 0
    assert f.text.startswith("*3 CPU machine(s) have a missing or short /etc/vy/peers*: ")
    assert f"no file on {B}; short on {A} (lacks 1), {C} (lacks 2)." in f.text


def test_the_watch_raises_with_the_fix_only_when_no_machine_could_be_read(tmp_path):
    down = {m: f"{auto.NO_SSH} (Connection timed out)" for m in CPUS}
    ctx = M.Ctx(now=T0, run=lambda argv, **kw: subprocess.CompletedProcess(argv, 255, b"", down[argv[0]].encode()))
    with pytest.raises(RuntimeError, match="no CPU machine could be read"):
        stranded.findings(ctx)
    one_down = quiet(**{C: down[C]})
    ctx = M.Ctx(now=T0, run=lambda argv, **kw: subprocess.CompletedProcess(argv, 0, json.dumps(one_down[argv[0]]).encode(), b"")
                if isinstance(one_down[argv[0]], dict) else subprocess.CompletedProcess(argv, 1, b"", b"down"))
    assert [f.key for f in stranded.findings(ctx)] == [f"shut|{C}|*"], "one machine not read is a shut finding, never a raise"
