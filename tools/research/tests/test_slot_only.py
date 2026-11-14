"""pods/nebius/slot_only.py: on a machine that takes runs only through its check slots, which commands count, and which live runs it
stops (a run outside slot.py beside the merge check, such as one from a tree older than the launcher's refusal); and a machine's
own slot.py, which a run that names a pool takes its slot by."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from research import remote, run_records
from research.pods.nebius import slot_only as S

NEBIUS = Path(S.__file__).resolve().parent
REPO = NEBIUS.parents[5]


def test_only_a_command_that_runs_cmd_in_a_check_slot_counts():
    assert S.through_slot(["python3", "tools/verity/check/slot.py", "--", "uv", "run", "python", "tools/verity/check/check.py"])
    assert S.through_slot(["python3", "tools/verity/check/slot.py", "--priority", "--by", "x", "--", "true"])
    for c in (["python3", "tools/verity/check/slot.py", "--status"], ["python3", "tools/verity/check/slot.py", "--by", "x", "--"],
              ["uv", "run", "circuit-check", "--all"], ["bash", "-c", "python3 tools/verity/check/slot.py -- true"],
              ["python", "tools/verity/check/slot.py", "--", "true"], ["python3", "/elsewhere/slot.py", "--", "true"], None, "python3"):
        assert not S.through_slot(c), c


def test_a_run_may_name_a_pool_of_its_own_under_the_locks_directory_and_nothing_else():
    pool = ["env", "CHECK_SLOTS=/workspace/research/locks/proofs/slots"]
    assert S.through_slot([*pool, "python3", "tools/verity/check/slot.py", "--by", "proofs", "--", "nice", "-n", "19", "bash", "j.sh"])
    for env in (["env", "CHECK_SLOTS=/tmp/slots"], ["env", "CHECK_SLOTS=/workspace/research/locks/../x/slots"],
                ["env", "CHECK_SLOTS=/workspace/research/locks/proofs/slots", "X=1"], ["env", "LEASE_DIR=/x"]):
        assert not S.through_slot([*env, "python3", "tools/verity/check/slot.py", "--", "true"]), env
    assert not S.through_slot([*pool, "python3", "tools/verity/check/slot.py", "--status"])


def test_the_refusal_names_the_pools_this_machine_has(tmp_path):
    """A `verity lean try --on vy-nebius-1` without `--pool` is refused naming the pools to add, not only slot.py by hand."""
    (tmp_path / "proofs").write_text("p0 0-15\n")
    why = S.refusal(tmp_path / "marker", {"proofs": str(tmp_path / "proofs"), "soundness": str(tmp_path / "soundness")})
    assert "add `--pool proofs` to `research run` or `verity lean try`" in why
    assert "slot.py [--by NAME] -- CMD" in S.refusal(tmp_path / "marker", {"soundness": str(tmp_path / "soundness")})


def test_research_run_pool_wraps_the_command_in_a_named_pool_that_the_launcher_accepts(capsys):
    """A lane's run on cpu-1 takes the `research` pool with one flag, not a five-layer command line."""
    c = S.in_pool("research", "compute-accounting", ["python3", "search.py", "--n", "4"])
    assert c == ["env", "CHECK_SLOTS=/workspace/research/locks/research/slots", "python3", "tools/verity/check/slot.py", "--by",
                 "compute-accounting", "--", "nice", "-n", "19", "python3", "search.py", "--n", "4"]
    assert S.through_slot(c) and all(S.through_slot(S.in_pool(p, "x", ["true"])) for p in S.POOLS)
    leanrec = S.in_pool("leanrec", "lean", ["true"])
    assert leanrec[1] == "CHECK_SLOTS=/workspace/research/locks/leanrec/slots" and S.through_slot(leanrec), "a node's own pool"
    for bad in ("../x", "a b", "", "x/y"):
        try:
            S.in_pool(bad, "x", ["true"])
            raise AssertionError(f"pool {bad!r} must be refused")
        except ValueError as e:
            assert "a pool is a name of letters, digits" in str(e)
    from research.cli import main
    for argv, why in ((["--pool", "research", "--", "true"], "needs --on MACHINE and --source"),
                      (["--on", "m", "--source", ".", "--pool", "research", "--", "true"], "needs a lane"),
                      (["--on", "m", "--source", ".", "--pool", "research", "--as", "x", "--cwd", "clone/sub", "--", "true"], "not --cwd clone/sub"),
                      (["--on", "m", "--source", ".", "--pool", "research", "--as", "x", "--cwd", "/tmp", "--", "true"], "not --cwd /tmp"),
                      (["--on", "m", "--source", ".", "--pool", "../gpu", "--as", "x", "--", "true"], "no pool '../gpu'")):
        assert main(["run", *argv]) == 2 and why in capsys.readouterr().err, argv
    # a job that writes into its tree (a Lean check rewriting a lock) runs in a private clone, so the shared tree stays clean
    main(["run", "--on", "m", "--source", ".", "--pool", "proofs", "--as", "x", "--cwd", "clone", "--", "true"])
    assert "--pool" not in capsys.readouterr().err


def test_a_pool_run_takes_no_launch_time_memory_reservation_beside_its_slots(tmp_path, monkeypatch, capsys):
    """A `--pool` run's --mem-gb reserved at launch would be held through its wait in the slot line, beside the slot's own
    reservation once it ran, so a few waiting runs drive admission below zero. The slot admits it, with the run's --mem-gb when
    that is more than its mem= (a 450 GB run in a 193 GB slot reserves 450, not 193)."""
    from research import remote
    from research.cli import main

    seen = []

    def launch(**kw):
        seen.append((kw["mem_gb"], kw["env"].get("CHECK_RUN_MEM_GB")))
        return {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r", "launch": {"runner_pid": 1}}

    monkeypatch.setattr(remote, "launch", launch)
    argv = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--no-custody-r2", "--source", ".",
            "--pool", "proofs", "--as", "proofs"]
    assert main([*argv, "--mem-gb", "96", "--", "true"]) == 0 and main([*argv, "--", "true"]) == 0
    assert seen == [(None, "96"), (None, None)] and "--mem-gb 96 goes to the slot" in capsys.readouterr().err


def test_a_pool_runs_timeout_counts_from_taking_its_slot_not_from_its_wait(monkeypatch, tmp_path, capsys):
    """A `--pool` run's --timeout must not count its wait in the pool's line too, or a run that waited 50 of its 60 min runs 10."""
    from research import remote
    from research.cli import main

    assert S.in_pool("research", "x", ["true"], 2400)[4:11] == ["--by", "x", "--wait-max", f"{S.POOL_WAIT_S}", "--run-max", "2400", "--"]
    assert S.through_slot(S.in_pool("research", "x", ["true"], 2400))
    seen = []
    monkeypatch.setattr(remote, "launch", lambda **kw: seen.append(kw) or {"phase": "launched", "pod_id": "p1", "remote_run_dir": "/r",
                                                                            "launch": {"runner_pid": 1}})
    argv = ["run", "--runs-dir", str(tmp_path / "runs"), "--on", "m", "--project", "p", "--no-custody-r2", "--source", ".",
            "--pool", "research", "--as", "infra"]
    assert main([*argv, "--timeout", "40m", "--", "true"]) == 0 and main([*argv, "--", "true"]) == 0
    timed, untimed = seen
    assert timed["command"][-7:] == ["--run-max", "2400", "--", "nice", "-n", "19", "true"]
    assert remote.flag_value(timed["harness_args"], "--timeout") == f"{S.POOL_WAIT_S + 2400}", "the runner's covers the wait and the run"
    assert "--timeout 2400 s counts from taking a slot (slot.py --run-max)" in capsys.readouterr().err
    assert "--run-max" not in untimed["command"] and remote.flag_value(untimed["harness_args"], "--timeout") is None
    assert S.POOL_WAIT_S >= 4 * 3600, "an hour's wait cost proofs' runs their place in a saturated pool each hour (INF-153)"
    assert main([*argv, "--timeout", "40m", "--pool-wait", "10800", "--", "true"]) == 0
    waited = seen[-1]
    assert waited["command"][4:11] == ["--by", "infra", "--wait-max", "10800", "--run-max", "2400", "--"]
    assert remote.flag_value(waited["harness_args"], "--timeout") == f"{10800 + 2400}"


def test_a_command_that_names_a_pool_and_the_trees_slot_py_names_the_machines_own_in_its_place():
    """`research run --pool`'s command, and any other built to the same words (a quick tier placed in a pool, a benchmark's):
    its waiter stands in a pool's line, so it goes by the machine's copy. A recorded check names its pool in the run's
    environment and none in its command, and keeps its tree's slot.py. Every reader of a run's command still knows it."""
    pool = S.in_pool("research", "lane", ["python3", "search.py"], 2400)
    own = S.node_copy(pool)
    assert own == [*pool[:3], S.NODE_SLOT, *pool[4:]] and own[4:] == pool[4:]
    assert S.through_slot(own) and S.node_copy(own) is None, "already the machine's"
    by_hand = ["env", f"CHECK_SLOTS={S.POOLS['proofs']}", "python3", S.SLOT_PATHS[0], "--wait-max", "60", "--", "sh", "-c", "x"]
    assert S.node_copy(by_hand)[3] == S.NODE_SLOT
    for other in (["python3", S.SLOT_PATHS[0], "--", "uv", "run", "python", "tools/verity/check/check.py"],
                  ["python3", S.SLOT_PATHS[0], "--priority", "--rank", "3", "--", "true"],
                  ["env", f"CHECK_SLOTS={S.POOLS['research']}", "python3", S.SLOT_PATHS[0], "--status"],
                  ["env", "CHECK_SLOTS=/tmp/slots", "python3", S.SLOT_PATHS[0], "--", "true"],
                  ["env", f"CHECK_SLOTS={S.POOLS['research']}", "python3", "elsewhere/slot.py", "--", "true"], ["true"], []):
        assert S.node_copy(other) is None, other
    assert S.through_slot(["python3", S.NODE_SLOT, "--", "true"]) and not S.through_slot(["python3", S.NODE_SLOT, "--status"])
    assert S.NODE_SLOT.startswith("/") and S.NODE_SLOT.endswith("/" + S.SLOT_PATHS[0]), "below slot.py's path in the repository"
    assert run_records.pool(own) == "research" and run_records.pool(["python3", S.NODE_SLOT, "--", "true"]) == run_records.OWN_POOL


@pytest.fixture
def node(tmp_path, monkeypatch):
    """A machine under tmp_path: where its own slot.py, its rule for a run's command (slot_only.py) and its marker would be.
    ``node.install(text=None)`` puts a slot.py there (the repository's), ``node.rule(knows)`` a rule that knows the copy or
    predates it, ``node.takes(command)`` runs the launch's test of the machine here."""
    class Node:
        copy = tmp_path / "lib" / "check-slot" / S.SLOT_PATHS[0]
        marker = tmp_path / "etc" / "runs-via-slot"
        rules = tmp_path / "lib" / "slot_only.py"

        def install(self, text: str | None = None) -> None:
            self.copy.parent.mkdir(parents=True, exist_ok=True)
            self.copy.write_text((REPO / S.SLOT_PATHS[0]).read_text() if text is None else text)

        def rule(self, knows: bool) -> None:
            text = Path(S.__file__).read_text()
            assert text.count(REAL) == 1, "the deployed rule names the copy's path whole: the launch's test looks for it there"
            self.rules.write_text(text.replace(REAL, str(self.copy) if knows else "/usr/local/lib/elsewhere/slot.py"))

        def asked(self, command: list[str]) -> tuple[int, str]:
            r = subprocess.run(["sh", "-c", remote.node_slot_test(command)], capture_output=True, cwd=tmp_path, text=True)
            return r.returncode, r.stdout.strip()

        def package(self, name: str, asks_the_machines_rule: bool, knows: bool) -> Path:
            """A research package as a run's tree ships it: from before or after the launch-request asked the machine's own
            rule, and from before or after its own rule knew the machine's slot.py."""
            pkg = tmp_path / name
            (pkg / "pods" / "nebius").mkdir(parents=True)
            (pkg / "remote.py").write_text("def node_slot_only():\n    pass\n" if asks_the_machines_rule else "")
            (pkg / "pods" / "nebius" / "slot_only.py").write_text(f"NODE_SLOT = {str(self.copy)!r}\n" if knows else "")
            return pkg

    REAL = S.NODE_SLOT
    n = Node()
    n.marker.parent.mkdir()
    monkeypatch.setattr(S, "NODE_SLOT", str(n.copy))
    monkeypatch.setattr(S, "SLOT_PATHS", (S.SLOT_PATHS[0], str(n.copy)))
    monkeypatch.setattr(remote, "RUNS_VIA_SLOT_FILE", str(n.marker))
    monkeypatch.setattr(remote, "NODE_SLOT_ONLY_FILE", str(n.rules))
    return n


def test_a_machines_own_slot_py_takes_a_run_only_if_it_is_there_and_knows_the_runs_options(node, tmp_path):
    """The launch asks the machine before it names the machine's copy. A copy from before an option was added would exit
    with its usage, so the run keeps its tree's slot.py. Where the copy takes the run, the machine says who reads a run's
    command there."""
    plain = S.in_pool("research", "lane", ["true"])
    timed = S.in_pool("research", "lane", ["true"], 2400)
    budget = [*timed[:timed.index("--")], "--budget-gb", "64", *timed[timed.index("--"):]]
    assert [node.asked(c) for c in (plain, timed, budget)] == [(1, "")] * 3, "no copy there"
    node.install()
    assert [node.asked(c) for c in (plain, timed, budget)] == [(0, "open none")] * 3
    node.install((REPO / S.SLOT_PATHS[0]).read_text().replace("--budget-gb", "--budget"))
    assert [node.asked(c)[0] for c in (plain, timed, budget)] == [0, 0, 1], "a copy from before --budget-gb takes no run that gives it"
    node.install()
    node.marker.write_text("")
    assert node.asked(timed) == (0, "marked none")
    node.rule(knows=False)
    assert node.asked(timed) == (0, "marked lacks")
    node.marker.unlink()
    assert node.asked(timed) == (0, "open lacks")
    node.rule(knows=True)
    assert node.asked(timed) == (0, "open names")
    node.marker.write_text("")
    assert node.asked(timed) == (0, "marked names")
    # a word of the command reaches the machine's shell quoted, whatever it holds
    odd = [*plain[:4], "--by", "lane", "--x'; touch pwned; echo '", "--", "true"]
    assert S.node_copy(odd) is not None and node.asked(odd) == (1, "") and not (tmp_path / "pwned").exists()


def test_a_run_waits_by_the_machines_slot_py_only_where_whatever_reads_its_command_there_knows_that_copy(node):
    """A rule from before the copy takes a run through it for one outside a check slot. The machine's own rule is what its
    timer stops or renices such a run by; and on a machine that starts a run only through a check slot, the research package
    that runs there refuses the run by the machine's rule where it asks it, else by its own."""
    ours = None                                                    # no package shipped: this one runs there
    new = node.package("new", asks_the_machines_rule=True, knows=True)
    asks = node.package("asks", asks_the_machines_rule=True, knows=False)
    old = node.package("old", asks_the_machines_rule=False, knows=False)
    predates = "has a slot_only.py that predates its own slot.py, and would take this run for one outside a check slot"
    refuses = ("starts a run only through a check slot, and the research package this run ships predates the machine's own "
               "slot.py: it would refuse this run")
    for pkg in (ours, new, asks, old):
        assert remote.node_slot_unruled("open none\n", pkg) is None, "nothing there reads a run's command"
        assert remote.node_slot_unruled("open names\n", pkg) is None, "its timer knows the copy, and nothing refuses a run"
        assert remote.node_slot_unruled("open lacks\n", pkg) == predates
        assert remote.node_slot_unruled("marked lacks\n", pkg) == predates
    for pkg in (ours, new):
        assert remote.node_slot_unruled("marked none\n", pkg) is None, "the package's own rule judges, and knows the copy"
        assert remote.node_slot_unruled("marked names\n", pkg) is None
    assert remote.node_slot_unruled("marked names\n", asks) is None, "it asks the machine's rule, which knows the copy"
    assert remote.node_slot_unruled("marked none\n", asks) == refuses, "no rule there to ask: its own, which predates the copy"
    assert remote.node_slot_unruled("marked names\n", old) == refuses, "it asks no rule but its own"
    assert remote.node_slot_unruled("marked none\n", old) == refuses
    assert remote.node_slot_unruled("marked names\n", node.copy.parent / "no-such-package") == refuses, "a package unread knows nothing"
    for odd in ("", "marked", "open names more", "motd: welcome\nopen names"):
        assert remote.node_slot_unruled(odd, ours) == f"answered {odd.strip()[:80]!r} when asked whether its own slot.py takes this run"


def _launch(tmp_path, monkeypatch, command: list[str], ssh=None) -> tuple[dict, dict]:
    """`remote.launch` of ``command`` on a machine that is this one (each ssh runs here) up to the staged request; the
    request and the launch's remote.json."""
    monkeypatch.setattr(remote, "resolve_machine", lambda name, project: {"root": "/r", "python": "python3"})
    monkeypatch.setattr(remote, "pod_target", lambda m: {"host": "h", "port": 22, "user": "research", "pod_name": "n1",
                                                         "cost_per_hr": 0, "rate_observed_utc": "t"})
    monkeypatch.setattr(remote, "wake", lambda m, ssh: None)
    monkeypatch.setattr(remote, "ship_tool", lambda *a: "/r/tool/x")
    monkeypatch.setattr(remote, "ssh_run", ssh or (lambda ssh, argv, **kw: subprocess.run(argv, capture_output=True, cwd=tmp_path)))
    staged = []

    def stage(ssh, root, run_id, request, sends, private=None):
        staged.append(request)
        raise remote.Refused("staged")

    monkeypatch.setattr(remote, "stage_request", stage)
    local = tmp_path / "runs" / f"r{len(list((tmp_path / 'runs').glob('*'))) if (tmp_path / 'runs').is_dir() else 0}"
    with pytest.raises(remote.Refused, match="staged"):
        remote.launch(run_id=local.name, local_dir=local, machine_name="vy-nebius-cpu-9", project="verity", command=command,
                      harness_args=[], source=None, sends=[], cwd=None, env={}, exclusive=False)
    return staged[0], json.loads((local / "remote.json").read_text())


def test_a_pool_run_is_sent_by_the_machines_own_slot_py_where_it_has_one_and_says_so_in_a_line_where_it_has_none(
        node, tmp_path, monkeypatch, capsys):
    """One copy's rules for every waiter of a pool's line, whatever tree each run ships: the launch names the machine's
    copy where its test of the machine passes, and its record says which copy the command names. Where the machine has none
    the command is the tree's, and the launch says so once. A command that names no pool is sent as it is, unasked."""
    pool = S.in_pool("research", "lane", ["python3", "search.py"], 2400)
    node.install()
    request, rec = _launch(tmp_path, monkeypatch, pool)
    assert request["command"] == [*pool[:3], str(node.copy), *pool[4:]] == rec["command"] and rec["slot_copy"] == "machine"
    assert "slot.py" not in capsys.readouterr().err

    node.copy.unlink()
    request, rec = _launch(tmp_path, monkeypatch, pool)
    assert request["command"] == pool == rec["command"] and rec["slot_copy"] == "tree"
    assert [ln for ln in capsys.readouterr().err.splitlines() if "slot.py" in ln] == [
        f"research: vy-nebius-cpu-9 has no slot.py of its own that takes this run ({node.copy}): the run waits in its pool's line "
        f"by its tree's {S.SLOT_PATHS[0]} (`research deploy drift --on vy-nebius-cpu-9` lists what the machine has)"]

    node.install()
    lost = subprocess.CompletedProcess([], 255, b"", b"ssh: connect to host h port 22: Connection timed out")
    request, rec = _launch(tmp_path, monkeypatch, pool, ssh=lambda ssh, argv, **kw: lost)
    assert request["command"] == pool and rec["slot_copy"] == "tree"
    assert "vy-nebius-cpu-9 didn't say whether its own slot.py takes this run (exit 255)" in capsys.readouterr().err

    node.rule(knows=False)                                    # the machine's rule for a run's command predates its slot.py
    request, rec = _launch(tmp_path, monkeypatch, pool)
    assert request["command"] == pool == rec["command"] and rec["slot_copy"] == "tree"
    assert "vy-nebius-cpu-9 has a slot_only.py that predates its own slot.py" in capsys.readouterr().err
    node.rule(knows=True)
    node.marker.write_text("")                                # ... and a machine that starts a run only through a check slot
    request, rec = _launch(tmp_path, monkeypatch, pool)
    assert request["command"][3] == str(node.copy) and rec["slot_copy"] == "machine" and "slot.py" not in capsys.readouterr().err
    node.rules.unlink()
    node.marker.unlink()

    check = ["python3", S.SLOT_PATHS[0], "--", "uv", "run", "python", "tools/verity/check/check.py"]
    request, rec = _launch(tmp_path, monkeypatch, check, ssh=lambda *a, **kw: pytest.fail("a command that names no pool asks nothing"))
    assert request["command"] == check == rec["command"] and "slot_copy" not in rec and capsys.readouterr().err == ""


def test_deploy_installs_slot_py_whole_as_each_pool_machines_own_copy_where_slot_py_and_the_launch_look_for_it():
    """One entry installs slot.py, at the path the launch names and slot.py knows itself by, on both nodes and every CPU
    node, whose record is the one slot.py reads its commit from. The copy is one file, which one rename replaces whole:
    slot.py imports the standard library alone, and the one file it looks for beside itself (common.py, for the check
    cache's name) is never installed there, nor is anything else, so a waiter reads nothing an install replaces. A run
    that waits by the copy is its own process, which no install restarts."""
    manifest = tomllib.loads((NEBIUS / "deploy.toml").read_text())
    rows = [e for e in manifest["file"] if e.get("src") == S.SLOT_PATHS[0]]
    assert [e["path"] for e in rows] == [S.NODE_SLOT], "one copy a machine"
    beside = S.NODE_SLOT.rsplit("/", 4)[0] + "/"               # .../check-slot/, which holds the copy below its repository path
    assert beside.endswith("/check-slot/") and S.NODE_SLOT == beside + S.SLOT_PATHS[0]
    assert [e["path"] for e in [*manifest["file"], *manifest.get("tree", [])]
            if e["path"].startswith(beside) or beside.startswith(e["path"].rstrip("/") + "/")] == [S.NODE_SLOT], "and nothing beside it"
    (row,) = rows
    cpu = {n for n in manifest["nodes"] if re.fullmatch(r"vy-nebius-cpu-\d+", n)}
    assert set(row["nodes"]) == cpu | {"vy-nebius-1", "vy-nebius-2"} and len(row["nodes"]) == len(set(row["nodes"]))
    assert row["no_restart"] and "run" not in row and "hold" not in row
    spec = importlib.util.spec_from_file_location("check_slot_for_its_copy", REPO / S.SLOT_PATHS[0])
    slot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(slot)
    assert str(slot.NODE_COPY) == S.NODE_SLOT
    assert {manifest["nodes"][n]["record"] for n in row["nodes"]} == {str(slot.DEPLOY_RECORD)}
    tree = ast.parse((REPO / S.SLOT_PATHS[0]).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {
        (n.module or ".").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= set(sys.stdlib_module_names), sorted(imported - set(sys.stdlib_module_names))
    reads = {f.name: [ast.unparse(c) for c in ast.walk(f) if isinstance(c, ast.Call) and "__file__" in ast.unparse(c)]
             for f in tree.body if isinstance(f, ast.FunctionDef) and "__file__" in ast.unparse(f)}
    assert set(reads) == {"cache_disk", "copy"}, "slot.py names its own path to say which copy it is, and to look for common.py"
    assert any("with_name('common.py')" in c for c in reads["cache_disk"]) and not any("with_name" in c for c in reads["copy"])


def _run(runs, rid, state, command, *, pgid=None, runner=None):
    d = runs / rid
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({"state": state, "pgid": pgid}))
    (d / "job.json").write_text(json.dumps({"execution": {"command": command}}))
    if runner:
        (d / "launch.json").write_text(json.dumps({"runner_pid": runner}))
    return d


def _proc(proc, pid, pgid, run_dir=None):
    p = proc / str(pid)
    p.mkdir(parents=True)
    (p / "stat").write_text(f"{pid} (py thon) S 1 {pgid} {pgid} 0 -1\n")
    (p / "environ").write_bytes(b"PATH=/bin\0" + (f"RESEARCH_RUN_DIR={run_dir.resolve()}\0".encode() if run_dir else b""))


def test_it_stops_every_process_of_a_live_run_outside_the_slots_but_its_runner(tmp_path, monkeypatch, capsys):
    runs, proc, marker, log = tmp_path / "runs", tmp_path / "proc", tmp_path / "runs-via-slot", tmp_path / "slot-only.jsonl"
    slot = ["python3", "tools/verity/check/slot.py", "--", "true"]
    gate = _run(runs, "r-gate", "running", slot, pgid=10, runner=9)
    off = _run(runs, "r-off", "running", ["uv", "run", "circuit-check", "--all"], pgid=20, runner=19)
    _run(runs, "r-done", "done", ["true"], pgid=30)
    _proc(proc, 9, 9)
    _proc(proc, 10, 10, gate)
    _proc(proc, 19, 19, off)          # a runner never carries RESEARCH_RUN_DIR; were it to, it is still spared
    _proc(proc, 20, 20, off)
    _proc(proc, 21, 20, off)
    _proc(proc, 40, 40, off)          # left in another group (a lake build's setsid): stopped too
    _proc(proc, 30, 30)
    assert [o["run"] for o in S.offenders(runs)] == ["r-off"]
    sent = []
    monkeypatch.setattr(S.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(S, "stop", lambda o, dry_run=False, _stop=S.stop: _stop(o, proc=proc, kill=S.os.kill, dry_run=dry_run))
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log)]) == 0
    assert sent == [] and not log.exists(), "no marker: the machine takes any run"
    marker.write_text("")
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log), "--dry-run"]) == 0
    assert sent == [] and not log.exists() and '"r-off"' in capsys.readouterr().out
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log)]) == 0
    assert sent == [(20, signal.SIGTERM), (21, signal.SIGTERM), (40, signal.SIGTERM)]
    rec = json.loads(log.read_text())
    assert rec["run"] == "r-off" and rec["workload_group_seen"] and rec["sigterm"] == [20, 21, 40]


def test_renice_moves_a_direct_runs_processes_to_nice_19_once_and_leaves_slot_runs_alone(tmp_path, monkeypatch, capsys):
    """On node 1 lanes' direct runs must not take the cores the train checks need. `--renice 19` stops nothing: it renices every
    process of a live run outside the slots, its runner too, and logs a run only while it moved a process."""
    runs, proc, marker, log = tmp_path / "runs", tmp_path / "proc", tmp_path / "direct-runs-nice", tmp_path / "direct-nice.jsonl"
    gate = _run(runs, "r-gate", "running", ["python3", "tools/verity/check/slot.py", "--", "true"], pgid=10, runner=9)
    off = _run(runs, "r-off", "running", ["uv", "run", "pytest", "-x"], pgid=20, runner=19)
    for pid, run in ((10, gate), (19, off), (20, off), (21, off)):
        _proc(proc, pid, pid, run)
    nice = {10: 0, 19: 0, 20: 0, 21: 5}
    monkeypatch.setattr(S.os, "getpriority", lambda which, pid: nice[pid])
    monkeypatch.setattr(S.os, "setpriority", lambda which, pid, n: nice.__setitem__(pid, n))
    monkeypatch.setattr(S.os, "kill", lambda *a: (_ for _ in ()).throw(AssertionError("--renice stops nothing")))
    monkeypatch.setattr(S, "renice", lambda o, n, dry_run=False, _r=S.renice: _r(o, n, proc=proc, dry_run=dry_run))
    marker.write_text("")
    argv = ["--runs", str(runs), "--marker", str(marker), "--log", str(log), "--renice", "19"]
    assert S.main(argv) == 0
    assert nice == {10: 0, 19: 19, 20: 19, 21: 19}
    rec = json.loads(log.read_text())
    assert rec["run"] == "r-off" and rec["reniced"] == [19, 20, 21] and rec["nice"] == 19
    assert S.main(argv) == 0 and len(log.read_text().splitlines()) == 1, "a run already at 19 isn't logged again"


def test_an_auditor_host_stops_every_live_run_but_the_commands_its_marker_lists(tmp_path, monkeypatch):
    """On vy-challenger-1 a run launched from a tree older than the launcher's auditor refusal is stopped, unless its command
    is exactly a line of /etc/vy/auditor; slot.py's runs get no pass there."""
    runs, proc, marker, log = tmp_path / "runs", tmp_path / "proc", tmp_path / "auditor", tmp_path / "auditor-only.jsonl"
    ok = _run(runs, "r-ok", "running", ["bash", "verity/composite/device/live_challenger/run.sh"], pgid=10, runner=9)
    slot = _run(runs, "r-slot", "running", ["python3", "tools/verity/check/slot.py", "--", "true"], pgid=20, runner=19)
    extra = _run(runs, "r-extra", "running", ["bash", "verity/composite/device/live_challenger/run.sh", "x"], pgid=30, runner=29)
    for pid, run in ((10, ok), (20, slot), (30, extra)):
        _proc(proc, pid, pid, run)
    marker.write_text("# comment\n\nbash verity/composite/device/live_challenger/run.sh\n")
    assert S.auditor_commands(marker) == [["bash", "verity/composite/device/live_challenger/run.sh"]]
    assert [o["run"] for o in S.offenders(runs, lambda c: S.auditor_allows(c, marker))] == ["r-extra", "r-slot"]
    sent = []
    monkeypatch.setattr(S.os, "kill", lambda pid, sig: sent.append(pid))
    monkeypatch.setattr(S, "stop", lambda o, dry_run=False, why="", _stop=S.stop: _stop(o, proc=proc, kill=S.os.kill, dry_run=dry_run,
                                                                                       why=why))
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log), "--auditor"]) == 0
    assert sent == [30, 20]
    assert {json.loads(ln)["why"] for ln in log.read_text().splitlines()} == {f"not a command of the auditor host's {marker}"}
    marker.write_text("")
    assert S.auditor_commands(marker) == [] and not S.auditor_allows(["bash", "verity/composite/device/live_challenger/run.sh"], marker)
    assert S.auditor_commands(tmp_path / "absent") == []


def test_deploy_makes_the_challenger_and_verifier_hosts_auditor_hosts_that_take_no_research_run():
    """The deployed marker lists no command (the challenger runs only through vy-challenger-gate, the verifier through
    vy-verifier-gate) and goes, with the timer that enforces it, to the auditor hosts only, each host's gate to it alone (the
    challenger's and certifier's units also go to vy-auditor-2, which runs a certifier); vy-admin's sshd drop-in goes to them and
    to each CPU node that carries the lock-out, whose sshd lets only research log in without it; the nodes' shared peers file and
    the cluster's description name no auditor host."""
    import tomllib
    from pathlib import Path

    hosts = ["vy-challenger-1", "vy-auditor-1", "vy-auditor-2"]
    nebius = Path(S.__file__).resolve().parent
    assert (nebius / "auditor").read_text().startswith("#") and S.auditor_commands(nebius / "auditor") == []
    rows = tomllib.loads((nebius / "deploy.toml").read_text())["file"]
    on = {e["path"]: e["nodes"] for e in rows if e.get("src", "").endswith(("/auditor", "/vy-auditor-only.service",
                                                                            "/vy-auditor-only.timer", "/11-vy-admin.conf"))}
    locked_cpu = [n for e in rows if e["path"] == "/usr/local/sbin/vy-lockout" for n in e["nodes"] if n.startswith("vy-nebius-cpu-")]
    assert on == {**{p: hosts for p in ("/etc/vy/auditor", "/etc/systemd/system/vy-auditor-only.service",
                                        "/etc/systemd/system/vy-auditor-only.timer")},
                  "/etc/ssh/sshd_config.d/11-vy-admin.conf": hosts + locked_cpu}
    assert {e["path"]: e["nodes"] for e in rows if e.get("src", "").endswith("/peers-challenger")} == {
        "/etc/vy/peers": ["vy-challenger-1"]}
    assert "--auditor --marker /etc/vy/auditor" in (nebius / "vy-auditor-only.service").read_text()
    gate = {e["path"]: e["nodes"] for e in rows if "/pods/nebius/challenger/" in e.get("src", "")
            and not e["src"].endswith("/11-vy-admin.conf")}
    assert gate == {**{p: ["vy-challenger-1"] for p in ("/usr/local/sbin/vy-challenger-gate", "/usr/local/sbin/vy-challenger-setup",
                                                        "/usr/local/sbin/vy-challenger-pin")},
                    **{p: ["vy-challenger-1", "vy-auditor-2"] for p in ("/etc/systemd/system/vy-challenger@.service",
                                                                        "/etc/systemd/system/vy-certifier@.service")}}
    verifier = {e["path"]: e["nodes"] for e in rows if "/pods/nebius/verifier/" in e.get("src", "")}
    assert verifier == {p: ["vy-auditor-1", "vy-auditor-2"] for p in (
        "/usr/local/sbin/vy-verifier-gate", "/usr/local/sbin/vy-verifier-setup", "/usr/local/sbin/vy-verifier-pin",
        "/usr/local/sbin/vy-verifier-tunnel", "/etc/systemd/system/vy-verifier@.service",
        "/etc/systemd/system/vy-verifier-replay@.service", "/etc/systemd/system/vy-verifier-publish@.service",
        "/etc/ssh/sshd_config.d/12-vy-verifier.conf", "/etc/vy/verifier.json")}
    for unit in ("vy-challenger@.service", "vy-certifier@.service"):
        assert "User=auditor" in (nebius / "challenger" / unit).read_text()
    for unit in ("vy-verifier@.service", "vy-verifier-replay@.service", "vy-verifier-publish@.service"):
        assert "User=auditor" in (nebius / "verifier" / unit).read_text()
    for host in hosts:
        assert not [e["path"] for e in rows if e.get("src", "").endswith("/peers") and host in e["nodes"]]
    for name in ("vy-challenger", "vy-auditor"):
        assert name not in (nebius / "peers").read_text()
        assert name not in (nebius.parents[5] / "infra/cluster/descriptions/nebius.toml").read_text()
