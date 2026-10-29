"""pods/nebius/slot_only.py: on a machine that takes runs only through its check slots, which commands count, and which live runs it
stops (cpu-1, 8 Oct: three runs outside slot.py beside the merge check, two from trees older than the launcher's refusal)."""
from __future__ import annotations

import json
import signal

from research.pods.nebius import slot_only as S


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
    """27 Oct: `verity lean try --on vy-nebius-1` without `--pool` was refused, and the refusal named only slot.py by hand."""
    (tmp_path / "proofs").write_text("p0 0-15\n")
    why = S.refusal(tmp_path / "marker", {"proofs": str(tmp_path / "proofs"), "soundness": str(tmp_path / "soundness")})
    assert "add `--pool proofs` to `research run` or `verity lean try`" in why
    assert "slot.py [--by NAME] -- CMD" in S.refusal(tmp_path / "marker", {"soundness": str(tmp_path / "soundness")})


def test_research_run_pool_wraps_the_command_in_a_named_pool_that_the_launcher_accepts(capsys):
    """INF-27/28: a lane's run on cpu-1 takes the `research` pool with one flag, not a five-layer command line."""
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
    """INF-61 (cpu-3, 18 Oct): a `--pool` run's --mem-gb reserved at launch and held that through its wait in the slot line, beside
    the slot's own reservation once it ran; two waiting runs held 128 GB and admission sat at -89 GB. The slot admits it, with the
    run's --mem-gb when that is more than its mem= (24 Oct: a 450 GB run in proofs' 193 GB slot p5 reserved 193)."""
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
    """INF-106: a `--pool` run's --timeout counted its wait in the pool's line too, so a run that waited 50 of its 60 min ran 10."""
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
    """Node 1 (top, 18 Oct): lanes' direct runs took the cores the train checks need. `--renice 19` stops nothing: it renices every
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
    """vy-challenger-1 (INF-154): a run launched from a tree older than the launcher's auditor refusal is stopped, unless its command
    is exactly a line of /etc/vy/auditor; slot.py's runs get no pass there."""
    runs, proc, marker, log = tmp_path / "runs", tmp_path / "proc", tmp_path / "auditor", tmp_path / "auditor-only.jsonl"
    ok = _run(runs, "r-ok", "running", ["bash", "benchmarks/live_challenger/run.sh"], pgid=10, runner=9)
    slot = _run(runs, "r-slot", "running", ["python3", "tools/verity/check/slot.py", "--", "true"], pgid=20, runner=19)
    extra = _run(runs, "r-extra", "running", ["bash", "benchmarks/live_challenger/run.sh", "x"], pgid=30, runner=29)
    for pid, run in ((10, ok), (20, slot), (30, extra)):
        _proc(proc, pid, pid, run)
    marker.write_text("# comment\n\nbash benchmarks/live_challenger/run.sh\n")
    assert S.auditor_commands(marker) == [["bash", "benchmarks/live_challenger/run.sh"]]
    assert [o["run"] for o in S.offenders(runs, lambda c: S.auditor_allows(c, marker))] == ["r-extra", "r-slot"]
    sent = []
    monkeypatch.setattr(S.os, "kill", lambda pid, sig: sent.append(pid))
    monkeypatch.setattr(S, "stop", lambda o, dry_run=False, why="", _stop=S.stop: _stop(o, proc=proc, kill=S.os.kill, dry_run=dry_run,
                                                                                       why=why))
    assert S.main(["--runs", str(runs), "--marker", str(marker), "--log", str(log), "--auditor"]) == 0
    assert sent == [30, 20]
    assert {json.loads(ln)["why"] for ln in log.read_text().splitlines()} == {f"not a command of the auditor host's {marker}"}
    marker.write_text("")
    assert S.auditor_commands(marker) == [] and not S.auditor_allows(["bash", "benchmarks/live_challenger/run.sh"], marker)
    assert S.auditor_commands(tmp_path / "absent") == []


def test_deploy_makes_the_challenger_and_verifier_hosts_auditor_hosts_that_take_no_research_run():
    """The deployed marker lists no command (the challenger runs only through vy-challenger-gate, the verifier through
    vy-verifier-gate) and goes, with the timer that enforces it, to the auditor hosts only, each host's gate and units to it
    alone; the nodes' shared peers file and the cluster's description name no auditor host."""
    import tomllib
    from pathlib import Path

    hosts = ["vy-challenger-1", "vy-auditor-1", "vy-auditor-2"]
    nebius = Path(S.__file__).resolve().parent
    assert (nebius / "auditor").read_text().startswith("#") and S.auditor_commands(nebius / "auditor") == []
    rows = tomllib.loads((nebius / "deploy.toml").read_text())["file"]
    on = {e["path"]: e["nodes"] for e in rows if e.get("src", "").endswith(("/auditor", "/vy-auditor-only.service",
                                                                            "/vy-auditor-only.timer", "/11-vy-admin.conf"))}
    assert on == {p: hosts for p in ("/etc/vy/auditor", "/etc/systemd/system/vy-auditor-only.service",
                                     "/etc/systemd/system/vy-auditor-only.timer", "/etc/ssh/sshd_config.d/11-vy-admin.conf")}
    assert {e["path"]: e["nodes"] for e in rows if e.get("src", "").endswith("/peers-challenger")} == {
        "/etc/vy/peers": ["vy-challenger-1"]}
    assert "--auditor --marker /etc/vy/auditor" in (nebius / "vy-auditor-only.service").read_text()
    gate = {e["path"]: e["nodes"] for e in rows if "/pods/nebius/challenger/" in e.get("src", "")
            and not e["src"].endswith("/11-vy-admin.conf")}
    assert gate == {p: ["vy-challenger-1"] for p in ("/usr/local/sbin/vy-challenger-gate", "/usr/local/sbin/vy-challenger-setup",
                                                     "/usr/local/sbin/vy-challenger-pin", "/etc/systemd/system/vy-challenger@.service",
                                                     "/etc/systemd/system/vy-certifier@.service")}
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
