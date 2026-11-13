import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from synthetic import run_job, tiny_registry, write_config

from training_slowdown import config, run, sandbox, starts

INSIDE = """
import json, os, sys
from pathlib import Path
from training_slowdown import sandbox
hidden = [Path(p) for p in sys.argv[1:]]
try:
    parents = os.listdir(f"/proc/{os.getppid()}/root{hidden[0]}")
except OSError as e:
    parents = e.strerror
print(json.dumps({"interfaces": sandbox.interfaces(), "refusals": sandbox.refusals(hidden),
                  "entries": [os.listdir(d) for d in hidden if d.exists()], "through_its_parent": parents,
                  "operators": {str(d): os.listdir(d) for d in sandbox.operators() if d.exists()},
                  "capabilities": sandbox.capabilities()}))
"""


@pytest.fixture
def no_operators(monkeypatch):
    """`refusals` with no directory or socket of this machine's own to name: a test outside the sandbox reads what it
    says of the paths it is given."""
    monkeypatch.setattr(sandbox, "operators", lambda uid=None: [])
    monkeypatch.setattr(sandbox, "ways_out", lambda uid=None: [])


def test_refusals_name_each_interface_that_is_up_and_each_final_set_with_an_entry(tmp_path, no_operators):
    final, empty, absent = tmp_path / "final", tmp_path / "empty", tmp_path / "absent"
    final.mkdir()
    empty.mkdir()
    (final / "tiny.bin").write_bytes(b"x")
    down = {"lo": False}
    assert sandbox.refusals([empty, absent], down, 0) == []
    assert sandbox.refusals([empty], down, 0x1FFFFFFFFFF) == [
        "the process holds capabilities (000001ffffffffff) with which it could undo the sandbox: run the job as a user "
        "other than root"]
    assert sandbox.refusals([empty, absent], {"lo": True, "eth0": True, "docker0": False}, 0) == [
        "the network is reachable (interface eth0 is up)", "the network is reachable (interface lo is up)"]
    assert sandbox.refusals([empty, final], down, 0) == [f"{final}, which the sandbox hides, is readable (tiny.bin)"]
    with pytest.raises(NotADirectoryError):
        sandbox.refusals([final / "tiny.bin"], down, 0)
    assert sandbox.final_sets({"data": {"b": {"final": "/x/b"}, "a": {"final": "/x/a"}, "c": {"final": "/x/a"}}}) == [
        Path("/x/a"), Path("/x/b")]


def test_refusals_name_each_of_the_operators_directories_with_an_entry_and_each_socket_left_visible(tmp_path,
                                                                                                    monkeypatch):
    """Outside the sandbox `refusals` names each directory the sandbox hides on every machine that has an entry, and
    each socket through which the user has a process started outside it. Stand-ins for both; the machine's own are
    what the entry's test reads."""
    assert sandbox.operators(7) == [Path("/run"), Path("/tmp/tmux-7")]
    assert sandbox.ways_out(7) == [Path("/run/user/7/systemd/private"), Path("/run/user/7/bus"),
                                   Path("/tmp/tmux-7/default")]
    assert sandbox.operators() == sandbox.operators(os.getuid()) and sandbox.ways_out() == sandbox.ways_out(os.getuid())
    under, tmux = tmp_path / "run", tmp_path / "tmux-7"
    manager, bus, server = under / "user/7/systemd/private", under / "user/7/bus", tmux / "default"
    manager.parent.mkdir(parents=True)
    tmux.mkdir()
    monkeypatch.setattr(sandbox, "operators", lambda uid=None: [under, tmux])
    monkeypatch.setattr(sandbox, "ways_out", lambda uid=None: [manager, bus, server])
    down = {"lo": False}
    assert sandbox.refusals([], down, 0) == [f"{under}, which the sandbox hides, is readable (user)"]
    bus.touch()
    server.touch()
    assert sandbox.refusals([under], down, 0) == [
        f"{under}, which the sandbox hides, is readable (user)", f"{tmux}, which the sandbox hides, is readable (default)",
        f"a process outside the sandbox is reachable (socket {bus}): what it starts has the network and every hidden "
        f"directory",
        f"a process outside the sandbox is reachable (socket {server}): what it starts has the network and every "
        f"hidden directory"]


def test_the_entrys_command_finds_no_interface_up_and_each_hidden_directory_empty(tmp_path):
    """What a process started through the entry reaches: one loopback interface, down, an empty directory where the
    final set is and where each of the operators' directories on this machine is, no capability to undo either, and
    no way to the final set through its parent's view of the file system. Outside it the final set and /run are as
    readable as before."""
    final = tmp_path / "final"
    final.mkdir()
    (final / "tiny.bin").write_bytes(b"x")
    here = Path(run.__file__).parents[1]
    p = subprocess.run([sys.executable, "-m", "training_slowdown.sandbox", "--hide", final, "--hide",
                        tmp_path / "absent", "--", sys.executable, "-c", INSIDE, final, tmp_path / "absent"],
                       cwd=here, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout) == {"interfaces": {"lo": False}, "refusals": [], "entries": [[]],
                                    "through_its_parent": "Permission denied",
                                    "operators": {str(d): [] for d in sandbox.operators() if d.exists()},
                                    "capabilities": 0}
    outside = sandbox.refusals([final], {"lo": False}, 0)
    assert outside[0] == f"{final}, which the sandbox hides, is readable (tiny.bin)"
    assert outside[1].startswith("/run, which the sandbox hides, is readable (")
    file = tmp_path / "file"
    file.write_bytes(b"x")
    p = subprocess.run([sys.executable, "-m", "training_slowdown.sandbox", "--hide", file, "--", "true"], cwd=here,
                       capture_output=True, text=True)
    assert p.returncode == 1 and f"{file} is not a directory" in p.stderr


def test_a_machine_that_refuses_the_namespaces_fails_the_job_with_what_it_needs(tmp_path, monkeypatch):
    """Where the kernel refuses the user namespace the entry stops with the kind's requirement, and a job whose
    route's process never started is a failed job with no record. The kernel's refusal is a stand-in here: a libc
    whose `unshare` fails, then an entry that exits 1."""
    class Refusing:
        def unshare(self, flags):
            return -1

    with monkeypatch.context() as m:
        m.setattr(sandbox.ctypes, "CDLL", lambda *a, **k: Refusing())
        m.setattr(sandbox.ctypes, "get_errno", lambda: errno.EPERM)
        with pytest.raises(SystemExit, match=r"sandbox: unshare\(user, mount, network\): Operation not permitted. a "
                                             r"machine that runs a training job lets an unprivileged process make a "
                                             r"user namespace: sysctl kernel.unprivileged_userns_clone=1"):
            sandbox.enter([])
    reg = tiny_registry(tmp_path / "reg")
    monkeypatch.setattr(run.sys, "executable", "false")
    assert run.main(["--config", write_config(tmp_path / "a.toml"), "--seed", 0, "--split", "search", "--out",
                     tmp_path / "out", "--starts", reg]) == 1
    assert not (tmp_path / "out").exists()


def test_the_routes_process_gets_no_name_of_the_stores_key_and_finds_its_directory_empty(tmp_path, monkeypatch):
    """A job whose environment holds a run's read-only key to the store, as `research run --store-read` gives it
    (names that start AWS_, and RESEARCH_STORE_CONFIG, which names the store's configuration, the key lying in a
    file beside it), starts its route's process with none of those names and every other, and with that directory
    among what the sandbox hides: through the entry a process finds it empty. A configuration beside what the
    route's process needs is refused before anything starts. The values are dummies, and only names are read: the
    route's process is a stand-in that writes down its argv and its environment's names."""
    staged, said, standin = tmp_path / "requests" / "store-read", tmp_path / "said.json", tmp_path / "bin" / "python"
    for d in (staged, standin.parent):
        d.mkdir(parents=True)
    for name in ("store.toml", "cred.json"):
        (staged / name).write_text("dummy")
    standin.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                       f"open({str(said)!r}, 'w').write(json.dumps({{'argv': sys.argv[1:], 'names': sorted(os.environ)}}))\n"
                       f"sys.exit(1)\n")
    standin.chmod(0o755)
    dummy = {"AWS_ACCESS_KEY_ID": "dummy", "AWS_SECRET_ACCESS_KEY": "not-to-be-copied", "AWS_SESSION_TOKEN": "dummy",
             "AWS_PROFILE": "dummy", "RESEARCH_STORE_CONFIG": str(staged / "store.toml"),
             "RESEARCH_RUN_DIR": "dummy", "NOT_AWS_": "dummy"}
    for name, value in dummy.items():
        monkeypatch.setenv(name, value)
    reg, python = tiny_registry(tmp_path / "reg"), sys.executable
    argv = ["--config", write_config(tmp_path / "a.toml"), "--seed", 0, "--split", "search", "--out", tmp_path / "out",
            "--starts", reg]
    monkeypatch.setattr(run.sys, "executable", str(standin))
    assert run.main(argv) == 1 and not (tmp_path / "out").exists()
    route = json.loads(said.read_text())
    names, kept = set(route["names"]), {"RESEARCH_RUN_DIR", "NOT_AWS_", "PATH", "PYTHONPATH"}
    assert not [n for n in names if n.startswith("AWS_")] and "RESEARCH_STORE_CONFIG" not in names
    inherited = {n for n in os.environ if n not in dummy}
    assert inherited | kept <= names
    entry, command = route["argv"][:route["argv"].index("--")], route["argv"][route["argv"].index("--") + 1:]
    assert entry[:2] == ["-m", "training_slowdown.sandbox"] and entry[-2:] == ["--hide", str(staged)]
    assert command[:3] == [str(standin), "-m", "training_slowdown.run"]
    seen = subprocess.run([python, *entry, "--", python, "-c", f"import os; print(os.listdir({str(staged)!r}))"],
                          cwd=Path(run.__file__).parents[1], capture_output=True, text=True)
    assert (seen.returncode, seen.stdout) == (0, "[]\n"), seen.stderr[-2000:]
    assert sorted(p.name for p in staged.iterdir()) == ["cred.json", "store.toml"]

    said.unlink()
    monkeypatch.setenv("RESEARCH_STORE_CONFIG", str(tmp_path / "store.toml"))
    with pytest.raises(SystemExit) as refused:
        run.main(argv)
    assert str(refused.value).endswith(f"so the job hides {tmp_path} from it, which holds {standin}, which that "
                                       f"process needs. Keep the store's configuration in a directory of its own.")
    assert not said.exists() and not (tmp_path / "out").exists()


def test_a_routes_process_outside_the_sandbox_stops_before_it_reads_its_config(tmp_path, monkeypatch, no_operators):
    """The route's process, started by hand where the final set is readable or an interface is up, is a failed job
    with the reason, and it has imported no route by then."""
    reg = tiny_registry(tmp_path / "reg")
    path = write_config(tmp_path / "a.toml")
    final = starts.registry(reg)["data"]["tiny"]["final"]
    argv = ["--config", path, "--seed", 0, "--split", "search", "--out", tmp_path / "out", "--starts", reg, "--inside",
            tmp_path]
    monkeypatch.setattr(config, "load", lambda *a, **k: pytest.fail("the route's process read its config"))
    monkeypatch.setattr(sandbox, "interfaces", lambda: {"lo": False})
    monkeypatch.setattr(sandbox, "capabilities", lambda: 0)
    with pytest.raises(SystemExit, match=rf"outside it: {final}, which the sandbox hides, is readable "
                                         rf"\(manifest.json, tiny.bin\)"):
        run.main(argv)
    for f in Path(final).iterdir():
        f.unlink()
    monkeypatch.setattr(sandbox, "interfaces", lambda: {"lo": True, "eth0": True})
    with pytest.raises(SystemExit, match=r"outside it: the network is reachable \(interface eth0 is up\); the network "
                                         r"is reachable \(interface lo is up\). Start the job without --inside"):
        run.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.slow
def test_a_final_job_is_scored_on_the_final_set_its_routes_process_cant_read(tmp_path):
    """A tiny start through the whole job on the final split: the route's process trained with no interface up and
    the final set's directory empty, which its record states, and every checkpoint it handed out was scored on the
    final set by the job's own process; the trace reads back. The search job test_run shares shows the sandbox and
    the trace; only a final job shows scores from the set the route's process can't read."""
    rec, cfg, reg, out = run_job(tmp_path, split="final", seeds={"search": [], "final": [0]})
    final = starts.registry(reg)["data"]["tiny"]["final"]
    assert rec["invalid"] == [] and rec["split"] == "final" and len(rec["steps"]) == 3
    assert rec["sandbox"] == {"interfaces": {"lo": False}, "final_sets": {final: "empty"}}
    assert sorted(p.name for p in Path(final).iterdir()) == ["manifest.json", "tiny.bin"]
    for cp in rec["checkpoints"]:
        assert cp["score"]["split"] == "final" and cp["bpb"] == cp["score"]["bpb"] > 0
        assert cp["score"]["targets"] == 2 * 1024 + 50 - 1
    assert len({cp["bpb"] for cp in rec["checkpoints"]}) == 3
    assert rec["trace"]["tokens"] == rec["steps"][0]["tokens"] and rec["trace"]["rows"]
