import json
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
                  "capabilities": sandbox.capabilities()}))
"""


def test_refusals_name_each_interface_that_is_up_and_each_final_set_with_an_entry(tmp_path):
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
    assert sandbox.refusals([empty, final], down, 0) == [f"the final set at {final} is readable (tiny.bin)"]
    assert sandbox.final_sets({"data": {"b": {"final": "/x/b"}, "a": {"final": "/x/a"}, "c": {"final": "/x/a"}}}) == [
        Path("/x/a"), Path("/x/b")]


def test_the_entrys_command_finds_no_interface_up_and_each_hidden_directory_empty(tmp_path):
    """What a process started through the entry reaches: one loopback interface, down, an empty directory where the
    final set is, no capability to undo either, and no way to the final set through its parent's view of the file
    system. Outside it the final set is as readable as before."""
    final = tmp_path / "final"
    final.mkdir()
    (final / "tiny.bin").write_bytes(b"x")
    here = Path(run.__file__).parents[1]
    p = subprocess.run([sys.executable, "-m", "training_slowdown.sandbox", "--hide", final, "--hide",
                        tmp_path / "absent", "--", sys.executable, "-c", INSIDE, final, tmp_path / "absent"],
                       cwd=here, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout) == {"interfaces": {"lo": False}, "refusals": [], "entries": [[]],
                                    "through_its_parent": "Permission denied", "capabilities": 0}
    assert sandbox.refusals([final], {"lo": False}, 0) == [f"the final set at {final} is readable (tiny.bin)"]
    file = tmp_path / "file"
    file.write_bytes(b"x")
    p = subprocess.run([sys.executable, "-m", "training_slowdown.sandbox", "--hide", file, "--", "true"], cwd=here,
                       capture_output=True, text=True)
    assert p.returncode == 1 and f"{file} is not a directory" in p.stderr


def test_a_routes_process_outside_the_sandbox_stops_before_it_reads_its_config(tmp_path, monkeypatch):
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
    with pytest.raises(SystemExit, match=rf"outside it: the final set at {final} is readable \(manifest.json, tiny.bin\)"):
        run.main(argv)
    for f in Path(final).iterdir():
        f.unlink()
    monkeypatch.setattr(sandbox, "interfaces", lambda: {"lo": True, "eth0": True})
    with pytest.raises(SystemExit, match=r"outside it: the network is reachable \(interface eth0 is up\); the network "
                                         r"is reachable \(interface lo is up\). Start the job without --inside"):
        run.main(argv)
    assert not (tmp_path / "out").exists()


def test_a_final_job_is_scored_on_the_final_set_its_routes_process_cant_read(tmp_path):
    """A tiny start through the whole job on the final split: the route's process trained with no interface up and
    the final set's directory empty, which its record states, and every checkpoint it handed out was scored on the
    final set by the job's own process; the trace reads back."""
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
