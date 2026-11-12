"""A queued job's script as bash runs it, around stand-ins for `uv`, `research` and the job: where the job's Python
environment is made, that the tree is left as it was, and what is left when the job ends or is told to stop."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from synthetic import tiny_registry, write_config
from test_command import ART, Research, cmd, command, stop

from training_slowdown import config, starts, stop_record, told

#: `uv`, on the script's PATH: it says what it was asked and where, and puts the two programs the script then starts
#: into the environment it was told to make. Under STANDIN=sync it then waits to be told to stop, as a sync still
#: running is
UV = """#!{python}
import json, os, signal, sys
from pathlib import Path
env = Path(os.environ["UV_PROJECT_ENVIRONMENT"])
(env / "bin").mkdir(parents=True)
with open({log!r}, "a") as f:
    f.write(json.dumps({{"program": "uv", "argv": sys.argv[1:], "cwd": os.getcwd(), "environment": str(env)}}) + "\\n")
for name in ("python", "research"):
    (env / "bin" / name).write_text(Path({program!r}).read_text())
    (env / "bin" / name).chmod(0o755)
if os.environ["STANDIN"] == "sync":
    os.write(int(os.environ["UP"]), b"up")
    signal.pause()
"""
#: the environment's `research` and `python`: each says what it was asked and where. The job (`python`) ends by
#: itself under STANDIN=ends; else it says it is up and waits, and told to stop writes its record and exits with 75,
#: as `training_slowdown.run` does
PROGRAM = """#!{python}
import json, os, signal, sys
from pathlib import Path
name = Path(sys.argv[0]).name
with open({log!r}, "a") as f:
    f.write(json.dumps({{"program": name, "argv": sys.argv[1:], "cwd": os.getcwd()}}) + "\\n")
if name == "research" or os.environ["STANDIN"] == "ends":
    sys.exit(0)

def stop(signum, frame):
    Path(os.environ["OUT"]).mkdir(parents=True, exist_ok=True)
    (Path(os.environ["OUT"]) / "run.json").write_text({record!r})
    sys.exit(75)

signal.signal(signal.SIGTERM, stop)
os.write(int(os.environ["UP"]), b"up")
signal.pause()
"""


@pytest.mark.parametrize("standin, status", [("ends", 0), ("job", 75), ("sync", -signal.SIGTERM)])
def test_a_queued_jobs_environment_is_its_runs_own_under_tmp_and_gone_when_it_ends(tmp_path, monkeypatch, standin,
                                                                                    status):
    """The script the command queues, in a tree like the one a run ships: `uv sync` is told to make the environment
    in a directory of the run's own under the command's `TMP`, the kept state is fetched and the job started with
    that environment's programs, and the tree holds what it held. The directory is gone when the script ends: after a
    job that ended by itself, after one told to stop (SIGTERM to the script's process group, as a lease out of time
    sends it), whose record and exit status are then the run's, and after a stop that came while `uv sync` ran, which
    leaves no record. What a test of the command's text can't show is that bash, told to stop, waits for the job and
    still removes the directory."""
    root, made, out = tmp_path.resolve() / "tree", tmp_path / "tmp", tmp_path / "run" / "out"
    here, bin_, log = root / "benchmarks" / "training_slowdown", tmp_path / "bin", tmp_path / "log"
    monkeypatch.setattr(cmd, "HERE", here)
    monkeypatch.setattr(cmd, "ROOT", root)
    monkeypatch.setattr(cmd, "TMP", str(made))
    reg = tiny_registry(tmp_path / "reg")
    (job,) = config.load(write_config(here / "attacks" / "a.toml"), starts.registry(reg)).jobs()
    r = Research()
    r.states.add(ART)
    command(r, "a", "--starts", str(reg))
    stop(r, job)
    command(r, "a", "--starts", str(reg))
    script = r.runs[1][-1]
    assert r.runs[1][-3:-1] == ["bash", "-lc"] and f"mktemp -d {made}/training-slowdown.XXXXXX" in script

    for d in (made, bin_, out.parent):
        d.mkdir()
    record = json.dumps(stop_record(job.seed, job.split))
    (tmp_path / "program").write_text(PROGRAM.format(python=sys.executable, log=str(log), record=record))
    (bin_ / "uv").write_text(UV.format(python=sys.executable, log=str(log), program=str(tmp_path / "program")))
    (bin_ / "uv").chmod(0o755)
    before = sorted(p.relative_to(root) for p in root.rglob("*"))
    up, tell = os.pipe()
    p = subprocess.Popen(["bash", "-c", script], cwd=root, pass_fds=(tell,), start_new_session=True,
                         env={"PATH": f"{bin_}{os.pathsep}{os.defpath}", "OUT": str(out), "UP": str(tell),
                              "STANDIN": standin})
    os.close(tell)
    with open(up, "rb") as said:
        if standin != "ends":
            assert said.read(2) == b"up"
            (venv,) = made.glob("training-slowdown.*/venv")
            assert sorted(x.name for x in (venv / "bin").iterdir()) == ["python", "research"]
            os.killpg(p.pid, signal.SIGTERM)
        assert said.read() == b""        # every process of the script has ended
    assert p.wait() == (status if status >= 0 else 128 - status)
    assert list(made.iterdir()) == [] and sorted(x.relative_to(root) for x in root.rglob("*")) == before
    if standin == "job":
        assert told(json.loads((out / "run.json").read_text()))
    else:
        assert not out.exists()

    said = [json.loads(line) for line in log.read_text().splitlines()]
    made_in = Path(said[0].pop("environment"))
    assert said[0] == {"program": "uv", "argv": ["sync", "-q", "--frozen", "--all-packages", "--extra", "torch"],
                       "cwd": str(root)}
    assert made_in.name == "venv" and made_in.parent.parent == made
    assert made_in.parent.name.startswith("training-slowdown.") and ".venv" not in script
    assert [x["program"] for x in said[1:]] == ([] if standin == "sync" else ["research", "python"])
    for x in said[1:]:
        assert x["argv"] == ({"research": ["data", "fetch", ART, "--path", "out/state/*", "--to", str(out.parent)],
                              "python": ["-m", "training_slowdown.run", "--config", "attacks/a.toml", "--seed", "0",
                                         "--split", "search", "--starts", str(reg.resolve()), "--state",
                                         f"{out}/state", "--continues", r.sha256, "--out", str(out)]}[x["program"]])
        assert x["cwd"] == str(root if x["program"] == "research" else here)
