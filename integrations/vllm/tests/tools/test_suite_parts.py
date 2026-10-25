"""`tools/suite_parts.py`: every test module of the suite is in exactly one part, whatever the number of parts; the modules are
the ones pytest collects; the parts are longest first, each to the least loaded; and `launch` runs each part as a suites.py run
of verity-vllm with the part's modules as its pytest arguments."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "suite_parts.py"
_spec = importlib.util.spec_from_file_location("suite_parts", SCRIPT)
SP = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("suite_parts", SP)
_spec.loader.exec_module(SP)


@pytest.mark.parametrize("n", [1, 2, 3, 6, 8])
def test_every_module_of_the_suite_is_in_exactly_one_part(n):
    names = SP.modules()
    got = SP.parts(names, n)
    flat = [m for p in got for m in p]
    assert len(got) == n and sorted(flat) == names and len(set(flat)) == len(flat)
    assert [m for k in range(1, n + 1) for m in SP.part(f"{k}/{n}", names)] == flat


def test_the_suite_has_its_test_modules_and_none_of_its_helpers():
    names = SP.modules()
    assert "tests/conftest.py" not in names and "tests/program/crosscheck.py" not in names
    assert "tests/tools/test_suite_parts.py" in names
    assert all(Path(m).name.startswith("test_") or m.endswith("_test.py") for m in names)


def test_the_modules_are_those_pytest_collects(tmp_path):
    """pytest's own collection of a tree with every case `modules` decides: both file patterns, a helper, a hidden and a
    `build` directory (default `norecursedirs`), a nested package."""
    files = ["tests/test_a.py", "tests/b_test.py", "tests/helper.py", "tests/sub/test_c.py", "tests/sub/deeper/test_d.py",
             "tests/.hidden/test_e.py", "tests/build/test_f.py", "tests/venv/test_g.py", "tests/sub/__init__.py"]
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        test = Path(f).name.startswith("test_") or f.endswith("_test.py")
        (tmp_path / f).write_text("def test_x():\n    pass\n" if test else "")
    (tmp_path / "pyproject.toml").write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "-p", "no:xdist"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    collected = sorted({line.split("::", 1)[0] for line in r.stdout.splitlines() if "::" in line})
    assert collected and SP.modules(tmp_path) == collected, r.stdout + r.stderr


def test_parts_go_longest_first_to_the_least_loaded():
    seconds = {"a": 10.0, "b": 7.0, "c": 5.0, "d": 4.0}
    assert SP.parts(["d", "c", "b", "a", "e", "f"], 2, seconds) == [["a", "d", "f"], ["b", "c", "e"]]   # e, f at DEFAULT_S
    assert SP.parts(["a"], 3, seconds) == [["a"], [], []]


@pytest.mark.parametrize("spec", ["0/3", "4/3", "3", "a/b", "1/0"])
def test_a_part_is_k_of_n(spec):
    with pytest.raises(ValueError):
        SP.part(spec, ["tests/test_a.py"])


def test_the_table_sums_each_modules_tests_from_a_report():
    report = {"passed": {"tests/a/test_x.py::t1": 15.0, "tests/a/test_x.py::t2[1]": 6.0, "tests/test_y.py::t": 3.0}}
    assert SP.table(report) == {"tests/a/test_x.py": 21}
    assert SP.table(report, floor_s=1.0) == {"tests/a/test_x.py": 21, "tests/test_y.py": 3}


def test_launch_runs_each_part_as_a_suites_run_of_its_modules():
    cmds = SP.jobs(3, ["--fresh"])
    assert len(cmds) == 3
    for k, cmd in enumerate(cmds, 1):
        at = cmd.index("--")
        assert cmd[:at] == ["research", "run", "--on", "auto", "--mem-gb", str(SP.JOB_MEM_GB), "--cwd", "source"]
        assert cmd[at + 1:at + 3] == ["bash", "-lc"] and len(cmd) == at + 4
        assert f"P=$(python3 integrations/vllm/tools/suite_parts.py {k}/3) && test -n \"$P\" && " in cmd[-1]
        assert cmd[-1].endswith("tools/verity/check/suites.py verity-vllm"
                                " --jobs \"$(nproc)\" --logs \"$OUT/logs\" --fresh -- $P")


def test_a_parts_job_runs_its_modules_and_stops_on_an_empty_part(tmp_path):
    """The job's shell, with `uv` a stand-in that prints its arguments: suites.py gets the part's modules, one argument each,
    and a part that lists nothing runs nothing (suites.py with no modules would run the whole suite)."""
    (tmp_path / "uv").write_text('#!/bin/sh\n[ "$1" = sync ] || printf "%s\\n" "$@"\n')
    (tmp_path / "uv").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    job = SP.jobs(3)[1][-1].replace("python3 ", f"{sys.executable} ")
    r = subprocess.run(["bash", "-c", job], cwd=SP.ROOT, env=env, capture_output=True, text=True)
    out = r.stdout.split("\n")
    assert r.returncode == 0 and out[out.index("--") + 1:-1] == SP.part("2/3"), r.stderr
    for lister in ("true", "false"):
        r = subprocess.run(["bash", "-c", job.replace(f"{sys.executable} integrations/vllm/tools/suite_parts.py 2/3", lister)],
                           cwd=SP.ROOT, env=env, capture_output=True, text=True)
        assert r.returncode != 0 and "verity-vllm" not in r.stdout


def test_the_cli_prints_a_parts_modules(capsys):
    assert SP.main(["2/3"]) == 0
    assert capsys.readouterr().out.split() == SP.part("2/3")
    assert SP.main(["--list", "3"]) == 0
    assert capsys.readouterr().out.count("part ") == 3


def test_the_seconds_table_prints_from_a_report(tmp_path, capsys):
    (tmp_path / "r.json").write_text(json.dumps({"passed": {"tests/test_a.py::t": 25.0}}))
    assert SP.main(["--seconds", str(tmp_path / "r.json")]) == 0
    assert capsys.readouterr().out.strip() == '"tests/test_a.py": 25,'


def test_the_seconds_table_sums_the_reports_of_every_part(tmp_path, capsys):
    (tmp_path / "1.json").write_text(json.dumps({"passed": {"tests/test_a.py::t": 25.0, "tests/test_b.py::t": 12.0}}))
    (tmp_path / "2.json").write_text(json.dumps({"passed": {"tests/test_b.py::u": 12.0}}))
    assert SP.main(["--seconds", str(tmp_path / "1.json"), str(tmp_path / "2.json")]) == 0
    assert capsys.readouterr().out.split("\n")[:2] == ['    "tests/test_a.py": 25,', '    "tests/test_b.py": 24,']
