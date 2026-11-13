"""`case_files`: a case's blobs reach the executable once each under short names, with the stack limit raised, and a
case that names a missing file has no run."""
from __future__ import annotations

import json
import os
import resource
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import case_files as CF

#: a stand-in executable: its arguments, each file's bytes in hex and its stack limit, as one line
ECHO = """import json, resource, sys
from pathlib import Path
print(json.dumps({"args": sys.argv[1:], "files": [Path(a).read_bytes().hex() for a in sys.argv[2:]],
                  "stack": resource.getrlimit(resource.RLIMIT_STACK)[0]}))
"""


@pytest.fixture
def exe(tmp_path: Path) -> Path:
    p = tmp_path / "echo"
    p.write_text(f"#!{sys.executable}\n{ECHO}")
    p.chmod(0o755)
    return p


def case(blobs: list[Path]) -> dict:
    return {"blobs": [str(b) for b in blobs], "commitment": "ab", "evidence": "cd", "params": {"b": 1, "a": [2]}}


def test_each_file_reaches_the_executable_once_under_a_short_name(tmp_path, exe):
    src = tmp_path / "files"
    src.mkdir()
    for n in ("x", "y"):
        (src / n).write_bytes(f"bytes of {n}".encode())
    os.link(src / "x", src / "x-again")
    got = CF.run(case([src / "x", src / "y", src / "x", src / "x-again"]), exe, "refusal", tmp_path / "run")
    line = json.loads(got["line"])
    assert line["args"] == ["refusal", "p", "c", "e", "0", "1"]
    assert [bytes.fromhex(f) for f in line["files"]] == [b'{"a":[2],"b":1}', b"\xab", b"\xcd", b"bytes of x",
                                                        b"bytes of y"]
    assert (got["rc"], got["listed"], got["linked"], got["bytes"]) == (0, 4, 2, 20)
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    inf = resource.RLIM_INFINITY
    assert line["stack"] == (soft if soft == inf or soft >= CF.STACK else CF.STACK if hard == inf else min(CF.STACK, hard))


def test_a_case_naming_a_missing_file_has_no_run(tmp_path, exe):
    with pytest.raises(FileNotFoundError, match="not there"):
        CF.run(case([tmp_path / "gone"]), exe, "refusal", tmp_path / "run")


def test_arguments_past_linuxs_cap_are_refused_before_the_run(tmp_path, exe, monkeypatch):
    monkeypatch.setattr(CF, "ARGS_CAP", 10)
    (tmp_path / "x").write_bytes(b"x")
    with pytest.raises(ValueError, match="past Linux's 10"):
        CF.run(case([tmp_path / "x"]), exe, "refusal", tmp_path / "run")
