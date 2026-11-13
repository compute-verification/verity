"""The benchmarks' runs of the Lean verifier (`a0.lean_verify`, `server_rate.Lean`) start its `verify` in
`verifier_env`: under the verifier's allocator setting where this process's environment sets none, and under the
caller's own value where it does, with the rest of the environment as it is. The children are stand-ins that record the
environment they were given."""
from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.spot.verifier.one_stage.proofs import EXE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import a0  # noqa: E402
import server_rate as SR  # noqa: E402

NAME = "MIMALLOC_PURGE_DELAY"
VERDICT = 'VERDICT {"accepted": true}\n'
CASES = pytest.mark.parametrize("own, want", [(None, "-1"), ("0", "0")], ids=["unset", "the caller's own"])


def caller(monkeypatch, own: str | None) -> None:
    if own is None:
        monkeypatch.delenv(NAME, raising=False)
    else:
        monkeypatch.setenv(NAME, own)


@CASES
def test_a0s_lean_verify_runs_under_the_allocator_setting_unless_the_caller_sets_its_own(tmp_path, monkeypatch, own, want):
    children = {}

    def child(cmd, **kw):
        children[cmd[1]] = kw["env"]
        return subprocess.CompletedProcess(cmd, 0, VERDICT if cmd[1] == "verify" else "", "")

    caller(monkeypatch, own)
    monkeypatch.setattr(a0.subprocess, "run", child)
    st, out = {"dir": tmp_path / "m"}, tmp_path / "session"
    st["dir"].mkdir()
    (st["dir"] / "circuit.txt").write_text("circuit")
    (out / "sessions" / "s").mkdir(parents=True)
    (out / "sessions" / "s" / "session.json").write_text("{}")
    assert a0.lean_verify(EXE, st, 8, out, zk=True)["accepted"]
    assert children["verify"] == {**os.environ, NAME: want}


@CASES
def test_server_rates_lean_runs_under_the_allocator_setting_unless_the_caller_sets_its_own(tmp_path, monkeypatch, own, want):
    children = []

    class Child:
        pid = 0

        def __init__(self, cmd, **kw) -> None:
            children.append(kw["env"])
            self.stdout = io.StringIO(VERDICT)

    caller(monkeypatch, own)
    monkeypatch.setattr(SR.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(SR.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(SR.subprocess, "Popen", Child)
    monkeypatch.setattr(SR.os, "wait4", lambda pid, options: (pid, 0, SimpleNamespace(ru_utime=0.0, ru_stime=0.0, ru_maxrss=0)))
    lean = SR.Lean(EXE, tmp_path, tmp_path)
    try:
        assert lean(SimpleNamespace(dir=tmp_path / "s", label="t/0"))["accepted"]
    finally:
        lean.close()
    assert children == [{**os.environ, NAME: want}]
