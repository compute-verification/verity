"""`check.py`'s program cache (`--program-cache`, `kept_program`): the tool's compiled Lean programs are kept for later
processes under a key of everything they are compiled from (`program_key`), so a changed source or toolchain is never
served an old program, processes that ask at once all get one whole program from one compile, and none a process holds
is evicted. The real-Lean tests compile a one-line program with the installed toolchain, in about a second."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHECK = HERE.parent / "check.py"
spec = importlib.util.spec_from_file_location("lean_audit_programs", CHECK)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

TOOLCHAIN = (HERE / "fixtures" / "control" / "lean-toolchain").read_text()
needs_lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def hello(tool: Path, text: str) -> None:
    (tool / "Hello.lean").write_text(f'def main : IO Unit := IO.println "{text}"\n')


@pytest.fixture
def setup(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """(The package the program is compiled for, the tool directory holding `Hello.lean`), with `PROGRAMS` in
    ``tmp_path`` and taken as one no build writes: these tests are of the key and the store, `sandbox_reads_only` of the
    probe."""
    pkg, tool = tmp_path / "pkg", tmp_path / "tool"
    pkg.mkdir()
    tool.mkdir()
    (pkg / "lean-toolchain").write_text(TOOLCHAIN)
    monkeypatch.setattr(C, "TOOL", tool)
    monkeypatch.setattr(C, "PROGRAMS", tmp_path / "programs")
    monkeypatch.setattr(C, "sandbox_reads_only", lambda d: True)
    monkeypatch.setattr(C, "_programs", {})
    monkeypatch.setattr(C, "_TAKEN", [])
    (tmp_path / "programs").mkdir()
    return pkg, tool


@needs_lean
def test_a_source_changed_by_one_byte_is_compiled_again_and_runs_as_changed(setup, monkeypatch):
    pkg, tool = setup
    compiled, compile_program = [], C.compile_program
    monkeypatch.setattr(C, "compile_program", lambda src, tc, exe: (compiled.append(exe), compile_program(src, tc, exe)))
    hello(tool, "a")
    first = C.program(pkg, "Hello.lean")
    assert Path(first).parent == C.PROGRAMS and subprocess.run([first], capture_output=True, text=True).stdout == "a\n"
    C._programs.clear()                                 # a later process: it takes the kept program, compiling nothing
    assert C.program(pkg, "Hello.lean") == first and len(compiled) == 1
    hello(tool, "b")
    C._programs.clear()
    second = C.program(pkg, "Hello.lean")
    assert second != first and len(compiled) == 2, "a source one byte apart is another key"
    assert subprocess.run([second], capture_output=True, text=True).stdout == "b\n"


CHILD = """
import hashlib, importlib.util, subprocess, sys
from pathlib import Path
check, pkg, tool, programs = sys.argv[1:]
spec = importlib.util.spec_from_file_location("check", check)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
C.TOOL, C.PROGRAMS, C.sandbox_reads_only = Path(tool), Path(programs), lambda d: True
compile_program = C.compile_program
def counted(*a):
    print("compiled", flush=True)
    compile_program(*a)
C.compile_program = counted
print("ready", flush=True)
sys.stdin.readline()
exe = C.program(Path(pkg), "Hello.lean")
print("program", exe, hashlib.sha256(Path(exe).read_bytes()).hexdigest(), flush=True)
print("ran", subprocess.run([exe], capture_output=True, text=True).stdout.strip(), flush=True)
"""


@needs_lean
def test_two_processes_asking_at_once_both_get_the_whole_program_from_one_compile(setup, tmp_path):
    pkg, tool = setup
    hello(tool, "a")
    kids = [subprocess.Popen([sys.executable, "-c", CHILD, str(CHECK), str(pkg), str(tool), str(C.PROGRAMS)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) for _ in range(2)]
    for k in kids:
        assert k.stdout.readline() == "ready\n"
    for k in kids:                                      # both released once both are ready
        k.stdin.write("go\n")
        k.stdin.close()
    outs = [k.stdout.read().splitlines() for k in kids]
    assert [k.wait() for k in kids] == [0, 0], outs
    assert sum(line == "compiled" for o in outs for line in o) == 1, outs
    got = [next(line.split()[1:] for line in o if line.startswith("program ")) for o in outs]
    assert got[0] == got[1] and hashlib.sha256(Path(got[0][0]).read_bytes()).hexdigest() == got[0][1]
    assert [next(line for line in o if line.startswith("ran ")) for o in outs] == ["ran a", "ran a"]


def toolchain(root: Path) -> Path:
    for f in ("bin/lean", "bin/leanc", "lib/lean/Init.olean"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(f)
    return root


def test_the_key_moves_with_the_source_and_any_file_of_the_toolchain_and_needs_both(setup, tmp_path, monkeypatch):
    pkg, tool = setup
    tc = toolchain(tmp_path / "tc")
    monkeypatch.setattr(C, "elan_toolchain", lambda p: tc)
    hello(tool, "a")
    src = tool / "Hello.lean"
    key, at = C.program_key(pkg, src)
    assert at == tc and C.program_key(pkg, src) == (key, tc)
    hello(tool, "b")
    assert C.program_key(pkg, src)[0] != key, "the source's bytes"
    hello(tool, "a")
    assert C.program_key(pkg, src)[0] == key
    (tc / "lib" / "lean" / "Init.olean").write_text("other")
    assert C.program_key(pkg, src)[0] != key, "a module of the toolchain"
    (tc / "bin" / "leanc").unlink()
    assert C.program_key(pkg, src) == f"the toolchain {tc} has no bin/leanc", "no `leanc`: no key"
    monkeypatch.setattr(C, "elan_toolchain", lambda p: "`elan which lean` named no toolchain")
    assert C.program_key(pkg, src) == "`elan which lean` named no toolchain", "no toolchain `elan` names: no key"
    assert C.kept_program(pkg, src) == "`elan which lean` named no toolchain", "the caller compiles its own"


def test_the_compile_reads_none_of_the_caller_s_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LEAN_CC", "cc")
    monkeypatch.setenv("LEAN_PATH", str(tmp_path))
    env = C.program_env(tmp_path / "tc")
    assert sorted(env) == ["PATH"] and env["PATH"] == str(tmp_path / "tc" / "bin")


def test_no_program_is_kept_where_a_build_could_write_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROGRAMS", tmp_path / "programs")
    monkeypatch.setattr(C, "_READS_ONLY", {})
    monkeypatch.setattr(C, "sandboxed", lambda: False)
    hello(tmp_path, "a")
    assert C.sandbox_reads_only(tmp_path / "programs") is False
    assert "could have written it" in C.kept_program(tmp_path, tmp_path / "Hello.lean")


def test_eviction_keeps_the_most_recently_taken_and_every_program_a_process_holds(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "PROGRAMS", tmp_path)
    monkeypatch.setattr(C, "PROGRAMS_KEPT", 3)
    for i in range(8):
        (tmp_path / f"facts-{i}").write_text(str(i))
        os.utime(tmp_path / f"facts-{i}", ns=(i, i))
    (tmp_path / "unread-0").write_text("another source's")
    with open(tmp_path / ".facts-1.lock", "a") as held:
        fcntl.flock(held, fcntl.LOCK_SH)
        assert C.evict("facts") == ["facts-1"]
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith(".")) == [
        "facts-1", "facts-5", "facts-6", "facts-7", "unread-0"]
