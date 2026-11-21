"""PoUW's verifier (`verifier`): one assignment that names the side of record, in a module that imports only the
standard library, so undoing the switch is one line and what decides an execution phase, or flips its coins, never reaches
`verity.ml`."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from verity.protocols.accounting.compute.pouw.verifier import verifier as V


def _imports(path: str) -> tuple[ast.Module, set[str]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    return tree, {m for m in imported if m.split(".")[0] not in sys.stdlib_module_names}


def test_the_switch_is_one_line_in_a_module_that_imports_only_the_standard_library() -> None:
    tree, beyond = _imports(V.__file__)
    assert beyond == {"verity"}, beyond
    assert V.LC.__name__ == "verity.lean_process" and not _imports(V.LC.__file__)[1]
    assigned = [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    assert assigned.count("RECORD_VERDICT") == 1 and V.RECORD_VERDICT in ("lean", "python")


_FAKE = """import json, sys
for line in sys.stdin:
    sys.stderr.write({said!r} + "\\n")
    sys.stderr.flush()
    print(json.dumps({{"output": {{"verdict": "accept"}}}}), flush=True)
"""


@pytest.mark.parametrize("said", ["PANIC at PouwVerifier.Credit.tileOf PouwVerifier.Credit:540:4: index out of bounds",
                                  "Error: index out of bounds"])
def test_a_verifier_that_panics_and_answers_accept_with_exit_0_is_refused_by_name(monkeypatch, tmp_path, said) -> None:
    """A verifier that writes a panic line, a ``PANIC`` one or an out-of-range index's with none, then answers accept
    and ends 0, gives no answer: `lean_process.LEAN_PANIC`, whatever it answered."""
    fake = tmp_path / "pouw-verifier-difftest"
    fake.write_text(f"#!{sys.executable}\n{_FAKE.format(said=said)}", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(V, "executable", lambda exe=V.EXE: fake)
    monkeypatch.setattr(V, "_server", V._Server())
    got = V.of_record("window", {})
    assert got["refused"][0] == V.NO_ANSWER and f"{V.LC.LEAN_PANIC}: " in got["refused"][1], got
    assert said in got["refused"][1], got
    V._server.close()


@pytest.mark.parametrize("end, how", [("os.kill(os.getpid(), signal.SIGABRT)", "killed by SIGABRT"),
                                      ("sys.exit(3)", "status 3"), ("os._exit(134)", "status 134 (abort)")])
def test_a_verifier_killed_by_a_signal_or_ending_with_no_verdict_is_refused_by_name(monkeypatch, tmp_path, end,
                                                                                   how) -> None:
    """A verifier killed by a signal, or ending silently with a status `SERVE_EXITS` doesn't hold (134 an abort's), gives
    no answer: `lean_process.LEAN_EXIT`."""
    fake = tmp_path / "pouw-verifier-difftest"
    fake.write_text(f"#!{sys.executable}\nimport os, signal, sys\nsys.stdin.readline()\n{end}\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(V, "executable", lambda exe=V.EXE: fake)
    monkeypatch.setattr(V, "_server", V._Server())
    got = V.of_record("window", {})
    assert got["refused"][0] == V.NO_ANSWER and f"{V.LC.LEAN_EXIT}: {how}: " in got["refused"][1], got


def test_without_lake_the_refusal_names_the_path_it_searched(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(V, "BUILD_LOCK", tmp_path / "build.lock")
    try:
        V.executable.__wrapped__("pouw-verifier-difftest")
    except V.LeanError as e:
        assert f"lake is not on the PATH ({tmp_path})" in str(e) and "~/.elan/bin" in str(e)
    else:
        raise AssertionError("built without lake")
