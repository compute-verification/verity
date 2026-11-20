"""PoUW's verifier (`verifier`): one assignment that names the side of record, in a module that imports only the
standard library, so undoing the switch is one line and what decides an execution phase, or flips its coins, never reaches
`verity.ml`."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from verity.protocols.resource_accounting.compute.pouw.verifier import verifier as V


def _imports(path: str) -> tuple[ast.Module, set[str]]:
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    return tree, {m for m in imported if m.split(".")[0] not in sys.stdlib_module_names}


def test_the_switch_is_one_line_in_a_module_that_imports_only_the_standard_library() -> None:
    tree, beyond = _imports(V.__file__)
    assert beyond == {"verity.protocols.proofs.spot.verifier.one_stage"}, beyond
    assert not _imports(V.LC.__file__)[1] and not _imports(V.LC.lean_env.__code__.co_filename)[1]
    assigned = [t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
    assert assigned.count("RECORD_VERDICT") == 1 and V.RECORD_VERDICT in ("lean", "python")


_FAKE = """import json, sys
for line in sys.stdin:
    sys.stderr.write("PANIC at PouwVerifier.Credit.tileOf PouwVerifier.Credit:540:4: index out of bounds\\n")
    sys.stderr.flush()
    print(json.dumps({"output": {"verdict": "accept"}}), flush=True)
"""


def test_a_verifier_that_panics_and_answers_accept_with_exit_0_is_refused_by_name(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "pouw-verifier-difftest"
    fake.write_text(f"#!{sys.executable}\n{_FAKE}", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(V, "executable", lambda exe=V.EXE: fake)
    monkeypatch.setattr(V, "_server", V._Server())
    got = V.of_record("window", {}, {"verdict": "accept"})
    assert got["refused"][0] == V.DISAGREE and f"{V.LC.LEAN_PANIC}: " in got["refused"][1], got
    assert "PANIC at PouwVerifier.Credit.tileOf" in got["refused"][1], got
    V._server.close()


def test_without_lake_the_refusal_names_the_path_it_searched(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(V, "BUILD_LOCK", tmp_path / "build.lock")
    try:
        V.executable.__wrapped__("pouw-verifier-difftest")
    except V.LeanError as e:
        assert f"lake is not on the PATH ({tmp_path})" in str(e) and "~/.elan/bin" in str(e)
    else:
        raise AssertionError("built without lake")
