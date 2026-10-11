"""`tools/verity/check/mutants.py`: which check calls it deletes, and how it judges each deletion, on a stand-in source."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("check_mutants", Path(__file__).resolve().parents[1] / "mutants.py")
M = sys.modules[spec.name] = importlib.util.module_from_spec(spec)      # its dataclass looks itself up there
spec.loader.exec_module(M)

SOURCE = """def buildStmt (st : Stmt) : V Stmt := do
  checkRegistered st registered
  Registered.checkFull registered draw   -- U4
  -- checkCommented st
  let x ← checkReturns st
  checked (← checkCircuit st) n
  if st.ok then checkInIf st else pure ()
  Draw.check a b n
  check (a.size == 2
    && b.size == 3) "malformed"
  check (a.all ok)
    "S11: the streams are not the tables' streams"
  return st

/-- every relation the verifier will
  check holds of the value -/
structure Leaf where
  checkLayout : Circuit → V Unit

theorem checkInRange_ok :
    checkInRange k n m = .ok () ↔ k ≤ 27 := by
  simp [checkInRange]
"""


def test_a_mutant_is_each_bare_check_call_of_a_do_block():
    assert [(m.line, m.text) for m in M.mutants(SOURCE)] == [
        (2, "checkRegistered st registered"), (3, "Registered.checkFull registered draw   -- U4"), (8, "Draw.check a b n"),
        (9, 'check (a.size == 2'), (11, 'check (a.all ok)')]
    assert [m.last for m in M.mutants(SOURCE)][-2:] == [10, 12]
    assert [m.line for m in M.mutants(SOURCE, only="checkFull")] == [3]
    assert M.mutate(SOURCE, 3).splitlines()[2] == "  pure ()"
    assert M.mutate(SOURCE, 3).splitlines()[3] == "  -- checkCommented st"
    assert M.mutate(SOURCE, 9, 10).splitlines()[8:11] == ["  pure ()", "", "  check (a.all ok)"]


def test_a_check_no_test_needs_survives_and_the_source_is_restored(tmp_path):
    """The test refuses only when `checkRegistered` is gone, so deleting `checkFull` survives: the dropped-call case."""
    src = tmp_path / "Verify.lean"
    src.write_text(SOURCE)
    built = tmp_path / "built"
    out = tmp_path / "out.json"
    rc = M.main([str(src), "--build", f"cp {src} {built}", "--test", f"grep -q checkRegistered {built}", "--json", str(out)])
    assert rc == 1
    got = {m["line"]: m["outcome"] for m in json.loads(out.read_text())["mutants"]}
    assert got == {2: "killed", 3: "survived", 8: "survived", 9: "survived", 11: "survived"}
    assert src.read_text() == SOURCE and built.read_text() == SOURCE
    rc = M.main([str(src), "--build", f"cp {src} {built}", "--test", f"grep -q checkRegistered {built}", "--json", str(out),
                 "--line", "2", "--line", "11"])
    assert rc == 1
    assert {m["line"]: m["outcome"] for m in json.loads(out.read_text())["mutants"]} == {2: "killed", 11: "survived"}
    assert M.main([str(src), "--build", "true", "--test", "true", "--line", "4"]) == 2


def test_a_failing_baseline_or_build_is_not_a_kill(tmp_path, capsys):
    src = tmp_path / "Verify.lean"
    src.write_text(SOURCE)
    assert M.main([str(src), "--build", "true", "--test", "false"]) == 2
    assert "would kill every mutant" in capsys.readouterr().err
    flag = tmp_path / "built-once"
    rc = M.main([str(src), "--build", f"test ! -e {flag} && touch {flag} || grep -q checkFull {src}", "--test", "true",
                 "--only", "checkFull"])
    assert rc == 1 and "unbuilt " in capsys.readouterr().out
    assert src.read_text() == SOURCE
