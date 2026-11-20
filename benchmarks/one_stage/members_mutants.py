"""The members road's refusals deleted from this tree's Lean verifier one at a time (`OneStageZk.headerHeldAt`'s and
`headerHeld`'s, and the null refusals of `checkMembersAt` and `checkOneAt`), each mutant judged by the Spot package's
build, whose `#guard`s and proofs read the refusals, and then by the Spot verifier's small tests of them (`TESTS`). A
mutant survives where the build and every test pass; the line names what killed the others. The tree is restored after
each mutant. A tool run by hand: no script or gate runs it.

    members_mutants.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "verity/protocols/proofs/spot/verifier/lean"
LEAN = PACKAGE / "Spot"
SUITE = ROOT / "tests/verity/protocols/proofs/spot/verifier"
#: the small tests of the members road's refusals
TESTS = [SUITE / "lean/test_members_twin.py", SUITE / "lean/test_header_held_at_difftest.py", SUITE / "test_members_run.py"]
STATEMENT, ZK = LEAN / "OneStageZkStatement.lean", LEAN / "OneStageZk.lean"

#: (name, file, the refusal's text, its deletion)
MUTANTS = [
    ("SHARED_UNREAD", STATEMENT, "      throw (SHARED_UNREAD p t)\n", "      pure ()\n"),
    ("UNREGISTERED_WHY", STATEMENT, "      if !registeredAt held p v then throw (UNREGISTERED_WHY p v)\n",
     "      if false then throw (UNREGISTERED_WHY p v)\n"),
    ("POPULATION_WHY", STATEMENT, "  if !jeq (getD h \"instances\") (getD rec \"population\") then throw POPULATION_WHY\n",
     "  if false then throw POPULATION_WHY\n"),
    ("headerHeld-matchesPublic", STATEMENT,
     "  | none => if c.codes.isEmpty then pure () else throw s!\"public file: not the registration's: {\", \".intercalate c.codes.toList}\"\n",
     "  | none => if c.codes.isEmpty || true then pure () else throw s!\"public file: not the registration's: {\", \".intercalate c.codes.toList}\"\n"),
    ("headerHeld-UNWIRED", STATEMENT, "  | some p => throw (UNWIRED_WHY p)\n", "  | some _ => pure ()\n"),
    ("NO_SESSION_ROOTS", ZK, "  if e.base.sessionRoots.isNone then .reject NO_SESSION_ROOTS else\n",
     "  if false then .reject NO_SESSION_ROOTS else\n"),
    ("NO_SERVED", ZK, "  if e.base.served.isNone then .reject NO_SERVED else\n", "  if false then .reject NO_SERVED else\n"),
    ("checkOneAt-NO_SESSION_ROOTS", STATEMENT, "  if e.base.sessionRoots.isNone then .reject NO_SESSION_ROOTS else\n",
     "  if false then .reject NO_SESSION_ROOTS else\n"),
    ("checkOneAt-NO_SERVED", STATEMENT, "  if e.base.served.isNone then .reject NO_SERVED else\n",
     "  if false then .reject NO_SERVED else\n"),
]


def killed_by() -> str | None:
    """What refuses the tree as it stands: the package's build (a `#guard` or a proof), a test, or nothing."""
    b = subprocess.run(["lake", "build", "Spot", "spot-one-stage-zk"], cwd=PACKAGE, capture_output=True, text=True, check=False)
    if b.returncode:
        return "its build: " + (b.stdout + b.stderr)[-600:]
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *map(str, TESTS)],
                       cwd=ROOT, capture_output=True, text=True, check=False)
    return None if t.returncode == 0 else "a test: " + t.stdout[-600:]


def main() -> int:
    if (why := killed_by()) is not None:
        print(f"HONEST refused, so no mutant is judged: {why}", flush=True)
        return 2
    survivors = []
    for name, path, old, new in MUTANTS:
        text = path.read_text()
        if text.count(old) != 1:
            print(f"MUTANT {name} not applied: its refusal's text is in {path.name} {text.count(old)} times", flush=True)
            survivors.append(name)
            continue
        try:
            path.write_text(text.replace(old, new))
            why = killed_by()
        finally:
            path.write_text(text)
        print(f"MUTANT {name} " + ("SURVIVES" if why is None else f"killed by {why}"), flush=True)
        if why is None:
            survivors.append(name)
    print("SURVIVORS " + str(survivors), flush=True)
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
