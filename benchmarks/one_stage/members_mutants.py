"""The members road's coverage refusals deleted from this tree's Lean verifier one at a time (`OneStageZk.headerHeldAt`'s
and `headerHeld`'s, and `checkMembersAt`'s null refusals), each mutant's `spot-one-stage-zk` built by `members.check`,
and a5's member negatives (`a5.run_member_negatives`) run on the case an audit kept (`A5_MEMBERS_NEGATIVES`). A mutant
survives where every negative's verdict and reason is the honest tree's; one whose build fails is killed by the proof
or the in-file check that reads it, which the line names. The tree is restored after each mutant.

    members_mutants.py CASE_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a5  # noqa: E402

LEAN = Path(__file__).resolve().parents[2] / "verity/protocols/proofs/spot/verifier/lean/Spot"
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
]


def main(keep: Path) -> int:
    case = json.loads((keep / "case.json").read_text())
    honest = {k: (v["lean"]["verdict"], v["lean"]["reason"]) for k, v in a5.run_member_negatives(case).items()}
    print("HONEST " + json.dumps(honest, sort_keys=True), flush=True)
    survivors = []
    for name, path, old, new in MUTANTS:
        texts = {p: p.read_text() for p in (STATEMENT, ZK)}
        if texts[path].count(old) != 1:
            print(f"MUTANT {name} not applied: its refusal's text is in {path.name} {texts[path].count(old)} times", flush=True)
            survivors.append(name)
            continue
        try:
            mutated = {**texts, path: texts[path].replace(old, new)}
            # the in-file guards read the refusals too; as #eval they print and fail nothing, so a negative decides
            for p, t in mutated.items():
                p.write_text(t.replace("#guard ", "#eval "))
            got = {k: (v["lean"]["verdict"], v["lean"]["reason"]) for k, v in a5.run_member_negatives(case).items()}
        finally:
            for p, t in texts.items():
                p.write_text(t)
        if all(v == "lean-not-run" for v, _ in got.values()):
            print(f"MUTANT {name} killed at its build: {next(iter(got.values()))[1][-600:]}", flush=True)
            continue
        killed = sorted(k for k in got if got[k] != honest.get(k))
        print(f"MUTANT {name} {'killed by ' + ', '.join(killed) if killed else 'SURVIVES'} "
              + json.dumps({k: got[k] for k in killed}, sort_keys=True)[:1500], flush=True)
        if not killed:
            survivors.append(name)
    print("SURVIVORS " + json.dumps(survivors), flush=True)
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
