/-
A plain session draws by the subset law at hidden outputs (`ProvedScope.lawWhy`, the one rule for which law each form
draws by), at one table, the form `flock_verify_sound`'s plain hidden-output branch covers (`SoundExecH` at its subset
draw design): a plain subset draw passes at hidden outputs and under `--zk`, and is still refused at public outputs; a
plain session at hidden outputs is still refused at two tables, the form just past the proof.
-/
import Flock.ProvedScope

namespace Flock.Test.PlainHidden

def subsetDraw : Draw.UnitDraw := { law := .subset 2, population := 8, units := #[1, 6] }

def lawWhy (zk : Bool) (tags : Tags) : String :=
  match ProvedScope.drawLaw zk tags false (some subsetDraw.toJson) with
  | .error e => e
  | .ok () => ""

def tablesWhy (count : Nat) : String :=
  match ProvedScope.check Tags.circuit false ByteArray.empty ByteArray.empty [] count false false with
  | .error e => e
  | .ok () => ""

#guard Tags.circuit.hiddenOutputs && !Tags.circuit967b8d06.hiddenOutputs
#guard lawWhy false Tags.circuit == ""
#guard lawWhy true Tags.circuit == ""
#guard lawWhy true Tags.circuit967b8d06 == ""
#guard lawWhy false Tags.circuit967b8d06 == "refused: a subset draw without --zk: no soundness claim draws by this law"
#guard ProvedScope.firstFailure (ProvedScope.statementRules Tags.circuit false 1 false) == none
#guard tablesWhy 2 ==
  "refused: 2 tables in a session without --zk: a statement form the soundness proof does not cover"

end Flock.Test.PlainHidden
