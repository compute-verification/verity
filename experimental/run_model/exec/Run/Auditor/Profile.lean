/-
The profile: the run's one output. On the verifier's verdict it states what verification established, from the
agreement and the draw alone: the exhaustive facts, the law over the drawn level, the units drawn, and the claim with
its δ. A rejecting verdict gives no profile. It issues one and takes nothing after.
-/
import Run.Auditor.Verifier

namespace Run.Profile

/-- The facts the verifier checked in full, whatever the draw. -/
def exhaustive : Array String :=
  #["registration committed before the draw", "every inner round committed before its coins",
    "outer session: Flock.Zk.verify on V* pinned by digest", "V*'s public words: the record's hashes, coins and comb"]

def ofDraw (A : Agreement) (d : Flock.Draw.UnitDraw) : IntegrityProfile :=
  { program := A.program, vstar := A.vstar, exhaustive, level := A.level, law := A.law, population := A.population,
    drawn := d.proved, threshold := A.threshold, deltaNum := A.deltaNum, deltaDen := A.deltaDen }

def profileOf (A : Agreement) (ok : Bool) (log : Array Entry) : Option IntegrityProfile :=
  if !ok then none else
  match Verifier.parse log with
  | .ok { draw := some d, .. } => some (ofDraw A d)
  | _ => none

structure State where
  issued : Bool := false
  deriving Inhabited

def step (A : Agreement) (_ : State) (src : Node) (b : Body) (_ : ByteArray) :
    Except String (State × List (Node × Body)) := do
  if src != .party .verifier then throw "profile: it takes the verifier's verdict only"
  let .verdict ok _ log := b | throw "profile: not a verdict"
  return ({ issued := true }, [(.outside, .profile (profileOf A ok log))])

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh _ _ _ := 0
  step := step A
  halted s := s.issued

def machine (A : Agreement) : Machine := (core A).seq (.party .profile)

end Run.Profile
