/-
The recheck: the boundary recomputes the outer proof's level-0 commitment itself, from V*'s witness (the proxy's) and
the outer session's masks and salts (the ZK layer's), so the one outer message that commits to the masked witness is a
function of those alone and carries nothing the GPU rack chose. The send gate releases the level-0 message only when the
commitment it carries is the recomputed one.
-/
import Run.Boundary.Proxy

namespace Run.Recheck

structure State where
  witness : Option (Array ByteArray × Array Salt) := none
  masks : Option (ByteArray × Array Salt) := none
  /-- the recomputed commitment, once both are in -/
  level0 : Option Digest := none
  deriving Inhabited

/-- The outer prover's level-0 commitment as C-Flock computes it: the masked witness encoded and its rows committed
under the ZK layer's salts (`Flock.MerkleScheme.hm96Sha512`'s leaves). The executable verifier holds only the checks;
the encoding is new Lean. -/
def level0Of (A : Agreement) (rows : Array ByteArray) (rowSalts : Array Salt) (masks : ByteArray) (salts : Array Salt) :
    Digest := sorry

def step (A : Agreement) (s : State) (src : Node) (b : Body) (_ : ByteArray) :
    Except String (State × List (Node × Body)) := do
  match src, b with
  | .party .proxy, .witness rows salts =>
    if s.witness.isSome then throw "recheck: the witness arrives once"
    return ({ s with witness := some (rows, salts) }, [])
  | .party .zkLayer, .masks m salts =>
    if s.masks.isSome then throw "recheck: the masks arrive once"
    return ({ s with masks := some (m, salts) }, [])
  | .party .sendGate, .level0 root =>
    let (some (rows, rowSalts), some (m, salts)) := (s.witness, s.masks) | throw "recheck: witness or masks missing"
    let want := level0Of A rows rowSalts m salts
    if root != want then throw "recheck: the level-0 commitment is not the boundary's"
    return ({ s with level0 := some want }, [(.party .sendGate, .level0 want)])
  | _, _ => throw "recheck: not a message it takes"

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh _ _ _ := 0
  step := step A
  halted s := s.level0.isSome

def machine (A : Agreement) : Machine := (core A).seq (.party .recheck)

end Run.Recheck
