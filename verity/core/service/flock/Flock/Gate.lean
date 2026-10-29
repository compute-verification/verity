/-
A gate of the canonical layout (PROTOCOL.md §6), as `Flock.Program.Prog.gate` locates it: `layout.Gate`, which
`verity-evaluate`'s `gate` function holds to it.
-/
import Flock.Program

namespace Flock.Program

/-- `layout.Gate`: gate `index`'s primitive (its Definition's index), its operands (gate indices, per argument in order,
repeats kept) and the activation that holds the primitive's node. -/
structure Gate where
  index : Nat
  prim : Nat
  operands : Array Nat
  scope : Scope
  deriving Inhabited

/-- `layout.Circuit.gate`: gate `g` of the Program's canonical layout. -/
def Prog.gateAt (P : Prog) (g : Nat) : V Gate := do
  let (prim, operands, scope) ← P.gate g
  return { index := g, prim := prim, operands := operands, scope := scope }

end Flock.Program
