import Definitions.Flock.Circuit

/-!
# The wire values a witness opens

* `posOf C g`: wire `g`'s position in the circuit's lowered program, its last lowered gate (`gateSize`: three for a
  row, one for any other gate; `off`: where a wire's lowered gates start).
* `openedOf C κ val`: the wire values opened at those positions, an input the constants `κ` fix its constant, `false`
  where nothing is opened.

Moved unchanged from `Proofs.Flock.Bridge.Defs` and `Proofs.Flock.Bridge.CircuitOf`; the lemmas about them stay
there.
-/

namespace Flock.Bridge

/-- The Boolean-profile gates one C-Flock gate lowers to: three for a row, one for any other. -/
def gateSize {N : ℕ} : FlockSoundness.Audit.Op N → ℕ
  | .row _ _ => 3
  | _ => 1

/-- Where wire `g`'s lowered gates start: the lowered sizes of the gates before it. -/
def off (C : FlockSoundness.Audit.Circuit) (g : ℕ) : ℕ :=
  (((List.finRange C.N).filter fun h => decide (h.val < g)).map fun h => gateSize (C.op h)).sum

/-- **Wire `g`'s position** in the lowered program: its last lowered gate. -/
def posOf (C : FlockSoundness.Audit.Circuit) (g : Fin C.N) : ℕ := off C g.val + gateSize (C.op g) - 1

/-- **The wire values a witness opens**: an input the constants `κ` fix, its constant; every other wire, the value
opened at its position (`posOf`), `false` where none is. -/
def openedOf (C : FlockSoundness.Audit.Circuit) (κ : Fin C.N → Option Bool) (val : ℕ → Option Bool) (g : Fin C.N) :
    Bool :=
  if (C.op g).isInput then (κ g).getD ((val (posOf C g)).getD false) else (val (posOf C g)).getD false

end Flock.Bridge
