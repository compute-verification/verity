module

public import Proofs.Flock.Bridge.Lowers

/-!
# The lowered program is in the readable scope's order

* `posOf_lt_length`: every wire's position is a gate of `progOf C κ`.
* `topological_progOf`: `progOf C κ` reads only earlier gates (`reads_lt`) and returns only its own gates, so it is
  topologically ordered (`Prog.Topological`), for every circuit and constants.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

variable {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool}

theorem posOf_lt_length (g : Fin C.N) : posOf C g < (progOf C κ).gates.length := by
  have h1 := one_le_gateSize (C.op g)
  have hl : gateSize (C.op g) - 1 < (lower C κ g).length := by rw [length_lower]; omega
  have h : (progOf C κ).gates[posOf C g]? = some (lower C κ g)[gateSize (C.op g) - 1] := by
    rw [posOf_eq, gates_off g (by omega), List.getElem?_eq_getElem hl]
  exact (List.getElem?_eq_some_iff.1 h).1

/-- **The lowered program is topologically ordered.** -/
theorem topological_progOf : (progOf C κ).Topological :=
  ⟨fun _ _ hx _ hi => reads_lt hx hi, fun o ho => by
    obtain ⟨g, -, rfl⟩ := List.mem_map.1 ho
    exact posOf_lt_length g⟩

end Flock.Bridge
