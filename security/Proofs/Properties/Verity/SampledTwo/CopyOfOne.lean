module

public import Proofs.Properties.Verity.SampledTwo.HeldOfAtDesc

/-!
# One input's copy is every input's

`heldOf` reads an input and its interior through seven fields: the boundary verifier, the boundary record, the rule,
the closure map, the interior verifier, the replay block and the templates' circuits (`heldOf_congr`). So where one
input and interior that carry the parameters `P` and the boundary record `j` in those fields build `A.h`, every one
does: `CopyOf A P j u` (`copyOf_of_one`), the premise `heldOfAtOr_at` (`RefsOfSent`) takes of the session of record.
-/

@[expose] public section

namespace SpotFlock.HeldOf

/-- **`heldOf` reads seven fields**: two inputs and interiors with one boundary verifier, boundary record, rule,
closure map, interior verifier, replay block and templates' circuits build one copy, or none. -/
theorem heldOf_congr {i i' : Spot.Run.Input} {it it' : Spot.Run.Interior} (u : Unread)
    (h1 : i.params.verifier = i'.params.verifier) (h2 : i.record = i'.record) (h3 : i.rule = i'.rule)
    (h4 : i.closure = i'.closure) (h5 : it.params.verifier = it'.params.verifier) (h6 : it.replay = it'.replay)
    (h7 : it.circuits = it'.circuits) : heldOf i it u = heldOf i' it' u := by
  have hf : firstOf i = firstOf i' := by
    unfold firstOf
    simp only [h1, h2, h3, h4]
  have hc : cellsOf i it = cellsOf i' it' := by
    unfold cellsOf
    simp only [h3, h5, h6]
  have hh : ∀ first ns ks r hz, held₀ i it u first ns ks r hz = held₀ i' it' u first ns ks r hz := by
    intro first ns ks r hz
    unfold held₀
    simp only [h1, h2, h5, h6, h7]
  have hk : ∀ h, keep u it h = keep u it' h := by
    intro h
    unfold keep decided
    simp only [h7]
  unfold heldOf
  rw [hf, hc]
  simp only [hh, hk]

end SpotFlock.HeldOf

namespace Verity.SampledTwo

open SpotFlock.HeldOf (Unread heldOf)

/-- **One input's copy gives `CopyOf`**: where an input and interior that carry the parameters `P` and the boundary
record `j` in the seven fields `heldOf` reads build `A.h`, every such input and interior builds `A.h` or none
(`HeldOf.heldOf_congr`). -/
theorem copyOf_of_one {A : Live₂} {P : VStar.Recursive.SampledTwo.Params} {j : Lean.Json} {u : Unread}
    {i₀ : Spot.Run.Input} {it₀ : Spot.Run.Interior} (g1 : i₀.params.verifier = P.boundary.verifier)
    (g2 : i₀.record = j) (g3 : i₀.rule = P.rule) (g4 : i₀.closure = none)
    (g5 : it₀.params.verifier = P.interior.verifier) (g6 : it₀.replay = P.replay) (g7 : it₀.circuits = P.circuits)
    (h₀ : heldOf i₀ it₀ u = some A.h) : CopyOf A P j u := by
  intro i it h h1 h2 h3 h4 h5 h6 h7 hh
  rw [SpotFlock.HeldOf.heldOf_congr u (h1.trans g1.symm) (h2.trans g2.symm) (h3.trans g3.symm)
    (h4.trans g4.symm) (h5.trans g5.symm) (h6.trans g6.symm) (h7.trans g7.symm), h₀] at hh
  exact (Option.some.inj hh).symm

end Verity.SampledTwo
