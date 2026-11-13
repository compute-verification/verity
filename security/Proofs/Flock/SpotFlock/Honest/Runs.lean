module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.Connected

/-!
# The design premises at the instance, from the row

At any held code whose reference reads are well formed (`RefReadsWf`), the address premise follows from the row at
the address map (`addressesConnected_of_row`). At the instance the reference reads are `R`'s (`refReadsWf_held`), so
the row and `R`'s reads being well formed give the design premises (`premises_of_row`) and, with an accepted run,
`HonestExists` (`honest_of_row`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The instance's reference reads are well formed**, where `R`'s are. -/
theorem refReadsWf_held (hWf : Claims.Json.Wf R.reads) : RefReadsWf (held R B desc q c₀) := fun _ _ => hWf

/-- **The design premises at the instance**, from the row at the address map and `R`'s reference reads being well
formed. -/
theorem premises_of_row (hWf : Claims.Json.Wf R.reads)
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) : Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) hRow⟩

/-- **`HonestExists`**, from the row at the instance's address map, `R`'s reference reads being well formed and a
run the instance accepts. -/
theorem honest_of_row (hWf : Claims.Json.Wf R.reads)
    (hRow : RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀))) (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) hRow, hA⟩

end SpotFlock.Honest
