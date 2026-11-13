module

public import Proofs.Flock.SpotFlock.Honest.Runs
public import Proofs.Flock.SpotFlock.Honest.Written
public import Proofs.Flock.SpotFlock.AddressFacts.RowOf

/-!
# The row at the instance

`rowFact_of_runs` proves the row at the address map from the wiring facts of each accepted run and three premises.
Two of them speak of a written bit at a port no link names (`AddressRow.OwnLeaves`, `AddressRow.PublishedRows`). At the
instance every port of `R` is linked, and where the design evaluates every row the member's units write is one of
`R`'s ports (`written_of_two`), so no written bit is at such a port (`linked_of_two`) and both hold with nothing to
check (`ownLeaves_held`, `publishedRows_held`).

So the row at the instance's address map takes `R`'s reference reads being well formed and where the second stage
puts a checked proof block (`AddressRow.Sits`): `rowFact_of_sits`. The address premise, the design premises and
`HonestExists` follow from the same two (`addressesConnected_of_sits`, `premises_of_sits`, `honest_of_sits`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **A written bit's port is linked**, where the design evaluates: its row is one the member's units write
(`writtenBits_row`), so one of `R`'s ports, and each has a link. -/
theorem linked_of_two {s : AuditStatement} {d : Design.Two} (h2 : two (held R B desc q c₀) s = some d)
    {y t : ℕ} {rp : Spot.Bits.RowPos} (hy : (y, t, rp) ∈ writtenBits (held R B desc q c₀).ms) :
    heldLink (held R B desc q c₀) ((held R B desc q c₀).names.getD t "") rp.row ≠ none := by
  obtain ⟨ht, hrow⟩ := writtenBits_row hy
  have hp : R.ports.contains rp.row = true := written_of_two R B desc q c₀ h2 rp.row hrow
  have hs := heldLink_held R B desc q c₀ rp.row hp
  rw [name_held R B desc q c₀ ht]
  intro e
  rw [e] at hs
  simp at hs

/-- **No registered read is at a written port no link names**, so the leaf count asked of one is asked of none. -/
theorem ownLeaves_held : AddressRow.OwnLeaves (held R B desc q c₀) := by
  intro r hr own reg _ _ hw hl
  obtain ⟨y, rp, hy, hrow⟩ := hw
  obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
  rw [← hrow] at hl
  exact absurd hl (linked_of_two R B desc q c₀ h2 hy)

/-- **No written port is without a link**, so the published rows asked at one are asked at none. -/
theorem publishedRows_held : AddressRow.PublishedRows (held R B desc q c₀) := by
  intro r hr y rp hy hl _
  obtain ⟨d, h2⟩ := CheckedRun.design_of_ok hr
  exact absurd hl (linked_of_two R B desc q c₀ h2 hy)

/-- **The row at the instance's address map**, from `R`'s reference reads being well formed and where the second
stage puts a checked proof block. -/
theorem rowFact_of_sits (hWf : Claims.Json.Wf R.reads) (hS : AddressRow.Sits (held R B desc q c₀)) :
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  rowFact_of_runs (fun _ hr => wiring_of_refReads (refReadsWf_held R B desc q c₀ hWf) hr) hS
    (ownLeaves_held R B desc q c₀) (publishedRows_held R B desc q c₀)

/-- **The address premise at the instance**, from the same two. -/
theorem addressesConnected_of_sits (hWf : Claims.Json.Wf R.reads) (hS : AddressRow.Sits (held R B desc q c₀)) :
    AddressesConnected (held R B desc q c₀) :=
  addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) (rowFact_of_sits R B desc q c₀ hWf hS)

/-- **The design premises at the instance**, from the same two. -/
theorem premises_of_sits (hWf : Claims.Json.Wf R.reads) (hS : AddressRow.Sits (held R B desc q c₀)) :
    Premises (held R B desc q c₀) :=
  premises_of_row R B desc q c₀ hWf (rowFact_of_sits R B desc q c₀ hWf hS)

/-- **`HonestExists`**, from the same two and a run the instance accepts. -/
theorem honest_of_sits (hWf : Claims.Json.Wf R.reads) (hS : AddressRow.Sits (held R B desc q c₀))
    (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_row R B desc q c₀ hWf (rowFact_of_sits R B desc q c₀ hWf hS) hA

end SpotFlock.Honest
