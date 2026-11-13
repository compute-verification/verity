module

public import Proofs.Flock.SpotFlock.Honest.Runs
public import Proofs.Flock.SpotFlock.Honest.Written
public import Proofs.Flock.SpotFlock.AddressFacts.RowLeaves
public import Proofs.Flock.SpotFlock.SampleBelow

/-!
# The row at the instance

`rowFact_of_wires` proves the row at the address map from the wiring facts of each accepted run, one condition
decided on the registered wires (`AddressRow.wiresChecked`) and two premises. The instance has no wire, so the
condition holds (`wiresChecked_held`). One premise speaks of a written bit at a port no link names
(`AddressRow.PublishedRows`). At the instance every port of `R` is linked, and where the design evaluates every row
the member's units write is one of `R`'s ports (`written_of_two`), so no written bit is at such a port
(`linked_of_two`) and it holds with nothing to check (`publishedRows_held`).

So the row at the instance's address map takes `R`'s reference reads being well formed and where the second stage
puts a checked proof block (`AddressRow.Sits`): `rowFact_of_sits`. The address premise, the design premises and
`HonestExists` follow from the same two (`addressesConnected_of_sits`, `premises_of_sits`, `honest_of_sits`).

Where the second stage puts a checked proof block is owed of a registration with no proof block always checked whose
first stage samples only slots of the layout that name themselves. The instance is one: none is always checked
(`always_held`), and its first stage samples one unit of a population of one (`Bnd.units`), so the one outer block
(`sample_held`), which the layout's one proof block names (`sampleSlots_held`).
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

/-- **Each wire is one the second stage's check reads**: there is no wire. -/
theorem wiresChecked_held : AddressRow.wiresChecked (held R B desc q c₀) = true := by
  simp [AddressRow.wiresChecked, held, held₀]

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
  rowFact_of_wires (fun _ hr => wiring_of_refReads (refReadsWf_held R B desc q c₀ hWf) hr) hS
    (wiresChecked_held R B desc q c₀) (publishedRows_held R B desc q c₀)

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

/-! ## The first stage's sample -/

/-- **No proof block is always checked** at the instance. -/
theorem always_held (v : ℕ) : (held R B desc q c₀).reg.always v = false := rfl

/-- **The first stage samples the one outer block**: its population is one unit (`Bnd.units`), and a `subset:1`
sample is below its population (`RegTwo.sample_lt`). -/
theorem sample_held (c : ByteArray) : ∀ ρ ∈ (held R B desc q c₀).reg.sample c, ρ = 0 := by
  intro ρ hρ
  obtain ⟨n, hn, hlt⟩ := RegTwo.sample_lt (reg B) B.firstK c
  have hn1 : n = 1 := Except.ok.inj (hn.symm.trans B.units)
  have hρn : ρ < n := hlt ρ hρ
  omega

/-- **Each block the first stage samples is a slot of the layout that names itself**: the layout is one proof block,
unit `0`, and the sample is that block. -/
theorem sampleSlots_held (l : Spot.Slots.SlotOrder)
    (hl : (Spot.Slots.layout (held R B desc q c₀).reg.first.program
      (held R B desc q c₀).reg.first.query).toOption = some l) (c : ByteArray) :
    ∀ ρ ∈ (held R B desc q c₀).reg.sample c, ∃ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ := by
  intro ρ hρ
  rw [layout_held R B desc q c₀] at hl
  obtain rfl := Option.some.inj hl
  obtain rfl := sample_held R B desc q c₀ c ρ hρ
  refine ⟨⟨some 0, B.gs⟩, ?_, rfl⟩
  simp [Spot.Slots.SlotOrder.blocks, oneLayout]

end SpotFlock.Honest
