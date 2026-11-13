module

public import Proofs.Flock.SpotFlock.Honest.Runs
public import Proofs.Flock.SpotFlock.Honest.Written
public import Proofs.Flock.SpotFlock.AddressFacts.RowPlace
public import Proofs.Flock.SpotFlock.SampleBelow

/-!
# The row at the instance

`rowFact_of_wires` proves the row at the address map from the wiring facts of each accepted run, one condition
decided on the registered wires (`AddressRow.wiresChecked`) and two premises. The instance has no wire, so the
condition holds (`wiresChecked_held`). Both premises hold at the instance.

* One speaks of a written bit at a port no link names (`AddressRow.PublishedRows`). At the instance every port of
  `R` is linked, and where the design evaluates every row the member's units write is one of `R`'s ports
  (`written_of_two`), so no written bit is at such a port (`linked_of_two`) and it holds with nothing to check
  (`publishedRows_held`).
* The other is where the second stage puts a checked proof block (`AddressRow.Sits`). `AddressRow.sits_of_slots`
  proves it of a registration with no proof block always checked whose first stage samples only slots of the layout
  that name themselves (`AddressRow.SampleSlots`). The instance is one: none is always checked (`always_held`), and
  its first stage samples one unit of a population of one (`Bnd.units`), so the one outer block (`sample_held`),
  which the layout's one proof block names (`sampleSlots_held`). So `sits_held`.

So the row at the instance's address map takes only `R`'s reference reads being well formed: `rowFact_held`. The
address premise and the design premises follow from the same (`addressesConnected_held`, `premises_held`), and
`HonestExists` from it and a run the instance accepts (`honest_of_accepts`).
-/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-! ## The published rows -/

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
theorem sampleSlots_held : AddressRow.SampleSlots (held R B desc q c₀) := by
  intro l hl c ρ hρ
  have hl' : Spot.Slots.layout B.g.program B.g.query = .ok l := hl
  rw [B.layout] at hl'
  obtain rfl := Except.ok.inj hl'
  obtain rfl := sample_held R B desc q c₀ c ρ hρ
  refine ⟨⟨some 0, B.gs⟩, ?_, rfl⟩
  simp [Spot.Slots.SlotOrder.blocks, oneLayout]

/-- **Where the second stage puts a checked proof block**, at the instance: none is always checked and the first
stage samples the layout's one slot. -/
theorem sits_held : AddressRow.Sits (held R B desc q c₀) :=
  AddressRow.sits_of_slots (always_held R B desc q c₀) (sampleSlots_held R B desc q c₀)

/-! ## The row and the premises -/

/-- **The row at the instance's address map**, from `R`'s reference reads being well formed. -/
theorem rowFact_held (hWf : Claims.Json.Wf R.reads) :
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  rowFact_of_wires (fun _ hr => wiring_of_refReads (refReadsWf_held R B desc q c₀ hWf) hr)
    (sits_held R B desc q c₀) (wiresChecked_held R B desc q c₀) (publishedRows_held R B desc q c₀)

/-- **The address premise at the instance**, from `R`'s reference reads being well formed. -/
theorem addressesConnected_held (hWf : Claims.Json.Wf R.reads) : AddressesConnected (held R B desc q c₀) :=
  addressesConnected_of_row (refReadsWf_held R B desc q c₀ hWf) (rowFact_held R B desc q c₀ hWf)

/-- **The design premises at the instance**, from `R`'s reference reads being well formed. -/
theorem premises_held (hWf : Claims.Json.Wf R.reads) : Premises (held R B desc q c₀) :=
  premises_of_row R B desc q c₀ hWf (rowFact_held R B desc q c₀ hWf)

/-- **`HonestExists`**, from `R`'s reference reads being well formed and a run the instance accepts. -/
theorem honest_of_accepts (hWf : Claims.Json.Wf R.reads) (hA : Accepts (held R B desc q c₀)) : HonestExists :=
  honest_of_row R B desc q c₀ hWf (rowFact_held R B desc q c₀ hWf) hA

end SpotFlock.Honest
