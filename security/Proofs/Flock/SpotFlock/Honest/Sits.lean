module

public import Proofs.Flock.SpotFlock.Honest.Instance
public import Proofs.Flock.SpotFlock.AddressFacts.RowFirst

/-! The address premise and the design premises at the instance take only `R`'s reference reads being well formed
(`addressesConnected_held`, `premises_held`). -/

@[expose] public section

namespace SpotFlock.Honest

open Claims Claims.Sampling SamplingDeploy

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

theorem refReadsWf_held (hWf : Claims.Json.Wf R.reads) : RefReadsWf (held R B desc q c₀) := fun _ _ => hWf

/-- **No proof block is always checked** at the instance. -/
theorem always_held (v : ℕ) : (held R B desc q c₀).reg.always v = false := rfl

theorem wiresChecked_held : AddressRow.wiresChecked (held R B desc q c₀) = true := by
  simp [AddressRow.wiresChecked, held, held₀]

/-- **The address premise at the instance**, from `R`'s reference reads being well formed. -/
theorem addressesConnected_held (hWf : Claims.Json.Wf R.reads) : AddressesConnected (held R B desc q c₀) :=
  addressesConnected_of_firstK (refReadsWf_held R B desc q c₀ hWf) (always_held R B desc q c₀)
    (popOk_held R B desc q c₀) B.firstK (wiresChecked_held R B desc q c₀)

/-- **The design premises at the instance**, from `R`'s reference reads being well formed. -/
theorem premises_held (hWf : Claims.Json.Wf R.reads) : Premises (held R B desc q c₀) :=
  ⟨builtWired_held R B desc q c₀, addressesConnected_held R B desc q c₀ hWf⟩

theorem honest_of_accepts (hWf : Claims.Json.Wf R.reads) (hA : AcceptsPlay (held R B desc q c₀)) : HonestExists :=
  honest_of_owed ⟨R, B, desc, q, c₀, addressesConnected_held R B desc q c₀ hWf, hA⟩

/-- **Each block the first stage samples is a slot of the layout that names itself** at the instance. -/
theorem sampleSlots_held : AddressRow.SampleSlots (held R B desc q c₀) :=
  AddressRow.sampleSlots_of_firstK (popOk_held R B desc q c₀) B.firstK

/-- **Where the second stage puts a checked proof block**, at the instance. -/
theorem sits_held : AddressRow.Sits (held R B desc q c₀) :=
  AddressRow.sits_of_slots (always_held R B desc q c₀) (sampleSlots_held R B desc q c₀)

/-- **The row at the instance's address map**, from `R`'s reference reads being well formed. -/
theorem rowFact_held (hWf : Claims.Json.Wf R.reads) :
    RowFact (held R B desc q c₀) (wireMap (held R B desc q c₀)) (returned (held R B desc q c₀))
      (addressOf (held R B desc q c₀)) :=
  rowFact_of_firstK (fun _ hr => wiring_of_refReads (refReadsWf_held R B desc q c₀ hWf) hr)
    (always_held R B desc q c₀) (popOk_held R B desc q c₀) B.firstK (wiresChecked_held R B desc q c₀)

end SpotFlock.Honest
