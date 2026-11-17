module

public import Proofs.Flock.SpotFlock.AddressFacts

/-! The five facts of `AddressFacts` at one address map give `AddressesWired`, and at the code's own wiring the premise
`AddressesConnected`. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held} {ω : ℕ → ℕ} {keep : ℕ → Bool}

theorem CheckedRun.addressHoldsWired_of_facts {r : CheckedRun h.stage} {S : Address} {i g : ℕ}
    (hw : r.WidthAt h S) (hc : r.CopyAt h S i) (hrow : r.RowHolds h ω keep S g)
    (he : r.EntryHolds h ω keep S g) (ht : r.RootsHold h ω keep S g) : r.AddressHoldsWired h ω keep S i g := by
  obtain ⟨hn, hreg₁, hint₁⟩ := hrow
  obtain ⟨hreg₂, hint₂⟩ := he
  obtain ⟨hreg₃, hint₃⟩ := ht
  refine ⟨hn, ⟨hw, hc⟩, ?_⟩
  cases hq : regNamed r.ins r.sts (S.port r.u) with
  | some reg =>
    obtain ⟨hfirst, hkey, hsch⟩ := hreg₂ reg hq
    exact Or.inl ⟨reg, hq, hfirst, hreg₃ reg hq, hkey, hsch, hreg₁ reg hq⟩
  | none =>
    obtain ⟨hlt, hkey, hsch⟩ := hint₂ hq
    obtain ⟨hone, hleaf⟩ := hint₁ hq
    exact Or.inr ⟨hq, hone, hlt, hint₃ hq, hkey, fun pop hp => ⟨hsch pop hp, hleaf pop hp⟩⟩

theorem addressesWired_of_facts {addr : ℕ → Address} (hw : WidthFact h addr) (hc : CopyFact h addr)
    (hr : RowFact h ω keep addr) (he : EntryFact h ω keep addr) (ht : OneTreeFact h ω keep addr) :
    AddressesWired h ω keep := by
  obtain ⟨hshare, hroots⟩ := ht
  exact ⟨addr, hshare, fun r hok i hi =>
    CheckedRun.addressHoldsWired_of_facts (hw r hok i hi) (hc r hok i hi) (hr r hok i hi) (he r hok i hi)
      (hroots r hok i hi)⟩

variable (h) in
theorem addressesConnected_of_facts
    (hf : sourcesNamed h = true → ∃ addr : ℕ → Address, WidthFact h addr ∧ CopyFact h addr ∧
      RowFact h (wireMap h) (returned h) addr ∧ EntryFact h (wireMap h) (returned h) addr ∧
      OneTreeFact h (wireMap h) (returned h) addr) : AddressesConnected h := by
  intro hn
  obtain ⟨addr, hw, hc, hr, he, ht⟩ := hf hn
  exact addressesWired_of_facts hw hc hr he ht

end SpotFlock
