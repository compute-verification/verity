module

public import Proofs.Flock.SpotFlock.AddressFacts

/-!
# The five facts give the address premise

The five facts of `AddressFacts`, at one address map, give `AddressesWired` (`addressesWired_of_facts`): at each
accepted run's checked unit and each gate it needs, the address holds (`CheckedRun.addressHoldsWired_of_facts`), read
as a registered value where a registered read is named like its port and off the interior record where none is. At
the code's own wiring map, returning the gates the boundary record holds, they give the premise
`AddressesConnected` (`addressesConnected_of_facts`), which asks the property only of a held code whose every source
bit is at a port a link or a wire names.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable {h : Held} {ω : ℕ → ℕ} {keep : ℕ → Bool}

/-- **The five facts at one address give that the address holds**, at the unit's gate `i` and the wired program's
gate `g`: as a registered read where one is named like the address's port, else off the interior record. -/
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

/-- **The five facts at one address map give the property at the wired program.** -/
theorem addressesWired_of_facts {addr : ℕ → Address} (hw : WidthFact h addr) (hc : CopyFact h addr)
    (hr : RowFact h ω keep addr) (he : EntryFact h ω keep addr) (ht : OneTreeFact h ω keep addr) :
    AddressesWired h ω keep := by
  obtain ⟨hshare, hroots⟩ := ht
  exact ⟨addr, hshare, fun r hok i hi =>
    CheckedRun.addressHoldsWired_of_facts (hw r hok i hi) (hc r hok i hi) (hr r hok i hi) (he r hok i hi)
      (hroots r hok i hi)⟩

variable (h) in
/-- **The five facts at the code's own wiring give the premise**: where every source bit is at a port a link or a
wire names, one address map with the five facts at the code's wiring map (`wireMap h`), returning the gates the
boundary record holds (`returned h`), gives `AddressesConnected h`. -/
theorem addressesConnected_of_facts
    (hf : sourcesNamed h = true → ∃ addr : ℕ → Address, WidthFact h addr ∧ CopyFact h addr ∧
      RowFact h (wireMap h) (returned h) addr ∧ EntryFact h (wireMap h) (returned h) addr ∧
      OneTreeFact h (wireMap h) (returned h) addr) : AddressesConnected h := by
  intro hn
  obtain ⟨addr, hw, hc, hr, he, ht⟩ := hf hn
  exact addressesWired_of_facts hw hc hr he ht

end SpotFlock
