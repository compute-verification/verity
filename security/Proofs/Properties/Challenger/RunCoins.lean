module

public import Proofs.Flock.SpotFlock.RunsAs.Coins
public import Proofs.Properties.Challenger.Drawn

/-! An accepted two-stage run's challenges are the device's, or a break (`RunCoinsFromDevice`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem fromDevice_of_reason (dev : Properties.Challenger.History) {key : Spot.Challenger.ChallengerKey}
    {main coins : Json} {w label : String} {after read : ByteArray}
    (h : Spot.Challenger.coinsReason key main coins w label after read = none) :
    FromDevice dev key w label after coins read := by
  obtain ⟨ξ, hd, hle, hpre⟩ := Properties.Challenger.Coins.coinsReason_sample h
  have hst : Starts read ξ := And.intro hle hpre
  rcases Properties.Challenger.Drawn.sampledByDevice dev key w label after coins ξ hd with ⟨s, hs, ho, he⟩ | hb
  · exact Or.inl ⟨s, hs, ho, he ▸ hst⟩
  · exact Or.inr hb

theorem fromDevice_of_code (dev : Properties.Challenger.History) {key : Option Spot.Challenger.ChallengerKey}
    {main coins record : Json} {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : 0 < read.size) :
    ∃ k w label d, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Docket.digest record = .ok d ∧ FromDevice dev k w label d coins read := by
  obtain ⟨k, w, label, d, hk, hw, hf, hd, hn⟩ := coinsCode_read h hr
  exact ⟨k, w, label, d, hk, hw, hf, hd, fromDevice_of_reason dev hn⟩

theorem runCoinsFromDevice : RunCoinsFromDevice := by
  intro i it j dev ha
  obtain ⟨_, workTable, resolved, draw, rest, read, -, -, hsel, hc₁, -, -, -, ns, -, ks, draw₂, read₂, -, -, -, -, -,
    hss, hc₂, -⟩ := accepted_steps ha
  exact ⟨read, read₂, ⟨resolved, workTable, draw, rest, hsel⟩, ⟨_, ns, ks, draw₂, hss⟩,
    fun hr => fromDevice_of_code dev hc₁ hr, fun hr => fromDevice_of_code dev hc₂ hr⟩

end SpotFlock.RunsAs
