module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Properties.Challenger.Drawn

/-!
# An accepted two-stage run's coins are the device's, or a break

Each stage of an accepted two-stage run passed `Run.coinsHold` on the bytes it read (`accepted_steps`); where it read
any, the verifier holds a window, the record has a digest and the coin check finds no reason (`coinsHold_read`). Where
the check reads the challenger's draw under the parameters' key (`CheckReadsDraw`) and the evidence carries that key
(`EvidenceKeyIs`), the draw's coins are the device's
or the model's finder outputs a break (`Properties.Challenger.Drawn.drawnByDevice`), so the stage's bytes are
(`fromDevice_of_reason`, `runCoinsFromDevice`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **Coins that hold on bytes read**: where `coinsHold` passes on a non-empty read, the verifier holds a window, the
record has a digest, and the coin check finds no reason after it. -/
theorem coinsHold_read {coins record : Json} {window : Option String} {read : ByteArray}
    (h : ¬(!Spot.Run.coinsHold coins record window read) = true) (hr : 0 < read.size) :
    ∃ w after, window = some w ∧ Spot.Registration.digest record = .ok after ∧
      Spot.Challenger.coinsReason coins w after read = none := by
  have h' : Spot.Run.coinsHold coins record window read = true := by simpa using h
  have hne : read.isEmpty = false := by
    unfold ByteArray.isEmpty
    simp only [beq_eq_false_iff_ne]
    omega
  unfold Spot.Run.coinsHold at h'
  rw [hne] at h'
  cases window with
  | none => cases hd : Spot.Registration.digest record <;> simp [hd] at h'
  | some w =>
    cases hd : Spot.Registration.digest record with
    | error e => simp [hd] at h'
    | ok after => exact ⟨w, after, rfl, hd, by simpa [hd] using h'⟩

/-- **Bytes the coin check passes are the device's, or a break**, where the check reads the draw under `key`. -/
theorem fromDevice_of_reason {key : Spot.Challenger.ChallengerKey} (hc : CheckReadsDraw key)
    (dev : Properties.Challenger.History) {coins : Json} {w : String} {after read : ByteArray}
    (hk : EvidenceKeyIs key coins) (h : Spot.Challenger.coinsReason coins w after read = none) :
    FromDevice dev key w after coins read := by
  obtain ⟨label, ξ, hd, hs⟩ := hc coins w after read hk h
  refine ⟨label, ?_⟩
  rcases Properties.Challenger.Drawn.drawnByDevice dev key w label after (drawEvidence coins) ξ hd with
    ⟨s, hs', ho, he⟩ | hb
  · exact Or.inl ⟨s, hs', ho, he ▸ hs⟩
  · exact Or.inr hb

/-- **A stage's coins that hold**, read, are the device's after its record's digest, or a break. -/
theorem fromDevice_of_hold {key : Spot.Challenger.ChallengerKey} (hc : CheckReadsDraw key)
    (dev : Properties.Challenger.History) {coins record : Json} {window : Option String} {read : ByteArray}
    (hk : EvidenceKeyIs key coins) (h : ¬(!Spot.Run.coinsHold coins record window read) = true) (hr : 0 < read.size) :
    ∃ w d, window = some w ∧ Spot.Registration.digest record = .ok d ∧ FromDevice dev key w d coins read := by
  obtain ⟨w, d, hw, hd, hn⟩ := coinsHold_read h hr
  exact ⟨w, d, hw, hd, fromDevice_of_reason hc dev hk hn⟩

/-- **`RunCoinsFromDevice`**: an accepted two-stage run's stages read the device's coins after their records' digests,
or a break. -/
theorem runCoinsFromDevice (key : Spot.Challenger.ChallengerKey) : RunCoinsFromDevice key := by
  intro i it j dev ha hc hk₁ hk₂
  obtain ⟨_, workTable, resolved, draw, rest, read, -, -, hsel, hc₁, -, -, -, ns, -, ks, draw₂, read₂, -, -, -, -, -,
    hss, hc₂, -⟩ := accepted_steps ha
  exact ⟨read, read₂, ⟨resolved, workTable, draw, rest, hsel⟩, ⟨_, ns, ks, draw₂, hss⟩,
    fromDevice_of_hold hc dev hk₁ hc₁, fromDevice_of_hold hc dev hk₂ hc₂⟩

end SpotFlock.RunsAs
