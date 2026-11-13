module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Proofs.Properties.Challenger.Drawn

/-!
# An accepted two-stage run's challenges are the device's, or a break

Each stage of an accepted two-stage run got no code from `Run.coinsCode` on the bytes it read (`accepted_steps`); where
it read any, the verifier holds a key, a window and a family, the record has a digest and the challenger's check finds
no reason under them (`coinsCode_read`). The check reads the challenger's sample under that key, window and family
(`Properties.Challenger.Coins.coinsReason_draw`), and the sample's challenges are the device's or the model's finder
outputs a break (`Properties.Challenger.Drawn.drawnByDevice`), so the stage's bytes are (`fromDevice_of_reason`,
`runCoinsFromDevice`). No premise names the key: it is the run's own (`Input.key`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **A check of challenges with no code on bytes read**: where `coinsCode` gives no code on a non-empty read, the
verifier holds a key, a window and a family, the record has a digest, and the challenger's check finds no reason
under them after it. -/
theorem coinsCode_read {key : Option Spot.Challenger.ChallengerKey} {main coins record : Json}
    {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : 0 < read.size) :
    ∃ k w label after, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Registration.digest record = .ok after ∧
      Spot.Challenger.coinsReason k main coins w label after read = none := by
  have hne : read.isEmpty = false := by
    unfold ByteArray.isEmpty
    simp only [beq_eq_false_iff_ne]
    omega
  unfold Spot.Run.coinsCode at h
  rw [hne] at h
  cases hd : Spot.Registration.digest record with
  | error e => cases key <;> cases window <;> cases family <;> simp [hd] at h
  | ok after =>
    cases key with
    | none => cases window <;> cases family <;> simp at h
    | some k =>
      cases window with
      | none => cases family <;> simp at h
      | some w =>
        cases family with
        | none => simp at h
        | some label => exact ⟨k, w, label, after, rfl, rfl, rfl, rfl, by simpa [hd] using h⟩

/-- **A check of challenges with no code on no byte** carries no evidence. -/
theorem coinsCode_empty {key : Option Spot.Challenger.ChallengerKey} {main coins record : Json}
    {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : read.size = 0) :
    coins = Json.null := by
  have he : read.isEmpty = true := by
    unfold ByteArray.isEmpty
    simp only [beq_iff_eq]
    exact hr
  unfold Spot.Run.coinsCode at h
  rw [he] at h
  cases coins <;> first | rfl | simp at h

/-- **Bytes the challenger's check passes are the device's, or a break**: of the family and on the window it is handed,
under its key. -/
theorem fromDevice_of_reason (dev : Properties.Challenger.History) {key : Spot.Challenger.ChallengerKey}
    {main coins : Json} {w label : String} {after read : ByteArray}
    (h : Spot.Challenger.coinsReason key main coins w label after read = none) :
    FromDevice dev key w label after coins read := by
  obtain ⟨ξ, hd, hle, hpre⟩ := Properties.Challenger.Coins.coinsReason_draw h
  have hst : Starts read ξ := And.intro hle hpre
  rcases Properties.Challenger.Drawn.drawnByDevice dev key w label after coins ξ hd with ⟨s, hs, ho, he⟩ | hb
  · exact Or.inl ⟨s, hs, ho, he ▸ hst⟩
  · exact Or.inr hb

/-- **A stage's bytes that got no code**, read, are the device's after its record's digest, of the verifier's family, on
its window and under its key, or a break. -/
theorem fromDevice_of_code (dev : Properties.Challenger.History) {key : Option Spot.Challenger.ChallengerKey}
    {main coins record : Json} {window family : Option String} {read : ByteArray}
    (h : Spot.Run.coinsCode key main coins record window family read = none) (hr : 0 < read.size) :
    ∃ k w label d, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Registration.digest record = .ok d ∧ FromDevice dev k w label d coins read := by
  obtain ⟨k, w, label, d, hk, hw, hf, hd, hn⟩ := coinsCode_read h hr
  exact ⟨k, w, label, d, hk, hw, hf, hd, fromDevice_of_reason dev hn⟩

/-- **`RunCoinsFromDevice`**: an accepted two-stage run's stages read the device's coins after their records' digests,
or a break. -/
theorem runCoinsFromDevice : RunCoinsFromDevice := by
  intro i it j dev ha
  obtain ⟨_, workTable, resolved, draw, rest, read, -, -, hsel, hc₁, -, -, -, ns, -, ks, draw₂, read₂, -, -, -, -, -,
    hss, hc₂, -⟩ := accepted_steps ha
  exact ⟨read, read₂, ⟨resolved, workTable, draw, rest, hsel⟩, ⟨_, ns, ks, draw₂, hss⟩,
    fun hr => fromDevice_of_code dev hc₁ hr, fun hr => fromDevice_of_code dev hc₂ hr⟩

end SpotFlock.RunsAs
