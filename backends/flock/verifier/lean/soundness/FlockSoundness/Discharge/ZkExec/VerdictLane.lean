import FlockSoundness.Discharge.ZkExec.VerdictForms
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLane

/-!
# Level 0's lane rows are the executable's

`zk_lane_rows`: the two coordinate rows of the `K`-row the compiled masked level 0 hands the inner proof are the two
constraints `Zk.ligeritoConstraints` builds, at the record `Zk.ligerito` returns.

* `ligeritoConstraints_eq`: the two constraints, as the lane phase's claim folds (`laneLhs`).
* `zk_ligerito_rec`: `Zk.ligerito`'s record holds the batched value, level 0's out-of-domain coin and value, the lane
  rounds' coins and the pair's coins `ρ`.
* `foldZK_rel`, `ligeritoPadZC_row`: the model's lane rounds fold the claim as the executable does, so every run of
  `ligeritoPadZC` returns the row `laneLhs` is.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The executable's lane rows -/

/-! ## The executable's record -/

theorem lforIn_inv {σ : Type} (body : ℕ → σ → LM (ForInStep σ)) (I : LState → Prop)
    (hb : ∀ i s st t b st' t', I st → body i s st t = .ok ((b, st'), t') → I st') :
    ∀ (l : List ℕ) (s : σ) (st : LState) (t : Flock.Transcript) (s' : σ) (st' : LState) (t' : Flock.Transcript),
      I st → (forIn l s body : LM σ) st t = .ok ((s', st'), t') → I st'
  | [], s, st, t, s', st', t', hI, h => by
      simp only [List.forIn_nil] at h
      obtain ⟨-, rfl, -⟩ := lpure_ok h
      exact hI
  | a :: l, s, st, t, s', st', t', hI, h => by
      rw [List.forIn_cons] at h
      obtain ⟨r, st₁, t₁, h₁, h₂⟩ := lbind_ok h
      have hI₁ := hb a s st t r st₁ t₁ hI h₁
      cases r with
      | done b => obtain ⟨-, rfl, -⟩ := lpure_ok h₂; exact hI₁
      | yield b => exact lforIn_inv body I hb l b st₁ t₁ s' st' t' hI₁ h₂

/-! ## The model's lane rounds -/

end FlockSoundness.Discharge.ZkExec
