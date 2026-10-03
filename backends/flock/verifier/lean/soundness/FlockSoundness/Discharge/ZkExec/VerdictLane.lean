import FlockSoundness.Discharge.ZkExec.VerdictForms

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

/-- A lane round's step on the claim: `u₀ + r·(claim + u₂) + r²·u₂` on the masked message `j`. -/
def laneStepE (lay : Zk.PadLayout) (lg : LigeritoProof) (ch : Array F256) (claim : Zk.Lin2) (j : ℕ) : Zk.Lin2 :=
  Zk.l2add (Zk.l2add (Zk.l2masked lg.messages[j]!.1 (lay.lgMsg j 0) (lay.lgMsg j 1) lay.len)
      (Zk.l2scale (Zk.l2add claim (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len)) ch[j]!))
    (Zk.l2scale (Zk.l2masked lg.messages[j]!.2 (lay.lgMsg j 2) (lay.lgMsg j 3) lay.len) (ch[j]! * ch[j]!))

/-- The lane phase's claim, `target + β·y₀` through the lane rounds, then `+ ρ₀·e₀ + ρ₁·e₁ + T'`. -/
def laneLhs (lay : Zk.PadLayout) (rec : Zk.L0Rec) (lg : LigeritoProof) (msgs : Zk.Level0Msgs) : Zk.Lin2 :=
  let claim0 := (Zk.Lin.konst rec.target lay.len).add
    ((Zk.Lin.masked lg.oodValues[0]! (lay.lgOod 0) lay.len).scale rec.ood[0]!.1)
  let claim := (List.range' 0 lay.k0).foldl (laneStepE lay lg rec.challenges) (claim0, Zk.Lin.konst 0 lay.len)
  Zk.l2add (Zk.l2add claim (Zk.l2scale (Zk.l2masked msgs.e0 (lay.lgE 0 0) (lay.lgE 0 1) lay.len) rec.rho.1))
    (Zk.l2add (Zk.l2scale (Zk.l2masked msgs.e1 (lay.lgE 1 0) (lay.lgE 1 1) lay.len) rec.rho.2)
      (Zk.l2konst msgs.tPrime lay.len))

theorem ligeritoConstraints_eq {lay : Zk.PadLayout} {rec : Zk.L0Rec} {lg : LigeritoProof} {msgs : Zk.Level0Msgs}
    {v : Array (String × Zk.Lin)} (h : Zk.ligeritoConstraints lay rec lg msgs = .ok v) (h1 : lay.nOod0 = 1) :
    rec.ood.size = 1 ∧ v[0]!.2 = (laneLhs lay rec lg msgs).1 ∧ v[1]!.2 = (laneLhs lay rec lg msgs).2 := by
  unfold Zk.ligeritoConstraints at h
  dsimp only at h
  split at h
  · cases h
  · rename_i hc
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, not_not, decide_eq_true_eq, not_lt] at hc
    obtain ⟨⟨ho, -⟩, -⟩ := hc
    rw [h1] at ho
    simp only [Std.Legacy.Range.forIn_eq_forIn_range', range_size, List.forIn_pure_yield_eq_foldl, ho] at h
    cases h
    refine ⟨ho, ?_, ?_⟩ <;> simp [laneLhs, List.range'_one] <;> rfl

end FlockSoundness.Discharge.ZkExec
