import Definitions.Core.TC.Bridge
import Proofs.Core.TC.Proofs.Group

/-!
# One group, the semantics' side

`groupSum_eq` is `group_sum` on the Hopper adder in terms of the relation's total `T` and maximum exponent `E`.
`finResult_admissible`: its finite result is `Admissible` (`Verity.Protocol.TC.Bridge`).
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- `group_sum` on the Hopper adder (`width = 26`, floor `-133`), with its folds named. -/
def groupSumH (ts : List Term) : Term :=
  let mx := ts.foldl maxStep (-133)
  let total := ts.foldl (totStep mx) 0
  if total = 0 then ZERO
  else
    let bl := bitLength total.natAbs
    let exponent : Int := mx + bl - 26
    let magnitude := if bl > 26 then total.natAbs >>> (bl - 26) else total.natAbs <<< (26 - bl)
    let magnitude' := if exponent < -126 then magnitude >>> (-126 - exponent).toNat else magnitude
    let exponent' : Int := if exponent < -126 then -126 else exponent
    let magnitude'' := magnitude' >>> 2
    if magnitude'' = 0 then ZERO else ⟨decide (total < 0), exponent', magnitude''⟩

theorem groupSum_H (ts : List Term) : groupSum ts 26 (-133) = groupSumH ts := rfl

/-- The finite result a group's total and maximum exponent give. -/
def finResult (T E : Int) : Term :=
  if T = 0 then ZERO else if mag24Of T E = 0 then ZERO else ⟨decide (T < 0), eFin T E, mag24Of T E⟩

theorem groupSum_eq (ts : List Term) (xs : List TermIn) (hc : Corr (-133) ts xs)
    (hT : ((xs.map (contrib (maxE (-133) xs))).sum).natAbs < 2 ^ 32) :
    groupSum ts 26 (-133) = finResult (xs.map (contrib (maxE (-133) xs))).sum (maxE (-133) xs) := by
  rw [groupSum_H]
  unfold groupSumH
  have hmx : ts.foldl maxStep (-133) = maxE (-133) xs := maxExp_eq _ ts xs _ hc (Int.le_refl _)
  have htot : ts.foldl (totStep (maxE (-133) xs)) 0 = (xs.map (contrib (maxE (-133) xs))).sum := by
    rw [total_eq _ _ ts xs 0 hc (le_maxE xs (-133)).2]; simp
  simp only [hmx, htot]
  generalize maxE (-133) xs = E at *
  generalize (xs.map (contrib E)).sum = T at *
  unfold finResult
  split
  · rfl
  · rename_i hT0
    have hN0 : T.natAbs ≠ 0 := by omega
    obtain ⟨hw, _, hhi⟩ := window T.natAbs hN0 hT
    rw [hw] at hhi
    have hfs := final_shift _ hhi (E + bitLength T.natAbs - 26)
    have hm : mag24Of T E =
        ((if E + ↑(bitLength T.natAbs) - 26 < -126 then
            (if bitLength T.natAbs > 26 then T.natAbs >>> (bitLength T.natAbs - 26)
              else T.natAbs <<< (26 - bitLength T.natAbs)) >>> (-126 - (E + ↑(bitLength T.natAbs) - 26)).toNat
          else (if bitLength T.natAbs > 26 then T.natAbs >>> (bitLength T.natAbs - 26)
              else T.natAbs <<< (26 - bitLength T.natAbs))) >>> 2) := by
      rw [hfs, mag24Of, magWOf, hw, eFin, eOut]
    have he : eFin T E = if E + ↑(bitLength T.natAbs) - 26 < -126 then -126 else E + ↑(bitLength T.natAbs) - 26 := by
      unfold eFin eOut; split <;> omega
    rw [← hm, ← he]

theorem finResult_admissible (T E : Int) (hT : T.natAbs < 2 ^ 32) (hov : ¬ eFin T E > 127) :
    Admissible (.fin (finResult T E)) := by
  unfold finResult
  split
  · simp [Admissible, ZERO]
  · split
    · simp [Admissible, ZERO]
    · rename_i hT0 hm0
      have hN0 : T.natAbs ≠ 0 := by omega
      obtain ⟨_, hlo, hhi⟩ := window T.natAbs hN0 hT
      simp only [Admissible]
      unfold mag24Of magWOf at *
      generalize T.natAbs * 2 ^ (32 - bitLength T.natAbs) / 2 ^ 6 = m at *
      have hsh : 2 ≤ (min (eFin T E - eOut T E + 2) 36).toNat := by unfold eFin; omega
      have hlt : m / 2 ^ (min (eFin T E - eOut T E + 2) 36).toNat < 2 ^ 24 := by
        apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
        have : 2 ^ 2 ≤ 2 ^ (min (eFin T E - eOut T E + 2) 36).toNat := Nat.pow_le_pow_right (by decide) hsh
        have : 2 ^ 26 ≤ 2 ^ 24 * 2 ^ (min (eFin T E - eOut T E + 2) 36).toNat := by
          calc 2 ^ 26 = 2 ^ 24 * 2 ^ 2 := by decide
            _ ≤ _ := Nat.mul_le_mul_left _ this
        omega
      refine ⟨hlt, fun _ => ⟨by unfold eFin; omega, by omega, ?_⟩⟩
      by_cases hsub : eOut T E < -126
      · right; unfold eFin; omega
      · left
        have : (min (eFin T E - eOut T E + 2) 36).toNat = 2 := by unfold eFin; omega
        rw [this]
        exact (Nat.le_div_iff_mul_le (by decide)).mpr (by omega)

end Verity.TC.Proofs
