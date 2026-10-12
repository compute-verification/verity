import Proofs.Core.TC.Ampere.Group
import Proofs.Core.TC.Proofs.SemGroup

/-!
# One Ampere group, the semantics' side

`group_sum` on the Ampere adder in terms of the relation's total and maximum exponent (`groupSum_eq`).
-/

namespace Verity.TC.Proofs.Ampere

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- `group_sum` on the Ampere adder (`width = 25`, floor `-132`), with its folds named. -/
def groupSumH (ts : List Term) : Term :=
  let mx := ts.foldl maxStep (-132)
  let total := ts.foldl (totStep mx) 0
  if total = 0 then ZERO
  else
    let bl := bitLength total.natAbs
    let exponent : Int := mx + bl - 25
    let magnitude := if bl > 25 then total.natAbs >>> (bl - 25) else total.natAbs <<< (25 - bl)
    let magnitude' := if exponent < -126 then magnitude >>> (-126 - exponent).toNat else magnitude
    let exponent' : Int := if exponent < -126 then -126 else exponent
    let magnitude'' := magnitude' >>> 1
    if magnitude'' = 0 then ZERO else ⟨decide (total < 0), exponent', magnitude''⟩

theorem groupSum_H (ts : List Term) : groupSum ts 25 (-132) = groupSumH ts := rfl

/-- The finite result a group's total and maximum exponent give. -/
def finResult (T E : Int) : Term :=
  if T = 0 then ZERO else if mag24Of T E = 0 then ZERO else ⟨decide (T < 0), eFin T E, mag24Of T E⟩

theorem groupSum_eq (ts : List Term) (xs : List TermIn) (hc : Corr (-132) ts xs)
    (hT : ((xs.map (contrib (maxE (-132) xs))).sum).natAbs < 2 ^ 30) :
    groupSum ts 25 (-132) = finResult (xs.map (contrib (maxE (-132) xs))).sum (maxE (-132) xs) := by
  rw [groupSum_H]
  unfold groupSumH
  have hmx : ts.foldl maxStep (-132) = maxE (-132) xs := maxExp_eq _ ts xs _ hc (Int.le_refl _)
  have htot : ts.foldl (totStep (maxE (-132) xs)) 0 = (xs.map (contrib (maxE (-132) xs))).sum := by
    rw [total_eq _ _ ts xs 0 hc (le_maxE xs (-132)).2]; simp
  simp only [hmx, htot]
  generalize maxE (-132) xs = E at *
  generalize (xs.map (contrib E)).sum = T at *
  unfold finResult
  split
  · rfl
  · rename_i hT0
    have hN0 : T.natAbs ≠ 0 := by omega
    obtain ⟨hw, _, hhi⟩ := window T.natAbs hN0 hT
    rw [hw] at hhi
    have hfs := final_shift _ hhi (E + bitLength T.natAbs - 25)
    have hm : mag24Of T E =
        ((if E + ↑(bitLength T.natAbs) - 25 < -126 then
            (if bitLength T.natAbs > 25 then T.natAbs >>> (bitLength T.natAbs - 25)
              else T.natAbs <<< (25 - bitLength T.natAbs)) >>> (-126 - (E + ↑(bitLength T.natAbs) - 25)).toNat
          else (if bitLength T.natAbs > 25 then T.natAbs >>> (bitLength T.natAbs - 25)
              else T.natAbs <<< (25 - bitLength T.natAbs))) >>> 1) := by
      rw [hfs, mag24Of, magWOf, hw, eFin, eOut]
    have he : eFin T E = if E + ↑(bitLength T.natAbs) - 25 < -126 then -126 else E + ↑(bitLength T.natAbs) - 25 := by
      unfold eFin eOut; split <;> omega
    rw [← hm, ← he]

theorem finResult_admissible (T E : Int) (hT : T.natAbs < 2 ^ 30) (hov : ¬ eFin T E > 127) :
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
      generalize T.natAbs * 2 ^ (30 - bitLength T.natAbs) / 2 ^ 5 = m at *
      have hsh : 1 ≤ (min (eFin T E - eOut T E + 1) 35).toNat := by unfold eFin; omega
      have hlt : m / 2 ^ (min (eFin T E - eOut T E + 1) 35).toNat < 2 ^ 24 := by
        apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
        have : 2 ^ 1 ≤ 2 ^ (min (eFin T E - eOut T E + 1) 35).toNat := Nat.pow_le_pow_right (by decide) hsh
        have : 2 ^ 25 ≤ 2 ^ 24 * 2 ^ (min (eFin T E - eOut T E + 1) 35).toNat := by
          calc 2 ^ 25 = 2 ^ 24 * 2 ^ 1 := by decide
            _ ≤ _ := Nat.mul_le_mul_left _ this
        omega
      refine ⟨hlt, fun _ => ⟨by unfold eFin; omega, by omega, ?_⟩⟩
      by_cases hsub : eOut T E < -126
      · right; unfold eFin; omega
      · left
        have : (min (eFin T E - eOut T E + 1) 35).toNat = 1 := by unfold eFin; omega
        rw [this]
        exact (Nat.le_div_iff_mul_le (by decide)).mpr (by omega)

end Verity.TC.Proofs.Ampere
