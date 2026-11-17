module

public import Proofs.Flock.Soundness.ExecFlatCopies
import Mathlib.Combinatorics.Matroid.Init

/-! The two cases in which Δ copies a forced-zero row into a unit copy's input bit (`ZeroPort`, `flatSrc_zero`). -/

@[expose] public section

open Flock

namespace FlockSoundness.Discharge.Zero

def ZeroPort (c : Circuit) (s₀ p t : ℕ) : Prop :=
  (p < (c.unit.inGroups.getD 0 default).bits.size ∧ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0 ∨
    ¬ 16 * (((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat -
          (Typed.starts c)[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat)).getD 0]!) + t <
      c.ports[((List.range (Typed.starts c).size).reverse.find?
        (fun q => (Typed.starts c)[q]! ≤ ((c.leavesIn.getD (s₀ % c.upv) #[])[p]!).toNat)).getD 0]!.bits)) ∨
    ((c.unit.inGroups.getD 0 default).bits.size ≤ p ∧
      ∃ e ∈ (c.leafCuts.toList.filter fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)).head?,
        ¬ (t < 16 ∧ 16 * (e.2.2 - (Typed.starts c)[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0]!) + t <
          c.ports[((List.range (Typed.starts c).size).reverse.find?
            (fun q => (Typed.starts c)[q]! ≤ e.2.2)).getD 0]!.bits))

def ZeroAt (c : Circuit) (s₀ off : ℕ) : Prop :=
  ZeroPort c s₀ (portOf (portCols c.unit.inGroups) off)
    (off - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) off]!.1)

theorem flatSrc_zero {c : Circuit} {s₀ p t : ℕ} (h : ZeroPort c s₀ p t) :
    Flat.flatSrc c s₀ p t = c.unitRange.slot s₀ + c.unit.useful ∨
      Flat.flatSrc c s₀ p t = c.unitRange.slot 0 + c.unit.useful := by
  unfold Flat.flatSrc
  rcases h with ⟨hp, hl | hb⟩ | ⟨hp, e, he, hz⟩
  · rw [ite_eq_left hp]
    left
    unfold Flat.leafSrc
    rw [ite_eq_left hl]
  · rw [ite_eq_left hp]
    left
    unfold Flat.leafSrc
    by_cases hl : (c.leavesIn.getD (s₀ % c.upv) #[])[p]! < 0
    · rw [ite_eq_left hl]
    · simp only [hl, hb, ↓reduceIte]
  · rw [ite_eq_right (by omega)]
    rcases hc : c.leafCuts.toList.filter (fun e => decide (e.1 = s₀ % c.upv ∧ e.2.1 = p)) with _ | ⟨e', rest⟩
    · rw [hc] at he
      cases he
    · rw [hc] at he
      cases he
      right
      simp only [hc, Flat.cutSrc]
      rw [ite_eq_right (by simpa only [Bool.and_eq_true, decide_eq_true_eq] using hz)]

theorem flatSrcAt_zero {c : Circuit} {s₀ off : ℕ} (h : ZeroAt c s₀ off) :
    flatSrcAt c s₀ off = c.unitRange.slot s₀ + c.unit.useful ∨
      flatSrcAt c s₀ off = c.unitRange.slot 0 + c.unit.useful :=
  flatSrc_zero h

end FlockSoundness.Discharge.Zero
