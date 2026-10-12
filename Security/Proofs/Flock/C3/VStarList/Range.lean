import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Level3.Assembly

/-!
# Rulings 2 and 4: `ℓ` and the slots' units from check 5's refusals

Check 5 (`RecCoins.checkSparse`) refuses the inner shape's `sparse` entry off RecSparse's shape (`Shape.valid`,
`6 ≤ ELL`), inner points short of `ELL − 6` (`RecCoins.sparseV`) and slots off the inner circuit's unit range
(`RecCoins.layoutFactors`), and the event carries its verdict (`SparseRangeOk`). So:

* `ellOk_of_sparseRange`: `ℓ` is in the class's range `[6, k_log]` (`HoloParams.EllOk`), where `holoAt` takes it.
* `units_of_sparseRange`: where the slots' units are the unit range's (`UnitsOf`), the class keeps every one of them
  (`HoloClass.ofExec`'s filter): the range ends inside the `2^(k_log − ℓ)` units of the block.
-/

namespace FlockC3.VStarList

open FlockSoundness.Discharge.Recursion

theorem ok_of_isOk {ε α : Type} {x : Except ε α} (h : x.isOk = true) : ∃ a, x = .ok a := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl⟩

/-- **`layoutFactors` passes inside the block**: the unit range ends within the equality table of `x_rest` past its first
`ELL − 6` coordinates. -/
theorem layoutFactors_end {sh : Flock.RecSparse.Shape} {xs ys : Array Flock.F128} {rng : Flock.Range}
    (h : (Flock.RecCoins.layoutFactors sh xs ys rng).isOk = true) :
    rng.pos0 + rng.count ≤ 2 ^ (xs.size - (sh.ell - Flock.RecSparse.PRE_ROUNDS)) := by
  obtain ⟨a, h⟩ := ok_of_isOk h
  unfold Flock.RecCoins.layoutFactors at h
  dsimp only at h
  split at h
  · cases h
  rename_i hc
  have hle : rng.pos0 + rng.count ≤
      min (Flock.eqTable (xs.extract (sh.ell - Flock.RecSparse.PRE_ROUNDS) xs.size)).size
        (Flock.eqTable (ys.extract (sh.ell - Flock.RecSparse.PRE_ROUNDS) ys.size)).size := by
    by_contra hn
    exact hc (by simp [Nat.lt_of_not_le hn])
  simp only [FlockLevel3.eqTable_size, Array.size_extract, min_self] at hle
  exact hle.trans (min_le_left _ _)

/-- **`ℓ` is in the class's range**: RecSparse's shape check gives `6 ≤ ℓ`, and points of `k_log − 6` coordinates not
short of `ℓ − 6` give `ℓ ≤ k_log`. -/
theorem ellOk_of_sparseRange {ish : Flock.RecVStar.Inner} {st : Flock.Stmt} {hp : HoloParams}
    (h : SparseRangeOk ish st) (hsp : SparseIs ish hp) : hp.EllOk (execClassAt st).cls.kLog := by
  obtain ⟨hv, hs, -⟩ := h _ hsp
  have h6 := (execClassAt st).cls.hk6
  simp only [Flock.RecSparse.Shape.valid, Flock.RecSparse.PRE_ROUNDS, Bool.and_eq_true, decide_eq_true_eq] at hv
  simp only [Array.size_replicate, Flock.RecSparse.PRE_ROUNDS] at hs
  refine ⟨?_, ?_⟩ <;> omega

/-- **The class keeps the slots' units**: at the unit range's units (`UnitsOf`), every slot's unit is inside the block,
so `HoloClass.ofExec`'s filter is the identity. -/
theorem units_of_sparseRange {ish : Flock.RecVStar.Inner} {st : Flock.Stmt} {hp : HoloParams}
    (h : SparseRangeOk ish st) (hsp : SparseIs ish hp) (hu : UnitsOf st hp) (σ : Fin (2 ^ hp.sLog)) :
    (holoAt st hp).cls.units σ = hp.units σ := by
  have hell := ellOk_of_sparseRange h hsp
  obtain ⟨-, -, rng, hr, hl⟩ := h _ hsp
  obtain ⟨rng', hr', hσ⟩ := hu
  obtain rfl : rng = rng' := Except.ok.inj (hr.symm.trans hr')
  have hend := layoutFactors_end hl
  have h6 := hell.1
  have he : (execClassAt st).cls.kLog - 6 - (hp.ell - 6) = (execClassAt st).cls.kLog - hp.ell := by omega
  simp only [Array.size_replicate, Flock.RecSparse.PRE_ROUNDS, he] at hend
  show ((hp.units σ).filter fun u => u < 2 ^ ((execClassAt st).cls.kLog -
    (if hp.EllOk (execClassAt st).cls.kLog then hp.ell else 6))) = hp.units σ
  rw [if_pos hell]
  apply Finset.filter_true_of_mem
  intro u hu'
  rw [hσ σ] at hu'
  split_ifs at hu' with hc
  · obtain rfl := Finset.mem_singleton.1 hu'
    exact lt_of_lt_of_le (Nat.add_lt_add_left hc _) hend
  · simp at hu'

end FlockC3.VStarList
