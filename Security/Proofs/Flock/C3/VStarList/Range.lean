import Proofs.Flock.C3.VStarList.Shape
import Proofs.Flock.Level3.Assembly

/-!
# Rulings 2 and 4: `ℓ` and the slots' units from check 5's refusals

Check 5 (`RecCoins.checkSparse`) refuses the inner shape's `sparse` entry off RecSparse's shape (`Shape.valid`,
`6 ≤ ELL`), inner points short of `ELL − 6` (`RecCoins.sparseV`) and slots off the inner circuit's unit range
(`RecCoins.layoutFactors`), and the event carries its verdict (`SparseRangeOk`). So:

* `sparseRangeOk_of_check`: the verifier evaluates that verdict on the inner statement at every candidate-3 session's
  setup (`RecCoins.checkAtC3`, which sets the statement up from its files, `checkAtC3_spec`, and runs
  `RecCoins.checkInnerRange`, `checkInnerRange_spec`), so where the statement's class checks pass (`execClassAt_kLog`)
  its acceptance is `SparseRangeOk`, and the inner shape is candidate 3's at the statement (`checkInnerStmt`).
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

/-- **`checkInnerRange` accepting**: the inner shape is candidate 3's at the statement, and at its RecSparse shape, which
is valid, zero points of the statement's `k_log − 6` coordinates are not short of `ELL − 6` and give layout factors at
its unit range. -/
theorem checkInnerRange_spec {i : Flock.RecVStar.Inner} {st : Flock.Stmt}
    (h : Flock.RecCoins.checkInnerRange i st = .ok ()) :
    Flock.RecVStar.checkInnerStmt i st = .ok () ∧ ∀ sp, i.sparse = some sp →
      sp.valid = true ∧ ¬ (Array.replicate (st.c.kLog - 6) (0 : Flock.F128)).size < sp.ell - Flock.RecSparse.PRE_ROUNDS ∧
        ∃ rng, Flock.RecCoins.unitRangeOf st = .ok rng ∧
          (Flock.RecCoins.layoutFactors sp (Array.replicate (st.c.kLog - 6) (0 : Flock.F128))
            (Array.replicate (st.c.kLog - 6) (0 : Flock.F128)) rng).isOk = true := by
  unfold Flock.RecCoins.checkInnerRange at h
  cases hc : Flock.RecVStar.checkInnerStmt i st with
  | error e => rw [hc] at h; cases h
  | ok u =>
    refine ⟨rfl, fun sp hsp => ?_⟩
    rw [hc] at h
    rw [Array.size_replicate]
    simp only [bind, Except.bind, hsp, Array.size_replicate] at h
    by_cases hv : sp.valid = true
    · by_cases hs : st.c.kLog - 6 < sp.ell - Flock.RecSparse.PRE_ROUNDS
      · simp [hv, hs, throw, throwThe, MonadExceptOf.throw] at h
      · refine ⟨hv, hs, ?_⟩
        cases hr : Flock.RecCoins.unitRangeOf st with
        | error e => simp [hv, hs, hr, Except.mapError] at h
        | ok rng =>
          refine ⟨rng, rfl, ?_⟩
          cases hl : Flock.RecCoins.layoutFactors sp (Array.replicate (st.c.kLog - 6) (0 : Flock.F128))
            (Array.replicate (st.c.kLog - 6) (0 : Flock.F128)) rng with
          | error e => simp [hv, hs, hr, hl, Except.mapError, throw, throwThe, MonadExceptOf.throw] at h
          | ok _ => rfl
    · simp [hv, throw, throwThe, MonadExceptOf.throw] at h

/-- **`checkAtC3` accepting at candidate 3's inner shape**: the verifier holds the inner statement's files, the inner
statement `st` sets up from them, and `checkInnerRange` accepts it. -/
theorem checkAtC3_spec {i : Flock.RecVStar.Inner} {src : Option Flock.RecCoins.Source}
    (h : Flock.RecCoins.checkAtC3 (some i) src = .ok ()) (hs : i.sparse.isSome = true) :
    ∃ f spec st, src = some (.files f) ∧ f.stmt "the inner session" = .ok (spec, st) ∧
      Flock.RecCoins.checkInnerRange i st = .ok () := by
  have hn : i.sparse.isNone = false := by cases hsp : i.sparse <;> simp_all
  unfold Flock.RecCoins.checkAtC3 at h
  simp only [hn, Bool.false_eq_true, ↓reduceIte] at h
  rcases src with _ | (f | ⟨j, r⟩)
  · cases h
  · cases hf : f.stmt "the inner session" with
    | error e => simp [hf, bind, Except.bind] at h
    | ok x =>
      obtain ⟨spec, st⟩ := x
      simp only [hf, bind, Except.bind] at h
      exact ⟨f, spec, st, rfl, hf, h⟩
  · cases h

/-- `execClassAt`'s `k_log` is the statement's where its class checks pass. -/
theorem execClassAt_kLog {st : Flock.Stmt} (h : ClassOk (shapeOf st)) : (execClassAt st).cls.kLog = st.c.kLog := by
  rw [execClassAt, dite_eq_left h]
  rfl

/-- **The verifier's range check gives `SparseRangeOk`**: where the inner statement's class checks pass (at an accepted
statement, `classOk_of_accepts`), `checkInnerRange` accepting is check 5's sparse refusals' verdict at zero points of the
class's `k_log` (`execClassAt_kLog`), and the inner shape is candidate 3's at the statement. -/
theorem sparseRangeOk_of_check {ish : Flock.RecVStar.Inner} {st : Flock.Stmt} (hc : ClassOk (shapeOf st))
    (h : Flock.RecCoins.checkInnerRange ish st = .ok ()) :
    SparseRangeOk ish st ∧ (Flock.RecVStar.checkInnerStmt ish st).isOk = true := by
  obtain ⟨hs, hr⟩ := checkInnerRange_spec h
  refine ⟨fun sp hsp => ?_, by rw [hs]; rfl⟩
  rw [execClassAt_kLog hc]
  exact hr sp hsp

/-- **`ℓ` is in the class's range**: RecSparse's shape check gives `6 ≤ ℓ`, and points of `k_log − 6` coordinates not
short of `ℓ − 6` give `ℓ ≤ k_log`. -/
theorem ellOk_of_sparseRange {ish : Flock.RecVStar.Inner} {st : Flock.Stmt} {hp : HoloParams}
    (h : SparseRangeOk ish st) (hsp : SparseIs ish hp) : hp.EllOk (execClassAt st).cls.kLog := by
  obtain ⟨hv, hs, -⟩ := h _ hsp
  have h6 := (execClassAt st).cls.hk6
  have hv' : (decide (1 ≤ hp.nu) && decide (6 ≤ hp.ell) && decide (hp.tLog ≤ 16)) = true := hv
  have hs' : ¬ (execClassAt st).cls.kLog - 6 < hp.ell - 6 := by
    rw [Array.size_replicate] at hs
    exact hs
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hv'
  exact ⟨hv'.1.2, by omega⟩

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
  rw [ite_eq_left hell]
  apply Finset.filter_true_of_mem
  intro u hu'
  rw [hσ σ] at hu'
  split_ifs at hu' with hc
  · obtain rfl := Finset.mem_singleton.1 hu'
    exact lt_of_lt_of_le (Nat.add_lt_add_left hc _) hend
  · simp at hu'

end FlockC3.VStarList
