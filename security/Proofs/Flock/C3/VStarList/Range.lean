module

public import Proofs.Flock.C3.VStarList.Shape
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Flock.Level3.Assembly

/-! `ℓ` and the slots' units from `RecCoins.checkSparse`'s refusals. -/

@[expose] public section

namespace FlockC3.VStarList

open FlockSoundness.Discharge.Recursion

theorem ok_of_isOk {ε α : Type} {x : Except ε α} (h : x.isOk = true) : ∃ a, x = .ok a := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl⟩

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

theorem execClassAt_kLog {st : Flock.Stmt} (h : ClassOk (shapeOf st)) : (execClassAt st).cls.kLog = st.c.kLog := by
  rw [execClassAt, dite_eq_left h]
  rfl

theorem sparseRangeOk_of_check {ish : Flock.RecVStar.Inner} {st : Flock.Stmt} (hc : ClassOk (shapeOf st))
    (h : Flock.RecCoins.checkInnerRange ish st = .ok ()) :
    SparseRangeOk ish st ∧ (Flock.RecVStar.checkInnerStmt ish st).isOk = true := by
  obtain ⟨hs, hr⟩ := checkInnerRange_spec h
  refine ⟨fun sp hsp => ?_, by rw [hs]; rfl⟩
  rw [execClassAt_kLog hc]
  exact hr sp hsp

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
