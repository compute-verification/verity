module

public import Proofs.Flock.Claims.KS.ZkXferJ
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
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
import Mathlib.Topology.Sheaves.Init

/-! The masked level-0 rows' extra lanes: a verifying level-0 opening with a nonzero extra lane is off every table of
witness-lane columns, so the mixture `ZkXferJ` asks for is not the deployed prover's live image. -/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

theorem off₀Z_oracleOf_of_extra {D Sl : Type} {mPts N : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (C : Fin N → List 𝔽) (hC : ∀ q, (C q).length ≤ 2 ^ T.sch.k0) (o : OutZC 𝔽 D Sl)
    (p : Fin (2 ^ (lvOf T.sch 0).logLen)) (v : Fin (width₀ T) → 𝔽)
    (hv : op₀Z Hs.H Hs.E Hs.Lf T cap₀ o p = some v) (i : Fin (width₀ T)) (hi : 2 ^ T.sch.k0 ≤ i.val)
    (hne : v i ≠ 0) : Off₀Z Hs T cap₀ (Claims.Flock.oracleOf C) o := by
  refine ⟨p, by rw [hv]; rfl, fun h => hne ?_⟩
  rw [hv] at h
  have h1 := congrFun (Option.some.inj h) i
  rw [h1]
  show Claims.Flock.oracleOf C i.val p.val = 0
  unfold Claims.Flock.oracleOf
  split
  · exact List.getD_eq_default _ _ ((hC _).trans hi)
  · rfl

theorem offNoClashZ_oracleOf_of_extra {D Sl : Type} {mPts N : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (C : Fin N → List 𝔽) (hC : ∀ q, (C q).length ≤ 2 ^ T.sch.k0) (o : OutZC 𝔽 D Sl)
    (p : Fin (2 ^ (lvOf T.sch 0).logLen)) (v : Fin (width₀ T) → 𝔽)
    (hv : op₀Z Hs.H Hs.E Hs.Lf T cap₀ o p = some v) (i : Fin (width₀ T)) (hi : 2 ^ T.sch.k0 ≤ i.val)
    (hne : v i ≠ 0) (hsc : ¬ SelfClashZ Hs T o) : OffNoClashZ Hs T cap₀ (Claims.Flock.oracleOf C) o :=
  ⟨off₀Z_oracleOf_of_extra Hs T cap₀ C hC o p v hv i hi hne, hsc⟩

theorem committed_congr_lanes {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K]
    [DecidableEq K] {mPts : ℕ} (A : Model.Arith F K) (T : TabZK F mPts) {C C' : Model.Oracle F}
    (h : ∀ ℓ < 2 ^ T.sch.k0, ∀ q, C ℓ q = C' ℓ q) : T.Committed A C ↔ T.Committed A C' := by
  have hr : padRows T.sch.k0 0 0 T.sch.level₀.logLen C = padRows T.sch.k0 0 0 T.sch.level₀.logLen C' := by
    funext ℓ q
    refine Fin.addCases (fun ℓ => ?_) (fun i => i.elim0) ℓ
    simp only [padRows, padLane, Fin.addCases_left]
    exact h ℓ.val ℓ.isLt q.val
  unfold TabZK.Committed CommittedPad padWits padList
  rw [hr]

end FlockDeploy.KS
