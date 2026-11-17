module

public import Batteries.Data.ByteArray
public import Flock.ZkServer.Finish
public import Proofs.Flock.Claims.ZK.FinishStop
public import Proofs.Flock.Claims.ZK.Worker
public import Proofs.Flock.Soundness.Discharge.Composed.Inner.Holds
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

/-! The ZK server's stop at a table's own sample. -/

@[expose] public section

namespace FlockSoundness.Discharge.Composed

open Flock Model ZkSession ZkView ZK Refine

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem arrOf_padsAt {nE s : ℕ} (hs : s = nE + 4) (hp : Fin (nE + 4) → 𝔽) : arrOf (padsAt hs hp) = arrOf hp := by
  subst hs
  rfl

variable (P : LigProver) {mPts : ℕ}

theorem viewPads (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c (r : Fin 2) p (ω : (C.T c).tab.onVals.Rand),
      (∀ i, ((C.T c).tab.onVals.view p ω).pre.2.1 r i = (C.T c).tab.onVals.vals r p ω.1 ω.2.1 i +
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm) (Fin.castAdd 4 i)) ∧
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).hab) =
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).fa) *
          ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).fb) ∧
      ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
          (ix (C.T c).tab.onVals.nE (layOf Tz st).tw) =
        ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).tx) *
          ((C.T c).tab.onVals.hFull (ω.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)
            (ix (C.T c).tab.onVals.nE (layOf Tz st).ty) := by
  intro C c r p ω
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  obtain ⟨hfa, hfb, htx, hty, htw⟩ := lay_triple _ _ hE
  refine ⟨fun i => ?_, ?_, ?_⟩
  · show _ + _ = _ + _
    refine congrArg (_ + ·) ?_
    rw [Function.comp_apply, show (padEquiv (C.T c).tab.onVals.nE).symm (Fin.castAdd 4 i) = .inl i from
      (Equiv.symm_apply_eq _).2 (Fin.ext (by simp [padEquiv]))]
    rfl
  · rw [comp_symm_lt _ _ hfa, comp_symm_lt _ _ hfb,
      show ix (C.T c).tab.onVals.nE (layOf Tz st).hab = ix (C.T c).tab.onVals.nE ((C.T c).tab.onVals.nE + 0) from
        congrArg _ hE, comp_symm_tail _ 0 (by omega)]
    rfl
  · rw [htw, htx, hty, comp_symm_tail _ 1 (by omega), comp_symm_tail _ 2 (by omega), comp_symm_tail _ 3 (by omega)]
    rfl

theorem rowsAt_of_cons (Tz : TabZK 𝔽 mPts) (st : Stmt) :
    let C := coinedOf P Tz st
    ∀ c p (ω₀ : (C.T c).tab.onVals.Rand),
      (∀ r : Fin 2, ∃ cons : Array (String × Zk.Lin), cons.size = 7 ∧
        (∀ q ∈ cons, q.2.a.size = (layOf Tz st).len) ∧
        (∀ k : Fin 7, (ofZk cons[k.val]!.2 : ZkSession.Lin 𝔽 ((C.T c).tab.onVals.nE + 4)) =
          rowsOf Tz st c r (((C.T c).tab.onVals.view p ω₀).pre.2.1 r) ((C.T c).tab.onVals.view p ω₀).pre.1
            (((C.T c).tab.onVals.view p ω₀).pre.2.2.2.2.2 r).1 (0, 0) (Fin.castAdd 3 k)) ∧
        Flock.ZkServer.Finish.failingOn cons
          (arrOf ((C.T c).tab.onVals.hFull (ω₀.2.2.1 r) ∘ (padEquiv (C.T c).tab.onVals.nE).symm)) = #[]) →
      (C.T c).tab.onVals.RowsAt p ω₀.1 ω₀.2.1 := by
  intro C c p ω₀ h
  refine rowsAt_of_rows P Tz st c p ω₀.1 ω₀.2.1 fun r => ?_
  obtain ⟨cons, hsz, hlen, hform, hf⟩ := h r
  have hE : (layOf Tz st).hab = (C.T c).tab.onVals.nE := by
    show _ = padLenOf _ _ - 4
    simp only [padLenOf, layOf, Zk.PadLayout.len, Zk.PadLayout.tw]
    omega
  have hs := len_eq Tz st hE
  obtain ⟨hvv, hab, htw⟩ := viewPads P Tz st c r p ω₀
  rw [← arrOf_padsAt hs] at hf
  exact (noFail_iff_rows Tz st c r _ _ _ hE hvv hab htw _ _ hs cons hsz hlen hform).1 hf

end FlockSoundness.Discharge.Composed
