module

public import Proofs.Flock.Claims.Recursion.NonVacuous
public import Proofs.Flock.Claims.Recursion.RoundCounts
public import Proofs.Flock.Claims.Recursion.VerdictRowL
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
import Proofs.Flock.Claims.Recursion.VerdictLoop

/-! Row L's target holds (`targetL_holds`), with its coin tie `TargetZ` proved (`targetZ`). -/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP ReadOkR VerdictR holdsR)
open FlockSoundness.Discharge.Recursion (HClass)
open RecDeploy.Rows (DecAt)

namespace Top

def TargetZ (rd : Msg → Option FinalP) (P : Msg → List Msg → List Coin → Prop) : Prop :=
  ∀ (x : Msg) (ms : List Msg) (cs : List Coin), At rd P x ms cs → ∀ (D : DecAt rd x ms cs) (cls : HClass) (i : ℕ),
    i < D.sts.size → ∀ r : Fin 2,
      (((D.files cls i).stream r).rounds[2]!).coins.getD 0 0 = (D.ctx (2 * i + r)).z

variable {rd : Msg → Option FinalP} {P : Msg → List Msg → List Coin → Prop}

theorem after_of_base {x : Msg} {ms : List Msg} {cs : List Coin} (hr : ReadOkR rd x ms cs)
    (hb : ∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) (D : DecAt rd x ms cs) :
    VerdictR.At FlockDeploy.sha512 .after D.d.s x D.d.link D.d.c₁ D.d.mids D.d.f := by
  have h := hb .after (by simp [VerdictP.bases])
  unfold holdsR at h
  rw [actR_of_dec hr D] at h
  exact h

theorem targetL (acc : Msg → List Msg → List Coin → ℕ → Array F128) (hZ : TargetZ rd P) : TargetL rd P acc := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, hr, hb, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact D.rowL_at cls (acc x ms cs) i r hst hf hx hi hti D.points_eq (idx_lt D hi r)
    (shaped_of_base hr hb D _ (idx_lt D hi r)) (after_of_base hr hb D) (hZ x ms cs hA D cls i hi r)

theorem targetZ : TargetZ rd P := by
  intro x ms cs hA D cls i hi r
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact D.hz cls i (D.names_nodup hst hf hx) hi hti r

theorem targetL_holds (acc : Msg → List Msg → List Coin → ℕ → Array F128) : TargetL rd P acc :=
  targetL acc targetZ

theorem hsz_of_sched (acc : Msg → List Msg → List Coin → ℕ → Array F128) {x : Msg} {ms : List Msg} {cs : List Coin}
    (hA : At rd P x ms cs) (D : DecAt rd x ms cs) (cls : HClass) {i : ℕ} (hi : i < D.sts.size)
    (hsched : (RecDeploy.repSchedAt D.spec D.sts[i]!).map (·.1) = FlockVBridge.Family.roundSizes (D.sh cls i)) :
    ∀ r : Fin 2, (D.rep cls (acc x ms cs) i r).rounds.toList.map Array.size =
      FlockVBridge.Family.roundSizes (D.sh cls i) := by
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -, -, -⟩ := id hA
  have hti : i < D.spec.tables.size := by
    obtain ⟨h1, h2⟩ := D.tables_of_dense hst hf hx
    omega
  exact fun r => D.hsz_of_sched cls i (acc x ms cs) (D.names_nodup hst hf hx) hi hti r hsched

end Top

end RecDeploy
