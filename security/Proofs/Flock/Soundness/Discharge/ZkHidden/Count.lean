module

public import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
public import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ
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

/-! The compiled `--zk` count curve at the hidden-output tables. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkHidden

open Game Model Accounting Audit ZkSession StrictCR ZkLink Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {I : Inputs}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : SampleSetupZKH I mPts S₀ (dj S₀))

theorem zk_flock_countC_execOS_crH (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (SamplingDesign.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : SamplingDesign.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S →
      UnitPlace P (tabOfZ (planAtZKH (Reg := Reg) dj y₀) (tabAtZKH dj y₀) S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P (planAtZKH dj y₀) (tabAtZKH dj y₀)
      (decodeOfPlace A P (planAtZKH dj y₀) (tabAtZKH dj y₀) place) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs
      (planAtZKH dj y₀ ((SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.incorrect (XplurZC A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (audit (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC A Hs (planAtZKH dj y₀))) σ ≤
      (SamplingDesign.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (SamplingDesign.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) vb k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) hA σs ks hks St hS Kd pop Lb src hA3 place vb
    hk hRw hM hρ hr ht K σ hCR tr hT

end FlockSoundness.Discharge.ZkHidden
