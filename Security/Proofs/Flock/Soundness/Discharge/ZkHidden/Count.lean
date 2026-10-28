import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
import Proofs.Flock.Soundness.Discharge.ZkLink.StrictZ

/-!
# The compiled `--zk` count curve at the hidden-output tables

`zk_flock_countC_execOS_cr` reads its tables only through the plan, so it holds at the one-table hidden-output plan
`planAtZKH` as stated. The hidden-output binding enters through `place` and `vb`, which read the drawn units' places
in `tableAtZKH`'s statements (`AcceptsZKAcceptedH`).
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Model Accounting Audit ZkSession StrictCR ZkLink Exec
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {I : Inputs}
  (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))

/-- **#874's compiled `--zk` count curve at the hidden-output tables**: `zk_flock_countC_execOS_cr` at
`planAtZKH dj y₀` (each draw's table `tableAtZKH dj y₀ S`, every unit in it). -/
theorem zk_flock_countC_execOS_crH (hA : A.Correct)
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    (place : ∀ S R u, u ∈ S →
      UnitPlace P (tabOfZ (planAtZKH (Reg := Reg) dj y₀) (tabAtZKH dj y₀) S R u).S u)
    {Dh : Type} {Hc : List UInt8 → Dh}
    (vb : ValueBindingZ A P (planAtZKH dj y₀) (tabAtZKH dj y₀)
      (decodeOfPlace A P (planAtZKH dj y₀) (tabAtZKH dj y₀) place) Hc)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ A Hs
      (planAtZKH dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat A Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC A Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧
        K ≤ (P.wrong (XplurZC A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) vb (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC A Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) vb k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_cr A Hs P (planAtZKH dj y₀) (tabAtZKH dj y₀) hA σs ks hks St hS Kd pop Lb src hA3 place vb
    hk hRw hM hρ hr ht K σ hCR tr hT

end FlockSoundness.Discharge.ZkHidden
