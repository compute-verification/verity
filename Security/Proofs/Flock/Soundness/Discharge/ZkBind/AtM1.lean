import Proofs.Flock.Soundness.Discharge.ZkBind.At
import Proofs.Flock.Soundness.Discharge.ZkLink.M1

/-!
# The compiled `--zk` count curve at the `--zk` tables, at C-Flock ZK's parameters

`zk_flock_countC_execOS_at` with its knowledge term in closed form (`ksAvgStrictZm1`): every table of the one-table
plan `planZAt` is at C-Flock ZK's parameters (`planZAt_M1`, from `ZkLink.tableAtZK_M1`: what `verify --zk` checks at
setup), so `ksAvgStrictZ_le_m1` applies with no hypothesis on the tables.
-/

namespace FlockSoundness.Discharge.ZkBind

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec Integrate Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}

include ht in
/-- **Every table of the `--zk` plan is at C-Flock ZK's parameters** (`tableAtZK_M1`). -/
theorem planZAt_M1 :
    ∀ S (R : Reg) (j : Fin (planZAt (Reg := Reg) dj y₀ S R).length), ((planZAt dj y₀ S R).get j).M1 := by
  intro S R j
  rcases j with ⟨_ | k, hk⟩
  · exact tableAtZK_M1 ht dj y₀ S
  · have : k + 1 < 1 := hk
    omega

variable {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : Layout.scopeOk c = true)

/-- **The compiled `--zk` count curve at the `--zk` tables, at C-Flock ZK's parameters**: `zk_flock_countC_execOS_at`
with the knowledge term `ksAvgStrictZm1` (`ksAvgStrictZ_le_m1` at `planZAt_M1`). -/
theorem zk_flock_countC_execOS_at_m1 (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planZAt dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planZAt dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planZAt dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planZAt dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planZAt dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZm1 (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  (zk_flock_countC_execOS_at ht dj y₀ hc hU hpub hpt hscope Hs σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht'
      K σ hCR tr hT).trans
    (add_le_add (add_le_add (add_le_add le_rfl
      (ksAvgStrictZ_le_m1 execArith_correct.card_F (planZAt dj y₀) (planZAt_M1 ht dj y₀) _ k qF qS (reg σ))) le_rfl)
      le_rfl)

end FlockSoundness.Discharge.ZkBind
