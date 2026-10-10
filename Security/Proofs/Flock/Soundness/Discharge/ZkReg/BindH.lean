import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Registered.ReadsX

set_option Elab.async false

/-!
# The rows at the hidden-output `--zk` tables, with or without shared rows

`ZkHidden.rsZH` takes `pub.tables = none`, as `ZkBind.rsZAt` does (`Bind.lean`), so here are `rsShH`, `rsZH`, their
`Computes` and `zk_flock_countC_execOS_atH` without it: `sitesShH_bcSiteS` is `sitesShH_bcSite` from
`HiddenExec.tableH_bcSiteS`. `layShH` is the shadow layout `layZH` moves to the `--zk` tables, where the registered reads
(`Registered.ReadsX`) are read.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOkH c = true)
  (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hU hpub in
/-- **`BcSiteH` at every satisfiable site** of the public plan, with or without shared rows (`tableH_bcSiteS`). -/
theorem sitesShH_bcSiteS :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShH dj y₀ ht hc hscope).acc S R u h) pub ((sitesShH dj y₀ ht hc hscope).o S R u h)
          ((sitesShH dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).acc (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteS (zkTyped ht) hc hscope dj hU hpub _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`rs` at the public plan, with or without shared rows**: `rsShH` with `hsite` from `sitesShH_bcSiteS`. -/
noncomputable abbrev rsShHS : Binding.HmRowsX (layShH dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShH_bcSiteS dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShHS_computes : (rsShHS dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH (zkTyped ht) hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShH_bcSiteS dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`rs` at the hidden-output `--zk` tables, with or without shared rows**: `rsShHS` moved along `planAtZKH_S`. -/
noncomputable abbrev rsZHS : HmRowsZX (layZH dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planAtZKH_S dj y₀) (decAgreeH dj y₀ ht hc hscope hU) (rsShHS dj y₀ ht hc hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' rows, with or without shared rows** (`rsZH_computes`). -/
theorem rsZHS_computes : (rsZHS dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeH dj y₀ ht hc hscope hU) _
    (rsShHS_computes dj y₀ ht hc hscope hU hpub)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables, with or without shared rows**:
`zk_flock_countC_execOS_atH` without `pub.tables = none`, at `rsZHS`. -/
theorem zk_flock_countC_execOS_atHS (Hs : Hashes 𝔽 Dig (List UInt8))
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K : ℕ)
    (σ : Strategy (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith Hs
      (planAtZKH dj y₀ ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω) (reg σ))
      j (zcStrat execArith Hs (planAtZKH dj y₀) (cont σ ω)) qF qS)
    {execAccept : Bool × Finset (Fin n) → Prop}
    (tr : StrictCR.TreeRewind Hs.H (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ fun o => execAccept o ∧ o.1 = false)
    {qT : ℝ} (hT : tr.finder.CR Hs.H qT) :
    prob (fun o => execAccept o ∧ K ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
        (XplurZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZH dj y₀ ht hc hscope hU)
    ((rsZHS dj y₀ ht hc hscope hU hpub).bindingZ (rsZHS_computes dj y₀ ht hc hscope hU hpub))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkReg
