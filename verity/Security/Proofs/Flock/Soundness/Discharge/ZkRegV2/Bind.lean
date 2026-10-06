import Proofs.Flock.Soundness.Discharge.ZkReg.Bind
import Proofs.Flock.Soundness.Discharge.ZkRegV2.Scope
import Proofs.Flock.Soundness.Discharge.ZkRegV2.Site

/-!
# The rows at the `--zk` tables, for `sha512/row/v2` rows

`ZkReg/Bind.lean` at a circuit `scopeOkV2` admits: `rowsShV2` is `rowsSh` with the rows' `sha512/row/v2` bytes
(`Layout.keyedRowsOutsP Hm.preV2`), `rsZAtSV2` moves it to the `--zk` tables, `rsZAtSV2_computes` is `hHm` there
(`Hm.keyedRowsOutsV2_computes`), and `zk_flock_countC_execOS_atSV2` is the compiled `--zk` count curve at `rsZAtSV2`.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind Integrate Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hscope : ZkRegV2.scopeOkV2 c = true)

include hc hU hpub hscope in
theorem sitesSh_bcSiteV2 :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planSh ht dj y₀) (tabSh ht dj y₀) S R u),
      Binding.Sat (planSh ht dj y₀) (tabSh ht dj y₀) S R u M →
        Layout.BcSite ((sitesSh ht dj y₀).acc S R u h) pub ((sitesSh ht dj y₀).o S R u h)
          ((sitesSh ht dj y₀).vu S R u h) u.val := by
  intro S R u h M hM
  show Layout.BcSite (tableSh ht dj y₀ S).acc pub ((tableSh ht dj y₀ S).o u h) ((tableSh ht dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZK I mPts S (dj S))
  · have e : tableSh ht dj y₀ S = (Classical.choice hS).x.table (zkTyped ht) := by
      unfold tableSh; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact table_bcSiteS (zkTyped ht) dj hc (ZkRegV2.scopeOkV2_spec hU hscope).2.1 hpub _ h
  · have e : tableSh ht dj y₀ S = y₀.x.refused (zkTyped ht) S := by
      unfold tableSh; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableSh ht dj y₀ S).T.S.Satisfies z := by
      rw [e]; exact y₀.x.refused_unsat (zkTyped ht) S
    exact absurd hM (hns _)

/-- The public plan's rows at a `scopeOkV2` circuit: `rowsSh` with the rows' `sha512/row/v2` bytes. -/
noncomputable abbrev rowsShV2 : Binding.HmRows (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub)) :=
  Layout.keyedRowsOutsP (sitesSh_parse ht dj y₀ hc) Hm.saltCol Hm.preV2 Hm.preV2_size hc
    (ZkRegV2.scopeOkV2_spec hU hscope).1 (ZkRegV2.scopeOkV2_spec hU hscope).2.2 (ZkRegV2.scopeOkV2_spec hU hscope).2.1
    (sitesSh_bcSiteV2 ht dj y₀ hc hU hpub hscope)

/-- **`rs` at the `--zk` tables, for `sha512/row/v2` rows**: `rowsShV2` moved to the `--zk` tables. -/
noncomputable abbrev rsZAtSV2 : HmRowsZ (layZAt (Reg := Reg) ht dj y₀ hc hU (pub := pub)) :=
  HmRowsZ.ofShadow (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU) (rowsShV2 ht dj y₀ hc hU hpub hscope)

/-- **`hHm` at the `--zk` tables' `sha512/row/v2` rows.** -/
theorem rsZAtSV2_computes : (rsZAtSV2 (Reg := Reg) ht dj y₀ hc hU hpub hscope).Computes :=
  HmRowsZ.ofShadow_computes (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU) _
    (Hm.keyedRowsOutsV2_computes (sitesSh_parse ht dj y₀ hc) hc (ZkRegV2.scopeOkV2_spec hU hscope).1
      (ZkRegV2.scopeOkV2_spec hU hscope).2.2 (ZkRegV2.scopeOkV2_spec hU hscope).2.1 (ZkRegV2.scopeOkV2_ports hscope)
      (sitesSh_bcSiteV2 ht dj y₀ hc hU hpub hscope))

/-- **The compiled `--zk` count curve at the `--zk` tables' `sha512/row/v2` rows**: `zk_flock_countC_execOS_atS` at
`rsZAtSV2`. -/
theorem zk_flock_countC_execOS_atSV2 (Hs : Hashes 𝔽 Dig (List UInt8))
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
          ((rsZAtSV2 ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtSV2_computes ht dj y₀ hc hU hpub hscope))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planZAt dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtSV2 ht dj y₀ hc hU hpub hscope).bindingZ (rsZAtSV2_computes ht dj y₀ hc hU hpub hscope))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_keyProg execArith Hs (planZAt dj y₀) (tabZAt dj y₀) execArith_correct hU (sitesZAt ht dj y₀)
    (sitesZAt_parse ht dj y₀ hc) (Layout.outsOf c hU n) (layZAt ht dj y₀ hc hU) (rsZAtSV2 ht dj y₀ hc hU hpub hscope)
    (rsZAtSV2_computes ht dj y₀ hc hU hpub hscope) σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ZkReg
