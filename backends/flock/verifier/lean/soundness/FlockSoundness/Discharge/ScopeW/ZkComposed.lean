import FlockSoundness.Discharge.Composed.Custody
import FlockSoundness.Discharge.ScopeW.Keyed

set_option Elab.async false

/-!
# One public-output `--zk` sampled session is sound, under the widened scope

`zk_session_soundW`, `zk_session_sound_custodyW`: `Composed.zk_session_sound` and `zk_session_sound_custody` with
`hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`, and the units bound to their rows at `rsZAtW`:
`keyedRowsOutsW` at `ZkBind.sitesSh`, each row under its own schema, moved to the `--zk` tables as `ZkBind.rsZAt` is
(`HmRowsZ.ofShadow` at `HmRowsP.toHmRows`). `layZAt`, `planZAt`, `sitesZAt` and every other hypothesis, the law and the
bound are the base forms', and so are the proofs: the count curve at `rsZAtW` (`zk_flock_countC_execOS_atW`) carried to
flock-verify's acceptance by `zk_headline_exec_on`.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

section At

variable {n : ℕ} {I : Inputs} (ht : I.tags.typed = false) {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) {Reg : Type}
  {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : scopeOkW c = true)

include hc hU hpub hpt hscope in
theorem sitesSh_bcSiteW :
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
    exact table_bcSite (zkTyped ht) dj hc (scopeOkW_spec hU hscope).2.1 hpub hpt _ h
  · have e : tableSh ht dj y₀ S = y₀.x.refused (zkTyped ht) S := by
      unfold tableSh; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableSh ht dj y₀ S).T.S.Satisfies z := by
      rw [e]; exact y₀.x.refused_unsat (zkTyped ht) S
    exact absurd hM (hns _)

/-- **`rs` at the `--zk` tables, at the widened scope**: `keyedRowsOutsW` at the public sites, its circuit conditions
from `scopeOkW` (`scopeOkW_spec`), moved to the `--zk` tables at `HmRowsP.toHmRows`. -/
noncomputable abbrev rsZAtW : HmRowsZ (layZAt (Reg := Reg) ht dj y₀ hc hU (pub := pub)) :=
  HmRowsZ.ofShadow (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU)
    ((keyedRowsOutsW (sitesSh_parse ht dj y₀ hc) Hm.saltCol hc (scopeOkW_spec hU hscope).1
      (scopeOkW_spec hU hscope).2.2 (scopeOkW_spec hU hscope).2.1
      (sitesSh_bcSiteW ht dj y₀ hc hU hpub hpt hscope)).toHmRows
      (keyedRowsOutsW_computes (sitesSh_parse ht dj y₀ hc) hc (scopeOkW_spec hU hscope).1
        (scopeOkW_spec hU hscope).2.2 (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope)
        (sitesSh_bcSiteW ht dj y₀ hc hU hpub hpt hscope)))

/-- **`hHm` at the `--zk` tables' rows, at the widened scope** (`keyedRowsOutsW_computes`, moved along). -/
theorem rsZAtW_computes : (rsZAtW (Reg := Reg) ht dj y₀ hc hU hpub hpt hscope).Computes :=
  HmRowsZ.ofShadow_computes (planZAt_S ht dj y₀) (decAgreeAt ht dj y₀ hc hU) _ (HmRowsP.toHmRows_computes _ _)

/-- **The compiled `--zk` count curve at the `--zk` tables, at the widened scope.** `ZkBind.zk_flock_countC_execOS_at`
at `rsZAtW`. -/
theorem zk_flock_countC_execOS_atW (Hs : Hashes 𝔽 Dig (List UInt8))
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
          ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planZAt dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_keyProg execArith Hs (planZAt dj y₀) (tabZAt dj y₀) execArith_correct hU (sitesZAt ht dj y₀) (sitesZAt_parse ht dj y₀ hc)
    (Layout.outsOf c hU n) (layZAt ht dj y₀ hc hU) (rsZAtW ht dj y₀ hc hU hpub hpt hscope)
    (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope) σs ks hks St hS Kd pop Lb src hA3 hk hRw hM hρ hr ht' K σ hCR tr hT

end At

variable {mPts n : ℕ} {Reg : Type}

/-- **One `--zk` sampled session is sound, at the widened scope.** `Composed.zk_session_sound` with
`hscope : scopeOkW c = true` and the binding `rsZAtW`'s. -/
theorem zk_session_soundW
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : scopeOkW c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) τ)) := by
  have hdec := decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)
  have hnever : ∀ o : Bool × Finset (Fin n), ¬ (o.1 = true ∧ o.1 = false) := fun o h => by
    rw [h.1] at h; exact Bool.noConfusion h.2
  have hc' := zk_flock_countC_execOS_atW ht dj y₀ hc hU hpub hpt hscope hs512 σs ks hks St hS Kd pop Lb src hA3 hk hRw hM
    hρ hr ht' K₀ (simZ execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ) hCR (execAccept := fun o => o.1 = true)
    (StrictCR.TreeRewind.ofNever hnever) (ofNever_finderCR hs512.H hnever 0)
  rw [StrictCR.TreeRewind.ofNever_bound, ENNReal.ofReal_zero, add_zero] at hc'
  exact zk_headline_exec_on execArith hs512 _ I wr _ dg _ _ _ _ N hdec τ _ _ hc'

/-- **One `--zk` sampled session is sound, under `live-verifier`, at the widened scope.**
`Composed.zk_session_sound_custody` with `hscope : scopeOkW c = true` and the binding `rsZAtW`'s. -/
theorem zk_session_sound_custodyW
    (I : Inputs) (ht : I.tags.typed = false)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : scopeOkW c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the record, at the link points of the law's draws
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I (mPtsOf c.g (∑ s, ks s)) S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut (mPtsOf c.g (∑ s, ks s)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verify command's tags, and live-verifier: the record and its draw file are the live session's
    (hTags : TagsOk I) (hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover, and cr/sha-512 for the knowledge finders of its simulation
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
      (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)))
    {qF qS : ℝ}
    (hCR : ∀ ω j, TableCRZ execArith hs512 (planZAt dj y₀
        ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))) j
      (zcStrat execArith hs512 (planZAt dj y₀)
        (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ) ω)) qF qS) :
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
            ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
            (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
            (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg (mPtsOf c.g (∑ s, ks s)) N wr)) τ ≤
      (Law.stratified σs ks hks).miss K₀ +
        ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
          ((rsZAtW ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAtW_computes ht dj y₀ hc hU hpub hpt hscope))
          k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ))
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hCust.1
          (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω))) τ)) := by
  exact zk_session_soundW I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St hS Kd pop Lb src hA3
    (fun ω => drawOkZK_of_exec I ht hc σs ks hks St hS Kd pop Lb src dj hCust.2 ω) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ
    hCR

end FlockSoundness.Discharge.ScopeW
