import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ScopeW.HeadlineExec

set_option Elab.async false

/-!
# The sites, the layout and the rows at the hidden-output `--zk` tables, under the widened scope

`ZkHidden/SoundBind.lean` with `hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`:

* `sitesShHW`: `ZkHidden.sitesShH` at `accW`, the drawn tables' acceptances with their unit slots' rows from
  `unitRowsFlatW`;
* `layShHW`, `rsShHW`, `rsShHW_computes`: `keyLayoutHW` and `keyRowsHW` at `sitesShHW`, each row under its own schema
  (`HmRowsP`);
* `placeZHW`, `decAgreeHW`, `layZHW`: `ZkHidden.placeZH`, `decAgreeH` and `layZH` at `sitesShHW`;
* `rsZHW`, `rsZHW_computes`: `rsShHW` as `HmRows` (`HmRowsP.toHmRows`, the same binding) moved to the `--zk` tables
  (`HmRowsZ.ofShadow`);
* `zk_flock_countC_execOS_atHW`: `ZkHidden.zk_flock_countC_execOS_atH` at `rsZHW`.
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Game Model Accounting Refine Audit Placed
open FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec
  FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.HiddenExec FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : scopeOkW c = true)

/-- **The public plan's sites, at the widened scope** (`ZkHidden.sitesShH` at `accW`). -/
noncomputable def sitesShHW : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  acc S _ _ _ := accW (zkTyped ht) hc (scopeOkW_ports hscope) (tableShH dj y₀ S)
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

/-- **One circuit per session**, as `ZkHidden.sitesShH_parse`. -/
theorem sitesShHW_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHW dj y₀ ht hc hscope).acc S R u h).tags ((sitesShHW dj y₀ ht hc hscope).acc S R u h).cf
        ((sitesShHW dj y₀ ht hc hscope).acc S R u h).tables = .ok c :=
  fun _ _ _ _ => (parse_untyped (zkTyped ht) hc).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable site** of the public plan, at the widened scope (`ZkHidden.sitesShH_bcSite`, with
`tableH_bcSiteW` for `tableH_bcSite`). -/
theorem sitesShHW_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShHW dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHW dj y₀ ht hc hscope).o S R u h)
          ((sitesShHW dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH (accW (zkTyped ht) hc (scopeOkW_ports hscope) (tableShH dj y₀ S)) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteW (zkTyped ht) hc hscope dj hU hpub hpt _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`lay` at the public plan, at the widened scope**: `keyLayoutHW` at `sitesShHW`. -/
noncomputable abbrev layShHW :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShH (Reg := Reg) dj y₀) (tabShH dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutHW (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan, at the widened scope**: `keyRowsHW` at `sitesShHW`, its circuit conditions from
`scopeOkW`, with `hsite` from `sitesShHW_bcSite`. -/
noncomputable abbrev rsShHW : HmRowsP (layShHW dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsHW (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHW_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (scopeOkW_spec hU hscope).1
      (scopeOkW_spec hU hscope).2.2)

theorem rsShHW_computes : (rsShHW dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsHW_computes (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHW_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (scopeOkW_spec hU hscope).1
      (scopeOkW_spec hU hscope).2.2)

/-- **Each drawn unit's place in its hidden-output `--zk` table, at the widened scope**: `keyPlaceH`'s place at
`sitesShHW`, along `planAtZKH_S`. -/
noncomputable def placeZHW (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u)
    (keyPlaceH (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**, as `ZkHidden.decAgreeH`. -/
theorem decAgreeHW :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀) (placeZHW dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShHW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHW, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables, at the widened scope**: `layShHW` moved along `planAtZKH_S`. -/
noncomputable abbrev layZHW :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH (Reg := Reg) dj y₀)
      (tabAtZKH dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        (placeZHW dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planAtZKH_S dj y₀) (layShHW dj y₀ ht hc hscope hU (pub := pub)) (decAgreeHW dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables, at the widened scope**: `rsShHW` as `HmRows` (`HmRowsP.toHmRows`, which
binds as `rsShHW` does), moved along `planAtZKH_S`. -/
noncomputable abbrev rsZHW : HmRowsZ (layZHW dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planAtZKH_S dj y₀) (decAgreeHW dj y₀ ht hc hscope hU)
    ((rsShHW dj y₀ ht hc hscope hU hpub hpt).toHmRows (rsShHW_computes dj y₀ ht hc hscope hU hpub hpt))

/-- **`hHm` at the hidden-output `--zk` tables' rows, at the widened scope** (`HmRowsP.toHmRows_computes`, moved
along). -/
theorem rsZHW_computes : (rsZHW dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeHW dj y₀ ht hc hscope hU) _
    (HmRowsP.toHmRows_computes _ _)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables, at the widened scope**:
`ZkHidden.zk_flock_countC_execOS_atH` with `place := placeZHW` and `vb := rsZHW.bindingZ rsZHW_computes`. -/
theorem zk_flock_countC_execOS_atHW (Hs : Hashes 𝔽 Dig (List UInt8))
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
          ((rsZHW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHW_computes dj y₀ ht hc hscope hU hpub hpt))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
          (tabAtZKH dj y₀)
          ((rsZHW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHW_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((keyProg c hU n).partition (Layout.outsOf c hU n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZHW dj y₀ ht hc hscope hU)
    ((rsZHW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHW_computes dj y₀ ht hc hscope hU hpub hpt))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end FlockSoundness.Discharge.ScopeW
