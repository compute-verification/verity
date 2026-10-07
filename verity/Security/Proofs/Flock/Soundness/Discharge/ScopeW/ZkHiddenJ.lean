import Proofs.Flock.Soundness.Discharge.ScopeW.ZkHidden
import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJ

set_option Elab.async false

/-!
# One hidden-output `--zk` sampled session of `J` tables, under the widened scope

`ZkHidden/DefsJ.lean`'s sites, layout and rows, `zk_session_soundHJ` and `zk_session_soundHJ_custody`, with
`hscope : scopeOkW c = true` in place of `Layout.scopeOk c = true`:

* `tableShHJW_bcSite`, `sitesShHJW`, `sitesShHJW_bcSite`: `tableShHJ_bcSite`, `sitesShHJ`, `sitesShHJ_bcSite` at
  `accW`, with `tableH_bcSiteW` for `tableH_bcSite`;
* `layShHJW`, `rsShHJW`, `rsShHJW_computes`: `keyLayoutHW` and `keyRowsHW` at `sitesShHJW`;
* `placeZHJW`, `decAgreeHJW`, `layZHJW`, `rsZHJW`, `rsZHJW_computes`: `placeZHJ`, `decAgreeHJ`, `layZHJ`, `rsZHJ` and
  `rsZHJ_computes` at `sitesShHJW`, `rsShHJW` moved as `HmRows` (`HmRowsP.toHmRows`, the same binding);
* `zk_session_soundHJW`, `zk_session_soundHJ_custodyW`: the headlines, the units bound to their rows at `rsZHJW`. Every
  other hypothesis, the law and the bound are the base forms', and so are the proofs.
-/

namespace FlockSoundness.Discharge.ScopeW

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed StrictCR ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

variable {n mPts : ℕ} {Reg : Type}

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : scopeOkW c = true)

/-- **The public plan's sites at `J` tables, at the widened scope** (`ZkHidden.sitesShHJ` at `accW`). -/
noncomputable def sitesShHJW : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  acc S R u _ := recastAccH (accW (tableShHJ_typed dj y₀ ht S (partOf I (dj S) u.val))
    (tableShHJ_parse dj y₀ ht hc S (partOf I (dj S) u.val)) (scopeOkW_ports hscope)
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2)
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session**, as `ZkHidden.sitesShHJ_parse`. -/
theorem sitesShHJW_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHJW dj y₀ ht hc hscope).acc S R u h).tags
        ((sitesShHJW dj y₀ ht hc hscope).acc S R u h).cf ((sitesShHJW dj y₀ ht hc hscope).acc S R u h).tables =
        .ok c :=
  fun S _ _ _ => (parse_untyped (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

include hU hpub hpt in
/-- `BcSiteH` at a table set up at table `j`'s inputs, at the widened scope (`tableH_bcSiteW` at `inputsJ I j`). -/
theorem tableShHJW_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (accW p₁ p₂ (scopeOkW_ports hscope) Tb.2) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteW p₁ p₂ hscope (fun _ => d) hU ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) hpt x h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables, at the widened scope
(`ZkHidden.sitesShHJ_bcSite`). -/
theorem sitesShHJW_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJW dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHJW dj y₀ ht hc hscope).o S R u h)
          ((sitesShHJW dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJW_bcSite hscope hU hpub hpt _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan of `J` tables, at the widened scope**: `keyLayoutHW` at `sitesShHJW`. -/
noncomputable abbrev layShHJW :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutHW (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan of `J` tables, at the widened scope**: `keyRowsHW` at `sitesShHJW`, with `hsite` from
`sitesShHJW_bcSite`. -/
noncomputable abbrev rsShHJW : HmRowsP (layShHJW dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsHW (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJW_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (scopeOkW_spec hU hscope).1
      (scopeOkW_spec hU hscope).2.2)

theorem rsShHJW_computes : (rsShHJW dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  keyRowsHW_computes (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (scopeOkW_spec hU hscope).2.1 (scopeOkW_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJW_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (scopeOkW_spec hU hscope).1
      (scopeOkW_spec hU hscope).2.2)

/-- **Each drawn unit's place in its hidden-output `--zk` table at `J` tables, at the widened scope**: `keyPlaceH`'s
place at `sitesShHJW`, along `planZHJ_S`. -/
noncomputable def placeZHJW (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**, as `ZkHidden.decAgreeHJ`. -/
theorem decAgreeHJW :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJW dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJW_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJW, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables, at the widened scope**: `layShHJW` moved along
`planZHJ_S`. -/
noncomputable abbrev layZHJW :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJW dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJW dj y₀ ht hc hscope hU (pub := pub))
    (decAgreeHJW dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables, at the widened scope**: `rsShHJW` as `HmRows`
(`HmRowsP.toHmRows`, which binds as `rsShHJW` does), moved along `planZHJ_S`. -/
noncomputable abbrev rsZHJW : HmRowsZ (layZHJW dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planZHJ_S dj y₀) (decAgreeHJW dj y₀ ht hc hscope hU)
    ((rsShHJW dj y₀ ht hc hscope hU hpub hpt).toHmRows (rsShHJW_computes dj y₀ ht hc hscope hU hpub hpt))

/-- **`hHm` at the hidden-output `--zk` tables' rows, at `J` tables, at the widened scope**
(`HmRowsP.toHmRows_computes`, moved along). -/
theorem rsZHJW_computes : (rsZHJW dj y₀ ht hc hscope hU hpub hpt (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJW dj y₀ ht hc hscope hU) _
    (HmRowsP.toHmRows_computes _ _)

end Shadow

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at the widened scope.**
`ZkHidden.zk_session_soundHJ` with `hscope : scopeOkW c = true` and the binding `rsZHJW`'s. -/
theorem zk_session_soundHJW
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : scopeOkW c = true)
    -- the law (the live subset draw) and A3
    (kd : ℕ) (hkd : kd ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZKHJ I dj mPts ((Law.execOS (n := n) src (.subset kd) n).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOkZHJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the live prover
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (scopeOkW_spec hU hscope).2.1 dj y₀ wr _ hTags hRec hDraw)) τ
    -- cr/sha-512 for the knowledge finders of its simulation
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJW_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJW_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) := by
  intro σ hCR
  refine zk_headline_exec_tables execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg
    (decAtZJ (planZHJ dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht hc (scopeOkW_spec hU hscope).2.1 dj y₀ wr _ hTags hRec hDraw)) τ _ _ ?_
  refine (zk_flock_countC execArith hs512 _ (planZHJ dj y₀) (tabZHJ dj y₀) _
    (loweringSoundZK_of_place execArith _ (planZHJ dj y₀) (tabZHJ dj y₀) (placeZHJW dj y₀ ht hc hscope hU))
    hRw hM hρ hr (jointZC_ksBound execArith execArith_correct hs512 (planZHJ dj y₀) hk) ht' K₀ σ).trans ?_
  exact add_le_add (add_le_add (Tables.execOS_subset_miss_le kd hkd Lb src hA3 K₀)
    (ksAvgZC_le_strict execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) hCR)) le_rfl

/-- **One hidden-output `--zk` sampled session of `J` tables is sound, at the coin server's custody, at the widened
scope.** `ZkHidden.zk_session_soundHJ_custody` with `hscope : scopeOkW c = true` and the binding `rsZHJW`'s. -/
theorem zk_session_soundHJ_custodyW
    (I : Inputs) (ht : I.tags.typed = false)
    {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : scopeOkW c = true)
    (kd : ℕ) (hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZKH I (mPtsOf c.g (kd / nTab I)) S₀ (dj S₀))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf c.g (kd / nTab I)) → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ)
    (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256) (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and the coin server's custody of the record and its draw file
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    {qF qS : ℝ} :
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht hc (scopeOkW_spec hU hscope).2.1 dj y₀ wr _ hTags hCust.1
          fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _)) τ
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ ((Law.execOS (n := n) src (.subset kd) n).draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    prob (fun o => LiveAcceptsZK I o ∧
        K₀ ≤ (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
          (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
            ((rsZHJW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJW_computes dj y₀ ht hc hscope hU hpub hpt))
            (reg σ) (cont σ) k Rw)).card)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      (Law.subset n kd hkd).miss K₀ +
        ksAvgStrictZ (planZHJ dj y₀) (Law.execOS (n := n) src (.subset kd) n) k qF qS (reg σ) +
        linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
          ((rsZHJW dj y₀ ht hc hscope hU hpub hpt).bindingZ (rsZHJW_computes dj y₀ ht hc hscope hU hpub hpt))
          k Rw M ρ t' (reg σ) (cont σ) :=
  zk_session_soundHJW I ht dj y₀ wr dg N hc hU hpub hpt hscope kd hkd Lb src hA3
    (fun _ => drawOkZKHJ_of_file I ht hc kd n hdiv dj hCust.2 _) hTags hCust.1 hk hRw hM hρ hr ht' K₀ τ

end FlockSoundness.Discharge.ScopeW
