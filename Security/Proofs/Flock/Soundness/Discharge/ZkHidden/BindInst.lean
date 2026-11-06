import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst

set_option Elab.async false

/-!
# The sites, the layout and the rows at the hidden-output `--zk` tables, at the template instance's program

`SoundBind.lean` and `DefsJ.lean`'s shadow at one table and at `J` tables, at `instProg c n` (`Inst/Prog.lean`) in
place of `Placed.keyProg c hU n`, as `HiddenExec.HeadlineInst` is at the public path's tables. The circuit conditions
are the rows' fit (`hrf`), the unit's reads (`hpr`) and the wires' order (`hw`), with no condition on the unit's input
groups, on where the outputs are returned, or on shared-row tables:

* `sitesShHI`, `sitesShHJI`: the public tables' sites, each acceptance at `hrf`;
* `sitesShHI_bcSite`, `sitesShHJI_bcSite`: `BcSiteH` at every satisfiable site (`HiddenExec.tableH_bcSiteI`);
* `layShHI`, `rsShHI`, `rsShHI_computes` (and `…HJI`): `instLayoutH` and `instRowsH` at those sites, `InstKeysRow` from
  the parse at the reference setup's statement (`instKeysRowH`), a unit per VU from the check;
* `placeZHI`, `decAgreeHI`, `layZHI`, `rsZHI`, `rsZHI_computes` (and `…HJI`): moved to the `--zk` tables
  (`LayoutZ.ofShadow`, `HmRowsZX.ofShadow`);
* `zk_flock_countC_execOS_atHI`: `zk_flock_countC_execOS_atH` at those.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind HiddenExec Placed Composed
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Layout (portRowW)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## One table -/

section One

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hrf : Hidden.Any.RowsFitS c)

/-- **The public plan's sites** (`sitesShH`'s, each acceptance at the rows' fit `hrf`). -/
noncomputable def sitesShHI : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  A S _ _ _ := (tableShH dj y₀ S).twinA
  B S _ _ _ := (tableShH dj y₀ S).twinB
  sat S _ _ _ := (tableShH dj y₀ S).sat (zkTyped ht) hc hrf
  acc S _ _ _ := (tableShH dj y₀ S).acc (zkTyped ht) hc hrf
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

theorem sitesShHI_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      ((sitesShHI dj y₀ ht hc hrf).acc S R u h).st.c = c :=
  fun S R u h => ((sitesShHI dj y₀ ht hc hrf).acc S R u h).c_eq (HmOut.parse_isTmpl hc)
    (parse_untyped (zkTyped ht) hc).1

variable {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hpub in
/-- **`BcSiteH` at every satisfiable site** of the public plan: at a draw `verify --zk` sets up by `tableH_bcSiteI`; at
a refused draw no message satisfies the table. -/
theorem sitesShHI_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u), Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShHI dj y₀ ht hc hrf).acc S R u h) pub ((sitesShHI dj y₀ ht hc hrf).o S R u h)
          ((sitesShHI dj y₀ ht hc hrf).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).acc (zkTyped ht) hc hrf) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteI (zkTyped ht) hc hrf dj hpub _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`lay` at the public plan, at `instProg`**: `instLayoutH` at `sitesShHI`, the program's outputs the output
words' bits, the registration the population's public file. -/
noncomputable abbrev layShHI :
    Binding.Layout ((instProg c n).partition (outsOfI c n)) (planShH (Reg := Reg) dj y₀) (tabShH dj y₀)
      (((instProg c n).partition (outsOfI c n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (instPlaceH (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n))) :=
  instLayoutH (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub)

variable (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)

/-- **`rs` at the public plan, at `instProg`**: `instRowsH` at `sitesShHI`, with `hsite` from `sitesShHI_bcSite` and
`InstKeysRow` from the parse at the reference setup's statement. -/
noncomputable abbrev rsShHI : Binding.HmRowsX (layShHI dj y₀ ht hc hrf (pub := pub) (Reg := Reg)) :=
  instRowsH (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (sitesShHI_bcSite dj y₀ ht hc hrf hpub) (instKeysRowH (zkTyped ht) hc dj y₀.x hU hpr hw)

theorem rsShHI_computes : (rsShHI dj y₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).Computes :=
  instRowsH_computes (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (checkFactsH (zkTyped ht) hc).upv_pos (sitesShHI_bcSite dj y₀ ht hc hrf hpub)
    (instKeysRowH (zkTyped ht) hc dj y₀.x hU hpr hw)

/-- **Each drawn unit's place in its hidden-output `--zk` table, at `instProg`**: `instPlaceH`'s place in the public
table, along `planAtZKH_S`. -/
noncomputable def placeZHI (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((instProg c n).partition (outsOfI c n)) (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u) (instPlaceH (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHI :
    DecAgree (P := (instProg c n).partition (outsOfI c n)) (planAtZKH_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀) (placeZHI dj y₀ ht hc hrf))
      (decS := ((instProg c n).partition (outsOfI c n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (instPlaceH (sitesShHI_parse dj y₀ ht hc hrf) hrf (outsOfI c n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHI, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables, at `instProg`**: `layShHI` moved along `planAtZKH_S`. -/
noncomputable abbrev layZHI :
    LayoutZ execArith ((instProg c n).partition (outsOfI c n)) (planAtZKH (Reg := Reg) dj y₀) (tabAtZKH dj y₀)
      (decodeOfPlace execArith ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
        (placeZHI dj y₀ ht hc hrf)) :=
  LayoutZ.ofShadow (planAtZKH_S dj y₀) (layShHI dj y₀ ht hc hrf (pub := pub)) (decAgreeHI dj y₀ ht hc hrf)

/-- **`rs` at the hidden-output `--zk` tables, at `instProg`**: `rsShHI` moved along `planAtZKH_S`. -/
noncomputable abbrev rsZHI : HmRowsZX (layZHI dj y₀ ht hc hrf (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planAtZKH_S dj y₀) (decAgreeHI dj y₀ ht hc hrf) (rsShHI dj y₀ ht hc hrf hpub hU hpr hw)

/-- **`hHm` at the hidden-output `--zk` tables' rows, at `instProg`** (`rsShHI_computes`, moved along). -/
theorem rsZHI_computes : (rsZHI dj y₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeHI dj y₀ ht hc hrf) _
    (rsShHI_computes dj y₀ ht hc hrf hpub hU hpr hw)

/-- **The compiled `--zk` count curve at the hidden-output `--zk` tables, at `instProg`**:
`zk_flock_countC_execOS_crH` with `place := placeZHI` and `vb := rsZHI.bindingZ rsZHI_computes`. Its circuit
conditions are `hrf`, `hpr` and `hw`; the public file may carry shared-row tables. -/
theorem zk_flock_countC_execOS_atHI (Hs : Hashes 𝔽 Dig (List UInt8))
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
    prob (fun o => execAccept o ∧ K ≤ (((instProg c n).partition (outsOfI c n)).wrong
        (XplurZC execArith Hs ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          (reg σ) (cont σ) k Rw)).card)
        (audit (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg
          (batchedSessionZC execArith Hs (planAtZKH dj y₀))) σ ≤
      (Law.stratified σs ks hks).miss K +
        ksAvgStrictZ (planAtZKH dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg σ) +
        linkBoundZC execArith Hs ((instProg c n).partition (outsOfI c n)) (planAtZKH dj y₀) (tabAtZKH dj y₀)
          ((rsZHI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHI_computes dj y₀ ht hc hrf hpub hU hpr hw))
          k Rw M ρ t' (reg σ) (cont σ) +
        ENNReal.ofReal (tr.bound qT) :=
  zk_flock_countC_execOS_crH execArith Hs ((instProg c n).partition (outsOfI c n)) dj y₀
    execArith_correct σs ks hks St hS Kd pop Lb src hA3 (placeZHI dj y₀ ht hc hrf)
    ((rsZHI dj y₀ ht hc hrf hpub hU hpr hw).bindingZ (rsZHI_computes dj y₀ ht hc hrf hpub hU hpr hw))
    hk hRw hM hρ hr ht' K σ hCR tr hT

end One

/-! ## `J` tables -/

section J

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hrf : Hidden.Any.RowsFitS c)

/-- **The public plan's sites at `J` tables** (`sitesShHJ`'s, each acceptance at the rows' fit `hrf`). -/
noncomputable def sitesShHJI : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  A S R u _ := castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinA
  B S R u _ := castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinB
  sat S R u _ := twin_sat_cast (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.sat (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _) hrf)
  acc S R u _ := recastAccH ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.acc
    (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _) hrf)
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

theorem sitesShHJI_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      ((sitesShHJI dj y₀ ht hc hrf).acc S R u h).st.c = c :=
  fun S R u h => ((sitesShHJI dj y₀ ht hc hrf).acc S R u h).c_eq
    (HmOut.parse_isTmpl (tableShHJ_parse dj y₀ ht hc S (partOf I (dj S) u.val)))
    (parse_untyped (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)).1

variable {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hpub in
/-- `BcSiteH` at a table set up at table `j`'s inputs (`tableH_bcSiteI` at `inputsJ I j`). -/
theorem tableShHJ_bcSiteI {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.acc p₁ p₂ hrf) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteI p₁ p₂ hrf (fun _ => d) ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hpub in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables: at a table `verify --zk` sets up by
`tableShHJ_bcSiteI`; at a refused one no message satisfies the table. -/
theorem sitesShHJI_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJI dj y₀ ht hc hrf).acc S R u h) pub ((sitesShHJI dj y₀ ht hc hrf).o S R u h)
          ((sitesShHJI dj y₀ ht hc hrf).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSiteI hrf hpub _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan of `J` tables, at `instProg`**: `instLayoutH` at `sitesShHJI`. -/
noncomputable abbrev layShHJI :
    Binding.Layout ((instProg c n).partition (outsOfI c n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((instProg c n).partition (outsOfI c n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (instPlaceH (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n))) :=
  instLayoutH (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub)

variable (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)

include ht hc y₀ hU hpr hw in
/-- **`InstKeysRow` at the session's circuit**, from the parse at table 0's reference setup (`instKeysRowH` at
`inputsJ I 0`). -/
theorem instKeysRowHJ : InstKeysRow c (portRowW c) :=
  instKeysRowH (typedJ ht 0) ((parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc) (fun _ => d₀) y₀.x
    hU hpr hw

/-- **`rs` at the public plan of `J` tables, at `instProg`**: `instRowsH` at `sitesShHJI`, with `hsite` from
`sitesShHJI_bcSite` and `InstKeysRow` from `instKeysRowHJ`. -/
noncomputable abbrev rsShHJI : Binding.HmRowsX (layShHJI dj y₀ ht hc hrf (pub := pub) (Reg := Reg)) :=
  instRowsH (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (sitesShHJI_bcSite dj y₀ ht hc hrf hpub) (instKeysRowHJ y₀ ht hc hU hpr hw)

theorem rsShHJI_computes : (rsShHJI dj y₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).Computes :=
  instRowsH_computes (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (checkFactsH (zkTyped ht) hc).upv_pos (sitesShHJI_bcSite dj y₀ ht hc hrf hpub) (instKeysRowHJ y₀ ht hc hU hpr hw)

/-- **Each drawn unit's place in its hidden-output `--zk` table at `J` tables, at `instProg`**: `instPlaceH`'s place
in the public table, along `planZHJ_S`. -/
noncomputable def placeZHJI (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((instProg c n).partition (outsOfI c n)) (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u) (instPlaceH (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n) S R u h)

/-- **The decoders agree** at `J` tables. -/
theorem decAgreeHJI :
    DecAgree (P := (instProg c n).partition (outsOfI c n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJI dj y₀ ht hc hrf))
      (decS := ((instProg c n).partition (outsOfI c n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (instPlaceH (sitesShHJI_parse dj y₀ ht hc hrf) hrf (outsOfI c n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJI, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables, at `instProg`**: `layShHJI` moved along `planZHJ_S`. -/
noncomputable abbrev layZHJI :
    LayoutZ execArith ((instProg c n).partition (outsOfI c n)) (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀)
      (decodeOfPlace execArith ((instProg c n).partition (outsOfI c n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJI dj y₀ ht hc hrf)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJI dj y₀ ht hc hrf (pub := pub)) (decAgreeHJI dj y₀ ht hc hrf)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables, at `instProg`**: `rsShHJI` moved along `planZHJ_S`. -/
noncomputable abbrev rsZHJI : HmRowsZX (layZHJI dj y₀ ht hc hrf (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planZHJ_S dj y₀) (decAgreeHJI dj y₀ ht hc hrf) (rsShHJI dj y₀ ht hc hrf hpub hU hpr hw)

/-- **`hHm` at the hidden-output `--zk` tables' rows of `J` tables, at `instProg`** (`rsShHJI_computes`, moved
along). -/
theorem rsZHJI_computes : (rsZHJI dj y₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJI dj y₀ ht hc hrf) _
    (rsShHJI_computes dj y₀ ht hc hrf hpub hU hpr hw)

end J

end FlockSoundness.Discharge.ZkHidden
