import Proofs.Flock.Soundness.Discharge.PlainHJ.Defs
import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineInst

set_option Elab.async false

/-!
# The sites, the layout and the rows at a plain session of `J` hidden-output tables, at the template instance's program

`ZkHidden/BindInst.lean`'s `J`-table shadow (`sitesShHJI`, `layShHJI`, `rsShHJI`) at the plain inputs
`PlainJ.inputsPJ I j` and the plan `planPHJ` itself, with no move to compiled tables: the plain session's tables are the
public tables the binding reads. The circuit conditions are `HiddenExec.HeadlineInst`'s: the rows' fit (`hrf`), the
unit's reads (`hpr`) and the wires' order (`hw`).

* `tablePHJ_typed`, `tablePHJ_parse`: each table's inputs are untyped and parse the session's circuit (`tagsAt` changes
  the identity alone, which an untyped `HmOut.parse` does not read).
* `sitesPHJI`, `sitesPHJI_parse`, `sitesPHJI_bcSite`: each drawn unit's acceptance, VU and block, the one circuit they
  parse, and `BcSiteH` at every satisfiable site (`HiddenExec.tableH_bcSiteI` at table `j`'s inputs).
* `ppPHJI`, `layPHJI`, `rsPHJI`, `rsPHJI_computes`: `SitesH.progPlaces_instProg`, `instLayoutH` and `instRowsH` at those
  sites, `InstKeysRow` from the parse at the reference setup's statement (`instKeysRowPHJ`).
* `ppPHJ`, `layPHJ`, `rsPHJ`, `rsPHJ_computes`: where the wires feed neither the unit nor the outputs, the same sites at
  the rows' fit `scopeOkH` gives, with `keyLayoutH` and `keyRowsH` at the unit's key program `Placed.keyProg c hU n`.
-/

namespace FlockSoundness.Discharge.PlainHJ

open Flock FlockLevel3 Game Model Accounting Refine Audit HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open FlockSoundness.Discharge.Inst
open FlockSoundness.Discharge.Composed (nTab drawJ partOf partS mem_partS)
open FlockSoundness.Discharge.PlainJ (inputsPJ typedPJ)
open FlockSoundness.Discharge.ZkHidden (recastAccH bcSiteH_recast parseH_tagsAt loadPublicH_tagsAt)
open FlockSoundness.Discharge.Layout (portRowW)

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  {d₀ : Lean.Json} (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hrf : Hidden.Any.RowsFitS c)

include ht in
theorem tablePHJ_typed (S : Finset (Fin n)) (j : Fin (nTab I)) : (tablePHJ dj x₀ S j).1.tags.typed = false := by
  by_cases h : Nonempty (DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tablePHJ_set dj x₀ h]
    exact typedPJ ht j
  · rw [tablePHJ_refused dj x₀ h]
    exact typedPJ ht 0

include ht hc in
/-- **One circuit per session, at every table**: each table's inputs parse the session's circuit file to `c`. -/
theorem tablePHJ_parse (S : Finset (Fin n)) (j : Fin (nTab I)) :
    HmOut.parse (tablePHJ dj x₀ S j).1.tags (tablePHJ dj x₀ S j).1.circuitFile (tablePHJ dj x₀ S j).1.tables =
      .ok c := by
  by_cases h : Nonempty (DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tablePHJ_set dj x₀ h]
    exact (parseH_tagsAt ht _ _ _ _).trans hc
  · rw [tablePHJ_refused dj x₀ h]
    exact (parseH_tagsAt ht _ _ _ _).trans hc

/-- **The plan's sites**: each drawn unit's acceptance (at the rows' fit `hrf`), VU and block are its table's. -/
noncomputable def sitesPHJI : SitesH (planPHJ (Reg := Reg) dj x₀) (tabPHJ dj x₀) where
  A S R u _ := castM (congrArg TabSpec.S (tabOfPHJ_eq dj x₀ S R u)).symm
    (tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.twinA
  B S R u _ := castM (congrArg TabSpec.S (tabOfPHJ_eq dj x₀ S R u)).symm
    (tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.twinB
  sat S R u _ := twin_sat_cast (congrArg TabSpec.S (tabOfPHJ_eq dj x₀ S R u)).symm
    ((tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.sat (tablePHJ_typed dj x₀ ht S _) (tablePHJ_parse dj x₀ ht hc S _) hrf)
  acc S R u _ := recastAccH ((tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.acc
    (tablePHJ_typed dj x₀ ht S _) (tablePHJ_parse dj x₀ ht hc S _) hrf)
    (congrArg TabSpec.S (tabOfPHJ_eq dj x₀ S R u)).symm
  vu S _ u h := (tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfPHJ_eq dj x₀ S R u)).symm
    ((tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

theorem sitesPHJI_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).tags ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).cf
        ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).tables = .ok c :=
  fun S _ _ _ => (HiddenExec.parse_untyped (tablePHJ_typed dj x₀ ht S _) (tablePHJ_parse dj x₀ ht hc S _)).1

/-- Every site's committed circuit is the session's circuit `c` (the flat `instProg` path). -/
theorem sitesPHJI_c :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).st.c = c :=
  fun S R u h => ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).c_eq
    (HmOut.parse_isTmpl (tablePHJ_parse dj x₀ ht hc S (partOf I (dj S) u.val))) (sitesPHJI_parse dj x₀ ht hc hrf S R u h)

/-- Every site's statement circuit is the session's circuit `c`, which is flat (the `keyProg` path). -/
theorem sitesPHJI_stmt :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesPHJI dj x₀ ht hc hrf).acc S R u h).st.c = c :=
  fun S R u h => by
    rw [sitesPHJI_c dj x₀ ht hc hrf S R u h]
    exact unrolled_flat (parse_tmpl_none ht hc)

variable {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)

include hpub in
/-- `BcSiteH` at a table set up at table `j`'s inputs (`tableH_bcSiteI` at `inputsPJ I j`). -/
theorem tablePHJ_bcSiteI {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsPJ I j) mPts S' d), Tb = ⟨inputsPJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.acc p₁ p₂ hrf) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteI p₁ p₂ hrf (fun _ => d) ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hpub in
/-- **`BcSiteH` at every satisfiable site** at `J` hidden-output tables: at a table flock-verify sets up by
`tablePHJ_bcSiteI`; at a refused one no message satisfies the table. -/
theorem sitesPHJI_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planPHJ dj x₀) (tabPHJ dj x₀) S R u),
      Binding.Sat (planPHJ dj x₀) (tabPHJ dj x₀) S R u M →
        BcSiteH ((sitesPHJI dj x₀ ht hc hrf).acc S R u h) pub ((sitesPHJI dj x₀ ht hc hrf).o S R u h)
          ((sitesPHJI dj x₀ ht hc hrf).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfPHJ_eq dj x₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfPHJ_eq dj x₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupH (inputsPJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tablePHJ_bcSiteI hrf hpub _ _ _ ⟨_, _, Classical.choice hS, tablePHJ_set dj x₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planPHJ (Reg := Reg) dj x₀) (tabPHJ dj x₀) S R u).S.Satisfies z := by
      rw [tabOfPHJ_eq, tablePHJ_refused dj x₀ hS]
      exact x₀.refused_unsat _
    exact absurd hM (hns _)

/-- **`pp` at the plan, at `instProg`** (`SitesH.progPlaces_instProg` at `sitesPHJI`). -/
noncomputable abbrev ppPHJI :
    ProgPlaces (instProg c n) (outsOfI c n) (planPHJ (Reg := Reg) dj x₀) (tabPHJ dj x₀) :=
  SitesH.progPlaces_instProg (sitesPHJI_c dj x₀ ht hc hrf) hrf.1 (outsOfI c n)

/-- **`lay` at the plan, at `instProg`**: `instLayoutH` at `sitesPHJI`, the program's outputs the output words' bits, the
registration the population's public file. -/
noncomputable abbrev layPHJI :
    Binding.Layout ((instProg c n).partition (outsOfI c n)) (planPHJ (Reg := Reg) dj x₀) (tabPHJ dj x₀)
      (((instProg c n).partition (outsOfI c n)).placeDecoder (planPHJ dj x₀) (tabPHJ dj x₀)
        (ppPHJI dj x₀ ht hc hrf).derived.place) :=
  instLayoutH (sitesPHJI_c dj x₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub)

variable (hU : Built c.unit) (hpr : Layout.PortReads c hU) (hw : ProvedScope.wiresAcyclic c = true)

include ht hc x₀ hU hpr hw in
/-- **`InstKeysRow` at the session's circuit**, from the parse at table 0's reference setup (`instKeysRowH` at
`inputsPJ I 0`). -/
theorem instKeysRowPHJ : InstKeysRow c (portRowW c) :=
  instKeysRowH (typedPJ ht 0) ((parseH_tagsAt ht _ _ _ _).trans hc) (fun _ => d₀) x₀ hU hpr hw

/-- **`rs` at the plan, at `instProg`**: `instRowsH` at `sitesPHJI`, with `hsite` from `sitesPHJI_bcSite` and
`InstKeysRow` from `instKeysRowPHJ`. -/
noncomputable abbrev rsPHJI : Binding.HmRowsX (layPHJI dj x₀ ht hc hrf (pub := pub) (Reg := Reg)) :=
  instRowsH (sitesPHJI_c dj x₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (sitesPHJI_bcSite dj x₀ ht hc hrf hpub) (instKeysRowPHJ x₀ ht hc hU hpr hw)

/-- **`hHm` at the plan's rows, at `instProg`** (`instRowsH_computes`, a unit per VU from the check). -/
theorem rsPHJI_computes : (rsPHJI dj x₀ ht hc hrf hpub hU hpr hw (Reg := Reg)).Computes :=
  instRowsH_computes (sitesPHJI_c dj x₀ ht hc hrf) hrf (outsOfI c n) (fun _ => pub) outsOutI_outsOfI
    (checkFactsH ht hc).upv_pos (sitesPHJI_bcSite dj x₀ ht hc hrf hpub) (instKeysRowPHJ x₀ ht hc hU hpr hw)

/-! ## Where the wires feed neither the unit nor the outputs: the unit's key program

`HiddenExec.HeadlineExec`'s `ppH`, `layAtH`, `rsAtH` at the plan of `J` tables: the sites `sitesPHJI` at the rows' fit
`scopeOkH` gives (`rowsFit_of_scopeH`), `keyLayoutH` and `keyRowsH` at the key program `Placed.keyProg c hU n`, its
circuit conditions from `scopeOkH`. -/

variable (hscope : Layout.scopeOkH c = true)

/-- **`pp` at the plan, at `keyProg`** (`SitesH.progPlaces_keyProg` at `sitesPHJI`). -/
noncomputable abbrev ppPHJ :
    ProgPlaces (Placed.keyProg c hU n) (Layout.outsOf c hU n) (planPHJ (Reg := Reg) dj x₀) (tabPHJ dj x₀) :=
  SitesH.progPlaces_keyProg hU (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope)) (Layout.outsOf c hU n)

/-- **`lay` at the plan, at `keyProg`**: `keyLayoutH` at `sitesPHJI`, the program's outputs the returned outputs' bits,
the registration the population's public file. -/
noncomputable abbrev layPHJ :
    Binding.Layout ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (planPHJ (Reg := Reg) dj x₀)
      (tabPHJ dj x₀)
      (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planPHJ dj x₀) (tabPHJ dj x₀)
        (ppPHJ dj x₀ ht hc hU hscope).derived.place) :=
  keyLayoutH (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope)) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the plan, at `keyProg`**: `keyRowsH` at `sitesPHJI`, with `hsite` from `sitesPHJI_bcSite` and the rows'
keys from the parse (`Layout.keysRow_of_parse`). -/
noncomputable abbrev rsPHJ : Binding.HmRowsX (layPHJ dj x₀ ht hc hU hscope (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope)) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope) (Layout.outsOut_outsOf c hU n)
    (sitesPHJI_bcSite dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope) hpub)
    (Layout.keysRow_of_parse (HiddenExec.parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

/-- **`hHm` at the plan's rows, at `keyProg`** (`keyRowsH_computes`). -/
theorem rsPHJ_computes : (rsPHJ dj x₀ ht hc hpub hU hscope (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesPHJI_stmt dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope)) (Layout.outsOf c hU n)
    (fun _ => pub) (Layout.scopeOkH_spec hU hscope).2.1 (rowsFit_of_scopeH ht hc hscope)
    (Layout.outsOut_outsOf c hU n) (sitesPHJI_bcSite dj x₀ ht hc (rowsFit_of_scopeH ht hc hscope) hpub)
    (Layout.keysRow_of_parse (HiddenExec.parse_untyped ht hc).1 (Layout.scopeOkH_spec hU hscope).1
      (Layout.scopeOkH_spec hU hscope).2.2)

end FlockSoundness.Discharge.PlainHJ
