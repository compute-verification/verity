import Proofs.Flock.Soundness.Discharge.ZkReg.BindHJ
import Proofs.Flock.Soundness.Discharge.ZkRegV2.KeyH

set_option Elab.async false

/-!
# The rows at the hidden-output `--zk` tables of `J` tables, for `sha512/row/v2` rows

`ZkHidden/DefsJ.lean`'s shadow and `ZkReg/BindHJ.lean` at a circuit `ZkRegV2.scopeOkHV2` admits: `sitesShHJV2` is
`sitesShHJ` with each drawn table's acceptance `TableH.accV2`; `layShHJV2`, `rsShHJSV2` and `rsShHJSV2_computes` are
`layShHJ`, `rsShHJS` and `rsShHJS_computes` with `keyLayoutHV2`, `keyRowsHV2` and `keyRowsH_computesV2` (the rows'
`sha512/row/v2` bytes); `placeZHJV2`, `decAgreeHJV2`, `layZHJV2`, `rsZHJSV2` and `rsZHJSV2_computes` move them to the
`--zk` tables.
-/

namespace FlockSoundness.Discharge.ZkReg

open Flock FlockLevel3 Game Model Accounting Refine Audit ZkSession ZkLink ZkExec ZkBind ZkHidden HiddenExec Placed
  Composed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) {Reg : Type}
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : ZkRegV2.scopeOkHV2 c = true)

/-- **The public plan's sites at `J` tables, at `sha512/row/v2` rows**: `sitesShHJ` with each drawn table's acceptance
`TableH.accV2`. -/
noncomputable def sitesShHJV2 : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  acc S R u _ := recastAccH ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.accV2
    (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _) (ZkRegV2.scopeOkHV2_ports hscope))
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesShHJ_parseV2 :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHJV2 dj y₀ ht hc hscope).acc S R u h).tags
        ((sitesShHJV2 dj y₀ ht hc hscope).acc S R u h).cf ((sitesShHJV2 dj y₀ ht hc hscope).acc S R u h).tables =
        .ok c :=
  fun S _ _ _ => (parse_untyped (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hU hpub in
/-- `BcSiteH` at a table set up at table `j`'s inputs, at `sha512/row/v2` rows (`tableH_bcSiteSV2`). -/
theorem tableShHJ_bcSiteSV2 {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.accV2 p₁ p₂ (ZkRegV2.scopeOkHV2_ports hscope)) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteSV2 p₁ p₂ (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_spec hU hscope).2.1
    (fun _ => d) ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hU hpub in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables, at `sha512/row/v2` rows. -/
theorem sitesShHJ_bcSiteSV2 :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJV2 dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHJV2 dj y₀ ht hc hscope).o S R u h)
          ((sitesShHJV2 dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSiteSV2 hscope hU hpub _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan of `J` tables, at `sha512/row/v2` rows**: `keyLayoutHV2` at `sitesShHJV2`. -/
noncomputable abbrev layShHJV2 :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutHV2 (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan of `J` tables, at `sha512/row/v2` rows**: `keyRowsHV2` at `sitesShHJV2`, with `hsite`
from `sitesShHJ_bcSiteSV2`. -/
noncomputable abbrev rsShHJSV2 : Binding.HmRows (layShHJV2 dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsHV2 (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n) (sitesShHJ_bcSiteSV2 dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (ZkRegV2.scopeOkHV2_spec hU hscope).1
      (ZkRegV2.scopeOkHV2_spec hU hscope).2.2)

theorem rsShHJSV2_computes : (rsShHJSV2 dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computesV2 (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (ZkRegV2.scopeOkHV2_spec hU hscope).2.1 (ZkRegV2.scopeOkHV2_ports hscope) (ZkRegV2.scopeOkHV2_rows hscope)
    (Layout.outsOut_outsOf c hU n) (sitesShHJ_bcSiteSV2 dj y₀ ht hc hscope hU hpub)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (ZkRegV2.scopeOkHV2_spec hU hscope).1
      (ZkRegV2.scopeOkHV2_spec hU hscope).2.2)

/-! ## The `--zk` places, layout and rows -/

/-- **Each drawn unit's place in its hidden-output `--zk` table at `J` tables, at `sha512/row/v2` rows**: `keyPlaceH`'s
place at `sitesShHJV2`, along `planZHJ_S`. -/
noncomputable def placeZHJV2 (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHJV2 :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJV2 dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parseV2 dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJV2, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables, at `sha512/row/v2` rows**: `layShHJV2` moved along
`planZHJ_S`. -/
noncomputable abbrev layZHJV2 :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJV2 dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJV2 dj y₀ ht hc hscope hU (pub := pub))
    (decAgreeHJV2 dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables, at `sha512/row/v2` rows**: `rsShHJSV2` moved along
`planZHJ_S`. -/
noncomputable abbrev rsZHJSV2 : HmRowsZ (layZHJV2 dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planZHJ_S dj y₀) (decAgreeHJV2 dj y₀ ht hc hscope hU) (rsShHJSV2 dj y₀ ht hc hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' `sha512/row/v2` rows of `J` tables** (`rsShHJSV2_computes`, moved
along). -/
theorem rsZHJSV2_computes : (rsZHJSV2 dj y₀ ht hc hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJV2 dj y₀ ht hc hscope hU) _
    (rsShHJSV2_computes dj y₀ ht hc hscope hU hpub)

end FlockSoundness.Discharge.ZkReg
