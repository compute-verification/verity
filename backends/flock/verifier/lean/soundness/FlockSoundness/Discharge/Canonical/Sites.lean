import FlockSoundness.Discharge.Canonical.Format
import FlockSoundness.Discharge.Canonical.Shared
import FlockSoundness.Discharge.ZkHidden.DefsJ

set_option Elab.async false

/-!
# The sites of a canonical session

`HiddenExec/Sites.lean`'s `TableH.acc`, `HeadlineExec.lean`'s `tableH_bcSite` and `ZkHidden/DefsJ.lean`'s
`sitesShHJ`, `sitesShHJ_parse`, `sitesShHJ_bcSite` at a circuit in the canonical format (`formatOk`), in place of
`sha512/row/v1` rows (`hv1`) and `Layout.scopeOk` (`hscope`), with `BcSiteC` (any public file, shared rows or not) for
`BcSiteH`:

* `unitRowsFlatC`: a canonical statement's unit slots' rows are its flat twin's (`HiddenExec.unitRowsFlat` without
  `hv1`).
* `accC`, `tableC_bcSite`: a drawn table's acceptance, and the site fact at it.
* `sitesShC`, `sitesShC_parse`, `tableShC_bcSite`, `sitesShC_bcSite`: the same at the public plan of `J` tables.

**Proof work.** `unitRowsFlatC` is the one `sorry`. Route: `HiddenExec.unitRowsFlat`'s proof, with
`outPairs_ne_unit` restated without `hv1`. An output pair of Δ is at a message column of an output row
(`Hidden.deltaMsgCol`). A byte below `128 · chunks` is in a `sha512x3` slot, as today (`deltaMsgCol_ne_unit`). A byte of
the last block is in its `hm96` slot's pad input (`Typed.msgCol`'s `hin[1]`), on the hm96 net, which is not the unit net
either (`Layout.ne_unit_at` at the hm96 net, `C.hm_in` for `C.sha_in`). The rest are proved here.
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden

section Flat

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- **A canonical hidden statement reads its flat twin's rows at every unit slot's row**
(`HiddenExec.unitRowsFlat` at any row width): no output pair of Δ is at a unit slot's row, whether the output byte
is in a compression's message block or in the last block's `hm96` pad input. -/
theorem unitRowsFlatC (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (hfmt : formatOk st.c = true) :
    UnitRowsFlat st := by
  sorry

end Flat

section OneTable

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)

/-- **A drawn canonical table's acceptance** (`TableH.acc` with `unitRowsFlatC` for `unitRowsFlat`). -/
noncomputable def accC {S : Finset (Fin n)} (x : TableH I mPts S) : AcceptedH x.T.S where
  tags := I.tags
  cf := I.circuitFile
  pf := I.publicFile
  coins := I.coins
  tables := I.tables
  draw := some x.d
  partition := I.partition
  program := I.program
  st := x.st
  ht := ht
  h := x.h
  L := Hidden.hidden_layout ht x.h
  hR := unitRowsFlatC ht x.h (by rw [x.c_eq hc]; exact hfmt)
  regs := x.regs
  hT := (Integrate.model_eq (parse_facts (HiddenExec.parse_untyped ht (setupHidden_parseC x.h)).1) _
    (Hidden.setupHidden_spec x.h).leaf x.hwf x.regs).trans x.hT.symm

variable (dj : Finset (Fin n) → Lean.Json) {pub : Public} (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub)

include hpub in
/-- **The site fact at a drawn canonical table** (`tableH_bcSite` without `pub.tables = none`): the VU bound
`2^⌊log₂ c.g⌋` is `formatOk`'s `vusPow2`, not `c.outNet = c.unitNet`. -/
theorem tableC_bcSite {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S)) {u : Fin n} (h : u ∈ S) :
    BcSiteC (accC ht hc hfmt x.table) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := x.table.c_eq hc
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    rw [hcx]
    exact hpub
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [hcx]
    exact formatOk_vu_lt hfmt (hcx ▸ Nat.mod_lt _ x.hg)
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

end OneTable

section Shadow

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false) {c : Flock.Circuit}
  (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hfmt : formatOk c = true)

/-- **The public plan's sites at a canonical circuit** (`sitesShHJ` with `accC`). -/
noncomputable def sitesShC : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  acc S R u _ := recastAccH (accC (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _) hfmt
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2) (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session** (`sitesShHJ_parse`). -/
theorem sitesShC_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShC dj y₀ ht hc hfmt).acc S R u h).tags ((sitesShC dj y₀ ht hc hfmt).acc S R u h).cf
        ((sitesShC dj y₀ ht hc hfmt).acc S R u h).tables = .ok c :=
  fun S _ _ _ => (HiddenExec.parse_untyped (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)).1

omit ht hfmt in
theorem bcSiteC_recast {T T' : Model.Statement} (a : AcceptedH T) (e : T = T') (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : BcSiteC a pop o g u) :
    BcSiteC (recastAccH a e) pop (Fin.cast h' o) g u :=
  hs

variable {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hfmt hpub in
/-- `BcSiteC` at a table set up at table `j`'s inputs (`tableShHJ_bcSite`). -/
theorem tableShC_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteC (accC p₁ p₂ hfmt Tb.2) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableC_bcSite p₁ p₂ hfmt (fun _ => d) ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hpub in
/-- **`BcSiteC` at every satisfiable public site** at `J` canonical tables (`sitesShHJ_bcSite`, with no
`pub.tables = none`). -/
theorem sitesShC_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteC ((sitesShC dj y₀ ht hc hfmt).acc S R u h) pub ((sitesShC dj y₀ ht hc hfmt).o S R u h)
          ((sitesShC dj y₀ ht hc hfmt).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteC_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShC_bcSite hfmt hpub _ _ _ ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

end Shadow

end FlockSoundness.Discharge.Canonical
