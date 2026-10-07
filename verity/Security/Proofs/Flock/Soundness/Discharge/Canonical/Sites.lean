import Proofs.Flock.Soundness.Discharge.Canonical.Format
import Proofs.Flock.Soundness.Discharge.Canonical.OutCopy
import Proofs.Flock.Soundness.Discharge.Canonical.Shared
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

set_option Elab.async false

/-!
# The sites of a canonical session

`HiddenExec/Sites.lean`'s `TableH.acc`, `HeadlineExec.lean`'s `tableH_bcSite` and `ZkHidden/DefsJ.lean`'s
`sitesShHJ`, `sitesShHJ_parse`, `sitesShHJ_bcSite` at a circuit in the canonical format (`formatOk`), in place of
`sha512/row/v1` rows (`hv1`) and `Layout.scopeOk` (`hscope`), with `BcSiteC` (any public file, shared rows or not) for
`BcSiteH`:

* `outPairs_ne_unitC`, `unitRowsFlatC`: a canonical statement's unit slots' rows are its flat twin's
  (`HiddenExec.unitRowsFlat` without `hv1`).
* `accC`, `tableC_bcSite`: a drawn table's acceptance, and the site fact at it.
* `sitesShC`, `sitesShC_parse`, `tableShC_bcSite`, `sitesShC_bcSite`: the same at the public plan of `J` tables.

Every declaration here is proved. `unitRowsFlatC` is `HiddenExec.unitRowsFlat`'s proof with `outPairs_ne_unitC` for
`outPairs_ne_unit`: an output pair of Δ is at a message column of an output row (`Hidden.deltaMsgCol`); a byte below
`128 · chunks` is in a `sha512x3` slot, as before (`deltaMsgCol_ne_unit`), and a byte of the last block is in its `hm96`
slot's pad input (`deltaMsgCol_hm`), on the hm96 net, which is not the unit net either (`Layout.ne_unit_at`).
-/

namespace FlockSoundness.Discharge.Canonical

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden

section Flat

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Flock.Stmt}

/-- **No output pair of Δ is at a unit slot's row**, at canonical output rows (`outPairs_ne_unit` without `hv1`): an
output byte is in a compression's message block (`deltaMsgCol_ne_unit`) or in the last block's `hm96` pad input, on
the `hm96` net, which is not the unit's. -/
theorem outPairs_ne_unitC (L : Layout st) (P : ParseFacts tags st.c) (C : CheckFacts st.c)
    (hsha : st.c.unitNet ≠ st.c.shaNet) (hhm : st.c.unitNet ≠ st.c.hmNet) (hrows : OutRowsC st.c) {q : ℕ × ℕ}
    (hq : q ∈ Hidden.outPairs st.c) {s₀ b : ℕ} (hs₀ : s₀ < st.c.unitRange.count) (hb : b < 2 ^ st.c.unit.unitLog) :
    q.1 ≠ st.c.unitRange.slot s₀ + b := by
  obtain ⟨g, hg, j, hj, k, hk, t, ht, hq1⟩ := outPairs_fst hq
  obtain ⟨hp, hbc, -, ht8⟩ := outRowC_bound hrows hj hk ht
  have hs := Hidden.segs_getD hrows.1 hp
  rw [hq1]
  rcases Nat.lt_or_ge ((2 * k + t / 8) / 128) st.c.ports[HmOut.nIn st.c + j]!.chunks with hlt | hge
  · exact deltaMsgCol_ne_unit L P C hsha hg hp hs hlt ht8 hs₀ hb
  · rw [deltaMsgCol_hm P.rowNets hs hp hge]
    have hsz : (portCols (st.c.nets.getD st.c.hmNet default).2.inGroups).size = 3 := by
      simpa using congrArg Array.size C.hm_in
    have hmem := getElem_mem_toList' (xs := portCols (st.c.nets.getD st.c.hmNet default).2.inGroups) (i := 1)
      (by omega)
    rw [(Hm.hmIn_cols P.rowNets).2] at hmem
    have hne := L.ne_unit_at P C P.hmNet hhm.symm hg
      (by rw [C.perVu_hm]; exact lt_of_lt_of_le hp (le_nextPow2 _)) hmem
      (tt := Hidden.msgOff (2 * k + t / 8) (t % 8)) (Hidden.msgOff_lt ht8) hs₀ hb
    rwa [Nat.add_assoc] at hne

/-- **A canonical hidden statement reads its flat twin's rows at every unit slot's row**
(`HiddenExec.unitRowsFlat` at any row width): no output pair of Δ is at a unit slot's row, whether the output byte
is in a compression's message block or in the last block's `hm96` pad input. -/
theorem unitRowsFlatC (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (hfmt : formatOk st.c = true) :
    UnitRowsFlat st := by
  have s := Hidden.setupHidden_spec' h
  obtain ⟨hp, hcr⟩ := Hidden.parse_untyped ht s.parse
  have L := flat_layout ht h
  have F := flat_unitFacts ht h
  have hrows := (outRowsC_of (unit_built F.parse F.check) hp hcr hfmt).1
  intro s₀ hs₀ b hb r j hr
  have hout : ∀ q ∈ Hidden.outPairs st.c, q.1 ≠ r.val := fun q hq => by
    rw [hr]; exact outPairs_ne_unitC L F.parse F.check F.sha F.hm hrows hq hs₀ hb
  have hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray := by rw [s.da, Hidden.hidden_delta']
  have hdb : st.db = (HmRow.delta st.c st.pin).2 ++ (Hidden.outPairs st.c).toArray := by rw [s.db, Hidden.hidden_delta']
  refine ⟨?_, ?_⟩
  · show rangesEntry st (slotEntryA st) r.val j.val + deltaEntry st.da r.val j.val =
      rangesEntry st (slotEntryA st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).1 r.val j.val
    rw [hda, Hidden.deltaEntry_append_out hout]
  · show rangesEntry st (slotEntryB st) r.val j.val + deltaEntry st.db r.val j.val =
      rangesEntry st (slotEntryB st) r.val j.val + deltaEntry (HmRow.delta st.c st.pin).2 r.val j.val
    rw [hdb, Hidden.deltaEntry_append_out hout]

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
  {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
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
