import Proofs.Flock.Soundness.Discharge.ZkPubIn.RowsP
import Proofs.Flock.Soundness.Discharge.ZkPubIn.TablesJ
import Proofs.Flock.Soundness.Discharge.Canonical.Sites

set_option Elab.async false

/-!
# The sites, the layout and the rows of a canonical `--zk` session of `J` tables with the verifier's public inputs

`Canonical/Sites.lean`'s `tableC_bcSite`, `sitesShC`, `sitesShC_parse`, `sitesShC_bcSite` and `Canonical/Session.lean`'s
`layShC`, `rsShC`, `placeZC`, `decAgreeC`, `layZC`, `rsZC` at `keyProgP` (`RowsP.lean`) and the `J` tables with `own`
(`TablesJ.lean`):

* `recastAccHP`, `bcSiteHP_recast`, `loadPublicIn_tagsAt`: an acceptance at an equal statement, and `HmIn.loadPublic`
  reads no table identity;
* `DrawSetupHP.proved_rank`, `tableCP_bcSite`: a drawn canonical table's site fact (`BcSiteHP`), the VU bound
  `formatOk`'s `vusPow2`;
* `sitesShCP`, `sitesShCP_parse`, `sitesShCP_own`, `sitesShCP_bcSite`: the public plan's sites, every acceptance's
  words `own`;
* `layShCP`, `rsShCP`, `rsShCP_computes`, `placeZCP`, `decAgreeCP`, `layZCP`, `rsZCP`, `rsZCP_computes`:
  `keyLayoutCP` and `keyRowsCP` at `sitesShCP`, moved to the `--zk` tables.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed ZkHidden
open FlockSoundness.Discharge.Canonical (formatOk formatOk_vu_lt keysRowC_of_parse)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- A hidden-output acceptance with the verifier's public inputs at `T`, as one at `T' = T`. -/
def recastAccHP {T T' : Model.Statement} (a : AcceptedHP T) (e : T = T') : AcceptedHP T' :=
  { a with hT := a.hT.trans e }

theorem bcSiteHP_recast {T T' : Model.Statement} (a : AcceptedHP T) (e : T = T') (pi : HmIn.Ports) (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : BcSiteHP a pi pop o g u) :
    BcSiteHP (recastAccHP a e) pi pop (Fin.cast h' o) g u :=
  hs

/-- `HmIn.loadPublic` reads no identity, so it agrees at every table's tags. -/
theorem loadPublicIn_tagsAt (tags : Tags) (count j : ℕ) (c : Flock.Circuit) (pi : HmIn.Ports) (b : ByteArray) :
    HmIn.loadPublic (tagsAt tags count j) c pi b = HmIn.loadPublic tags c pi b := by
  unfold tagsAt
  split_ifs <;> rfl

section OneTable

variable {n mPts : ℕ} {I : Inputs} {own : ByteArray}

/-- A drawn unit's rank indexes it among the draw's proved units. -/
theorem DrawSetupHP.proved_rank {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupHP I own mPts S d) {u : Fin n}
    (h : u ∈ S) : x.D.proved[x.rank u]? = some u.val := by
  have hl : x.rank u < x.D.proved.toList.length := by simpa using x.rank_lt h
  have := List.getElem_idxOf (x := u.val) (xs := x.D.proved.toList) hl
  rw [Array.getElem?_eq_getElem (x.rank_lt h), ← Array.getElem_toList]
  exact congrArg some this

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
  (hfmt : formatOk c = true) {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
  (hpub : HmIn.loadPublic I.tags c pi I.publicFile = .ok pub)

include hpi hpub in
/-- **`BcSiteHP` at a drawn canonical table** (`Canonical.tableC_bcSite` with `own`): the table's draw decodes to
`D`, its regions are the executable's, the circuit's public input ports are `pi` and its public file loads to `pub`,
and the unit's block and VU index its rank in `D.proved`, the VU below `2^⌊log₂ c.g⌋` (`formatOk`'s `vusPow2`). -/
theorem tableCP_bcSite {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupHP I own mPts S d) {u : Fin n}
    (h : u ∈ S) : BcSiteHP (x.table.acc ht hc hfmt) pi pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : x.st.c = c := TableHP.c_eq hc x.table
  refine ⟨d, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_, ?_⟩
  · show HmIn.parse x.st.c = .ok pi
    rw [hcx]
    exact hpi
  · show HmIn.loadPublic I.tags x.st.c pi I.publicFile = .ok pub
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

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : ByteArray} (dj : Finset (Fin n) → Lean.Json)
  {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hfmt : formatOk c = true)

/-- **The public plan's sites at a canonical circuit, with the verifier's public inputs** (`Canonical.sitesShC` with
`TableHP.acc`). -/
noncomputable def sitesShCP : SitesHP (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) where
  acc S R u _ := recastAccHP ((tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.acc (tableShHJP_typed dj y₀ ht S _)
    (tableShHJP_parse dj y₀ ht hc S _) hfmt) (congrArg TabSpec.S (tabOfShHJP_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJP_eq dj y₀ S R u)).symm
    ((tableShHJP dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session** (`Canonical.sitesShC_parse`). -/
theorem sitesShCP_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShCP dj y₀ ht hc hfmt).acc S R u h).tags ((sitesShCP dj y₀ ht hc hfmt).acc S R u h).cf
        ((sitesShCP dj y₀ ht hc hfmt).acc S R u h).tables = .ok c :=
  fun S _ _ _ => (HiddenExec.parse_untyped (tableShHJP_typed dj y₀ ht S _) (tableShHJP_parse dj y₀ ht hc S _)).1

/-- **Every site checks the verifier's own words**: each acceptance is `setupHidden … (some own)`'s. -/
theorem sitesShCP_own :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S), ((sitesShCP dj y₀ ht hc hfmt).acc S R u h).own = own :=
  fun _ _ _ _ => rfl

variable {pi : HmIn.Ports} (hpi : HmIn.parse c = .ok pi) {pub : Public}
  (hpub : HmIn.loadPublic (zkTags I) c pi I.publicFile = .ok pub)

include hfmt hpi hpub in
/-- `BcSiteHP` at a table set up at table `j`'s inputs (`Canonical.tableShC_bcSite`). -/
theorem tableShCP_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableHP I' own mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupHP (inputsJ I j) own mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteHP (Tb.2.acc p₁ p₂ hfmt) pi pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableCP_bcSite p₁ p₂ hfmt hpi ((loadPublicIn_tagsAt _ _ _ _ _ _).trans hpub) x h

include hpi hpub in
/-- **`BcSiteHP` at every satisfiable public site** at `J` canonical tables (`Canonical.sitesShC_bcSite`): at a draw
`verify --zk --public-inputs F` sets up, the unit's rank in its table's draw; at a refused one no message satisfies
the table. -/
theorem sitesShCP_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJP dj y₀) (tabShHJP dj y₀) S R u),
      Binding.Sat (planShHJP dj y₀) (tabShHJP dj y₀) S R u M →
        BcSiteHP ((sitesShCP dj y₀ ht hc hfmt).acc S R u h) pi pub ((sitesShCP dj y₀ ht hc hfmt).o S R u h)
          ((sitesShCP dj y₀ ht hc hfmt).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteHP_recast _ (congrArg TabSpec.S (tabOfShHJP_eq dj y₀ S R u)).symm _ _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJP_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJP (inputsJ I (partOf I (dj S) u.val)) own mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShCP_bcSite hfmt hpi hpub _ _ _ ⟨_, _, (Classical.choice hS).x, tableShHJP_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJP (Reg := Reg) dj y₀) (tabShHJP dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJP_eq, tableShHJP_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

variable (hU : Built c.unit)

/-- **`lay` at the public plan**: `keyLayoutCP` at `sitesShCP`, the registration the population's public file. -/
noncomputable abbrev layShCP :
    Binding.Layout ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planShHJP (Reg := Reg) dj y₀)
      (tabShHJP dj y₀)
      (((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).placeDecoder (planShHJP dj y₀) (tabShHJP dj y₀)
        (keyPlaceHP (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi (Layout.outsOf c hU n))) :=
  keyLayoutCP (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi own (Layout.outsOf c hU n) (fun _ => pub) hfmt
    (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsCP` at `sitesShCP`, with `hsite` from `sitesShCP_bcSite`, the keys from
`keysRowC_of_parse` and every site's words `own` (`sitesShCP_own`). -/
noncomputable abbrev rsShCP : Binding.HmRows (layShCP dj y₀ ht hc hfmt hpi hU (pub := pub) (Reg := Reg)) :=
  keyRowsCP (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi own (Layout.outsOf c hU n) (fun _ => pub) hfmt
    (Layout.outsOut_outsOf c hU n) (sitesShCP_bcSite dj y₀ ht hc hfmt hpi hpub)
    (keysRowC_of_parse hU (HiddenExec.parse_untyped (zkTyped ht) hc).1 hfmt) (sitesShCP_own dj y₀ ht hc hfmt)

theorem rsShCP_computes : (rsShCP dj y₀ ht hc hfmt hpi hpub hU (Reg := Reg)).Computes :=
  keyRowsCP_computes (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi own (Layout.outsOf c hU n) (fun _ => pub) hfmt
    (Layout.outsOut_outsOf c hU n) (sitesShCP_bcSite dj y₀ ht hc hfmt hpi hpub)
    (keysRowC_of_parse hU (HiddenExec.parse_untyped (zkTyped ht) hc).1 hfmt) (sitesShCP_own dj y₀ ht hc hfmt)

/-- **Each drawn unit's place in its canonical `--zk` table** (`Canonical.placeZC` at `sitesShCP`). -/
noncomputable def placeZCP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProgP c hU n pi).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S u :=
  placeCast (planZHJP_S dj y₀ S R u)
    (keyPlaceHP (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree** (`Canonical.decAgreeC`). -/
theorem decAgreeCP :
    DecAgree (P := (keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZCP dj y₀ ht hc hfmt hpi hU))
      (decS := ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)).placeDecoder (planShHJP dj y₀)
        (tabShHJP dj y₀) (keyPlaceHP (sitesShCP_parse dj y₀ ht hc hfmt) pi hpi (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZCP, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the canonical `--zk` tables with the verifier's public inputs** (`Canonical.layZC`). -/
noncomputable abbrev layZCP :
    LayoutZ execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP (Reg := Reg) dj y₀)
      (tabZHJP dj y₀)
      (decodeOfPlace execArith ((keyProgP c hU n pi).partition (Layout.outsOf c hU n)) (planZHJP dj y₀)
        (tabZHJP dj y₀) (placeZCP dj y₀ ht hc hfmt hpi hU)) :=
  LayoutZ.ofShadow (planZHJP_S dj y₀) (layShCP dj y₀ ht hc hfmt hpi hU (pub := pub))
    (decAgreeCP dj y₀ ht hc hfmt hpi hU)

/-- **`rs` at the canonical `--zk` tables with the verifier's public inputs** (`Canonical.rsZC`). -/
noncomputable abbrev rsZCP : HmRowsZ (layZCP dj y₀ ht hc hfmt hpi hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planZHJP_S dj y₀) (decAgreeCP dj y₀ ht hc hfmt hpi hU) (rsShCP dj y₀ ht hc hfmt hpi hpub hU)

/-- **`hHm` at the canonical `--zk` tables' rows with the verifier's public inputs** (`Canonical.rsZC_computes`). -/
theorem rsZCP_computes : (rsZCP dj y₀ ht hc hfmt hpi hpub hU (Reg := Reg)).Computes :=
  HmRowsZ.ofShadow_computes (planZHJP_S dj y₀) (decAgreeCP dj y₀ ht hc hfmt hpi hU) _
    (rsShCP_computes dj y₀ ht hc hfmt hpi hpub hU)

end Shadow

end FlockSoundness.Discharge.ZkPubIn
