import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundBind
import Proofs.Flock.Soundness.Discharge.ZkHidden.ParseStmt
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Discharge.Hidden.Unrolled

set_option Elab.async false

/-!
# The binding half at the session's statement circuit

`SoundBind.lean`'s `sitesShH` … `rsZH` and `DefsJ.lean`'s `sitesShHJ` … `rsZHJ` take the session's circuit
from `HmOut.parse`, which refuses a templated file. Here they are at the statement circuit `c`
(`HmOut.parseStmt (zkTags I) … = .ok c`, a templated file's committed circuit unrolled), with the rows' fit `hv1`
given (`TableH.accS`'s):

* `vu_lt_pow2_stmt`: every template instance is below `2^⌊log₂ c.g⌋` at the statement circuit, from the committed
  circuit's check (`vu_lt_pow2_check`); `loadPublic_unrolled`: the public file loads alike at the committed circuit and
  the statement circuit; `tableH_bcSiteSt`: `tableH_bcSite` at `TableH.accS`.
* `sitesShHSt`, `layShHSt`, `rsShHSt`, `placeZHSt`, `decAgreeHSt`, `layZHSt`, `rsZHSt`, `rsZHSt_computes`: the
  one-table plan.
* `tableShHJ_parseStmt`, `sitesShHJSt` … `rsZHJSt_computes`: the J-table plan, `count = 1` included (table `j`'s tags
  differ from the session's in the identity alone).
* `unrolled_ports` …: `HmOut.unrolled` changes only the unit net and `tmpl`.

The flat forms are these at `parseStmt_of_parseH hc` and `rowsFit_of_scopeH … hc hscope`.
-/

namespace FlockSoundness

open Flock

section unrolled

variable (c : Circuit)

theorem unrolled_ports : (HmOut.unrolled c).ports = c.ports := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_outPorts : (HmOut.unrolled c).outPorts = c.outPorts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_outRowPorts : (HmOut.unrolled c).outRowPorts = c.outRowPorts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leavesIn : (HmOut.unrolled c).leavesIn = c.leavesIn := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leavesOut : (HmOut.unrolled c).leavesOut = c.leavesOut := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_upv : (HmOut.unrolled c).upv = c.upv := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_unitNet : (HmOut.unrolled c).unitNet = c.unitNet := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_outNet : (HmOut.unrolled c).outNet = c.outNet := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

theorem unrolled_leafCuts : (HmOut.unrolled c).leafCuts = c.leafCuts := by
  unfold HmOut.unrolled; split
  · split <;> rfl
  · rfl

end unrolled

end FlockSoundness

namespace FlockSoundness.Discharge.Layout

open Flock

/-- `vu_lt_pow2` from `HmRow.check` alone. -/
theorem vu_lt_pow2_check {c : Flock.Circuit} {tags : Tags} {ls : Lean.Json} (ht : c.typed = none)
    (hls : HmRow.check c tags ls = .ok ()) (ho : c.outNet = c.unitNet) {g : ℕ} (hg : g < c.g) :
    g < 2 ^ c.g.log2 := by
  have C := check_facts ht hls
  have hp := unit_count_pow2 ht hls ho
  rw [C.unit_range.2] at hp
  simp only [isPow2, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hp
  have hd : c.g ∣ 2 ^ (c.g * c.upv).log2 := by rw [hp.2]; exact Dvd.intro c.upv rfl
  obtain ⟨i, -, hi⟩ := (Nat.dvd_prime_pow Nat.prime_two).1 hd
  rw [hi, Nat.log2_two_pow, ← hi]
  exact hg

/-- **Every template instance is below `2^⌊log₂ c.g⌋` at the statement circuit**: a templated file's committed
circuit was checked before its template was attached, and unrolling keeps `g`, `outNet` and `unitNet`. -/
theorem vu_lt_pow2_stmt {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {c : Flock.Circuit}
    (hty : tags.typed = false) (h : HmOut.parseStmt tags cf tables = .ok c) (ho : c.outNet = c.unitNet) {g : ℕ}
    (hg : g < c.g) : g < 2 ^ c.g.log2 := by
  rcases parseStmt_cases h with ⟨-, hp⟩ | ⟨-, ct, hp, rfl⟩
  · exact vu_lt_pow2 (Hidden.parse_untyped hty hp).1 ho hg
  · obtain ⟨tu, c₀, P, -, rfl, -, -⟩ := TmplValue.parseTmpl_facts hp
    obtain ⟨ls, hls⟩ := P.check
    rw [unrolled_g] at hg ⊢
    rw [unrolled_outNet, unrolled_unitNet] at ho
    exact vu_lt_pow2_check P.typed hls ho hg

end FlockSoundness.Discharge.Layout

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Model Audit
open FlockSoundness.Discharge.Exec (Inputs)

/-- **The public file loads alike at the committed and the statement circuit**: `HmOut.loadPublic` reads the circuit's
hash, ports, output ports and class, none of which unrolling changes. -/
theorem loadPublic_unrolled {tags : Tags} {cf : ByteArray} {tables : List (String × ByteArray)} {cs : Flock.Circuit}
    (hty : tags.typed = false) (h : Hidden.Committed tags cf tables cs) (t : Tags) (pf : ByteArray) :
    HmOut.loadPublic t (HmOut.unrolled cs) pf = HmOut.loadPublic t cs pf := by
  obtain ⟨_nets, he⟩ := h.unrolled_eq hty
  rw [he]
  rfl

/-- **The site fact at a drawn table**, at the session's statement circuit (`tableH_bcSiteS` at `TableH.accS`). -/
theorem tableH_bcSiteSt {n : ℕ} {I : Inputs} {mPts : ℕ} (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hcS : HmOut.parseStmt I.tags I.circuitFile I.tables = .ok c) (hv1 : Hidden.Any.RowsFitS c)
    (hscope : Layout.scopeOkH c = true) (dj : Finset (Fin n) → Lean.Json) (hU : Built c.unit) {pub : Public}
    (hpub : HmOut.loadPublic I.tags c I.publicFile = .ok pub) {S : Finset (Fin n)} (x : DrawSetupH I mPts S (dj S))
    {u : Fin n} (h : u ∈ S) :
    BcSiteH (x.table.accS ht hcS hv1) pub (x.table.o u h) (x.table.vu u h) u.val := by
  have hcx : HmOut.unrolled x.st.c = c := x.table.unrolled_eq ht hcS
  have hC : Hidden.Committed I.tags I.circuitFile I.tables x.st.c := (Hidden.setupHidden_specCC x.table.h).committed
  refine ⟨dj S, x.D, rfl, x.hD, Layout.proved_lt_population x.hD, fun r hr => List.mem_map_of_mem hr, ?_, ?_, ?_⟩
  · show HmOut.loadPublic I.tags x.st.c I.publicFile = .ok pub
    exact (loadPublic_unrolled ht hC _ _).symm.trans (by rw [hcx]; exact hpub)
  · show x.rank u % x.st.c.g < 2 ^ x.st.c.g.log2
    rw [← unrolled_g x.st.c, hcx]
    exact Layout.vu_lt_pow2_stmt ht hcS (Layout.scopeOkH_spec hU hscope).2.1
      (Nat.mod_lt _ (by rw [← hcx, unrolled_g]; exact x.hg))
  · show x.D.proved[x.rank u / x.st.c.g * x.st.c.g + x.rank u % x.st.c.g]? = some u.val
    rw [Nat.div_add_mod']
    exact x.proved_rank h

end FlockSoundness.Discharge.HiddenExec

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Exec (Inputs)

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {Reg : Type}

/-! ## The one-table plan -/

section One

variable {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKH I mPts S₀ (dj S₀)) (ht : I.tags.typed = false)
  {c : Flock.Circuit}
  (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (hv1 : Hidden.Any.RowsFitS c)
  (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites** at the session's statement circuit (`sitesShH` at `TableH.accS`). -/
noncomputable def sitesShHSt : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  A S _ _ _ := (tableShH dj y₀ S).twinA
  B S _ _ _ := (tableShH dj y₀ S).twinB
  sat S _ _ _ := (tableShH dj y₀ S).satS (zkTyped ht) hcS hv1
  pin S _ _ _ := (tableShH dj y₀ S).pinS (zkTyped ht) hcS hv1
  acc S _ _ _ := (tableShH dj y₀ S).accS (zkTyped ht) hcS hv1
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

/-- **One circuit per session**: every site's statement circuit is the session's statement circuit `c`. -/
theorem sitesShHSt_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesShHSt dj y₀ ht hcS hv1).acc S R u h).st.c = c :=
  fun S _ _ _ => (tableShH dj y₀ S).accS_unrolled (zkTyped ht) hcS hv1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hscope hU hpub in
/-- **`BcSiteH` at every satisfiable site** of the public plan (`sitesShH_bcSite` at `tableH_bcSiteSt`). -/
theorem sitesShHSt_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShH dj y₀) (tabShH dj y₀) S R u),
      Binding.Sat (planShH dj y₀) (tabShH dj y₀) S R u M →
        BcSiteH ((sitesShHSt dj y₀ ht hcS hv1).acc S R u h) pub ((sitesShHSt dj y₀ ht hcS hv1).o S R u h)
          ((sitesShHSt dj y₀ ht hcS hv1).vu S R u h) u.val := by
  intro S R u h M hM
  show BcSiteH ((tableShH dj y₀ S).accS (zkTyped ht) hcS hv1) pub ((tableShH dj y₀ S).o u h)
    ((tableShH dj y₀ S).vu u h) u.val
  by_cases hS : Nonempty (DrawSetupZKH I mPts S (dj S))
  · have e : tableShH dj y₀ S = (Classical.choice hS).x.table := by
      unfold tableShH
      rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact tableH_bcSiteSt (zkTyped ht) hcS hv1 hscope dj hU hpub _ h
  · have e : tableShH dj y₀ S = y₀.x.refused S := by
      unfold tableShH
      rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (tableShH dj y₀ S).T.S.Satisfies z := by
      rw [e]
      exact y₀.x.refused_unsat S
    exact absurd hM (hns _)

/-- **`lay` at the public plan**, at the statement circuit (`layShH`). -/
noncomputable abbrev layShHSt :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShH (Reg := Reg) dj y₀) (tabShH dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**, at the statement circuit (`rsShH`), its keys from `Hidden.keysRow_stmt`. -/
noncomputable abbrev rsShHSt : Binding.HmRowsX (layShHSt dj y₀ ht hcS hv1 hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)
    (sitesShHSt_bcSite dj y₀ ht hcS hv1 hscope hU hpub)
    (Hidden.keysRow_stmt (zkTyped ht) hcS (Layout.scopeOkH_spec hU hscope).1 (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShHSt_computes : (rsShHSt dj y₀ ht hcS hv1 hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)
    (sitesShHSt_bcSite dj y₀ ht hcS hv1 hscope hU hpub)
    (Hidden.keysRow_stmt (zkTyped ht) hcS (Layout.scopeOkH_spec hU hscope).1 (Layout.scopeOkH_spec hU hscope).2.2)

/-- **Each drawn unit's place in its `--zk` table**, at the statement circuit (`placeZH`). -/
noncomputable def placeZHSt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u)
    (keyPlaceH (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**, at the statement circuit (`decAgreeH`). -/
theorem decAgreeHSt :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀) (placeZHSt dj y₀ ht hcS hv1 hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShH dj y₀) (tabShH dj y₀)
        (keyPlaceH (sitesShHSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHSt, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables**, at the statement circuit (`layZH`). -/
noncomputable abbrev layZHSt :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH (Reg := Reg) dj y₀)
      (tabAtZKH dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planAtZKH dj y₀)
        (tabAtZKH dj y₀)
        (placeZHSt dj y₀ ht hcS hv1 hU)) :=
  LayoutZ.ofShadow (planAtZKH_S dj y₀) (layShHSt dj y₀ ht hcS hv1 hscope hU (pub := pub))
    (decAgreeHSt dj y₀ ht hcS hv1 hU)

/-- **`rs` at the hidden-output `--zk` tables**, at the statement circuit (`rsZH`). -/
noncomputable abbrev rsZHSt : HmRowsZX (layZHSt dj y₀ ht hcS hv1 hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planAtZKH_S dj y₀) (decAgreeHSt dj y₀ ht hcS hv1 hU)
    (rsShHSt dj y₀ ht hcS hv1 hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' rows**, at the statement circuit (`rsZH_computes`). -/
theorem rsZHSt_computes : (rsZHSt dj y₀ ht hcS hv1 hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planAtZKH_S dj y₀) (decAgreeHSt dj y₀ ht hcS hv1 hU) _
    (rsShHSt_computes dj y₀ ht hcS hv1 hscope hU hpub)

end One

/-! ## The J-table plan -/

section J

variable {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)
  (ht : I.tags.typed = false) {c : Flock.Circuit}

include ht in
/-- **Every public table's inputs read the statement circuit** (`tableShHJ_parse` at `HmOut.parseStmt`). -/
theorem tableShHJ_parseStmt (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (S : Finset (Fin n))
    (j : Fin (nTab I)) :
    HmOut.parseStmt (tableShHJ dj y₀ S j).1.tags (tableShHJ dj y₀ S j).1.circuitFile
      (tableShHJ dj y₀ S j).1.tables = .ok c := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    exact (parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hcS
  · rw [tableShHJ_refused dj y₀ h]
    exact (parseStmtH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hcS

variable (hcS : HmOut.parseStmt (zkTags I) I.circuitFile I.tables = .ok c) (hv1 : Hidden.Any.RowsFitS c)
  (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites** at `J` tables, at the statement circuit (`sitesShHJ` at `TableH.accS`). -/
noncomputable def sitesShHJSt : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  A S R u _ := HiddenExec.castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinA
  B S R u _ := HiddenExec.castM (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.twinB
  sat S R u _ := HiddenExec.twin_sat_cast (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.satS (tableShHJ_typed dj y₀ ht S _)
      (tableShHJ_parseStmt dj y₀ ht hcS S _) hv1)
  pin S R u _ := HiddenExec.pin_cast (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.pinS (tableShHJ_typed dj y₀ ht S _)
      (tableShHJ_parseStmt dj y₀ ht hcS S _) hv1)
  acc S R u _ := recastAccH ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.accS
    (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parseStmt dj y₀ ht hcS S _) hv1)
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session**: every site's statement circuit is the session's statement circuit `c`. -/
theorem sitesShHJSt_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesShHJSt dj y₀ ht hcS hv1).acc S R u h).st.c = c :=
  fun S _ u _ => (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.accS_unrolled (tableShHJ_typed dj y₀ ht S _)
    (tableShHJ_parseStmt dj y₀ ht hcS S _) hv1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)

include hscope hU hpub in
/-- `BcSiteH` at a table set up at table `j`'s inputs, at the statement circuit (`tableShHJ_bcSite`). -/
theorem tableShHJ_bcSiteSt {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parseStmt Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.accS p₁ p₂ hv1) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSiteSt p₁ p₂ hv1 hscope (fun _ => d) hU ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) x h

include hscope hU hpub in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables, at the statement circuit (`sitesShHJ_bcSite`). -/
theorem sitesShHJSt_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJSt dj y₀ ht hcS hv1).acc S R u h) pub ((sitesShHJSt dj y₀ ht hcS hv1).o S R u h)
          ((sitesShHJSt dj y₀ ht hcS hv1).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSiteSt hv1 hscope hU hpub _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan** at `J` tables, at the statement circuit (`layShHJ`). -/
noncomputable abbrev layShHJSt :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan** at `J` tables, at the statement circuit (`rsShHJ`). -/
noncomputable abbrev rsShHJSt : Binding.HmRowsX (layShHJSt dj y₀ ht hcS hv1 hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)
    (sitesShHJSt_bcSite dj y₀ ht hcS hv1 hscope hU hpub)
    (Hidden.keysRow_stmt (zkTyped ht) hcS (Layout.scopeOkH_spec hU hscope).1 (Layout.scopeOkH_spec hU hscope).2.2)

theorem rsShHJSt_computes : (rsShHJSt dj y₀ ht hcS hv1 hscope hU hpub (Reg := Reg)).Computes :=
  keyRowsH_computes (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOkH_spec hU hscope).2.1 hv1 (Layout.outsOut_outsOf c hU n)
    (sitesShHJSt_bcSite dj y₀ ht hcS hv1 hscope hU hpub)
    (Hidden.keysRow_stmt (zkTyped ht) hcS (Layout.scopeOkH_spec hU hscope).1 (Layout.scopeOkH_spec hU hscope).2.2)

/-- **Each drawn unit's place in its `--zk` table** at `J` tables, at the statement circuit (`placeZHJ`). -/
noncomputable def placeZHJSt (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree** at `J` tables, at the statement circuit (`decAgreeHJ`). -/
theorem decAgreeHJSt :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJSt dj y₀ ht hcS hv1 hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJSt_parse dj y₀ ht hcS hv1) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJSt, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables**, at the statement circuit (`layZHJ`). -/
noncomputable abbrev layZHJSt :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJSt dj y₀ ht hcS hv1 hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJSt dj y₀ ht hcS hv1 hscope hU (pub := pub))
    (decAgreeHJSt dj y₀ ht hcS hv1 hU)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables**, at the statement circuit (`rsZHJ`). -/
noncomputable abbrev rsZHJSt : HmRowsZX (layZHJSt dj y₀ ht hcS hv1 hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZX.ofShadow (planZHJ_S dj y₀) (decAgreeHJSt dj y₀ ht hcS hv1 hU)
    (rsShHJSt dj y₀ ht hcS hv1 hscope hU hpub)

/-- **`hHm` at the hidden-output `--zk` tables' rows, at `J` tables**, at the statement circuit (`rsZHJ_computes`). -/
theorem rsZHJSt_computes : (rsZHJSt dj y₀ ht hcS hv1 hscope hU hpub (Reg := Reg)).Computes :=
  HmRowsZX.ofShadow_computes (planZHJ_S dj y₀) (decAgreeHJSt dj y₀ ht hcS hv1 hU) _
    (rsShHJSt_computes dj y₀ ht hcS hv1 hscope hU hpub)

end J

end FlockSoundness.Discharge.ZkHidden
