import FlockSoundness.Protocol.Discharge.Composed.DefsJ
import FlockSoundness.Protocol.Discharge.ZkHidden.SoundBind
import FlockSoundness.Protocol.Discharge.ZkHidden.SoundDefs

set_option Elab.async false

/-!
# One hidden-output `--zk` sampled session of `J` tables: the pieces `zk_session_soundHJ` reads

`Composed/DefsJ.lean`'s pieces at `J` tables, under hidden outputs (`Stmt.setupHidden`, the default `--statement`), each
the one-table hidden-output piece (`ZkHidden/Defs.lean`, `SoundBind.lean`, `SoundRuns.lean`) at table `j`:

* `setupTables_partsH`: `Tables.setupTables_parts` under hidden outputs. At `count > 1`, table `j` is `setupHidden`'s
  statement under `Tags.tableIdentity count j` at the subset draw of part `j`; `setupOf_tableJH` is `setupOf_tableJ`'s
  counterpart.
* `parseH_tagsAt`, `loadPublicH_tagsAt`: `HmOut.parse` (untyped) and `HmOut.loadPublic` read no identity.
* `DrawSetupZHJ`, `tableZHJ`, `planZHJ`, `tabZHJ`: `DrawSetupZJ`, `tableZJ`, `planZJ`, `tabZJ` with a `DrawSetupH` at
  table `j`'s inputs (`inputsJ I j`), and the refused table of the reference setup `y₀ : DrawSetupZHJ` at table 0's
  inputs.
* `tableShHJ`, `planShHJ`, `tabShHJ`, `sitesShHJ`: the public hidden-output tables (`HiddenExec.TableH`) at the same
  draws, each at the inputs it is set up on (table `j`'s, or table 0's, `y₀`'s, at a refused table), and their sites.
* `placeZHJ`, `layZHJ`, `rsZHJ`, `rsZHJ_computes`: `keyPlaceH`, `keyLayoutH` and `keyRowsH` at those sites, moved to the
  `--zk` tables (`SoundBind.lean`'s `placeZH`, `layZH`, `rsZH` at `J` tables).
* `TagsOkZHJ`, `DrawOkZKHJ`, `opnAtZHJ`, `runsZCJH_live`: `TagsOkZJ` with hidden outputs, `DrawOkZKJ` and `opnAtZJ` with
  `setupHidden`, and `runsZCJ_live` under hidden outputs.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Placed DecodeZL
  HiddenExec Composed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

/-! ## flock-verify's tables at `count > 1` under hidden outputs -/

section Parts

open Lean (Json)

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {count : ℕ} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {work : Option (ℕ × Array (String × ℕ × ℕ))}
  {stratified : Option ℕ} {closure : Option (Std.HashMap ℕ (Array ℕ))} {spec : SessionSpec} {sts : Array Flock.Stmt}

end Parts

/-! ## Table `j`'s tags under hidden outputs -/

theorem parseH_identity {tags : Tags} (ht : tags.typed = false) (x : Lean.Json) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse { tags with identity := x } b t = HmOut.parse tags b t := by
  simp only [HmOut.parse, parse_identity]
  simp only [ht, Bool.not_false, ite_true]

/-- An untyped `HmOut.parse` reads no identity, so it agrees at every table's tags. -/
theorem parseH_tagsAt {tags : Tags} (ht : tags.typed = false) (count j : ℕ) (b : ByteArray)
    (t : List (String × ByteArray)) : HmOut.parse (tagsAt tags count j) b t = HmOut.parse tags b t := by
  unfold tagsAt
  split_ifs
  · rfl
  · exact parseH_identity ht _ _ _

/-- `HmOut.loadPublic` reads no identity, so it agrees at every table's tags. -/
theorem loadPublicH_tagsAt (tags : Tags) (count j : ℕ) (c : Flock.Circuit) (b : ByteArray) :
    HmOut.loadPublic (tagsAt tags count j) c b = HmOut.loadPublic tags c b := by
  unfold tagsAt
  split_ifs <;> rfl

variable {n mPts : ℕ} {Reg : Type}

/-! ## The tables -/

/-- **A hidden-output table `verify --zk` sets up**, on table `j`'s inputs `I'` (`inputsJ I j`): `DrawSetupZKH`'s
fields at a `DrawSetupH` of `I'`, and the statement passes the `--zk` shape check (`Zk.shapeOk`). -/
structure DrawSetupZHJ (I' : Inputs) (mPts : ℕ) (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetupH I' mPts S d
  sch : Flock.Schedule
  tPad : ℕ
  hz : Zk.schedule x.st.m = .ok (sch, tPad)
  hshape : 2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
    (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
    levelsChained (schedOf sch).levels (schedOf sch).ext = true
  hmsch : (schedOf sch).m = x.st.m
  hpad : Accounting.PadOK (schedOf sch).level₀ tPad
  hN : innerN (layOf x.st sch) < 2 ^ 64
  hL : (layOf x.st sch).len + Zk.K_PAD ≤ innerN (layOf x.st sch)
  hpos : ∀ D : Finset (Fin (innerN (layOf x.st sch))),
    prCoin (fun w => innerPos _ (lt_of_lt_of_le (by simp [Zk.K_PAD]) hL) w ∈ D) ≤
      (D.card : ℝ≥0∞) / innerN (layOf x.st sch)
  hzk : Flock.Zk.shapeOk x.st = .ok ()

namespace DrawSetupZHJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZHJ I' mPts S d)

theorem innerN_pos : 0 < innerN (layOf y.x.st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64`. -/
def innerDom : Fin (innerN (layOf y.x.st y.sch)) ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin (innerN (layOf y.x.st y.sch)), j.val + 1 < 2 ^ 64 := fun j => by
      have := y.hN
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- **The `--zk` table of a hidden-output table `verify --zk` sets up**, at statement regions `regs`
(`DrawSetupZKH.tabAt`). -/
noncomputable def tabAt (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabZK 𝔽 mPts where
  S := stmtOf y.x.st y.x.wf.1 regs
  sch := schedOf y.sch
  hshape := y.hshape
  hmsch := y.hmsch
  hm := y.x.hm
  t := y.tPad
  hpad := y.hpad
  c₀ := fOf Zk.PAD_C
  e := 2
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    have h := (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).2
    rw [y.x.hmPts] at h
    exact h⟩
  np := (layOf y.x.st y.sch).len
  pads := padsOfLay (layOf y.x.st y.sch)
  padsA := padsAOfLay (layOf y.x.st y.sch)
  tr := tripleOfLay (layOf y.x.st y.sch)
  N := innerN (layOf y.x.st y.sch)
  dom := y.innerDom
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ y.innerN_pos
  hpos := y.hpos
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code y.innerDom ((layOf y.x.st y.sch).len + Zk.K_PAD) : Set (Fin (innerN (layOf y.x.st y.sch)) → 𝔽))
  hL := y.hL
  hδ := le_rfl

/-- The table's `--zk` table: setupHidden's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused table's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

end DrawSetupZHJ

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZHJ (inputsJ I 0) mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`, under hidden outputs** (`tableZJ` with `setupHidden`): the table `verify --zk` sets up at
table `j`'s inputs and draw when it sets one up, and otherwise the reference setup `y₀`'s refused table (a setup at
table 0's inputs, which every session `verify --zk` accepts has: `setupZHJ_of_verifiesZK`). -/
noncomputable def tableZHJ (S : Finset (Fin n)) (j : Fin (nTab I)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table
  else y₀.refused

/-- **The hidden-output plan of `J` tables**: at draw `S`, table `j` is `tableZHJ … S j`, for every registration. -/
noncomputable def planZHJ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => List.ofFn (tableZHJ dj y₀ S)

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabZHJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZHJ (Reg := Reg) dj y₀ S R).length :=
  fun S R u => Fin.cast (planZHJ_length dj y₀ S R).symm (partOf I (dj S) u.val)

theorem tabOfZHJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u = tableZHJ dj y₀ S (partOf I (dj S) u.val) := by
  simp only [tabOfZ, planZHJ, tabZHJ, List.get_ofFn]
  rfl

/-! ## The public shadow, the sites, the layout and the rows -/

open Classical in
/-- **The public hidden-output table at table `j`'s draw**, with the inputs it is set up on: its `DrawSetupZHJ`'s
table at table `j`'s inputs, or `y₀`'s refused one at table 0's inputs. -/
noncomputable def tableShHJ (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (I' : Inputs) × TableH I' mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    ⟨inputsJ I j, (Classical.choice h).x.table⟩
  else ⟨inputsJ I 0, y₀.x.refused _⟩

theorem tableShHJ_set {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJ dj y₀ S j = ⟨inputsJ I j, (Classical.choice h).x.table⟩ := by
  unfold tableShHJ
  rw [dite_eq_left_of_eq_true (eq_true h)]

theorem tableShHJ_refused {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : ¬ Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tableShHJ dj y₀ S j = ⟨inputsJ I 0, y₀.x.refused _⟩ := by
  unfold tableShHJ
  rw [dite_eq_right_of_eq_false (eq_false h)]

theorem tableZHJ_S (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (tableZHJ dj y₀ S j).S = (tableShHJ dj y₀ S j).2.T.S := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    unfold tableZHJ
    rw [dite_eq_left_of_eq_true (eq_true h)]
    rfl
  · rw [tableShHJ_refused dj y₀ h]
    unfold tableZHJ
    rw [dite_eq_right_of_eq_false (eq_false h)]
    rfl

/-- The public plan at the same draws. -/
noncomputable def planShHJ : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tableShHJ dj y₀ S j).2.T

noncomputable def tabShHJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShHJ (Reg := Reg) dj y₀ S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfShHJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planShHJ dj y₀) (tabShHJ dj y₀) S R u = (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.T := by
  simp only [Partition.tabOf, planShHJ, tabShHJ, List.get_ofFn]
  rfl

theorem tableShHJ_tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.T.S =
      (tabOfZ (planZHJ (Reg := Reg) dj y₀) (tabZHJ dj y₀) S R u).S := by
  rw [tabOfZHJ_eq, tableZHJ_S dj y₀]

theorem planZHJ_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S = (Partition.tabOf (planShHJ dj y₀) (tabShHJ dj y₀) S R u).S :=
  fun S R u => (tableShHJ_tabOfZ dj y₀ S R u).symm.trans (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm

variable (ht : I.tags.typed = false)

include ht in
theorem tableShHJ_typed (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableShHJ dj y₀ S j).1.tags.typed = false := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    exact typedJ ht j
  · rw [tableShHJ_refused dj y₀ h]
    exact typedJ ht 0

variable {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)

include ht hc in
/-- **One circuit per session, at every table**: each public table's inputs parse the session's circuit file to `c`
(table `j`'s tags differ from the session's in the identity alone, which an untyped `HmOut.parse` does not read). -/
theorem tableShHJ_parse (S : Finset (Fin n)) (j : Fin (nTab I)) :
    HmOut.parse (tableShHJ dj y₀ S j).1.tags (tableShHJ dj y₀ S j).1.circuitFile (tableShHJ dj y₀ S j).1.tables =
      .ok c := by
  by_cases h : Nonempty (DrawSetupZHJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))
  · rw [tableShHJ_set dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc
  · rw [tableShHJ_refused dj y₀ h]
    exact (parseH_tagsAt ((zkTags_typed I).trans ht) _ _ _ _).trans hc

/-- A hidden-output acceptance at `T`, as one at `T' = T`. -/
def recastAccH {T T' : Model.Statement} (a : AcceptedH T) (e : T = T') : AcceptedH T' := { a with hT := a.hT.trans e }

variable (hscope : Layout.scopeOk c = true)

/-- **The public plan's sites**: each drawn unit's acceptance, VU and block are its table's public table's. -/
noncomputable def sitesShHJ : SitesH (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) where
  acc S R u _ := recastAccH ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.acc
    (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _) (v2_of_scope hscope))
    (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm
  vu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.vu u (mem_partS h)
  hvu S _ u h := (tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm
    ((tableShHJ dj y₀ S (partOf I (dj S) u.val)).2.o u (mem_partS h))

/-- **One circuit per session**: every site's acceptance parses the session's one circuit file to `c`. -/
theorem sitesShHJ_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmRow.parse ((sitesShHJ dj y₀ ht hc hscope).acc S R u h).tags ((sitesShHJ dj y₀ ht hc hscope).acc S R u h).cf
        ((sitesShHJ dj y₀ ht hc hscope).acc S R u h).tables = .ok c :=
  fun S _ _ _ => (parse_untyped (tableShHJ_typed dj y₀ ht S _) (tableShHJ_parse dj y₀ ht hc S _)).1

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

omit ht in
theorem bcSiteH_recast {T T' : Model.Statement} (a : AcceptedH T) (e : T = T') (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : BcSiteH a pop o g u) :
    BcSiteH (recastAccH a e) pop (Fin.cast h' o) g u :=
  hs

include hU hpub hpt in
/-- `BcSiteH` at a table set up at table `j`'s inputs (`tableH_bcSite` at `inputsJ I j`). -/
theorem tableShHJ_bcSite {S' : Finset (Fin n)} (Tb : (I' : Inputs) × TableH I' mPts S')
    (p₁ : Tb.1.tags.typed = false) (p₂ : HmOut.parse Tb.1.tags Tb.1.circuitFile Tb.1.tables = .ok c)
    (hX : ∃ (j : ℕ) (d : Lean.Json) (x : DrawSetupH (inputsJ I j) mPts S' d), Tb = ⟨inputsJ I j, x.table⟩)
    {u : Fin n} (h : u ∈ S') :
    BcSiteH (Tb.2.acc p₁ p₂ (v2_of_scope hscope)) pub (Tb.2.o u h) (Tb.2.vu u h) u.val := by
  obtain ⟨j, d, x, rfl⟩ := hX
  exact tableH_bcSite p₁ p₂ hscope (fun _ => d) hU ((loadPublicH_tagsAt _ _ _ _ _).trans hpub) hpt x h

include hU hpub hpt in
/-- **`BcSiteH` at every satisfiable public site** at `J` tables (`sitesShH_bcSite`): at a table `verify --zk` sets up
by `tableH_bcSite` at table `j`'s inputs; at a refused one no message satisfies the table. -/
theorem sitesShHJ_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShHJ dj y₀) (tabShHJ dj y₀) S R u),
      Binding.Sat (planShHJ dj y₀) (tabShHJ dj y₀) S R u M →
        BcSiteH ((sitesShHJ dj y₀ ht hc hscope).acc S R u h) pub ((sitesShHJ dj y₀ ht hc hscope).o S R u h)
          ((sitesShHJ dj y₀ ht hc hscope).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSiteH_recast _ (congrArg TabSpec.S (tabOfShHJ_eq dj y₀ S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShHJ_eq dj y₀ S R u)).symm _ _ ?_
  by_cases hS : Nonempty (DrawSetupZHJ (inputsJ I (partOf I (dj S) u.val)) mPts
    (partS I (dj S) S (partOf I (dj S) u.val)) (drawJ I (dj S) (partOf I (dj S) u.val)))
  · exact tableShHJ_bcSite hscope hU hpub hpt _ _ _
      ⟨_, _, (Classical.choice hS).x, tableShHJ_set dj y₀ hS⟩ (mem_partS h)
  · have hns : ∀ z, ¬ (Partition.tabOf (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀) S R u).S.Satisfies z := by
      rw [tabOfShHJ_eq, tableShHJ_refused dj y₀ hS]
      exact y₀.x.refused_unsat _
    exact absurd hM (hns _)

/-- **`lay` at the public plan**: `keyLayoutH` at `sitesShHJ`. -/
noncomputable abbrev layShHJ :
    Binding.Layout ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planShHJ (Reg := Reg) dj y₀) (tabShHJ dj y₀)
      (((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) :=
  keyLayoutH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)

/-- **`rs` at the public plan**: `keyRowsH` at `sitesShHJ`, with `hsite` from `sitesShHJ_bcSite`. -/
noncomputable abbrev rsShHJ : Binding.HmRows (layShHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  keyRowsH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) (fun _ => pub)
    (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope) (Layout.outsOut_outsOf c hU n)
    (sitesShHJ_bcSite dj y₀ ht hc hscope hU hpub hpt)
    (Layout.keysRow_of_parse (parse_untyped (zkTyped ht) hc).1 (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2)

/-- **Each drawn unit's place in its hidden-output `--zk` table at `J` tables**: `keyPlaceH`'s place in the public
table, along `planZHJ_S`. -/
noncomputable def placeZHJ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planZHJ dj y₀) (tabZHJ dj y₀) S R u).S u :=
  placeCast (planZHJ_S dj y₀ S R u)
    (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

/-- **The decoders agree**: the `--zk` places and the public places read one witness alike. -/
theorem decAgreeHJ :
    DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ_S (Reg := Reg) dj y₀)
      (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀)
        (tabZHJ dj y₀) (placeZHJ dj y₀ ht hc hscope hU))
      (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShHJ dj y₀) (tabShHJ dj y₀)
        (keyPlaceH (sitesShHJ_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM _ g _
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, placeZHJ, decode_placeCast]
  exact congrArg (fun z => UnitPlace.decode _ z g) hM.symm

/-- **`lay` at the hidden-output `--zk` tables of `J` tables**: `layShHJ` moved along `planZHJ_S`. -/
noncomputable abbrev layZHJ :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ (Reg := Reg) dj y₀)
      (tabZHJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZHJ dj y₀) (tabZHJ dj y₀)
        (placeZHJ dj y₀ ht hc hscope hU)) :=
  LayoutZ.ofShadow (planZHJ_S dj y₀) (layShHJ dj y₀ ht hc hscope hU (pub := pub)) (decAgreeHJ dj y₀ ht hc hscope hU)

/-- **`rs` at the hidden-output `--zk` tables of `J` tables**: `rsShHJ` moved along `planZHJ_S`. -/
noncomputable abbrev rsZHJ : HmRowsZ (layZHJ dj y₀ ht hc hscope hU (pub := pub) (Reg := Reg)) :=
  HmRowsZ.ofShadow (planZHJ_S dj y₀) (decAgreeHJ dj y₀ ht hc hscope hU) (rsShHJ dj y₀ ht hc hscope hU hpub hpt)

end Shadow

/-! ## The tags, the coin server's draws, and the openings -/

/-- **The coin server's draw at `S`, at `J` hidden-output tables** (`DrawOkZKJ` with `setupHidden` for `setupH`): each
table's draw proves the drawn units it is given (`partS`), and setupHidden sets each table up at it with `mPts` link
points. -/
def DrawOkZKHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j)) I.partition I.program = .ok st → st.mPts = mPts

/-- **The `--zk` openings at hidden-output table `j`** (`opnAtZJ` with `setupHidden`): `opnZ` at the statement
setupHidden sets up under table `j`'s tags at its draw, and its `--zk` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnAtZHJ (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (planZ S R).length) : OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupHidden (tagsAt (zkTags I) I.count j.val) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZJ ((planZ S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | .error _ => default

/-! ## The message half at the live hidden-output tables -/

end FlockSoundness.Discharge.ZkHidden
