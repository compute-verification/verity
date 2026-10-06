import Proofs.Flock.Soundness.Discharge.Composed.Defs
import Proofs.Flock.Soundness.Discharge.DecodeZLJ
import Proofs.Flock.Soundness.Discharge.Tables
import Proofs.Flock.Soundness.Discharge.Composed.TagsJ

/-!
# One `--zk` sampled session of `J` tables: the pieces `zk_session_soundJ` reads

`zk_session_sound` (`Composed/Sound.lean`) reads one table. `flock-circuit serve|prove --zk --session-tables J` on the
CPU prover (`--gpu` refuses `J > 1`), and `flock-verify verify --zk --session-tables J`, run `J` tables in one session.
`Stmt.setupTables` then sets table `j` up under `tagsAt` (`Tags.tableIdentity count j`) at part `j` of the session's
subset draw (`partJson`, `accepts_tables`); at one table it is `zk_session_sound`'s table. Here are the pieces at `J`
tables, each the one-table piece's counterpart:

* `nTab`, `inputsJ`, `drawJ`, `partOf`, `partS`: the number of tables, table `j`'s inputs and draw, the table unit `u`
  is drawn into (its position among the draw's units over the parts' size), and the drawn units of table `j`.
* `DrawSetupZJ`, `tableZJ`, `planZJ`, `tabZJ`: `DrawSetupZK`, `tableAtZK`, `planZAt` and `tabZAt` at table `j`'s inputs
  and draw (where `verify --zk` sets none up, the refused table of `y₀`, a `DrawSetupZJ` at table 0's inputs).
* `tableShJ`, `planShJ`, `sitesShJ`, `sitesZJ`, `layZJ`, `rsZJ`, `rsZJ_computes`: `ZkBind`'s public shadow, sites,
  layout and rows (`At.lean`) at `J` tables.
* `liveOfZJ`, `DecodesOnZKJ`, `simZJ`, `zk_headline_exec_tables`: `Tables.liveOfJ`, `DecodesOnJ`, `simAuditJ` and
  `execRefines_tables` at the compiled `--zk` tables.
* `RunsOnZCJ`, `decodesOnZKJ_of_runsOnZCJ`: `RunsOnZC` and `decodesOnZK_of_runsOnZC` table by table.
* `TagsOkZJ`, `RecordsLiveZKJ`, `DrawOkZKJ`: `TagsOk` without `count ≤ 1`, and `RecordsLiveZK`, `DrawOkZK` at `J` tables.
* `runsZCJ_live`: `runsZC_live` at `J` tables.
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession modelSession posOf DecodesTable simSession live_le_session
  AllOk repTapeJ)
open FlockSoundness.Discharge.DecodeZLJ (partJson)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

variable {n mPts : ℕ} {Reg : Type}

/-! ## The session's tables -/

/-- The number of tables `verify --zk` sets a session up with (`Tags.tableNames`): one at `count ≤ 1`, else `count`. -/
def nTab (I : Inputs) : ℕ := if I.count ≤ 1 then 1 else I.count

theorem nTab_pos (I : Inputs) : 0 < nTab I := by
  unfold nTab
  split <;> omega

/-- Table `j`'s inputs: `zkInputs I` under table `j`'s tags (`tagsAt`: the statement's at one table, under
`Tags.tableIdentity count j` at several). -/
def inputsJ (I : Inputs) (j : ℕ) : Inputs := { zkInputs I with tags := tagsAt (zkTags I) I.count j }

/-- Table `j`'s draw: the session's draw at one table, its part `j` (`partJson`) at several. -/
def drawJ (I : Inputs) (d : Lean.Json) (j : ℕ) : Lean.Json := if I.count ≤ 1 then d else partJson d I.count j

/-- **The table unit `u` is drawn into** at the session's draw `d`: its position among `d`'s units over the parts' size
(`partDraw`'s equal consecutive parts), within the tables; table 0 at one table. -/
def partOf (I : Inputs) (d : Lean.Json) (u : ℕ) : Fin (nTab I) :=
  ⟨min (match Draw.UnitDraw.ofJson d with
      | .ok D => D.units.toList.idxOf u / (D.units.size / I.count)
      | .error _ => 0) (nTab I - 1), by have := nTab_pos I; omega⟩

/-- The drawn units of `S` that table `j` proves. -/
def partS (I : Inputs) (d : Lean.Json) (S : Finset (Fin n)) (j : Fin (nTab I)) : Finset (Fin n) :=
  S.filter fun u => partOf I d u.val = j

theorem mem_partS {I : Inputs} {d : Lean.Json} {S : Finset (Fin n)} {u : Fin n} (h : u ∈ S) :
    u ∈ partS I d S (partOf I d u.val) :=
  Finset.mem_filter.2 ⟨h, rfl⟩

/-- **A table `verify --zk` sets up**, on table `j`'s inputs `I'` (`inputsJ I j`): `DrawSetupZK`'s fields at a
`DrawSetup` of `I'`. -/
structure DrawSetupZJ (I' : Inputs) (mPts : ℕ) (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetup I' mPts S d
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

namespace DrawSetupZJ

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZJ I' mPts S d)

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

/-- **The `--zk` table of a table `verify --zk` sets up**, at statement regions `regs` (`DrawSetupZK.tabAt`). -/
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

/-- The table's `--zk` table: setupH's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused table's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

end DrawSetupZJ

section Shadow

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`** (`tableAtZK` at table `j`): the table `verify --zk` sets up at table `j`'s inputs and draw
when it sets one up, and otherwise the reference setup `y₀`'s refused table (a setup at table 0's inputs, which every
session `verify --zk` accepts at a draw `DrawOkZKJ` admits has: `setupZJ_of_acceptsZK`). -/
noncomputable def tableZJ (S : Finset (Fin n)) (j : Fin (nTab I)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).table
  else y₀.refused

/-- **The plan of `J` tables**: at draw `S`, table `j` is `tableZJ … S j`, for every registration. -/
noncomputable def planZJ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => List.ofFn (tableZJ dj y₀ S)

theorem planZJ_length (S : Finset (Fin n)) (R : Reg) : (planZJ (Reg := Reg) dj y₀ S R).length = nTab I :=
  List.length_ofFn

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabZJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZJ (Reg := Reg) dj y₀ S R).length :=
  fun S R u => Fin.cast (planZJ_length dj y₀ S R).symm (partOf I (dj S) u.val)

theorem tabOfZJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    tabOfZ (planZJ dj y₀) (tabZJ dj y₀) S R u = tableZJ dj y₀ S (partOf I (dj S) u.val) := by
  simp only [tabOfZ, planZJ, tabZJ, List.get_ofFn]
  rfl

/-! ## The public shadow, the sites, the layout and the rows -/

variable (ht : I.tags.typed = false)

include ht in
theorem typedJ (j : ℕ) : (inputsJ I j).tags.typed = false := (Exec.tagsAt_typed _ _ _).trans (zkTyped ht)

include ht in
open Classical in
/-- **The public table at table `j`'s draw** (`tableSh`): its `DrawSetupZJ`'s table, or `y₀`'s refused one. -/
noncomputable def tableShJ (S : Finset (Fin n)) (j : Fin (nTab I)) : Integrate.Table mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetupZJ (inputsJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    (Classical.choice h).x.table (typedJ ht j)
  else y₀.x.refused (typedJ ht 0) _

theorem tableZJ_S (S : Finset (Fin n)) (j : Fin (nTab I)) : (tableZJ dj y₀ S j).S = (tableShJ dj y₀ ht S j).T.S := by
  unfold tableZJ tableShJ
  split_ifs with h
  · exact (table_stmtOf (typedJ ht j) _).symm
  · exact (model_eq_stmtOf (typedJ ht 0) y₀.x.h y₀.x.wf.1 _).symm

/-- The public plan at the same draws. -/
noncomputable def planShJ : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tableShJ dj y₀ ht S j).T

noncomputable def tabShJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShJ (Reg := Reg) dj y₀ ht S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfShJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u = (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).T := by
  simp only [Partition.tabOf, planShJ, tabShJ, List.get_ofFn]
  rfl

theorem tableShJ_tabOfZ (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).T.S = (tabOfZ (planZJ (Reg := Reg) dj y₀) (tabZJ dj y₀) S R u).S := by
  rw [tabOfZJ_eq, tableZJ_S dj y₀ ht]

theorem planZJ_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planZJ dj y₀) (tabZJ dj y₀) S R u).S = (Partition.tabOf (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u).S :=
  fun S R u => (tableShJ_tabOfZ dj y₀ ht S R u).symm.trans (congrArg TabSpec.S (tabOfShJ_eq dj y₀ ht S R u)).symm

/-- The public plan's sites: each drawn unit's acceptance, VU and block are its table's. -/
noncomputable def sitesShJ : Placed.Sites (planShJ (Reg := Reg) dj y₀ ht) (tabShJ dj y₀ ht) where
  acc S R u _ := recastAcc (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc
    (congrArg TabSpec.S (tabOfShJ_eq dj y₀ ht S R u)).symm
  vu S _ u h := (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).vu u (mem_partS h)
  hvu S _ u h := (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShJ_eq dj y₀ ht S R u)).symm
    ((tableShJ dj y₀ ht S (partOf I (dj S) u.val)).o u (mem_partS h))

/-- **The `--zk` sites at `J` tables**: each drawn unit's acceptance, VU and block are its table's public table's. -/
noncomputable def sitesZJ : SitesZ (planZJ (Reg := Reg) dj y₀) (tabZJ dj y₀) where
  acc S R u _ := recastAcc (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc (tableShJ_tabOfZ dj y₀ ht S R u)
  vu S _ u h := (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).vu u (mem_partS h)
  hvu S _ u h := (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).hvu u (mem_partS h)
  o S R u h := Fin.cast (congrArg (fun T : Model.Statement => 2 ^ T.nbl) (tableShJ_tabOfZ dj y₀ ht S R u))
    ((tableShJ dj y₀ ht S (partOf I (dj S) u.val)).o u (mem_partS h))

variable {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c)

include hc in
/-- **One circuit per session, at every table**: every public site's acceptance parses the session's circuit file to
`c` (table `j`'s tags differ from the session's in the identity alone, which the parse does not read). -/
theorem sitesShJ_parse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse ((sitesShJ dj y₀ ht).acc S R u h).tags ((sitesShJ dj y₀ ht).acc S R u h).cf
      ((sitesShJ dj y₀ ht).acc S R u h).tables = .ok c := by
  intro S R u h
  show HmRow.parse (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc.tags
    (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc.cf (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc.tables = .ok c
  unfold tableShJ
  by_cases hS : Nonempty (DrawSetupZJ (inputsJ I (partOf I (dj S) u.val)) mPts (partS I (dj S) S (partOf I (dj S) u.val))
    (drawJ I (dj S) (partOf I (dj S) u.val)))
  · rw [dite_eq_left_of_eq_true (eq_true hS)]; exact (parse_tagsAt _ _ _ _ _).trans hc
  · rw [dite_eq_right_of_eq_false (eq_false hS)]; exact (parse_tagsAt _ _ _ _ _).trans hc

include hc in
theorem sitesZJ_parse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse ((sitesZJ dj y₀ ht).acc S R u h).tags ((sitesZJ dj y₀ ht).acc S R u h).cf
      ((sitesZJ dj y₀ ht).acc S R u h).tables = .ok c :=
  sitesShJ_parse dj y₀ ht hc

variable (hU : Built c.unit) {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none) (hscope : Layout.scopeOk c = true)

/-- **The decoders agree** at `J` tables (`decAgreeAt`). -/
theorem decAgreeJ : DecAgree (P := (keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ_S (Reg := Reg) dj y₀ ht)
    (decZ := decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
      ((sitesZJ dj y₀ ht).placeKey hU (sitesZJ_parse dj y₀ ht hc) (Layout.outsOf c hU n)))
    (decS := ((keyProg c hU n).partition (Layout.outsOf c hU n)).placeDecoder (planShJ dj y₀ ht) (tabShJ dj y₀ ht)
      (Layout.keyPlace (sitesShJ_parse dj y₀ ht hc) (Layout.outsOf c hU n))) := by
  intro S R u hu M' M hM hM' g hg
  simp only [decodeOfPlace, Partition.placeDecoder, hu, ↓reduceDIte, UnitPlace.decode]
  congr 1
  funext x
  congr 1
  have hM2 : witnessOf execArith (Partition.tabOf (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u).S.m
      (Partition.tabOf (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u).sch.k0
      (Partition.tabOf (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u).sch.level₀.logCols M =
      witOf (planZJ_S dj y₀ ht S R u) (witZ execArith (planZJ dj y₀) (tabZJ dj y₀) S R u M') := hM
  rw [hM2]
  exact block_witOf _ _ _ _ _ _ rfl rfl

/-- **`lay` at `J` tables**: e2e-layout's `keyedLayout` at the public sites, moved to the `--zk` tables. -/
noncomputable abbrev layZJ :
    LayoutZ execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ (Reg := Reg) dj y₀) (tabZJ dj y₀)
      (decodeOfPlace execArith ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZJ dj y₀) (tabZJ dj y₀)
        ((sitesZJ dj y₀ ht).placeKey hU (sitesZJ_parse dj y₀ ht hc) (Layout.outsOf c hU n))) :=
  LayoutZ.ofShadow (planZJ_S dj y₀ ht)
    (Layout.keyedLayout (sitesShJ_parse dj y₀ ht hc) (Layout.outsOf c hU n) Hm.saltCol fun _ => pub)
    (decAgreeJ dj y₀ ht hc hU)

omit ht in
theorem bcSite_recast {T T' : Model.Statement} (a : Placed.Accepted T) (e : T = T') (pop : Public)
    (o : Fin (2 ^ T.nbl)) (h' : 2 ^ T.nbl = 2 ^ T'.nbl) (g u : ℕ) (hs : Layout.BcSite a pop o g u) :
    Layout.BcSite (recastAcc a e) pop (Fin.cast h' o) g u :=
  hs

include hc hU hpub hpt hscope in
/-- **`BcSite` at every satisfiable public site** at `J` tables (`sitesSh_bcSite`): at a table `verify --zk` sets up by
`table_bcSite` at table `j`'s inputs; at a refused one no message satisfies the table. -/
theorem sitesShJ_bcSite :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S)
      (M : Binding.Msg (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u),
      Binding.Sat (planShJ dj y₀ ht) (tabShJ dj y₀ ht) S R u M →
        Layout.BcSite ((sitesShJ dj y₀ ht).acc S R u h) pub ((sitesShJ dj y₀ ht).o S R u h)
          ((sitesShJ dj y₀ ht).vu S R u h) u.val := by
  intro S R u h M hM
  refine bcSite_recast _ (congrArg TabSpec.S (tabOfShJ_eq dj y₀ ht S R u)).symm _ _
    (congrArg (fun T : TabSpec mPts => 2 ^ T.S.nbl) (tabOfShJ_eq dj y₀ ht S R u)).symm _ _ ?_
  show Layout.BcSite (tableShJ dj y₀ ht S (partOf I (dj S) u.val)).acc pub
    ((tableShJ dj y₀ ht S (partOf I (dj S) u.val)).o u (mem_partS h))
    ((tableShJ dj y₀ ht S (partOf I (dj S) u.val)).vu u (mem_partS h)) u.val
  by_cases hS : Nonempty (DrawSetupZJ (inputsJ I (partOf I (dj S) u.val)) mPts (partS I (dj S) S (partOf I (dj S) u.val))
    (drawJ I (dj S) (partOf I (dj S) u.val)))
  · have e : tableShJ dj y₀ ht S (partOf I (dj S) u.val) = (Classical.choice hS).x.table (typedJ ht _) := by
      unfold tableShJ; rw [dite_eq_left_of_eq_true (eq_true hS)]
    rw [e]
    exact table_bcSite (typedJ ht _) (fun _ => drawJ I (dj S) (partOf I (dj S) u.val))
      ((parse_tagsAt _ _ _ _ _).trans hc) (Layout.scopeOk_spec hU hscope).2.1 ((loadPublic_tagsAt _ _ _ _ _).trans hpub)
      hpt (Classical.choice hS).x (mem_partS h)
  · have e : tableShJ dj y₀ ht S (partOf I (dj S) u.val) = y₀.x.refused (typedJ ht 0) _ := by
      unfold tableShJ; rw [dite_eq_right_of_eq_false (eq_false hS)]
    have hns : ∀ z, ¬ (Partition.tabOf (planShJ (Reg := Reg) dj y₀ ht) (tabShJ dj y₀ ht) S R u).S.Satisfies z := by
      rw [tabOfShJ_eq, e]; exact y₀.x.refused_unsat (typedJ ht 0) _
    exact absurd hM (hns _)

/-- **`rs` at `J` tables**: e2e-layout's `keyedRowsOuts` at the public sites, moved to the `--zk` tables. -/
noncomputable abbrev rsZJ : HmRowsZ (layZJ (Reg := Reg) dj y₀ ht hc hU (pub := pub)) :=
  HmRowsZ.ofShadow (planZJ_S dj y₀ ht) (decAgreeJ dj y₀ ht hc hU)
    (Layout.keyedRowsOuts (sitesShJ_parse dj y₀ ht hc) Hm.saltCol hc (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1
      (sitesShJ_bcSite dj y₀ ht hc hU hpub hpt hscope))

/-- **`hHm` at `J` tables' rows** (`rsZAt_computes`). -/
theorem rsZJ_computes : (rsZJ (Reg := Reg) dj y₀ ht hc hU hpub hpt hscope).Computes :=
  HmRowsZ.ofShadow_computes (planZJ_S dj y₀ ht) (decAgreeJ dj y₀ ht hc hU) _
    (Hm.keyedRowsOuts_computes (sitesShJ_parse dj y₀ ht hc) hc (Layout.scopeOk_spec hU hscope).1
      (Layout.scopeOk_spec hU hscope).2.2 (Layout.scopeOk_spec hU hscope).2.1 (Layout.scopeOk_ports hscope)
      (sitesShJ_bcSite dj y₀ ht hc hU hpub hpt hscope))

end Shadow

/-! ## The compiled session of `J` tables, and the live one simulated in it -/

section Session

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K]

/-- Table `j`'s compiled output in a session on the tables `ts`. -/
noncomputable def outJ (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) :
    Fin ts.length → Cap D → PendZ 𝔽 D Sl × PendZ 𝔽 D Sl → OpensZ 𝔽 D Sl → OutZC 𝔽 D Sl :=
  fun j cap pend os => outZC Hs (ts.get j) cap pend os

/-- Table `j`'s compiled reps in a session on the tables `ts`. -/
noncomputable def repJ (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) :
    Fin ts.length → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (PendZ 𝔽 D Sl) :=
  fun j => repZCAt A Hs (ts.get j)

omit [Fintype K] in
/-- **The compiled session on the tables `ts` is `modelSession`'s** (`sessionBL_eq_modelSession` at `--zk`). -/
theorem sessionZC_eq_modelSession (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts)) :
    modelSession (J := ts.length) (outJ Hs ts) (repJ A Hs ts) = ZkLink.sessionZC A Hs ts := by
  unfold modelSession ZkLink.sessionZC sessAfterZC posOf
  congr 1
  funext caps
  congr 1
  funext pts
  simp only [gamesZC, List.map_ofFn]
  rfl

variable [Inhabited D] [Inhabited Sl]

/-- The simulated compiled session strategy against a live prover of `liveSession` over the tables `ts`. -/
noncomputable def simSessZJ (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => o.accepted = true) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) : Strategy (ZkLink.sessionZC A Hs ts) :=
  sessionZC_eq_modelSession A Hs ts ▸
    (simSession (outJ Hs ts) (fun _ o => o.accepted = true) dfltOutZC pL opn dg dec one fin capOf N (repJ A Hs ts)
      (fun j pts r => allInh_repZCAt A Hs (ts.get j) pts r) hdec).strategy Pr

/-- **The transfer at `J` compiled `--zk` tables** (`live_le_sessionBLJ` at `--zk`). -/
theorem live_le_sessionZJ (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (ts : List (TabZK 𝔽 mPts))
    (pL : LiveOutJ ts.length mPts → Prop) (opn : Array ByteArray → Fin ts.length → OpensZ 𝔽 D Sl)
    (dg : ByteArray → ByteArray) (dec : Fin ts.length → ℕ → ByteArray → List Msg) (one : Fin ts.length → ℕ → Bool)
    (fin : Fin ts.length → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin ts.length → Cap D)
    (N : ℕ)
    (hdec : ∀ j, DecodesTable (outJ Hs ts) (fun _ o => o.accepted = true) pL opn dec one fin capOf (repJ A Hs ts) j)
    (Pr : Strategy (liveSession dg ts.length mPts N)) :
    prob pL (liveSession dg ts.length mPts N) Pr ≤
      prob (acceptedZC ts) (ZkLink.sessionZC A Hs ts)
        (simSessZJ A Hs ts pL opn dg dec one fin capOf N hdec Pr) := by
  unfold simSessZJ
  rw [Exec.prob_cast]
  exact live_le_session _ _ _ pL opn dg dec one fin capOf N _ _ hdec Pr

/-- **The live audit's session at a draw**: `liveSession` over the plan's `J` tables, written out by `wr`. -/
def liveOfZJ (dg : ByteArray → ByteArray) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (N : ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
    (S : Finset (Fin n)) (R : Reg) : Game Exec.Transcript :=
  (liveSession dg (planZ S R).length mPts N).map (wr S R)

variable (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dg : ByteArray → ByteArray)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (one : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → Bool)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (N : ℕ) (L : Law n)

/-- **The decoder is right at the law's draws, table by table** (`DecodesOnJ` at the compiled `--zk` tables): at every
draw `L.draw ω`, registration and table `j`, on every live outcome flock-verify `--zk` accepts, table `j`'s compiled
reps run on the tapes decoded from its streams `(j, 0)`, `(j, 1)` and its proof files, and table `j`'s compiled verdict
on those runs holds. -/
def DecodesOnZKJ : Prop :=
  ∀ ω R (j : Fin (planZ (L.draw ω) R).length),
    DecodesTable (outJ Hs (planZ (L.draw ω) R)) (fun _ o => o.accepted = true)
      (fun o => AcceptsZK I (wr (L.draw ω) R o)) (opn (L.draw ω) R) (dec (L.draw ω) R) (one (L.draw ω) R)
      (fin (L.draw ω) R) (capOf (L.draw ω) R) (repJ A Hs (planZ (L.draw ω) R)) j

variable {L}

/-- **The simulated audit strategy in the compiled batched session of `J` tables** (`simAuditJ` at `--zk`). -/
noncomputable def simZJ (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) :
    Strategy (audit L Reg (batchedSessionZC A Hs planZ)) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessZJ A Hs (planZ (L.draw ω) τ.1) (fun o => AcceptsZK I (wr (L.draw ω) τ.1 o)) (opn (L.draw ω) τ.1)
      dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

/-- **R11 at `--zk` for sessions of `J` tables** (`execRefines_tables` at the compiled `--zk` tables). -/
theorem execRefinesZK_tables (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) :
    ExecRefines (auditLive L Reg (liveOfZJ dg planZ N wr)) τ (LiveAcceptsZK I) (fun _ => False)
      (audit L Reg (batchedSessionZC A Hs planZ)) (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) := by
  have h0 : prob (fun _ => False) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ = 0 :=
    prob_eq_zero_of_out (fun _ _ h => h) τ
  unfold ExecRefines
  rw [h0, add_zero]
  show ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (LiveAcceptsZK I) ((liveOfZJ dg planZ N wr (L.draw ω) τ.1).map fun t => (τ.1, L.draw ω, t)) (τ.2 ω) ≤
    ∑ ω : L.Ω, (Fintype.card L.Ω : ℝ≥0∞)⁻¹ *
      prob (fun o => o.1 = true) ((batchedSessionZC A Hs planZ (L.draw ω) τ.1).map
        fun ok => (ok, L.draw ω)) ((simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ).2 ω)
  refine Finset.sum_le_sum fun ω _ => mul_le_mul_right ?_ _
  unfold simZJ batchedSessionZC
  rw [prob_strategy_map, prob_strategy_map, prob_map]
  unfold liveOfZJ
  rw [prob_map]
  refine (live_le_sessionZJ A Hs (planZ (L.draw ω) τ.1) _ (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1)
    (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N (hdec ω τ.1) _).trans_eq ?_
  congr 1
  funext outs
  classical
  exact propext decide_eq_true_iff.symm

/-- **The `--zk` headline at flock-verify's acceptance, for sessions of `J` tables**: a bound on the compiled batched
audit's acceptance with any event `Q` at `simZJ … τ` bounds flock-verify `--zk`'s acceptance with `Q`. -/
theorem zk_headline_exec_tables (hdec : DecodesOnZKJ A Hs planZ I wr opn dec one fin capOf L)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg planZ N wr))) (Q : Prop) (B : ℝ≥0∞)
    (hB : prob (fun o => o.1 = true ∧ Q) (audit L Reg (batchedSessionZC A Hs planZ))
      (simZJ A Hs planZ I wr opn dg dec one fin capOf N hdec τ) ≤ B) :
    prob (fun o => LiveAcceptsZK I o ∧ Q) (auditLive L Reg (liveOfZJ dg planZ N wr)) τ ≤ B := by
  have h := exec_le_of_refines (execRefinesZK_tables A Hs planZ I wr opn dg dec one fin capOf N hdec τ) Q hB
  rwa [prob_eq_zero_of_out (fun _ _ h => h) τ, add_zero] at h

end Session

/-! ## The message half, table by table -/

section Runs

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl]
  (A : Arith 𝔽 K) (Hs : Hashes 𝔽 D Sl) (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
  (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZ S R).length mPts → Exec.Transcript)
  (opn : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → OpensZ 𝔽 D Sl)
  (dec : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → ℕ → ByteArray → List Msg)
  (fin : (S : Finset (Fin n)) → (R : Reg) → Fin (planZ S R).length → Fin 2 → Array ByteArray → List Msg)
  (capOf : (S : Finset (Fin n)) → (R : Reg) → Array ByteArray → Fin (planZ S R).length → Cap D) (L : Law n)

/-- **`RunsOnZC` at `J` tables**: at the law's draws, on every live outcome flock-verify `--zk` accepts, each table's
reps have a `RepRun` on their streams' rounds along `repShapeZ`, with `fin`'s messages as the tail, and the table's
compiled verdict accepts at the two runs' ends, with its cap and openings. -/
def RunsOnZCJ : Prop :=
  ∀ ω R (o : LiveOutJ (planZ (L.draw ω) R).length mPts), AcceptsZK I (wr (L.draw ω) R o) →
    ∀ j, ∃ pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl,
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 0)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 0) (fin (L.draw ω) R j 0 o.proofs) pend.1 ∧
      PlaysCoin.RepRun (repZCAt A Hs ((planZ (L.draw ω) R).get j) o.pts 1)
        (repShapeZ 𝔽 K D ((planZ (L.draw ω) R).get j)
          (extraClaims ((planZ (L.draw ω) R).get j).S ((planZ (L.draw ω) R).get j).ptLocal mPts o.pts).length)
        (dec (L.draw ω) R j) (o.reps j 1) (fin (L.draw ω) R j 1 o.proofs) pend.2 ∧
      (outZC Hs ((planZ (L.draw ω) R).get j) (capOf (L.draw ω) R o.rootB j) pend
        (opn (L.draw ω) R o.proofs j)).accepted = true

omit [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] in
open Classical in
/-- A compiled table that accepts has no rejected rep (`accZ_lig`). -/
theorem outZC_lig {T : TabZK 𝔽 mPts} {cap : Cap D} {pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl} {os : OpensZ 𝔽 D Sl}
    (h : (outZC Hs T cap pend os).accepted = true) :
    pend.1.lig ≠ PendL.reject ∧ pend.2.lig ≠ PendL.reject := by
  simp only [ZkLink.outZC, Bool.and_eq_true] at h
  exact ⟨ne_reject_of_verdict h.1.1, ne_reject_of_verdict h.1.2⟩

omit [Fintype K] [Inhabited D] [Inhabited Sl] in
/-- **The decoder at the law's draws, table by table, from `RunsOnZCJ`** (`decodesOnZK_of_runsOnZC`), with the coin
step `true`. -/
theorem decodesOnZKJ_of_runsOnZCJ (h : RunsOnZCJ A Hs planZ I wr opn dec fin capOf L) :
    DecodesOnZKJ A Hs planZ I wr opn dec (fun _ _ _ _ => true) fin capOf L := by
  intro ω R j o ho
  obtain ⟨pend, h₀, h₁, hacc⟩ := h ω R o ho j
  obtain ⟨hl₀, hl₁⟩ := outZC_lig Hs hacc
  exact ⟨pend, PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 0 hl₀) h₀,
    PlaysCoin.playsT_of_repRunAt (repZCAt_spineAt A Hs _ o.pts 1 hl₁) h₁, hacc⟩

end Runs

/-! ## The live session's record, the coin server's draws, and the tags -/

/-- **The verify command's tags at any number of tables** (`TagsOk` without `count ≤ 1`): public outputs, retained
rounds, `hm96-sha512/v1` leaves. -/
def TagsOkZJ (I : Inputs) : Prop :=
  I.tags.hiddenOutputs = false ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = "hm96-sha512/v1"

/-- **`wr` is the coin server's record of the live `--zk` session of `J` tables** (`RecordsLiveZK` at `J` tables, as
`RecordsLiveJ` is at the public path): whenever flock-verify `--zk` accepts what `wr` writes for the outcome `o` at draw
`S`, the record carries the draw `dj S`, the proof files are `o`'s, and the record decodes to a session whose link
points are `o`'s and whose table `j`'s `root_B` and rep streams' rounds are `o`'s. -/
def RecordsLiveZKJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (Jf : Finset (Fin n) → Reg → ℕ)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (Jf S R) mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZK I (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOf (zkInputs I) (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      ptsOf mPts sess.points = o.pts ∧
      ∀ j : Fin (Jf S R), sess.rootB[j.val]! = o.rootB[j.val]! ∧ ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[j.val]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = o.reps j r

/-- **The coin server's draw at `S`, at `J` tables** (`DrawOkZK` at table `j`'s inputs and draw): each table's draw
proves the drawn units it is given (`partS`), and setupH sets each table up at it with `mPts` link points. At one table
it is `DrawOkZK`. -/
def DrawOkZKJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupH (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j)) I.partition I.program = .ok st → st.mPts = mPts

/-! ## The decoder and openings at table `j` -/

/-- The number of table `T`'s extra claims. -/
noncomputable def nXZJ (T : TabZK 𝔽 mPts) : ℕ := (Model.extraClaims T.S T.ptLocal mPts (fun _ _ => (0 : 𝔽))).length

/-- **The `--zk` decoder at table `j`** (`decAtZ`): `decZ` at table `j`'s dimensions, claims, schedule and pads. -/
noncomputable def decAtZJ (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planZ S R).length) : ℕ → ByteArray → List Msg :=
  decZ (wRestZ ((planZ S R).get j).S.m ((planZ S R).get j).S.kLog (2 + nXZJ ((planZ S R).get j))
    ((planZ S R).get j).sch ((planZ S R).get j).t ((planZ S R).get j).np)

/-- **The `--zk` openings at table `j`** (`opnAtZ`): `opnZ` at the statement setupH sets up under table `j`'s tags at
its draw, and its `--zk` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnAtZJ (planZ : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts)) (I : Inputs)
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (planZ S R).length) : OpensZ 𝔽 Dig (List UInt8) :=
  match Stmt.setupH (tagsAt (zkTags I) I.count j.val) I.circuitFile I.publicFile (zkCoins I) I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program with
  | .ok st => match Zk.schedule st.m with
    | .ok (sch, _) => opnZ (Setup.ofCircuit st) sch (nXZJ ((planZ S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | .error _ => default
  | .error _ => default

/-! ## Table `j` where `verify --zk` sets it up -/

section Live

variable {I' : Inputs} {S : Finset (Fin n)} {d : Lean.Json}

/-- A `DrawSetup` with `Zk.schedule`'s schedule at its `m` is a `DrawSetupZJ` (`drawSetupZK_of`). -/
theorem drawSetupZJ_of (ht' : I'.tags.typed = false) (x : DrawSetup I' mPts S d) {sch : Flock.Schedule} {tPad : ℕ}
    (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZJ I' mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨hshape, hmsch, hpad⟩ := schedule_zk hz
  obtain ⟨-, hm35, -⟩ := schedule_spec hz
  have hk := (setupH_spec ht' x.h).2.2.2.2.2
  have h0 : sch.k0 ≤ 35 := by
    have := hshape.2.2.1
    simp only [schedOf] at this hmsch
    omega
  have hlen := layLen_le hm35 hk h0
  obtain ⟨k, hk', hk13⟩ := innerN_pow _ hlen
  have hN : ZkExec.innerN (layOf x.st sch) < 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hd : ZkExec.innerN (layOf x.st sch) ∣ 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_dvd_pow 2 (by omega)
  exact ⟨⟨x, sch, tPad, hz, hshape, hmsch, hpad, hN, innerN_ge _,
    fun D => prCoin_innerPos_le _ _ hd D⟩, rfl, rfl, rfl⟩

/-- A table's `--zk` table is the same at every `DrawSetupZJ` of it (`tableZK_eq`). -/
theorem DrawSetupZJ.table_eq (y₁ y₂ : DrawSetupZJ I' mPts S d) : y₁.table = y₂.table := by
  obtain ⟨⟨st₁, h₁, D₁, hD₁, hS₁, hg₁, hb₁, m₁, k₁, a₁, b₁, r₁, e₁, hs₁, hm₁, hp₁⟩, sch₁, t₁, hz₁, _, _, _, _, _, _⟩ := y₁
  obtain ⟨⟨st₂, h₂, D₂, hD₂, hS₂, hg₂, hb₂, m₂, k₂, a₂, b₂, r₂, e₂, hs₂, hm₂, hp₂⟩, sch₂, t₂, hz₂, _, _, _, _, _, _⟩ := y₂
  obtain rfl : st₁ = st₂ := Except.ok.inj (h₁.symm.trans h₂)
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz₁.symm.trans hz₂))
  rfl

theorem DrawSetupZJ.table_seam (y : DrawSetupZJ I' mPts S d) :
    ZkTabSeam y.table (Setup.ofCircuit y.x.st) y.sch y.tPad :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem DrawSetupZJ.table_lay (y : DrawSetupZJ I' mPts S d) : ZkTabLay y.table (Setup.ofCircuit y.x.st) y.sch := by
  constructor <;> intros <;> rfl

end Live

/-- **`verify --zk` sets table `j` up under its tags at its draw**: at the record's draw `d`, `setupTables` at
`zkInputs I` has `nTab I` tables, table `j` setupH's statement under `tagsAt` at `drawJ` (`setupTables_ok` at one table,
`setupTables_parts` at several). -/
theorem setupOf_tableJ {I : Inputs} {record : String} {d : Lean.Json} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hh : I.tags.hiddenOutputs = false) (hd : recordDraw record = some d)
    (hs : setupOf (zkInputs I) record = .ok (spec, sts)) :
    sts.size = nTab I ∧ ∀ (j : ℕ) (hj : j < sts.size),
      Stmt.setupH (tagsAt (zkTags I) I.count j) I.circuitFile I.publicFile (zkCoins I) I.tables
        (some (drawJ I d j)) I.partition I.program = .ok sts[j] := by
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  by_cases hc : I.count ≤ 1
  · rcases setupTables_ok hh' hs with ⟨-, st, hst, -, rfl⟩ | ⟨hc', -⟩
    · refine ⟨by simp [nTab, hc], fun j hj => ?_⟩
      have : j = 0 := by simpa using hj
      subst this
      have e1 : tagsAt (zkTags I) I.count 0 = zkTags I := by simp [tagsAt, hc]
      have e2 : drawJ I d 0 = d := by simp [drawJ, hc]
      rw [e1, e2, ← hd]
      exact hst
    · exact absurd hc' (by change ¬ 1 < I.count; omega)
  · have hc1 : 1 < I.count := by omega
    obtain ⟨-, -, population, units, hD1, -, -, -, hsz, hall⟩ := Tables.setupTables_parts hh' hc1 hs
    obtain ⟨u, hu, -, hpop, hun⟩ := hD1 _ hd
    refine ⟨by simp only [nTab, hc, ite_false]; exact hsz, fun j hj => ?_⟩
    have hpart : drawJ I d j = Tables.partDraw population units I.count j := by
      simp only [drawJ, hc, ite_false]
      unfold partJson
      rw [hu]
      dsimp only
      rw [hpop, hun]
    have e1 : tagsAt (zkTags I) I.count j = { zkTags I with identity := (zkTags I).tableIdentity I.count j } := by
      simp [tagsAt, hc]
    rw [e1, hpart]
    exact hall j hj

open Classical in
theorem outZC_of_accZ {D Sl : Type} [Inhabited D] [Inhabited Sl]
    {Hs : Hashes 𝔽 D Sl} {T : Finset (Fin n) → Reg → TabZK 𝔽 mPts} {S : Finset (Fin n)} {R : Reg}
    {caps : Fin 1 → Cap D} {pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl} {os : OpensZ 𝔽 D Sl}
    (h : accZ T S R (ZkExec.outZ Hs T S R caps pend os) = true) : (outZC Hs (T S R) (caps 0) pend os).accepted = true := by
  unfold accZ ZkExec.outZ at h
  exact (of_decide_eq_true h) ⟨0, by simp⟩

/-- **`RunsZC` at the live `--zk` tables of a session of `J` tables, at the law's draws** (`runsZC_live` at `J`
tables): on untyped inputs under the tags `TagsOkZJ` fixes, with `wr` the coin server's record of the live session and
each of the law's draws setting each table up at `mPts` link points, `RunsOnZCJ` holds at the tables `planZJ` under
`hm96-sha512/v1`, with the decoder `decAtZJ`, the openings `opnAtZJ`, no tail and table `j`'s cap from `root_B[j]`. -/
theorem runsZCJ_live (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} {d₀ : Lean.Json} (y₀ : DrawSetupZJ (inputsJ I 0) mPts S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (L : Law n) (hT : TagsOkZJ I) (hRec : RecordsLiveZKJ I dj (fun S R => (planZJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, DrawOkZKJ I dj mPts (L.draw ω)) :
    RunsOnZCJ (K := 𝕂) execArith hs512 (planZJ dj y₀) I wr (opnAtZJ (planZJ dj y₀) I dj) (decAtZJ (planZJ dj y₀))
      (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) L := by
  intro ω R o hacc j
  obtain ⟨hh, hrr, hl⟩ := hT
  obtain ⟨hd, hprf, hrec⟩ := hRec _ R o hacc
  obtain ⟨spec, sts, hs, -, hv⟩ := hacc
  have hh' : (zkTags I).hiddenOutputs = false := (zkTags_hiddenOutputs I).trans hh
  have hpts2 := setupTables_points hh' hs
  have hsr : spec.retainRounds = true := ((setupTables_retain hh' hs).trans (zkTags_retainRounds I)).trans hrr
  obtain ⟨hsz, hstJ⟩ := setupOf_tableJ hh hd hs
  have hjn : j.val < nTab I := planZJ_length dj y₀ (L.draw ω) R ▸ j.isLt
  have hj : j.val < sts.size := hsz ▸ hjn
  have hst := hstJ j.val hj
  obtain ⟨-, sess, schs, hdecS, -, -, -, -, hall⟩ := zk_verify_ok _ _ _ _ hv
  obtain ⟨hs0, c, hz0, hv0, hv1⟩ := hall j.val (by simpa using hj)
  simp only [Array.getElem_map] at hz0 hv0 hv1
  have hz' : Zk.schedule sts[j.val].m = .ok (schs[j.val].1, schs[j.val].2) := hz0
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (inputsJ I j.val).tags.typed = false := typedJ _ht j.val
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, hx⟩ := drawSetup_of_setupH_g (I := inputsJ I j.val)
    (S := partS I (dj (L.draw ω)) (L.draw ω) ⟨j.val, hjn⟩) (mPts := mPts) ht' hst hf hD
    ((hDraw ω ⟨j.val, hjn⟩).1 D hD) ((hDraw ω ⟨j.val, hjn⟩).2 _ hst)
  obtain ⟨y, hyx, hysch, hytp⟩ := drawSetupZJ_of ht' x (sch := schs[j.val].1) (tPad := schs[j.val].2)
    (by rw [hx]; exact hz')
  have hyst : y.x.st = sts[j.val] := by rw [hyx]; exact hx
  have hTy : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = y.table := by
    have hget : (planZJ (Reg := Reg) dj y₀ (L.draw ω) R).get j = tableZJ dj y₀ (L.draw ω) ⟨j.val, hjn⟩ := by
      exact List.get_ofFn _ j
    rw [hget]
    unfold tableZJ
    rw [dite_eq_left_of_eq_true (eq_true ⟨y⟩)]
    exact DrawSetupZJ.table_eq _ _
  rw [← hysch, ← hytp, ← hyst] at hv0 hv1
  rw [← hyst] at hst
  obtain ⟨hpts, hrj⟩ := hrec spec sts sess hs hdecS
  obtain ⟨hroot, hstreams⟩ := hrj j
  obtain ⟨hp2, hmp⟩ := hpts2
  have hpd := decode_points hdecS
  have hmP : sts[j.val].mPts = mPts := by rw [← hyst]; exact y.x.hmPts
  have hsz' : ∀ p ∈ sess.points.toList, p.size = mPts := fun p hp =>
    (hpd.2 p hp).trans ((hmp j.val hj).symm.trans hmP)
  have hwf := setupH_wf y.x.h
  have hExt : ExtraRealizes (Setup.ofCircuit y.x.st) y.table.S sess.points
      (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts) := by
    have := ofCircuit_extra y.x.st hwf.1 hwf.2 sess.points (hpd.1.trans hp2) (by rw [y.x.hmPts]; exact hsz')
    rw [y.x.hmPts, hpts] at this
    exact this
  have hms : (Setup.ofCircuit y.x.st).merkle = MerkleScheme.hm96Sha512 :=
    setupH_tagsAt_hm96 ((zkTags_merkleLeaf I).trans hl) hst
  have hdl : (Setup.ofCircuit y.x.st).merkle.digestLen = 64 := by rw [hms]; rfl
  have hseam := y.table_seam
  have hlay := y.table_lay
  have hfold : FoldRealizes (Setup.ofCircuit y.x.st).fold y.table.S :=
    ofCircuit_fold y.x.st y.x.wf.1 (regionsOf y.x.st)
  have hwfA := decode_roundsWF hdecS
  have hretA := decode_allRetained hsr hdecS
  obtain ⟨s₀, ex₀, zp₀, tr₀, hf₀, hex₀, hdz₀, -, hrt₀, hrun₀, hk₀, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv0
  obtain ⟨s₁, ex₁, zp₁, tr₁, hf₁, hex₁, hdz₁, -, hrt₁, hrun₁, hk₁, -, -⟩ := zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hv1
  have e₁ : ex₁ = ex₀ := Except.ok.inj (hex₁.symm.trans hex₀)
  subst ex₁
  obtain ⟨hlen₀, hcr₀⟩ := hExt ex₀ hex₀
  have hm₀ : s₀ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₀)
  have hm₁ : s₁ ∈ sess.streams.toList := Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf₁)
  obtain ⟨p₀, hpl₀, hrr₀⟩ := zk_rep_run_plays hs512 y.table o.pts 0 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₀
    s₀.rounds _ tr₀ y.hz hdl hseam hlen₀ hcr₀ hdz₀ hrt₀ (hwfA s₀ hm₀) (hretA s₀ hm₀) hrun₀ hk₀
  obtain ⟨p₁, hpl₁, hrr₁⟩ := zk_rep_run_plays hs512 y.table o.pts 1 (Setup.ofCircuit y.x.st) y.sch y.tPad ex₀ _ zp₁
    s₁.rounds _ tr₁ y.hz hdl hseam hlen₀ hcr₀ hdz₁ hrt₁ (hwfA s₁ hm₁) (hretA s₁ hm₁) hrun₁ hk₁
  obtain ⟨s₀', s₁', zp₀', zp₁', hf₀', hf₁', hd₀', hd₁', -, hall'⟩ :=
    zk_table_verdict (fun (_ : Finset (Fin n)) (_ : Reg) => y.table) (L.draw ω) R o.pts spec (Setup.ofCircuit y.x.st)
      y.sch y.tPad sess j.val _ _ c y.hz hms hseam hlay hfold hExt hwfA hv0 hv1
  have e₂ : s₀' = s₀ := Option.some.inj (hf₀'.symm.trans hf₀)
  subst s₀'
  have e₃ : s₁' = s₁ := Option.some.inj (hf₁'.symm.trans hf₁)
  subst s₁'
  have hshp : shapeZK (Setup.ofCircuit y.x.st) y.sch ex₀ =
      shapeZKn (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length := by
    rw [← hlen₀]
    rfl
  rw [hshp] at hdz₀ hdz₁
  have e₄ : zp₀' = zp₀ := Except.ok.inj (hd₀'.symm.trans hdz₀)
  subst zp₀'
  have e₅ : zp₁' = zp₁ := Except.ok.inj (hd₁'.symm.trans hdz₁)
  subst zp₁'
  rw [hlen₀] at hpl₀ hpl₁
  have hacc' := hall' (p₀, p₁) hpl₀ hpl₁
  have hnX : nXZJ y.table = ex₀.size := by
    unfold nXZJ
    rw [hlen₀, DecodeZL.extraClaims_length, DecodeZL.extraClaims_length]
  have hdecEq : decAtZJ (planZJ (Reg := Reg) dj y₀) (L.draw ω) R j =
      decZ (wRestZ (Setup.ofCircuit y.x.st).m (Setup.ofCircuit y.x.st).kLog (2 + ex₀.size) (schedOf y.sch) y.tPad
        (layZ (Setup.ofCircuit y.x.st) y.sch).len) := by
    unfold decAtZJ
    rw [hTy, hseam.hm, hseam.hk, hseam.hsch, hseam.ht, hseam.hnp, hnX]
  obtain ⟨e₀, he₀, hr₀⟩ := hstreams 0
  obtain ⟨e₁, he₁, hr₁⟩ := hstreams 1
  have e₆ : e₀ = s₀ := Option.some.inj (he₀.symm.trans hf₀)
  subst e₀
  have e₇ : e₁ = s₁ := Option.some.inj (he₁.symm.trans hf₁)
  subst e₁
  have hopn : opnAtZJ (planZJ (Reg := Reg) dj y₀) I dj (L.draw ω) R o.proofs j =
      opnZ (Setup.ofCircuit y.x.st) y.sch (Model.extraClaims y.table.S y.table.ptLocal mPts o.pts).length
        #[(wr (L.draw ω) R o).proofs[2 * j.val]!, (wr (L.draw ω) R o).proofs[2 * j.val + 1]!] := by
    rw [hprf]
    unfold opnAtZJ
    rw [hst]
    dsimp only
    rw [y.hz]
    dsimp only
    rw [hTy, hnX, hlen₀]
  refine ⟨(p₀, p₁), ?_, ?_, ?_⟩
  · rw [hTy, hdecEq, ← hr₀]
    exact hrr₀
  · rw [hTy, hdecEq, ← hr₁]
    exact hrr₁
  · dsimp only
    rw [hTy, hopn, ← hroot]
    exact outZC_of_accZ hacc'

end FlockSoundness.Discharge.Composed
