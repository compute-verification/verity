module

public import Proofs.Properties.Verity.Sampled.Moves
public import Proofs.Properties.Verity.Sampled.CoinMap
public import Proofs.Flock.Recursive.Witness
public import Proofs.Flock.Soundness.Refine.Walk
public import Proofs.Flock.Soundness.Refine.Lookups
public import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
public import Proofs.Properties.Certifier.Check.Bytes
public import Proofs.Properties.Spot.Tables

/-! v2's verdict is verity-verify-proof `--zk`'s acceptance on the session's files, which gives the decoders'
premise of the logged coupling (`Decodes`). -/

@[expose] public section

set_option Elab.async false

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Discharge
open FlockSoundness.Discharge.ZkHidden (AcceptsZKP LiveAcceptsZKP)
open FlockSoundness.Discharge.Recursion.Witness (toExec setupOf_eq)
open FlockSoundness.Discharge.Tables (LiveOutJ)
open FlockSoundness.Refine (ebind_ok ptsOf OkLe TablesExtend setupTables_back extend_filterMap mufu_nodup)
open Claims.Bridge.CustodyZkP (repRounds)

theorem setup_if (tags : Flock.Tags) (cf pf : ByteArray) (coins : Flock.Coins) (count : ℕ)
    (tables : List (String × ByteArray)) (draw partition : Option Json) (program : Option ByteArray) :
    (if tags.hiddenOutputs then
        Flock.Stmt.setupTablesOf (Flock.ProvedScope.circuitOf tags cf tables) tags cf pf coins count tables draw
          partition program
      else Flock.Stmt.setupTables tags cf pf coins count tables draw partition program :
        Flock.V (Flock.SessionSpec × Array Flock.Stmt)) =
      Flock.Stmt.setupTables tags cf pf coins count tables draw partition program := by
  by_cases hh : tags.hiddenOutputs = true
  · rw [ite_eq_left hh, Flock.Stmt.setupTables_eq, Flock.ProvedScope.circuitOf, ite_eq_left hh]
  · rw [ite_eq_right hh]

theorem buildSessionOf_ok {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins}
    {tables : List (String × ByteArray)} {count : ℕ} {draw partition : Option Json} {program : Option ByteArray}
    {registered : Option (Array Flock.Registered.Port)} {r : Flock.SessionSpec × Array Flock.Stmt}
    (h : _root_.Spot.OneStageZk.buildSessionOf (Flock.ProvedScope.circuitOf tags cf tables) tags cf pf coins
      tables count draw partition program registered = .ok r) :
    Flock.Stmt.setupTables tags cf pf coins count tables draw partition program = .ok r := by
  unfold _root_.Spot.OneStageZk.buildSessionOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨r', hr, h⟩ := ebind_ok h
  rw [setup_if] at hr
  repeat' walk_step
  all_goals (cases h; exact hr)

theorem sessionAt_ok {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {draw : Option Json}
    {r : Flock.SessionSpec × Array Flock.Stmt} (h : _root_.Spot.OneStageZk.sessionAt W rd draw = .ok r) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables draw W.partition W.program = .ok r ∧
      r.2.forM Flock.Zk.shapeOk = .ok () := by
  unfold _root_.Spot.OneStageZk.sessionAt at h
  try dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨r', hr, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hs, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  exact ⟨buildSessionOf_ok hr, hs⟩

theorem setupAt_ok {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {draw : Option Json}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (h : _root_.Spot.OneStageZk.setupAt W rd draw = .ok (spec, tabs, x)) :
    ∃ sts, _root_.Spot.OneStageZk.sessionAt W rd draw = .ok (spec, sts) ∧
      tabs = sts.map Flock.Setup.ofCircuit := by
  unfold _root_.Spot.OneStageZk.setupAt at h
  obtain ⟨⟨spec', sts⟩, h1, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  cases h
  refine ⟨sts, ?_, rfl⟩
  cases hs : _root_.Spot.OneStageZk.sessionAt W rd draw with
  | error e => rw [hs] at h1; cases h1
  | ok b => rw [hs] at h1; cases h1; rfl

theorem setup_eq (W : Flock.Witness.Inputs) (record : String) :
    Flock.Stmt.setupTables (Flock.Witness.zkTags W) W.circuitFile W.publicFile (Flock.Witness.zkCoins W) W.count
        W.tables (Flock.Witness.recordDraw record) W.partition W.program =
      Exec.setupOfP (ZkExec.zkInputs (toExec W)) none record :=
  setupOf_eq W record

theorem setupAt_setupOfP {W : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads} {record : String}
    {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {x : Json × Json}
    (h : _root_.Spot.OneStageZk.setupAt W rd (Flock.Witness.recordDraw record) = .ok (spec, tabs, x)) :
    ∃ sts, Exec.setupOfP (ZkExec.zkInputs (toExec W)) none record = .ok (spec, sts) ∧
      sts.forM Flock.Zk.shapeOk = .ok () ∧ tabs = sts.map Flock.Setup.ofCircuit := by
  obtain ⟨sts, hs, rfl⟩ := setupAt_ok h
  obtain ⟨hset, hshape⟩ := sessionAt_ok hs
  rw [setup_eq] at hset
  exact ⟨sts, hset, hshape, rfl⟩

theorem zkVerify_decode {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} (h : Flock.Zk.verify spec tabs record proofs = .ok ()) :
    ∃ sess, Flock.Record.decode spec record = .ok sess := by
  obtain ⟨-, sess, -, hs, -⟩ := ZkExec.zk_verify_ok spec tabs record proofs h
  exact ⟨sess, hs⟩

theorem v2Verified_accepts {p : _root_.Spot.OneStageZk.Params} {e : _root_.Spot.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {record : String} {spec : Flock.SessionSpec} {tabs : Array Flock.Setup}
    {rest : Json × Json} {proofs : Array ByteArray} (h : V2Verified p e bs record spec tabs rest proofs) :
    ∃ tags0 circuitB pubB sts, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      Exec.setupOfP (ZkExec.zkInputs (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)))
        none record = .ok (spec, sts) ∧
      AcceptsZKP (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)) none
        { record := record, proofs := proofs } := by
  obtain ⟨_, tags0, circuitB, pubB, _, -, ht, hc, hp, -, -, hst, -, hv⟩ := h
  obtain ⟨sts, hs, hshape, rfl⟩ := setupAt_setupOfP hst
  exact ⟨tags0, circuitB, pubB, sts, ht, hc, hp, hs, spec, sts, hs, hshape, hv⟩

theorem zkInputs_toExec {W : Flock.Witness.Inputs} {I : Exec.Inputs} (htags : W.tags = I.tags)
    (hcf : W.circuitFile = I.circuitFile) (hpf : W.publicFile = I.publicFile) (hcount : W.count = I.count)
    (hpart : W.partition = I.partition) (hprog : W.program = I.program)
    (hwork : I.work = none) (hstrat : I.stratified = none) (hclo : I.closure = none) :
    ZkExec.zkInputs (toExec W) = ZkExec.zkInputs { I with tables := W.tables } := by
  obtain ⟨tags, cf, pf, count, tables, partition, program⟩ := W
  obtain ⟨tags', cf', pf', coins', count', tables', partition', program', work', strat', clo'⟩ := I
  dsimp only at htags hcf hpf hcount hpart hprog hwork hstrat hclo
  subst htags hcf hpf hcount hpart hprog hwork hstrat hclo
  rfl

theorem acceptsZKP_congr {I I' : Exec.Inputs} (h : ZkExec.zkInputs I = ZkExec.zkInputs I') {own : Option ByteArray}
    {t : Exec.Transcript} : AcceptsZKP I own t ↔ AcceptsZKP I' own t := by
  unfold AcceptsZKP
  rw [h]

theorem setupOfP_back {I : Exec.Inputs} {T' : List (String × ByteArray)} (hext : TablesExtend I.tables T')
    (hcount : I.count ≤ 1) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hp : Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) (own : Option ByteArray)
    (record : String) :
    OkLe (Exec.setupOfP (ZkExec.zkInputs { I with tables := T' }) own record)
      (Exec.setupOfP (ZkExec.zkInputs I) own record) :=
  setupTables_back hext hcount ((ZkExec.zkTags_typed I).trans ht) hp _ _ _ _ _ _ _ _ _

theorem acceptsZKP_back {I : Exec.Inputs} {T' : List (String × ByteArray)} (hext : TablesExtend I.tables T')
    (hcount : I.count ≤ 1) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hp : Flock.HmOut.parse (ZkExec.zkTags I) I.circuitFile I.tables = .ok c) {own : Option ByteArray}
    {t : Exec.Transcript} (h : AcceptsZKP { I with tables := T' } own t) : AcceptsZKP I own t := by
  obtain ⟨spec, sts, hs, hshape, hv⟩ := h
  exact ⟨spec, sts, setupOfP_back hext hcount ht hp own t.record _ hs, hshape, hv⟩

structure FilesOn (A : Live) (s : Verity.Sent) : Prop where
  circuit : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.circuit "circuit" = .ok b → b = A.I.circuitFile
  publicFile : ∀ p, ParamsAt s.params p → ∀ b,
    (_root_.Core.Blobs.ofArray s.blobs).get p.own.pub "public file" = .ok b → b = A.I.publicFile
  tables : TablesExtend A.I.tables (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs))

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

theorem getD_inj {a b : ByteArray} (ha : a.size = 64) (hb : b.size = 64)
    (h : (fun i : Fin 64 => a.data.getD i.val 0) = fun i => b.data.getD i.val 0) : a = b := by
  apply ByteArray.ext
  apply Array.ext (ha.trans hb.symm)
  intro i hi1 hi2
  have hi : i < 64 := by rw [← ha]; exact hi1
  have e := congrFun h ⟨i, hi⟩
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi1, Array.getElem?_eq_getElem hi2,
    Option.getD_some] at e
  exact e

theorem sha512_data (f : ByteArray) :
    Verity.sha512 f.data.toList = fun i => (Flock.Sha512.hash f).data.getD i.val 0 := by
  unfold Verity.sha512
  rw [Array.toArray_toList]

theorem get_first {F B : Array ByteArray} {s what : String} {b : ByteArray} {q : ByteArray → Bool}
    (hq : ∀ dg, Flock.unhex s = .ok dg → dg.size = 64 → ∀ f, (Flock.Sha512.hash f == dg) = q f)
    (hreg : ∃ f ∈ F, q f = true) (hg : (_root_.Core.Blobs.ofArray (F ++ B)).get s what = .ok b) :
    b = (F.find? q).getD .empty := by
  unfold _root_.Core.Blobs.get at hg
  obtain ⟨dg, hu, hg⟩ := ebind_ok hg
  dsimp only at hg
  by_cases c0 : (dg.size != 64 || Flock.hexOf dg != s) = true
  · rw [ite_eq_left c0] at hg
    obtain ⟨_, h0, _⟩ := ebind_ok hg
    cases h0
  rw [ite_eq_right c0] at hg
  have hp := hq dg hu (by
    by_contra hne
    exact c0 (by simp [hne]))
  obtain ⟨f, hf, hfq⟩ := hreg
  obtain ⟨f0, hf0⟩ := Option.isSome_iff_exists.1 (Array.find?_isSome.2 ⟨f, hf, hfq⟩)
  have hfind : (_root_.Core.Blobs.ofArray (F ++ B)).entries.find? (fun x => x.1 == dg) =
      some (Flock.Sha512.hash f0, f0) := by
    show ((F ++ B).map fun b => (Flock.Sha512.hash b, b)).find? (fun x => x.1 == dg) = _
    simp only [Array.find?_map, Array.find?_append, Function.comp_def, hp, hf0, Option.some_or, Option.map_some]
  simp only [hfind] at hg
  have hb : f0 = b := by cases hg; rfl
  rw [← hb, hf0, Option.getD_some]

theorem get_named {F B : Array ByteArray} {s what : String} {h : Option Hash.Digest} {b : ByteArray}
    (hd : ∀ dg, Flock.unhex s = .ok dg → h = some fun i => dg.data.getD i.val 0)
    (hreg : ∃ f ∈ F, h = some (Verity.sha512 f.data.toList))
    (hg : (_root_.Core.Blobs.ofArray (F ++ B)).get s what = .ok b) : b = namedFile h F := by
  unfold namedFile
  refine get_first (fun dg hu h64 f => ?_) ?_ hg
  · try dsimp only
    rw [hd dg hu, sha512_data, Bool.eq_iff_iff, beq_bytes, decide_eq_true_iff, Option.some.injEq]
    constructor
    · rintro rfl
      rfl
    · intro e
      exact (getD_inj h64 (Properties.Certifier.Bytes.hash_size f) e).symm
  · obtain ⟨f, hf, hfh⟩ := hreg
    exact ⟨f, hf, decide_eq_true hfh⟩

theorem tables_registered {V : Vocab d α} {k : Verity.Registration.Composite} {o : Verity.Exchange}
    {p₀ : _root_.Spot.OneStageZk.Params} (hθ : ParamsAt o.params p₀) (hone : OneStatement o.params)
    (hv : V.Verdict k .sampled o) :
    Properties.Spot.HoldsRegisteredTables p₀.own (_root_.Core.Blobs.ofArray o.sent.blobs) := by
  obtain ⟨_, _, _, p, _, _, _, _, hcase, hp, -, -, -, -, pub, prof, hck⟩ := case_drew hv.2.1 hθ (verdict_case hv)
  cases params_of_case hv.2.1 hcase hp hθ
  have h := Properties.Spot.SecurityProofs.tablesRegistered.2.1 _ _ _ _ _ _ _ hck
  unfold Properties.Spot.AcceptHoldsTables at h
  rw [hone _ hθ] at h
  exact h

theorem pinnedTables_extend {T : PinnedTables} {s : Verity.Sent} {circuitB : ByteArray} {names : List String}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB = .ok names)
    (hn : names = (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)).map (·.1))
    (ht : TablesSent T s) :
    TablesExtend (pinnedTables T circuitB)
      (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)) := by
  classical
  have hp : pinnedTables T circuitB = Flock.MUFU_TABLES.filterMap fun t =>
      if t ∈ Flock.MUFU_TABLES ∧ names.contains t.name = true then (T.file t.sha512).map (fun b => (t.name, b))
      else none := by
    simp only [pinnedTables, hl]
    refine List.filterMap_congr fun t ht => ?_
    by_cases hc : names.contains t.name = true
    · rw [if_pos hc, if_pos ⟨ht, hc⟩]
    · rw [if_neg hc, if_neg fun h => hc h.2]
  rw [hp, tablesOf_ofArray]
  refine extend_filterMap mufu_nodup (fun t x h => ?_) (fun t x h => ?_) (fun t x h => ?_)
  · split at h
    · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rfl
    · cases h
  · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
    rfl
  · split at h
    · rename_i hc
      obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.1 h
      have hm : t.name ∈ names := by simpa using hc.2
      rw [hn, tablesOf_ofArray] at hm
      obtain ⟨y, hy, hyn⟩ := List.mem_map.1 hm
      obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
      obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
      have htt : t' = t := List.inj_on_of_nodup_map mufu_nodup ht' hc.1 hyn
      subst htt
      simp only [ha, ht _ ht' a b ha hb, Option.map_some]
    · cases h

theorem pinnedTables_among {T : PinnedTables} {s : Verity.Sent} {circuitB : ByteArray} {names : List String}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB = .ok names)
    (hn : (_root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray s.blobs)).map (·.1) = names)
    (ht : TablesSent T s) :
    TablesExtend (pinnedTables T circuitB)
      (_root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray s.blobs)) := by
  classical
  have hp : pinnedTables T circuitB = Flock.MUFU_TABLES.filterMap fun t =>
      if t ∈ Flock.MUFU_TABLES ∧ names.contains t.name = true then (T.file t.sha512).map (fun b => (t.name, b))
      else none := by
    simp only [pinnedTables, hl]
    refine List.filterMap_congr fun t ht => ?_
    by_cases hc : names.contains t.name = true
    · rw [if_pos hc, if_pos ⟨ht, hc⟩]
    · rw [if_neg hc, if_neg fun h => hc h.2]
  have hq : _root_.Spot.OneStageZk.tablesAmong names (_root_.Core.Blobs.ofArray s.blobs) =
      Flock.MUFU_TABLES.filterMap fun t =>
        if names.contains t.name = true then (tableFile t.sha512 s.blobs).map (fun b => (t.name, b)) else none := by
    unfold _root_.Spot.OneStageZk.tablesAmong
    rw [tablesOf_ofArray, List.filter_filterMap]
    refine List.filterMap_congr fun t _ => ?_
    cases tableFile t.sha512 s.blobs <;> simp [Option.filter]
  have hn' : ∀ n ∈ names, n ∈ (Flock.MUFU_TABLES.filterMap fun t =>
      if names.contains t.name = true then (tableFile t.sha512 s.blobs).map (fun b => (t.name, b)) else none).map
        (·.1) := by
    intro n hmem
    rw [← hq, hn]
    exact hmem
  rw [hp, hq]
  refine extend_filterMap mufu_nodup (fun t x h => ?_) (fun t x h => ?_) (fun t x h => ?_)
  · split at h
    · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rfl
    · cases h
  · split at h
    · obtain ⟨_, -, rfl⟩ := Option.map_eq_some_iff.1 h
      rfl
    · cases h
  · split at h
    · rename_i hc
      obtain ⟨b, hb, rfl⟩ := Option.map_eq_some_iff.1 h
      have hm : t.name ∈ names := by simpa using hc.2
      obtain ⟨y, hy, hyn⟩ := List.mem_map.1 (hn' _ hm)
      obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
      split at hft'
      · obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
        have htt : t' = t := List.inj_on_of_nodup_map mufu_nodup ht' hc.1 hyn
        subst htt
        rw [if_pos hc.2]
        simp only [ha, ht _ ht' a b ha hb, Option.map_some]
      · cases hft'
    · cases h

theorem filesOn_of {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F B : Array ByteArray} {A : Live}
    {s : Verity.Sent} (hA : SetsUp V θ F A) (hD : DigestsAt V T θ) (hreg : V.Registered θ F) (hθ : s.params = θ)
    (hs : s.blobs = F ++ B) {p₀ : _root_.Spot.OneStageZk.Params} (hp₀ : ParamsAt s.params p₀)
    (hheld : Properties.Spot.HoldsRegisteredTables p₀.own (_root_.Core.Blobs.ofArray s.blobs))
    (ht : TablesSent T s) : FilesOn A s := by
  have hc : ∀ p, ParamsAt s.params p → ∀ b,
      (_root_.Core.Blobs.ofArray s.blobs).get p.own.circuit "circuit" = .ok b → b = A.I.circuitFile := by
    intro p hp b hg
    rw [hs] at hg
    rw [hA.circuit]
    exact get_named (hD.circuit p (hθ ▸ hp)) hreg.1 hg
  refine ⟨hc, fun p hp b hg => ?_, ?_⟩
  · rw [hs] at hg
    rw [hA.publicFile]
    exact get_named (hD.publicFile p (hθ ▸ hp)) hreg.2 hg
  · obtain ⟨circuitB, hg, hl⟩ := hheld
    have h : A.I.tables = pinnedTables T A.I.circuitFile := by
      rw [hA.circuit]
      exact (congrArg (fun r : Verity.ExecutionPhaseFiles => r.first.tables) hA.read).symm.trans (hD.tables F)
    rw [h, ← hc p₀ hp₀ circuitB hg]
    exact pinnedTables_extend hl rfl ht

theorem accepts_sent {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) :
    ∃ record proofs spec sts, sentRecord o.sent = some record ∧ sentProofs o.sent = some proofs ∧
      Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts) ∧
      AcceptsZKP A.I A.own { record := record, proofs := proofs } := by
  subst hθ
  obtain ⟨p', hp', hpart, hprog, -⟩ := hA.params
  obtain ⟨a, b, e, p, c, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp' (verdict_case hv)
  have hpp : p' = p := paramsAt_case hv.2.1 hcase hacc hp'
  rw [hpp] at hp' hpart hprog
  obtain ⟨record, spec, tabs, rest, proofs, -, hrec, hpr, hver⟩ :=
    sent_verified hv.2.1 hcase hacc (single_of_case hv.2.1 hcase hacc hA.single)
  obtain ⟨tags0, circuitB, pubB, sts, ht0, hc, hpb, hs, hat⟩ := v2Verified_accepts hver
  have hz := zkInputs_toExec (W := _root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB
      (_root_.Core.Blobs.ofArray o.sent.blobs)) (Option.some.inj (ht0.symm.trans hA.tags))
    (hF.circuit p hp' circuitB hc) (hF.publicFile p hp' pubB hpb) hA.count.symm hpart.symm hprog.symm hA.work
    hA.stratified hA.closure
  rw [hz] at hs
  rw [hA.own]
  have hp := Flock.Desc.parse_flat A.hD
  exact ⟨record, proofs, spec, sts, hrec, hpr, setupOfP_back hF.tables hA.count.le A.ht hp none record _ hs,
    acceptsZKP_back hF.tables hA.count.le A.ht hp ((acceptsZKP_congr hz).1 hat)⟩

def MovesOf (A : Live) (s : Verity.Sent) (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP) : Prop :=
  ∀ record proofs spec sts sess, sentRecord s = some record → sentProofs s = some proofs →
    Exec.setupOfP (ZkExec.zkInputs A.I) A.own record = .ok (spec, sts) →
    Flock.Record.decode spec record = .ok sess →
    Exec.recordDraw record = some (A.dj S) ∧ out.rootB = sess.rootB ∧ out.pts = ptsOf A.mP sess.points ∧
      (∀ j r, out.reps j r = repRounds spec sess j.val r.val) ∧ out.proofs = proofs

theorem wr_accepts_of_verdict {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live} {θ : ByteArray}
    {F : Array ByteArray} {o : Verity.Exchange} (hA : SetsUp V θ F A) (hθ : o.params = θ)
    (hv : V.Verdict k .sampled o) (hF : FilesOn A o.sent) (hW : WritesView A) {S : Finset (Fin A.n)} {R : A.Reg}
    {out : LiveOutJ (A.J S R) A.mP} (hM : MovesOf A o.sent S R out) : AcceptsZKP A.I A.own (A.wr S R out) := by
  obtain ⟨record, proofs, spec, sts, hrec, hpr, hs, hat⟩ := accepts_sent hA hθ hv hF
  have hd : ∃ sess, Flock.Record.decode spec record = .ok sess := by
    obtain ⟨spec', sts', hs', -, hver⟩ := hat
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hs.symm.trans hs'))
    exact zkVerify_decode hver
  obtain ⟨sess, hd⟩ := hd
  obtain ⟨hdr, hB, hP, hR, hPf⟩ := hM record proofs spec sts sess hrec hpr hs hd
  exact wr_accepts hW hat hdr hs hd hB hP hR hPf

theorem run_reaches {β : Type} : ∀ (g : Verity.Game β) (s : Verity.Game.Strategy g)
    (x : Verity.Game.CoinTape g s), Verity.Game.PlayReaches g s (Verity.Game.CoinTape.run g s x)
  | .ret _, _, _ => rfl
  | .send _ k, s, x => run_reaches (k s.1) s.2 x
  | @Verity.Game.coin _ _ _ _ k, s, x => ⟨x.1, run_reaches (k x.1) (s x.1) (x.2 x.1)⟩

theorem play_ends (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (V.game θ F c)) (x : Verity.Game.CoinTape (V.game θ F c) σ) :
    (Verity.Game.CoinTape.run (V.game θ F c) σ x).params = θ ∧
      (Verity.Game.CoinTape.run (V.game θ F c) σ x).files = F := by
  obtain ⟨log, s, h⟩ := Verity.Game.PlayReaches.of_bind (f := finish (V.signer θ) θ F c σ.1)
    (releases (frames (V.families θ)) (V.families θ) []) σ.2 (run_reaches (V.game θ F c) σ x)
  obtain ⟨ρ, ho⟩ : ∃ ρ : Fin (V.signer θ).draws → Bool, Verity.Game.CoinTape.run (V.game θ F c) σ x =
      ⟨θ, F, c, σ.1, log, s.1, List.ofFn ρ, (V.signer θ).signatures σ.1 log s.1 (List.ofFn ρ), (s.2 ρ).1.1,
        (s.2 ρ).1.2⟩ := h
  rw [ho]
  exact ⟨rfl, rfl⟩

def MovesReach (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (k : Verity.Registration.Composite)
    (P : Verity.Exchange → Prop) (A : Live) (rd : Reading A.rounds) (σ : Verity.Game.Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    (B : RoundHold) : Prop :=
  ∀ o, ReachesOff V θ F c A rd σ D M BL B o →
    V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ o.2) ∧ NoMiss V θ F c rd σ o.2 ∧
      P (Verity.Game.CoinTape.run (V.game θ F c) σ o.2) →
    ∃ (S : Finset (Fin A.n)) (R : A.Reg) (out : LiveOutJ (A.J S R) A.mP), o.1 = (R, S, A.wr S R out) ∧
      MovesOf A (Verity.Game.CoinTape.run (V.game θ F c) σ o.2).sent S R out

theorem decodes_of {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {P : Verity.Exchange → Prop} {A : Live} {rd : Reading A.rounds}
    {σ : Verity.Game.Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold} (hA : SetsUp V θ F A)
    (hF : ∀ x, V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ x) →
      P (Verity.Game.CoinTape.run (V.game θ F c) σ x) → FilesOn A (Verity.Game.CoinTape.run (V.game θ F c) σ x).sent)
    (hW : WritesView A) (hM : MovesReach V θ F c k P A rd σ D M BL B) : Decodes V θ F c k P A rd σ D M BL B := by
  intro o ho hacc
  obtain ⟨S, R, out, h1, hmo⟩ := hM o ho hacc
  rw [h1]
  exact wr_accepts_of_verdict hA (play_ends V θ F c σ o.2).1 hacc.1 (hF o.2 hacc.1 hacc.2.2) hW hmo

theorem filesOn_play {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {A : Live} {σ : Verity.Game.Strategy (V.game θ F c)}
    (hA : SetsUp V θ F A) (hD : DigestsAt V T θ) {x : Verity.Game.CoinTape (V.game θ F c) σ}
    (hv : V.Verdict k .sampled (Verity.Game.CoinTape.run (V.game θ F c) σ x))
    (ht : TablesSent T (Verity.Game.CoinTape.run (V.game θ F c) σ x).sent) :
    FilesOn A (Verity.Game.CoinTape.run (V.game θ F c) σ x).sent := by
  obtain ⟨hp, hf⟩ := play_ends V θ F c σ x
  have hreg : V.Registered θ F := by
    have h := hv.1
    rwa [hp, hf] at h
  obtain ⟨p₀, hp₀, -⟩ := hA.params
  have hp₀' : ParamsAt (Verity.Game.CoinTape.run (V.game θ F c) σ x).params p₀ := by rw [hp]; exact hp₀
  exact filesOn_of hA hD hreg hp (B := (Verity.Game.CoinTape.run (V.game θ F c) σ x).blobs)
    (by show (Verity.Game.CoinTape.run (V.game θ F c) σ x).files ++ _ = _; rw [hf]) hp₀'
    (tables_registered hp₀' (by rw [hp]; exact hA.single) hv) ht

theorem decodes_of_digests {V : Vocab d α} {T : PinnedTables} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    {k : Verity.Registration.Composite} {P : Verity.Exchange → Prop} {A : Live} {rd : Reading A.rounds}
    {σ : Verity.Game.Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold} (hA : SetsUp V θ F A)
    (hD : DigestsAt V T θ) (hP : ∀ o, P o → TablesSent T o.sent) (hW : WritesView A)
    (hM : MovesReach V θ F c k P A rd σ D M BL B) : Decodes V θ F c k P A rd σ D M BL B :=
  decodes_of hA (fun _ hv hp => filesOn_play hA hD hv (hP _ hp)) hW hM

end Verity.Sampled
