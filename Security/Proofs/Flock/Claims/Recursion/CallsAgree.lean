import Proofs.Flock.Claims.Recursion.VerdictLog
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.DeployK

/-!
# Two accepted plain decisions on one set of proof files make one set of calls

* `tables_nodup`, `repNames_nodup`: a session the statement sets up names its tables apart (`Tags.tableNames`, at most
  `Tags.MAX_TABLES`), so its rep streams too.
* `streams_pinned`: S11's count and names then leave a decoded session no stream but the one it finds at each rep
  stream's name.
* `decode_publics`: the public words `Record.decode` reads (S7) are `liveOf`'s.
-/

namespace FlockDeploy.KS

open Flock FlockSoundness.Refine

/-! ## The tables' names are distinct -/

/-- A table index's digits read back. -/
def digitsVal (l : List Char) : ℕ := l.foldl (fun n c => 10 * n + (c.toNat - 48)) 0

theorem digitsVal_pad : ∀ j < 64, digitsVal ((if j < 10 then ['0'] else []) ++ Nat.toDigits 10 j) = j := by
  decide +kernel

theorem pad_toList (j : ℕ) : ((if j < 10 then "0" else "") ++ toString j).toList =
    (if j < 10 then ['0'] else []) ++ Nat.toDigits 10 j := by
  rw [String.toList_append, Nat.toString_eq_repr, Nat.toList_repr]
  split <;> simp

theorem tableNames_nodup (t : Tags) {count : ℕ} (h : count ≤ Tags.MAX_TABLES) :
    (t.tableNames count).toList.Nodup := by
  unfold Tags.tableNames
  split
  · simp
  · rw [Array.toList_map, Array.toList_range]
    refine List.Nodup.map_on (fun i hi j hj e => ?_) List.nodup_range
    rw [List.mem_range] at hi hj
    unfold Tags.MAX_TABLES at h
    have e' := congrArg String.toList e
    simp only [String.toList_append, List.append_assoc, List.append_cancel_left_eq] at e'
    have e2 : ((if i < 10 then "0" else "") ++ toString i).toList =
        ((if j < 10 then "0" else "") ++ toString j).toList := by
      rw [String.toList_append, String.toList_append]
      exact e'
    rw [pad_toList, pad_toList] at e2
    rw [← digitsVal_pad i (by omega), ← digitsVal_pad j (by omega), e2]

/-- **A session the statement sets up names its tables apart.** -/
theorem tables_nodup {s : Ins} {spec : SessionSpec} {sts : Array Stmt} (hs : s.setup = .ok (spec, sts))
    (hpi : s.read.publicInputs = none) : spec.tables.toList.Nodup := by
  have h := Ins.setup_read hs
  unfold Readable.Read.setup at h
  rw [hpi] at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny h with ⟨-, st, -, rfl, -⟩ | ⟨hc, -, -, -, -⟩
  · simp [Stmt.spec]
  · obtain ⟨hn, hmax⟩ := FlockComplete.ZkTables.setupTables_names hc h
    rw [hn]
    exact tableNames_nodup _ hmax

theorem toDigits_two_inj : ∀ r < 2, ∀ r' < 2, Nat.toDigits 10 r = Nat.toDigits 10 r' → r = r' := by decide

/-- **Tables named apart name their rep streams apart.** -/
theorem repNames_nodup {spec : SessionSpec} (h : spec.tables.toList.Nodup) :
    (spec.repStreams.toList.map (·.2.2)).Nodup := by
  rw [repStreams_toList, List.map_map]
  refine List.Nodup.map_on (fun k hk k' hk' e => ?_) List.nodup_range
  rw [List.mem_range] at hk hk'
  have e' := congrArg String.toList e
  simp only [Function.comp, String.toList_append, Nat.toString_eq_repr, Nat.toList_repr] at e'
  have hlen : ∀ r < 2, (Nat.toDigits 10 r).length = 1 := by decide
  obtain ⟨e1, e2⟩ :=
    List.append_inj' e' (by rw [hlen _ (Nat.mod_lt _ (by omega)), hlen _ (Nat.mod_lt _ (by omega))])
  have et : spec.tables[k / 2]! = spec.tables[k' / 2]! := String.toList_injective (List.append_cancel_right e1)
  have hi : k / 2 < spec.tables.size := by have := repStreams_size spec; omega
  have hi' : k' / 2 < spec.tables.size := by have := repStreams_size spec; omega
  rw [getElem!_pos spec.tables _ hi, getElem!_pos spec.tables _ hi', ← Array.getElem_toList,
    ← Array.getElem_toList, h.getElem_inj] at et
  have := toDigits_two_inj _ (Nat.mod_lt _ (by omega)) _ (Nat.mod_lt _ (by omega)) e2
  omega

/-! ## S11 leaves no stream but the rep streams -/

theorem find?_self_of_nodup {α β : Type} [BEq β] [LawfulBEq β] (key : α → β) :
    ∀ {l : List α}, (l.map key).Nodup → ∀ {s : α}, s ∈ l → l.find? (fun x => key x == key s) = some s
  | [], _, _, h => by cases h
  | a :: l, hn, s, h => by
    rw [List.map_cons, List.nodup_cons] at hn
    rcases List.mem_cons.1 h with rfl | h
    · simp
    · have hne : ¬(key a == key s) = true := by
        rw [beq_iff_eq]
        intro e
        exact hn.1 (e ▸ List.mem_map_of_mem h)
      rw [List.find?_cons_of_neg (p := fun x => key x == key s) hne]
      exact find?_self_of_nodup key hn.2 h

/-- **S11 pins a decoded session's streams**: with the rep streams named apart, as many streams as rep streams, and
one found at each rep stream's name, every stream is the one found at some rep stream's name. -/
theorem streams_pinned {spec : SessionSpec} {sess : Session} (hn : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hsz : sess.streams.size = spec.repStreams.size)
    (hf : ∀ x ∈ spec.repStreams.toList, ∃ s, sess.streams.find? (·.name == x.2.2) = some s) :
    ∀ s ∈ sess.streams.toList, ∃ q, ∃ hq : q < spec.repStreams.size,
      sess.streams.find? (·.name == spec.repStreams[q].2.2) = some s := by
  classical
  have hsub : (spec.repStreams.toList.map (·.2.2)).toFinset ⊆ (sess.streams.toList.map StreamRec.name).toFinset := by
    intro w hw
    rw [List.mem_toFinset, List.mem_map] at hw ⊢
    obtain ⟨x, hx, rfl⟩ := hw
    obtain ⟨s, hs⟩ := hf x hx
    rw [← Array.find?_toList] at hs
    exact ⟨s, List.mem_of_find?_eq_some hs, by simpa using List.find?_some hs⟩
  have hcW : (spec.repStreams.toList.map (·.2.2)).toFinset.card = (sess.streams.toList.map StreamRec.name).length := by
    rw [List.toFinset_card_of_nodup hn]
    simp [hsz]
  have hnN : (sess.streams.toList.map StreamRec.name).Nodup := by
    have hc : (sess.streams.toList.map StreamRec.name).toFinset.card =
        (sess.streams.toList.map StreamRec.name).length :=
      le_antisymm (List.toFinset_card_le _) (hcW ▸ Finset.card_le_card hsub)
    exact Multiset.coe_nodup.1 (Multiset.toFinset_card_eq_card_iff_nodup.1 hc)
  have heq := Finset.eq_of_subset_of_card_le hsub (by rw [hcW]; exact List.toFinset_card_le _)
  intro s hs
  have hsW : s.name ∈ spec.repStreams.toList.map (·.2.2) := by
    rw [← List.mem_toFinset, heq, List.mem_toFinset]
    exact List.mem_map_of_mem hs
  obtain ⟨x, hx, hxs⟩ := List.mem_map.1 hsW
  obtain ⟨q, hq, rfl⟩ := List.mem_iff_getElem.1 hx
  refine ⟨q, by simpa using hq, ?_⟩
  rw [← Array.find?_toList]
  simp only [Array.getElem_toList] at hxs
  rw [hxs]
  exact find?_self_of_nodup StreamRec.name hnN hs

/-! ## The decoded public words are `liveOf`'s -/

section Publics

open Lean (Json)
open Claims.Bridge.Custody

/-- Table `t`'s public words as the record's link has them: empty where absent. -/
def PubField (pubs : Json) (t : String) (p : ByteArray) : Prop :=
  (Record.isNull pubs t = true → p = ByteArray.empty) ∧
    (Record.isNull pubs t = false → ∃ s, Record.str pubs t = .ok s ∧ unhex s = .ok p)

theorem pubField_unique {pubs : Json} {t : String} {a b : ByteArray} (ha : PubField pubs t a)
    (hb : PubField pubs t b) : a = b := by
  cases hn : Record.isNull pubs t
  · obtain ⟨s, h1, h2⟩ := ha.2 hn
    obtain ⟨s', h1', h2'⟩ := hb.2 hn
    rw [h1] at h1'
    cases h1'
    rw [h2] at h2'
    cases h2'
    rfl
  · rw [ha.1 hn, hb.1 hn]

theorem pubField_of {pubs : Json} {t : String} {p : ByteArray}
    (h : (if Record.isNull pubs t = true then pure ByteArray.empty else do unhex (← Record.str pubs t) :
      Flock.V ByteArray) = .ok p) : PubField pubs t p := by
  refine ⟨fun hn => ?_, fun hn => ?_⟩
  · rw [ite_eq_left hn] at h
    exact (Except.ok.inj h).symm
  · rw [ite_eq_right (by simp [hn])] at h
    obtain ⟨s, hs, h⟩ := ebind_ok h
    exact ⟨s, hs, h⟩

/-- **What `Record.decode` reads of the public words.** -/
theorem decode_pubs {spec : SessionSpec} {text : String} {sess : Session} (h : Record.decode spec text = .ok sess) :
    ∃ j l pubs, Json.parse text = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList sess.publics.toList := by
  revert sess
  show Ret (Record.decode spec text) _
  rw [Record.decode.eq_def]
  cases hp : Json.parse text with
  | error e => intro sess h; cases h
  | ok j =>
    simp only []
    rw [pure_bind]
    ret_walk [Record.fld, Array.mapM]
    all_goals
      refine ⟨j, _, _, rfl, by assumption, by assumption, ?_⟩
      refine List.Forall₂.imp (fun t p h => ?_) (mapM_forall₂ (by assumption))
      beta_reduce at h
      split at h
      · rename_i hn
        rw [pure_bind] at h
        refine ⟨fun _ => ?_, fun hn' => by simp [hn] at hn'⟩
        split at h
        · obtain ⟨_, _, h⟩ := ebind_ok h
          obtain ⟨_, _, h⟩ := ebind_ok h
          exact (Except.ok.inj h).symm
        · exact (Except.ok.inj h).symm
      · rename_i hn
        obtain ⟨s, hs, h⟩ := ebind_ok h
        obtain ⟨p', hp', h⟩ := ebind_ok h
        refine ⟨fun hn' => absurd hn' hn, fun _ => ⟨s, hs, ?_⟩⟩
        split at h
        · obtain ⟨_, _, h⟩ := ebind_ok h
          obtain ⟨_, _, h⟩ := ebind_ok h
          rw [← Except.ok.inj h]
          exact hp'
        · rw [← Except.ok.inj h]
          exact hp'

/-- **What `liveOf` reads of the public words.** -/
theorem liveOf_pubs {spec : SessionSpec} {record : String} {lv : Live} (h : liveOf spec record = .ok lv) :
    ∃ j l pubs, Json.parse record = .ok j ∧ Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (PubField pubs) spec.tables.toList lv.publics.toList := by
  revert lv
  show Ret (liveOf spec record) _
  rw [liveOf.eq_def]
  cases hp : Json.parse record with
  | error e => intro lv h; cases h
  | ok j =>
    dsimp only
    rw [pure_bind]
    ret_walk [Record.fld, Array.mapM]
    refine ⟨j, _, _, rfl, by assumption, by assumption, ?_⟩
    exact List.Forall₂.imp (fun t p h => pubField_of h) (mapM_forall₂ (by assumption))

/-- **The decoded session's public words are `liveOf`'s.** -/
theorem decode_publics {spec : SessionSpec} {text : String} {sess : Session} {lv : Live}
    (hd : Record.decode spec text = .ok sess) (hl : liveOf spec text = .ok lv) : sess.publics = lv.publics := by
  obtain ⟨j, l, pubs, hp, hl1, hp1, hf⟩ := decode_pubs hd
  obtain ⟨j', l', pubs', hp', hl1', hp1', hf'⟩ := liveOf_pubs hl
  rw [hp] at hp'
  cases hp'
  rw [hl1] at hl1'
  cases hl1'
  rw [hp1] at hp1'
  cases hp1'
  exact Array.ext' (forall₂_unique (R := PubField pubs) (fun _ _ _ => pubField_unique) hf hf')

end Publics

end FlockDeploy.KS
