import Proofs.Flock.Claims.Recursion.VerdictLog
import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.DeployK

/-!
# Two accepted plain decisions on one set of proof files make one set of calls

* `tables_nodup`, `repNames_nodup`: a session the statement sets up names its tables apart (`Tags.tableNames`, at most
  `Tags.MAX_TABLES`), so its rep streams too.
* `streams_pinned`: S11's count and names then leave a decoded session no stream but the one it finds at each rep
  stream's name.
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
  obtain ⟨e1, e2⟩ := List.append_inj' e' (by rw [hlen _ (Nat.mod_lt _ (by omega)), hlen _ (Nat.mod_lt _ (by omega))])
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

end FlockDeploy.KS
