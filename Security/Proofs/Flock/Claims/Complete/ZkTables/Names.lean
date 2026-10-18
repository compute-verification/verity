import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.ZkTables.Count
import Proofs.Flock.Claims.Complete.Json.Record

/-!
# The tables' names and the rep streams' are distinct

The record keys each table's `root_B` by the table's name and finds each rep's stream and proof digest by the stream's
name, so at several tables those names must differ.

* `tableNames_nodup`: `Tags.tableNames` names each table `{table}-{j}`, `j` padded to two digits, so each name's
  suffix reads back as its index (`pad_toNat?`, through `String.toNat?`) and no two names agree (`tableName_inj`).
* `repNames_nodup`: rep `r < 2` of table `t` is `{t}/rep{r}`, so distinct tables give distinct rep streams
  (`repName_inj`, the last digit then the prefix).
* `rootBList_distinct`: the record's `root_b` object has one key per table.
* `Case.names`: at a good case of several tables the spec names its tables `Tags.tableNames`, so they are distinct
  (`Case.tables_nodup`, at any number of tables).
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

open FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec JsonRT

/-- **A table name's suffix reads back as its index**: `j` padded to two digits. -/
theorem pad_toNat? (j : ℕ) : (toString (if j < 10 then "0" else "") ++ toString j).toNat? = some j := by
  show ((if j < 10 then "0" else "") ++ Nat.repr j).toNat? = some j
  split
  · have hd : ∀ c ∈ ("0" ++ Nat.repr j).toList, c.isDigit := by
      intro c hc
      rw [String.toList_append] at hc
      rcases List.mem_append.1 hc with h | h
      · simp at h; subst h; decide
      · exact Nat.isDigit_of_mem_toDigits (b := 10) (by decide) (by decide) (by simpa [Nat.toList_repr] using h)
    rw [String.toNat?_eq_some_ofDigitChars (String.isNat_of_isDigit (by simp) hd)]
    simp only [String.toList_append, Nat.toList_repr, String.reduceToList, List.cons_append, List.nil_append,
      List.filter_cons, bne_iff_ne, ne_eq, show ¬ ('0' = '_') by decide, not_false_eq_true, ite_true,
      Option.some.injEq]
    rw [Nat.ofDigitChars_cons, List.filter_bne_eq_self_of_not_mem (by simp)]
    simp
  · simp

theorem tableName_inj (t : String) {j k : ℕ}
    (h : s!"{t}-{if j < 10 then "0" else ""}{j}" = s!"{t}-{if k < 10 then "0" else ""}{k}") : j = k := by
  simp only [String.append_assoc, String.append_right_inj] at h
  have := congrArg String.toNat? h
  rw [pad_toNat?, pad_toNat?] at this
  exact Option.some.inj this

/-- **`Tags.tableNames` names no two tables alike.** -/
theorem tableNames_nodup (t : Tags) (count : ℕ) : (t.tableNames count).toList.Nodup := by
  unfold Tags.tableNames
  split
  · simp
  · rw [Array.toList_map, Array.toList_range]
    exact List.Nodup.map_on (fun j _ k _ h => tableName_inj t.table h) List.nodup_range

theorem repName_inj {a b : String} {r r' : ℕ} (hr : r < 2) (hr' : r' < 2) (h : s!"{a}/rep{r}" = s!"{b}/rep{r'}") :
    a = b ∧ r = r' := by
  interval_cases r <;> interval_cases r'
  all_goals
    have := congrArg (fun s => s.toList.reverse) h
    simp [String.toList_append, show Nat.toDigits 10 1 = ['1'] from rfl] at this
    try exact ⟨String.toList_inj.1 this, rfl⟩

/-- **Distinct tables give distinct rep streams.** -/
theorem repNames_nodup {spec : SessionSpec} (h : spec.tables.toList.Nodup) :
    (spec.repStreams.toList.map (·.2.2)).Nodup := by
  rw [FlockDeploy.repStreams_toList, List.map_map]
  refine List.Nodup.map_on (fun j hj k hk e => ?_) List.nodup_range
  rw [List.mem_range] at hj hk
  obtain ⟨h1, h2⟩ := repName_inj (Nat.mod_lt j (by decide)) (Nat.mod_lt k (by decide)) e
  have hj' : j / 2 < spec.tables.size := by omega
  have hk' : k / 2 < spec.tables.size := by omega
  rw [getElem!_pos spec.tables _ hj', getElem!_pos spec.tables _ hk', ← Array.getElem_toList,
    ← Array.getElem_toList] at h1
  have := (List.Nodup.getElem_inj_iff h).1 h1
  omega

/-- **The record's `root_b` object has one key per table.** -/
theorem rootBList_distinct {spec : SessionSpec} (h : spec.tables.toList.Nodup) (f : ByteArray → ByteArray)
    {rootB : Array ByteArray} (hs : rootB.size = spec.tables.size) : JsonRT.Distinct (rootBList spec f rootB) := by
  unfold rootBList JsonRT.Distinct
  rw [List.pairwise_map]
  have hfst : ((spec.tables.zip rootB).toList.map Prod.fst) = spec.tables.toList := by
    rw [Array.toList_zip, List.map_fst_zip (by simp [hs])]
  have := hfst ▸ h
  refine (List.pairwise_map.1 this).imp fun hne heq => hne ?_
  exact compare_eq_iff_eq.1 heq

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **At a good case of several tables the spec names its tables `Tags.tableNames`.** -/
theorem Case.names (c : Case H rel) (hc : 1 < c.sts.size) : ∃ t : Tags, c.spec.tables = t.tableNames c.sts.size := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  obtain ⟨hck, -⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hck).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [← ht] at hset
  unfold Readable.Read.setup at hset
  rw [hpi] at hset
  rcases HiddenExec.setupTables_okAny hset with ⟨-, st, -, -, hst⟩ | ⟨hc', hsz, -, -, -⟩
  · rw [hst] at hc
    simp at hc
  · exact ⟨_, by rw [(setupTables_names hc' hset).1, hsz]⟩

/-- **At a good case no two tables are named alike.** -/
theorem Case.tables_nodup (c : Case H rel) : c.spec.tables.toList.Nodup := by
  rcases (show c.sts.size = 1 ∨ 1 < c.sts.size by have := c.tables.1; omega) with h1 | hc
  · obtain ⟨-, -, hsp, -⟩ := c.tables
    rw [h1] at hsp
    obtain ⟨t, ht⟩ : ∃ t, c.spec.tables = #[t] := ⟨c.spec.tables[0], by
      apply Array.ext (by simp [hsp]) fun i h₁ h₂ => by
        simp at h₂
        subst h₂
        rfl⟩
    simp [ht]
  · obtain ⟨t, ht⟩ := c.names hc
    rw [ht]
    exact tableNames_nodup t _

end FlockComplete.ZkTables
