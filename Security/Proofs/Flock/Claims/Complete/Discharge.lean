import Proofs.Flock.Claims.Complete.Skeleton
import Proofs.Flock.Soundness.Refine.Seam

/-!
# The skeleton's goals discharged

The goals of `Goals` proved at every honest run, from the scope alone:

* `Case.table`: the case's statement has one table, of `22 ≤ m ≤ 35`, and the session spec is its own (the scope's
  `count ≤ 1` and table-0 bounds, through `setupTables_okAny`).
* `tables_goal` (`Tables`, §3): one statement per table.
* `schedules_goal` (`Schedules`): `fast100` has a schedule at the table's `m` (`fast100_isSome`).
-/

open Flock FlockDeploy

namespace FlockComplete

/-- **The scope at the case's setup**: one table, of `22 ≤ m ≤ 35`, and the session spec is its own. -/
theorem Case.table {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : Case H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧ 22 ≤ st.m ∧ st.m ≤ 35 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployPlain rel) rel (readOf false) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  have hzk : p.zk = false := readOf_zk (x := c.x) (by
    unfold Readable.parseBytes at hp
    obtain ⟨r, hr, hrp⟩ := Option.bind_eq_some_iff.1 hp
    rw [hr, ← Readable.toRead_of_parse hrp])
  unfold Readable.parseBytes readOf at hp
  rw [hi] at hp
  have ht := Readable.toRead_of_parse (r := c.s.read) hp
  have htab := Readable.tablesOf_of_parses (Readable.parses_of_parse hp)
  obtain ⟨hc, hsz⟩ := (Readable.scope_ok_iff p).1 hsc
  have hR := ((Readable.check_ok_iff _ _ _ _ _ _ _ _).1 hc).1
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hR.2.2.2.2.2])
  obtain ⟨spec', hset'⟩ := Readable.setup_of_tablesOf htab
  have hset := Ins.setup_read (setupOf_some c.setup).2
  rw [ht] at hset'
  have hpe : p.sts = c.sts := by
    have := hset'.symm.trans hset
    simp only [Except.ok.injEq, Prod.mk.injEq] at this
    exact this.2
  rw [hpe] at hsz htab
  have hcount : p.I.count ≤ 1 := by simpa [hzk] using hR.2.2.2.2.1
  obtain ⟨st0, h0, -, -, -, -, -, hm35, -⟩ := Readable.table0 htab hR.2.1 hpi
  have hset2 := hset
  rw [← ht] at hset2
  unfold Readable.Read.setup at hset2
  rw [hpi] at hset2
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset2 with ⟨-, st, -, hs, hsts⟩ | ⟨hc1, -⟩
  · refine ⟨st, hsts, hs, ?_, ?_⟩
    · have := (Readable.sizesOk_iff _ _).1 hsz st.m (by rw [hsts]; simp [Setup.ofCircuit])
      exact (Readable.mMin_ge _).trans this
    · rw [hsts] at h0
      simp only [List.getElem?_toArray, List.getElem?_cons_zero, Option.some.injEq] at h0
      rw [h0]
      exact hm35
  · omega

/-- `fast100` has a schedule at every `m` from 22 to 35. -/
theorem fast100_isSome {m : ℕ} (h1 : 22 ≤ m) (h2 : m ≤ 35) : (fast100 m).isSome = true := by
  have h : (FlockSoundness.Accounting.fast100 m).isSome = true := by interval_cases m <;> rfl
  rw [← FlockSoundness.Refine.fast100_schedOf, Option.isSome_map] at h
  exact h

/-- **One statement per table** (§3): the goal `Tables`. -/
theorem tables_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Tables H rel := by
  intro o
  obtain ⟨st, hsts, hspec, -, -⟩ := o.table
  unfold Run.tabs tabsOf
  rw [hsts, hspec]
  simp [Stmt.spec]

/-- **Each table has its fast100 schedule**: the goal `Schedules`. -/
theorem schedules_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Schedules H rel := by
  intro o st hst
  obtain ⟨st0, hsts, -, h1, h2⟩ := o.table
  unfold Run.tabs tabsOf at hst
  rw [hsts] at hst
  simp only [Array.map_singleton, List.mem_singleton] at hst
  subst hst
  exact fast100_isSome h1 h2

end FlockComplete
