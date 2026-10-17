import Proofs.Flock.Claims.Complete.Shape
import Proofs.Flock.Claims.Complete.Zk.Skeleton

/-!
# The masked skeleton's goals discharged

The goals of `GoalsZk` proved at every honest masked run, from the scope and `readyZk`'s one table:

* `CaseZk.table`: the case's statement has one table, of `25 ≤ m ≤ 35`, and the session spec is its own (the `--zk`
  scope's `mMin` and table-0 bounds, `readyZk`'s one table, through `setupTables_okAny`).
* `tablesZk_goal` (`TablesZk`, §3): one statement per table.
* `schedulesZk_goal` (`SchedulesZk`): `Zk.schedule` has a schedule at the table's `m` (`zkSchedule_ok`).
* `proofCountZk_goal` (`ProofCountZk`): the masked session returns one proof file per rep stream (`sessionPZk_proofs`).
-/

open Flock FlockDeploy

namespace FlockComplete

/-- **The masked scope at the case's setup**: one table, of `25 ≤ m ≤ 35`, and the session spec is its own. -/
theorem CaseZk.table {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation} (c : CaseZk H rel) :
    ∃ st, c.sts = #[st] ∧ c.spec = st.spec ∧ 25 ≤ st.m ∧ st.m ≤ 35 := by
  obtain ⟨p, hp, hsc⟩ := (Readable.scope_iff (deployZk rel) rel (readOf true) rfl
    (fun _ _ h => readOf_zk h) c.x).1 c.scope
  have hi := (setupOf_some c.setup).1
  have hzk : p.zk = true := readOf_zk (x := c.x) (by
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
  obtain ⟨st0, h0, -, -, -, -, -, hm35, -⟩ := Readable.table0 htab hR.2.1 hpi
  have hset2 := hset
  rw [← ht] at hset2
  unfold Readable.Read.setup at hset2
  rw [hpi] at hset2
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hset2 with ⟨-, st, -, hs, hsts⟩ | ⟨hc1, hn, -⟩
  · refine ⟨st, hsts, hs, ?_, ?_⟩
    · have := (Readable.sizesOk_iff _ _).1 hsz st.m (by rw [hsts]; simp [Setup.ofCircuit])
      rw [hzk] at this
      simpa [ProvedScope.mMin] using this
    · rw [hsts] at h0
      simp only [List.getElem?_toArray, List.getElem?_cons_zero, Option.some.injEq] at h0
      rw [h0]
      exact hm35
  · have := c.ready_parts.1
    omega

/-- `Zk.schedule` has a schedule at every `m` from 25 to 35. -/
theorem zkSchedule_ok {m : ℕ} (h1 : 25 ≤ m) (h2 : m ≤ 35) : ∃ v, Zk.schedule m = .ok v := by
  obtain ⟨q, hq⟩ : ∃ q, Zk.queries0 m = .ok q := by
    unfold Zk.queries0
    interval_cases m <;> exact ⟨_, rfl⟩
  obtain ⟨sch, hs⟩ := Option.isSome_iff_exists.1 (fast100_isSome (by omega) h2)
  unfold Zk.schedule
  simp only [hq, hs]
  exact ⟨_, rfl⟩

/-- **One statement per table** (§3): the goal `TablesZk`. -/
theorem tablesZk_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : TablesZk H rel := by
  intro o
  obtain ⟨st, hsts, hspec, -, -⟩ := o.table
  unfold RunZk.tabs tabsOf
  rw [hsts, hspec]
  simp [Stmt.spec]

/-- **Each table has its `--zk` schedule**: the goal `SchedulesZk`. -/
theorem schedulesZk_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SchedulesZk H rel := by
  intro o st hst
  obtain ⟨st0, hsts, -, h1, h2⟩ := o.table
  unfold RunZk.tabs tabsOf at hst
  rw [hsts] at hst
  simp only [Array.map_singleton, List.mem_singleton] at hst
  subst hst
  exact zkSchedule_ok h1 h2

/-- **The masked session returns one proof file per rep stream.** -/
theorem sessionPZk_proofs (h : ByteArray → ByteArray) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (trs : Array Trace) (tape : TapeZk) :
    Ret (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec tabs trs tape) fun r => r.2.size = spec.repStreams.size := by
  unfold sessionPZk
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.true _) fun _ _ => ?_
  refine Ret.bind (Ret.forIn_array _ _ _ (fun k b => b.snd.size = k) rfl ?_) ?_
  · rintro k ⟨i, r, name⟩ ⟨ss, ps⟩ hk
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    exact Ret.pure ⟨_, rfl, by simp [hk]⟩
  · rintro ⟨ss, ps⟩ hk
    refine Ret.bind (Ret.true _) fun _ _ => ?_
    exact Ret.pure hk

/-- **One proof file per rep stream**: the goal `ProofCountZk`. -/
theorem proofCountZk_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    ProofCountZk H rel := by
  intro o
  have := sessionPZk_proofs (Claims.hOf H) (fun j => (Claims.coinSeq o.r j).bytes) o.spec (tabsOf o.sts)
    (tracesOf o.sts o.w) o.ω #[] _ _ o.ran
  simp [this]

end FlockComplete
