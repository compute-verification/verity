import Proofs.Flock.Claims.Complete.ZkTables.Scope
import Proofs.Flock.Claims.Complete.Zk.Count

/-!
# The honest masked session's sends fit the masked verifier's rounds, at several tables

`Zk/Count.lean`'s per-step bounds at every table, summed over the rep streams:

* `forIn_array_sum`: a loop whose step at `a` costs `k a` costs their sum.
* `rounds_eq`: the session's recorded rounds are each table's two reps' (`repRounds`), summed over the tables; the rep
  streams, two per table, sum to the same (`repStreams_sum`).
* `afterLinkPZk_sendsJ`: after the link message, table `i`'s rep sends at most `repRounds true spec sts[i]` rounds
  (`repPZk_sends`, `zkSchedule_shape`), so the session after the link sends at most `rounds true spec sts`.
* `sends_goal` (`SendCountZk`) at any number of tables.
-/

open Flock FlockDeploy

namespace FlockComplete.ZkTables

open With

/-- **A loop whose step at `a` costs `k a`** costs their sum. -/
theorem forIn_list_sum {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M)
    {α β : Type} (l : List α) (b : β) {f : α → β → M (ForInStep β)} (k : α → ℕ)
    (h : ∀ a ∈ l, ∀ b, C.P (f a b) (k a)) : C.P (forIn l b f) (l.map k).sum := by
  induction l generalizing b with
  | nil => rw [List.forIn_nil]; exact C.pure b
  | cons a l ih =>
    rw [List.forIn_cons, List.map_cons, List.sum_cons]
    refine C.bind (h a List.mem_cons_self b) fun r => ?_
    cases r with
    | done b => exact C.mono (C.pure b) (Nat.zero_le _)
    | yield b => exact ih b fun a ha => h a (List.mem_cons_of_mem _ ha)

/-- **A loop over an array whose step at `a` costs `k a`** costs their sum. -/
theorem forIn_array_sum {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (C : Cost M)
    {α β : Type} (xs : Array α) (b : β) {f : α → β → M (ForInStep β)} (k : α → ℕ)
    (h : ∀ a ∈ xs, ∀ b, C.P (f a b) (k a)) : C.P (forIn xs b f) (xs.toList.map k).sum := by
  rw [← Array.forIn_toList]
  exact forIn_list_sum C xs.toList b k fun a ha => h a (Array.mem_toList_iff.1 ha)

theorem sum_range_two (n : ℕ) (K : ℕ → ℕ) :
    ((List.range (2 * n)).map fun k => K (k / 2)).sum = 2 * ((List.range n).map K).sum := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [show 2 * (n + 1) = 2 * n + 1 + 1 by ring, List.range_succ, List.range_succ, List.range_succ]
    simp only [List.map_append, List.sum_append, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, ih]
    rw [show (2 * n) / 2 = n by omega, show (2 * n + 1) / 2 = n by omega]
    ring

theorem foldl_add_two {α : Type} (g : α → ℕ) (l : List α) (acc : ℕ) :
    l.foldl (fun acc a => acc + 2 * g a) acc = acc + 2 * (l.map g).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons a l ih =>
    rw [List.foldl_cons, ih, List.map_cons, List.sum_cons]
    ring

theorem toList_range {α : Type} [Inhabited α] (xs : Array α) :
    xs.toList = (List.range xs.size).map fun i => xs[i]! := by
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  rw [getElem!_pos xs i (by simpa using h1)]

/-- **The session's recorded rounds** are each table's two reps', summed over the tables. -/
theorem rounds_eq (zk : Bool) (spec : SessionSpec) (sts : Array Stmt) :
    rounds zk spec sts = 2 * ((List.range sts.size).map fun i => repRounds zk spec sts[i]!).sum := by
  unfold rounds
  rw [← Array.foldl_toList, foldl_add_two (repRounds zk spec), toList_range sts, List.map_map, Nat.zero_add]
  rfl

/-- **The rep streams, two per table**, sum a table's cost twice. -/
theorem repStreams_sum (spec : SessionSpec) (K : ℕ → ℕ) :
    (spec.repStreams.toList.map fun a => K a.1).sum = 2 * ((List.range spec.tables.size).map K).sum := by
  rw [FlockDeploy.repStreams_toList, List.map_map]
  exact sum_range_two _ K

/-- **The masked session after the link** sends at most each table's two masked reps' rounds. -/
theorem afterLinkPZk_sendsJ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Stmt) (words : Array (List ℕ)) (schs : Array (Schedule × ℕ)) (trs : Array Trace) (tape : TapeZk)
    (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128) (hspec : spec.tables.size = sts.size)
    (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m)
    (hschs : ∀ j (hj : j < sts.size), Zk.schedule sts[j].m = .ok schs[j]!) :
    sendsC.P (afterLinkPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape schs tbls cms c1)
      (rounds true spec sts) := by
  unfold afterLinkPZk
  extract_lets rootB points streams proofs
  have hpts : points.size = spec.points := by simp [points]
  refine sendsC.mono (sendsC.bind (forIn_array_sum sendsC _ _ (fun a => repRounds true spec sts[a.1]!)
      fun a ha b => ?_) fun s => ?_) (n := _ + 0) ?_
  · obtain ⟨i, r, nm⟩ := a
    have hi : i < sts.size := by
      rw [← Array.mem_toList_iff, FlockDeploy.repStreams_toList, List.mem_map] at ha
      obtain ⟨k, hk, he⟩ := ha
      rw [List.mem_range, hspec] at hk
      have := congrArg Prod.fst he
      simp only at this
      omega
    have ht : (tabsOf sts)[i]! = Setup.ofCircuit sts[i] := by
      rw [getElem!_pos (tabsOf sts) i (by simpa [tabsOf] using hi)]
      simp [tabsOf]
    have hst : sts[i]! = sts[i] := getElem!_pos sts i hi
    have hv := hschs i hi
    obtain ⟨hk0, hL⟩ := zkSchedule_shape hv
    have hmi := hm i hi
    extract_lets streams' proofs'
    dsimp only
    simp only [ht]
    apply sendsC_lift_bind
    intro extras he
    have hx : extras.size = spec.points * sts[i].regions.size := by
      rw [FlockDeploy.extraClaims_size (st := sts[i]) he, hpts, Nat.mul_comm]
    refine sendsC_quiet_bind (innerCommitP_quiet h _ _) fun ic _ => ?_
    apply sendsC.mono
    · cost_walk [repPZk_sends resp h]
    · simp only [hst, repRounds, hv, Setup.ofCircuit, Stmt.circuitFold, hx]
      simp only [↓reduceIte]
      omega
  · cost_walk [recordJson_sends h]
  · rw [repStreams_sum spec (fun i => repRounds true spec sts[i]!), hspec, rounds_eq]
    omega

/-- **The masked session sends at most `1 + rounds`**, whether it returns or stops, at any answers. -/
theorem sessionPZk_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Stmt) (words : Array (List ℕ)) (trs : Array Trace) (ω : TapeZk)
    (hspec : spec.tables.size = sts.size) (hm : ∀ j (hj : j < sts.size), 6 ≤ sts[j].m) :
    sendsC.P (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs ω)
      (1 + rounds true spec sts) := by
  rw [sessionPZk_eq]
  refine sendsC_quiet_bind (schedsPZk_quiet _) fun schs hs => ?_
  unfold schedsPZk at hs
  obtain ⟨hs1, hs2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hs
  have hschs : ∀ j (hj : j < sts.size), Zk.schedule sts[j].m = .ok schs[j]! := by
    intro j hj
    have hv := hs2 j (by simpa [tabsOf] using hj) (by rw [hs1]; simpa [tabsOf] using hj)
    simp only [tabsOf, Array.getElem_map, Setup.ofCircuit, monadLift_self] at hv
    rw [getElem!_pos schs j (by rw [hs1]; simpa [tabsOf] using hj)]
    exact hv
  refine sendsC_quiet_bind (commitsPZk_quiet _ _ _ _ _) fun cms _ => ?_
  exact sendsC.bind (sendsC_link _ _ _) fun c1 =>
    afterLinkPZk_sendsJ _ _ spec sts _ schs _ _ _ cms c1 hspec hm hschs

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked sends fit the masked verifier's rounds**, at any number of tables: the goal `SendCountZk`. -/
theorem sends_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SendCountZk H rel := by
  intro o
  obtain ⟨-, -, hspec, hall⟩ := o.tables
  exact sendsC_ran (sessionPZk_sends o.resp (Claims.hOf H) o.spec o.sts (MaskRank.masksOf o.sts) (traces o.sts o.w)
    o.ω hspec fun j hj => by have := (hall j hj).1; omega) o.ran

end FlockComplete.ZkTables
