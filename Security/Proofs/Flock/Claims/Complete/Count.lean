import Proofs.Flock.Claims.Complete.Link
import Proofs.Flock.Claims.Complete.CostWalk
import Proofs.Flock.Claims.DeployK

/-!
# The honest session's sends fit the verifier's rounds

* **The sends** (`sendsC`): at most `n` bodies pushed by a run of the pure session that returns, a cost bound that
  `cost_walk` reads off each prover step: a squeeze sends at most one round, a commitment (quiet) and the record
  (`recordJson_sends`) none.
* **Per step**: zerocheck `3 + (m − 6)`, lincheck `3 + (kLog − 6)`, the ring switch one per claim and one more, Ligerito
  `k0 + 9` and then `11` per recursive level but the last, which sends `7` (`ligeritoP_sends`); a rep their sum
  (`repP_sends`).
* **The session** (`afterLinkP_sends`): one table's two reps, each with the statement's extra claims (one per region
  and link point), after the link message; with fast100's at least two levels that is `repRounds`, and
  `sendCount_goal` is the goal `SendCount`.
-/

open Flock FlockDeploy

namespace FlockComplete

open With

/-! ## The sends -/

/-- **At most `n` sends**: a run of the pure session that returns pushes at most `n` bodies. -/
def sendsC : Cost PS where
  P y n := ∀ l a l', y.go l = (.ok a, l') → l'.size ≤ l.size + n
  mono h hnm l a l' e := (h l a l' e).trans (by omega)
  pure a l b l' e := by
    simp only [StV.go, Id.run] at e
    cases e
    simp
  bind {α β x f n m} hx hf l b l' e := by
    rw [go_bind] at e
    rcases hy : x.go l with ⟨r, l''⟩
    rw [hy] at e
    cases r with
    | error _ => cases e
    | ok a =>
      have h1 := hx l a l'' hy
      have h2 := hf a l'' b l' e
      omega
  throw _ _ _ _ _ e := by cases e
  lift v l a l' e := by
    cases v with
    | error _ => cases e
    | ok b =>
      have : l' = l := (Prod.mk.inj e).2.symm
      subst this
      simp

theorem sendsC_round (resp : ℕ → Claims.Msg) (c : ByteArray) : sendsC.P ((asksS resp).round c) 1 := by
  intro l a l' e
  cases e
  simp

theorem sendsC_quiet {α : Type} {v : V α} {y : PS α} (h : quietS.R v y) : sendsC.P y 0 := by
  intro l a l' e
  rw [h l] at e
  cases e
  simp

theorem squeezeP_sends (resp : ℕ → Claims.Msg) (kind : UInt8) (n : ℕ) :
    (sendsC.st Transcript).P (squeezeP (asksS resp) kind n) 1 := by
  intro t
  unfold squeezeP
  extract_lets c
  refine sendsC.mono (sendsC.ite (fun _ => sendsC.pure _) fun _ =>
    sendsC.bind (sendsC_round resp c) fun _ => sendsC.pure _) (by omega)

theorem sampleP_sends (resp : ℕ → Claims.Msg) : (sendsC.st Transcript).P (sampleP (asksS resp)) 1 := by
  unfold sampleP
  exact (sendsC.st Transcript).bind (squeezeP_sends resp 1 1) fun _ => (sendsC.st Transcript).pure _

theorem sampleVecP_sends (resp : ℕ → Claims.Msg) (n : ℕ) :
    (sendsC.st Transcript).P (sampleVecP (asksS resp) n) 1 :=
  squeezeP_sends resp 2 n

theorem sampleKP_sends (resp : ℕ → Claims.Msg) : (sendsC.st Transcript).P (sampleKP (asksS resp)) 1 := by
  unfold sampleKP
  exact (sendsC.st Transcript).bind (squeezeP_sends resp 2 2) fun _ => (sendsC.st Transcript).pure _

theorem zerocheckP_sends (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap : ByteArray) (tr : Trace) :
    (sendsC.st Transcript).P (zerocheckP (asksS resp) m digest cap tr) (3 + (m - 6)) := by
  apply (sendsC.st Transcript).mono
  · unfold zerocheckP sampleVecP
    cost_walk [sampleP_sends resp, squeezeP_sends resp, Cost.forIn_range]
  · simp only [Std.Legacy.Range.size]
    omega

theorem lincheckP_sends (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (tr : Trace) (zc : ZcOut) :
    (sendsC.st Transcript).P (lincheckP (asksS resp) c m tr zc) (3 + (c.kLog - 6)) := by
  apply (sendsC.st Transcript).mono
  · unfold lincheckP
    cost_walk [sampleP_sends resp, Cost.forIn_range]
  · simp only [Std.Legacy.Range.size]
    omega

theorem ringSwitchP_sends (resp : ℕ → Claims.Msg) (tr : Trace) (claims : Array Claim) :
    (sendsC.st Transcript).P (ringSwitchP (asksS resp) tr claims) (claims.size + 1) := by
  apply (sendsC.st Transcript).mono
  · unfold ringSwitchP sampleVecP
    cost_walk [squeezeP_sends resp, Cost.forIn_range]
  · simp only [Std.Legacy.Range.size]
    omega

/-! ## Ligerito -/

theorem commitP_sends (h : ByteArray → ByteArray) (ms : MerkleScheme) (lv : Level) (g : Array F128)
    (salt : ℕ → ByteArray) : sendsC.P (commitP (Hashes.ofSha512 h) ms lv g salt : PS _) 0 :=
  sendsC_quiet (commitP_quiet h ms lv g salt)

theorem sendMsg_sends (m : F256 × F256) : ((sendsC.st Transcript).st LigState).P (sendMsg m : LP PS Unit) 0 := by
  apply ((sendsC.st Transcript).st LigState).mono
  · unfold sendMsg liftT
    cost_walk
  · omega

theorem powP_sends (nonce : ℕ → ℕ → UInt64) (kind : ℕ) :
    ((sendsC.st Transcript).st LigState).P (powP nonce kind : LP PS Unit) 0 := by
  apply ((sendsC.st Transcript).st LigState).mono
  · unfold powP liftT
    cost_walk
  · omega

theorem challengeP_sends (resp : ℕ → Claims.Msg) (split : Bool) :
    ((sendsC.st Transcript).st LigState).P (challengeP (asksS resp) split) 1 := by
  apply ((sendsC.st Transcript).st LigState).mono
  · unfold challengeP liftT
    cost_walk [sampleKP_sends resp, sendMsg_sends]
  · omega

theorem oodIntroP_sends (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (g : Array F128) (dim : ℕ) :
    ((sendsC.st Transcript).st LigState).P (oodIntroP (asksS resp) nonce g dim) 2 := by
  apply ((sendsC.st Transcript).st LigState).mono
  · unfold oodIntroP liftT
    cost_walk [sampleVecP_sends resp, sampleP_sends resp, sendMsg_sends, powP_sends]
  · omega

theorem queryP_sends (resp : ℕ → Claims.Msg) (nonce : ℕ → ℕ → UInt64) (ms : MerkleScheme) (lv : Level) (cm : Commit) :
    ((sendsC.st Transcript).st LigState).P (queryP (asksS resp) nonce ms lv cm) 2 := by
  apply ((sendsC.st Transcript).st LigState).mono
  · unfold queryP liftT
    cost_walk [sampleVecP_sends resp, powP_sends]
  · omega

/-- **Ligerito's sends**: `k0 + 9` before the recursive levels, then `11` at each level but the last, which sends `7`. -/
theorem ligeritoP_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (sendsC.st Transcript).P (ligeritoP (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b salt nonce)
      (sch.k0 + 9 + ((sch.levels.size - 1 - 1) * 11 + 7)) := by
  apply (sendsC.st Transcript).mono
  · unfold ligeritoP liftT
    extract_lets n R lv
    cost_walk [sampleVecP_sends resp, sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends,
      challengeP_sends resp, oodIntroP_sends resp, queryP_sends resp, commitP_sends h, Cost.forIn_upto _ sch.k0]
    refine Cost.forIn_last _ R _ (k := 11) (k' := 7) (fun i hi b => ?_) (fun hn b => ?_)
    · apply ((sendsC.st Transcript).st LigState).mono
      · cost_walk [sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends, challengeP_sends resp,
          oodIntroP_sends resp, queryP_sends resp, commitP_sends h, Cost.forIn_upto _ 4]
      · omega
    · simp only [show R - 1 + 1 = R by omega, beq_self_eq_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte]
      apply ((sendsC.st Transcript).st LigState).mono
      · cost_walk [sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends, challengeP_sends resp,
          queryP_sends resp, Cost.forIn_upto _ 4]
      · omega
    all_goals cost_walk
  · omega

/-! ## A rep, the record and the session -/

theorem ringSwitchP_abc_sends (resp : ℕ → Claims.Msg) (tr : Trace) (m kLog : ℕ) (zc : ZcOut) (lc : LcOut)
    (extras : Array Claim) :
    (sendsC.st Transcript).P (ringSwitchP (asksS resp) tr (abcClaims m kLog zc lc ++ extras)) (2 + extras.size + 1) := by
  have := ringSwitchP_sends resp tr (abcClaims m kLog zc lc ++ extras)
  rwa [Array.size_append, show (abcClaims m kLog zc lc).size = 2 from rfl] at this

/-- **A rep's sends**: zerocheck's, lincheck's, the ring switch's on the two claims and the extras, and Ligerito's. -/
theorem repP_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (sendsC.st Transcript).P (repP (asksS resp) (Hashes.ofSha512 h) st sch tr t cm0 extras salt nonce)
      ((3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (2 + extras.size + 1) +
        (sch.k0 + 9 + ((sch.levels.size - 1 - 1) * 11 + 7))) := by
  apply (sendsC.st Transcript).mono
  · unfold repP
    cost_walk [zerocheckP_sends resp, lincheckP_sends resp, ringSwitchP_abc_sends resp, ligeritoP_sends resp h]
  · omega

theorem roundJ_quiet (h : ByteArray → ByteArray) (spec : SessionSpec) (k : ℕ) (r : Round) :
    quietS.R (roundJ (Hashes.ofSha512 h) spec k r : V _) (roundJ (Hashes.ofSha512 h) spec k r : PS _) := by
  have hh := quietS_hashes h
  unfold roundJ SessionSpec.dgWith
  sim_walk

/-- **The record sends nothing.** -/
theorem recordJson_sends (h : ByteArray → ByteArray) (spec : SessionSpec) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    sendsC.P (recordJson (Hashes.ofSha512 h) spec rootB points streams proofs : PS _) 0 := by
  apply sendsC_quiet (v := (recordJson (Hashes.ofSha512 h) spec rootB points streams proofs : V _))
  have hh := quietS_hashes h
  unfold recordJson SessionSpec.dgWith
  sim_walk [roundJ_quiet h, Sim.mapM_list]

/-- A lifted verdict's value: what follows it is bounded at the value it returns. -/
theorem sendsC_lift_bind {α β : Type} (v : V α) {f : α → PS β} {n : ℕ} (h : ∀ a, v = .ok a → sendsC.P (f a) n) :
    sendsC.P (monadLift v >>= f) n := by
  intro l b l' e
  rw [go_bind] at e
  cases v with
  | error _ => cases e
  | ok a => exact h a rfl l b l' e

/-- A quiet run's value: what follows it is bounded at the value it returns. -/
theorem sendsC_quiet_bind {α β : Type} {v : V α} {y : PS α} (hq : quietS.R v y) {f : α → PS β} {n : ℕ}
    (h : ∀ a, v = .ok a → sendsC.P (f a) n) : sendsC.P (y >>= f) n := by
  intro l b l' e
  rw [go_bind, hq l] at e
  cases v with
  | error _ => cases e
  | ok a => exact h a rfl l b l' e

theorem sendsC_link (resp : ℕ → Claims.Msg) (a b : Array ByteArray) : sendsC.P ((asksS resp).link a b) 1 := by
  intro l c l' e
  cases e
  simp

/-- **The sends after the link message**, one table's two reps. -/
theorem afterLinkP_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Stmt)
    (sch : Schedule) (trs : Array Trace) (tape : Tape) (tbls : Array (Array F128)) (cms : Array Commit)
    (c1 : ℕ → F128) (hspec : spec.tables.size = 1) :
    sendsC.P (afterLinkP (asksS resp) (Hashes.ofSha512 h) spec #[Setup.ofCircuit st] trs tape #[sch] tbls cms c1)
      (2 * ((3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) + (2 + spec.points * st.regions.size + 1) +
        (sch.k0 + 9 + ((sch.levels.size - 1 - 1) * 11 + 7)))) := by
  unfold afterLinkP
  extract_lets rootB points streams proofs
  have hpts : points.size = spec.points := by simp [points]
  refine sendsC.mono (sendsC.bind (sendsC.forIn_array _ _ (k := (3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) +
      (2 + spec.points * st.regions.size + 1) + (sch.k0 + 9 + ((sch.levels.size - 1 - 1) * 11 + 7)))
      fun a ha b => ?_) fun s => ?_) (n := spec.repStreams.size * _ + 0) ?_
  · obtain ⟨i, r, nm⟩ := a
    have hi : i = 0 := by
      rw [← Array.mem_toList_iff, FlockDeploy.repStreams_toList, List.mem_map] at ha
      obtain ⟨k, hk, he⟩ := ha
      rw [List.mem_range, hspec] at hk
      have := congrArg Prod.fst he
      simp only at this
      omega
    subst hi
    extract_lets streams' proofs'
    dsimp only
    apply sendsC_lift_bind
    intro extras he
    have hx : extras.size = spec.points * st.regions.size := by
      rw [FlockDeploy.extraClaims_size (st := st) he, hpts, Nat.mul_comm]
    apply sendsC.mono
    · cost_walk [repP_sends resp h]
    · simp only [List.getElem!_toArray, List.getElem!_cons_zero, Setup.ofCircuit, Stmt.circuitFold, hx]
      omega
  · cost_walk [recordJson_sends h]
  · rw [FlockDeploy.repStreams_size, hspec]
    omega

/-- fast100's schedules have at least two levels. -/
theorem fast100_levels {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : 2 ≤ sch.levels.size := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, (fast100 m.val).all (fun s => decide (2 ≤ s.levels.size)) = true := by
      decide +kernel
    have := key ⟨m, hm⟩
    simp only [h, Option.all_some, decide_eq_true_eq] at this
    exact this
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

/-! ## The goal -/

/-- **The sends fit the verifier's rounds**: the goal `SendCount`. -/
theorem sendCount_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SendCount H rel := by
  intro o
  obtain ⟨st, hsts, hspec, -⟩ := o.leaf
  have htabs : tabsOf o.sts = #[Setup.ofCircuit st] := by rw [hsts]; simp [tabsOf]
  have h1 : o.spec.tables.size = 1 := by rw [hspec]; rfl
  have hP : sendsC.P (sessionP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.spec (tabsOf o.sts)
      (tracesOf o.sts o.w) o.ω) (1 + rounds false o.spec o.sts) := by
    rw [sessionP_eq, htabs]
    refine sendsC_quiet_bind (schedsP_quiet _) fun schs hs => ?_
    obtain ⟨sch, hf, rfl⟩ := schedsP_one hs
    refine sendsC_quiet_bind (commitsP_quiet _ _ _ _ _) fun cms _ => ?_
    refine sendsC.mono (sendsC.bind (sendsC_link _ _ _) fun c1 =>
      afterLinkP_sends _ _ o.spec st sch _ _ _ cms c1 h1) ?_
    have hL := fast100_levels hf
    simp only [hsts, rounds, repRounds, hf, List.foldl_toArray', List.foldl_cons, List.foldl_nil,
      Bool.false_eq_true, ↓reduceIte]
    omega
  have := hP #[] _ _ o.ran
  simpa using this

end FlockComplete
