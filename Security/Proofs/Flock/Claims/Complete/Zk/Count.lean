import Proofs.Flock.Claims.Complete.Count
import Proofs.Flock.Claims.Complete.Zk.Link

/-!
# The honest masked session's sends fit the masked verifier's rounds

`Count.lean` at the masked session, the goal `SendCountZk`:

* **Per step**: the masked zerocheck `3 + (m − 6)` (`zerocheckPZk_sends`), lincheck `3 + (kLog − 6)`
  (`lincheckPZk_sends`), level 0's masked Ligerito `k0 + 11` and then `11` per recursive level but the last, which
  sends `7` (`ligeritoPZk_sends`), the inner proof `4` (`innerP_sends`; the tau commitment is quiet), a rep their sum
  with the ring switch's (`repPZk_sends`).
* **The session** (`afterLinkPZk_sends`): one table's two reps after the link message, the pads' commitment quiet
  (`innerCommitP_quiet`); the table's `--zk` schedule keeps fast100's at least one lane round and two levels
  (`zkSchedule_shape`), so that is `repRounds true`, and `sendCountZk_goal` is the goal `SendCountZk`.
-/

open Flock FlockDeploy

namespace FlockComplete

open With

/-- **The masked zerocheck** sends at most `3 + (m − 6)` rounds. -/
theorem zerocheckPZk_sends (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) :
    (sendsC.st Transcript).P (zerocheckPZk (asksS resp) m digest cap root lay h tr) (3 + (m - 6)) := by
  apply (sendsC.st Transcript).mono
  · unfold zerocheckPZk sampleVecP
    cost_walk [sampleP_sends resp, squeezeP_sends resp, Cost.forIn_range]
  · simp only [Std.Legacy.Range.size]
    omega

/-- **The masked lincheck** sends at most `3 + (kLog − 6)` rounds. -/
theorem lincheckPZk_sends (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ) (lay : Zk.PadLayout) (h : Array F128)
    (tr : Trace) (zc : ZcOut) :
    (sendsC.st Transcript).P (lincheckPZk (asksS resp) c m lay h tr zc) (3 + (c.kLog - 6)) := by
  apply (sendsC.st Transcript).mono
  · unfold lincheckPZk
    cost_walk [sampleP_sends resp, Cost.forIn_range]
  · simp only [Std.Legacy.Range.size]
    omega

/-- The tau commitment sends nothing. -/
theorem tau_sends (h : ByteArray → ByteArray) (tau : F128) (salt : ByteArray) :
    sendsC.P (Zk.tauCommitmentWith (Hashes.ofSha512 h : Hashes PS) tau salt) 0 :=
  sendsC_quiet (zk_tauCommitmentWith_sim quietS (quietS_hashes h) tau salt)

/-- A stop sends nothing. -/
theorem sendsC_st_throw {α : Type} (e : String) : (sendsC.st Transcript).P (throw e : TMW PS α) 0 :=
  (sendsC.st Transcript).throw e 0

/-- **The inner proof** sends at most `4` rounds. -/
theorem innerP_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    (sendsC.st Transcript).P (innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons) 4 := by
  apply (sendsC.st Transcript).mono
  · unfold innerP sampleVecP
    cost_walk [sampleP_sends resp, squeezeP_sends resp, tau_sends h, sendsC_st_throw]
  · omega

/-- A stop inside Ligerito sends nothing. -/
theorem sendsC_lp_throw {α : Type} (e : String) : ((sendsC.st Transcript).st LigState).P (throw e : LP PS α) 0 :=
  ((sendsC.st Transcript).st LigState).throw e 0

/-- **Level 0's masked Ligerito** sends at most `k0 + 11`, then `11` per recursive level but the last's `7`. -/
theorem ligeritoPZk_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (hp : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    (sendsC.st Transcript).P (ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 t b lay hp rep tPad extra
      lanePad salt nonce) (2 + ((sch.k0 - 1) * 1 + 3) + 7 + ((sch.levels.size - 1 - 1) * 11 + 7)) := by
  apply (sendsC.st Transcript).mono
  · unfold ligeritoPZk liftT
    extract_lets n R lv lanes
    cost_walk [sampleVecP_sends resp, sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends,
      challengeP_sends resp, oodIntroP_sends resp, queryP_sends resp, commitP_sends h, sendsC_lp_throw]
    refine Cost.forIn_last _ sch.k0 _ (k := 1) (k' := 3) (fun i hi b => ?_) (fun hn b => ?_)
    · apply ((sendsC.st Transcript).st LigState).mono
      · simp only [show (i + 1 == sch.k0) = false from by simp; omega, Bool.false_eq_true, ↓reduceIte]
        cost_walk [sampleVecP_sends resp, sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends,
          challengeP_sends resp, oodIntroP_sends resp, queryP_sends resp, commitP_sends h, sendsC_lp_throw]
      · omega
    · apply ((sendsC.st Transcript).st LigState).mono
      · cost_walk [sampleVecP_sends resp, sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends,
          challengeP_sends resp, oodIntroP_sends resp, queryP_sends resp, commitP_sends h, sendsC_lp_throw]
      · omega
    all_goals cost_walk [sampleVecP_sends resp, sampleP_sends resp, sampleKP_sends resp, sendMsg_sends, powP_sends,
      challengeP_sends resp, oodIntroP_sends resp, queryP_sends resp, commitP_sends h, sendsC_lp_throw]
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

/-- **A masked rep** sends at most the sum of its steps. -/
theorem repPZk_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) :
    (sendsC.st Transcript).P (repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic
      extra lanePad salt nonce)
      ((3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (2 + extras.size + 1) +
        (2 + ((sch.k0 - 1) * 1 + 3) + 7 + ((sch.levels.size - 1 - 1) * 11 + 7)) + 4) := by
  apply (sendsC.st Transcript).mono
  · unfold repPZk
    cost_walk [zerocheckPZk_sends resp, lincheckPZk_sends resp, ringSwitchP_abc_sends resp,
      ligeritoPZk_sends resp h, innerP_sends resp h]
  · omega

/-- The pads' commitment is quiet. -/
theorem innerCommitP_quiet (h : ByteArray → ByteArray) (p : PadsZk) (salt : ℕ → ByteArray) :
    quietS.R (innerCommitP (Hashes.ofSha512 h : Hashes V) p salt)
      (innerCommitP (Hashes.ofSha512 h : Hashes PS) p salt) := by
  have hh := quietS_hashes h
  unfold innerCommitP MerkleOps.hm96With
  sim_walk [leafWith_sim quietS hh]

/-- **The masked session after the link** sends at most two masked reps' rounds. -/
theorem afterLinkPZk_sends (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Stmt)
    (v : Schedule × ℕ) (trs : Array Trace) (tape : TapeZk) (tbls : Array (Array F128)) (cms : Array Commit)
    (c1 : ℕ → F128) (hspec : spec.tables.size = 1) :
    sendsC.P (afterLinkPZk (asksS resp) (Hashes.ofSha512 h) spec #[Setup.ofCircuit st] trs tape #[v] tbls cms c1)
      (2 * ((3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) + (2 + spec.points * st.regions.size + 1) +
        (2 + ((v.1.k0 - 1) * 1 + 3) + 7 + ((v.1.levels.size - 1 - 1) * 11 + 7)) + 4)) := by
  unfold afterLinkPZk
  extract_lets rootB points streams proofs
  have hpts : points.size = spec.points := by simp [points]
  refine sendsC.mono (sendsC.bind (sendsC.forIn_array _ _ (k := (3 + (st.m - 6)) + (3 + (st.c.kLog - 6)) +
      (2 + spec.points * st.regions.size + 1) +
      (2 + ((v.1.k0 - 1) * 1 + 3) + 7 + ((v.1.levels.size - 1 - 1) * 11 + 7)) + 4)
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
    refine sendsC_quiet_bind (innerCommitP_quiet h _ _) fun ic _ => ?_
    apply sendsC.mono
    · cost_walk [repPZk_sends resp h]
    · simp only [List.getElem!_toArray, List.getElem!_cons_zero, Setup.ofCircuit, Stmt.circuitFold, hx]
      omega
  · cost_walk [recordJson_sends h]
  · rw [FlockDeploy.repStreams_size, hspec]
    omega

/-- One table's `--zk` schedules: its `--zk` schedule. -/
theorem schedsPZk_one {st : Stmt} {schs : Array (Schedule × ℕ)}
    (hs : (schedsPZk #[Setup.ofCircuit st] : V _) = .ok schs) : ∃ v, Zk.schedule st.m = .ok v ∧ schs = #[v] := by
  unfold schedsPZk at hs
  obtain ⟨h1, h2⟩ := FlockSoundness.Discharge.Exec.array_mapM_ok_getElem hs
  have h0 := h2 0 (by simp) (by rw [h1]; simp)
  simp only [List.getElem_toArray, List.getElem_cons_zero, Setup.ofCircuit] at h0
  refine ⟨schs[0]'(by rw [h1]; simp), h0, ?_⟩
  apply Array.ext (by simpa using h1)
  intro j hj _
  have : j = 0 := by simp at h1; omega
  subst this
  rfl

/-- fast100's schedules have at least one lane round. -/
theorem fast100_k0 {m : ℕ} {sch : Schedule} (h : fast100 m = some sch) : 1 ≤ sch.k0 := by
  by_cases hm : m < 40
  · have key : ∀ m : Fin 40, (fast100 m.val).all (fun s => decide (1 ≤ s.k0)) = true := by
      decide +kernel
    have := key ⟨m, hm⟩
    simp only [h, Option.all_some, decide_eq_true_eq] at this
    exact this
  · rw [FlockSoundness.Refine.exec_fast100_none m (by omega)] at h
    cases h

/-- The `--zk` schedule keeps fast100's lane rounds and levels: at least one, and at least two. -/
theorem zkSchedule_shape {m : ℕ} {v : Schedule × ℕ} (h : Zk.schedule m = .ok v) :
    1 ≤ v.1.k0 ∧ 2 ≤ v.1.levels.size := by
  unfold Zk.schedule at h
  obtain ⟨q0, -, h⟩ := except_ok_bind h
  cases hf : fast100 m with
  | none => rw [hf] at h; cases h
  | some sch =>
    rw [hf] at h
    cases h
    exact ⟨(fast100_k0 hf : 1 ≤ sch.k0), by simpa using fast100_levels hf⟩

/-- **The masked sends fit the masked verifier's rounds**: the goal `SendCountZk`. -/
theorem sendCountZk_goal (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    SendCountZk H rel := by
  intro o
  obtain ⟨st, hsts, hspec, hm1, -⟩ := o.table
  have htabs : tabsOf o.sts = #[Setup.ofCircuit st] := by rw [hsts]; simp [tabsOf]
  have h1 : o.spec.tables.size = 1 := by rw [hspec]; rfl
  have hP : sendsC.P (sessionPZk (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.spec (tabsOf o.sts)
      (MaskRank.masksOf o.sts) (tracesOf o.sts o.w) o.ω) (1 + rounds true o.spec o.sts) := by
    rw [sessionPZk_eq, htabs]
    refine sendsC_quiet_bind (schedsPZk_quiet _) fun schs hs => ?_
    obtain ⟨⟨sch, tp⟩, hv, rfl⟩ := schedsPZk_one hs
    refine sendsC_quiet_bind (commitsPZk_quiet _ _ _ _ _) fun cms _ => ?_
    refine sendsC.mono (sendsC.bind (sendsC_link _ _ _) fun c1 =>
      afterLinkPZk_sends _ _ o.spec st (sch, tp) _ _ _ cms c1 h1) ?_
    obtain ⟨hk0, hL⟩ := zkSchedule_shape hv
    simp only [hsts, rounds, repRounds, hv, List.foldl_toArray', List.foldl_cons, List.foldl_nil, ↓reduceIte]
    simp only at hk0 hL ⊢
    omega
  have := hP #[] _ _ o.ran
  simpa using this

end FlockComplete
