import Proofs.Flock.Claims.Complete.Walk

/-!
# The session walk on the packed table

The honest session commits each table's packed trace at level 0 and runs both reps on it (`sessionP_reps` at
`sessionS`), so the session walk holds on the packed table at every rep (`Run.repWalkP`, `RepWalkP`). With ring
switching's claim and the verifier's basis proved at the deployed relation (`lgFinal_relation`), Ligerito's final
identity there rests on the honest Ligerito's alone (`lgFinal_walk`), and so does Flock's completeness at the deployed
plain verifier, beside the record's JSON and the oracle's 64-byte outputs (`complete_plain_honest`). With the honest
Ligerito's final identity proved (`Lg.honest`), `LgFinal` rests on the record's JSON and the oracle's 64-byte outputs
alone (`lgFinal_json`, by `lgFinal_walkP`), and Flock's completeness at the deployed plain verifier on the record's JSON
alone (`complete_plain_json`): `Claims.Complete` reads the oracle's 64-byte outputs (`Out64`) itself.
-/

namespace FlockComplete.Replay

open Flock Transcript FlockDeploy

/-- **The session walk at a run's rep, on the packed table**: rep `r` of the run's one table, run on the trace's packed
table and the table's level-0 commitment of it, from the rep's first send on and its own transcript's start, returns
the decoded proof. -/
theorem _root_.FlockComplete.Run.walkP {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : JsonRT.ProverOk H rel) (o : Run H rel)
    {r : ℕ} (hr : r < 2) : RepWalkP o 0 r := by
  obtain ⟨st, hst, hsp, -, -⟩ := o.table
  have ht : o.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  obtain ⟨hct, hsig, hdraw, -⟩ := JsonRT.stmtOk o.toCase
  obtain ⟨hC, hRk⟩ := hP o.toCase st (by rw [hst]; simp)
  have htabs : o.tabs = #[Setup.ofCircuit st] := by simp [Run.tabs, tabsOf, hst]
  have htab0 : o.tabs[0]! = Setup.ofCircuit st := by rw [htabs]; rfl
  have hrun := o.ran
  unfold sessionS at hrun
  rw [show tabsOf o.sts = #[Setup.ofCircuit st] from htabs] at hrun
  obtain ⟨sch, cm, e, hsch, hcm, he, p₀, t₀, l₁, p₁, t₁, hr₀, hr₁, hps, hrec⟩ :=
    JsonRT.sessionP_reps H o.resp ht hct (Setup.ofCircuit st) _ _ hrun
  dsimp only at hps hrec
  have hok₀ := JsonRT.repP_rounds _ _ _ _ _ _ _ _ _ _ _ _ (hRk sch hsch _ _ _ _ _ _ _ he) hr₀
  have hok₁ := JsonRT.repP_rounds _ _ _ _ _ _ _ _ _ _ _ _ (hRk sch hsch _ _ _ _ _ _ _ he) hr₁
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, JsonRT.ptsOf_coin] at he hrec
  have hcap := hC sch hsch _ _ cm hcm
  have hrs1 := JsonRT.repStreams_one ht
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  have hdec : Record.decodeWith o.hs o.spec o.record = .ok (JsonRT.sessOf o.spec (Claims.hOf H) #[flat cm.cap]
      (pointsOf o.spec (Claims.coinSeq o.r 0)) #[t₀.rounds, t₁.rounds] o.proofs) := by
    rw [hrec]
    refine JsonRT.decode_recordObj S (JsonRT.hOf_size h64) ht hct hdraw hsig hcap.1 hpts hpm (by rw [hrs1]; rfl) ?_
      o.proofs
    intro rs hrs
    simp only [List.mem_toArray, List.mem_cons, List.not_mem_nil, or_false] at hrs
    rcases hrs with rfl | rfl
    · exact hok₀.1
    · exact hok₁.1
  have hsess : o.sess = JsonRT.sessOf o.spec (Claims.hOf H) #[flat cm.cap] (pointsOf o.spec (Claims.coinSeq o.r 0))
      #[t₀.rounds, t₁.rounds] o.proofs := by
    unfold Run.sess
    rw [hdec]
    rfl
  have hsch0 : o.sch 0 = sch := by simp [Run.sch, schedsOf, htabs, hsch]
  have hext0 : o.extras 0 = e := by
    unfold Run.extras
    rw [htab0, hsess]
    simp only [JsonRT.sessOf, he]
    rfl
  have hdp : ∀ {p salt nonce d l t' l'}, ((repP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st)
      sch (tracesOf o.sts o.w)[0]! (packed (tracesOf o.sts o.w)[0]! (Setup.ofCircuit st).m) cm e salt nonce).run
      (Transcript.init #[] d)).go l = (.ok (p, t'), l') →
      decodeProof (shapeOf o.tabs[0]! (o.sch 0) (o.extras 0).size)
        (FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p) = .ok p := by
    intro p salt nonce d l t' l' h
    have hR' := JsonRT.repShaped h64 o.toCase st sch cm e hst hsch _ hcm he salt nonce d
    have h1 := JsonRT.Ret.go hR' h
    rw [hsch0, hext0, htab0]
    exact FlockVBridge.Family.decodeProof_encodeProof _ _ h1
  have hm : 14 ≤ (Setup.ofCircuit st).m := by
    have := (o.tab_m (i := 0) (by rw [htabs]; simp)).2.1
    rw [htab0] at this
    omega
  obtain ⟨s0, hcm0⟩ : ∃ s0, commitP o.hs o.tabs[0]!.merkle ((o.sch 0).levels.getD 0 default)
      (packed (o.trace 0) o.tabs[0]!.m) s0 = .ok cm :=
    ⟨_, by rw [htab0, hsch0]; exact hcm⟩
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · have hp : o.proof 0 0 = p₀ := by
      unfold Run.proof Run.bytes
      rw [hps]
      rw [show (#[FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₀,
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₁])[2 * 0 + 0]! =
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₀ from rfl, hdp hr₀]
      rfl
    refine ⟨cm, s0, o.ω.salt 0 0, o.ω.nonce 0 0, #[linkBody [flat cm.cap] [ByteArray.empty]], t₀, l₁, hcm0,
      by simp [repStart], ?_⟩
    rw [htab0, hsch0, hext0, hp]
    exact hr₀
  · have hp : o.proof 0 1 = p₁ := by
      unfold Run.proof Run.bytes
      rw [hps]
      rw [show (#[FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₀,
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₁])[2 * 0 + 1]! =
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₁ from rfl, hdp hr₁]
      rfl
    have hcount := repP_count _ _ _ _ _ _ _ _ _ _ hm hsch _ _ hr₀
    have hx : e.size = o.spec.points * st.regions.size := by
      rw [FlockDeploy.extraClaims_size (st := st) he, hpts, Nat.mul_comm]
    refine ⟨cm, s0, o.ω.salt 0 1, o.ω.nonce 0 1, l₁, t₁, o.bodies, hcm0, ?_, ?_⟩
    · rw [hcount, hx]
      simp only [repStart, repRounds, hst]
      simp only [Setup.ofCircuit, Stmt.circuitFold] at hsch ⊢
      simp [hsch]
    · rw [htab0, hsch0, hext0, hp]
      exact hr₁

/-- **The session walk on the packed table at every rep** (`RepWalkP`). -/
theorem _root_.FlockComplete.Run.repWalkP {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : JsonRT.ProverOk H rel) (o : Run H rel)
    {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) : RepWalkP o i r := by
  obtain rfl : i = 0 := by have := o.tabs_size_one; omega
  exact o.walkP S h64 hP hr

/-- **Ligerito's final identity at the deployed relation** (`LgFinal`), from the honest Ligerito's (`Lg.Honest`), at the
record's JSON and the oracle's 64-byte outputs. -/
theorem lgFinal_walk (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H)
    (hH : Lg.Honest) : LgFinal H Readable.relation :=
  lgFinal_relation H (fun o _ hi _ hr => o.repWalkP S h64 (proverOk h64) hi hr) hH

/-- **Flock's completeness at the deployed plain verifier**, at the deployed relation and the record's error, from the
record's JSON (`JsonSpec`), the oracle's 64-byte outputs (`Out64`) and the honest Ligerito's final identity
(`Lg.Honest`). -/
theorem complete_plain_honest (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (h64 : Claims.Out64 H) (hH : Lg.Honest) : Claims.Flock.Complete H (deployPlain Readable.relation) :=
  complete_plain H S h64 (lgFinal_walk H S h64 hH)

/-- **Ligerito's final identity at the deployed relation** (`LgFinal`), at the record's JSON and the oracle's 64-byte
outputs. -/
theorem lgFinal_json (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) :
    LgFinal H Readable.relation :=
  lgFinal_walkP H fun o _ hi _ hr => o.repWalkP S h64 (proverOk h64) hi hr

/-- **Flock's completeness at the deployed plain verifier**, at the deployed relation, of the record's reference prover
(`honestP`) at the record's error (`δc H`), from the record's JSON (`JsonSpec`) alone: the oracle's 64-byte outputs
(`Out64`) are `Claims.Complete`'s own antecedent. -/
theorem complete_plain_json (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) :
    Claims.Flock.Complete H (deployPlain Readable.relation) := fun h64 =>
  complete_plain H S h64 (lgFinal_json H S h64) h64

end FlockComplete.Replay
