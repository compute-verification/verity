import Proofs.Flock.Claims.Complete.Zk.Walk
import Proofs.Flock.Claims.Complete.Zk.Transfer
import Proofs.Flock.Claims.Complete.Replay.ZerocheckGoal

/-!
# The masked replay passes (`Replay`)

The masked prover runs the verifier's own replay (`Zk.replay`) on its rounds so far, inside the rep, before its
constraints and inner proof; a rep that returns had that replay return (`repPZk_replay`), on rounds its final rounds
begin with: what follows adds rounds only (`innerP_rec`). At the run, rep `r`'s recorded stream is the rounds the rep
ended at, every round retained (`RunZk.walk`), so the stream carries the rounds that replay ran on, and the verifier's
replay at the run's oracle returns too (`replayWith_xfer`, from `Zk.replay` as `Zk.replayWith` at SHA-512,
`zk_replayWith_std`): `replay_goal`.
-/

namespace FlockComplete

open Flock Transcript FlockComplete.Replay

theorem get_ret (t : Transcript) : Ret ((get : TMW PS Transcript).run t) fun at' => at'.1 = t ∧ at'.2 = t := by
  intro l a l' h
  simp only [StateT.run_get, JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at h
  obtain ⟨rfl, -⟩ := h
  exact ⟨rfl, rfl⟩

theorem liftV_ret {α : Type} (v : V α) (t : Transcript) :
    Ret ((StateT.lift (monadLift v : PS α) : TMW PS α).run t) fun at' => v = .ok at'.1 ∧ at'.2 = t := by
  intro l a l' e
  change ((monadLift v : PS _) >>= fun a => Pure.pure (a, t)).go l = _ at e
  obtain ⟨b, l₁, e1, e2⟩ := JsonRT.go_bind_ok e
  rw [JsonRT.monadLift_go] at e1
  simp only [Prod.mk.injEq] at e1
  obtain ⟨hv, rfl⟩ := e1
  cases e2
  exact ⟨hv, rfl⟩

/-- **The inner proof only adds rounds.** -/
theorem innerP_pre (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) (t : Transcript) :
    Ret ((innerP (asksS resp) (Hashes.ofSha512 h) lay p ic finalC cons).run t) fun pt =>
      Prefix t.rounds pt.2.rounds := by
  intro l a l' e
  obtain ⟨rs, bs, hrs, -, -⟩ := JsonRT.Rec.go (JsonRT.innerP_rec resp h lay p ic finalC cons) e
  exact ⟨rs.toArray, Array.toList_inj.1 (by simp [hrs])⟩

/-- **A masked rep that returns had its replay return**, on rounds its final rounds begin with. -/
theorem repPZk_replay (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (t₀ : Transcript) :
    Ret ((repPZk (asksS resp) (Hashes.ofSha512 h) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad
      salt nonce prior words).run t₀) fun pt => ∃ R rp t₁,
        (Zk.replay st (Zk.PadLayout.new st.m st.kLog sch.k0) pt.1.proof ic.root).run (Transcript.init R dom) =
          .ok (rp, t₁) ∧ Prefix R pt.2.rounds := by
  unfold repPZk
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨zp, zc⟩ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨lp, lc⟩ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun _ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine Replay.Ret.run_bind (Ret.true _) fun ⟨lg, zl, rec⟩ s₄ _ => ?_
  refine Replay.Ret.run_bind (get_ret s₄) fun tr' s₅ ⟨h1, h2⟩ => ?_
  refine Replay.Ret.run_bind (liftV_ret _ s₅) fun ⟨rp, x⟩ s₆ ⟨h3, h4⟩ => ?_
  refine Replay.Ret.run_bind (liftV_ret _ s₆) fun lgc s₇ ⟨_, h5⟩ => ?_
  refine Replay.Ret.run_bind (innerP_pre resp h _ _ _ _ _ s₇) fun inner s₈ h6 => ?_
  rw [StateT.run_pure]
  refine Ret.pure ⟨tr'.rounds, rp, x, h3, ?_⟩
  dsimp only at h1 h2 h4 h5 ⊢
  rw [h1, ← h2, ← h4, ← h5]
  exact h6

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The masked replay passes** from the rep's recorded rounds: the goal `Replay`. -/
theorem replay_goal (h64 : Claims.Out64 H) (hS : JsonRT.StmtOkZk H rel)
    (hP : JsonRT.ProverOkZk H rel) (hR : JsonRT.RepShapedZk H rel) : Replay H rel := by
  intro o i hi r hr
  obtain ⟨st, v, cm, e, streams, htabs, hsch, hss, hbytes, hstream, -, -, hrep⟩ := JsonRT.RunZk.walk h64 hS hP hR o
  rw [htabs] at hi
  obtain rfl : i = 0 := by simpa using hi
  obtain ⟨p, ic, l, t', l', prior, ws, -, hzp, hst', hrun⟩ := hrep r hr
  have hroot := (JsonRT.Ret.go (JsonRT.repPZk_fields _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun).2.2
  obtain ⟨R, rp, t₁, hre, C, hC⟩ := JsonRT.Ret.go (repPZk_replay _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun
  refine Replay.ok_val ?_
  have hlay : o.lay 0 = Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0 := by
    simp [RunZk.lay, htabs, hsch]
  rw [hzp, hroot, hlay, show o.tabs[0]! = Setup.ofCircuit st by rw [htabs]; rfl]
  rw [← With.zk_replayWith_std] at hre
  have hcarry : Carry (Transcript.init R (o.spec.domain o.spec.tables[0]! r).toUTF8) (o.t₀ 0 r) := by
    refine ⟨rfl, rfl, fun j hj => ?_⟩
    change j < R.size at hj
    have hj' : j < t'.rounds.size := by
      rw [hC, Array.size_append]
      exact Nat.lt_of_lt_of_le hj (Nat.le_add_right _ _)
    have hRj : R[j] = t'.rounds[j] := by
      have h1 : t'.rounds[j]? = R[j]? := by rw [hC, Array.getElem?_append_left hj]
      rw [Array.getElem?_eq_getElem hj', Array.getElem?_eq_getElem hj] at h1
      exact (Option.some.inj h1).symm
    have hmem : t'.rounds[j] ∈ streams[r]! := by
      rw [hst']
      exact Array.getElem_mem hj'
    obtain ⟨c, hc⟩ := hbytes _ (by rw [getElem!_pos streams r (by omega)]; exact Array.getElem_mem _) _ hmem
    change R[j].bytes.isSome = true ∧ Carries (o.stream 0 r).rounds j R[j]
    rw [hRj, hc]
    refine ⟨rfl, JsonRT.roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [JsonRT.roundRec, hc]⟩
    rw [hstream r hr, hst']
    simp [hj']
  obtain ⟨u', hu', -⟩ := replayWith_xfer With.std o.hs _ _ _ _ _ _ _ _ hcarry hre
  exact ⟨_, hu'⟩

end FlockComplete
