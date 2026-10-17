import Proofs.Flock.Claims.Complete.Replay.LigeritoProver
import Proofs.Flock.Claims.Complete.Replay.LigeritoGoal

/-!
# A returning rep carries the model's Ligerito

`repP_ligerito`: a rep that returns (`repP`) ends its transcript where the verifier's model (`Lg.out`) does on the rep's
own Ligerito proof, at the ring switch's `T` and from where the prover's ring switch stopped, at the replies' coins from
the rep's first send on (`ligeritoP_go` after `zerocheckP_go`, `lincheckP_go` and `ringSwitchP_go`); its first cap is
the table's commitment's. A rep that returns has its circuit fold return at the lincheck's `α` (`repP_fold`).

At a run (`Run.lgOut_eq`): once the honest rep, run from the rep's first send on and its own transcript's start,
returns the decoded proof, the model's transcript at the rep (`Run.lgOut`) is where the rep's ended. So the session
walk's Ligerito facts (`LgAt`, `LgCount`) follow from the recorded stream carrying the rep's rounds and holding as many
(`Run.lgAt_of`, `Run.lgCount_of`), and lincheck's circuit fold from the run alone (`Run.lcFold_of`).
-/

namespace FlockComplete.Replay

open Flock Transcript

/-- **A rep that returns has its circuit fold return** at the lincheck's `α`, the reply to its first send after
zerocheck's. -/
theorem repP_fold (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t).go l = (.ok (p, t'), l')) :
    ∃ full, st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  unfold repP at h
  rw [bind_go hz, StateT.run_bind] at h
  obtain ⟨⟨⟨lp, lc⟩, t₂⟩, l₂, h1, -⟩ := JsonRT.go_bind_ok h
  unfold lincheckP at h1
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _), bind_go (sampleP_go resp _ _),
    bind_go (sampleP_go resp _ _)] at h1
  dsimp only at h1
  rw [hl₁] at h1
  revert h1
  split
  · intro _; exact ⟨_, ‹_›⟩
  · intro h1
    rw [StateT.run_bind, go_bind] at h1
    cases h1

/-- **A rep that returns carries the model's Ligerito**: its transcript ends at `Lg.out`'s on its Ligerito proof, at
the ring switch's `T`, from where the prover's ring switch stopped, at the replies' coins from the rep's first send on;
it keeps one send per round, and its cap and its proof's first cap are the commitment's. -/
theorem repP_ligerito (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (hn : sch.n ≠ 0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) (t : Transcript) (ht : t.k = 0)
    (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    p.cap = cm0.cap ∧ p.ligerito.initialCap = cm0.cap ∧
      t' = (Lg.out p.ligerito (Lg.cfOf resp l.size) sch
        (Rs.T tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size))
        cm0.cap
        (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
          (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
            (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
              (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
                (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)
          (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
            (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)))).2.2 ∧
      l'.size = l.size + t'.k := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, hl₂, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl₂ hl
  obtain ⟨l₃, hl₃, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) (by simp [abcClaims])
    (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
      (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)) l₂
  rw [hl₂] at hl₃ hrs
  unfold repP at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [bind_go hrs, StateT.run_bind] at hrun
  obtain ⟨⟨lg, t₄⟩, l₄, h1, h2⟩ := JsonRT.go_bind_ok hrun
  rw [pure_go] at h2
  simp only [Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := h2
  obtain ⟨hcap, ht₄, hl₄⟩ := Lg.ligeritoP_go resp h st.merkle sch _ cm0 tbl _ salt nonce l.size hn hL hq
    (by rw [hl₃, Rs.tEnd_k, Lc.tEnd_k, Zc.tEnd_k, ht]; omega) h1
  exact ⟨rfl, hcap, ht₄, hl₄⟩

/-- **Table `i`'s rep `r`'s Ligerito from the rep's run**: once the honest rep, run from the rep's first send on and its
own transcript's start, returns the decoded proof, the model's transcript at the rep is where the rep's ended, and the
proof's first cap is its cap. -/
theorem _root_.FlockComplete.Run.lgOut_eq {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (h : ByteArray → ByteArray) (tbl : Array F128)
    (cm0 : Commit) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {l : Array Claims.Msg}
    (hl : l.size = repStart o.spec o.sts (2 * i + r)) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS o.resp) (Hashes.ofSha512 h) o.tabs[i]! (o.sch i) (o.trace i) tbl cm0 (o.extras i) salt
      nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l = (.ok (o.proof i r, t'), l')) :
    (o.lgOut i r).2.2 = t' ∧ (o.proof i r).ligerito.initialCap = (o.proof i r).cap := by
  have hm : 14 ≤ o.tabs[i]!.m := by have := (o.tab_m hi).2.1; omega
  obtain ⟨hn, hL, -, hq, -⟩ := fast100_ligerito (o.sch_fast100 hi)
  obtain ⟨full, hfold⟩ := repP_fold o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨hcap, hinit, ht', -⟩ := repP_ligerito o.resp h _ _ _ tbl cm0 _ salt nonce hm hn hL hq _ rfl l hfold hrun
  rw [hl] at hfold
  have hfull : o.full i r = full := congrArg val hfold
  refine ⟨?_, hinit.trans hcap.symm⟩
  rw [ht', hl, ← hcap, ← hfull]
  rfl

/-- **The walk's `LgAt` from the rep's run**: the recorded stream carries the rep's rounds. -/
theorem _root_.FlockComplete.Run.lgAt_of {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (h : ByteArray → ByteArray) (tbl : Array F128)
    (cm0 : Commit) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {l : Array Claims.Msg}
    (hl : l.size = repStart o.spec o.sts (2 * i + r)) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS o.resp) (Hashes.ofSha512 h) o.tabs[i]! (o.sch i) (o.trace i) tbl cm0 (o.extras i) salt
      nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l = (.ok (o.proof i r, t'), l'))
    (hag : Agrees (o.stream i r).rounds t'.rounds 0) : LgAt o i r := by
  obtain ⟨e, hcap⟩ := o.lgOut_eq hi r h tbl cm0 salt nonce hl hrun
  exact ⟨hcap, by rw [e]; exact hag⟩

/-- **The walk's `LgCount` from the rep's run**: the recorded stream holds as many rounds as the rep's. -/
theorem _root_.FlockComplete.Run.lgCount_of {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (h : ByteArray → ByteArray) (tbl : Array F128)
    (cm0 : Commit) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {l : Array Claims.Msg}
    (hl : l.size = repStart o.spec o.sts (2 * i + r)) {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS o.resp) (Hashes.ofSha512 h) o.tabs[i]! (o.sch i) (o.trace i) tbl cm0 (o.extras i) salt
      nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[i]! r).toUTF8)).go l = (.ok (o.proof i r, t'), l'))
    (hsz : (o.stream i r).rounds.size = t'.rounds.size) : LgCount o i r := by
  obtain ⟨e, -⟩ := o.lgOut_eq hi r h tbl cm0 salt nonce hl hrun
  show _ = _
  rw [e]
  exact hsz

/-- **Lincheck's circuit fold returns** at table `i`'s rep `r` (`LcAt`'s first part), once the honest rep, run from
the rep's first send on, returns. -/
theorem _root_.FlockComplete.Run.lcFold_of {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (o : Run H rel) {i : ℕ} (hi : i < o.tabs.size) (r : ℕ) (hsP : Hashes PS) (tbl : Array F128) (cm0 : Commit)
    (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) {l : Array Claims.Msg}
    (hl : l.size = repStart o.spec o.sts (2 * i + r)) {t : Transcript} {p : Proof} {t' : Transcript}
    {l' : Array Claims.Msg}
    (hrun : ((repP (asksS o.resp) hsP o.tabs[i]! (o.sch i) (o.trace i) tbl cm0 (o.extras i) salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    o.tabs[i]!.fold.fold ((o.lcCoins i r).ca 0) (skipWeights (o.zcOut i r).z)
      ((o.zcOut i r).rho.extract 0 (o.tabs[i]!.fold.kLog - 6)) = .ok (o.full i r) := by
  have hm : 14 ≤ o.tabs[i]!.m := by have := (o.tab_m hi).2.1; omega
  obtain ⟨full, hfold⟩ := repP_fold o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  rw [hl] at hfold
  have hfull : o.full i r = full := congrArg val hfold
  rw [hfull]
  exact hfold

end FlockComplete.Replay
