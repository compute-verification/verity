import Proofs.Flock.Claims.Complete.Count
import Proofs.Flock.Claims.Complete.Custody
import Proofs.Flock.Claims.Complete.Discharge
import Proofs.Flock.Claims.Complete.Files
import Proofs.Flock.Claims.Complete.Link
import Proofs.Flock.Claims.Complete.Runs
import Proofs.Flock.Claims.Complete.Shape
import Proofs.Flock.Claims.Complete.Replay.LigeritoHonestRun
import Proofs.Flock.Claims.Complete.Replay.ProverOk

/-!
# The session walk

What the honest session records for each rep, read at a run: Ligerito's walk premise `RepWalk` (`Run.repWalk`) and
the walk's facts `ZcAt`, `LcAt`, `RsAt`, `LgAt` and `LgCount` at every rep (`Run.walkAt`), and from them and the goals
proved beside it, Flock's completeness at the deployed plain verifier and the deployed relation from named premises
(`complete_plain`).

* `sessionP_reps`: the one-table session's two reps, unrolled: each runs from the sends before it, the proof files are
  their proofs' encodings, and the record is the honest record of their rounds.
* `repP_count`, `repP_prefix`: a returning rep makes one send per round of its four phases, and its rounds begin with
  those where its ring switch stopped (the honest Ligerito only adds rounds).
* `Run.walk`: at a run, rep `r` runs from `repStart`'s send on and returns the decoded proof, and the recorded stream is
  its rounds as the decoder reads them back (`roundRec`), which carry them, on the table's level-0 commitment.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

/-- **The one-table session's two reps**: the table's schedule, its level-0 commitment `cm` and its extra claims `e`
at the link's points; rep `0` runs from the link's send, rep `1` from where rep `0` stopped; the proof files are their
proofs' encodings, and the record is the honest record of their rounds. -/
theorem sessionP_reps (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) {spec : SessionSpec} {t : String}
    (ht : spec.tables = #[t]) (hct : spec.coinTree = none) (st : Setup) (trs : Array Trace) (tape : Tape)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec #[st] trs tape).go #[] = (.ok r, bodies)) :
    ∃ sch cm e, fast100 st.m = some sch ∧
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle (sch.levels.getD 0 default)
        (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm ∧
      st.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
      ∃ p₀ t₀ l₁ p₁ t₁,
        ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch trs[0]! (packed trs[0]! st.m) cm e (tape.salt 0 0)
          (tape.nonce 0 0)).run (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)).go
          #[linkBody [flat cm.cap] [ByteArray.empty]] = (.ok (p₀, t₀), l₁) ∧
        ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch trs[0]! (packed trs[0]! st.m) cm e (tape.salt 0 1)
          (tape.nonce 0 1)).run (Transcript.init #[] (spec.domain spec.tables[0]! 1).toUTF8)).go l₁ =
          (.ok (p₁, t₁), bodies) ∧
        r.2 = #[FlockVBridge.Family.encodeProof (shapeOf st sch e.size) p₀,
          FlockVBridge.Family.encodeProof (shapeOf st sch e.size) p₁] ∧
        r.1 = printJ (recordObj spec (Claims.hOf H) #[flat cm.cap] (ptsOf spec (coinOf (resp 0)))
          #[t₀.rounds, t₁.rounds] r.2) := by
  unfold sessionP at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  have hschs := Ret.go (Ret.mapM_array _ (fun st' sch => fast100 st'.m = some sch) _ fun a _ => by
    cases hf : fast100 a.m with
    | none => exact Ret.throw' _
    | some sch => exact Ret.pure rfl) hA
  have hl1 := Still.go (Still.mapM_array (fun a => by
    split
    · exact Still.pure _
    · exact Still.throw _) _) hA
  subst hl1
  obtain ⟨sch, rfl, hsch⟩ : ∃ sch, schs = #[sch] ∧ fast100 st.m = some sch := by
    obtain ⟨hsz, hf⟩ := hschs
    have h1 : schs.size = 1 := by simpa using hsz
    exact ⟨schs[0], eq_singleton_of h1, by simpa using hf 0 (by simp) (by omega)⟩
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun _ cm => commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (sch.levels.getD 0 default) (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm) _
    fun i hi => by
      have : i = 0 := by simpa using hi
      subst this
      exact Ret.of_pureS (commitP_pureS _ _ _ _ _) fun cm e => by simpa using e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitP_pureS _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨cm, rfl, hcm⟩ : ∃ cm, cms = #[cm] ∧ commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) st.merkle
      (sch.levels.getD 0 default) (packed trs[0]! st.m) (tape.salt 0 0 0) = .ok cm := by
    obtain ⟨hsz, hf⟩ := hcms
    have h1 : cms.size = 1 := by simpa using hsz
    exact ⟨cms[0], eq_singleton_of h1, hf 0 (by simp) (by omega)⟩
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  have hrs1 := repStreams_one ht
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  rw [hrs1, ← Array.forIn_toList] at hF
  simp only [List.forIn_cons, List.forIn_nil] at hF
  obtain ⟨y₀, l₅, h0, hF⟩ := go_bind_ok hF
  obtain ⟨e₀, l₆, he₀, h0⟩ := go_bind_ok h0
  rw [monadLift_go] at he₀
  simp only [Prod.mk.injEq] at he₀
  obtain ⟨he₀, rfl⟩ := he₀
  obtain ⟨⟨p₀, t₀⟩, l₁, hr₀, h0⟩ := go_bind_ok h0
  simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h0
  obtain ⟨rfl, rfl⟩ := h0
  dsimp only at hF
  obtain ⟨y₁, l₇, h1, hF⟩ := go_bind_ok hF
  obtain ⟨e₁, l₈, he₁, h1⟩ := go_bind_ok h1
  rw [monadLift_go] at he₁
  simp only [Prod.mk.injEq] at he₁
  obtain ⟨he₁, rfl⟩ := he₁
  rw [he₀] at he₁
  cases he₁
  obtain ⟨⟨p₁, t₁⟩, l₂, hr₁, h1⟩ := go_bind_ok h1
  simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at h1
  obtain ⟨rfl, rfl⟩ := h1
  simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hF
  obtain ⟨⟨rfl, rfl⟩, rfl⟩ := hF
  dsimp only at hrun
  rw [recordJson_eq _ hct] at hrun
  obtain ⟨rec, l₅, hJ, hrun⟩ := go_bind_ok hrun
  simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hJ hrun
  obtain ⟨rfl, rfl⟩ := hJ
  obtain ⟨rfl, rfl⟩ := hrun
  refine ⟨sch, cm, e₀, hsch, hcm, by simpa [ptsOf] using he₀, p₀, t₀, l₁, p₁, t₁, by simpa using hr₀,
    by simpa using hr₁, by simp, ?_⟩
  simp
  rfl

/-- Rounds that are sends retain their bytes. -/
theorem Recs.bytes {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg}, Recs resp i rs bs → ∀ r ∈ rs, ∃ c, r.bytes = some c
  | _, [], [], _, _, hr => by simp at hr
  | _, _ :: _, _ :: _, h, r, hr => by
    rcases List.mem_cons.1 hr with rfl | hr
    · obtain ⟨c, hc, -⟩ := h.1
      exact ⟨c, hc⟩
    · exact Recs.bytes h.2 r hr
  | _, [], _ :: _, h, _, _ => h.elim
  | _, _ :: _, [], h, _, _ => h.elim

/-- **A rep's recorded rounds**, run from its own transcript's start: a round, each of `1 ≤ n ≤ 65536` coins, and each
round's bytes retained. -/
theorem repP_rounds (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (d : ByteArray) (l : Array Claims.Msg) {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hok : Ret ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run
      (Transcript.init #[] d)) fun pt => RepOk pt.2.rounds)
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run
      (Transcript.init #[] d)).go l = (.ok (p, t'), l')) :
    (0 < t'.rounds.size ∧ ∀ x ∈ t'.rounds, 1 ≤ x.n ∧ x.n ≤ 65536 ∧ x.coins.size = x.n) ∧
      ∀ x ∈ t'.rounds, ∃ c, x.bytes = some c := by
  have hk := Ret.go hok hrun
  obtain ⟨rs, bs, hrs, -, hrec⟩ := Rec.go (repP_rec resp h _ _ _ _ _ _ _ _) hrun
  have hrs' : t'.rounds.toList = rs := by simpa [Transcript.init] using hrs
  refine ⟨⟨hk.1, fun x hx => ?_⟩, fun x hx => ?_⟩
  · have := Recs.mem hrec x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)
    exact ⟨this.1, hk.2 x hx, this.2⟩
  · exact Recs.bytes hrec x (by rw [← hrs']; exact Array.mem_toList_iff.2 hx)

end FlockComplete.JsonRT

namespace FlockComplete.Replay

open Flock Transcript FlockDeploy

/-- **A returning rep's sends**: one per round of its zerocheck, lincheck, ring switch and Ligerito. -/
theorem repP_count (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (hf : fast100 st.m = some sch) (d : ByteArray) (l : Array Claims.Msg) {p : Proof}
    {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run
      (Transcript.init #[] d)).go l = (.ok (p, t'), l')) :
    l'.size = l.size + ((3 + (st.m - 6)) + (3 + (st.fold.kLog - 6)) + (2 + extras.size + 1) +
      (sch.k0 + 5 + 11 * (sch.levels.size - 1))) := by
  obtain ⟨hn, hL, hsd, hq, -⟩ := fast100_ligerito hf
  obtain ⟨full, hfold⟩ := repP_fold resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  obtain ⟨-, -, ht', hl'⟩ := repP_ligerito resp h _ _ _ tbl cm0 _ salt nonce hm hn hL hq _ rfl l hfold hrun
  rw [hl', ht', (Lg.done_out _ _ sch _ _ _ hL hsd).k, Rs.tEnd_k, Lc.tEnd_k, Zc.tEnd_k]
  simp only [Transcript.init, Array.size_append, abcClaims, List.size_toArray, List.length_cons, List.length_nil]
  omega

/-- **A returning rep's rounds begin with its ring switch's**: the honest Ligerito only adds rounds
(`ligeritoP_rec`). -/
theorem repP_prefix (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (hm : 14 ≤ st.m) (t : Transcript) (l : Array Claims.Msg) {full : Array F128}
    (hfold : st.fold.fold (coinOf (resp (l.size + 3 + (st.m - 6))) 0)
      (skipWeights (Zc.out st.m (Zc.coinsAt resp l.size) tr).z)
      ((Zc.out st.m (Zc.coinsAt resp l.size) tr).rho.extract 0 (st.fold.kLog - 6)) = .ok full)
    {p : Proof} {t' : Transcript} {l' : Array Claims.Msg}
    (hrun : ((repP (asksS resp) (Hashes.ofSha512 h) st sch tr tbl cm0 extras salt nonce).run t).go l =
      (.ok (p, t'), l')) :
    Prefix (Rs.tEnd tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
        (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras)
      (Rs.coinsAt resp (l.size + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6))
        (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
          (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
            (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras).size)
      (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
        (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
        (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t))).rounds t'.rounds := by
  obtain ⟨l₁, hl₁, hz⟩ := zerocheckP_go resp hm st.digest (flat cm0.cap) tr t l
  obtain ⟨l₂, hl₂, hl⟩ := lincheckP_go resp st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
    (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t) l₁ (by rw [hl₁]; exact hfold)
  rw [hl₁] at hl₂ hl
  obtain ⟨l₃, -, hrs⟩ := ringSwitchP_go resp tr (abcClaims st.m st.kLog (Zc.out st.m (Zc.coinsAt resp l.size) tr)
    (Lc.out st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr)
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))) ++ extras) (by simp [abcClaims])
    (Lc.tEnd st.fold st.m tr (Zc.out st.m (Zc.coinsAt resp l.size) tr) full
      (Lc.coinsAt resp (l.size + 3 + (st.m - 6)) (st.fold.kLog - 6))
      (Zc.tEnd st.m st.digest (flat cm0.cap) (Zc.coinsAt resp l.size) tr t)) l₂
  rw [hl₂] at hrs
  unfold repP at hrun
  rw [bind_go hz] at hrun
  dsimp only at hrun
  rw [bind_go hl] at hrun
  dsimp only at hrun
  rw [bind_go hrs, StateT.run_bind] at hrun
  obtain ⟨⟨lg, t₄⟩, l₄, h1, h2⟩ := JsonRT.go_bind_ok hrun
  rw [pure_go] at h2
  simp only [Prod.mk.injEq, Except.ok.injEq] at h2
  obtain ⟨⟨-, rfl⟩, -⟩ := h2
  obtain ⟨rs, -, hrs', -, -⟩ := JsonRT.Rec.go (JsonRT.ligeritoP_rec resp h st.merkle sch _ cm0 tbl _ salt nonce) h1
  exact ⟨rs.toArray, Array.toList_inj.1 (by simpa using hrs')⟩

theorem Lc.prefix_tEnd {c : CircuitFold} {m : ℕ} {tr : Trace} {zc : ZcOut} {full : Array F128} {cs : Lc.Coins}
    (t : Transcript) : Prefix t.rounds (Lc.tEnd c m tr zc full cs t).rounds :=
  (Prefix.push _ _).trans (Lc.prefix_t₂ t)

theorem Rs.prefix_tEnd {tr : Trace} {claims : Array Claim} {cs : Rs.Coins} (t : Transcript) :
    Prefix t.rounds (Rs.tEnd tr claims cs t).rounds :=
  Rs.prefix_loop (i := 0) t (Nat.zero_le _)

/-- **The session walk at a run's rep**: rep `r` of the run's one table, run from its first send on and its own
transcript's start, returns the decoded proof; the recorded stream is its rounds as the decoder reads them back, each
of its rounds retains its bytes, and the rep's level-0 commitment is a `commitP` of the schedule's level `0`. -/
theorem _root_.FlockComplete.Run.walk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : JsonRT.ProverOk H rel) (o : Run H rel)
    {r : ℕ} (hr : r < 2) :
    ∃ (tbl : Array F128) (cm0 : Commit) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
      (l : Array Claims.Msg) (t' : Transcript) (l' : Array Claims.Msg),
      l.size = repStart o.spec o.sts (2 * 0 + r) ∧
      ((repP (asksS o.resp) (Hashes.ofSha512 (Claims.hOf H)) o.tabs[0]! (o.sch 0) (o.trace 0) tbl cm0 (o.extras 0)
        salt nonce).run (Transcript.init #[] (o.spec.domain o.spec.tables[0]! r).toUTF8)).go l =
        (.ok (o.proof 0 r, t'), l') ∧
      (o.stream 0 r).rounds = t'.rounds.map (JsonRT.roundRec o.spec (Claims.hOf H)) ∧
      (∀ x ∈ t'.rounds, ∃ c, x.bytes = some c) ∧
      ∃ s0, commitP o.hs o.tabs[0]!.merkle ((o.sch 0).levels.getD 0 default) tbl s0 = .ok cm0 := by
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
  have hstream : ∀ r < 2, (o.stream 0 r).rounds =
      (#[t₀.rounds, t₁.rounds])[r]!.map (JsonRT.roundRec o.spec (Claims.hOf H)) := by
    intro r hr
    unfold Run.stream
    rw [hsess]
    simp only [JsonRT.sessOf, hrs1, ht, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray,
      List.map_cons, List.map_nil, List.find?_toArray, JsonRT.streamRec]
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
    · simp
    · simp
  have hm : 14 ≤ (Setup.ofCircuit st).m := by
    have := (o.tab_m (i := 0) (by rw [htabs]; simp)).2.1
    rw [htab0] at this
    omega
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · have hp : o.proof 0 0 = p₀ := by
      unfold Run.proof Run.bytes
      rw [hps]
      rw [show (#[FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₀,
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₁])[2 * 0 + 0]! =
        FlockVBridge.Family.encodeProof (shapeOf (Setup.ofCircuit st) sch e.size) p₀ from rfl, hdp hr₀]
      rfl
    refine ⟨packed (tracesOf o.sts o.w)[0]! (Setup.ofCircuit st).m, cm, o.ω.salt 0 0, o.ω.nonce 0 0,
      #[linkBody [flat cm.cap] [ByteArray.empty]], t₀, l₁, by simp [repStart], ?_, by rw [hstream 0 (by omega)]; rfl,
      hok₀.2, _, by rw [htab0, hsch0]; exact hcm⟩
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
    refine ⟨packed (tracesOf o.sts o.w)[0]! (Setup.ofCircuit st).m, cm, o.ω.salt 0 1, o.ω.nonce 0 1, l₁, t₁, o.bodies,
      ?_, ?_, by rw [hstream 1 (by omega)]; rfl, hok₁.2, _, by rw [htab0, hsch0]; exact hcm⟩
    · rw [hcount, hx]
      simp only [repStart, repRounds, hst]
      simp only [Setup.ofCircuit, Stmt.circuitFold] at hsch ⊢
      simp [hsch]
    · rw [htab0, hsch0, hext0, hp]
      exact hr₁

theorem _root_.FlockComplete.Run.tabs_size_one {H : Claims.Msg → Claims.Msg}
    {rel : Readable.Parsed → Readable.Relation} (o : Run H rel) : o.tabs.size = 1 := by
  obtain ⟨st, hst, -⟩ := o.table
  simp [Run.tabs, tabsOf, hst]

/-- **The session walk at every rep** (`RepWalk`): table `i`'s rep `r`, on the table's level-0 commitment, run from the
rep's first send on and its own transcript's start, returns the decoded proof, and the recorded stream carries its
rounds and holds as many. -/
theorem _root_.FlockComplete.Run.repWalk {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : JsonRT.ProverOk H rel) (o : Run H rel)
    {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    RepWalk o i r := by
  obtain rfl : i = 0 := by have := o.tabs_size_one; omega
  obtain ⟨tbl, cm0, salt, nonce, l, t', l', hl, hrun, hst, hby, s0, h0⟩ := o.walk S h64 hP hr
  refine ⟨tbl, cm0, s0, salt, nonce, l, t', l', h0, hl, hrun, ?_, by rw [hst, Array.size_map]⟩
  rw [hst]
  intro j hj _
  obtain ⟨c, hc⟩ := hby _ (Array.getElem_mem hj)
  refine ⟨JsonRT.roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [JsonRT.roundRec, hc]⟩
  rw [Array.getElem?_map, Array.getElem?_eq_getElem hj]
  rfl

/-- **The session walk** at table `i`'s rep `r` of a run: the decoded proof's zerocheck, lincheck and ring switches are
the prover's, the statement's circuit fold returns, and the recorded stream carries the prover's rounds through each
phase (`ZcAt`, `LcAt`, `RsAt`), and the model's Ligerito rounds, as many as it records (`LgAt`, `LgCount`). -/
theorem _root_.FlockComplete.Run.walkAt {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}
    (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : JsonRT.ProverOk H rel) (o : Run H rel)
    {i : ℕ} (hi : i < o.tabs.size) {r : ℕ} (hr : r < 2) :
    ZcAt o i r ∧ LcAt o i r ∧ RsAt o i r ∧ LgAt o i r ∧ LgCount o i r := by
  obtain ⟨tbl, cm0, -, salt, nonce, l, t', l', -, hl, hrun, hag, hsz⟩ := o.repWalk S h64 hP hi hr
  obtain rfl : i = 0 := by have := o.tabs_size_one; omega
  have hm : 14 ≤ o.tabs[0]!.m := by have := (o.tab_m hi).2.1; omega
  obtain ⟨full, hfold⟩ := repP_fold o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  have hpre := repP_prefix o.resp _ _ _ _ _ _ _ _ _ hm _ l hfold hrun
  obtain ⟨hzc, hcap⟩ := repP_zerocheck o.resp _ _ _ _ _ _ _ _ _ hm _ l hrun
  have hlc := repP_lincheck o.resp _ _ _ _ _ _ _ _ _ hm _ l hfold hrun
  have hrs := repP_ringSwitch o.resp _ _ _ _ _ _ _ _ _ hm _ l hfold hrun
  rw [hl] at hfold hpre hzc hlc hrs
  have hfull : o.full 0 r = full := congrArg val hfold
  rw [← hcap] at hpre
  have hrsP : Prefix (o.rsEnd 0 r).rounds t'.rounds := by
    rw [← hfull] at hpre
    exact hpre
  have hlcP : Prefix (o.lcEnd 0 r).rounds t'.rounds := (Rs.prefix_tEnd _).trans hrsP
  have hzcP : Prefix (o.zcEnd 0 r).rounds t'.rounds := (Lc.prefix_tEnd _).trans hlcP
  refine ⟨⟨hzc, hag.of_prefix hzcP⟩, ⟨o.lcFold_of hi r _ tbl cm0 salt nonce hl hrun, ?_, hag.of_prefix hlcP⟩,
    ⟨?_, hag.of_prefix hrsP⟩, o.lgAt_of hi r _ tbl cm0 salt nonce hl hrun hag,
    o.lgCount_of hi r _ tbl cm0 salt nonce hl hrun hsz⟩
  · rw [hlc, hfull]
    rfl
  · rw [hrs]
    rfl

/-- **Flock's completeness at the deployed plain verifier**, at the deployed relation (`Readable.relation`) and the
record's error (`δc H`, `Complete.Ready`), from the named premises: the record's JSON (`JsonSpec`), the oracle's
64-byte outputs (`Out64`), and Ligerito's final identity at every rep (`LgFinal`). What the record needs of the
prover's pieces (`ProverOk`, `proverOk`) and the Merkle schemes' digest widths (`SchemeWidths`, `schemeWidths`) follow
from `Out64`, and the session walk (`Run.walkAt`, `Run.repWalk`) aligns every rep. -/
theorem complete_plain (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H)
    (hfin : LgFinal H Readable.relation) :
    Claims.Flock.Complete H (deployPlain Readable.relation) := by
  have hP : JsonRT.ProverOk H Readable.relation := proverOk h64
  have hW : SchemeWidths H Readable.relation := schemeWidths h64
  have w : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2,
      ZcAt o i r ∧ LcAt o i r ∧ RsAt o i r ∧ LgAt o i r ∧ LgCount o i r :=
    fun o _ hi _ hr => o.walkAt S h64 hP hi hr
  have hzc := fun o i hi r hr => (w o i hi r hr).1
  have hlc := fun o i hi r hr => (w o i hi r hr).2.1
  have hrs := fun o i hi r hr => (w o i hi r hr).2.2.1
  have hw : ∀ o : Run H Readable.relation, ∀ i < o.tabs.size, ∀ r < 2, RepWalk o i r :=
    fun o _ hi _ hr => o.repWalk S h64 hP hi hr
  have hD := JsonRT.repDecodes_goal S h64 hP
  exact complete_of_goals H Readable.relation
    { session := JsonRT.sessionRuns_goal
      sends := sendCount_goal H _
      linkFirst := linkFirst_goal H h64 _
      finalFits := JsonRT.finalFits_goal S h64 hP
      draw := JsonRT.drawMatches_goal S h64 hP
      liveReads := JsonRT.liveReads_goal S h64 hP
      canonical := JsonRT.canonical_goal S h64 hP
      rootB := JsonRT.custodyRootB_goal S h64 hP
      publicsLink := JsonRT.custodyPublics_goal S h64 hP
      points := JsonRT.custodyPoints_goal S h64 hP
      count := JsonRT.custodyCount_goal S h64 hP
      rounds := JsonRT.custodyRounds_goal S h64 hP
      tables := tables_goal H _
      decodes := JsonRT.decodes_goal S h64 hP
      publics := JsonRT.publics_goal S h64 hP
      schedules := schedules_goal H _
      proofCount := proofCount_goal H _
      proofList := JsonRT.proofList_goal S h64 hP
      proofDigests := JsonRT.proofDigests_goal S h64 hP
      repStream := JsonRT.repStream_goal S h64 hP
      repExtras := JsonRT.repExtras_goal S h64 hP
      repDecodes := hD
      repParams := JsonRT.repParams_goal S h64 hP
      zerocheck := zerocheck_goal' H hzc
      lincheck := lincheck_goal' H hzc hlc (lcAlgebra H)
      ringSwitch := ringSwitch_goal' H hzc hlc (lcAlgebra H) hrs (rsAlgebra H)
      ligerito := ligerito_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW hfin
      consumed := consumed_goal_walk H _ (traceR1CS H) hzc hlc (lcAlgebra H) hrs (rsAlgebra H) hD hw hW hfin
      capCommitted := JsonRT.capCommitted_goal S h64 hP }

end FlockComplete.Replay

