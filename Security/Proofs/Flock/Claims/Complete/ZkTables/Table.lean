import Proofs.Flock.Claims.Complete.ZkTables.StmtOne

/-!
# The masked completeness premises, one table at a time

`complete_zk`'s open premises (`RepShapedZk`, `ProverRunsZk`, `LigeritoZk`, `Inner`) and their forms at several tables
(`RepShapedAt`, `ProverRunsAt`, `LgInnerAt`) each say something of one rep of one table: on the table, its trace, its
row of the tape, the rep's domain, and the coins from the rep's first send on. Here they are stated so, once for every
number of tables (`RepShapedT`, `ProverRunsT`, `LigeritoT`, `InnerT`), and the per-case forms follow at each table `i`
of a good case, its rep `r` from the sends after the link's and the `2 i + r` earlier reps' (`repStartZk`).

* `TableT`: a table the masked scope admits (`25 ≤ m ≤ 35`, hm96-SHA-512, regions that fit) and its trace, the
  one-table trace from bit `0` of a witness that satisfies it alone at the deployed relation (`Case.tableT`, from
  `Readable.relS_only`), the table a parsed one (`StmtT`: well formed, its matrices `matA`/`matB`, its ranges
  foldable).
* `RepIn`: the rep's inputs: the table's `--zk` schedule, its masked level-0 commitment of its packed trace under the
  tape's row, its extra claims at link points of its `m_pts`, and its pads' commitment.
* `CoinsT`: the coins a good case gives a rep from `n` sends (`Case.coinsT`): its sends fit the verifier's coins (the
  last rep ends at `1 + rounds`, `repStart_last`), and, from `¬ BadZk`, its zerocheck outer coins are not `1`, its
  level-0 pair coins are independent, and its inner `β` is not `0`.
* `MaskRankT`: the rep's rank check passes at the coins from `n` sends after the table's earlier claim points
  (`Case.maskRankT`, from `¬ BadZk`'s rank slot: rep `1`'s four points, and rep `0`'s first two, have full rank over
  the table's mask words). The points the prover reads off a returning rep's rounds are its points at the coins
  (`repT_points`: each round's coins are its send's, and round `1` holds the zerocheck's `m − 13`, `repPZk_round1`),
  so rep `1`'s prior points are rep `0`'s at the coins.
* `repRoundsT`: a rep's recorded rounds, `repRounds true` at its extra claims (`repRoundsT_eq`). `ProverRunsT` says a
  returning rep records that many, so table `i`'s rep `r` starts at `repStartZk` (`Run.start_eq`).
* `repShapedAt_of`, `proverRunsAt_of`, `lgInnerAt_of`: `RepShapedAt`, `ProverRunsAt` and `LgInnerAt` at every good case
  of the deployed relation, from the table-level forms. `complete_jsonT`: masked completeness at any number of tables
  from the JSON library's specification and the four table-level forms.
* `repShapedZk_of`, `proverRunsZk_of`, `lgInnerZk_of`: `complete_zk`'s `RepShapedZk`, `ProverRunsZk`, `LigeritoZk` and
  `Inner` at the deployed relation, from the table-level forms at table `0` of a good masked case (`CaseZk.toCase`) or
  run (`RunZk.toRun`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay
open FlockSoundness.Discharge.Composed (maskWordsOf)

/-! ## A table, its rep's inputs, and its coins -/

/-- **What a parsed table's circuit gives its statement**: it is well formed, its regions are, its placed matrices
are `matA` and `matB`, and the fold folds each of its ranges (`SlotOk`). -/
def StmtT (st : Flock.Stmt) : Prop :=
  (∃ _ : FlockSoundness.Refine.StmtWF st, FlockSoundness.Refine.RegionsWF st ∧
    ∀ a b : Fin (2 ^ st.c.kLog), FlockLevel3.placedA st a b = FlockSoundness.Refine.matA st a.val b.val ∧
      FlockLevel3.placedB st a b = FlockSoundness.Refine.matB st a.val b.val) ∧
  ∀ r ∈ st.c.ranges.toList, SlotOk st.c r

/-- Each table the verifier's parser reads has `StmtT` (`Readable.tableAt_model`, and `HmRow.check`'s range loop). -/
theorem stmtT_of {r : Readable.Read} {sts : Array Flock.Stmt} (hs : r.tablesOf = some sts)
    (ht : r.tags.typed = false) (hpi : r.publicInputs = none) {j : ℕ} (hj : j < sts.size) : StmtT sts[j] := by
  obtain ⟨hwf, hrw, hmat⟩ := Readable.tableAt_model hs ht hpi hj
  obtain ⟨hc, -⟩ := Readable.tableAt hs ht hpi hj
  obtain ⟨ls, hls⟩ := circuitOf_check ht hc
  exact ⟨⟨hwf, hrw, hmat⟩, hm_check_kinds hls⟩

/-- **A table and its honest trace**: the masked scope admits the table, and the trace is the one-table trace from bit
`0` of a witness that satisfies the table alone at the deployed relation; the table is a parsed one (`StmtT`). -/
def TableT (st : Flock.Stmt) (tr : Trace) : Prop :=
  25 ≤ st.m ∧ st.m ≤ 35 ∧ st.merkle = MerkleScheme.hm96Sha512 ∧ RegionsFit st ∧
    ∃ (p : Readable.Parsed) (w : Claims.Flock.Wit), p.sts = #[st] ∧
      Claims.Flock.RelS (Readable.toReadable Readable.relation p) w ∧ tr = traceOf st w.hid.toArray 0 ∧ StmtT st

/-- **A rep's inputs** at the tape's row `i`: the table's `--zk` schedule `v`, its masked level-0 commitment `cm` of
its packed trace, its extra claims `e` at the points `pts`, each of the table's `m_pts`, and its pads' commitment
`ic`. -/
structure RepIn (H : Claims.Msg → Claims.Msg) (st : Flock.Stmt) (tr : Trace) (ω : TapeZk) (i : ℕ)
    (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ) (ic : InnerCommit) :
    Prop where
  schedule : Zk.schedule (Setup.ofCircuit st).m = .ok v
  commit : commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle
    (v.1.levels.getD 0 default) v.2 (packed tr (Setup.ofCircuit st).m) (ω.extra i) (ω.lanePad i) (ω.salt i 0 0) =
      .ok cm
  points : ∀ a ∈ pts, a.size = st.mPts
  extra : (Setup.ofCircuit st).extra pts = .ok e
  innerCommit : innerCommitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
    (padsOf ω i rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0))
    (ω.innerSalt i rep) = .ok ic

/-- A masked rep's rounds before Ligerito at its extra claims `e`: `piopRoundsZk` at `e`. -/
def piopT (st : Flock.Stmt) (e : Array Claim) : ℕ := (st.m + st.c.kLog - 6) + (2 + e.size + 1)

/-- **A masked rep's recorded rounds** at its extra claims `e`: `repRounds true` there. -/
def repRoundsT (st : Flock.Stmt) (v : Schedule × ℕ) (e : Array Claim) : ℕ :=
  (st.m + (st.c.kLog - 6)) + ((2 + e.size) + 1) + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 4

/-- **The rep's good coins** from `n` sends (`¬ BadZk`'s at a rep starting there): its sends fit the coins, its
zerocheck outer coins (its second round, elements `0` to `m − 14`) are not `1`, its level-0 pair coins (Ligerito's
rounds `k_0 + 2` and `k_0 + 3`) are independent, and its inner `β` is not `0`. -/
def CoinsT (st : Flock.Stmt) (v : Schedule × ℕ) (e : Array Claim) {k : ℕ} (r : Fin k → Coin) (n : ℕ) : Prop :=
  n + repRoundsT st v e < k ∧ (∀ j < st.m - 13, (Claims.coinSeq r (n + 1)).at j ≠ 1) ∧
    ¬ rhoDep r (n + piopT st e + (v.1.k0 + 2)) ∧
    (n + piopT st e + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 2 < k →
      (Claims.coinSeq r (n + piopT st e + (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 2)).at 0 ≠ 0)

/-- **The rep's rank check passes** from `n` sends, after the claim points `prior` of the table's earlier reps: those
points and the rep's own at the coins `r` (`MaskRank.claimsAt`) have full rank over the table's mask words. -/
def MaskRankT (st : Flock.Stmt) {k : ℕ} (r : Fin k → Coin) (n : ℕ) (prior : Array (List F128)) : Prop :=
  MaskRank.maskRankOk (prior ++ MaskRank.claimsAt st r n) (maskWordsOf st) = true

variable (H : Claims.Msg → Claims.Msg) in
/-- **Rep `rep` of a table** on its trace `tr`, at the tape's row `i`, from the empty transcript at its domain `d`,
each ask answered by the coins `r`, its rank check at the earlier claim points `prior` over the words `ws`. -/
def repT {k : ℕ} (r : Fin k → Coin) (st : Flock.Stmt) (tr : Trace) (ω : TapeZk) (i : ℕ) (d : ByteArray)
    (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (rep : ℕ) (ic : InnerCommit) (prior : Array (List F128))
    (ws : List ℕ) : PS (Zk.ZkProof × Transcript) :=
  (repPZk (asksS fun j => (Claims.coinSeq r j).bytes) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2
    rep d tr (packed tr (Setup.ofCircuit st).m) cm e
    (padsOf ω i rep (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0)) ic (ω.extra i)
    (ω.lanePad i) (ω.salt i rep) (ω.nonce i rep) prior ws).run (Transcript.init #[] d)

/-! ## The table-level premises -/

section Props

variable (H : Claims.Msg → Claims.Msg)

/-- **The honest masked rep proof fits its shape**, at a table: on its inputs, at most 2,048 extra claims (a good
case's, `Case.extras_le`), each masked proof the rep returns, at any earlier claim points and words, is `Shaped` at
the shape the masked verifier decodes it at. -/
def RepShapedT : Prop :=
  Claims.Out64 H → ∀ (st : Flock.Stmt) (tr : Trace), TableT st tr → ∀ (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ}
    (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit), RepIn H st tr ω i v cm pts e rep ic → e.size ≤ 2048 → ∀ prior ws,
    Ret (repT H r st tr ω i d v cm e rep ic prior ws) fun pt =>
      FlockSoundness.Discharge.ZkView.Shaped (shapeZk (Setup.ofCircuit st) v.1 e.size) pt.1

/-- **The masked prover's rep returns**, at a table: on its inputs, from any sends after the link's where its coins
are good (`CoinsT`) and its rank check over the table's mask words passes after the earlier claim points `prior`
(`MaskRankT`), and wherever it returns it records `repRoundsT` rounds. -/
def ProverRunsT : Prop :=
  Claims.Out64 H → ∀ (st : Flock.Stmt) (tr : Trace), TableT st tr → ∀ (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ}
    (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit), RepIn H st tr ω i v cm pts e rep ic → rep < 2 →
    (∀ (l : Array Claims.Msg) (prior : Array (List F128)), 0 < l.size → CoinsT st v e r l.size →
      MaskRankT st r l.size prior →
        ∃ pt l', (repT H r st tr ω i d v cm e rep ic prior (maskWordsOf st)).go l = (.ok pt, l')) ∧
    ∀ prior ws, Ret (repT H r st tr ω i d v cm e rep ic prior ws) fun pt => pt.2.rounds.size = repRoundsT st v e

/-- **The masked Ligerito replays**, at a table: where the rep returned its masked proof `p` from sends at which its
coins are good, and the verifier, on rounds that carry the prover's, replayed it (`q`) and its ring switch (`sw`),
the verifier's masked Ligerito returns. -/
def LigeritoT : Prop :=
  Claims.Out64 H → ∀ (st : Flock.Stmt) (tr : Trace), TableT st tr → ∀ (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ}
    (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit), RepIn H st tr ω i v cm pts e rep ic → rep < 2 →
    ∀ (prior : Array (List F128)) (ws : List ℕ) (l : Array Claims.Msg) (p : Zk.ZkProof) (t' : Transcript)
      (l' : Array Claims.Msg), 0 < l.size →
      CoinsT st v e r l.size → (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l') →
    ∀ R : Array Round, Agrees R t'.rounds 0 →
    ∀ q, (Zk.replayWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st)
        (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0) p.proof p.inner.root).run
        (Transcript.init R d) = .ok q →
    ∀ sw, (ringSwitchWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).m
        (ringClaimsZk q.1 p.proof ++ e) p.proof.ringSwitches).run q.2 = .ok sw →
    ∃ lg, (Zk.ligeritoWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle v.1 sw.1.1
      p.proof.cap p.proof.ligerito (basis sw.1.2) ⟨v.2, rep, p.level0⟩).run sw.2 = .ok lg

/-- **The inner proof passes**, at a table: as `LigeritoT`, and where the masked Ligerito returned `lg` and level 0's
lane phase gave its constraints, the inner proof passes on every constraint. -/
def InnerT : Prop :=
  Claims.Out64 H → ∀ (st : Flock.Stmt) (tr : Trace), TableT st tr → ∀ (ω : TapeZk) (i : ℕ) (d : ByteArray) {k : ℕ}
    (r : Fin k → Coin) (v : Schedule × ℕ) (cm : Commit) (pts : Array (Array F128)) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit), RepIn H st tr ω i v cm pts e rep ic → rep < 2 →
    ∀ (prior : Array (List F128)) (ws : List ℕ) (l : Array Claims.Msg) (p : Zk.ZkProof) (t' : Transcript)
      (l' : Array Claims.Msg), 0 < l.size →
      CoinsT st v e r l.size → (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l') →
    ∀ R : Array Round, Agrees R t'.rounds 0 →
    ∀ q, (Zk.replayWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st)
        (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0) p.proof p.inner.root).run
        (Transcript.init R d) = .ok q →
    ∀ sw, (ringSwitchWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).m
        (ringClaimsZk q.1 p.proof ++ e) p.proof.ringSwitches).run q.2 = .ok sw →
    ∀ lg, (Zk.ligeritoWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle v.1 sw.1.1
      p.proof.cap p.proof.ligerito (basis sw.1.2) ⟨v.2, rep, p.level0⟩).run sw.2 = .ok lg →
    ∀ lanes, Zk.ligeritoConstraints (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0) lg.1
      p.proof.ligerito p.level0 = .ok lanes →
    ∃ u, (Zk.verifyInnerWith (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      (Zk.PadLayout.new (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog v.1.k0) p.inner p.proof.zerocheck.finalC
      (repConsZk q.1 p.proof ++ lanes)).run lg.2 = .ok u

end Props

/-! ## Each table of a good case -/

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **Each table of a good case of the deployed relation and its trace** (`TableT`): table `i`'s share of the witness
satisfies table `i` alone (`Readable.relS_only`), and its trace is that share's from bit `0` (`traces_getElem`). -/
theorem Case.tableT (c : Case H Readable.relation) {i : ℕ} (hi : i < c.sts.size) :
    TableT c.sts[i]! (traces c.sts c.w)[i]! := by
  obtain ⟨-, -, -, hall⟩ := c.tables
  obtain ⟨h1, h2, h3⟩ := hall i hi
  obtain ⟨p, hps, hty, hpi, hpe, hR⟩ := c.parsed
  have hJ := (Readable.relS_only hps hty hpi c.w).1 hR i (by rw [hpe]; exact hi)
  rw [hpe] at hJ
  have hone : (p.only i).sts = #[c.sts[i]] := by
    show #[p.sts[i]!] = _
    rw [hpe, getElem!_pos c.sts i hi]
  have htab := Readable.tablesOf_of_parses hps
  rw [hpe] at htab
  rw [traces_getElem c.sts c.w hi, getElem!_pos c.sts i hi]
  exact ⟨h1, h2, h3, c.ready_parts.2.2.1 _ (Array.getElem_mem hi), p.only i, _, hone, hJ, rfl,
    stmtT_of htab hty hpi hi⟩

/-- The link's points have each table's `m_pts`. -/
theorem Case.points_mPts (c : Case H rel) {i : ℕ} (hi : i < c.sts.size) :
    ∀ a ∈ pointsOf c.spec (Claims.coinSeq c.r 0), a.size = c.sts[i]!.mPts := by
  intro a ha
  obtain ⟨j, -, rfl⟩ := Array.mem_map.1 ha
  rw [getElem!_pos c.sts i hi, c.mPts hi]
  simp

/-- A table's extra claims at the link's points: `points · |regions|` of them. -/
theorem extras_size {spec : SessionSpec} {st : Flock.Stmt} {c : Coin} {e : Array Claim}
    (he : (Setup.ofCircuit st).extra (pointsOf spec c) = .ok e) : e.size = spec.points * st.regions.size := by
  rw [FlockDeploy.extraClaims_size (st := st) he]
  simp [pointsOf, Nat.mul_comm]

/-- **A good case's table has at most 2,048 extra claims** at the link's points (`Case.points_regions`). -/
theorem Case.extras_le (c : Case H rel) {st : Flock.Stmt} (hst : st ∈ c.sts) {e : Array Claim}
    (he : (Setup.ofCircuit st).extra (pointsOf c.spec (Claims.coinSeq c.r 0)) = .ok e) : e.size ≤ 2048 := by
  rw [extras_size he, c.points_regions.1]
  have := c.points_regions.2 st hst
  omega

theorem repRoundsT_eq {spec : SessionSpec} {st : Flock.Stmt} {v : Schedule × ℕ} {e : Array Claim}
    (hv : Zk.schedule st.m = .ok v) (he : e.size = spec.points * st.regions.size) :
    repRoundsT st v e = repRounds true spec st := by
  rcases v with ⟨sch, tp⟩
  simp [repRoundsT, repRounds, hv, he]

theorem repStart_succ (spec : SessionSpec) (sts : Array Flock.Stmt) (q : ℕ) :
    repStartZk spec sts (q + 1) = repStartZk spec sts q + repRounds true spec sts[q / 2]! := by
  unfold repStartZk
  rw [List.range_succ, List.map_append, List.sum_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  omega

theorem repStart_mono (spec : SessionSpec) (sts : Array Flock.Stmt) {a b : ℕ} (h : a ≤ b) :
    repStartZk spec sts a ≤ repStartZk spec sts b := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [repStart_succ]; omega

/-- The last rep ends at the session's recorded rounds, after the link's send. -/
theorem repStart_last (spec : SessionSpec) (sts : Array Flock.Stmt) :
    repStartZk spec sts (2 * sts.size) = 1 + rounds true spec sts := by
  have h : ∀ n, repStartZk spec sts (2 * n) =
      1 + 2 * ((List.range n).map fun i => repRounds true spec sts[i]!).sum := by
    intro n
    induction n with
    | zero => simp [repStartZk]
    | succ n ih =>
      rw [show 2 * (n + 1) = 2 * n + 1 + 1 by omega, repStart_succ, repStart_succ, ih, List.range_succ,
        List.map_append, List.sum_append, show (2 * n + 1) / 2 = n by omega, show 2 * n / 2 = n by omega]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
      omega
  rw [h, rounds_eq]

/-- **Table `q / 2`'s rep `q % 2` of a good case has good coins** (`CoinsT`) from `repStartZk` sends: its sends end
by the last rep's, inside the verifier's coins, and its slots are `BadZk`'s. -/
theorem Case.coinsT (c : Case H rel) {q : ℕ} (hq : q < 2 * c.sts.size) {v : Schedule × ℕ}
    (hv : Zk.schedule c.sts[q / 2]!.m = .ok v) {e : Array Claim}
    (he : e.size = c.spec.points * c.sts[q / 2]!.regions.size) :
    CoinsT c.sts[q / 2]! v e c.r (repStartZk c.spec c.sts q) := by
  rcases v with ⟨sch, tp⟩
  have hp : piopT c.sts[q / 2]! e = piopRoundsZk c.spec c.sts[q / 2]! := by
    simp only [piopT, piopRoundsZk, he]
  have hk : (V true).k c.x = 2 + rounds true c.spec c.sts := by
    show (match setupOf true c.x with
      | some (_, spec, sts) => 2 + rounds true spec sts
      | none => 1) = _
    rw [c.setup]
  have hfit : repStartZk c.spec c.sts q + repRoundsT c.sts[q / 2]! (sch, tp) e < (V true).k c.x := by
    rw [repRoundsT_eq hv he, ← repStart_succ, hk]
    have h1 := repStart_mono c.spec c.sts (show q + 1 ≤ 2 * c.sts.size by omega)
    have h2 := repStart_last c.spec c.sts
    omega
  refine ⟨hfit, fun j hj h1 => c.good (Or.inl ⟨(repStartZk c.spec c.sts q + 1, j), ?_, h1⟩),
    fun hd => c.good (Or.inr (Or.inr (Or.inl ⟨_, ?_, hd⟩))), fun hb h0 => c.good (Or.inr (Or.inl ⟨_, ?_, hb, h0⟩))⟩
  · unfold outerSlotsZk
    rw [c.setup]
    simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
    exact ⟨q, hq, j, hj, rfl, rfl⟩
  · unfold rhoSlotsZk
    rw [c.setup]
    simp only [List.mem_flatMap, List.mem_range]
    refine ⟨q, hq, ?_⟩
    rw [hv]
    show _ ∈ [_]
    rw [hp]
    exact List.mem_singleton_self _
  · unfold betaSlotsZk
    rw [c.setup]
    simp only [List.mem_flatMap, List.mem_range]
    refine ⟨q, hq, ?_⟩
    rw [hv]
    show _ ∈ [_]
    rw [hp]
    exact List.mem_singleton_self _

theorem repRoundsT_ge (st : Flock.Stmt) (v : Schedule × ℕ) (e : Array Claim) :
    st.m + st.c.kLog ≤ repRoundsT st v e := by
  unfold repRoundsT
  omega

/-- The claim points the session's `q`th rep checks ahead of its own, at the case's coins: at a table's rep `1`, its
rep `0`'s. -/
def Case.priorT (c : Case H rel) (q : ℕ) : Array (List F128) :=
  if q % 2 = 1 then MaskRank.claimsAt c.sts[q / 2]! c.r (repStartZk c.spec c.sts (q - 1)) else #[]

/-- **Table `q / 2`'s rep `q % 2` of a good case passes its rank check** (`MaskRankT`) from `repStartZk` sends after
its earlier claim points (`Case.priorT`): the table's rank slot is `BadZk`'s, inside the verifier's coins, and rep
`0`'s check reads the first two of rep `1`'s four points (`maskRankOk_append_left`). -/
theorem Case.maskRankT (c : Case H rel) {q : ℕ} (hq : q < 2 * c.sts.size) {v : Schedule × ℕ}
    (hv : Zk.schedule c.sts[q / 2]!.m = .ok v) {e : Array Claim}
    (he : e.size = c.spec.points * c.sts[q / 2]!.regions.size) :
    MaskRankT c.sts[q / 2]! c.r (repStartZk c.spec c.sts q) (c.priorT q) := by
  have hk : (V true).k c.x = 2 + rounds true c.spec c.sts := by
    show (match setupOf true c.x with
      | some (_, spec, sts) => 2 + rounds true spec sts
      | none => 1) = _
    rw [c.setup]
  have hAt : MaskRank.MaskRankAt c.sts[q / 2]! c.r (repStartZk c.spec c.sts (2 * (q / 2)))
      (repStartZk c.spec c.sts (2 * (q / 2) + 1)) := by
    by_contra h
    refine c.good (Or.inr (Or.inr (Or.inr ⟨(c.sts[q / 2]!, repStartZk c.spec c.sts (2 * (q / 2)),
      repStartZk c.spec c.sts (2 * (q / 2) + 1)), ?_, ?_, h⟩)))
    · unfold rankSlotsZk
      rw [c.setup]
      simp only [List.mem_flatMap, List.mem_range]
      refine ⟨q / 2, by omega, ?_⟩
      rw [hv]
      exact List.mem_singleton_self _
    · have h1 := repRoundsT_ge c.sts[q / 2]! v e
      rw [repRoundsT_eq hv he] at h1
      have h2 := repStart_succ c.spec c.sts (2 * (q / 2) + 1)
      rw [show (2 * (q / 2) + 1) / 2 = q / 2 by omega] at h2
      have h3 := repStart_mono c.spec c.sts (show 2 * (q / 2) + 1 + 1 ≤ 2 * c.sts.size by omega)
      have h4 := repStart_last c.spec c.sts
      dsimp only
      omega
  unfold MaskRankT Case.priorT
  rcases Nat.mod_two_eq_zero_or_one q with h0 | h1
  · have hn : ¬ q % 2 = 1 := by omega
    simp only [hn, ↓reduceIte, Array.empty_append]
    have hq0 : 2 * (q / 2) = q := by omega
    rw [hq0] at hAt
    exact MaskRank.maskRankOk_append_left hAt
  · simp only [h1, ↓reduceIte]
    have hq1 : 2 * (q / 2) + 1 = q := by omega
    have hq1' : 2 * (q / 2) = q - 1 := by omega
    rw [hq1, hq1'] at hAt
    exact hAt

/-- **A returning masked rep's round `1` is its zerocheck's outer coins**, `m − 13` of them (`repPZk_rs`). -/
theorem repPZk_round1 (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m)
    {l : Array Claims.Msg} {P : Zk.ZkProof} {t' : Transcript} {l' : Array Claims.Msg}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run (Transcript.init #[] dom)).go l = (.ok (P, t'), l')) :
    t'.rounds[1]!.n = st.m - 13 := by
  obtain ⟨full, -, hpre⟩ := repPZk_rs resp hsh st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt nonce
    prior words hm _ l h
  have hp : Prefix ((ZcZ.t₂ st.digest (flat cm0.cap) ic.root (Zc.coinsAt resp l.size)
      (Transcript.init #[] dom)).rounds.push (sent (ZcZ.t₂ st.digest (flat cm0.cap) ic.root (Zc.coinsAt resp l.size)
        (Transcript.init #[] dom)) 2 (st.m - 13) (Zc.coinsAt resp l.size).c1))
      (ZcZ.tEnd (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m st.digest (flat cm0.cap) ic.root
        (Zc.coinsAt resp l.size) tr (Transcript.init #[] dom)).rounds :=
    (Prefix.push _ _).trans (ZcZ.prefix_loop (i := 0) _ (Nat.zero_le _))
  obtain ⟨C, hC⟩ := hp.trans ((LcZ.prefix_start _).trans ((Rs.prefix_tEnd _).trans hpre))
  have hs : (ZcZ.t₂ st.digest (flat cm0.cap) ic.root (Zc.coinsAt resp l.size)
      (Transcript.init #[] dom)).rounds.size = 1 := by
    simp [ZcZ.t₂, ZcZ.t₁, afterSq, Transcript.init]
  rw [hC, getElem!_pos _ 1 (by simp [hs]; omega), Array.getElem_append_left (by simp [hs])]
  simp [Array.getElem_push, hs, sent]

theorem claimPoints_congr {m kLog : ℕ} {f g : ℕ → ℕ → F128} (h0 : ∀ j < m + kLog, f j 0 = g j 0)
    (h1 : ∀ e < m - 13, f 1 e = g 1 e) : MaskRank.claimPoints m kLog f = MaskRank.claimPoints m kLog g := by
  unfold MaskRank.claimPoints
  have hr : (Array.range (m - 6)).map (fun j => f (3 + j) 0) = (Array.range (m - 6)).map fun j => g (3 + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have ht : (Array.range (kLog - 6)).map (fun j => f (5 + (m - 6) + j) 0) =
      (Array.range (kLog - 6)).map fun j => g (5 + (m - 6) + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have hc : (Array.range (m - 13)).map (f 1) = (Array.range (m - 13)).map (g 1) :=
    Array.map_congr_left fun e he => h1 e (by rw [Array.mem_range] at he; exact he)
  simp only [hr, ht, hc]

theorem Recs.getElem {resp : ℕ → Claims.Msg} :
    ∀ {i : ℕ} {rs : List Round} {bs : List Claims.Msg}, Recs resp i rs bs →
      ∀ j (hj : j < rs.length), ∃ b, RoundIs resp (i + j) b rs[j]
  | _, [], [], _, _, hj => absurd hj (by simp)
  | _, _ :: _, b :: _, h, 0, _ => ⟨b, h.1⟩
  | i, _ :: rs, _ :: _, h, j + 1, hj => by
    obtain ⟨b, hb⟩ := Recs.getElem h.2 j (by simpa using hj)
    exact ⟨b, by rw [show i + (j + 1) = i + 1 + j by omega]; exact hb⟩
  | _, [], _ :: _, h, _, _ => h.elim
  | _, _ :: _, [], h, _, _ => h.elim

/-- **A returning rep's claim points read off its rounds are its claim points at the coins** (`MaskRank.claimsAt`,
from its first send `l`): each recorded round's coins are its send's (`repPZk_rec`), its round `1` holds the
zerocheck's `m − 13` outer coins (`repPZk_round1`), and the points read rounds before `m + k_log`. -/
theorem repT_points (H : Claims.Msg → Claims.Msg) {k : ℕ} (r : Fin k → Coin) (st : Flock.Stmt) (tr : Trace)
    (ω : TapeZk) (i : ℕ) (d : ByteArray) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit) (prior : Array (List F128)) (ws : List ℕ) {l l' : Array Claims.Msg} {p : Zk.ZkProof}
    {t' : Transcript} (hm : 14 ≤ st.m)
    (hgo : (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l'))
    (hsz : st.m + st.c.kLog ≤ t'.rounds.size) :
    MaskRank.roundPoints st.m st.c.kLog t'.rounds = MaskRank.claimsAt st r l.size := by
  unfold repT at hgo
  have hm' : 14 ≤ (Setup.ofCircuit st).m := hm
  have h1 : t'.rounds[1]!.n = st.m - 13 := repPZk_round1 (hm := hm') (h := hgo)
  obtain ⟨rs, bs, hrs, -, hrec⟩ := repPZk_rec (fun j => (Claims.coinSeq r j).bytes) (Claims.hOf H) _ _ _ _ _ _ _
    _ _ _ _ _ _ _ _ _ _ _ l p t' l' hgo
  have hR : t'.rounds.toList = rs := by simpa [Transcript.init] using hrs
  have hround : ∀ j (hj : j < t'.rounds.size), ∃ b, RoundIs (fun j => (Claims.coinSeq r j).bytes) (l.size + j) b
      t'.rounds[j]! := by
    intro j hj
    have hj' : j < rs.length := by rw [← hR]; simpa using hj
    obtain ⟨b, hb⟩ := Recs.getElem hrec j hj'
    refine ⟨b, ?_⟩
    rw [getElem!_pos t'.rounds j hj]
    simpa only [← hR, Array.getElem_toList] using hb
  have key : ∀ j (hj : j < t'.rounds.size) a, a < t'.rounds[j]!.n →
      (t'.rounds[j]!.coins).getD a 0 = (Claims.coinSeq r (l.size + j)).at a := by
    intro j hj a ha
    obtain ⟨b, c, -, -, -, hc⟩ := hround j hj
    rw [hc, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
    simpa [ha] using coinOf_bytes (Claims.coinSeq r (l.size + j)) a
  unfold MaskRank.roundPoints MaskRank.claimsAt
  refine claimPoints_congr (fun j hj => ?_) (fun a ha => ?_)
  · have hj' : j < t'.rounds.size := by omega
    obtain ⟨b, c, -, -, hn, -⟩ := hround j hj'
    exact key j hj' 0 hn
  · exact key 1 (by omega) a (by omega)

/-- **A returning rep's sends grow by its recorded rounds.** -/
theorem repT_size (H : Claims.Msg → Claims.Msg) {k : ℕ} (r : Fin k → Coin) (st : Flock.Stmt) (tr : Trace)
    (ω : TapeZk) (i : ℕ) (d : ByteArray) (v : Schedule × ℕ) (cm : Commit) (e : Array Claim) (rep : ℕ)
    (ic : InnerCommit) (prior : Array (List F128)) (ws : List ℕ) {l l' : Array Claims.Msg} {p : Zk.ZkProof}
    {t' : Transcript} (hgo : (repT H r st tr ω i d v cm e rep ic prior ws).go l = (.ok (p, t'), l')) :
    l'.size = l.size + t'.rounds.size := by
  unfold repT at hgo
  obtain ⟨rs, bs, hrs, hl', hrec⟩ := repPZk_rec (fun j => (Claims.coinSeq r j).bytes) (Claims.hOf H) _ _ _ _ _ _ _
    _ _ _ _ _ _ _ _ _ _ _ l p t' l' hgo
  have h1 := Recs.length_eq hrec
  have h2 : t'.rounds.size = rs.length := by
    rw [← Array.length_toList, hrs]
    simp [Transcript.init]
  rw [← Array.length_toList, hl', List.length_append, Array.length_toList, h2, h1]

/-! ## The per-case premises from the table-level ones -/

/-- **`RepShapedAt` at every good case** of the deployed relation, from `RepShapedT` at each table. -/
theorem repShapedAt_of (hR : RepShapedT H) (h64 : Claims.Out64 H) (c : Case H Readable.relation) :
    RepShapedAt c := fun i hi v cm e hv hcm hex rep ic hic prior =>
  hR h64 _ _ (c.tableT hi) c.ω i (c.spec.domain c.spec.tables[i]! rep).toUTF8 c.r v cm _ e rep ic
    ⟨hv, hcm, c.points_mPts hi, hex, hic⟩ (c.extras_le (getElem!_pos c.sts i hi ▸ Array.getElem_mem hi) hex) prior _

/-- **`ProverRunsAt` at every good case** of the deployed relation, from `ProverRunsT` at each table: table `q / 2`'s
rep `q % 2` runs from `repStartZk` sends, where `¬ BadZk` gives its coins (`Case.coinsT`) and its rank check after
its rep `0`'s claim points at the coins (`Case.maskRankT`), which are the ones read off rep `0`'s rounds
(`repT_points`), and ends `repRounds` sends later, where the next rep starts. -/
theorem proverRunsAt_of (hPR : ProverRunsT H) (h64 : Claims.Out64 H) (c : Case H Readable.relation) :
    ProverRunsAt c := by
  intro v cm e ic hin l₀ hl₀
  have hstep : ∀ q < 2 * c.sts.size, ∀ l : Array Claims.Msg, l.size = repStartZk c.spec c.sts q →
      ∃ x : (Zk.ZkProof × Transcript) × Array Claims.Msg,
        (repRunJ H Readable.relation c (q / 2) (v (q / 2)) (cm (q / 2)) (e (q / 2)) (q % 2) (ic (q / 2) (q % 2))
          (c.priorT q)).go l = (.ok x.1, x.2) ∧ x.2.size = repStartZk c.spec c.sts (q + 1) ∧
        MaskRank.roundPoints c.sts[q / 2]!.m c.sts[q / 2]!.c.kLog x.1.2.rounds =
          MaskRank.claimsAt c.sts[q / 2]! c.r (repStartZk c.spec c.sts q) := by
    intro q hq l hl
    have hi : q / 2 < c.sts.size := by omega
    obtain ⟨hv, hcm, hex, hic⟩ := hin (q / 2) hi
    have hx := extras_size hex
    have hT := c.tableT hi
    obtain ⟨hrun, hcount⟩ := hPR h64 _ _ hT c.ω (q / 2)
      (c.spec.domain c.spec.tables[q / 2]! (q % 2)).toUTF8 c.r (v (q / 2)) (cm (q / 2)) _ (e (q / 2)) (q % 2)
      (ic (q / 2) (q % 2)) ⟨hv, hcm, c.points_mPts hi, hex, hic (q % 2)⟩ (Nat.mod_lt _ (by norm_num))
    have hpos : 0 < l.size := by rw [hl]; unfold repStartZk; omega
    obtain ⟨⟨p, t'⟩, l', hgo⟩ := hrun l (c.priorT q) hpos (by rw [hl]; exact c.coinsT hq hv hx)
      (by rw [hl]; exact c.maskRankT hq hv hx)
    have hc : t'.rounds.size = repRoundsT c.sts[q / 2]! (v (q / 2)) (e (q / 2)) := hcount _ _ l _ l' hgo
    refine ⟨((p, t'), l'), hgo, ?_, ?_⟩
    · rw [repT_size H c.r _ _ _ _ _ _ _ _ _ _ _ _ hgo, hc, hl, repStart_succ,
        repRoundsT_eq (st := c.sts[q / 2]!) hv hx]
    · rw [← hl]
      exact repT_points H c.r _ _ _ _ _ _ _ _ _ _ _ _ (by have := hT.1; omega) hgo
        (by rw [hc]; exact repRoundsT_ge _ _ _)
  classical
  let g : ℕ → Array Claims.Msg → (Zk.ZkProof × Transcript) × Array Claims.Msg := fun q l =>
    if h : q < 2 * c.sts.size ∧ l.size = repStartZk c.spec c.sts q then Classical.choose (hstep q h.1 l h.2)
    else (default, l)
  have hg : ∀ q l (h : q < 2 * c.sts.size ∧ l.size = repStartZk c.spec c.sts q),
      g q l = Classical.choose (hstep q h.1 l h.2) := fun q l h => dite_eq_left h
  let ls : ℕ → Array Claims.Msg := fun q => Nat.rec (motive := fun _ => Array Claims.Msg) l₀ (fun q l => (g q l).2) q
  have hls : ∀ q, q ≤ 2 * c.sts.size → (ls q).size = repStartZk c.spec c.sts q := by
    intro q
    induction q with
    | zero => intro _; show l₀.size = _; rw [hl₀]; simp [repStartZk]
    | succ q ih =>
      intro hq
      have h : q < 2 * c.sts.size ∧ (ls q).size = repStartZk c.spec c.sts q := ⟨by omega, ih (by omega)⟩
      show (g q (ls q)).2.size = _
      rw [hg q (ls q) h]
      exact (Classical.choose_spec (hstep q h.1 (ls q) h.2)).2.1
  let pts : ℕ → Zk.ZkProof × Transcript := fun q => (g q (ls q)).1
  have hpts : ∀ q < 2 * c.sts.size,
      (repRunJ H Readable.relation c (q / 2) (v (q / 2)) (cm (q / 2)) (e (q / 2)) (q % 2) (ic (q / 2) (q % 2))
        (c.priorT q)).go (ls q) = (.ok (pts q), ls (q + 1)) ∧
      MaskRank.roundPoints c.sts[q / 2]!.m c.sts[q / 2]!.c.kLog (pts q).2.rounds =
        MaskRank.claimsAt c.sts[q / 2]! c.r (repStartZk c.spec c.sts q) := by
    intro q hq
    have h : q < 2 * c.sts.size ∧ (ls q).size = repStartZk c.spec c.sts q := ⟨hq, hls q (by omega)⟩
    have e1 : g q (ls q) = Classical.choose (hstep q h.1 (ls q) h.2) := hg q (ls q) h
    have hs := Classical.choose_spec (hstep q h.1 (ls q) h.2)
    rw [← e1] at hs
    exact ⟨hs.1, hs.2.2⟩
  refine ⟨ls, pts, rfl, fun q hq => ?_⟩
  have hprior : priorJ c pts q = c.priorT q := by
    unfold priorJ Case.priorT
    by_cases h1 : q % 2 = 1
    · simp only [h1, ↓reduceIte]
      have h2 := (hpts (q - 1) (by omega)).2
      rw [show (q - 1) / 2 = q / 2 by omega] at h2
      exact h2
    · simp only [h1, ↓reduceIte]
  rw [hprior]
  exact (hpts q hq).1

theorem take_flatMap_length {α : Type} (xs : Array (Array α)) :
    ∀ q, q ≤ xs.size → ((xs.toList.take q).flatMap Array.toList).length = ((List.range q).map fun j => xs[j]!.size).sum
  | 0, _ => by simp
  | q + 1, hq => by
    have hq' : q < xs.toList.length := by simpa using hq
    rw [List.take_add_one, List.flatMap_append, List.length_append, take_flatMap_length xs q (by omega),
      List.range_succ, List.map_append, List.sum_append, List.getElem?_eq_getElem hq']
    simp [getElem!_pos xs q (by omega)]

/-- **Each rep of a run, as a table-level rep**: table `i`'s rep `r`'s recorded rounds are the `2 i + r`th stream's
as the decoder reads them back, its decoded masked proof is the one it returned, from `1 +` the earlier streams' rounds
sends, on its inputs (`RepIn`). -/
theorem Run.repIn (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hR : RepShapedT H) (o : Run H Readable.relation) :
    ∃ (schs : Array (Schedule × ℕ)) (cms : Array Commit) (streams : Array (Array Round)),
      streams.size = 2 * o.sts.size ∧ (∀ rs ∈ streams, ∀ x ∈ rs, ∃ c, x.bytes = some c) ∧
      ∀ i < o.sts.size, o.tabs[i]! = Setup.ofCircuit o.sts[i]! ∧ Zk.schedule o.sts[i]!.m = .ok schs[i]! ∧
        o.sch i = schs[i]!.1 ∧ ∀ r < 2, ∃ p ic l t' l' prior ws,
          (o.stream i r).rounds = streams[2 * i + r]!.map (roundRec o.spec (Claims.hOf H)) ∧ o.zp i r = p ∧
          streams[2 * i + r]! = t'.rounds ∧
          l.size = 1 + ((streams.toList.take (2 * i + r)).flatMap Array.toList).length ∧
          RepIn H o.sts[i]! (traces o.sts o.w)[i]! o.ω i schs[i]! cms[i]! (pointsOf o.spec (Claims.coinSeq o.r 0))
            (o.extras i) r ic ∧
          (repT H o.r o.sts[i]! (traces o.sts o.w)[i]! o.ω i (o.spec.domain o.spec.tables[i]! r).toUTF8 schs[i]! cms[i]!
            (o.extras i) r ic prior ws).go l = (.ok (p, t'), l') := by
  have hS := stmtAt_all S o.toCase
  obtain ⟨schs, cms, streams, hss, hbytes, -, hpts, hrep⟩ := Run.walkJ S h64 o hS (repShapedAt_of hR h64 o.toCase)
  have hJ := recordJsonJ S h64 o hS
  refine ⟨schs, cms, streams, hss, hbytes, fun i hi => ?_⟩
  obtain ⟨htabi, hv, hsch, hreps⟩ := hrep i hi
  refine ⟨htabi, hv, hsch, fun r hr => ?_⟩
  obtain ⟨hstream, p, ic, l, t', l', prior, ws, hic, hzp, hst', hl, hcm, hrun⟩ := hreps r hr
  have hi' : i < o.tabs.size := by simpa [Run.tabs, tabsOf_size] using hi
  have hex := hJ.repExtras i hi'
  rw [htabi, hpts] at hex
  exact ⟨p, ic, l, t', l', prior, ws, hstream, hzp, hst', hl, ⟨hv, hcm, o.toCase.points_mPts hi, hex, hic⟩, hrun⟩

/-- **Table `i`'s rep `r` of a run starts at `repStartZk`**: every earlier rep recorded `repRounds` rounds
(`ProverRunsT`). -/
theorem Run.start_eq (hPR : ProverRunsT H) (h64 : Claims.Out64 H) {o : Run H Readable.relation}
    {schs : Array (Schedule × ℕ)} {cms : Array Commit} {streams : Array (Array Round)}
    (hss : streams.size = 2 * o.sts.size)
    (hrep : ∀ i < o.sts.size, o.tabs[i]! = Setup.ofCircuit o.sts[i]! ∧ Zk.schedule o.sts[i]!.m = .ok schs[i]! ∧
        o.sch i = schs[i]!.1 ∧ ∀ r < 2, ∃ p ic l t' l' prior ws,
          (o.stream i r).rounds = streams[2 * i + r]!.map (roundRec o.spec (Claims.hOf H)) ∧ o.zp i r = p ∧
          streams[2 * i + r]! = t'.rounds ∧
          l.size = 1 + ((streams.toList.take (2 * i + r)).flatMap Array.toList).length ∧
          RepIn H o.sts[i]! (traces o.sts o.w)[i]! o.ω i schs[i]! cms[i]! (pointsOf o.spec (Claims.coinSeq o.r 0))
            (o.extras i) r ic ∧
          (repT H o.r o.sts[i]! (traces o.sts o.w)[i]! o.ω i (o.spec.domain o.spec.tables[i]! r).toUTF8 schs[i]! cms[i]!
            (o.extras i) r ic prior ws).go l = (.ok (p, t'), l'))
    {q : ℕ} (hq : q ≤ 2 * o.sts.size) :
    1 + ((streams.toList.take q).flatMap Array.toList).length = repStartZk o.spec o.sts q := by
  have hcnt : ∀ j < 2 * o.sts.size, streams[j]!.size = repRounds true o.spec o.sts[j / 2]! := by
    intro j hj
    have hi : j / 2 < o.sts.size := by omega
    obtain ⟨-, hv, -, hreps⟩ := hrep (j / 2) hi
    obtain ⟨p, ic, l, t', l', prior, ws, -, -, hst', -, hin, hrun⟩ := hreps (j % 2) (Nat.mod_lt _ (by norm_num))
    have hc : t'.rounds.size = repRoundsT o.sts[j / 2]! schs[j / 2]! (o.extras (j / 2)) :=
      (hPR h64 _ _ (o.toCase.tableT hi) _ _ _ _ _ _ _ _ _ _ hin (Nat.mod_lt _ (by norm_num))).2 prior ws l _ l' hrun
    rw [show 2 * (j / 2) + j % 2 = j by omega] at hst'
    rw [hst', hc, repRoundsT_eq hv (extras_size hin.extra)]
  rw [take_flatMap_length streams q (by omega)]
  unfold repStartZk
  congr 1
  refine congrArg List.sum (List.map_congr_left fun j hj => ?_)
  exact hcnt j (by have := List.mem_range.1 hj; omega)

/-- **`LgInnerAt` at every run** of the deployed relation, from `LigeritoT` and `InnerT` at each table: each rep's
inputs and run are the walk's (`Run.repIn`), from `repStartZk` sends (`Run.start_eq`), where `¬ BadZk` gives its coins
(`Case.coinsT`); its recorded rounds carry the prover's, the masked replay (`replayJ`) and the ring switch
(`ringSwitchJ`) return, and level 0's lanes are its constraints (`lanesJ`). -/
theorem lgInnerAt_of (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hR : RepShapedT H) (hPR : ProverRunsT H)
    (hLg : LigeritoT H) (hIn : InnerT H) (o : Run H Readable.relation) : LgInnerAt H Readable.relation o := by
  have hS := stmtAt_all S o.toCase
  have hRA := repShapedAt_of hR h64 o.toCase
  obtain ⟨schs, cms, streams, hss, hbytes, hrep⟩ := Run.repIn S h64 hR o
  have key : ∀ i < o.tabs.size, ∀ r < 2,
      (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
        (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r) ∧
      (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
        (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r) := by
    intro i hi r hr
    have hi' : i < o.sts.size := Run.lt_sts hi
    obtain ⟨htabi, hv, hsch, hreps⟩ := hrep i hi'
    obtain ⟨p, ic, l, t', l', prior, ws, hstream, hzp, hst', hl, hin, hrun⟩ := hreps r hr
    have hq : 2 * i + r < 2 * o.sts.size := by omega
    have hstart := Run.start_eq hPR h64 hss hrep (q := 2 * i + r) (by omega)
    rw [← hl] at hstart
    have hqi : (2 * i + r) / 2 = i := by omega
    have hcoins : CoinsT o.sts[i]! schs[i]! (o.extras i) o.r l.size := by
      have := o.toCase.coinsT (q := 2 * i + r) hq (v := schs[i]!) (by rw [hqi]; exact hv) (e := o.extras i)
        (by rw [hqi]; exact extras_size hin.extra)
      rw [hqi, ← hstart] at this
      exact this
    have hpos : 0 < l.size := by rw [hl]; omega
    have hk : 2 * i + r < streams.size := by omega
    have hAg : Agrees (o.stream i r).rounds t'.rounds 0 := by
      intro j hj _
      have hmem : t'.rounds[j] ∈ streams[2 * i + r]! := by
        rw [hst']
        exact Array.getElem_mem hj
      obtain ⟨c, hc⟩ := hbytes _ (by rw [getElem!_pos streams _ hk]; exact Array.getElem_mem _) _ hmem
      refine ⟨roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [roundRec, hc]⟩
      rw [hstream, hst']
      simp [hj]
    have hlay : o.lay i = Zk.PadLayout.new (Setup.ofCircuit o.sts[i]!).m (Setup.ofCircuit o.sts[i]!).kLog
        schs[i]!.1.k0 := by
      unfold Run.lay
      rw [htabi, hsch]
    have hv' : Zk.schedule (Setup.ofCircuit o.sts[i]!).m = .ok schs[i]! := hv
    have htp : o.tPad i = schs[i]!.2 := by
      unfold Run.tPad zkSchedsOf
      rw [getElem!_pos _ i (by simpa using hi), Array.getElem_map, ← getElem!_pos o.tabs i hi, htabi]
      unfold schedZkOf
      rw [hv']
    have hRp := replayJ S h64 o hS hRA i hi r hr
    have hRS := ringSwitchJ S h64 o hS hRA o.rsExtras i hi r hr
    have hD := ((Run.repJ S h64 o hS hRA).2.2 i hi r hr).1
    rw [htabi, hlay, hzp] at hRp
    rw [htabi, hzp] at hRS
    have hT := o.toCase.tableT hi'
    obtain ⟨lg, hlg⟩ := hLg h64 _ _ hT o.ω i _ o.r _ _ _ _ r ic hin hr prior ws l p t' l' hpos hcoins hrun _ hAg _ hRp
      _ hRS
    have hLgOk : (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap
        (o.zp i r).proof.ligerito (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 =
          .ok (o.lg i r) := by
      refine FlockComplete.Replay.ok_val ⟨lg, ?_⟩
      rw [htabi, hsch, htp, hzp]
      exact hlg
    refine ⟨hLgOk, ?_⟩
    have hlanes := lanesJ hLgOk hD
    have hLgOk' := hLgOk
    rw [htabi, hsch, htp, hzp] at hLgOk'
    rw [hlay, hzp] at hlanes
    obtain ⟨u, hu⟩ := hIn h64 _ _ hT o.ω i _ o.r _ _ _ _ r ic hin hr prior ws l p t' l' hpos hcoins hrun _ hAg _ hRp
      _ hRS _ hLgOk' _ hlanes
    refine FlockComplete.Replay.ok_val ⟨u, ?_⟩
    rw [hlay, hzp]
    exact hu
  exact ⟨fun i hi r hr => (key i hi r hr).1, fun i hi r hr => (key i hi r hr).2⟩

/-- **Flock's masked verifier is complete at the deployed relation** for the honest masked prover on each table's
share, with error `δc H`, at any number of tables, under the JSON library's specification, from the four premises at
each table: the rep proofs' shapes (`RepShapedT`), the masked prover's reps (`ProverRunsT`), the masked Ligerito
(`LigeritoT`) and the inner proof (`InnerT`). -/
theorem complete_jsonT (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon) (hR : RepShapedT H)
    (hPR : ProverRunsT H) (hLg : LigeritoT H) (hIn : InnerT H) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) := fun h64 =>
  completeRelS H S h64 (proverRunsAt_of hPR h64) (repShapedAt_of hR h64) (lgInnerAt_of S h64 hR hPR hLg hIn) hMR
    h64

/-! ## One table: `complete_zk`'s premises from the table-level ones -/

/-- **A good masked case is a good case** (`readyZk` gives `ready`). -/
def _root_.FlockComplete.CaseZk.toCase (c : CaseZk H rel) : Case H rel :=
  ⟨c.x, c.w, c.ω, c.r, c.s, c.spec, c.sts, c.scope, c.rel_w, ready_of_readyZk c.ready, c.good, c.setup⟩

/-- **A masked run is a run** of its one table. -/
def _root_.FlockComplete.RunZk.toRun (o : RunZk H rel) : Run H rel :=
  ⟨o.toCaseZk.toCase, o.record, o.proofs, o.bodies, (sessionS_one o.toCaseZk.ready_all.1 o.w o.ω).trans o.ran⟩

/-- **A good masked case's one table and its trace** (`TableT`). -/
theorem _root_.FlockComplete.CaseZk.tableT (c : CaseZk H Readable.relation) {st : Flock.Stmt} (hst : c.sts = #[st]) :
    TableT st (tracesOf c.sts c.w)[0]! := by
  have h1 : c.sts.size = 1 := by rw [hst]; rfl
  have h := c.toCase.tableT (i := 0) (by show 0 < c.sts.size; omega)
  change TableT c.sts[0]! (traces c.sts c.w)[0]! at h
  rw [traces_one h1, show c.sts[0]! = st by rw [hst]; rfl] at h
  exact h

/-- The link's points of a good masked case have its one table's `m_pts`. -/
theorem _root_.FlockComplete.CaseZk.points_mPts (c : CaseZk H rel) {st : Flock.Stmt} (hst : c.sts = #[st]) :
    ∀ a ∈ pointsOf c.spec (Claims.coinSeq c.r 0), a.size = st.mPts := by
  have h := c.toCase.points_mPts (i := 0) (by show 0 < c.sts.size; rw [hst]; simp)
  change ∀ a ∈ pointsOf c.spec (Claims.coinSeq c.r 0), a.size = c.sts[0]!.mPts at h
  rwa [show c.sts[0]! = st by rw [hst]; rfl] at h

/-- **`RepShapedZk` at the deployed relation**, from `RepShapedT` at the one table. -/
theorem repShapedZk_of (hR : RepShapedT H) (h64 : Claims.Out64 H) : JsonRT.RepShapedZk H Readable.relation := by
  intro c st v cm e hst hv hcm hex rep ic hic d prior ws
  exact hR h64 st _ (c.tableT hst) c.ω 0 d c.r v cm _ e rep ic ⟨hv, hcm, c.points_mPts hst, hex, hic⟩
    (c.toCase.extras_le (st := st) (by show st ∈ c.sts; rw [hst]; simp) hex) prior ws

/-- **`ProverRunsZk` at the deployed relation**, from `ProverRunsT` at the one table: rep `0` runs from the link's
send, where `¬ BadZk` gives its coins, and ends `repRounds` sends later, where rep `1`'s coins are good. -/
theorem proverRunsZk_of (hPR : ProverRunsT H) (h64 : Claims.Out64 H) : JsonRT.ProverRunsZk H Readable.relation := by
  intro c st v hst hv cm e hcm hex ic hic l₀ hl₀
  have h1 : c.sts.size = 1 := by rw [hst]; rfl
  have hst0 : c.sts[0]! = st := by rw [hst]; rfl
  have hx := extras_size hex
  have hin : ∀ rep, RepIn H st (tracesOf c.sts c.w)[0]! c.ω 0 v cm (pointsOf c.spec (Claims.coinSeq c.r 0)) e rep
      (ic rep) := fun rep => ⟨hv, hcm, c.points_mPts hst, hex, hic rep⟩
  have hv0 : Zk.schedule c.toCase.sts[0 / 2]!.m = .ok v := by
    show Zk.schedule c.sts[0]!.m = .ok v
    rw [hst0]
    exact hv
  have he0 : e.size = c.toCase.spec.points * c.toCase.sts[0 / 2]!.regions.size := by
    show e.size = c.spec.points * c.sts[0]!.regions.size
    rw [hst0]
    exact hx
  have hco0 := c.toCase.coinsT (q := 0) (by show 0 < 2 * c.sts.size; omega) hv0 he0
  have hco1 := c.toCase.coinsT (q := 1) (by show 1 < 2 * c.sts.size; omega) hv0 he0
  have hmr0 := c.toCase.maskRankT (q := 0) (by show 0 < 2 * c.sts.size; omega) hv0 he0
  have hmr1 := c.toCase.maskRankT (q := 1) (by show 1 < 2 * c.sts.size; omega) hv0 he0
  change CoinsT c.sts[0]! v e c.r (repStartZk c.spec c.sts 0) at hco0
  change CoinsT c.sts[0]! v e c.r (repStartZk c.spec c.sts 1) at hco1
  change MaskRankT c.sts[0]! c.r (repStartZk c.spec c.sts 0) #[] at hmr0
  change MaskRankT c.sts[0]! c.r (repStartZk c.spec c.sts 1)
    (MaskRank.claimsAt c.sts[0]! c.r (repStartZk c.spec c.sts 0)) at hmr1
  rw [hst0] at hco0 hco1 hmr0 hmr1
  have hs0 : repStartZk c.spec c.sts 0 = 1 := by simp [repStartZk]
  have hs1 : repStartZk c.spec c.sts 1 = 1 + repRounds true c.spec st := by
    have h := repStart_succ c.spec c.sts 0
    rw [Nat.zero_add, Nat.zero_div, hst0, hs0] at h
    exact h
  obtain ⟨hrun0, hcnt0⟩ := hPR h64 st _ (c.tableT hst) c.ω 0 (c.spec.domain c.spec.tables[0]! 0).toUTF8 c.r v cm _ e
    0 (ic 0) (hin 0) (by norm_num)
  obtain ⟨⟨p₀, t₀⟩, l₁, hgo0⟩ := hrun0 l₀ #[] (by omega) (by rw [hl₀, ← hs0]; exact hco0)
    (by rw [hl₀, ← hs0]; exact hmr0)
  refine ⟨_, l₁, hgo0, ?_⟩
  have hc : t₀.rounds.size = repRoundsT st v e := hcnt0 _ _ l₀ _ l₁ hgo0
  have hsz : l₁.size = repStartZk c.spec c.sts 1 := by
    rw [repT_size H c.r _ _ _ _ _ _ _ _ _ _ _ _ hgo0, hc, hl₀, hs1, repRoundsT_eq (st := st) hv hx]
  have hpt : MaskRank.roundPoints st.m st.c.kLog t₀.rounds = MaskRank.claimsAt st c.r (repStartZk c.spec c.sts 0) :=
    (repT_points H c.r _ _ _ _ _ _ _ _ _ _ _ _ (by have := (c.tableT hst).1; omega) hgo0
      (by rw [hc]; exact repRoundsT_ge _ _ _)).trans (by rw [hl₀, hs0])
  obtain ⟨hrun1, -⟩ := hPR h64 st _ (c.tableT hst) c.ω 0 (c.spec.domain c.spec.tables[0]! 1).toUTF8 c.r v cm _ e
    1 (ic 1) (hin 1) (by norm_num)
  obtain ⟨pt₁, l₂, hgo1⟩ := hrun1 l₁ (MaskRank.roundPoints st.m st.c.kLog t₀.rounds) (by rw [hsz, hs1]; omega)
    (by rw [hsz]; exact hco1) (by rw [hsz, hpt]; exact hmr1)
  exact ⟨pt₁, l₂, hgo1⟩

/-- **`LigeritoZk` and `Inner` at the deployed relation**, from `LigeritoT` and `InnerT` at the one table: a masked run
is a run of one table (`RunZk.toRun`), at which `lgInnerAt_of` gives both. -/
theorem lgInnerZk_of (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hR : RepShapedT H) (hPR : ProverRunsT H)
    (hLg : LigeritoT H) (hIn : InnerT H) :
    FlockComplete.LigeritoZk H Readable.relation ∧ FlockComplete.Inner H Readable.relation :=
  ⟨fun o => (lgInnerAt_of S h64 hR hPR hLg hIn o.toRun).ligerito,
    fun o => (lgInnerAt_of S h64 hR hPR hLg hIn o.toRun).inner⟩

end FlockComplete.ZkTables
