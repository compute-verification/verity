import Proofs.Flock.Claims.Recursion.NextRSetup
import Proofs.Flock.Claims.Recursion.NextRRep
import Proofs.Flock.Claims.Complete.Zk.Walk
import Proofs.Flock.Claims.Complete.Sim
import Proofs.Flock.Claims.Complete.Json.Goals

/-!
# The honest session's proofs are its reps'

`honestRun_decides` decides at `honestPs`, proofs of the reps' sizes. Here the proofs are the honest session's own.

* **File `k` is table `k / 2`'s rep** (`sessionP_repRuns`): at any number of tables, the plain session's `k`th proof
  file encodes the proof `repP` returns on table `k / 2`'s setup, schedule, trace, packed table and level-0
  commitment, with its extra claims at the link's points and rep `k % 2`'s salts and nonces, from the empty transcript
  at the rep's domain, after the link's send and the earlier reps' (`sendsP`, `RepRunP`). A rep's proof carries its
  commitment's cap (`repP_cap`).
* **The files decode to honest reps** (`sessionS_reps`): at a `sessionS` run whose tables are `TableOk` with at most
  2,048 extra claims at the first coin's link points, file `k` decodes at `shapeAt` to that proof (`filePs`), whose
  cap, its table's, has the table's level-0 node count, and which is an honest rep of `repCtx` (`honestRep_of_repP`)
  wherever its `finalC` is the zerocheck's at the reader's `z`; that `finalC` is the zerocheck's at the reply the
  rep's `z` is read from (`zAtP`, `JsonRT.repP_finalC`). So the honest run's last message carries `filesAt` of
  those proofs (`honestRun_files`).
* **The honest run decides at its own proofs** (`sessionRun_decides`): `ReadsBack` at `filePs` and each table's cap
  (`fileCaps`), wherever their rows fit a message (`RowsFit`), the last message reads as `filePs`, and the files'
  `finalC` are at the reader's `z` (`FilesFinalC`, which `NextRCoins` proves where the replies are the coins' bytes).

The session's tables are taken to number the spec's (`sts.size = spec.tables.size`), as the masked scope proves
(`ZkTables.tables_of_setup`).
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy

theorem range_map_getElem! {β : Type} [Inhabited β] (g : ℕ → β) {n i : ℕ} (hi : i < n) :
    ((Array.range n).map g)[i]! = g i := by
  rw [getElem!_pos ((Array.range n).map g) i (by simpa using hi)]
  simp

/-- **A rep's proof carries its commitment's cap.** -/
theorem repP_cap (resp : ℕ → Claims.Msg) (hsP : Hashes PS) (st : Setup) (sch : Schedule) (tr : Trace)
    (tbl : Array F128) (cm0 : Commit) (extras : Array Claim) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (t : Transcript) :
    Ret ((repP (asksS resp) hsP st sch tr tbl cm0 extras salt nonce).run t) fun pt => pt.1.cap = cm0.cap := by
  unfold repP
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨zp, zc⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨lp, lc⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun ⟨ss, T, sws⟩ _ _ => ?_
  refine FlockComplete.Replay.Ret.run_bind (FlockComplete.Ret.true _) fun lg _ _ => ?_
  rw [StateT.run_pure]
  exact FlockComplete.Ret.pure rfl

/-- **Rep `k`'s sends** in the plain session (`Replay.repP_count`): its zerocheck's, lincheck's, its ring switch's at
its table's extra claims at the link's points, and Ligerito's at its table's `fast100` schedule. -/
def sendsP (resp : ℕ → Claims.Msg) (spec : SessionSpec) (tabs : Array Setup) (k : ℕ) : ℕ :=
  (3 + (tabs[k / 2]!.m - 6)) + (3 + (tabs[k / 2]!.fold.kLog - 6)) +
    (2 + (match tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) with
      | .ok e => e.size
      | .error _ => 0) + 1) +
    (((fast100 tabs[k / 2]!.m).getD default).k0 + 5 + 11 * (((fast100 tabs[k / 2]!.m).getD default).levels.size - 1))

/-- **Rep `k`'s run** in the plain session: on table `k / 2`'s setup, schedule, trace, packed table and level-0
commitment, with the table's extra claims `e` at the link's points and rep `k % 2`'s salts and nonces, `repP` returns
`pt` from the empty transcript at the rep's domain, after the link's send and the earlier reps' (`sendsP`), and `f` is
its proof's file. -/
def RepRunP (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (schs : Array Schedule) (trs : Array Trace) (cms : Array Commit) (tape : Tape) (k : ℕ) (f : ByteArray) : Prop :=
  ∃ e pt l l', tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) = .ok e ∧
    l.size = 1 + ((List.range k).map (sendsP resp spec tabs)).sum ∧
    ((repP (asksS resp) (Hashes.ofSha512 h) tabs[k / 2]! schs[k / 2]! trs[k / 2]!
      (packed trs[k / 2]! tabs[k / 2]!.m) cms[k / 2]! e (tape.salt (k / 2) (k % 2)) (tape.nonce (k / 2) (k % 2))).run
        (Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8)).go l = (.ok pt, l') ∧
    f = FlockVBridge.Family.encodeProof (FlockComplete.shapeOf tabs[k / 2]! schs[k / 2]! e.size) pt.1

/-- **The plain session, rep by rep**, at any number of tables: each table's `fast100` schedule and level-0
commitment, and the `k`th proof file is rep `k`'s (`RepRunP`). -/
theorem sessionP_repRuns (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (trs : Array Trace) (tape : Tape) (hts : tabs.size = spec.tables.size)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs trs tape).go #[] = (.ok r, bodies)) :
    ∃ schs cms, (∀ i < tabs.size, fast100 tabs[i]!.m = some schs[i]! ∧
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) tabs[i]!.merkle (schs[i]!.levels.getD 0 default)
        (packed trs[i]! tabs[i]!.m) (tape.salt i 0 0) = .ok cms[i]!) ∧
      r.2.size = 2 * tabs.size ∧
      ∀ k < 2 * tabs.size, RepRunP resp (Claims.hOf H) spec tabs schs trs cms tape k r.2[k]! := by
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
  obtain ⟨hsz₁, hf₁⟩ := hschs
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hcms := Ret.go (Ret.mapM_array _ (fun i cm => commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V)
      tabs[i]!.merkle (schs[i]!.levels.getD 0 default) (packed trs[i]! tabs[i]!.m) (tape.salt i 0 0) = .ok cm) _
    fun i hi => by
      simp only [range_map_getElem! _ (Array.mem_range.1 hi)]
      exact Ret.of_pureS (commitP_pureS _ _ _ _ _) fun cm e => e) hB
  have hl2 := Still.go (Still.mapM_array (fun i => Still.of_pureS (commitP_pureS _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨hsz₂, hf₂⟩ := hcms
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, hrun⟩ := go_bind_ok hrun
  have hsch : ∀ i < tabs.size, fast100 tabs[i]!.m = some schs[i]! := fun i hi => by
    rw [getElem!_pos tabs i hi, getElem!_pos schs i (by omega)]
    exact hf₁ i hi (by omega)
  have hinv := forIn_array_idx _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      l.size = 1 + ((List.range k).map (sendsP resp spec tabs)).sum ∧
      b.2.size = k ∧ ∀ j < k, RepRunP resp (Claims.hOf H) spec tabs schs trs cms tape j b.2[j]!) ?_
    (by exact ⟨by simp, rfl, by simp⟩) hF
  · obtain ⟨-, h₁, h₂⟩ := hinv
    rw [repStreams_size, ← hts] at h₁ h₂
    dsimp only at hrun
    obtain ⟨rec, l₅, -, hrun⟩ := go_bind_ok hrun
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at hrun
    obtain ⟨rfl, rfl⟩ := hrun
    refine ⟨schs, cms, fun i hi => ⟨hsch i hi, ?_⟩, h₁, h₂⟩
    · have := hf₂ i (by simpa using hi) (by simp at hsz₂; omega)
      simp only [Array.getElem_range] at this
      rw [getElem!_pos cms i (by simp at hsz₂; omega)]
      exact this
  · rintro k ⟨i, rp, name⟩ ⟨ss, ps⟩ l res l' hk ⟨h₀, h₂, h₅⟩ e
    dsimp only at h₂ h₅
    obtain ⟨hk', hkv⟩ := Array.getElem?_eq_some_iff.1 hk
    rw [repStreams_getElem spec hk'] at hkv
    simp only [Prod.mk.injEq] at hkv
    obtain ⟨rfl, rfl, -⟩ := hkv
    have hk2 : k / 2 < tabs.size := by
      rw [repStreams_size] at hk'
      omega
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [range_map_getElem! _ hk2] at e2
    have hex' : tabs[k / 2]!.extra (ptsOf spec (coinOf (resp 0))) = .ok extras := by simpa [ptsOf] using hex
    refine ⟨_, rfl, ?_, by simp [h₂], fun j hj => ?_⟩
    · have hm := (RecDeploy.fast100_sizes (hsch _ hk2)).1
      rw [FlockComplete.Replay.repP_count resp _ _ _ _ _ _ _ _ _ (by omega) (hsch _ hk2) _ l e2, h₀,
        List.range_succ, List.map_append, List.sum_append]
      simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, sendsP, hsch _ hk2, Option.getD_some,
        hex']
      omega
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · rw [push_getElem!_lt _ _ (by omega)]
      exact h₅ j hj
    · refine ⟨extras, (pf, tr'), _, l₂, by simpa [ptsOf] using hex, h₀, e2, ?_⟩
      rw [← h₂, push_getElem!_size]

end FlockComplete.JsonRT

namespace FlockDeploy

open Flock FlockComplete

/-- **The proofs the session's files decode to**: file `k` at `shapeAt`, table `k / 2`'s shape at the first coin's
link points. -/
def filePs (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) : Array Flock.Proof :=
  pfs.mapIdx fun k b => match decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) b with
    | .ok p => p
    | .error _ => default

/-- Each table's cap: its first rep's. -/
def fileCaps (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    Array (Array ByteArray) :=
  (Array.range sts.size).map fun i => (filePs spec sts c₁ pfs)[2 * i]!.cap

theorem filePs_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    (filePs spec sts c₁ pfs).size = pfs.size := by
  simp only [filePs, Array.size_mapIdx]

theorem filePs_get {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin} {pfs : Array ByteArray} {k : ℕ}
    {p : Flock.Proof} (hk : k < pfs.size)
    (h : decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok p) :
    (filePs spec sts c₁ pfs)[k]! = p := by
  rw [getElem!_pos (filePs spec sts c₁ pfs) k (by rw [filePs_size]; exact hk)]
  simp only [filePs, Array.getElem_mapIdx]
  rw [← getElem!_pos pfs k hk, h]

theorem fileCaps_size (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (pfs : Array ByteArray) :
    (fileCaps spec sts c₁ pfs).size = sts.size := by
  simp only [fileCaps, Array.size_map, Array.size_range]

/-- The reply rep `k`'s zerocheck reads its `z` from in the plain session: after the link's send and the earlier reps'
(`JsonRT.sendsP`), the rep's third. -/
def zAtP (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Stmt) (k : ℕ) : ℕ :=
  1 + ((List.range k).map (JsonRT.sendsP resp spec (tabsOf sts))).sum + 2

/-- **The honest session's files decode to honest reps**: at a `sessionS` run whose replies start with the coin `c₁`,
at tables that are `TableOk` with at most 2,048 extra claims at `c₁`'s link points, there are two files a table, and
file `k` decodes at `shapeAt` to the proof table `k / 2`'s rep returned (`filePs`), whose cap is its table's, of the
table's level-0 node count, which is an honest rep of `repCtx` at the coins `cs` wherever its `finalC` is the
zerocheck's at that rep's `z`, whose `finalC` is the zerocheck's at the reply `zAtP`, and which is shaped at
`shapeAt`. -/
theorem sessionS_reps {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {resp : ℕ → Claims.Msg}
    {spec : SessionSpec} {sts : Array Stmt} {w : Claims.Flock.Wit} {ω : Tape} {c₁ : Coin} (hc : resp 0 = c₁.bytes)
    (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (r₀ : List ByteArray) (cs : List Coin) :
    pfs.size = 2 * sts.size ∧ ∀ k < 2 * sts.size,
      decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok (filePs spec sts c₁ pfs)[k]! ∧
      SchedOf (repCtx spec sts c₁ r₀ cs k) ∧
      ((filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
          dot (lagrange nodesL den6Inv (repCtx spec sts c₁ r₀ cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c →
        HonestRep (repCtx spec sts c₁ r₀ cs k) (filePs spec sts c₁ pfs)[k]!) ∧
      (filePs spec sts c₁ pfs)[k]!.cap = (filePs spec sts c₁ pfs)[2 * (k / 2)]!.cap ∧
      (filePs spec sts c₁ pfs)[k]!.cap.size = RecDeploy.level0Nodes sts[k / 2]! ∧
      (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC = dot (lagrange nodesL den6Inv
        (FlockComplete.coinOf (resp (zAtP resp spec sts k)) 0)) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c ∧
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (filePs spec sts c₁ pfs)[k]! := by
  have htabs : (tabsOf sts).size = sts.size := by
    unfold tabsOf
    rw [Array.size_map]
  obtain ⟨schs, cms, hsc, hsz, hk⟩ := JsonRT.sessionP_repRuns H resp spec (tabsOf sts) (tracesOf sts w) ω
    (by rw [htabs, hts]) hS
  rw [htabs] at hsc hsz hk
  replace hsz : pfs.size = 2 * sts.size := hsz
  have hone : ∀ k < 2 * sts.size,
      decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok (filePs spec sts c₁ pfs)[k]! ∧
      SchedOf (repCtx spec sts c₁ r₀ cs k) ∧
      ((filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
          dot (lagrange nodesL den6Inv (repCtx spec sts c₁ r₀ cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c →
        HonestRep (repCtx spec sts c₁ r₀ cs k) (filePs spec sts c₁ pfs)[k]!) ∧
      (filePs spec sts c₁ pfs)[k]!.cap = cms[k / 2]!.cap ∧
      (filePs spec sts c₁ pfs)[k]!.cap.size = RecDeploy.level0Nodes sts[k / 2]! ∧
      (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC = dot (lagrange nodesL den6Inv
        (FlockComplete.coinOf (resp (zAtP resp spec sts k)) 0)) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c ∧
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf sts) (pointsOf spec c₁) k) (filePs spec sts c₁ pfs)[k]! := by
    intro k hk'
    have hk2 : k / 2 < sts.size := by omega
    obtain ⟨e, pt, l, l', he, hls, hrun, hf⟩ := hk k hk'
    obtain ⟨hs, hcm⟩ := hsc (k / 2) hk2
    rw [hc, JsonRT.ptsOf_coin] at he
    have hsh : shapeAt (tabsOf sts) (pointsOf spec c₁) k =
        FlockComplete.shapeOf (tabsOf sts)[k / 2]! schs[k / 2]! e.size := by
      simp only [shapeAt, hs, he, Option.getD_some]
      rfl
    have hcap := JsonRT.Ret.go (JsonRT.repP_cap _ _ _ _ _ _ _ _ _ _ _) hrun
    have ht := tabsOf_get! hk2
    rw [ht] at hs hcm he hrun hf hsh
    have hs' : fast100 sts[k / 2].m = some schs[k / 2]! := hs
    have hon := honestRep_of_repP (spec := spec) (c₁ := c₁) (r₀ := r₀) (cs := cs) (k := k) h64 hk2 (htab _ hk2).1 hs'
      he ((htab _ hk2).2 e he) resp _ hcm _ _ _ hrun
    have hp := JsonRT.repShapedOk h64 (htab _ hk2).1 hs resp _ hcm ((htab _ hk2).2 e he) _ _ _ l pt l' hrun
    have hdec : decodeProof (shapeAt (tabsOf sts) (pointsOf spec c₁) k) pfs[k]! = .ok pt.1 := by
      rw [hsh, show pfs[k]! = (rec, pfs).2[k]! from rfl, hf]
      exact FlockVBridge.Family.decodeProof_encodeProof _ _ hp
    have hget := filePs_get (by rw [hsz]; exact hk') hdec
    have hfz := JsonRT.repP_finalC (p := pt.1) (t' := pt.2) resp _ _ _ _ _ _ _ _
      (by exact le_trans (by decide) (htab _ hk2).1.m22) _ hrun
    rw [hls] at hfz
    rw [hget]
    refine ⟨hdec, schedOf_repCtx hk2 (by simp [hs']) he, hon, hcap, ?_, hfz, by rw [hsh]; exact hp⟩
    have hsz' := FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _
      (FlockVBridge.Family.decodeProof_encodeProof _ _ hp)
    rw [hsz'.cap]
    show 2 ^ (schs[k / 2]!.levels.getD 0 default).cap =
      2 ^ ((fast100 sts[k / 2]!.m).map fun s => (s.levels.getD 0 default).cap).getD 0
    rw [getElem!_pos sts (k / 2) hk2, hs', Option.map_some, Option.getD_some]
  refine ⟨hsz, fun k hk' => ?_⟩
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := hone k hk'
  obtain ⟨-, -, -, h4', -, -⟩ := hone (2 * (k / 2)) (by omega)
  refine ⟨h1, h2, h3, ?_, h5, h6⟩
  rw [h4, h4', show 2 * (k / 2) / 2 = k / 2 by omega]

/-- **The honest run's last message carries its reps' proofs**: at a statement that sets up, where the pure session
returns, the last message is `Final.encode` of the statement, the record and `filesAt` of the proofs its files decode
to (`filePs`), each an honest rep (`sessionS_reps`). -/
theorem honestRun_files {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {x : Claims.Msg}
    {w : Claims.Flock.Wit} {ω : Tape} {k : ℕ} (r : Fin k → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hx : setupOf false x = some (s, spec, sts)) (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec (Claims.coinSeq r 0)) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (rec, pfs), bs)) :
    FlockComplete.honestRun H x w ω r = (bs.toList.map (List.cons 0), Final.encode ⟨x, rec,
      filesAt (tabsOf sts) (pointsOf spec (Claims.coinSeq r 0)) (filePs spec sts (Claims.coinSeq r 0) pfs)⟩) := by
  obtain ⟨hsz, hall⟩ := sessionS_reps (c₁ := Claims.coinSeq r 0) h64 (by rfl) hts htab hS [] []
  have hf : filesAt (tabsOf sts) (pointsOf spec (Claims.coinSeq r 0)) (filePs spec sts (Claims.coinSeq r 0) pfs) =
      pfs := filesAt_eq (by rw [filePs_size]) fun k hk => by
    have hk' : k < 2 * sts.size := by simpa [filePs_size, hsz] using hk
    have h := (hall k hk').1
    rw [getElem!_pos pfs k (by omega),
      getElem!_pos (filePs spec sts (Claims.coinSeq r 0) pfs) k (by rw [filePs_size]; omega)] at h
    exact h
  rw [hf]
  exact FlockComplete.honestRun_eq H x w ω r hx hS

/-- **The honest files' `finalC` is the zerocheck's at the reader's `z`**: each proof the files decode to
(`filePs`) has `finalC` at its rep's `z` in the coins `cs` (`repCtx`). This is plain zerocheck completeness at the
reader's coin: `sessionRun_decides` takes it, and `filesFinalC` proves it where the replies are the coins' bytes. -/
def FilesFinalC (spec : SessionSpec) (sts : Array Stmt) (c₁ : Coin) (cs : List Coin) (pfs : Array ByteArray) :
    Prop :=
  ∀ k < pfs.size, (filePs spec sts c₁ pfs)[k]!.zerocheck.finalC =
    dot (lagrange nodesL den6Inv (repCtx spec sts c₁ [] cs k).z) (filePs spec sts c₁ pfs)[k]!.zerocheck.round1c

/-- **The honest run decides at its own proofs**: at a statement that sets up, where the pure session returns at
replies that start with the first coin `c₁` and its tables are `TableOk` with at most 2,048 extra claims at `c₁`'s link
points, `decR` decides on the honest messages from the proofs the session's files decode to (`filePs`) and their
tables' caps (`fileCaps`), wherever their rows fit a message, the last message reads as `filePs` at the statement and
the files' `finalC` are at the reader's `z` (`FilesFinalC`). The files decode to proofs shaped for them. -/
theorem sessionRun_decides {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {rd : Claims.Msg → Option FinalP}
    {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {Ts : ℕ → F128} {fl : FinalP}
    (hx : setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁) (hc : resp 0 = c₁.bytes)
    (hts : sts.size = spec.tables.size)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs))
    (hfc : FilesFinalC spec sts c₁ cs.tail.dropLast pfs)
    (hfit0 : RowsFit ((capRows (fileCaps spec sts c₁ pfs)).map nodeRow))
    (hfit : ∀ k < (filePs spec sts c₁ pfs).size,
      ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
        (filePs spec sts c₁ pfs)[k]! (Ts k), RowsFit rs)
    (hrd : rd last = some fl) (hps : fl.ps = filePs spec sts c₁ pfs) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
      repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
        (Ts k)) last) cs = some d := by
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts htab hS (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast
  have hsh : ShapedAt spec sts c₁ (filePs spec sts c₁ pfs) := fun k hk => by
    have hk' : k < 2 * sts.size := by
      have h := hk
      rw [filePs_size, hsz] at h
      exact h
    have h := (hall k hk').2.2.2.2.2.2
    rwa [getElem!_pos (filePs spec sts c₁ pfs) k hk] at h
  have hcaps : ∀ i < (fileCaps spec sts c₁ pfs).size,
      (fileCaps spec sts c₁ pfs)[i]!.size = RecDeploy.level0Nodes sts[i]! := by
    intro i hi
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < (filePs spec sts c₁ pfs).size,
      SchedOf (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) ∧
        HonestRep (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! ∧
          (filePs spec sts c₁ pfs)[k]!.cap = (fileCaps spec sts c₁ pfs)[k / 2]! := by
    intro k hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < sts.size by omega)]
    exact h4
  exact ⟨_, readsBack rd x last cs s spec sts c₁ (fileCaps spec sts c₁ pfs) (filePs spec sts c₁ pfs) Ts fl hx hcs
    (fileCaps_size ..) hcaps (by rw [filePs_size, hsz]) hrep hfit0 hfit hrd hps hst hsh⟩

end FlockDeploy
