import Proofs.Flock.Claims.Complete.ZkTables.Files
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.Zk.Exact
import Proofs.Flock.Soundness.Discharge.ZkExec.RunsSeams

/-!
# A masked session that returns passed every refusal at its slot

`sessionPZk_pass`: one induction over the session's reps. Where the pure masked session at the replies `resp` returns,
each rep `q` started at `repStartZk q` sends and passed its refusals there (`PassAt`): level 0's pair coins, the
replies at `rhoSlotsZk`'s slot and the next, are independent; the inner `β`, the reply at `betaSlotsZk`'s slot, is
not `0`; and the rank check passed at the claim points of the replies from the rep's first send (`ptsAt`), rep `1`'s
after rep `0`'s. Each rep that returns sent exactly its `repRounds` (`repPZk_exact`), so the next starts where
`repStartZk` says; its rank check read its claim points at its sends' coins (`repPZk_rank`), and its rounds carry them
for rep `1`'s check (`repPZk_points`). Read the other way, at replies where some rep's slot refuses, the session does
not return: it stops at that refusal or at an earlier one.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay

/-! ## A rep's claim points at its sends' coins -/

/-- `claimPoints` reads coin `0` of its first `m + k_log − 6` rounds and the first `m − 13` coins of round `1`. -/
theorem claimPoints_congr_lt {m kLog : ℕ} (hm : 6 ≤ m) (hk : 6 ≤ kLog) {f g : ℕ → ℕ → F128}
    (h0 : ∀ j < m + kLog - 6, f j 0 = g j 0) (h1 : ∀ e < m - 13, f 1 e = g 1 e) :
    MaskRank.claimPoints m kLog f = MaskRank.claimPoints m kLog g := by
  unfold MaskRank.claimPoints
  have hr : (Array.range (m - 6)).map (fun j => f (3 + j) 0) = (Array.range (m - 6)).map fun j => g (3 + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have ht : (Array.range (kLog - 6)).map (fun j => f (5 + (m - 6) + j) 0) =
      (Array.range (kLog - 6)).map fun j => g (5 + (m - 6) + j) 0 :=
    Array.map_congr_left fun j hj => h0 _ (by rw [Array.mem_range] at hj; omega)
  have hc : (Array.range (m - 13)).map (f 1) = (Array.range (m - 13)).map (g 1) :=
    Array.map_congr_left fun e he => h1 e (by rw [Array.mem_range] at he; exact he)
  simp only [hr, ht, hc]

/-- **Round `1` after a masked rep's lincheck is the zerocheck's `m − 13` outer coins.** -/
theorem lcZ_round1 {lay lay' : Zk.PadLayout} {hz hz' : Array F128} {c : CircuitFold} {m m' : ℕ}
    {digest cap root : ByteArray} {cs : Zc.Coins} {tr tr' : Trace} {zc : ZcOut} {full : Array F128}
    {cs' : Lc.Coins} (dom : ByteArray) :
    (LcZ.tEnd lay' hz' c m' tr' zc full cs'
      (ZcZ.tEnd lay hz m digest cap root cs tr (Transcript.init #[] dom))).rounds[1]!.n = m - 13 := by
  have hp : Prefix ((ZcZ.t₂ digest cap root cs (Transcript.init #[] dom)).rounds.push
      (sent (ZcZ.t₂ digest cap root cs (Transcript.init #[] dom)) 2 (m - 13) cs.c1))
      (ZcZ.tEnd lay hz m digest cap root cs tr (Transcript.init #[] dom)).rounds :=
    (Prefix.push _ _).trans (ZcZ.prefix_loop (i := 0) _ (Nat.zero_le _))
  obtain ⟨C, hC⟩ := hp.trans (LcZ.prefix_start _)
  have hs : (ZcZ.t₂ digest cap root cs (Transcript.init #[] dom)).rounds.size = 1 := by
    simp [ZcZ.t₂, ZcZ.t₁, afterSq, Transcript.init]
  rw [hC, getElem!_pos _ 1 (by simp [hs]; omega), Array.getElem_append_left (by simp [hs])]
  simp [Array.getElem_push, hs, sent]

/-- **Rounds recorded from send `i` on carry their sends' coins as claim points**, where they reach the points'
`m + k_log − 6` rounds and round `1` holds `m − 13` coins. -/
theorem roundPoints_recs {resp : ℕ → Claims.Msg} {m kLog i : ℕ} {R : Array Round} {bs : List Claims.Msg}
    (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hrec : Recs resp i R.toList bs) (hsz : m + kLog - 6 ≤ R.size)
    (h1 : R[1]!.n = m - 13) :
    MaskRank.roundPoints m kLog R = MaskRank.claimPoints m kLog fun j e => coinOf (resp (i + j)) e := by
  have hround : ∀ j (hj : j < R.size), ∃ b, RoundIs resp (i + j) b R[j]! := by
    intro j hj
    obtain ⟨b, hb⟩ := Recs.getElem hrec j (by simpa using hj)
    exact ⟨b, by rw [getElem!_pos R j hj]; simpa only [Array.getElem_toList] using hb⟩
  have key : ∀ j (hj : j < R.size) a, a < R[j]!.n → (R[j]!.coins).getD a 0 = coinOf (resp (i + j)) a := by
    intro j hj a ha
    obtain ⟨b, c, -, -, -, hc⟩ := hround j hj
    rw [hc, Array.getD_eq_getD_getElem?, Array.getElem?_map, Array.getElem?_range]
    simp [ha]
  unfold MaskRank.roundPoints
  refine claimPoints_congr_lt (by omega) hk (fun j hj => ?_) (fun a ha => ?_)
  · obtain ⟨b, c, -, -, hn, -⟩ := hround j (by omega)
    exact key j (by omega) 0 hn
  · exact key 1 (by omega) a (by omega)

/-- **A returning masked rep's rank check read its claim points at its sends' coins**, from its first send `l`: the
rounds through the lincheck are its sends there (`zerocheckPZk_rec`, `lincheckPZk_rec`), `m + k_log − 6` of them,
round `1` the zerocheck's outer coins. -/
theorem repPZk_rank (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hk : 6 ≤ st.kLog)
    (hfk : st.fold.kLog = st.kLog) {l l' : Array Claims.Msg} {P : Zk.ZkProof} {t' : Transcript}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run (Transcript.init #[] dom)).go l = (.ok (P, t'), l')) :
    MaskRank.maskRankOk (prior ++ MaskRank.claimPoints st.m st.kLog fun j e => coinOf (resp (l.size + j)) e)
      words = true := by
  obtain ⟨l₁, -, hz⟩ := zerocheckPZk_go resp hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr (Transcript.init #[] dom) l
  unfold repPZk at h
  rw [bind_go hz] at h
  dsimp only at h
  have h' := h
  rw [StateT.run_bind] at h'
  obtain ⟨_, _, h₁, -⟩ := JsonRT.go_bind_ok h'
  obtain ⟨full, hfold⟩ := lincheckPZk_fold resp _ _ _ _ _ _ _ _ h₁
  obtain ⟨l₂, -, hl⟩ := lincheckPZk_go resp st.fold st.m (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr
    (Zc.out st.m (Zc.coinsAt resp l.size) tr) full _ l₁ hfold
  rw [bind_go hl] at h
  dsimp only at h
  rw [StateT.run_bind] at h
  obtain ⟨⟨u, t₃⟩, l₃', hk', -⟩ := JsonRT.go_bind_ok h
  have hrk := rankCheckP_ok hk'
  obtain ⟨rs₁, bs₁, hr₁, hb₁, hrec₁⟩ := JsonRT.Rec.go (zerocheckPZk_rec resp _ _ _ _ _ _ _) hz
  obtain ⟨rs₂, bs₂, hr₂, -, hrec₂⟩ := JsonRT.Rec.go (lincheckPZk_rec resp _ _ _ _ _ _) hl
  have hlen : l.size + rs₁.length = l₁.size := by
    have := congrArg List.length hb₁
    simp only [Array.length_toList, List.length_append] at this
    rw [Recs.length_eq hrec₁]
    omega
  have hrec := Recs.append hrec₁ (by rw [hlen]; exact hrec₂)
  have hinit : (Transcript.init #[] dom).rounds.toList = [] := by simp [Transcript.init]
  rw [hinit, List.nil_append] at hr₁
  rw [hr₁] at hr₂
  rw [← hr₂] at hrec
  have h0 : (Transcript.init #[] dom).rounds.size = 0 := by simp [Transcript.init]
  rw [roundPoints_recs hm hk hrec (by rw [LcZ.tEnd_size, ZcZ.tEnd_size, h0, hfk]; omega) (lcZ_round1 dom)] at hrk
  exact hrk

/-- **A returning masked rep's rounds carry its sends' coins as claim points**, from its first send `l`
(`repPZk_rec`), where it sent at least the points' `m + k_log − 6` rounds. -/
theorem repPZk_points (resp : ℕ → Claims.Msg) (hsh : ByteArray → ByteArray) (st : Setup) (sch : Schedule)
    (tPad rep : ℕ) (dom : ByteArray) (tr : Trace) (tb : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray)
    (nonce : ℕ → ℕ → UInt64) (prior : Array (List F128)) (words : List ℕ) (hm : 14 ≤ st.m) (hk : 6 ≤ st.kLog)
    {l l' : Array Claims.Msg} {P : Zk.ZkProof} {t' : Transcript}
    (h : ((repPZk (asksS resp) (Hashes.ofSha512 hsh) st sch tPad rep dom tr tb cm0 extras pads ic extra lanePad salt
      nonce prior words).run (Transcript.init #[] dom)).go l = (.ok (P, t'), l'))
    (hsz : l.size + (st.m + st.kLog - 6) ≤ l'.size) :
    MaskRank.roundPoints st.m st.kLog t'.rounds =
      MaskRank.claimPoints st.m st.kLog fun j e => coinOf (resp (l.size + j)) e := by
  obtain ⟨rs, bs, hrs, hl', hrec⟩ := JsonRT.Rec.go (repPZk_rec resp hsh _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) h
  have hR : t'.rounds.toList = rs := by simpa [Transcript.init] using hrs
  rw [← hR] at hrec
  refine roundPoints_recs hm hk hrec ?_ (repPZk_round1 (hm := hm) (h := h))
  have h1 := congrArg List.length hl'
  have h2 := Recs.length_eq hrec
  simp only [Array.length_toList, List.length_append] at h1 h2
  omega

/-! ## The session -/

/-- **Table `i`'s claim points at the replies from send `repStartZk n`.** -/
def ptsAt (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) (i n : ℕ) : Array (List F128) :=
  MaskRank.claimPoints sts[i]!.m sts[i]!.c.kLog fun j e => coinOf (resp (repStartZk spec sts n + j)) e

/-- **Rep `q` passed its refusals at its slots**: at its table's `--zk` schedule, level 0's pair coins (`rhoSlotsZk`'s
slot and the next) are independent and its inner `β` (`betaSlotsZk`'s slot) is not `0`; and its rank check passed, at
its claim points from its first send (`ptsAt`), after rep `q − 1`'s at rep `1`, over its table's `words`. -/
def PassAt (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt) (words : Array (List ℕ))
    (q : ℕ) : Prop :=
  (∀ v : Schedule × ℕ, Zk.schedule sts[q / 2]!.m = .ok v →
    Zk.rhoIndependent (coinKOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (v.1.k0 + 2))),
      coinKOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! + (v.1.k0 + 2) + 1))) = true ∧
    (coinOf (resp (repStartZk spec sts q + piopRoundsZk spec sts[q / 2]! +
      (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 2)) 0).isZero = false) ∧
  MaskRank.maskRankOk ((if q % 2 = 1 then ptsAt resp spec sts (q / 2) (q - 1) else #[]) ++
    ptsAt resp spec sts (q / 2) q) words[q / 2]! = true

/-- **A masked session that returns passed every rep's refusals at their slots** (`PassAt`), at tables of `14 ≤ m`
and `6 ≤ k_log`, one spec table per statement. -/
theorem sessionPZk_pass (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec)
    (sts : Array Flock.Stmt) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk)
    (hspec : spec.tables.size = sts.size) (hsts : ∀ j (hj : j < sts.size), 14 ≤ sts[j].m ∧ 6 ≤ sts[j].c.kLog)
    {r : String × Array ByteArray} {bodies : Array Claims.Msg}
    (hrun : (sessionPZk (asksS resp) (Hashes.ofSha512 h) spec (tabsOf sts) words trs tape).go #[] =
      (.ok r, bodies)) :
    ∀ q < 2 * sts.size, PassAt resp spec sts words q := by
  unfold sessionPZk at hrun
  obtain ⟨schs, l1, hA, hrun⟩ := go_bind_ok hrun
  obtain ⟨hssz, hsf⟩ := Ret.go (Ret.mapM_array _ (fun st' v => Zk.schedule st'.m = .ok v) _ fun a _ =>
    Ret.monadLift (Zk.schedule a.m)) hA
  have hl1 := Still.go (Still.mapM_array (fun a l => by rw [monadLift_go]) _) hA
  subst hl1
  have htabs : (tabsOf sts).size = sts.size := tabsOf_size sts
  have htab : ∀ i (hi : i < sts.size), (tabsOf sts)[i]! = Setup.ofCircuit sts[i]! := fun i hi => by
    rw [getElem!_pos (tabsOf sts) i (by omega), getElem!_pos sts i hi]
    simp [tabsOf]
  have hv : ∀ i (hi : i < sts.size), Zk.schedule sts[i]!.m = .ok schs[i]! := fun i hi => by
    have := hsf i (by omega) (by omega)
    rw [← getElem!_pos (tabsOf sts) i (by omega), htab i hi, ← getElem!_pos schs i (by omega)] at this
    exact this
  dsimp only at hrun
  obtain ⟨cms, l2, hB, hrun⟩ := go_bind_ok hrun
  have hl2 := Still.go (Still.mapM_array (fun i => by
    exact Still.of_pureS (commitPZk_pureS _ _ _ _ _ _ _ _)) _) hB
  subst hl2
  obtain ⟨c₁, l₃, hL, hrun⟩ := go_bind_ok hrun
  rw [link_go] at hL
  simp only [Prod.mk.injEq, Except.ok.injEq, Array.size_empty] at hL
  obtain ⟨rfl, rfl⟩ := hL
  obtain ⟨⟨ss, ps⟩, l₄, hF, -⟩ := go_bind_ok hrun
  have hinv := forIn_array_idx _ _ (fun k (b : Array (Array Round) × Array ByteArray) (l : Array Claims.Msg) =>
      l.size = repStartZk spec sts k ∧
        (k % 2 = 1 → MaskRank.roundPoints sts[k / 2]!.m sts[k / 2]!.c.kLog b.1.back! =
          ptsAt resp spec sts (k / 2) (k - 1)) ∧ ∀ q < k, PassAt resp spec sts words q)
    ?_ (by exact ⟨by simp [repStartZk], fun h => absurd h (by decide), fun q hq => absurd hq (Nat.not_lt_zero _)⟩) hF
  · intro q hq
    exact hinv.2.2 q (by rw [repStreams_size, hspec]; exact hq)
  · rintro k a ⟨ss, ps⟩ l res l' ha ⟨hl, hback, hpass⟩ e
    obtain ⟨hk, rfl⟩ := Array.getElem?_eq_some_iff.1 ha
    have hk' : k < 2 * sts.size := by rw [repStreams_size, hspec] at hk; exact hk
    have hi : k / 2 < sts.size := by omega
    rw [repStreams_getElem _ hk] at e
    obtain ⟨extras, l₁, e1, e⟩ := go_bind_ok e
    rw [monadLift_go] at e1
    simp only [Prod.mk.injEq] at e1
    obtain ⟨hex, rfl⟩ := e1
    obtain ⟨ic, l₀, e0, e3⟩ := go_bind_ok e
    rw [Still.go (Still.of_pureS (innerCommitP_pureS _ _ _)) e0] at e3
    obtain ⟨⟨pf, tr'⟩, l₂, e2, e⟩ := go_bind_ok e3
    simp only [go_pure, Prod.mk.injEq, Except.ok.injEq] at e
    obtain ⟨rfl, rfl⟩ := e
    rw [htab _ hi] at e2 hex
    obtain ⟨hm, hk6⟩ := hsts (k / 2) hi
    rw [← getElem!_pos sts (k / 2) hi] at hm hk6
    obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ :=
      FlockSoundness.Discharge.ZkExec.zk_schedule_seams sts[k / 2]!.m schs[k / 2]!.1 schs[k / 2]!.2 (hv _ hi)
    obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
    obtain ⟨hl₂, hrho, hb, -⟩ := repPZk_exact (sch := schs[k / 2]!.1) (hm := hm) (hn := by rw [hn7, hsm]; omega) (hk0 := hk0) (hL := h2)
      (hq := fun i hi => (hlv i hi).q_ne_zero) (h := e2)
    have hx : extras.size = spec.points * sts[k / 2]!.regions.size := by
      rw [JsonRT.extraClaims_size hex]
      simp [Nat.mul_comm]
    have hp : piopZk (Setup.ofCircuit sts[k / 2]!) extras = piopRoundsZk spec sts[k / 2]! := by
      have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = sts[k / 2]!.c.kLog := rfl
      have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
      unfold piopZk piopRoundsZk
      rw [e6, e7, hx]
      omega
    have hrank := repPZk_rank (hm := hm) (hk := hk6) (hfk := rfl) (h := e2)
    have hpts := repPZk_points (hm := hm) (hk := hk6) (h := e2) (hsz := by
      have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = (Setup.ofCircuit sts[k / 2]!).kLog := rfl
      rw [hl₂]
      unfold piopZk
      rw [e6]
      omega)
    refine ⟨_, rfl, ?_, ?_, fun q hq => ?_⟩
    · rw [repStart_succ, ← repRoundsT_eq (hv _ hi) hx, hl₂, hl]
      have e6 : (Setup.ofCircuit sts[k / 2]!).fold.kLog = sts[k / 2]!.c.kLog := rfl
      have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
      unfold repRoundsT piopZk
      rw [e6, e7, hx]
      omega
    · intro hk1
      dsimp only
      rw [show (k + 1) / 2 = k / 2 by omega, Array.back!_push, show k + 1 - 1 = k by omega]
      unfold ptsAt
      rw [← hl]
      exact hpts
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hq with hq | rfl
      · exact hpass q hq
      refine ⟨fun v hv' => ?_, ?_⟩
      · rw [hv _ hi] at hv'
        cases hv'
        rw [← hl, ← hp]
        refine ⟨?_, ?_⟩
        · rw [show l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + (schs[q / 2]!.1.k0 + 2) =
              l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 2 + schs[q / 2]!.1.k0 by omega,
            show l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 2 + schs[q / 2]!.1.k0 + 1 =
              l.size + piopZk (Setup.ofCircuit sts[q / 2]!) extras + 3 + schs[q / 2]!.1.k0 by omega]
          exact hrho
        · exact hb
      · have e8 : ptsAt resp spec sts (q / 2) q = MaskRank.claimPoints (Setup.ofCircuit sts[q / 2]!).m
            (Setup.ofCircuit sts[q / 2]!).kLog fun j e => coinOf (resp (l.size + j)) e := by
          unfold ptsAt
          rw [← hl]
          rfl
        unfold priorOf at hrank
        rw [e8]
        by_cases h1 : q % 2 = 1
        · have e7 : MaskRank.roundPoints (Setup.ofCircuit sts[q / 2]!).m (Setup.ofCircuit sts[q / 2]!).kLog
              ss.back! = ptsAt resp spec sts (q / 2) (q - 1) := hback h1
          simp only [h1, ↓reduceIte] at hrank ⊢
          rw [e7] at hrank
          exact hrank
        · simp only [h1, ↓reduceIte] at hrank ⊢
          exact hrank

end FlockComplete.ZkTables
