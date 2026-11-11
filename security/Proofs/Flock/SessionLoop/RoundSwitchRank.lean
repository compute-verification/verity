module

public import Proofs.Flock.SessionLoop.RoundPiop
public import Proofs.Flock.SessionLoop.RoundSwitchSched

/-!
# A rep's rank check at its ring switch's first round

At a rep's rank check's round (`Masked.rankSlot`), the first of its ring switch, the frame takes the coins of the rep's
last algebraic round past the rank stop (`rankStop`), which checks the rank there.

* **`Loop.coinsAt_words`**: the coins the loop feeds a frame are the last opening's words.
* **`LeanRelease.rank_unrefused_pcs`**: where the frame takes the coins after a history at a rep's rank check's round,
  the rep's rank check holds at the opened coins.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128)

namespace Loop

/-- The coins the loop feeds a frame after `e0 :: ops`, `q + 1` openings past `e0`, are opening `q`'s words. -/
theorem coinsAt_words (x : Msg) (e0 : Msg × Msg) (ops : List (Msg × Msg)) {q : ℕ} (hq : ops.length = q + 1) :
    coinsAt x (e0 :: ops) = wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) q := by
  rw [← readCoins_eq x _ (by rw [List.length_map]; exact hq)]
  unfold coinsAt readCoins
  simp only [List.tail_cons, List.getLast?_map, List.length_map]
  rcases ops.getLast? with _ | ⟨a, b⟩
  · rfl
  · exact feedOf_rd _ b

end Loop

namespace LeanRelease

open _root_.Flock.Firewall.Frame (K_SKIP)
open Loop Ranked

set_option maxHeartbeats 1600000 in
/-- **The frame passes a rep's rank check at its ring switch's first round**: on a history of at most `refK x`
openings whose last coins the frame takes, at table `j`'s rep `r`'s rank check (`Masked.rankSlot`), the round after
the rep's algebraic phase, the rep's rank check holds at the opened coins. The frame took the algebraic phase's last
coins past the rank stop, in the algebraic phase at its last position, holding the phase's coins after the earlier
rep's claim points (`repState_pre_pcs`); there the rank stop passing is the rank check (`rank_of_rankFails`), on the
opened coins (`wordsAt_slot`), as at the rep's later rounds (`rank_unrefused_sched`). -/
theorem rank_unrefused_pcs (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length ≤ refK x)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : ops.length = Masked.rankSlot x j r) :
    ¬ Masked.RankBad (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r := by
  intro hbad
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hstj : (Masked.stsOf x)[j]! = (Masked.stsOf x)[j] := getElem!_pos _ j hj
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  have hm14 : 14 ≤ (Masked.stsOf x)[j].m := by have := (hall j hj).1; omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  obtain ⟨N, hNd⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrR : ∀ q, q / 2 = (2 * j + r) / 2 → FlockDeploy.repRounds true spec (Masked.stsOf x)[q / 2]! = N + 4 := by
    intro q hq'
    rw [hq']
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hrq : ∀ q, q / 2 = (2 * j + r) / 2 → q < spec.repStreams.size →
      (roundsOf spec (Masked.stsOf x)).getD q 0 = N + 4 := by
    intro q hq' hq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
      Option.getD_some]
    exact hrR q hq'
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨A, hAd⟩ : ∃ A, A = (Masked.stsOf x)[j]!.m + (Masked.stsOf x)[j]!.c.kLog - 6 := ⟨_, rfl⟩
  have hAN : A < N := by
    rw [hNd, hAd, hst, hstj]
    unfold FlockComplete.piopRoundsZk
    omega
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + A := ⟨_, rfl⟩
  have hbs : Masked.rankSlot x j r = s := by
    unfold Masked.rankSlot
    rw [Masked.repBase_eq hsx, hs, hAd]
  rw [hbs] at hN
  have hA2 : 2 ≤ A := by rw [hAd, hstj]; omega
  obtain ⟨i, hi⟩ : ∃ i, s = i + 2 := ⟨s - 2, by omega⟩
  obtain ⟨a, ha1⟩ : ∃ a, A = a + 1 := ⟨A - 1, by omega⟩
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1) = (2 * j + r, A) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) A (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    have := repStartZk_eq spec (Masked.stsOf x) hq2.le
    omega
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).2 = a + 1 := by rw [hloc, ha1]
  -- the frame took the round's coins past the rank stop
  unfold refusesCoins at hr
  obtain ⟨m, hA⟩ : ∃ m, (ofLoop T L₀ real).answered x ρ ((m₀, ρ₀) :: ops) = .ok m := by
    cases hA : (ofLoop T L₀ real).answered x ρ ((m₀, ρ₀) :: ops) with
    | error e => rw [hA] at hr; cases hr
    | ok m => exact ⟨m, rfl⟩
  obtain ⟨ho, ht, -⟩ := answered_ok hA
  have hA' := hA
  rw [ofLoop_answered_same T L₀ real x ρ (m₀, ρ₀) ops hsx (q := i + 1) (by omega) hj1 ho ht] at hA'
  obtain ⟨hA', hrf⟩ := bind_rankStop_ok hA'
  -- the loop's placeholder table and its state there
  have hcsl : (ops.map fun p => ZKMal.CoinLayer.rd p.2).length = i + 2 := by rw [List.length_map]; omega
  have ho' : (progOf x (ops.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun h => answer_owed_none _ h hA'
  obtain ⟨c', t, dom, xs, his, hph0⟩ := progOf_placeholder x _ hsx hcsl ho'
  rw [locate_pred _ 0 i a hj1, hloc] at hph0
  dsimp only at hph0
  have hph : placeholder spec (Masked.stsOf x) (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (i + 1)).1
      (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', t, dom, xs, his) := by
    rw [hloc]
    exact hph0
  rw [progOf_at x _ hsx (q := i + 1) (by rw [List.length_map]; omega) hj1 hph, hloc] at hA'
  dsimp only at hA'
  obtain ⟨alg, -, hal, -⟩ := repState_ok hA'
  obtain ⟨hts', ⟨tp, htp⟩, -⟩ := Count.placeholder_tab hph0
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  obtain ⟨hm36, -, -, -⟩ := Count.sched_facts (htsch ▸ hz : _root_.Flock.Zk.schedule _ = .ok (t.sch, y.tPad))
  obtain ⟨-, hdom', -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph0
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  obtain ⟨sch', tp', -, hlay0, hms⟩ := start_eq hds
  have hmask : c'.mask = some mask := by rw [hc']; rfl
  have htm : t.st.m = (Masked.stsOf x)[j]!.m := by rw [hts', hst, hstj]; rfl
  have htk : t.st.fold.kLog = (Masked.stsOf x)[j]!.c.kLog := by rw [hts', hst, hstj]; rfl
  have hlay : c'.lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.fold.kLog sch'.k0 := by
    rw [hc']
    show lay = _
    rw [hlay0, hts']
    rfl
  have hkl : c'.lay.nLc + K_SKIP = (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by
    rw [hc']
    show lay.nLc + K_SKIP = _
    rw [hlay0]
    show _ - 6 + 6 = _
    omega
  have hnM : lay.nMlv = t.st.m - 6 := by rw [hlay0, hts']; rfl
  have hnL : lay.nLc = t.st.fold.kLog - 6 := by rw [hlay0, hts']; rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by rw [hc']; exact ofDraws_order _ _ _
  have hm14t : 14 ≤ t.st.m := by rw [htm, hstj]; exact hm14
  have hk6t : 6 ≤ t.st.fold.kLog := by rw [htk, hstj]; exact y.wf.1.hk6
  have hAA : alg.size = A := by rw [(alg_cnt hm14t hk6t hmb hal).1, htm, htk, hAd]
  obtain ⟨hsz, -⟩ := Sched.rounds_piop hm14t hmb hal
  have hb : (piopV t.st.m t.st.fold.kLog t.st.digest (_root_.Flock.Firewall.Finish.flat t.cm0.cap) Sched.root0
      (_root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0) (fun _ => 0) alg.size).length ≤ c'.order.size := by
    rw [hc', Loop.ofDraws_order, absorbed_size]
    exact le_trans (piopV_length_le lay hm14t hk6t hnM hnL hsz.le) (by omega)
  -- the frame's state there: the algebraic phase's coins at its last position
  rw [← hAA] at hA'
  obtain ⟨CR0, CR, hCR0, hCR, hA'⟩ := repState_args hA'
  have hcs : coinsAt x ((m₀, ρ₀) :: ops) = CR.getD (alg.size - 1) #[] := by
    rw [coinsAt_words x _ ops (q := i + 1) (by omega), hCR, crMap_getD _ _ (by omega),
      show i + 1 - alg.size + (alg.size - 1) + 1 = i + 1 by omega]
  obtain ⟨hA'', hcnt, hon, hd, hl0, hpos, hco, hp0, hp1⟩ := repState_pre_pcs hcc hal hm14t hk6t hmb hdg
    (placeholder_cap hph0) (hdm ▸ hdom') hb hcs hA'
  -- the rank stop there is the rank check
  obtain ⟨hCA, -, -, -⟩ := ofLoop_cfg_place T L₀ real x ρ ((m₀, ρ₀) :: ops) hsx (q := i + 1) (k := 2 * j + r)
    (j := a) (by simp only [List.length_cons]; omega) (by rw [hloc, ha1]) hph0
  have hrf' : rankFails c' m = false := (rankFails_congr m hCA.lay hCA.nPiop hCA.mask).symm.trans hrf
  have hpN : m.pos = c'.nPiop := by
    rw [hpos, hsz, piopV_length_congr t.st.m t.st.fold.kLog (digest' := t.st.digest)
        (cap' := _root_.Flock.Firewall.Finish.flat t.cm0.cap) (root' := Sched.root0) (lay' := lay) (fun _ => 0)
        (fun _ => 0), piopV_length _ _ _ _ _ _ _ hm14t hk6t hnM hnL, hc', Loop.ofDraws_nPiop, absorbed_size]
    omega
  have hm14' : 14 ≤ (Masked.stsOf x)[j]!.m := by rw [hstj]; exact hm14
  have hk6' : 6 ≤ (Masked.stsOf x)[j]!.c.kLog := by rw [hstj]; exact y.wf.1.hk6
  rw [htm, htk] at hA'' hlay
  rw [htm] at hcnt
  have hW : ∀ n < A, CR.getD n #[] =
      wordsAt x ((ops.take ops.length).map fun p => ZKMal.CoinLayer.rd p.2)
        (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + n) := by
    intro n hn'
    rw [List.take_length, hCR, crMap_getD _ _ (by omega)]
    generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) = R0
    rw [show i + 1 - alg.size + n + 1 = R0 + n by omega]
  obtain ⟨P, hP, hPm⟩ := frame_points sch'.k0 hm14' hk6' (fun k => CR.getD k #[]) m.coins (hA'' ▸ hco)
    (hA'' ▸ hcnt) (fun n e => Masked.slotAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.repBase x (2 * j + r) + n) e)
    (fun n hn' e he => by
      rw [hW n (by omega), wordsAt_slot T x ops hl, Masked.repBase_eq hsx]
      · omega
      · omega
      · have := hcnt n (by omega)
        rw [hW n (by omega)] at this
        omega)
  have hrk := rank_of_rankFails hrf' hon hd hl0 hpN hmask (P := P) (by rw [hlay]; exact hP)
  have hok := Rank.maskRankOk_of_maskRank (m.prior ++ P) mask (c'.lay.nLc + K_SKIP) (by rw [hrk]; omega)
  rw [hkl, Rank.words_maskSlot hms, Array.map_append, hjj] at hok
  have hprior : m.prior.map Array.toList =
      if r = 1 then Masked.ptsOpen (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
        (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.stsOf x)[j]! (2 * j) else #[] := by
    rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
    · rw [hp0 (by omega)]
      simp
    · rw [ite_eq_left rfl]
      have hq0 : 2 * j < spec.repStreams.size := by omega
      have hr0 : (roundsOf spec (Masked.stsOf x)).getD (2 * j + 1 - 1) 0 = N + 4 :=
        hrq _ (by omega) (by omega)
      have hst0 := FlockComplete.ZkTables.repStart_succ spec (Masked.stsOf x) (2 * j)
      rw [show (2 * j) / 2 = (2 * j + 1) / 2 by omega, hrR _ rfl] at hst0
      have hs10 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) := by
        unfold FlockComplete.repStartZk; omega
      have hW0 : ∀ n < N + 4, CR0.getD n #[] =
          wordsAt x ((ops.take ops.length).map fun p => ZKMal.CoinLayer.rd p.2)
            (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) + n) := by
        intro n hn'
        rw [List.take_length, hCR0, crMap_getD _ _ (by rw [hr0]; exact hn'), hr0]
        generalize hR0 : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j) = R0
        rw [show i + 1 - alg.size - (N + 4) + n + 1 = R0 + n by omega]
      have hCR0s : alg.size ≤ CR0.size := by
        rw [hCR0, Array.size_map, Array.size_range, hr0]
        omega
      have hcs0 : ∀ n < alg.size, (CR0.getD n #[]).size = cnt (Masked.stsOf x)[j]!.m n := fun n hn' => by
        have h1 := hcnt n hn'
        rw [hW n (by omega), wordsAt_size x ops (by omega) (by omega),
          loopNr_rep hsx (q := 2 * j + 1) (by omega) (by rw [hrq _ (by omega) (by omega)]; omega)] at h1
        rw [hW0 n (by omega), wordsAt_size x ops (by omega) (by omega),
          loopNr_rep hsx (q := 2 * j) hq0 (by rw [hrq _ (by omega) hq0]; omega),
          show (2 * j) / 2 = (2 * j + 1) / 2 by omega]
        exact h1
      have hP0 := prior_toList CR0 hCR0s
      have hne : ((CR0.extract 0 alg.size).foldl (· ++ ·) #[]).isEmpty = false := by
        refine Bool.eq_false_iff.2 fun h => ?_
        rw [Array.isEmpty_iff_size_eq_zero] at h
        have h1 := congrArg List.length hP0
        rw [Array.length_toList, h, show alg.size = (alg.size - 2) + 2 by omega,
          flatR_len (hA'' ▸ hcs0) (alg.size - 2) (by omega)] at h1
        omega
      have hpc := hp1 (by omega) hne
      rw [words_wordBytes, hlay] at hpc
      obtain ⟨P0, hP0', hP0m⟩ := frame_points sch'.k0 hm14' hk6' (fun k => CR0.getD k #[]) _ (hA'' ▸ hP0)
        (hA'' ▸ hcs0) (fun n e => Masked.slotAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
          (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) (Masked.repBase x (2 * j) + n) e)
        (fun n hn' e he => by
          rw [hW0 n (by omega), wordsAt_slot T x ops hl, Masked.repBase_eq hsx]
          · omega
          · omega
          · have := hcs0 n (by omega)
            rw [hW0 n (by omega)] at this
            omega)
      rw [← hA''] at hP0'
      rw [hP0'] at hpc
      obtain rfl : P0 = m.prior := Except.ok.inj hpc
      exact hP0m
  rw [hprior, hPm] at hok
  unfold Masked.RankBad Masked.ptsOpen at hbad
  unfold Masked.ptsOpen at hok
  rw [hok] at hbad
  exact Bool.noConfusion hbad

end LeanRelease

end Claims.ZKServer
