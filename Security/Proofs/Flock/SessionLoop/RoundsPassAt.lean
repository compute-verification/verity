module

public import Proofs.Flock.SessionLoop.RhoUnrefused
public import Proofs.Flock.SessionLoop.ScheduleTotal
public import Proofs.Flock.SessionLoop.FinL0Vals
public import Proofs.Flock.SessionLoop.FinInnerRet
public import Proofs.Flock.SessionLoop.InnerSlot
public import Proofs.Flock.Claims.ZK.FrameReal
public import Proofs.Flock.Claims.Complete.Zk.RepRounds

/-!
# A taken round's rep, table and schedule

`Loop.RoundsPass` reads every history whose last coins the frame takes, live or not. There the round in the device's
place is the located rep's next scheduled round, on the rep's own table (`Loop.realOf`), wherever its schedule so far is
not refused. Without liveness nothing says the frame's check passed a device round there, so the schedule's totality is
the premise `Loop.ScheduleTotal` instead:

* `LeanRelease.coins_taken_of_unrefused`: where the frame takes the last coins of a history, it took those of each of its
  prefixes of at least three rounds (`coins_taken_of_live` without liveness: the coin stop sticks,
  `refusesCoins_of_take`).
* `LeanRelease.rho_words_unrefused`: so a rep's `ρ₀, ρ₁` words among the opened coins are independent where the frame took
  `ρ₁`'s (`rho_unrefused_sched_pf`'s argument from `coins_taken_of_unrefused`).
* `LeanRelease.ofLoop_answered_at`: where the coins after a history reach the frame, the round after it is a rep's past
  its first, whose frame state answered, or its first, whose fresh state is the answer.
* `LeanRelease.ofLoop_sched_placeholder`: there the rep's schedule on its rounds' coins, at its placeholder table, is not
  refused: on no coins it is the algebraic phase's; past them `ScheduleTotal` carries the schedule `repState` read on
  the coins before the last to all of them, each coin of the plan's count (`loopNr_succ`, `Count.plan_sched`) and
  `ρ₀, ρ₁` independent (`rho_words_unrefused`).
* `Loop.realOf_ok`: where `src` samples the tape's level 0 (`Loop.Level0Src`), the rep's own table is answered, of the
  placeholder's shape (`Sched.tabOfLevel0_alike`).
* **`LeanRelease.ofLoop_sched_unrefused`**: so the rep's own table is answered and its schedule so far not refused
  (`Sched.schedSoFar_alike`): `ofLoop_sched_live`'s conclusion, at the loop's own tables, without liveness.
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg Masker)
open _root_.Flock.Firewall.Finish (Tab)
open Loop

theorem rangeMap_getElem! {α : Type} [Inhabited α] {f : ℕ → α} {n i : ℕ} (h : i < n) :
    ((Array.range n).map f)[i]! = f i := by
  rw [getElem!_pos ((Array.range n).map f) i (by simpa using h), Array.getElem_map, Array.getElem_range]

theorem wordsAt_size_lt (x : Msg) (h1 : List (Msg × Msg)) {p : ℕ} (hp : p < h1.length) :
    (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) p).size =
      min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX := by
  have e := wordsAt_size x h1 (s := h1.length) (p := p) hp (Nat.le_refl _)
  rwa [List.take_of_length_le (Nat.le_refl _)] at e

/-- **Where the frame takes a history's last coins it took each prefix's**, from three rounds on: the coin stop sticks
(`refusesCoins_of_take`), and where the frame takes the coins after a prefix, its rep's state answered them
(`coins_taken_of_answered`). -/
theorem coins_taken_of_unrefused {W R : Type} (L : LeanRelease W R) (x : Msg) (ρ : R) (h : List (Msg × Msg))
    (hr : L.refusesCoins x ρ h = false) {i : ℕ} (h3 : 3 ≤ i) (hi : i ≤ h.length) :
    L.opens x ρ (h.take i) = true ∧
      ∃ s, (L.prog x (h.take i)).answer (L.cfgAt x ρ (i - 1)) (L.coinsOf x (h.take i)) = .ok s := by
  have hlen : (h.take i).length = i := by
    rw [List.length_take]
    omega
  have hr' : L.refusesCoins x ρ (h.take i) = false := by
    cases hb : L.refusesCoins x ρ (h.take i) with
    | false => rfl
    | true =>
      rw [L.refusesCoins_of_take x ρ h i hb] at hr
      cases hr
  unfold refusesCoins at hr'
  cases ha : L.answered x ρ (h.take i) with
  | error e =>
    rw [ha] at hr'
    cases hr'
  | ok m =>
    have hc := L.coins_taken_of_answered x ρ (h.take i) (by omega) ha
    rwa [hlen] at hc

/-- **A placeholder's sizes at the session's link**: its link's points are the session's, and its domain and extra claims
fit a `u64`'s count. -/
theorem placeholder_sizes {x : Msg} (hSv : Serves x) {s0 : _} {spec : SessionSpec} {sts : Array Stmt}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, sts)) {k : ℕ} (hk : k < spec.repStreams.size)
    {h1 : List (Msg × Msg)} (hpos : 0 < h1.length) {c' : Cfg} {tb : Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) =
      .ok (c', tb, dom, xs, hi)) :
    (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0).size / sts[k / 2]!.mPts = spec.points ∧
      dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64 := by
  have hls := wordsAt_size_lt x h1 hpos
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec sts hsx) hk
  obtain ⟨lay, mask, -, -, hdm, hxs⟩ := placeholder_ok hph
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  refine ⟨by rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0], by rw [hdm]; exact hdom', ?_⟩
  rw [hxs, Loop.extras_size]
  have := Nat.mul_le_mul_left sts[k / 2]!.regions.size
    ((linkPoints_size_le sts[k / 2]! _).trans (hls ▸ Nat.min_le_right _ _))
  omega

set_option maxHeartbeats 800000 in
/-- **A rep's `ρ₀, ρ₁` words are independent where the frame took `ρ₁`'s coins**: `rho_unrefused_sched_pf`'s argument,
the frame's taking of `ρ₁`'s coins from `coins_taken_of_unrefused`, on the opened words themselves. -/
theorem rho_words_unrefused (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg))
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hN : Masked.rhoSlot x j r + 1 < ops.length) :
    _root_.Flock.Zk.rhoIndependent
      (⟨(wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.rhoSlot x j r))[0]!,
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.rhoSlot x j r))[1]!⟩,
       ⟨(wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.rhoSlot x j r + 1))[0]!,
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (Masked.rhoSlot x j r + 1))[1]!⟩) = true := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hstj : (Masked.stsOf x)[j]! = (Masked.stsOf x)[j] := getElem!_pos _ j hj
  have hm14 : 14 ≤ (Masked.stsOf x)[j].m := by have := (hall j hj).1; omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
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
  obtain ⟨P', hP'⟩ : ∃ P', P' = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! + (y.sch.k0 + 3) :=
    ⟨_, rfl⟩
  obtain ⟨s, hs⟩ : ∃ s, s = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + P' := ⟨_, rfl⟩
  have hbs : Masked.rhoSlot x j r + 1 = s := by
    unfold Masked.rhoSlot
    rw [Masked.repSlot_eq hsx hz]
    omega
  rw [hbs] at hN
  obtain ⟨-, m, ha⟩ := coins_taken_of_unrefused (ofLoop T L₀ real) x ρ ((m₀, ρ₀) :: ops) hr (i := s + 2) (by omega)
    (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer
    ((ofLoop T L₀ real).cfgAt x ρ (s + 1))
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  have hca : coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)])) =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold coinsAt
    simp only [List.tail_cons, List.getLast?_append, List.getLast?_singleton, List.length_append,
      hts, List.length_singleton, Nat.add_sub_cancel]
    exact feedOf_rd _ _
  rw [hca, List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, P') := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) P' (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  have hlay : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay =
      ((startOf x (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1)).1).toOption.map
        (·.1)).getD default := by
    show ((startOf x (repOf x (s + 1))).toOption.map (·.1)).getD default = _
    rw [show s + 1 = s - 1 + 2 by omega, repOf_add_two hsx]
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  rw [hloc] at hlay
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e' | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨CR0, CR, -, hCR, ha⟩ := repState_args ha
  obtain ⟨alg, -, hal, -⟩ := repState_ok ha
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  have hzt : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (t.sch, y.tPad) := htsch ▸ hz
  obtain ⟨hm36, -, -, -⟩ := Count.sched_facts hzt
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := by
    obtain ⟨p0, rest, hpr⟩ : ∃ p0 rest, ops.take s = p0 :: rest := by
      cases h' : ops.take s with
      | nil => rw [h', List.length_nil] at hts; omega
      | cons p0 rest => exact ⟨p0, rest, rfl⟩
    rw [hpr]
    simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hpts : (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0).size /
      (Masked.stsOf x)[(2 * j + r) / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : xs.size = spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : xs.size + 2 < 2 ^ 64 := by
    rw [hxs, Loop.extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[(2 * j + r) / 2]! _).trans (hls ▸ Nat.min_le_right _ _))
    omega
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  have hA : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
      (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have eA := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at eA
    exact eA
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) hzt)
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.Firewall.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)
      t.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [hA, hNd]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hx2 : xs.size + 2 = 2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size := by omega
  have hkk : y.sch.k0 = t.sch.k0 := by rw [htsch]
  have hll : y.sch.levels.size = t.sch.levels.size := by rw [htsch]
  have hOL := Count.openList_length hzt
    (c := (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)) (by omega)
  have hAP : alg.size + (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 =
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! := by
    rw [hOL] at hFO
    omega
  have hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hts']
    show (Masked.stsOf x)[(2 * j + r) / 2]!.merkle = _
    rw [hst]
    exact (hall j hj).2.2
  have htm : t.st.m = (Masked.stsOf x)[j]!.m := by rw [hts', hst, hstj]; rfl
  have hm14t : 14 ≤ t.st.m := by rw [htm, hstj]; exact hm14
  have hzT : _root_.Flock.Zk.schedule t.st.m = .ok (t.sch, y.tPad) := by rw [hts']; exact hzt
  obtain ⟨sch', tp', hz', hlay0, -⟩ := start_eq hds
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at hz'
  have hlayT : lay = _root_.Flock.Zk.PadLayout.new t.st.m t.st.kLog t.sch.k0 := by
    rw [hlay0, hts', ← hz'.1, htsch]
    all_goals rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by rw [hc']; exact ofDraws_order _ _ _
  have hlay' : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay = lay := by
    rw [hlay]
    unfold startOf
    rw [hsx]
    dsimp only
    rw [hds]
    all_goals rfl
  have hord : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).order.size = c'.order.size := by
    rw [hc']
    show (_root_.Flock.Firewall.Frame.absorbed ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay).size = _
    rw [hlay']
    all_goals rfl
  have hCRs : CR.size = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1 := by
    rw [hCR, Array.size_push, Array.size_map, Array.size_range]
    omega
  rw [show P' + 1 = alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 3) + 1 by omega] at ha
  have hR := RhoKeep.repState_rho hcc hal hm14t hdg (placeholder_cap hph) (hdm ▸ hdom') hmk hxb hzT hlayT hc'
    hord hCRs ha
  have hW' : CR.getD (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) #[] =
      wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (s - 1) := by
    rw [hCR, crAt_getD _ _ _ (show alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2) < P' by omega)]
    rw [show s - 1 - P' + (alg.size + (xs.size + 2) + 1 + (t.sch.k0 + 2)) + 1 = s - 1 by omega]
  have hw1 : wordsAt x ((ops.take (s + 1)).map fun p => ZKMal.CoinLayer.rd p.2) s =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold wordsAt
    rw [List.getElem?_map, List.getElem?_take_of_lt (Nat.lt_succ_self s), hd]
    rfl
  rw [hbs, show Masked.rhoSlot x j r = s - 1 by omega,
    ← wordsAt_take x ops (s := s) (p := s - 1) (by omega),
    ← wordsAt_take x ops (s := s + 1) (p := s) (by omega), hw1, ← hW']
  exact hR

/-- **Where the loop's frame takes the coins after a history, its rep answered them**: the round after it is located at
rep stream `k`'s round `j`; past the rep's first round the rep's state after the history answered the coins
(`takeAt`'s same-rep branch), and at its first the frame starts the rep at the fresh state (`freshOf`). -/
theorem ofLoop_answered_at (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (e0 : Msg × Msg) (h1 : List (Msg × Msg)) {m : Masker}
    (hA : (ofLoop T L₀ real).answered x ρ (e0 :: h1) = .ok m)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) {k j : ℕ}
    (hl : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j)) :
    (∃ j', j = j' + 1 ∧ ∃ c cs', (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer c cs' = .ok m) ∨
      (j = 0 ∧ freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m) := by
  have hF : (ofLoop T L₀ real).frameAnswer x ρ (e0 :: h1) = .ok m := by
    unfold answered at hA
    split at hA
    · exact hA
    · cases hA
  rw [ofLoop_frameAnswer] at hF
  have e1 : (e0 :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  rcases j with _ | j'
  · refine Or.inr ⟨rfl, ?_⟩
    rcases takeAt_ok hF with ⟨-, hf⟩ | ⟨hn, hb, -⟩ | ⟨-, -, -, hf⟩
    · exact hf
    · exfalso
      obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by rw [e1] at hn; omega⟩
      rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx] at hb
      have := locate_first (roundsOf spec sts) 0 q' (of_decide_eq_true hb)
      rw [hl] at this
      exact this rfl
    · exact hf
  · refine Or.inl ⟨j', rfl, ?_⟩
    rcases takeAt_ok hF with ⟨hn, -⟩ | ⟨-, -, ha⟩ | ⟨hn, hb, -, -⟩
    · exfalso
      rw [e1] at hn
      obtain rfl : q = 0 := by omega
      have := locate_zero (roundsOf spec sts) 0
      rw [hl] at this
      exact Nat.succ_ne_zero _ this
    · exact ⟨_, _, ha⟩
    · exfalso
      obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by rw [e1] at hn; omega⟩
      have hp := locate_pred (roundsOf spec sts) 0 q' j' (by rw [hl])
      rw [e1, show q' + 1 + 2 - 1 = q' + 2 by omega, repOf_add_two hx, repOf_add_two hx, hp, hl] at hb
      simp at hb

set_option maxHeartbeats 800000 in
/-- **Where the loop's frame takes a history's last coins, the rep's schedule so far is not refused**, at its placeholder
table and salts, the round after the history located at rep stream `k`'s round `j`: at the rep's first round its
schedule on no coins is the algebraic phase's (`rounds_pos`); past it the rep's state after the history answered the
coins (`ofLoop_answered_at`), so its schedule on the coins before the last is not refused (`repState_ok`), and
`ScheduleTotal` carries it to them all: each coin has its plan's count (`loopNr_succ`, `Count.plan_sched`), and the
rep's `ρ₀, ρ₁`, where taken, are independent (`rho_words_unrefused`). -/
theorem ofLoop_sched_placeholder (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg) (hx : Masked.Live x) (hSv : Serves x) (hST : ScheduleTotal x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (m₀ ρ₀ : Msg) (h1 : List (Msg × Msg))
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: h1) = false)
    (hk : (ofLoop T L₀ real).kindAt x (h1.length + 1) = .masked)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) {k j : ℕ}
    (hl : ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q = (k, j)) :
    ∃ c' tb dom xs hi s,
      placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) = .ok (c', tb, dom, xs, hi) ∧
      schedSoFar tb (k % 2) dom xs hi
        ((Array.range j).map fun i => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - j + i + 1)) = .ok s ∧
      k < 2 * sts.size ∧ dom.size < 2 ^ 64 ∧ xs.size + 2 < 2 ^ 64 := by
  have hss : sts = Masked.stsOf x := by unfold Masked.stsOf; rw [hsx]
  obtain ⟨m, hA⟩ : ∃ m, (ofLoop T L₀ real).answered x ρ ((m₀, ρ₀) :: h1) = .ok m := by
    unfold refusesCoins at hr
    cases hA : (ofLoop T L₀ real).answered x ρ ((m₀, ρ₀) :: h1) with
    | error e => rw [hA] at hr; cases hr
    | ok m => exact ⟨m, rfl⟩
  have hne : ((m₀, ρ₀) :: h1).length ≠ lastOf x := (kindAt_masked hk).2.2
  have hq2 : ((m₀, ρ₀) :: h1).length = q + 2 := by simp only [List.length_cons, hq]
  have hqs := loop_round_in_plan T (withFrames T L₀ real) x ρ ((m₀, ρ₀) :: h1) hA hne hsx hq2
  have hcs : (h1.map fun p => ZKMal.CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = k := by rw [hl]
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 = j := by rw [hl]
  obtain ⟨-, -, hspec, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  obtain ⟨hpts2, -, hreg⟩ := Masked.live_bounds hx hsx
  have hkr : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec sts) 0 q hqs
    rw [hl] at this
    simpa [roundsOf] using this
  have hrsl : (roundsOf spec sts).length = spec.repStreams.size := by simp [roundsOf]
  have hk2s : k < 2 * sts.size := by
    have := hkr
    rw [FlockDeploy.repStreams_size, hspec] at this
    exact this
  have hkj2 : k / 2 < sts.size := by omega
  have hstk : sts[k / 2]! = sts[k / 2] := getElem!_pos sts (k / 2) hkj2
  obtain ⟨hB, hjlt⟩ := locate_inv (roundsOf spec sts) 0 q hqs
  rw [hk1, hj1, Nat.sub_zero] at hB hjlt
  have hans := ofLoop_answered_at T L₀ real x ρ (m₀, ρ₀) h1 hA hsx hq hl
  rcases j with _ | j'
  · have hm : freshOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2) = .ok m := by
      rcases hans with ⟨j', hj', -⟩ | ⟨-, hm⟩
      · exact absurd hj' (by omega)
      · exact hm
    obtain ⟨c', tb, dom, xs, hi, alg, hph, hal, -⟩ := freshOf_ok x _ hsx hcs hm
    rw [hk1] at hph
    obtain ⟨-, hdom, hxb⟩ := placeholder_sizes hSv hsx hkr (by omega) hph
    refine ⟨c', tb, dom, xs, hi, alg, hph, ?_, hk2s, hdom, hxb⟩
    rw [range_zero_map]
    unfold schedSoFar
    rw [hal]
    simp only [bind, Except.bind, Array.size_empty, rounds_pos hal, ↓reduceIte, pure, Except.pure]
  · obtain ⟨c0, cs0, ha⟩ : ∃ c cs', (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).answer c cs' = .ok m := by
      rcases hans with ⟨j'', -, ha⟩ | ⟨hj0, -⟩
      · exact ha
      · exact absurd hj0 (by omega)
    obtain ⟨q', rfl⟩ : ∃ q', q = q' + 1 := ⟨q - 1, by omega⟩
    have ho : (progOf x (h1.map fun p => ZKMal.CoinLayer.rd p.2)).owed ≠ none := fun ho => answer_owed_none _ ho ha
    have hp := locate_pred (roundsOf spec sts) 0 q' j' (by rw [hl])
    obtain ⟨c', tb, dom, xs, hi, hph⟩ := progOf_placeholder x _ hsx (by rw [hcs]) ho
    rw [hp, hl] at hph
    have hph' : placeholder spec sts k (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0) =
        .ok (c', tb, dom, xs, hi) := hph
    obtain ⟨hpts, hdom, hxb⟩ := placeholder_sizes hSv hsx hkr (by omega) hph'
    have hpa := progOf_at x _ hsx hcs (j := j') (c := c') (t := tb) (dom := dom) (xs := xs) (hi := hi)
      (by rw [hl]) (by rw [hk1]; exact hph')
    rw [hk1, hj1] at hpa
    rw [hpa] at ha
    obtain ⟨-, sched, -, hpop⟩ := repState_ok ha
    generalize hCR : (Array.range (j' + 1)).map
      (fun i => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q' + 1 - (j' + 1) + i + 1)) = CR at hpop ⊢
    have hCRs : CR.size = j' + 1 := by rw [← hCR, Array.size_map, Array.size_range]
    obtain ⟨-, ⟨tp, htp⟩, -⟩ := Count.placeholder_tab hph'
    have h25 : 25 ≤ sts[k / 2]!.m := by rw [hstk]; exact (hall _ hkj2).1
    have h35 : sts[k / 2]!.m ≤ 35 := by rw [hstk]; exact (hall _ hkj2).2.1
    obtain ⟨hn6, hq6⟩ := FlockComplete.JsonRT.zkSchedule_small h25 h35 htp
    have hc65 : 2 + spec.points * sts[k / 2]!.regions.size ≤ 65536 := by
      have := hreg (k / 2) hkj2
      rw [← hstk] at this
      rw [hpts2]
      omega
    have hcn := placeholder_cap hph'
    have hcnt : ∀ i (h : i < sched.size), i < CR.size → (CR[i]!).size = sched[i].2 := by
      intro i hi' hic
      have hij : i < j' + 1 := by omega
      have hiq : i < (roundsOf spec sts).getD k 0 := by omega
      have hloc := locate_take_sum (roundsOf spec sts) 0 k i (by rw [hrsl]; exact hkr) hiq
      rw [Nat.zero_add] at hloc
      have hidx : q' + 1 - (j' + 1) + i + 1 = ((roundsOf spec sts).take k).sum + i + 1 := by omega
      rw [← hCR, rangeMap_getElem! hij, hidx,
        wordsAt_size_lt x h1 (p := ((roundsOf spec sts).take k).sum + i + 1) (by omega),
        loopNr_succ hsx hloc (by omega),
        Nat.min_eq_left (Loop.repPlan_le htp h35 hn6 hq6 hc65 i),
        Count.plan_sched spec sts k i _ c' tb dom xs hi CR.pop sched hph' hpop hi' hpts hdom hcn hxb]
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hi', Option.getD_some]
    have hrho : ∀ I, I = FlockComplete.piopRoundsZk spec sts[k / 2]! + tb.sch.k0 + 2 → I + 1 < CR.size →
        _root_.Flock.Zk.rhoIndependent (⟨(CR[I]!)[0]!, (CR[I]!)[1]!⟩, ⟨(CR[I + 1]!)[0]!, (CR[I + 1]!)[1]!⟩) =
          true := by
      intro I hI hIc
      rw [← hCR, rangeMap_getElem! (by omega : I < j' + 1), rangeMap_getElem! (by omega : I + 1 < j' + 1)]
      subst hss
      have hslot : Masked.rhoSlot x (k / 2) (k % 2) = q' + 1 - (j' + 1) + I + 1 := by
        unfold Masked.rhoSlot
        rw [Nat.div_add_mod k 2, Masked.repSlot_eq hsx htp, repStartZk_eq spec _ hkr.le]
        omega
      have hR := rho_words_unrefused T L₀ real x hx hSv ρ m₀ ρ₀ h1 hr (j := k / 2) (r := k % 2) hkj2
        (Nat.mod_lt _ (by decide)) (by rw [hslot]; omega)
      rw [hslot, show q' + 1 - (j' + 1) + I + 1 + 1 = q' + 1 - (j' + 1) + (I + 1) + 1 by omega] at hR
      exact hR
    obtain ⟨s, hs⟩ := hST s0 spec sts hsx k hk2s _ hpts c' tb dom xs hi hph' CR sched hpop hcnt (hrho _ rfl)
    exact ⟨c', tb, dom, xs, hi, s, hph', hs, hk2s, hdom, hxb⟩

end Claims.ZKServer.LeanRelease

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.Firewall.Frame (Cfg)
open _root_.Flock.Firewall.Finish (Tab)

/-- **The loop's own table is answered** at a rep stream of the session, where `src` samples the tape's level 0
(`Level0Src`): its level 0 is sampled, and `tabOfLevel0` of it is of the placeholder's shape, which is answered
(`Sched.tabOfLevel0_alike`). -/
theorem realOf_ok {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)}
    {x : Msg} {w : Flock.Wit} (hx : Masked.Live x) (hS0 : Level0Src src x w) (ν : finS.Nonce) (t : finS.Tape)
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {k : ℕ} (hk : k < 2 * (Masked.stsOf x).size) {link : Array F128} {c' : Cfg} {tb : Tab} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray}
    (hph : placeholder spec (Masked.stsOf x) k link = .ok (c', tb, dom, xs, hi)) :
    ∃ tr his, realOf src x w (ν, t) k = .ok (tr, his) := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    exact ite_eq_left hx
  obtain ⟨l, hl, -⟩ := hS0 ν t ⟨k / 2, by rw [hJ]; omega⟩ ⟨k % 2, Nat.mod_lt _ (by decide)⟩
  have hl' : level0At src x w (ν, t) (Masked.stsOf x) (2 * (k / 2) + k % 2) = .ok l := hl
  rw [Nat.div_add_mod k 2] at hl'
  unfold level0At at hl'
  rcases hs : src x w (ν, t) k with e | ⟨g, d, his⟩
  · rw [hs] at hl'
    cases hl'
  rw [hs] at hl'
  simp only [bind, Except.bind] at hl'
  unfold placeholder at hph
  obtain ⟨⟨lay, mask⟩, -, hph⟩ := bok hph
  obtain ⟨⟨sch, tPad⟩, -, hph⟩ := bok hph
  obtain ⟨l0, hl0, hph⟩ := bok hph
  obtain ⟨t0, ht0, -⟩ := bok hph
  obtain ⟨sg, tpg, hsg, hg⟩ := Count.ofDraws_gsize hl'
  obtain ⟨sg', tpg', hsg', hg'⟩ := Count.ofDraws_gsize hl0
  rw [hsg] at hsg'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hsg')
  obtain ⟨-, -, ⟨s1, hs1, hlv⟩, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok hl'
  rw [hsg] at hs1
  obtain ⟨rfl, -⟩ := Prod.mk.inj (Except.ok.inj hs1)
  have hd : 64 * 2 ^ l.lv.d < 2 ^ 64 := by
    rw [hlv]
    exact (Count.sched_bounds hsg).2 0
  have hA := Sched.tabOfLevel0_alike (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!) hl' hl0
    (by rw [hg, hg']) hd
  rw [ht0] at hA
  cases htr : _root_.Flock.Firewall.Finish.tabOfLevel0 (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!) l with
  | error e =>
    rw [htr] at hA
    exact (hA : False).elim
  | ok tr =>
    refine ⟨tr, his, ?_⟩
    unfold realOf
    rw [hsx]
    dsimp only
    simp only [hs, hl', htr, bind, Except.bind, pure, Except.pure]

end Claims.ZKServer.Loop

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open Loop

/-- **Where the loop's frame takes a history's last coins, the round's rep has its own table and schedule**, at the loop's
own tables (`Loop.realOf`) where `src` samples the tape's level 0: `ofLoop_sched_live`'s conclusion without liveness. The
rep's schedule at its placeholder table is not refused (`ofLoop_sched_placeholder`), its own table is answered
(`Loop.realOf_ok`) and of the placeholder's shape (`Count.real_tabAlike`), so its schedule there is not refused either
(`Sched.schedSoFar_alike`). -/
theorem ofLoop_sched_unrefused (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x) (hS0 : Loop.Level0Src src x w)
    (hST : Loop.ScheduleTotal x) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (m₀ ρ₀ : Msg) (h1 : List (Msg × Msg))
    (hr : (ofLoop Masked.dataE L₀ (Loop.realOf src)).refusesCoins x (ν, t) ((m₀, ρ₀) :: h1) = false)
    (hk : (ofLoop Masked.dataE L₀ (Loop.realOf src)).kindAt x (h1.length + 1) = .masked)
    {s0 : _} {spec : SessionSpec} {sts : Array Stmt} {q : ℕ}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : h1.length = q + 1) :
    ∃ tr hi sched, Loop.realOf src x w (ν, t) (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 = .ok (tr, hi) ∧
      schedSoFar tr ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)
        (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!)
          ((ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
          (_root_.Flock.Firewall.Drive.linkPoints sts[(ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).1 / 2]!
            (wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi
        ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2).map
          fun j => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2)
            (q - (ZKMal.CoinLayer.locate (roundsOf spec sts) 0 q).2 + j + 1)) = .ok sched := by
  have hss : sts = Masked.stsOf x := by unfold Masked.stsOf; rw [hsx]
  subst hss
  obtain ⟨k, j, hl⟩ : ∃ k j, ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (k, j) := ⟨_, _, rfl⟩
  have hk1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).1 = k := by rw [hl]
  have hj1 : (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q).2 = j := by rw [hl]
  rw [hk1, hj1]
  obtain ⟨c', tb, dom, xs, hi, s, hph, hsp, hk2s, hdom, hxs⟩ :=
    ofLoop_sched_placeholder Masked.dataE L₀ (Loop.realOf src) x hx hSv hST (ν, t) m₀ ρ₀ h1 hr hk hsx hq hl
  obtain ⟨lay, mask, -, -, hdm, hxe⟩ := placeholder_ok hph
  obtain ⟨tr, his, hre⟩ := Loop.realOf_ok hx hS0 ν t hsx hk2s hph
  have hT := Count.real_tabAlike (Loop.realTabs_of src x w (ν, t) hsx) hph hre
  obtain ⟨hdg, hmb, hn, hlq, hp, hd⟩ := Count.real_bounds hph hT
  have hA := Sched.schedSoFar_alike tr tb (k % 2) dom xs his hi
    ((Array.range j).map fun i => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - j + i + 1))
    hT.1 hT.2.1 hT.2.2.1 hT.2.2.2.1 hT.2.2.2.2 hdom hdg hmb hxs hn hlq hp hd
  rw [hsp] at hA
  cases hre' : schedSoFar tr (k % 2) dom xs his
      ((Array.range j).map fun i => wordsAt x (h1.map fun p => ZKMal.CoinLayer.rd p.2) (q - j + i + 1)) with
  | error e =>
    rw [hre'] at hA
    exact (hA : False).elim
  | ok s' =>
    refine ⟨tr, his, s', hre, ?_⟩
    rw [← hdm, ← hxe]
    exact hre'

end Claims.ZKServer.LeanRelease
