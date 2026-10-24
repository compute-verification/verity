import Proofs.Flock.Claims.ZK.L0PinChain
import Proofs.Flock.Claims.ZK.FrameRho
import Proofs.Flock.Claims.ZK.FrameReal
import Proofs.Flock.Claims.ZK.FrameInner
import Proofs.Flock.Claims.ZK.FrameOuter

/-!
# Level 0 at the session loop

`L0PinChain`'s `l0_chain` at the loop's release (`ofLoop`): on a live history of `refK x` openings whose last coins the
frame takes, at a rep's binding round `J` (past its algebraic phase and ring switch), the sent rounds from `J` through
level 0's are the binding and level 0's bodies at the rep's own table and draws (**`LeanRelease.ofLoop_l0`**).

* The loop's counts there: `z`'s `n` (under `64` at every `--zk` schedule, `Count.sched_n`), then level 0's `1, 1, 2, …`
  (`Count.openList_l0`, `Count.repPlan_l0`).
* The rep's schedules at the round's prefixes agree where both reach (`Loop.sched_mono`), and read the opened coins
  (`Loop.wordsAt_take`).
* The bodies' coins are the opened coins from the binding's (`Loop.l0Body_clean`).
-/

namespace Claims.ZKServer.Count

/-- Every `--zk` schedule's `n` (the packed table's variables, `m − 7`) is under `64`. -/
def nOk (m : ℕ) : Bool :=
  match _root_.Flock.Zk.schedule m with
  | .ok (sch, _) => decide (sch.n < 64)
  | .error _ => true

/-- Every `--zk` schedule, `m < 36`, by the kernel's evaluation. -/
theorem nOk_lt : ∀ m, m < 36 → nOk m = true := by decide +kernel

theorem sched_n {m tp : ℕ} {sch : _root_.Flock.Schedule} (h : _root_.Flock.Zk.schedule m = .ok (sch, tp)) :
    sch.n < 64 := by
  have hok := nOk_lt m (sched_facts h).1
  unfold nOk at hok
  rw [h] at hok
  simpa only [decide_eq_true_eq] using hok

theorem rep2_get {k0 e : ℕ} (rest : List ℕ) (he : e ≤ k0) : (List.replicate k0 2 ++ 2 :: rest)[e]? = some 2 := by
  rcases Nat.lt_or_ge e k0 with h | h
  · rw [List.getElem?_append_left (by simpa using h)]
    simp [h]
  · rw [List.getElem?_append_right (by simp only [List.length_replicate]; omega), List.length_replicate,
      show e - k0 = 0 by omega]
    rfl

/-- **The opening's counts at level 0**: past the ring switch's `N + 1` rounds, `z`'s `n` and then level 0's
(`l0Nc`). -/
theorem openList_l0 (sch : _root_.Flock.Schedule) {N : ℕ} (hN : N ≠ 0) (hn0 : sch.n ≠ 0) {i : ℕ}
    (hi : i ≤ sch.k0 + 2) :
    (openList N sch)[N + 1 + i]? = some (if i = 0 then sch.n else l0Nc i) := by
  obtain ⟨rest, hr⟩ := Bind.ligeritoCoins_l0 sch
  have hf : (sch.n :: 1 :: (List.replicate sch.k0 2 ++ 2 :: 2 :: rest)).filter (· != 0) =
      sch.n :: 1 :: (List.replicate sch.k0 2 ++ 2 :: 2 :: rest.filter (· != 0)) := by
    simp [List.filter_cons, List.filter_append, List.filter_replicate, hn0]
  rw [Bind.openList_eq sch hN, hr, hf, List.getElem?_append_right (by simp only [List.length_replicate]; omega),
    List.length_replicate, show N + 1 + i - N = i + 1 by omega, List.getElem?_cons_succ]
  rcases i with _ | _ | e
  · rfl
  · rfl
  · rw [List.getElem?_cons_succ, List.getElem?_cons_succ, rep2_get _ (by omega)]
    simp [l0Nc]

/-- **A rep's plan at its binding and level 0**: `z`'s `n`, then level 0's counts. -/
theorem repPlan_l0 {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) {i : ℕ} (hi : i ≤ sch.k0 + 2) :
    (_root_.Flock.Firewall.Session.repPlan spec st).getD
      (((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)).length +
        ((2 + spec.points * st.regions.size) + 1 + i)) 0 = if i = 0 then sch.n else l0Nc i := by
  have hn0 : sch.n ≠ 0 := by have := (sched_rho h).1; omega
  have hlt : ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)).length +
      ((2 + spec.points * st.regions.size) + 1 + i) <
      (((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)) ++
        openList (2 + spec.points * st.regions.size) sch).length := by
    rw [List.length_append, openList_length h (by omega)]
    omega
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, repPlan_list h, List.getElem?_append_left hlt,
    List.getElem?_append_right (Nat.le_add_right _ _), Nat.add_sub_cancel_left, openList_l0 sch (by omega) hn0 hi]
  rfl

theorem flatten_ones (n : ℕ) : (List.replicate n [1]).flatten = List.replicate n 1 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [List.replicate_succ, List.flatten_cons, ih]; rfl

/-- **The algebraic phase's counts**: the zerocheck's `6`, `m − 13` and `1`, one for each of its `m − 6` residual rounds,
the lincheck's `1, 1`, one for each of its `k_log − 6` rounds, and `r'`'s `1`. -/
theorem piop_list {m kLog : ℕ} (hm : 14 ≤ m) :
    (_root_.Flock.Firewall.Session.replayCoins m kLog).toList.filter (· != 0) =
      6 :: (m - 13) :: 1 :: (List.replicate (m - 6) 1 ++ 1 :: 1 :: (List.replicate (kLog - 6) 1 ++ [1])) := by
  rw [← piop_counts]
  simp [zcCounts, lcCounts, sq, show m - 13 ≠ 0 by omega, flatten_ones]

theorem plan_get {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) {p : ℕ}
    (hp : p < ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
      openList (2 + spec.points * st.regions.size) sch).length) :
    (_root_.Flock.Firewall.Session.repPlan spec st).getD p 0 =
      (((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
        openList (2 + spec.points * st.regions.size) sch)[p]?).getD 0 := by
  rw [Array.getD_eq_getD_getElem?, ← Array.getElem?_toList, repPlan_list h, List.getElem?_append_left hp]

/-- **A rep's plan in its PIOP and ring switch**: the zerocheck's `r_outer` round squeezes `m − 13` coins, its residual
rounds and the lincheck's rounds one each, then each of the `N` claims' ring switch `7` and their batching `N`. -/
theorem repPlan_piop {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} {sch : _root_.Flock.Schedule} {tp : ℕ}
    (h : _root_.Flock.Zk.schedule st.m = .ok (sch, tp)) (hm : 14 ≤ st.m) :
    (_root_.Flock.Firewall.Session.repPlan spec st).getD 1 0 = st.m - 13 ∧
      (∀ i < st.m - 6, (_root_.Flock.Firewall.Session.repPlan spec st).getD (3 + i) 0 = 1) ∧
      (∀ i < st.c.kLog - 6, (_root_.Flock.Firewall.Session.repPlan spec st).getD (3 + (st.m - 6) + 2 + i) 0 = 1) ∧
      (∀ k < 2 + spec.points * st.regions.size,
        (_root_.Flock.Firewall.Session.repPlan spec st).getD (3 + (st.m - 6) + 3 + (st.c.kLog - 6) + k) 0 = 7) ∧
      (_root_.Flock.Firewall.Session.repPlan spec st).getD
        (3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + spec.points * st.regions.size)) 0 =
        2 + spec.points * st.regions.size := by
  have hL := piop_list (kLog := st.c.kLog) hm
  have hlen : ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0)).length =
      3 + (st.m - 6) + 3 + (st.c.kLog - 6) := by
    rw [hL]
    simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil]
    omega
  have hOL := openList_length h (c := 2 + spec.points * st.regions.size) (by omega)
  have hall : ∀ p, p ≤ 3 + (st.m - 6) + 3 + (st.c.kLog - 6) + (2 + spec.points * st.regions.size) →
      p < ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
        openList (2 + spec.points * st.regions.size) sch).length := by
    intro p hp
    rw [List.length_append, hlen, hOL]
    omega
  have hleft : ∀ p, p < 3 + (st.m - 6) + 3 + (st.c.kLog - 6) →
      ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
        openList (2 + spec.points * st.regions.size) sch)[p]? =
      (6 :: (st.m - 13) :: 1 ::
        (List.replicate (st.m - 6) 1 ++ 1 :: 1 :: (List.replicate (st.c.kLog - 6) 1 ++ [1])))[p]? := by
    intro p hp
    rw [List.getElem?_append_left (by rw [hlen]; exact hp), hL]
  have hright : ∀ k, ((_root_.Flock.Firewall.Session.replayCoins st.m st.c.kLog).toList.filter (· != 0) ++
        openList (2 + spec.points * st.regions.size) sch)[3 + (st.m - 6) + 3 + (st.c.kLog - 6) + k]? =
      (openList (2 + spec.points * st.regions.size) sch)[k]? := by
    intro k
    rw [List.getElem?_append_right (by rw [hlen]; omega), hlen, Nat.add_sub_cancel_left]
  refine ⟨?_, fun i hi => ?_, fun i hi => ?_, fun k hk => ?_, ?_⟩
  · rw [plan_get h (hall 1 (by omega)), hleft 1 (by omega)]
    rfl
  · rw [plan_get h (hall _ (by omega)), hleft _ (by omega), show 3 + i = i + 1 + 1 + 1 by omega,
      List.getElem?_cons_succ, List.getElem?_cons_succ, List.getElem?_cons_succ,
      List.getElem?_append_left (by simpa using hi)]
    simp [hi]
  · rw [plan_get h (hall _ (by omega)), hleft _ (by omega),
      show 3 + (st.m - 6) + 2 + i = ((st.m - 6) + (i + 1 + 1)) + 1 + 1 + 1 by omega,
      List.getElem?_cons_succ, List.getElem?_cons_succ, List.getElem?_cons_succ,
      List.getElem?_append_right (by simp), List.length_replicate, Nat.add_sub_cancel_left,
      List.getElem?_cons_succ, List.getElem?_cons_succ, List.getElem?_append_left (by simpa using hi)]
    simp [hi]
  · rw [plan_get h (hall _ (by omega)), hright, ← List.getD_eq_getElem?_getD]
    exact Bind.openList_lt sch (by omega) hk
  · rw [plan_get h (hall _ le_rfl), hright, ← List.getD_eq_getElem?_getD]
    exact Bind.openList_at sch (by omega)

end Claims.ZKServer.Count

namespace Claims.ZKServer.Loop

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

/-- The words a prefix of the openings reads at an earlier round are the openings'. -/
theorem wordsAt_take (x : Msg) (ops : List (Msg × Msg)) {s p : ℕ} (hp : p < s) :
    wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) p =
      wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) p := by
  unfold wordsAt
  rw [List.getElem?_map, List.getElem?_map, List.getElem?_take_of_lt hp]

theorem rangeMap_congr {α : Type} {n : ℕ} {f g : ℕ → α} (h : ∀ i, i < n → f i = g i) :
    (Array.range n).map f = (Array.range n).map g := by
  apply Array.ext (by simp)
  intro i h1 _
  simp only [Array.size_map, Array.size_range] at h1
  simp [h i h1]

/-- A rep's schedule so far reads its table's algebraic rounds. -/
theorem schedSoFar_rounds {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {sched : Array (ByteArray × ℕ)}
    (h : schedSoFar t rep dom xs hi cr = .ok sched) :
    ∃ alg, _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg := by
  unfold schedSoFar at h
  obtain ⟨alg, hal, -⟩ := bindV_ok h
  exact ⟨alg, hal⟩

/-- **A rep's schedules so far agree where both reach**: fed the coins of `a` rounds and of `b ≥ a` rounds, every
schedule between answered, the shorter's rounds are the longer's (`schedSoFar_pop_getD`, one round at a time). -/
theorem sched_mono {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} (f : ℕ → Array F128) {a : ℕ} {sa : Array (ByteArray × ℕ)}
    (hsa : schedSoFar t rep dom xs hi ((Array.range a).map f) = .ok sa) {k : ℕ} (hk : k < sa.size) :
    ∀ b, a ≤ b → (∀ n, a ≤ n → n ≤ b → ∃ s, schedSoFar t rep dom xs hi ((Array.range n).map f) = .ok s) →
      ∀ sb, schedSoFar t rep dom xs hi ((Array.range b).map f) = .ok sb →
        sa.getD k (.empty, 0) = sb.getD k (.empty, 0) ∧ k < sb.size := by
  intro b hab
  induction b, hab using Nat.le_induction with
  | base =>
    intro _ sb hsb
    rw [hsa] at hsb
    cases hsb
    exact ⟨rfl, hk⟩
  | succ b hab ih =>
    intro hall sb hsb
    obtain ⟨s', hs'⟩ := hall b hab (by omega)
    obtain ⟨e1, e2⟩ := ih (fun n h1 h2 => hall n h1 (by omega)) s' hs'
    have hpop : ((Array.range (b + 1)).map f).pop = (Array.range b).map f := by
      rw [Array.range_succ, Array.append_singleton, Array.map_push, Array.pop_push]
    rw [← hpop] at hs'
    obtain ⟨e3, e4⟩ := schedSoFar_pop_getD hs' hsb (by simp) e2
    exact ⟨e1.trans e3, e4⟩

/-- **A rep stream's start fixes its layout**: `Drive.start` lays the pads out at its statement's schedule. -/
theorem start_lay {st : Stmt} {lay : _root_.Flock.Zk.PadLayout} {mask : _root_.Flock.Firewall.Frame.MaskSlot}
    (h : _root_.Flock.Firewall.Drive.start st = .ok (lay, mask)) :
    ∃ sch tp, _root_.Flock.Zk.schedule st.m = .ok (sch, tp) ∧
      lay = _root_.Flock.Zk.PadLayout.new st.m st.c.kLog sch.k0 := by
  unfold _root_.Flock.Firewall.Drive.start at h
  obtain ⟨⟨sch, tp⟩, hs, h⟩ := bindV_ok h
  obtain ⟨ms, -, h⟩ := bindV_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  exact ⟨sch, tp, hs, h.1.symm⟩

theorem crN_extract_getD (f : ℕ → Array F128) {A C p : ℕ} (h : A + p < C) :
    (((Array.range C).map f).extract A ((Array.range C).map f).size).toList.getD p #[] = f (A + p) := by
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getElem?_extract]
  split
  · simp [h]
  · rename_i hc
    exfalso
    simp only [Array.size_map, Array.size_range, Nat.min_self] at hc
    omega

theorem coin_raw (W : ℕ → Array F128) {R A C p e : ℕ} (h : A + p < C) :
    ((((Array.range C).map fun q => W (R + q)).extract A ((Array.range C).map fun q => W (R + q)).size).toList.getD
      p #[])[e]! = (W (R + (A + p)))[e]! := by
  rw [crN_extract_getD _ h]

theorem rec_raw (W : ℕ → Array F128) {R A C k S : ℕ} (hk : k < A) (hA : A ≤ C) (hS : R + k = S) (e : ℕ) :
    FlockComplete.LigLock.Fed.recCoin (((Array.range C).map fun q => W (R + q)).extract 0 A) k e = (W S)[e]! := by
  unfold FlockComplete.LigLock.Fed.recCoin
  rw [List.getD_eq_getElem?_getD, Array.getElem?_toList, Array.getElem?_extract]
  split
  · subst hS
    simp [show k < C by omega]
  · rename_i hc
    exfalso
    simp only [Array.size_map, Array.size_range] at hc
    omega

/-- The algebraic phase's points at the recorded coins are its points at the opened coins from `R` on. -/
theorem pts_raw (W : ℕ → Array F128) {m kLog R A C : ℕ} {zc : _root_.Flock.ZcOut}
    {lc : _root_.Flock.LcOut} (hA : 3 + (m - 6) + 3 + (kLog - 6) ≤ A) (hC : A ≤ C)
    (h : FlockComplete.LigLock.Fed.PtsAt m kLog
      (FlockComplete.LigLock.Fed.recCoin (((Array.range C).map fun q => W (R + q)).extract 0 A)) 0 zc lc) :
    FlockComplete.LigLock.Fed.PtsAt m kLog (fun p e => (W p)[e]!) R zc lc :=
  FlockComplete.LigLock.Fed.ptsAt_congr (fun i _ => rec_raw W (by omega) hC (by omega) i)
    (fun i _ => rec_raw W (by omega) hC (by omega) 0) (fun i _ => rec_raw W (by omega) hC (by omega) 0) h

/-- The ring switch at the schedule's coins past the algebraic phase `A` is the one at the opened coins. -/
theorem sws_raw (W : ℕ → Array F128) {R A C N L S : ℕ} (cl : Array _root_.Flock.Claim) (hN : cl.size = N)
    (hC : A + N < C) (hL : R + A = L) (hS : R + A + N = S) :
    FlockComplete.Replay.Rs.sws cl
        ⟨fun k i => ((((Array.range C).map fun q => W (R + q)).extract A
          ((Array.range C).map fun q => W (R + q)).size).toList.getD k #[])[i]!,
        fun i => ((((Array.range C).map fun q => W (R + q)).extract A
          ((Array.range C).map fun q => W (R + q)).size).toList.getD N #[])[i]!⟩ =
      FlockComplete.Replay.Rs.sws cl ⟨fun k e => (W (L + k))[e]!, fun e => (W S)[e]!⟩ := by
  apply FlockComplete.LigLock.Fed.sws_coins_congr
  · intro k hk i _
    rw [hN] at hk
    exact (coin_raw W (R := R) (e := i) (show A + k < C by omega)).trans
      (by show _ = (W (L + k))[i]!; rw [← hL, Nat.add_assoc])
  · intro k _
    exact (coin_raw W (R := R) (e := k) hC).trans (by show _ = (W S)[k]!; rw [← hS, Nat.add_assoc])

/-- **Level 0's bodies at the schedule's coins are its bodies at the opened coins** from the binding's on: the
schedule reads round `p`'s coins at the `p`-th of the rounds past the algebraic phase's `A`. -/
theorem l0Body_clean (W : ℕ → Array F128) {R A N C S : ℕ} {P : List UInt8} {T : F128} {cap : ByteArray}
    {sch : _root_.Flock.Schedule} {t b : Array F128} {lay : _root_.Flock.Zk.PadLayout} {h : Array F128} {rep : ℕ}
    {extra : ℕ → ℕ → F128} {nonce : ℕ → ℕ → UInt64} (hC : A + N + 1 + (sch.k0 + 2) ≤ C)
    (hS : R + A + N + 1 = S) (i : ℕ) :
    FlockComplete.LigLock.l0BodyAt
        (fun j i => ((((Array.range C).map fun q => W (R + q)).extract A
          ((Array.range C).map fun q => W (R + q)).size).toList.getD j #[])[i]!)
        (N + 1) P T cap sch t b lay h rep extra nonce i =
      FlockComplete.LigLock.l0BodyAt (fun j i => (W (S + j))[i]!) 0 P T cap sch t b lay h rep extra nonce i := by
  apply FlockComplete.LigLock.Fed.l0BodyAt_congr
  · intro e _
    exact (coin_raw W (show A + (N + 1) < C by omega)).trans
      (by rw [show R + (A + (N + 1)) = S + 0 by omega])
  · exact (coin_raw W (show A + (N + 1 + 1) < C by omega)).trans
      (by rw [show R + (A + (N + 1 + 1)) = S + (0 + 1) by omega])
  · intro j hj e _
    exact (coin_raw W (show A + (N + 1 + 2 + j) < C by omega)).trans
      (by rw [show R + (A + (N + 1 + 2 + j)) = S + (0 + 2 + j) by omega])

end Claims.ZKServer.Loop

namespace Claims.ZKServer.LeanRelease

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open FlockSoundness.Refine (Frame encs)
open Loop

/-- A live history at a masked round has the frame's coins answered. -/
theorem answered_ok_of_live {W R : Type} (L : LeanRelease W R) (wk : L.toSrv.Worker) (x : Msg) (w : W) (ρ : R)
    (κ : L.toSrv.Chk) (h : List (Msg × Msg)) (hk : L.kindAt x h.length = .masked)
    (hl : L.toSrv.live wk x w ρ κ h = true) : ∃ m, L.answered x ρ h = .ok m := by
  obtain ⟨v, -, -, hp, -⟩ := L.live_some wk x w ρ κ h hk hl
  unfold pass at hp
  cases hA : L.answered x ρ h with
  | error e => rw [hA] at hp; cases hp
  | ok m => exact ⟨m, rfl⟩

set_option maxHeartbeats 2000000 in
/-- **The loop's level-0 messages**: at a `Live` statement the loop serves, on a live history of `refK x` openings
whose last coins the frame takes, whose messages are the release's own and whose tables are the loop's (`RealTabs`),
table `j`'s rep `r`'s binding round `J` (past its algebraic phase and ring switch) sends the binding at its table's cap
and `z`'s squeeze of `n`, and its level-0 rounds `J + i` send level 0's bodies at its table, its draws' pads and the
opened coins from the binding's on, each tagged `0`. -/
theorem ofLoop_l0 (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hrf : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    (hown : (ofLoop T L₀ real).toSrv.OwnHist (ofLoop T L₀ real).fillOf wk x w ρ κ ((m₀, ρ₀) :: ops))
    (hP : (ofLoop T L₀ real).ProgChecks x w) (hR : Count.RealTabs (Masked.stsOf x) (real x w ρ))
    {s0 : _} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {j r : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) {J : ℕ}
    (hJ : J = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! + 1) :
    ∃ tr hr' T₀ T₁ sws, real x w ρ (2 * j + r) = .ok (tr, hr') ∧ J + tr.sch.k0 + 2 ≤ ops.length ∧
      ((m₀, ρ₀) :: ops)[J]?.map Prod.fst =
        some (0 :: (encs (openBind T₀ (FlockComplete.flat tr.cm0.toC.cap)) ++ (Frame.sq 2 tr.sch.n).enc)) ∧
      (∀ i, 1 ≤ i → i ≤ tr.sch.k0 + 2 → ((m₀, ρ₀) :: ops)[J + i]?.map Prod.fst =
        some (0 :: FlockComplete.LigLock.l0BodyAt
          (fun p e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) (J - 1 + p))[e]!) 0 []
          T₁ (FlockComplete.flat tr.cm0.toC.cap) tr.sch tr.g (FlockComplete.basisVals sws (tr.st.m - 7))
          (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0)
          ((ofLoop T L₀ real).drawsAt x ρ (2 * j + r)).h ((2 * j + r) % 2) tr.extra (fun _ _ => 0) i)) ∧
      J = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
          (3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6)) + (2 + (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j + r) / 2]!
            (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j + r) / 2]!
              (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) 0))).size) + 1 + 1 ∧
      ∃ zc lc, FlockComplete.LigLock.Fed.PtsAt tr.st.m tr.st.fold.kLog
          (fun p e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) p)[e]!)
          (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r)) zc lc ∧
        sws = FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++
            _root_.Flock.Firewall.Finish.claimsOf (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j + r) / 2]!
            (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j + r) / 2]!
              (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) 0))))
          ⟨fun k e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2)
              (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
                (3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6)) + k))[e]!,
            fun e => (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2)
              (FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
                (3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6)) + (2 + (_root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j + r) / 2]!
            (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j + r) / 2]!
              (wordsAt x (ops.map fun q => ZKMal.CoinLayer.rd q.2) 0))).size)))[e]!⟩ := by
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, htab⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hK := Masked.refK_eq hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j + r) / 2 = j := by omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [hjj]
    exact getElem!_pos _ j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hrR : ∀ q, q / 2 = (2 * j + r) / 2 → FlockDeploy.repRounds true spec (Masked.stsOf x)[q / 2]! = N + 4 := by
    intro q hq'
    rw [hq']
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hrq : ∀ q, q / 2 = (2 * j + r) / 2 → q < spec.repStreams.size →
      (roundsOf spec (Masked.stsOf x)).getD q 0 = N + 4 := by
    intro q hq' hq
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq, Option.map_some,
      Option.getD_some]
    exact hrR q hq'
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hend : FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
      FlockDeploy.repRounds true spec (Masked.stsOf x)[(2 * j + r) / 2]! ≤ refK x := by
    rw [← FlockComplete.ZkTables.repStart_succ, hK, ← FlockComplete.ZkTables.repStart_last]
    exact FlockComplete.ZkTables.repStart_mono spec (Masked.stsOf x) (by omega)
  rw [hrR _ rfl] at hend
  obtain ⟨R, hRd⟩ : ∃ R, R = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := ⟨_, rfl⟩
  obtain ⟨P, hPd⟩ : ∃ P, P = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! := ⟨_, rfl⟩
  rw [← hRd, ← hPd] at hJ
  rw [← hRd] at hend
  have hs1 : 1 ≤ R := by rw [hRd]; unfold FlockComplete.repStartZk; omega
  have hP1 : 1 ≤ P := by rw [hPd]; unfold FlockComplete.piopRoundsZk; omega
  -- the bounds of level 0's rounds
  have hJK : J + y.sch.k0 + 2 ≤ refK x := by omega
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  have hlen : ((m₀, ρ₀) :: ops).length = refK x + 1 := by simp [hl]
  have hle : ((m₀, ρ₀) :: ops).length ≤ (ofLoop T L₀ real).last x := by rw [hlast]; omega
  have hRs := repStartZk_eq spec (Masked.stsOf x) hq2.le
  have hloc : ∀ q i, q + 1 = R + i → i < N + 4 →
      ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 q = (2 * j + r, i) := by
    intro q i hqi hi
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) i (by rw [hrsl]; exact hq2)
      (by rw [hrq _ rfl hq2]; omega)
    rw [Nat.zero_add] at e
    rw [← e]
    congr 1
    omega
  have hkd : ∀ c, J ≤ c → c ≤ J + y.sch.k0 + 2 → (ofLoop T L₀ real).kindAt x c = .masked := by
    intro c h1 h2
    unfold kindAt
    rw [hlast]
    split_ifs <;> first | rfl | omega
  have hrepk : ∀ c, J ≤ c → c ≤ J + y.sch.k0 + 2 → (ofLoop T L₀ real).rep x c = 2 * j + r := by
    intro c h1 h2
    obtain ⟨c', rfl⟩ : ∃ c', c = c' + 2 := ⟨c - 2, by omega⟩
    show repOf x (c' + 2) = _
    rw [repOf_add_two hsx, hloc c' (c' + 1 - R) (by omega) (by omega)]
  have htk : ∀ c, 1 ≤ c → ((m₀, ρ₀) :: ops).take c = (m₀, ρ₀) :: ops.take (c - 1) := by
    intro c hc
    obtain ⟨c', rfl⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
    rw [List.take_succ_cons, Nat.add_sub_cancel]
  have htl : ∀ c, c ≤ ((m₀, ρ₀) :: ops).length → (((m₀, ρ₀) :: ops).take c).length = c := fun c hc => by
    rw [List.length_take]
    omega
  have htl' : ∀ c, c ≤ ops.length → (ops.take c).length = c := fun c hc => by
    rw [List.length_take]
    omega
  -- the placeholder at the binding: the frame's state there
  have hopJ := (coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    hle hrf (i := J) (by omega) (by omega)).1
  obtain ⟨mA, hA⟩ := answered_ok_of_live (ofLoop T L₀ real) wk x w ρ κ (((m₀, ρ₀) :: ops).take J)
    (by rw [htl J (by omega)]; exact hkd J le_rfl (by omega))
    ((ofLoop T L₀ real).live_take wk x w ρ κ _ hlive J)
  have hAJ := hA
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[J - 2]? = some d := ⟨ops[J - 2]'(by omega), List.getElem?_eq_getElem _⟩
  have htk2 : ops.take (J - 1) = ops.take (J - 2) ++ [(dm, da)] := by
    rw [show J - 1 = J - 2 + 1 by omega, List.take_add_one, hd]; rfl
  have hts : (ops.take (J - 2)).length = J - 2 := by rw [List.length_take]; omega
  rw [answered_of_opens _ _ _ _ hopJ, htk J (by omega),
    ofLoop_frameAnswer_same T L₀ real x ρ (m₀, ρ₀) (ops.take (J - 1)) hsx (q := J - 2) (j := P - 1)
      (by rw [List.length_take]; omega)
      (by rw [hloc (J - 2) (J - 1 - R) (by omega) (by omega)]; show J - 1 - R = P - 1 + 1; omega),
    htk2, List.map_append, List.map_cons, List.map_nil,
    progOf_snoc x ((ops.take (J - 2)).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := J - 3)
      (by rw [List.length_map, hts]; omega),
    hloc (J - 3) (P - 1) (by omega) (by omega)] at hAJ
  dsimp only at hAJ
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take (J - 2)).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at hAJ
    exact absurd hAJ (answer_owed_none _ rfl)
  rw [hph] at hAJ
  dsimp only at hAJ
  obtain ⟨alg, -, hal, -⟩ := repState_ok hAJ
  obtain ⟨hts', ⟨tp, htp⟩, hxs⟩ := Count.placeholder_tab hph
  rw [hz, Except.ok.injEq, Prod.mk.injEq] at htp
  obtain ⟨htsch, -⟩ := htp
  have hzt : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (t.sch, y.tPad) := htsch ▸ hz
  obtain ⟨hm36, hk0, hn, hq⟩ := Count.sched_facts hzt
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom', hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph
  have hls : (wordsAt x ((ops.take (J - 2)).map fun p => ZKMal.CoinLayer.rd p.2) 0).size =
      min (loopNr x 0) _root_.Flock.Firewall.Frames.COIN_MAX := wordsAt_size x ops (by omega) (by omega)
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hpts : (wordsAt x ((ops.take (J - 2)).map fun p => ZKMal.CoinLayer.rd p.2) 0).size /
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
  have hmk : t.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    rw [hts']
    show (Masked.stsOf x)[(2 * j + r) / 2]!.merkle = _
    rw [hst]
    exact (htab j hj).2.2
  have hm14 : 14 ≤ t.st.m := by
    rw [hts']
    show 14 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.m
    rw [hst]
    have := (htab j hj).1
    omega
  have hAs : alg.size = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
      (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts', List.length_map, Array.length_toList] at e
    exact e
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
    rw [hAs, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hOL := Count.openList_length hzt
    (c := (2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size)) (by omega)
  have hAP : alg.size + (xs.size + 2) + 1 = P := by
    rw [hOL, hN, ← hPd, ← htsch] at hFO
    omega
  have hkk : y.sch.k0 = t.sch.k0 := by rw [htsch]
  -- the frame's state at the binding
  rw [show P - 1 + 1 = alg.size + (xs.size + 2) + 1 by omega] at hAJ
  obtain ⟨hdA, honA, -, hl0A, hiA, hpA, -⟩ := Bind.repState_opened
    (by rw [hc']; exact ofDraws_order _ _ _) hal hm14 hmb hdg (placeholder_cap hph) (hdm ▸ hdom') hmk hxb hk0 hn hq
    (by simp only [Array.size_push, Array.size_map, Array.size_range]; omega) hAJ
  have hcfg := (ofLoop_cfg_place T L₀ real x ρ (((m₀, ρ₀) :: ops).take J) hsx (q := J - 2) (j := P - 1)
    (by rw [List.length_take]; omega) (by rw [hloc (J - 2) (P - 1 + 1) (by omega) (by omega)]) hph).1
  -- the rep's own table, and its schedules at level 0's prefixes
  have hxs0 : _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[(2 * j + r) / 2]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[(2 * j + r) / 2]!
        (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) 0)) = xs := by
    rw [hxs, wordsAt_take x ops (show 0 < J - 2 by omega)]
  have hnorm : ∀ c, J ≤ c → c ≤ J + y.sch.k0 + 2 → ∀ (t' : _root_.Flock.Firewall.Finish.Tab) (hi' : Array ByteArray),
      schedSoFar t' ((ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).1 % 2)
        (spec.domain (spec.tables[(ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).1 / 2]!)
          ((ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).1 % 2)).toUTF8
        (_root_.Flock.Firewall.Drive.extras
          (Masked.stsOf x)[(ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).1 / 2]!
          (_root_.Flock.Firewall.Drive.linkPoints
            (Masked.stsOf x)[(ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).1 / 2]!
            (wordsAt x ((ops.take (c - 1)).map fun p => ZKMal.CoinLayer.rd p.2) 0))) hi'
        ((Array.range (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).2).map
          fun i => wordsAt x ((ops.take (c - 1)).map fun p => ZKMal.CoinLayer.rd p.2)
            (c - 2 - (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (c - 2)).2 + i + 1)) =
      schedSoFar t' ((2 * j + r) % 2) dom xs hi'
        ((Array.range (c - 1 - R)).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) := by
    intro c h1 h2 t' hi'
    rw [hloc (c - 2) (c - 1 - R) (by omega) (by omega)]
    dsimp only
    rw [wordsAt_take x ops (show 0 < c - 1 by omega), hxs0, ← hdm]
    congr 1
    exact rangeMap_congr fun i hi => by
      rw [wordsAt_take x ops (show c - 2 - (c - 1 - R) + i + 1 < c - 1 by omega)]
      congr 1
      omega
  have hex : ∀ c, J ≤ c → c ≤ J + y.sch.k0 + 2 → ∃ t' hi' sched, real x w ρ (2 * j + r) = .ok (t', hi') ∧
      schedSoFar t' ((2 * j + r) % 2) dom xs hi'
        ((Array.range (c - 1 - R)).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) =
        .ok sched := by
    intro c h1 h2
    have hlc := (ofLoop T L₀ real).live_take wk x w ρ κ _ hlive c
    rw [htk c (by omega)] at hlc
    obtain ⟨t', hi', sched, hre, hss⟩ := ofLoop_sched_live T L₀ real wk x w ρ κ (m₀, ρ₀) (ops.take (c - 1)) hlc
      (by rw [htl' (c - 1) (by omega), Nat.sub_add_cancel (show 1 ≤ c by omega)]; exact hkd c h1 h2)
      hsx (q := c - 2) (by rw [List.length_take]; omega)
    rw [hnorm c h1 h2] at hss
    rw [hloc (c - 2) (c - 1 - R) (by omega) (by omega)] at hre
    exact ⟨t', hi', sched, hre, hss⟩
  obtain ⟨tr, hr', sx, hreal, -⟩ := hex J le_rfl (by omega)
  have hex' : ∀ n, P ≤ n → n ≤ P + y.sch.k0 + 2 → ∃ sched, schedSoFar tr ((2 * j + r) % 2) dom xs hr'
      ((Array.range n).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) = .ok sched := by
    intro n h1 h2
    obtain ⟨t', hi', s, hre', hs'⟩ := hex (n + 1 + R) (by omega) (by omega)
    rw [hreal, Except.ok.injEq, Prod.mk.injEq] at hre'
    obtain ⟨rfl, rfl⟩ := hre'
    rw [show n + 1 + R - 1 - R = n by omega] at hs'
    exact ⟨s, hs'⟩
  obtain ⟨htrst, htrsch, -, -, hcapA⟩ := Count.real_tabAlike hR hph hreal
  have hzr : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (tr.sch, y.tPad) := by
    rw [htrsch]; exact hzt
  obtain ⟨-, hk0r, hnr, hqr⟩ := Count.sched_facts hzr
  have hn0r : tr.sch.n ≠ 0 := by have := (Count.sched_rho hzr).1; omega
  have hn64 := Count.sched_n hzr
  have hkr : tr.sch.k0 = y.sch.k0 := by rw [htrsch, hkk]
  have hmr : tr.st.m < 2 ^ 64 := by rw [htrst]; exact hmb
  have hdgr : tr.st.digest.size < 2 ^ 64 := by rw [htrst]; exact hdg
  have hmkr : tr.st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by rw [htrst]; exact hmk
  obtain ⟨sJ, hsJ⟩ := hex' P le_rfl (by omega)
  obtain ⟨algr, halr⟩ := schedSoFar_rounds hsJ
  have hAr : algr.size = alg.size := by
    have e1 := congrArg List.length (Count.rounds_counts hmb hal)
    have e2 := congrArg List.length (Count.rounds_counts hmr halr)
    rw [htrst] at e2
    simp only [List.length_map, Array.length_toList] at e1 e2
    omega
  have hdom : dom.size < 2 ^ 64 := hdm ▸ hdom'
  -- the binding and level 0 at the rep's own table
  obtain ⟨hJlt, T₀, hT₀, -⟩ := Bind.sched_binding halr hsJ hdom hdgr hcapA.2 hmr hmkr hxb hk0r hnr hqr hn0r
    (by simp only [Array.size_map, Array.size_range]; omega)
  obtain ⟨sX, hsX⟩ := hex' (P + y.sch.k0 + 2) (by omega) le_rfl
  have hm14r : 14 ≤ tr.st.m := by rw [htrst]; exact hm14
  obtain ⟨-, T₁, sws, ⟨zc, lc, hpts, hsws⟩, hX⟩ := Bind.sched_l0 halr hsX hdom hdgr hcapA.2 hmr hmkr hxb hk0r hnr hqr
    hn0r hm14r
    (j := tr.sch.k0 + 2) (by omega) (by simp only [Array.size_map, Array.size_range]; omega)
  have hsJ' : schedSoFar tr ((2 * j + r) % 2) dom xs hr'
      ((Array.range (J - 1 - R)).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) =
      .ok sJ := by
    rw [show J - 1 - R = P by omega]
    exact hsJ
  have hhist : ∀ c, J ≤ c → c ≤ J + y.sch.k0 + 2 → ∀ {sched : Array (ByteArray × ℕ)},
      schedSoFar tr ((2 * j + r) % 2) dom xs hr'
        ((Array.range (c - 1 - R)).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) =
        .ok sched → c - 1 - R < sched.size →
      (ofLoop T L₀ real).scheduled x w ρ (((m₀, ρ₀) :: ops).take c)
          (computedOf ((ofLoop T L₀ real).computed x w ρ (((m₀, ρ₀) :: ops).take c))) =
        (sched.getD (c - 1 - R) (.empty, 0)).1.data.toList := by
    intro c h1 h2 sched hs hlt
    rw [htk c (by omega)]
    rw [ofLoop_scheduled T L₀ real x w ρ (m₀, ρ₀) (ops.take (c - 1)) _ hsx (q := c - 2)
      (by rw [List.length_take]; omega) (t := tr) (hi := hr')
      (by rw [hloc (c - 2) (c - 1 - R) (by omega) (by omega)]; exact hreal) (sched := sched)
      (by rw [hnorm c h1 h2]; exact hs)
      (by rw [hloc (c - 2) (c - 1 - R) (by omega) (by omega)]; exact hlt),
      hloc (c - 2) (c - 1 - R) (by omega) (by omega)]
  have hcoins : ∀ i, i ≤ tr.sch.k0 + 2 → loopNr x (J - 1 + i) = if i = 0 then tr.sch.n else l0Nc i := by
    intro i hi
    have e := loopNr_rep hsx hq2 (i := P + i) (by rw [hrq _ rfl hq2]; omega)
    rw [show J - 1 + i = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + (P + i) by omega, e,
      show P + i = ((_root_.Flock.Firewall.Session.replayCoins (Masked.stsOf x)[(2 * j + r) / 2]!.m
        (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog).toList.filter (· != 0)).length +
        ((2 + spec.points * (Masked.stsOf x)[(2 * j + r) / 2]!.regions.size) + 1 + i) by omega]
    exact Count.repPlan_l0 hzr hi
  have hcJ : (ofLoop T L₀ real).coins x J = tr.sch.n := by
    show loopNr x (J - 1) = _
    have := hcoins 0 (Nat.zero_le _)
    rw [Nat.add_zero, if_pos (rfl : (0 : ℕ) = 0)] at this
    exact this
  have hrepJ := hrepk J le_rfl (by omega)
  have hso : startOf x (2 * j + r) = .ok (lay, mask) := by
    unfold startOf
    rw [hsx]
    exact hds
  have hlay : (ofLoop T L₀ real).lay x (2 * j + r) =
      _root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0 := by
    show ((startOf x (2 * j + r)).toOption.map (·.1)).getD default = _
    rw [hso]
    show lay = _
    obtain ⟨sch', tp', hs', hl'⟩ := start_lay hds
    rw [hzr, Except.ok.injEq, Prod.mk.injEq] at hs'
    rw [hl', ← hs'.1, htrst, hts']
    try rfl
  obtain ⟨hb, hbodies⟩ := l0_chain (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive hown hle hrf hP
    (J := J) (sch := tr.sch) (by omega) (by rw [hlen]; omega)
    (fun c h1 h2 => hkd c h1 (by omega))
    (fun c h1 h2 => by rw [hrepk c h1 (by omega), hrepJ])
    (fun c h1 h2 => by
      obtain ⟨c', rfl⟩ : ∃ c', c = c' + 1 := ⟨c - 1, by omega⟩
      obtain ⟨⟨cm, ca⟩, hcd⟩ : ∃ d, ops[c']? = some d := ⟨ops[c']'(by omega), List.getElem?_eq_getElem _⟩
      have htc : ops.take (c' + 1) = ops.take c' ++ [(cm, ca)] := by
        rw [List.take_add_one, hcd]
        rfl
      have hlc : (ops.take c').length = c' := htl' c' (by omega)
      rw [htk (c' + 1 + 1) (by omega), Nat.add_sub_cancel]
      show (coinsAt x ((m₀, ρ₀) :: ops.take (c' + 1))).size = loopNr x (c' + 1 - 1)
      rw [htc, Nat.add_sub_cancel]
      unfold coinsAt
      simp only [List.tail_cons, List.getLast?_concat, List.length_append, hlc, List.length_singleton,
        Nat.add_sub_cancel]
      rw [feedOf_rd]
      simp only [FlockDeploy.Coin.vals, Array.size_map, Array.size_range]
      have hc := hcoins (c' + 1 - J) (by omega)
      rw [show J - 1 + (c' + 1 - J) = c' by omega] at hc
      have hb : loopNr x c' ≤ _root_.Flock.Firewall.Frames.COIN_MAX := by
        rw [hc]
        unfold _root_.Flock.Firewall.Frames.COIN_MAX l0Nc
        split_ifs <;> omega
      exact Nat.min_eq_left hb)
    (by omega) (by rw [hrepJ, hlay]; try rfl) (by rw [hrepJ, hlay]; try rfl) hA honA hdA hl0A hiA
    (by rw [hpA, hcfg.nPiop])
    T₀ (FlockComplete.flat tr.cm0.toC.cap) hcapA.2
    (by
      rw [hhist J le_rfl (by omega) hsJ' (by omega), show J - 1 - R = algr.size + (xs.size + 2) + 1 by omega, hT₀,
        hcJ])
    (by rw [hcJ]; exact hn0r) (by rw [hcJ]; omega)
    (coin := fun j i => ((((Array.range (P + y.sch.k0 + 2)).map
      fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)).extract algr.size
      ((Array.range (P + y.sch.k0 + 2)).map
        fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)).size).toList.getD j #[])[i]!)
    (p0 := xs.size + 2 + 1) (P := []) (T := T₁) (cap := FlockComplete.flat tr.cm0.toC.cap) (t := tr.g)
    (b := FlockComplete.basisVals sws (tr.st.m - 7)) (rep := (2 * j + r) % 2) (extra := tr.extra)
    (nonce := fun _ _ => 0)
    (_root_.Flock.Zk.PadLayout.new tr.st.m tr.st.kLog tr.sch.k0).len
    (fun i h1 h2 => by
      obtain ⟨si, hsi⟩ := hex' (P + i) (by omega) (by omega)
      obtain ⟨hsz, -⟩ := Bind.sched_l0 halr hsi hdom hdgr hcapA.2 hmr hmkr hxb hk0r hnr hqr hn0r hm14r (j := i)
        (by omega)
        (by simp only [Array.size_map, Array.size_range]; omega)
      have hsi' : schedSoFar tr ((2 * j + r) % 2) dom xs hr'
          ((Array.range (J + i - 1 - R)).map fun i => wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2) (R + i)) =
          .ok si := by
        rw [show J + i - 1 - R = P + i by omega]
        exact hsi
      rw [hhist (J + i) (by omega) (by omega) hsi' (by omega), hrepJ, hlay,
        show J + i - 1 - R = algr.size + (xs.size + 2) + 1 + i by omega]
      obtain ⟨e1, -⟩ := sched_mono _ hsi (k := algr.size + (xs.size + 2) + 1 + i) (by omega) (P + y.sch.k0 + 2)
        (by omega) (fun n h1 h2 => hex' n (by omega) h2) sX hsX
      rw [e1]
      exact (hX i (by omega)).1)
    (fun i h1 h2 => by
      show loopNr x (J + i - 1) = _
      have hc := hcoins i (by omega)
      rw [show J - 1 + i = J + i - 1 by omega, if_neg (show i ≠ 0 by omega)] at hc
      exact hc)
  have hm14s : 14 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.m := by
    rw [hst]
    have := (htab j hj).1
    omega
  have hA3 : algr.size = 3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6) := by
    have e1 : tr.st.m = (Masked.stsOf x)[(2 * j + r) / 2]!.m := by rw [htrst, hts']; rfl
    have e2 : tr.st.fold.kLog = (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [htrst, hts']; rfl
    rw [e1, e2, hAr, hAs, Count.piop_list hm14s]
    simp only [List.length_cons, List.length_append, List.length_replicate, List.length_nil]
    omega
  refine ⟨tr, hr', T₀, T₁, sws, hreal, by omega, ?_, fun i h1 h2 => ?_, ?_, ?_⟩
  · rw [List.getElem?_eq_getElem (show J < ((m₀, ρ₀) :: ops).length by omega)]
    exact congrArg some (hb.trans (by rw [hcJ]))
  · rw [List.getElem?_eq_getElem (show J + i < ((m₀, ρ₀) :: ops).length by omega)]
    refine congrArg some ((hbodies i h1 h2).trans ?_)
    rw [hrepJ, hlay, l0Body_clean (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) (R := R) (A := algr.size)
      (N := xs.size + 2) (C := P + y.sch.k0 + 2) (S := J - 1) (by omega) (by omega)]
  · rw [← hRd, hxs0]
    omega
  · rw [hxs0]
    have hcsz : (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs).size =
        xs.size + 2 := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
      omega
    have hAC : algr.size + (xs.size + 2) < P + y.sch.k0 + 2 := by omega
    have hL : R + algr.size = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
        (3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6)) := by rw [hA3, hRd]
    have hS : R + algr.size + (xs.size + 2) = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) +
        (3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6)) + (2 + xs.size) := by rw [hL, Nat.add_comm xs.size 2]
    refine ⟨zc, lc, ?_, hsws.trans ?_⟩
    · have hp := pts_raw (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) (R := R) (A := algr.size)
        (C := P + y.sch.k0 + 2) (m := tr.st.m) (kLog := tr.st.fold.kLog) (zc := zc) (lc := lc) (le_of_eq hA3.symm)
        (Nat.le_of_lt (Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hAC)) hpts
      rw [← hRd]
      exact hp
    have e := sws_raw (wordsAt x (ops.map fun p => ZKMal.CoinLayer.rd p.2)) _ hcsz hAC hL hS
    refine Eq.trans ?_ e
    with_reducible rfl

end Claims.ZKServer.LeanRelease
