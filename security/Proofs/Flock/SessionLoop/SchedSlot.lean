module

public import Proofs.Flock.SessionLoop.InnerSlot
public import Proofs.Flock.Claims.ZK.L0PinLive
public import Proofs.Flock.SessionLoop.TakenVals

/-! The loop's own rounds sit within a rep's rounds, before its inner proof's (`LeanRelease.sched_slot`). -/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)
open _root_.Flock.ZkServer.Finish (Tab)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoopLink`'s session types, as in `TakenVals`

namespace Loop

theorem sched_slot_at {real : ℕ → V (Tab × Array ByteArray)} {x : Msg} (hx : Masked.Live x) (hSv : Serves x)
    {cs : List FlockDeploy.Coin} {s0 : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {q k i : ℕ}
    {tr : Tab} {hr : Array ByteArray} {rsched : Array (ByteArray × ℕ)}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) (hq : cs.length = q + 1) (hK : cs.length < refK x)
    (hloc : CoinLayer.locate (roundsOf spec sts) 0 q = (k, i)) (hR : Count.RealTabs sts real)
    (hreal : real k = .ok (tr, hr))
    (hrs : schedSoFar tr (k % 2) (spec.domain (spec.tables[k / 2]!) (k % 2)).toUTF8
      (_root_.Flock.ZkServer.Drive.extras sts[k / 2]!
        (_root_.Flock.ZkServer.Drive.linkPoints sts[k / 2]! (wordsAt x cs 0)))
      hr ((Array.range i).map fun j => wordsAt x cs (q - i + j + 1)) = .ok rsched)
    (hlt : i < rsched.size) :
    ∃ (j : Fin (Masked.jOf x)) (r : Fin 2), k = 2 * j.val + r.val ∧
      Masked.repBase x (2 * j.val + r.val) ≤ q + 1 ∧
      q + 1 < Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0) := by
  have hsts : Masked.stsOf x = sts := by
    unfold Masked.stsOf
    rw [hs]
  subst hsts
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hrsz : spec.repStreams.size = 2 * (Masked.stsOf x).size := by rw [FlockDeploy.repStreams_size, hspec]
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hsum : (roundsOf spec (Masked.stsOf x)).sum + 1 = refK x := by
    rw [Masked.refK_eq hs, ← FlockComplete.ZkTables.repStart_last, ← hrsz]
    unfold FlockComplete.repStartZk roundsOf
    omega
  have hinv := locate_inv (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
  rw [hloc] at hinv
  simp only [Nat.sub_zero] at hinv
  obtain ⟨hqs, -⟩ := hinv
  have hk : k < spec.repStreams.size := by
    have := locate_lt (roundsOf spec (Masked.stsOf x)) 0 q (by omega)
    rw [hloc] at this
    simp only [hrsl, Nat.zero_add] at this
    omega
  have hj : k / 2 < (Masked.stsOf x).size := by omega
  obtain ⟨y⟩ := hx.2 (k / 2) hj
  have hst : (Masked.stsOf x)[k / 2]! = (Masked.stsOf x)[k / 2] := getElem!_pos _ (k / 2) hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.wideLeaf (Masked.stsOf x)[k / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[k / 2]!.c.kLog := by
    rw [hst]
    exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[k / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  obtain ⟨g, d, l, hl, htr⟩ := hR k tr hr hreal
  obtain ⟨hts, hsch⟩ := Count.tabOfLevel0_sched htr
  have htsch : tr.sch = y.sch := by
    have e : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.wideLeaf (Masked.stsOf x)[k / 2]!.m = .ok (tr.sch, tr.tPad) := hsch
    rw [hz, Except.ok.injEq, Prod.mk.injEq] at e
    exact e.1.symm
  have hcn : (_root_.Flock.ZkServer.Finish.flat tr.cm0.cap).size < 2 ^ 64 := by
    obtain ⟨s, tp, hs', -⟩ := Count.ofSamples_gsize hl
    obtain ⟨-, -, ⟨s1, hs1, hlv⟩, -⟩ := _root_.Flock.ZkServer.Level0.ofSamples_ok hl
    rw [hs'] at hs1
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Except.ok.inj hs1)
    have hd : 64 * 2 ^ l.lv.d < 2 ^ 64 := by
      rw [hlv]
      exact (Count.sched_bounds hs').2 0
    have hA := Sched.tabOfLevel0_alike (_root_.Flock.Setup.ofCircuit (Masked.stsOf x)[k / 2]!) hl hl rfl hd
    rw [htr] at hA
    have hA' : Sched.TabAlike tr tr := hA
    exact hA'.2.2.2.2.2
  obtain ⟨hm36, hk0, hn, hqb⟩ := Count.sched_facts hz
  obtain ⟨⟨hmp, hmp0, hpc⟩, hdom, hxr, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hs) hk
  have hdg : tr.st.digest.size < 2 ^ 64 := by
    rw [hts]
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : tr.st.m < 2 ^ 64 := by
    rw [hts]
    exact Nat.lt_trans hm36 (by decide)
  have hls : (wordsAt x cs 0).size = min (loopNr x 0) _root_.Flock.ZkServer.Frames.COIN_MAX := by
    cases cs with
    | nil => simp at hq
    | cons p rest => simp [wordsAt, FlockDeploy.Coin.vals]
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hs]
    rfl
  have hpts : (wordsAt x cs 0).size / (Masked.stsOf x)[k / 2]!.mPts = spec.points := by
    rw [hls, hl0, Nat.min_eq_left hpc, hmp, Nat.mul_div_cancel _ hmp0]
  have hxn : (_root_.Flock.ZkServer.Drive.extras (Masked.stsOf x)[k / 2]!
      (_root_.Flock.ZkServer.Drive.linkPoints (Masked.stsOf x)[k / 2]! (wordsAt x cs 0))).size =
      spec.points * (Masked.stsOf x)[k / 2]!.regions.size := by
    rw [Count.extras_size, Count.linkPoints_size, hpts, Nat.mul_comm]
  have hxb : (_root_.Flock.ZkServer.Drive.extras (Masked.stsOf x)[k / 2]!
      (_root_.Flock.ZkServer.Drive.linkPoints (Masked.stsOf x)[k / 2]! (wordsAt x cs 0))).size + 2 < 2 ^ 64 := by
    rw [extras_size]
    have := Nat.mul_le_mul_left (Masked.stsOf x)[k / 2]!.regions.size
      ((linkPoints_size_le (Masked.stsOf x)[k / 2]! (wordsAt x cs 0)).trans (hls.trans_le (Nat.min_le_right _ _)))
    omega
  obtain ⟨alg, hal⟩ := schedSoFar_ok_rounds hrs
  have hA : alg.size = ((_root_.Flock.ZkServer.Session.replayCoins (Masked.stsOf x)[k / 2]!.m
      (Masked.stsOf x)[k / 2]!.c.kLog).toList.filter (· != 0)).length := by
    have e := congrArg List.length (Count.rounds_counts hmb hal)
    rw [Count.piop_counts, hts, List.length_map, Array.length_toList] at e
    exact e
  have hz' : _root_.Flock.Zk.schedule (Masked.stsOf x)[k / 2]!.wideLeaf (Masked.stsOf x)[k / 2]!.m = .ok (tr.sch, y.tPad) := by
    rw [htsch]
    exact hz
  have hplan := congrArg List.length (Count.repPlan_list (spec := spec) hz')
  rw [Array.length_toList, repPlan_size] at hplan
  simp only [repRounds_true, hz, List.length_append] at hplan
  have h4 : ([1, _root_.Flock.ZkServer.Session.INNER_CONS, 1, _root_.Flock.Zk.Q].filter (· != 0)).length = 4 := by
    decide
  have hFO : alg.size + (Count.openList (2 + spec.points * (Masked.stsOf x)[k / 2]!.regions.size)
      tr.sch).length = N := by
    rw [h4] at hplan
    have e1 : (Masked.stsOf x)[k / 2]!.m + (Masked.stsOf x)[k / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[k / 2]!.m + ((Masked.stsOf x)[k / 2]!.c.kLog - 6) := by omega
    rw [hA, hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
    exact (Nat.add_right_cancel hplan).symm
  have hiN : i < N := by
    by_contra hNi
    have hS := Count.schedSoFar_size tr _ _ _ hr _ hal hrs hdom hdg hcn hmb hxb (by rw [htsch]; exact hk0)
      (by rw [htsch]; exact hn) (by rw [htsch]; exact hqb) (by
        rw [hxn, show spec.points * (Masked.stsOf x)[k / 2]!.regions.size + 2 =
          2 + spec.points * (Masked.stsOf x)[k / 2]!.regions.size by omega, hFO]
        simp only [Array.size_map, Array.size_range]
        omega)
    rw [hxn, show spec.points * (Masked.stsOf x)[k / 2]!.regions.size + 2 =
      2 + spec.points * (Masked.stsOf x)[k / 2]!.regions.size by omega, hFO] at hS
    omega
  have hj' : k / 2 < Masked.jOf x := by
    unfold Masked.jOf
    rw [if_pos hx]
    exact hj
  refine ⟨⟨k / 2, hj'⟩, ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩, (Nat.div_add_mod k 2).symm, ?_, ?_⟩
  · show Masked.repBase x (2 * (k / 2) + k % 2) ≤ q + 1
    rw [Nat.div_add_mod k 2, Masked.repBase_eq hs, repStartZk_eq spec _ hk.le]
    omega
  · show q + 1 < Masked.repSlot x (2 * (k / 2) + k % 2) (Masked.innerOff 0)
    rw [Nat.div_add_mod k 2, Masked.repSlot_eq hs hz, repStartZk_eq spec _ hk.le]
    unfold Masked.innerOff
    omega

end Loop

namespace LeanRelease

open Loop in
theorem sched_slot (L₁ : LeanRelease Flock.Wit
      ((CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand))
    (real : Msg → Flock.Wit →
      (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand →
      ℕ → V (Tab × Array ByteArray))
    (wk : (ofLoop Masked.dataE L₁ real).toSrv.Worker) {x : Msg} {w : Flock.Wit} (hx : Masked.Live x)
    (hSv : Loop.Serves x)
    (ρ : (CoinLayer.session loopNr Masked.dataE).refP.Nonce × (CoinLayer.session loopNr Masked.dataE).refP.Rand)
    (κ : (ofLoop Masked.dataE L₁ real).toSrv.Chk) (m₀ ρ₀ : Msg)
    (ops : List (Msg × Msg))
    (hRT : ∀ (s0 : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt),
      FlockDeploy.setupOf true x = some (s0, spec, sts) → Count.RealTabs sts (real x w ρ))
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoop Masked.dataE L₁ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true) {b : ByteArray}
    (hsc : Loop.schedOf (real x w ρ) x (ops.map fun p => CoinLayer.rd p.2) = some b) :
    ∃ (j : Fin (Masked.jOf x)) (r : Fin 2), Masked.repBase x (2 * j.val + r.val) ≤ ops.length ∧
      ops.length < Masked.repSlot x (2 * j.val + r.val) (Masked.innerOff 0) := by
  rcases hs : FlockDeploy.setupOf true x with _ | ⟨s0, spec, sts⟩
  · obtain ⟨_, hp, -⟩ := hx.1
    exact absurd (refK_of_none hs (decodeRec_none_of_parse hp)) (by omega)
  obtain ⟨q, hq⟩ : ∃ q, ops.length = q + 1 := ⟨ops.length - 1, by omega⟩
  have hlast : (ofLoop Masked.dataE L₁ real).last x = refK x + 1 :=
    loop_last loopNr Masked.dataE (withFrames Masked.dataE L₁ real) x
  have hk : (ofLoop Masked.dataE L₁ real).kindAt x (ops.length + 1) = .masked :=
    kindAt_masked_of _ (by omega) (by rw [hlast]; omega)
  obtain ⟨tr, hir, rsched, hre, hrs⟩ :=
    ofLoop_sched_live Masked.dataE L₁ real wk x w ρ κ (m₀, ρ₀) ops hl hk hs hq
  have hcs : (ops.map fun p => CoinLayer.rd p.2).length = q + 1 := by rw [List.length_map, hq]
  rcases hloc : CoinLayer.locate (roundsOf spec sts) 0 q with ⟨k, i⟩
  rw [hloc] at hre hrs
  dsimp only at hre hrs
  have hi : i < rsched.size := by
    by_contra hle
    have hin : Loop.InnerAt (real x w ρ) x (ops.map fun p => CoinLayer.rd p.2) :=
      ⟨s0, spec, sts, q, k, i, tr, hir, rsched, hs, hcs, hloc, hre, hrs, by omega⟩
    rw [schedOf_none_of_innerAt hin] at hsc
    cases hsc
  obtain ⟨j, r, -, h1, h2⟩ := sched_slot_at hx hSv hs hcs (by rw [List.length_map]; exact hlt) hloc
    (hRT s0 spec sts hs) hre hrs hi
  exact ⟨j, r, by omega, by omega⟩

end LeanRelease

end Claims.ZKServer
