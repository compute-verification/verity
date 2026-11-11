module

public import Proofs.Flock.SessionLoop.RoundL0Frame
public import Proofs.Flock.SessionLoop.RoundL0Cut
public import Proofs.Flock.SessionLoop.RoundL0Vals
public import Proofs.Flock.SessionLoop.RoundSwitchSv
public import Proofs.Flock.SessionLoop.SchedUnrefused

/-!
# A rep's level-0 rounds past the binding

`LiveSched`'s equation at rep stream `2j + r`'s level-0 rounds `1 ≤ i ≤ k₀ + 3` (`Masked.repSlot` `i`; at `k₀ + 3`,
`ρ_1`'s squeeze alone), where `vs`'s level-0 values are the ZK server's (`Loop.L0Fixed`): the release of the worker's
round is the switched session's message without its leading domain frame.

* `Masked.l0BodyAt_ligTab`: at table `j`, level 0's bodies read the extra lanes only below `2^cols`.
* **`LeanRelease.release_sched_l0_st`**: where no refusal shows by the round. The frame releases body `i` at the
  loop's table, the basis of its switched vectors at the openings' words, the rep's pads and the opened coins from the
  binding on (`l0_frame`). The
  session sends body `i` at the switched source's pads (`Loop.swMsgAt_l0`), which under `L0Fixed` are the rep's pads
  (`Loop.l0BodyAt_vs`, `Loop.RepSrc.pads`), at the tape's packed table and extra lanes, which are the loop's table's
  within level 0's sizes (`Loop.Level0Src.realOf`, `Loop.Level0Src.realOf_lanes`, `Masked.l0BodyAt_ligTab`), the
  basis of the switched source's vectors at the opened coins, which is the loop's
  (`Loop.sws_of_pts`, the loop's extra claims at the session's link points, `Loop.linkPoints_slot`), and the opened
  coins, which are the openings' words (`wordsAt_slot`). Past the binding a body has no domain frame
  (`Loop.dropDom_l0Body`).
* **`LeanRelease.release_sched_l0_pf`**: `release_sched_l0`'s statement. Level 0's rounds are past the rep's PCS round
  and before its `β` round, where no refusal shows (`unrefused_sched`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal.Masked

/-- **`l0BodyAt` at table `j` reads the extra lanes only below `2^cols`**, `cols` level 0's. -/
theorem l0BodyAt_ligTab (coin : ℕ → ℕ → _root_.Flock.F128) (p0 : ℕ) (x : Msg) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (P : List UInt8) (T : _root_.Flock.F128) (cap : ByteArray) (b : Array _root_.Flock.F128)
    (lay : _root_.Flock.Zk.PadLayout) (h : Array _root_.Flock.F128) (rep : ℕ)
    {extra extra' : ℕ → ℕ → _root_.Flock.F128} (he : ∀ e c, c < 2 ^ (lv0 x j).cols → extra e c = extra' e c)
    (nonce : ℕ → ℕ → UInt64) (i : ℕ) :
    FlockComplete.LigLock.l0BodyAt coin p0 P T cap (zkTab x j).sch (ligTab x w τ j) b lay h rep extra nonce i =
      FlockComplete.LigLock.l0BodyAt coin p0 P T cap (zkTab x j).sch (ligTab x w τ j) b lay h rep extra' nonce i := by
  have hA : ∀ e, (Array.range (FlockComplete.LigLock.tabS coin p0 (ligTab x w τ j) (zkTab x j).sch.k0).size).map
      (extra e) =
      (Array.range (FlockComplete.LigLock.tabS coin p0 (ligTab x w τ j) (zkTab x j).sch.k0).size).map (extra' e) :=
    fun e => Array.map_congr_left fun c hc =>
      he e c (by rw [← tabS_ligTab_size coin p0 x w τ j]; exact Array.mem_range.1 hc)
  unfold FlockComplete.LigLock.l0BodyAt
  simp only [hA]

end Claims.ZKMal.Masked

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open FlockSoundness.Refine (Frame encs)

open Loop Ranked in
/-- **Level 0's rounds past the binding, where no refusal shows**: at rep stream `2j + r`'s level-0 round
`1 ≤ i ≤ k₀ + 3` (`Masked.repSlot` `i`), where `vs`'s level-0 values are the ZK server's (`Loop.L0Fixed`) and no refusal
shows by the round, the release of the worker's round is the switched session's message without its leading domain
frame: body `i` at the rep's pads, the tape's packed table and extra lanes, the basis of the switched source's vectors
and the opened coins from the binding on. -/
theorem release_sched_l0_st (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hL0 : Loop.L0Fixed vs x w) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) {v : Msg} (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : 1 ≤ i)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3)
    (hN : ¬ Masked.Refused x (Loop.finS.pad (K := refK x) (ops.map fun p => Loop.finS.decOf p.2)) ops.length) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hS0 : Loop.Level0Src src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).1
  have hRS : Loop.RepSrc L₀ src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).2
  -- the frame's release: body `i` at the loop's table, its basis, the rep's pads and the openings' words
  obtain ⟨tr, hir, zc, lc, T, xs, lay, mask, hre, hT1, hpts, hxs, hrep, hds, hrel⟩ :=
    l0_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn h1 hi
  have hrel' : (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = _ := hrel
  rw [hrel']
  -- the session's round: body `i` at the switched source's pads
  have hP0 : Masked.ligP0 x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! :=
    (Masked.repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  have hPi : ops.length = Masked.ligP0 x j r + i := by rw [hn, Masked.repSlot_eq hsx hz, hP0]
  -- `Masked.swSrcE` at `finS.pad (ops.map …)` itself exhausts `whnf`; at a variable list it elaborates
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  rw [hcs] at hN
  have hcl : cs.length = ops.length := by rw [← hcs, List.length_map]
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx (Loop.finS.pad (K := refK x) cs) hj
  obtain ⟨P', T', cap', hmsg⟩ := Loop.swMsgAt_l0 vs w hx hsx ν t cs (by omega) j r (i := i) (by omega)
    (by rw [← hPi]; exact hN) hext hi
  rw [hmsg]
  rcases Nat.lt_or_ge i ((Masked.zkTab x j).sch.k0 + 3) with h2 | h2
  swap
  · obtain rfl : i = (Masked.zkTab x j).sch.k0 + 3 := by omega
    simp only [FlockComplete.LigLock.l0BodyAt_r1]
    have e := dropDom_noDom [] 2 2 fun f hf => by simp at hf
    rw [show encs ([] : List Frame) ++ (Frame.sq 2 2).enc = (Frame.sq 2 2).enc by simp [encs]] at e
    exact e.symm
  rw [dropDom_l0Body h1 (by omega)]
  -- `congr 1` would retry `rfl` at default transparency on the two bodies, which exhausts `whnf`
  refine congrArg (List.cons _) ?_
  -- the table and extra lanes are the tape's
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by rw [hreal] at hre; exact hre
  have hg := (hS0.realOf ν t j r hre').2.2.2
  have hex := (Loop.Level0Src.realOf_lanes hS0 ν t j r hre').1
  -- the frame's pads are the tape's
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hds
  have hzD := Loop.RepSrc.size hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
  have hH : ((ofLoop Masked.dataE (withFiles (withLink L₀ src) src) real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h =
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h := by
    have e := (cfg_tapeOfDraws (ofLoopLink L₀ src real) x (ν, t) ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t)
      Loop.frepOf j.val r.val hrep hzD).1
    rw [hlay, ← Loop.RepSrc.pads hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
      (Masked.UE.tapeZk t)] at e
    exact e
  -- the openings' words are the opened coins
  have hCM : _root_.Flock.Firewall.Frames.COIN_MAX = 65536 := rfl
  have hW : ∀ p e, p < ops.length → e < min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX →
      (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]! = Masked.slotAt (Loop.finS.pad (K := refK x) cs) p e := by
    intro p e hp he
    have hw := wordsAt_slot Masked.dataE x ops hlt.le (s := ops.length) (p := p) (e := e) hp le_rfl
      (by rw [wordsAt_size x ops hp le_rfl]; exact he)
    rw [List.take_length] at hw
    exact hw.trans (congrArg (fun c => Masked.slotAt (Loop.finS.pad (K := refK x) c) p e) hcs)
  have htm : tr.st.m = (Masked.stAt x j).m := by rw [hT1]; rfl
  have htk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1]; rfl
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  have hLR : Masked.ligP0 x j r = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := by
    have hPR : FlockComplete.piopRoundsZk spec (Masked.stAt x j) = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := rfl
    rw [hP0, Masked.repBase_eq hsx, hst, hPR]
    omega
  -- the loop's counts at the algebraic phase's and the ring switch's rounds
  obtain ⟨hp1, hp3, hp5, hp7, hpN⟩ := loopNr_piop hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6)
  rw [hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at hp1 hp3 hp5 hp7 hpN
  -- the extra claims: the table's regions at each link point, the session's
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hq2
  have hmp' : (Masked.stAt x j).mPts = spec.mPts := by rw [← hst]; exact hmp
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hW0 : (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  have hXS : xs.size = spec.points * (Masked.stAt x j).regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hW0, hmp', Nat.mul_div_cancel _ hmp0]
    exact Nat.mul_comm _ _
  obtain ⟨hpts2, -, hregs⟩ := Masked.live_bounds hx hsx
  have hpr : spec.points * (Masked.stAt x j).regions.size ≤ 2048 := by
    have hrg : (Masked.stAt x j).regions.size ≤ 1024 := hregs j.val hj
    rw [hpts2]
    omega
  have heS : (Masked.stAt x j).extraClaims (Masked.ptsE (Loop.finS.pad (K := refK x) cs) spec) = .ok extras := by
    have hext' := hext
    rw [Masked.tabsOf_stAt] at hext'
    exact hext'
  have hXE : extras.map (·.x) = xs := by
    have e := Loop.linkPoints_slot hx hsx hpos hlt.le hq2
    rw [Masked.stsOf_half] at e
    rw [hxs, extraClaims_xs heS, ← hcs, ← e]
  -- the basis of the loop's switched vectors is the switched source's
  have hsw := sws_of_pts (st := tr.st) (W := fun p e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]!)
    (C := Masked.ligCoin (Loop.finS.pad (K := refK x) cs)) (R := Masked.repBase x (2 * j.val + r.val))
    (L₁ := Masked.rankSlot x j.val r.val)
    (L₂ := Masked.repBase x (2 * j.val + r.val) + 3 + (tr.st.m - 6) + 3 + (tr.st.fold.kLog - 6))
    (zc := zc) (lc := lc) (xs := xs) (extras := extras) (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! hpts
    (fun e he => hW _ e (by omega) (by
      rw [hp1, hCM, Nat.min_eq_left (by omega)]
      omega))
    (fun e he => hW _ 0 (by omega) (by
      rw [hp3 e (by omega), hCM]
      decide))
    (fun e he => hW _ 0 (by omega) (by
      rw [htm, hp5 e (by omega), hCM]
      decide))
    (by rw [hRk, htm, htk]; omega)
    (fun k hk e he => hW _ e (by omega) (by
      rw [show Masked.rankSlot x j.val r.val + k = Masked.repBase x (2 * j.val + r.val) + 3 +
          ((Masked.stAt x j).m - 6) + 3 + ((Masked.stAt x j).c.kLog - 6) + k by rw [hRk]; omega,
        hp7 k (by omega), hCM]
      exact lt_of_lt_of_le he (by decide)))
    (fun e he => hW _ e (by omega) (by
      rw [show Masked.rankSlot x j.val r.val + (2 + xs.size) = Masked.repBase x (2 * j.val + r.val) + 3 +
          ((Masked.stAt x j).m - 6) + 3 + ((Masked.stAt x j).c.kLog - 6) +
          (2 + spec.points * (Masked.stAt x j).regions.size) by rw [hRk, hXS]; omega,
        hpN, hCM, Nat.min_eq_left (by omega)]
      omega))
    hXE
  beta_reduce at hsw
  have htabj : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
    rw [getElem!_pos _ j.val hj']
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have htabst : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = tr.st := htabj.trans hT1.symm
  have hbasis : FlockComplete.basisVals (FlockComplete.Replay.Rs.sws
        (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs)
        ⟨fun k e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) (Masked.rankSlot x j.val r.val + k))[e]!,
          fun e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2)
            (Masked.rankSlot x j.val r.val + (2 + xs.size)))[e]!⟩) (tr.st.m - 7) =
      Masked.ligBasis x (Loop.finS.pad (K := refK x) cs) w t j r extras := by
    have hzc : ∀ L, FlockComplete.Replay.Zc.coinsAt (Masked.ligRespE (Loop.finS.pad (K := refK x) cs)) L =
        FlockComplete.zcCoins (Masked.ligCoin (Loop.finS.pad (K := refK x) cs)) L :=
      fun L => Masked.zcCoins_ligRespE _ L
    have hlc : ∀ L n, FlockComplete.Replay.Lc.coinsAt (Masked.ligRespE (Loop.finS.pad (K := refK x) cs)) L n =
        FlockComplete.lcCoins (Masked.ligCoin (Loop.finS.pad (K := refK x) cs)) L n :=
      fun L n => Masked.lcCoins_ligRespE _ L n
    have hrc : ∀ L n, FlockComplete.Replay.Rs.coinsAt (Masked.ligRespE (Loop.finS.pad (K := refK x) cs)) L n =
        ⟨fun k => Masked.ligCoin (Loop.finS.pad (K := refK x) cs) (L + k),
          Masked.ligCoin (Loop.finS.pad (K := refK x) cs) (L + n)⟩ :=
      fun L n => Masked.rsCoins_ligRespE _ L n
    unfold Masked.ligBasis Masked.ligSt Masked.swsE
    rw [htabst, hzc, hlc, hrc]
    refine congrArg (fun s => FlockComplete.basisVals s (tr.st.m - 7)) ?_
    refine Eq.trans ?_ (hsw.trans ?_)
    · with_reducible rfl
    · with_reducible rfl
  -- the loop's counts at level 0's rounds
  have hsn := Count.sched_n hz
  have hn0 : loopNr x (Masked.ligP0 x j r) = (Masked.zkTab x j).sch.n := by
    have e := loopNr_l0 hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) (i := 0) (Nat.zero_le _)
    rw [Nat.add_zero, if_pos rfl] at e
    rw [hP0]
    exact e
  have hnl : ∀ a, 1 ≤ a → a ≤ (Masked.zkTab x j).sch.k0 + 3 → loopNr x (Masked.ligP0 x j r + a) = l0Nc a := by
    intro a ha1 ha2
    rw [hP0]
    exact loopNr_l0r hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) ha1 ha2
  -- body `i` at the session's pads, table, extra lanes and basis, then at the opened coins
  rw [hbasis, hg, Masked.l0BodyAt_ligTab _ _ x w t j _ _ _ _ _ _ _ hex _ i, hH,
    Loop.l0BodyAt_vs hL0 ν t (Loop.finS.pad (K := refK x) cs) hN hsx j r hext P' T' cap' (fun _ _ => 0) i]
  refine (FlockComplete.LigLock.l0BodyAt_bind_congr h1 [] P' T T' _ cap').trans
    (FlockComplete.LigLock.l0BodyAt_congr_lt ?_ ?_ ?_)
  · intro _ e he
    exact hW (Masked.ligP0 x j r) e (by omega) (by
      rw [hn0, Nat.min_eq_left (show (Masked.zkTab x j).sch.n ≤ _root_.Flock.Firewall.Frames.COIN_MAX by
        rw [hCM]; omega)]
      exact he)
  · intro hi2
    exact hW (Masked.ligP0 x j r + 1) 0 (by omega) (by
      rw [hnl 1 le_rfl (by omega), hCM]
      decide)
  · intro jj _ hj3 e he
    exact hW (Masked.ligP0 x j r + 2 + jj) e (by omega) (by
      rw [show Masked.ligP0 x j r + 2 + jj = Masked.ligP0 x j r + (2 + jj) by omega,
        hnl (2 + jj) (by omega) (by omega), hCM]
      unfold l0Nc
      rw [if_neg (by omega)]
      exact lt_of_lt_of_le he (by decide))

/-- **Level 0's rounds**: at rep stream `2j + r`'s level-0 round `i`, `1 ≤ i ≤ k0 + 3` (`Masked.repSlot` `i`; at
`k0 + 3`, `ρ_1`'s squeeze alone), where `vs`'s level-0 values are the ZK server's (`Loop.L0Fixed`), the release of the
worker's round is the switched session's message without its leading domain frame. The round is past the rep's PCS
round and before its `β` round, where no refusal shows (`unrefused_sched`), so it is `release_sched_l0_st`'s. -/
theorem release_sched_l0_pf (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
    (src : Msg → Flock.Wit → Loop.Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
    (real : Msg → Flock.Wit → Loop.Tp → ℕ → V (Tab × Array ByteArray)) (hreal : real = Loop.realOf src)
    (wk : (ofLoopLink L₀ src real).toSrv.Worker) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hSv : Loop.Serves x)
    (hSc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x) (hS : Loop.Src L₀ src x w)
    (hL0 : Loop.L0Fixed vs x w) (ν : Loop.finS.Nonce) (t : Loop.finS.Tape) (κ : (ofLoopLink L₀ src real).toSrv.Chk)
    (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hTV : Loop.TakesValsAt vs (ofLoopLink L₀ src real) wk x w ν t κ m₀ ρ₀ ops) {v : Msg}
    (hpos : 0 < ops.length) (hlt : ops.length < refK x)
    (hl : (ofLoopLink L₀ src real).toSrv.live wk x w (ν, t) κ ((m₀, ρ₀) :: ops) = true)
    (hok : ((allOk (Loop.finS.refP.chk x ν ρ₀) 0 (ops.map Prod.snd)).eval CoinOpen.shaH).1 = true)
    (href : (ofLoopLink L₀ src real).refusesCoins x (ν, t) ((m₀, ρ₀) :: ops) = false)
    (hv : wk x w (ν, t) ((m₀, ρ₀) :: ops) 0 = some (v, false)) (j : Fin (Masked.jOf x)) (r : Fin 2) {i : ℕ}
    (hn : ops.length = Masked.repSlot x (2 * j.val + r.val) fun _ => i) (h1 : 1 ≤ i)
    (hi : i ≤ (Masked.zkTab x j).sch.k0 + 3) :
    (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v =
      dropDom (Loop.finS.swMsgAt x (Loop.finS.wt x w) (vs x w ν) t (ops.map fun p => Loop.finS.decOf p.2)) := by
  have hx : Masked.Live x := Masked.live_of hSc hR
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  have hjh : (2 * j.val + r.val) / 2 = j.val := by omega
  have hrm : (2 * j.val + r.val) % 2 = r.val := by omega
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half x j r]
    exact (Masked.zkTab x j).hz
  have hRS : ∀ f : _root_.Flock.Schedule → ℕ, Masked.repSlot x (2 * j.val + r.val) f =
      Masked.repBase x (2 * j.val + r.val) + FlockComplete.piopRoundsZk spec (Masked.stsOf x)[j.val]! +
        f (Masked.zkTab x j).sch := by
    intro f
    rw [Masked.repSlot_eq hsx hz, Masked.repBase_eq hsx, hjh]
  have hPR : FlockComplete.piopRoundsZk spec (Masked.stsOf x)[j.val]! =
      (Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6 +
        (2 + spec.points * (Masked.stsOf x)[j.val]!.regions.size + 1) := rfl
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) := rfl
  have hN := unrefused_sched Masked.dataE (withFiles (withLink L₀ src) src) real wk x w hx hSv (ν, t) κ m₀ ρ₀ ops
    hlt.le hl href (q := 2 * j.val + r.val) (by omega) (by rw [hn, hRS]; omega)
    (by
      rw [hjh, hrm]
      unfold Masked.betaSlot
      rw [hn, hRS, hRS]
      try dsimp only
      omega)
    (by
      rw [hjh, hrm, hn, hRS, hRk, hPR]
      try dsimp only
      omega)
  exact release_sched_l0_st vs L₀ src real hreal wk hR hSv hSc hS hL0 ν t κ m₀ ρ₀ ops hpos hlt hl hv j r hn h1 hi hN

end LeanRelease

end Claims.ZKServer
