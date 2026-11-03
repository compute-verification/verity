import Proofs.Flock.SessionLoop.RoundL0Frame
import Proofs.Flock.SessionLoop.RoundL0Cut
import Proofs.Flock.SessionLoop.RoundL0Vals
import Proofs.Flock.SessionLoop.RoundSwitchSv
import Proofs.Flock.SessionLoop.SchedUnrefused

open Lean Elab Tactic in
elab "hb_mark " s:str : tactic => do
  logInfo m!"hb {s.getString}: {(← IO.getNumHeartbeats) / 1000}"

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)
open _root_.Flock.Firewall.Finish (Tab)

namespace LeanRelease

open FlockSoundness.Refine (Frame encs)

open Loop Ranked in
set_option Elab.async false in
set_option maxHeartbeats 4000000 in
/-- **Level 0's rounds past the binding, where no refusal shows**: at rep stream `2j + r`'s level-0 round
`1 ≤ i ≤ k₀ + 3` (`Masked.repSlot` `i`), where `vs`'s level-0 values are the firewall's (`Loop.L0Fixed`) and no refusal
shows by the round, the release of the worker's round is the switched session's message without its leading domain
frame: body `i` at the rep's pads, the tape's packed table and extra lanes, the basis of the switched source's vectors
and the opened coins from the binding on. -/
theorem release_sched_l0_st_probe (vs : Loop.finS.SwVals) (L₀ : LeanRelease Flock.Wit Loop.Tp)
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
  hb_mark "L62"
  have hx : Masked.Live x := Masked.live_of hSc hR
  hb_mark "L63"
  have hj := Masked.lt_of_jOf j
  hb_mark "L64"
  have hr2 := r.isLt
  hb_mark "L65"
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  hb_mark "L66"
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  hb_mark "L67"
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  hb_mark "L68"
  have hst : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  hb_mark "L69"
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  hb_mark "L70"
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst]
    exact (Masked.zkTab x j).hz
  hb_mark "L74"
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  hb_mark "L75"
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  hb_mark "L76"
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  hb_mark "L77"
  have hS0 : Loop.Level0Src src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).1
  hb_mark "L78"
  have hRS : Loop.RepSrc L₀ src x w := (show Loop.Level0Src src x w ∧ Loop.RepSrc L₀ src x w from hS).2
  -- the frame's release: body `i` at the loop's table, its basis, the rep's pads and the openings' words
  hb_mark "L80"
  obtain ⟨tr, hir, zc, lc, T, xs, lay, mask, hre, hT1, hpts, hxs, hrep, hds, hrel⟩ :=
    l0_frame (withFiles (withLink L₀ src) src) real wk hx hSv ν t κ m₀ ρ₀ ops
      (by rw [hreal]; exact Loop.realTabs_of src x w (ν, t) hsx) hlt hl hv j r hn h1 hi
  hb_mark "L83"
  have hrel' : (ofLoopLink L₀ src real).release x (ν, t) ((m₀, ρ₀) :: ops) false v = _ := hrel
  hb_mark "L84"
  rw [hrel']
  -- the session's round: body `i` at the switched source's pads
  hb_mark "L86"
  have hP0 : Masked.ligP0 x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! :=
    (Masked.repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  hb_mark "L89"
  have hPi : ops.length = Masked.ligP0 x j r + i := by rw [hn, Masked.repSlot_eq hsx hz, hP0]
  -- `Masked.swSrcE` at `finS.pad (ops.map …)` itself exhausts `whnf`; at a variable list it elaborates
  hb_mark "L91"
  generalize hcs : (ops.map fun p => Loop.finS.decOf p.2) = cs
  hb_mark "L92"
  rw [hcs] at hN
  hb_mark "L93"
  have hcl : cs.length = ops.length := by rw [← hcs, List.length_map]
  hb_mark "L94"
  obtain ⟨extras, hext⟩ := Masked.extra_ok hx hsx (Loop.finS.pad (K := refK x) cs) hj
  hb_mark "L95"
  obtain ⟨P', T', cap', hmsg⟩ := Loop.swMsgAt_l0 vs w hx hsx ν t cs (by omega) j r (i := i) (by omega)
    (by rw [← hPi]; exact hN) hext hi
  hb_mark "L97"
  rw [hmsg]
  hb_mark "L98"
  rcases Nat.lt_or_ge i ((Masked.zkTab x j).sch.k0 + 3) with h2 | h2
  hb_mark "L99"
  swap
  hb_mark "L100"
  · obtain rfl : i = (Masked.zkTab x j).sch.k0 + 3 := by omega
    hb_mark "L101"
    simp only [FlockComplete.LigLock.l0BodyAt_r1]
    hb_mark "L102"
    have e := dropDom_noDom [] 2 2 fun f hf => by simp at hf
    hb_mark "L103"
    rw [show encs ([] : List Frame) ++ (Frame.sq 2 2).enc = (Frame.sq 2 2).enc by simp [encs]] at e
    hb_mark "L104"
    exact e.symm
  hb_mark "L105"
  rw [dropDom_l0Body h1 (by omega)]
  -- `congr 1` would retry `rfl` at default transparency on the two bodies, which exhausts `whnf`
  hb_mark "L107"
  refine congrArg (List.cons _) ?_
  -- the table and extra lanes are the tape's
  hb_mark "L109"
  have hre' : Loop.realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir) := by rw [hreal] at hre; exact hre
  hb_mark "L110"
  have hg := (hS0.realOf ν t j r hre').2.2.2
  hb_mark "L111"
  have hex := (Loop.Level0Src.realOf_lanes hS0 ν t j r hre').1
  -- the frame's pads are the tape's
  hb_mark "L113"
  have hlay := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real j r hds
  hb_mark "L114"
  have hzD := Loop.RepSrc.size hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
  hb_mark "L115"
  have hH : ((ofLoop Masked.dataE (withFiles (withLink L₀ src) src) real).cfg x (ν, t) ((m₀, ρ₀) :: ops)).h =
      (FlockComplete.padsOf (Masked.UE.tapeZk t) j.val r.val (Masked.layE x j)).h := by
    have e := (cfg_tapeOfDraws (ofLoopLink L₀ src real) x (ν, t) ((m₀, ρ₀) :: ops) (Masked.UE.tapeZk t)
      Loop.frepOf j.val r.val hrep hzD).1
    rw [hlay, ← Loop.RepSrc.pads hRS (ofLoopLink L₀ src real) (LeanRelease.draws_ofLoopLink src real) ν t j r hlay
      (Masked.UE.tapeZk t)] at e
    exact e
  -- the openings' words are the opened coins
  hb_mark "L123"
  have hCM : _root_.Flock.Firewall.Frames.COIN_MAX = 65536 := rfl
  hb_mark "L124"
  have hW : ∀ p e, p < ops.length → e < min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX →
      (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]! = Masked.slotAt (Loop.finS.pad (K := refK x) cs) p e := by
    intro p e hp he
    have hw := wordsAt_slot Masked.dataE x ops hlt.le (s := ops.length) (p := p) (e := e) hp le_rfl
      (by rw [wordsAt_size x ops hp le_rfl]; exact he)
    rw [List.take_length] at hw
    exact hw.trans (congrArg (fun c => Masked.slotAt (Loop.finS.pad (K := refK x) c) p e) hcs)
  hb_mark "L131"
  have htm : tr.st.m = (Masked.stAt x j).m := by rw [hT1]; rfl
  hb_mark "L132"
  have htk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hT1]; rfl
  hb_mark "L133"
  have hRk : Masked.rankSlot x j.val r.val = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  hb_mark "L137"
  have hLR : Masked.ligP0 x j r = Masked.repBase x (2 * j.val + r.val) +
      ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := by
    have hPR : FlockComplete.piopRoundsZk spec (Masked.stAt x j) = (Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6 +
        (2 + spec.points * (Masked.stAt x j).regions.size + 1) := rfl
    rw [hP0, Masked.repBase_eq hsx, hst, hPR]
    omega
  -- the loop's counts at the algebraic phase's and the ring switch's rounds
  hb_mark "L145"
  obtain ⟨hp1, hp3, hp5, hp7, hpN⟩ := loopNr_piop hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6)
  hb_mark "L146"
  rw [hst, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at hp1 hp3 hp5 hp7 hpN
  -- the extra claims: the table's regions at each link point, the session's
  hb_mark "L148"
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hq2
  hb_mark "L149"
  have hmp' : (Masked.stAt x j).mPts = spec.mPts := by rw [← hst]; exact hmp
  hb_mark "L150"
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  hb_mark "L154"
  have hW0 : (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  hb_mark "L158"
  have hXS : xs.size = spec.points * (Masked.stAt x j).regions.size := by
    rw [hxs, Count.extras_size, Count.linkPoints_size, hW0, hmp', Nat.mul_div_cancel _ hmp0]
    exact Nat.mul_comm _ _
  hb_mark "L161"
  obtain ⟨hpts2, -, hregs⟩ := Masked.live_bounds hx hsx
  hb_mark "L162"
  have hpr : spec.points * (Masked.stAt x j).regions.size ≤ 2048 := by
    have hrg : (Masked.stAt x j).regions.size ≤ 1024 := hregs j.val hj
    rw [hpts2]
    omega
  hb_mark "L166"
  have heS : (Masked.stAt x j).extraClaims (Masked.ptsE (Loop.finS.pad (K := refK x) cs) spec) = .ok extras := by
    have hext' := hext
    rw [Masked.tabsOf_stAt] at hext'
    exact hext'
  hb_mark "L170"
  have hXE : extras.map (·.x) = xs := by
    have e := Loop.linkPoints_slot hx hsx hpos hlt.le hq2
    rw [Masked.stsOf_half] at e
    rw [hxs, extraClaims_xs heS, ← hcs, ← e]
  -- the basis of the loop's switched vectors is the switched source's
  hb_mark "L175"
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
  hb_mark "L202"
  beta_reduce at hsw
  hb_mark "L203"
  have htabj : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
    rw [getElem!_pos _ j.val hj']
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  hb_mark "L208"
  have htabst : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = tr.st := htabj.trans hT1.symm
  hb_mark "L209"
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
  hb_mark "L232"
  have hsn := Count.sched_n hz
  hb_mark "L233"
  have hn0 : loopNr x (Masked.ligP0 x j r) = (Masked.zkTab x j).sch.n := by
    have e := loopNr_l0 hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) (i := 0) (Nat.zero_le _)
    rw [Nat.add_zero, if_pos rfl] at e
    rw [hP0]
    exact e
  hb_mark "L238"
  have hnl : ∀ a, 1 ≤ a → a ≤ (Masked.zkTab x j).sch.k0 + 3 → loopNr x (Masked.ligP0 x j r + a) = l0Nc a := by
    intro a ha1 ha2
    rw [hP0]
    exact loopNr_l0r hsx hq2 hz (by rw [hst]; omega) (by rw [hst]; exact hk6) ha1 ha2
  -- body `i` at the session's pads, table, extra lanes and basis, then at the opened coins
  hb_mark "L243"
  rw [hbasis, hg, hex, hH, Loop.l0BodyAt_vs hL0 ν t (Loop.finS.pad (K := refK x) cs) hN hsx j r hext P' T' cap'
    (fun _ _ => 0) i]
  hb_mark "L245"
  refine (FlockComplete.LigLock.l0BodyAt_bind_congr h1 [] P' T T' _ cap').trans
    (FlockComplete.LigLock.l0BodyAt_congr_lt ?_ ?_ ?_)
  hb_mark "L247"
  · intro _ e he
    exact hW (Masked.ligP0 x j r) e (by omega) (by
      rw [hn0, Nat.min_eq_left (show (Masked.zkTab x j).sch.n ≤ _root_.Flock.Firewall.Frames.COIN_MAX by
        rw [hCM]; omega)]
      exact he)
  hb_mark "L252"
  · intro hi2
    exact hW (Masked.ligP0 x j r + 1) 0 (by omega) (by
      rw [hnl 1 le_rfl (by omega), hCM]
      decide)
  hb_mark "L256"
  · intro jj _ hj3 e he
    exact hW (Masked.ligP0 x j r + 2 + jj) e (by omega) (by
      rw [show Masked.ligP0 x j r + 2 + jj = Masked.ligP0 x j r + (2 + jj) by omega,
        hnl (2 + jj) (by omega) (by omega), hCM]
      unfold l0Nc
      rw [if_neg (by omega)]
      exact lt_of_lt_of_le he (by decide))
  hb_mark "end"

end LeanRelease

end Claims.ZKServer
