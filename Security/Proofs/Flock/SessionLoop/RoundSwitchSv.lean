module

public import Proofs.Flock.SessionLoop.SchedAt
public import Proofs.Flock.SessionLoop.Tables
public import Proofs.Flock.Claims.ZK.FrameRankFault

/-!
# The ring switch's switched vectors at the loop's coins and at the session's

At rep stream `2j + r`'s ring switch the loop's table schedules each claim's switched vector at the points of the
openings' words, and the switched session sends the switched source's. They agree at every claim the session switches
before its stop.

* `Loop.linkPoints_slot`: the link points at the first opening's words are the session's (`Masked.ptsE`).
* `Loop.rsBody_congr`: the switch's body `k` reads its switched vector `k` alone.
* **`Loop.switch_sv`**: on the loop's table (`Level0Src.realOf`), the switched vector of the loop's claim `k`, at the
  points of the openings' words, is the switched source's (`Masked.swSrc_sv`): the loop's claims are at the session's
  points (`ptsAt_claims_x`), its extra claims at the session's link points (`extraClaims_xs`).
-/

@[expose] public section

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 SessionSpec)

namespace Loop

/-- **The link points at the first opening's words are the session's**: after openings `ops`, at most `refK x`, rep
stream `k`'s link points off opening `0`'s words are `Masked.ptsE` at the opened coins. -/
theorem linkPoints_slot {x : Msg} (hx : Masked.Live x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {ops : List (Msg × Msg)}
    (hpos : 0 < ops.length) (hl : ops.length ≤ refK x) {k : ℕ} (hk : k < spec.repStreams.size) :
    _root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]!
        (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0) =
      Masked.ptsE (finS.pad (K := refK x) (ops.map fun p => finS.decOf p.2)) spec := by
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx hk
  have hl0 : loopNr x 0 = spec.points * spec.mPts := by
    show (reqOf x 0).2 = _
    rw [reqOf_eq hsx]
    rfl
  have hsz : (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size = spec.points * spec.mPts := by
    have e := wordsAt_size x ops (s := ops.length) (p := 0) hpos le_rfl
    rw [List.take_length] at e
    rw [e, hl0, Nat.min_eq_left hpc]
  unfold _root_.Flock.Firewall.Drive.linkPoints Masked.ptsE
  rw [hsz, hmp, Nat.mul_div_cancel _ hmp0]
  refine Array.map_congr_left fun p hp => ?_
  have hp' : p < spec.points := by simpa using hp
  have hle : (p + 1) * spec.mPts ≤ (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by
    rw [hsz]
    exact Nat.mul_le_mul_right _ hp'
  rw [Nat.succ_mul] at hle
  apply Array.ext
  · simp only [Array.size_extract, Array.size_map, Array.size_range]
    rw [Nat.succ_mul]
    omega
  · intro i h1 h2
    have hi : i < spec.mPts := by simpa using h2
    have hlt : p * spec.mPts + i < (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by omega
    have hw := wordsAt_slot Masked.dataE x ops hl (s := ops.length) (p := 0) (e := p * spec.mPts + i) hpos le_rfl
      (by rw [List.take_length]; exact hlt)
    rw [List.take_length] at hw
    simp only [Array.getElem_extract, Array.getElem_map, Array.getElem_range]
    exact (getElem!_pos (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0) (p * spec.mPts + i) hlt).symm.trans hw

/-- **The switch's body `k` reads its switched vector `k` alone.** -/
theorem rsBody_congr {sv sv' : ℕ → Array F128} {N : ℕ} (P : List UInt8) {k : ℕ} (h : k < N → sv k = sv' k) :
    Switch.rsBody sv N P k = Switch.rsBody sv' N P k := by
  unfold Switch.rsBody
  by_cases hk : k < N
  · rw [if_pos hk, if_pos hk, h hk]
  · rw [if_neg hk, if_neg hk]

set_option maxHeartbeats 1000000 in
/-- **The loop's switched vectors are the switched source's**: on the loop's table `tr` at rep stream `2j + r`, where
`src` samples the tape's level 0, with the zerocheck's and lincheck's claims at the points of the openings' words from
the rep's start, after openings `ops` that show no refusal by round `rankSlot + k`, the loop's extra claims (at the
link points off the first opening's words) are as many as the session's, and the loop's claim `k`'s switched vector,
`k` at most the claims' count, is the switched source's. -/
theorem switch_sv (vs : finS.SwVals) {x : Msg} {w : Flock.Wit}
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hx : Masked.Live x)
    {s0 : FlockDeploy.Ins} {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x))
    {src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray)} (hS0 : Level0Src src x w)
    (ν : finS.Nonce) (t : finS.Tape) (ops : List (Msg × Msg)) (hpos : 0 < ops.length) (hl : ops.length ≤ refK x)
    {cs : List finS.Coin} (hcs : (ops.map fun p => finS.decOf p.2) = cs) (j : Fin (Masked.jOf x)) (r : Fin 2)
    {tr : _root_.Flock.Firewall.Finish.Tab} {hir : Array ByteArray}
    (hre : realOf src x w (ν, t) (2 * j.val + r.val) = .ok (tr, hir))
    (hst : tr.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x j)) {zc : _root_.Flock.ZcOut}
    {lc : _root_.Flock.LcOut}
    (hpts : FlockComplete.LigLock.Fed.PtsAt tr.st.m tr.st.fold.kLog
      (fun p e => (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]!) (Masked.repBase x (2 * j.val + r.val)) zc lc)
    {k : ℕ} (hn : Masked.rankSlot x j.val r.val + k ≤ ops.length)
    (hN : ¬ Masked.Refused x (finS.pad (K := refK x) cs) ops.length) {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finS.pad (K := refK x) cs) spec) =
      .ok extras) :
    (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j) (_root_.Flock.Firewall.Drive.linkPoints (Masked.stAt x j)
      (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))).size = extras.size ∧
    (k < extras.size + 2 →
      _root_.Flock.Firewall.Finish.switchVec tr.g (_root_.Flock.abcClaims tr.st.m tr.st.kLog zc lc ++
        _root_.Flock.Firewall.Finish.claimsOf (_root_.Flock.Firewall.Drive.extras (Masked.stAt x j)
          (_root_.Flock.Firewall.Drive.linkPoints (Masked.stAt x j)
            (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0))))[k]! =
        ((Masked.swSrcE x (finS.pad (K := refK x) cs) w (vs x w ν (finS.pad (K := refK x) cs) (finS.seenOf x t)) t j.val).rep
          r.val).sv k) := by
  have hj := Masked.lt_of_jOf j
  have hr2 := r.isLt
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hstj : (Masked.stsOf x)[j.val]! = Masked.stAt x j := getElem!_pos (Masked.stsOf x) j.val hj
  have hm25 : 25 ≤ (Masked.stAt x j).m := (htabs j.val hj).1
  have hm35 : (Masked.stAt x j).m ≤ 35 := (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stAt x j).c.kLog := (Masked.zkTab x j).wf.1.hk6
  have hkm : (Masked.stAt x j).c.kLog ≤ (Masked.stAt x j).m := (Masked.zkTab x j).wf.1.hk
  have hfk : (Masked.stAt x j).circuitFold.kLog = (Masked.stAt x j).c.kLog := rfl
  have hzq : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [Masked.stsOf_half]
    exact (Masked.zkTab x j).hz
  have hrk : Masked.rankSlot x j.val r.val =
      Masked.repBase x (2 * j.val + r.val) + ((Masked.stAt x j).m + (Masked.stAt x j).c.kLog - 6) := by
    show Masked.repBase x (2 * j.val + r.val) + ((Masked.stsOf x)[j.val]!.m + (Masked.stsOf x)[j.val]!.c.kLog - 6) = _
    rw [hstj]
  -- the opened coins' words at the algebraic phase's rounds
  obtain ⟨h1, h3, h5, -, -⟩ := loopNr_piop hsx hq2 hzq (by rw [Masked.stsOf_half]; omega)
    (by rw [Masked.stsOf_half]; exact hk6)
  rw [Masked.stsOf_half, ← Masked.repBase_eq hsx (2 * j.val + r.val)] at h1 h3 h5
  have hCM : _root_.Flock.Firewall.Frames.COIN_MAX = 65536 := rfl
  have hW : ∀ p e, p < ops.length → e < min (loopNr x p) _root_.Flock.Firewall.Frames.COIN_MAX →
      (wordsAt x (ops.map fun p => CoinLayer.rd p.2) p)[e]! = Masked.slotAt (finS.pad (K := refK x) cs) p e := by
    intro p e hp he
    have hw := wordsAt_slot Masked.dataE x ops hl (s := ops.length) (p := p) (e := e) hp le_rfl
      (by rw [wordsAt_size x ops hp le_rfl]; exact he)
    rw [List.take_length] at hw
    exact hw.trans (congrArg (fun c => Masked.slotAt (finS.pad (K := refK x) c) p e) hcs)
  have htm : tr.st.m = (Masked.stAt x j).m := by rw [hst]; rfl
  have htk : tr.st.fold.kLog = (Masked.stAt x j).c.kLog := by rw [hst]; rfl
  have htk' : tr.st.kLog = (Masked.stAt x j).c.kLog := by rw [hst]; rfl
  rw [htm, htk] at hpts
  have hptsC : FlockComplete.LigLock.Fed.PtsAt (Masked.stAt x j).m (Masked.stAt x j).c.kLog
      (fun p e => Masked.slotAt (finS.pad (K := refK x) cs) p e) (Masked.repBase x (2 * j.val + r.val)) zc lc :=
    FlockComplete.LigLock.Fed.ptsAt_congr
      (fun i hi => hW _ _ (by omega) (by rw [h1, hCM, Nat.min_eq_left (by omega)]; exact hi))
      (fun i hi => hW _ _ (by omega) (by rw [h3 i hi, hCM]; decide))
      (fun i hi => hW _ _ (by omega) (by rw [h5 i hi, hCM]; decide)) hpts
  -- the session's claims at the same points
  have hpS : FlockComplete.LigLock.Fed.PtsAt (Masked.stAt x j).m (Masked.stAt x j).c.kLog
      (fun p e => Masked.slotAt (finS.pad (K := refK x) cs) p e) (Masked.repBase x (2 * j.val + r.val))
      (FlockComplete.Replay.Zc.out (Masked.stAt x j).m (Masked.zcE (finS.pad (K := refK x) cs) j r)
        (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!)
      (FlockComplete.Replay.Lc.out (Masked.stAt x j).circuitFold (Masked.stAt x j).m
        (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!
        (FlockComplete.Replay.Zc.out (Masked.stAt x j).m (Masked.zcE (finS.pad (K := refK x) cs) j r)
          (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!)
        (Masked.lcE (finS.pad (K := refK x) cs) j r)) :=
    FlockComplete.LigLock.Fed.ptsAt_out (Masked.stAt x j).m (Masked.stAt x j).circuitFold
      (fun p e => Masked.slotAt (finS.pad (K := refK x) cs) p e) (Masked.repBase x (2 * j.val + r.val))
      (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!
  -- the loop's extra claims are at the session's link points
  have heS : (Masked.stAt x j).extraClaims (Masked.ptsE (finS.pad (K := refK x) cs) spec) = .ok extras := by
    rw [Masked.tabsOf_stAt] at hext
    exact hext
  have hXE : extras.map (·.x) = _root_.Flock.Firewall.Drive.extras (Masked.stAt x j)
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stAt x j) (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
    have e := linkPoints_slot hx hsx hpos hl hq2
    rw [Masked.stsOf_half] at e
    rw [extraClaims_xs heS, ← hcs, ← e]
  refine ⟨by rw [← hXE, Array.size_map], fun hk => ?_⟩
  have hkE : k < (_root_.Flock.abcClaims (Masked.stAt x j).m (Masked.stAt x j).c.kLog
      (FlockComplete.Replay.Zc.out (Masked.stAt x j).m (Masked.zcE (finS.pad (K := refK x) cs) j r)
        (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!)
      (FlockComplete.Replay.Lc.out (Masked.stAt x j).circuitFold (Masked.stAt x j).m
        (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!
        (FlockComplete.Replay.Zc.out (Masked.stAt x j).m (Masked.zcE (finS.pad (K := refK x) cs) j r)
          (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!)
        (Masked.lcE (finS.pad (K := refK x) cs) j r)) ++ extras).size := by
    rw [Array.size_append, FlockComplete.abcClaims_size]
    omega
  have hxk := FlockComplete.LigLock.Fed.ptsAt_claims_x (kL := (Masked.stAt x j).c.kLog) hptsC hpS
    (fun k => claimsOf_x hXE k) k
  have hmem : ∀ (a : Array _root_.Flock.Claim) (i : ℕ), i < a.size → a[i]! ∈ a := fun a i hi => by
    rw [getElem!_pos a i hi]
    exact Array.getElem_mem hi
  have hx6 := FlockComplete.Replay.claims_xsize' (Masked.stAt x j) hkm (by omega)
    (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! (Masked.zcE (finS.pad (K := refK x) cs) j r)
    (Masked.lcE (finS.pad (K := refK x) cs) j r) (Masked.ptsE (finS.pad (K := refK x) cs) spec) extras heS _
    (hmem _ k hkE)
  rw [← hxk] at hx6
  have hpk : Masked.ligTab x w t j =
      FlockComplete.packed (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! (Masked.stAt x j).m := by
    show FlockComplete.packed (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]!
      (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.m = _
    rw [Masked.tabsOf_stAt]
    rfl
  rw [htm, htk', (hS0.realOf ν t j r hre).2.2.2, hpk,
    FlockComplete.switchVec_packed (by omega) (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! _ hx6,
    FlockComplete.Replay.switchVec_x hxk]
  exact (Masked.swSrc_sv (finS.pad (K := refK x) cs) hR t j r
    (vs x w ν (finS.pad (K := refK x) cs) (finS.seenOf x t)) hx hN hsx heS k hkE (by omega)).symm

end Loop

end Claims.ZKServer
