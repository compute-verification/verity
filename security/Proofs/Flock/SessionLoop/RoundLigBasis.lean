module

public import Proofs.Flock.SessionLoop.FinL0Vals
public import Proofs.Flock.Claims.ZK.FinishSwRep

/-!
# The basis at a rep's Ligerito rounds, from the frame's coins

Past the ring switch the frame's run (`LeanRelease.release_lig_frame`) opens at the basis of `Rs.sws` on the rep's
claims at its algebraic coins `algC` and on the history's coins `cs` from the switch on; the switched session's basis is
`Masked.ligBasis`, in closed form at the opened coins. Below the history's length both read the same coins.

* `Loop.linkPoints_ptsE`: at a history of at most `refK x` openings, the link points off the `Commit`'s coin are the
  session's at the opened coins (`linkPoints_wordsAt` at a full history).
* **`Loop.basis_lig`**: the frame's basis is `Masked.ligBasis` (`sws_of_pts`, `loopNr_piop`).
-/

@[expose] public section

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **The request's link points are the session's at a shorter history**: at a statement in the masked scope with a
witness, where at least one and at most `refK x` openings came, rep stream `k`'s link points off the `Commit`'s coin are
the session's at the opened coins. -/
theorem linkPoints_ptsE {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    {ops : List (Msg × Msg)} (hl : ops.length ≤ refK x) (hpos : 0 < ops.length) {s0 : FlockDeploy.Ins}
    {spec : SessionSpec} (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) {k : ℕ}
    (hk : k < 2 * spec.tables.size) :
    _root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]!
        (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0) = Masked.ptsE (finCoins x ops) spec := by
  have hx : Masked.Live x := Masked.live_of hsc hR
  have hrs : spec.repStreams.size = 2 * spec.tables.size := FlockDeploy.repStreams_size spec
  obtain ⟨hmp, hmp0, hpc⟩ := Masked.live_links hx hsx (k := k) (by omega)
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
  · intro j _ h2
    have hj : j < spec.mPts := by simpa using h2
    have hlt : p * spec.mPts + j < (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0).size := by omega
    have hw := wordsAt_slot Masked.dataE x ops hl (s := ops.length) (p := 0) (e := p * spec.mPts + j) hpos le_rfl
      (by rw [List.take_length]; exact hlt)
    rw [List.take_length] at hw
    simp only [Array.getElem_extract, Array.getElem_map, Array.getElem_range]
    exact (getElem!_pos (wordsAt x (ops.map fun e => CoinLayer.rd e.2) 0) (p * spec.mPts + j) hlt).symm.trans hw

set_option maxHeartbeats 1000000 in
/-- **The frame's basis is the session's**: at a statement in the masked scope with a witness and a history of at most
`refK x` openings past table `j`'s rep `r`'s first Ligerito round (`Masked.ligP0`), where the rep's algebraic coins
`algC` are the history's from its first round (`Masked.repBase`), the coins `cs` the history's from there on, `xs` the
request's extra coordinates and the claims' points `PtsAt` `algC`'s, the basis of `Rs.sws` on the claims at `cs` is
`Masked.ligBasis` at the opened coins. -/
theorem basis_lig {x : Msg} {w : Flock.Wit}
    (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w)
    {ops : List (Msg × Msg)} (hl : ops.length ≤ refK x) {s0 : FlockDeploy.Ins} {spec : SessionSpec}
    (hsx : FlockDeploy.setupOf true x = some (s0, spec, Masked.stsOf x)) (j : Fin (Masked.jOf x)) (r : Fin 2)
    (t : finS.Tape) {st : _root_.Flock.Setup} (hst : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = st)
    {extras : Array _root_.Flock.Claim}
    (hext : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]!.extra (Masked.ptsE (finCoins x ops) spec) = .ok extras)
    {xs algC : Array (Array F128)} {base : ℕ} {cs : List (Array F128)}
    (hxs : xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[j.val]!
      (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[j.val]!
        (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)))
    (hanchor : Masked.repBase x (2 * j.val + r.val) + algC.size = base)
    (halg : ∀ a (ha : a < algC.size),
      algC[a] = wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base - algC.size + a))
    (hbcs : base + cs.length = ops.length)
    (hcsv : ∀ a (ha : a < cs.length), cs[a] = wordsAt x (ops.map fun p => CoinLayer.rd p.2) (base + a))
    (hbase : base + (xs.size + 2) + 1 = Masked.ligP0 x j r) (hlen : Masked.ligP0 x j r < ops.length)
    {zc : _root_.Flock.ZcOut} {lc : _root_.Flock.LcOut}
    (hpts : FlockComplete.LigLock.Fed.PtsAt st.m st.fold.kLog (FlockComplete.LigLock.Fed.recCoin algC) 0 zc lc) :
    FlockComplete.basisVals
        (FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims st.m st.kLog zc lc ++
            _root_.Flock.Firewall.Finish.claimsOf xs)
          ⟨fun k i => (cs.getD k #[])[i]!, fun i => (cs.getD (xs.size + 2) #[])[i]!⟩) (st.m - 7) =
      Masked.ligBasis x (finCoins x ops) w t j r extras := by
  have hx := Masked.live_of hsc hR
  have hj := Masked.lt_of_jOf j
  obtain ⟨-, -, hspec, htabs⟩ := FlockComplete.ZkTables.tables_of_setup hsc hsx
  have hq : 2 * j.val + r.val < 2 * spec.tables.size := by rw [hspec]; omega
  have hq2 : 2 * j.val + r.val < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hjj : (2 * j.val + r.val) / 2 = j.val := by omega
  have hst' : (Masked.stsOf x)[(2 * j.val + r.val) / 2]! = Masked.stAt x j := Masked.stsOf_half x j r
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((Masked.zkTab x j).sch, (Masked.zkTab x j).tPad) := by
    rw [hst']
    exact (Masked.zkTab x j).hz
  have hm25 : 25 ≤ (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m := by
    rw [hst']
    exact (htabs j.val hj).1
  have hm35 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m ≤ 35 := by
    rw [hst']
    exact (htabs j.val hj).2.1
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog := by
    rw [hst']
    exact (Masked.zkTab x j).wf.1.hk6
  obtain ⟨hpts2, -, hregs⟩ := Masked.live_bounds hx hsx
  have hpr : spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size ≤ 2048 := by
    rw [hst', hpts2]
    have := hregs j.val hj
    exact Nat.le_trans (Nat.mul_le_mul_left 2 this) (by norm_num)
  have hj' : j.val < (FlockDeploy.tabsOf (Masked.stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have htabj : (FlockDeploy.tabsOf (Masked.stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (Masked.stAt x j) := by
    rw [getElem!_pos _ j.val hj']
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hm1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m = st.m := by rw [hst', ← hst, htabj]; rfl
  have hk1 : (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog = st.fold.kLog := by rw [hst', ← hst, htabj]; rfl
  -- the extra claims
  have hlp := linkPoints_ptsE hsc hR hl (by omega) hsx hq
  rw [hjj] at hlp
  have hxn : xs.size = spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size := by
    rw [hxs, Count.extras_size, hlp, show (Masked.ptsE (finCoins x ops) spec).size = spec.points by
      simp [Masked.ptsE], Nat.mul_comm, hjj]
  have hxe : extras.map (·.x) = xs := by
    rw [hxs, hlp]
    rw [getElem!_pos (FlockDeploy.tabsOf (Masked.stsOf x)) j.val hj'] at hext
    simp only [FlockDeploy.tabsOf, Array.getElem_map] at hext
    apply extraClaims_xs
    rw [getElem!_pos (Masked.stsOf x) j.val hj]
    exact hext
  -- where the rep's rounds sit
  have hP0 : Masked.ligP0 x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) +
      FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! :=
    (Masked.repSlot_eq hsx (q := 2 * j.val + r.val) hz (fun _ => 0)).trans (Nat.add_zero _)
  have hPR : FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j.val + r.val) / 2]! =
      (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.m + (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.c.kLog - 6 +
        (2 + spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size + 1) := rfl
  obtain ⟨Rz, hRz⟩ : ∃ Rz, FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j.val + r.val) = Rz := ⟨_, rfl⟩
  have hRB : Masked.repBase x (2 * j.val + r.val) = Rz := (Masked.repBase_eq hsx _).trans hRz
  rw [hP0, hPR, hRz] at hbase
  rw [hP0, hPR, hRz] at hlen
  rw [hRB] at hanchor
  have hA : algC.size = 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) := by omega
  have hbR : base = Rz + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) := by omega
  obtain ⟨hp1, hp3, hp5, hp7, hpN⟩ := loopNr_piop hsx hq2 hz (by omega) hk6
  simp only [hm1, hk1, hRz] at hp1 hp3 hp5 hp7 hpN
  -- the history's words are the opened coins
  have hWC : ∀ p e, p < ops.length → e < loopNr x p → e < 65536 →
      (wordsAt x (ops.map fun q => CoinLayer.rd q.2) p)[e]! = Masked.ligCoin (finCoins x ops) p e := by
    intro p e hp he hc
    have hs := wordsAt_size x ops hp le_rfl
    rw [List.take_length] at hs
    have h := wordsAt_slot Masked.dataE x ops hl hp le_rfl (e := e)
      (by rw [List.take_length, hs]; exact Nat.lt_min.mpr ⟨he, hc⟩)
    rw [List.take_length] at h
    exact h
  -- the claims' points at the history's words
  have hrec : ∀ k i, k < algC.size → FlockComplete.LigLock.Fed.recCoin algC k i =
      (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (Rz + k))[i]! := by
    intro k i hk
    rw [congrFun (FlockComplete.LigLock.Fed.recCoin_of (Array.getElem?_eq_getElem hk)) i, halg k hk,
      show base - algC.size + k = Rz + k by omega]
  have hptsW : FlockComplete.LigLock.Fed.PtsAt st.m st.fold.kLog
      (fun p e => (wordsAt x (ops.map fun q => CoinLayer.rd q.2) p)[e]!) Rz zc lc := by
    refine FlockComplete.LigLock.Fed.ptsAt_congr (fun i hi => ?_) (fun i hi => ?_) (fun i hi => ?_) hpts
    · exact hrec (0 + 1) i (by omega)
    · exact (hrec (0 + 3 + i) 0 (by omega)).trans (by rw [show Rz + (0 + 3 + i) = Rz + 3 + i by omega])
    · exact (hrec (0 + 3 + (st.m - 6) + 2 + i) 0 (by omega)).trans
        (by rw [show Rz + (0 + 3 + (st.m - 6) + 2 + i) = Rz + 3 + (st.m - 6) + 2 + i by omega])
  -- the switched vectors
  have h1 : ∀ i < st.m - 13, (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (Rz + 1))[i]! =
      Masked.ligCoin (finCoins x ops) (Rz + 1) i :=
    fun i hi => hWC _ _ (by omega) (by rw [hp1]; omega) (by omega)
  have h3 : ∀ i < st.m - 6, (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (Rz + 3 + i))[0]! =
      Masked.ligCoin (finCoins x ops) (Rz + 3 + i) 0 :=
    fun i hi => hWC _ _ (by omega) (by rw [hp3 i hi]; omega) (by omega)
  have h5 : ∀ i < st.fold.kLog - 6,
      (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (Rz + 3 + (st.m - 6) + 2 + i))[0]! =
        Masked.ligCoin (finCoins x ops) (Rz + 3 + (st.m - 6) + 2 + i) 0 :=
    fun i hi => hWC _ _ (by omega) (by rw [hp5 i hi]; omega) (by omega)
  have hr : ∀ k < 2 + xs.size, ∀ i < 7, (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + k))[i]! =
      Masked.ligCoin (finCoins x ops) (base + k) i := by
    intro k hk i hi
    have e7 := hp7 k (by omega)
    rw [show Rz + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) + k = base + k by omega] at e7
    exact hWC _ _ (by omega) (by rw [e7]; omega) (by omega)
  have hg : ∀ i < 2 + xs.size, (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + (2 + xs.size)))[i]! =
      Masked.ligCoin (finCoins x ops) (base + (2 + xs.size)) i := by
    intro i hi
    have eN := hpN
    rw [show Rz + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6) +
        (2 + spec.points * (Masked.stsOf x)[(2 * j.val + r.val) / 2]!.regions.size) = base + (2 + xs.size) by
      omega] at eN
    exact hWC _ _ (by omega) (by rw [eN]; omega) (by omega)
  have e := sws_of_pts (st := st) (W := fun p e => (wordsAt x (ops.map fun q => CoinLayer.rd q.2) p)[e]!)
    (C := Masked.ligCoin (finCoins x ops)) (R := Rz) (L₁ := base)
    (L₂ := Rz + 3 + (st.m - 6) + 3 + (st.fold.kLog - 6)) (zc := zc) (lc := lc) (xs := xs) (extras := extras)
    (Masked.trsE x w (Masked.UE.tapeZk t))[j.val]! hptsW h1 h3 h5 hbR hr hg hxe
  beta_reduce at e
  -- the frame's coins from the switch on are the history's words there
  have hcs : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ _root_.Flock.Firewall.Finish.claimsOf xs).size =
      2 + xs.size := by
    rw [Array.size_append, FlockComplete.abcClaims_size]
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
  have hfw : FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims st.m st.kLog zc lc ++
        _root_.Flock.Firewall.Finish.claimsOf xs)
        ⟨fun k i => (cs.getD k #[])[i]!, fun i => (cs.getD (xs.size + 2) #[])[i]!⟩ =
      FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims st.m st.kLog zc lc ++
        _root_.Flock.Firewall.Finish.claimsOf xs)
        ⟨fun k e => (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + k))[e]!,
          fun e => (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + (2 + xs.size)))[e]!⟩ := by
    refine FlockComplete.LigLock.Fed.sws_coins_congr _ (fun k hk i _ => ?_) (fun k _ => ?_)
    · rw [hcs] at hk
      show (cs.getD k #[])[i]! = (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + k))[i]!
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some, hcsv k (by omega)]
    · show (cs.getD (xs.size + 2) #[])[k]! =
        (wordsAt x (ops.map fun q => CoinLayer.rd q.2) (base + (2 + xs.size)))[k]!
      rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (by omega), Option.getD_some,
        hcsv _ (by omega), Nat.add_comm xs.size 2]
  have hzc : ∀ L, FlockComplete.Replay.Zc.coinsAt (Masked.ligRespE (finCoins x ops)) L =
      FlockComplete.zcCoins (Masked.ligCoin (finCoins x ops)) L := fun L => Masked.zcCoins_ligRespE _ L
  have hlc : ∀ L n, FlockComplete.Replay.Lc.coinsAt (Masked.ligRespE (finCoins x ops)) L n =
      FlockComplete.lcCoins (Masked.ligCoin (finCoins x ops)) L n := fun L n => Masked.lcCoins_ligRespE _ L n
  have hrc : ∀ L n, FlockComplete.Replay.Rs.coinsAt (Masked.ligRespE (finCoins x ops)) L n =
      ⟨fun k => Masked.ligCoin (finCoins x ops) (L + k), Masked.ligCoin (finCoins x ops) (L + n)⟩ :=
    fun L n => Masked.rsCoins_ligRespE _ L n
  unfold Masked.ligBasis Masked.ligSt Masked.swsE
  rw [hst, hzc, hlc, hrc, hRB]
  refine congrArg (fun s => FlockComplete.basisVals s (st.m - 7)) ?_
  rw [hfw]
  refine Eq.trans ?_ (e.trans ?_)
  · with_reducible rfl
  · with_reducible rfl

end Loop

end Claims.ZKServer
