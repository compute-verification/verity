import Proofs.Flock.Claims.ZK.FinishFwTarget
import Proofs.Flock.Claims.Complete.ZkTables.LaneRows

/-!
# A deployed rep's level-0 restart is `Σ y₁'·bFold`

`fwRows_at` takes the firewall's level-0 restart `t'` as `Σ_col y₁'(col)·bFold(col)`. Here that is read off the deployed
rep: `repFwC`'s Ligerito on table `j`'s rep `r`, answered by the opened coins (`ligCoin`) from the rep's base, on the
packed table of the witness's trace (`ligTab`) and the tape's extra lanes, with extra claims at the statement's region
claims' points.

* `FlockComplete.Replay.rsSws_congr_x`: the switched claims read only the claims' points.
* `Masked.xs_pt`: extra claims at the link's region points (`Drive.extras`) are the points `openB_basisOf` reads.
* `Masked.wS_foldKs`: the weights after `k` lane challenges are the executable's iterated fold.
* **`Masked.fwRestart_at`**: at the deployed rep, `t' = Σ_col y₁'(col)·bFold(col)`. The run's lane phase is
  `LaneForm`'s (`ligeritoPZk_lanes`): `t' = ⟨W_{k₀}, f_{k₀} + ρ₀R_{2r} + ρ₁R_{2r+1}⟩`. Its lane coins, pair coins and
  out-of-domain point are `czExe`'s from `p₀` (`repSlot_eq`), its switched claims the deployed basis's (`swsE`, by
  `rsSws_congr_x`), so `W_{k₀}` is `bFold` (`laneB_wSOf`, `openB_basisOf`) and the blinded table is `y₁'` (`tab_y1_L`,
  `wordOf_packed`, the tape's extra lanes by `agrees0_exe`).

Neither reads the worker's clear values: `y₁'` reads the witness lanes, the mask slot and the extra lanes.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete.Replay

/-- **The switched claims read only the claims' points.** -/
theorem rsSws_congr_x {A B : Array _root_.Flock.Claim} (cs : Rs.Challenges) (hsz : A.size = B.size)
    (hx : ∀ k : ℕ, A[k]!.x = B[k]!.x) : Rs.sws A cs = Rs.sws B cs := by
  have hp : Rs.perClaim A cs = Rs.perClaim B cs := by
    unfold Rs.perClaim
    rw [hsz]
    congr 1
    funext k
    rw [hx k]
  unfold Rs.sws
  rw [hp, hsz]

end FlockComplete.Replay

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)

namespace Masked

/-- **Extra claims at the link's region points** (`Drive.extras` at the coins' link points) have the points `extras_pt`
reads off the statement's extra claims. -/
theorem xs_pt {x : Msg} (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) {s spec}
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (xs : Array _root_.Flock.Claim)
    (hxsz : xs.size = spec.points * (stAt x j).regions.size)
    (hxs : xs.map (·.x) = _root_.Flock.Firewall.Drive.extras (stAt x j) (ptsE c spec)) (i : ℕ)
    (hi : i < 2 * (stAt x j).regions.size) :
    (xs[i]!).x.size = (stAt x j).m - 6 ∧
      ((xs[i]!).x.extract 1).toList = Composed.regionPtOf (stAt x j)
        ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link ⟨i, hi⟩ := by
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm14' : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  obtain ⟨hpts, hmp⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  set ρ := ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) with hρdef
  have hmP : (stAt x j).mPts = spec.mPts := hmp j.val hj
  have hmx : mPtsOf x = spec.mPts := by unfold mPtsOf; rw [hs]
  have hlink : (ptsE c spec).toList = [Array.ofFn (ρ.link 0), Array.ofFn (ρ.link 1)] := by
    have hp : ∀ p : Fin 2, (Array.range spec.mPts).map (fun i => slotAt c 0 (p.val * spec.mPts + i)) =
        Array.ofFn (ρ.link p) := by
      intro p
      apply Array.ext
      · simp [hmP]
      · intro i h1 h2
        rw [Array.getElem_map, Array.getElem_range, Array.getElem_ofFn]
        show slotAt c 0 (p.val * spec.mPts + i) = slotAt c 0 (p.val * mPtsOf x + i)
        rw [hmx]
    apply List.ext_getElem
    · simp [ptsE, hpts]
    · intro i h1 h2
      have hi2 : i < 2 := by simpa [ptsE, hpts] using h1
      rw [Array.getElem_toList]
      simp only [ptsE, Array.getElem_map, Array.getElem_range]
      interval_cases i
      · exact hp 0
      · exact hp 1
  have hptsE : ptsE c spec = #[Array.ofFn (ρ.link 0), Array.ofFn (ρ.link 1)] := Array.ext' hlink
  have hlt : i < xs.size := by rw [hxsz, hpts]; omega
  have hif : (if i % 2 = 0 then Array.ofFn (ρ.link 0) else Array.ofFn (ρ.link 1)) =
      Array.ofFn (ρ.link (if i % 2 = 0 then 0 else 1)) := by
    split <;> rfl
  have hxe : (xs[i]!).x = ((stAt x j).regionPoint (stAt x j).regions[i / 2]!
      (Array.ofFn (ρ.link (if i % 2 = 0 then 0 else 1)))).extract 6
      ((stAt x j).regionPoint (stAt x j).regions[i / 2]!
        (Array.ofFn (ρ.link (if i % 2 = 0 then 0 else 1)))).size := by
    have h1 : (xs[i]!).x = (xs.map (·.x))[i]! := by
      rw [getElem!_pos xs i hlt, getElem!_pos _ i (by rw [Array.size_map]; exact hlt), Array.getElem_map]
    rw [h1, hxs, hptsE, driveExtras_get _ _ _ i hi, hif]
  refine ⟨?_, ?_⟩
  · rw [hxe, Array.size_extract, FlockSoundness.Refine.regionPoint_size]; omega
  · rw [hxe, extract_six_one]
    rfl

/-- **The weights are the executable's iterated fold**: the weights after `k` lane challenges, lowest variable first. -/
theorem wS_foldKs (coin : ℕ → ℕ → _root_.Flock.F128) (p0 : ℕ) (b : Array _root_.Flock.F128) (n : ℕ) :
    ∀ k, FlockComplete.LigLock.wS coin p0 b n k =
      FlockComplete.Replay.Lg.foldKs ((List.range k).map (FlockComplete.LigLock.chS coin p0))
        (FlockComplete.addScaled (FlockComplete.embedF b)
          (FlockComplete.embedF (_root_.Flock.eqTable (FlockComplete.LigLock.zS coin p0 n))) (coin (p0 + 1) 0))
  | 0 => rfl
  | k + 1 => by
    rw [FlockComplete.LigLock.wS, wS_foldKs coin p0 b n k, List.range_succ, List.map_append]
    unfold FlockComplete.Replay.Lg.foldKs
    rw [List.foldl_append]
    rfl

set_option maxHeartbeats 4000000 in
/-- **A deployed rep's level-0 restart is `Σ y₁'·bFold`**: where `repFwC` on table `j`'s rep `r`, answered by the opened
coins from the rep's base at the packed table of the witness's trace (`hg`) and the tape's extra lanes (`hext`),
returns with extra claims at the region points of the link (`Drive.extras`, `hxsz`, `hxs`), its proof's level-0
restart `t'` is `Σ_col y₁'(col)·bFold(col)` at the tape's mask slot and extra lanes. -/
theorem fwRestart_at (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (xs : Array _root_.Flock.Claim) (hxsz : xs.size = spec.points * (stAt x j).regions.size)
    (hxs : xs.map (·.x) = _root_.Flock.Firewall.Drive.extras (stAt x j) (ptsE c spec))
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array _root_.Flock.F128)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (hext : extra = (exeTape τ).extra j.val)
    (salt : ℕ → ℕ → ByteArray) {t₀ : _root_.Flock.Transcript} {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript}
    {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch tPad r.val dom g cm0 xs pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l')) :
    gOf p.level0.tPrime = ∑ col, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.y1' (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (czExe PE x c j) r col.val := by
  subst hg hext
  -- the run's steps
  dsimp only [_root_.Flock.Firewall.Finish.repFwC] at h
  obtain ⟨⟨zp, zc⟩, t₁, l₁, h₁, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lp, lc⟩, t₂, l₂, h₂, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨ss, T, sws⟩, t₃, l₃, h₃, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨⟨lg, zl, rec⟩, t₄, l₄, h₄, h⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  try dsimp only at h
  obtain ⟨tr', t₅, l₅, h₅, hg⟩ := _root_.Flock.Firewall.Finish.run_bind_ok h
  rw [_root_.Flock.Firewall.Finish.get_run_go] at h₅
  simp only [Prod.mk.injEq, Except.ok.injEq] at h₅
  obtain ⟨⟨htr, ht₅⟩, hl₅⟩ := h₅
  subst tr' t₅ l₅
  try dsimp only at hg
  obtain ⟨⟨rp, tt⟩, t₆, l₆, h₆, hr₆⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hg
  obtain ⟨hrp, ht₆, hl₆⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₆
  subst t₆ l₆
  try dsimp only at hr₆
  obtain ⟨lgc, t₇, l₇, h₇, hc⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hr₆
  obtain ⟨hlgc, ht₇, hl₇⟩ := _root_.Flock.Firewall.Finish.liftV_run_ok h₇
  subst t₇ l₇
  try dsimp only at hc
  split at hc
  · obtain ⟨_, _, _, h', _⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
    rw [_root_.Flock.Firewall.Finish.throw_run_go] at h'
    cases h'
  have hzl : p.level0 = zl := by
    first
    | (obtain ⟨_, _, _, -, hp⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
       simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
       obtain ⟨⟨rfl, -⟩, -⟩ := hp
       rfl)
    | (obtain ⟨_, _, _, -, hc'⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
       obtain ⟨_, _, _, -, hp⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc'
       simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
       obtain ⟨⟨rfl, -⟩, -⟩ := hp
       rfl)
  -- the setup is the statement's
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, h2, -⟩, hlv, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hk6' : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm14' : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
  have hm14 : 14 ≤ st.m := by
    rw [hSm]
    exact hm14'
  -- the runs' lengths, two extra claims per region
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hA2 : (_root_.Flock.abcClaims st.m st.kLog zc lc).size = 2 := FlockComplete.abcClaims_size _ _ _ _
  have hN : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size = 2 + 2 * (stAt x j).regions.size := by
    rw [Array.size_append, hA2, hxsz, hpts]
  have hxs2 : xs.size = (stAt x j).regions.size * 2 := by rw [hxsz, hpts, Nat.mul_comm]
  obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
  obtain ⟨hlc, hl₂⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
  obtain ⟨hsws, hl₃, -⟩ := FlockComplete.ringSwitchCut_sws (ligCoin c) n _ _ (by rw [hN]; omega) h₃
  have hNx : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size = 2 + xs.size := by
    rw [Array.size_append, hA2]
  rw [hNx] at hsws
  rw [hN] at hl₃
  have hL2 : l₂.size = repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).c.kLog - 6) := by
    rw [hl₂, hl₁, hl, hkL, hSk, hSm]
  -- level 0 starts at `p₀`
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hp0 : l₃.size = ligP0 x j r := by
    show l₃.size = repSlot x (2 * j.val + r.val) (fun _ => 0)
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk
    rw [hpts]
    try dsimp only
    omega
  -- the switched claims are the deployed basis's
  have hswsE : sws = swsE x c w τ j r xs := by
    have hzc' : FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val)) = zcE c j r := by
      unfold FlockComplete.Replay.Zc.coinsAt
      simp only [coinOf_ligRespE]
      rfl
    have hlc' : FlockComplete.Replay.Lc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6))
        ((stAt x j).c.kLog - 6) = lcE c j r := by
      unfold FlockComplete.Replay.Lc.coinsAt
      simp only [coinOf_ligRespE]
      rfl
    subst hst
    set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
    have hzE : FlockComplete.zcCoins (ligCoin c) l.size = zcE c j r := by
      rw [hl]
    have hlE : FlockComplete.lcCoins (ligCoin c) l₁.size
        ((_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog - 6) = lcE c j r := by
      rw [hl₁, hl]
      rfl
    rw [hzE] at hzc
    rw [hlE] at hlc
    have hab' := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
      (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
      (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
      (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc.2.1 hzc.2.2 hlc.2
    have hxk := FlockComplete.Replay.claims_x (e := xs) (FlockComplete.abcClaims_size _ _ _ _)
      (FlockComplete.abcClaims_size _ _ _ _) hab'.1 hab'.2
    rw [hsws, FlockComplete.Replay.rsSws_congr_x _
      (by simp only [Array.size_append, FlockComplete.abcClaims_size]) hxk]
    unfold swsE
    rw [tabsOf_stAt]
    show _ = FlockComplete.Replay.Rs.sws (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
      (FlockComplete.Replay.Zc.out (stAt x j).m
        (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) tr)
      (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m
          (FlockComplete.Replay.Zc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val))) tr)
        (FlockComplete.Replay.Lc.coinsAt (ligRespE c) (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6))
          ((stAt x j).c.kLog - 6))) ++ xs)
      (FlockComplete.Replay.Rs.coinsAt (ligRespE c)
        (repBase x (2 * j.val + r.val) + 3 + ((stAt x j).m - 6) + 3 + ((stAt x j).c.kLog - 6)) (2 + xs.size))
    rw [hzc', hlc']
    refine congrArg₂ FlockComplete.Replay.Rs.sws rfl ?_
    rw [hL2]
    unfold FlockComplete.Replay.Rs.coinsAt
    simp only [coinOf_ligRespE] <;> rfl
  have hb : FlockComplete.basisVals sws (st.m - 7) = ligBasis x c w τ j r xs := by
    rw [hswsE]
    show _ = FlockComplete.basisVals (swsE x c w τ j r xs) ((FlockDeploy.tabsOf (stsOf x))[j.val]!.m - 7)
    rw [tabsOf_stAt, hst]
  -- the lane phase in closed form
  have hS4 := FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _
    _ _ _ t₃ l₃ _ _ h₄
  have hLF := FlockComplete.L0.ligeritoPZk_lanes (respE c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by omega) hk0 hS4
  have hch : FlockComplete.ZkTables.Level0.laneCh (respE c) (zkTab x j).sch.k0 (ligP0 x j r) =
      (List.range (zkTab x j).sch.k0).map (FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r)) := by
    unfold FlockComplete.ZkTables.Level0.laneCh
    refine List.map_congr_left fun q _ => ?_
    simp only [FlockComplete.coinKOf, coinOf_respE, FlockComplete.LigLock.chS, ligCoin] <;> rfl
  have hρL : FlockComplete.ZkTables.Level0.laneRho (respE c) (zkTab x j).sch.k0 (ligP0 x j r) =
      FlockComplete.LigLock.rhoS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.k0 := by
    unfold FlockComplete.ZkTables.Level0.laneRho FlockComplete.LigLock.rhoS
    simp only [FlockComplete.coinKOf, coinOf_respE, ligCoin] <;> rfl
  have hW0 : FlockComplete.ZkTables.Level0.laneW0 (respE c) (zkTab x j).sch.n (ligP0 x j r)
        (FlockComplete.basisVals sws (st.m - 7)) =
      FlockComplete.addScaled (FlockComplete.embedF (FlockComplete.basisVals sws (st.m - 7)))
        (FlockComplete.embedF (_root_.Flock.eqTable (FlockComplete.LigLock.zS (ligCoin c) (ligP0 x j r)
          (zkTab x j).sch.n))) (ligCoin c (ligP0 x j r + 1) 0) := by
    unfold FlockComplete.ZkTables.Level0.laneW0 FlockComplete.ZkTables.Level0.lanePoint FlockComplete.LigLock.zS
    simp only [coinOf_respE, ligCoin] <;> rfl
  rw [hzl, hLF.tPrime, hp0, hch, hρL, hW0,
    ← tabS_foldKs (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0,
    ← wS_foldKs (ligCoin c) (ligP0 x j r) (FlockComplete.basisVals sws (st.m - 7)) (zkTab x j).sch.n
      (zkTab x j).sch.k0, hb, FlockComplete.Replay.gOf_dotK]
  -- the folded weights are `bFold`
  have hB := laneB_wSOf w τ c hs j r xs fun y hy =>
    openB_basisOf hx w τ c hs j r xs hxs2 (fun i hi => xs_pt hx c hs j xs hxsz hxs i hi) y hy
  obtain ⟨hWsz, hW⟩ := hB (zkTab x j).sch.k0 le_rfl
  -- the blinded table is `y₁'`
  have hY : ∀ col (hcol : col < (tabC PE x j).z.L),
      ((tabC PE x j).T (czExe PE x c j)).tab.y1' (wtOf PE x w j) ((UE.tape x τ).1 j).1
          ((UE.tape x τ).1 j).2.1.2.1 r ⟨col, hcol⟩ =
        FlockComplete.Replay.Lg.gv (FlockComplete.ZkTables.Level0.blinded
          (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0)
          (FlockComplete.LigLock.rhoS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.k0)
          (FlockComplete.ZkTables.Level0.pairLane ((exeTape τ).extra j.val) (2 * r.val)
            (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0).size)
          (FlockComplete.ZkTables.Level0.pairLane ((exeTape τ).extra j.val) (2 * r.val + 1)
            (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0).size)) col := by
    obtain ⟨hnL, hL, -, -, -, hm⟩ := tabC_z PE x j
    obtain ⟨-, -, hl0, -, -, -⟩ := zkLevel0 (zkTab x j).hz
    unfold lv0 at hnL hL hm
    rw [hl0] at hnL hm
    have hmL : (ligSt x j).m = (tabC PE x j).S.m := by
      unfold ligSt
      rw [tabsOf_stAt]
      rfl
    have hesz : (FlockComplete.embedF (ligTab x w τ j)).size = 2 ^ ((tabC PE x j).S.m - 7) := by
      simp [FlockComplete.embedF, ligTab, FlockComplete.JsonRT.packed_size, hmL]
    have hpow : 2 ^ ((tabC PE x j).S.m - 7) =
        2 ^ (zkTab x j).sch.k0 * 2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
      rw [← pow_add]
      congr 1
      omega
    have hdiv : 2 ^ ((tabC PE x j).S.m - 7) / 2 ^ (zkTab x j).sch.k0 =
        2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
      rw [hpow, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
    have hlen : (((List.range (zkTab x j).sch.k0).map (FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r))).map
        FlockSoundness.Refine.gOf).length = (zkTab x j).sch.k0 := by simp
    have htsz : (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j) (zkTab x j).sch.k0).size =
        2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by
      rw [tabS_foldKs, FlockComplete.Replay.Lg.foldKs_size, hesz]
      simpa using hdiv
    intro col hcol
    have h2 : col < 2 ^ ((zkTab x j).sch.levels.getD 0 default).cols := by rw [← hL]; exact hcol
    have hit : col < (FlockComplete.LigLock.tabS (ligCoin c) (ligP0 x j r) (ligTab x w τ j)
        (zkTab x j).sch.k0).size := by rw [htsz]; exact h2
    rw [FlockComplete.Replay.gv_blinded _ _ _ _ hit]
    refine (tab_y1_L x c w τ j r ⟨col, hcol⟩).trans ?_
    rw [add_assoc]
    refine congrArg₂ (· + ·) ?_ ?_
    · rw [tabS_foldKs, FlockComplete.Replay.Lg.gv_foldKs _ _ col (by
          rw [hesz, List.length_map, List.length_range, hdiv]; exact h2),
        FlockSoundness.foldLoL_eq_sum, hlen]
      refine sum_fin_range hnL _ _ fun l hl => ?_
      refine congrArg₂ (· * ·) ?_ ?_
      · rw [FlockSoundness.Refine.eqList_eq]
        unfold Composed.eqLane
        refine prod_fin_range (by rw [hlen]; rfl) _ _ fun q hq => ?_
        have hq' : q < (zkTab x j).sch.k0 := hq
        rw [List.getD_eq_getElem _ _ (by simpa using hq')]
        simp only [List.getElem_map, List.getElem_range]
        have e : (czExe PE x c j).rest.lane r ⟨q, hq⟩ = FlockComplete.LigLock.chS (ligCoin c) (ligP0 x j r) q := by
          show kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + q) = _
          rw [repSlot_p0 hs j r]
          simp only [kAt, FlockComplete.LigLock.chS, ligCoin, Nat.add_assoc]
        rw [e]
      · have hl' : l < 2 ^ (zkTab x j).sch.k0 := by rw [← hnL]; exact hl
        have hy : l + 2 ^ (zkTab x j).sch.k0 * col < 2 ^ ((tabC PE x j).S.m - 7) := by
          have hc : 2 ^ (zkTab x j).sch.k0 * (col + 1) ≤
              2 ^ (zkTab x j).sch.k0 * 2 ^ ((zkTab x j).sch.levels.getD 0 default).cols :=
            Nat.mul_le_mul_left _ h2
          rw [hpow]
          rw [Nat.mul_succ] at hc
          omega
        show FlockLevel3.embed (Composed.wordOf (zwL x w τ j) (l + (tabC PE x j).z.nL * col)) = _
        rw [FlockComplete.Replay.Lg.gv_embedF _ (by rw [FlockComplete.JsonRT.packed_size, hmL]; exact hy)]
        refine congrArg FlockLevel3.embed ?_
        have hM7 : 7 ≤ (zkTab x j).table.S.m := by
          have e : (zkTab x j).table.S.m = (tabC PE x j).S.m := rfl
          omega
        unfold ligTab
        rw [hnL, hmL]
        exact FlockComplete.wordOf_packed _ _ (zwL_trace w τ j) hy hM7
    · have hρ : rhoAt c j.val r.val = FlockComplete.LigLock.rhoS (ligCoin c) (ligP0 x j r) (zkTab x j).sch.k0 := by
        unfold rhoAt rhoSlot
        rw [repSlot_p0 hs j r]
        unfold FlockComplete.LigLock.rhoS ligCoin
        rw [show ligP0 x j r + ((zkTab x j).sch.k0 + 2) = ligP0 x j r + 2 + (zkTab x j).sch.k0 by omega,
          show ligP0 x j r + 2 + (zkTab x j).sch.k0 + 1 = ligP0 x j r + 3 + (zkTab x j).sch.k0 by omega]
      have e0 := (agrees0_exe PE).ext x τ j r 0 ⟨col, hcol⟩
      have e1 := (agrees0_exe PE).ext x τ j r 1 ⟨col, hcol⟩
      simp only [Fin.val_zero, Fin.val_one, Nat.add_zero] at e0 e1
      unfold FlockComplete.ZkTables.Level0.pairLane
      rw [hρ, FlockSoundness.Refine.getElem!_range_map _ hit, FlockSoundness.Refine.getElem!_range_map _ hit]
      exact congrArg₂ (· + ·) (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e0))
        (congrArg₂ (· * ·) rfl (congrArg FlockLevel3.embed e1))
  -- the restart, column by column
  obtain ⟨-, hL, -, -, -, -⟩ := tabC_z PE x j
  obtain ⟨-, -, -, -, -, hk7⟩ := zkLevel0 (zkTab x j).hz
  unfold lv0 at hL
  have hLm : (tabC PE x j).z.L = 2 ^ ((stAt x j).m - 7 - (zkTab x j).sch.k0) := by
    rw [hL]
    congr 1
    omega
  have hLsz : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.L =
      (FlockComplete.LigLock.wS (ligCoin c) (ligP0 x j r) (ligBasis x c w τ j r xs) (zkTab x j).sch.n
        (zkTab x j).sch.k0).size := by
    show (tabC PE x j).z.L = _
    rw [hLm, hWsz]
  symm
  refine sum_fin_range hLsz _ _ fun col hcol => ?_
  have hc' : col < 2 ^ ((stAt x j).m - 7 - (zkTab x j).sch.k0) := by
    have e : col < (tabC PE x j).z.L := hcol
    rw [hLm] at e
    exact e
  exact (mul_comm _ _).trans (congrArg₂ (· * ·) (hW col hc') (hY col hcol))

end Masked
end Claims.ZKMal
