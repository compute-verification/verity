import Proofs.Flock.SessionLoop.FinRepRows
import Proofs.Flock.SessionLoop.FinHmkLive
import Proofs.Flock.Claims.ZK.FinishSwFw
import Proofs.Flock.Claims.ZK.FinTau

/-!
# (c.3)'s `FinRepTau`: `τ` from the views is the session's

`Loop.FinRepTau` asks that where the session's rep (`repFwC` on the loop's table and request) and the run from the views
return proofs equal up to `eraseZ`, their `τ` agree. `eraseZ` forgets `τ`, so each side's is computed:

* the views' run sends its source's `τ` (`FinRep.repCut_map_tau`), the switched view's. Where table `k / 2`'s rows hold
  at its base (`Loop.FinRows`), that is the view at the workers' values (`Coined.swExe_tau`), the worker's view's `τ`
  (`Masked.swViewE`; `Loop.finSrc_tau`);
* the session's rep's `τ` is `InZ.tau` over its constraints at the coins from the round its inner proof begins
  (`FinTau.innerP_cut_tau`). Those constraints are the worker's view's first seven rows (`Composed.tauFw_at_coef`,
  `Masked.swRows_coef`) wherever the masked values below `zp 64` are the switched source's (`Loop.swSrcE_hmk_live`), the
  pads are the tape's (`Masked.pads_frame_eq`) and the inner proof begins at `repSlot … (innerOff 0)`; `InZ.tau` over
  them is the view's `τ` (`Masked.swTau_of_rows`; `Masked.fwTau_at`).

**`Loop.finRepTau`**: in the masked scope, with a witness, at a statement the loop serves whose level 0 and pads the
source draws from the tape (`Loop.Level0Src`, `Loop.RepSrc`), `FinRepTau` holds wherever every table's rows hold at its
base (`Loop.FinRepRows`).
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockSoundness.ZK.Table

theorem toBase_tau {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)
    (v : T.onVals.View) : (T.toBase v).tau = v.tau := rfl

/-- A real view's `τ` is the view's: `viewWith` changes only the hidden leaves. -/
theorem Hm96.viewR_tau {F K W Salt B D Hi VL C : Type} [Field F] [Field K] [AddCommGroup B] [CharP F 2]
    {T : Table F K W Salt (B × Salt) D Hi VL} (M : T.Hm96 C) (p : W) (ω : T.Rand) (ys : T.Hid → Salt) :
    (M.viewR p ω ys).tau = (T.view p ω).tau := rfl

end FlockSoundness.ZK.Table

/-- **Where its rows hold at its base, a switched view's `τ` is the view's at the workers' values** (`Hm96.swViewR_eq`):
the real view keeps the view's `τ` and changes the leaves alone. -/
theorem FlockSoundness.Discharge.ZkView.Coined.swExe_tau {K W Hi VL : Type} [Field K]
    {z : FlockSoundness.Discharge.ZkView.Sizes} (T : FlockSoundness.Discharge.ZkView.Coined K W Hi VL z) (w : W)
    (v : T.tab.Vals) (ω : T.tab.Rand) (ys : T.tab.Hid → FlockSoundness.Discharge.ZkView.Salt)
    (hr : T.tab.onVals.RowsAt (w, v) ω.1 ω.2.1.2.1) :
    (T.swExe w v ω ys).tau = (T.tab.toBase (T.tab.onVals.view (w, v) ω)).tau := by
  have e1 : T.swExe w v ω ys = T.tab.toBase (T.hm96.onVals.viewR (w, v) ω ys) :=
    congrArg T.tab.toBase
      ((FlockSoundness.ZK.Table.Hm96.swViewR_eq T.hm96.onVals (w, v) (w, T.tab.vals) ω ys).trans (ite_eq_left hr))
  rw [e1]
  exact (T.tab.toBase_tau _).trans
    ((FlockSoundness.ZK.Table.Hm96.viewR_tau T.hm96.onVals (w, v) ω ys).trans (T.tab.toBase_tau _).symm)

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)

namespace Masked

/-- **A deployed rep's `τ` from the firewall's run**: where `repFwC` on table `j`'s rep `r`, answered by the opened
coins from the rep's base at the packed table of the witness's trace (`hg`), at the tape's pads (`hP`), returns, and
below `zp 64` the passed values plus the pads are the switched source's masked values (`hmk`), its `τ` is the worker's
view's at the clear values `v`, wherever table `j`'s rows hold at its base there (`hrows`). The inner proof's `τ` is
`InZ.tau` at the coins from the round it begins (`FinTau.innerP_cut_tau`), `repSlot … (innerOff 0)`, over the run's
constraints, the view's first seven rows (`tauFw_at_coef`, `swRows_coef`), so the view's `τ` (`swTau_of_rows`). -/
theorem fwTau_at (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (xs : Array _root_.Flock.Claim) (hxsz : xs.size = spec.points * (stAt x j).regions.size)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (pads : FlockComplete.PadsZk) (hP : pads = innerPadsE x τ j r) (ic : FlockComplete.InnerCommit)
    (vals : Array _root_.Flock.F128) (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript}
    {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch tPad r.val dom g cm0 xs pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l'))
    (hmk : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      ((swSrcE x c w v τ j.val).rep r.val).masked k = vals[k]! + pads.h[k]!) :
    p.inner.tau = FlockLevel3.GF128.val ((swViewE x c w v τ j).tau r) := by
  subst hg
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, h2, -⟩, hlv, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hk6' : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  -- the setup is the statement's
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
    omega
  have hk6 : 6 ≤ st.kLog := by
    rw [hSk]
    exact hk6'
  have hlay : Composed.layOf (zkTab x j).table (stAt x j) =
      _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 := by
    subst hst
    rfl
  have hlen : (Composed.layOf (zkTab x j).table (stAt x j)).len = (tabC PE x j).z.nE + 4 := by
    have h := innerLayE_len x j
    rw [innerLayE_layE] at h
    exact h
  -- the coins from the rep's base are `czExe`'s
  have ez : ligCoin c (l.size + 2) 0 = Composed.zcF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    rw [hl]
    rfl
  have er : _root_.Flock.pinned ++ (Array.range (st.m - 13)).map (ligCoin c (l.size + 1)) =
      Composed.rF (zkTab x j).table (stAt x j) (czExe PE x c j) r := by
    subst hst
    rw [rF_czExe c hs j r, hl]
    rfl
  have eρ : Composed.rhoF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      FlockComplete.Replay.pushes (fun k => ligCoin c (l.size + (3 + k)) 0) (st.m - 6) := by
    subst hst
    rw [hl]
    refine arr_eq_pushes (by unfold Composed.rhoF; exact Array.size_ofFn) fun k hk => ?_
    unfold Composed.rhoF
    rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
    rfl
  have eα : ligCoin c (l.size + (3 + (st.m - 6))) 0 = (czExe PE x c j).rest.lcAlpha r := by
    subst hst
    rw [hl]
    rfl
  have eβ : ligCoin c (l.size + (4 + (st.m - 6))) 0 = (czExe PE x c j).rest.lcBeta r := by
    subst hst
    rw [hl]
    rfl
  have eτ : Composed.tsF (zkTab x j).table (stAt x j) (czExe PE x c j) r =
      FlockComplete.Replay.pushes (fun k => ligCoin c (l.size + (5 + (st.m - 6) + k)) 0) (st.kLog - 6) := by
    subst hst
    rw [hl]
    refine arr_eq_pushes (by unfold Composed.tsF; exact Array.size_ofFn) fun k hk => ?_
    unfold Composed.tsF
    rw [getElem!_pos _ k (by rw [Array.size_ofFn]; exact hk), Array.getElem_ofFn]
    rfl
  have er' : ligCoin c (l.size + (5 + (st.m - 6) + (st.kLog - 6))) 0 = (czExe PE x c j).rest.rPrime r := by
    subst hst
    rw [hl]
    rfl
  -- the masked values below `zp 64` are the view's
  have hmk' : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      (Composed.msgAt ((swViewE x c w v τ j).pre.2.1 r) k).val = vals[k]! + pads.h[k]! := fun k hk => by
    rw [swViewMasked x c w v τ j r hrows k]
    exact hmk k hk
  -- the ring switch's `ŝ`s for claims `0` and `1` are the view's
  have hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => vals[i]! + pads.h[i]!).run t₀).go l = (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (ligCoin c) n)
          (fun k => _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
          (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val) ∧
        ss[1]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + 1) := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
    have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
    obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
      (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
      (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) t₀ (l := l) rfl hl0.le
      (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
    obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
      (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
      (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) zc t₁ (l := l₁) rfl
      (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
    have hrank : rankSlot x j.val r.val ≤ n := by
      unfold rankSlot
      rw [hstj]
      omega
    -- the switched source's switched vectors are the firewall's
    have hsv : ∀ k < 2,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]! := by
      intro k hk2
      obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
      obtain ⟨hlc, -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
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
        (FlockComplete.abcClaims_size _ _ _ _) hab'.1 hab'.2 k
      have hA2 : (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r))).size = 2 :=
        FlockComplete.abcClaims_size _ _ _ _
      have hx6E : ((_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ xs)[k]!).x.size =
          (stAt x j).m - 6 := by
        rw [FlockComplete.Replay.get_left (lt_of_lt_of_eq hk2 hA2.symm)]
        interval_cases k
        · show ((FlockComplete.Replay.pushes (lcE c j r).τ ((stAt x j).circuitFold.kLog - 6)).reverse ++
            (FlockComplete.Replay.pushes (zcE c j r).ρ ((stAt x j).m - 6)).extract ((stAt x j).c.kLog - 6)
              ((stAt x j).m - 6)).size = (stAt x j).m - 6
          rw [Array.size_append, Array.size_reverse, FlockComplete.Replay.pushes_size, Array.size_extract,
            FlockComplete.Replay.pushes_size, Nat.min_self]
          omega
        · exact FlockComplete.Replay.Zc.r_size (stAt x j).m (by omega) (zcE c j r)
      have hx6 : ((_root_.Flock.abcClaims (_root_.Flock.Setup.ofCircuit (stAt x j)).m
          (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog zc lc ++ xs)[k]!).x.size = (stAt x j).m - 6 := by
        rw [hxk]
        exact hx6E
      have hpk : ligTab x w τ j = FlockComplete.packed tr (stAt x j).m := by
        show FlockComplete.packed tr (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
        rw [tabsOf_stAt]
        rfl
      rw [hpk, FlockComplete.switchVec_packed (by omega) tr _ hx6, FlockComplete.Replay.switchVec_x hxk, swSrcE_sv]
      have hl₃ := (FlockComplete.ringSwitchCut_sws (ligCoin c) n _ _
        (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) h₃).2
      rw [Array.size_append, FlockComplete.abcClaims_size] at hl₃
      exact src_sv_abc c hw τ j r hx hnr hs xs k hk2 (by rw [hfk]; omega)
    have hss := FlockComplete.ringSwitchCut_ss (ligCoin c) n _ _ h₃
    have hsz : 2 ≤ (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      omega
    have e : ∀ k < 2, ss[k]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + k) := by
      intro k hk
      rw [hss]
      refine (FlockComplete.Replay.pushes_get _ _ k (by omega)).trans ?_
      refine (hsv k (by omega)).symm.trans ?_
      exact swSrcE_sv_rank x hx c w v τ hnr hs j r hrank k hk
    have e0 := e 0 (by norm_num)
    rw [Nat.add_zero] at e0
    exact ⟨e0, e 1 (by norm_num)⟩
  -- the run: the PIOP, Ligerito, the replay and level 0's constraints, then the inner proof
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
  -- the inner proof's `τ` is `InZ.tau` at the coins from the round it begins
  have htau : p.inner.tau = FlockComplete.Replay.InZ.tau
      (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) pads
      (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
        ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
      (FlockComplete.SimInner.cutCoins (ligCoin c) l₄.size) := by
    obtain ⟨ip, t₈, l₈, h₈, hp⟩ := _root_.Flock.Firewall.Finish.run_bind_ok hc
    simp only [StateT.run_pure, FlockComplete.JsonRT.go_pure, Prod.mk.injEq, Except.ok.injEq] at hp
    obtain ⟨⟨rfl, -⟩, -⟩ := hp
    exact (FinTau.innerP_cut_tau h₈).2
  -- the inner proof's slot: the PIOP's sends, then Ligerito's
  have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
  obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
    (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) t₀ (l := l) rfl hl0.le
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => vals[i]! + pads.h[i]!) (fun i => vals[i]! + pads.h[i]!) zc t₁ (l := l₁) rfl
    (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  obtain ⟨hl3, -, -⟩ := (FlockComplete.CutEq.ringSwitchCut_eqAt (ligCoin c) n
    (fun k => _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
      (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
    (fun k => _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
      (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
    (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)
    (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) t₂ (l := l₂) rfl (by omega)
    (fun _ _ _ => rfl)).2 _ _ h₃
  rw [Array.size_append, FlockComplete.abcClaims_size] at hl3
  have hS4 := FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _ _
    _ _ t₃ l₃ _ _ h₄
  have hl4 := (FlockComplete.ligeritoPZk_exact (respE c) _root_.Flock.Sha512.hash _root_.Flock.MerkleScheme.hm96Sha512
    (zkTab x j).sch T cm0 (ligTab x w τ j) (FlockComplete.basisVals sws (st.m - 7))
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    pads.h r.val tPad extra lanePad salt (fun _ _ => 0) (by omega) hk0 h2
    (fun i hi => (hlv i hi).q_ne_zero) hS4).1
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hL₀ : l₄.size = repSlot x (2 * j.val + r.val) (innerOff 0) := by
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk innerOff
    omega
  have hslot : ∀ f, repSlot x (2 * j.val + r.val) f = l₃.size + f (zkTab x j).sch := by
    intro f
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk
    omega
  -- level 0's record holds the rep's coins at its slots (`ligeritoPZk_lanes`), `czExe`'s
  have hrec := (FlockComplete.L0.ligeritoPZk_lanes (respE c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by omega) hk0 hS4).record
  have hLC : LaneCoins (zkTab x j).table (stAt x j) (czExe PE x c j) r rec := by
    refine ⟨fun i => ?_, fun i hi => ?_, ?_⟩
    · rw [hrec]
      show fOf (FlockComplete.coinOf (respE c (l₃.size + 1)) 0) =
        fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 1)) 0)
      rw [coinOf_respE, hslot]
    · rw [hrec]
      show (FlockComplete.ZkTables.Level0.laneCh (respE c) (zkTab x j).sch.k0 l₃.size).toArray[i]! =
        kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + i)
      unfold FlockComplete.ZkTables.Level0.laneCh
      rw [getElem!_pos _ i (by simp only [List.size_toArray, List.length_map, List.length_range]; exact hi)]
      simp only [List.getElem_toArray, List.getElem_map, List.getElem_range, FlockComplete.coinKOf, coinOf_respE, kAt,
        hslot, Nat.add_assoc]
    · rw [hrec]
      show FlockComplete.ZkTables.Level0.laneRho (respE c) (zkTab x j).sch.k0 l₃.size = rhoAt c j.val r.val
      unfold FlockComplete.ZkTables.Level0.laneRho rhoAt rhoSlot
      simp only [FlockComplete.coinKOf, coinOf_respE, hslot]
      rw [show l₃.size + 2 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) by omega,
        show l₃.size + 3 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) + 1 by omega]
  -- the run's constraints are the view's first seven rows
  obtain ⟨hlsz, hlasz, hrow⟩ := Composed.tauFw_at_coef (zkTab x j).table (stAt x j) (czExe PE x c j) r
    ((swViewE x c w v τ j).pre.2.1 r) (swViewE x c w v τ j).pre.1
    ((swViewE x c w v τ j).pre.2.2.2.2.2 r).1 ((swViewE x c w v τ j).pre.2.2.2.2.1 r)
    (ligCoin c) n st (zkTab x j).sch tPad r.val dom (ligTab x w τ j) cm0 xs pads ic
    vals extra lanePad salt ht₀ hm14 hk6 hSm hSk hkL hlay hmk' ez er eρ eα eβ eτ er'
    (fun combv hcomb => by
      subst hst
      have hkm := (zkTab x j).table.S.hk
      have hS : (zkTab x j).table.S = FlockSoundness.Refine.stmtOf (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)) := rfl
      refine folded_unique (zkTab x j).table.S _ (Composed.modelFold (zkTab x j).table.S)
        (hS ▸ FlockSoundness.Refine.stmtOf_fold (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)))
        (Composed.modelFold_realizes _) _ _ _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ hcomb rfl
      simp only [Composed.rhoF, Array.size_extract, Array.size_ofFn]
      show min ((zkTab x j).table.S.kLog - 6) ((zkTab x j).table.S.m - 6) - 0 = (zkTab x j).table.S.kLog - 6
      omega)
    hsw hlen h₁ h₂ h₃ h₄ hrp hlgc hLC
  have hLy : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = innerLayE x j := by
    subst hst
    exact (innerLayE_eq x j).symm
  have hrp' := hrp
  rw [hLy] at hrp'
  have hcons := swRows_coef x c w v τ j r hm14 hrp' ss ⟨hlsz, hlasz⟩ hrow
  rw [htau, hP, hL₀, hLy]
  exact swTau_of_rows hs c w v τ j r _ hcons

end Masked
end Claims.ZKMal

namespace Claims.ZKServer

open Claims Claims.ZKMal
open FlockSoundness.Discharge
open _root_.Flock (V F128 Stmt SessionSpec)

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000 -- `ofLoop`'s session types, as in `FrameZK`

namespace Loop

/-- **Where its rows hold, the views' `τ` is the worker's view's**: table `i`'s rep `r` source from the switched views
reads the switched view's `τ`, the view's at the workers' values (`Coined.swExe_tau`), whose `τ` reads the base's mask,
lanes, pads and `μ` alone, as `Masked.swViewE`'s. -/
theorem finSrc_tau (vs : finS.SwVals) (x : Msg) (w : Flock.Wit) (ν : finS.Nonce) (t : finS.Tape)
    (ops : List (Msg × Msg)) {i r : ℕ} (hi : i < Masked.jOf x) (hr : r < 2) (hrows : FinRows vs x w ν t ops ⟨i, hi⟩) :
    ((finSrc vs x w ν t ops i).rep r).tau =
      FlockLevel3.GF128.val ((Masked.swViewE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t ⟨i, hi⟩).tau ⟨r, hr⟩) := by
  unfold finSrc Masked.tabSrcView
  rw [dite_eq_left hi]
  dsimp only
  rw [dite_eq_left hr]
  try dsimp only
  simp only [RefSession.swViewsAt]
  exact congrArg (fun g : Fin 2 → _ => FlockLevel3.GF128.val (g ⟨r, hr⟩))
    (ZkView.Coined.swExe_tau ((finS.C x ⟨i, hi⟩).T (finS.cz x (finCoins x ops) ⟨i, hi⟩))
      (finS.wt x w ⟨i, hi⟩) (vs x w ν (finCoins x ops) ⟨i, hi⟩) (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).1
      (finS.randAt x (finCoins x ops) t ⟨i, hi⟩).2 hrows)

variable (vs : finS.SwVals) (L₀ : LeanRelease Flock.Wit Tp)
  (src : Msg → Flock.Wit → Tp → ℕ → V (Array F128 × ByteArray × Array ByteArray))
  (real : Msg → Flock.Wit → Tp → ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
  (wk : (LeanRelease.ofLoopLink L₀ src real).toSrv.Worker) (x : Msg) (w : Flock.Wit)

/-- **`FinRepTau` where every table's rows hold at its base**: in the masked scope, with a witness, at a statement the
loop serves (no coin fault at the opened coins, `frameUnrefused_ofLoop`) whose level 0 and pads the source draws from
the tape (`Level0Src`, `RepSrc`). The run from the views sends the views' `τ` (`FinRep.repCut_map_tau`), the worker's
view's (`finSrc_tau`); the session's rep runs at the session's table, schedule and packed table (`finRepTab`,
`schedsPZk_live`, `Level0Src.realOf`), at the tape's pads (`pads_frame_eq`), with the switched source's masked values
below `zp 64` (`swSrcE_hmk_live`), so its `τ` is the worker's view's too (`Masked.fwTau_at`). -/
theorem finRepTau (hsc : Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x)
    (hR : Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w) (hServe : Loop.Serves x)
    (hS0 : Loop.Level0Src src x w) (hS : Loop.RepSrc L₀ src x w) (hRows : FinRepRows vs L₀ src real wk x w) :
    FinRepTau vs L₀ src real wk x w := by
  have bok : ∀ {α β : Type} {y : V α} {f : α → V β} {b : β}, y >>= f = .ok b → ∃ a, y = .ok a ∧ f a = .ok b := by
    intro α β y f b hb
    cases y with
    | error e => cases hb
    | ok a => exact ⟨a, rfl, hb⟩
  have hx : Masked.Live x := Masked.live_of hsc hR
  intro ν t κ m₀ ρ₀ ops hctx s0 spec hsx schs hschs cms hcms k hk extras hxe ic l hic hl tb hi q f hRep ic' p tr hic'
    hrun p' l' hv hz
  have hmk := swSrcE_hmk_live vs wk hsc hR hServe hctx hsx hk hRep hrun hv hz
  -- the views' `τ`
  have e1 : p'.inner.tau = ((finSrc vs x w ν t ops (k / 2)).rep (k % 2)).tau :=
    Masked.FinRep.field_of_map (fun z => z.inner.tau) (Masked.FinRep.repCut_map_tau _ _ _ _ _ _ _ _ _ _ _ _) hv
  obtain ⟨-, -, hT, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hJ : Masked.jOf x = (Masked.stsOf x).size := by
    unfold Masked.jOf
    rw [if_pos hx]
  have hj : k / 2 < Masked.jOf x := by omega
  have hr : k % 2 < 2 := Nat.mod_lt _ (by decide)
  have hk2 : Loop.frepOf (k / 2) (k % 2) = k := by
    unfold Loop.frepOf
    omega
  have e2 := finSrc_tau vs x w ν t ops hj hr (hRows ν t κ m₀ ρ₀ ops hctx ⟨k / 2, hj⟩)
  -- the request's pads are the tape's
  have hq := hRep.2.1
  unfold repInOf at hq
  obtain ⟨⟨lay, ms⟩, hst0, hq⟩ := bok hq
  have hqd : q.draws = _root_.Flock.Firewall.Frame.Draws.ofBytes lay (L₀.draws (ν, t) k) := by
    cases hq
    rfl
  have hLk : (LeanRelease.ofLoopLink L₀ src real).lay x k = lay := by
    have e : Loop.startOf x k = _root_.Flock.Firewall.Drive.start (Masked.stsOf x)[k / 2]! := by
      unfold Loop.startOf
      rw [hsx]
      try dsimp only
    show ((Loop.startOf x k).toOption.map (·.1)).getD default = lay
    rw [e]
    rw [hst0]
    all_goals rfl
  have hlay0 := LeanRelease.lay_ofLoopLink (L₀ := L₀) src real ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hst0
  have hd := LeanRelease.draws_ofLoopLink (L₀ := L₀) src real
  have hzs := hS.size (LeanRelease.ofLoopLink L₀ src real) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay0
  have hpads := hS.pads (LeanRelease.ofLoopLink L₀ src real) hd ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hlay0
    (Masked.UE.tapeZk t)
  have hq' : q.draws = (LeanRelease.ofLoopLink L₀ src real).drawsAt x (ν, t) (Loop.frepOf (k / 2) (k % 2)) := by
    rw [hk2, hqd]
    show _ = _root_.Flock.Firewall.Frame.Draws.ofBytes ((LeanRelease.ofLoopLink L₀ src real).lay x k)
      (L₀.draws (ν, t) k)
    rw [hLk]
  have hP := Masked.pads_frame_eq (LeanRelease.ofLoopLink L₀ src real) (ν, t) (Masked.UE.tapeZk t) Loop.frepOf x t
    ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ q hq' hlay0 hzs hpads
  have hP' : (_root_.Flock.Firewall.Finish.padsOf q.draws).toC = Masked.innerPadsE x t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ :=
    hP.trans (congrArg (FlockComplete.padsOf (Masked.UE.tapeZk t) (k / 2) (k % 2))
      (Masked.innerLayE_layE x ⟨k / 2, hj⟩)).symm
  -- the session's table, schedule and packed table
  have hnr : ¬ Masked.Refused x (finCoins x ops) (refK x) :=
    LeanRelease.frameUnrefused_ofLoop Masked.dataE (LeanRelease.withFiles (LeanRelease.withLink L₀ src) src) real
      CoinOpen.shaH wk x w hServe hx ν t κ m₀ ρ₀ ops hctx.1 hctx.2.1 hctx.2.2.1 hctx.2.2.2.1
  obtain ⟨schs', hschs', hschZ⟩ := Masked.schedsPZk_live hx
  obtain ⟨htb, hsb⟩ := finRepTab (L₀ := L₀) (src := src) (x := x) (w := w) hsc hsx hschs' hk hRep
  have hst : tb.st = _root_.Flock.Setup.ofCircuit (Masked.stAt x ⟨k / 2, hj⟩) :=
    htb.symm.trans (Masked.tabsOf_stAt x ⟨k / 2, hj⟩)
  have hsch : tb.sch = (Masked.zkTab x ⟨k / 2, hj⟩).sch := (Prod.mk.inj (hsb.symm.trans (hschZ ⟨k / 2, hj⟩))).1
  have hR1 : Loop.realOf src x w (ν, t) (2 * (⟨k / 2, hj⟩ : Fin (Masked.jOf x)).val + (⟨k % 2, hr⟩ : Fin 2).val) =
      .ok (tb, hi) := by
    show Loop.realOf src x w (ν, t) (2 * (k / 2) + k % 2) = _
    rw [Nat.div_add_mod]
    exact hRep.1
  obtain ⟨-, -, -, hg⟩ := hS0.realOf ν t ⟨k / 2, hj⟩ ⟨k % 2, hr⟩ hR1
  -- the masked values below `zp 64` are the switched source's
  have hlay : _root_.Flock.Zk.PadLayout.new tb.st.m tb.st.kLog tb.sch.k0 =
      Composed.layOf (Masked.zkTab x ⟨k / 2, hj⟩).table (Masked.stAt x ⟨k / 2, hj⟩) := by
    rw [hst, hsch]
    rfl
  have hmk' : ∀ i < (Composed.layOf (Masked.zkTab x ⟨k / 2, hj⟩).table (Masked.stAt x ⟨k / 2, hj⟩)).zp 64,
      ((Masked.swSrcE x (finCoins x ops) w (vs x w ν (finCoins x ops)) t (k / 2)).rep (k % 2)).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := by
    rw [← hlay]
    exact hmk
  -- as many extra claims as the statement
  have hxsz : (_root_.Flock.Firewall.Finish.claimsOf q.xs).size =
      spec.points * (Masked.stAt x ⟨k / 2, hj⟩).regions.size := by
    have hjs : k / 2 < (Masked.stsOf x).size := by omega
    have hq := hRep.2.1
    unfold repInOf at hq
    obtain ⟨⟨lay, ms⟩, -, hq⟩ := bok hq
    have hqx : q.xs = _root_.Flock.Firewall.Drive.extras (Masked.stsOf x)[k / 2]!
        (_root_.Flock.Firewall.Drive.linkPoints (Masked.stsOf x)[k / 2]!
          (wordsAt x (ops.map fun p => CoinLayer.rd p.2) 0)) := by
      cases hq
      rfl
    rw [linkPoints_wordsAt x w hsc hR hctx.1 hsx hk m₀ ρ₀,
      show (Masked.stsOf x)[k / 2]! = Masked.stAt x ⟨k / 2, hj⟩ from getElem!_pos (Masked.stsOf x) (k / 2) hjs] at hqx
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map]
    rw [hqx]
    unfold _root_.Flock.Firewall.Drive.extras
    rw [flatMap_size_const _ _ _ fun _ => Array.size_map]
    simp only [viewPoints, Array.size_map, Array.size_range]
    exact Nat.mul_comm _ _
  -- the session's rep at table `k / 2`'s rep `k % 2`
  have hl' : l.size = Masked.repBase x (2 * (⟨k / 2, hj⟩ : Fin (Masked.jOf x)).val + (⟨k % 2, hr⟩ : Fin 2).val) := by
    rw [hl]
    show Masked.repBase x k = Masked.repBase x (2 * (k / 2) + k % 2)
    rw [Nat.div_add_mod]
  have hgo := go_of_fst hrun
  rw [hsch] at hgo
  have ht₀ : (_root_.Flock.Transcript.init #[] (spec.domain spec.tables[k / 2]! (k % 2)).toUTF8).rounds = #[] := rfl
  have e3 := Masked.fwTau_at x hx w hR t (finCoins x ops) (refK x) hnr s0 spec hsx ⟨k / 2, hj⟩ ⟨k % 2, hr⟩
    (vs x w ν (finCoins x ops)) (hRows ν t κ m₀ ρ₀ ops hctx ⟨k / 2, hj⟩) (_root_.Flock.Firewall.Finish.claimsOf q.xs)
    hxsz tb.st hst tb.tPad _ tb.g hg tb.cm0.toC (_root_.Flock.Firewall.Finish.padsOf q.draws).toC hP' ic'.toC q.vals
    tb.extra tb.lanePad _ ht₀ hl' hgo hmk'
  exact e1.trans (e2.trans e3.symm)

end Loop

end Claims.ZKServer
