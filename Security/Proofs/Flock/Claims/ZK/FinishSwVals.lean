import Proofs.Flock.Claims.ZK.FinishSwRep

/-!
# The firewall's rep is the switched source's cut, at any clear values

`FinishSwRep.swSrcE_repCut` takes the switched source at the firewall's clear values (`valsFw`), which read the
request's values below `zp 64` and level 0's `l0Clear` above. The views' source is at the workers' clear values, so
here the chain is restated at any clear values `v`, with the two facts `valsFw` supplied as premises: below `zp 64` the
masked values are the request's values plus its pads (`hmkV`), and from `zp 64` to `h_ab` they are `l0Clear` at the
coins plus the pads (`hl0`). The rest of the chain
(`swSrc_sv`, `swViewMasked`, `swRows_coef`, `swSrcE_sv_rank`, `swSrcE_restart`, `swSrcE_innerSrc_rows`) is already
at any clear values.

* `swSrcE_hinner_v`, `swSrcE_hinner_sv_v`: `hinner` at `v`.
* `swSrcE_ligHyp_frame_v`, `swSrcE_ligHyp_E_v`: `hlig` at `v`.
* **`swSrcE_repCut_v`**: `repCut_fw` at the switched source at `v`.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)

namespace Masked

/-- **`swSrcE_hinner` at any clear values `v`**: as `swSrcE_hinner`, with the masked values below `zp 64` given
(`hmkV`) in place of `valsFw`'s and the tape's pads. -/
theorem swSrcE_hinner_v {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (hmkV : ∀ k < (layE x j).zp 64, ((swSrcE x c w v τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array _root_.Flock.Claim)
    (he : extras.size = spec.points * (stAt x j).regions.size) (ic : FlockComplete.InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) {t₀ : _root_.Flock.Transcript}
    (ht₀ : t₀.rounds = #[]) {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    (hsw : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (ligCoin c) n)
          (fun k => _root_.Flock.Firewall.Finish.switchVec g (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]!)
          (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      ss[0]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val) ∧
        ss[1]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + 1)) :
    ∀ zp lp ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc,
      ((_root_.Flock.Firewall.Finish.piopFw (FlockComplete.asksCut (ligCoin c) n) st
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) g
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) (FlockComplete.flat cm0.cap)
          ic.root extras).run t₀).go l = (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut (ligCoin c) n)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 (zkTab x j).sch
          T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h r.val tPad extra lanePad salt fun _ _ => 0).run t₃).go
          l₃ = (.ok ((lg, zl, rec), t₄), l₄) →
      (_root_.Flock.Zk.replay st (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          { cap := cm0.cap, params := _root_.Flock.pcsParams st.merkle (zkTab x j).sch, zerocheck := zp,
            lincheck := lp, ringSwitches := ss, ligerito := lg } ic.root).run
          (_root_.Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) →
      _root_.Flock.Zk.ligeritoConstraints (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) rec lg zl =
        .ok lgc →
      ((FlockComplete.Replay.InZ.lin (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins (ligCoin c) l₄.size)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins (ligCoin c) l₄.size)).a
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h).isZero = true →
      FlockComplete.SimInner.InnerSrc (ligCoin c) n _root_.Flock.Sha512.hash
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
        (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
        ((swSrcE x c w v τ j.val).rep r.val) l₄.size := by
  intro zp lp ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc hP h₄ hrp hlgc hbat
  obtain ⟨zc, t₁, l₁, lc, t₂, l₂, h₁, h₂, h₃⟩ := _root_.Flock.Firewall.Finish.piopFw_split hP
  obtain ⟨h25, -, hsm, hk0, -, ⟨hn7, h2, -⟩, hlv, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hk6' : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  -- the setup is the statement's
  have hm14 : 14 ≤ st.m := by
    subst hst
    show 14 ≤ (stAt x j).m
    omega
  have hk6 : 6 ≤ st.kLog := by
    subst hst
    exact hk6'
  have hSm : st.m = (stAt x j).m := by
    subst hst
    rfl
  have hSk : st.kLog = (stAt x j).c.kLog := by
    subst hst
    rfl
  have hkL : st.fold.kLog = st.kLog := by
    subst hst
    rfl
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
  have hmk : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      (Composed.msgAt ((swViewE x c w v τ j).pre.2.1 r) k).val =
        q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[k]! := fun k hk => by
    rw [swViewMasked x c w v τ j r hrows k]
    exact hmkV k hk
  -- the inner proof's slot: the PIOP's sends, then Ligerito's
  have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
  obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
    (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) t₀ (l := l) rfl hl0.le
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc t₁ (l := l₁) rfl
    (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  obtain ⟨hl3, -, -⟩ := (FlockComplete.CutEq.ringSwitchCut_eqAt (ligCoin c) n
    (fun k => _root_.Flock.Firewall.Finish.switchVec g (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    (fun k => _root_.Flock.Firewall.Finish.switchVec g (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]!)
    (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)
    (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) t₂ (l := l₂) rfl (by omega)
    (fun _ _ _ => rfl)).2 _ _ h₃
  rw [Array.size_append, FlockComplete.abcClaims_size] at hl3
  have hS4 := FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _ _
    _ _ t₃ l₃ _ _ h₄
  have hl4 := (FlockComplete.ligeritoPZk_exact (respE c) _root_.Flock.Sha512.hash _root_.Flock.MerkleScheme.hm96Sha512
    (zkTab x j).sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h r.val tPad extra lanePad salt (fun _ _ => 0) (by omega) hk0 h2
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
  -- the rows
  obtain ⟨hlsz, hlasz, hrow⟩ := Composed.tauFw_at_coef (zkTab x j).table (stAt x j) (czExe PE x c j) r
    ((swViewE x c w v τ j).pre.2.1 r) (swViewE x c w v τ j).pre.1
    ((swViewE x c w v τ j).pre.2.2.2.2.2 r).1 ((swViewE x c w v τ j).pre.2.2.2.2.1 r)
    (ligCoin c) n st (zkTab x j).sch tPad r.val dom g cm0 extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic
    q.vals extra lanePad salt ht₀ hm14 hk6 hSm hSk hkL hlay hmk ez er eρ eα eβ eτ er'
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
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  rw [hLyE] at hbat ⊢
  exact swSrcE_innerSrc_rows Lr ρ tz frep x hx w τ c n hnr s spec hs j r v q hq hlayR hz hpads hrows _
    hcons hL₀ fun _ => hbat

/-- **`swSrcE_hinner_sv` at any clear values `v`**: `swSrcE_hinner_v` with `hsw` from `hpiop`'s switched vectors. -/
theorem swSrcE_hinner_sv_v {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (hmkV : ∀ k < (layE x j).zp 64, ((swSrcE x c w v τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (cm0 : FlockComplete.Commit) (extras : Array _root_.Flock.Claim)
    (he : extras.size = spec.points * (stAt x j).regions.size) (ic : FlockComplete.InnerCommit)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray) {t₀ : _root_.Flock.Transcript}
    (ht₀ : t₀.rounds = #[]) {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val))
    (hpsv : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec g (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]!) :
    ∀ zp lp ss T sws t₃ l₃ lg zl rec t₄ l₄ rp tt lgc,
      ((_root_.Flock.Firewall.Finish.piopFw (FlockComplete.asksCut (ligCoin c) n) st
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) g
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) (FlockComplete.flat cm0.cap)
          ic.root extras).run t₀).go l = (.ok ((zp, lp, ss, T, sws), t₃), l₃) →
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut (ligCoin c) n)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 (zkTab x j).sch
          T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h r.val tPad extra lanePad salt fun _ _ => 0).run t₃).go
          l₃ = (.ok ((lg, zl, rec), t₄), l₄) →
      (_root_.Flock.Zk.replay st (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          { cap := cm0.cap, params := _root_.Flock.pcsParams st.merkle (zkTab x j).sch, zerocheck := zp,
            lincheck := lp, ringSwitches := ss, ligerito := lg } ic.root).run
          (_root_.Flock.Transcript.init t₄.rounds dom) = .ok (rp, tt) →
      _root_.Flock.Zk.ligeritoConstraints (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) rec lg zl =
        .ok lgc →
      ((FlockComplete.Replay.InZ.lin (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins (ligCoin c) l₄.size)).c +
        _root_.Flock.dot (FlockComplete.Replay.InZ.lin (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
          (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
            ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
          (FlockComplete.SimInner.cutCoins (ligCoin c) l₄.size)).a
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h).isZero = true →
      FlockComplete.SimInner.InnerSrc (ligCoin c) n _root_.Flock.Sha512.hash
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) (_root_.Flock.Firewall.Finish.padsOf q.draws).toC
        (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[0]! ss[0]!)),
          ("ring switch c", rp.cValue.addConst (_root_.Flock.Zk.claimCheck rp.claims[1]! ss[1]!))] ++ lgc)
        ((swSrcE x c w v τ j.val).rep r.val) l₄.size := by
  refine swSrcE_hinner_v Lr ρ tz frep x hx w τ c n hnr s spec hs j r v q hq hlayR hz hpads hrows hmkV st hst tPad
    dom g cm0 extras he ic extra lanePad salt ht₀ hl ?_
  intro zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
  obtain ⟨h25, -, -, -, -, -, -, -, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
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
  have hl0 := (FlockComplete.RankAt.zerocheckCut_round1 (ligCoin c) n hm14 _ _ _ _ _ h₁).1
  obtain ⟨hl1, hl1n, -⟩ := (FlockComplete.CutEq.zerocheckCut_eqAt (ligCoin c) n hm14 st.digest
    (FlockComplete.flat cm0.cap) ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) t₀ (l := l) rfl hl0.le
    (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ => ⟨rfl, rfl, rfl⟩)).2 _ _ h₁
  obtain ⟨hl2, hl2n, -⟩ := (FlockComplete.CutEq.lincheckCut_eqAt (ligCoin c) n st.fold st.m
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!)
    (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc t₁ (l := l₁) rfl
    (by omega) (fun _ _ _ => ⟨rfl, rfl⟩) (fun _ _ _ => rfl)).2 _ _ h₂
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val (lt_of_jOf j)
  have hrank : rankSlot x j.val r.val ≤ n := by
    unfold rankSlot
    rw [hstj]
    omega
  have hss := FlockComplete.ringSwitchCut_ss (ligCoin c) n _ _ h₃
  have hsz : 2 ≤ (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size := by
    rw [Array.size_append, FlockComplete.abcClaims_size]
    omega
  have e : ∀ k < 2, ss[k]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + k) := by
    intro k hk
    rw [hss]
    refine (FlockComplete.Replay.pushes_get _ _ k (by omega)).trans ?_
    refine (hpsv zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ k (by omega)).symm.trans ?_
    exact swSrcE_sv_rank x hx c w v τ hnr hs j r hrank k hk
  have e0 := e 0 (by norm_num)
  rw [Nat.add_zero] at e0
  exact ⟨e0, e 1 (by norm_num)⟩

/-- **`swSrcE_ligHyp_frame` at any clear values `v`**: where rep `r`'s masked values from `zp 64` to `h_ab` are
`l0Clear` at the coins from round `p₀`, the packed table `t` and the basis `b`, plus the request's pads (`hl0`), rep
`r`'s source holds level 0's opening, given the schedule's facts and the source's `y₁'`, `ȳ` and salts. -/
theorem swSrcE_ligHyp_frame_v (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (coin : ℕ → ℕ → _root_.Flock.F128) (n p0 : ℕ) (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule)
    (t b : Array _root_.Flock.F128) (rp tPad : ℕ) (extra lanePad : ℕ → ℕ → _root_.Flock.F128)
    (salt : ℕ → ℕ → ByteArray)
    (hl0 : ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w v τ j.val).rep r.val).masked k =
      FlockComplete.LigLock.l0Clear coin p0 sch t b (layE x j) extra rp k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    (hk0 : 1 ≤ sch.k0) (hn0 : sch.n ≠ 0) (hlv : 2 ≤ sch.levels.size) (hlk : (zkTab x j).sch.k0 = sch.k0)
    (hy1 : p0 + 3 + sch.k0 < n → _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ((swSrcE x c w (v) τ j.val).rep r.val).y1 =
        (Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map fun i =>
          (FlockComplete.LigLock.tabS coin p0 t sch.k0)[i]! +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).1.smul
              ((Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map (extra (2 * rp)))[i]! +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).2.smul
              ((Array.range (FlockComplete.LigLock.tabS coin p0 t sch.k0).size).map (extra (2 * rp + 1)))[i]!)
    (hybar : p0 + 3 + sch.k0 < n →
      _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ((swSrcE x c w (v) τ j.val).rep r.val).ybar = (Array.range tPad).map fun i =>
        (List.range (2 ^ (sch.levels.getD 0 default).lanes)).foldl
          (fun acc l => acc + ((_root_.Flock.eqTableK
            ((Array.range sch.k0).map (FlockComplete.LigLock.chS coin p0)))[l]!).smul (lanePad l i))
          ((FlockComplete.LigLock.rhoS coin p0 sch.k0).1.smul
              (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rp) i) +
            (FlockComplete.LigLock.rhoS coin p0 sch.k0).2.smul
              (lanePad (2 ^ (sch.levels.getD 0 default).lanes + 2 * rp + 1) i)))
    (hsalt : p0 + 3 + sch.k0 < n →
      _root_.Flock.Zk.rhoIndependent (FlockComplete.LigLock.rhoS coin p0 sch.k0) = true →
      ∀ i, 1 ≤ i → i < sch.levels.size → ∀ p < 2 ^ (sch.levels.getD i default).d,
        FlockComplete.fitBytes ms.saltBytes (((swSrcE x c w (v) τ j.val).rep r.val).hiSalt i p) =
          FlockComplete.fitBytes ms.saltBytes (salt i p)) :
    FlockComplete.LigLock.LigHyp coin n p0 ms sch t b (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).h rp
      tPad extra lanePad salt ((swSrcE x c w (v) τ j.val).rep r.val) :=
  FlockComplete.LigLock.ligHyp_of_l0 coin n p0 ms sch t b (layE x j)
    (_root_.Flock.Firewall.Finish.padsOf q.draws).h rp tPad extra lanePad salt
    ((swSrcE x c w v τ j.val).rep r.val) hk0 hn0 hlv rfl hlk hl0 hy1 hybar hsalt

/-- **`swSrcE_ligHyp_E` at any clear values `v`**: as `swSrcE_ligHyp_E`, with level 0's masked values from `zp 64` to
`h_ab` given (`hl0`) in place of `valsFw`'s level 0 part and the tape's
pads. -/
theorem swSrcE_ligHyp_E_v (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hn : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hl0 : ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w v τ j.val).rep r.val).masked k =
      FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
        (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val k +
        (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!) :
    FlockComplete.LigLock.LigHyp (ligCoin c) n (ligP0 x j r) (ligSt x j).merkle (zkTab x j).sch (ligTab x w τ j)
      (ligBasis x c w τ j r extras) (layE x j) (_root_.Flock.Firewall.Finish.padsOf q.draws).h r.val (zkTab x j).tPad
      ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val)
      ((swSrcE x c w v τ j.val).rep r.val) := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, -⟩ := ZkExec.zk_schedule_seams _ _ _ (zkTab x j).hz
  obtain ⟨hn7, h2, -⟩ := hdim
  obtain ⟨ey1, eybar, esalt⟩ := swSrcE_restart x c w v τ j.val r.val
  have hy1 := ligY1E x hx w hw τ c n hn s spec hs j r extras he
  have hybar := ligYbarE x hx w hw τ c n hn s spec hs j r extras he
  have hsalt := ligSaltE hiSaltE x hx w hw τ c n hn s spec hs j r extras he
  dsimp only at hy1 hybar hsalt
  refine swSrcE_ligHyp_frame_v x c w v τ j r q (ligCoin c) n (ligP0 x j r) (ligSt x j).merkle
    (zkTab x j).sch (ligTab x w τ j) (ligBasis x c w τ j r extras) r.val (zkTab x j).tPad ((UE.tapeZk τ).extra j.val)
    ((UE.tapeZk τ).lanePad j.val) ((UE.tapeZk τ).salt j.val r.val) hl0 hk0 (by omega) h2 rfl
    ?_ ?_ ?_
  · intro a b
    rw [ey1]
    exact hy1 a b
  · intro a b
    rw [eybar]
    exact hybar a b
  · intro a b
    rw [esalt]
    exact hsalt a b

/-- **`swSrcE_repCut` at any clear values `v`**: where table `j`'s rows hold at its base at `v`, rep `r`'s masked values
below `zp 64` are the request's values plus its pads (`hmkV`) and, wherever the source's PIOP returns, from `zp 64` to
`h_ab` level 0's `l0Clear` at the coins plus the pads (`hl0`), the firewall's
rep returning `p` gives the cut from the switched source at `v` the same transcript and log and a proof `p'` with
`eraseZ p' = eraseZ p`. -/
theorem swSrcE_repCut_v {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (hAd : ∀ j, (tabC PE x j).Admits (czExe PE x c j)) (n : ℕ)
    (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (extras : Array _root_.Flock.Claim) (he : (FlockDeploy.tabsOf (stsOf x))[j.val]!.extra (ptsE c spec) = .ok extras)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (q : _root_.Flock.Firewall.Finish.RepIn)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
    (hmkV : ∀ k < (layE x j).zp 64, ((swSrcE x c w v τ j.val).rep r.val).masked k =
      q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (ic : FlockComplete.InnerCommit) (salt : ℕ → ℕ → ByteArray)
    (hsalt : ∀ i, 1 ≤ i → i < (zkTab x j).sch.levels.size → ∀ q < 2 ^ ((zkTab x j).sch.levels.getD i default).d,
      salt i q = (UE.tapeZk τ).salt j.val r.val i q)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val))
    (hl0 : ∀ res t₃ l₃, ((FlockComplete.piopCut (FlockComplete.asksCut (ligCoin c) n) st
        (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) ((swSrcE x c w v τ j.val).rep r.val)
        (FlockComplete.flat cm0.cap) ic.root extras (priorE c j.val r.val)
        (Composed.maskWordsOf (stsOf x)[j.val]!)).run t₀).go l = (.ok (res, t₃), l₃) →
      ∀ k, (layE x j).zp 64 ≤ k → k < (layE x j).hab → ((swSrcE x c w v τ j.val).rep r.val).masked k =
        FlockComplete.LigLock.l0Clear (ligCoin c) (ligP0 x j r) (zkTab x j).sch (ligTab x w τ j)
          (ligBasis x c w τ j r extras) (layE x j) ((UE.tapeZk τ).extra j.val) r.val k +
          (_root_.Flock.Firewall.Finish.padsOf q.draws).h[k]!)
    {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript} {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch (zkTab x j).tPad r.val dom g cm0
      extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals ((UE.tapeZk τ).extra j.val)
      ((UE.tapeZk τ).lanePad j.val) salt).run t₀).go l = (.ok (p, t), l')) :
    ∃ p', ((FlockComplete.repCut (FlockComplete.asksCut (ligCoin c) n)
        (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch r.val dom
        (FlockComplete.eraseC cm0) extras ic ((swSrcE x c w v τ j.val).rep r.val)
        (priorE c j.val r.val) (Composed.maskWordsOf (stsOf x)[j.val]!)).run t₀).go l = (.ok (p', t), l') ∧
      FlockComplete.eraseZ p' = FlockComplete.eraseZ p := by
  subst hg
  have hnr : ¬ Refused x c n := not_refused_of_admits hx hAd n
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm25 : 25 ≤ (stAt x j).m := (hall j.val hj).1
  have hk6 : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
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
  have hmerk : st.merkle = _root_.Flock.MerkleScheme.hm96Sha512 := by
    subst hst
    exact (hall j.val hj).2.2
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  have heS : (stAt x j).extraClaims (ptsE c spec) = .ok extras := by
    rw [tabsOf_stAt] at he
    exact he
  obtain ⟨hpts, -⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hpsz : (ptsE c spec).size = 2 := by
    unfold ptsE
    simp [hpts]
  have hesz : extras.size = spec.points * (stAt x j).regions.size := by
    rw [FlockDeploy.extraClaims_size heS, hpsz, hpts, Nat.mul_comm]
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  -- rep `r`'s PIOP: `m + k_log − 6` bodies of zerocheck and lincheck, then `2 + |extras| + 1` of ring switch
  have hP : ligP0 x j r =
      repBase x (2 * j.val + r.val) + ((stAt x j).m + (stAt x j).c.kLog - 6) + (2 + extras.size + 1) := by
    show repSlot x (2 * j.val + r.val) (fun _ => 0) = _
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val)]
    unfold FlockComplete.piopRoundsZk
    rw [hesz]
    omega
  -- the firewall's PIOP returned, so it sent all its bodies before the stop
  have hP0 : ligP0 x j r ≤ n := by
    obtain ⟨_, _, _, _, _, _, _, hpf⟩ := _root_.Flock.Firewall.Finish.repFwC_piopFw h
    obtain ⟨-, hl₃, hn₃⟩ := FlockComplete.piopFw_at hm14 hpf (trsE x w (UE.tapeZk τ))[j.val]!
    omega
  have hP0' := hP0
  rw [hP] at hP0'
  have hln : l.size ≤ n := by
    rw [hl]
    omega
  -- the masked values below `zp 64`
  have hmk : ∀ i < (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0).zp 64,
      ((swSrcE x c w v τ j.val).rep r.val).masked i =
        q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]! := fun i hi =>
    hmkV i (by rw [← hLyE]; exact hi)
  -- the switched vectors at the firewall's zerocheck and lincheck
  have hsv : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
            (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ k hk
    obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
    obtain ⟨hlc, -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
    have hk2 : k < 2 + extras.size := by
      rw [Array.size_append, FlockComplete.abcClaims_size] at hk
      exact hk
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
    have hab := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
      (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
      (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
      (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc.2.1 hzc.2.2 hlc.2
    have hxk := FlockComplete.Replay.claims_x (e := extras) (FlockComplete.abcClaims_size _ _ _ _)
      (FlockComplete.abcClaims_size _ _ _ _) hab.1 hab.2 k
    have hkE : k < (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
        (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
        (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ extras).size := by
      rw [Array.size_append, FlockComplete.abcClaims_size]
      exact hk2
    have hmem : ∀ (a : Array _root_.Flock.Claim) (i : ℕ), i < a.size → a[i]! ∈ a := fun a i hi => by
      rw [getElem!_pos a i hi]
      exact Array.getElem_mem hi
    have hx6E := FlockComplete.Replay.claims_xsize' (stAt x j) hkm (by omega) tr (zcE c j r) (lcE c j r)
      (ptsE c spec) extras heS _ (hmem _ k hkE)
    have hx6 : ((_root_.Flock.abcClaims (_root_.Flock.Setup.ofCircuit (stAt x j)).m
        (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog zc lc ++ extras)[k]!).x.size = (stAt x j).m - 6 := by
      rw [hxk]
      exact hx6E
    have hpk : ligTab x w τ j = FlockComplete.packed tr (stAt x j).m := by
      show FlockComplete.packed tr (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
      rw [tabsOf_stAt]
      rfl
    rw [hpk, FlockComplete.switchVec_packed (by omega) tr _ hx6, FlockComplete.Replay.switchVec_x hxk]
    exact swSrc_sv c hw τ j r v hx hnr hs heS k hkE (by rw [hfk]; omega)
  -- the rank check
  have hpiop : ∀ zp zc t₁ l₁ lp lc t₂ l₂,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!).run t₀).go l =
        (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => q.vals[i]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[i]!) zc).run t₁).go l₁ =
        (.ok ((lp, lc), t₂), l₂) →
      FlockComplete.MaskRank.maskRankOk
          (priorE c j.val r.val ++ FlockComplete.MaskRank.roundPoints st.m st.kLog t₂.rounds)
          (Composed.maskWordsOf (stsOf x)[j.val]!) = true ∧
        ∀ k < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size,
          ((swSrcE x c w v τ j.val).rep r.val).sv k =
            _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j)
              (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras)[k]! := fun zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ =>
    ⟨rank_at_admits hx hAd hj r.isLt (ligCoin c) n (fun _ _ _ => rfl) st (by rw [hstj, hSm]) (by rw [hstj, hSk]) hkL
      _ _ _ _ ht₀ hl h₁ h₂, hsv zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂⟩
  refine _root_.Flock.Firewall.Finish.repCut_fw (ligCoin c) n st hmerk (zkTab x j).sch (zkTab x j).tPad r.val dom
    (ligTab x w τ j) cm0 extras (_root_.Flock.Firewall.Finish.padsOf q.draws).toC ic q.vals
    ((UE.tapeZk τ).extra j.val) ((UE.tapeZk τ).lanePad j.val) salt
    ((swSrcE x c w v τ j.val).rep r.val) (priorE c j.val r.val)
    (Composed.maskWordsOf (stsOf x)[j.val]!) t₀ l hm14 hln hkL hmk hpiop ?_
    (swSrcE_hinner_sv_v Lr ρ tz frep x hx w τ c n hnr s spec hs j r v q hq hlayR hz hpads hrows hmkV st hst
      (zkTab x j).tPad dom (ligTab x w τ j) cm0 extras hesz ic _ _ _ ht₀ hl hsv) h
  -- level 0's opening: the firewall's PIOP ends at `ligP0` and switched `swsE`
  intro zp lp ss T sws t₃ l₃ h₃
  have hPC := _root_.Flock.Firewall.Finish.piopCut_fw (ligCoin c) n st _ (ligTab x w τ j) _
    ((swSrcE x c w v τ j.val).rep r.val) _ _ extras (priorE c j.val r.val) (Composed.maskWordsOf (stsOf x)[j.val]!)
    hm14 (by exact Nat.le_refl _) (by rw [hkL]; exact Nat.le_refl _) hmk hln hpiop h₃
  obtain ⟨hsws, hl₃, -⟩ := FlockComplete.piopFw_at hm14 h₃ (trsE x w (UE.tapeZk τ))[j.val]!
  have hl3 : l₃.size = ligP0 x j r := by
    rw [hl₃, hl, hP, hkL, hSm, hSk, hesz]
    omega
  have hsE : sws = swsE x c w τ j r extras := by
    rw [hsws]
    unfold swsE
    rw [tabsOf_stAt, ← hst, hl, zcCoins_ligRespE, lcCoins_ligRespE, rsCoins_ligRespE]
  rw [hl3, hsE]
  have H := swSrcE_ligHyp_E_v x hx w hw τ c n hnr s spec hs j r extras he v q (hl0 _ _ _ hPC)
  unfold ligBasis at H
  rw [show ligSt x j = st from (tabsOf_stAt x j).trans hst.symm, ← hLyE] at H
  exact H.ofSalt hsalt

end Masked
end Claims.ZKMal
