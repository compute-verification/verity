import Proofs.Flock.Claims.ZK.FinishSwRows
import Proofs.Flock.Claims.ZK.InnerTauRep

/-!
# `hinner` at the switched source

`FinishSrc.repCut_fw` asks of its source, at the firewall's Ligerito and constraints, the inner proof's (`hinner`,
`InnerSrc`). At the switched prover's source (`swSrcE`) at the firewall's clear values (`valsFw`), `InnerSrc` holds
where the firewall's constraints are the worker's view's first seven rows (`swSrcE_innerSrc_rows`, `SwRows`), and
`Composed.tauFw_at` gives those rows from what the deployed rep fixes: the coins are the opened coins from the rep's
base (`czExe`'s), the masked values below `zp 64` are the view's (`swViewMasked`, `swSrcE_hmk_frame`) and the fold is
the model's (`folded_unique`). The inner proof begins at `repSlot … (innerOff 0)`: the PIOP's `piopZk` sends, then
Ligerito's `k₀ + 7 + 11 R`.

* `swViewMasked`: where table `j`'s rows hold at its base, the worker's view's masked message `k` of rep `r` is the
  switched source's.
* **`swSrcE_hinner`**: `repCut_fw`'s `hinner` at the switched source and the request's pads, at the ring switch's
  vectors (`hsw`) and level 0 read at the worker's view (`hlane`), as `tauFw_at` takes them.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf)

namespace Masked

/-- **The worker's view's masked message is the switched source's**, where table `j`'s rows hold at its base: the
switched cut's masked values are the view's there (`swCutE_masked_of_rowsAt`), and past `n_E` both are `0`. -/
theorem swViewMasked (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) (r : Fin 2)
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) (k : ℕ) :
    (Composed.msgAt ((swViewE x c w v τ j).pre.2.1 r) k).val = ((swSrcE x c w v τ j.val).rep r.val).masked k := by
  unfold swSrcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show _ = (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (swCutE x c w v τ) j ⟨r.val, hr⟩
    else default).masked k
  rw [dite_eq_left r.isLt]
  show _ = (if h : k < ((tabC PE x j).T (czExe PE x c j)).tab.nE then
      FlockLevel3.GF128.val ((swCutE x c w v τ j).masked r ⟨k, h⟩) else 0)
  rw [swCutE_masked_of_rowsAt x c w v τ j hrows]
  unfold Composed.msgAt
  split_ifs with h₁ h₂
  all_goals first | rfl | (exfalso; exact h₂ h₁) | (exfalso; exact h₁ h₂) | (exfalso; contradiction) |
    exact FlockLevel3.GF128.zero_eq

/-- **`repCut_fw`'s `hinner` at the switched source**, for table `j`'s rep `r` at the firewall's request `q` (frame rep
`frep j r` of the release `Lr`), at the setup `st` of table `j`'s statement and the log `l` at the rep's base: where
table `j`'s rows hold at its base at the firewall's clear values (`valsFw`), the passed values are `q`'s and the pads
the request's (`swSrcE_hmk_frame`'s and `swSrcE_innerSrc_rows`'s premises), then wherever the firewall's PIOP and
Ligerito return, its replay and level 0's constraints return, and the batched constraint holds, `InnerSrc` holds at the
switched source. The firewall's constraints are the worker's view's first seven rows (`tauFw_at`, `swRows_lane`) at
the ring switch's vectors the view's (`hsw`) and level 0 read at the view (`hlane`), and the inner proof begins at
`repSlot … (innerOff 0)`. -/
theorem swSrcE_hinner {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (q : _root_.Flock.Firewall.Finish.RepIn) (hA : A j r = q.vals)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (htape : (UE.tapeZk τ).h j.val r.val =
      (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep).h j.val r.val)
    (hpads : FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (layE x j) =
      FlockComplete.padsOf (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep) j.val r.val (layE x j))
    (hrows : ((tabC PE x j).T (czExe PE x c j)).tab.onVals.RowsAt (wtOf PE x w j, valsFw x c A B j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1)
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
      ss[0]! = Composed.shArr (swViewE x c w (valsFw x c A B) τ j).pre.1 (2 * r.val) ∧
        ss[1]! = Composed.shArr (swViewE x c w (valsFw x c A B) τ j).pre.1 (2 * r.val + 1))
    (hlane : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄,
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
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut (ligCoin c) n)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512 (zkTab x j).sch
          T cm0 g (FlockComplete.basisVals sws (st.m - 7))
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h r.val tPad extra lanePad salt (fun _ _ => 0)).run t₃).go
          l₃ = (.ok ((lg, zl, rec), t₄), l₄) →
      LaneAt (zkTab x j).table (stAt x j) (czExe PE x c j) r ((swViewE x c w (valsFw x c A B) τ j).pre.2.1 r)
        (swViewE x c w (valsFw x c A B) τ j).pre.1 ((swViewE x c w (valsFw x c A B) τ j).pre.2.2.2.2.2 r).1 lg rec
        zl) :
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
        ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) l₄.size := by
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
      (Composed.msgAt ((swViewE x c w (valsFw x c A B) τ j).pre.2.1 r) k).val =
        q.vals[k]! + (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h[k]! := fun k hk => by
    rw [swViewMasked x c w (valsFw x c A B) τ j r hrows k]
    exact swSrcE_hmk_frame Lr ρ tz frep x c w A B τ j r q hA hq hlayR hz htape hrows k hk
  -- the rows
  have hrow := Composed.tauFw_at (zkTab x j).table (stAt x j) (czExe PE x c j) r
    ((swViewE x c w (valsFw x c A B) τ j).pre.2.1 r) (swViewE x c w (valsFw x c A B) τ j).pre.1
    ((swViewE x c w (valsFw x c A B) τ j).pre.2.2.2.2.2 r).1 ((swViewE x c w (valsFw x c A B) τ j).pre.2.2.2.2.1 r)
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
    hsw hlen hlane h₁ h₂ h₃ h₄ hrp hlgc
  have hLy : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = innerLayE x j := by
    subst hst
    exact (innerLayE_eq x j).symm
  have hrp' := hrp
  have hlgc' := hlgc
  rw [hLy] at hrp' hlgc'
  have hcons := swRows_lane x c w (valsFw x c A B) τ j r hm14 hrp' ss (innerLayE_layE x j).symm hlen
    (hlane zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ h₁ h₂ h₃ h₄) hlgc' hrow
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
  have hl4 := (FlockComplete.ligeritoPZk_exact (respE c) _root_.Flock.Sha512.hash _root_.Flock.MerkleScheme.hm96Sha512
    (zkTab x j).sch T cm0 g (FlockComplete.basisVals sws (st.m - 7))
    (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
    (_root_.Flock.Firewall.Finish.padsOf q.draws).toC.h r.val tPad extra lanePad salt (fun _ _ => 0) (by omega) hk0 h2
    (fun i hi => (hlv i hi).q_ne_zero)
    (FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
      t₃ l₃ _ _ h₄)).1
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hL₀ : l₄.size = repSlot x (2 * j.val + r.val) (innerOff 0) := by
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk innerOff
    omega
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  rw [hLyE] at hbat ⊢
  exact swSrcE_innerSrc_rows Lr ρ tz frep x hx w τ c n hnr s spec hs j r (valsFw x c A B) q hq hlayR hz hpads hrows _
    hcons hL₀ fun _ => hbat

end Masked
end Claims.ZKMal
