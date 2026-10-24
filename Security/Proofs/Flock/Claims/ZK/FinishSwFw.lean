import Proofs.Flock.Claims.ZK.FinishSwRows
import Proofs.Flock.Claims.ZK.FinishSwSv
import Proofs.Flock.Claims.ZK.InnerTauRep

/-!
# `hinner` at the switched source

`FinishSrc.repCut_fw` asks of its source, at the firewall's Ligerito and constraints, the inner proof's (`hinner`,
`InnerSrc`). At the switched prover's source (`swSrcE`) at the firewall's clear values (`valsFw`), `InnerSrc` holds
where the firewall's constraints are the worker's view's first seven rows (`swSrcE_innerSrc_rows`, `SwRows`), and
`Composed.tauFw_at_coef` gives those rows from what the deployed rep fixes: the coins are the opened coins from the
rep's base (`czExe`'s), the masked values below `zp 64` are the view's (`swViewMasked`, `swSrcE_hmk_frame`), the fold
is the model's (`folded_unique`), and level 0's record holds the coins at the rep's slots (`ligeritoPZk_lanes`'
`record`, `LaneCoins`). The PIOP's `piopZk` sends come first, then Ligerito's `k₀ + 7 + 11 R`, so the inner proof
begins at `repSlot … (innerOff 0)`.

* `swViewMasked`: where table `j`'s rows hold at its base, the worker's view's masked message `k` of rep `r` is the
  switched source's.
* **`swSrcE_hinner`**: `repCut_fw`'s `hinner` at the switched source and the request's pads, where the ring switch's
  vectors are the view's (`hsw`).
* `FlockComplete.ringSwitchCut_ss`: a returned ring switch's `ŝ`s are its switched vectors in turn.
* `swSrcE_sv_rank`: past rep `r`'s rank check, the switched source's switched vector `k < 2` is the view's `ŝ_{2r+k}`.
* **`swSrcE_hinner_sv`**: `swSrcE_hinner` with `hsw` from `hpiop`'s switched vectors (`hpsv`): no premise on the
  firewall's runs is left.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- A view at clear values, as the table's: its `ŝ` is `ν + maskHom` at its mask. -/
theorem FlockSoundness.ZK.Table.toBase_onVals_sHat {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2]
    (T : FlockSoundness.ZK.Table F K W Salt Seed D Hi VL) (w : W) (v : T.Vals) (ω : T.onVals.Rand) :
    (T.toBase (T.onVals.view (w, v) ω)).pre.1 = T.nu w + T.maskHom ω.1 :=
  rfl

namespace FlockComplete

open Flock Transcript Replay

/-- **A returned ring switch's `ŝ`s are its switched vectors in turn**: claim `k`'s step pushes `sv k`. -/
theorem ringSwitchCut_ss (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128) (claims : Array Claim)
    {t t' : Transcript} {l l' : Array Claims.Msg} {ss : Array (Array F128)} {T : F128} {sws : Array Switched}
    (h : ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.ok ((ss, T, sws), t'), l')) :
    ss = pushes sv claims.size := by
  have hA : GoAlike (fun a b _ => a = b ∧ a.1 = pushes sv claims.size)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l) := by
    unfold ringSwitchCut
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind (forIn_alike _ _
      (fun k (s s' : Array (Array F128) × Array F128 × Array (Array F128 × Array F128)) =>
        s = s' ∧ s.1 = pushes sv k)
      l.size claims.size ?hb _ _ ?hl ?h0) ?tail
    case hl => exact rfl
    case h0 => exact ⟨rfl, rfl⟩
    case hb =>
      intro k hk s s' t' l' hl' hI
      obtain ⟨rfl, hss⟩ := hI
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      refine alike_bind_eq rfl fun rpp t₃ l₃ e₃ => ?_
      obtain ⟨hlt, -, -, rfl⟩ := sampleVecP_cut_inv coin (by decide) e₃
      refine .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, ?_⟩, by simp only [Array.size_push, hl']⟩
      show ss.push (sv k) = (pushes sv k).push (sv k)
      exact congrArg (fun a => a.push (sv k)) hss
    case tail =>
      intro s s' t' l' _ _ hR
      obtain ⟨⟨hs, hss⟩, -⟩ := hR
      obtain rfl : s = s' := hs
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun gamma t₁ l₁ _ => ?_
      exact alike_pure _ _ ⟨rfl, hss⟩
  rcases hA with ⟨e, L, hx, -⟩ | ⟨a, b, t₁, L, hx, -, hR⟩
  · rw [h] at hx
    simp at hx
  · rw [h] at hx
    simp only [Prod.mk.injEq, Except.ok.injEq] at hx
    obtain ⟨⟨rfl, -⟩, -⟩ := hx
    exact hR.2

end FlockComplete

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
switched source. The firewall's constraints are the worker's view's first seven rows (`tauFw_at_coef`, `swRows_coef`)
where the ring switch's vectors are the view's (`hsw`), level 0's record holding the rep's coins
(`ligeritoPZk_lanes`), and the inner proof begins at `repSlot … (innerOff 0)`. -/
theorem swSrcE_hinner {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (q : _root_.Flock.Firewall.Finish.RepIn) (hA : A j r = q.vals)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep).h j.val r.val k)
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
        ss[1]! = Composed.shArr (swViewE x c w (valsFw x c A B) τ j).pre.1 (2 * r.val + 1)) :
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
    hsw hlen h₁ h₂ h₃ h₄ hrp hlgc hLC
  have hLy : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = innerLayE x j := by
    subst hst
    exact (innerLayE_eq x j).symm
  have hrp' := hrp
  rw [hLy] at hrp'
  have hcons := swRows_coef x c w (valsFw x c A B) τ j r hm14 hrp' ss ⟨hlsz, hlasz⟩ hrow
  have hLyE : _root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0 = layE x j := by
    subst hst
    rfl
  rw [hLyE] at hbat ⊢
  exact swSrcE_innerSrc_rows Lr ρ tz frep x hx w τ c n hnr s spec hs j r (valsFw x c A B) q hq hlayR hz hpads hrows _
    hcons hL₀ fun _ => hbat

/-- The worker's view's `ŝ` is `ν + maskHom` at the tape's mask, at any clear values. -/
theorem swViewE_sHat (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) (j : Fin (jOf x)) :
    (swViewE x c w v τ j).pre.1 = ((tabC PE x j).T (czExe PE x c j)).tab.nu (wtOf PE x w j) +
      ((tabC PE x j).T (czExe PE x c j)).tab.maskHom ((UE.tape x τ).1 j).1 := by
  show (((tabC PE x j).T (czExe PE x c j)).tab.toBase (((tabC PE x j).T (czExe PE x c j)).tab.onVals.view
    (wtOf PE x w j, v j) (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0))).pre.1 = _
  exact (FlockSoundness.ZK.Table.toBase_onVals_sHat _ _ _ _).trans rfl

/-- **The switched source's `ŝ` vector `k < 2` is the worker's view's `ŝ_{2r+k}`**, where rep `r`'s rank check is
reached before the stop: the switched source's switched vectors are the honest source's (`swSrcE_sv`), whose `ŝ` the
guard passes there, `ν + maskHom` at the tape's mask, the view's `ŝ` at any clear values. -/
theorem swSrcE_sv_rank (x : Msg) (hx : Live x) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals) (τ : UE.Tape) {n : ℕ} (hn : ¬ Refused x c n) {s spec}
    (hsx : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (hrank : rankSlot x j.val r.val ≤ n) (k : ℕ) (hk : k < 2) :
    ((swSrcE x c w v τ j.val).rep r.val).sv k = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + k) := by
  rw [swViewE_sHat, swSrcE_sv]
  have h4 : 2 * r.val + k < _root_.Flock.Zk.RANK_POINTS := by
    have := r.isLt; simp only [_root_.Flock.Zk.RANK_POINTS]; omega
  have hpass := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hsx j r .opening hrank
  have hrep : ((tabC PE x j).T (czExe PE x c j)).rep ⟨2 * r.val + k, h4⟩ = r :=
    Composed.rep_of_lt r (2 * r.val + k) (by omega)
  have hs : (cutE x c w τ j).sHat ⟨2 * r.val + k, h4⟩ =
      some ((((tabC PE x j).T (czExe PE x c j)).tab.nu (wtOf PE x w j) +
        ((tabC PE x j).T (czExe PE x c j)).tab.maskHom ((UE.tape x τ).1 j).1) ⟨2 * r.val + k, h4⟩) := by
    have hpass' : famPasses (fun j => ((tabC PE x j).T (czExe PE x c j)).tab)
        (fun j => ((tabC PE x j).T (czExe PE x c j)).rep) j
        (((tabC PE x j).T (czExe PE x c j)).rep ⟨2 * r.val + k, h4⟩) .opening := by
      rw [hrep]; exact hpass
    exact ite_eq_left hpass'
  have hC : 2 * r.val + k < ((tabC PE x j).T (czExe PE x c j)).tab.nC := h4
  unfold srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  show (if hr : r.val < 2 then repSrcCut PE x (czExe PE x c) (cutE x c w τ) j ⟨r.val, hr⟩ else default).sv k = _
  rw [dite_eq_left r.isLt]
  dsimp only [repSrcCut]
  rw [ite_eq_left hk, dite_eq_left hC, hs]
  dsimp only
  unfold Composed.shArr Composed.sHatAt
  rw [dite_eq_left hC]

/-- **`swSrcE_hinner` at the ring switch's vectors from `repCut_fw`'s `hpiop`**: where the switched source's switched
vectors are the firewall's (`hpsv`, `hpiop`'s second half), the firewall's `ŝ`s for claims `0` and `1` are the
worker's view's `ŝ_{2r}` and `ŝ_{2r+1}` (`ringSwitchCut_ss`, `swSrcE_sv_rank`), the rank check's slot being the ring
switch's start. -/
theorem swSrcE_hinner_sv {W R : Type} (Lr : Claims.ZKServer.LeanRelease W R) (ρ : R) (tz : FlockComplete.TapeZk)
    (frep : ℕ → ℕ → ℕ) (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (A : Fin (jOf x) → Fin 2 → Array _root_.Flock.F128) (B : Fin (jOf x) → Fin 2 → ℕ → _root_.Flock.F128)
    (q : _root_.Flock.Firewall.Finish.RepIn) (hA : A j r = q.vals)
    (hq : q.draws = Lr.drawsAt x ρ (frep j.val r.val)) (hlayR : Lr.lay x (frep j.val r.val) = layE x j)
    (hz : (Lr.draws ρ (frep j.val r.val)).size =
      _root_.Flock.Firewall.Frame.drawBytes (Lr.lay x (frep j.val r.val)))
    (htape : ∀ k < (layE x j).hab, (UE.tapeZk τ).h j.val r.val k =
      (Claims.ZKServer.tapeOfDraws tz (Lr.lay x) (Lr.draws ρ) frep).h j.val r.val k)
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
        ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val).sv k =
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
        ((swSrcE x c w (valsFw x c A B) τ j.val).rep r.val) l₄.size := by
  refine swSrcE_hinner Lr ρ tz frep x hx w τ c n hnr s spec hs j r A B q hA hq hlayR hz htape hpads hrows st hst tPad
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
  have e : ∀ k < 2, ss[k]! = Composed.shArr (swViewE x c w (valsFw x c A B) τ j).pre.1 (2 * r.val + k) := by
    intro k hk
    rw [hss]
    refine (FlockComplete.Replay.pushes_get _ _ k (by omega)).trans ?_
    refine (hpsv zp zc t₁ l₁ lp lc t₂ l₂ h₁ h₂ k (by omega)).symm.trans ?_
    exact swSrcE_sv_rank x hx c w (valsFw x c A B) τ hnr hs j r hrank k hk
  have e0 := e 0 (by norm_num)
  rw [Nat.add_zero] at e0
  exact ⟨e0, e 1 (by norm_num)⟩

end Masked
end Claims.ZKMal
