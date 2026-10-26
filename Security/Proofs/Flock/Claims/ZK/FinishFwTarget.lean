import Proofs.Flock.Claims.ZK.FinishFwRep
import Flock.Firewall.Drive

/-!
# A deployed rep's ring-switch target is the opening's

`fwRows_at` takes the firewall's ring-switch target `T` as the opening's (`Composed.openT`) at the view's masked claims'
`ŝ`. Here that is read off the deployed rep: `repFwC`'s ring switch on table `j`'s rep `r`, answered by the opened coins
(`ligCoin`) from the rep's base, on the packed table of the witness's trace (`ligTab`), with extra claims at the
statement's region claims' points.

* `FlockComplete.ringSwitchCut_T`: a returned ring switch's `T` is `Σ_k γ_k·β_k`, `β_k = dot (eq r''_k) (t(s_k))`, at
  its coins, whatever the switched vectors.
* **`Masked.fwTarget_at`**: at the deployed rep, `T = openT` at the view's `ν + maskHom`: claim `k`'s `β_k` is the
  model's `recombine` (`Refine.recombine_eq`); claims `0` and `1` send the view's `ŝ` (as `fwRows_at` reads them), and
  region claim `k − 2` sends the witness's region message (`sv_region`), which is the public one at a valid witness
  (`Composed.coinedOf_reg`, `complete_tabC`); the coins are `czExe`'s `rpp` and `rsGamma` (`repSlot_eq`).

Neither reads the worker's clear values: `openT` reads `ν`, which reads the witness alone.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace FlockComplete

open Flock Transcript Replay

/-- **A returned ring switch's target at its coins**: `T = Σ_k γ_k·dot (eq r''_k) (t(s_k))`, claim `k`'s `r''` at
`l.size + k` and the batching `γ` at `l.size + |claims|`, whatever the switched vectors. -/
theorem ringSwitchCut_T (coin : ℕ → ℕ → F128) (n : ℕ) (sv : ℕ → Array F128) (claims : Array Claim)
    (hc : claims.size ≠ 0) {t t' : Transcript} {l l' : Array Claims.Msg} {ss : Array (Array F128)} {T : F128}
    {sws : Array Switched}
    (h : ((ringSwitchCut (asksCut coin n) sv claims).run t).go l = (.ok ((ss, T, sws), t'), l')) :
    T = dot ((Array.range claims.size).map (coin (l.size + claims.size)))
      (pushes (fun k => dot (eqTable ((Array.range 7).map (coin (l.size + k))))
        (FlockSoundness.Refine.tOf (sv k))) claims.size) := by
  have hA : GoAlike (fun (a b : Array (Array F128) × F128 × Array Switched) _ => a = b ∧
        a.2.1 = dot ((Array.range claims.size).map (coin (l.size + claims.size)))
          (pushes (fun k => dot (eqTable ((Array.range 7).map (coin (l.size + k))))
            (FlockSoundness.Refine.tOf (sv k))) claims.size))
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l)
      (((ringSwitchCut (asksCut coin n) sv claims).run t).go l) := by
    unfold ringSwitchCut
    refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
    obtain rfl := liftTM_inv e₁
    refine alike_bind (forIn_alike _ _
      (fun k (s s' : Array (Array F128) × Array F128 × Array (Array F128 × Array F128)) =>
        s = s' ∧ s.2.1 = pushes (fun k => dot (eqTable ((Array.range 7).map (coin (l.size + k))))
          (FlockSoundness.Refine.tOf (sv k))) k)
      l.size claims.size ?hb _ _ ?hl ?h0) ?tail
    case hl => exact rfl
    case h0 => exact ⟨rfl, rfl⟩
    case hb =>
      intro k hk s s' t' l' hl' hI
      obtain ⟨rfl, hpc⟩ := hI
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun _ t₁ l₁ e₁ => ?_
      obtain rfl := liftTM_inv e₁
      refine alike_bind_eq rfl fun _ t₂ l₂ e₂ => ?_
      obtain rfl := liftTM_inv e₂
      refine alike_bind_eq rfl fun rpp t₃ l₃ e₃ => ?_
      obtain ⟨-, rfl, -, rfl⟩ := sampleVecP_cut_inv coin (by decide) e₃
      refine .inr ⟨_, _, _, _, rfl, rfl, _, _, rfl, rfl, ⟨rfl, ?_⟩, by simp only [Array.size_push, hl']⟩
      dsimp only at hpc ⊢
      rw [hpc, hl']
      all_goals rfl
    case tail =>
      intro s s' t' l' _ _ hR
      obtain ⟨⟨hs, hpc⟩, hsz⟩ := hR
      obtain rfl : s = s' := hs
      obtain ⟨ss, betas, perClaim⟩ := s
      try dsimp only
      refine alike_bind_eq rfl fun gamma t₁ l₁ e₁ => ?_
      obtain ⟨hlt, rfl, -, rfl⟩ := sampleVecP_cut_inv coin hc e₁
      refine alike_pure _ _ ⟨rfl, ?_⟩
      dsimp only at hpc ⊢
      rw [hpc, hsz]
  rcases hA with ⟨e, L, hx, -⟩ | ⟨a, b, t₁, L, hx, -, hR⟩
  · rw [h] at hx
    simp at hx
  · rw [h] at hx
    simp only [Prod.mk.injEq, Except.ok.injEq] at hx
    obtain ⟨⟨rfl, -⟩, rfl⟩ := hx
    exact hR.2

end FlockComplete

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)

namespace Masked

/-- Entry `i` of a list of pairs is pair `i / 2`'s at `i % 2`. -/
theorem pairs_get {α β γ : Type} [Inhabited α] [Inhabited γ] (g : α → β → γ) (p0 p1 : β) :
    ∀ (L : List α) (i : ℕ), i < 2 * L.length →
      (L.flatMap fun a => [g a p0, g a p1])[i]! = g L[i / 2]! (if i % 2 = 0 then p0 else p1)
  | [], i, hi => by simp at hi
  | a :: L, i, hi => by
    rcases Nat.lt_or_ge i 2 with h2 | h2
    · interval_cases i <;> simp
    · obtain ⟨i', rfl⟩ : ∃ i', i = i' + 2 := ⟨i - 2, by omega⟩
      rw [show (i' + 2) / 2 = i' / 2 + 1 by omega, show (i' + 2) % 2 = i' % 2 by omega]
      simp only [List.flatMap_cons, List.cons_append, List.nil_append, List.getElem!_cons_succ]
      exact pairs_get g p0 p1 L i' (by simp only [List.length_cons] at hi; omega)

/-- **Extra coordinate `i` at two link points** is region `i / 2`'s point at link point `i % 2`, past its six skip
coordinates. -/
theorem driveExtras_get (st : _root_.Flock.Stmt) (p0 p1 : Array _root_.Flock.F128) :
    ∀ i < 2 * st.regions.size, (_root_.Flock.Firewall.Drive.extras st #[p0, p1])[i]! =
      (st.regionPoint st.regions[i / 2]! (if i % 2 = 0 then p0 else p1)).extract 6
        (st.regionPoint st.regions[i / 2]! (if i % 2 = 0 then p0 else p1)).size := by
  intro i hi
  have h := pairs_get (fun reg (q : Array _root_.Flock.F128) => (st.regionPoint reg q).extract 6
    (st.regionPoint reg q).size) p0 p1 st.regions.toList i (by rw [Array.length_toList]; exact hi)
  have e : st.regions.toList[i / 2]! = st.regions[i / 2]! := Array.getElem!_toList
  rw [e] at h
  have hl : (_root_.Flock.Firewall.Drive.extras st #[p0, p1]).toList = st.regions.toList.flatMap fun reg =>
      [(st.regionPoint reg p0).extract 6 (st.regionPoint reg p0).size,
        (st.regionPoint reg p1).extract 6 (st.regionPoint reg p1).size] := by
    unfold _root_.Flock.Firewall.Drive.extras
    simp
  rw [← Array.getElem!_toList, hl]
  exact h

set_option maxHeartbeats 1000000 in
/-- **A deployed rep's ring-switch target is the opening's**: where `repFwC`'s zerocheck, lincheck and ring switch on
table `j`'s rep `r`, answered by the opened coins from the rep's base at the packed table of the witness's trace (`hg`),
return with extra claims at the region points of the link (`Drive.extras`, `hxsz`, `hxs`), the ring switch's `T` is
`Composed.openT` at the view's `ν + maskHom`. -/
theorem fwTarget_at (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (xs : Array _root_.Flock.Claim) (hxsz : xs.size = spec.points * (stAt x j).regions.size)
    (hxs : xs.map (·.x) = _root_.Flock.Firewall.Drive.extras (stAt x j) (ptsE c spec))
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j))
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array _root_.Flock.F128)
    {t₀ : _root_.Flock.Transcript} {l : Array Msg} (hl : l.size = repBase x (2 * j.val + r.val)) :
    ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
      ((FlockComplete.zerocheckCut (FlockComplete.asksCut (ligCoin c) n) st.m st.digest (FlockComplete.flat cm0.cap)
          ic.root (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          fun i => vals[i]! + pads.h[i]!).run t₀).go l = (.ok ((zp, zc), t₁), l₁) →
      ((FlockComplete.lincheckCut (FlockComplete.asksCut (ligCoin c) n) st.fold st.m
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0)
          (fun i => vals[i]! + pads.h[i]!) zc).run t₁).go l₁ = (.ok ((lp, lc), t₂), l₂) →
      ((FlockComplete.ringSwitchCut (FlockComplete.asksCut (ligCoin c) n)
          (fun k => _root_.Flock.Firewall.Finish.switchVec g (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!)
          (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)).run t₂).go l₂ = (.ok ((ss, T, sws), t₃), l₃) →
      fOf T = Composed.openT (zkTab x j).table (stAt x j) (czExe PE x c j) r
        (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
          ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) := by
  subst hg
  intro zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
  have hk6' : 6 ≤ (stAt x j).c.kLog := (zkTab x j).wf.1.hk6
  have hkm : (stAt x j).c.kLog ≤ (stAt x j).m := (zkTab x j).wf.1.hk
  have hfk : (stAt x j).circuitFold.kLog = (stAt x j).c.kLog := rfl
  have hj : j.val < (stsOf x).size := lt_of_jOf j
  have hstj : (stsOf x)[j.val]! = stAt x j := getElem!_pos (stsOf x) j.val hj
  obtain ⟨-, -, -, hall⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hs
  have hm14' : 14 ≤ (stAt x j).m := le_trans (by norm_num) (hall j.val hj).1
  have hm7 : 7 ≤ (stAt x j).m := by omega
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
  -- the extra claims: two per region
  obtain ⟨hpts, hmp⟩ := HiddenExec.setupTables_pointsAny (FlockComplete.ZkTables.setupTables_of hx.1 hs)
  have hA2 : (_root_.Flock.abcClaims st.m st.kLog zc lc).size = 2 := FlockComplete.abcClaims_size _ _ _ _
  have hN : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size = 2 + 2 * (stAt x j).regions.size := by
    rw [Array.size_append, hA2, hxsz, hpts]
  -- the runs' lengths
  obtain ⟨hzc, hl₁⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
  obtain ⟨hlc, hl₂⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
  obtain ⟨-, hl₃, -⟩ := FlockComplete.ringSwitchCut_sws (ligCoin c) n _ _ (by rw [hN]; omega) h₃
  rw [hN] at hl₃
  -- claims `0` and `1` send the view's `ŝ`
  have hsw : ss[0]! = Composed.shArr (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
          ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) (2 * r.val) ∧
        ss[1]! = Composed.shArr (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
          ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) (2 * r.val + 1) := by
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
    have hsv : ∀ k < 2,
        ((swSrcE x c w v τ j.val).rep r.val).sv k =
          _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j) (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]! := by
      intro k hk2
      obtain ⟨hzc', hl₁'⟩ := FlockComplete.zerocheckCut_at (ligCoin c) n hm14 _ _ _ _ _ h₁
      obtain ⟨hlc', -⟩ := FlockComplete.lincheckCut_at (ligCoin c) n _ _ _ _ _ h₂
      subst hst
      set tr := (trsE x w (UE.tapeZk τ))[j.val]! with htr
      have hzE : FlockComplete.zcCoins (ligCoin c) l.size = zcE c j r := by
        rw [hl]
      have hlE : FlockComplete.lcCoins (ligCoin c) l₁.size
          ((_root_.Flock.Setup.ofCircuit (stAt x j)).fold.kLog - 6) = lcE c j r := by
        rw [hl₁', hl]
        rfl
      rw [hzE] at hzc'
      rw [hlE] at hlc'
      have hab' := FlockComplete.Replay.abc_x (m := (_root_.Flock.Setup.ofCircuit (stAt x j)).m)
        (kLog := (_root_.Flock.Setup.ofCircuit (stAt x j)).kLog) (z₁ := zc) (l₁ := lc)
        (z₂ := FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
        (l₂ := FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) hzc'.2.1 hzc'.2.2 hlc'.2
      have hxk := FlockComplete.Replay.claims_x (e := xs) (FlockComplete.abcClaims_size _ _ _ _)
        (FlockComplete.abcClaims_size _ _ _ _) hab'.1 hab'.2 k
      have hA2' : (_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r))).size = 2 :=
        FlockComplete.abcClaims_size _ _ _ _
      have hx6E : ((_root_.Flock.abcClaims (stAt x j).m (stAt x j).c.kLog
          (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr)
          (FlockComplete.Replay.Lc.out (stAt x j).circuitFold (stAt x j).m tr
            (FlockComplete.Replay.Zc.out (stAt x j).m (zcE c j r) tr) (lcE c j r)) ++ xs)[k]!).x.size =
          (stAt x j).m - 6 := by
        rw [FlockComplete.Replay.get_left (lt_of_lt_of_eq hk2 hA2'.symm)]
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
      have hl₃' := (FlockComplete.ringSwitchCut_sws (ligCoin c) n _ _
        (by rw [Array.size_append, FlockComplete.abcClaims_size]; omega) h₃).2
      rw [Array.size_append, FlockComplete.abcClaims_size] at hl₃'
      exact src_sv_abc c hw τ j r hx hnr hs xs k hk2 (by rw [hfk]; omega)
    have hss := FlockComplete.ringSwitchCut_ss (ligCoin c) n _ _ h₃
    have e : ∀ k < 2, ss[k]! = Composed.shArr (swViewE x c w v τ j).pre.1 (2 * r.val + k) := by
      intro k hk
      rw [hss]
      refine (FlockComplete.Replay.pushes_get _ _ k (by rw [hN]; omega)).trans ?_
      refine (hsv k (by omega)).symm.trans ?_
      exact swSrcE_sv_rank x hx c w v τ hnr hs j r hrank k hk
    have e0 := e 0 (by norm_num)
    rw [Nat.add_zero, swViewE_sHat] at e0
    have e1 := e 1 (by norm_num)
    rw [swViewE_sHat] at e1
    exact ⟨e0, e1⟩
  have hss := FlockComplete.ringSwitchCut_ss (ligCoin c) n _ _ h₃
  -- region claim `k − 2` sends the public message
  have hreg : ((tabC PE x j).T (czExe PE x c j)).tab.sReg (wtOf PE x w j) =
      ((tabC PE x j).T (czExe PE x c j)).tab.pubReg :=
    Composed.coinedOf_reg PE (zkTab x j).table (stAt x j) (zkTab x j).wf.1 (zkTab x j).wf.2 rfl (zkTab x j).whole
      (czExe PE x c j) (wtOf PE x w j) (complete_tabC PE hw j)
  have hregk : ∀ k, 2 ≤ k → ∀ hk : k - 2 < 2 * (stAt x j).regions.size,
      FlockSoundness.Refine.vecOf 128 (_root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j) (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]!) =
        Composed.pubRegOf (stAt x j) ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).link
          ⟨k - 2, hk⟩ := by
    intro k hk2 hkr
    set ρ := ((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)) with hρdef
    have hmP : (stAt x j).mPts = spec.mPts := hmp j.val hj
    have hmx : mPtsOf x = spec.mPts := by unfold mPtsOf; rw [hs]
    have hxk : (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k]! = xs[k - 2]! := by
      rw [FlockComplete.Replay.get_right (show (_root_.Flock.abcClaims st.m st.kLog zc lc).size ≤ k by rw [hA2]; omega),
        hA2]
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
    have hlt : k - 2 < xs.size := by rw [hxsz, hpts]; omega
    have hif : (if (k - 2) % 2 = 0 then Array.ofFn (ρ.link 0) else Array.ofFn (ρ.link 1)) =
        Array.ofFn (ρ.link (if (k - 2) % 2 = 0 then 0 else 1)) := by
      split <;> rfl
    have hxe : (xs[k - 2]!).x = ((stAt x j).regionPoint (stAt x j).regions[(k - 2) / 2]!
        (Array.ofFn (ρ.link (if (k - 2) % 2 = 0 then 0 else 1)))).extract 6
        ((stAt x j).regionPoint (stAt x j).regions[(k - 2) / 2]!
          (Array.ofFn (ρ.link (if (k - 2) % 2 = 0 then 0 else 1)))).size := by
      have h1 : (xs[k - 2]!).x = (xs.map (·.x))[k - 2]! := by
        rw [getElem!_pos xs (k - 2) hlt, getElem!_pos _ (k - 2) (by rw [Array.size_map]; exact hlt),
          Array.getElem_map]
      rw [h1, hxs, hptsE, driveExtras_get _ _ _ (k - 2) hkr, hif]
    have hx6 : (xs[k - 2]!).x.size = (stAt x j).m - 6 := by
      rw [hxe, Array.size_extract, FlockSoundness.Refine.regionPoint_size]; omega
    have hpt : ((xs[k - 2]!).x.extract 1).toList = Composed.regionPtOf (stAt x j) ρ.link ⟨k - 2, hkr⟩ := by
      rw [hxe, extract_six_one]
      rfl
    have hpk : ligTab x w τ j = FlockComplete.packed (trsE x w (UE.tapeZk τ))[j.val]! (stAt x j).m := by
      show FlockComplete.packed _ (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = _
      rw [tabsOf_stAt]
      rfl
    have hsr := sv_region c hw τ j hm7 ⟨k - 2, hkr⟩ _ hx6 hpt
    rw [hxk, hpk, FlockComplete.switchVec_packed hm7 _ _ hx6, ← hsr, hreg]
    funext v'
    simp only [FlockSoundness.Refine.vecOf]
    rw [getElem!_pos _ v'.val (by rw [Array.size_ofFn]; exact v'.isLt), Array.getElem_ofFn]
    rfl
  -- the target, claim by claim
  have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
      .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
    rw [stsOf_half]
    exact (zkTab x j).hz
  have hslot : repSlot x (2 * j.val + r.val) (fun _ => 0) = l₃.size := by
    rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
    unfold FlockComplete.piopRoundsZk
    rw [hpts]
    try dsimp only
    omega
  have hT := FlockComplete.ringSwitchCut_T (ligCoin c) n _ _ (by rw [hN]; omega) h₃
  rw [hT, FlockSoundness.Refine.fOf_dot _ _ (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size (by rw [Array.size_map, Array.size_range])]
  unfold Composed.openT
  refine Fintype.sum_equiv (finCongr hN) _ _ fun k => ?_
  have hk : k.val < (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size := k.isLt
  rw [FlockSoundness.Refine.getElem!_range_map _ hk, FlockComplete.Replay.pushes_get _ _ k.val hk,
    FlockSoundness.Refine.recombine_eq _ _ (by rw [Array.size_map, Array.size_range])]
  refine congrArg₂ (· * ·) ?_ ?_
  · show fOf (slotAt c (l₂.size + (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs).size) k.val) = fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 0) - 1) k.val)
    rw [hslot, hl₃]
    exact congrArg (fun q => fOf (slotAt c q k.val)) (by omega)
  · have hrpp : FlockSoundness.Refine.vecOf 7 ((Array.range 7).map (ligCoin c (l₂.size + k.val))) =
        fun i => fOf (((czExe PE x c j).rest : Composed.RestOf (zkTab x j).table (stAt x j)).rpp r
          (finCongr hN k) i) := by
      funext i
      simp only [FlockSoundness.Refine.vecOf]
      rw [FlockSoundness.Refine.getElem!_range_map _ i.isLt]
      show fOf (slotAt c (l₂.size + k.val) i.val) =
        fOf (slotAt c (repBase x (2 * j.val + r.val) + (6 + ((stAt x j).m - 6) + ((stAt x j).c.kLog - 6) + k.val))
          i.val)
      exact congrArg (fun q => fOf (slotAt c q i.val)) (by omega)
    rw [hrpp]
    refine congrArg (FlockSoundness.Model.recombine FlockSoundness.execArith · _) ?_
    show FlockSoundness.Refine.vecOf 128 (_root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j) (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k.val]!) =
      Composed.swAt (zkTab x j).table (stAt x j) (czExe PE x c j) r _ k.val
    rcases Nat.lt_or_ge k.val 2 with hk2 | hk2
    · have hsk : _root_.Flock.Firewall.Finish.switchVec (ligTab x w τ j) (_root_.Flock.abcClaims st.m st.kLog zc lc ++ xs)[k.val]! = ss[k.val]! := by
        rw [hss, FlockComplete.Replay.pushes_get _ _ k.val hk]
      rw [hsk]
      have hsh : ∀ i, FlockSoundness.Refine.vecOf 128 (Composed.shArr
          (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
            ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
              (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) i) =
          Composed.sHatAt (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
            ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
              (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) i := by
        intro i
        funext v'
        simp only [FlockSoundness.Refine.vecOf, Composed.shArr]
        rw [getElem!_pos _ v'.val (by rw [Array.size_ofFn]; exact v'.isLt), Array.getElem_ofFn]
        rfl
      unfold Composed.swAt
      interval_cases hkv : k.val
      · rw [hsw.1, hsh, ite_eq_left rfl]
      · rw [hsw.2, hsh, ite_eq_right (by norm_num), ite_eq_left rfl]
    · have hkr : k.val - 2 < 2 * (stAt x j).regions.size := by
        have := k.isLt
        omega
      rw [hregk k.val hk2 hkr]
      unfold Composed.swAt
      rw [ite_eq_right (by omega), ite_eq_right (by omega)]
      funext v'
      rw [dif_pos hkr]

end Masked
end Claims.ZKMal
