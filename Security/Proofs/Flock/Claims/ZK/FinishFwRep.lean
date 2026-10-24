import Proofs.Flock.Claims.ZK.FinishFwRows
import Proofs.Flock.Claims.ZK.FinishSwRep

/-!
# A deployed rep's rows from the firewall's run

`Composed.rowsAt_of_fw` decides table `j`'s rows at its base from what the firewall's run on each rep fixes. Here that
is read off the deployed rep: `repFwC` on table `j`'s rep `r`, answered by the opened coins (`ligCoin`) from the rep's
base, on the packed table of the witness's trace (`ligTab`), at pads with `h_ab = h_a·h_b` and `h_w = h_x·h_y` on the
layout's length.

**`fwRows_at`**: where that run returns, its constraints and pads are `rowsAt_of_fw`'s at rep `r`, at the worker's clear
values `v`, the tape's mask and extra lanes, given what the clear values `v` and the opening's sums are at the run:

* `hM`: below `zp 64`, the clear values are the request's;
* `hV`: level 0's out-of-domain value, lane messages and pair's sums are the masked values (`LaneVals`);
* `hT`: the ring switch's `T` is the opening's (`Composed.openT`);
* `hTP`: level 0's restart is `Σ y₁'·bFold`.

The rest is the deployed rep's own: the coins from the rep's base are `czExe`'s (as `swSrcE_hinner` reads them), the
fold is the model's (`folded_unique`), the ring switch's `ŝ`s are the view's (`ringSwitchCut_ss`, `src_sv_abc`,
`swSrcE_sv_rank`: they read claims `0` and `1` alone, not the extra claims' values), and level 0's record holds the rep's coins (`ligeritoPZk_lanes`' `record`).
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKMal
open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer
open FlockSoundness.Refine (fOf gOf)

namespace Masked

/-- **A deployed rep's rows from the firewall's run**: where `repFwC` on table `j`'s rep `r`, answered by the opened
coins from the rep's base at the packed table of the witness's trace (`hg`), with as many extra claims `xs` as the
statement's (`hxsz`), returns at pads `pads` with `h_ab = h_a·h_b` and `h_w = h_x·h_y` on the layout's length, the pads
and the run's seven constraints are `rowsAt_of_fw`'s at rep `r`, at the clear values `v` and the tape's mask and extra
lanes, wherever below `zp 64` the clear values are the request's (`hM`), level 0's values are the masked values (`hV`),
the ring switch's `T` is the opening's (`hT`) and level 0's restart is `Σ y₁'·bFold` (`hTP`). -/
theorem fwRows_at (x : Msg) (hx : Live x) (w : Claims.Flock.Wit) (hw : RZk x w) (τ : UE.Tape)
    (c : Fin (refK x) → FlockDeploy.Coin) (n : ℕ) (hnr : ¬ Refused x c n) (s : _) (spec : _root_.Flock.SessionSpec)
    (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (j : Fin (jOf x)) (r : Fin 2)
    (v : ∀ j, ((tabC PE x j).T (czExe PE x c j)).tab.Vals)
    (xs : Array _root_.Flock.Claim) (hxsz : xs.size = spec.points * (stAt x j).regions.size)
    (st : _root_.Flock.Setup) (hst : st = _root_.Flock.Setup.ofCircuit (stAt x j)) (tPad : ℕ) (dom : ByteArray)
    (g : Array _root_.Flock.F128) (hg : g = ligTab x w τ j) (cm0 : FlockComplete.Commit)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array _root_.Flock.F128)
    (extra lanePad : ℕ → ℕ → _root_.Flock.F128) (salt : ℕ → ℕ → ByteArray)
    (hps : pads.h.size = (Composed.layOf (zkTab x j).table (stAt x j)).len)
    (hab : pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).hab]! =
      pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).fa]! *
        pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).fb]!)
    (htw : pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).tw]! =
      pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).tx]! *
        pads.h[(Composed.layOf (zkTab x j).table (stAt x j)).ty]!)
    {t₀ : _root_.Flock.Transcript} (ht₀ : t₀.rounds = #[]) {l : Array Msg}
    (hl : l.size = repBase x (2 * j.val + r.val)) {p : _root_.Flock.Zk.ZkProof} {t : _root_.Flock.Transcript}
    {l' : Array Msg}
    (h : ((_root_.Flock.Firewall.Finish.repFwC (FlockComplete.asksCut (ligCoin c) n)
      (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) st (zkTab x j).sch tPad r.val dom g cm0 xs pads ic
      vals extra lanePad salt).run t₀).go l = (.ok (p, t), l'))
    (hM : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      (Composed.msgAt (((tabC PE x j).T (czExe PE x c j)).tab.onVals.vals r (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1) k).val = vals[k]!)
    (hV : LaneVals (zkTab x j).table (stAt x j)
      (fun i => ((tabC PE x j).T (czExe PE x c j)).tab.onVals.vals r (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 i + fOf pads.h[i.val]!)
      p.proof.ligerito p.level0)
    (hT : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃,
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
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1))
    (hTP : gOf p.level0.tPrime = ∑ col, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.y1' (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (czExe PE x c j) r col.val) :
    ∃ (hh : Array _root_.Flock.F128) (cons : Array (String × _root_.Flock.Zk.Lin)),
      hh.size = (Composed.layOf (zkTab x j).table (stAt x j)).len ∧
      hh[(Composed.layOf (zkTab x j).table (stAt x j)).hab]! =
        hh[(Composed.layOf (zkTab x j).table (stAt x j)).fa]! *
          hh[(Composed.layOf (zkTab x j).table (stAt x j)).fb]! ∧
      hh[(Composed.layOf (zkTab x j).table (stAt x j)).tw]! =
        hh[(Composed.layOf (zkTab x j).table (stAt x j)).tx]! *
          hh[(Composed.layOf (zkTab x j).table (stAt x j)).ty]! ∧
      cons.size = 7 ∧ (∀ q ∈ cons, q.2.a.size = (Composed.layOf (zkTab x j).table (stAt x j)).len) ∧
      (∀ k : Fin 7, (Composed.ofZk cons[k.val]!.2 :
          ZkSession.Lin FlockLevel3.GF128 (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nE + 4)) =
        Composed.rowsOf (zkTab x j).table (stAt x j) (czExe PE x c j) r
          (fun i => ((tabC PE x j).T (czExe PE x c j)).tab.onVals.vals r (wtOf PE x w j, v j)
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 i + fOf hh[i.val]!)
          (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
            ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
              (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1)
          (∑ col, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.y1' (wtOf PE x w j, v j)
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r col *
              Composed.bFold (zkTab x j).table (stAt x j) (czExe PE x c j) r col.val) (0, 0) (Fin.castAdd 3 k)) ∧
      _root_.Flock.Firewall.Finish.failingOn cons hh = #[] := by
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
  have hE : (Composed.layOf (zkTab x j).table (stAt x j)).hab =
      ((tabC PE x j).T (czExe PE x c j)).tab.onVals.nE := by
    show _ = Composed.padLenOf _ _ - 4
    simp only [Composed.padLenOf, Composed.layOf, _root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen := Composed.len_eq (zkTab x j).table (stAt x j) hE
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
  -- the masked values below `zp 64` are the clear values plus the pads
  have hzp : (Composed.layOf (zkTab x j).table (stAt x j)).zp 64 <
      (Composed.layOf (zkTab x j).table (stAt x j)).hab := by
    simp only [_root_.Flock.Zk.PadLayout.hab]
    omega
  have hmk : ∀ k < (Composed.layOf (zkTab x j).table (stAt x j)).zp 64,
      (Composed.msgAt (fun i => ((tabC PE x j).T (czExe PE x c j)).tab.onVals.vals r (wtOf PE x w j, v j)
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 i + fOf pads.h[i.val]!) k).val =
        vals[k]! + pads.h[k]! := by
    intro k hk
    have hkE : k < ((tabC PE x j).T (czExe PE x c j)).tab.onVals.nE := by omega
    have e := hM k hk
    unfold Composed.msgAt at e ⊢
    rw [dif_pos hkE] at e ⊢
    show (_ + fOf pads.h[k]!).val = _
    rw [FlockLevel3.val_add, e]
    rfl
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
      ss[0]! = Composed.shArr (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
          ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) (2 * r.val) ∧
        ss[1]! = Composed.shArr (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
          ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
            (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) (2 * r.val + 1) := by
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
    rw [Nat.add_zero, swViewE_sHat] at e0
    have e1 := e 1 (by norm_num)
    rw [swViewE_sHat] at e1
    exact ⟨e0, e1⟩
  -- level 0's record: its target is the ring switch's `T`, its coins the rep's
  have hrec : ∀ zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄,
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
      ((FlockComplete.ligeritoPZk (FlockComplete.asksCut (ligCoin c) n)
          (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash) _root_.Flock.MerkleScheme.hm96Sha512
          (zkTab x j).sch T cm0 (ligTab x w τ j) (FlockComplete.basisVals sws (st.m - 7))
          (_root_.Flock.Zk.PadLayout.new st.m st.kLog (zkTab x j).sch.k0) pads.h r.val tPad extra lanePad salt
          (fun _ _ => 0)).run t₃).go l₃ = (.ok ((lg, zl, rec), t₄), l₄) →
      fOf rec.target = Composed.openT (zkTab x j).table (stAt x j) (czExe PE x c j) r
          (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
            ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
              (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1) ∧
        LaneCoins (zkTab x j).table (stAt x j) (czExe PE x c j) r rec := by
    intro zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ lg zl rec t₄ l₄ h₁ h₂ h₃ h₄
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
    have hS4 := FlockComplete.SimInner.ligeritoPZk_ok (ligCoin c) n (respE c) (coinOf_respE c) _ _ _ _ _ _ _ _ _ _ _ _
      _ _ _ t₃ l₃ _ _ h₄
    have hzq : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m =
        .ok ((zkTab x j).sch, (zkTab x j).tPad) := by
      rw [stsOf_half]
      exact (zkTab x j).hz
    have hslot : ∀ f, repSlot x (2 * j.val + r.val) f = l₃.size + f (zkTab x j).sch := by
      intro f
      rw [repSlot_eq hs hzq, stsOf_half, ← repBase_eq hs (2 * j.val + r.val), ← hl]
      unfold FlockComplete.piopRoundsZk
      omega
    have hLF := FlockComplete.L0.ligeritoPZk_lanes (respE c) _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ (by omega) hk0 hS4
    have hrc := hLF.record
    refine ⟨?_, fun i => ?_, fun i hi => ?_, ?_⟩
    · rw [congrArg _root_.Flock.Zk.L0Rec.target hrc]
      exact hT zp zc t₁ l₁ lp lc t₂ l₂ ss T sws t₃ l₃ h₁ h₂ h₃
    · rw [hrc]
      show fOf (FlockComplete.coinOf (respE c (l₃.size + 1)) 0) =
        fOf (slotAt c (repSlot x (2 * j.val + r.val) (fun _ => 1)) 0)
      rw [coinOf_respE, hslot]
    · rw [hrc]
      show (FlockComplete.ZkTables.Level0.laneCh (respE c) (zkTab x j).sch.k0 l₃.size).toArray[i]! =
        kAt c (repSlot x (2 * j.val + r.val) fun _ => 2 + i)
      unfold FlockComplete.ZkTables.Level0.laneCh
      rw [getElem!_pos _ i (by simp only [List.size_toArray, List.length_map, List.length_range]; exact hi)]
      simp only [List.getElem_toArray, List.getElem_map, List.getElem_range, FlockComplete.coinKOf, coinOf_respE, kAt,
        hslot, Nat.add_assoc]
    · rw [hrc]
      show FlockComplete.ZkTables.Level0.laneRho (respE c) (zkTab x j).sch.k0 l₃.size = rhoAt c j.val r.val
      unfold FlockComplete.ZkTables.Level0.laneRho rhoAt rhoSlot
      simp only [FlockComplete.coinKOf, coinOf_respE, hslot]
      rw [show l₃.size + 2 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) by omega,
        show l₃.size + 3 + (zkTab x j).sch.k0 = l₃.size + ((zkTab x j).sch.k0 + 2) + 1 by omega]
  -- the run's seven constraints
  obtain ⟨rp, ss, lgc, hf, hc7, hasz, hform⟩ := Composed.tauFw_vals (zkTab x j).table (stAt x j) (czExe PE x c j) r
    (fun i => ((tabC PE x j).T (czExe PE x c j)).tab.onVals.vals r (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 i + fOf pads.h[i.val]!)
    (((tabC PE x j).T (czExe PE x c j)).tab.onVals.nu (wtOf PE x w j, v j) +
      ((tabC PE x j).T (czExe PE x c j)).tab.onVals.maskHom
        (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1)
    (∑ col, ((tabC PE x j).T (czExe PE x c j)).tab.onVals.y1' (wtOf PE x w j, v j)
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).1
      (((tabC PE x j).T (czExe PE x c j)).randOf ((UE.tape x τ).1 j) 0).2.1.2.1 r col *
        Composed.bFold (zkTab x j).table (stAt x j) (czExe PE x c j) r col.val) (0, 0)
    (ligCoin c) n st (zkTab x j).sch tPad r.val dom (ligTab x w τ j) cm0 xs pads ic vals extra lanePad salt ht₀
    hm14 hk6 hSm hSk hkL hlay hmk ez er eρ eα eβ eτ er'
    (fun combv hcomb => by
      subst hst
      have hkm' := (zkTab x j).table.S.hk
      have hS : (zkTab x j).table.S = FlockSoundness.Refine.stmtOf (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)) := rfl
      refine folded_unique (zkTab x j).table.S _ (Composed.modelFold (zkTab x j).table.S)
        (hS ▸ FlockSoundness.Refine.stmtOf_fold (stAt x j) (zkTab x j).wf.1
          (FlockSoundness.Refine.regionsOf (stAt x j)))
        (Composed.modelFold_realizes _) _ _ _ _ _ _ _ (FlockSoundness.Refine.skipWeights_size _) ?_ hcomb rfl
      simp only [Composed.rhoF, Array.size_extract, Array.size_ofFn]
      show min ((zkTab x j).table.S.kLog - 6) ((zkTab x j).table.S.m - 6) - 0 = (zkTab x j).table.S.kLog - 6
      omega)
    hsw hlen hrec h hV hTP
  refine ⟨pads.h, _, hps, hab, htw, hc7, fun q hq => ?_, fun k => hform k.val k.isLt, hf⟩
  obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hq
  have e := hasz i (by rw [← hc7]; exact hi)
  rwa [getElem!_pos _ i hi] at e

end Masked
end Claims.ZKMal
