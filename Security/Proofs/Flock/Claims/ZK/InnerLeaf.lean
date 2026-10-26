import Proofs.Flock.Claims.ZK.SplitInner

/-!
# The triple's openings and `τ`'s leaf on the cut

Two of `SplitInner`'s statements, each read off the tape at the executable's layout:

* `innerTriple`: the cut's `ρ_in`, `σ_in` (`Table.innerMsg` of the base's pads) are `ε·h_fa + h_tx` and `h_fb + h_ty` at
  the honest pads (`padsOf`), with `ε` the coin `czExe` reads `ζ` from.
* `innerTauLeaf`: the cut's `τ` commitment is the real view's (`cutV_viewsAt`), `leafT` of its `τ` and `y_τ`, and
  `y_τ`'s bytes are the honest pads' `τ` salt (`ytau_saltBytes`).
-/

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

/-- **`y_τ`'s bytes are the honest pads' `τ` salt**: table `j`'s rep `r` real view reads `y_τ` from the tape's cell `j`,
and `padsOf` fits the same bytes to `TAU_SALT_BYTES`. -/
theorem ytau_saltBytes (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    ZkView.saltBytes (((session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j).ytau r) =
      (innerPadsE x τ j r).tauSalt := by
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  show ZkView.saltBytes ((τ ⟨j.val, hc⟩).2.1.2.2 r) =
    FlockComplete.fitBytes 192 (fAt ByteArray.empty (fun r => ZkView.saltBytes ((cellAt τ j.val).2.1.2.2 r)) r.val)
  rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt, fitBytes_of_size (saltBytes_size _)]

/-- Table `j`'s rep `r` source on the cut is `repSrcCut` of the cut. -/
private theorem innerSrcRepE_eq (x : Msg) (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    innerSrcRepE x c w τ j r = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
  unfold innerSrcRepE srcE tabSrcCut
  rw [dite_eq_left j.isLt]
  exact dite_eq_left r.isLt

private theorem tcOf_sha512 (t : _root_.Flock.F128) (s : ByteArray) :
    FlockComplete.Replay.InZ.tcOf _root_.Flock.Sha512.hash t s = _root_.Flock.Zk.tauCommitment t s := rfl

/-- **`InnerTauLeaf`**: the cut's `τ` commitment is the real view's, `leafT` of its `τ` under `y_τ`, which is the
executable's `τ` commitment through SHA-512 under the honest pads' `τ` salt (`ytau_saltBytes`). -/
theorem innerTauLeaf : InnerTauLeaf := by
  intro x w τ c n j r cons L _
  have hV := (RefSession.cutV_viewsAt (session (fun _ _ => 0) (tableData PE QE UE)) x c (wtOf PE x w) τ).trans
    (cutOf_eq (fun _ _ => 0) x c w τ)
  rw [innerSrcRepE_eq, ← ytau_saltBytes x c w τ j r, tcOf_sha512]
  show (cutE x c w τ j).tauCom r = _
  rw [← hV]
  show ((session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j).tauCom r = _
  show ZkView.leafT (((session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j).tau r)
    (((session (fun _ _ => 0) (tableData PE QE UE)).viewsAt x c (wtOf PE x w) τ j).ytau r) = _
  rfl

open FlockSoundness.Refine in
/-- **`InnerTriple`**: the cut's `ρ_in`, `σ_in` are `Table.innerMsg` of the base's pads, which the tape's cell `j`
holds at the layout's `fa`, `fb`, `tx`, `ty` (`readH`); there the honest pads (`padsOf`) are the tape's, and `ζ` is the
coin at the rep's `ε` slot (`restAt`'s `zeta`). -/
theorem innerTriple : InnerTriple := by
  intro x w τ c n j r cons L₀ _ _
  have hj := lt_of_jOf j
  have hjt : j.val < (FlockDeploy.tabsOf (stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have hget : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val hjt]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hm : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stAt x j).m := by rw [hget]; rfl
  have hk : (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog = (stAt x j).c.kLog := by rw [hget]; rfl
  have hLay : innerLayE x j = _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 := by
    show _root_.Flock.Zk.PadLayout.new (FlockDeploy.tabsOf (stsOf x))[j.val]!.m
      (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog (zkTab x j).sch.k0 = _
    rw [hm, hk]
  rw [innerSrcRepE_eq]
  show FlockLevel3.GF128.val ((cutE x c w τ j).inner r).1 =
      FlockComplete.Replay.InZ.rho (innerLayE x j) (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (innerLayE x j))
        (innerCoinsE c j r) ∧
    FlockLevel3.GF128.val ((cutE x c w τ j).inner r).2 =
      FlockComplete.Replay.InZ.sigma (innerLayE x j) (FlockComplete.padsOf (UE.tapeZk τ) j.val r.val (innerLayE x j))
  rw [hLay]
  set L := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  set T := UE.tapeZk τ with hT
  set B := (UE.tape x τ).1 j with hB
  have hz := fits_zkTab PE (zkTab x j)
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE : (tabC PE x j).z.nE = L.hab := by
    show L.len - 4 = L.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hfa : L.fa < L.fb := by simp only [_root_.Flock.Zk.PadLayout.fb]; omega
  have hfb : L.fb < L.hab := Composed.ZerocheckMasked.fb_lt_hab L
  have htx : L.tx = L.hab + 1 := rfl
  have hty : L.ty = L.hab + 2 := rfl
  have htw : L.tw = L.hab + 3 := rfl
  have hlen : L.len = L.hab + 4 := rfl
  have hget0 : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw →
      (FlockComplete.padsOf T j.val r.val L).h[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => FlockComplete.padsOf_get T _ _ L k hk h1 h2
  have tH : ∀ k : Fin (2 ^ 64 + 4), fOf (T.h j.val r.val k.val) = (τ ⟨j.val, hc⟩).1.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt τ j.val).1.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt τ hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have hEv : ∀ b : Fin 2, (hEmb hz (.inr b)).val = L.hab + 1 + b.val := fun b => by
    show (tabC PE x j).z.nE + 1 + b.val = _
    rw [hE]
  have hE64 : (tabC PE x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hfaE : L.fa < (tabC PE x j).z.nE := by omega
  have hfbE : L.fb < (tabC PE x j).z.nE := by omega
  have hA : (B.2.2.1 r).1 ⟨L.fa, hfaE⟩ = fOf (FlockComplete.padsOf T j.val r.val L).h[L.fa]! := by
    rw [hget0 L.fa (by omega) (by omega) (by omega), tH ⟨L.fa, by omega⟩]
    rfl
  have hB' : (B.2.2.1 r).1 ⟨L.fb, hfbE⟩ = fOf (FlockComplete.padsOf T j.val r.val L).h[L.fb]! := by
    rw [hget0 L.fb (by omega) (by omega) (by omega), tH ⟨L.fb, by omega⟩]
    rfl
  have hX : (B.2.2.1 r).2.1 = fOf (FlockComplete.padsOf T j.val r.val L).h[L.tx]! := by
    rw [hget0 L.tx (by omega) (by omega) (by omega), show L.tx = (hEmb hz (.inr 0)).val by rw [hEv]; rfl, tH]
    rfl
  have hY : (B.2.2.1 r).2.2 = fOf (FlockComplete.padsOf T j.val r.val L).h[L.ty]! := by
    rw [hget0 L.ty (by omega) (by omega) (by omega), show L.ty = (hEmb hz (.inr 1)).val by rw [hEv]; rfl, tH]
    rfl
  have hin : (cutE x c w τ j).inner r =
      ((tabC PE x j).T (czExe PE x c j)).tab.innerMsg r (B.2.2.1 r) := rfl
  rw [hin]
  refine ⟨fOf_injective ?_, fOf_injective ?_⟩
  · show fOf (slotAt c (innerSlotE x j r) 0) * (B.2.2.1 r).1 ⟨L.fa, hfaE⟩ + (B.2.2.1 r).2.1 =
      fOf ((innerCoinsE c j r).ce 0 * (FlockComplete.padsOf T j.val r.val L).h[L.fa]! +
        (FlockComplete.padsOf T j.val r.val L).h[L.tx]!)
    rw [fOf_add, fOf_mul, hA, hX]
    rfl
  · show (B.2.2.1 r).1 ⟨L.fb, hfbE⟩ + (B.2.2.1 r).2.2 =
      fOf ((FlockComplete.padsOf T j.val r.val L).h[L.fb]! + (FlockComplete.padsOf T j.val r.val L).h[L.ty]!)
    rw [fOf_add, hB', hY]

end Masked

end Claims.ZKMal
