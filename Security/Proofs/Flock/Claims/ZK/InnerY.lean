import Proofs.Flock.Claims.ZK.SplitInner
import Proofs.Flock.Claims.ZK.SentE

/-!
# The cut's `Y` is the session's (`Masked.InnerY`)

Where the shared phases reach table `j`'s rep `r`'s inner proof and `β`'s round (the slot's `+ 2`) is answered, the cut's
source sends `Y` (`repSrcCut`'s `y`) as the session's `InZ.y` at the honest pads:

* the cut sent rep `r`'s `Y` (`famSent`'s `fin r`): no refusal is seen by round `n`, past `β`'s round, so every check up
  to rep `r`'s inner proof passes (`famPasses_of`);
* the cut's `Y₁`, `Y₂` are the view's, `h + β·μ` and `pad_h + β·pad_μ` at the cell's base, with `β` the coin at
  `β`'s round (`CoinReads`);
* in `yAt`'s order (the exposed pads, then `PadIdx.inr ![2, 0, 1, 3]`, then the padding) the cell's base is the tape's
  `h ‖ pad_h` and `μ ‖ pad_μ` entry by entry (`pads_exe`, as `SimCommit`'s `exeOf_pads` reads the pads columns), so
  `Y` is `InZ.y`'s array, entry `i` being `(h ‖ pad_h)[i] + β·(μ ‖ pad_μ)[i]`.
-/

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge
open CoinLayer

local notation "𝔽" => FlockLevel3.GF128

/-! ## `yAt`'s order against the arrays' -/

private theorem get_append_right' (a b : Array _root_.Flock.F128) {k : ℕ} (hk : k < b.size) :
    (a ++ b)[a.size + k]! = b[k]! := by
  rw [getElem!_pos (a ++ b) _ (by simp; omega), getElem!_pos b k hk, Array.getElem_append_right (by omega)]
  simp

private theorem padIdx_lt' {nE : ℕ} (p : ZK.PadIdx nE) : Composed.padIdx p < nE + 4 := by
  rcases p with k | j
  · show k.val < nE + 4
    omega
  · show nE + (![1, 2, 0, 3] : Fin 4 → ℕ) j < nE + 4
    fin_cases j <;> simp

/-- `yAt`'s pad at entry `n_E + k` sits at layout index `n_E + k`. -/
private theorem padIdx_yAt {nE : ℕ} (k : Fin 4) :
    Composed.padIdx (nE := nE) (.inr ((![2, 0, 1, 3] : Fin 4 → Fin 4) k)) = nE + k.val := by
  fin_cases k <;> rfl

private theorem padIdx_yAt_mk {nE : ℕ} (m : ℕ) (hm : m < 4) :
    Composed.padIdx (nE := nE) (.inr ((![2, 0, 1, 3] : Fin 4 → Fin 4) ⟨m, hm⟩)) = nE + m :=
  padIdx_yAt ⟨m, hm⟩

/-- **`yAt` is an array's `a + β·m`** where pad `i` is entry `padIdx i` of `a + β·m` and padding symbol `k` entry
`n_E + 4 + k`. -/
private theorem yAt_eq {nE nPad : ℕ} (Y1 : ZK.PadIdx nE → 𝔽) (Y2 : Fin nPad → 𝔽) (a m : Array _root_.Flock.F128)
    (β : _root_.Flock.F128) (ha : a.size = nE + 4 + nPad)
    (h1 : ∀ i, Y1 i = fOf a[Composed.padIdx i]! + fOf β * fOf m[Composed.padIdx i]!)
    (h2 : ∀ k : Fin nPad, Y2 k = fOf a[nE + 4 + k.val]! + fOf β * fOf m[nE + 4 + k.val]!) :
    Array.ofFn (yAt Y1 Y2) = (Array.range a.size).map fun i => a[i]! + β * m[i]! := by
  refine Array.ext (by simp [ha]) fun i hi _ => ?_
  rw [Array.getElem_ofFn, Array.getElem_map, Array.getElem_range]
  unfold yAt
  dsimp only
  split_ifs with h h'
  · rw [h1]
    show FlockLevel3.GF128.val (fOf a[i]! + fOf β * fOf m[i]!) = _
    rw [← fOf_mul, ← fOf_add]
    rfl
  · rw [h1, padIdx_yAt_mk, Nat.add_sub_cancel' (by omega), ← fOf_mul, ← fOf_add]
    rfl
  · rw [h2, Fin.val_mk, Nat.add_sub_cancel' (by omega), ← fOf_mul, ← fOf_add]
    rfl

/-! ## The tape's pads at the cell's base -/

/-- **The tape's pads are the cell's base, entry by entry**: at the executable's layout, `h ‖ pad_h` at pad `i`'s layout
index is the table's pad `i` (`hFull`, with `h_ab = h_a·h_b` and `w = x·y`), `μ ‖ pad_μ` there is the base's `μ`, and
entry `n_E + 4 + k` of each is padding symbol `k`. -/
private theorem pads_exe (P : Composed.LigProver) {x : Msg} (t : Fin Claims.Flock.zkTableCap → ExeCell)
    (j : Fin (jOf x)) (r : Fin 2) (cz : Composed.CoinsZK (tabC P x j).Rest) {L : _root_.Flock.Zk.PadLayout}
    (hLy : L = _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)
    (pd : FlockComplete.PadsZk)
    (hpd : pd = FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val L) :
    (pd.h ++ pd.padH).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad ∧
      (∀ i, fOf (pd.h ++ pd.padH)[Composed.padIdx i]! =
        ((tabC P x j).T cz).tab.hFull ((((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j).2.2.1 r) i) ∧
      (∀ i, fOf (pd.mu ++ pd.padMu)[Composed.padIdx i]! =
        (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j).2.2.2.2.2.1 r i) ∧
      (∀ k : Fin (tabC P x j).z.nPad, fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 4 + k.val]! =
        (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j).2.2.2.1 r k) ∧
      ∀ k : Fin (tabC P x j).z.nPad, fOf (pd.mu ++ pd.padMu)[(tabC P x j).z.nE + 4 + k.val]! =
        (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j).2.2.2.2.1 r k := by
  subst hLy hpd
  set T := (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t with hT
  set L := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  set B := ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j with hB
  have hz := fits_zkTab P (zkTab x j)
  have hE64 : (tabC P x j).z.nE ≤ 2 ^ 64 := hz.nE
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE : (tabC P x j).z.nE = L.hab := by
    show L.len - 4 = L.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : L.len = (tabC P x j).z.nE + 4 := by
    rw [hE]
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  have hfa : L.fa < L.fb := by simp only [_root_.Flock.Zk.PadLayout.fb]; omega
  have hfb : L.fb < L.hab := Composed.ZerocheckMasked.fb_lt_hab L
  have htx : L.tx = L.hab + 1 := rfl
  have hty : L.ty = L.hab + 2 := rfl
  have htw : L.tw = L.hab + 3 := rfl
  have hhs := FlockComplete.padsOf_size T j.val r.val L
  have hpH : (FlockComplete.padsOf T j.val r.val L).padH.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have hms : (FlockComplete.padsOf T j.val r.val L).mu.size = L.len := by simp [FlockComplete.padsOf]
  have hpM : (FlockComplete.padsOf T j.val r.val L).padMu.size = _root_.Flock.Zk.K_PAD := by
    simp [FlockComplete.padsOf]
  have tH : ∀ k : Fin (2 ^ 64 + 4), fOf (T.h j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tM : ∀ k : Fin (2 ^ 64 + 4), fOf (T.mu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.2.1 r k := fun k => by
    show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.2.1 r) k.val) r.val) = _
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPH : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padH j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  have tPM : ∀ k : Fin _root_.Flock.Zk.K_PAD, fOf (T.padMu j.val r.val k.val) = (t ⟨j.val, hc⟩).1.2.2.2.2.1 r k :=
    fun k => by
      show fOf (fAt 0 (fun r => fAt 0 ((cellAt t j.val).1.2.2.2.2.1 r) k.val) r.val) = _
      rw [cellAt_lt t hc, fAt_lt _ _ r.isLt]; exact fAt_lt _ _ k.isLt
  set pd := FlockComplete.padsOf T j.val r.val L with hpd
  have hsz : (pd.h ++ pd.padH).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hhs, hpH, hlen]; rfl
  have hget0 : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → pd.h[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => FlockComplete.padsOf_get T _ _ L k hk h1 h2
  have hget : ∀ k, k < L.len → k ≠ L.hab → k ≠ L.tw → (pd.h ++ pd.padH)[k]! = T.h j.val r.val k :=
    fun k hk h1 h2 => by rw [FlockComplete.get_append_left _ _ (by rw [hhs]; exact hk), hget0 k hk h1 h2]
  have hfaE : L.fa < (tabC P x j).z.nE := by omega
  have hfbE : L.fb < (tabC P x j).z.nE := by omega
  have hEv : ∀ b : Fin 2, (hEmb hz (.inr b)).val = L.hab + 1 + b.val := fun b => by
    show (tabC P x j).z.nE + 1 + b.val = _
    rw [hE]
  have hf : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.h ++ pd.padH)[Composed.padIdx i]! =
      ((tabC P x j).T cz).tab.hFull (B.2.2.1 r) i := by
    rintro (k | i)
    · have hk := k.isLt
      show fOf (pd.h ++ pd.padH)[k.val]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl k))
      rw [hget k.val (by omega) (by omega) (by omega)]
      exact tH (hEmb hz (.inl k))
    · fin_cases i
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 1]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0))
        rw [hget ((tabC P x j).z.nE + 1) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 2]! = (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [hget ((tabC P x j).z.nE + 2) (by omega) (by omega) (by omega), ← tH, hEv, hE]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 0]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fa, hfaE⟩)) *
            (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inl ⟨L.fb, hfbE⟩))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 0 = L.hab by omega,
          FlockComplete.padsOf_hab, hget0 L.fa (by omega) (by omega) (by omega),
          hget0 L.fb (by omega) (by omega) (by omega), fOf_mul, tH ⟨L.fa, by omega⟩, tH ⟨L.fb, by omega⟩]
        rfl
      · show fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 3]! =
          (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 0)) * (t ⟨j.val, hc⟩).1.2.2.1 r (hEmb hz (.inr 1))
        rw [FlockComplete.get_append_left _ _ (by rw [hhs]; omega), show (tabC P x j).z.nE + 3 = L.tw by omega,
          FlockComplete.padsOf_tw, hget0 L.tx (by omega) (by omega) (by omega),
          hget0 L.ty (by omega) (by omega) (by omega), fOf_mul, ← tH, ← tH, hEv, hEv]
        rfl
  have hfm : ∀ i : ZK.PadIdx (tabC P x j).z.nE, fOf (pd.mu ++ pd.padMu)[Composed.padIdx i]! = B.2.2.2.2.2.1 r i := by
    intro i
    have hi := padIdx_lt' i
    rw [FlockComplete.get_append_left _ _ (by rw [hms]; omega),
      show pd.mu[Composed.padIdx i]! = T.mu j.val r.val (Composed.padIdx i) from
        Composed.Zerocheck.get_map_range _ (by omega)]
    exact tM (padEmb' hz i)
  have hg : ∀ k : Fin (tabC P x j).z.nPad, fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.h.size + k.val by rw [hhs, hlen],
      get_append_right' _ _ (by rw [hpH]; exact hk),
      show pd.padH[k.val]! = T.padH j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPH ⟨k.val, hk⟩
  have hgm : ∀ k : Fin (tabC P x j).z.nPad,
      fOf (pd.mu ++ pd.padMu)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.mu.size + k.val by rw [hms, hlen],
      get_append_right' _ _ (by rw [hpM]; exact hk),
      show pd.padMu[k.val]! = T.padMu j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPM ⟨k.val, hk⟩
  exact ⟨hsz, hf, hfm, hg, hgm⟩

/-! ## The cut's `Y` -/

/-- **The cut sent rep `r`'s `Y₁`** once `β`'s round is seen: `h + β·μ` at the cell's base, `β` the coin at its round. -/
private theorem cutE_Y1 {x : Msg} (hx : Live x) {c : Fin (refK x) → FlockDeploy.Coin} {n : ℕ} (hn : ¬ Refused x c n)
    {s spec} (hs : FlockDeploy.setupOf true x = some (s, spec, stsOf x)) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) (h : betaSlot x j.val r.val + 1 ≤ n) :
    (cutE x c w τ j).Y1 r = some fun i =>
      ((tabC PE x j).T (czExe PE x c j)).tab.hFull (((UE.tape x τ).1 j).2.2.1 r) i +
        fOf (betaAt c j.val r.val) * ((UE.tape x τ).1 j).2.2.2.2.2.1 r i := by
  have hpass := famPasses_of (pointReads_czExe PE) (coinReads_czExe PE) hx hn hs j r .inner h
  rw [← ((coinReads_czExe PE) x c j r).1]
  exact ite_eq_left hpass

/-- **The cut's `Y₂`**: `pad_h + β·pad_μ` at the cell's base. -/
private theorem cutE_Y2 {x : Msg} (c : Fin (refK x) → FlockDeploy.Coin) (w : Claims.Flock.Wit) (τ : UE.Tape)
    (j : Fin (jOf x)) (r : Fin 2) :
    (cutE x c w τ j).Y2 r = fun k =>
      ((UE.tape x τ).1 j).2.2.2.1 r k + fOf (betaAt c j.val r.val) * ((UE.tape x τ).1 j).2.2.2.2.1 r k := by
  rw [← ((coinReads_czExe PE) x c j r).1]
  rfl

/-- **`InnerY`**: once `β`'s round is answered, the cut's `Y` is `InZ.y` at the honest pads. -/
theorem innerY : InnerY := by
  intro x w τ c n j r cons L hA hlt _
  obtain ⟨hx, -, hn, s, spec, hs, -⟩ := hA
  have hj := lt_of_jOf j
  obtain ⟨y⟩ := hx.2 j.val hj
  have hst : (stsOf x)[(2 * j.val + r.val) / 2]! = (stsOf x)[j.val] := by
    rw [show (2 * j.val + r.val) / 2 = j.val by omega]
    exact getElem!_pos (stsOf x) j.val hj
  have hz : _root_.Flock.Zk.schedule (stsOf x)[(2 * j.val + r.val) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]
    exact y.hz
  have hβs : betaSlot x j.val r.val = innerSlotE x j r + 2 := by
    show repSlot x _ _ = repSlot x _ _ + 2
    rw [repSlot_eq hs hz, repSlot_eq hs hz]
    simp only [innerOff]
    omega
  have hjt : j.val < (FlockDeploy.tabsOf (stsOf x)).size := by simpa [FlockDeploy.tabsOf] using hj
  have hget : (FlockDeploy.tabsOf (stsOf x))[j.val]! = _root_.Flock.Setup.ofCircuit (stAt x j) := by
    rw [getElem!_pos (FlockDeploy.tabsOf (stsOf x)) j.val hjt]
    simp only [FlockDeploy.tabsOf, Array.getElem_map]
    rfl
  have hm : (FlockDeploy.tabsOf (stsOf x))[j.val]!.m = (stAt x j).m := by rw [hget]; rfl
  have hk : (FlockDeploy.tabsOf (stsOf x))[j.val]!.kLog = (stAt x j).c.kLog := by rw [hget]; rfl
  have hLy : innerLayE x j = _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 := by
    show _root_.Flock.Zk.PadLayout.new _ _ _ = _
    rw [hm, hk]
  obtain ⟨hsz, hf, hfm, hg, hgm⟩ := pads_exe PE τ j r (czExe PE x c j) hLy (innerPadsE x τ j r) rfl
  have hrep : innerSrcRepE x c w τ j r = repSrcCut PE x (czExe PE x c) (cutE x c w τ) j r := by
    show (srcE x c w τ j.val).rep r.val = _
    unfold srcE tabSrcCut
    rw [dite_eq_left j.isLt]
    exact dite_eq_left r.isLt
  have hβ : betaAt c j.val r.val = (innerCoinsE c j r).cb 0 := by
    show slotAt c (betaSlot x j.val r.val) 0 = slotAt c (innerSlotE x j r + 2) 0
    rw [hβs]
  rw [hrep]
  unfold repSrcCut
  dsimp only
  rw [cutE_Y1 hx hn hs w τ j r (by omega), cutE_Y2 c w τ j r, hβ]
  dsimp only
  exact yAt_eq _ _ _ _ _ hsz
    (fun i => (congrArg₂ (· + fOf ((innerCoinsE c j r).cb 0) * ·) (hf i) (hfm i)).symm)
    (fun k => (congrArg₂ (· + fOf ((innerCoinsE c j r).cb 0) * ·) (hg k) (hgm k)).symm)

end Claims.ZKMal.Masked
