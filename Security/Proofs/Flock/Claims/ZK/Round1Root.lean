import Proofs.Flock.Claims.ZK.Round1Pads
import Proofs.Flock.Claims.ZK.ExeLayout

/-!
# Round `1` of the masked session: the pads root at the executable's layout

The masked session's second body binds table `0`'s rep `0` pads root (`Masked.liveRun_round1`), `innerCommitP`'s root
of the tape's pads. Here that root is the model's:

* `FlockComplete.innerCommitP_root_layerOf`: `innerCommitP`'s root is `layerOf`'s, of the leaves
  `H(H(c_h(p) ‖ c_μ(p)), salt p)`, where `c_h`, `c_μ` are the codes of `h ‖ pad_h` and `μ ‖ pad_μ` at `X = p + 1`
  (`Zk.codeAt`).
* `Composed.sum_padCode`: `padColAt`'s sums, each pad at its layout index (`padIdx`, a bijection onto the layout's
  `n_E + 4` positions: `Composed.padIdxEquiv`) and padding symbol `j` at `len + j`, are the code's sum over the array.
* `Masked.exeOf_pads`: so at the executable's layout (`TapeLayout.exe`), the executable's pads leaf of the cut's view at
  tree position `p` is `innerCommitP`'s leaf `p` of the tape's pads, under the tape's pads salt `p`, at every coin
  vector: the tape's pads are the cell's, with `h_ab = h_a·h_b` and `w = x·y` (`FlockComplete.padsOf_hab`,
  `FlockComplete.padsOf_tw`) as `Composed.padH` puts them.
* `Masked.innerRoot_exe`: the root of the tape's pads is `layerOf`'s of the cut's pads leaves by tree position
  (`Masked.leafAtP`).
-/

namespace FlockComplete

open Flock FlockDeploy

/-- Entry `|a| + k` of `a ++ b` is `b`'s entry `k`. -/
theorem get_append_right (a b : Array F128) {k : ℕ} (hk : k < b.size) : (a ++ b)[a.size + k]! = b[k]! := by
  rw [getElem!_pos (a ++ b) _ (by simp; omega), getElem!_pos b k hk, Array.getElem_append_right (by omega)]
  simp

/-- `layerOf`'s level `h ≤ d` has `2^(d-h)` nodes. -/
theorem layerOf_size (nd : ByteArray → ByteArray → ByteArray) (d : ℕ) (lf : ℕ → ByteArray) :
    ∀ h ≤ d, (layerOf nd d lf h).size = 2 ^ (d - h)
  | 0, _ => by simp [layerOf]
  | h + 1, hh => by
    have ih := layerOf_size nd d lf h (by omega)
    simp only [layerOf, Array.size_map, Array.size_range]
    rw [ih, show d - h = d - (h + 1) + 1 by omega, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]

/-- **A pads tree of `n = 2^k` leaves is `layerOf`'s** of its leaves, level by level. -/
theorem innerTree_layerOf {n : ℕ} {ic : InnerCommit} (hT : InnerTree (Hashes.ofSha512 Sha512.hash) n ic)
    (lf : ℕ → ByteArray) (hlf : ∀ j < n, (ic.layers[0]!)[j]! = lf j) (hn : n = 2 ^ Nat.log2 n) :
    ∀ h ≤ Nat.log2 n, ∀ i < 2 ^ (Nat.log2 n - h),
      (ic.layers[h]!)[i]! = (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[i]!
  | 0, _, i, hi => by
    rw [hlf i (by rw [hn]; simpa using hi)]
    simp only [layerOf]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp
  | h + 1, hh, i, hi => by
    have hi' : i < 2 ^ (Nat.log2 n - h - 1) := by
      rwa [show Nat.log2 n - h - 1 = Nat.log2 n - (h + 1) by omega]
    have hnode := hT.node h (by omega) i hi'
    have hs := layerOf_size (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h (by omega)
    have hd : 2 ^ (Nat.log2 n - h) = 2 * 2 ^ (Nat.log2 n - (h + 1)) := by
      rw [show Nat.log2 n - h = Nat.log2 n - (h + 1) + 1 by omega, Nat.pow_succ, Nat.mul_comm]
    rw [innerTree_layerOf hT lf hlf hn h (by omega) (2 * i) (by omega),
      innerTree_layerOf hT lf hlf hn h (by omega) (2 * i + 1) (by omega)] at hnode
    have e2 : (MerkleOps.hm96With (Hashes.ofSha512 Sha512.hash : Hashes V)).node
        (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i]!
        (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i + 1]! =
      .ok (Sha512.hash ((layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i]! ++
        (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i + 1]!)) := rfl
    rw [e2, Except.ok.injEq] at hnode
    rw [← hnode]
    have hk : i < (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h).size / 2 := by
      rw [hs, hd, Nat.mul_div_cancel_left _ (by norm_num)]
      exact hi
    show _ = ((Array.range ((layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h).size / 2)).map fun i =>
      Sha512.hash ((layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i]! ++
        (layerOf (fun l r => Sha512.hash (l ++ r)) (Nat.log2 n) lf h)[2 * i + 1]!))[i]!
    rw [getElem!_pos _ i (by simpa using hk)]
    simp

/-- `innerCommitP`'s salts are the tape's, each fitted to 192 bytes. -/
theorem innerCommitP_salts {hs : Hashes V} {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : innerCommitP hs p salt = .ok ic) :
    ic.salts = (Array.range ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)).map
      fun j => fitBytes 192 (salt j) := by
  unfold innerCommitP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

/-- **`innerCommitP`'s root is `layerOf`'s**, of the leaves `H(H(c_h(j) ‖ c_μ(j)), salt j)`. -/
theorem innerCommitP_root_layerOf {p : PadsZk} {salt : ℕ → ByteArray} {ic : InnerCommit}
    (h : (innerCommitP (Hashes.ofSha512 Sha512.hash) p salt : V InnerCommit) = .ok ic) :
    ic.root = (layerOf (fun l r => Sha512.hash (l ++ r))
      (Nat.log2 ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE))
      (fun j => Hm96.Default512.leaf (Sha512.hash ((Zk.codeAt (p.h ++ p.padH) j).toBytes ++
        (Zk.codeAt (p.mu ++ p.padMu) j).toBytes)) (fitBytes 192 (salt j)))
      (Nat.log2 ((p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE)))[0]! := by
  have hT := innerCommitP_tree h
  obtain ⟨hch, hcm⟩ := Replay.innerCommitP_rows h
  have hsl := innerCommitP_salts h
  set n := (p.h ++ p.padH).size.nextPowerOfTwo * 2 ^ Zk.INNER_LOG_INV_RATE with hN
  have hpow : n = 2 ^ Nat.log2 n := innerCols_pow _
  rw [hT.root]
  refine innerTree_layerOf hT _ (fun j hj => ?_) hpow (Nat.log2 n) le_rfl 0 (by simp)
  have e := hT.leaf j hj
  change Except.ok (Hm96.Default512.leaf (Sha512.hash (ic.ch[j]!.toBytes ++ ic.cm[j]!.toBytes)) ic.salts[j]!) = _ at e
  rw [Except.ok.injEq] at e
  rw [← e, hch, hcm, hsl, Replay.map_get _ _ (by simpa using hj), Replay.map_get _ _ (by simpa using hj),
    Replay.map_get _ _ (by simpa using hj)]
  rw [getElem!_pos (Array.range n) j (by simpa using hj), Array.getElem_range]

end FlockComplete

namespace FlockSoundness.Discharge.Composed

open Flock FlockSoundness.Refine

local notation "𝔽" => FlockLevel3.GF128

/-- A pad's layout index is below `n_E + 4`. -/
theorem padIdx_lt {nE : ℕ} (p : ZK.PadIdx nE) : padIdx p < nE + 4 := by
  rcases p with k | j
  · show k.val < nE + 4
    omega
  · show nE + (![1, 2, 0, 3] : Fin 4 → ℕ) j < nE + 4
    fin_cases j <;> simp

theorem padIdx_injective {nE : ℕ} : Function.Injective (padIdx (nE := nE)) := by
  rintro (a | a) (b | b) h <;> simp only [padIdx] at h
  · exact congrArg Sum.inl (Fin.ext h)
  · have := a.isLt; fin_cases b <;> simp at h <;> omega
  · have := b.isLt; fin_cases a <;> simp at h <;> omega
  · fin_cases a <;> fin_cases b <;> simp at h ⊢

/-- **The pads layout's indices**: pad `i` at `padIdx i`, a bijection onto the layout's `n_E + 4` positions. -/
noncomputable def padIdxEquiv (nE : ℕ) : ZK.PadIdx nE ≃ Fin (nE + 4) :=
  Equiv.ofBijective (fun p => ⟨padIdx p, padIdx_lt p⟩)
    ((Fintype.bijective_iff_injective_and_card _).2
      ⟨fun a b h => padIdx_injective (congrArg Fin.val h), by simp [ZK.PadIdx]⟩)

/-- **`padColAt`'s sums are a code's sum over its array** `y` of `n_E + 4 + K` symbols, where pad `i` is `y`'s entry
`padIdx i` and padding symbol `j` is entry `n_E + 4 + j`. -/
theorem sum_padCode {nE K padLen : ℕ} (hpl : padLen = nE + 4) (y : Array F128) (ξ : 𝔽) (f : ZK.PadIdx nE → 𝔽)
    (g : Fin K → 𝔽) (hf : ∀ i, fOf y[padIdx i]! = f i) (hg : ∀ j : Fin K, fOf y[nE + 4 + j.val]! = g j) :
    ∑ i, ξ ^ padIdx i * f i + ∑ j : Fin K, ξ ^ (padLen + j.val) * g j =
      ∑ i : Fin (nE + 4 + K), fOf y[i.val]! * ξ ^ (i : ℕ) := by
  subst hpl
  rw [Fin.sum_univ_add]
  refine congrArg₂ (· + ·) ?_ (Finset.sum_congr rfl fun j _ => ?_)
  · exact Fintype.sum_equiv (padIdxEquiv nE) _ _ fun p => by
      show ξ ^ padIdx p * f p = fOf y[padIdx p]! * ξ ^ padIdx p
      rw [hf, mul_comm]
  · simp only [Fin.val_natAdd]
    rw [hg, mul_comm]

end FlockSoundness.Discharge.Composed

namespace Claims.ZKMal.Masked

open Claims FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge

/-- **Table `j`'s rep `r` pads leaf at tree position `p` is the masked prover's**: at the executable's layout, the
executable's leaf of the cut's view (`exeOf`) is `verifyInner`'s leaf of `innerCommitP`'s column `p`, the codes of
`h ‖ pad_h` and `μ ‖ pad_μ` at `X = p + 1`, under the tape's pads salt `p`, at every coin vector. -/
theorem exeOf_pads (P : Composed.LigProver) {x : Msg} (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → ExeCell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    (r : Fin 2) (p : Fin ((tabC P x j).z.nQ + (tabC P x j).z.nPU)) :
    exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
        ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩ =
      _root_.Flock.Hm96.Default512.leaf (_root_.Flock.Sha512.hash
        ((_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val
            r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).h ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padH) p.val).toBytes ++
        (_root_.Flock.Zk.codeAt ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val
            r.val (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).mu ++
          (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
            (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padMu) p.val).toBytes))
        (FlockComplete.fitBytes 192 (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).innerSalt j.val r.val
          p.val)) := by
  set T := (TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t with hT
  set L := _root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0 with hLy
  set B := ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1 j with hB
  have hz := fits_zkTab P (zkTab x j)
  have hc : j.val < Claims.Flock.zkTableCap := lt_of_lt_of_le j.isLt (jOf_le_cap x)
  have hE : (tabC P x j).z.nE = L.hab := by
    show L.len - 4 = L.hab
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
    omega
  have hlen : L.len = (tabC P x j).z.nE + 4 := by
    rw [hE]
    simp only [_root_.Flock.Zk.PadLayout.len, _root_.Flock.Zk.PadLayout.tw]
  have hE64 : (tabC P x j).z.nE ≤ 2 ^ 64 := hz.nE
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
  have hpl : Composed.padLenOf (zkTab x j).table (stAt x j) = (tabC P x j).z.nE + 4 := hlen
  set pd := FlockComplete.padsOf T j.val r.val L with hpd
  have hsz : (pd.h ++ pd.padH).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hhs, hpH, hlen]; rfl
  have hszm : (pd.mu ++ pd.padMu).size = (tabC P x j).z.nE + 4 + (tabC P x j).z.nPad := by
    rw [Array.size_append, hms, hpM, hlen]; rfl
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
      Composed.padH (nE := (tabC P x j).z.nE) ⟨L.fa, hfaE⟩ ⟨L.fb, hfbE⟩ (B.2.2.1 r) i := by
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
    have hi := Composed.padIdx_lt i
    rw [FlockComplete.get_append_left _ _ (by rw [hms]; omega),
      show pd.mu[Composed.padIdx i]! = T.mu j.val r.val (Composed.padIdx i) from Composed.Zerocheck.get_map_range _ (by omega)]
    exact tM (padEmb' hz i)
  have hg : ∀ k : Fin (tabC P x j).z.nPad, fOf (pd.h ++ pd.padH)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.h.size + k.val by rw [hhs, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpH]; exact hk),
      show pd.padH[k.val]! = T.padH j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPH ⟨k.val, hk⟩
  have hgm : ∀ k : Fin (tabC P x j).z.nPad,
      fOf (pd.mu ++ pd.padMu)[(tabC P x j).z.nE + 4 + k.val]! = B.2.2.2.2.1 r k := by
    intro k
    have hk := k.isLt
    rw [show (tabC P x j).z.nE + 4 + k.val = pd.mu.size + k.val by rw [hms, hlen],
      FlockComplete.get_append_right _ _ (by rw [hpM]; exact hk),
      show pd.padMu[k.val]! = T.padMu j.val r.val k.val from Composed.Zerocheck.get_map_range _ hk]
    exact tPM ⟨k.val, hk⟩
  have h1 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).1 =
      fOf (_root_.Flock.Zk.codeAt (pd.h ++ pd.padH) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hsz]
    exact Composed.sum_padCode hpl (pd.h ++ pd.padH) (ZkLink.innerPt p.val) _ _ hf hg
  have h2 : (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).2 =
      fOf (_root_.Flock.Zk.codeAt (pd.mu ++ pd.padMu) p.val) := by
    rw [colPAt_tabC, ZkExec.fOf_codeAt _ p.val _ hszm]
    exact Composed.sum_padCode hpl (pd.mu ++ pd.padMu) (ZkLink.innerPt p.val) _ _ hfm hgm
  have hp : p.val < _root_.Flock.Zk.Q + 2 ^ 64 := by
    have h1 := p.isLt
    have h2 : (tabC P x j).z.nQ ≤ _root_.Flock.Zk.Q := hz.nQ
    have h3 : (tabC P x j).z.nPU ≤ 2 ^ 64 := hz.nPU
    omega
  have hs : ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩) =
      FlockComplete.fitBytes 192 (T.innerSalt j.val r.val p.val) := by
    show ZkView.saltBytes ((t ⟨j.val, hc⟩).2.1.2.1 r (Fin.castLE (Nat.add_le_add hz.nQ hz.nPU) p)) =
      FlockComplete.fitBytes 192 (fAt ByteArray.empty (fun r => fAt ByteArray.empty
        (fun p => ZkView.saltBytes ((cellAt t j.val).2.1.2.1 r p)) p.val) r.val)
    rw [cellAt_lt t hc, fAt_lt _ _ r.isLt, fitBytes_of_size (by rw [fAt_lt _ _ hp]; exact saltBytes_size _),
      fAt_lt _ _ hp]
    rfl
  show _root_.Flock.Hm96.Default512.leaf (_root_.Flock.Sha512.hash
      (_root_.Flock.F128.toBytes (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).1 ++
        _root_.Flock.F128.toBytes (((tabC P x j).T (cz j)).colPAt (wtOf P x w j) B r p).2))
      (ZkView.saltBytes (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1 ⟨j, .inr (.inl (r, p))⟩)) = _
  rw [h1, h2, hs]
  rfl

/-- **Table `j`'s rep `r` pads root at the executable's layout is the model's**: where `innerCommitP` commits the tape's
pads, its root is `layerOf`'s of the cut's pads leaves by tree position (`leafAtP`), at every coin vector. -/
theorem innerRoot_exe (P : Composed.LigProver) {x : Msg} (w : Claims.Flock.Wit)
    (t : Fin Claims.Flock.zkTableCap → ExeCell) (cz : ∀ j, Composed.CoinsZK (tabC P x j).Rest) (j : Fin (jOf x))
    (r : Fin 2) {ic : FlockComplete.InnerCommit}
    (h : (FlockComplete.innerCommitP (_root_.Flock.Hashes.ofSha512 _root_.Flock.Sha512.hash)
        (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
          (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0))
        (((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t).innerSalt j.val r.val) :
          _root_.Flock.V FlockComplete.InnerCommit) = .ok ic) :
    ic.root = (FlockComplete.layerOf (fun l r => _root_.Flock.Sha512.hash (l ++ r))
        (Nat.log2 (Composed.padNOf (zkTab x j).table (stAt x j)))
        (leafAtP P x cz (cutAt (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
          (exeOf (tabC P x) cz (wtOf P x w) ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).1
            ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tape x t).2.1)) j r)
        (Nat.log2 (Composed.padNOf (zkTab x j).table (stAt x j))))[0]! := by
  have hn : ((FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
        (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).h ++
      (FlockComplete.padsOf ((TapePlan.ofLayout (TapeLayout.exe P) exeTape).tapeZk t) j.val r.val
        (_root_.Flock.Zk.PadLayout.new (stAt x j).m (stAt x j).c.kLog (zkTab x j).sch.k0)).padH).size.nextPowerOfTwo *
        2 ^ _root_.Flock.Zk.INNER_LOG_INV_RATE = Composed.padNOf (zkTab x j).table (stAt x j) := by
    rw [Array.size_append, FlockComplete.padsOf_size]
    simp only [FlockComplete.padsOf, Array.size_map, Array.size_range]
    rfl
  rw [FlockComplete.innerCommitP_root_layerOf h, hn]
  have hN := padN_bounds x j
  have hpow := FlockComplete.innerCols_pow (Composed.padLenOf (zkTab x j).table (stAt x j) + _root_.Flock.Zk.K_PAD)
  have hNq : (tabC P x j).z.nQ + (tabC P x j).z.nPU = Composed.padNOf (zkTab x j).table (stAt x j) := by
    have e1 : (tabC P x j).z.nQ = _root_.Flock.Zk.Q := rfl
    have e2 : (tabC P x j).z.nPU = Composed.padNOf (zkTab x j).table (stAt x j) - _root_.Flock.Zk.Q := rfl
    omega
  rw [FlockComplete.layerOf_congr _ _ (fun q hq => ?_)]
  have hq' : q < (tabC P x j).z.nQ + (tabC P x j).z.nPU := by
    rw [hNq]
    exact lt_of_lt_of_eq hq hpow.symm
  rw [leafAtP_cutAt P x cz _ _ _ j r hq']
  exact (exeOf_pads P w t cz j r ⟨q, hq'⟩).symm

end Claims.ZKMal.Masked
