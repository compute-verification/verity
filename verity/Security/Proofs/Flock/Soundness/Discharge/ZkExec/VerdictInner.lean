import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictStatements
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysDet
import Proofs.Flock.Soundness.Discharge.ZkExec.Inner
import Proofs.Flock.Soundness.Discharge.ZkExec.PlaysInner
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.Soundness.Refine.Salted
import Mathlib.Algebra.CharP.Two

/-!
# The inner proof's check holds on its openings (`ZkInnerAccepts`)

`zk_inner_accepts`: when `Zk.verifyInner` accepts, the compiled inner proof's one result (`plays_det`), at the coins of
the four rounds it read (`zk_inner_peel`), is a check that holds on the proof's inner openings:

* `τ` opens its commitment: `encTau`'s `hm96` leaf is `Zk.tauCommitment` (`leaf_tau`).
* The batched constraint: `Lin.combo`'s coefficients and constant are the `γ`-sums of the constraints' (`combo_a`,
  `combo_c`); the model's rows are the seven constraints (`LinRel`) and the sacrifice's three rows (`triple_rel`, at
  the pads layout's distinct positions); `batchT`'s sign is characteristic 2's.
* Each query: `codeAt` is `enc` at the point `j + 1` (`fOf_codeAt`), and the pair's path climbs to the pads' root
  under `encPair`'s `hm96` leaf (`leaf_pair`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-! ## Arrays -/

theorem set_get {α : Type} [Inhabited α] (a : Array α) {j : ℕ} (v : α) (hj : j < a.size) {i : ℕ} (hi : i < a.size) :
    (a.set! j v)[i]! = if i = j then v else a[i]! := by
  rw [Array.set!_eq_setIfInBounds, getElem!_pos _ i (by rw [Array.size_setIfInBounds]; exact hi),
    getElem!_pos a i hi, Array.getElem_setIfInBounds hi]
  by_cases h : i = j
  · subst h; simp
  · simp [h, Ne.symm h]

theorem repl_get (n : ℕ) {i : ℕ} (hi : i < n) : (Array.replicate n (0 : F128))[i]! = 0 := by
  rw [getElem!_pos _ i (by simpa using hi), Array.getElem_replicate]

theorem map_get! {α β : Type} [Inhabited α] [Inhabited β] (a : Array α) (f : α → β) {k : ℕ} (hk : k < a.size) :
    (a.map f)[k]! = f a[k]! := by
  rw [getElem!_pos _ k (by simpa using hk), Array.getElem_map, getElem!_pos a k hk]

theorem append_get!_left {α : Type} [Inhabited α] (a b : Array α) {k : ℕ} (hk : k < a.size) :
    (a ++ b)[k]! = a[k]! := by
  rw [getElem!_pos _ k (by simp; omega), Array.getElem_append_left hk, getElem!_pos a k hk]

theorem append_get!_right {α : Type} [Inhabited α] (a b : Array α) {k j : ℕ} (hk : k = a.size + j)
    (hj : j < b.size) : (a ++ b)[k]! = b[j]! := by
  subst hk
  rw [getElem!_pos _ _ (by simp; omega), Array.getElem_append_right (by omega), getElem!_pos b j hj]
  simp only [Nat.add_sub_cancel_left]

/-! ## `Lin.combo` -/

theorem setLoop_size (w : F128) (x : Array F128) : ∀ (n : ℕ) (a : Array F128),
    (List.foldl (fun b j => b.set! j (b[j]! + w * x[j]!)) a (List.range n)).size = a.size
  | 0, _ => rfl
  | n + 1, a => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Array.set!_eq_setIfInBounds,
        Array.size_setIfInBounds, setLoop_size w x n a]

theorem setLoop_get (w : F128) (x : Array F128) : ∀ (n : ℕ) (a : Array F128), n ≤ a.size → ∀ i < a.size,
    (List.foldl (fun b j => b.set! j (b[j]! + w * x[j]!)) a (List.range n))[i]! =
      if i < n then a[i]! + w * x[i]! else a[i]!
  | 0, _, _, _, _ => by simp
  | n + 1, a, hn, i, hi => by
      have hs := setLoop_size w x n a
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        set_get _ _ (by rw [hs]; omega) (by rw [hs]; exact hi), setLoop_get w x n a (by omega) i hi,
        setLoop_get w x n a (by omega) n (by omega)]
      by_cases hin : i = n
      · subst hin; simp
      · by_cases h : i < n
        · simp [hin, h, show i < n + 1 by omega]
        · simp [hin, h, show ¬ i < n + 1 by omega]

/-- One term of `Lin.combo`'s loop: the constant and the coefficients after adding `w_k·l_k`. -/
def comboStep (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) (b : F128 × Array F128) (k : ℕ) : F128 × Array F128 :=
  (b.1 + ws[k]! * ls[k]!.c,
    List.foldl (fun a j => a.set! j (a[j]! + ws[k]! * ls[k]!.a[j]!)) b.2 (List.range (min s ls[k]!.a.size)))

theorem combo_eq (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) :
    Zk.Lin.combo ws ls s = ⟨(List.foldl (comboStep ws ls s) (0, Array.replicate s 0) (List.range (min ws.size ls.size))).1,
      (List.foldl (comboStep ws ls s) (0, Array.replicate s 0) (List.range (min ws.size ls.size))).2⟩ := by
  simp only [Zk.Lin.combo, Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl,
    Std.Legacy.Range.size, pure_bind]
  simp only [Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range']
  rfl

theorem comboLoop_size (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) :
    ∀ (m : ℕ) (b : F128 × Array F128), (List.foldl (comboStep ws ls s) b (List.range m)).2.size = b.2.size
  | 0, _ => rfl
  | m + 1, b => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, comboStep, setLoop_size,
        comboLoop_size ws ls s m b]

theorem comboLoop_c (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) :
    ∀ (m : ℕ) (b : F128 × Array F128), fOf (List.foldl (comboStep ws ls s) b (List.range m)).1 =
      fOf b.1 + ∑ k : Fin m, fOf ws[k.val]! * fOf (ls[k.val]!).c
  | 0, b => by simp
  | m + 1, b => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, comboStep, fOf_add, fOf_mul,
        comboLoop_c ws ls s m b, Fin.sum_univ_castSucc]
      simp only [Fin.val_castSucc, Fin.val_last, add_assoc]

theorem comboLoop_a (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) :
    ∀ (m : ℕ) (b : F128 × Array F128), b.2.size = s → (∀ k < m, (ls[k]!).a.size = s) → ∀ i < s,
      fOf (List.foldl (comboStep ws ls s) b (List.range m)).2[i]! =
        fOf b.2[i]! + ∑ k : Fin m, fOf ws[k.val]! * fOf (ls[k.val]!).a[i]!
  | 0, b, _, _, i, _ => by simp
  | m + 1, b, hb, hls, i, hi => by
      have hsz := comboLoop_size ws ls s m b
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, comboStep, hls m (by omega),
        Nat.min_self, setLoop_get _ _ s _ (by rw [hsz, hb]) i (by rw [hsz, hb]; exact hi), ite_eq_left hi, fOf_add,
        fOf_mul, comboLoop_a ws ls s m b hb (fun k hk => hls k (by omega)) i hi, Fin.sum_univ_castSucc]
      simp only [Fin.val_castSucc, Fin.val_last, add_assoc]

theorem combo_size (ws : Array F128) (ls : Array Zk.Lin) (s : ℕ) : (Zk.Lin.combo ws ls s).a.size = s := by
  rw [combo_eq]
  exact (comboLoop_size ws ls s _ _).trans (Array.size_replicate ..)

/-- **`Lin.combo`'s constant** is the `w`-sum of the forms' constants. -/
theorem combo_c (ws : Array F128) (ls : Array Zk.Lin) (s m : ℕ) (hw : ws.size = m) (hl : ls.size = m) :
    fOf (Zk.Lin.combo ws ls s).c = ∑ k : Fin m, fOf ws[k.val]! * fOf (ls[k.val]!).c := by
  rw [combo_eq]
  dsimp only
  rw [hw, hl, Nat.min_self, comboLoop_c, fOf_zero, zero_add]

/-- **`Lin.combo`'s coefficients** are the `w`-sums of the forms' coefficients, for forms on `s` pads. -/
theorem combo_a (ws : Array F128) (ls : Array Zk.Lin) (s m : ℕ) (hw : ws.size = m) (hl : ls.size = m)
    (hls : ∀ k < m, (ls[k]!).a.size = s) (i : ℕ) (hi : i < s) :
    fOf (Zk.Lin.combo ws ls s).a[i]! = ∑ k : Fin m, fOf ws[k.val]! * fOf (ls[k.val]!).a[i]! := by
  rw [combo_eq]
  dsimp only
  rw [hw, hl, Nat.min_self, comboLoop_a ws ls s m (0, Array.replicate s 0) (Array.size_replicate ..) hls i hi,
    repl_get s hi, fOf_zero, zero_add]

/-! ## The code -/

theorem horner_list (x : F128) : ∀ l : List F128,
    fOf (l.foldr (fun c acc => acc * x + c) 0) = ∑ i ∈ Finset.range l.length, fOf (l.getD i 0) * fOf x ^ i
  | [] => by simp [fOf_zero]
  | c :: l => by
      rw [List.foldr_cons, fOf_add, fOf_mul, horner_list x l, List.length_cons, Finset.sum_range_succ']
      refine congrArg₂ (· + ·) ((Finset.sum_mul _ _ _).trans (Finset.sum_congr rfl fun i _ => ?_)) ?_
      · rw [List.getD_cons_succ, pow_succ, mul_assoc]
      · simp

/-- **The inner code at column `j`** is `Σ y_i·X^i` at `X = j + 1`. -/
theorem fOf_codeAt (y : Array F128) (j n : ℕ) (hn : y.size = n) :
    fOf (Zk.codeAt y j) = ∑ i : Fin n, fOf y[i.val]! * fOf ⟨(j + 1).toUInt64, 0⟩ ^ (i : ℕ) := by
  show fOf (y.foldr (fun c acc => acc * (⟨(j + 1).toUInt64, 0⟩ : F128) + c) 0) = _
  rw [← Array.foldr_toList, horner_list, Array.length_toList, hn,
    ← Fin.sum_univ_eq_sum_range (fun i => fOf (y.toList.getD i 0) * fOf ⟨(j + 1).toUInt64, 0⟩ ^ i) n]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hi : i.val < y.size := by rw [hn]; exact i.isLt
  have e : y.toList.getD i.val 0 = y[i.val]! := by
    simp [List.getD_eq_getElem?_getD, hi]
  rw [e]

/-! ## The leaves -/

/-- hm96's leaf at the executable's key, for any inner bytes `x`, is `Hm96.Default512.leaf` of `SHA-512(x)`. -/
theorem hm96_bytes (x : List UInt8) (salt : ByteArray) :
    (H512 (K512.leafPrefix ++ (Merkle.xorBytes (enc512.dig (H512 x)) (K512.mask salt.data.toList) ++
      enc512.dig (H512 (K512.saltPrefix ++ salt.data.toList))))).1 =
      Hm96.Default512.leaf (Sha512.hash ⟨x.toArray⟩) salt := by
  rw [H512_val, dig_eq, dig_eq, H512_val, H512_val, K512_mask, bytes_of_list,
    ← xor_range _ _ 64 (sha512_size _) (mask_size salt)]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ (_ ++
      (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList)).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash ⟨x.toArray⟩) salt)
  rw [← data_append Hm96.Default512.saltPrefix salt, bytes_of_list, ← data_append, ← data_append, bytes_of_list]
  rfl

theorem pair_bytes (a b : F128) : (⟨(colBytes [a, b]).toArray⟩ : ByteArray) = a.toBytes ++ b.toBytes := by
  have e : colBytes [a, b] = (a.toBytes ++ b.toBytes).data.toList := by rw [data_append]; simp [colBytes]
  rw [e, bytes_of_list]

theorem tau_bytes (τ : F128) :
    (⟨(Zk.TAU_TAG.data.toList ++ colBytes [τ]).toArray⟩ : ByteArray) = Zk.TAU_TAG ++ τ.toBytes := by
  have e : Zk.TAU_TAG.data.toList ++ colBytes [τ] = (Zk.TAU_TAG ++ τ.toBytes).data.toList := by
    rw [data_append]; simp [colBytes]
  rw [e, bytes_of_list]

theorem leafPair_eq (c : 𝔽 × 𝔽) (y : List UInt8) :
    (Merkle.Leaf.hm96 H512 encPair K512 hL512).leaf c y = Merkle.hm96Leaf (H := H512) (E := encPair) K512 c y := rfl

theorem leafTau_eq (c : 𝔽) (y : List UInt8) :
    (Merkle.Leaf.hm96 H512 encTau K512 hL512).leaf c y = Merkle.hm96Leaf (H := H512) (E := encTau) K512 c y := rfl

theorem encPair_col (p : 𝔽 × 𝔽) : encPair.col p = enc512.col [p.1, p.2] := by dsimp only [encPair]

theorem encTau_col (τ : 𝔽) : encTau.col τ = Zk.TAU_TAG.data.toList ++ enc512.col [τ] := by dsimp only [encTau]

theorem encPair_dig (d : Dig) : encPair.dig d = enc512.dig d := rfl

theorem encTau_dig (d : Dig) : encTau.dig d = enc512.dig d := rfl

theorem pair_col (o : F128 × F128) : encPair.col (pairOf o) = colBytes [o.1, o.2] :=
  (encPair_col _).trans (col_eq _)

theorem tau_col (τ : F128) : encTau.col (fOf τ) = Zk.TAU_TAG.data.toList ++ colBytes [τ] := by
  rw [encTau_col]
  exact congrArg (Zk.TAU_TAG.data.toList ++ ·) (col_eq _)

/-- **The pads pair's leaf** is the executable's: `hm96` of `SHA-512(a.toBytes ‖ b.toBytes)` under the salt. -/
theorem leaf_pair (o : F128 × F128) (salt : ByteArray) :
    ((Merkle.Leaf.hm96 H512 encPair K512 hL512).leaf (pairOf o) salt.data.toList).1 =
      Hm96.Default512.leaf (Sha512.hash (o.1.toBytes ++ o.2.toBytes)) salt := by
  rw [leafPair_eq, Merkle.hm96Leaf, pair_col, encPair_dig, encPair_dig, hm96_bytes, pair_bytes]

/-- **`τ`'s leaf** is the executable's `τ` commitment. -/
theorem leaf_tau (τ : F128) (salt : ByteArray) :
    ((Merkle.Leaf.hm96 H512 encTau K512 hL512).leaf (fOf τ) salt.data.toList).1 = Zk.tauCommitment τ salt := by
  rw [leafTau_eq, Merkle.hm96Leaf, tau_col, encTau_dig, encTau_dig, hm96_bytes, tau_bytes, Zk.tauCommitment]

theorem walk_pair : ∀ (sibs : List Dig) (idx : ℕ) (acc : Dig),
    Merkle.walk H512 encPair idx acc sibs = Merkle.walk H512 enc512 idx acc sibs
  | [], _, _ => rfl
  | s :: rest, idx, acc => by
      simp only [Merkle.walk]
      exact walk_pair rest (idx / 2) _

/-! ## The triple's rows -/

theorem lay_order (l : Zk.PadLayout) : l.fa + 1 = l.fb ∧ l.fb < l.hab ∧ l.hab + 1 = l.tx ∧ l.tx + 1 = l.ty ∧
    l.ty + 1 = l.tw ∧ l.tw + 1 = l.len := by
  refine ⟨rfl, ?_, rfl, rfl, rfl, rfl⟩
  unfold Zk.PadLayout.fb Zk.PadLayout.hab Zk.PadLayout.zp Zk.PadLayout.lc Zk.PadLayout.fc
  omega

section Triple

variable {mPts : ℕ} {T : ZkSession.TabZK 𝔽 mPts} {st : Setup} {sch : Flock.Schedule} {tPad : ℕ}

theorem tr_pos (htab : ZkTabLay T st sch) :
    T.tr.fa.val = (layZ st sch).fa ∧ T.tr.fb.val = (layZ st sch).fb ∧ T.tr.ab.val = (layZ st sch).hab ∧
      T.tr.tx.val = (layZ st sch).tx ∧ T.tr.ty.val = (layZ st sch).ty ∧ T.tr.tw.val = (layZ st sch).tw := by
  have ho := lay_order (layZ st sch)
  rw [htab.hfa, htab.hfb, htab.hab, htab.htx, htab.hty, htab.htw]
  refine ⟨Nat.mod_eq_of_lt ?_, Nat.mod_eq_of_lt ?_, Nat.mod_eq_of_lt ?_, Nat.mod_eq_of_lt ?_, Nat.mod_eq_of_lt ?_,
    Nat.mod_eq_of_lt ?_⟩ <;> omega

theorem rel_rho (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch) (e r : F128) :
    LinRel (⟨Pi.single T.tr.fa (fOf e) + Pi.single T.tr.tx 1, fOf r⟩ : ZkSession.Lin 𝔽 T.np)
      ⟨r, ((Array.replicate (layZ st sch).len 0).set! (layZ st sch).fa e).set! (layZ st sch).tx 1⟩ := by
  have ho := lay_order (layZ st sch)
  obtain ⟨hfa, -, -, htx, -, -⟩ := tr_pos htab
  refine ⟨by simp [hseam.hnp], rfl, fun i => ?_⟩
  have hi : i.val < (layZ st sch).len := by rw [← hseam.hnp]; exact i.isLt
  rw [set_get _ _ (by simp; omega) (by simp; omega), set_get _ _ (by simp; omega) (by simp; omega),
    repl_get _ hi]
  simp only [Pi.add_apply, Pi.single_apply, Fin.ext_iff, hfa, htx]
  by_cases h1 : i.val = (layZ st sch).fa
  · simp only [ite_eq_left h1, ite_eq_right (show ¬ i.val = (layZ st sch).tx by omega), add_zero]
  by_cases h2 : i.val = (layZ st sch).tx
  · simp only [ite_eq_left h2, ite_eq_right h1, zero_add, fOf_one]
  · simp only [ite_eq_right h1, ite_eq_right h2, add_zero, fOf_zero]

theorem rel_sigma (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch) (s : F128) :
    LinRel (⟨Pi.single T.tr.fb 1 + Pi.single T.tr.ty 1, fOf s⟩ : ZkSession.Lin 𝔽 T.np)
      ⟨s, ((Array.replicate (layZ st sch).len 0).set! (layZ st sch).fb 1).set! (layZ st sch).ty 1⟩ := by
  have ho := lay_order (layZ st sch)
  obtain ⟨-, hfb, -, -, hty, -⟩ := tr_pos htab
  refine ⟨by simp [hseam.hnp], rfl, fun i => ?_⟩
  have hi : i.val < (layZ st sch).len := by rw [← hseam.hnp]; exact i.isLt
  rw [set_get _ _ (by simp; omega) (by simp; omega), set_get _ _ (by simp; omega) (by simp; omega),
    repl_get _ hi]
  simp only [Pi.add_apply, Pi.single_apply, Fin.ext_iff, hfb, hty]
  by_cases h1 : i.val = (layZ st sch).fb
  · simp only [ite_eq_left h1, ite_eq_right (show ¬ i.val = (layZ st sch).ty by omega), add_zero, fOf_one]
  by_cases h2 : i.val = (layZ st sch).ty
  · simp only [ite_eq_left h2, ite_eq_right h1, zero_add, fOf_one]
  · simp only [ite_eq_right h1, ite_eq_right h2, add_zero, fOf_zero]

theorem rel_sacrifice (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch) (e r s : F128) :
    LinRel (⟨Pi.single T.tr.ab (fOf e) + Pi.single T.tr.tw 1 + Pi.single T.tr.tx (fOf s) + Pi.single T.tr.ty (fOf r),
        fOf r * fOf s⟩ : ZkSession.Lin 𝔽 T.np)
      ⟨r * s, ((((Array.replicate (layZ st sch).len 0).set! (layZ st sch).hab e).set! (layZ st sch).tw 1).set!
        (layZ st sch).tx s).set! (layZ st sch).ty r⟩ := by
  have ho := lay_order (layZ st sch)
  obtain ⟨-, -, hab, htx, hty, htw⟩ := tr_pos htab
  refine ⟨by simp [hseam.hnp], (fOf_mul r s).symm, fun i => ?_⟩
  have hi : i.val < (layZ st sch).len := by rw [← hseam.hnp]; exact i.isLt
  rw [set_get _ _ (by simp; omega) (by simp; omega), set_get _ _ (by simp; omega) (by simp; omega),
    set_get _ _ (by simp; omega) (by simp; omega), set_get _ _ (by simp; omega) (by simp; omega), repl_get _ hi]
  simp only [Pi.add_apply, Pi.single_apply, Fin.ext_iff, hab, htx, hty, htw]
  by_cases h1 : i.val = (layZ st sch).hab
  · simp only [ite_eq_left h1, ite_eq_right (show ¬ i.val = (layZ st sch).tw by omega),
      ite_eq_right (show ¬ i.val = (layZ st sch).tx by omega), ite_eq_right (show ¬ i.val = (layZ st sch).ty by omega), add_zero]
  by_cases h2 : i.val = (layZ st sch).tw
  · simp only [ite_eq_left h2, ite_eq_right h1, ite_eq_right (show ¬ i.val = (layZ st sch).tx by omega),
      ite_eq_right (show ¬ i.val = (layZ st sch).ty by omega), zero_add, add_zero, fOf_one]
  by_cases h3 : i.val = (layZ st sch).tx
  · simp only [ite_eq_left h3, ite_eq_right h1, ite_eq_right h2, ite_eq_right (show ¬ i.val = (layZ st sch).ty by omega), zero_add,
      add_zero]
  by_cases h4 : i.val = (layZ st sch).ty
  · simp only [ite_eq_left h4, ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, zero_add]
  · simp only [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3, ite_eq_right h4, add_zero, fOf_zero]

/-- **The sacrifice's three rows are the executable's `triple`**, at the pads layout's positions. -/
theorem triple_rel (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch) (e r s : F128) :
    ∀ k : Fin 3, LinRel (ZkSession.tripleRows T.tr (fOf e) (fOf r) (fOf s) k)
      ((Zk.triple (layZ st sch) e r s)[k.val]!).2 := by
  intro k
  fin_cases k
  · exact rel_rho hseam htab e r
  · exact rel_sigma hseam htab s
  · exact rel_sacrifice hseam htab e r s

end Triple

/-! ## The run -/

/-- **`Zk.verifyInner`'s run, with its coins**: it reads four rounds, `ε`'s (one coin), the ten batching coins,
`β`'s (one coin) and the `Q` query words, and its checks hold at them. -/
theorem zk_inner_peel {lay : Zk.PadLayout} {pr : Zk.InnerProof} {finalC : F128} {cons : Array (String × Zk.Lin)}
    {t t' : Flock.Transcript} (hcons : cons.size = 7) (hwf : RoundsWF t.rounds)
    (h : Zk.verifyInner lay pr finalC cons t = .ok ((), t')) :
    ∃ c₀ c₁ c₂ c₃ : Array F128, c₀.size = 1 ∧ c₁.size = 7 + 3 ∧ c₂.size = 1 ∧ c₃.size = Zk.Q ∧
      consumed t.rounds t.k t'.k = [c₀, c₁, c₂, c₃] ∧
      dot (innerRow lay pr cons c₀[0]! c₁).a pr.y = (innerRow lay pr cons c₀[0]! c₁).c + c₂[0]! * pr.tau ∧
      ∀ k < Zk.Q,
        Zk.codeAt pr.y ((c₃[k]!).lo.toNat % innerN lay) = (pr.opened[k]!).1 + c₂[0]! * (pr.opened[k]!).2 ∧
        climb MerkleScheme.hm96Sha512
          (Hm96.Default512.leaf (Sha512.hash ((pr.opened[k]!).1.toBytes ++ (pr.opened[k]!).2.toBytes)) pr.salts[k]!)
          ((c₃[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList = pr.root := by
  unfold Zk.verifyInner at h
  dsimp only at h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbF_bind h
  obtain ⟨-, n₂, t₃, r₃, kk₃, h⟩ := sample_bind h
  obtain ⟨t₄, k₄, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₄, t₅, r₅, kk₅, h⟩ := sampleVec_bind (by simp [Zk.triple]) h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨-, n₆, t₇, r₇, kk₇, h⟩ := sample_bind h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨t₈, k₈, h⟩ := absorbSlice_bind h
  obtain ⟨-, n₈, t₉, r₉, kk₉, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨-, h⟩ := check_bind h
  obtain ⟨hdot, h⟩ := check_bind h
  rw [forIn_range_eq] at h
  obtain ⟨_, tL, hloop, hE⟩ := bind_ok h
  have hL := forIn_keeps _ (by
    intro i s tt b tt' hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨-, hb⟩ := check_bind hb
    obtain ⟨rfl, rfl⟩ := pure_ok hb
    exact ⟨_, rfl, rfl⟩) Zk.Q 0 _ _ _ _ hloop
  have hk := forIn_each (fun k =>
      Zk.codeAt pr.y ((((t₈.rounds[t₈.k]!).coins)[k]!).lo.toNat % innerN lay) =
        (pr.opened[k]!).1 + (t₆.rounds[t₆.k]!).coins.getD 0 0 * (pr.opened[k]!).2 ∧
      climb MerkleScheme.hm96Sha512
        (Hm96.Default512.leaf (Sha512.hash ((pr.opened[k]!).1.toBytes ++ (pr.opened[k]!).2.toBytes)) pr.salts[k]!)
        ((((t₈.rounds[t₈.k]!).coins)[k]!).lo.toNat % innerN lay) (pr.paths[k]!).toList = pr.root) _
    (fun i s t b t' hb => by
      obtain ⟨c1, hb⟩ := check_bind hb
      obtain ⟨-, hb⟩ := check_bind hb
      obtain ⟨c3, hb⟩ := check_bind hb
      obtain ⟨rfl, -⟩ := pure_ok hb
      rw [map_lo_getElem!] at c1 c3
      exact ⟨⟨eq_of_beq c1, bytes_eq_of_beq c3⟩, _, rfl⟩) Zk.Q 0 _ _ _ _ hloop
  have ht' : t' = t₉ := (pure_ok hE).2.trans hL
  have hr₂ : t₂.rounds = t.rounds := by rw [k₂.2, k₁.2]
  have hk₂ : t₂.k = t.k := by rw [k₂.1, k₁.1]
  have hr₄ : t₄.rounds = t.rounds := by rw [k₄.2, r₃, hr₂]
  have hk₄ : t₄.k = t.k + 1 := by rw [k₄.1, kk₃, hk₂]
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, r₅, hr₄]
  have hk₆ : t₆.k = t.k + 2 := by rw [k₆.1, kk₅, hk₄]
  have hr₈ : t₈.rounds = t.rounds := by rw [k₈.2, r₇, hr₆]
  have hk₈ : t₈.k = t.k + 3 := by rw [k₈.1, kk₇, hk₆]
  have hk₉ : t₉.k = t.k + 4 := by rw [kk₉, hk₈]
  have hsz₀ : (t₂.rounds[t₂.k]!).coins.size = 1 := by rw [RoundsWF.get (hr₂ ▸ hwf), n₂]
  have hsz₁ : (t₄.rounds[t₄.k]!).coins.size = 7 + 3 := by
    rw [RoundsWF.get (hr₄ ▸ hwf), n₄]; simp [Zk.triple, hcons]
  have hsz₂ : (t₆.rounds[t₆.k]!).coins.size = 1 := by rw [RoundsWF.get (hr₆ ▸ hwf), n₆]
  have hsz₃ : (t₈.rounds[t₈.k]!).coins.size = Zk.Q := by rw [RoundsWF.get (hr₈ ▸ hwf), n₈]
  have hcoins : consumed t.rounds t.k t'.k = [(t₂.rounds[t₂.k]!).coins, (t₄.rounds[t₄.k]!).coins,
      (t₆.rounds[t₆.k]!).coins, (t₈.rounds[t₈.k]!).coins] := by
    rw [ht', hk₉, consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 3 by omega) (by omega), consumed_succ, consumed_succ,
      consumed_succ, consumed_succ, hr₂, hk₂, hr₄, hk₄, hr₆, hk₆, hr₈, hk₈]
    rfl
  refine ⟨_, _, _, _, hsz₀, hsz₁, hsz₂, hsz₃, hcoins, ?_, fun k hk' => ?_⟩
  · rw [getElem!_eq_getD (t₂.rounds[t₂.k]!).coins, getElem!_eq_getD (t₆.rounds[t₆.k]!).coins]
    exact eq_of_beq hdot
  · rw [getElem!_eq_getD (t₆.rounds[t₆.k]!).coins]
    exact hk k (List.mem_range'_1.2 ⟨Nat.zero_le _, by simpa using hk'⟩)

/-! ## The check -/

theorem zk_inner_accepts : ZkInnerAccepts := by
  intro lay pr finalC cons t t' mPts T st sch tPad rows hlay hseam htab hcons hrows hpaths hwf h v hv
  subst hlay
  obtain ⟨hy, -, -, hp, hr, htc, -, eps, beta, gammas, ws, -, htau, -, hcolsOk⟩ := verifyInner_ok _ _ _ _ _ _ h
  obtain ⟨c₀, c₁, c₂, c₃, hs₀, hs₁, hs₂, hs₃, hcs, hdot, hcol⟩ := zk_inner_peel hcons hwf h
  rw [hcs] at hv
  have hq1 : T.q ≠ 1 := by rw [hseam.hq]; decide
  have hv₀ : Plays fOf (ZkLink.innerGameC hs512.H hs512.Ep hs512.Lp hs512.Lτ T.dom T.np T.nP T.q T.pos rows T.tr
      (toDig pr.root))
      [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩]
      [c₀, c₁, c₂, c₃]
      (ZkLink.innerCheckC hs512.H hs512.Ep hs512.Lp hs512.Lτ T.dom T.pos rows T.tr (toDig pr.root)
        (toDig pr.tauCommit) (fOf c₀[0]!) (fOf pr.rho, fOf pr.sigma) (fun i : Fin (7 + 3) => fOf c₁[i.val]!)
        (fOf c₂[0]!) (vecOf (T.np + T.nP) pr.y) (fun i : Fin T.q => fOf c₃[i.val]!)) := by
    unfold ZkLink.innerGameC
    refine Plays.bind' (ms₂ := [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩]) (cs₂ := [c₁, c₂, c₃]) (Plays.drawOne c₀ hs₀) ?_ rfl rfl
    refine Plays.bind' (ms₂ := [⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩])
      (cs₂ := [c₁, c₂, c₃]) (Plays.recv _ (fOf pr.rho, fOf pr.sigma)) ?_ rfl rfl
    refine Plays.bind' (ms₂ := [⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩])
      (cs₂ := [c₂, c₃]) (Plays.drawVec c₁ hs₁ (by decide)) ?_ rfl rfl
    refine Plays.bind' (ms₂ := [⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩]) (cs₂ := [c₂, c₃])
      (Plays.recv _ (toDig pr.tauCommit)) ?_ rfl rfl
    refine Plays.bind' (ms₂ := [⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) pr.y⟩]) (cs₂ := [c₃])
      (Plays.drawOne c₂ hs₂) ?_ rfl rfl
    refine Plays.bind' (ms₂ := []) (cs₂ := [c₃]) (Plays.recv _ (vecOf (T.np + T.nP) pr.y)) ?_ rfl rfl
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec c₃ (hs₃.trans hseam.hq.symm) hq1) ?_ rfl rfl
    exact Plays.pure (ι := fOf) _
  obtain rfl := plays_det hv hv₀
  -- `τ` opens its commitment.
  have hT : (Merkle.Leaf.hm96 H512 encTau K512 hL512).leaf (fOf pr.tau) pr.tauSalt.data.toList =
      toDig pr.tauCommit := by
    apply Subtype.ext
    rw [toDig_val htc, leaf_tau, htau]
  -- The batched constraint.
  have hts : (Zk.triple (layZ st sch) c₀[0]! pr.rho pr.sigma).size = 3 := rfl
  have hcs7 : (cons ++ Zk.triple (layZ st sch) c₀[0]! pr.rho pr.sigma).size = 7 + 3 := by
    rw [Array.size_append, hcons, hts]
  have hR : ∀ k : Fin (7 + 3), LinRel (Fin.append rows
      (ZkSession.tripleRows T.tr (fOf c₀[0]!) (fOf pr.rho) (fOf pr.sigma)) k)
      ((cons ++ Zk.triple (layZ st sch) c₀[0]! pr.rho pr.sigma).map (·.2))[k.val]! := by
    intro k
    refine Fin.addCases (fun i => ?_) (fun i => ?_) k
    · rw [Fin.append_left, Fin.val_castAdd, map_get! _ _ (by rw [hcs7]; have := i.isLt; omega),
        append_get!_left _ _ (by rw [hcons]; exact i.isLt)]
      exact hrows i
    · rw [Fin.append_right, Fin.val_natAdd, map_get! _ _ (by rw [hcs7]; have := i.isLt; omega),
        append_get!_right (j := i.val) _ _ (by rw [hcons]) (by rw [hts]; exact i.isLt)]
      exact triple_rel hseam htab _ _ _ i
  have hLs : ((cons ++ Zk.triple (layZ st sch) c₀[0]! pr.rho pr.sigma).map (·.2)).size = 7 + 3 := by
    rw [Array.size_map, hcs7]
  obtain ⟨L, hL⟩ : ∃ L, (cons ++ Zk.triple (layZ st sch) c₀[0]! pr.rho pr.sigma).map (·.2) = L := ⟨_, rfl⟩
  have hrow : innerRow (layZ st sch) pr cons c₀[0]! c₁ = Zk.Lin.combo c₁ L (layZ st sch).len := by
    subst hL; rfl
  rw [hL] at hR hLs
  rw [hrow] at hdot
  obtain ⟨R, hRdef⟩ : ∃ R, Fin.append rows (ZkSession.tripleRows T.tr (fOf c₀[0]!) (fOf pr.rho) (fOf pr.sigma)) = R :=
    ⟨_, rfl⟩
  rw [hRdef] at hR
  have hla : ∀ k < 7 + 3, (L[k]!).a.size = (layZ st sch).len := fun k hk => (hR ⟨k, hk⟩).1.trans hseam.hnp
  have hdot' := congrArg fOf hdot
  rw [fOf_add, fOf_mul, fOf_dot _ _ T.np (by rw [combo_size]; exact hseam.hnp.symm)] at hdot'
  have hcoef : ∀ i : Fin T.np, ZkSession.batchCoef (fun k : Fin (7 + 3) => fOf c₁[k.val]!) R i =
      fOf (Zk.Lin.combo c₁ L (layZ st sch).len).a[i.val]! := by
    intro i
    rw [combo_a c₁ L _ (7 + 3) hs₁ hLs hla i.val (by rw [← hseam.hnp]; exact i.isLt)]
    exact Finset.sum_congr rfl fun k _ => congrArg₂ (· * ·) rfl ((hR k).2.2 i)
  have hconst : ZkSession.batchT (fun k : Fin (7 + 3) => fOf c₁[k.val]!) R =
      fOf (Zk.Lin.combo c₁ L (layZ st sch).len).c := by
    rw [ZkSession.batchT, CharTwo.neg_eq, combo_c c₁ L _ (7 + 3) hs₁ hLs]
    exact Finset.sum_congr rfl fun k _ => congrArg₂ (· * ·) rfl (hR k).2.1
  have hB : ∑ i, ZkSession.batchCoef (fun k : Fin (7 + 3) => fOf c₁[k.val]!) R i *
      ZkSession.head (vecOf (T.np + T.nP) pr.y) i =
      ZkSession.batchT (fun k : Fin (7 + 3) => fOf c₁[k.val]!) R + fOf c₂[0]! * fOf pr.tau := by
    rw [hconst, ← hdot']
    exact Finset.sum_congr rfl fun i _ => congrArg₂ (· * ·) (hcoef i) rfl
  subst hRdef
  -- Each query.
  have hC : ∀ j : Fin T.q,
      ZkSession.enc T.dom (vecOf (T.np + T.nP) pr.y) (T.pos (fOf c₃[j.val]!)) =
        fOf (pr.opened[j.val]!).1 + fOf c₂[0]! * fOf (pr.opened[j.val]!).2 ∧
      Merkle.VerifiesL H512 encPair (Merkle.Leaf.hm96 H512 encPair K512 hL512) (fun _ => toDig pr.root)
        (Nat.log 2 T.N) (T.pos (fOf c₃[j.val]!)).val (pairOf pr.opened[j.val]!) (pr.salts[j.val]!).data.toList
        ((pr.paths[j.val]!).toList.map toDig) := by
    intro j
    have hjq : j.val < Zk.Q := hseam.hq ▸ j.isLt
    obtain ⟨hc, hcl⟩ := hcol j.val hjq
    have hpv : (T.pos (fOf c₃[j.val]!)).val = (c₃[j.val]!).lo.toNat % innerN (layZ st sch) := by
      rw [htab.hpos (fOf c₃[j.val]!), htab.hN]
      rfl
    refine ⟨?_, ?_⟩
    · have e1 : ZkSession.enc T.dom (vecOf (T.np + T.nP) pr.y) (T.pos (fOf c₃[j.val]!)) =
          fOf (Zk.codeAt pr.y ((c₃[j.val]!).lo.toNat % innerN (layZ st sch))) := by
        rw [fOf_codeAt pr.y _ (T.np + T.nP) (by rw [hy, hseam.hnp, hseam.hnP]), ← hpv, ← htab.hdom]
        rfl
      rw [e1, hc, fOf_add, fOf_mul]
    unfold Merkle.VerifiesL
    refine ⟨?_, ?_⟩
    · rw [List.length_map, Array.length_toList, (hcolsOk j.val hjq).2.1, htab.hN, Nat.log2_eq_log_two]
    · have hsib : ∀ b ∈ (pr.paths[j.val]!).toList, b.size = 64 := hpaths j.val (by rw [hp]; exact hjq)
      have hmap : ((pr.paths[j.val]!).toList.map toDig).map (·.1) = (pr.paths[j.val]!).toList := by
        rw [List.map_map]
        exact (List.map_congr_left fun b hb => toDig_val (hsib b hb)).trans (List.map_id _)
      apply Subtype.ext
      show (Merkle.walk H512 encPair _ _ _).1 = (toDig pr.root).1
      rw [walk_pair, walk_val "", hmap, leaf_pair, ← climb_hm96 "", hpv, hcl, toDig_val hr]
  dsimp only [ZkLink.innerCheckC]
  exact ⟨hT, hB, hC⟩

end FlockSoundness.Discharge.ZkExec
