import Proofs.Accounting.Tag.Basic
import Definitions.Accounting.Guarantees.Tag

/-!
# The polynomial hash's properties

Proofs of `Accounting.Guarantees.Tag`'s statements about the keyed polynomial hash over `GF(2^31 − 1)`: non-collision
for fixed blocks and for fixed lists of hashes (root counting), unpredictability under partial reads (a residue class mod
`p` holds at most three 32-bit words), the digest (the two together) and the Horner loop.
-/

namespace Accounting.TagProofs

open Accounting.Tag Accounting.Guarantees.Tag
open scoped ENNReal

/-! ## Non-collision for fixed blocks -/

theorem blockHash_sub {W : ℕ} (g : ZMod P) (x y : Chunk W) :
    blockHash g x - blockHash g y = ∑ j : Fin W, (residues x j - residues y j) * g ^ (W - 1 - (j : ℕ)) := by
  simp [blockHash, residues, ← Finset.sum_sub_distrib, sub_mul]

/-- Two blocks with different residues hash alike at at most `W − 1` keys. -/
theorem card_blockHash_eq_le {W : ℕ} (x y : Chunk W) (hxy : residues x ≠ residues y) :
    (Finset.univ.filter fun g : ZMod P => blockHash g x = blockHash g y).card ≤ W - 1 := by
  classical
  refine le_trans (Finset.card_le_card fun g hg => ?_)
    (card_zeros_le (fun j : Fin W => residues x j - residues y j) (fun j => W - 1 - (j : ℕ)) (fun i j h => ?_) (W - 1)
      (fun j => Nat.sub_le _ _) fun h => hxy (funext fun j => sub_eq_zero.1 (congrFun h j)))
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hg ⊢
    rw [← blockHash_sub, hg, sub_self]
  · have := i.isLt
    have := j.isLt
    exact Fin.ext (by simp only at h; omega)

theorem polyNonCollisionKeyed : PolyNonCollisionKeyed := by
  intro W ν x y
  classical
  apply prob_le_card_mul_iSup
  by_cases hxy : residues x = residues y
  · simp [hxy]
  · simpa [hxy] using card_blockHash_eq_le x y hxy

theorem polyNonCollision : PolyNonCollision := by
  intro W x y
  have h := polyNonCollisionKeyed W (PMF.uniformOfFintype (ZMod P)) x y
  rwa [iSup_uniform, card_zmod, ← div_eq_mul_inv] at h

/-! ## Unpredictability under partial reads -/

/-- A residue class mod `p` holds at most three 32-bit words: `r`, `r + p` and `r + 2p`, since `2^32 < 3p`. -/
theorem card_residue_le (r : ZMod P) : (Finset.univ.filter fun b : Word => residue b = r).card ≤ 3 := by
  classical
  have hmod : ∀ b : Word, residue b = r → (b : ℕ) % P = r.val := by
    intro b hb
    rw [← hb, residue, ZMod.val_natCast]
  calc (Finset.univ.filter fun b : Word => residue b = r).card ≤ (Finset.range 3).card := by
        refine Finset.card_le_card_of_injOn (fun b : Word => (b : ℕ) / P) (fun b _ => ?_) ?_
        · have hb : (b : ℕ) < 4294967296 := b.isLt
          simp only [Finset.coe_range, Set.mem_Iio]
          rw [Nat.div_lt_iff_lt_mul (by norm_num)]
          show (b : ℕ) < 3 * 2147483647
          omega
        · intro b hb b' hb' h
          simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hb hb'
          have e1 := Nat.div_add_mod (b : ℕ) P
          have e2 := Nat.div_add_mod (b' : ℕ) P
          simp only at h
          rw [hmod b hb] at e1
          rw [hmod b' hb'] at e2
          exact Fin.ext (by rw [← e1, ← e2, h])
    _ = 3 := Finset.card_range 3

/-- The hash of a block with word `j` replaced by `b`: a constant plus `(b mod p) · g^(W − 1 − j)`. -/
theorem blockHash_update {W : ℕ} (g : ZMod P) (y : Chunk W) (j : Fin W) (b : Word) :
    blockHash g (Function.update y j b) =
      residue b * g ^ (W - 1 - (j : ℕ)) + ∑ i ∈ Finset.univ.erase j, residue (y i) * g ^ (W - 1 - (i : ℕ)) := by
  unfold blockHash
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ j), Function.update_self]
  congr 1
  refine Finset.sum_congr rfl fun i hi => ?_
  rw [Function.update_of_ne (Finset.ne_of_mem_erase hi)]

/-- At a nonzero key, at most three values of word `j` give the hash any one value. -/
theorem card_update_eq_le {W : ℕ} (g : (ZMod P)ˣ) (y : Chunk W) (j : Fin W) (t : ZMod P) :
    (Finset.univ.filter fun b : Word => blockHash (g : ZMod P) (Function.update y j b) = t).card ≤ 3 := by
  classical
  set c := ∑ i ∈ Finset.univ.erase j, residue (y i) * (g : ZMod P) ^ (W - 1 - (i : ℕ))
  set e := (g : ZMod P) ^ (W - 1 - (j : ℕ))
  have he : e ≠ 0 := pow_ne_zero _ g.ne_zero
  refine le_trans (Finset.card_le_card fun b hb => ?_) (card_residue_le ((t - c) / e))
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb ⊢
  rw [blockHash_update] at hb
  rw [eq_div_iff he]
  exact eq_sub_of_add_eq hb

/-- Each point mass is at most `wordGuess` times its class's mass. -/
theorem le_wordGuess_mul {W : ℕ} (μ : PMF (Chunk W)) (j : Fin W) (x : Chunk W) :
    μ x ≤ wordGuess μ j * ∑ a : Word, μ (Function.update x j a) := by
  set s := ∑ a : Word, μ (Function.update x j a)
  have hxs : μ x ≤ s := by
    have := Finset.single_le_sum (f := fun a : Word => μ (Function.update x j a)) (fun _ _ => zero_le)
      (Finset.mem_univ (x j))
    simpa [s, Function.update_eq_self] using this
  by_cases hs : s = 0
  · rw [hs] at hxs ⊢
    simpa using hxs
  have hs' : s ≠ ⊤ := by
    refine ne_top_of_le_ne_top ENNReal.one_ne_top ?_
    calc s = ∑ a : Word, μ (Function.update x j a) := rfl
      _ = ∑ y ∈ Finset.univ.image (fun a : Word => Function.update x j a), μ y := by
          rw [Finset.sum_image fun a _ b _ h => by simpa using congrFun h j]
      _ ≤ ∑ y : Chunk W, μ y := Finset.sum_le_sum_of_subset (Finset.subset_univ _)
      _ = 1 := sum_pmf μ
  have h1 : μ x / s ≤ wordGuess μ j := le_iSup (fun y : Chunk W => μ y / ∑ a : Word, μ (Function.update y j a)) x
  calc μ x = μ x / s * s := (ENNReal.div_mul_cancel hs hs').symm
    _ ≤ wordGuess μ j * s := by gcongr

/-- Re-pairing a block and a word: `(x, a) ↦ (x with word j set to a, x's word j)`, its own inverse. -/
def swapWord {W : ℕ} (j : Fin W) : Chunk W × Word ≃ Chunk W × Word where
  toFun p := (Function.update p.1 j p.2, p.1 j)
  invFun p := (Function.update p.1 j p.2, p.1 j)
  left_inv p := by simp [Function.update_eq_self]
  right_inv p := by simp [Function.update_eq_self]

theorem polyUnpredictable : PolyUnpredictable := by
  intro W μ j g t
  classical
  set G := wordGuess μ j
  set H : Chunk W → Prop := fun x => blockHash (g : ZMod P) x = t
  rw [prob_eq_sum]
  calc ∑ x, (if H x then μ x else 0)
      ≤ ∑ x, (if H x then G * ∑ a : Word, μ (Function.update x j a) else 0) := by
        refine Finset.sum_le_sum fun x _ => ?_
        split_ifs
        · exact le_wordGuess_mul μ j x
        · exact le_rfl
    _ = G * ∑ p : Chunk W × Word, (if H p.1 then μ (Function.update p.1 j p.2) else 0) := by
        rw [Fintype.sum_prod_type, Finset.mul_sum]
        refine Finset.sum_congr rfl fun x _ => ?_
        split_ifs <;> simp [Finset.mul_sum]
    _ = G * ∑ p : Chunk W × Word, (if H (Function.update p.1 j p.2) then μ p.1 else 0) := by
        refine congrArg (G * ·) ?_
        rw [← (swapWord j).sum_comp]
        refine Finset.sum_congr rfl fun p _ => ?_
        simp [swapWord, Function.update_eq_self]
    _ = G * ∑ y, μ y * ((Finset.univ.filter fun b : Word => H (Function.update y j b)).card : ℝ≥0∞) := by
        rw [Fintype.sum_prod_type]
        refine congrArg (G * ·) ?_
        refine Finset.sum_congr rfl fun y _ => ?_
        rw [← Finset.sum_filter]
        dsimp only
        rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ G * ∑ y, μ y * 3 := by
        gcongr with y
        exact_mod_cast card_update_eq_le g y j t
    _ = 3 * G := by
        rw [← Finset.sum_mul, sum_pmf μ, one_mul, mul_comm]

/-! ## The fold -/

theorem fold_eq_sum_getD (z : ZMod P) (hs : List (ZMod P)) (N : ℕ) (hN : hs.length ≤ N) :
    fold z hs = ∑ r : Fin N, hs.getD r 0 * z ^ (r : ℕ) := by
  unfold fold
  have h1 : ∑ r : Fin hs.length, hs[r] * z ^ (r : ℕ) = ∑ r : Fin hs.length, hs.getD r 0 * z ^ (r : ℕ) := by
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [List.getD_eq_getElem hs 0 r.isLt]
    rfl
  rw [h1, Fin.sum_univ_eq_sum_range (fun r => hs.getD r 0 * z ^ r),
    Fin.sum_univ_eq_sum_range (fun r => hs.getD r 0 * z ^ r)]
  refine Finset.sum_subset (Finset.range_subset_range.2 hN) fun r _ hr => ?_
  rw [Finset.mem_range, not_lt] at hr
  rw [List.getD_eq_default hs 0 hr, zero_mul]

theorem foldNonCollision : FoldNonCollision := by
  intro hs hs'
  classical
  set N := max hs.length hs'.length
  refine le_trans (prob_uniform_le _ (N - 1) ?_) (le_of_eq (by rw [card_zmod]))
  by_cases hd : ∃ r, hs.getD r 0 ≠ hs'.getD r 0
  · obtain ⟨r0, hr0⟩ := hd
    have hr0N : r0 < N := by
      by_contra hge
      push Not at hge
      apply hr0
      rw [List.getD_eq_default hs 0 ((le_max_left _ _).trans hge),
        List.getD_eq_default hs' 0 ((le_max_right _ _).trans hge)]
    refine le_trans (Finset.card_le_card fun z hz => ?_)
      (card_zeros_le (fun r : Fin N => hs.getD r 0 - hs'.getD r 0) (fun r => (r : ℕ)) Fin.val_injective (N - 1)
        (fun r => by have := r.isLt; omega)
        fun h0 => hr0 (sub_eq_zero.1 (congrFun h0 ⟨r0, hr0N⟩)))
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hz ⊢
    obtain ⟨_, hz⟩ := hz
    rw [fold_eq_sum_getD z hs N (le_max_left _ _), fold_eq_sum_getD z hs' N (le_max_right _ _)] at hz
    have h0 := sub_eq_zero.2 hz
    rw [← Finset.sum_sub_distrib] at h0
    simpa [sub_mul] using h0
  · rw [Finset.filter_false_of_mem fun z _ h => hd h.1, Finset.card_empty]
    exact Nat.zero_le _

/-! ## The digest -/

theorem digest_eq_sum {R W : ℕ} (g z : ZMod P) (xs : Fin R → Chunk W) :
    digest g z xs = ∑ r : Fin R, blockHash g (xs r) * z ^ (r : ℕ) := by
  unfold digest
  rw [fold_eq_sum_getD z _ R (by simp)]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

theorem digest_sub {R W : ℕ} (g z : ZMod P) (xs ys : Fin R → Chunk W) :
    digest g z xs - digest g z ys = ∑ r : Fin R, (blockHash g (xs r) - blockHash g (ys r)) * z ^ (r : ℕ) := by
  simp [digest_eq_sum, ← Finset.sum_sub_distrib, sub_mul]

/-- If some block hash differs at `g`, the digests agree at at most `R − 1` points `z`. -/
theorem card_digest_eq_le {R W : ℕ} (g : ZMod P) (xs ys : Fin R → Chunk W)
    (h : ∃ r, blockHash g (xs r) ≠ blockHash g (ys r)) :
    (Finset.univ.filter fun z : ZMod P => digest g z xs = digest g z ys).card ≤ R - 1 := by
  classical
  obtain ⟨r, hr⟩ := h
  refine le_trans (Finset.card_le_card fun z hz => ?_)
    (card_zeros_le (fun r : Fin R => blockHash g (xs r) - blockHash g (ys r)) (fun r => (r : ℕ)) Fin.val_injective
      (R - 1) (fun r => by have := r.isLt; omega) fun h0 => hr (sub_eq_zero.1 (congrFun h0 r)))
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hz ⊢
  rw [← digest_sub, hz, sub_self]

theorem digestNonCollision : DigestNonCollision := by
  intro R W xs ys
  classical
  have hcardP : Fintype.card (ZMod P × ZMod P) = P * P := by rw [Fintype.card_prod, card_zmod]
  by_cases hd : ∃ r, residues (xs r) ≠ residues (ys r)
  swap
  · have : (fun k : ZMod P × ZMod P => (∃ r, residues (xs r) ≠ residues (ys r)) ∧
        digest k.1 k.2 xs = digest k.1 k.2 ys) = fun _ => False := by
      funext k
      simp only [hd, false_and]
    rw [this, prob_eq_sum]
    simp
  obtain ⟨r0, hr0⟩ := hd
  set S := Finset.univ.filter fun g : ZMod P => blockHash g (xs r0) = blockHash g (ys r0) with hSdef
  have hS : S.card ≤ W - 1 := card_blockHash_eq_le _ _ hr0
  have hslice : ∀ g : ZMod P, (Finset.univ.filter fun z : ZMod P => digest g z xs = digest g z ys).card ≤
      (if g ∈ S then P else 0) + (R - 1) := by
    intro g
    by_cases hall : ∀ r, blockHash g (xs r) = blockHash g (ys r)
    · have hg : g ∈ S := by simp [hSdef, hall r0]
      simp only [hg, ite_true]
      exact le_add_right ((Finset.card_le_univ _).trans card_zmod.le)
    · push Not at hall
      exact le_add_left (card_digest_eq_le g xs ys hall)
  have hcount : (Finset.univ.filter fun k : ZMod P × ZMod P => (∃ r, residues (xs r) ≠ residues (ys r)) ∧
      digest k.1 k.2 xs = digest k.1 k.2 ys).card ≤ (W - 1 + (R - 1)) * P := by
    have hev : (Finset.univ.filter fun k : ZMod P × ZMod P => (∃ r, residues (xs r) ≠ residues (ys r)) ∧
        digest k.1 k.2 xs = digest k.1 k.2 ys) =
        Finset.univ.filter fun k : ZMod P × ZMod P => digest k.1 k.2 xs = digest k.1 k.2 ys := by
      refine Finset.filter_congr fun k _ => ?_
      exact ⟨fun h => h.2, fun h => ⟨⟨r0, hr0⟩, h⟩⟩
    rw [hev, Finset.card_filter, Fintype.sum_prod_type]
    calc ∑ g : ZMod P, ∑ z : ZMod P, (if digest g z xs = digest g z ys then 1 else 0)
        = ∑ g : ZMod P, (Finset.univ.filter fun z : ZMod P => digest g z xs = digest g z ys).card := by
          refine Finset.sum_congr rfl fun g _ => ?_
          rw [Finset.card_filter]
      _ ≤ ∑ g : ZMod P, ((if g ∈ S then P else 0) + (R - 1)) := Finset.sum_le_sum fun g _ => hslice g
      _ = S.card * P + P * (R - 1) := by
          rw [Finset.sum_add_distrib, Finset.sum_ite_mem, Finset.univ_inter, Finset.sum_const, Finset.sum_const,
            smul_eq_mul, smul_eq_mul, Finset.card_univ, card_zmod]
      _ ≤ (W - 1) * P + P * (R - 1) := by gcongr
      _ = (W - 1 + (R - 1)) * P := by ring
  calc prob (PMF.uniformOfFintype (ZMod P × ZMod P)) _
      ≤ (((W - 1 + (R - 1)) * P : ℕ) : ℝ≥0∞) / Fintype.card (ZMod P × ZMod P) := prob_uniform_le _ _ hcount
    _ = ((W - 1 + (R - 1) : ℕ) : ℝ≥0∞) / P := by
        rw [hcardP]
        push_cast
        exact ENNReal.mul_div_mul_right _ _ (by norm_num) (ENNReal.natCast_ne_top P)

/-! ## The Horner loop -/

theorem hornerIsBlockHash : HornerIsBlockHash := by
  intro W g x
  induction W with
  | zero => simp [hornerHash, blockHash]
  | succ W ih =>
    rw [List.ofFn_succ', hornerHash, List.concat_eq_append, List.foldl_append, List.foldl_cons, List.foldl_nil, ← hornerHash, ih, blockHash, blockHash,
      Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last, Nat.add_sub_cancel, Nat.sub_self, pow_zero, mul_one]
    rw [Finset.sum_mul]
    congr 1
    refine Finset.sum_congr rfl fun j _ => ?_
    have hj := j.isLt
    rw [mul_assoc, ← pow_succ]
    congr 2
    omega

end Accounting.TagProofs
