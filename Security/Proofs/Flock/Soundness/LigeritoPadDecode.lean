import Proofs.Flock.Soundness.LigeritoPadLevel0

/-!
# Decoding the padded level-0 column

`decode_pad` is `decode` for the padded level 0. If no fold round and no extra coin was an MCA
event, and rep `r`'s padded column agrees with the encoding of `G` on a set `T` of at least
`(1 − γ')·2^d` positions, then `G` is the next-level message of a row family in rep `r`'s list:
* the target `Enc(G) − padCorr(ybar)` is a codeword of the padded code, so every row agrees with a
  padded codeword on `T` (`agree_of_extras`, then `rowsAgreeOn_of_foldAll`), and the rows form a
  family in the list;
* the family's combined encoding and the target agree on `T`, and `|T| ≥ 2^cols + t`, so their
  coefficients agree: at `x < 2^cols` that is `G`, at `2^cols + j` it is `ybar`.
-/

namespace FlockSoundness

open Game Model Accounting
open CodingTheory CoreDefinitions
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem encKN_sub (A : Arith F K) (d k : ℕ) (b b' : ℕ → K) (q : ℕ) :
    encKN A d k (fun c => b c - b' c) q = encKN A d k b q - encKN A d k b' q := by
  unfold encKN
  rw [← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- Agreement on `(1 − γ')·2^d` positions covers at least `2^cols + t` of them. -/
theorem card_ge_of_padAgree (l : Level) (t : ℕ) (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen)
    (T : Finset (Fin (2 ^ l.logLen)))
    (hT : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.padRadius t)) :
    2 ^ l.logCols + t ≤ T.card := by
  have hr : 0 < l.padRate t := padRate_pos l t
  have hr1 : l.padRate t ≤ 1 := by
    unfold Level.padRate
    rw [div_le_one (by positivity)]
    exact_mod_cast hk
  have hsq : l.padRate t ≤ Real.sqrt (l.padRate t) :=
    calc l.padRate t = Real.sqrt (l.padRate t ^ 2) := (Real.sqrt_sq hr.le).symm
      _ ≤ Real.sqrt (l.padRate t) := Real.sqrt_le_sqrt (by nlinarith)
  have hlen : (2 : ℝ) ^ l.logLen * l.padRate t = ((2 ^ l.logCols + t : ℕ) : ℝ) := by
    unfold Level.padRate
    field_simp
  rw [Fintype.card_fin] at hT
  push_cast at hT
  have h1 : 1 - l.padRadius t = Real.sqrt (l.padRate t) + Accounting.eta := by
    unfold Level.padRadius; ring
  rw [h1] at hT
  have heta : (0 : ℝ) < Accounting.eta := by unfold Accounting.eta; norm_num
  have : ((2 ^ l.logCols + t : ℕ) : ℝ) ≤ T.card := by
    have h2 : (0 : ℝ) < 2 ^ l.logLen := by positivity
    nlinarith
  exact_mod_cast this

/-- **Decoding the padded column.** -/
theorem decode_pad (A : Arith F K) (hA : A.Correct) (l : Level) (t : ℕ) (hd : l.logLen ≤ 40)
    (hd64 : l.logLen ≤ 64) (hcols : l.logCols ≤ l.logLen) (ht : t ≤ 2 ^ l.logCols)
    (hk : 2 ^ l.logCols + t ≤ 2 ^ l.logLen) (c₀ : F) (k0 e r : ℕ) (C : Oracle F) (ch : List K)
    (hch : ch.length = k0) (ρ : Fin e → K) (ybar : Fin t → K)
    (hgood : MCAGood (codePad A hA l t hd64) (l.padRadius t) k0 ch (rowsK A k0 l.logLen C))
    (hx : ExtrasGood (codePad A hA l t hd64) (l.padRadius t) e ρ (xrowsK A k0 e r l.logLen C)
      (foldAllRows k0 ch (rowsK A k0 l.logLen C)))
    (G : ℕ → K) (T : Finset (Fin (2 ^ l.logLen)))
    (hT : (T.card : ℝ) ≥ Fintype.card (Fin (2 ^ l.logLen)) * (1 - l.padRadius t))
    (hag : ∀ q ∈ T, padColumn A l.logLen l.logCols t c₀ ch ρ (extraLane k0 e r) ybar C q.val =
      encK A l.logCols G q.val) :
    ∃ M ∈ padList A l t k0 e r C, ∀ x < 2 ^ l.logCols, G x = padFold A c₀ ch ρ M x := by
  classical
  haveI : CharP K 2 := hA.charTwoK
  set L := 2 ^ l.logCols with hL
  set k := L + t with hkdef
  have hk0 : 0 < k := by have := Nat.two_pow_pos l.logCols; omega
  set code := codePad A hA l t hd64
  have hfold : ∀ q : Fin (2 ^ l.logLen), foldAllRows k0 ch (rowsK A k0 l.logLen C) q =
      foldColumn A ch (fun ℓ => C ℓ q.val) := by
    intro q
    rw [foldAllRows_eq_sum k0 ch hch, foldColumn, hch,
      ← Fin.sum_univ_eq_sum_range (fun ℓ => eqList ch ℓ * A.embed (C ℓ q.val))]
    rfl
  -- the target `Enc(G) − padCorr(ybar)`, a padded codeword
  set gN : ℕ → K := fun c => if c < L then G c else 0
  set pN := padCorrN A l.logCols t c₀ ybar
  set wN : ℕ → K := fun c => gN c - pN c
  let w : Fin (2 ^ l.logLen) → K := fun q => encKN A l.logLen k wN q.val
  have hw : w ∈ code := by
    have := encKAt_mem A hA l.logLen k hd hd64 hk hk0 (fun c => wN c.val)
    show (fun q : Fin (2 ^ l.logLen) => encKN A l.logLen k wN q.val) ∈ codePad A hA l t hd64
    unfold codePad
    simpa [encKN_eq_encKAt] using this
  have hwq : ∀ q : ℕ, encKN A l.logLen k wN q =
      encK A l.logCols G q - padCorr A l.logLen l.logCols t c₀ ybar q := by
    intro q
    rw [encKN_sub, encK_eq_encKN A l t hcols G q, padCorr_eq_encKN A l t ht c₀ ybar q]
  have hwT : ∀ q ∈ T, w q = foldAllRows k0 ch (rowsK A k0 l.logLen C) q +
      ∑ i, ρ i * xrowsK A k0 e r l.logLen C i q := by
    intro q hq
    have h := hag q hq
    unfold padColumn at h
    simp only [w, hwq, ← h, hfold, xrowsK]
    ring
  -- every row agrees with a padded codeword on `T`
  obtain ⟨⟨c₁, hc₁, hc₁T⟩, hxs⟩ := agree_of_extras code (l.padRadius t) e ρ _ _ hx T hT
    ⟨w, hw, hwT⟩
  have hrows := rowsAgreeOn_of_foldAll code (l.padRadius t) k0 ch _ hgood T hT ⟨c₁, hc₁, hc₁T⟩
  have hwit : ∀ ℓ : Fin (2 ^ k0), ∃ a : Fin k → F,
      ∀ q ∈ T, encAt A l.logLen k a q.val = C ℓ.val q.val := by
    intro ℓ
    obtain ⟨c, hc, hcT⟩ := hrows ℓ
    exact exists_encAt_of_agree A hA l.logLen k hd hd64 hk hk0 (fun q => C ℓ.val q.val) T c hc
      fun q hq => by rw [hcT q hq]; rfl
  have hext : ∀ i : Fin e, ∃ a : Fin k → F,
      ∀ q ∈ T, encAt A l.logLen k a q.val = C (extraLane k0 e r i) q.val := by
    intro i
    obtain ⟨c, hc, hcT⟩ := hxs i
    exact exists_encAt_of_agree A hA l.logLen k hd hd64 hk hk0
      (fun q => C (extraLane k0 e r i) q.val) T c hc fun q hq => by rw [hcT q hq]; rfl
  choose aw haw using hwit
  choose ax hax using hext
  let M : Fin (2 ^ k0 + e) → Fin k → F := Fin.addCases aw ax
  have hMT : ∀ q ∈ T, ∀ ℓ, padRows k0 e r l.logLen C ℓ q = encAt A l.logLen k (M ℓ) q.val := by
    intro q hq ℓ
    refine Fin.addCases (fun ℓ => ?_) (fun i => ?_) ℓ
    · simp only [M, padRows, padLane, Fin.addCases_left]; exact (haw ℓ q hq).symm
    · simp only [M, padRows, padLane, Fin.addCases_right]; exact (hax i q hq).symm
  have hM : M ∈ padList A l t k0 e r C := by
    rw [padList, Finset.mem_filter]
    refine ⟨Finset.mem_univ _, ?_⟩
    unfold CloseRows
    refine le_trans ?_ ((Nat.cast_le (α := ℝ)).2 (Finset.card_le_card (s := T) fun q hq => ?_))
    · rw [Fintype.card_fin] at hT; push_cast at hT; linarith
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and]; exact hMT q hq
  refine ⟨M, hM, ?_⟩
  -- the family's combined coefficients
  set bM : ℕ → K := fun c => if h : c < k then
      (∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (aw ℓ ⟨c, h⟩)) +
        ∑ i, ρ i * A.embed (ax i ⟨c, h⟩) else 0 with hbM
  have hcomb : ∀ q : ℕ, encKN A l.logLen k bM q =
      (∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (encAt A l.logLen k (aw ℓ) q)) +
        ∑ i, ρ i * A.embed (encAt A l.logLen k (ax i) q) := by
    intro q
    rw [encKN_eq_encKAt]
    unfold encKAt encAt
    simp only [bM, Fin.is_lt, dif_pos, Fin.eta, map_sum, map_mul, add_mul, Finset.sum_add_distrib,
      Finset.sum_mul, Finset.mul_sum]
    congr 1
    · rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
    · rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun _ _ => Finset.sum_congr rfl fun _ _ => by ring
  have hzero := encKN_eq_zero A hA l.logLen k hd hd64 hk (fun c => bM c - wN c) T
    (card_ge_of_padAgree l t hk T hT) fun q hq => by
      rw [encKN_sub, hcomb, sub_eq_zero]
      show _ = w q
      rw [hwT q hq, foldAllRows_eq_sum k0 ch hch]
      simp only [xrowsK, rowsK]
      congr 1
      · refine Finset.sum_congr rfl fun ℓ _ => ?_
        rw [haw ℓ q hq]
      · refine Finset.sum_congr rfl fun i _ => ?_
        rw [hax i q hq]
  -- read off `G`
  intro x hx
  have hxk : x < k := by omega
  have h1 : bM x - wN x = 0 := hzero x hxk
  have hw1 : wN x = G x - (if x < t then ybarN ybar x * A.embed c₀ else 0) := by
    simp only [wN, gN, pN, padCorrN, if_pos (show x < L by omega),
      show ¬ (2 ^ l.logCols ≤ x) from by omega, if_false, add_zero]
  have hy : x < t → ybarN ybar x = - bM (L + x) := by
    intro hxt
    have h2 : bM (L + x) - wN (L + x) = 0 := hzero _ (by omega)
    have hw2 : wN (L + x) = - ybarN ybar x := by
      simp only [wN, gN, pN, padCorrN, show ¬ (L + x < L) from by omega,
        show ¬ (L + x < t) from by omega, show 2 ^ l.logCols ≤ L + x from by omega,
        show L + x - 2 ^ l.logCols = x from by omega, if_true, if_false, zero_add, zero_sub]
    rw [hw2] at h2
    linear_combination h2
  have hbx : bM x = (∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (aw ℓ (Fin.castAdd t ⟨x, hx⟩))) +
      ∑ i, ρ i * A.embed (ax i (Fin.castAdd t ⟨x, hx⟩)) := by
    simp only [hbM, dif_pos hxk]
    rfl
  have hpf : padFold A c₀ ch ρ M x =
      (∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (padMsgAt c₀ (aw ℓ) ⟨x, hx⟩)) +
        ∑ i, ρ i * A.embed (padMsgAt c₀ (ax i) ⟨x, hx⟩) := by
    rw [padFold, foldMsg_eq A ch hch (padWit c₀ M) ⟨x, hx⟩]
    simp only [dif_pos hx, padWit, M, Fin.addCases_left, Fin.addCases_right]
  rw [hpf]
  by_cases hxt : x < t
  · have hLx : L + x < k := by omega
    have hbLx : bM (L + x) =
        (∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (aw ℓ ⟨L + x, hLx⟩)) +
          ∑ i, ρ i * A.embed (ax i ⟨L + x, hLx⟩) := by
      simp only [hbM, dif_pos hLx]
    have hG : G x = bM x - A.embed c₀ * bM (L + x) := by
      rw [hw1, if_pos hxt, hy hxt] at h1
      linear_combination -h1
    have key : ∀ a : Fin k → F, A.embed (padMsgAt c₀ a ⟨x, hx⟩) =
        A.embed (a (Fin.castAdd t ⟨x, hx⟩)) - A.embed c₀ * A.embed (a ⟨L + x, hLx⟩) := by
      intro a
      simp only [padMsgAt, dif_pos hxt, map_sub, map_mul]
      rfl
    simp only [key, mul_sub, Finset.sum_sub_distrib]
    have hs1 : ∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * (A.embed c₀ * A.embed (aw ℓ ⟨L + x, hLx⟩)) =
        A.embed c₀ * ∑ ℓ : Fin (2 ^ k0), eqList ch ℓ.val * A.embed (aw ℓ ⟨L + x, hLx⟩) := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
    have hs2 : ∑ i, ρ i * (A.embed c₀ * A.embed (ax i ⟨L + x, hLx⟩)) =
        A.embed c₀ * ∑ i, ρ i * A.embed (ax i ⟨L + x, hLx⟩) := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun _ _ => by ring
    rw [hs1, hs2, hG, hbx, hbLx]
    ring
  · have hG : G x = bM x := by
      rw [hw1, if_neg hxt] at h1
      linear_combination -h1
    rw [hG, hbx]
    simp only [padMsgAt, dif_neg hxt, sub_zero]

end FlockSoundness
