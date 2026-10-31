import Definitions.Pouw.Dimension.Ssn
import Proofs.Pouw.Dimension.H100FloorProofs

/-!
# Served sparse noise in the bit model: distinctness and shift separation (proofs)

**Draft semantics, pending review** (`MH100w`, `H100W.lean`). Every bound here is for one `MH100w` program fixed
before the coins and right on every coin value.

* **The floor charges every target** (`charged`): at `h100Floor`, a well-formed program on free inputs that matches a
  target set distinct and non-free on `S` costs at least 32 a target (`distinctWritesFloor`).
* **The full grid** (`ssnFull`): when every step's weights reach every column, step 0's tell the columns apart, the
  used lanes read distinct bits and no weight is `2^31`, the `m·n·S` committed words are pairwise distinct and none is
  free. Each word's coefficient at its row's last weighed bit is its own (`rows_win`, `rows_win0`). `ssnFull_shift`
  makes that shift-separated: no two words differ by a constant and none is a free value plus a constant, the
  hypotheses of carrier separation.
* **The served layouts.** Both are injective (`l4_inj`, `l5_inj`), and their codes tell the columns apart: the 4-lane
  code for `n ≤ 28,559` (`ssnCode_inj`) and the 5-lane balanced code for `n ≤ 371,292` (`code5_inj`). So the served
  words are distinct (`ssn1lServed_full`, `ssn1l5Served_full`), charged (`ssn1lServedB_charged`,
  `ssn1l5Served_charged`) and shift-separated (`ssn1lServed_shift`, `ssn1l5Served_shift`).
-/

namespace Pouw.Dimension.Ssn

open Finset Pouw.Dimension.H100

section Generic

variable {G : Type}

/-- The image of a family that differs pairwise on `S` has one element per index. -/
theorem card_image_of_differ {ι : Type} [DecidableEq (G → ℚ)] {S : Set G} (s : Finset ι) (t : ι → G → ℚ)
    (h : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → ¬ ∀ g ∈ S, t i g = t j g) : (s.image t).card = s.card := by
  classical
  refine Finset.card_image_of_injOn fun i hi j hj hij => ?_
  by_contra hne
  exact h i hi j hj hne fun g _ => by rw [hij]

/-- **The floor charges every target**: at `h100Floor` (the H200's measured table), a well-formed `MH100w` program on
free inputs that matches on `S` a target set distinct and non-free on `S` costs at least 32 a target. -/
theorem charged (F : Set (G → ℚ)) (hF : ∀ q : ℚ, (fun _ : G => q) ∈ F) (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
    (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)) (hL : ∀ f ∈ L, f ∈ F)
    (hP : ∀ o ∈ P, o.WFw h100Floor F) (hT : DistinctOn S F T) (hO : OutputsOn fpSem S P L T) :
    32 * T.card ≤ cost100 h100Floor P :=
  distinctWritesFloor G F hF S fpSem L P T hL hP hT hO

end Generic

section Affine

variable {ι : Type} [Fintype ι] {q : ℕ}

/-- Affine functions of the digits that agree everywhere have the same coefficients. -/
theorem coef_eq_of_aff_eq [DecidableEq ι] (hq : 1 < q) {c c' : ℚ} {a a' : ι → ℚ} (h : aff (q := q) c a = aff c' a') (k : ι) :
    a k = a' k := by
  have h0 := congrFun h fun _ => ⟨0, by omega⟩
  have h1 := congrFun h fun k' => ⟨if k' = k then 1 else 0, by split_ifs <;> omega⟩
  simp [aff] at h0 h1
  linarith

end Affine

section Bits

variable {m S C L : ℕ}

theorem fin0_le (hS : 0 < S) (t : Fin S) : (⟨0, hS⟩ : Fin S) ≤ t := Fin.le_iff_val_le_val.2 (Nat.zero_le _)

/-- A word's rank: row, then column, then step. -/
def rk (n : ℕ) (u : Fin m × ℕ × Fin S) : ℕ := (u.1 : ℕ) * (n * S) + (u.2.1 * S + u.2.2)

theorem lex2 {N a b a' b' : ℕ} (hb : b < N) (h : a' * N + b' < a * N + b) : a' ≤ a ∧ (a' = a → b' < b) := by
  refine ⟨?_, fun he => by subst he; omega⟩
  by_contra hlt
  have h1 : (a + 1) * N ≤ a' * N := Nat.mul_le_mul_right N (by omega)
  have h2 : (a + 1) * N = a * N + N := Nat.succ_mul a N
  omega

theorem lex2_eq {N a b a' b' : ℕ} (hb : b < N) (hb' : b' < N) (h : a * N + b = a' * N + b') : a = a' ∧ b = b' := by
  rcases lt_trichotomy a a' with hl | rfl | hl
  · have h1 : (a + 1) * N ≤ a' * N := Nat.mul_le_mul_right N (Nat.succ_le_of_lt hl)
    have h2 : (a + 1) * N = a * N + N := Nat.succ_mul a N
    omega
  · omega
  · have h1 : (a' + 1) * N ≤ a * N := Nat.mul_le_mul_right N (Nat.succ_le_of_lt hl)
    have h2 : (a' + 1) * N = a' * N + N := Nat.succ_mul a' N
    omega

theorem rank_lt {N a B b : ℕ} (ha : a < N) (hb : b < B) : a * B + b < N * B := by
  have h1 : (a + 1) * B ≤ N * B := Nat.mul_le_mul_right B (Nat.succ_le_of_lt ha)
  have h2 : (a + 1) * B = a * B + B := Nat.succ_mul a B
  omega

theorem rk_inj {n : ℕ} {u u' : Fin m × ℕ × Fin S} (hj : u.2.1 < n) (hj' : u'.2.1 < n) (h : rk n u = rk n u') :
    u = u' := by
  obtain ⟨i, j, t⟩ := u
  obtain ⟨i', j', t'⟩ := u'
  simp only [rk] at h hj hj'
  obtain ⟨hi, hr⟩ := lex2_eq (rank_lt hj t.isLt) (rank_lt hj' t'.isLt) h
  obtain ⟨hjj, htt⟩ := lex2_eq t.isLt t'.isLt hr
  rw [Fin.ext hi, hjj, Fin.ext htt]

variable {src : Fin S → Fin C} {used : Fin S → Fin L → Prop} [∀ s q, Decidable (used s q)]
  {pos : Fin S → Fin L → ℕ}

theorem bitCoef_row (W : Fin S → Fin L → ℕ → ℤ) {i : Fin m} (j : ℕ) (t : Fin S) {k : Fin m × Fin C × Fin 32}
    (hk : k.1 ≠ i) : bitCoef src used pos W i j t k = 0 := by
  simp [bitCoef, hk]

theorem bitCoef_at (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ) (t : Fin S) {s₀ : Fin S} {q₀ : Fin L} (hu : used s₀ q₀) :
    bitCoef src used pos W i j t (i, src s₀, ⟨pos s₀ q₀, hpos s₀ q₀⟩) = if s₀ ≤ t then (W s₀ q₀ j : ℚ) else 0 := by
  unfold bitCoef
  dsimp only
  rw [if_pos rfl, Finset.sum_eq_single s₀, Finset.sum_eq_single q₀]
  · simp [hu]
  · intro q _ hq
    rw [if_neg]
    rintro ⟨-, hu', -, hp⟩
    exact hq (hinj s₀ q s₀ q₀ hu' hu rfl hp).2
  · intro h
    exact absurd (Finset.mem_univ _) h
  · intro s _ hs
    refine Finset.sum_eq_zero fun q _ => ?_
    rw [if_neg]
    rintro ⟨-, hu', hsrc, hp⟩
    exact hs (hinj s q s₀ q₀ hu' hu hsrc hp).1
  · intro h
    exact absurd (Finset.mem_univ _) h

theorem bitCoef_none (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ) (t : Fin S) (c : Fin C) (b : Fin 32)
    (h : ∀ s q, used s q → src s = c → pos s q = b → False) : bitCoef src used pos W i j t (i, c, b) = 0 := by
  unfold bitCoef
  dsimp only
  rw [if_pos rfl]
  refine Finset.sum_eq_zero fun s _ => Finset.sum_eq_zero fun q _ => ?_
  rw [if_neg]
  rintro ⟨-, hu, hs, hp⟩
  exact h s q hu hs hp

/-- A word differs from every earlier-ranked word at a bit of its own row, and both are `0` on later rows. -/
theorem rows_win (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q) (hS : 0 < S) (n : ℕ)
    (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0)
    (hW2 : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j')
    {u u' : Fin m × ℕ × Fin S} (hj : u.2.1 < n) (hj' : u'.2.1 < n) (hlt : rk n u' < rk n u) :
    ∃ k : Fin m × Fin C × Fin 32, k.1 = u.1 ∧
      bitCoef src used pos W u.1 u.2.1 u.2.2 k ≠ bitCoef src used pos W u'.1 u'.2.1 u'.2.2 k ∧
      ∀ k' : Fin m × Fin C × Fin 32, (u.1 : ℕ) < k'.1 →
        bitCoef src used pos W u.1 u.2.1 u.2.2 k' = bitCoef src used pos W u'.1 u'.2.1 u'.2.2 k' := by
  obtain ⟨i, j, t⟩ := u
  obtain ⟨i', j', t'⟩ := u'
  simp only [rk] at hlt hj hj' ⊢
  obtain ⟨hii, hii'⟩ := lex2 (rank_lt hj t.isLt) hlt
  have htl : ∀ k' : Fin m × Fin C × Fin 32, (i : ℕ) < k'.1 →
      bitCoef src used pos W i j t k' = bitCoef src used pos W i' j' t' k' := by
    intro k' hk
    rw [bitCoef_row W _ _ (fun h => by rw [h] at hk; exact lt_irrefl _ hk),
      bitCoef_row W _ _ (fun h => by rw [h] at hk; omega)]
  suffices hk : ∃ k : Fin m × Fin C × Fin 32, k.1 = i ∧
      bitCoef src used pos W i j t k ≠ bitCoef src used pos W i' j' t' k by
    obtain ⟨k, hk1, hk2⟩ := hk
    exact ⟨k, hk1, hk2, htl⟩
  by_cases hi : i' = i
  · subst hi
    obtain ⟨-, hjj'⟩ := lex2 t.isLt (hii' rfl)
    by_cases hj0 : j' = j
    · subst hj0
      have ht : t' < t := Fin.lt_iff_val_lt_val.2 (hjj' rfl)
      obtain ⟨q, hq⟩ := hW1 t j' (Finset.mem_range.2 hj)
      refine ⟨(_, src t, ⟨pos t q, hpos t q⟩), rfl, ?_⟩
      rw [bitCoef_at hpos hinj W _ _ _ (hU _ _ _ hq), bitCoef_at hpos hinj W _ _ _ (hU _ _ _ hq), if_pos le_rfl,
        if_neg (not_le.2 ht)]
      exact_mod_cast hq
    · obtain ⟨q, hq⟩ := hW2 j (Finset.mem_range.2 hj) j' (Finset.mem_range.2 hj') (Ne.symm hj0)
      have hu0 : used ⟨0, hS⟩ q := by
        by_cases h0 : W ⟨0, hS⟩ q j = 0
        · exact hU _ _ j' (fun h' => hq (by rw [h0, h']))
        · exact hU _ _ j h0
      refine ⟨(_, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ _⟩), rfl, ?_⟩
      rw [bitCoef_at hpos hinj W _ _ _ hu0, bitCoef_at hpos hinj W _ _ _ hu0, if_pos (fin0_le hS t),
        if_pos (fin0_le hS t')]
      exact_mod_cast hq
  · obtain ⟨q, hq⟩ := hW1 ⟨0, hS⟩ j (Finset.mem_range.2 hj)
    refine ⟨(i, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ _⟩), rfl, ?_⟩
    rw [bitCoef_at hpos hinj W _ _ _ (hU _ _ _ hq), if_pos (fin0_le hS t),
      bitCoef_row W _ _ (show (i, src ⟨0, hS⟩, (⟨pos ⟨0, hS⟩ q, hpos _ _⟩ : Fin 32)).1 ≠ i' from fun h => hi h.symm)]
    exact_mod_cast hq

/-- A word is nonzero at a bit of its own row and `0` on later rows. -/
theorem rows_win0 (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q) (hS : 0 < S) (n : ℕ)
    (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0) {u : Fin m × ℕ × Fin S} (hj : u.2.1 < n) :
    ∃ k : Fin m × Fin C × Fin 32, k.1 = u.1 ∧ bitCoef src used pos W u.1 u.2.1 u.2.2 k ≠ 0 ∧
      ∀ k' : Fin m × Fin C × Fin 32, (u.1 : ℕ) < k'.1 → bitCoef src used pos W u.1 u.2.1 u.2.2 k' = 0 := by
  obtain ⟨i, j, t⟩ := u
  obtain ⟨q, hq⟩ := hW1 ⟨0, hS⟩ j (Finset.mem_range.2 hj)
  refine ⟨(i, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ _⟩), rfl, ?_,
    fun k' hk => bitCoef_row W _ _ (fun h => by rw [h] at hk; exact lt_irrefl _ hk)⟩
  rw [bitCoef_at hpos hinj W _ _ _ (hU _ _ _ hq), if_pos (fin0_le hS t)]
  exact_mod_cast hq

/-- **SSN's words on the full grid**: as `ssnRows` with no weight `2^31`, all `m·n·S` words are pairwise distinct
and none is free. -/
theorem ssnFull (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q)
    (hS : 0 < S) (n : ℕ) (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0)
    (hW2 : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j')
    (hW31 : ∀ s q j, W s q j ≠ 2 ^ 31) :
    DistinctOn Set.univ (FreeB m C) ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
        ssnWordB src used pos κ W x.1 x.2.1 x.2.2) ∧
      ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
        ssnWordB src used pos κ W x.1 x.2.1 x.2.2).card = m * (n * S) := by
  classical
  have mem : ∀ u : Fin m × ℕ × Fin S, u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)) ↔ u.2.1 < n := by
    intro u
    simp
  have hne : ∀ u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)),
      ∀ u' ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)), u ≠ u' →
      ¬ ∀ x ∈ (Set.univ : Set (Bits m C)),
        ssnWordB src used pos κ W u.1 u.2.1 u.2.2 x = ssnWordB src used pos κ W u'.1 u'.2.1 u'.2.2 x := by
    intro u hu u' hu' huu heq
    have hf : ssnWordB src used pos κ W u.1 u.2.1 u.2.2 = ssnWordB src used pos κ W u'.1 u'.2.1 u'.2.2 :=
      funext fun x => heq x trivial
    have hrk : rk n u ≠ rk n u' := fun h => huu (rk_inj ((mem u).1 hu) ((mem u').1 hu') h)
    rcases lt_or_gt_of_ne hrk with h | h
    · obtain ⟨k, -, hk, -⟩ := rows_win hpos hinj W hU hS n hW1 hW2 ((mem u').1 hu') ((mem u).1 hu) h
      exact hk (coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ u'.1 u'.2.1 u'.2.2 : ℚ))
        (a := bitCoef src used pos W u'.1 u'.2.1 u'.2.2) (c' := (κ u.1 u.2.1 u.2.2 : ℚ))
        (a' := bitCoef src used pos W u.1 u.2.1 u.2.2) hf.symm k)
    · obtain ⟨k, -, hk, -⟩ := rows_win hpos hinj W hU hS n hW1 hW2 ((mem u).1 hu) ((mem u').1 hu') h
      exact hk (coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ u.1 u.2.1 u.2.2 : ℚ))
        (a := bitCoef src used pos W u.1 u.2.1 u.2.2) (c' := (κ u'.1 u'.2.1 u'.2.2 : ℚ))
        (a' := bitCoef src used pos W u'.1 u'.2.1 u'.2.2) hf k)
  refine ⟨⟨fun w hw w' hw' h => ?_, fun w hw f hf heq => ?_⟩, ?_⟩
  · obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨u', hu', rfl⟩ := Finset.mem_image.1 hw'
    by_contra huu
    exact hne u hu u' hu' (fun h' => huu (by rw [h'])) h
  · obtain ⟨⟨i, j, t⟩, hu, rfl⟩ := Finset.mem_image.1 hw
    have hj : j < n := (mem _).1 hu
    have hf' : ssnWordB src used pos κ W i j t = f := funext fun x => heq x trivial
    rcases hf with ⟨c, rfl⟩ | ⟨r, c, rfl⟩
    · obtain ⟨k, -, hk, -⟩ := rows_win0 hpos hinj W hU hS n hW1 (u := (i, j, t)) hj
      refine hk (coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ i j t : ℚ)) (a := bitCoef src used pos W i j t)
        (c' := c) (a' := fun _ => 0) (hf'.trans (funext fun x => ?_)) k)
      simp [aff]
    · have hk := coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ i j t : ℚ)) (a := bitCoef src used pos W i j t)
        (c' := 0) (a' := coinCoef r c) hf' (r, c, ⟨31, by norm_num⟩)
      simp only [coinCoef, and_self, if_true] at hk
      by_cases hri : r = i
      · subst hri
        by_cases hown : ∃ s₀ q₀, used s₀ q₀ ∧ src s₀ = c ∧ pos s₀ q₀ = 31
        · obtain ⟨s₀, q₀, hu₀, rfl, hp⟩ := hown
          have hpt : (⟨31, by norm_num⟩ : Fin 32) = ⟨pos s₀ q₀, hpos s₀ q₀⟩ := Fin.ext hp.symm
          rw [hpt, bitCoef_at hpos hinj W _ _ _ hu₀] at hk
          split_ifs at hk
          · exact hW31 s₀ q₀ j (by exact_mod_cast hk)
          · norm_num at hk
        · rw [bitCoef_none W _ j t c _ (fun s q hu hs hp => hown ⟨s, q, hu, hs, hp⟩)] at hk
          norm_num at hk
      · rw [bitCoef_row W j t (show (r, c, (⟨31, by norm_num⟩ : Fin 32)).1 ≠ i from hri)] at hk
        norm_num at hk
  · rw [card_image_of_differ _ _ hne]
    simp [Finset.card_product]

theorem aff_shift {ι : Type} [Fintype ι] {c c' q : ℚ} {a a' : ι → ℚ}
    (h : ∀ g ∈ (Set.univ : Set (ι → Fin 2)), aff (q := 2) c a g = aff c' a' g + q) :
    aff (q := 2) c a = aff (c' + q) a' := by
  funext g
  have := h g trivial
  simp only [aff] at this ⊢
  linarith

/-- **SSN's words are shift-separated on the full grid**: under `ssnFull`'s hypotheses no two words differ by a
constant, and no word is a free value plus a constant. These are the two hypotheses of carrier separation
(`SepBy (Carries Set.univ) (FreeB m C) T`): a register that carries one word carries no other and no free value. -/
theorem ssnFull_shift (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q)
    (hS : 0 < S) (n : ℕ) (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0)
    (hW2 : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j')
    (hW31 : ∀ s q j, W s q j ≠ 2 ^ 31) :
    (∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (C := C) src used pos κ W x.1 x.2.1 x.2.2),
      ∀ t' ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (C := C) src used pos κ W x.1 x.2.1 x.2.2),
      (∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), t g = t' g + q) → t = t') ∧
    ∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (C := C) src used pos κ W x.1 x.2.1 x.2.2),
      ∀ f ∈ FreeB m C, ¬ ∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), f g = t g + q := by
  classical
  have mem : ∀ u : Fin m × ℕ × Fin S, u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)) ↔ u.2.1 < n := by
    intro u
    simp
  refine ⟨fun w hw w' hw' ⟨q, hq⟩ => ?_, fun w hw f hf ⟨q, hq⟩ => ?_⟩
  · obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hw
    obtain ⟨u', hu', rfl⟩ := Finset.mem_image.1 hw'
    have hf := aff_shift hq
    by_cases huu : u = u'
    · rw [huu]
    exfalso
    have hrk : rk n u ≠ rk n u' := fun h => huu (rk_inj ((mem u).1 hu) ((mem u').1 hu') h)
    have hc := fun k => coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ u.1 u.2.1 u.2.2 : ℚ))
      (a := bitCoef src used pos W u.1 u.2.1 u.2.2) (c' := (κ u'.1 u'.2.1 u'.2.2 : ℚ) + q)
      (a' := bitCoef src used pos W u'.1 u'.2.1 u'.2.2) hf k
    rcases lt_or_gt_of_ne hrk with h | h
    · obtain ⟨k, -, hk, -⟩ := rows_win hpos hinj W hU hS n hW1 hW2 ((mem u').1 hu') ((mem u).1 hu) h
      exact hk (hc k).symm
    · obtain ⟨k, -, hk, -⟩ := rows_win hpos hinj W hU hS n hW1 hW2 ((mem u).1 hu) ((mem u').1 hu') h
      exact hk (hc k)
  · obtain ⟨⟨i, j, t⟩, hu, rfl⟩ := Finset.mem_image.1 hw
    have hj : j < n := (mem _).1 hu
    have hq' : ∀ g ∈ (Set.univ : Set (Bits m C)), ssnWordB src used pos κ W i j t g = f g + -q := fun g hg => by
      rw [hq g hg]
      ring
    rcases hf with ⟨c, rfl⟩ | ⟨r, c, rfl⟩
    · obtain ⟨k, -, hk, -⟩ := rows_win0 hpos hinj W hU hS n hW1 (u := (i, j, t)) hj
      have hf' : ssnWordB src used pos κ W i j t = aff (q := 2) (c + -q) (fun _ => 0) := by
        funext x
        rw [hq' x trivial]
        simp [aff]
      exact hk (coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ i j t : ℚ)) (a := bitCoef src used pos W i j t)
        (c' := c + -q) (a' := fun _ => 0) hf' k)
    · have hf' := aff_shift (c := (κ i j t : ℚ)) (a := bitCoef src used pos W i j t) (c' := 0)
        (a' := coinCoef r c) hq'
      have hk := coef_eq_of_aff_eq (q := 2) (by norm_num) (c := (κ i j t : ℚ)) (a := bitCoef src used pos W i j t)
        (c' := 0 + -q) (a' := coinCoef r c) hf' (r, c, ⟨31, by norm_num⟩)
      simp only [coinCoef, and_self, if_true] at hk
      by_cases hri : r = i
      · subst hri
        by_cases hown : ∃ s₀ q₀, used s₀ q₀ ∧ src s₀ = c ∧ pos s₀ q₀ = 31
        · obtain ⟨s₀, q₀, hu₀, rfl, hp⟩ := hown
          have hpt : (⟨31, by norm_num⟩ : Fin 32) = ⟨pos s₀ q₀, hpos s₀ q₀⟩ := Fin.ext hp.symm
          rw [hpt, bitCoef_at hpos hinj W _ _ _ hu₀] at hk
          split_ifs at hk
          · exact hW31 s₀ q₀ j (by exact_mod_cast hk)
          · norm_num at hk
        · rw [bitCoef_none W _ j t c _ (fun s q hu hs hp => hown ⟨s, q, hu, hs, hp⟩)] at hk
          norm_num at hk
      · rw [bitCoef_row W j t (show (r, c, (⟨31, by norm_num⟩ : Fin 32)).1 ≠ i from hri)] at hk
        norm_num at hk

end Bits

section Served

variable {m S C : ℕ}

theorem l4_pos_lt (s : Fin S) (q : Fin 4) : l4Pos s q < 32 := by
  have := q.isLt
  unfold l4Pos
  split_ifs <;> omega

theorem l4_inj (hC : S + 3 ≤ 32 * C) : ∀ (s : Fin S) (q : Fin 4) (s' : Fin S) (q' : Fin 4),
    ((s : ℕ) = 0 ∨ q = 0) → ((s' : ℕ) = 0 ∨ q' = 0) → l4Src hC s = l4Src hC s' → l4Pos s q = l4Pos s' q' →
      s = s' ∧ q = q' := by
  intro s q s' q' hu hu' hw hp
  have hq := q.isLt
  have hq' := q'.isLt
  simp only [Fin.ext_iff, l4Src, l4Word, l4Pos, Fin.val_zero] at hu hu' hw hp ⊢
  split_ifs at hw hp <;> omega

theorem l5_pos_lt (s : Fin S) (q : Fin 5) : l5Pos s q < 32 := by
  have := q.isLt
  unfold l5Pos
  split_ifs <;> omega

theorem l5_inj (hC : S + 4 ≤ 32 * C) : ∀ (s : Fin S) (q : Fin 5) (s' : Fin S) (q' : Fin 5),
    ((s : ℕ) = 0 ∨ q = 0) → ((s' : ℕ) = 0 ∨ q' = 0) → l5Src hC s = l5Src hC s' → l5Pos s q = l5Pos s' q' →
      s = s' ∧ q = q' := by
  intro s q s' q' hu hu' hw hp
  have hq := q.isLt
  have hq' := q'.isLt
  simp only [Fin.ext_iff, l5Src, l5Word, l5Pos, Fin.val_zero] at hu hu' hw hp ⊢
  split_ifs at hw hp <;> omega

theorem ssnDigit_eq (c : ℕ) : ssnDigit c 0 = c % 13 ∧ ssnDigit c 1 = c / 13 % 13 ∧ ssnDigit c 2 = c / 169 % 13 ∧
    ssnDigit c 3 = c / 2197 % 13 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> simp [ssnDigit]

theorem ssnCode_ne_zero {j : ℕ} (hj : j < 28559) : ∃ q, ssnCode q j ≠ 0 := by
  by_contra h
  push_neg at h
  obtain ⟨d0, d1, d2, d3⟩ := ssnDigit_eq (ssnC j)
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  have h3 := h 3
  simp only [ssnCode] at h0 h1 h2 h3
  unfold ssnC at d0 d1 d2 d3 h0 h1 h2 h3
  split_ifs at d0 d1 d2 d3 h0 h1 h2 h3 <;> omega

theorem ssnCode_inj {j j' : ℕ} (hj : j < 28559) (hj' : j' < 28559) (hne : j ≠ j') :
    ∃ q, ssnCode q j ≠ ssnCode q j' := by
  by_contra h
  push_neg at h
  obtain ⟨d0, d1, d2, d3⟩ := ssnDigit_eq (ssnC j)
  obtain ⟨d0', d1', d2', d3'⟩ := ssnDigit_eq (ssnC j')
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  have h3 := h 3
  simp only [ssnCode] at h0 h1 h2 h3
  unfold ssnC at d0 d1 d2 d3 d0' d1' d2' d3' h0 h1 h2 h3
  split_ifs at d0 d1 d2 d3 d0' d1' d2' d3' h0 h1 h2 h3 <;> omega

theorem ssnCode_le (e : Fin 4) (j : ℕ) : ssnCode e j ≤ 6 := by
  have h := Nat.mod_lt (ssnC j / 13 ^ (e : ℕ)) (by norm_num : 13 > 0)
  simp only [ssnCode, ssnDigit]
  generalize ssnC j / 13 ^ (e : ℕ) % 13 = d at h ⊢
  omega

theorem w1l_reach (n : ℕ) (hn : n ≤ 28559) : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, w1l s q j ≠ 0 := by
  intro s j hj
  have hj' : j < 28559 := by have := Finset.mem_range.1 hj; omega
  by_cases hs : (s : ℕ) = 0
  · obtain ⟨q, hq⟩ := ssnCode_ne_zero hj'
    exact ⟨q, by simp only [w1l, if_pos hs]; exact hq⟩
  · exact ⟨0, by simp [w1l, hs]⟩

theorem code_cols (hS : 0 < S) (W : Fin S → Fin 4 → ℕ → ℤ) (hW : ∀ q j, W ⟨0, hS⟩ q j = ssnCode q j) (n : ℕ)
    (hn : n ≤ 28559) : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j' := by
  intro j hj j' hj' hne
  obtain ⟨q, hq⟩ := ssnCode_inj (by have := Finset.mem_range.1 hj; omega) (by have := Finset.mem_range.1 hj'; omega) hne
  exact ⟨q, by rw [hW, hW]; exact hq⟩

theorem w1l_ne (s : Fin S) (q : Fin 4) (j : ℕ) : w1l s q j ≠ 2 ^ 31 := by
  have := ssnCode_le q j
  unfold w1l
  split_ifs <;> norm_num <;> omega

/-- `ssn_1l` weighs step 0's four lanes and lane 0 of every later step: `S + 3` coin bits a row. -/
theorem w1l_used (s : Fin S) (q : Fin 4) (j : ℕ) (h : w1l s q j ≠ 0) : (s : ℕ) = 0 ∨ q = 0 := by
  unfold w1l at h
  split_ifs at h with h1 h2
  · exact Or.inl h1
  · exact Or.inr h2
  · exact absurd rfl h

theorem digit5_eq (c : ℕ) : c / 13 ^ ((0 : Fin 5) : ℕ) % 13 = c % 13 ∧ c / 13 ^ ((1 : Fin 5) : ℕ) % 13 = c / 13 % 13 ∧
    c / 13 ^ ((2 : Fin 5) : ℕ) % 13 = c / 169 % 13 ∧ c / 13 ^ ((3 : Fin 5) : ℕ) % 13 = c / 2197 % 13 ∧
    c / 13 ^ ((4 : Fin 5) : ℕ) % 13 = c / 28561 % 13 := by
  refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;> simp

theorem code5_ne_zero {j : ℕ} (hj : j < 371292) : ∃ e, code5 e j ≠ 0 := by
  by_contra h
  push_neg at h
  obtain ⟨d0, d1, d2, d3, d4⟩ := digit5_eq (z5 j)
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  have h3 := h 3
  have h4 := h 4
  simp only [code5] at h0 h1 h2 h3 h4
  unfold z5 at d0 d1 d2 d3 d4 h0 h1 h2 h3 h4
  split_ifs at d0 d1 d2 d3 d4 h0 h1 h2 h3 h4 <;> omega

theorem code5_inj {j j' : ℕ} (hj : j < 371292) (hj' : j' < 371292) (hne : j ≠ j') :
    ∃ e, code5 e j ≠ code5 e j' := by
  by_contra h
  push_neg at h
  obtain ⟨d0, d1, d2, d3, d4⟩ := digit5_eq (z5 j)
  obtain ⟨d0', d1', d2', d3', d4'⟩ := digit5_eq (z5 j')
  have h0 := h 0
  have h1 := h 1
  have h2 := h 2
  have h3 := h 3
  have h4 := h 4
  simp only [code5] at h0 h1 h2 h3 h4
  unfold z5 at d0 d1 d2 d3 d4 d0' d1' d2' d3' d4' h0 h1 h2 h3 h4
  split_ifs at d0 d1 d2 d3 d4 d0' d1' d2' d3' d4' h0 h1 h2 h3 h4 <;> omega

theorem code5_le (e : Fin 5) (j : ℕ) : code5 e j ≤ 6 := by
  have h := Nat.mod_lt (z5 j / 13 ^ (e : ℕ)) (by norm_num : 13 > 0)
  simp only [code5]
  generalize z5 j / 13 ^ (e : ℕ) % 13 = d at h ⊢
  omega

theorem w5_used (s : Fin S) (q : Fin 5) (j : ℕ) (h : w5 s q j ≠ 0) : (s : ℕ) = 0 ∨ q = 0 := by
  unfold w5 at h
  split_ifs at h with h1 h2
  · exact Or.inl h1
  · exact Or.inr h2
  · exact absurd rfl h

theorem w5_reach (n : ℕ) (hn : n ≤ 371292) : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, w5 s q j ≠ 0 := by
  intro s j hj
  have hj' : j < 371292 := by have := Finset.mem_range.1 hj; omega
  by_cases hs : (s : ℕ) = 0
  · obtain ⟨q, hq⟩ := code5_ne_zero hj'
    exact ⟨q, by simp only [w5, if_pos hs]; exact hq⟩
  · exact ⟨0, by simp [w5, hs]⟩

theorem w5_cols (hS : 0 < S) (n : ℕ) (hn : n ≤ 371292) :
    ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, w5 ⟨0, hS⟩ q j ≠ w5 ⟨0, hS⟩ q j' := by
  intro j hj j' hj' hne
  obtain ⟨q, hq⟩ := code5_inj (by have := Finset.mem_range.1 hj; omega) (by have := Finset.mem_range.1 hj'; omega) hne
  exact ⟨q, by simp only [w5]; exact hq⟩

theorem w5_ne (s : Fin S) (q : Fin 5) (j : ℕ) : w5 s q j ≠ 2 ^ 31 := by
  have := code5_le q j
  unfold w5
  split_ifs <;> norm_num <;> omega

/-- **The 4-lane layout's words are distinct** (bit model, full grid, `n ≤ 28,559`): pairwise distinct, none free, and
`m·n·S` of them. -/
theorem ssn1lServed_full (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ) (hn : n ≤ 28559) :
    DistinctOn Set.univ (FreeB m C) (ssnWords (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l n) ∧
      (ssnWords (C := C) (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l n).card =
        m * (n * S) :=
  ssnFull (src := l4Src hC) (used := fun s q => (s : ℕ) = 0 ∨ q = 0) (pos := fun s q => l4Pos s q) l4_pos_lt
    (l4_inj hC) κ w1l w1l_used hS n (w1l_reach n hn) (code_cols hS w1l (fun q j => by simp [w1l]) n hn) w1l_ne

/-- **The 5-lane layout's words are distinct** (bit model, full grid, `n ≤ 371,292`): pairwise distinct, none free,
and `m·n·S` of them. -/
theorem ssn1l5Served_full (hC : S + 4 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 371292) :
    DistinctOn Set.univ (FreeB m C) (ssnWords (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 n) ∧
      (ssnWords (C := C) (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 n).card =
        m * (n * S) :=
  ssnFull (src := l5Src hC) (used := fun s q => (s : ℕ) = 0 ∨ q = 0) (pos := fun s q => l5Pos s q) l5_pos_lt
    (l5_inj hC) κ w5 w5_used hS n (w5_reach n hn) (w5_cols hS n hn) w5_ne

/-- **The 4-lane layout is charged** (bit model, full grid, `n ≤ 28,559`): every well-formed `MH100w` program on
constants and raw coin words that outputs the committed words on the full coin grid costs at least `32·m·n·S`. -/
theorem ssn1lServedB_charged (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 28559) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (Bits m C → ℚ)) (Pg : List (Op100 (Bits m C)))
    (hL : ∀ f ∈ Fs, f ∈ FreeB m C) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeB m C))
    (hO : OutputsOn fpSem Set.univ Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg := by
  obtain ⟨hD, hc⟩ := ssn1lServed_full hC κ hS n hn
  rw [← hc]
  exact charged _ (fun q => Or.inl ⟨q, rfl⟩) _ fpSem Fs Pg _ hL hPg hD hO

/-- **The 5-lane code's served layout is charged**: with SW-NOISE's 5-lane layout (`l5Src`, `l5Pos`) on `C` coin words
a row, `S + 4 ≤ 32·C`, every well-formed `MH100w` program on constants and raw coin words that outputs the committed
words on the full coin grid costs at least `32·m·n·S`, for `n ≤ 371,292` columns. -/
theorem ssn1l5Served_charged (hC : S + 4 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 371292) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (Bits m C → ℚ)) (Pg : List (Op100 (Bits m C)))
    (hL : ∀ f ∈ Fs, f ∈ FreeB m C) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeB m C))
    (hO : OutputsOn fpSem Set.univ Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordB (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg := by
  obtain ⟨hD, hc⟩ := ssnFull (src := l5Src hC) (used := fun s q => (s : ℕ) = 0 ∨ q = 0)
    (pos := fun s q => l5Pos s q) l5_pos_lt (l5_inj hC) κ w5 w5_used hS n (w5_reach n hn) (w5_cols hS n hn) w5_ne
  rw [← hc]
  exact charged _ (fun q => Or.inl ⟨q, rfl⟩) _ fpSem Fs Pg _ hL hPg hD hO

/-- **`ssn_1l`'s served words are shift-separated** (bit model, full grid, `n ≤ 28,559`): `ssnFull_shift` for
`l4Src`/`l4Pos` and the 4-lane code. -/
theorem ssn1lServed_shift (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 28559) :
    (∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2),
      ∀ t' ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2),
      (∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), t g = t' g + q) → t = t') ∧
    ∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2),
      ∀ f ∈ FreeB m C, ¬ ∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), f g = t g + q :=
  ssnFull_shift l4_pos_lt (l4_inj hC) κ w1l w1l_used hS n (w1l_reach n hn)
    (code_cols hS w1l (fun q j => by simp [w1l]) n hn) w1l_ne

/-- **The 5-lane code's served words are shift-separated** (bit model, full grid, `n ≤ 371,292`). -/
theorem ssn1l5Served_shift (hC : S + 4 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 371292) :
    (∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 x.1 x.2.1 x.2.2),
      ∀ t' ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 x.1 x.2.1 x.2.2),
      (∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), t g = t' g + q) → t = t') ∧
    ∀ t ∈ (univ ×ˢ range n ×ˢ univ).image (fun x : Fin m × ℕ × Fin S =>
        ssnWordB (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 x.1 x.2.1 x.2.2),
      ∀ f ∈ FreeB m C, ¬ ∃ q : ℚ, ∀ g ∈ (Set.univ : Set (Bits m C)), f g = t g + q :=
  ssnFull_shift l5_pos_lt (l5_inj hC) κ w5 w5_used hS n (w5_reach n hn) (w5_cols hS n hn) w5_ne

end Served

end Pouw.Dimension.Ssn
