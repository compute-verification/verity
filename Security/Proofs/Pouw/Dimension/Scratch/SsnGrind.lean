import Proofs.Pouw.Dimension.Scratch.SwLean
import Proofs.Pouw.Dimension.Scratch.Grind

/-!
# Served sparse noise on a success set, and SW-NOISE's served layouts (scratch; not in a lock)

**Draft, pending review.** `ssnTargetG` charges every SSN word on the full coin grid. A prover that chooses `A` and `B`
before the coins can retry its inputs until the coins fall in a set it likes. This file bounds the words that set can
cost, in the bit model, shows the bound is of the right order for `ssn_1l`, and proves distinctness for the served bit
layouts of the 4-lane and 5-lane column codes.

* **The bit model.** The grid is every value of the call's coin bits `Bits m C` (row, coin word, bit), a
  reparametrization of `ssnGridG m C`. The free values are the constants and the raw coin words
  `Σ_b 2^b·x(r, c, b)` (`FreeB`). A committed word (`ssnWordB`) is its data case's constant plus, at every bit, the
  weight of the used lane that reads it (`bitCoef`), over `L` lanes a step.
* **Rows** (`ssnRows`, `ssnRows_charged`). When every step's weights reach every column, step 0's tell the columns
  apart and the used lanes read distinct bits, then on every set of coin values of measure at least `2^−g` all but
  `g·n·S + m·C` of the `m·n·S` words stay distinct and non-free, so a program right on that set costs at least
  `32·(m·n·S − g·n·S − m·C)`. The window of a word is its row: only step 0's code tells its columns apart.
* **The full grid** (`ssnFull`): with no weight `2^31`, all `m·n·S` words are distinct and non-free.
* **Served layouts.** `l4Src`/`l4Pos` is `ssn_1l`'s layout (step 0's lanes at bits `0, 8, 16, 24` of word 0, later
  steps at word 0's other 28 bits in order, then words 1, 2, …) and `l5Src`/`l5Pos` the 5-lane code's (step 0's fifth
  lane at bit 1, later steps at word 0's other 27 bits). Both are injective (`l4_inj`, `l5_inj`), so
  `ssn1lServed_charged` (`ssnTargetG`'s model) and `ssn1l5Served_charged` (the bit model) charge every word, and
  `ssn1lServedRows_charged` and `ssn1l5ServedRows_charged` hold on every success set. The 5-lane code `code5` is the
  balanced base-13 code of `z_j = (⌊j/2⌋ + 1)·(−1)^j`, injective and nonzero for `n ≤ 371,292` columns.
* **`ssn_1l` loses whole rows** (`zeroRow_agree`, `ssn1l_zero_row`). Fixing row `i₀`'s step-0 bits to `0`, a set of
  measure at least `2^−4` (`zeroSet_measure`), makes the words of any two columns with equal data constants agree at
  every step. So the rows bound's `g·n·S` term can't drop to `g·n`.
-/

namespace Pouw.Dimension.SwLean

open Finset Pouw.NCP Pouw.Dimension Pouw.Dimension.H100 Pouw.Dimension.Grind

section Bits

variable {m S C L : ℕ}

/-- The call's coin bits: `x (i, c, b)` is bit `b` of row `i`'s coin word `c`. -/
abbrev Bits (m C : ℕ) : Type := Fin m × Fin C × Fin 32 → Fin 2

/-- Row `r`'s raw coin word `c`, as coefficients on the bits: `2^b` at `(r, c, b)`. -/
def coinCoef (r : Fin m) (c : Fin C) : Fin m × Fin C × Fin 32 → ℚ :=
  fun k => if k.1 = r ∧ k.2.1 = c then 2 ^ (k.2.2 : ℕ) else 0

/-- The free values in the bit model: the constants and the raw coin words. -/
def FreeB (m C : ℕ) : Set (Bits m C → ℚ) :=
  {f | (∃ q : ℚ, f = fun _ => q) ∨ ∃ r c, f = aff 0 (coinCoef (m := m) (C := C) r c)}

/-- The weight bit `k` carries in row `i`'s word of column `j` after step `t`: the weight of the used lane that reads it
(step `s` reads lane `q` from bit `pos s q` of coin word `src s`). -/
def bitCoef (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ) (t : Fin S) :
    Fin m × Fin C × Fin 32 → ℚ :=
  fun k => if k.1 = i then
    ∑ s : Fin S, ∑ q : Fin L, if s ≤ t ∧ used s q ∧ src s = k.2.1 ∧ pos s q = k.2.2 then (W s q j : ℚ) else 0
  else 0

/-- SSN's committed word of row `i`, column `j`, after step `t`, in the bit model. -/
def ssnWordB (src : Fin S → Fin C) (used : Fin S → Fin L → Prop) [∀ s q, Decidable (used s q)]
    (pos : Fin S → Fin L → ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (i : Fin m) (j : ℕ)
    (t : Fin S) : Bits m C → ℚ :=
  aff (κ i j t : ℚ) (bitCoef src used pos W i j t)

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

/-- **SSN's words on a success set, by rows**: with the used lanes on distinct bits, every step's weights reaching
every column and step 0's telling the columns apart, on every set of coin values of measure at least `2^−g` some
`D` of the `m·n·S` words, all but `g·n·S + m·C`, is pairwise distinct and non-free there. -/
theorem ssnRows (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q)
    (hS : 0 < S) (n : ℕ) (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0)
    (hW2 : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j')
    (Sg : Finset (Bits m C)) (g : ℕ) (hSg : 2 ^ (m * (C * 32)) ≤ 2 ^ g * Sg.card) :
    ∃ D ⊆ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)),
      DistinctOn (↑Sg) (FreeB m C) (D.image fun x => ssnWordB src used pos κ W x.1 x.2.1 x.2.2) ∧
      (D.image fun x => ssnWordB src used pos κ W x.1 x.2.1 x.2.2).card = D.card ∧
      m * (n * S) ≤ D.card + g * (n * S) + m * C := by
  classical
  have hcard : Fintype.card (Fin m × Fin C × Fin 32) = m * (C * 32) := by simp
  have mem : ∀ u : Fin m × ℕ × Fin S, u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)) ↔ u.2.1 < n := by
    intro u
    simp
  have hν : Set.InjOn (rk n) ↑(univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)) :=
    fun u hu u' hu' h => rk_inj ((mem u).1 (Finset.mem_coe.1 hu)) ((mem u').1 (Finset.mem_coe.1 hu')) h
  have hwin : ∀ u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)),
      ∀ u' ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)), rk n u' < rk n u →
      ∃ k : Fin m × Fin C × Fin 32, k.1 ∈ ({u.1} : Finset (Fin m)) ∧
        bitCoef src used pos W u.1 u.2.1 u.2.2 k ≠ bitCoef src used pos W u'.1 u'.2.1 u'.2.2 k ∧
        ∀ k' : Fin m × Fin C × Fin 32, (k.1 : ℕ) < k'.1 →
          bitCoef src used pos W u.1 u.2.1 u.2.2 k' = bitCoef src used pos W u'.1 u'.2.1 u'.2.2 k' := by
    intro u hu u' hu' hlt
    obtain ⟨k, hk1, hk2, hk3⟩ := rows_win hpos hinj W hU hS n hW1 hW2 ((mem u).1 hu) ((mem u').1 hu') hlt
    exact ⟨k, Finset.mem_singleton.2 hk1, hk2, fun k' h => hk3 k' (by rw [← hk1]; exact h)⟩
  have hwin0 : ∀ u ∈ (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)),
      ∃ k : Fin m × Fin C × Fin 32, k.1 ∈ ({u.1} : Finset (Fin m)) ∧
        bitCoef src used pos W u.1 u.2.1 u.2.2 k ≠ 0 ∧
        ∀ k' : Fin m × Fin C × Fin 32, (k.1 : ℕ) < k'.1 → bitCoef src used pos W u.1 u.2.1 u.2.2 k' = 0 := by
    intro u hu
    obtain ⟨k, hk1, hk2, hk3⟩ := rows_win0 hpos hinj W hU hS n hW1 ((mem u).1 hu)
    exact ⟨k, Finset.mem_singleton.2 hk1, hk2, fun k' h => hk3 k' (by rw [← hk1]; exact h)⟩
  have hF : ∀ f ∈ FreeB m C, (∃ c : ℚ, f = fun _ => c) ∨
      ∃ e ∈ (univ : Finset (Fin m × Fin C)),
        f = aff (q := 2) ((fun _ => (0 : ℚ)) e) ((fun e : Fin m × Fin C => coinCoef e.1 e.2) e) := by
    rintro f (⟨c, rfl⟩ | ⟨r, c, rfl⟩)
    · exact Or.inl ⟨c, rfl⟩
    · exact Or.inr ⟨(r, c), Finset.mem_univ _, rfl⟩
  have hM : ∀ b : Fin m,
      ((univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)).filter fun u => b ∈ ({u.1} : Finset (Fin m))).card ≤
        n * S := by
    intro b
    calc ((univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)).filter
            fun u => b ∈ ({u.1} : Finset (Fin m))).card
        ≤ (({b} : Finset (Fin m)) ×ˢ range n ×ˢ (univ : Finset (Fin S))).card := by
          refine Finset.card_le_card fun u hu => ?_
          rw [Finset.mem_filter, mem, Finset.mem_singleton] at hu
          obtain ⟨i, j, t⟩ := u
          simp only [Finset.mem_product, Finset.mem_singleton, Finset.mem_range, Finset.mem_univ, and_true]
          exact ⟨hu.2.symm, hu.1⟩
      _ = n * S := by simp [Finset.card_product]
  obtain ⟨D, hD, h1, h2, h3⟩ := lost_le (q := 2) (by norm_num) Sg g (by rw [hcard]; exact hSg)
    (fun k : Fin m × Fin C × Fin 32 => k.1) (fun i : Fin m => (i : ℕ)) Fin.val_injective
    (univ ×ˢ range n ×ˢ univ) (rk n) hν (fun u => (κ u.1 u.2.1 u.2.2 : ℚ))
    (fun u => bitCoef src used pos W u.1 u.2.1 u.2.2) (fun u => {u.1}) hwin hwin0
    (univ : Finset (Fin m × Fin C)) (fun _ => 0) (fun e => coinCoef e.1 e.2) (FreeB m C) hF (n * S) hM
  have hΩc : (univ ×ˢ range n ×ˢ univ : Finset (Fin m × ℕ × Fin S)).card = m * (n * S) := by
    simp [Finset.card_product]
  have hEc : (univ : Finset (Fin m × Fin C)).card = m * C := by simp
  rw [hΩc, hEc] at h3
  exact ⟨D, hD, h1, h2, h3⟩

/-- **SSN is charged on a success set, by rows**: as `ssnRows`, every well-formed `MH100w` program on constants and
raw coin words that outputs SSN's committed words on the set costs at least `32·(m·n·S − g·n·S − m·C)`. -/
theorem ssnRows_charged (hpos : ∀ s q, pos s q < 32)
    (hinj : ∀ s q s' q', used s q → used s' q' → src s = src s' → pos s q = pos s' q' → s = s' ∧ q = q')
    (κ : Fin m → ℕ → Fin S → ℤ) (W : Fin S → Fin L → ℕ → ℤ) (hU : ∀ s q j, W s q j ≠ 0 → used s q)
    (hS : 0 < S) (n : ℕ) (hW1 : ∀ s : Fin S, ∀ j ∈ range n, ∃ q, W s q j ≠ 0)
    (hW2 : ∀ j ∈ range n, ∀ j' ∈ range n, j ≠ j' → ∃ q, W ⟨0, hS⟩ q j ≠ W ⟨0, hS⟩ q j')
    (Sg : Finset (Bits m C)) (g : ℕ) (hSg : 2 ^ (m * (C * 32)) ≤ 2 ^ g * Sg.card)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (Bits m C → ℚ)) (Pg : List (Op100 (Bits m C)))
    (hL : ∀ f ∈ Fs, f ∈ FreeB m C) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeB m C))
    (hO : OutputsOn fpSem (↑Sg) Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordB src used pos κ W x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg + 32 * (g * (n * S) + m * C) := by
  obtain ⟨D, hD, h1, h2, h3⟩ := ssnRows hpos hinj κ W hU hS n hW1 hW2 Sg g hSg
  have h := charged (FreeB m C) (fun q => Or.inl ⟨q, rfl⟩) _ fpSem Fs Pg _ hL hPg h1
    (fun t ht => hO t (Finset.image_subset_image hD ht))
  rw [h2] at h
  omega

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

/-! ### `ssn_1l` loses whole rows -/

/-- The coin values with every used step-0 lane of row `i₀` at `0`. -/
def zeroSet (hpos : ∀ s q, pos s q < 32) (hS : 0 < S) (i₀ : Fin m) : Finset (Bits m C) :=
  univ.filter fun x => ∀ q, used ⟨0, hS⟩ q → x (i₀, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ q⟩) = 0

/-- Fixing at most `L` bits leaves a set of measure at least `2^−L`. -/
theorem zeroSet_measure (hpos : ∀ s q, pos s q < 32) (hS : 0 < S) (i₀ : Fin m) :
    2 ^ (m * (C * 32)) ≤ 2 ^ L * (zeroSet (src := src) (used := used) hpos hS i₀).card := by
  classical
  set P : Finset (Fin m × Fin C × Fin 32) :=
    univ.image fun q : Fin L => (i₀, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ q⟩) with hP
  have hPc : P.card ≤ L := Finset.card_image_le.trans (by simp)
  have hPN : P.card ≤ m * (C * 32) := by
    have := Finset.card_le_univ P
    simpa using this
  have hle := Finset.card_le_card_of_injOn
    (fun y : {k // k ∉ P} → Fin 2 => fun k : Fin m × Fin C × Fin 32 => if h : k ∈ P then 0 else y ⟨k, h⟩)
    (s := Finset.univ) (t := zeroSet (src := src) (used := used) hpos hS i₀)
    (fun y _ => Finset.mem_coe.2 (Finset.mem_filter.2 ⟨Finset.mem_univ _, fun q _ => by
      dsimp only
      rw [dif_pos (Finset.mem_image.2 ⟨q, Finset.mem_univ _, rfl⟩)]⟩))
    (fun y _ y' _ h => funext fun k => by
      have := congrFun h k.1
      simp only [dif_neg k.2] at this
      exact this)
  have hcard : (Finset.univ : Finset ({k // k ∉ P} → Fin 2)).card = 2 ^ (m * (C * 32) - P.card) := by
    simp [Finset.card_univ, Fintype.card_fun, Fintype.card_subtype_compl, Fintype.card_coe]
  rw [hcard] at hle
  calc 2 ^ (m * (C * 32)) = 2 ^ P.card * 2 ^ (m * (C * 32) - P.card) := by
        rw [← pow_add, Nat.add_sub_cancel' hPN]
    _ ≤ 2 ^ L * (zeroSet (src := src) (used := used) hpos hS i₀).card :=
        Nat.mul_le_mul (Nat.pow_le_pow_right (by norm_num) hPc) hle

/-- **Columns with equal constants coincide on `zeroSet`**: when only step 0's weights depend on the column, the
words of row `i₀` at columns `j`, `j'` with equal constants agree after every step at every coin value with row
`i₀`'s step-0 bits `0`. -/
theorem zeroRow_agree (hpos : ∀ s q, pos s q < 32) (hS : 0 < S) (κ : Fin m → ℕ → Fin S → ℤ)
    (W : Fin S → Fin L → ℕ → ℤ) (hWj : ∀ s : Fin S, (s : ℕ) ≠ 0 → ∀ q j j', W s q j = W s q j') (i₀ : Fin m)
    {j j' : ℕ} (t : Fin S) (hκ : κ i₀ j t = κ i₀ j' t) :
    ∀ x ∈ zeroSet (src := src) (used := used) hpos hS i₀,
      ssnWordB src used pos κ W i₀ j t x = ssnWordB src used pos κ W i₀ j' t x := by
  intro x hx
  have hx0 := (Finset.mem_filter.1 hx).2
  unfold ssnWordB aff
  rw [hκ]
  congr 1
  refine Finset.sum_congr rfl fun k _ => ?_
  by_cases hxk : ((x k : ℕ) : ℚ) = 0
  · rw [hxk, mul_zero, mul_zero]
  · congr 1
    unfold bitCoef
    split_ifs with hk
    · refine Finset.sum_congr rfl fun s _ => Finset.sum_congr rfl fun q _ => ?_
      split_ifs with hc
      · by_cases hs : (s : ℕ) = 0
        · exfalso
          have hs' : s = ⟨0, hS⟩ := Fin.ext hs
          subst hs'
          have hkpt : k = (i₀, src ⟨0, hS⟩, ⟨pos ⟨0, hS⟩ q, hpos _ q⟩) := by
            obtain ⟨k1, k2, k3⟩ := k
            exact Prod.ext hk (Prod.ext hc.2.2.1.symm (Fin.ext hc.2.2.2.symm))
          apply hxk
          rw [hkpt, hx0 q hc.2.1]
          rfl
        · exact_mod_cast hWj s hs q j j'
      · rfl
    · rfl

/-- **`ssn_1l` loses whole rows**: for every layout, fixing row `i₀`'s four step-0 bits to `0` leaves a set of measure
at least `2^−4` on which row `i₀`'s words of any two columns with equal constants agree after every step. At a zero
row the constant is `−β_j` at every step, so the row keeps at most one word a value of `β` and a step. -/
theorem ssn1l_zero_row {src : Fin S → Fin C} {pos : Fin S → Fin 4 → ℕ} (hpos : ∀ s q, pos s q < 32) (hS : 0 < S)
    (κ : Fin m → ℕ → Fin S → ℤ) (i₀ : Fin m) :
    2 ^ (m * (C * 32)) ≤
        2 ^ 4 * (zeroSet (src := src) (used := fun s q => (s : ℕ) = 0 ∨ q = 0) hpos hS i₀).card ∧
      ∀ j j' (t : Fin S), κ i₀ j t = κ i₀ j' t →
        ∀ x ∈ zeroSet (src := src) (used := fun s q => (s : ℕ) = 0 ∨ q = 0) hpos hS i₀,
          ssnWordB src (fun s q => (s : ℕ) = 0 ∨ q = 0) pos κ w1l i₀ j t x =
            ssnWordB src (fun s q => (s : ℕ) = 0 ∨ q = 0) pos κ w1l i₀ j' t x :=
  ⟨zeroSet_measure hpos hS i₀, fun _ _ t hκ => zeroRow_agree hpos hS κ w1l
    (fun s hs q j j' => by simp [w1l, hs]) i₀ t hκ⟩

end Bits

/-! ### SW-NOISE's served layouts -/

section Served

variable {m S C : ℕ}

/-- `ssn_1l`'s coin word for step `s`: word 0 through step 28, then words 1, 2, … 32 steps each. -/
def l4Word (s : ℕ) : ℕ := if s < 29 then 0 else 1 + (s - 29) / 32

/-- `ssn_1l`'s bit for step `s`, lane `q`: step 0's lanes at bits `8q` of word 0; steps 1–28 at word 0's other 28 bits
in order (1–7, 9–15, 17–23, 25–31); later steps at bits 0–31 of each next word. -/
def l4Pos (s q : ℕ) : ℕ := if s = 0 then 8 * q else if s < 29 then s + (s - 1) / 7 else (s - 29) % 32

theorem l4Word_lt (hC : S + 3 ≤ 32 * C) {s : ℕ} (hs : s < S) : l4Word s < C := by
  unfold l4Word
  split_ifs <;> omega

/-- `ssn_1l`'s served coin word for each step, with `C ≥ 1 + ⌈max(0, S − 29)/32⌉` words a row. -/
def l4Src (hC : S + 3 ≤ 32 * C) (s : Fin S) : Fin C := ⟨l4Word s, l4Word_lt hC s.isLt⟩

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

/-- The 5-lane code's coin word for step `s`: word 0 through step 27, then words 1, 2, … 32 steps each. -/
def l5Word (s : ℕ) : ℕ := if s < 28 then 0 else 1 + (s - 28) / 32

/-- The 5-lane code's bit for step `s`, lane `q`: step 0's lanes 0–3 at bits `8q` and lane 4 at bit 1 of word 0;
steps 1–27 at word 0's other 27 bits in order (2–7, 9–15, 17–23, 25–31); later steps at bits 0–31 of each next
word. -/
def l5Pos (s q : ℕ) : ℕ :=
  if s = 0 then (if q < 4 then 8 * q else 1) else if s < 28 then s + 1 + s / 7 else (s - 28) % 32

theorem l5Word_lt (hC : S + 4 ≤ 32 * C) {s : ℕ} (hs : s < S) : l5Word s < C := by
  unfold l5Word
  split_ifs <;> omega

/-- The 5-lane code's served coin word for each step, with `C ≥ 1 + ⌈max(0, S − 28)/32⌉` words a row. -/
def l5Src (hC : S + 4 ≤ 32 * C) (s : Fin S) : Fin C := ⟨l5Word s, l5Word_lt hC s.isLt⟩

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

/-- Column `j`'s balanced code number `z_j = (⌊j/2⌋ + 1)·(−1)^j`, shifted by `(13⁵ − 1)/2` into `[0, 13⁵)`. -/
def z5 (j : ℕ) : ℕ := if j % 2 = 0 then 185646 + (j / 2 + 1) else 185646 - (j / 2 + 1)

/-- The 5-lane balanced base-13 code: `C[e, j]` is `z_j`'s `e`-th balanced digit, in `[−6, 6]`. -/
def code5 (e : Fin 5) (j : ℕ) : ℤ := ((z5 j / 13 ^ (e : ℕ) % 13 : ℕ) : ℤ) - 6

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

/-- The 5-lane code's weights: step 0 the code on five lanes, every later step one lane of weight 1 for every column. -/
def w5 (s : Fin S) (q : Fin 5) (j : ℕ) : ℤ := if (s : ℕ) = 0 then code5 q j else if q = 0 then 1 else 0

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

/-- **`ssn_1l`'s served layout is charged**: with SW-NOISE's bit layout (`l4Src`, `l4Pos`) on `C` coin words a row,
`S + 3 ≤ 32·C`, every well-formed `MH100w` program on constants and raw coin words that outputs `ssn_1l`'s committed
words on the full coin grid costs at least `32·m·n·S`, for `n ≤ 28,559` columns. -/
theorem ssn1lServed_charged [DecidableEq (SsnCoinsG m C → ℚ)] (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ)
    (hS : 0 < S) (n : ℕ) (hn : n ≤ 28559) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (SsnCoinsG m C → ℚ))
    (Pg : List (Op100 (SsnCoinsG m C))) (hL : ∀ f ∈ Fs, f ∈ FreeSsnG m C)
    (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeSsnG m C))
    (hO : OutputsOn fpSem (ssnGridG m C) Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordG (l4Src hC) (laneBit (fun s q => (s : ℕ) = 0 ∨ q = 0) fun s q => l4Pos s q) κ w1l
        x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg :=
  ssn1lBit_charged (src := l4Src hC) (pos := fun s q => l4Pos s q) l4_pos_lt (l4_inj hC) κ hS n hn fpSem Fs Pg
    hL hPg hO

/-- **`ssn_1l`'s served layout on a success set**: as `ssn1lServed_charged` in the bit model, on every set of coin
values of measure at least `2^−g`, at least `32·(m·n·S − g·n·S − m·C)`. -/
theorem ssn1lServedRows_charged (hC : S + 3 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 28559) (Sg : Finset (Bits m C)) (g : ℕ) (hSg : 2 ^ (m * (C * 32)) ≤ 2 ^ g * Sg.card)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (Bits m C → ℚ)) (Pg : List (Op100 (Bits m C)))
    (hL : ∀ f ∈ Fs, f ∈ FreeB m C) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeB m C))
    (hO : OutputsOn fpSem (↑Sg) Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordB (l4Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l4Pos s q) κ w1l x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg + 32 * (g * (n * S) + m * C) :=
  ssnRows_charged l4_pos_lt (l4_inj hC) κ w1l w1l_used hS n (w1l_reach n hn)
    (code_cols hS w1l (fun q j => by simp [w1l]) n hn) Sg g hSg fpSem Fs Pg hL hPg hO

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

/-- **The 5-lane code's served layout on a success set**: as `ssn1l5Served_charged`, on every set of coin values of
measure at least `2^−g`, at least `32·(m·n·S − g·n·S − m·C)`. -/
theorem ssn1l5ServedRows_charged (hC : S + 4 ≤ 32 * C) (κ : Fin m → ℕ → Fin S → ℤ) (hS : 0 < S) (n : ℕ)
    (hn : n ≤ 371292) (Sg : Finset (Bits m C)) (g : ℕ) (hSg : 2 ^ (m * (C * 32)) ≤ 2 ^ g * Sg.card)
    (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (Fs : List (Bits m C → ℚ)) (Pg : List (Op100 (Bits m C)))
    (hL : ∀ f ∈ Fs, f ∈ FreeB m C) (hPg : ∀ o ∈ Pg, o.WFw h100Floor (FreeB m C))
    (hO : OutputsOn fpSem (↑Sg) Pg Fs ((univ ×ˢ range n ×ˢ univ).image fun x : Fin m × ℕ × Fin S =>
      ssnWordB (l5Src hC) (fun s q => (s : ℕ) = 0 ∨ q = 0) (fun s q => l5Pos s q) κ w5 x.1 x.2.1 x.2.2)) :
    32 * (m * (n * S)) ≤ cost100 h100Floor Pg + 32 * (g * (n * S) + m * C) :=
  ssnRows_charged l5_pos_lt (l5_inj hC) κ w5 w5_used hS n (w5_reach n hn) (w5_cols hS n hn) Sg g hSg fpSem Fs Pg
    hL hPg hO

end Served

end Pouw.Dimension.SwLean
