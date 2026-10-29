import Proofs.Pouw.Dimension.Scratch.SwLean
import Proofs.Pouw.Dimension.GrindProofs

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
  `ssnFull_shift` makes that shift-separated: no two words differ by a constant and none is a free value plus a
  constant, the hypotheses of carrier separation (`ssn1lServed_shift`, `ssn1l5Served_shift` for the served layouts).
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

open Finset Pouw.NCP Pouw.Dimension Pouw.Dimension.H100 Pouw.Dimension.Grind Pouw.Dimension.Ssn

section Bits

variable {m S C L : ℕ}

variable {src : Fin S → Fin C} {used : Fin S → Fin L → Prop} [∀ s q, Decidable (used s q)]
  {pos : Fin S → Fin L → ℕ}

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
