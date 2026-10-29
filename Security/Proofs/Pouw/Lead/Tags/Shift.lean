import Definitions.Pouw.Lead.Layout
import Definitions.Pouw.Dimension.Ssn
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith

/-!
# The lead's words are shift-separated on its coins

The grid is every value of the lead's coins (`Pouw.Lead.Coins`): the tag bits `b`, the step-1 bits `a₁`, the tag weights
`G` and the coin lanes `s`. A committed word is affine in them, so a coin's bump from the zero value moves each word by
that coin's coefficient (`word_bumpB`, `word_bumpA`, `word_bumpG`, `word_bumpU`), and two words that differ by a constant
move alike under every bump (`shift_probe`). The free values (`FreeLead`) are the constants and every function of one
coin: a raw coin, its lane value, or any encoding of it.

**The words are shift-separated** (`leadWords_shift`): no two of a call's words differ by a constant on the coins, and
none is a free value plus a constant, for at most 28,560 rows, any number of columns, any data and any number of steps,
with nonzero coin tiles and coin lanes of at least one bit. Words of different steps are separated by the later step's
tag bits `b_t`, and words of one step in different columns by the step's coin lanes under `-rot`'s weights. In one
column, two rows' step-0 words are separated by the row tag against `G`, whose digits are distinct and nonzero below
28,560 rows (`rowTag_inj`, `rowTag_ne_zero`), and their later words by the rows' tag bits. Every word reads a coin lane
and a second coin, `G` at step 0 and the tag bit later, so none is a function of one coin plus a constant.
-/

namespace Pouw.SecurityProofs.Lead.Tags

open Finset Pouw.Lead Pouw.Dimension.Ssn

variable {L : Layout}

/-- A coin of the lead: a tag bit, a step-1 bit, a tag weight or a coin lane. -/
abbrev Coord (L : Layout) : Type :=
  (Fin L.m × Fin L.S) ⊕ (Fin L.m × Fin 4) ⊕ (Fin 4 × Fin L.n) ⊕ (Fin L.S × Fin L.tiles × Fin 3 × Fin L.n)

/-- A coin's digit at a value of the coins. -/
def coord (ω : Coins L) : Coord L → ℕ
  | .inl x => ω.1 x.1 x.2
  | .inr (.inl x) => ω.2.1 x.1 x.2
  | .inr (.inr (.inl x)) => ω.2.2.1 x.1 x.2
  | .inr (.inr (.inr x)) => ω.2.2.2 x.1 x.2.1 x.2.2.1 x.2.2.2

/-- The free values on the lead's coins: the constants, and every function of one coin. -/
def FreeLead (L : Layout) : Set (Coins L → ℚ) :=
  {f | (∃ c : ℚ, f = fun _ => c) ∨ ∃ x : Coord L, ∀ ω ω' : Coins L, coord ω x = coord ω' x → f ω = f ω'}

open Classical in
/-- A call's committed words on the coins: every row, step and column. -/
noncomputable def leadWords (L : Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) : Finset (Coins L → ℚ) :=
  (univ : Finset (Fin L.m × Fin L.S × Fin L.n)).image fun x ω => (word L κ ω x.1 x.2.1 x.2.2 : ℚ)

/-! ## Probes -/

/-- The zero coin lane digit. -/
def zeroU (L : Layout) : Fin (2 ^ L.e) := ⟨0, Nat.two_pow_pos _⟩

/-- The zero value of the coins. -/
def zeroC (L : Layout) : Coins L := (fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ _ _ => zeroU L)

/-- The coins with tag bit `b_{t₀}[i₀]` set. -/
def bumpB (i₀ : Fin L.m) (t₀ : Fin L.S) : Coins L :=
  (fun i t => if i = i₀ ∧ t = t₀ then 1 else 0, fun _ _ => 0, fun _ _ => 0, fun _ _ _ _ => zeroU L)

/-- The coins with step-1 bit `a₁[i₀, q₀]` set. -/
def bumpA (i₀ : Fin L.m) (q₀ : Fin 4) : Coins L :=
  (fun _ _ => 0, fun i q => if i = i₀ ∧ q = q₀ then 1 else 0, fun _ _ => 0, fun _ _ _ _ => zeroU L)

/-- The coins with tag weight digit `g[q₀, j₀]` one. -/
def bumpG (q₀ : Fin 4) (j₀ : Fin L.n) : Coins L :=
  (fun _ _ => 0, fun _ _ => 0, fun q j => if q = q₀ ∧ j = j₀ then 1 else 0, fun _ _ _ _ => zeroU L)

/-- The coins with coin lane digit `u_{t₀}[k₀, ℓ₀, j₀]` one. -/
def bumpU (he : 1 < 2 ^ L.e) (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) (j₀ : Fin L.n) : Coins L :=
  (fun _ _ => 0, fun _ _ => 0, fun _ _ => 0,
    fun t k ℓ j => if t = t₀ ∧ k = k₀ ∧ ℓ = ℓ₀ ∧ j = j₀ then ⟨1, he⟩ else zeroU L)

/-- Two functions that differ by a constant move alike between any two points. -/
theorem shift_probe {G : Type} {t t' : G → ℚ} {q : ℚ} (h : ∀ g ∈ (Set.univ : Set G), t g = t' g + q)
    (g₀ g₁ : G) : t g₁ - t g₀ = t' g₁ - t' g₀ := by
  rw [h g₁ trivial, h g₀ trivial]; ring

theorem word_sub (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω₁ ω₀ : Coins L) (i : Fin L.m) (t : Fin L.S)
    (j : Fin L.n) : word L κ ω₁ i t j - word L κ ω₀ i t j =
      ∑ τ : Fin L.S, if τ ≤ t then coinTerm L ω₁ i τ j - coinTerm L ω₀ i τ j else 0 := by
  unfold word
  rw [add_sub_add_left_eq_sub, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun τ _ => ?_
  split_ifs <;> ring

theorem tile_lt (htile : 0 < L.tile) (i : Fin L.m) : (i : ℕ) / L.tile < L.tiles := by
  unfold Layout.tiles
  refine Nat.lt_of_succ_le ((Nat.le_div_iff_mul_le htile).2 ?_)
  have h1 := Nat.div_mul_le_self (i : ℕ) L.tile
  have h2 := i.isLt
  have : ((i : ℕ) / L.tile).succ * L.tile = (i : ℕ) / L.tile * L.tile + L.tile := Nat.succ_mul _ _
  omega

/-- A row's coin tile. -/
def rowTile (htile : 0 < L.tile) (i : Fin L.m) : Fin L.tiles := ⟨(i : ℕ) / L.tile, tile_lt htile i⟩

/-! ## Each coin's coefficient in a word -/

/-- Two values of the coins with the same coin lanes differ in a coin term only through step 0's tag and code or a
later step's tag lane. -/
theorem coinTerm_sub_of_lanes (ω₁ ω₀ : Coins L) (h : ω₁.2.2.2 = ω₀.2.2.2) (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    coinTerm L ω₁ i τ j - coinTerm L ω₀ i τ j =
      (if (τ : ℕ) = 0 then
        (∑ q : Fin 4, rowTag i q * ω₁.G q j) + ∑ q : Fin 4, ((ω₁.2.1 i q : ℕ) : ℤ) * colCode j q
      else 1 + ((ω₁.1 i τ : ℕ) : ℤ)) -
      (if (τ : ℕ) = 0 then
        (∑ q : Fin 4, rowTag i q * ω₀.G q j) + ∑ q : Fin 4, ((ω₀.2.1 i q : ℕ) : ℤ) * colCode j q
      else 1 + ((ω₀.1 i τ : ℕ) : ℤ)) := by
  have : laneTerm L ω₁ i τ j = laneTerm L ω₀ i τ j := by unfold laneTerm Coins.s; rw [h]
  unfold coinTerm
  rw [this]
  ring

/-- Two values of the coins that differ only in the coin lanes differ in a coin term by the lanes. -/
theorem coinTerm_sub_of_tags (ω₁ ω₀ : Coins L) (h1 : ω₁.1 = ω₀.1) (h2 : ω₁.2.1 = ω₀.2.1) (h3 : ω₁.2.2.1 = ω₀.2.2.1)
    (i : Fin L.m) (τ : Fin L.S) (j : Fin L.n) :
    coinTerm L ω₁ i τ j - coinTerm L ω₀ i τ j = laneTerm L ω₁ i τ j - laneTerm L ω₀ i τ j := by
  unfold coinTerm Coins.G
  rw [h1, h2, h3]
  ring

theorem word_bumpB (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (i₀ : Fin L.m) {t₀ : Fin L.S} (h0 : (t₀ : ℕ) ≠ 0)
    (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L κ (bumpB i₀ t₀) i t j - word L κ (zeroC L) i t j = if i = i₀ ∧ t₀ ≤ t then 1 else 0 := by
  rw [word_sub, Finset.sum_eq_single t₀]
  · rw [coinTerm_sub_of_lanes (bumpB i₀ t₀) (zeroC L) rfl, ite_eq_right h0, ite_eq_right h0]
    simp only [bumpB, zeroC, and_true]
    by_cases hi : i = i₀ <;> by_cases ht : t₀ ≤ t <;> simp [hi, ht]
  · intro τ _ hτ
    rw [coinTerm_sub_of_lanes (bumpB i₀ t₀) (zeroC L) rfl]
    by_cases h : (τ : ℕ) = 0
    · simp [h, bumpB, zeroC, Coins.G]
    · simp [h, bumpB, zeroC, hτ]
  · simp

theorem word_bumpA (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (i₀ : Fin L.m) (q₀ : Fin 4) (i : Fin L.m)
    (t : Fin L.S) (j : Fin L.n) :
    word L κ (bumpA i₀ q₀) i t j - word L κ (zeroC L) i t j = if i = i₀ then colCode j q₀ else 0 := by
  have hS : 0 < L.S := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  rw [word_sub, Finset.sum_eq_single ⟨0, hS⟩]
  · rw [ite_eq_left (Fin.le_def.2 (Nat.zero_le _)), coinTerm_sub_of_lanes (bumpA i₀ q₀) (zeroC L) rfl]
    by_cases hi : i = i₀
    · subst hi
      simp [bumpA, zeroC, Coins.G, apply_ite, ite_mul, Finset.sum_ite_eq']
    · simp [hi, bumpA, zeroC, Coins.G]
  · intro τ _ hτ
    rw [coinTerm_sub_of_lanes (bumpA i₀ q₀) (zeroC L) rfl]
    have h : (τ : ℕ) ≠ 0 := fun h => hτ (Fin.ext h)
    simp [h, bumpA, zeroC]
  · simp

theorem word_bumpG (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (q₀ : Fin 4) (j₀ : Fin L.n) (i : Fin L.m)
    (t : Fin L.S) (j : Fin L.n) :
    word L κ (bumpG q₀ j₀) i t j - word L κ (zeroC L) i t j = if j = j₀ then rowTag i q₀ else 0 := by
  have hS : 0 < L.S := Nat.lt_of_le_of_lt (Nat.zero_le _) t.isLt
  rw [word_sub, Finset.sum_eq_single ⟨0, hS⟩]
  · rw [ite_eq_left (Fin.le_def.2 (Nat.zero_le _)), coinTerm_sub_of_lanes (bumpG q₀ j₀) (zeroC L) rfl]
    by_cases hj : j = j₀
    · subst hj
      simp [bumpG, zeroC, Coins.G, apply_ite, mul_sub, Finset.sum_sub_distrib, Finset.sum_ite_eq']
    · simp [hj, bumpG, zeroC, Coins.G]
  · intro τ _ hτ
    rw [coinTerm_sub_of_lanes (bumpG q₀ j₀) (zeroC L) rfl]
    have h : (τ : ℕ) ≠ 0 := fun h => hτ (Fin.ext h)
    simp [h, bumpG, zeroC]
  · simp

theorem word_bumpU (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (he : 1 < 2 ^ L.e) (htile : 0 < L.tile)
    (t₀ : Fin L.S) (k₀ : Fin L.tiles) (ℓ₀ : Fin 3) (j₀ : Fin L.n) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n) :
    word L κ (bumpU he t₀ k₀ ℓ₀ j₀) i t j - word L κ (zeroC L) i t j =
      if j = j₀ ∧ rowTile htile i = k₀ ∧ t₀ ≤ t then 2 * rotWeight i ℓ₀ else 0 := by
  rw [word_sub, Finset.sum_eq_single t₀]
  · rw [coinTerm_sub_of_tags (bumpU he t₀ k₀ ℓ₀ j₀) (zeroC L) rfl rfl rfl]
    unfold laneTerm
    simp only [dite_eq_left (tile_lt htile i)]
    by_cases ht : t₀ ≤ t
    · rw [ite_eq_left ht]
      by_cases hc : j = j₀ ∧ rowTile htile i = k₀
      · obtain ⟨rfl, hk⟩ := hc
        rw [ite_eq_left ⟨rfl, hk, ht⟩]
        simp only [rowTile] at hk
        simp [bumpU, zeroC, Coins.s, hk, zeroU, apply_ite, mul_sub, Finset.sum_sub_distrib, Finset.sum_ite_eq']
        ring
      · rw [ite_eq_right (fun h => hc ⟨h.1, h.2.1⟩)]
        have : ∀ ℓ : Fin 3, ¬ ((⟨(i : ℕ) / L.tile, tile_lt htile i⟩ : Fin L.tiles) = k₀ ∧ ℓ = ℓ₀ ∧
            j = j₀) := fun ℓ h => hc ⟨h.2.2, h.1⟩
        simp [bumpU, zeroC, Coins.s, this]
    · rw [ite_eq_right ht, ite_eq_right (fun h => ht h.2.2)]
  · intro τ _ hτ
    rw [coinTerm_sub_of_tags (bumpU he t₀ k₀ ℓ₀ j₀) (zeroC L) rfl rfl rfl]
    simp [laneTerm, bumpU, zeroC, Coins.s, hτ]
  · simp

/-! ## The coins a probe leaves alone -/

theorem coord_zeroC (x : Coord L) : coord (zeroC L) x = 0 := by
  rcases x with x | x | x | x <;> rfl

theorem coord_bumpB {i₀ : Fin L.m} {t₀ : Fin L.S} {x : Coord L} (h : x ≠ .inl (i₀, t₀)) :
    coord (bumpB i₀ t₀) x = 0 := by
  rcases x with ⟨i, t⟩ | x | x | x
  · have : ¬ (i = i₀ ∧ t = t₀) := fun h' => h (by rw [h'.1, h'.2])
    simp [coord, bumpB, this]
  all_goals rfl

theorem coord_bumpA {i₀ : Fin L.m} {q₀ : Fin 4} {x : Coord L} (h : x ≠ .inr (.inl (i₀, q₀))) :
    coord (bumpA i₀ q₀) x = 0 := by
  rcases x with x | ⟨i, q⟩ | x | x
  · rfl
  · have : ¬ (i = i₀ ∧ q = q₀) := fun h' => h (by rw [h'.1, h'.2])
    simp [coord, bumpA, this]
  all_goals rfl

theorem coord_bumpG {q₀ : Fin 4} {j₀ : Fin L.n} {x : Coord L} (h : x ≠ .inr (.inr (.inl (q₀, j₀)))) :
    coord (bumpG q₀ j₀) x = 0 := by
  rcases x with x | x | ⟨q, j⟩ | x
  · rfl
  · rfl
  · have : ¬ (q = q₀ ∧ j = j₀) := fun h' => h (by rw [h'.1, h'.2])
    simp [coord, bumpG, this]
  · rfl

theorem coord_bumpU (he : 1 < 2 ^ L.e) {t₀ : Fin L.S} {k₀ : Fin L.tiles} {ℓ₀ : Fin 3} {j₀ : Fin L.n}
    {x : Coord L} (h : x ≠ .inr (.inr (.inr (t₀, k₀, ℓ₀, j₀)))) : coord (bumpU he t₀ k₀ ℓ₀ j₀) x = 0 := by
  rcases x with x | x | x | ⟨t, k, ℓ, j⟩
  · rfl
  · rfl
  · rfl
  · have : ¬ (t = t₀ ∧ k = k₀ ∧ ℓ = ℓ₀ ∧ j = j₀) := fun h' => h (by rw [h'.1, h'.2.1, h'.2.2.1, h'.2.2.2])
    simp [coord, bumpU, this, zeroU]

/-! ## The tag and the code -/

theorem bal13_inj {z z' : ℤ} (h1 : -14280 ≤ z) (h2 : z ≤ 14280) (h1' : -14280 ≤ z') (h2' : z' ≤ 14280)
    (h : bal13 z = bal13 z') : z = z' := by
  have e0 := congrFun h 0
  have e1 := congrFun h 1
  have e2 := congrFun h 2
  have e3 := congrFun h 3
  simp only [bal13, Fin.val_zero, Fin.val_one, Fin.val_two, pow_zero, pow_one] at e0 e1 e2 e3
  norm_num at e0 e1 e2 e3
  omega

theorem bal13_ne_zero {z : ℤ} (h1 : -14280 ≤ z) (h2 : z ≤ 28560) (hz : z ≠ 0) : ∃ q, bal13 z q ≠ 0 := by
  by_contra hc
  push Not at hc
  have e0 := hc 0
  have e1 := hc 1
  have e2 := hc 2
  have e3 := hc 3
  simp only [bal13, Fin.val_zero, Fin.val_one, Fin.val_two, pow_zero, pow_one] at e0 e1 e2 e3
  norm_num at e0 e1 e2 e3
  omega

/-- Row `i`'s tag value `(⌊i/2⌋ + 1)·(−1)^i`. -/
theorem rowTag_eq (i : ℕ) :
    rowTag i = bal13 (if i % 2 = 0 then ((i / 2 + 1 : ℕ) : ℤ) else -((i / 2 + 1 : ℕ) : ℤ)) := rfl

theorem rowTag_inj {i i' : ℕ} (hi : i < 28560) (hi' : i' < 28560) (h : rowTag i = rowTag i') : i = i' := by
  rw [rowTag_eq, rowTag_eq] at h
  have := bal13_inj (by split_ifs <;> omega) (by split_ifs <;> omega) (by split_ifs <;> omega)
    (by split_ifs <;> omega) h
  split_ifs at this <;> omega

theorem rowTag_ne_zero {i : ℕ} (hi : i < 28560) : ∃ q, rowTag i q ≠ 0 := by
  rw [rowTag_eq]
  exact bal13_ne_zero (by split_ifs <;> omega) (by split_ifs <;> omega) (by split_ifs <;> omega)

theorem rotWeight_ne_zero (i : ℕ) (ℓ : Fin 3) : rotWeight i ℓ ≠ 0 := by
  unfold rotWeight; split_ifs <;> norm_num

/-! ## Shift separation -/

/-- **The lead's words are shift-separated** on its coins, for at most 28,560 rows, any number of columns, nonzero
coin tiles and coin lanes of at least one bit. -/
theorem leadWords_shift (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (hm : L.m ≤ 28560) (he : 0 < L.e)
    (htile : 0 < L.tile) : ShiftSeparatedOn Set.univ (FreeLead L) (leadWords L κ) := by
  have he' : 1 < 2 ^ L.e := Nat.one_lt_two_pow (by omega)
  have cast_sub : ∀ (ω₁ ω₀ : Coins L) (i : Fin L.m) (t : Fin L.S) (j : Fin L.n),
      (word L κ ω₁ i t j : ℚ) - word L κ ω₀ i t j = ((word L κ ω₁ i t j - word L κ ω₀ i t j : ℤ) : ℚ) :=
    fun _ _ _ _ _ => by push_cast; ring
  refine ⟨?_, ?_⟩
  · simp only [leadWords, Finset.mem_image, Finset.mem_univ, true_and]
    rintro _ ⟨⟨i, t, j⟩, rfl⟩ _ ⟨⟨i', t', j'⟩, rfl⟩ ⟨q, hq⟩
    have P := fun ω₁ => shift_probe hq (zeroC L) ω₁
    simp only [cast_sub] at P
    by_contra hne
    have hne' : (i, t, j) ≠ (i', t', j') := fun h => hne (by simp only [Prod.mk.injEq] at h; rw [h.1, h.2.1, h.2.2])
    rcases lt_trichotomy t t' with htt | rfl | htt
    · have h0 : (t' : ℕ) ≠ 0 := by have := Fin.lt_def.1 htt; omega
      have := P (bumpB i' t')
      rw [word_bumpB κ i' h0, word_bumpB κ i' h0, ite_eq_right (fun h => absurd h.2 (not_le.2 htt)),
        ite_eq_left ⟨rfl, le_rfl⟩] at this
      norm_num at this
    · by_cases hj : j' = j
      · subst hj
        have hi : i ≠ i' := fun h => hne' (by rw [h])
        by_cases ht0 : (t : ℕ) = 0
        · obtain ⟨q₀, hq₀⟩ : ∃ q₀, rowTag i q₀ ≠ rowTag i' q₀ := by
            by_contra hc; push Not at hc
            exact hi (Fin.ext (rowTag_inj (by omega) (by omega) (funext hc)))
          have := P (bumpG q₀ j')
          rw [word_bumpG, word_bumpG, ite_eq_left rfl, ite_eq_left rfl] at this
          exact hq₀ (by exact_mod_cast this)
        · have := P (bumpB i t)
          rw [word_bumpB κ i ht0, word_bumpB κ i ht0, ite_eq_left ⟨rfl, le_rfl⟩,
            ite_eq_right (fun h => hi h.1.symm)] at this
          norm_num at this
      · have := P (bumpU he' t (rowTile htile i) 0 j)
        rw [word_bumpU κ he' htile, word_bumpU κ he' htile, ite_eq_left ⟨rfl, rfl, le_rfl⟩,
          ite_eq_right (fun h => hj h.1)] at this
        have h2 : (2 * rotWeight i 0 : ℤ) = 0 := by exact_mod_cast this
        exact rotWeight_ne_zero i 0 (by omega)
    · have h0 : (t : ℕ) ≠ 0 := by have := Fin.lt_def.1 htt; omega
      have := P (bumpB i t)
      rw [word_bumpB κ i h0, word_bumpB κ i h0, ite_eq_left ⟨rfl, le_rfl⟩,
        ite_eq_right (fun h => absurd h.2 (not_le.2 htt))] at this
      norm_num at this
  · simp only [leadWords, Finset.mem_image, Finset.mem_univ, true_and]
    rintro _ ⟨⟨i, t, j⟩, rfl⟩ f hf ⟨q, hq⟩
    have P : ∀ ω₁, f ω₁ - f (zeroC L) = ((word L κ ω₁ i t j - word L κ (zeroC L) i t j : ℤ) : ℚ) := fun ω₁ => by
      rw [← cast_sub]; exact shift_probe hq (zeroC L) ω₁
    -- two coins that move the word, as probes with the coin each sets
    obtain ⟨y₁, ω₁, hy₁, hω₁, y₂, ω₂, hy₂, hω₂, hy⟩ : ∃ (y₁ : Coord L) (ω₁ : Coins L),
        (∀ x, x ≠ y₁ → coord ω₁ x = 0) ∧ word L κ ω₁ i t j - word L κ (zeroC L) i t j ≠ 0 ∧
        ∃ (y₂ : Coord L) (ω₂ : Coins L),
        (∀ x, x ≠ y₂ → coord ω₂ x = 0) ∧ word L κ ω₂ i t j - word L κ (zeroC L) i t j ≠ 0 ∧ y₁ ≠ y₂ := by
      have hU : word L κ (bumpU he' t (rowTile htile i) 0 j) i t j - word L κ (zeroC L) i t j ≠ 0 := by
        rw [word_bumpU κ he' htile, ite_eq_left ⟨rfl, rfl, le_rfl⟩]
        have := rotWeight_ne_zero i 0; omega
      by_cases ht0 : (t : ℕ) = 0
      · obtain ⟨q₀, hq₀⟩ := rowTag_ne_zero (i := i) (by omega)
        refine ⟨_, bumpG q₀ j, fun x hx => coord_bumpG hx, ?_, _, bumpU he' t (rowTile htile i) 0 j,
          fun x hx => coord_bumpU he' hx, hU, by simp⟩
        rw [word_bumpG, ite_eq_left rfl]; exact hq₀
      · refine ⟨_, bumpB i t, fun x hx => coord_bumpB hx, ?_, _, bumpU he' t (rowTile htile i) 0 j,
          fun x hx => coord_bumpU he' hx, hU, by simp⟩
        rw [word_bumpB κ i ht0, ite_eq_left ⟨rfl, le_rfl⟩]; norm_num
    rcases hf with ⟨c, rfl⟩ | ⟨x, hx⟩
    · have := P ω₁
      simp only [sub_self] at this
      exact hω₁ (by exact_mod_cast this.symm)
    · have key : ∀ ω y, (∀ x', x' ≠ y → coord ω x' = 0) → y ≠ x → f ω = f (zeroC L) := fun ω y hω hyx =>
        hx ω (zeroC L) (by rw [hω x (Ne.symm hyx), coord_zeroC])
      by_cases h1 : y₁ = x
      · have := P ω₂
        rw [key ω₂ y₂ hy₂ (fun h => hy (h1.trans h.symm)), sub_self] at this
        exact hω₂ (by exact_mod_cast this.symm)
      · have := P ω₁
        rw [key ω₁ y₁ hy₁ h1, sub_self] at this
        exact hω₁ (by exact_mod_cast this.symm)

end Pouw.SecurityProofs.Lead.Tags
