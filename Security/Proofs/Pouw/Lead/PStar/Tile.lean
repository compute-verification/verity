import Proofs.Pouw.Lead.PStar.Walks

/-!
# The exact `p*` of an MMA tile under `-rot` (proofs)

At tile grain a free copy moves a whole MMA tile, 16 rows × 8 columns, so what a value fixed before the draw must hit
is a tile. At a step `t ≥ 1` the tile's rows each draw their own bit and its columns each their own triple, shared
by the tile's rows: `2^16 · (2^21)^8 = 2^184` draws, the tile's word at row `r`, column `q` being
`c + b_r + Σ_ℓ A′[r, ℓ]·s_q,ℓ` with `c` fixed before the draw.

* **The two classes fix the triple** (`rotPair_inj`): `(127, 1, 1)·s` and `(1, 127, 1)·s` together determine `s`.
* **The step is injective** (`tileStep_injOn`): on any tile holding an even and an odd row and a column, the parity of
  a word fixes its row's bit, since the coin sum is odd, and then an even and an odd row fix each column's triple.
* **`p*` exactly** (`tile_fiber_le`, `mmaTile_pstar`): a value fixed before the draw is the tile's step on at most one
  of its draws, and on exactly one when it is a step's value: `p* = 2^−184` on a 16 × 8 tile, whatever the data
  and the earlier steps.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Fintype

/-- **The two row classes fix the triple**: two coin triples with the same even rows' sum and the same odd rows'
sum are equal. -/
theorem rotPair_inj {s s' : ℤ × ℤ × ℤ} (hs : s ∈ triples) (hs' : s' ∈ triples) (he : evenSum s = evenSum s')
    (ho : oddSum s = oddSum s') : s = s' := by
  obtain ⟨x, y, z⟩ := s
  obtain ⟨x', y', z'⟩ := s'
  rw [mem_triples] at hs hs'
  obtain ⟨a, -, -, rfl⟩ := mem_coins.mp hs.1
  obtain ⟨b, -, -, rfl⟩ := mem_coins.mp hs.2.1
  obtain ⟨c, hc1, hc2, rfl⟩ := mem_coins.mp hs.2.2
  obtain ⟨a', -, -, rfl⟩ := mem_coins.mp hs'.1
  obtain ⟨b', -, -, rfl⟩ := mem_coins.mp hs'.2.1
  obtain ⟨c', hc1', hc2', rfl⟩ := mem_coins.mp hs'.2.2
  simp only [evenSum, oddSum] at he ho
  have h : a = a' ∧ b = b' ∧ c = c' := by omega
  rw [h.1, h.2.1, h.2.2]

variable {R Q : Type*} [Fintype R] [DecidableEq R] [Fintype Q] [DecidableEq Q]

/-- A tile's draws at one step: a bit a row and a coin triple a column. -/
def tileDraws (R Q : Type*) [Fintype R] [DecidableEq R] [Fintype Q] [DecidableEq Q] :
    Finset ((R → ℤ) × (Q → ℤ × ℤ × ℤ)) :=
  (piFinset fun _ : R => ({0, 1} : Finset ℤ)) ×ˢ piFinset fun _ : Q => triples

theorem card_tileDraws : (tileDraws R Q).card = 2 ^ card R * (128 ^ 3) ^ card Q := by
  rw [tileDraws, card_product, card_steps, card_steps, card_bits, card_triples]

/-- The tile's words at one step: row `r` (the call's row `row r`) and column `q` add `c r q`, fixed before the draw, to
their row's increment from the row's bit and the column's triple. -/
def tileStep (row : R → ℕ) (c : R → Q → ℤ) (x : (R → ℤ) × (Q → ℤ × ℤ × ℤ)) : R → Q → ℤ :=
  fun r q => rotLane (c r q) (row r) (x.1 r) (x.2 q)

/-- **The step is injective** on a tile holding an even row, an odd row and a column. -/
theorem tileStep_injOn (row : R → ℕ) {r₀ r₁ : R} (h₀ : row r₀ % 2 = 0) (h₁ : row r₁ % 2 = 1) [Nonempty Q]
    (c : R → Q → ℤ) : Set.InjOn (tileStep row c) (tileDraws R Q) := by
  rintro ⟨b, s⟩ hx ⟨b', s'⟩ hx' he
  rw [mem_coe, tileDraws, mem_product, mem_piFinset, mem_piFinset] at hx hx'
  dsimp only at hx hx'
  have hrq : ∀ r q, b r + rotSum (row r) (s q) = b' r + rotSum (row r) (s' q) := fun r q => by
    have := congrFun (congrFun he r) q
    simp only [tileStep, rotLane] at this
    omega
  have hb : b = b' := funext fun r => by
    obtain ⟨q⟩ := ‹Nonempty Q›
    obtain ⟨m, hm⟩ := rotSum_odd (row r) (hx.2 q)
    obtain ⟨m', hm'⟩ := rotSum_odd (row r) (hx'.2 q)
    have h := hrq r q
    have hr := hx.1 r
    have hr' := hx'.1 r
    simp only [mem_insert, mem_singleton] at hr hr'
    omega
  subst hb
  have hs : s = s' := funext fun q => by
    have he₀ := hrq r₀ q
    have ho₁ := hrq r₁ q
    simp only [rotSum, h₀, ite_true, show ¬ row r₁ % 2 = 0 by omega, ite_false, add_right_inj] at he₀ ho₁
    exact rotPair_inj (hx.2 q) (hx'.2 q) he₀ ho₁
  rw [hs]

/-- **No value is the tile's step on more than one draw.** -/
theorem tile_fiber_le (row : R → ℕ) {r₀ r₁ : R} (h₀ : row r₀ % 2 = 0) (h₁ : row r₁ % 2 = 1) [Nonempty Q]
    (c : R → Q → ℤ) (y : R → Q → ℤ) : ((tileDraws R Q).filter fun x => tileStep row c x = y).card ≤ 1 :=
  card_le_one.mpr fun _ hx _ hx' =>
    tileStep_injOn row h₀ h₁ c (mem_filter.mp hx).1 (mem_filter.mp hx').1
      ((mem_filter.mp hx).2.trans (mem_filter.mp hx').2.symm)

/-- The call's rows of the MMA tile in row block `rb`: `16·rb + r`. -/
def mmaRow (rb : ℕ) (r : Fin 16) : ℕ := 16 * rb + r

/-- **`p*` exactly, on a 16 × 8 MMA tile**: of the `2^184` draws of a step `t ≥ 1` (a bit for each of the tile's 16
rows, the call's rows `16·rb` to `16·rb + 15`, and a triple for each of its 8 columns), a value fixed before the draw
is the tile's step on at most one, and some value is it on one. -/
theorem mmaTile_pstar (rb : ℕ) (c : Fin 16 → Fin 8 → ℤ) :
    (∀ y, 2 ^ 184 * ((tileDraws (Fin 16) (Fin 8)).filter fun x => tileStep (mmaRow rb) c x = y).card ≤
        (tileDraws (Fin 16) (Fin 8)).card) ∧
      ∃ y, 2 ^ 184 * ((tileDraws (Fin 16) (Fin 8)).filter fun x => tileStep (mmaRow rb) c x = y).card ≥
        (tileDraws (Fin 16) (Fin 8)).card := by
  have hcard : (tileDraws (Fin 16) (Fin 8)).card = 2 ^ 184 := by
    rw [card_tileDraws, Fintype.card_fin, Fintype.card_fin]; norm_num
  have h₀ : mmaRow rb 0 % 2 = 0 := by simp only [mmaRow, Fin.val_zero]; omega
  have h₁ : mmaRow rb 1 % 2 = 1 := by simp only [mmaRow, Fin.val_one]; omega
  rw [hcard]
  refine ⟨fun y => ?_, tileStep (mmaRow rb) c ((fun _ => 0), fun _ => (1, 1, 1)), ?_⟩
  · have := tile_fiber_le (mmaRow rb) h₀ h₁ c y
    omega
  · have h1 : (1 : ℤ) ∈ coins := mem_coins.mpr ⟨0, by norm_num, by norm_num, by norm_num⟩
    have hmem : ((fun _ => (0 : ℤ)), fun _ => ((1 : ℤ), (1 : ℤ), (1 : ℤ))) ∈ tileDraws (Fin 16) (Fin 8) := by
      rw [tileDraws, mem_product, mem_piFinset, mem_piFinset]
      exact ⟨fun _ => by simp, fun _ => mem_triples.mpr ⟨h1, h1, h1⟩⟩
    have : 1 ≤ ((tileDraws (Fin 16) (Fin 8)).filter fun x => tileStep (mmaRow rb) c x =
        tileStep (mmaRow rb) c ((fun _ => 0), fun _ => (1, 1, 1))).card :=
      card_pos.mpr ⟨_, mem_filter.mpr ⟨hmem, rfl⟩⟩
    omega

end Pouw.SecurityProofs.Lead.PStar
