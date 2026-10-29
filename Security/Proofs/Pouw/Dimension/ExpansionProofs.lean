import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.LinearCombination

/-!
# The affine coin expansion: two steps' offsets are uniform on a prime modulus (proofs)

A column's step offsets `w_t = u + t·v` expand two coins `u, v ∈ ℤ/q` drawn after the data.

* **Prime `q`** (`pair_card`, `pair_mem_card`): for steps `t ≠ t'` the map `(u, v) ↦ (w_t, w_t')` is a bijection of
  `(ℤ/q)²`, so each pair of offsets has exactly one preimage, and for any target sets `X, Y` fixed before the draw the
  events `w_t ∈ X` and `w_t' ∈ Y` hold together on `|X|·|Y|` of the `q²` draws: independent, each at its marginal
  (`step_mem_card`, `|X|·q`).
* **Composite `q`** (`pair_card_255`): at `q = 255` and steps 0 and 3, the offsets `(0, 0)` have three preimages, three
  times the independent share, because `3` is no unit mod 255.

This is the pairwise independence a second-moment tail over a column's step events reads.
-/

namespace Pouw.Dimension.Expansion

open Finset

variable {q : ℕ}

/-- The offset of step `t` from the coins `(u, v)`. -/
def offset (t : ZMod q) (uv : ZMod q × ZMod q) : ZMod q := uv.1 + t * uv.2

theorem pair_card [Fact q.Prime] {t t' : ZMod q} (h : t ≠ t') (a b : ZMod q) :
    (univ.filter fun uv => offset t uv = a ∧ offset t' uv = b).card = 1 := by
  have hd : t - t' ≠ 0 := sub_ne_zero.mpr h
  rw [card_eq_one]
  refine ⟨(a - t * ((a - b) / (t - t')), (a - b) / (t - t')), ?_⟩
  ext ⟨u, v⟩
  rw [mem_filter, mem_singleton]
  simp only [offset, mem_univ, true_and, Prod.mk.injEq]
  constructor
  · rintro ⟨ha, hb⟩
    have hv : v = (a - b) / (t - t') := by
      rw [eq_div_iff hd, ← ha, ← hb]; ring
    exact ⟨by rw [← hv, ← ha]; ring, hv⟩
  · rintro ⟨rfl, rfl⟩
    refine ⟨by ring, ?_⟩
    field_simp
    ring

theorem pair_mem_card [Fact q.Prime] {t t' : ZMod q} (h : t ≠ t') (X Y : Finset (ZMod q)) :
    (univ.filter fun uv => offset t uv ∈ X ∧ offset t' uv ∈ Y).card = X.card * Y.card := by
  rw [card_eq_sum_card_fiberwise (f := fun uv => (offset t uv, offset t' uv)) (t := X ×ˢ Y)
    (fun uv huv => by rw [mem_coe, mem_filter] at huv; rw [mem_coe]; exact mem_product.mpr huv.2), ← card_product,
    card_eq_sum_ones (X ×ˢ Y)]
  refine sum_congr rfl fun ab hab => ?_
  rw [← pair_card h ab.1 ab.2]
  congr 1
  ext uv
  rw [mem_filter, mem_filter, mem_filter]
  simp only [mem_univ, true_and, Prod.ext_iff]
  constructor
  · rintro ⟨-, h1, h2⟩; exact ⟨h1, h2⟩
  · rintro ⟨h1, h2⟩
    have := mem_product.mp hab
    exact ⟨⟨h1 ▸ this.1, h2 ▸ this.2⟩, h1, h2⟩

theorem step_mem_card [NeZero q] (t : ZMod q) (X : Finset (ZMod q)) :
    (univ.filter fun uv => offset t uv ∈ X).card = X.card * q := by
  have : (univ.filter fun uv : ZMod q × ZMod q => offset t uv ∈ X) =
      (X ×ˢ (univ : Finset (ZMod q))).map ⟨fun xv => (xv.1 - t * xv.2, xv.2), by
        rintro ⟨x1, x2⟩ ⟨y1, y2⟩ h
        simp only [Prod.mk.injEq] at h ⊢
        obtain ⟨h1, rfl⟩ := h
        exact ⟨by linear_combination h1, rfl⟩⟩ := by
    ext ⟨u, v⟩
    rw [mem_filter, mem_map]
    simp only [offset, mem_univ, true_and, mem_product, Prod.exists]
    constructor
    · intro hx
      exact ⟨u + t * v, v, ⟨hx, trivial⟩, Prod.ext (show u + t * v - t * v = u by ring) rfl⟩
    · rintro ⟨x, w, ⟨hx, -⟩, he⟩
      have he' : (x - t * w, w) = (u, v) := he
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he'
      rwa [sub_add_cancel]
  rw [this, card_map, card_product, card_univ, ZMod.card]

/-- At `q = 255`, steps 0 and 3 send `(0, 0)`, `(0, 85)` and `(0, 170)` to the same offsets. -/
theorem pair_card_255 : 3 ≤ (univ.filter fun uv => offset (0 : ZMod 255) uv = 0 ∧ offset 3 uv = 0).card := by
  have hL : ({(0, 0), (0, 85), (0, 170)} : Finset (ZMod 255 × ZMod 255)).card = 3 := by decide
  rw [← hL]
  refine card_le_card fun uv huv => ?_
  simp only [mem_insert, mem_singleton] at huv
  rw [mem_filter]
  simp only [offset, mem_univ, true_and]
  rcases huv with rfl | rfl | rfl <;> exact ⟨rfl, rfl⟩

end Pouw.Dimension.Expansion
