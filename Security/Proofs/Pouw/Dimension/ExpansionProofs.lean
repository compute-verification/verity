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

* **The two moments** (`sum_hitSteps`, `sum_hitPairs`): for `S` distinct steps with targets of at most `P` residues
  each, the hit count `C` has `E[C] ≤ S·p*` and, on a prime modulus, `E[C(C − 1)] ≤ S(S − 1)·p*²` with `p* = P/q`.
  These are what a second-moment tail over a column's step events reads; the tail itself isn't proved here.
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

variable {S : ℕ}

/-- The steps whose offsets land in their targets. -/
def hitSteps (t : Fin S → ZMod q) (X : Fin S → Finset (ZMod q)) (uv : ZMod q × ZMod q) : Finset (Fin S) :=
  univ.filter fun i => offset (t i) uv ∈ X i

/-- **First moment**: if each step's target holds at most `P` residues, the hit steps summed over the `q²` draws are at
most `S·P·q`, so `E[C] ≤ S·p*` with `p* = P/q`. -/
theorem sum_hitSteps [NeZero q] (t : Fin S → ZMod q) (X : Fin S → Finset (ZMod q)) (P : ℕ)
    (hX : ∀ i, (X i).card ≤ P) : ∑ uv, (hitSteps t X uv).card ≤ S * P * q := by
  simp only [hitSteps, card_filter]
  rw [sum_comm]
  calc ∑ i : Fin S, ∑ uv : ZMod q × ZMod q, (if offset (t i) uv ∈ X i then 1 else 0)
      = ∑ i : Fin S, (X i).card * q := by
        refine sum_congr rfl fun i _ => ?_
        rw [← card_filter, step_mem_card]
    _ ≤ ∑ _i : Fin S, P * q := sum_le_sum fun i _ => Nat.mul_le_mul_right _ (hX i)
    _ = S * P * q := by rw [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id, mul_assoc]

/-- **Second moment**: for distinct steps on a prime modulus, the ordered pairs of distinct hit steps summed over the
draws are at most `(S² − S)·P²`, so `E[C(C − 1)] ≤ S(S − 1)·p*²`, the pairwise term a second-moment tail reads. -/
theorem sum_hitPairs [Fact q.Prime] (t : Fin S → ZMod q) (ht : Function.Injective t) (X : Fin S → Finset (ZMod q))
    (P : ℕ) (hX : ∀ i, (X i).card ≤ P) : ∑ uv, (hitSteps t X uv).offDiag.card ≤ (S * S - S) * (P * P) := by
  have hoff : ∀ uv, (hitSteps t X uv).offDiag = (univ : Finset (Fin S)).offDiag.filter
      fun ii => offset (t ii.1) uv ∈ X ii.1 ∧ offset (t ii.2) uv ∈ X ii.2 := by
    intro uv
    ext ii
    rw [mem_filter, mem_offDiag, mem_offDiag, hitSteps, mem_filter, mem_filter]
    simp only [mem_univ, true_and]
    tauto
  simp only [hoff, card_filter]
  rw [sum_comm]
  calc ∑ ii ∈ (univ : Finset (Fin S)).offDiag, ∑ uv : ZMod q × ZMod q,
        (if offset (t ii.1) uv ∈ X ii.1 ∧ offset (t ii.2) uv ∈ X ii.2 then 1 else 0)
      = ∑ ii ∈ (univ : Finset (Fin S)).offDiag, (X ii.1).card * (X ii.2).card := by
        refine sum_congr rfl fun ii hii => ?_
        rw [← card_filter, pair_mem_card (ht.ne (mem_offDiag.mp hii).2.2)]
    _ ≤ ∑ _ii ∈ (univ : Finset (Fin S)).offDiag, P * P := sum_le_sum fun ii _ => Nat.mul_le_mul (hX _) (hX _)
    _ = (S * S - S) * (P * P) := by
        rw [sum_const, offDiag_card, card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_id]

end Pouw.Dimension.Expansion
