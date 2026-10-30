import Proofs.Pouw.Lead.Tail.Words
import Proofs.Pouw.Lead.Tail.Fiber

/-!
# One column block's slots with a tile repeated within a block

`slots_tail`: at a call of the lead whose coin tiles are whole (or which has one row block), `x` or more of one column
block's slots (step, coin tile) hold two equal tiles on at most a `λ_b^x / x!` share of the coins,
`λ_b = pairs · S · tiles / 2^16` (`lamB`).

A slot's own coordinates are 64 fresh bits, one a row of its coin tile (`setBits`): digit 0 of the step-0 bits `a₁` at
step 0, the tag bits `b` of its step at a later step. With whole coin tiles the coins are the rest (`b` at step 0,
`a₁`'s digits 1 to 3, `G`, the coin lanes), drawn first, and a product over the slots in slot order (step, then coin
tile). `count_tail_fiber` then needs:

* **Reads**: a slot's tiles are prefix sums over steps up to its own, of its own coin tile's rows: the bits of slots
  at or before it.
* **The fiber** (`slots_fiber`): two tiles at one step, column block and coin tile differ, at each of their 16 row
  pairs, by a term the slot's bits don't move and the two rows' fresh bits' difference: `b` once, or `a₁`'s digit 0
  against a column code digit 0 that is nonzero at one of the block's first two columns (it is zero only at
  `j ≡ 12 (mod 13)`). With every other coin fixed, each pair of rows meets it on at most half of the first row's bit
  values: `2^−16` a pair of tiles, over `pairs` pairs.
* **The sum**: `S · tiles · pairs / 2^16 = lamB`.
-/

namespace Pouw.SecurityProofs.Lead.Tail

open Pouw.Lead

/-- `ω` with slot `(t, ct)`'s fresh bits set to `v`, row `i` of coin tile `ct` taking `v (i % 64)`: digit 0 of the
step-0 bits `a₁` at step 0, and the tag bits `b` of step `t` at a later step. -/
def setBits (L : Pouw.Lead.Layout) (ω : Coins L) (t : Fin L.S) (ct : Fin L.tiles) (v : Fin 64 → Fin 2) : Coins L :=
  (fun i τ => if (t : ℕ) ≠ 0 ∧ τ = t ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.1 i τ,
    fun i q => if (t : ℕ) = 0 ∧ (q : ℕ) = 0 ∧ (i : ℕ) / 64 = ct then v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩
      else ω.2.1 i q,
    ω.2.2.1, ω.2.2.2)

open Finset in
/-- **A slot's bits move its coin tile's words at its step** by `b` once, or at step 0 by `a₁`'s digit 0 against the
column code's digit 0. -/
theorem word_setBits (L : Pouw.Lead.Layout) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (t : Fin L.S)
    (ct : Fin L.tiles) (v v' : Fin 64 → Fin 2) (i : Fin L.m) (hi : (i : ℕ) / 64 = ct) (j : Fin L.n) :
    word L κ (setBits L ω t ct v) i t j - word L κ (setBits L ω t ct v') i t j =
      (if (t : ℕ) = 0 then colCode j 0 else 1) *
        (((v ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ) -
          ((v' ⟨(i : ℕ) % 64, Nat.mod_lt _ (by norm_num)⟩ : ℕ) : ℤ)) := by
  rw [Tags.word_sub, sum_eq_single t]
  · rw [ite_eq_left le_rfl, Tags.coinTerm_sub_of_lanes (setBits L ω t ct v) (setBits L ω t ct v') rfl]
    by_cases ht : (t : ℕ) = 0
    · simp only [ht, ite_true, setBits, hi, Fin.sum_univ_four, true_and, and_true, Coins.G]
      simp
      ring
    · simp [ht, setBits, hi]
  · intro τ _ hτ
    split_ifs with h
    · rw [Tags.coinTerm_sub_of_lanes (setBits L ω t ct v) (setBits L ω t ct v') rfl]
      by_cases ht : (t : ℕ) = 0
      · have : (τ : ℕ) ≠ 0 := fun h0 => hτ (Fin.ext (h0.trans ht.symm))
        simp [setBits, ht, this]
      · simp [setBits, ht, hτ, Coins.G]
    · rfl
  · intro h
    exact absurd (mem_univ t) h

open Classical Finset in
/-- `slots_fiber` on any layout of whole MMA tiles. -/
theorem slots_fiber_of (L : Pouw.Lead.Layout) (hLm : L.m = 16 * L.rowBlocks) (hLn : L.n = 8 * L.colBlocks)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) (cb : Fin L.colBlocks) (t : Fin L.S) (ct : Fin L.tiles) :
    ((Finset.univ.filter fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)).card : ℝ) ≤
      (pairs L : ℝ) / 2 ^ 16 * Fintype.card (Fin 64 → Fin 2) := by
  obtain ⟨c₀, hc₀, hcoef⟩ : ∃ c₀ < 8, (if (t : ℕ) = 0 then colCode (8 * cb + c₀) 0 else 1) ≠ 0 := by
    by_cases h12 : (8 * (cb : ℕ)) % 13 = 12
    · refine ⟨1, by norm_num, ?_⟩
      split_ifs
      · simp only [colCode, bal13, Fin.val_zero, pow_zero, Int.ediv_one]; push_cast; omega
      · norm_num
    · refine ⟨0, by norm_num, ?_⟩
      split_ifs
      · simp only [colCode, bal13, Fin.val_zero, pow_zero, Int.ediv_one]; push_cast; omega
      · norm_num
  have hcard := card_filter_le_cover (V := Fin 64 → Fin 2) (W := Fin 48 → Fin 2)
    (fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)) (pairsAt L ct)
    (fun p v => tileOf L κ (setBits L ω t ct v) (4 * ct + p.1) cb t =
      tileOf L κ (setBits L ω t ct v) (4 * ct + p.2) cb t)
    (fun v hv => withinAt_pair hv) fun p hp => ⟨offBlock p.2, fun v v' hv hv' hvv => funext fun k => by
      simp only [pairsAt, mem_filter, mem_product, mem_range] at hp
      obtain ⟨⟨ha4, hb4⟩, hab, hbR⟩ := hp
      have hoff := offBlock_eq hb4 hvv
      by_cases hk : (k : ℕ) / 16 = p.2
      · have hr : (k : ℕ) % 16 < 16 := Nat.mod_lt _ (by norm_num)
        have hrbA : 4 * (ct : ℕ) + p.1 < L.rowBlocks := by omega
        have hcb := cb.isLt
        have eA := congrFun hv ⟨8 * ((k : ℕ) % 16) + c₀, by unfold tileWords; omega⟩
        have eA' := congrFun hv' ⟨8 * ((k : ℕ) % 16) + c₀, by unfold tileWords; omega⟩
        rw [tileOf_at hLm hLn _ _ hrbA hcb t hr hc₀, tileOf_at hLm hLn _ _ hbR hcb t hr hc₀] at eA eA'
        have dA := word_setBits L κ ω t ct v v' ⟨16 * (4 * ct + p.1) + (k : ℕ) % 16, by omega⟩ (by simp; omega)
          ⟨8 * cb + c₀, by omega⟩
        have dB := word_setBits L κ ω t ct v v' ⟨16 * (4 * ct + p.2) + (k : ℕ) % 16, by omega⟩ (by simp; omega)
          ⟨8 * cb + c₀, by omega⟩
        have hvA := hoff ⟨(16 * (4 * ct + p.1) + (k : ℕ) % 16) % 64, Nat.mod_lt _ (by norm_num)⟩ (by simp; omega)
        rw [hvA, sub_self, mul_zero] at dA
        have hk' : (⟨(16 * (4 * ct + p.2) + (k : ℕ) % 16) % 64, Nat.mod_lt _ (by norm_num)⟩ : Fin 64) = k :=
          Fin.ext (by simp only; omega)
        rw [hk'] at dB
        have h0 : (if (t : ℕ) = 0 then colCode (8 * cb + c₀) 0 else 1) * (((v k : ℕ) : ℤ) - ((v' k : ℕ) : ℤ)) = 0 := by
          rw [← dB]
          linarith
        rcases mul_eq_zero.1 h0 with h | h
        · exact absurd h hcoef
        · exact Fin.ext (by exact_mod_cast sub_eq_zero.1 h)
      · exact hoff k hk⟩
  rw [Fintype.card_fun, Fintype.card_fin, Fintype.card_fin] at hcard ⊢
  calc (#(univ.filter fun v => withinAt L κ (setBits L ω t ct v) (t, ct, cb)) : ℝ) ≤ (#(pairsAt L ct) * 2 ^ 48 : ℕ) := by
        exact_mod_cast hcard
    _ ≤ (pairs L * 2 ^ 48 : ℕ) := by exact_mod_cast Nat.mul_le_mul_right _ (pairsAt_card L ct)
    _ = (pairs L : ℝ) / 2 ^ 16 * ((2 ^ 64 : ℕ) : ℝ) := by push_cast; ring

open Classical in
/-- **A slot's fresh bits rarely make two tiles of a block agree**: with every other coin fixed, slot `(t, ct)` of
column block `cb` holds two equal tiles on at most a `pairs / 2^16` share of its fresh bits' values. -/
theorem slots_fiber (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (ω : Coins (lead T m n K)) (cb : Fin (lead T m n K).colBlocks) (t : Fin (lead T m n K).S)
    (ct : Fin (lead T m n K).tiles) :
    ((Finset.univ.filter fun v => withinAt _ κ (setBits _ ω t ct v) (t, ct, cb)).card : ℝ) ≤
      (pairs (lead T m n K) : ℝ) / 2 ^ 16 * Fintype.card (Fin 64 → Fin 2) :=
  slots_fiber_of _ (lead_m T m n K) (lead_n T m n K) κ ω cb t ct

/-- **One column block's slots with a repeat inside a block are rarely many**: `x` or more of them on at most a
`λ_b^x / x!` share of the coins, when the coin tiles are whole or there is one row block. -/
theorem slots_tail (T m n K : ℕ) (hm : 64 ∣ m ∨ m ≤ 16)
    (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (cb : Fin (lead T m n K).colBlocks) (x : ℕ) :
    Pouw.pr (fun ω : Coins (lead T m n K) => x ≤ (hitSlots _ κ ω cb).card) ≤
      lamB (lead T m n K) ^ x / x.factorial := by
  sorry

end Pouw.SecurityProofs.Lead.Tail
