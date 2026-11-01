import Proofs.Pouw.Lead.PStar.AlphabetProofs

/-!
# The exact `p*` of `blc4a-t64-e7`'s three coin lanes under `-rot`'s weights (proofs)

`blc4a-t64-e7` draws three odd 7-bit coins `s = (s₀, s₁, s₂)`, each in `±127`, for each 64-row tile, step and column,
after `A_j` is held, and every row of the tile reads the same coins. `-rot` weights them `(127, 1, 1)` on even rows and
`(1, 127, 1)` on odd rows, so an entry's step increment is `c + b + 127·s₀ + s₁ + s₂` on an even row and
`c + b + s₀ + 127·s₁ + s₂` on an odd one, with `b ∈ {0, 1}` the row's bit and `c` the row lane's constant. Its `p*` is
the largest share of the draws `(b, s)` on which the increment equals a value fixed before the draw.

* **The coin sum** (`evenSum_fiber_le`, `evenSum_fiber_127`): no value is `127·s₀ + s₁ + s₂` for more than 130 of the
  128³ coin triples, and 127 is it for exactly 130. A solution with `s₂ ≠ −127` is fixed by `s₁`, since two would need
  `s₂` values 254 apart; one with `s₂ = −127` is fixed by the sign of `s₁`. The two extra preimages are the 127 weight's
  boundary cases: `127·s₀` steps by 254 and `s₁ + s₂` reaches `±254`.
* **Both classes** (`oddSum_fiber_card`): swapping `s₀` and `s₁` carries the odd rows' fibers onto the even rows', so
  both have the same law.
* **The increment** (`rotLane_fiber_le`, `rotLane_fiber_attained`): the coin sum is odd, so the parity of
  `c + b + Σ` fixes `b`, and `p* = 130/(2·128³) = 65/2,097,152` on either class, for any constant `c`
  (`rotLane_pstar`).
* **The range** (`rotSum_abs_le`): the 127 weight lets a row's coin sum reach `16,383` in magnitude.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Dimension.Alphabet

/-- The coins of one 7-bit lane: odd and in `±127`. Irreducible, so that unfolding never enumerates its 128 elements. -/
@[irreducible] def coins : Finset ℤ := oddLane 7

theorem mem_coins {x : ℤ} : x ∈ coins ↔ ∃ k : ℤ, -64 ≤ k ∧ k ≤ 63 ∧ x = 2 * k + 1 := by
  unfold coins
  simp only [oddLane, mem_image, mem_Icc]
  constructor
  · rintro ⟨k, ⟨h1, h2⟩, rfl⟩
    exact ⟨k, by norm_num at h1; omega, by norm_num at h2; omega, rfl⟩
  · rintro ⟨k, h1, h2, rfl⟩
    exact ⟨k, ⟨by norm_num; omega, by norm_num; omega⟩, rfl⟩

theorem card_coins : coins.card = 128 := by
  have h := card_oddLane 6
  unfold coins
  exact h

/-- A tile's three coins, one triple a step and column. -/
def triples : Finset (ℤ × ℤ × ℤ) := coins ×ˢ coins ×ˢ coins

theorem card_triples : triples.card = 128 ^ 3 := by
  rw [triples, card_product, card_product, card_coins]; norm_num

/-- The even rows' coin sum, weights `(127, 1, 1)`. -/
def evenSum (s : ℤ × ℤ × ℤ) : ℤ := 127 * s.1 + s.2.1 + s.2.2

/-- The odd rows' coin sum, weights `(1, 127, 1)`. -/
def oddSum (s : ℤ × ℤ × ℤ) : ℤ := s.1 + 127 * s.2.1 + s.2.2

/-- A row's coin sum under `-rot`: `(127, 1, 1)` on even rows, `(1, 127, 1)` on odd rows. -/
def rotSum (i : ℕ) (s : ℤ × ℤ × ℤ) : ℤ := if i % 2 = 0 then evenSum s else oddSum s

theorem mem_triples {s : ℤ × ℤ × ℤ} :
    s ∈ triples ↔ s.1 ∈ coins ∧ s.2.1 ∈ coins ∧ s.2.2 ∈ coins := by
  simp only [triples, mem_product]

/-- **The coin sum's largest fiber has at most 130 triples.** -/
theorem evenSum_fiber_le (v : ℤ) : (triples.filter fun s => evenSum s = v).card ≤ 130 := by
  set F := triples.filter fun s => evenSum s = v with hF
  have hmem : ∀ s ∈ F, ∃ a b c : ℤ, -64 ≤ a ∧ a ≤ 63 ∧ -64 ≤ b ∧ b ≤ 63 ∧ -64 ≤ c ∧ c ≤ 63 ∧
      s = (2 * a + 1, 2 * b + 1, 2 * c + 1) ∧ 127 * (2 * a + 1) + (2 * b + 1) + (2 * c + 1) = v := by
    intro s hs
    rw [hF, mem_filter, mem_triples] at hs
    obtain ⟨⟨h0, h1, h2⟩, hv⟩ := hs
    obtain ⟨a, ha1, ha2, ha⟩ := mem_coins.mp h0
    obtain ⟨b, hb1, hb2, hb⟩ := mem_coins.mp h1
    obtain ⟨c, hc1, hc2, hc⟩ := mem_coins.mp h2
    refine ⟨a, b, c, ha1, ha2, hb1, hb2, hc1, hc2, ?_, ?_⟩
    · rw [← ha, ← hb, ← hc]
    · rw [← ha, ← hb, ← hc]; exact hv
  have hlow : (F.filter fun s => ¬ s.2.2 = -127).card ≤ 128 := by
    rw [← card_coins]
    refine card_le_card_of_injOn (fun s => s.2.1) (fun s hs => ?_) ?_
    · rw [mem_coe, mem_filter, hF, mem_filter, mem_triples] at hs
      exact hs.1.1.2.1
    · intro s hs t ht hst
      rw [mem_coe, mem_filter] at hs ht
      obtain ⟨a, b, c, -, -, -, -, hc1, hc2, rfl, hv⟩ := hmem s hs.1
      obtain ⟨a', b', c', -, -, -, -, hc1', hc2', rfl, hv'⟩ := hmem t ht.1
      simp only at hst hs ht
      have := hs.2
      have := ht.2
      have hcc : c = c' ∧ a = a' := by omega
      rw [hcc.1, hcc.2, hst]
  have hhigh : (F.filter fun s => s.2.2 = -127).card ≤ 2 := by
    calc (F.filter fun s => s.2.2 = -127).card ≤ (univ : Finset Bool).card :=
          card_le_card_of_injOn (fun s => decide (0 < s.2.1)) (fun _ _ => mem_coe.mpr (mem_univ _)) ?_
      _ = 2 := rfl
    intro s hs t ht hst
    rw [mem_coe, mem_filter] at hs ht
    obtain ⟨a, b, c, -, -, hb1, hb2, -, -, rfl, hv⟩ := hmem s hs.1
    obtain ⟨a', b', c', -, -, hb1', hb2', -, -, rfl, hv'⟩ := hmem t ht.1
    simp only [decide_eq_decide] at hst hs ht
    have := hs.2
    have := ht.2
    have hall : a = a' ∧ b = b' ∧ c = c' := by omega
    rw [hall.1, hall.2.1, hall.2.2]
  have := card_filter_add_card_filter_not (s := F) (fun s : ℤ × ℤ × ℤ => s.2.2 = -127)
  omega

/-- **The bound is attained**: 127 is `127·s₀ + s₁ + s₂` for 130 triples: `(1, x, −x)` for each of the 128 coins `x`,
and the boundary pair `(3, −127, −127)` and `(−1, 127, 127)` that the 127 weight opens. -/
theorem evenSum_fiber_127 : 130 ≤ (triples.filter fun s => evenSum s = 127).card := by
  have hneg : ∀ x ∈ coins, -x ∈ coins := fun x hx => by
    obtain ⟨k, h1, h2, rfl⟩ := mem_coins.mp hx
    exact mem_coins.mpr ⟨-k - 1, by omega, by omega, by ring⟩
  have h127 : (127 : ℤ) ∈ coins := mem_coins.mpr ⟨63, by norm_num, by norm_num, by norm_num⟩
  have hm127 : (-127 : ℤ) ∈ coins := mem_coins.mpr ⟨-64, by norm_num, by norm_num, by norm_num⟩
  have h1 : (1 : ℤ) ∈ coins := mem_coins.mpr ⟨0, by norm_num, by norm_num, by norm_num⟩
  have h3 : (3 : ℤ) ∈ coins := mem_coins.mpr ⟨1, by norm_num, by norm_num, by norm_num⟩
  have hm1 : (-1 : ℤ) ∈ coins := mem_coins.mpr ⟨-1, by norm_num, by norm_num, by norm_num⟩
  set L : Finset (ℤ × ℤ × ℤ) :=
    insert (3, -127, -127) (insert (-1, 127, 127) (coins.image fun x => (1, x, -x))) with hL
  have hcard : L.card = 130 := by
    have hinj : Function.Injective fun x : ℤ => ((1 : ℤ), x, -x) := fun x y h => by
      simp only [Prod.mk.injEq] at h; exact h.2.1
    rw [hL, card_insert_of_notMem, card_insert_of_notMem, card_image_of_injective _ hinj, card_coins]
    · simp only [mem_image, Prod.mk.injEq, not_exists, not_and]; intro x _ h; norm_num at h
    · simp only [mem_insert, mem_image, Prod.mk.injEq, not_or, not_exists, not_and]
      exact ⟨by norm_num, fun x _ h => by norm_num at h⟩
  rw [← hcard]
  refine card_le_card fun s hs => ?_
  rw [mem_filter, mem_triples]
  rw [hL, mem_insert, mem_insert, mem_image] at hs
  rcases hs with rfl | rfl | ⟨x, hx, rfl⟩
  · exact ⟨⟨h3, hm127, hm127⟩, by norm_num [evenSum]⟩
  · exact ⟨⟨hm1, h127, h127⟩, by norm_num [evenSum]⟩
  · exact ⟨⟨h1, hx, hneg x hx⟩, by simp only [evenSum]; ring⟩

/-- Swapping the first two coins carries the odd rows' coin sum onto the even rows'. -/
private def swap01 (s : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ := (s.2.1, s.1, s.2.2)

/-- **Both classes have one law**: for every value, the odd rows' fiber has as many triples as the even rows'. -/
theorem oddSum_fiber_card (v : ℤ) :
    (triples.filter fun s => oddSum s = v).card = (triples.filter fun s => evenSum s = v).card := by
  refine card_nbij' swap01 swap01 (fun s hs => ?_) (fun s hs => ?_) (fun s _ => rfl) (fun s _ => rfl)
  · rw [mem_coe, mem_filter, mem_triples] at hs ⊢
    exact ⟨⟨hs.1.2.1, hs.1.1, hs.1.2.2⟩, by rw [← hs.2]; simp only [swap01, evenSum, oddSum]; ring⟩
  · rw [mem_coe, mem_filter, mem_triples] at hs ⊢
    exact ⟨⟨hs.1.2.1, hs.1.1, hs.1.2.2⟩, by rw [← hs.2]; simp only [swap01, evenSum, oddSum]; ring⟩

theorem rotSum_fiber_le (i : ℕ) (v : ℤ) : (triples.filter fun s => rotSum i s = v).card ≤ 130 := by
  by_cases hi : i % 2 = 0
  · simp only [rotSum, hi, ite_true]; exact evenSum_fiber_le v
  · simp only [rotSum, hi, ite_false]; rw [oddSum_fiber_card]; exact evenSum_fiber_le v

theorem rotSum_fiber_127 (i : ℕ) : 130 ≤ (triples.filter fun s => rotSum i s = 127).card := by
  by_cases hi : i % 2 = 0
  · simp only [rotSum, hi, ite_true]; exact evenSum_fiber_127
  · simp only [rotSum, hi, ite_false]; rw [oddSum_fiber_card]; exact evenSum_fiber_127

/-- **The 127 weight's range**: a row's coin sum is at most `127·127 + 2·127 = 16,383` in magnitude, against `3·127`
unweighted. -/
theorem rotSum_abs_le (i : ℕ) {s : ℤ × ℤ × ℤ} (hs : s ∈ triples) : |rotSum i s| ≤ 16383 := by
  rw [mem_triples] at hs
  obtain ⟨a, ha1, ha2, ha⟩ := mem_coins.mp hs.1
  obtain ⟨b, hb1, hb2, hb⟩ := mem_coins.mp hs.2.1
  obtain ⟨c, hc1, hc2, hc⟩ := mem_coins.mp hs.2.2
  rw [abs_le]
  by_cases hi : i % 2 = 0
  · simp only [rotSum, hi, ite_true, evenSum, ha, hb, hc]; constructor <;> omega
  · simp only [rotSum, hi, ite_false, oddSum, ha, hb, hc]; constructor <;> omega

/-- Row `i`'s step increment `c + b + Σ` from its bit `b` and the tile's coins `s`, weighted by its class. -/
def rotLane (c : ℤ) (i : ℕ) (b : ℤ) (s : ℤ × ℤ × ℤ) : ℤ := c + b + rotSum i s

theorem rotSum_odd (i : ℕ) {s : ℤ × ℤ × ℤ} (hs : s ∈ triples) : ∃ m, rotSum i s = 2 * m + 1 := by
  rw [mem_triples] at hs
  obtain ⟨a, -, -, ha⟩ := mem_coins.mp hs.1
  obtain ⟨b, -, -, hb⟩ := mem_coins.mp hs.2.1
  obtain ⟨c, -, -, hc⟩ := mem_coins.mp hs.2.2
  by_cases hi : i % 2 = 0
  · exact ⟨127 * a + b + c + 64, by simp only [rotSum, hi, ite_true, evenSum, ha, hb, hc]; ring⟩
  · exact ⟨a + 127 * b + c + 64, by simp only [rotSum, hi, ite_false, oddSum, ha, hb, hc]; ring⟩

/-- **The increment's largest fiber**: over the row bits `{0, 1}` and the coin triples, no value is row `i`'s increment
for more than 130 of the `2·128³` draws, whatever `c` and the class. -/
theorem rotLane_fiber_le (c : ℤ) (i : ℕ) (v : ℤ) :
    (fiber (rotLane c i) ({0, 1} : Finset ℤ) triples v).card ≤ 130 := by
  refine le_trans ?_ (rotSum_fiber_le i (v - c - (v - c - 1) % 2))
  refine card_le_card_of_injOn (fun p => p.2) (fun p hp => ?_) ?_
  · rw [mem_coe, fiber, mem_filter, mem_product] at hp
    obtain ⟨⟨hb, hs⟩, hv⟩ := hp
    obtain ⟨m, hm⟩ := rotSum_odd i hs
    rw [mem_coe, mem_filter]
    refine ⟨hs, ?_⟩
    simp only [rotLane] at hv
    rw [mem_insert, mem_singleton] at hb
    dsimp only
    omega
  · rintro ⟨b, s⟩ hp ⟨b', s'⟩ hp' hss
    simp only at hss
    subst hss
    rw [mem_coe, fiber, mem_filter, mem_product] at hp hp'
    simp only [rotLane, mem_insert, mem_singleton] at hp hp'
    rw [show b = b' by omega]

/-- **It is attained**: `c + 127` is row `i`'s increment for 130 draws, with `b = 0` and the 130 triples of
`evenSum_fiber_127` (swapped on odd rows). -/
theorem rotLane_fiber_attained (c : ℤ) (i : ℕ) :
    130 ≤ (fiber (rotLane c i) ({0, 1} : Finset ℤ) triples (c + 127)).card := by
  refine le_trans (rotSum_fiber_127 i) (card_le_card_of_injOn (fun s => ((0 : ℤ), s)) (fun s hs => ?_)
    (fun s _ t _ h => (Prod.mk.inj h).2))
  rw [mem_coe, mem_filter] at hs
  rw [mem_coe, fiber, mem_filter, mem_product]
  exact ⟨⟨by simp, hs.1⟩, by simp only [rotLane, hs.2]; ring⟩

/-- **`p*` exactly**: the largest share of the `2·128³ = 4,194,304` draws on which row `i`'s increment hits a value fixed
before the draw is `130/4,194,304 = 65/2,097,152`, on even and odd rows alike. -/
theorem rotLane_pstar (c : ℤ) (i : ℕ) :
    (∀ v, 2097152 * (fiber (rotLane c i) ({0, 1} : Finset ℤ) triples v).card ≤
        65 * (({0, 1} : Finset ℤ) ×ˢ triples).card) ∧
      ∃ v, 2097152 * (fiber (rotLane c i) ({0, 1} : Finset ℤ) triples v).card ≥
        65 * (({0, 1} : Finset ℤ) ×ˢ triples).card := by
  have h2 : ({0, 1} : Finset ℤ).card = 2 := by decide
  have hcard : (({0, 1} : Finset ℤ) ×ˢ triples).card = 4194304 := by
    rw [card_product, h2, card_triples]; norm_num
  rw [hcard]
  refine ⟨fun v => ?_, c + 127, ?_⟩
  · have := rotLane_fiber_le c i v; omega
  · have := rotLane_fiber_attained c i; omega

end Pouw.SecurityProofs.Lead.PStar
