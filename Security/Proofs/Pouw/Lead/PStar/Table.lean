import Definitions.Pouw.Lead.Gap
import Proofs.Pouw.Lead.PStar.Lanes

/-!
# The coin lanes' guess, and tables against the committed tiles

`pStar_eq`: the exact `p*` of `blc4a-t64-e7`'s three lanes under `-rot`'s weight `(127, 1, 1)` with the tag bit, 130
of `2 · 128³`, so `65 / 2^21`. `table_le`: a memory fixed before the coins, or a coin stream of raw coins, holds a
committed tile rarely: at most `tableGap` in words averaged over the coins.
-/

namespace Pouw.SecurityProofs.Lead.PStar

open Finset Pouw.Lead

/-- A coin lane's value from its draw `u`: `2u − 127`. -/
def lane (u : Fin 128) : ℤ := 2 * (u : ℤ) - 127

theorem lane_mem (u : Fin 128) : lane u ∈ coins := by
  have hu := u.isLt
  refine mem_coins.mpr ⟨(u : ℤ) - 64, by omega, by omega, ?_⟩
  simp only [lane]
  ring

theorem lane_injective : Function.Injective lane := fun u u' h => by
  simp only [lane] at h
  exact Fin.ext (by omega)

theorem exists_lane {x : ℤ} (hx : x ∈ coins) : ∃ u, lane u = x := by
  obtain ⟨k, h1, h2, rfl⟩ := mem_coins.mp hx
  refine ⟨⟨(k + 64).toNat, by omega⟩, ?_⟩
  simp only [lane]
  omega

set_option linter.constructorNameAsVariable false in
/-- The statement's draws of a step's lanes and tag bit are `rotLane 1 0`'s on an even row. -/
theorem pStar_fiber_card (v : ℤ) :
    (univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card =
      (Pouw.Dimension.Alphabet.fiber (rotLane 1 0) ({0, 1} : Finset ℤ) triples v).card := by
  refine card_bij (fun x _ => (((x.2.2.2 : ℕ) : ℤ), lane x.1, lane x.2.1, lane x.2.2.1)) (fun x hx => ?_)
    (fun x _ y _ h => ?_) (fun p hp => ?_)
  · simp only [mem_filter, mem_univ, true_and] at hx
    rw [Pouw.Dimension.Alphabet.fiber, mem_filter, mem_product, mem_triples]
    refine ⟨⟨?_, lane_mem _, lane_mem _, lane_mem _⟩, ?_⟩
    · have := x.2.2.2.isLt
      simp only [mem_insert, mem_singleton]
      omega
    · rw [← hx]
      simp only [rotLane, rotSum, evenSum, Nat.zero_mod, ite_true]
      ring
  · simp only [Prod.mk.injEq] at h
    obtain ⟨hb, h0, h1, h2⟩ := h
    exact Prod.ext (lane_injective h0) (Prod.ext (lane_injective h1)
      (Prod.ext (lane_injective h2) (Fin.ext (by omega))))
  · obtain ⟨b, s0, s1, s2⟩ := p
    rw [Pouw.Dimension.Alphabet.fiber, mem_filter, mem_product, mem_triples] at hp
    obtain ⟨⟨hb, h0, h1, h2⟩, hv⟩ := hp
    obtain ⟨u0, rfl⟩ := exists_lane h0
    obtain ⟨u1, rfl⟩ := exists_lane h1
    obtain ⟨u2, rfl⟩ := exists_lane h2
    have hb' : b = 0 ∨ b = 1 := by simpa only [mem_insert, mem_singleton] using hb
    refine ⟨(u0, u1, u2, ⟨b.toNat, by omega⟩), ?_, ?_⟩
    · simp only [mem_filter, mem_univ, true_and]
      rw [← hv]
      simp only [rotLane, rotSum, evenSum, Nat.zero_mod, ite_true]
      have : (((⟨b.toNat, by omega⟩ : Fin 2) : ℕ) : ℤ) = b := by simp only; omega
      rw [this]
      ring
    · simp only [Prod.mk.injEq, and_true]
      omega

/-- **The exact `p*`**: a value of `127 s₀ + s₁ + s₂ + 1 + b` is taken by at most 130 of the
`2 · 128³` draws, and one value by exactly 130 (`p* = 65 / 2^21`). -/
theorem pStar_eq :
    (∀ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card ≤ 130) ∧
      ∃ v : ℤ, (Finset.univ.filter fun x : Fin 128 × Fin 128 × Fin 128 × Fin 2 =>
        127 * lane x.1 + lane x.2.1 + lane x.2.2.1 + 1 + (x.2.2.2 : ℤ) = v).card = 130 := by
  refine ⟨fun v => ?_, 1 + 127, ?_⟩
  · rw [pStar_fiber_card]
    exact rotLane_fiber_le 1 0 v
  · rw [pStar_fiber_card]
    exact le_antisymm (rotLane_fiber_le 1 0 _) (rotLane_fiber_attained 1 0)

/-- **Tables rarely hold a committed tile**: for any data, every strategy's pre-coin tables
and the certifier's coin stream, the committed tiles its memory starts with, in words and averaged over the coins,
are at most `tableGap`. A step-0 tile is a given value with chance at most `49^-8` (`7^-8` against the at most 32
stream tiles holding its block's `G`), a later step's at most `2^-100` (each column's even and odd rows fix its three
lanes); a union over the `2 · mem + 1` values. Reads `coinTerm`, `word`, `tileOf`, `CoinStream` and `bal13`. -/
theorem table_le (T m n K : ℕ) (κ : Fin (lead T m n K).m → Fin (lead T m n K).S → Fin (lead T m n K).n → ℤ)
    (σ : Strategy (lead T m n K)) (stream : CoinStream (lead T m n K)) :
    ∑ ω : Coins (lead T m n K), (tileWords * tableHits (lead T m n K) κ σ stream ω : ℚ) ≤
      Fintype.card (Coins (lead T m n K)) * tableGap (lead T m n K) := by
  sorry

end Pouw.SecurityProofs.Lead.PStar
